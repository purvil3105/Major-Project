"""
orbit.py — SGP4 pass geometry from real TLEs (Skyfield).

For every pass of a satellite over the ground site the module returns, on a 1-s grid,
elevation eps(t), slant range d(t), range rate (from Skyfield's topocentric rates), one-way
delay tau = d/c and raw Doppler f_D = -rdot * f_c / c.

All qualifying passes (peak elevation >= min_pass_peak_deg) in the search window are
returned, so experiments cover several orbital geometries rather than one hand-picked pass.
Synthetic circular-orbit satellites at other altitudes are built for the altitude sweep.
"""

import math
from functools import lru_cache

import numpy as np
from sgp4.api import Satrec, WGS72
from skyfield.api import EarthSatellite, load, wgs84

from src.config_loader import resolve_path

C_M_S = 299792458.0
MU_EARTH_KM3_S2 = 398600.4418
RE_EQ_KM = 6378.137

_TS = None


def timescale():
    global _TS
    if _TS is None:
        _TS = load.timescale(builtin=True)
    return _TS


def load_tle_file(tle_path: str) -> dict:
    """Parse a 3-line TLE file into {name: EarthSatellite}."""
    ts = timescale()
    with open(resolve_path(tle_path), "r", encoding="utf-8") as f:
        lines = [ln.rstrip() for ln in f if ln.strip()]
    sats = {}
    for i in range(len(lines) - 2):
        if lines[i + 1].startswith("1 ") and lines[i + 2].startswith("2 "):
            name = lines[i].strip()
            sats[name] = EarthSatellite(lines[i + 1], lines[i + 2], name, ts)
    if not sats:
        raise ValueError(f"No satellites parsed from {tle_path}")
    return sats


def synthetic_satellite(reference: EarthSatellite, altitude_km: float, name: str) -> EarthSatellite:
    """Circular orbit at `altitude_km` sharing the reference's epoch, inclination and RAAN."""
    ref = reference.model
    a_km = RE_EQ_KM + altitude_km
    n_rad_min = math.sqrt(MU_EARTH_KM3_S2 / a_km ** 3) * 60.0
    epoch_days = (ref.jdsatepoch + ref.jdsatepochF) - 2433281.5
    sat = Satrec()
    sat.sgp4init(WGS72, "i", 99000 + int(altitude_km) % 1000, epoch_days,
                 0.0, 0.0, 0.0, 1e-4, ref.argpo, ref.inclo, ref.mo, n_rad_min, ref.nodeo)
    return EarthSatellite.from_satrec(sat, timescale())


def mean_altitude_km(sat: EarthSatellite) -> float:
    n_rad_s = sat.model.no_kozai / 60.0
    return (MU_EARTH_KM3_S2 / n_rad_s ** 2) ** (1.0 / 3.0) - RE_EQ_KM


def find_passes(sat: EarthSatellite, cfg: dict, min_peak_deg: float = None) -> list:
    """All passes above the mask whose culmination >= min_peak_deg, in time order."""
    ts = timescale()
    site = wgs84.latlon(cfg["ground_lat_deg"], cfg["ground_lon_deg"], elevation_m=cfg["ground_elev_m"])
    t0 = sat.epoch
    t1 = ts.tt_jd(t0.tt + float(cfg["pass_search_days"]))
    times, events = sat.find_events(site, t0, t1, altitude_degrees=float(cfg["elev_mask_deg"]))
    min_peak = float(cfg["min_pass_peak_deg"] if min_peak_deg is None else min_peak_deg)
    passes, cur = [], {}
    for ti, ev in zip(times, events):
        if ev == 0:
            cur = {"rise": ti}
        elif ev == 1 and "rise" in cur:
            alt = (sat - site).at(ti).altaz()[0].degrees
            cur["peak_deg"] = max(cur.get("peak_deg", -90.0), float(alt))
        elif ev == 2 and "peak_deg" in cur:
            cur["set"] = ti
            if cur["peak_deg"] >= min_peak:
                passes.append(cur)
            cur = {}
    return passes


def pass_trajectory(sat: EarthSatellite, pass_info: dict, cfg: dict, step_s: float = 1.0) -> dict:
    """Geometry time series for one pass on a `step_s` grid (vectorised)."""
    ts = timescale()
    site = wgs84.latlon(cfg["ground_lat_deg"], cfg["ground_lon_deg"], elevation_m=cfg["ground_elev_m"])
    dur = (pass_info["set"].tt - pass_info["rise"].tt) * 86400.0
    t_s = np.arange(0.0, dur + 1e-9, step_s)
    t = ts.tt_jd(pass_info["rise"].tt + t_s / 86400.0)
    alt, _, dist, _, _, range_rate = (sat - site).at(t).frame_latlon_and_rates(site)
    rng_km = dist.km
    rdot_km_s = range_rate.km_per_s
    return {
        "satellite": sat.name,
        "rise_utc": pass_info["rise"].utc_strftime("%Y-%m-%d %H:%M:%S"),
        "time_s": t_s,
        "elevation_deg": alt.degrees,
        "slant_range_km": rng_km,
        "range_rate_km_s": rdot_km_s,
        "delay_s": rng_km * 1e3 / C_M_S,
        "delay_ms": rng_km * 1e6 / C_M_S,
        "raw_doppler_hz": -rdot_km_s * 1e3 / C_M_S * float(cfg["carrier_hz"]),
        "duration_s": float(t_s[-1]),
        "max_elevation_deg": float(np.max(alt.degrees)),
        "altitude_km": mean_altitude_km(sat),
    }


@lru_cache(maxsize=32)
def _cached_passes(tle_file, sat_names, altitude_km, lat, lon, elev_m, days, peak, mask, fc, limit):
    cfg = {"ground_lat_deg": lat, "ground_lon_deg": lon, "ground_elev_m": elev_m,
           "pass_search_days": days, "min_pass_peak_deg": peak, "elev_mask_deg": mask, "carrier_hz": fc}
    sats = load_tle_file(tle_file)
    out = []
    for name in sat_names:
        sat = sats[name]
        if altitude_km is not None:
            sat = synthetic_satellite(sat, altitude_km, f"SYNTH_{int(altitude_km)}KM")
        for p in find_passes(sat, cfg):
            out.append(pass_trajectory(sat, p, cfg))
    out.sort(key=lambda tr: tr["rise_utc"])
    return tuple(out[:limit] if limit else out)


def get_passes(cfg: dict, role: str = "test", altitude_km: float = None, limit: int = None) -> list:
    """Pass trajectories for role 'test' (test satellite) or 'train' (training satellites)."""
    names = (cfg["test_satellite"],) if role == "test" else tuple(cfg["train_satellites"])
    if role == "test" and limit is None:
        limit = int(cfg["max_test_passes"])
    return list(_cached_passes(cfg["tle_file"], names, altitude_km, cfg["ground_lat_deg"],
                               cfg["ground_lon_deg"], cfg["ground_elev_m"], cfg["pass_search_days"],
                               cfg["min_pass_peak_deg"], cfg["elev_mask_deg"], cfg["carrier_hz"], limit))
