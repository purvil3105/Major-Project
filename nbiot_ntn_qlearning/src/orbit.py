"""
orbit.py — SGP4 orbit propagation using Skyfield from real TLE data.

Calculates time-series of:
- Elevation ε(t) [deg]
- Slant range d(t) [km]
- Range rate r_dot(t) [km/s]
- One-way propagation delay τ(t) = d(t) / c [s]
- Raw Doppler shift f_D(t) = - (r_dot(t) / c) * f_c [Hz]

Acceptance criteria (Phase 1):
- Pass duration: ~2–7 minutes (120–420 s) for elevation >= 10 deg mask.
- One-way delay: ~2.0 ms (zenith) to ~6.4 ms (10° edge).
- Peak raw Doppler: <= ~41–48 kHz for 2 GHz carrier.
"""

import os
import numpy as np
from skyfield.api import load, EarthSatellite, wgs84
from skyfield.timelib import Time
import datetime

# Speed of light in m/s and km/s
SPEED_OF_LIGHT_M_S = 299792458.0
SPEED_OF_LIGHT_KM_S = 299792.458


def load_satellite_from_tle(tle_path: str, sat_name: str = "SATELIOT_2") -> EarthSatellite:
    """Load an EarthSatellite from a TLE file."""
    if not os.path.isabs(tle_path):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        tle_path = os.path.join(project_root, tle_path)

    ts = load.timescale()
    with open(tle_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    # Parse 3-line format (Name, Line 1, Line 2)
    sats = {}
    i = 0
    while i < len(lines):
        if lines[i].startswith("1 ") and i > 0:
            name = lines[i - 1].strip()
            l1 = lines[i]
            l2 = lines[i + 1]
            sats[name] = EarthSatellite(l1, l2, name, ts)
            i += 2
        elif i + 2 < len(lines) and lines[i + 1].startswith("1 "):
            name = lines[i].strip()
            l1 = lines[i + 1]
            l2 = lines[i + 2]
            sats[name] = EarthSatellite(l1, l2, name, ts)
            i += 3
        else:
            i += 1

    if sat_name in sats:
        return sats[sat_name]
    if len(sats) > 0:
        # Fallback to first available satellite
        first_key = list(sats.keys())[0]
        return sats[first_key]
    raise ValueError(f"Could not parse any satellites from {tle_path}")


def find_high_elevation_pass(
    satellite: EarthSatellite,
    ground_lat: float,
    ground_lon: float,
    ground_elev_m: float = 53.0,
    min_elev_deg: float = 10.0,
    search_days: float = 7.0,
    min_culmination_deg: float = 40.0
):
    """
    Find the best visible pass (highest culmination angle >= min_culmination_deg)
    starting from satellite epoch over search_days.
    """
    ts = load.timescale()
    ground_station = wgs84.latlon(ground_lat, ground_lon, elevation_m=ground_elev_m)
    epoch_dt = satellite.epoch.utc_datetime()

    # Search window: from epoch to epoch + search_days
    t0 = ts.from_datetime(epoch_dt)
    t1 = ts.from_datetime(epoch_dt + datetime.timedelta(days=search_days))

    t, events = satellite.find_events(ground_station, t0, t1, altitude_degrees=min_elev_deg)

    # Group events: 0 = rise, 1 = culminate, 2 = set
    passes = []
    current_pass = {}
    for ti, event in zip(t, events):
        if event == 0:  # Rise
            current_pass = {'rise': ti}
        elif event == 1 and 'rise' in current_pass:  # Culminate
            current_pass['culminate'] = ti
            # check elevation at culmination
            diff = (satellite - ground_station).at(ti)
            alt, az, dist = diff.altaz()
            current_pass['max_alt_deg'] = alt.degrees
        elif event == 2 and 'culminate' in current_pass:  # Set
            current_pass['set'] = ti
            passes.append(current_pass)
            current_pass = {}

    # Filter by culmination requirement and sort by highest elevation
    good_passes = [p for p in passes if p.get('max_alt_deg', 0) >= min_culmination_deg]
    if not good_passes:
        # Fall back to best available pass
        if passes:
            passes.sort(key=lambda x: x.get('max_alt_deg', 0), reverse=True)
            return passes[0]
        raise RuntimeError("No pass found matching criteria.")

    good_passes.sort(key=lambda x: x['max_alt_deg'], reverse=True)
    return good_passes[0]


def compute_pass_trajectory(
    satellite: EarthSatellite,
    pass_info: dict,
    ground_lat: float,
    ground_lon: float,
    ground_elev_m: float = 53.0,
    carrier_hz: float = 2.0e9,
    time_step_s: float = 1.0,
    elev_mask_deg: float = 10.0
) -> dict:
    """
    Propagate the orbit along the selected pass at time_step_s increments.
    
    Returns
    -------
    dict with numpy arrays:
        time_s: array of elapsed seconds from rise [0, 1, 2, ...]
        timestamps: list of Skyfield Time objects
        elevation_deg: array of elevation angles [deg]
        slant_range_km: array of distance from ground to satellite [km]
        range_rate_km_s: array of radial velocity dr/dt [km/s] (positive = moving away)
        delay_s: one-way delay tau(t) = d(t)/c [s]
        delay_ms: tau(t) [ms]
        raw_doppler_hz: Doppler shift f_D(t) = - (r_dot / c) * f_c [Hz]
        duration_s: total pass duration in seconds
        max_elevation_deg: peak elevation during pass
    """
    ts = load.timescale()
    ground_station = wgs84.latlon(ground_lat, ground_lon, elevation_m=ground_elev_m)

    t_rise = pass_info['rise']
    t_set = pass_info['set']
    duration_s = (t_set.utc_datetime() - t_rise.utc_datetime()).total_seconds()

    dt_step = datetime.timedelta(seconds=time_step_s)
    current_dt = t_rise.utc_datetime()
    end_dt = t_set.utc_datetime()

    time_s = []
    times = []
    elevations = []
    ranges = []
    range_rates = []

    elapsed = 0.0
    while current_dt <= end_dt + datetime.timedelta(seconds=0.1):
        ti = ts.from_datetime(current_dt)
        times.append(ti)
        time_s.append(elapsed)

        diff = (satellite - ground_station).at(ti)
        alt, az, dist = diff.altaz()
        elevations.append(alt.degrees)
        ranges.append(dist.km)

        # Range rate dr/dt via central or forward/backward difference with +/- 0.05 s
        dt_eps = datetime.timedelta(seconds=0.05)
        ti_plus = ts.from_datetime(current_dt + dt_eps)
        ti_minus = ts.from_datetime(current_dt - dt_eps)
        dist_plus = (satellite - ground_station).at(ti_plus).altaz()[2].km
        dist_minus = (satellite - ground_station).at(ti_minus).altaz()[2].km
        r_dot = (dist_plus - dist_minus) / 0.1  # km/s
        range_rates.append(r_dot)

        current_dt += dt_step
        elapsed += time_step_s

    time_s = np.array(time_s)
    elevations = np.array(elevations)
    ranges = np.array(ranges)
    range_rates = np.array(range_rates)

    # Delay: tau(t) = range / c
    delay_s = (ranges * 1000.0) / SPEED_OF_LIGHT_M_S
    delay_ms = delay_s * 1000.0

    # Doppler: f_D = - (range_rate / c) * f_c
    # When satellite approaches, range_rate < 0, f_D > 0 (frequency increases)
    raw_doppler_hz = - (range_rates / SPEED_OF_LIGHT_KM_S) * carrier_hz

    return {
        'time_s': time_s,
        'elevation_deg': elevations,
        'slant_range_km': ranges,
        'range_rate_km_s': range_rates,
        'delay_s': delay_s,
        'delay_ms': delay_ms,
        'raw_doppler_hz': raw_doppler_hz,
        'duration_s': time_s[-1] if len(time_s) > 0 else 0.0,
        'max_elevation_deg': float(np.max(elevations)) if len(elevations) > 0 else 0.0
    }


def generate_pass_trajectory_from_config(cfg: dict) -> dict:
    """Convenience helper to compute trajectory directly from config."""
    sat = load_satellite_from_tle(cfg['tle_file'], cfg.get('tle_satellite_name', 'SATELIOT_2'))
    best_pass = find_high_elevation_pass(
        satellite=sat,
        ground_lat=cfg['ground_lat_deg'],
        ground_lon=cfg['ground_lon_deg'],
        ground_elev_m=cfg.get('ground_elev_m', 53.0),
        min_elev_deg=cfg['elev_mask_deg'],
        search_days=7.0,
        min_culmination_deg=35.0
    )
    traj = compute_pass_trajectory(
        satellite=sat,
        pass_info=best_pass,
        ground_lat=cfg['ground_lat_deg'],
        ground_lon=cfg['ground_lon_deg'],
        ground_elev_m=cfg.get('ground_elev_m', 53.0),
        carrier_hz=cfg['carrier_hz'],
        time_step_s=1.0,
        elev_mask_deg=cfg['elev_mask_deg']
    )
    return traj
