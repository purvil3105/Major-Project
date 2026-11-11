"""
channel.py — uplink link budget and NPRACH detection model.

1. Link budget (3GPP TR 38.821 Set-4 payload, as tabulated by Dwivedi et al. [10]):
       SNR = EIRP_UE + G/T - k_B - FSPL(d) - L_atm - L_shadow - L_scint - L_pol
             - L_beam(theta) - 10 log10(B_NPRACH)
   with the SGP4 slant range d and the satellite off-nadir angle theta. The beam roll-off is
   L_beam = 12 (theta / theta_3dB)^2 dB (3 dB at the beam edge). Reproduces [10]: -3.14 dB at
   30 deg (beam edge) and +4.93 dB at 90 deg for a 600 km orbit.

2. Detection probability, logistic in SNR:
       P_d(SNR, N) = 1 / (1 + exp(-k (SNR - SNR99(N) + ln(99)/k)))
       SNR99(N)   = SNR99_ref + s * log2(N / N_ref) + Delta_regime
   - GNSS-assisted regime (Delta = 0): passes exactly through the TS 36.104 Table 8.5.3.2.1-1
     AWGN requirements (99 % at -2.1 dB / 8 reps and -6.8 dB / 32 reps, 2 Rx), shifted +3 dB
     for one Rx branch.
   - GNSS-free regime: least-squares offset to Chougrani et al. [5] (MD-RTD 800 us).
   Residual Doppler after pre-compensation (10-250 Hz << 3.75 kHz SCS) is not modelled:
   its ICI loss is < 0.1 dB (see REVIEW_ISSUES D-2).

3. Differential RTD for GNSS-degraded meters: MD_RTD(eps) = 2 D cos(eps) / c.
"""

import numpy as np

K_BOLTZMANN_DBW = -228.6
RE_KM = 6371.0
C_M_S = 299792458.0
LN99 = np.log(99.0)


def off_nadir_deg(elevation_deg, altitude_km):
    """Satellite off-nadir angle towards the ground site (spherical Earth)."""
    e = np.radians(np.asarray(elevation_deg, dtype=float))
    s = np.clip(RE_KM / (RE_KM + altitude_km) * np.cos(e), -1.0, 1.0)
    return np.degrees(np.arcsin(s))


def outdoor_snr_db(slant_range_km, elevation_deg, altitude_km, cfg):
    """Single-tone (3.75 kHz) uplink SNR for an outdoor meter (no extra coupling loss)."""
    d_m = np.asarray(slant_range_km, dtype=float) * 1e3
    lam = C_M_S / float(cfg["carrier_hz"])
    fspl = 20.0 * np.log10(4.0 * np.pi * d_m / lam)
    theta = off_nadir_deg(elevation_deg, altitude_km)
    beam = np.minimum(12.0 * (theta / float(cfg["sat_beamwidth_deg"])) ** 2, 20.0)
    losses = (cfg["atmospheric_loss_db"] + cfg["shadow_margin_db"]
              + cfg["scintillation_loss_db"] + cfg["polarization_loss_db"])
    return (cfg["ue_eirp_dbw"] + cfg["sat_gt_dbk"] - K_BOLTZMANN_DBW - fspl - losses - beam
            - 10.0 * np.log10(cfg["nprach_bw_hz"]))


def snr99_db(n_rep, cfg, regime: str = None):
    """SNR [dB] giving 99 % preamble detection with `n_rep` repetitions."""
    regime = regime or cfg["detection_regime"]
    n = np.asarray(n_rep, dtype=float)
    return (cfg["snr99_ref_db"] + cfg["snr99_slope_db"] * np.log2(n / cfg["snr99_ref_reps"])
            + cfg["detection_regimes"][regime])


def detection_probability(snr_db, n_rep, cfg, regime: str = None):
    """Preamble detection probability P_d(SNR, N_rep) (vectorised)."""
    k = float(cfg["pd_logistic_k"])
    mid = snr99_db(n_rep, cfg, regime) - LN99 / k
    x = np.clip(k * (np.asarray(snr_db, dtype=float) - mid), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x))


def md_rtd_s(elevation_deg, cfg):
    """Maximum differential round-trip delay seen by a meter without a valid GNSS fix."""
    return 2.0 * cfg["beam_equiv_diameter_km"] * 1e3 * np.cos(np.radians(elevation_deg)) / C_M_S


def timeout_probability(elevation_deg, cfg):
    """P(residual timing error > CP) for a GNSS-degraded meter; error ~ U(0, MD_RTD)."""
    md = md_rtd_s(elevation_deg, cfg)
    cp = cfg["cp_us"] * 1e-6
    return np.where(md > cp, 1.0 - cp / np.maximum(md, 1e-12), 0.0)
