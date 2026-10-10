"""
channel.py — NTN Channel model and NPRACH detection probability calibration.

Models:
1. Elevation-dependent Uplink SNR:
   SNR(ε) [dB] varies from snr_edge_db (at 10°) to snr_zenith_db (at 90°),
   grounded in Dwivedi et al. (Set-4 UL SNR: -3 dB edge to +5 dB zenith).

2. Residual Doppler shift δf:
   Nominal: ~10 Hz; Stress: ~250 Hz (Lee et al., arXiv:2512.07053).
   Subcarrier spacing SCS = 3750 Hz (NPRACH Format 0/1, 3GPP TS 36.211).

3. Detection probability P_d(SNR, δf, N_rep):
   Calibrated strictly to Chougrani et al. (IEEE IoT-J 2022) miss-detection curves:
   - Anchor 1: SNR = 8.0 dB, N_rep = 64, δf = 10 Hz  -> P_miss <= 0.01 (P_d >= 0.99)
   - Anchor 2: SNR = 15.0 dB, N_rep = 32, δf = 10 Hz -> P_miss <= 0.01 (P_d >= 0.99)
   - P_d increases monotonically with N_rep and SNR.
   - P_d degrades with residual Doppler δf due to ICI and phase drift across repetitions.
"""

import numpy as np


def elevation_to_snr_db(
    elevation_deg: float or np.ndarray,
    snr_edge_db: float = -3.0,
    snr_zenith_db: float = 5.0,
    elev_min_deg: float = 10.0,
    elev_max_deg: float = 90.0
) -> float or np.ndarray:
    """
    Compute uplink SNR as a function of elevation angle.
    
    Uses standard NTN path loss & antenna gain elevation profile,
    interpolated smoothly between edge (-3 dB at 10°) and zenith (+5 dB at 90°).
    """
    elev = np.clip(elevation_deg, elev_min_deg, elev_max_deg)
    # Sinusoidal / geometric elevation scaling (higher elevation = shorter slant range & better antenna gain)
    norm_elev = (elev - elev_min_deg) / (elev_max_deg - elev_min_deg)
    snr = snr_edge_db + (snr_zenith_db - snr_edge_db) * (norm_elev ** 0.8)
    return snr


def compute_detection_probability(
    snr_db: float or np.ndarray,
    residual_doppler_hz: float or np.ndarray,
    n_rep: int or np.ndarray,
    scs_hz: float = 3750.0
) -> float or np.ndarray:
    """
    Calculate NPRACH preamble detection probability P_d(SNR, δf, N_rep).
    
    Calibration basis (Chougrani et al., 2022, Figs. 6-8):
    In NPRACH preamble detection over NTN with residual CFO:
    - Post-combining effective SNR includes repetition gain:
        SNR_post = SNR + 10 * log10(N_rep) - non_coherent_combining_loss(N_rep)
    - Residual Doppler causes ICI power reduction:
        L_ICI = 10 * log10( sinc^2(delta_f / scs) )
      and differential phase rotation across repeated symbol groups.
    - Detection curve follows logistic CDF:
        P_d = 1 / (1 + exp(-k * (SNR_eff - gamma_th)))
    
    Parameters
    ----------
    snr_db : float or ndarray
        Single-tone uplink SNR in dB.
    residual_doppler_hz : float or ndarray
        Residual Doppler shift δf in Hz (after ephemeris pre-compensation).
    n_rep : int or ndarray
        NPRACH repetition count (e.g. 2, 8, 32, 64).
    scs_hz : float
        NPRACH subcarrier spacing in Hz (default: 3750 Hz).
        
    Returns
    -------
    float or ndarray
        Detection probability P_d in [0.0, 1.0].
    """
    n_rep_arr = np.asarray(n_rep, dtype=float)
    snr_arr = np.asarray(snr_db, dtype=float)
    cfo_arr = np.asarray(residual_doppler_hz, dtype=float)

    # 1. Repetition accumulation gain with non-coherent loss across groups:
    rep_gain_db = 10.0 * np.log10(np.maximum(n_rep_arr, 1.0)) - 0.5 * np.log2(np.maximum(n_rep_arr, 1.0))

    # 2. Residual Doppler loss:
    # Normalized offset relative to subcarrier spacing (SCS = 3750 Hz)
    cfo_norm = np.abs(cfo_arr) / scs_hz
    # Sinc power reduction
    ici_loss_db = 14.3 * (cfo_norm ** 2)
    # Phase decorrelation across symbol groups for residual frequency offset
    # Grounded in Chougrani [5] and Lee [13]: 10 Hz is negligible, 250 Hz stress incurs ~2-3 dB loss
    decorr_loss_db = 2.8 * ((np.abs(cfo_arr) / 250.0) ** 1.5)

    # Total effective SNR at detector output
    snr_eff = snr_arr + rep_gain_db - ici_loss_db - decorr_loss_db

    # 3. Detector operating characteristic:
    # Calibrated to Chougrani et al. [5] anchors:
    # - Anchor 1: SNR = 8 dB, N_rep = 64, delta_f = 10 Hz -> P_miss <= 0.01 (P_d >= 0.99)
    # - Anchor 2: SNR = 15 dB, N_rep = 32, delta_f = 10 Hz -> P_miss <= 0.01 (P_d >= 0.99)
    gamma_th = 2.0   # dB threshold
    k_slope = 0.45   # logistic slope

    # Logistic function
    logit = k_slope * (snr_eff - gamma_th)
    logit_clipped = np.clip(logit, -30.0, 30.0)
    p_d = 1.0 / (1.0 + np.exp(-logit_clipped))

    # Numerical safeguard
    p_d = np.clip(p_d, 0.0001, 0.9999)

    if np.isscalar(snr_db) and np.isscalar(residual_doppler_hz) and np.isscalar(n_rep):
        return float(p_d)
    return p_d


def compute_miss_probability(
    snr_db: float or np.ndarray,
    residual_doppler_hz: float or np.ndarray,
    n_rep: int or np.ndarray,
    scs_hz: float = 3750.0
) -> float or np.ndarray:
    """Detection miss probability P_m = 1 - P_d."""
    return 1.0 - compute_detection_probability(snr_db, residual_doppler_hz, n_rep, scs_hz)
