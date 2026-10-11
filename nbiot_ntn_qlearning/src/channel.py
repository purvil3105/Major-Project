"""
channel.py — 3GPP NTN Link Budget and Calibrated NPRACH Detection Model.

Models:
1. Link-Budget-Derived Uplink SNR:
   Grounds SNR in orbital geometry, free-space path loss (FSPL), satellite
   antenna beam roll-off, and receiver thermal noise (3GPP TR 38.821 & TR 38.811):
       SNR(ε) = P_tx + G_tx + G_rx(θ(ε)) - FSPL(d(ε)) - N_0

2. Sinc Inter-Carrier Interference (ICI) Loss:
   Grounded in subcarrier spacing SCS = 3750 Hz (TS 36.211 Format 0/1) and
   residual CFO δf (3GPP R1-2100601 / Lee et al., arXiv:2512.07053).

3. Calibrated Detection Probability P_d(SNR, δf, N_rep):
   Calibrated in the operational range [-5 dB, +5 dB] matching Chougrani et al.
   (IEEE IoT-J 2022) preamble detection operating curves.
"""

import numpy as np


def compute_slant_range_km(
    elevation_deg: float or np.ndarray,
    altitude_km: float = 600.0,
    earth_radius_km: float = 6371.0
) -> float or np.ndarray:
    """Calculate slant range from ground station to satellite in km."""
    elev_rad = np.radians(np.clip(elevation_deg, 0.0, 90.0))
    # Law of cosines in spherical geometry
    re_sin = earth_radius_km * np.sin(elev_rad)
    slant_km = np.sqrt(re_sin**2 + 2.0 * earth_radius_km * altitude_km + altitude_km**2) - re_sin
    return slant_km


def elevation_to_snr_db(
    elevation_deg: float or np.ndarray,
    altitude_km: float = 600.0,
    carrier_hz: float = 2.0e9,
    p_tx_dbm: float = 23.0,
    g_tx_dbi: float = 0.0,
    g_rx_zenith_dbi: float = 6.0,
    antenna_3db_beamwidth_deg: float = 65.0,
    noise_figure_db: float = 5.0,
    temperature_k: float = 290.0,
    scs_hz: float = 3750.0,
    **kwargs
) -> float or np.ndarray:
    """
    Compute single-tone uplink SNR from 3GPP NTN link budget.
    
    Parameters
    ----------
    elevation_deg : float or ndarray
        Satellite elevation angle from ground station in degrees.
    altitude_km : float
        Satellite orbital altitude in km.
    carrier_hz : float
        Carrier frequency in Hz (default: 2.0 GHz, S-band n256).
    p_tx_dbm : float
        UE transmit power in dBm (default: 23 dBm for NB-IoT Class 3).
    g_tx_dbi : float
        UE antenna gain in dBi (default: 0 dBi isotropic).
    g_rx_zenith_dbi : float
        Satellite antenna boresight gain in dBi.
    antenna_3db_beamwidth_deg : float
        Satellite antenna 3 dB beamwidth in degrees.
    noise_figure_db : float
        Satellite receiver noise figure in dB.
    temperature_k : float
        System noise temperature in Kelvin.
    scs_hz : float
        NPRACH subcarrier bandwidth in Hz (3750 Hz).
        
    Returns
    -------
    snr_db : float or ndarray
        Uplink single-tone SNR in dB.
    """
    elev = np.clip(elevation_deg, 10.0, 90.0)
    re_km = 6371.0

    # 1. Slant range
    d_km = compute_slant_range_km(elev, altitude_km, re_km)
    d_m = d_km * 1000.0

    # 2. Free-Space Path Loss: FSPL = 20*log10(4*pi*d / lambda)
    c_speed = 299792458.0
    wavelength = c_speed / carrier_hz
    fspl_db = 20.0 * np.log10(4.0 * np.pi * d_m / wavelength)

    # 3. Satellite nadir angle theta and 3GPP shaped-beam / isoflux roll-off
    elev_rad = np.radians(elev)
    # sin(theta) = Re / (Re + h) * cos(elev)
    sin_theta = (re_km / (re_km + altitude_km)) * np.cos(elev_rad)
    theta_deg = np.degrees(np.arcsin(np.clip(sin_theta, 0.0, 1.0)))

    # In 3GPP LEO payloads (TR 38.821), isoflux / shaped beams limit beam-edge roll-off to ~3-4 dB
    # to compensate for path loss toward the horizon.
    beam_rolloff_db = np.clip(12.0 * ((theta_deg / antenna_3db_beamwidth_deg) ** 2), 0.0, 3.5)
    g_rx_dbi = g_rx_zenith_dbi - beam_rolloff_db

    # 4. Thermal noise power in dBm: N0 = k * T * B + NF
    k_boltzmann = 1.380649e-23
    noise_power_w = k_boltzmann * temperature_k * scs_hz
    noise_power_dbm = 10.0 * np.log10(noise_power_w) + 30.0 + noise_figure_db

    # 5. Received single-tone SNR in dB
    snr_db = (p_tx_dbm + g_tx_dbi + g_rx_dbi) - fspl_db - noise_power_dbm

    if np.isscalar(elevation_deg):
        return float(snr_db)
    return snr_db


def compute_detection_probability(
    snr_db: float or np.ndarray,
    residual_doppler_hz: float or np.ndarray,
    n_rep: int or np.ndarray,
    scs_hz: float = 3750.0,
    gamma_th_db: float = -1.0,
    k_slope: float = 0.60
) -> float or np.ndarray:
    """
    Calculate NPRACH preamble detection probability P_d(SNR, δf, N_rep).
    
    Calibration basis (Chougrani et al., 2022):
    In NPRACH preamble detection over NTN with residual CFO:
    - Effective accumulation gain across repetitions:
        Gain_rep = 10*log10(N_rep) - 0.5*log2(N_rep)
    - Sinc ICI power reduction from residual CFO:
        L_ICI = 10 * log10( sinc^2(delta_f / scs) )
    - Operating characteristic calibrated to [-5 dB, +5 dB] range:
        P_d = 1 / (1 + exp(-k * (SNR_eff - gamma_th)))
    """
    n_rep_arr = np.asarray(n_rep, dtype=float)
    snr_arr = np.asarray(snr_db, dtype=float)
    cfo_arr = np.asarray(residual_doppler_hz, dtype=float)

    # 1. Repetition accumulation gain (coherent within symbol group, non-coherent across groups)
    rep_gain_db = 10.0 * np.log10(np.maximum(n_rep_arr, 1.0)) - 0.5 * np.log2(np.maximum(n_rep_arr, 1.0))

    # 2. Sinc ICI loss: 10 * log10( (sin(pi*x) / (pi*x))^2 ) ~ 14.3 * (x^2) in dB
    cfo_norm = np.abs(cfo_arr) / scs_hz
    ici_loss_db = 14.3 * (cfo_norm ** 2)

    # Effective post-combining SNR
    snr_eff = snr_arr + rep_gain_db - ici_loss_db

    # 3. Logistic operating characteristic
    logit = k_slope * (snr_eff - gamma_th_db)
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
    scs_hz: float = 3750.0,
    gamma_th_db: float = -1.0,
    k_slope: float = 0.60
) -> float or np.ndarray:
    """Detection miss probability P_miss = 1 - P_d."""
    return 1.0 - compute_detection_probability(
        snr_db, residual_doppler_hz, n_rep, scs_hz, gamma_th_db, k_slope
    )
