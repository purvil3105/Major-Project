"""
config_loader.py — Load and validate the master YAML configuration.

Provides a single `load_config()` function that returns a flat dict with
all parameters, plus a `set_global_seed()` helper for reproducibility.
"""

import os
import yaml
import numpy as np


def load_config(path: str = None) -> dict:
    """Load configuration from YAML file.

    Parameters
    ----------
    path : str, optional
        Path to YAML config file. Defaults to config/default.yaml
        relative to the project root.

    Returns
    -------
    dict
        Configuration dictionary with all parameters.
    """
    if path is None:
        # Default: project_root/config/default.yaml
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(project_root, "config", "default.yaml")

    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    # Cast numeric values that might be parsed as strings
    float_keys = [
        "carrier_hz", "scs_hz", "altitude_km", "elev_mask_deg", "earth_radius_km",
        "ground_lat_deg", "ground_lon_deg", "ground_elev_m", "snr_edge_db", "snr_zenith_db",
        "activation_window_s", "alpha", "gamma", "eps0", "eps_min", "eps_decay",
        "w_s", "w_c", "w_m", "w_t", "w_d", "d_ref_ms", "doppler_threshold"
    ]
    for k in float_keys:
        if k in cfg:
            cfg[k] = float(cfg[k])

    int_keys = [
        "n_preambles", "preamble_trans_max", "rar_window_ms", "rar_window_ntn_ms",
        "processing_time_ms", "beta_a", "beta_b", "load_bins", "elev_bins",
        "doppler_bins", "pass_phase_bins", "default_n_devices", "default_residual_doppler_hz",
        "default_rao_period_ms", "default_seed", "warmstart_terrestrial_raos"
    ]
    for k in int_keys:
        if k in cfg:
            cfg[k] = int(cfg[k])

    cfg["acb_levels"] = [float(x) for x in cfg["acb_levels"]]
    cfg["bi_levels_ms"] = [int(x) for x in cfg["bi_levels_ms"]]
    cfg["nrep_levels"] = [int(x) for x in cfg["nrep_levels"]]
    cfg["load_thresholds"] = [float(x) for x in cfg["load_thresholds"]]
    cfg["elev_thresholds"] = [float(x) for x in cfg["elev_thresholds"]]
    cfg["pass_phase_thresholds"] = [float(x) for x in cfg["pass_phase_thresholds"]]

    # Validate critical parameters
    assert cfg["n_preambles"] > 0, "n_preambles must be positive"
    assert cfg["altitude_km"] > 0, "altitude_km must be positive"
    assert cfg["elev_mask_deg"] >= 0, "elev_mask_deg must be non-negative"
    assert 0 < cfg["alpha"] <= 1, "learning rate alpha must be in (0, 1]"
    assert 0 <= cfg["gamma"] < 1, "discount gamma must be in [0, 1)"
    assert len(cfg["acb_levels"]) > 0, "acb_levels must be non-empty"
    assert len(cfg["bi_levels_ms"]) > 0, "bi_levels_ms must be non-empty"
    assert len(cfg["nrep_levels"]) > 0, "nrep_levels must be non-empty"

    # Compute derived quantities
    cfg["n_acb"] = len(cfg["acb_levels"])
    cfg["n_bi"] = len(cfg["bi_levels_ms"])
    cfg["n_nrep"] = len(cfg["nrep_levels"])
    cfg["n_actions"] = cfg["n_acb"] * cfg["n_bi"] * cfg["n_nrep"]
    cfg["n_states"] = (cfg["load_bins"] * cfg["elev_bins"]
                       * cfg["doppler_bins"] * cfg["pass_phase_bins"])

    return cfg


def set_global_seed(seed: int):
    """Set global random seed for reproducibility.

    Parameters
    ----------
    seed : int
        Random seed value.

    Returns
    -------
    np.random.Generator
        A seeded numpy random generator.
    """
    np.random.seed(seed)
    return np.random.default_rng(seed)
