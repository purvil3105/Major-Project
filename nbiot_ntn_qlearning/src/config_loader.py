"""Load and validate the master YAML configuration."""

import hashlib
import json
import os

import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolve_path(path: str) -> str:
    """Resolve a project-relative path."""
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)


def load_config(path: str = None) -> dict:
    """Load config/default.yaml (or `path`) and add derived quantities."""
    path = resolve_path(path or os.path.join("config", "default.yaml"))
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    profiles = [[int(x) for x in p] for p in cfg["rep_profiles"]]
    cfg["rep_profiles"] = profiles
    assert all(len(p) == len(cfg["ce_subcarriers"]) for p in profiles), \
        "each repetition profile needs one value per CE level"
    assert sum(cfg["ce_subcarriers"]) <= 48, "NB-IoT NPRACH has at most 48 subcarriers"
    assert 0 < cfg["alpha"] <= 1 and 0 <= cfg["gamma"] < 1
    assert cfg["detection_regime"] in cfg["detection_regimes"]

    cfg["n_actions"] = len(cfg["acb_levels"]) * len(cfg["bi_levels_ms"]) * len(profiles)
    cfg["n_states"] = ((len(cfg["load_thresholds"]) + 1) * (len(cfg["elev_thresholds"]) + 1)
                       * (len(cfg["phase_thresholds"]) + 1) * 4)
    return cfg


def config_hash(cfg: dict) -> str:
    """Short, stable hash of the configuration (for run manifests)."""
    blob = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]
