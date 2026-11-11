"""
Acceptance tests (03_Implementation_Plan.md). Run: python -m pytest -q
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent import TabularQLearningAgent  # noqa: E402
from src.channel import (detection_probability, outdoor_snr_db, snr99_db,  # noqa: E402
                         timeout_probability)
from src.config_loader import load_config  # noqa: E402
from src.experiment import run_episode  # noqa: E402
from src.mdp import RACH_MDP, estimate_pool  # noqa: E402
from src.nprach import NPRACHSimulator  # noqa: E402
from src.orbit import get_passes  # noqa: E402


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def passes(cfg):
    return get_passes(cfg, "test")


def _slant(e, h=600.0):
    s = 6371.0 * np.sin(np.radians(e))
    return np.sqrt(s * s + 2 * 6371.0 * h + h * h) - s


# ----------------------------------------------------------------- Phase 1: orbit
def test_pass_geometry_bounds(passes):
    assert len(passes) >= 8, "need several test geometries"
    for p in passes:
        assert 300.0 <= p["duration_s"] <= 540.0          # 10 deg mask, ~577 km: <= ~8.5 min
        assert p["max_elevation_deg"] >= 20.0
        assert 1.8 <= p["delay_ms"].min() and p["delay_ms"].max() <= 6.6
        assert np.abs(p["raw_doppler_hz"]).max() <= 48e3   # [5],[11]: up to ~41-46 kHz
    assert max(np.abs(p["raw_doppler_hz"]).max() for p in passes) >= 40e3


def test_train_and_test_passes_are_disjoint(cfg, passes):
    train = get_passes(cfg, "train")
    assert {p["satellite"] for p in train}.isdisjoint({p["satellite"] for p in passes})


# ----------------------------------------------------------------- Phase 2: channel
def test_link_budget_reproduces_dwivedi_set4(cfg):
    assert outdoor_snr_db(_slant(90.0), 90.0, 600.0, cfg) == pytest.approx(4.93, abs=0.05)
    assert outdoor_snr_db(_slant(30.0), 30.0, 600.0, cfg) == pytest.approx(-3.14, abs=0.15)


def test_detection_anchors_3gpp(cfg):
    # TS 36.104 Table 8.5.3.2.1-1 (AWGN, 2 Rx) + 3 dB for one Rx branch
    assert detection_probability(-2.1 + 3.0, 8, cfg, "gnss_assisted") == pytest.approx(0.99, abs=1e-3)
    assert detection_probability(-6.8 + 3.0, 32, cfg, "gnss_assisted") == pytest.approx(0.99, abs=1e-3)


def test_detection_gnss_free_fit_to_chougrani(cfg):
    fit = snr99_db(np.array([32, 64]), cfg, "gnss_free")
    target = np.array([7.9, 2.8])                          # [5], MD-RTD 800 us
    assert np.max(np.abs(fit - target)) <= 1.5             # documented residual +/-1.4 dB


def test_detection_monotone(cfg):
    pds = detection_probability(-4.0, np.array([2, 8, 32, 128]), cfg)
    assert np.all(np.diff(pds) > 0)
    assert detection_probability(-4.0, 32, cfg, "gnss_free") < detection_probability(-4.0, 32, cfg)


def test_timeout_probability_shape(cfg):
    assert timeout_probability(90.0, cfg) == 0.0
    assert timeout_probability(10.0, cfg) > timeout_probability(60.0, cfg) > 0.0


# ----------------------------------------------------------------- Phase 3: access model
def test_outcomes_mutually_exclusive_and_observables_consistent(cfg, passes):
    sim = NPRACHSimulator(cfg, passes[1], 1500, [1, 2, 3])
    for e in range(sim.n_epochs):
        r = sim.step(e, 1.0, 1024, cfg["rep_profiles"][1])
        assert r["n_trans"] == r["n_success"] + r["n_collision"] + r["n_miss"] + r["n_timeout"]
        for pl in r["pools"]:
            assert 0 <= pl["idle_obs"] <= pl["R"]
            assert pl["flagged"] + pl["succ"] <= pl["R"]
    assert sim.served.sum() + sim.dropped.sum() <= sim.n


def test_no_timeouts_without_degraded_meters(cfg, passes):
    sim = NPRACHSimulator(cfg, passes[0], 500, [4, 0, 1], gnss_degraded_frac=0.0)
    assert sum(sim.step(e, 1.0, 1024, cfg["rep_profiles"][2])["n_timeout"] for e in range(sim.n_epochs)) == 0


def test_heavy_load_without_barring_collides(cfg, passes):
    sim = NPRACHSimulator(cfg, passes[1], 5000, [5, 0, 1])
    tot = [sim.step(e, 1.0, 256, cfg["rep_profiles"][2]) for e in range(sim.n_epochs)]
    n_c = sum(r["n_collision"] for r in tot)
    n_t = sum(r["n_trans"] for r in tot)
    assert n_c / n_t > 0.5


def test_more_repetitions_fewer_misses(cfg, passes):
    def miss_rate(prof):
        sim = NPRACHSimulator(cfg, passes[0], 300, [6, 0, 1], coupling_loss_mean_db=5.0)
        rs = [sim.step(e, 1.0, 4096, cfg["rep_profiles"][prof]) for e in range(sim.n_epochs)]
        return sum(r["n_miss"] for r in rs) / max(1, sum(r["n_trans"] for r in rs))
    assert miss_rate(2) < miss_rate(0)


# ----------------------------------------------------------------- Phase 4: MDP / agent
def test_estimator_tracks_load_and_misses():
    rng = np.random.default_rng(0)
    R, pcd = 24, 0.95
    for n_true, pd in [(10, 0.7), (24, 0.7), (48, 1.0)]:
        est, tru = [], []
        for _ in range(1500):
            sc = rng.integers(0, R, n_true)
            cnt = np.bincount(sc, minlength=R)
            single = cnt[sc] == 1
            det = rng.random(n_true) < pd
            coll = np.flatnonzero(cnt >= 2)
            cdet = rng.random(coll.size) < 1 - (1 - pd) ** cnt[coll]
            flag = cdet & (rng.random(coll.size) < pcd)
            succ = (single & det).sum()
            n, _, m, _, _ = estimate_pool(R, succ, flag.sum(), cdet.sum() - flag.sum(), pcd)
            est.append((n, m))
            tru.append((n_true, (single & ~det).sum()))
        e, t = np.mean(est, 0), np.mean(tru, 0)
        assert abs(e[0] - t[0]) / t[0] < 0.15
        assert abs(e[1] - t[1]) < 1.0


def test_reward_uses_only_observables(cfg, passes):
    """Changing ground-truth fields that a gNB cannot see must not change the reward."""
    mdp = RACH_MDP(cfg)
    sim = NPRACHSimulator(cfg, passes[1], 1000, [7, 0, 1])
    for e in range(40):
        rao = sim.step(e, 1.0, 1024, cfg["rep_profiles"][1])
    r1 = RACH_MDP(cfg).observe(rao)
    tampered = dict(rao, n_collision=999, n_miss=999, n_timeout=999, n_success=999, n_pending=999)
    tampered["pools"] = [{k: v for k, v in p.items() if k != "n_tx_true"} for p in rao["pools"]]
    assert RACH_MDP(cfg).observe(tampered) == r1


def test_state_and_action_spaces(cfg):
    mdp = RACH_MDP(cfg)
    assert mdp.n_states == cfg["n_states"] == 108
    assert mdp.n_actions == cfg["n_actions"] == 60
    for a in range(mdp.n_actions):
        p, bi, reps = mdp.decode_action(a)
        assert p in cfg["acb_levels"] and bi in cfg["bi_levels_ms"] and reps in cfg["rep_profiles"]


def test_q_update():
    ag = TabularQLearningAgent(4, 3, 0.1, 0.9, 1.0, 0.05, 0.5, np.random.default_rng(0))
    ag.update(0, 1, 1.0, 2)
    assert ag.q[0, 1] == pytest.approx(0.1) and ag.eps == 0.5


# ----------------------------------------------------------------- reproducibility
def test_byte_identical_reruns(cfg, passes):
    a = run_episode(cfg, passes[2], "Proposed_Cold", 500, [9, 2, 1])[0]
    b = run_episode(cfg, passes[2], "Proposed_Cold", 500, [9, 2, 1])[0]
    assert a == b


def test_common_random_numbers_across_policies(cfg, passes):
    s1 = NPRACHSimulator(cfg, passes[0], 300, [3, 0, 1])
    s2 = NPRACHSimulator(cfg, passes[0], 300, [3, 0, 1])
    assert np.array_equal(s1.arrival, s2.arrival) and np.array_equal(s1.loss_db, s2.loss_db)
