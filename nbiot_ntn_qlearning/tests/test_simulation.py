"""
test_simulation.py — Automated verification and acceptance test suite.

Encodes all acceptance checks from 03_Implementation_Plan.md:
1. Orbit physics sanity: pass duration in [2, 10] min, delay in [1.8, 7.0] ms, peak Doppler in [35, 55] kHz.
2. Detection anchors: calibrated to Chougrani et al. miss <= 0.01 at anchors.
3. Failure taxonomy mutual exclusivity: n_trans == n_s + n_c + n_m + n_t strictly.
4. MDP dimensions: state space |S| = 72, action space |A| = 60.
5. Q-learning Bellman update convergence and TD error behavior.
"""

import os
import sys
import pytest
import numpy as np

# Ensure project root in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config_loader import load_config
from src.orbit import generate_pass_trajectory_from_config
from src.channel import (
    elevation_to_snr_db,
    compute_detection_probability,
    compute_miss_probability
)
from src.nprach import NPRACHSimulator
from src.mdp import RACH_MDP
from src.agent import TabularQLearningAgent
from src.baselines import (
    FixedTerrestrialController,
    Rel17NTNDefaultController,
    HeuristicBenchmarkController,
    OracleController
)
from src.experiment import run_single_pass_simulation


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def traj(cfg):
    return generate_pass_trajectory_from_config(cfg)


def test_orbit_physics(traj):
    """Phase 1 Acceptance Check: Orbit propagation sanity against literature."""
    # Pass duration ~2-10 minutes
    assert 120.0 <= traj['duration_s'] <= 600.0, f"Unexpected pass duration: {traj['duration_s']} s"

    # Peak elevation >= 35 deg (good pass)
    assert traj['max_elevation_deg'] >= 35.0, f"Peak elevation too low: {traj['max_elevation_deg']}"

    # One-way delay ~2.0 ms (zenith) to ~6.4 ms (edge)
    min_delay = float(np.min(traj['delay_ms']))
    max_delay = float(np.max(traj['delay_ms']))
    assert 1.5 <= min_delay <= 2.5, f"Zenith delay out of bounds: {min_delay} ms"
    assert 5.5 <= max_delay <= 7.0, f"Edge delay out of bounds: {max_delay} ms"

    # Raw Doppler magnitude at 2.0 GHz carrier should reach ~40-50 kHz
    peak_doppler_khz = float(np.max(np.abs(traj['raw_doppler_hz']))) / 1000.0
    assert 38.0 <= peak_doppler_khz <= 52.0, f"Peak Doppler out of bounds: {peak_doppler_khz} kHz"


def test_channel_detection_anchors(cfg):
    """Phase 2 Acceptance Check: Calibration to Chougrani et al. [5] miss curves."""
    # Anchor 1: 8.0 dB, 64 reps, 10 Hz residual Doppler -> P_miss <= 0.01
    p_miss_1 = compute_miss_probability(snr_db=8.0, residual_doppler_hz=10.0, n_rep=64)
    assert p_miss_1 <= 0.01, f"Anchor 1 failed: P_miss = {p_miss_1} > 0.01"

    # Anchor 2: 15.0 dB, 32 reps, 10 Hz residual Doppler -> P_miss <= 0.01
    p_miss_2 = compute_miss_probability(snr_db=15.0, residual_doppler_hz=10.0, n_rep=32)
    assert p_miss_2 <= 0.01, f"Anchor 2 failed: P_miss = {p_miss_2} > 0.01"

    # Monotonicity with N_rep: higher repetition should increase P_d
    pd_2 = compute_detection_probability(snr_db=-3.0, residual_doppler_hz=10.0, n_rep=2)
    pd_8 = compute_detection_probability(snr_db=-3.0, residual_doppler_hz=10.0, n_rep=8)
    pd_32 = compute_detection_probability(snr_db=-3.0, residual_doppler_hz=10.0, n_rep=32)
    assert pd_2 < pd_8 < pd_32, f"P_d not monotonic with repetition: {pd_2}, {pd_8}, {pd_32}"

    # Degradation with residual Doppler
    pd_nom = compute_detection_probability(snr_db=2.0, residual_doppler_hz=10.0, n_rep=8)
    pd_stress = compute_detection_probability(snr_db=2.0, residual_doppler_hz=250.0, n_rep=8)
    assert pd_stress < pd_nom, "P_d did not degrade under Doppler stress"


def test_nprach_failure_taxonomy_mutual_exclusivity(cfg, traj):
    """Phase 3 Acceptance Check: Every attempt is classified into exactly one cause."""
    sim = NPRACHSimulator(cfg, traj, seed=300)
    devices = sim.generate_traffic(500, traffic_model="beta")

    step_s = 0.160
    t = 0.0
    total_trans = 0
    total_causes = 0

    while t < 30.0:  # first 30 seconds
        res = sim.simulate_rao(
            t_now_s=t, devices=devices, acb_p=0.5, bi_ms=1024, n_rep=8,
            residual_doppler_hz=10.0, rar_window_s=0.040
        )
        total_trans += res['n_trans']
        total_causes += (res['n_success'] + res['n_collision'] + res['n_miss'] + res['n_timeout'])
        assert res['n_trans'] == (res['n_success'] + res['n_collision'] + res['n_miss'] + res['n_timeout'])
        t += step_s

    assert total_trans > 0, "No transmissions occurred"
    assert total_trans == total_causes, f"Taxonomy violation: {total_trans} != {total_causes}"


def test_mdp_dimensions_and_encoding(cfg):
    """Phase 4 Acceptance Check: Compact state and action spaces."""
    mdp = RACH_MDP(cfg)
    assert mdp.n_states == 72, f"Expected 72 states, got {mdp.n_states}"
    assert mdp.n_actions == 60, f"Expected 60 actions, got {mdp.n_actions}"

    # Test state encoding
    s_idx = mdp.encode_state(load_ratio=0.5, elevation_deg=30.0, residual_doppler_hz=20.0, pass_progress=0.5)
    assert 0 <= s_idx < 72, f"State index out of bounds: {s_idx}"

    # Test action decoding
    for a in range(mdp.n_actions):
        p, bi, nrep = mdp.decode_action(a)
        assert p in cfg['acb_levels']
        assert bi in cfg['bi_levels_ms']
        assert nrep in cfg['nrep_levels']


def test_qlearning_agent_update(cfg):
    """Phase 4 Acceptance Check: Bellman update and exploration decay."""
    mdp = RACH_MDP(cfg)
    agent = TabularQLearningAgent(mdp, alpha=0.1, gamma=0.9, eps0=1.0, eps_decay=0.95, seed=400)

    initial_eps = agent.eps
    td_error = agent.update(state_idx=5, action_idx=10, reward=1.0, next_state_idx=6, done=False)

    assert agent.q_table[5, 10] > 0.0, "Q-value was not updated"
    assert agent.eps < initial_eps, "Epsilon did not decay"
    assert np.isfinite(td_error), "TD error is non-finite"


def test_link_budget_snr_bounds(cfg):
    """Audit Verification: SNR derived from link budget spans ~[-5.5, +8.2] dB."""
    snr_edge = elevation_to_snr_db(
        10.0,
        altitude_km=cfg.get('altitude_km', 600.0),
        carrier_hz=cfg.get('carrier_hz', 2.0e9),
        p_tx_dbm=cfg.get('p_tx_dbm', 23.0),
        g_tx_dbi=cfg.get('g_tx_dbi', 0.0),
        g_rx_zenith_dbi=cfg.get('g_rx_zenith_dbi', 6.0),
        antenna_3db_beamwidth_deg=cfg.get('antenna_3db_beamwidth_deg', 65.0),
        noise_figure_db=cfg.get('noise_figure_db', 5.0),
        temperature_k=cfg.get('temperature_k', 290.0)
    )
    snr_zenith = elevation_to_snr_db(
        90.0,
        altitude_km=cfg.get('altitude_km', 600.0),
        carrier_hz=cfg.get('carrier_hz', 2.0e9),
        p_tx_dbm=cfg.get('p_tx_dbm', 23.0),
        g_tx_dbi=cfg.get('g_tx_dbi', 0.0),
        g_rx_zenith_dbi=cfg.get('g_rx_zenith_dbi', 6.0),
        antenna_3db_beamwidth_deg=cfg.get('antenna_3db_beamwidth_deg', 65.0),
        noise_figure_db=cfg.get('noise_figure_db', 5.0),
        temperature_k=cfg.get('temperature_k', 290.0)
    )

    # Edge SNR should be around -5.5 dB (in [-7.0, -4.0])
    assert -7.0 <= snr_edge <= -4.0, f"Edge SNR out of expected range: {snr_edge:.2f} dB"
    # Zenith SNR should be around +8.2 dB (in [+6.5, +10.0])
    assert 6.5 <= snr_zenith <= 10.0, f"Zenith SNR out of expected range: {snr_zenith:.2f} dB"
    # Dynamic range should be in [11.0, 16.0] dB
    dyn_range = snr_zenith - snr_edge
    assert 11.0 <= dyn_range <= 16.0, f"Dynamic range unexpected: {dyn_range:.2f} dB"


def test_repetition_timing_feasibility(cfg):
    """Audit Verification: Max repetition transmission time fits within RAO period."""
    # 32 repetitions * 5.6 ms per preamble symbol group = 179.2 ms
    max_nrep = max(cfg['nrep_levels'])
    symbol_group_dur_ms = 5.6
    tx_time_ms = max_nrep * symbol_group_dur_ms

    rao_period_ms = cfg.get('default_rao_period_ms', 320)
    assert rao_period_ms >= tx_time_ms, (
        f"Timing infeasible: Max repetition {max_nrep} takes {tx_time_ms} ms, "
        f"exceeding RAO period {rao_period_ms} ms"
    )
    assert rao_period_ms == 320, f"Expected calibrated RAO period 320 ms, got {rao_period_ms}"


def test_multiphase_traffic_coverage(cfg, traj):
    """Audit Verification: Multiphase traffic populates all 4 elevation bins."""
    sim = NPRACHSimulator(cfg, traj, seed=555)
    devices = sim.generate_traffic(1000, traffic_model="multiphase")

    # Map arrivals to elevation at time of arrival
    elev_bins = [(10, 25), (25, 45), (45, 70), (70, 90)]
    counts = {b: 0 for b in elev_bins}

    for d in devices:
        elev, _, _ = sim.get_orbit_at_time(d.arrival_time_s)
        for (emin, emax) in elev_bins:
            if emin <= elev < emax:
                counts[(emin, emax)] += 1
                break

    for b, cnt in counts.items():
        assert cnt >= 150, f"Bin {b} has insufficient arrivals: {cnt} < 150"


def test_byte_identical_reproducibility(cfg, traj):
    """Audit Verification: Seed determinism produces byte-identical simulation outcomes."""
    fixed_seed = 777
    m1, _, _ = run_single_pass_simulation(
        cfg=cfg, traj=traj, policy_name="B2_Rel17_NTN_Default",
        n_devices=200, residual_doppler_hz=10.0, rao_period_ms=320,
        seed=fixed_seed
    )
    m2, _, _ = run_single_pass_simulation(
        cfg=cfg, traj=traj, policy_name="B2_Rel17_NTN_Default",
        n_devices=200, residual_doppler_hz=10.0, rao_period_ms=320,
        seed=fixed_seed
    )

    assert m1['total_success'] == m2['total_success']
    assert m1['p_s'] == m2['p_s']
    assert np.isclose(m1['mean_delay_ms'], m2['mean_delay_ms'], rtol=1e-12, atol=1e-12)
    assert np.isclose(m1['p95_delay_ms'], m2['p95_delay_ms'], rtol=1e-12, atol=1e-12)
    assert m1['total_collision'] == m2['total_collision']
    assert m1['total_miss'] == m2['total_miss']
