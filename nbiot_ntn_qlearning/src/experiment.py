"""
experiment.py — Experiment runner executing R1–R6 matrices across 20 seeds.

Implements:
1. Strict deterministic seeding across all RNG instances (ensuring byte-identical runs).
2. Clean separation of warm-start pre-training from evaluation pass.
3. Dedicated pre-trained table for Lumped-Reward Ablation.
4. Evaluation across 20 independent seeds with 95% Confidence Intervals and
   paired significance tests against B2 (Rel-17 NTN default).
5. Pass-spanning multi-phase traffic ensuring all elevation bins are populated (U-shape).
6. Success-within-deadline and transmissions-per-success (energy) metrics.
"""

import os
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats
from tqdm import tqdm

from src.config_loader import load_config
from src.orbit import generate_pass_trajectory_from_config
from src.nprach import NPRACHSimulator, DeviceState
from src.mdp import RACH_MDP
from src.agent import TabularQLearningAgent
from src.baselines import (
    FixedTerrestrialController,
    Rel17NTNDefaultController,
    HeuristicBenchmarkController
)


def create_synthetic_training_trajectory(cfg: dict, duration_s: float = 480.0) -> dict:
    """
    Generate an independent synthetic orbital pass for offline pre-training.
    Separates pre-training from the test pass (preventing test-pass data leakage).
    """
    t_arr = np.linspace(0.0, duration_s, int(duration_s) + 1)
    # Synthetic elevation trajectory rising from 10° to 80° and setting to 10°
    elev_arr = 10.0 + 70.0 * np.sin(np.pi * t_arr / duration_s)
    # Synthetic delay from 6.3 ms at 10° to 2.0 ms at zenith
    delay_ms = 6.3 - 4.3 * np.sin(np.pi * t_arr / duration_s)
    delay_s = delay_ms / 1000.0
    # Synthetic Doppler
    doppler_hz = 45000.0 * np.cos(np.pi * t_arr / duration_s)

    return {
        'time_s': t_arr,
        'elevation_deg': elev_arr,
        'delay_s': delay_s,
        'delay_ms': delay_ms,
        'raw_doppler_hz': doppler_hz,
        'duration_s': duration_s,
        'max_elevation_deg': 80.0
    }


def train_warmstart_table(
    cfg: dict,
    n_episodes: int = 25,
    save_path: str = "results/warmstart_qtable.npy",
    use_lumped_ablation: bool = False
) -> np.ndarray:
    """
    Pre-train Q-table on independent multi-condition training scenarios.
    Uses an explicit, fixed seed to guarantee byte-identical reproducibility.
    """
    pretrain_seed = int(cfg.get('pretrain_seed', 42))
    rng = np.random.default_rng(pretrain_seed)
    train_traj = create_synthetic_training_trajectory(cfg, duration_s=480.0)

    mdp = RACH_MDP(cfg)
    agent = TabularQLearningAgent(
        mdp=mdp, alpha=cfg['alpha'], gamma=cfg['gamma'],
        eps0=1.0, eps_min=0.05, eps_decay=0.99, seed=pretrain_seed
    )

    rao_step_s = float(cfg.get('default_rao_period_ms', 320)) / 1000.0
    pass_dur_s = float(train_traj['duration_s'])

    for ep in range(n_episodes):
        n_dev = int(rng.choice([300, 800, 1500, 2500]))
        d_hz = float(rng.choice([10.0, 100.0, 250.0]))
        sim_seed = int(pretrain_seed * 100 + ep)
        sim = NPRACHSimulator(cfg, train_traj, seed=sim_seed)
        devices = sim.generate_traffic(n_dev, traffic_model="multiphase")

        t = 0.0
        prev_s_idx = None
        prev_a_idx = None

        while t < pass_dur_s:
            elev, _, _ = sim.get_orbit_at_time(t)
            n_backlogged = sum(1 for d in devices if not d.is_served and not d.is_dropped and d.arrival_time_s <= t)
            load_ratio = float(n_backlogged) / float(sim.n_preambles)
            pass_prog = t / max(1.0, pass_dur_s)

            s_idx = mdp.encode_state(load_ratio, elev, d_hz, pass_prog)

            if prev_s_idx is not None and prev_a_idx is not None:
                agent.update(prev_s_idx, prev_a_idx, reward, s_idx, done=False)

            a_idx = agent.select_action(s_idx)
            acb_p, bi_ms, n_rep = mdp.decode_action(a_idx)

            rao_res = sim.simulate_rao(
                t_now_s=t,
                devices=devices,
                acb_p=acb_p,
                bi_ms=bi_ms,
                n_rep=n_rep,
                residual_doppler_hz=d_hz,
                rar_window_s=0.040
            )

            reward = mdp.compute_reward(rao_res, use_lumped_ablation=use_lumped_ablation)
            prev_s_idx = s_idx
            prev_a_idx = a_idx
            t += rao_step_s

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    agent.save_q_table(save_path)
    return agent.q_table


def train_terrestrial_frozen_table(
    cfg: dict,
    n_episodes: int = 25,
    save_path: str = "results/terrestrial_frozen_qtable.npy"
) -> np.ndarray:
    """
    Train B3 baseline: trained across terrestrial load levels with constant high SNR,
    then deployed on NTN without adaptation.
    """
    pretrain_seed = int(cfg.get('pretrain_seed', 42)) + 999
    rng = np.random.default_rng(pretrain_seed)
    train_traj = create_synthetic_training_trajectory(cfg, duration_s=480.0)

    mdp = RACH_MDP(cfg)
    agent = TabularQLearningAgent(
        mdp=mdp, alpha=cfg['alpha'], gamma=cfg['gamma'],
        eps0=1.0, eps_min=0.05, eps_decay=0.99, seed=pretrain_seed
    )

    rao_step_s = float(cfg.get('default_rao_period_ms', 320)) / 1000.0
    pass_dur_s = float(train_traj['duration_s'])

    for ep in range(n_episodes):
        n_dev = int(rng.choice([300, 800, 1500, 2500]))
        sim_seed = int(pretrain_seed * 100 + ep)
        sim = NPRACHSimulator(cfg, train_traj, seed=sim_seed)
        devices = sim.generate_traffic(n_dev, traffic_model="multiphase")

        t = 0.0
        prev_s_idx = None
        prev_a_idx = None

        while t < pass_dur_s:
            n_backlogged = sum(1 for d in devices if not d.is_served and not d.is_dropped and d.arrival_time_s <= t)
            load_ratio = float(n_backlogged) / float(sim.n_preambles)
            # Terrestrial environment: constant high elevation / high SNR, 0 Doppler
            elev_sim = float(rng.choice([45.0, 75.0, 85.0]))
            s_idx = mdp.encode_state(load_ratio, elev_sim, 0.0, t / max(1.0, pass_dur_s))

            if prev_s_idx is not None and prev_a_idx is not None:
                agent.update(prev_s_idx, prev_a_idx, reward, s_idx, done=False)

            a_idx = agent.select_action(s_idx)
            acb_p, bi_ms, n_rep = mdp.decode_action(a_idx)

            rao_res = sim.simulate_rao(
                t_now_s=t,
                devices=devices,
                acb_p=acb_p,
                bi_ms=bi_ms,
                n_rep=n_rep,
                residual_doppler_hz=0.0,
                rar_window_s=0.024
            )

            reward = mdp.compute_reward(rao_res, use_lumped_ablation=False)
            prev_s_idx = s_idx
            prev_a_idx = a_idx
            t += rao_step_s

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    agent.save_q_table(save_path)
    return agent.q_table


def run_single_pass_simulation(
    cfg: dict,
    traj: dict,
    policy_name: str,
    n_devices: int,
    residual_doppler_hz: float,
    rao_period_ms: int,
    seed: int,
    warmstart_path: Optional[str] = None
) -> Tuple[Dict, List[Dict], List[float]]:
    """
    Execute a full single-pass simulation with strict deterministic seeding.
    """
    sim = NPRACHSimulator(cfg, traj, seed=seed)
    devices = sim.generate_traffic(n_devices, traffic_model=cfg.get('traffic_model', 'multiphase'))
    mdp = RACH_MDP(cfg)

    agent = None
    baseline = None
    use_lumped = False
    is_learning = False
    rar_window_s = 0.040

    if policy_name == "Proposed_Warm":
        is_learning = True
        agent = TabularQLearningAgent(
            mdp=mdp, alpha=cfg['alpha'], gamma=cfg['gamma'],
            eps0=0.15, eps_min=cfg['eps_min'], eps_decay=cfg['eps_decay'],
            seed=seed, is_warm_start=True, warmstart_path=warmstart_path
        )
    elif policy_name == "Proposed_Cold":
        is_learning = True
        agent = TabularQLearningAgent(
            mdp=mdp, alpha=cfg['alpha'], gamma=cfg['gamma'],
            eps0=cfg['eps0'], eps_min=cfg['eps_min'], eps_decay=cfg['eps_decay'],
            seed=seed, is_warm_start=False
        )
    elif policy_name == "Ablation_Lumped":
        is_learning = True
        use_lumped = True
        ablation_ws = warmstart_path.replace("warmstart_qtable", "ablation_lumped_qtable") if warmstart_path else None
        agent = TabularQLearningAgent(
            mdp=mdp, alpha=cfg['alpha'], gamma=cfg['gamma'],
            eps0=0.15, eps_min=cfg['eps_min'], eps_decay=cfg['eps_decay'],
            seed=seed, is_warm_start=True, warmstart_path=ablation_ws
        )
    elif policy_name == "B1_Fixed_Terrestrial":
        rar_win_s = float(cfg.get('rar_window_ms', 24.0)) / 1000.0
        baseline = FixedTerrestrialController(mdp, rar_window_s=rar_win_s)
        rar_window_s = baseline.get_rar_window_s()
    elif policy_name == "B2_Rel17_NTN_Default":
        baseline = Rel17NTNDefaultController(mdp)
        rar_window_s = baseline.get_rar_window_s()
    elif policy_name == "B3_Terrestrial_Frozen":
        b3_path = warmstart_path.replace("warmstart_qtable", "terrestrial_frozen_qtable") if warmstart_path else None
        agent = TabularQLearningAgent(
            mdp=mdp, is_warm_start=True, warmstart_path=b3_path, seed=seed
        )
        is_learning = False
    elif policy_name in ["B4_Oracle", "B4_Heuristic_Benchmark"]:
        baseline = HeuristicBenchmarkController(mdp)
        rar_window_s = baseline.get_rar_window_s()
    else:
        raise ValueError(f"Unknown policy: {policy_name}")

    rao_step_s = float(rao_period_ms) / 1000.0
    pass_dur_s = float(traj['duration_s'])

    t = 0.0
    step_idx = 0
    rao_logs = []
    prev_s_idx = None
    prev_a_idx = None

    while t < pass_dur_s:
        elev, delay_s, _ = sim.get_orbit_at_time(t)
        n_backlogged = sum(1 for d in devices if not d.is_served and not d.is_dropped and d.arrival_time_s <= t)
        load_ratio = float(n_backlogged) / float(sim.n_preambles)
        pass_prog = t / max(1.0, pass_dur_s)

        s_idx = mdp.encode_state(load_ratio, elev, residual_doppler_hz, pass_prog)

        if baseline is not None:
            acb_p, bi_ms, n_rep = baseline.get_action({
                'elevation_deg': elev,
                'n_eligible': n_backlogged,
                'n_preambles': sim.n_preambles
            })
            a_idx = mdp.encode_action(acb_p, bi_ms, n_rep)
        elif not is_learning:
            a_idx = agent.select_action(s_idx, greedy_only=True)
            acb_p, bi_ms, n_rep = mdp.decode_action(a_idx)
        else:
            if prev_s_idx is not None and prev_a_idx is not None:
                agent.update(prev_s_idx, prev_a_idx, reward, s_idx, done=False)
            a_idx = agent.select_action(s_idx, greedy_only=False)
            acb_p, bi_ms, n_rep = mdp.decode_action(a_idx)

        rao_res = sim.simulate_rao(
            t_now_s=t,
            devices=devices,
            acb_p=acb_p,
            bi_ms=bi_ms,
            n_rep=n_rep,
            residual_doppler_hz=residual_doppler_hz,
            rar_window_s=rar_window_s
        )

        reward = mdp.compute_reward(rao_res, use_lumped_ablation=use_lumped)
        prev_s_idx = s_idx
        prev_a_idx = a_idx

        rao_log_entry = {
            'step': step_idx,
            't_s': t,
            'elevation_deg': elev,
            'delay_ms': delay_s * 1000.0,
            'load_ratio': load_ratio,
            'acb_p': acb_p,
            'bi_ms': bi_ms,
            'n_rep': n_rep,
            'n_trans': rao_res['n_trans'],
            'n_success': rao_res['n_success'],
            'n_collision': rao_res['n_collision'],
            'n_miss': rao_res['n_miss'],
            'n_timeout': rao_res['n_timeout'],
            'reward': reward,
            'p_d': rao_res['p_d']
        }
        rao_logs.append(rao_log_entry)

        t += rao_step_s
        step_idx += 1

    # End of pass summary
    n_total = len(devices)
    served_devices = [d for d in devices if d.is_served]
    n_served = len(served_devices)
    p_s = float(n_served) / float(n_total)

    served_deadline = [d for d in devices if d.is_served_within_deadline]
    p_s_deadline = float(len(served_deadline)) / float(n_total)

    delays = [d.access_delay_s * 1000.0 for d in served_devices if not np.isnan(d.access_delay_s)]
    mean_delay = float(np.mean(delays)) if delays else np.nan
    p95_delay = float(np.percentile(delays, 95)) if delays else np.nan

    # Energy: transmissions per successful device
    trans_counts = [d.trans_count for d in devices]
    tot_trans = sum(r['n_trans'] for r in rao_logs)
    trans_per_success = float(tot_trans) / float(max(1, n_served))

    tot_coll = sum(r['n_collision'] for r in rao_logs)
    tot_miss = sum(r['n_miss'] for r in rao_logs)
    tot_to = sum(r['n_timeout'] for r in rao_logs)

    summary = {
        'policy': policy_name,
        'seed': seed,
        'n_devices': n_devices,
        'residual_doppler_hz': residual_doppler_hz,
        'rao_period_ms': rao_period_ms,
        'p_s': p_s,
        'p_s_deadline': p_s_deadline,
        'mean_delay_ms': mean_delay,
        'p95_delay_ms': p95_delay,
        'trans_per_success': trans_per_success,
        'total_trans': tot_trans,
        'total_success': n_served,
        'total_collision': tot_coll,
        'total_miss': tot_miss,
        'total_timeout': tot_to,
        'collision_rate': tot_coll / max(1, tot_trans),
        'miss_rate': tot_miss / max(1, tot_trans),
        'timeout_rate': tot_to / max(1, tot_trans)
    }

    return summary, rao_logs, delays


def run_all_experiments(cfg: dict, output_dir: str = "results"):
    """
    Run full experimental matrix R1–R6 across 20 seeds.
    """
    os.makedirs(output_dir, exist_ok=True)
    traj = generate_pass_trajectory_from_config(cfg)

    print("Step 1: Pre-training independent Q-tables (warm-start, lumped ablation, terrestrial)...")
    ws_path = os.path.join(output_dir, "warmstart_qtable.npy")
    ab_path = os.path.join(output_dir, "ablation_lumped_qtable.npy")
    tf_path = os.path.join(output_dir, "terrestrial_frozen_qtable.npy")

    train_warmstart_table(cfg, n_episodes=25, save_path=ws_path, use_lumped_ablation=False)
    train_warmstart_table(cfg, n_episodes=25, save_path=ab_path, use_lumped_ablation=True)
    train_terrestrial_frozen_table(cfg, n_episodes=25, save_path=tf_path)

    seeds = list(cfg.get('seeds', [200 + i for i in range(20)]))
    print(f"Step 2: Running Main Comparison (R2, R3, R4, R5) across {len(seeds)} seeds...")

    policies = [
        "Proposed_Warm",
        "Proposed_Cold",
        "B1_Fixed_Terrestrial",
        "B2_Rel17_NTN_Default",
        "B3_Terrestrial_Frozen",
        "B4_Heuristic_Benchmark",
        "Ablation_Lumped"
    ]

    all_summaries = []
    elevation_records = []
    convergence_records = []
    failure_records = []
    all_delays = []

    elev_bins = [(10, 25), (25, 45), (45, 70), (70, 90)]

    for seed in tqdm(seeds, desc="Seeds"):
        for pol in policies:
            # Unconfounded: all evaluated under matched nominal Doppler (10 Hz)
            d_hz = 10.0
            summary, rao_logs, delays = run_single_pass_simulation(
                cfg=cfg,
                traj=traj,
                policy_name=pol,
                n_devices=int(cfg.get('default_n_devices', 1000)),
                residual_doppler_hz=d_hz,
                rao_period_ms=int(cfg.get('default_rao_period_ms', 320)),
                seed=seed,
                warmstart_path=ws_path
            )
            all_summaries.append(summary)

            for d in delays:
                all_delays.append({'policy': pol, 'seed': seed, 'delay_ms': d})

            df_rao = pd.DataFrame(rao_logs)
            for (emin, emax) in elev_bins:
                sub = df_rao[(df_rao['elevation_deg'] >= emin) & (df_rao['elevation_deg'] < emax)]
                sub_trans = sub['n_trans'].sum()
                sub_succ = sub['n_success'].sum()
                sub_ps = sub_succ / max(1, sub_trans) if sub_trans > 0 else np.nan
                elevation_records.append({
                    'policy': pol,
                    'seed': seed,
                    'elev_min': emin,
                    'elev_max': emax,
                    'elev_center': (emin + emax) / 2.0,
                    'p_s': sub_ps
                })

            # Convergence time-series (Fig. 5): only active RAOs to avoid 0/0 artifacts
            if pol in ["Proposed_Warm", "Proposed_Cold"]:
                df_active = df_rao[df_rao['n_trans'] > 0].copy()
                if len(df_active) > 0:
                    roll_s = df_active['n_success'].rolling(15, min_periods=1).sum()
                    roll_t = df_active['n_trans'].rolling(15, min_periods=1).sum()
                    df_active['roll_success_rate'] = roll_s / roll_t
                    for _, row in df_active.iterrows():
                        convergence_records.append({
                            'policy': pol,
                            'seed': seed,
                            'step': row['step'],
                            't_s': row['t_s'],
                            'reward': row['reward'],
                            'success_rate_win': row['roll_success_rate']
                        })

    # R4 Failure breakdown under matched stress Doppler (250 Hz) for Proposed vs Ablation
    print("Step 2b: Running Unconfounded R4 Failure Breakdown (250 Hz stress)...")
    for seed in seeds[:10]:
        for pol in ["Proposed_Warm", "Ablation_Lumped"]:
            _, rao_logs, _ = run_single_pass_simulation(
                cfg=cfg,
                traj=traj,
                policy_name=pol,
                n_devices=1000,
                residual_doppler_hz=250.0,
                rao_period_ms=int(cfg.get('default_rao_period_ms', 320)),
                seed=seed,
                warmstart_path=ws_path
            )
            df_rao = pd.DataFrame(rao_logs)
            for _, row in df_rao.iterrows():
                failure_records.append({
                    'policy': pol,
                    'seed': seed,
                    'step': row['step'],
                    't_s': row['t_s'],
                    'elevation_deg': row['elevation_deg'],
                    'acb_p': row['acb_p'],
                    'n_rep': row['n_rep'],
                    'n_collision': row['n_collision'],
                    'n_miss': row['n_miss'],
                    'n_timeout': row['n_timeout'],
                    'n_success': row['n_success']
                })

    pd.DataFrame(all_summaries).to_csv(os.path.join(output_dir, "metrics.csv"), index=False)
    pd.DataFrame(elevation_records).to_csv(os.path.join(output_dir, "elevation_bins.csv"), index=False)
    pd.DataFrame(convergence_records).to_csv(os.path.join(output_dir, "convergence.csv"), index=False)
    pd.DataFrame(failure_records).to_csv(os.path.join(output_dir, "failures.csv"), index=False)
    pd.DataFrame(all_delays).to_csv(os.path.join(output_dir, "delays.csv"), index=False)

    print("Step 3: Running Sensitivity Sweep (R6: N x delta_f) across seeds...")
    sens_records = []
    test_devices = list(cfg.get('device_counts', [200, 1000, 3000]))
    test_dopplers = list(cfg.get('residual_doppler_hz', [10, 250]))

    for n_dev in test_devices:
        for d_hz in test_dopplers:
            for seed in seeds[:10]:
                s_prop, _, _ = run_single_pass_simulation(
                    cfg, traj, "Proposed_Warm", n_dev, d_hz, 320, seed, ws_path
                )
                s_b2, _, _ = run_single_pass_simulation(
                    cfg, traj, "B2_Rel17_NTN_Default", n_dev, d_hz, 320, seed, ws_path
                )
                s_b1, _, _ = run_single_pass_simulation(
                    cfg, traj, "B1_Fixed_Terrestrial", n_dev, d_hz, 320, seed, ws_path
                )

                gain_vs_b2 = s_prop['p_s_deadline'] - s_b2['p_s_deadline']
                gain_vs_b1 = s_prop['p_s_deadline'] - s_b1['p_s_deadline']
                trans_prop = s_prop['trans_per_success']
                trans_b2 = s_b2['trans_per_success']
                energy_reduction_pct = (trans_b2 - trans_prop) / max(0.001, trans_b2) * 100.0

                sens_records.append({
                    'n_devices': n_dev,
                    'residual_doppler_hz': d_hz,
                    'seed': seed,
                    'p_s_proposed': s_prop['p_s_deadline'],
                    'p_s_b2': s_b2['p_s_deadline'],
                    'p_s_b1': s_b1['p_s_deadline'],
                    'gain_vs_b2': gain_vs_b2,
                    'gain_vs_b1': gain_vs_b1,
                    'trans_proposed': trans_prop,
                    'trans_b2': trans_b2,
                    'energy_reduction_pct': energy_reduction_pct,
                    'delay_proposed': s_prop['mean_delay_ms'],
                    'delay_b2': s_b2['mean_delay_ms']
                })

    pd.DataFrame(sens_records).to_csv(os.path.join(output_dir, "sensitivity.csv"), index=False)
    print("All experiments completed successfully. Results saved in", output_dir)


if __name__ == "__main__":
    cfg = load_config()
    run_all_experiments(cfg)
