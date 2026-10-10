"""
experiment.py — Experiment runner executing R1–R6 matrices across seeds.

Implements:
1. Pre-training routines:
   - Warm-start Q-table: trained on mixed conditions to cover state bins.
   - Terrestrial-frozen Q-table (B3): trained on terrestrial conditions only.
2. Single-pass simulation engine supporting:
   - Proposed (Cause-Separated Tabular Q-Learning)
   - B1 (Fixed Terrestrial)
   - B2 (Rel-17 NTN Default)
   - B3 (Terrestrial-Frozen Q-table)
   - B4 (Oracle)
   - Ablation (Lumped-Reward Q-Learning)
   - Proposed Cold-Start vs Warm-Start
3. Comprehensive logging:
   - results/metrics.csv (headline comparison table)
   - results/elevation_bins.csv (data for Fig. 2 U-shape)
   - results/delays.csv (data for Fig. 3 delay CDF)
   - results/failures.csv (data for Fig. 4 failure cause breakdown)
   - results/convergence.csv (data for Fig. 5 cold vs warm convergence)
   - results/sensitivity.csv (data for Fig. 6 N x delta_f sensitivity)
"""

import os
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

from src.config_loader import load_config
from src.orbit import generate_pass_trajectory_from_config
from src.nprach import NPRACHSimulator, DeviceState
from src.mdp import RACH_MDP
from src.agent import TabularQLearningAgent
from src.baselines import (
    FixedTerrestrialController,
    Rel17NTNDefaultController,
    OracleController
)


def train_warmstart_table(cfg: dict, traj: dict, n_episodes: int = 15, save_path: str = "results/warmstart_qtable.npy") -> np.ndarray:
    """
    Pre-train warm-start Q-table across varied load and channel conditions
    to ensure all state bins are well-visited before pass deployment.
    """
    mdp = RACH_MDP(cfg)
    agent = TabularQLearningAgent(mdp=mdp, alpha=cfg['alpha'], gamma=cfg['gamma'],
                                  eps0=1.0, eps_min=0.05, eps_decay=0.99, seed=101)

    rao_step_s = float(cfg.get('default_rao_period_ms', 160)) / 1000.0
    pass_dur_s = float(traj['duration_s'])

    for ep in range(n_episodes):
        # Vary device load and Doppler across episodes for diversity
        n_dev = int(np.random.choice([300, 800, 1500, 2500]))
        d_hz = float(np.random.choice([10.0, 100.0, 250.0]))
        sim = NPRACHSimulator(cfg, traj, seed=1000 + ep)
        devices = sim.generate_traffic(n_dev, traffic_model="beta")

        t = 0.0
        prev_s_idx = None
        prev_a_idx = None

        while t < pass_dur_s:
            elev, _, _ = sim.get_orbit_at_time(t)
            # Count backlogged devices
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

            reward = mdp.compute_reward(rao_res, use_lumped_ablation=False)
            prev_s_idx = s_idx
            prev_a_idx = a_idx
            t += rao_step_s

    agent.save_q_table(save_path)
    return agent.q_table


def train_terrestrial_frozen_table(cfg: dict, traj: dict, n_episodes: int = 15, save_path: str = "results/terrestrial_frozen_qtable.npy") -> np.ndarray:
    """
    Train B3 baseline table: trained strictly under terrestrial conditions
    (constant high SNR = 15 dB, zero propagation delay, zero Doppler).
    """
    mdp = RACH_MDP(cfg)
    agent = TabularQLearningAgent(mdp=mdp, alpha=cfg['alpha'], gamma=cfg['gamma'],
                                  eps0=1.0, eps_min=0.05, eps_decay=0.99, seed=202)

    rao_step_s = float(cfg.get('default_rao_period_ms', 160)) / 1000.0
    pass_dur_s = float(traj['duration_s'])

    for ep in range(n_episodes):
        n_dev = int(np.random.choice([300, 800, 1500, 2500]))
        sim = NPRACHSimulator(cfg, traj, seed=2000 + ep)
        devices = sim.generate_traffic(n_dev, traffic_model="beta")

        t = 0.0
        prev_s_idx = None
        prev_a_idx = None

        while t < pass_dur_s:
            n_backlogged = sum(1 for d in devices if not d.is_served and not d.is_dropped and d.arrival_time_s <= t)
            load_ratio = float(n_backlogged) / float(sim.n_preambles)
            # Under terrestrial conditions: elevation is nominally zenith (90°), doppler is 0 Hz
            s_idx = mdp.encode_state(load_ratio, 85.0, 0.0, t / max(1.0, pass_dur_s))

            if prev_s_idx is not None and prev_a_idx is not None:
                agent.update(prev_s_idx, prev_a_idx, reward, s_idx, done=False)

            a_idx = agent.select_action(s_idx)
            acb_p, bi_ms, n_rep = mdp.decode_action(a_idx)

            # Terrestrial channel: high SNR, short terrestrial RAR
            rao_res = sim.simulate_rao(
                t_now_s=t,
                devices=devices,
                acb_p=acb_p,
                bi_ms=bi_ms,
                n_rep=n_rep,
                residual_doppler_hz=0.0,
                rar_window_s=0.040
            )

            reward = mdp.compute_reward(rao_res, use_lumped_ablation=False)
            prev_s_idx = s_idx
            prev_a_idx = a_idx
            t += rao_step_s

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
    Execute a full single-pass simulation.
    
    Returns
    -------
    summary : dict
        Aggregated metrics (P_s, mean delay, 95th pct delay, failure split, etc.)
    rao_logs : list of dict
        Per-RAO metrics time-series
    delays_ms : list of float
        Individual served device access delays
    """
    sim = NPRACHSimulator(cfg, traj, seed=seed)
    devices = sim.generate_traffic(n_devices, traffic_model=cfg.get('traffic_model', 'beta'))
    mdp = RACH_MDP(cfg)

    # Initialize controller based on policy_name
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
        agent = TabularQLearningAgent(
            mdp=mdp, alpha=cfg['alpha'], gamma=cfg['gamma'],
            eps0=0.15, eps_min=cfg['eps_min'], eps_decay=cfg['eps_decay'],
            seed=seed, is_warm_start=True, warmstart_path=warmstart_path
        )
    elif policy_name == "B1_Fixed_Terrestrial":
        baseline = FixedTerrestrialController(mdp)
        rar_window_s = baseline.get_rar_window_s()
    elif policy_name == "B2_Rel17_NTN_Default":
        baseline = Rel17NTNDefaultController(mdp)
        rar_window_s = baseline.get_rar_window_s()
    elif policy_name == "B3_Terrestrial_Frozen":
        # Frozen policy: greedy on terrestrial table, no online updates
        b3_path = warmstart_path.replace("warmstart_qtable", "terrestrial_frozen_qtable") if warmstart_path else None
        agent = TabularQLearningAgent(
            mdp=mdp, is_warm_start=True, warmstart_path=b3_path, seed=seed
        )
        is_learning = False
    elif policy_name == "B4_Oracle":
        baseline = OracleController(mdp)
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
            # Frozen policy greedy evaluation
            a_idx = agent.select_action(s_idx, greedy_only=True)
            acb_p, bi_ms, n_rep = mdp.decode_action(a_idx)
        else:
            # Active learning agent
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

        # Log RAO step
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

    delays = [d.access_delay_s * 1000.0 for d in served_devices if not np.isnan(d.access_delay_s)]
    mean_delay = float(np.mean(delays)) if delays else np.nan
    p95_delay = float(np.percentile(delays, 95)) if delays else np.nan

    # Energy proxy: mean transmissions per device
    trans_counts = [d.trans_count for d in devices]
    mean_trans = float(np.mean(trans_counts)) if trans_counts else 0.0

    # Total failure counts
    tot_coll = sum(r['n_collision'] for r in rao_logs)
    tot_miss = sum(r['n_miss'] for r in rao_logs)
    tot_to = sum(r['n_timeout'] for r in rao_logs)
    tot_trans = sum(r['n_trans'] for r in rao_logs)

    summary = {
        'policy': policy_name,
        'seed': seed,
        'n_devices': n_devices,
        'residual_doppler_hz': residual_doppler_hz,
        'rao_period_ms': rao_period_ms,
        'p_s': p_s,
        'mean_delay_ms': mean_delay,
        'p95_delay_ms': p95_delay,
        'mean_transmissions': mean_trans,
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
    Run full experimental matrix R1–R6 across >=10 seeds.
    Emits all CSVs needed for plotting and analysis.
    """
    os.makedirs(output_dir, exist_ok=True)
    traj = generate_pass_trajectory_from_config(cfg)

    print("Step 1: Pre-training warm-start and terrestrial-frozen Q-tables...")
    ws_path = os.path.join(output_dir, "warmstart_qtable.npy")
    tf_path = os.path.join(output_dir, "terrestrial_frozen_qtable.npy")
    train_warmstart_table(cfg, traj, n_episodes=20, save_path=ws_path)
    train_terrestrial_frozen_table(cfg, traj, n_episodes=20, save_path=tf_path)

    seeds = list(cfg.get('seeds', [200, 201, 202, 203, 204, 205, 206, 207, 208, 209]))
    print(f"Step 2: Running Main Comparison (R2, R3, R4, R5) across {len(seeds)} seeds...")

    policies = [
        "Proposed_Warm",
        "Proposed_Cold",
        "B1_Fixed_Terrestrial",
        "B2_Rel17_NTN_Default",
        "B3_Terrestrial_Frozen",
        "B4_Oracle",
        "Ablation_Lumped"
    ]

    all_summaries = []
    elevation_records = []
    convergence_records = []
    failure_records = []
    all_delays = []

    # Elevation bin definitions for Fig. 2
    elev_bins = [(10, 25), (25, 45), (45, 70), (70, 90)]

    for seed in tqdm(seeds, desc="Seeds"):
        for pol in policies:
            # Default run conditions: N = 1000, delta_f = 10 Hz (nominal) or 250 Hz (stress for R4)
            d_hz = 250.0 if pol == "Ablation_Lumped" else 10.0
            summary, rao_logs, delays = run_single_pass_simulation(
                cfg=cfg,
                traj=traj,
                policy_name=pol,
                n_devices=1000,
                residual_doppler_hz=d_hz,
                rao_period_ms=160,
                seed=seed,
                warmstart_path=ws_path
            )
            all_summaries.append(summary)

            # Record delays for Fig. 3
            for d in delays:
                all_delays.append({'policy': pol, 'seed': seed, 'delay_ms': d})

            # Record elevation-binned performance for Fig. 2
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

            # Record convergence time-series for Fig. 5 (Cold vs Warm)
            if pol in ["Proposed_Warm", "Proposed_Cold"]:
                # Moving window success rate and cumulative reward
                w_size = 20
                roll_s = df_rao['n_success'].rolling(w_size, min_periods=1).sum()
                roll_t = df_rao['n_trans'].rolling(w_size, min_periods=1).sum()
                for idx, row in df_rao.iterrows():
                    convergence_records.append({
                        'policy': pol,
                        'seed': seed,
                        'step': row['step'],
                        't_s': row['t_s'],
                        'reward': row['reward'],
                        'success_rate_win': roll_s.iloc[idx] / max(1, roll_t.iloc[idx])
                    })

            # Record failure breakdown for Fig. 4 (Proposed vs Ablation at 250 Hz)
            if pol in ["Proposed_Warm", "Ablation_Lumped"]:
                for idx, row in df_rao.iterrows():
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

    # Save dataframes
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
            for seed in seeds[:5]:  # 5 seeds for sensitivity sweep
                # Run Proposed_Warm
                s_prop, _, _ = run_single_pass_simulation(
                    cfg, traj, "Proposed_Warm", n_dev, d_hz, 160, seed, ws_path
                )
                # Run B1_Fixed_Terrestrial
                s_b1, _, _ = run_single_pass_simulation(
                    cfg, traj, "B1_Fixed_Terrestrial", n_dev, d_hz, 160, seed, ws_path
                )
                # Run B2_Rel17_NTN_Default
                s_b2, _, _ = run_single_pass_simulation(
                    cfg, traj, "B2_Rel17_NTN_Default", n_dev, d_hz, 160, seed, ws_path
                )

                gain_vs_b1 = s_prop['p_s'] - s_b1['p_s']
                gain_vs_b2 = s_prop['p_s'] - s_b2['p_s']

                sens_records.append({
                    'n_devices': n_dev,
                    'residual_doppler_hz': d_hz,
                    'seed': seed,
                    'p_s_proposed': s_prop['p_s'],
                    'p_s_b1': s_b1['p_s'],
                    'p_s_b2': s_b2['p_s'],
                    'gain_vs_b1': gain_vs_b1,
                    'gain_vs_b2': gain_vs_b2,
                    'delay_proposed': s_prop['mean_delay_ms'],
                    'delay_b1': s_b1['mean_delay_ms']
                })

    pd.DataFrame(sens_records).to_csv(os.path.join(output_dir, "sensitivity.csv"), index=False)
    print("All experiments completed successfully. Results saved in", output_dir)


if __name__ == "__main__":
    cfg = load_config()
    run_all_experiments(cfg)
