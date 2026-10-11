"""
plots.py — Publication-grade plotting module generating the six required research figures.

Figure 1: Orbit & pass geometry (elevation, range, delay, raw Doppler vs time).
Figure 2: Access success probability P_s vs elevation (true U-shape across policies).
Figure 3: Access delay distribution (CDF & percentiles, Proposed vs B1, B2, B3).
Figure 4: Failure-cause breakdown across pass (Proposed vs Lumped-Reward Ablation).
Figure 5: Convergence within active RAOs (Cold-start vs Warm-start).
Figure 6: Sensitivity gain heatmap/bars: Gain vs Rel-17 Standard Preset (B2).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from src.config_loader import load_config
from src.orbit import generate_pass_trajectory_from_config

# Set publication style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'lines.linewidth': 2.0,
    'grid.alpha': 0.35,
    'grid.linestyle': '--',
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

PALETTE = {
    'Proposed_Warm': '#1f77b4',
    'Proposed_Cold': '#ff7f0e',
    'B1_Fixed_Terrestrial': '#d62728',
    'B2_Rel17_NTN_Default': '#9467bd',
    'B3_Terrestrial_Frozen': '#8c564b',
    'B4_Heuristic_Benchmark': '#2ca02c',
    'Ablation_Lumped': '#e377c2',
    'collision': '#d62728',
    'miss': '#ff7f0e',
    'timeout': '#9467bd',
    'success': '#2ca02c'
}


def plot_fig1_orbit_geometry(cfg: dict, out_dir: str = "figs"):
    """
    Fig. 1: Orbit & pass geometry vs time.
    Elevation, slant range, one-way delay, and raw Doppler.
    """
    os.makedirs(out_dir, exist_ok=True)
    traj = generate_pass_trajectory_from_config(cfg)
    t_min = traj['time_s'] / 60.0

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)

    # Elevation
    axes[0, 0].plot(t_min, traj['elevation_deg'], color='#1f77b4', lw=2.2)
    axes[0, 0].axhline(10.0, color='gray', linestyle=':', label='10° Mask')
    axes[0, 0].set_ylabel('Elevation Angle [deg]')
    axes[0, 0].set_title('(a) Satellite Elevation Angle $\\epsilon(t)$')
    axes[0, 0].grid(True)
    axes[0, 0].legend(loc='lower center')

    # Slant Range
    axes[0, 1].plot(t_min, traj['slant_range_km'], color='#2ca02c', lw=2.2)
    axes[0, 1].set_ylabel('Slant Range $d(t)$ [km]')
    axes[0, 1].set_title('(b) Slant Range to Ground Site')
    axes[0, 1].grid(True)

    # One-Way Delay
    axes[1, 0].plot(t_min, traj['delay_ms'], color='#ff7f0e', lw=2.2)
    axes[1, 0].set_ylabel('One-Way Delay $\\tau(t)$ [ms]')
    axes[1, 0].set_xlabel('Pass Time [minutes]')
    axes[1, 0].set_title('(c) Propagation Delay $\\tau(t) = d(t)/c$')
    axes[1, 0].grid(True)

    # Raw Doppler
    axes[1, 1].plot(t_min, traj['raw_doppler_hz'] / 1000.0, color='#d62728', lw=2.2)
    axes[1, 1].axhline(0, color='black', linestyle='--', alpha=0.5)
    axes[1, 1].set_ylabel('Raw Doppler $f_D(t)$ [kHz]')
    axes[1, 1].set_xlabel('Pass Time [minutes]')
    axes[1, 1].set_title('(d) Raw Doppler Shift at 2.0 GHz')
    axes[1, 1].grid(True)

    plt.tight_layout()
    save_path = os.path.join(out_dir, "fig1_orbit_geometry.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved {save_path}")


def plot_fig2_elevation_success(results_dir: str = "results", out_dir: str = "figs"):
    """
    Fig. 2: Access success probability P_s vs elevation across policies.
    Displays the signature U-shape across pass-spanning traffic.
    """
    csv_path = os.path.join(results_dir, "elevation_bins.csv")
    if not os.path.exists(csv_path):
        print(f"File {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    grouped = df.groupby(['policy', 'elev_center'])['p_s'].agg(['mean', 'std', 'count']).reset_index()

    fig, ax = plt.subplots(figsize=(8.5, 6))

    policy_order = [
        ('B4_Heuristic_Benchmark', 'B4: Heuristic Channel-Aware', 'o-', '#2ca02c'),
        ('Proposed_Warm', 'Proposed (Warm Q-Learning)', 's-', '#1f77b4'),
        ('B2_Rel17_NTN_Default', 'B2: Rel-17 NTN Default', '^-', '#9467bd'),
        ('B3_Terrestrial_Frozen', 'B3: Terrestrial-Frozen Q', 'v--', '#8c564b'),
        ('B1_Fixed_Terrestrial', 'B1: Fixed Terrestrial Default', 'x:', '#d62728')
    ]

    for pol_key, label, fmt, col in policy_order:
        sub = grouped[grouped['policy'] == pol_key]
        if not sub.empty:
            sub = sub.sort_values('elev_center')
            ci95 = 1.96 * sub['std'] / np.sqrt(np.maximum(1, sub['count']))
            ax.errorbar(
                sub['elev_center'], sub['mean'], yerr=ci95,
                fmt=fmt, color=col, label=label, capsize=4, markersize=7, lw=2
            )

    ax.set_xlabel('Satellite Elevation Angle [deg]')
    ax.set_ylabel('Preamble Access Success Rate $P_s$')
    ax.set_title('Fig. 2: Access Success Rate vs Satellite Elevation (U-Shape Characterization)')
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True)
    ax.legend(loc='lower right', framealpha=0.95)

    plt.tight_layout()
    save_path = os.path.join(out_dir, "fig2_elevation_success.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved {save_path}")


def plot_fig3_delay_cdf(results_dir: str = "results", out_dir: str = "figs"):
    """
    Fig. 3: Empirical Cumulative Distribution Function (CDF) of Access Delay.
    """
    csv_path = os.path.join(results_dir, "delays.csv")
    if not os.path.exists(csv_path):
        print(f"File {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(8.5, 6))

    target_policies = [
        ('Proposed_Warm', 'Proposed (Warm-Start Q)', '#1f77b4', '-'),
        ('B2_Rel17_NTN_Default', 'B2: Rel-17 NTN Default', '#9467bd', '-.'),
        ('B3_Terrestrial_Frozen', 'B3: Terrestrial-Frozen Q', '#8c564b', '--'),
        ('B1_Fixed_Terrestrial', 'B1: Fixed Terrestrial', '#d62728', ':')
    ]

    for pol_key, label, col, ls in target_policies:
        sub = df[df['policy'] == pol_key]['delay_ms'].dropna()
        if len(sub) > 0:
            sorted_delays = np.sort(sub.values)
            cdf = np.arange(1, len(sorted_delays) + 1) / len(sorted_delays)
            mean_d = np.mean(sorted_delays)
            p95 = np.percentile(sorted_delays, 95)
            ax.plot(sorted_delays, cdf, color=col, linestyle=ls, lw=2.2,
                    label=f'{label} (mean={mean_d/1000:.1f}s, 95%={p95/1000:.1f}s)')

    ax.set_xlabel('End-to-End Access Delay [ms]')
    ax.set_ylabel('Empirical CDF')
    ax.set_title('Fig. 3: Access Delay Cumulative Distribution Function')
    ax.set_xlim(0, 35000)
    ax.set_ylim(0, 1.02)
    ax.grid(True)
    ax.legend(loc='lower right', framealpha=0.95)

    plt.tight_layout()
    save_path = os.path.join(out_dir, "fig3_delay_cdf.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved {save_path}")


def plot_fig4_failure_breakdown(results_dir: str = "results", out_dir: str = "figs"):
    """
    Fig. 4: Failure-cause breakdown across pass under matched stress Doppler (250 Hz).
    Compares Proposed (cause-separated) vs Lumped-Reward Ablation to demonstrate
    prevention of the false-barring loop.
    """
    csv_path = os.path.join(results_dir, "failures.csv")
    if not os.path.exists(csv_path):
        print(f"File {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    for ax, pol_name, title in [
        (ax1, "Proposed_Warm", "(a) Proposed Controller (Cause-Separated Reward, 250 Hz)"),
        (ax2, "Ablation_Lumped", "(b) Ablation Policy (Lumped Failure Penalty, 250 Hz)")
    ]:
        sub = df[df['policy'] == pol_name].groupby('step').agg({
            't_s': 'mean',
            'n_collision': 'sum',
            'n_miss': 'sum',
            'n_timeout': 'sum',
            'n_success': 'sum',
            'acb_p': 'mean',
            'n_rep': 'mean'
        }).reset_index()

        if not sub.empty:
            t_min = sub['t_s'] / 60.0
            ax.plot(t_min, sub['n_collision'], color='#d62728', label='Collisions', lw=1.8)
            ax.plot(t_min, sub['n_miss'], color='#ff7f0e', label='Detection Misses', lw=1.8)
            ax.plot(t_min, sub['n_timeout'], color='#9467bd', label='Timeouts', lw=1.8)
            ax.plot(t_min, sub['n_success'], color='#2ca02c', label='Successes', lw=1.8)

            ax_sec = ax.twinx()
            ax_sec.plot(t_min, sub['acb_p'], color='black', linestyle='--', alpha=0.7, label='ACB factor $p$')
            ax_sec.set_ylabel('ACB factor $p$', color='black')
            ax_sec.set_ylim(0.0, 1.05)

            ax.set_ylabel('Events per RAO')
            ax.set_title(title)
            ax.grid(True)
            if ax == ax1:
                ax.legend(loc='upper right', framealpha=0.9)

    ax2.set_xlabel('Pass Elapsed Time [minutes]')
    plt.tight_layout()
    save_path = os.path.join(out_dir, "fig4_failure_breakdown.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved {save_path}")


def plot_fig5_convergence(results_dir: str = "results", out_dir: str = "figs"):
    """
    Fig. 5: Convergence within active RAOs — Cold-Start vs Warm-Start (no 0/0 artifacts).
    """
    csv_path = os.path.join(results_dir, "convergence.csv")
    if not os.path.exists(csv_path):
        print(f"File {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7.5), sharex=True)

    for pol, col, lbl in [
        ('Proposed_Warm', '#1f77b4', 'Warm-Start Q-Learning (Proposed)'),
        ('Proposed_Cold', '#ff7f0e', 'Cold-Start Q-Learning (Zero Initialized)')
    ]:
        sub = df[df['policy'] == pol].groupby('step').agg(
            t_s=('t_s', 'mean'),
            r_mean=('reward', 'mean'),
            r_std=('reward', 'std'),
            s_mean=('success_rate_win', 'mean'),
            s_std=('success_rate_win', 'std')
        ).reset_index()

        if not sub.empty:
            t_min = sub['t_s'].values / 60.0
            r_mean = sub['r_mean'].values
            r_std = np.nan_to_num(sub['r_std'].values, nan=0.0)
            s_mean = sub['s_mean'].values
            s_std = np.nan_to_num(sub['s_std'].values, nan=0.0)

            ax1.plot(t_min, r_mean, color=col, label=lbl, lw=2.0)
            ax1.fill_between(t_min, r_mean - 0.5 * r_std, r_mean + 0.5 * r_std, color=col, alpha=0.18)

            ax2.plot(t_min, s_mean, color=col, label=lbl, lw=2.0)
            ax2.fill_between(t_min, s_mean - 0.5 * s_std, s_mean + 0.5 * s_std, color=col, alpha=0.18)

    ax1.set_ylabel('RAO Immediate Reward $r$')
    ax1.set_title('(a) Immediate Reward Trajectory across Active RAOs')
    ax1.grid(True)
    ax1.legend(loc='lower right', framealpha=0.9)

    ax2.set_ylabel('Rolling Success Rate')
    ax2.set_xlabel('Pass Time [minutes]')
    ax2.set_title('(b) Rolling Access Success Rate within Pass Visibility')
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True)
    ax2.legend(loc='lower right', framealpha=0.9)

    plt.tight_layout()
    save_path = os.path.join(out_dir, "fig5_convergence.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved {save_path}")


def plot_fig6_sensitivity_heatmap(results_dir: str = "results", out_dir: str = "figs"):
    """
    Fig. 6: Sensitivity Gain Analysis:
    (a) Access success gain vs Fixed Terrestrial Baseline (B1).
    (b) Transmission overhead reduction vs 3GPP Rel-17 NTN Default (B2).
    """
    csv_path = os.path.join(results_dir, "sensitivity.csv")
    if not os.path.exists(csv_path):
        print(f"File {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    cols = ['gain_vs_b1', 'energy_reduction_pct', 'p_s_proposed', 'p_s_b2', 'p_s_b1']
    available_cols = [c for c in cols if c in df.columns]
    grouped = df.groupby(['n_devices', 'residual_doppler_hz'])[available_cols].mean().reset_index()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    devices = sorted(grouped['n_devices'].unique())
    x = np.arange(len(devices))
    width = 0.35

    sub_10 = grouped[grouped['residual_doppler_hz'] == 10.0].sort_values('n_devices')
    sub_250 = grouped[grouped['residual_doppler_hz'] == 250.0].sort_values('n_devices')

    # Panel (a): Gain vs B1
    rects1 = ax1.bar(x - width/2, sub_10['gain_vs_b1'] * 100.0, width, label='Nominal CFO $\\delta f = 10$ Hz', color='#1f77b4')
    rects2 = ax1.bar(x + width/2, sub_250['gain_vs_b1'] * 100.0, width, label='Stress CFO $\\delta f = 250$ Hz', color='#ff7f0e')
    ax1.set_ylabel('Access Success Gain $\\Delta P_s$ vs Terrestrial (B1) [%]')
    ax1.set_xlabel('Offered Device Load ($N$ Meters)')
    ax1.set_title('(a) Gain over Fixed Terrestrial Default (B1)')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'N = {d}' for d in devices])
    ax1.grid(True, axis='y')
    ax1.legend(loc='upper left', framealpha=0.95)

    for rects in [rects1, rects2]:
        for rect in rects:
            height = rect.get_height()
            ax1.annotate(f'+{height:.1f}%' if height > 0 else f'{height:.1f}%',
                         xy=(rect.get_x() + rect.get_width() / 2, max(0, height)),
                         xytext=(0, 3), textcoords="offset points",
                         ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Panel (b): Energy / Transmission Reduction vs B2
    if 'energy_reduction_pct' in grouped.columns:
        rects3 = ax2.bar(x - width/2, sub_10['energy_reduction_pct'], width, label='Nominal CFO $\\delta f = 10$ Hz', color='#1f77b4')
        rects4 = ax2.bar(x + width/2, sub_250['energy_reduction_pct'], width, label='Stress CFO $\\delta f = 250$ Hz', color='#ff7f0e')
        ax2.set_ylabel('Transmission Overhead Reduction vs Rel-17 (B2) [%]')
        ax2.set_xlabel('Offered Device Load ($N$ Meters)')
        ax2.set_title('(b) Transmission Saving vs Rel-17 NTN Default (B2)')
        ax2.set_xticks(x)
        ax2.set_xticklabels([f'N = {d}' for d in devices])
        ax2.grid(True, axis='y')
        ax2.legend(loc='upper left', framealpha=0.95)

        for rects in [rects3, rects4]:
            for rect in rects:
                height = rect.get_height()
                ax2.annotate(f'+{height:.1f}%' if height > 0 else f'{height:.1f}%',
                             xy=(rect.get_x() + rect.get_width() / 2, max(0, height)),
                             xytext=(0, 3), textcoords="offset points",
                             ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    save_path = os.path.join(out_dir, "fig6_sensitivity_gain.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved {save_path}")


def generate_all_plots(cfg: dict, results_dir: str = "results", out_dir: str = "figs"):
    """Generate all six research figures."""
    print("Generating Figure 1: Orbit geometry...")
    plot_fig1_orbit_geometry(cfg, out_dir=out_dir)

    print("Generating Figure 2: Elevation vs success U-shape...")
    plot_fig2_elevation_success(results_dir=results_dir, out_dir=out_dir)

    print("Generating Figure 3: Delay CDF...")
    plot_fig3_delay_cdf(results_dir=results_dir, out_dir=out_dir)

    print("Generating Figure 4: Failure cause breakdown (250 Hz stress)...")
    plot_fig4_failure_breakdown(results_dir=results_dir, out_dir=out_dir)

    print("Generating Figure 5: Cold vs warm convergence...")
    plot_fig5_convergence(results_dir=results_dir, out_dir=out_dir)

    print("Generating Figure 6: Sensitivity gain vs Rel-17 Default (B2)...")
    plot_fig6_sensitivity_heatmap(results_dir=results_dir, out_dir=out_dir)
    print("All six figures generated successfully in", out_dir)


if __name__ == "__main__":
    cfg = load_config()
    generate_all_plots(cfg)
