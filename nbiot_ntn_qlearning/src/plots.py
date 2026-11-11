"""
plots.py — IEEE-format figures (3.5 in single column / 7.16 in double column, PDF + PNG).

Fig. 1  Pass geometry: elevation, one-way delay, raw Doppler, outdoor SNR (all test passes).
Fig. 2  Per-attempt success rate vs elevation, all schemes (mean, 95 % CI over runs).
Fig. 3  Access-delay CDF.
Fig. 4  Ground-truth failure causes and chosen actions over the pass: Proposed vs ablations.
Fig. 5  Convergence within a pass: warm vs cold start.
Fig. 6  Sensitivity heatmaps: deadline-success gain of Proposed over B2 and over B5.
Each scheme keeps one colour, marker and line style in every figure. No embedded titles:
captions belong in the LaTeX source.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.orbit import get_passes  # noqa: E402
from src.stats import LABELS, ci95  # noqa: E402

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 6.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "lines.linewidth": 1.2, "lines.markersize": 4, "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.4,
    "axes.linewidth": 0.6, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "legend.frameon": False,
})
COL1, COL2 = 3.5, 7.16
INK = "#52514e"
STYLE = {  # colour, marker, linestyle — fixed per scheme; hues from the validated palette order
    "Proposed": ("#2a78d6", "o", "-"),
    "B1_FixedTerrestrial": ("#eb6834", "x", ":"),
    "B2_StaticNTN": ("#1baf7a", "^", "--"),
    "B3_TerrestrialFrozen": ("#eda100", "P", "-."),
    "B4_Heuristic": ("#e87ba4", "<", "--"),
    "B5_BestFixed": ("#008300", "*", "-"),
    "Proposed_Cold": ("#e34948", "s", "--"),
    "Abl_LumpedReward": ("#6250d6", "D", "-."),
    "Abl_NoCause": (INK, "v", ":"),
}
CAUSE_COL = {"n_collision": "#e34948", "n_miss": "#eda100", "n_timeout": "#2a78d6"}


def _save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"))
    plt.close(fig)
    print(f"  saved {name}")


def fig1_geometry(cfg, out_dir):
    passes = get_passes(cfg, "test")
    ref = max(passes, key=lambda p: p["max_elevation_deg"])
    from src.channel import outdoor_snr_db
    fig, ax = plt.subplots(2, 2, figsize=(COL2, 3.6), sharex=True)
    for p in passes:
        tm = p["time_s"] / 60
        snr = outdoor_snr_db(p["slant_range_km"], p["elevation_deg"], p["altitude_km"], cfg)
        lw, c, z = (1.4, STYLE["Proposed"][0], 3) if p is ref else (0.6, "#b9b8b2", 1)
        ax[0, 0].plot(tm, p["elevation_deg"], color=c, lw=lw, zorder=z)
        ax[0, 1].plot(tm, p["delay_ms"], color=c, lw=lw, zorder=z)
        ax[1, 0].plot(tm, p["raw_doppler_hz"] / 1e3, color=c, lw=lw, zorder=z)
        ax[1, 1].plot(tm, snr, color=c, lw=lw, zorder=z)
    ax[0, 0].set_ylabel("Elevation [deg]")
    ax[0, 1].set_ylabel("One-way delay [ms]")
    ax[1, 0].set_ylabel("Raw Doppler at 2 GHz [kHz]")
    ax[1, 1].set_ylabel("Outdoor UL SNR (3.75 kHz) [dB]")
    for a in ax[1]:
        a.set_xlabel("Time since rise [min]")
    for a, lab in zip(ax.flat, "abcd"):
        a.text(0.0, 1.02, f"({lab})", transform=a.transAxes, va="bottom", fontsize=7.5)
    ax[0, 0].text(0.98, 0.95, f"bold: peak {ref['max_elevation_deg']:.0f}$^\\circ$ pass\ngrey: other {len(passes)-1} test passes",
                  transform=ax[0, 0].transAxes, ha="right", va="top", fontsize=6.5, color=INK)
    fig.tight_layout()
    _save(fig, out_dir, "fig1_geometry")


def fig2_elevation(res, out_dir):
    df = pd.read_csv(os.path.join(res, "elevation_bins.csv"))
    df = df[df["experiment"] == "main"]
    df["rate"] = df["successes"] / df["attempts"].replace(0, np.nan)
    pols = ["Proposed", "B1_FixedTerrestrial", "B2_StaticNTN", "B3_TerrestrialFrozen", "B4_Heuristic", "B5_BestFixed"]
    fig, ax = plt.subplots(figsize=(COL1, 2.4))
    for pol in pols:
        sub = df[df["policy"] == pol]
        xs, ys, hs = [], [], []
        for (lo, hi), g in sub.groupby(["elev_lo", "elev_hi"]):
            m, h = ci95(g["rate"])
            xs.append((lo + hi) / 2)
            ys.append(m)
            hs.append(h)
        c, mk, ls = STYLE[pol]
        ax.errorbar(xs, ys, yerr=hs, color=c, marker=mk, ls=ls, capsize=2, label=LABELS[pol])
    ax.set_xlabel("Satellite elevation [deg]")
    ax.set_ylabel("Per-attempt success rate")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="lower center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    _save(fig, out_dir, "fig2_success_vs_elevation")


def fig3_delay_cdf(res, out_dir, deadline):
    df = pd.read_csv(os.path.join(res, "delays.csv.gz"))
    pols = ["Proposed", "B1_FixedTerrestrial", "B2_StaticNTN", "B3_TerrestrialFrozen", "B4_Heuristic", "B5_BestFixed"]
    fig, ax = plt.subplots(figsize=(COL1, 2.4))
    for pol in pols:
        d = np.sort(df.loc[df["policy"] == pol, "delay_s"].to_numpy())
        if d.size == 0:
            continue
        c, _, ls = STYLE[pol]
        ax.plot(d, np.arange(1, d.size + 1) / d.size, color=c, ls=ls, label=LABELS[pol])
    ax.axvline(deadline, color=INK, lw=0.6, ls=":")
    ax.text(deadline * 1.05, 0.03, f"deadline {deadline:.0f} s", fontsize=6.5, color=INK)
    ax.set_xscale("log")
    ax.set_xlabel("Access delay of served meters [s]")
    ax.set_ylabel("Empirical CDF")
    ax.set_ylim(0, 1.01)
    ax.legend(loc="lower center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    _save(fig, out_dir, "fig3_delay_cdf")


def _binned(log, col, nb=30):
    log = log.copy()
    dur = log.groupby(["seed", "pass_idx"])["t_s"].transform("max")
    log["phase_bin"] = np.minimum((log["t_s"] / dur * nb).astype(int), nb - 1)
    per_run = log.groupby(["seed", "pass_idx", "phase_bin"])[col].mean().reset_index()
    out = per_run.groupby("phase_bin")[col].agg(["mean", "std", "count"]).reset_index()
    out["x"] = (out["phase_bin"] + 0.5) / nb
    out["h"] = 1.96 * out["std"] / np.sqrt(out["count"].clip(lower=1))
    return out


def fig4_failures(res, out_dir):
    log = pd.read_csv(os.path.join(res, "rao_log.csv.gz"))
    pols = ["Proposed", "Abl_LumpedReward", "Abl_NoCause"]
    fig, ax = plt.subplots(3, 3, figsize=(COL2, 4.2), sharex=True, sharey="row")
    for j, pol in enumerate(pols):
        sub = log[log["policy"] == pol]
        bottom = None
        for col, lab in [("n_collision", "Collision"), ("n_miss", "Miss"), ("n_timeout", "Timeout")]:
            b = _binned(sub, col)
            y = b["mean"].to_numpy()
            base = np.zeros_like(y) if bottom is None else bottom
            ax[0, j].fill_between(b["x"], base, base + y, color=CAUSE_COL[col], alpha=0.85, lw=0,
                                  label=lab, step="mid")
            bottom = base + y
        ax[0, j].set_title(LABELS[pol], fontsize=7.5)
        for row, col, ylab in [(1, "acb_p", "Barring factor $p$"), (2, "rep_ce2", "CE2 repetitions")]:
            b = _binned(sub, col)
            c = STYLE[pol][0]
            ax[row, j].plot(b["x"], b["mean"], color=c)
            ax[row, j].fill_between(b["x"], b["mean"] - b["h"], b["mean"] + b["h"], color=c, alpha=0.2, lw=0)
            if j == 0:
                ax[row, j].set_ylabel(ylab)
        ax[2, j].set_xlabel("Pass progress (rise = 0, set = 1)")
    ax[0, 0].set_ylabel("Failed attempts per RAO")
    h, lab = ax[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.035))
    fig.tight_layout()
    _save(fig, out_dir, "fig4_failure_breakdown")


def fig5_convergence(res, out_dir):
    log = pd.read_csv(os.path.join(res, "rao_log.csv.gz"))
    log = log[log["n_trans"] > 0].copy()
    log["succ_rate"] = log["n_success"] / log["n_trans"]
    fig, ax = plt.subplots(2, 1, figsize=(COL1, 3.2), sharex=True)
    for pol in ["Proposed", "Proposed_Cold", "B5_BestFixed"]:
        sub = log[log["policy"] == pol]
        c, _, ls = STYLE[pol]
        for row, col in [(0, "reward"), (1, "succ_rate")]:
            if col == "reward" and pol == "B5_BestFixed":
                continue
            b = _binned(sub, col, nb=25)
            ax[row].plot(b["x"], b["mean"], color=c, ls=ls, label=LABELS[pol])
            ax[row].fill_between(b["x"], b["mean"] - b["h"], b["mean"] + b["h"], color=c, alpha=0.18, lw=0)
    ax[0].set_ylabel("Per-RAO reward")
    ax[1].set_ylabel("Per-attempt success rate")
    ax[1].set_xlabel("Pass progress (rise = 0, set = 1)")
    ax[1].legend(loc="lower center", ncol=3)
    fig.tight_layout()
    _save(fig, out_dir, "fig5_convergence")


def fig6_sensitivity(res, out_dir):
    df = pd.read_csv(os.path.join(res, "sensitivity.csv"))
    df = df[df["experiment"] == "load_x_coupling"]
    key = ["n_devices", "coupling_loss_mean_db", "pass_idx", "seed"]
    piv = df.pivot_table(index=key, columns="policy", values="p_s_deadline").reset_index()
    fig, ax = plt.subplots(1, 2, figsize=(COL2, 2.2))
    vmax = 0.0
    grids = []
    for ref in ["B2_StaticNTN", "B5_BestFixed"]:
        piv["gain"] = piv["Proposed"] - piv[ref]
        g = piv.groupby(["coupling_loss_mean_db", "n_devices"])["gain"].mean().unstack()
        grids.append((ref, g))
        vmax = max(vmax, float(np.nanmax(np.abs(g.to_numpy()))))
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("div", ["#e34948", "#f2f1ee", "#2a78d6"])
    for a, (ref, g) in zip(ax, grids):
        im = a.imshow(g.to_numpy(), cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto", origin="lower")
        a.set_xticks(range(g.shape[1]), [str(c) for c in g.columns])
        a.set_yticks(range(g.shape[0]), [f"{r:g}" for r in g.index])
        a.set_xlabel("Meters per pass $N$")
        a.set_ylabel("Mean coupling loss [dB]")
        a.grid(False)
        for i in range(g.shape[0]):
            for j in range(g.shape[1]):
                a.text(j, i, f"{g.iat[i, j]*100:+.1f}", ha="center", va="center", fontsize=6.5, color="#0b0b0b")
        a.text(0.0, 1.03, f"Proposed $-$ {LABELS[ref]} (pp)", transform=a.transAxes, fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="$\\Delta P_s^{\\leq D}$")
    _save(fig, out_dir, "fig6_sensitivity")


def generate_all_plots(cfg, results_dir="results", out_dir="figs"):
    print("Generating figures ...")
    fig1_geometry(cfg, out_dir)
    for fn in (fig2_elevation, fig4_failures, fig5_convergence, fig6_sensitivity):
        try:
            fn(results_dir, out_dir)
        except FileNotFoundError as exc:
            print(f"  skipped {fn.__name__}: {exc}")
    try:
        fig3_delay_cdf(results_dir, out_dir, cfg["deadline_s"])
    except FileNotFoundError as exc:
        print(f"  skipped fig3: {exc}")
