"""
stats.py — summary tables with 95 % confidence intervals and paired significance tests.

Unit of analysis: one (test pass, traffic seed) run. All policies share the same traffic for a
given (pass, seed), so comparisons are paired. CI = mean +/- t_{0.975, n-1} s / sqrt(n).
Tests: two-sided Wilcoxon signed-rank, Proposed vs every other policy, Holm-corrected per metric.
"""

import os

import numpy as np
import pandas as pd
from scipy import stats

METRICS = [
    ("p_s_deadline", "P_s (<= deadline)", 1.0),
    ("p_s", "P_s (within pass)", 1.0),
    ("p_drop", "Drop rate", 1.0),
    ("mean_delay_s", "Mean delay [s]", 1.0),
    ("p95_delay_s", "95th-pct delay [s]", 1.0),
    ("airtime_per_success_ms", "Preamble airtime / success [ms]", 1.0),
    ("collision_rate", "Collision share", 1.0),
    ("miss_rate", "Miss share", 1.0),
    ("timeout_rate", "Timeout share", 1.0),
]
ORDER = ["Proposed", "Proposed_Cold", "Abl_LumpedReward", "Abl_NoCause", "B1_FixedTerrestrial",
         "B2_StaticNTN", "B3_TerrestrialFrozen", "B4_Heuristic", "B5_BestFixed"]
LABELS = {
    "Proposed": "Proposed (warm)", "Proposed_Cold": "Proposed (cold)",
    "Abl_LumpedReward": "Ablation: lumped reward", "Abl_NoCause": "Ablation: no cause info",
    "B1_FixedTerrestrial": "B1 Fixed terrestrial", "B2_StaticNTN": "B2 Static NTN-aware",
    "B3_TerrestrialFrozen": "B3 Terrestrial-frozen Q", "B4_Heuristic": "B4 Heuristic",
    "B5_BestFixed": "B5 Best fixed (genie)",
}


def ci95(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if x.size < 2:
        return float(np.mean(x)) if x.size else np.nan, np.nan
    h = stats.t.ppf(0.975, x.size - 1) * x.std(ddof=1) / np.sqrt(x.size)
    return float(x.mean()), float(h)


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj


def summary_table(df):
    rows = []
    for pol in [p for p in ORDER if p in set(df["policy"])]:
        sub = df[df["policy"] == pol]
        row = {"policy": pol, "n_runs": len(sub)}
        for m, _, _ in METRICS:
            mu, h = ci95(sub[m])
            row[m] = mu
            row[m + "_ci"] = h
        rows.append(row)
    return pd.DataFrame(rows)


def paired_tests(df, reference="Proposed"):
    rows = []
    key = ["pass_idx", "seed"]
    ref = df[df["policy"] == reference].set_index(key)
    for m, _, _ in METRICS:
        tmp = []
        for pol in [p for p in ORDER if p in set(df["policy"]) and p != reference]:
            other = df[df["policy"] == pol].set_index(key)
            j = ref[[m]].join(other[[m]], lsuffix="_ref", rsuffix="_oth").dropna()
            d = j[m + "_ref"] - j[m + "_oth"]
            if len(d) < 5 or np.allclose(d, 0):
                p = 1.0
            else:
                p = float(stats.wilcoxon(j[m + "_ref"], j[m + "_oth"]).pvalue)
            tmp.append({"metric": m, "vs": pol, "n_pairs": len(d), "mean_diff": float(d.mean()),
                        "median_diff": float(d.median()), "p_value": p})
        adj = holm([t["p_value"] for t in tmp])
        for t, a in zip(tmp, adj):
            t["p_holm"] = float(a)
            rows.append(t)
    return pd.DataFrame(rows)


def latex_table(summ):
    cols = [("p_s_deadline", "$P_s^{\\le D}$", 3), ("p_drop", "Drop", 3), ("mean_delay_s", "$\\bar d$ [s]", 2),
            ("p95_delay_s", "$d_{95}$ [s]", 1), ("airtime_per_success_ms", "Airtime [ms]", 0)]
    lines = ["\\begin{tabular}{l" + "c" * len(cols) + "}", "\\toprule",
             "Scheme & " + " & ".join(c[1] for c in cols) + " \\\\", "\\midrule"]
    for _, r in summ.iterrows():
        cells = []
        for m, _, dec in cols:
            cells.append(f"{r[m]:.{dec}f}$\\pm${r[m + '_ci']:.{dec}f}")
        lines.append(f"{LABELS.get(r['policy'], r['policy'])} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def write_tables(results_dir="results"):
    path = os.path.join(results_dir, "metrics.csv")
    if not os.path.exists(path):
        print(f"{path} missing; skipping tables")
        return
    df = pd.read_csv(path)
    summ = summary_table(df)
    summ.to_csv(os.path.join(results_dir, "summary_table.csv"), index=False)
    tests = paired_tests(df)
    tests.to_csv(os.path.join(results_dir, "stats_tests.csv"), index=False)
    with open(os.path.join(results_dir, "table_main.tex"), "w", encoding="utf-8") as f:
        f.write(latex_table(summ))
    print(summ[["policy", "p_s_deadline", "p_s_deadline_ci", "p_drop", "mean_delay_s",
                "p95_delay_s", "airtime_per_success_ms"]].round(3).to_string(index=False))
