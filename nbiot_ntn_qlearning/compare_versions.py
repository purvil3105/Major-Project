"""
compare_versions.py — side-by-side v1 (pre-declared design) vs v2 (one documented iteration).

    python compare_versions.py [--v1 results_v1] [--v2 results]

Writes results/version_comparison.csv and results/table_versions.tex.
"""

import argparse
import os

import pandas as pd

from src.stats import LABELS, ORDER, paired_tests, summary_table

METRICS = [("p_s_deadline", "$P_s^{\\le D}$", 3), ("p_s", "$P_s$", 3), ("p_drop", "Drop", 3),
           ("mean_delay_s", "$\\bar d$ [s]", 1), ("airtime_per_success_ms", "Airtime [ms]", 0)]
LEARNERS = ["Proposed", "Proposed_Cold", "Abl_LumpedReward", "Abl_NoCause", "B3_TerrestrialFrozen"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1", default="results_v1")
    ap.add_argument("--v2", default="results")
    a = ap.parse_args()
    rows = []
    for ver, d in (("v1", a.v1), ("v2", a.v2)):
        df = pd.read_csv(os.path.join(d, "metrics.csv"))
        s = summary_table(df)
        s.insert(0, "version", ver)
        t = paired_tests(df)
        t = t[t["metric"] == "p_s_deadline"].set_index("vs")[["mean_diff", "p_holm"]]
        s = s.merge(t.rename(columns={"mean_diff": "d_vs_proposed", "p_holm": "p_vs_proposed"}),
                    left_on="policy", right_index=True, how="left")
        rows.append(s)
    out = pd.concat(rows)
    out.to_csv(os.path.join(a.v2, "version_comparison.csv"), index=False)

    lines = ["\\begin{tabular}{l" + "cc" * len(METRICS) + "}", "\\toprule",
             "Scheme & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{m[1]}}}" for m in METRICS) + " \\\\",
             " & " + " & ".join(["v1 & v2"] * len(METRICS)) + " \\\\", "\\midrule"]
    for pol in [p for p in ORDER if p in set(out["policy"])]:
        cells = []
        for m, _, dec in METRICS:
            for ver in ("v1", "v2"):
                r = out[(out["version"] == ver) & (out["policy"] == pol)]
                cells.append(f"{r[m].iloc[0]:.{dec}f}" if len(r) else "--")
        lines.append(f"{LABELS[pol]} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    with open(os.path.join(a.v2, "table_versions.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    show = out[["version", "policy", "p_s_deadline", "p_s", "p_drop", "mean_delay_s", "p95_delay_s",
                "airtime_per_success_ms", "d_vs_proposed", "p_vs_proposed"]]
    print(show.round(3).to_string(index=False))
    print("\nNote: baselines B1/B2/B4/B5 do not learn, so they are identical in v1 and v2;"
          " only " + ", ".join(LEARNERS) + " can change.")


if __name__ == "__main__":
    main()
