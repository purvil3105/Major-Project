# Major Project: Cause-Aware Q-Learning RACH Control for NB-IoT NTN

7th-semester major project. A tabular Q-learning controller jointly sets NB-IoT access barring, the Backoff Indicator and the NPRACH repetition profile for smart meters served by a LEO satellite. It is evaluated on real Sateliot SGP4 passes against static, heuristic and genie baselines.

## Contents

| Path | What it is |
|---|---|
| [`01_Literature_Review.md`](01_Literature_Review.md) | Literature review (19 papers + 3GPP specs), gaps G1–G5, corrected after review |
| [`02_Methodology.md`](02_Methodology.md) | System model, MDP, baselines and metrics, exactly as implemented |
| [`03_Implementation_Plan.md`](03_Implementation_Plan.md) | Build phases, acceptance checks (all automated), commands |
| [`REVIEW_ISSUES.md`](REVIEW_ISSUES.md) | Review of the original project: every issue found, with its resolution status |
| [`RESOLUTION_PLAN.md`](RESOLUTION_PLAN.md) | The plan used to fix them |
| [`ASSUMPTIONS.md`](ASSUMPTIONS.md) | Every modelling and process assumption, and the pre-declared v2 iteration |
| [`ProblemStatement_Errata.md`](ProblemStatement_Errata.md) | Corrections to the original problem statement |
| [`paper/`](paper/) | IEEE conference paper draft (LaTeX, figures, references) |
| [`nbiot_ntn_qlearning/`](nbiot_ntn_qlearning/) | Simulation code, results, figures (see its README) |
| `_extracted/`, `papers/` | Reference papers (local only, not published) |
| `Project_Audit.pdf` | Earlier audit (11 Oct 2026) |

## Headline results (v2; N = 1 000 meters; 14 passes × 10 seeds)

- **Against the static NTN-aware configuration (B2):** about equal deadline success (0.668 vs 0.663), with 9 % less preamble airtime and 3.9 pp fewer dropped meters.
- **Pareto check:** none of the 60 fixed configurations is better on both deadline success and airtime.
- **At nominal load:** 6–8 pp below the heuristic and the genie best-fixed configuration, which use ~60 % more airtime.
- **At 3–5× the nominal load:** 4.6–8.1 pp above the genie best-fixed configuration tuned for nominal load.
- **Cause information** adds 1.4–1.6 pp.
- **Negative result:** satellite geometry in the state did not help. A terrestrial-trained load + cause table was 1.3 pp better.

Details, confidence intervals and tests are in [`nbiot_ntn_qlearning/README.md`](nbiot_ntn_qlearning/README.md).

## Quick start

```bash
cd nbiot_ntn_qlearning
python -m pip install -r requirements.txt
python -m pytest -q
python main.py --plots-only
```
