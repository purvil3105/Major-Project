# Implementation Plan (as built)
### NB-IoT NTN Q-learning RACH controller

This plan describes how the code is built and how each phase is checked. Every acceptance check is now an automated test in `nbiot_ntn_qlearning/tests/test_simulation.py` (17 tests). The earlier plan is in git history. Design details are in `02_Methodology.md`.

## 1. Software stack (pinned in `requirements.txt`)
Python 3.12.10, with skyfield 1.55, sgp4 2.27, numpy 2.3.1, scipy 1.16.2, pandas 2.3.1, matplotlib 3.10.6, PyYAML 6.0.3, pytest 8.2.2 and tqdm 4.67.1. The project uses no deep-learning framework.

## 2. Repository layout
```
nbiot_ntn_qlearning/
├── config/default.yaml      single source of truth, every value tagged [3GPP]/[paper]/[assump]/[design]/[sweep]
├── data/tle/sateliot.tle    real TLEs (4 Sateliot satellites, epoch Oct 2026)
├── src/orbit.py             SGP4 passes (all qualifying passes, vectorised); synthetic altitude orbits
├── src/channel.py           Set-4 link budget; P_d fit (TS 36.104 / Chougrani); timing-error model
├── src/nprach.py            vectorised NB-IoT RA simulator: CE levels, pools, barring, backoff, outcomes, observables
├── src/mdp.py               observable ML estimator; state; action; cause-separated / lumped reward
├── src/agent.py             tabular Q-learning
├── src/baselines.py         B1, B2, B4 rules (B3 and B5 live in experiment.py)
├── src/experiment.py        pre-training, B5 search, main evaluation, sweeps (multiprocessing, paired seeds)
├── src/stats.py             95 % CIs, Wilcoxon + Holm, LaTeX table
├── src/plots.py             six IEEE-format figures (PDF + PNG)
├── tests/test_simulation.py acceptance tests
├── results/                 CSVs, Q-tables, manifest.json (config hash, seeds, versions, TLE passes)
├── results_archive_2026-10/ previous results and figures (kept for comparison)
└── figs/                    figures
```

## 3. Phases and acceptance checks
| Phase | Deliverable | Acceptance check (test) |
|---|---|---|
| 0 Setup | pinned environment, git, config loader | byte-identical reruns (`test_byte_identical_reruns`); identical traffic across policies (`test_common_random_numbers_across_policies`) |
| 1 Orbit | 14 test passes + 40 disjoint training passes | duration 300–540 s, delay 1.8–6.6 ms, Doppler ≤ 48 kHz (`test_pass_geometry_bounds`); training and test satellites disjoint |
| 2 Channel | link budget and P_d | reproduces [10] Set-4 at 30°/90°; 99 % detection at the TS 36.104 points; GNSS-free fit within ±1.5 dB of [5]; P_d rises with N; timeout probability is 0 at zenith and grows toward the horizon |
| 3 Access model | simulator | outcomes mutually exclusive and observables consistent; no timeouts when f_deg = 0; collisions dominate at N = 5 000 without barring; more repetitions give fewer misses |
| 4 MDP / agent | estimator, state, reward, Q-learning | estimator load bias < 15 % and miss error < 1; the reward is unchanged when ground-truth fields are altered (`test_reward_uses_only_observables`); 108 states × 60 actions; Q-update correct |
| 5 Baselines | B1–B5, ablations | B5 is the best fixed action by construction (exhaustive search) |
| 6 Experiments | `python main.py` | 9 policies × 14 passes × 10 seeds; sweeps; manifest written |
| 7 Figures / tables | `figs/`, `results/summary_table.csv`, `stats_tests.csv`, `table_main.tex` | rendered and inspected for overlaps; no figure shows numbers that aren't in the CSVs |
| 8 Results gate | review with the student before any claim is written | done: v1 reviewed, one pre-declared iteration (v2, ASSUMPTIONS §D), framing agreed |

## 4. Experiment matrix
| Run | Policies | Passes × seeds | Varied |
|---|---|---|---|
| B5 search | 60 fixed actions | 14 × 2 | — |
| Main | all 9 | 14 × 10 | — |
| Load × coupling | Proposed, B2, B4, B5 | 4 × 4 | N ∈ {200, 1 000, 3 000, 5 000} × μ_L ∈ {0, 5, 10} dB |
| Detection regime | Proposed, B2, B4, B5 | 4 × 4 | GNSS-free |
| GNSS-degraded share | Proposed, No-cause ablation, B2, B5 | 4 × 4 | f_deg ∈ {0, 0.1, 0.3} |
| RAO period | Proposed, B2, B4 | 4 × 4 | 160 / 320 / 640 ms |
| Altitude | Proposed, B2, B4 | 4 × 4 (synthetic passes) | 500 / 600 / 800 km |

Runtime on 4 CPU cores (about 1 s per pass simulation): about 38 min for v1 and about 50 min for v2, whose pre-training is 4× longer. v1 is reproduced with `python main.py --config config/v1.yaml --results-dir results_v1 --figs-dir figs_v1`. `python compare_versions.py` builds the side-by-side table.

## 5. Commands
```
python -m pytest -q          # acceptance tests (~10 s)
python main.py --quick       # reduced smoke run (~9 min) -> results_quick/, figs_quick/
python main.py               # full reproduction -> results/, figs/
python main.py --plots-only  # re-draw figures and tables from results/
```

## 6. Known limitations (to be stated in the paper)
- One ground site and a single satellite visible at a time. There is no inter-satellite handover.
- No capture effect. All meters on a collided subcarrier fail.
- The detection curve is logistic, anchored to two 3GPP AWGN points. Its slope k is assumed.
- The GNSS-degraded timing model is stylised (uniform error up to the beam's differential RTD).
- The estimator ignores collisions where all colliding preambles go undetected, and it is biased at very low load.
- B5 is a genie (tuned on the test passes). It is a reference, not a deployable scheme.
