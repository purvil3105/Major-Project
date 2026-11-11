# Cause-Aware Tabular Q-Learning for NB-IoT NTN Random Access

Simulation code for a lightweight tabular Q-learning controller that jointly sets NB-IoT access barring, the Backoff Indicator, and the NPRACH repetition profile during a LEO satellite pass. The controller acts only on quantities a gNB can observe:
- NPRACH idle / detected / collided subcarrier counts;
- Msg3 failures;
- ephemeris-derived elevation and pass phase.

## What is modelled

| Part | Model | Main sources |
|---|---|---|
| Orbit | SGP4 (Skyfield) on real Sateliot TLEs (~577 km). 14 test passes (peak 20–86°) and 40 disjoint training passes over Ahmedabad, 10° mask | CelesTrak TLE epoch Oct 2026 |
| Link budget | 3GPP TR 38.821 Set-4 payload, reproducing Dwivedi et al. (−3.14 dB at 30°, +4.93 dB at 90°, 3.75 kHz) | TR 38.821; Dwivedi et al. 2024 |
| Detection | Logistic P_d(SNR, N_rep) through the TS 36.104 NPRACH 99 % points. A GNSS-free regime is fitted to Chougrani et al. 2022 | TS 36.104 Table 8.5.3.2.1-1 |
| Access | 3 CE levels (24/12/12 subcarriers), RSRP-based CE selection, per-CE periodicity, AB-bitmap barring at initiation, backoff, CE escalation, preambleTransMax = 10 | TS 36.211 / 36.321 / 36.331 |
| Failure causes | collision / detection miss / timeout. Timeouts come from GNSS-degraded meters whose timing error exceeds the CP | Chougrani et al.; Kodheli et al. |
| Traffic | 70 % periodic reports spread over the pass + a 30 % Beta(3,4) alarm burst at a random time | TR 37.868 |
| Controller | 108 states (load × elevation × pass phase × dominant failure cause) × 60 actions. Q-table size 52 KB | — |

All parameters and their sources are in [`config/default.yaml`](config/default.yaml). The full design is in [`../02_Methodology.md`](../02_Methodology.md).

## Results (v2, N = 1 000 meters, 14 passes × 10 seeds, mean ± 95 % CI)

| Scheme | Served ≤ 10 s | Served in pass | Drop rate | Mean delay [s] | Airtime / served meter [ms] |
|---|---|---|---|---|---|
| **Proposed (warm start)** | 0.668 ± 0.019 | 0.756 | 0.223 | 4.06 ± 0.20 | 2034 ± 164 |
| Proposed (cold start) | 0.607 ± 0.022 | 0.794 | 0.164 | 7.37 ± 0.35 | 1803 ± 143 |
| Ablation: lumped reward | 0.652 ± 0.021 | 0.737 | 0.231 | 3.97 ± 0.24 | 2040 ± 149 |
| Ablation: no cause information | 0.654 ± 0.019 | 0.745 | 0.229 | 4.20 ± 0.20 | 2254 ± 174 |
| B1 Fixed terrestrial-style | 0.619 ± 0.016 | 0.619 | 0.377 | 0.85 ± 0.03 | 1687 ± 93 |
| B2 Static NTN-aware | 0.663 ± 0.018 | 0.722 | 0.262 | 3.49 ± 0.07 | 2246 ± 147 |
| B3 Terrestrial-trained Q (frozen) | 0.681 ± 0.021 | 0.755 | 0.217 | 3.84 ± 0.25 | 1912 ± 140 |
| B4 Channel-aware heuristic | 0.730 ± 0.018 | 0.773 | 0.204 | 2.71 ± 0.13 | 3259 ± 268 |
| B5 Best fixed configuration (genie) | 0.745 ± 0.017 | 0.775 | 0.208 | 2.14 ± 0.08 | 3367 ± 236 |

All comparisons use a paired Wilcoxon signed-rank test with Holm correction. Full numbers are in `results/summary_table.csv` and `results/stats_tests.csv`.

**What the results show:**
- **Against B2:** deadline success is about equal (+0.5 pp, p < 0.001), with 9 % less preamble airtime and 3.9 pp fewer drops.
- **Against B4/B5 at nominal load:** 6–8 pp lower deadline success, with ~38–40 % less airtime.
- **Pareto check:** none of the 60 fixed configurations is better on both deadline success and airtime. Two fixed configurations dominate B2.
- **Load robustness:**
  - At 3 000–5 000 meters (load × coupling-loss sweep, Holm p < 0.05), the controller beats the nominal-tuned best fixed configuration by 4.6–8.1 pp and the heuristic by 4.4–6.5 pp.
  - At 200–1 000 meters it is 8–10 pp below them.
- **Cause information:** +1.4–1.6 pp over both ablations (p < 0.001).
- **Negative result:** a table trained on a terrestrial channel with load + cause state only (B3) is 1.3 pp better than the satellite-aware table. Elevation and pass phase in the state gave no measurable benefit in this setting.
- **Warm start:** pre-training on other satellites' passes beats in-pass cold-start learning by 6.1 pp.

**Versions.**
- v1 is the pre-declared design. It had no delay term and only 30 pre-training passes; there, Proposed reached 0.492 deadline success.
- v2 is one documented iteration, declared in [`../ASSUMPTIONS.md`](../ASSUMPTIONS.md) §D before it was run.
- Both are reproducible. `python compare_versions.py` prints them side by side.

## Reproduce

```bash
python -m pip install -r requirements.txt
python -m pytest -q                       # 18 acceptance tests, ~10 s
python main.py                            # full v2 run (~50 min on 4 cores) -> results/, figs/
python main.py --config config/v1.yaml --results-dir results_v1 --figs-dir figs_v1   # v1
python compare_versions.py                # v1 vs v2 table
python main.py --plots-only               # redraw figures/tables from results/
```

Each run writes `results/manifest.json`: config hash, seeds, test-pass list, package versions and runtime. Runs are byte-identical for a given config.

## Layout

```
config/            default.yaml (v2), v1.yaml
data/tle/          Sateliot TLEs
src/               orbit, channel, nprach, mdp, agent, baselines, experiment, stats, plots
tests/             acceptance tests
results/, figs/    v2 outputs          results_v1/, figs_v1/   v1 outputs
results_archive_2026-10/   outputs of the pre-review code (superseded, kept for the record)
```

## Limitations

- One ground site and one satellite in view at a time, with no handover.
- No capture effect.
- The detection curve slope is assumed (3GPP specifies only the 99 % points).
- The timing-error model for GNSS-degraded meters is stylised.
- B5 is tuned on the test passes, so it is a reference, not a deployable scheme.

## License

MIT, see [LICENSE](LICENSE). Citation metadata is in [CITATION.cff](CITATION.cff).
