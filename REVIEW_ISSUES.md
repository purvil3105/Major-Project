# Project Review — Issues Found

## Resolution status (updated 11 Nov 2026)

Every issue below has been addressed. Git commits `5805bb6` (baseline) → `f7551ab` and later record each change.

| Issues | Resolution |
|---|---|
| P-1…P-6 | `ProblemStatement_Errata.md`. The original `.docx` is left unchanged. |
| L-1…L-11 | Applied in `01_Literature_Review.md`. [9] verified via Crossref. R1-2100601 and TS 36.104 added. Absolute novelty claims softened, with overlaps stated. The NB-IoT AB-bitmap note added. The four missing PDFs remain bibliographic-only (L-10). |
| D-1 | Timeouts are now caused by GNSS-degraded meters (timing error > CP). They are elevation-dependent and observable as Msg3 failures. |
| D-2 | The residual-Doppler axis and state bin were removed. The coupling-loss severity is the new stress axis. |
| D-3, D-7 | NB-IoT subcarrier pools per CE level (24/12/12), RSRP-based CE selection and escalation, per-CE periodicity, and an action that is a repetition *profile*. |
| D-4, C-1, I-2 | P_d now passes exactly through the TS 36.104 points (tested). The GNSS-free regime is fitted to Chougrani [5] within ±1.4 dB (tested). |
| D-5, C-8 | Set-4 link budget from the SGP4 slant range; it reproduces [10] (tested). |
| D-6, C-3 | Reward normalised per active subcarrier. It uses only observables (tested). The v2 deferral-cost term was pre-declared. |
| L-6, C-7 | ML estimator from idle/single/collided counts. Load and cause come from observables only. B4 also uses the estimate. |
| D-8…D-11, C-5, C-6 | B2 renamed. B3 is trained on a true terrestrial channel with a load + cause state. B5 is an exhaustive-search genie. Warm start is trained on other satellites' passes. |
| D-12, D-13 | Docs rewritten to match the code. Energy is measured as preamble airtime. |
| C-4 | Pass-agnostic traffic: uniform reports + a Beta(3,4) burst at a random time. |
| C-9, C-10 | Delay includes airtime, RTT and processing. Barring is checked only at initiation. |
| C-11, C-12 | 14 real passes × 10 seeds, 95 % CIs, paired Wilcoxon + Holm, sweep tests. |
| C-13…C-17 | Agent and config fixes. Consistent bins and citations. |
| I-1…I-5 | All acceptance checks automated (18 tests). Sweeps over N, coupling loss, GNSS share, regime, RAO period and altitude. Per-RAO log as `rao_log.csv.gz` (no pyarrow, so no parquet). |
| E-1, E-2, E-10 | READMEs rewritten with current numbers only. Q-table size stated correctly (52 KB). Placeholders and badges removed. Versions pinned. |
| E-3…E-7 | Figures redrawn: IEEE sizes, no embedded titles, validated palette, CIs, heatmap, no dual axes. |
| E-8, E-9 | Import-time scripts removed. Tests rewritten. |
| E-11 | Git initialised. Caches removed. PDFs and zip are git-ignored. Old outputs archived. |

**Results outcome.** The headline claims of the original README did not survive. The rebuilt evaluation gives a more modest result, reported honestly in `paper/main.tex` and `nbiot_ntn_qlearning/README.md`: Pareto-efficient and load-robust, but not better than the best static configuration at nominal load.

---

Review date: 11 Nov 2026. Scope: problem statement, `01_Literature_Review.md`, `02_Methodology.md`, `03_Implementation_Plan.md`, the code in `nbiot_ntn_qlearning/`, the stored results (`results/*.csv`), the figures, the READMEs, and the earlier `Project_Audit.pdf` (11 Oct 2026).

How it was checked:
- I read every source file.
- I ran the test suite: 9/9 tests pass.
- I compared the stored CSVs with the claims in the README.
- I checked the cited numbers against the extracted paper texts in `_extracted/`.
- I re-ran the full experiment from scratch to test reproducibility (see §0).

Severity levels:
- **[C] Critical**: invalidates a paper claim, or a reviewer would reject on it.
- **[M] Major**: modelling or method flaw that changes the results.
- **[m] Minor**: hygiene or wording.

---

## 0. Headline summary

1. **The stored results do not support the claims in either README.** The README claims a 36.4 % transmission saving vs B2, +53.8 % success at N = 3000, and a halved collision rate (21.8 % vs 40.0 %). None of these numbers appear in `results/*.csv`. They come from an older run that the Oct-11 audit already said does not reproduce. What the current data shows:

| Scheme (N = 1000, 10 Hz, 20 seeds) | P_s | Mean delay | 95th-pct delay | Tx / success | Collision rate | Miss rate |
|---|---|---|---|---|---|---|
| B1 Fixed terrestrial | 0.988 | **381 ms** | 1.47 s | 1.78 | 19.1 % | 24.7 % |
| B2 "Rel-17 NTN default" | **1.000** | 2.19 s | 9.33 s | **1.11** | 7.7 % | 2.1 % |
| B3 Terrestrial-frozen | 1.000 | 4.31 s | 21.0 s | 1.19 | 9.4 % | 6.5 % |
| B4 Heuristic | **1.000** | **245 ms** | **0.57 s** | **1.11** | 9.3 % | **0.5 %** |
| **Proposed (warm)** | 0.994* | **4.60 s (worst)** | **21.2 s (worst)** | 1.21 | 8.8 % | 8.1 % |
| Ablation (lumped) | 0.991* | 4.35 s | 21.4 s | **1.14** | 9.9 % | **2.0 %** |

\* P_s within the 45 s deadline.

   - The proposed controller is **the slowest scheme**. It uses **more** transmissions than B2.
   - It **does not beat the lumped-reward ablation**: the ablation has fewer misses and fewer transmissions. This contradicts gap G3, the paper's core novelty claim.
   - In the sensitivity sweep, the proposed scheme is **below B2 in every cell** (−0.6 to −2.3 points). Its "energy saving" vs B2 is **−35 %** at N = 200 and only +4 to +7 % at N = 3000.
2. **Two of the three "failure causes" have no effect in the simulator.**
   - Timeouts are **impossible**: the maximum RTT is 15.6 ms, but the RAR window is 24 ms or 40 ms.
   - Residual Doppler of 10 Hz vs 250 Hz changes P_d by **less than 0.01**. The sinc loss at 250 Hz is 0.06 dB.
   - So the "Doppler stress" axis, the Doppler state bin, and the timeout reward term are all inactive.
3. **The detection model is about 14–21 dB more optimistic than Chougrani et al. [5]**, which it claims to be calibrated to.
   - The model gives a 1 % miss rate at −5.9 dB with 32 repetitions.
   - Chougrani needs +7.9 to +15.5 dB (32 reps) and +2.8 to +9.3 dB (64 reps).
4. **Many items from the Oct-11 audit are still open** in the documents, for example: "48 kHz [8]", the Chen "collapse" misquote, Deep Dyna-Q "400–600 steps", "64 − 16 reserved", "2–7 min pass", and absolute novelty claims.
5. **Reproducibility is now fine.** A fresh re-run gives byte-identical `metrics.csv`, `elevation_bins.csv`, `delays.csv`, `failures.csv`, `convergence.csv`, and all three Q-tables. So the CSVs are trustworthy, but the README numbers are not (E-1).

---

## 1. Problem statement (`_extracted/ProblemStatement.txt` / `.docx`)

| ID | Sev | Issue | Evidence | Fix |
|---|---|---|---|---|
| P-1 | M | The required pass window is "~2–7 min", which is physically wrong for a 600 km orbit with a 10° mask. An overhead pass lasts about 8.5 min. | Geometry: the Earth central angle at 10° is 15.8°, so 2 × 15.8° / 360° × 96.2 min ≈ 8.5 min. The simulated pass is 8.17 min. | Restate as "≈ 3–8.5 min, depending on peak elevation". |
| P-2 | m | Section 7 / "Section 7.1–7.3" is referenced but missing from the extracted text. | | Check that the .docx is complete. |
| P-3 | m | Ref [1] author initial is "J. Tello-Oquendo"; it should be "L.". | Tello paper | Fix. |
| P-4 | m | [12] is described as "learning-automata Q-learning". "LA" means *linear approximation*. | Audit | Fix. |
| P-5 | m | [9] is given as art. 102611, 2025. The literature review gives art. 102584, 2024. | | Use one form, verified (I'll check it online). |
| P-6 | M | The "U-shape" expectation is mis-named. Plotted against elevation, P_s rises monotonically from horizon to zenith. Only a plot **against time** gives an inverted-U (∩). | Data: the B1 curve rises 0.27 → 0.91 | Call it "elevation dependence" / "∩-shape over time". |

## 2. Literature review (`01_Literature_Review.md`)

Numbers verified against `_extracted/`:
- Tello: 31.305 % → 97.44 %, c(54) = 20.05 ✔
- Kodheli: 970 ms ✔
- Shen: 698.62 µs, ±41 kHz ✔
- Liu: 700 km, RAO set up to 540, K = 400 ✔
- eRACH: 54.6 %, 1.49×, 550 km ✔
- Feng: 40 000 devices, 0.5 % of 600 s ✔
- Chen: AoI −30.23 % ✔
- Dwivedi: 230 devices, 246.9 s, ≈4 min at a 30° mask, 13 min at a 0° mask, −3.13…4.93 dB ✔
- Survey: 950 Hz, 8 kHz, ~10 min ✔
- Lee-DL: 10 Hz / 250 Hz, −13 dB ✔
- Abera: 0.9998, 40 250 vs 35 000 ✔

| ID | Sev | Issue | Fix |
|---|---|---|---|
| L-1 | M | Doppler "±41–48 kHz", with "[8] (48 kHz)". The 48 kHz figure is **not in the survey**, which gives 950 Hz and 8 kHz. | Drop the 48 kHz attribution. Quote our own SGP4 value (±46 kHz at 2 GHz) plus [5] and [11]. |
| L-2 | M | G4: "Chen [16] shows pretrained eRACH collapsing to ≈0 on a fresh cell". The collapse came from a **load shift**, and both fresh and pretrained policies collapse. | Rephrase as a *risk* of warm-starting, not as support for it. |
| L-3 | M | Deep Dyna-Q "~400–600 steps" is cited as policy convergence. It is **world-model** convergence. | Rephrase. |
| L-4 | M | Jiang [12] is called "single-knob". It tunes several NPRACH parameters per CE group. | Fix. This also weakens G5, so G5 must be worded precisely. |
| L-5 | M | Absolute novelty claims ("intersection is currently empty", "the only entry that…"). | Soften. Acknowledge the overlaps with Chen [16] (ACB + RACH period), Jiang [12], and Lee [13], which suggests RL + collision-type estimation as future work. |
| L-6 | M | G3 says collisions and misses are "observable at the gNB" via [18],[19],[13]. Those works detect *whether* a collision happened, not *how many devices* collided. A missed preamble leaves no trace at all. The code uses ground-truth device counts. | State exactly what is observable (collided-subcarrier flags, idle subcarriers), and make the code match (see M-4). |
| L-7 | M | ACB in NB-IoT. Rel-13+ NB-IoT has **no `ac-BarringFactor`**. It uses AB (access barring) bitmaps per access class (SIB14-NB `ab-BarringBitmap`). | Reframe the barring factor p as "fraction of access classes not barred via the AB bitmap" (10 ACs → steps of 0.1), or cite LTE-M / an NB-IoT ACB extension. See the questions. |
| L-8 | m | [13] arXiv:2512.07053 is dated 2026, but the ID means Dec 2025. | Use 2025 unless a 2026 journal version exists. |
| L-9 | m | 3GPP primary source for the 10 Hz residual (R1-2100601 / TR 38.821) is not cited directly. | Add it. |
| L-10 | m | The PDFs for [2] Bui-Pham, [4] Daraseliya, [9] SCMA-Q, and [12] Jiang are not in `papers/`, so their claims can't be checked locally. | Download them, or verify online (bibliographic data only). |
| L-11 | m | Some references are never cited by number in the body. The numbering differs from the problem statement. | Renumber to match the paper's order of citation. |

## 3. Methodology (`02_Methodology.md`) vs literature and vs code

| ID | Sev | Issue | Fix |
|---|---|---|---|
| D-1 | C | "Timeout: `2τ + t_proc > T_RAR`; right response: NTN-extended RAR / BI spacing". In Rel-17 NTN the UE applies TA pre-compensation, and the RAR window start is offset by the UE–gNB RTT. So a correctly configured NTN UE **never** hits this timeout. None of the three actions (ACB, BI, N_rep) can fix a deterministic timeout either. | Drop timeout from the controller's reward, or redefine it as something real (see the questions). |
| D-2 | C | "Residual Doppler drives misses" with δf = 10 / 250 Hz. Physically, 250 Hz / 3.75 kHz = 0.067 SCS, which gives 0.06 dB ICI loss. Lee [13] confirms performance is stable up to 250 Hz. So the Doppler bin and the Doppler sweep can't produce an effect. | Remove the Doppler state bin. Replace the stress axis with something that matters: a timing/frequency-error severity ("GNSS-degraded"), a per-device coupling-loss spread, or a larger CFO. See the questions. |
| D-3 | M | "48 = 64 − 16 reserved" is LTE PRACH logic. In NB-IoT, the NPRACH "preambles" are **subcarriers**: 12/24/36/48 per CE level, minus those reserved for multi-tone Msg3. | Fix the justification. |
| D-4 | M | P_d is "calibrated to [5]", but the code isn't (see C-1). Also, Chougrani's curves are **GNSS-free**, with differential-RTD timing uncertainty. Rel-17 NB-IoT NTN assumes GNSS. | Decide on an anchor (see the questions) and document the mapping. |
| D-5 | M | "UL SNR −3 → +5 dB [Dwivedi Set-4]". That range is beam **edge vs centre**, not 10° vs 90° elevation. The code now uses a link budget (−5.4 → +8.2 dB). The document and the code disagree. | Document the link budget (TR 38.821 Table 6.1.1.1-x style), including the missing losses: polarisation 3 dB, atmospheric/scintillation, shadowing margin. |
| D-6 | M | The reward is a **ratio per transmission**: `(…)/n_trans`. Barred devices don't count, so heavy barring with few transmissions scores well. This is why the proposed controller over-bars and has 4.6 s mean delay. | Normalise by preamble resources (a throughput-style reward), and add a backlog/age penalty. |
| D-7 | M | A single N_rep for every device, chosen by the network each RAO. In NB-IoT, each UE picks its CE level from RSRP thresholds, and repetitions are configured per CE level. Also, all meters sit at **one point**, so every device has the same SNR. | Add per-device coupling loss (indoor penetration + shadowing), and make the action a per-CE repetition **profile** (see the questions). |
| D-8 | M | B2 is called the "Rel-17 NPRACH-NTN default" (p = 0.7, BI = 1024, N_rep = 8, RAR = 40 ms), but no 3GPP spec defines such a preset. | Rename it to "static NTN-aware configuration", and justify each value. |
| D-9 | M | B4 is called an "Oracle / upper bound", but it is a hand-written heuristic. | Add a real reference: the best fixed configuration by exhaustive search over all 60 actions. Keep the heuristic as a separate, named baseline. |
| D-10 | M | B3 (terrestrial-frozen) is defined as "trained terrestrially". In the code the channel stays the NTN pass. Only the *state* is faked (a random elevation of 45/75/85°). | Train it on a true terrestrial channel (constant high SNR, about zero delay), with a load-only state projection. |
| D-11 | M | Warm start is "terrestrial / early-pass". The code pre-trains for 25 synthetic NTN passes (about 37 k updates). The claim "converges within one pass" is therefore really about **cold** start. | Pre-train on *other* passes (a different satellite or day), report the training budget, and treat cold start as the in-pass convergence test. |
| D-12 | m | Small numeric mismatches: RAO period 160 ms (doc) vs 320 ms (code); terrestrial RAR 8 ms (doc) vs 24 ms (code); traffic Beta + Poisson (doc) vs a hard-coded "multiphase" (code). | Make the docs match the final code. |
| D-13 | m | The energy proxy is "transmissions per success". It ignores repetitions: 32 reps cost 16× the airtime of 2 reps. | Use preamble airtime per success, Σ N_rep × T_preamble. |

## 4. Implementation plan (`03_Implementation_Plan.md`) acceptance checks

| ID | Sev | Check | Status |
|---|---|---|---|
| I-1 | M | Pass 2–7 min | Fails (8.17 min). The test bound was widened to 10 min instead. |
| I-2 | M | 1 − P_d matches [5] at the anchors | Not met. The test only checks miss ≤ 1e-2, which a far-too-optimistic curve passes trivially. |
| I-3 | M | Timeouts appear when 2τ + t_proc > T_RAR | Never exercised in the experiments. |
| I-4 | M | B4 ≥ proposed ≥ B1; B3 < proposed; the ablation shows false barring | Not tested. The data shows proposed < B2 and proposed ≈ ablation. |
| I-5 | M | RAO-period sweep, altitude sweep, Poisson traffic, fairness, `attempts.parquet` | Not implemented. |

## 5. Code (`nbiot_ntn_qlearning/src`)

| ID | Sev | File:line | Issue |
|---|---|---|---|
| C-1 | C | `channel.py:115-148` | The detection curve is not calibrated to [5] (see D-4). The rep-gain term `10log10(N) − 0.5log2(N)` is ad hoc. The ICI term `14.3·x²` is negligible over the swept range. |
| C-2 | C | `nprach.py:166-186`, config `rar_window_ms: 24` | The timeout branch can't trigger, because the max RTT is 15.6 ms. |
| C-3 | C | `mdp.py:126-163` | Reward is a ratio per transmission, so it favours over-barring (D-6). The estimated miss count `n_s(1−P_d)/P_d` uses the simulator's **true** P_d, which the gNB can't know. `n_c` is the true count of collided **devices**, which is also not observable. |
| C-4 | M | `nprach.py:97-114` | "Multiphase" traffic uses hard-coded arrival times (20–50 s, 115–145 s, 180–210 s, 235–265 s), tuned to this one pass. No arrivals happen after 265 s of the 490 s pass, so the descending half is never loaded. It breaks for any other pass or altitude, and it looks engineered. |
| C-5 | M | `experiment.py:127-189` | The B3 "terrestrial" training still uses the NTN pass channel. Only the state elevation is randomised (D-10). |
| C-6 | M | `experiment.py:34-56` | Warm start is trained on a synthetic sinusoidal pass whose elevation, delay, and Doppler are not geometrically consistent. |
| C-7 | M | `experiment.py:266-272`, `baselines.py:200` | The state's load estimate and the B4 heuristic both read the **true backlog**, which is not observable. Use an estimator from idle/collided subcarriers. |
| C-8 | M | `channel.py:47-70` vs `orbit.py` | SNR is recomputed from elevation with a fixed 600 km altitude and Re = 6371 km. It should use the SGP4 slant range (the satellite is at about 585 km). |
| C-9 | M | `nprach.py:236, 284` | The access delay leaves out the preamble airtime (N_rep × 5.6 ms) and the Msg2–Msg4 time. |
| C-10 | M | `nprach.py:190-201` | ACB is re-applied on every retry. In LTE, the ACB check happens only at RA initiation, not on preamble retransmissions. |
| C-11 | M | `experiment.py:1-12` | The docstring promises "95 % CIs and paired significance tests vs B2", but none are computed. |
| C-12 | M | `orbit.py:167-217`, `generate_pass_trajectory_from_config` | Only **one** cherry-picked pass is used: the best of 7 days, peaking at 85.9°. The problem statement asks for "different orbital geometries". The 20 seeds only vary the traffic draw. |
| C-13 | m | `agent.py:52-55` | Warm start forces ε = 0.15, ignoring the `eps0` argument. |
| C-14 | m | `experiment.py` | The final transition of each pass is never updated. Idle RAOs (n_trans = 0) give reward 0 and still update Q, which biases values. |
| C-15 | m | `mdp.py:76` | The lumped weight is the mean of (w_c, w_m, w_t). This is defensible but undocumented. |
| C-16 | m | `config/default.yaml` | Wrong citations: `rar_window_ms 24 [TS 36.321]` (NB-IoT windows are in PDCCH periods, pp2–pp10); band n256 attributed to TR 38.811. |
| C-17 | m | `config.yaml` / `nprach.py` comments | The 3 elevation state bins (<20, 20–60, >60) are inconsistent with the 4 analysis bins (10–25, 25–45, 45–70, 70–90). |

## 6. Results, figures, tests, READMEs

| ID | Sev | Item | Issue |
|---|---|---|---|
| E-1 | C | README key results | The 36.4 %, +53.8 %, 21.8 % vs 40.0 %, and "1.38 vs 2.17" numbers are **not in the current CSVs** (see §0). The table in the README is current, but the bullets above it are stale. |
| E-2 | C | README claim "<1 KB RAM" | A 72 × 60 float64 table is 34.6 KB. |
| E-3 | M | Fig. 2 | The y-value is per-attempt success (n_s / n_trans), labelled as P_s. It is called a "U-shape". The ablation curve is missing. |
| E-4 | M | Fig. 4 | The y-axis "events per RAO" is actually a **sum over 10 seeds**. |
| E-5 | M | Fig. 5 | The shaded band is ±0.5 σ, which is misleading. Use a 95 % CI. |
| E-6 | M | Fig. 6 | It is bars, not the heatmap the plan promised. It compares against B1, which is a weak baseline. |
| E-7 | m | All figures | Titles are embedded ("Fig. 2: …"). IEEE captions belong in LaTeX. The fonts are not IEEE column width. |
| E-8 | M | `tests/test_*_quick.py` | These are scripts that run at **import**, so pytest collection runs simulations. They contain no assertions. |
| E-9 | M | `test_channel_detection_anchors` | Tautological: see I-2. `test_repetition_timing_feasibility` hard-codes 320. |
| E-10 | m | READMEs | Inconsistent test counts ("5/5" vs "9/9"). Placeholder BibTeX ("Your Name", `your-username`). Emoji headers and promotional wording. `requirements.txt` says "pinned" but uses `>=`. |
| E-11 | m | Repo hygiene | `__pycache__/` and `.pytest_cache/` are present. A 4 MB `nbiot_ntn_qlearning.zip` duplicate sits in the root. The `.docx` copies of the `.md` files will drift. |

---

## 7. What is solid and should be kept

- The SGP4 orbit module is correct. It uses a real Sateliot TLE (NORAD 60534). Delay is 1.94–6.30 ms and Doppler is ±46 kHz at 2 GHz, which is consistent with [5], [11], and [13].
- The mutually-exclusive outcome taxonomy and the per-RAO logging.
- Deterministic seeding: tests pass, and the re-run reproduces the CSVs.
- The literature review's core numbers are accurate (§2), and the gap structure G1–G5 is reasonable once softened.
- The project structure (config → modules → experiment → plots) is clean, and the runtime is small (~0.6 s per pass run).
