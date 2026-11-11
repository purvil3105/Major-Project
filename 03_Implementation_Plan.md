# Implementation Plan
### Building & validating the NB-IoT NTN Q-learning RACH controller

A phase-by-phase plan to implement the methodology in `02_Methodology.md`. Each phase has concrete tasks, a deliverable, and an **acceptance check** that must pass before moving on. The guiding principle: *fixed values come from 3GPP, derived values come from orbital mechanics, swept values are the experiment — and the channel comes from a real TLE via SGP4, never a closed-form stand-in.*

---

## 1. Software stack & environment

| Component | Library | Role | From |
|---|---|---|---|
| Orbit propagation | **Skyfield** (SGP4) | real TLE → elevation, range, range-rate | PS §6 |
| Numerics / Monte-Carlo | **NumPy, SciPy** | contention sim + Q-update + stats | PS §6 |
| Logging / aggregation | **pandas** | per-attempt (collision/miss/timeout) logs → figures | PS §6 |
| Plotting | **Matplotlib** | the six figures | PS §6 |
| Config | **PyYAML** (or dataclass) | one config file drives every run | reproducibility |

No MATLAB, no deep-learning framework (tabular Q-learning needs none — and avoiding one answers the on-device-overhead objection). Pin versions in `requirements.txt`; set a global seed; log the config + seed with every result.

```
python>=3.10
skyfield, sgp4, numpy, scipy, pandas, matplotlib, pyyaml, pytest
```

---

## 2. Repository layout

```
nbiot_ntn_qlearning/
├── config/
│   └── default.yaml            # all parameters (§3) — single source of truth
├── data/
│   └── tle/                    # real TLE(s) for the chosen 600 km LEO sat
├── src/
│   ├── orbit.py                # SGP4 pass geometry  (Phase 1)
│   ├── channel.py              # SNR(ε), Doppler, residual, P_d calibration (Phase 2)
│   ├── nprach.py               # RAO contention engine + failure taxonomy (Phase 3)
│   ├── mdp.py                  # state encode, action decode, reward (Phase 4)
│   ├── agent.py                # tabular Q-learning; cold/warm start (Phase 4)
│   ├── baselines.py            # B1–B4 + lumped-reward ablation (Phase 5)
│   ├── experiment.py           # run matrix, seeds, CSV logging (Phase 6)
│   └── plots.py                # six figures (Phase 7)
├── results/                    # metrics.csv, sensitivity.csv, per-attempt logs
├── figs/                       # fig1…fig6
├── tests/                      # pytest sanity checks (Phase 8)
└── README.md                   # how to reproduce every figure
```

---

## 3. Configuration (single source of truth)

One YAML holds everything, tagged by provenance so a reviewer sees it at a glance:

```yaml
# --- FIXED (3GPP) ---
carrier_hz: 2.0e9            # S-band n256         [TS/TR NTN]
scs_hz: 3750                 # NPRACH format 0/1   [TS 36.211]
n_preambles: 48              # 64 - 16 reserved    [TS 36.211]
preamble_trans_max: 10       # CE                  [TS 36.321]
acb_levels: [0.1,0.3,0.5,0.7,1.0]      # from p00..p95 [TS 36.331]
bi_levels_ms: [256,1024,4096,16384]    #              [TS 36.321 T7.2-2]
nrep_levels: [2,8,32]        # CE0/1/2 reps        [TS 36.211]
# --- DERIVED (SGP4 / physics) ---
altitude_km: 600             # TLE-consistent      [TR 38.811]
elev_mask_deg: 10            #                     [TR 38.821]
# calibration anchors for P_d from [5]: (snr_db, nrep) -> miss<=1e-2
# --- SWEPT (the experiment) ---
device_counts: [200,1000,3000]
residual_doppler_hz: [10,250]          #            [arXiv 2512.07053]
rao_period_ms: [160,640]
seeds: [200,201,...,209]
# --- LEARNING ---
alpha: 0.12 ; gamma: 0.9 ; eps0: 1.0 ; eps_min: 0.05 ; eps_decay: 0.995
```

---

## 4. Phase plan

### Phase 0 — Setup & reproducibility (0.5 wk)
- Repo, venv, pinned `requirements.txt`, config loader, global RNG seeding.
- **Acceptance:** `pytest` runs; config loads; seed makes two runs byte-identical.

### Phase 1 — Orbit module (`orbit.py`) (1 wk)
- Load a **real TLE** (CelesTrak/Space-Track) for a 600 km NB-IoT-NTN-class satellite + a ground AMI site; use Skyfield to compute, per 1 s step over one pass above the 10° mask: elevation `ε(t)`, slant range `d(t)`, range-rate `ṙ(t)`.
- Derive one-way delay `τ(t)=d(t)/c` and Doppler `f_D(t)=ṙ(t)·f_c/c`.
- **Acceptance (sanity vs literature):** pass length **2–7 min**; `τ` ∈ **~2 ms (zenith) … ~6.4 ms (edge)** [13]; peak `|f_D|` **≤ ~41–48 kHz** [5],[8],[11]. If any is off, the TLE/geometry is wrong — stop and fix. Produces **Fig. 1**.

### Phase 2 — Channel & detection (`channel.py`) (1 wk)
- Map elevation → UL SNR (edge ≈ −3 dB, zenith ≈ +5 dB single-tone [10]).
- Model residual Doppler `δf` as a swept input (10 / 250 Hz [13]) — *not* raw Doppler (UE pre-compensates ephemeris).
- Implement `P_d(SNR, δf, N_rep)` and **calibrate it to Chougrani's miss-detection curves** [5] (anchor points: miss ≤1e-2 at ~8 dB/64-rep and ~15 dB/32-rep; degrade with `δf/SCS`). Document the calibration in a docstring + a test.
- **Acceptance:** at the anchor points `1−P_d` matches [5] within a stated tolerance; `P_d` rises with `N_rep`, falls with `δf`.

### Phase 3 — NPRACH contention engine + failure taxonomy (`nprach.py`) (1.5 wk)
- Event-driven loop over RAOs for a full pass. Beta(3,4) synchronized activation [S6] + optional Poisson; backlog with BI-derived re-tries; drop after `preambleTransMax`.
- Per attempt, classify into **exactly one** of {success, collision, miss, timeout} per §2.3 (timeout when `2τ+t_proc > T_RAR`). Log every attempt to a pandas frame (`t, elev, cause, delay, N, δf, policy`).
- **Acceptance:** with ACB off and `N` large, collisions dominate and `P_s` collapses toward the Tello-Oquendo regime (~0.3 under heavy sync load [1]); with `N` tiny, `P_s`→~1; miss-rate tracks `δf`/elevation; timeouts appear only when `2τ+t_proc > T_RAR`. Counts are mutually exclusive and sum to attempts.

### Phase 4 — MDP + tabular agent (`mdp.py`, `agent.py`) (1.5 wk)
- `encode_state(load, elev, δf, pass_phase)` → index over the compact `|S|` (§3.1); `decode_action(a)` → `(p, BI, N_rep)` over `|A|≈60`.
- Cause-separated reward (§3.3) with configurable weights `w_s,w_c,w_m,w_t,w_d`.
- Q-learning update, ε-greedy decay; **cold-start** (Q=0) and **warm-start** (load a pre-trained terrestrial/early-pass table) entry points.
- **Acceptance:** on a fixed easy scenario the agent beats B1 after training; Q-values finite/stable; warm-start reaches B1-beating performance in fewer RAOs than cold-start.

### Phase 5 — Baselines & ablation (`baselines.py`) (0.5 wk)
- B1 fixed ACB+BI; B2 Rel-17 NPRACH-NTN default (extended RAR); **B3 terrestrial-frozen Q-table** (train terrestrially, apply unmodified); B4 oracle (perfect per-RAO channel → greedy best action); lumped-reward ablation (same agent, single failure penalty).
- **Acceptance:** B4 ≥ proposed ≥ B1 on `P_s` (ordering sane); B3 underperforms proposed on the satellite channel (the adaptation claim); lumped-reward ablation shows more barring on miss-heavy states.

### Phase 6 — Experiment runner (`experiment.py`) (1 wk)
- Train the warm-start table on mixed conditions so all state bins are visited; then evaluate every policy over **≥10 seeds**; report mean ± std.
- Sensitivity sweep over `N × δf` (and an altitude variant) with a short per-cell fine-tune for the proposed scheme (honest, pass-length-respecting).
- Emit `results/metrics.csv`, `results/sensitivity.csv`, `results/attempts.parquet`.
- **Acceptance:** CSVs populated; confidence intervals (not single runs); runtime acceptable.

### Phase 7 — Figures (`plots.py`) (0.5 wk)
- Generate Fig. 1–6 per the methodology table. **Fig. 2 and Fig. 4 must be measured from the simulator's own logged outcomes**, never from a closed-form shortcut.
- **Acceptance:** all six render; Fig. 2 shows the U-shape with proposed lifting the horizon tails; Fig. 4 shows cause-correct responses vs the lumped ablation.

### Phase 8 — Validation, tests & write-up (1 wk)
- `pytest` sanity suite encoding every acceptance check above (orbit magnitudes, detection anchors, mutually-exclusive causes, baseline ordering).
- Cross-check table: simulated Doppler/delay vs [5],[11],[13]; overload `P_s` vs [1].
- README: one command per figure; record TLE epoch, seeds, config hash.
- **Acceptance:** tests green; numbers reproduce; results map 1-to-1 onto the problem statement's six required graphs.

---

## 5. Experiment matrix (runs → figures)

| Run | Policies | `N` | `δf` | Seeds | Produces |
|---|---|---|---|---|---|
| R1 geometry | — | — | — | — | Fig. 1 |
| R2 success vs elevation | proposed, B1–B4 | 1000 | 10 Hz | 10 | Fig. 2 |
| R3 delay dist. | proposed, B1, B3 | 1000 | 10 Hz | 10 | Fig. 3 |
| R4 failure breakdown | proposed, lumped-ablation | 1000 | 250 Hz | 10 | Fig. 4 |
| R5 convergence | proposed cold, proposed warm | 1000 | 250 Hz | 10 | Fig. 5 |
| R6 sensitivity | proposed, B1 | 200/1000/3000 | 10/250 Hz | 10 | Fig. 6 |

Headline table: `P_s`, mean & 95th-pct delay, failure split, convergence RAOs — proposed vs B1–B4, mean ± std.

---

## 6. Risk register

| Risk | Impact | Mitigation |
|---|---|---|
| Real TLE gives an atypical pass | geometry unrepresentative | pick a TLE whose pass sanity-checks against Phase-1 bounds; report epoch; test 2–3 TLEs |
| Q-table too large to converge in a pass | core claim fails | keep `|S|` ~50–150; warm-start; treat bin count as a tested hypothesis (Fig. 5/6) |
| Detection model un-grounded | reviewer rejects `P_d` | calibrate strictly to [5]; expose anchors in a test |
| Reward weights look hand-tuned | "cherry-picked" critique | fix weights before final runs; justify ordering by cause→knob; include lumped ablation |
| Single-run results | not falsifiable | ≥10 seeds, mean ± std everywhere |
| Scope creep (power ramping, HARQ, beam size) | 10× action space, no convergence | explicitly out of scope; action stays the ACB/BI/NPRACH triple |

---

## 7. Timeline (≈9 weeks, single student)

| Wk | Phase |
|---|---|
| 1 | P0 + P1 (orbit, Fig. 1) |
| 2 | P2 (channel/detection, calibrated to [5]) |
| 3–4 | P3 (NPRACH engine + failure taxonomy) |
| 5–6 | P4 (MDP + tabular agent, cold/warm) |
| 6 | P5 (baselines + ablation) |
| 7 | P6 (experiments, seeds, sensitivity) |
| 8 | P7 (six figures) |
| 9 | P8 (tests, cross-validation, write-up) |

---

## 8. Definition of done

- Six figures reproduce from one command each, from logged simulator outputs.
- Proposed scheme beats B1/B2, beats B3 on the satellite channel (adaptation proven), and sits below the B4 oracle.
- Fig. 4 shows the controller responds to the *right* cause; the lumped-reward ablation shows the false-barring loop it avoids.
- Warm-start converges within one pass where cold-start does not (Fig. 5).
- Every parameter traces to a 3GPP spec, a cited paper, or an explicit sweep; simulated Doppler/delay cross-check against [5],[11],[13].
