# Methodology
### A satellite-aware tabular Q-learning controller for NB-IoT NTN random access

This document defines **what to build and why each choice is defensible**. Every system value carries a source (a 3GPP specification or a cited paper). Values marked *(sweep)* are deliberately varied in the experiments rather than fixed. Nothing here depends on any earlier prototype code.

---

## 1. Design rationale — why this method

| Decision | Why | Grounded in |
|---|---|---|
| **Tabular Q-learning** (not deep RL) | A LEO pass gives only **~2–7 min** of learning. Deep schemes need hundreds–thousands of stationary episodes (eRACH ~200–500 [15]; Deep Dyna-Q ~400–600 steps [14]) and a pretrained deep policy *collapsed to ≈0 throughput* on a fresh cell [16]. A small Q-table converges in hundreds of updates and runs on a gateway/on-board with no GPU. | Jiang [12] (tabular-Q works for NB-IoT config); bounded-visibility gap G4 |
| **Joint action over ACB + BI + NPRACH** | The three failure modes each need a *different* knob; tuning one in isolation (all prior work) cannot fix the other two. | Gap G5; TS 36.331 / 36.321 / 36.211 |
| **Reward that isolates collision vs detection-miss vs timeout** | A terrestrial controller reads a Doppler/low-SNR *miss* as congestion and raises barring — a false-barring loop. Collisions and misses are physically distinguishable at the gNB. | Gap G3; collision detectors [18],[19]; classifier [13]; Daraseliya [4] |
| **State includes satellite geometry + pass clock** | Only way the controller can anticipate delay/Doppler and stop learning before the satellite sets. | Problem statement §4; survey open problems [8] |
| **Real-TLE SGP4 orbit** (not constant-delay or closed-form) | A reviewer must be able to sanity-check elevation/delay/Doppler against real orbital mechanics. | Problem statement §6; magnitudes cross-checked vs [5],[11],[13] |

---

## 2. System model

### 2.1 Orbit & channel (built from a real TLE)

Propagate a **real LEO TLE at ≈600 km** with SGP4 (Skyfield) over one visible pass; from the geometry derive, per time-step, the elevation `ε(t)`, slant range `d(t)`, range-rate `ṙ(t)`, one-way delay `τ(t)=d(t)/c`, and Doppler `f_D(t)=ṙ(t)·f_c/c`.

| Parameter | Value / range | Source |
|---|---|---|
| Orbit altitude | **600 km** LEO | TR 38.811 reference orbit [S1]; used by [5],[10],[13] |
| TLE | Real element set for an NB-IoT-NTN-class LEO sat (e.g. Sateliot / a 600 km LEO), from CelesTrak/Space-Track | Problem statement §6 ("pull a real TLE") |
| Carrier frequency `f_c` | **2 GHz**, S-band (n256 UL 1980–2010 MHz) | [5],[8],[11]; 3GPP NTN band |
| Min elevation (mask) | **10°** | 3GPP NTN min elevation [S2]; survey [8] |
| Pass duration `T_pass` | **~2–7 min** (follows from mask & altitude) | Dwivedi [10] (~4 min @30°, ~13 min @0°); problem statement §7 |
| One-way delay `τ` | **~2 ms** (zenith) → **~6.4 ms** (10° edge) | computed; matches [13] (2–6.44 ms), [7] |
| Raw Doppler `f_D` | up to **±41–48 kHz** | [5] (±41 kHz @30°), [8] (48 kHz), [11] (±41 kHz) |
| **Residual Doppler** after ephemeris/GNSS pre-compensation `δf` | **~10 Hz** nominal; **up to ~250 Hz** stress *(sweep)* | [13] (10 Hz nominal, 250 Hz stress) |
| Differential delay across beam | up to **~700 µs** | [11] (698.62 µs); [5] |
| UL SNR vs elevation | **~−3 dB (edge) → ~+5 dB (zenith)** single-tone, lower with repetition | Dwivedi [10] (Set-4 UL SNR −3.13…4.93 dB) |

> **Why residual Doppler, not raw, drives misses:** the UE pre-compensates raw Doppler from broadcast ephemeris; what degrades detection is the *residual* `δf` (tens–hundreds of Hz) relative to the 3.75 kHz NPRACH subcarrier spacing [5],[6],[13].

### 2.2 NB-IoT NPRACH contention model

A Monte-Carlo NPRACH engine advances in Random-Access Opportunities (RAOs). At each RAO, each backlogged meter: (i) draws an ACB check against barring factor `p`; (ii) if allowed, picks one of `R` preambles; (iii) the outcome is classified into exactly one cause (next section); (iv) failed meters back off by a BI-derived delay and retry up to `preambleTransMax`.

| Parameter | Value | Source |
|---|---|---|
| Contention preambles `R` | **48** (64 − 16 reserved) | TS 36.211 [S3]; [10] |
| NPRACH subcarrier spacing | **3.75 kHz** (format 0/1) | TS 36.211 [S3]; [5] |
| RAO periodicity | **160 ms** default *(sweep 160–640 ms)* | [10]; TS 36.211 [S3] |
| `preambleTransMax` (CE) | **10** | TS 36.321 [S4] |
| RAR window | **terrestrial 8 ms vs NTN-extended ≥20 ms** | TS 36.321 [S4]; NTN extension [S2] |
| Detection probability `P_d(SNR, δf, N_rep)` | **calibrated to published miss-detection curves** (miss ≤1e-2 target at ~8 dB/64-rep, ~15 dB/32-rep) | [5] (Chougrani detection curves) |

### 2.3 Failure taxonomy (the core instrumentation)

Every attempt is logged as exactly one of:

- **Collision** — ≥2 allowed meters chose the same preamble. *Right response: raise ACB barring (and/or spread with BI).*
- **Detection miss** — a cleanly-sent preamble not decoded, probability `1−P_d(SNR,δf,N_rep)`. *Right response: raise NPRACH repetitions / CE level — barring does nothing here.*
- **Timeout** — reply could not arrive within the RAR window: `2τ + t_proc > T_RAR`. *Right response: NTN-extended RAR / BI spacing.*
- **Success** — allowed, unique preamble, detected, in-window.

That a collision (multi-peak) is separable from a miss (no peak) at the gNB is exactly what the early collision detectors [18],[19] and the DL classifier [13] establish — so this state is "something the network can plausibly know."

---

## 3. MDP formulation

**Agent:** the gNB/gateway RACH controller. **Decision epoch:** one RAO (or one SIB period). **Objective:** maximise access success and minimise delay across the pass while attributing each failure to its true cause.

### 3.1 State `s = (load, geometry, doppler, pass-phase)`

| Component | Discretisation (design; *validated by sensitivity*) | Observable because |
|---|---|---|
| Load estimate `L̂` | 4 bins: low / moderate / high / overload (from backlog or recent attempts/RAO) | gNB counts attempts & detected preambles/RAO |
| Elevation (⇔ delay) `ε` | 3 bins: low (<20°) / mid / high (>60°) | broadcast ephemeris |
| Residual Doppler `δf` | 2 bins: nominal (≤~50 Hz) / degraded | from CFO estimate [5],[13] |
| Pass phase `φ` | 2–3 bins by time-remaining (early / late) | pass clock from ephemeris |

Keep `|S|` small (order **~50–150 states**) so the table is *visitable within one pass*. Adding a bin multiplies the table and risks non-convergence inside `T_pass` — a bin count is itself a hypothesis to test (§6, Fig. 5/6).

### 3.2 Action `a = (ACB, BI, N_rep)`

| Knob | Candidate levels (design) | Source |
|---|---|---|
| ACB barring factor `p` | subset of {0.1, 0.3, 0.5, 0.7, 1.0} from the TS 36.331 `ac-BarringFactor` grid (p00…p95) | TS 36.331 [S5] |
| Backoff Indicator `BI` | log-spaced NB-IoT backoff values (e.g. {256, 1024, 4096, 16384} ms) | TS 36.321 Table 7.2-2 [S4] |
| NPRACH repetition / CE level `N_rep` | 3 CE levels → representative {low, mid, high} repetitions (powers of 2, e.g. {2, 8, 32}) | TS 36.211 CE groups [S3] |

`|A| ≈ 5 × 4 × 3 = 60`. Table size `|S|·|A|` on the order of a few thousand entries — small enough to converge in hundreds of updates (cf. tabular-Q NB-IoT [12]).

### 3.3 Reward — cause-separated (the novelty)

Per RAO, with `n` attempts and counts `n_s, n_c, n_m, n_t` (success/collision/miss/timeout):

```
r = ( w_s·n_s − w_c·n_c − w_m·n_m − w_t·n_t ) / n   −   w_d · (d̄_recent / d_ref)
```

The **separation of `n_c`, `n_m`, `n_t`** — not the exact weights — is the contribution. Rationale for the ordering of weights (to be tuned, then fixed):

- `w_c` **largest** → collisions are the only cause barring can fix, so they must dominate the barring gradient.
- `w_m` **small** → a miss must *not* trigger barring (that is the false-barring loop); it should instead push `N_rep` up. Keeping `w_m` light prevents the agent from barring away a channel problem.
- `w_t` **middle** → timeouts call for RAR/BI spacing, not barring.
- `w_d` delay regulariser so the agent does not over-bar.

An ablation with a **single lumped failure penalty** (no separation) is run to show the separation is what prevents the false-barring loop (Fig. 4).

### 3.4 Learning rule & schedule

Standard Q-learning:
`Q(s,a) ← Q(s,a) + α [ r + γ·max_{a'}Q(s',a') − Q(s,a) ]`, ε-greedy with decay.

| Hyperparameter | Starting value | Source |
|---|---|---|
| Learning rate `α` | ~0.1–0.15 | tabular-Q NB-IoT precedent [12]; retune |
| Discount `γ` | ~0.9 | [12]; retune |
| Exploration `ε` | 1.0 → 0.05, decay ~0.995 | high-explore early pass, exploit late |

**Convergence strategy (bounded-visibility, gap G4):**
- **Cold-start** — Q initialised to zero at the start of each pass.
- **Warm-start** — Q initialised from a table trained on terrestrial / early-pass conditions, then fine-tuned online during the pass.
The experiments test whether warm-start converges inside `T_pass` where cold-start may not — the evidence for "satellite conditions genuinely require adaptation."

---

## 4. Baselines (mandatory for a fair claim)

| Baseline | Definition | Purpose |
|---|---|---|
| **B1 Fixed ACB+BI** | terrestrial default `p`, `BI`, `N_rep`, no adaptation | today's deployment |
| **B2 Rel-17 NPRACH-NTN default** | NTN-extended RAR + fixed NTN preset | standards-compliant non-learning |
| **B3 Terrestrial-frozen Q-table** | Q-table trained terrestrially, applied to the satellite channel **unmodified** | **critical ablation** — proves adaptation is needed |
| **B4 Oracle** | controller with perfect per-RAO channel knowledge | honest upper bound |
| **Ablation** | proposed reward with **lumped** failure penalty | isolates the value of cause-separation |

---

## 5. Input / output parameters

**Inputs (independent variables / config)**

| Input | Role | Default / sweep |
|---|---|---|
| Device count `N` | load | *(sweep)* 200 / 1000 / 3000 (rural → congested → beyond-capacity) |
| Traffic model | arrival process | **Beta(3,4)** synchronized burst (TR 37.868 [S6]) + Poisson routine |
| Orbit altitude | delay/Doppler/pass-length | 600 km *(sweep for sensitivity)* |
| Residual Doppler `δf` | detection stress | *(sweep)* 10 / 250 Hz [13] |
| RAO periodicity | decision cadence | 160 ms *(sweep 160–640)* |
| Policy | which controller | {proposed, B1–B4, ablation} |
| Random seed | variance | ≥10 independent seeds |

**Outputs (dependent variables / metrics — all reported with mean ± std over seeds)**

| Output | Definition |
|---|---|
| Access success probability `P_s` | served meters / total meters |
| Mean & **95th-percentile** access delay | from first attempt to successful RAR |
| **Failure-cause breakdown** | counts/fractions of collision / miss / timeout |
| Convergence | RAOs (or passes) until reward/`P_s` stabilises; cold vs warm |
| Energy proxy | mean preamble transmissions per success (battery-relevant) |
| Fairness (optional) | Jain index across meters |

---

## 6. Graphs (the six the study must produce)

| Fig | Content | Expected shape / what it proves | Source req. |
|---|---|---|---|
| **1** | Orbit & pass geometry: elevation, one-way delay, Doppler vs time | sanity-check the channel driving everything (from SGP4) | PS §5 |
| **2** | `P_s` vs elevation across a pass — proposed vs B1–B4 | **U-shape** (poor near horizon, good near zenith); proposed lifts the horizon tails | PS §5 (headline) |
| **3** | Access-delay distribution (CDF/box) — proposed vs B1, B3 | proposed shifts the tail left (esp. 95th pct) | PS §5 |
| **4** | Failure-cause breakdown over the pass (stacked) — proposed vs lumped-reward ablation | proposed converts misses→repetitions & collisions→barring correctly; lumped reward shows false-barring | PS §5 (key) |
| **5** | Convergence within one pass — reward/`P_s` vs RAO index, **cold vs warm start** overlaid | warm-start stabilises inside `T_pass`; cold may not | PS §5 |
| **6** | Sensitivity heatmap: success gain (proposed − B1) over `N` × `δf` (and altitude) | where the advantage grows/shrinks | PS §5 |

---

## 7. Where every reference comes from (summary)

- **Method lineage (tabular-Q for NB-IoT, ACB foundations):** Jiang [12], Tello-Oquendo [1], Abera [3], Bui-Pham [2], Daraseliya [4].
- **Channel magnitudes (delay, Doppler, residual, detection curves):** Chougrani [5], Shen [11], Lee-DL [13], Kodheli [7], Dwivedi [10], survey [8].
- **Failure-cause observability at the gNB:** collision detectors [18],[19], DL classifier [13], Daraseliya [4].
- **Bounded-visibility constraint & why deep RL struggles:** Dwivedi [10], eRACH [15], Deep Dyna-Q [14], Chen [16].
- **Closest competitors to differentiate against:** Chen channel-aware ACB [16], Feng DDPG joint [17], SCMA-Q-learning [9].
- **Protocol/parameter values:** 3GPP TS 36.211 [S3], TS 36.321 [S4], TS 36.331 [S5], TR 37.868 [S6], TR 38.811/38.821 [S1,S2].

Full citations are in `01_Literature_Review.md` §5.
