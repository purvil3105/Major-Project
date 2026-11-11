# Methodology
### A satellite-aware, cause-aware tabular Q-learning controller for NB-IoT NTN random access

This document describes exactly what the code in `nbiot_ntn_qlearning/` implements. Each parameter is tagged with its source: a 3GPP document, a cited paper ([n] numbering as in `01_Literature_Review.md`), a design choice, or an assumption listed in `ASSUMPTIONS.md`. The previous version of this document is in git history. `REVIEW_ISSUES.md` lists what changed and why.

---

## 1. Design rationale

| Decision | Why | Grounded in |
|---|---|---|
| **Tabular Q-learning** | A pass gives ~1 100–1 600 NPRACH opportunities. A 108 × 60 table (52 KB) can be updated per opportunity on a gateway or on board, with no GPU. Deep RL needs far more data [14],[15], and it is fragile under load shift [16]. | Jiang [12]; gap G4 |
| **Joint action: barring + BI + repetition profile** | The three failure causes respond to different knobs: collisions to barring and backoff, misses to repetitions, timeouts to backoff spacing. | Gap G5; TS 36.321 / 36.331 |
| **Cause information from observables** | A controller that sees only "failures" cannot tell congestion from coverage loss. The gNB can observe collided subcarriers [18],[19] and infer misses statistically. | Gap G3 |
| **Satellite geometry and pass phase in the state** | Elevation (from ephemeris) predicts SNR and timing error. The pass phase tells the controller how much visibility is left. | Gap G1 |
| **Real-TLE SGP4 geometry, several passes** | Results must hold across orbital geometries, not just one hand-picked pass. | Problem statement §4 |

---

## 2. System model

### 2.1 Orbit and geometry
- **Satellites and passes.** Real Sateliot TLEs (NORAD 60534 etc., epoch Oct 2026, ~577 km, 97.7° inclination) are propagated with SGP4 (Skyfield) for 7 days after the TLE epoch.
- **Ground site.** Ahmedabad, India.
- **Test geometry.** All 14 passes of SATELIOT_2 that rise above a 10° mask and peak at ≥ 20°. Peaks range from 20° to 86° and durations from 364 to 490 s.
- **Warm-start training.** 40 passes of SATELIOT_1/3/4. These are disjoint from the test set.
- **Per second along each pass:** elevation ε, slant range d, range rate (Skyfield topocentric rates), delay τ = d/c, and Doppler f_D = −ṙ f_c / c.
- **Sanity check:** τ is 1.94–6.30 ms and |f_D| ≤ 46 kHz at 2 GHz, consistent with [5], [11] and [13].
- **Altitude sweep:** synthetic circular orbits at 500 / 600 / 800 km with the same inclination and node (assumption A-9).

### 2.2 Uplink link budget (TR 38.821 Set-4, as tabulated in [10] Table III)

SNR = EIRP_UE + G/T − k_B − FSPL(d) − L_atm − L_SM − L_scint − L_pol − L_beam(θ) − 10 log₁₀ B

| Parameter | Value | Source |
|---|---|---|
| Carrier | 2 GHz (S-band, band 256) | TS 36.102; [5], [10] |
| UE EIRP | −7 dBW (23 dBm, 0 dBi) | [10]; NB-IoT power class 3 |
| Satellite G/T | −18.6 dB/K | Set-4 [10] |
| 3-dB beamwidth | 104.7° | Set-4 [10] |
| L_beam | 12 (θ/104.7°)² dB, θ = off-nadir angle | 3 dB at beam edge, as in [10] |
| Atmospheric / shadow margin / scintillation / polarisation | 0.1 / 3 / 2.2 / 3 dB | [10] Table III |
| B | 3.75 kHz (single-tone NPRACH) | TS 36.211 |

This reproduces [10]: −3.14 dB at 30° and +4.93 dB at 90° for 600 km. Across a pass, the outdoor SNR runs from −9.7 dB (10°) to +5 dB.

**Per-meter coupling loss.** Each meter gets an extra loss L_i ~ N(μ_L, 4 dB), fixed for the pass. It represents installation, building penetration and shadowing beyond the margin. The mean μ_L ∈ {0, 5, 10} dB is the **stress axis** (A-7).

### 2.3 NPRACH detection model

P_d(SNR, N) = 1 / (1 + exp(−k (SNR − SNR₉₉(N)) − ln 99))

SNR₉₉(N) = SNR₉₉,ref + s · log₂(N / 8) + Δ_regime

| Symbol | Value | Source |
|---|---|---|
| SNR₉₉,ref (8 reps, 1 Rx) | 0.9 dB | TS 36.104 Table 8.5.3.2.1-1: −2.1 dB (2 Rx, AWGN), +3 dB for a single Rx branch |
| s | −2.35 dB per doubling | same table: 32 reps → −6.8 dB (2 Rx) |
| k | 1.2 dB⁻¹ (miss drops from 10 % to 1 % over 2 dB) | assumption A-18 |
| Δ, GNSS-assisted (default) | 0 dB | Rel-17 NB-IoT NTN assumes GNSS |
| Δ, GNSS-free | +10.3 dB, a least-squares fit to Chougrani [5] at MD-RTD 800 µs (99 % at 7.9 dB / 32 reps and 2.8 dB / 64 reps; residual ±1.4 dB) | [5] |

Residual Doppler after ephemeris/GNSS pre-compensation (~10 Hz nominal, 3GPP R1-2100601; up to 250 Hz [13]) is below 7 % of the 3.75 kHz subcarrier spacing. Its ICI loss is under 0.1 dB, so it is **not** a stress axis.

### 2.4 NB-IoT random-access procedure (per 320 ms decision epoch)

| Element | Model | Source |
|---|---|---|
| Preamble | Format 1: CP 266.7 µs, 6.4 ms per repetition | TS 36.211 |
| CE levels | 3 levels with 24 / 12 / 12 subcarriers (48 total) | TS 36.331 NPRACH-ConfigSIB-NB; A-6 |
| CE selection | At RA initiation, the meter compares its measured SNR (σ = 1 dB error) with thresholds T_k = SNR₉₉(N_k) | TS 36.321 (RSRP thresholds); design |
| Per-CE periodicity | m_k × 320 ms ≥ N_k × 6.4 ms, with m_k ∈ {1, 2, 4, 8} | TS 36.331 |
| Barring | AB bitmap modelled as "pass with probability p". Checked **only at initiation**. A barred meter waits (0.7 + 0.6U) × 4 s | TS 36.331; A-5 |
| Contention | Uniform choice of subcarrier within the CE pool. Two or more on the same subcarrier: all fail (no capture) | A-2 |
| Retries | Wait U(0, BI) after a failure. After 4 failures at a CE level, move up one level. Drop after 10 attempts | TS 36.321 / 36.331 (`maxNumPreambleAttemptCE`, `preambleTransMax-CE`) |
| GNSS-degraded meters (timeout cause) | A fraction f_deg (default 10 %) has a stale GNSS/ephemeris fix. Its residual RTT error ~ U(0, MD-RTD(ε)), with MD-RTD = 2 D cos ε / c and D = 250 km. If the error exceeds the CP, the detected preamble's Msg3 misses its grant: **timeout**. After a timeout, the meter re-acquires GNSS with probability 0.5 | [5] Table IV; [7]; A-1 |
| Access delay | arrival → Msg4, including barring waits, backoffs, preamble airtime, 2 × RTT and processing | design |

**Ground-truth outcomes** are mutually exclusive: success, collision, miss, or timeout. They are logged for evaluation only.

**gNB observables** per CE pool:
- idle-looking subcarriers;
- successes;
- collided subcarriers flagged by an early collision detector (P_cd = 0.95 [18],[19]; A-3);
- Msg3 failures (timeouts plus unflagged collisions).

### 2.5 Traffic
Per pass, N meters need access (N ∈ {200, 1 000, 3 000, 5 000}; default 1 000):
- 70 % are periodic reports, uniform over the pass (Poisson-like);
- 30 % are an alarm-storm burst, Beta(3,4) over 10 s (TR 37.868 model 2), starting at a uniformly random time.

All policies see the same arrivals, losses and GNSS flags for a given (pass, seed). This is a paired design.

---

## 3. MDP formulation

**Agent:** the gNB/gateway RACH controller. **Decision epoch:** 320 ms.

### 3.1 Observable estimator
A slotted-ALOHA model gives, per subcarrier, the probabilities of "idle-looking", "detected single" and "collided" as functions of the attempt rate λ and the mean detection probability P̄_d. A grid maximum-likelihood fit to the EWMA-smoothed counts (β = 0.5) yields λ̂. From λ̂:
- attempts n̂ = Rλ̂;
- colliding meters ĉ = Rλ̂(1 − e^{−λ̂});
- misses m̂ = max(0, Rλ̂e^{−λ̂} − detected singles);
- timeouts t̂ = Msg3 failures − unflagged collisions.

Monte Carlo check (unit test): the load bias is < 15 % and the miss error is < 1 meter per opportunity.

### 3.2 State (108 states)
| Component | Bins | Observable because |
|---|---|---|
| Load λ̂ per subcarrier | 3: < 0.3 / 0.3–1 / > 1 | NPRACH observations (§3.1) |
| Elevation | 3: < 20° / 20–50° / > 50° | broadcast ephemeris |
| Pass phase | 3 thirds | ephemeris |
| Dominant failure cause | 4: low (< 20 % failures) / collision / miss / timeout | §3.1 estimates |

### 3.3 Action (60 actions)
| Knob | Levels | Source |
|---|---|---|
| Barring factor p | {0.1, 0.3, 0.5, 0.7, 1.0} | AB bitmap fraction (TS 36.331) |
| Backoff Indicator | {256, 1 024, 4 096, 16 384} ms | TS 36.321 Table 7.2-2 (NB-IoT) |
| Repetition profile (CE0/1/2) | low {2, 8, 32}, mid {4, 16, 64}, high {8, 32, 128} | `numRepetitionsPerPreambleAttempt` (TS 36.331) |

### 3.4 Reward (cause-separated)

r = [w_s n_s − w_c ĉ − w_m m̂ − w_t t̂] / R_active − w_e · (Σ_k n̂_k N_k) / (R_active N_max) − w_d · min(d̂ / D, 3)

d̂ = T̄_bar (1 − p) / p + f̂ · BI / 2

with w = (w_s, w_c, w_m, w_t, w_e, w_d) = (1, 1, 0.5, 0.5, 0.1, 0.5), T̄_bar = 4 s and D = 10 s.
- The normalisation is per *active subcarrier*, not per transmission. In the earlier per-transmission version, heavy barring raised the reward by shrinking the denominator.
- The w_e term is the estimated preamble airtime, an energy proxy.
- The w_d term (**v2**, ASSUMPTIONS V2-1) is the expected extra wait that the chosen barring and backoff impose on an attempting meter: a geometric number of barring periods, plus the estimated failed share f̂ times the mean backoff. It is normalised by the deadline. The gNB knows every quantity in it.
- **v1** (the pre-declared design) had w_d = 0. It is reproducible with `config/v1.yaml`, and both versions are reported.

**Ablations:**
1. *Lumped reward*: w_c = w_m = w_t = 0.667 (their mean), with the cause state kept.
2. *No cause information*: lumped reward, and the cause bin removed from the state.

### 3.5 Learning
- Q-learning with α = 0.12 and γ = 0.9.
- ε-greedy exploration. Cold start: ε from 1.0 to 0.05 with decay 0.995 per update. Warm start: ε = 0.1.
- Updates happen only on epochs with observable activity, with a terminal update at the end of the pass.
- **Warm start:** pre-trained on passes of the *training* satellites with randomised N, μ_L and f_deg, then fine-tuned online during the test pass (ε = 0.1).
  - **v2** (V2-2): 120 passes (the 40 training passes cycled 3×). ε is set per pass, falling geometrically from 1.0 to 0.05 by 80 % of the passes and held there.
  - v1: 30 passes, with per-update decay 0.9999. ε was still ≈ 0.2 at the end, so the tables were under-trained.
- **Cold start:** zero-initialised and learned within one pass.

Hyperparameters are fixed in advance, with no per-scenario tuning (A-14).

---

## 4. Baselines
| ID | Definition | Purpose |
|---|---|---|
| B1 | Fixed p = 1, BI = 256 ms, low profile | terrestrial-style default |
| B2 | Fixed p = 0.7, BI = 1 024 ms, mid profile | static NTN-aware configuration (not a 3GPP preset) |
| B3 | Q-table trained on a **terrestrial** channel (fixed SNR 4.93 dB, no delay, no GNSS errors), with a load + cause state only; deployed frozen | does a transplanted terrestrial learner suffice? |
| B4 | Rules: profile from elevation (< 25° high, < 50° mid); p and BI from λ̂ | hand-crafted channel-aware control |
| B5 | Best single fixed action by exhaustive search over all 60 actions on the test passes (2 seeds) | **genie** best static configuration |

---

## 5. Metrics (mean ± 95 % CI over 14 passes × 10 seeds; paired Wilcoxon with Holm correction)
- **P_s^{≤D}:** fraction of meters served within the deadline D = 10 s (primary metric).
- **P_s:** fraction served within the pass. **Drop rate.**
- **Delay:** mean and 95th percentile, over served meters.
- **Preamble airtime per served meter** (energy).
- **Ground-truth failure shares:** collision / miss / timeout.
- **Convergence:** reward and per-attempt success over the pass, cold vs warm.

## 6. Figures
| Fig. | Content |
|---|---|
| 1 | Pass geometry for all 14 test passes: elevation, delay, Doppler, outdoor SNR |
| 2 | Per-attempt success rate vs elevation, all schemes |
| 3 | Access-delay CDF (served meters) with the deadline marked |
| 4 | Ground-truth failure causes and chosen barring and repetitions over the pass: Proposed vs both ablations |
| 5 | Convergence within a pass: warm vs cold start (and B5 for reference) |
| 6 | Heatmaps of the P_s^{≤D} gain of Proposed over B2 and over B5, across N × coupling loss |

Additional sweeps (table form): GNSS-free regime, f_deg ∈ {0, 0.1, 0.3}, RAO period ∈ {160, 320, 640} ms, altitude ∈ {500, 600, 800} km.
