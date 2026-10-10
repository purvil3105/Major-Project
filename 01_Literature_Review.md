# Literature Review
### Q-Learning-Based RACH Parameter Optimization for NB-IoT Non-Terrestrial Networks

> Scope: learning-based and analytical random-access (RA) control for cellular / NB-IoT and LEO-satellite IoT, read against the proposed contribution — a **lightweight tabular Q-learning controller that jointly adapts ACB + Backoff Indicator (BI) + NPRACH configuration for NB-IoT NTN, isolates detection failures from contention collisions, and converges inside a single LEO pass.**

---

## 1. How the field is organised

The relevant work splits into three bodies that have not yet been joined:

1. **Learning-based ACB / RACH tuning on terrestrial cellular & NB-IoT.** Strong controllers, but every one assumes a channel that does not move.
2. **NB-IoT / NTN random access at the physical and timing layer.** Deep understanding of delay and Doppler, but it stops at the receiver — it never asks the *access-control* parameters to adapt.
3. **Machine learning for LEO-satellite random access more broadly.** Learning is applied to satellites, but for a different multiple-access scheme (SCMA/ALOHA), or at the resource/association layer, never to the NB-IoT ACB-and-NPRACH triple.

The proposed work sits at the intersection of all three, which — on the evidence below — is currently empty.

---

## 2. Comparison of the 10 most relevant papers

Columns: **State/Input · Action · Reward/Objective · Traffic & Channel · Method · Key quantitative result · Gap it leaves open.** Grouped by cluster. (Nine further works are discussed in §3.)

### Cluster A — Terrestrial learning-based ACB / RACH tuning

| # | Paper | State / Input | Action | Reward / Objective | Traffic & Channel | Method | Key quantitative result | Gap left open |
|---|-------|---------------|--------|--------------------|-------------------|--------|-------------------------|---------------|
| 1 | **Tello-Oquendo et al., 2018** — Optimal ACB config, LTE-A massive M2M (IEEE TVT 67(4)) | Offered load, RACH statistics | ACB barring rate `P_ACB` + barring time `T_ACB` (static) | Keep access success ≥ 0.95 at min delay | Beta(3,4)/10 s & uniform/60 s (TR 37.868); 54 preambles; terrestrial, static | Analytical PRACH model + exhaustive parameter sweep (**no learning**) | Baseline P_s ≈ **31%** at 30 000 UEs → **97.4%** with P_ACB=0.5, T_ACB=4 s; PRACH capacity ≈ 20.05 preambles/RAO | Static params; no Doppler/elevation/delay; no learning; no NTN |
| 2 | **Jiang et al., 2019** — RL for real-time optimisation in NB-IoT (IEEE JSAC 37(6)) | Per-CE-group traffic & resource state | NPRACH resource / repetition configuration | Maximise served devices / resource efficiency | NB-IoT, terrestrial, multi-CE-group | **Tabular Q-learning**, linear-Q and DQN compared | Establishes tabular-Q as effective, low-complexity baseline for NB-IoT config; near-optimal with no channel knowledge | Terrestrial only; no satellite geometry, Doppler, elevation; single-knob (not ACB+BI+NPRACH jointly) |
| 3 | **Abera et al., 2025** — DRL-based ACB for mMTC (Elsevier *Array* 28:100533) | Mean successful preambles/RAO, Δ, σ, current barring factor β | ACB barring factor β ∈ {0.05…1.0} (16 levels) | Maximise access success (P_s ≥ 0.95) | M2M Beta(3,4)/10 s + **real** H2H (Telecom Italia); LTE-A/5G-NR FR1, terrestrial | **Dueling Double DQN** (value+advantage streams; experience replay) | **P_s = 0.9998**; optimal up to **40 250** M2M UEs vs 35 000 for Q-learning baseline | Static terrestrial channel; no NTN delay/Doppler; deep net (heavier than tabular); no failure isolation |

### Cluster B — NB-IoT / NTN random access at PHY & timing layer

| # | Paper | State / Input | Action | Reward / Objective | Traffic & Channel | Method | Key quantitative result | Gap left open |
|---|-------|---------------|--------|--------------------|-------------------|--------|-------------------------|---------------|
| 4 | **Chougrani et al., 2022** — NB-IoT RA for NTN: preamble detection & UL sync (IEEE IoT-J 9(16)) | Received NPRACH signal; residual CFO | Receiver config (CP extension, Doppler-rate-aided ToA) | Meet 3GPP detection ≥99%, FA ≤0.1%, ToA err ≤3.646 µs | LEO 600 km, S-band 2 GHz, 30–90° elev, **no GNSS**; NPRACH format 1 (3.75 kHz) | PHY detector + 1D-DFT ToA + Doppler-rate ML; **no MAC/ACB adaptation** | Doppler **±41 kHz** @30°; diff. RTD up to **1.4 ms**; miss ≤1e-2 at ~8–15 dB (32/64 reps) | Pure PHY; access-control parameters untouched; no learning |
| 5 | **Kodheli et al., 2021** — RA over NTN: theory to practice (IEEE Access 9) | Propagation-delay / timer state | SF-level TA, RAO postponement, **RA-timer extension** | Make the 4-step handshake complete under NTN delay | LEO 600 km → GEO; LTE Band 7 testbed; single-user | Protocol fixes + OAI + HW channel emulator (**deterministic, not learned**) | End-to-end RA demonstrated to GEO; single-user access time up to **970 ms** | Single-user (no congestion/collision); Doppler out of scope; timers reconfigured deterministically, not by load/learning |
| 6 | **Shen et al., 2025** — LEO RA with large differential delay & Doppler (IEEE TWC 24(4)) | Rx signal (delay-Doppler domain) | Joint device-ID + channel-est. + symbol detection | Grant-free detection under overspread channel | LEO (3GPP Set-2), **no GNSS**, OTFS; 40 devices, p=0.1 | Message-passing (GAMP + MRF) **+ CNN** for detection (not RL, not access control) | Diff. delay up to **698.62 µs**, Doppler **±41 kHz** (τ·ν≈28.6, overspread); SER < 0.01 at Q=12 | Grant-free detector; no ACB/BI/NPRACH; no bounded-visibility learning |

### Cluster C — Learning / control for LEO-satellite random access

| # | Paper | State / Input | Action | Reward / Objective | Traffic & Channel | Method | Key quantitative result | Gap left open |
|---|-------|---------------|--------|--------------------|-------------------|--------|-------------------------|---------------|
| 7 | **Liu et al., 2022** — Deep Dyna-RL RA control, LEO satellite IoT (IEEE IoT-J 9(16)) | # devices requesting each LEO `[N₁…N_J]` | **RAO allocation** {27…540} + EAB barring | Reward ±1 on change in RA efficiency `r_eff` | LEO 700 km; Poisson (async) + Beta (sync, TR 37.868); 54 preambles | **Deep Dyna-Q** (world-model DNN + Q-DNN; planning K=400) | Dyna-Q converges faster/higher `r_eff` than model-free DQN; world model stable after ~400–600 steps | Not NB-IoT NPRACH; tunes RAO+EAB, not ACB+BI+NPRACH; reward aggregate (no failure isolation); deep nets |
| 8 | **Lee et al., 2023** — Learning emergent RA protocol (eRACH), LEO (IEEE TWC 22(1)) | Local: SAT positions, throughput, collision flag, prev. action | SAT-plane association + {backoff,1…K} | `g(rate − ρ·collision)` normalised | Mega-constellation 550 km, 7.59 km/s; 5 agents, 2 preambles | **Multi-agent deep RL (A2C)**, fully distributed | **+54.6%** throughput vs RACH; latency 1.49× lower; converges in **~200–500 episodes** ("one day") | Not NB-IoT; replaces (not tunes) backoff/ACB; binary collision flag (no detection-vs-collision split); 2-preamble toy |
| 9 | **Feng et al., 2026** — Joint access-control + resource allocation, LEO IoT, DRL (IEEE TCCN 12) | Per-sub-channel used/idle slot counts | Channel-selection weights (+ ACB barring factor for delay-tolerant devices) | Channels saved + normalised throughput | LEO 700 km; multi-channel slotted-ALOHA; delay-tolerant + delay-sensitive; up to 40 000 devices | **DDPG** (actor-critic) + Lyapunov opt. + matching theory | Connects **>40 000** devices within 100 steps; runtime 0.5% of 600 s dwell | Not NB-IoT NPRACH; ACB is a single DT-only knob (no BI, no NPRACH); heavy DDPG+Lyapunov; no failure isolation |
| 10 | **Chen et al., 2026** — Channel-aware access & RACH enhancement, LEO IoT (IEEE IoT-J 13(11)) | Predicted per-RAO SNR from ephemeris; device class | **Multi-class ACB factors** + optimised RO/PRACH period | Maximise expected rate / AoI / fairness | LEO; Poisson, delay-tolerant, finite buffer; GNSS-synced; 2 000–12 000 UEs | Analytical KKT / reverse water-filling (**zero-shot, not RL**) | AoI −30.2% at 12 000 UEs; Jain fairness 0.85→0.90; beats DRL eRACH's cold-start (eRACH ≈0 at 8 000 UEs) | Analytic (needs near-perfect prediction); **BI set to zero**; no NB-IoT NPRACH; no detection-vs-collision isolation |

**Reading of the table.** Cluster A has the learning machinery and the ACB/NPRACH knobs but a static terrestrial channel. Cluster B has the real NTN channel but no adaptive access control and (mostly) no learning. Cluster C puts learning on satellites but for ALOHA/SCMA/association/RAO — never the NB-IoT ACB+BI+NPRACH triple, and never with the failure cause isolated inside the reward. The two 2026 competitors (Feng, Chen) are the closest in spirit, yet Feng uses heavyweight DDPG with ACB as a single side-knob and no BI/NPRACH, while Chen is analytical, zeros the BI, and is not NB-IoT.

---

## 3. Further relevant works (not in the top-10 table)

- **Lin, 2025** — *Unified RA procedure for NB-IoT in NTN & TN* (IEEE IoT-J 12(3)). GNSS-free satellite-side Doppler pre/post-compensation + a matched-filter detector tolerant to ±0.5·SCS frequency error. PHY only; motivates the "residual error after compensation" that drives detection misses. No access-control adaptation.
- **Daraseliya et al., 2026** — *Reactive access-barring for NB-IoT via ML inference* (Sensors 26(2):636). An XGBoost classifier infers collision count from the received SNR pattern and drives reactive barring, holding preamble success ≈ 0.3 under overload where conventional ACB falls below 0.05. The clearest terrestrial precedent for **inferring the failure cause**, but still a static channel and no NTN state.
- **Dwivedi et al., 2024** — *Efficient transmission for LEO NB-IoT: a data-driven perspective* (arXiv:2406.14107). NB-IoT/NPRACH-specific; Shewhart change-detection + ML payload reduction cut simultaneous transmitters 230→8 and extend battery to 10–15 yr. Names the **bounded visibility window explicitly** (≈4 min @30° mask, ≈246.9 s beam window), the constraint the proposed learner must respect — but solves it by transmitting less, not by learning.
- **Jeong et al., 2025 (MT-eCD)** & **Jeong & Yun, 2026 (ISC-eCD)** — *Early preamble-collision detectors for NTN IoT* (IEEE TVT 74(8); IEEE TAES 62). Multi-threshold and inter-satellite-cooperative detectors that separate a true collision from a single-user miss at **Step 1**, analytically deriving successful-collision-detection, false-alarm and miss-detection probabilities. They establish that **collision vs detection failure is physically observable** at the gNB — the observability the proposed reward relies on — but they are PHY detectors, not controllers, and use no learning.
- **Le et al., 2024** — *Survey on RA protocols in direct-access LEO satellite IoT* (IEEE COMST). Taxonomy of five MAC design challenges (resource limits, high Doppler, sporadic traffic, sparse-data recovery, heterogeneity) and eight optimisation objectives; names ~10-min visibility, differential Doppler (up to 8 kHz vs a 950 Hz supported limit) and timing-advance as recurring open problems; lists online learning-based adaptation for the dynamic topology as unresolved.
- **Lee et al., 2026** — *RA for LEO via deep learning* (arXiv:2512.07053). A lightweight 1D-CNN classifies preamble-collision multiplicity at Step 1 (misdetection <1%, FA <0.1% down to −13 dB) and resolves it probabilistically. Source for the **~10 Hz nominal residual Doppler** (and 250 Hz stress case) used as a channel sanity check; explicitly defers RL-based cross-layer access control to future work.
- **Bui & Pham, 2020** — *DRL-based ACB for energy-efficient mMTC* (IEEE Access). Dueling DQN trading access delay against device listening energy; relevant to a battery-bound meter, but terrestrial and single-knob.
- **"SCMA-Q-learning with overload control for LEO mMTC," 2025** (*Physical Communication* 69:102611). The closest existing *satellite Q-learning* for overload control — but built on SCMA/NOMA, so its state and action spaces differ fundamentally from a Rel-17 NB-IoT ACB/NPRACH deployment.

---

## 4. Research-gap synthesis

Mapping the evidence onto the proposal yields five gaps. The first four are the problem statement's; the fifth is sharpened by the 2026 competitors.

**G1 — Learning-based RACH controllers assume a channel that never moves.**
Clusters A ([1],[2],[3],[12], Bui-Pham, Daraseliya) deliver excellent controllers but validate only on a static terrestrial channel. None ingests elevation, residual Doppler, propagation delay or a pass clock. *Open:* does a Q-learning ACB/BI/NPRACH controller still behave sensibly when the channel is time-varying across a pass?

**G2 — NB-IoT/NTN work stops at the physical/timing layer.**
Cluster B ([5],[6],[7],[11], Lin) models the NTN channel precisely but never lets the *access-control* parameters respond to what the receiver knows. The access layer is left on terrestrial defaults.

**G3 — No scheme separates a detection failure from a contention collision when tuning access parameters.**
Transplanted as-is to NTN, a terrestrial controller reads a Doppler/low-SNR **detection miss** as congestion and raises barring — a false-barring loop that suppresses throughput while doing nothing for the real cause. The collision detectors [18],[19] and the classifier [13] prove the two causes are *observable* at the gNB; Daraseliya infers collisions terrestrially — but **no learning controller puts that separation inside its reward**.

**G4 — Existing RL is not designed for the short LEO visibility window.**
Deep schemes need many stationary episodes: eRACH [15] ~200–500 episodes, Deep Dyna-Q [14] ~400–600 steps, DDQN [3] offline training; Chen [16] shows pretrained eRACH **collapsing to ≈0 throughput at 8 000 UEs** on a fresh cell. A LEO pass lasts **~2–7 min** (≈ hundreds of RAOs). *Open:* a lightweight, fast-converging controller that handles **cold-start vs warm-start** within one pass.

**G5 — Joint ACB + BI + NPRACH control is unoccupied, especially for NB-IoT NTN.**
Prior work tunes one knob or a different one: ACB only [1,3,4,16], RAO/EAB [14], association/backoff [15], channel selection + a single ACB side-knob [17], or NPRACH config alone [12]. **No work co-adapts the Rel-17 NB-IoT triple (ACB barring, Backoff Indicator, NPRACH repetition/CE level) together** — the combination needed because collisions, misses and timeouts each call for a *different* one of the three.

### Gap-coverage matrix (● addressed · ◐ partial · ○ not addressed)

| Capability | TN RL [2,3,12] | TN ML [4] | NTN PHY [5–7,11] | LEO RL [14,15,17] | LEO analytic [16] | **Proposed** |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| Time-varying satellite channel in state | ○ | ○ | ◐ | ◐ | ● | ● |
| NB-IoT NPRACH-specific | ◐ | ● | ● | ○ | ○ | ● |
| Joint ACB + BI + NPRACH action | ○ | ○ | ○ | ○ | ○ | ● |
| Detection-miss vs collision isolation | ○ | ◐ | ◐(PHY) | ○ | ○ | ● |
| Converges within one LEO pass | — | — | — | ○ | ●(zero-shot) | ● |
| Lightweight (tabular, on-board/gateway) | ○ | ◐ | ● | ○ | ● | ● |

The proposed controller is the only entry that is simultaneously NB-IoT-NPRACH-specific, satellite-aware, joint over all three knobs, failure-cause-aware **inside the reward**, pass-length-constrained, and light enough to run tabular at a gateway/on-board.

---

## 5. References

**Journal / conference**

[1] L. Tello-Oquendo *et al.*, "Performance Analysis and Optimal Access Class Barring Parameter Configuration in LTE-A Networks With Massive M2M Traffic," *IEEE Trans. Veh. Technol.*, vol. 67, no. 4, pp. 3505–3520, 2018, doi:10.1109/TVT.2017.2776868.
[2] A.-T. H. Bui and A. T. Pham, "Deep Reinforcement Learning-Based Access Class Barring for Energy-Efficient mMTC Random Access in LTE Networks," *IEEE Access*, 2020.
[3] W. G. Abera, T. O. Olwal, Y. W. Marye, "Deep reinforcement learning-based access class barring for mMTC access control in cellular networks," *Array (Elsevier)*, vol. 28, 100533, 2025, doi:10.1016/j.array.2025.100533.
[4] A. Daraseliya *et al.*, "Performance of the Novel Reactive Access-Barring Scheme for NB-IoT Systems Based on the Machine Learning Inference," *Sensors*, vol. 26, no. 2, art. 636, 2026, doi:10.3390/s26020636.
[5] H. Chougrani, S. Kisseleff, W. A. Martins, S. Chatzinotas, "NB-IoT Random Access for Nonterrestrial Networks: Preamble Detection and Uplink Synchronization," *IEEE Internet Things J.*, vol. 9, no. 16, pp. 14913–14927, 2022, doi:10.1109/JIOT.2021.3123376.
[6] J.-C. Lin, "A Unified Random Access Procedure for an NB-IoT Protocol in Nonterrestrial and Terrestrial Networks," *IEEE Internet Things J.*, vol. 12, no. 3, pp. 3424–3440, 2025, doi:10.1109/JIOT.2024.3481064.
[7] O. Kodheli *et al.*, "Random Access Procedure Over Non-Terrestrial Networks: From Theory to Practice," *IEEE Access*, vol. 9, pp. 109130–109143, 2021, doi:10.1109/ACCESS.2021.3101291.
[8] T. T. T. Le, N. Ul Hassan, X. Chen, M.-S. Alouini, Z. Han, C. Yuen, "A Survey on Random Access Protocols in Direct-Access LEO Satellite-Based IoT Communication," *IEEE Commun. Surveys Tuts.*, 2024, doi:10.1109/COMST.2024.3385347.
[9] "SCMA-Q-Learning with Overload Control for Random Access in LEO Satellite mMTC Networks," *Physical Communication*, vol. 69, art. 102611, 2025.
[10] A. K. Dwivedi, H. Chougrani, S. Chaudhari, N. Varshney, S. Chatzinotas, "Efficient Transmission Scheme for LEO Satellite-Based NB-IoT: A Data-Driven Perspective," *arXiv:2406.14107*, 2024.
[11] B. Shen, Y. Wu, W. Zhang, S. Chatzinotas, B. Ottersten, "LEO Satellite-Enabled Random Access With Large Differential Delay and Doppler Shift," *IEEE Trans. Wireless Commun.*, vol. 24, no. 4, pp. 2876–2893, 2025, doi:10.1109/TWC.2025.3525738.
[12] N. Jiang, Y. Deng, A. Nallanathan, J. A. Chambers, "Reinforcement Learning for Real-Time Optimization in NB-IoT Networks," *IEEE J. Sel. Areas Commun.*, vol. 37, no. 6, 2019, doi:10.1109/JSAC.2019.2904366.
[13] H. Lee, I. P. Roberts, J. Jeong, D. Hong, "Random Access for LEO Satellite Communication Systems via Deep Learning," *arXiv:2512.07053*, 2026.
[14] X. Liu, H. Zhang, K. Long, A. Nallanathan, V. C. M. Leung, "Deep Dyna-Reinforcement Learning Based on Random Access Control in LEO Satellite IoT Networks," *IEEE Internet Things J.*, vol. 9, no. 16, pp. 14818–14827, 2022, doi:10.1109/JIOT.2021.3112907.
[15] J.-H. Lee, H. Seo, J. Park, M. Bennis, Y.-C. Ko, "Learning Emergent Random Access Protocol for LEO Satellite Networks," *IEEE Trans. Wireless Commun.*, vol. 22, no. 1, pp. 257–269, 2023, doi:10.1109/TWC.2022.3192365.
[16] Y.-W. Chen, K.-H. Lin, H.-Y. Wei, "Channel-Aware Access and RACH Enhancement for Efficient LEO Satellite IoT," *IEEE Internet Things J.*, vol. 13, no. 11, pp. 24806–24822, 2026, doi:10.1109/JIOT.2026.3674729.
[17] X. Feng, G. Yin, H. Quan, Z. Li, F. Gong, "Joint Optimization of Access Control and Resource Allocation for LEO Satellite IoT Using Deep Reinforcement Learning," *IEEE Trans. Cogn. Commun. Netw.*, vol. 12, pp. 6670–6684, 2026, doi:10.1109/TCCN.2026.3670194.
[18] J. Jeong, H. Lee, D. Hong, "Enhanced Early Preamble Collision Detector for LEO Satellite-Based NTN IoT Random Access," *IEEE Trans. Veh. Technol.*, vol. 74, no. 8, pp. 12496–12511, 2025, doi:10.1109/TVT.2025.3554531.
[19] J. Jeong, S. Yun, "Intersatellite Cooperation for Early Preamble Collision Detection in NTN IoT Random Access," *IEEE Trans. Aerosp. Electron. Syst.*, vol. 62, pp. 2151–2165, 2026, doi:10.1109/TAES.2025.3634485.

**Standards**

[S1] 3GPP TR 38.811 — *Study on New Radio to support Non-Terrestrial Networks.*
[S2] 3GPP TR 38.821 — *Solutions for NR to support Non-Terrestrial Networks.*
[S3] 3GPP TS 36.211 — *E-UTRA Physical channels and modulation* (NPRACH).
[S4] 3GPP TS 36.321 — *E-UTRA MAC protocol* (Backoff Indicator, Table 7.2-2).
[S5] 3GPP TS 36.331 — *E-UTRA RRC* (Access Class Barring).
[S6] 3GPP TR 37.868 — *Study on RAN improvements for machine-type communications* (Beta(3,4) traffic).
