# Assumptions Register

Every assumption made during the review and the planned fixes is listed here.

**Status values:**
- **Proposed**: my default. It applies unless you object.
- **Asked**: waiting for your answer.
- **Confirmed**: you agreed.
- **Rejected**: you said no.

Each entry says what changes if the assumption is wrong.

## A. Review assumptions (already used in `REVIEW_ISSUES.md`)

| ID | Assumption | Basis | Status |
|---|---|---|---|
| R-1 | The `.md` files are the master copies of the literature review, methodology, and plan. The `.docx` files are exports of them. | The text in each `.docx` matches the start of its `.md`. | Proposed |
| R-2 | `_extracted/*.txt` is a faithful text extraction of the PDFs in `papers/`. | The file names match. Spot checks found the quoted numbers. | Proposed |
| R-3 | `Project_Audit.pdf` (11 Oct) is an earlier review. Items it raised that are still visible in the files count as open. | Content comparison. | Proposed |
| R-4 | The README bullet claims (36.4 %, +53.8 %, 21.8 % vs 40.0 %) come from an older run. | They are absent from every current CSV. | Proposed |

## B. Modelling assumptions for the fixes

| ID | Assumption | Basis | If wrong |
|---|---|---|---|
| A-1 | **Confirmed (Q1).** Most UEs have GNSS and apply Rel-17 TA/Doppler pre-compensation, with the RAR window offset by the RTT, so they don't time out. A fraction f_deg (default 10 %, swept 0–30 %) has a stale GNSS/ephemeris fix. Their residual timing error grows with the differential delay toward the horizon. When it exceeds the CP/window tolerance, the attempt counts as a **timeout**. | TS 36.321 / TS 36.331 Rel-17 NTN. TR 38.821 §6.3. Kodheli [7] (timer failures). Chougrani [5] (differential RTD). | f_deg and the error model are sensitivity parameters. |
| A-1b | **Confirmed (Q2).** Two detection regimes: (i) GNSS-free, fitted to Chougrani's 1 % miss points (32 reps: 7.9–15.5 dB; 64 reps: 2.8–9.3 dB); (ii) GNSS-assisted, with the same curve shape shifted by a documented offset. The Doppler state bin is removed. The stress axis is the coupling-loss severity. | [5], [13]. | Values are documented in the paper's parameter table. |
| A-1c | **Confirmed (Q3).** Per-device coupling loss and RSRP-based CE0/1/2 selection, with per-CE subcarrier pools. The action is (p, BI, repetition profile). | TS 36.331 NPRACH-ConfigSIB-NB. | — |
| A-1d | **Confirmed (Q5).** The deliverable is an IEEEtran conference LaTeX draft. Venue unknown, so the default is a 6-page IEEE conference format. | — | The format is adjusted once you name the venue. |
| A-2 | Collided meters on the same subcarrier all fail at Msg4. There is no capture effect. | A common conservative simplification (Tello [1], Liu [14]). | Collision losses are slightly overstated. Stated in the limitations. |
| A-3 | The gNB knows whether a subcarrier collided, through an early collision detector with P_cd = 0.95 and false alarm 0.01. It does **not** know how many devices collided. | Jeong et al. [18], [19]. Lee [13]. | If the detector is absent, collisions are inferred only from Msg4 failures, one RAO later. |
| A-4 | Misses are not directly observable. They are estimated as (estimated attempts − detected − collided × 2) clipped at ≥ 0, where attempts come from the Poisson inversion of idle subcarriers. | Standard ALOHA load estimation. | The reward gets noisier, but stays honest. |
| A-5 | NB-IoT has no probabilistic ACB factor. p is modelled as the fraction of the 10 access classes left unbarred by the AB bitmap (SIB14-NB). The barring wait is (0.7 + 0.6U) × 4 s, and the check runs only at RA initiation. | TS 36.331 AB-Config-NB. LTE ACB timer formula. | If you prefer plain "ACB as in LTE-M", only the wording changes. |
| A-6 | The 48 NPRACH subcarriers are split across CE levels as 24/12/12, and each CE level has its own repetition count. The agent picks a repetition **profile** (low/mid/high) that scales all three. | TS 36.331 NPRACH-ConfigSIB-NB. 3 CE levels. | It could fall back to the current single global N_rep (Q3). |
| A-7 | Per-meter extra coupling loss = building penetration ~ N(10, 4) dB clipped to [0, 25] dB, plus log-normal shadowing σ = 4 dB (LOS-dominant suburban, TR 38.811 Table 6.6.2-1). | AMI meters are often indoors or in basements. | The SNR spread changes. This is a sensitivity parameter. |
| A-8 | Link-budget extras: 3 dB polarisation, 0.1 dB atmospheric, 2.2 dB scintillation, 3 dB shadow margin. Satellite G/T is derived from a 6 dBi antenna gain, a 5 dB noise figure, and 290 K. | TR 38.821 Table 6.1.1.1-1 style. | Absolute SNR shifts by a few dB. This is documented. |
| A-9 | The altitude sweep uses synthetic circular-orbit TLEs at 500/600/800 km with the same inclination (97.67°). The ground site is unchanged. | No real NB-IoT LEO satellites exist at all three altitudes. | Labelled "synthetic" in the paper. |
| A-10 | Test geometry is every SATELIOT_2 pass over Ahmedabad with peak elevation ≥ 20° within 7 days of the TLE epoch. Warm-start training passes come from SATELIOT_1/3/4. | SGP4 accuracy degrades after ~1 week. Keeps training and test separate. | It could also add a second ground site. |
| A-11 | Traffic: periodic AMI reports as Poisson arrivals over the whole pass at a rate λ = N / T_pass, plus one Beta(3,4) burst of 30 % of N over 10 s at a uniformly random time. | TR 37.868 traffic models 1 and 2. | It could fall back to pure Poisson or pure Beta. |
| A-12 | Access-success deadline = 10 s, because AMI alarm reporting is delay-tolerant but bounded. The 45 s deadline is also reported. | There is no standard deadline. 45 s did not separate the schemes. | The deadline is a sweep parameter. |
| A-13 | B2's static NTN values are kept (p = 0.7, BI = 1024 ms, mid-repetition profile) and renamed. They are not presented as a 3GPP default. | No 3GPP preset exists. | Only the wording changes. |
| A-14 | Q-learning hyperparameters (α = 0.12, γ = 0.9, ε decay 0.995) stay fixed. There is no per-scenario tuning, to avoid cherry-picking. | Jiang [12] precedent. | A small, documented grid could be run if the results are unstable. |
| A-15 | Statistics: 95 % CI = mean ± t₀.₉₇₅ × s / √n over (pass, seed) runs. Significance uses a paired Wilcoxon signed-rank test with Holm correction. | Standard. | None. |
| A-16 | Reference [9] is Wu, Jing, Ding, Zhao, *Phys. Commun.* vol. 69, art. 102584 (2024/25), doi 10.1016/j.phycom.2024.102584. | The literature review and the earlier audit agree. To be confirmed online. | The citation is fixed after checking. |
| A-17 | arXiv 2512.07053 (Lee et al.) is dated 2025. | The arXiv ID encodes Dec 2025. | Changes to 2026 if there is a later journal version. |

## C. Process assumptions

| ID | Assumption |
|---|---|
| P-1 | I do not edit your original problem-statement `.docx`. Corrections go into an errata section. |
| P-2 | Old results and figures are archived, never deleted. |
| P-3 | No result is chosen or tuned to look favourable. Every variant that is tried gets reported. |
| P-4 | Creating a git repository locally is fine. Nothing is pushed to GitHub without your OK. |
