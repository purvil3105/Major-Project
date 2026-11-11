# Problem Statement — Errata (11 Nov 2026)

These corrections apply to `20260928_NBIoT_NTN_RACH_ProblemStatement.docx`. The original is left unchanged as submitted.

| # | Original text | Correction | Basis |
|---|---|---|---|
| E1 | "usable window of roughly two to seven minutes" | Above a 10° mask, a ~580 km LEO pass lasts **≈6–8.5 min**, depending on peak elevation. The 14 simulated passes peak at 20–86° and last 364–490 s. A ~4 min pass corresponds to a 30° mask (Dwivedi [10]). | SGP4; [10] |
| E2 | "expected to show a U-shape: poor near the horizon, good near zenith" | Plotted against **elevation**, success rises with elevation. Plotted against **time**, it is ∩-shaped. "U-shape" is the wrong term. | Fig. 2 |
| E3 | Residual Doppler of ~10 Hz nominal and 250 Hz stress drives detection misses | Relative to the 3.75 kHz NPRACH subcarrier spacing, these residuals cost < 0.1 dB, and Lee [13] reports stable detection up to 250 Hz. Misses are driven by link budget and repetitions, so residual Doppler is not used as a stress axis. | [13]; 3GPP R1-2100601 |
| E4 | Timeout when the response arrives after the RAR window | In Rel-17 NTN the RAR window start is offset by the UE–gNB RTT, so a correctly synchronised UE does not time out. Timeouts are modelled for meters with a stale GNSS/ephemeris fix, whose timing error exceeds the CP. | TS 36.321 Rel-17; [5] |
| E5 | "ACB barring factor" for NB-IoT | NB-IoT (Rel-13+) uses access barring with an access-class bitmap (SIB14-NB), not a probabilistic barring factor. The factor p is used here as the fraction of unbarred access classes. | TS 36.331 |
| E6 | "Rel-17-style NPRACH-NTN default configuration" as a baseline | 3GPP defines no default access-control preset. The baseline is renamed "static NTN-aware configuration (B2)". A genie best-fixed configuration (B5) is added as the honest static reference. | — |
| E7 | "oracle controller with perfect channel knowledge as an honest upper bound" | A true per-RAO oracle is not computable for this problem. It is replaced by B5, the best of all 60 fixed configurations found by exhaustive search on the test passes. | — |
| E8 | Ref. [1] "J. Tello-Oquendo" | **L.** Tello-Oquendo | [1] |
| E9 | Ref. [12] "learning-automata Q-learning" | "LA" means **linear-approximation** Q-learning | [12] |
| E10 | Ref. [9] "vol. 69, article 102611, 2025" | Z. Wu, G. Jing, J. Ding, X. Zhao, *Phys. Commun.* vol. 69, art. **102584**, 2025, doi 10.1016/j.phycom.2024.102584 | Crossref |
| E11 | Ref. [13] "arXiv:2512.07053, 2026" | arXiv preprint dated **Dec. 2025** | arXiv ID |
| E12 | Sections 7.1–7.3 are cross-referenced but absent from the extracted text | Check the final document for the missing section. | — |
