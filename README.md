# Major Project: Q-Learning-Based RACH Parameter Optimization for NB-IoT NTN

This repository contains the complete research artifacts, literature review, methodology, and implementation codebase for the 7th/8th Semester Major Project on **Q-Learning-Based Random Access Channel (RACH) Parameter Optimization for NB-IoT Non-Terrestrial Networks (NTN)**.

---

## 📁 Repository Overview

```text
├── 01_Literature_Review.md      # Comprehensive literature review & 5 research gaps
├── 02_Methodology.md            # Mathematical system model, SGP4 orbit, MDP formulation
├── 03_Implementation_Plan.md     # 8-phase implementation roadmap & acceptance checks
├── _extracted/                  # Extracted text & summaries of 17 foundational IEEE papers
└── nbiot_ntn_qlearning/         # Complete Python simulation codebase & research figures
    ├── config/default.yaml      # Master 3GPP parameters & SGP4 ephemeris
    ├── data/tle/sateliot.tle    # Real commercial Sateliot LEO constellation TLEs
    ├── src/                     # Core simulation modules (orbit, channel, NPRACH, Q-learning)
    ├── results/                 # Simulation metrics CSVs across 10 Monte-Carlo seeds
    ├── figs/                    # All 6 publication-ready research figures (300 DPI)
    ├── tests/                   # Pytest automated validation test suite (5/5 passing)
    ├── main.py                  # Single-command simulation & figure generation entry point
    └── requirements.txt         # Pinned Python package dependencies
```

---

## 🚀 Quick Start (Running the Code)

### 1. Setup Environment
```bash
cd nbiot_ntn_qlearning
python -m pip install -r requirements.txt
```

### 2. Run Acceptance Tests
```bash
python -m pytest tests/test_simulation.py -v
```

### 3. Generate Publication Figures
To re-render all 6 publication figures from the validated simulation datasets:
```bash
python main.py --plots-only
```

---

## 📊 Key Results at a Glance

* **36.4% reduction in preamble transmissions** vs 3GPP Rel-17 NTN default ($1.38$ vs $2.17$ attempts/device), directly saving IoT battery life.
* **+53.8% access success gain** under 3,000-device smart-metering overload under Doppler stress ($250$ Hz).
* **Elimination of the false-barring loop** via cause-separated reward formulation isolating collisions from detection misses.
* **Convergence within an 8.17-minute LEO pass** via compact tabular MDP ($|S|=72, |A|=60$) and warm-start initialization.
