"""
main.py — Single command entry point to run all experiments and generate all 6 figures.

Usage:
    python main.py             # Runs experiments and generates figures
    python main.py --plots-only # Generates figures from existing results
"""

import os
import sys
import argparse

from src.config_loader import load_config
from src.experiment import run_all_experiments
from src.plots import generate_all_plots


def main():
    parser = argparse.ArgumentParser(description="NB-IoT NTN Q-Learning RACH Optimization")
    parser.add_argument("--config", type=str, default="config/default.yaml", help="Path to config YAML")
    parser.add_argument("--plots-only", action="store_true", help="Only generate figures from existing results")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory for simulation CSV results")
    parser.add_argument("--figs-dir", type=str, default="figs", help="Directory to save generated figures")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if not args.plots_only:
        print("=" * 70)
        print("Starting Simulation Matrix (R1 to R6 across seeds)...")
        print("=" * 70)
        run_all_experiments(cfg, output_dir=args.results_dir)

    print("=" * 70)
    print("Generating Publication-Quality Research Figures (Fig 1 to Fig 6)...")
    print("=" * 70)
    generate_all_plots(cfg, results_dir=args.results_dir, out_dir=args.figs_dir)
    print("=" * 70)
    print(f"All done! Review your results in {args.results_dir}/ and figures in {args.figs_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
