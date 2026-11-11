"""
main.py — reproduce every result and figure.

    python main.py                 # full run: pre-training, B5 search, evaluation, sweeps, figures
    python main.py --quick         # smoke test on a reduced matrix (results_quick/)
    python main.py --plots-only    # re-draw figures and tables from existing results/
"""

import argparse

from src.config_loader import load_config


def main():
    ap = argparse.ArgumentParser(description="NB-IoT NTN Q-learning RACH controller")
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--figs-dir", default=None)
    ap.add_argument("--plots-only", action="store_true")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    res = args.results_dir or ("results_quick" if args.quick else "results")
    figs = args.figs_dir or ("figs_quick" if args.quick else "figs")

    if not args.plots_only:
        from src.experiment import run_all_experiments
        run_all_experiments(args.config, output_dir=res, quick=args.quick)

    from src.plots import generate_all_plots
    from src.stats import write_tables
    cfg = load_config(args.config)
    write_tables(res)
    generate_all_plots(cfg, results_dir=res, out_dir=figs)


if __name__ == "__main__":
    main()
