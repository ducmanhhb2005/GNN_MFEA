"""Run the full CEC 2020 experiment: Tables II-IV and Figures 7-8.

Original data and four paper components only. EA is the single-task control.
Results are measured independently; full coverage is not a numerical-match claim.
"""
import argparse
import importlib.util
import json
from pathlib import Path

from experiments.run_mfea import ROOT, load_reference, main as run_experiment
from experiments._support.validate_artifacts import audit_outputs
from experiments._support.paper_report import build_paper_report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT/"experiments"/"paper_run")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--population", type=int, default=100)
    parser.add_argument("--generations", type=int, default=500)
    parser.add_argument("--rmp", type=float, default=.5)
    parser.add_argument("--mutation-rate", type=float, default=.05)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-plots", action="store_true", help="Export numerical results without optional matplotlib figures")
    args = parser.parse_args(argv)
    if not args.no_plots and importlib.util.find_spec("matplotlib") is None:
        parser.error("Figure export requires: python -m pip install -r requirements-plots.txt (or use --no-plots)")
    output = args.out_dir.resolve()
    common = ["--methods", "ea", "mfea"]
    for option in ("workers", "runs", "seed", "population", "generations", "rmp", "mutation_rate"):
        common += ["--"+option.replace("_", "-"), str(getattr(args, option))]
    if args.resume:
        common += ["--resume"]
    reference = load_reference()
    pair = [ROOT/"data"/"benchmark"/reference["instances"][name]["set"]/name for name in reference["convergence"]["instances"]]
    print("[1/4] EA/MFEA on all 42 original instances", flush=True)
    run_experiment(common+["--set", "all", "--out-dir", str(output/"benchmark")])
    print("[2/4] Published convergence task pair, with EA controls", flush=True)
    run_experiment(common+["--instances", *map(str, pair), "--out-dir", str(output/"convergence")])
    print("[3/4] Independently audit paths, traces, evaluation budgets, statistics and hashes", flush=True)
    validation = audit_outputs([output/"benchmark", output/"convergence"])
    (output/"validation.json").write_text(json.dumps(validation, indent=2)+"\n", encoding="utf-8")
    print("[4/4] Export Tables II-IV, normalized convergence, and Figures 7-8", flush=True)
    metrics = build_paper_report(output, make_plots=not args.no_plots)
    print("Finished. BF/AVG matches:", metrics["matching_BF_AVG_rows"], flush=True)
    print("Full report:", output/"REPORT.md", flush=True)


if __name__ == "__main__":
    main()
