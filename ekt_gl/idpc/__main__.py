"""Run one EKT-GL-IDPC-DU seed and print a JSON report."""

from argparse import ArgumentParser
from dataclasses import asdict
import json
import math
from pathlib import Path

import torch

from . import CompactIDPCGraph, IDPCEKTGL, IDPCEKTGLConfig


def main() -> None:
    defaults = IDPCEKTGLConfig()
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("graphs", nargs="*", type=Path, help=".idpc or demo JSON files")
    parser.add_argument("--set", choices=("set1", "set2", "all"), help="Original IDPC-DU benchmark group")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--population", type=int, default=defaults.population_size)
    parser.add_argument("--generations", type=int, default=defaults.generations)
    parser.add_argument("--transfer-interval", type=int, default=defaults.transfer_interval)
    parser.add_argument("--elite-count", type=int, default=defaults.elite_count)
    parser.add_argument("--train-epochs", type=int, default=defaults.train_epochs)
    parser.add_argument("--device", default=defaults.device)
    parser.add_argument("--negative-samples", type=int, default=defaults.negative_samples)
    parser.add_argument("--edge-batch-size", type=int, default=defaults.edge_batch_size)
    parser.add_argument("--no-transfer", action="store_true", help="Run independent EA (STO) with the same local search")
    args = parser.parse_args()
    if bool(args.graphs) == bool(args.set):
        parser.error("provide graph files or --set")
    files = args.graphs
    if args.set:
        benchmark = Path("data/benchmark")
        pattern = "set*/*.idpc" if args.set == "all" else f"{args.set}/*.idpc"
        files = sorted(benchmark.glob(pattern))
        if not files:
            parser.error(f"no matching benchmark files in {benchmark}")
    names = [path.stem for path in files]
    if len(set(names)) != len(names):
        parser.error("graph filenames must have distinct stems")
    try:
        config = IDPCEKTGLConfig(
            population_size=args.population, generations=args.generations,
            transfer_interval=args.transfer_interval, elite_count=args.elite_count,
            train_epochs=args.train_epochs, device=args.device,
            negative_samples=args.negative_samples, edge_batch_size=args.edge_batch_size,
        )
    except ValueError as error:
        parser.error(str(error))
    torch.set_num_threads(1)
    tasks = {name: CompactIDPCGraph.from_file(path) for name, path in zip(names, files)}
    result = IDPCEKTGL(config).solve(tasks, seed=args.seed, transfer=not args.no_transfer)
    report = asdict(result)
    # JSON null marks a task/generation for which no feasible path was found.
    for best in report["best"].values():
        if not math.isfinite(best["cost"]):
            best["cost"] = None
    report["history"] = {
        name: [cost if math.isfinite(cost) else None for cost in costs]
        for name, costs in result.history.items()
    }
    report["config"] = asdict(config)
    report["transfer_enabled"] = not args.no_transfer
    report["seed"] = args.seed
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
