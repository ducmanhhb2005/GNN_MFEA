"""Compare EKT-GL, STO, independent online learning and MFEA on IDPC-DU."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import sys
from tempfile import NamedTemporaryFile
from time import perf_counter

import numpy as np
import torch

from ekt_gl.idpc import CompactIDPCGraph, IDPCEKTGL, IDPCEKTGLConfig
from experiments._support.ektgl_idpc_report import write_reports
from experiments.run_mfea import build_groups, json_safe, load_reference, sha256
from idpc_du.validation import validate_path
from mfea.solver import solve_mfea


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("ektgl", "sto", "individual", "mfea")


def _write_json(path, payload):
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False) as output:
        temporary = Path(output.name)
        json.dump(payload, output, allow_nan=False, separators=(",", ":"))
        output.write("\n")
    temporary.replace(path)


def _identity(paths, config, mfea_config, groups, threads):
    files = [Path(__file__), Path(__file__).parent / "_support/ektgl_idpc_report.py",
             Path(__file__).parent / "run_mfea.py", ROOT / "ekt_gl/network.py", ROOT / "ekt_gl/config.py"]
    for folder in ("idpc_du", "mfea", "ekt_gl/idpc"):
        files.extend((ROOT / folder).glob("*.py"))
    return dict(
        tasks=[dict(name=path.stem, path=str(path), sha256=sha256(path)) for path in paths],
        ektgl_config=asdict(config),
        mfea_config=mfea_config,
        mfea_groups=groups,
        torch_threads=threads,
        environment=dict(python=sys.version, platform=platform.platform(),
                         torch=torch.__version__, numpy=np.__version__),
        sources={str(path.relative_to(ROOT)): sha256(path) for path in sorted(files)},
    )


def _graph_rows(result):
    rows = {}
    for task, solution in result.best.items():
        outcomes = Counter(report.outcomes[task] for report in result.transfers)
        rows[task] = dict(
            cost=solution.cost, edge_ids=solution.edge_ids,
            history=result.history[task], evaluation_history=result.evaluation_history[task],
            evaluations=result.evaluations[task], local_search_checks=result.local_search_checks[task],
            transfer_outcomes=dict(outcomes),
        )
    return rows


def execute(method, graphs, config, mfea_config, groups, seed, progress):
    training_seconds = inference_seconds = 0.0
    transfers = []
    rows = {}
    rng = random.Random(seed)
    population_seeds = {name: rng.getrandbits(64) for name in graphs}
    start = perf_counter()
    if method == "mfea":
        for index_group, group in enumerate(groups, start=1):
            diagnostics = {}
            best, history = solve_mfea(
                [graphs[name] for name in group], seed=seed, diagnostics=diagnostics, **mfea_config,
            )
            for index, name in enumerate(group):
                cost, path = best[index]
                rows[name] = dict(
                    cost=cost, edge_ids=path, history=[entry[index] for entry in history],
                    evaluation_history=[entry[index] for entry in diagnostics["evaluation_history"]],
                    evaluations=diagnostics["evaluations"][index], local_search_checks=0,
                    transfer_outcomes={},
                )
            if index_group % 5 == 0 or index_group == len(groups):
                print(f"  mfea/{seed}: groups {index_group}/{len(groups)}", flush=True)
    else:
        solver = IDPCEKTGL(config)
        task_groups = [{name: graph} for name, graph in graphs.items()] if method == "individual" else [graphs]
        for index_group, tasks in enumerate(task_groups, start=1):
            result = solver.solve(
                tasks, seed=seed, transfer=method != "sto",
                population_seeds={name: population_seeds[name] for name in tasks},
                progress=progress if len(task_groups) == 1 else None,
            )
            rows.update(_graph_rows(result))
            for report in result.transfers:
                training_seconds += report.training_seconds
                inference_seconds += report.inference_seconds
                transfers.append(asdict(report))
            if method == "individual" and (index_group % 10 == 0 or index_group == len(task_groups)):
                print(f"  individual/{seed}: tasks {index_group}/{len(task_groups)}", flush=True)
    return json_safe(dict(
        method=method, seed=seed, tasks=rows, transfers=transfers,
        timing=dict(solve_seconds=perf_counter()-start, training_seconds=training_seconds,
                    inference_decode_seconds=inference_seconds),
    ))


def validate_record(record, graphs, method, seed, generations, fingerprint):
    """Recompute path cost and check the best-so-far trace before accepting a run."""
    if (record["method"], record["seed"], record["fingerprint"]) != (method, seed, fingerprint):
        raise ValueError(f"Checkpoint identity mismatch for {method}/{seed}")
    if record["tasks"].keys() != graphs.keys():
        raise ValueError(f"Task set mismatch for {method}/{seed}")
    for name, row in record["tasks"].items():
        path, reported = row["edge_ids"], row["cost"]
        cost = validate_path(graphs[name], path)
        if reported is None:
            valid = not path and cost == math.inf
        else:
            valid = math.isfinite(reported) and math.isclose(cost, reported, rel_tol=1e-12)
        history, counts = row["history"], row["evaluation_history"]
        valid = (
            valid and len(history) == len(counts) == generations + 1
            and history[-1] == reported and counts[-1] == row["evaluations"]
            and all(0 < a <= b for a, b in zip(counts, counts[1:]))
            and row["local_search_checks"] >= 0
        )
        costs = [math.inf if value is None else value for value in history]
        if not valid or any(a < b for a, b in zip(costs, costs[1:])):
            raise ValueError(f"Invalid path or trace for {method}/{seed}/{name}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    data = parser.add_mutually_exclusive_group()
    data.add_argument("--set", choices=("set1", "set2", "all"))
    data.add_argument("--instances", nargs="+", type=Path)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--population", type=int, default=30)
    parser.add_argument("--generations", type=int, default=300)
    parser.add_argument("--transfer-interval", type=int, default=10)
    parser.add_argument("--elite-count", type=int, default=5)
    parser.add_argument("--train-epochs", type=int, default=1)
    parser.add_argument("--mfea-population", type=int, default=100)
    parser.add_argument("--mfea-generations", type=int, default=500)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--cache-size", type=int, default=2048, help="Exact memoization entries per task")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results/ektgl_idpc")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if min(args.runs, args.threads, args.mfea_generations) < 1 or args.mfea_population < 4:
        parser.error("runs, threads and generations must be positive, MFEA population must be >= 4")
    if len(set(args.methods)) != len(args.methods):
        parser.error("methods must be unique")
    try:
        config = IDPCEKTGLConfig(
            population_size=args.population, generations=args.generations,
            transfer_interval=args.transfer_interval, elite_count=args.elite_count,
            train_epochs=args.train_epochs, device=args.device, cache_size=args.cache_size,
        )
    except ValueError as error:
        parser.error(str(error))

    if args.instances:
        paths = [path.resolve() for path in args.instances]
        grouping = "explicit"
    else:
        subset = args.set or "all"
        paths = sorted(
            ROOT / "data/benchmark" / values["set"] / name
            for name, values in load_reference()["instances"].items()
            if subset == "all" or values["set"] == subset
        )
        grouping = "adjacent"
    if any(not path.is_file() for path in paths) or len({path.stem for path in paths}) != len(paths):
        parser.error("provide existing files with distinct stems")
    groups = [[path.stem for path in group] for group in build_groups(paths, grouping)]
    mfea_config = dict(population=args.mfea_population, generations=args.mfea_generations,
                       rmp=0.5, mutation_rate=0.05, cache_size=args.cache_size)
    torch.set_num_threads(args.threads)

    output = args.out_dir.resolve()
    identity = _identity(paths, config, mfea_config, groups, args.threads)
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "manifest.json"
    if manifest.exists():
        if not args.resume or json.loads(manifest.read_text(encoding="utf-8"))["fingerprint"] != fingerprint:
            parser.error("existing output requires --resume and matching data, code and configuration")
    else:
        if any(output.iterdir()):
            parser.error("select an empty output directory")
        _write_json(manifest, dict(fingerprint=fingerprint, **identity))

    checkpoint_dir = output / "runs"
    checkpoint_dir.mkdir(exist_ok=True)
    print(f"Loading {len(paths)} IDPC-DU tasks", flush=True)
    graphs = {path.stem: CompactIDPCGraph.from_file(path) for path in paths}
    records = []
    for seed in range(args.seed, args.seed + args.runs):
        for method in args.methods:
            checkpoint = checkpoint_dir / f"{method}_{seed}.json"
            generations = args.mfea_generations if method == "mfea" else config.generations
            if checkpoint.exists():
                record = json.loads(checkpoint.read_text(encoding="utf-8"))
                status = "verified checkpoint"
            else:
                print(f"Running {method}, seed={seed}", flush=True)

                def progress(generation):
                    if generation % 50 == 0:
                        print(f"  {method}/{seed}: generation {generation}/{generations}", flush=True)

                record = execute(method, graphs, config, mfea_config, groups, seed, progress)
                record["fingerprint"] = fingerprint
                status = f"{record['timing']['solve_seconds']:.2f}s"
                validate_record(record, graphs, method, seed, generations, fingerprint)
                _write_json(checkpoint, record)
            if status == "verified checkpoint":
                validate_record(record, graphs, method, seed, generations, fingerprint)
            records.append(record)
            # Rebuild summaries from completed runs on resume, not from old CSV rows.
            print(f"Finished {method}, seed={seed}: {status}", flush=True)
        write_reports(output, records)
    print(f"Report: {output / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
