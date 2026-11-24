"""Recompute saved tour lengths independently from raw TSPLIB coordinates."""
import argparse
from dataclasses import asdict
import hashlib
import json
import platform

import numpy as np
import scipy
import torch

from ekt_gl import EKTGLConfig
from run_tsp import BENCHMARKS, ROOT, load_tasks, source_digest, validate_record


def raw_coordinates_float64(path):
    """Independent read without the solver runner's float32 coordinate parser."""
    section = path.read_text(encoding="utf-8").split("NODE_COORD_SECTION", 1)[1]
    rows = {}
    for line in section.splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "EOF":
            break
        node, x, y = parts
        rows[int(node)] = (float(x), float(y))
    return np.asarray([rows[node] for node in sorted(rows)], dtype=np.float64)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--generations", type=int, default=300)
    arguments = parser.parse_args()
    config = EKTGLConfig(generations=arguments.generations)
    directory = ROOT / "results" / "tsp"
    datasets = {b: load_tasks(b, ROOT / "tsp_data") for b in BENCHMARKS}
    digest = source_digest()
    versions = {"python": platform.python_version(), "numpy": np.__version__,
                "torch": torch.__version__, "scipy": scipy.__version__}
    records = {}
    errors = []
    post_errors = []
    for path in sorted((directory / "raw").glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        identity = record["identity"]
        if (identity["source_sha256"] != digest or identity["config"] != asdict(config)
                or identity["versions"] != versions
                or identity["seed"] >= arguments.runs):
            continue
        benchmark = identity["benchmark"]
        key = (benchmark, identity["mode"], identity["seed"])
        if key in records:
            raise ValueError(f"Duplicate seed: {key}")
        tasks, inputs = datasets[benchmark]
        if inputs != identity["inputs"]:
            raise ValueError(f"Input manifest mismatch: {path}")
        validate_record(record, tasks, config)
        post_path = directory / "post_2opt" / path.name
        post = None
        if identity["mode"] == "ektgl" and post_path.exists():
            post = json.loads(post_path.read_text(encoding="utf-8"))
            if (post["identity"] != identity
                    or post["input_sha256"] != hashlib.sha256(path.read_bytes()).hexdigest()):
                raise ValueError(f"Mismatched postprocessing source: {post_path}")
        for task in tasks:
            raw = raw_coordinates_float64(ROOT / "tsp_data" / f"{task.name}.tsp")
            route = np.asarray(record["best"][task.name]["route"])
            # Do not use TSPTask's normalization, distance matrix or evaluate().
            edges = raw[route] - raw[np.roll(route, -1)]
            cost = np.sqrt(np.square(edges).sum(axis=1)).sum() / (raw.max() - raw.min())
            error = abs(float(cost) - record["best"][task.name]["cost"])
            if error > 5e-5:
                raise ValueError(f"Independent tour cost mismatch: {path}, {task.name}, {error}")
            errors.append(error)
            if post is not None:
                solution = post["best"][task.name]
                task.validate_route(tuple(solution["route"]))
                refined = np.asarray(solution["route"])
                edges = raw[refined] - raw[np.roll(refined, -1)]
                refined_cost = np.sqrt(np.square(edges).sum(axis=1)).sum() / (raw.max()-raw.min())
                post_error = abs(float(refined_cost)-solution["cost"])
                if (post_error > 5e-5 or solution["cost"] > record["best"][task.name]["cost"]+1e-7
                        or solution["before_cost"] != record["best"][task.name]["cost"]
                        or not solution["converged"]):
                    raise ValueError(f"Invalid postprocessing: {post_path}, {task.name}")
                post_errors.append(post_error)
        records[key] = str(path.relative_to(ROOT))
    expected = {(b, m, s) for b in BENCHMARKS for m in ("ektgl", "sto") for s in range(arguments.runs)}
    if set(records) != expected:
        raise ValueError(f"Missing runs: {sorted(expected-set(records))}")
    if post_errors and len(post_errors) != 18 * arguments.runs:
        raise ValueError("Incomplete postprocessing; rerun experiments.refine_tsp_results")
    report = {"passed": True, "runs_per_mode_per_benchmark": arguments.runs,
              "generations": arguments.generations, "seed_files": len(records),
              "validated_tours": len(errors), "max_independent_cost_error": max(errors),
              "post_validated_tours": len(post_errors),
              "post_max_independent_cost_error": max(post_errors, default=0),
              "cost_tolerance": 5e-5, "source_sha256": digest,
              "versions": versions,
              "summary_sha256": hashlib.sha256((directory / "summary.json").read_bytes()).hexdigest(),
              "files": list(records.values())}
    post_summary = directory / "post_2opt" / "summary.json"
    if post_summary.exists():
        report["post_summary_sha256"] = hashlib.sha256(post_summary.read_bytes()).hexdigest()
    (directory / "audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
