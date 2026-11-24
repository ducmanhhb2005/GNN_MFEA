"""Separate post-run 2-opt diagnostic; never overwrites primary EKT-GL results."""
import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import scipy
import torch

from ekt_gl import EKTGLConfig
from ekt_gl.routing import two_opt
from run_tsp import BENCHMARKS, PAPER, ROOT, load_tasks, source_digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()
    directory = ROOT / "results" / "tsp"
    destination = directory / "post_2opt"
    destination.mkdir(parents=True, exist_ok=True)
    datasets = {b: load_tasks(b, ROOT / "tsp_data")[0] for b in BENCHMARKS}
    versions = {"python": platform.python_version(), "numpy": np.__version__,
                "torch": torch.__version__, "scipy": scipy.__version__}
    refinement_source = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    records = {}
    for path in sorted((directory / "raw").glob("*.json")):
        original = json.loads(path.read_text(encoding="utf-8"))
        identity = original["identity"]
        if (identity["mode"] != "ektgl" or identity["seed"] >= args.runs
                or identity["config"] != asdict(EKTGLConfig()) or identity["versions"] != versions
                or identity["source_sha256"] != source_digest()):
            continue
        key = (identity["benchmark"], identity["seed"])
        if key in records:
            raise ValueError(f"Duplicate seed: {key}")
        signature = hashlib.sha256(path.read_bytes()).hexdigest()
        target = destination / path.name
        record = json.loads(target.read_text(encoding="utf-8")) if target.exists() else None
        if (record is None or record["input_sha256"] != signature
                or record.get("refinement_source_sha256") != refinement_source):
            start = time.perf_counter()
            solutions = {}
            for task in datasets[identity["benchmark"]]:
                old = original["best"][task.name]
                route = two_opt(task, tuple(old["route"]), 10000)
                cost = task.evaluate(route)
                if cost > old["cost"] + 1e-7:
                    raise ValueError("Postprocessing worsened the tour")
                converged = task.evaluate(two_opt(task, route, 1)) == cost
                solutions[task.name] = {"before_cost": old["cost"], "cost": cost,
                                         "route": route, "converged": converged}
            record = {"identity": identity, "input_sha256": signature,
                      "refinement_source_sha256": refinement_source,
                      "description": "Extra first-improvement 2-opt on final EKT-GL tours; not part of primary run",
                      "max_improvements": 10000, "seconds": time.perf_counter()-start,
                      "best": solutions}
            temporary = target.with_suffix(".tmp")
            temporary.write_text(json.dumps(record, indent=2, allow_nan=False), encoding="utf-8")
            temporary.replace(target)
        records[key] = record
        print(f"Post-2opt {identity['benchmark']}/seed={identity['seed']}: {record['seconds']:.1f}s", flush=True)
    expected = {(b, s) for b in BENCHMARKS for s in range(args.runs)}
    if set(records) != expected:
        print(f"Processed {len(records)}/{len(expected)} EKT-GL seeds; rerun when remaining seeds finish.")
        return
    rows = []
    for benchmark in BENCHMARKS:
        for index, name in enumerate(BENCHMARKS[benchmark]):
            items = [records[benchmark, s]["best"][name] for s in range(args.runs)]
            a = np.asarray([i["before_cost"] for i in items])
            b = np.asarray([i["cost"] for i in items])
            rows.append({"benchmark": benchmark, "task": name, "runs": args.runs,
                         "original_mean": float(a.mean()), "original_std": float(a.std()),
                         "refined_mean": float(b.mean()), "refined_std": float(b.std()),
                         "paper_mean": PAPER[benchmark]["mean"][index],
                         "original_delta_paper_pct": float(100*(a.mean()/PAPER[benchmark]["mean"][index]-1)),
                         "refined_delta_paper_pct": float(100*(b.mean()/PAPER[benchmark]["mean"][index]-1)),
                         "gain_postprocessing_pct": float(100*(1-b.mean()/a.mean())),
                         "best_refined_gap_lkh_pct": float(100*(b.min()/PAPER[benchmark]["lkh"][index]-1)),
                         "all_converged": all(i["converged"] for i in items)})
    (destination / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with (destination / "summary.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("Separate postprocessing diagnostic complete.")


if __name__ == "__main__":
    main()
