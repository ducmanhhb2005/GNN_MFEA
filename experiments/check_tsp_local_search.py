"""Metric and local-search diagnostic; this is NOT an EKT-GL benchmark run.

python -m experiments.check_tsp_local_search
"""
import json
from pathlib import Path
import time

from ekt_gl.routing import two_opt
from run_tsp import BENCHMARKS, ROOT, load_tasks


def main():
    rows = []
    for benchmark in BENCHMARKS:
        tasks, inputs = load_tasks(benchmark, ROOT / "tsp_data")
        for task in tasks:
            start = time.perf_counter()
            route, unvisited = [0], set(task.route_nodes) - {0}
            while unvisited:
                node = min(unvisited, key=lambda n: (float(task._distance[route[-1], n]), n))
                route.append(node)
                unvisited.remove(node)
            initial = tuple(route)
            one = two_opt(task, initial, 1)
            converged = two_opt(task, initial, 10000)
            extra = two_opt(task, converged, 1)
            row = {"benchmark": benchmark, "task": task.name, "input": inputs[task.name],
                   "nearest_neighbor_cost": task.evaluate(initial),
                   "one_improvement_cost": task.evaluate(one),
                   "up_to_10000_improvements_cost": task.evaluate(converged),
                   "converged": task.evaluate(extra) == task.evaluate(converged),
                   "route": converged, "seconds": time.perf_counter() - start}
            rows.append(row)
            print(f"{task.name}: NN={row['nearest_neighbor_cost']:.3f}, "
                  f"1-pass={row['one_improvement_cost']:.3f}, "
                  f"converged={row['up_to_10000_improvements_cost']:.3f}", flush=True)
    output = ROOT / "results" / "tsp" / "local_search_diagnostic.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"description": "Deterministic nearest neighbor from node 0, then "
                                            "first-improvement 2-opt. Not EKT-GL or a paper baseline.",
                                  "max_improvements": 10000, "results": rows}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
