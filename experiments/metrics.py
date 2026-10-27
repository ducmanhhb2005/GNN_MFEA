"""Paper Tables I/II metrics and reference [4] convergence normalization.

Reporting only: nothing in this module is imported by the optimizer.
"""
from collections import defaultdict
import math
import statistics


def comparison_metrics(summaries, reference=None):
    result = []
    for subset in sorted({r["subset"] for r in summaries}):
        grouped = defaultdict(dict)
        for row in summaries:
            if row["subset"] == subset:
                grouped[row["instance"]][row["algorithm"]] = row
        pairs = [p for p in grouped.values() if "EA" in p and "MFEA" in p]
        if not pairs:
            continue
        comparable = [p for p in pairs if all(math.isfinite(p[a]["AVG"]) for a in ("EA", "MFEA"))]
        equal = lambda p: math.isclose(p["EA"]["AVG"], p["MFEA"]["AVG"], rel_tol=1e-12, abs_tol=1e-9)
        rpds = [100 * (p["EA"]["AVG"] - p["MFEA"]["AVG"]) / p["EA"]["AVG"]
                for p in comparable if p["EA"]["AVG"] > 0]
        improved_rpds = [100 * (p["EA"]["AVG"] - p["MFEA"]["AVG"]) / p["EA"]["AVG"]
                         for p in comparable if p["EA"]["AVG"] > 0 and not equal(p)
                         and p["MFEA"]["AVG"] < p["EA"]["AVG"]]
        row = dict(subset=subset, paired=len(pairs), finite_pairs=len(comparable),
                   NIB=sum(not equal(p) and p["MFEA"]["AVG"] < p["EA"]["AVG"] for p in comparable),
                   NIE=sum(equal(p) for p in comparable),
                   NIW=sum(not equal(p) and p["MFEA"]["AVG"] > p["EA"]["AVG"] for p in comparable),
                   rpd_pairs=len(rpds), maximum_RPD=max(rpds, default=None),
                   average_RPD=statistics.mean(rpds) if rpds else None,
                   average_RPD_improved=statistics.mean(improved_rpds) if improved_rpds else None)
        paper = (reference or {}).get("table_II", {}).get(subset, {})
        for key in ("NIB", "NIE", "maximum_RPD", "average_RPD"):
            row["paper_" + key] = paper.get(key)
        result.append(row)
    return result


def normalized_convergence(rows, generations):
    """Normalize with extrema of ALL evaluated finite objectives, EA and MFEA.

    Gupta/Ong/Feng [4], page 10: (f - f_min) / (f_max - f_min), then
    average over tasks and independent runs. Extrema never enter the search.
    A constant task maps to zero; an infeasible best maps to None and is NOT
    silently dropped from an average. Nonfinite candidate evaluations are
    recorded separately, since finite min-max normalization cannot scale inf.
    """
    by_task = defaultdict(list)
    by_run = {}
    for row in rows:
        key = (row["algorithm"], row["instance"], row["seed"])
        if key in by_run:
            raise ValueError("Duplicate convergence record")
        if len(row["convergence"]) != generations + 1:
            raise ValueError("Wrong convergence length")
        by_run[key] = row
        by_task[row["instance"]].append(row)
    tasks = sorted(by_task)
    algorithms = ("EA", "MFEA")
    seeds = sorted({r["seed"] for r in rows})
    expected = {(a, t, s) for a in algorithms for t in tasks for s in seeds}
    if set(by_run) != expected:
        raise ValueError("Convergence requires both algorithms on every task and seed")
    extrema = {}
    for task, records in by_task.items():
        low = [r["objective_min"] for r in records if r["objective_min"] is not None]
        high = [r["objective_max"] for r in records if r["objective_max"] is not None]
        if not low or not high:
            raise ValueError(f"No finite evaluated objectives for {task}")
        extrema[task] = dict(minimum=min(low), maximum=max(high),
                             constant=min(low) == max(high),
                             evaluations=sum(r["evaluations"] for r in records),
                             infeasible_evaluations=sum(r["infeasible_evaluations"] for r in records))
    def normalize(task, value):
        if value is None or not math.isfinite(value):
            return None
        low, high = extrema[task]["minimum"], extrema[task]["maximum"]
        if not low <= value <= high:
            raise ValueError("Objective outside recorded evaluation extrema")
        return (value - low) / (high - low) if high != low else 0.0
    individual, averaged = [], []
    for algorithm in algorithms:
        for generation in range(generations + 1):
            per_task = {}
            for task in tasks:
                values = [normalize(task, by_run[(algorithm, task, s)]["convergence"][generation]) for s in seeds]
                per_task[task] = values
                individual.append(dict(algorithm=algorithm, instance=task, generation=generation,
                    normalized_mean=statistics.mean(values) if all(v is not None for v in values) else None,
                    valid_runs=sum(v is not None for v in values), total_runs=len(seeds)))
            per_seed = []
            for index, seed in enumerate(seeds):
                values = [per_task[t][index] for t in tasks]
                per_seed.append(statistics.mean(values) if all(v is not None for v in values) else None)
            averaged.append(dict(algorithm=algorithm, generation=generation,
                normalized_mean=statistics.mean(per_seed) if all(v is not None for v in per_seed) else None,
                valid_runs=sum(v is not None for v in per_seed), total_runs=len(seeds), tasks=len(tasks)))
    return extrema, individual, averaged
