"""Summarize matched IDPC-DU runs without excluding infeasible seeds."""

from __future__ import annotations

from bisect import bisect_right
from collections import Counter, defaultdict
import csv
from pathlib import Path
from statistics import mean, stdev


def _cost_stats(costs):
    finite = [cost for cost in costs if cost is not None]
    complete = len(finite) == len(costs)
    return {
        "BF": min(finite, default=None),
        "AVG": mean(finite) if complete else None,
        "STD_sample": stdev(finite) if complete and len(finite) > 1 else None,
        "feasible_runs": len(finite),
        "feasible_rate": len(finite) / len(costs),
    }


def _write_csv(path: Path, rows: list[dict], columns=None) -> None:
    if not rows and columns is None:
        return
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=columns or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _winner(ektgl_cost, control_cost):
    if ektgl_cost is None and control_cost is None:
        return "both_infeasible"
    if control_cost is None or (ektgl_cost is not None and ektgl_cost < control_cost):
        return "ektgl"
    if ektgl_cost is None or ektgl_cost > control_cost:
        return "control"
    return "tie"


def matched_budgets(records: list[dict]) -> list[dict]:
    rows = []
    by_seed = defaultdict(dict)
    for record in records:
        by_seed[record["seed"]][record["method"]] = record["tasks"]
    for seed, methods in sorted(by_seed.items()):
        if "ektgl" not in methods:
            continue
        for method, tasks in methods.items():
            if method == "ektgl":
                continue
            for task, control in tasks.items():
                ektgl = methods["ektgl"][task]
                shared = min(ektgl["evaluations"], control["evaluations"])
                ektgl_gen = bisect_right(ektgl["evaluation_history"], shared) - 1
                control_gen = bisect_right(control["evaluation_history"], shared) - 1
                ektgl_cost = ektgl["history"][ektgl_gen] if ektgl_gen >= 0 else None
                control_cost = control["history"][control_gen] if control_gen >= 0 else None
                rows.append(dict(
                    seed=seed, task=task, control=method, budget=shared,
                    ektgl_evaluations=ektgl["evaluation_history"][ektgl_gen] if ektgl_gen >= 0 else None,
                    control_evaluations=control["evaluation_history"][control_gen] if control_gen >= 0 else None,
                    ektgl_cost=ektgl_cost, control_cost=control_cost,
                    budget_winner=(
                        _winner(ektgl_cost, control_cost) if min(ektgl_gen, control_gen) >= 0
                        else "before_initial_snapshot"
                    ),
                    final_winner=_winner(ektgl["cost"], control["cost"]),
                ))
    return rows


def write_reports(output: Path, records: list[dict]) -> None:
    """Export aggregates and explicit per-seed comparisons."""
    buckets = defaultdict(list)
    timings = []
    runs = []
    for record in records:
        method, seed = record["method"], record["seed"]
        timings.append(dict(method=method, seed=seed, **record["timing"]))
        for task, row in record["tasks"].items():
            buckets[method, task].append(row)
            runs.append(dict(
                method=method, seed=seed, task=task, cost=row["cost"],
                feasible=row["cost"] is not None, evaluations=row["evaluations"],
                local_search_checks=row["local_search_checks"],
                accepted=row["transfer_outcomes"].get("accepted", 0),
                rejected=sum(
                    count for status, count in row["transfer_outcomes"].items()
                    if status not in {"accepted", "no_elites"}
                ),
                skipped=row["transfer_outcomes"].get("no_elites", 0),
                edge_ids=" ".join(map(str, row["edge_ids"])),
            ))

    summary, convergence = [], []
    for (method, task), rows in sorted(buckets.items()):
        summary.append(dict(
            method=method, task=task, runs=len(rows),
            **_cost_stats([row["cost"] for row in rows]),
            mean_evaluations=mean(row["evaluations"] for row in rows),
            mean_local_search_checks=mean(row["local_search_checks"] for row in rows),
        ))
        for generation, costs in enumerate(zip(*(row["history"] for row in rows), strict=True)):
            convergence.append(dict(
                method=method, task=task, generation=generation, **_cost_stats(costs),
                mean_evaluations=mean(row["evaluation_history"][generation] for row in rows),
            ))

    comparisons = matched_budgets(records)
    _write_csv(output / "runs.csv", runs)
    _write_csv(output / "summary.csv", summary)
    _write_csv(output / "convergence.csv", convergence)
    _write_csv(output / "timings.csv", timings)
    _write_csv(output / "matched_budgets.csv", comparisons, [
        "seed", "task", "control", "budget", "ektgl_evaluations", "control_evaluations",
        "ektgl_cost", "control_cost", "budget_winner", "final_winner",
    ])
    report = [
        "# EKT-GL-IDPC-DU benchmark", "",
        f"Paths and traces checked: {len(runs)}. Feasible paths: {sum(row['feasible'] for row in runs)}.",
        "", "| Method | Seeds | Feasible task runs | Total task runs | Mean solver seconds per seed |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in sorted({row["method"] for row in timings}):
        times = [row["solve_seconds"] for row in timings if row["method"] == method]
        selected = [row for row in runs if row["method"] == method]
        report.append(f"| {method} | {len(times)} | {sum(r['feasible'] for r in selected)} | "
                      f"{len(selected)} | {mean(times):.2f} |")
    if comparisons:
        report += [
            "", "| Control | EKT-GL wins / ties / losses at final state | At matched evaluation cap | Unavailable at cap |",
            "|---|---:|---:|---:|",
        ]
        for method in sorted({row["control"] for row in comparisons}):
            selected = [row for row in comparisons if row["control"] == method]
            final = Counter(row["final_winner"] for row in selected)
            budget = Counter(row["budget_winner"] for row in selected)
            triple = lambda c: f"{c['ektgl']} / {c['tie']} / {c['control']}"
            report.append(f"| {method} | {triple(final)} | {triple(budget)} | "
                          f"{budget['before_initial_snapshot'] + budget['both_infeasible']} |")
        report += ["", "Pairs in which both methods are infeasible are excluded from wins, ties and losses."]
    report += [
        "", "Each matched pair uses the last generation snapshot within the smaller final GPA budget.",
        "Evaluation requests and local search checks include memoized repeats. Solver timings exclude graph loading and output verification.",
        "A missing AVG or STD indicates an infeasible seed or insufficient repetitions for sample STD.",
        "See the manifest for the two algorithm configurations and MFEA groups. Seed IDs and results are in the run files.",
        "", "[Per-run paths](runs.csv) ? [Task statistics](summary.csv) ? [Convergence](convergence.csv) ? "
        "[Evaluation caps](matched_budgets.csv) ? [Runtime](timings.csv)", "",
    ]
    (output / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
