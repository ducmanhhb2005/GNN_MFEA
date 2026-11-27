"""Capture per-event diagnostics without changing the solver's decisions."""

from collections import Counter
import hashlib
import json
from pathlib import Path
from statistics import median
from time import perf_counter

from ekt_gl.idpc.research import engine
from experiments import run_idpc_v3 as runner
from experiments._support.idpc_v3_archive import ROOT, sha


TRANSFER_KEYS = ("transfer_evaluations", "ls_evaluations", "greedy_edge_checks",
                 "ls_actual_edge_expansions", "encode_attempts", "encode_successes",
                 "encode_failures", "roundtrip_successes", "roundtrip_failures")


def observed_transfer(evolution, network, pairs, lower, diagnostic):
    task_counts = evolution.tracker.per_task
    before = [Counter(row) for row in task_counts]
    members = [[m.solution.cost for m in p] for p in evolution.populations]
    diversity = [(len({m.solution.path for m in p if m.solution.feasible}),
                  len({(tuple(m.chromosome.priority), tuple(m.chromosome.edge_index)) for m in p}))
                 for p in evolution.populations]
    task_by_counter = {id(counts): t for t, counts in enumerate(task_counts)}
    construction_seconds, evaluation_seconds = {}, {}
    evaluate = engine.evaluate_candidates
    construct = engine.construct

    def measure_construction(graph, lb, variant, scores, counts, seed, event, task):
        started = perf_counter()
        result = construct(graph, lb, variant, scores, counts, seed, event, task)
        if id(counts) in task_by_counter:
            construction_seconds[task_by_counter[id(counts)]] = perf_counter() - started
        return result

    def measure_evaluation(engine_state, task, proposal, graph_pairs):
        started = perf_counter()
        result = evaluate(engine_state, task, proposal, graph_pairs)
        if engine_state is evolution:
            evaluation_seconds[task] = perf_counter() - started
        return result

    engine.construct, engine.evaluate_candidates = measure_construction, measure_evaluation
    try:
        event = engine.transfer(evolution, network, pairs, lower, diagnostic)
    finally:
        engine.construct, engine.evaluate_candidates = construct, evaluate
    for t, row in enumerate(event["rows"]):
        after = task_counts[t]
        row.update(population_best=members[t][0], population_median=median(members[t]),
                   population_worst=members[t][-1],
                   unique_feasible_paths=diversity[t][0], unique_chromosomes=diversity[t][1],
                   construction_seconds=construction_seconds.get(t, 0.0),
                   evaluation_seconds=evaluation_seconds.get(t, 0.0),
                   **{key: after[key] - before[t][key] for key in TRANSFER_KEYS})
        row["encoded"] = bool(row["encode_successes"])
        row["objective_requests"] = row["transfer_evaluations"] + row["ls_evaluations"]
        row["construction_reason"] = ("no feasible directed continuation in greedy walk"
                                      if row["outcome"] == "dead_end" else "")
    return event


def main():
    original = runner.campaign_identity

    def identity(*args, **kwargs):
        data = original(*args, **kwargs)
        data["sources"][Path(__file__).relative_to(ROOT).as_posix()] = sha(__file__)
        data["full_diagnostics"] = True
        data.pop("fingerprint")
        data["fingerprint"] = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
        return data

    runner.transfer, runner.campaign_identity = observed_transfer, identity
    return runner.main()


if __name__ == "__main__":
    raise SystemExit(main())
