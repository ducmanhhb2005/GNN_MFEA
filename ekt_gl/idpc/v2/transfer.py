"""One online graph-learning event followed by verified elitist insertion."""

from time import perf_counter
import random

from .core import seed_for
from .learning import greedy
from .local_search import improve


def transfer(evolution, network, pairs):
    config = evolution.config
    tracker = evolution.tracker
    generation = evolution.generation
    labels = network.labels(evolution.populations)
    loss, train_time = network.fit(labels)
    scores, attention, inference_time = (network.scores() if loss is not None else (None, None, 0.0))
    rows = []
    for task, (name, population, label) in enumerate(zip(
            evolution.names, evolution.populations, labels)):
        graph = tracker.graphs[task]
        positives, elites, duplicates = label
        counters = tracker.per_task[task]
        best_before = evolution.best[task].cost
        row = dict(generation=generation, task=name, positive_labels=len(positives),
                   negative_labels=len(graph.edges) - len(positives) if elites else 0,
                   elite_count=elites, duplicate_elites=duplicates, train_loss=loss,
                   attempt=False, feasible=False, encoded=False, roundtrip=False,
                   accepted=False, improvement=False, pre_ls_cost=None, post_ls_cost=None,
                   best_before=best_before, best_after=best_before)
        rows.append(row)
        if loss is None or tracker.exhausted:
            row["outcome"] = "no_elites" if loss is None else "budget_exhausted"
            continue
        row["attempt"] = True
        counters["transfer_attempts"] += 1
        started = perf_counter()
        edge_ids = greedy(graph, scores[task], counters)
        inference_time += perf_counter() - started
        if edge_ids is None:
            row["outcome"] = "dead_end"
            continue
        proposed = tracker.path(task, edge_ids, "transfer")
        if not proposed.feasible:
            row["outcome"] = "invalid"
            continue
        row.update(feasible=True, pre_ls_cost=proposed.cost)
        counters["valid_candidates"] += 1
        old_success = counters["encode_successes"]
        candidate = tracker.roundtrip(task, proposed, evolution.local_limits[task], "transfer",
                                      phase="pre_transfer")
        row["encoded"] = counters["encode_successes"] > old_success
        if candidate is None:
            row["outcome"] = ("budget_exhausted" if tracker.exhausted else
                              "roundtrip_failed" if row["encoded"] else "unrepresentable")
            continue
        row["roundtrip"] = True
        rng = random.Random(seed_for(evolution.seed, name, "pre_transfer", generation))
        candidate = improve(tracker, task, candidate, evolution.local_limits[task], pairs[task],
                            config.ls, rng, "pre_transfer")
        row["post_ls_cost"] = candidate.solution.cost
        if candidate.solution.cost < population[-1].solution.cost:
            evolution.populations[task] = evolution.survive(population + [candidate], len(population))
            counters["accepted_candidates"] += 1
            row["accepted"] = True
        evolution.record(task, candidate.solution)
        row["best_after"] = evolution.best[task].cost
        row["improvement"] = row["best_after"] < best_before
        counters["strict_improvements"] += row["improvement"]
        row["outcome"] = "accepted" if row["accepted"] else "elitist_reject"
    return dict(event_generation=generation, rows=rows, attention=attention,
                train_time_s=train_time, inference_time_s=inference_time, train_loss=loss)
