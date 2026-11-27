"""Reconcile N3 candidate work and actual neural activity from frozen traces."""
from __future__ import annotations

import argparse
from collections import Counter

from ekt_gl.idpc.v2.checkpoint import read_checkpoint
from experiments._support.idpc_v3_archive import json_read, write
from experiments.report_idpc_v3 import put
from experiments.run_idpc_matched_n3 import METHODS, OUT, REQUESTS


def row_accounting(method, seed, result, trace, *, detail=False):
    """Split one objective ledger without treating enumerated paths as evaluations."""
    g_rows, transfer_rows = [], []
    previous = 0
    totals = Counter()
    for row in trace:
        g = row["generation"]
        n3 = Counter()
        events = row["n3_events"]
        for event in events:
            n3.update({k: v for k, v in event.items() if k not in ("task_index", "phase")})
        rejected = n3["ls_du_rejects"]
        candidates = n3["ls_feasible_neighbors"] + rejected
        roundtrips = n3["ls_evaluations"] - candidates
        if roundtrips < 0 or n3["ls_complete_paths"] < candidates:
            raise ValueError(f"Invalid N3 candidate ledger for {method}/{seed}/{g}")
        fitness = row["suite_evaluations"] - previous
        init = row["suite_evaluations"] if g == 0 else 0
        offspring = row["ordinary_offspring"]
        transfer_requests = fitness - init - offspring - n3["ls_evaluations"]
        if transfer_requests < 0 or (transfer_requests and row["transfer"] is None):
            raise ValueError(f"Unreconciled objective requests at {method}/{seed}/{g}")
        event = row["transfer"]
        attempts = sum(r["attempt"] for r in event["rows"]) if event else 0
        trained = int(event is not None and event["train_loss"] is not None)
        counts = dict(init_requests=init, offspring_requests=offspring, transfer_requests=transfer_requests,
                      n3_requests=n3["ls_evaluations"], n3_calls=n3["ls_calls"],
                      n3_improved_calls=n3["ls_accepted"], n3_enumerated_paths=n3["ls_complete_paths"],
                      n3_candidate_path_requests=candidates, n3_rejected_infeasible=rejected,
                      n3_gpa_roundtrip_requests=roundtrips, n3_roundtrip_failures=n3["n3_roundtrip_failures"],
                      n3_encode_failures=n3["n3_encode_failures"], n3_edge_expansions=n3["ls_edge_expansions"],
                      n3_actual_expansions=n3["ls_subpath_actual_expansions"],
                      n3_ranking_edge_reads=n3["ls_ranking_edge_reads"],
                      n3_actual_ranking_edge_reads=n3["ls_subpath_actual_ranking_reads"],
                      n3_time_s=n3["ls_time_s"], scheduled_transfer_events=int(event is not None),
                      training_events=trained, actual_transfer_events=int(attempts > 0),
                      task_transfer_attempts=attempts)
        if event:
            for key in ("feasible","roundtrip","accepted","improvement","duplicate"):
                counts["task_transfer_"+key] = sum(bool(r.get(key)) for r in event["rows"])
            counts["task_transfer_dead_end"] = sum(r["outcome"]=="dead_end" for r in event["rows"])
            if detail:
                for r in event["rows"]:
                    transfer_rows.append(dict(method=method, seed=seed, trained=bool(trained),
                                              **{k:v for k,v in r.items() if k!="paths"}))
        previous = row["suite_evaluations"]
        totals.update(counts)
        g_rows.append(dict(method=method,seed=seed,generation=g,objective_requests=previous,
                           request_delta=fitness, **counts))
    record = result["counters"]
    for observed, expected in (("init_requests","init_evaluations"), ("offspring_requests","offspring_evaluations"),
                               ("transfer_requests","transfer_evaluations"), ("n3_requests","ls_evaluations"),
                               ("n3_calls","ls_calls"), ("task_transfer_attempts","transfer_attempts")):
        if totals[observed] != record.get(expected,0):
            raise ValueError(f"Total {observed} does not reconcile for {method}/{seed}")
    if totals["scheduled_transfer_events"] != result["transfer_events"]:
        raise ValueError(f"Scheduled transfer count differs at {method}/{seed}")
    if totals["training_events"] != result["model_train_steps"]:
        raise ValueError(f"Training count differs at {method}/{seed} with one epoch per trained event")
    if previous != record[REQUESTS] or totals["init_requests"] + totals["offspring_requests"] + \
            totals["transfer_requests"] + totals["n3_requests"] != previous:
        raise ValueError(f"Suite budget does not reconcile for {method}/{seed}")
    return (dict(method=method,seed=seed,actual_generations=result["actual_generations"],
                 stop_rule=result["stop_rule"],objective_requests=previous,
                 actual_objective_computations=record["actual_objective_computations"],
                 cache_hits=record.get("cache_hits",0), model_train_steps=result["model_train_steps"],
                 train_time_s=result["train_time_s"], inference_time_s=result["inference_time_s"],
                 wall_clock_s=result["wall_clock_s"], **totals),
            g_rows, transfer_rows)


def records(out, manifest, seeds, methods=METHODS, *, detail=False):
    for method in methods:
        for seed in seeds:
            file = out / "checkpoints" / f"{method}_n3_{seed}.sqlite"
            metadata, trace = read_checkpoint(file,manifest["fingerprint"],method,seed)
            if metadata["status"]!="complete":
                raise ValueError(f"The checkpoint is unfinished: {file}")
            result = metadata["result"]
            yield row_accounting(method,seed,result,trace,detail=detail)


def write_budget(out, full=False):
    manifest = json_read(out/"experiment_manifest.json")
    status = json_read(out/"coverage.json")
    if full and status["status"]!="COMPLETE":
        raise ValueError("Wait for the 60 budget-suite review")
    first = list(records(out,manifest,[4300],detail=True))
    mfea, timeline,_ = next(row for row in first if row[0]["method"]=="mfea")
    ekt, transfer_ticks, _ = next(row for row in first if row[0]["method"]=="ektgl")
    rows = list(records(out,manifest,manifest["seeds"])) if full else first
    supplement = out / "budget_accounting"
    put(supplement / "seed4300_mfea_by_generation.csv",timeline)
    put(supplement / "seed4300_framework_totals.csv",[row[0] for row in first])
    put(supplement / "seed4300_ektgl_training_transfer_by_generation.csv",
        [row for row in transfer_ticks if row["scheduled_transfer_events"]])
    if full:
        put(supplement/"objective_work_by_suite.csv",[r[0] for r in rows])
        put(supplement/"ektgl_training_transfer_by_suite.csv",[r[0] for r in rows if r[0]["method"]=="ektgl"])
        put(supplement/"ektgl_training_transfer_by_generation.csv",[
            event for summary,ticks,_ in rows if summary["method"]=="ektgl"
            for event in ticks if event["scheduled_transfer_events"]])
    text = [
        "# Matched-N3-Budget accounting", "",
        "The objective cap counts cache hits. Search expansions, complete paths enumerated, "
        "charged path validations and charged GPA round-trips are distinct counters.", "",
        "## Why MFEA seed 4300 stopped during generation 8", "",
        "| Component | Objective requests |", "|---|---:|",
        f"| Initialization on all 42 tasks | {mfea['init_requests']:,} |",
        f"| Offspring GPA evaluation | {mfea['offspring_requests']:,} |",
        f"| N3 path candidate evaluation | {mfea['n3_candidate_path_requests']:,} |",
        f"| N3 round-trip GPA | {mfea['n3_gpa_roundtrip_requests']:,} |",
        f"| Total | {mfea['objective_requests']:,} |", "",
        f"N3 was invoked {mfea['n3_calls']:,} times and {mfea['n3_improved_calls']:,} calls improved their "
        f"incumbent. Of {mfea['n3_enumerated_paths']:,} complete paths enumerated, "
        f"{mfea['n3_candidate_path_requests']:,} reached objective evaluation. "
        f"The full directed path validator rejected {mfea['n3_rejected_infeasible']:,}. "
        f"N3 accounted for {mfea['n3_requests']/mfea['objective_requests']:.2%} of the suite cap.", "",
        f"Generation 7 ended at {timeline[-2]['objective_requests']:,} requests, leaving "
        f"{mfea['objective_requests']-timeline[-2]['objective_requests']:,}. Generation 8 used "
        f"{timeline[-1]['offspring_requests']:,} offspring requests plus "
        f"{timeline[-1]['n3_requests']:,} N3 requests. The ordinary batch nominally offers 1,260 offspring.", "",
        f"The suite ledger records {mfea['actual_objective_computations']:,} actual computations and "
        f"{mfea['cache_hits']:,} cache hits. The beam records {mfea['n3_edge_expansions']:,} logical "
        f"expansions and {mfea['n3_actual_expansions']:,} physical expansions after memoization.", "",
        "## EKT-GL actual training and transfer", "",
        f"Seed 4300 completed {ekt['actual_generations']} generations. It trained {ekt['training_events']} times "
        f"({ekt['model_train_steps']} optimizer steps) and made task attempts in {ekt['actual_transfer_events']} "
        f"transfer events, with {ekt['task_transfer_attempts']} task attempts and "
        f"{ekt.get('task_transfer_accepted',0)} acceptances. Training and attempted transfer occurred at "
        f"generations {', '.join(str(t['generation']) for t in transfer_ticks if t['actual_transfer_events'])}.", "",
    ]
    if full:
        ekt_rows=[r[0] for r in rows if r[0]["method"]=="ektgl"]
        text += ["Across the 20 EKT-GL suites, the actual training count was "
                 f"{sum(r['training_events'] for r in ekt_rows)}, the optimizer step count was "
                 f"{sum(r['model_train_steps'] for r in ekt_rows)}, scheduled events totaled "
                 f"{sum(r['scheduled_transfer_events'] for r in ekt_rows)}, events with attempts totaled "
                 f"{sum(r['actual_transfer_events'] for r in ekt_rows)}, task attempts totaled "
                 f"{sum(r['task_transfer_attempts'] for r in ekt_rows)}, and acceptances totaled "
                 f"{sum(r.get('task_transfer_accepted',0) for r in ekt_rows)}.", "",
                 "[All-suite neural and transfer counts](ektgl_training_transfer_by_suite.csv)", ""]
    text += ["[MFEA generation ledger](seed4300_mfea_by_generation.csv) · "
             "[EKT-GL event ledger](seed4300_ektgl_training_transfer_by_generation.csv)", ""]
    write(supplement/"BUDGET_BREAKDOWN.md","\n".join(text))
    print(f"Reconciled N3 and training counts for {len(rows)} matched-budget suites")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=type(OUT),default=OUT)
    parser.add_argument("--full",action="store_true")
    args=parser.parse_args()
    write_budget(args.out,args.full)
