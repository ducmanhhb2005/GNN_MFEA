"""Publish the independent 300-generation N3 profile from actual checkpoint traces."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
import sqlite3

import numpy as np

from ekt_gl.idpc.task import CompactIDPCGraph
from ekt_gl.idpc.v2.checkpoint import read_checkpoint
from experiments._support.idpc_n3_accounting import row_accounting
from experiments._support.idpc_v3_archive import independently_validate, json_read, sha, structured, write
from experiments.report_idpc_matched_n3 import combat, holm, mean, table_by_task
from experiments.report_idpc_v3 import put
from experiments.run_idpc_matched_n3 import INIT, LS_FIELDS, METHODS, PHASES, REQUESTS, ROOT
from experiments.run_idpc_matched_n3_generation import OUT

OBJECTIVE_PHASES = ("init", "offspring", "ls", "transfer", "other")


def collect(out, manifest):
    tasks = manifest["tasks"]
    names = [t["task"] for t in tasks]
    graphs = [CompactIDPCGraph.from_file(ROOT/t["path"]) for t in tasks]
    raw, workloads, curves, details, local, coverage = [], [], [], [], [], []
    for method in METHODS:
        for seed in manifest["seeds"]:
            name = f"{method}_n3_{seed}.sqlite"
            path=out/"checkpoints"/name
            base=dict(method=method, seed=seed)
            if not path.exists():
                coverage.append(dict(**base,status="missing"))
                continue
            try:
                meta,trace=read_checkpoint(path,manifest["fingerprint"],method,seed)
            except sqlite3.OperationalError as exc:
                coverage.append(dict(**base,status="temporarily_unreadable",error=str(exc)))
                continue
            if meta["status"] != "complete":
                coverage.append(dict(**base,status=meta["status"],error=meta.get("error"),
                                     generation=trace[-1]["generation"] if trace else None))
                continue
            result=meta["result"]
            if (result["evaluation_budget"] is not None or result["actual_generations"]!=300
                    or result["stop_rule"]!="generations"
                    or [point["generation"] for point in trace] != list(range(301))
                    or [t["task"] for t in result["tasks"]]!=names):
                raise ValueError(f"Incorrect uncapped run at {path}")
            account, generations, events = row_accounting(method,seed,result,trace,detail=True)
            batches = 270 if method=="ektgl" else 300
            if (account["offspring_requests"] != 42*30*batches or account["init_requests"] != 42*INIT[method]
                    or account["scheduled_transfer_events"] != (30 if method=="ektgl" else 0)
                    or sum(t["counters"][REQUESTS] for t in result["tasks"]) != account["objective_requests"]):
                raise ValueError(f"Invalid invocation or init policy at {path}")
            expected_transfer_ticks = list(range(10,301,10)) if method == "ektgl" else []
            if [point["generation"] for point in trace if point["transfer"] is not None] != expected_transfer_ticks:
                raise ValueError(f"Incorrect transfer cadence in {path}")
            by_phase = defaultdict(Counter)
            for point in trace:
                distribution = [size for size, _ in point["populations"]]
                if (len(distribution) != 42 or sum(distribution) != 1260
                        or (method != "mfea" and any(size != 30 for size in distribution))):
                    raise ValueError(f"Incorrect population size in {path} at {point['generation']}")
                for event in point["n3_events"]:
                    if event["phase"] not in PHASES or not 0 <= event["task_index"] < 42:
                        raise ValueError(f"Invalid N3 phase or task in {path}")
                    by_phase[event["task_index"], event["phase"]].update(
                        {key: event[key] for key in LS_FIELDS})
            workloads.append(dict(**account, uncapped=True,
                                  cpu_peak_process_bytes=result["cpu_peak_process_bytes"],
                                  gpu_peak_allocated_bytes=result["gpu_peak_allocated_bytes"],
                                  started_at=result["started_at"],completed_at=result["completed_at"]))
            for generation, point in zip(generations,trace):
                costs=point["best_costs"]
                curves.append(dict(**generation,
                                   suite_mean_archive_cost=mean(costs) if all(map(math.isfinite,costs)) else None,
                                   feasible_tasks=sum(map(math.isfinite,costs))))
            for i,(task, graph, descriptor) in enumerate(zip(result["tasks"],graphs,tasks)):
                value = independently_validate(graph,task["best_path_edge_ids"])
                if value != task["best_cost"] or task["feasible"]!=math.isfinite(value):
                    raise ValueError(f"Invalid final directed path at {path}:{task['task']}")
                c=task["counters"]
                if (c["init_evaluations"]!=INIT[method]
                        or c[REQUESTS]!=sum(c.get(k+"_evaluations",0) for k in OBJECTIVE_PHASES)
                        or c[REQUESTS]!=c["actual_objective_computations"]+c.get("cache_hits",0)
                        or c[REQUESTS]!=trace[-1]["evaluations"][i]):
                    raise ValueError(f"Task ledger does not reconcile {path}:{task['task']}")
                for key in LS_FIELDS:
                    total = sum(by_phase.get((i, phase), {}).get(key, 0) for phase in PHASES)
                    actual = c.get(key, 0)
                    equal = math.isclose(total, actual, rel_tol=1e-9, abs_tol=1e-7) if key == "ls_time_s" else total == actual
                    if not equal:
                        raise ValueError(f"Unreconciled N3 {key} at {path}:{task['task']}")
                for phase in PHASES:
                    totals = by_phase.get((i, phase))
                    if totals is not None:
                        local.append(dict(**base, task=task["task"], phase=phase,
                                          improvement_rate=totals["ls_accepted"]/totals["ls_calls"],
                                          **{key: totals[key] for key in LS_FIELDS}))
                raw.append(dict(**base,configuration=method+"_n3",task=task["task"],set=descriptor["subset"],
                                cost=value,feasible=task["feasible"],independently_validated=math.isfinite(value),
                                initial_cost=trace[0]["best_costs"][i],best_generation=task["best_generation"],
                                actual_generations=300,best_path_edge_ids=json.dumps(task["best_path_edge_ids"]),**c))
            details.extend(events)
            coverage.append(dict(**base,status="complete",generation=300,request_count=account["objective_requests"],
                                 validated_feasible_paths=sum(t["feasible"] for t in result["tasks"])))
    return raw, workloads, curves, details, local, coverage


def workload_report(workloads):
    keys = ("objective_requests", "actual_objective_computations", "cache_hits", "wall_clock_s",
            "init_requests", "offspring_requests", "n3_requests", "n3_calls", "n3_improved_calls",
            "n3_candidate_path_requests", "n3_gpa_roundtrip_requests", "n3_edge_expansions", "n3_actual_expansions",
            "n3_ranking_edge_reads", "n3_actual_ranking_edge_reads", "n3_time_s", "transfer_requests",
            "model_train_steps", "training_events", "actual_transfer_events", "scheduled_transfer_events",
            "task_transfer_attempts", "train_time_s", "inference_time_s")
    result=[]
    for method in METHODS:
        rows=[r for r in workloads if r["method"]==method]
        if rows:
            result.append(dict(method=method,runs=len(rows),median_wall_clock_s=float(
                np.median([r["wall_clock_s"] for r in rows])),
                **{"mean_"+k:mean(r.get(k,0) for r in rows) for k in keys}))
    return result


def publication(out, summaries, battles, inference, ranks, work, workloads, status):
    lines = ["# IDPC-DU N3 benchmark at 300 generations", "", f"**{status}**", "",
             "EA raw + N3, MFEA raw + N3 and EKT-GL + N3 each start from their original initialization "
             "on all 42 graphs with seeds 4300–4319. EA and EKT-GL maintain 42 populations of 30. "
             "MFEA uses a unified population of 1,260. All runs have a 300-generation horizon. "
             "Each N3 call uses the same bounds and objective requests are accounted for without a suite cap. "
             "EKT-GL schedules its transfer branch every 10 generations with one training epoch per trained event.", "",
             "## Solution quality over the 42 graphs", "",
             "| Method | Seeds | Mean BF | Mean AVG | Mean sample STD |",
             "|---|---:|---:|---:|---:|"]
    for m in METHODS:
        complete = [r for r in summaries if r.get(m+"_runs")==20]
        if len(complete) == 42 and all(r[m + "_feasible"] == 1 for r in complete):
            bf,avg,std=[mean(r[f"{m}_{k}"] for r in complete) for k in ("BF","AVG","STD")]
            lines.append(f"| {m} | 20 | {bf:.3f} | {avg:.3f} | {std:.3f} |")
    lines += ["", "## Win, tie and loss", "",
              "| Scope | Comparison | Metric | W/T/L |", "|---|---|---|---|"]
    for r in battles:
        lines.append(f"| {r['scope']} | {r['comparison']} | {r['metric']} | "
                     f"{r['wins']}/{r['ties']}/{r['losses']} |")
    lines += ["", "## Mean rank and statistical significance", "",
              "Mean rank uses per-instance AVG costs and gives equal methods average ranks. "
              "For each pair and seed, the test averages relative cost differences over all 42 graphs. "
              "A positive effect favors the first method. The analysis uses paired two-sided Wilcoxon tests, "
              "Holm correction over three comparisons and 95% bootstrap intervals over seeds.", "",
              "| Method | Mean rank on all42 |", "|---|---:|"]
    lines.extend(f"| {r['method']} | {r['mean_rank']:.3f} |" for r in ranks if r["scope"]=="all42")
    lines += ["", "| Comparison | Mean relative gain (%) | 95% CI (%) | Rank-biserial | Holm p |",
              "|---|---:|---:|---:|---:|"]
    for r in inference:
        if r["p_holm"] is not None:
            lines.append(f"| {r['comparison']} | {r['effect']:.3f} | "
                         f"[{r['ci_low']:.3f}, {r['ci_high']:.3f}] | "
                         f"{r['rank_biserial']:.3f} | {r['p_holm']:.4g} |")
    lines += ["", "## Workload", "",
              "| Method | Init requests | Offspring requests | N3 requests | Total requests | Actual computations | Cache hits | Median runtime (s) |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in work:
        lines.append(f"| {r['method']} | {r['mean_init_requests']:,.0f} | {r['mean_offspring_requests']:,.0f} | "
                     f"{r['mean_n3_requests']:,.0f} | {r['mean_objective_requests']:,.0f} | "
                     f"{r['mean_actual_objective_computations']:,.0f} | {r['mean_cache_hits']:,.0f} | "
                     f"{r['median_wall_clock_s']:.2f} |")
    lines += ["", "## N3 diagnostics", "",
              "| Method | Mean calls | Mean improving calls | Mean evaluated paths | Mean GPA round-trips | Mean physical expansions | Mean time (s) |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for r in work:
        lines.append(f"| {r['method']} | {r['mean_n3_calls']:,.0f} | {r['mean_n3_improved_calls']:,.0f} | "
                     f"{r['mean_n3_candidate_path_requests']:,.0f} | {r['mean_n3_gpa_roundtrip_requests']:,.0f} | "
                     f"{r['mean_n3_actual_expansions']:,.0f} | {r['mean_n3_time_s']:.2f} |")
    ekt = [r for r in workloads if r["method"]=="ektgl"]
    if ekt:
        lines += ["", "## Knowledge transfer", "", f"Across {len(ekt)} EKT-GL suites, the traces recorded "
                  f"{sum(r['scheduled_transfer_events'] for r in ekt)} scheduled transfer events, "
                  f"{sum(r['training_events'] for r in ekt)} actual training events, "
                  f"{sum(r['model_train_steps'] for r in ekt)} optimizer steps, and "
                  f"{sum(r['actual_transfer_events'] for r in ekt)} transfer events with task attempts. "
                  f"The event ledger contains {sum(r['task_transfer_attempts'] for r in ekt):,} task attempts "
                  f"and {sum(r.get('task_transfer_accepted',0) for r in ekt):,} accepted candidates."]
    lines += ["", "Runtime is elapsed solver time on this machine under the parallel campaign. "
              "EA and MFEA use CPU and EKT-GL trains on CUDA. The solver timer excludes checkpoint writes. "
              "Logical objective requests include cache hits. N3 expansions are recorded both before and after "
              "subpath memoization. Shared generation counts yield different amounts of objective and search work.", "",
              "- [Per-instance BF, AVG, sample STD and feasibility](publication_table_all42.csv)",
              "- [Independently validated paths and per-task objective counts](full_raw_results.csv)",
              "- [Runtime and objective ledger per suite](workload.csv)",
              "- [N3 diagnostics by task and phase](local_search_diagnostics.csv)",
              "- [Convergence data and figures](figures/index.md)",
              "- [Statistical tests](statistical_tests.csv)",
              "- [Actual training and transfer](ektgl_training_transfer_by_suite.csv)", ""]
    write(out/"FINAL_BENCHMARK_REPORT.md","\n".join(lines))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=type(OUT),default=OUT)
    args=p.parse_args(argv)
    out=args.out
    manifest=json_read(out/"experiment_manifest.json")
    if any(sha(ROOT/k)!=v for k,v in manifest["sources"].items()):
        raise ValueError("Frozen generation source changed")
    if any(sha(ROOT/t["path"]) != t["sha256"] for t in manifest["tasks"]):
        raise ValueError("Frozen benchmark instances changed")
    raw, workloads, curves, detail, local, coverage=collect(out,manifest)
    states=Counter(r["status"] for r in coverage)
    summary, task_stats=table_by_task(raw)
    battles, inference, ranks=combat(task_stats,raw,summary,manifest["seeds"])
    holm(inference)
    work=workload_report(workloads)
    put(out/"full_raw_results.csv",raw)
    put(out/"publication_table_all42.csv",summary)
    put(out/"win_tie_loss.csv",battles)
    put(out/"statistical_tests.csv",inference)
    put(out/"mean_rank.csv",ranks)
    put(out/"convergence.csv",curves)
    put(out/"workload.csv",workloads)
    put(out/"workload_summary.csv",work)
    put(out/"local_search_diagnostics.csv", local)
    put(out/"transfer_events.csv",detail)
    put(out/"ektgl_training_transfer_by_suite.csv",[r for r in workloads if r["method"]=="ektgl"])
    put(out/"ektgl_training_transfer_by_generation.csv",[
        r for r in curves if r["method"]=="ektgl" and r["scheduled_transfer_events"]])
    for m in METHODS:
        put(out/(m+"_n3")/"workload.csv",[r for r in workloads if r["method"]==m])
        put(out/(m+"_n3")/"raw_results.csv",[r for r in raw if r["method"]==m])
    feasible = sum(r["independently_validated"] for r in raw)
    complete = (states=={"complete":60} and len(raw)==feasible==2520 and len(curves)==18060
                and len(battles)==18 and len(ranks)==9 and len(inference)==3
                and all(r["p_holm"] is not None for r in inference))
    status = ("SOLVER_COMPLETE" if complete else
              "VALIDATION_FAILED" if states == {"complete": 60} else "PARTIAL")
    publication(out,summary,battles,inference,ranks,work,workloads,status)
    structured(out/"coverage.json",dict(status=status, counts=dict(states), suites=coverage,
               task_results=len(raw), validated_feasible_paths=feasible))
    print(f"{status}: {dict(states)}, {feasible}/{len(raw)} independently validated feasible paths")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
