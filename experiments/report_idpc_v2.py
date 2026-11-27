"""Validate checkpoint coverage, export raw V2 results and reproduce statistics/figures."""

from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import csv
from itertools import combinations
import json
import math
from pathlib import Path
from statistics import mean, median, stdev

import numpy as np
from scipy.stats import mannwhitneyu, rankdata, wilcoxon

from ekt_gl.idpc.v2.checkpoint import read_checkpoint
from ekt_gl.idpc.v2.config import METHODS
from ekt_gl.idpc.v2.core import REASONS, REQUESTS
from experiments.run_idpc_v2 import atomic_json, json_clean

PAIRS = tuple(combinations(METHODS, 2))
METHOD_LABELS = dict(ea="EA", mfea="MFEA-EDU", ektgl="EKT-GL")
COUNTERS = (REQUESTS, "actual_objective_computations", "cache_hits", "gpa_decode_calls",
            "validator_calls", *(reason + "_evaluations" for reason in REASONS),
            "local_search_checks", "ls_calls", "ls_cut_attempts", "ls_edge_expansions",
            "ls_actual_edge_expansions", "ls_beam_cache_hits", "ls_partial_cost_updates", "ls_complete_paths", "ls_missing_edge", "ls_du_rejects",
            "ls_truncated", "ls_accepted", "ls_feasible_neighbors", "ls_cost_reduction", "ls_time_s",
            "greedy_edge_checks", "encode_attempts", "encode_successes", "encode_failures",
            "roundtrip_failures", "roundtrip_successes", "transfer_attempts", "valid_candidates",
            "accepted_candidates", "strict_improvements")


def write_csv(path, rows, fields=None):
    rows = iter(rows)
    if fields is None:
        rows = list(rows)
        fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(json_clean(row))


def assert_integrity(manifest, result, traces, method, seed):
    config = manifest["effective_config"]
    k = len(manifest["tasks"])
    n = config["population_per_task"]
    generations = result["actual_generations"]
    assert result["seed"] == seed and result["method"] == method
    assert result["fingerprint"] == manifest["fingerprint"]
    assert [row["task"] for row in result["tasks"]] == [row["task"] for row in manifest["tasks"]]
    assert [row["generation"] for row in traces] == list(range(generations + 1))
    assert all(sum(n for n, _ in row["populations"]) == k * n for row in traces)
    assert all(sum(row["evaluations"]) == row["suite_evaluations"] for row in traces)
    assert all(0 <= feasible <= count for row in traces for count, feasible in row["populations"])
    count = result["counters"]
    budget = manifest["evaluation_budget"]
    assert (generations == config["generations"] or count[REQUESTS] == budget)
    assert budget is None or count[REQUESTS] <= budget
    assert count["init_evaluations"] == n * k * (k if method == "mfea" else 1)
    assert count[REQUESTS] == traces[-1]["suite_evaluations"]
    assert count[REQUESTS] == sum(count.get(r + "_evaluations", 0) for r in REASONS)
    assert count[REQUESTS] == count["actual_objective_computations"] + count.get("cache_hits", 0)
    events = [row["generation"] for row in traces if row["transfer"] is not None]
    assert len(events) == result["transfer_events"]
    expected = (list(range(config["transfer_interval"], generations+1, config["transfer_interval"]))
                if method == "ektgl" else [])
    assert events == expected
    assert all(0 <= row["ordinary_offspring"] <= n*k for row in traces)
    assert all(row["ordinary_offspring"] == 0 for row in traces if row["transfer"])
    if budget is None:
        assert count["offspring_evaluations"] == n*k*(generations-len(events))
    assert all(b["suite_evaluations"] >= a["suite_evaluations"] for a,b in zip(traces,traces[1:]))
    if not manifest["method_config"][method]["ordinary_ls"]:
        assert count.get("ls_calls", 0) == count.get("local_search_checks", 0) == 0
    for i, row in enumerate(result["tasks"]):
        assert row["best_cost"] == traces[-1]["best_costs"][i]
        assert row["feasible"] == math.isfinite(row["best_cost"])
        assert all(b["best_costs"][i] <= a["best_costs"][i] for a,b in zip(traces,traces[1:]))
    assert all(not entry["accepted"] or entry["feasible"] and entry["roundtrip"]
               for row in traces if row["transfer"] for entry in row["transfer"]["rows"])


def read_runs(out, manifest):
    completed = {}
    coverage = []
    for seed in manifest["seeds"]:
        for method in METHODS:
            name = out / "checkpoints" / f"{method}_{seed}.sqlite"
            if not name.is_file():
                coverage.append(dict(method=method, seed=seed, status="pending"))
                continue
            metadata,traces = read_checkpoint(name, manifest["fingerprint"], method, seed)
            status = metadata.get("status")
            if status == "complete":
                result = metadata["result"]
                assert_integrity(manifest,result,traces,method,seed)
                completed[method,seed] = result,traces
            coverage.append(dict(method=method, seed=seed, status=status,
                                 last_generation=traces[-1]["generation"] if traces else None,
                                 error=metadata.get("error")))
    return completed, coverage


def group_tasks(manifest, runs):
    groups = defaultdict(list)
    for (method,_), (result,_) in runs.items():
        for row in result["tasks"]:
            groups[method,row["task"]].append(row)
    return groups


def summarize(manifest, runs):
    groups = group_tasks(manifest, runs)
    out = {}
    for labels, observations in groups.items():
        costs = [r["best_cost"] for r in observations if r["feasible"]]
        counters = [r["counters"] for r in observations]
        evaluations = [r[REQUESTS] for r in counters]
        row = dict(method=labels[0], task=labels[1], runs=len(observations),
                   feasible_runs=len(costs), feasible_rate=len(costs)/len(observations),
                   BF=min(costs,default=None), AVG=mean(costs) if costs else None,
                   STD_sample=stdev(costs) if len(costs)>1 else None,
                   median=median(costs) if costs else None,
                   IQR=float(np.percentile(costs,75)-np.percentile(costs,25)) if costs else None,
                   min_evaluations=min(evaluations), max_evaluations=max(evaluations),
                   **{"mean_"+key:mean(r.get(key,0) for r in counters) for key in COUNTERS})
        out[labels] = row
    for task in manifest["tasks"]:
        name = task["task"]
        eligible = [out[m,name] for m in METHODS if (m,name) in out and out[m,name]["AVG"] is not None]
        if not eligible:
            continue
        ref = min(row["AVG"] for row in eligible)
        ranks = rankdata([row["AVG"] for row in eligible], method="average")
        for row,rank in zip(eligible,ranks):
            row["AVG_gap_percent"] = 100*(row["AVG"]-ref)/ref if ref else None
            row["AVG_rank"] = float(rank)
    return out, groups


def outcome(a,b):
    if a is None or b is None:
        return "missing"
    if math.isclose(a,b,rel_tol=0,abs_tol=1e-9):
        return "tie"
    return "win" if a<b else "loss"


def holm(rows):
    tests = sorted((r for r in rows if r["p_raw"] is not None),key=lambda r:r["p_raw"])
    cumulative = 0
    for i,row in enumerate(tests):
        cumulative = max(cumulative,min(1.0,(len(tests)-i)*row["p_raw"]))
        row["p_holm"] = cumulative
        effect = row["rank_biserial_A"]
        row["statistical"] = ("win" if effect>0 else "loss") if cumulative<0.05 else "ns"


def comparisons(manifest, summary, groups, runs):
    rows,budget_rows = [],[]
    expected = len(manifest["seeds"])
    for a,b in PAIRS:
        for task in manifest["tasks"]:
            t = task["task"]
            x,y = summary.get((a,t)), summary.get((b,t))
            if not x or not y:
                continue
            xa,ya = x["AVG"],y["AVG"]
            row = dict(method_a=a, method_b=b, task=t, subset=task["subset"],
                       AVG_A=xa, AVG_B=ya, BF_A=x["BF"], BF_B=y["BF"],
                       feasible_A=x["feasible_runs"], feasible_B=y["feasible_runs"],
                       AVG_outcome=outcome(xa,ya), BF_outcome=outcome(x["BF"],y["BF"]),
                       RPD_A_vs_B=100*(ya-xa)/ya if xa is not None and ya and ya>0 else None,
                       p_raw=None, p_holm=None, rank_biserial_A=None, statistical="insufficient")
            ax = [r["best_cost"] for r in groups[a,t] if r["feasible"]]
            bx = [r["best_cost"] for r in groups[b,t] if r["feasible"]]
            if len(ax) == len(bx) == expected:
                test = mannwhitneyu(ax,bx,alternative="two-sided",method="asymptotic")
                row["p_raw"] = float(test.pvalue)
                row["rank_biserial_A"] = 1-2*float(test.statistic)/(len(ax)*len(bx))
            rows.append(row)
        for seed in manifest["seeds"]:
            if (a,seed) not in runs or (b,seed) not in runs:
                continue
            ta,tb = (runs[m,seed][1] for m in (a,b))
            common = min(ta[-1]["suite_evaluations"],tb[-1]["suite_evaluations"])
            applicable = [trace[bisect_right([r["suite_evaluations"] for r in trace],common)-1]
                          if trace[0]["suite_evaluations"]<=common else None for trace in (ta,tb)]
            if any(row is None for row in applicable):
                continue
            sa,sb = applicable
            for i,task in enumerate(manifest["tasks"]):
                x,y = sa["best_costs"][i],sb["best_costs"][i]
                budget_rows.append(dict(method_a=a,method_b=b,seed=seed,task=task["task"],
                    shared_suite_cap=common, suite_evaluations_a=sa["suite_evaluations"],
                    suite_evaluations_b=sb["suite_evaluations"],generations_a=sa["generation"],
                    generations_b=sb["generation"],task_evaluations_a=sa["evaluations"][i],
                    task_evaluations_b=sb["evaluations"][i],cost_a=x,cost_b=y,
                    outcome=outcome(x,y) if math.isfinite(x) or math.isfinite(y) else "infeasible_both"))
    holm(rows)
    return rows,budget_rows


def paired_ea_ektgl(manifest, runs):
    """Secondary within-seed test for the two methods with identical initial genomes."""
    seeds = manifest["seeds"]
    rows = []
    for i, task in enumerate(manifest["tasks"]):
        diffs = []
        for seed in seeds:
            if ("ea", seed) not in runs or ("ektgl", seed) not in runs:
                continue
            a, b = (runs[method, seed][0]["tasks"][i] for method in ("ea","ektgl"))
            if a["feasible"] and b["feasible"]:
                diffs.append(b["best_cost"] - a["best_cost"])
        row = dict(task=task["task"], paired_seeds=len(diffs),
                   median_ektgl_minus_ea=median(diffs) if diffs else None,
                   p_raw=None, p_holm=None)
        if len(diffs) == len(seeds):
            row["p_raw"] = (float(wilcoxon(diffs, zero_method="pratt", alternative="two-sided").pvalue)
                            if any(diffs) else 1.0)
        rows.append(row)
    order = sorted((r for r in rows if r["p_raw"] is not None),key=lambda r:r["p_raw"])
    last = 0.0
    for i,row in enumerate(order):
        last = max(last,min(1.0,(len(order)-i)*row["p_raw"]))
        row["p_holm"] = last
    return rows


def write_raw(out,manifest,runs):
    identity = {k:manifest[k] for k in ("experiment_id","suite_id","profile","fingerprint")}
    tasks,suites,events = [],[],[]
    for (method,seed),(result,trace) in runs.items():
        prefix = dict(**identity,method=method,seed=seed)
        suites.append({**prefix,**{k:v for k,v in result.items() if k not in ("tasks","counters","mating")},
                       **{key:result["counters"].get(key,0) for key in COUNTERS},
                       **{"mating_"+k:v for k,v in result["mating"].items()}})
        for raw,task in zip(result["tasks"],manifest["tasks"]):
            data = {key:v for key,v in raw.items() if key!="counters"}
            data["best_path_edge_ids"] = json.dumps(data["best_path_edge_ids"])
            tasks.append(dict(**prefix,**data,actual_generations=result["actual_generations"],
                              status="complete",dataset_hash=task["sha256"],n_nodes=task["n_nodes"],
                              n_domains=task["n_domains"],n_edges=task["n_edges"],
                              **{key:raw["counters"].get(key,0) for key in COUNTERS},
                              **{key:val for key,val in raw["counters"].items() if key not in COUNTERS}))
        for row in trace:
            if row["transfer"]:
                events.extend({**prefix,**t} for t in row["transfer"]["rows"])
    write_csv(out/"task_runs.csv",tasks)
    write_csv(out/"suite_runs.csv",suites)
    write_csv(out/"transfer_events.csv",events)

    def convergence():
        for (method,seed),(_,trace) in runs.items():
            for row in trace:
                for i,task in enumerate(manifest["tasks"]):
                    count,feasible = row["populations"][i]
                    yield dict(method=method,seed=seed,task=task["task"], generation=row["generation"],
                               best_cost=row["best_costs"][i], evaluated=row["evaluations"][i],
                               suite_evaluations=row["suite_evaluations"], population_count=count,
                               feasible_fraction=feasible/count if count else None)

    write_csv(out/"convergence.csv",convergence(),
              ("method","seed","task","generation","best_cost","evaluated","suite_evaluations",
               "population_count","feasible_fraction"))
    return suites, events


def curves(manifest,runs,summary):
    reference = []
    for task in manifest["tasks"]:
        t = task["task"]
        values = [summary[m,t]["AVG"] for m in METHODS if (m,t) in summary and summary[m,t]["AVG"] is not None]
        reference.append(min(values) if values else None)

    def gap(row):
        values = [100*(cost-ref)/ref for cost,ref in zip(row["best_costs"],reference)
                  if ref and math.isfinite(cost)]
        return mean(values) if values else None

    points = defaultdict(list)
    for (method,_),(_,trace) in runs.items():
        for row in trace:
            cost = gap(row)
            if cost is not None:
                points["generation",method,row["generation"]].append(cost)
    initial = max(trace[0]["suite_evaluations"] for _,trace in runs.values())
    final = min(trace[-1]["suite_evaluations"] for _,trace in runs.values())
    if final>=initial:
        budgets = np.unique(np.linspace(initial,final,61,dtype=int))
        for (method,_),(_,trace) in runs.items():
            evaluations = [row["suite_evaluations"] for row in trace]
            for budget in budgets:
                row = trace[bisect_right(evaluations,budget)-1]
                cost = gap(row)
                if cost is not None:
                    points["suite_evaluations",method,int(budget)].append(cost)
    return [dict(axis=axis,method=method,x=x,mean_gap=mean(values),
                 lower_95=float(np.percentile(values,2.5)),upper_95=float(np.percentile(values,97.5)),
                 n=len(values)) for (axis,method,x),values in sorted(points.items())]


def grouping_results(comparison, manifest):
    buckets = defaultdict(Counter)
    for row in comparison:
        for scope in ("all",row["subset"]):
            for measure,key in (("AVG","AVG_outcome"),("BF","BF_outcome"),("Mann–Whitney-Holm","statistical")):
                buckets[row["method_a"],row["method_b"],scope,measure][row[key]] += 1
    return [dict(method_a=a,method_b=b,subset=scope,metric=metric,win=count["win"],
                 tie=count["tie"],loss=count["loss"],not_significant=count["ns"],
                 insufficient=count["insufficient"]+count["missing"])
            for (a,b,scope,metric),count in buckets.items()]


def report_text(path,manifest,runs,summary,wtl,budgets,events):
    expected = len(manifest["seeds"])*len(METHODS)
    complete = len(runs)==expected
    labels = METHOD_LABELS if manifest["profile"] == "native" else {
        **METHOD_LABELS, "ea": "EA+LS", "mfea": "MFEA+LS"}
    lines = [
        "# IDPC-DU V2 benchmark",
        "",
        f"Profile **{manifest['profile']}** · suite **{manifest['suite_id']}**",
        "",
        f"Coverage **{len(runs)}/{expected} suite runs** and "
        f"**{sum(len(data[0]['tasks']) for data in runs.values())}/{expected*len(manifest['tasks'])} task runs**.",
        "",
        "Complete coverage." if complete else "**PRELIMINARY**. See coverage.json for incomplete runs.",
        "",
        "| Method | Suite runs | Feasible task runs | Mean suite solver seconds | Mean objective requests/suite | Mean LS cuts/suite | Mean AVG gap (%) | Mean rank |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        records = [r for (m,_),(r,_) in runs.items() if m==method]
        if not records:
            continue
        gaps = [v["AVG_gap_percent"] for (m,_),v in summary.items()
                if m==method and v.get("AVG_gap_percent") is not None]
        ranks = [v["AVG_rank"] for (m,_),v in summary.items() if m==method and "AVG_rank" in v]
        lines.append(f"| {labels[method]} | {len(records)} | "
                     f"{sum(t['feasible'] for r in records for t in r['tasks'])}/{sum(len(r['tasks']) for r in records)} | "
                     f"{mean(r['wall_clock_s'] for r in records):.2f} | "
                     f"{mean(r['counters'][REQUESTS] for r in records):.2f} | "
                     f"{mean(r['counters'].get('local_search_checks',0) for r in records):.2f} | "
                     f"{mean(gaps) if gaps else math.nan:.4f} | {mean(ranks) if ranks else math.nan:.3f} |")
    lines.extend(["","| Pair (A vs B) | Scope | Metric | A wins | Exact ties | A losses | Insignificant | Insufficient |",
                  "|---|---|---|---:|---:|---:|---:|---:|"])
    for r in wtl:
        lines.append(f"| {r['method_a']} vs {r['method_b']} | {r['subset']} | {r['metric']} | "
                     f"{r['win']} | {r['tie']} | {r['loss']} | {r['not_significant']} | {r['insufficient']} |")
    lines.extend(["", "## Work and search diagnostics", "",
                  "| Method | Generations min–max | Mean actual computations | LS improvements | LS truncated cuts | LS cut attempts |",
                  "|---|---:|---:|---:|---:|---:|"])
    for method in METHODS:
        records = [r for (m,_),(r,_) in runs.items() if m==method]
        if not records:
            continue
        n_g = [r["actual_generations"] for r in records]
        counters = [r["counters"] for r in records]
        ls = [sum(c.get(key,0) for c in counters)
              for key in ("ls_accepted","ls_truncated","ls_cut_attempts")]
        computations = mean(c["actual_objective_computations"] for c in counters)
        lines.append(f"| {labels[method]} | {min(n_g)}–{max(n_g)} | {computations:.2f} | "
                     f"{ls[0]} | {ls[1]} | {ls[2]} |")
    if events:
        outcomes = Counter(event["outcome"] for event in events)
        attempts = sum(event["attempt"] for event in events)
        verified = sum(event["roundtrip"] for event in events)
        accepted = sum(event["accepted"] for event in events)
        improved = sum(event["improvement"] for event in events)
        lines.extend(["", f"EKT-GL attempted {attempts} candidates. {verified} passed exact GPA "
                      f"round-trip, {accepted} entered a population, and {improved} improved an archive.",
                      "Transfer outcomes by event/task slot",
                      " · ".join(f"{key}={value}" for key,value in sorted(outcomes.items()))])
    mfea = [trace for (method,_),(_,trace) in runs.items() if method == "mfea"]
    if mfea:
        starved = sum(n==0 for trace in mfea for row in trace for n,_ in row["populations"])
        slots = len(manifest["tasks"])*sum(map(len,mfea))
        lines.extend(["", f"MFEA had {starved}/{slots} task-generation slots with zero "
                      "individuals assigned by skill factor. Archives remain available for those tasks."])
    if budgets:
        lines.extend(["", "### At a shared suite objective cap", "",
                      "| Pair (A vs B) | A wins | Ties | A losses | Both infeasible |",
                      "|---|---:|---:|---:|---:|"])
        for a,b in PAIRS:
            c = Counter(r["outcome"] for r in budgets if (r["method_a"],r["method_b"]) == (a,b))
            lines.append(f"| {labels[a]} vs {labels[b]} | {c['win']} | {c['tie']} | "
                         f"{c['loss']} | {c['infeasible_both']} |")
        lines.append("Rows count task/seed pairs at the last recorded generation under each seed's common suite cap.")
    lines += ["", "BF/AVG and sample STD use feasible observations. Feasibility is reported separately.",
              "The two-sided Mann–Whitney U family comprises the three method pairs across all tasks "
              "within this profile. Holm correction uses comparisons with all expected feasible runs.",
              "The supplementary EA/EKT-GL signed-rank test pairs runs by seed "
              "because the methods share random initial chromosomes. It uses a separate Holm family.",
              "An insignificant test does not establish equivalence.",
              "Cost gaps use the lowest observed method AVG per task and do not assume a known optimum.",
              "Solver wall time excludes checkpoint I/O and shared dataset loading. LS cuts and requests "
              "are separate counters. Cache hits count toward the shared budget.",
              "Evaluation-axis convergence uses the last completed snapshot up to a common suite budget. "
              "The displayed 2.5–97.5 percentiles describe seed dispersion.",
              "Native includes a different LS policy for EKT-GL. Matched-LS compares complete algorithm "
              "configurations with the same LS operator and caps. Per-task budgets can still differ.",
              "", "[Coverage](coverage.json) · [Summary](summary_by_task.csv) · "
              "[Comparisons](pairwise_comparison.csv) · [Figures](figures/)"]
    path.write_text("\n".join(lines)+"\n",encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir",type=Path)
    parser.add_argument("--preliminary",action="store_true")
    parser.add_argument("--no-plots",action="store_true")
    args = parser.parse_args(argv)
    root = args.out_dir
    manifest = json.loads((root/"manifest.json").read_text(encoding="utf-8"))
    runs,coverage = read_runs(root,manifest)
    atomic_json(root/"coverage.json",dict(suite_runs=coverage,complete_suite_runs=len(runs),
        expected_suite_runs=len(coverage),complete_task_runs=len(runs)*len(manifest["tasks"]),
        expected_task_runs=len(coverage)*len(manifest["tasks"])))
    if not args.preliminary and len(runs)!=len(coverage):
        raise SystemExit("Incomplete coverage. Reporting requires --preliminary until all checkpoints are complete.")
    if not runs:
        return
    summary,groups = summarize(manifest,runs)
    pairs,budgets = comparisons(manifest,summary,groups,runs)
    wtl = grouping_results(pairs,manifest)
    suites,events = write_raw(root,manifest,runs)
    for name,rows in (("summary_by_task",summary.values()),("pairwise_comparison",pairs),
                      ("matched_suite_budgets",budgets),("win_tie_loss",wtl),
                      ("paired_wilcoxon_ea_ektgl",paired_ea_ektgl(manifest,runs))):
        write_csv(root/(name+".csv"),rows)
    curve_data = curves(manifest,runs,summary)
    write_csv(root/"convergence_normalized.csv",curve_data)
    report_text(root/"report.md",manifest,runs,summary,wtl,budgets,events)
    if not args.no_plots:
        from experiments._support.idpc_v2_figures import plot_all
        plot_all(root,manifest,runs,summary,groups,wtl,curve_data,suites,events)


if __name__ == "__main__":
    main()
