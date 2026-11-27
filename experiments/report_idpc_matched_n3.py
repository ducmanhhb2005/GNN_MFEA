"""Audit checkpoint ledgers and publish the fixed Matched-N3 comparisons."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import closing
import json
import math
import sqlite3

import numpy as np
from scipy.stats import rankdata, wilcoxon

from ekt_gl.idpc.task import CompactIDPCGraph
from ekt_gl.idpc.v2.checkpoint import load, read_checkpoint
from experiments._support.idpc_v3_archive import independently_validate, json_read, read, sha, structured, write
from experiments.report_idpc_v3 import put
from experiments.run_idpc_matched_n3 import BUDGET, INIT, METHODS, OUT, PHASES, REFERENCE, REQUESTS, RESEARCH, ROOT

COMBAT = (("ektgl", "mfea"), ("ektgl", "ea"), ("mfea", "ea"))
SCOPES = ("set1", "set2", "all42")
DISPLAY = {"ea": "EA+N3 (STO-N3)", "mfea": "MFEA+N3", "ektgl": "EKT-GL+N3"}
BUDGET_PARTS = ("init", "offspring", "ls", "transfer", "other")
AGGREGATE_FIELDS = (
    REQUESTS, "init_evaluations", "offspring_evaluations", "ls_evaluations", "transfer_evaluations",
    "actual_objective_computations", "cache_hits", "ls_edge_expansions", "ls_ranking_edge_reads",
    "ls_subpath_actual_expansions", "ls_subpath_actual_ranking_reads", "actual_generations",
    "wall_clock_s", "local_search_time_s", "train_time_s", "inference_time_s", "budget_utilization",
)


def mean(values):
    values = list(values)
    return float(np.mean(values)) if values else None


def confidence_and_test(label, family, effects, seeds):
    values = np.asarray(effects, dtype=float)
    row = dict(comparison=label, family=family, unit="suite seed", seeds=len(seeds),
               seed_ids=" ".join(map(str, seeds)), effect=None, ci_low=None, ci_high=None,
               rank_biserial=None, p=None, p_holm=None)
    if not len(values) or not np.all(np.isfinite(values)):
        return row
    row["effect"] = float(values.mean())
    if len(values) < 20:
        return row
    rng = np.random.default_rng(5291)
    row["ci_low"], row["ci_high"] = np.quantile(rng.choice(values, (10000, len(values))).mean(1), [.025, .975])
    nonzero = values[values != 0]
    if len(nonzero):
        ranks = rankdata(abs(nonzero))
        row["rank_biserial"] = float(np.dot(ranks, np.sign(nonzero)) / ranks.sum())
        row["p"] = float(wilcoxon(values, alternative="two-sided", zero_method="wilcox").pvalue)
    else:
        row.update(rank_biserial=0.0, p=1.0)
    return row


def holm(rows):
    for family, size in (("combat", 3), ("operator", 3), ("benefit_contrast", 1), ("transfer", 3)):
        previous = 0.0
        indices = sorted((r for r in rows if r["family"] == family and r["p"] is not None),
                         key=lambda row: row["p"])
        for i, row in enumerate(indices):
            previous = max(previous, min(1.0, (size - i) * row["p"]))
            row["p_holm"] = previous


def prior_initial(reference, method, seed):
    """Read generation zero and the terminal metadata at their frozen identity."""
    path = REFERENCE / reference["folder"] / f"{method}_{seed}.sqlite"
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as conn:
        metadata = {k: load(v) for k, v in conn.execute(
            "SELECT key, value FROM meta WHERE key IN ('identity','status','result')")}
        if metadata["identity"] != (reference["fingerprint"], method, seed) or metadata["status"] != "complete":
            raise ValueError(f"Wrong archived comparator {path}")
        start = load(conn.execute("SELECT row FROM trace WHERE generation = 0").fetchone()[0])
    return start, metadata["result"]


def collect(out, manifest, only=None):
    sources = manifest["tasks"]
    names = [t["task"] for t in sources]
    graphs = [CompactIDPCGraph.from_file(ROOT / t["path"]) for t in sources]
    fingerprint = manifest["fingerprint"]
    raw, workload, trajectories, local, transfer, coverage = [], [], [], [], [], []
    seeds = [only] if only is not None else manifest["seeds"]
    for method in METHODS:
        for seed in seeds:
            path = out / "checkpoints" / f"{method}_n3_{seed}.sqlite"
            base = dict(method=method, configuration=method+"_n3", seed=seed)
            if not path.exists():
                coverage.append(dict(**base, status="missing"))
                continue
            try:
                meta, trace = read_checkpoint(path, fingerprint, method, seed)
            except sqlite3.OperationalError as error:
                coverage.append(dict(**base, status="temporarily_unreadable", error=str(error)))
                continue
            result = meta.get("result")
            if result is None:
                coverage.append(dict(**base, status=meta.get("status", "pending"), error=meta.get("error"),
                                     generation=trace[-1]["generation"] if trace else None))
                continue
            tasks = result["tasks"]
            if [t["task"] for t in tasks] != names or meta["status"] != "complete":
                raise ValueError(f"Invalid task coverage {path}")
            initial, archived = prior_initial(manifest["references"][method], method, seed)
            g0, end = trace[0], trace[-1]
            comparable_init = all(initial[k] == g0[k] for k in (
                "generation", "evaluations", "suite_evaluations", "best_costs", "populations"))
            if not comparable_init:
                raise ValueError(f"Initialization differs from V3 Matched-LS for {path}")
            c = result["counters"]
            requests = c[REQUESTS]
            same_stop = (result["evaluation_budget"] == archived["evaluation_budget"] == BUDGET
                         and requests == archived["counters"][REQUESTS] == BUDGET
                         and result["stop_rule"] == archived["stop_rule"] == "budget"
                         and result["actual_generations"] <= manifest["generation_ceiling"]
                         and archived["actual_generations"] <= manifest["generation_ceiling"])
            if not 0 < requests <= BUDGET or requests != end["suite_evaluations"]:
                raise ValueError(f"Invalid suite accounting in {path}")
            if not 0 < result["actual_generations"] <= 300 or end["generation"] != result["actual_generations"]:
                raise ValueError(f"Invalid stopping record in {path}")
            if sum(t["counters"][REQUESTS] for t in tasks) != requests:
                raise ValueError(f"Suite/task request counts disagree in {path}")
            event_counts = defaultdict(Counter)
            for row in trace:
                generation = row["generation"]
                trajectories.append(dict(**base, generation=generation,
                    objective_requests=row["suite_evaluations"],
                    suite_mean_archive_cost=mean(row["best_costs"]) if all(map(math.isfinite, row["best_costs"])) else None,
                    feasible_tasks=sum(map(math.isfinite, row["best_costs"])),
                    ordinary_offspring=row["ordinary_offspring"]))
                for event in row["n3_events"]:
                    values = {k: v for k, v in event.items() if k not in ("task_index", "phase")}
                    phase = event["phase"]
                    idx = event["task_index"]
                    event_counts[idx, phase].update(values)
                    local.append(dict(**base, generation=generation, task=names[idx], phase=phase, **values))
                if row["transfer"] is not None:
                    for evt in row["transfer"]["rows"]:
                        transfer.append(dict(**base, **{k: v for k, v in evt.items() if k != "paths"},
                                             candidates=json.dumps([
                                                 {**x, "cost": x["cost"] if math.isfinite(x["cost"]) else None}
                                                 for x in evt.get("paths", [])])))
            for i, (t, graph, descriptor) in enumerate(zip(tasks, graphs, sources)):
                tc = t["counters"]
                if (tc[REQUESTS] != sum(tc.get(k+"_evaluations", 0) for k in BUDGET_PARTS)
                        or tc[REQUESTS] != tc["actual_objective_computations"] + tc.get("cache_hits", 0)
                        or tc[REQUESTS] != end["evaluations"][i]
                        or tc["init_evaluations"] != INIT[method]):
                    raise ValueError(f"Invalid task accounting {path}:{t['task']}")
                for key in ("ls_calls", "ls_accepted", "ls_cost_reduction", "ls_evaluations",
                            "ls_edge_expansions", "ls_du_rejects",
                            "n3_encode_failures", "n3_roundtrip_failures", "ls_subpath_memo_hits"):
                    if sum(event_counts[i, phase][key] for phase in PHASES) != tc.get(key, 0):
                        raise ValueError(f"N3 event ledger differs on {key} at {path}:{t['task']}")
                feasible = t["feasible"]
                verified = independently_validate(graph, t["best_path_edge_ids"])
                if verified != t["best_cost"] or feasible != math.isfinite(verified):
                    raise ValueError(f"Independent DU validation failed on {path}:{t['task']}")
                raw.append(dict(**base, task=t["task"], set=descriptor["subset"], cost=t["best_cost"],
                                feasible=feasible, independently_validated=feasible,
                                initial_cost=g0["best_costs"][i], best_generation=t["best_generation"],
                                actual_generations=end["generation"], best_path_edge_ids=json.dumps(t["best_path_edge_ids"]),
                                paired_2opt_eligible=same_stop, **tc))
            metrics = {k: result[k] for k in ("actual_generations", "wall_clock_s", "local_search_time_s",
                       "train_time_s", "inference_time_s", "stop_rule", "started_at", "completed_at",
                       "cpu_peak_process_bytes", "gpu_peak_allocated_bytes", "model_train_steps")}
            workload.append(dict(**base, paired_2opt_eligible=same_stop, init_archive_exact=comparable_init,
                                 budget=BUDGET, budget_utilization=requests/BUDGET, **metrics, **c))
            coverage.append(dict(**base, status="complete" if same_stop else "budget_incompatible",
                                 generation=end["generation"], actual_requests=requests,
                                 initial_archive_exact=comparable_init,
                                 validated_feasible_paths=sum(t["feasible"] for t in tasks)))
    return raw, workload, trajectories, local, transfer, coverage


def table_by_task(raw):
    groups = defaultdict(list)
    for row in raw:
        groups[row["task"], row["method"]].append(row)
    summary = {}
    for (task, method), rows in groups.items():
        costs = np.array([r["cost"] for r in rows])
        finite = costs[np.isfinite(costs)]
        summary[task, method] = dict(
            BF=float(finite.min()) if len(finite) else None, AVG=float(costs.mean()),
            STD=float(costs.std(ddof=1)) if len(costs) > 1 and len(costs)==len(finite) else None,
            feasible=len(finite)/len(costs), runs=len(costs))
    tasks = [dict(task=t, set=next(r["set"] for r in raw if r["task"]==t),
                 **{f"{m}_{key}": value for m in METHODS
                    for key,value in summary.get((t,m), {}).items()})
             for t in sorted({r["task"] for r in raw})]
    return tasks, summary


def wtl(values_a, values_b):
    return dict(wins=sum(a < b for a,b in zip(values_a, values_b)),
                ties=sum(a == b for a,b in zip(values_a, values_b)),
                losses=sum(a > b for a,b in zip(values_a, values_b)))


def cost_matrix(rows, method, tasks, seeds):
    ledger = {(int(r["seed"]), r["task"]): float(r["cost"]) for r in rows if r["method"]==method}
    return np.array([[ledger[s,t] for t in tasks] for s in seeds])


def relative(new, old):
    return (old - new) / np.maximum(1., np.abs(old)) * 100


def combat(summary, raw, tasks, seeds):
    if any(len([r for r in raw if r["method"]==m]) != len(tasks)*len(seeds) for m in METHODS):
        return [], [], []
    matrices = {m: cost_matrix(raw, m, [t["task"] for t in tasks], seeds) for m in METHODS}
    records, tests, ranks = [], [], []
    for scope in SCOPES:
        chosen = [t["task"] for t in tasks if scope=="all42" or t["set"]==scope]
        means = np.array([[summary[t,m]["AVG"] for m in METHODS] for t in chosen])
        rank = rankdata(means, axis=1)
        ranks.extend(dict(scope=scope, method=m, mean_rank=float(rank[:,i].mean())) for i,m in enumerate(METHODS))
        for a,b in COMBAT:
            for metric in ("AVG", "BF"):
                values = [[summary[t,m][metric] if summary[t,m][metric] is not None else math.inf
                           for t in chosen] for m in (a,b)]
                records.append(dict(scope=scope, metric=metric, comparison=f"{a} vs {b}", **wtl(*values)))
    for a,b in COMBAT:
        tests.append(confidence_and_test(f"{a} vs {b}", "combat",
                     relative(matrices[a], matrices[b]).mean(axis=1), seeds))
    return records, tests, ranks


def historical_comparison(manifest, raw, workload):
    prior = [r for r in read(RESEARCH/"09_final_benchmark/full_raw_results.csv") if r["profile"]=="matched_ls"]
    tasks = [t["task"] for t in manifest["tasks"]]
    seeds = manifest["seeds"]
    if len(prior) != len(METHODS)*len(tasks)*len(seeds):
        raise ValueError("V3 Matched-LS raw archive incomplete")
    old_seeds = {int(r["seed"]) for r in prior}
    if old_seeds != set(seeds):
        raise ValueError("Prior seed list differs")
    records, tests, gains = [], [], {}
    for m in METHODS:
        rows = [r for r in workload if r["method"]==m]
        if len(rows) != len(seeds):
            continue
        old, new = (cost_matrix(x,m,tasks,seeds) for x in (prior,raw))
        exact = all(r["paired_2opt_eligible"] for r in rows)
        count = wtl(new.mean(0), old.mean(0))
        effects = relative(new,old).mean(1)
        label = f"{m}: N3 vs archived 2-opt"
        records.append(dict(method=m, paired=exact, comparison=label, prior_2opt_AVG=float(old.mean()),
                            n3_AVG=float(new.mean()), task_relative_gain_pct=float(effects.mean()), **count))
        if exact:
            tests.append(confidence_and_test(label, "operator", effects, seeds))
            gains[m] = effects
    if "ektgl" in gains and "mfea" in gains:
        tests.append(confidence_and_test("EKT-GL N3 gain minus MFEA N3 gain", "benefit_contrast",
                                         gains["ektgl"]-gains["mfea"], seeds))
    return records, tests


def is_true(value):
    return value is True or value == "True"


def transfer_totals(events, method_name):
    grouped = defaultdict(list)
    for event in events:
        grouped[int(event["seed"])].append(event)
    result = []
    for seed, rows in sorted(grouped.items()):
        attempted = [r for r in rows if is_true(r["attempt"])]
        roundtrip = [r for r in attempted if is_true(r["roundtrip"])]
        costs = [r for r in roundtrip if r.get("pre_ls_cost") not in ("", None)
                 and r.get("post_ls_cost") not in ("", None)]
        count = lambda key: sum(is_true(r.get(key)) for r in attempted)
        rate = lambda numerator, denominator: numerator / denominator if denominator else None
        result.append(dict(operator=method_name, seed=seed, attempts=len(attempted), feasible=count("feasible"),
                           representable=len(roundtrip), accepted=count("accepted"), strict_improvements=count("improvement"),
                           duplicate_candidates=count("duplicate"), dead_ends=sum(r["outcome"]=="dead_end" for r in attempted),
                           acceptance_rate=rate(count("accepted"), len(attempted)),
                           strict_rate=rate(count("improvement"), len(attempted)),
                           dead_end_rate=rate(sum(r["outcome"]=="dead_end" for r in attempted),len(attempted)),
                           pre_ls_mean_cost=mean(float(r["pre_ls_cost"]) for r in costs),
                           post_ls_mean_cost=mean(float(r["post_ls_cost"]) for r in costs),
                           paired_paths=len(costs),
                           immediate_gain_pct=mean(relative(float(r["post_ls_cost"]),
                                                       float(r["pre_ls_cost"])) for r in costs)))
    return result


def transfer_comparison(transfer, eligibility, seeds):
    prior = [r for r in read(RESEARCH/"09_final_benchmark/online_transfer_outcomes.csv")
             if r["profile"]=="matched_ls" and r["method"]=="ektgl"]
    records = transfer_totals(prior,"2-opt") + transfer_totals(transfer,"N3")
    tests = []
    idx = {(r["operator"], r["seed"]):r for r in records}
    if eligibility:
        for metric in ("acceptance_rate","strict_rate","immediate_gain_pct"):
            rows = [(idx.get(("N3",s),{}).get(metric),idx.get(("2-opt",s),{}).get(metric)) for s in seeds]
            if all(a is not None and b is not None for a,b in rows):
                scale = 1 if metric=="immediate_gain_pct" else 100
                tests.append(confidence_and_test("EKT transfer "+metric+" N3 minus 2-opt", "transfer",
                                                  [scale*(a-b) for a,b in rows], seeds))
    return records, tests


def workload_summary(workload):
    result = []
    for method in METHODS:
        rows = [r for r in workload if r["method"]==method]
        if rows:
            row = dict(method=method, runs=len(rows))
            row.update({"mean_"+key:mean(r.get(key,0) for r in rows) for key in AGGREGATE_FIELDS})
            for key in ("wall_clock_s","actual_generations"):
                row["median_"+key] = float(np.median([r[key] for r in rows]))
            result.append(row)
    return result


def previous_workload(work):
    prior = [r for r in read(RESEARCH/"09_final_benchmark/suite_workload.csv") if r["profile"]=="matched_ls"]
    flags = {(r["method"],r["seed"]) for r in read(RESEARCH/"09_final_benchmark/runtime_flags.csv")
             if r["profile"]=="matched_ls"}
    result=[]
    for row in work:
        m = row["method"]
        previous = [r for r in prior if r["method"]==m]
        if row["runs"] != 20 or len(previous) != 20:
            continue
        old_elapsed = float(np.median([float(r["wall_clock_s"]) for r in previous]))
        result.append(dict(method=m, previous_median_runtime_s=old_elapsed,
                           n3_median_runtime_s=row["median_wall_clock_s"],
                           median_runtime_ratio=row["median_wall_clock_s"]/old_elapsed,
                           previous_mean_computations=mean(float(r["actual_objective_computations"])
                                                            for r in previous),
                           n3_mean_computations=row["mean_actual_objective_computations"],
                           previous_mean_generations=mean(float(r["actual_generations"]) for r in previous),
                           n3_mean_generations=row["mean_actual_generations"],
                           previous_timing_flags=sum((m,r["seed"]) in flags for r in previous)))
    return result


def ls_summary(local):
    output = []
    for method in METHODS:
        for phase in (*PHASES,"all"):
            rows = [r for r in local if r["method"]==method and (phase=="all" or r["phase"]==phase)]
            if rows:
                counts = {k:sum(r[k] for r in rows) for k in ("ls_calls","ls_accepted","ls_cost_reduction",
                         "ls_time_s","ls_du_rejects","n3_encode_failures","n3_roundtrip_failures")}
                output.append(dict(method=method, phase=phase, **counts,
                                   improvement_rate=counts["ls_accepted"]/counts["ls_calls"],
                                   mean_cost_reduction_per_call=counts["ls_cost_reduction"]/counts["ls_calls"],
                                   mean_cost_reduction_per_improved_call=(
                                       counts["ls_cost_reduction"]/counts["ls_accepted"] if counts["ls_accepted"] else None)))
    return output


def report(out, tasks, raw, workload, combat_rows, operator_rows, statistics, work, prior_work, ls, kt, done):
    title = "# IDPC-DU Matched-N3 benchmark\n\n"
    lines = [title, f"**{'COMPLETE' if done else 'PARTIAL'}** for the solver and analysis gates.",
             "", "EA, canonical USS MFEA and EKT-GL use the same V3 Native N3 subpath replacement "
             "and identical cut sampling, caps, exact edge-ID validation, and per-suite 32,768-entry search "
             "memo. All frameworks use their original initialization. EKT-GL keeps its online transfer "
             "and charges its transfer refinement in the same ledger.",
             "", f"Recorded {len(workload)}/60 suites and {len(raw)}/2520 task results. "
             f"{sum(r['independently_validated'] for r in raw)} feasible paths passed independent validation.",
             "", "Size 30 per task, MFEA USS population 1,260, all42, seeds 4300–4319. "
             "The stopping rule is 379,260 objective requests or 300 generations, whichever occurs first.",
             "", "## Solution quality", "",
             "| Method | Mean task BF | Mean task AVG | Mean sample STD | Feasibility |",
             "|---|---:|---:|---:|---:|"]
    for m in METHODS:
        n = [r for r in tasks if r.get(f"{m}_runs")==20]
        if len(n)==42:
            means = [mean(r[f"{m}_{metric}"] for r in n) for metric in ("BF","AVG","STD","feasible")]
            lines.append(f"| {DISPLAY[m]} | {means[0]:.3f} | {means[1]:.3f} | {means[2]:.3f} | {means[3]:.1%} |")
    lines += ["", "[All42 BF/AVG/STD](publication_table_all42.csv) · [Ranks](mean_rank.csv) · [Full raw data](full_raw_results.csv)", "",
              "## Win, tie and loss", "",
              "| Scope | Comparison | AVG W/T/L | BF W/T/L |", "|---|---|---|---|"]
    for r in combat_rows:
        if r["metric"]=="AVG":
            b = next(row for row in combat_rows if row["scope"]==r["scope"]
                     and row["comparison"]==r["comparison"] and row["metric"]=="BF")
            lines.append(f"| {r['scope']} | {r['comparison']} | {win(r)} | {win(b)} |")
    lines += ["", "## N3 versus archived V3 Matched-LS", "",
              "The paired gate requires matching seeds, initial archives, source and data fingerprints, "
              "framework configuration, and the exact objective cap/ceiling. The historical comparator is "
              "read from V3. Each effect averages task-relative cost changes within seed.", "",
              "| Method | Archived AVG | N3 AVG | Relative gain | AVG W/T/L | Paired |", "|---|---:|---:|---:|---|---|"]
    for r in operator_rows:
        lines.append(f"| {r['method']} | {r['prior_2opt_AVG']:.3f} | {r['n3_AVG']:.3f} | "
                     f"{r['task_relative_gain_pct']:+.2f}% | {win(r)} | {r['paired']} |")
    test_lookup = {r["comparison"]:r for r in statistics}
    delta = test_lookup.get("EKT-GL N3 gain minus MFEA N3 gain")
    if delta and delta["p_holm"] is not None:
        lines += ["", f"The gain contrast (EKT-GL minus MFEA) was {delta['effect']:+.2f} percentage points, "
                  f"with 95% CI [{delta['ci_low']:+.2f}, {delta['ci_high']:+.2f}] and "
                  f"Holm p={delta['p_holm']:.4g}."]
    lines += ["", "[Operator changes](comparison_with_v3_2opt.csv) · [Paired tests, confidence intervals and effects](statistical_tests.csv)", "",
              "## N3 workload", "",
              "| Method | Init requests | Total requests | Computations | Cache hits | LS expansions (logical / actual) | Median time (s) | Mean generations | Utilization |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in work:
        lines.append(f"| {r['method']} | {r['mean_init_evaluations']:,.0f} | "
                     f"{r['mean_'+REQUESTS]:,.0f} | {r['mean_actual_objective_computations']:,.0f} | "
                     f"{r['mean_cache_hits']:,.0f} | {r['mean_ls_edge_expansions']:,.0f} / "
                     f"{r['mean_ls_subpath_actual_expansions']:,.0f} | {r['median_wall_clock_s']:.2f} | "
                     f"{r['mean_actual_generations']:.1f} | {r['mean_budget_utilization']:.2%} |")
    if prior_work:
        lines += ["", "| Method | 2-opt median (s) | N3 median (s) | N3 / 2-opt | Archived timing flags |",
                  "|---|---:|---:|---:|---:|"]
        for row in prior_work:
            lines.append(f"| {row['method']} | {row['previous_median_runtime_s']:.2f} | "
                         f"{row['n3_median_runtime_s']:.2f} | {row['median_runtime_ratio']:.2f} | "
                         f"{row['previous_timing_flags']} |")
        lines += ["", "These medians come from concurrent workloads on the same host. "
                  "The V3 archive flags power-related elapsed-time outliers. The workload ledger keeps "
                  "every observation and records objective evaluations, physical expansions and elapsed time separately.",
                  "", "[Workload comparison](comparison_workload_with_v3_2opt.csv)"]
    lines += ["", "| Method and phase | Calls | Improved calls | Rate | Reduction per call | Reduction per improved call | LS time (s, summed) | Infeasible candidates | Encode failures | GPA failures |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in ls:
        drop = r["mean_cost_reduction_per_improved_call"]
        lines.append(f"| {r['method']} {r['phase']} | {r['ls_calls']:,} | {r['ls_accepted']:,} | "
                     f"{r['improvement_rate']:.2%} | {r['mean_cost_reduction_per_call']:.3f} | "
                     f"{drop if drop is None else round(drop,3)} | {r['ls_time_s']:.2f} | "
                     f"{r['ls_du_rejects']:,} | {r['n3_encode_failures']:,} | {r['n3_roundtrip_failures']:,} |")
    lines += ["", "[Suite workloads](workload.csv) · [N3 events by generation, task and phase](local_search_events.csv)", "",
              "## Transfer", "",
              "| Operator | Attempts | Feasible | Round-trip | Accepted | Archive improvements | Duplicates | Dead-end rate | Mean pre-LS cost | Mean post-LS cost | Mean immediate refinement gain |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for operator in ("2-opt","N3"):
        rows = [r for r in kt if r["operator"]==operator]
        if rows:
            fields = ("attempts","feasible","representable","accepted","strict_improvements","duplicate_candidates")
            totals = " | ".join(f"{sum(r[k] for r in rows):,}" for k in fields)
            immediate = mean(r["immediate_gain_pct"] for r in rows if r["immediate_gain_pct"] is not None)
            attempts = sum(r["attempts"] for r in rows)
            dead = sum(r["dead_ends"] for r in rows)
            rate = f"{dead/attempts:.2%}" if attempts else "NA"
            costs = [mean(r[key] for r in rows if r[key] is not None)
                     for key in ("pre_ls_mean_cost","post_ls_mean_cost")]
            cost_text = " | ".join(f"{c:.3f}" if c is not None else "NA" for c in costs)
            lines.append(f"| {operator} | {totals} | {rate} | {cost_text} | "
                         f"{immediate if immediate is None else f'{immediate:+.2f}%'} |")
    lines += ["", "Immediate gain uses representable transfer paths with both costs recorded. Relative rates "
              "use attempts as denominator. The number of transfer opportunities can change at the same "
              "budget because N3 changes how quickly evaluations are spent. The comparisons measure the "
              "observed transfer pipeline. The contribution of learned scores would require a dedicated control.",
              "", "| Transfer measure | N3 minus 2-opt (percentage points) | Bootstrap 95% CI | Holm p |",
              "|---|---:|---:|---:|"]
    for r in statistics:
        if r["family"]=="transfer" and r["p_holm"] is not None:
            lines.append(f"| {r['comparison']} | {r['effect']:+.2f} | "
                         f"[{r['ci_low']:+.2f}, {r['ci_high']:+.2f}] | {r['p_holm']:.4g} |")
    lines += ["", "[Events](transfer_events.csv) · [Transfer metrics over seeds](transfer_summary.csv)", "",
              "## Interpretation", ""]
    if done:
        significant = [r for r in statistics if r["family"]=="operator" and r["p_holm"] is not None
                       and r["effect"]>0 and r["p_holm"]<.05]
        if len(significant)==3:
            lines.append("N3 is a supported common local-search baseline for the all42 IDPC-DU suite. "
                         "All three paired operator comparisons yield lower relative cost with Holm p below 0.05. "
                         "Treat its actual search work and elapsed time as separate resource measures.")
        else:
            names = ", ".join(r["comparison"].split(":")[0] for r in significant) or "none"
            lines.append(f"The paired evidence for improvement under N3 is clearest for {names}. "
                         "N3 serves as a common-operator condition on all42. Adoption as a default "
                         "depends on each framework's effect and runtime.")
        duel = test_lookup.get("ektgl vs mfea",{})
        if duel.get("p_holm") is not None:
            lines.append(f"EKT-GL+N3 versus MFEA+N3 has mean relative gain {duel['effect']:+.2f}% "
                         f"with 95% interval [{duel['ci_low']:+.2f}%, {duel['ci_high']:+.2f}%] "
                         f"and Holm p={duel['p_holm']:.4g}.")
            if duel["ci_low"] <= 0 <= duel["ci_high"]:
                lines.append("The confidence interval allows effects in both directions.")
    lines += ["", "[Figures and figure sources](figures/index.md) · [Effective configurations and commands](README.md)", ""]
    write(out/"FINAL_BENCHMARK_REPORT.md","\n".join(lines))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=type(OUT), default=OUT)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args(argv)
    out = args.out
    manifest = json_read(out/"experiment_manifest.json")
    for relative, digest in manifest["reference_artifacts"].items():
        if sha(RESEARCH/relative) != digest:
            raise ValueError(f"V3 artifact changed: {relative}")
    raw, workload, trace, local, transfer, coverage = collect(out,manifest,4300 if args.smoke else None)
    flags = Counter(r["status"] for r in coverage)
    if args.smoke:
        passed = flags=={"complete":3} and len(raw)==126 and all(r["feasible"] for r in raw)
        structured(out/"logs/smoke_validation.json", dict(passed=passed, suites=coverage,
                   task_results=len(raw), init_requests={r["method"]:r["init_evaluations"] for r in workload}))
        print(f"Smoke gate {'PASS' if passed else 'PENDING/FAILED'}: {flags}, {len(raw)} task results")
        return 0 if passed else 2
    solver_done = flags=={"complete":60} and len(raw)==2520
    structured(out/"coverage.json",dict(status="SOLVER_COMPLETE" if solver_done else "PARTIAL",
               counts=dict(flags), suites=coverage, task_results=len(raw),
               feasible_results=sum(r["independently_validated"] for r in raw)))
    files = {"full_raw_results":raw, "workload":workload, "convergence":trace,
             "local_search_events":local, "transfer_events":transfer}
    for name, rows in files.items():
        put(out/(name+".csv"), rows)
        for m in METHODS:
            put(out/(m+"_n3")/(name+".csv"), [r for r in rows if r["method"]==m])
    tasks, by_task = table_by_task(raw)
    complete_seed_set = manifest["seeds"]
    combat_rows, stats, ranks = combat(by_task,raw,tasks,complete_seed_set)
    op_rows, op_tests = historical_comparison(manifest,raw,workload)
    transfer_rows, transfer_tests = transfer_comparison(
        transfer, any(r["method"]=="ektgl" and r["paired"] for r in op_rows), complete_seed_set)
    stats += op_tests + transfer_tests
    holm(stats)
    work, ls = workload_summary(workload), ls_summary(local)
    prior_work = previous_workload(work)
    for name, rows in (("publication_table_all42",tasks),("win_tie_loss",combat_rows),
                       ("comparison_with_v3_2opt",op_rows),("statistical_tests",stats),("mean_rank",ranks),
                       ("workload_summary",work),("local_search_summary",ls),("transfer_summary",transfer_rows),
                       ("comparison_workload_with_v3_2opt",prior_work)):
        put(out/(name+".csv"), rows)
    report(out,tasks,raw,workload,combat_rows,op_rows,stats,work,prior_work,ls,transfer_rows,solver_done)
    print(f"{dict(flags)} suites, {len(raw)} task results, {len(trace)} trace rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
