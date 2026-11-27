"""Summaries for search-budget, initialization and transfer mechanism controls."""

from collections import Counter, defaultdict
from pathlib import Path
import math
import statistics

from experiments._support.idpc_v3_archive import PILOT, RESEARCH, json_read, read, write
from experiments.report_idpc_v3 import put


def number(row, key):
    return float(row[key]) if row.get(key) else None


def rows(path):
    return read(path) if path.is_file() else ()


def common_generation(root):
    source = root / "10_figures/convergence_source.csv"
    observed = defaultdict(dict)
    for row in rows(source):
        if row["phase"] == "pilot" and row["profile"] == "matched_ls" and row["method"] == "ektgl":
            if row["arm"] in ("A0", "A1"):
                observed[row["arm"], int(row["seed"])][int(row["generation"]), row["task"]] = row
    export, summary = [], []
    for seed in PILOT:
        control, adapt = observed["A0", seed], observed["A1", seed]
        if not control or not adapt:
            continue
        g = min(max(n for n, _ in control), max(n for n, _ in adapt)) - 1
        # The last recorded generation may contain a partial budget-limited batch.
        effects = []
        for (_, t), row in ((k, v) for k, v in control.items() if k[0] == g):
            other = adapt[g, t]
            b, a = (number(r, "cost") for r in (row, other))
            effect = (b-a)/max(abs(b),1)*100 if all(math.isfinite(x) for x in (a,b)) else None
            if effect is not None:
                effects.append(effect)
            export.append(dict(seed=seed, generation=g, task=t, A0_cost=b, A1_cost=a, relative_gain_pct=effect,
                               A0_suite_requests=row["suite_requests"], A1_suite_requests=other["suite_requests"]))
        if effects:
            summary.append((seed, g, statistics.mean(effects), len(effects)))
    put(root / "08_v3_pilot/local_search/common_generation_comparison.csv", export)
    return summary


def transfer_summary(root):
    base = root / "07_transfer_diagnostics"
    attempts = [r for r in rows(base/"transfer_events.csv") if r["attempt"] == "True"]
    if not attempts:
        return []
    funnel = Counter(r["outcome"] for r in attempts)
    funnel_rows = [dict(outcome=k, count=v, attempts=len(attempts), rate=v/len(attempts))
                   for k,v in sorted(funnel.items())]
    put(base/"transfer_outcomes.csv", funnel_rows)
    candidate_gaps = []
    for r in attempts:
        a,b,w = (number(r,name) for name in ("pre_ls_cost","post_ls_cost","population_worst"))
        if all(v is not None and math.isfinite(v) for v in (a,b,w)):
            candidate_gaps.append((a,b,w))
    learning = []
    for r in rows(base/"learning_metrics.csv"):
        positive, negative = (number(r,key) for key in ("positive_labels","negative_labels"))
        average_precision = number(r,"PR_AUC_AP")
        if positive+negative and average_precision is not None:
            rate = positive/(positive+negative)
            learning.append((rate, average_precision, average_precision/rate if rate else None))
    slices = list(rows(base/"local_search_diagnostics.csv"))
    measures = {key: sum(number(r,key) or 0 for r in slices) for key in (
        "ls_calls", "ls_cut_attempts", "ls_truncated", "ls_beam_pruned_cuts",
        "ls_expansion_capped_cuts", "ls_complete_capped_cuts", "ls_cut_sampling_truncated",
        "ls_missing_edge", "ls_no_feasible_reconnection_cuts", "ls_no_improvement",
        "ls_roundtrip_failures", "ls_accepted", "ls_evaluations",
    )}
    lines = [
        f"The detailed A0 ledger currently includes {len(attempts)} attempted transfers.",
        f"{sum(r['accepted']=='True' for r in attempts)} entered the population and "
        f"{sum(r['improvement']=='True' for r in attempts)} improved the archive.",
        "",
    ]
    if candidate_gaps:
        lines += [
            f"Among {len(candidate_gaps)} candidates with comparable cost and a finite worst member, "
            f"{sum(b<a for a,b,_ in candidate_gaps)} received a strict local-search improvement. "
            f"{sum(a<w for a,_,w in candidate_gaps)} beat the worst member before refinement and "
            f"{sum(b<w for _,b,w in candidate_gaps)} beat it after refinement.", "",
        ]
    if learning:
        rate, ap, lifts = zip(*learning)
        lines += [f"The median positive fraction is {statistics.median(rate):.3g}. The median average precision "
                  f"on current elite labels is {statistics.median(ap):.3g}. Median AP relative to the "
                  f"positive fraction is {statistics.median(x for x in lifts if x is not None):.3g}. "
                  "These are in-sample diagnostics for an online learner.", ""]
    if slices and measures["ls_calls"]:
        lines += ["| Local-search counter | Count |", "|---|---:|"]
        lines += [f"| {key} | {val:,.0f} |" for key,val in measures.items()]
        lines += ["", "The three per-cut truncation flags can overlap. Cut sampling, missing directed "
                  "reconnections, and calls without an improved neighbor have their own denominators.", ""]
    return lines


def initialization_report(root):
    folder = root / "05_initialization"
    verified = list(rows(folder / "matched_init_pool_audit.csv"))
    observed = [r for r in rows(folder / "matched_init_statistics.csv") if r["complete"] == "True"]
    if not verified or not observed:
        return
    heading = "## Matched initialization control"
    report = folder / "initialization_report.md"
    original = report.read_text(encoding="utf-8").split(heading, 1)[0].rstrip()
    text = [original, "", heading, "",
            f"The control matches the initial archive cost and 1,260 charged evaluations across all three "
            f"methods on {len(verified)} checked task-seed pairs. EA and EKT-GL retain the best 30 local "
            "projections per task, while MFEA uses the canonical skill assignment. The comparison uses "
            "379,260 total objective requests and the shared V2 neighborhood.", "",
            "| Contrast | Paired relative gain (%) | W/T/L |",
            "|---|---:|---|"]
    text.extend(f"| {r['comparison']} | {number(r, 'effect_pct'):.3f} | "
                f"{r['wins']}/{r['ties']}/{r['losses']} |" for r in observed)
    text += ["", "The first method benefits when the value is positive. The estimates use the pilot seeds "
             "and retain the original baseline MFEA results under the same effective profile.", "",
             "[Pool audit](matched_init_pool_audit.csv) · [Raw results](matched_init_results.csv) · "
             "[Paired statistics](matched_init_statistics.csv)", ""]
    write(report, "\n".join(text))


def main():
    root=RESEARCH
    initialization_report(root)
    generations = common_generation(root)
    transfer = transfer_summary(root)
    lines=["# Controlled mechanism checks", ""]
    if generations:
        lines+=["The last generation with complete offspring batches in both A0 and A1 is used for a "
                "secondary comparison. The experiment's primary comparison uses the fixed objective cap.", "",
                "| Seed | Common generation | A1 relative gain (%) | Comparable tasks |",
                "|---:|---:|---:|---:|"]
        lines += [f"| {seed} | {g} | {effect:.3f} | {count} |" for seed,g,effect,count in generations]
        lines += ["", "[Raw fixed-generation comparison](../08_v3_pilot/local_search/common_generation_comparison.csv)", ""]
    if transfer:
        lines += transfer
        lines += ["[Transfer outcomes](../07_transfer_diagnostics/transfer_outcomes.csv) · "
                  "[Candidate ledger](../07_transfer_diagnostics/candidate_quality.csv) · "
                  "[Diagnostics](../07_transfer_diagnostics/learning_metrics.csv)", ""]
    if len(lines)>2:
        write(root/"14_reports/CONTROLLED_MECHANISMS.md","\n".join(lines))


if __name__ == "__main__":
    main()
