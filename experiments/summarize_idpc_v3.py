"""Build publication tables and evidence statements from the checkpoint exports."""

from collections import Counter, defaultdict
from datetime import datetime
import itertools

import numpy as np
from scipy.stats import friedmanchisquare, mannwhitneyu, rankdata

from experiments._support.idpc_v3_archive import RESEARCH, json_read, outcome, read, write
from experiments.report_idpc_v3 import PROFILES, put


METHODS = ("ea", "mfea", "ektgl")
LABELS = {"ea": "EA", "mfea": "MFEA-EDU V2", "ektgl": "EKT-GL V3"}
BASE = "09_final_benchmark"
METRICS = ("BF", "AVG", "STD", "median", "IQR", "worst", "feasible_rate")


def observations(path):
    return list(read(path)) if path.is_file() else []


def publication(root):
    final = observations(root / BASE / "summary_by_task.csv")
    archive = observations(root / "04_cross_version/summary_by_task.csv")
    if not final:
        return []
    full, ranking, matrix, group_scope = [], [], [], {}
    for profile in PROFILES:
        stats = [r for r in final if r["profile"] == profile and int(r["runs"]) == 20]
        task_rows = defaultdict(dict)
        for row in stats:
            task_rows[row["task"]][row["method"]] = row
            group_scope[row["task"]] = row["set"]
        all42 = [(task, cells) for task, cells in sorted(task_rows.items()) if all(m in cells for m in METHODS)]
        if len(all42) != 42:
            continue
        for task, cells in all42:
            row = {"profile": profile, "set": group_scope[task], "task": task}
            for method, values in cells.items():
                row.update({f"{method}_{metric}": values[metric] for metric in METRICS})
            full.append(row)
        for scope in ("set1", "set2", "all42"):
            tasks = [(t, s) for t, s in all42 if scope == "all42" or group_scope[t] == scope]
            values = np.array([[float(cells[m]["AVG"]) for m in METHODS] for _, cells in tasks])
            if np.all(np.isfinite(values)):
                ranks = rankdata(values, axis=1)
                if np.all(values == values[:, :1]):
                    statistic, p = 0.0, 1.0
                else:
                    f = friedmanchisquare(*values.T)
                    statistic, p = float(f.statistic), float(f.pvalue)
                for column, method in enumerate(METHODS):
                    baseline = values[:, 1]
                    gaps = (values[:, column]-baseline)/np.maximum(np.abs(baseline),1.)*100
                    ranking.append(dict(profile=profile, scope=scope, method=method, tasks=len(tasks),
                                        mean_rank=ranks[:,column].mean(), mean_gap_to_mfea_pct=gaps.mean(),
                                        friedman_chi_square=statistic,
                                        friedman_p_exploratory=p))
            for left, right in itertools.combinations(METHODS[::-1], 2):
                for metric in ("AVG", "BF"):
                    counts = Counter(outcome(float(c[left][metric]), float(c[right][metric])) for _,c in tasks)
                    matrix.append(dict(profile=profile, scope=scope, metric=metric, method_a=left, method_b=right,
                                       wins=counts["win"], ties=counts["tie"], losses=counts["loss"]))
    put(root / BASE / "publication_table_all42.csv", full)
    for scope in ("set1", "set2"):
        put(root / BASE / f"publication_table_{scope}.csv", [r for r in full if r["set"] == scope])
    put(root / BASE / "rank_and_gap.csv", ranking)
    put(root / BASE / "win_tie_loss_by_set.csv", matrix)
    if not full:
        return []
    historic = defaultdict(dict)
    keys = {"legacy:ektgl": "ekt_legacy", "native:ektgl": "ekt_v2", "native:mfea": "mfea_v2"}
    for row in archive:
        if row["method"] in keys:
            historic[row["task"]].update({f"{keys[row['method']]}_{key}": row[key] for key in ("BF","AVG","STD")})
    for row in full:
        if row["profile"] == "native":
            historic[row["task"]].update({f"ekt_v3_{key}": row["ektgl_"+key] for key in ("BF","AVG","STD")})
    edition = [dict(task=t, **cells) for t,cells in sorted(historic.items()) if "ekt_v3_AVG" in cells]
    put(root / BASE / "historical_comparison.csv", edition)
    columns = [(m, metric) for m in METHODS for metric in ("BF", "AVG", "STD")]
    tex = [r"% Generated from full 20-seed profile tables."]
    for profile in sorted({r["profile"] for r in full}):
        tex += [rf"\paragraph{{{profile.replace('_', ' ')}}}",
                r"\begin{tabular}{lrrrrrrrrr}",
                "Instance & " + " & ".join(f"{LABELS[m]} {field}" for m,field in columns) + r" \\ \hline"]
        for r in full:
            if r["profile"] == profile:
                display = [f"{float(r[m+'_'+metric]):.3f}" if r[m+"_"+metric] else "NA" for m,metric in columns]
                tex.append(r["task"].removeprefix("idpc_") + " & " + " & ".join(display) + r" \\")
        tex += [r"\end{tabular}", ""]
    write(root / "14_reports/results_tables.tex", "\n".join(tex))
    return matrix


def root_causes(root):
    pilot = observations(root / "06_ablation/ablation_statistics.csv")
    by_test = {row["comparison"]: row for row in pilot if row["complete"] == "True"}
    legacy = json_read(root / "experiment_status.json")
    lines = [
        "# Root cause analysis of EKT-GL-IDPC-DU", "",
        f"**Legacy recovery.** Verified source commit 0a4e11d against all 26 archived source hashes. "
        f"Independent reruns match the archive on {legacy['legacy_exact']}/80 suite-runs after excluding timing. "
        "[Replay ledger](../02_legacy_reproduction/replay_verification.csv)", "",
        "**Confirmed absolute regression.** Legacy EKT-GL has lower AVG than native V2 EKT-GL "
        "on 40/42 instances, one tie and one loss. The V2 MFEA change alone cannot account for this "
        "difference. On the 42 instances V2 MFEA has 18 wins, 12 ties and 12 losses against legacy MFEA by AVG. "
        "Cross-version comparisons use compatible inputs and path semantics, with different optimization "
        "protocols. [Cross-version results](../04_cross_version/cross_version_report.md)", "",
        "**Confirmed initialization mechanism.** Canonical V2 MFEA evaluates 1,260 chromosomes on every task. "
        "Standard EKT-GL evaluates 30 per task. Its MFEA comparison archive at generation zero wins/ties/loses "
        "against final matched-LS EKT-GL in 469/223/148 task-seed pairs. "
        "This establishes the starting archive gap. The causal effect on the final result is measured by "
        "the separate matched-initialization intervention. [Initialization audit](../05_initialization/initialization_report.md)", "",
        "**Observed transfer and local-search signals.** The native V2 archive records 329 accepted events "
        "and 287 strict archive improvements over 25,200 task-transfer attempts. It records 2,590 construction "
        "dead ends. The approximately 99.44% truncation figure is a union of three independent flags "
        "on processed cuts. Low acceptance and cut truncation are descriptive observations. "
        "[Local-search audit](../03_version_diff/local_search_audit.md)", "",
        "| Hypothesis | Evidence at current coverage |",
        "|---|---|",
    ]
    topics = [
        ("Legacy neighborhood improves quality", "A1 vs A0"),
        ("Matched initialization improves EKT-GL", "Matched-init vs standard ektgl"),
        ("32-dimensional network improves quality", "A2_32 vs A0"),
        ("Sampled balanced loss improves quality", "A3 vs A0"),
        ("Class balancing alone improves quality", "G2 vs A0"),
        ("Incremental effect of negative subsampling after balancing", "A3 vs G2"),
        ("Legacy features improve quality", "A4 vs A0"),
        ("Cost-aware decoder improves quality", "D1 vs A0"),
        ("Trained model adds benefit to same-cap beam", "D3 vs D1"),
        ("Cross-task attention provides benefit", "A8_identity vs A0"),
        ("Learned scores affect greedy performance", "A8_shuffled vs A0"),
    ]
    for topic, comparison in topics:
        row = by_test.get(comparison)
        if row:
            effect = float(row["effect_pct"])
            if comparison in ("A8_identity vs A0", "A8_shuffled vs A0"):
                useful = effect < 0
            else:
                useful = effect > 0
            shuffled = by_test.get("D3 vs D2")
            if comparison == "D3 vs D1" and shuffled:
                useful = useful and float(shuffled["effect_pct"]) > 0
            classification = "Supported hypothesis" if useful else "Unsupported in this pilot"
            if comparison == "A3 vs G2":
                classification = "Descriptive incremental effect"
            text = (f"{classification}. First arm of {comparison} gives {effect:+.3f}% "
                    f"paired relative gain. W/T/L {row['wins']}/{row['ties']}/{row['losses']}.")
            if comparison == "D3 vs D1" and shuffled:
                text += (f" Against the shuffled same-cap control, W/T/L is "
                         f"{shuffled['wins']}/{shuffled['ties']}/{shuffled['losses']}.")
        else:
            text = "Unresolved. A full paired estimate is pending."
        lines.append(f"| {topic} | {text} |")
    lines += ["", "Three seed-level observations per full pilot contrast limit statistical power. Mechanism attribution "
              "is restricted to the settings of the intervention. Architectures, features and neighborhoods can interact.",
              "", "[Paired effects by task](../06_ablation/ablation_effects_by_task.csv) · "
              "[Coverage](../coverage.json) ? [Matched-generation and transfer mechanisms](CONTROLLED_MECHANISMS.md)", ""]
    write(root / "14_reports/ROOT_CAUSE_ANALYSIS.md", "\n".join(lines))


def independent_confirmation(root):
    old = observations(root / "04_cross_version/raw_results.csv")
    new = observations(root / BASE / "full_raw_results.csv")
    summary = observations(root / "04_cross_version/summary_by_task.csv")
    scale = {r["task"]: max(abs(float(r["AVG"])), 1.)
             for r in summary if r["method"] == "native:mfea"}
    if len(scale) != 42 or not new:
        return []
    rng = np.random.default_rng(4320)
    comparisons = []
    for profile in ("native", "matched_ls"):
        paired_costs = []
        for cohort in (new, old):
            by_seed = defaultdict(dict)
            for row in cohort:
                if row["profile"] == profile and row["method"] == "ektgl":
                    by_seed[int(row["seed"])][row["task"]] = float(row["cost"]) / scale[row["task"]] * 100
            if len(by_seed) != 20 or any(len(costs) != 42 for costs in by_seed.values()):
                break
            paired_costs.append(np.array([np.mean(list(c.values())) for c in by_seed.values()]))
        if len(paired_costs) != 2 or any(not np.all(np.isfinite(v)) for v in paired_costs):
            continue
        fresh, historical = paired_costs
        test = mannwhitneyu(fresh, historical, alternative="two-sided")
        draws = rng.choice(historical, (10000, 20)).mean(1) - rng.choice(fresh, (10000, 20)).mean(1)
        low, high = np.quantile(draws, [.025, .975])
        comparisons.append(dict(profile=profile, V3_runs=len(fresh), V2_runs=len(historical),
                                V3_mean_normalized_cost=fresh.mean(), V2_mean_normalized_cost=historical.mean(),
                                V3_gain_points=historical.mean()-fresh.mean(), ci_lower=low, ci_upper=high,
                                rank_biserial_favors_V3=1-2*test.statistic/(len(fresh)*len(historical)),
                                mann_whitney_p=test.pvalue, p_holm=None))
    previous = 0
    for i, r in enumerate(sorted(comparisons, key=lambda r: r["mann_whitney_p"])):
        previous = max(previous, min(1.0, (2-i)*r["mann_whitney_p"]))
        r["p_holm"] = previous
    put(root/BASE/"independent_ekt_version_tests.csv", comparisons)
    return comparisons


def workload_summary(root):
    source = root / BASE / "suite_workload.csv"
    groups = defaultdict(list)
    for row in observations(source):
        groups[row["profile"], row["method"]].append(row)
    keys = ("wall_clock_s", "actual_generations", "objective_evaluation_requests",
            "actual_objective_computations", "train_time_s", "inference_time_s",
            "local_search_time_s", "ls_evaluations", "transfer_evaluations")
    events = [datetime.fromisoformat(r["utc"]) for r in
              observations(root / "12_logs/confirmation_host_power_events.csv")]
    summary, timing_flags = [], []
    for (profile, method), completed in sorted(groups.items()):
        if len(completed) != 20:
            continue
        row = dict(profile=profile, method=method, runs=len(completed))
        row.update({"mean_"+k: np.mean([float(r.get(k) or 0) for r in completed]) for k in keys})
        seconds = np.array([float(r["wall_clock_s"]) for r in completed])
        typical = float(np.median(seconds))
        unflagged = []
        for run, elapsed in zip(completed, seconds):
            crosses_event = run.get("started_at") and any(
                datetime.fromisoformat(run["started_at"]) <= event <= datetime.fromisoformat(run["completed_at"])
                for event in events)
            outlier = bool(elapsed > 5 * typical)
            if crosses_event or outlier:
                timing_flags.append(dict(profile=profile, method=method, seed=run["seed"],
                                         wall_clock_s=elapsed, host_power_event=bool(crosses_event),
                                         exceeds_five_times_median=outlier))
            else:
                unflagged.append(elapsed)
        row.update(median_wall_clock_s=typical, timing_flag_count=len(seconds)-len(unflagged),
                   unflagged_mean_wall_clock_s=np.mean(unflagged) if unflagged else None)
        summary.append(row)
    put(root/BASE/"workload_summary.csv", summary)
    put(root/BASE/"runtime_flags.csv", timing_flags)
    return summary


def describe_final(root, matrix):
    config = root / "13_configs/frozen_v3.json"
    coverage = json_read(root / "experiment_status.json")
    chosen = json_read(config)["selected_arm"] if config.is_file() else None
    holdout = independent_confirmation(root)
    workload = workload_summary(root)
    lines = [
        "# Confirmation benchmark", "",
        "The suite has 42 directed IDPC-DU multigraphs, including 24 small instances and 18 large "
        "instances. The fixed canonical EA and MFEA sources were audited and hashed before the pilot. "
        "Confirmation uses seeds 4300–4319. Native has a limit of 300 generations. Matched-LS shares "
        "the 2-opt implementation and objective cap of 379,260. Matched-init-LS additionally evaluates a "
        "common USS source pool while retaining canonical MFEA initialization and skill assignment.", "",
        f"The frozen V3 selection is {chosen or 'pending'}.",
        f"Confirmation checkpoint counts are {coverage['checkpoints']['confirmation']}.", "",
    ]
    if matrix:
        lines += ["| Profile | First method | Second method | AVG W/T/L |",
                  "|---|---|---|---|"]
        for r in matrix:
            if r["metric"] == "AVG" and r["scope"] == "all42":
                lines.append(f"| {r['profile']} | {LABELS[r['method_a']]} | {LABELS[r['method_b']]} | "
                             f"{r['wins']}/{r['ties']}/{r['losses']} |")
        lines += ["", "[Per-instance table](../09_final_benchmark/publication_table_all42.csv) · "
                  "[Paired tests](../09_final_benchmark/statistical_tests.csv) · "
                  "[Workload](../09_final_benchmark/suite_workload.csv)", ""]
    else:
        lines += ["A publication table requires all twenty seeds for each method in a profile. "
                  "The publication table is pending.", ""]
    confirm_stats = [r for r in observations(root/BASE/"statistical_tests.csv")
                     if r["comparison"] == "ektgl vs mfea" and r["complete"] == "True"]
    if confirm_stats:
        lines += ["## Interpretation", ""]
        for row in confirm_stats:
            if not row["effect_pct"]:
                continue
            effect = float(row["effect_pct"])
            profile = row["profile"]
            interval = f"[{float(row['lower_95']):+.2f}%, {float(row['upper_95']):+.2f}%]"
            gain = f"{effect:+.2f}%"
            lines.append(f"- **{profile}** gives {gain} mean relative gain for V3 against canonical MFEA "
                         f"with seed bootstrap interval {interval}. "
                         f"The signed-rank Holm p-value is {float(row['p_holm']):.4g}.")
        lines += ["", "Native V3 uses the N3 heuristic and has its own measured objective workload. "
                  "Matched-LS applies V2 2-opt to all three methods. Matched-init-LS also matches the "
                  "source initialization pool and archive. The MFEA active population uses its canonical skill "
                  "assignment in both matched profiles. These conditions answer distinct comparisons.", "",
                  "The trained beam tied the shuffled beam on all 42 pilot instance averages. "
                  "The improvement adopted in native V3 is bounded cost-aware subpath search. The role "
                  "of state-conditioned neural scores remains unresolved.", "",
                  "[Same-cap decoder controls](../08_v3_pilot/beam_search/replay_summary.csv) ? "
                  "[Initialization charges and initial archives](../09_final_benchmark/matched_init_ls/initialization_pool_audit.csv)", ""]
    if workload:
        lines += ["## Computational workload", "",
                  "Every value averages twenty suites. Native has a shared generation ceiling and records "
                  "the actual objective workload of each method. Matched profiles account for the same cap.",
                  "",  "| Profile | Method | Requests/suite | Computations/suite | Mean time (s) | Median time (s) | Timing flags |",
                  "|---|---|---:|---:|---:|---:|---:|"]
        for row in workload:
            lines.append(f"| {row['profile']} | {LABELS[row['method']]} | "
                         f"{row['mean_objective_evaluation_requests']:,.0f} | "
                         f"{row['mean_actual_objective_computations']:,.0f} | {row['mean_wall_clock_s']:.2f} | "
                         f"{row['median_wall_clock_s']:.2f} | {row['timing_flag_count']} |")
        lines += ["", "CPU search and CUDA workers ran concurrently on the experiment host. Wall-clock times "
                   "describe the recorded workload on that host. Windows power transitions occurred during execution. "
                  "The timing audit flags runs spanning a recorded event or exceeding five times their group "
                  "median. Quality, budget and feasibility keep every seed. "
                  "[Workload summaries](../09_final_benchmark/workload_summary.csv) ? "
                  "[Convergence by generation and objective requests](../10_figures/index.md)", ""]
    if holdout:
        lines += ["## EKT-GL V3 versus archived EKT-GL V2", "",
                  "Independent seed cohorts, with per-task normalization fixed using archived native MFEA AVG. "
                  "These supplementary tests compare the same V2 profile to the corresponding V3 profile.", "",
                  "| Profile | V3 gain (normalized points) | 95% bootstrap interval | Holm p |",
                  "|---|---:|---|---:|"]
        for row in holdout:
            lines.append(f"| {row['profile']} | {row['V3_gain_points']:.3f} | "
                         f"[{row['ci_lower']:.3f}, {row['ci_upper']:.3f}] | {row['p_holm']:.4g} |")
        lines += ["", "[Independent version comparisons](../09_final_benchmark/independent_ekt_version_tests.csv)", ""]
    lines += ["The paired Wilcoxon test uses per-seed mean task-relative effects. Holm correction is "
              "applied within the three prespecified comparisons in each profile. The Friedman test "
              "uses instance AVG ranks and is exploratory. Runtime includes model training and path "
              "construction. Objective request totals alone do not measure total compute.", "",
              "[Coverage](../coverage.json) · [Protocol](../13_configs/preregistered_pilot.json)", ""]
    write(root / "14_reports/FINAL_BENCHMARK_REPORT.md", "\n".join(lines))


def publish_frozen_algorithm(root):
    path = root / "13_configs/frozen_v3.json"
    if not path.is_file():
        return
    frozen = json_read(path)
    arm = frozen["selected_arm"]
    heading = "## Frozen confirmation configuration"
    report = root / "14_reports/EKT_GL_V3_ALGORITHM.md"
    original = report.read_text(encoding="utf-8").split(heading, 1)[0].rstrip()
    notes = [original, "", heading, "",
             f"Pilot selection adopted **{arm}** from the documented interventions.", "",
             "| Profile | Local search | Embedding width | Training policy | Feature policy | Decoder |",
             "|---|---|---:|---|---|---|"]
    for profile, row in frozen["effective_configs"].items():
        v = row["variant"]
        notes.append(f"| {profile} | {v['ls']} | {v['width']} | {v['loss']} | "
                     f"{v['features']} | {v['decoder']} |")
    if frozen.get("native_subpath_memoization"):
        notes += ["", "Native subpath search uses a 32,768-entry cache of exact cost-only search results. "
                  "The key includes the graph, target bound, search caps and full DU/visit state. Each "
                  "logical path still goes through the same objective ledger and round-trip. Physical "
                  "ranking reads and expansions have separate counters. "
                  "[Exact trajectory parity](../08_v3_pilot/local_search/n3_memoization_parity.json)"]
    notes += ["", "[Frozen source, seeds and effective configurations](../13_configs/EKT_GL_V3_CONFIG.yaml)", ""]
    write(report, "\n".join(notes))
    changelog = root / "14_reports/EKT_GL_V3_CHANGELOG.md"
    previous = changelog.read_text(encoding="utf-8").split(heading, 1)[0].rstrip()
    write(changelog, f"{previous}\n\n{heading}\n\n**{arm}** is selected for confirmation on seeds "
          "4300–4319. Selection used only paired pilot outcomes and the preregistered decoder controls. "
          "[Selection evidence](../08_v3_pilot/configuration_selection.md)\n")


if __name__ == "__main__":
    publish_frozen_algorithm(RESEARCH)
    root_causes(RESEARCH)
    describe_final(RESEARCH, publication(RESEARCH))
