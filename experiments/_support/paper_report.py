"""Build measured Tables II-IV and Figures 7-8 from the complete paper suite."""
import csv
import json
import math
from pathlib import Path

from experiments.metrics import comparison_metrics, normalized_convergence
from experiments.run_mfea import load_reference, write_csv


def read_summaries(directory):
    with (Path(directory)/"summary.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key in ("BF", "AVG", "STD_sample", "seconds_mean", "paper_BF", "paper_AVG"):
            row[key] = float(row[key]) if row[key] else None
        for key in ("runs", "feasible"):
            row[key] = int(row[key])
        for key in ("matches_BF", "matches_AVG"):
            row[key] = row[key] == "True"
    return rows


def checkpoint_rows(directory):
    manifest = json.loads((Path(directory)/"manifest.json").read_text(encoding="utf-8"))
    rows = []
    for path in sorted((Path(directory)/"checkpoints").glob("*.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        if result["fingerprint"] != manifest["fingerprint"]:
            raise ValueError(f"Mismatched checkpoint: {path}")
        rows.extend(result["rows"])
    return rows


def write_convergence(directory, reference):
    directory = Path(directory)
    manifest = json.loads((directory/"manifest.json").read_text(encoding="utf-8"))
    extrema, individual, averaged = normalized_convergence(checkpoint_rows(directory), manifest["config"]["generations"])
    write_csv(directory/"normalized_per_task.csv", individual)
    write_csv(directory/"normalized_average.csv", averaged)
    metadata = dict(formula="(f - minimum) / (maximum - minimum)",
        extrema_source="All evaluated finite candidate objectives across both EA and MFEA and all runs, including initialization",
        aggregation="Normalize, average tasks within each seed, then average seeds",
        constant_task_policy="0 when maximum == minimum",
        infeasible_policy="Exclude infinite candidates only from finite extrema; an infinite best remains missing and invalidates its aggregate mean",
        reference=reference["convergence"], tasks=extrema)
    (directory/"normalization.json").write_text(json.dumps(metadata, indent=2)+"\n", encoding="utf-8")
    return metadata, individual, averaged


def plot_convergence(output, individual, averaged, generations):
    # Plotting is optional for the solver, required for the full figure export.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"EA": "#d85c41", "MFEA": "#2464a6"}
    names = {"EA": "ST (EA)", "MFEA": "MT (MFEA)"}
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 180, "svg.fonttype": "none"})
    def points(rows):
        return ([r["generation"] for r in rows],
                [r["normalized_mean"] if r["normalized_mean"] is not None else math.nan for r in rows])
    fig, ax = plt.subplots(figsize=(7.5, 4.2), constrained_layout=True)
    for algorithm in ("EA", "MFEA"):
        rows = [r for r in averaged if r["algorithm"] == algorithm]
        ax.plot(*points(rows), label=names[algorithm], color=colors[algorithm], linewidth=1.8)
    ax.set(xlabel="Generation", ylabel="Mean normalized best objective", xlim=(0, generations),
           title="Figure 7 experiment: measured EA and MFEA convergence")
    ax.grid(alpha=.2); ax.legend()
    for extension in ("png", "svg"):
        fig.savefig(output/f"figure_7.{extension}")
    plt.close(fig)
    tasks = sorted({r["instance"] for r in individual}, key=lambda name: int(name.split("_")[1].split("x")[0]), reverse=True)
    fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    styles = [("#d85c41", "-"), ("#7aa20a", "--"), ("#00a6b4", "-"), ("#aa66cc", "--")]
    for index, (algorithm, task) in enumerate((a, t) for a in ("EA", "MFEA") for t in tasks):
        rows = [r for r in individual if r["algorithm"] == algorithm and r["instance"] == task]
        color, style = styles[index % len(styles)]
        ax.plot(*points(rows), label=f"{names[algorithm]}: {task.removesuffix('.idpc')}",
                color=color, linestyle=style, linewidth=1.8)
    ax.set(xlabel="Generation", ylabel="Normalized best objective", xlim=(0, generations),
           title="Figure 8 experiment: measured convergence of both tasks")
    ax.grid(alpha=.2); ax.legend(fontsize=8)
    for extension in ("png", "svg"):
        fig.savefig(output/f"figure_8.{extension}")
    plt.close(fig)


def build_paper_report(output, make_plots=True):
    output = Path(output)
    reference = load_reference()
    benchmark = output/"benchmark"
    convergence = output/"convergence"
    manifest = json.loads((benchmark/"manifest.json").read_text(encoding="utf-8"))
    config = manifest["config"]
    summaries = read_summaries(benchmark)
    lookup = {(r["algorithm"], r["instance"]): r for r in summaries}
    expected = {(a, name) for a in ("EA", "MFEA") for name in reference["instances"]}
    if set(lookup) != expected or len(summaries) != len(expected):
        raise ValueError("Full benchmark requires exactly both algorithms on all 42 original instances")
    if any(r["runs"] != config["runs"] for r in summaries):
        raise ValueError("Incomplete run counts")
    metrics = comparison_metrics(summaries, reference)
    write_csv(output/"table_II.csv", metrics)
    wide_tables = {"set1": [], "set2": []}
    for name, ref in reference["instances"].items():
        row = dict(instance=name)
        for algorithm in ("EA", "MFEA"):
            actual = lookup[(algorithm, name)]
            for field in ("BF", "AVG", "STD_sample", "seconds_mean", "feasible"):
                row[f"{algorithm}_{field}"] = actual[field]
            for field in ("BF", "AVG", "seconds"):
                row[f"paper_{algorithm}_{field}"] = ref[algorithm][field]
            row[f"{algorithm}_matches_BF_AVG"] = actual["matches_BF"] and actual["matches_AVG"]
        ea, mt = row["EA_AVG"], row["MFEA_AVG"]
        row["RPD"] = 100*(ea-mt)/ea if math.isfinite(ea) and math.isfinite(mt) and ea > 0 else None
        wide_tables[ref["set"]].append(row)
    write_csv(output/"table_III.csv", wide_tables["set1"])
    write_csv(output/"table_IV.csv", wide_tables["set2"])
    normalization, individual, averaged = write_convergence(convergence, reference)
    if set(normalization["tasks"]) != set(reference["convergence"]["instances"]):
        raise ValueError("C-experiment task pair differs from the published pair")
    if make_plots:
        plot_convergence(output, individual, averaged, config["generations"])
    matches = {a: sum(r["matches_BF"] and r["matches_AVG"] for r in summaries if r["algorithm"] == a) for a in ("EA", "MFEA")}
    finite = {a: sum(r["feasible"] for r in summaries if r["algorithm"] == a) for a in ("EA", "MFEA")}
    published_setup = all(config.get(k) == v for k, v in reference["parameters"].items())
    summary = dict(full_published_parameters=published_setup, instances=42, runs=config["runs"],
                   matching_BF_AVG_rows=matches, feasible_benchmark_runs=finite,
                   benchmark_path_records=len(summaries)*config["runs"],
                   convergence_path_records=4*config["runs"],
                   all_table_statistics_match=all(matches[a] == 42 for a in matches),
                   table_II=metrics, plots_exported=make_plots,
                   normalization=normalization)
    (output/"metrics.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
    lines = ["# Full CEC 2020 experiment: measured results", "",
             "All four component algorithms are used: GPA, random initialization, PMX/TPX crossover, and swap/reset mutation.",
             "EA optimizes each instance independently; MFEA jointly optimizes inferred adjacent-size pairs. No pruning, GNN, oracle answers, local search or repair enters either search.", "",
             f"Setup: {config['runs']} independent seeds per scenario; population {config['population']}, generations {config['generations']}, rmp {config['rmp']}, mutation rate {config['mutation_rate']}.",
             f"Published parameter values used: **{published_setup}**. Benchmark: 42 instances, 84 algorithm/instance rows, {summary['benchmark_path_records']} path records.",
             f"C-experiment: the published 100x100 + 50x100 pair, both EA controls and joint MFEA, {summary['convergence_path_records']} path records.", "",
             "## Agreement with Tables III and IV", "",
             "| Method | BF and printed AVG both match | Feasible runs |", "|---|---:|---:|"]
    for a in ("EA", "MFEA"):
        lines.append(f"| {a} | {matches[a]}/42 | {finite[a]}/{42*config['runs']} |")
    lines += ["", "**Running the complete experiment does not establish reproduction of the published numbers.** Numerical agreement is reported above; remaining gaps are visible in the tables.", "",
              "## Table II comparison", "",
              "| Set | NIB | NIE | NIW | Max RPD % | Mean RPD all % | Mean RPD improved % | Paper NIB/NIE | Paper max/mean RPD % |",
              "|---|---:|---:|---:|---:|---:|---:|---|---|"]
    def number(value):
        return f"{value:.3f}" if value is not None else "undefined"
    for r in metrics:
        lines.append(f"| {r['subset']} | {r['NIB']} | {r['NIE']} | {r['NIW']} | {number(r['maximum_RPD'])} | {number(r['average_RPD'])} | {number(r['average_RPD_improved'])} | {r['paper_NIB']}/{r['paper_NIE']} | {r['paper_maximum_RPD']}/{r['paper_average_RPD']} |")
    lines += ["", "RPD = 100*(EA AVG - MFEA AVG)/EA AVG. Mean-all includes every finite pair; mean-improved includes only strict MFEA improvements.",
              "The printed reference rows yield mean-all 9.948% / 4.296% and mean-improved 13.264% / 4.549%, whereas Table II prints 13.2% / 4.6%. The article does not specify the averaging subset or publish unrounded run values, so this discrepancy is reported rather than forced to match."]
    url = reference["convergence"]["normalization_reference_url"]
    lines += ["", "## C-experiment and figures", "",
              "Normalize each task as `(f - f_min)/(f_max - f_min)` using the finite minimum and maximum of all evaluated candidates from both methods and all runs; then average tasks and seeds.",
              f"This follows the normalization described in [Gupta, Ong and Feng, reference 4, page 10]({url}). Extrema are measured after the search, never used as optimization guidance.",
              "The zero-range convention maps a constant task to zero. Failed best values stay missing, invalidate the aggregate, and are never silently discarded.",
              "Raw and normalized curves are exported even when they disagree with the paper. Generations 0..500 are used, without smoothing, selecting favorable seeds or copying figure coordinates.", "",
              "- [Table II](table_II.csv), [Table III](table_III.csv), [Table IV](table_IV.csv).",
              "- [Benchmark report](benchmark/REPORT.md), [all final paths](benchmark/paths.json).",
              "- [C-experiment report](convergence/REPORT.md), [raw curves](convergence/convergence.csv).",
              "- [Per-task normalized curves](convergence/normalized_per_task.csv), [averaged normalized curves](convergence/normalized_average.csv), [normalization extrema](convergence/normalization.json).",
              "- [Independent validation](validation.json), [summary metrics](metrics.json)."]
    if make_plots:
        lines += ["- [Figure 7 PNG](figure_7.png), [SVG](figure_7.svg); [Figure 8 PNG](figure_8.png), [SVG](figure_8.svg).", "",
                  "![Measured Figure 7 experiment](figure_7.png)", "", "![Measured Figure 8 experiment](figure_8.png)"]
    lines += ["", "## Reproducibility limits", "",
              "Algorithm 1 is interpreted literally: arriving-node sigma, literal modulo, and original file edge order. Sigma at a node without outgoing edges is stored as numeric 1, resolving the article's undefined '-' modulo case.",
              "Adjacent task pairing is inferred from repeated reported times, not author-confirmed. The convergence pair is stated explicitly in the paper. Seeds are 100..129 by default; author seeds are unavailable.",
              "The framework follows reference [4]'s assortative mating, skill inheritance, selective evaluation and elitist scalar-fitness selection, with the CEC paper's custom representation/operators. Parent sampling, extra mutation scheduling and population allocation remain documented implementation choices.",
              "Unevaluated tasks are excluded from rank calculation, equivalent to assigning infinite cost when feasible evaluated solutions exist; this differs from random ranking of unevaluated infinite costs in infeasible-only populations.",
              "Population 100 is shared by a multitask group. Each EA control has 100 individuals of its own. Evaluation counts are exported per task; no claim of equal aggregate evaluation budgets is made.",
              "Rs values are measured Python solver time; MFEA time is shared across tasks in a group. They are not directly comparable with the paper's Java/i7 runtime. Loading graphs and report generation are outside solver timings.",
              "Only the complete benchmark and published convergence scenario are retained; there is no parameter grid or extra method outside the paper.", ""]
    (output/"REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return summary
