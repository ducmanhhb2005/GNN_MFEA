"""Produce publication figures from archived and controlled IDPC-DU observations."""

from collections import defaultdict
from itertools import groupby
from pathlib import Path
import argparse
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1] / "results/idpc_v3_research"
COLORS = {"Legacy EKT": "#6e7d8e", "V2 EKT": "#b27237", "V3 EKT": "#237d69",
          "V2 MFEA": "#42568b", "A0": "#b27237", "A1": "#33588d", "D1": "#57658d",
          "D2": "#8d8675", "D3": "#237d69", "ea": "#858178", "mfea": "#42568b"}
HISTORY = "04_cross_version/summary_by_task.csv"
HISTORY_RAW = "04_cross_version/raw_results.csv"
INIT = "05_initialization/initialization_by_task_seed.csv"
FINAL = "09_final_benchmark/summary_by_task.csv"
FINAL_RAW = "09_final_benchmark/full_raw_results.csv"
FINAL_WTL = "09_final_benchmark/win_tie_loss.csv"
FINAL_WORK = "09_final_benchmark/suite_workload.csv"
TRACE = "10_figures/convergence_source.csv"
STATS = "08_v3_pilot/pilot_results.csv"
WORK = "06_ablation/suite_workload.csv"
EVENTS = "07_transfer_diagnostics/transfer_events.csv"
LEARN = "07_transfer_diagnostics/learning_metrics.csv"
ATTENTION = "07_transfer_diagnostics/attention_diagnostics.csv"


def group(rows, key):
    return {k: list(items) for k, items in groupby(sorted(rows, key=key), key)}


def v(row, key):
    return float(row[key]) if row.get(key) else np.nan


def coords(task):
    return tuple(int(x) for x in task.split("_")[-1].split("x"))


def label(task):
    n, d, _ = coords(task)
    return f"{n}×{d}"


def avg(values):
    values = [x for x in values if np.isfinite(x)]
    return np.mean(values) if values else np.nan


class Ledger:
    def __init__(self, root):
        self.root = root
        self.folder = root / "10_figures"
        self.folder.mkdir(parents=True, exist_ok=True)
        self.captions = []

    def read(self, name):
        path = self.root / name
        if not path.exists():
            return []
        with path.open(encoding="utf-8-sig", newline="") as file:
            return list(csv.DictReader(file))

    def save(self, fig, name, title, detail, *sources):
        fig.suptitle(title, fontsize=13, x=.05, ha="left")
        fig.tight_layout()
        for suffix in ("png", "pdf"):
            fig.savefig(self.folder / f"{name}.{suffix}", dpi=300, bbox_inches="tight")
        plt.close(fig)
        refs = " · ".join(f"[{Path(p).name}](../{p})" for p in sources)
        self.captions.append(f"- [{title}]({name}.pdf). {detail} {refs}")

    def finish(self):
        intro = "# Figures\n\nReproduce with python -m experiments.plot_idpc_v3. Each figure cites its input CSV. PNGs use 300 DPI.\n\n"
        (self.folder / "index.md").write_text(intro + "\n".join(self.captions) + "\n", encoding="utf-8")


def task_ticks(ax, tasks):
    ax.set_xticks(range(len(tasks)), [label(t) for t in tasks], rotation=90, fontsize=7)
    ax.set_xlabel("Instance (vertices × domains)")


def history(book):
    codes = {"legacy:ektgl": "Legacy EKT", "native:ektgl": "V2 EKT", "native:mfea": "V2 MFEA"}
    data = [{**r, "series": codes[r["method"]]} for r in book.read(HISTORY) if r["method"] in codes]
    final = [{**r, "series": "V3 EKT"} for r in book.read(FINAL)
             if r["profile"] == "native" and r["method"] == "ektgl"]
    series = {s: {r["task"]: r for r in rows} for s, rows in group(data + final, lambda r: r["series"]).items()}
    tasks = sorted(series["V2 MFEA"], key=coords)
    refs = (HISTORY, FINAL) if final else (HISTORY,)
    for i, metric in enumerate(("AVG", "BF", "STD"), 3):
        fig, ax = plt.subplots(figsize=(12, 4.4))
        for name, task_rows in series.items():
            ax.plot([v(task_rows.get(task, {}), metric) for task in tasks], "o-", ms=3, lw=1,
                    color=COLORS[name], label=name)
        task_ticks(ax, tasks)
        ax.set_ylabel(f"{metric} (path cost)")
        ax.legend(frameon=False, ncols=len(series))
        book.save(fig, f"{i:02d}_{metric.lower()}", f"{metric} by instance",
                  "One point per instance and method. Historical settings differ.", *refs)
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    ax.bar(list(series), [avg(v(r, "AVG") for r in rows.values()) for rows in series.values()],
           color=[COLORS[s] for s in series])
    ax.set_ylabel("Mean of instance AVG (path cost)")
    book.save(fig, "01_versions", "Solution quality across versions",
              "Every available task contributes equally to its method average.", *refs)
    source = "04_cross_version/win_tie_loss.csv"
    rows = [r for r in book.read(source) if r["scope"] == "all" and r["metric"] == "avg"]
    confirmed = [r for r in book.read(FINAL_WTL) if r["complete"] == "True"]
    historical_names = {"legacy:ektgl": "Legacy EKT", "legacy:mfea": "Legacy MFEA",
                        "legacy:individual": "Legacy Individual", "legacy:sto": "Legacy STO",
                        "native:ektgl": "Native EKT V2", "native:mfea": "Native MFEA V2", "native:ea": "Native EA",
                        "matched_ls:ektgl": "Matched-LS EKT V2", "matched_ls:mfea": "Matched-LS MFEA V2"}
    displayed = [{**r, "comparison": historical_names[r["method_a"]] + " vs " + historical_names[r["method_b"]]}
                 for r in rows]
    groups = [("Historical", displayed)]
    if confirmed:
        labels = {"native": "Native", "matched_ls": "Matched-LS", "matched_init_ls": "Matched-init-LS"}
        final_names = {"ektgl": "EKT V3", "mfea": "MFEA V2", "ea": "EA"}
        final_labels = (" vs ".join(final_names[w] for w in r["comparison"].split(" vs ")) for r in confirmed)
        groups.append(("Independent confirmation",
                       [dict(comparison=labels[r["profile"]] + " · " + comparison,
                             win=r["wins"], tie=r["ties"], loss=r["losses"])
                        for r, comparison in zip(confirmed, final_labels)]))
    fig, subplots = plt.subplots(1, len(groups), figsize=(8.2*len(groups), 5.1), squeeze=False)
    for ax, (title, values) in zip(subplots[0], groups):
        offset = np.zeros(len(values))
        for name, color in (("win", "#237d69"), ("tie", "#bbb7ac"), ("loss", "#ad6343")):
            count = [int(r[name]) for r in values]
            bars = ax.barh([r["comparison"] for r in values], count, left=offset, color=color, label=name)
            ax.bar_label(bars, labels=[str(n) if n >= 4 else "" for n in count], label_type="center",
                         color="#201f1c" if name == "tie" else "#ffffff", fontsize=8)
            offset += count
        ax.set_xlabel("Number of instances")
        ax.set_title(title, loc="right")
        ax.legend(frameon=False, ncols=3, loc="lower left", bbox_to_anchor=(0, 1.0), borderaxespad=0)
    book.save(fig, "02_wtl", "Win, tie and loss by instance AVG",
              "Each bar counts instances for the named comparison in the source.",
              source, *((FINAL_WTL,) if confirmed else ()))
    base = {t: v(r, "AVG") for t, r in series["V2 MFEA"].items()}
    gap = lambda cost, t: 100 * (cost - base[t]) / max(abs(base[t]), 1)
    matrix = np.array([[gap(v(scores.get(t, {}), "AVG"), t) for t in tasks] for scores in series.values()])
    fig, ax = plt.subplots(figsize=(12, 3.8))
    scale = np.nanmax(np.abs(matrix)) or 1.
    image = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-scale, vmax=scale)
    ax.set_yticks(range(len(series)), list(series))
    task_ticks(ax, tasks)
    fig.colorbar(image, ax=ax, label="Gap to V2 MFEA (%)")
    book.save(fig, "11_heatmap", "Solution quality relative to V2 MFEA",
              "Cells compare instance AVG. Lower values are better.", *refs)
    for name, xlabel, axis in (
        ("12_size", "Vertices", lambda task: coords(task)[0]),
        ("13_ratio", "Domain ratio class", lambda task: .5 if coords(task)[1] == coords(task)[0] // 2
         else coords(task)[1] / coords(task)[0]),
    ):
        fig, ax = plt.subplots(figsize=(7.3, 4.1))
        for method, scores in series.items():
            by_x = group(list(scores), axis)
            ax.plot(list(by_x), [avg(gap(v(scores[t], "AVG"), t) for t in ts) for ts in by_x.values()],
                    "o-", color=COLORS[method], label=method, ms=4)
        ax.set(xlabel=xlabel, ylabel="Mean gap to V2 MFEA (%)")
        ax.legend(frameon=False, ncols=2)
        book.save(fig, name, f"Solution quality by {xlabel.lower()}",
                  "The mean uses the instances in each size or ratio class.", *refs)
    raw = [{**r, "series": codes[r["profile"] + ":" + r["method"]]} for r in book.read(HISTORY_RAW)
           if r["profile"] + ":" + r["method"] in codes]
    confirm_raw = [{**r, "series": "V3 EKT"} for r in book.read(FINAL_RAW)
                   if r["profile"] == "native" and r["method"] == "ektgl"]
    runs = group(raw + confirm_raw, lambda r: (r["series"], int(r["seed"])))
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    labels = list(dict.fromkeys(m for m, _ in runs))
    values = [[avg(gap(v(r, "cost"), r["task"]) for r in rows)
               for (method, _), rows in runs.items() if method == m] for m in labels]
    ax.boxplot(values, tick_labels=labels)
    ax.set_ylabel("Per-seed relative cost gap (%)")
    book.save(fig, "10_seed_boxplots", "Distribution across seeds",
              "Each observation is a seed mean relative to V2 MFEA instance AVG.", HISTORY_RAW, HISTORY,
              *((FINAL_RAW,) if confirm_raw else ()))


def initial(book):
    groups = group(book.read(INIT), lambda r: r["method"])
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = np.arange(len(groups))
    for i, (key, name) in enumerate((("evaluated", "Evaluated"), ("active_count", "Active slots"))):
        ax.bar(x + (i-.5)*.35, [avg(v(r, key) for r in rows) for rows in groups.values()],
               width=.35, label=name)
    ax.set_xticks(x, list(groups))
    ax.set_yscale("log")
    ax.set_ylabel("Mean chromosomes per task (log scale)")
    ax.legend(frameon=False)
    book.save(fig, "06_initialization", "Chromosomes evaluated and retained at initialization",
              "MFEA evaluates all USS chromosomes on every task before skill assignment.", INIT)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for profile in ("native", "matched_ls"):
        reductions = [avg((v(r, "initial_archive_cost")-v(r, profile+"_final")) /
                          max(abs(v(r, "initial_archive_cost")), 1)*100 for r in rows)
                      for rows in groups.values()]
        ax.plot(list(groups), reductions, "o-", label=profile)
    ax.set_ylabel("Mean relative cost reduction (%)")
    ax.legend(frameon=False)
    book.save(fig, "07_initialization_to_final", "Archive improvement after initialization",
              "Per-task, per-seed reductions are computed from its generation-zero archive.", INIT)


def convergence(book):
    raw = book.read(TRACE)
    final = [r for r in raw if r["phase"] == "confirmation"
             and r["profile"] in ("native", "matched_ls")]
    data = final or [r for r in raw if r["profile"] == "matched_ls" and r["phase"] == "pilot"]
    groups = group(data, lambda r: (r["profile"], r["method"] if r["method"] != "ektgl" or final else r["arm"],
                                    int(r["seed"]), int(r["generation"])))
    traces = defaultdict(list)
    for (profile, method, seed, generation), rows in groups.items():
        if len(rows) == 42:
            traces[profile, method,seed].append((generation, int(rows[0]["suite_requests"]),
                                                avg(v(r,"cost") for r in rows)))
    if not traces:
        return
    profiles = sorted({p for p, _, _ in traces}, reverse=True)
    cohort = "Confirmation" if final else "Pilot"
    for axis, key, xlabel in ((0, "08_convergence_generation", "Generation"),
                              (1, "09_convergence_budget", "Suite objective requests")):
        fig, axes = plt.subplots(1, len(profiles), figsize=(7.8*len(profiles),4.3), squeeze=False)
        for profile, ax in zip(profiles, axes[0]):
            shown = set()
            for (p, method, _), rows in traces.items():
                if p != profile:
                    continue
                ax.step([r[axis] for r in rows], [r[2] for r in rows], where="post",
                        alpha=.23 if final else .6, label=method if method not in shown else None,
                        color=COLORS.get(method, "#237d69"))
                shown.add(method)
            ax.set(xlabel=xlabel, ylabel="Mean feasible archive cost", title=profile)
            ax.legend(frameon=False, ncols=3)
        book.save(fig, key, f"{cohort} convergence by {xlabel.lower()}",
                  "Lines are individual seed trajectories. Means use feasible task archives.", TRACE)


def diagnostics(book):
    transfers = [r for r in book.read(EVENTS) if r["attempt"] == "True"]
    if not transfers:
        return
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    labels = ("attempt", "feasible", "roundtrip", "accepted", "improvement")
    ax.bar(labels, [sum(r[k] == "True" for r in transfers) for k in labels], color="#237d69")
    ax.bar_label(ax.containers[0])
    ax.set_ylabel("Task-transfer events")
    book.save(fig, "14_transfer", "A0 transfer outcomes",
              "Acceptance and archive improvement are distinct measures.", EVENTS)
    eligible = [r for r in transfers if r["pre_ls_cost"] and r["post_ls_cost"] and np.isfinite(v(r,"best_before"))]
    if eligible:
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        ax.boxplot([[100*(v(r, name)-v(r, "best_before"))/max(abs(v(r, "best_before")), 1)
                     for r in eligible] for name in ("pre_ls_cost", "post_ls_cost")],
                   tick_labels=["Before LS", "After LS"])
        ax.set_ylabel("Gap to pre-transfer archive (%)")
        book.save(fig, "15_candidate_cost", "Cost of encodable transfer candidates",
                  "Both values use the same pre-transfer archive as the denominator.", EVENTS)
    rows = book.read(LEARN)
    if rows:
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        unique = {(r["seed"], int(r["generation"])): r for r in rows}
        for seed, events in group(list(unique), lambda k: k[0]).items():
            events.sort(key=lambda k: k[1])
            ax.plot([e[1] for e in events], [v(unique[e], "train_loss") for e in events], "o-", label=seed)
        ax.set(xlabel="Generation", ylabel="Training loss")
        ax.legend(frameon=False, title="Pilot seed")
        book.save(fig, "16_loss", "Loss on the online elite label set", "Labels may change at each event.", LEARN)
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        ax.boxplot([[v(r,k) for r in rows if np.isfinite(v(r,k))]
                    for k in ("positive_mean", "negative_mean")],
                   tick_labels=["Elite edge", "Other edge"])
        ax.set_ylabel("Per-task mean predicted probability")
        book.save(fig, "17_probabilities", "Predicted probabilities by elite label",
                  "Each observation is a task-event probability mean.", LEARN)
    rows = book.read(ATTENTION)
    if rows:
        seed, g = min((int(r["seed"]), int(r["generation"])) for r in rows)
        weights = {(r["task"],r["source_task"]): v(r,"weight") for r in rows
                   if (int(r["seed"]), int(r["generation"])) == (seed,g)}
        tasks = sorted({t for t, _ in weights}, key=coords)
        fig, ax = plt.subplots(figsize=(7.5, 6.4))
        image = ax.imshow([[weights[t,s] for s in tasks] for t in tasks], cmap="viridis")
        ticks = range(0,len(tasks),3)
        ax.set_xticks(ticks, [label(tasks[i]) for i in ticks], rotation=90)
        ax.set_yticks(ticks, [label(tasks[i]) for i in ticks])
        ax.set(xlabel="Source task", ylabel="Query task")
        fig.colorbar(image, ax=ax, label="Attention weight")
        book.save(fig, "18_attention", f"Cross-task attention at seed {seed}, generation {g}",
                  "Each row sums to one.", ATTENTION)


def workload(book):
    rows = [r for r in book.read(WORK) if r["profile"] == "matched_ls"]
    groups = group(rows, lambda r: r["arm"] if r["method"] == "ektgl" else r["method"])
    s = [r for r in book.read(STATS) if r["complete"] == "True" and r["effect_pct"]]
    if not groups:
        return
    final = [r for r in book.read(FINAL_WORK) if r["profile"] in ("native", "matched_ls", "matched_init_ls")]
    compute = group(final, lambda r: r["profile"] + " " + r["method"]) if final else groups
    fig, axs = plt.subplots(2,1,figsize=(max(7,.48*len(compute)),6), sharex=True)
    for ax, key, ylabel in zip(axs, ("wall_clock_s","actual_objective_computations"),
                               ("Elapsed solver time (s)", "Actual objective computations")):
        ax.bar(list(compute), [avg(v(r,key) for r in a) for a in compute.values()], color="#42568b")
        ax.set_ylabel(ylabel)
    axs[-1].tick_params(axis="x", rotation=35)
    cohort = "Confirmation" if final else "Pilot"
    book.save(fig, "21_workload", f"{cohort} runtime and objective computations",
              "Each bar is a mean of completed suites. Elapsed time includes neural work.", FINAL_WORK if final else WORK)
    fig, ax = plt.subplots(figsize=(7.4,4.3))
    for name, records in groups.items():
        count = sum(int(r["ls_calls"]) for r in records if r.get("ls_calls"))
        if count:
            success = sum(int(r["ls_accepted"]) for r in records if r.get("ls_accepted"))
            reduction = avg(v(r,"ls_cost_reduction") for r in records)
            ax.scatter(100*success/count, reduction, c=COLORS.get(name, "#42568b"))
            ax.annotate(name, (100*success/count,reduction), fontsize=8, xytext=(3,3), textcoords="offset points")
    ax.set(xlabel="Local search calls that improved cost (%)", ylabel="Mean cumulative cost reduction")
    book.save(fig, "19_local_search", "Local search effectiveness",
              "Reductions include accepted refinements outside the best archive.", WORK)
    if not s:
        return
    s.sort(key=lambda r: v(r,"effect_pct"))
    fig, ax = plt.subplots(figsize=(8,max(3.8,.31*len(s))))
    ax.axvline(0,color="#999999",lw=1)
    ax.scatter([v(r,"effect_pct") for r in s], np.arange(len(s)), color="#237d69")
    ax.set_yticks(np.arange(len(s)), [r["comparison"] for r in s])
    ax.set_xlabel("Paired mean relative gain (%)")
    book.save(fig, "20_ablation", "Controlled pilot comparisons",
              "Positive values favor the first arm. Each contrast has all three pilot seeds.", STATS)
    fig, ax = plt.subplots(figsize=(7.4,4.3))
    for r in s:
        arm = r["comparison"].split()[0]
        if r["comparison"].endswith(" vs A0") and arm in groups:
            x = avg(v(q,"wall_clock_s") for q in groups[arm])
            y = v(r,"effect_pct")
            ax.scatter(x,y, color=COLORS.get(arm,"#42568b"))
            ax.annotate(arm,(x,y),xytext=(4,3),textcoords="offset points",fontsize=8)
    ax.set(xlabel="Elapsed solver time (s)", ylabel="Paired mean gain over A0 (%)")
    book.save(fig, "22_tradeoff", "Pilot solution quality and runtime",
              "Arms are plotted only when paired effects and elapsed times are available.", STATS, WORK)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    book = Ledger(parser.parse_args().out)
    for render in (history, initial, convergence, diagnostics, workload):
        render(book)
    book.finish()


if __name__ == "__main__":
    main()
