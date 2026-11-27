"""Rebuild the eight Matched-N3 figures from published ledgers."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments._support.idpc_v3_archive import read, write
from experiments.report_idpc_v3 import put
from experiments.run_idpc_matched_n3 import BUDGET, METHODS, OUT, RESEARCH

COLORS = {"ea": "#4973a1", "mfea": "#aa5e38", "ektgl": "#237a68"}
LABELS = {"ea": "EA+N3", "mfea": "MFEA+N3", "ektgl": "EKT-GL+N3"}
STATUS_COLORS = {"wins": "#237a68", "ties": "#b7b2a6", "losses": "#aa5e38"}
COMBAT_LABELS = {"ektgl vs mfea": "EKT-GL vs MFEA", "ektgl vs ea": "EKT-GL vs EA",
                 "mfea vs ea": "MFEA vs EA"}


def plot_name(task):
    return task.removeprefix("idpc_").rsplit("x", 1)[0].replace("x", "×")


class Figures:
    def __init__(self, root):
        self.root = root
        self.out = root / "figures"
        self.index = ["# Figures", "", "PNG at 300 DPI. Reproduce with the Matched-N3 figure script.", ""]

    def read(self, name):
        return list(read(self.root / (name + ".csv")))

    def save(self, fig, key, title, source, caption):
        fig.suptitle(title, x=.06, ha="left", fontsize=13)
        fig.tight_layout()
        fig.savefig(self.out / (key+".png"), dpi=300, bbox_inches="tight")
        fig.savefig(self.out / (key+".pdf"), bbox_inches="tight")
        plt.close(fig)
        self.index.append(f"- [{title}]({key}.png). {caption} [Source CSV]({source}).")


def stacked(ax, rows, labels):
    offset = np.zeros(len(rows))
    for k,color in STATUS_COLORS.items():
        n = [int(row[k]) for row in rows]
        bars = ax.barh(labels, n, left=offset, color=color, label=k.capitalize())
        ax.bar_label(bars, [str(x) if x>=3 else "" for x in n], label_type="center",
                     color="#20201f" if k=="ties" else "white", fontsize=8)
        offset += n
    ax.set_xlabel("Instances")
    ax.invert_yaxis()


def figures(out):
    book = Figures(out)
    battle = book.read("win_tie_loss")
    tasks = sorted(book.read("publication_table_all42"),
                   key=lambda r: tuple(map(int, r["task"].removeprefix("idpc_").split("x"))))
    workload = book.read("workload_summary")
    ls = book.read("local_search_summary")
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})

    fig, axes = plt.subplots(2,3,figsize=(13,5.8))
    for i,metric in enumerate(("AVG","BF")):
        for area, ax in zip(("all42","set1","set2"), axes[i]):
            rows = [r for r in battle if r["scope"]==area and r["metric"]==metric]
            stacked(ax,rows,[COMBAT_LABELS[r["comparison"]] for r in rows])
            ax.set_title(f"{area.capitalize()} · {metric}")
    axes[0,0].legend(frameon=False,ncols=3,loc="lower left",bbox_to_anchor=(0,1.10))
    book.save(fig,"01_combat","Win, tie and loss across the three N3 frameworks","../win_tie_loss.csv",
              "Each bar compares method AVG or BF on the labeled instance set.")

    archived = list(read(RESEARCH/"09_final_benchmark/win_tie_loss_by_set.csv"))
    previous = next(r for r in archived if r["profile"]=="matched_ls" and r["scope"]=="all42"
                    and r["metric"]=="AVG" and r["method_a"]=="ektgl" and r["method_b"]=="mfea")
    current = next(r for r in battle if r["scope"]=="all42" and r["metric"]=="AVG"
                   and r["comparison"]=="ektgl vs mfea")
    paired_bars = [dict(operator=label, **{k:r[k] for k in STATUS_COLORS})
                   for label,r in (("2-opt",previous),("N3",current))]
    put(book.out/"02_ekt_vs_mfea_source.csv",paired_bars)
    fig,ax = plt.subplots(figsize=(7.3,3.3))
    stacked(ax,paired_bars,[r["operator"] for r in paired_bars])
    ax.legend(frameon=False,ncols=3,loc="lower left",bbox_to_anchor=(0,1.01))
    book.save(fig,"02_ekt_vs_mfea","EKT-GL vs MFEA with a common local-search operator",
              "02_ekt_vs_mfea_source.csv","AVG W/T/L on the same 42 instances and 20 seeds.")

    fig,axes = plt.subplots(3,1,figsize=(12,8),sharex=True)
    for field,ax in zip(("BF","AVG","STD"),axes):
        for m in METHODS:
            ax.plot([float(row[f"{m}_{field}"]) for row in tasks],marker="o",ms=2,lw=.9,
                    label=LABELS[m],color=COLORS[m])
        ax.set_ylabel(f"{field} (cost)")
    axes[0].legend(frameon=False,ncols=3)
    axes[-1].set_xticks(range(len(tasks)),[plot_name(r["task"]) for r in tasks],rotation=90)
    axes[-1].set_xlabel("Instance (vertices × domains)")
    book.save(fig,"03_bf_avg_std","Per-instance best, mean and sample standard deviation",
              "../publication_table_all42.csv","Each measure summarizes 20 runs.")

    gap_rows = [dict(task=row["task"], **{m:100*(float(row[m+"_AVG"])-float(row["mfea_AVG"]))/
                 max(1.,abs(float(row["mfea_AVG"]))) for m in METHODS}) for row in tasks]
    put(book.out/"04_heatmap_source.csv",gap_rows)
    gap = np.array([[r[m] for r in gap_rows] for m in METHODS])
    limit = max(1.,float(np.abs(gap).max()))
    fig,ax=plt.subplots(figsize=(12,3.5))
    im=ax.imshow(gap,cmap="RdBu_r",vmin=-limit,vmax=limit,aspect="auto")
    ax.set_yticks(range(len(METHODS)),[LABELS[m] for m in METHODS])
    ax.set_xticks(range(len(tasks)),[plot_name(r["task"]) for r in tasks],rotation=90)
    ax.set_xlabel("Instance (vertices × domains)")
    fig.colorbar(im,ax=ax,label="AVG cost gap to MFEA+N3 (%)")
    book.save(fig,"04_heatmap","Per-instance quality relative to MFEA+N3","04_heatmap_source.csv",
              "Negative values indicate lower instance AVG.")

    traces = defaultdict(list)
    for r in book.read("convergence"):
        traces[r["method"],int(r["seed"])].append((int(r["objective_requests"]),r["suite_mean_archive_cost"]))
    grid=np.unique(np.append(np.arange(0,BUDGET+1,1000),BUDGET))
    curve=[]
    fig,ax=plt.subplots(figsize=(8,4.6))
    for m in METHODS:
        histories=[]
        for (method,seed), rows in sorted(traces.items()):
            if method!=m:
                continue
            x,y=zip(*rows)
            idx=np.searchsorted(x,grid,side="right")-1
            costs=np.array([float(value) if value else np.nan for value in y])
            histories.append(np.where(idx>=0,costs[np.maximum(idx,0)],np.nan))
        cohort=np.array(histories)
        valid=np.isfinite(cohort).any(axis=0)
        x=grid[valid]
        quantiles=np.nanquantile(cohort[:,valid],[.25,.5,.75],axis=0)
        low,mid,high=quantiles
        curve.extend(dict(method=m,objective_requests=int(q),q25=l,median=v,q75=h,
                          contributors=int(n)) for q,l,v,h,n in zip(
            x,low,mid,high,np.isfinite(cohort[:,valid]).sum(axis=0)))
        ax.step(x,mid,where="post",label=LABELS[m],color=COLORS[m])
        ax.fill_between(x,low,high,step="post",alpha=.15,color=COLORS[m])
    ax.set(xlabel="Objective requests per suite",ylabel="Mean archive cost across the 42 tasks")
    ax.legend(frameon=False)
    put(book.out/"05_convergence_source.csv",curve)
    book.save(fig,"05_convergence","Convergence on the common objective-request axis",
              "05_convergence_source.csv","Median and interquartile range across seeds. The trace aligns by the most recent recorded checkpoint.")

    cohort=defaultdict(list)
    for r in book.read("full_raw_results"):
        cohort[r["method"],int(r["seed"])].append(float(r["cost"]))
    values=[dict(method=m,seed=s,suite_mean_cost=float(np.mean(v))) for (m,s),v in sorted(cohort.items())]
    put(book.out/"06_seed_boxplots_source.csv",values)
    fig,ax=plt.subplots(figsize=(6.8,4.1))
    slices=[[r["suite_mean_cost"] for r in values if r["method"]==m] for m in METHODS]
    boxes=ax.boxplot(slices,tick_labels=[LABELS[m] for m in METHODS],patch_artist=True)
    rng=np.random.default_rng(5292)
    for i,(box,section,m) in enumerate(zip(boxes["boxes"],slices,METHODS),1):
        box.set_facecolor(COLORS[m])
        box.set_alpha(.36)
        ax.scatter(i+rng.uniform(-.11,.11,len(section)),section,s=11,color=COLORS[m],alpha=.8)
    ax.set_ylabel("Mean cost across the 42 tasks")
    book.save(fig,"06_seed_boxplots","Solution quality across the 20 seeds","06_seed_boxplots_source.csv",
              "Each point is one independently seeded suite.")

    fig,axes=plt.subplots(1,3,figsize=(12.1,4))
    panels=(("mean_actual_objective_computations","Actual objective computations","Computations per suite"),
            ("mean_ls_subpath_actual_expansions","Physical N3 expansions","Expansions per suite"),
            ("median_wall_clock_s","Median solver runtime","Seconds per suite"))
    for ax,(key,title,unit) in zip(axes,panels):
        ax.bar([LABELS[r["method"]] for r in workload],[float(r[key]) for r in workload],
               color=[COLORS[r["method"]] for r in workload])
        ax.set(title=title,ylabel=unit)
        ax.tick_params(axis="x",rotation=20)
    book.save(fig,"07_workload","Objective computations, native search work and elapsed time",
              "../workload_summary.csv","All three frameworks charge the same logical request cap.")

    fig,ax=plt.subplots(figsize=(7.4,4.2))
    ordinary=[r for r in ls if r["phase"]=="ordinary"]
    labels=[LABELS[r["method"]]+" · offspring" for r in ordinary]
    special=[r for r in ls if r["phase"]=="pre_transfer"]
    labels.extend(LABELS[r["method"]]+" · transfer" for r in special)
    complete=ordinary+special
    bars=ax.barh(labels,[100*float(r["improvement_rate"]) for r in complete],
                 color=[COLORS[r["method"]] for r in complete])
    ax.bar_label(bars,fmt="%.1f%%",padding=3)
    ax.set_xlim(0,max(10,max(100*float(r["improvement_rate"]) for r in complete)*1.22))
    ax.invert_yaxis()
    ax.set_xlabel("N3 calls with strict improvement (%)")
    book.save(fig,"08_n3_improvement","N3 improvement rates by framework and phase",
              "../local_search_summary.csv","Each rate pools completed local-search calls within its phase.")
    write(book.out/"index.md","\n".join(book.index)+"\n")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=OUT)
    figures(parser.parse_args().out)


if __name__ == "__main__":
    main()
