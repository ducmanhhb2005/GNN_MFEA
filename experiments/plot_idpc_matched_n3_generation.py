"""Render the 300-generation N3 figures and mark checked coverage complete."""
from __future__ import annotations
import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from experiments._support.idpc_v3_archive import json_read, structured, write
from experiments.plot_idpc_matched_n3 import COLORS, COMBAT_LABELS, LABELS, Figures, plot_name, stacked
from experiments.report_idpc_v3 import put
from experiments.run_idpc_matched_n3 import METHODS
from experiments.run_idpc_matched_n3_generation import OUT


def figures(root):
    coverage=json_read(root/"coverage.json")
    if (coverage["status"] not in ("SOLVER_COMPLETE", "COMPLETE")
            or coverage["counts"] != {"complete": 60}
            or coverage["validated_feasible_paths"] != 2520):

        raise ValueError("All 60 uncapped suites are required")
    book=Figures(root)
    book.index = ["# Figures", "", "Figures are available as 300 DPI PNG and vector PDF.", ""]
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    rows=book.read("convergence")
    largest = max(int(r["objective_requests"]) for r in rows)
    grouped=defaultdict(list)
    for r in rows:
        grouped[r["method"],int(r["seed"])].append(r)
    fig,axes=plt.subplots(1,2,figsize=(12,4.2))
    sources=[]
    for ax,axis in zip(axes,("generation","objective_requests")):
        grid=np.arange(301) if axis=="generation" else np.linspace(0,largest,301,dtype=np.int64)
        for m in METHODS:
            cohort=[]
            for (method,_),records in sorted(grouped.items()):
                if m!=method:
                    continue
                x=[int(r[axis]) for r in records]
                y=np.array([float(r["suite_mean_archive_cost"]) if r["suite_mean_archive_cost"] else np.nan
                            for r in records])
                at=np.searchsorted(x,grid,side="right")-1
                cohort.append(np.where((at>=0)&(grid<=x[-1]),y[np.maximum(at,0)],np.nan))
            data=np.array(cohort)
            shown=np.isfinite(data).any(axis=0)
            q1,mid,q3=np.nanquantile(data[:,shown],[.25,.5,.75],axis=0)
            x=grid[shown]
            ax.plot(x,mid,color=COLORS[m],label=LABELS[m])
            ax.fill_between(x,q1,q3,color=COLORS[m],alpha=.15)
            sources.extend(dict(method=m,axis=axis,x=int(v),q25=l,median=y,q75=h,contributors=int(n))
                           for v,l,y,h,n in zip(x,q1,mid,q3,np.isfinite(data[:,shown]).sum(axis=0)))
        ax.set(xlabel="Generation" if axis=="generation" else "Objective requests per suite",
               ylabel="Mean archive cost across 42 tasks")
        ax.legend(frameon=False)
    put(book.out/"01_convergence_source.csv",sources)
    book.save(fig,"01_convergence","Convergence at 300 generations","01_convergence_source.csv",
              "Medians and interquartile ranges use only suites observed at each x-value.")

    workload=book.read("workload_summary")
    fig,axes=plt.subplots(1,3,figsize=(12,4.2))
    for ax,(key,ylabel) in zip(axes,(("mean_objective_requests","Logical objective requests"),
                                    ("mean_actual_objective_computations","Actual objective computations"),
                                    ("median_wall_clock_s","Median elapsed solver seconds"))):
        ax.bar([LABELS[r["method"]] for r in workload],[float(r[key]) for r in workload],
               color=[COLORS[r["method"]] for r in workload])
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x",labelrotation=20)
    book.save(fig,"02_workload","Workload and runtime at the shared generation ceiling",
              "../workload_summary.csv","Each group summarizes 20 runs.")

    combat=book.read("win_tie_loss")
    fig,axes=plt.subplots(2,3,figsize=(13,5.8))
    for i,metric in enumerate(("AVG","BF")):
        for scope,ax in zip(("all42","set1","set2"),axes[i]):
            r=[row for row in combat if row["scope"]==scope and row["metric"]==metric]
            stacked(ax,r,[COMBAT_LABELS[x["comparison"]] for x in r])
            ax.set_title(f"{scope.capitalize()} · {metric}")
    axes[0,0].legend(frameon=False,ncols=3,loc="lower left",bbox_to_anchor=(0,1.08))
    book.save(fig,"03_combat","Pairwise win, tie and loss at 300 generations","../win_tie_loss.csv",
              "Each bar counts problem instances.")
    raw = book.read("full_raw_results")
    outcomes = book.read("publication_table_all42")
    if (len(raw)!=2520 or len({(r["method"],r["seed"],r["task"]) for r in raw})!=2520
            or len(outcomes)!=42 or any(r[f"{m}_runs"]!="20" for r in outcomes for m in METHODS)
            or len(combat)!=18 or len(rows)!=18060
            or any(r["feasible"] != "True" or r["independently_validated"] != "True" for r in raw)):

        raise ValueError("Publication tables have incomplete coverage")
    outcomes.sort(key=lambda r: (r["set"],
                                  tuple(map(int, r["task"].removeprefix("idpc_").split("x")))))
    fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
    for field, ax in zip(("BF", "AVG", "STD"), axes):
        for m in METHODS:
            ax.plot([float(row[f"{m}_{field}"]) for row in outcomes], marker="o",
                    ms=2, lw=.9, label=LABELS[m], color=COLORS[m])
        ax.set_ylabel(f"{field} (cost)")
    axes[0].legend(frameon=False, ncols=3)
    axes[-1].set_xticks(range(len(outcomes)), [plot_name(r["task"]) for r in outcomes], rotation=90)
    book.save(fig, "04_quality", "Solution quality by instance", "../publication_table_all42.csv",
              "BF is the best of 20 runs. AVG and sample STD summarize the same 20 costs.")
    write(book.out/"index.md", "\n".join(book.index)+"\n")
    for name in ("01_convergence","02_workload","03_combat","04_quality"):
        with Image.open(book.out/(name+".png")) as im:
            if round(im.info.get("dpi",(0,0))[0])!=300:
                raise ValueError("Expected 300-DPI figures")
            im.verify()
    coverage["status"]="COMPLETE"
    coverage["accepted"] = dict(checked_60_uncapped_suites=True, all_2520_paths_validated=True,
                                actual_neural_activity_reconciled=True, tables_and_figures=True)
    structured(root/"coverage.json",coverage)
    report=root/"FINAL_BENCHMARK_REPORT.md"
    write(report,report.read_text(encoding="utf-8").replace("**SOLVER_COMPLETE**","**COMPLETE**"))
    print("COMPLETE: generation profile, 60 suites and 2,520 task records")


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=OUT)
    figures(parser.parse_args().out)
