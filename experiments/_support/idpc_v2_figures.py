"""Publication-sized static figures for the V2 report."""

from collections import defaultdict
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LABELS = dict(ea="EA",mfea="MFEA-EDU",ektgl="EKT-GL")
COLORS = dict(ea="#3569A7",mfea="#CC7942",ektgl="#2B827D")


def plot_all(root,manifest,runs,summary,groups,wtl,curves,suites,events):
    output = root/"figures"
    output.mkdir(exist_ok=True)
    tasks = manifest["tasks"]
    names = [r["task"] for r in tasks]
    short = [f"{r['n_nodes']}×{r['n_domains']}" for r in tasks]
    labels = LABELS if manifest["profile"] == "native" else {
        **LABELS, "ea": "EA+LS", "mfea": "MFEA+LS"}
    methods = [m for m in labels if any(record["method"]==m for record in suites)]
    counts = [sum(1 for record in suites if record["method"]==m) for m in methods]
    title = f"{manifest['profile']} · {manifest['suite_id']} · " + ", ".join(
        f"{labels[m]} {n} seeds" for m,n in zip(methods,counts))

    def save(fig,name):
        fig.suptitle(title,fontsize=11)
        fig.tight_layout()
        fig.savefig(output/(name+".png"),dpi=240,bbox_inches="tight")
        plt.close(fig)

    fig,ax = plt.subplots(figsize=(max(6,len(tasks)*0.33),3))
    heat = np.array([[summary.get((m,t),{}).get("AVG_gap_percent",np.nan) for t in names]
                     for m in methods],dtype=float)
    pos = ax.imshow(np.ma.masked_invalid(heat),cmap="YlOrRd",aspect="auto")
    ax.set_xticks(range(len(names)),short,rotation=90,fontsize=8)
    ax.set_yticks(range(len(methods)),[labels[m] for m in methods])
    ax.set_xlabel("Nodes × domains")
    fig.colorbar(pos,ax=ax,label="AVG gap to best observed method (%)")
    save(fig,"01_avg_gap")

    if wtl:
        fig,axes = plt.subplots(2,1,figsize=(9,5))
        for ax,metric in zip(axes,("AVG","Mann–Whitney-Holm")):
            selected = [r for r in wtl if r["subset"]=="all" and r["metric"]==metric]
            left = np.zeros(len(selected))
            tags = [f"{labels[r['method_a']]} vs {labels[r['method_b']]}" for r in selected]
            for key,color,label in (("win","#2B827D","A wins"),("tie","#B6BAC3","Exact ties"),
                                     ("loss","#CC7942","A losses"),("not_significant","#777B8A","Insignificant"),
                                     ("insufficient","#DDDDE4","Insufficient")):
                values = [r[key] for r in selected]
                if any(values):
                    ax.barh(tags,values,left=left,color=color,label=label)
                    left += values
            ax.set_title(metric)
            ax.set_xlabel("Tasks")
            if ax.get_legend_handles_labels()[0]:
                ax.legend(loc="upper center",bbox_to_anchor=(0.5,-0.25),ncol=4,fontsize=8)
        save(fig,"02_win_tie_loss")

    fig,axes = plt.subplots(3,1,figsize=(max(8,len(tasks)*0.33),8),sharex=True)
    for ax,metric in zip(axes,("BF","AVG","STD_sample")):
        for m in methods:
            values = [summary.get((m,t),{}).get(metric) for t in names]
            ax.plot(range(len(names)),values,label=labels[m],color=COLORS[m],marker=".",linewidth=1)
        ax.set_ylabel(f"{metric} (path cost)")
    axes[-1].set_xticks(range(len(names)),short,rotation=90,fontsize=8)
    axes[-1].set_xlabel("Nodes × domains")
    axes[0].legend(ncol=3)
    save(fig,"03_bf_avg_std")

    fig,axes = plt.subplots(1,2,figsize=(11,4),sharey=True)
    for ax,axis,label in zip(axes,("generation","suite_evaluations"),
                             ("Generation","Suite objective requests")):
        for m in methods:
            rows = [r for r in curves if r["axis"]==axis and r["method"]==m]
            if rows:
                x,y,lo,hi = ([r[k] for r in rows] for k in ("x","mean_gap","lower_95","upper_95"))
                ax.plot(x,y,color=COLORS[m],label=labels[m])
                ax.fill_between(x,lo,hi,color=COLORS[m],alpha=0.16)
        ax.set_xlabel(label)
        ax.set_yscale("symlog",linthresh=1)
        if ax.get_legend_handles_labels()[0]:
            ax.legend()
    axes[0].set_ylabel("Mean gap over feasible paths (%)")
    save(fig,"04_convergence")

    targets = [r for r in tasks if (r["n_nodes"],r["n_domains"]) in {
        (20,10),(35,17),(70,70),(80,80),(90,45),(100,100)}] or tasks[:min(6,len(tasks))]
    cols = min(3,len(targets))
    fig,axes = plt.subplots((len(targets)+cols-1)//cols,cols,figsize=(4*cols,6),squeeze=False)
    for ax,task in zip(axes.flat,targets):
        t = task["task"]
        values = [[r["best_cost"] for r in groups[m,t] if r["feasible"]] for m in methods]
        ax.boxplot(values,tick_labels=[labels[m] for m in methods])
        ax.set_title(f"{task['n_nodes']} nodes, {task['n_domains']} domains")
        ax.set_ylabel("Final cost (feasible seeds)")
    for i,ax in enumerate(axes.flat):
        if i>=len(targets):
            ax.set_visible(False)
    save(fig,"05_final_cost_boxplots")

    fig,axes = plt.subplots(1,3,figsize=(11,4))
    for ax,key,label in zip(axes,("wall_clock_s","objective_evaluation_requests","local_search_checks"),
                            ("Solver time (s)","Objective requests","LS cut attempts")):
        values = [mean(r.get(key,0) for r in suites if r["method"]==m) for m in methods]
        ax.bar([labels[m] for m in methods],values,color=[COLORS[m] for m in methods])
        ax.set_ylabel(label+" per suite")
        ax.tick_params(axis="x",rotation=20)
    save(fig,"06_runtime_and_work")

    if events:
        by_generation = defaultdict(list)
        for event in events:
            by_generation[event["generation"]].append(event)
        x = sorted(by_generation)
        fig,(a,b) = plt.subplots(1,2,figsize=(10,4))
        for key,label in (("feasible","Feasible"),("accepted","Accepted"),("improvement","Improved archive")):
            a.plot(x,[100*mean(r[key] for r in by_generation[g]) for g in x],label=label)
        losses = [[r["train_loss"] for r in by_generation[g] if r["train_loss"] is not None] for g in x]
        b.plot(x,[mean(v) if v else np.nan for v in losses],color=COLORS["ektgl"])
        a.set_ylabel("Percent of event/task slots")
        b.set_ylabel("Online training cross entropy")
        for ax in (a,b):
            ax.set_xlabel("Transfer generation")
        a.legend()
        save(fig,"07_transfers")

    mfea_runs = [trace for (m,_),(_,trace) in runs.items() if m=="mfea"]
    if mfea_runs:
        generation = max(trace[-1]["generation"] for trace in mfea_runs)
        values = np.full((len(names),generation+1),np.nan)
        for g in range(generation+1):
            entries = [trace[g]["populations"] for trace in mfea_runs if len(trace)>g]
            values[:,g] = np.mean(np.array(entries)[:,:,0],axis=0)
        fig,ax = plt.subplots(figsize=(9,max(4,len(tasks)*0.15)))
        heat = ax.imshow(values,aspect="auto",cmap="viridis",interpolation="nearest")
        ax.set_yticks(range(len(names)),short,fontsize=6)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Nodes × domains")
        fig.colorbar(heat,ax=ax,label="Mean individuals assigned by skill factor")
        save(fig,"08_mfea_task_allocations")
