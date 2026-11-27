"""Isolated V3 experiment engine. Frozen V2 implementations are imported in place."""

from dataclasses import dataclass, replace
from time import perf_counter
from collections import Counter
import math
import random

import numpy as np

from ekt_gl.idpc.decoding import greedy_path
from ekt_gl.idpc.v2.core import Candidate, EvaluationTracker, initialize, limits_for, project, seed_for
from ekt_gl.idpc.v2.evolution import Evolution
from ekt_gl.idpc.v2.learning import elite_labels, greedy
from .beam import BeamConfig, SearchResult, decode_beam
from .neighborhoods import improve


@dataclass(frozen=True)
class Variant:
    ls: str = "2opt"
    width: int = 128
    loss: str = "full"
    features: str = "v2"
    decoder: str = "greedy"
    insertion: str = "v2"
    matched_init: bool = False
    fusion: str = "shared"
    beam: BeamConfig = BeamConfig()


ARMS = {
    "A0": Variant(),
    "A1": Variant(ls="legacy"),
    "A2_32": Variant(width=32),
    "A2_64": Variant(width=64),
    "A3": Variant(loss="sampled"),
    "A4": Variant(features="legacy"),
    "A5": Variant(decoder="legacy_greedy"),
    "A6": Variant(insertion="distinct"),
    "A7": Variant(matched_init=True),
    "A8_identity": Variant(fusion="identity"),
    "A8_independent": Variant(fusion="independent"),
    "A8_shuffled": Variant(decoder="shuffled_greedy"),
    "A9_legacy_32": Variant(ls="legacy",width=32),
    "D1": Variant(decoder="cost_beam",beam=BeamConfig(beta=0)),
    "D2": Variant(decoder="shuffled_beam"),
    "D3": Variant(decoder="trained_beam"),
    "G1": Variant(features="distances"), "G2": Variant(loss="balanced"),
    "G3": Variant(loss="frequency"), "G4": Variant(loss="elite_cost"),
    "N1": Variant(ls="parallel"), "N2": Variant(ls="shortcut"), "N3": Variant(ls="subpath"),
}


class ResearchEvolution(Evolution):
    def __init__(self, *args, variant, diagnostic, **kwargs):
        super().__init__(*args, **kwargs)
        self.variant=variant
        self.diagnostic=diagnostic

    def initialize(self):
        if not self.variant.matched_init or self.method=="mfea":
            return super().initialize()
        n,k=self.config.population_per_task,len(self.names)
        if self.tracker.budget is not None and self.tracker.budget<n*k*k:
            raise ValueError("Budget cannot fund matched initialization")
        rng=random.Random(seed_for(self.seed,"mfea"))
        bounds=limits_for(self.tracker.graphs)
        genomes=[initialize(bounds,rng) for _ in range(n*k)]
        populations=[]
        for t,g in enumerate(self.tracker.graphs):
            pop=[]
            for genome in genomes:
                chromosome=project(genome,g.n)
                solution=self.tracker.gpa(t,chromosome,"init")
                self.record(t,solution)
                pop.append(Candidate(chromosome,solution))
            populations.append(self.survive(pop,n))
        self.populations=populations

    def improve(self,task,candidate,pairs,rng,phase):
        counts=self.tracker.per_task[task]
        before=counts["roundtrip_failures"]
        result=improve(self.tracker,task,candidate,self.local_limits[task],pairs,
                       self.config.ls,rng,phase,self.variant.ls,unified=self.method=="mfea")
        if self.diagnostic and candidate.solution.feasible:
            if self.variant.ls=="2opt":
                counts["ls_no_improvement"]+=result is candidate
            counts["ls_roundtrip_failures"]+=counts["roundtrip_failures"]-before
        return result

    def _evaluate(self, chromosome, task, index, pairs):
        candidate=Candidate(chromosome,self.tracker.gpa(task,chromosome,"offspring"))
        self.record(task,candidate.solution)
        if self.config.ordinary_ls(self.method) and candidate.solution.feasible:
            rng=random.Random(seed_for(self.seed,self.names[task],"ordinary",self.generation,index))
            candidate=self.improve(task,candidate,pairs[task],rng,"ordinary")
            self.record(task,candidate.solution)
        return candidate


def shuffled(scores,seed,event,task):
    return np.random.default_rng(seed_for(seed,"heatmap",event,task)).permutation(scores)


def construct(graph,lower,variant,scores,counters,seed,event,task):
    mode=variant.decoder
    if mode.endswith("beam"):
        if mode=="shuffled_beam":
            scores=shuffled(scores,seed,event,task)
        return decode_beam(graph,lower,variant.beam,None if mode=="cost_beam" else scores)
    if mode=="shuffled_greedy":
        scores=shuffled(scores,seed,event,task)
    before=counters["greedy_edge_checks"]
    if mode=="legacy_greedy":
        def score(batch):
            counters["greedy_edge_checks"]+=len(batch)
            return scores[batch.positions]
        path=greedy_path(graph,score,32768)
    else:
        path=greedy(graph,scores,counters)
    return SearchResult(() if path is None else (path,),len(path or ()),
                        counters["greedy_edge_checks"]-before,False)


def evaluate_candidates(evolution,task,proposal,pairs):
    tracker=evolution.tracker
    accepted=[]
    rows=[]
    for ids in proposal.paths:
        if tracker.exhausted:
            break
        solution=tracker.path(task,ids,"transfer")
        verified=(tracker.roundtrip(task,solution,evolution.local_limits[task],"transfer",
                                   phase="pre_transfer") if solution.feasible else None)
        rows.append(dict(edge_ids=ids,feasible=solution.feasible,cost=solution.cost,
                         roundtrip=verified is not None))
        if verified is not None:
            accepted.append(verified)
    if not accepted:
        return None,None,rows
    chosen=min(accepted,key=lambda c:(c.solution.cost,c.solution.path))
    rng=random.Random(seed_for(evolution.seed,evolution.names[task],"pre_transfer",evolution.generation))
    return chosen,evolution.improve(task,chosen,pairs,rng,"pre_transfer"),rows


def score_diagnostic(graph,scores,positive,seed):
    mapping=graph.by_id.positions
    positive=sorted(positive)
    where=np.asarray(positive if mapping is None else [mapping[e] for e in positive],dtype=np.int64)
    s=np.asarray(scores)
    p=s[where]
    metrics=dict(positive_mean=float(p.mean()) if len(p) else None,
                 negative_mean=float((s.sum(dtype=np.float64)-p.sum(dtype=np.float64))/(len(s)-len(p)))
                 if len(s)>len(p) else None)
    positives_at_half=int((p>=.5).sum())
    predicted=int((s>=.5).sum())
    metrics.update(precision_at_half=positives_at_half/max(predicted,1),
                   recall_at_half=positives_at_half/max(len(p),1),PR_AUC_AP=None)
    if len(p):
        levels,positives=np.unique(p,return_counts=True)
        hist=np.histogram(s,bins=np.concatenate(([-np.inf],levels,[np.inf])))[0]
        at_least=np.cumsum(hist[1:][::-1])[::-1]
        true_at_least=np.cumsum(positives[::-1])[::-1]
        metrics["PR_AUC_AP"]=float(np.sum(positives*true_at_least/at_least)/len(p))
    rng=random.Random(seed)
    pos_slots=set(map(int,where))
    neg=[i for i in rng.sample(range(len(s)),min(len(s),len(where)+32)) if i not in pos_slots][:32]
    ids=graph.edges.table.rows["id"]
    metrics["edge_sample"]=[(int(ids[i]),float(s[i]),int(i in pos_slots)) for i in [*where,*neg]]
    return metrics


def replay(evolution,task,graph,lower,scores,pairs):
    """Evaluate decoder alternatives against the same heatmap and population."""
    variant=evolution.variant
    seed,g=evolution.seed,evolution.generation
    choices={key:replace(variant,decoder=ARMS[key].decoder,beam=ARMS[key].beam)
             for key in ("A0","D1","D2","D3")}
    choices["legacy_greedy"]=replace(variant,decoder="legacy_greedy")
    if g==10:
        for width in (1,8,16):
            for beta in (0,.1,.5,1):
                choices[f"grid_w{width}_b{beta:g}"]=replace(variant,
                    decoder="trained_beam" if beta else "cost_beam",beam=BeamConfig(width,beta))
    result=[]
    pop=evolution.populations[task]
    for key,variant in choices.items():
        tracker=EvaluationTracker([graph],cache_size=evolution.config.gpa_cache_per_task,
                                  decoders=None if evolution.tracker.decoders is None else
                                  [evolution.tracker.decoders[task]])
        trial=ResearchEvolution("ektgl",[graph],[evolution.names[task]],evolution.config,seed,tracker,
                                variant=variant,diagnostic=False)
        trial.generation=g
        start=perf_counter()
        candidate=construct(graph,lower,variant,scores,tracker.per_task[0],seed,g,evolution.names[task])
        pre,post,details=evaluate_candidates(trial,0,candidate,pairs)
        result.append(dict(variant=key,paths=len(candidate.paths),expansions=candidate.expansions,
                           ranking_reads=candidate.ranking_reads,requests=tracker.requests,
                           best_before=evolution.best[task].cost,pre_cost=pre.solution.cost if pre else None,
                           post_cost=post.solution.cost if post else None,
                           improved=bool(post and post.solution.cost<evolution.best[task].cost),
                           beats_worst=bool(post and post.solution.cost<pop[-1].solution.cost),
                           seconds=perf_counter()-start))
    return result


def transfer(evolution, network, pairs, lower, diagnostic=False):
    tracker=evolution.tracker
    seed,g=evolution.seed,evolution.generation
    if network is None:
        labels=[elite_labels(p,evolution.config.elite_count) for p in evolution.populations]
        loss=train_time=inference_time=0.0
        scores=attention=None
    else:
        network.event=g
        labels=network.labels(evolution.populations)
        loss,train_time=network.fit(labels)
        scores,attention,inference_time=(network.scores() if loss is not None else (None,None,0.0))
    rows=[]
    for t,(name,pop,graph,(positives,elites,duplicates)) in enumerate(zip(
            evolution.names,evolution.populations,tracker.graphs,labels)):
        counts=tracker.per_task[t]
        best_before=evolution.best[t].cost
        row=dict(generation=g,task=name,positive_labels=len(positives),
                 negative_labels=len(graph.edges)-len(positives) if elites else 0,
                 elite_count=elites,duplicate_elites=duplicates,train_loss=loss,
                 attempt=False,feasible=False,encoded=False,roundtrip=False,
                 accepted=False,improvement=False,pre_ls_cost=None,post_ls_cost=None,
                 best_before=best_before,best_after=best_before)
        rows.append(row)
        if loss is None or tracker.exhausted:
            row["outcome"]="budget_exhausted" if tracker.exhausted else "no_elites"
            continue
        row["attempt"]=True
        counts["transfer_attempts"]+=1
        start=perf_counter()
        s=None if scores is None else scores[t]
        candidate=construct(graph,lower[t] if lower else None,evolution.variant,s,counts,seed,g,name)
        inference_time+=perf_counter()-start
        counts["beam_expansions"]+=candidate.expansions if evolution.variant.decoder.endswith("beam") else 0
        counts["ranking_edge_reads"]+=candidate.ranking_reads
        proposed,refined,details=evaluate_candidates(evolution,t,candidate,pairs[t])
        row["paths"]=details
        row["ranking_edge_reads"]=candidate.ranking_reads
        row["beam_expansions"]=candidate.expansions if evolution.variant.decoder.endswith("beam") else 0
        row["feasible"]=any(x["feasible"] for x in details)
        counts["valid_candidates"]+=row["feasible"]
        row["roundtrip"]=proposed is not None
        row["encoded"]=proposed is not None
        if diagnostic and s is not None:
            row["learning"]=score_diagnostic(graph,s,positives,seed_for(seed,"diagnosis",g,name))
            if g in (10,20):
                # Replay does not change or consume the evolving solver's budget.
                row["replay"]=replay(evolution,t,graph,lower[t],s,pairs[t])
        if proposed is None:
            row["outcome"]=("budget_exhausted" if tracker.exhausted else
                            "unrepresentable" if candidate.paths else "dead_end")
            continue
        row.update(pre_ls_cost=proposed.solution.cost,post_ls_cost=refined.solution.cost)
        distinct=not any(m.solution.path==refined.solution.path for m in pop)
        row["duplicate"]=not distinct
        if refined.solution.cost<pop[-1].solution.cost and (distinct or evolution.variant.insertion=="v2"):
            evolution.populations[t]=evolution.survive(pop+[refined],len(pop))
            counts["accepted_candidates"]+=1
            row["accepted"]=True
        evolution.record(t,refined.solution)
        row["best_after"]=evolution.best[t].cost
        row["improvement"]=row["best_after"]<best_before
        counts["strict_improvements"]+=row["improvement"]
        row["outcome"]=("accepted" if row["accepted"] else
                        "duplicate" if not distinct and evolution.variant.insertion=="distinct" else "elitist_reject")
    return dict(event_generation=g,rows=rows,attention=attention,
                train_time_s=train_time,inference_time_s=inference_time,train_loss=loss,
                gradients=network.gradients if network else {},
                positive_examples=network.examples[0] if network else 0,
                negative_examples=network.examples[1] if network else 0)
