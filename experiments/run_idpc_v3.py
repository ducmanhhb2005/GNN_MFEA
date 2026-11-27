"""Reproducible IDPC-DU research runner backed by the frozen V2 evaluator."""

from __future__ import annotations

import argparse
from contextlib import closing
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import random
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")

import numpy as np
import torch

from ekt_gl.idpc.task import CompactIDPCGraph
from ekt_gl.idpc.v2.checkpoint import Checkpoint, model_state, restore_model
from ekt_gl.idpc.v2.config import Config
from ekt_gl.idpc.v2.core import EvaluationTracker, REQUESTS, seed_for
from ekt_gl.idpc.v2.gpa_fast import PackedGPA, warmup
from ekt_gl.idpc.v2.local_search import DirectedPairs
from ekt_gl.idpc.research.beam import lower_bound
from ekt_gl.idpc.research.engine import ARMS, ResearchEvolution, transfer
from ekt_gl.idpc.research.learning import IndependentLearning, ResearchLearning, feature_set
from ekt_gl.idpc.research.neighborhoods import ObservedPairs
from experiments.run_idpc_v2 import ROOT, finish, manifest_data, restore_rng, rng_state, run_suite, utc
from experiments._support.idpc_v3_archive import (
    CONFIRMATION, PILOT, RESEARCH, independently_validate, json_read, sha, structured,
)


def run_research(checkpoint, method, seed, graphs, names, config, fingerprint,
                 features, decoders, variant, diagnostic=False, lower=None, pause_after=None,
                 prepared_pairs=None):
    start=time.perf_counter()
    random.seed(seed_for(seed,"global"))
    np.random.seed(seed_for(seed,"numpy")%2**32)
    torch.manual_seed(seed_for(seed,"torch"))
    if config.device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed_for(seed,"cuda"))
        torch.cuda.reset_peak_memory_stats(config.device)
    tracker=EvaluationTracker(graphs,config.budget(len(graphs)),config.gpa_cache_per_task,decoders)
    evolution=ResearchEvolution(method,graphs,names,config,seed,tracker,
                                variant=variant,diagnostic=diagnostic)
    if method=="ektgl" and variant.decoder!="cost_beam":
        network=(IndependentLearning(graphs,features,config,seed,variant) if variant.fusion=="independent"
                 else ResearchLearning(graphs,features,config,seed,variant,diagnostic))
    else:
        network=None
    pairs_type=ObservedPairs if diagnostic else DirectedPairs
    pairs=prepared_pairs if prepared_pairs is not None else [pairs_type(g,config.ls_beam_cache_per_task) for g in graphs]
    for pair in pairs:
        pair.beam_cache.clear()
    saved=checkpoint.get("state")
    timing=dict(wall_clock_s=0.0,train_time_s=0.0,inference_time_s=0.0,local_search_time_s=0.0)
    transfers=0
    started=utc()
    if saved is not None:
        evolution.restore(saved["evolution"])
        timing,transfers,started=saved["timing"],saved["transfers"],saved["started"]
        if network is not None:
            restore_model(network,checkpoint.get("model"),torch)
        restore_rng(saved["rng"])

    def record(event,offspring,save_model):
        nonlocal start
        timing["wall_clock_s"]+=time.perf_counter()-start
        timing["local_search_time_s"]=tracker.totals()["ls_time_s"]
        trace=dict(generation=evolution.generation,suite_evaluations=tracker.requests,
                   best_costs=[s.cost for s in evolution.best],populations=evolution.trace(),
                   evaluations=[c[REQUESTS] for c in tracker.per_task],transfer=event,
                   ordinary_offspring=offspring)
        result=None
        if evolution.generation==config.generations or tracker.exhausted:
            result=finish(evolution,tracker,timing,started,transfers,network,fingerprint)
            for g,r in zip(graphs,result["tasks"]):
                if r["feasible"] and independently_validate(g,r["best_path_edge_ids"])!=r["best_cost"]:
                    raise ValueError(f"Failed independent path check {r['task']}")
        state=dict(evolution=evolution.snapshot(),timing=timing,started=started,
                   transfers=transfers,rng=rng_state())
        checkpoint.commit(state,trace,model_state(network,torch) if save_model else None,result)
        start=time.perf_counter()
        return result

    if saved is None:
        evolution.initialize()
        result=record(None,0,network is not None)
        if result:
            return result
    for g in range(evolution.generation+1,config.generations+1):
        if pause_after is not None and g>pause_after:
            return None
        evolution.generation=g
        for pair in pairs:
            pair.beam_cache.clear()
        tracker.clear_caches()
        if method=="ektgl" and g%config.transfer_interval==0:
            event=transfer(evolution,network,pairs,lower,diagnostic)
            transfers+=1
            timing["train_time_s"]+=event["train_time_s"]
            timing["inference_time_s"]+=event["inference_time_s"]
            result=record(event,0,network is not None and event["train_loss"] is not None)
        else:
            before=tracker.totals()["offspring_evaluations"]
            evolution.ordinary(pairs)
            result=record(None,tracker.totals()["offspring_evaluations"]-before,False)
        if g%25==0 or result:
            print(f"{method}/{seed} gen {g} requests {tracker.requests} solver {timing['wall_clock_s']:.2f}s",flush=True)
        if result:
            return result
    raise RuntimeError("Run ended with no committed result")


def arguments(argv):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=("pilot","confirmation","smoke"),required=True)
    p.add_argument("--profile", choices=("native","matched_ls","matched_init_ls"),required=True)
    p.add_argument("--arms",nargs="+",default=["A0"])
    p.add_argument("--methods",nargs="+",choices=("ea","mfea","ektgl"),default=["ektgl"])
    p.add_argument("--device",default="cuda")
    p.add_argument("--threads",type=int,default=4)
    p.add_argument("--out",type=Path,default=RESEARCH/"11_checkpoints")
    p.add_argument("--only-seeds",nargs="+",type=int)
    p.add_argument("--resume",action="store_true")
    p.add_argument("--diagnostic",action="store_true")
    p.add_argument("--smoke-files",nargs="+",type=Path)
    p.add_argument("--smoke-generations",type=int,default=12)
    p.add_argument("--smoke-population",type=int,default=6)
    p.add_argument("--pause-after",type=int)
    p.add_argument("--frozen-config",type=Path)
    a=p.parse_args(argv)
    if a.phase=="smoke" and not a.smoke_files:
        p.error("--smoke-files is required for smoke")
    if a.phase!="smoke" and a.smoke_files:
        p.error("A partial suite is allowed for smoke only")
    if a.phase=="confirmation" and not a.frozen_config:
        p.error("--frozen-config is required for confirmation")
    return a


def campaign_identity(files,graphs,config,seeds,phase,profile,arm,variant,diagnostic):
    identity=manifest_data(files,graphs,config,seeds,"all42" if len(graphs)==42 else "smoke",arm)
    identity.update(phase=phase,comparison_profile=profile,arm=arm,variant=asdict(variant),
                    diagnostic=diagnostic,source_root="V2 frozen shared algorithms + isolated research modules")
    files=[Path(__file__).resolve(),*ROOT.glob("ekt_gl/idpc/research/*.py"),
           ROOT/"ekt_gl/idpc/decoding.py"]
    identity["sources"].update({f.relative_to(ROOT).as_posix():sha(f) for f in files})
    return dict(identity,fingerprint=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest())


def check_frozen():
    original=json_read(RESEARCH/"01_baseline_audit/baseline_freeze_manifest.json")
    for manifest in original["profiles"].values():
        for path,digest in manifest["sources"].items():
            if sha(ROOT/path)!=digest:
                raise ValueError(f"Frozen baseline mismatch {path}")
    return original["profiles"]["native"]


def main(argv=None):
    cli=arguments(argv)
    if cli.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA build is not visible")
    torch.set_num_threads(cli.threads)
    torch.use_deterministic_algorithms(True)
    reference=check_frozen()
    seeds=PILOT if cli.phase=="pilot" else CONFIRMATION if cli.phase=="confirmation" else (4200,)
    chosen=tuple(cli.only_seeds or seeds)
    if len(set(chosen))!=len(chosen) or not set(chosen)<=set(seeds):
        raise ValueError(f"Invalid seeds for {cli.phase}")
    files=[ROOT/r["path"] for r in reference["tasks"]] if cli.phase!="smoke" else [p.resolve() for p in cli.smoke_files]
    if cli.phase!="smoke" and any(sha(f)!=r["sha256"] for f,r in zip(files,reference["tasks"])):
        raise ValueError("Dataset mismatch")
    names=[f.stem for f in files]
    print(f"Loading {len(names)} graphs, phase={cli.phase} profile={cli.profile}",flush=True)
    graphs=[CompactIDPCGraph.from_file(f) for f in files]
    warmup()
    decoders=[PackedGPA(g) for g in graphs]
    registry=dict(ARMS)
    if cli.frozen_config:
        chosen_config=json_read(cli.frozen_config)
        if tuple(chosen_config["confirmation_seeds"])!=CONFIRMATION:
            raise ValueError("Unexpected confirmation seeds")
        for path,digest in chosen_config["sources"].items():
            if sha(ROOT/path)!=digest:
                raise ValueError(f"Confirmation source mismatch {path}")
        registry["V3"]=registry[chosen_config["selected_arm"]]
        if cli.profile!="native":
            registry["V3"]=replace(registry["V3"],ls="2opt")
    features=cached_feature=lower=None
    canonical_pairs=[DirectedPairs(g) for g in graphs]
    failure=False
    for arm in cli.arms:
        if arm not in registry:
            raise ValueError(f"Unknown arm {arm}")
        variant=registry[arm]
        if cli.profile=="matched_init_ls":
            variant=replace(variant,matched_init=True)
        if cli.phase=="confirmation" and arm not in ("A0","V3"):
            raise ValueError("Confirmation accepts only frozen baselines and V3")
        config=Config(profile="native" if cli.profile=="native" else "matched_ls",
                      population_per_task=cli.smoke_population if cli.phase=="smoke" else 30,
                      generations=cli.smoke_generations if cli.phase=="smoke" else 300,
                      hidden_dim=variant.width,device=cli.device,threads=cli.threads,gpa_backend="numba")
        folder=cli.out.resolve()/cli.phase/cli.profile/arm
        folder.mkdir(parents=True,exist_ok=True)
        manifest=campaign_identity(files,graphs,config,seeds,cli.phase,cli.profile,arm,variant,cli.diagnostic)
        path=folder/"manifest.json"
        if path.exists():
            if not cli.resume or json_read(path)!=manifest:
                raise ValueError(f"Use --resume with matching configuration for {folder}")
        else:
            structured(path,manifest)
        if "ektgl" in cli.methods and variant.decoder!="cost_beam" and cached_feature!=variant.features:
            features=None
            if cli.device.startswith("cuda"):
                torch.cuda.empty_cache()
            features=[feature_set(g,cli.device,variant.features) for g in graphs]
            cached_feature=variant.features
        if (variant.decoder.endswith("beam") or cli.diagnostic) and lower is None:
            lower=[lower_bound(g) for g in graphs]
        fp=manifest["fingerprint"]
        observed=[ObservedPairs(g) for g in graphs] if cli.diagnostic else canonical_pairs
        for seed in chosen:
            for method in cli.methods:
                if method!="ektgl" and arm!="A0":
                    raise ValueError("EA and MFEA baselines run only with A0")
                with closing(Checkpoint(folder/f"{method}_{seed}.sqlite",(fp,method,seed))) as checkpoint:
                    if checkpoint.get("status")=="complete":
                        print(f"{arm} {method}/{seed} already complete",flush=True)
                        continue
                    print(f"{arm} {method}/{seed} started",flush=True)
                    checkpoint.status("running")
                    try:
                        if method=="mfea" or (method=="ea" and not variant.matched_init):
                            result=run_suite(checkpoint,method,seed,graphs,names,config,fp,None,
                                             canonical_pairs,cli.pause_after,25,decoders)
                        else:
                            result=run_research(checkpoint,method,seed,graphs,names,config,fp,features,
                                                decoders,variant,cli.diagnostic,lower,cli.pause_after,observed)
                        print(f"{arm} {method}/{seed} {'complete' if result else 'paused'}",flush=True)
                    except Exception as error:
                        checkpoint.status("failed",repr(error))
                        failure=True
                        import traceback
                        traceback.print_exc()
    return int(failure)


if __name__=="__main__":
    raise SystemExit(main())
