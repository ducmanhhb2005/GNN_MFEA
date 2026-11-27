"""Run the N3 comparison at exactly 300 generations with no objective cap."""
from __future__ import annotations

import argparse
from contextlib import closing
import hashlib
import json
from pathlib import Path

from experiments._support.idpc_v3_archive import CONFIRMATION, ROOT, json_read, sha, structured
from experiments.run_idpc_matched_n3 import (
    EventCheckpoint, METHODS, REQUESTS, n3_observation, reopen_config, v3, verify_reference,
)
from ekt_gl.idpc.research.engine import Variant
from ekt_gl.idpc.research.learning import feature_set
from ekt_gl.idpc.task import CompactIDPCGraph
from ekt_gl.idpc.v2.gpa_fast import PackedGPA, warmup

OUT = ROOT / "results/idpc_n3_300gen"


class GenerationConfig(v3.Config):
    """Apply N3 to all three methods and account for evaluation costs without a cap."""

    def budget(self, task_count):
        return None


def generation_config(effective):
    previous = reopen_config(effective)
    return GenerationConfig(**vars(previous))


def prepare(out):
    reference = verify_reference()
    settings = {
        key: reference[key]
        for key in ("tasks", "seeds", "sources", "init_requests_per_task", "caps",
                    "local_search", "memoization", "init_policy", "native_n3_fingerprint")
    }
    settings.update(profile="idpc_n3_300gen", objective_budget=None, generations=300,
                    population_per_task=30, unified_population=1260, transfer_interval=10,
                    training_epochs=1, stop="exactly 300 generations")
    settings["statistics"] = {
        **reference["statistics"],
        "multiplicity": "Holm across the three pairwise comparisons",
    }
    relative = Path(__file__).resolve().relative_to(ROOT).as_posix()
    settings["sources"][relative] = sha(__file__)
    settings["references"] = reference["references"]
    for method, ref in settings["references"].items():
        config = generation_config(ref["effective_config"])
        if (config.budget(42) is not None or config.population_per_task != 30
                or config.generations != 300 or config.transfer_interval != 10
                or config.train_epochs != 1 or not config.ordinary_ls(method)):
            raise ValueError(f"Incompatible uncapped framework settings: {method}")
        ref.update(active_objective_cap=None, active_generation_limit=300)
    settings["fingerprint"] = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    path = out / "experiment_manifest.json"
    if path.exists():
        if json_read(path) != settings:
            raise ValueError("Existing generation campaign has a different fingerprint")
    else:
        structured(path,settings)
        structured(out/"configs/statistical_plan.json", settings["statistics"])
        for m, ref in settings["references"].items():
            structured(out/"configs"/f"{m}_n3_generation.json", ref)
    for name in ("checkpoints", "logs", "figures", *(f"{m}_n3" for m in METHODS)):
        (out / name).mkdir(parents=True, exist_ok=True)
    return settings


def run(method, seeds, out, manifest):
    config = generation_config(manifest["references"][method]["effective_config"])
    if config.device.startswith("cuda") and not v3.torch.cuda.is_available():
        raise RuntimeError("The frozen EKT-GL CUDA device is required")
    v3.torch.set_num_threads(config.threads)
    v3.torch.use_deterministic_algorithms(True)
    print(f"Preparing all42: {method}_n3, 300 generations on {config.device}", flush=True)
    graphs = [CompactIDPCGraph.from_file(ROOT/t["path"]) for t in manifest["tasks"]]
    names = [t["task"] for t in manifest["tasks"]]
    warmup()
    decoders = [PackedGPA(g) for g in graphs]
    features = [feature_set(g, config.device, "v2") for g in graphs] if method=="ektgl" else None
    pairs = [v3.DirectedPairs(g, config.ls_beam_cache_per_task) for g in graphs]
    fingerprint = manifest["fingerprint"]
    for seed in seeds:
        path = out/"checkpoints"/f"{method}_n3_{seed}.sqlite"
        with closing(EventCheckpoint(path, (fingerprint,method,seed))) as checkpoint:
            if checkpoint.get("status") == "complete":
                result = checkpoint.get("result")
                if result["actual_generations"] != 300 or result["evaluation_budget"] is not None:
                    raise ValueError(f"Incompatible terminal checkpoint: {path}")
                print(f"{method}/{seed} already complete", flush=True)
                continue
            checkpoint.status("running")
            print(f"{method}/{seed} started", flush=True)
            try:
                with n3_observation():
                    result = v3.run_research(checkpoint,method,seed,graphs,names,config,fingerprint,
                                             features,decoders,Variant(ls="subpath"),prepared_pairs=pairs)
                if (result["stop_rule"] != "generations" or result["actual_generations"] != 300
                        or result["evaluation_budget"] is not None):
                    raise ValueError("The uncapped suite did not stop at generation 300")
                print(f"{method}/{seed} complete 300 generations {result['counters'][REQUESTS]} requests", flush=True)
            except Exception as error:
                checkpoint.status("failed",repr(error))
                raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only",action="store_true")
    parser.add_argument("--method",choices=METHODS)
    parser.add_argument("--seeds",nargs="+",type=int,choices=CONFIRMATION)
    parser.add_argument("--out",type=Path,default=OUT)
    args=parser.parse_args(argv)
    if not args.verify_only and (not args.method or not args.seeds):
        parser.error("Supply a method and a subset of the registered seeds")
    if args.seeds and len(set(args.seeds))!=len(args.seeds):
        parser.error("Seeds must be unique")
    manifest=prepare(args.out)
    print(f"Verified uncapped generation config {manifest['fingerprint']}",flush=True)
    if not args.verify_only:
        run(args.method,args.seeds,args.out,manifest)


if __name__=="__main__":
    main()
