"""Run the unchanged uncapped 300-generation protocol with exact accelerated N3."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

from ekt_gl.idpc.research import n3_fast
from ekt_gl.idpc.v2 import core
from experiments import run_idpc_matched_n3 as base
from experiments import run_idpc_matched_n3_generation as generation
from experiments._support.idpc_v3_archive import ROOT, json_read, sha, structured

OUT = ROOT / "results/idpc_n3_300gen_optimized"
AUDIT = ROOT / "results/idpc_n3_performance_audit"


@contextmanager
def optimized_observation():
    beam = n3_fast.CostOnlyBeam()
    original_factory = base.SubpathMemo

    def make_memo(_):
        memo = original_factory(beam)
        original_work = memo.work

        def work():
            values = original_work()
            values["ls_subpath_actual_ranking_reads"] = beam.ranking_reads
            return values

        memo.work = work
        return memo

    with (patch.object(base, "SubpathMemo", make_memo),
          patch.object(base.neighborhoods, "replace_segment", n3_fast.replace_segment),
          patch.object(core, "validate_path", n3_fast.validate_path),
          base.n3_observation()):
        yield


def prepare(out):
    manifest = json_read(generation.OUT / "experiment_manifest.json")
    if manifest["objective_budget"] is not None or manifest["generations"] != 300:
        raise ValueError("Expected the uncapped 300-generation reference")
    manifest["reference_fingerprint"] = manifest.pop("fingerprint")
    manifest.update(profile="idpc_n3_300gen_optimized",
                    implementation="exact_n3_rank_reuse_compiled_filter_and_validator",
                    rank_cache_bytes=n3_fast.RANK_CACHE_BYTES,
                    rank_cached_prefix=n3_fast.RANK_PREFIX)
    for p in (Path(__file__), Path(n3_fast.__file__)):
        manifest["sources"][p.resolve().relative_to(ROOT).as_posix()] = sha(p)
    for path, digest in manifest["sources"].items():
        if sha(ROOT / path) != digest:
            raise ValueError(f"Frozen source changed: {path}")
    for task in manifest["tasks"]:
        if sha(ROOT / task["path"]) != task["sha256"]:
            raise ValueError(f"Frozen dataset changed: {task['task']}")
    manifest["fingerprint"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    path = out / "experiment_manifest.json"
    if path.exists():
        if json_read(path) != manifest:
            raise ValueError("This output already belongs to a different implementation")
    else:
        structured(path, manifest)
        structured(out/"configs/statistical_plan.json", manifest["statistics"])
        for method, reference in manifest["references"].items():
            structured(out/"configs"/f"{method}_n3_generation.json", reference)
    for folder in ("checkpoints", "logs", "figures", *(m+"_n3" for m in base.METHODS)):
        (out / folder).mkdir(parents=True, exist_ok=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=base.METHODS)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    if not args.verify_only and (not args.method or not args.seeds):
        parser.error("Supply a method and seeds")
    manifest = prepare(args.out)
    if args.verify_only:
        print(manifest["fingerprint"], flush=True)
        return
    if len(set(args.seeds)) != len(args.seeds) or not set(args.seeds) <= set(manifest["seeds"]):
        parser.error("Each requested seed must be unique and registered")
    if args.seeds != [4300]:
        approval = json_read(AUDIT/"audit_gate.json")
        if approval["status"] != "PASS" or approval["fingerprint"] != manifest["fingerprint"]:
            raise ValueError("Full campaign needs the targeted audit acceptance for this implementation")
    n3_fast.warmup_filter()
    with patch.object(generation, "n3_observation", optimized_observation):
        generation.run(args.method, args.seeds, args.out, manifest)


if __name__ == "__main__":
    main()
