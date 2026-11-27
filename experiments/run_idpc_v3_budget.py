"""Continue a generation-limited matched pilot to its already declared objective cap."""

import argparse
from contextlib import closing
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

from ekt_gl.idpc.v2.checkpoint import Checkpoint, load
from experiments import run_idpc_v3 as runner
from experiments._support.idpc_v3_archive import PILOT, RESEARCH, ROOT, json_read, sha, structured


BUDGET = 379260


def copy_for_extension(source, target, identity_config, seed, ordinary_only):
    if target.exists():
        return
    old_manifest = json_read(source.parent / "manifest.json")
    with closing(sqlite3.connect(source.resolve().as_uri() + "?mode=ro", uri=True)) as prior:
        meta = {k: load(v) for k, v in prior.execute(
            "SELECT key, value FROM meta WHERE key IN ('identity', 'status', 'result')")}
        if meta["identity"] != (old_manifest["fingerprint"], "ektgl", seed):
            raise ValueError(f"Unexpected checkpoint identity {source}")
        result = meta.get("result")
        if meta.get("status") != "complete" or result["stop_rule"] != "generations":
            raise ValueError(f"{source} did not finish below the objective cap")
        if not result["counters"]["objective_evaluation_requests"] < BUDGET:
            raise ValueError(f"{source} already used its objective budget")
        if ordinary_only and not (result["actual_generations"] == 300 and
                                  BUDGET - result["counters"]["objective_evaluation_requests"] <= 9 * 1260):
            raise ValueError("CPU continuation requires exhausting budget before the next transfer")
        with closing(sqlite3.connect(target)) as destination:
            prior.backup(destination)
    with closing(Checkpoint(target, (old_manifest["fingerprint"], "ektgl", seed))) as extended:
        with extended.connection:
            extended.set("identity", (identity_config["fingerprint"], "ektgl", seed))
            extended.set("result", None)
            extended.set("status", "running")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=runner.ARMS)
    parser.add_argument("--seeds", nargs="+", type=int, choices=PILOT, required=True)
    parser.add_argument("--ordinary-only-on-cpu", action="store_true")
    args = parser.parse_args()
    original = RESEARCH / "11_checkpoints/pilot/matched_ls" / args.arm
    base = RESEARCH / "11_checkpoints/budget_extensions"
    output = base / "pilot/matched_ls" / args.arm
    baseline_manifest = json_read(original / "manifest.json")
    initialize_manifest = runner.campaign_identity
    config_class = runner.Config
    device = "cpu" if args.ordinary_only_on_cpu else "cuda"

    def exact_budget(**options):
        original_config = config_class(**options)
        return replace(original_config, generations=340, budget_override=BUDGET)

    def setup(*args_, **kwargs):
        manifest = initialize_manifest(*args_, **kwargs)
        if manifest["sources"] != baseline_manifest["sources"]:
            raise ValueError("Continuation must use the same algorithm sources")
        manifest["sources"][Path(__file__).relative_to(ROOT).as_posix()] = sha(__file__)
        manifest["prefix_fingerprint"] = baseline_manifest["fingerprint"]
        manifest["extension_reason"] = "nominal generation limit reached below fixed objective cap"
        manifest["extension_device"] = device
        manifest["extension_has_transfer"] = None if device == "cuda" else False
        manifest.pop("fingerprint")
        manifest["fingerprint"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
        for seed in args.seeds:
            path = f"ektgl_{seed}.sqlite"
            copy_for_extension(original / path, output / path, manifest, seed, args.ordinary_only_on_cpu)
        if not (output / "manifest.json").exists():
            structured(output / "manifest.json", manifest)
        return manifest

    runner.Config = exact_budget
    runner.campaign_identity = setup
    suffix = ["--pause-after", "309"] if args.ordinary_only_on_cpu else []
    return runner.main(["--phase", "pilot", "--profile", "matched_ls", "--device", device,
                        "--arms", args.arm, "--only-seeds", *map(str, args.seeds), "--resume",
                        "--out", str(base), *suffix])


if __name__ == "__main__":
    sys.exit(main())
