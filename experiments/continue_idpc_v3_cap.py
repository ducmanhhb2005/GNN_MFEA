"""Continue an EKT-GL checkpoint that hit its generation ceiling before a matched cap."""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

from experiments import run_idpc_v3 as runner
from experiments._support.idpc_v3_archive import RESEARCH, ROOT, json_read, sha, structured
from experiments.run_idpc_v3_budget import BUDGET, copy_for_extension


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--source", required=True, type=Path, help="Directory with the original arm manifest and checkpoint")
    cli.add_argument("--seeds", nargs="+", required=True, type=int)
    args = cli.parse_args()
    source = args.source.resolve()
    source.relative_to(RESEARCH / "11_checkpoints")
    original = json_read(source / "manifest.json")
    phase, profile, arm = (original[k] for k in ("phase", "comparison_profile", "arm"))
    if phase not in ("pilot", "confirmation") or profile not in ("matched_ls", "matched_init_ls"):
        raise ValueError("The fixed-cap continuation is restricted to matched all42 designs")
    if original["diagnostic"]:
        raise ValueError("Complete an observational replay with its diagnostic runner")
    for path, digest in original["sources"].items():
        if sha(ROOT / path) != digest:
            raise ValueError(f"Continuation requires original source {path}")
    config = original["effective_config"]
    destination = RESEARCH / "11_checkpoints/cap_continuations"
    target = destination / phase / profile / arm
    ctor, make_manifest = runner.Config, runner.campaign_identity

    def exact_cap(**provided):
        if any(provided[k] != v for k, v in config.items() if k in provided):
            raise ValueError("This resume requires the exact original optimizer setup")
        return replace(ctor(**provided), generations=340, budget_override=BUDGET)

    def extended_manifest(*provided, **named):
        record = make_manifest(*provided, **named)
        existing = record["sources"]
        if any(original["sources"].get(k) != v for k, v in existing.items()):
            raise ValueError("Core continuation sources differ")
        if record["tasks"] != original["tasks"] or record["variant"] != original["variant"]:
            raise ValueError("Continuation data or variant differs")
        record["sources"] = {**original["sources"], Path(__file__).relative_to(ROOT).as_posix(): sha(__file__),
                             "experiments/run_idpc_v3_budget.py": sha(ROOT/"experiments/run_idpc_v3_budget.py")}
        record["prefix_fingerprint"] = original["fingerprint"]
        record["extension_reason"] = "generation ceiling reached before fixed matched objective cap"
        record.pop("fingerprint")
        record["fingerprint"] = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
        for seed in args.seeds:
            name = f"ektgl_{seed}.sqlite"
            copy_for_extension(source/name, target/name, record, seed, False)
        if not (target/"manifest.json").exists():
            structured(target/"manifest.json", record)
        return record

    runner.Config, runner.campaign_identity = exact_cap, extended_manifest
    extra = ["--frozen-config", str(RESEARCH/"13_configs/frozen_v3.json")] if phase == "confirmation" else []
    return runner.main(["--phase", phase, "--profile", profile, "--arms", arm, "--methods", "ektgl",
                        "--device", config["device"], "--threads", str(config["threads"]), "--resume",
                        "--out", str(destination), "--only-seeds", *map(str,args.seeds), *extra])


if __name__ == "__main__":
    raise SystemExit(main())
