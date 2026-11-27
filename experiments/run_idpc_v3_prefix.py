"""Reuse a verified pre-transfer evolutionary prefix in pilot learning variants."""

from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3

from ekt_gl.idpc.v2.checkpoint import Checkpoint, load
from experiments import run_idpc_v3 as runner
from experiments._support.idpc_v3_archive import PILOT, RESEARCH, ROOT, json_read, sha, structured


PREFIX = RESEARCH / "11_checkpoints/shared_prefix"
BRANCHED = RESEARCH / "11_checkpoints/shared_prefix_arms"
PREPARED = PREFIX / "pilot/matched_ls/D1"


def prefix_rows(db):
    with closing(sqlite3.connect(db.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        return [load(row[0]) for row in connection.execute(
            "SELECT row FROM trace WHERE generation <= 9 ORDER BY generation")]


def branch(source, destination, old_id, new_id):
    if destination.exists():
        return
    with closing(sqlite3.connect(source.resolve().as_uri() + "?mode=ro", uri=True)) as prior:
        metadata = {k: load(v) for k, v in prior.execute(
            "SELECT key, value FROM meta WHERE key IN ('identity', 'status', 'state', 'model')")}
        state = metadata["state"]
        if metadata["identity"] != old_id or metadata["status"] != "running":
            raise ValueError(f"Expected a paused, identified ordinary-evolution checkpoint at {source}")
        if state["transfers"] or state["evolution"]["generation"] != 9 or metadata.get("model") is not None:
            raise ValueError("The shared prefix must stop immediately before the first training event")
        with closing(sqlite3.connect(destination)) as dst:
            prior.backup(dst)
    with closing(Checkpoint(destination, old_id)) as new:
        with new.connection:
            state["rng"]["ordinary_prefix_without_model"] = True
            new.set("state", state)
            new.set("identity", new_id)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", required=True, choices=runner.ARMS)
    parser.add_argument("--only-seeds", nargs="+", type=int, choices=PILOT, default=list(PILOT))
    args = parser.parse_args()
    # Shared ordinary code and local search are required for an identical prefix.
    for arm in args.arms:
        variant = runner.ARMS[arm]
        if variant.ls != "2opt" or variant.matched_init:
            raise ValueError(f"{arm} changes the first nine generations")
    base = json_read(PREPARED/"manifest.json")
    factory = runner.campaign_identity
    restore_model, restore_rng = runner.restore_model, runner.restore_rng

    def check_compatibility(identity):
        if identity["tasks"] != base["tasks"] or identity["sources"] != base["sources"]:
            raise ValueError("Shared prefix source or dataset differs")
        effective = identity["effective_config"]
        initial = base["effective_config"]
        if any(v != initial[k] for k,v in effective.items() if k not in ("device", "threads", "hidden_dim")):
            raise ValueError("Shared prefix ordinary configuration differs")

    def campaign_identity(*args_, **kwargs):
        manifest = factory(*args_, **kwargs)
        check_compatibility(manifest)
        manifest["sources"][Path(__file__).relative_to(ROOT).as_posix()] = sha(__file__)
        manifest["branch_prefix_fingerprint"] = base["fingerprint"]
        manifest["branch_prefix_generation"] = 9
        manifest["prefix_verification"] = manifest["arm"] == "A0"
        manifest.pop("fingerprint")
        manifest["fingerprint"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
        folder = BRANCHED / "pilot/matched_ls" / manifest["arm"]
        for seed in args.only_seeds:
            name = f"ektgl_{seed}.sqlite"
            control = RESEARCH / "11_checkpoints/pilot/matched_ls/A0" / name
            if prefix_rows(PREPARED/name) != prefix_rows(control):
                raise ValueError(f"Ordinary trajectory disagrees before first transfer for seed {seed}")
            branch(PREPARED/name, folder/name, (base["fingerprint"], "ektgl", seed),
                   (manifest["fingerprint"], "ektgl", seed))
        if not (folder/"manifest.json").exists():
            structured(folder/"manifest.json", manifest)
        return manifest

    def deferred_model(network, data, torch):
        if data is not None:
            restore_model(network, data, torch)

    def deterministic_rng(state):
        if state.get("ordinary_prefix_without_model"):
            # run_research has just seeded and created the actual model. Keep its Torch RNG.
            import random
            import numpy as np
            random.setstate(state["python"])
            np.random.set_state(state["numpy"])
        else:
            restore_rng(state)

    runner.campaign_identity = campaign_identity
    runner.restore_model = deferred_model
    runner.restore_rng = deterministic_rng
    return runner.main(["--phase","pilot","--profile","matched_ls","--device","cuda","--resume",
                        "--out",str(BRANCHED),"--arms",*args.arms,"--only-seeds",*map(str,args.only_seeds)])


if __name__ == "__main__":
    raise SystemExit(main())
