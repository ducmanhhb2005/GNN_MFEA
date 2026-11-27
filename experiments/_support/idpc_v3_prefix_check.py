"""Check an accelerated pilot against its independent, full-prefix control."""

from pathlib import Path
import argparse

from ekt_gl.idpc.v2.checkpoint import read_checkpoint
from experiments._support.idpc_v3_archive import RESEARCH, json_read, structured


IGNORED = {
    "started_at", "completed_at", "wall_clock_s", "train_time_s", "inference_time_s",
    "local_search_time_s", "cpu_peak_process_bytes", "gpu_peak_allocated_bytes", "fingerprint",
}


def normalize(value):
    if isinstance(value, dict):
        return {k: normalize(v) for k, v in value.items() if k not in IGNORED and not k.endswith("ls_time_s")}
    if isinstance(value, (list, tuple)):
        return [normalize(v) for v in value]
    return value


def check(base, replay, seed):
    paths = [(d / "pilot/matched_ls/A0") for d in (base, replay)]
    data = []
    for folder in paths:
        fingerprint = json_read(folder / "manifest.json")["fingerprint"]
        meta, traces = read_checkpoint(folder / f"ektgl_{seed}.sqlite", fingerprint, "ektgl", seed)
        if meta["status"] != "complete":
            raise ValueError(f"Still incomplete {folder}")
        data.append((normalize(meta["result"]), normalize(traces)))
    actual, expected = data
    exact = actual == expected
    structured(RESEARCH / "06_ablation/shared_prefix_parity.json",
               dict(seed=seed, exact=exact, trace_entries=[len(r[1]) for r in data],
                    result_matches=actual[0] == expected[0], paths=[str(r) for r in paths]))
    if not exact:
        raise ValueError("Branched pilot cannot replace an independent run")
    print(f"A0/{seed} exact across task solutions, counters, generation traces and transfer outputs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=4200)
    args = parser.parse_args()
    check(RESEARCH/"11_checkpoints", RESEARCH/"11_checkpoints/shared_prefix_arms", args.seed)
