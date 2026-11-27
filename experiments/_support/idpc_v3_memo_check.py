"""Verify that memoized N3 replays the unmodified pilot trajectory exactly."""

from ekt_gl.idpc.v2.checkpoint import read_checkpoint
from experiments._support.idpc_v3_archive import RESEARCH, json_read, structured


TIMING_FIELDS = {
    "started_at", "completed_at", "wall_clock_s", "train_time_s", "inference_time_s",
    "local_search_time_s", "cpu_peak_process_bytes", "gpu_peak_allocated_bytes", "fingerprint",
}


def normalized(value):
    if isinstance(value, dict):
        return {k: normalized(v) for k,v in value.items()
                if k not in TIMING_FIELDS and not k.endswith("ls_time_s") and not k.startswith("ls_subpath_")}
    if isinstance(value, (list, tuple)):
        return [normalized(v) for v in value]
    return value


def case(folder):
    dir_ = folder / "pilot/matched_ls/N3"
    fingerprint = json_read(dir_/"manifest.json")["fingerprint"]
    meta, traces = read_checkpoint(dir_/"ektgl_4200.sqlite", fingerprint, "ektgl", 4200)
    if meta["status"] != "complete":
        raise ValueError(f"Incomplete N3 regression {dir_}")
    return meta["result"], traces, fingerprint


def main():
    base = RESEARCH/"11_checkpoints"
    direct, a, direct_id = case(base)
    memo, b, memo_id = case(base/"performance_validation")
    exact = normalized(direct) == normalized(memo) and normalized(a) == normalized(b)
    structured(RESEARCH/"08_v3_pilot/local_search/n3_memoization_parity.json",
               dict(seed=4200, equivalent=exact, baseline_fingerprint=direct_id,
                    memoized_fingerprint=memo_id, trace_rows=len(a),
                    original_wall_clock_s=direct["wall_clock_s"], memoized_wall_clock_s=memo["wall_clock_s"],
                    memoized_search_counters={k:v for k,v in memo["counters"].items()
                                              if k.startswith("ls_subpath_")}))
    if not exact:
        raise ValueError("Memoization changed the trajectory or logical evaluation ledger")
    print("Memoized N3 matches the full raw paths, transfer outputs and objective ledger for seed 4200")


if __name__ == "__main__":
    main()
