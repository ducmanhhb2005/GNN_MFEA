"""Verify the technical online repeats performed after the frozen decoder replay."""

from datetime import datetime

from ekt_gl.idpc.v2.checkpoint import read_checkpoint
from experiments._support.idpc_v3_archive import PILOT, RESEARCH, json_read, table
from experiments._support.idpc_v3_prefix_check import normalize


CHECKPOINTS = RESEARCH / "11_checkpoints"
ARM_PATH = "pilot/matched_ls"


def completed(root, arm, seed):
    folder = root / ARM_PATH / arm
    identity = json_read(folder / "manifest.json")["fingerprint"]
    record, trace = read_checkpoint(folder / f"ektgl_{seed}.sqlite", identity, "ektgl", seed)
    if record["status"] != "complete":
        raise ValueError(f"Incomplete post-replay validation for {arm}/{seed}")
    return record["result"], trace, identity


def main():
    replay_last = max(datetime.fromisoformat(completed(CHECKPOINTS/"detailed_diagnostics", "A0", seed)[0]["completed_at"])
                      for seed in PILOT)
    rows = []
    for arm in ("A0", "D1", "D2", "D3"):
        for seed in PILOT:
            control, original, fingerprint = completed(CHECKPOINTS, arm, seed)
            repeat, trace, repeat_identity = completed(CHECKPOINTS/"post_replay_online", arm, seed)
            exact = normalize(control) == normalize(repeat) and normalize(original) == normalize(trace)
            after = datetime.fromisoformat(repeat["started_at"]) > replay_last
            rows.append(dict(arm=arm, seed=seed, matched=exact, completed_after_replay=after,
                             traces=len(trace), fingerprint=fingerprint,
                             matched_manifest=fingerprint == repeat_identity,
                             started=repeat["started_at"], completed=repeat["completed_at"]))
    table(RESEARCH/"08_v3_pilot/beam_search/post_replay_online_verification.csv", rows)
    if not all(r["matched"] and r["completed_after_replay"] and r["matched_manifest"] for r in rows):
        raise ValueError("At least one post-replay online repeat differs")
    print("Twelve online repeats match paths, objective ledger and transfer outcomes after all42 replay.")


if __name__ == "__main__":
    main()
