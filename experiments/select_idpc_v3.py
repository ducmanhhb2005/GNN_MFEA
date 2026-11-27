"""Assess frozen score replays and freeze a supported EKT-GL configuration."""

from collections import defaultdict
from dataclasses import asdict, replace
from pathlib import Path
import argparse
import math

from experiments._support.idpc_v3_archive import CONFIRMATION, PILOT, RESEARCH, ROOT, json_read, read, sha, structured, write
from experiments.report_idpc_v3 import COMPONENTS, put
from ekt_gl.idpc.research.engine import ARMS
from ekt_gl.idpc.v2.config import Config


REPLAY_ARMS = ("A0", "D1", "D2", "D3", "legacy_greedy")
NOT_CANDIDATES = {"A0", "D1", "D2", "A7", "A8_identity", "A8_independent", "A8_shuffled", "A9_legacy_32"}
REPLAY_SOURCE = "08_v3_pilot/beam_search/decoder_replay.csv"


def replay_outcomes(folder):
    path = folder / REPLAY_SOURCE
    if not path.is_file():
        return {}, []
    examples = defaultdict(dict)
    all_arms = set()
    for r in read(path):
        key = int(r["seed"]), int(r["generation"]), r["task"]
        before = float(r["best_before"])
        post = float(r["post_cost"]) if r["post_cost"] else math.inf
        gain = max(0, (before - post) / max(abs(before), 1)) * 100 if math.isfinite(before) else None
        examples[key][r["variant"]] = dict(improvement=gain, before=before, candidate=post,
                                           expansions=int(r["expansions"]), requests=int(r["requests"]),
                                           inserted=r["beats_worst"] == "True")
        all_arms.add(r["variant"])
    signatures = {}
    outcomes = []
    for arm in sorted(all_arms):
        for seed in PILOT:
            observations = [block[arm] for k, block in examples.items() if k[0] == seed and arm in block]
            if observations:
                gains = [r["improvement"] for r in observations if r["improvement"] is not None]
                # A construction failure contributes zero archive improvement.
                record = dict(decoder=arm, seed=seed, events=len(observations),
                              finite_archive_events=len(gains), mean_archive_gain_pct=sum(gains)/len(gains) if gains else 0.,
                              chromosome_paths=sum(math.isfinite(r["candidate"]) for r in observations),
                              beats_worst=sum(r["inserted"] for r in observations),
                              calls_improving_archive=sum(r > 0 for r in gains),
                              gained_feasibility=sum(not math.isfinite(r["before"]) and math.isfinite(r["candidate"])
                                                     for r in observations),
                              mean_expansions=sum(r["expansions"] for r in observations)/len(observations),
                              objective_requests=sum(r["requests"] for r in observations))
                signatures[arm, seed] = record
                outcomes.append(record)
    put(folder / "08_v3_pilot/beam_search/replay_summary.csv", outcomes)
    exact_replay = {row["seed"] for row in outcomes if row["decoder"] == "D3" and row["events"] == 84}
    positive = len(exact_replay) == len(PILOT) and all(
        signatures["D3", seed]["mean_archive_gain_pct"] >
        signatures[control, seed]["mean_archive_gain_pct"]
        for seed in PILOT for control in ("D1", "D2"))
    return dict(replay_complete=len(exact_replay)==len(PILOT), trained_positive=positive), outcomes


def per_seed(rows):
    index = defaultdict(dict)
    for r in rows:
        if r["profile"] == "matched_ls" and r["method"] == "ektgl":
            index[r["arm"], int(r["seed"])][r["task"]] = float(r["cost"])
    gains = defaultdict(list)
    for arm in COMPONENTS:
        for seed in PILOT:
            a, b = index[arm, seed], index["A0", seed]
            if len(a) == 42 and a.keys() == b.keys() and all(math.isfinite(a[t]) and math.isfinite(b[t]) for t in a):
                gains[arm].append(sum((b[t] - a[t])/max(abs(b[t]), 1)*100 for t in a)/len(a))
    return gains


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=RESEARCH)
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    folder = args.out
    replay, samples = replay_outcomes(folder)
    raw = folder / "06_ablation/raw_results.csv"
    effects = per_seed(read(raw)) if raw.is_file() else {}
    rows = []
    complete = all(len(effects.get(a, ())) == len(PILOT) for a in COMPONENTS)
    if samples and replay.get("replay_complete") and complete:
        online_trained = all(t > control for c in ("D1", "D2") for t, control in zip(effects["D3"], effects[c]))
        replay["online_trained_positive"] = online_trained
    eligible = {}
    for arm, vector in sorted(effects.items()):
        if not vector:
            continue
        no_gnn_claim = arm != "D3" or replay.get("trained_positive") and replay.get("online_trained_positive")
        viable = arm not in NOT_CANDIDATES and len(vector) == len(PILOT) and min(vector) > 0 and no_gnn_claim
        rows.append(dict(arm=arm, paired_seeds=len(vector), paired_effects=" ".join(f"{x:.8f}" for x in vector),
                         mean_gain_pct=sum(vector)/len(vector), eligible=bool(viable)))
        if viable:
            eligible[arm] = sum(vector)/len(vector)
    put(folder / "08_v3_pilot/selection_evidence.csv", rows)
    result = ["# Selection using pilot seeds only", "",
              "A candidate must preserve shared graph learning, cross-task attention and independent task populations. "
              "A change must reduce the mean task-relative cost at the exact cap on every pilot seed. "
              "A trained decoder must also demonstrate archive gains over both same-cap controls in frozen "
              "replay and online. A cost-only control cannot establish a graph-learning benefit.", "",
              f"Full-control coverage is {complete}. Frozen replay checks yielded {replay}.", ""]
    if not complete or not replay.get("replay_complete"):
        result.append("Selection is pending the complete pilot matrix and paired replay.")
    elif not eligible:
        result.append("The complete pilot did not produce an eligible V3 configuration.")
    else:
        costs = defaultdict(list)
        for r in read(folder / "06_ablation/suite_workload.csv"):
            if r["profile"] == "matched_ls" and r["method"] == "ektgl":
                costs[r["arm"]].append(float(r["wall_clock_s"]))
        def priority(arm):
            seconds = costs[arm]
            average = sum(seconds) / len(seconds) if seconds else math.inf
            return eligible[arm], -average
        chosen = max(sorted(eligible), key=priority)
        result.append(f"Best eligible single intervention **{chosen}**, mean paired gain **{eligible[chosen]:.3f}%**.")
        if args.freeze:
            config = folder / "13_configs/frozen_v3.json"
            if config.exists():
                raise ValueError(f"Freeze already exists at {config}")
            reference = json_read(folder / "11_checkpoints/pilot/matched_ls/A0/manifest.json")
            current = {name: sha(ROOT / name) for name in reference["sources"]}
            if current != reference["sources"]:
                raise ValueError("Pilot source does not match the current candidate source")
            identities = sorted({row["artifact"] for row in json_read(folder / "coverage.json")
                                 if row["phase"] == "pilot" and row["status"] == "complete"})
            required = json_read(folder / "experiment_status.json")["checkpoints"]["pilot"]
            if required.get("complete", 0) != sum(required.values()):
                raise ValueError("Pilot coverage gate is incomplete")
            variant = ARMS[chosen]
            profiles = {"native": variant, "matched_ls": replace(variant, ls="2opt"),
                        "matched_init_ls": replace(variant, ls="2opt", matched_init=True)}
            effective = {name: dict(variant=asdict(policy), solver=asdict(Config(
                profile="native" if name == "native" else "matched_ls", hidden_dim=policy.width,
                device="cuda", gpa_backend="numba"))) for name, policy in profiles.items()}
            memo_report = folder / "08_v3_pilot/local_search/n3_memoization_parity.json"
            memo = json_read(memo_report) if memo_report.is_file() else {}
            accelerated = chosen == "N3" and memo.get("equivalent") is True
            if accelerated:
                memo_source = "experiments/run_idpc_v3_memoized.py"
                current[memo_source] = sha(ROOT / memo_source)
            snapshot = dict(
                effective_configs=effective, native_subpath_memoization=accelerated,
                selected_arm=chosen, pilot_seeds=list(PILOT), confirmation_seeds=list(CONFIRMATION),
                sources=current, tasks=reference["tasks"], profiles=["native", "matched_ls", "matched_init_ls"],
                kept_matched_operator="2opt", matched_objective_requests=379260, max_native_generations=300,
                pilot_checkpoint_artifacts=identities, evidence=rows, trained_decoder_gate=replay,
                pilot_table_sha256=sha(raw),
                selection_source={Path(__file__).relative_to(ROOT).as_posix(): sha(__file__)},
            )
            structured(config, snapshot)
            # JSON is also a valid YAML 1.2 representation.
            structured(folder / "13_configs/EKT_GL_V3_CONFIG.yaml", snapshot)
            status_path = folder / "experiment_status.json"
            status = json_read(status_path)
            status.update(confirmation_frozen=True, selected_arm=chosen)
            structured(status_path, status)
            result.append("The effective sources, tasks, controls and seed list are frozen before confirmation.")
    result += ["", "[Per-arm seed effects](selection_evidence.csv) · [Decoder replay](beam_search/replay_summary.csv)"]
    write(folder / "08_v3_pilot/configuration_selection.md", "\n".join(result) + "\n")


if __name__ == "__main__":
    main()
