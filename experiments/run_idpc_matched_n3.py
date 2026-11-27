"""Matched-N3 on the original V3 confirmation seeds and stopping policy."""
from __future__ import annotations

import argparse
from contextlib import closing, contextmanager
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from experiments import run_idpc_v3 as v3
from experiments._support.idpc_v3_archive import CONFIRMATION, RESEARCH, ROOT, json_read, sha, structured
from experiments.run_idpc_v3_memoized import SUBPATH_CACHE_SIZE, SubpathMemo
from ekt_gl.idpc.research import neighborhoods
from ekt_gl.idpc.research.engine import ResearchEvolution, Variant
from ekt_gl.idpc.research.learning import feature_set
from ekt_gl.idpc.task import CompactIDPCGraph
from ekt_gl.idpc.v2.checkpoint import Checkpoint
from ekt_gl.idpc.v2.config import Config, LocalSearch
from ekt_gl.idpc.v2.gpa_fast import PackedGPA, warmup

OUT = ROOT / "results/idpc_matched_n3"
METHODS = ("ea", "mfea", "ektgl")
BUDGET = 379260
INIT = {"ea": 30, "mfea": 1260, "ektgl": 30}
REQUESTS = "objective_evaluation_requests"
REFERENCE = RESEARCH / "11_checkpoints"
MATCHED = {
    "ea": "final_baselines/ea/confirmation/matched_ls/A0",
    "mfea": "final_baselines/mfea/confirmation/matched_ls/A0",
    "ektgl": "confirmation/matched_ls/V3",
}
NATIVE = "confirmation/native/V3"
PHASES = ("ordinary", "pre_transfer")
LS_FIELDS = (
    "ls_calls", "ls_accepted", "ls_no_improvement", "ls_cost_reduction", "ls_time_s",
    "ls_evaluations", "ls_edge_expansions", "ls_ranking_edge_reads", "ls_cut_attempts",
    "ls_cut_sampling_truncated", "ls_truncated", "ls_complete_paths", "ls_feasible_neighbors",
    "ls_du_rejects", "n3_encode_failures", "n3_roundtrip_failures",
    "ls_subpath_memo_hits", "ls_subpath_memo_misses", "ls_subpath_actual_ranking_reads",
    "ls_subpath_actual_expansions",
)
STATISTICS = {
    "unit": "one seed, averaging task-relative effects over the 42 tasks",
    "primary_metric": "100 * (comparator - candidate) / max(1, abs(comparator))",
    "test": "two-sided paired Wilcoxon signed rank, zero_method=wilcox",
    "multiplicity": "Holm within combat (3), paired operator (3), relative gain contrast (1), transfer (3) families",
    "effect": "paired rank-biserial correlation, positive favors candidate",
    "interval": "10,000 percentile bootstrap resamples of paired seeds, RNG 5291",
    "alpha": 0.05,
    "feasibility": "raw rows retained; aggregate signed tests require all paired objectives finite",
    "timing": "use suite medians, retain recorded elapsed times and timing flags",
}


def reopen_config(data):
    return Config(**{**data, "ls": LocalSearch(**data["ls"])})


def verify_reference():
    """Bind each new arm to the exact historical comparator and N3 implementation."""
    original = v3.check_frozen()
    matched = {m: json_read(REFERENCE / p / "manifest.json") for m, p in MATCHED.items()}
    native = json_read(REFERENCE / NATIVE / "manifest.json")
    sources = {}
    for doc in (*matched.values(), native):
        for path, digest in doc["sources"].items():
            if path in sources and sources[path] != digest:
                raise ValueError(f"V3 sources disagree on {path}")
            sources[path] = digest
    for path, digest in sources.items():
        if sha(ROOT / path) != digest:
            raise ValueError(f"Frozen source changed: {path}")
    tasks = original["tasks"]
    if len(tasks) != 42 or sum(t["subset"] == "set1" for t in tasks) != 24:
        raise ValueError("Expected the exact all42 dataset")
    expected_variant = asdict(Variant())
    for method, doc in matched.items():
        cfg = reopen_config(doc["effective_config"])
        expected_method = dict(population_total=1260, population_count=1 if method == "mfea" else 42,
                               ordinary_ls=True, pre_transfer_ls=method == "ektgl")
        if (doc["tasks"] != tasks or tuple(doc["seeds"]) != CONFIRMATION or doc["variant"] != expected_variant
                or doc["comparison_profile"] != "matched_ls" or cfg.profile != "matched_ls"
                or cfg.budget(len(tasks)) != BUDGET or doc["evaluation_budget"] != BUDGET
                or cfg.population_per_task != 30 or cfg.generations != 300
                or doc["method_config"][method] != expected_method):
            raise ValueError(f"Incompatible paired reference: {method}")
        if cfg.ls != reopen_config(native["effective_config"]).ls:
            raise ValueError(f"V3 native N3 caps differ for {method}")
    if (native["tasks"] != tasks or native["variant"] != asdict(Variant(ls="subpath"))
            or native["runtime_optimization"]["subpath_cache_entries"] != SUBPATH_CACHE_SIZE):
        raise ValueError("The exact V3 Native N3 reference is required")
    for task in tasks:
        if sha(ROOT / task["path"]) != task["sha256"]:
            raise ValueError(f"Dataset changed: {task['task']}")
    for relative in (
        "experiments/run_idpc_matched_n3.py", "ekt_gl/idpc/research/neighborhoods.py",
        "experiments/run_idpc_v3_memoized.py",
    ):
        sources[relative] = sha(ROOT / relative)
    reference_files = [
        "14_reports/ROOT_CAUSE_ANALYSIS.md", "14_reports/FINAL_BENCHMARK_REPORT.md",
        "09_final_benchmark/publication_table_all42.csv", "09_final_benchmark/full_raw_results.csv",
        "09_final_benchmark/suite_workload.csv", "09_final_benchmark/online_transfer_outcomes.csv",
        "09_final_benchmark/runtime_flags.csv",
    ]
    manifest = dict(
        profile="matched_n3", tasks=tasks, seeds=list(CONFIRMATION), sources=sources,
        objective_budget=BUDGET, generation_ceiling=300, init_requests_per_task=INIT,
        caps=matched["ektgl"]["effective_config"]["ls"], local_search="V3 Native replace_segment",
        memoization=dict(entries=SUBPATH_CACHE_SIZE, scope="per-suite, all methods, logical requests still charged"),
        init_policy="original framework-specific initialization",
        stop="earlier of 379260 objective requests or 300 generations",
        statistics=STATISTICS,
        reference_artifacts={name: sha(RESEARCH / name) for name in reference_files},
        native_n3_fingerprint=native["fingerprint"],
        references={
            m: dict(folder=MATCHED[m], fingerprint=doc["fingerprint"], effective_config=doc["effective_config"],
                    invocation=doc["method_config"][m], variant=asdict(Variant(ls="subpath")),
                    historical_manifest_sha256=sha(REFERENCE / MATCHED[m] / "manifest.json"))
            for m, doc in matched.items()
        },
    )
    manifest["fingerprint"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    return manifest


@contextmanager
def n3_observation():
    """Attach accounting to the exact frozen N3, scoped to a single suite."""
    beam = neighborhoods.decode_beam
    original = v3.ResearchEvolution
    memo = SubpathMemo(beam)

    class ObservedN3(ResearchEvolution):
        def improve(self, task, candidate, pairs, rng, phase):
            counts = self.tracker.per_task[task]
            rejections = ("encode_failures", "roundtrip_failures")
            before = {key: counts[key] for key in rejections}
            memo_before = memo.work()
            result = super().improve(task, candidate, pairs, rng, phase)
            for key in rejections:
                delta = counts[key] - before[key]
                counts["n3_" + key] += delta
                counts[phase + "_n3_" + key] += delta
            for key, total in memo.work().items():
                delta = total - memo_before[key]
                counts[key] += delta
                counts[phase + "_" + key] += delta
            return result

    try:
        v3.ResearchEvolution = ObservedN3
        neighborhoods.decode_beam = memo
        yield
    finally:
        v3.ResearchEvolution = original
        neighborhoods.decode_beam = beam


class EventCheckpoint(Checkpoint):
    def __init__(self, path, identity):
        super().__init__(path, identity)
        saved = self.get("state")
        self.previous = saved["evolution"]["counters"] if saved else None

    def commit(self, state, trace, model=None, result=None):
        counters = state["evolution"]["counters"]
        events = []
        for task, count in enumerate(counters):
            for phase in PHASES:
                if self.previous is None:
                    continue
                before = self.previous[task]
                delta = {key: count[phase + "_" + key] - before[phase + "_" + key] for key in LS_FIELDS}
                if delta["ls_calls"]:
                    events.append(dict(task_index=task, phase=phase, **delta))
        trace["n3_events"] = events
        super().commit(state, trace, model, result)
        self.previous = [values.copy() for values in counters]


def prepare(out):
    manifest = verify_reference()
    filename = out / "experiment_manifest.json"
    if filename.exists():
        if json_read(filename) != manifest:
            raise ValueError("Existing Matched-N3 experiment has a different fingerprint")
    else:
        structured(filename, manifest)
        structured(out / "configs/statistical_plan.json", STATISTICS)
        for method, reference in manifest["references"].items():
            structured(out / "configs" / f"{method}_n3.json", reference)
    for relative in ("checkpoints", "logs", "figures", *(f"{m}_n3" for m in METHODS)):
        (out / relative).mkdir(parents=True, exist_ok=True)
    return manifest


def run(method, seeds, out, manifest):
    cfg = reopen_config(manifest["references"][method]["effective_config"])
    if cfg.device.startswith("cuda") and not v3.torch.cuda.is_available():
        raise RuntimeError("V3 EKT-GL CUDA device is required")
    v3.torch.set_num_threads(cfg.threads)
    v3.torch.use_deterministic_algorithms(True)
    print(f"Preparing all42 for {method}_n3 on {cfg.device}", flush=True)
    graphs = [CompactIDPCGraph.from_file(ROOT / t["path"]) for t in manifest["tasks"]]
    names = [t["task"] for t in manifest["tasks"]]
    warmup()
    decoders = [PackedGPA(g) for g in graphs]
    features = [feature_set(g, cfg.device, "v2") for g in graphs] if method == "ektgl" else None
    pairs = [v3.DirectedPairs(g, cfg.ls_beam_cache_per_task) for g in graphs]
    for seed in seeds:
        fingerprint = manifest["fingerprint"]
        filename = out / "checkpoints" / f"{method}_n3_{seed}.sqlite"
        with closing(EventCheckpoint(filename, (fingerprint, method, seed))) as checkpoint:
            if checkpoint.get("status") == "complete":
                print(f"{method}/{seed} already complete", flush=True)
                continue
            checkpoint.status("running")
            print(f"{method}/{seed} started", flush=True)
            try:
                with n3_observation():
                    result = v3.run_research(checkpoint, method, seed, graphs, names, cfg, fingerprint,
                                             features, decoders, Variant(ls="subpath"), prepared_pairs=pairs)
                print(f"{method}/{seed} complete {result['stop_rule']} "
                      f"{result['counters'][REQUESTS]}/{BUDGET}", flush=True)
            except Exception as exc:
                checkpoint.status("failed", repr(exc))
                raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--seeds", nargs="+", type=int, choices=CONFIRMATION)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    if not args.verify_only and (not args.method or not args.seeds):
        parser.error("Supply --method and an explicit subset of the frozen seeds")
    if args.seeds and len(set(args.seeds)) != len(args.seeds):
        parser.error("Seeds must be distinct")
    manifest = prepare(args.out)
    print(f"Verified frozen inputs and reference fingerprints, {manifest['fingerprint']}", flush=True)
    if not args.verify_only:
        run(args.method, args.seeds, args.out, manifest)


if __name__ == "__main__":
    main()
