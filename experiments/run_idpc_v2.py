"""Run, resume and validate the V2 all42 IDPC-DU benchmark."""

from __future__ import annotations

import argparse
from contextlib import closing
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import sys
from time import perf_counter
import traceback

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

from ekt_gl.idpc.task import CompactIDPCGraph
from ekt_gl.idpc.v2.checkpoint import Checkpoint, model_state, restore_model
from ekt_gl.idpc.v2.config import Config, METHODS, SEEDS
from ekt_gl.idpc.v2.core import EvaluationTracker, REQUESTS, initialize, limits_for, seed_for
from ekt_gl.idpc.v2.evolution import Evolution
from ekt_gl.idpc.v2.learning import Features, FEATURE_POLICY, OnlineLearning
from ekt_gl.idpc.v2.local_search import DirectedPairs
from ekt_gl.idpc.v2.transfer import transfer
from idpc_du.gpa import decode_gpa
from idpc_du.validation import validate_path
from mfea.chromosome import Chromosome

ROOT = Path(__file__).resolve().parents[1]


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def json_clean(value):
    if isinstance(value, dict):
        return {key: json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_clean(item) for item in value]
    if isinstance(value, np.generic):
        return json_clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def atomic_json(path, payload):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(json_clean(payload), indent=2, ensure_ascii=False, allow_nan=False),
                    encoding="utf-8")
    temp.replace(path)


def manifest_data(files, graphs, config, seeds, suite, experiment):
    sources = sorted({Path(__file__).resolve(),
                      *ROOT.glob("ekt_gl/idpc/v2/*.py"),
                      ROOT / "ekt_gl/idpc/task.py", ROOT / "ekt_gl/network.py",
                      *ROOT.glob("idpc_du/*.py"), ROOT / "mfea/chromosome.py",
                      ROOT / "mfea/operators.py"})
    graph_rows = [
        dict(task=path.stem, path=os.path.relpath(path, ROOT), sha256=digest(path),
             subset=path.parent.name, n_nodes=graph.n, n_domains=len(graph.domains),
             n_edges=len(graph.edges), source=graph.source, target=graph.target)
        for path, graph in zip(files, graphs)
    ]
    k = len(graphs)
    return dict(protocol="IDPC-DU-V2", experiment_id=experiment, suite_id=suite,
                profile=config.profile, seeds=list(seeds), methods=list(METHODS), tasks=graph_rows,
                sources={path.relative_to(ROOT).as_posix(): digest(path) for path in sources},
                effective_config=asdict(config), evaluation_budget=config.budget(k),
                method_config={m: dict(population_total=config.population_per_task * k,
                                       population_count=1 if m == "mfea" else k,
                                       ordinary_ls=config.ordinary_ls(m), pre_transfer_ls=m == "ektgl")
                               for m in METHODS},
                decoder="CEC GPA Algorithm 1 arriving sigma, file order, empty -1 read as 1; stable USS filter",
                operators="PMX, TPX, Swap+One-Point, tournament 2, stable elitist union",
                feature_policy=FEATURE_POLICY, model_policy="GCN/residual/BatchNorm; QKV; MLP/Adam",
                hardware=dict(platform=platform.platform(), processor=platform.processor(),
                              logical_cpus=os.cpu_count(), device=config.device,
                              gpu=(torch.cuda.get_device_name(config.device)
                                   if config.device.startswith("cuda") else None)),
                dependencies=dict(python=sys.version, numpy=np.__version__, torch=torch.__version__,
                                  cuda=torch.version.cuda,
                                  numba=(__import__("numba").__version__
                                         if config.gpa_backend == "numba" else None)),
                statistical_plan=dict(test="two-sided Mann–Whitney U",
                                      adjustment="Holm across all three pairs × tasks per profile",
                                      alpha=0.05, test_requires_all_feasible=True))


def memory_peak():
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class Memory(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in ("peak", "working", "quota_peak", "quota",
                                                   "nonpaged_peak", "nonpaged", "pagefile", "pagefile_peak")]
        memory = Memory()
        memory.cb = ctypes.sizeof(memory)
        kernel = ctypes.WinDLL("kernel32")
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL("psapi")
        psapi.GetProcessMemoryInfo.argtypes = (wintypes.HANDLE, ctypes.POINTER(Memory), wintypes.DWORD)
        if psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(memory), memory.cb):
            return memory.peak
        return None
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None)


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def finish(evolution, tracker, timing, started, transfers, network, fingerprint):
    tracker.check_invariants()
    tasks = []
    for task, name in enumerate(evolution.names):
        solution = evolution.best[task]
        if solution.feasible:
            verified = validate_path(tracker.graphs[task], solution.path)
            if verified != solution.cost:
                raise ValueError(f"Independent path verification failed for {name}")
        counts = tracker.per_task[task]
        tasks.append(dict(task=name, feasible=solution.feasible, best_cost=solution.cost,
                          best_path_edge_ids=solution.path,
                          best_generation=evolution.best_generation[task], counters=dict(counts)))
    cuda = network is not None and network.config.device.startswith("cuda")
    return dict(method=evolution.method, seed=evolution.seed, status="complete",
                actual_generations=evolution.generation, stop_rule=("budget" if tracker.exhausted else "generations"),
                evaluation_budget=tracker.budget, counters=dict(tracker.totals()),
                rng_stream="blake2b(run seed, task, phase, generation, candidate)",
                fingerprint=fingerprint, tasks=tasks, mating=dict(evolution.mating),
                transfer_events=transfers, model_train_steps=network.steps if network else 0,
                **timing, cpu_peak_process_bytes=memory_peak(),
                gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(network.config.device) if cuda else None,
                started_at=started, completed_at=utc())


def run_suite(checkpoint, method, seed, graphs, names, config, fingerprint, features,
              pairs, pause_after=None, progress=30, decoders=None):
    clock = perf_counter()
    random.seed(seed_for(seed, "global"))
    np.random.seed(seed_for(seed, "numpy") % 2**32)
    torch.manual_seed(seed_for(seed, "torch"))
    if config.device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed_for(seed, "cuda"))
        torch.cuda.reset_peak_memory_stats(config.device)
    tracker = EvaluationTracker(graphs, config.budget(len(graphs)), config.gpa_cache_per_task, decoders)
    evolution = Evolution(method, graphs, names, config, seed, tracker)
    network = OnlineLearning(graphs, features, config) if method == "ektgl" else None
    timing = dict(wall_clock_s=0.0, train_time_s=0.0, inference_time_s=0.0,
                  local_search_time_s=0.0)
    started = utc()
    transfers = 0
    saved = checkpoint.get("state")
    if saved is not None:
        evolution.restore(saved["evolution"])
        timing, started, transfers = saved["timing"], saved["started"], saved["transfers"]
        if network is not None:
            restore_model(network, checkpoint.get("model"), torch)
        restore_rng(saved["rng"])

    def save(event=None, update_model=False, ordinary_offspring=0):
        nonlocal clock
        timing["wall_clock_s"] += perf_counter() - clock
        timing["local_search_time_s"] = tracker.totals()["ls_time_s"]
        done = evolution.generation == config.generations or tracker.exhausted
        trace = dict(generation=evolution.generation, suite_evaluations=tracker.requests,
                     best_costs=[s.cost for s in evolution.best], populations=evolution.trace(),
                     evaluations=[c[REQUESTS] for c in tracker.per_task], transfer=event,
                     ordinary_offspring=ordinary_offspring)
        result = (finish(evolution, tracker, timing, started, transfers, network, fingerprint)
                  if done else None)
        state = dict(evolution=evolution.snapshot(), timing=timing, started=started,
                     transfers=transfers, rng=rng_state())
        checkpoint.commit(state, trace, model_state(network, torch) if update_model else None, result)
        clock = perf_counter()
        return result

    if saved is None:
        evolution.initialize()
        result = save(update_model=network is not None)
        if result:
            return result
    for generation in range(evolution.generation + 1, config.generations + 1):
        if pause_after is not None and generation > pause_after:
            return None
        evolution.generation = generation
        tracker.clear_caches()
        for index in pairs:
            index.beam_cache.clear()
        event, offspring = None, 0
        if network is not None and generation % config.transfer_interval == 0:
            event = transfer(evolution, network, pairs)
            transfers += 1
            timing["train_time_s"] += event["train_time_s"]
            timing["inference_time_s"] += event["inference_time_s"]
        else:
            before = tracker.totals()["offspring_evaluations"]
            evolution.ordinary(pairs)
            offspring = tracker.totals()["offspring_evaluations"] - before
        result = save(event, update_model=event is not None and event["train_loss"] is not None,
                      ordinary_offspring=offspring)
        if progress and (generation % progress == 0 or result):
            print(f"{method}/{seed} g={generation} eval={tracker.requests} "
                  f"solver={timing['wall_clock_s']:.2f}s", flush=True)
        if result:
            return result
    raise RuntimeError("Run did not record a stop condition")


def arguments(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("native", "matched_ls"), required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--instances", nargs="+", type=Path)
    parser.add_argument("--generations", type=int)
    parser.add_argument("--population", type=int)
    parser.add_argument("--budget", type=int)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--methods", choices=METHODS, nargs="+", default=METHODS)
    parser.add_argument("--only-seeds", type=int, nargs="+")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--gpa-backend", choices=("reference", "numba"), default="reference")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--pause-after", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress", type=int, default=30)
    args = parser.parse_args(argv)
    if not args.debug and any(x is not None for x in (
            args.instances, args.generations, args.population, args.budget)):
        parser.error("Instance and algorithm overrides require --debug")
    if len(set(args.methods)) != len(args.methods):
        parser.error("Method list contains duplicates")
    if args.only_seeds and (len(set(args.only_seeds)) != len(args.only_seeds) or
                           not set(args.only_seeds) <= set(SEEDS)):
        parser.error("Select distinct seeds from the frozen list 100–119")
    return args


def main(argv=None):
    args = arguments(argv)
    config = Config(profile=args.profile, threads=args.threads, device=args.device,
                    gpa_backend=args.gpa_backend)
    config = replace(config,
                     population_per_task=args.population if args.population is not None else config.population_per_task,
                     generations=args.generations if args.generations is not None else config.generations,
                     budget_override=args.budget)
    torch.set_num_threads(config.threads)
    torch.use_deterministic_algorithms(True)
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("The selected CUDA device is unavailable in this PyTorch build")
    files = ([p.resolve() for p in args.instances] if args.instances else
             sorted((ROOT / "data/benchmark").glob("set*/*.idpc")))
    if not args.debug and len(files) != 42:
        raise ValueError("The all42 suite must contain exactly 42 original instances")
    names = [p.stem for p in files]
    if len(names) != len(set(names)):
        raise ValueError("Task names must be unique")
    print(f"Loading {len(files)} graphs on {config.device}; profile={config.profile}", flush=True)
    graphs = [CompactIDPCGraph.from_file(path) for path in files]
    suite = "debug" if args.debug else "all42"
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    checkpoints = out / "checkpoints"
    manifest = manifest_data(files, graphs, config, SEEDS, suite, out.name)
    fingerprint = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    manifest["fingerprint"] = fingerprint
    manifest_file = out / "manifest.json"
    if manifest_file.exists():
        if not args.resume:
            raise ValueError("Output already has a manifest; select --resume or a new directory")
        if json.loads(manifest_file.read_text(encoding="utf-8")) != manifest:
            raise ValueError("Existing experiment does not match effective code/config/data")
    else:
        atomic_json(manifest_file, manifest)
    print(json.dumps({key: manifest[key] for key in (
        "fingerprint", "suite_id", "profile", "method_config", "effective_config", "seeds",
        "evaluation_budget")}, ensure_ascii=False), flush=True)
    checkpoints.mkdir(exist_ok=True)
    features = ([Features.build(graph, config.device) for graph in graphs]
                if "ektgl" in args.methods else None)
    pairs = [DirectedPairs(graph, config.ls_beam_cache_per_task) for graph in graphs]
    if config.gpa_backend == "numba":
        from ekt_gl.idpc.v2.gpa_fast import PackedGPA, warmup
        warmup()
        decoders = [PackedGPA(graph) for graph in graphs]
        for task, graph, decoder in zip(names, graphs, decoders):
            probe_rng = random.Random(seed_for(0, task, "decoder-check"))
            limits = limits_for([graph])
            for _ in range(8):
                genome = initialize(limits, probe_rng)
                reference = Chromosome(genome.priority, [max(1, x) for x in genome.edge_index])
                if decoder(genome) != decode_gpa(graph, reference):
                    raise ValueError(f"GPA backend differs from reference on {task}")
        print(f"GPA backend checked on {8*len(graphs)} independent probes", flush=True)
    else:
        decoders = None
    had_error = False
    for seed in args.only_seeds or SEEDS:
        for method in args.methods:
            path = checkpoints / f"{method}_{seed}.sqlite"
            with closing(Checkpoint(path, (fingerprint, method, seed))) as checkpoint:
                status = checkpoint.get("status")
                if status == "complete":
                    print(f"{method}/{seed} already complete", flush=True)
                    continue
                try:
                    checkpoint.status("running")
                    result = run_suite(checkpoint, method, seed, graphs, names, config, fingerprint,
                                       features, pairs, args.pause_after, args.progress, decoders)
                    print(f"{method}/{seed} {'complete' if result else 'paused'}", flush=True)
                except Exception:
                    had_error = True
                    checkpoint.status("failed", traceback.format_exc())
                    traceback.print_exc()
    return int(had_error)


if __name__ == "__main__":
    raise SystemExit(main())
