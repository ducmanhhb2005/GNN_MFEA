"""Profile exact N3 on deterministic continuations of archived population snapshots."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import closing
import cProfile
import io
import linecache
import math
from pathlib import Path
import pstats
import sqlite3
import sys
from time import perf_counter
from unittest.mock import patch

from ekt_gl.idpc.task import CompactIDPCGraph
from ekt_gl.idpc.v2.checkpoint import load
from ekt_gl.idpc.v2.core import EvaluationTracker
from ekt_gl.idpc.v2.gpa_fast import PackedGPA, warmup
from ekt_gl.idpc.research import neighborhoods
from ekt_gl.idpc.research.engine import Variant
from experiments._support.idpc_v3_archive import ROOT, json_read, structured
from experiments import run_idpc_matched_n3 as n3
from experiments.run_idpc_matched_n3_generation import OUT as ARCHIVE, generation_config
from experiments.run_idpc_v3_memoized import SubpathMemo

OUT = ROOT / "results/idpc_n3_performance_audit"


class KeyProbe(SubpathMemo):
    """Count context diversity and distinguish cold misses from LRU evictions."""

    def __init__(self, search):
        super().__init__(search)
        self.seen = set()
        self.endpoints = set()
        self.evicted_misses = 0

    def __call__(self, graph, lower, config, scores=None, **restrictions):
        key = (id(graph), id(lower), config, tuple(sorted(restrictions.items())))
        self.endpoints.add((id(graph), restrictions["start"], restrictions["target"]))
        if key in self.seen and key not in self.cache:
            self.evicted_misses += 1
        self.seen.add(key)
        return super().__call__(graph, lower, config, scores, **restrictions)

    def summary(self):
        return dict(**self.work(), evicted_misses=self.evicted_misses,
                    distinct_contexts=len(self.seen), distinct_endpoints=len(self.endpoints))


class LineProbe:
    """Measure inclusive time at selected beam and N3 source lines."""

    def __init__(self):
        self.codes = {neighborhoods.replace_segment.__code__, neighborhoods.decode_beam.__code__}
        self.seconds = Counter()
        self.calls = Counter()

    def trace(self, frame, event, _argument):
        if event != "call" or frame.f_code not in self.codes:
            return None
        source = frame.f_code.co_filename
        previous = None
        start = perf_counter()

        def record(frame, event, _argument):
            nonlocal previous, start
            end = perf_counter()
            if previous is not None:
                key = source, previous
                self.seconds[key] += end - start
                self.calls[key] += 1
            previous = frame.f_lineno if event == "line" else None
            start = perf_counter()
            return record
        return record

    def output(self):
        return [dict(file=Path(file).name, line=line, hits=self.calls[file,line], seconds=seconds,
                     code=linecache.getline(file,line).strip())
                for (file,line),seconds in self.seconds.most_common()]


def read_state(path, identity):
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        return_value = dict(connection.execute("SELECT key,value FROM meta WHERE key IN ('identity','state')"))
    if load(return_value["identity"]) != identity:
        raise ValueError("Wrong archived population identity")
    return return_value["state"]


def measure(method, seed, state, graphs, names, config, decoders, steps, backend, profiler=None, keys=False):
    archived = load(state)["evolution"]
    tracker = EvaluationTracker(graphs, config.budget(len(graphs)), config.gpa_cache_per_task, decoders)
    pairs = [n3.v3.DirectedPairs(g, config.ls_beam_cache_per_task) for g in graphs]
    before = Counter()
    for counter in archived["counters"]:
        before.update(counter)
    if backend == "optimized":
        from experiments.run_idpc_n3_optimized import optimized_observation
        observation = optimized_observation
    else:
        observation = n3.n3_observation
    memos = []

    def memo_factory(search):
        memo = KeyProbe(search)
        memos.append(memo)
        return memo

    with patch.object(n3, "SubpathMemo", memo_factory if keys else SubpathMemo), observation():
        evolution = n3.v3.ResearchEvolution(method, graphs, names, config, seed, tracker,
                                            variant=Variant(ls="subpath"), diagnostic=False)
        evolution.restore(archived)
        start = perf_counter()
        if profiler:
            profiler.enable()
        for _ in range(steps):
            evolution.generation += 1
            tracker.clear_caches()
            evolution.ordinary(pairs)
        if profiler:
            profiler.disable()
        elapsed = perf_counter() - start
    totals = tracker.totals()
    fields = ("ls_time_s", "ls_calls", "ls_evaluations", "ls_edge_expansions", "ls_ranking_edge_reads",
              "ls_subpath_actual_expansions", "ls_subpath_actual_ranking_reads",
              "ls_subpath_memo_hits", "ls_subpath_memo_misses")
    return dict(method=method, seed=seed, snapshot_generation=archived["generation"], steps=steps,
                backend=backend, seconds=elapsed,
                counters={k: totals[k]-before[k] for k in fields},
                keys=memos[0].summary() if memos else None,
                requests=tracker.requests,
                best=[(solution.cost if math.isfinite(solution.cost) else None, solution.path)
                      for solution in evolution.best],
                state=evolution.snapshot())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", choices=("ea", "mfea"), default=["ea", "mfea"])
    parser.add_argument("--seed", type=int, default=4300)
    parser.add_argument("--steps", type=int, default=3)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--compare", action="store_true")
    mode.add_argument("--lines", action="store_true")
    args = parser.parse_args(argv)
    manifest = json_read(ARCHIVE/"experiment_manifest.json")
    names = [task["task"] for task in manifest["tasks"]]
    graphs = [CompactIDPCGraph.from_file(ROOT/task["path"]) for task in manifest["tasks"]]
    warmup()
    decoders = [PackedGPA(g) for g in graphs]
    for method in args.methods:
        config = generation_config(manifest["references"][method]["effective_config"])
        state = read_state(ARCHIVE/"checkpoints"/f"{method}_n3_{args.seed}.sqlite",
                           (manifest["fingerprint"], method, args.seed))
        def replay(backend, **options):
            return measure(method, args.seed, state, graphs, names, config, decoders, args.steps,
                           backend, **options)
        if args.compare:
            reference = replay("reference")
            from ekt_gl.idpc.research.n3_fast import warmup_filter
            warmup_filter()
            optimized = replay("optimized")
            for data in (reference, optimized):
                state_value = data.pop("state")
                data["populations"] = state_value["populations"]
                data["rngs"] = (state_value["rngs"], state_value["mating_rng"])
            for k in ("best", "requests", "populations", "rngs"):
                if reference.pop(k) != optimized.pop(k):
                    raise ValueError(f"Replay differs at {method}:{k}")
            for key in reference["counters"]:
                if key not in ("ls_time_s", "ls_subpath_actual_ranking_reads"):
                    if reference["counters"][key] != optimized["counters"][key]:
                        raise ValueError(f"Replay counter differs at {method}:{key}")
            result = dict(parity=True, reference=reference, optimized=optimized)
            dest = OUT/"profile"/f"{method}_replay_comparison.json"
        elif args.lines:
            probe = LineProbe()
            previous = sys.gettrace()
            try:
                sys.settrace(probe.trace)
                profiled = replay("reference")
            finally:
                sys.settrace(previous)
            result = dict(method=method, seed=args.seed, steps=args.steps,
                          snapshot_generation=profiled["snapshot_generation"],
                          timing="inclusive, instrumented execution", lines=probe.output())
            dest = OUT/"profile"/f"{method}_reference_lines.json"
        else:
            profiler = cProfile.Profile()
            profiled = replay("reference", profiler=profiler, keys=True)
            profiled.pop("state")
            text = io.StringIO()
            stats = pstats.Stats(profiler, stream=text).strip_dirs().sort_stats("cumulative")
            stats.print_stats(55)
            selected = {"replace_segment", "decode_beam", "pair_costs", "shortest",
                        "validate_path", "roundtrip"}
            profiled["hotspots_s"] = {name: metrics[3] for (_, _, name), metrics in stats.stats.items()
                                      if name in selected}
            (OUT/"profile").mkdir(parents=True, exist_ok=True)
            (OUT/"profile"/f"{method}_reference_cprofile.txt").write_text(text.getvalue(), encoding="utf-8")
            result = profiled
            dest = OUT/"profile"/f"{method}_reference_profile.json"
        structured(dest,result)
        print(dest, flush=True)


if __name__ == "__main__":
    main()
