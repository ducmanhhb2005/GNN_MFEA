"""Memoize exact cost-only subpath searches; charge every logical objective request."""

from collections import OrderedDict
import hashlib
import json
from pathlib import Path

from ekt_gl.idpc.research import neighborhoods
from experiments import run_idpc_v3 as runner
from experiments._support.idpc_v3_archive import ROOT, sha


SUBPATH_CACHE_SIZE = 32768


class SubpathMemo:
    def __init__(self, search, size=SUBPATH_CACHE_SIZE):
        self.search = search
        self.size = size
        self.cache = OrderedDict()
        self.inputs = {}
        self.hits = self.misses = self.expansions = self.reads = 0

    def __call__(self, graph, lower, config, scores=None, **restrictions):
        if config.beta or scores is not None:
            raise ValueError("Memoization is defined for immutable cost-only rankings")
        base = id(graph), id(lower)
        key = (*base, config, tuple(sorted(restrictions.items())))
        if key in self.cache:
            self.hits += 1
            self.cache.move_to_end(key)
            return self.cache[key]
        result = self.search(graph, lower, config, **restrictions)
        self.misses += 1
        self.expansions += result.expansions
        self.reads += result.ranking_reads
        self.inputs.setdefault(base, (graph, lower))
        self.cache[key] = result
        if len(self.cache) > self.size:
            self.cache.popitem(last=False)
        return result

    def work(self):
        return dict(ls_subpath_memo_hits=self.hits, ls_subpath_memo_misses=self.misses,
                    ls_subpath_actual_ranking_reads=self.reads, ls_subpath_actual_expansions=self.expansions)


def main():
    original_run = runner.run_research
    original_manifest = runner.campaign_identity

    def run_with_memo(checkpoint, method, seed, graphs, names, config, fingerprint,
                      features, decoders, variant, *args, **kwargs):
        inputs = (checkpoint, method, seed, graphs, names, config, fingerprint,
                  features, decoders, variant, *args)
        if variant.ls != "subpath":
            return original_run(*inputs, **kwargs)
        original_search = neighborhoods.decode_beam
        original_evolution = runner.ResearchEvolution
        memo = SubpathMemo(original_search)

        class ObservedEvolution(original_evolution):
            def improve(self, task, candidate, pairs, rng, phase):
                before = memo.work()
                result = super().improve(task, candidate, pairs, rng, phase)
                counts = self.tracker.per_task[task]
                for key, after in memo.work().items():
                    counts[key] += after - before[key]
                return result

        runner.ResearchEvolution, neighborhoods.decode_beam = ObservedEvolution, memo
        try:
            return original_run(*inputs, **kwargs)
        finally:
            runner.ResearchEvolution, neighborhoods.decode_beam = original_evolution, original_search

    def manifest_with_memo(*args, **kwargs):
        data = original_manifest(*args, **kwargs)
        data["sources"][Path(__file__).relative_to(ROOT).as_posix()] = sha(__file__)
        data["runtime_optimization"] = {
            "subpath_cache_entries": SUBPATH_CACHE_SIZE,
            "scope": "exact cost-only subpath search with immutable graph and fixed directed DU history",
            "budget": "no change to logical objective calls, ordering, round-trip, or search caps",
        }
        data["performance_validation"] = data["phase"] == "pilot"
        data.pop("fingerprint")
        data["fingerprint"] = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
        return data

    runner.run_research, runner.campaign_identity = run_with_memo, manifest_with_memo
    return runner.main()


if __name__ == "__main__":
    raise SystemExit(main())
