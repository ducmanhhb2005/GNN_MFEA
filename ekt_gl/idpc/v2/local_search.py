"""Bounded directed 2-opt with exact edge assignments and GPA round-trips."""

from collections import Counter
from itertools import combinations
from time import perf_counter

import numpy as np


class DirectedPairs:
    """Build at most one weight/edge-ID index per directed pair on demand."""

    def __init__(self, graph, cache_size=512):
        self.graph = graph
        self.codes = {domain: i for i, domain in enumerate(graph.domains)}
        self.cache = {}
        self.cache_size = cache_size
        self.beam_cache = {}

    def reconstruct(self, key, indexes, prefix, suffix, config, counts):
        cache_key = (key, config)
        cached = self.beam_cache.get(cache_key)
        if cached is None:
            work = Counter()
            neighbors = _rebuild(indexes, prefix, suffix, self.codes, config, work)
            if self.cache_size:
                if len(self.beam_cache) >= self.cache_size:
                    del self.beam_cache[next(iter(self.beam_cache))]
                self.beam_cache[cache_key] = (neighbors, work)
            counts["ls_actual_edge_expansions"] += work["ls_edge_expansions"]
        else:
            neighbors, work = cached
            counts["ls_beam_cache_hits"] += 1
        counts.update(work)
        return neighbors

    def get(self, u, v):
        key = (u, v)
        if key in self.cache:
            return self.cache[key]
        edges = self.graph.by_neighbor.get(u, {}).get(v)
        if not edges:
            return None
        if hasattr(edges, "table"):
            rows = edges.table.rows[edges.positions]
            rows = rows[np.lexsort((rows["id"], rows["weight"]))]
            index = rows["id"].copy(), rows["weight"].copy(), rows["domain"].copy()
        else:
            ordered = sorted(edges, key=lambda e: (e.weight, e.id))
            index = tuple(np.array(values) for values in zip(
                *((e.id, e.weight, self.codes[e.domain]) for e in ordered)))
        self.cache[key] = index
        return index


def _step(domain, active, closed):
    if domain in closed:
        return None
    return domain, (closed | {active} if active is not None and active != domain else closed)


def _rebuild(indexes, prefix, suffix, codes, config, counts):
    active, closed = None, frozenset()
    for edge in prefix:
        active, closed = _step(codes[edge.domain], active, closed)
    beam = [(0.0, (), active, closed)]
    remaining = config.max_edge_expansions_per_cut
    truncated = False
    for ids, weights, domains in indexes:
        children = []
        for partial, path, active, closed in beam:
            visits = min(len(ids), remaining)
            truncated |= visits < len(ids)
            remaining -= visits
            counts["ls_edge_expansions"] += visits
            if not visits:
                break
            blocked = np.zeros(len(codes), dtype=bool)
            if closed:
                blocked[list(closed)] = True
            valid = np.flatnonzero(~blocked[domains[:visits]])
            counts["ls_du_rejects"] += visits - len(valid)
            counts["ls_partial_cost_updates"] += min(len(valid), config.beam_width)
            # A single state's first beam_width eligible edges dominate its
            # later edges for this layer's global cost/edge-ID ordering.
            truncated |= len(valid) > config.beam_width
            for pos in valid[:config.beam_width]:
                domain = int(domains[pos])
                new_active, new_closed = _step(domain, active, closed)
                children.append((partial + float(weights[pos]), path + (int(ids[pos]),),
                                 new_active, new_closed))
        children.sort(key=lambda item: item[:2])
        truncated |= len(children) > config.beam_width
        beam = children[:config.beam_width]
        if not beam:
            break
    prefix_ids = tuple(edge.id for edge in prefix)
    suffix_ids = tuple(edge.id for edge in suffix)
    complete = []
    for _, path, active, closed in beam:
        if len(path) != len(indexes):
            continue
        for edge in suffix:
            state = _step(codes[edge.domain], active, closed)
            if state is None:
                counts["ls_du_rejects"] += 1
                break
            active, closed = state
        else:
            complete.append(prefix_ids + path + suffix_ids)
    if truncated or len(complete) > config.max_complete_paths_per_cut:
        counts["ls_truncated"] += 1
    return complete[:config.max_complete_paths_per_cut]


def cut_pairs(node_count, cap, rng):
    pairs = list(combinations(range(1, node_count - 1), 2))
    return sorted(rng.sample(pairs, cap)) if len(pairs) > cap else pairs


def improve(tracker, task, candidate, limits, pairs, config, rng, phase, unified=False):
    """Return the best strictly better verified candidate within the fixed caps."""
    if not candidate.solution.feasible:
        return candidate
    started = perf_counter()
    counts = Counter(ls_calls=1)
    ls_before = tracker.per_task[task]["ls_evaluations"]
    original = best = candidate
    graph = tracker.graphs[task]
    for _ in range(config.max_passes):
        baseline = best
        edges = [graph.by_id[eid] for eid in baseline.solution.path]
        nodes = [graph.source] + [edge.v for edge in edges]
        cuts = cut_pairs(len(nodes), config.max_cut_pairs_per_candidate, rng)
        counts["ls_cut_sampling_truncated"] += len(cuts) < (len(nodes) - 2) * (len(nodes) - 3) // 2
        for i, j in cuts:
            if tracker.exhausted:
                break
            counts["ls_cut_attempts"] += 1
            counts["local_search_checks"] += 1
            replacement = [nodes[i - 1], *reversed(nodes[i:j + 1]), nodes[j + 1]]
            indexes = [pairs.get(u, v) for u, v in zip(replacement, replacement[1:])]
            if any(index is None for index in indexes):
                counts["ls_missing_edge"] += 1
                continue
            neighbors = pairs.reconstruct((baseline.solution.path, i, j), indexes, edges[:i-1],
                                          edges[j+1:], config, counts)
            counts["ls_complete_paths"] += len(neighbors)
            for path in neighbors:
                evaluated = tracker.path(task, path, "ls")
                if evaluated is None:
                    break
                counts["ls_feasible_neighbors"] += evaluated.feasible
                if evaluated.cost >= best.solution.cost:
                    continue
                encoded = tracker.roundtrip(
                    task, evaluated, limits, "ls",
                    unified=baseline.chromosome if unified else None, phase=phase)
                if encoded is not None:
                    best = encoded
        if best is baseline or tracker.exhausted:
            break
    if best is not original:
        counts["ls_accepted"] += 1
        counts["ls_cost_reduction"] += original.solution.cost - best.solution.cost
    counts["ls_time_s"] = perf_counter() - started
    task_counts = tracker.per_task[task]
    task_counts[phase + "_ls_evaluations"] += task_counts["ls_evaluations"] - ls_before
    task_counts.update(counts)
    task_counts.update({phase + "_" + key: value for key, value in counts.items()})
    return best
