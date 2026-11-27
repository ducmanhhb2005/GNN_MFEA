"""Exact cost-only N3 beam with reusable outgoing ranks and compiled filtering."""
from collections import Counter, OrderedDict
import math

import numpy as np
from numba import njit

from ekt_gl.idpc.task import CompactIDPCGraph
from idpc_du.validation import validate_path as reference_validate

from . import neighborhoods
from .beam import BeamConfig, SearchResult, State

RANK_CACHE_BYTES = 128 * 1024 * 1024
RANK_PREFIX = 128


@njit(cache=False)
def eligible_edges(rows, ranked, finite, visited, closed, target, count, remaining_complete):
    """Filter the ranked prefix in the same order and with the same stop checks."""
    selected = np.empty(count, dtype=np.int64)
    size = 0
    for row_index in ranked:
        row = rows[row_index]
        dst = row["v"]
        domain = row["domain"]
        if ((visited[dst >> 3] >> (dst & 7)) & 1
                or (closed[domain >> 3] >> (domain & 7)) & 1
                or not finite[dst]):
            continue
        selected[size] = row_index
        size += 1
        if dst == target:
            remaining_complete -= 1
        if size == count or remaining_complete == 0:
            break
    return selected[:size], remaining_complete


@njit(cache=False)
def path_cost(rows, positions, source, target, vertices, domains):
    visited = np.zeros(vertices, dtype=np.bool_)
    closed = np.zeros(domains, dtype=np.bool_)
    current = source
    visited[current] = True
    active = -1
    cost = 0.0
    for position in positions:
        if position < 0 or position >= len(rows):
            return np.inf
        row = rows[position]
        dst, domain = row["v"], row["domain"]
        if row["u"] != current or visited[dst] or closed[domain]:
            return np.inf
        if active >= 0 and active != domain:
            closed[active] = True
        active = domain
        current = dst
        visited[current] = True
        cost += row["weight"]
    return cost if current == target else np.inf


def validate_path(graph, edge_ids):
    """Validate directed edge IDs from compact rows in the original addition order."""
    if not isinstance(graph, CompactIDPCGraph):
        return reference_validate(graph, edge_ids)
    indexes = graph.by_id.positions
    if indexes is not None:
        positions = np.fromiter((indexes.get(edge, -1) for edge in edge_ids), np.int64, len(edge_ids))
    else:
        try:
            positions = np.asarray(edge_ids, dtype=np.int64)
        except OverflowError:
            return math.inf
    return float(path_cost(graph.edges.table.rows, positions, graph.source, graph.target,
                           graph.n, len(graph.domains)))


def warmup_filter():
    from ekt_gl.idpc.task import _EDGE_DTYPE

    rows = np.empty(0, dtype=_EDGE_DTYPE)
    rows.flags.writeable = False
    for dtype in (np.int32, np.int64):
        ranked = np.empty(0, dtype=dtype)
        ranked.flags.writeable = False
        finite = np.array([True])
        mask = np.frombuffer(b"\0", dtype=np.uint8)
        eligible_edges(rows, ranked, finite, mask, mask, 0, 1, 1)
    path_cost(rows, np.empty(0, dtype=np.int64), 0, 0, 1, 1)


class CostOnlyBeam:
    """One solver-suite cache of exact vertex rankings in a fixed graph."""

    def __init__(self, cache_bytes=RANK_CACHE_BYTES):
        self.cache_bytes = cache_bytes
        self.cache = OrderedDict()
        self.size = 0
        self.ranking_reads = 0
        self.graphs = {}
        self.targets = {}

    def target(self, graph, lower):
        key = id(graph), id(lower)
        if key not in self.targets:
            graph_key = id(graph)
            if graph_key not in self.graphs:
                weights = graph.edges.table.rows["weight"]
                maximum = float(weights.max()) if len(weights) else 0.0
                integral = (maximum <= 2**50 and np.all(weights >= 0)
                            and np.all(weights == np.floor(weights)))
                self.graphs[graph_key] = (graph, maximum if integral else None)
            _, maximum = self.graphs[graph_key]
            finite = np.isfinite(lower)
            values = lower[finite]
            # In [0, 2**50], distinct integer keys differ by at least 1.
            # A common divisor >=1 cannot collapse that gap under float64
            # rounding: each quotient ulp is bounded by 1/(4*scale).
            shared = (maximum is not None and np.all(values >= 0)
                      and np.all(values == np.floor(values))
                      and (len(values) == 0 or maximum + float(values.max()) <= 2**50))
            self.targets[key] = (lower, finite, shared)
        return self.targets[key]

    def ranks(self, graph, lower, vertex, scale, shared, limit=None):
        key = id(graph), id(lower), vertex, 1.0 if shared else scale, limit
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        outgoing = graph.adj.get(vertex)
        positions = np.asarray(outgoing.positions if outgoing else (), dtype=np.int64 if outgoing is None else None)
        table = graph.edges.table.rows
        keys = table["weight"][positions] + lower[table["v"][positions]]
        if not shared:
            keys = keys / scale
        self.ranking_reads += len(positions)
        if limit is not None and len(positions) > limit:
            boundary = np.partition(keys, limit-1)[limit-1]
            # Preserve the original ordering when the boundary contains ties.
            if not np.isnan(boundary):
                top = keys <= boundary
                positions, keys = positions[top], keys[top]
        order = np.lexsort((table["id"][positions], keys))[:limit]
        ranked = positions[order]
        ranked.flags.writeable = False
        if ranked.nbytes <= self.cache_bytes:
            while self.size + ranked.nbytes > self.cache_bytes:
                _, previous = self.cache.popitem(last=False)
                self.size -= previous.nbytes
            self.cache[key] = ranked
            self.size += ranked.nbytes
        return ranked

    def __call__(self, graph, lower, config, scores=None, *, start=None, target=None,
                 visited=0, active=-1, closed=0):
        if config.beta or scores is not None:
            raise ValueError("N3 expects immutable cost-only ranks")
        start = graph.source if start is None else start
        target = graph.target if target is None else target
        if not np.isfinite(lower[start]):
            return SearchResult((), 0, 0, False)
        scale = max(1.0, float(lower[start]))
        _, finite, shared = self.target(graph, lower)
        table = graph.edges.table.rows
        vertex_bytes = max(1, (graph.n + 7) // 8)
        domain_bytes = max(1, (len(graph.domains) + 7) // 8)
        layers = [State(start, (), visited | (1 << start), active, closed, 0.0, 0.0)]
        rankings = {}
        tails = {}
        complete = []
        expanded = reads = 0
        while layers and expanded < config.max_expansions and len(complete) < config.max_complete:
            children = []
            for state in layers:
                vertex = state.vertex
                if vertex not in rankings:
                    degree = len(graph.adj[vertex]) if vertex in graph.adj else 0
                    prefix = self.ranks(graph, lower, vertex, scale, shared, RANK_PREFIX)
                    rankings[vertex] = prefix, degree
                    reads += degree
                state_visited = np.frombuffer(state.visited.to_bytes(vertex_bytes, "little"), dtype=np.uint8)
                state_closed = np.frombuffer(state.closed.to_bytes(domain_bytes, "little"), dtype=np.uint8)
                ranking, degree = rankings[vertex]
                count = min(config.width, config.max_expansions - expanded)
                remaining = config.max_complete - len(complete)
                positions, complete_left = eligible_edges(table, ranking, finite, state_visited,
                                                           state_closed, target, count, remaining)
                if len(positions) < count and complete_left and len(ranking) < degree:
                    if vertex not in tails:
                        full = self.ranks(graph, lower, vertex, scale, shared)
                        tails[vertex] = full[len(ranking):]
                    suffix, _ = eligible_edges(table, tails[vertex], finite, state_visited,
                                                state_closed, target, count-len(positions), complete_left)
                    positions = np.concatenate((positions, suffix))
                expanded += len(positions)
                for index in positions:
                    row = table[index]
                    dst, domain = int(row["v"]), int(row["domain"])
                    left = state.closed
                    if state.active >= 0 and state.active != domain:
                        left |= 1 << state.active
                    child = State(dst, state.edge_ids + (int(row["id"]),),
                                  state.visited | (1 << dst), domain, left,
                                  state.cost + float(row["weight"]), 0.0)
                    if dst == target:
                        complete.append(child)
                    else:
                        children.append(child)
                if expanded == config.max_expansions or len(complete) == config.max_complete:
                    break
            children.sort(key=lambda state: ((state.cost + lower[state.vertex]) / scale,
                                             state.cost, state.edge_ids))
            layers = children[:config.width]
        complete.sort(key=lambda state: (state.cost, state.edge_ids))
        return SearchResult(tuple(state.edge_ids for state in complete), expanded, reads,
                            expanded == config.max_expansions or len(complete) == config.max_complete)


def replace_segment(tracker, task, candidate, limits, pairs, config, rng, phase, unified=False):
    """N3 with cut context precomputed once on the original candidate."""
    graph = tracker.graphs[task]
    counts = Counter(ls_calls=1)
    started = neighborhoods.perf_counter()
    before = tracker.per_task[task]["ls_evaluations"]
    path = candidate.solution.path
    edges = [graph.by_id[edge_id] for edge_id in path]
    nodes = [graph.source] + [edge.v for edge in edges]
    cuts = [(i,j) for i in range(len(edges)) for j in range(i+1,len(nodes))]
    if len(cuts) > config.max_cut_pairs_per_candidate:
        counts["ls_cut_sampling_truncated"] += 1
        cuts = sorted(rng.sample(cuts, config.max_cut_pairs_per_candidate))
    if not hasattr(pairs, "lower_by_target"):
        pairs.lower_by_target = {}
        pairs.vertex_costs = neighborhoods.pair_costs(graph)
    domains = [(-1, 0)]
    active, closed = domains[0]
    for edge in edges:
        domain = pairs.codes[edge.domain]
        if active >= 0 and active != domain:
            closed |= 1 << active
        active = domain
        domains.append((active, closed))
    vertices = [0]
    for node in nodes:
        vertices.append(vertices[-1] + (1 << node))
    beam = BeamConfig(config.beam_width, 0, config.max_edge_expansions_per_cut,
                      config.max_complete_paths_per_cut)
    best = candidate
    for i,j in cuts:
        if tracker.exhausted:
            break
        counts["ls_cut_attempts"] += 1
        counts["local_search_checks"] += 1
        target = nodes[j]
        if target not in pairs.lower_by_target:
            pairs.lower_by_target[target] = neighborhoods.shortest(pairs.vertex_costs.T, target)
        active, closed = domains[i]
        visited = vertices[i] + vertices[-1] - vertices[j+1]
        result = neighborhoods.decode_beam(graph, pairs.lower_by_target[target], beam,
                                           start=nodes[i], target=target, visited=visited,
                                           active=active, closed=closed)
        counts["ls_edge_expansions"] += result.expansions
        counts["ls_ranking_edge_reads"] += result.ranking_reads
        counts["ls_truncated"] += result.cap_hit
        counts["ls_complete_paths"] += len(result.paths)
        prefix, suffix = path[:i], path[j:]
        for subpath in result.paths:
            if tracker.exhausted:
                break
            proposed = tracker.path(task, prefix + subpath + suffix, "ls")
            if not proposed.feasible:
                counts["ls_du_rejects"] += 1
                continue
            counts["ls_feasible_neighbors"] += 1
            if proposed.cost >= best.solution.cost:
                continue
            encoded = tracker.roundtrip(task, proposed, limits, "ls",
                                        unified=candidate.chromosome if unified else None, phase=phase)
            if encoded is not None:
                best = encoded
    neighborhoods.finish_stats(tracker, task, phase, candidate, best, before, counts, started)
    return best
