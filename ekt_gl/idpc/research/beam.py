"""Budgeted cost-aware search over exact directed multigraph edge IDs."""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class BeamConfig:
    width: int = 8
    beta: float = 0.1
    max_expansions: int = 4096
    max_complete: int = 4

    def __post_init__(self):
        if min(self.width, self.max_expansions, self.max_complete) < 1:
            raise ValueError("Invalid beam cap")
        if not math.isfinite(self.beta) or self.beta < 0:
            raise ValueError("Invalid neural contribution")


@dataclass(frozen=True)
class State:
    vertex: int
    edge_ids: tuple[int, ...]
    visited: int
    active: int
    closed: int
    cost: float
    nll: float


@dataclass(frozen=True)
class SearchResult:
    paths: tuple[tuple[int, ...], ...]
    expansions: int
    ranking_reads: int
    cap_hit: bool


def pair_costs(graph):
    rows = graph.edges.table.rows
    matrix = np.full((graph.n, graph.n), np.inf)
    if np.any(rows["weight"] < 0):
        raise ValueError("The lower bound requires nonnegative edge weights")
    np.minimum.at(matrix, (rows["u"], rows["v"]), rows["weight"])
    return matrix


def shortest(matrix, start):
    """Dense Dijkstra for the small vertex set, including zero-weight edges."""
    distance = np.full(len(matrix), np.inf)
    done = np.zeros(len(matrix), dtype=bool)
    distance[start] = 0
    for _ in distance:
        candidate = np.where(done, np.inf, distance)
        v = candidate.argmin()
        if not np.isfinite(candidate[v]):
            break
        done[v] = True
        distance = np.minimum(distance, distance[v] + matrix[v])
    return distance


def lower_bound(graph, target=None):
    return shortest(pair_costs(graph).T, graph.target if target is None else target)


def decode_beam(graph, lower, config, scores=None, *, start=None, target=None,
                visited=0, active=-1, closed=0):
    """Apply the configured work caps, then sort completed paths by exact cost.

    Every expansion uses one valid directed edge. Ranking scans are accounted
    separately, including all parallel edges read to construct row rankings.
    No state is merged across different visit or domain histories.
    """
    start = graph.source if start is None else start
    target = graph.target if target is None else target
    if not np.isfinite(lower[start]):
        return SearchResult((), 0, 0, False)
    scale = max(1.0, float(lower[start]))
    if config.beta and scores is None:
        raise ValueError("A nonzero beta requires a supplied score per edge")
    table = graph.edges.table.rows
    if scores is not None:
        if np.shape(scores) != (len(table),) or not np.all(np.isfinite(scores)):
            raise ValueError("Invalid edge scores")
        penalty = -np.log(np.clip(scores, 1e-8, 1.0)) / -math.log(1e-8)
    else:
        penalty = None

    layers = [State(start, (), visited | (1 << start), active, closed, 0.0, 0.0)]
    rankings = {}
    complete = []
    expanded = reads = 0
    while layers and expanded < config.max_expansions and len(complete) < config.max_complete:
        children = []
        for state in layers:
            v = state.vertex
            if v not in rankings:
                outgoing = graph.adj.get(v)
                positions = np.asarray(outgoing.positions if outgoing else (), dtype=np.int64)
                reads += len(positions)
                subset = table[positions]
                keys = (subset["weight"] + lower[subset["v"]]) / scale
                if penalty is not None:
                    keys += config.beta * penalty[positions]
                rankings[v] = positions[np.lexsort((subset["id"], keys))]
            chosen = 0
            for index in rankings[v]:
                row = table[index]
                dst, domain = int(row["v"]), int(row["domain"])
                if state.visited & (1 << dst) or state.closed & (1 << domain) or not np.isfinite(lower[dst]):
                    continue
                expanded += 1
                chosen += 1
                left = state.closed
                if state.active >= 0 and state.active != domain:
                    left |= 1 << state.active
                child = State(dst, state.edge_ids + (int(row["id"]),),
                              state.visited | (1 << dst), domain, left,
                              state.cost + float(row["weight"]),
                              state.nll + (float(penalty[index]) if penalty is not None else 0.0))
                if dst == target:
                    complete.append(child)
                else:
                    children.append(child)
                if (chosen == config.width or expanded == config.max_expansions
                        or len(complete) == config.max_complete):
                    break
            if expanded == config.max_expansions or len(complete) == config.max_complete:
                break
        children.sort(key=lambda s: ((s.cost + lower[s.vertex]) / scale + config.beta*s.nll,
                                    s.cost, s.edge_ids))
        layers = children[:config.width]
    complete.sort(key=lambda s: (s.cost, s.edge_ids))
    return SearchResult(tuple(s.edge_ids for s in complete), expanded, reads,
                        expanded == config.max_expansions or len(complete) == config.max_complete)
