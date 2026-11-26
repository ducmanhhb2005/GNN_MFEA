"""Decode heatmap scores and encode paths for the existing GPA."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from itertools import islice
import math

from idpc_du.gpa import eligible_edges
from idpc_du.graph import Edge
from idpc_du.validation import validate_path
from mfea.chromosome import Chromosome

from .task import IDPCGraph, _EdgeSequence


def greedy_path(
    graph: IDPCGraph,
    score_edges: Callable[[Sequence[Edge]], Sequence[float]],
    batch_size: int,
) -> tuple[int, ...] | None:
    """Score only outgoing, currently feasible edges and stop at the target."""
    current = graph.source
    visited = {current}
    closed: set[str] = set()
    active = None
    path: list[int] = []
    while current != graph.target:
        outgoing = graph.adj.get(current, ())
        best = None
        best_key = (-math.inf, -math.inf, -math.inf)
        if isinstance(outgoing, _EdgeSequence):
            eligible = eligible_edges(outgoing.unvisited(visited), closed)
            for offset in range(0, len(eligible), batch_size):
                batch = eligible[offset:offset + batch_size]
                score, edge = batch.best_scored_edge(score_edges(batch))
                key = (score, -edge.weight, -edge.id)
                if key > best_key:
                    best, best_key = edge, key
        else:
            allowed = (
                edge for edge in outgoing if edge.v not in visited and edge.domain not in closed
            )
            while batch := list(islice(allowed, batch_size)):
                for edge, score in zip(batch, score_edges(batch), strict=True):
                    if not math.isfinite(score):
                        raise ValueError("Edge predictor returned a non-finite score")
                    key = (score, -edge.weight, -edge.id)
                    if key > best_key:
                        best, best_key = edge, key
        if best is None:
            return None
        if active is not None and best.domain != active:
            closed.add(active)
        active = best.domain
        current = best.v
        visited.add(current)
        path.append(best.id)
    return tuple(path)


def encode_path(graph: IDPCGraph, path: tuple[int, ...], limits: Sequence[int]) -> Chromosome | None:
    """Construct genes that guide GPA along this exact sequence of edge IDs."""
    if not math.isfinite(validate_path(graph, path)):
        return None
    destinations = [graph.by_id[edge_id].v for edge_id in path]
    path_vertices = set(destinations)
    order = destinations + [v for v in range(graph.n) if v not in path_vertices]
    priority = [0] * graph.n
    for rank, vertex in enumerate(order):
        priority[vertex] = graph.n - rank

    sigma = [1] * graph.n
    closed: set[str] = set()
    active = None
    for edge_id in path:
        chosen = graph.by_id[edge_id]
        allowed = eligible_edges(graph.by_neighbor[chosen.u][chosen.v], closed)
        packed_index = getattr(allowed, "index_of_id", None)
        index = (packed_index(edge_id) if packed_index is not None
                 else next(i for i, edge in enumerate(allowed) if edge.id == edge_id))
        # GPA reads sigma at the arriving vertex. A positive multiple is
        # required to select the first eligible edge (modulo residue zero).
        gene = index or len(allowed)
        if gene > limits[chosen.v]:
            return None
        sigma[chosen.v] = gene
        if active is not None and chosen.domain != active:
            closed.add(active)
        active = chosen.domain
    return Chromosome(priority, sigma)
