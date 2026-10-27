"""Growing Path Algorithm: CEC 2020, Algorithm 1 and Section IV-F."""
from __future__ import annotations
import math
from typing import Protocol
from .graph import Graph
from .validation import validate_path

class PathChromosome(Protocol):
    """Input contract: priority and edge-index genes, independent of MFEA."""
    priority: list[int]
    edge_index: list[int]

def decode_gpa(g: Graph, c: PathChromosome) -> tuple[float, list[int]]:
    """Follow Algorithm 1, including its arriving-node sigma and literal modulo.

    Line 13 specifies k = sigma[v] % |E_hv| + 1 (one-based edge index).
    In Python this is candidates[sigma[v] % len(candidates)]. The prose's
    departing-node description conflicts with this line; use the pseudocode
    consistently. Parallel edges retain their original input order.
    Return (+inf, []) when GPA cannot reach the target under DU.
    """
    pi = [x for x in c.priority if x <= g.n]
    if len(pi) != g.n or set(pi) != set(range(1, g.n + 1)):
        return math.inf, []
    if len(c.edge_index) < g.n:
        return math.inf, []
    # Paper Section IV-F: stable filtering of permutation for a task of n nodes.
    # pi[v] is priority of the vertex v; descending priority visits first.
    sorted_neighbors = None  # lazily used only for dense neighborhoods
    curr = g.source
    seen = {curr}
    left_domains = set()
    active_domain = None
    path = []
    while curr != g.target:
        candidates = None
        chosen_v = None
        neighbors = g.by_neighbor.get(curr, {})
        if len(neighbors) < max(4, g.n // 4):
            # Sparse graphs: inspect actual neighbors, not all n vertices at each
            # step. Unique priorities make this exactly the same GPA choice.
            highest = -1
            for v, pair_edges in neighbors.items():
                if v in seen or pi[v] <= highest:
                    continue
                allowed = ([e for e in pair_edges if e.domain not in left_domains]
                           if left_domains else pair_edges)
                if allowed:
                    chosen_v, candidates, highest = v, allowed, pi[v]
        else:
            if sorted_neighbors is None:
                sorted_neighbors = sorted((v for v in range(g.n) if v not in seen),
                                          key=lambda v: (-pi[v], v))
            for v in sorted_neighbors:
                pair_edges = neighbors.get(v)
                if not pair_edges:
                    continue
                allowed = ([e for e in pair_edges if e.domain not in left_domains]
                           if left_domains else pair_edges)
                if allowed:
                    chosen_v, candidates = v, allowed
                    break
        if candidates is None:
            return math.inf, []
        # Algorithm 1, lines 13-14; chosen_v is the arriving vertex.
        e = candidates[c.edge_index[chosen_v] % len(candidates)]
        if active_domain is not None and active_domain != e.domain:
            left_domains.add(active_domain)
        if e.domain in left_domains:
            return math.inf, []
        path.append(e.id)
        # Keep the priority order but remove visited vertices once. This avoids
        # rescanning the same visited prefix on every step of a dense path;
        # Algorithm 1's choice among unvisited vertices is unchanged.
        if sorted_neighbors is not None:
            sorted_neighbors.remove(chosen_v)
        seen.add(chosen_v)
        active_domain, curr = e.domain, chosen_v
    return validate_path(g, path), path
