"""IDPC-DU path validation (independent of optimizers)."""
from __future__ import annotations
import math
from .graph import Graph

def validate_path(g: Graph, edge_ids: list[int] | tuple[int, ...]) -> float:
    """Return finite cost only for a connected simple source-target path respecting DU.

    A domain may occupy several consecutive edges. Once exited, it cannot reappear.
    """
    current = g.source
    visited_nodes = {current}
    closed_domains: set[str] = set()
    active_domain = None
    cost = 0.0
    for eid in edge_ids:
        e = g.by_id.get(eid)
        if e is None or e.u != current or e.v in visited_nodes:
            return math.inf
        if e.domain in closed_domains:
            return math.inf
        if active_domain is not None and e.domain != active_domain:
            closed_domains.add(active_domain)
            if e.domain in closed_domains:
                return math.inf
        active_domain = e.domain
        current = e.v
        visited_nodes.add(current)
        cost += e.weight
    return cost if current == g.target else math.inf
