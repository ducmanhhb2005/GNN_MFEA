"""First improvement by replacing a path segment with a directed edge."""

import math
from idpc_du.validation import validate_path

from .task import IDPCGraph


def refine_path(graph: IDPCGraph, path: tuple[int, ...], passes: int) -> tuple[tuple[int, ...], int]:
    """Try cheaper parallel edges and forward shortcuts under the DU constraint."""
    checks = 1
    cost = validate_path(graph, path)
    if not math.isfinite(cost):
        return path, checks
    for _ in range(passes):
        improvement, tried = _first_improvement(graph, path, cost)
        checks += tried
        if improvement is None:
            break
        path, cost = improvement
    return path, checks


def _first_improvement(graph: IDPCGraph, path: tuple[int, ...], cost: float):
    vertices = [graph.source]
    prefix_cost = [0.0]
    for edge_id in path:
        edge = graph.by_id[edge_id]
        vertices.append(edge.v)
        prefix_cost.append(prefix_cost[-1] + edge.weight)
    checks = 0
    for start, source in enumerate(vertices[:-1]):
        neighbors = graph.by_neighbor.get(source, {})
        for end in range(start + 1, len(vertices)):
            old_cost = prefix_cost[end] - prefix_cost[start]
            checked_domains = set()
            edges = neighbors.get(vertices[end], ())
            filter_lighter = getattr(edges, "lighter_than", None)
            if filter_lighter is not None:
                edges = filter_lighter(old_cost)
            for edge in edges:
                if edge.weight >= old_cost or edge.domain in checked_domains:
                    continue
                candidate = path[:start] + (edge.id,) + path[end:]
                checks += 1
                value = validate_path(graph, candidate)
                if value < cost:
                    return (candidate, value), checks
                # With fixed endpoints and domain, feasibility is identical.
                if value == float("inf"):
                    checked_domains.add(edge.domain)
    return None, checks
