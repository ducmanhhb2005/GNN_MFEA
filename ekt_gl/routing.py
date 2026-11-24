"""Permutation operators, heatmap decoding, and local search."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .problems import CVRPTask, Route, RoutingTask, TSPTask

if TYPE_CHECKING:
    from numpy.random import Generator


def order_crossover(first: Route, second: Route, rng: Generator) -> Route:
    """Order crossover (OX), preserving every node exactly once."""
    if len(first) != len(second) or set(first) != set(second):
        raise ValueError("OX requires two permutations of the same node set")
    size = len(first)
    left, right = sorted(rng.choice(size, size=2, replace=False).tolist())
    child: list[int | None] = [None] * size
    child[left : right + 1] = first[left : right + 1]
    remaining = [node for node in second if node not in child]
    positions = list(range(right + 1, size)) + list(range(0, left))
    for position, node in zip(positions, remaining):
        child[position] = node
    return tuple(int(node) for node in child if node is not None)


def swap_mutation(route: Route, rng: Generator) -> Route:
    """Swap two genes; the resulting route remains a valid permutation."""
    left, right = rng.choice(len(route), size=2, replace=False)
    result = list(route)
    result[left], result[right] = result[right], result[left]
    return tuple(result)


def two_opt(task: RoutingTask, route: Route, max_passes: int = 1) -> Route:
    """First-improvement 2-opt, evaluated against the task's full objective.

    For CVRP this also permits moving customers over a depot split. Feasibility is
    guaranteed because ``CVRPTask.evaluate`` re-splits the customer permutation.
    """
    best = route
    best_cost = task.evaluate(best)
    for _ in range(max_passes):
        improved = False
        for left in range(len(best) - 1):
            for right in range(left + 2, len(best) + 1):
                candidate = best[:left] + tuple(reversed(best[left:right])) + best[right:]
                candidate_cost = task.evaluate(candidate)
                if candidate_cost + 1e-12 < best_cost:
                    best, best_cost, improved = candidate, candidate_cost, True
                    break
            if improved:
                break
        if not improved:
            break
    return best


def _best_unvisited(scores: np.ndarray, candidates: set[int], distance: np.ndarray) -> int:
    """Select max probability, breaking ties by shorter edge then node id."""
    return min(candidates, key=lambda node: (-float(scores[node]), float(distance[node]), node))


def greedy_decode(task: RoutingTask, heatmap: np.ndarray, rng: Generator) -> Route:
    """Convert edge probabilities into a valid TSP/CVRP permutation.

    TSP starts from a random node, as in the paper. CVRP starts from the depot;
    when no unvisited customer fits its remaining capacity it closes that vehicle
    loop and opens a new one at depot.
    """
    heatmap = np.asarray(heatmap, dtype=np.float64)
    if heatmap.shape != (task.node_count, task.node_count):
        raise ValueError("heatmap shape must match task adjacency matrix")

    if isinstance(task, TSPTask):
        start = int(rng.integers(task.node_count))
        route = [start]
        unvisited = set(task.route_nodes)
        unvisited.remove(start)
        current = start
        while unvisited:
            next_node = _best_unvisited(heatmap[current], unvisited, task._distance[current])
            route.append(next_node)
            unvisited.remove(next_node)
            current = next_node
        return tuple(route)

    if not isinstance(task, CVRPTask):  # defensive for future RoutingTask implementations
        raise TypeError(f"unsupported routing task {type(task).__name__}")
    unvisited = set(task.route_nodes)
    route: list[int] = []
    current, remaining = 0, float(task.capacity)
    while unvisited:
        feasible = {node for node in unvisited if task.demands[node] <= remaining + 1e-7}
        if not feasible:
            current, remaining = 0, float(task.capacity)
            continue
        next_node = _best_unvisited(heatmap[current], feasible, task._distance[current])
        route.append(next_node)
        unvisited.remove(next_node)
        remaining -= float(task.demands[next_node])
        current = next_node
    return tuple(route)
