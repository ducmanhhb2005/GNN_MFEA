"""Permutation operators, heatmap decoding, and local search."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np

from .problems import CVRPTask, Route, RoutingTask, TSPTask

if TYPE_CHECKING:
    from numpy.random import Generator


def order_crossover(
    first: Route, second: Route, rng: Generator, *, cyclic_scan: bool = True,
) -> Route:
    """OX with cyclic scanning from the second cut, preserving node order.

    ``cyclic_scan=False`` retains the prior CVRP implementation for backwards
    compatibility. Standard OX must scan and fill from the same cut position.
    """
    if len(first) != len(second) or set(first) != set(second):
        raise ValueError("OX requires two permutations of the same node set")
    size = len(first)
    left, right = sorted(rng.choice(size, size=2, replace=False).tolist())
    child: list[int | None] = [None] * size
    child[left : right + 1] = first[left : right + 1]
    scan = second[right + 1:] + second[:right + 1] if cyclic_scan else second
    remaining = [node for node in scan if node not in child]
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
    if isinstance(task, TSPTask):
        return _tsp_two_opt(task, route, max_passes)
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


@lru_cache(maxsize=64)
def _reversal_indices(count: int) -> tuple[np.ndarray, np.ndarray]:
    # Same lexicographic left/right order as the original nested loops.
    return np.triu_indices(count + 1, k=2)


def _tsp_two_opt(task: TSPTask, route: Route, max_passes: int) -> Route:
    """Screen reversals using four edges; preserve the original search order.

    Evaluate every potentially improving candidate with the original float32
    objective. The conservative summation error bound also retains candidates
    that improve only because of float32 rounding. CVRP cannot use this delta
    because a reversal may change capacity splits.
    """
    best = route
    best_cost = task.evaluate(best)
    count = len(best)
    distance = task._distance
    # eps, rather than eps/2, deliberately overbounds float32 summation error.
    neps = count * np.finfo(distance.dtype).eps
    if neps >= 1:
        raise ValueError("TSP is too large for the float32 summation bound")
    gamma = neps / (1 - neps)
    maximum_edge = float(distance.max())
    lefts, rights = _reversal_indices(count)
    for _ in range(max_passes):
        tour = np.asarray(best)
        exact_sum = distance[tour, np.roll(tour, -1)].sum(dtype=np.float64)
        error_bound = gamma * (exact_sum + 2 * maximum_edge) + 1e-10
        improved = False
        before, first = tour[lefts - 1], tour[lefts]
        last, after = tour[rights - 1], tour[rights % count]
        # Cast each edge before arithmetic to avoid float32 delta rounding.
        delta = (distance[before, last].astype(np.float64)
                 + distance[first, after].astype(np.float64)
                 - distance[before, first].astype(np.float64)
                 - distance[last, after].astype(np.float64))
        delta[count - 2] = 0.0  # full-cycle reversal (left=0, right=count)
        promising = np.flatnonzero(exact_sum + delta - error_bound < best_cost)
        for index in promising:
            left, right = int(lefts[index]), int(rights[index])
            candidate = best[:left] + tuple(reversed(best[left:right])) + best[right:]
            candidate_cost = task.evaluate(candidate)
            if candidate_cost + 1e-12 < best_cost:
                best, best_cost, improved = candidate, candidate_cost, True
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
