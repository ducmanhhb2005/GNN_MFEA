"""Independent evolutionary solvers used by EKT-GL for each routing task."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .problems import Route, RoutingTask, TSPTask
from .routing import order_crossover, swap_mutation, two_opt

if TYPE_CHECKING:
    from numpy.random import Generator


@dataclass(frozen=True, slots=True)
class Individual:
    route: Route
    cost: float


class Population:
    """Elitist population belonging to exactly one routing task."""

    def __init__(self, task: RoutingTask, size: int, rng: Generator) -> None:
        self.task = task
        self.size = size
        self._rng = rng
        self.members = sorted(
            (self._make(tuple(rng.permutation(task.route_nodes).tolist())) for _ in range(size)),
            key=lambda individual: individual.cost,
        )

    def _make(self, route: Route) -> Individual:
        return Individual(route=route, cost=self.task.evaluate(route))

    @property
    def best(self) -> Individual:
        return self.members[0]

    def elites(self, count: int) -> list[Individual]:
        return self.members[: min(count, self.size)]

    def _tournament(self) -> Individual:
        contenders = self._rng.choice(self.members, size=min(3, self.size), replace=False)
        return min(contenders, key=lambda individual: individual.cost)

    def evolve(self, mutation_rate: float, local_search_passes: int) -> None:
        offspring: list[Individual] = []
        while len(offspring) < self.size:
            left, right = self._tournament(), self._tournament()
            route = order_crossover(
                left.route, right.route, self._rng,
                cyclic_scan=isinstance(self.task, TSPTask),
            )
            if self._rng.random() < mutation_rate:
                route = swap_mutation(route, self._rng)
            route = two_opt(self.task, route, local_search_passes)
            offspring.append(self._make(route))
        self.members = sorted(self.members + offspring, key=lambda individual: individual.cost)[: self.size]

    def inject(self, route: Route) -> Individual:
        """Transfer a valid candidate by replacing the current worst member."""
        candidate = self._make(route)
        if all(member.route != candidate.route for member in self.members):
            self.members[-1] = candidate
            self.members.sort(key=lambda individual: individual.cost)
        return candidate
