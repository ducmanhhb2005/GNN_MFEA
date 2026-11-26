"""Independent evolutionary search for one IDPC-DU task."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache, partial
import math
import random

from idpc_du.gpa import cached_gpa, decode_gpa
from mfea.chromosome import Chromosome, edge_index_limits, init_chrom
from mfea.operators import crossover, mutation

from .decoding import encode_path
from .local_search import refine_path
from .task import IDPCGraph


@dataclass(frozen=True, slots=True)
class IDPCSolution:
    cost: float
    edge_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _Individual:
    chromosome: Chromosome
    solution: IDPCSolution


class IDPCPopulation:
    def __init__(self, graph: IDPCGraph, size: int, rng: random.Random, cache_size: int = 0) -> None:
        self.graph = graph
        self.size = size
        self.rng = rng
        self.limits = edge_index_limits([graph])
        self.decode = cached_gpa(graph, cache_size) if cache_size else partial(decode_gpa, graph)
        refine = partial(refine_path, graph)
        self.refinement = lru_cache(maxsize=cache_size)(refine) if cache_size else refine
        self.evaluations = 0
        self.local_search_checks = 0
        self._members = sorted(
            (self._evaluate(init_chrom(graph.n, self.limits, rng)) for _ in range(size)),
            key=lambda member: member.solution.cost,
        )
        self.best = self._members[0].solution

    def _evaluate(self, chromosome: Chromosome) -> _Individual:
        cost, path = self.decode(chromosome)
        self.evaluations += 1
        return _Individual(chromosome, IDPCSolution(cost, tuple(path)))

    def evolve(self, mutation_rate: float, local_search_passes: int = 1) -> None:
        offspring = []
        while len(offspring) < self.size:
            left, right = self.rng.sample(self._members, 2)
            children = crossover(left.chromosome, right.chromosome, self.rng)
            for child in children[:self.size - len(offspring)]:
                if self.rng.random() < mutation_rate:
                    child = mutation(child, self.limits, self.rng)
                candidate = self._evaluate(child)
                if math.isfinite(candidate.solution.cost):
                    path = self.refine(candidate.solution.edge_ids, local_search_passes)
                    if path != candidate.solution.edge_ids:
                        candidate = self._from_path(path) or candidate
                offspring.append(candidate)
        self._members = sorted(
            self._members + offspring, key=lambda member: member.solution.cost
        )[:self.size]
        self._update_best()

    def elites(self, count: int) -> list[IDPCSolution]:
        elite_paths = set()
        elites = []
        for member in self._members:
            solution = member.solution
            if len(elites) == count or not math.isfinite(solution.cost):
                break
            if solution.edge_ids not in elite_paths:
                elite_paths.add(solution.edge_ids)
                elites.append(solution)
        return elites

    def refine(self, path: tuple[int, ...], passes: int) -> tuple[int, ...]:
        path, checks = self.refinement(path, passes)
        self.local_search_checks += checks
        return path

    def _from_path(self, path: tuple[int, ...]) -> _Individual | None:
        chromosome = encode_path(self.graph, path, self.limits)
        if chromosome is None:
            return None
        candidate = self._evaluate(chromosome)
        if candidate.solution.edge_ids != path or not math.isfinite(candidate.solution.cost):
            return None
        return candidate

    def inject(self, path: tuple[int, ...]) -> str:
        candidate = self._from_path(path)
        if candidate is None:
            return "unrepresentable"
        if any(member.solution.edge_ids == path for member in self._members):
            return "duplicate"
        if candidate.solution.cost >= self._members[-1].solution.cost:
            return "not_better"
        self._members[-1] = candidate
        self._members.sort(key=lambda member: member.solution.cost)
        self._update_best()
        return "accepted"

    def _update_best(self) -> None:
        if self._members[0].solution.cost < self.best.cost:
            self.best = self._members[0].solution
