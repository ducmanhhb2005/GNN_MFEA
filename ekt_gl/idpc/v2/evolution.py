"""Independent EA and one-population MFEA using the same IDPC-DU operators."""

from collections import Counter
from dataclasses import dataclass
import random

from mfea.chromosome import Chromosome
from mfea.operators import crossover

from .core import Candidate, INFEASIBLE, Solution, initialize, limits_for, seed_for, vary
from .local_search import improve


@dataclass
class Member:
    chromosome: Chromosome
    costs: dict[int, float]
    skill: int | None = None
    scalar_fitness: float = 0.0


def assign_ranks(members, task_count, counters=None):
    """Recompute fitness using only evaluated task entries, even if infeasible."""
    columns = [[] for _ in range(task_count)]
    for i, member in enumerate(members):
        for task, cost in member.costs.items():
            columns[task].append((cost, i))
    best = [task_count * len(members) + 1] * len(members)
    skills = [None] * len(members)
    for task, column in enumerate(columns):
        column.sort()
        if counters is not None:
            counters["factorial_rank_ties"] += sum(
                left[0] == right[0] for left, right in zip(column, column[1:]))
        for position, (_, i) in enumerate(column, 1):
            if position < best[i] or (position == best[i] and task == members[i].skill):
                best[i], skills[i] = position, task
    for i, member in enumerate(members):
        member.skill = skills[i]
        member.scalar_fitness = 1.0 / best[i] if skills[i] is not None else 0.0


def select(population, rng, key):
    """Binary tournament. A smaller key and then an earlier index wins."""
    a, b = rng.sample(range(len(population)), 2)
    winner = min((a, b), key=lambda i: (key(population[i]), i))
    return population[winner]


def offspring_ea(population, count, rate, limits, rng, counters):
    cost = lambda member: member.solution.cost
    for start in range(0, count, 2):
        a, b = (select(population, rng, cost) for _ in range(2))
        counters["crossover_matings"] += 1
        for child in crossover(a.chromosome, b.chromosome, rng)[:count - start]:
            counters["crossover_offspring"] += 1
            if rng.random() < rate:
                counters["mutation_calls"] += 1
                child = vary(child, limits, rng)
            yield child


def offspring_mfea(population, count, config, limits, rng, counters):
    fitness = lambda member: -member.scalar_fitness
    for start in range(0, count, 2):
        a, b = (select(population, rng, fitness) for _ in range(2))
        same = a.skill == b.skill
        counters["intra_matings" if same else "inter_matings"] += 1
        if same or rng.random() < config.rmp:
            counters["crossover_matings"] += 1
            counters["inter_crossover_matings"] += not same
            for child in crossover(a.chromosome, b.chromosome, rng)[:count - start]:
                task = rng.choice((a.skill, b.skill))
                counters["crossover_offspring"] += 1
                counters["inherited_offspring"] += 1
                if rng.random() < config.mutation_rate:
                    counters["mutation_calls"] += 1
                    child = vary(child, limits, rng)
                yield child, task
        else:
            counters["mutation_only_matings"] += 1
            for parent in (a, b)[:count - start]:
                counters["mutation_only_offspring"] += 1
                counters["mutation_calls"] += 1
                counters["inherited_offspring"] += 1
                yield vary(parent.chromosome, limits, rng), parent.skill


class Evolution:
    def __init__(self, method, graphs, names, config, seed, tracker):
        self.method = method
        self.config = config
        self.tracker = tracker
        self.seed = seed
        self.names = tuple(names)
        self.local_limits = [limits_for([graph]) for graph in graphs]
        self.unified_limits = limits_for(graphs) if method == "mfea" else None
        self.rngs = [random.Random(seed_for(seed, name, "ea")) for name in names]
        self.mating_rng = random.Random(seed_for(seed, "mfea"))
        self.best = [Solution(INFEASIBLE)] * len(graphs)
        self.best_generation = [None] * len(graphs)
        self.mating = Counter()
        self.generation = 0
        self.populations = None

    def record(self, task, solution):
        if solution.cost < self.best[task].cost:
            self.best[task] = solution
            self.best_generation[task] = self.generation

    def initialize(self):
        n = self.config.population_per_task
        k = len(self.names)
        required = n * k * (k if self.method == "mfea" else 1)
        if self.tracker.budget is not None and self.tracker.budget < required:
            raise ValueError(f"Budget cannot cover {required} initialization requests")
        if self.method == "mfea":
            population = []
            for _ in range(n * k):
                chromosome = initialize(self.unified_limits, self.mating_rng)
                costs = {}
                for task in range(k):
                    solution = self.tracker.gpa(task, chromosome, "init")
                    self.record(task, solution)
                    costs[task] = solution.cost
                population.append(Member(chromosome, costs))
            assign_ranks(population, k, self.mating)
            self.populations = population
        else:
            populations = [[] for _ in range(k)]
            for _ in range(n):
                for task, rng in enumerate(self.rngs):
                    chromosome = initialize(self.local_limits[task], rng)
                    solution = self.tracker.gpa(task, chromosome, "init")
                    self.record(task, solution)
                    populations[task].append(Candidate(chromosome, solution))
            self.populations = [self.survive(pop, n) for pop in populations]

    @staticmethod
    def survive(population, size):
        return sorted(population, key=lambda member: member.solution.cost)[:size]

    def _evaluate(self, chromosome, task, index, pairs):
        solution = self.tracker.gpa(task, chromosome, "offspring")
        self.record(task, solution)
        candidate = Candidate(chromosome, solution)
        if self.config.ordinary_ls(self.method) and solution.feasible:
            rng = random.Random(seed_for(self.seed, self.names[task], "ordinary", self.generation, index))
            candidate = improve(self.tracker, task, candidate, self.local_limits[task], pairs[task],
                                self.config.ls, rng, "ordinary", unified=self.method == "mfea")
            self.record(task, candidate.solution)
        return candidate

    def ordinary(self, pairs):
        n = self.config.population_per_task
        k = len(self.names)
        if self.method == "mfea":
            children = []
            iterator = offspring_mfea(self.populations, n * k, self.config, self.unified_limits,
                                      self.mating_rng, self.mating)
            for i in range(n * k):
                if self.tracker.exhausted:
                    break
                chromosome, task = next(iterator)
                candidate = self._evaluate(chromosome, task, i, pairs)
                children.append(Member(candidate.chromosome, {task: candidate.solution.cost}, task))
            union = self.populations + children
            assign_ranks(union, k, self.mating)
            ordered = sorted(union, key=lambda member: member.scalar_fitness, reverse=True)
            if len(ordered) > n * k and ordered[n * k - 1].scalar_fitness == ordered[n * k].scalar_fitness:
                self.mating["survival_boundary_ties"] += 1
            self.populations = ordered[:n * k]
        else:
            children = [[] for _ in range(k)]
            streams = [offspring_ea(pop, n, self.config.mutation_rate, self.local_limits[task],
                                    self.rngs[task], self.tracker.per_task[task])
                       for task, pop in enumerate(self.populations)]
            for i in range(n):
                if self.tracker.exhausted:
                    break
                for task, stream in enumerate(streams):
                    if self.tracker.exhausted:
                        break
                    children[task].append(self._evaluate(next(stream), task, i, pairs))
            self.populations = [self.survive(pop + kids, n)
                                for pop, kids in zip(self.populations, children)]

    def trace(self):
        if self.method == "mfea":
            count = Counter(member.skill for member in self.populations)
            feasible = Counter(member.skill for member in self.populations
                               if member.costs[member.skill] < INFEASIBLE)
            return [(count[i], feasible[i]) for i in range(len(self.names))]
        return [(len(pop), sum(member.solution.feasible for member in pop)) for pop in self.populations]

    def snapshot(self):
        return dict(populations=self.populations, best=self.best, best_generation=self.best_generation,
                    mating=self.mating, generation=self.generation,
                    rngs=[rng.getstate() for rng in self.rngs], mating_rng=self.mating_rng.getstate(),
                    counters=self.tracker.per_task, requests=self.tracker.requests)

    def restore(self, state):
        for name in ("populations", "best", "best_generation", "mating", "generation"):
            setattr(self, name, state[name])
        for rng, rng_state in zip(self.rngs, state["rngs"]):
            rng.setstate(rng_state)
        self.mating_rng.setstate(state["mating_rng"])
        self.tracker.per_task = state["counters"]
        self.tracker.requests = state["requests"]
