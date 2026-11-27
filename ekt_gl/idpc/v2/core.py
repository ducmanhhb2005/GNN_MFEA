"""CEC chromosome helpers and shared budgeted exact evaluations."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import math

from idpc_du.gpa import decode_gpa, eligible_edges
from idpc_du.validation import validate_path
from mfea.chromosome import Chromosome


REASONS = ("init", "offspring", "ls", "transfer", "other")
REQUESTS = "objective_evaluation_requests"
INFEASIBLE = float("inf")


def seed_for(seed, *parts):
    key = repr((seed, *parts)).encode()
    return int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "little")


def limits_for(graphs):
    limits = [0] * max(graph.n for graph in graphs)
    for graph in graphs:
        for u, neighbors in graph.by_neighbor.items():
            limits[u] = max(limits[u], max(map(len, neighbors.values()), default=0))
    return limits


def initialize(limits, rng):
    pi = list(range(1, len(limits) + 1))
    rng.shuffle(pi)
    return Chromosome(pi, [rng.randint(1, limit) if limit else -1 for limit in limits])


def vary(chromosome, limits, rng):
    child = chromosome.clone()
    if len(limits) >= 2:
        i, j = rng.sample(range(len(limits)), 2)
        child.priority[i], child.priority[j] = child.priority[j], child.priority[i]
    valid = [i for i, limit in enumerate(limits) if limit]
    if valid:
        i = rng.choice(valid)
        child.edge_index[i] = rng.randint(1, limits[i])
    child.scores = None
    child.scalar_fitness = 0.0
    return child


def project(chromosome, n):
    return Chromosome([p for p in chromosome.priority if p <= n], chromosome.edge_index[:n])


def lift(local, unified):
    """Embed a local priority permutation into the slots retained by projection."""
    n = len(local.priority)
    priorities = iter(local.priority)
    sigma = [value if value != -1 else (unified.edge_index[i] if unified.edge_index[i] == -1 else 1)
             for i, value in enumerate(local.edge_index)]
    return Chromosome([next(priorities) if p <= n else p for p in unified.priority],
                      sigma + unified.edge_index[n:], skill=unified.skill)


def encode(graph, path, limits):
    """Encode an independently validated path with a constructive priority order.

    The next vertex outranks every alternative unvisited vertex at each step:
    all later path vertices and all off-path vertices have lower priority.
    """
    edges = [graph.by_id[edge_id] for edge_id in path]
    destinations = [edge.v for edge in edges]
    used = set(destinations)
    order = destinations + [v for v in range(graph.n) if v not in used]
    priorities = [0] * graph.n
    for priority, v in enumerate(reversed(order), 1):
        priorities[v] = priority
    sigma = [1 if lim else -1 for lim in limits]
    left = set()
    active = None
    for edge in edges:
        allowed = eligible_edges(graph.by_neighbor[edge.u][edge.v], left)
        get_index = getattr(allowed, "index_of_id", None)
        position = (get_index(edge.id) if get_index else
                    next(i for i, item in enumerate(allowed) if item.id == edge.id))
        limit = limits[edge.v]
        if limit:
            gene = position or len(allowed)
            if gene > limit:
                return None
            sigma[edge.v] = gene
        elif 1 % len(allowed) != position:
            return None
        if active is not None and active != edge.domain:
            left.add(active)
        active = edge.domain
    return Chromosome(priorities, sigma)


@dataclass(frozen=True)
class Solution:
    cost: float
    path: tuple[int, ...] = ()

    @property
    def feasible(self):
        return math.isfinite(self.cost)


@dataclass(frozen=True)
class Candidate:
    chromosome: Chromosome
    solution: Solution


class EvaluationTracker:
    """Use one objective request for a cache hit and one for a cache miss."""

    def __init__(self, graphs, budget=None, cache_size=512, decoders=None):
        self.graphs = graphs
        self.decoders = decoders
        self.budget = budget
        self.requests = 0
        self.per_task = [Counter() for _ in graphs]
        self.cache_size = cache_size
        self.clear_caches()

    def clear_caches(self):
        self.cache = [{} for _ in self.graphs]
        self.path_cache = [{} for _ in self.graphs]

    @property
    def exhausted(self):
        return self.budget is not None and self.requests >= self.budget

    def _charge(self, task, reason):
        if reason not in REASONS:
            raise ValueError(f"Unknown evaluation reason {reason}")
        if self.exhausted:
            return None
        counters = self.per_task[task]
        counters[REQUESTS] += 1
        counters[reason + "_evaluations"] += 1
        self.requests += 1
        return counters

    def gpa(self, task, chromosome, reason):
        counters = self._charge(task, reason)
        if counters is None:
            return None
        graph = self.graphs[task]
        local = project(chromosome, graph.n) if len(chromosome.priority) != graph.n else chromosome
        sigma = tuple(local.edge_index)
        if -1 in sigma:
            sigma = tuple(1 if s == -1 else s for s in sigma)
        key = tuple(local.priority), sigma
        cache = self.cache[task]
        if key in cache:
            counters["cache_hits"] += 1
            return cache[key]
        counters["actual_objective_computations"] += 1
        counters["gpa_decode_calls"] += 1
        local = Chromosome(local.priority, sigma)
        score, path = (decode_gpa(graph, local) if self.decoders is None
                       else self.decoders[task](local))
        if path or math.isfinite(score):
            counters["validator_calls"] += 1
        solution = Solution(score, tuple(path))
        if self.cache_size:
            if len(cache) >= self.cache_size:
                del cache[next(iter(cache))]
            cache[key] = solution
        return solution

    def path(self, task, edge_ids, reason):
        counters = self._charge(task, reason)
        if counters is None:
            return None
        cache = self.path_cache[task]
        if edge_ids in cache:
            counters["cache_hits"] += 1
            return cache[edge_ids]
        counters["actual_objective_computations"] += 1
        counters["validator_calls"] += 1
        evaluated = Solution(validate_path(self.graphs[task], edge_ids), edge_ids)
        if self.cache_size:
            if len(cache) >= self.cache_size:
                del cache[next(iter(cache))]
            cache[edge_ids] = evaluated
        return evaluated

    def roundtrip(self, task, solution, limits, reason, unified=None, phase=None):
        if self.exhausted or not solution.feasible:
            return None
        counts = self.per_task[task]

        def count(key):
            counts[key] += 1
            if phase:
                counts[phase + "_" + key] += 1

        count("encode_attempts")
        chromosome = encode(self.graphs[task], solution.path, limits)
        if chromosome is None:
            count("encode_failures")
            return None
        count("encode_successes")
        if unified is not None:
            chromosome = lift(chromosome, unified)
        decoded = self.gpa(task, chromosome, reason)
        if decoded != solution:
            count("roundtrip_failures")
            return None
        count("roundtrip_successes")
        return Candidate(chromosome, decoded)

    def totals(self):
        result = Counter()
        for counters in self.per_task:
            result.update(counters)
        return result

    def check_invariants(self):
        for c in self.per_task:
            assert c[REQUESTS] == sum(c[name + "_evaluations"] for name in REASONS)
            assert c[REQUESTS] == c["actual_objective_computations"] + c["cache_hits"]
        assert self.requests == sum(c[REQUESTS] for c in self.per_task)
        assert self.budget is None or self.requests <= self.budget
