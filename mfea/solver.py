"""MFEA-EDU reconstruction from CEC 2020; table reproduction is unverified."""
from __future__ import annotations
import math
import random
from idpc_du.graph import Graph
from idpc_du.gpa import decode_gpa
from .chromosome import edge_index_limits, init_chrom
from .operators import crossover, mutation
from .ranking import assign_ranks

def _validate_parameters(tasks, population, generations, rmp, mutation_rate):
    if not tasks:
        raise ValueError("Need at least one task")
    if population < 4 or generations < 1 or not 0 <= rmp <= 1 or not 0 <= mutation_rate <= 1:
        raise ValueError("invalid MFEA parameters")


def solve_mfea(tasks: list[Graph], seed: int = 7, population: int = 100,
               generations: int = 500, rmp: float = 0.5,
               mutation_rate: float = 0.05, *, diagnostics: dict | None = None):
    """MFEA with random initialization (Algorithm 2) and the paper's operators.

    The article omits detailed parent/tie selection and task grouping. This
    implementation uses uniform parents, random rank ties, skill inheritance,
    inherited-task offspring evaluation, and elitist scalar-fitness survival.
    These are explicit MFEA framework assumptions, not verified author code.
    GNN initialization is exposed separately by integration.solver.
    diagnostics, when supplied, receives evaluation counts and objective extrema;
    it never affects variation, selection, or random draws.
    """
    _validate_parameters(tasks, population, generations, rmp, mutation_rate)
    rng = random.Random(seed)
    limits = edge_index_limits(tasks)
    n = max(g.n for g in tasks)
    population_list = [init_chrom(n, limits, rng) for _ in range(population)]
    return _evolve(tasks, population_list, limits, rng, generations, rmp, mutation_rate, diagnostics)


def solve_ea(task: Graph, seed: int = 7, population: int = 100,
             generations: int = 500, mutation_rate: float = 0.05, *, diagnostics=None):
    """Single-task EA control with the same representation and operators.

    A single task has no cross-task mating/transfer. The return shape matches
    solve_mfea: a one-element best list and per-generation history.
    """
    return solve_mfea([task], seed=seed, population=population, generations=generations,
                      mutation_rate=mutation_rate, diagnostics=diagnostics)


def _evolve(tasks, population_list, limits, rng, generations, rmp, mutation_rate, diagnostics=None):
    """Shared evolutionary loop; callers own the initialization step."""
    population = len(population_list)
    K = len(tasks)
    counts = [0] * K
    minima = [math.inf] * K
    maxima = [-math.inf] * K
    infeasible = [0] * K

    def evaluate(task, chromosome):
        score, path = decode_gpa(tasks[task], chromosome)
        counts[task] += 1
        if math.isfinite(score):
            minima[task] = min(minima[task], score)
            maxima[task] = max(maxima[task], score)
        else:
            infeasible[task] += 1
        return score, path

    best = [(math.inf, []) for _ in tasks]
    for c in population_list:
        c.scores = {}
        for t, g in enumerate(tasks):
            score, path = evaluate(t, c)
            c.scores[t] = score
            if score < best[t][0]:
                best[t] = (score, path)
    assign_ranks(population_list, K, rng=rng, initial=True)
    history = [tuple(x[0] for x in best)]
    for _ in range(generations):
        kids = []
        while len(kids) < population:
            pa, pb = rng.sample(population_list, 2)
            if pa.skill == pb.skill or rng.random() < rmp:
                ca, cb = crossover(pa, pb, rng)
                ca.skill = rng.choice((pa.skill, pb.skill))
                cb.skill = rng.choice((pa.skill, pb.skill))
            else:
                ca, cb = mutation(pa, limits, rng), mutation(pb, limits, rng)
                ca.skill, cb.skill = pa.skill, pb.skill
            for child in (ca, cb):
                if rng.random() < mutation_rate:
                    child = mutation(child, limits, rng)
                inherited_task = child.skill
                cost, path = evaluate(inherited_task, child)
                child.scores = {inherited_task: cost}
                if cost < best[inherited_task][0]:
                    best[inherited_task] = (cost, path)
                kids.append(child)
                if len(kids) >= population:
                    break
        both = population_list + kids
        assign_ranks(both, K, rng=rng)
        rng.shuffle(both)
        population_list = sorted(both, key=lambda x: -x.scalar_fitness)[:population]
        history.append(tuple(x[0] for x in best))
    if diagnostics is not None:
        diagnostics.update(evaluations=counts, infeasible_evaluations=infeasible,
                           objective_min=[v if math.isfinite(v) else None for v in minima],
                           objective_max=[v if math.isfinite(v) else None for v in maxima])
    return best, history
