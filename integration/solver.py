"""GNN extension entry point. This is outside the CEC 2020 reconstruction."""
import random
from mfea.chromosome import edge_index_limits
from mfea.solver import _evolve, _validate_parameters


def solve_guided_mfea(tasks, *, initial_population, seed=7, population=100,
                      generations=500, rmp=0.5, mutation_rate=0.05):
    """Reuse the MFEA loop with externally guided P0; never call this paper-only."""
    _validate_parameters(tasks, population, generations, rmp, mutation_rate)
    if len(initial_population) != population:
        raise ValueError("initial_population must match population")
    limits = edge_index_limits(tasks)
    n = max(g.n for g in tasks)
    chromosomes = [c.clone() for c in initial_population]
    for c in chromosomes:
        if len(c.priority) != n or sorted(c.priority) != list(range(1, n+1)) or len(c.edge_index) != n:
            raise ValueError("invalid supplied chromosome dimension or permutation")
        if any(not 1 <= value <= limits[i] for i, value in enumerate(c.edge_index)):
            raise ValueError("supplied edge-index gene is out of range")
        c.scores = None
        c.skill = None
        c.scalar_fitness = 0.0
    return _evolve(tasks, chromosomes, limits, random.Random(seed), generations, rmp, mutation_rate)
