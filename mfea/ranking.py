"""Factorial rank, skill factor and scalar fitness for MFEA."""
from __future__ import annotations
import random
from .chromosome import Chromosome

def assign_ranks(pop: list[Chromosome], task_count: int,
                 rng: random.Random | None = None, initial: bool = False) -> None:
    """MFEA scalar fitness = 1 / min factorial rank among EVALUATED tasks.

    Offspring are evaluated only on inherited skill, while original individuals
    are evaluated on all tasks. Missing task costs are NOT ranked as valid
    entries; assigning them a finite rank produces fictitious fitness.
    """
    ranks = [dict() for _ in range(task_count)]
    for task in range(task_count):
        order = [i for i, c in enumerate(pop) if c.scores is not None and task in c.scores]
        if rng is not None:
            rng.shuffle(order)  # randomly break equal fitness ties (incl. +inf)
        order.sort(key=lambda i: pop[i].scores[task])
        for rank, idx in enumerate(order, 1):
            ranks[task][idx] = rank
    for i, c in enumerate(pop):
        candidates = [(ranks[t][i], t) for t in range(task_count) if i in ranks[t]]
        if not candidates:
            raise ValueError("Individual was not evaluated on any task")
        # In the original MFEA, skill factor is argmin of factorial ranks.
        min_rank = min(rank for rank, _ in candidates)
        tied = [t for rank, t in candidates if rank == min_rank]
        c.skill = rng.choice(tied) if rng is not None and len(tied) > 1 else tied[0]
        c.scalar_fitness = 1.0 / min_rank
