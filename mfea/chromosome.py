"""Two-segment (pi | sigma) chromosome and random P0 creation."""
from __future__ import annotations
from dataclasses import dataclass
import random
from idpc_du.graph import Graph

@dataclass
class Chromosome:
    priority: list[int]
    edge_index: list[int]
    skill: int | None = None
    scores: dict[int,float] | None = None
    scalar_fitness: float = 0.0

    def clone(self):
        return Chromosome(self.priority[:], self.edge_index[:], self.skill,
                          None if self.scores is None else dict(self.scores), self.scalar_fitness)


def edge_index_limits(tasks: list[Graph]) -> list[int]:
    n=max(g.n for g in tasks)
    # A node with no outgoing edges has '-' in Algorithm 2. Store 1 as a
    # numeric placeholder so Algorithm 1's arriving-node modulo is defined;
    # this convention matters for sink vertices and is documented explicitly.
    limits=[1]*n
    for g in tasks:
        for u in range(g.n):
            # The parser already groups parallel edges; reuse these exact
            # counts instead of recounting millions of edges for every seed.
            parallel = g.by_neighbor.get(u, {})
            limits[u] = max(limits[u], max((len(edges) for edges in parallel.values()), default=1))
    return limits


def init_chrom(n: int, limits: list[int], rng: random.Random) -> Chromosome:
    priorities=list(range(1,n+1))
    rng.shuffle(priorities)
    return Chromosome(priorities, [rng.randint(1,limit) for limit in limits])
