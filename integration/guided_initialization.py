"""Task-specific GNN-guided initialization for the *existing* CEC-2020 MFEA baseline.

No oracle/ground-truth labels are used at search time. `edge_scores` must come
from a trained scorer/model; scores do NOT directly constitute MFEA chromosome.

IMPORTANT: CEC-2020 projects a unified priority permutation onto task j by
stable-filtering values <= n_j. We explicitly construct a global permutation
whose task-j projection equals our proposed priority array. This is a
research hypothesis, not a published operation from the CEC-2020 algorithm.
"""
from __future__ import annotations
import math
import random
from idpc_du.graph import Graph
from mfea.chromosome import Chromosome, edge_index_limits, init_chrom


def edge_scores_to_node_priority(g: Graph, edge_scores: list[float],
                                 rng: random.Random, jitter: float = 0.0) -> list[int]:
    """Heuristic map edge goodness -> priorities assigned to vertex indices.

    Each destination node receives maximum predicted score among its incoming
    edges; unmatched nodes receive 0.0. Low scored vertices get low integer
    priority; GPA chooses the largest feasible neighbor priority first.

    The max aggregation is an explicit design decision, NOT an established
    theorem or a mapping prescribed by either CEC paper.
    """
    if len(edge_scores) != len(g.edges):
        raise ValueError("edge scores must match number of graph edges")
    if not 0 <= jitter <= 1:
        raise ValueError("jitter must be in [0,1]")
    scores = [0.0] * g.n
    for edge, score in zip(g.edges, edge_scores):
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("each edge score must lie in [0,1]")
        scores[edge.v] = max(scores[edge.v], float(score))
    # For the source node the incoming score is irrelevant to a path starting here.
    scores[g.source] = -1.0
    noise = [rng.uniform(-jitter, jitter) for _ in range(g.n)]
    order = sorted(range(g.n), key=lambda v: (scores[v] + noise[v], v))
    priorities = [0] * g.n
    for rank, vertex in enumerate(order, start=1):
        priorities[vertex] = rank
    assert sorted(priorities) == list(range(1, g.n+1))
    return priorities


def lift_task_priority(task_pi: list[int], max_nodes: int,
                       rng: random.Random) -> list[int]:
    """Embed a task priority as an MFEA unified genotype under stable filtering.

    Given pi_j, randomly interleave symbols n_j+1..N while preserving the
    order of [1..n_j] priority values. Stable-filtering the output for <= n_j
    therefore gives back pi_j *exactly*.
    """
    n = len(task_pi)
    if not (1 <= n <= max_nodes) or sorted(task_pi) != list(range(1, n+1)):
        raise ValueError("task_pi must be a permutation of 1..n")
    priority = list(task_pi)
    extras = list(range(n+1, max_nodes+1))
    rng.shuffle(extras)
    for gene in extras:
        priority.insert(rng.randrange(len(priority)+1), gene)
    assert [x for x in priority if x <= n] == task_pi
    return priority


def make_guided_population(tasks: list[Graph], edge_score_vectors: list[list[float]],
                           population: int, seed: int,
                           guided_fraction: float = 0.5,
                           jitter: float = 0.04) -> tuple[list[Chromosome], dict]:
    """Create mixed guided/random P0; different guided individuals target different tasks.

    Edge Index (sigma) remains random, so improvement in initial population
    specifically tests learned priority (pi) rather than changed GPA/fitness.
    """
    if not tasks or len(tasks) != len(edge_score_vectors):
        raise ValueError("one edge-score vector per task is required")
    if population < 4 or not 0 <= guided_fraction <= 1:
        raise ValueError("population>=4 and guided_fraction in [0,1] required")
    rng = random.Random(seed)
    n = max(t.n for t in tasks)
    limits = edge_index_limits(tasks)
    n_guided = round(population * guided_fraction)
    # Pre-compute one score-per-node ordering for each task; random jitter makes
    # independently guided chromosomes nonidentical without losing permutation.
    guided_tasks = [j % len(tasks) for j in range(n_guided)]
    rng.shuffle(guided_tasks)
    chromosomes = []
    for target_task in guided_tasks:
        chromosome = init_chrom(n, limits, rng)
        task_pi = edge_scores_to_node_priority(tasks[target_task],
                    edge_score_vectors[target_task], rng, jitter=jitter)
        chromosome.priority = lift_task_priority(task_pi, n, rng)
        chromosomes.append(chromosome)
    for _ in range(population - n_guided):
        chromosomes.append(init_chrom(n, limits, rng))
    rng.shuffle(chromosomes)
    return chromosomes, {"guided_count": n_guided,
                         "random_count": population-n_guided,
                         "guided_task_allocations": [guided_tasks.count(j) for j in range(len(tasks))]}
