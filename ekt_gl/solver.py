"""Algorithm 1 orchestration for evolutionary knowledge transfer via graph learning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .config import EKTGLConfig
from .evolution import Individual, Population
from .network import EKTGraphNetwork
from .problems import RoutingTask
from .routing import greedy_decode, two_opt
from .training import OnlineTrainer, TrainingReport


@dataclass(frozen=True, slots=True)
class EKTGLResult:
    """Best solution per task plus optimization traces for reproducibility."""

    best: dict[str, Individual]
    history: dict[str, tuple[float, ...]]
    transfer_reports: tuple[TrainingReport, ...]


class EKTGL:
    """Many-task heterogeneous routing solver described in Algorithm 1.

    The network—not a permutation—is the common knowledge carrier. Each target
    task nevertheless retains its own population and objective evaluation.
    """

    def __init__(self, config: EKTGLConfig | None = None) -> None:
        self.config = config or EKTGLConfig()

    def solve(self, tasks: Sequence[RoutingTask], seed: int | None = None) -> EKTGLResult:
        if not tasks:
            raise ValueError("EKTGL.solve requires one or more routing tasks")
        names = [task.name for task in tasks]
        if len(names) != len(set(names)):
            raise ValueError("routing task names must be unique")

        rng = np.random.default_rng(seed)
        populations = [Population(task, self.config.population_size, rng) for task in tasks]
        network = EKTGraphNetwork(self.config.embedding_dim, self.config.gcn_layers)
        trainer = OnlineTrainer(network, self.config.learning_rate, self.config.device)
        history: dict[str, list[float]] = {task.name: [population.best.cost] for task, population in zip(tasks, populations)}
        reports: list[TrainingReport] = []

        for generation in range(self.config.generations):
            if generation % self.config.transfer_interval == 0:
                reports.append(
                    trainer.fit(tasks, populations, self.config.elite_count, self.config.train_epochs)
                )
                heatmaps, _ = trainer.heatmaps(tasks)
                for task, population, heatmap in zip(tasks, populations, heatmaps):
                    transferred = greedy_decode(task, heatmap, rng)
                    refined = two_opt(task, transferred, self.config.local_search_passes)
                    population.inject(refined)
            else:
                for population in populations:
                    population.evolve(self.config.mutation_rate, self.config.local_search_passes)
            for task, population in zip(tasks, populations):
                history[task.name].append(population.best.cost)

        return EKTGLResult(
            best={task.name: population.best for task, population in zip(tasks, populations)},
            history={name: tuple(costs) for name, costs in history.items()},
            transfer_reports=tuple(reports),
        )
