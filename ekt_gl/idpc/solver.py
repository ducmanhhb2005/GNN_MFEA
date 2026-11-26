"""EKT-GL loop for IDPC-DU with one population per graph."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from functools import partial
import math
import random
from time import perf_counter

from idpc_du.validation import validate_path

from .config import IDPCEKTGLConfig
from .decoding import greedy_path
from .population import IDPCPopulation, IDPCSolution
from .task import IDPCGraph
from .training import OnlineIDPCTrainer, elite_edge_labels


@dataclass(frozen=True, slots=True)
class IDPCTransferReport:
    generation: int
    loss: float | None
    labelled_tasks: tuple[str, ...]
    attention: tuple[tuple[float, ...], ...] | None
    outcomes: dict[str, str]
    training_seconds: float = field(default=0.0, compare=False)
    inference_seconds: float = field(default=0.0, compare=False)


@dataclass(frozen=True, slots=True)
class IDPCEKTGLResult:
    best: dict[str, IDPCSolution]
    history: dict[str, tuple[float, ...]]
    evaluations: dict[str, int]
    evaluation_history: dict[str, tuple[int, ...]]
    local_search_checks: dict[str, int]
    transfers: tuple[IDPCTransferReport, ...]


class IDPCEKTGL:
    def __init__(self, config: IDPCEKTGLConfig | None = None) -> None:
        self.config = config or IDPCEKTGLConfig()

    def solve(
        self, tasks: Mapping[str, IDPCGraph], seed: int = 7,
        *, transfer: bool = True, progress: Callable[[int], None] | None = None,
        population_seeds: Mapping[str, int] | None = None,
    ) -> IDPCEKTGLResult:
        if not tasks:
            raise ValueError("At least one IDPC-DU task is required")
        names, graphs = zip(*tasks.items())
        config = self.config
        rng = random.Random(seed)
        if population_seeds is None:
            population_seeds = {name: rng.getrandbits(64) for name in names}
        elif population_seeds.keys() != tasks.keys():
            raise ValueError("population_seeds must match the task names")
        populations = [
            IDPCPopulation(graph, config.population_size, random.Random(population_seeds[name]), config.cache_size)
            for name, graph in tasks.items()
        ]
        trainer = OnlineIDPCTrainer(list(graphs), config, rng.getrandbits(64)) if transfer else None
        history = {name: [population.best.cost] for name, population in zip(names, populations)}
        evaluation_history = {name: [p.evaluations] for name, p in zip(names, populations)}
        transfers = []
        for generation in range(config.generations):
            transferred = False
            if trainer is not None and generation % config.transfer_interval == 0:
                report = self._transfer(generation, names, populations, trainer)
                transfers.append(report)
                transferred = report.loss is not None
            if not transferred:
                for population in populations:
                    population.evolve(config.mutation_rate, config.local_search_passes)
            for name, population in zip(names, populations):
                history[name].append(population.best.cost)
                evaluation_history[name].append(population.evaluations)
            if progress is not None:
                progress(generation + 1)
        return IDPCEKTGLResult(
            best={name: population.best for name, population in zip(names, populations)},
            history={name: tuple(costs) for name, costs in history.items()},
            evaluations={name: population.evaluations for name, population in zip(names, populations)},
            evaluation_history={name: tuple(counts) for name, counts in evaluation_history.items()},
            local_search_checks={name: p.local_search_checks for name, p in zip(names, populations)},
            transfers=tuple(transfers),
        )

    def _transfer(
        self, generation: int, names: tuple[str, ...], populations: list[IDPCPopulation],
        trainer: OnlineIDPCTrainer,
    ) -> IDPCTransferReport:
        config = self.config
        labels = [elite_edge_labels(population.elites(config.elite_count)) for population in populations]
        labelled = tuple(name for name, edges in zip(names, labels) if edges)
        if not labelled:
            return IDPCTransferReport(generation, None, (), None, dict.fromkeys(names, "no_elites"))

        started = perf_counter()
        loss = trainer.fit(labels)
        training_seconds = perf_counter() - started
        started = perf_counter()
        nodes, fused, attention = trainer.encode()
        inference_seconds = perf_counter() - started
        outcomes = {}
        for index, (name, population) in enumerate(zip(names, populations)):
            graph = population.graph
            scorer = partial(trainer.score_edges, index, nodes[index], fused[index])
            started = perf_counter()
            path = greedy_path(graph, scorer, config.edge_batch_size)
            inference_seconds += perf_counter() - started
            if path is None:
                outcomes[name] = "dead_end"
                continue
            path = population.refine(path, config.local_search_passes)
            if not math.isfinite(validate_path(graph, path)):
                outcomes[name] = "invalid"
                continue
            outcomes[name] = population.inject(path)
        return IDPCTransferReport(
            generation, loss, labelled, tuple(tuple(row) for row in attention.cpu().tolist()), outcomes,
            training_seconds, inference_seconds,
        )
