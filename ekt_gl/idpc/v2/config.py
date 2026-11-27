"""Frozen comparison profiles and explicit debug overrides."""

from dataclasses import dataclass, field

METHODS = ("ea", "mfea", "ektgl")
SEEDS = tuple(range(100, 120))


@dataclass(frozen=True)
class LocalSearch:
    max_cut_pairs_per_candidate: int = 16
    beam_width: int = 8
    max_edge_expansions_per_cut: int = 4096
    max_complete_paths_per_cut: int = 4
    max_passes: int = 1

    def __post_init__(self):
        if min(vars(self).values()) < 1:
            raise ValueError("All DU-aware 2-opt caps must be positive")


@dataclass(frozen=True)
class Config:
    profile: str = "native"
    population_per_task: int = 30
    generations: int = 300
    mutation_rate: float = 0.05
    rmp: float = 0.5
    transfer_interval: int = 10
    hidden_dim: int = 128
    gcn_layers: int = 4
    train_epochs: int = 1
    elite_count: int = 5
    learning_rate: float = 0.001
    edge_batch_size: int = 32768
    gpa_cache_per_task: int = 512
    ls_beam_cache_per_task: int = 512
    gpa_backend: str = "reference"
    device: str = "cpu"
    threads: int = 4
    budget_override: int | None = None
    ls: LocalSearch = field(default_factory=LocalSearch)

    def __post_init__(self):
        if self.gpa_backend not in ("reference", "numba"):
            raise ValueError("Unknown GPA backend")
        if self.profile not in ("native", "matched_ls"):
            raise ValueError("Unknown benchmark profile")
        if not (0 <= self.rmp <= 1 and 0 <= self.mutation_rate <= 1):
            raise ValueError("Invalid mating or mutation probability")
        if min(self.population_per_task - 1, self.generations, self.transfer_interval,
               self.hidden_dim, self.gcn_layers, self.train_epochs, self.elite_count,
               self.learning_rate, self.edge_batch_size, self.threads) <= 0:
            raise ValueError("Nonpositive benchmark parameter")
        if (min(self.gpa_cache_per_task, self.ls_beam_cache_per_task) < 0
                or (self.budget_override is not None and self.budget_override < 1)):
            raise ValueError("Invalid evaluator cache or budget")

    def budget(self, task_count):
        return (self.budget_override or self.population_per_task * task_count * (self.generations + 1)
                if self.profile == "matched_ls" else None)

    def ordinary_ls(self, method):
        return method == "ektgl" or self.profile == "matched_ls"
