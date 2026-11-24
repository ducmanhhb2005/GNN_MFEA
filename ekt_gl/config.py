"""Configuration for the EKT-GL workflow."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EKTGLConfig:
    """Hyperparameters from the paper, with conservative CPU-friendly defaults."""

    population_size: int = 30
    generations: int = 300
    transfer_interval: int = 10
    elite_count: int = 5
    train_epochs: int = 1
    embedding_dim: int = 128
    gcn_layers: int = 4
    learning_rate: float = 1e-3
    mutation_rate: float = 0.2
    local_search_passes: int = 1
    device: str = "cpu"

    def __post_init__(self) -> None:
        positive = (
            "population_size", "generations", "transfer_interval", "elite_count",
            "train_epochs", "embedding_dim", "gcn_layers", "local_search_passes",
        )
        if any(getattr(self, field) <= 0 for field in positive):
            raise ValueError("all integer EKTGLConfig values must be positive")
        if not 0.0 <= self.mutation_rate <= 1.0:
            raise ValueError("mutation_rate must be in [0, 1]")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
