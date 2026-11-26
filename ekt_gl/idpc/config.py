"""Configuration for online transfer on directed, domain-constrained graphs."""

from dataclasses import dataclass

from ekt_gl.config import EKTGLConfig


@dataclass(frozen=True, slots=True)
class IDPCEKTGLConfig(EKTGLConfig):
    embedding_dim: int = 32
    gcn_layers: int = 2
    mutation_rate: float = 0.05
    negative_samples: int = 512
    edge_batch_size: int = 4096
    cache_size: int = 0

    def __post_init__(self) -> None:
        EKTGLConfig.__post_init__(self)
        if self.population_size < 2:
            raise ValueError("IDPC-DU needs at least two individuals per task")
        if self.cache_size < 0:
            raise ValueError("cache_size must be non-negative")
        if self.negative_samples < 1 or self.edge_batch_size < 1:
            raise ValueError("negative_samples and edge_batch_size must be positive")
