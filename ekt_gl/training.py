"""Online supervised fitting of EKT-GL using current population elites."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from .evolution import Population
from .network import EKTGraphNetwork
from .problems import RoutingTask


@dataclass(frozen=True, slots=True)
class TrainingReport:
    loss: float
    attention: np.ndarray


def elite_edge_labels(task: RoutingTask, population: Population, elite_count: int) -> np.ndarray:
    """Build the binary edge label in Eq. (11) from the union of top-ε routes."""
    label = np.zeros((task.node_count, task.node_count), dtype=np.float32)
    for individual in population.elites(elite_count):
        label = np.maximum(label, task.edge_matrix(individual.route))
    np.fill_diagonal(label, 0.0)
    return label


class OnlineTrainer:
    """Keeps Adam state across migrations, so the model is genuinely online-updated."""

    def __init__(self, network: EKTGraphNetwork, learning_rate: float, device: str = "cpu") -> None:
        self.network = network.to(device)
        self.device = torch.device(device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)

    def _coordinates(self, tasks: Sequence[RoutingTask]) -> list[Tensor]:
        return [torch.as_tensor(task.coordinates, dtype=torch.float32, device=self.device) for task in tasks]

    def fit(
        self,
        tasks: Sequence[RoutingTask],
        populations: Sequence[Population],
        elite_count: int,
        epochs: int,
    ) -> TrainingReport:
        if len(tasks) != len(populations):
            raise ValueError("each task must have exactly one population")
        labels = [
            torch.as_tensor(elite_edge_labels(task, population, elite_count), device=self.device)
            for task, population in zip(tasks, populations)
        ]
        masks = [~torch.eye(task.node_count, dtype=torch.bool, device=self.device) for task in tasks]
        coordinates = self._coordinates(tasks)
        self.network.train()
        final_loss = torch.tensor(0.0, device=self.device)
        for _ in range(epochs):
            output = self.network(coordinates)
            losses = [
                F.binary_cross_entropy_with_logits(logit[mask], label[mask])
                for logit, label, mask in zip(output.edge_logits, labels, masks)
            ]
            final_loss = torch.stack(losses).mean()
            self.optimizer.zero_grad(set_to_none=True)
            final_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
            self.optimizer.step()
        return TrainingReport(loss=float(final_loss.detach().cpu()), attention=output.attention.detach().cpu().numpy())

    @torch.no_grad()
    def heatmaps(self, tasks: Sequence[RoutingTask]) -> tuple[list[np.ndarray], np.ndarray]:
        self.network.eval()
        output = self.network(self._coordinates(tasks))
        heatmaps = [torch.sigmoid(logits).cpu().numpy() for logits in output.edge_logits]
        return heatmaps, output.attention.cpu().numpy()
