"""Build online edge labels from current elites and update the shared network."""

from __future__ import annotations

from collections.abc import Sequence
import random

import torch
from torch import Tensor
from torch.nn import functional as F

from idpc_du.graph import Edge

from .config import IDPCEKTGLConfig
from .features import GraphFeatures
from .network import IDPCGraphNetwork
from .population import IDPCSolution
from .task import IDPCGraph


def elite_edge_labels(elites: list[IDPCSolution]) -> set[int]:
    """The union of elite paths is the positive class. All other edges are zero."""
    return {edge_id for solution in elites for edge_id in solution.edge_ids}


def _training_edges(
    graph: IDPCGraph, positives: set[int], max_negatives: int, rng: random.Random
) -> tuple[list[Edge], int]:
    # Drawing at most k + number of positives guarantees k distinct negatives
    # when enough exist. Taking the first k surviving draws is uniform.
    negative_count = min(max_negatives, len(graph.edges) - len(positives))
    negatives = []
    if negative_count:
        sample = rng.sample(range(len(graph.edges)), negative_count + len(positives))
        for index in sample:
            edge = graph.edges[index]
            if edge.id not in positives:
                negatives.append(edge)
                if len(negatives) == negative_count:
                    break
    positive_edges = [graph.by_id[edge_id] for edge_id in sorted(positives)]
    return positive_edges + negatives, len(positive_edges)


class OnlineIDPCTrainer:
    def __init__(self, graphs: list[IDPCGraph], config: IDPCEKTGLConfig, seed: int) -> None:
        self.graphs = graphs
        self.config = config
        self.features = [GraphFeatures.from_graph(graph, config.device) for graph in graphs]
        self.rng = random.Random(seed)
        # Initialize weights reproducibly without changing the caller's RNG.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.network = IDPCGraphNetwork(config.embedding_dim, config.gcn_layers)
        self.network.to(config.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=config.learning_rate)

    def fit(self, labels: list[set[int]]) -> float:
        examples = {}
        for index, positives in enumerate(labels):
            if positives:
                edges, split = _training_edges(
                    self.graphs[index], positives, self.config.negative_samples, self.rng
                )
                examples[index] = self.features[index].edge_inputs(edges), split
        if not examples:
            raise ValueError("Online training requires at least one task with an elite path")

        self.network.train()
        for _ in range(self.config.train_epochs):
            nodes, fused, _ = self.network.encode(self.features)
            losses = []
            for index, (batch, split) in examples.items():
                logits = self.network.edge_logits(nodes[index], fused[index], batch)
                # Binary cross-entropy with equal weight for each available class.
                positive_loss = F.softplus(-logits[:split]).mean()
                if split == len(logits):
                    losses.append(positive_loss)
                else:
                    negative_loss = F.softplus(logits[split:]).mean()
                    losses.append((positive_loss + negative_loss) / 2)
            loss = torch.stack(losses).mean()
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
        return float(loss.detach().cpu())

    @torch.no_grad()
    def encode(self) -> tuple[list[Tensor], Tensor, Tensor]:
        self.network.eval()
        return self.network.encode(self.features)

    @torch.no_grad()
    def score_edges(self, index: int, nodes: Tensor, context: Tensor, edges: Sequence[Edge]) -> list[float]:
        logits = self.network.edge_logits(nodes, context, self.features[index].edge_inputs(edges))
        # Sigmoid preserves the order of logits used by greedy decoding.
        return logits.cpu().tolist()
