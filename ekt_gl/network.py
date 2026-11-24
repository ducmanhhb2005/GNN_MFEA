"""The GCN-attention-MLP model used as EKT-GL's knowledge carrier.

The equations in this module correspond to Eq. (2)--(10) of the paper.  Task
graphs intentionally remain a list rather than padded batches: this preserves
their heterogeneous node counts and avoids treating padding as graph nodes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class EdgeGatedGraphConv(nn.Module):
    """One anisotropic edge-gated graph convolutional layer (Eq. 2--3)."""

    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.node_self = nn.Linear(dimension, dimension, bias=False)  # U^l
        self.node_neighbour = nn.Linear(dimension, dimension, bias=False)  # V^l
        self.edge_self = nn.Linear(dimension, dimension, bias=False)  # A^l
        self.edge_source = nn.Linear(dimension, dimension, bias=False)  # B^l
        self.edge_target = nn.Linear(dimension, dimension, bias=False)  # C^l
        self.node_norm = nn.BatchNorm1d(dimension)
        self.edge_norm = nn.BatchNorm1d(dimension)

    def forward(self, nodes: Tensor, edges: Tensor) -> tuple[Tensor, Tensor]:
        node_count = nodes.shape[0]
        gated_messages = torch.sigmoid(edges) * self.node_neighbour(nodes).unsqueeze(0)
        # A complete graph's neighbourhood excludes itself.
        diagonal = torch.eye(node_count, dtype=torch.bool, device=nodes.device)
        gated_messages = gated_messages.masked_fill(diagonal.unsqueeze(-1), 0.0)
        aggregate = gated_messages.sum(dim=1) / max(node_count - 1, 1)
        node_delta = F.relu(self.node_norm(self.node_self(nodes) + aggregate))

        edge_pre_activation = (
            self.edge_self(edges)
            + self.edge_source(nodes).unsqueeze(1)
            + self.edge_target(nodes).unsqueeze(0)
        )
        edge_delta = F.relu(self.edge_norm(edge_pre_activation.flatten(0, 1))).view_as(edges)
        return nodes + node_delta, edges + edge_delta


class GraphEncoder(nn.Module):
    """Four-layer edge-gated GCN encoder, unless configured otherwise."""

    def __init__(self, dimension: int = 128, layers: int = 4) -> None:
        super().__init__()
        self.node_projection = nn.Linear(2, dimension)
        self.edge_projection = nn.Linear(1, dimension)
        self.layers = nn.ModuleList(EdgeGatedGraphConv(dimension) for _ in range(layers))

    def forward(self, coordinates: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        distances = torch.cdist(coordinates, coordinates, p=2).unsqueeze(-1)
        nodes, edges = self.node_projection(coordinates), self.edge_projection(distances)
        for layer in self.layers:
            nodes, edges = layer(nodes, edges)
        return nodes, edges, nodes.mean(dim=0)


class KnowledgeFusion(nn.Module):
    """Single-head scaled self-attention across task graph embeddings (Eq. 5--9)."""

    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.query = nn.Linear(dimension, dimension, bias=False)
        self.key = nn.Linear(dimension, dimension, bias=False)
        self.value = nn.Linear(dimension, dimension, bias=False)
        self.scale = dimension**-0.5

    def forward(self, graph_embeddings: Tensor) -> tuple[Tensor, Tensor]:
        query, key, value = self.query(graph_embeddings), self.key(graph_embeddings), self.value(graph_embeddings)
        weights = torch.softmax((query @ key.transpose(0, 1)) * self.scale, dim=-1)
        return weights @ value, weights


class EdgeDecoder(nn.Module):
    """The two-layer MLP edge classifier from Eq. (10)."""

    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.hidden = nn.Linear(3 * dimension, dimension)
        self.output = nn.Linear(dimension, 1)

    def forward(self, nodes: Tensor, fused_graph: Tensor) -> Tensor:
        count = nodes.shape[0]
        context = fused_graph.view(1, 1, -1).expand(count, count, -1)
        source = nodes.unsqueeze(1).expand(count, count, -1)
        target = nodes.unsqueeze(0).expand(count, count, -1)
        logits = self.output(F.relu(self.hidden(torch.cat((context, source, target), dim=-1)))).squeeze(-1)
        return logits.masked_fill(torch.eye(count, device=nodes.device, dtype=torch.bool), -20.0)


@dataclass(frozen=True, slots=True)
class NetworkOutput:
    edge_logits: tuple[Tensor, ...]
    attention: Tensor


class EKTGraphNetwork(nn.Module):
    """End-to-end GCN encoder, inter-task fusion, and edge heatmap decoder."""

    def __init__(self, embedding_dim: int = 128, gcn_layers: int = 4) -> None:
        super().__init__()
        self.encoder = GraphEncoder(embedding_dim, gcn_layers)
        self.fusion = KnowledgeFusion(embedding_dim)
        self.decoder = EdgeDecoder(embedding_dim)

    def forward(self, coordinate_sets: Sequence[Tensor]) -> NetworkOutput:
        if not coordinate_sets:
            raise ValueError("at least one routing task is required")
        encoded = [self.encoder(coordinates) for coordinates in coordinate_sets]
        nodes, _, graphs = zip(*encoded)
        fused, attention = self.fusion(torch.stack(graphs))
        logits = tuple(self.decoder(node, graph) for node, graph in zip(nodes, fused))
        return NetworkOutput(edge_logits=logits, attention=attention)
