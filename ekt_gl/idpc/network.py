"""Directed graph encoder, cross-task fusion and original-edge scorer."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ekt_gl.network import KnowledgeFusion

from .features import GraphFeatures


class _DirectedLayer(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.node_update = nn.Linear(dimension, dimension)
        self.incoming = nn.Linear(dimension, dimension, bias=False)
        self.outgoing = nn.Linear(dimension, dimension, bias=False)
        self.gate = nn.Linear(2, dimension)
        self.norm = nn.LayerNorm(dimension)

    def forward(self, nodes: Tensor, graph: GraphFeatures, degrees: tuple[Tensor, Tensor]) -> Tensor:
        source, target = graph.sources, graph.targets
        gate = torch.sigmoid(self.gate(graph.pairs))
        incoming = torch.zeros_like(nodes).index_add(0, target, self.incoming(nodes)[source] * gate)
        outgoing = torch.zeros_like(nodes).index_add(0, source, self.outgoing(nodes)[target] * gate)
        in_degree, out_degree = degrees
        delta = self.node_update(nodes) + incoming / in_degree + outgoing / out_degree
        return nodes + torch.relu(self.norm(delta))


class IDPCGraphNetwork(nn.Module):
    def __init__(self, dimension: int, layers: int) -> None:
        super().__init__()
        self.project_nodes = nn.Linear(4, dimension)
        self.layers = nn.ModuleList(_DirectedLayer(dimension) for _ in range(layers))
        self.fusion = KnowledgeFusion(dimension)
        self.decoder = nn.Sequential(
            nn.Linear(3 * dimension + 3, dimension), nn.ReLU(), nn.Linear(dimension, 1)
        )

    def encode(self, graphs: list[GraphFeatures]) -> tuple[list[Tensor], Tensor, Tensor]:
        encoded = []
        for graph in graphs:
            nodes = self.project_nodes(graph.nodes)
            degrees = (
                torch.bincount(graph.targets, minlength=len(nodes)).clamp_min(1).unsqueeze(1),
                torch.bincount(graph.sources, minlength=len(nodes)).clamp_min(1).unsqueeze(1),
            )
            for layer in self.layers:
                nodes = layer(nodes, graph, degrees)
            encoded.append(nodes)
        fused, attention = self.fusion(torch.stack([nodes.mean(dim=0) for nodes in encoded]))
        return encoded, fused, attention

    def edge_logits(
        self, nodes: Tensor, context: Tensor, batch: tuple[Tensor, Tensor, Tensor]
    ) -> Tensor:
        source, target, attributes = batch
        inputs = torch.cat((nodes[source], nodes[target], context.expand(len(source), -1), attributes), dim=1)
        return self.decoder(inputs).squeeze(-1)
