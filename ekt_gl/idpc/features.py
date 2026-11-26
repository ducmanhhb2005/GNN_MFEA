"""Features for a directed multigraph without dense per-edge embeddings."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import Tensor

from idpc_du.graph import Edge

from .task import CompactIDPCGraph, IDPCGraph, _EdgeSequence


@dataclass(frozen=True, slots=True)
class GraphFeatures:
    nodes: Tensor
    sources: Tensor
    targets: Tensor
    pairs: Tensor
    weight_scale: float
    domain_attributes: dict[str, tuple[float, float]]
    domain_matrix: np.ndarray

    @classmethod
    def from_graph(cls, graph: IDPCGraph, device: str) -> GraphFeatures:
        if not isinstance(graph, CompactIDPCGraph):
            graph = CompactIDPCGraph.from_graph(graph)
        rows = graph.edges.table.rows
        weights = rows["weight"]
        weights_at_log_scale = np.log1p(weights)
        incoming = np.bincount(rows["v"], minlength=graph.n)
        sources, targets, pairs = [], [], []
        domain_counts = np.bincount(rows["domain"], minlength=len(graph.domains))
        weight_scale = float(weights_at_log_scale.max(initial=0.0)) or 1.0
        domain_weights = np.bincount(
            rows["domain"], weights=weights_at_log_scale / weight_scale, minlength=len(graph.domains)
        )
        edge_count = max(len(graph.edges), 1)
        degree_scale = math.log1p(edge_count)
        for source, neighbors in graph.by_neighbor.items():
            for target, edges in neighbors.items():
                sources.append(source)
                targets.append(target)
                pairs.append((
                    math.log1p(len(edges)) / degree_scale,
                    math.log1p(edges.minimum_weight()) / weight_scale,
                ))
        nodes = [
            (
                float(vertex == graph.source),
                float(vertex == graph.target),
                math.log1p(incoming[vertex]) / degree_scale,
                math.log1p(len(graph.adj.get(vertex, ()))) / degree_scale,
            )
            for vertex in range(graph.n)
        ]
        domain_matrix = np.array([
            (domain_counts[index] / edge_count, domain_weights[index] / domain_counts[index])
            for index in range(len(graph.domains))
        ], dtype=np.float32).reshape(-1, 2)
        return cls(
            torch.tensor(nodes, dtype=torch.float32, device=device),
            torch.tensor(sources, dtype=torch.long, device=device),
            torch.tensor(targets, dtype=torch.long, device=device),
            torch.tensor(pairs, dtype=torch.float32, device=device).reshape(-1, 2),
            weight_scale,
            {domain: tuple(map(float, row)) for domain, row in zip(graph.domains, domain_matrix)},
            domain_matrix,
        )

    def edge_inputs(self, edges: Sequence[Edge]) -> tuple[Tensor, Tensor, Tensor]:
        """Build one feature row per original edge in the supplied order."""
        device = self.nodes.device
        if isinstance(edges, _EdgeSequence):
            rows = edges.table.rows[edges.positions]
            features = np.column_stack((
                np.log1p(rows["weight"]) / self.weight_scale, self.domain_matrix[rows["domain"]]
            )).astype(np.float32)
            return (
                torch.as_tensor(rows["u"].copy(), dtype=torch.long, device=device),
                torch.as_tensor(rows["v"].copy(), dtype=torch.long, device=device),
                torch.as_tensor(features, dtype=torch.float32, device=device),
            )
        return (
            torch.tensor([edge.u for edge in edges], dtype=torch.long, device=device),
            torch.tensor([edge.v for edge in edges], dtype=torch.long, device=device),
            torch.tensor([
                (math.log1p(edge.weight) / self.weight_scale, *self.domain_attributes[edge.domain])
                for edge in edges
            ], dtype=torch.float32, device=device).reshape(-1, 3),
        )
