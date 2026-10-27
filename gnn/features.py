"""Extract node and edge features for GNN input."""
from __future__ import annotations
from collections import Counter
import math
from idpc_du.graph import Graph
from .model import require_torch

def graph_features(graph: Graph):
    torch, _ = require_torch()
    n, m = graph.n, len(graph.edges)
    if m == 0:
        raise ValueError("graph must have at least one edge")
    indeg, outdeg = [0]*n, [0]*n
    in_w = [0.0]*n
    dom = Counter(e.domain for e in graph.edges)
    parallel = Counter((e.u, e.v) for e in graph.edges)
    max_w = max(1., max(e.weight for e in graph.edges))
    node_x, edge_x, src, dst = [], [], [], []
    for e in graph.edges:
        indeg[e.v] += 1
        outdeg[e.u] += 1
        in_w[e.v] += e.weight / max_w
        src.append(e.u)
        dst.append(e.v)
        edge_x.append([
            math.log1p(e.weight) / math.log1p(max_w),
            1. / parallel[(e.u,e.v)],
            dom[e.domain] / m,
            float(e.u == graph.source),
            float(e.v == graph.target),
            1. / (1. + e.weight),
        ])
    for v in range(n):
        node_x.append([
            indeg[v] / m,
            outdeg[v] / m,
            float(v == graph.source),
            float(v == graph.target),
            (in_w[v]/indeg[v]) if indeg[v] else 0.,
        ])
    return (torch.tensor(node_x, dtype=torch.float32),
            torch.tensor(edge_x, dtype=torch.float32),
            torch.tensor(src, dtype=torch.long),
            torch.tensor(dst, dtype=torch.long))
