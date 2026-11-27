"""Online graph learning with exhaustive, chunked per-edge classification."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ekt_gl.idpc.task import CompactIDPCGraph
from ekt_gl.network import KnowledgeFusion


FEATURE_POLICY = {
    "node": ["source", "target", "in_degree", "out_degree", "in_domain_count",
             "out_domain_count", "mean_in_cost", "mean_out_cost"],
    "edge": ["normalized_log_weight", "normalized_log_multiplicity", "domain_frequency"],
    "message_index": "weighted directed pairs with mean edge features, exact edge IDs in decoder",
    "normalization": "per-graph log-weight scale and feature-specific max, BatchNorm in GCN",
    "loss": "all exact edges, unweighted mean 2-class cross entropy per graph then per task",
}


@dataclass
class Features:
    nodes: torch.Tensor
    pair_u: torch.Tensor
    pair_v: torch.Tensor
    pair_features: torch.Tensor
    multiplicity: torch.Tensor
    edge_pair: torch.Tensor
    edge_features: torch.Tensor

    @classmethod
    def build(cls, graph, device):
        packed = graph if isinstance(graph, CompactIDPCGraph) else CompactIDPCGraph.from_graph(graph)
        rows = packed.edges.table.rows
        n = graph.n
        u, v, domains = rows["u"], rows["v"], rows["domain"]
        d = len(graph.domains)
        count = len(rows)
        keys, inverse, multiplicities = np.unique(
            u * n + v, return_inverse=True, return_counts=True)
        weight = np.log1p(rows["weight"])
        weight /= max(1.0, weight.max(initial=0))
        domain_frequency = np.bincount(domains, minlength=d) / max(1, count)
        pair_mult = np.log1p(multiplicities)
        pair_mult /= max(1.0, pair_mult.max(initial=0))
        edge_features = np.column_stack((weight, pair_mult[inverse], domain_frequency[domains]))
        pair_features = np.column_stack([
            np.bincount(inverse, weights=column, minlength=len(keys)) / multiplicities
            for column in edge_features.T
        ])

        def node_attributes(endpoints):
            degree = np.bincount(endpoints, minlength=n)
            domain_counts = (np.bincount(np.unique(endpoints * d + domains) // d, minlength=n)
                             if d else np.zeros(n))
            cost = np.bincount(endpoints, weights=weight, minlength=n) / np.maximum(degree, 1)
            return degree / max(1, degree.max()), domain_counts / max(1, domain_counts.max()), cost

        ins, ind, inc = node_attributes(v)
        outs, outd, outc = node_attributes(u)
        ids = np.arange(n)
        nodes = np.column_stack((ids == graph.source, ids == graph.target, ins, outs,
                                 ind, outd, inc, outc))

        def tensor(data, dtype=torch.float32):
            return torch.as_tensor(np.ascontiguousarray(data), dtype=dtype, device=device)

        return cls(tensor(nodes), tensor(keys // n, torch.long), tensor(keys % n, torch.long),
                   tensor(pair_features), tensor(multiplicities),
                   tensor(inverse, torch.long), tensor(edge_features))


class DirectedLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.node_self = nn.Linear(dim, dim, bias=False)
        self.node_in = nn.Linear(dim, dim, bias=False)
        self.node_out = nn.Linear(dim, dim, bias=False)
        self.edge_self = nn.Linear(dim, dim, bias=False)
        self.edge_in = nn.Linear(dim, dim, bias=False)
        self.edge_out = nn.Linear(dim, dim, bias=False)
        self.node_norm = nn.BatchNorm1d(dim)
        self.edge_norm = nn.BatchNorm1d(dim)

    @staticmethod
    def normalize(norm, values):
        if values.shape[0] > 1:
            return norm(values)
        # A singleton message uses running statistics; empty edge lists stay empty.
        if values.shape[0] == 1:
            return F.batch_norm(values, norm.running_mean, norm.running_var, norm.weight,
                                norm.bias, training=False, eps=norm.eps)
        return values

    def forward(self, nodes, edges, u, v, mult, in_degree, out_degree):
        gates = edges.sigmoid() * mult
        incoming = torch.zeros_like(nodes).index_add(0, v, self.node_in(nodes)[u] * gates)
        outgoing = torch.zeros_like(nodes).index_add(0, u, self.node_out(nodes)[v] * gates)
        dn = self.node_self(nodes) + incoming / in_degree + outgoing / out_degree
        de = self.edge_self(edges) + self.edge_in(nodes)[u] + self.edge_out(nodes)[v]
        return nodes + self.normalize(self.node_norm, dn).relu(), edges + self.normalize(self.edge_norm, de).relu()


class EdgeNetwork(nn.Module):
    def __init__(self, dim=128, layers=4):
        super().__init__()
        self.nodes = nn.Linear(8, dim)
        self.pairs = nn.Linear(3, dim)
        self.exact_edge = nn.Linear(3, dim, bias=False)
        self.layers = nn.ModuleList(DirectedLayer(dim) for _ in range(layers))
        self.fusion = KnowledgeFusion(dim)
        self.first = nn.Linear(4 * dim, dim)
        self.last = nn.Linear(dim, 2)

    def contexts(self, graphs):
        node_labels = torch.cat([features.nodes for features in graphs])
        pair_labels = torch.cat([features.pair_features for features in graphs])
        node_counts = [len(features.nodes) for features in graphs]
        pair_counts = [len(features.pair_u) for features in graphs]
        offsets = np.cumsum([0] + node_counts[:-1]).tolist()
        u = torch.cat([f.pair_u + offset for f, offset in zip(graphs, offsets)])
        v = torch.cat([f.pair_v + offset for f, offset in zip(graphs, offsets)])
        mult = torch.cat([features.multiplicity for features in graphs]).unsqueeze(1)
        in_degree = torch.zeros_like(node_labels[:, :1]).index_add(0, v, mult).clamp_min(1)
        out_degree = torch.zeros_like(in_degree).index_add(0, u, mult).clamp_min(1)
        nodes, edges = self.nodes(node_labels), self.pairs(pair_labels)
        for layer in self.layers:
            nodes, edges = layer(nodes, edges, u, v, mult, in_degree, out_degree)
        pooled = torch.stack([chunk.mean(0) for chunk in nodes.split(node_counts)])
        fused, attention = self.fusion(pooled)
        w_graph, w_u, w_v, w_edge = self.first.weight.chunk(4, dim=1)
        # Algebraic factorization of Linear(concat(graph,u,v,h_e)). Compute
        # graph/node/pair terms once, and the raw edge features for every ID.
        per_pair = (F.linear(nodes, w_u)[u] + F.linear(nodes, w_v)[v] +
                    F.linear(edges, w_edge) + self.first.bias)
        bases = [base + context for base, context in zip(
            per_pair.split(pair_counts), F.linear(fused, w_graph))]
        edge_weights = w_edge @ self.exact_edge.weight
        return bases, edge_weights, attention

    @staticmethod
    def logits(base, feature_weight, pair, features, last_weight, last_bias):
        hidden = base[pair] + F.linear(features, feature_weight)
        return F.linear(hidden.relu(), last_weight, last_bias)


def elite_labels(population, count):
    valid = [member.solution.path for member in population if member.solution.feasible][:count]
    positives = {edge for path in valid for edge in path}
    return positives, len(valid), len(valid) - len(set(valid))


class OnlineLearning:
    def __init__(self, graphs, features, config):
        self.graphs = graphs
        self.features = features
        self.config = config
        self.model = EdgeNetwork(config.hidden_dim, config.gcn_layers).to(config.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.steps = 0

    def _sync(self):
        if self.config.device.startswith("cuda"):
            torch.cuda.synchronize(self.config.device)

    def labels(self, populations):
        statistics = [elite_labels(pop, self.config.elite_count) for pop in populations]
        return statistics

    def fit(self, labels):
        active = sum(feasible > 0 for _, feasible, _ in labels)
        if not active:
            return None, 0.0
        self._sync()
        start = perf_counter()
        self.model.train()
        for _ in range(self.config.train_epochs):
            self.optimizer.zero_grad(set_to_none=True)
            bases, weight, _ = self.model.contexts(self.features)
            original = [*bases, weight, self.model.last.weight, self.model.last.bias]
            # Accumulate exact feature/decoder gradients for each edge chunk
            # without retaining each chunk's tape. Apply the chain rule through
            # all node/pair embeddings and attention once after the edge pass.
            leaves = [item.detach().requires_grad_() for item in original]
            feature_w, last_w, last_b = leaves[-3:]
            loss_value = torch.zeros((), dtype=feature_w.dtype, device=feature_w.device)
            for graph, features, base, (positive, feasible, _) in zip(
                    self.graphs, self.features, leaves, labels):
                if not feasible:
                    continue
                n = len(features.edge_pair)
                targets = torch.zeros(n, dtype=torch.long, device=feature_w.device)
                positions = graph.by_id.positions
                ids = sorted(positive) if positions is None else sorted(positions[i] for i in positive)
                targets[ids] = 1
                for offset in range(0, n, self.config.edge_batch_size):
                    batch = slice(offset, offset + self.config.edge_batch_size)
                    logits = self.model.logits(base, feature_w, features.edge_pair[batch],
                                               features.edge_features[batch], last_w, last_b)
                    loss = F.cross_entropy(logits, targets[batch], reduction="sum") / (n * active)
                    loss.backward()
                    loss_value += loss.detach()
            gradients = [(value, leaf.grad) for value, leaf in zip(original, leaves) if leaf.grad is not None]
            torch.autograd.backward(*(list(values) for values in zip(*gradients)))
            self.optimizer.step()
            self.steps += 1
        self._sync()
        return float(loss_value), perf_counter() - start

    @torch.no_grad()
    def scores(self):
        self._sync()
        start = perf_counter()
        self.model.eval()
        bases, weight, attention = self.model.contexts(self.features)
        scores = []
        for base, features in zip(bases, self.features):
            n = len(features.edge_pair)
            result = torch.empty(n, device=weight.device)
            for offset in range(0, n, self.config.edge_batch_size):
                batch = slice(offset, offset + self.config.edge_batch_size)
                logits = self.model.logits(base, weight, features.edge_pair[batch],
                                           features.edge_features[batch],
                                           self.model.last.weight, self.model.last.bias)
                result[batch] = logits.softmax(1)[:, 1]
            scores.append(result.cpu().numpy())
        self._sync()
        return scores, attention.cpu().numpy().tolist(), perf_counter() - start


def greedy(graph, scores, counters):
    current = graph.source
    visited = np.zeros(graph.n, dtype=bool)
    visited[current] = True
    closed = np.zeros(len(graph.domains), dtype=bool)
    active = None
    path = []
    while current != graph.target:
        outgoing = graph.adj.get(current)
        if not outgoing:
            return None
        positions = outgoing.positions
        table = outgoing.table.rows
        counters["greedy_edge_checks"] += len(positions)
        allowed = positions[~visited[table["v"][positions]] & ~closed[table["domain"][positions]]]
        if not len(allowed):
            return None
        values = scores[allowed]
        if not np.isfinite(values).all():
            raise ValueError("Non-finite transfer score")
        best = allowed[values == values.max()]
        best = best[table["id"][best].argmin()]
        edge = table[best]
        domain = int(edge["domain"])
        if active is not None and active != domain:
            closed[active] = True
        active = domain
        current = int(edge["v"])
        visited[current] = True
        path.append(int(edge["id"]))
    return tuple(path)
