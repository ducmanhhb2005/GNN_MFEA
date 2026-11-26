"""Compact storage for IDPC-DU edges with the graph indexes used by GPA."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from idpc_du.graph import Edge, Graph


_EDGE_DTYPE = np.dtype([
    ("id", np.int64), ("u", np.int32), ("v", np.int32),
    ("weight", np.float64), ("domain", np.int32),
])


@dataclass(frozen=True, slots=True)
class _EdgeTable:
    rows: NDArray
    domains: tuple[str, ...]
    domain_codes: dict[str, int]

    def edge(self, position: int) -> Edge:
        row = self.rows[position]
        return Edge(
            int(row["id"]), int(row["u"]), int(row["v"]),
            float(row["weight"]), self.domains[int(row["domain"])],
        )


@dataclass(frozen=True, slots=True)
class _EdgeSequence(Sequence[Edge]):
    table: _EdgeTable
    positions: range | NDArray

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, index: int | slice) -> Edge | _EdgeSequence:
        if isinstance(index, slice):
            return _EdgeSequence(self.table, self.positions[index])
        return self.table.edge(int(self.positions[index]))

    def __iter__(self) -> Iterator[Edge]:
        for position in self.positions:
            yield self.table.edge(position)

    def excluding_domains(self, excluded: set[str]) -> _EdgeSequence:
        codes = [self.table.domain_codes[domain] for domain in excluded]
        positions = np.asarray(self.positions)
        domains = self.table.rows["domain"][positions]
        allowed = np.ones(len(self.table.domains), dtype=np.bool_)
        allowed[codes] = False
        mask = allowed[domains]
        return _EdgeSequence(self.table, positions[mask])

    def unvisited(self, visited: set[int]) -> _EdgeSequence:
        positions = np.asarray(self.positions)
        allowed = ~np.isin(self.table.rows["v"][positions], list(visited))
        return _EdgeSequence(self.table, positions[allowed])

    def best_scored_edge(self, scores: Sequence[float]) -> tuple[float, Edge]:
        values = np.asarray(scores)
        if values.shape != (len(self),) or not np.isfinite(values).all():
            raise ValueError("Edge predictor returned invalid scores")
        best_score = values.max()
        candidates = np.asarray(self.positions)[values == best_score]
        weights = self.table.rows["weight"][candidates]
        candidates = candidates[weights == weights.min()]
        position = candidates[self.table.rows["id"][candidates].argmin()]
        return float(best_score), self.table.edge(position)

    def lighter_than(self, weight: float) -> _EdgeSequence:
        positions = np.asarray(self.positions)
        mask = self.table.rows["weight"][positions] < weight
        return _EdgeSequence(self.table, positions[mask])

    def index_of_id(self, edge_id: int) -> int:
        ids = self.table.rows["id"][self.positions]
        return int(np.flatnonzero(ids == edge_id)[0])

    def minimum_weight(self) -> float:
        return float(self.table.rows["weight"][self.positions].min())


@dataclass(frozen=True, slots=True)
class _EdgesByID(Mapping[int, Edge]):
    table: _EdgeTable
    positions: dict[int, int] | None

    def __len__(self) -> int:
        return len(self.table.rows)

    def __getitem__(self, key: int) -> Edge:
        if self.positions is not None:
            return self.table.edge(self.positions[key])
        if not 0 <= key < len(self):
            raise KeyError(key)
        return self.table.edge(key)

    def __iter__(self) -> Iterator[int]:
        if self.positions is None:
            return iter(range(len(self)))
        return iter(self.positions)


@dataclass(frozen=True, slots=True)
class CompactIDPCGraph:
    """Graph indexes backed by arrays, with edges reconstructed as needed."""

    n: int
    source: int
    target: int
    edges: _EdgeSequence
    by_id: _EdgesByID
    adj: dict[int, _EdgeSequence]
    by_neighbor: dict[int, dict[int, _EdgeSequence]]
    domains: tuple[str, ...]

    @classmethod
    def from_file(cls, path: str | Path) -> CompactIDPCGraph:
        # Reuse the baseline parser so validation and edge IDs stay identical.
        return cls.from_graph(Graph.from_file(path))

    @classmethod
    def from_graph(cls, graph: Graph) -> CompactIDPCGraph:
        domains = tuple(graph.domains)
        domain_codes = {domain: index for index, domain in enumerate(domains)}
        count = len(graph.edges)
        rows = np.fromiter(
            ((e.id, e.u, e.v, e.weight, domain_codes[e.domain]) for e in graph.edges),
            dtype=_EDGE_DTYPE,
            count=count,
        )
        rows.flags.writeable = False
        table = _EdgeTable(rows, domains, domain_codes)
        # Official files have sequential IDs, so they need no Python ID dictionary.
        positions = None
        if not all(int(edge_id) == i for i, edge_id in enumerate(rows["id"])):
            positions = {int(edge_id): i for i, edge_id in enumerate(rows["id"])}

        index_dtype = np.int32 if count <= np.iinfo(np.int32).max else np.int64
        pair_keys = rows["u"].astype(np.int64) * graph.n + rows["v"]
        adj = {
            node: _EdgeSequence(table, indices)
            for node, indices in _group_positions(rows["u"], index_dtype)
        }
        by_neighbor: dict[int, dict[int, _EdgeSequence]] = {}
        for key, indices in _group_positions(pair_keys, index_dtype):
            source, target = divmod(key, graph.n)
            by_neighbor.setdefault(source, {})[target] = _EdgeSequence(table, indices)
        return cls(
            graph.n, graph.source, graph.target,
            _EdgeSequence(table, range(count)), _EdgesByID(table, positions),
            adj, by_neighbor, domains,
        )

    @property
    def storage_bytes(self) -> int:
        """Size of the edge table and the two position indexes."""
        return self.edges.table.rows.nbytes + sum(
            view.positions.nbytes for view in self.adj.values()
        ) + sum(
            view.positions.nbytes for neighbors in self.by_neighbor.values() for view in neighbors.values()
        )


def _group_positions(keys: NDArray, dtype) -> Iterator[tuple[int, NDArray]]:
    if not len(keys):
        return
    # A stable sort preserves file order inside every parallel-edge group.
    order = np.argsort(keys, kind="stable").astype(dtype, copy=False)
    order.flags.writeable = False
    ordered_keys = keys[order]
    boundaries = np.flatnonzero(ordered_keys[1:] != ordered_keys[:-1]) + 1
    start = 0
    for end in (*boundaries, len(order)):
        yield int(ordered_keys[start]), order[start:end]
        start = end


IDPCGraph = Graph | CompactIDPCGraph
