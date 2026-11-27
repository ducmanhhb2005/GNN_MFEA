"""Optional compiled GPA traversal, equivalent to the Python implementation."""

import numpy as np
from numba import njit

from idpc_du.validation import validate_path


@njit
def _grow(n, domain_count, source, target, offsets, positions, domains, edge_ids, pi, sigma):
    path = np.empty(n - 1, np.int64)
    order = np.full(n, -1, np.int32)
    if len(pi) != n or len(sigma) < n:
        return path[:0], False
    for i in range(n):
        key = pi[i] - 1
        if key < 0 or key >= n or order[key] != -1:
            return path[:0], False
        order[key] = i
    visited = np.zeros(n, np.bool_)
    closed = np.zeros(domain_count, np.bool_)
    current, active, used, closed_count = source, -1, 0, 0
    visited[current] = True
    while current != target:
        selected = -1
        for rank in range(n - 1, -1, -1):
            v = order[rank]
            if visited[v]:
                continue
            pair = current * n + v
            lo, hi = offsets[pair], offsets[pair + 1]
            if lo == hi:
                continue
            selector = 1 if sigma[v] == -1 else sigma[v]
            if closed_count == 0:
                selected = positions[lo + selector % (hi - lo)]
            else:
                eligible = 0
                for p in range(lo, hi):
                    if not closed[domains[positions[p]]]:
                        eligible += 1
                if eligible == 0:
                    continue
                index = selector % eligible
                for p in range(lo, hi):
                    row = positions[p]
                    if not closed[domains[row]]:
                        if index == 0:
                            selected = row
                            break
                        index -= 1
            if selected >= 0:
                current = v
                break
        if selected < 0:
            return path[:0], False
        new_domain = domains[selected]
        if active >= 0 and new_domain != active:
            closed[active] = True
            closed_count += 1
        active = new_domain
        visited[current] = True
        path[used] = edge_ids[selected]
        used += 1
    return path[:used], True


class PackedGPA:
    """Index the provided directed edge views without changing their file order."""

    def __init__(self, graph):
        self.graph = graph
        rows = graph.edges.table.rows
        n = graph.n
        lengths = np.zeros(n * n + 1, dtype=np.int64)
        pairs = []
        for u in range(n):
            for v, edges in sorted(graph.by_neighbor.get(u, {}).items()):
                lengths[u * n + v + 1] = len(edges)
                pairs.append(np.asarray(edges.positions, dtype=np.int32))
        self.offsets = np.cumsum(lengths)
        self.positions = np.concatenate(pairs) if pairs else np.empty(0, dtype=np.int32)
        self.domains = np.ascontiguousarray(rows["domain"])
        self.edge_ids = np.ascontiguousarray(rows["id"])

    def __call__(self, chromosome):
        path, ok = _grow(
            self.graph.n, len(self.graph.domains), self.graph.source, self.graph.target,
            self.offsets, self.positions, self.domains, self.edge_ids,
            np.asarray(chromosome.priority, dtype=np.int32),
            np.asarray(chromosome.edge_index, dtype=np.int32),
        )
        if not ok:
            return float("inf"), []
        edge_ids = path.tolist()
        return validate_path(self.graph, edge_ids), edge_ids


def warmup():
    """Compile traversal outside timed solver runs, on a two-node empty graph."""
    _grow(2, 0, 0, 1, np.zeros(5, dtype=np.int64), np.zeros(0, dtype=np.int32),
          np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int64),
          np.array([1, 2], dtype=np.int32), np.array([-1, -1], dtype=np.int32))
