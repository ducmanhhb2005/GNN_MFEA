"""Compare the optimized GPA against a direct transcription of Algorithm 1."""
import math
import random
import unittest
from idpc_du import Graph, Edge, validate_path
from idpc_du.gpa import decode_gpa
from mfea.chromosome import init_chrom, edge_index_limits


def algorithm_1(graph, chromosome):
    pi = [p for p in chromosome.priority if p <= graph.n]
    current = graph.source
    visited = set()
    exited = set()
    domain = None
    path = []
    while current != graph.target:
        visited.add(current)
        adjacent = [e for e in graph.adj[current]
                    if e.v not in visited and e.domain not in exited]
        if not adjacent:
            return math.inf, []
        vertex = max({e.v for e in adjacent}, key=lambda v: pi[v])
        parallel = [e for e in adjacent if e.v == vertex]
        edge = parallel[chromosome.edge_index[vertex] % len(parallel)]
        path.append(edge.id)
        if domain is not None and edge.domain != domain:
            exited.add(domain)
        domain = edge.domain
        current = vertex
    return validate_path(graph, path), path


class TestGPAOptimization(unittest.TestCase):
    def test_sparse_dense_and_domain_filtering_match_algorithm_1(self):
        rng = random.Random(992)
        for density in [.08, .3, .9]:
            for trial in range(10):
                n = 12
                edges = []
                for u in range(n):
                    for v in range(n):
                        if u == v or rng.random() >= density:
                            continue
                        for k in range(rng.randrange(1, 4)):
                            edges.append(Edge(len(edges), u, v,
                                              float(rng.randrange(15)), str(rng.randrange(4))))
                graph = Graph(n, 0, n-1, edges)
                limits = edge_index_limits([graph])
                for k in range(20):
                    chromosome = init_chrom(n, limits, rng)
                    self.assertEqual(decode_gpa(graph, chromosome), algorithm_1(graph, chromosome))
