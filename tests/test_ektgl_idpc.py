"""Regression tests for the IDPC-DU specialization of EKT-GL."""

import math
from pathlib import Path
import random
import unittest

import torch

from ekt_gl.idpc import CompactIDPCGraph, IDPCEKTGL, IDPCEKTGLConfig, IDPCSolution
from ekt_gl.idpc.decoding import encode_path, greedy_path
from ekt_gl.idpc.local_search import refine_path
from ekt_gl.idpc.population import IDPCPopulation
from ekt_gl.idpc.training import elite_edge_labels
from idpc_du.gpa import decode_gpa
from idpc_du.graph import Edge, Graph
from idpc_du.validation import validate_path
from mfea.chromosome import edge_index_limits, init_chrom


ROOT = Path(__file__).resolve().parents[1]


class TestIDPCTransfer(unittest.TestCase):
    def test_greedy_and_refinement_respect_direction_and_domain(self):
        graph = Graph(5, 0, 4, [
            Edge(12, 0, 1, 1, "B"), Edge(11, 0, 1, 2, "A"), Edge(10, 0, 1, 4, "A"),
            Edge(20, 1, 2, 1, "A"), Edge(21, 1, 0, 0, "B"),
            Edge(30, 2, 3, 1, "B"), Edge(31, 3, 4, 0, "A"), Edge(32, 3, 4, 1, "B"),
        ])
        scores = {10: 5, 21: 10, 31: 10}
        path = greedy_path(graph, lambda edges: [scores.get(edge.id, 0) for edge in edges], 2)
        self.assertEqual(path, (10, 20, 30, 32))
        refined, checks = refine_path(graph, path, 1)
        self.assertEqual(refined, (11, 20, 30, 32))
        self.assertGreater(checks, 1)
        self.assertEqual(validate_path(graph, refined), 5)
        chromosome = encode_path(graph, refined, edge_index_limits([graph]))
        self.assertEqual(decode_gpa(graph, chromosome), (5, list(refined)))
        self.assertEqual(
            elite_edge_labels([IDPCSolution(7, path), IDPCSolution(5, refined)]),
            {10, 11, 20, 30, 32},
        )

    def test_valid_but_unrepresentable_candidate_is_rejected(self):
        graph = Graph(2, 0, 1, [Edge(10, 0, 1, 1, "A"), Edge(11, 0, 1, 2, "B")])
        population = IDPCPopulation(graph, 5, random.Random(10))
        self.assertEqual(validate_path(graph, (10,)), 1)
        self.assertEqual(population.inject((10,)), "unrepresentable")
        self.assertEqual(population.inject((11,)), "duplicate")
        self.assertEqual(population.best, IDPCSolution(2, (11,)))
        population.evolve(0.05)
        self.assertEqual(len(population._members), 5)

    def test_real_online_learning_and_separate_tasks(self):
        graphs = {name: Graph.from_file(ROOT / "data" / (name + ".json"))
                  for name in ("demo1", "demo2")}
        graphs["empty"] = Graph(2, 0, 1, [])
        config = IDPCEKTGLConfig(population_size=6, generations=5, transfer_interval=2)
        solver = IDPCEKTGL(config)
        result = solver.solve(graphs, seed=7)
        self.assertEqual(result, solver.solve(graphs, seed=7))
        self.assertEqual([r.generation for r in result.transfers], [0, 2, 4])
        for report in result.transfers:
            self.assertTrue(math.isfinite(report.loss))
            self.assertEqual(report.labelled_tasks, ("demo1", "demo2"))
            self.assertEqual(report.outcomes["empty"], "dead_end")
            self.assertEqual(len(report.attention), 3)
            for row in report.attention:
                self.assertEqual(len(row), 3)
                self.assertAlmostEqual(sum(row), 1.0, places=6)
        for name, solution in result.best.items():
            history = result.history[name]
            self.assertEqual(len(history), 6)
            self.assertTrue(all(a >= b for a, b in zip(history, history[1:])))
            self.assertEqual(history[-1], solution.cost)
            self.assertEqual(solution.cost, validate_path(graphs[name], solution.edge_ids))
        self.assertTrue(math.isfinite(result.best["demo1"].cost))
        self.assertTrue(math.isfinite(result.best["demo2"].cost))
        self.assertEqual(result.evaluations["empty"], 18)

    def test_no_elite_falls_back_to_evolution(self):
        config = IDPCEKTGLConfig(population_size=4, generations=2, transfer_interval=1)
        result = IDPCEKTGL(config).solve({"empty": Graph(2, 0, 1, [])})
        self.assertEqual(result.evaluations, {"empty": 12})
        for report in result.transfers:
            self.assertIsNone(report.loss)
            self.assertEqual(report.outcomes, {"empty": "no_elites"})

    def test_compact_graph_preserves_original_ids_and_gpa_order(self):
        graph = Graph(4, 0, 3, [
            Edge(92, 0, 2, 9, "B"), Edge(81, 0, 1, 2, "A"),
            Edge(17, 0, 2, 3, "A"), Edge(31, 2, 3, 1, "A"),
            Edge(43, 1, 2, 1, "B"), Edge(26, 2, 0, 0, "A"),
        ])
        packed = CompactIDPCGraph.from_graph(graph)
        self.assertEqual(list(packed.edges), graph.edges)
        self.assertEqual(list(packed.adj[0]), graph.adj[0])
        self.assertEqual(list(packed.by_neighbor[0][2]), graph.by_neighbor[0][2])
        self.assertEqual(packed.by_id[17], graph.by_id[17])
        self.assertIsNone(packed.by_id.get(1))
        rng = random.Random(5)
        limits = edge_index_limits([graph])
        self.assertEqual(edge_index_limits([packed]), limits)
        for _ in range(30):
            chromosome = init_chrom(graph.n, limits, rng)
            self.assertEqual(decode_gpa(packed, chromosome), decode_gpa(graph, chromosome))

    def test_sto_uses_the_same_local_search(self):
        graph = CompactIDPCGraph.from_graph(Graph(3, 0, 2, [
            Edge(0, 0, 1, 3, "A"), Edge(1, 1, 2, 3, "A"), Edge(2, 0, 2, 1, "A"),
        ]))
        config = IDPCEKTGLConfig(population_size=4, generations=3, transfer_interval=2)
        solver = IDPCEKTGL(config)
        result = solver.solve({"shortcuts": graph}, seed=8, transfer=False)
        self.assertEqual(result.transfers, ())
        self.assertGreaterEqual(result.local_search_checks["shortcuts"], 12)
        self.assertEqual(result.best["shortcuts"], IDPCSolution(1, (2,)))
        with_transfer = solver.solve({"shortcuts": graph}, seed=8)
        self.assertEqual(result.history["shortcuts"][0], with_transfer.history["shortcuts"][0])

    @classmethod
    def setUpClass(cls):
        old_threads = torch.get_num_threads()
        cls.addClassCleanup(torch.set_num_threads, old_threads)
        torch.set_num_threads(1)
