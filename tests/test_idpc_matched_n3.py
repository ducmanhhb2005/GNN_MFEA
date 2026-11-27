"""N3 integration regressions without regenerating the V3 reference benchmark."""
from collections import Counter
from contextlib import closing
from pathlib import Path
import random
import tempfile
import unittest

from ekt_gl.idpc.research.engine import ResearchEvolution, Variant
from ekt_gl.idpc.research.learning import feature_set
from ekt_gl.idpc.task import CompactIDPCGraph
from ekt_gl.idpc.v2.config import Config
from ekt_gl.idpc.v2.core import EvaluationTracker, Solution, project
from ekt_gl.idpc.v2.evolution import Evolution
from ekt_gl.idpc.v2.local_search import DirectedPairs
from experiments._support.idpc_v3_archive import independently_validate
from experiments.run_idpc_matched_n3 import EventCheckpoint, n3_observation, v3
from idpc_du.graph import Graph
from mfea.chromosome import Chromosome
from test_idpc_v3 import fixture


class MatchedN3Tests(unittest.TestCase):
    def test_framework_initialization_preserves_archive_and_budget(self):
        graph = fixture()
        graphs = [graph, CompactIDPCGraph.from_graph(Graph(6, 0, 3, list(graph.edges)))]
        config = Config(profile="matched_ls", population_per_task=6)
        for method in ("ea", "mfea", "ektgl"):
            with self.subTest(method=method):
                origin = Evolution(method, graphs, ("a", "b"), config, 4200, EvaluationTracker(graphs))
                adapted = ResearchEvolution(method, graphs, ("a", "b"), config, 4200, EvaluationTracker(graphs),
                                            variant=Variant(ls="subpath"), diagnostic=False)
                origin.initialize()
                adapted.initialize()
                self.assertEqual(origin.snapshot(), adapted.snapshot())
                self.assertEqual(adapted.tracker.requests, 24 if method == "mfea" else 12)

    def test_n3_exact_lift_keeps_unified_slots_and_skill(self):
        graph = fixture()
        larger = CompactIDPCGraph.from_graph(Graph(6, 0, 3, list(graph.edges)))
        tracker = EvaluationTracker([graph, larger], budget=200)
        config = Config(profile="matched_ls", population_per_task=6)
        with n3_observation():
            method = v3.ResearchEvolution("mfea", [graph, larger], ("a", "b"), config, 4200, tracker,
                                          variant=Variant(ls="subpath"), diagnostic=False)
            original = Chromosome([5, 4, 1, 6, 3, 2], [1] * 6, skill=0)
            initial = tracker.roundtrip(0, Solution(19, (7,)), method.local_limits[0], "offspring", unified=original)
            improved = method.improve(0, initial, DirectedPairs(graph), random.Random(1), "ordinary")
            self.assertLess(improved.solution.cost, initial.solution.cost)
            self.assertEqual(improved.solution.cost, independently_validate(graph, improved.solution.path))
            self.assertEqual(improved.chromosome.skill, 0)
            self.assertEqual(improved.chromosome.edge_index[4:], original.edge_index[4:])
            for i, value in enumerate(original.priority):
                if value > 4:
                    self.assertEqual(improved.chromosome.priority[i], value)
            decoded = tracker.gpa(0, project(improved.chromosome, 4), "other")
            self.assertEqual(decoded, improved.solution)
            self.assertGreater(tracker.per_task[0]["ordinary_ls_evaluations"], 0)
            tracker.check_invariants()

    def test_n3_checks_resume_and_phase_work(self):
        graph = fixture()
        graphs = [graph, graph]
        config = Config(profile="matched_ls", population_per_task=6, generations=12,
                        hidden_dim=8, gcn_layers=1, budget_override=1900, threads=1)
        v3.torch.set_num_threads(1)
        features = [feature_set(g, "cpu", "v2") for g in graphs]
        with tempfile.TemporaryDirectory() as folder:
            for method in ("ea", "mfea", "ektgl"):
                with self.subTest(method=method):
                    path = Path(folder) / f"{method}.sqlite"
                    identity = ("tiny", method, 4399)
                    args = (method, 4399, graphs, ("a", "b"), config, "tiny",
                            features if method == "ektgl" else None, None, Variant(ls="subpath"))
                    with closing(EventCheckpoint(path, identity)) as checkpoint, n3_observation():
                        self.assertIsNone(v3.run_research(checkpoint, *args, pause_after=5))
                    with closing(EventCheckpoint(path, identity)) as checkpoint, n3_observation():
                        result = v3.run_research(checkpoint, *args)
                        traces = list(checkpoint.generations())
                    counts = result["counters"]
                    self.assertLessEqual(counts["objective_evaluation_requests"], result["evaluation_budget"])
                    self.assertEqual(counts["objective_evaluation_requests"],
                                     counts.get("cache_hits", 0) + counts["actual_objective_computations"])
                    self.assertEqual(traces[-1]["suite_evaluations"], counts["objective_evaluation_requests"])
                    event_total = Counter()
                    for entry in traces:
                        for event in entry["n3_events"]:
                            event_total["calls"] += event["ls_calls"]
                            event_total["requests"] += event["ls_evaluations"]
                    self.assertGreater(event_total["calls"], 0)
                    self.assertEqual(event_total["calls"], counts["ls_calls"])
                    self.assertEqual(event_total["requests"], counts["ls_evaluations"])
                    if method == "ektgl":
                        self.assertEqual(result["transfer_events"], 1)
                        self.assertTrue(any(entry["transfer"] for entry in traces))
                    else:
                        self.assertEqual(result["transfer_events"], 0)


if __name__ == "__main__":
    unittest.main()
