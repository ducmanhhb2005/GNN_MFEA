"""Full-paper metrics, evaluation-budget and end-to-end artifact checks."""
import copy
import json
import math
from pathlib import Path
import random
import tempfile
import unittest

from idpc_du import Graph, Edge
from mfea.chromosome import edge_index_limits
from mfea.solver import solve_mfea, solve_ea
from experiments.metrics import comparison_metrics, normalized_convergence
from experiments.run_mfea import load_reference, main as run_experiment
from experiments._support.validate_artifacts import audit_outputs
from experiments._support.paper_report import checkpoint_rows, write_convergence

ROOT = Path(__file__).resolve().parents[1]


class TestPaperMetrics(unittest.TestCase):
    def test_reference_counts_and_maximum_match_but_mean_definition_is_unresolved(self):
        reference = load_reference()
        rows = [dict(algorithm=a, instance=name, subset=r["set"], AVG=r[a]["AVG"])
                for name, r in reference["instances"].items() for a in ("EA", "MFEA")]
        for row in comparison_metrics(rows, reference):
            self.assertEqual(row["NIB"], row["paper_NIB"])
            self.assertEqual(row["NIE"], row["paper_NIE"])
            self.assertEqual(round(row["maximum_RPD"], 1), row["paper_maximum_RPD"])
            # Printed Tables III/IV do not recover Table II's reported mean.
            # Never force a statistic to the paper's answer to make it match.
            expected = {"set1": 9.947768764819422, "set2": 4.2962248891964645}
            self.assertAlmostEqual(row["average_RPD"], expected[row["subset"]])
            self.assertNotEqual(round(row["average_RPD"], 1), row["paper_average_RPD"])
            self.assertLess(abs(row["average_RPD_improved"] - row["paper_average_RPD"]), .1)
            self.assertEqual(row["NIB"]+row["NIE"]+row["NIW"], row["paired"])

    def test_worse_and_infeasible_methods_are_not_counted_as_improvements(self):
        rows = [dict(algorithm=a, instance=t, subset="set1", AVG=v)
                for t, ea, mt in [("a", 10, 8), ("b", 10, 12), ("c", 5, 5), ("d", 10, math.inf)]
                for a, v in [("EA", ea), ("MFEA", mt)]]
        r = comparison_metrics(rows)[0]
        self.assertEqual((r["paired"], r["finite_pairs"], r["NIB"], r["NIE"], r["NIW"]), (4, 3, 1, 1, 1))
        self.assertEqual(r["average_RPD"], 0)

    def curves(self):
        return [dict(algorithm=a, instance=t, seed=0, convergence=values,
                     objective_min=lo, objective_max=hi, evaluations=100, infeasible_evaluations=0)
                for a, t, values, lo, hi in [("EA", "A", [20,10], 0,100), ("MFEA", "A", [10,0], 0,200),
                                            ("EA", "B", [30,20], 10,50), ("MFEA", "B", [20,10], 10,40)]]

    def test_normalization_uses_all_candidate_extrema_and_equal_task_weights(self):
        extrema, individual, averaged = normalized_convergence(self.curves(), 1)
        self.assertEqual(extrema["A"]["maximum"], 200)
        means = {(r["algorithm"], r["generation"]):r["normalized_mean"] for r in averaged}
        self.assertAlmostEqual(means[("EA",0)], .3)
        self.assertAlmostEqual(means[("MFEA",0)], .15)
        self.assertAlmostEqual(means[("EA",1)], .15)
        self.assertEqual(means[("MFEA",1)], 0)

    def test_infeasible_convergence_is_not_silently_dropped(self):
        rows = self.curves()
        extra = copy.deepcopy(rows)
        for row in extra:
            row["seed"] = 1
        extra[0]["convergence"][0] = None
        extra[0]["infeasible_evaluations"] = 1
        _, individual, averaged = normalized_convergence(rows+extra, 1)
        ea = next(r for r in averaged if r["algorithm"] == "EA" and r["generation"] == 0)
        self.assertIsNone(ea["normalized_mean"])
        self.assertEqual(ea["valid_runs"], 1)
        with self.assertRaises(ValueError):
            normalized_convergence(rows[:-1], 1)

    def test_constant_task_normalization_is_defined(self):
        rows = [dict(algorithm=a, instance="A", seed=0, convergence=[7,7], objective_min=7,
                     objective_max=7, evaluations=8, infeasible_evaluations=0) for a in ("EA", "MFEA")]
        extrema, _, averaged = normalized_convergence(rows, 1)
        self.assertTrue(extrema["A"]["constant"])
        self.assertTrue(all(r["normalized_mean"] == 0 for r in averaged))


class TestPaperExecution(unittest.TestCase):
    def test_index_limits_match_explicit_edge_counts_on_heterogeneous_tasks(self):
        rng = random.Random(51)
        for trial in range(12):
            tasks = []
            for n in (3, 8):
                edges = [Edge(i, rng.randrange(n), rng.randrange(n), 1., "A") for i in range(40)]
                tasks.append(Graph(n, 0, n-1, edges))
            expected = [max([sum(e.u == u and e.v == v for e in g.edges)
                             for g in tasks for v in range(g.n)] + [1]) for u in range(8)]
            self.assertEqual(edge_index_limits(tasks), expected)

    def test_diagnostics_preserve_rng_and_count_every_evaluation(self):
        graphs = [Graph.from_file(ROOT/"data"/f"demo{i}.json") for i in (1,2)]
        params = dict(seed=42, population=12, generations=9)
        expected = solve_mfea(graphs, **params)
        diagnostics = {}
        self.assertEqual(solve_mfea(graphs, diagnostics=diagnostics, **params), expected)
        self.assertEqual(sum(diagnostics["evaluations"]), 12*(2+9))
        self.assertEqual(diagnostics["objective_min"], [v for v, p in expected[0]])
        self.assertTrue(all(count >= 12 for count in diagnostics["evaluations"]))
        self.assertEqual(solve_ea(graphs[0], **params), solve_mfea([graphs[0]], **params))

    def test_both_algorithms_checkpoint_resume_audit_and_normalized_exports(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)/"experiment"
            args = ["--instances", str(ROOT/"data/demo1.json"), str(ROOT/"data/demo2.json"),
                    "--methods", "ea", "mfea", "--runs", "2", "--population", "8", "--generations", "3",
                    "--workers", "1", "--out-dir", str(output)]
            run_experiment(args)
            audit = audit_outputs([output])
            self.assertEqual(audit["path_records_checked"], 8)
            before = {p.name:p.read_bytes() for p in (output/"checkpoints").glob("*.json")}
            run_experiment(args+["--resume"])
            self.assertEqual(before, {p.name:p.read_bytes() for p in (output/"checkpoints").glob("*.json")})
            records = checkpoint_rows(output)
            self.assertEqual({r["algorithm"] for r in records}, {"EA", "MFEA"})
            meta, individual, averaged = write_convergence(output, load_reference())
            self.assertEqual(len(individual), 2*2*4)
            self.assertEqual(len(averaged), 2*4)
            self.assertTrue((output/"normalization.json").is_file())
