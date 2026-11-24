"""Checks for TSP benchmark loading, real solver output and result provenance."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from ekt_gl import CVRPTask, EKTGLConfig, TSPTask
from ekt_gl.routing import order_crossover, two_opt
from run_tsp import BENCHMARKS, PAPER, ROOT, holm_adjust, load_tasks, parse_tsp, run_seed, validate_record


class TestTSPExperiment(unittest.TestCase):
    def test_standard_ox_preserves_identical_parents(self):
        rng = np.random.default_rng(42)
        for size in (3, 9, 100):
            route = tuple(rng.permutation(size).tolist())
            for _ in range(30):
                self.assertEqual(order_crossover(route, route, rng), route)

    def test_ox_cyclic_order_and_legacy_cvrp_scan(self):
        first = tuple(range(10))
        second = tuple(reversed(first))
        # seed=0 selects cuts 6..7 inclusive. Fill 8,9,0,...,5 from
        # second's cyclic scan starting at index 8 (then skip retained 6,7).
        self.assertEqual(order_crossover(first, second, np.random.default_rng(0)),
                         (9, 8, 5, 4, 3, 2, 6, 7, 1, 0))
        self.assertEqual(order_crossover(first, first, np.random.default_rng(0), cyclic_scan=False),
                         (2, 3, 4, 5, 8, 9, 6, 7, 0, 1))
    def test_tsp_preserves_aspect_ratio_and_cvrp_normalization(self):
        xy = np.array([[10, 20], [30, 20], [30, 30]], dtype=np.float32)
        tsp = TSPTask("rectangle", xy)
        np.testing.assert_allclose(tsp._distance[0, 1], 1.)
        np.testing.assert_allclose(tsp._distance[1, 2], .5)
        vrp = CVRPTask("rectangle", xy, np.array([0, 1, 1]), 2)
        np.testing.assert_allclose(vrp._distance[0, 1], 1.)
        np.testing.assert_allclose(vrp._distance[1, 2], 1.)

    def test_normalized_optimum_scale_matches_all_paper_lkh_references(self):
        for benchmark in BENCHMARKS:
            tasks, inputs = load_tasks(benchmark, ROOT / "tsp_data")
            for index, task in enumerate(tasks):
                value = inputs[task.name]["normalized_optimum_scale_check"]
                self.assertEqual(round(value, 2), PAPER[benchmark]["lkh"][index])

    def test_accelerated_two_opt_matches_full_objective_scan(self):
        def reference(task, route, passes):
            best, best_cost = route, task.evaluate(route)
            for _ in range(passes):
                improved = False
                for left in range(len(best) - 1):
                    for right in range(left + 2, len(best) + 1):
                        candidate = best[:left] + tuple(reversed(best[left:right])) + best[right:]
                        cost = task.evaluate(candidate)
                        if cost + 1e-12 < best_cost:
                            best, best_cost, improved = candidate, cost, True
                            break
                    if improved:
                        break
                if not improved:
                    break
            return best

        rng = np.random.default_rng(123)
        for size in (3, 4, 9, 20, 76, 150, 299):
            for shape in ("random", "duplicates", "circle"):
                xy = rng.random((size, 2))
                if shape == "duplicates":
                    xy = rng.integers(0, 3, size=(size, 2))
                if shape == "circle":
                    angles = np.arange(size) * 2 * np.pi / size
                    xy = np.column_stack([np.cos(angles), np.sin(angles)])
                task = TSPTask(shape, xy)
                route = tuple(rng.permutation(size).tolist()) if shape != "circle" else tuple(range(size))
                for passes in (1, 3):
                    self.assertEqual(two_opt(task, route, passes), reference(task, route, passes))

    def test_tsplib_parser_reorders_ids_and_rejects_missing_nodes(self):
        text = ("NAME : tiny\nDIMENSION : 3\nEDGE_WEIGHT_TYPE : EUC_2D\n"
                "NODE_COORD_SECTION\n3 0 1\n1 0 0\n2 1 0\nEOF\n")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.tsp"
            path.write_text(text)
            np.testing.assert_array_equal(parse_tsp(path), [[0, 0], [1, 0], [0, 1]])
            path.write_text(text.replace("3 0 1\n", ""))
            with self.assertRaises(ValueError):
                parse_tsp(path)

    def test_closed_tour_and_holm(self):
        square = TSPTask("square", np.array([[0, 0], [10, 0], [10, 10], [0, 10]]))
        self.assertAlmostEqual(square.evaluate((0, 1, 2, 3)), 4.)
        np.testing.assert_allclose(holm_adjust([.04, .001, .03]), [.06, .003, .06])

    def test_real_solver_cache_and_output_validation(self):
        tasks = [TSPTask("small", np.array([[0, 0], [2, 0], [2, 1], [0, 1]])),
                 TSPTask("other", np.random.default_rng(8).random((6, 2)))]
        config = EKTGLConfig(generations=3, population_size=4, transfer_interval=2,
                             embedding_dim=8, gcn_layers=1)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            for mode in ("ektgl", "sto"):
                record = run_seed(tasks, {}, "tiny", mode, 7, config, output)
                again = run_seed(tasks, {}, "tiny", mode, 7, config, output)
                self.assertEqual(json.loads(json.dumps(record)), again)
                validate_record(record, tasks, config)
                self.assertEqual(len(record["transfers"]), 2 if mode == "ektgl" else 0)
                # Independently rerun the same seed, rather than just hitting cache.
                repeat = run_seed(tasks, {}, "tiny", mode, 7, config, output / "repeat")
                self.assertEqual(record["best"], repeat["best"])
                self.assertEqual(record["history"], repeat["history"])
                record["best"]["small"]["cost"] += 1
                with self.assertRaises(ValueError):
                    validate_record(record, tasks, config)
            self.assertEqual(len(list((output / "raw").glob("*.json"))), 2)


if __name__ == "__main__":
    unittest.main()
