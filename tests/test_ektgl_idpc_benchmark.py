"""Exercise the benchmark runner, resume validation and budget comparisons."""

import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

from experiments.run_ektgl_idpc import METHODS, main


ROOT = Path(__file__).resolve().parents[1]


class TestIDPCBenchmark(unittest.TestCase):
    def test_reports_and_verified_resume_for_all_methods(self):
        with TemporaryDirectory() as directory:
            output = Path(directory)
            argv = [
                "--instances", str(ROOT / "data/demo1.json"), str(ROOT / "data/demo2.json"),
                "--out-dir", directory, "--seed", "100", "--runs", "2",
                "--population", "4", "--generations", "3", "--transfer-interval", "2",
                "--mfea-population", "8", "--mfea-generations", "3",
            ]
            threads = torch.get_num_threads()
            self.addCleanup(torch.set_num_threads, threads)
            main(argv)
            files = sorted((output / "runs").glob("*.json"))
            self.assertEqual(len(files), 2 * len(METHODS))
            records = [json.loads(path.read_bytes()) for path in files]
            for seed in (100, 101):
                methods = {r["method"]: r for r in records if r["seed"] == seed}
                for name in ("demo1", "demo2"):
                    starts = {methods[m]["tasks"][name]["history"][0]
                              for m in ("ektgl", "sto", "individual")}
                    self.assertEqual(len(starts), 1)
                self.assertEqual(len(methods["individual"]["transfers"]), 4)
                self.assertTrue(all(len(report["outcomes"]) == 1
                                    for report in methods["individual"]["transfers"]))
                self.assertFalse(methods["sto"]["transfers"])
            with (output / "summary.csv").open(newline="", encoding="utf-8") as stream:
                summary = list(csv.DictReader(stream))
            self.assertEqual(len(summary), 8)
            self.assertTrue(all(row["runs"] == row["feasible_runs"] == "2" for row in summary))
            with (output / "matched_budgets.csv").open(newline="", encoding="utf-8") as stream:
                comparisons = list(csv.DictReader(stream))
            self.assertEqual(len(comparisons), 12)
            for row in comparisons:
                for method in ("ektgl", "control"):
                    if row[f"{method}_evaluations"]:
                        self.assertLessEqual(int(row[f"{method}_evaluations"]), int(row["budget"]))
            before = [path.read_bytes() for path in files]
            main(argv + ["--resume"])
            self.assertEqual([path.read_bytes() for path in files], before)
            broken = records[0]
            broken["tasks"]["demo1"]["edge_ids"] = [-100]
            files[0].write_text(json.dumps(broken), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Invalid path"):
                main(argv + ["--resume"])
