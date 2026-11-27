"""The generation profile preserves phase settings and removes only the budget."""
from dataclasses import asdict
import unittest

from ekt_gl.idpc.v2.config import Config
from ekt_gl.idpc.v2.core import EvaluationTracker
from experiments.run_idpc_matched_n3_generation import generation_config
from test_idpc_v3 import fixture


class GenerationProfileTests(unittest.TestCase):
    def test_generation_uses_identical_settings_and_uncapped_ledger(self):
        base = Config(profile="matched_ls", population_per_task=30, gpa_backend="numba")
        current = generation_config(asdict(base))
        self.assertEqual(asdict(current), asdict(base))
        self.assertEqual(base.budget(42), 379260)
        self.assertIsNone(current.budget(42))
        self.assertEqual(current.generations, 300)
        self.assertTrue(all(current.ordinary_ls(m) for m in ("ea", "mfea", "ektgl")))
        tracker = EvaluationTracker([fixture()], current.budget(42))
        for _ in range(379261):
            tracker.path(0, (7,), "ls")
        self.assertFalse(tracker.exhausted)
        self.assertEqual(tracker.requests, 379261)
        tracker.check_invariants()
