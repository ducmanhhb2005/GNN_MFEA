"""Import-boundary regression tests for the modular repository."""
import ast
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

class TestModuleBoundaries(unittest.TestCase):
    def test_no_legacy_monolith(self):
        self.assertFalse((ROOT/'src').exists())
        self.assertFalse((ROOT/'mfea2020.py').exists())
        self.assertFalse((ROOT/'gacob2022.py').exists())

    def test_baseline_does_not_import_graph_learning(self):
        for package in ("idpc_du", "mfea"):
            forbidden = {"ekt_gl", "torch", "numpy"}
            if package == "idpc_du":
                forbidden.add("mfea")
            for source in (ROOT / package).glob("*.py"):
                parsed = ast.parse(source.read_text(encoding="utf-8"))
                for node in ast.walk(parsed):
                    if isinstance(node, ast.Import):
                        modules = [alias.name for alias in node.names]
                    elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                        modules = [node.module]
                    else:
                        continue
                    for module in modules:
                        self.assertNotIn(module.split(".")[0], forbidden, str(source))
