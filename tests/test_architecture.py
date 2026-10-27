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

    def test_core_does_not_import_gnn_or_optimizer(self):
        for source in (ROOT/'idpc_du').glob('*.py'):
            parsed=ast.parse(source.read_text(encoding='utf8'))
            for node in ast.walk(parsed):
                if isinstance(node,ast.ImportFrom) and node.module:
                    self.assertFalse(node.module.startswith(('gnn','integration','mfea')), source.name)

    def test_gnn_does_not_import_optimizer_or_integration(self):
        for source in (ROOT/'gnn').glob('*.py'):
            parsed=ast.parse(source.read_text(encoding='utf8'))
            for node in ast.walk(parsed):
                if isinstance(node,ast.ImportFrom) and node.module:
                    self.assertFalse(node.module.startswith(('mfea','integration')), source.name)
