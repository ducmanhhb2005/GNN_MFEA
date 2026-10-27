import math, unittest
from pathlib import Path
from idpc_du import Graph, validate_path, exact_du_small_domains
from mfea.solver import solve_mfea
DATA=Path(__file__).resolve().parents[1] / 'data'


class TestOriginalMendeleyDataset(unittest.TestCase):
    def test_parse_original_10x5(self):
        p=DATA/'benchmark'/'set1'/'idpc_10x5x425.idpc'
        if not p.exists():
            self.skipTest('Original dataset not unpacked')
        from idpc_du.exact import exact_du_small_domains
        g=Graph.from_file(str(p))
        self.assertEqual((g.n,len(g.edges),len(g.domains)),(10,425,5))
        self.assertEqual((g.source,g.target),(0,4))
        optimum,path=exact_du_small_domains(g)
        self.assertEqual(validate_path(g,path),optimum)
        self.assertTrue(math.isfinite(optimum))

    def test_original_mfea_paths_are_valid(self):
        p=DATA/'benchmark'/'set1'/'idpc_10x5x425.idpc'
        if not p.exists():
            self.skipTest('Original dataset not unpacked')
        g=Graph.from_file(str(p))
        gm=Graph.from_file(str(DATA/'benchmark'/'set1'/'idpc_10x10x1000.idpc'))
        multi,_=solve_mfea([g,gm],seed=7,population=16,generations=8)
        for task,(cost,path) in zip([g,gm],multi):
            if math.isfinite(cost):
                self.assertEqual(validate_path(task,path),cost)
