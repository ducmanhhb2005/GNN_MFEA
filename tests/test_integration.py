import math, random, unittest
from pathlib import Path
from idpc_du import Graph, validate_path
from integration.solver import solve_guided_mfea
from integration.guided_initialization import edge_scores_to_node_priority, lift_task_priority, make_guided_population
DATA=Path(__file__).resolve().parents[1] / 'data'


class TestGNNGuidance(unittest.TestCase):
    def setUp(self):
        self.g=Graph.from_file(str(DATA/'demo1.json'))
        self.h=Graph.from_file(str(DATA/'demo2.json'))

    def test_scores_to_vertex_priority(self):
        scores=[0.05]*len(self.g.edges)
        scores[0]=0.97  # 0->1
        scores[3]=0.7   # 0->3
        pi=edge_scores_to_node_priority(self.g,scores,random.Random(2),jitter=0.)
        self.assertEqual(sorted(pi),list(range(1,self.g.n+1)))
        self.assertGreater(pi[1],pi[3])

    def test_stable_filter_embedding(self):
        rng=random.Random(4)
        priority=[2,4,1,3]
        for _ in range(20):
            unified=lift_task_priority(priority,7,rng)
            self.assertEqual([v for v in unified if v<=4],priority)
            self.assertEqual(sorted(unified),list(range(1,8)))

    def test_mixed_initialization_and_mfea(self):
        p,stat=make_guided_population([self.g,self.h],
            [[0.8]*len(self.g.edges),[0.6]*len(self.h.edges)],16,seed=7,
            guided_fraction=.5)
        self.assertEqual((stat['guided_count'],stat['random_count']),(8,8))
        self.assertEqual(stat['guided_task_allocations'],[4,4])
        self.assertEqual(len(p),16)
        self.assertTrue(all(sorted(c.priority)==list(range(1,8)) for c in p))
        best,_=solve_guided_mfea([self.g,self.h],seed=7,population=16,generations=8,
                           initial_population=p)
        for g,(cost,path) in zip([self.g,self.h],best):
            if math.isfinite(cost):
                self.assertEqual(validate_path(g,path),cost)

    def test_reject_wrong_scores(self):
        with self.assertRaises(ValueError):
            edge_scores_to_node_priority(self.g,[0.5],random.Random(3))

    def test_reject_invalid_supplied_chromosomes(self):
        bad,_=make_guided_population([self.g,self.h],[[0.5]*len(self.g.edges),
              [0.5]*len(self.h.edges)],8,seed=1)
        bad[0].priority=[1]*7
        with self.assertRaises(ValueError):
            solve_guided_mfea([self.g,self.h],population=8,generations=1,initial_population=bad)

class TestBundledTorchEdgeScorer(unittest.TestCase):
    def test_checkpoint_shapes_and_probs(self):
        try:
            import torch
        except ImportError:
            self.skipTest('PyTorch optional for GNN tests')
        from gnn.inference import load_edge_scorer, score_edges
        file=Path(__file__).resolve().parents[1]/'models'/'edge_gnn_synthetic.pt'
        if not file.is_file():
            self.skipTest('Bundled demo checkpoint not found')
        model=load_edge_scorer(str(file))
        graph=Graph.from_file(str(DATA/'demo1.json'))
        probs=score_edges(model,graph)
        self.assertEqual(len(probs),len(graph.edges))
        self.assertTrue(all(0 <= s <= 1 and math.isfinite(s) for s in probs))
