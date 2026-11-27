"""Tests of the research adapters with immutable V2 algorithms."""

from collections import Counter
from contextlib import closing
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from ekt_gl.idpc.research.beam import BeamConfig, decode_beam, lower_bound
from ekt_gl.idpc.research.engine import ResearchEvolution, Variant, score_diagnostic
from ekt_gl.idpc.research.learning import ResearchLearning, feature_set
from ekt_gl.idpc.research.neighborhoods import inspect_rebuild
from ekt_gl.idpc.task import CompactIDPCGraph
from ekt_gl.idpc.v2.checkpoint import Checkpoint
from ekt_gl.idpc.v2.config import Config, LocalSearch
from ekt_gl.idpc.v2.core import EvaluationTracker
from ekt_gl.idpc.v2.evolution import Evolution
from ekt_gl.idpc.v2.local_search import _rebuild, DirectedPairs
from experiments.run_idpc_v2 import run_suite
from experiments.run_idpc_v3 import run_research
from experiments._support.idpc_v3_archive import independently_validate
from idpc_du.graph import Edge, Graph


def fixture():
    e=[Edge(i,u,v,w,d) for i,(u,v,w,d) in enumerate([
        (0,1,1,"a"),(0,1,4,"b"),(1,2,0,"b"),(2,3,1,"a"),(2,3,3,"c"),
        (1,3,6,"a"),(2,0,0,"c"),(0,3,19,"b"),(1,0,1,"a"),(3,2,4,"b")])]
    return CompactIDPCGraph.from_graph(Graph(4,0,3,e))


class ResearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_constrained_beam_preserves_domains_and_exact_parallel_ids(self):
        g=fixture()
        h=lower_bound(g)
        self.assertTrue(np.array_equal(h,np.array([2.,1.,1.,0.])))
        result=decode_beam(g,h,BeamConfig(8,0,max_expansions=36,max_complete=8))
        self.assertLessEqual(result.expansions,36)
        self.assertLessEqual(len(result.paths),8)
        costs=[independently_validate(g,p) for p in result.paths]
        self.assertTrue(all(np.isfinite(c) for c in costs))
        self.assertEqual(costs,sorted(costs))
        self.assertIn((0,2,4),result.paths)
        self.assertNotIn((0,2,3),result.paths)
        cap=decode_beam(g,h,BeamConfig(8,0,max_expansions=1,max_complete=1))
        self.assertEqual(cap.expansions,1)
        self.assertTrue(cap.cap_hit)
        with self.assertRaises(ValueError):
            decode_beam(g,h,BeamConfig(beta=.1))

    def test_matched_initialization_matches_mfea_archive(self):
        g=fixture()
        graphs=[g,g]
        config=Config(profile="matched_ls",population_per_task=6,budget_override=60)
        trackers=[EvaluationTracker(graphs,60) for _ in range(2)]
        old=Evolution("mfea",graphs,["x","y"],config,412,trackers[0])
        new=ResearchEvolution("ea",graphs,["x","y"],config,412,trackers[1],
                              variant=Variant(matched_init=True),diagnostic=False)
        old.initialize()
        new.initialize()
        self.assertEqual(old.best,new.best)
        self.assertEqual([len(p) for p in new.populations],[6,6])
        self.assertEqual([t.requests for t in trackers],[24,24])
        self.assertTrue(all(c["init_evaluations"]==12 for t in trackers for c in t.per_task))

    def test_split_truncation_preserves_v2_paths_and_tie_order(self):
        g=fixture()
        pairs=DirectedPairs(g)
        indexes=[pairs.get(u,v) for u,v in [(0,1),(1,2),(2,3)]]
        for cap in (1,3,50):
            config=LocalSearch(beam_width=1,max_edge_expansions_per_cut=cap,max_complete_paths_per_cut=1)
            expected,actual=Counter(),Counter()
            self.assertEqual(_rebuild(indexes,(),(),pairs.codes,config,expected),
                             inspect_rebuild(indexes,(),(),pairs.codes,config,actual))
            self.assertTrue(all(actual[k]==v for k,v in expected.items()))
            self.assertEqual(actual["ls_truncated"],int(any(actual[k] for k in
                ("ls_beam_pruned_cuts","ls_expansion_capped_cuts","ls_complete_capped_cuts"))))

    def test_exact_average_precision_with_probability_ties(self):
        g=fixture()
        scores=np.array([.4,.4,.07,.32,.4,.6,.7,.8,.9,.2])
        positive={0,4,5}
        metrics=score_diagnostic(g,scores,positive,100)
        expected=(1/4+2*(3/7))/3
        self.assertAlmostEqual(metrics["PR_AUC_AP"],expected)
        self.assertAlmostEqual(metrics["recall_at_half"],1/3)

    def test_learner_variants_finite_gradient(self):
        g=fixture()
        for policy,loss in (("legacy","sampled"),("distances","balanced"),("v2","full")):
            with self.subTest(policy=policy,loss=loss):
                c=Config(hidden_dim=8,edge_batch_size=6)
                learner=ResearchLearning([g],[feature_set(g,"cpu",policy)],c,100,
                                         Variant(features=policy,loss=loss),True)
                value,_=learner.fit([({0,2,4},1,0)])
                self.assertTrue(np.isfinite(value))
                self.assertGreater(learner.gradients["encoder"],0)
                self.assertGreater(learner.gradients["decoder"],0)
                scores,attention,_=learner.scores()
                self.assertTrue(np.all(np.isfinite(scores[0])))
                self.assertEqual(attention,[[1.0]])

    def test_a0_matches_v2_on_budget_and_trajectory(self):
        g=fixture()
        graphs=[g,g]
        names=["g","h"]
        config=Config(hidden_dim=8,gcn_layers=1,population_per_task=6,generations=12)
        features=[feature_set(g,"cpu","v2") for g in graphs]
        with tempfile.TemporaryDirectory() as d:
            with closing(Checkpoint(Path(d)/"a.sqlite",("a","ektgl",120))) as ck:
                a=run_suite(ck,"ektgl",120,graphs,names,config,"a",features,
                            [DirectedPairs(g) for g in graphs],progress=0)
                ta=list(ck.generations())
            with closing(Checkpoint(Path(d)/"b.sqlite",("b","ektgl",120))) as ck:
                b=run_research(ck,"ektgl",120,graphs,names,config,"b",features,None,Variant())
                tb=list(ck.generations())
        self.assertEqual([r["best_path_edge_ids"] for r in a["tasks"]],
                         [r["best_path_edge_ids"] for r in b["tasks"]])
        for key in ("best_costs","suite_evaluations","evaluations","populations"):
            self.assertEqual([r[key] for r in ta],[r[key] for r in tb])


if __name__=="__main__":
    unittest.main()
