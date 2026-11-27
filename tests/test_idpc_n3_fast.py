"""Deterministic parity between archived N3 and its faster implementation."""
from dataclasses import asdict
import random
import unittest
from unittest.mock import patch

import numpy as np

from ekt_gl.idpc.research import neighborhoods
from ekt_gl.idpc.research.beam import BeamConfig, decode_beam, lower_bound
from ekt_gl.idpc.research.n3_fast import CostOnlyBeam, replace_segment, validate_path
from ekt_gl.idpc.task import CompactIDPCGraph
from ekt_gl.idpc.v2.config import LocalSearch
from ekt_gl.idpc.v2.core import Candidate, EvaluationTracker, Solution, limits_for
from ekt_gl.idpc.v2.local_search import DirectedPairs
from idpc_du.graph import Edge, Graph
from idpc_du.validation import validate_path as reference_validate
from mfea.chromosome import Chromosome
from test_idpc_v3 import fixture


class ExactN3OptimizationTests(unittest.TestCase):
    def test_order_caps_parallel_edges_and_large_domain_masks(self):
        rng = random.Random(2901)
        for vertex_count, divisor in ((7,1), (75,1), (12,7)):
            edges = [Edge(i, u, v, i % 9 / divisor, str(i % 94))
                     for i, (u,v) in enumerate((rng.randrange(vertex_count), rng.randrange(vertex_count))
                                                for _ in range(450)) if u != v]
            graph = CompactIDPCGraph.from_graph(Graph(vertex_count, 0, vertex_count-1, edges))
            fast = CostOnlyBeam()
            for _ in range(45):
                start,target = rng.sample(range(vertex_count), 2)
                lower = lower_bound(graph,target)
                visited = sum(1 << v for v in rng.sample(range(vertex_count), 3) if v != target)
                closed = sum(1 << d for d in rng.sample(range(len(graph.domains)), 15))
                config = BeamConfig(rng.choice((1,3,8)), 0, rng.choice((1,8,4096)), rng.choice((1,2,4)))
                args = dict(start=start,target=target,visited=visited,
                            closed=closed,active=rng.randrange(-1,len(graph.domains)))
                self.assertEqual(decode_beam(graph,lower,config,**args), fast(graph,lower,config,**args))

    def test_integral_rank_reuse_and_exact_fractional_fallback(self):
        graph = fixture()
        fast = CostOnlyBeam()
        lower = lower_bound(graph)
        shared_lower, _, shared = fast.target(graph,lower)
        self.assertIs(shared_lower,lower)
        self.assertTrue(shared)
        ranks = fast.ranks(graph,lower,0,1,shared)
        count = fast.ranking_reads
        self.assertIs(ranks,fast.ranks(graph,lower,0,100,shared))
        self.assertEqual(fast.ranking_reads,count)
        large = np.array([3., 2**52, 1., 2**52 + 2])
        self.assertFalse(fast.target(graph,large)[2])
        params = BeamConfig(8,0,4096,4)
        self.assertEqual(decode_beam(graph,large,params), fast(graph,large,params))

    def test_exact_replacement_compares_candidate_counters_and_cut_order(self):
        graph = fixture()
        limits = limits_for([graph])
        for path, cost in (((7,),19), ((0,5),7), ((1,2,4),7)):
            trackers = [EvaluationTracker([graph]), EvaluationTracker([graph])]
            self.assertEqual(reference_validate(graph,path),cost)
            chromosome = Chromosome([1,2,3,4], [1,1,1,1])
            candidates = [Candidate(chromosome, Solution(cost,path)) for _ in trackers]
            for seed in (4300,4301,4310):
                config = LocalSearch(**{**asdict(LocalSearch()), "max_cut_pairs_per_candidate":2})
                rng1,rng2 = random.Random(seed),random.Random(seed)
                a = neighborhoods.replace_segment(trackers[0],0,candidates[0],limits,DirectedPairs(graph),
                                                    config,rng1,"ordinary")
                with patch.object(neighborhoods,"decode_beam",CostOnlyBeam()):
                    b = replace_segment(trackers[1],0,candidates[1],limits,DirectedPairs(graph),
                                        config,rng2,"ordinary")
                self.assertEqual(a,b)
                self.assertEqual(rng1.getstate(),rng2.getstate())
                self.assertEqual(trackers[0].requests,trackers[1].requests)
                def stable(tracker):
                    return {k:v for k,v in tracker.totals().items() if not k.endswith("_time_s")}
                self.assertEqual(stable(trackers[0]),stable(trackers[1]))

    def test_validator_checks_exact_directed_ids_and_du(self):
        for graph in (fixture(), CompactIDPCGraph.from_graph(Graph(4,0,3,[
                Edge(91,0,1,.1,"a"), Edge(4,1,2,.2,"b"), Edge(22,2,3,.3,"a"),
                Edge(23,2,3,.3,"b"), Edge(11,0,3,10.1,"c"), Edge(12,2,1,.2,"b")
        ]))):
            for path in ((),(7,),(0,5),(1,2,4),(0,2,3),(4,),(91,4,22),(91,4,23),
                         (91,4,12,4,23),(91,23),(11,),(2**70,),(-1,)):
                self.assertEqual(reference_validate(graph,path),validate_path(graph,path))

    def test_filtered_rank_prefix_falls_back_to_exact_tail(self):
        edges = [Edge(i,0,1,0,"closed") for i in range(180)]
        edges.extend([Edge(200,1,3,0,"open"), Edge(201,0,2,1,"open"),
                      Edge(202,2,3,1,"open")])
        graph = CompactIDPCGraph.from_graph(Graph(4,0,3,edges))
        beam = BeamConfig(8,0,4096,4)
        lower = lower_bound(graph)
        closed = 1 << graph.domains.index("closed")
        expected = decode_beam(graph,lower,beam,closed=closed)
        self.assertEqual(expected.paths,((201,202),))
        fast = CostOnlyBeam()
        self.assertEqual(expected,fast(graph,lower,beam,closed=closed))
        self.assertTrue(any(key[-1] is None for key in fast.cache))
