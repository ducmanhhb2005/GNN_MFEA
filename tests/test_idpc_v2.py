"""V2 correctness gates for algorithms, data, budget and checkpoint resume."""

from collections import Counter
from contextlib import closing
from copy import deepcopy
from dataclasses import replace
import math
import importlib.util
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from ekt_gl.idpc.task import CompactIDPCGraph
from ekt_gl.idpc.v2.checkpoint import Checkpoint, read_checkpoint
from ekt_gl.idpc.v2.config import Config, LocalSearch
from ekt_gl.idpc.v2.core import (EvaluationTracker, INFEASIBLE, REQUESTS, encode, initialize,
                                lift, limits_for, project, vary)
from ekt_gl.idpc.v2.evolution import Evolution, Member, assign_ranks, offspring_mfea
from ekt_gl.idpc.v2.learning import EdgeNetwork, Features, OnlineLearning, greedy
from ekt_gl.idpc.v2.local_search import DirectedPairs, cut_pairs, improve
from experiments.run_idpc_v2 import run_suite
from idpc_du.graph import Edge, Graph
from idpc_du.gpa import decode_gpa
from idpc_du.validation import validate_path
from mfea.chromosome import Chromosome
from mfea.operators import crossover


def graph(n, rows):
    return CompactIDPCGraph.from_graph(Graph(n, 0, n - 1,
        [Edge(i, u, v, w, d) for i, (u, v, w, d) in enumerate(rows)]))


def two_opt_graph(reverse=True, conflict=False):
    # 0->1 prefix, [2,3] reversed, 4->5 suffix.
    rows = [(0, 1, 1, "A"), (1, 2, 10, "B"), (2, 3, 10, "B"), (3, 4, 10, "D"),
            (4, 5, 1, "D"), (1, 3, 1, "D"), (1, 3, 2, "C"),
            (3, 2, 1, "C"), (2, 4, 1, "D")]
    if not reverse:
        rows[7] = (3, 1, 1, "C")
    if conflict:
        rows[6] = (1, 3, 2, "D")
    return graph(6, rows)


def evaluate_path(g, path, budget=None):
    tracker = EvaluationTracker([g], budget, cache_size=8)
    current = tracker.path(0, path, "init")
    candidate = tracker.roundtrip(0, current, limits_for([g]), "init")
    return tracker, candidate


class CoreTests(unittest.TestCase):
    def test_gpa_uss_validator_and_parallel_order(self):
        g = graph(4, [(0,1,1,"A"), (0,1,2,"B"), (1,2,3,"A"), (2,3,4,"B"),
                      (0,2,2,"B"), (1,3,1,"A"), (1,3,1,"B")])
        self.assertEqual([e.id for e in g.by_neighbor[0][1]], [0, 1])
        self.assertEqual(limits_for([g]), [2, 2, 1, 0])
        self.assertEqual(validate_path(g, (0, 2, 3)), 8)
        for bad in ((1, 2, 3), (0, 2), (2, 3), (999,)):
            self.assertEqual(validate_path(g, bad), INFEASIBLE)
        limits = limits_for([g])
        c = encode(g, (0, 2, 3), limits)
        self.assertIsNotNone(c)
        self.assertEqual(c.edge_index[-1], -1)
        unified = Chromosome([6, 2, 5, 4, 1, 3], [2, 2, 3, 2, 5, 4])
        local = project(unified, 4)
        self.assertEqual(local.priority, [2, 4, 1, 3])
        self.assertEqual(local.edge_index, [2, 2, 3, 2])
        lifted = lift(c, unified)
        self.assertEqual(project(lifted, 4).priority, c.priority)
        self.assertEqual(lifted.edge_index[4:], [5, 4])
        tracker = EvaluationTracker([g])
        self.assertEqual(tracker.gpa(0, lifted, "init").path, (0, 2, 3))
        self.assertEqual(unified.priority, [6, 2, 5, 4, 1, 3])
        unreachable = graph(3, [(0, 1, 1, "A")])
        self.assertEqual(decode_gpa(unreachable, Chromosome([1,2,3], [1,1,1])), (INFEASIBLE, []))

    @unittest.skipUnless(importlib.util.find_spec("numba"), "Numba backend is optional")
    def test_compiled_gpa_matches_reference_random_multigraphs(self):
        from ekt_gl.idpc.v2.gpa_fast import PackedGPA
        rng = random.Random(921)
        for n in range(2, 17):
            rows = [(u,v,rng.randrange(1,91),str(rng.randrange(5)))
                    for u in range(n) for v in range(n) if u != v
                    for _ in range(rng.randrange(4))]
            g = graph(n, rows)
            compiled = PackedGPA(g)
            limits = limits_for([g])
            for _ in range(18):
                c = initialize(limits,rng)
                reference = Chromosome(c.priority,[max(1,s) for s in c.edge_index])
                self.assertEqual(compiled(c),decode_gpa(g,reference))

    def test_operator_properties(self):
        rng = random.Random(176)
        for n in range(1, 30):
            limits = [i % 5 for i in range(n)]
            for _ in range(4):
                a, b = [initialize(limits, rng) for _ in range(2)]
                backup = a.clone(), b.clone()
                for child in crossover(a, b, rng):
                    mutant = vary(child, limits, rng)
                    self.assertEqual(sorted(mutant.priority), list(range(1, n+1)))
                    for gene, limit in zip(mutant.edge_index, limits):
                        self.assertTrue(1 <= gene <= limit if limit else gene == -1)
                    self.assertEqual(sum(x != y for x,y in zip(mutant.priority, child.priority)),
                                     2 if n > 1 else 0)
                    self.assertLessEqual(sum(x != y for x,y in zip(mutant.edge_index, child.edge_index)), 1)
                self.assertEqual((a,b), backup)

    def test_shared_budget_and_cache(self):
        g = graph(3, [(0,1,2,"A"), (1,2,1,"A")])
        t = EvaluationTracker([g], budget=3, cache_size=4)
        c = Chromosome([1,2,3], [1,1,-1])
        self.assertEqual(t.gpa(0,c,"init").cost, 3)
        self.assertEqual(t.gpa(0,c,"offspring").cost, 3)
        self.assertEqual(t.path(0,(1,),"transfer").cost, INFEASIBLE)
        self.assertIsNone(t.gpa(0,c,"ls"))
        t.check_invariants()
        self.assertEqual(t.totals()["cache_hits"], 1)
        self.assertEqual(t.totals()["actual_objective_computations"], 2)
        self.assertEqual(t.totals()["gpa_decode_calls"], 1)

    def test_rank_mask_ties_and_assortative_mating(self):
        p = [
            Member(Chromosome([1,2], [1,1]), {0: 5, 1: 12}),
            Member(Chromosome([2,1], [1,1]), {0: 3}),
            Member(Chromosome([2,1], [1,1]), {1: INFEASIBLE}),
            Member(Chromosome([1,2], [1,1]), {}),
        ]
        assign_ranks(p, 2)
        self.assertEqual([(m.skill, m.scalar_fitness) for m in p],
                         [(1,1.0),(0,1.0),(1,0.5),(None,0.0)])
        for rmp, parents, expect_cross in [(0, (p[0], p[1]), False), (1, (p[0], p[1]), True),
                                            (0, (p[0], p[0]), True)]:
            counts = Counter()
            cfg = Config(rmp=rmp, mutation_rate=1)
            with patch("ekt_gl.idpc.v2.evolution.select", side_effect=parents):
                children = list(offspring_mfea(p, 2, cfg, [1,1], random.Random(1), counts))
            self.assertEqual(counts["mutation_calls"], 2)
            self.assertEqual(counts["crossover_matings"], int(expect_cross))
            self.assertEqual(counts["mutation_only_matings"], int(not expect_cross))
            self.assertTrue(all(skill in {a.skill for a in parents} for _,skill in children))
            if not expect_cross:
                self.assertEqual([s for _,s in children], [parents[0].skill, parents[1].skill])


class DirectedTwoOptTests(unittest.TestCase):
    def test_improvement_exact_edges_and_oracle(self):
        g = two_opt_graph()
        tracker, candidate = evaluate_path(g, (0,1,2,3,4))
        self.assertEqual(candidate.solution.cost, 32)
        before = candidate.chromosome.clone()
        pairs = DirectedPairs(g)
        improvement = improve(tracker,0,candidate,limits_for([g]),pairs,LocalSearch(),
                              random.Random(9),"ordinary")
        self.assertEqual(improvement.solution.path, (0,6,7,8,4))
        self.assertEqual(improvement.solution.cost, 6)
        # Exhaustive edge choices only on this tiny fixed reversed sequence.
        from itertools import product
        choices = [g.by_neighbor[u][v] for u,v in [(1,3),(3,2),(2,4)]]
        oracle = min(validate_path(g,(0,*(e.id for e in edges),4)) for edges in product(*choices))
        self.assertEqual(improvement.solution.cost, oracle)
        self.assertEqual(candidate.chromosome, before)
        old_nodes = {g.source, *(g.by_id[e].v for e in candidate.solution.path)}
        self.assertEqual(old_nodes, {g.source, *(g.by_id[e].v for e in improvement.solution.path)})
        self.assertEqual(tracker.totals()["ls_accepted"],1)
        self.assertGreater(tracker.totals()["ls_du_rejects"],0)
        tracker.check_invariants()

    def test_missing_direction_du_conflict_short_and_capped_moves(self):
        for g, name in [(two_opt_graph(reverse=False),"ls_missing_edge"),
                        (two_opt_graph(conflict=True),"ls_du_rejects")]:
            tracker, candidate = evaluate_path(g,(0,1,2,3,4))
            self.assertIs(improve(tracker,0,candidate,limits_for([g]),DirectedPairs(g),
                                  LocalSearch(),random.Random(11),"ordinary"),candidate)
            self.assertGreater(tracker.totals()[name],0)
        g = two_opt_graph()
        for budget, expansion_cap in [(3,4096),(None,1)]:
            tracker, candidate = evaluate_path(g,(0,1,2,3,4),budget)
            caps = LocalSearch(max_cut_pairs_per_candidate=1,max_edge_expansions_per_cut=expansion_cap,
                               max_complete_paths_per_cut=1,beam_width=1)
            result = improve(tracker,0,candidate,limits_for([g]),DirectedPairs(g),caps,
                             random.Random(11),"ordinary")
            self.assertIs(result,candidate)
            self.assertLessEqual(tracker.totals()["ls_edge_expansions"],expansion_cap)
            self.assertLessEqual(tracker.totals()["ls_complete_paths"],1)
            self.assertLessEqual(tracker.totals()["ls_cut_attempts"],1)
            tracker.check_invariants()
        a = cut_pairs(15, 16, random.Random(8))
        self.assertEqual(len(set(a)),16)
        self.assertEqual(a,cut_pairs(15,16,random.Random(8)))
        short = graph(3,[(0,1,2,"A"),(1,2,2,"A")])
        tracker,candidate = evaluate_path(short,(0,1))
        self.assertIs(improve(tracker,0,candidate,limits_for([short]),DirectedPairs(short),
                              LocalSearch(),random.Random(4),"ordinary"),candidate)
        self.assertEqual(tracker.totals()["ls_cut_attempts"],0)

    def test_memoized_beam_preserves_candidates_and_objective_requests(self):
        g = two_opt_graph()
        seen = []
        for cache_size in (0, 64):
            t, candidate = evaluate_path(g,(0,1,2,3,4))
            pairs,gains = DirectedPairs(g,cache_size),[]
            for _ in range(2):
                gains.append(improve(t,0,candidate,limits_for([g]),pairs,LocalSearch(),
                                     random.Random(9),"ordinary"))
            seen.append((gains,t.totals()))
        self.assertEqual(seen[0][0],seen[1][0])
        self.assertEqual(seen[0][1][REQUESTS],seen[1][1][REQUESTS])
        self.assertEqual(seen[0][1]["ls_edge_expansions"],seen[1][1]["ls_edge_expansions"])
        self.assertLess(seen[1][1]["ls_actual_edge_expansions"],seen[0][1]["ls_actual_edge_expansions"])
        self.assertGreater(seen[1][1]["ls_beam_cache_hits"],0)

    def test_uss_lift_and_stale_costs(self):
        g = two_opt_graph()
        tracker, original = evaluate_path(g,(0,1,2,3,4))
        extended = Chromosome([7]+original.chromosome.priority, original.chromosome.edge_index+[2],
                              skill=0,scores={0:32.0,1:100.0},scalar_fitness=1.0)
        candidate = type(original)(extended,original.solution)
        result = improve(tracker,0,candidate,limits_for([g]),DirectedPairs(g),LocalSearch(),
                         random.Random(9),"ordinary",unified=True)
        self.assertEqual(len(result.chromosome.priority),7)
        self.assertIsNone(result.chromosome.scores)
        self.assertEqual(result.solution.path,(0,6,7,8,4))
        self.assertEqual(extended.scores,{0:32.0,1:100.0})


class LearningAndResumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_edge_shapes_labels_attention_gradients_and_domain_invariance(self):
        g = two_opt_graph()
        relabel = {domain: str(17-i) for i,domain in enumerate(g.domains)}
        other = graph(6,[(e.u,e.v,e.weight,relabel[e.domain]) for e in g.edges])
        a,b = (Features.build(item,"cpu") for item in (g,other))
        torch.testing.assert_close(a.nodes,b.nodes)
        torch.testing.assert_close(a.edge_features,b.edge_features)
        self.assertFalse(torch.equal(a.edge_features[5], a.edge_features[6]))
        model = EdgeNetwork(16,4)
        bases,weights,attention = model.contexts([a,b])
        logits = model.logits(bases[0],weights,a.edge_pair,a.edge_features,
                              model.last.weight,model.last.bias)
        self.assertEqual(logits.shape,(len(g.edges),2))
        torch.testing.assert_close(attention.sum(1),torch.ones(2))
        logits.square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        config = Config(hidden_dim=16,gcn_layers=2,train_epochs=130,learning_rate=0.015)
        network = OnlineLearning([g],[a],config)
        loss,_ = network.fit([({0,1,2,3,4},1,0)])
        self.assertLess(loss,0.08)
        scores,_,_ = network.scores()
        self.assertEqual(scores[0].shape,(len(g.edges),))
        self.assertEqual(greedy(g,scores[0],Counter()),(0,1,2,3,4))


    def test_chunked_loss_matches_unchunked_autograd(self):
        first = two_opt_graph()
        second = graph(4, [(0,1,2,"A"), (1,2,3,"A"), (2,3,1,"B"), (0,3,8,"C")])
        graphs = [first,second]
        features = [Features.build(g,"cpu") for g in graphs]
        config = Config(hidden_dim=12,gcn_layers=2,edge_batch_size=3)
        torch.manual_seed(915)
        learner = OnlineLearning(graphs,features,config)
        replica = deepcopy(learner.model)
        optimizer = torch.optim.Adam(replica.parameters(),lr=config.learning_rate)
        labels = [({0,1,2,3,4},1,0),({0,1,2},1,0)]
        bases,weights,_ = replica.contexts(features)
        losses = []
        for features_i,base,label in zip(features,bases,labels):
            logits = replica.logits(base,weights,features_i.edge_pair,features_i.edge_features,
                                    replica.last.weight,replica.last.bias)
            target = torch.zeros(len(logits),dtype=torch.long)
            target[sorted(label[0])] = 1
            losses.append(torch.nn.functional.cross_entropy(logits,target))
        expected = torch.stack(losses).mean()
        expected.backward()
        optimizer.step()
        loss,_ = learner.fit(labels)
        self.assertAlmostEqual(loss,float(expected.detach()),places=6)
        for actual,reference in zip(learner.model.parameters(),replica.parameters()):
            torch.testing.assert_close(actual,reference,rtol=1e-4,atol=2e-5)

    def test_checkpoint_rejects_mismatch(self):
        with tempfile.TemporaryDirectory() as root:
            target = Path(root)/"run.sqlite"
            with closing(Checkpoint(target,("original","ea",100))):
                pass
            with self.assertRaisesRegex(ValueError,"fingerprint mismatch"):
                Checkpoint(target,("different","ea",100))

    def test_suite_counts_local_search_native_and_resume(self):
        g = two_opt_graph()
        graphs = [g,graph(4,[(0,1,1,"A"),(1,2,2,"A"),(2,3,2,"B"),(0,3,7,"C")])]
        names = ["x","y"]
        features = [Features.build(item,"cpu") for item in graphs]
        pairs = [DirectedPairs(item) for item in graphs]
        config = Config(population_per_task=6,generations=5,transfer_interval=2,
                        hidden_dim=8,gcn_layers=2,threads=1)
        with tempfile.TemporaryDirectory() as root:
            for method in ("ea","mfea","ektgl"):
                full = Path(root)/f"{method}.sqlite"
                resume = Path(root)/f"{method}-resume.sqlite"
                args = (method,100,graphs,names,config,"fp",features,pairs)
                with closing(Checkpoint(full,("fp",method,100))) as c:
                    result = run_suite(c,*args,progress=0)
                with closing(Checkpoint(resume,("fp",method,100))) as c:
                    self.assertIsNone(run_suite(c,*args,pause_after=2,progress=0))
                with closing(Checkpoint(resume,("fp",method,100))) as c:
                    restored = run_suite(c,*args,progress=0)
                self.assertEqual([(t["best_cost"],t["best_path_edge_ids"]) for t in result["tasks"]],
                                 [(t["best_cost"],t["best_path_edge_ids"]) for t in restored["tasks"]])
                _,trace = read_checkpoint(full,"fp",method,100)
                _,trace_replayed = read_checkpoint(resume,"fp",method,100)
                for x,y in zip(trace,trace_replayed):
                    for key in ("generation","populations","best_costs","evaluations","suite_evaluations"):
                        self.assertEqual(x[key],y[key])
                    self.assertEqual(sum(n for n,_ in x["populations"]),12)
                    if x["transfer"]:
                        self.assertEqual(x["transfer"]["rows"],y["transfer"]["rows"])
                counts = result["counters"]
                self.assertEqual(counts["init_evaluations"],24 if method=="mfea" else 12)
                self.assertEqual(counts["offspring_evaluations"],36 if method=="ektgl" else 60)
                if method=="ektgl":
                    self.assertEqual(result["transfer_events"],2)
                    self.assertGreater(counts["ordinary_ls_calls"],0)
                    self.assertGreater(counts["pre_transfer_ls_calls"],0)
                else:
                    self.assertEqual(counts.get("local_search_checks",0),0)
                    self.assertEqual(result["transfer_events"],0)
            limited = replace(config,profile="matched_ls",budget_override=25)
            path = Path(root)/"limited.sqlite"
            with closing(Checkpoint(path,("fp","mfea",100))) as c:
                record = run_suite(c,"mfea",100,graphs,names,limited,"fp",None,pairs,progress=0)
            self.assertEqual(record["counters"][REQUESTS],25)
            self.assertEqual(record["actual_generations"],1)
            self.assertEqual(record["stop_rule"],"budget")


if __name__ == "__main__":
    unittest.main()
