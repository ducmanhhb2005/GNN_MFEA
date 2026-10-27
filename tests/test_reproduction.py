"""Regression checks for article interpretation and auditable experiment output."""
import json
import math
import tempfile
import unittest
from pathlib import Path

from idpc_du.graph import Graph, Edge
from idpc_du.gpa import decode_gpa
from idpc_du.validation import validate_path
from mfea.chromosome import Chromosome
from mfea.operators import crossover
from experiments.run_mfea import build_groups, load_reference, summarize, run_group
from tools.unpack_idpc import _safe_target


class TestPaperProtocol(unittest.TestCase):
    def test_parallel_edges_keep_input_indices(self):
        edges=[Edge(10,0,1,8,"2"),Edge(11,0,1,1,"10"),Edge(12,1,2,2,"10")]
        graph=Graph(3,0,2,edges)
        chromosome=Chromosome([3,2,1],[1,1,1])
        self.assertEqual([e.id for e in graph.by_neighbor[0][1]],[10,11])
        self.assertEqual(decode_gpa(graph,chromosome),(3,[11,12]))

    def test_algorithm_1_line_13_arriving_literal(self):
        graph=Graph(3,0,2,[Edge(0,0,1,8,"A"),Edge(1,0,1,1,"B"),Edge(2,1,2,2,"B")])
        chromosome=Chromosome([3,2,1],[2,1,1])
        cost,path=decode_gpa(graph,chromosome)
        self.assertEqual(path,[1,2])
        self.assertEqual(cost,validate_path(graph,path))
        self.assertEqual(cost,3)

    def test_pmx_and_tpx_can_use_different_sections(self):
        class CutPoints:
            def __init__(self): self.points=iter(([0,1],[2,4]))
            def sample(self,values,count): return next(self.points)
        a=Chromosome([1,2,3,4],[1,2,3,4])
        b=Chromosome([4,3,2,1],[5,6,7,8])
        first,second=crossover(a,b,CutPoints())
        self.assertEqual(first.edge_index,[1,2,7,8])
        self.assertEqual(second.edge_index,[5,6,3,4])
        self.assertEqual(sorted(first.priority),[1,2,3,4])

    def test_all_reference_rows_and_pairing(self):
        ref=load_reference()
        self.assertEqual(len(ref["instances"]),42)
        files=[Path(r["set"])/name for name,r in ref["instances"].items()]
        groups=build_groups(files,"adjacent")
        self.assertEqual(len(groups),21)
        self.assertEqual(len({p for group in groups for p in group}),42)
        for group in groups:
            self.assertEqual(len(group),2)
            a,b=[int(p.stem.split("_")[1].split("x")[0]) for p in group]
            self.assertEqual(b-a,5 if a<50 else 10)
        target=Path("set1/idpc_10x5x425.idpc")
        pair=next(g for g in groups if target in g)
        self.assertEqual(pair[1].name,"idpc_15x7x1504.idpc")

    def test_no_silent_odd_group(self):
        with self.assertRaises(ValueError):
            build_groups([Path("set1/idpc_10x5x425.idpc")],"adjacent")

    def test_summary_retains_infeasible_runs(self):
        base=dict(algorithm="MFEA",instance="idpc_10x5x425.idpc",subset="set1",seconds=1.)
        rows=[dict(base,cost=7.,valid=True),dict(base,cost=None,valid=False)]
        row=summarize(rows,load_reference())[0]
        self.assertEqual(row["BF"],7)
        self.assertEqual(row["AVG"],math.inf)
        self.assertEqual(row["feasible"],1)
        self.assertTrue(row["matches_BF"])
        self.assertFalse(row["matches_AVG"])

    def test_seed_checkpoint_resume_and_mismatch(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            graph=root/"graph.json"
            graph.write_text(json.dumps(dict(n=3,source=0,target=2,edges=[[0,1,1,"A"],[1,2,1,"A"],[0,2,4,"B"]])),encoding="utf-8")
            config=dict(runs=1,seed=3,population=4,generations=2,rmp=.5,mutation_rate=.05)
            job=("EA",[str(graph)],config,str(root/"out"),"test-fingerprint")
            paths=run_group(job)
            content=Path(paths[0]).read_bytes()
            self.assertEqual(run_group(job),paths)
            self.assertEqual(Path(paths[0]).read_bytes(),content)
            result=json.loads(content)
            row=result["rows"][0]
            self.assertTrue(row["valid"])
            self.assertEqual(len(row["convergence"]),3)
            with self.assertRaises(ValueError):
                run_group(job[:-1]+("wrong-fingerprint",))

    def test_unpack_rejects_path_traversal(self):
        for name in ("../file.idpc","set1/../../file.idpc","C:/file.idpc"):
            with self.assertRaises(ValueError):
                _safe_target(name,Path("output"))


if __name__ == "__main__":
    unittest.main()
