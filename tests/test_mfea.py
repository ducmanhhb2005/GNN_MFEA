import math, random, unittest
from pathlib import Path
from idpc_du import Graph, Edge, validate_path, exact_small, exact_du_small_domains
from idpc_du.gpa import decode_gpa
from mfea.chromosome import Chromosome, edge_index_limits
from mfea.operators import pmx_one, mutation
from mfea.ranking import assign_ranks
from mfea.solver import solve_mfea
DATA=Path(__file__).resolve().parents[1] / 'data'


class TestAlgorithms(unittest.TestCase):
    def setUp(self):
        self.g=Graph.from_json(str(DATA/'demo1.json'))

    def test_domain_reentry(self):
        # blue -> red -> blue is forbidden even though graph edges connect
        eids=[0,6,7]
        self.assertEqual(validate_path(self.g,eids),math.inf)

    def test_valid_path(self):
        self.assertEqual(validate_path(self.g,[0,1,2]),4)

    def test_exact(self):
        v,p=exact_small(self.g)
        self.assertTrue(math.isfinite(v))
        self.assertEqual(v,validate_path(self.g,p))

    def test_pmx_property(self):
        rng=random.Random(2)
        for n in range(3,16):
            for _ in range(30):
                a=list(range(1,n+1));b=a[:]
                rng.shuffle(a);rng.shuffle(b)
                lo,hi=sorted(rng.sample(range(n+1),2))
                self.assertEqual(sorted(pmx_one(a,b,lo,hi)),list(range(1,n+1)))

    def test_mutation_domain(self):
        rng=random.Random(2)
        g2=Graph.from_json(str(DATA/'demo2.json'))
        limits=edge_index_limits([self.g,g2])
        c=Chromosome(list(range(1,8)),[1]*7)
        child=mutation(c,limits,rng)
        self.assertEqual(sorted(child.priority),list(range(1,8)))

    def test_gpa_validity(self):
        c=Chromosome([6,5,4,3,2,1],[1]*6)
        v,path=decode_gpa(self.g,c)
        self.assertEqual(validate_path(self.g,path),v)

    def test_mfea_smoke(self):
        g2=Graph.from_json(str(DATA/'demo2.json'))
        best,_=solve_mfea([self.g,g2],seed=7,population=24,generations=15)
        for g,(v,p) in zip([self.g,g2],best):
            self.assertTrue(math.isfinite(v))
            self.assertEqual(validate_path(g,p),v)

class TestGPAReproduction(unittest.TestCase):
    """Deterministic tests for CEC 2020 Alg. 1 choices and task decoding."""
    def setUp(self):
        from idpc_du.graph import Edge
        self.g=Graph(3,0,2,[Edge(0,0,1,9.,'A'),Edge(1,0,1,1.,'B'),
                             Edge(2,1,2,2.,'B'),Edge(3,0,2,20.,'C')])

    def test_algorithm_1_arriving_gene_literal_modulo(self):
        # sigma[source]=1 would select the cheap B edge if the departing gene
        # were used. Algorithm 1 uses sigma[1]=2: 2 % 2 = 0, selecting A.
        c=Chromosome([3,2,1],[1,2,1])
        cost,path=decode_gpa(self.g,c)
        self.assertEqual(cost,11.)
        self.assertEqual(path,[0,2])

    def test_du_reentry_dead_end_is_infeasible(self):
        from idpc_du.graph import Edge
        graph=Graph(4,0,3,[Edge(0,0,1,1.,'A'),Edge(1,1,2,1.,'B'),
                             Edge(2,2,3,1.,'A')])
        for i in range(1,4):
            c=Chromosome([4,3,2,1],[i]*4)
            cost,path=decode_gpa(graph,c)
            self.assertFalse(math.isfinite(cost))
            self.assertEqual(path,[])

    def test_task_space_stable_filter(self):
        # Keep values <= 3 and preserve order: [5,2,4,1,3] => [2,1,3]
        # The task has the same edges, but a larger unified chromosome.
        c=Chromosome([5,2,4,1,3],[2,1,1,1,1])
        cost,path=decode_gpa(self.g,c)
        self.assertEqual(validate_path(self.g,path),cost)

    def test_missing_task_cost_has_no_fictitious_rank(self):
        from mfea.ranking import assign_ranks
        a=Chromosome([1,2],[1,1],scores={0:10.})
        b=Chromosome([1,2],[1,1],scores={1:5.})
        c=Chromosome([1,2],[1,1],scores={1:8.})
        assign_ranks([a,b,c],2,random.Random(0))
        self.assertEqual((a.skill,b.skill,c.skill),(0,1,1))
        self.assertEqual((a.scalar_fitness,b.scalar_fitness,c.scalar_fitness),(1.,1.,0.5))

class TestPaperBestRepresentability(unittest.TestCase):
    """Diagnostic ONLY: exact known optimum can be encoded into GPA chromosome.

    The chromosome is constructed from oracle path solely in test code.
    This is NOT fed to MFEA and does not leak benchmark optimum into search.
    """
    def test_paper_small_optimum_is_representable(self):
        from idpc_du.exact import exact_du_small_domains
        for filename in ('idpc_10x5x425.idpc','idpc_10x10x1000.idpc'):
            g=Graph.from_file(str(DATA/'benchmark'/'set1'/filename))
            optimal,edge_ids=exact_du_small_domains(g)
            self.assertEqual(optimal,7.)
            sequence=[g.by_id[e].v for e in edge_ids]
            visited=[g.source]+sequence
            others=[v for v in range(g.n) if v not in visited]
            ordered=sequence+others+[g.source]
            pi=[0]*g.n
            for priority,vertex in enumerate(reversed(ordered),1):
                pi[vertex]=priority
            sigma=[1]*g.n
            closed=set();active=None
            for eid in edge_ids:
                edge=g.by_id[eid]
                eligible=[e.id for e in g.by_neighbor[edge.u][edge.v] if e.domain not in closed]
                index=eligible.index(eid)
                # Positive 1..Si gene whose literal modulo selects index.
                sigma[edge.v]=index if index else len(eligible)
                if active is not None and active!=edge.domain:
                    closed.add(active)
                active=edge.domain
            cost,path=decode_gpa(g,Chromosome(pi,sigma))
            self.assertEqual(cost,7.)
            self.assertEqual(path,edge_ids)
