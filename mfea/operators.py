"""Genetic operators: PMX, two-part crossover and mutation."""
from __future__ import annotations
import random
from .chromosome import Chromosome

def pmx_one(p1:list[int], p2:list[int], a:int, b:int)->list[int]:
    n=len(p1)
    c=[None]*n
    c[a:b]=p1[a:b]
    for i in range(a,b):
        if p2[i] in c:
            continue
        value=p2[i]
        position=i
        while a <= position < b:
            position=p2.index(p1[position])
        c[position]=value
    for i in range(n):
        if c[i] is None:
            c[i]=p2[i]
    assert sorted(c)==sorted(p1)
    return c


def crossover(a:Chromosome,b:Chromosome,rng:random.Random)->tuple[Chromosome,Chromosome]:
    n=len(a.priority)
    if n<2:
        return a.clone(), b.clone()
    i,j=sorted(rng.sample(range(n+1),2))
    p1=pmx_one(a.priority,b.priority,i,j)
    p2=pmx_one(b.priority,a.priority,i,j)
    # Alg. 3 invokes PMX and TPX separately; Fig. 3/4 describe separate cuts.
    i,j=sorted(rng.sample(range(n+1),2))
    e1=a.edge_index[:i]+b.edge_index[i:j]+a.edge_index[j:]
    e2=b.edge_index[:i]+a.edge_index[i:j]+b.edge_index[j:]
    return Chromosome(p1,e1),Chromosome(p2,e2)


def mutation(c:Chromosome,limits:list[int],rng:random.Random)->Chromosome:
    o=c.clone()
    if len(o.priority)>=2:
        i,j=rng.sample(range(len(o.priority)),2)
        o.priority[i],o.priority[j]=o.priority[j],o.priority[i]
    idx=rng.randrange(len(o.edge_index))
    o.edge_index[idx]=rng.randint(1,limits[idx])
    o.scores=None
    o.scalar_fitness=0
    return o
