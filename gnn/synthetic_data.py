"""Toy graph generation for prototype GNN supervision; NOT benchmark labels."""
from __future__ import annotations
import random
from idpc_du.graph import Edge, Graph

def generate_training_graph(rng: random.Random, n_min: int = 5, n_max: int = 8) -> Graph:
    """Build toy connected graphs with at least one DU-valid route; NOT original benchmark."""
    n = rng.randint(n_min, n_max)
    interior = list(range(1,n-1))
    rng.shuffle(interior)
    k = rng.randint(1, len(interior))
    path = [0] + interior[:k] + [n-1]
    edges: list[Edge] = []
    for step,(u,v) in enumerate(zip(path, path[1:])):
        edges.append(Edge(len(edges),u,v,float(rng.randint(1,4)),f"P{step//2}"))
    for u in range(n):
        for v in range(n):
            if u==v or rng.random() > 0.22:
                continue
            for _ in range(1+(rng.random()<0.12)):
                edges.append(Edge(len(edges),u,v,float(rng.randint(1,12)),
                                  f"P{rng.randrange(0,4)}"))
    return Graph(n, 0, n-1, edges)
