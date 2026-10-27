"""Small-instance exact solvers used ONLY as verification oracles."""
from __future__ import annotations
import math
from .graph import Graph
from .validation import validate_path

def exact_small(g: Graph, max_nodes: int = 12) -> tuple[float, list[int]]:
    """Exhaustive reference solver for tiny instances; NOT meant for paper benchmarks."""
    if g.n > max_nodes:
        raise ValueError(f"exact_small is limited to <= {max_nodes} nodes")
    best=math.inf
    best_path=[]
    def dfs(u, seen, closed, active, c, path):
        nonlocal best,best_path
        if c>=best:
            return
        if u==g.target:
            best=c
            best_path=path[:]
            return
        for e in g.adj[u]:
            if e.v in seen or e.domain in closed:
                continue
            next_closed=closed if active is None or active==e.domain else closed | {active}
            if e.domain in next_closed:
                continue
            dfs(e.v, seen|{e.v}, next_closed, e.domain, c+e.weight, path+[e.id])
    dfs(g.source, {g.source}, set(), None, 0., [])
    return best, best_path


def exact_du_small_domains(g:Graph,max_domains:int=12,max_states:int=2_000_000)->tuple[float,list[int]]:
    """Exact state-space Dijkstra for small number of domains (verification oracle).

    State = (current node, active domain, bitmask of closed domains). Nonnegative
    edge weights mean an optimal path exists without a repeated state. This oracle
    is independent of both paper heuristics; it is not the paper's solver.
    Large instances are intentionally unsupported because states are exponential.
    """
    import heapq
    domains={name:i for i,name in enumerate(g.domains)}
    if len(domains)>max_domains:
        raise ValueError(f"Exact DU solver limited to {max_domains} domains; graph has {len(domains)}")
    # Distinct closed domains are marked in a compact bitmask.
    start=(g.source,-1,0)
    dist={start:0.0}
    predecessor={}
    heap=[(0.0,start)]
    expansions=0
    while heap:
        cost,state=heapq.heappop(heap)
        if cost!=dist.get(state):continue
        node,active,closed=state
        if node==g.target:
            path=[]
            while state!=start:
                prior,eid=predecessor[state]
                path.append(eid)
                state=prior
            path.reverse()
            assert validate_path(g,path)==cost
            return cost,path
        expansions+=1
        if expansions>max_states:
            raise ValueError(f"Exceeded exact solver cap {max_states} expanded states")
        for e in g.adj[node]:
            dom=domains[e.domain]
            bit=1<<dom
            next_closed=closed
            if dom!=active:
                if closed&bit:continue
                if active!=-1:next_closed|=1<<active
            next_state=(e.v,dom,next_closed)
            nextcost=cost+e.weight
            if nextcost<dist.get(next_state,math.inf):
                dist[next_state]=nextcost
                predecessor[next_state]=(state,e.id)
                heapq.heappush(heap,(nextcost,next_state))
    return math.inf,[]
