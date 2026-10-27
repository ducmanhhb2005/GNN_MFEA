"""CLI for MFEA-EDU and small exact IDPC-DU checks."""
import argparse, math, statistics
from idpc_du import Graph, exact_small, validate_path
from mfea.solver import solve_mfea, solve_ea

def main():
    p=argparse.ArgumentParser(description="IDPC-DU: MFEA-EDU (CEC 2020)")
    p.add_argument("algorithm",choices=["exact","ea","mfea"])
    p.add_argument("graphs",nargs="+",help="JSON or official .idpc files")
    p.add_argument("--population",type=int,default=100)
    p.add_argument("--generations",type=int,default=500)
    p.add_argument("--seed",type=int,default=7)
    p.add_argument("--runs",type=int,default=1)
    a=p.parse_args()
    if a.runs < 1:
        p.error("--runs must be >= 1")
    graphs=[Graph.from_file(x) for x in a.graphs]
    if a.algorithm=="exact":
        for name,g in zip(a.graphs,graphs):
            cost,path=exact_small(g)
            print(name,{"cost":cost,"edges":path})
        return
    label=a.algorithm.upper()
    all_cost=[]
    for i in range(a.runs):
        kwargs={"population":a.population,"generations":a.generations}
        if a.algorithm == "ea":
            results = [solve_ea(g, seed=a.seed+i, **kwargs) for g in graphs]
            best = [result[0][0] for result in results]
            last_generation = tuple(result[1][-1][0] for result in results)
        else:
            best,history=solve_mfea(graphs,seed=a.seed+i,**kwargs)
            last_generation = history[-1]
        for g,(cost,path) in zip(graphs,best):
            if math.isfinite(cost):
                assert math.isclose(validate_path(g,path),cost)
        print(label,[(x[0],x[1]) for x in best],"last-gen",last_generation)
        all_cost.append([x[0] for x in best])
    if a.runs>1:
        print("Mean per task:",[statistics.mean(v) for v in zip(*all_cost)])

if __name__=="__main__":main()
