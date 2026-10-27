"""Paired-seed ablation: baseline random initialization vs GNN-guided P0.

Does not change MFEA evolution or GPA. Model must be pretrained from a separate
training dataset. This experiment makes no claim of statistical significance.
"""
import argparse, csv, json, math, statistics, time
from pathlib import Path
from idpc_du import Graph, validate_path
from mfea.solver import solve_mfea
from integration.solver import solve_guided_mfea
from integration.guided_initialization import make_guided_population
from gnn.inference import load_edge_scorer, score_edges

p=argparse.ArgumentParser()
p.add_argument('tasks', nargs='+', help='Two or more .idpc or .json graph files')
p.add_argument('--checkpoint',default='models/edge_gnn_synthetic.pt')
p.add_argument('--methods',default='baseline,heuristic,guided',help='Comma-separated methods: baseline,heuristic,guided')
p.add_argument('--population',type=int,default=100)
p.add_argument('--generations',type=int,default=500)
p.add_argument('--seed',type=int,default=100)
p.add_argument('--runs',type=int,default=30)
p.add_argument('--guided-fraction',type=float,default=0.5)
p.add_argument('--jitter',type=float,default=0.04)
p.add_argument('--output',default='experiments/latest_gnn.csv')
args=p.parse_args()
if len(args.tasks)<2: p.error('use >=2 tasks to evaluate multitasking')
if args.runs<1:p.error('runs must be >=1')
graphs=[Graph.from_file(path) for path in args.tasks]
def heuristic_scores(graph):
    # Control condition: cheap edges have high scores; NO learned GNN.
    return [1.0/(1.0+edge.weight) for edge in graph.edges]

methods=[m.strip() for m in args.methods.split(',') if m.strip()]
if not methods or len(set(methods)) != len(methods) or any(m not in ('baseline','heuristic','guided') for m in methods):
    p.error('methods must be a unique subset of baseline,heuristic,guided')
if 'guided' in methods:
    model=load_edge_scorer(args.checkpoint)
    scores=[score_edges(model,g) for g in graphs]
else:
    scores=[]
rows=[]
for trial in range(args.runs):
    seed=args.seed+trial
    for mode in methods:
        init=None
        meta={}
        if mode in ('guided','heuristic'):
            vectors = scores if mode == 'guided' else [heuristic_scores(g) for g in graphs]
            init,meta=make_guided_population(graphs,vectors,args.population,seed,
                                guided_fraction=args.guided_fraction,jitter=args.jitter)
        start=time.perf_counter()
        solver = solve_mfea if init is None else solve_guided_mfea
        initialization = {} if init is None else {"initial_population": init}
        best,history=solver(graphs,seed=seed,population=args.population,
              generations=args.generations,**initialization)
        elapsed=time.perf_counter()-start
        for task_idx,(g,(cost,path)) in enumerate(zip(graphs,best)):
            valid=math.isfinite(cost) and validate_path(g,path)==cost
            if math.isfinite(cost) and not valid:raise AssertionError('Invalid returned path')
            rows.append({'seed':seed,'method':mode,'task':args.tasks[task_idx],
                'cost':cost,'feasible':valid,'seconds':round(elapsed,4),
                'population':args.population,'generations':args.generations,
                'decoder':'algorithm_1',
                'guided_fraction':args.guided_fraction if mode=='guided' else 0.0,
                'checkpoint':args.checkpoint if mode=='guided' else '',
                'init_allocation':str(meta.get('guided_task_allocations',''))})
        print(f'{mode:8} seed={seed:<4} cost={[round(z[0],4) for z in best]} {elapsed:.2f}s',flush=True)
Path(args.output).parent.mkdir(parents=True,exist_ok=True)
with open(args.output,'w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
for task in args.tasks:
    for mode in methods:
        arr=[r['cost'] for r in rows if r['task']==task and r['method']==mode]
        vals=[v for v in arr if math.isfinite(v)]
        print(f'{Path(task).name} {mode}: valid={len(vals)}/{len(arr)}, best={min(vals,default=math.inf)}, mean_valid={statistics.mean(vals) if vals else math.inf:.4f}')
print(f'CSV: {args.output}')
