"""Train a supervised toy-edge scorer on synthetic IDPC-DU graphs only.

Run from repository root: python -m experiments.train_gnn
"""
import argparse, json
from gnn.train import train_synthetic

p=argparse.ArgumentParser(description='Toy GNN proof-of-concept: NOT a validated research model')
p.add_argument('--checkpoint',default='models/edge_gnn_synthetic.pt')
p.add_argument('--graphs',type=int,default=80)
p.add_argument('--epochs',type=int,default=16)
p.add_argument('--seed',type=int,default=3)
p.add_argument('--hidden',type=int,default=32)
p.add_argument('--lr',type=float,default=0.002)
args=p.parse_args()
print(json.dumps(train_synthetic(args.checkpoint,seed=args.seed,graph_count=args.graphs,
  epochs=args.epochs,hidden=args.hidden,lr=args.lr),indent=2,ensure_ascii=False))
