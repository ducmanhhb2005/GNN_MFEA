"""Train toy GNN edge scorer on synthetic graphs. Research prototype only."""
from __future__ import annotations
import math
import random
from pathlib import Path
from idpc_du.exact import exact_small
from .model import require_torch, build_model
from .features import graph_features
from .synthetic_data import generate_training_graph

def train_synthetic(checkpoint: str, *, seed: int = 3, graph_count: int = 80,
                    epochs: int = 16, lr: float = 0.002, hidden: int = 32) -> dict:
    torch, nn = require_torch()
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng = random.Random(seed)
    if graph_count < 8 or epochs < 1:
        raise ValueError("graph_count >= 8 and epochs >= 1")
    dataset = []
    for _ in range(graph_count):
        graph = generate_training_graph(rng)
        cost, path = exact_small(graph)
        if not math.isfinite(cost):
            raise AssertionError("synthetic graph should have feasible path")
        label = torch.tensor([1. if e.id in set(path) else 0. for e in graph.edges],dtype=torch.float32)
        dataset.append((graph_features(graph), label))
    # Held-out synthetic split; no benchmark data or benchmark optima used in training.
    rng.shuffle(dataset)
    split = max(1, int(graph_count*0.8))
    training, validation = dataset[:split], dataset[split:]
    model = build_model(hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    last_train_loss = 0.
    for _ in range(epochs):
        rng.shuffle(training)
        model.train()
        losses=[]
        for feat,y in training:
            logits=model(*feat)
            count_pos=float(y.sum().item())
            pos_weight=torch.tensor(min(12.,max(1.,(len(y)-count_pos)/max(1.,count_pos))))
            loss=nn.functional.binary_cross_entropy_with_logits(logits,y,pos_weight=pos_weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        last_train_loss=sum(losses)/len(losses)
    model.eval()
    with torch.no_grad():
        val=[]
        for feat,y in validation:
            val.append(nn.functional.binary_cross_entropy_with_logits(model(*feat),y).item())
    Path(checkpoint).parent.mkdir(parents=True,exist_ok=True)
    torch.save({"model_state":model.state_dict(),"hidden":hidden,
                "metadata":{"seed":seed,"graph_count":graph_count,"epochs":epochs,
                            "source":"SYNTHETIC_EXACT_PATH_LABELS", "not_paper_reproduction":True}},checkpoint)
    return {"train_loss":round(last_train_loss,5),"val_bce":round(sum(val)/len(val),5),
            "train_graphs":len(training),"heldout_graphs":len(validation),"checkpoint":checkpoint}
