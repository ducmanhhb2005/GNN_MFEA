"""Load GNN checkpoint and predict a score for each graph edge."""
from __future__ import annotations
from idpc_du.graph import Graph
from .model import require_torch, build_model
from .features import graph_features

def load_edge_scorer(checkpoint: str):
    torch, _ = require_torch()
    pack=torch.load(checkpoint,map_location="cpu",weights_only=True)
    model=build_model(int(pack["hidden"]))
    model.load_state_dict(pack["model_state"])
    model.eval()
    return model


def score_edges(model, graph: Graph) -> list[float]:
    torch, _ = require_torch()
    features=graph_features(graph)
    with torch.no_grad():
        scores=torch.sigmoid(model(*features)).cpu().tolist()
    return [float(s) for s in scores]
