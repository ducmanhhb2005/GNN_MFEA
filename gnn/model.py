"""PyTorch-only edge-scoring GNN architecture."""
from __future__ import annotations


def require_torch():
    try:
        import torch
        from torch import nn
    except ImportError as e:
        raise RuntimeError("GNN requires PyTorch. Install pip install torch") from e
    return torch, nn


def build_model(hidden: int = 32):
    torch, nn = require_torch()
    class EdgeGNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.node_input = nn.Linear(5, hidden)
            self.message1 = nn.Linear(hidden+6, hidden)
            self.update1 = nn.Linear(hidden*2, hidden)
            self.message2 = nn.Linear(hidden+6, hidden)
            self.update2 = nn.Linear(hidden*2, hidden)
            self.out = nn.Sequential(nn.Linear(hidden*2+6, hidden), nn.ReLU(),
                                     nn.Linear(hidden, 1))

        def propagate(self, h, ef, src, dst, message, update):
            msg = torch.relu(message(torch.cat((h[src], ef), dim=1)))
            accum = h.new_zeros(h.shape)
            accum.index_add_(0, dst, msg)
            count = h.new_zeros(h.shape[0], 1)
            count.index_add_(0, dst, h.new_ones(dst.shape[0], 1))
            mean = accum / count.clamp_min(1.)
            return torch.relu(update(torch.cat((h, mean), dim=1)))

        def forward(self, x, ef, src, dst):
            h = torch.relu(self.node_input(x))
            h = self.propagate(h, ef, src, dst, self.message1, self.update1)
            h = self.propagate(h, ef, src, dst, self.message2, self.update2)
            return self.out(torch.cat((h[src], h[dst], ef), dim=1)).squeeze(-1)
    return EdgeGNN()
