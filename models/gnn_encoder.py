"""Lightweight structural encoder placeholder for graph data."""

from __future__ import annotations

import torch
import torch.nn as nn


class GNNEncoder(nn.Module):
    """Minimal GNN-style encoder (currently implemented as an MLP placeholder)."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor | None = None) -> torch.Tensor:
        # ``edge_index`` is ignored for now (placeholder for a real GNN implementation).
        return self.net(x)