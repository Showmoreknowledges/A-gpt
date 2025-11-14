"""Fusion utilities for combining structural/text embeddings."""

from __future__ import annotations

import torch
import torch.nn as nn


class GatedFusion(nn.Module):
    def __init__(self, dim_struct: int, dim_text: int, dim_out: int) -> None:
        super().__init__()
        self.proj_struct = nn.Linear(dim_struct, dim_out)
        self.proj_text = nn.Linear(dim_text, dim_out)
        self.gate = nn.Linear(dim_struct + dim_text, dim_out)

    def forward(self, h_struct: torch.Tensor, h_text: torch.Tensor) -> torch.Tensor:
        hs = self.proj_struct(h_struct)
        ht = self.proj_text(h_text)
        gate_inp = torch.cat([h_struct, h_text], dim=-1)
        g = torch.sigmoid(self.gate(gate_inp))
        return g * hs + (1 - g) * ht