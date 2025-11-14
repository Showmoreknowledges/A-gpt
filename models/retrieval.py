"""Similarity-based coarse retrieval from fused node embeddings."""

from __future__ import annotations
from typing import Mapping
import torch


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute cosine similarities between every pair of rows in ``a`` and ``b``."""

    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("cosine_sim expects 2D tensors (num_nodes, dim).")
    if a.size(1) != b.size(1):
        raise ValueError("Input tensors must share the embedding dimension.")

    def _normalize(x: torch.Tensor) -> torch.Tensor:
        return x / (x.norm(dim=-1, keepdim=True).clamp_min(1e-8))

    a_norm = _normalize(a)
    b_norm = _normalize(b)
    return a_norm @ b_norm.T


class Retriever:
    """Combine multiple similarity channels to build coarse candidate lists."""

    def __init__(self, alpha: float = 1.0, beta: float = 0.5, gamma: float = 0.5, delta: float = 0.5):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

    def _get_extra(
        self,
        name: str,
        reference: torch.Tensor,
        extra_sims: Mapping[str, torch.Tensor] | None,
    ) -> torch.Tensor:
        if not extra_sims or name not in extra_sims:
            return torch.zeros_like(reference)
        value = extra_sims[name]
        if not torch.is_tensor(value):
            value = torch.as_tensor(value)
        return value

    def compute_scores(
        self,
        h_a: torch.Tensor,
        h_b: torch.Tensor,
        extra_sims: Mapping[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        base = cosine_sim(h_a, h_b)
        scores = self.alpha * base

        if extra_sims:
            neighbor = self._get_extra("neighbor", base, extra_sims)
            attr = self._get_extra("attr", base, extra_sims)
            text = self._get_extra("text", base, extra_sims)

            for tensor in (neighbor, attr, text):
                if tensor.shape != base.shape:
                    raise ValueError("Extra similarity matrices must match cosine map shape.")

            scores = scores + self.beta * neighbor + self.gamma * attr + self.delta * text

        return scores

    def topk_candidates(
        self,
        h_a: torch.Tensor,
        h_b: torch.Tensor,
        k: int,
        extra_sims: Mapping[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if k <= 0:
            raise ValueError("k must be positive for top-k retrieval.")

        scores = self.compute_scores(h_a, h_b, extra_sims=extra_sims)
        topk_scores, topk_indices = scores.topk(k, dim=-1)
        return topk_indices, topk_scores