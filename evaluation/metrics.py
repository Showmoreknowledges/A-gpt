"""Ranking-based evaluation metrics for graph alignment."""

from __future__ import annotations

from typing import Dict, Iterable, Sequence

import numpy as np


def _validate_inputs(rank_lists: Sequence[Sequence[int]], gt_indices: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    ranks = np.asarray(rank_lists)
    gt = np.asarray(gt_indices)
    if ranks.ndim != 2:
        raise ValueError("rank_lists must be a 2-D array-like with shape (num_queries, list_len)")
    if gt.ndim != 1:
        raise ValueError("gt_indices must be a 1-D array-like of ground-truth indices")
    if ranks.shape[0] != gt.shape[0]:
        raise ValueError(
            "rank_lists and gt_indices must contain the same number of queries. "
            f"Got {ranks.shape[0]} vs {gt.shape[0]}."
        )
    return ranks, gt


def compute_all_metrics(
    rank_lists: Sequence[Sequence[int]],
    gt_indices: Sequence[int],
    ks: Iterable[int] = (1, 5, 10),
) -> Dict[str, float]:
    """Compute Hit@k for several ``k`` values and overall MRR."""

    ranks, gt = _validate_inputs(rank_lists, gt_indices)
    if ranks.size == 0:
        raise ValueError("rank_lists is empty; unable to compute metrics")

    num_queries, list_len = ranks.shape
    eq_matrix = ranks == gt[:, None]

    metrics: Dict[str, float] = {}
    for k in ks:
        if k <= 0:
            continue
        k_eff = min(int(k), list_len)
        hits = eq_matrix[:, :k_eff].any(axis=1).astype(np.float32)
        metrics[f"hit@{k_eff}"] = float(hits.mean())

    first_pos = eq_matrix.argmax(axis=1)
    found = eq_matrix.any(axis=1)
    rr = np.zeros(num_queries, dtype=np.float32)
    rr[found] = 1.0 / (first_pos[found] + 1)
    metrics["mrr"] = float(rr.mean())
    return metrics