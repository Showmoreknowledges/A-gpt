"""End-to-end evaluation script for the graph-alignment pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..data.dataset_loader import load_graph_pair
from .logger import log_metrics_to_csv
from .metrics import compute_all_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate graph alignment results")
    parser.add_argument("--npz_path", type=str, required=True, help="Path to the *.npz dataset")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Logical dataset name for logging (e.g., douban, amazon_clothing)",
    )
    parser.add_argument(
        "--log_csv",
        type=str,
        default="results/metrics_log.csv",
        help="Where to append the evaluation metrics",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="gnn_fusion_llm",
        help="Human-readable identifier for the evaluated model",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=50,
        help="Candidate list length produced by the alignment pipeline",
    )
    parser.add_argument(
        "--rank_path",
        type=str,
        default=None,
        help="Optional path to the rank_lists_final npz produced by the inference step",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed when falling back to random rankings")
    return parser.parse_args()


def _placeholder_rankings(num_queries: int, num_nodes_b: int, topk: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed=seed)
    rank_lists = []
    for _ in range(num_queries):
        perm = rng.permutation(num_nodes_b)[:topk]
        rank_lists.append(perm)
    return np.stack(rank_lists, axis=0)


def _load_rank_lists(path: str | None) -> np.ndarray | None:
    if path is None:
        return None
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(resolved)
    data = np.load(resolved, allow_pickle=True)
    if isinstance(data, np.ndarray):
        ranks = data
    else:
        if "rank_lists" not in data.files:
            raise KeyError("npz file must contain 'rank_lists'")
        ranks = data["rank_lists"]
    return np.asarray(ranks, dtype=np.int64)


def main() -> None:
    args = parse_args()
    graph_pair = load_graph_pair(args.npz_path)

    aligned = graph_pair.aligned_pairs
    if aligned.size == 0:
        raise ValueError("The provided dataset does not contain any aligned node pairs.")

    gt_target = aligned[:, 1]
    gt_source = aligned[:, 0]
    num_queries = aligned.shape[0]

    num_nodes_b = graph_pair.graph_b.num_nodes()
    if num_nodes_b == 0:
        raise ValueError("Graph B has no nodes; cannot form ranking lists.")
    topk = max(1, min(int(args.topk), num_nodes_b))

    rank_lists = _load_rank_lists(args.rank_path)
    if rank_lists is not None:
        if rank_lists.shape[0] <= int(gt_source.max()):
            raise ValueError(
                "rank_lists array does not cover all source node indices present in aligned pairs"
            )
        if rank_lists.shape[1] < topk:
            print(
                "⚠️ Provided rank list topk smaller than requested, truncating to available length."
            )
        rank_lists_final = rank_lists[gt_source]
        rank_lists_final = rank_lists_final[:, : min(rank_lists_final.shape[1], topk)]
        rank_source = f"Loaded from {args.rank_path}"
    else:
        rank_lists_final = _placeholder_rankings(num_queries, num_nodes_b, topk, args.seed)
        rank_source = "Random baseline (provide --rank_path to evaluate model outputs)"

    metrics = compute_all_metrics(rank_lists_final, gt_target, ks=(1, 5, 10))

    print("=== Evaluation ===")
    print(f"Dataset: {args.dataset}")
    print(f"Queries: {num_queries}")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    print(f"Rank source: {rank_source}")

    log_metrics_to_csv(
        save_path=args.log_csv,
        dataset_name=args.dataset,
        metrics=metrics,
        extra_info={
            "npz_path": args.npz_path,
            "model": args.model_name,
            "topk": topk,
            "num_queries": num_queries,
            "rank_path": args.rank_path,
            "rank_source": rank_source,
        },
    )
    print(f"Metrics appended to: {args.log_csv}")


if __name__ == "__main__":
    main()