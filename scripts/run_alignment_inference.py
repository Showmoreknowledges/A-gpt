"""Compute H_a/H_b, run coarse retrieval, and optionally LLM reranking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.data.dataset_loader import load_graph_pair
from src.llm.reranker import LLMReranker, rerank_candidate_lists
from src.llm.runtime import build_llm_client
from src.models.fusion import GatedFusion
from src.models.gnn_encoder import GNNEncoder
from src.models.representation_pipeline import (
    compute_pair_embeddings,
    load_description_records,
)
from src.models.retrieval import Retriever
from src.models.text_encoder import TextEncoder


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz-path", type=Path, required=True, help="Path to the *.npz dataset")
    parser.add_argument(
        "--desc-a",
        type=Path,
        default=Path("cache/descriptions/layer_a.jsonl"),
        help="JSONL file with descriptions for graph A",
    )
    parser.add_argument(
        "--desc-b",
        type=Path,
        default=Path("cache/descriptions/layer_b.jsonl"),
        help="JSONL file with descriptions for graph B",
    )
    parser.add_argument(
        "--text-model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer checkpoint used for encoding descriptions",
    )
    parser.add_argument("--gnn-hidden", type=int, default=256, help="Hidden dimension of the GNN MLP")
    parser.add_argument("--gnn-out", type=int, default=256, help="Output dimension of the GNN features")
    parser.add_argument(
        "--fusion-dim",
        type=int,
        default=256,
        help="Output dimension of the gated fusion layer",
    )
    parser.add_argument("--text-strategy", choices=["concat", "sum"], default="concat")
    parser.add_argument("--struct-weight", type=float, default=0.5)
    parser.add_argument("--sem-weight", type=float, default=0.5)
    parser.add_argument("--topk", type=int, default=50, help="Number of coarse candidates to keep")
    parser.add_argument("--retriever-alpha", type=float, default=1.0)
    parser.add_argument("--retriever-beta", type=float, default=0.5)
    parser.add_argument("--retriever-gamma", type=float, default=0.5)
    parser.add_argument("--retriever-delta", type=float, default=0.5)
    parser.add_argument(
        "--skip-rerank",
        action="store_true",
        help="Disable the LLM reranker and return coarse rank lists",
    )
    parser.add_argument(
        "--rerank-backend",
        choices=["echo", "openai", "deepseek", "hf-local"],
        default="echo",
    )
    parser.add_argument("--rerank-model", type=str, default="gpt-4o-mini")
    parser.add_argument("--rerank-api-key", type=str, default=None)
    parser.add_argument("--rerank-base-url", type=str, default=None)
    parser.add_argument("--rerank-echo-suffix", type=str, default="")
    parser.add_argument("--rerank-hf-model", type=str, default=None)
    parser.add_argument("--rerank-hf-task", type=str, default="text-generation")
    parser.add_argument("--rerank-hf-device", type=str, default=None)
    parser.add_argument("--rerank-max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--save-npz",
        type=Path,
        default=Path("cache/rank_lists_final.npz"),
        help="Where to save the resulting rank lists",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default=None,
        help="Optional JSON string stored alongside the rank lists",
    )
    return parser.parse_args()


def _gnn_for_layer(features, hidden: int, out: int) -> GNNEncoder:
    in_dim = features.shape[1]
    return GNNEncoder(in_dim=in_dim, hidden_dim=hidden, out_dim=out)


def _build_reranker(args: argparse.Namespace) -> LLMReranker:
    client = build_llm_client(
        args.rerank_backend,
        api_key=args.rerank_api_key,
        base_url=args.rerank_base_url,
        echo_suffix=args.rerank_echo_suffix,
        hf_model=args.rerank_hf_model or args.rerank_model,
        hf_task=args.rerank_hf_task,
        hf_device=args.rerank_hf_device,
        generation_kwargs={"max_new_tokens": args.rerank_max_new_tokens},
    )
    return LLMReranker(client=client, model=args.rerank_model)


def _save_rank_lists(path: Path, ranks: np.ndarray, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, rank_lists=ranks, metadata=json.dumps(metadata, ensure_ascii=False))


def main() -> None:
    args = _parse_args()
    torch.set_grad_enabled(False)

    graph_pair = load_graph_pair(args.npz_path)
    descriptions_a = load_description_records(args.desc_a)
    descriptions_b = load_description_records(args.desc_b)

    text_encoder = TextEncoder(args.text_model)
    gnn_a = _gnn_for_layer(graph_pair.graph_a.features, args.gnn_hidden, args.gnn_out)
    gnn_b = _gnn_for_layer(graph_pair.graph_b.features, args.gnn_hidden, args.gnn_out)

    text_dim = text_encoder.output_dim(strategy=args.text_strategy)
    fusion_a = GatedFusion(dim_struct=args.gnn_out, dim_text=text_dim, dim_out=args.fusion_dim)
    fusion_b = GatedFusion(dim_struct=args.gnn_out, dim_text=text_dim, dim_out=args.fusion_dim)

    with torch.no_grad():
        h_a, h_b = compute_pair_embeddings(
            graph_pair,
            descriptions_a,
            descriptions_b,
            gnn_encoder_a=gnn_a,
            gnn_encoder_b=gnn_b,
            text_encoder=text_encoder,
            fusion_a=fusion_a,
            fusion_b=fusion_b,
            text_strategy=args.text_strategy,
            struct_text_weight=args.struct_weight,
            sem_text_weight=args.sem_weight,
        )

    retriever = Retriever(
        alpha=args.retriever_alpha,
        beta=args.retriever_beta,
        gamma=args.retriever_gamma,
        delta=args.retriever_delta,
    )
    topk = min(args.topk, h_b.shape[0])
    coarse_idx, _ = retriever.topk_candidates(h_a, h_b, topk)
    rank_lists = coarse_idx.cpu().numpy()

    if not args.skip_rerank:
        reranker = _build_reranker(args)
        final_lists = rerank_candidate_lists(
            descriptions_a,
            descriptions_b,
            rank_lists.tolist(),
            reranker,
        )
        rank_lists = np.asarray(final_lists, dtype=np.int64)

    metadata = {
        "npz_path": str(args.npz_path),
        "desc_a": str(args.desc_a),
        "desc_b": str(args.desc_b),
        "text_model": args.text_model,
        "topk": int(rank_lists.shape[1]),
        "rerank": not args.skip_rerank,
    }
    if args.metadata:
        metadata.update(json.loads(args.metadata))

    _save_rank_lists(args.save_npz, rank_lists, metadata)

    print("=== Inference Summary ===")
    print(f"Nodes A: {rank_lists.shape[0]} | topk: {rank_lists.shape[1]}")
    print(f"Descriptions: {args.desc_a} / {args.desc_b}")
    print(f"Embeddings saved to: {args.save_npz}")


if __name__ == "__main__":
    main()