"""End-to-end helpers that combine structural/text encoders into node embeddings."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, MutableMapping, Sequence

import torch

from ..data.dataset_loader import GraphLayer, GraphPair
from .fusion import GatedFusion
from .gnn_encoder import GNNEncoder
from .text_encoder import TextEncoder

DescriptionRecord = MutableMapping[str, str]


def load_description_records(path: str | Path) -> list[DescriptionRecord]:
    """Load cached node descriptions produced by the LLM stage."""

    records: list[DescriptionRecord] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _features_to_tensor(layer: GraphLayer) -> torch.Tensor:
    if layer.features is None:
        raise ValueError("GraphLayer is missing node features.")
    features = torch.as_tensor(layer.features, dtype=torch.float32)
    return features


def _edge_index_tensor(layer: GraphLayer) -> torch.Tensor:
    edges = list(layer.graph.edges())
    if not edges:
        return torch.zeros((2, 0), dtype=torch.long)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index


def _texts_from_descriptions(descriptions: Sequence[Mapping[str, str]]) -> tuple[list[str], list[str]]:
    struct_texts: list[str] = []
    sem_texts: list[str] = []
    for record in descriptions:
        struct_texts.append(str(record.get("struct_desc", "")))
        sem_texts.append(str(record.get("sem_desc", "")))
    return struct_texts, sem_texts


def _encode_text_features(
    descriptions: Sequence[Mapping[str, str]],
    text_encoder: TextEncoder,
    *,
    strategy: str = "concat",
    struct_weight: float = 0.5,
    sem_weight: float = 0.5,
) -> torch.Tensor:
    struct_texts, sem_texts = _texts_from_descriptions(descriptions)
    struct_emb = text_encoder.encode(struct_texts)
    sem_emb = text_encoder.encode(sem_texts)
    if strategy == "concat":
        return torch.cat([struct_weight * struct_emb, sem_weight * sem_emb], dim=-1)
    if strategy == "sum":
        return struct_weight * struct_emb + sem_weight * sem_emb
    raise ValueError("strategy must be either 'concat' or 'sum'.")


def compute_layer_embeddings(
    layer: GraphLayer,
    descriptions: Sequence[Mapping[str, str]],
    *,
    gnn_encoder: GNNEncoder,
    text_encoder: TextEncoder,
    fusion_module: GatedFusion,
    text_strategy: str = "concat",
    struct_text_weight: float = 0.5,
    sem_text_weight: float = 0.5,
) -> torch.Tensor:
    """Compute fused node embeddings for a single graph layer."""

    num_nodes = layer.num_nodes()
    if len(descriptions) != num_nodes:
        raise ValueError(
            "Number of descriptions does not match node count: "
            f"{len(descriptions)} vs {num_nodes}."
        )

    x = _features_to_tensor(layer)
    edge_index = _edge_index_tensor(layer)
    h_struct = gnn_encoder(x, edge_index)
    if h_struct.shape[0] != num_nodes:
        raise ValueError("GNN encoder output does not align with node count.")

    h_text = _encode_text_features(
        descriptions,
        text_encoder,
        strategy=text_strategy,
        struct_weight=struct_text_weight,
        sem_weight=sem_text_weight,
    )
    if h_text.shape[0] != num_nodes:
        raise ValueError("Text encoder output does not align with node count.")

    fused = fusion_module(h_struct, h_text)
    return fused


def compute_pair_embeddings(
    graph_pair: GraphPair,
    descriptions_a: Sequence[Mapping[str, str]],
    descriptions_b: Sequence[Mapping[str, str]],
    *,
    gnn_encoder_a: GNNEncoder,
    gnn_encoder_b: GNNEncoder | None = None,
    text_encoder: TextEncoder,
    fusion_a: GatedFusion,
    fusion_b: GatedFusion | None = None,
    text_strategy: str = "concat",
    struct_text_weight: float = 0.5,
    sem_text_weight: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute ``H_a`` and ``H_b`` embeddings for both graph layers."""

    gnn_encoder_b = gnn_encoder_b or gnn_encoder_a
    fusion_b = fusion_b or fusion_a

    h_a = compute_layer_embeddings(
        graph_pair.graph_a,
        descriptions_a,
        gnn_encoder=gnn_encoder_a,
        text_encoder=text_encoder,
        fusion_module=fusion_a,
        text_strategy=text_strategy,
        struct_text_weight=struct_text_weight,
        sem_text_weight=sem_text_weight,
    )
    h_b = compute_layer_embeddings(
        graph_pair.graph_b,
        descriptions_b,
        gnn_encoder=gnn_encoder_b,
        text_encoder=text_encoder,
        fusion_module=fusion_b,
        text_strategy=text_strategy,
        struct_text_weight=struct_text_weight,
        sem_text_weight=sem_text_weight,
    )
    return h_a, h_b