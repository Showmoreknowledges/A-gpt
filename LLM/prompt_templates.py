"""Prompt helpers for turning :class:`dataset.schema.NodeProperty` into text."""

from __future__ import annotations

from typing import Mapping, Sequence, Tuple

from dataset.schema import CrossLayerInfo, NodeProperty
from special_tokens import HOP_UNKNOWN_TOKEN


def _cross_layer_hint(info: CrossLayerInfo | None) -> Tuple[str, str]:
    if info is None or info.hop_distance is None:
        hint = "- No reachable anchor was found; treat the hop level as unknown and reason from local evidence."
        return hint, HOP_UNKNOWN_TOKEN

    candidate_preview = ", ".join(str(n) for n in info.mirror_candidates[:5]) or "None"
    overflow = ""
    if len(info.mirror_candidates) > 5:
        overflow = f", ... (+{len(info.mirror_candidates) - 5} more)"
    hint = (
        f"- Hop distance to nearest anchor: {info.hop_distance} (token {info.hop_token})\n"
        f"- Mirrored anchor on paired graph: {info.mirror_anchor}\n"
        f"- Nodes at the same hop in the paired graph: [{candidate_preview}{overflow}]"
    )
    if info.path_to_anchor:
        path_desc = " -> ".join(str(x) for x in info.path_to_anchor)
        hint += f"\n- Shortest path to that anchor (within this layer): {path_desc}"
    return hint, info.hop_token


def make_struct_desc_prompt(prop: NodeProperty) -> str:
    s = prop.struct
    n = prop.neigh
    cross_hint, hop_token = _cross_layer_hint(prop.cross_layer)
    return f"""
You are an expert in network science and cross-layer graph alignment.

Below are structural statistics and neighborhood summary of a node:
- Graph type: {prop.graph_type}
- Degree: {s.degree}
- Clustering coefficient: {s.clustering}
- Square clustering coefficient: {s.square_clustering}
- Closeness centrality: {s.closeness}
- Top-k neighbors: {n.topk_neighbors}

Cross-layer alignment context:
{cross_hint}

Instructions:
- Begin the first sentence with the hop token {hop_token} so downstream models can read it directly.
- Describe the structural role of this node in 2-3 sentences and explicitly mention how the cross-layer hint (including the path) could guide alignment decisions.
""".strip()


def make_semantic_desc_prompt(raw_text: str, graph_meta: Mapping[str, str]) -> str:
    return f"""
You are an expert at summarizing user/item descriptions for graph alignment.

Graph type: {graph_meta.get("graph_type", "unknown")}

Original text:
\"\"\"{raw_text}\"\"\"

Provide a concise 1-2 sentence semantic summary. If relevant, reference any <hop-k> style structural cues mentioned above to keep the summary aligned with structural context.
""".strip()


def make_rerank_prompt(source_desc: str, candidate_descs: Sequence[str]) -> str:
    candidates = "\n".join(f"{idx}. {desc}" for idx, desc in enumerate(candidate_descs, start=1))
    return f"""
You are aligning a source node to the correct target node across two graphs.
Each description may contain special <hop-k> tokens that indicate the hop distance to the nearest anchor; smaller hop values mean the node is closer to reliable cross-layer anchors.

Source node description:
{source_desc}

Candidate node descriptions:
{candidates}

Rank the candidates from best to worst match considering both semantics and the <hop-k> structural cues. Reply strictly in the form:
Ranking: X > Y > Z
""".strip()
