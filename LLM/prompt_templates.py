"""Prompt helpers for turning :class:`~src.data.schema.NodeProperty` into text."""

from __future__ import annotations

from ..data.schema import NodeProperty


def make_struct_desc_prompt(prop: NodeProperty) -> str:
    s = prop.struct
    n = prop.neigh
    return f"""
You are an expert in network science.

Below are structural statistics and neighborhood summary of a node in a graph:

- Graph type: {prop.graph_type}
- Degree: {s.degree}
- Clustering coefficient: {s.clustering}
- Square clustering coefficient: {s.square_clustering}
- Closeness centrality: {s.closeness}
- Top-k neighbors: {n.topk_neighbors}

Describe the structural role of this node in 2-3 sentences.
""".strip()


def make_semantic_desc_prompt(raw_text: str, graph_meta: dict) -> str:
    return f"""
You are an expert at summarizing user/item descriptions.

Graph type: {graph_meta.get("graph_type", "unknown")}

Original text:
\"\"\"{raw_text}\"\"\"

Please give a concise semantic summary in 1-2 sentences.
""".strip()