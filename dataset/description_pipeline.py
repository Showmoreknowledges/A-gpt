"""Generate and cache LLM-based descriptions for graph nodes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from ..llm.prompt_templates import make_semantic_desc_prompt, make_struct_desc_prompt
from ..llm.providers import LLMClient
from .schema import NodeProperty


def _ensure_lengths(node_props: Sequence[NodeProperty], node_texts: Sequence[str] | None) -> None:
    if node_texts is not None and len(node_texts) != len(node_props):
        raise ValueError(
            "node_props and node_texts must have the same length when semantic descriptions are requested"
        )


def generate_descriptions_for_graph(
    node_props: Sequence[NodeProperty],
    node_texts: Sequence[str] | None,
    client: LLMClient,
    model: str,
    save_path: str | Path,
) -> None:
    """Generate structural/semantic descriptions for every node and cache to disk."""

    _ensure_lengths(node_props, node_texts)
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for idx, prop in enumerate(node_props):
            prompt_struct = make_struct_desc_prompt(prop)
            struct_desc = client.complete(model=model, prompt=prompt_struct)

            sem_desc = ""
            if node_texts is not None:
                prompt_sem = make_semantic_desc_prompt(
                    raw_text=node_texts[idx], graph_meta={"graph_type": prop.graph_type}
                )
                sem_desc = client.complete(model=model, prompt=prompt_sem)

            record = {
                "id": idx,
                "struct_desc": struct_desc,
                "sem_desc": sem_desc,
            }
            cross_layer = prop.cross_layer
            if cross_layer is not None:
                record["hop_token"] = cross_layer.hop_token
                record["hop_distance"] = cross_layer.hop_distance
                record["nearest_anchor"] = cross_layer.nearest_anchor
                if cross_layer.path_to_anchor:
                    record["path_to_anchor"] = cross_layer.path_to_anchor
                if cross_layer.path_struct_profile:
                    record["path_struct_profile"] = cross_layer.path_struct_profile
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
