"""Generate structural (and optional semantic) descriptions for both graph layers."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Sequence

from dataset.cross_layer import compute_cross_layer_contexts
from dataset.dataset_loader import load_graph_pair
from dataset.description_pipeline import generate_descriptions_for_graph
from dataset.node_property import compute_node_properties
from dataset.schema import NodeProperty
from LLM.providers import LLMClient
from LLM.runtime import build_llm_client
from special_tokens import collect_hop_tokens


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz-path", type=Path, required=True, help="Path to the *.npz bundle")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("cache/descriptions"),
        help="Directory to store the generated JSONL files",
    )
    parser.add_argument(
        "--graph-a-label",
        type=str,
        default="graph_a",
        help="Logical name for the first graph layer (used in prompts)",
    )
    parser.add_argument(
        "--graph-b-label",
        type=str,
        default="graph_b",
        help="Logical name for the second graph layer",
    )
    parser.add_argument(
        "--text-a",
        type=Path,
        default=None,
        help="Optional text file containing one raw text per node for graph A",
    )
    parser.add_argument(
        "--text-b",
        type=Path,
        default=None,
        help="Optional text file containing one raw text per node for graph B",
    )
    parser.add_argument(
        "--llm-backend",
        choices=["echo", "openai", "deepseek", "hf-local"],
        default="echo",
        help="Backend used to materialize the LLM client",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="gpt-4o-mini",
        help="Model identifier passed to the selected backend",
    )
    parser.add_argument("--llm-api-key", type=str, default=None, help="API key for OpenAI/DeepSeek")
    parser.add_argument("--llm-base-url", type=str, default=None, help="Custom base URL for OpenAI-compatible APIs")
    parser.add_argument(
        "--llm-echo-suffix",
        type=str,
        default="",
        help="Suffix appended by the echo backend (useful for smoke tests)",
    )
    parser.add_argument(
        "--hf-model",
        type=str,
        default=None,
        help="Model name when using the hf-local backend (defaults to --llm-model)",
    )
    parser.add_argument(
        "--hf-task",
        type=str,
        default="text-generation",
        help="Transformers pipeline task name for hf-local",
    )
    parser.add_argument(
        "--hf-device",
        type=str,
        default=None,
        help="Device spec understood by transformers.pipeline (e.g., cpu, cuda:0)",
    )
    parser.add_argument(
        "--hf-max-new-tokens",
        type=int,
        default=256,
        help="max_new_tokens passed to the hf-local backend",
    )
    return parser.parse_args()


def _load_optional_texts(path: Optional[Path], expected: int) -> Optional[List[str]]:
    if path is None:
        return None
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) != expected:
        raise ValueError(
            f"Text file {path} contains {len(lines)} entries but {expected} nodes were detected."
        )
    return lines


def _build_client(args: argparse.Namespace, *, special_tokens: Sequence[str] | None) -> LLMClient:
    return build_llm_client(
        args.llm_backend,
        api_key=args.llm_api_key,
        base_url=args.llm_base_url,
        echo_suffix=args.llm_echo_suffix,
        hf_model=args.hf_model or args.llm_model,
        hf_task=args.hf_task,
        hf_device=args.hf_device,
        generation_kwargs={"max_new_tokens": args.hf_max_new_tokens},
        special_tokens=special_tokens,
    )


def _write_descriptions(
    save_path: Path,
    graph_label: str,
    node_props: List[NodeProperty],
    node_texts: Optional[List[str]],
    args: argparse.Namespace,
    client: LLMClient,
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    generate_descriptions_for_graph(
        node_props=node_props,
        node_texts=node_texts,
        client=client,
        model=args.llm_model,
        save_path=str(save_path),
    )
    print(f"✅ Wrote {len(node_props)} descriptions for {graph_label} -> {save_path}")


def main() -> None:
    args = _parse_args()
    graph_pair = load_graph_pair(args.npz_path)
    ctx_a, ctx_b = compute_cross_layer_contexts(graph_pair)
    hop_tokens = collect_hop_tokens(list(ctx_a) + list(ctx_b))
    client = _build_client(args, special_tokens=hop_tokens)

    props_a = compute_node_properties(graph_pair.graph_a, graph_type=args.graph_a_label)
    props_b = compute_node_properties(graph_pair.graph_b, graph_type=args.graph_b_label)

    for prop, ctx in zip(props_a, ctx_a):
        prop.cross_layer = ctx
    for prop, ctx in zip(props_b, ctx_b):
        prop.cross_layer = ctx

    texts_a = _load_optional_texts(args.text_a, len(props_a)) if args.text_a else None
    texts_b = _load_optional_texts(args.text_b, len(props_b)) if args.text_b else None

    out_a = args.output_dir / "layer_a.jsonl"
    out_b = args.output_dir / "layer_b.jsonl"

    _write_descriptions(out_a, args.graph_a_label, props_a, texts_a, args, client)
    _write_descriptions(out_b, args.graph_b_label, props_b, texts_b, args, client)

    print("🎯 Description generation finished.")


if __name__ == "__main__":
    main()
