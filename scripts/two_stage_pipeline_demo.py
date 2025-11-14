"""CLI demo for the first two pipeline stages (data + node features)."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data.dataset_loader import load_graph_pair
from src.data.node_property import compute_node_properties


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset",
        type=Path,
        default=Path("data/ACM-DBLP.npz"),
        nargs="?",
        help="Path to the *.npz dataset that contains two-layer graphs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    graph_pair = load_graph_pair(args.dataset)
    props_a = compute_node_properties(graph_pair.graph_a, graph_type="graph_a")
    props_b = compute_node_properties(graph_pair.graph_b, graph_type="graph_b")

    dataset_label = Path(graph_pair.metadata["path"]).name
    print(f"Loaded dataset: {dataset_label}")
    print(f"Aligned pairs: {graph_pair.num_aligned_pairs()}")
    print("Graph A -> nodes:{:d} edges:{:d}".format(
        graph_pair.graph_a.graph.number_of_nodes(),
        graph_pair.graph_a.graph.number_of_edges(),
    ))
    print("Graph B -> nodes:{:d} edges:{:d}".format(
        graph_pair.graph_b.graph.number_of_nodes(),
        graph_pair.graph_b.graph.number_of_edges(),
    ))

    for name, node_props in (("Graph A", props_a), ("Graph B", props_b)):
        if not node_props:
            print(f"{name} stats -> graph is empty")
            continue
        avg_degree = sum(p.struct.degree for p in node_props) / len(node_props)
        avg_cluster = sum(p.struct.clustering for p in node_props) / len(node_props)
        avg_close = sum(p.struct.closeness for p in node_props) / len(node_props)
        print(
            f"{name} stats -> avg_degree: {avg_degree:.4f}, "
            f"avg_cluster: {avg_cluster:.4f}, avg_closeness: {avg_close:.4f}"
        )


if __name__ == "__main__":
    main()