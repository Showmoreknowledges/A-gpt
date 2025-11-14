"""Utilities for loading dual-layer graphs and their alignment labels.

The loader only depends on the information contained in the provided
``npz_path`` bundle.  As long as the dataset exposes the expected keys (two sets
of node features/edges and one of the supported alignment annotations) the
graph pair can be reconstructed without caring about how the file is named.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import networkx as nx
import numpy as np

ArrayLike = Union[np.ndarray, Sequence[Sequence[int]]]


@dataclass
class GraphLayer:
    """Container that bundles a NetworkX graph with optional features."""

    graph: nx.Graph
    features: Optional[np.ndarray] = None
    name: Optional[str] = None

    def num_nodes(self) -> int:
        return self.graph.number_of_nodes()


@dataclass
class GraphPair:
    """Two aligned graphs and their reference node mapping."""

    graph_a: GraphLayer
    graph_b: GraphLayer
    aligned_pairs: np.ndarray
    test_pairs: Optional[np.ndarray] = None
    metadata: Dict[str, Union[str, int, float]] = field(default_factory=dict)

    def num_aligned_pairs(self) -> int:
        return int(self.aligned_pairs.shape[0])


def _ensure_edge_array(edge_index: ArrayLike) -> np.ndarray:
    array = np.asarray(edge_index)
    if array.ndim != 2:
        raise ValueError("edge_index must be a 2-D array with shape (2, E) or (E, 2)")
    if array.shape[0] != 2:
        if array.shape[1] != 2:
            raise ValueError(f"edge_index array has invalid shape {array.shape}")
        array = array.T
    return array.astype(np.int64, copy=False)


def _ensure_pair_array(pairs: ArrayLike, *, allow_empty: bool = True) -> np.ndarray:
    array = np.asarray(pairs)
    if array.size == 0:
        if allow_empty:
            return np.zeros((0, 2), dtype=np.int64)
        raise ValueError("Alignment pairs array is empty.")
    if array.ndim != 2:
        raise ValueError("Pairs must be a 2-D array with shape (N, 2) or (2, N).")
    if array.shape[1] != 2:
        if array.shape[0] == 2:
            array = array.T
        else:
            raise ValueError(f"Pairs array has invalid shape {array.shape}.")
    return array.astype(np.int64, copy=False)


def _build_graph(edge_index: np.ndarray, num_nodes: int) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    if edge_index.size:
        edges = edge_index.T.tolist()
        graph.add_edges_from(edges)
    return graph


def _load_feature(value: ArrayLike, num_nodes: int) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim == 1:
        arr = arr.reshape(num_nodes, -1)
    if arr.shape[0] != num_nodes:
        raise ValueError(
            "Feature matrix row count must match inferred node count. "
            f"Got {arr.shape[0]} rows vs {num_nodes} nodes."
        )
    return arr.astype(np.float32, copy=False)


def _select_alignment_pairs(npz_dict: np.lib.npyio.NpzFile) -> np.ndarray:
    if "aligned_pair" in npz_dict.files:
        return _ensure_pair_array(npz_dict["aligned_pair"], allow_empty=False)
    if "aligned_pairs" in npz_dict.files:
        return _ensure_pair_array(npz_dict["aligned_pairs"], allow_empty=False)
    if "pos_pairs" in npz_dict.files and "test_pairs" in npz_dict.files:
        pos_pairs = _ensure_pair_array(npz_dict["pos_pairs"], allow_empty=True)
        test_pairs = _ensure_pair_array(npz_dict["test_pairs"], allow_empty=True)
        return np.concatenate([pos_pairs, test_pairs], axis=0)
    raise KeyError(
        "Unable to find alignment labels in the .npz file. "
        "Expected `aligned_pair`, `aligned_pairs`, or both `pos_pairs` and `test_pairs`."
    )


def _load_layer(npz_dict: np.lib.npyio.NpzFile, layer_idx: int) -> GraphLayer:
    edge_key = f"edge_index{layer_idx}"
    feature_key = f"x{layer_idx}"
    if edge_key not in npz_dict.files or feature_key not in npz_dict.files:
        raise KeyError(
            f"Missing required keys for layer {layer_idx}: expected `{edge_key}` and `{feature_key}`."
        )

    edge_index = _ensure_edge_array(npz_dict[edge_key])
    num_nodes = int(edge_index.max()) + 1 if edge_index.size else 0
    features_raw = np.asarray(npz_dict[feature_key])
    inferred_nodes = max(num_nodes, features_raw.shape[0])
    features = _load_feature(features_raw, inferred_nodes)
    num_nodes = max(num_nodes, features.shape[0])
    graph = _build_graph(edge_index, num_nodes)
    return GraphLayer(graph=graph, features=features, name=f"layer_{layer_idx}")


def load_graph_pair(npz_path: Union[str, Path]) -> GraphPair:
    """Load a pair of graphs and alignment labels from a ``.npz`` bundle."""

    npz_path = Path(npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)

    data = np.load(npz_path, allow_pickle=True)
    graph_a = _load_layer(data, layer_idx=1)
    graph_b = _load_layer(data, layer_idx=2)

    aligned_pairs = _select_alignment_pairs(data)
    metadata = {
        "path": str(npz_path),
    }

    return GraphPair(
        graph_a=graph_a,
        graph_b=graph_b,
        aligned_pairs=aligned_pairs,
        test_pairs=None,
        metadata=metadata,
    )
