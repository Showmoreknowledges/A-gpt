"""Cross-layer anchor propagation helpers."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import DefaultDict, Dict, List, Sequence

import networkx as nx
import numpy as np

from .dataset_loader import GraphPair
from .schema import CrossLayerInfo
from special_tokens import HOP_UNKNOWN_TOKEN, hop_token


def _multi_source_bfs(
    graph: nx.Graph,
    anchor_to_pair: Dict[int, int],
    num_nodes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assign each node to the closest anchor (in hop distance)."""

    owners = np.full(num_nodes, -1, dtype=np.int64)
    hops = np.full(num_nodes, -1, dtype=np.int64)
    parents = np.full(num_nodes, -1, dtype=np.int64)
    queue: deque[int] = deque()
    for node, pair_idx in anchor_to_pair.items():
        if node < 0 or node >= num_nodes:
            continue
        owners[node] = pair_idx
        hops[node] = 0
        parents[node] = -1
        queue.append(node)

    while queue:
        current = queue.popleft()
        base_owner = owners[current]
        base_hop = hops[current]
        for nbr in graph.neighbors(current):
            nbr = int(nbr)
            if nbr < 0 or nbr >= num_nodes:
                continue
            if hops[nbr] != -1:
                continue
            owners[nbr] = base_owner
            hops[nbr] = base_hop + 1
            parents[nbr] = current
            queue.append(nbr)

    return owners, hops, parents


def _group_by_anchor_and_hop(owners: np.ndarray, hops: np.ndarray) -> Dict[int, Dict[int, List[int]]]:
    grouped: DefaultDict[int, DefaultDict[int, List[int]]] = defaultdict(lambda: defaultdict(list))
    for node, (owner_idx, hop) in enumerate(zip(owners.tolist(), hops.tolist())):
        if owner_idx < 0 or hop < 0:
            continue
        grouped[owner_idx][hop].append(node)
    return grouped


def _build_contexts(
    owners: np.ndarray,
    hops: np.ndarray,
    anchor_nodes: Sequence[int],
    mirror_anchor_nodes: Sequence[int],
    parents: np.ndarray,
    graph: nx.Graph,
    mirror_grouped: Dict[int, Dict[int, List[int]]],
) -> List[CrossLayerInfo]:
    contexts: List[CrossLayerInfo] = []
    for node_idx, (owner_idx, hop_value) in enumerate(zip(owners.tolist(), hops.tolist())):
        if owner_idx < 0 or hop_value < 0:
            contexts.append(
                CrossLayerInfo(
                    anchor_pair_index=None,
                    nearest_anchor=None,
                    mirror_anchor=None,
                    hop_distance=None,
                    hop_token=HOP_UNKNOWN_TOKEN,
                    mirror_candidates=[],
                    path_to_anchor=[],
                    path_struct_profile=[],
                )
            )
            continue

        mirror_candidates = mirror_grouped.get(owner_idx, {}).get(hop_value, [])
        path_nodes = _reconstruct_path(node_idx, parents)
        path_profile = _path_struct_profile(path_nodes, graph)
        contexts.append(
            CrossLayerInfo(
                anchor_pair_index=owner_idx,
                nearest_anchor=int(anchor_nodes[owner_idx]),
                mirror_anchor=int(mirror_anchor_nodes[owner_idx]),
                hop_distance=int(hop_value),
                hop_token=hop_token(int(hop_value)),
                mirror_candidates=list(mirror_candidates),
                path_to_anchor=path_nodes,
                path_struct_profile=path_profile,
            )
        )
    return contexts


def _reconstruct_path(node_idx: int, parents: np.ndarray) -> List[int]:
    if node_idx < 0 or node_idx >= parents.shape[0]:
        return []
    path: List[int] = []
    current = node_idx
    visited = set()
    while current != -1:
        if current in visited:
            break
        visited.add(current)
        path.append(int(current))
        next_parent = int(parents[current])
        if next_parent == -1:
            break
        current = next_parent
    return path


def _path_struct_profile(path: List[int], graph: nx.Graph) -> List[float]:
    if not path:
        return []
    degrees = graph.degree()
    profile: List[float] = []
    for node in path:
        if graph.has_node(node):
            profile.append(float(degrees[node]))
        else:
            profile.append(0.0)
    return profile


def compute_cross_layer_contexts(graph_pair: GraphPair) -> tuple[list[CrossLayerInfo], list[CrossLayerInfo]]:
    """Compute hop-aware cross-layer summaries for both graphs."""

    num_nodes_a = graph_pair.graph_a.num_nodes()
    num_nodes_b = graph_pair.graph_b.num_nodes()
    if graph_pair.aligned_pairs.size == 0:
        fallback_a = [
            CrossLayerInfo(hop_token=HOP_UNKNOWN_TOKEN) for _ in range(num_nodes_a)
        ]
        fallback_b = [
            CrossLayerInfo(hop_token=HOP_UNKNOWN_TOKEN) for _ in range(num_nodes_b)
        ]
        return fallback_a, fallback_b

    anchors = np.asarray(graph_pair.aligned_pairs, dtype=np.int64)
    anchor_nodes_a = anchors[:, 0].tolist()
    anchor_nodes_b = anchors[:, 1].tolist()
    anchor_map_a = {int(node): idx for idx, node in enumerate(anchor_nodes_a)}
    anchor_map_b = {int(node): idx for idx, node in enumerate(anchor_nodes_b)}

    owners_a, hops_a, parents_a = _multi_source_bfs(graph_pair.graph_a.graph, anchor_map_a, num_nodes_a)
    owners_b, hops_b, parents_b = _multi_source_bfs(graph_pair.graph_b.graph, anchor_map_b, num_nodes_b)

    grouped_a = _group_by_anchor_and_hop(owners_a, hops_a)
    grouped_b = _group_by_anchor_and_hop(owners_b, hops_b)

    contexts_a = _build_contexts(
        owners_a,
        hops_a,
        parents_a,
        graph_pair.graph_a.graph,
        anchor_nodes_a,
        anchor_nodes_b,
        grouped_b,
    )
    contexts_b = _build_contexts(
        owners_b,
        hops_b,
        parents_b,
        graph_pair.graph_b.graph,
        anchor_nodes_b,
        anchor_nodes_a,
        grouped_a,
    )
    return contexts_a, contexts_b
