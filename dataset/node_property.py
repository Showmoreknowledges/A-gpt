"""Compute the structured node-property schema from graph data."""

from __future__ import annotations

from collections import Counter
from typing import List, Optional, Protocol

import networkx as nx

from .schema import NeighborhoodSummary, NodeProperty, StructuralFeatures


class GraphData(Protocol):
    """Minimal protocol describing the graph container expected by this module."""

    graph: nx.Graph

    def num_nodes(self) -> int: ...  # pragma: no cover - protocol definition


def _ensure_graph(graph_like: GraphData | nx.Graph) -> nx.Graph:
    if isinstance(graph_like, nx.Graph):
        return graph_like
    graph = getattr(graph_like, "graph", None)
    if isinstance(graph, nx.Graph):
        return graph
    raise TypeError("compute_node_properties expects a NetworkX graph or GraphData with a `.graph`." )


def _topk_neighbors(graph: nx.Graph, node: int, k: int) -> List[int]:
    neighbors = list(graph.neighbors(node))
    if not neighbors:
        return []
    degrees = graph.degree()
    neighbors.sort(key=lambda n: (-degrees[n], n))
    return neighbors[:k]


def _neighbor_type_hist(graph: nx.Graph, node: int) -> Optional[dict[str, float]]:
    types = [graph.nodes[v].get("type") for v in graph.neighbors(node)]
    types = [t for t in types if t is not None]
    if not types:
        return None
    counts = Counter(types)
    total = float(sum(counts.values()))
    return {k: v / total for k, v in counts.items()}


def _ego_density(graph: nx.Graph, node: int) -> Optional[float]:
    ego = nx.ego_graph(graph, node)
    n = ego.number_of_nodes()
    if n <= 1:
        return None
    possible_edges = n * (n - 1) / 2
    if possible_edges == 0:
        return None
    return ego.number_of_edges() / possible_edges


def compute_node_properties(
    graph: GraphData | nx.Graph,
    graph_type: str,
    *,
    topk: int = 5,
    include_pagerank: bool = True,
    include_betweenness: bool = False,
) -> list[NodeProperty]:
    """Compute :class:`NodeProperty` descriptors for each node in ``graph``."""

    nx_graph = _ensure_graph(graph)
    num_nodes = nx_graph.number_of_nodes()
    if num_nodes == 0:
        return []

    degree_dict = dict(nx_graph.degree())
    clustering = nx.clustering(nx_graph)
    square_clustering = nx.square_clustering(nx_graph)
    closeness = nx.closeness_centrality(nx_graph)
    pagerank = nx.pagerank(nx_graph) if include_pagerank else {}
    betweenness = nx.betweenness_centrality(nx_graph) if include_betweenness else {}

    properties: list[NodeProperty] = []
    for node in nx_graph.nodes():
        struct = StructuralFeatures(
            degree=float(degree_dict.get(node, 0.0)),
            clustering=float(clustering.get(node, 0.0)),
            square_clustering=float(square_clustering.get(node, 0.0)),
            closeness=float(closeness.get(node, 0.0)),
            pagerank=float(pagerank[node]) if node in pagerank else None,
            betweenness=float(betweenness[node]) if node in betweenness else None,
        )
        neigh = NeighborhoodSummary(
            topk_neighbors=_topk_neighbors(nx_graph, node, topk),
            neighbor_type_hist=_neighbor_type_hist(nx_graph, node),
            ego_density=_ego_density(nx_graph, node),
        )
        node_type = nx_graph.nodes[node].get("type")
        properties.append(
            NodeProperty(
                struct=struct,
                neigh=neigh,
                node_type=node_type,
                graph_type=graph_type,
            )
        )

    return properties