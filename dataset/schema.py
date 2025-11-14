"""Dataclasses describing the normalized node-property schema."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(slots=True)
class StructuralFeatures:
    """Structural statistics computed for every node in a graph."""

    degree: float
    clustering: float
    square_clustering: float
    closeness: float
    pagerank: Optional[float] = None
    betweenness: Optional[float] = None


@dataclass(slots=True)
class NeighborhoodSummary:
    """Information about a node's ego network and neighbour types."""

    topk_neighbors: List[int]
    neighbor_type_hist: Optional[Dict[str, float]] = None
    ego_density: Optional[float] = None


@dataclass(slots=True)
class NodeProperty:
    """Full node-level descriptor used by downstream stages."""

    struct: StructuralFeatures
    neigh: NeighborhoodSummary
    node_type: Optional[str] = None
    graph_type: Optional[str] = None
