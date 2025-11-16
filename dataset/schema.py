"""Dataclasses describing the normalized node-property schema."""

from __future__ import annotations

from dataclasses import dataclass, field
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
    cross_layer: Optional["CrossLayerInfo"] = None


@dataclass(slots=True)
class CrossLayerInfo:
    """Hop-aware cross-layer cues derived from anchor supervision."""

    anchor_pair_index: Optional[int] = None
    nearest_anchor: Optional[int] = None
    mirror_anchor: Optional[int] = None
    hop_distance: Optional[int] = None
    hop_token: str = "<hop-unk>"
    mirror_candidates: List[int] = field(default_factory=list)
    path_to_anchor: List[int] = field(default_factory=list)
    path_struct_profile: List[float] = field(default_factory=list)
