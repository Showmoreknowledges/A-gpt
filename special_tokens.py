"""Utility helpers for hop-specific special tokens shared across modules."""

from __future__ import annotations

from typing import Iterable, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from dataset.schema import CrossLayerInfo

HOP_UNKNOWN_TOKEN = "<hop-unk>"


def hop_token(distance: int | None) -> str:
    """Return the canonical ``<hop-k>`` token for ``distance``."""

    if distance is None or distance < 0:
        return HOP_UNKNOWN_TOKEN
    return f"<hop-{int(distance)}>"


def collect_hop_tokens(contexts: Iterable["CrossLayerInfo"]) -> list[str]:
    """Aggregate the distinct hop tokens referenced by ``contexts``."""

    tokens = {HOP_UNKNOWN_TOKEN}
    for ctx in contexts:
        if ctx is None:
            continue
        token = getattr(ctx, "hop_token", None)
        if token:
            tokens.add(token)
    return sorted(tokens)
