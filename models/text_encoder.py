"""Sentence-Transformer powered text encoder for node descriptions."""

from __future__ import annotations

from typing import Sequence

import torch
from sentence_transformers import SentenceTransformer


class TextEncoder:
    """Wrapper around :class:`SentenceTransformer` with a tiny API surface."""

    def __init__(self, model_name: str, *, device: str | None = None, normalize: bool = True) -> None:
        self.model = SentenceTransformer(model_name, device=device)
        self.normalize = normalize

    def encode(self, texts: Sequence[str]) -> torch.Tensor:
        """Encode ``texts`` into a dense tensor of shape ``[N, d_text]``."""

        if not isinstance(texts, (list, tuple)):
            texts = list(texts)
        embeddings = self.model.encode(
            texts,
            convert_to_tensor=True,
            normalize_embeddings=self.normalize,
        )
        return embeddings