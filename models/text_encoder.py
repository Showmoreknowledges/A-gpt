"""Sentence-Transformer powered text encoder for node descriptions."""

from __future__ import annotations

from typing import Sequence

import torch
from sentence_transformers import SentenceTransformer


class TextEncoder:
    """Wrapper around :class:`SentenceTransformer` with a tiny API surface."""

    def __init__(
        self,
        model_name: str,
        *,
        device: str | None = None,
        normalize: bool = True,
        special_tokens: Sequence[str] | None = None,
    ) -> None:
        self.model = SentenceTransformer(model_name, device=device)
        self.normalize = normalize
        if special_tokens:
            self._register_special_tokens(special_tokens)

    def _register_special_tokens(self, tokens: Sequence[str]) -> None:
        tokenizer = getattr(self.model, "tokenizer", None)
        if tokenizer is None or not hasattr(tokenizer, "add_special_tokens"):
            return
        vocab = set()
        if hasattr(tokenizer, "get_vocab"):
            vocab.update(tokenizer.get_vocab().keys())
        new_tokens = [tok for tok in tokens if tok not in vocab]
        if not new_tokens:
            return
        tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})

        model = None
        try:
            first_module = self.model._first_module()
        except AttributeError:
            first_module = None
        if first_module is not None and hasattr(first_module, "auto_model"):
            model = first_module.auto_model
        elif hasattr(self.model, "auto_model"):
            model = getattr(self.model, "auto_model")
        if model is not None and hasattr(model, "resize_token_embeddings"):
            model.resize_token_embeddings(len(tokenizer))

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
