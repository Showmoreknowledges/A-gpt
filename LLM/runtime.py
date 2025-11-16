"""Convenience helpers for instantiating :class:`LLMClient` objects."""

from __future__ import annotations

import os
from typing import Any, Mapping, MutableMapping, Sequence

from .providers import (
    DeepSeekClient,
    EchoClient,
    HFLocalClient,
    LLMClient,
    OpenAIClient,
)


def _register_special_tokens_for_pipeline(pipeline: Any, tokens: Sequence[str] | None) -> None:
    if not tokens:
        return

    tokenizer = getattr(pipeline, "tokenizer", None)
    model = getattr(pipeline, "model", None)
    if tokenizer is None or not hasattr(tokenizer, "add_special_tokens"):
        return

    vocab = set()
    if hasattr(tokenizer, "get_vocab"):
        vocab_dict = tokenizer.get_vocab()
        vocab.update(vocab_dict.keys())

    unique: list[str] = []
    for tok in tokens:
        if tok in vocab:
            continue
        if tok not in unique:
            unique.append(tok)

    if not unique:
        return

    tokenizer.add_special_tokens({"additional_special_tokens": unique})
    if model is not None and hasattr(model, "resize_token_embeddings"):
        model.resize_token_embeddings(len(tokenizer))


def build_llm_client(kind: str, **kwargs: Any) -> LLMClient:
    """Return an initialized :class:`LLMClient` for ``kind``.

    Parameters
    ----------
    kind:
        One of ``"echo"``, ``"openai"``, ``"deepseek"``, or ``"hf-local"``.
    kwargs:
        Backend-specific configuration such as API keys or HuggingFace model names.
    """

    special_tokens = kwargs.pop("special_tokens", None)
    kind = kind.lower()
    if kind == "echo":
        return EchoClient(suffix=str(kwargs.get("echo_suffix", "")))

    if kind in {"openai", "deepseek"}:
        api_key = kwargs.get("api_key")
        if not api_key:
            env_key = "DEEPSEEK_API_KEY" if kind == "deepseek" else "OPENAI_API_KEY"
            api_key = os.getenv(env_key)
        if not api_key:
            raise ValueError("API key is required for OpenAI/DeepSeek clients.")

        default_kwargs = kwargs.get("default_kwargs")
        if default_kwargs is not None and not isinstance(default_kwargs, Mapping):
            raise TypeError("default_kwargs must be a mapping of generation parameters.")

        base_url = kwargs.get("base_url")
        client_cls = DeepSeekClient if kind == "deepseek" else OpenAIClient
        return client_cls(  # type: ignore[return-value]
            api_key=api_key,
            base_url=base_url,
            default_kwargs=default_kwargs,
        )

    if kind == "hf-local":
        model_name = kwargs.get("hf_model") or kwargs.get("model_name")
        if not model_name:
            raise ValueError("hf-local backend requires `hf_model`/`model_name`.")
        task = kwargs.get("hf_task", "text-generation")
        generation_kwargs: MutableMapping[str, Any] = {}
        provided_kwargs = kwargs.get("generation_kwargs")
        if provided_kwargs is not None:
            if not isinstance(provided_kwargs, Mapping):
                raise TypeError("generation_kwargs must be a mapping")
            generation_kwargs.update(provided_kwargs)

        try:
            from transformers import pipeline as hf_pipeline  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("transformers package is required for hf-local backend") from exc

        pipeline_kwargs: dict[str, Any] = {}
        if kwargs.get("hf_device") is not None:
            pipeline_kwargs["device"] = kwargs["hf_device"]
        if kwargs.get("hf_revision") is not None:
            pipeline_kwargs["revision"] = kwargs["hf_revision"]

        pipe = hf_pipeline(task, model=model_name, **pipeline_kwargs)
        _register_special_tokens_for_pipeline(pipe, special_tokens)
        return HFLocalClient(pipeline=pipe, generation_kwargs=dict(generation_kwargs))

    raise ValueError(f"Unsupported LLM backend: {kind}")
