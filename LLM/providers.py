"""Unified large language model client abstractions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional


class LLMClient(ABC):
    """Abstract interface implemented by all LLM providers used in the pipeline."""

    @abstractmethod
    def complete(self, *, model: str, prompt: str, **kwargs: Any) -> str:
        """Return the generated text for ``prompt`` using ``model``."""


@dataclass
class OpenAIClient(LLMClient):
    """Wrapper around the official OpenAI Python SDK."""

    api_key: str
    base_url: Optional[str] = None
    default_kwargs: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("openai package is required for OpenAIClient") from exc
        self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def complete(self, *, model: str, prompt: str, **kwargs: Any) -> str:
        payload = dict(self.default_kwargs or {})
        payload.update(kwargs)
        response = self._client.chat.completions.create(  # type: ignore[attr-defined]
            model=model,
            messages=[{"role": "user", "content": prompt}],
            **payload,
        )
        choice = response.choices[0]
        content = getattr(choice.message, "content", None) or ""
        return content.strip()


@dataclass
class DeepSeekClient(OpenAIClient):
    """DeepSeek compatible client that reuses the OpenAI SDK surface."""

    base_url: Optional[str] = "https://api.deepseek.com"


@dataclass
class HFLocalClient(LLMClient):
    """Thin wrapper on top of a ``transformers`` text-generation pipeline."""

    pipeline: Any
    generation_kwargs: Dict[str, Any] = field(default_factory=dict)

    def complete(self, *, model: str, prompt: str, **kwargs: Any) -> str:
        payload = dict(self.generation_kwargs)
        payload.update(kwargs)
        payload.setdefault("max_new_tokens", 256)
        outputs = self.pipeline(prompt, **payload)
        if isinstance(outputs, list) and outputs:
            text = outputs[0].get("generated_text", "")
        else:
            text = str(outputs)
        return text.strip()


@dataclass
class EchoClient(LLMClient):
    """Deterministic client useful for tests and offline development."""

    suffix: str = ""

    def complete(self, *, model: str, prompt: str, **kwargs: Any) -> str:  # noqa: D401
        return f"{prompt.strip()} {self.suffix}".strip()
