from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import anthropic


class _LLM(ABC):
    """Internal LLM contract shared by all three Typologist LLM roles.

    Always callable with a prompt string plus optional kwargs
    (``system_prompt``, ``temperature``, ``max_tokens``). Backends that can
    honor the kwargs do so; backends that can't (user-provided callables)
    discard them.
    """

    @property
    @abstractmethod
    def model_name(self) -> str | None: ...

    @abstractmethod
    def __call__(self, prompt: str, **options: Any) -> str: ...


class _AnthropicLLM(_LLM):
    """Anthropic-SDK-backed LLM resolved from a model-name string."""

    def __init__(self, model: str, client: anthropic.Anthropic | None = None):
        self._model = model
        self._client = client or anthropic.Anthropic()

    @property
    def model_name(self) -> str:
        return self._model

    def __call__(self, prompt: str, **options: Any) -> str:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": options.get("max_tokens", 1024),
            "messages": [{"role": "user", "content": prompt}],
            "temperature": options.get("temperature", 1.0),
        }
        system_prompt = options.get("system_prompt")
        if system_prompt:
            kwargs["system"] = system_prompt

        resp = self._client.messages.create(**kwargs)
        return resp.content[0].text


class _CallableLLM(_LLM):
    """Wraps a user-provided ``(prompt) -> str`` callable.

    Options (temperature etc.) are discarded at this boundary; the callable is
    expected to encapsulate whatever configuration it needs.
    """

    def __init__(self, fn: Callable[[str], str]):
        self._fn = fn

    @property
    def model_name(self) -> None:
        return None

    def __call__(self, prompt: str, **options: Any) -> str:
        return self._fn(prompt)


def _resolve_llm(spec: str | Callable[[str], str]) -> _LLM:
    """Turn a user-facing LLM spec (model name or callable) into an internal ``_LLM``."""
    if isinstance(spec, str):
        return _AnthropicLLM(model=spec)
    if callable(spec):
        return _CallableLLM(fn=spec)
    raise TypeError(f"LLM spec must be a model-name string or callable, got {type(spec).__name__}")


def _wrap_for_toponymy(llm: _LLM) -> Any:
    """Adapt our ``_LLM`` to Toponymy's ``LLMWrapper`` contract.

    Toponymy expects ``_call_llm(prompt, temperature, max_tokens)`` and
    ``_call_llm_with_system_prompt(system, user, temperature, max_tokens)``.
    We forward each of those through our ``_LLM.__call__`` with the
    corresponding options.
    """
    from toponymy.llm_wrappers import LLMWrapper

    class _Adapter(LLMWrapper):
        def _call_llm(self, prompt: str, temperature: float, max_tokens: int) -> str:
            return llm(prompt, temperature=temperature, max_tokens=max_tokens)

        def _call_llm_with_system_prompt(
            self,
            system_prompt: str,
            user_prompt: str,
            temperature: float,
            max_tokens: int,
        ) -> str:
            return llm(
                user_prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )

    return _Adapter()
