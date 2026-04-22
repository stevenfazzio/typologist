from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import anthropic


class LLMOutputError(RuntimeError):
    """Raised when an LLM returned output that couldn't be parsed as expected."""


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

    def call_structured(self, prompt: str, response_schema: dict) -> dict:
        """Return a structured response matching ``response_schema``.

        Default (callable) path: append a JSON-instructions suffix to the prompt
        and parse the string response. Subclasses that can do better (e.g.
        ``_AnthropicLLM`` via tool-use) should override.
        """
        instructions = (
            "\n\nRespond with only valid JSON matching this schema "
            "(no code fences, no prose before or after):\n"
            f"{json.dumps(response_schema)}"
        )
        raw = self(prompt + instructions)
        return _parse_json_response(raw)


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

    def call_structured(self, prompt: str, response_schema: dict) -> dict:
        """Anthropic tool-use path: define a single tool with ``response_schema`` and
        force the model to call it. The tool input is the structured response.
        """
        tool = {
            "name": "record_response",
            "description": "Record the structured response.",
            "input_schema": response_schema,
        }
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=2048,
            tools=[tool],
            tool_choice={"type": "tool", "name": "record_response"},
            messages=[{"role": "user", "content": prompt}],
        )
        for block in resp.content:
            if getattr(block, "type", None) == "tool_use":
                return dict(block.input)
        raise LLMOutputError(
            f"Anthropic response contained no tool_use block. Content: {resp.content!r}"
        )


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


def _parse_json_response(raw: str) -> dict:
    """Extract and parse a JSON object from an LLM response.

    Tolerates ```json code fences around the JSON body; anything else that
    doesn't parse cleanly raises ``LLMOutputError`` with the raw response
    included for debugging.
    """
    stripped = raw.strip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.DOTALL)
        if match is None:
            raise LLMOutputError(
                f"LLM response contained no parseable JSON. Response:\n{raw[:500]}"
            ) from None
        try:
            parsed = json.loads(match.group(1))
        except json.JSONDecodeError as e:
            raise LLMOutputError(
                f"LLM response had a code-fence but contents did not parse as JSON. "
                f"Response:\n{raw[:500]}"
            ) from e
    if not isinstance(parsed, dict):
        raise LLMOutputError(
            f"LLM response parsed as JSON but was not an object. Parsed: {parsed!r}"
        )
    return parsed


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
