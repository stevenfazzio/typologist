from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any


class LLMOutputError(RuntimeError):
    """Raised when an LLM returned output that couldn't be parsed as expected."""


class LLM(ABC):
    """Provider-agnostic LLM contract used by all three Typologist LLM roles.

    Always callable with a prompt string plus optional kwargs (``system_prompt``,
    ``temperature``, ``max_tokens``). Backends that can honor the kwargs do so;
    backends that can't (user-provided callables) discard them.

    Subclasses set ``provider`` (e.g., ``"anthropic"``) as a class attribute and
    expose a ``model_name`` property. When this LLM is used as the
    ``labeling_llm`` for a fitted Typologist, the pair is stored in
    ``schema_[i]["labeling_model"]`` as ``"{provider}:{model_name}"`` for
    provenance.
    """

    provider: str | None = None

    @property
    @abstractmethod
    def model_name(self) -> str | None: ...

    @abstractmethod
    def __call__(self, prompt: str, **options: Any) -> str: ...

    def call_structured(self, prompt: str, response_schema: dict) -> dict:
        """Return a structured response matching ``response_schema``.

        Default path: append a JSON-instructions suffix to the prompt and
        parse the string response. Subclasses with native structured-output
        support should override (see ``AnthropicLLM`` and ``OpenAILLM``).
        """
        instructions = (
            "\n\nRespond with only valid JSON matching this schema "
            "(no code fences, no prose before or after):\n"
            f"{json.dumps(response_schema)}"
        )
        raw = self(prompt + instructions)
        return _parse_json_response(raw)


class AnthropicLLM(LLM):
    """Anthropic-SDK-backed LLM.

    Requires the ``anthropic`` extra: ``pip install typologist[anthropic]``.
    """

    provider = "anthropic"

    def __init__(self, model: str, client: Any | None = None):
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self._model = model
        self._client = client

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
        """Anthropic tool-use path: define a single tool with ``response_schema``
        and force the model to call it. The tool input is the structured response.
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


class OpenAILLM(LLM):
    """OpenAI-SDK-backed LLM.

    Requires the ``openai`` extra: ``pip install typologist[openai]``.

    Structured outputs go through OpenAI's function-calling API (mirrors
    ``AnthropicLLM``'s tool-use path), which accepts full JSON Schema without
    the constraints of ``response_format`` strict mode.
    """

    provider = "openai"

    def __init__(self, model: str, client: Any | None = None):
        if client is None:
            import openai

            client = openai.OpenAI()
        self._model = model
        self._client = client

    @property
    def model_name(self) -> str:
        return self._model

    def __call__(self, prompt: str, **options: Any) -> str:
        messages: list[dict[str, str]] = []
        system_prompt = options.get("system_prompt")
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        resp = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            max_tokens=options.get("max_tokens", 1024),
            temperature=options.get("temperature", 1.0),
        )
        return resp.choices[0].message.content

    def call_structured(self, prompt: str, response_schema: dict) -> dict:
        """OpenAI function-calling path: define a single function whose
        ``parameters`` are ``response_schema`` and force the model to call it.
        The function arguments are the structured response.
        """
        function_def = {
            "type": "function",
            "function": {
                "name": "record_response",
                "description": "Record the structured response.",
                "parameters": response_schema,
            },
        }
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048,
            tools=[function_def],
            tool_choice={"type": "function", "function": {"name": "record_response"}},
        )
        message = resp.choices[0].message
        tool_calls = getattr(message, "tool_calls", None) or []
        if not tool_calls:
            raise LLMOutputError(
                f"OpenAI response contained no tool_calls. Content: {message.content!r}"
            )
        arguments = tool_calls[0].function.arguments
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError as e:
            raise LLMOutputError(
                f"OpenAI tool_call arguments did not parse as JSON. Arguments:\n{arguments[:500]}"
            ) from e
        if not isinstance(parsed, dict):
            raise LLMOutputError(
                f"OpenAI tool_call arguments parsed but were not an object. Parsed: {parsed!r}"
            )
        return parsed


class _CallableLLM(LLM):
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


def _resolve_llm(spec: LLM | Callable[[str], str]) -> LLM:
    """Turn a user-facing LLM spec into an internal ``LLM``.

    Accepts an ``LLM`` instance (passed through) or a ``Callable[[str], str]``
    (wrapped in ``_CallableLLM``). Strings are no longer accepted; they used to
    resolve to Anthropic by convention, which made the API implicitly
    Anthropic-defaulting. Pass ``AnthropicLLM(...)`` or ``OpenAILLM(...)``
    explicitly instead.
    """
    if isinstance(spec, LLM):
        return spec
    if isinstance(spec, str):
        raise TypeError(
            f"LLM spec is a string ({spec!r}). String specs are no longer supported. "
            "Pass typologist.AnthropicLLM(...), typologist.OpenAILLM(...), "
            "a custom typologist.LLM subclass, or a Callable[[str], str] instead."
        )
    if callable(spec):
        return _CallableLLM(fn=spec)
    raise TypeError(f"LLM spec must be an LLM instance or callable, got {type(spec).__name__}")


def _wrap_for_toponymy(llm: LLM) -> Any:
    """Adapt our ``LLM`` to Toponymy's ``LLMWrapper`` contract.

    Toponymy expects ``_call_llm(prompt, temperature, max_tokens)`` and
    ``_call_llm_with_system_prompt(system, user, temperature, max_tokens)``.
    We forward each of those through our ``LLM.__call__`` with the
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
