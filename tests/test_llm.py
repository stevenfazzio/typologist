from __future__ import annotations

import pytest

from typologist._llm import (
    _AnthropicLLM,
    _CallableLLM,
    _resolve_llm,
    _wrap_for_toponymy,
)


class _FakeTextBlock:
    def __init__(self, text: str):
        self.text = text


class _FakeResponse:
    def __init__(self, text: str):
        self.content = [_FakeTextBlock(text)]


class _FakeAnthropicClient:
    """Records the last call to messages.create and returns a canned text."""

    def __init__(self, response_text: str = "ok"):
        self._response_text = response_text
        self.last_call: dict | None = None
        self.messages = self._Messages(self)

    class _Messages:
        def __init__(self, parent):
            self._parent = parent

        def create(self, **kwargs):
            self._parent.last_call = kwargs
            return _FakeResponse(self._parent._response_text)


def test_resolve_string_yields_anthropic_llm():
    llm = _resolve_llm("claude-haiku-4-5")
    assert isinstance(llm, _AnthropicLLM)
    assert llm.model_name == "claude-haiku-4-5"


def test_resolve_callable_yields_callable_llm():
    llm = _resolve_llm(lambda p: "ok")
    assert isinstance(llm, _CallableLLM)
    assert llm.model_name is None


def test_resolve_rejects_non_str_non_callable():
    with pytest.raises(TypeError, match="string or callable"):
        _resolve_llm(42)


def test_callable_llm_forwards_prompt_only():
    captured = {}

    def fn(prompt: str) -> str:
        captured["prompt"] = prompt
        return "response"

    llm = _CallableLLM(fn)
    out = llm("hello", temperature=0.3, max_tokens=50, system_prompt="sys")
    assert out == "response"
    assert captured["prompt"] == "hello"


def test_anthropic_llm_sends_expected_kwargs():
    client = _FakeAnthropicClient(response_text="synthesized")
    llm = _AnthropicLLM("claude-haiku-4-5", client=client)

    out = llm("hi", temperature=0.2, max_tokens=256, system_prompt="You are a helper.")

    assert out == "synthesized"
    assert client.last_call is not None
    assert client.last_call["model"] == "claude-haiku-4-5"
    assert client.last_call["temperature"] == 0.2
    assert client.last_call["max_tokens"] == 256
    assert client.last_call["system"] == "You are a helper."
    assert client.last_call["messages"] == [{"role": "user", "content": "hi"}]


def test_anthropic_llm_defaults_when_options_absent():
    client = _FakeAnthropicClient()
    llm = _AnthropicLLM("claude-opus-4-7", client=client)

    llm("hello")
    assert "system" not in client.last_call
    assert client.last_call["max_tokens"] == 1024
    assert client.last_call["temperature"] == 1.0


def test_wrap_for_toponymy_forwards_plain_prompt():
    recorded = {}

    def fn(prompt: str) -> str:
        recorded["prompt"] = prompt
        return "x"

    llm = _CallableLLM(fn)
    wrapper = _wrap_for_toponymy(llm)

    out = wrapper._call_llm("a prompt", temperature=0.4, max_tokens=100)
    assert out == "x"
    assert recorded["prompt"] == "a prompt"


def test_wrap_for_toponymy_forwards_system_prompt_to_anthropic():
    client = _FakeAnthropicClient(response_text="named")
    llm = _AnthropicLLM("claude-haiku-4-5", client=client)
    wrapper = _wrap_for_toponymy(llm)

    out = wrapper._call_llm_with_system_prompt(
        "you are a topic namer",
        "cluster: quantum, qubits, entanglement",
        temperature=0.4,
        max_tokens=128,
    )

    assert out == "named"
    assert client.last_call["system"] == "you are a topic namer"
    assert client.last_call["messages"] == [
        {"role": "user", "content": "cluster: quantum, qubits, entanglement"}
    ]
    assert client.last_call["temperature"] == 0.4
    assert client.last_call["max_tokens"] == 128


def test_wrap_for_toponymy_returns_llm_wrapper_subclass():
    from toponymy.llm_wrappers import LLMWrapper

    wrapper = _wrap_for_toponymy(_CallableLLM(lambda p: "x"))
    assert isinstance(wrapper, LLMWrapper)
