from __future__ import annotations

import pytest

from typologist._llm import (
    LLMOutputError,
    _AnthropicLLM,
    _CallableLLM,
    _parse_json_response,
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


def test_parse_json_response_plain_object():
    out = _parse_json_response('{"a": 1, "b": "x"}')
    assert out == {"a": 1, "b": "x"}


def test_parse_json_response_strips_whitespace():
    out = _parse_json_response('  \n{"a": 1}\n  ')
    assert out == {"a": 1}


def test_parse_json_response_extracts_from_code_fence():
    raw = 'Here is my response:\n```json\n{"a": 1}\n```\n'
    out = _parse_json_response(raw)
    assert out == {"a": 1}


def test_parse_json_response_rejects_non_object():
    with pytest.raises(LLMOutputError, match="not an object"):
        _parse_json_response("[1, 2, 3]")


def test_parse_json_response_rejects_unparseable():
    with pytest.raises(LLMOutputError, match="no parseable JSON"):
        _parse_json_response("I cannot help with that.")


def test_callable_llm_call_structured_roundtrips_json():
    def fn(prompt: str) -> str:
        return '{"name": "f", "values": ["a", "b"]}'

    llm = _CallableLLM(fn)
    out = llm.call_structured("propose a facet", {"type": "object"})
    assert out == {"name": "f", "values": ["a", "b"]}


def test_callable_llm_call_structured_raises_on_malformed():
    llm = _CallableLLM(lambda p: "not json at all")
    with pytest.raises(LLMOutputError):
        llm.call_structured("propose a facet", {"type": "object"})


class _FakeToolUseBlock:
    def __init__(self, input_dict):
        self.type = "tool_use"
        self.input = input_dict


class _FakeTextOnlyResponse:
    def __init__(self, text):
        self.content = [_FakeTextBlock(text)]


class _FakeToolUseResponse:
    def __init__(self, input_dict):
        self.content = [_FakeToolUseBlock(input_dict)]


class _FakeAnthropicClientWithTools:
    def __init__(self, response):
        self._response = response
        self.last_call: dict | None = None
        self.messages = self._Messages(self)

    class _Messages:
        def __init__(self, parent):
            self._parent = parent

        def create(self, **kwargs):
            self._parent.last_call = kwargs
            return self._parent._response


def test_anthropic_llm_call_structured_uses_tool_use():
    resp = _FakeToolUseResponse({"name": "f", "values": ["a", "b"]})
    client = _FakeAnthropicClientWithTools(resp)
    llm = _AnthropicLLM("claude-opus-4-7", client=client)

    schema = {"type": "object", "properties": {"name": {"type": "string"}}}
    out = llm.call_structured("propose a facet", schema)

    assert out == {"name": "f", "values": ["a", "b"]}
    assert client.last_call["tools"][0]["input_schema"] == schema
    assert client.last_call["tool_choice"] == {"type": "tool", "name": "record_response"}


def test_anthropic_llm_call_structured_raises_when_tool_use_absent():
    resp = _FakeTextOnlyResponse("I refuse.")
    client = _FakeAnthropicClientWithTools(resp)
    llm = _AnthropicLLM("claude-opus-4-7", client=client)

    with pytest.raises(LLMOutputError, match="no tool_use block"):
        llm.call_structured("propose a facet", {"type": "object"})
