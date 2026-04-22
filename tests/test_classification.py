from __future__ import annotations

import pandas as pd

from typologist._llm import _CallableLLM
from typologist._pipeline import _classify_docs


def _facet(name="sentiment", values=None):
    return {
        "name": name,
        "type": "categorical",
        "values": values or ["positive", "neutral", "negative"],
        "definition": "Overall tone.",
        "labeling_prompt_template": "Classify:\n{document}\nRespond with one value.",
        "labeling_model": "claude-haiku-4-5",
    }


def test_classify_docs_happy_path():
    responses = {"doc a": "positive", "doc b": "negative", "doc c": "neutral"}
    llm = _CallableLLM(lambda p: _extract_doc_and_respond(p, responses))
    docs = pd.Series(["doc a", "doc b", "doc c"])

    out = _classify_docs(_facet(), docs, llm, noise_label="Unlabelled")

    assert list(out) == ["positive", "negative", "neutral"]
    assert out.dtype == "category"


def test_classify_docs_case_insensitive_match():
    responses = {"d1": "Positive", "d2": "  NEGATIVE  ", "d3": "Neutral\n"}
    llm = _CallableLLM(lambda p: _extract_doc_and_respond(p, responses))
    docs = pd.Series(["d1", "d2", "d3"])

    out = _classify_docs(_facet(), docs, llm, noise_label="Unlabelled")
    assert list(out) == ["positive", "negative", "neutral"]


def test_classify_docs_falls_back_to_noise_label_on_miss():
    responses = {"d1": "positive", "d2": "nonsense response", "d3": "negative"}
    llm = _CallableLLM(lambda p: _extract_doc_and_respond(p, responses))
    docs = pd.Series(["d1", "d2", "d3"])

    out = _classify_docs(_facet(), docs, llm, noise_label="Unlabelled")
    assert list(out) == ["positive", "Unlabelled", "negative"]


def test_classify_docs_preserves_index():
    llm = _CallableLLM(lambda p: "positive")
    docs = pd.Series(["d1", "d2"], index=[100, 200])

    out = _classify_docs(_facet(), docs, llm, noise_label="Unlabelled")
    assert list(out.index) == [100, 200]


def test_classify_docs_categorical_includes_noise_label():
    llm = _CallableLLM(lambda p: "positive")
    docs = pd.Series(["d1"])

    out = _classify_docs(_facet(), docs, llm, noise_label="Unlabelled")
    assert "Unlabelled" in out.cat.categories
    assert set(out.cat.categories) == {"positive", "neutral", "negative", "Unlabelled"}


def test_classify_docs_does_not_double_add_noise_when_in_values():
    facet = _facet(values=["a", "b", "Unlabelled"])
    llm = _CallableLLM(lambda p: "a")
    docs = pd.Series(["d1"])

    out = _classify_docs(facet, docs, llm, noise_label="Unlabelled")
    assert list(out.cat.categories) == ["a", "b", "Unlabelled"]


def test_classify_docs_uses_facet_name_as_series_name():
    llm = _CallableLLM(lambda p: "positive")
    out = _classify_docs(_facet(name="my_facet"), pd.Series(["d"]), llm, "Unlabelled")
    assert out.name == "my_facet"


def _extract_doc_and_respond(prompt: str, responses: dict[str, str]) -> str:
    """Find the document in the prompt and return its canned response."""
    for doc, response in responses.items():
        if doc in prompt:
            return response
    raise AssertionError(f"no matching doc in prompt: {prompt[:200]}")
