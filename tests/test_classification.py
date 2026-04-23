from __future__ import annotations

import pandas as pd

from typologist._llm import _CallableLLM
from typologist._pipeline import _classify_docs


def _facet(name="sentiment", values=None):
    return {
        "name": name,
        "kind": "categorical",
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


def test_classify_docs_uses_threadpool_when_max_concurrency_above_one():
    """Verify that concurrent calls actually overlap on multiple threads."""
    import threading
    import time

    observed_threads: set[int] = set()
    lock = threading.Lock()

    def slow_llm(prompt: str, **kwargs) -> str:
        with lock:
            observed_threads.add(threading.get_ident())
        time.sleep(0.05)
        return "positive"

    llm = _CallableLLM(slow_llm)
    docs = pd.Series([f"d{i}" for i in range(20)])

    out = _classify_docs(_facet(), docs, llm, noise_label="Unlabelled", max_concurrency=5)

    assert len(out) == 20
    # With max_concurrency=5 and 20 docs, we expect more than one thread to
    # have handled calls.
    assert len(observed_threads) > 1


def test_classify_docs_preserves_order_under_concurrency():
    """Even with concurrent dispatch, labels must align with input order."""
    import threading
    import time

    def ordered_llm(prompt: str, **kwargs) -> str:
        # Longer docs sleep longer to invert the natural call order
        for idx in range(10):
            if f"doc_{idx:02d}" in prompt:
                time.sleep((10 - idx) * 0.01)
                return f"value_{idx:02d}"
        return "nope"

    facet = {
        "name": "ordered",
        "kind": "categorical",
        "values": [f"value_{i:02d}" for i in range(10)],
        "definition": "",
        "labeling_prompt_template": "{document}",
        "labeling_model": "m",
    }
    llm = _CallableLLM(ordered_llm)
    docs = pd.Series([f"doc_{i:02d}" for i in range(10)])

    _ = threading  # keep import used

    out = _classify_docs(facet, docs, llm, noise_label="Unlabelled", max_concurrency=5)
    assert list(out) == [f"value_{i:02d}" for i in range(10)]
