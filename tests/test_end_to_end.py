from __future__ import annotations

import json

import numpy as np
import pandas as pd

from typologist import Typologist, apply_schema


class _FakeTopicEmbedder:
    def encode(self, texts, show_progress_bar=False):
        return np.zeros((len(texts), 4), dtype=np.float32)


class _FakeToponymy:
    """Returns canned topic names so schema synthesis has something to work with."""

    def __init__(self, **kwargs):
        pass

    def fit(self, objects, embedding_vectors, clusterable_vectors):
        n = len(objects)
        self.topic_names_ = [
            ["fine_topic_a", "fine_topic_b", "fine_topic_c"],
            ["coarse_topic_x"],
        ]
        self.topic_name_vectors_ = [
            np.array(["fine_topic_a"] * n, dtype=object),
            np.array(["coarse_topic_x"] * n, dtype=object),
        ]


class _FakeEVoCClusterer:
    def __init__(self, **kwargs):
        pass


def _install_toponymy_mocks(monkeypatch):
    from typologist import _pipeline

    monkeypatch.setattr(_pipeline, "Toponymy", _FakeToponymy)
    monkeypatch.setattr(_pipeline, "EVoCClusterer", _FakeEVoCClusterer)


def _make_schema_llm(facet_dicts: list[dict]):
    """Returns JSON-formatted canned facets, one per call."""
    queue = list(facet_dicts)

    def call(prompt: str, **kwargs) -> str:
        return json.dumps(queue.pop(0))

    return call


def _make_labeling_llm(doc_to_value: dict[str, str], default: str = "none-of-the-above"):
    """Returns the canned value for whichever document appears in the prompt."""

    def call(prompt: str, **kwargs) -> str:
        for doc, value in doc_to_value.items():
            if doc in prompt:
                return value
        return default

    return call


def test_fit_populates_all_fitted_attributes(monkeypatch):
    _install_toponymy_mocks(monkeypatch)

    t = Typologist(
        n_facets=2,
        topic_embedder=_FakeTopicEmbedder(),
        naming_llm=lambda p: "ignored",
        schema_llm=_make_schema_llm(
            [
                {
                    "name": "facet_one",
                    "type": "categorical",
                    "values": ["a", "b"],
                    "definition": "first axis",
                },
                {
                    "name": "facet_two",
                    "type": "categorical",
                    "values": ["x", "y"],
                    "definition": "second axis",
                },
            ]
        ),
        labeling_llm=_make_labeling_llm({"doc_1": "a", "doc_2": "b", "doc_3": "a"}, default="x"),
    )

    docs = ["doc_1", "doc_2", "doc_3"]
    emb = np.random.default_rng(0).random((3, 4)).astype(np.float32)
    t.fit(docs, emb)

    assert len(t.schema_) == 2
    assert [f["name"] for f in t.schema_] == ["facet_one", "facet_two"]
    assert t.schema_[0]["values"] == ["a", "b", "Other"]
    assert "{document}" in t.schema_[0]["labeling_prompt_template"]

    assert t.labels_df_.shape == (3, 2)
    assert list(t.labels_df_.columns) == ["facet_one", "facet_two"]

    assert t.embeddings_residualized_.shape == emb.shape

    assert len(t.facet_diagnostics_) == 2
    assert "synthesis_prompt" in t.facet_diagnostics_[0]
    assert "entropy_bits" in t.facet_diagnostics_[0]
    assert "exemplars_per_value" in t.facet_diagnostics_[0]


def test_fit_preserves_series_index_onto_labels_df(monkeypatch):
    _install_toponymy_mocks(monkeypatch)

    t = Typologist(
        n_facets=1,
        topic_embedder=_FakeTopicEmbedder(),
        naming_llm=lambda p: "x",
        schema_llm=_make_schema_llm(
            [{"name": "f", "type": "categorical", "values": ["a", "b"], "definition": "d"}]
        ),
        labeling_llm=lambda p: "a",
    )

    docs = pd.Series(["d1", "d2", "d3"], index=[100, 200, 300])
    emb = np.zeros((3, 4), dtype=np.float32)
    t.fit(docs, emb)

    assert list(t.labels_df_.index) == [100, 200, 300]


def test_fit_records_callable_labeling_model_as_none(monkeypatch):
    _install_toponymy_mocks(monkeypatch)

    t = Typologist(
        n_facets=1,
        topic_embedder=_FakeTopicEmbedder(),
        naming_llm=lambda p: "x",
        schema_llm=_make_schema_llm(
            [{"name": "f", "type": "categorical", "values": ["a", "b"], "definition": "d"}]
        ),
        labeling_llm=lambda p: "a",  # callable => model unknown
    )
    t.fit(["d1", "d2"], np.zeros((2, 4), dtype=np.float32))
    assert t.schema_[0]["labeling_model"] is None


def test_fit_with_metadata_runs_pre_erasure(monkeypatch):
    _install_toponymy_mocks(monkeypatch)

    t = Typologist(
        n_facets=1,
        topic_embedder=_FakeTopicEmbedder(),
        naming_llm=lambda p: "x",
        schema_llm=_make_schema_llm(
            [{"name": "f", "type": "categorical", "values": ["a", "b"], "definition": "d"}]
        ),
        labeling_llm=lambda p: "a",
    )

    rng = np.random.default_rng(0)
    n = 20
    docs = [f"doc_{i}" for i in range(n)]
    emb = rng.standard_normal((n, 8)).astype(np.float32)
    meta = pd.DataFrame({"source": ["s1"] * 10 + ["s2"] * 10})

    t.fit(docs, emb, metadata=meta)
    # working embeddings were through LEACE; shape preserved
    assert t.embeddings_residualized_.shape == (n, 8)


def test_apply_schema_applies_stored_template():
    schema = [
        {
            "name": "sentiment",
            "type": "categorical",
            "values": ["positive", "negative"],
            "definition": "tone",
            "labeling_prompt_template": "Classify:\n{document}\nPick one.",
            "labeling_model": "claude-haiku-4-5",
        }
    ]

    result = apply_schema(
        schema,
        documents=["Great!", "Terrible."],
        llm=lambda p: "positive" if "Great" in p else "negative",
    )

    assert result.shape == (2, 1)
    assert list(result["sentiment"]) == ["positive", "negative"]


def test_apply_schema_accepts_single_facet_dict():
    facet = {
        "name": "f",
        "type": "categorical",
        "values": ["a", "b"],
        "definition": "d",
        "labeling_prompt_template": "{document}",
        "labeling_model": "m",
    }
    result = apply_schema(facet, documents=["x"], llm=lambda p: "a")
    assert result.shape == (1, 1)
    assert list(result.columns) == ["f"]


def test_apply_schema_requires_llm_when_facet_has_no_labeling_model():
    import pytest

    schema = [
        {
            "name": "f",
            "type": "categorical",
            "values": ["a", "b"],
            "definition": "d",
            "labeling_prompt_template": "{document}",
            "labeling_model": None,
        }
    ]
    with pytest.raises(RuntimeError, match="labeling_model=None"):
        apply_schema(schema, documents=["x"])


def test_apply_schema_preserves_series_index():
    schema = [
        {
            "name": "f",
            "type": "categorical",
            "values": ["a", "b"],
            "definition": "d",
            "labeling_prompt_template": "{document}",
            "labeling_model": "m",
        }
    ]
    docs = pd.Series(["d1", "d2"], index=[42, 99])
    result = apply_schema(schema, docs, llm=lambda p: "a")
    assert list(result.index) == [42, 99]
