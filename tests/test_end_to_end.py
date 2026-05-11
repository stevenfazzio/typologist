from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

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


def _make_schema_llm(facets: list[dict]):
    """Return a callable that emits a one-shot multi-facet JSON response.

    The new fit() does a single call to schema_llm.call_structured, expecting
    a response of the form ``{"facets": [...]}`` with n_facets entries.
    """

    def call(prompt: str, **kwargs) -> str:
        return json.dumps({"facets": facets})

    return call


def test_fit_populates_schema_and_diagnostics(monkeypatch):
    _install_toponymy_mocks(monkeypatch)

    t = Typologist(
        n_facets=2,
        topic_embedder=_FakeTopicEmbedder(),
        naming_llm=lambda p: "ignored",
        schema_llm=_make_schema_llm(
            [
                {
                    "name": "facet_one",
                    "kind": "categorical",
                    "values": ["a", "b"],
                    "definition": "first axis",
                },
                {
                    "name": "facet_two",
                    "kind": "categorical",
                    "values": ["x", "y"],
                    "definition": "second axis",
                },
            ]
        ),
        labeling_llm=lambda p: "ignored",  # provenance only; fit never calls it
    )

    docs = ["doc_1", "doc_2", "doc_3"]
    emb = np.random.default_rng(0).random((3, 4)).astype(np.float32)
    t.fit(docs, emb)

    assert len(t.schema_) == 2
    assert [f["name"] for f in t.schema_] == ["facet_one", "facet_two"]
    assert t.schema_[0]["values"] == ["a", "b", "Other"]
    assert t.schema_[1]["values"] == ["x", "y", "Other"]
    assert "{document}" in t.schema_[0]["labeling_prompt_template"]

    # New diagnostics shape: a single dict for the whole run, not per-facet
    assert isinstance(t.diagnostics_, dict)
    assert "synthesis_prompt" in t.diagnostics_
    assert "cluster_count" in t.diagnostics_
    assert "hierarchy_depth" in t.diagnostics_

    # No more labels_df_ or embeddings_residualized_ attributes on the fitted object
    assert not hasattr(t, "labels_df_")
    assert not hasattr(t, "embeddings_residualized_")


def test_fit_records_callable_labeling_model_as_none(monkeypatch):
    _install_toponymy_mocks(monkeypatch)

    t = Typologist(
        n_facets=1,
        topic_embedder=_FakeTopicEmbedder(),
        naming_llm=lambda p: "x",
        schema_llm=_make_schema_llm(
            [{"name": "f", "kind": "categorical", "values": ["a", "b"], "definition": "d"}]
        ),
        labeling_llm=lambda p: "ignored",
    )
    t.fit(["d1", "d2"], np.zeros((2, 4), dtype=np.float32))
    assert t.schema_[0]["labeling_model"] is None


def test_fit_appends_other_when_absent(monkeypatch):
    _install_toponymy_mocks(monkeypatch)

    t = Typologist(
        n_facets=1,
        topic_embedder=_FakeTopicEmbedder(),
        naming_llm=lambda p: "x",
        schema_llm=_make_schema_llm(
            [{"name": "f", "kind": "categorical", "values": ["a", "b", "c"], "definition": "d"}]
        ),
        labeling_llm=lambda p: "ignored",
    )
    t.fit(["d1", "d2"], np.zeros((2, 4), dtype=np.float32))
    assert t.schema_[0]["values"] == ["a", "b", "c", "Other"]


def test_fit_dedupes_other_case_insensitively(monkeypatch):
    _install_toponymy_mocks(monkeypatch)

    t = Typologist(
        n_facets=1,
        topic_embedder=_FakeTopicEmbedder(),
        naming_llm=lambda p: "x",
        schema_llm=_make_schema_llm(
            [{"name": "f", "kind": "categorical", "values": ["a", "b", "other"], "definition": "d"}]
        ),
        labeling_llm=lambda p: "ignored",
    )
    t.fit(["d1", "d2"], np.zeros((2, 4), dtype=np.float32))
    # LLM lowercased "other" already present; the duplicate "Other" should not be appended
    assert t.schema_[0]["values"] == ["a", "b", "other"]


def test_fit_rejects_facet_name_collision(monkeypatch):
    _install_toponymy_mocks(monkeypatch)

    t = Typologist(
        n_facets=2,
        topic_embedder=_FakeTopicEmbedder(),
        naming_llm=lambda p: "x",
        schema_llm=_make_schema_llm(
            [
                {"name": "f", "kind": "categorical", "values": ["a", "b"], "definition": "d"},
                {"name": "f", "kind": "categorical", "values": ["x", "y"], "definition": "d"},
            ]
        ),
        labeling_llm=lambda p: "ignored",
    )
    with pytest.raises(RuntimeError, match="duplicate facet name"):
        t.fit(["d1", "d2"], np.zeros((2, 4), dtype=np.float32))


def test_fit_rejects_duplicate_values(monkeypatch):
    _install_toponymy_mocks(monkeypatch)

    t = Typologist(
        n_facets=1,
        topic_embedder=_FakeTopicEmbedder(),
        naming_llm=lambda p: "x",
        schema_llm=_make_schema_llm(
            [{"name": "f", "kind": "categorical", "values": ["a", "a"], "definition": "d"}]
        ),
        labeling_llm=lambda p: "ignored",
    )
    with pytest.raises(RuntimeError, match="duplicate values"):
        t.fit(["d1", "d2"], np.zeros((2, 4), dtype=np.float32))


def test_fit_rejects_too_few_values(monkeypatch):
    _install_toponymy_mocks(monkeypatch)

    t = Typologist(
        n_facets=1,
        topic_embedder=_FakeTopicEmbedder(),
        naming_llm=lambda p: "x",
        schema_llm=_make_schema_llm(
            [{"name": "f", "kind": "categorical", "values": ["only_one"], "definition": "d"}]
        ),
        labeling_llm=lambda p: "ignored",
    )
    with pytest.raises(RuntimeError, match="fewer than 2 values"):
        t.fit(["d1", "d2"], np.zeros((2, 4), dtype=np.float32))


def test_fit_rejects_missing_required_field(monkeypatch):
    _install_toponymy_mocks(monkeypatch)

    # missing "values" and "definition"
    t = Typologist(
        n_facets=1,
        topic_embedder=_FakeTopicEmbedder(),
        naming_llm=lambda p: "x",
        schema_llm=_make_schema_llm([{"name": "f", "kind": "categorical"}]),
        labeling_llm=lambda p: "ignored",
    )
    with pytest.raises(RuntimeError, match="missing required field"):
        t.fit(["d1", "d2"], np.zeros((2, 4), dtype=np.float32))


def test_fit_rejects_wrong_facet_count(monkeypatch):
    _install_toponymy_mocks(monkeypatch)

    # Asks for 3 but the schema_llm only returns 2
    t = Typologist(
        n_facets=3,
        topic_embedder=_FakeTopicEmbedder(),
        naming_llm=lambda p: "x",
        schema_llm=_make_schema_llm(
            [
                {"name": "a", "kind": "categorical", "values": ["x", "y"], "definition": "d"},
                {"name": "b", "kind": "categorical", "values": ["x", "y"], "definition": "d"},
            ]
        ),
        labeling_llm=lambda p: "ignored",
    )
    with pytest.raises(RuntimeError, match="expected 3"):
        t.fit(["d1", "d2"], np.zeros((2, 4), dtype=np.float32))


def test_fit_with_use_toponymy_false_routes_through_homemade(monkeypatch):
    """When use_toponymy=False, Toponymy must not be touched; EVoC drives naming."""
    from typologist import _homemade, _pipeline

    class _ExplodeIfCalled:
        def __init__(self, **kwargs):
            pass

        def fit(self, *args, **kwargs):
            raise AssertionError("Toponymy should not be invoked when use_toponymy=False")

    monkeypatch.setattr(_pipeline, "Toponymy", _ExplodeIfCalled)

    class _FakeEVoC:
        def __init__(self, **kwargs):
            pass

        def fit(self, x):
            self.labels_ = np.array([0, 0, 1, 1])
            self.cluster_layers_ = [self.labels_]
            return self

    monkeypatch.setattr(_homemade, "EVoC", _FakeEVoC)

    t = Typologist(
        n_facets=1,
        topic_embedder=_FakeTopicEmbedder(),
        naming_llm=lambda p: "named_cluster",
        schema_llm=_make_schema_llm(
            [{"name": "f", "kind": "categorical", "values": ["a", "b"], "definition": "d"}]
        ),
        labeling_llm=lambda p: "ignored",
        use_toponymy=False,
    )

    t.fit(["d1", "d2", "d3", "d4"], np.eye(4, 8, dtype=np.float32))

    assert t.diagnostics_["hierarchy_depth"] == 1
    assert t.diagnostics_["cluster_count"] == 2


# --- apply_schema ---


def test_apply_schema_applies_stored_template():
    schema = [
        {
            "name": "sentiment",
            "kind": "categorical",
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
        "kind": "categorical",
        "values": ["a", "b"],
        "definition": "d",
        "labeling_prompt_template": "{document}",
        "labeling_model": "m",
    }
    result = apply_schema(facet, documents=["x"], llm=lambda p: "a")
    assert result.shape == (1, 1)
    assert list(result.columns) == ["f"]


def test_apply_schema_requires_llm_even_when_labeling_model_set():
    schema = [
        {
            "name": "f",
            "kind": "categorical",
            "values": ["a", "b"],
            "definition": "d",
            "labeling_prompt_template": "{document}",
            "labeling_model": "anthropic:claude-haiku-4-5",
        }
    ]
    with pytest.raises(TypeError, match="apply_schema requires `llm="):
        apply_schema(schema, documents=["x"])


def test_apply_schema_requires_llm_when_labeling_model_is_none():
    schema = [
        {
            "name": "f",
            "kind": "categorical",
            "values": ["a", "b"],
            "definition": "d",
            "labeling_prompt_template": "{document}",
            "labeling_model": None,
        }
    ]
    with pytest.raises(TypeError, match="apply_schema requires `llm="):
        apply_schema(schema, documents=["x"])


def test_apply_schema_preserves_series_index():
    schema = [
        {
            "name": "f",
            "kind": "categorical",
            "values": ["a", "b"],
            "definition": "d",
            "labeling_prompt_template": "{document}",
            "labeling_model": "m",
        }
    ]
    docs = pd.Series(["d1", "d2"], index=[42, 99])
    result = apply_schema(schema, docs, llm=lambda p: "a")
    assert list(result.index) == [42, 99]
