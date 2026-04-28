from __future__ import annotations

import numpy as np
import pandas as pd

from typologist._pipeline import _run_toponymy
from typologist.llm import _CallableLLM


class _FakeToponymy:
    """Replaces Toponymy for boundary-wiring tests without real LLM/clustering calls."""

    last_init_kwargs: dict = {}
    last_fit_args: tuple = ()

    def __init__(self, **kwargs):
        _FakeToponymy.last_init_kwargs = kwargs

    def fit(self, objects, embedding_vectors, clusterable_vectors):
        _FakeToponymy.last_fit_args = (objects, embedding_vectors, clusterable_vectors)
        n = len(objects)
        self.topic_names_ = [
            ["fine_a", "fine_b", "fine_c"],
            ["coarse_x", "coarse_y"],
        ]
        self.topic_name_vectors_ = [
            np.array(["fine_a"] * n, dtype=object),
            np.array(["coarse_x"] * n, dtype=object),
        ]
        return self


class _FakeEVoCClusterer:
    last_init_kwargs: dict = {}

    def __init__(self, **kwargs):
        _FakeEVoCClusterer.last_init_kwargs = kwargs


class _FakeEmbedder:
    def encode(self, texts, show_progress_bar=False):
        return np.zeros((len(texts), 8))


def test_run_toponymy_wires_construction_kwargs(monkeypatch):
    from typologist import _pipeline

    monkeypatch.setattr(_pipeline, "Toponymy", _FakeToponymy)
    monkeypatch.setattr(_pipeline, "EVoCClusterer", _FakeEVoCClusterer)

    docs = pd.Series(["a", "b", "c"], index=[10, 20, 30])
    emb = np.zeros((3, 8), dtype=np.float32)

    _run_toponymy(
        documents=docs,
        embeddings=emb,
        topic_embedder=_FakeEmbedder(),
        naming_llm=_CallableLLM(lambda p: "x"),
        object_description="paper",
        corpus_description="papers",
        verbose=False,
    )

    assert _FakeToponymy.last_init_kwargs["object_description"] == "paper"
    assert _FakeToponymy.last_init_kwargs["corpus_description"] == "papers"
    assert _FakeToponymy.last_init_kwargs["verbose"] is False
    assert isinstance(_FakeToponymy.last_init_kwargs["clusterer"], _FakeEVoCClusterer)
    assert _FakeEVoCClusterer.last_init_kwargs["verbose"] is False


def test_run_toponymy_fits_on_documents_list_and_embeddings():
    from typologist import _pipeline

    # monkeypatch via setattr so the test is self-contained
    orig_topo = _pipeline.Toponymy
    orig_cl = _pipeline.EVoCClusterer
    _pipeline.Toponymy = _FakeToponymy
    _pipeline.EVoCClusterer = _FakeEVoCClusterer
    try:
        docs = pd.Series(["doc1", "doc2"], index=[100, 200])
        emb = np.ones((2, 4), dtype=np.float32) * 0.5

        _run_toponymy(
            documents=docs,
            embeddings=emb,
            topic_embedder=_FakeEmbedder(),
            naming_llm=_CallableLLM(lambda p: "x"),
            object_description="obj",
            corpus_description="corpus",
            verbose=False,
        )

        objects, emb_v, clusterable = _FakeToponymy.last_fit_args
        assert objects == ["doc1", "doc2"]
        np.testing.assert_array_equal(emb_v, emb)
        # EVoCClusterer ignores clusterable_vectors but Toponymy.fit still takes it;
        # we pass the same embeddings for both
        np.testing.assert_array_equal(clusterable, emb)
    finally:
        _pipeline.Toponymy = orig_topo
        _pipeline.EVoCClusterer = orig_cl


def test_run_toponymy_extracts_expected_result_shape(monkeypatch):
    from typologist import _pipeline

    monkeypatch.setattr(_pipeline, "Toponymy", _FakeToponymy)
    monkeypatch.setattr(_pipeline, "EVoCClusterer", _FakeEVoCClusterer)

    docs = pd.Series(["a", "b", "c"])
    emb = np.zeros((3, 8), dtype=np.float32)

    result = _run_toponymy(
        documents=docs,
        embeddings=emb,
        topic_embedder=_FakeEmbedder(),
        naming_llm=_CallableLLM(lambda p: "x"),
        object_description="doc",
        corpus_description="corpus",
        verbose=False,
    )

    assert result.hierarchy_depth == 2
    # layer 0 had 3 clusters, layer 1 had 2; max is 3
    assert result.cluster_count == 3
    assert len(result.topic_names) == 2
    assert len(result.topic_name_vectors) == 2
    assert result.topic_name_vectors[0].shape == (3,)


def test_run_toponymy_forwards_verbose_flag(monkeypatch):
    from typologist import _pipeline

    monkeypatch.setattr(_pipeline, "Toponymy", _FakeToponymy)
    monkeypatch.setattr(_pipeline, "EVoCClusterer", _FakeEVoCClusterer)

    _run_toponymy(
        documents=pd.Series(["a"]),
        embeddings=np.zeros((1, 4), dtype=np.float32),
        topic_embedder=_FakeEmbedder(),
        naming_llm=_CallableLLM(lambda p: "x"),
        object_description="doc",
        corpus_description="corpus",
        verbose=True,
    )

    assert _FakeToponymy.last_init_kwargs["verbose"] is True
    assert _FakeEVoCClusterer.last_init_kwargs["verbose"] is True
