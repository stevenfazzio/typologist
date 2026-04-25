from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from typologist._homemade import _run_homemade_naming, _select_exemplars
from typologist._llm import _CallableLLM


class _FakeEVoC:
    """Returns canned cluster labels passed in via class attribute."""

    next_labels: np.ndarray = np.array([])
    last_fit_input: np.ndarray | None = None

    def __init__(self, **kwargs):
        pass

    def fit(self, x):
        _FakeEVoC.last_fit_input = x
        self.labels_ = _FakeEVoC.next_labels
        return self


def _patch_evoc(monkeypatch, labels: np.ndarray) -> None:
    from typologist import _homemade

    _FakeEVoC.next_labels = labels
    monkeypatch.setattr(_homemade, "EVoC", _FakeEVoC)


def test_select_exemplars_returns_centroid_nearest():
    docs = pd.Series([f"doc_{i}" for i in range(5)])
    embeddings = np.array(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [10.0, 0.0]],
        dtype=np.float32,
    )
    positions = np.array([0, 1, 2, 3])  # exclude the outlier at position 4
    # centroid of those 4 = (1.5, 0.0); distances 1.5, 0.5, 0.5, 1.5
    out = _select_exemplars(docs, embeddings, positions, exemplars_k=2)
    assert set(out) == {"doc_1", "doc_2"}


def test_select_exemplars_caps_at_cluster_size():
    docs = pd.Series([f"doc_{i}" for i in range(3)])
    embeddings = np.zeros((3, 4), dtype=np.float32)
    out = _select_exemplars(docs, embeddings, np.array([0, 1, 2]), exemplars_k=10)
    assert len(out) == 3


def test_run_homemade_naming_skips_noise_and_names_each_cluster(monkeypatch):
    # 6 docs: cluster 0 has docs 0-1, cluster 1 has docs 2-3, noise (-1) docs 4-5.
    labels = np.array([0, 0, 1, 1, -1, -1])
    _patch_evoc(monkeypatch, labels)

    docs = pd.Series([f"doc_{i}" for i in range(6)])
    embeddings = np.eye(6, 8, dtype=np.float32)

    seen_prompts: list[str] = []

    def fake_naming(prompt: str) -> str:
        seen_prompts.append(prompt)
        # Return a deterministic name based on which exemplars are present
        for i in range(6):
            if f"doc_{i}" in prompt:
                return f"name_for_{i}"
        return "unknown"

    result = _run_homemade_naming(
        documents=docs,
        embeddings=embeddings,
        naming_llm=_CallableLLM(fake_naming),
        object_description="thing",
        corpus_description="things",
        exemplars_k=4,
        max_concurrency=1,
        verbose=False,
    )

    # Two non-noise clusters -> two prompts -> two names in a single layer.
    assert len(seen_prompts) == 2
    assert result.hierarchy_depth == 1
    assert result.cluster_count == 2
    assert len(result.topic_names) == 1
    assert len(result.topic_names[0]) == 2
    # Each prompt should have rendered the corpus_description and the object plural.
    assert all("things" in p for p in seen_prompts)
    assert all("thing" in p for p in seen_prompts)


def test_run_homemade_naming_passes_embedding_to_evoc(monkeypatch):
    labels = np.array([0, 0, 0])
    _patch_evoc(monkeypatch, labels)

    embeddings = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    _run_homemade_naming(
        documents=pd.Series(["a", "b", "c"]),
        embeddings=embeddings,
        naming_llm=_CallableLLM(lambda p: "n"),
        object_description="o",
        corpus_description="c",
        max_concurrency=1,
    )
    np.testing.assert_array_equal(_FakeEVoC.last_fit_input, embeddings)


def test_run_homemade_naming_strips_whitespace_from_names(monkeypatch):
    _patch_evoc(monkeypatch, np.array([0, 0]))

    result = _run_homemade_naming(
        documents=pd.Series(["a", "b"]),
        embeddings=np.zeros((2, 4), dtype=np.float32),
        naming_llm=_CallableLLM(lambda p: "  padded name  \n"),
        object_description="o",
        corpus_description="c",
        max_concurrency=1,
    )
    assert result.topic_names[0] == ["padded name"]


def test_run_homemade_naming_raises_when_all_noise(monkeypatch):
    _patch_evoc(monkeypatch, np.array([-1, -1, -1]))

    with pytest.raises(RuntimeError, match="no clusters"):
        _run_homemade_naming(
            documents=pd.Series(["a", "b", "c"]),
            embeddings=np.zeros((3, 4), dtype=np.float32),
            naming_llm=_CallableLLM(lambda p: "n"),
            object_description="o",
            corpus_description="c",
            max_concurrency=1,
        )


def test_run_homemade_naming_concurrent_path_preserves_cluster_order(monkeypatch):
    # Three clusters; order of names_ in the result must match the sorted cluster ids.
    labels = np.array([2, 2, 0, 0, 1, 1])
    _patch_evoc(monkeypatch, labels)

    docs = pd.Series([f"doc_{i}" for i in range(6)])
    embeddings = np.eye(6, 4, dtype=np.float32)

    def fake_naming(prompt: str) -> str:
        for i in range(6):
            if f"doc_{i}" in prompt:
                # The first doc per cluster identifies it
                return f"cluster_for_{i}"
        return "?"

    result = _run_homemade_naming(
        documents=docs,
        embeddings=embeddings,
        naming_llm=_CallableLLM(fake_naming),
        object_description="o",
        corpus_description="c",
        exemplars_k=2,
        max_concurrency=4,
    )
    # sorted cluster ids: 0, 1, 2 -> first doc indices: 2, 4, 0
    assert result.topic_names[0] == ["cluster_for_2", "cluster_for_4", "cluster_for_0"]
