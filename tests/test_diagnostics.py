from __future__ import annotations

import math

import numpy as np
import pandas as pd

from typologist._pipeline import _build_facet_diagnostics, _NamingResult


def _naming_result(cluster_count=8, hierarchy_depth=3):
    return _NamingResult(
        topic_names=[["x"]],
        topic_name_vectors=[np.array([], dtype=object)],
        cluster_count=cluster_count,
        hierarchy_depth=hierarchy_depth,
    )


def test_diagnostics_passes_through_synthesis_prompt_and_counts():
    labels = pd.Series(pd.Categorical(["a"] * 4, categories=["a", "b"]))
    out = _build_facet_diagnostics(
        synthesis_prompt="prompt text",
        naming_result=_naming_result(cluster_count=12, hierarchy_depth=4),
        labels=labels,
        embeddings_pre_erasure=np.zeros((4, 8)),
        values=["a", "b"],
    )
    assert out["synthesis_prompt"] == "prompt text"
    assert out["cluster_count"] == 12
    assert out["hierarchy_depth"] == 4


def test_diagnostics_entropy_uniform_is_log2_n_values():
    labels = pd.Series(pd.Categorical(["a", "b", "c", "d"], categories=["a", "b", "c", "d"]))
    out = _build_facet_diagnostics(
        synthesis_prompt="",
        naming_result=_naming_result(),
        labels=labels,
        embeddings_pre_erasure=np.zeros((4, 8)),
        values=["a", "b", "c", "d"],
    )
    assert out["entropy_bits"]["uniform"] == math.log2(4)


def test_diagnostics_entropy_observed_matches_distribution():
    # 8 docs: 4 "a", 2 "b", 2 "c" -> probs 0.5, 0.25, 0.25 -> entropy 1.5 bits
    labels = pd.Series(
        pd.Categorical(["a", "a", "a", "a", "b", "b", "c", "c"], categories=["a", "b", "c"])
    )
    out = _build_facet_diagnostics(
        synthesis_prompt="",
        naming_result=_naming_result(),
        labels=labels,
        embeddings_pre_erasure=np.zeros((8, 8)),
        values=["a", "b", "c"],
    )
    assert math.isclose(out["entropy_bits"]["observed"], 1.5, abs_tol=1e-6)


def test_diagnostics_entropy_delta_is_observed_minus_uniform():
    # uniform distribution -> entropy == uniform -> delta == 0
    labels = pd.Series(pd.Categorical(["a", "b", "c", "d"], categories=["a", "b", "c", "d"]))
    out = _build_facet_diagnostics(
        synthesis_prompt="",
        naming_result=_naming_result(),
        labels=labels,
        embeddings_pre_erasure=np.zeros((4, 8)),
        values=["a", "b", "c", "d"],
    )
    assert math.isclose(out["entropy_bits"]["delta"], 0.0, abs_tol=1e-6)

    # all one label -> observed entropy 0 -> delta = -log2(4)
    labels_concentrated = pd.Series(pd.Categorical(["a"] * 4, categories=["a", "b", "c", "d"]))
    out2 = _build_facet_diagnostics(
        synthesis_prompt="",
        naming_result=_naming_result(),
        labels=labels_concentrated,
        embeddings_pre_erasure=np.zeros((4, 8)),
        values=["a", "b", "c", "d"],
    )
    assert math.isclose(out2["entropy_bits"]["observed"], 0.0, abs_tol=1e-6)
    assert math.isclose(out2["entropy_bits"]["delta"], -math.log2(4), abs_tol=1e-6)


def test_diagnostics_exemplars_are_nearest_to_centroid():
    # Construct 4 "a" docs where doc at position 0 is at the centroid and
    # others are progressively farther; verify exemplar order.
    embeddings = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
        ],
        dtype=np.float32,
    )
    # centroid is (1.5, 0.0); distance from each: 1.5, 0.5, 0.5, 1.5
    # so positions 1 and 2 are nearest, then 0 and 3
    labels = pd.Series(
        pd.Categorical(["a"] * 4, categories=["a", "b"]),
        index=[10, 20, 30, 40],
    )
    out = _build_facet_diagnostics(
        synthesis_prompt="",
        naming_result=_naming_result(),
        labels=labels,
        embeddings_pre_erasure=embeddings,
        values=["a", "b"],
        exemplars_k=2,
    )
    # exemplars should correspond to positions 1 and 2, i.e., index values 20 and 30
    assert set(out["exemplars_per_value"]["a"]) == {20, 30}
    assert len(out["exemplars_per_value"]["a"]) == 2
    assert out["exemplars_per_value"]["b"] == []


def test_diagnostics_exemplars_respect_k_limit():
    labels = pd.Series(pd.Categorical(["a"] * 10, categories=["a", "b"]))
    out = _build_facet_diagnostics(
        synthesis_prompt="",
        naming_result=_naming_result(),
        labels=labels,
        embeddings_pre_erasure=np.random.default_rng(0).random((10, 4)),
        values=["a", "b"],
        exemplars_k=3,
    )
    assert len(out["exemplars_per_value"]["a"]) == 3
    assert len(out["exemplars_per_value"]["b"]) == 0


def test_diagnostics_exemplars_handle_fewer_docs_than_k():
    labels = pd.Series(pd.Categorical(["a", "b", "a"], categories=["a", "b"]))
    out = _build_facet_diagnostics(
        synthesis_prompt="",
        naming_result=_naming_result(),
        labels=labels,
        embeddings_pre_erasure=np.ones((3, 4)),
        values=["a", "b"],
        exemplars_k=10,
    )
    # only 2 docs with "a", 1 with "b"; exemplars must cap at actual counts
    assert len(out["exemplars_per_value"]["a"]) == 2
    assert len(out["exemplars_per_value"]["b"]) == 1


def test_diagnostics_exemplars_preserve_user_index():
    labels = pd.Series(
        pd.Categorical(["a"] * 3, categories=["a"]),
        index=["doc_x", "doc_y", "doc_z"],
    )
    out = _build_facet_diagnostics(
        synthesis_prompt="",
        naming_result=_naming_result(),
        labels=labels,
        embeddings_pre_erasure=np.ones((3, 4)),
        values=["a"],
        exemplars_k=5,
    )
    assert set(out["exemplars_per_value"]["a"]) == {"doc_x", "doc_y", "doc_z"}
