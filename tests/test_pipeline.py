import numpy as np
import pandas as pd
import pytest

from typologist._pipeline import (
    _describe_erased_metadata,
    _erase_metadata,
    _is_l2_normalized,
    _l2_normalize,
    _normalize_inputs,
)


def _embeddings(n=3, d=4):
    return np.random.default_rng(0).random((n, d))


def _unit_embeddings(n=50, d=16, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, d)).astype(np.float32)
    return _l2_normalize(x)


def test_list_documents_get_range_index():
    result = _normalize_inputs(["a", "b", "c"], _embeddings(3, 4), None)
    assert isinstance(result.documents, pd.Series)
    assert list(result.documents.index) == [0, 1, 2]
    assert result.documents.tolist() == ["a", "b", "c"]


def test_series_documents_preserve_index():
    docs = pd.Series(["a", "b", "c"], index=[10, 20, 30])
    result = _normalize_inputs(docs, _embeddings(3, 4), None)
    assert list(result.documents.index) == [10, 20, 30]


def test_empty_documents_rejected():
    with pytest.raises(ValueError, match="non-empty"):
        _normalize_inputs([], np.zeros((0, 4)), None)


def test_non_string_documents_rejected():
    with pytest.raises(TypeError, match="must be str"):
        _normalize_inputs([1, 2, 3], _embeddings(3, 4), None)


def test_non_array_embeddings_rejected():
    with pytest.raises(TypeError, match="numpy ndarray"):
        _normalize_inputs(["a", "b"], [[1.0, 2.0], [3.0, 4.0]], None)


def test_one_dim_embeddings_rejected():
    with pytest.raises(ValueError, match="2-dimensional"):
        _normalize_inputs(["a", "b"], np.array([1.0, 2.0]), None)


def test_int_embeddings_rejected():
    with pytest.raises(TypeError, match="floating-point"):
        _normalize_inputs(["a", "b"], np.array([[1, 2], [3, 4]]), None)


def test_length_mismatch_rejected():
    with pytest.raises(ValueError, match="rows but documents"):
        _normalize_inputs(["a", "b", "c"], _embeddings(2, 4), None)


def test_metadata_accepted_when_valid():
    docs = pd.Series(["a", "b", "c"], index=[10, 20, 30])
    metadata = pd.DataFrame({"source": ["x", "y", "z"]}, index=[10, 20, 30])
    result = _normalize_inputs(docs, _embeddings(3, 4), metadata)
    assert result.metadata is not None
    assert list(result.metadata.index) == [10, 20, 30]


def test_non_dataframe_metadata_rejected():
    with pytest.raises(TypeError, match="DataFrame"):
        _normalize_inputs(["a", "b"], _embeddings(2, 4), {"source": ["x", "y"]})


def test_metadata_length_mismatch_rejected():
    metadata = pd.DataFrame({"source": ["x", "y"]})
    with pytest.raises(ValueError, match="rows but documents"):
        _normalize_inputs(["a", "b", "c"], _embeddings(3, 4), metadata)


def test_metadata_index_mismatch_rejected():
    docs = pd.Series(["a", "b", "c"], index=[10, 20, 30])
    metadata = pd.DataFrame({"source": ["x", "y", "z"]}, index=[0, 1, 2])
    with pytest.raises(ValueError, match="index must match"):
        _normalize_inputs(docs, _embeddings(3, 4), metadata)


def test_embeddings_defensively_copied():
    original = _embeddings(3, 4)
    result = _normalize_inputs(["a", "b", "c"], original, None)
    result.embeddings[0, 0] = 999.0
    assert original[0, 0] != 999.0


def test_normalized_input_detected():
    x = _unit_embeddings(10, 8)
    result = _normalize_inputs(["a"] * 10, x, None)
    assert result.was_normalized is True


def test_unnormalized_input_detected():
    x = _embeddings(10, 8) * 17.0
    result = _normalize_inputs(["a"] * 10, x, None)
    assert result.was_normalized is False


def test_is_l2_normalized_helper():
    assert _is_l2_normalized(_unit_embeddings(10, 4)) is True
    assert _is_l2_normalized(_embeddings(10, 4) * 3.0) is False


def test_l2_normalize_helper_produces_unit_rows():
    x = _embeddings(10, 4) * 17.0
    normalized = _l2_normalize(x)
    np.testing.assert_allclose(np.linalg.norm(normalized, axis=1), 1.0, atol=1e-6)


def test_l2_normalize_handles_zero_rows():
    x = np.zeros((3, 4))
    normalized = _l2_normalize(x)
    np.testing.assert_array_equal(normalized, np.zeros((3, 4)))


def test_erase_metadata_equalizes_class_means():
    rng = np.random.default_rng(42)
    n = 80
    d = 16
    class_ids = rng.integers(0, 4, size=n)
    shifts = rng.standard_normal((4, d)) * 2.0
    x = rng.standard_normal((n, d)).astype(np.float32) + shifts[class_ids]
    metadata = pd.DataFrame({"klass": class_ids})

    erased = _erase_metadata(x, metadata, was_normalized=False)

    assert erased.shape == x.shape
    per_class_means_before = np.stack([x[class_ids == k].mean(axis=0) for k in range(4)])
    per_class_means_after = np.stack([erased[class_ids == k].mean(axis=0) for k in range(4)])
    before_spread = per_class_means_before.std(axis=0).mean()
    after_spread = per_class_means_after.std(axis=0).mean()
    assert after_spread < before_spread * 0.1


def test_erase_metadata_renormalizes_when_input_was_normalized():
    rng = np.random.default_rng(7)
    x = _l2_normalize(rng.standard_normal((40, 16)).astype(np.float32))
    metadata = pd.DataFrame({"klass": rng.integers(0, 3, size=40)})

    erased = _erase_metadata(x, metadata, was_normalized=True)
    np.testing.assert_allclose(np.linalg.norm(erased, axis=1), 1.0, atol=1e-5)


def test_erase_metadata_leaves_norms_alone_when_input_not_normalized():
    rng = np.random.default_rng(7)
    x = rng.standard_normal((40, 16)).astype(np.float32) * 5.0
    metadata = pd.DataFrame({"klass": rng.integers(0, 3, size=40)})

    erased = _erase_metadata(x, metadata, was_normalized=False)
    assert np.linalg.norm(erased, axis=1).mean() > 1.5


def test_erase_metadata_handles_multi_column_metadata():
    rng = np.random.default_rng(13)
    n = 60
    x = rng.standard_normal((n, 12)).astype(np.float32)
    metadata = pd.DataFrame(
        {
            "region": rng.choice(["us", "eu", "asia"], size=n),
            "tier": rng.choice(["free", "pro"], size=n),
        }
    )
    erased = _erase_metadata(x, metadata, was_normalized=False)
    assert erased.shape == x.shape


def test_erase_metadata_preserves_dtype():
    x32 = _embeddings(20, 8).astype(np.float32)
    x64 = _embeddings(20, 8).astype(np.float64)
    metadata = pd.DataFrame({"klass": [0, 1] * 10})

    assert _erase_metadata(x32, metadata, was_normalized=False).dtype == np.float32
    assert _erase_metadata(x64, metadata, was_normalized=False).dtype == np.float64


def test_describe_erased_metadata_infers_string_column_as_categorical():
    meta = pd.DataFrame({"color": ["red", "blue", "red", "green"]})
    descs = _describe_erased_metadata(meta)
    assert len(descs) == 1
    assert descs[0]["name"] == "color"
    assert descs[0]["type"] == "categorical"
    assert set(descs[0]["values_shown"]) == {"red", "blue", "green"}
    assert descs[0]["total_unique"] == 3
    assert descs[0]["truncated"] is False


def test_describe_erased_metadata_infers_numeric_column_as_ordinal():
    meta = pd.DataFrame({"rating": [3.0, 1.0, 5.0, 2.0, 4.0, 3.0]})
    descs = _describe_erased_metadata(meta)
    assert descs[0]["type"] == "ordinal"
    # sorted, deduplicated
    assert descs[0]["values_shown"] == ["1.0", "2.0", "3.0", "4.0", "5.0"]


def test_describe_erased_metadata_infers_ordered_categorical_as_ordinal():
    meta = pd.DataFrame(
        {
            "size": pd.Categorical(
                ["M", "S", "L", "M"],
                categories=["S", "M", "L", "XL"],
                ordered=True,
            )
        }
    )
    descs = _describe_erased_metadata(meta)
    assert descs[0]["type"] == "ordinal"
    assert descs[0]["values_shown"] == ["S", "M", "L", "XL"]


def test_describe_erased_metadata_treats_unordered_categorical_as_categorical():
    meta = pd.DataFrame(
        {
            "tag": pd.Categorical(
                ["a", "b", "a"],
                categories=["a", "b"],
                ordered=False,
            )
        }
    )
    descs = _describe_erased_metadata(meta)
    assert descs[0]["type"] == "categorical"


def test_describe_erased_metadata_truncates_high_cardinality():
    meta = pd.DataFrame({"country": [f"C{i:03d}" for i in range(50)]})
    descs = _describe_erased_metadata(meta)
    assert descs[0]["truncated"] is True
    assert len(descs[0]["values_shown"]) == 8
    assert descs[0]["total_unique"] == 50
