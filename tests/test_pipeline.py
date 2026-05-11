import numpy as np
import pandas as pd
import pytest

from typologist._pipeline import _normalize_inputs


def _embeddings(n=3, d=4):
    return np.random.default_rng(0).random((n, d))


def test_list_documents_get_range_index():
    result = _normalize_inputs(["a", "b", "c"], _embeddings(3, 4))
    assert isinstance(result.documents, pd.Series)
    assert list(result.documents.index) == [0, 1, 2]
    assert result.documents.tolist() == ["a", "b", "c"]


def test_series_documents_preserve_index():
    docs = pd.Series(["a", "b", "c"], index=[10, 20, 30])
    result = _normalize_inputs(docs, _embeddings(3, 4))
    assert list(result.documents.index) == [10, 20, 30]


def test_empty_documents_rejected():
    with pytest.raises(ValueError, match="non-empty"):
        _normalize_inputs([], np.zeros((0, 4)))


def test_non_string_documents_rejected():
    with pytest.raises(TypeError, match="must be str"):
        _normalize_inputs([1, 2, 3], _embeddings(3, 4))


def test_non_array_embeddings_rejected():
    with pytest.raises(TypeError, match="numpy ndarray"):
        _normalize_inputs(["a", "b"], [[1.0, 2.0], [3.0, 4.0]])


def test_one_dim_embeddings_rejected():
    with pytest.raises(ValueError, match="2-dimensional"):
        _normalize_inputs(["a", "b"], np.array([1.0, 2.0]))


def test_int_embeddings_rejected():
    with pytest.raises(TypeError, match="floating-point"):
        _normalize_inputs(["a", "b"], np.array([[1, 2], [3, 4]]))


def test_length_mismatch_rejected():
    with pytest.raises(ValueError, match="rows but documents"):
        _normalize_inputs(["a", "b", "c"], _embeddings(2, 4))


def test_embeddings_defensively_copied():
    original = _embeddings(3, 4)
    result = _normalize_inputs(["a", "b", "c"], original)
    result.embeddings[0, 0] = 999.0
    assert original[0, 0] != 999.0
