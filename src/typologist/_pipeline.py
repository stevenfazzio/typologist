from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from concept_erasure import LeaceEraser


@dataclass(frozen=True)
class _Inputs:
    """Normalized inputs to the Typologist pipeline."""

    documents: pd.Series
    embeddings: np.ndarray
    metadata: pd.DataFrame | None
    was_normalized: bool


def _normalize_inputs(
    documents: list[str] | pd.Series,
    embeddings: np.ndarray,
    metadata: pd.DataFrame | None,
) -> _Inputs:
    if isinstance(documents, pd.Series):
        docs_series = documents
    elif isinstance(documents, list):
        docs_series = pd.Series(documents)
    else:
        raise TypeError(
            f"documents must be a list or pandas Series, got {type(documents).__name__}"
        )

    if not all(isinstance(d, str) for d in docs_series):
        raise TypeError("all entries in documents must be str")

    n_docs = len(docs_series)
    if n_docs == 0:
        raise ValueError("documents must be non-empty")

    if not isinstance(embeddings, np.ndarray):
        raise TypeError(f"embeddings must be a numpy ndarray, got {type(embeddings).__name__}")
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2-dimensional, got shape {embeddings.shape}")
    if not np.issubdtype(embeddings.dtype, np.floating):
        raise TypeError(f"embeddings must have a floating-point dtype, got {embeddings.dtype}")
    if embeddings.shape[0] != n_docs:
        raise ValueError(
            f"embeddings has {embeddings.shape[0]} rows but documents has {n_docs} entries"
        )

    if metadata is not None:
        if not isinstance(metadata, pd.DataFrame):
            raise TypeError(f"metadata must be a pandas DataFrame, got {type(metadata).__name__}")
        if len(metadata) != n_docs:
            raise ValueError(
                f"metadata has {len(metadata)} rows but documents has {n_docs} entries"
            )
        if not metadata.index.equals(docs_series.index):
            raise ValueError("metadata index must match the documents index")

    return _Inputs(
        documents=docs_series,
        embeddings=embeddings.copy(),
        metadata=metadata,
        was_normalized=_is_l2_normalized(embeddings),
    )


def _is_l2_normalized(x: np.ndarray, atol: float = 1e-3) -> bool:
    """Return True iff every row of x has unit L2 norm within `atol`."""
    norms = np.linalg.norm(x, axis=1)
    return bool(np.allclose(norms, 1.0, atol=atol))


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    """Return a copy of x with each row rescaled to unit L2 norm (zero rows left as-is)."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return x / norms


def _erase_metadata(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    was_normalized: bool,
) -> np.ndarray:
    """Erase all metadata columns from embeddings via LEACE.

    Every column is treated as categorical and one-hot encoded before LEACE fit
    (the LEACE-with-int-labels gotcha: integer class labels get treated as one
    continuous axis otherwise). Bucket continuous metadata yourself before
    passing it in.
    """
    one_hot = pd.get_dummies(metadata.astype(str), dtype=float).to_numpy()

    x = torch.from_numpy(embeddings.astype(np.float32))
    z = torch.from_numpy(one_hot.astype(np.float32))

    eraser = LeaceEraser.fit(x, z)
    erased = eraser(x).numpy().astype(embeddings.dtype, copy=False)

    if was_normalized:
        erased = _l2_normalize(erased)

    return erased
