from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class _Inputs:
    """Normalized inputs to the Typologist pipeline."""

    documents: pd.Series
    embeddings: np.ndarray
    metadata: pd.DataFrame | None


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
    )
