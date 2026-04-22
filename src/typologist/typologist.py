from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd


class Typologist:
    """Extract a categorical schema and per-document labels from a corpus.

    See ``docs/design.md`` for the full 0.1 contract.
    """

    def __init__(
        self,
        n_facets: int,
        topic_embedder: Any,
        object_description: str = "objects",
        corpus_description: str = "collection of objects",
        naming_llm: str | Callable[..., str] = "claude-haiku-4-5",
        schema_llm: str | Callable[..., str] = "claude-opus-4-7",
        labeling_llm: str | Callable[..., str] = "claude-haiku-4-5",
        random_state: int | None = None,
        noise_label: str = "Unlabelled",
        verbose: bool = False,
    ) -> None:
        self.n_facets = n_facets
        self.topic_embedder = topic_embedder
        self.object_description = object_description
        self.corpus_description = corpus_description
        self.naming_llm = naming_llm
        self.schema_llm = schema_llm
        self.labeling_llm = labeling_llm
        self.random_state = random_state
        self.noise_label = noise_label
        self.verbose = verbose

    def fit(
        self,
        documents: list[str] | pd.Series,
        embeddings: np.ndarray,
        metadata: pd.DataFrame | None = None,
    ) -> Typologist:
        raise NotImplementedError("Typologist.fit is not yet implemented")
