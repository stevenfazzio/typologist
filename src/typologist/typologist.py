from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from typologist._llm import _resolve_llm
from typologist._pipeline import (
    _build_facet_diagnostics,
    _classify_docs,
    _erase_metadata,
    _normalize_inputs,
    _residualize_facet,
    _run_toponymy,
    _synthesize_field,
)


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
        max_concurrency: int = 10,
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
        self.max_concurrency = max_concurrency

    def fit(
        self,
        documents: list[str] | pd.Series,
        embeddings: np.ndarray,
        metadata: pd.DataFrame | None = None,
    ) -> Typologist:
        if self.random_state is not None:
            np.random.seed(self.random_state)

        naming = _resolve_llm(self.naming_llm)
        schema_llm = _resolve_llm(self.schema_llm)
        labeling_llm = _resolve_llm(self.labeling_llm)

        inputs = _normalize_inputs(documents, embeddings, metadata)
        working = inputs.embeddings
        if inputs.metadata is not None:
            working = _erase_metadata(working, inputs.metadata, inputs.was_normalized)

        schema: list[dict] = []
        label_series: list[pd.Series] = []
        diagnostics: list[dict] = []

        for _ in range(self.n_facets):
            topo = _run_toponymy(
                documents=inputs.documents,
                embeddings=working,
                topic_embedder=self.topic_embedder,
                naming_llm=naming,
                object_description=self.object_description,
                corpus_description=self.corpus_description,
                verbose=self.verbose,
            )

            facet, synthesis_prompt = _synthesize_field(
                cluster_hierarchy=topo.topic_names,
                schema_llm=schema_llm,
                labeling_llm_model_name=labeling_llm.model_name,
                object_description=self.object_description,
                corpus_description=self.corpus_description,
                prior_facet_names=[f["name"] for f in schema],
            )

            labels = _classify_docs(
                facet=facet,
                documents=inputs.documents,
                labeling_llm=labeling_llm,
                noise_label=self.noise_label,
                max_concurrency=self.max_concurrency,
                verbose=self.verbose,
            )

            diagnostics.append(
                _build_facet_diagnostics(
                    synthesis_prompt=synthesis_prompt,
                    toponymy_result=topo,
                    labels=labels,
                    embeddings_pre_erasure=working,
                    values=facet["values"],
                )
            )

            working = _residualize_facet(
                embeddings=working,
                facet_labels=labels,
                was_normalized=inputs.was_normalized,
            )

            schema.append(facet)
            label_series.append(labels)

        self.schema_ = schema
        self.labels_df_ = (
            pd.concat(label_series, axis=1)
            if label_series
            else pd.DataFrame(index=inputs.documents.index)
        )
        self.embeddings_residualized_ = working
        self.facet_diagnostics_ = diagnostics

        return self
