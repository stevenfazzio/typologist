from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from typologist._homemade import _run_homemade_naming
from typologist._pipeline import (
    _build_facet_diagnostics,
    _classify_docs,
    _describe_erased_metadata,
    _erase_metadata,
    _normalize_inputs,
    _residualize_facet,
    _run_toponymy,
    _synthesize_facet,
)
from typologist.llm import LLM, _resolve_llm


class Typologist:
    """Extract a categorical schema and per-document labels from a corpus.

    See ``docs/design.md`` for the full 0.1 contract.
    """

    def __init__(
        self,
        n_facets: int,
        topic_embedder: Any,
        *,
        naming_llm: LLM | Callable[..., str],
        schema_llm: LLM | Callable[..., str],
        labeling_llm: LLM | Callable[..., str],
        object_description: str = "objects",
        corpus_description: str = "collection of objects",
        random_state: int | None = None,
        noise_label: str = "Unlabelled",
        verbose: bool = False,
        max_concurrency: int = 10,
        use_toponymy: bool = True,
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
        self.use_toponymy = use_toponymy

    def fit(
        self,
        documents: list[str] | pd.Series,
        embeddings: np.ndarray,
        metadata: pd.DataFrame | None = None,
    ) -> Typologist:
        """Discover ``n_facets`` facets and per-doc labels from the corpus.

        When ``metadata`` is provided, each column is both (a) erased from the
        embeddings via LEACE and (b) described to the synthesis LLM in the
        prompt so it steers off those axes. Both effects are partial by
        construction; see ``docs/design.md`` ("Erasure: scope and limits") for
        the two-lever model and when erasure does and does not do what users
        expect.
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        naming = _resolve_llm(self.naming_llm)
        schema_llm = _resolve_llm(self.schema_llm)
        labeling_llm = _resolve_llm(self.labeling_llm)

        inputs = _normalize_inputs(documents, embeddings, metadata)
        working = inputs.embeddings
        erased_metadata_descriptions: list[dict] | None = None
        if inputs.metadata is not None:
            working = _erase_metadata(working, inputs.metadata, inputs.was_normalized)
            erased_metadata_descriptions = _describe_erased_metadata(inputs.metadata)

        schema: list[dict] = []
        label_series: list[pd.Series] = []
        diagnostics: list[dict] = []

        for _ in range(self.n_facets):
            if self.use_toponymy:
                naming_result = _run_toponymy(
                    documents=inputs.documents,
                    embeddings=working,
                    topic_embedder=self.topic_embedder,
                    naming_llm=naming,
                    object_description=self.object_description,
                    corpus_description=self.corpus_description,
                    verbose=self.verbose,
                )
            else:
                naming_result = _run_homemade_naming(
                    documents=inputs.documents,
                    embeddings=working,
                    naming_llm=naming,
                    object_description=self.object_description,
                    corpus_description=self.corpus_description,
                    max_concurrency=self.max_concurrency,
                    verbose=self.verbose,
                )

            facet, synthesis_prompt = _synthesize_facet(
                cluster_hierarchy=naming_result.topic_names,
                schema_llm=schema_llm,
                labeling_llm=labeling_llm,
                object_description=self.object_description,
                corpus_description=self.corpus_description,
                prior_facet_names=[f["name"] for f in schema],
                erased_metadata_descriptions=erased_metadata_descriptions,
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
                    naming_result=naming_result,
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
