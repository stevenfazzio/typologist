from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from typologist._homemade import _run_homemade_naming
from typologist._pipeline import (
    _FACET_RESPONSE_SCHEMA,
    _normalize_inputs,
    _run_toponymy,
)
from typologist._prompts import render_labeling_template, render_synthesis_prompt
from typologist.llm import LLM, _resolve_llm


class Typologist:
    """Discover a multi-facet categorical schema from a corpus.

    ``fit(documents, embeddings)`` runs one Toponymy pass over the corpus and
    asks ``schema_llm`` to propose ``n_facets`` mutually orthogonal categorical
    axes in a single call. The fitted Typologist exposes ``schema_`` and
    ``diagnostics_``; per-document labels are obtained separately via
    ``apply_schema(t.schema_, documents, llm=...)``.

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
        use_toponymy: bool = True,
    ) -> None:
        self.n_facets = n_facets
        self.topic_embedder = topic_embedder
        self.object_description = object_description
        self.corpus_description = corpus_description
        self.naming_llm = naming_llm
        self.schema_llm = schema_llm
        # labeling_llm is provenance only: stored on each facet as
        # ``"provider:model"`` so apply_schema users see which model the schema
        # was designed for. fit() itself never calls labeling_llm.
        self.labeling_llm = labeling_llm
        self.random_state = random_state
        self.noise_label = noise_label
        self.verbose = verbose
        self.use_toponymy = use_toponymy

    def fit(
        self,
        documents: list[str] | pd.Series,
        embeddings: np.ndarray,
    ) -> Typologist:
        """Discover ``n_facets`` mutually orthogonal categorical facets.

        One Toponymy run produces a cluster hierarchy; one ``schema_llm`` call
        proposes all facets at once with the instruction that they be
        mutually orthogonal. No per-document labeling happens here; call
        ``apply_schema(t.schema_, documents, llm=...)`` to label.
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        naming = _resolve_llm(self.naming_llm)
        schema_llm = _resolve_llm(self.schema_llm)
        labeling_llm = _resolve_llm(self.labeling_llm)

        inputs = _normalize_inputs(documents, embeddings)

        if self.use_toponymy:
            naming_result = _run_toponymy(
                documents=inputs.documents,
                embeddings=inputs.embeddings,
                topic_embedder=self.topic_embedder,
                naming_llm=naming,
                object_description=self.object_description,
                corpus_description=self.corpus_description,
                verbose=self.verbose,
            )
        else:
            naming_result = _run_homemade_naming(
                documents=inputs.documents,
                embeddings=inputs.embeddings,
                naming_llm=naming,
                object_description=self.object_description,
                corpus_description=self.corpus_description,
                verbose=self.verbose,
            )

        prompt = render_synthesis_prompt(
            cluster_hierarchy=naming_result.topic_names,
            object_description=self.object_description,
            corpus_description=self.corpus_description,
            n_facets=self.n_facets,
        )

        response_schema = {
            "type": "object",
            "properties": {
                "facets": {
                    "type": "array",
                    "items": _FACET_RESPONSE_SCHEMA,
                    "minItems": self.n_facets,
                    "maxItems": self.n_facets,
                }
            },
            "required": ["facets"],
        }

        response = schema_llm.call_structured(prompt, response_schema)
        facets_raw = response.get("facets", [])
        if len(facets_raw) != self.n_facets:
            raise RuntimeError(
                f"schema_llm returned {len(facets_raw)} facets; expected {self.n_facets}"
            )

        if labeling_llm.provider is not None and labeling_llm.model_name is not None:
            labeling_model_id: str | None = f"{labeling_llm.provider}:{labeling_llm.model_name}"
        else:
            labeling_model_id = None

        schema: list[dict] = []
        seen_names: set[str] = set()
        for raw in facets_raw:
            for key in ("name", "kind", "values", "definition"):
                if key not in raw:
                    raise RuntimeError(f"facet missing required field '{key}': {raw!r}")
            name = raw["name"]
            if name in seen_names:
                raise RuntimeError(f"duplicate facet name '{name}' in schema_llm response")
            seen_names.add(name)

            values = list(raw["values"])
            if len(values) < 2:
                raise RuntimeError(f"facet '{name}' has fewer than 2 values: {values}")
            if len(set(values)) != len(values):
                raise RuntimeError(f"facet '{name}' has duplicate values: {values}")
            if "other" not in {v.lower() for v in values}:
                values.append("Other")

            labeling_template = render_labeling_template(
                facet_name=name,
                facet_definition=raw["definition"],
                values=values,
                object_description=self.object_description,
            )
            schema.append(
                {
                    "name": name,
                    "kind": raw["kind"],
                    "values": values,
                    "definition": raw["definition"],
                    "labeling_prompt_template": labeling_template,
                    "labeling_model": labeling_model_id,
                }
            )

        self.schema_ = schema
        self.diagnostics_ = {
            "synthesis_prompt": prompt,
            "cluster_count": naming_result.cluster_count,
            "hierarchy_depth": naming_result.hierarchy_depth,
        }
        return self
