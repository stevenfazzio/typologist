from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from toponymy import Toponymy
from toponymy.clustering import EVoCClusterer
from tqdm.auto import tqdm

from typologist._prompts import render_labeling_prompt
from typologist.llm import LLM, _wrap_for_toponymy


@dataclass(frozen=True)
class _Inputs:
    """Normalized inputs to the Typologist pipeline."""

    documents: pd.Series
    embeddings: np.ndarray


def _normalize_inputs(
    documents: list[str] | pd.Series,
    embeddings: np.ndarray,
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

    return _Inputs(
        documents=docs_series,
        embeddings=embeddings.copy(),
    )


@dataclass(frozen=True)
class _NamingResult:
    """Topic-naming output from one naming-stage run (Toponymy or homemade)."""

    topic_names: list[list[str]]
    topic_name_vectors: list[np.ndarray]
    cluster_count: int
    hierarchy_depth: int


def _run_toponymy(
    documents: pd.Series,
    embeddings: np.ndarray,
    topic_embedder: Any,
    naming_llm: LLM,
    object_description: str,
    corpus_description: str,
    verbose: bool,
) -> _NamingResult:
    """Construct a Toponymy instance with EVoCClusterer and run it on embeddings.

    Toponymy's ``fit`` expects a third ``clusterable_vectors`` argument, which
    EVoCClusterer ignores (EVoC does its own dim reduction internally). We pass
    ``embeddings`` for both slots.
    """
    topo = Toponymy(
        llm_wrapper=_wrap_for_toponymy(naming_llm),
        text_embedding_model=topic_embedder,
        clusterer=EVoCClusterer(verbose=verbose),
        object_description=object_description,
        corpus_description=corpus_description,
        verbose=verbose,
    )
    topo.fit(documents.tolist(), embeddings, embeddings)
    return _NamingResult(
        topic_names=topo.topic_names_,
        topic_name_vectors=topo.topic_name_vectors_,
        cluster_count=max(len(layer) for layer in topo.topic_names_),
        hierarchy_depth=len(topo.topic_names_),
    )


_FACET_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "kind": {"type": "string", "enum": ["categorical"]},
        "values": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 2,
        },
        "definition": {"type": "string"},
    },
    "required": ["name", "kind", "values", "definition"],
}


def _classify_docs(
    facet: dict,
    documents: pd.Series,
    labeling_llm: LLM,
    noise_label: str,
    max_concurrency: int = 1,
    verbose: bool = False,
) -> pd.Series:
    """Apply a facet's labeling template to every document.

    Matches each LLM response against ``facet["values"]`` case-insensitively;
    unmatched responses become ``noise_label``. When ``max_concurrency > 1``
    the per-doc LLM calls are dispatched through a ``ThreadPoolExecutor``;
    order is preserved. Returns a Categorical Series whose index matches
    ``documents``.
    """
    template = facet["labeling_prompt_template"]
    values = list(facet["values"])
    canonical_by_lower = {v.lower(): v for v in values}

    def classify_one(doc: str) -> str:
        raw = labeling_llm(render_labeling_prompt(template, doc))
        return canonical_by_lower.get(raw.strip().lower(), noise_label)

    if max_concurrency > 1:
        with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
            results_iter = executor.map(classify_one, documents)
            if verbose:
                results_iter = tqdm(
                    results_iter,
                    total=len(documents),
                    desc=f"labeling {facet['name']}",
                    unit="doc",
                )
            labels = list(results_iter)
    else:
        iterator = (
            tqdm(documents, desc=f"labeling {facet['name']}", unit="doc") if verbose else documents
        )
        labels = [classify_one(d) for d in iterator]

    categories = list(values)
    if noise_label not in categories:
        categories.append(noise_label)

    return pd.Series(
        pd.Categorical(labels, categories=categories),
        index=documents.index,
        name=facet["name"],
    )
