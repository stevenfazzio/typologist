from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from concept_erasure import LeaceEraser
from toponymy import Toponymy
from toponymy.clustering import EVoCClusterer
from tqdm.auto import tqdm

from typologist._llm import _LLM, _wrap_for_toponymy
from typologist._prompts import (
    render_labeling_prompt,
    render_labeling_template,
    render_synthesis_prompt,
)


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


_MAX_VALUES_SHOWN_IN_PROMPT = 8


def _describe_erased_metadata(metadata: pd.DataFrame) -> list[dict]:
    """Describe each metadata column for the synthesis prompt.

    Type inference:
    - ``pd.CategoricalDtype(ordered=True)`` -> ordinal (use declared category order)
    - numeric dtypes -> ordinal (sorted unique values have a natural order)
    - everything else -> categorical

    Returns one dict per column with the fields consumed by
    ``_format_accounted_for_entry`` in ``_prompts.py``.
    """
    out: list[dict] = []
    for col in metadata.columns:
        series = metadata[col]
        if isinstance(series.dtype, pd.CategoricalDtype) and series.dtype.ordered:
            col_type = "ordinal"
            unique_all = [v for v in series.cat.categories if pd.notna(v)]
        elif pd.api.types.is_numeric_dtype(series):
            col_type = "ordinal"
            unique_all = sorted(series.dropna().unique().tolist())
        else:
            col_type = "categorical"
            unique_all = list(pd.unique(series.dropna()))

        total_unique = len(unique_all)
        truncated = total_unique > _MAX_VALUES_SHOWN_IN_PROMPT
        shown = unique_all[:_MAX_VALUES_SHOWN_IN_PROMPT]
        out.append(
            {
                "name": str(col),
                "type": col_type,
                "values_shown": [str(v) for v in shown],
                "truncated": truncated,
                "total_unique": total_unique,
            }
        )
    return out


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

    LEACE removes the entire linear subspace that predicts the labels, not
    just the centroid offset between value groups. Signals in the embedding
    that are linearly correlated with the labels (even if they aren't what
    the labels directly encode) also get removed. See ``docs/design.md``
    (``embeddings_residualized_`` section) for the user-facing implication.
    """
    one_hot = pd.get_dummies(metadata.astype(str), dtype=float).to_numpy()

    x = torch.from_numpy(embeddings.astype(np.float32))
    z = torch.from_numpy(one_hot.astype(np.float32))

    eraser = LeaceEraser.fit(x, z)
    erased = eraser(x).numpy().astype(embeddings.dtype, copy=False)

    if was_normalized:
        erased = _l2_normalize(erased)

    return erased


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
    naming_llm: _LLM,
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


def _synthesize_facet(
    cluster_hierarchy: list[list[str]],
    schema_llm: _LLM,
    labeling_llm_model_name: str | None,
    object_description: str,
    corpus_description: str,
    prior_facet_names: list[str],
    erased_metadata_descriptions: list[dict] | None = None,
) -> tuple[dict, str]:
    """Call ``schema_llm`` to propose a new facet from Toponymy cluster names.

    Returns a (facet_dict, synthesis_prompt) tuple. The synthesis prompt is
    returned so the caller can record it in ``facet_diagnostics_``. The
    facet_dict is the entry that goes into ``schema_``; it records
    ``labeling_llm_model_name`` as the ``labeling_model`` field (the
    classification-step model, not the synthesis-step one).
    """
    prompt = render_synthesis_prompt(
        cluster_hierarchy=cluster_hierarchy,
        object_description=object_description,
        corpus_description=corpus_description,
        prior_facet_names=prior_facet_names,
        erased_metadata_descriptions=erased_metadata_descriptions,
    )

    response = schema_llm.call_structured(prompt, _FACET_RESPONSE_SCHEMA)

    for key in ("name", "kind", "values", "definition"):
        if key not in response:
            raise RuntimeError(f"schema_llm response missing required field '{key}': {response!r}")

    name = response["name"]
    if name in prior_facet_names:
        raise RuntimeError(
            f"schema_llm proposed facet name '{name}' which collides with an "
            f"already-discovered facet. Prior facets: {prior_facet_names}"
        )

    values = list(response["values"])
    if len(values) < 2:
        raise RuntimeError(f"schema_llm proposed facet '{name}' with fewer than 2 values: {values}")
    if len(set(values)) != len(values):
        raise RuntimeError(f"schema_llm proposed duplicate values in facet '{name}': {values}")

    # Always include "Other" as a catch-all. The synthesis prompt asks the LLM
    # not to include one itself, but we dedup case-insensitively just in case.
    if "other" not in {v.lower() for v in values}:
        values.append("Other")

    labeling_template = render_labeling_template(
        facet_name=name,
        facet_definition=response["definition"],
        values=values,
        object_description=object_description,
    )

    facet = {
        "name": name,
        "kind": response["kind"],
        "values": values,
        "definition": response["definition"],
        "labeling_prompt_template": labeling_template,
        "labeling_model": labeling_llm_model_name,
    }
    return facet, prompt


def _classify_docs(
    facet: dict,
    documents: pd.Series,
    labeling_llm: _LLM,
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


def _build_facet_diagnostics(
    synthesis_prompt: str,
    naming_result: _NamingResult,
    labels: pd.Series,
    embeddings_pre_erasure: np.ndarray,
    values: list[str],
    exemplars_k: int = 5,
) -> dict:
    """Assemble the per-facet diagnostics dict that lands in facet_diagnostics_.

    Entropy is computed from the observed label distribution (in bits); uniform
    is the max-entropy ceiling over the value vocabulary (``log2(n_values)``);
    delta is ``observed - uniform`` (<=0). Exemplars per value are the
    ``exemplars_k`` documents nearest to that value's centroid in
    ``embeddings_pre_erasure``, identified by their index in the original
    ``labels`` Series.
    """
    counts = labels.value_counts(normalize=True)
    probs = counts.values
    probs = probs[probs > 0.0]  # exclude zero-count Categorical members
    if probs.size > 0:
        observed = -float((probs * np.log2(probs)).sum())
    else:
        observed = 0.0
    n_vals = len(values)
    uniform = float(np.log2(n_vals)) if n_vals > 1 else 0.0

    exemplars_per_value: dict[str, list] = {}
    for value in values:
        mask = (labels == value).to_numpy()
        count = int(mask.sum())
        if count == 0:
            exemplars_per_value[value] = []
            continue
        value_positions = np.where(mask)[0]
        value_embeddings = embeddings_pre_erasure[value_positions]
        centroid = value_embeddings.mean(axis=0)
        dists = np.linalg.norm(value_embeddings - centroid, axis=1)
        k = min(exemplars_k, count)
        top_within = np.argsort(dists)[:k]
        top_positions = value_positions[top_within]
        exemplars_per_value[value] = labels.index[top_positions].tolist()

    return {
        "synthesis_prompt": synthesis_prompt,
        "cluster_count": naming_result.cluster_count,
        "hierarchy_depth": naming_result.hierarchy_depth,
        "entropy_bits": {
            "observed": observed,
            "uniform": uniform,
            "delta": observed - uniform,
        },
        "exemplars_per_value": exemplars_per_value,
    }


def _residualize_facet(
    embeddings: np.ndarray,
    facet_labels: pd.Series,
    was_normalized: bool,
) -> np.ndarray:
    """Erase a facet's per-doc labels from embeddings via LEACE.

    Noise-labeled rows are treated as their own category for erasure purposes
    (the one-hot vector has a column for ``noise_label``), so the LEACE
    projection removes any variance aligned with "couldn't classify" as well.
    """
    metadata = pd.DataFrame({"facet": facet_labels.astype(str).to_numpy()})
    return _erase_metadata(embeddings, metadata, was_normalized)
