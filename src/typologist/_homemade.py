"""Toponymy-replacement naming path for evaluating whether Toponymy is load-bearing.

This module exists to support the experiment in issue #11: does a minimal,
single-K, exemplar-based cluster-naming pipeline match Toponymy's facet quality?
Routed via ``Typologist(use_toponymy=False)``. Not part of the public API.

What it does:
- EVoC clustering at a single layer (no multi-scale hierarchy)
- For each non-noise cluster, pick the ``exemplars_k`` documents nearest the
  cluster's centroid in the embedding space
- One LLM call per cluster to produce a short topic name from those exemplars

Returns a ``_NamingResult`` with ``topic_names`` of length 1 (single layer),
which the schema-synthesis prompt consumes the same way it consumes Toponymy's
multi-layer hierarchy.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from evoc import EVoC
from tqdm.auto import tqdm

from typologist._llm import _LLM
from typologist._pipeline import _NamingResult

_NAMING_PROMPT = """\
Below are {n_exemplars} {object_description} from one cluster within {corpus_description}.

{exemplars_block}

Propose a short topic name (3 to 7 words) describing what these {object_description} \
have in common. Respond with the topic name only, no explanation, no quotes."""


def _format_exemplars(exemplars: list[str], max_chars_per_exemplar: int = 600) -> str:
    """Render the exemplars block for the naming prompt.

    Long documents get truncated so the prompt stays bounded; the trade-off is
    accepted because cluster-naming is a gist task and the cluster geometry has
    already done most of the work of pulling related documents together.
    """
    lines: list[str] = []
    for i, doc in enumerate(exemplars, start=1):
        text = doc if len(doc) <= max_chars_per_exemplar else doc[:max_chars_per_exemplar] + "..."
        lines.append(f"--- Exemplar {i} ---\n{text}")
    return "\n\n".join(lines)


def _select_exemplars(
    documents: pd.Series,
    embeddings: np.ndarray,
    cluster_positions: np.ndarray,
    exemplars_k: int,
) -> list[str]:
    """Return up to ``exemplars_k`` documents nearest to the cluster centroid.

    Operates in the cluster's own subset of ``embeddings`` and returns the
    documents themselves (text), in centroid-distance order.
    """
    cluster_embeddings = embeddings[cluster_positions]
    centroid = cluster_embeddings.mean(axis=0)
    dists = np.linalg.norm(cluster_embeddings - centroid, axis=1)
    k = min(exemplars_k, cluster_positions.size)
    nearest_within = np.argsort(dists)[:k]
    nearest_positions = cluster_positions[nearest_within]
    return [documents.iloc[p] for p in nearest_positions]


def _run_homemade_naming(
    documents: pd.Series,
    embeddings: np.ndarray,
    naming_llm: _LLM,
    object_description: str,
    corpus_description: str,
    exemplars_k: int = 8,
    max_concurrency: int = 10,
    verbose: bool = False,
) -> _NamingResult:
    """Cluster with EVoC, name each cluster from its centroid-nearest exemplars.

    Uses EVoC's finest (most-clusters) layer rather than ``labels_`` (EVoC's
    auto-pick of the layer with the fewest noise points): the auto-pick
    sometimes returns 4 coarse clusters when the underlying data has 11+
    natural ones, and the schema-synthesis LLM benefits from finer-grained
    cluster names. Toponymy similarly feeds all layers (finest included) into
    its synthesis prompt; this is the single-layer analog. Returns a
    single-layer ``_NamingResult`` whose ``topic_names[0]`` lists the named
    clusters.
    """
    evoc = EVoC()
    evoc.fit(embeddings)
    layers = list(evoc.cluster_layers_) or [np.asarray(evoc.labels_)]
    finest_layer = max(
        layers, key=lambda layer: len(set(int(lbl) for lbl in np.unique(layer) if lbl != -1))
    )
    raw_labels = np.asarray(finest_layer)

    cluster_ids = sorted(int(lbl) for lbl in np.unique(raw_labels) if lbl != -1)
    if not cluster_ids:
        raise RuntimeError(
            "EVoC produced no clusters (all points were noise). The homemade "
            "naming path needs at least one non-noise cluster to propose a facet."
        )

    exemplars_per_cluster: list[list[str]] = []
    for cid in cluster_ids:
        positions = np.where(raw_labels == cid)[0]
        exemplars_per_cluster.append(
            _select_exemplars(documents, embeddings, positions, exemplars_k)
        )

    def name_one(exemplars: list[str]) -> str:
        prompt = _NAMING_PROMPT.format(
            n_exemplars=len(exemplars),
            object_description=object_description,
            corpus_description=corpus_description,
            exemplars_block=_format_exemplars(exemplars),
        )
        return naming_llm(prompt).strip()

    if max_concurrency > 1:
        with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
            results_iter = executor.map(name_one, exemplars_per_cluster)
            if verbose:
                results_iter = tqdm(
                    results_iter,
                    total=len(exemplars_per_cluster),
                    desc="naming clusters",
                    unit="cluster",
                )
            names = list(results_iter)
    else:
        iterator = (
            tqdm(exemplars_per_cluster, desc="naming clusters", unit="cluster")
            if verbose
            else exemplars_per_cluster
        )
        names = [name_one(ex) for ex in iterator]

    return _NamingResult(
        topic_names=[names],
        topic_name_vectors=[np.array(names, dtype=object)],
        cluster_count=len(names),
        hierarchy_depth=1,
    )
