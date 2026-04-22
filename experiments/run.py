"""Run one Typologist experiment.

Usage:
    uv run python experiments/run.py <corpus> <mode> <seed>

Examples:
    uv run python experiments/run.py arxiv no_erase 0
    uv run python experiments/run.py amazon erase 1

Embeddings are cached per (corpus, seed) pair under experiments/artifacts/cache/
so the two modes for a given (corpus, seed) reuse them. Delete the cache dir
to force a re-embed.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from datetime import UTC, datetime
from pathlib import Path

import cohere
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from loaders import LoadedCorpus, load_amazon, load_arxiv  # noqa: E402

ARTIFACTS_ROOT = Path(__file__).parent / "artifacts"
CACHE_ROOT = ARTIFACTS_ROOT / "cache"
RUNS_ROOT = ARTIFACTS_ROOT / "runs"

COHERE_MODEL = "embed-v4.0"
COHERE_INPUT_TYPE = "search_document"
COHERE_BATCH_SIZE = 96

TOPIC_EMBEDDER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

DEFAULT_N_PER_CLASS = {"arxiv": 200, "amazon": 167}


def _embed_with_cohere(texts: list[str]) -> np.ndarray:
    client = cohere.ClientV2(api_key=os.environ["CO_API_KEY"])
    parts: list[np.ndarray] = []
    for i in range(0, len(texts), COHERE_BATCH_SIZE):
        batch = texts[i : i + COHERE_BATCH_SIZE]
        resp = client.embed(
            texts=batch,
            model=COHERE_MODEL,
            input_type=COHERE_INPUT_TYPE,
            embedding_types=["float"],
        )
        parts.append(np.array(resp.embeddings.float_, dtype=np.float32))
    return np.vstack(parts)


def _load_or_cache_corpus(
    corpus_name: str, seed: int, n_per_class: int
) -> tuple[LoadedCorpus, np.ndarray]:
    cache_dir = CACHE_ROOT / f"{corpus_name}_seed{seed}"
    embeddings_path = cache_dir / "embeddings.npy"
    documents_path = cache_dir / "documents.parquet"
    curator_path = cache_dir / "curator_labels.parquet"
    meta_path = cache_dir / "meta.json"

    if embeddings_path.exists() and documents_path.exists():
        print(f"[cache] loading {corpus_name}_seed{seed}")
        embeddings = np.load(embeddings_path)
        docs_df = pd.read_parquet(documents_path)
        documents = docs_df["document"]
        curator_labels = pd.read_parquet(curator_path)
        with open(meta_path) as f:
            meta = json.load(f)
        corpus = LoadedCorpus(
            name=corpus_name,
            documents=documents,
            curator_labels=curator_labels,
            object_description=meta["object_description"],
            corpus_description=meta["corpus_description"],
        )
        return corpus, embeddings

    print(f"[load] {corpus_name} fresh (seed={seed}, n_per_class={n_per_class})")
    if corpus_name == "arxiv":
        corpus = load_arxiv(n_per_class=n_per_class, random_state=seed)
    elif corpus_name == "amazon":
        corpus = load_amazon(n_per_class=n_per_class, random_state=seed)
    else:
        raise ValueError(f"unknown corpus: {corpus_name}")

    print(f"[embed] {len(corpus.documents)} docs via Cohere {COHERE_MODEL}")
    embeddings = _embed_with_cohere(corpus.documents.tolist())

    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(embeddings_path, embeddings)
    corpus.documents.to_frame("document").to_parquet(documents_path)
    corpus.curator_labels.to_parquet(curator_path)
    with open(meta_path, "w") as f:
        json.dump(
            {
                "object_description": corpus.object_description,
                "corpus_description": corpus.corpus_description,
                "n_per_class": n_per_class,
            },
            f,
            indent=2,
        )
    return corpus, embeddings


def run_experiment(
    corpus_name: str,
    mode: str,
    seed: int,
    n_facets: int,
    n_per_class: int,
) -> Path:
    from sentence_transformers import SentenceTransformer

    from typologist import Typologist

    corpus, embeddings = _load_or_cache_corpus(corpus_name, seed, n_per_class)

    metadata = corpus.curator_labels if mode == "erase" else None

    print(
        f"[fit] corpus={corpus_name} mode={mode} seed={seed} "
        f"n_facets={n_facets} n_docs={len(corpus.documents)}"
    )
    topic_embedder = SentenceTransformer(TOPIC_EMBEDDER_MODEL)

    t = Typologist(
        n_facets=n_facets,
        topic_embedder=topic_embedder,
        object_description=corpus.object_description,
        corpus_description=corpus.corpus_description,
        random_state=seed,
        verbose=True,
    )
    t.fit(corpus.documents, embeddings, metadata=metadata)

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_ROOT / f"{corpus_name}_{mode}_seed{seed}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    from typologist import __version__ as typologist_version

    config = {
        "corpus": corpus_name,
        "mode": mode,
        "seed": seed,
        "n_facets": n_facets,
        "n_per_class": n_per_class,
        "n_docs": len(corpus.documents),
        "cohere_model": COHERE_MODEL,
        "topic_embedder": TOPIC_EMBEDDER_MODEL,
        "timestamp_utc": timestamp,
        "typologist_version": typologist_version,
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    with open(run_dir / "schema.json", "w") as f:
        json.dump(t.schema_, f, indent=2)

    t.labels_df_.to_parquet(run_dir / "labels_df.parquet")
    np.save(run_dir / "residualized.npy", t.embeddings_residualized_)

    with open(run_dir / "diagnostics.pkl", "wb") as f:
        pickle.dump(t.facet_diagnostics_, f)

    print(f"[done] saved to {run_dir}")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", choices=["arxiv", "amazon"])
    parser.add_argument("mode", choices=["erase", "no_erase"])
    parser.add_argument("seed", type=int)
    parser.add_argument("--n-facets", type=int, default=3)
    parser.add_argument("--n-per-class", type=int, default=None)
    args = parser.parse_args()

    n_per_class = args.n_per_class or DEFAULT_N_PER_CLASS[args.corpus]
    run_experiment(args.corpus, args.mode, args.seed, args.n_facets, n_per_class)


if __name__ == "__main__":
    main()
