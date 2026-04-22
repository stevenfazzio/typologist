"""End-to-end smoke test against the predecessor's 1000-doc arxiv corpus.

Runs the full Typologist pipeline with real Anthropic API calls so we can
sanity-check the wiring and see what schema falls out. Prints schema_,
per-facet label distributions, and key diagnostics.

Requirements:
    - Predecessor data at ~/repos/_archived/strata/data/
    - ANTHROPIC_API_KEY in the environment
    - sentence-transformers installed (dev dep)

Expected runtime: 30-60 min (3000 serial Haiku labeling calls + Toponymy
naming + Opus schema synthesis). Expected Anthropic spend: ~$1.50-2.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from typologist import Typologist

DATA_DIR = Path("~/repos/_archived/strata/data").expanduser()


def load_corpus() -> tuple[pd.Series, np.ndarray, pd.DataFrame]:
    docs_df = pd.read_parquet(DATA_DIR / "arxiv-2026-03.parquet")
    emb_df = pd.read_parquet(DATA_DIR / "embeddings-2026-03.parquet")

    merged = docs_df.merge(emb_df, on="arxiv_id", how="inner").set_index("arxiv_id")

    documents = (merged["title"] + "\n\n" + merged["abstract"]).rename("document")
    embeddings = np.stack(merged["embedding"].to_numpy()).astype(np.float32)
    metadata = merged[["primary_category"]]

    return documents, embeddings, metadata


def _section(title: str) -> None:
    bar = "=" * 70
    print(f"\n{bar}\n{title}\n{bar}")


def print_schema(t: Typologist) -> None:
    _section("DISCOVERED SCHEMA")
    for i, facet in enumerate(t.schema_):
        print(f"\n--- Facet {i}: {facet['name']} ({facet['type']}) ---")
        print(f"Definition:    {facet['definition']}")
        print(f"Values:        {facet['values']}")
        print(f"Labeling model: {facet['labeling_model']}")


def print_distributions(t: Typologist) -> None:
    _section("LABEL DISTRIBUTIONS")
    for col in t.labels_df_.columns:
        print(f"\n{col}:")
        print(t.labels_df_[col].value_counts().to_string())


def print_diagnostics(t: Typologist, documents: pd.Series) -> None:
    _section("DIAGNOSTICS")
    for i, diag in enumerate(t.facet_diagnostics_):
        facet = t.schema_[i]
        print(f"\n--- Facet {i}: {facet['name']} ---")
        print(f"Toponymy clusters: {diag['cluster_count']} across {diag['hierarchy_depth']} layers")
        e = diag["entropy_bits"]
        print(
            f"Entropy (bits):  observed={e['observed']:.3f}  "
            f"uniform={e['uniform']:.3f}  delta={e['delta']:+.3f}"
        )
        print("Top-3 exemplars per value (arxiv_id: title preview):")
        for value, arxiv_ids in diag["exemplars_per_value"].items():
            if not arxiv_ids:
                print(f"  {value}: (no docs)")
                continue
            print(f"  {value}:")
            for aid in arxiv_ids[:3]:
                doc = documents.loc[aid]
                title_line = doc.split("\n\n", 1)[0]
                preview = title_line[:110].strip()
                print(f"    {aid}  {preview}")


def main() -> None:
    assert os.environ.get("ANTHROPIC_API_KEY"), "ANTHROPIC_API_KEY not set"

    print("Loading corpus from predecessor archive...")
    documents, embeddings, metadata = load_corpus()
    print(
        f"  {len(documents)} documents, "
        f"embedding dim {embeddings.shape[1]}, "
        f"metadata columns {list(metadata.columns)}"
    )

    print("\nInitializing MiniLM topic embedder...")
    topic_embedder = SentenceTransformer("all-MiniLM-L6-v2")

    print("\nRunning Typologist.fit ...")
    print("  Expect 30-60 min; 3000 labeling calls + naming + synthesis are serial.")
    t0 = time.time()
    t = Typologist(
        n_facets=3,
        topic_embedder=topic_embedder,
        object_description="scientific paper",
        corpus_description="machine-learning arxiv papers",
        random_state=0,
        verbose=True,
    ).fit(documents, embeddings, metadata=metadata)
    elapsed = time.time() - t0
    print(f"\nfit() completed in {elapsed:.1f}s ({elapsed / 60:.1f} min)")

    print_schema(t)
    print_distributions(t)
    print_diagnostics(t, documents)


if __name__ == "__main__":
    main()
