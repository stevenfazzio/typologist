"""Empirical evaluation for issue #11: is Toponymy load-bearing for facet quality?

Runs Typologist with ``use_toponymy={True, False}`` x ``erasure={False, True}``
across one or more corpora, holding ``random_state=0`` everywhere. Per arm,
captures the discovered schema, a crosstab of facet 0 against the curator-
assigned metadata column, per-facet diagnostics, and per-value exemplar
documents (so the JSON artifact is auditable without re-running).

Decision criteria (from issue #11):
  - Wash or homemade-wins -> drop Toponymy in v0.2.
  - Toponymy wins clearly -> keep, document the empirical justification.
  - Mixed -> probably keep, document what would unblock a revisit.

Phase 1 (this script): vary corpus domain at n~500 with seed=0.
Phase 2 (planned): vary corpus size on whichever corpus shows the biggest delta.
Phase 3 (planned): vary document length.

Required env: ``ANTHROPIC_API_KEY``.
Required extra installs: ``uv pip install datasets sentence-transformers``.

Run from the repo root with one or more corpus names (default: all):
    uv run python experiments/toponymy_vs_homemade.py amazon arxiv ag_news

Per-corpus expected runtime: ~10-15 min. Per-corpus expected cost: ~$3-12
(depends on n_docs and per-doc length).
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "examples"))

OUT_DIR = REPO_ROOT / "experiments" / "artifacts" / "runs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RANDOM_SEED = 0
N_DOCS_TARGET = 500
N_EXEMPLARS_PER_VALUE = 3


@dataclass
class _Corpus:
    """Per-corpus inputs threaded into ``_run_arm``."""

    name: str
    documents: pd.Series
    embeddings: np.ndarray
    metadata_for_erasure: pd.DataFrame
    curator_category: np.ndarray
    object_description: str
    corpus_description: str
    metadata_column_name: str


@dataclass
class _ArmResult:
    label: str
    use_toponymy: bool
    erasure: bool
    schema: list[dict]
    diagnostics_summary: list[dict]
    crosstab_facet0_vs_category: dict
    exemplars: list[dict]
    elapsed_seconds: float


# --- corpus loaders ---------------------------------------------------------


def _load_amazon_n(n_per_category: int, name: str) -> _Corpus:
    """Stratified sample of Amazon reviews; size set via ``n_per_category`` x 6 categories."""
    from amazon_reviews import embed_documents, load_reviews

    df = load_reviews(seed=RANDOM_SEED, n_per_category=n_per_category).reset_index(drop=True)
    print(f"  {name}: {len(df)} reviews, {df['product_category'].nunique()} categories")
    print(f"  embedding {name} with sentence-transformers all-MiniLM-L6-v2...")
    embeddings = embed_documents(df["text"].tolist())
    return _Corpus(
        name=name,
        documents=df["text"],
        embeddings=embeddings,
        metadata_for_erasure=df[["product_category"]],
        curator_category=df["product_category"].to_numpy(),
        object_description="product reviews",
        corpus_description="Amazon product reviews",
        metadata_column_name="product_category",
    )


def _load_amazon() -> _Corpus:
    """Stratified sample of ~500 Amazon reviews (the Phase 1 baseline size)."""
    return _load_amazon_n(83, "amazon_reviews")


def _load_amazon_1500() -> _Corpus:
    """Stratified sample of ~1500 Amazon reviews (Phase 2 size-variation arm)."""
    return _load_amazon_n(250, "amazon_reviews_1500")


def _load_arxiv() -> _Corpus:
    """~500 arxiv abstracts from the top-6 primary_categories of the predecessor parquet.

    Restricted to the top-6 categories (cs.CV, cs.LG, cs.CL, stat.ML, cs.RO,
    cs.AI) for two reasons: it matches Amazon's 6-category structure for a
    cleaner cross-corpus comparison, and it keeps LEACE's one-hot rank low
    enough that the residualized embedding doesn't collapse below EVoC's
    minimum-cluster-detection threshold (the n=48-cat version crashed with
    "argmax of an empty sequence" inside EVoC during the erasure pass).

    Re-embeds with MiniLM rather than reusing the parquet's Cohere embeddings,
    so the corpus comparison holds the embedder constant across all corpora.
    """
    parquet = Path("~/repos/_archived/strata/data/arxiv-2026-03.parquet").expanduser()
    if not parquet.exists():
        raise FileNotFoundError(
            f"arxiv parquet not found at {parquet}. The arxiv corpus assumes "
            "the predecessor harness data is available locally."
        )
    df_full = pd.read_parquet(parquet)
    df_full["text"] = df_full["title"].str.strip() + "\n\n" + df_full["abstract"].str.strip()

    top_categories = df_full["primary_category"].value_counts().head(6).index.tolist()
    df_top = df_full[df_full["primary_category"].isin(top_categories)].reset_index(drop=True)

    rng = np.random.default_rng(RANDOM_SEED)
    per_cat = N_DOCS_TARGET // len(top_categories)
    parts: list[pd.DataFrame] = []
    for cat in top_categories:
        group = df_top[df_top["primary_category"] == cat]
        if len(group) <= per_cat:
            parts.append(group)
        else:
            picked = rng.choice(len(group), size=per_cat, replace=False)
            parts.append(group.iloc[picked])
    df = pd.concat(parts, ignore_index=True)

    print(f"  arxiv: {len(df)} abstracts, {df['primary_category'].nunique()} categories")
    print("  embedding arxiv with sentence-transformers all-MiniLM-L6-v2...")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(
        df["text"].tolist(),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)

    return _Corpus(
        name="arxiv_abstracts",
        documents=df["text"],
        embeddings=embeddings,
        metadata_for_erasure=df[["primary_category"]],
        curator_category=df["primary_category"].to_numpy(),
        object_description="scientific paper",
        corpus_description="machine-learning arxiv papers",
        metadata_column_name="primary_category",
    )


def _load_ag_news() -> _Corpus:
    """500 ag_news items stratified across 4 topic labels (World/Sports/Business/SciTech)."""
    from datasets import load_dataset

    ds = load_dataset("ag_news", split="train", streaming=True)
    label_names = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
    per_label = N_DOCS_TARGET // len(label_names)
    rng = np.random.default_rng(RANDOM_SEED)

    pool: dict[int, list[dict]] = {k: [] for k in label_names}
    stream_window = per_label * 8
    for row in ds:
        lbl = int(row["label"])
        if len(pool[lbl]) < stream_window:
            pool[lbl].append({"text": row["text"], "label": label_names[lbl]})
        if all(len(v) >= stream_window for v in pool.values()):
            break

    parts: list[pd.DataFrame] = []
    for lbl, rows in pool.items():
        picked = rng.choice(len(rows), size=min(per_label, len(rows)), replace=False)
        parts.append(pd.DataFrame([rows[i] for i in picked]))
    df = pd.concat(parts, ignore_index=True)

    print(f"  ag_news: {len(df)} items, {df['label'].nunique()} topic labels")
    print("  embedding ag_news with sentence-transformers all-MiniLM-L6-v2...")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(
        df["text"].tolist(),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)

    return _Corpus(
        name="ag_news",
        documents=df["text"],
        embeddings=embeddings,
        metadata_for_erasure=df[["label"]],
        curator_category=df["label"].to_numpy(),
        object_description="news article",
        corpus_description="news headline and snippet excerpts",
        metadata_column_name="label",
    )


CORPUS_LOADERS: dict[str, callable] = {
    "amazon": _load_amazon,
    "amazon_1500": _load_amazon_1500,
    "arxiv": _load_arxiv,
    "ag_news": _load_ag_news,
}


# --- per-arm execution ------------------------------------------------------


def _summarize_diagnostics(diagnostics: list[dict]) -> list[dict]:
    """Strip the heavy fields from facet_diagnostics_ for the JSON artifact."""
    return [
        {
            "cluster_count": d["cluster_count"],
            "hierarchy_depth": d["hierarchy_depth"],
            "entropy_bits": d["entropy_bits"],
        }
        for d in diagnostics
    ]


def _crosstab_to_dict(crosstab: pd.DataFrame) -> dict:
    return {
        "rows": list(crosstab.index.astype(str)),
        "columns": list(crosstab.columns.astype(str)),
        "values": crosstab.values.tolist(),
    }


def _capture_exemplars(
    diagnostics: list[dict],
    schema: list[dict],
    documents: pd.Series,
    k: int,
    max_chars: int = 240,
) -> list[dict]:
    """Per facet, per value, return up to k exemplar texts (truncated for JSON-readability).

    The diagnostics already carry per-value exemplar document indices via
    ``exemplars_per_value``; here we look up the actual text so the JSON is
    auditable without rerunning.
    """
    out: list[dict] = []
    for facet, diag in zip(schema, diagnostics):
        facet_entry: dict = {"facet_name": facet["name"], "values": {}}
        for value, doc_indices in diag["exemplars_per_value"].items():
            texts: list[str] = []
            for idx in doc_indices[:k]:
                raw = documents.loc[idx]
                texts.append(raw if len(raw) <= max_chars else raw[:max_chars] + "...")
            facet_entry["values"][value] = texts
        out.append(facet_entry)
    return out


def _run_arm(
    label: str,
    use_toponymy: bool,
    erasure: bool,
    topic_embedder,
    corpus: _Corpus,
) -> _ArmResult:
    from typologist import Typologist

    print(f"\n{'=' * 70}\nArm: {corpus.name}/{label}\n{'=' * 70}")
    t0 = time.time()
    t = Typologist(
        n_facets=3,
        topic_embedder=topic_embedder,
        object_description=corpus.object_description,
        corpus_description=corpus.corpus_description,
        random_state=RANDOM_SEED,
        verbose=True,
        use_toponymy=use_toponymy,
    ).fit(
        corpus.documents,
        corpus.embeddings,
        metadata=corpus.metadata_for_erasure if erasure else None,
    )
    elapsed = time.time() - t0

    print(f"\n--- Schema ({corpus.name}/{label}) ---")
    for i, facet in enumerate(t.schema_):
        print(f"\nFacet {i}: {facet['name']} ({facet['kind']})")
        print(f"  {facet['definition']}")
        for value in facet["values"]:
            print(f"  - {value}")

    primary_facet = t.schema_[0]["name"]
    labels = t.labels_df_.reset_index(drop=True)
    labels["curator_category"] = corpus.curator_category
    crosstab = pd.crosstab(labels["curator_category"], labels[primary_facet])
    print(f"\n--- Facet 0 ({primary_facet}) vs curator {corpus.metadata_column_name} ---\n")
    print(crosstab.to_string())

    print(f"\n--- per-facet diagnostics ({corpus.name}/{label}) ---")
    for i, d in enumerate(t.facet_diagnostics_):
        e = d["entropy_bits"]
        print(
            f"  Facet {i}: clusters={d['cluster_count']} "
            f"depth={d['hierarchy_depth']} "
            f"entropy obs={e['observed']:.2f} unif={e['uniform']:.2f} "
            f"delta={e['delta']:+.2f} bits"
        )
    print(f"\n  elapsed: {elapsed:.1f}s")

    return _ArmResult(
        label=label,
        use_toponymy=use_toponymy,
        erasure=erasure,
        schema=t.schema_,
        diagnostics_summary=_summarize_diagnostics(t.facet_diagnostics_),
        crosstab_facet0_vs_category=_crosstab_to_dict(crosstab),
        exemplars=_capture_exemplars(
            t.facet_diagnostics_, t.schema_, corpus.documents, k=N_EXEMPLARS_PER_VALUE
        ),
        elapsed_seconds=elapsed,
    )


def _run_corpus(corpus: _Corpus, topic_embedder, out_path: Path) -> dict:
    """Run all four arms for one corpus, writing the JSON artifact incrementally.

    Each arm runs in a try/except so one arm's failure (e.g. EVoC failing on
    LEACE-residualized embeddings) doesn't sink the whole corpus. After every
    arm we rewrite the artifact, so a later crash still leaves the completed
    arms' results on disk.
    """
    arms_spec: list[tuple[str, bool, bool]] = [
        ("toponymy_vanilla", True, False),
        ("toponymy_erasure", True, True),
        ("homemade_vanilla", False, False),
        ("homemade_erasure", False, True),
    ]
    arms_data: list[dict] = []
    artifact = {
        "corpus": corpus.name,
        "n_docs": int(len(corpus.documents)),
        "embedding_dim": int(corpus.embeddings.shape[1]),
        "metadata_column": corpus.metadata_column_name,
        "random_state": RANDOM_SEED,
        "n_facets": 3,
        "arms": arms_data,
    }
    for label, use_toponymy, erasure in arms_spec:
        try:
            result = _run_arm(label, use_toponymy, erasure, topic_embedder, corpus)
            arms_data.append(asdict(result))
        except Exception as exc:
            print(f"\n!! arm {corpus.name}/{label} FAILED: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            arms_data.append(
                {
                    "label": label,
                    "use_toponymy": use_toponymy,
                    "erasure": erasure,
                    "failed": True,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
        with out_path.open("w") as f:
            json.dump(artifact, f, indent=2, default=str)
    return artifact


def main() -> None:
    global RANDOM_SEED
    from sentence_transformers import SentenceTransformer

    args = list(sys.argv[1:])
    if "--seed" in args:
        i = args.index("--seed")
        RANDOM_SEED = int(args[i + 1])
        del args[i : i + 2]

    requested = args or list(CORPUS_LOADERS.keys())
    unknown = [c for c in requested if c not in CORPUS_LOADERS]
    if unknown:
        raise SystemExit(f"Unknown corpus name(s): {unknown}. Known: {list(CORPUS_LOADERS.keys())}")

    topic_embedder = SentenceTransformer("all-MiniLM-L6-v2")
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    summary_lines: list[str] = []
    for corpus_name in requested:
        print(f"\n{'#' * 70}\n# Loading corpus: {corpus_name} (seed={RANDOM_SEED})\n{'#' * 70}")
        corpus = CORPUS_LOADERS[corpus_name]()
        out_path = (
            OUT_DIR / f"toponymy_vs_homemade_{corpus_name}_seed{RANDOM_SEED}_{timestamp}.json"
        )
        artifact = _run_corpus(corpus, topic_embedder, out_path)
        print(f"\nWrote artifact: {out_path.relative_to(REPO_ROOT)}")

        for arm in artifact["arms"]:
            if arm.get("failed"):
                summary_lines.append(
                    f"  {corpus_name:18s} {arm['label']:22s} "
                    f"FAILED: {arm['error_type']}: {arm['error_message'][:60]}"
                )
            else:
                names = [f["name"] for f in arm["schema"]]
                summary_lines.append(
                    f"  {corpus_name:18s} {arm['label']:22s} "
                    f"({arm['elapsed_seconds']:5.1f}s)  facets: {names}"
                )

    print("\n=== Summary ===")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
