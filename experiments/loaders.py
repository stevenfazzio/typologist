"""Dataset loaders for the initial-corpora experiment.

Each loader returns a LoadedCorpus with:
- documents: Series of document text
- curator_labels: DataFrame of the curator-assigned metadata (primary_category,
  rating, etc.) that we will either erase via LEACE or leave unerased and
  compare against the discovered schema.
- object_description / corpus_description: passed to Typologist.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from datasets import load_dataset

ARXIV_TOP_LEVELS = ["cs", "math", "physics", "stat", "q-bio"]

AMAZON_CATEGORIES = [
    "Books",
    "Electronics",
    "All_Beauty",
    "Clothing_Shoes_and_Jewelry",
    "Home_and_Kitchen",
    "Toys_and_Games",
]


@dataclass
class LoadedCorpus:
    name: str
    documents: pd.Series
    curator_labels: pd.DataFrame
    object_description: str
    corpus_description: str


def _extract_arxiv_top_level(categories_field) -> str:
    """Pull the top-level category from the messy `categories` field.

    The field arrives as a sequence (list or numpy array) of strings where each
    string may itself be space-separated multi-categories
    (e.g. ["math.CO cs.CG"]). We take the first token of the first entry and
    strip any sub-category suffix.
    """
    if categories_field is None or len(categories_field) == 0:
        return ""
    first = categories_field[0]
    if not isinstance(first, str) or not first:
        return ""
    primary = first.split()[0]
    return primary.split(".")[0]


def load_arxiv(
    n_per_class: int = 200,
    top_levels: list[str] | None = None,
    random_state: int = 0,
    min_abstract_chars: int = 200,
) -> LoadedCorpus:
    top_levels = top_levels or ARXIV_TOP_LEVELS
    ds = load_dataset("gfissore/arxiv-abstracts-2021", split="train")
    keep = [c for c in ["id", "abstract", "categories"] if c in ds.column_names]
    ds = ds.select_columns(keep)
    df = ds.to_pandas()

    df["raw_primary"] = df["categories"].apply(_extract_arxiv_top_level)
    df["primary_category"] = df["raw_primary"]
    df = df[df["primary_category"].isin(top_levels)]
    df = df[df["abstract"].str.len() >= min_abstract_chars]

    parts = []
    for cat in top_levels:
        cat_df = df[df["primary_category"] == cat]
        if len(cat_df) < n_per_class:
            raise ValueError(
                f"arxiv: only {len(cat_df)} docs available for {cat}, need {n_per_class}"
            )
        parts.append(cat_df.sample(n_per_class, random_state=random_state))

    out = pd.concat(parts).reset_index(drop=True)
    documents = out["abstract"].rename("document")
    documents.index = pd.Index(out["id"], name="arxiv_id")

    curator_labels = pd.DataFrame(
        {"primary_category": out["primary_category"].to_numpy()},
        index=documents.index,
    )

    return LoadedCorpus(
        name="arxiv",
        documents=documents,
        curator_labels=curator_labels,
        object_description="research paper abstracts",
        corpus_description="collection of arXiv scientific paper abstracts",
    )


_AMAZON_JSONL_URL = (
    "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023"
    "/resolve/main/raw/review_categories/{category}.jsonl"
)


def load_amazon(
    n_per_class: int = 167,
    categories: list[str] | None = None,
    random_state: int = 0,
    stream_window: int = 3000,
    min_text_chars: int = 200,
) -> LoadedCorpus:
    categories = categories or AMAZON_CATEGORIES
    rng = np.random.default_rng(random_state)

    parts = []
    for cat in categories:
        url = _AMAZON_JSONL_URL.format(category=cat)
        ds = load_dataset("json", data_files=url, split="train", streaming=True)
        rows = []
        for row in ds:
            text = row.get("text") or ""
            if (
                row.get("verified_purchase")
                and len(text) >= min_text_chars
                and row.get("rating") is not None
            ):
                rows.append(
                    {
                        "text": text,
                        "title": row.get("title") or "",
                        "rating": float(row["rating"]),
                        "asin": row.get("asin") or "",
                    }
                )
            if len(rows) >= stream_window:
                break
        if len(rows) < n_per_class:
            raise ValueError(f"amazon: only {len(rows)} usable rows for {cat}, need {n_per_class}")
        picked_idx = rng.choice(len(rows), size=n_per_class, replace=False)
        cat_df = pd.DataFrame([rows[i] for i in picked_idx])
        cat_df["product_category"] = cat
        parts.append(cat_df)

    out = pd.concat(parts, ignore_index=True)
    documents = out["text"].rename("document")
    documents.index = pd.RangeIndex(len(documents), name="row")

    curator_labels = pd.DataFrame(
        {
            "product_category": out["product_category"].to_numpy(),
            "rating": out["rating"].to_numpy(),
        },
        index=documents.index,
    )

    return LoadedCorpus(
        name="amazon",
        documents=documents,
        curator_labels=curator_labels,
        object_description="product reviews",
        corpus_description="collection of Amazon product reviews",
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Smoke-test loaders.")
    parser.add_argument("corpus", choices=["arxiv", "amazon"])
    parser.add_argument("--n-per-class", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.corpus == "arxiv":
        corpus = load_arxiv(n_per_class=args.n_per_class, random_state=args.seed)
    else:
        corpus = load_amazon(n_per_class=args.n_per_class, random_state=args.seed)

    print(f"=== {corpus.name} ===")
    print(f"n_docs: {len(corpus.documents)}")
    print(f"curator_labels columns: {list(corpus.curator_labels.columns)}")
    print(f"curator_labels head:\n{corpus.curator_labels.head()}")
    print(f"first doc preview:\n{corpus.documents.iloc[0][:300]}...")
