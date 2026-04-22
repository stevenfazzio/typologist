"""Discover categorical facets in a sample of Amazon product reviews.

What this does:
1. Streams a stratified sample of ~500 Amazon reviews across 6 product
   categories from HuggingFace.
2. Embeds the reviews with Cohere embed-v4.0.
3. Runs Typologist to discover 3 categorical facets.
4. Prints the discovered schema and a crosstab of facet 0 against the
   curator-assigned product category (to show the rediscovery effect).
5. Renders an interactive DataMapPlot HTML you can open in a browser,
   colored by each discovered facet (toggle between layers).

Expected runtime: ~5 minutes on a laptop.
Expected cost: ~$3 (Cohere embedding + Anthropic Haiku labeling + Opus synthesis).

Required environment variables:
    CO_API_KEY         Cohere API key, for embeddings
    ANTHROPIC_API_KEY  Anthropic API key, for the three LLM roles

Required extra installs (on top of `typologist` itself):
    uv pip install datasets sentence-transformers cohere umap-learn datamapplot

Run from the repo root:
    uv run python examples/amazon_reviews.py
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

# --- configuration ----------------------------------------------------------

CATEGORIES = [
    "Books",
    "Electronics",
    "All_Beauty",
    "Clothing_Shoes_and_Jewelry",
    "Home_and_Kitchen",
    "Toys_and_Games",
]
N_PER_CATEGORY = 83  # 6 x 83 = 498 reviews total
STREAM_WINDOW = 3000
MIN_TEXT_CHARS = 200
RANDOM_SEED = 0

COHERE_MODEL = "embed-v4.0"
COHERE_BATCH = 96

OUTPUT_DIR = Path(__file__).parent
HTML_OUTPUT = OUTPUT_DIR / "amazon_reviews_map.html"

_AMAZON_JSONL_URL = (
    "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023"
    "/resolve/main/raw/review_categories/{category}.jsonl"
)


# --- helpers ----------------------------------------------------------------


def load_reviews(seed: int) -> pd.DataFrame:
    """Stream a stratified sample of Amazon reviews from HuggingFace.

    For each category, streams the first STREAM_WINDOW rows from the public
    JSONL dump, filters to verified purchases with text >= MIN_TEXT_CHARS,
    and samples N_PER_CATEGORY rows. Returns a DataFrame with columns:
    text, rating, product_category.
    """
    from datasets import load_dataset

    rng = np.random.default_rng(seed)
    parts: list[pd.DataFrame] = []

    for category in CATEGORIES:
        print(f"  streaming {category}...", flush=True)
        url = _AMAZON_JSONL_URL.format(category=category)
        ds = load_dataset("json", data_files=url, split="train", streaming=True)

        rows: list[dict] = []
        for row in ds:
            text = row.get("text") or ""
            if (
                row.get("verified_purchase")
                and len(text) >= MIN_TEXT_CHARS
                and row.get("rating") is not None
            ):
                rows.append({"text": text, "rating": float(row["rating"])})
            if len(rows) >= STREAM_WINDOW:
                break

        picked = rng.choice(len(rows), size=N_PER_CATEGORY, replace=False)
        cat_df = pd.DataFrame([rows[i] for i in picked])
        cat_df["product_category"] = category
        parts.append(cat_df)

    return pd.concat(parts, ignore_index=True)


def embed_documents(texts: list[str]) -> np.ndarray:
    """Embed documents via Cohere embed-v4.0, batched."""
    import cohere

    client = cohere.ClientV2(api_key=os.environ["CO_API_KEY"])
    parts: list[np.ndarray] = []
    for i in range(0, len(texts), COHERE_BATCH):
        batch = texts[i : i + COHERE_BATCH]
        resp = client.embed(
            texts=batch,
            model=COHERE_MODEL,
            input_type="search_document",
            embedding_types=["float"],
        )
        parts.append(np.array(resp.embeddings.float_, dtype=np.float32))
    return np.vstack(parts)


def render_map(
    embeddings: np.ndarray,
    labels_df: pd.DataFrame,
    hover_text: list[str],
    output_path: Path,
    seed: int,
) -> None:
    """Run UMAP on the embeddings and render a multi-layer DataMapPlot.

    Each facet in ``labels_df`` becomes a separate label layer in the
    interactive plot, so the viewer can switch between colorings.
    """
    import datamapplot
    import umap

    print("  projecting to 2D with UMAP...", flush=True)
    coords = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
        random_state=seed,
    ).fit_transform(embeddings)

    # Each column of labels_df (one per discovered facet) becomes one
    # positional *label_layers argument to create_interactive_plot.
    label_layers = [labels_df[col].astype(str).to_numpy() for col in labels_df.columns]

    fig = datamapplot.create_interactive_plot(
        coords,
        *label_layers,
        hover_text=hover_text,
        title="Amazon reviews, colored by Typologist-discovered facets",
        sub_title="Toggle layers to switch between facets.",
        inline_data=True,
    )
    fig.save(str(output_path))


# --- main -------------------------------------------------------------------


def main() -> None:
    from sentence_transformers import SentenceTransformer

    from typologist import Typologist

    print("Loading Amazon reviews from HuggingFace...")
    df = load_reviews(seed=RANDOM_SEED)
    print(f"  loaded {len(df)} reviews across {df['product_category'].nunique()} categories\n")

    print("Embedding with Cohere embed-v4.0...")
    embeddings = embed_documents(df["text"].tolist())
    print(f"  embeddings shape: {embeddings.shape}\n")

    print("Fitting Typologist (n_facets=3)...")
    t = Typologist(
        n_facets=3,
        topic_embedder=SentenceTransformer("all-MiniLM-L6-v2"),
        object_description="product reviews",
        corpus_description="Amazon product reviews",
        random_state=RANDOM_SEED,
        verbose=True,
    ).fit(df["text"], embeddings)

    print("\n=== Discovered schema ===")
    for i, facet in enumerate(t.schema_):
        print(f"\nFacet {i}: {facet['name']} ({facet['type']})")
        print(f"  {facet['definition']}")
        for value in facet["values"]:
            print(f"  - {value}")

    primary_facet = t.schema_[0]["name"]
    print(f"\n=== Facet 0 ({primary_facet}) vs curator-assigned product category ===\n")
    # Use a distinct column name so a facet named "product_category" doesn't
    # collide with the curator column when we join them.
    labels = t.labels_df_.reset_index(drop=True)
    labels["curator_category"] = df["product_category"].values
    crosstab = pd.crosstab(labels["curator_category"], labels[primary_facet])
    print(crosstab.to_string())

    print("\nRendering interactive map...")
    hover = [t[:200] for t in df["text"].tolist()]
    render_map(
        embeddings=embeddings,
        labels_df=t.labels_df_.reset_index(drop=True),
        hover_text=hover,
        output_path=HTML_OUTPUT,
        seed=RANDOM_SEED,
    )
    print(f"  saved {HTML_OUTPUT.relative_to(Path.cwd())}")
    print(f"  open in a browser: file://{HTML_OUTPUT}")


if __name__ == "__main__":
    main()


# Sample output from a run on 2026-04-22 with seed=0 (exact wording varies
# seed-to-seed because EVoC clustering is non-deterministic):
#
# === Discovered schema ===
#
# Facet 0: product_category (categorical)
#   The category of product being reviewed, based on the item's primary use
#   and market segment.
#   - apparel_and_footwear
#   - books_and_cookbooks
#   - bags_and_luggage
#   - electronics_and_accessories
#   - kitchen_tools
#   - beauty_and_personal_care
#   - toys_and_plush
#   - Other
#
# Facet 1: review_sentiment (categorical)
#   The overall sentiment the reviewer expresses toward the product, ranging
#   from enthusiastic praise to strong dissatisfaction.
#   - strongly_positive
#   - mildly_positive
#   - mixed
#   - mildly_negative
#   - strongly_negative
#   - Other
#
# Facet 2: review_focus_aspect (categorical)
#   The primary product attribute the reviewer emphasizes when evaluating
#   the item.
#   - fit_and_sizing
#   - durability_and_longevity
#   - ease_of_use
#   - value_for_money
#   - aesthetic_appearance
#   - performance_effectiveness
#   - comfort_and_feel
#   - instructions_and_setup
#   - Other
#
# Facet 0's crosstab against Amazon's own product_category shows heavy
# diagonal concentration: Typologist rediscovers the curators' categorization
# from the text alone, sometimes refining it further (Amazon's
# Clothing_Shoes_and_Jewelry splits into apparel_and_footwear and
# bags_and_luggage in the discovered schema). Facets 1 and 2 add orthogonal
# axes that product_category alone doesn't capture.
