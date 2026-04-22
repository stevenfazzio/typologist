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
    """Run UMAP on the embeddings and render an interactive DataMapPlot.

    Each Typologist-discovered facet becomes one selectable colormap in the
    ``colormaps=`` dict; a dropdown in the plot lets the viewer switch point
    coloring between facets. Region text annotations (``*label_layers``) are
    deliberately left off: the correct source for those is Toponymy fit on
    the 2D coords, which is outside the scope of this example.
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

    colormaps = {col: labels_df[col].astype(str).to_numpy() for col in labels_df.columns}

    fig = datamapplot.create_interactive_plot(
        coords,
        colormaps=colormaps,
        hover_text=hover_text,
        title="Amazon reviews",
        sub_title="Point colors switchable between Typologist-discovered facets.",
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


# Sample output from a run on 2026-04-22 with seed=0. Facets 0 and 1
# (product_category and review_sentiment) are stable across seeds; Facet 2
# varies more (EVoC clustering is non-deterministic and the third facet is
# the farthest from the embedding's dominant axes, so it picks up whichever
# orthogonal structure the LLM finds most discriminating on a given run).
#
# === Discovered schema ===
#
# Facet 0: product_category (categorical)
#   The broad Amazon product category that the review pertains to.
#   - electronics_and_tech
#   - books
#   - apparel_and_footwear
#   - beauty_and_personal_care
#   - toys_and_games
#   - kitchen_and_home
#   - bags_and_accessories
#   - Other
#
# Facet 1: review_sentiment (categorical)
#   The overall sentiment and evaluative tone the reviewer expresses about
#   the product.
#   - highly_positive
#   - mixed_with_caveats
#   - disappointed_negative
#   - neutral_informational
#   - Other
#
# Facet 2: intended_user (categorical)
#   Who the reviewer indicates the product was purchased for or used by.
#   - self
#   - child
#   - spouse_or_partner
#   - parent_or_elderly_relative
#   - friend_as_gift
#   - household_shared
#   - Other
#
# Facet 0's crosstab against Amazon's own product_category shows heavy
# diagonal concentration: Typologist rediscovers the curators' categorization
# from the text alone, and sometimes refines it (Amazon's
# Clothing_Shoes_and_Jewelry splits into apparel_and_footwear + bags_and_
# accessories; Electronics spills a bit into bags_and_accessories for tech
# accessories/cases). Facets 1 and 2 add orthogonal axes that
# product_category alone doesn't capture.
