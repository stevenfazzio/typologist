"""Discover categorical facets in a sample of Amazon product reviews.

**Step 3 is where Typologist actually runs.** Steps 1, 2, 4, and 5 are
plumbing around this particular example (sampling Amazon reviews, embedding
them, printing and visualizing the output). If you're skimming to see what
Typologist does, start with Step 3's ~10-line block.

What this does:
1. Streams a stratified sample of ~500 Amazon reviews across 6 product
   categories from HuggingFace.
2. Embeds the reviews locally with sentence-transformers all-MiniLM-L6-v2.
   Swappable, see the ``embed_documents`` docstring.
3. **Runs Typologist to discover 3 categorical facets.**
4. Prints the discovered schema and a crosstab of facet 0 against the
   curator-assigned product category (to show the rediscovery effect).
5. Renders an interactive DataMapPlot HTML: point colors switch between
   Typologist facets via a dropdown; hovering on a point shows the
   assigned facet values followed by the review text itself.

Expected runtime: ~5 minutes on a laptop.
Expected cost: ~$3 (Anthropic LLM calls for schema synthesis and per-doc
labeling; embedding runs locally and is free).

Required environment variables:
    ANTHROPIC_API_KEY  Anthropic API key (Typologist's default LLM provider)

Required extra installs (on top of `typologist` itself):
    uv pip install datasets sentence-transformers umap-learn datamapplot

Run from the repo root:
    uv run python examples/amazon_reviews.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUTPUT_DIR = Path(__file__).parent
HTML_OUTPUT = OUTPUT_DIR / "amazon_reviews_map.html"
RANDOM_SEED = 0


# --- helpers ----------------------------------------------------------------


def load_reviews(seed: int) -> pd.DataFrame:
    """Stream a stratified sample of Amazon reviews from HuggingFace.

    The ``SOURCE_PRODUCT_CATEGORIES`` list below is *input to sampling*, not
    configuration for Typologist. We pick six Amazon-curator-assigned
    product categories and draw a balanced number of reviews from each so
    the corpus isn't dominated by one type. Typologist will then discover
    its own categorization from the review text alone (see Step 3 in main).
    """
    from datasets import load_dataset

    source_product_categories = [
        "Books",
        "Electronics",
        "All_Beauty",
        "Clothing_Shoes_and_Jewelry",
        "Home_and_Kitchen",
        "Toys_and_Games",
    ]
    n_per_category = 83  # 6 x 83 = ~498 reviews total
    stream_window = 3000  # Stream this many per category before sampling
    min_text_chars = 200  # Drop one-liner reviews; they embed poorly

    jsonl_url = (
        "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023"
        "/resolve/main/raw/review_categories/{category}.jsonl"
    )

    rng = np.random.default_rng(seed)
    parts: list[pd.DataFrame] = []
    for category in source_product_categories:
        print(f"  streaming {category}...", flush=True)
        ds = load_dataset(
            "json",
            data_files=jsonl_url.format(category=category),
            split="train",
            streaming=True,
        )
        rows: list[dict] = []
        for row in ds:
            text = row.get("text") or ""
            if (
                row.get("verified_purchase")
                and len(text) >= min_text_chars
                and row.get("rating") is not None
            ):
                rows.append({"text": text, "rating": float(row["rating"])})
            if len(rows) >= stream_window:
                break

        picked = rng.choice(len(rows), size=n_per_category, replace=False)
        cat_df = pd.DataFrame([rows[i] for i in picked])
        cat_df["product_category"] = category
        parts.append(cat_df)

    return pd.concat(parts, ignore_index=True)


def embed_documents(texts: list[str]) -> np.ndarray:
    """Embed documents. **Swap this function if you prefer a different embedder.**

    Typologist doesn't care which embedder produced its inputs. The only
    contract is: take ``list[str]`` of length n, return ``np.ndarray`` of
    shape ``(n, d)`` with floating-point dtype. The default below uses
    sentence-transformers' ``all-MiniLM-L6-v2`` (384-dim, local, free, CPU-
    fine) because it keeps the example key-free beyond Anthropic.

    Two common alternatives you can paste in to replace this function:

        # Cohere (remote, paid, 1536-dim; reads CO_API_KEY):
        import cohere
        def embed_documents(texts):
            client = cohere.ClientV2()
            parts = []
            for i in range(0, len(texts), 96):
                resp = client.embed(
                    texts=texts[i:i+96],
                    model="embed-v4.0",
                    input_type="search_document",
                    embedding_types=["float"],
                )
                parts.append(np.array(resp.embeddings.float_, dtype=np.float32))
            return np.vstack(parts)

        # OpenAI (remote, paid, 1536-dim):
        import openai
        def embed_documents(texts):
            client = openai.OpenAI()
            resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
            return np.array([d.embedding for d in resp.data], dtype=np.float32)
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)


def _build_hover_text(
    texts: list[str],
    labels_df: pd.DataFrame,
    max_text_chars: int = 500,
) -> list[str]:
    """Per-point tooltip: Typologist facet values, then the (truncated) text.

    Labels go first so readers see Typologist's output on hover at a glance;
    the review text follows in case they want to check the labels against
    the source.
    """
    hovers: list[str] = []
    for i, raw in enumerate(texts):
        parts = [f"{col}: {labels_df[col].iloc[i]}" for col in labels_df.columns]
        text = raw if len(raw) <= max_text_chars else raw[:max_text_chars] + "..."
        hovers.append("\n".join(parts) + "\n\n" + text)
    return hovers


def render_map(
    embeddings: np.ndarray,
    labels_df: pd.DataFrame,
    hover_text: list[str],
    output_path: Path,
    seed: int,
) -> None:
    """Render an interactive DataMapPlot of the corpus.

    Each Typologist-discovered facet becomes one selectable colormap in the
    ``colormaps=`` dict; a dropdown in the plot lets the viewer switch point
    coloring between facets. Region text annotations (``*label_layers``) are
    deliberately left off: the right source for those is Toponymy fit on the
    2D coords, which is its own can of worms (picks a clusterer, manages
    upstream compat, another ~$0.25 and a minute of runtime) and risks
    implying the region labels are Typologist output when they are not.
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

    # === Step 1: load a sample corpus ===
    print("Step 1: loading Amazon reviews from HuggingFace...")
    df = load_reviews(seed=RANDOM_SEED)
    print(f"  loaded {len(df)} reviews across {df['product_category'].nunique()} categories\n")

    # === Step 2: embed ===
    print("Step 2: embedding with sentence-transformers all-MiniLM-L6-v2...")
    embeddings = embed_documents(df["text"].tolist())
    print(f"  embeddings shape: {embeddings.shape}\n")

    # === Step 3: run Typologist ==============================================
    # This is the part the example is actually demonstrating. Everything else
    # in this file is plumbing. Typologist takes the documents and their
    # embeddings, discovers n_facets categorical axes, and assigns each
    # document a value on each axis.
    #
    # naming_llm, schema_llm, and labeling_llm default to Anthropic model
    # strings (so ANTHROPIC_API_KEY is read from the env). If you use a
    # different provider, pass a callable(prompt: str) -> str to any of those
    # three kwargs; see docs/design.md for the full contract.
    print("Step 3: fitting Typologist (n_facets=3)...")
    t = Typologist(
        n_facets=3,
        topic_embedder=SentenceTransformer("all-MiniLM-L6-v2"),
        object_description="product reviews",
        corpus_description="Amazon product reviews",
        random_state=RANDOM_SEED,
        verbose=True,
    ).fit(df["text"], embeddings)
    # =========================================================================

    # === Step 4: print the output ===
    print("\n=== Discovered schema ===")
    for i, facet in enumerate(t.schema_):
        print(f"\nFacet {i}: {facet['name']} ({facet['kind']})")
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

    # === Step 5: render the map ===
    print("\nStep 5: rendering interactive map...")
    labels_for_map = t.labels_df_.reset_index(drop=True)
    hover = _build_hover_text(df["text"].tolist(), labels_for_map)
    render_map(
        embeddings=embeddings,
        labels_df=labels_for_map,
        hover_text=hover,
        output_path=HTML_OUTPUT,
        seed=RANDOM_SEED,
    )
    print(f"  saved {HTML_OUTPUT.relative_to(Path.cwd())}")
    print(f"  open in a browser: file://{HTML_OUTPUT}")


if __name__ == "__main__":
    main()


# Sample output from a run on 2026-04-22 with seed=0 and MiniLM embeddings.
# Facets 0 and 1 (product_category and review_sentiment) are stable across
# seeds; Facet 2 varies more (EVoC clustering is non-deterministic and the
# third facet is the farthest from the embedding's dominant axes, so it
# picks up whichever orthogonal structure the LLM finds most discriminating
# on a given run).
#
# === Discovered schema ===
#
# Facet 0: product_category (categorical)
#   The general product category that the Amazon review is about.
#   - books_and_cookbooks
#   - apparel_and_footwear
#   - kitchen_and_cookware
#   - toys_and_games
#   - personal_care_and_beauty
#   - hair_accessories
#   - electronics_and_tech_accessories
#   - Other
#
# Facet 1: review_sentiment (categorical)
#   The overall evaluative tone the reviewer expresses toward the product,
#   independent of what the product is.
#   - highly_positive
#   - mixed_with_reservations
#   - disappointed_negative
#   - neutral_descriptive
#   - Other
#
# Facet 2: review_focus_aspect (categorical)
#   The primary evaluative dimension the reviewer emphasizes when assessing
#   the product.
#   - fit_and_sizing
#   - durability_and_build_quality
#   - ease_of_use_and_assembly
#   - value_for_money
#   - sensory_experience
#   - content_and_storytelling
#   - functional_performance
#   - aesthetic_and_design
#   - Other
#
# Facet 0's crosstab against Amazon's own product_category shows heavy
# diagonal concentration and meaningful refinement: Typologist splits
# All_Beauty into personal_care + hair_accessories, and lumps
# Clothing_Shoes_and_Jewelry's apparel/footwear into a single bucket with
# hair_accessories pulled out. Facets 1 and 2 add sentiment and
# evaluation-focus axes that product_category alone doesn't capture.
