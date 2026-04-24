"""Discover categorical facets in a sample of Amazon product reviews.

**Step 3 is where Typologist actually runs.** Steps 1, 2, 4, and 5 are
plumbing around this particular example (sampling Amazon reviews, embedding
them, printing and visualizing the output). If you're skimming to see what
Typologist does, start with the ``Typologist(...).fit(...)`` call inside
``_run_pass``.

What this does:
1. Streams a stratified sample of ~500 Amazon reviews across 6 product
   categories from HuggingFace.
2. Embeds the reviews locally with sentence-transformers all-MiniLM-L6-v2.
   Swappable, see the ``embed_documents`` docstring.
3. **Runs Typologist twice to discover 3 categorical facets per run:**
   - Vanilla pass: discovers facets from the review text alone.
   - Erasure pass: discovers facets after concept-erasing the curator-
     assigned ``product_category`` column, so the result is orthogonal
     to what the corpus already came tagged with.
4. Prints each discovered schema and a crosstab of facet 0 against the
   curator-assigned product category (to show the rediscovery effect in
   the vanilla pass and its absence in the erasure pass).
5. Renders one interactive DataMapPlot HTML per pass: point colors switch
   between Typologist facets via a dropdown; hovering on a point shows
   the assigned facet values followed by the review text itself.

Expected runtime: ~10 minutes on a laptop.
Expected cost: ~$6 (Anthropic LLM calls for schema synthesis and per-doc
labeling, two passes; embedding runs locally and is free).

Required environment variables:
    ANTHROPIC_API_KEY  Anthropic API key (Typologist's default LLM provider)

Required extra installs (on top of `typologist` itself):
    uv pip install datasets sentence-transformers umap-learn datamapplot

Run from the repo root:
    uv run python examples/amazon_reviews.py

The two ``*_map.html`` outputs land in ``examples/`` (gitignored) so
dev runs don't touch the published versions. To refresh the README's
linked interactive map after a run you're happy with, copy them into
``docs/`` by hand:
    cp examples/amazon_reviews_{vanilla,erased}_map.html docs/
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUTPUT_DIR = Path(__file__).parent
HTML_OUTPUT_VANILLA = OUTPUT_DIR / "amazon_reviews_vanilla_map.html"
HTML_OUTPUT_ERASED = OUTPUT_DIR / "amazon_reviews_erased_map.html"
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
    sub_title: str,
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
        sub_title=sub_title,
        inline_data=True,
    )
    fig.save(str(output_path))


# --- main -------------------------------------------------------------------


def _run_pass(
    label: str,
    topic_embedder,
    documents: pd.Series,
    embeddings: np.ndarray,
    curator_category: np.ndarray,
    output_html: Path,
    sub_title: str,
    metadata: pd.DataFrame | None,
) -> None:
    """Fit Typologist once, print the schema and crosstab, render the map.

    Called twice from ``main``: once with ``metadata=None`` (vanilla) and once
    with ``metadata=df[["product_category"]]`` (erasure). The only difference
    between the two calls is the ``metadata=`` keyword.
    """
    from typologist import Typologist

    print(f"Fitting Typologist ({label}, n_facets=3)...")
    t = Typologist(
        n_facets=3,
        topic_embedder=topic_embedder,
        object_description="product reviews",
        corpus_description="Amazon product reviews",
        random_state=RANDOM_SEED,
        verbose=True,
    ).fit(documents, embeddings, metadata=metadata)

    print(f"\n=== Discovered schema ({label}) ===")
    for i, facet in enumerate(t.schema_):
        print(f"\nFacet {i}: {facet['name']} ({facet['kind']})")
        print(f"  {facet['definition']}")
        for value in facet["values"]:
            print(f"  - {value}")

    primary_facet = t.schema_[0]["name"]
    print(f"\n=== {label}: Facet 0 ({primary_facet}) vs curator-assigned product category ===\n")
    # Use a distinct column name so a facet named "product_category" doesn't
    # collide with the curator column when we join them.
    labels = t.labels_df_.reset_index(drop=True)
    labels["curator_category"] = curator_category
    crosstab = pd.crosstab(labels["curator_category"], labels[primary_facet])
    print(crosstab.to_string())

    print(f"\nRendering interactive map ({label})...")
    labels_for_map = t.labels_df_.reset_index(drop=True)
    hover = _build_hover_text(documents.tolist(), labels_for_map)
    render_map(
        embeddings=embeddings,
        labels_df=labels_for_map,
        hover_text=hover,
        output_path=output_html,
        seed=RANDOM_SEED,
        sub_title=sub_title,
    )
    print(f"  saved {output_html.relative_to(Path.cwd())}")
    print(f"  open in a browser: file://{output_html}")


def main() -> None:
    from sentence_transformers import SentenceTransformer

    # === Step 1: load a sample corpus ===
    print("Step 1: loading Amazon reviews from HuggingFace...")
    df = load_reviews(seed=RANDOM_SEED)
    print(f"  loaded {len(df)} reviews across {df['product_category'].nunique()} categories\n")

    # === Step 2: embed ===
    print("Step 2: embedding with sentence-transformers all-MiniLM-L6-v2...")
    embeddings = embed_documents(df["text"].tolist())
    print(f"  embeddings shape: {embeddings.shape}\n")

    # === Step 3: run Typologist (vanilla + erasure) ==========================
    # This is the part the example is actually demonstrating. Everything else
    # in this file is plumbing. Typologist takes the documents and their
    # embeddings, discovers n_facets categorical axes, and assigns each
    # document a value on each axis.
    #
    # We run it twice on the same corpus to illustrate the effect of
    # metadata erasure:
    #   - Vanilla pass: no metadata, discovery is driven by the embeddings
    #     alone. Facet 0 tends to rediscover product_category.
    #   - Erasure pass: pass product_category to ``metadata=``. LEACE
    #     projects its linear signal out of the embeddings and the synthesis
    #     prompt is told to avoid it, so discovery finds something else.
    #
    # naming_llm, schema_llm, and labeling_llm default to Anthropic model
    # strings (so ANTHROPIC_API_KEY is read from the env). If you use a
    # different provider, pass a callable(prompt: str) -> str to any of those
    # three kwargs; see docs/design.md for the full contract.
    topic_embedder = SentenceTransformer("all-MiniLM-L6-v2")

    _run_pass(
        label="vanilla",
        topic_embedder=topic_embedder,
        documents=df["text"],
        embeddings=embeddings,
        curator_category=df["product_category"].values,
        output_html=HTML_OUTPUT_VANILLA,
        sub_title="Typologist facets, discovered from review text alone.",
        metadata=None,
    )

    _run_pass(
        label="erasure",
        topic_embedder=topic_embedder,
        documents=df["text"],
        embeddings=embeddings,
        curator_category=df["product_category"].values,
        output_html=HTML_OUTPUT_ERASED,
        sub_title="Typologist facets, discovered after erasing product_category.",
        metadata=df[["product_category"]],
    )
    # =========================================================================


if __name__ == "__main__":
    main()


# Sample output from a run on 2026-04-24 with seed=0 and MiniLM embeddings.
# The first two facets in each pass are broadly stable across seeds (a
# product/category axis in the vanilla pass, sentiment in both); the third
# facet varies more because EVoC clustering is non-deterministic and the
# third facet is the farthest from the embedding's dominant axes.
#
# === Discovered schema (vanilla) ===
#
# Facet 0: product_category (categorical)
#   The broad product domain that the review is about, distinguishing
#   reviews by the type of item being evaluated.
#   - books
#   - apparel_and_footwear
#   - kitchen_and_cookware
#   - toys
#   - personal_care_and_beauty
#   - consumer_electronics
#   - Other
#
# Facet 1: reviewer_sentiment (categorical)
#   Captures the overall sentiment polarity and intensity expressed by the
#   reviewer toward the product.
#   - strongly_positive
#   - mildly_positive
#   - mixed
#   - mildly_negative
#   - strongly_negative
#   - Other
#
# Facet 2: review_focus_aspect (categorical)
#   The primary product attribute or dimension the reviewer focuses their
#   evaluation on.
#   - fit_and_sizing
#   - durability_and_build_quality
#   - ease_of_use_and_instructions
#   - sensory_experience
#   - functional_performance
#   - value_for_money
#   - aesthetic_appearance
#   - content_and_informational_value
#   - Other
#
# Vanilla Facet 0's crosstab against the curator-assigned product_category
# shows heavy diagonal concentration: Typologist rediscovers ~80% of the
# curator's buckets from the review text alone. This is the "you got back
# what you already had" failure mode that the erasure pass below addresses.
#
#                             books  apparel  kitchen  toys  personal_care  electronics  Other
# All_Beauty                      0        8        2     2             70            1      0
# Books                          76        0        1     1              0            0      5
# Clothing_Shoes_and_Jewelry      0       75        0     0              2            1      5
# Electronics                     0        6        0     0              0           75      2
# Home_and_Kitchen                0        9       53     2              0            7     12
# Toys_and_Games                  0        2        0    78              0            1      2
#
# === Discovered schema (erasure, metadata=df[["product_category"]]) ===
#
# Facet 0: review_sentiment (categorical)
#   The overall sentiment polarity and intensity expressed by the reviewer
#   toward the product.
#   - highly_positive
#   - mildly_positive
#   - mixed
#   - mildly_negative
#   - highly_negative
#   - Other
#
# Facet 1: review_focus_aspect (categorical)
#   The primary product attribute the reviewer evaluates or focuses on in
#   their review.
#   - fit_and_sizing
#   - durability_and_build_quality
#   - ease_of_use_and_setup
#   - value_for_money
#   - aesthetics_and_design
#   - comfort_and_feel
#   - performance_and_effectiveness
#   - content_and_storytelling
#   - Other
#
# Facet 2: reviewer_purchase_motivation (categorical)
#   Captures the reviewer's stated reason or context for purchasing the
#   product, reflecting who the item is for and the situation of use.
#   - gift_for_others
#   - personal_use
#   - replacement_or_upgrade
#   - professional_or_work_use
#   - hobby_or_leisure
#   - child_or_family_use
#   - travel_or_outdoor_use
#   - Other
#
# With product_category erased, Facet 0 is no longer product-aligned:
#
# review_sentiment            highly_pos  mildly_pos  mixed  mildly_neg  highly_neg
# All_Beauty                          40          12      9           9          13
# Books                               43          16      9           8           7
# Clothing_Shoes_and_Jewelry          38          20      8           7          10
# Electronics                         41          15     12           7           8
# Home_and_Kitchen                    37          16     12          12           6
# Toys_and_Games                      45          13     15           1           9
#
# Facets 1 and 2 are the erasure pass's payoff: review_focus_aspect (what
# the reviewer is evaluating) and reviewer_purchase_motivation (why they
# bought it) are axes of variation the vanilla pass never surfaced, because
# the embedding geometry was dominated by product-category structure.
