"""Discover categorical facets in a sample of Amazon product reviews.

**Step 3 is where Typologist actually runs.** Steps 1, 2, 4, and 5 are
plumbing around this particular example (sampling Amazon reviews, embedding
them, printing the schema, labeling docs, visualizing the result). If you're
skimming to see what Typologist does, start with the ``Typologist(...).fit(...)``
call inside ``main``.

What this does:
1. Streams a stratified sample of ~500 Amazon reviews across 6 product
   categories from HuggingFace.
2. Embeds the reviews locally with sentence-transformers all-MiniLM-L6-v2.
   Swappable, see the ``embed_documents`` docstring.
3. **Runs Typologist.fit on the corpus to discover 3 categorical facets.**
   This is one Toponymy run + one schema_llm call; no per-doc labeling.
4. **Calls apply_schema to label every doc on every facet.** This is the
   cost-dominant step (one LLM call per (doc, facet) pair, threadpooled).
5. Renders an interactive DataMapPlot HTML: a 2D UMAP of the embeddings
   colored by each Typologist facet (dropdown to switch coloring).

This example wires up Anthropic models because that's what the worked
output below was generated with; OpenAI works the same way via
``OpenAILLM(...)``, and any provider can be supplied via a
``Callable[[str], str]``. See the README's "Model choice" section.

Expected runtime: ~3-5 minutes on a laptop.
Expected cost: ~$0.50-1 (Anthropic LLM calls; embedding runs locally and
is free; cost dominated by per-doc labeling).

Required environment variables:
    ANTHROPIC_API_KEY  Anthropic API key (used by this example)

Required extra installs (on top of `typologist[anthropic]`):
    uv pip install datasets sentence-transformers umap-learn datamapplot

Run from the repo root:
    uv run python examples/amazon_reviews.py

The ``amazon_reviews_map.html`` output lands in ``examples/`` (gitignored).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUTPUT_DIR = Path(__file__).parent
HTML_OUTPUT = OUTPUT_DIR / "amazon_reviews_map.html"
RANDOM_SEED = 0


# --- helpers ----------------------------------------------------------------


def load_reviews(seed: int, n_per_category: int = 83) -> pd.DataFrame:
    """Stream a stratified sample of Amazon reviews from HuggingFace.

    The ``SOURCE_PRODUCT_CATEGORIES`` list below is *input to sampling*, not
    configuration for Typologist. We pick six Amazon-curator-assigned
    product categories and draw a balanced number of reviews from each so
    the corpus isn't dominated by one type. Typologist will then discover
    its own categorization from the review text alone.

    ``n_per_category`` defaults to 83 (6 x 83 = ~498 reviews total) for the
    example here; experiments that want a different size pass a larger value.
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
    stream_window = max(3000, n_per_category * 6)
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


def build_hover_text(
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
    """Render an interactive DataMapPlot of the corpus colored by facet labels.

    Each Typologist-discovered facet becomes one selectable colormap in the
    ``colormaps=`` dict; a dropdown in the plot lets the viewer switch point
    coloring between facets.
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
        sub_title="Typologist facets discovered from review text.",
        inline_data=True,
    )
    fig.save(str(output_path))


# --- main -------------------------------------------------------------------


def main() -> None:
    from sentence_transformers import SentenceTransformer

    from typologist import AnthropicLLM, Typologist, apply_schema

    # === Step 1: load a sample corpus ===
    print("Step 1: loading Amazon reviews from HuggingFace...")
    df = load_reviews(seed=RANDOM_SEED)
    print(f"  loaded {len(df)} reviews across {df['product_category'].nunique()} categories\n")

    # === Step 2: embed ===
    print("Step 2: embedding with sentence-transformers all-MiniLM-L6-v2...")
    embeddings = embed_documents(df["text"].tolist())
    print(f"  embeddings shape: {embeddings.shape}\n")

    # === Step 3: discover the schema =========================================
    # This is the part the example is actually demonstrating. Typologist takes
    # the documents and their embeddings, runs Toponymy once, and asks the
    # schema_llm to propose n_facets mutually orthogonal categorical axes in
    # a single structured-output call. No per-document labeling happens here.
    #
    # naming_llm, schema_llm, and labeling_llm are required keyword-only
    # arguments. labeling_llm is provenance-only at fit time (stamped onto
    # each facet's `labeling_model` field); fit() never calls it.
    #
    # This example uses AnthropicLLM (reading ANTHROPIC_API_KEY from the env).
    # For other providers, swap in OpenAILLM(...) or a callable
    # (prompt: str) -> str; see docs/design.md for the contract.
    topic_embedder = SentenceTransformer("all-MiniLM-L6-v2")

    print("Step 3: discovering schema (Typologist.fit)...")
    t = Typologist(
        n_facets=3,
        topic_embedder=topic_embedder,
        object_description="product reviews",
        corpus_description="Amazon product reviews",
        naming_llm=AnthropicLLM("claude-haiku-4-5"),
        schema_llm=AnthropicLLM("claude-opus-4-7"),
        labeling_llm=AnthropicLLM("claude-haiku-4-5"),
        random_state=RANDOM_SEED,
        verbose=True,
    ).fit(df["text"], embeddings)

    print("\n=== Discovered schema ===")
    for i, facet in enumerate(t.schema_):
        print(f"\nFacet {i}: {facet['name']} ({facet['kind']})")
        print(f"  {facet['definition']}")
        for value in facet["values"]:
            print(f"  - {value}")

    # === Step 4: label every doc on every facet ==============================
    # apply_schema runs one LLM call per (document, facet) pair, dispatched
    # through a threadpool. This is the cost-dominant step.
    print(f"\nStep 4: labeling {len(df)} docs via apply_schema...")
    labels_df = apply_schema(
        t.schema_,
        df["text"],
        llm=AnthropicLLM("claude-haiku-4-5"),
        verbose=True,
    )
    df_labeled = df.join(labels_df.set_axis(df.index))
    print(f"  labels_df shape: {labels_df.shape}\n")

    primary_facet = t.schema_[0]["name"]
    print(f"=== Crosstab: Facet 0 ({primary_facet}) vs curator-assigned product_category ===\n")
    crosstab = pd.crosstab(df_labeled["product_category"], df_labeled[primary_facet])
    print(crosstab.to_string())

    # === Step 5: render the interactive map ==================================
    print("\nStep 5: rendering interactive DataMapPlot...")
    hover = build_hover_text(df["text"].tolist(), labels_df)
    render_map(
        embeddings=embeddings,
        labels_df=labels_df,
        hover_text=hover,
        output_path=HTML_OUTPUT,
        seed=RANDOM_SEED,
    )
    print(f"  saved {HTML_OUTPUT.relative_to(Path.cwd())}")
    print(f"  open in a browser: file://{HTML_OUTPUT}")


if __name__ == "__main__":
    main()
