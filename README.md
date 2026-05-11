# Typologist

Induce a multi-facet categorical schema from a corpus of documents.

## Status

**Alpha.** 0.1.0 on PyPI. The public API may change as we iterate. See [`docs/design.md`](docs/design.md) for the current contract.

## What it does

You give it documents and their embeddings. It gives you back a handful of *facets* (each a named categorical axis with a short list of values), discovered from the corpus in one shot. The output shape mirrors faceted classification, the library-science approach to multi-axis categorical description (Ranganathan's colon classification, 1933); Typologist is the automated form of that activity. Run it on a corpus of Amazon product reviews and you get three facets (`product_category`, `reviewer_sentiment`, and `review_focus_aspect`), each with a definition and 5-10 values. Worked example with real output below.

The pipeline is a single pass: cluster the embeddings with [Toponymy](https://github.com/TutteInstitute/toponymy) (which composes EVoC clustering + LLM cluster naming), then ask a schema-synthesis LLM to propose `n_facets` mutually orthogonal categorical axes from the resulting cluster hierarchy. Per-document labels are obtained separately, via `apply_schema(t.schema_, documents, llm=...)`, so discovery and labeling can be sequenced and budgeted independently.

## Schema induction vs. topic modeling

Topic modeling is the per-facet primitive Typologist uses under the hood (specifically Toponymy on top of [EVoC](https://github.com/TutteInstitute/evoc)), not the activity it performs. A topic model partitions a corpus into topics; schema induction synthesizes such a partition into a named facet and stacks multiple orthogonal facets in a single output. If you've used BERTopic or similar:

- **Multiple orthogonal axes, not one partition.** A topic model gives you one bucket per document. Typologist's `n_facets` axes are designed to be mutually orthogonal (a document's value on one facet shouldn't predict its value on another), so cross-cutting analysis is a one-line `groupby` rather than a second clustering pass.
- **Concept-labeled values, not keyword bags.** Each facet's values are short phrases (`fit_and_sizing`, `value_for_money`) backed by an LLM-written labeling prompt that can be reapplied to new documents. There's no c-TF-IDF keyword list to interpret.
- **Discovery and labeling are separable steps.** `fit()` returns just the schema (cheap; one Toponymy run + one schema LLM call). `apply_schema()` does the per-document labeling on demand, so you can label the same corpus you discovered on, label a held-out sample, or label streaming new documents without re-running discovery.

## Install

Requires Python 3.11+. Pick a provider extra; the package itself is provider-neutral.

```bash
uv add 'typologist[anthropic]'
# or: pip install 'typologist[anthropic]'
```

Extras: `[anthropic]`, `[openai]`, `[all]`. None of them are activated by default. If you want to wire up your own provider via a `Callable[[str], str]`, install bare `typologist` and skip the extras entirely.

You'll also want:

- An API key for whichever provider you picked: `ANTHROPIC_API_KEY` for `AnthropicLLM`, `OPENAI_API_KEY` for `OpenAILLM`. The example below uses Anthropic.
- A sentence-embedding model that Toponymy (the cluster-naming library Typologist builds on) can use internally for keyphrases and topic names. `sentence-transformers` with MiniLM is cheap and good enough for most use cases:

  ```bash
  uv pip install sentence-transformers
  ```

> [!IMPORTANT]
> Fitting Typologist makes paid LLM API calls. `fit()` is cheap (one Toponymy run + one schema LLM call) — typically well under a dollar for a 500-1000 doc corpus. `apply_schema()` is the cost-dominant step: one LLM call per (document, facet) pair, threadpooled. See [Performance](#performance) for the cost shape and knobs.

## Quick start

```python
import numpy as np
from sentence_transformers import SentenceTransformer
from typologist import AnthropicLLM, Typologist, apply_schema

documents = [...]              # list[str], one per document
embeddings = np.array(...)     # shape (n_docs, d), float

t = Typologist(
    n_facets=3,
    topic_embedder=SentenceTransformer("all-MiniLM-L6-v2"),
    naming_llm=AnthropicLLM("claude-haiku-4-5"),
    schema_llm=AnthropicLLM("claude-opus-4-7"),
    labeling_llm=AnthropicLLM("claude-haiku-4-5"),
).fit(documents, embeddings)

# Per-document labels: separate step
labels = apply_schema(t.schema_, documents, llm=AnthropicLLM("claude-haiku-4-5"))
```

The three LLM kwargs are required and have no defaults. `labeling_llm` is provenance-only at fit time — it stamps each facet's `labeling_model` field with a `"provider:model"` string so `apply_schema()` callers know which model the schema was designed for; `fit()` itself never calls it. Swap any of the three for `OpenAILLM("gpt-4.1-mini")`, a custom `LLM` subclass, or a plain `Callable[[str], str]` to use a different provider; the rest of the API stays the same.

Run on a corpus of Amazon product reviews ([`examples/amazon_reviews.py`](examples/amazon_reviews.py) is the runnable script), `t.schema_` looks like:

```
Facet 0: product_category (categorical)
  The broad product domain that the review is about, distinguishing
  reviews by the type of item being evaluated.
  - books
  - apparel_and_footwear
  - kitchen_and_cookware
  - toys
  - personal_care_and_beauty
  - consumer_electronics
  - Other

Facet 1: reviewer_sentiment (categorical)
  The overall sentiment polarity and intensity expressed by the reviewer
  toward the product.
  - strongly_positive
  - mildly_positive
  - mixed
  - mildly_negative
  - strongly_negative
  - Other

Facet 2: review_focus_aspect (categorical)
  The primary product attribute or dimension the reviewer focuses their
  evaluation on.
  - fit_and_sizing
  - durability_and_build_quality
  - ease_of_use_and_instructions
  - sensory_experience
  - functional_performance
  - value_for_money
  - aesthetic_appearance
  - content_and_informational_value
  - Other
```

Each facet entry is a JSON-serializable dict (`name`, `kind`, `values`, `definition`, plus a stored `labeling_prompt_template` and `labeling_model` for reuse; see [Applying a schema](#applying-a-schema)). The output of `apply_schema(t.schema_, documents, llm=...)` is a `(n_docs, n_facets)` DataFrame of pandas Categoricals, positionally aligned with the input documents and ready to join back onto the source corpus.

## Applying a schema

Every facet entry stores its own `labeling_prompt_template` and a `labeling_model` provenance string (e.g., `"anthropic:claude-haiku-4-5"`), so you can apply the same schema to new documents without re-running discovery. `apply_schema` is provider-neutral, so you pass the LLM you want to label with:

```python
from typologist import AnthropicLLM, apply_schema

labels = apply_schema(
    schema=t.schema_,
    documents=documents,
    llm=AnthropicLLM("claude-haiku-4-5"),
)
# labels is a (n_docs, n_facets) DataFrame of pandas Categoricals
df_labeled = df.join(labels.set_axis(df.index))
```

The same call shape works for held-out documents, streaming new documents, or any other corpus the schema makes sense for.

See [`docs/design.md`](docs/design.md) for the full schema entry shape and `apply_schema` contract.

## Performance

`fit()` is cheap: one Toponymy run (which makes a handful of LLM cluster-naming calls) plus one schema-synthesis call. On a typical 500-1000 doc corpus with `n_facets=3`, expect a few minutes and well under a dollar in API spend.

`apply_schema()` is the cost-dominant step: it makes one LLM call per `(document, facet)` pair, dispatched through a threadpool (`max_concurrency=10` by default). On 1000 docs with `n_facets=3` that's 3000 LLM calls, typically a few minutes wall time. Cost on the Anthropic Haiku path runs roughly $0.50-2 per 1000 docs at `n_facets=3`, depending on document length. Per-document length is the dominant cost variable; pre-summarizing long documents is the most effective cost-management knob if your inputs are long.

Three other levers for managing cost on large corpora: sample to a representative subset before fitting (smaller corpus for discovery), apply the schema only to the subset you care about labels for (you don't have to label everything you fit on), or lower `n_facets`.

## Model choice

**LLMs.** Typologist is provider-neutral. Pick whichever you have keys for. `AnthropicLLM` and `OpenAILLM` ship in the package; subclass `LLM` (or pass a `Callable[[str], str]`) for anything else. The three roles trade off differently:

- `schema_llm` runs only once per `fit()` and is the quality-dominant step. Use the strongest model you'll pay for here.
- `naming_llm` (used by Toponymy for cluster naming) is called a handful of times per fit. Mid-tier is fine.
- `labeling_llm` is called by `apply_schema()` once per (document, facet) — orders of magnitude more often than the other two. If you're cost-cutting, downgrade this first; Haiku-class models are typically the right trade-off.

**Embeddings.** The input embeddings (your `(n_docs, d)` array) and the `topic_embedder` Toponymy uses for keyphrases and exemplar selection are separate slots and don't have to come from the same model. In practice a local sentence-transformers `topic_embedder` works fine even when input embeddings come from a stronger remote model (Cohere, OpenAI), so you can put your embedding budget on the input embeddings without losing quality on the cluster-naming side.

## Related

Typologist is an independent project with no affiliation to the authors of the libraries it builds on:

- [Toponymy](https://github.com/TutteInstitute/toponymy): cluster naming and hierarchy
- [EVoC](https://github.com/TutteInstitute/evoc): hierarchical clustering

[DataMapPlot](https://github.com/TutteInstitute/datamapplot) pairs naturally with the stack: Toponymy supplies hierarchical region labels for `label_layers=`, and `apply_schema()` supplies per-document facet labels for `colormaps=`.

## Questions and bug reports

Open an issue at [github.com/stevenfazzio/typologist/issues](https://github.com/stevenfazzio/typologist/issues). Feedback is especially welcome while the API is still settling.

## License

BSD-3-Clause. See [LICENSE](LICENSE).
