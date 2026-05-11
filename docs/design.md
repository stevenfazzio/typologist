# Typologist 0.1 design

This document is the public-API contract the 0.1 implementation works against. It is intentionally narrow: what ships in 0.1, what the types and shapes are, and what is deliberately parked for later.

## Scope

**In scope for 0.1.**

- A single `Typologist` class (sklearn-style, `fit`-then-attributes), returning a discovered schema
- A top-level `apply_schema` helper for labeling documents with a discovered schema
- Single-pass discovery: one Toponymy run + one schema-synthesis LLM call propose all `n_facets` mutually orthogonal categorical axes
- Synchronous execution only

**Out of scope for 0.1.** See "Parking lot" at the end.

## Architecture

Typologist's architectural shape is a deliberate commitment that shows up in nearly every API decision: the LLM is used only at the leaves of the pipeline as a stateless function, while all integration across the corpus happens in a geometric substrate.

**LLM calls are small, local, and stateless.** There are exactly three LLM roles (`naming_llm`, `schema_llm`, `labeling_llm`), and each call operates on a narrow input: one cluster to name (via Toponymy), one cluster hierarchy from which to synthesize `n_facets` axes, one document to label on one facet (via `apply_schema`). No call sees the full corpus. No call holds memory of other calls. Each is a parallelizable, cacheable, retryable function invocation.

**Integration happens in the geometric substrate.** The embedding space (via similarity and EVoC clustering), the topic hierarchy (via Toponymy), and the structural operations on them (nearest-neighbor graphs, exemplar selection) are where documents are woven into a coherent typology. This layer does not reason; it operates on shape.

This split produces corpus-level exchangeability: the pipeline is approximately order-invariant across documents, so every document gets equal weight on the emergent typology rather than being read through the lens of whatever came before it in an LLM's context. The inductive bias is the same one that justifies bag-of-words models, de Finetti exchangeability, and hierarchical Bayes: given a population of cases, treat them as order-invariant when you're trying to recover latent structure.

The commitment is intentional. Agentic architectures are useful for path-dependent problems (planning, search, long-form summarization); typology extraction is not that kind of problem. The cost is correctly priced: Typologist cannot exploit long-range correlations between documents, which is the right trade-off for building orthogonal category systems but a fatal one for, say, summarizing a novel.

## Public surface

```python
from typologist import AnthropicLLM, Typologist, apply_schema

t = Typologist(
    n_facets=3,
    topic_embedder=...,                              # required, no default
    naming_llm=AnthropicLLM("claude-haiku-4-5"),     # required, keyword-only
    schema_llm=AnthropicLLM("claude-opus-4-7"),      # required, keyword-only
    labeling_llm=AnthropicLLM("claude-haiku-4-5"),   # required, keyword-only; provenance only
    object_description="objects",
    corpus_description="collection of objects",
    random_state=None,
    noise_label="Unlabelled",
    verbose=False,
).fit(documents, embeddings)

t.schema_         # list[dict]
t.diagnostics_    # dict

labels_df = apply_schema(t.schema_, documents, llm=AnthropicLLM("claude-haiku-4-5"))
```

## Constructor parameters

| param | type | default | notes |
|---|---|---|---|
| `n_facets` | `int` | required | number of categorical facets to discover; no `"auto"` in 0.1 |
| `topic_embedder` | object implementing `TextEmbedderProtocol` | required | no default; matches Toponymy's stance. Intended for Toponymy's internal keyphrase/topic-name embedding. `sentence_transformers.SentenceTransformer("all-MiniLM-L6-v2")` is a recommended baseline |
| `naming_llm` | `LLM \| Callable` | required, keyword-only | names Toponymy's clusters. Called O(n_clusters x n_layers) times per fit. Pass `AnthropicLLM(...)`, `OpenAILLM(...)`, a custom `LLM` subclass, or a `Callable[[str], str]` |
| `schema_llm` | `LLM \| Callable` | required, keyword-only | synthesizes all `n_facets` facets from the cluster hierarchy in a single structured-output call. Called once per fit; quality-dominant step |
| `labeling_llm` | `LLM \| Callable` | required, keyword-only | provenance only at fit time. Stamped onto each facet's `labeling_model` field as `"provider:model"`; never called by `fit()`. Used by `apply_schema` if the caller passes the same LLM through |
| `object_description` | `str` | `"objects"` | describes what each document is. Passed through to Toponymy; also rendered into our schema-synthesis and labeling prompts |
| `corpus_description` | `str` | `"collection of objects"` | describes the collection as a whole. Same passthrough behavior |
| `random_state` | `int \| None` | `None` | threaded into NumPy RNG / sampling. EVoC has no `random_state` and is the residual source of non-determinism; this is documented, not fixed |
| `noise_label` | `str` | `"Unlabelled"` | sentinel string for docs that couldn't be classified into any value during `apply_schema`. British spelling matches Toponymy and DataMapPlot |
| `verbose` | `bool` | `False` | cascades to Toponymy and other noisy subcomponents |
| `use_toponymy` | `bool` | `True` | when `False`, swaps Toponymy for a minimal single-layer EVoC + per-cluster naming. Exposed for benchmarking; default-on is the supported path |

## `fit(documents, embeddings)`

Returns `self`.

- `documents`: `list[str]` or `pd.Series[str]`. If a Series, its index is preserved if you later call `apply_schema` on the same Series.
- `embeddings`: `np.ndarray` of shape `(n_docs, dim)`. Always row-aligned positionally with `documents`; never index-joined.

The fit is one Toponymy run + one schema-synthesis LLM call. Per-document labels are *not* produced at fit time; call `apply_schema(t.schema_, documents, llm=...)` to label.

## Fitted attributes

### `schema_: list[dict]`

One entry per discovered facet:

```python
{
    "name": str,                      # column name used by apply_schema; unique across facets
    "kind": "categorical",            # 0.1 only emits categorical; ordinal is parked
    "values": list[str],              # value vocabulary for this facet, "Other" appended
    "definition": str,                # human-readable semantic definition
    "labeling_prompt_template": str,  # f-string template containing "{document}"
    "labeling_model": str | None,     # "{provider}:{model_name}" provenance string;
                                      # None if labeling_llm was a callable
}
```

**`labeling_model`.** Provenance only: a `"provider:model"` string (e.g., `"anthropic:claude-haiku-4-5"`, `"openai:gpt-4o-mini"`) identifying the LLM intended for labeling this schema. `apply_schema` does *not* auto-resolve this back to an LLM instance; you always pass `llm=` explicitly. The string is informational, intended for humans reading a JSON-serialized schema.

**`labeling_prompt_template` format.** Plain Python f-string with a single `{document}` variable. No Jinja, no additional placeholders, no hidden context substituted at call time. The template is fully self-contained at storage time: `object_description` and `corpus_description` are rendered into the template when the schema is generated, so downstream users reusing a schema do not need to know the original descriptions.

**Uniqueness.** Facet names must be unique across `schema_`. If LLM synthesis produces a duplicate, Typologist raises loudly rather than silently disambiguating.

**Serialization.** `schema_` is JSON-serializable as written. No custom types, no callable references.

### `diagnostics_: dict`

A single dict with provenance from the fit. Currently:

- `synthesis_prompt`: the prompt sent to `schema_llm`
- `cluster_count`: max cluster count across Toponymy's hierarchy layers
- `hierarchy_depth`: number of Toponymy hierarchy layers

Exact fields may evolve across minor versions; `diagnostics_` is **not** part of the stable persisted-state contract.

## `apply_schema(schema, documents, llm, noise_label="Unlabelled", max_concurrency=10, verbose=False) -> pd.DataFrame`

Applies an existing schema to documents.

- `schema`: a `list[dict]` in the `schema_` format, or a single facet dict.
- `documents`: `list[str]` or `pd.Series[str]`. If a Series, its index is preserved onto the returned DataFrame; if a list, the result uses a range index.
- `llm`: `LLM | Callable[[str], str]`. Required. The schema's stored `labeling_model` is provenance only and is not auto-resolved.
- `noise_label`: sentinel for parse failures; defaults to the same string `Typologist` uses by default.
- `max_concurrency`: per-doc labeling calls run through a threadpool. Set to `1` for serial.
- `verbose`: if true, shows a tqdm progress bar per facet.
- Raises `TypeError` if `llm` is omitted.

Returns a `pd.DataFrame` of shape `(n_docs, n_facets)` with `pd.Categorical` columns named after the facets, values drawn from each facet's `values` list (plus `noise_label`).

## Persisted-state compatibility

Across 0.x minor versions, `schema_` keeps read-compatibility, so a JSON-serialized schema produced by an earlier 0.x release can be passed straight into a later `apply_schema`. `diagnostics_` is explicitly excluded from this contract.

## Conventions and non-obvious commitments

- **Noise encoding.** Noise is always the `noise_label` string. `pd.NA` was considered and rejected because DataMapPlot's label-equality logic does not round-trip `pd.NA`. This also matches Toponymy and preserves CSV readability.
- **Column naming.** `apply_schema` output columns are literal `schema_[i]["name"]`. LLM synthesis is responsible for producing unique names; Typologist fails loud on collision.
- **No LLM client shortcut.** The three `*_llm` kwargs are the only way to configure LLMs; there is no `llm_client=` meta-kwarg that fills all three. The three-way split is a core design decision, not a leaky abstraction.
- **Embedding alignment.** `embeddings` is always positionally row-aligned with `documents`. Even when `documents` is a `Series` with a non-default index, we do not index-join against `embeddings`.
- **Discovery and labeling are separable.** `fit()` returns only a schema; per-document labels are obtained via `apply_schema()`. This makes the cost shape explicit (discovery is cheap; labeling is the volume-dominant step) and lets callers budget the two independently.
- **Dependency ecosystem gotchas.** See `CLAUDE.md` for the inherited pins (Toponymy `>=0.5.0,<0.6.0`, `evoc==0.1.3`).

## Parking lot (post-0.1)

Each item is tracked as a GitHub issue with the `parking-lot` label; click through for full design notes. Filter: [is:issue label:parking-lot](https://github.com/stevenfazzio/typologist/issues?q=is%3Aissue+label%3Aparking-lot).

- **Async support (0.2+).** Threadpool covers most labeling-throughput; full async rework with `AsyncLLMWrapper` is the 0.2 story. (see #15)
- **Ordinal facets (0.2+).** Adds `"ordinal"` kind, value-ordering invariant, and `pd.Categorical(..., ordered=True)` in the output. (see #16)
- **Stability helper `stability_check(docs, embeddings, n_seeds=5)` (0.2).** Reports cross-seed agreement on facets, values, and per-doc labels. (see #17)
- **Functional `discover()` (needs separate design pass).** Open question: rich result object alongside the sklearn-style class. (see #19)
- **`predefined_facets=` one-call sugar.** Already composable via `apply_schema`; sugar later if clunky in practice. (see #20)
- **Tier-1 map output (0.2).** Surface 2D coords and hierarchical region labels for direct DataMapPlot use; adds `umap-learn` as optional extra. (see #22)
