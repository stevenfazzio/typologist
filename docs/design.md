# Typologist 0.1 design

This document is the public-API contract the 0.1 implementation works against. It is intentionally narrow: what ships in 0.1, what the types and shapes are, and what is deliberately parked for later.

## Scope

**In scope for 0.1.**

- A single `Typologist` class (sklearn-style, `fit`-then-attributes)
- A top-level `apply_schema` helper for labeling new documents with a previously-discovered schema
- LEACE pre-erasure of user-supplied known metadata
- Between-facet LEACE residualization against per-doc classifications under synthesized facets (not cluster labels; matches predecessor evidence)
- Synchronous execution only

**Out of scope for 0.1.** See "Parking lot" at the end.

## Architecture

Typologist's architectural shape is a deliberate commitment that shows up in nearly every API decision: the LLM is used only at the leaves of the pipeline as a stateless function, while all integration across the corpus happens in a geometric substrate.

**LLM calls are small, local, and stateless.** There are exactly three LLM roles (`naming_llm`, `schema_llm`, `labeling_llm`), and each call operates on a narrow input: one cluster to name, one set of cluster names to synthesize a facet from, one document to label on one facet. No call sees the full corpus. No call holds memory of other calls. Each is a parallelizable, cacheable, retryable function invocation.

**Integration happens in the geometric substrate.** The embedding space (via similarity, EVoC clustering, and LEACE projections), the topic hierarchy (via Toponymy), and the structural operations on them (nearest-neighbor graphs, facility-location sampling, concept erasure) are where documents are woven into a coherent typology. This layer does not reason; it operates on shape.

This split produces corpus-level exchangeability: the pipeline is approximately order-invariant across documents, so every document gets equal weight on the emergent typology rather than being read through the lens of whatever came before it in an LLM's context. The inductive bias is the same one that justifies bag-of-words models, de Finetti exchangeability, and hierarchical Bayes: given a population of cases, treat them as order-invariant when you're trying to recover latent structure.

The commitment is intentional. Agentic architectures are useful for path-dependent problems (planning, search, long-form summarization); typology extraction is not that kind of problem. The cost is correctly priced: Typologist cannot exploit long-range correlations between documents, which is the right trade-off for building orthogonal category systems but a fatal one for, say, summarizing a novel.

## Public surface

```python
from typologist import AnthropicLLM, Typologist, apply_schema

t = Typologist(
    n_facets=3,
    topic_embedder=...,                         # required, no default
    naming_llm=AnthropicLLM("claude-haiku-4-5"),  # required, keyword-only
    schema_llm=AnthropicLLM("claude-opus-4-7"),   # required, keyword-only
    labeling_llm=AnthropicLLM("claude-haiku-4-5"),  # required, keyword-only
    object_description="objects",
    corpus_description="collection of objects",
    random_state=None,
    noise_label="Unlabelled",
    verbose=False,
).fit(documents, embeddings, metadata=None)

t.schema_                    # list[dict]
t.labels_df_                 # pd.DataFrame, shape (n_docs, n_facets)
t.embeddings_residualized_   # np.ndarray, shape (n_docs, dim)
t.facet_diagnostics_         # list[dict]

labels_df = apply_schema(schema, documents, llm=AnthropicLLM("claude-haiku-4-5"))
```

## Constructor parameters

| param | type | default | notes |
|---|---|---|---|
| `n_facets` | `int` | required | number of categorical facets to discover; no `"auto"` in 0.1 |
| `topic_embedder` | object implementing `TextEmbedderProtocol` | required | no default; matches Toponymy's stance. Intended for Toponymy's internal keyphrase/topic-name embedding. `sentence_transformers.SentenceTransformer("all-MiniLM-L6-v2")` is a recommended baseline (see predecessor evidence) |
| `naming_llm` | `LLM \| Callable` | required, keyword-only | names Toponymy's clusters. Called O(n_clusters x n_layers x n_facets) times. Pass `AnthropicLLM(...)`, `OpenAILLM(...)`, a custom `LLM` subclass, or a `Callable[[str], str]` |
| `schema_llm` | `LLM \| Callable` | required, keyword-only | synthesizes a facet from cluster names. Called O(n_facets) times; quality-dominant step |
| `labeling_llm` | `LLM \| Callable` | required, keyword-only | classifies each document into a facet's value vocabulary. Called O(n_docs x n_facets) times |
| `object_description` | `str` | `"objects"` | describes what each document is. Passed through to Toponymy; also rendered into our schema-synthesis and labeling prompts |
| `corpus_description` | `str` | `"collection of objects"` | describes the collection as a whole. Same passthrough behavior |
| `random_state` | `int \| None` | `None` | threaded into LEACE fit, sampling, NumPy RNG. EVoC has no `random_state` and is the residual source of non-determinism; this is documented, not fixed |
| `noise_label` | `str` | `"Unlabelled"` | sentinel string for docs that couldn't be classified into any value. British spelling matches Toponymy and DataMapPlot |
| `verbose` | `bool` | `False` | cascades to Toponymy and other noisy subcomponents, and to a tqdm bar around the per-doc labeling loop |
| `max_concurrency` | `int` | `10` | number of concurrent threads used to dispatch per-doc labeling calls. Set to `1` for serial |

## `fit(documents, embeddings, metadata=None)`

Returns `self`.

- `documents`: `list[str]` or `pd.Series[str]`. If a Series, its index is preserved onto `labels_df_`. If a list, outputs use a range index.
- `embeddings`: `np.ndarray` of shape `(n_docs, dim)`. Always row-aligned positionally with `documents`; never index-joined.
- `metadata`: `pd.DataFrame | None`. If present, all columns are concept-erased via LEACE before discovery proceeds. Presence of `metadata` is the sole signal of intent to erase; there is no separate `erase_known_metadata` flag.

## Fitted attributes

### `schema_: list[dict]`

One entry per discovered facet, in discovery order:

```python
{
    "name": str,                      # column name in labels_df_; must be unique across facets
    "kind": "categorical",            # 0.1 only emits categorical; ordinal is parked
    "values": list[str],              # value vocabulary for this facet, "Other" appended
    "definition": str,                # human-readable semantic definition
    "labeling_prompt_template": str,  # f-string template containing "{document}"
    "labeling_model": str | None,     # "{provider}:{model_name}" provenance string;
                                      # None if labeling_llm was a callable
}
```

**`labeling_model`.** Provenance only: a `"provider:model"` string (e.g., `"anthropic:claude-haiku-4-5"`, `"openai:gpt-4o-mini"`) identifying the LLM that produced the schema. `apply_schema` does *not* auto-resolve this back to an LLM instance; you always pass `llm=` explicitly. The string is informational, intended for humans reading a JSON-serialized schema.

**`labeling_prompt_template` format.** Plain Python f-string with a single `{document}` variable. No Jinja, no additional placeholders, no hidden context substituted at call time. The template is fully self-contained at storage time: `object_description` and `corpus_description` are rendered into the template when the schema is generated, so downstream users reusing a schema do not need to know the original descriptions.

**Uniqueness.** Facet names must be unique across `schema_`. If LLM synthesis produces a duplicate, Typologist raises loudly rather than silently disambiguating.

**Serialization.** `schema_` is JSON-serializable as written. No custom types, no callable references.

### `labels_df_: pd.DataFrame`

- Shape: `(n_docs, n_facets)`
- Columns: facet names from `schema_` directly (`labels_df_["contribution_type"]`)
- Column dtype: `pd.Categorical`, with `categories` set to the facet's value list plus the `noise_label`
- Index: `documents.index` if `documents` was a `pd.Series`; otherwise `pd.RangeIndex(n_docs)`
- Values: members of `schema_[i]["values"]` or the `noise_label` string

### `embeddings_residualized_: np.ndarray`

- Shape: `(n_docs, dim)`, matching the input `embeddings`
- Contents: the embeddings after all erasure passes (metadata + all facets). Intermediate per-facet residualizations are not retained.

**What "erased" means here.** Each erasure pass is a LEACE projection on either a metadata column (during pre-erasure) or a facet's per-document labels (during between-facet residualization). LEACE removes the *entire* linear subspace of the embeddings that predicts the one-hot encoding of those labels, not just the centroid offset between value groups. In practice this means signals that are linearly correlated with the erased facet or metadata column (even if they aren't it themselves) also get removed. Users chaining `embeddings_residualized_` into downstream tasks whose targets correlate with an erased facet or metadata column (e.g., predicting a 1-5 rating after erasing a sentiment facet) should expect partial erasure of those targets as well.

This compounds across passes. After K facets have been residualized iteratively, the embedding has had K linear subspaces projected out. Predicting any one of those K facets from `embeddings_residualized_` typically drops to chance; correlated signals survive better but still degrade meaningfully. Downstream-predictive uses of `embeddings_residualized_` should account for this.

### `facet_diagnostics_: list[dict]`

One entry per facet, mirroring `schema_` order. Contains provenance and per-facet signals:

- synthesis prompts used (for the `schema_llm` call)
- cluster count and cluster hierarchy depth at the time the facet was proposed
- any entropy or quality signals we compute
- exemplar document ids per value (if cheap to produce)

Exact fields may evolve across minor versions; `facet_diagnostics_` is **not** part of the stable persisted-state contract.

## `apply_schema(schema, documents, llm, noise_label="Unlabelled", max_concurrency=10, verbose=False) -> pd.DataFrame`

Applies an existing schema to new documents without running discovery.

- `schema`: a `list[dict]` in the `schema_` format, or a single facet dict.
- `documents`: `list[str]` or `pd.Series[str]`. Index-preservation behavior matches `fit`.
- `llm`: `LLM | Callable[[str], str]`. Required. The schema's stored `labeling_model` is provenance only and is not auto-resolved.
- `noise_label`: sentinel for parse failures; defaults to the same string `Typologist` uses by default.
- `max_concurrency`: per-doc labeling calls run through a threadpool. Set to `1` for serial.
- `verbose`: if true, shows a tqdm progress bar per facet.
- Raises `TypeError` if `llm` is omitted.

Returns a `pd.DataFrame` in the same shape as `labels_df_` (Categorical columns, noise-labeled per the schema at generation time).

## Persisted-state compatibility

Across 0.x minor versions, the following three attributes keep read-compatibility, so saved-and-loaded Typologist state can be resumed or extended by later releases:

- `schema_`
- `labels_df_`
- `embeddings_residualized_`

`facet_diagnostics_` is explicitly excluded from this contract.

## Conventions and non-obvious commitments

- **Noise encoding.** Noise is always the `noise_label` string. `pd.NA` was considered and rejected because DataMapPlot's label-equality logic does not round-trip `pd.NA`. This also matches Toponymy and preserves CSV readability.
- **Column naming.** `labels_df_` column names are literal `schema_[i]["name"]`. LLM synthesis is responsible for producing unique names; Typologist fails loud on collision.
- **No LLM client shortcut.** The three `*_llm` kwargs are the only way to configure LLMs; there is no `llm_client=` meta-kwarg that fills all three. The three-way split is a core design decision, not a leaky abstraction.
- **Metadata as erasure signal.** Passing `metadata` to `fit` is the only way to request pre-erasure; there is no separate flag. Absent metadata, no pre-erasure runs.
- **Embedding alignment.** `embeddings` is always positionally row-aligned with `documents`. Even when `documents` is a `Series` with a non-default index, we do not index-join against `embeddings`.
- **Dependency ecosystem gotchas.** See `CLAUDE.md` for the inherited pins (Toponymy from git main, `evoc==0.1.3`, LEACE-with-one-hot requirements, re-normalization after LEACE for cosine-clustering downstream).

## Erasure: scope and limits

The `metadata=` parameter on `fit()` is documented as "concept erasure," but in practice it has two independent effects and users should know about both because they fail in different ways.

**Lever 1: embedding-side LEACE.** Each metadata column is one-hot encoded and LEACE projects out the linear subspace of the embeddings that predicts those one-hots. Toponymy then clusters the erased embeddings, so clusters are less cleanly organized along the erased axis. Two caveats:
- LEACE is linear only. If the erased axis has structure that is nonlinear in the embedding (common for semantic dimensions like sentiment), that structure survives.
- Even if erasure were perfect at the embedding level, it wouldn't touch the text. Every LLM step downstream of clustering (cluster naming, schema synthesis, per-document labeling) reads the original documents.

**Lever 2: synthesis-prompt steering.** When metadata is passed, the synthesis prompt lists each erased column with its inferred type (ordinal if numeric or `pd.CategoricalDtype(ordered=True)`, categorical otherwise) and up to 8 example values, and instructs the LLM to propose an axis orthogonal to them. Caveats:
- Only steers the synthesis step. The per-document labeling LLM still reads the text and can still classify docs into sentiment-like or subject-like values if the cluster the LLM named really is organized that way.
- Success depends on whether the LLM can infer *what to avoid* from name + dtype + sample values. Discrete and text-reflected categoricals (e.g., product category) steer well; broad semantic dimensions (e.g., sentiment correlated with a 1-5 rating) steer weakly because the LLM reasonably interprets "rating accounted for" as "don't re-propose a 1-5 scale" rather than "avoid all sentiment/valence axes."

**Practical consequences.**
- Erasure works best on metadata that is both linearly predictable in the embedding and describable at the LLM level via name + type + values. Discrete categorical metadata that shows up in the text is the sweet spot.
- Erasure works least well on semantically loud axes the LLM can find in the text regardless. If you need to steer away from such an axis, user-provided column descriptions would help and are tracked as a future enhancement.
- Both levers are additive. There is no toggle; passing `metadata=` turns both on.

Measured reductions (baseline vs with both levers, n=500 per run, 2 seeds averaged):

| erased column | type | max-NMI before | max-NMI after | reduction |
|---|---|---|---|---|
| arxiv `primary_category` | categorical | 0.39 | 0.24 | -39% |
| amazon `product_category` | categorical | 0.58 | 0.15 | -74% |
| amazon `rating` | ordinal (1-5) | 0.42 | 0.38 | -8% |

## Parking lot (post-0.1)

Each item is tracked as a GitHub issue with the `parking-lot` label; click through for full design notes. Filter: [is:issue label:parking-lot](https://github.com/stevenfazzio/typologist/issues?q=is%3Aissue+label%3Aparking-lot).

- **Async support (0.2+).** Threadpool covers most labeling-throughput; full async rework with `AsyncLLMWrapper` is the 0.2 story. (see #15)
- **Ordinal facets (0.2+).** Adds `"ordinal"` kind, value-ordering invariant, and `pd.Categorical(..., ordered=True)` in `labels_df_`. (see #16)
- **Stability helper `stability_check(docs, embeddings, n_seeds=5)` (0.2).** Reports cross-seed agreement on facets, values, and per-doc labels. (see #17)
- **Functional `discover()` (needs separate design pass).** Open question: rich result object that surfaces all four fitted artifacts. (see #19)
- **`predefined_facets=` one-call sugar.** Already composable via `apply_schema` + `metadata=`; sugar later if clunky in practice. (see #20)
- **Extension / incremental `n_facets` (0.2+).** Resume a previous fit via `prior_schema=` / `prior_labels_df=` / `prior_residualized=`. (see #21)
- **Tier-1 map output (0.2).** Surface 2D coords and hierarchical region labels for direct DataMapPlot use; adds `umap-learn` as optional extra. (see #22)
