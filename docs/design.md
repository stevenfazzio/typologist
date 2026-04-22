# Typologist 0.1 design

This document is the public-API contract the 0.1 implementation works against. It is intentionally narrow: what ships in 0.1, what the types and shapes are, and what is deliberately parked for later.

## Scope

**In scope for 0.1.**

- A single `Typologist` class (sklearn-style, `fit`-then-attributes)
- A top-level `apply_schema` helper for labeling new documents with a previously-discovered schema
- LEACE pre-erasure of user-supplied known metadata
- Between-facet LEACE residualization against per-doc classifications under synthesized schema fields (not cluster labels; matches predecessor evidence)
- Synchronous execution only

**Out of scope for 0.1.** See "Parking lot" at the end.

## Public surface

```python
from typologist import Typologist, apply_schema

t = Typologist(
    n_facets=3,
    topic_embedder=...,                         # required, no default
    object_description="objects",
    corpus_description="collection of objects",
    naming_llm="claude-haiku-4-5",
    schema_llm="claude-opus-4-7",
    labeling_llm="claude-haiku-4-5",
    random_state=None,
    noise_label="Unlabelled",
    verbose=False,
).fit(documents, embeddings, metadata=None)

t.schema_                    # list[dict]
t.labels_df_                 # pd.DataFrame, shape (n_docs, n_facets)
t.embeddings_residualized_   # np.ndarray, shape (n_docs, dim)
t.facet_diagnostics_         # list[dict]

labels_df = apply_schema(schema, documents, llm=None)
```

## Constructor parameters

| param | type | default | notes |
|---|---|---|---|
| `n_facets` | `int` | required | number of categorical facets to discover; no `"auto"` in 0.1 |
| `topic_embedder` | object implementing `TextEmbedderProtocol` | required | no default; matches Toponymy's stance. Intended for Toponymy's internal keyphrase/topic-name embedding. `sentence_transformers.SentenceTransformer("all-MiniLM-L6-v2")` is a recommended baseline (see predecessor evidence) |
| `object_description` | `str` | `"objects"` | describes what each document is. Passed through to Toponymy; also rendered into our schema-synthesis and labeling prompts |
| `corpus_description` | `str` | `"collection of objects"` | describes the collection as a whole. Same passthrough behavior |
| `naming_llm` | `str \| Callable` | `"claude-haiku-4-5"` | names Toponymy's clusters. Called O(n_clusters x n_layers x n_facets) times |
| `schema_llm` | `str \| Callable` | `"claude-opus-4-7"` | synthesizes a schema field from cluster names. Called O(n_facets) times; quality-dominant step |
| `labeling_llm` | `str \| Callable` | `"claude-haiku-4-5"` | classifies each document into a facet's value vocabulary. Called O(n_docs x n_facets) times |
| `random_state` | `int \| None` | `None` | threaded into LEACE fit, sampling, NumPy RNG. EVoC has no `random_state` and is the residual source of non-determinism; this is documented, not fixed |
| `noise_label` | `str` | `"Unlabelled"` | sentinel string for docs that couldn't be classified into any value. British spelling matches Toponymy and DataMapPlot |
| `verbose` | `bool` | `False` | cascades to Toponymy and other noisy subcomponents |

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
    "type": "categorical" | "ordinal",
    "values": list[str],              # value vocabulary for this facet
    "definition": str,                # human-readable semantic definition
    "labeling_prompt_template": str,  # f-string template containing "{document}"
    "labeling_model": str | None,     # None if labeling_llm was a callable
}
```

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

### `facet_diagnostics_: list[dict]`

One entry per facet, mirroring `schema_` order. Contains provenance and per-facet signals:

- synthesis prompts used (for the `schema_llm` call)
- cluster count and cluster hierarchy depth at the time the facet was proposed
- any entropy or quality signals we compute
- exemplar document ids per value (if cheap to produce)

Exact fields may evolve across minor versions; `facet_diagnostics_` is **not** part of the stable persisted-state contract.

## `apply_schema(schema, documents, llm=None) -> pd.DataFrame`

Applies an existing schema to new documents without running discovery.

- `schema`: a `list[dict]` in the `schema_` format, or a single facet dict.
- `documents`: `list[str]` or `pd.Series[str]`. Index-preservation behavior matches `fit`.
- `llm`: `str | Callable | None`. If `None`, each facet uses its own `labeling_model`. If `str` or `Callable`, that overrides every facet's stored model.
- Raises a clear error if any facet has `labeling_model=None` and no `llm=` is passed.

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

## Parking lot (post-0.1)

Each item has a one-line "why deferred" note.

- **Async support (0.2+).** Real throughput concern at typical corpus sizes (1000+ docs, multiple facets, iteration). Deferred to keep the isinstance-ladder pattern Toponymy fell into out of 0.1; revisit with a cleaner abstraction.
- **Stability helper `stability_check(docs, embeddings, n_seeds=5)` (0.2).** Predecessor evidence shows this is cheap and produces robust signal; deferred only to minimize 0.1 surface.
- **`anthropic_llm(model)` convenience factory (0.1.x).** Collapses the 9-line `make_tracked_llm` pattern into one line; easy addition once we see real usage.
- **Functional `discover()` (needs separate design pass).** Thin wrapper over the class didn't feel worth it given our multi-artifact output. If added later, the open design question is what a rich result object looks like so users don't miss `schema_` / `labels_df_` / `embeddings_residualized_` / `facet_diagnostics_`.
- **`predefined_facets=` one-call sugar.** Already composable in 0.1 via `apply_schema` + `metadata=` (the user labels with a predefined schema, then passes those labels as `metadata` to erase before discovering more). Syntactic sugar later if the composition proves clunky in practice.
- **Extension / incremental `n_facets` (0.2+).** Running `Typologist(n_facets=2).fit(..., prior_schema=t1.schema_, prior_labels_df=t1.labels_df_, prior_residualized=t1.embeddings_residualized_)`. 0.1 persisted-state contract (above) already makes this addable without breaking changes.
- **Tier-1 map output (0.2).** Optional `produce_map_output=True` that surfaces `coords_2d_` (via UMAP), `topic_names_`, and `topic_name_vectors_` (already computed internally) so users go straight to DataMapPlot without running UMAP and Toponymy separately. Adds `umap-learn` as an optional extra (`typologist[viz]`); reuses the high-dim Toponymy output rather than re-running on 2D.
