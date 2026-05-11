# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-05-11

Pivot release. Ablation evidence (see `experiments/artifacts/ablations/` and `experiments/artifacts/holdout_eval/`) showed that the iterative LEACE-residualization workflow did not deliver the orthogonality value it was sold on: facets 2..N had near-zero geometric alignment with the residualized embeddings they were nominally discovered from, and LEACE's collateral-erasure of correlated signals sometimes *reduced* the geometric grounding of later facets. The schema-level orthogonality the methodology was measured against was an LLM-prompt artifact, not a property of the residualized embedding space. 0.1.0 strips the iterative loop and LEACE in favor of a single-pass workflow.

### Added

- Public `LLM` ABC and provider-specific wrapper classes `AnthropicLLM` and `OpenAILLM` with lazy SDK imports. Subclass `LLM` to support additional providers.
- `use_toponymy: bool = True` constructor parameter on `Typologist`. With `use_toponymy=False`, swaps Toponymy for a minimal single-layer EVoC + per-cluster naming path. Default `True` is the supported path; `False` exists for benchmarking.
- `diagnostics_` attribute on the fitted Typologist: a single dict with `synthesis_prompt`, `cluster_count`, and `hierarchy_depth` from the Toponymy run. Not part of the stable persisted-state contract.

### Changed

- **Breaking:** `Typologist.fit()` is now a single-pass workflow: one Toponymy run + one structured-output `schema_llm` call propose all `n_facets` mutually orthogonal categorical axes. The previous iterative loop (one Toponymy + synthesis + LEACE residualization per facet) is removed.
- **Breaking:** `Typologist.fit()` no longer produces per-document labels. The fitted Typologist exposes `schema_` and `diagnostics_` only. To label documents, call `apply_schema(t.schema_, documents, llm=...)` separately.
- **Breaking:** `naming_llm`, `schema_llm`, and `labeling_llm` are now required keyword-only arguments on `Typologist`. The previous Anthropic string defaults are removed. `labeling_llm` is provenance-only at fit time (stamped onto each facet's `labeling_model` field); `fit()` never calls it.
- **Breaking:** LLM arguments no longer accept model-name strings. Pass an `AnthropicLLM(...)`, `OpenAILLM(...)`, custom `LLM` subclass, or `Callable[[str], str]`.
- **Breaking:** `schema_[i]["labeling_model"]` now stores a `"provider:model"` string (e.g., `"anthropic:claude-haiku-4-5"`) instead of a bare model name, and is provenance only.
- **Breaking:** `apply_schema(...)` requires an `llm=` argument. Auto-resolution from the stored `labeling_model` is removed.
- **Breaking:** `anthropic` is no longer a runtime dependency. Install with `pip install typologist[anthropic]`, `typologist[openai]`, or `typologist[all]`. Bare `pip install typologist` ships only the provider-neutral core.
- **Breaking:** renamed the `schema_[i]["type"]` dict key to `schema_[i]["kind"]`. "Type" is too loaded in Python and pandas; "kind" is the canonical term for a facet's categorical-vs-ordinal shape per `docs/glossary.md`. Saved schemas from 0.0.1 need their `"type"` key renamed to `"kind"` before loading.

### Removed

- **Breaking:** the `metadata=` parameter on `fit()` and the entire pre-erasure machinery. Users who previously erased known nuisance metadata before discovery have no equivalent in 0.1.0; pre-erasure may return in a future minor release if there's clear evidence it earns its keep on its own (separately from the iterative loop, which the ablation evidence showed was the load-bearing problem).
- **Breaking:** the `concept-erasure` package is no longer a runtime dependency. LEACE is no longer used anywhere in the pipeline.
- **Breaking:** `Typologist.labels_df_` attribute. `fit()` no longer labels; call `apply_schema()` to get a labels DataFrame.
- **Breaking:** `Typologist.embeddings_residualized_` attribute. No residualization happens, so there's nothing to expose.
- **Breaking:** `Typologist.facet_diagnostics_` attribute (list of per-facet dicts) is replaced by `Typologist.diagnostics_` (single dict). Single-pass discovery produces one synthesis prompt and one Toponymy run for the whole schema, not one per facet.
- **Breaking:** the `max_concurrency` constructor parameter on `Typologist`. It only governed fit-time labeling, which no longer happens. `apply_schema()` has its own `max_concurrency`.
- **Breaking:** the `concept_erasure` and `inform_synthesis_of_prior_facets` constructor parameters on `Typologist`. These were ablation toggles for the iterative-loop study and have no meaning in the single-pass design.

## [0.0.1] - 2026-04-22

Initial alpha release.

### Added
- `Typologist` class: discovers `n_facets` categorical facets from a corpus of documents and embeddings. Facets are kept mutually orthogonal via concept erasure (LEACE) between facets.
- Optional metadata pre-erasure: pass a `pandas.DataFrame` to `fit()` and Typologist erases those axes before discovery, so the facets end up orthogonal to structure you already have.
- `apply_schema()` helper: apply a previously-discovered schema to new documents using each facet's stored labeling prompt template, without re-running discovery.
- Separate LLM roles (`naming_llm`, `schema_llm`, `labeling_llm`), each accepting either a model-name string (resolved to Anthropic) or a callable.
- Threadpool-concurrent labeling via a `max_concurrency` kwarg (default 10).
- Per-facet diagnostics (`facet_diagnostics_`): synthesis prompt, Toponymy cluster counts, label entropy, exemplar documents per value.

[Unreleased]: https://github.com/stevenfazzio/typologist/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/stevenfazzio/typologist/releases/tag/v0.1.0
[0.0.1]: https://github.com/stevenfazzio/typologist/releases/tag/v0.0.1
