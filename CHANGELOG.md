# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Public `LLM` ABC and provider-specific wrapper classes `AnthropicLLM` and `OpenAILLM` with lazy SDK imports. Subclass `LLM` to support additional providers.
- `use_toponymy: bool = True` constructor parameter on `Typologist`. With `use_toponymy=False`, swaps Toponymy for a homemade naming path (single-layer EVoC clustering, centroid-nearest exemplars, one LLM call per cluster). Default `True` preserves existing behavior.
- `examples/amazon_reviews.py`: end-to-end runnable walkthrough that loads Amazon reviews from HuggingFace, fits Typologist, and renders an interactive DataMapPlot HTML. README's worked example is grounded in this script's actual output.

### Changed
- **Breaking:** `naming_llm`, `schema_llm`, and `labeling_llm` are now required keyword-only arguments on `Typologist`. The previous Anthropic string defaults are removed.
- **Breaking:** LLM arguments no longer accept model-name strings. Pass an `AnthropicLLM(...)`, `OpenAILLM(...)`, custom `LLM` subclass, or `Callable[[str], str]`.
- **Breaking:** `schema_[i]["labeling_model"]` now stores a `"provider:model"` string (e.g., `"anthropic:claude-haiku-4-5"`) instead of a bare model name, and is provenance only.
- **Breaking:** `apply_schema(...)` requires an `llm=` argument. Auto-resolution from the stored `labeling_model` is removed.
- **Breaking:** `anthropic` is no longer a runtime dependency. Install with `pip install typologist[anthropic]`, `typologist[openai]`, or `typologist[all]`. Bare `pip install typologist` ships only the provider-neutral core.
- **Breaking:** renamed the `schema_[i]["type"]` dict key to `schema_[i]["kind"]`. "Type" is too loaded in Python and pandas; "kind" is the canonical term for a facet's categorical-vs-ordinal shape per `docs/glossary.md`. Saved schemas from 0.0.1 need their `"type"` key renamed to `"kind"` before loading.
- Synthesis prompt now describes erased metadata columns to the schema LLM, so `metadata=` on `fit()` steers the LLM at both the embedding level (via LEACE) and at the prompt level. Measured reductions in curator-label rediscovery: amazon `product_category` -74%, arxiv `primary_category` -39%, amazon `rating` -8% (sentiment-like axes are harder to steer from dtype alone). See `docs/design.md` for the two-lever erasure model.
- Clarified what `embeddings_residualized_` actually contains: LEACE removes the entire linear subspace that predicts the erased facet (or metadata column), not just the centroid offset between value groups. Signals linearly correlated with the erased facet or metadata column also get partially removed. See `docs/design.md` (`embeddings_residualized_` section) for the user-facing implication when chaining residualized embeddings into downstream tasks.

## [0.0.1] - 2026-04-22

Initial alpha release.

### Added
- `Typologist` class: discovers `n_facets` categorical facets from a corpus of documents and embeddings. Facets are kept mutually orthogonal via concept erasure (LEACE) between facets.
- Optional metadata pre-erasure: pass a `pandas.DataFrame` to `fit()` and Typologist erases those axes before discovery, so the facets end up orthogonal to structure you already have.
- `apply_schema()` helper: apply a previously-discovered schema to new documents using each facet's stored labeling prompt template, without re-running discovery.
- Separate LLM roles (`naming_llm`, `schema_llm`, `labeling_llm`), each accepting either a model-name string (resolved to Anthropic) or a callable.
- Threadpool-concurrent labeling via a `max_concurrency` kwarg (default 10).
- Per-facet diagnostics (`facet_diagnostics_`): synthesis prompt, Toponymy cluster counts, label entropy, exemplar documents per value.

[Unreleased]: https://github.com/stevenfazzio/typologist/compare/v0.0.1...HEAD
[0.0.1]: https://github.com/stevenfazzio/typologist/releases/tag/v0.0.1
