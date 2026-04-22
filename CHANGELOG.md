# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- Synthesis prompt now describes erased metadata columns to the schema LLM, so `metadata=` on `fit()` steers the LLM at both the embedding level (via LEACE) and at the prompt level. Measured reductions in curator-label rediscovery: amazon `product_category` -74%, arxiv `primary_category` -39%, amazon `rating` -8% (sentiment-like axes are harder to steer from dtype alone). See `docs/design.md` for the two-lever erasure model.

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
