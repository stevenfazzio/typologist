# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Typologist is a Python FOSS tool for *schema induction* over document corpora: given a corpus and its embeddings, it returns a multi-facet categorical schema (a list of named axes, each with a vocabulary of values and a definition). Built on [Toponymy](https://github.com/TutteInstitute/toponymy) and [EVoC](https://github.com/TutteInstitute/evoc).

Schema induction is the automated form of faceted classification, the library-science approach to multi-axis categorical description (Ranganathan's colon classification, 1933). The project is positioned for ML/data-science audiences and for the taxonomy/ontology/archives/library-science world alike.

The pipeline is a single pass: Toponymy clusters and names the corpus; a schema-synthesis LLM proposes `n_facets` mutually orthogonal categorical axes from the resulting cluster hierarchy in one call. Per-document labels are obtained separately via `apply_schema(t.schema_, documents, llm=...)`, so discovery and labeling are budgeted independently.

Audience: data scientists, ML engineers, taxonomists/ontologists/archivists/librarians, social scientists, marketing/product analysts, and users of McInnes/Tutte tools.

The 0.1 public-API contract lives in `docs/design.md`.

## Project state (2026-05-11)

Alpha. 0.1.0 on PyPI. The public API is still subject to change as we iterate.

0.1.0 is a pivot release that removed the iterative LEACE-residualization workflow in favor of single-pass synthesis (`fit_single_pass` semantics promoted to `fit`). See `CHANGELOG.md` and `experiments/artifacts/` for the ablation evidence that motivated the change.

## Key decisions locked

- License: BSD-3-Clause (matches UMAP, HDBSCAN, Toponymy, EVoC, DataMapPlot).
- Python floor: 3.11+.
- Alpha / 0.x branding for 6-12 months. "API may change."
- Name casing: `typologist` in code, imports, CLI, PyPI; `Typologist` in prose, class names, README titles.
- Support commitments: respond to issues within a week, tagged PyPI releases, semver discipline, maintained CHANGELOG.

## Terminology

@docs/glossary.md

## Inherited gotchas from the dependency ecosystem

**Toponymy pinned to `>=0.5.0,<0.6.0`.** PyPI 0.5.0 works with `evoc==0.1.3`; earlier PyPI versions (0.4.0) had API drifts. Don't bump to 0.6.x without re-verifying evoc compatibility, since Toponymy's `EVoCClusterer` adapter ties the two together tightly.

**EVoC pinned to `==0.1.3`.** Toponymy's `EVoCClusterer` adapter passes `min_num_clusters` and `next_cluster_size_quantile` kwargs that newer evoc (0.3.x) removed. If either pin changes, expect breakage.

**Toponymy hard-imports `tokenizers`, `transformers`, and `jinja2` at module load** via its templates and llm-wrapper modules. None are in Toponymy's declared deps, so `uv sync` won't pull them in transitively. All three are listed as direct deps in our `pyproject.toml`. Matplotlib and `anywidget` are only needed if you import `toponymy.plotting`, which we don't.

**EVoC has no `random_state`.** Clustering is non-deterministic within a session. Wire a `random_state` at the Typologist level where we can (any sampling, NumPy RNG) and document EVoC as the residual source of non-determinism.

## Provider API keys

None are required by Typologist itself; each is required only if you use the corresponding provider.

- `ANTHROPIC_API_KEY`: for `AnthropicLLM`.
- `OPENAI_API_KEY`: for `OpenAILLM`.

## Commands

Sync dependencies: `uv sync`

Run tests: `uv run pytest`

Lint and format check: `uv run ruff check src tests && uv run ruff format --check src tests`

Apply format: `uv run ruff format src tests`

## Before committing

Check whether `CHANGELOG.md` needs an entry under `[Unreleased]`. The bar: would someone upgrading from the previous version benefit from knowing? If yes, add to the appropriate Keep-a-Changelog section (Added / Changed / Deprecated / Removed / Fixed / Security); breaking changes get a `**Breaking:**` prefix inside Changed. Skip for internal refactors, dev tooling, CI, test-only, or pure doc tidying.
