# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Typologist is a Python FOSS tool that takes a corpus of documents (plus embeddings, and optionally existing structured metadata to concept-erase) and returns a schema of discovered categorical facets with per-document labels. Built on [Toponymy](https://github.com/TutteInstitute/toponymy) and [EVoC](https://github.com/TutteInstitute/evoc).

Audience: data scientists, ML engineers, taxonomists/ontologists/archivists/librarians, social scientists, marketing/product analysts, and users of McInnes/Tutte tools.

## Project state (2026-04-22)

Alpha. 0.0.1 is live on PyPI; the public API is still subject to change as we iterate (see `docs/design.md` for the current contract). Successor to an earlier research harness (see memory for design rationale and empirical findings that inform current defaults).

## Key decisions locked

- License: BSD-3-Clause (matches UMAP, HDBSCAN, Toponymy, EVoC, DataMapPlot).
- Python floor: 3.11+.
- Alpha / 0.x branding for 6-12 months. "API may change."
- Name casing: `typologist` in code, imports, CLI, PyPI; `Typologist` in prose, class names, README titles.
- Support commitments: respond to issues within a week, tagged PyPI releases, semver discipline, maintained CHANGELOG.

## Terminology

@docs/glossary.md

## Inherited gotchas from the dependency ecosystem

These apply to any code built on Toponymy + EVoC + LEACE, carried over from prior work on the predecessor harness:

**LEACE fit with int labels.** `LeaceEraser.fit(X, z_int)` accepts integer class labels but treats them as a single continuous feature, so only one axis of variance gets projected out. For multi-class erasure Z must be one-hot.

**Re-normalization after LEACE.** Many embedders (including Cohere) return L2-normalized unit vectors. LEACE's affine projection pushes points slightly off the unit sphere, which matters for cosine-similarity clustering downstream. Renormalize after LEACE unless there's a specific reason not to.

**Toponymy pinned to `>=0.5.0,<0.6.0`.** PyPI 0.5.0 works with `evoc==0.1.3`; earlier PyPI versions (0.4.0) had API drifts. Don't bump to 0.6.x without re-verifying evoc compatibility, since Toponymy's `EVoCClusterer` adapter ties the two together tightly.

**EVoC pinned to `==0.1.3`.** Toponymy's `EVoCClusterer` adapter passes `min_num_clusters` and `next_cluster_size_quantile` kwargs that newer evoc (0.3.x) removed. If either pin changes, expect breakage.

**Toponymy hard-imports `tokenizers` and `transformers` at module load** via `toponymy/llm_wrappers.py`; neither is in Toponymy's declared deps, so `uv sync` won't pull them in transitively. Both are listed as direct deps in our `pyproject.toml`. Matplotlib and `anywidget` are only needed if you import `toponymy.plotting`, which we don't.

**EVoC has no `random_state`.** Clustering is non-deterministic within a session. Wire a `random_state` at the Typologist level where we can (LEACE fit, any sampling, NumPy RNG) and document EVoC as the residual source of non-determinism.

## Required env vars (when code is added)

- `ANTHROPIC_API_KEY`: for cluster naming and axis synthesis
- `CO_API_KEY`: for Cohere embeddings. Note: NOT the more common `COHERE_API_KEY`. The Cohere Python SDK v5+ uses `CO_API_KEY`; some older docs still say `COHERE_API_KEY` and that silently fails to authenticate.

## Commands

Sync dependencies: `uv sync`

Run tests: `uv run pytest`

Lint and format check: `uv run ruff check src tests && uv run ruff format --check src tests`

Apply format: `uv run ruff format src tests`

## Before committing

Check whether `CHANGELOG.md` needs an entry under `[Unreleased]`. The bar: would someone upgrading from the previous version benefit from knowing? If yes, add to the appropriate Keep-a-Changelog section (Added / Changed / Deprecated / Removed / Fixed / Security); breaking changes get a `**Breaking:**` prefix inside Changed. Skip for internal refactors, dev tooling, CI, test-only, or pure doc tidying.

## Reading order for a fresh agent

1. This file
2. `README.md` for user-facing framing
3. The memory directory for locked decisions, API design heuristics, empirical findings, and collaboration preferences
4. `docs/design.md` for the 0.1 public-API contract (the implementation target)
5. `pyproject.toml` for build/deps config
6. Once there's code: `src/typologist/__init__.py` to see the public surface
