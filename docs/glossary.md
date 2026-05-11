# Glossary

This document locks the project's terminology. It governs code identifiers, variable names, docstrings, prompt text, and user-facing prose. CLAUDE.md imports this file via `@docs/glossary.md`, so the canonical terms below are always in context when Claude is assisting on this repository.

The core shape: a *corpus* of *documents* (plus their *embeddings*) goes in to `Typologist.fit()`; a *schema* comes out. A schema is a list of *facets*. Each facet has *values*. Separately, `apply_schema(schema, documents, llm=...)` returns per-document *labels* on each facet (the value each doc was assigned).

## Canonical terms

### Input side

- **corpus**: the full collection of input documents.
- **document**: one input string. Prefer "document" over "doc" or "item" in public API and docs; "doc" is acceptable in internal variable names for brevity.
- **embedding**: a vector representation of a document. Always positionally row-aligned with the documents; never index-joined.

### Output side

- **schema**: the full discovered output, an ordered list of facets. Stored on the fitted `Typologist` as `schema_`. JSON-serializable.
- **facet**: one categorical axis in the schema. Has a name, a kind, a vocabulary of values, and a definition. A discovered facet is one entry of `schema_`.
- **kind**: the shape of a facet. In 0.1, always `categorical`; `ordinal` is planned for 0.2+.
- **value**: one element of a facet's vocabulary, for example `highly_positive`. Distinct from a label: a value is a vocabulary entry, a label is a specific document's assignment.
- **label**: the value a document has been assigned on a specific facet. Returned by `apply_schema()` as a DataFrame whose `iloc[i, j]` is document `i`'s label on facet `j`.
- **noise label**: the sentinel string (default `"Unlabelled"`) written by `apply_schema` when a document couldn't be classified. British spelling matches Toponymy and DataMapPlot.

## Avoid, and what to use instead

- **field** → facet. "Field" is SQL/DataFrame vocabulary. Reserve it for pandas column names, not our facets.
- **category** / **categorical** as a noun for a value → value. "Categorical" is fine as an adjective describing a facet's kind; avoid it as a count noun ("the categories of this facet"). Pandas' `pd.Categorical` is a separate concept (see collisions below).
- **dimension** / **axis** in public API, code, and user-facing prose → facet. LLM prompts are the deliberate exception (see "Prompt vocabulary" below).
- **topic** → facet. Toponymy emits "topic names" for clusters, which are inputs to facet synthesis, not facets themselves. The parameter `topic_embedder` is named for Toponymy's use of the embedder, not because we adopt the word for our own outputs.
- **type** for a facet's kind → kind. "Type" is too loaded in Python and pandas; reserve it for its usual senses. This applies to prose, dict keys, and variable names.

## Prompt vocabulary: a deliberate exception

LLM prompts (in `src/typologist/_prompts.py` and inside stored `labeling_prompt_template` strings) use "axis", "dimension", and "orthogonal" as discovery language. A user reading a stored template will see phrases like `along the "review_sentiment" dimension`. This is intentional: those words read naturally to the model as discovery instructions, and "facet" is house vocabulary the LLM doesn't benefit from being taught.

Implication for future prompt edits: keep prompt text in discovery language; translate back to "facet" anywhere that text leaves the prompt (docstrings, error messages, public API names, docs).

## Collisions we live with

- **DataMapPlot `label_layers`.** DataMapPlot's `*label_layers` parameter means text that renders over regions of the 2D map. It is not the same thing as our labels DataFrame. Our `apply_schema` output feeds DataMapPlot's `colormaps=` parameter (per-point categorical coloring), not `label_layers`. When writing example or integration code near DataMapPlot, qualify ("Typologist labels", "region labels") if surrounding context doesn't make the meaning unambiguous.
- **`pd.Categorical`.** The pandas dtype used for columns of the DataFrame returned by `apply_schema`. Unrelated to our `categorical` facet kind. Say `pd.Categorical` or "Categorical dtype" when referring to the pandas concept; reserve bare "categorical" for the facet kind.
