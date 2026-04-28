from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from typologist._pipeline import _classify_docs
from typologist.llm import LLM, _resolve_llm


def apply_schema(
    schema: list[dict] | dict,
    documents: list[str] | pd.Series,
    llm: LLM | Callable[..., str] | None = None,
    noise_label: str = "Unlabelled",
    max_concurrency: int = 10,
    verbose: bool = False,
) -> pd.DataFrame:
    """Apply a previously-discovered schema to new documents.

    The schema's stored ``labeling_model`` is provenance only (a
    ``"provider:model"`` string identifying what produced the schema), so
    ``llm`` is required to actually label new documents. Pass an
    ``AnthropicLLM(...)``, ``OpenAILLM(...)``, custom ``LLM`` subclass, or a
    ``Callable[[str], str]``.

    Per-doc LLM calls are dispatched concurrently via a threadpool with up to
    ``max_concurrency`` workers. Pass ``max_concurrency=1`` for serial.

    See ``docs/design.md`` for the full 0.1 contract.
    """
    if isinstance(schema, dict):
        facets = [schema]
    else:
        facets = list(schema)

    if isinstance(documents, pd.Series):
        docs_series = documents
    elif isinstance(documents, list):
        docs_series = pd.Series(documents)
    else:
        raise TypeError(
            f"documents must be a list or pandas Series, got {type(documents).__name__}"
        )

    if not all(isinstance(d, str) for d in docs_series):
        raise TypeError("all entries in documents must be str")

    if llm is None:
        raise TypeError(
            "apply_schema requires `llm=`. The schema's stored labeling_model is "
            "provenance only and is no longer auto-resolved. Pass typologist."
            "AnthropicLLM(...), typologist.OpenAILLM(...), a custom LLM subclass, "
            "or a Callable[[str], str]."
        )
    facet_llm = _resolve_llm(llm)

    label_series: list[pd.Series] = []
    for facet in facets:
        label_series.append(
            _classify_docs(
                facet=facet,
                documents=docs_series,
                labeling_llm=facet_llm,
                noise_label=noise_label,
                max_concurrency=max_concurrency,
                verbose=verbose,
            )
        )

    if label_series:
        return pd.concat(label_series, axis=1)
    return pd.DataFrame(index=docs_series.index)
