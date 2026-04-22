from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from typologist._llm import _resolve_llm
from typologist._pipeline import _classify_docs


def apply_schema(
    schema: list[dict] | dict,
    documents: list[str] | pd.Series,
    llm: str | Callable[..., str] | None = None,
    noise_label: str = "Unlabelled",
    max_concurrency: int = 10,
    verbose: bool = False,
) -> pd.DataFrame:
    """Apply a previously-discovered schema to new documents.

    Each facet uses its stored ``labeling_model`` by default; pass ``llm`` to
    override every facet's model. A facet whose ``labeling_model`` is ``None``
    (created with a callable ``labeling_llm``) requires ``llm`` to be passed.

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

    override_llm = _resolve_llm(llm) if llm is not None else None

    label_series: list[pd.Series] = []
    for facet in facets:
        if override_llm is not None:
            facet_llm = override_llm
        elif facet.get("labeling_model") is not None:
            facet_llm = _resolve_llm(facet["labeling_model"])
        else:
            raise RuntimeError(
                f"Facet {facet.get('name', '<unnamed>')!r} has labeling_model=None "
                "(created with a callable labeling_llm). Pass `llm=` to apply_schema."
            )
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
