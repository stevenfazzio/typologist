from __future__ import annotations

from collections.abc import Callable

import pandas as pd


def apply_schema(
    schema: list[dict] | dict,
    documents: list[str] | pd.Series,
    llm: str | Callable[..., str] | None = None,
) -> pd.DataFrame:
    """Apply a previously-discovered schema to new documents.

    See ``docs/design.md`` for the full 0.1 contract.
    """
    raise NotImplementedError("apply_schema is not yet implemented")
