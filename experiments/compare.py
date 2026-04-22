"""Compare discovered schema against curator labels.

Usage:
    uv run python experiments/compare.py <run_dir>

Prints:
- The discovered schema (qualitative read).
- For each facet x curator-column pair, a contingency table plus NMI and
  adjusted Rand index.

For a `no_erase` run we hope to see at least one facet correlated with each
curator column. For an `erase` run we hope the erased columns show *low*
correlation with every facet.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

ARTIFACTS_ROOT = Path(__file__).parent / "artifacts"
CACHE_ROOT = ARTIFACTS_ROOT / "cache"
RUNS_ROOT = ARTIFACTS_ROOT / "runs"


def _load_run(run_dir: Path) -> dict:
    with open(run_dir / "config.json") as f:
        config = json.load(f)
    with open(run_dir / "schema.json") as f:
        schema = json.load(f)
    labels_df = pd.read_parquet(run_dir / "labels_df.parquet")
    return {"config": config, "schema": schema, "labels_df": labels_df}


def _load_cache(corpus_name: str, seed: int) -> pd.DataFrame:
    cache_dir = CACHE_ROOT / f"{corpus_name}_seed{seed}"
    return pd.read_parquet(cache_dir / "curator_labels.parquet")


def _align(labels_df: pd.DataFrame, curator_labels: pd.DataFrame) -> pd.DataFrame:
    """Return a single DataFrame with both label sets, positionally aligned.

    We rely on positional alignment: both were produced from the same sampled
    corpus in the same order.
    """
    if len(labels_df) != len(curator_labels):
        raise ValueError(
            f"length mismatch: labels_df={len(labels_df)}, curator={len(curator_labels)}"
        )
    merged = labels_df.reset_index(drop=True).copy()
    for col in curator_labels.columns:
        merged[f"curator__{col}"] = curator_labels[col].to_numpy()
    return merged


def _contingency(merged: pd.DataFrame, facet_col: str, curator_col: str) -> pd.DataFrame:
    return pd.crosstab(
        merged[f"curator__{curator_col}"],
        merged[facet_col],
        dropna=False,
    )


def _rediscovery_scores(merged: pd.DataFrame, facet_col: str, curator_col: str) -> dict:
    facet = merged[facet_col].astype(str).to_numpy()
    curator = merged[f"curator__{curator_col}"].astype(str).to_numpy()
    return {
        "nmi": float(normalized_mutual_info_score(curator, facet)),
        "ari": float(adjusted_rand_score(curator, facet)),
        "n": int(len(facet)),
    }


def _format_schema(schema: list[dict]) -> str:
    lines = []
    for i, f in enumerate(schema):
        lines.append(f"Facet {i}: {f['name']} ({f['type']})")
        lines.append(f"  definition: {f['definition']}")
        lines.append(f"  values: {f['values']}")
        lines.append("")
    return "\n".join(lines)


def _format_comparison(run_info: dict, curator_labels: pd.DataFrame) -> str:
    config = run_info["config"]
    schema = run_info["schema"]
    labels_df = run_info["labels_df"]
    merged = _align(labels_df, curator_labels)

    out = []
    out.append("=" * 70)
    out.append(f"Run: {config['corpus']} / mode={config['mode']} / seed={config['seed']}")
    out.append(f"n_docs={config['n_docs']} n_facets={config['n_facets']}")
    out.append(f"timestamp_utc={config['timestamp_utc']}")
    out.append("")
    out.append("Discovered schema:")
    out.append("-" * 70)
    out.append(_format_schema(schema))

    facet_cols = [f["name"] for f in schema]
    curator_cols = list(curator_labels.columns)

    out.append("Facet x curator NMI / ARI (higher = more correlated):")
    out.append("-" * 70)
    score_rows = []
    for facet_col in facet_cols:
        for curator_col in curator_cols:
            scores = _rediscovery_scores(merged, facet_col, curator_col)
            score_rows.append({"facet": facet_col, "curator": curator_col, **scores})
    scores_df = pd.DataFrame(score_rows)
    out.append(scores_df.to_string(index=False))
    out.append("")

    out.append("Contingency tables:")
    out.append("-" * 70)
    for facet_col in facet_cols:
        for curator_col in curator_cols:
            table = _contingency(merged, facet_col, curator_col)
            out.append(f"[facet={facet_col}] x [curator={curator_col}]")
            out.append(table.to_string())
            out.append("")

    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()

    run_info = _load_run(args.run_dir)
    config = run_info["config"]
    curator_labels = _load_cache(config["corpus"], config["seed"])
    print(_format_comparison(run_info, curator_labels))


if __name__ == "__main__":
    main()
