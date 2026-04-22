"""Cross-run summary over all experiments in experiments/artifacts/runs/.

For each run, compute max-NMI between any discovered facet and each curator
column. Produces a wide table where you can read off:

- Rediscovery signal (no_erase rows): high max-NMI = curator schema was
  rediscovered.
- Erasure effectiveness (erase rows): low max-NMI on the erased column(s) =
  LEACE successfully stripped that structure from the embeddings.
- Seed stability: seed 0 vs seed 1 rows should be close if the pipeline is
  stable under random-state variation.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import normalized_mutual_info_score

ARTIFACTS_ROOT = Path(__file__).parent / "artifacts"
CACHE_ROOT = ARTIFACTS_ROOT / "cache"
RUNS_ROOT = ARTIFACTS_ROOT / "runs"


def _load_run(run_dir: Path) -> dict:
    with open(run_dir / "config.json") as f:
        config = json.load(f)
    with open(run_dir / "schema.json") as f:
        schema = json.load(f)
    labels_df = pd.read_parquet(run_dir / "labels_df.parquet")
    return {"config": config, "schema": schema, "labels_df": labels_df, "dir": run_dir}


def _load_curator(corpus_name: str, seed: int) -> pd.DataFrame:
    return pd.read_parquet(CACHE_ROOT / f"{corpus_name}_seed{seed}" / "curator_labels.parquet")


def _max_nmi(labels_df: pd.DataFrame, curator_col: pd.Series) -> tuple[float, str]:
    best_nmi = -1.0
    best_facet = ""
    curator_str = curator_col.astype(str).to_numpy()
    for facet in labels_df.columns:
        facet_str = labels_df[facet].astype(str).to_numpy()
        score = float(normalized_mutual_info_score(curator_str, facet_str))
        if score > best_nmi:
            best_nmi = score
            best_facet = facet
    return best_nmi, best_facet


def main() -> None:
    rows = []
    for run_dir in sorted(RUNS_ROOT.iterdir()):
        if not run_dir.is_dir():
            continue
        run = _load_run(run_dir)
        config = run["config"]
        curator = _load_curator(config["corpus"], config["seed"])
        facet_names = [f["name"] for f in run["schema"]]
        row = {
            "corpus": config["corpus"],
            "mode": config["mode"],
            "seed": config["seed"],
            "timestamp": config["timestamp_utc"],
            "facets": ", ".join(facet_names),
        }
        for curator_col in curator.columns:
            nmi, best_facet = _max_nmi(
                run["labels_df"].reset_index(drop=True),
                curator[curator_col].reset_index(drop=True),
            )
            row[f"max_nmi_{curator_col}"] = round(nmi, 3)
            row[f"best_facet_for_{curator_col}"] = best_facet
        rows.append(row)

    df = (
        pd.DataFrame(rows)
        .sort_values(["corpus", "mode", "seed", "timestamp"])
        .reset_index(drop=True)
    )
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
