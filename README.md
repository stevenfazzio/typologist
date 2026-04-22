# Typologist

Extract a categorical schema from a corpus of documents. Built on [Toponymy](https://github.com/TutteInstitute/toponymy) and [EVoC](https://github.com/TutteInstitute/evoc).

## Status

**Pre-alpha.** The public API is subject to change without notice. See [`docs/design.md`](docs/design.md) for the current contract.

## What it does

Given a corpus of text documents and their embeddings, Typologist discovers a set of orthogonal categorical facets (e.g., `contribution_type`, `data_modality`, `application_domain`) and labels each document along each facet. Optionally, it concept-erases user-supplied metadata via [LEACE](https://github.com/EleutherAI/concept-erasure), so the discovered schema is orthogonal to structure the user already knows about.

A typical run on ~1000 arxiv ML papers produces three facets with 6-10 values each, plus a DataFrame of per-doc labels that joins back to the original corpus on index.

## Install

Requires Python 3.11+.

```bash
uv add git+https://github.com/stevenfazzio/typologist.git
# or: pip install git+https://github.com/stevenfazzio/typologist.git
```

Also needed:

- `ANTHROPIC_API_KEY` in the environment (or supply your own LLM callable for each of the three roles).
- A sentence-embedding model that Toponymy can use to embed keyphrases and topic names. `sentence-transformers` with MiniLM is a common and cheap pick:

  ```bash
  uv pip install sentence-transformers
  ```

## Quick start

```python
import numpy as np
from sentence_transformers import SentenceTransformer
from typologist import Typologist

documents = [...]              # list[str], one per document
embeddings = np.array(...)     # shape (n_docs, d), float

t = Typologist(
    n_facets=3,
    topic_embedder=SentenceTransformer("all-MiniLM-L6-v2"),
).fit(documents, embeddings)

print(t.schema_)               # list[dict]: discovered facet definitions
print(t.labels_df_)            # (n_docs, n_facets) DataFrame of categorical labels
```

## Discovery with metadata erasure

If your documents come with known structured metadata (source, category, year), you probably don't want Typologist to "discover" those axes — you want the facets it finds to be **orthogonal** to what you already know. Pass a `metadata` DataFrame and Typologist will concept-erase those axes before running discovery.

```python
import pandas as pd
from sentence_transformers import SentenceTransformer
from typologist import Typologist

df = pd.read_parquet("arxiv_sample.parquet")   # title, abstract, primary_category, ...
embeddings = np.load("arxiv_cohere_v4.npy")    # shape (len(df), 1536)
docs = (df["title"] + "\n\n" + df["abstract"]).rename("document")

t = Typologist(
    n_facets=3,
    topic_embedder=SentenceTransformer("all-MiniLM-L6-v2"),
    object_description="scientific paper",
    corpus_description="machine-learning arxiv papers",
    random_state=0,
    verbose=True,
).fit(
    docs,
    embeddings,
    metadata=df[["primary_category"]],
)

# Join labels back onto the original DataFrame
df_labeled = df.join(t.labels_df_)

# Inspect the schema
for facet in t.schema_:
    print(f"{facet['name']} ({len(facet['values'])} values): {facet['definition']}")
    for value in facet["values"]:
        print(f"  - {value}")

# Cross-tab a discovered facet against held-out metadata
pd.crosstab(df_labeled[t.schema_[0]["name"]], df_labeled["primary_category"])
```

Per-facet diagnostics — cluster counts, label entropy, exemplar documents — are available via `t.facet_diagnostics_`.

## Reusing a discovered schema

Each facet entry includes a stored `labeling_prompt_template` and `labeling_model`, so a schema can be applied to new documents without re-running discovery:

```python
from typologist import apply_schema

new_labels = apply_schema(schema=t.schema_, documents=new_docs)
```

See [`docs/design.md`](docs/design.md) for the full schema entry shape and `apply_schema` contract.

## Performance

Per-document labeling dispatches through a threadpool (`max_concurrency=10` by default). On a 1000-doc, n_facets=3 corpus this typically completes in ~6-8 minutes. Toponymy's cluster naming and the schema-synthesis LLM calls are still serial; fuller async is a 0.2 item.

## Related

Typologist is an independent project with no affiliation to the authors of the libraries it builds on:

- [Toponymy](https://github.com/TutteInstitute/toponymy) — cluster naming and hierarchy
- [EVoC](https://github.com/TutteInstitute/evoc) — hierarchical clustering
- [concept-erasure](https://github.com/EleutherAI/concept-erasure) — LEACE implementation

If you want to visualize Typologist labels over a 2D embedding projection, [DataMapPlot](https://github.com/TutteInstitute/datamapplot) pairs naturally.

## License

BSD-3-Clause. See [LICENSE](LICENSE).
