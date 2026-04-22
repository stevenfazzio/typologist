# Typologist

Extract a categorical schema from a corpus of documents. Built on [Toponymy](https://github.com/TutteInstitute/toponymy) and [EVoC](https://github.com/TutteInstitute/evoc).

## Status

**Pre-alpha.** The public API will change as we figure things out. See [`docs/design.md`](docs/design.md) for the current contract.

## What it does

You give it documents and their embeddings. It gives you back a handful of categorical facets and a per-document label for each. For example, run it on ~1000 arxiv ML papers and you'll typically get three facets (say `contribution_type`, `primary_data_modality`, and `application_domain`), each with 6-10 values, plus a DataFrame of per-doc labels you can join straight back onto the original corpus.

If you already have known metadata that you don't want rediscovered (existing category tags, publication year, source, whatever), you pass that in too and Typologist concept-erases it first via [LEACE](https://github.com/EleutherAI/concept-erasure), so the facets it finds are orthogonal to what you already had.

## Install

Requires Python 3.11+.

```bash
uv add git+https://github.com/stevenfazzio/typologist.git
# or: pip install git+https://github.com/stevenfazzio/typologist.git
```

You'll also want:

- `ANTHROPIC_API_KEY` in the environment (or your own LLM callable for each of the three roles; see below).
- A sentence-embedding model that Toponymy can use internally for keyphrases and topic names. `sentence-transformers` with MiniLM is cheap and good enough for most use cases:

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

If your documents come with known metadata (source, category, year), you usually don't want Typologist to rediscover those axes. You want the facets it finds to be *orthogonal* to what you already have. Pass a `metadata` DataFrame and Typologist concept-erases those axes before running discovery.

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

Per-facet diagnostics (cluster counts, label entropy, exemplar documents) live on `t.facet_diagnostics_`.

## Reusing a discovered schema

Every facet entry stores its own `labeling_prompt_template` and `labeling_model`, so you can apply a schema to new documents without re-running discovery:

```python
from typologist import apply_schema

new_labels = apply_schema(schema=t.schema_, documents=new_docs)
```

See [`docs/design.md`](docs/design.md) for the full schema entry shape and `apply_schema` contract.

## Performance

Per-document labeling runs through a threadpool (`max_concurrency=10` by default). On 1000 docs with `n_facets=3` you should see roughly 6-8 minutes end to end. Toponymy's cluster naming and the schema-synthesis LLM calls are still serial; full async is a 0.2 item.

## Related

Typologist is an independent project with no affiliation to the authors of the libraries it builds on:

- [Toponymy](https://github.com/TutteInstitute/toponymy): cluster naming and hierarchy
- [EVoC](https://github.com/TutteInstitute/evoc): hierarchical clustering
- [concept-erasure](https://github.com/EleutherAI/concept-erasure): LEACE implementation

If you want a 2D embedding projection with your Typologist labels on top, [DataMapPlot](https://github.com/TutteInstitute/datamapplot) is a natural match.

## License

BSD-3-Clause. See [LICENSE](LICENSE).
