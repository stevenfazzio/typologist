# Typologist

Extract a categorical schema from a corpus of documents.

## Status

**Pre-alpha.** The public API will change as we figure things out. See [`docs/design.md`](docs/design.md) for the current contract.

## What it does

You give it documents and their embeddings. It gives you back a handful of categorical facets and a per-document label for each. For example, run it on ~1000 arxiv ML papers and you'll typically get three facets (say `contribution_type`, `primary_data_modality`, and `application_domain`), each with 6-10 values, plus a DataFrame of per-doc labels you can join straight back onto the original corpus.

What makes Typologist's facets mutually orthogonal rather than redundant is concept erasure. After each facet is discovered, its per-document labels are erased from the embeddings via [LEACE](https://github.com/EleutherAI/concept-erasure), and the next facet is discovered against the residual. If you pass in known metadata (existing category tags, publication year, source) it's erased the same way up front, so discovery starts from embeddings orthogonal to what you already have.

## Install

Requires Python 3.11+.

```bash
uv add typologist
# or: pip install typologist
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

For a runnable end-to-end example against real data (500 Amazon reviews, Cohere embeddings, an interactive DataMapPlot of the result), see [`examples/amazon_reviews.py`](examples/amazon_reviews.py).

## Discovery with metadata erasure

Pass a `metadata` DataFrame to erase known axes before discovery starts, so the facets Typologist finds are orthogonal to what you already had. A fuller example:

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

### How much does erasure actually erase?

Erasure is partial, not absolute. Passing `metadata=` activates two independent effects: LEACE removes the linearly-predictable structure from the embeddings (so Toponymy's clustering is less aligned with the erased axis), and the synthesis prompt tells the LLM "these axes are accounted for, find something else." Both help, but neither reaches the per-document labeling LLM, which reads the original text. So erasure is most effective on discrete, text-reflected metadata (product category, subject area) and least effective on broad semantic axes the LLM can find in the text regardless of the metadata signal (sentiment correlated with a 1-5 rating). See [`docs/design.md`](docs/design.md#erasure-scope-and-limits) for the full two-lever model and measured reductions.

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
