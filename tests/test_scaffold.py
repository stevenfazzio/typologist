import numpy as np
import pandas as pd
import pytest

from typologist import Typologist, apply_schema


class _DummyEmbedder:
    def encode(self, texts):
        return np.zeros((len(texts), 8))


def test_typologist_defaults():
    t = Typologist(n_facets=3, topic_embedder=_DummyEmbedder())
    assert t.n_facets == 3
    assert t.object_description == "objects"
    assert t.corpus_description == "collection of objects"
    assert t.naming_llm == "claude-haiku-4-5"
    assert t.schema_llm == "claude-opus-4-7"
    assert t.labeling_llm == "claude-haiku-4-5"
    assert t.random_state is None
    assert t.noise_label == "Unlabelled"
    assert t.verbose is False


def test_typologist_fit_not_implemented():
    t = Typologist(n_facets=3, topic_embedder=_DummyEmbedder())
    with pytest.raises(NotImplementedError):
        t.fit(["doc a", "doc b"], np.zeros((2, 8)))


def test_apply_schema_not_implemented():
    with pytest.raises(NotImplementedError):
        apply_schema([], [])


def test_documents_series_typing_accepted_at_signature_level():
    t = Typologist(n_facets=1, topic_embedder=_DummyEmbedder())
    docs = pd.Series(["a", "b"], index=[10, 20])
    with pytest.raises(NotImplementedError):
        t.fit(docs, np.zeros((2, 8)))
