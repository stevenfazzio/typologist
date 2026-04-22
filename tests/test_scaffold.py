import numpy as np

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
    assert t.max_concurrency == 10


def test_public_surface_exports():
    assert callable(Typologist)
    assert callable(apply_schema)
