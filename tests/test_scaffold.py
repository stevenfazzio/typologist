import numpy as np
import pytest

from typologist import LLM, AnthropicLLM, LLMOutputError, OpenAILLM, Typologist, apply_schema


class _DummyEmbedder:
    def encode(self, texts):
        return np.zeros((len(texts), 8))


def test_typologist_constructor_stores_kwargs():
    naming = lambda p: "n"  # noqa: E731
    schema = lambda p: "s"  # noqa: E731
    labeling = lambda p: "l"  # noqa: E731

    t = Typologist(
        n_facets=3,
        topic_embedder=_DummyEmbedder(),
        naming_llm=naming,
        schema_llm=schema,
        labeling_llm=labeling,
    )
    assert t.n_facets == 3
    assert t.object_description == "objects"
    assert t.corpus_description == "collection of objects"
    assert t.naming_llm is naming
    assert t.schema_llm is schema
    assert t.labeling_llm is labeling
    assert t.random_state is None
    assert t.noise_label == "Unlabelled"
    assert t.verbose is False
    assert t.max_concurrency == 10


def test_typologist_requires_llm_kwargs():
    with pytest.raises(TypeError, match="missing.*required keyword.*argument"):
        Typologist(n_facets=3, topic_embedder=_DummyEmbedder())


def test_public_surface_exports():
    assert callable(Typologist)
    assert callable(apply_schema)
    assert isinstance(LLM, type)
    assert issubclass(AnthropicLLM, LLM)
    assert issubclass(OpenAILLM, LLM)
    assert issubclass(LLMOutputError, RuntimeError)
