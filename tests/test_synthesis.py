from __future__ import annotations

import pytest

from typologist._pipeline import _synthesize_facet
from typologist.llm import LLM, _CallableLLM


class _StubLLM(LLM):
    """Test double that stubs ``call_structured`` and exposes provider/model.

    Used both as the synthesis-side ``schema_llm`` (where ``call_structured`` is
    invoked) and as the ``labeling_llm`` provenance source (where only
    ``provider`` and ``model_name`` are read).
    """

    def __init__(
        self,
        response: dict | None = None,
        *,
        provider: str | None = "anthropic",
        model_name: str | None = "claude-opus-4-7",
    ):
        self._response = response or {}
        self.provider = provider
        self._model_name = model_name

    @property
    def model_name(self) -> str | None:
        return self._model_name

    def __call__(self, prompt: str, **options):
        return ""

    def call_structured(self, prompt: str, response_schema: dict) -> dict:
        return dict(self._response)


def _labeling_llm(provider: str | None = "anthropic", model: str | None = "claude-haiku-4-5"):
    return _StubLLM(provider=provider, model_name=model)


def test_synthesize_facet_assembles_schema_entry():
    schema_llm = _StubLLM(
        {
            "name": "contribution_type",
            "kind": "categorical",
            "values": ["empirical_study", "method_paper", "theory"],
            "definition": "What the paper primarily contributes.",
        }
    )

    facet, synthesis_prompt = _synthesize_facet(
        cluster_hierarchy=[["topic_a", "topic_b"]],
        schema_llm=schema_llm,
        labeling_llm=_labeling_llm(),
        object_description="scientific paper",
        corpus_description="arxiv ML papers",
        prior_facet_names=[],
    )

    assert facet["name"] == "contribution_type"
    assert facet["kind"] == "categorical"
    assert facet["values"] == ["empirical_study", "method_paper", "theory", "Other"]
    assert facet["definition"] == "What the paper primarily contributes."
    assert facet["labeling_model"] == "anthropic:claude-haiku-4-5"
    assert "{document}" in facet["labeling_prompt_template"]
    assert "contribution_type" in facet["labeling_prompt_template"]
    assert "scientific paper" in synthesis_prompt


def test_synthesize_facet_records_callable_llm_as_none_labeling_model():
    schema_llm = _StubLLM(
        {
            "name": "f",
            "kind": "categorical",
            "values": ["a", "b"],
            "definition": "d",
        }
    )
    facet, _ = _synthesize_facet(
        cluster_hierarchy=[["t"]],
        schema_llm=schema_llm,
        labeling_llm=_CallableLLM(lambda p: ""),
        object_description="doc",
        corpus_description="corpus",
        prior_facet_names=[],
    )
    assert facet["labeling_model"] is None


def test_synthesize_facet_emits_provider_model_for_openai_labeling():
    schema_llm = _StubLLM(
        {
            "name": "f",
            "kind": "categorical",
            "values": ["a", "b"],
            "definition": "d",
        }
    )
    facet, _ = _synthesize_facet(
        cluster_hierarchy=[["t"]],
        schema_llm=schema_llm,
        labeling_llm=_labeling_llm(provider="openai", model="gpt-4o-mini"),
        object_description="doc",
        corpus_description="corpus",
        prior_facet_names=[],
    )
    assert facet["labeling_model"] == "openai:gpt-4o-mini"


def test_synthesize_facet_rejects_name_collision():
    schema_llm = _StubLLM(
        {
            "name": "contribution_type",
            "kind": "categorical",
            "values": ["a", "b"],
            "definition": "d",
        }
    )
    with pytest.raises(RuntimeError, match="collides"):
        _synthesize_facet(
            cluster_hierarchy=[["t"]],
            schema_llm=schema_llm,
            labeling_llm=_labeling_llm(),
            object_description="doc",
            corpus_description="corpus",
            prior_facet_names=["contribution_type"],
        )


def test_synthesize_facet_rejects_duplicate_values():
    schema_llm = _StubLLM(
        {
            "name": "f",
            "kind": "categorical",
            "values": ["a", "a", "b"],
            "definition": "d",
        }
    )
    with pytest.raises(RuntimeError, match="duplicate values"):
        _synthesize_facet(
            cluster_hierarchy=[["t"]],
            schema_llm=schema_llm,
            labeling_llm=_labeling_llm(),
            object_description="doc",
            corpus_description="corpus",
            prior_facet_names=[],
        )


def test_synthesize_facet_rejects_too_few_values():
    schema_llm = _StubLLM(
        {
            "name": "f",
            "kind": "categorical",
            "values": ["only_one"],
            "definition": "d",
        }
    )
    with pytest.raises(RuntimeError, match="fewer than 2 values"):
        _synthesize_facet(
            cluster_hierarchy=[["t"]],
            schema_llm=schema_llm,
            labeling_llm=_labeling_llm(),
            object_description="doc",
            corpus_description="corpus",
            prior_facet_names=[],
        )


def test_synthesize_facet_rejects_missing_required_field():
    schema_llm = _StubLLM(
        {
            "name": "f",
            "kind": "categorical",
            # missing "values" and "definition"
        }
    )
    with pytest.raises(RuntimeError, match="missing required field"):
        _synthesize_facet(
            cluster_hierarchy=[["t"]],
            schema_llm=schema_llm,
            labeling_llm=_labeling_llm(),
            object_description="doc",
            corpus_description="corpus",
            prior_facet_names=[],
        )


def test_synthesize_facet_appends_other_when_absent():
    schema_llm = _StubLLM(
        {
            "name": "f",
            "kind": "categorical",
            "values": ["a", "b", "c"],
            "definition": "d",
        }
    )
    facet, _ = _synthesize_facet(
        cluster_hierarchy=[["t"]],
        schema_llm=schema_llm,
        labeling_llm=_labeling_llm(),
        object_description="doc",
        corpus_description="corpus",
        prior_facet_names=[],
    )
    assert facet["values"] == ["a", "b", "c", "Other"]


def test_synthesize_facet_dedupes_other_case_insensitively():
    # If the LLM ignores instructions and includes "other", we don't double-append.
    schema_llm = _StubLLM(
        {
            "name": "f",
            "kind": "categorical",
            "values": ["a", "b", "other"],
            "definition": "d",
        }
    )
    facet, _ = _synthesize_facet(
        cluster_hierarchy=[["t"]],
        schema_llm=schema_llm,
        labeling_llm=_labeling_llm(),
        object_description="doc",
        corpus_description="corpus",
        prior_facet_names=[],
    )
    assert facet["values"] == ["a", "b", "other"]


def test_synthesize_facet_other_appears_in_labeling_template():
    schema_llm = _StubLLM(
        {
            "name": "f",
            "kind": "categorical",
            "values": ["a", "b"],
            "definition": "d",
        }
    )
    facet, _ = _synthesize_facet(
        cluster_hierarchy=[["t"]],
        schema_llm=schema_llm,
        labeling_llm=_labeling_llm(),
        object_description="doc",
        corpus_description="corpus",
        prior_facet_names=[],
    )
    # Template should enumerate all values including the appended "Other".
    assert "a, b, Other" in facet["labeling_prompt_template"]


def test_synthesize_facet_passes_prior_names_to_prompt():
    """Verify that prior facet names show up in the synthesis prompt the LLM sees."""
    captured_prompts: list[str] = []

    class _CaptureStub(_StubLLM):
        def call_structured(self, prompt, response_schema):
            captured_prompts.append(prompt)
            return dict(self._response)

    stub = _CaptureStub(
        {
            "name": "f",
            "kind": "categorical",
            "values": ["a", "b"],
            "definition": "d",
        }
    )
    _synthesize_facet(
        cluster_hierarchy=[["t"]],
        schema_llm=stub,
        labeling_llm=_labeling_llm(),
        object_description="doc",
        corpus_description="corpus",
        prior_facet_names=["contribution_type", "data_modality"],
    )

    assert len(captured_prompts) == 1
    assert "contribution_type" in captured_prompts[0]
    assert "data_modality" in captured_prompts[0]
