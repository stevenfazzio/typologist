from __future__ import annotations

import pytest

from typologist._llm import _LLM
from typologist._pipeline import _synthesize_field


class _StubSchemaLLM(_LLM):
    def __init__(self, response: dict, model_name: str | None = "claude-opus-4-7"):
        self._response = response
        self._model_name = model_name

    @property
    def model_name(self) -> str | None:
        return self._model_name

    def __call__(self, prompt: str, **options):
        return ""

    def call_structured(self, prompt: str, response_schema: dict) -> dict:
        return dict(self._response)


def test_synthesize_field_assembles_schema_entry():
    schema_llm = _StubSchemaLLM(
        {
            "name": "contribution_type",
            "type": "categorical",
            "values": ["empirical_study", "method_paper", "theory"],
            "definition": "What the paper primarily contributes.",
        }
    )

    facet, synthesis_prompt = _synthesize_field(
        cluster_hierarchy=[["topic_a", "topic_b"]],
        schema_llm=schema_llm,
        labeling_llm_model_name="claude-haiku-4-5",
        object_description="scientific paper",
        corpus_description="arxiv ML papers",
        prior_facet_names=[],
    )

    assert facet["name"] == "contribution_type"
    assert facet["type"] == "categorical"
    assert facet["values"] == ["empirical_study", "method_paper", "theory"]
    assert facet["definition"] == "What the paper primarily contributes."
    assert facet["labeling_model"] == "claude-haiku-4-5"
    assert "{document}" in facet["labeling_prompt_template"]
    assert "contribution_type" in facet["labeling_prompt_template"]
    assert "scientific paper" in synthesis_prompt


def test_synthesize_field_records_callable_llm_as_none_labeling_model():
    schema_llm = _StubSchemaLLM(
        {
            "name": "f",
            "type": "categorical",
            "values": ["a", "b"],
            "definition": "d",
        }
    )
    facet, _ = _synthesize_field(
        cluster_hierarchy=[["t"]],
        schema_llm=schema_llm,
        labeling_llm_model_name=None,  # simulates callable labeling_llm
        object_description="doc",
        corpus_description="corpus",
        prior_facet_names=[],
    )
    assert facet["labeling_model"] is None


def test_synthesize_field_rejects_name_collision():
    schema_llm = _StubSchemaLLM(
        {
            "name": "contribution_type",
            "type": "categorical",
            "values": ["a", "b"],
            "definition": "d",
        }
    )
    with pytest.raises(RuntimeError, match="collides"):
        _synthesize_field(
            cluster_hierarchy=[["t"]],
            schema_llm=schema_llm,
            labeling_llm_model_name="m",
            object_description="doc",
            corpus_description="corpus",
            prior_facet_names=["contribution_type"],
        )


def test_synthesize_field_rejects_duplicate_values():
    schema_llm = _StubSchemaLLM(
        {
            "name": "f",
            "type": "categorical",
            "values": ["a", "a", "b"],
            "definition": "d",
        }
    )
    with pytest.raises(RuntimeError, match="duplicate values"):
        _synthesize_field(
            cluster_hierarchy=[["t"]],
            schema_llm=schema_llm,
            labeling_llm_model_name="m",
            object_description="doc",
            corpus_description="corpus",
            prior_facet_names=[],
        )


def test_synthesize_field_rejects_too_few_values():
    schema_llm = _StubSchemaLLM(
        {
            "name": "f",
            "type": "categorical",
            "values": ["only_one"],
            "definition": "d",
        }
    )
    with pytest.raises(RuntimeError, match="fewer than 2 values"):
        _synthesize_field(
            cluster_hierarchy=[["t"]],
            schema_llm=schema_llm,
            labeling_llm_model_name="m",
            object_description="doc",
            corpus_description="corpus",
            prior_facet_names=[],
        )


def test_synthesize_field_rejects_missing_required_field():
    schema_llm = _StubSchemaLLM(
        {
            "name": "f",
            "type": "categorical",
            # missing "values" and "definition"
        }
    )
    with pytest.raises(RuntimeError, match="missing required field"):
        _synthesize_field(
            cluster_hierarchy=[["t"]],
            schema_llm=schema_llm,
            labeling_llm_model_name="m",
            object_description="doc",
            corpus_description="corpus",
            prior_facet_names=[],
        )


def test_synthesize_field_passes_prior_names_to_prompt():
    """Verify that prior facet names show up in the synthesis prompt the LLM sees."""
    captured_prompts: list[str] = []

    class _CaptureStub(_StubSchemaLLM):
        def call_structured(self, prompt, response_schema):
            captured_prompts.append(prompt)
            return dict(self._response)

    stub = _CaptureStub(
        {
            "name": "f",
            "type": "categorical",
            "values": ["a", "b"],
            "definition": "d",
        }
    )
    _synthesize_field(
        cluster_hierarchy=[["t"]],
        schema_llm=stub,
        labeling_llm_model_name="m",
        object_description="doc",
        corpus_description="corpus",
        prior_facet_names=["contribution_type", "data_modality"],
    )

    assert len(captured_prompts) == 1
    assert "contribution_type" in captured_prompts[0]
    assert "data_modality" in captured_prompts[0]
