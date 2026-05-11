from typologist._prompts import (
    render_labeling_prompt,
    render_labeling_template,
    render_synthesis_prompt,
)


def test_synthesis_prompt_includes_descriptions():
    out = render_synthesis_prompt(
        cluster_hierarchy=[["topic_a", "topic_b"]],
        object_description="scientific paper",
        corpus_description="machine-learning arxiv papers",
        n_facets=3,
    )
    assert "scientific paper" in out
    assert "machine-learning arxiv papers" in out


def test_synthesis_prompt_renders_hierarchy_layers():
    out = render_synthesis_prompt(
        cluster_hierarchy=[["fine_a", "fine_b"], ["coarse_a"]],
        object_description="document",
        corpus_description="corpus",
        n_facets=2,
    )
    assert "Layer 0:" in out
    assert "Layer 1:" in out
    assert "fine_a" in out
    assert "fine_b" in out
    assert "coarse_a" in out


def test_synthesis_prompt_asks_for_exact_facet_count():
    out = render_synthesis_prompt(
        cluster_hierarchy=[["t"]],
        object_description="doc",
        corpus_description="corpus",
        n_facets=4,
    )
    # n_facets must show up where the prompt instructs the LLM to return that many
    assert "Identify 4 mutually orthogonal" in out
    assert "Return exactly 4 facets" in out


def test_synthesis_prompt_instructs_llm_not_to_include_other():
    out = render_synthesis_prompt(
        cluster_hierarchy=[["t"]],
        object_description="doc",
        corpus_description="corpus",
        n_facets=3,
    )
    # "Other" is appended programmatically after synthesis; the LLM should not
    # add its own catch-all value.
    assert 'Do not include "Other"' in out


def test_labeling_template_bakes_descriptions():
    template = render_labeling_template(
        facet_name="contribution_type",
        facet_definition="What the paper primarily contributes to the literature.",
        values=["empirical_study", "method_paper", "theory"],
        object_description="scientific paper",
    )
    assert "contribution_type" in template
    assert "scientific paper" in template
    assert "empirical_study, method_paper, theory" in template
    assert "What the paper primarily contributes" in template


def test_labeling_template_preserves_document_placeholder():
    template = render_labeling_template(
        facet_name="f",
        facet_definition="d",
        values=["a", "b"],
        object_description="doc",
    )
    assert "{document}" in template


def test_labeling_prompt_substitutes_document():
    template = "Prompt:\n{document}\nEnd."
    out = render_labeling_prompt(template, "actual document text")
    assert out == "Prompt:\nactual document text\nEnd."


def test_labeling_prompt_tolerates_braces_in_document():
    template = "Prompt:\n{document}\nEnd."
    tricky_doc = 'Some JSON: {"key": "value"} with braces.'
    out = render_labeling_prompt(template, tricky_doc)
    assert out == f"Prompt:\n{tricky_doc}\nEnd."


def test_labeling_template_is_self_contained_end_to_end():
    template = render_labeling_template(
        facet_name="sentiment",
        facet_definition="Overall tone.",
        values=["positive", "negative"],
        object_description="review",
    )
    prompt = render_labeling_prompt(template, "Great product!")
    assert "sentiment" in prompt
    assert "Great product!" in prompt
    assert "positive, negative" in prompt
