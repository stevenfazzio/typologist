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
        prior_facet_names=[],
    )
    assert "scientific paper" in out
    assert "machine-learning arxiv papers" in out


def test_synthesis_prompt_renders_hierarchy_layers():
    out = render_synthesis_prompt(
        cluster_hierarchy=[["fine_a", "fine_b"], ["coarse_a"]],
        object_description="document",
        corpus_description="corpus",
        prior_facet_names=[],
    )
    assert "Layer 0:" in out
    assert "Layer 1:" in out
    assert "fine_a" in out
    assert "fine_b" in out
    assert "coarse_a" in out


def test_synthesis_prompt_omits_accounted_for_block_when_empty():
    out = render_synthesis_prompt(
        cluster_hierarchy=[["t"]],
        object_description="doc",
        corpus_description="corpus",
        prior_facet_names=[],
    )
    assert "already been accounted for" not in out


def test_synthesis_prompt_includes_prior_facet_names():
    out = render_synthesis_prompt(
        cluster_hierarchy=[["t"]],
        object_description="doc",
        corpus_description="corpus",
        prior_facet_names=["contribution_type", "data_modality"],
    )
    assert "already been accounted for" in out
    assert "contribution_type" in out
    assert "data_modality" in out


def test_synthesis_prompt_describes_erased_categorical_metadata():
    out = render_synthesis_prompt(
        cluster_hierarchy=[["t"]],
        object_description="doc",
        corpus_description="corpus",
        prior_facet_names=[],
        erased_metadata_descriptions=[
            {
                "name": "product_category",
                "type": "categorical",
                "values_shown": ["Books", "Electronics", "All_Beauty"],
                "truncated": False,
                "total_unique": 3,
            }
        ],
    )
    assert "already been accounted for" in out
    assert "product_category" in out
    assert "categorical" in out
    assert "Books, Electronics, All_Beauty" in out


def test_synthesis_prompt_describes_erased_ordinal_metadata():
    out = render_synthesis_prompt(
        cluster_hierarchy=[["t"]],
        object_description="doc",
        corpus_description="corpus",
        prior_facet_names=[],
        erased_metadata_descriptions=[
            {
                "name": "rating",
                "type": "ordinal",
                "values_shown": ["1.0", "2.0", "3.0", "4.0", "5.0"],
                "truncated": False,
                "total_unique": 5,
            }
        ],
    )
    assert '"rating"' in out
    assert "ordinal" in out
    assert "ordered values: 1.0, 2.0, 3.0, 4.0, 5.0" in out


def test_synthesis_prompt_shows_truncation_tail_for_high_cardinality_metadata():
    out = render_synthesis_prompt(
        cluster_hierarchy=[["t"]],
        object_description="doc",
        corpus_description="corpus",
        prior_facet_names=[],
        erased_metadata_descriptions=[
            {
                "name": "country",
                "type": "categorical",
                "values_shown": ["AR", "AT", "AU", "BE", "BR", "CA", "CH", "CN"],
                "truncated": True,
                "total_unique": 200,
            }
        ],
    )
    assert "... and 192 more" in out


def test_synthesis_prompt_combines_prior_facets_and_erased_metadata():
    out = render_synthesis_prompt(
        cluster_hierarchy=[["t"]],
        object_description="doc",
        corpus_description="corpus",
        prior_facet_names=["tone"],
        erased_metadata_descriptions=[
            {
                "name": "product_category",
                "type": "categorical",
                "values_shown": ["Books"],
                "truncated": False,
                "total_unique": 1,
            }
        ],
    )
    assert "already been accounted for" in out
    assert "tone" in out
    assert "product_category" in out


def test_synthesis_prompt_instructs_llm_not_to_include_other():
    out = render_synthesis_prompt(
        cluster_hierarchy=[["t"]],
        object_description="doc",
        corpus_description="corpus",
        prior_facet_names=[],
    )
    # "Other" is appended programmatically after synthesis; the LLM should not
    # add its own catch-all value.
    assert 'Do not include "Other"' in out


def test_synthesis_prompt_warns_against_broad_umbrella_values():
    out = render_synthesis_prompt(
        cluster_hierarchy=[["t"]],
        object_description="paper",
        corpus_description="corpus",
        prior_facet_names=[],
    )
    # predecessor smoke test showed "novel_method_or_architecture" absorbed
    # the majority of docs; the prompt should explicitly counter that pattern.
    assert "Novel method" in out


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
