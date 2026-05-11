from __future__ import annotations

_SYNTHESIS_PROMPT = """\
You are analyzing a corpus of {object_description} from {corpus_description}.

Clustered topics discovered in this corpus at multiple levels of granularity:

{hierarchy_block}

Identify {n_facets} mutually orthogonal categorical axes along which these {object_description} vary.

Respond with only valid JSON of the form:
{{
  "facets": [
    {{
      "name": "<lowercase snake_case identifier>",
      "kind": "categorical",
      "values": ["<value1>", "<value2>", ...],
      "definition": "<one short sentence explaining what this facet captures>"
    }}
  ]
}}

Return exactly {n_facets} facets in the array.

Rules:
- The {n_facets} facets must be distinct categorical axes, orthogonal to each other (a document's value on one facet should not predict its value on another).
- Do not include "Other" or any similar catch-all value. A catch-all will be appended automatically after your response.
- Choose value names that are specific and discriminating; avoid vague umbrella terms like "General", "Miscellaneous", or anything broad enough to absorb the majority of {object_description}.
- Use lowercase snake_case for value names unless proper nouns are required.
- The "name" field should describe the axis itself, not a specific value on it.
"""


_LABELING_TEMPLATE = """\
Classify the following {object_description} along the "{facet_name}" dimension.

{facet_definition}

Choose exactly one of: {values_joined}

{object_description}:
{{document}}

Respond with the chosen value name only, no explanation.\
"""


def render_synthesis_prompt(
    cluster_hierarchy: list[list[str]],
    object_description: str,
    corpus_description: str,
    n_facets: int,
) -> str:
    """Build the one-shot prompt that ``schema_llm`` sees when proposing facets.

    ``cluster_hierarchy`` is Toponymy's ``topic_names_``: layers from finest
    (layer 0) to coarsest.
    """
    hierarchy_lines: list[str] = []
    for i, layer in enumerate(cluster_hierarchy):
        hierarchy_lines.append(f"Layer {i}:")
        for name in layer:
            hierarchy_lines.append(f"  - {name}")
    hierarchy_block = "\n".join(hierarchy_lines)

    return _SYNTHESIS_PROMPT.format(
        object_description=object_description,
        corpus_description=corpus_description,
        hierarchy_block=hierarchy_block,
        n_facets=n_facets,
    )


def render_labeling_template(
    facet_name: str,
    facet_definition: str,
    values: list[str],
    object_description: str,
) -> str:
    """Build the per-facet template that gets stored in ``schema_[i]["labeling_prompt_template"]``.

    The returned string contains a single ``{document}`` placeholder; everything
    else is baked in at discovery time so the stored template is self-contained
    for later replay (see ``apply_schema``).
    """
    values_joined = ", ".join(values)
    return _LABELING_TEMPLATE.format(
        object_description=object_description,
        facet_name=facet_name,
        facet_definition=facet_definition,
        values_joined=values_joined,
    )


def render_labeling_prompt(template: str, document: str) -> str:
    """Substitute a specific document into a stored labeling template.

    Uses string replace rather than ``str.format`` so that documents containing
    literal curly braces (JSON snippets, code) don't blow up.
    """
    return template.replace("{document}", document)
