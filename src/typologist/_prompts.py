from __future__ import annotations

_SYNTHESIS_PROMPT = """\
You are analyzing a corpus of {object_description} from {corpus_description}.

Clustered topics discovered in this corpus at multiple levels of granularity:

{hierarchy_block}

Identify a single categorical axis along which these {object_description} vary.{accounted_for_block}

Respond with only valid JSON of the form:
{{
  "name": "<lowercase snake_case identifier>",
  "kind": "categorical",
  "values": ["<value1>", "<value2>", ...],
  "definition": "<one short sentence explaining what this facet captures>"
}}

Rules:
- Do not include "Other" or any similar catch-all value. A catch-all will
  be appended automatically after your response.
- Choose value names that are specific and discriminating; avoid vague
  umbrella terms like "General", "Miscellaneous", "Novel method", or
  anything broad enough to absorb the majority of {object_description}.
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


def _format_accounted_for_entry(entry: dict) -> str:
    """Format one bullet of the accounted-for-axes block.

    ``entry`` is one of:
    - ``{"kind": "prior_facet", "name": str}``
    - ``{"kind": "erased_metadata", "name": str, "type": "categorical" | "ordinal",
         "values_shown": list[str], "truncated": bool, "total_unique": int}``
    """
    name = entry["name"]
    if entry["kind"] == "prior_facet":
        return f'- "{name}"'

    values_joined = ", ".join(entry["values_shown"])
    remaining = entry["total_unique"] - len(entry["values_shown"])
    trailing = f", ... and {remaining} more" if entry["truncated"] else ""
    if entry["type"] == "ordinal":
        return f'- "{name}" (ordinal; ordered values: {values_joined}{trailing})'
    return f'- "{name}" (categorical; values include: {values_joined}{trailing})'


def render_synthesis_prompt(
    cluster_hierarchy: list[list[str]],
    object_description: str,
    corpus_description: str,
    prior_facet_names: list[str],
    erased_metadata_descriptions: list[dict] | None = None,
) -> str:
    """Build the one-shot prompt that ``schema_llm`` sees when proposing a new facet.

    ``cluster_hierarchy`` is Toponymy's ``topic_names_``: layers from finest
    (layer 0) to coarsest.

    ``erased_metadata_descriptions`` is an optional list of dicts produced by
    ``_describe_erased_metadata`` describing columns the caller passed to
    ``fit(metadata=...)``. They appear alongside prior facets in a unified
    "do not re-propose" block so the LLM is steered off axes that have already
    been accounted for, whether by Typologist itself or by the user.
    """
    hierarchy_lines: list[str] = []
    for i, layer in enumerate(cluster_hierarchy):
        hierarchy_lines.append(f"Layer {i}:")
        for name in layer:
            hierarchy_lines.append(f"  - {name}")
    hierarchy_block = "\n".join(hierarchy_lines)

    entries: list[dict] = []
    for name in prior_facet_names:
        entries.append({"kind": "prior_facet", "name": name})
    for desc in erased_metadata_descriptions or []:
        entries.append({"kind": "erased_metadata", **desc})

    if entries:
        bullets = "\n".join(_format_accounted_for_entry(e) for e in entries)
        accounted_for_block = (
            "\n\nThe following axes have already been accounted for in this "
            "corpus and should NOT be re-proposed. Propose an axis that is "
            "orthogonal to all of them:\n" + bullets
        )
    else:
        accounted_for_block = ""

    return _SYNTHESIS_PROMPT.format(
        object_description=object_description,
        corpus_description=corpus_description,
        hierarchy_block=hierarchy_block,
        accounted_for_block=accounted_for_block,
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
