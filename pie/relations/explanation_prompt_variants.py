"""Discovery-only role prompt variants for an isolated diagnostic.

The original renderer owns evidence extraction. Variants change instructions
and JSON formatting, never the supplied labels, evidence rows, or measurements.
No API client is instantiated and no gold/reference files are read here.
"""
from __future__ import annotations

import json

from .description_prompts import FIELD_CAP, ROLE_PROTOCOL, role_messages


ORGANIZED_ROLE_PROMPT_VERSION = "pie-B-cached-discovery-role-organized-v1"
ROLE_PROMPT_PROTOCOLS = {
    "original": ROLE_PROTOCOL,
    "organized-v1": ORGANIZED_ROLE_PROMPT_VERSION,
}

ORGANIZED_ROLE_SYSTEM = (
    "Describe the supplied measured discovery evidence as a neural cross-layer attribution role. "
    "Return one JSON object with exactly cross_layer_role and scope_and_uncertainty, "
    f"each a nonempty string of at most {FIELD_CAP} characters. "
    "Use concise clauses and count characters before returning. "
    "Keep the fixed activation condition unchanged; it is context for the source, not an output field.\n\n"
    "Measurement meaning: each value is a signed local gradient-times-source-write contribution "
    "to the downstream target feature. Positive values contribute positively to target activity "
    "(locally promote it); negative values contribute negatively to target activity "
    "(locally inhibit it). These are local attribution values, "
    "not measured nonlinear removal effects. A fictional giver or recipient is a narrative role, "
    "distinct from the direction of a neural contribution.\n\n"
    "Priority for cross_layer_role, which is shown to the relationship evaluator: "
    "state the measured positive, negative, or mixed contribution direction; identify the affected "
    "downstream activation type using neighbor_activation_condition and its observed target layer; "
    "then add the source/target token or position context supported by the rows. "
    "When signs differ, connect direction to the observed downstream type, layer, or context; "
    "if the supplied rows do not distinguish those conditions, state that direction varies within "
    "the sampled evidence. Include essential sign uncertainty in cross_layer_role itself. "
    "Use qualitative relative strength only for comparable measurements to the same downstream "
    "target and reference units; otherwise leave strength unresolved. Do not output raw measured "
    "values or numeric feature indices.\n\n"
    "Scope: these are a restricted sample of discovery examples selected for large absolute "
    "contributions. Their signs and magnitudes describe the shown sample; they do not establish "
    "frequencies, universal behavior, or contributions to other targets. "
    "Use scope_and_uncertainty for additional sampling limits and conditional ambiguity. "
    "Describe only the displayed measurements, labels, layers, positions, and token spellings; "
    "do not invent target identities, thresholds, edges, or causal intervention results. "
    "Preserve uncertainty rather than completing a plausible transfer story."
)


def role_messages_variant(
    base: str,
    packet: dict,
    labels: dict,
    displays: dict | None = None,
    variant: str = "organized-v1",
) -> list[dict[str, str]]:
    """Render a versioned variant from exactly the original public evidence.

    ``original`` returns the existing messages unchanged. ``organized-v1``
    retains the original two user-JSON keys and every evidence-row field,
    numeric value, label, and row order, with readable JSON indentation.
    ``base``, ``packet``, ``labels``, and ``displays`` are never mutated.
    """
    if variant not in ROLE_PROMPT_PROTOCOLS:
        raise ValueError(f"Unsupported role message variant: {variant}")
    original = role_messages(base, packet, labels, displays)
    if variant == "original":
        return original
    public_evidence = json.loads(original[1]["content"])
    return [
        {"role": "system", "content": ORGANIZED_ROLE_SYSTEM},
        {
            "role": "user",
            "content": json.dumps(public_evidence, ensure_ascii=False, indent=2),
        },
    ]
