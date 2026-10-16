"""Format-first role instructions; the original organized evidence is unchanged.

This pure renderer changes the system instruction and prompt namespace only.
It does not call a model, validate/edit a reply, read gold, or modify caches.
"""
from __future__ import annotations

from .explanation_prompt_variants import (ORGANIZED_ROLE_SYSTEM,
                                         ROLE_PROMPT_PROTOCOLS as ORIGINAL_PROTOCOLS,
                                         role_messages_variant as original_variant)

VARIANT = "organized-brief-v3"
PROTOCOL = "pie-B-cached-discovery-role-organized-brief-v3"
ROLE_PROMPT_PROTOCOLS = {**ORIGINAL_PROTOCOLS, VARIANT: PROTOCOL}
FIELD_CAP = 280
ROLE_SOFT_TARGET = 160
SCOPE_SOFT_TARGET = 80
FORMATTING_PROTOCOL = {
    "variant": VARIANT, "prompt_version": PROTOCOL, "hard_field_cap": FIELD_CAP,
    "role_soft_character_target": ROLE_SOFT_TARGET,
    "scope_soft_character_target": SCOPE_SOFT_TARGET,
    "word_limit": None, "one_sentence_per_field": True,
    "instruction_order": "format_first_scientific_rules_preserved_check_last",
    "evidence_and_row_order_changed": False, "activation_labels_changed": False,
    "thresholds_changed": False, "model_or_profile_changed": False,
    "response_schema_changed": False, "received_reply_editing": False,
}

FORMAT_FIRST = (
    "Return a compact attribution summary in one JSON object with exactly "
    "cross_layer_role and scope_and_uncertainty. Both values must be nonempty strings.\n"
    "FORMAT: each field is one short sentence. Target at most 160 characters in "
    "cross_layer_role and at most 80 characters in scope_and_uncertainty. "
    "The unchanged hard limit is 280 characters for each field. Characters include "
    "spaces and punctuation in the decoded string; no word-count limit applies.\n"
    "ROLE: prioritize measured positive, negative, or mixed contribution direction "
    "and the downstream activation type. Keep essential conditional direction and "
    "sign uncertainty in cross_layer_role itself. If mixed signs cannot be resolved "
    "by the supplied conditions, say that direction varies within the sample. "
    "Add source/target context only when it helps explain the role and fits.\n"
    "BREVITY: summarize across the rows. Do not enumerate examples, individual "
    "layer-position transitions, token positions, or feature indices. Omit redundant "
    "details and repeated context; preserve the supported direction and uncertainty.\n"
    "SCOPE: use one short clause for the restricted discovery sample and any "
    "additional ambiguity. Do not repeat the role, directions, or evidence examples."
)

FINAL_CHECK = (
    "Before sending: check the length of each decoded field. If the draft exceeds "
    "its target, rewrite it concisely while retaining measured direction, downstream "
    "type, and essential uncertainty. Keep both fields within the 280-character "
    "hard limit. Return only the two-field JSON object, without counts or commentary."
)

# The scientific rules, measurement semantics and restricted-support caveats
# remain verbatim. Clear formatting instructions bracket them.
SYSTEM = FORMAT_FIRST + "\n\n" + ORGANIZED_ROLE_SYSTEM + "\n\n" + FINAL_CHECK


def role_messages_variant(base, packet, labels, displays=None, variant=VARIANT):
    """Replace only the system message; preserve the original user bytes."""
    if variant == VARIANT:
        messages = original_variant(base, packet, labels, displays, variant="organized-v1")
        return [{"role": "system", "content": SYSTEM}, messages[1]]
    return original_variant(base, packet, labels, displays, variant=variant)
