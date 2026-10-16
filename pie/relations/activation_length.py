"""Discovery-only activation evidence and character-budget factorial renderer."""
from __future__ import annotations

import json
import re
from .description_prompts import activation_messages
from . import organized_brief as brief

VARIANTS = {
    "brief_v3_280": {"field_cap": 280, "role_soft_target": 160,
                     "scope_soft_target": 80, "activation_examples": False},
    "examples_280": {"field_cap": 280, "role_soft_target": 160,
                     "scope_soft_target": 80, "activation_examples": True},
    "brief_v3_800": {"field_cap": 800, "role_soft_target": 600,
                     "scope_soft_target": 200, "activation_examples": False},
    "examples_800": {"field_cap": 800, "role_soft_target": 600,
                     "scope_soft_target": 200, "activation_examples": True},
}
EXAMPLE_INSTRUCTION = (
    "\n\nSource activation context: source_activation_evidence contains the source's "
    "cached discovery high-activation examples and absent-feature contrasts. Use "
    "these examples to identify the source-token context relevant to the measured "
    "role. Keep the fixed activation condition unchanged. If the label and examples "
    "are ambiguous, preserve that uncertainty. Activation or absence alone does "
    "not establish a cross-layer relation, sign, or strength: derive those only "
    "from measured_discovery_evidence. Summarize the role rather than listing "
    "activation examples."
)


def protocol(variant):
    if variant == "brief_v3_280":
        return brief.PROTOCOL
    if variant not in VARIANTS:
        raise ValueError("Unknown fixed ablation variant")
    return "pie-B-cached-discovery-role-activation-length-v1-" + variant


def role_messages(base, packet, labels, displays, variant):
    design = VARIANTS[variant]
    messages = brief.role_messages_variant(base, packet, labels, displays)
    if variant == "brief_v3_280":
        return messages
    system = messages[0]["content"]
    if design["field_cap"] == 800:
        replacements = {"280": "800", "160": "600", "80": "200"}
        system = re.sub(r"\b(?:280|160|80)\b", lambda m: replacements[m.group()], system)
        system = system.replace("The unchanged hard limit", "The hard limit")
    evidence = json.loads(messages[1]["content"])
    if design["activation_examples"]:
        system += EXAMPLE_INSTRUCTION
        evidence["source_activation_evidence"] = json.loads(activation_messages(packet, displays)[1]["content"])
        user = json.dumps(evidence, ensure_ascii=False, indent=2)
    else:
        user = messages[1]["content"]
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def validate_role(answer, variant):
    cap = VARIANTS[variant]["field_cap"]
    return (isinstance(answer, dict) and set(answer) == {"cross_layer_role", "scope_and_uncertainty"}
            and all(isinstance(text, str) and text.strip() and len(text) <= cap for text in answer.values()))
