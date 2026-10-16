"""Pure cached-discovery description prompts; no collection or I/O."""
from __future__ import annotations
import json
from .relation import key_string

ACTIVATION_PROTOCOL = "pie-B-cached-discovery-activation-v2"

ROLE_PROTOCOL = "pie-B-cached-discovery-role-v2"

FIELD_CAP = 280

def activation_messages(packet: dict, displays: dict | None = None) -> list[dict[str, str]]:
    def example(record):
        item = {key: record[key] for key in ("prompt", "source_layer", "source_position", "activation") if key in record}
        if displays and "source_token_id" in record:
            item["source_token_spelling"] = displays["token_spellings_by_id"].get(str(record["source_token_id"]))
        return item
    evidence = {"high_activation_examples": [example(x) for x in packet["activation_examples"]],
                "absent_feature_contrasts": [example(x) for x in packet["contrast_examples"]]}
    return [{"role": "system", "content":
             'Describe the observed activation condition in IOI discovery examples. Return JSON with '
             '"activation_condition" and "scope_and_uncertainty", each a nonempty string of at most 280 characters. '
             'Use one short sentence per field, aiming for 20 words or fewer and 140 characters or fewer. '
             'Do not list examples or repeat the evidence. Count characters before returning JSON. '
             'Describe only visible patterns; do not infer relations or unseen examples. '
             'Token spellings preserve tokenizer markers and do not provide character offsets.'},
            {"role": "user", "content": json.dumps(evidence, ensure_ascii=False, separators=(",", ":"))}]

def role_messages(base: str, packet: dict, labels: dict, displays: dict | None = None) -> list[dict[str, str]]:
    evidence = []
    for row in packet["measured_role_examples"]:
        neighbor = key_string(row["target_feature_key"])
        if neighbor not in labels:
            raise ValueError("Missing discovery neighbor activation label")
        item = {key: row[key] for key in ("prompt", "source_layer", "source_position", "target_layer",
                                        "target_position", "value", "measurement_status")}
        item["neighbor_activation_condition"] = labels[neighbor]["activation_condition"]
        if displays:
            ids = displays["discovery_input_token_ids"][str(row["prompt_pair_id"])]
            item["source_token_spelling"] = displays["token_spellings_by_id"][str(ids[row["source_position"]])]
            item["target_token_spelling"] = displays["token_spellings_by_id"][str(ids[row["target_position"]])]
        evidence.append(item)
    return [{"role": "system", "content":
             'Describe only measured discovery attribution evidence. Return JSON with '
             '"cross_layer_role" and "scope_and_uncertainty", each nonempty and at most 280 characters. '
             'Use one short sentence per field, aiming for 20 words or fewer and 140 characters or fewer. '
             'Do not list examples or repeat the evidence. Count characters before returning JSON. '
             'Preserve uncertainty and the restricted sampled neighborhood. These values are signed local '
             'gradient-times-write contributions, not nonlinear removal effects. Do not infer an unmeasured edge.'},
            {"role": "user", "content": json.dumps({"fixed_activation_condition": base,
             "measured_discovery_evidence": evidence}, ensure_ascii=False, separators=(",", ":"))}]
