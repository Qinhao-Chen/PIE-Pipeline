"""Binary History/History+role/Role prompts for user-supplied relation evidence."""
from __future__ import annotations
from .api_client import canonical, sha256
from .relation import key_string

CONDITIONS = ("history", "history_role", "role")

VERSION = "pie-B-FAP-rank-band-evaluator-v1"

SYSTEM = (
    'Use the supplied IOI prompt, source information and target activation description to predict '
    'the measured signed cross-layer local replacement-gradient times source-write relation. '
    'The sources share a layer and exact token position. Choose the source with the larger absolute '
    'measured contribution to the target. Predict a sign for each source and the probability that '
    'its absolute measured contribution meets relation_threshold in raw target encoder-input units. '
    'Probabilities are independent and need not sum to one. Discovery histories show activation '
    'contexts; activation or absence alone does not establish relation strength or sign. '
    'Marked tokenizer positions refer to the complete answer-free input including BOS, without padding; '
    'token spellings are vocabulary markers, not character offsets. Role uncertainty describes '
    'restricted discovery evidence. Return only JSON with exactly choice, signs, relation_probabilities: '
    '{"choice":"A" or "B","signs":{"A":"positive" or "negative" or "uncertain",'
    '"B":"positive" or "negative" or "uncertain"},"relation_probabilities":{"A":number,'
    '"B":number}}. Every probability must be a finite number from 0 to 1.'
)

def require(test, message):
    if not test:
        raise ValueError(message)

def history(packet, display):
    examples = []
    for row in packet["activation_examples"]:
        ids = display["discovery_input_token_ids"][str(row["prompt_pair_id"])]
        tokens = [display["token_spellings_by_id"][str(token)] for token in ids]
        position = row["source_position"]
        require(ids[position] == row["source_token_id"], "History token alignment differs")
        examples.append({"prompt": row["prompt"], "source_layer": row["source_layer"],
            "source_position": position, "source_token_spelling": tokens[position],
            "highlighted_tokens": " ".join(("[[" + token + "]]" if i == position else token)
                                           for i, token in enumerate(tokens)),
            "activation": row["activation"]})
    # Absence is prompt-wide and has no invented position or measured token zero.
    contrasts = [{"prompt": row["prompt"], "scope": row["scope"]}
                 for row in packet["contrast_examples"]]
    require(len(examples) <= 6 and len(contrasts) <= 6, "History exceeds fixed exemplar budget")
    return {"high_activation_examples": examples, "absent_feature_contrasts": contrasts}

def shuffled_keys(sources):
    # Fixed global cyclic derangement, independent of gold and outcomes.
    ordered = sorted(sources, key=lambda key: sha256([20261011, "role_derangement", key]))
    require(len(ordered) >= 2, "Shuffled roles need at least two source identities")
    return {key: ordered[(i + 1) % len(ordered)] for i, key in enumerate(ordered)}

def render(question, reference, histories, labels, condition, presentation, assignments):
    target_key = key_string(reference["target_feature_key"])
    target = labels[target_key]
    public = {"prompt": question["prompt"], "target": {**question["target"],
        "activation_condition": target["activation_condition"],
        "scope_and_uncertainty": target["activation_scope_and_uncertainty"]},
        "sources": {}, "relation_threshold": question["relation_threshold"]}
    by_label = {row["label"]: row for row in reference["candidate_feature_keys"]}
    for row in question["candidates"]:
        label = row["label"]
        key = key_string(by_label[label]["feature_key"])
        source = {"layer": row["source_layer"], "position": row["source_position"]}
        if condition in ("history", "history_role", "shuffled"):
            source["history"] = histories[key]
        if condition in ("history_role", "role", "shuffled"):
            role = labels[assignments[key] if condition == "shuffled" else key]
            source["role"] = {"cross_layer_role": role["cross_layer_role"],
                              "scope_and_uncertainty": role["scope_and_uncertainty"]}
        public["sources"][label] = source
    if presentation == "reversed":
        public["sources"] = {"A": public["sources"]["B"], "B": public["sources"]["A"]}
    else:
        require(presentation == "original", "Unknown candidate presentation")
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": canonical(public).decode()}]
