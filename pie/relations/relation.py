"""CPU-only, reference-blind relation-question construction for Task B."""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from .api_client import canonical, sha256


PROMPT_VERSION = "pie-B-relation-choice-v1"
THREE_METRICS_PROMPT_VERSION = "pie-B-three-metrics-v1"
VARIABLE_THREE_METRICS_PROMPT_VERSION = "pie-B-three-metrics-v2"
EVIDENCE_EVALUATOR_PROMPT_REVISION = "pie-B-evaluator-evidence-v2"
DESCRIPTION_VERSION = "pie-B-discovery-role-v1"
LETTERS = "ABCD"


def candidate_labels(labels: Iterable[str]) -> tuple[str, ...]:
    """Validate a nonempty prefix of A-D and return its canonical ordering."""
    try:
        provided = tuple(labels)
    except TypeError as exc:
        raise ValueError("Candidate labels must be a prefix of A-D") from exc
    expected = tuple(LETTERS[:len(provided)])
    if (not 1 <= len(provided) <= len(LETTERS) or
            any(type(label) is not str for label in provided) or
            len(set(provided)) != len(provided) or set(provided) != set(expected)):
        raise ValueError("Candidate labels must be a nonempty prefix of A-D")
    return expected


def feature_key(model_revision: str, clt_revision: str, node: dict[str, Any]) -> list[Any]:
    return [model_revision, clt_revision, node["layer"], node["feature_index"]]


def key_string(key: list[Any]) -> str:
    return canonical(key).decode()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(canonical(row).decode() + "\n")


def validate_graph(graph: dict[str, Any]) -> None:
    required = ("graph_id", "prompt_pair_id", "prompt", "model_revision", "clt_revision",
                "pair_dataset_sha256", "edge_semantics", "normalization", "position_policy",
                "nodes", "edges", "support_policy")
    absent = [field for field in required if field not in graph or graph[field] in (None, "")]
    if absent:
        raise ValueError(f"Graph lacks required provenance: {absent}")
    if graph["support_policy"] not in ("explicit_computed_entries", "complete_measured_submatrix"):
        raise ValueError("Graph support policy does not identify measured entries")
    if not isinstance(graph["prompt_pair_id"], (str, int)):
        raise ValueError("Prompt pair ID must be scalar")
    nodes = {node["id"]: node for node in graph["nodes"]}
    if len(nodes) != len(graph["nodes"]):
        raise ValueError("Duplicate node ID")
    seen_edges: set[tuple[str, str]] = set()
    for node in nodes.values():
        if node["type"] == "feature":
            for field in ("layer", "position", "feature_index"):
                if not isinstance(node.get(field), int) or node[field] < 0:
                    raise ValueError(f"Invalid feature {field}")
    for edge in graph["edges"]:
        source, target = edge["source"], edge["target"]
        if source not in nodes or target not in nodes:
            raise ValueError("Edge endpoint missing from node map")
        if (source, target) in seen_edges:
            raise ValueError("Duplicate edge measurement")
        seen_edges.add((source, target))
        state = edge.get("measurement_status")
        if state not in ("computed_zero", "computed_nonzero", "unmeasured_omitted"):
            raise ValueError("Missing explicit edge measurement status")
        value = edge.get("value")
        if state == "unmeasured_omitted":
            if value is not None:
                raise ValueError("Unmeasured edge has a value")
        elif not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("Computed edge needs a finite signed value")
        elif (state == "computed_zero") != (value == 0):
            raise ValueError("Zero status and edge value disagree")


def inspect_legacy_graph(path: Path) -> dict[str, Any]:
    """Summarize old graph JSON, without promoting sparse links to measured zeros."""
    obj = json.loads(path.read_text(encoding="utf-8"))
    result: dict[str, Any] = {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    if isinstance(obj, list) and obj and "inbound" in obj[0]:
        result.update({"format": "pruned_feature_connectivity", "nodes": len(obj),
                       "listed_feature_links": sum(len(x["inbound"]["features"]) for x in obj),
                       "eligible_targets_with_4_listed_links": sum(len(x["inbound"]["features"]) >= 4 for x in obj),
                       "limitations": ["one graph with no pair ID", "top-k pruned link lists",
                                       "unlisted edges have unknown measurement status",
                                       "model/CLT revisions and token alignment absent"]})
    elif isinstance(obj, dict) and "nodes" in obj and "links" in obj:
        meta = obj.get("metadata") or {}
        result.update({"format": "circuit_tracer_visualization", "nodes": len(obj["nodes"]),
                       "listed_links": len(obj["links"]), "prompt": meta.get("prompt"),
                       "scan": meta.get("scan"),
                       "limitations": ["one example prompt, not an approved pair ID",
                                       "listed links do not certify absent edges as measured zero",
                                       "model revision and full acquisition settings absent"]})
    else:
        raise ValueError("Unsupported legacy graph structure")
    return result


def make_provisional_split(pairs: list[dict[str, Any]], dataset_hash: str,
                           seed: int = 0) -> dict[str, Any]:
    """Template-index groups prevent template reuse; never an approved final split."""
    by_template: dict[int, list[int]] = defaultdict(list)
    seen_ids: set[int] = set()
    for pair in pairs:
        pair_id = pair["id"]
        if pair_id in seen_ids:
            raise ValueError("Duplicate pair ID")
        seen_ids.add(pair_id)
        by_template[pair["template_idx"]].append(pair_id)
    if len(by_template) < 6:
        raise ValueError("Too few template groups for three-way split")
    groups = sorted(by_template)
    random.Random(seed).shuffle(groups)
    n = len(groups)
    assignment = {group: ("discovery" if i < round(0.60 * n) else
                          "development" if i < round(0.80 * n) else "final")
                  for i, group in enumerate(groups)}
    split_ids: dict[str, list[int]] = {name: [] for name in ("discovery", "development", "final")}
    for group in groups:
        split_ids[assignment[group]].extend(by_template[group])
    for name in split_ids:
        split_ids[name].sort()
    # Exact duplicate texts, including corrupted strings, cannot cross the boundary.
    text_split: dict[str, str] = {}
    duplicate_conflicts = []
    for pair in pairs:
        current = assignment[pair["template_idx"]]
        for field in ("text_clean", "text_corr"):
            value = re.sub(r"\s+", " ", pair[field].strip())
            previous = text_split.setdefault(value, current)
            if previous != current:
                duplicate_conflicts.append({"pair_id": pair["id"], "field": field})
    return {"status": "provisional_cpu_only", "pair_dataset_sha256": dataset_hash,
            "split_strategy": "template_idx_grouped_seeded_60_20_20",
            "template_to_split": {str(k): v for k, v in sorted(assignment.items())},
            "pair_ids": split_ids, "exact_duplicate_conflicts": duplicate_conflicts,
            "seed": seed, "near_duplicate_check": "pending_manual_or_task_specific_review",
            "final_sealed": True}


def build_question_bank(graphs: list[dict[str, Any]], split: dict[str, Any], *,
                        question_split: str, seed: int, min_abs: float, min_gap: float,
                        descriptions: dict[str, str] | None = None) -> tuple[list[dict[str, Any]],
                                                                            list[dict[str, Any]],
                                                                            list[dict[str, Any]], dict[str, int]]:
    if question_split not in ("development", "final"):
        raise ValueError("Question split must be held out from discovery")
    if min_abs < 0 or min_gap < 0:
        raise ValueError("Signal and separation thresholds must be nonnegative")
    if split.get("status") == "provisional_cpu_only" and question_split == "final":
        raise ValueError("Provisional final split remains sealed")
    allowed_ids = set(split["pair_ids"][question_split])
    counts: Counter[str] = Counter()
    public: list[dict[str, Any]] = []
    refs: list[dict[str, Any]] = []
    gold: list[dict[str, Any]] = []
    for graph in graphs:
        validate_graph(graph)
        if graph["pair_dataset_sha256"] != split["pair_dataset_sha256"]:
            raise ValueError("Graph and pair dataset hashes disagree")
        if graph["prompt_pair_id"] not in allowed_ids:
            counts["outside_question_split"] += 1
            continue
        nodes = {node["id"]: node for node in graph["nodes"]}
        incoming: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for edge in graph["edges"]:
            src, dst = nodes[edge["source"]], nodes[edge["target"]]
            if edge["measurement_status"] == "unmeasured_omitted":
                continue
            if src["type"] != "feature" or dst["type"] != "feature":
                continue
            if src["layer"] >= dst["layer"]:
                continue
            incoming[edge["target"]].append(edge)
        for target_id, edges in incoming.items():
            target = nodes[target_id]
            groups: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
            for edge in edges:
                src = nodes[edge["source"]]
                groups[(src["layer"], str(src.get("position_role", src["position"])))].append(edge)
            eligible = [group for group in groups.values() if len(group) >= 4]
            if not eligible:
                counts["fewer_than_4_matched_measured_sources"] += 1
                continue
            chosen_group = max(eligible, key=lambda group: max(abs(e["value"]) for e in group))
            four = sorted(chosen_group, key=lambda e: (-abs(e["value"]), e["source"]))[:4]
            if abs(four[0]["value"]) < min_abs:
                counts["weak_signal"] += 1
                continue
            if abs(four[0]["value"]) - abs(four[1]["value"]) < min_gap:
                counts["tie_or_small_gap"] += 1
                continue
            target_key = feature_key(graph["model_revision"], graph["clt_revision"], target)
            if descriptions is not None and key_string(target_key) not in descriptions:
                counts["missing_target_activation_description"] += 1
                continue
            if descriptions is not None and any(
                key_string(feature_key(graph["model_revision"], graph["clt_revision"],
                                       nodes[edge["source"]])) not in descriptions for edge in four):
                counts["missing_candidate_activation_description"] += 1
                continue
            question_id = sha256([graph["graph_id"], target_id, sorted(e["source"] for e in four), seed])[:20]
            rng = random.Random(int(sha256([question_id, seed])[:16], 16))
            rng.shuffle(four)
            candidates = []
            ref_candidates = []
            labels: dict[str, dict[str, Any]] = {}
            for letter, edge in zip(LETTERS, four):
                source = nodes[edge["source"]]
                source_key = feature_key(graph["model_revision"], graph["clt_revision"], source)
                candidates.append({"label": letter, "source_layer": source["layer"],
                                   "source_position": source["position"],
                                   "source_position_role": source.get("position_role")})
                ref_candidates.append({"label": letter, "feature_key": source_key})
                labels[letter] = {"value": edge["value"], "sign": ("positive" if edge["value"] > 0 else
                                                                     "negative" if edge["value"] < 0 else "uncertain"),
                                  "source_node_id": edge["source"]}
            correct = max(LETTERS, key=lambda letter: abs(labels[letter]["value"]))
            public.append({"question_id": question_id, "prompt_pair_id": graph["prompt_pair_id"],
                           "prompt": graph["prompt"], "target": {"layer": target["layer"],
                                                                    "position": target["position"]},
                           "candidates": candidates, "prompt_version": PROMPT_VERSION})
            refs.append({"question_id": question_id, "target_feature_key": target_key,
                         "candidate_feature_keys": ref_candidates})
            gold.append({"question_id": question_id, "graph_id": graph["graph_id"],
                         "prompt_pair_id": graph["prompt_pair_id"], "target_node_id": target_id,
                         "correct_label": correct, "candidate_values_and_signs": labels,
                         "strongest_abs": abs(labels[correct]["value"]),
                         "separation": abs(labels[correct]["value"]) -
                                       max(abs(v["value"]) for k, v in labels.items() if k != correct)})
            counts["questions"] += 1
    if len({q["question_id"] for q in public}) != len(public):
        raise ValueError("Question ID collision or duplicate graph target")
    return public, refs, gold, dict(counts)


def make_role_messages(activation_condition: str, discovery_packet: dict[str, Any],
                       *, max_evidence_chars: int = 10000) -> list[dict[str, str]]:
    if not activation_condition.strip():
        raise ValueError("Activation condition missing")
    packet = canonical(discovery_packet).decode()
    if len(packet) > max_evidence_chars:
        raise ValueError("Discovery packet exceeds fixed evidence cap")
    return [{"role": "system", "content":
             "Describe only the measured discovery evidence. Return a JSON object with keys "
             "cross_layer_role and scope_and_uncertainty. Never infer an unmeasured edge."},
            {"role": "user", "content":
             "Existing activation condition (fixed): " + activation_condition +
             "\nDiscovery graph neighborhood with signed measured values and neighbor activation labels:\n" + packet}]


def validate_role_record(record: dict[str, Any], activation_condition: str,
                         *, max_chars: int = 550) -> dict[str, str]:
    if not isinstance(record, dict):
        raise ValueError("Role response must be an object")
    role = record.get("cross_layer_role")
    uncertainty = record.get("scope_and_uncertainty")
    if not isinstance(role, str) or not isinstance(uncertainty, str):
        raise ValueError("Role response missing string fields")
    if len(role) > max_chars or len(uncertainty) > max_chars:
        raise ValueError("Role field exceeds fixed cap")
    return {"activation_condition": activation_condition,
            "cross_layer_role": role.strip(), "scope_and_uncertainty": uncertainty.strip()}


def render_evaluator_messages(question: dict[str, Any], refs: dict[str, Any],
                              descriptions: dict[str, dict[str, str]], condition: str,
                              shuffle_seed: int = 0, max_description_chars: int = 700,
                              *, evaluator_prompt_revision: str | None = None) -> tuple[list[dict[str, str]], dict[str, str]]:
    """Render from public question and ID bridge only; gold is never an argument."""
    if condition not in ("metadata", "activation", "relational", "shuffled"):
        raise ValueError("Unknown condition")
    if evaluator_prompt_revision not in (None, EVIDENCE_EVALUATOR_PROMPT_REVISION):
        raise ValueError("Unsupported evaluator prompt revision")
    if evaluator_prompt_revision is not None and question.get("prompt_version") not in (
            THREE_METRICS_PROMPT_VERSION, VARIABLE_THREE_METRICS_PROMPT_VERSION):
        raise ValueError("Evidence evaluator revision requires a three-metric question")
    if question["question_id"] != refs["question_id"]:
        raise ValueError("Question/reference join mismatch")
    letters = candidate_labels(candidate["label"] for candidate in question["candidates"])
    variable = question.get("prompt_version") == VARIABLE_THREE_METRICS_PROMPT_VERSION
    if not variable and letters != tuple(LETTERS):
        raise ValueError("Variable candidate counts require the versioned variable prompt")
    if (type(question.get("candidate_count", len(letters))) is not int or
            question.get("candidate_count", len(letters)) != len(letters)):
        raise ValueError("Question candidate count disagrees with its labels")
    if condition == "shuffled" and len(letters) == 1:
        raise ValueError("Singleton shuffled role control is unavailable: no derangement exists")
    target_key = key_string(refs["target_feature_key"])
    target = descriptions.get(target_key, {})
    if condition != "metadata" and not target.get("activation_condition"):
        raise ValueError("Missing target activation description")
    mapped_refs = {x["label"]: key_string(x["feature_key"]) for x in refs["candidate_feature_keys"]}
    if len(refs["candidate_feature_keys"]) != len(letters) or set(mapped_refs) != set(letters):
        raise ValueError("Candidate reference labels invalid")
    role_assignment = {letter: letter for letter in letters}
    if condition == "shuffled":
        rng = random.Random(int(sha256([question["question_id"], shuffle_seed])[:16], 16))
        shift = rng.choice(tuple(range(1, len(letters))))
        role_assignment = {letter: letters[(i + shift) % len(letters)]
                           for i, letter in enumerate(letters)}
    lines = ["Prompt: " + question["prompt"],
             f"Target feature at layer {question['target']['layer']}, position {question['target']['position']}."]
    if condition != "metadata":
        lines.append("Target activation condition: " + target["activation_condition"][:max_description_chars])
    for candidate in question["candidates"]:
        letter = candidate["label"]
        line = f"{letter}: upstream feature at layer {candidate['source_layer']}, position {candidate['source_position']}."
        if condition != "metadata":
            desc = descriptions.get(mapped_refs[letter], {})
            activation = desc.get("activation_condition")
            if not activation:
                raise ValueError("Missing candidate activation description")
            line += " Activation condition: " + activation[:max_description_chars]
            if condition in ("relational", "shuffled"):
                assigned = descriptions.get(mapped_refs[role_assignment[letter]], {})
                role = assigned.get("cross_layer_role")
                if not role:
                    raise ValueError("Missing candidate role description")
                line += " Cross-layer attribution role: " + role[:max_description_chars]
        lines.append(line)
    label_range = "A-D" if len(letters) == 4 else f"A-{letters[-1]}"
    instruction = ("The only candidate is A; return choice A. "
                   "Predict its contribution sign as positive, negative, or uncertain. "
                   if len(letters) == 1 else
                   f"Choose the candidate with the greatest absolute contribution to this target among {label_range}. "
                   "Predict the sign of each candidate contribution as positive, negative, or uncertain. ")
    if question.get("prompt_version") in (THREE_METRICS_PROMPT_VERSION, VARIABLE_THREE_METRICS_PROMPT_VERSION):
        threshold = question["relation_threshold"]
        instruction += (
            f"A meaningful relation has an absolute measured contribution of at least {threshold} "
            "in this target's reference units, using a threshold frozen on development data. "
            "For each candidate, give the probability from 0 to 1 that its relation is meaningful. "
            "Candidates may all have weak effects; still choose the most likely strongest source. ")
        if evaluator_prompt_revision == EVIDENCE_EVALUATOR_PROMPT_REVISION:
            instruction += (
                "Infer each answer from the visible evidence. Candidate labels convey no strength or sign. "
                "Positive contributions promote target feature activity; negative contributions inhibit it. "
                "Use uncertain when the evidence does not establish a direction. A relation can be meaningful "
                "with either sign; its probability concerns absolute magnitude, not positivity. "
                "Return one JSON object with exactly three fields: "
                "choice is a string containing your predicted strongest candidate label; "
                "signs is an object mapping every candidate label to positive, negative, or uncertain; "
                "relation_probabilities is an object mapping every candidate label to a finite number "
                "between 0 and 1. "
                "The candidate labels are " + ", ".join(letters) + ". "
                "Supply your predictions as the field values and omit explanations.")
        elif variable:
            instruction += "Return JSON with exactly these answer fields: "
            example_signs = dict(zip(LETTERS, ("positive", "negative", "uncertain", "positive")))
            instruction += json.dumps({"choice": "A", "signs": {label: example_signs[label] for label in letters},
                                       "relation_probabilities": {label: 0.5 for label in letters}},
                                      separators=(",", ":"))
        else:
            instruction += "Return JSON with exactly these answer fields: "
            instruction += ('{"choice":"A","signs":{"A":"positive","B":"negative","C":"uncertain","D":"positive"},'
                            '"relation_probabilities":{"A":0.5,"B":0.5,"C":0.5,"D":0.5}}')
    else:
        instruction += ('Return JSON: {"choice":"A","signs":{"A":"positive","B":"negative",'
                        '"C":"uncertain","D":"positive"}}')
    lines.append(instruction)
    system = "Use only the visible descriptions and prompt. Return one JSON object."
    if evaluator_prompt_revision == EVIDENCE_EVALUATOR_PROMPT_REVISION:
        system = ("Predict held-out feature contributions using only the visible descriptions and prompt. "
                  "Use the evidence to choose each answer value. Return one JSON object matching the requested fields.")
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": "\n".join(lines)}]
    return messages, role_assignment


def validate_evaluator_answer(answer: Any, *, require_probabilities: bool = False,
                              expected_labels: Iterable[str] = LETTERS) -> dict[str, Any]:
    letters = candidate_labels(expected_labels)
    if not isinstance(answer, dict) or answer.get("choice") not in letters:
        raise ValueError("Missing candidate choice")
    signs = answer.get("signs")
    if not isinstance(signs, dict) or set(signs) != set(letters):
        raise ValueError("Need one sign per candidate")
    if any(sign not in ("positive", "negative", "uncertain") for sign in signs.values()):
        raise ValueError("Invalid sign label")
    result = {"choice": answer["choice"], "signs": signs}
    if require_probabilities:
        probabilities = answer.get("relation_probabilities")
        if not isinstance(probabilities, dict) or set(probabilities) != set(letters):
            raise ValueError("Need one relation probability per candidate")
        if any(type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1
               for p in probabilities.values()):
            raise ValueError("Relation probabilities must be finite numbers in [0,1]")
        result["relation_probabilities"] = probabilities
    return result


def score_responses(gold: list[dict[str, Any]], responses: list[dict[str, Any]],
                    *, strong_edge_min_abs: float) -> dict[str, Any]:
    by_id = {r["question_id"]: r for r in responses}
    if len(by_id) != len(responses):
        raise ValueError("Duplicate response")
    if set(by_id) - {g["question_id"] for g in gold}:
        raise ValueError("Responses contain unknown questions")
    n = len(gold)
    correct = 0
    valid = 0
    sign_n = 0
    sign_correct = 0
    by_prompt: dict[str, list[int]] = defaultdict(list)
    paired_rows = []
    for g in gold:
        response = by_id.get(g["question_id"])
        parsed = None
        if response is not None:
            try:
                parsed = validate_evaluator_answer(response.get("answer"),
                                                   expected_labels=g["candidate_values_and_signs"])
            except ValueError:
                pass
        top_correct = int(parsed is not None and parsed["choice"] == g["correct_label"])
        correct += top_correct
        valid += int(parsed is not None)
        by_prompt[str(g["prompt_pair_id"])].append(top_correct)
        if g["strongest_abs"] >= strong_edge_min_abs:
            sign_n += 1
            sign_correct += int(parsed is not None and parsed["signs"][g["correct_label"]] ==
                                g["candidate_values_and_signs"][g["correct_label"]]["sign"])
        paired_rows.append({"question_id": g["question_id"], "prompt_pair_id": g["prompt_pair_id"],
                            "valid": parsed is not None, "top1_correct": top_correct})
    return {"questions": n, "valid": valid, "invalid_or_missing": n - valid,
            "top1_correct": correct, "top1_accuracy_all": correct / n if n else None,
            "strong_sign_questions": sign_n, "strong_sign_correct": sign_correct,
            "strong_sign_accuracy_all": sign_correct / sign_n if sign_n else None,
            "prompt_count": len(by_prompt), "paired_rows": paired_rows}
