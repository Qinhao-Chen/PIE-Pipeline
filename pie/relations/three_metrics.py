"""Offline, versioned CLT relation detection, strength and direction metrics.

This module consumes measured caches. It never runs the subject model or an API.
See docs/plans/THREE_METRICS_PROTOCOL.md for populations and missing-data rules.
"""

from __future__ import annotations

import math
import json
import random
from collections import Counter, defaultdict
from typing import Any

from .api_client import sha256, utc_now
from .relation import (LETTERS, THREE_METRICS_PROMPT_VERSION, VARIABLE_THREE_METRICS_PROMPT_VERSION,
                      candidate_labels, feature_key,
                      key_string, validate_evaluator_answer, validate_graph)

METRIC_PROTOCOL = "pie-B-three-metrics-v1"
VARIABLE_METRIC_PROTOCOL = VARIABLE_THREE_METRICS_PROMPT_VERSION
SUPPORTED_METRIC_PROTOCOLS = (METRIC_PROTOCOL, VARIABLE_METRIC_PROTOCOL)
POLICY_SCHEMA = "pie-B-three-metrics-policy-v1"
METRIC_NAMES = ("relation_detection_ap", "strongest_source_accuracy", "sign_accuracy")
CONTEXT_FIELDS = ("model_revision", "clt_revision", "pair_dataset_sha256",
                  "edge_semantics", "normalization", "position_policy")
THRESHOLD_FIELDS = ("relation_min_abs", "winner_min_gap", "sign_min_abs")


def _number(value: Any, name: str, *, positive: bool = False) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if value < 0 or (positive and value == 0):
        raise ValueError(f"{name} must be {'positive' if positive else 'nonnegative'}")
    return float(value)


def _thresholds(values: dict[str, Any]) -> dict[str, float]:
    if set(values) != set(THRESHOLD_FIELDS):
        raise ValueError("Threshold policy must specify relation, winner-gap and sign thresholds")
    result = {name: _number(values[name], name, positive=name != "winner_min_gap")
              for name in THRESHOLD_FIELDS}
    if result["sign_min_abs"] < result["relation_min_abs"]:
        raise ValueError("Sign threshold must be at least the relation threshold")
    return result


def _split_members(split: dict[str, Any]) -> dict[str, set[Any]]:
    if not split.get("pair_dataset_sha256"):
        raise ValueError("Split lacks dataset provenance")
    members: dict[str, set[Any]] = {}
    for name in ("discovery", "development", "final"):
        ids = split.get("pair_ids", {}).get(name)
        if not isinstance(ids, list) or any(type(x) not in (str, int) for x in ids):
            raise ValueError("Split must contain scalar prompt IDs in all three partitions")
        members[name] = set(ids)
        if len(members[name]) != len(ids):
            raise ValueError("Split contains duplicate prompt IDs")
    if any(members[a] & members[b] for a, b in
           (("discovery", "development"), ("discovery", "final"), ("development", "final"))):
        raise ValueError("Discovery, development and final splits overlap")
    if split.get("exact_duplicate_conflicts"):
        raise ValueError("Split has cross-partition duplicate texts")
    return members


def _context(graph: dict[str, Any]) -> dict[str, Any]:
    validate_graph(graph)
    if any(type(e.get("value")) is bool for e in graph["edges"]):
        raise ValueError("A Boolean is not a measured contribution")
    return {name: graph[name] for name in CONTEXT_FIELDS}


def freeze_metric_policy(graphs: list[dict[str, Any]], split: dict[str, Any], *,
                         relation_min_abs: float, winner_min_gap: float, sign_min_abs: float,
                         split_sha256: str, graph_sha256: str,
                         per_target: dict[str, Any] | None = None) -> dict[str, Any]:
    """Record owner-chosen thresholds with development-only evidence provenance.

    This records a freeze; it does not fit or claim to validate the thresholds.
    Discovery records may accompany development data, but final records may not.
    """
    members = _split_members(split)
    thresholds = _thresholds(dict(relation_min_abs=relation_min_abs,
                                  winner_min_gap=winner_min_gap, sign_min_abs=sign_min_abs))
    if not split_sha256 or not graph_sha256:
        raise ValueError("Policy requires input file hashes")
    development = []
    for graph in graphs:
        _context(graph)
        prompt_id = graph["prompt_pair_id"]
        if prompt_id in members["final"]:
            raise ValueError("Final graphs cannot be used to freeze thresholds")
        if prompt_id not in members["development"] | members["discovery"]:
            raise ValueError("Policy input prompt is outside the discovery/development split")
        if graph["pair_dataset_sha256"] != split["pair_dataset_sha256"]:
            raise ValueError("Policy graph and split dataset mismatch")
        if prompt_id in members["development"]:
            development.append(graph)
    if not development:
        raise ValueError("Need development graphs to record threshold provenance")
    context = _context(development[0])
    if any(_context(g) != context for g in graphs):
        raise ValueError("Policy graphs mix model, CLT, units or position policies")
    overrides = {}
    for target_key, override in (per_target or {}).items():
        try:
            identity = json.loads(target_key)
        except (TypeError, ValueError) as exc:
            raise ValueError("Per-target thresholds need canonical feature identity keys") from exc
        if (not isinstance(identity, list) or len(identity) != 4 or
                identity[:2] != [context["model_revision"], context["clt_revision"]] or
                any(type(x) is not int or x < 0 for x in identity[2:]) or
                key_string(identity) != target_key):
            raise ValueError("Per-target threshold identity or model/CLT mismatch")
        if not isinstance(override, dict) or set(override) - set(THRESHOLD_FIELDS):
            raise ValueError("Invalid per-target threshold fields")
        overrides[target_key] = _thresholds({**thresholds, **override})
    return {"schema_version": POLICY_SCHEMA, "metric_protocol": METRIC_PROTOCOL,
            "frozen_on_split": "development", "created_at_utc": utc_now(),
            "threshold_source": "owner_chosen_on_development_not_automatically_fitted",
            "split_sha256": split_sha256, "split_content_sha256": sha256(split),
            "development_graph_sha256": graph_sha256,
            "development_prompt_ids": sorted({g["prompt_pair_id"] for g in development},
                                             key=lambda x: key_string([x])),
            "reference_context": context, "thresholds": thresholds, "per_target": overrides}


def validate_metric_policy(policy: dict[str, Any], split: dict[str, Any],
                           split_sha256: str | None = None) -> None:
    members = _split_members(split)
    if (policy.get("schema_version") != POLICY_SCHEMA or
            policy.get("metric_protocol") != METRIC_PROTOCOL or
            policy.get("frozen_on_split") != "development"):
        raise ValueError("Unrecognized or unfrozen three-metric policy")
    if policy.get("split_content_sha256") != sha256(split):
        raise ValueError("Policy and split content hash mismatch")
    if split_sha256 is not None and policy.get("split_sha256") != split_sha256:
        raise ValueError("Policy and split file hash mismatch")
    ids = policy.get("development_prompt_ids", [])
    if not ids or not set(ids) <= members["development"]:
        raise ValueError("Threshold provenance is not development-only")
    context = policy.get("reference_context", {})
    if any(not context.get(name) for name in CONTEXT_FIELDS):
        raise ValueError("Policy lacks reference context")
    if context["pair_dataset_sha256"] != split["pair_dataset_sha256"]:
        raise ValueError("Policy and split dataset mismatch")
    _thresholds(policy.get("thresholds", {}))
    for override in policy.get("per_target", {}).values():
        _thresholds(override)


def build_three_metric_bank(graphs: list[dict[str, Any]], split: dict[str, Any], *,
                            policy: dict[str, Any], question_split: str, seed: int,
                            descriptions: dict[str, Any] | None = None,
                            split_sha256: str | None = None):
    """Sample quartets without inspecting effect ranks, then attach private labels.

    Sampling is within verified measured graph support, not a whole-model universe.
    Measured weak and zero effects stay in the detection bank.
    """
    validate_metric_policy(policy, split, split_sha256)
    members = _split_members(split)
    if question_split not in ("development", "final"):
        raise ValueError("Questions must be held out from discovery")
    if question_split == "final" and split.get("status") == "provisional_cpu_only":
        raise ValueError("Provisional final split remains sealed")
    public, refs, gold = [], [], []
    counts: Counter[str] = Counter()
    seen_prompts = set()
    for graph in graphs:
        if _context(graph) != policy["reference_context"]:
            raise ValueError("Graph/policy model, CLT, units or position policy mismatch")
        if graph["prompt_pair_id"] not in members[question_split]:
            counts["outside_question_split"] += 1
            continue
        if graph["prompt_pair_id"] in seen_prompts:
            raise ValueError("More than one reference graph for the same prompt")
        seen_prompts.add(graph["prompt_pair_id"])
        nodes = {n["id"]: n for n in graph["nodes"]}
        occurrence_ids = [(n["layer"], n["position"], n["feature_index"])
                          for n in nodes.values() if n["type"] == "feature"]
        if len(set(occurrence_ids)) != len(occurrence_ids):
            raise ValueError("Graph contains duplicate feature occurrences")
        incoming: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for edge in graph["edges"]:
            source, target = nodes[edge["source"]], nodes[edge["target"]]
            if edge["measurement_status"] == "unmeasured_omitted":
                counts["omitted_measurements"] += 1
                continue
            if source["type"] == target["type"] == "feature" and source["layer"] < target["layer"]:
                incoming[edge["target"]].append(edge)
        for target_id in sorted(incoming, key=str):
            target = nodes[target_id]
            target_key = feature_key(graph["model_revision"], graph["clt_revision"], target)
            thresholds = policy["per_target"].get(key_string(target_key), policy["thresholds"])
            groups: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
            for edge in incoming[target_id]:
                source = nodes[edge["source"]]
                groups[(source["layer"], str(source.get("position_role", source["position"])))].append(edge)
            eligible = sorted(key for key, edges in groups.items() if len(edges) >= 4)
            if not eligible:
                counts["fewer_than_4_matched_measured_sources"] += 1
                continue
            metadata_seed = [METRIC_PROTOCOL, graph["pair_dataset_sha256"], graph["prompt_pair_id"],
                             target_key, target["position"], seed]
            rng = random.Random(int(sha256(metadata_seed)[:16], 16))
            group = groups[rng.choice(eligible)]
            group = sorted(group, key=lambda e: key_string([
                feature_key(graph["model_revision"], graph["clt_revision"], nodes[e["source"]]),
                nodes[e["source"]]["position"]]))
            four = rng.sample(group, 4)
            if descriptions is not None:
                needed = [target_key] + [feature_key(graph["model_revision"], graph["clt_revision"],
                                                    nodes[e["source"]]) for e in four]
                def has_activation(identity):
                    record = descriptions.get(key_string(identity))
                    return (record.get("activation_condition") if isinstance(record, dict) else record)
                if any(not has_activation(k) for k in needed):
                    counts["missing_activation_descriptions"] += 1
                    continue
            qid = sha256(metadata_seed + [sorted(e["source"] for e in four)])[:20]
            random.Random(int(sha256([qid, "labels", seed])[:16], 16)).shuffle(four)
            candidates, candidate_refs, labels = [], [], {}
            for letter, edge in zip(LETTERS, four):
                source, value = nodes[edge["source"]], edge["value"]
                candidates.append({"label": letter, "source_layer": source["layer"],
                                   "source_position": source["position"],
                                   "source_position_role": source.get("position_role")})
                candidate_refs.append({"label": letter, "feature_key": feature_key(
                    graph["model_revision"], graph["clt_revision"], source)})
                labels[letter] = {"value": value, "source_node_id": edge["source"],
                                  "sign": "positive" if value > 0 else "negative" if value < 0 else "uncertain",
                                  "relation_present": abs(value) >= thresholds["relation_min_abs"]}
            ordered = sorted(LETTERS, key=lambda x: -abs(labels[x]["value"]))
            strongest = abs(labels[ordered[0]]["value"])
            gap = strongest - abs(labels[ordered[1]]["value"])
            strength_eligible = (strongest >= thresholds["relation_min_abs"] and gap > 0 and
                                 gap >= thresholds["winner_min_gap"])
            sign_eligible = strength_eligible and strongest >= thresholds["sign_min_abs"]
            public.append({"question_id": qid, "prompt_pair_id": graph["prompt_pair_id"],
                           "prompt": graph["prompt"], "target": {
                               "layer": target["layer"], "position": target["position"]},
                           "candidates": candidates, "relation_threshold": thresholds["relation_min_abs"],
                           "prompt_version": THREE_METRICS_PROMPT_VERSION,
                           "metric_protocol": METRIC_PROTOCOL})
            refs.append({"question_id": qid, "target_feature_key": target_key,
                         "candidate_feature_keys": candidate_refs})
            gold.append({"question_id": qid, "prompt_pair_id": graph["prompt_pair_id"],
                         "graph_id": graph["graph_id"], "target_node_id": target_id,
                         "metric_protocol": METRIC_PROTOCOL, "policy_sha256": sha256(policy),
                         "relation_threshold": thresholds["relation_min_abs"],
                         "strength_eligible": strength_eligible, "sign_eligible": sign_eligible,
                         "correct_label": ordered[0] if strength_eligible else None,
                         "candidate_values_and_signs": labels,
                         "strongest_abs": strongest, "separation": gap})
            counts["questions"] += 1
            counts["strength_eligible"] += strength_eligible
            counts["sign_eligible"] += sign_eligible
            counts["weak_or_tied_strength_exclusions"] += not strength_eligible
            counts["detection_candidates"] += 4
            counts["detection_positives"] += sum(v["relation_present"] for v in labels.values())
    if len({q["question_id"] for q in public}) != len(public):
        raise ValueError("Duplicate three-metric question ID")
    return public, refs, gold, dict(counts)


def average_precision(labels: list[bool], probabilities: list[float]) -> float | None:
    """Stepwise AP with tied confidences grouped, including the whole tie group."""
    if len(labels) != len(probabilities):
        raise ValueError("Label and probability lengths differ")
    if any(x not in (0, 1, False, True) for x in labels):
        raise ValueError("AP labels must be binary")
    for p in probabilities:
        if type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1:
            raise ValueError("AP probabilities must be finite numbers in [0,1]")
    total_positive = sum(labels)
    if not total_positive:
        return None
    groups: dict[float, list[bool]] = defaultdict(list)
    for label, probability in zip(labels, probabilities):
        groups[probability].append(label)
    true_positive, seen, ap = 0, 0, 0.0
    for probability in sorted(groups, reverse=True):
        positives = sum(groups[probability])
        true_positive += positives
        seen += len(groups[probability])
        ap += (positives / total_positive) * (true_positive / seen)
    return ap


def score_three_metrics(gold: list[dict[str, Any]], responses: list[dict[str, Any]], *,
                        metric_protocol: str | None = None,
                        candidate_count: int | None = None) -> dict[str, Any]:
    if metric_protocol is not None and metric_protocol not in SUPPORTED_METRIC_PROTOCOLS:
        raise ValueError("Unrecognized three-metric protocol")
    if candidate_count is not None and (type(candidate_count) is not int or not 1 <= candidate_count <= 4):
        raise ValueError("Candidate count must be an integer from one through four")
    protocols = {g.get("metric_protocol") for g in gold}
    arities = {len(candidate_labels(g["candidate_values_and_signs"])) for g in gold}
    if len(protocols) > 1 or len(arities) > 1:
        raise ValueError("Three-metric scores require separate banks for each protocol and candidate count")
    observed_protocol = next(iter(protocols), metric_protocol or METRIC_PROTOCOL)
    observed_arity = next(iter(arities), candidate_count if candidate_count is not None else
                          4 if observed_protocol == METRIC_PROTOCOL else None)
    if observed_protocol not in SUPPORTED_METRIC_PROTOCOLS:
        raise ValueError("Legacy gold cannot be scored as three-metric data")
    if metric_protocol is not None and metric_protocol != observed_protocol:
        raise ValueError("Gold and requested metric protocol disagree")
    if candidate_count is not None and candidate_count != observed_arity:
        raise ValueError("Gold and requested candidate count disagree")
    if observed_protocol == METRIC_PROTOCOL and observed_arity != 4:
        raise ValueError("The original three-metric protocol requires four candidates")
    by_id = {r["question_id"]: r for r in responses}
    gold_ids = {g["question_id"] for g in gold}
    if len(gold_ids) != len(gold) or len(by_id) != len(responses):
        raise ValueError("Duplicate gold or response question ID")
    if set(by_id) - gold_ids:
        raise ValueError("Response contains unknown question")
    counts: Counter[str] = Counter()
    sign_total: Counter[str] = Counter()
    sign_correct: Counter[str] = Counter()
    all_labels, available_labels, probabilities, paired_rows = [], [], [], []
    for g in gold:
        labels = g["candidate_values_and_signs"]
        letters = candidate_labels(labels)
        if (g.get("candidate_count", len(letters)) != len(letters) or
                type(g.get("candidate_count", len(letters))) is not int):
            raise ValueError("Gold candidate count disagrees with its labels")
        if any(type(labels[x].get("relation_present")) is not bool for x in letters):
            raise ValueError("Three-metric gold requires measured binary relation labels")
        if type(g["sign_eligible"]) is not bool or type(g["strength_eligible"]) is not bool:
            raise ValueError("Gold eligibility flags must be Boolean")
        if len(letters) == 1 and g["strength_eligible"]:
            raise ValueError("Singleton strongest-source questions are trivial and must be excluded")
        if len(letters) > 1 and g["sign_eligible"] and not g["strength_eligible"]:
            raise ValueError("Sign subset must be contained in the clear-winner subset")
        if (g["strength_eligible"] or g["sign_eligible"]) and g["correct_label"] not in letters:
            raise ValueError("Eligible question lacks a strongest label")
        parsed = None
        response = by_id.get(g["question_id"])
        if response is not None:
            try:
                parsed = validate_evaluator_answer(response.get("answer"), require_probabilities=True,
                                                   expected_labels=letters)
            except ValueError:
                pass
        valid = parsed is not None
        counts["valid"] += valid
        detection_labels = [labels[x]["relation_present"] for x in letters]
        all_labels.extend(detection_labels)
        if valid:
            available_labels.extend(detection_labels)
            probabilities.extend(parsed["relation_probabilities"][x] for x in letters)
        top_correct = int(g["strength_eligible"] and valid and parsed["choice"] == g["correct_label"])
        sign_ok, true_sign = 0, None
        if g["strength_eligible"]:
            counts["strength_questions"] += 1
            counts["strength_correct"] += top_correct
        if g["sign_eligible"]:
            true_sign = labels[g["correct_label"]]["sign"]
            if true_sign not in ("positive", "negative"):
                raise ValueError("Sign-eligible gold must have a nonzero signed effect")
            sign_ok = int(valid and parsed["signs"][g["correct_label"]] == true_sign)
            sign_total[true_sign] += 1
            sign_correct[true_sign] += sign_ok
        paired_rows.append({"question_id": g["question_id"], "prompt_pair_id": g["prompt_pair_id"],
                            "valid": valid, "top1_correct": top_correct,
                            "strength_eligible": g["strength_eligible"], "sign_eligible": g["sign_eligible"],
                            "sign_correct": sign_ok, "true_sign": true_sign})
    n = len(gold)
    invalid = n - counts["valid"]
    positives, candidates = sum(all_labels), len(all_labels)
    reason = ("empty_bank" if not candidates else "incomplete_probabilities" if invalid else
              "single_gold_class" if positives in (0, candidates) else None)
    available_positives = sum(available_labels)
    available_classes = 0 < available_positives < len(available_labels)
    ap = average_precision(available_labels, probabilities) if available_classes else None
    sign_n = sum(sign_total.values())
    recalls = {sign: sign_correct[sign] / sign_total[sign] if sign_total[sign] else None
               for sign in ("positive", "negative")}
    balanced = sum(recalls.values()) / 2 if all(v is not None for v in recalls.values()) else None
    return {"metric_protocol": observed_protocol, "candidate_count": observed_arity, "questions": n,
            "valid": counts["valid"], "invalid_or_missing": invalid,
            "invalid_response_rate": invalid / n if n else None,
            "relation_detection_ap": ap if reason is None else None,
            "relation_detection_ap_undefined_reason": reason,
            "available_relation_detection_ap": ap,
            "available_detection_candidates": len(available_labels),
            "available_detection_positives": available_positives,
            "available_detection_prevalence": available_positives / len(available_labels)
                if available_labels else None,
            "detection_candidates": candidates, "detection_positives": positives,
            "detection_prevalence": positives / candidates if candidates else None,
            "detection_prediction_coverage": len(probabilities) / candidates if candidates else None,
            "strength_questions": counts["strength_questions"],
            "strength_correct": counts["strength_correct"],
            "strongest_source_accuracy": counts["strength_correct"] / counts["strength_questions"]
                if counts["strength_questions"] else None,
            "strength_excluded_questions": n - counts["strength_questions"],
            "singleton_strength_exclusions": n if observed_arity == 1 else 0,
            "strongest_source_accuracy_undefined_reason": ("single_candidate_trivial" if observed_arity == 1 else
                "empty_strength_population" if not counts["strength_questions"] else None),
            "sign_questions": sign_n, "sign_correct": sum(sign_correct.values()),
            "sign_accuracy": sum(sign_correct.values()) / sign_n if sign_n else None,
            "sign_balanced_accuracy": balanced,
            "sign_balanced_accuracy_undefined_reason": None if balanced is not None else "missing_sign_class",
            "sign_class_counts": dict(sign_total), "sign_class_accuracy": recalls,
            "prompt_count": len({key_string([g["prompt_pair_id"]]) for g in gold}),
            "paired_rows": paired_rows}


def paired_three_metric_bootstrap(gold: list[dict[str, Any]], responses: list[dict[str, Any]],
                                  condition_a: str, condition_b: str, seed: int = 0,
                                  n_boot: int = 2000, *, metric_protocol: str | None = None,
                                  candidate_count: int | None = None) -> dict[str, Any]:
    """Resample whole prompts and recompute nonlinear AP and balanced sign scores."""
    if n_boot < 1:
        raise ValueError("Need positive bootstrap replicates")
    rows_a = [r for r in responses if r["condition"] == condition_a]
    rows_b = [r for r in responses if r["condition"] == condition_b]
    score_options = {"metric_protocol": metric_protocol, "candidate_count": candidate_count}
    observed_a, observed_b = score_three_metrics(gold, rows_a, **score_options), score_three_metrics(gold, rows_b, **score_options)
    score_options = {"metric_protocol": observed_a["metric_protocol"], "candidate_count": observed_a["candidate_count"]}
    metrics = (*METRIC_NAMES, "sign_balanced_accuracy")
    clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for g in gold:
        clusters[key_string([g["prompt_pair_id"]])].append(g)
    prompts = sorted(clusters)
    by_a, by_b = ({r["question_id"]: r for r in rows} for rows in (rows_a, rows_b))
    draws: dict[str, list[float]] = {name: [] for name in metrics}
    rng = random.Random(seed)
    if prompts:
        for _ in range(n_boot):
            sampled_gold, sampled_a, sampled_b = [], [], []
            for occurrence, prompt in enumerate(rng.choice(prompts) for _ in prompts):
                for g in clusters[prompt]:
                    qid = f"{occurrence}:{g['question_id']}"
                    sampled_gold.append({**g, "question_id": qid})
                    for by_id, dest in ((by_a, sampled_a), (by_b, sampled_b)):
                        if g["question_id"] in by_id:
                            dest.append({**by_id[g["question_id"]], "question_id": qid})
            a, b = score_three_metrics(sampled_gold, sampled_a, **score_options), score_three_metrics(sampled_gold, sampled_b, **score_options)
            for name in metrics:
                if a[name] is not None and b[name] is not None:
                    draws[name].append(a[name] - b[name])
    results = {}
    for name in metrics:
        values = sorted(draws[name])
        difference = (observed_a[name] - observed_b[name]
                      if observed_a[name] is not None and observed_b[name] is not None else None)
        ci = ([values[int(0.025 * (len(values) - 1))], values[int(0.975 * (len(values) - 1))]]
              if values and difference is not None else None)
        results[name] = {"difference": difference, "ci_95": ci,
                         "defined_replicates": len(values), "undefined_replicates": n_boot - len(values),
                         "ci_population": "replicates_where_both_scores_are_defined"}
    return {"metric_protocol": observed_a["metric_protocol"], "candidate_count": observed_a["candidate_count"],
            "condition_a": condition_a, "condition_b": condition_b,
            "prompt_clusters": len(prompts), "question_count": len(gold), "metrics": results,
            "bootstrap_seed": seed, "bootstrap_replicates": n_boot}
