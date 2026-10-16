"""CPU-only baselines, clustered uncertainty, and exact-key relation coverage."""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from statistics import mean
from typing import Any

from .relation import (THREE_METRICS_PROMPT_VERSION, feature_key, key_string,
                      score_responses, validate_graph)
from .three_metrics import SUPPORTED_METRIC_PROTOCOLS


def occurrence_key(dataset_hash: str, prompt_pair_id: str | int,
                   node: dict[str, Any], model_revision: str,
                   clt_revision: str) -> list[Any]:
    return [dataset_hash, prompt_pair_id, node["position"],
            feature_key(model_revision, clt_revision, node)]


def _computed_feature_edges(graph: dict[str, Any]):
    nodes = {n["id"]: n for n in graph["nodes"]}
    for edge in graph["edges"]:
        if edge["measurement_status"] == "unmeasured_omitted":
            continue
        source, target = nodes[edge["source"]], nodes[edge["target"]]
        if source["type"] == target["type"] == "feature" and source["layer"] < target["layer"]:
            yield source, target, edge["value"]


def discovery_average_predictor(discovery_graphs: list[dict[str, Any]],
                                split: dict[str, Any], questions: list[dict[str, Any]],
                                refs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Exact source-target-context mean; unseen pairs use layer/position-role mean."""
    allowed = set(split["pair_ids"]["discovery"])
    exact: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    fallback: dict[tuple[int, int, str], list[float]] = defaultdict(list)
    sign_counts: Counter[str] = Counter()
    for graph in discovery_graphs:
        validate_graph(graph)
        if graph["pair_dataset_sha256"] != split["pair_dataset_sha256"]:
            raise ValueError("Discovery graph dataset hash mismatch")
        if graph["prompt_pair_id"] not in allowed:
            continue
        for source, target, value in _computed_feature_edges(graph):
            src_key = key_string(feature_key(graph["model_revision"], graph["clt_revision"], source))
            dst_key = key_string(feature_key(graph["model_revision"], graph["clt_revision"], target))
            role = source.get("position_role")
            role = str(source["position"] if role is None else role)
            exact[(src_key, dst_key, role)].append(value)
            fallback[(source["layer"], target["layer"], role)].append(value)
            sign_counts["positive" if value > 0 else "negative" if value < 0 else "uncertain"] += 1
    refs_by_id = {r["question_id"]: r for r in refs}
    rows = []
    coverage = Counter()
    for question in questions:
        ref = refs_by_id[question["question_id"]]
        target_key = key_string(ref["target_feature_key"])
        candidate_keys = {r["label"]: key_string(r["feature_key"])
                          for r in ref["candidate_feature_keys"]}
        predictions = {}
        signs = {}
        relation_probabilities = {}
        for candidate in question["candidates"]:
            letter = candidate["label"]
            role = candidate.get("source_position_role")
            role = str(candidate["source_position"] if role is None else role)
            source_key = candidate_keys[letter]
            values = exact.get((source_key, target_key, role))
            if values:
                coverage["exact_pair"] += 1
            else:
                values = fallback.get((candidate["source_layer"], question["target"]["layer"], role))
                coverage["layer_role_fallback" if values else "zero_fallback"] += 1
            score = mean(values) if values else 0.0
            predictions[letter] = score
            signs[letter] = "positive" if score > 0 else "negative" if score < 0 else "uncertain"
            if (question.get("prompt_version") == THREE_METRICS_PROMPT_VERSION or
                    question.get("metric_protocol") in SUPPORTED_METRIC_PROTOCOLS):
                relation_probabilities[letter] = (
                    sum(abs(v) >= question["relation_threshold"] for v in values) / len(values)
                    if values else 0.0)
        choice = max(predictions, key=lambda letter: abs(predictions[letter]))
        answer = {"choice": choice, "signs": signs}
        if relation_probabilities:
            answer["relation_probabilities"] = relation_probabilities
        rows.append({"question_id": question["question_id"], "condition": "discovery_average",
                     "status": "cpu_baseline", "answer": answer,
                     "fallback_policy": "exact_pair_role_else_layer_role_else_zero"})
    return rows, {**dict(coverage), "discovery_sign_positive": sign_counts["positive"],
                  "discovery_sign_negative": sign_counts["negative"],
                  "discovery_sign_zero": sign_counts["uncertain"]}


def paired_prompt_bootstrap(gold: list[dict[str, Any]], responses: list[dict[str, Any]],
                            condition_a: str, condition_b: str, seed: int = 0,
                            n_boot: int = 2000) -> dict[str, Any]:
    if n_boot < 1:
        raise ValueError("Need positive bootstrap replicates")
    score_a = score_responses(gold, [r for r in responses if r["condition"] == condition_a],
                              strong_edge_min_abs=float("inf"))
    score_b = score_responses(gold, [r for r in responses if r["condition"] == condition_b],
                              strong_edge_min_abs=float("inf"))
    by_a = {r["question_id"]: r["top1_correct"] for r in score_a["paired_rows"]}
    by_b = {r["question_id"]: r["top1_correct"] for r in score_b["paired_rows"]}
    clusters: dict[str, list[float]] = defaultdict(list)
    for g in gold:
        qid = g["question_id"]
        clusters[str(g["prompt_pair_id"])].append(by_a[qid] - by_b[qid])
    prompts = sorted(clusters)
    observed = mean([d for rows in clusters.values() for d in rows]) if prompts else None
    if not prompts:
        return {"condition_a": condition_a, "condition_b": condition_b,
                "difference": None, "prompt_clusters": 0, "ci_95": None}
    rng = random.Random(seed)
    draws = []
    for _ in range(n_boot):
        sampled = [rng.choice(prompts) for _ in prompts]
        values = [d for prompt in sampled for d in clusters[prompt]]
        draws.append(mean(values))
    draws.sort()
    lo = draws[int(0.025 * (n_boot - 1))]
    hi = draws[int(0.975 * (n_boot - 1))]
    return {"condition_a": condition_a, "condition_b": condition_b,
            "difference": observed, "prompt_clusters": len(prompts),
            "question_count": len(gold), "ci_95": [lo, hi],
            "bootstrap_seed": seed, "bootstrap_replicates": n_boot}


def selection_relation_coverage(graphs: list[dict[str, Any]],
                                selections: list[dict[str, Any]],
                                selection_manifest: dict[str, Any],
                                questions: list[dict[str, Any]],
                                refs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Macro measured-edge coverage and strict target-plus-presented-source support."""
    arities = {len(q["candidates"]) for q in questions}
    if len(arities) > 1:
        raise ValueError("Selection coverage requires separate candidate-count populations")
    candidate_count = next(iter(arities), 4)
    for field in ("model_revision", "clt_revision", "pair_dataset_sha256", "position_policy"):
        if not selection_manifest.get(field):
            raise ValueError(f"Selection handoff missing {field}")
    graph_by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for graph in graphs:
        validate_graph(graph)
        for field in ("model_revision", "clt_revision", "pair_dataset_sha256", "position_policy"):
            if graph[field] != selection_manifest[field]:
                raise ValueError(f"Graph/selection {field} mismatch")
        graph_by_prompt[str(graph["prompt_pair_id"])].append(graph)
    refs_by_id = {r["question_id"]: r for r in refs}
    questions_by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for q in questions:
        questions_by_prompt[str(q["prompt_pair_id"])].append(q)
    out = []
    for selection in selections:
        prompt_id = str(selection["prompt_pair_id"])
        retained = {key_string(k) for k in selection["retained_occurrences"]}
        if len(retained) != len(selection["retained_occurrences"]):
            raise ValueError("Selection has duplicate occurrence keys")
        target_groups: dict[str, list[int]] = defaultdict(list)
        n_edges = 0
        for graph in graph_by_prompt[prompt_id]:
            for source, target, value in _computed_feature_edges(graph):
                if value == 0:
                    continue
                source_occ = occurrence_key(graph["pair_dataset_sha256"], graph["prompt_pair_id"],
                                            source, graph["model_revision"], graph["clt_revision"])
                target_occ = occurrence_key(graph["pair_dataset_sha256"], graph["prompt_pair_id"],
                                            target, graph["model_revision"], graph["clt_revision"])
                target_groups[key_string(target_occ)].append(int(key_string(source_occ) in retained and
                                                                key_string(target_occ) in retained))
                n_edges += 1
        target_means = [mean(group) for group in target_groups.values()]
        supported = 0
        for question in questions_by_prompt[prompt_id]:
            ref = refs_by_id[question["question_id"]]
            target_occ = [selection_manifest["pair_dataset_sha256"], question["prompt_pair_id"],
                          question["target"]["position"], ref["target_feature_key"]]
            occs = [target_occ]
            ref_candidates = {r["label"]: r["feature_key"] for r in ref["candidate_feature_keys"]}
            for candidate in question["candidates"]:
                occs.append([selection_manifest["pair_dataset_sha256"], question["prompt_pair_id"],
                             candidate["source_position"], ref_candidates[candidate["label"]]])
            supported += int(all(key_string(occ) in retained for occ in occs))
        out.append({"run_id": selection.get("run_id"), "prompt_pair_id": selection["prompt_pair_id"],
                    "method_version": selection["method_version"],
                    "requested_budget": selection["requested_budget"],
                    "achieved_occurrences": selection["achieved_occurrences"],
                    "reference_relation_edges": n_edges,
                    "reference_relation_targets": len(target_groups),
                    "induced_reference_relation_coverage_prompt_target_macro":
                        mean(target_means) if target_means else None,
                    "candidate_count": candidate_count,
                    "questions": len(questions_by_prompt[prompt_id]),
                    "fully_supported_questions": supported,
                    "question_coverage": supported / len(questions_by_prompt[prompt_id])
                    if questions_by_prompt[prompt_id] else None})
        if candidate_count == 4:
            out[-1].update({"four_way_questions": out[-1]["questions"],
                           "fully_supported_four_way_questions": supported,
                           "four_way_question_coverage": out[-1]["question_coverage"]})
    return out


def interpretation_cost_requirements(selections: list[dict[str, Any]],
                                     selection_manifest: dict[str, Any],
                                     questions: list[dict[str, Any]],
                                     refs: list[dict[str, Any]],
                                     descriptions: dict[str, dict[str, Any]],
                                     conditions: tuple[str, ...] =
                                     ("metadata", "activation", "relational", "shuffled")) -> list[dict[str, Any]]:
    """Cold-start key demand by method; no invented token or graph-acquisition cost."""
    dataset_hash = selection_manifest.get("pair_dataset_sha256")
    if not dataset_hash:
        raise ValueError("Selection dataset hash missing")
    refs_by_id = {r["question_id"]: r for r in refs}
    questions_by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for question in questions:
        questions_by_prompt[str(question["prompt_pair_id"])].append(question)
    groups: dict[tuple[str, Any], list[dict[str, Any]]] = defaultdict(list)
    for selection in selections:
        groups[(selection["method_version"], selection["requested_budget"])].append(selection)
    out = []
    for (method, budget), rows in sorted(groups.items(), key=lambda item: str(item[0])):
        identities: set[str] = set()
        neighboring: set[str] = set()
        question_keys: set[tuple[str, str]] = set()
        missing_activation: set[str] = set()
        missing_role: set[str] = set()
        for row in rows:
            retained = {key_string(occ) for occ in row["retained_occurrences"]}
            for occurrence in row["retained_occurrences"]:
                if occurrence[0] != dataset_hash or str(occurrence[1]) != str(row["prompt_pair_id"]):
                    raise ValueError("Selection occurrence dataset/prompt mismatch")
                identity = key_string(occurrence[3])
                identities.add(identity)
                description = descriptions.get(identity, {})
                if not description.get("activation_condition"):
                    missing_activation.add(identity)
                if not description.get("cross_layer_role"):
                    missing_role.add(identity)
                provenance = description.get("provenance") or {}
                neighboring.update(provenance.get("neighbor_feature_keys") or [])
            for question in questions_by_prompt[str(row["prompt_pair_id"])]:
                ref = refs_by_id[question["question_id"]]
                target_occ = [dataset_hash, question["prompt_pair_id"],
                              question["target"]["position"], ref["target_feature_key"]]
                candidate_keys = {candidate["label"]: candidate["feature_key"]
                                  for candidate in ref["candidate_feature_keys"]}
                occs = [target_occ] + [[dataset_hash, question["prompt_pair_id"],
                                        c["source_position"], candidate_keys[c["label"]]]
                                       for c in question["candidates"]]
                if all(key_string(occ) in retained for occ in occs):
                    question_keys.update((question["question_id"], condition)
                                         for condition in conditions)
        out.append({"method_version": method, "requested_budget": budget,
                    "selection_rows": len(rows), "unique_selected_identity_keys": len(identities),
                    "required_activation_description_keys": len(identities | neighboring),
                    "required_role_description_keys": len(identities),
                    "neighbor_activation_description_keys": len(neighboring),
                    "missing_activation_description_keys": len(missing_activation),
                    "missing_role_description_keys": len(missing_role),
                    "cold_start_evaluator_condition_keys_on_strict_support": len(question_keys),
                    "graph_acquisition_cost": None,
                    "cost_scope": "raw_required_keys_no_token_price_or_upstream_cost"})
    return out
