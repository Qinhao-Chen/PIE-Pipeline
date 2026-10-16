#!/usr/bin/env python3
"""Task B CLI: CPU graph audit, sealed questions, API descriptions and evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickletools
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .api_client import APIClient, Budget, DEFAULT_BASE_URL, atomic_json, canonical, sha256, utc_now
from .analysis import (discovery_average_predictor, paired_prompt_bootstrap,
                      interpretation_cost_requirements, selection_relation_coverage)
from .relation import (DESCRIPTION_VERSION, PROMPT_VERSION, build_question_bank, feature_key,
                      inspect_legacy_graph, key_string, make_provisional_split,
                      make_role_messages, read_jsonl, render_evaluator_messages,
                      score_responses, validate_evaluator_answer, validate_graph,
                      validate_role_record, write_jsonl)
from .three_metrics import (METRIC_PROTOCOL, SUPPORTED_METRIC_PROTOCOLS, build_three_metric_bank, freeze_metric_policy,
                           paired_three_metric_bootstrap, score_three_metrics)


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def file_hash(path: Path) -> str:
    stream = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            stream.update(block)
    return stream.hexdigest()


def load_dict(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def verify_metric_file(meta: dict[str, Any], path: Path, filename: str) -> None:
    if meta.get("metric_protocol") not in SUPPORTED_METRIC_PROTOCOLS:
        raise ValueError("Expected a three-metric question manifest")
    if meta.get("file_sha256", {}).get(filename) != file_hash(path):
        raise ValueError(f"Three-metric bank file/hash mismatch: {filename}")


def metric_scoring_meta(args: argparse.Namespace, gold: list[dict[str, Any]]) -> dict[str, Any] | None:
    meta = load_dict(args.question_meta) if args.question_meta else None
    is_three = ((meta or {}).get("metric_protocol") in SUPPORTED_METRIC_PROTOCOLS or
                any(g.get("metric_protocol") in SUPPORTED_METRIC_PROTOCOLS for g in gold))
    if not is_three:
        return None
    if meta is None:
        raise ValueError("Three-metric scoring requires --question-meta")
    verify_metric_file(meta, args.gold, "private_gold.jsonl")
    if args.status != meta.get("status"):
        raise ValueError("Result status must match the three-metric question bank status")
    if any(g.get("metric_protocol") != meta["metric_protocol"] or
           len(g["candidate_values_and_signs"]) != meta.get("candidate_count", 4) for g in gold):
        raise ValueError("Question manifest and gold protocol/candidate count mismatch")
    return meta


def cmd_freeze_metric_policy(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise ValueError("Frozen policy already exists; choose a new versioned output")
    policy = freeze_metric_policy(
        read_jsonl(args.development_graphs), load_dict(args.split),
        relation_min_abs=args.relation_min_abs, winner_min_gap=args.winner_min_gap,
        sign_min_abs=args.sign_min_abs, split_sha256=file_hash(args.split),
        graph_sha256=file_hash(args.development_graphs),
        per_target=load_dict(args.per_target) if args.per_target else None)
    atomic_json(args.output, policy)
    print(json.dumps({"policy": str(args.output), "metric_protocol": METRIC_PROTOCOL,
                      "development_prompts": len(policy["development_prompt_ids"]),
                      "network_requests": 0}))


def verify_graph_handoff(manifest_path: Path, graph_path: Path,
                         dataset_hash: str, adapter_provenance: Path | None) -> dict[str, Any]:
    manifest = load_dict(manifest_path)
    if manifest.get("schema_version") != "pie-parallel-v1" or manifest.get("owner") != "C":
        raise ValueError("Graph manifest is not a C graph handoff")
    if manifest.get("status") not in ("real_pilot", "real_final"):
        raise ValueError("Graph handoff has no real measured status")
    if manifest.get("pair_dataset_sha256") != dataset_hash:
        raise ValueError("Graph handoff and split dataset hashes disagree")
    if not manifest.get("measured_support_policy") or not manifest.get("edge_semantics"):
        raise ValueError("Graph handoff lacks measured support or edge semantics")
    listed = [item.get("sha256") if isinstance(item, dict) else item
              for item in (manifest.get("files") or {}).values()]
    adapted_hash = file_hash(graph_path)
    if adapted_hash not in listed:
        if adapter_provenance is None:
            raise ValueError("Adapted graph needs a B adapter-provenance receipt")
        receipt = load_dict(adapter_provenance)
        if (receipt.get("schema_version") != "pie-B-graph-adapter-v1" or
                receipt.get("source_manifest_sha256") != file_hash(manifest_path) or
                receipt.get("source_graph_sha256") not in listed or
                receipt.get("adapted_graph_sha256") != adapted_hash):
            raise ValueError("B graph-adapter provenance mismatch")
        source_path = Path(receipt["source_graph_path"])
        if file_hash(source_path) != receipt["source_graph_sha256"]:
            raise ValueError("C source graph file/hash mismatch")
        code_path = Path(receipt["adapter_code_path"])
        if not code_path.resolve().is_relative_to(HERE) or file_hash(code_path) != receipt["adapter_code_sha256"]:
            raise ValueError("B adapter code path/hash mismatch")
    return manifest


def api(args: argparse.Namespace) -> APIClient:
    budget = Budget(args.max_requests, args.max_total_tokens, args.max_usd,
                    args.input_rate, args.output_rate, state_path=args.budget_state)
    return APIClient(model=args.model, base_url=args.base_url, key_env=args.key_env,
                     cache_dir=args.cache_dir, ledger_path=args.usage_ledger, budget=budget,
                     timeout_s=args.timeout, max_retries=args.max_retries,
                     max_concurrent=args.max_concurrent, reasoning_effort=args.reasoning_effort,
                     omit_temperature=args.omit_temperature)


def cmd_split(args: argparse.Namespace) -> None:
    pairs = read_jsonl(args.pairs)
    split = make_provisional_split(pairs, file_hash(args.pairs), args.seed)
    atomic_json(args.output, split)
    print(json.dumps({"split": str(args.output), "counts": {k: len(v) for k, v in split["pair_ids"].items()},
                      "exact_duplicate_conflicts": len(split["exact_duplicate_conflicts"])}))


def cmd_import_cantor(args: argparse.Namespace) -> None:
    raw = load_dict(args.input)
    converted = {}
    for paired_id, description in raw.items():
        if not isinstance(description, str):
            continue
        z = int(paired_id)
        w = (math.isqrt(8 * z + 1) - 1) // 2
        t = w * (w + 1) // 2
        feat = z - t
        layer = w - feat
        key = [args.model_revision, args.clt_revision, layer, feat]
        converted[key_string(key)] = {"activation_condition": description,
                                      "cross_layer_role": "", "scope_and_uncertainty": "",
                                      "source_sha256": file_hash(args.input),
                                      "provenance_status": "unverified_prior_description"}
    atomic_json(args.output, converted)
    print(json.dumps({"imported": len(converted), "status": "provenance_unverified",
                      "output": str(args.output)}))


def cmd_questions(args: argparse.Namespace) -> None:
    split = load_dict(args.split)
    graphs = read_jsonl(args.graphs)
    descriptions = load_dict(args.descriptions) if args.descriptions else None
    status = "fixture" if args.fixture else "audit_only"
    verified = False
    graph_manifest_hash = None
    if args.graph_manifest:
        manifest = verify_graph_handoff(args.graph_manifest, args.graphs,
                                        split["pair_dataset_sha256"], args.adapter_provenance)
        graph_manifest_hash = file_hash(args.graph_manifest)
        status = manifest["status"]
        verified = True
    if args.fixture and args.graph_manifest:
        raise ValueError("Fixture mode cannot take a real graph handoff")
    if status == "real_final" and split.get("status") == "provisional_cpu_only":
        raise ValueError("Provisional split cannot produce real-final questions")
    policy = load_dict(args.metric_policy) if args.metric_policy else None
    if policy is not None:
        if args.output_dir.exists() and any(args.output_dir.iterdir()):
            raise ValueError("Three-metric bank already exists; choose a new versioned output directory")
        public, refs, gold, counts = build_three_metric_bank(
            graphs, split, policy=policy, question_split=args.question_split, seed=args.seed,
            descriptions=descriptions, split_sha256=file_hash(args.split))
    else:
        public, refs, gold, counts = build_question_bank(
            graphs, split, question_split=args.question_split, seed=args.seed,
            min_abs=args.min_abs, min_gap=args.min_gap, descriptions=descriptions)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "public_questions.jsonl", public)
    write_jsonl(args.output_dir / "private_refs.jsonl", refs)
    write_jsonl(args.output_dir / "private_gold.jsonl", gold)
    meta = {"status": status, "verified_graph_evidence": verified,
            "graph_manifest_sha256": graph_manifest_hash,
            "graph_input_sha256": file_hash(args.graphs), "split_sha256": file_hash(args.split),
            "description_input_sha256": file_hash(args.descriptions) if args.descriptions else None,
            "question_split": args.question_split, "min_abs": args.min_abs, "min_gap": args.min_gap,
            "seed": args.seed, "counts": counts, "prompt_version": PROMPT_VERSION,
            "reference_visibility": "private_gold_never_read_by_evaluator"}
    if policy is not None:
        meta.pop("min_abs")
        meta.pop("min_gap")
        meta.update({"metric_protocol": METRIC_PROTOCOL, "prompt_version": METRIC_PROTOCOL,
                     "metric_policy_sha256": file_hash(args.metric_policy),
                     "metric_policy_content_sha256": sha256(policy),
                     "reference_context": policy["reference_context"],
                     "candidate_sampling": "seeded_metadata_within_matched_measured_support",
                     "reference_population": "cached_graph_measured_support_not_whole_model",
                     "file_sha256": {name: file_hash(args.output_dir / name) for name in
                                     ("public_questions.jsonl", "private_refs.jsonl", "private_gold.jsonl")},
                     "code_sha256": {name: file_hash(HERE / name) for name in
                                     ("relation.py", "three_metrics.py", "run_task_b.py")}})
    atomic_json(args.output_dir / "question_meta.json", meta)
    print(json.dumps({"questions": len(public), "counts": counts, "output_dir": str(args.output_dir)}))


def discovery_packets(graphs: list[dict[str, Any]], split: dict[str, Any],
                      activation: dict[str, dict[str, str]], max_edges: int) -> dict[str, dict[str, Any]]:
    allowed = set(split["pair_ids"]["discovery"])
    packets: dict[str, dict[str, Any]] = defaultdict(lambda: {"evidence": [], "discovery_prompt_ids": []})
    for graph in graphs:
        validate_graph(graph)
        if graph["pair_dataset_sha256"] != split["pair_dataset_sha256"]:
            raise ValueError("Graph dataset hash mismatch")
        if graph["prompt_pair_id"] not in allowed:
            continue
        nodes = {node["id"]: node for node in graph["nodes"]}
        for edge in graph["edges"]:
            if edge["measurement_status"] == "unmeasured_omitted":
                continue
            source, target = nodes[edge["source"]], nodes[edge["target"]]
            if source["type"] != "feature" or target["type"] != "feature":
                continue
            if source["layer"] >= target["layer"]:
                continue
            source_key = key_string(feature_key(graph["model_revision"], graph["clt_revision"], source))
            target_key = key_string(feature_key(graph["model_revision"], graph["clt_revision"], target))
            packet = packets[source_key]
            packet["discovery_prompt_ids"].append(graph["prompt_pair_id"])
            packet["evidence"].append({"graph_id": graph["graph_id"],
                                       "prompt_pair_id": graph["prompt_pair_id"],
                                       "source_position": source["position"],
                                       "target_layer": target["layer"],
                                       "target_position": target["position"],
                                       "target_feature_key": feature_key(graph["model_revision"],
                                                                         graph["clt_revision"], target),
                                       "target_activation_label": activation.get(target_key, {}).get("activation_condition"),
                                       "signed_value": edge["value"],
                                       "edge_semantics": graph["edge_semantics"],
                                       "normalization": graph["normalization"]})
    for packet in packets.values():
        packet["discovery_prompt_ids"] = sorted(set(packet["discovery_prompt_ids"]))
        packet["evidence"] = sorted(packet["evidence"], key=lambda e: -abs(e["signed_value"]))[:max_edges]
    return dict(packets)


def cmd_describe(args: argparse.Namespace) -> None:
    split = load_dict(args.split)
    graphs = read_jsonl(args.graphs)
    activation = load_dict(args.activation)
    packets = discovery_packets(graphs, split, activation, args.max_edges)
    if args.dry_run:
        print(json.dumps({"eligible_source_identities": len(packets),
                          "with_activation_labels": sum(k in activation for k in packets),
                          "max_edges_per_packet": args.max_edges, "network_requests": 0}))
        return
    if not args.graph_manifest:
        raise ValueError("Real API descriptions require a verified C graph handoff manifest")
    verify_graph_handoff(args.graph_manifest, args.graphs,
                         split["pair_dataset_sha256"], args.adapter_provenance)
    client = api(args)
    output = load_dict(args.output) if args.output.exists() else {}
    source_hashes = {"discovery_graph_sha256": file_hash(args.graphs),
                     "split_sha256": file_hash(args.split),
                     "activation_source_sha256": file_hash(args.activation)}
    generation_hash = sha256({**source_hashes, "prompt_version": DESCRIPTION_VERSION,
                              "api_model": args.model, "base_url": args.base_url,
                              "max_output_tokens": args.max_output_tokens, "max_edges": args.max_edges,
                              "reasoning_effort": args.reasoning_effort,
                              "omit_temperature": args.omit_temperature})
    if any((r.get("provenance") or {}).get("generation_configuration_hash") != generation_hash
           for r in output.values()):
        raise ValueError("Description task/evidence/API configuration changed; use a separate output namespace")
    missing = 0
    for source_key, packet in sorted(packets.items()):
        base = activation.get(source_key, {}).get("activation_condition")
        if not base:
            missing += 1
            continue
        if source_key in output:
            continue
        messages = make_role_messages(base, packet)
        try:
            result = client.chat(messages, prompt_version=DESCRIPTION_VERSION,
                                 max_output_tokens=args.max_output_tokens)
            fields = validate_role_record(result.parsed, base)
            output[source_key] = {**fields, "provenance": {
                "discovery_prompt_ids": packet["discovery_prompt_ids"],
                **source_hashes,
                "generation_configuration_hash": generation_hash,
                "neighbor_feature_keys": sorted({key_string(e["target_feature_key"])
                                                  for e in packet["evidence"]}),
                "prompt_version": DESCRIPTION_VERSION, "api_model": args.model,
                "api_cache_key": result.cache_key, "api_usage": result.usage}}
            atomic_json(args.output, output)
        except Exception as exc:
            print(json.dumps({"feature_key": source_key, "status": "failed", "error": str(exc)}))
            break
    print(json.dumps({"described": len(output), "missing_activation_labels": missing,
                      "eligible_source_identities": len(packets), "output": str(args.output)}))


def cmd_evaluate(args: argparse.Namespace) -> None:
    meta = load_dict(args.question_meta)
    if meta.get("status") == "fixture" and not args.dry_run:
        raise ValueError("Fixture questions cannot be sent to a real API")
    if not args.dry_run and (meta.get("status") not in ("real_pilot", "real_final") or
                             not meta.get("verified_graph_evidence")):
        raise ValueError("Real API evaluation requires a verified real-pilot question manifest")
    questions = read_jsonl(args.questions)
    three_metrics = meta.get("metric_protocol") in SUPPORTED_METRIC_PROTOCOLS
    if three_metrics:
        verify_metric_file(meta, args.questions, "public_questions.jsonl")
        verify_metric_file(meta, args.refs, "private_refs.jsonl")
    if len(set(args.conditions)) != len(args.conditions):
        raise ValueError("Evaluation conditions must be unique")
    refs = {r["question_id"]: r for r in read_jsonl(args.refs)}
    desc = load_dict(args.descriptions)
    work = []
    available = set(meta.get("available_conditions", ("metadata", "activation", "relational", "shuffled")))
    unavailable = sorted(set(args.conditions) - available)
    for question in questions:
        for condition in args.conditions:
            if condition not in available:
                continue
            messages, assignment = render_evaluator_messages(question, refs[question["question_id"]],
                                                              desc, condition, args.shuffle_seed)
            labels = tuple(c["label"] for c in question["candidates"])
            work.append((question["question_id"], condition, messages, assignment, labels))
    bound = sum(len(canonical(messages)) + 64 * len(messages) + 128 + args.max_output_tokens
                for _, _, messages, _, _ in work)
    if args.dry_run:
        print(json.dumps({"requests": len(work), "conservative_token_reservation": bound,
                          "max_output_tokens_per_request": args.max_output_tokens,
                          "conditions": [c for c in args.conditions if c in available],
                          "unavailable_conditions": unavailable, "network_requests": 0}))
        return
    config_hash = sha256({"model": args.model, "base_url": args.base_url,
                          "max_output_tokens": args.max_output_tokens,
                          "reasoning_effort": args.reasoning_effort,
                          "omit_temperature": args.omit_temperature,
                          "description_sha256": file_hash(args.descriptions),
                          "shuffle_seed": args.shuffle_seed, "metric_protocol": meta.get("metric_protocol")})
    completed = {(r["question_id"], r["condition"]) for r in read_jsonl(args.output)
                 if r.get("status") == "ok" and
                 (not three_metrics or r.get("evaluation_configuration_hash") == config_hash)} \
        if args.output.exists() else set()
    remaining = [item for item in work if item[:2] not in completed]
    client = api(args)

    def one(item: tuple[str, str, list[dict[str, str]], dict[str, str], tuple[str, ...]]) -> dict[str, Any]:
        question_id, condition, messages, assignment, labels = item
        try:
            prompt_version = (meta["metric_protocol"] if three_metrics else PROMPT_VERSION) + "-" + condition
            result = client.chat(messages, prompt_version=prompt_version,
                                 max_output_tokens=args.max_output_tokens)
            answer = validate_evaluator_answer(result.parsed, require_probabilities=three_metrics,
                                               expected_labels=labels)
            return {"question_id": question_id, "condition": condition, "status": "ok",
                    "answer": answer, "usage": result.usage, "cache_key": result.cache_key,
                    "cache_hit": result.cache_hit, "role_assignment": assignment,
                    "rendered_chars": sum(len(m["content"]) for m in messages),
                    "metric_protocol": meta.get("metric_protocol"),
                    "evaluation_configuration_hash": config_hash}
        except Exception as exc:
            return {"question_id": question_id, "condition": condition, "status": "failed",
                    "answer": None, "error": str(exc), "role_assignment": assignment}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.max_concurrent) as pool, args.output.open("a", encoding="utf-8") as out:
        for future in as_completed([pool.submit(one, item) for item in remaining]):
            out.write(canonical(future.result()).decode() + "\n")
            out.flush()
    print(json.dumps({"scheduled": len(remaining), "output": str(args.output)}))


def cmd_score(args: argparse.Namespace) -> None:
    gold = read_jsonl(args.gold)
    meta = metric_scoring_meta(args, gold)
    if meta and args.output.exists():
        raise ValueError("Three-metric score output exists; choose a new versioned output")
    responses = list({(r["question_id"], r["condition"]): r
                      for r in read_jsonl(args.responses)}.values())
    results = {}
    conditions = (meta.get("available_conditions", ["metadata", "activation", "relational", "shuffled"]) if meta else
                  sorted({r["condition"] for r in responses}))
    if meta and any(r["condition"] not in conditions for r in responses):
        raise ValueError("Response uses an unavailable question-bank condition")
    conditions = sorted(set(conditions) | {r["condition"] for r in responses})
    for condition in conditions:
        rows = [r for r in responses if r["condition"] == condition]
        result = (score_three_metrics(gold, rows, metric_protocol=meta["metric_protocol"],
                                      candidate_count=meta.get("candidate_count", 4)) if meta else
                  score_responses(gold, rows, strong_edge_min_abs=args.strong_edge_min_abs))
        result.pop("paired_rows")
        results[condition] = result
    atomic_json(args.output, {"status": args.status, "results": results,
                              "metric_protocol": meta["metric_protocol"] if meta else PROMPT_VERSION,
                              "unavailable_conditions": meta.get("unavailable_conditions", []) if meta else [],
                              "question_meta_sha256": file_hash(args.question_meta) if args.question_meta else None,
                              "gold_sha256": file_hash(args.gold),
                              "responses_sha256": file_hash(args.responses)})
    print(json.dumps(results, sort_keys=True))


def cmd_cost(args: argparse.Namespace) -> None:
    entries = read_jsonl(args.usage_ledger)
    usage = Counter()
    for e in entries:
        usage["entries"] += 1
        usage["cache_hits"] += int(e.get("cache_hit", False))
        usage["requests"] += int(not e.get("cache_hit", False))
        if not e.get("cache_hit", False):
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                usage[key] += (e.get("usage") or {}).get(key) or 0
    print(json.dumps({"raw_usage": dict(usage), "ledger": str(args.usage_ledger)}))


def cmd_baselines(args: argparse.Namespace) -> None:
    split = load_dict(args.split)
    questions = read_jsonl(args.questions)
    refs = read_jsonl(args.refs)
    gold = read_jsonl(args.gold)
    meta = metric_scoring_meta(args, gold)
    if meta:
        if args.output_dir.exists() and any(args.output_dir.iterdir()):
            raise ValueError("Three-metric baseline output exists; choose a new versioned directory")
        verify_metric_file(meta, args.questions, "public_questions.jsonl")
        verify_metric_file(meta, args.refs, "private_refs.jsonl")
        if file_hash(args.split) != meta.get("split_sha256"):
            raise ValueError("Baseline and question bank split hashes disagree")
        for graph in read_jsonl(args.discovery_graphs):
            if any(graph.get(k) != v for k, v in meta["reference_context"].items()):
                raise ValueError("Discovery baseline and question bank reference contexts disagree")
    rows, fallback_counts = discovery_average_predictor(read_jsonl(args.discovery_graphs),
                                                         split, questions, refs)
    result = (score_three_metrics(gold, rows, metric_protocol=meta["metric_protocol"],
                                  candidate_count=meta.get("candidate_count", 4)) if meta else
              score_responses(gold, rows, strong_edge_min_abs=args.strong_edge_min_abs))
    result.pop("paired_rows")
    signs = {"positive": fallback_counts["discovery_sign_positive"],
             "negative": fallback_counts["discovery_sign_negative"],
             "uncertain": fallback_counts["discovery_sign_zero"]}
    majority = max(("positive", "negative", "uncertain"), key=lambda label: signs[label])
    strong = [g for g in gold if (g["sign_eligible"] if meta else
                                 g["strongest_abs"] >= args.strong_edge_min_abs)]
    majority_sign_accuracy = sum(g["candidate_values_and_signs"][g["correct_label"]]["sign"] == majority
                                 for g in strong) / len(strong) if strong else None
    write_jsonl(args.output_dir / "discovery_average_responses.jsonl", rows)
    arities = {len(q["candidates"]) for q in questions}
    arity = next(iter(arities)) if len(arities) == 1 else None
    atomic_json(args.output_dir / "baseline_summary.json", {
        "status": args.status, "discovery_average": result,
        "fallback_counts": fallback_counts, "random_four_way_chance": 0.25 if arity == 4 else None,
        "candidate_count": arity, "random_source_choice_chance": 1 / arity if arity and arity >= 2 else None,
        "majority_discovery_sign": majority,
        "majority_sign_accuracy_on_strong_subset": majority_sign_accuracy,
        "strong_sign_questions": len(strong),
        "discovery_graph_sha256": file_hash(args.discovery_graphs),
        "split_sha256": file_hash(args.split)})
    print(json.dumps({"baseline_summary": str(args.output_dir / "baseline_summary.json"),
                      "questions": len(rows), "fallback_counts": fallback_counts}))


def cmd_paired(args: argparse.Namespace) -> None:
    responses = list({(r["question_id"], r["condition"]): r
                      for r in read_jsonl(args.responses)}.values())
    gold = read_jsonl(args.gold)
    meta = metric_scoring_meta(args, gold)
    if meta and args.output.exists():
        raise ValueError("Three-metric bootstrap output exists; choose a new versioned output")
    if meta and (args.condition_a not in meta.get("available_conditions", ("metadata", "activation", "relational", "shuffled")) or
                 args.condition_b not in meta.get("available_conditions", ("metadata", "activation", "relational", "shuffled"))):
        raise ValueError("Bootstrap uses an unavailable question-bank condition")
    bootstrap = paired_three_metric_bootstrap if meta else paired_prompt_bootstrap
    result = bootstrap(gold, responses, args.condition_a, args.condition_b,
                       seed=args.seed, n_boot=args.replicates,
                       **({"metric_protocol": meta["metric_protocol"], "candidate_count": meta.get("candidate_count", 4)}
                          if meta else {}))
    atomic_json(args.output, {"status": args.status, "comparison": result})
    print(json.dumps(result))


def cmd_coverage(args: argparse.Namespace) -> None:
    manifest = load_dict(args.selection_manifest)
    if manifest.get("schema_version") != "pie-parallel-v1" or manifest.get("owner") not in ("A", "C", "D"):
        raise ValueError("Selection manifest is not a compatible A/C/D handoff")
    listed = [item.get("sha256") if isinstance(item, dict) else item
              for item in (manifest.get("files") or {}).values()]
    if file_hash(args.selections) not in listed:
        raise ValueError("Selection file hash missing from handoff manifest")
    rows = selection_relation_coverage(read_jsonl(args.graphs), read_jsonl(args.selections),
                                       manifest, read_jsonl(args.questions), read_jsonl(args.refs))
    write_jsonl(args.output, rows)
    print(json.dumps({"selection_rows": len(rows), "output": str(args.output)}))


def cmd_cost_requirements(args: argparse.Namespace) -> None:
    manifest = load_dict(args.selection_manifest)
    if manifest.get("schema_version") != "pie-parallel-v1" or manifest.get("owner") not in ("A", "C", "D"):
        raise ValueError("Selection manifest is not a compatible A/C/D handoff")
    listed = [item.get("sha256") if isinstance(item, dict) else item
              for item in (manifest.get("files") or {}).values()]
    if file_hash(args.selections) not in listed:
        raise ValueError("Selection file hash missing from handoff manifest")
    rows = interpretation_cost_requirements(read_jsonl(args.selections), manifest,
                                            read_jsonl(args.questions), read_jsonl(args.refs),
                                            load_dict(args.descriptions))
    for row in rows:
        row["selection_artifact_id"] = manifest.get("artifact_id")
        row["task"] = manifest.get("task")
        row["description_protocol"] = args.description_protocol
        row["question_bank_sha256"] = file_hash(args.questions)
    write_jsonl(args.output, rows)
    print(json.dumps({"method_budget_rows": len(rows), "output": str(args.output)}))


def add_api_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--key-env", default="OPENAI_API_KEY")
    parser.add_argument("--cache-dir", type=Path, default=HERE / "cache" / "api")
    parser.add_argument("--usage-ledger", type=Path, default=HERE / "exports" / "api_usage.jsonl")
    parser.add_argument("--max-requests", type=int, default=1)
    parser.add_argument("--max-total-tokens", type=int, default=1000)
    parser.add_argument("--max-usd", type=float)
    parser.add_argument("--input-rate", type=float, help="Documented USD per million input tokens")
    parser.add_argument("--output-rate", type=float, help="Documented USD per million output tokens")
    parser.add_argument("--budget-state", type=Path,
                        help="Shared persistent reservations; resume cannot reset the total cap")
    parser.add_argument("--reasoning-effort", choices=("minimal", "low", "medium", "high"))
    parser.add_argument("--omit-temperature", action="store_true")
    parser.add_argument("--max-output-tokens", type=int, default=180)
    parser.add_argument("--max-concurrent", type=int, default=1)
    parser.add_argument("--max-retries", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=25)
    parser.add_argument("--dry-run", action="store_true")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    split = sub.add_parser("split")
    split.add_argument("--pairs", type=Path, required=True)
    split.add_argument("--seed", type=int, default=0)
    split.add_argument("--output", type=Path, default=HERE / "exports" / "provisional_ioi_split.json")
    split.set_defaults(func=cmd_split)
    importer = sub.add_parser("import-cantor-descriptions")
    importer.add_argument("--input", type=Path, required=True)
    importer.add_argument("--model-revision", required=True)
    importer.add_argument("--clt-revision", required=True)
    importer.add_argument("--output", type=Path, required=True)
    importer.set_defaults(func=cmd_import_cantor)
    policy = sub.add_parser("freeze-metrics-policy", help="Freeze owner-chosen thresholds on development inputs")
    policy.add_argument("--development-graphs", type=Path, required=True)
    policy.add_argument("--split", type=Path, required=True)
    policy.add_argument("--relation-min-abs", type=float, required=True)
    policy.add_argument("--winner-min-gap", type=float, required=True)
    policy.add_argument("--sign-min-abs", type=float, required=True)
    policy.add_argument("--per-target", type=Path)
    policy.add_argument("--output", type=Path, required=True)
    policy.set_defaults(func=cmd_freeze_metric_policy)
    questions = sub.add_parser("questions")
    questions.add_argument("--graphs", type=Path, required=True)
    questions.add_argument("--split", type=Path, required=True)
    questions.add_argument("--descriptions", type=Path)
    questions.add_argument("--graph-manifest", type=Path,
                           help="Verified C handoff manifest; required to promote real graph status")
    questions.add_argument("--adapter-provenance", type=Path,
                           help="B receipt if graph JSONL was adapted from C's native format")
    questions.add_argument("--question-split", choices=("development", "final"), default="development")
    questions.add_argument("--seed", type=int, default=0)
    questions.add_argument("--min-abs", type=float, default=0.05)
    questions.add_argument("--min-gap", type=float, default=0.02)
    questions.add_argument("--metric-policy", type=Path,
                           help="Frozen three-metric policy; omit for the legacy graph-ranked question bank")
    questions.add_argument("--fixture", action="store_true")
    questions.add_argument("--output-dir", type=Path, required=True)
    questions.set_defaults(func=cmd_questions)
    describe = sub.add_parser("describe")
    describe.add_argument("--graphs", type=Path, required=True)
    describe.add_argument("--split", type=Path, required=True)
    describe.add_argument("--activation", type=Path, required=True)
    describe.add_argument("--graph-manifest", type=Path,
                          help="Verified C handoff required for any real API request")
    describe.add_argument("--adapter-provenance", type=Path)
    describe.add_argument("--max-edges", type=int, default=8)
    describe.add_argument("--output", type=Path, required=True)
    add_api_args(describe)
    describe.set_defaults(func=cmd_describe)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--questions", type=Path, required=True)
    evaluate.add_argument("--refs", type=Path, required=True)
    evaluate.add_argument("--question-meta", type=Path, required=True)
    evaluate.add_argument("--descriptions", type=Path, required=True)
    evaluate.add_argument("--conditions", nargs="+", choices=("metadata", "activation", "relational", "shuffled"),
                          default=["metadata", "activation", "relational", "shuffled"])
    evaluate.add_argument("--shuffle-seed", type=int, default=0)
    evaluate.add_argument("--output", type=Path, required=True)
    add_api_args(evaluate)
    evaluate.set_defaults(func=cmd_evaluate)
    score = sub.add_parser("score")
    score.add_argument("--gold", type=Path, required=True)
    score.add_argument("--responses", type=Path, required=True)
    score.add_argument("--question-meta", type=Path, help="Required for three-metric gold")
    score.add_argument("--strong-edge-min-abs", type=float, default=0.1)
    score.add_argument("--status", choices=("fixture", "audit_only", "real_pilot", "real_final"), required=True)
    score.add_argument("--output", type=Path, required=True)
    score.set_defaults(func=cmd_score)
    cost = sub.add_parser("cost")
    cost.add_argument("--usage-ledger", type=Path, required=True)
    cost.set_defaults(func=cmd_cost)
    baselines = sub.add_parser("baselines")
    baselines.add_argument("--discovery-graphs", type=Path, required=True)
    baselines.add_argument("--split", type=Path, required=True)
    baselines.add_argument("--questions", type=Path, required=True)
    baselines.add_argument("--refs", type=Path, required=True)
    baselines.add_argument("--gold", type=Path, required=True)
    baselines.add_argument("--question-meta", type=Path, help="Required for three-metric gold")
    baselines.add_argument("--strong-edge-min-abs", type=float, default=0.1)
    baselines.add_argument("--status", choices=("fixture", "audit_only", "real_pilot", "real_final"), required=True)
    baselines.add_argument("--output-dir", type=Path, required=True)
    baselines.set_defaults(func=cmd_baselines)
    paired = sub.add_parser("paired-bootstrap")
    paired.add_argument("--gold", type=Path, required=True)
    paired.add_argument("--responses", type=Path, required=True)
    paired.add_argument("--question-meta", type=Path, help="Required for three-metric gold")
    paired.add_argument("--condition-a", required=True)
    paired.add_argument("--condition-b", required=True)
    paired.add_argument("--seed", type=int, default=0)
    paired.add_argument("--replicates", type=int, default=2000)
    paired.add_argument("--status", choices=("fixture", "audit_only", "real_pilot", "real_final"), required=True)
    paired.add_argument("--output", type=Path, required=True)
    paired.set_defaults(func=cmd_paired)
    coverage = sub.add_parser("coverage")
    coverage.add_argument("--graphs", type=Path, required=True)
    coverage.add_argument("--selections", type=Path, required=True)
    coverage.add_argument("--selection-manifest", type=Path, required=True)
    coverage.add_argument("--questions", type=Path, required=True)
    coverage.add_argument("--refs", type=Path, required=True)
    coverage.add_argument("--output", type=Path, required=True)
    coverage.set_defaults(func=cmd_coverage)
    cost_keys = sub.add_parser("cost-requirements")
    cost_keys.add_argument("--selections", type=Path, required=True)
    cost_keys.add_argument("--selection-manifest", type=Path, required=True)
    cost_keys.add_argument("--questions", type=Path, required=True)
    cost_keys.add_argument("--refs", type=Path, required=True)
    cost_keys.add_argument("--descriptions", type=Path, required=True)
    cost_keys.add_argument("--description-protocol", default=DESCRIPTION_VERSION)
    cost_keys.add_argument("--output", type=Path, required=True)
    cost_keys.set_defaults(func=cmd_cost_requirements)
    args = parser.parse_args(argv)
    try:
        args.func(args)
        return 0
    except Exception as exc:
        print(f"task_b_error: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
