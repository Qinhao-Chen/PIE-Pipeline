#!/usr/bin/env python3
"""Exercise all three CLT metrics through the CLI with synthetic, scripted data."""

from __future__ import annotations

import argparse
import hashlib
import json
import contextlib
import io
import sys
from pathlib import Path

from .api_client import atomic_json, sha256, utc_now
from .fixture_graph import CLT, DATASET_HASH, MODEL, graph
from .relation import (feature_key, key_string, read_jsonl, render_evaluator_messages,
                      write_jsonl)
from .three_metrics import METRIC_PROTOCOL
from . import run_task_b

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("Choose a new empty output directory; existing fixtures are immutable")
    output.mkdir(parents=True, exist_ok=True)
    graphs = [graph(1, "three-discovery", [0.7, -0.4, 0.04, 0.0]),
              graph(2, "three-positive", [0.9, 0.2, 0.03, 0.0]),
              graph(4, "three-negative", [-0.85, 0.15, 0.02, 0.0]),
              graph(5, "three-weak", [0.04, -0.03, 0.02, 0.0]),
              graph(6, "three-tie", [0.7, -0.7, 0.03, 0.0])]
    for record in graphs:
        for edge in record["edges"]:
            if edge["value"] == 0:
                edge["measurement_status"] = "computed_zero"
    split = {"status": "fixture", "pair_dataset_sha256": DATASET_HASH,
             "pair_ids": {"discovery": [1], "development": [2, 4, 5, 6], "final": [3]},
             "final_sealed": True}
    descriptions = {key_string(feature_key(MODEL, CLT, node)): {
        "activation_condition": "Synthetic activation text for this fixture feature.",
        "cross_layer_role": "Synthetic downstream role; not a real feature explanation.",
        "scope_and_uncertainty": "Scripted fixture only; no empirical interpretation evidence."}
        for node in graphs[0]["nodes"] if node["type"] == "feature"}
    write_jsonl(output / "graphs.jsonl", graphs)
    atomic_json(output / "split.json", split)
    atomic_json(output / "descriptions.json", descriptions)
    commands = []

    def run(*arguments):
        # Exercise the real CLI handlers in process; no child process/environment capture.
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            result = run_task_b.main([str(arg) for arg in arguments])
        commands.append({"command": arguments[0], "exit_code": result})
        if result:
            atomic_json(output / "commands.json", commands)
            raise RuntimeError(captured.getvalue())

    run("freeze-metrics-policy", "--development-graphs", output / "graphs.jsonl",
        "--split", output / "split.json", "--relation-min-abs", "0.1",
        "--winner-min-gap", "0.05", "--sign-min-abs", "0.3", "--output", output / "policy.json")
    bank = output / "bank"
    run("questions", "--graphs", output / "graphs.jsonl", "--split", output / "split.json",
        "--descriptions", output / "descriptions.json", "--metric-policy", output / "policy.json",
        "--fixture", "--seed", "7", "--output-dir", bank)
    public = read_jsonl(bank / "public_questions.jsonl")
    refs = {r["question_id"]: r for r in read_jsonl(bank / "private_refs.jsonl")}
    gold = {g["question_id"]: g for g in read_jsonl(bank / "private_gold.jsonl")}
    mocks = []
    for question in public:
        g = gold[question["question_id"]]
        for condition in ("metadata", "activation", "relational", "shuffled"):
            messages, assignment = render_evaluator_messages(question, refs[question["question_id"]],
                                                               descriptions, condition)
            answer = {"choice": g["correct_label"] or "A",
                      "signs": {x: "uncertain" for x in "ABCD"},
                      "relation_probabilities": {x: 0.5 for x in "ABCD"}}
            if condition == "relational":
                answer["signs"] = {x: g["candidate_values_and_signs"][x]["sign"] for x in "ABCD"}
                answer["relation_probabilities"] = {
                    x: 0.9 if g["candidate_values_and_signs"][x]["relation_present"] else 0.1 for x in "ABCD"}
            else:
                answer["choice"] = next(x for x in "ABCD" if x != g["correct_label"])
            mocks.append({"question_id": question["question_id"], "condition": condition,
                          "status": "mock_fixture", "answer": answer, "metric_protocol": METRIC_PROTOCOL,
                          "role_assignment": assignment, "rendered_messages_sha256": sha256(messages)})
    write_jsonl(output / "mock_responses.jsonl", mocks)
    run("evaluate", "--questions", bank / "public_questions.jsonl", "--refs", bank / "private_refs.jsonl",
        "--question-meta", bank / "question_meta.json", "--descriptions", output / "descriptions.json",
        "--output", output / "never_sent.jsonl", "--dry-run")
    run("score", "--gold", bank / "private_gold.jsonl", "--responses", output / "mock_responses.jsonl",
        "--question-meta", bank / "question_meta.json", "--status", "fixture", "--output", output / "scores.json")
    run("baselines", "--discovery-graphs", output / "graphs.jsonl", "--split", output / "split.json",
        "--questions", bank / "public_questions.jsonl", "--refs", bank / "private_refs.jsonl",
        "--gold", bank / "private_gold.jsonl", "--question-meta", bank / "question_meta.json",
        "--status", "fixture", "--output-dir", output / "baselines")
    run("paired-bootstrap", "--gold", bank / "private_gold.jsonl", "--responses", output / "mock_responses.jsonl",
        "--question-meta", bank / "question_meta.json", "--condition-a", "relational",
        "--condition-b", "activation", "--replicates", "200", "--seed", "7",
        "--status", "fixture", "--output", output / "paired.json")
    atomic_json(output / "commands.json", commands)
    code_names = ("api_client.py", "relation.py", "analysis.py", "three_metrics.py", "run_task_b.py",
                  "fixture_graph.py", "fixture.py")
    files = {str(p.relative_to(output)): file_hash(p) for p in sorted(output.rglob("*")) if p.is_file()}
    atomic_json(output / "manifest.json", {
        "schema_version": "pie-parallel-v1", "owner": "B", "artifact_id": output.name,
        "artifact_kind": "three_metrics_pipeline_fixture", "created_at_utc": utc_now(), "status": "fixture",
        "base_commit": None,
        "changed_code_hashes": {name: file_hash(HERE / name) for name in code_names},
        "method_version_or_description_protocol": METRIC_PROTOCOL,
        "model_revision": MODEL, "clt_revision": CLT, "task": "synthetic_ioi_like",
        "pair_dataset_sha256": DATASET_HASH,
        "split_manifest_path_and_sha256": {"path": "split.json", "sha256": files["split.json"]},
        "seeds": {"question": 7, "bootstrap": 7}, "protocol_label": "synthetic_fixture_not_empirical",
        "evaluator_path_and_hash": None, "intervention_metadata": None,
        "inputs": [], "files": files, "cost_log_path": None,
        "resource_usage": {"network_requests": 0, "subject_model_forwards": 0},
        "missing_fields_or_limitations": ["Synthetic graphs and deliberately scripted answers; no research result",
                                          "Fixture thresholds are software examples, not calibrated research thresholds"]})
    print(json.dumps({"fixture_dir": str(output), "questions": len(public), "responses": len(mocks),
                      "network_requests": 0, "subject_model_forwards": 0}))


if __name__ == "__main__":
    main()
