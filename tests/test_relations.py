#!/usr/bin/env python3
"""CPU-only behavioral checks for the three-metric CLT benchmark.

All subject-model measurements below are synthetic; no fixture is a research result.
The tests leave existing fixtures and published bundles untouched.
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from pie.relations.api_client import canonical
from pie.relations.fixture_graph import CLT, DATASET_HASH, MODEL, graph
from pie.relations.relation import (feature_key, key_string, render_evaluator_messages,
                      validate_evaluator_answer)
from pie.relations.three_metrics import (average_precision, build_three_metric_bank,
                           freeze_metric_policy, paired_three_metric_bootstrap,
                           score_three_metrics)


PROTOCOL = "pie-B-three-metrics-v1"
LETTERS = "ABCD"


def synthetic_graph(pair_id: int, graph_id: str, values: list[float]) -> dict:
    result = graph(pair_id, graph_id, values)
    for edge in result["edges"]:
        if edge["value"] == 0:
            edge["measurement_status"] = "computed_zero"
    return result


def digest(value: object) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


class ThreeMetricTests(unittest.TestCase):
    def setUp(self):
        self.discovery = synthetic_graph(1, "synthetic-discovery", [0.7, -0.4, 0.3, 0.1])
        self.development = synthetic_graph(2, "synthetic-development",
                                           [0.8123456, -0.5432197, 0.0812376, 0.0])
        self.split = self.make_split([2])
        self.graphs = [self.discovery, self.development]
        self.descriptions = {}
        for index, node in enumerate(self.development["nodes"]):
            if node["type"] == "feature":
                self.descriptions[key_string(feature_key(MODEL, CLT, node))] = {
                    "activation_condition": f"Synthetic concept number {index}",
                    "cross_layer_role": ("PRIVATE_TARGET_ROLE_SENTINEL" if node["id"] == "t"
                                         else f"Synthetic downstream role number {index}"),
                    "scope_and_uncertainty": "Synthetic fixture only",
                }

    @staticmethod
    def make_split(development_ids):
        return {"status": "fixture", "pair_dataset_sha256": DATASET_HASH,
                "pair_ids": {"discovery": [1], "development": development_ids, "final": [3]},
                "final_sealed": True, "seed": 0}

    def freeze(self, graphs=None, split=None, **overrides):
        graphs = self.graphs if graphs is None else graphs
        split = self.split if split is None else split
        arguments = {"relation_min_abs": 0.2, "winner_min_gap": 0.05,
                     "sign_min_abs": 0.4, "split_sha256": digest(split),
                     "graph_sha256": digest(graphs)}
        arguments.update(overrides)
        return freeze_metric_policy(graphs, split, **arguments)

    def bank(self, development=None, *, seed=17, policy=None, split=None,
             question_split="development", graphs=None):
        if graphs is None:
            development = [self.development] if development is None else development
            graphs = [self.discovery, *development]
        if split is None:
            split = self.make_split(sorted({g["prompt_pair_id"] for g in graphs
                                            if g["prompt_pair_id"] not in (1, 3)}))
        if policy is None:
            policy = self.freeze(graphs, split)
        return build_three_metric_bank(
            graphs, split, policy=policy, question_split=question_split, seed=seed,
            descriptions={key: value["activation_condition"]
                          for key, value in self.descriptions.items()},
            split_sha256=digest(split))

    @staticmethod
    def response(gold, *, correct_choice=True, correct_sign=True,
                 correct_probabilities=True, condition="activation"):
        strongest = gold["correct_label"]
        choice = strongest if strongest is not None else "A"
        if not correct_choice:
            choice = next(label for label in LETTERS if label != choice)
        signs = {label: entry["sign"] for label, entry
                 in gold["candidate_values_and_signs"].items()}
        if not correct_sign and strongest is not None:
            signs[strongest] = ("negative" if signs[strongest] == "positive" else "positive")
        probabilities = {
            label: (0.9 if entry["relation_present"] else 0.1)
            for label, entry in gold["candidate_values_and_signs"].items()
        }
        if not correct_probabilities:
            probabilities = {label: 1 - value for label, value in probabilities.items()}
        return {"question_id": gold["question_id"], "condition": condition,
                "status": "mock_fixture", "answer": {"choice": choice, "signs": signs,
                "relation_probabilities": probabilities}}

    def test_average_precision_groups_ties_and_is_order_invariant(self):
        self.assertAlmostEqual(average_precision([1, 0, 1, 0], [0.9, 0.8, 0.7, 0.6]), 5 / 6)
        self.assertAlmostEqual(average_precision([1, 0, 1, 0], [0.5] * 4), 0.5)
        self.assertAlmostEqual(average_precision([1, 0, 1, 0], [0.9, 0.9, 0.8, 0.7]), 7 / 12)
        self.assertAlmostEqual(average_precision([0, 1, 1, 0], [0.9, 0.9, 0.8, 0.7]), 7 / 12)
        self.assertIsNone(average_precision([0, 0], [0.8, 0.1]))
        self.assertIsNone(average_precision([], []))

    def test_probability_schema_requires_exact_finite_numeric_values(self):
        answer = {"choice": "B", "signs": {label: "uncertain" for label in LETTERS},
                  "relation_probabilities": {label: 0.5 for label in LETTERS}}
        self.assertEqual(validate_evaluator_answer(answer, require_probabilities=True)
                         ["relation_probabilities"], answer["relation_probabilities"])
        missing = copy.deepcopy(answer)
        missing.pop("relation_probabilities")
        with self.assertRaises(ValueError):
            validate_evaluator_answer(missing, require_probabilities=True)
        for bad_value in (True, False, float("nan"), float("inf"), -0.01, 1.01, "0.5", None):
            with self.subTest(bad_value=bad_value):
                invalid = copy.deepcopy(answer)
                invalid["relation_probabilities"]["A"] = bad_value
                with self.assertRaises(ValueError):
                    validate_evaluator_answer(invalid, require_probabilities=True)
        for remove, extra in ((True, False), (False, True)):
            invalid = copy.deepcopy(answer)
            if remove:
                invalid["relation_probabilities"].pop("D")
            if extra:
                invalid["relation_probabilities"]["E"] = 0.1
            with self.assertRaises(ValueError):
                validate_evaluator_answer(invalid, require_probabilities=True)

    def test_frozen_policy_has_source_and_split_provenance(self):
        policy = self.freeze()
        serialized = json.dumps(policy)
        for value in (PROTOCOL, MODEL, CLT, DATASET_HASH, digest(self.split), digest(self.graphs),
                      self.development["edge_semantics"], self.development["normalization"],
                      self.development["position_policy"]):
            self.assertIn(value, serialized)
        public, _, gold, _ = self.bank(policy=policy)
        self.assertEqual(public[0]["prompt_version"], PROTOCOL)
        self.assertEqual(gold[0]["metric_protocol"], PROTOCOL)
        self.assertEqual(gold[0]["relation_threshold"], 0.2)

    def test_policy_rejects_invalid_thresholds(self):
        for name in ("relation_min_abs", "winner_min_gap", "sign_min_abs"):
            values = (-0.01, float("nan"), float("inf"), True, "0.1")
            if name != "winner_min_gap":
                values = (*values, 0.0)
            for value in values:
                with self.subTest(name=name, value=value):
                    with self.assertRaises(ValueError):
                        self.freeze(**{name: value})

    def test_policy_rejects_final_measurements_and_overlapping_split(self):
        final = synthetic_graph(3, "synthetic-final", [0.9, -0.5, 0.1, 0.0])
        with self.assertRaises(ValueError):
            self.freeze([*self.graphs, final])
        overlap = copy.deepcopy(self.split)
        overlap["pair_ids"]["development"].append(1)
        with self.assertRaises(ValueError):
            self.freeze(split=overlap)
        with self.assertRaises(ValueError):
            self.bank(policy=self.freeze(), split=overlap)

    def test_bank_rejects_mismatched_units_context_and_split_hash(self):
        policy = self.freeze()
        for field in ("model_revision", "clt_revision", "pair_dataset_sha256", "edge_semantics",
                      "normalization", "position_policy"):
            bad = copy.deepcopy(self.development)
            bad[field] = "incompatible-value"
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    self.bank([bad], policy=policy)
            mixed_discovery = copy.deepcopy(self.discovery)
            mixed_discovery[field] = "incompatible-value"
            with self.subTest(discovery_field=field):
                with self.assertRaises(ValueError):
                    self.freeze([mixed_discovery, self.development])
        with self.assertRaises(ValueError):
            build_three_metric_bank(self.graphs, self.split, policy=policy,
                                    question_split="development", seed=17,
                                    split_sha256="f" * 64)
        changed_split = copy.deepcopy(self.split)
        changed_split["seed"] = 99
        with self.assertRaises(ValueError):
            build_three_metric_bank(self.graphs, changed_split, policy=policy,
                                    question_split="development", seed=17)

    def test_final_remains_unavailable_under_provisional_split(self):
        provisional = copy.deepcopy(self.split)
        provisional["status"] = "provisional_cpu_only"
        policy = self.freeze(split=provisional)
        final = synthetic_graph(3, "synthetic-final", [0.9, -0.5, 0.1, 0.0])
        with self.assertRaises(ValueError):
            self.bank(policy=policy, split=provisional, question_split="final", graphs=[final])

    def test_target_specific_thresholds_use_canonical_identity(self):
        target = next(node for node in self.development["nodes"] if node["id"] == "t")
        target_key = key_string(feature_key(MODEL, CLT, target))
        policy = self.freeze(per_target={target_key: {"relation_min_abs": 0.9,
                                                     "winner_min_gap": 0.02,
                                                     "sign_min_abs": 0.9}})
        _, _, gold, _ = self.bank(policy=policy)
        self.assertEqual(gold[0]["relation_threshold"], 0.9)
        self.assertFalse(gold[0]["sign_eligible"])
        self.assertTrue(all(not entry["relation_present"]
                            for entry in gold[0]["candidate_values_and_signs"].values()))
        with self.assertRaises(ValueError):
            self.freeze(per_target={"not-a-canonical-feature-key": {"relation_min_abs": 0.9}})

    def test_sampling_ignores_measured_values_and_node_or_edge_order(self):
        original = synthetic_graph(2, "same-metadata", [0.9, -0.5, 0.2, 0.1, 0.04])
        for index in range(4):
            source = {"id": f"u{index}", "type": "feature", "layer": 2, "position": 3,
                      "position_role": "name", "feature_index": 110 + index}
            original["nodes"].append(source)
            original["edges"].append({"source": source["id"], "target": "t", "value": 10 - index,
                                      "measurement_status": "computed_nonzero"})
            self.descriptions[key_string(feature_key(MODEL, CLT, source))] = {
                "activation_condition": "Additional matched source concept",
                "cross_layer_role": "Additional synthetic source role",
            }
        policy = self.freeze([self.discovery, original])
        public_a, refs_a, gold_a, _ = self.bank([original], policy=policy)
        changed = copy.deepcopy(original)
        for index, edge in enumerate(changed["edges"]):
            if edge["target"] == "t" and edge["source"].startswith("s"):
                edge["value"] = -(index + 3) ** 2
            elif edge["target"] == "t" and edge["source"].startswith("u"):
                edge["value"] = 0.001 * (index + 1)
        changed["nodes"].reverse()
        changed["edges"].reverse()
        public_b, refs_b, gold_b, _ = self.bank([changed], policy=policy)
        self.assertEqual(public_a, public_b)
        self.assertEqual(refs_a, refs_b)
        self.assertNotEqual(gold_a[0]["candidate_values_and_signs"],
                            gold_b[0]["candidate_values_and_signs"])

    def test_computed_zero_is_a_negative_but_omitted_is_never_a_candidate(self):
        measured = copy.deepcopy(self.development)
        measured["edges"].append({"source": "s4", "target": "t", "value": None,
                                  "measurement_status": "unmeasured_omitted"})
        public, refs, gold, _ = self.bank([measured])
        self.assertEqual(len(public), 1)
        self.assertEqual(len(refs[0]["candidate_feature_keys"]), 4)
        entries = gold[0]["candidate_values_and_signs"].values()
        self.assertNotIn("s4", [entry["source_node_id"] for entry in entries])
        zero = next(entry for entry in entries if entry["source_node_id"] == "s3")
        self.assertEqual(zero["value"], 0)
        self.assertFalse(zero["relation_present"])
        measured["edges"] = [edge for edge in measured["edges"] if not
                             (edge["source"] == "s2" and edge["target"] == "t")]
        self.assertEqual(len(self.bank([measured])[0]), 0)

    def test_all_conditions_hide_private_values_identity_and_target_role(self):
        public, refs, gold, _ = self.bank()
        self.assertEqual(len(public), 1)
        self.assertEqual(gold[0]["prompt_pair_id"], 2)
        for condition in ("metadata", "activation", "relational", "shuffled"):
            with self.subTest(condition=condition):
                messages, assignment = render_evaluator_messages(public[0], refs[0],
                                                                 self.descriptions, condition)
                visible = json.dumps(messages)
                self.assertIn("relation_probabilities", visible)
                for forbidden in ("correct_label", "candidate_values_and_signs", "feature_index",
                                  "source_node_id", "graph_id", MODEL, CLT, DATASET_HASH,
                                  "PRIVATE_TARGET_ROLE_SENTINEL", "0.8123456", "-0.5432197",
                                  "0.0812376", self.development["graph_id"]):
                    self.assertNotIn(forbidden, visible)
                self.assertEqual(set(assignment), set(LETTERS))
                if condition == "shuffled":
                    self.assertTrue(all(label != assigned for label, assigned in assignment.items()))
                if condition == "metadata":
                    self.assertNotIn("Synthetic concept", visible)

    def test_strength_and_sign_are_scored_independently(self):
        _, _, gold, _ = self.bank()
        wrong_choice = score_three_metrics(gold, [self.response(gold[0], correct_choice=False)])
        self.assertEqual(wrong_choice["relation_detection_ap"], 1)
        self.assertEqual(wrong_choice["strongest_source_accuracy"], 0)
        self.assertEqual(wrong_choice["sign_accuracy"], 1)
        self.assertIsNone(wrong_choice["sign_balanced_accuracy"])
        wrong_sign = score_three_metrics(gold, [self.response(gold[0], correct_sign=False)])
        self.assertEqual(wrong_sign["strongest_source_accuracy"], 1)
        self.assertEqual(wrong_sign["sign_accuracy"], 0)

    def test_weak_and_tied_questions_remain_for_detection_only(self):
        weak = synthetic_graph(4, "weak", [0.15, -0.1, 0.04, 0.0])
        tied = synthetic_graph(5, "tied", [0.8, -0.8, 0.04, 0.0])
        public, _, gold, _ = self.bank([self.development, weak, tied])
        self.assertEqual(len(public), 3)
        for row in gold:
            if row["prompt_pair_id"] != 2:
                self.assertIsNone(row["correct_label"])
                self.assertFalse(row["strength_eligible"])
                self.assertFalse(row["sign_eligible"])
        responses = [self.response(row, correct_choice=(row["prompt_pair_id"] == 2)) for row in gold]
        result = score_three_metrics(gold, responses)
        self.assertEqual(result["relation_detection_ap"], 1)
        self.assertEqual(result["detection_candidates"], 12)
        self.assertEqual(result["detection_positives"], 4)
        self.assertAlmostEqual(result["detection_prevalence"], 1 / 3)
        self.assertEqual(result["strength_questions"], 1)
        self.assertEqual(result["strength_excluded_questions"], 2)
        self.assertEqual(result["sign_questions"], 1)
        self.assertEqual(result["strongest_source_accuracy"], 1)
        self.assertEqual(result["sign_accuracy"], 1)
        zero_gap_policy = self.freeze([self.discovery, tied], self.make_split([5]), winner_min_gap=0)
        tied_gold = self.bank([tied], policy=zero_gap_policy)[2]
        self.assertFalse(tied_gold[0]["strength_eligible"])

    def test_missing_or_invalid_answers_keep_denominators_and_null_headline_ap(self):
        other = synthetic_graph(4, "second-strong", [-0.9, 0.5, 0.1, 0.0])
        _, _, gold, _ = self.bank([self.development, other])
        valid = self.response(gold[0])
        for responses in ([valid], [valid, {"question_id": gold[1]["question_id"],
                                           "answer": {"choice": "X"}}]):
            result = score_three_metrics(gold, responses)
            self.assertIsNone(result["relation_detection_ap"])
            self.assertEqual(result["available_relation_detection_ap"], 1)
            self.assertEqual(result["invalid_or_missing"], 1)
            self.assertEqual(result["invalid_response_rate"], 0.5)
            self.assertEqual(result["detection_prediction_coverage"], 0.5)
            self.assertEqual(result["detection_candidates"], 8)
            self.assertEqual(result["detection_prevalence"], 0.5)
            self.assertEqual(result["strength_questions"], 2)
            self.assertEqual(result["sign_questions"], 2)
            self.assertEqual(sum(row["valid"] for row in result["paired_rows"]), 1)
            self.assertEqual(result["strongest_source_accuracy"], 0.5)
            self.assertEqual(result["sign_accuracy"], 0.5)
            self.assertEqual(result["sign_balanced_accuracy"], 0.5)
        incomplete_probability = self.response(gold[1])
        incomplete_probability["answer"]["relation_probabilities"].pop("D")
        result = score_three_metrics(gold, [valid, incomplete_probability])
        self.assertIsNone(result["relation_detection_ap"])
        self.assertEqual(result["strongest_source_accuracy"], 0.5)

    def test_single_class_detection_and_empty_population_return_null(self):
        for values in ([0.9, -0.8, 0.7, 0.6], [0.15, -0.1, 0.04, 0.0]):
            one_class = synthetic_graph(2, "one-class", values)
            _, _, gold, _ = self.bank([one_class])
            result = score_three_metrics(gold, [self.response(gold[0])])
            self.assertIsNone(result["relation_detection_ap"])
        empty = score_three_metrics([], [])
        for metric in ("relation_detection_ap", "strongest_source_accuracy", "sign_accuracy",
                       "sign_balanced_accuracy"):
            self.assertIsNone(empty[metric])

    def test_balanced_sign_weights_sign_classes_equally(self):
        positive = [synthetic_graph(pair, f"positive-{pair}", [0.9, -0.5, 0.1, 0.0])
                    for pair in (2, 4, 5)]
        negative = synthetic_graph(6, "negative-6", [-0.9, 0.5, 0.1, 0.0])
        _, _, gold, _ = self.bank([*positive, negative])
        responses = [self.response(row, correct_sign=row["prompt_pair_id"] != 6) for row in gold]
        result = score_three_metrics(gold, responses)
        self.assertEqual(result["sign_accuracy"], 0.75)
        self.assertEqual(result["sign_balanced_accuracy"], 0.5)

    def test_unknown_duplicate_responses_and_legacy_gold_are_rejected(self):
        _, _, gold, _ = self.bank()
        row = self.response(gold[0])
        with self.assertRaises(ValueError):
            score_three_metrics(gold, [row, copy.deepcopy(row)])
        unknown = copy.deepcopy(row)
        unknown["question_id"] = "unknown-question"
        with self.assertRaises(ValueError):
            score_three_metrics(gold, [unknown])
        legacy = copy.deepcopy(gold)
        legacy[0].pop("metric_protocol")
        with self.assertRaises(ValueError):
            score_three_metrics(legacy, [row])

    def test_paired_bootstrap_resamples_whole_prompts_and_reports_undefined_draws(self):
        repeated = synthetic_graph(2, "three-targets-one-prompt", [0.9, -0.5, 0.1, 0.0])
        target = next(node for node in repeated["nodes"] if node["id"] == "t")
        for index in (1, 2):
            additional_target = {**target, "id": f"t{index}", "feature_index": 91 + index}
            repeated["nodes"].append(additional_target)
            repeated["edges"].extend({**edge, "target": additional_target["id"]}
                                     for edge in list(repeated["edges"])
                                     if edge["target"] == "t" and edge["source"].startswith("s"))
            self.descriptions[key_string(feature_key(MODEL, CLT, additional_target))] = {
                "activation_condition": "Additional synthetic target concept",
                "cross_layer_role": "PRIVATE_TARGET_ROLE_SENTINEL",
            }
        negative = synthetic_graph(4, "negative-prompt", [-0.9, 0.5, 0.1, 0.0])
        _, _, gold, _ = self.bank([repeated, negative])
        self.assertEqual(len(gold), 4)
        responses = []
        for row in gold:
            for condition in ("activation", "relational"):
                correct = ((row["prompt_pair_id"] == 2) == (condition == "activation"))
                responses.append(self.response(row, correct_choice=correct, correct_sign=correct,
                                               condition=condition))
        seed, n_boot = 19, 80
        result = paired_three_metric_bootstrap(gold, responses, "activation", "relational",
                                               seed=seed, n_boot=n_boot)
        self.assertEqual(result["prompt_clusters"], 2)
        self.assertEqual(result["metrics"]["strongest_source_accuracy"]["difference"], 0.5)
        strength = result["metrics"]["strongest_source_accuracy"]
        self.assertEqual(strength["ci_95"], [-1, 1])
        self.assertEqual(strength["defined_replicates"], n_boot)
        self.assertEqual(strength["undefined_replicates"], 0)
        balanced = result["metrics"]["sign_balanced_accuracy"]
        self.assertEqual(balanced["difference"], 0)
        rng = random.Random(seed)
        defined = sum(rng.choice([2, 4]) != rng.choice([2, 4]) for _ in range(n_boot))
        self.assertEqual(balanced["defined_replicates"], defined)
        self.assertEqual(balanced["undefined_replicates"], n_boot - defined)
        self.assertEqual(balanced["ci_95"], [0, 0])
        self.assertEqual(result, paired_three_metric_bootstrap(
            gold, responses, "activation", "relational", seed=seed, n_boot=n_boot))

    def test_offline_fixture_completes_without_api_or_environment_records(self):
        from pie.relations.fixture import main
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "fixture"
            with contextlib.redirect_stdout(io.StringIO()):
                main(["--output-dir", str(output)])
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "fixture")
            self.assertEqual(manifest["resource_usage"],
                             {"network_requests": 0, "subject_model_forwards": 0})
            self.assertIsNone(manifest["base_commit"])
            commands = json.loads((output / "commands.json").read_text())
            self.assertTrue(all(record["exit_code"] == 0 for record in commands))
            self.assertNotIn(str(Path(temporary)), json.dumps(commands))
            public = (output / "bank/public_questions.jsonl").read_text()
            self.assertNotIn("candidate_values_and_signs", public)
            self.assertNotIn("correct_label", public)


if __name__ == "__main__":
    unittest.main()
