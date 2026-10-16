"""Focused checks for the stage CLI and corrected explanation evaluation."""
import asyncio
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from pie.evaluation.fade import eval_clarity, eval_purity
from pie.evaluation.metrics import average_precision, gini_abs
from pie.models.activations import compute_activations


class ReplyClient:
    def __init__(self, text):
        self.text = text

    async def generate(self, messages, **kwargs):
        return SimpleNamespace(text=self.text)


class CliEvaluationTests(unittest.TestCase):
    def test_stage_help_and_script_wrappers_work_without_loading_models(self):
        root = Path(__file__).resolve().parents[1]
        commands = [["-m", "pie", stage, "--help"] for stage in ("prune", "interpret", "evaluate")]
        commands += [[str(root / "scripts" / name), "--help"] for name in (
            "run_pruning.py", "run_description.py", "run_evaluation.py", "run_behavior.py")]
        for command in commands:
            with self.subTest(command=command):
                result = subprocess.run([sys.executable, "-S", "-B", *command], cwd=root,
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage:", result.stdout)

    def test_current_pruning_output_requires_one_budget_and_deduplicates_identities(self):
        # Feature loading does not need an instantiated subject model.
        with patch.dict(sys.modules, {"circuit_tracer": SimpleNamespace(ReplacementModel=object)}):
            from pie.interpretation.data import load_pruned_features
        row = {"by_k": {"1": {"retained_occurrences": [[0, 1, 1]]},
                        "2": {"retained_occurrences": [[0, 1, 1], [0, 2, 1]]}}}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "per_prompt.jsonl"
            path.write_text(json.dumps(row) + "\n")
            with self.assertRaisesRegex(ValueError, "Multiple pruning budgets"):
                load_pruned_features(path)
            self.assertEqual(load_pruned_features(path, keep_topk=2), [2])
            with self.assertRaisesRegex(ValueError, "missing budget"):
                load_pruned_features(path, keep_topk=3)

    def test_activation_collection_uses_encoder_hooks_thresholds_and_excludes_bos(self):
        class Model:
            cfg = SimpleNamespace(device="cpu")

            def ensure_tokenized(self, text):
                return torch.tensor([1, 2, 3])

            def run_with_hooks(self, tokens, fwd_hooks):
                values = torch.tensor([[[99.0, 99.0], [1.0, 2.0], [3.0, 2.0]]])
                for name, function in fwd_hooks:
                    self.last_hook = name
                    function(values, None)

        class CLT:
            n_layers = 1
            feature_input_hook = "hook_resid_mid"

            def _get_encoder_weights(self, layer):
                return torch.eye(2)

            def encode_layer(self, value, layer):
                return value * (value > 2.5)

        model = Model()
        with contextlib.redirect_stderr(io.StringIO()):
            result = compute_activations(model, None, CLT(), ["short text"],
                                         {0: {"layer": 0, "index": 0}, 2: {"layer": 0, "index": 1}})
        self.assertEqual(model.last_hook, "blocks.0.hook_resid_mid")
        self.assertEqual(result[0].tolist(), [3.0])
        self.assertEqual(result[2].tolist(), [0.0])

    def test_tied_scores_are_order_invariant_and_inactive_populations_score_zero(self):
        for labels in ([1, 0], [0, 1]):
            self.assertEqual(average_precision(np.array(labels), np.array([0.5, 0.5])), 0.5)
        self.assertEqual(gini_abs(np.zeros(2), np.zeros(3)), 0)

    def test_small_corpus_and_undefined_class_comparisons(self):
        client = ReplyClient("{0: 0, 1: 1}")
        self.assertEqual(asyncio.run(eval_purity(client, "concept", ["a", "b"], np.array([0.0, 0.0]))),
                         (None, None))
        client = ReplyClient("{0: 2, 1: 0}")
        self.assertEqual(asyncio.run(eval_purity(client, "concept", ["a", "b"], np.array([0.0, 0.0]))),
                         (0.5, 0.0))

    def test_missing_and_invalid_ratings_raise_instead_of_scoring_zero(self):
        for response in ("", "{0: 2}", "{0: True, 1: 0}", "{0: 3, 1: 0}"):
            with self.subTest(response=response), self.assertRaises(ValueError):
                asyncio.run(eval_purity(ReplyClient(response), "concept", ["a", "b"], np.array([1.0, 0.0])))

    def test_invalid_synthetic_examples_raise_before_model_inference(self):
        for response in ("", "['only one']", "[1, 2, 3, 4, 5]"):
            with self.subTest(response=response), self.assertRaises(ValueError):
                asyncio.run(eval_clarity(ReplyClient(response), "concept", None, None, None, {}, np.zeros(2)))


if __name__ == "__main__":
    unittest.main()
