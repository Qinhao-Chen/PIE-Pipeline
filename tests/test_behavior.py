"""CPU fixtures for the revision's scientific kernels; no models or network."""
from __future__ import annotations

import math
from pathlib import Path
import subprocess
import sys
import unittest
from types import SimpleNamespace

import torch

from pie.behavior.protocol import prepare_prompt, tokenize_prefix
from pie.behavior.metrics import dual_kl
from pie.behavior.fixed_write import (
    _union_delta_alignment, build_removed_feature_patches, fap_score_clt,
)
from pie.behavior.activation import rank_clean_activation_occurrences, select_topk_rows, candidate_universe_record
from pie.behavior.factp import SingleOccurrencePatchBuilder, make_score_vectors, select_topk_occurrences
from pie.behavior.native_replay import kl_with_witness
from pie.behavior.runner import aggregate, prepare_row, run_one
from pie.exchange.search import SearchConfig, occurrence_key, run_exchange


class TokenizerFixture:
    bos_token_id = 1
    pad_token_id = None
    eos_token_id = 2
    all_special_ids = [1, 2]

    def encode(self, text, add_special_tokens=True):
        return [1] + [3 + ord(char) for char in text]


def components(triples, values, rows, receiver_layers, positions, row_map):
    return {"activation_matrix": torch.sparse_coo_tensor(torch.tensor(triples).T,
             torch.tensor(values, dtype=torch.float32), (2, 4, 12)).coalesce(),
            "decoder_vecs": torch.tensor(rows, dtype=torch.float32),
            "decoder_locations": torch.tensor([receiver_layers, positions]),
            "encoder_to_decoder_map": torch.tensor(row_map)}


class TinyModel(torch.nn.Module):
    """A differentiable one-layer two-feature model for orchestration fixtures."""
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(256, 2)
        self.unembedding = torch.nn.Parameter(torch.randn(2, 5))
        self.cfg = SimpleNamespace(device=torch.device("cpu"), n_layers=1)
        self.tokenizer = TokenizerFixture()
        self.hooks = []

    def ensure_tokenized(self, text):
        return torch.tensor(self.tokenizer.encode(text))

    def get_caching_hooks(self, predicate):
        cache = {}
        return cache, [cache], None

    def run_with_hooks(self, tokens, fwd_hooks):
        fwd_hooks[0]["blocks.0.feature_in"] = self.embed(tokens).unsqueeze(0)
        return self(tokens)

    def add_hook(self, name, function, direction):
        self.hooks.append(function)

    def reset_hooks(self, **kwargs):
        self.hooks = []

    def forward(self, tokens):
        features = self.embed(tokens).unsqueeze(0)
        for hook in self.hooks:
            features = hook(features, None)
        return features @ self.unembedding


class TinyTranscoder:
    n_layers = 1
    feature_input_hook = "feature_in"
    feature_output_hook = "feature_out"

    def compute_attribution_components(self, inputs):
        value = inputs.squeeze(0)
        length = value.shape[0]
        positions = torch.arange(length).repeat_interleave(2)
        features = torch.arange(2).repeat(length)
        indices = torch.stack((torch.zeros(2 * length, dtype=torch.long), positions, features))
        vectors = torch.zeros(2 * length, 2)
        vectors[torch.arange(2 * length), features] = value.flatten()
        return {"activation_matrix": torch.sparse_coo_tensor(indices, value.flatten(), (1, length, 2)).coalesce(),
                "decoder_vecs": vectors, "decoder_locations": torch.stack((torch.zeros_like(positions), positions)),
                "encoder_to_decoder_map": torch.arange(2 * length)}


class BehaviorTests(unittest.TestCase):
    def test_help_does_not_import_torch(self):
        root = Path(__file__).resolve().parents[1]
        # -S omits site packages, so --help cannot accidentally depend on torch.
        result = subprocess.run([sys.executable, "-S", "-B", str(root / "scripts/run_behavior.py"), "--help"],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("lastlse-swap", result.stdout)

    def test_preanswer_removes_same_literal_answer_from_both_ioi_sides(self):
        raw = {"id": 7, "text_clean": "After meeting Jane, Robert gave a note to Jane",
               "text_corr": "After meeting Robert, Jane gave a note to Jane",
               "io_clean": "Jane", "io_corr": "Robert"}
        row = prepare_prompt(raw, "ioi", 7)
        self.assertTrue(row["text_clean"].endswith(" to"))
        self.assertTrue(row["text_corr"].endswith(" to"))
        self.assertEqual(row["removed_corr_suffix"], " Jane")
        self.assertEqual(raw["text_clean"].split()[-1], "Jane")
        self.assertEqual(row["answer_corr_label"], "Robert")
        tokens = tokenize_prefix(TokenizerFixture(), row["text_clean"], row["continuation_clean"])
        self.assertEqual(tokens["prediction_position"], tokens["token_count"] - 1)
        self.assertTrue(tokens["answer_token_ids"])

    def test_invalid_prefix_boundary_is_rejected(self):
        raw = {"id": 0, "text_clean": "Jane", "text_corr": "Robert", "io_clean": "Jane", "io_corr": "Robert"}
        with self.assertRaises(ValueError):
            prepare_prompt(raw, "ioi", 0)
        with self.assertRaises(ValueError):
            tokenize_prefix(TokenizerFixture(), "too short" * 20, " continuation")

    def test_dual_kl_matches_probability_reference_and_excludes_nonfinite_padding(self):
        torch.manual_seed(11)
        full, selected = torch.randn(2, 4, 7), torch.randn(2, 4, 7)
        mask = torch.tensor([[1, 1, 0, 0], [0, 1, 1, 1]])
        full[0, 2:] = float("nan")
        selected[0, 2:] = float("inf")
        full[1, :1] = float("inf")
        selected[1, :1] = float("nan")
        actual = dual_kl(full, selected, mask)
        expected_last, expected_mean = [], []
        for index in range(2):
            p = full[index][mask[index].bool()].softmax(-1)
            q = selected[index][mask[index].bool()].softmax(-1)
            kl = (p * (p.log() - q.log())).sum(-1)
            expected_last.append(kl[-1]); expected_mean.append(kl.mean())
        torch.testing.assert_close(actual["kl_pre_answer_last_token"], torch.stack(expected_last))
        torch.testing.assert_close(actual["kl_sequence_mean"], torch.stack(expected_mean))
        self.assertEqual(actual["final_real_position"].tolist(), [1, 3])

    def test_union_retains_corruption_only_occurrences_and_ties_decoder_writes(self):
        clean = components([[0, 1, 2]], [2.0], [[1.0, 2.0], [3.0, 4.0]], [0, 1], [1, 1], [0, 0])
        corrupt = components([[0, 1, 2], [1, 2, 7]], [1.0, 3.0],
                             [[0.5, 1.0], [1.5, 2.0], [5.0, 6.0]], [0, 1, 1], [1, 1, 2], [0, 0, 1])
        delta, layers, positions, count, row_map, sl, sp, sf = _union_delta_alignment(clean, corrupt)
        self.assertEqual(count, 2)
        self.assertEqual(list(zip(sl.tolist(), sp.tolist(), sf.tolist())), [(0, 1, 2), (1, 2, 7)])
        self.assertEqual(row_map.tolist(), [0, 0, 1])
        torch.testing.assert_close(delta, torch.tensor([[0.5, 1.0], [1.5, 2.0], [-5.0, -6.0]]))
        patches = build_removed_feature_patches(torch.tensor([False, True]), layers, positions, row_map, delta, 2)
        torch.testing.assert_close(patches[0][1], delta[0])
        torch.testing.assert_close(patches[1][1], delta[1])
        torch.testing.assert_close(patches[1][2], torch.zeros(2))
        grads = {0: torch.ones(4, 2), 1: torch.full((4, 2), 2.0)}
        score = fap_score_clt(delta, layers, positions, row_map, count, grads, 2)
        torch.testing.assert_close(score, torch.tensor([8.5, -22.0]))

    def test_magnitude_ranking_keeps_corruption_only_zero_score(self):
        activation = torch.sparse_coo_tensor(torch.tensor([[0, 1], [1, 2], [2, 7]]),
                                             torch.tensor([-2.0, 4.0]), (2, 4, 12))
        triples = [[0, 1, 2], [1, 2, 7], [1, 3, 9]]
        scores, sorted_scores, order, present = rank_clean_activation_occurrences(activation, triples)
        self.assertEqual(scores.tolist(), [2.0, 4.0, 0.0])
        self.assertEqual(present, [True, True, False])
        keep, rows, _ = select_topk_rows(triples, scores, order, 2)
        self.assertEqual(int(keep.sum()), 2)
        self.assertEqual([row["triple"] for row in rows], [[1, 2, 7], [0, 1, 2]])
        self.assertEqual(candidate_universe_record(triples, scores, order, present)["count"], 3)

    def test_factp_cached_single_patches_and_independent_topk(self):
        delta = torch.tensor([[0.25, -1.0], [1.5, 0.5], [-0.75, 2.0], [1.0, 1.0]])
        layers, positions, mapping = torch.tensor([0, 0, 1, 1]), torch.tensor([1, 1, 2, 2]), torch.tensor([0, 0, 1, 2])
        builder = SingleOccurrencePatchBuilder(layers, positions, mapping, delta, 3, 2)
        for index in range(3):
            keep = torch.ones(3, dtype=torch.bool); keep[index] = False
            reference = build_removed_feature_patches(keep, layers, positions, mapping, delta, 2)
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(reference, builder.build_single_occurrence(index))))
        signed, scores = make_score_vectors(16.25, [17.0, 15.5, 16.25])
        self.assertEqual(signed, [0.75, -0.75, 0.0])
        triples = [[0, 1, 2], [1, 2, 7], [1, 3, 9]]
        for k in (1, 2, 3):
            keep, selected, _ = select_topk_occurrences(triples, signed, scores, k)
            self.assertEqual(int(keep.sum()), k)
            self.assertEqual([row["candidate_index"] for row in selected], torch.topk(scores, k).indices.tolist())

    def test_single_exchange_cap_monotonicity_and_tied_writes(self):
        dataset = "d" * 64
        keys = [occurrence_key([dataset, "p1", 3, ["m", "c", i, i]]) for i in range(5)]
        context = {"model_revision": "m", "clt_revision": "c", "pair_dataset_sha256": dataset,
                   "prompt_pair_id": "p1", "protocol_label": "fixture", "evaluator_sha256": "e",
                   "evaluator_config_sha256": "cfg", "intervention_sha256": "i"}
        universe = {key: (f"{i}/write0", f"{i}/write1") for i, key in enumerate(keys)}
        attribution = {key: float(5 - i) for i, key in enumerate(keys)}
        calls = []

        def scorer(view):
            self.assertEqual(view.retained_write_ids, frozenset(w for k in view.retained_occurrences for w in universe[k]))
            calls.append(view.retained_occurrences)
            return 2.0 - sum(0.4 for k in keys[2:] if k in view.retained_occurrences)

        result = run_exchange(mode="single", initial=keys[:2], universe=universe, attribution=attribution,
                              context=context, scorer=scorer,
                              config=SearchConfig(max_candidate_evaluations=4, max_rounds=2, seed=0))
        self.assertLess(result.final_j, result.initial_j)
        self.assertEqual(len(result.final), 2)
        self.assertLessEqual(result.candidate_evaluations, 4)
        self.assertEqual(result.verification_calls, 1)
        self.assertEqual(len(calls), result.initialization_calls + result.uncached_candidate_calls + result.verification_calls)

    def test_portable_runner_executes_multiple_scientific_methods_on_cpu_fixture(self):
        raw = {"id": "fixture", "text_clean": "Earlier Robert gave a note to Jane",
               "text_corr": "Earlier Alicia gave a note to Jane", "io_clean": "Jane", "io_corr": "Alicia"}
        for method in ("fap", "lastlse", "activation", "factp", "fap-swap", "lastlse-swap", "random"):
            with self.subTest(method=method):
                torch.manual_seed(9)
                model, trans = TinyModel(), TinyTranscoder()
                row = prepare_row(raw, "ioi", 0, model.tokenizer)
                result = run_one(model, trans, row, method=method, k_values=(2, 4), dataset_sha256="a" * 64,
                                 swap_config=SearchConfig(max_candidate_evaluations=2, max_rounds=1))
                self.assertEqual(result["all_retained_parity"]["max_abs_logit_error"], 0.0)
                self.assertEqual(model.hooks, [])
                for k, values in result["by_k"].items():
                    self.assertEqual(len(values["retained_occurrences"]), int(k))
                    self.assertTrue(math.isfinite(values["kl_pre_answer_last_token"]))
                    self.assertTrue(math.isfinite(values["kl_sequence_mean"]))
                    if method.endswith("-swap"):
                        self.assertLessEqual(values["kl_pre_answer_last_token"],
                                             values["initial_metrics"]["kl_pre_answer_last_token"] + 1e-6)

    def test_native_fp64_witness_matches_independent_softcap_kl(self):
        raw = torch.tensor([1.0, 0.0, -2.0], dtype=torch.float64)
        delta = torch.tensor([0.3, -0.1, 0.2], dtype=torch.float64)
        cap = 2.0
        normal = cap * torch.tanh(raw / cap)
        actual, witness = kl_with_witness(raw, normal, delta, cap)
        p = normal.softmax(-1); q = (cap * torch.tanh((raw + delta) / cap)).softmax(-1)
        expected = float((p * (p.log() - q.log())).sum())
        self.assertAlmostEqual(actual, expected, places=12)
        self.assertAlmostEqual(witness["witness_minus_direct"], 0.0, places=12)

    def test_aggregate_uses_equal_prompt_weights_and_sample_sd(self):
        rows = [{"by_k": {"50": {"kl_pre_answer_last_token": 1.0, "kl_sequence_mean": 2.0}}},
                {"by_k": {"50": {"kl_pre_answer_last_token": 3.0, "kl_sequence_mean": 4.0}}}]
        result = aggregate(rows)["50"]["kl_sequence_mean"]
        self.assertEqual(result["mean"], 3.0)
        self.assertAlmostEqual(result["sd_sample"], math.sqrt(2.0))
        self.assertEqual(result["sd_population"], 1.0)
        self.assertIsNone(aggregate(rows[:1])["50"]["kl_sequence_mean"]["sd_sample"])


if __name__ == "__main__":
    unittest.main()
