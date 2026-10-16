"""Saved-final and fresh earlier-position completion of immutable native masks.

Saved native terminal gates apply only to the final output position. Earlier
positions retain every direct output effect from the replayed graph sources.
The reported final term uses the historical saved completion and clean anchors;
fresh final attribution remains a drift diagnostic. This is a composite local
graph completion diagnostic, not an edge-gated Transformer or fresh final rerun.
"""

from __future__ import annotations

from contextlib import nullcontext
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import time
from types import SimpleNamespace
import uuid

import torch

from .edge_full_vocab_kl import (
    anchored_softcap_kl, complete_raw_logit_delta, graph_source_weights,
    raw_selected_logit_deltas, replay_feature_graph,
)
from .native_adapter import decode_native_edge_mask, graph_shape, native_counts


K_VALUES = (50, 100, 200, 400, 800)
DEFAULT_SETTINGS = {
    "basis_batch_size": 64,
    "unembed_chunk_size": 8192,
    "all_keep_atol": 1e-8,
    "source_tensor_kl_atol": 1e-6,
    "source_clean_anchor_atol": 1e-6,
    "saved_top_feature_atol": 1e-4,
    "saved_top_nonfeature_atol": 0.5,
    "saved_top_raw_atol": 0.5,
    "graph_probability_atol": 0.05,
    "direct_top_atol": 0.5,
    "witness_kl_atol": 1e-9,
}
PROTOCOL = "saved_final_output_graph_full_prefix_direct_completion_v2"


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _finite(tensor: torch.Tensor, label: str) -> None:
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"nonfinite {label}")


def _max_abs(tensor: torch.Tensor) -> float:
    return float(tensor.abs().max()) if tensor.numel() else 0.0


def resolve_settings(settings: dict | None = None) -> dict:
    resolved = {**DEFAULT_SETTINGS, **(settings or {})}
    for name in DEFAULT_SETTINGS:
        value = resolved[name]
        if name in ("basis_batch_size", "unembed_chunk_size"):
            if isinstance(value, bool) or int(value) != value or int(value) < 1:
                raise ValueError(f"{name} must be a positive integer")
            resolved[name] = int(value)
        elif not math.isfinite(float(value)) or float(value) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    return resolved


def kl_with_witness(raw_clean: torch.Tensor, normal_clean: torch.Tensor,
                    raw_delta: torch.Tensor, softcap: float,
                    atol: float = 1e-9) -> tuple[float, dict]:
    """Check direct FP64 KL against its independently reducible scalar witness.

    Inputs may be CUDA tensors; all arithmetic remains on their device. The
    witness identifies KL as logZ(candidate)-logZ(clean)-E_clean[logit shift].
    """
    raw = raw_clean.to(torch.float64).flatten()
    normal = normal_clean.to(device=raw.device, dtype=torch.float64).flatten()
    delta = raw_delta.to(device=raw.device, dtype=torch.float64).flatten()
    if raw.shape != normal.shape or raw.shape != delta.shape or raw.numel() < 1:
        raise ValueError("raw, normal and delta logits must be equal nonempty vectors")
    _finite(raw, "raw clean logits")
    _finite(normal, "normal clean logits")
    _finite(delta, "raw delta")
    if not math.isfinite(float(softcap)):
        raise ValueError("nonfinite softcap")
    shift = (float(softcap) * (torch.tanh((raw + delta) / float(softcap)) -
                             torch.tanh(raw / float(softcap)))
             if softcap > 0 else delta)
    candidate = normal + shift
    logp = torch.log_softmax(normal, dim=-1)
    logq = torch.log_softmax(candidate, dim=-1)
    p = logp.exp()
    direct = float(torch.sum(p * (logp - logq)))
    clean_z = float(torch.logsumexp(normal, dim=-1))
    candidate_z = float(torch.logsumexp(candidate, dim=-1))
    expected_shift = float(torch.sum(p * shift))
    witness_kl = candidate_z - clean_z - expected_shift
    if not all(math.isfinite(v) for v in (direct, clean_z, candidate_z,
                                         expected_shift, witness_kl)) or \
            min(direct, witness_kl) < -atol or abs(direct - witness_kl) > atol:
        raise ValueError(f"KL/witness disagreement: direct={direct}, witness={witness_kl}")
    return max(0.0, direct), {
        "logZ_clean": clean_z, "logZ_candidate": candidate_z,
        "expected_shift_under_clean": expected_shift,
        "witness_minus_direct": witness_kl - direct,
    }


def complete_position_delta(direct: torch.Tensor, position: int, token_length: int,
                            selected_token_ids: torch.Tensor,
                            final_selected_raw: torch.Tensor) -> torch.Tensor:
    """Keep early direct effects, applying saved native gates at the final position."""
    if not 0 <= position < token_length:
        raise ValueError("output position outside real prefix")
    if position == token_length - 1:
        return complete_raw_logit_delta(direct, selected_token_ids, final_selected_raw)
    return direct.to(torch.float64).flatten()


def full_prefix_residual_effects(graph, context, model, tokens,
                                current_indices: torch.Tensor,
                                weights_by_budget: dict[int, torch.Tensor],
                                budgets: tuple[int, ...], settings: dict):
    """Measure every output position with a shared five-budget source contraction.

    The attribution forward is installed once. Every backward except the last
    coordinate block of the last position retains that same forward graph.
    A zero-source-weight contraction runs through every measured basis block.
    """
    cfg = resolve_settings(settings)
    batch_size = cfg["basis_batch_size"]
    n_layers, n_pos = int(model.cfg.n_layers), int(tokens.numel())
    n_features, n_saved = len(current_indices), len(graph.selected_features)
    n_nonfeatures = (n_layers + 1) * n_pos
    n_sources, n_logits = n_features + n_nonfeatures, len(graph.logit_tokens)
    weights = torch.stack([weights_by_budget[k] for k in budgets], dim=1)
    if weights.shape != (n_sources, len(budgets)):
        raise ValueError("source weight layout mismatch")
    _finite(weights, "source weights")
    saved_indices = graph.active_features[graph.selected_features.long()].long()
    current_map = {tuple(map(int, triple)): i for i, triple
                   in enumerate(current_indices.tolist())}
    if len(current_map) != n_features:
        raise ValueError("duplicate fresh source occurrence")
    current_to_saved = torch.full((n_features,), -1, dtype=torch.long)
    for saved, triple in enumerate(saved_indices.tolist()):
        current = current_map.get(tuple(map(int, triple)))
        if current is not None:
            current_to_saved[current] = saved
    matching = current_to_saved >= 0
    if int(matching.sum()) != n_saved:
        raise ValueError("fresh attribution context lacks saved clean graph sources")

    source_all_keep = torch.zeros((n_sources, 1), device=tokens.device, dtype=torch.float32)
    residual_deltas = torch.empty((n_pos, int(model.cfg.d_model), len(budgets)),
                                  dtype=torch.float32)
    all_keep_basis_max = 0.0
    all_keep_position_max = torch.zeros(n_pos, dtype=torch.float64)
    with torch.enable_grad(), context.install_hooks(model):
        residual = model.forward(tokens.expand(batch_size, -1), stop_at_layer=n_layers)
        context._resid_activations[-1] = model.ln_final(residual)
        W_U = model.unembed.W_U
        token_ids = graph.logit_tokens.long().to(W_U.device)
        layer = torch.full((n_logits,), n_layers, device=tokens.device)
        final_position = torch.full((n_logits,), n_pos - 1, device=tokens.device)
        centered_vectors = (W_U[:, token_ids] - W_U.mean(dim=-1, keepdim=True)).T.contiguous()
        centered_rows = context.compute_batch(
            layers=layer, positions=final_position, inject_values=centered_vectors,
            retain_graph=True).float().cpu()
        _finite(centered_rows, "saved output attribution rows")
        direct_selected = centered_rows[:, :n_sources].to(torch.float64) @ weights
        feature_error = (centered_rows[:, :n_features][:, matching] -
                         graph.adjacency_matrix[-n_logits:, :n_saved].float()[
                             :, current_to_saved[matching]]).abs()
        nonfeature_error = (centered_rows[:, n_features:n_sources] -
                           graph.adjacency_matrix[-n_logits:, n_saved:n_saved + n_nonfeatures].float()).abs()
        mean_row = context.compute_batch(
            layers=torch.full((1,), n_layers, device=tokens.device),
            positions=torch.full((1,), n_pos - 1, device=tokens.device),
            inject_values=W_U.mean(dim=-1)[None, :], retain_graph=True,
        )[0, :n_sources].float().cpu()
        _finite(mean_row, "mean output attribution row")
        mean_effects = torch.zeros(n_saved + n_nonfeatures, dtype=torch.float64)
        mean_effects[current_to_saved[matching]] = mean_row[:n_features][matching].double()
        mean_effects[n_saved:] = mean_row[n_features:].double()
        raw_rows = context.compute_batch(
            layers=layer, positions=final_position,
            inject_values=W_U[:, token_ids].T.contiguous(), retain_graph=True,
        )[:, :n_sources].float().cpu()
        _finite(raw_rows, "raw saved-output attribution rows")
        raw_saved_order = torch.zeros((n_logits, n_saved + n_nonfeatures), dtype=torch.float64)
        raw_saved_order[:, current_to_saved[matching]] = raw_rows[:, :n_features][:, matching].double()
        raw_saved_order[:, n_saved:] = raw_rows[:, n_features:].double()
        saved_centered = graph.adjacency_matrix[-n_logits:, :n_saved + n_nonfeatures].double()
        raw_error = (raw_saved_order - (saved_centered + mean_effects[None, :])).abs()[saved_centered != 0]
        diagnostics = {
            "saved_top_row_common_feature_count": int(matching.sum()),
            "saved_top_row_feature_max_abs_difference": _max_abs(feature_error),
            "saved_top_row_nonfeature_max_abs_difference": _max_abs(nonfeature_error),
            "saved_top_raw_support_edge_count": int(raw_error.numel()),
            "saved_top_raw_support_max_abs_difference": _max_abs(raw_error),
        }
        for name, threshold in (("saved_top_row_feature_max_abs_difference", "saved_top_feature_atol"),
                                ("saved_top_row_nonfeature_max_abs_difference", "saved_top_nonfeature_atol"),
                                ("saved_top_raw_support_max_abs_difference", "saved_top_raw_atol")):
            if not math.isfinite(diagnostics[name]) or diagnostics[name] > cfg[threshold]:
                raise ValueError(f"fresh/saved terminal attribution mismatch: {diagnostics}")
        weights_gpu = weights.to(device=tokens.device, dtype=torch.float32)
        for position in range(n_pos):
            for start in range(0, int(model.cfg.d_model), batch_size):
                stop = min(start + batch_size, int(model.cfg.d_model))
                block = torch.zeros((stop - start, int(model.cfg.d_model)),
                                    device=tokens.device, dtype=W_U.dtype)
                index = torch.arange(stop - start, device=tokens.device)
                block[index, index + start] = 1
                rows = context.compute_batch(
                    layers=torch.full((stop - start,), n_layers, device=tokens.device),
                    positions=torch.full((stop - start,), position, device=tokens.device),
                    inject_values=block,
                    retain_graph=not (position == n_pos - 1 and stop == int(model.cfg.d_model)),
                )[:, :n_sources].float()
                _finite(rows, "prefix residual attribution rows")
                contracted = rows @ weights_gpu
                noop = rows @ source_all_keep
                _finite(contracted, "prefix residual source contraction")
                _finite(noop, "all-keep residual source contraction")
                position_gap = _max_abs(noop)
                all_keep_basis_max = max(all_keep_basis_max, position_gap)
                all_keep_position_max[position] = max(float(all_keep_position_max[position]), position_gap)
                residual_deltas[position, start:stop] = contracted.detach().cpu()
    diagnostics["all_keep_basis_contraction_max_abs"] = all_keep_basis_max
    diagnostics["all_keep_basis_per_position_max_abs"] = all_keep_position_max.tolist()
    diagnostics["basis_positions_measured"] = n_pos
    diagnostics["basis_coordinates_per_position"] = int(model.cfg.d_model)
    if all_keep_basis_max > cfg["all_keep_atol"]:
        raise ValueError("all-keep source weights fail measured prefix basis parity")
    return residual_deltas, mean_effects, diagnostics, direct_selected


def _unembed_delta(model, residual_delta: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """Preserve the original final-output worker's FP32 chunked unembedding."""
    W_U = model.unembed.W_U
    vector = residual_delta.to(device=W_U.device, dtype=torch.float32)
    result = torch.empty(W_U.shape[1], dtype=torch.float32)
    with torch.no_grad():
        for start in range(0, W_U.shape[1], chunk_size):
            stop = min(start + chunk_size, W_U.shape[1])
            result[start:stop] = (vector @ W_U[:, start:stop].float()).cpu()
    _finite(result, "unembedded direct delta")
    return result


def _load_checked_source(source_entry: dict):
    from circuit_tracer import Graph

    prompt_dir = Path(source_entry["source_prompt_dir"]).resolve()
    record_file = prompt_dir / "record.json"
    if _digest(record_file) != source_entry["record_sha256"]:
        raise ValueError("immutable source record hash differs")
    record = json.loads(record_file.read_text(encoding="utf-8"))
    if record.get("files_sha256") != source_entry["files_sha256"] or \
            record.get("row_index") != source_entry["row_index"] or \
            str(record.get("pair_id")) != str(source_entry["pair_id"]) or \
            record.get("model_name", record.get("model")) != source_entry["model"]:
        raise ValueError("immutable source prompt metadata differs")
    if record.get("task", source_entry["task"]) != source_entry["task"]:
        raise ValueError("immutable source task differs")
    for relative, expected in source_entry["files_sha256"].items():
        file = (prompt_dir / relative).resolve()
        if not file.is_relative_to(prompt_dir) or _digest(file) != expected:
            raise ValueError(f"immutable source file hash differs: {relative}")
    graph = Graph.from_pt(str(prompt_dir / "graph.pt"), map_location="cpu")
    source_tensors = torch.load(prompt_dir / "kl_tensors.pt", map_location="cpu", weights_only=False)
    source_masks = {
        k: torch.load(prompt_dir / f"native_mask_k{k}.pt", map_location="cpu", weights_only=False)
        for k in K_VALUES
    }
    return record, graph, source_tensors, source_masks


def evaluate_saved_prompt(model, trans, source_entry: dict,
                          output_dir: Path, settings: dict) -> dict:
    """Evaluate a saved native circuit at every real position; never mutate it."""
    cfg = resolve_settings(settings)
    started = time.perf_counter()
    output_dir = Path(output_dir).resolve()
    source_dir = Path(source_entry["source_prompt_dir"]).resolve()
    if output_dir == source_dir or output_dir.is_relative_to(source_dir):
        raise ValueError("sequence outputs must be outside the immutable source prompt")
    tensor_file = output_dir / "sequence_replay.pt"
    if tensor_file.exists():
        raise FileExistsError(f"refusing to overwrite existing sequence payload: {tensor_file}")
    record, graph, source_tensors, encoded_masks = _load_checked_source(source_entry)
    source_validation_seconds = time.perf_counter() - started
    shape = graph_shape(graph)
    if shape.features != len(graph.active_features) or int(graph.cfg.n_layers) != int(model.cfg.n_layers):
        raise ValueError("graph is truncated or model layer count differs")
    if int(graph.cfg.d_model) != int(model.cfg.d_model) or \
            model.unembed.W_U.shape[0] != int(model.cfg.d_model):
        raise ValueError("graph/model residual width differs")
    if model.training:
        raise ValueError("native sequence model must be in eval mode")
    if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
        raise ValueError("native sequence replay requires TF32 disabled")
    model_parameter = next(model.parameters())
    trans_parameter = next(trans.parameters())
    if model_parameter.dtype != torch.bfloat16 or trans_parameter.dtype != torch.bfloat16:
        raise ValueError("native sequence replay requires pinned BF16 model and CLT")
    for key in ("model_revision", "clt_revision"):
        expected_revision = cfg.get(key, source_entry.get(key))
        if expected_revision is not None and record.get(key) != expected_revision:
            raise ValueError(f"source {key} differs from pinned runtime profile")
    token_ids = graph.input_tokens.long().cpu().tolist()
    saved_ids = record.get("graph_input_token_ids", record.get("input_token_ids"))
    if token_ids != saved_ids or len(token_ids) < 1:
        raise ValueError("saved graph input tokens differ from source record")
    if "token_ids" in source_entry and token_ids != source_entry["token_ids"]:
        raise ValueError("saved graph input differs from indexed input token boundary")
    if record.get("prediction_position", len(token_ids) - 1) != len(token_ids) - 1:
        raise ValueError("saved native endpoint is not the final real prefix position")
    tokens = graph.input_tokens.long().to(model.cfg.device)
    softcap_value = getattr(model.cfg, "output_logits_soft_cap", None)
    softcap = float(softcap_value or 0.0)
    if not math.isfinite(softcap):
        raise ValueError("nonfinite model output softcap")
    softcap = max(0.0, softcap)
    if max(0.0, float(record["model_output_softcap"])) != softcap:
        raise ValueError("source/runtime model output softcap differs")
    source_scalar_gaps = {}
    source_completed = source_tensors["completed_raw_deltas"]
    for k in K_VALUES:
        delta = source_completed.get(str(k), source_completed.get(k))
        if delta is None:
            raise ValueError(f"source lacks saved final delta for K={k}")
        checked = anchored_softcap_kl(source_tensors["raw_clean"], source_tensors["normal_clean"], delta, softcap)
        source_scalar_gaps[str(k)] = checked - float(source_entry["source_final_kl"][str(k)])
        if abs(source_scalar_gaps[str(k)]) > cfg["source_tensor_kl_atol"]:
            raise ValueError(f"source final scalar does not reproduce for K={k}")
    context = None
    temporary = None
    try:
        forward_start = time.perf_counter()
        with torch.no_grad():
            raw_context = model.zero_softcap() if softcap > 0 else nullcontext()
            with raw_context:
                raw_clean = model(tokens)[0].float().cpu()
        context = model.setup_attribution(tokens)
        normal_clean = context.logits[0].float().cpu()
        if raw_clean.shape != normal_clean.shape or raw_clean.shape != (len(token_ids), model.unembed.W_U.shape[1]):
            raise ValueError("full-prefix clean logits have wrong token/vocabulary shape")
        _finite(raw_clean, "prefix raw clean logits")
        _finite(normal_clean, "prefix normal clean logits")
        source_anchor_gaps = {
            "raw_clean_max_abs": _max_abs(raw_clean[-1].double() - source_tensors["raw_clean"].double()),
            "normal_clean_max_abs": _max_abs(normal_clean[-1].double() - source_tensors["normal_clean"].double()),
        }
        if max(source_anchor_gaps.values()) > cfg["source_clean_anchor_atol"]:
            raise ValueError(f"fresh final clean anchors differ from immutable source: {source_anchor_gaps}")
        graph_prob = torch.softmax(normal_clean[-1], dim=-1)[graph.logit_tokens.long()]
        probability_gap = _max_abs(graph_prob - graph.logit_probabilities.float())
        if probability_gap > cfg["graph_probability_atol"]:
            raise ValueError("fresh final graph probabilities differ from saved seeds")
        clean_forward_seconds = time.perf_counter() - forward_start
        current_indices = context.activation_matrix.indices().T.cpu()
        current_values = context.activation_matrix.values().cpu()
        replays, weights_by_budget, feature_diagnostics, decoded = {}, {}, {}, {}
        replay_start = time.perf_counter()
        for k in K_VALUES:
            mask = encoded_masks[k]
            node_mask = mask["node_mask"]
            edge_mask = decode_native_edge_mask(mask)
            native = SimpleNamespace(node_mask=node_mask, edge_mask=edge_mask)
            achieved = native_counts(graph, native)["retained_feature_occurrences"]
            if achieved != int(source_entry["achieved_k"][str(k)]) or achieved < k:
                raise ValueError(f"saved native achieved K differs for K={k}")
            replay = replay_feature_graph(graph, node_mask, edge_mask)
            weights, diagnostics = graph_source_weights(graph, node_mask, current_indices, current_values, replay)
            replays[k], weights_by_budget[k], feature_diagnostics[str(k)] = replay, weights, diagnostics
            decoded[k] = native
        all_nodes = torch.ones(shape.nodes, dtype=torch.bool)
        all_edges = graph.adjacency_matrix.ne(0)
        all_replay = replay_feature_graph(graph, all_nodes, all_edges)
        all_weights, all_source_diag = graph_source_weights(
            graph, all_nodes, current_indices, current_values, all_replay)
        all_keep_source_gap = _max_abs(all_weights)
        if all_keep_source_gap > cfg["all_keep_atol"]:
            raise ValueError("all-keep graph replay produces nonzero source change")
        replay_seconds = time.perf_counter() - replay_start
        basis_start = time.perf_counter()
        residual_deltas, mean_effects, basis_diagnostics, direct_selected = full_prefix_residual_effects(
            graph, context, model, tokens, current_indices, weights_by_budget, K_VALUES, cfg)
        if tokens.device.type == "cuda":
            torch.cuda.synchronize(tokens.device)
        basis_seconds = time.perf_counter() - basis_start
        context = None
        gc.collect()
        del current_indices, current_values
        final_selected = {
            k: raw_selected_logit_deltas(graph, decoded[k].edge_mask, replays[k], mean_effects)
            for k in K_VALUES
        }
        all_keep_selected = raw_selected_logit_deltas(
            graph, all_edges, all_replay, torch.zeros_like(mean_effects))
        all_keep_terminal_gap = _max_abs(all_keep_selected)
        all_keep_kls = []
        per_position = {str(k): [] for k in K_VALUES}
        witnesses = {str(k): {"logZ_clean": [], "logZ_candidate": [],
                               "expected_shift_under_clean": []} for k in K_VALUES}
        final_completed = {}
        fresh_final_completed = {}
        fresh_final_values = {}
        fresh_final_witnesses = {}
        final_diagnostics = {}
        max_witness_gap = 0.0
        metric_start = time.perf_counter()
        for position in range(len(token_ids)):
            raw_gpu = raw_clean[position].to(tokens.device)
            normal_gpu = normal_clean[position].to(tokens.device)
            noop_cpu = complete_position_delta(
                torch.zeros(raw_clean.shape[1], dtype=torch.float32), position, len(token_ids),
                graph.logit_tokens, all_keep_selected)
            noop_kl, _ = kl_with_witness(raw_gpu, normal_gpu, noop_cpu.to(tokens.device),
                                       softcap, cfg["witness_kl_atol"])
            all_keep_kls.append(noop_kl)
            for column, k in enumerate(K_VALUES):
                direct = _unembed_delta(model, residual_deltas[position, :, column], cfg["unembed_chunk_size"])
                completed = complete_position_delta(
                    direct, position, len(token_ids), graph.logit_tokens, final_selected[k])
                value, witness = kl_with_witness(raw_gpu, normal_gpu, completed.to(tokens.device),
                                               softcap, cfg["witness_kl_atol"])
                max_witness_gap = max(max_witness_gap, abs(witness["witness_minus_direct"]))
                if position == len(token_ids) - 1:
                    fresh_final_completed[str(k)] = completed.cpu().double()
                    fresh_final_values[str(k)] = value
                    fresh_final_witnesses[str(k)] = {
                        name: witness[name] for name in witnesses[str(k)]
                    }
                    source_final = float(source_entry["source_final_kl"][str(k)])
                    parity_gap = value - source_final
                    direct_top_gap = _max_abs(direct_selected[:, column] - replays[k].node_only_logit_deltas)
                    if direct_top_gap > cfg["direct_top_atol"]:
                        raise ValueError(f"fresh direct-top source contraction differs for K={k}")
                    centered = direct[graph.logit_tokens.long()].double() - direct.double().mean()
                    basis_gap = _max_abs(centered - replays[k].node_only_logit_deltas)
                    final_diagnostics[str(k)] = {
                        "source_final_kl": source_final, "fresh_minus_source_final_kl": parity_gap,
                        "direct_top_vs_graph_node_only_selected_max_abs": direct_top_gap,
                        "basis_vs_graph_node_only_selected_max_abs": basis_gap,
                        "completion_basis_warning_over_0p5": basis_gap > 0.5,
                    }
                    # Preserve the historical final endpoint exactly. Fresh
                    # final attribution is stored separately and never silently
                    # substituted for this source-fixed term in the mean.
                    saved_delta = source_completed.get(str(k), source_completed.get(k))
                    final_completed[str(k)] = saved_delta.detach().cpu().clone().double()
                    source_value, source_witness = kl_with_witness(
                        source_tensors["raw_clean"].to(tokens.device),
                        source_tensors["normal_clean"].to(tokens.device),
                        final_completed[str(k)].to(tokens.device), softcap, cfg["witness_kl_atol"])
                    if abs(source_value - source_final) > cfg["source_tensor_kl_atol"]:
                        raise ValueError(f"immutable final witness does not reproduce source scalar for K={k}")
                    final_diagnostics[str(k)]["saved_final_tensor_minus_scalar_kl"] = source_value - source_final
                    final_diagnostics[str(k)]["fresh_final_kl"] = value
                    value, witness = source_final, source_witness
                per_position[str(k)].append(value)
                for name in witnesses[str(k)]:
                    witnesses[str(k)][name].append(witness[name])
        metric_seconds = time.perf_counter() - metric_start
        all_keep = {
            "passed": max(all_keep_source_gap, all_keep_terminal_gap,
                          basis_diagnostics["all_keep_basis_contraction_max_abs"],
                          max(all_keep_kls)) <= cfg["all_keep_atol"],
            "tolerance": cfg["all_keep_atol"],
            "source_weight_max_abs": all_keep_source_gap,
            "terminal_delta_max_abs": all_keep_terminal_gap,
            "basis_contraction_max_abs": basis_diagnostics["all_keep_basis_contraction_max_abs"],
            "per_position_kl": all_keep_kls,
            "max_kl_nats": max(all_keep_kls), "source_alignment": all_source_diag,
        }
        if not all_keep["passed"]:
            raise ValueError("all-keep prefix KL parity failed")
        payload = {
            "schema_version": "native-graph-sequence-replay-tensors-v2", "protocol": PROTOCOL,
            "raw_clean": source_tensors["raw_clean"].detach().cpu().clone(),
            "normal_clean": source_tensors["normal_clean"].detach().cpu().clone(),
            "residual_deltas": residual_deltas,
            "final_completed_raw_deltas": final_completed,
            "fresh_final_raw_clean": raw_clean[-1].clone(),
            "fresh_final_normal_clean": normal_clean[-1].clone(),
            "fresh_final_completed_raw_deltas": fresh_final_completed,
            "fresh_final_kl": fresh_final_values,
            "fresh_final_kl_witnesses": fresh_final_witnesses,
            "per_position_kl": {k: torch.tensor(v, dtype=torch.float64) for k, v in per_position.items()},
            "kl_witnesses": {k: {name: torch.tensor(v, dtype=torch.float64) for name, v in item.items()}
                             for k, item in witnesses.items()},
            "selected_token_ids": graph.logit_tokens.long().cpu(),
            "token_ids": graph.input_tokens.long().cpu(), "softcap": softcap,
            "all_keep_per_position_kl": torch.tensor(all_keep_kls, dtype=torch.float64),
            "all_keep_basis_per_position_max_abs": torch.tensor(
                basis_diagnostics["all_keep_basis_per_position_max_abs"], dtype=torch.float64),
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        temporary = tensor_file.with_name(tensor_file.name + f".tmp_{uuid.uuid4().hex}")
        torch.save(payload, temporary)
        os.replace(temporary, tensor_file)
        temporary = None
        return {
            "protocol": PROTOCOL, "task": source_entry["task"], "model": source_entry["model"],
            "row_index": source_entry["row_index"], "pair_id": str(source_entry["pair_id"]),
            "token_length": len(token_ids), "token_ids": token_ids,
            "prediction_position": len(token_ids) - 1, "softcap": softcap,
            "budgets": [{"requested_k": k, "achieved_k": int(source_entry["achieved_k"][str(k)]),
                         "kl_last_token": per_position[str(k)][-1],
                         "kl_sequence_mean": math.fsum(per_position[str(k)]) / len(token_ids),
                         "kl_per_position": per_position[str(k)],
                         "source_final_kl": float(source_entry["source_final_kl"][str(k)]),
                         "final_endpoint_origin": "immutable_saved_source_final"} for k in K_VALUES],
            "all_keep_parity": all_keep,
            "diagnostics": {"basis": basis_diagnostics, "feature_source_alignment": feature_diagnostics,
                            "final_position": final_diagnostics,
                            "source_saved_tensor_minus_scalar_kl": source_scalar_gaps,
                            "fresh_graph_top_probability_max_abs": probability_gap,
                            "witness_vs_direct_kl_max_abs": max_witness_gap,
                            "source_final_clean_anchor": source_anchor_gaps,
                            "fresh_vs_saved_final_raw_clean_max_abs": _max_abs(raw_clean[-1] - source_tensors["raw_clean"].float()),
                            "fresh_vs_saved_final_normal_clean_max_abs": _max_abs(normal_clean[-1] - source_tensors["normal_clean"].float())},
            "costs": {"source_validation_seconds": source_validation_seconds,
                      "clean_forward_seconds": clean_forward_seconds,
                      "feature_replay_seconds": replay_seconds, "prefix_basis_seconds": basis_seconds,
                      "completion_and_kl_seconds": metric_seconds,
                      "total_prompt_seconds": time.perf_counter() - started},
            "files_sha256": {tensor_file.name: _digest(tensor_file)},
        }
    finally:
        context = None
        if temporary is not None and temporary.exists():
            temporary.unlink()
        gc.collect()


def validate_sequence_payload(payload: dict, record: dict, source_tensors: dict,
                              source_entry: dict, atol: float = 1e-9,
                              source_tensor_atol: float = 1e-6,
                              source_anchor_atol: float = 1e-6) -> dict:
    """Check the composite endpoint against hash-verified immutable source data.

    Earlier witnesses retain the 1e-9 gate. The final saved scalar has the
    original 1e-6 source-tensor round-trip gate, while its actual reported
    scalar, raw/normal anchors and completed delta must equal source exactly.
    Fresh final values are independently validated diagnostics.
    """
    length = int(record["token_length"])
    if length < 1 or payload.get("protocol") != PROTOCOL or record.get("protocol") != PROTOCOL:
        raise ValueError("sequence payload protocol/token length differs")
    if payload["token_ids"].long().tolist() != record["token_ids"] or len(record["token_ids"]) != length:
        raise ValueError("sequence payload token IDs differ")
    residuals = payload["residual_deltas"]
    if residuals.ndim != 3 or residuals.shape[0] != length or residuals.shape[2] != len(K_VALUES) or residuals.dtype != torch.float32:
        raise ValueError("sequence residual payload shape/dtype differs")
    _finite(residuals, "persisted residual effects")
    if float(payload["softcap"]) != float(record["softcap"]):
        raise ValueError("sequence payload softcap differs")
    for name in ("raw_clean", "normal_clean"):
        stored = payload[name].detach().cpu().double()
        saved = source_tensors[name].detach().cpu().double()
        if stored.ndim != 1 or stored.shape != saved.shape or not torch.equal(stored, saved):
            raise ValueError(f"saved final {name} differs from immutable source")
        _finite(stored, "immutable final clean anchor")
        fresh = payload[f"fresh_final_{name}"].detach().cpu().double()
        if fresh.shape != saved.shape:
            raise ValueError("fresh final clean anchor shape differs")
        _finite(fresh, "fresh final clean anchor")
        if _max_abs(fresh - saved) > source_anchor_atol:
            raise ValueError("fresh final clean anchor differs from immutable source")
    budget_rows = {int(row["requested_k"]): row for row in record["budgets"]}
    if len(budget_rows) != len(record["budgets"]) or set(budget_rows) != set(K_VALUES):
        raise ValueError("sequence record budget coverage differs")
    if set(payload["per_position_kl"]) != {str(k) for k in K_VALUES} or \
            set(payload["final_completed_raw_deltas"]) != {str(k) for k in K_VALUES} or \
            set(payload["fresh_final_completed_raw_deltas"]) != {str(k) for k in K_VALUES} or \
            set(payload["fresh_final_kl"]) != {str(k) for k in K_VALUES} or \
            set(payload["fresh_final_kl_witnesses"]) != {str(k) for k in K_VALUES}:
        raise ValueError("sequence payload budget coverage differs")
    largest = 0.0
    largest_final = 0.0
    largest_fresh = 0.0
    for k in K_VALUES:
        key = str(k)
        values = payload["per_position_kl"][key].double()
        witness = payload["kl_witnesses"][key]
        if values.shape != (length,):
            raise ValueError("per-position KL shape differs")
        _finite(values, "persisted per-position KL")
        terms = [witness[name].double() for name in
                 ("logZ_candidate", "logZ_clean", "expected_shift_under_clean")]
        if any(term.shape != (length,) for term in terms):
            raise ValueError("KL witness shape differs")
        for term in terms:
            _finite(term, "persisted KL witness")
        checked = terms[0] - terms[1] - terms[2]
        largest = max(largest, _max_abs(values[:-1] - checked[:-1]))
        largest_final = max(largest_final, abs(float(values[-1] - checked[-1])))
        if bool((values < 0).any()) or bool((checked < -atol).any()) or \
                _max_abs(values[:-1] - checked[:-1]) > atol or \
                abs(float(values[-1] - checked[-1])) > source_tensor_atol:
            raise ValueError("per-position KL does not reproduce from witnesses")
        reported = torch.tensor(budget_rows[k]["kl_per_position"], dtype=torch.float64)
        if reported.shape != (length,) or _max_abs(reported - values) > atol:
            raise ValueError("record/payload per-position KL differs")
        if abs(float(values[-1]) - float(budget_rows[k]["kl_last_token"])) > atol or \
                abs(math.fsum(values.tolist()) / length - float(budget_rows[k]["kl_sequence_mean"])) > atol:
            raise ValueError("sequence/final reductions differ from per-position KL")
        source_final = float(source_entry["source_final_kl"][key])
        if not math.isfinite(source_final) or source_final < 0 or \
                float(values[-1]) != source_final or \
                float(budget_rows[k]["kl_last_token"]) != source_final or \
                float(budget_rows[k]["source_final_kl"]) != source_final or \
                budget_rows[k]["final_endpoint_origin"] != "immutable_saved_source_final":
            raise ValueError("reported final scalar differs from immutable source")
        saved_delta = source_tensors["completed_raw_deltas"].get(
            key, source_tensors["completed_raw_deltas"].get(k))
        if saved_delta is None:
            raise ValueError("immutable source lacks final completed delta")
        final_delta = payload["final_completed_raw_deltas"][key]
        if final_delta.dtype != torch.float64 or final_delta.shape != saved_delta.shape or \
                not torch.equal(final_delta.cpu(), saved_delta.detach().cpu().double()):
            raise ValueError("final completed delta differs from immutable source")
        _finite(final_delta, "immutable final completed delta")
        final = anchored_softcap_kl(payload["raw_clean"], payload["normal_clean"],
                                   payload["final_completed_raw_deltas"][key], float(payload["softcap"]))
        if abs(final - source_final) > source_tensor_atol:
            raise ValueError("final KL does not reproduce from saved final tensors")
        fresh_delta = payload["fresh_final_completed_raw_deltas"][key]
        if fresh_delta.dtype != torch.float64 or fresh_delta.shape != final_delta.shape:
            raise ValueError("fresh final diagnostic delta shape/dtype differs")
        _finite(fresh_delta, "fresh final diagnostic delta")
        fresh_value = float(payload["fresh_final_kl"][key])
        fresh_checked = anchored_softcap_kl(
            payload["fresh_final_raw_clean"], payload["fresh_final_normal_clean"],
            fresh_delta, float(payload["softcap"]))
        fresh_witness = payload["fresh_final_kl_witnesses"][key]
        fresh_terms = [float(fresh_witness[name]) for name in
                       ("logZ_candidate", "logZ_clean", "expected_shift_under_clean")]
        fresh_reduced = fresh_terms[0] - fresh_terms[1] - fresh_terms[2]
        if not all(math.isfinite(v) for v in (fresh_value, fresh_checked, fresh_reduced, *fresh_terms)) or \
                min(fresh_value, fresh_reduced) < -atol or \
                abs(fresh_value - fresh_checked) > atol or abs(fresh_value - fresh_reduced) > atol:
            raise ValueError("fresh final diagnostic KL fails tensor/witness recomputation")
        largest_fresh = max(largest_fresh, abs(fresh_value - fresh_reduced))
    for name in ("all_keep_per_position_kl", "all_keep_basis_per_position_max_abs"):
        tensor = payload[name].double()
        if tensor.shape != (length,):
            raise ValueError("all-keep witness position coverage differs")
        _finite(tensor, "all-keep prefix witness")
        if _max_abs(tensor) > float(record["all_keep_parity"]["tolerance"]):
            raise ValueError("all-keep payload fails parity")
    if record["all_keep_parity"].get("passed") is not True:
        raise ValueError("all-keep record did not pass")
    return {"status": "validated", "positions": length, "budgets": len(K_VALUES),
            "witness_vs_direct_kl_max_abs": largest,
            "saved_final_witness_vs_source_scalar_max_abs": largest_final,
            "fresh_final_witness_vs_direct_kl_max_abs": largest_fresh,
            "reported_final_origin": "immutable_saved_source_final"}
