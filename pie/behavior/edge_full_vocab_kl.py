"""Full-vocabulary completion of a circuit-tracer local edge-mask replay.

The saved graph has edge rows for clean-active features and selected output
tokens.  The pinned attribution context can additionally measure the direct
effect of its clean source nodes on each coordinate of the final residual
stream.  Those coordinates supply the untracked-vocabulary logit changes.

This is an explicitly defined local graph execution, not an edge-gated
Transformer forward.  In particular, the native mask contains no output
edges for unselected vocabulary tokens; they keep all direct edges from the
replayed feature sources under the completion rule below.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .edge_replay_kl import _check_causal_feature_edges, _check_graph


@dataclass(frozen=True)
class FeatureReplay:
    clean_activations: torch.Tensor
    replayed_activations: torch.Tensor
    graph_logit_deltas: torch.Tensor
    node_only_logit_deltas: torch.Tensor
    source_scales: torch.Tensor


def replay_feature_graph(graph, node_mask: torch.Tensor,
                         edge_mask: torch.Tensor) -> FeatureReplay:
    """Run native masks through signed feature edges in causal layer order."""
    n_features, n_logits = _check_graph(graph, node_mask, edge_mask)
    _check_causal_feature_edges(graph, n_features)
    adjacency = graph.adjacency_matrix.to(torch.float64)
    clean = graph.activation_values[graph.selected_features.long()].to(torch.float64)
    if clean.shape != (n_features,) or not bool(torch.isfinite(clean).all()) or bool((clean <= 0).any()):
        raise ValueError("graph features must be finite and clean-active")
    layers = graph.active_features[graph.selected_features.long(), 0].long()
    scale = torch.ones(adjacency.shape[0], dtype=torch.float64)
    scale[:n_features] = 0.0
    # Native pruning can remove error/embedding nodes too.  Their unmasked
    # contributions must disappear for output directions absent from the graph.
    scale[n_features:-n_logits] = node_mask[n_features:-n_logits].to(torch.float64)
    replayed = torch.zeros_like(clean)
    for index in torch.argsort(layers, stable=True).tolist():
        if not bool(node_mask[index]):
            continue
        change = torch.dot(adjacency[index],
                           edge_mask[index].to(torch.float64) * scale - 1.0)
        replayed[index] = torch.relu(clean[index] + change)
        if not bool(torch.isfinite(replayed[index])):
            raise ValueError("nonfinite replayed feature")
        scale[index] = replayed[index] / clean[index]
    graph_deltas = torch.sum(
        adjacency[-n_logits:] *
        (edge_mask[-n_logits:].to(torch.float64) * scale[None, :] - 1.0), dim=1)
    node_only_deltas = torch.sum(
        adjacency[-n_logits:] * (scale[None, :] - 1.0), dim=1)
    if not bool(torch.isfinite(graph_deltas).all()):
        raise ValueError("nonfinite output-graph delta")
    return FeatureReplay(clean, replayed, graph_deltas, node_only_deltas, scale)


def graph_source_weights(graph, node_mask: torch.Tensor,
                         context_feature_indices: torch.Tensor,
                         context_feature_activations: torch.Tensor,
                         replay: FeatureReplay) -> tuple[torch.Tensor, dict]:
    """Map saved graph activation changes to fresh attribution-source weights.

    Fresh decoder source vectors already contain their measured activation,
    so dividing by that activation converts a desired absolute activation
    change to a coefficient on the source vector.  A full-mask replay has all
    zero weights even if two BF16 clean forwards differ slightly.
    """
    graph_indices = graph.active_features[graph.selected_features.long()].long()
    current_indices = context_feature_indices.long()
    current_acts = context_feature_activations.to(torch.float64)
    if current_indices.ndim != 2 or current_indices.shape[1] != 3:
        raise ValueError("current feature indices must be [layer, position, feature]")
    if current_acts.shape != (len(current_indices),):
        raise ValueError("current feature activation shape mismatch")
    if not bool(torch.isfinite(current_acts).all()) or bool((current_acts <= 0).any()):
        raise ValueError("current features must be finite and active")
    current_map = {tuple(map(int, triple)): i for i, triple in enumerate(current_indices.tolist())}
    if len(current_map) != len(current_indices):
        raise ValueError("duplicate feature occurrence in fresh context")
    n_logits = len(graph.logit_tokens)
    n_nonfeatures = (graph.cfg.n_layers + 1) * graph.input_tokens.numel()
    if graph.adjacency_matrix.shape[0] != len(graph_indices) + n_nonfeatures + n_logits:
        raise ValueError("graph non-feature node layout mismatch")
    if node_mask.shape != (graph.adjacency_matrix.shape[0],):
        raise ValueError("graph node mask mismatch")
    weights = torch.zeros(len(current_indices) + n_nonfeatures, dtype=torch.float64)
    missing = 0
    nonzero_missing = 0
    activation_errors = []
    changed_count = 0
    for index, triple in enumerate(graph_indices.tolist()):
        key = tuple(map(int, triple))
        current = current_map.get(key)
        delta = float(replay.replayed_activations[index] - replay.clean_activations[index])
        if current is None:
            missing += 1
            if abs(delta) > 1e-9:
                nonzero_missing += 1
            continue
        activation_errors.append(abs(float(current_acts[current] - replay.clean_activations[index])))
        weights[current] = delta / current_acts[current]
        if abs(delta) > 1e-9:
            changed_count += 1
    if missing:
        # A missing clean graph source prevents accurate mean-unembedding
        # reconstruction of that source's saved top-token output edges.
        raise ValueError(f"{missing} graph sources absent from fresh context ({nonzero_missing} changed)")
    weights[len(current_indices):] = (
        node_mask[len(graph_indices):-n_logits].to(torch.float64) - 1.0)
    diagnostics = {
        "graph_features": len(graph_indices),
        "current_features": len(current_indices),
        "missing_graph_features": missing,
        "changed_features": changed_count,
        "removed_error_or_embedding_nodes": int((weights[len(current_indices):] == -1).sum()),
        "median_clean_activation_abs_difference":
            float(torch.tensor(activation_errors, dtype=torch.float64).median()) if activation_errors else None,
        "max_clean_activation_abs_difference": max(activation_errors) if activation_errors else None,
    }
    return weights, diagnostics


def complete_raw_logit_delta(direct_full_delta: torch.Tensor,
                             selected_token_ids: torch.Tensor,
                             graph_selected_raw_deltas: torch.Tensor) -> torch.Tensor:
    """Apply exact saved terminal-edge deltas at selected graph logits.

    `direct_full_delta` is the raw unembedding effect of changed graph source
    nodes.  The selected raw deltas are reconstructed from saved centered
    graph edge values and each source's mean-unembedding contribution.
    Unselected tokens keep their full direct source effect.
    """
    direct = direct_full_delta.to(torch.float64).flatten()
    indices = selected_token_ids.long().flatten()
    selected = graph_selected_raw_deltas.to(torch.float64).flatten()
    if direct.ndim != 1 or len(indices) != len(selected) or len(indices.unique()) != len(indices):
        raise ValueError("output-token/delta shape or uniqueness mismatch")
    if bool((indices < 0).any()) or bool((indices >= direct.numel()).any()):
        raise ValueError("selected output token outside vocabulary")
    if not bool(torch.isfinite(direct).all()) or not bool(torch.isfinite(selected).all()):
        raise ValueError("nonfinite output delta")
    result = direct.clone()
    result[indices] = selected
    return result


def raw_selected_logit_deltas(graph, edge_mask: torch.Tensor,
                              replay: FeatureReplay,
                              mean_source_effects: torch.Tensor) -> torch.Tensor:
    """Execute native terminal edge masks in the raw pre-softcap logit gauge.

    Original edge strengths are relative to vocabulary-mean unembedding.
    The attribution context supplies that mean direction for every source.
    Only nonzero edges in the saved graph get the mean term: its zero entries
    are not treated as newly invented edges.
    """
    n_logits = len(graph.logit_tokens)
    n_sources = graph.adjacency_matrix.shape[0] - n_logits
    mean = mean_source_effects.to(torch.float64).flatten()
    if mean.shape != (n_sources,):
        raise ValueError("mean source effect shape mismatch")
    if not bool(torch.isfinite(mean).all()):
        raise ValueError("nonfinite mean source effect")
    centered = graph.adjacency_matrix[-n_logits:, :n_sources].to(torch.float64)
    raw_coefficients = torch.where(centered != 0, centered + mean[None, :], 0.0)
    factor = (edge_mask[-n_logits:, :n_sources].to(torch.float64) *
              replay.source_scales[None, :n_sources] - 1.0)
    return torch.sum(raw_coefficients * factor, dim=1)


def anchored_softcap_kl(raw_clean_logits: torch.Tensor,
                        model_clean_logits: torch.Tensor,
                        raw_delta: torch.Tensor, softcap: float) -> float:
    """Compute KL using the actual clean model output as the reference.

    The nonlinear softcap *change* is applied at the clean raw logit point.
    This makes an all-edge/all-node mask exactly reproduce the measured clean
    distribution despite BF16 differences between two clean forwards.
    """
    raw = raw_clean_logits.to(torch.float64).flatten()
    normal = model_clean_logits.to(torch.float64).flatten()
    delta = raw_delta.to(torch.float64).flatten()
    if raw.shape != normal.shape or raw.shape != delta.shape:
        raise ValueError("raw, normal and delta logits must match")
    if not bool(torch.isfinite(raw).all()) or not bool(torch.isfinite(normal).all()) or not bool(torch.isfinite(delta).all()):
        raise ValueError("nonfinite logit input")
    if softcap > 0:
        shift = softcap * (torch.tanh((raw + delta) / softcap) - torch.tanh(raw / softcap))
    else:
        shift = delta
    changed = normal + shift
    logp = torch.log_softmax(normal, dim=-1)
    logq = torch.log_softmax(changed, dim=-1)
    value = float(torch.sum(logp.exp() * (logp - logq)))
    if not math.isfinite(value) or value < -1e-9:
        raise ValueError("invalid graph full-vocabulary KL")
    return max(0.0, value)
