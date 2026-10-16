"""Local signed-edge replay for a circuit-tracer graph.

This is a new diagnostic executor, not an executable native graph mask.  It
uses the prompt-specific signed adjacency as a first-order model anchored at
the clean feature activations.  Graph output rows exist only for selected
top-token logits.  All other vocabulary logits are held at their clean value;
the resulting KL is therefore a top-logit graph-replay KL.

The implementation deliberately never maps a graph edge to a CLT decoder write.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class ReplayResult:
    kl_nats: float
    logit_deltas: tuple[float, ...]
    top_probability_mass: float
    retained_features: int
    retained_edges: int
    replay_positive_features: int
    maximum_feature_ratio: float


def _check_graph(graph, node_mask: torch.Tensor, edge_mask: torch.Tensor) -> tuple[int, int]:
    adjacency = graph.adjacency_matrix
    n_features = len(graph.selected_features)
    n_logits = len(graph.logit_tokens)
    n_nodes = adjacency.shape[0]
    if adjacency.ndim != 2 or adjacency.shape != (n_nodes, n_nodes):
        raise ValueError("graph adjacency must be square")
    if node_mask.shape != (n_nodes,) or node_mask.dtype != torch.bool:
        raise ValueError("node mask shape/type mismatch")
    if edge_mask.shape != (n_nodes, n_nodes) or edge_mask.dtype != torch.bool:
        raise ValueError("edge mask shape/type mismatch")
    if len(graph.selected_features) != len(graph.active_features):
        raise ValueError("replay requires an untruncated graph with every active feature")
    if n_logits < 1 or len(graph.logit_probabilities) != n_logits:
        raise ValueError("graph logit nodes/probabilities mismatch")
    if n_features + n_logits > n_nodes:
        raise ValueError("invalid graph node layout")
    if not bool(node_mask[-n_logits:].all()):
        raise ValueError("native output-logit nodes must be retained")
    if bool((edge_mask & (adjacency == 0)).any()):
        raise ValueError("native edge mask includes unmeasured/zero graph entries")
    if bool((edge_mask & ~node_mask[:, None]).any()) or bool((edge_mask & ~node_mask[None, :]).any()):
        raise ValueError("edge mask refers to removed nodes")
    return n_features, n_logits


def _check_causal_feature_edges(graph, n_features: int) -> None:
    """Feature targets may depend only on features at earlier layers."""
    layers = graph.active_features[graph.selected_features.long(), 0].long()
    # Work one target layer at a time to avoid materializing all nonzero edges.
    adjacency = graph.adjacency_matrix[:n_features, :n_features]
    for layer in layers.unique(sorted=True).tolist():
        target = layers == layer
        non_earlier_source = layers >= layer
        if bool((adjacency[target][:, non_earlier_source] != 0).any()):
            raise ValueError(f"graph has same/future-layer feature edge into layer {layer}")


def _tilted_kl(top_probabilities: torch.Tensor, logit_deltas: torch.Tensor) -> float:
    """KL(p_clean || q_tilted), with zero logit shift outside graph output nodes."""
    p = top_probabilities.to(torch.float64)
    delta = logit_deltas.to(torch.float64)
    if p.ndim != 1 or delta.shape != p.shape:
        raise ValueError("top probabilities and deltas must match")
    if not bool(torch.isfinite(p).all()) or not bool(torch.isfinite(delta).all()):
        raise ValueError("nonfinite probability or logit delta")
    if bool((p <= 0).any()):
        raise ValueError("top probabilities must be positive")
    rest = 1.0 - float(p.sum())
    if rest < 0:
        raise ValueError("rounded graph top-token probabilities exceed one")
    log_rest = math.log(rest) if rest else -math.inf
    log_terms = torch.cat((p.log() + delta, torch.tensor([log_rest], dtype=torch.float64)))
    log_z = torch.logsumexp(log_terms, dim=0)
    kl = float(log_z - torch.dot(p, delta))
    if not math.isfinite(kl) or kl < -1e-9:
        raise ValueError("invalid replay KL")
    return max(0.0, kl)  # only suppress float-roundoff below zero


def kl_from_raw_logits(raw_clean_logits: torch.Tensor, selected_token_ids: torch.Tensor,
                       selected_presoftcap_deltas: torch.Tensor,
                       softcap: float) -> float:
    """Apply graph deltas before the model's output softcap, then compute full-vocab KL.

    Unrepresented output tokens keep their clean pre-softcap logits.  This is
    still a graph-local completion rule, not an edge-masked model forward.
    """
    raw = raw_clean_logits.to(torch.float64).flatten()
    indices = selected_token_ids.long().flatten()
    delta = selected_presoftcap_deltas.to(torch.float64).flatten()
    if len(indices) != len(delta) or len(indices.unique()) != len(indices):
        raise ValueError("selected output tokens/deltas mismatch or repeat")
    if bool((indices < 0).any()) or bool((indices >= raw.numel()).any()):
        raise ValueError("selected output token outside vocabulary")
    if not bool(torch.isfinite(raw).all()) or not bool(torch.isfinite(delta).all()):
        raise ValueError("nonfinite raw logits or graph deltas")
    changed = raw.clone()
    changed[indices] += delta
    if softcap > 0:
        clean = softcap * torch.tanh(raw / softcap)
        candidate = softcap * torch.tanh(changed / softcap)
    else:
        clean, candidate = raw, changed
    logp = torch.log_softmax(clean, dim=-1)
    logq = torch.log_softmax(candidate, dim=-1)
    result = float(torch.sum(logp.exp() * (logp - logq)))
    if not math.isfinite(result) or result < -1e-9:
        raise ValueError("invalid pre-softcap graph replay KL")
    return max(0.0, result)


def replay_edge_mask(graph, node_mask: torch.Tensor, edge_mask: torch.Tensor,
                     *, check_causal: bool = True) -> ReplayResult:
    """Replay retained signed messages through feature and selected-logit rows.

    Each active clean feature has an implicit clean-point intercept.  For a
    feature target t, its new preactivation is its clean activation plus the
    change in summed direct inputs.  A source feature message scales with its
    replayed/clean activation; error and embedding inputs remain fixed.  ReLU
    is the verified activation for the pinned Gemma CLT.  Output-logit deltas
    use the same message rule.  The graph's output directions are mean-centered
    pre-softcap logits; setting untracked-token deltas to zero is a declared
    completion convention, not a model forward.
    """
    n_features, n_logits = _check_graph(graph, node_mask, edge_mask)
    if check_causal:
        _check_causal_feature_edges(graph, n_features)
    adjacency = graph.adjacency_matrix.to(torch.float64)
    activation = graph.activation_values[graph.selected_features.long()].to(torch.float64)
    if activation.shape != (n_features,) or bool((activation <= 0).any()):
        raise ValueError("selected graph features must be clean-active")
    layers = graph.active_features[graph.selected_features.long(), 0].long()
    scale = torch.ones(adjacency.shape[0], dtype=torch.float64)
    scale[:n_features] = 0.0
    replayed = torch.zeros(n_features, dtype=torch.float64)
    for index in torch.argsort(layers, stable=True).tolist():
        if not bool(node_mask[index]):
            continue
        message_change = torch.dot(
            adjacency[index], edge_mask[index].to(torch.float64) * scale - 1.0
        )
        value = torch.relu(activation[index] + message_change)
        if not bool(torch.isfinite(value)):
            raise ValueError("nonfinite replayed feature activation")
        replayed[index] = value
        scale[index] = value / activation[index]
    logit_deltas = torch.sum(
        adjacency[-n_logits:] *
        (edge_mask[-n_logits:].to(torch.float64) * scale[None, :] - 1.0),
        dim=1,
    )
    kl = _tilted_kl(graph.logit_probabilities, logit_deltas)
    return ReplayResult(
        kl_nats=kl,
        logit_deltas=tuple(float(x) for x in logit_deltas),
        top_probability_mass=float(graph.logit_probabilities.to(torch.float64).sum()),
        retained_features=int(node_mask[:n_features].sum()),
        retained_edges=int(edge_mask.sum()),
        replay_positive_features=int((replayed > 0).sum()),
        maximum_feature_ratio=float(scale[:n_features].max()) if n_features else 0.0,
    )
