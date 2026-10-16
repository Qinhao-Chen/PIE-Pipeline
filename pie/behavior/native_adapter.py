"""Task C adapter for the installed circuit_tracer native graph pruning API.

The graph node mask is the native result.  Projecting its retained feature nodes
to occurrences is an explicit, separate operation; it does not turn native
graph edge pruning into an executable intervention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import torch


@dataclass(frozen=True)
class GraphShape:
    features: int
    errors: int
    embeddings: int
    logits: int
    nodes: int


def graph_shape(graph: Any) -> GraphShape:
    """Validate the installed Graph layout before interpreting native masks."""
    features = int(len(graph.selected_features))
    positions = int(len(graph.input_tokens))
    layers = int(graph.cfg.n_layers)
    logits = int(len(graph.logit_tokens))
    if positions < 1 or layers < 1 or logits < 1:
        raise ValueError("graph must have positions, layers, and logit nodes")
    if graph.active_features.ndim != 2 or graph.active_features.shape[1] != 3:
        raise ValueError("active_features must contain (layer, position, feature) triples")
    selected = graph.selected_features.to(dtype=torch.long, device="cpu")
    if selected.ndim != 1 or len(selected) != features:
        raise ValueError("selected_features must be a one-dimensional index tensor")
    if features and (selected.min() < 0 or selected.max() >= len(graph.active_features)):
        raise ValueError("selected_features indexes outside active_features")
    if selected.unique().numel() != features:
        raise ValueError("selected_features contains duplicate indexes")
    errors = layers * positions
    nodes = features + errors + positions + logits
    if tuple(graph.adjacency_matrix.shape) != (nodes, nodes):
        raise ValueError("adjacency shape does not match the installed graph node layout")
    if len(graph.logit_probabilities) != logits:
        raise ValueError("logit probability count differs from logit node count")
    return GraphShape(features, errors, positions, logits, nodes)


def graph_node_records(graph: Any) -> list[dict[str, Any]]:
    """Map every native node index without merging feature identities or positions."""
    shape = graph_shape(graph)
    selected = graph.selected_features.long().to(graph.active_features.device)
    chosen = graph.active_features[selected].cpu().tolist()
    records: list[dict[str, Any]] = []
    for index, (layer, position, feature_index) in enumerate(chosen):
        records.append({"node_index": index, "node_type": "feature", "source_layer": int(layer),
                        "source_position": int(position), "feature_index": int(feature_index),
                        "native_active_index": int(graph.selected_features[index])})
    for offset in range(shape.errors):
        layer, position = divmod(offset, shape.embeddings)
        records.append({"node_index": shape.features + offset, "node_type": "error",
                        "layer": layer, "position": position})
    start = shape.features + shape.errors
    for position in range(shape.embeddings):
        records.append({"node_index": start + position, "node_type": "embedding",
                        "position": position})
    start += shape.embeddings
    for offset in range(shape.logits):
        records.append({"node_index": start + offset, "node_type": "logit",
                        "position": shape.embeddings - 1,
                        "vocab_index": int(graph.logit_tokens[offset]),
                        "probability": float(graph.logit_probabilities[offset])})
    return records


def prune_native(graph: Any, node_threshold: float, edge_threshold: float) -> Any:
    """Call the installed author's function exactly, preserving both native knobs."""
    from circuit_tracer.graph import prune_graph

    graph_shape(graph)
    return prune_graph(graph, node_threshold=node_threshold, edge_threshold=edge_threshold)


def project_feature_occurrences(graph: Any, native_result: Any) -> tuple[tuple[int, int, int], ...]:
    """Project final retained *feature nodes* only, after native edge cleanup."""
    shape = graph_shape(graph)
    mask = native_result.node_mask
    if mask.ndim != 1 or mask.numel() != shape.nodes or mask.dtype != torch.bool:
        raise ValueError("native node mask shape or type is invalid")
    selected = graph.selected_features.long().to(graph.active_features.device)
    triples = graph.active_features[selected].cpu().tolist()
    return tuple(sorted({tuple(int(x) for x in triples[i])
                         for i in range(shape.features) if bool(mask[i])}))


def check_projection_universe(
    projected: Iterable[tuple[int, int, int]],
    common_universe: Iterable[tuple[int, int, int]],
) -> tuple[tuple[int, int, int], ...]:
    """Return unmapped graph nodes; callers must block common evaluation if nonempty."""
    universe = set(common_universe)
    return tuple(sorted(set(projected) - universe))


def feature_key(model_revision: str, clt_revision: str, occurrence: tuple[int, int, int]) -> list[Any]:
    if not model_revision or not clt_revision:
        raise ValueError("model and CLT revisions are mandatory for joinable keys")
    layer, _position, feature_index = occurrence
    return [model_revision, clt_revision, layer, feature_index]


def occurrence_key(
    pair_dataset_sha256: str,
    prompt_pair_id: int | str,
    model_revision: str,
    clt_revision: str,
    occurrence: tuple[int, int, int],
) -> list[Any]:
    if len(pair_dataset_sha256) != 64:
        raise ValueError("pair dataset SHA256 is mandatory for joinable keys")
    return [pair_dataset_sha256, prompt_pair_id, occurrence[1],
            feature_key(model_revision, clt_revision, occurrence)]


def native_counts(graph: Any, native_result: Any) -> dict[str, int]:
    shape = graph_shape(graph)
    if native_result.edge_mask.shape != graph.adjacency_matrix.shape:
        raise ValueError("native edge mask has wrong shape")
    projected = project_feature_occurrences(graph, native_result)
    return {
        "retained_feature_occurrences": len(projected),
        "retained_unique_identities": len({(layer, feature) for layer, _position, feature in projected}),
        "retained_edges": int(native_result.edge_mask.sum().item()),
        "retained_error_nodes": int(native_result.node_mask[shape.features:shape.features + shape.errors].sum().item()),
        "retained_embedding_nodes": int(native_result.node_mask[shape.features + shape.errors:-shape.logits].sum().item()),
        "retained_logit_nodes": int(native_result.node_mask[-shape.logits:].sum().item()),
    }


def encode_native_mask(native_result: Any) -> dict[str, Any]:
    """Losslessly store a sparse edge mask without a dense N-by-N bool file."""
    edge_mask = native_result.edge_mask.cpu()
    if edge_mask.ndim != 2 or edge_mask.shape[0] != edge_mask.shape[1]:
        raise ValueError("native edge mask must be square")
    return {
        "encoding": "edge_indices_target_source_v1",
        "node_mask": native_result.node_mask.cpu(),
        "edge_indices": edge_mask.nonzero(as_tuple=False).to(torch.int32),
        "edge_shape": list(edge_mask.shape),
        "cumulative_scores": native_result.cumulative_scores.cpu(),
    }


def decode_native_edge_mask(record: dict[str, Any]) -> torch.Tensor:
    """Read current sparse files and the earlier dense fixture/pilot files."""
    if "edge_mask" in record:
        return record["edge_mask"].bool()
    if record.get("encoding") != "edge_indices_target_source_v1":
        raise ValueError("unsupported native mask encoding")
    shape = tuple(int(x) for x in record["edge_shape"])
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError("invalid edge shape")
    indices = record["edge_indices"].long()
    if indices.ndim != 2 or indices.shape[1] != 2:
        raise ValueError("edge indices must be (target,source) pairs")
    mask = torch.zeros(shape, dtype=torch.bool)
    if indices.numel():
        if indices.min() < 0 or indices[:, 0].max() >= shape[0] or indices[:, 1].max() >= shape[1]:
            raise ValueError("edge index outside native matrix")
        mask[indices[:, 0], indices[:, 1]] = True
    return mask
