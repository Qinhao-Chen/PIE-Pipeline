"""Exhaustive single-occurrence FActP ablation and exact independent-K selection."""
from __future__ import annotations
import hashlib
import json
import math
import torch
def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _native_sequence_logsumexp(logits: torch.Tensor) -> float:
    """Match ACDC_CLT3.metric_factory's default scalar operation exactly."""
    return float(logits.logsumexp(dim=-1).sum().item())


def make_score_vectors(
    clean_metric_native: float,
    ablated_metrics_native: list[float],
) -> tuple[list[float], torch.Tensor]:
    """Return signed ablation deltas and the legacy CPU float32 abs scores."""
    clean_metric_native = float(clean_metric_native)
    if not math.isfinite(clean_metric_native):
        raise ValueError("clean sequence-logsumexp metric is nonfinite")
    signed = []
    scores = torch.empty(len(ablated_metrics_native), dtype=torch.float32, device="cpu")
    for index, raw_value in enumerate(ablated_metrics_native):
        ablated = float(raw_value)
        if not math.isfinite(ablated):
            raise ValueError(f"candidate {index} has a nonfinite ablated metric")
        delta = ablated - clean_metric_native
        if not math.isfinite(delta):
            raise ValueError(f"candidate {index} has a nonfinite signed delta")
        signed.append(delta)
        # Assignment reproduces CLT3's CPU float32 score vector.
        scores[index] = abs(delta)
    if not bool(torch.isfinite(scores).all()):
        raise ValueError("FActP float32 score vector is nonfinite")
    return signed, scores


def select_topk_occurrences(
    triples: list[list[int]],
    signed_deltas: list[float],
    scores_cpu_float32: torch.Tensor,
    k: int,
) -> tuple[torch.Tensor, list[dict], str]:
    """Use the source's exact independent ``torch.topk`` selection operation."""
    if scores_cpu_float32.dtype != torch.float32 or scores_cpu_float32.device.type != "cpu":
        raise ValueError("FActP scores must be a CPU float32 vector")
    if len(triples) != scores_cpu_float32.numel() or len(signed_deltas) != len(triples):
        raise ValueError("FActP triples and scores are misaligned")
    if type(k) is not int or not 0 <= k <= len(triples):
        raise ValueError(f"cannot select exact K={k} from {len(triples)} candidates")
    _values, indices = torch.topk(scores_cpu_float32, k, largest=True, sorted=True)
    chosen = [int(index) for index in indices.tolist()]
    if len(chosen) != k or len(set(chosen)) != k:
        raise ValueError(f"torch.topk returned an invalid exact-K mask at K={k}")
    keep = torch.zeros(len(triples), dtype=torch.bool, device="cpu")
    if chosen:
        keep[torch.tensor(chosen, dtype=torch.long)] = True
    selected_rows = []
    for rank, candidate_index in enumerate(chosen):
        selected_rows.append({
            "rank": rank,
            "candidate_index": candidate_index,
            "triple": [int(value) for value in triples[candidate_index]],
            "score_signed": float(signed_deltas[candidate_index]),
            "score_abs": float(scores_cpu_float32[candidate_index].item()),
        })
    selected_mask_sha256 = fingerprint(sorted(row["triple"] for row in selected_rows))
    return keep, selected_rows, selected_mask_sha256


class SingleOccurrencePatchBuilder:
    """Cache decoder-write coordinates once and build exact one-occurrence patches.

    ``build_removed_feature_patches`` allocates and scans all decoder rows for
    every candidate. This class preserves its CPU float32 accumulation order,
    target-layer shapes, and tied-write semantics while grouping row indices
    once, so exhaustive FActP scoring avoids repeated full-matrix transfers and
    masks.
    """

    def __init__(
        self,
        layer_ids: torch.Tensor,
        pos_ids: torch.Tensor,
        enc2rows: torch.Tensor,
        delta_dec: torch.Tensor,
        n_occurrences: int,
        n_layers: int,
    ) -> None:
        self.layer_ids = layer_ids.detach().to(device="cpu", dtype=torch.long).contiguous()
        self.pos_ids = pos_ids.detach().to(device="cpu", dtype=torch.long).contiguous()
        self.enc2rows = enc2rows.detach().to(device="cpu", dtype=torch.long).contiguous()
        self.delta = delta_dec.detach().to(device="cpu").contiguous()
        self.n_occurrences = int(n_occurrences)
        self.n_layers = int(n_layers)
        if self.delta.ndim != 2:
            raise ValueError("aligned decoder delta must be a rank-2 tensor")
        row_count = self.delta.shape[0]
        if any(vector.ndim != 1 or vector.numel() != row_count for vector in (
                self.layer_ids, self.pos_ids, self.enc2rows)):
            raise ValueError("decoder-write row maps have inconsistent lengths")
        if self.n_occurrences <= 0 or self.n_layers <= 0:
            raise ValueError("FActP requires a nonempty occurrence/layer universe")
        if bool((self.enc2rows < 0).any()) or bool((self.enc2rows >= self.n_occurrences).any()):
            raise ValueError("decoder write refers to an occurrence outside the candidate universe")
        if bool((self.layer_ids < 0).any()) or bool((self.layer_ids >= self.n_layers).any()):
            raise ValueError("decoder write refers to an invalid target layer")
        if bool((self.pos_ids < 0).any()):
            raise ValueError("decoder write has a negative target position")
        if not bool(torch.isfinite(self.delta.to(torch.float32)).all()):
            raise ValueError("aligned decoder deltas contain nonfinite values")

        self.shapes = []
        for layer in range(self.n_layers):
            mask = self.layer_ids == layer
            length = int(self.pos_ids[mask].max().item()) + 1 if bool(mask.any()) else 0
            self.shapes.append((length, int(self.delta.shape[1])))
        self.rows_by_occurrence: list[list[int]] = [[] for _ in range(self.n_occurrences)]
        for row_index, occurrence_index in enumerate(self.enc2rows.tolist()):
            self.rows_by_occurrence[int(occurrence_index)].append(row_index)

    def build_single_occurrence(self, occurrence_index: int) -> list[torch.Tensor]:
        if type(occurrence_index) is not int or not 0 <= occurrence_index < self.n_occurrences:
            raise ValueError("single-occurrence patch index is outside the candidate universe")
        patches = [torch.zeros(shape, dtype=torch.float32, device="cpu") for shape in self.shapes]
        for row_index in self.rows_by_occurrence[occurrence_index]:
            layer = int(self.layer_ids[row_index].item())
            position = int(self.pos_ids[row_index].item())
            patches[layer][position].add_(self.delta[row_index].to(torch.float32))
        return patches

    def build_all_retained(self) -> list[torch.Tensor]:
        return [torch.zeros(shape, dtype=torch.float32, device="cpu") for shape in self.shapes]
