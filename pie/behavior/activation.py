"""Clean-activation magnitude occurrence selector from the matched revision run."""
from __future__ import annotations
import hashlib
import json
import torch
CANDIDATE_DEFINITION = "aligned_clean_corrupt_union; all_decoder_writes_tied_per_source_occurrence"
RANKING_OPERATION = "torch.sort(scores_cpu_float32,descending=True,stable=False)"
def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def rank_clean_activation_occurrences(
    activation_matrix: torch.Tensor,
    candidate_triples: list[list[int]] | list[tuple[int, int, int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[bool]]:
    """Score union occurrences from abs(clean activation), then legacy torch.sort.

    The source sparse tensor is coalesced exactly as in Activation_Baseline.py.
    Candidate occurrences absent from the clean activation dictionary receive
    score 0.0, which is the legacy behavior for corrupt-only occurrences.
    Scores are a CPU float32 vector in candidate-triple order.
    """
    if activation_matrix.layout != torch.sparse_coo:
        raise TypeError("clean CLT activation matrix must be a sparse COO tensor")
    normalized = [tuple(int(value) for value in triple) for triple in candidate_triples]
    if any(len(triple) != 3 or any(value < 0 for value in triple) for triple in normalized):
        raise ValueError("candidate triples must be nonnegative [layer,position,feature] values")
    if normalized != sorted(normalized) or len(normalized) != len(set(normalized)):
        raise ValueError("candidate triples must be sorted and unique")

    clean = activation_matrix.coalesce()
    clean_indices = clean.indices().detach().to(device="cpu").T.tolist()
    clean_values = clean.values().detach().abs().to(device="cpu", dtype=torch.float32)
    if not bool(torch.isfinite(clean_values).all()):
        raise ValueError("clean activation magnitudes contain nonfinite values")
    clean_map = {
        tuple(int(value) for value in triple): float(magnitude)
        for triple, magnitude in zip(clean_indices, clean_values.tolist())
    }
    present = [triple in clean_map for triple in normalized]
    scores = torch.tensor([clean_map.get(triple, 0.0) for triple in normalized],
                          dtype=torch.float32, device="cpu")
    if not bool(torch.isfinite(scores).all()) or bool((scores < 0).any()):
        raise ValueError("activation-magnitude score vector is invalid")
    sorted_scores, sorted_indices = torch.sort(scores, descending=True, stable=False)
    return scores, sorted_scores, sorted_indices, present


def candidate_universe_record(candidate_triples, scores: torch.Tensor,
                              ranked_indices: torch.Tensor,
                              clean_present: list[bool]) -> dict:
    triples = [[int(value) for value in triple] for triple in candidate_triples]
    if scores.ndim != 1 or len(triples) != scores.numel() or len(clean_present) != len(triples):
        raise ValueError("candidate scores, triples, and clean-presence flags are misaligned")
    if not torch.equal(torch.sort(ranked_indices).values,
                       torch.arange(len(triples), dtype=torch.long)):
        raise ValueError("activation ranking is not a permutation of candidate indices")
    score_values = [float(value) for value in scores.tolist()]
    rows = [
        {"triple": triple, "clean_present": bool(clean_present[index]),
         "clean_activation_magnitude": score_values[index]}
        for index, triple in enumerate(triples)
    ]
    ranking = [int(index) for index in ranked_indices.tolist()]
    return {
        "definition": CANDIDATE_DEFINITION,
        "count": len(triples), "rows": rows,
        "rows_sha256": fingerprint(rows),
        "score_vector_sha256": fingerprint(score_values),
        "ranking_operation": RANKING_OPERATION,
        "ranked_candidate_indices": ranking,
        "ranking_sha256": fingerprint(ranking),
    }


def select_topk_rows(candidate_triples: list[list[int]], scores: torch.Tensor,
                     ranked_indices: torch.Tensor, k: int) -> tuple[torch.Tensor, list[dict], str]:
    """Return the exact legacy rank-prefix keep mask, output rows, and mask hash."""
    if type(k) is not int or k < 0 or k > len(candidate_triples):
        raise ValueError(f"cannot select exact K={k} from {len(candidate_triples)} candidates")
    if scores.dtype != torch.float32 or scores.device.type != "cpu":
        raise ValueError("selection scores must be a CPU float32 vector")
    if ranked_indices.dtype != torch.long or ranked_indices.device.type != "cpu":
        raise ValueError("ranking indices must be a CPU int64 vector")
    chosen = [int(index) for index in ranked_indices[:k].tolist()]
    if len(chosen) != k or len(set(chosen)) != k:
        raise ValueError("torch.sort prefix is not an exact-K unique candidate set")
    keep = torch.zeros(len(candidate_triples), dtype=torch.bool, device="cpu")
    keep[torch.tensor(chosen, dtype=torch.long)] = True
    selected_rows = []
    for rank, index in enumerate(chosen):
        magnitude = float(scores[index].item())
        selected_rows.append({
            "rank": rank,
            "triple": [int(value) for value in candidate_triples[index]],
            "score_signed": magnitude,
            "score_abs": magnitude,
        })
    canonical_mask = sorted(row["triple"] for row in selected_rows)
    return keep, selected_rows, fingerprint(canonical_mask)
