"""Task-local bridge from legacy aligned FAP rows to occurrence-atomic search.

This module keeps row order unchanged. It does not claim that historical
per-prompt feature-identity lists contain enough information to reconstruct
occurrence-level sets; the aligned tensors must come from the live FAP pass or
a compatible saved cache.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from .search import SelectionView, occurrence_key


@dataclass(frozen=True)
class AlignedFAPPrompt:
    ordered_keys: tuple[str, ...]
    universe: dict[str, tuple[str, ...]]
    attribution: dict[str, float]
    initial: tuple[str, ...]

    def mask_for(self, view: SelectionView) -> tuple[bool, ...]:
        if not view.retained_occurrences <= self.universe.keys():
            raise ValueError("selection includes unknown occurrences")
        expected_writes = frozenset(w for k in view.retained_occurrences for w in self.universe[k])
        if view.retained_write_ids != expected_writes:
            raise ValueError("selection does not tie all decoder writes to occurrences")
        return tuple(k in view.retained_occurrences for k in self.ordered_keys)

    def scorer(self, evaluate_keep_mask: Callable[[tuple[bool, ...]], float]):
        return lambda view: evaluate_keep_mask(self.mask_for(view))


def from_aligned_rows(
    *, pair_dataset_sha256: str, prompt_pair_id: str,
    model_revision: str, clt_revision: str,
    source_layers: Sequence[int], source_positions: Sequence[int],
    feature_indices: Sequence[int], receiver_layers: Sequence[int],
    receiver_positions: Sequence[int], encoder_to_decoder: Sequence[int],
    base_scores: Sequence[float], keep_base: Sequence[bool],
) -> AlignedFAPPrompt:
    """Map `_union_delta_alignment` outputs plus original global FAP mask.

    `encoder_to_decoder` maps each decoder-write row to an encoder occurrence
    index. Each write ID names its original row, receiver layer and position.
    """
    n = len(source_layers)
    if not all(len(x) == n for x in (source_positions, feature_indices,
                                    base_scores, keep_base)):
        raise ValueError("encoder occurrence array length mismatch")
    r = len(receiver_layers)
    if len(receiver_positions) != r or len(encoder_to_decoder) != r:
        raise ValueError("decoder row array length mismatch")
    ordered_keys = tuple(occurrence_key([pair_dataset_sha256, prompt_pair_id,
                                         int(source_positions[i]),
                                         [model_revision, clt_revision,
                                          int(source_layers[i]), int(feature_indices[i])]])
                         for i in range(n))
    if len(set(ordered_keys)) != n:
        raise ValueError("duplicate aligned occurrence keys")
    rows: dict[str, list[str]] = {k: [] for k in ordered_keys}
    for row_index, encoder_index in enumerate(encoder_to_decoder):
        if not 0 <= int(encoder_index) < n:
            raise ValueError("encoder-to-decoder mapping out of range")
        write_id = f"row={row_index};receiver={int(receiver_layers[row_index])}:{int(receiver_positions[row_index])}"
        rows[ordered_keys[int(encoder_index)]].append(write_id)
    if any(not value for value in rows.values()):
        raise ValueError("occurrence without decoder writes")
    return AlignedFAPPrompt(
        ordered_keys=ordered_keys,
        universe={key: tuple(value) for key, value in rows.items()},
        attribution={key: float(score) for key, score in zip(ordered_keys, base_scores)},
        initial=tuple(key for key, keep in zip(ordered_keys, keep_base) if keep),
    )
