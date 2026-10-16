"""Shared final-position and sequence-mean KL measurements for FAP runs.

The caller is responsible for constructing inputs whose last real token is
the desired prediction position. In the IOI answer-prefix protocol, that
means removing the answer suffix before tokenization.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F


def dual_kl(
    logits_full: torch.Tensor,
    logits_selected: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Return per-example full-to-selected KL at the last real token and mean.

    ``logits_full`` and ``logits_selected`` may be unbatched ``[T, V]`` or
    batched ``[B, T, V]`` tensors. ``attention_mask`` is optional and marks
    real input positions with nonzero values. Padding positions are excluded
    from both reductions and sanitized before log-softmax, so even nonfinite
    padding logits cannot contaminate the result.

    The last-position field names the intended answer-prefix endpoint; callers
    must ensure that the final real input position is immediately before the
    answer when using the IOI answer-prediction protocol.
    """
    if not isinstance(logits_full, torch.Tensor) or not isinstance(logits_selected, torch.Tensor):
        raise TypeError("logits_full and logits_selected must be torch tensors")
    if logits_full.ndim not in (2, 3) or logits_selected.ndim != logits_full.ndim:
        raise ValueError("logits must both have shape [T, V] or [B, T, V]")
    if not logits_full.is_floating_point() or not logits_selected.is_floating_point():
        raise ValueError("logits must use real floating-point dtypes")
    was_unbatched = logits_full.ndim == 2
    if was_unbatched:
        logits_full = logits_full.unsqueeze(0)
        logits_selected = logits_selected.unsqueeze(0)
    if logits_full.shape != logits_selected.shape:
        raise ValueError("full and selected logits must have identical shapes")
    if logits_full.shape[0] == 0 or logits_full.shape[1] == 0 or logits_full.shape[2] == 0:
        raise ValueError("logits must contain examples, time positions, and vocabulary items")
    if logits_full.device != logits_selected.device:
        raise ValueError("full and selected logits must be on the same device")

    batch_size, sequence_length, _ = logits_full.shape
    if attention_mask is None:
        valid = torch.ones(
            (batch_size, sequence_length), dtype=torch.bool, device=logits_full.device
        )
    else:
        if not isinstance(attention_mask, torch.Tensor):
            raise TypeError("attention_mask must be a torch tensor")
        mask = attention_mask
        if mask.ndim == 1 and was_unbatched:
            mask = mask.unsqueeze(0)
        if mask.shape != (batch_size, sequence_length):
            raise ValueError(
                "attention_mask must have shape [T] for unbatched logits or [B, T] for batched logits"
            )
        if mask.is_floating_point() and not bool(torch.isfinite(mask).all().item()):
            raise ValueError("attention_mask must contain only finite values")
        valid = mask.to(device=logits_full.device).bool()

    token_count = valid.sum(dim=1)
    if bool((token_count == 0).any().item()):
        raise ValueError("each example must have at least one real input position")

    # Mask before log_softmax. This prevents arbitrary padded logits, including
    # NaNs and infinities, from creating NaNs in the positionwise KL tensor.
    valid_3d = valid.unsqueeze(-1)
    full = logits_full.to(torch.float32)
    selected = logits_selected.to(torch.float32)
    full = torch.where(valid_3d, full, torch.zeros_like(full))
    selected = torch.where(valid_3d, selected, torch.zeros_like(selected))

    valid_full_finite = torch.where(valid_3d, torch.isfinite(full), torch.ones_like(valid_3d))
    valid_selected_finite = torch.where(
        valid_3d, torch.isfinite(selected), torch.ones_like(valid_3d)
    )
    if not bool(valid_full_finite.all().item()) or not bool(valid_selected_finite.all().item()):
        raise ValueError("valid-position logits must be finite")

    log_full = F.log_softmax(full, dim=-1)
    log_selected = F.log_softmax(selected, dim=-1)
    per_position_kl = (log_full.exp() * (log_full - log_selected)).sum(dim=-1)

    positions = torch.arange(sequence_length, device=logits_full.device).expand(batch_size, -1)
    final_real_position = positions.masked_fill(~valid, -1).max(dim=1).values
    kl_pre_answer_last_token = per_position_kl.gather(
        dim=1, index=final_real_position.unsqueeze(1)
    ).squeeze(1)
    kl_sequence_mean = (per_position_kl * valid.to(per_position_kl.dtype)).sum(dim=1) / token_count

    return {
        "kl_pre_answer_last_token": kl_pre_answer_last_token,
        "kl_sequence_mean": kl_sequence_mean,
        "final_real_position": final_real_position,
        "token_count": token_count,
    }
