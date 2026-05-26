import torch
import torch.nn.functional as F
from circuit_tracer import ReplacementModel
from typing import Tuple

def to_dtype(dtype_str: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32, "fp32": torch.float32,
        "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
        "float16": torch.float16, "fp16": torch.float16,
    }
    if dtype_str not in mapping:
        raise ValueError(f"Unsupported dtype {dtype_str}")
    return mapping[dtype_str]

def behavior_kl(logits_p: torch.Tensor, logits_q: torch.Tensor, last_token: bool = False) -> torch.Tensor:
    if last_token:
        if logits_p.ndim == 3:
            logits_p = logits_p[:, -1]
            logits_q = logits_q[:, -1]
    else:
        if logits_p.ndim == 2:
            logits_p = logits_p.unsqueeze(0)
            logits_q = logits_q.unsqueeze(0)
    logp = F.log_softmax(logits_p, dim=-1)
    logq = F.log_softmax(logits_q, dim=-1)
    p = logp.exp()
    kl = (p * (logp - logq)).sum(dim=-1)
    return kl.mean()

def cantor_unpair(z: int) -> Tuple[int, int]:
    w = int((int((8 * z + 1) ** 0.5) - 1) // 2)
    t = (w * w + w) // 2
    y = z - t
    x = w - y
    return x, y

def cantor_pair(x: int, y: int) -> int:
    return (x + y) * (x + y + 1) // 2 + y

def truncate_to_tokens(model: ReplacementModel, text: str, max_tokens: int) -> torch.Tensor:
    t = model.ensure_tokenized(text)
    return t[:max_tokens] if t.numel() > max_tokens else t