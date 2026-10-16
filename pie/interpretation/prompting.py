import random
import re
from typing import Dict, Any, List

SYSTEM_PROMPT = """You are a meticulous AI researcher. Analyze text examples where specific tokens are highlighted (<<token>>).
Describe the semantic or syntactic pattern common to these highlighted tokens.
- Concise description.
- Do not mention the markers << >>.
- End with [EXPLANATION]: <your description>
"""

def _highlight_tokens(tokens: List[str], acts: List[float], frac_threshold: float) -> str:
    if not tokens: return ""
    max_act = max(acts) if acts else 0.0
    thr = max_act * frac_threshold
    out = []
    i = 0
    n = min(len(tokens), len(acts))
    while i < n:
        if acts[i] > thr:
            out.append("<<")
            while i < n and acts[i] > thr:
                out.append(tokens[i]); i += 1
            out.append(">>")
        else:
            out.append(tokens[i]); i += 1
    return "".join(out)

def build_explanation_messages(feature: Dict[str, Any], n_train: int = 40) -> List[Dict[str, str]]:
    # Flatten quantiles
    examples = []
    quantiles = feature.get("examples_quantiles") or []
    # Take top bucket primarily
    if quantiles:
        examples = quantiles[0].get("examples", [])[:n_train]

    lines = []
    for i, e in enumerate(examples, 1):
        toks = e.get("tokens", [])
        acts = e.get("tokens_acts_list", [])
        txt = _highlight_tokens(toks, acts, 0.6)
        lines.append(f"Example {i}: {txt}")
    
    user_txt = "\n".join(lines) if lines else "No examples available."
    
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_txt}
    ]

def parse_explanation(text: str) -> str:
    m = re.search(r"\[EXPLANATION\]:\s*(.*)", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()