import numpy as np
import ast
from pie.evaluation.metrics import gini_abs, average_precision

CLARITY_PROMPT = """Concept: "{concept}"
Generate a Python list of 5 diverse strings that express this concept clearly.
Output ONLY the list. Example: ["sentence 1", "sentence 2"]"""

RATING_PROMPT = """Concept: "{concept}"
Rate how well each sequence expresses the concept (0=Not expressed, 1=Vague, 2=Clear).
Input:
{text_list}
Output a Python dictionary {{id: rating}}. Output ONLY the dict."""

async def eval_clarity(client, concept, model, tokenizer, clt, feature_loc, base_acts):
    """Generates synthetic examples and compares activations against baseline."""
    # 1. Generate
    msgs = [{"role": "user", "content": CLARITY_PROMPT.format(concept=concept)}]
    resp = await client.generate(msgs, temperature=0.7)
    try:
        synth_texts = ast.literal_eval(resp.text)
    except (ValueError, SyntaxError) as error:
        raise ValueError("Invalid synthetic-example response") from error
    if not isinstance(synth_texts, list) or len(synth_texts) != 5 or any(not isinstance(text, str) or not text.strip() for text in synth_texts):
        raise ValueError("Clarity requires five nonempty synthetic texts")

    # 2. Compute Acts (Circular import handling: assume compute_activations is passed or imported)
    from pie.models.activations import compute_activations 
    # For efficiency in a real loop, you'd batch this. Here is logic only.
    acts_dict = compute_activations(model, tokenizer, clt, synth_texts, {"target": feature_loc})
    s_acts = acts_dict["target"]
    
    return gini_abs(s_acts, base_acts) if s_acts.size and base_acts.size else None, synth_texts

async def eval_purity(client, concept, texts, acts, seed=42):
    """Rates natural examples to check if high activation == concept presence."""
    # 1. Sample Top and Random
    n = len(texts)
    if n == 0 or len(acts) != n:
        raise ValueError("Natural texts and activations must be nonempty and aligned")
    if not np.isfinite(acts).all():
        raise ValueError("Natural activations must be finite")
    order = np.argsort(-acts)
    top_k = order[:10]
    rand_k = np.random.RandomState(seed).choice(n, min(10, n), replace=False)
    indices = np.unique(np.concatenate([top_k, rand_k]))
    
    subset_texts = [texts[i] for i in indices]
    formatted_list = "\n".join([f"[{i}] {t}" for i, t in enumerate(subset_texts)])
    
    # 2. Rate
    msgs = [{"role": "user", "content": RATING_PROMPT.format(concept=concept, text_list=formatted_list)}]
    resp = await client.generate(msgs, temperature=0.0)
    
    try:
        ratings = ast.literal_eval(resp.text)
    except (ValueError, SyntaxError) as error:
        raise ValueError("Invalid concept-rating response") from error
    if not isinstance(ratings, dict):
        raise ValueError("Expected a dictionary of concept ratings")
    normalized = {str(key): value for key, value in ratings.items()}
    if len(normalized) != len(ratings) or set(normalized) != {str(i) for i in range(len(indices))}:
        raise ValueError("Concept ratings contain missing or extra example IDs")
    if any(type(value) is not int or value not in (0, 1, 2) for value in normalized.values()):
        raise ValueError("Concept ratings must be integers 0, 1 or 2")
    y_true = np.array([1 if normalized[str(i)] == 2 else 0 for i in range(len(indices))])
    y_score = acts[indices]
    
    ap = average_precision(y_true, y_score) if y_true.sum() else None
    
    # Responsiveness: Split acts by Concept vs Non-Concept
    ac = y_score[y_true == 1]
    an = y_score[y_true == 0]
    resp_score = gini_abs(ac, an) if ac.size and an.size else None
    
    return ap, resp_score
