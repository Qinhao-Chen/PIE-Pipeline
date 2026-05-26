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
        if not isinstance(synth_texts, list): return 0.0, []
    except: return 0.0, []

    # 2. Compute Acts (Circular import handling: assume compute_activations is passed or imported)
    from pie.models.activations import compute_activations 
    # For efficiency in a real loop, you'd batch this. Here is logic only.
    acts_dict = compute_activations(model, tokenizer, clt, synth_texts, {"target": feature_loc})
    s_acts = acts_dict["target"]
    
    return gini_abs(s_acts, base_acts), synth_texts

async def eval_purity(client, concept, texts, acts, seed=42):
    """Rates natural examples to check if high activation == concept presence."""
    # 1. Sample Top and Random
    n = len(texts)
    order = np.argsort(-acts)
    top_k = order[:10]
    rand_k = np.random.RandomState(seed).choice(n, 10, replace=False)
    indices = np.unique(np.concatenate([top_k, rand_k]))
    
    subset_texts = [texts[i] for i in indices]
    formatted_list = "\n".join([f"[{i}] {t}" for i, t in enumerate(subset_texts)])
    
    # 2. Rate
    msgs = [{"role": "user", "content": RATING_PROMPT.format(concept=concept, text_list=formatted_list)}]
    resp = await client.generate(msgs, temperature=0.0)
    
    try:
        ratings = ast.literal_eval(resp.text)
    except: return 0.0, 0.0
    
    y_true = np.array([1 if ratings.get(i, 0) == 2 else 0 for i in range(len(indices))])
    y_score = acts[indices]
    
    ap = average_precision(y_true, y_score)
    
    # Responsiveness: Split acts by Concept vs Non-Concept
    ac = y_score[y_true == 1]
    an = y_score[y_true == 0]
    resp_score = gini_abs(ac, an)
    
    return ap, resp_score