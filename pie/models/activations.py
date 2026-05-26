import torch
import numpy as np
from tqdm import tqdm

@torch.inference_mode()
def compute_activations(model, tokenizer, clt, texts, feature_locs, batch_size=64, device="cuda"):
    """
    Computes max-activation of specific features over a list of texts.
    feature_locs: {fid: {'layer': int, 'index': int}}
    """
    by_layer = {}
    for fid, loc in feature_locs.items():
        by_layer.setdefault(loc['layer'], []).append((fid, loc['index']))
    W_enc, b_enc = {}, {}
    dtype = clt.W_enc[0].dtype
    for L in by_layer:
        W_enc[L] = clt._get_encoder_weights(L).to(device, dtype)
        b_enc[L] = clt.b_enc[L].to(device, dtype)

    results = {fid: np.zeros(len(texts), dtype=np.float32) for fid in feature_locs}
    
    for start in tqdm(range(0, len(texts), batch_size), desc="Computing Acts"):
        batch = texts[start : start + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        out = model(**enc, output_hidden_states=True)

        
        for L, feats in by_layer.items():
            h = out.hidden_states[L + 1].to(dtype) 
            f_idxs = torch.tensor([f[1] for f in feats], device=device)
            W = W_enc[L].index_select(0, f_idxs)
            b = b_enc[L].index_select(0, f_idxs)
            pre_act = torch.einsum("btd,fd->btf", h, W) + b.view(1, 1, -1)
            act = torch.relu(pre_act)
            max_act = act.max(dim=1).values.float().cpu().numpy()
            
            for i, (fid, _) in enumerate(feats):
                results[fid][start : start + len(batch)] = max_act[:, i]
                
    return results