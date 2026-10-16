import torch
from typing import Dict, Callable, Optional, List

def collect_downstream_grads_at_hook(model, tokens_1d, metric_fn, hook_name) -> Dict[int, torch.Tensor]:
    dev = model.cfg.device
    grad_cache, fwd_tensors = {}, {}
    def fwd_passthrough(L):
        def _id(acts, hook):
            if torch.is_floating_point(acts) and not acts.requires_grad: acts = acts * 1.0
            acts.retain_grad(); fwd_tensors[L] = acts
            return acts
        return _id
    for L in range(model.cfg.n_layers):
        model.add_hook(f"blocks.{L}.{hook_name}", fwd_passthrough(L), "fwd")
    with torch.enable_grad():
        logits = model(tokens_1d.to(dev))
        val = metric_fn(logits)
        model.zero_grad(set_to_none=True)
        val.backward()
    model.reset_hooks()
    for L, acts in fwd_tensors.items():
        if acts.grad is not None:
            g = acts.grad.squeeze(0) if acts.grad.ndim == 3 else acts.grad
            grad_cache[L] = g.to(torch.float32).detach()
    return grad_cache

def fap_score_clt(delta_dec, layer_ids, pos_ids, enc2rows, n_enc_occ, grad_mlp_out, n_layers) -> torch.Tensor:
    if n_enc_occ == 0 or delta_dec.numel() == 0: return torch.zeros((0,), dtype=torch.float32)
    R, d = delta_dec.shape
    grads_rows = torch.empty((R, d), dtype=torch.float32, device="cpu")
    for t in range(n_layers):
        mask_t = (layer_ids.cpu() == t)
        if not mask_t.any(): continue
        pos_t = pos_ids.cpu()[mask_t]
        gmat = grad_mlp_out.get(t)
        if gmat is not None:
            grads_rows[mask_t] = gmat.to("cpu", dtype=torch.float32).index_select(0, pos_t)
        else:
            grads_rows[mask_t] = 0.0
    scores = torch.zeros((n_enc_occ,), dtype=torch.float32, device="cpu")
    scores.index_add_(0, enc2rows.cpu(), (delta_dec.to("cpu") * grads_rows).sum(dim=-1))
    return scores

# --- Corruption Policies ---
def _extract_occ_list(features_sparse):
    feats = features_sparse.coalesce()
    Lsrc, Penc, Fid = feats.indices()
    occ_keys = []
    last_layer = int(Lsrc.max().item()) if Lsrc.numel() > 0 else -1
    for L in range(last_layer + 1):
        mask = (Lsrc == L)
        for pos, fid in zip(Penc[mask].tolist(), Fid[mask].tolist()):
            occ_keys.append((L, pos, fid))
    return occ_keys

def _union_delta_alignment(comp_clean, comp_corr):
    dec_clean = comp_clean["decoder_vecs"]
    locs_c = comp_clean["decoder_locations"]
    enc2rows_c = comp_clean["encoder_to_decoder_map"].long()
    occ_clean = _extract_occ_list(comp_clean["activation_matrix"])
    clean_rows = {}
    for i in range(dec_clean.shape[0]):
        enc_idx = int(enc2rows_c[i].item())
        key = (occ_clean[enc_idx][0], occ_clean[enc_idx][1], occ_clean[enc_idx][2],
               int(locs_c[0][i].item()), int(locs_c[1][i].item()))
        clean_rows[key] = dec_clean[i].to(torch.float32).cpu()
    
    corr_rows = {}
    if comp_corr is not None:
        dec_corr = comp_corr["decoder_vecs"]
        locs_o = comp_corr["decoder_locations"]
        enc2rows_o = comp_corr["encoder_to_decoder_map"].long()
        occ_corr = _extract_occ_list(comp_corr["activation_matrix"])
        for i in range(dec_corr.shape[0]):
            enc_idx = int(enc2rows_o[i].item())
            key = (occ_corr[enc_idx][0], occ_corr[enc_idx][1], occ_corr[enc_idx][2],
                   int(locs_o[0][i].item()), int(locs_o[1][i].item()))
            corr_rows[key] = dec_corr[i].to(torch.float32).cpu()

    all_keys = sorted(list(set(clean_rows.keys()) | set(corr_rows.keys())))
    if not all_keys:
        z = torch.zeros(0, dtype=torch.long)
        return torch.zeros((0, 0)), z, z, 0, z, z, z, z
    enc_keys = sorted(list(set([(k[0], k[1], k[2]) for k in all_keys])))
    enc_key_to_idx = {ek: i for i, ek in enumerate(enc_keys)}
    
    delta_list, layer_list, pos_list, enc2rows_list = [], [], [], []
    d = next(iter(clean_rows.values())).numel() if clean_rows else next(iter(corr_rows.values())).numel()
    zero_vec = torch.zeros(d, dtype=torch.float32)
    for key in all_keys:
        v_c = clean_rows.get(key, zero_vec)
        v_o = corr_rows.get(key, zero_vec)
        delta_list.append((v_c - v_o).unsqueeze(0))
        layer_list.append(key[3]); pos_list.append(key[4])
        enc2rows_list.append(enc_key_to_idx[(key[0], key[1], key[2])])

    return (torch.cat(delta_list, dim=0), torch.tensor(layer_list), torch.tensor(pos_list),
            len(enc_keys), torch.tensor(enc2rows_list),
            torch.tensor([k[0] for k in enc_keys]), torch.tensor([k[1] for k in enc_keys]), torch.tensor([k[2] for k in enc_keys]))

class CorruptionPolicy:
    def build_delta_dec(self, trans, comp_clean, comp_corr=None): raise NotImplementedError

class ZeroFeaturesCorruption(CorruptionPolicy):
    def build_delta_dec(self, trans, comp_clean, comp_corr=None): return _union_delta_alignment(comp_clean, None)

class CorruptedPromptCorruption(CorruptionPolicy):
    def build_delta_dec(self, trans, comp_clean, comp_corr=None):
        assert comp_corr is not None
        return _union_delta_alignment(comp_clean, comp_corr)