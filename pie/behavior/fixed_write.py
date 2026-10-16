"""Pinned pre-answer fixed-write primitives used by the revision experiments."""
from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional
from collections import defaultdict
import types
import torch


class CorruptionPolicy:
    name: str = "base"
    def build_delta_dec(
        self, trans: CrossLayerTranscoder,
        comp_clean: Dict[str, torch.Tensor],
        comp_corr: Optional[Dict[str, torch.Tensor]] = None,
    ):
        raise NotImplementedError


def _extract_occ_list(features_sparse: torch.Tensor):
    feats = features_sparse.coalesce()
    Lsrc, Penc, Fid = feats.indices()
    occ_keys = []
    last_layer = int(Lsrc.max().item()) if Lsrc.numel() > 0 else -1
    for L in range(last_layer + 1):
        mask = (Lsrc == L)
        for pos, fid in zip(Penc[mask].tolist(), Fid[mask].tolist()):
            occ_keys.append((L, pos, fid))
    return occ_keys


def _union_delta_alignment(comp_clean: Dict[str, torch.Tensor], comp_corr: Optional[Dict[str, torch.Tensor]]):
    # Returns Δdecoder rows aligned on (enc_layer, enc_pos, feat_id, tgt_layer, tgt_pos)
    dec_clean = comp_clean["decoder_vecs"]                  # [R_c, d]
    locs_c    = comp_clean["decoder_locations"]             # [2, R_c]
    layer_ids_c = locs_c[0].long()
    pos_ids_c   = locs_c[1].long()
    enc2rows_c  = comp_clean["encoder_to_decoder_map"].long()
    occ_clean   = _extract_occ_list(comp_clean["activation_matrix"])

    clean_rows = {}
    for i in range(dec_clean.shape[0]):
        enc_idx = int(enc2rows_c[i].item())
        key = (occ_clean[enc_idx][0], occ_clean[enc_idx][1], occ_clean[enc_idx][2],
               int(layer_ids_c[i].item()), int(pos_ids_c[i].item()))
        clean_rows[key] = dec_clean[i].to(torch.float32).cpu()

    corr_rows = {}
    if comp_corr is not None:
        dec_corr = comp_corr["decoder_vecs"]
        locs_o   = comp_corr["decoder_locations"]
        layer_ids_o = locs_o[0].long()
        pos_ids_o   = locs_o[1].long()
        enc2rows_o  = comp_corr["encoder_to_decoder_map"].long()
        occ_corr    = _extract_occ_list(comp_corr["activation_matrix"])
        for i in range(dec_corr.shape[0]):
            enc_idx = int(enc2rows_o[i].item())
            key = (occ_corr[enc_idx][0], occ_corr[enc_idx][1], occ_corr[enc_idx][2],
                   int(layer_ids_o[i].item()), int(pos_ids_o[i].item()))
            corr_rows[key] = dec_corr[i].to(torch.float32).cpu()

    all_keys = list(set(clean_rows.keys()) | set(corr_rows.keys()))
    all_keys.sort()
    enc_keys = sorted(list(set([(k[0], k[1], k[2]) for k in all_keys])))
    enc_key_to_idx = {ek: i for i, ek in enumerate(enc_keys)}

    if not all_keys:
        zL = torch.zeros(0, dtype=torch.long)
        return (torch.zeros((0, 0), dtype=torch.float32), zL, zL, 0, zL, zL, zL, zL)

    occ_src_layer = torch.tensor([ek[0] for ek in enc_keys], dtype=torch.long)
    occ_src_pos   = torch.tensor([ek[1] for ek in enc_keys], dtype=torch.long)
    occ_src_feat  = torch.tensor([ek[2] for ek in enc_keys], dtype=torch.long)

    d = next(iter(clean_rows.values())).numel() if clean_rows else next(iter(corr_rows.values())).numel()
    delta_list, layer_list, pos_list, enc2rows_list = [], [], [], []
    zero_vec = torch.zeros(d, dtype=torch.float32)
    for key in all_keys:
        v_clean = clean_rows.get(key, zero_vec)
        v_corr  = corr_rows.get(key, zero_vec)
        delta_list.append((v_clean - v_corr).unsqueeze(0))
        layer_list.append(key[3])
        pos_list.append(key[4])
        enc2rows_list.append(enc_key_to_idx[(key[0], key[1], key[2])])

    delta_dec = torch.cat(delta_list, dim=0)   # [R, d]
    layer_ids = torch.tensor(layer_list, dtype=torch.long)  # [R]
    pos_ids   = torch.tensor(pos_list, dtype=torch.long)    # [R]
    enc2rows  = torch.tensor(enc2rows_list, dtype=torch.long)  # [R]
    n_enc_occ = len(enc_keys)
    return delta_dec, layer_ids, pos_ids, n_enc_occ, enc2rows, occ_src_layer, occ_src_pos, occ_src_feat


class ZeroFeaturesCorruption(CorruptionPolicy):
    name = "zero_features"
    def build_delta_dec(self, trans, comp_clean, comp_corr=None):
        return _union_delta_alignment(comp_clean, None)


class CorruptedPromptCorruption(CorruptionPolicy):
    name = "corrupted_prompt"
    def build_delta_dec(self, trans, comp_clean, comp_corr=None):
        assert comp_corr is not None, "CorruptedPromptCorruption requires comp_corr"
        return _union_delta_alignment(comp_clean, comp_corr)


def collect_downstream_grads_at_hook(model: ReplacementModel, tokens_1d: torch.Tensor,
                                     metric_fn: Callable[[torch.Tensor], torch.Tensor],
                                     hook_name: str) -> Dict[int, torch.Tensor]:
    dev = model.cfg.device
    grad_cache: Dict[int, torch.Tensor] = {}
    fwd_tensors: Dict[int, torch.Tensor] = {}

    def fwd_passthrough(L: int):
        def _id(acts, hook):
            if torch.is_floating_point(acts) and not acts.requires_grad:
                acts = acts * 1.0
            acts.retain_grad()
            fwd_tensors[L] = acts
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
        g = acts.grad
        if g is None: continue
        if g.ndim == 3: g = g.squeeze(0)
        grad_cache[L] = g.to(torch.float32).detach()
    assert any(g.abs().sum().item() > 0 for g in grad_cache.values()), "No grad captured."
    return grad_cache


def fap_score_clt(delta_dec: torch.Tensor, layer_ids: torch.Tensor, pos_ids: torch.Tensor,
                  enc2rows: torch.Tensor, n_enc_occ: int,
                  grad_mlp_out: Dict[int, torch.Tensor], n_layers: int) -> torch.Tensor:
    """Return per-feature scalar score (aggregate across target layers/positions)."""
    if n_enc_occ == 0 or delta_dec.numel() == 0:
        return torch.zeros((0,), dtype=torch.float32)
    R, d = delta_dec.shape

    # Gather grads rows per (layer, pos) from cache
    grads_rows = torch.empty((R, d), dtype=torch.float32, device="cpu")
    for t in range(n_layers):
        mask_t = (layer_ids.cpu() == t)
        if not mask_t.any(): continue
        pos_t = pos_ids.cpu()[mask_t]
        gmat = grad_mlp_out.get(t)
        if gmat is None:
            grads_rows[mask_t] = 0.0
        else:
            grads_rows[mask_t] = gmat.to("cpu", dtype=torch.float32).index_select(0, pos_t)

    delta_cpu = delta_dec.to("cpu", dtype=torch.float32)
    dots = (delta_cpu * grads_rows).sum(dim=-1)  # per-row dot
    # Sum over all rows that belong to same feature occurrence
    scores = torch.zeros((n_enc_occ,), dtype=torch.float32, device="cpu")
    scores.index_add_(0, enc2rows.cpu(), dots)
    return scores


def build_feature_patch_on_device(j_enc: int, layer_ids: torch.Tensor, pos_ids: torch.Tensor,
                                  enc2rows: torch.Tensor, delta_dec: torch.Tensor,
                                  n_layers: int, T_cap: int, device: torch.device):
    """Return list[P_tgt] where each P_tgt is [T, d] for all rows belonging to feature j_enc."""
    P = [torch.zeros((0, 0), device=device, dtype=torch.float32) for _ in range(n_layers)]
    if delta_dec.numel() == 0: return P

    mask = (enc2rows == int(j_enc))
    if not mask.any(): return P
    idxs = torch.nonzero(mask, as_tuple=False).squeeze(-1)

    # Separate per target layer
    for t in range(n_layers):
        tmask = idxs[layer_ids[idxs] == t]
        if tmask.numel() == 0: 
            continue
        max_p = int(min(T_cap - 1, int(pos_ids[tmask].max().item())))
        if max_p < 0: 
            continue
        d = int(delta_dec.shape[1])
        Pt = torch.zeros((max_p + 1, d), dtype=torch.float32, device=device)
        for r in tmask.tolist():
            p = int(pos_ids[r].item())
            if p <= max_p:
                Pt[p].add_(delta_dec[r].to(torch.float32).to(device))
        P[t] = Pt
    return P


def build_removed_feature_patches(keep_mask_feat: torch.Tensor, layer_ids: torch.Tensor, pos_ids: torch.Tensor,
                                  enc2rows: torch.Tensor, delta_dec: torch.Tensor, n_layers: int) -> List[torch.Tensor]:
    """Build summed per-layer [T,d] patches for *removed* features."""
    if delta_dec.numel() == 0:
        return [torch.zeros(0,0) for _ in range(n_layers)]
    R, d = delta_dec.shape
    T_per_layer = [0] * n_layers
    for t in range(n_layers):
        mask_t = (layer_ids == t)
        T_per_layer[t] = int(pos_ids[mask_t].max().item()) + 1 if mask_t.any() else 0
    P = [torch.zeros((T_per_layer[t], d), dtype=torch.float32, device="cpu") for t in range(n_layers)]

    enc_occ = enc2rows.to("cpu")        # [R]
    tgt     = layer_ids.to("cpu")       # [R]
    pos     = pos_ids.to("cpu")         # [R]
    delta   = delta_dec.to("cpu")       # [R,d]

    # rows to patch are those whose feature is *removed*
    removed = (~keep_mask_feat.to(torch.bool)).to(torch.bool)   # [n_occ]
    to_patch = removed[enc_occ]  # [R]
    if to_patch.any():
        rows = torch.nonzero(to_patch, as_tuple=False).squeeze(-1)
        for idx in rows.tolist():
            t = int(tgt[idx].item()); p = int(pos[idx].item())
            v = delta[idx].to(torch.float32)
            if p >= P[t].shape[0]:
                pad = p + 1 - P[t].shape[0]
                P[t] = torch.cat([P[t], torch.zeros((pad, v.numel()), dtype=torch.float32)], dim=0)
            P[t][p].add_(v)
    return P


def apply_patch(tokens_1d_dev, P_list, feature_out, metric_fn, scale, model):
    L = model.cfg.n_layers
    # sum same-layer patches among a list of candidates
    summed = []
    for layer in range(L):
        maxT, d = 0, 0
        for P in P_list:
            if layer < len(P) and P[layer].numel():
                maxT = max(maxT, P[layer].shape[0]); d = P[layer].shape[1]
        if maxT == 0:
            summed.append(None); continue
        S = torch.zeros((maxT, d), dtype=torch.float32, device=tokens_1d_dev.device)
        for P in P_list:
            if layer < len(P) and P[layer].numel():
                Tcap = min(maxT, P[layer].shape[0])
                if Tcap > 0: S[:Tcap].add_(P[layer][:Tcap])
        summed.append(S)

    def make_hook(Pl):
        if Pl is None or Pl.numel() == 0: return None
        Pl = (scale * Pl)
        def _hook(acts, hook):
            if Pl.numel() == 0: return acts
            B, T, d = acts.shape
            Tcap = min(T, Pl.shape[0])
            if Tcap > 0: acts[:, :Tcap, :] = acts[:, :Tcap, :] - Pl[:Tcap, :].to(acts.dtype)
            return acts
        return _hook

    for layer, S in enumerate(summed):
        hk = make_hook(S)
        if hk is not None:
            model.add_hook(f"blocks.{layer}.{feature_out}", hk, "fwd")

    with torch.no_grad(), torch.autocast(
        device_type=tokens_1d_dev.device.type,
        dtype=torch.bfloat16 if tokens_1d_dev.dtype==torch.bfloat16 else torch.float16,
        enabled=True
    ):
        logits = model(tokens_1d_dev)
        val = metric_fn(logits).item()

    model.reset_hooks()
    return float(val)


def _select_features_topk_global(scores: torch.Tensor, K: int) -> torch.Tensor:
    """Global feature selection: return keep_mask [n_occ]"""
    n = scores.numel()
    keep = torch.zeros((n,), dtype=torch.bool)
    if n == 0 or K <= 0: return keep
    K_eff = min(K, n)
    order = torch.topk(scores.abs(), K_eff, largest=True, sorted=False).indices
    keep[order] = True
    return keep


def metric_factory(model: ReplacementModel, choice: str, sample: Dict[str, Any]):
    if choice == "logsumexp":
        return lambda logits: logits.logsumexp(dim=-1).sum()
    if choice == "last_logsumexp":
        return lambda logits: logits[:, -1, :].logsumexp(dim=-1).sum()
    if choice == "last_logit_diff_json":
        pos_s = sample.get("pos_target")
        neg_s = sample.get("neg_target")
        if not pos_s or not neg_s:
            return lambda logits: logits.logsumexp(dim=-1).sum()
        pos_ids = model.tokenizer(pos_s, return_tensors="pt").input_ids.squeeze(0).to(model.cfg.device)
        neg_ids = model.tokenizer(neg_s, return_tensors="pt").input_ids.squeeze(0).to(model.cfg.device)
        pos_tok = int(pos_ids[-1].item())
        neg_tok = int(neg_ids[-1].item())
        def _metric(logits):
            last = logits[:, -1, :] if logits.ndim == 3 else logits[-1, :]
            return (last[..., pos_tok] - last[..., neg_tok]).mean()
        return _metric
    return lambda logits: logits.logsumexp(dim=-1).sum()


def attach_clt_device_patches(trans) -> None:
    """Mirror the working ten-prompt LastLSE producer's CLT device adapter."""
    original_decoder = trans._get_decoder_vectors
    original_encoder = trans._get_encoder_weights

    def decoder(self, layer_id, feat_ids=None):
        out = original_decoder(layer_id, feat_ids)
        return out.to(self.device, dtype=self.dtype, non_blocking=True)

    def encoder(self, layer_id=None):
        out = original_encoder(layer_id)
        if isinstance(out, torch.Tensor):
            return out.to(self.device, dtype=self.dtype, non_blocking=True)
        try:
            return type(out)(t.to(self.device, dtype=self.dtype, non_blocking=True) for t in out)
        except Exception:
            return out

    def select_decoder_vectors(self, features):
        if not isinstance(features, torch.sparse.Tensor):
            features = features.to_sparse()
        layer_idx, pos_idx, feat_idx = features.indices()
        activations = features.values()
        device = features.device
        pos_ids, layer_ids, feat_ids, decoder_vectors, encoder_mapping = [], [], [], [], []
        offset = 0
        for layer_id in range(features.shape[0]):
            current = layer_idx == layer_id
            if not current.any():
                continue
            layer_features = feat_idx[current]
            unique_features, inv = layer_features.unique(return_inverse=True)
            unique_decoders = self._get_decoder_vectors(layer_id, unique_features.cpu()).to(device, dtype=self.dtype)
            scaled = unique_decoders.index_select(0, inv.to(device)) * activations[current].to(device)[:, None, None]
            decoder_vectors.append(scaled.reshape(-1, self.d_model))
            n_output_layers = self.n_layers - layer_id
            pos_ids.append(pos_idx[current].repeat_interleave(n_output_layers))
            feat_ids.append(layer_features.repeat_interleave(n_output_layers))
            layer_ids.append(torch.arange(layer_id, self.n_layers, device=device).repeat(len(layer_features)))
            source_ids = torch.arange(len(layer_features), device=device) + offset
            offset += len(layer_features)
            encoder_mapping.append(torch.repeat_interleave(source_ids, n_output_layers))
        return (
            torch.cat(pos_ids) if pos_ids else torch.empty(0, dtype=torch.long, device=device),
            torch.cat(layer_ids) if layer_ids else torch.empty(0, dtype=torch.long, device=device),
            torch.cat(feat_ids) if feat_ids else torch.empty(0, dtype=torch.long, device=device),
            torch.cat(decoder_vectors) if decoder_vectors else torch.empty(0, self.d_model, dtype=self.dtype, device=device),
            torch.cat(encoder_mapping) if encoder_mapping else torch.empty(0, dtype=torch.long, device=device),
        )

    trans._get_decoder_vectors = types.MethodType(decoder, trans)
    trans._get_encoder_weights = types.MethodType(encoder, trans)
    trans.select_decoder_vectors = types.MethodType(select_decoder_vectors, trans)


def patched_logits(model, trans, tokens: torch.Tensor, patches: list[torch.Tensor]) -> torch.Tensor:
    feature_out = trans.feature_output_hook
    for layer in range(trans.n_layers):
        patch = patches[layer].to(model.cfg.device)

        def make_hook(value):
            def patch_hook(acts, hook):
                if value.numel():
                    length = min(acts.shape[1], value.shape[0])
                    if length:
                        acts[:, :length, :] = acts[:, :length, :] - value[:length].to(acts.dtype)
                return acts
            return patch_hook

        model.add_hook(f"blocks.{layer}.{feature_out}", make_hook(patch), "fwd")
    try:
        with torch.no_grad():
            return model(tokens.to(model.cfg.device))
    finally:
        model.reset_hooks()
