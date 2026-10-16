import torch
import types

def setup_clt_patching(trans):
    """Applies the custom _get_decoder_vectors logic to the transcoder."""
    _orig_get_dec = trans._get_decoder_vectors
    _orig_get_enc = trans._get_encoder_weights

    def _get_decoder_vectors_patched(self, layer_id, feat_ids=None):
        out = _orig_get_dec(layer_id, feat_ids)
        return out.to(self.device, dtype=self.dtype, non_blocking=True)

    def _get_encoder_weights_patched(self, layer_id=None):
        out = _orig_get_enc(layer_id)
        if isinstance(out, torch.Tensor):
            return out.to(self.device, dtype=self.dtype, non_blocking=True)
        try:
            return type(out)(t.to(self.device, dtype=self.dtype, non_blocking=True) for t in out)
        except Exception:
            return out

    def _select_decoder_vectors_patched(self, features):
        if not isinstance(features, torch.sparse.Tensor):
            features = features.to_sparse()

        layer_idx, pos_idx, feat_idx = features.indices()
        activations = features.values()
        n_layers = features.shape[0]
        device = features.device

        pos_ids, layer_ids, feat_ids, decoder_vectors, encoder_mapping = [], [], [], [], []
        st = 0

        for layer_id in range(n_layers):
            current_layer = (layer_idx == layer_id)
            if not current_layer.any(): continue
            
            # Filter for current layer
            current_layer_features = feat_idx[current_layer]
            
            # De-duplicate features for efficient decoder retrieval
            unique_feats, inv = current_layer_features.unique(return_inverse=True)
            unique_decoders = self._get_decoder_vectors(layer_id, unique_feats.cpu()).to(device, dtype=self.dtype)
            
            inv = inv.to(device)
            act = activations[current_layer].to(device)
            
            # Scale decoders by activation values
            scaled_decoders = unique_decoders.index_select(0, inv) * act[:, None, None]
            decoder_vectors.append(scaled_decoders.reshape(-1, self.d_model))
            
            # Map to output targets
            n_output_layers = self.n_layers - layer_id
            pos_ids.append(pos_idx[current_layer].repeat_interleave(n_output_layers))
            feat_ids.append(current_layer_features.repeat_interleave(n_output_layers))
            layer_ids.append(torch.arange(layer_id, self.n_layers, device=device).repeat(len(current_layer_features)))
            
            source_ids = torch.arange(len(current_layer_features), device=device) + st
            st += len(current_layer_features)
            encoder_mapping.append(torch.repeat_interleave(source_ids, n_output_layers))

        pos_ids = torch.cat(pos_ids, dim=0) if pos_ids else torch.empty(0, dtype=torch.long, device=device)
        layer_ids = torch.cat(layer_ids, dim=0) if layer_ids else torch.empty(0, dtype=torch.long, device=device)
        feat_ids = torch.cat(feat_ids, dim=0) if feat_ids else torch.empty(0, dtype=torch.long, device=device)
        decoder_vectors = torch.cat(decoder_vectors, dim=0) if decoder_vectors else torch.empty(0, self.d_model, dtype=self.dtype, device=device)
        encoder_mapping = torch.cat(encoder_mapping, dim=0) if encoder_mapping else torch.empty(0, dtype=torch.long, device=device)
        return pos_ids, layer_ids, feat_ids, decoder_vectors, encoder_mapping

    trans._get_decoder_vectors = types.MethodType(_get_decoder_vectors_patched, trans)
    trans._get_encoder_weights = types.MethodType(_get_encoder_weights_patched, trans)
    trans.select_decoder_vectors = types.MethodType(_select_decoder_vectors_patched, trans)

def build_feature_patch_on_device(j_enc, layer_ids, pos_ids, enc2rows, delta_dec, n_layers, T_cap, device):
    P = [torch.zeros((0, 0), device=device, dtype=torch.float32) for _ in range(n_layers)]
    if delta_dec.numel() == 0: return P
    mask = (enc2rows == int(j_enc))
    if not mask.any(): return P
    idxs = torch.nonzero(mask, as_tuple=False).squeeze(-1)
    for t in range(n_layers):
        tmask = idxs[layer_ids[idxs] == t]
        if tmask.numel() == 0: continue
        max_p = int(min(T_cap - 1, int(pos_ids[tmask].max().item())))
        if max_p < 0: continue
        d = int(delta_dec.shape[1])
        Pt = torch.zeros((max_p + 1, d), dtype=torch.float32, device=device)
        for r in tmask.tolist():
            p = int(pos_ids[r].item())
            if p <= max_p:
                Pt[p].add_(delta_dec[r].to(torch.float32).to(device))
        P[t] = Pt
    return P

def build_removed_feature_patches(keep_mask_feat, layer_ids, pos_ids, enc2rows, delta_dec, n_layers):
    if delta_dec.numel() == 0:
        return [torch.zeros(0,0) for _ in range(n_layers)]
    T_per_layer = [0] * n_layers
    for t in range(n_layers):
        mask_t = (layer_ids == t)
        T_per_layer[t] = int(pos_ids[mask_t].max().item()) + 1 if mask_t.any() else 0
    P = [torch.zeros((T_per_layer[t], delta_dec.shape[1]), dtype=torch.float32, device="cpu") for t in range(n_layers)]

    enc_occ = enc2rows.to("cpu")
    tgt = layer_ids.to("cpu")
    pos = pos_ids.to("cpu")
    delta = delta_dec.to("cpu")
    removed = (~keep_mask_feat.to(torch.bool)).to(torch.bool)
    to_patch = removed[enc_occ]
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

def apply_patches_and_get_logits(tokens_1d_dev, patches, feature_out, scale, model):
    """Apply per-layer patches (one tensor per layer) and return logits.

    Unlike apply_patch (which aggregates patches from multiple feature lists and
    returns a scalar metric), this helper consumes a single per-layer patch list
    of length n_layers and returns logits for downstream KL / faithfulness use.
    """
    L = model.cfg.n_layers

    def make_hook(Pl):
        if Pl is None or Pl.numel() == 0: return None
        Pl = (scale * Pl)
        def _hook(acts, hook):
            if Pl.numel() == 0: return acts
            Tcap = min(acts.shape[1], Pl.shape[0])
            if Tcap > 0: acts[:, :Tcap, :] = acts[:, :Tcap, :] - Pl[:Tcap, :].to(acts.dtype)
            return acts
        return _hook

    for layer in range(L):
        Pl = patches[layer] if layer < len(patches) else None
        if Pl is not None and Pl.numel() > 0:
            hk = make_hook(Pl.to(tokens_1d_dev.device))
            if hk is not None:
                model.add_hook(f"blocks.{layer}.{feature_out}", hk, "fwd")

    with torch.no_grad(), torch.autocast(
        device_type=tokens_1d_dev.device.type,
        dtype=torch.bfloat16 if tokens_1d_dev.dtype == torch.bfloat16 else torch.float16,
        enabled=True,
    ):
        logits = model(tokens_1d_dev)
    model.reset_hooks()
    return logits

def apply_patch(tokens_1d_dev, P_list, feature_out, metric_fn, scale, model):
    L = model.cfg.n_layers
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
            Tcap = min(acts.shape[1], Pl.shape[0])
            if Tcap > 0: acts[:, :Tcap, :] = acts[:, :Tcap, :] - Pl[:Tcap, :].to(acts.dtype)
            return acts
        return _hook

    for layer, S in enumerate(summed):
        hk = make_hook(S)
        if hk is not None:
            model.add_hook(f"blocks.{layer}.{feature_out}", hk, "fwd")

    with torch.no_grad(), torch.autocast(device_type=tokens_1d_dev.device.type, dtype=torch.bfloat16):
        val = metric_fn(model(tokens_1d_dev)).item()
    model.reset_hooks()
    return float(val)