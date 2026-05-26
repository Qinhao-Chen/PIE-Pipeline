import torch
import random
import numpy as np
from pie.models.patching import build_feature_patch_on_device, apply_patch

def apply_synergy_reranking(
    keep_base: torch.Tensor,
    scores_feat: torch.Tensor,
    args,
    model,
    tok: torch.Tensor,
    metric_fn,
    feature_out: str,
    layer_ids,
    pos_ids,
    enc2rows,
    delta_dec,
    trans,
    m_clean: float,
    seed: int
) -> torch.Tensor:
    """
    Reranks features near the pruning boundary based on pairwise interactions.
    Matches the logic from the original script exactly.
    """
    keep_mask = keep_base.clone()
    n = scores_feat.numel()
    
    # Standard check from original script
    if n == 0 or args.keep_topk <= 0:
        return keep_mask

    K = min(args.keep_topk, n)
    s_abs = scores_feat.abs()
    order = torch.argsort(s_abs, descending=True)
    T_cap = int(tok.shape[-1])

    # Boundary definitions
    core_K = max(0, K - int(round(args.boundary_percent / 100.0 * K * 0.5)))
    core_idx = order[:core_K]
    boundary_lo = core_K
    boundary_hi = min(n, K + int(round(args.boundary_percent / 100.0 * K * 0.5)))
    boundary_idx = order[boundary_lo:boundary_hi]

    if boundary_idx.numel() > args.boundary_cap:
        boundary_idx = boundary_idx[:args.boundary_cap]

    partner_pool = core_idx.tolist()
    rng = random.Random(seed)

    # Precompute partner single effects
    partners = partner_pool[:min(args.partners_per_refresh, len(partner_pool))]
    partner_P, partner_eff = [], []
    for jp in partners:
        Pp = build_feature_patch_on_device(jp, layer_ids, pos_ids, enc2rows, delta_dec, trans.n_layers, T_cap, tok.device)
        if not any(x.numel() for x in Pp):
            partner_P.append(Pp); partner_eff.append(0.0)
            continue
        mp = apply_patch(tok, [Pp], feature_out, metric_fn, args.patch_scale, model)
        partner_P.append(Pp); partner_eff.append(m_clean - mp)

    # Scaling
    base_vals = s_abs[order[:K]] if K > 0 else torch.tensor([], device=s_abs.device)
    base_scale = float(torch.median(base_vals).item()) if base_vals.numel() else 1.0
    base_scale = max(base_scale, 1e-6)
    med_partner = float(np.median(np.abs(partner_eff))) if partner_eff else 1.0
    med_partner = max(med_partner, 1e-6)

    cand_cache_P = {}
    cand_cache_eff = {}

    def get_cand_P_and_eff(jj):
        if jj in cand_cache_P: return cand_cache_P[jj], cand_cache_eff[jj]
        Pc = build_feature_patch_on_device(jj, layer_ids, pos_ids, enc2rows, delta_dec, trans.n_layers, T_cap, tok.device)
        if not any(x.numel() for x in Pc):
            cand_cache_P[jj] = Pc; cand_cache_eff[jj] = 0.0; return Pc, 0.0
        mc = apply_patch(tok, [Pc], feature_out, metric_fn, args.patch_scale, model)
        eff_c = m_clean - mc
        cand_cache_P[jj] = Pc; cand_cache_eff[jj] = eff_c; return Pc, eff_c

    def boosted_score(jj):
        base_z = float(s_abs[jj].item()) / base_scale
        Pc, eff_c = get_cand_P_and_eff(jj)
        if not any(x.numel() for x in Pc) or not partner_P: return base_z

        idxs = rng.sample(range(len(partner_P)), min(args.syn_partners_per_candidate, len(partner_P)))
        syn_vals = []
        for k_idx in idxs:
            Pp = partner_P[k_idx]
            eff_p = partner_eff[k_idx]
            if eff_p <= 0 or eff_c <= 0: continue
            mcp = apply_patch(tok, [Pc, Pp], feature_out, metric_fn, args.patch_scale, model)
            syn_vals.append((m_clean - mcp) - eff_c - eff_p)
        
        syn_stat = float(np.median(syn_vals)) if syn_vals else 0.0
        syn_z = max(0.0, syn_stat) / (med_partner + 1e-6)
        return base_z + args.lambda_syn * syn_z

    # Reconstruct mask
    keep_mask = torch.zeros_like(keep_base)
    keep_mask[core_idx] = True
    needed = K - int(keep_mask.sum().item())

    candidates = []
    for jj in boundary_idx.tolist():
        candidates.append((boosted_score(jj), jj))
    
    if needed > 0 and candidates:
        candidates.sort(key=lambda x: -x[0])
        for _, jj in candidates[:needed]:
            keep_mask[jj] = True

    # Fill remainder if under budget
    if int(keep_mask.sum().item()) < K:
        need = K - int(keep_mask.sum().item())
        base_order = order[~keep_mask[order]]
        keep_mask[base_order[:need]] = True
        
    return keep_mask