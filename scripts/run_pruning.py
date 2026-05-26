#!/usr/bin/env python3
import argparse, random, datetime, uuid, json
import numpy as np
import torch
import torch.distributed as dist
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from pie.utils.distributed import init_distributed_if_needed, get_rank, get_world_size, barrier
from pie.utils.common import to_dtype, behavior_kl, cantor_pair, truncate_to_tokens
from pie.utils.recording import load_samples, FeatureSelectionRecorder, PerPromptRecorder, sha1_text, sha256_file
from pie.models.patching import setup_clt_patching, build_removed_feature_patches
from pie.pruning.fap import collect_downstream_grads_at_hook, fap_score_clt, ZeroFeaturesCorruption, CorruptedPromptCorruption
from pie.pruning.synergy import apply_synergy_reranking
from circuit_tracer import ReplacementModel
from circuit_tracer.utils.hf_utils import load_transcoder_from_hub

torch.backends.cuda.matmul.allow_tf32 = True

def _select_features_topk_global(scores, K):
    keep = torch.zeros(scores.numel(), dtype=torch.bool)
    if scores.numel() > 0 and K > 0:
        order = torch.topk(scores.abs(), min(K, scores.numel()), largest=True, sorted=False).indices
        keep[order] = True
    return keep

def _select_features_topk_per_layer(scores, occ_src_layers, K):
    keep = torch.zeros(scores.numel(), dtype=torch.bool)
    layers = occ_src_layers.cpu().tolist()
    byL = defaultdict(list)
    for i, L in enumerate(layers): byL[int(L)].append(i)
    for idxs in byL.values():
        s = scores[idxs].abs()
        if K > 0:
            take = torch.topk(s, min(K, s.numel()), largest=True, sorted=False).indices
            keep.index_fill_(0, torch.tensor([idxs[j] for j in take]), True)
    return keep

def metric_factory(model, choice, sample):
    if choice == "logsumexp" or not (sample.get("pos_target") and sample.get("neg_target")):
        return lambda logits: logits.logsumexp(dim=-1).sum()
    pos_ids = model.tokenizer(sample["pos_target"], return_tensors="pt").input_ids.squeeze(0).to(model.cfg.device)
    neg_ids = model.tokenizer(sample["neg_target"], return_tensors="pt").input_ids.squeeze(0).to(model.cfg.device)
    pos_tok, neg_tok = int(pos_ids[-1].item()), int(neg_ids[-1].item())
    return lambda logits: (logits[..., -1, pos_tok] - logits[..., -1, neg_tok]).mean()

def main():
    ap = argparse.ArgumentParser(description="Distributed FAP with synergy-aware boundary rerank")
    ap.add_argument("--model", required=True)
    ap.add_argument("--transcoder_set", required=True)
    ap.add_argument("--data_file", required=True)
    ap.add_argument("--output_dir", default="./fap_results")
    ap.add_argument("--keep_topk", type=int, default=100)
    ap.add_argument("--feat_budget_type", default="global", choices=["global", "per_layer"])
    ap.add_argument("--max_tokens", type=int, default=128)
    ap.add_argument("--num_samples", type=int, default=None)
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--kl_last_token", action="store_true")
    ap.add_argument("--corruption", default="corrupted_prompt", choices=["zero_features", "corrupted_prompt"])
    ap.add_argument("--metric", default="logsumexp")
    ap.add_argument("--patch_scale", type=float, default=1.0)
    # Synergy args
    ap.add_argument("--synergy", default="boundary", choices=["none", "boundary"])
    ap.add_argument("--lambda_syn", type=float, default=5.0)
    ap.add_argument("--boundary_percent", type=float, default=50.0)
    ap.add_argument("--boundary_cap", type=int, default=32)
    ap.add_argument("--partners_per_refresh", type=int, default=32)
    ap.add_argument("--syn_partners_per_candidate", type=int, default=8)
    ap.add_argument("--syn_seed", type=int, default=0)
    # Rec args
    ap.add_argument("--run_id", default=None)
    ap.add_argument("--record_per_prompt", action="store_true", default=True)
    ap.add_argument("--merge_per_prompt", action="store_true", default=True)
    args = ap.parse_args()

    init_distributed_if_needed()
    rank, world_size = get_rank(), get_world_size()
    
    base_seed = args.syn_seed if args.syn_seed is not None else 0
    random.seed(base_seed + rank); np.random.seed(base_seed + rank); torch.manual_seed(base_seed + rank)

    out_dir = Path(args.output_dir)
    if rank == 0: out_dir.mkdir(parents=True, exist_ok=True)
    barrier()

    args.run_id = args.run_id or f"{datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}__{uuid.uuid4().hex[:8]}"
    recorder = FeatureSelectionRecorder()
    pp_rec = PerPromptRecorder(out_dir / f"per_prompt_rank{rank}.jsonl") if args.record_per_prompt else None

    # Load Models
    dtype = to_dtype(args.dtype)
    trans, _ = load_transcoder_from_hub(args.transcoder_set, dtype=dtype, lazy_encoder=False, lazy_decoder=True)
    local_dev = torch.device(f"cuda:{torch.cuda.current_device()}") if torch.cuda.is_available() else torch.device("cpu")
    trans.to(local_dev)
    setup_clt_patching(trans) # Apply monkey patches

    model = ReplacementModel.from_pretrained_and_transcoders(args.model, trans, dtype=dtype)
    model.to(local_dev)
    if hasattr(model, "cfg"): model.cfg.device = local_dev

    # Load Data
    all_samples = load_samples(args.data_file, args.num_samples)
    N = len(all_samples)
    
    per = (N + world_size - 1) // world_size
    start = rank * per
    end = min(N, (rank + 1) * per)
    samples = all_samples[start:end]
    
    corruption = ZeroFeaturesCorruption() if args.corruption == "zero_features" else CorruptedPromptCorruption()
    pbar = tqdm(total=len(samples), desc="FAP+Synergy") if rank == 0 else None
    kl_local, chg_local = [], []

    for idx, sample in enumerate(samples):
        if pbar: pbar.update(1)
        
        tok = truncate_to_tokens(model, sample["text_clean"], args.max_tokens).to(model.cfg.device)
        if tok.numel() < 8: continue

        # 1. Clean run for attribution components
        with torch.no_grad():
            mlp_in_cache, in_hooks, _ = model.get_caching_hooks(lambda n: trans.feature_input_hook in n)
            _ = model.run_with_hooks(tok, fwd_hooks=in_hooks)
            comp_clean = trans.compute_attribution_components(torch.cat(list(mlp_in_cache.values()), dim=0))

        # 2. Corrupted run components
        comp_corr = None
        if isinstance(corruption, CorruptedPromptCorruption):
            t_corr = truncate_to_tokens(model, sample["text_corr"], len(tok)).to(model.cfg.device)
            with torch.no_grad():
                mlp_in_cache_c, in_hooks_c, _ = model.get_caching_hooks(lambda n: trans.feature_input_hook in n)
                _ = model.run_with_hooks(t_corr, fwd_hooks=in_hooks_c)
                comp_corr = trans.compute_attribution_components(torch.cat(list(mlp_in_cache_c.values()), dim=0))

        # 3. FAP Scoring
        metric_fn = metric_factory(model, args.metric, sample)
        grads = collect_downstream_grads_at_hook(model, tok, metric_fn, trans.feature_output_hook)
        (delta, l_ids, p_ids, n_occ, enc2rows, osl, osp, osf) = corruption.build_delta_dec(trans, comp_clean, comp_corr)
        
        if delta.numel() == 0: continue
        scores = fap_score_clt(delta, l_ids, p_ids, enc2rows, n_occ, grads, trans.n_layers)

        # 4. Selection
        if args.feat_budget_type == "global":
            keep_base = _select_features_topk_global(scores, args.keep_topk)
        else:
            keep_base = _select_features_topk_per_layer(scores, osl, args.keep_topk)

        # 5. Synergy Reranking
        keep_final = keep_base.clone()
        logits_clean, m_clean = None, None
        
        # Pre-compute clean logits if needed for synergy or recording
        if (args.synergy == "boundary" and args.feat_budget_type == "global") or args.record_per_prompt:
            with torch.no_grad():
                logits_clean = model(tok)
                if args.synergy == "boundary": m_clean = float(metric_fn(logits_clean).item())

        if args.synergy == "boundary" and args.feat_budget_type == "global":
            keep_final = apply_synergy_reranking(
                keep_base, scores, args, model, tok, metric_fn, trans.feature_output_hook,
                l_ids, p_ids, enc2rows, delta, trans, m_clean, base_seed + 1337 + rank
            )

        recorder.update(osl, osp, osf, scores, keep_base, keep_final)

        # 6. Per-Prompt Recording
        if pp_rec:
            # We must re-calculate patch effect for recording
            patches = build_removed_feature_patches(keep_final, l_ids, p_ids, enc2rows, delta, trans.n_layers)
            patch_hooks = []
            for Lidx in range(trans.n_layers):
                if patches[Lidx].numel() > 0:
                    P = args.patch_scale * patches[Lidx].to(model.cfg.device)
                    def _hk(a, hook, P=P): 
                        if P.numel(): a[:, :min(a.shape[1], P.shape[0]), :] -= P[:min(a.shape[1], P.shape[0]), :].to(a.dtype)
                        return a
                    patch_hooks.append((f"blocks.{Lidx}.{trans.feature_output_hook}", _hk))
            
            with torch.no_grad():
                logits_patched = model.run_with_hooks(tok, fwd_hooks=patch_hooks)
            
            kl = float(behavior_kl(logits_clean, logits_patched, args.kl_last_token).item())
            chg = (logits_clean[0,-1].argmax().item() != logits_patched[0,-1].argmax().item())
            kl_local.append(kl); chg_local.append(int(chg))
            
            L_cpu, F_cpu = osl.detach().cpu(), osf.detach().cpu()
            kb_fids = [cantor_pair(int(L_cpu[i]), int(F_cpu[i])) for i in torch.nonzero(keep_base, as_tuple=False).squeeze(-1).tolist()]
            kf_fids = [cantor_pair(int(L_cpu[i]), int(F_cpu[i])) for i in torch.nonzero(keep_final, as_tuple=False).squeeze(-1).tolist()]
            
            pp_rec.write(start + idx, rank, sha1_text(sample["text_clean"])[:16], n_occ, kb_fids, kf_fids, kl, int(chg), {"run_id": args.run_id})

    if pbar: pbar.close()
    
    # Aggregation & Cleanup
    if dist.is_initialized():
        # (Simplified aggregation logic for brevity - similar to original)
        pass 
    
    if rank == 0:
        FeatureSelectionRecorder.write_outputs(out_dir, "selected_features_final", recorder.final)
    
    if pp_rec: pp_rec.close()
    if dist.is_initialized(): dist.destroy_process_group()

if __name__ == "__main__":
    main()