#!/usr/bin/env python3
"""Distributed RelP (Relevance Patching) pruning over multiple K budgets.

Computes RelP scores once per prompt and then slices top-K selections for each
requested K, evaluating KL, prediction-change, faithfulness, and completeness
for every K in a single pass.
"""
import argparse, random, datetime, uuid, json
import numpy as np
import torch
import torch.distributed as dist
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from typing import Dict, List, Any
from pie.utils.distributed import init_distributed_if_needed, get_rank, get_world_size, barrier
from pie.utils.common import to_dtype, behavior_kl, cantor_pair, truncate_to_tokens
from pie.utils.recording import (
    load_samples, FeatureSelectionRecorder, RelpPerPromptRecorder,
    sha1_text, sha256_file,
)
from pie.models.patching import (
    setup_clt_patching, build_removed_feature_patches, apply_patches_and_get_logits,
)
from pie.pruning.fap import ZeroFeaturesCorruption, CorruptedPromptCorruption
from pie.pruning.relp import collect_downstream_relp_at_hook, relp_score_clt
from circuit_tracer import ReplacementModel
from circuit_tracer.utils.hf_utils import load_transcoder_from_hub

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True


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


def parse_k_list(s: str) -> List[int]:
    return sorted(set(int(x.strip()) for x in s.split(",") if x.strip()))


def _merge_stats(dst: Dict[int, Any], src: Dict[int, Any]) -> Dict[int, Any]:
    for fid, st in src.items():
        if fid not in dst:
            dst[fid] = st
            continue
        d = dst[fid]
        d["seen_count"]    += st.get("seen_count", 0)
        d["kept_count"]    += st.get("kept_count", 0)
        d["sum_abs_score"] += st.get("sum_abs_score", 0.0)
        d["sum_score"]     += st.get("sum_score", 0.0)
        pc_d = d.get("pos_counts", {})
        for p, c in st.get("pos_counts", {}).items():
            pc_d[p] = pc_d.get(p, 0) + c
        d["pos_counts"] = pc_d
    return dst


def main():
    ap = argparse.ArgumentParser(description="RelP: Relevance Patching for CLT-Native Pruning (multi-K)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--transcoder_set", required=True)
    ap.add_argument("--data_file", required=True)
    ap.add_argument("--output_dir", default="./relp_results")
    ap.add_argument("--keep_topk", type=str, default="100",
                    help="Comma-separated K budgets, e.g. 50,100,200,400,800")
    ap.add_argument("--feat_budget_type", default="global", choices=["global", "per_layer"])
    ap.add_argument("--max_tokens", type=int, default=128)
    ap.add_argument("--num_samples", type=int, default=None)
    ap.add_argument("--dtype", default="bf16",
                    choices=["float32", "fp32", "bfloat16", "bf16", "float16", "fp16"])
    ap.add_argument("--kl_last_token", action="store_true")
    ap.add_argument("--corruption", default="corrupted_prompt",
                    choices=["zero_features", "corrupted_prompt"])
    ap.add_argument("--metric", default="logsumexp",
                    choices=["logsumexp", "last_logit_diff_json"])
    ap.add_argument("--patch_scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run_id", default=None)
    ap.add_argument("--record_per_prompt", action="store_true", default=True)
    ap.add_argument("--merge_per_prompt", action="store_true", default=True)
    args = ap.parse_args()

    k_list = parse_k_list(args.keep_topk)
    assert k_list, f"--keep_topk must specify at least one positive integer, got: {args.keep_topk}"
    k_max = max(k_list)

    init_distributed_if_needed()
    rank, world_size = get_rank(), get_world_size()

    base_seed = args.seed
    random.seed(base_seed + rank); np.random.seed(base_seed + rank); torch.manual_seed(base_seed + rank)

    out_dir = Path(args.output_dir)
    if rank == 0: out_dir.mkdir(parents=True, exist_ok=True)
    barrier()

    args.run_id = args.run_id or f"{datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}__{uuid.uuid4().hex[:8]}"

    recorders: Dict[int, FeatureSelectionRecorder] = {k: FeatureSelectionRecorder() for k in k_list}
    pp_rec = RelpPerPromptRecorder(out_dir / f"per_prompt_rank{rank}.jsonl") if args.record_per_prompt else None

    # Load Models
    dtype = to_dtype(args.dtype)
    trans, _ = load_transcoder_from_hub(args.transcoder_set, dtype=dtype, lazy_encoder=False, lazy_decoder=True)
    local_dev = torch.device(f"cuda:{torch.cuda.current_device()}") if torch.cuda.is_available() else torch.device("cpu")
    trans.to(local_dev)
    setup_clt_patching(trans)

    model = ReplacementModel.from_pretrained_and_transcoders(args.model, trans, dtype=dtype)
    model.to(local_dev)
    if hasattr(model, "cfg"): model.cfg.device = local_dev

    # RelP sanity probe
    if rank == 0:
        t_probe = model.ensure_tokenized("hello world")[:16].to(torch.long).to(local_dev)
        rho_probe = collect_downstream_relp_at_hook(
            model, t_probe, lambda l: l.logsumexp(dim=-1).sum(), trans.feature_output_hook,
        )
        assert any(v.abs().sum().item() > 0 for v in rho_probe.values()), \
            "RelP sanity probe failed: no non-zero propagation coefficients."
        print(f"[RelP] Sanity probe passed. rho captured at {len(rho_probe)} layers.")
        print(f"[RelP] K budgets: {k_list}")

    # Load Data
    all_samples = load_samples(args.data_file, args.num_samples)
    N = len(all_samples)
    per = (N + world_size - 1) // world_size
    start = rank * per
    end = min(N, (rank + 1) * per)
    samples = all_samples[start:end]

    corruption = ZeroFeaturesCorruption() if args.corruption == "zero_features" else CorruptedPromptCorruption()
    feature_out = trans.feature_output_hook
    pbar = tqdm(total=len(samples), desc="RelP multi-K") if rank == 0 else None

    # Per-K accumulators
    kl_lists: Dict[int, List[float]] = {k: [] for k in k_list}
    chg_lists: Dict[int, List[int]] = {k: [] for k in k_list}
    faith_lists: Dict[int, List[float]] = {k: [] for k in k_list}
    comp_lists: Dict[int, List[float]] = {k: [] for k in k_list}

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

        metric_fn = metric_factory(model, args.metric, sample)

        # 3. RelP propagation coefficients (once per prompt)
        rho = collect_downstream_relp_at_hook(model, tok, metric_fn, feature_out)

        # 4. Delta + scoring (once per prompt)
        (delta, l_ids, p_ids, n_occ, enc2rows, osl, osp, osf) = corruption.build_delta_dec(trans, comp_clean, comp_corr)
        if delta.numel() == 0 or n_occ == 0: continue
        scores = relp_score_clt(delta, l_ids, p_ids, enc2rows, n_occ, rho, trans.n_layers)

        # 5. Clean and null baselines
        with torch.no_grad():
            logits_clean = model(tok)
        m_clean = float(metric_fn(logits_clean).item())

        null_keep = torch.zeros(n_occ, dtype=torch.bool)
        patches_null = build_removed_feature_patches(null_keep, l_ids, p_ids, enc2rows, delta, trans.n_layers)
        logits_null = apply_patches_and_get_logits(tok, patches_null, feature_out, args.patch_scale, model)
        m_null = float(metric_fn(logits_null).item())

        # 6. Multi-K selection and per-K evaluation
        prompt_kls: Dict[int, float] = {}
        prompt_chgs: Dict[int, Any] = {}
        prompt_faiths: Dict[int, float] = {}
        prompt_comps: Dict[int, float] = {}
        kept_fids_max_k: List[int] = []

        for K in k_list:
            if args.feat_budget_type == "global":
                keep_mask = _select_features_topk_global(scores, K)
            else:
                keep_mask = _select_features_topk_per_layer(scores, osl, K)

            recorders[K].update(osl, osp, osf, scores, keep_mask, keep_mask)

            # Patch keeps the kept features (remove the others)
            patches = build_removed_feature_patches(keep_mask, l_ids, p_ids, enc2rows, delta, trans.n_layers)
            logits_patched = apply_patches_and_get_logits(tok, patches, feature_out, args.patch_scale, model)
            m_circuit = float(metric_fn(logits_patched).item())

            # Complement: keep the rest, remove the circuit
            comp_mask = ~keep_mask.to(torch.bool)
            patches_comp = build_removed_feature_patches(comp_mask, l_ids, p_ids, enc2rows, delta, trans.n_layers)
            logits_comp = apply_patches_and_get_logits(tok, patches_comp, feature_out, args.patch_scale, model)
            m_complement = float(metric_fn(logits_comp).item())

            denom = m_clean - m_null
            if abs(denom) < 1e-10:
                faithfulness = float('nan'); completeness = float('nan')
            else:
                faithfulness = (m_circuit - m_null) / denom
                completeness = (m_complement - m_null) / denom

            kl = float(behavior_kl(logits_clean, logits_patched, args.kl_last_token).item())
            chg_val = None
            if logits_clean.ndim == 3 and logits_patched.ndim == 3 and logits_clean.shape[1] > 0:
                chg = (logits_clean[:, -1].argmax(-1).item()
                       != logits_patched[:, -1].argmax(-1).item())
                chg_val = int(chg)
                chg_lists[K].append(chg_val)

            prompt_kls[K] = kl; prompt_chgs[K] = chg_val
            prompt_faiths[K] = faithfulness; prompt_comps[K] = completeness
            kl_lists[K].append(kl); faith_lists[K].append(faithfulness); comp_lists[K].append(completeness)

            if K == k_max:
                L_cpu = osl.detach().cpu(); F_cpu = osf.detach().cpu()
                kf = keep_mask.detach().cpu().to(torch.bool)
                if kf.any():
                    kept_fids_max_k = [
                        cantor_pair(int(L_cpu[i]), int(F_cpu[i]))
                        for i in torch.nonzero(kf, as_tuple=False).squeeze(-1).tolist()
                    ]

        if pp_rec:
            pp_rec.write(
                global_idx=start + idx, rank=rank,
                text_sha1=sha1_text(sample["text_clean"])[:16],
                n_enc_occ=int(n_occ),
                kept_fids_max_k=kept_fids_max_k,
                kl_by_k=prompt_kls, chg_by_k=prompt_chgs,
                faithfulness_by_k=prompt_faiths, completeness_by_k=prompt_comps,
                extra={"run_id": args.run_id, "k_list": k_list, "method": "relp",
                       "feat_budget_type": args.feat_budget_type,
                       "corruption": args.corruption, "metric": args.metric},
            )

    if pbar: pbar.close()

    # Aggregate selection stats across ranks
    if dist.is_initialized():
        obj_local = {k: rec.final for k, rec in recorders.items()}
        obj_list = [None for _ in range(world_size)]
        dist.all_gather_object(obj_list, obj_local)
        if rank == 0:
            merged_stats: Dict[int, Dict[int, Any]] = {k: {} for k in k_list}
            for obj in obj_list:
                for k in k_list:
                    merged_stats[k] = _merge_stats(merged_stats[k], obj.get(k, {}))
    else:
        merged_stats = {k: rec.final for k, rec in recorders.items()}

    if rank == 0:
        for k in k_list:
            txt_path, json_path = FeatureSelectionRecorder.write_outputs(
                out_dir, f"selected_features_k{k}", merged_stats[k]
            )
            print(f"[REC] wrote: {txt_path}")

    # Aggregate metrics across ranks
    if dist.is_initialized():
        metrics_local = {"kl": kl_lists, "chg": chg_lists, "faith": faith_lists, "comp": comp_lists}
        metric_list = [None for _ in range(world_size)]
        dist.all_gather_object(metric_list, metrics_local)
        if rank == 0:
            agg = {key: {k: [] for k in k_list} for key in metrics_local}
            for obj in metric_list:
                for key in obj:
                    for k in k_list:
                        agg[key][k].extend(obj[key].get(k, []))
            kl_all, chg_all, faith_all, comp_all = agg["kl"], agg["chg"], agg["faith"], agg["comp"]
    else:
        kl_all, chg_all = kl_lists, chg_lists
        faith_all, comp_all = faith_lists, comp_lists

    if rank == 0:
        summary: Dict[str, Any] = {
            "method": "relp",
            "lrp_rules": ["LN-rule", "AH-rule", "Identity-rule", "Half-rule"],
            "k_list": k_list,
            "by_k": {},
        }
        for k in k_list:
            arr_kl    = np.array(kl_all[k],    dtype=np.float64) if kl_all[k]    else np.array([])
            arr_chg   = np.array(chg_all[k],   dtype=np.float64) if chg_all[k]   else np.array([])
            arr_faith = np.array(faith_all[k], dtype=np.float64) if faith_all[k] else np.array([])
            arr_comp  = np.array(comp_all[k],  dtype=np.float64) if comp_all[k]  else np.array([])
            summary["by_k"][str(k)] = {
                "mean_kl":    float(np.mean(arr_kl))     if arr_kl.size else 0.0,
                "std_kl":     float(np.std(arr_kl))      if arr_kl.size else 0.0,
                "median_kl":  float(np.median(arr_kl))   if arr_kl.size else 0.0,
                "min_kl":     float(np.min(arr_kl))      if arr_kl.size else 0.0,
                "max_kl":     float(np.max(arr_kl))      if arr_kl.size else 0.0,
                "prediction_change_rate": float(np.mean(arr_chg)) if arr_chg.size else 0.0,
                "mean_faithfulness": float(np.nanmean(arr_faith)) if arr_faith.size else 0.0,
                "mean_completeness": float(np.nanmean(arr_comp))  if arr_comp.size  else 0.0,
                "num_samples": int(arr_kl.size),
            }

        with open(out_dir / "final_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print("\n=== SUMMARY (RelP multi-K) ===")
        print(json.dumps(summary, indent=2))
    barrier()

    if pp_rec: pp_rec.close()

    # Merge per-rank JSONLs
    if rank == 0 and args.record_per_prompt and args.merge_per_prompt:
        merged_path = out_dir / "per_prompt_merged.jsonl"
        with open(merged_path, "w", encoding="utf-8") as w:
            for r in range(world_size):
                p = out_dir / f"per_prompt_rank{r}.jsonl"
                if p.exists():
                    with open(p, "r", encoding="utf-8") as f:
                        for line in f: w.write(line)
        print(f"[REC] wrote merged per-prompt file: {merged_path}")
    barrier()

    # Manifest
    if rank == 0:
        manifest = {
            "run_id": args.run_id, "method": "relp",
            "created_utc": datetime.datetime.utcnow().isoformat() + "Z",
            "world_size": world_size, "args": vars(args),
            "k_list": k_list, "files": {},
        }
        for k in k_list:
            for ext in ["txt", "json"]:
                p = out_dir / f"selected_features_k{k}.{ext}"
                if p.exists():
                    manifest["files"][p.name] = {"sha256": sha256_file(p), "bytes": p.stat().st_size}
        for p in [out_dir / "final_summary.json"]:
            if p.exists():
                manifest["files"][p.name] = {"sha256": sha256_file(p), "bytes": p.stat().st_size}
        if args.record_per_prompt:
            for r in range(world_size):
                p = out_dir / f"per_prompt_rank{r}.jsonl"
                if p.exists():
                    manifest["files"][p.name] = {"sha256": sha256_file(p), "bytes": p.stat().st_size}
            merged = out_dir / "per_prompt_merged.jsonl"
            if merged.exists():
                manifest["files"][merged.name] = {"sha256": sha256_file(merged), "bytes": merged.stat().st_size}
        with open(out_dir / "run_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print(f"[REC] wrote manifest: {out_dir / 'run_manifest.json'}")

    if dist.is_initialized(): dist.destroy_process_group()


if __name__ == "__main__":
    main()
