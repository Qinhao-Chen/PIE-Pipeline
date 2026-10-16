"""Portable orchestration around the revision's extracted scientific kernels.

This runner produces new measurements. It does not replay or assert numerical
identity with the mixed-platform 2,000-prompt aggregate tables.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import time
from types import SimpleNamespace

import numpy as np
import torch

from .protocol import MAX_TOKENS, PROFILES, prepare_prompt, tokenize_prefix
from .metrics import dual_kl
from .fixed_write import (
    CorruptedPromptCorruption, ZeroFeaturesCorruption,
    _select_features_topk_global, attach_clt_device_patches,
    build_removed_feature_patches, collect_downstream_grads_at_hook,
    fap_score_clt, metric_factory, patched_logits,
)


METRICS = ("kl_pre_answer_last_token", "kl_sequence_mean")


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_hash(path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def source_fingerprints() -> dict:
    """Hash the code actually used without publishing machine-local paths."""
    root = Path(__file__).resolve().parents[2]
    paths = list((root / "pie" / "behavior").glob("*.py"))
    paths += list((root / "pie" / "exchange").glob("*.py"))
    paths += [root / name for name in (
        "scripts/run_behavior.py", "pie/pruning/relp.py", "pie/models/patching.py")]
    code = {str(path.relative_to(root)): file_hash(path) for path in sorted(set(paths)) if path.is_file()}
    external = {}
    for package, names in {
        "circuit_tracer": ("replacement_model.py", "utils/hf_utils.py", "transcoder/cross_layer_transcoder.py"),
        "transformer_lens": ("HookedTransformer.py",),
    }.items():
        spec = importlib.util.find_spec(package)
        if spec and spec.origin:
            directory = Path(spec.origin).parent
            for name in names:
                if (directory / name).is_file():
                    external[f"{package}/{name}"] = file_hash(directory / name)
    versions = {}
    for name in ("torch", "numpy", "transformers", "transformer-lens", "huggingface-hub"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"code_sha256": code, "external_source_sha256": external, "dependency_versions": versions}


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def load_model(model_name: str, *, device="cuda:0", allow_download=False, native=False):
    """Use the exact model/CLT commits and BF16 policy of the local runs."""
    if model_name not in PROFILES:
        raise ValueError("model must be gemma or llama")
    selected_device = torch.device(device)
    if selected_device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Subject-model measurements require a CUDA GPU; CPU tests use fixtures.")
    if not allow_download:
        os.environ["HF_HUB_OFFLINE"] = "1"
    from circuit_tracer import ReplacementModel
    from circuit_tracer.utils.hf_utils import load_transcoder_from_hub
    from transformers import AutoTokenizer

    profile = PROFILES[model_name]
    torch.cuda.set_device(selected_device)
    torch.backends.cuda.matmul.allow_tf32 = not native
    torch.backends.cudnn.allow_tf32 = not native
    torch.backends.cudnn.benchmark = not native
    tokenizer = AutoTokenizer.from_pretrained(
        profile["model"], revision=profile["model_commit"], local_files_only=not allow_download)
    trans, _ = load_transcoder_from_hub(
        profile["clt"] + "@" + profile["clt_commit"], dtype=torch.bfloat16,
        lazy_encoder=False, lazy_decoder=True)
    trans.to(selected_device)
    model = ReplacementModel.from_pretrained_and_transcoders(
        profile["model"], trans, dtype=torch.bfloat16,
        revision=profile["model_commit"], local_files_only=not allow_download,
        tokenizer=tokenizer)
    model.to(selected_device)
    model.cfg.device = selected_device
    if native:
        model.eval()
    else:
        # The verified fixed-write protocol uses the initial replacement-model
        # mode. Calling eval() here would change that frozen configuration.
        attach_clt_device_patches(trans)
    return model, trans


def prepare_row(raw: dict, task: str, index: int, tokenizer) -> dict:
    row = prepare_prompt(raw, task, index)
    row["tokenization"] = {
        side: tokenize_prefix(tokenizer, row[f"text_{side}"], row[f"continuation_{side}"])
        for side in ("clean", "corr")
    }
    if row["tokenization"]["clean"]["token_count"] != row["tokenization"]["corr"]["token_count"]:
        raise ValueError(f"Pair {row['id']}: clean and corrupted prefixes have different token lengths")
    return row


def _capture_components(model, trans, tokens):
    with torch.no_grad():
        cache, hooks, _ = model.get_caching_hooks(lambda name: trans.feature_input_hook in name)
        model.run_with_hooks(tokens, fwd_hooks=hooks)
        if not cache:
            raise ValueError("No CLT feature input activations were captured")
        return trans.compute_attribution_components(torch.cat(list(cache.values()), dim=0))


def _measure(full, selected) -> dict:
    measured = dual_kl(full, selected)
    result = {name: float(measured[name][0].item()) for name in METRICS}
    if any(not math.isfinite(value) or value < -1e-6 for value in result.values()):
        raise ValueError("Nonfinite or invalid full-vocabulary KL")
    return result


def _tensor_hash(*tensors):
    hasher = hashlib.sha256()
    for tensor in tensors:
        value = tensor.detach().cpu().contiguous()
        hasher.update(fingerprint({"shape": list(value.shape), "dtype": str(value.dtype)}).encode())
        hasher.update(value.view(torch.uint8).numpy().tobytes())
    return hasher.hexdigest()


def _rng_state(device):
    return (torch.random.get_rng_state().clone(), torch.cuda.get_rng_state(device).clone()
            if device.type == "cuda" else None)


def _restore_rng(state, device):
    torch.random.set_rng_state(state[0])
    if state[1] is not None:
        torch.cuda.set_rng_state(state[1], device)


def _swap(prepared, initial, k, context, config, model, trans):
    from pie.exchange.search import run_exchange
    from pie.exchange.adapter import from_aligned_rows
    aligned = from_aligned_rows(
        pair_dataset_sha256=context["pair_dataset_sha256"], prompt_pair_id=context["prompt_pair_id"],
        model_revision=context["model_revision"], clt_revision=context["clt_revision"],
        source_layers=prepared["source_layers"].tolist(), source_positions=prepared["source_positions"].tolist(),
        feature_indices=prepared["source_features"].tolist(), receiver_layers=prepared["layers"].tolist(),
        receiver_positions=prepared["positions"].tolist(), encoder_to_decoder=prepared["row_map"].tolist(),
        base_scores=prepared["scores"].tolist(), keep_base=initial.tolist())
    measurements = []

    def scorer(view):
        keep = torch.tensor(aligned.mask_for(view), dtype=torch.bool)
        patches = build_removed_feature_patches(keep, prepared["layers"], prepared["positions"],
                                                prepared["row_map"], prepared["delta"], trans.n_layers)
        logits = patched_logits(model, trans, prepared["tokens"], patches)
        measurements.append(_measure(prepared["clean_logits"], logits))
        return max(0.0, measurements[-1][METRICS[0]])

    result = run_exchange(mode="single", initial=aligned.initial, universe=aligned.universe,
                          attribution=aligned.attribution, context=context, scorer=scorer, config=config)
    if result.error or any(row.get("error") for row in result.trace) or result.status in (
            "error", "verification_mismatch", "candidate_errors"):
        raise RuntimeError(f"Swap failed strict verification: {result.status}: {result.error}")
    if len(result.final) != k or result.verification_calls != 1 or \
            result.final_j > result.initial_j + config.acceptance_tolerance:
        raise ValueError("Swap violated its budget, monotonicity, or final verification contract")
    final = torch.tensor([key in result.final for key in aligned.ordered_keys], dtype=torch.bool)
    return final, measurements[0], measurements[-1], result.to_json()


def run_one(model, trans, row: dict, *, method: str, k_values,
            dataset_sha256: str, seed=0, swap_config=None,
            synergy_weight=3.0, boundary_percent=25.0, boundary_cap=32,
            partners_per_refresh=32, partners_per_candidate=8) -> dict:
    """Select exact-K occurrences and measure both endpoints on one paired prefix.

    The random control retains the historical clean-active zero-ablation
    intervention. It is labeled separately from the paired fixed-write methods.
    FActP scores every candidate; there is no hidden candidate truncation.
    """
    started = time.perf_counter()
    allowed = {"fap", "lastlse", "fap-swap", "lastlse-swap", "synergy", "activation", "clt-relp", "factp", "random"}
    if method not in allowed:
        raise ValueError(f"Unsupported fixed-write method: {method}")
    k_values = tuple(k_values)
    if not k_values or len(set(k_values)) != len(k_values) or any(type(k) is not int or k < 1 for k in k_values):
        raise ValueError("K values must be distinct positive integers")
    device = torch.device(model.cfg.device)
    tokens = {}
    for side in ("clean", "corr"):
        actual = model.ensure_tokenized(row[f"text_{side}"]).long()
        if actual.tolist() != row["tokenization"][side]["input_ids"] or not 8 <= actual.numel() <= MAX_TOKENS:
            raise ValueError("Model tokenization differs from the checked pre-answer prefix")
        tokens[side] = actual.to(device)
    if tokens["clean"].numel() != tokens["corr"].numel():
        raise ValueError("Clean/corrupt prefix token lengths differ")

    early_reference = method in {"activation", "factp", "random", "lastlse", "lastlse-swap"}
    clean_components = _capture_components(model, trans, tokens["clean"])
    reference_rng = None
    clean_logits = None
    if method in {"activation", "factp", "random"}:
        reference_rng = _rng_state(device)
        with torch.no_grad():
            clean_logits = model(tokens["clean"])
    corr_components = None if method == "random" else _capture_components(model, trans, tokens["corr"])
    corruption = ZeroFeaturesCorruption() if method == "random" else CorruptedPromptCorruption()
    (delta, layers, positions, count, row_map, sl, sp, sf) = corruption.build_delta_dec(
        trans, clean_components, corr_components)
    if count < max(k_values):
        raise ValueError(f"Only {count} eligible occurrences; cannot retain exact K={max(k_values)}")
    triples = [[int(a), int(b), int(c)] for a, b, c in zip(sl.tolist(), sp.tolist(), sf.tolist())]
    scoring_forwards = scoring_backwards = 0
    scalar_name = "last_logsumexp" if method.startswith("lastlse") else "logsumexp"
    metric = metric_factory(model, scalar_name, row)
    if early_reference and clean_logits is None:
        reference_rng = _rng_state(device)
        with torch.no_grad():
            clean_logits = model(tokens["clean"])
    if method in {"fap", "lastlse", "fap-swap", "lastlse-swap", "synergy"}:
        try:
            grads = collect_downstream_grads_at_hook(model, tokens["clean"], metric, trans.feature_output_hook)
        finally:
            model.reset_hooks()
        scores = fap_score_clt(delta, layers, positions, row_map, count, grads, trans.n_layers)
        scoring_forwards = scoring_backwards = 1
    elif method == "clt-relp":
        from .clt_relp import collect_downstream_relp_at_hook_safe, relp_score_clt, sequence_sum_logsumexp
        from pie.pruning import relp
        rho, restoration = collect_downstream_relp_at_hook_safe(
            relp, model, tokens["clean"], sequence_sum_logsumexp, trans.feature_output_hook)
        if not restoration["cleanup_complete"]:
            raise ValueError("CLT-RelP did not restore the original model")
        scores = relp_score_clt(relp, delta, layers, positions, row_map, count, rho, trans.n_layers)
        scoring_forwards = scoring_backwards = 1
    elif method == "activation":
        from .activation import rank_clean_activation_occurrences
        scores, _, activation_order, _ = rank_clean_activation_occurrences(clean_components["activation_matrix"], triples)
    elif method == "factp":
        from .factp import SingleOccurrencePatchBuilder, make_score_vectors
        builder = SingleOccurrencePatchBuilder(layers, positions, row_map, delta, count, trans.n_layers)
        ablated = []
        for index in range(count):
            selected = patched_logits(model, trans, tokens["clean"], builder.build_single_occurrence(index))
            ablated.append(float(selected.logsumexp(dim=-1).sum().item()))
        signed, scores = make_score_vectors(float(clean_logits.logsumexp(dim=-1).sum().item()), ablated)
        scoring_forwards = count
    else:
        scores = torch.zeros(count, dtype=torch.float32)
    if not bool(torch.isfinite(scores).all()) or scores.numel() != count:
        raise ValueError("Occurrence score vector is nonfinite or misaligned")
    if clean_logits is None:
        reference_rng = _rng_state(device)
        with torch.no_grad():
            clean_logits = model(tokens["clean"])

    after_scoring_rng = _rng_state(device)
    _restore_rng(reference_rng, device)
    parity_logits = patched_logits(model, trans, tokens["clean"], build_removed_feature_patches(
        torch.ones(count, dtype=torch.bool), layers, positions, row_map, delta, trans.n_layers))
    _restore_rng(after_scoring_rng, device)
    parity_error = float((clean_logits.float() - parity_logits.float()).abs().max().item())
    parity_metrics = _measure(clean_logits, parity_logits)
    if parity_error != 0.0 or any(abs(value) > 1e-6 for value in parity_metrics.values()):
        raise ValueError(f"All-retained fixed-write replay failed exact clean-logit parity: {parity_error}")

    prepared = {"tokens": tokens["clean"], "clean_logits": clean_logits, "delta": delta,
                "layers": layers, "positions": positions, "row_map": row_map,
                "source_layers": sl, "source_positions": sp, "source_features": sf, "scores": scores}
    profile = PROFILES.get(getattr(model, "_pie_profile_name", ""), {})
    context = {"model_revision": profile.get("model", "fixture") + "@" + profile.get("model_commit", "fixture"),
               "clt_revision": profile.get("clt", "fixture") + "@" + profile.get("clt_commit", "fixture"),
               "pair_dataset_sha256": dataset_sha256, "prompt_pair_id": str(row["id"]),
               "protocol_label": "pre_answer_prefix_first_token_v1", "evaluator_sha256": file_hash(Path(__file__)),
               "evaluator_config_sha256": fingerprint({"method": method, "kl_dtype": "float32", "seed": seed}),
               "intervention_sha256": _tensor_hash(delta, layers, positions, row_map)}
    by_k = {}
    for k in k_values:
        extra = {}
        if method == "activation":
            from .activation import select_topk_rows
            keep, _, _ = select_topk_rows(triples, scores, activation_order, k)
        elif method == "factp":
            from .factp import select_topk_occurrences
            keep, _, _ = select_topk_occurrences(triples, signed, scores, k)
        elif method == "random":
            keep = torch.zeros(count, dtype=torch.bool)
            keep[torch.randperm(count)[:k]] = True
        else:
            keep = _select_features_topk_global(scores, k)
        if method == "synergy":
            from .synergy import apply_synergy_reranking
            arguments = SimpleNamespace(keep_topk=k, boundary_percent=boundary_percent, boundary_cap=boundary_cap,
                                        partners_per_refresh=partners_per_refresh,
                                        syn_partners_per_candidate=partners_per_candidate,
                                        patch_scale=1.0, lambda_syn=synergy_weight)
            try:
                keep = apply_synergy_reranking(keep, scores, arguments, model, tokens["clean"], metric,
                                               trans.feature_output_hook, layers, positions, row_map,
                                               delta, trans, float(metric(clean_logits).item()), seed)
            finally:
                model.reset_hooks()
            extra["synergy_parameters"] = vars(arguments)
        if method.endswith("-swap"):
            if swap_config is None:
                from pie.exchange.search import SearchConfig
                swap_config = SearchConfig(max_candidate_evaluations=32, max_rounds=2, incoming_pool_size=8,
                                           outgoing_pool_size=6, max_proposal_attempts=256, seed=seed,
                                           acceptance_tolerance=1e-6, verification_tolerance=1e-7)
            initial = keep.clone()
            keep, initial_metrics, measured, search = _swap(prepared, keep, k, context, swap_config, model, trans)
            extra.update(initial_metrics=initial_metrics,
                         initial_occurrences=[triples[i] for i in torch.nonzero(initial).flatten().tolist()], search=search)
        else:
            selected = patched_logits(model, trans, tokens["clean"], build_removed_feature_patches(
                keep, layers, positions, row_map, delta, trans.n_layers))
            measured = _measure(clean_logits, selected)
        if int(keep.sum()) != k:
            raise ValueError(f"Method returned {int(keep.sum())} occurrences for requested K={k}")
        by_k[str(k)] = {"requested_k": k, "achieved_k": k,
                       "retained_occurrences": [triples[i] for i in torch.nonzero(keep).flatten().tolist()],
                       **measured, **extra}
    synchronize()
    return {"schema_version": "pie-reviewer-behavior-v1", "method": method,
            "protocol": "pre_answer_prefix_first_token_v1", "pair_id": str(row["id"]),
            "row_index": row["row_index"], "task": row["task"], "tokenization": row["tokenization"],
            "source_clean_sha256": row["source_clean_sha256"], "source_corr_sha256": row["source_corr_sha256"],
            "prefix_clean_sha256": row["prefix_clean_sha256"], "prefix_corr_sha256": row["prefix_corr_sha256"],
            "prediction_position": int(tokens["clean"].numel()) - 1, "token_count": int(tokens["clean"].numel()),
            "candidate_universe": "clean_active_zero_ablation" if method == "random" else "aligned_clean_corrupt_union",
            "candidate_occurrences": count, "occurrence_triples": triples, "scores": scores.tolist(),
            "selection_scalar": scalar_name if method in {"fap", "lastlse", "fap-swap", "lastlse-swap", "synergy", "clt-relp"} else None,
            "intervention": "fixed_cached_clean_write_minus_removed_clean_corrupt_delta" if method != "random" else "fixed_clean_write_zero_ablation",
            "kl_reference": "full_clean_replacement_model", "kl_direction": "full_clean_to_selected",
            "kl_dtype": "float32", "by_k": by_k,
            "all_retained_parity": {"max_abs_logit_error": parity_error, **parity_metrics},
            "selector_cost": {"forwards": scoring_forwards, "backwards": scoring_backwards},
            "elapsed_seconds": time.perf_counter() - started}


def aggregate(rows) -> dict:
    """Equal prompt weights; sample SD uses n-1 and is undefined for n=1."""
    if not rows:
        raise ValueError("Cannot aggregate an empty run")
    result = {}
    for k in rows[0]["by_k"]:
        result[k] = {}
        for metric in METRICS:
            values = [row["by_k"][k][metric] for row in rows]
            mean = math.fsum(values) / len(values)
            ss = math.fsum((value - mean) ** 2 for value in values)
            result[k][metric] = {"n": len(values), "mean": mean,
                                  "sd_population": math.sqrt(ss / len(values)),
                                  "sd_sample": math.sqrt(ss / (len(values) - 1)) if len(values) > 1 else None}
    return result


def execute(args) -> dict:
    """Entry point called only after dependency-light argument parsing."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.method == "native-replay":
        return execute_native(args)
    source = Path(args.data_file)
    chosen = []
    with source.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index < args.start_index:
                continue
            chosen.append((index, json.loads(line)))
            if args.num_samples is not None and len(chosen) >= args.num_samples:
                break
    if not chosen:
        raise ValueError("No input pairs selected")
    if len({str(raw["id"]) for _, raw in chosen}) != len(chosen):
        raise ValueError("Selected input pairs contain duplicate IDs")
    for index, raw in chosen:
        prepare_prompt(raw, args.task, index)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    dataset_hash = file_hash(source)
    started = time.perf_counter()
    model, trans = load_model(args.model, device=args.device, allow_download=args.allow_download)
    model._pie_profile_name = args.model
    from pie.exchange.search import SearchConfig
    config = SearchConfig(max_candidate_evaluations=args.swap_candidate_cap, max_rounds=args.swap_rounds,
                          incoming_pool_size=8, outgoing_pool_size=6, max_proposal_attempts=256,
                          seed=args.seed, acceptance_tolerance=1e-6, verification_tolerance=1e-7)
    config.validate()
    rows = []
    with (output / "per_prompt.jsonl").open("x", encoding="utf-8") as handle:
        for index, raw in chosen:
            row = prepare_row(raw, args.task, index, model.tokenizer)
            record = run_one(model, trans, row, method=args.method, k_values=args.keep_topk,
                             dataset_sha256=dataset_hash, seed=args.seed, swap_config=config,
                             synergy_weight=args.lambda_syn, boundary_percent=args.boundary_percent,
                             boundary_cap=args.boundary_cap, partners_per_refresh=args.partners_per_refresh,
                             partners_per_candidate=args.syn_partners_per_candidate)
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            rows.append(record)
            print(json.dumps({"pair_id": record["pair_id"], "method": args.method,
                              "candidate_occurrences": record["candidate_occurrences"],
                              "by_k": {k: {m: v[m] for m in METRICS} for k, v in record["by_k"].items()}}), flush=True)
    summary = aggregate(rows)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    runtime = {"torch": str(torch.__version__), "model_training": bool(model.training),
               "transcoder_training": bool(getattr(trans, "training", False)), "model_dtype": str(next(model.parameters()).dtype),
               "transcoder_dtype": str(next(trans.parameters()).dtype), "gpu": torch.cuda.get_device_name(),
               "tf32_matmul": torch.backends.cuda.matmul.allow_tf32, "tf32_cudnn": torch.backends.cudnn.allow_tf32,
               "output_logits_soft_cap": getattr(model.cfg, "output_logits_soft_cap", None),
               "attn_scores_soft_cap": getattr(model.cfg, "attn_scores_soft_cap", None)}
    manifest = {"schema_version": "pie-reviewer-behavior-run-v1", "status": "complete",
                "profile": PROFILES[args.model], "method": args.method, "task": args.task,
                "num_prompts": len(rows), "source_dataset_sha256": dataset_hash,
                "selected_file_rows": [i for i, _ in chosen], "k_values": list(args.keep_topk),
                "seed": args.seed, "runtime": runtime, "swap_config": asdict(config),
                "endpoints": list(METRICS), "sample_sd_denominator": len(rows) - 1,
                "result_scope": "new measurements; historical aggregate reproduction is not asserted",
                "elapsed_seconds": time.perf_counter() - started,
                "source_fingerprints": source_fingerprints(),
                "files": {name: file_hash(output / name) for name in ("per_prompt.jsonl", "summary.json")}}
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return manifest


def execute_native(args) -> dict:
    """Replay a separately supplied, hash-checked historical native prompt bundle."""
    from .native_replay import K_VALUES, evaluate_saved_prompt
    if tuple(args.keep_topk) != K_VALUES:
        raise ValueError("Saved native replay requires all five historical K values: 50,100,200,400,800")
    if not args.native_source_entry:
        raise ValueError("--native-source-entry is required; native graph inputs are not synthesized")
    source = json.loads(Path(args.native_source_entry).read_text(encoding="utf-8"))
    if source.get("model") != args.model or source.get("task") != args.task:
        raise ValueError("Native source model/task differs from the command")
    model, trans = load_model(args.model, device=args.device, allow_download=args.allow_download, native=True)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    result = evaluate_saved_prompt(model, trans, source, output, {
        "model_revision": PROFILES[args.model]["model"] + "@" + PROFILES[args.model]["model_commit"],
        "clt_revision": PROFILES[args.model]["clt"] + "@" + PROFILES[args.model]["clt_commit"],
    })
    (output / "native_result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return {"status": "complete", "method": "native-replay",
            "protocol": "saved_final_output_graph_full_prefix_direct_completion_v2"}
