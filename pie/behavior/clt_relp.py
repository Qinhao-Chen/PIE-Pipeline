"""Exception-safe adapter for the frozen local PIE CLT-RelP implementation.

The scoring rule remains the one in ``PIE-Pipeline/pie/pruning/relp.py``.
Only coefficient collection is mirrored here so temporary hooks and MLP
forward overrides are restored even when the model forward/backward fails.
"""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import types
from typing import Callable

import torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_relp_source(source_path: Path):
    """Load the local, provenance-pinned RelP module by explicit file path."""
    source_path = Path(source_path).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if source_path.name != "relp.py" or source_path.parent.name != "pruning":
        raise ValueError(f"unexpected RelP source location: {source_path}")
    spec = importlib.util.spec_from_file_location("pie_taskc_pinned_relp", source_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load local RelP source: {source_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    required = ("_gated_mlp_forward_relp", "relp_score_clt")
    missing = [name for name in required if not callable(getattr(module, name, None))]
    if missing:
        raise ImportError(f"local RelP source lacks required functions: {missing}")
    module.__source_path__ = str(source_path)
    module.__source_sha256__ = sha256_file(source_path)
    return module


def sequence_sum_logsumexp(logits: torch.Tensor) -> torch.Tensor:
    """The original RelP/FAP distribution-agnostic sequence objective."""
    return logits.logsumexp(dim=-1).sum()


def _restore_model(model, mlps, original_forward_overrides) -> dict:
    """Always clear transient hooks, restore monkey patches, then rebuild TL state."""
    errors = []
    result = {
        "cleanup_complete": False,
        "hooks_reset": False,
        "mlp_forward_overrides_restored": False,
        "gradient_flow_configured": False,
        "setup_reinitialized": False,
        "mlp_modules_checked": len(mlps),
        "original_forward_overrides": sum(
            1 for existed, _ in original_forward_overrides if existed
        ),
    }
    try:
        model.reset_hooks()
        result["hooks_reset"] = True
    except BaseException as error:  # cleanup must continue to restore forwards
        errors.append(("reset_hooks", error))

    for mlp, (had_override, prior_override) in zip(mlps, original_forward_overrides):
        try:
            if had_override:
                mlp.__dict__["forward"] = prior_override
            else:
                mlp.__dict__.pop("forward", None)
        except BaseException as error:
            errors.append(("restore_mlp_forward", error))

    result["mlp_forward_overrides_restored"] = all(
        ("forward" in mlp.__dict__) == had_override and
        (not had_override or mlp.__dict__.get("forward") is prior_override)
        for mlp, (had_override, prior_override)
        in zip(mlps, original_forward_overrides)
    )

    try:
        model._configure_gradient_flow()
        result["gradient_flow_configured"] = True
    except BaseException as error:
        errors.append(("configure_gradient_flow", error))
    try:
        model.setup()
        result["setup_reinitialized"] = True
    except BaseException as error:
        errors.append(("setup", error))

    result["cleanup_complete"] = not errors and all(
        result[key] for key in (
            "hooks_reset", "mlp_forward_overrides_restored",
            "gradient_flow_configured", "setup_reinitialized",
        )
    )
    if errors:
        result["cleanup_errors"] = [
            {"step": label, "error_type": type(error).__name__, "message": str(error)}
            for label, error in errors
        ]
    return result


def collect_downstream_relp_at_hook_safe(
    relp_module,
    model,
    tokens_1d: torch.Tensor,
    metric_fn: Callable[[torch.Tensor], torch.Tensor],
    hook_name: str,
) -> tuple[dict[int, torch.Tensor], dict]:
    """Faithfully collect upstream RelP ``rho`` coefficients with safe restore.

    This mirrors the local upstream ``collect_downstream_relp_at_hook``:
    RelP-augmented gated-MLP forward, detached attention patterns and layernorm
    scales, an embedding-gradient path, and retained gradients at every CLT
    feature-output hook. The only behavioral change is unconditional cleanup.
    """
    if not callable(getattr(relp_module, "_gated_mlp_forward_relp", None)):
        raise TypeError("loaded RelP module lacks _gated_mlp_forward_relp")
    dev = model.cfg.device
    n_layers = int(model.cfg.n_layers)
    mlps = [block.mlp.old_mlp for block in model.blocks]
    if len(mlps) != n_layers:
        raise ValueError("model block/declared layer counts differ")
    original_forward_overrides = [
        ("forward" in mlp.__dict__, mlp.__dict__.get("forward")) for mlp in mlps
    ]
    fwd_tensors: dict[int, torch.Tensor] = {}
    body_error = None
    body_traceback = None

    try:
        model.reset_hooks(including_permanent=True)
        for mlp in mlps:
            mlp.forward = types.MethodType(relp_module._gated_mlp_forward_relp, mlp)

        def stop_gradient(acts, hook):
            return acts.detach()

        for block in model.blocks:
            block.attn.hook_pattern.add_hook(stop_gradient)
            block.ln1.hook_scale.add_hook(stop_gradient)
            block.ln2.hook_scale.add_hook(stop_gradient)
            if hasattr(block, "ln1_post"):
                block.ln1_post.hook_scale.add_hook(stop_gradient)
            if hasattr(block, "ln2_post"):
                block.ln2_post.hook_scale.add_hook(stop_gradient)
        model.ln_final.hook_scale.add_hook(stop_gradient)

        def enable_gradient(acts, hook):
            acts.requires_grad = True
            return acts

        model.hook_embed.add_hook(enable_gradient)

        def fwd_passthrough(layer: int):
            def retain(acts, hook):
                if torch.is_floating_point(acts) and not acts.requires_grad:
                    acts = acts * 1.0
                acts.retain_grad()
                fwd_tensors[layer] = acts
                return acts
            return retain

        for layer in range(n_layers):
            model.add_hook(f"blocks.{layer}.{hook_name}", fwd_passthrough(layer), "fwd")

        with torch.enable_grad():
            logits = model(tokens_1d.to(dev))
            value = metric_fn(logits)
            if not isinstance(value, torch.Tensor) or value.numel() != 1:
                raise ValueError("RelP metric must return one scalar tensor")
            model.zero_grad(set_to_none=True)
            value.backward()
    except BaseException as error:
        body_error = error
        body_traceback = error.__traceback__

    restoration = _restore_model(model, mlps, original_forward_overrides)
    if body_error is not None:
        if not restoration["cleanup_complete"]:
            raise RuntimeError(
                f"RelP collection failed ({type(body_error).__name__}) and model restoration was incomplete: "
                f"{restoration.get('cleanup_errors', restoration)}"
            ) from body_error
        raise body_error.with_traceback(body_traceback)
    if not restoration["cleanup_complete"]:
        raise RuntimeError(f"RelP model restoration was incomplete: {restoration}")

    rho_cache: dict[int, torch.Tensor] = {}
    for layer, activations in fwd_tensors.items():
        gradient = activations.grad
        if gradient is None:
            continue
        if gradient.ndim == 3:
            gradient = gradient.squeeze(0)
        rho_cache[layer] = gradient.to(torch.float32).detach()
    missing_layers = sorted(set(range(n_layers)) - set(rho_cache))
    if missing_layers:
        raise AssertionError(f"RelP propagation coefficients missing layers: {missing_layers}")
    if not any(bool(value.abs().sum().item() > 0) for value in rho_cache.values()):
        raise AssertionError("No RelP propagation coefficients captured.")
    restoration["rho_layers_with_gradient"] = len(rho_cache)
    restoration["rho_layers_expected"] = n_layers
    return rho_cache, restoration


def relp_score_clt(
    relp_module,
    delta_dec: torch.Tensor,
    layer_ids: torch.Tensor,
    pos_ids: torch.Tensor,
    enc2rows: torch.Tensor,
    n_enc_occ: int,
    rho_at_mlp_out: dict[int, torch.Tensor],
    n_layers: int,
) -> torch.Tensor:
    """Call the source-pinned upstream occurrence scorer and check its result."""
    scores = relp_module.relp_score_clt(
        delta_dec, layer_ids, pos_ids, enc2rows, n_enc_occ, rho_at_mlp_out, n_layers,
    )
    if not isinstance(scores, torch.Tensor) or scores.ndim != 1 or scores.numel() != n_enc_occ:
        raise ValueError("upstream RelP scorer returned a misaligned score vector")
    scores = scores.detach().to(device="cpu", dtype=torch.float32)
    if not bool(torch.isfinite(scores).all()):
        raise ValueError("upstream RelP scorer returned nonfinite scores")
    return scores
