import types
import torch
from typing import Callable, Dict, Any
from transformer_lens.utilities.addmm import batch_addmm
from circuit_tracer import ReplacementModel


# --------------------------- LRP rules ---------------------------

def stabilize(z: torch.Tensor) -> torch.Tensor:
    return z + ((z == 0.).to(z) + z.sign()) * 1e-6


def _identity_rule_act(act_fn: Callable, x: torch.Tensor) -> torch.Tensor:
    z = act_fn(x)
    zp = stabilize(x)
    return zp * (z / zp).data


def _gated_mlp_forward_relp(self, x):
    """LRP-augmented forward for transformer_lens GatedMLP.

    Replaces the activation with the Identity-rule and applies the Half-rule on
    the gated branch so that backward yields LRP relevance coefficients instead
    of raw gradients.
    """
    if self.W_gate.device != x.device:
        x = x.to(self.W_gate.device)
    pre_act = self.hook_pre(torch.matmul(x, self.W_gate))

    if (
        self.cfg.is_layer_norm_activation()
        and self.hook_mid is not None
        and self.ln is not None
    ):
        mid_act = self.hook_mid(self.act_fn(pre_act))
        post_act = self.hook_post(self.ln(mid_act))
    else:
        pre_linear = self.hook_pre_linear(torch.matmul(x, self.W_in))
        act_out = _identity_rule_act(self.act_fn, pre_act)
        gate_out = act_out * pre_linear
        gate_out = (gate_out / 2.) + (gate_out / 2.).detach()
        post_act = self.hook_post(gate_out + self.b_in)

    return batch_addmm(self.b_out, self.W_out, post_act)


# --------------------------- RelP coefficient collection ---------------------------

def collect_downstream_relp_at_hook(model: ReplacementModel, tokens_1d: torch.Tensor,
                                    metric_fn: Callable[[torch.Tensor], torch.Tensor],
                                    hook_name: str) -> Dict[int, torch.Tensor]:
    """Collect RelP propagation coefficients (rho) at every layer's hook_name.

    Temporarily monkey-patches each block's MLP with the LRP-augmented forward,
    detaches gradients through attention pattern and LayerNorm scales, runs
    backward of metric_fn, and returns per-layer relevance maps.
    """
    dev = model.cfg.device
    rho_cache: Dict[int, torch.Tensor] = {}
    fwd_tensors: Dict[int, torch.Tensor] = {}

    model.reset_hooks(including_permanent=True)

    saved_mlp_overrides: Dict[int, Any] = {}
    for i, block in enumerate(model.blocks):
        mlp = block.mlp.old_mlp
        if 'forward' in mlp.__dict__:
            saved_mlp_overrides[i] = mlp.__dict__['forward']
        mlp.forward = types.MethodType(_gated_mlp_forward_relp, mlp)

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

    for i, block in enumerate(model.blocks):
        mlp = block.mlp.old_mlp
        if i in saved_mlp_overrides:
            mlp.__dict__['forward'] = saved_mlp_overrides[i]
        else:
            mlp.__dict__.pop('forward', None)

    model._configure_gradient_flow()
    model.setup()

    for L, acts in fwd_tensors.items():
        g = acts.grad
        if g is None: continue
        if g.ndim == 3: g = g.squeeze(0)
        rho_cache[L] = g.to(torch.float32).detach()

    assert any(g.abs().sum().item() > 0 for g in rho_cache.values()), \
        "No RelP propagation coefficients captured."
    return rho_cache


# --------------------------- Scoring ---------------------------

def relp_score_clt(delta_dec: torch.Tensor, layer_ids: torch.Tensor, pos_ids: torch.Tensor,
                   enc2rows: torch.Tensor, n_enc_occ: int,
                   rho_at_mlp_out: Dict[int, torch.Tensor], n_layers: int) -> torch.Tensor:
    """Score each encoder occurrence by <delta_dec, rho> summed over decoder rows."""
    if n_enc_occ == 0 or delta_dec.numel() == 0:
        return torch.zeros((0,), dtype=torch.float32)
    R, d = delta_dec.shape

    rho_rows = torch.empty((R, d), dtype=torch.float32, device="cpu")
    for t in range(n_layers):
        mask_t = (layer_ids.cpu() == t)
        if not mask_t.any(): continue
        pos_t = pos_ids.cpu()[mask_t]
        rho_mat = rho_at_mlp_out.get(t)
        if rho_mat is None:
            rho_rows[mask_t] = 0.0
        else:
            rho_rows[mask_t] = rho_mat.to("cpu", dtype=torch.float32).index_select(0, pos_t)

    delta_cpu = delta_dec.to("cpu", dtype=torch.float32)
    dots = (delta_cpu * rho_rows).sum(dim=-1)
    scores = torch.zeros((n_enc_occ,), dtype=torch.float32, device="cpu")
    scores.index_add_(0, enc2rows.cpu(), dots)
    return scores
