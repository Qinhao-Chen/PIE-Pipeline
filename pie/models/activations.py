import torch
import numpy as np
from tqdm import tqdm

@torch.inference_mode()
def compute_activations(model, tokenizer, clt, texts, feature_locs, batch_size=64, device=None):
    """Max CLT activation per text at the actual encoder input hook.

    ``model`` must be a circuit-tracer ReplacementModel. The CLT applies its own
    encoder activation function, including JumpReLU thresholds. Texts run
    individually so padding and BOS cannot contribute to the maximum.
    ``tokenizer`` and ``batch_size`` remain accepted for caller compatibility.
    """
    by_layer = {}
    for fid, loc in feature_locs.items():
        layer, index = loc["layer"], loc["index"]
        if not 0 <= layer < clt.n_layers or not 0 <= index < clt._get_encoder_weights(layer).shape[0]:
            raise ValueError(f"Feature {fid} is outside the loaded CLT")
        by_layer.setdefault(layer, []).append((fid, index))
    results = {fid: np.zeros(len(texts), dtype=np.float32) for fid in feature_locs}
    selected_device = device or model.cfg.device
    for row, text in enumerate(tqdm(texts, desc="Computing Acts")):
        tokens = model.ensure_tokenized(text).to(selected_device)
        if tokens.numel() < 2:
            raise ValueError("Evaluation text must contain a token after BOS")
        # Match the earlier evaluation's 128-token window without padding.
        tokens = tokens[:128]
        captured = {}
        hooks = []
        for layer, features in by_layer.items():
            def capture(value, hook, layer=layer, features=features):
                encoded = clt.encode_layer(value, layer)
                indices = torch.tensor([index for _, index in features], device=encoded.device)
                maxima = encoded[0, 1:].index_select(-1, indices).amax(dim=0).float().cpu().tolist()
                captured.update({fid: activation for (fid, _), activation in zip(features, maxima)})
                return value
            hooks.append((f"blocks.{layer}.{clt.feature_input_hook}", capture))
        model.run_with_hooks(tokens, fwd_hooks=hooks)
        if set(captured) != set(results):
            raise ValueError("Some CLT encoder hooks were not reached")
        for fid, activation in captured.items():
            results[fid][row] = activation
    return results
