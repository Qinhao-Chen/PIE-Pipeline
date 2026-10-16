# -*- coding: utf-8 -*-
from __future__ import annotations

import glob
import logging
import os
import re  # === CHANGE: needed to parse layer indices
from typing import Dict, Iterable, NamedTuple, Optional, Tuple  # === CHANGE: consolidated typing imports
from urllib.parse import parse_qs, urlparse

import torch
import yaml
from safetensors.torch import safe_open  # kept (not used directly here but kept for parity)

from huggingface_hub import get_token, hf_api, hf_hub_download, snapshot_download
from huggingface_hub.constants import HF_HUB_ENABLE_HF_TRANSFER
from huggingface_hub.utils.tqdm import tqdm as hf_tqdm
from tqdm.contrib.concurrent import thread_map

logger = logging.getLogger(__name__)


# ----------------------------- HF URI helper -----------------------------
class HfUri(NamedTuple):
    """Structured representation of a HuggingFace URI."""

    repo_id: str
    file_path: Optional[str]
    revision: Optional[str]

    @classmethod
    def from_str(cls, hf_ref: str):
        if hf_ref.startswith("hf://"):
            return parse_hf_uri(hf_ref)

        parts = hf_ref.split("@", 1)
        repo_id = parts[0]
        revision = parts[1] if len(parts) > 1 else None
        return cls(repo_id, None, revision)


# ----------------------------- NEW: local helpers -----------------------------
# === CHANGE: new helpers to recognize and load local directories without calling snapshot_download.

def _is_local_dir(ref: str) -> bool:
    """Return True if `ref` is an existing local directory path."""
    try:
        return isinstance(ref, str) and os.path.isdir(ref)
    except Exception:
        return False


def _infer_consecutive_layers(layer_paths: Iterable[str]) -> Tuple[int, Dict[int, str]]:
    """Infer consecutive layer indices from file names 'layer_<i>.safetensors'."""
    idx_to_path: Dict[int, str] = {}
    for p in sorted(layer_paths):
        b = os.path.basename(p)
        m = re.match(r"layer_(\d+)\.safetensors$", b)
        if not m:
            continue
        idx = int(m.group(1))
        idx_to_path[idx] = p
    if not idx_to_path:
        raise RuntimeError("No layer_*.safetensors found.")
    # Require 0..N-1 consecutive
    indices = sorted(idx_to_path)
    if indices[0] != 0 or indices[-1] + 1 != len(indices):
        raise RuntimeError(
            f"Layer files must be consecutively numbered from 0..N-1. Found indices: {indices}"
        )
    return len(indices), idx_to_path


def _load_local_transcoders_dir(
    local_dir: str,
    device: Optional[torch.device],
    dtype: Optional[torch.dtype],
    lazy_encoder: bool,
    lazy_decoder: bool,
):
    """
    Load a local transcoder directory that contains:
      - config.yaml
      - layer_*.safetensors (required, 0..N-1)
      - features/ (optional)

    This path bypasses snapshot_download entirely.
    """
    cfg_path = os.path.join(local_dir, "config.yaml")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"Local transcoder dir missing config.yaml: {cfg_path}")

    with open(cfg_path, "r") as f:
        config = yaml.safe_load(f) or {}

    # === CHANGE: ensure expected keys exist for downstream ReplacementModel logic
    config.setdefault("feature_input_hook", "mlp.hook_in")
    config.setdefault("feature_output_hook", "mlp.hook_out")
    config.setdefault("model_kind", "transcoder_set")

    # Find consecutive layer files
    n_layers, idx_to_path = _infer_consecutive_layers(
        glob.glob(os.path.join(local_dir, "layer_*.safetensors"))
    )

    # === CHANGE: explicitly pass absolute paths so resolve_transcoder_paths won't try to download
    config["transcoders"] = [os.path.join(local_dir, f"layer_{i}.safetensors") for i in range(n_layers)]

    # Metadata for logging / UI
    config["repo_id"] = local_dir        # a local sentinel "repo id"
    config["revision"] = None
    config["scan"] = local_dir

    # === CHANGE: reuse the same loader path used by hub repos
    return load_transcoders(
        config,
        device=device,
        dtype=dtype,
        lazy_encoder=lazy_encoder,
        lazy_decoder=lazy_decoder,
    ), config


# ----------------------------- Public API -----------------------------
def load_transcoder_from_hub(
    hf_ref: str,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = torch.float32,
    lazy_encoder: bool = False,
    lazy_decoder: bool = True,
):
    """
    Enhanced loader:
      - === CHANGE: If `hf_ref` is a LOCAL DIRECTORY: load `config.yaml` + `layer_*.safetensors` from disk (no snapshot_download).
      - Else: treat as HF repo id (optionally with @revision), download files via hf_hub.

    Returns:
        (transcoders, config_dict)
    """
    # === CHANGE: local directory branch
    if _is_local_dir(hf_ref):
        return _load_local_transcoders_dir(
            hf_ref, device=device, dtype=dtype, lazy_encoder=lazy_encoder, lazy_decoder=lazy_decoder
        )

    # === CHANGE: support short-hands kept from project README
    if hf_ref == "gemma":
        hf_ref = "mntss/gemma-scope-transcoders"
    elif hf_ref == "llama":
        hf_ref = "mntss/transcoder-Llama-3.2-1B"

    # Parse user/repo[@rev]
    hf_uri = HfUri.from_str(hf_ref)

    # Download config.yaml from hub (this side keeps original behavior)
    try:
        config_path = hf_hub_download(
            repo_id=hf_uri.repo_id,
            revision=hf_uri.revision,
            filename="config.yaml",
        )
    except Exception as e:
        raise FileNotFoundError(f"Could not download config.yaml from {hf_uri.repo_id}") from e

    with open(config_path, "r") as f:
        config = yaml.safe_load(f) or {}

    # Metadata
    config["repo_id"] = hf_uri.repo_id
    config["revision"] = hf_uri.revision
    config["scan"] = f"{hf_uri.repo_id}@{hf_uri.revision}" if hf_uri.revision else hf_uri.repo_id

    # === CHANGE: ensure defaults for hooks/kind so downstream is robust
    config.setdefault("feature_input_hook", "mlp.hook_in")
    config.setdefault("feature_output_hook", "mlp.hook_out")
    config.setdefault("model_kind", "transcoder_set")

    # Delegate to common loader
    return load_transcoders(
        config,
        device=device,
        dtype=dtype,
        lazy_encoder=lazy_encoder,
        lazy_decoder=lazy_decoder,
    ), config


def load_transcoders(
    config: dict,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = torch.float32,
    lazy_encoder: bool = False,
    lazy_decoder: bool = True,
):
    """Load a transcoder from a config dict (local or hub)."""

    model_kind = config.get("model_kind", "transcoder_set")

    if model_kind == "transcoder_set":
        from circuit_tracer.transcoder.single_layer_transcoder import load_transcoder_set

        transcoder_paths = resolve_transcoder_paths(config)
        is_gemma_scope = "gemma-scope" in str(config.get("repo_id", ""))

        return load_transcoder_set(
            transcoder_paths,
            scan=config.get("scan"),
            feature_input_hook=config["feature_input_hook"],
            feature_output_hook=config["feature_output_hook"],
            gemma_scope=is_gemma_scope,
            dtype=dtype,
            device=device,
            lazy_encoder=lazy_encoder,
            lazy_decoder=lazy_decoder,
        )

    elif model_kind == "cross_layer_transcoder":
        from circuit_tracer.transcoder.cross_layer_transcoder import load_clt

        repo_id = config.get("repo_id")
        revision = config.get("revision", "main")

        # === CHANGE: allow local directory for cross-layer transcoders too
        if _is_local_dir(repo_id):
            local_path = repo_id
        else:
            local_path = snapshot_download(
                repo_id, revision=revision, allow_patterns=["*.safetensors"]
            )

        return load_clt(
            local_path,
            scan=config.get("scan"),
            feature_input_hook=config["feature_input_hook"],
            feature_output_hook=config["feature_output_hook"],
            lazy_decoder=lazy_decoder,
            lazy_encoder=lazy_encoder,
            dtype=dtype,
            device=device,
        )
    else:
        raise ValueError(f"Unknown model kind: {model_kind}")


def resolve_transcoder_paths(config: dict) -> Dict[int, str]:
    """
    Return a mapping {layer_idx: local_file_path} for a "transcoder_set".
    Prefers explicit paths in `config['transcoders']`. Otherwise resolves from repo/cache.
    """
    # === CHANGE: if explicit paths are provided, honor them (hf:// entries are downloaded; local/absolute kept as-is)
    if "transcoders" in config and config["transcoders"]:
        hf_uris = [p for p in config["transcoders"] if isinstance(p, str) and p.startswith("hf://")]
        local_map = download_hf_uris(hf_uris) if hf_uris else {}
        paths: Dict[int, str] = {}
        for i, p in enumerate(config["transcoders"]):
            paths[i] = local_map.get(p, p)  # keep local/absolute paths intact
        return paths

    # === CHANGE: if repo_id is a local directory, glob there without snapshot_download
    repo_id = config.get("repo_id", "")
    if _is_local_dir(repo_id):
        layer_files = sorted(glob.glob(os.path.join(repo_id, "layer_*.safetensors")))
        n, idx_to_path = _infer_consecutive_layers(layer_files)
        return {i: idx_to_path[i] for i in range(n)}

    # Fallback: download from hub
    local_path = snapshot_download(
        repo_id,
        revision=config.get("revision", "main"),
        allow_patterns=["layer_*.safetensors"],
    )
    layer_files = glob.glob(os.path.join(local_path, "layer_*.safetensors"))
    n, idx_to_path = _infer_consecutive_layers(layer_files)
    return {i: idx_to_path[i] for i in range(n)}


# ----------------------------- HF URI utilities (unchanged except doc & minor guards) -----------------------------
def parse_hf_uri(uri: str) -> HfUri:
    """Parse an HF URI into repo id, file path and revision.

    Args:
        uri: String like ``hf://org/repo/file?revision=main``.

    Returns:
        ``HfUri`` with repository id, file path and optional revision.
    """
    parsed = urlparse(uri)
    if parsed.scheme != "hf":
        raise ValueError(f"Not a huggingface URI: {uri}")
    path = parsed.path.lstrip("/")
    repo_parts = path.split("/", 1)
    if len(repo_parts) != 2:
        raise ValueError(f"Invalid huggingface URI: {uri}")
    repo_id = f"{parsed.netloc}/{repo_parts[0]}"
    file_path = repo_parts[1]
    revision = parse_qs(parsed.query).get("revision", [None])[0] or None
    return HfUri(repo_id, file_path, revision)


def download_hf_uri(uri: str) -> str:
    """Download a file referenced by a HuggingFace URI and return the local path."""
    info = parse_hf_uri(uri)
    return hf_hub_download(
        repo_id=info.repo_id,
        filename=info.file_path,
        revision=info.revision,
        force_download=False,
    )


def download_hf_uris(uris: Iterable[str], max_workers: int = 8) -> Dict[str, str]:
    """Download multiple HuggingFace URIs concurrently with pre-flight auth checks.

    Args:
        uris: Iterable of HF URIs.
        max_workers: Maximum number of parallel workers.

    Returns:
        Mapping from input URI to the local file path on disk.
    """
    if not uris:
        return {}

    uri_list = list(uris)
    if not uri_list:
        return {}

    parsed_map = {uri: parse_hf_uri(uri) for uri in uri_list}

    # Pre-flight check
    logger.info("Performing pre-flight metadata check...")
    unique_repos = {info.repo_id for info in parsed_map.values()}
    token = get_token()

    for repo_id in unique_repos:
        # This will raise if the repo is gated and no token is available.
        if hf_api.repo_info(repo_id=repo_id, token=token).gated is not False:
            if token is None:
                raise PermissionError("Cannot access a gated repo without a HF token.")

    logger.info("Pre-flight check complete. Starting downloads...")

    def _download(uri: str) -> str:
        info = parsed_map[uri]
        return hf_hub_download(
            repo_id=info.repo_id,
            filename=info.file_path,
            revision=info.revision,
            token=token,
            force_download=False,
        )

    if HF_HUB_ENABLE_HF_TRANSFER:
        results = [_download(uri) for uri in uri_list]
        return dict(zip(uri_list, results))

    results = thread_map(
        _download,
        uri_list,
        desc=f"Fetching {len(parsed_map)} files",
        max_workers=max_workers,
        tqdm_class=hf_tqdm,
    )
    return dict(zip(uri_list, results))
