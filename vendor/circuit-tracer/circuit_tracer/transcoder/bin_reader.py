#!/usr/bin/env python3
"""
inspect_hf_layer_bin.py

Inspect the structure of a HuggingFace-hosted feature store file:
  - features/index.json.gz
  - features/layer_*.bin

This script:
  1) downloads index.json.gz
  2) resolves the layer's .bin filename + offsets
  3) range-requests a single feature record (no full download)
  4) parses the record (uint32 length + deflate/zlib/gzip JSON)
  5) prints structure + head preview

Example:
  python inspect_hf_layer_bin.py \
    --repo mntss/clt-gemma-2-2b-426k \
    --revision main \
    --layer 0 \
    --feat 0
"""

import argparse
import gzip
import json
import struct
import zlib
from typing import Any, Dict, Tuple, Optional, List

import httpx


def hf_resolve(repo: str, revision: str, path: str) -> str:
    # Raw file URL (not the "blob" UI)
    return f"https://huggingface.co/{repo}/resolve/{revision}/{path}"


def _decompress_to_str(data: bytes) -> str:
    # Try gzip
    try:
        return gzip.decompress(data).decode("utf-8")
    except Exception:
        pass
    # Try zlib (with gzip header)
    try:
        return zlib.decompress(data, wbits=15 + 32).decode("utf-8")
    except Exception:
        pass
    # Try raw DEFLATE
    return zlib.decompress(data, wbits=-15).decode("utf-8")


def parse_index_json_gz(raw: bytes) -> Dict[str, Any]:
    s = _decompress_to_str(raw)
    return json.loads(s)


def parse_bin_record(raw: bytes) -> Dict[str, Any]:
    """
    Typical frontend format:
      bytes[0:4] -> little-endian uint32 dataLength
      bytes[4:4+dataLength] -> compressed stream (deflate/zlib/gzip) -> JSON string
    """
    if len(raw) < 4:
        raise ValueError("Record too short (<4 bytes).")
    (data_len,) = struct.unpack("<I", raw[:4])
    comp = raw[4 : 4 + data_len]
    if len(comp) != data_len:
        raise ValueError(f"Record truncated: need {data_len} compressed bytes, got {len(comp)}.")
    s = _decompress_to_str(comp)
    return json.loads(s)


def short_type(x: Any) -> str:
    if isinstance(x, dict):
        return f"dict(len={len(x)})"
    if isinstance(x, list):
        return f"list(len={len(x)})"
    if isinstance(x, str):
        return f"str(len={len(x)})"
    if isinstance(x, (int, float, bool)) or x is None:
        return type(x).__name__
    return type(x).__name__


def print_feature_head(feature: Dict[str, Any], max_logits: int = 10, max_tok: int = 80):
    print("\n=== FEATURE HEAD PREVIEW ===")
    keys = sorted(feature.keys())
    print("Top-level keys:", keys)

    # Common fields in your Neuronpedia-style feature JSON
    for k in ["index", "featureIndex", "scan", "act_min", "act_max", "activation_frequency"]:
        if k in feature:
            print(f"{k}: {feature.get(k)}")

    top_logits = feature.get("top_logits") or []
    bottom_logits = feature.get("bottom_logits") or []
    if top_logits:
        print("top_logits[:10]:", top_logits[:max_logits])
    if bottom_logits:
        print("bottom_logits[:10]:", bottom_logits[:max_logits])

    exq = feature.get("examples_quantiles") or []
    print("examples_quantiles:", short_type(exq))
    if exq and isinstance(exq, list):
        q0 = exq[0] if exq else None
        if isinstance(q0, dict):
            ex0s = q0.get("examples") or []
            if ex0s:
                e0 = ex0s[0]
                toks = (e0.get("tokens") or [])[:max_tok]
                acts = (e0.get("tokens_acts_list") or [])[:max_tok]
                print("\nExample0 tokens head:", toks)
                print("Example0 acts head:  ", acts)
            else:
                print("No examples in the first quantile bucket.")
        else:
            print("examples_quantiles[0] is not a dict.")
    else:
        print("No examples_quantiles found.")

    # Shallow “structure”
    print("\n=== SHALLOW STRUCTURE (key -> type) ===")
    for k in keys:
        print(f"{k:24s} -> {short_type(feature[k])}")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="HF repo id, e.g. mntss/clt-gemma-2-2b-426k")
    ap.add_argument("--revision", default="main", help="HF revision/branch/tag (default: main)")
    ap.add_argument("--layer", type=int, default=0, help="Layer id to inspect (default: 0)")
    ap.add_argument("--feat", type=int, default=0, help="Feature idx within layer to inspect (default: 0)")
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--print_bin_hex_head", action="store_true",
                    help="Also fetch and print first 256 bytes of layer_*.bin (hex).")
    args = ap.parse_args()

    index_url = hf_resolve(args.repo, args.revision, "features/index.json.gz")

    async with httpx.AsyncClient(timeout=args.timeout, follow_redirects=True) as client:
        # 1) Load index
        print("[1/3] Downloading index.json.gz ...")
        r = await client.get(index_url)
        r.raise_for_status()
        index = parse_index_json_gz(r.content)
        print("Index loaded. Layer keys:", list(index.keys())[:10], "...")

        layer_key = str(args.layer)
        if layer_key not in index:
            raise KeyError(f"Layer {args.layer} not found in index.json.gz. Available: {list(index.keys())[:20]}...")

        info = index[layer_key]
        offsets = info.get("offsets")
        bin_filename = info.get("filename")
        if not offsets or not bin_filename:
            raise KeyError(f"Missing offsets/filename for layer {args.layer} in index.json.gz")

        print(f"\n[Layer {args.layer}] filename = {bin_filename}")
        print(f"[Layer {args.layer}] offsets length = {len(offsets)} (usually d_transcoder+1)")
        print("offsets[:10] =", offsets[:10])

        bin_url = hf_resolve(args.repo, args.revision, f"features/{bin_filename}")

        # optional: print raw file head
        if args.print_bin_hex_head:
            print("\n[2a/3] Fetching first 256 bytes of bin file (hex head) ...")
            r0 = await client.get(bin_url, headers={"Range": "bytes=0-255"})
            r0.raise_for_status()
            head = r0.content
            print("bin head (len={}):".format(len(head)))
            print(head.hex())

        # 2) Range fetch a single feature record
        feat = args.feat
        if feat < 0 or feat + 1 >= len(offsets):
            raise IndexError(f"feat={feat} out of range. Must satisfy 0 <= feat < {len(offsets)-1}")

        start = int(offsets[feat])
        end = int(offsets[feat + 1])
        if end <= start:
            raise ValueError(f"Bad offsets: offsets[{feat}]={start}, offsets[{feat+1}]={end}")
        range_end = end - 1

        print(f"\n[2/3] Fetching feature record feat={feat} with Range bytes={start}-{range_end} ...")
        rr = await client.get(bin_url, headers={"Range": f"bytes={start}-{range_end}"})
        rr.raise_for_status()
        raw_record = rr.content
        print(f"Downloaded {len(raw_record)} bytes for this record.")

        # 3) Parse record
        print("\n[3/3] Parsing record ...")
        feature_json = parse_bin_record(raw_record)

        # attach useful metadata for display
        feature_json["__repo__"] = args.repo
        feature_json["__revision__"] = args.revision
        feature_json["__layer__"] = args.layer
        feature_json["__feat__"] = feat
        feature_json["__bin_filename__"] = bin_filename
        feature_json["__range__"] = [start, range_end]

        print_feature_head(feature_json)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
