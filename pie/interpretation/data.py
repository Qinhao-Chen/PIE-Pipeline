import json
import gzip
import zlib
import struct
import httpx
from typing import Dict, Any, List, Optional, Tuple, Counter
from pie.utils.common import cantor_unpair, cantor_pair

def _decompress_to_str(data: bytes) -> str:
    """Robust decompression handling gzip and zlib variants."""
    try: return gzip.decompress(data).decode("utf-8")
    except: pass
    try: return zlib.decompress(data, wbits=15 + 32).decode("utf-8")
    except: pass
    return zlib.decompress(data, wbits=-15).decode("utf-8")

class HFFeatureStore:
    """Fetches feature activation history from HuggingFace via range requests."""
    def __init__(self, scan: str, timeout_s: int = 60):
        self.scan = scan
        parts = scan.split("@", 1)
        self.repo_id = parts[0]
        self.revision = parts[1] if len(parts) == 2 else "main"
        self.timeout_s = timeout_s
        self._index: Optional[Dict[str, Any]] = None

    def _hf_resolve(self, path: str) -> str:
        return f"https://huggingface.co/{self.repo_id}/resolve/{self.revision}/features/{path}"

    async def load_index(self) -> Dict[str, Any]:
        if self._index is not None: return self._index
        url = self._hf_resolve("index.json.gz")
        async with httpx.AsyncClient(timeout=self.timeout_s, follow_redirects=True) as client:
            r = await client.get(url)
            r.raise_for_status()
            self._index = json.loads(_decompress_to_str(r.content))
        return self._index

    async def load_feature(self, feature_index: int) -> Dict[str, Any]:
        index = await self.load_index()
        layer_idx, feat_idx = cantor_unpair(feature_index)
        layer_info = index.get(str(layer_idx))
        
        if not layer_info: 
            raise KeyError(f"Layer {layer_idx} missing in index")
        
        offsets = layer_info.get("offsets")
        bin_file = layer_info.get("filename")
        
        if feat_idx < 0 or feat_idx + 1 >= len(offsets):
            raise IndexError(f"Feature index {feat_idx} out of bounds")

        start, end = offsets[feat_idx], offsets[feat_idx + 1]
        
        if start >= end:
            return {
                "featureIndex": feature_index,
                "examples_quantiles": [],
                "act_min": 0.0,
                "act_max": 0.0
            }
        range_end = max(start, end - 1)
        url = self._hf_resolve(bin_file)
        headers = {"Range": f"bytes={start}-{range_end}"}
        
        async with httpx.AsyncClient(timeout=self.timeout_s, follow_redirects=True) as client:
            r = await client.get(url, headers=headers)
            r.raise_for_status()
            raw = r.content
            
            if len(raw) < 4:
                return {"featureIndex": feature_index, "error": "empty_payload"}
            
            (data_len,) = struct.unpack("<I", raw[:4])
            feat_data = json.loads(_decompress_to_str(raw[4 : 4 + data_len]))
            
        feat_data["featureIndex"] = feature_index
        return feat_data

def load_pruned_features(jsonl_path: str, min_freq: int = 1) -> List[int]:
    """Loads feature indices from the pruning stage output."""
    freq = Counter()
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            # Support various key names used in pruning scripts
            vals = obj.get("kept_final") or obj.get("featureIndex_list") or []
            # Normalize to list of ints
            idx_list = []
            if isinstance(vals, list):
                for v in vals:
                    if isinstance(v, int): idx_list.append(v)
                    elif isinstance(v, dict) and "src_layer" in v:
                        idx_list.append(cantor_pair(v["src_layer"], v["src_feat"]))
            freq.update(idx_list)
    
    items = [(fid, c) for fid, c in freq.items() if c >= min_freq]
    # Sort by frequency desc
    items.sort(key=lambda x: -x[1])
    return [x[0] for x in items]