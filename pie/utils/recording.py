import json
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional
import torch
from .common import cantor_pair

def sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def load_samples(path: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit: break
            line = line.rstrip("\n")
            row = None
            if line and line.startswith("{") and line.endswith("}"):
                try: row = json.loads(line)
                except Exception: row = None
            if isinstance(row, dict) and "text_clean" in row:
                out.append({
                    "text_clean": row["text_clean"],
                    "text_corr": row.get("text_corr"),
                    "pos_target": row.get("io_clean"),
                    "neg_target": row.get("s_clean"),
                    "raw": row,
                })
            else:
                out.append({
                    "text_clean": line,
                    "text_corr": None,
                    "pos_target": None,
                    "neg_target": None,
                    "raw": line,
                })
    return out

class FeatureSelectionRecorder:
    """Records kept encoder features into explain-script-compatible featureIndex list."""
    def __init__(self):
        self.base: Dict[int, Any] = {}   # ordinary FAP keep_base
        self.final: Dict[int, Any] = {}  # actual keep_mask (after synergy)

    def _update_one(self, bucket, occ_src_layer, occ_src_pos, occ_src_feat, scores_feat, keep_mask):
        occ_src_layer = occ_src_layer.detach().to("cpu")
        occ_src_pos   = occ_src_pos.detach().to("cpu")
        occ_src_feat  = occ_src_feat.detach().to("cpu")
        scores_feat   = scores_feat.detach().to("cpu")
        keep_mask     = keep_mask.detach().to("cpu").to(torch.bool)

        n = int(scores_feat.numel())
        for i in range(n):
            L = int(occ_src_layer[i].item())
            P = int(occ_src_pos[i].item())
            FID = int(occ_src_feat[i].item())
            featureIndex = int(cantor_pair(L, FID))
            s = float(scores_feat[i].item())
            kept = bool(keep_mask[i].item())

            st = bucket.get(featureIndex)
            if st is None:
                st = {
                    "src_layer": L, "src_feat": FID,
                    "seen_count": 0, "kept_count": 0,
                    "sum_abs_score": 0.0, "sum_score": 0.0,
                    "pos_counts": {},
                }
                bucket[featureIndex] = st

            st["seen_count"] += 1
            st["kept_count"] += (1 if kept else 0)
            st["sum_abs_score"] += abs(s)
            st["sum_score"] += s
            pc = st["pos_counts"]
            pc[P] = pc.get(P, 0) + 1

    def update(self, occ_src_layer, occ_src_pos, occ_src_feat, scores_feat, keep_base, keep_final):
        self._update_one(self.base,  occ_src_layer, occ_src_pos, occ_src_feat, scores_feat, keep_base)
        self._update_one(self.final, occ_src_layer, occ_src_pos, occ_src_feat, scores_feat, keep_final)

    @staticmethod
    def write_outputs(out_dir: Path, prefix: str, stats: Dict[int, Any], top_n: Optional[int] = None):
        out_dir.mkdir(parents=True, exist_ok=True)
        # Helper to summarize stats (moved from global scope to static method or internal)
        def _finalize_stats(stats):
            out = {}
            for fid, st in stats.items():
                seen = int(st.get("seen_count", 0))
                kept = int(st.get("kept_count", 0))
                sum_abs = float(st.get("sum_abs_score", 0.0))
                sum_s   = float(st.get("sum_score", 0.0))
                pc = st.get("pos_counts", {}) or {}
                top_pos = sorted(pc.items(), key=lambda kv: -kv[1])[:20]
                out[fid] = {
                    "featureIndex": int(fid),
                    "src_layer": int(st.get("src_layer", -1)),
                    "src_feat": int(st.get("src_feat", -1)),
                    "seen_count": seen, "kept_count": kept,
                    "kept_rate": (kept / seen) if seen > 0 else 0.0,
                    "mean_abs_score": (sum_abs / seen) if seen > 0 else 0.0,
                    "mean_score": (sum_s / seen) if seen > 0 else 0.0,
                    "top_positions": [(int(p), int(c)) for p, c in top_pos],
                }
            return out

        summarized = _finalize_stats(stats)
        items = list(summarized.items())
        items.sort(key=lambda kv: (kv[1]["kept_count"], kv[1]["mean_abs_score"]), reverse=True)

        if top_n is not None and top_n > 0:
            items = items[:top_n]

        txt_path = out_dir / f"{prefix}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            for fid, _st in items:
                f.write(str(int(fid)) + "\n")

        json_path = out_dir / f"{prefix}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({str(fid): st for fid, st in items}, f, indent=2)

        return txt_path, json_path

class PerPromptRecorder:
    def __init__(self, path: Path):
        self.path = path
        self.f = open(path, "w", encoding="utf-8")
        self.n_written = 0

    def write(self, global_idx, rank, text_sha1, n_enc_occ, kept_base_fids, kept_final_fids, kl, chg, extra=None):
        rec = {
            "global_idx": int(global_idx), "rank": int(rank),
            "text_sha1": text_sha1, "n_enc_occ": int(n_enc_occ),
            "kept_base": kept_base_fids, "kept_final": kept_final_fids,
            "kl": float(kl), "pred_changed": None if chg is None else int(chg),
        }
        if extra: rec.update(extra)
        self.f.write(json.dumps(rec) + "\n")
        self.n_written += 1
        if (self.n_written % 50) == 0: self.f.flush()

    def close(self):
        try: self.f.flush()
        finally: self.f.close()


class RelpPerPromptRecorder:
    """Per-prompt JSONL recorder for the multi-K RelP pipeline.

    Records KL, prediction-change, faithfulness, and completeness for each K
    budget on a single prompt, plus the kept feature ids at the largest K.
    """
    def __init__(self, path: Path):
        self.path = path
        self.f = open(path, "w", encoding="utf-8")
        self.n_written = 0

    def write(self, global_idx, rank, text_sha1, n_enc_occ,
              kept_fids_max_k, kl_by_k, chg_by_k,
              faithfulness_by_k=None, completeness_by_k=None, extra=None):
        rec = {
            "global_idx": int(global_idx), "rank": int(rank),
            "text_sha1": text_sha1, "n_enc_occ": int(n_enc_occ),
            "kept_fids_max_k": kept_fids_max_k,
            "kl_by_k": {str(k): float(v) for k, v in kl_by_k.items()},
            "pred_changed_by_k": {str(k): (None if v is None else int(v)) for k, v in chg_by_k.items()},
        }
        if faithfulness_by_k is not None:
            rec["faithfulness_by_k"] = {str(k): float(v) for k, v in faithfulness_by_k.items()}
        if completeness_by_k is not None:
            rec["completeness_by_k"] = {str(k): float(v) for k, v in completeness_by_k.items()}
        if extra: rec.update(extra)
        self.f.write(json.dumps(rec) + "\n")
        self.n_written += 1
        if (self.n_written % 50) == 0: self.f.flush()

    def close(self):
        try: self.f.flush()
        finally: self.f.close()