"""Answer-free task prefixes, pinned model profiles, and tokenizer boundary checks."""
from __future__ import annotations
import hashlib
TASKS = ("ioi", "docstring")
MAX_TOKENS = 128
K_VALUES = (50, 100, 200, 400, 800)
PROFILES = {
    "gemma": {"model": "google/gemma-2-2b", "model_commit": "c5ebcd40d208330abc697524c919956e692655cf", "clt": "mntss/clt-gemma-2-2b-426k", "clt_commit": "b1e9ab376d07c90d780bf20b5fb1a0c89bd0f5e7"},
    "llama": {"model": "meta-llama/Llama-3.2-1B", "model_commit": "4e20de362430cd3b72f300e6b0f18e50e7166e08", "clt": "mntss/clt-llama-3.2-1b-524k", "clt_commit": "d6ab3ffa8184c9894b2ffb35093c50c098c0f223"},
}


def text_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def prepare_prompt(row: dict, task: str, index: int) -> dict:
    """Remove the verified IOI answer and its leading whitespace as one suffix.

    Retaining the space would create an extra input token instead of predicting
    the original space-prefixed answer token. DocString already stops at :param.
    """
    if task not in TASKS:
        raise ValueError(f"unknown task: {task}")
    answer_clean = row["io_clean"] if task == "ioi" else row["target_clean"]
    answer_corr = row["io_corr"] if task == "ioi" else row["target_corr"]
    result = {
        "id": str(row["id"]), "row_index": index, "task": task,
        "answer_clean": answer_clean, "answer_corr_label": answer_corr,
        "protocol": "pre_answer_prefix_first_token_v1",
    }
    for side in ("clean", "corr"):
        text = row[f"text_{side}"]
        if task == "ioi":
            if not answer_clean or not text.endswith(answer_clean):
                raise ValueError(f"IOI row {index} {side} lacks literal io_clean suffix")
            boundary = len(text) - len(answer_clean)
            if boundary == 0 or not text[boundary - 1].isspace():
                raise ValueError(f"IOI row {index} {side} answer has no whitespace boundary")
            prefix = text[:boundary].rstrip()
            continuation = text[len(prefix):]
            removed = continuation
        else:
            if not text.endswith(":param"):
                raise ValueError(f"DocString row {index} {side} does not end at :param")
            prefix = text
            continuation = " " + row[f"target_{side}"]
            removed = ""
        result[f"text_{side}"] = prefix
        result[f"source_{side}_sha256"] = text_digest(text)
        result[f"prefix_{side}_sha256"] = text_digest(prefix)
        result[f"removed_{side}_suffix"] = removed
        result[f"continuation_{side}"] = continuation
    return result


def tokenize_prefix(tokenizer, prefix: str, continuation: str) -> dict:
    """Match ReplacementModel.ensure_tokenized and verify the token boundary."""
    def tokenize(text):
        ids = tokenizer.encode(text, add_special_tokens=True)
        if not ids:
            raise ValueError("empty tokenized input")
        if ids[0] not in tokenizer.all_special_ids:
            bos = next((value for value in (
                tokenizer.bos_token_id, tokenizer.pad_token_id,
                tokenizer.eos_token_id, *tokenizer.all_special_ids,
            ) if value is not None and value != 0), None)
            if bos is None:
                raise ValueError("no usable special token for ReplacementModel BOS policy")
            ids.insert(0, bos)
        return ids

    ids = tokenize(prefix)
    completed_ids = tokenize(prefix + continuation)
    if completed_ids[:len(ids)] != ids or len(completed_ids) <= len(ids):
        raise ValueError("answer boundary does not coincide with a tokenizer boundary")
    if not 8 <= len(ids) <= MAX_TOKENS:
        raise ValueError(f"{len(ids)} input tokens; refuse truncation that loses the answer boundary")
    answer_ids = completed_ids[len(ids):]
    return {"input_ids": ids, "answer_token_ids": answer_ids,
            "token_count": len(ids), "prediction_position": len(ids) - 1}
