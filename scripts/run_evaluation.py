#!/usr/bin/env python3
import asyncio
import argparse
import torch
import json
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from circuit_tracer.utils.hf_utils import load_transcoder_from_hub

from pie.utils.llm import OpenAIChatClient
from pie.models.activations import compute_activations
from pie.evaluation.fade import eval_clarity, eval_purity
from pie.utils.common import cantor_unpair

def load_corpus(n=1000,corpus="sentence-transformers/wikipedia-en-sentences"):
    ds = load_dataset(corpus, split="train")
    return [ds[i]["sentence"] for i in range(n)]

async def main():
    parser = argparse.ArgumentParser(description="PIE Stage III: FADE Evaluation")
    parser.add_argument("--model", required=True)
    parser.add_argument("--clt", required=True)
    parser.add_argument("--descriptions", required=True)
    parser.add_argument("--out_file", default="evaluation_results.json")
    parser.add_argument("--llm_model", default="gpt-5-mini")
    parser.add_argument("--base_url", default=None, help="Custom API base URL")
    parser.add_argument("--num_samples", type=int, default=2000, help="Number of corpus samples to base evaluate on")
    parser.add_argument("--corpus", default="sentence-transformers/wikipedia-en-sentences", help="HuggingFace dataset for evaluation texts")
    args = parser.parse_args()

    # 1. Load Models
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[PIE] Loading Subject Model...")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print("[PIE] Loading CLT...")
    clt, _ = load_transcoder_from_hub(args.clt, lazy_decoder=True)
    clt.to(device)

    # 2. Load Descriptions & Data
    with open(args.descriptions, "r") as f:
        descriptions = json.load(f)
    
    wiki_texts = load_corpus(n=args.num_samples, corpus=args.corpus)
    client = OpenAIChatClient(model=args.llm_model, base_url=args.base_url)

    # 3. Pre-Compute Natural Activations
    locs = {}
    for fid_str in descriptions:
        fid = int(fid_str)
        L, idx = cantor_unpair(fid)
        locs[fid] = {"layer": L, "index": idx}
    
    print("[PIE] Computing Natural Activations...")
    nat_acts = compute_activations(model, tokenizer, clt, wiki_texts, locs, device=device)

    # 4. Eval
    results = {}
    for fid, desc in descriptions.items():
        fid = int(fid)
        print(f"[EVAL] Feature {fid}")
        
        # Clarity
        clarity_score, _ = await eval_clarity(client, desc, model, tokenizer, clt, locs[fid], nat_acts[fid])
        
        # Purity & Responsiveness
        purity_ap, resp_score = await eval_purity(client, desc, wiki_texts, nat_acts[fid])
        
        results[fid] = {
            "description": desc,
            "metrics": {
                "clarity": clarity_score,
                "purity": purity_ap,
                "responsiveness": resp_score
            }
        }
        print(f"       Clarity: {clarity_score:.2f} | Purity: {purity_ap:.2f}")

    with open(args.out_file, "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    asyncio.run(main())