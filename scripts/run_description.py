#!/usr/bin/env python3
import asyncio
import argparse
import json
import os
from pathlib import Path
from pie.utils.llm import OpenAIChatClient
from pie.interpretation.data import HFFeatureStore, load_pruned_features
from pie.interpretation.prompting import build_explanation_messages, parse_explanation

async def main():
    parser = argparse.ArgumentParser(description="PIE Stage II: Interpretation Generation")
    parser.add_argument("--scan", required=True, help="HF repo (e.g., org/repo@main)")
    parser.add_argument("--pruned_file", required=True, help="Output from pruning stage (jsonl)")
    parser.add_argument("--llm_model", default="gpt-4o")
    parser.add_argument("--out_file", default="descriptions.json")
    parser.add_argument("--base_url", default=None, help="Custom API base URL (e.g. https://api.xty.app/v1)")
    parser.add_argument("--openai_api_key", default=None, help="OpenAI API Key")
    parser.add_argument("--max_concurrent", type=int, default=20)
    
    args = parser.parse_args()

    # 1. Load Pruned Features
    feat_ids = load_pruned_features(args.pruned_file, min_freq=1)
    print(f"[PIE] Loaded {len(feat_ids)} features to explain.")

    # 2. Setup Resources
    client = OpenAIChatClient(
        model=args.llm_model, 
        base_url=args.base_url,
        api_key=args.openai_api_key,
        max_concurrent=args.max_concurrent
    )
    store = HFFeatureStore(args.scan)
    results = {}


    Path(args.out_file).parent.mkdir(parents=True, exist_ok=True)

    # 3. Main Loop
    tasks = []
    
    async def process_feature(fid):
        try:
            feature_data = await store.load_feature(fid)
            msgs = build_explanation_messages(feature_data)
            resp = await client.generate(msgs)
            
            if not resp.text:
                print(f"[WARN] {fid}: Empty response generated")
                return None
                
            explanation = parse_explanation(resp.text)
            print(f"[GEN] {fid}: {explanation[:50]}...")
            return (str(fid), explanation)
        except Exception as e:
            print(f"[ERR] {fid}: {e}")
            return None

    # Run in batches or all at once (client semaphore handles concurrency)
    batch_size = 50
    for i in range(0, len(feat_ids), batch_size):
        batch = feat_ids[i:i+batch_size]
        coros = [process_feature(fid) for fid in batch]
        batch_results = await asyncio.gather(*coros)
        for res in batch_results:
            if res:
                results[res[0]] = res[1]
        with open(args.out_file, "w") as f:
            json.dump(results, f, indent=2)

if __name__ == "__main__":
    asyncio.run(main())