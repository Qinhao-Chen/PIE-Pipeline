"""PIE Stage II: generate descriptions from retained feature activation histories."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan", required=True, help="feature-history HF repository, optionally @revision")
    parser.add_argument("--pruned-file", "--pruned_file", dest="pruned_file", required=True,
                        help="pruning JSONL: current per_prompt.jsonl or historical output")
    parser.add_argument("--keep-topk", "--keep_topk", dest="keep_topk", type=int,
                        help="select a budget when pruning results contain several K values")
    parser.add_argument("--llm-model", "--llm_model", dest="llm_model", default="gpt-4o")
    parser.add_argument("--out-file", "--out_file", dest="out_file", default="descriptions.json")
    parser.add_argument("--base-url", "--base_url", dest="base_url")
    parser.add_argument("--openai-api-key", "--openai_api_key", dest="openai_api_key")
    parser.add_argument("--max-concurrent", "--max_concurrent", dest="max_concurrent", type=int, default=20)
    return parser


async def run(args):
    from pie.utils.llm import OpenAIChatClient
    from pie.interpretation.data import HFFeatureStore, load_pruned_features
    from pie.interpretation.prompting import build_explanation_messages, parse_explanation

    feat_ids = load_pruned_features(args.pruned_file, min_freq=1, keep_topk=args.keep_topk)
    if not feat_ids:
        raise ValueError("No retained features found in the pruning input")
    print(f"[PIE] Loaded {len(feat_ids)} features to explain.")
    client = OpenAIChatClient(model=args.llm_model, base_url=args.base_url,
                              api_key=args.openai_api_key, max_concurrent=args.max_concurrent)
    store = HFFeatureStore(args.scan)
    results = {}
    Path(args.out_file).parent.mkdir(parents=True, exist_ok=True)

    async def process_feature(fid):
        try:
            feature_data = await store.load_feature(fid)
            response = await client.generate(build_explanation_messages(feature_data))
            if not response.text:
                print(f"[WARN] {fid}: Empty response generated")
                return None
            explanation = parse_explanation(response.text)
            print(f"[GEN] {fid}: {explanation[:50]}...")
            return str(fid), explanation
        except Exception as error:
            print(f"[ERR] {fid}: {error}")
            return None

    for start in range(0, len(feat_ids), 50):
        batch = await asyncio.gather(*(process_feature(fid) for fid in feat_ids[start:start + 50]))
        results.update(result for result in batch if result is not None)
        with open(args.out_file, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.max_concurrent < 1 or (args.keep_topk is not None and args.keep_topk < 1):
        parser.error("--max-concurrent and --keep-topk must be positive")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
