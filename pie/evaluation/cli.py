"""PIE Stage III: assess feature descriptions with FADE-style metrics."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="gemma, llama, or the corresponding full model ID")
    parser.add_argument("--clt", help="optional CLT ID; must match the selected model profile")
    parser.add_argument("--descriptions", required=True)
    parser.add_argument("--out-file", "--out_file", dest="out_file", default="evaluation_results.json")
    parser.add_argument("--llm-model", "--llm_model", dest="llm_model", default="gpt-5-mini")
    parser.add_argument("--base-url", "--base_url", dest="base_url")
    parser.add_argument("--num-samples", "--num_samples", dest="num_samples", type=int, default=2000)
    parser.add_argument("--corpus", default="sentence-transformers/wikipedia-en-sentences")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--allow-download", action="store_true")
    return parser


async def run(args):
    vendor = Path(__file__).resolve().parents[2] / "vendor" / "circuit-tracer"
    if vendor.is_dir():
        sys.path.insert(0, str(vendor))
    from datasets import load_dataset
    from pie.behavior.protocol import PROFILES
    from pie.behavior.runner import load_model
    from pie.models.activations import compute_activations
    from pie.evaluation.fade import eval_clarity, eval_purity
    from pie.utils.common import cantor_unpair
    from pie.utils.llm import OpenAIChatClient

    model_name = args.model if args.model in PROFILES else next(
        (name for name, profile in PROFILES.items() if profile["model"] == args.model), None)
    if model_name is None:
        raise ValueError("Use gemma or llama, or a full model ID from configs/models.json")
    profile = PROFILES[model_name]
    if args.clt and args.clt not in (profile["clt"], profile["clt"] + "@" + profile["clt_commit"]):
        raise ValueError("The CLT must match the selected model profile")
    descriptions = json.loads(Path(args.descriptions).read_text(encoding="utf-8"))
    if not isinstance(descriptions, dict) or not descriptions or any(not isinstance(text, str) or not text.strip() for text in descriptions.values()):
        raise ValueError("Descriptions must be a nonempty feature-ID-to-text mapping")
    locs = {}
    for identity in descriptions:
        if not identity.isdecimal():
            raise ValueError("Feature IDs must be nonnegative integers")
        layer, index = cantor_unpair(int(identity))
        locs[int(identity)] = {"layer": layer, "index": index}
    client = OpenAIChatClient(model=args.llm_model, base_url=args.base_url)
    model, clt = load_model(model_name, device=args.device, allow_download=args.allow_download)
    model.eval()
    clt.eval()
    dataset = load_dataset(args.corpus, split="train")
    texts = [dataset[i]["sentence"] for i in range(min(args.num_samples, len(dataset)))]
    natural = compute_activations(model, model.tokenizer, clt, texts, locs)
    results = {}
    errors = {}
    output = Path(args.out_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    for identity, description in descriptions.items():
        fid = int(identity)
        try:
            clarity, _ = await eval_clarity(client, description, model, model.tokenizer, clt, locs[fid], natural[fid])
            purity, responsiveness = await eval_purity(client, description, texts, natural[fid])
            results[identity] = {"description": description, "metrics": {
                "clarity": clarity, "purity": purity, "responsiveness": responsiveness}}
            print(f"[EVAL] Feature {identity}: {results[identity]['metrics']}")
        except Exception as error:
            errors[identity] = str(error)
            results[identity] = {"description": description, "error": str(error)}
            print(f"[ERR] Feature {identity}: {error}", file=sys.stderr)
        output.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if errors:
        raise RuntimeError(f"{len(errors)} feature evaluations failed; see {output}")


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.num_samples < 1:
        parser.error("--num-samples must be positive")
    try:
        asyncio.run(run(args))
    except (ValueError, RuntimeError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
