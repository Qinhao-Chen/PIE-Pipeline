#!/usr/bin/env python3
"""Select a budget of CLT occurrences and measure behavioral fidelity."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / "vendor" / "circuit-tracer"
if VENDOR.is_dir():
    sys.path.insert(0, str(VENDOR))


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def k_values(value):
    try:
        values = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("use a comma-separated list of integers") from error
    if not values or len(set(values)) != len(values) or any(k < 1 for k in values):
        raise argparse.ArgumentTypeError("K values must be distinct positive integers")
    return values


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("fap", "lastlse", "fap-swap", "lastlse-swap", "synergy",
                        "activation", "clt-relp", "factp", "random", "native-replay"), default="fap")
    parser.add_argument("--model", choices=("gemma", "llama"), required=True)
    parser.add_argument("--task", choices=("ioi", "docstring"), required=True)
    parser.add_argument("--data-file", type=Path, help="local task pairs; default: data/tasks/{task}_pairs.jsonl (not bundled)")
    parser.add_argument("--output-dir", type=Path, required=True, help="new output directory; existing results are preserved")
    parser.add_argument("--num-samples", type=positive, default=None, help="number of consecutive pairs; default: all remaining")
    parser.add_argument("--start-index", type=int, default=0, help="zero-based source file row")
    parser.add_argument("--keep-topk", type=k_values, default=(50, 100, 200, 400, 800), metavar="K1,K2,...")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--allow-download", action="store_true", help="allow fetching the pinned model/CLT snapshots")
    parser.add_argument("--swap-candidate-cap", type=positive, default=32, help="unique extra candidate forwards; paper setting: 32")
    parser.add_argument("--swap-rounds", type=positive, default=2, help="paper setting: 2")
    parser.add_argument("--lambda-syn", type=float, default=3.0)
    parser.add_argument("--boundary-percent", type=float, default=25.0)
    parser.add_argument("--boundary-cap", type=positive, default=32)
    parser.add_argument("--partners-per-refresh", type=positive, default=32)
    parser.add_argument("--syn-partners-per-candidate", type=positive, default=8)
    parser.add_argument("--native-source-entry", type=Path,
                        help="JSON entry for a saved graph/record/masks/KL-anchor bundle; required for native-replay")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.start_index < 0:
        parser.error("--start-index cannot be negative")
    if not 0 <= args.boundary_percent <= 100:
        parser.error("--boundary-percent must be between 0 and 100")
    if args.lambda_syn < 0:
        parser.error("--lambda-syn cannot be negative")
    if args.method == "native-replay" and not args.native_source_entry:
        parser.error("--native-source-entry is required for native-replay")
    if args.data_file is None:
        args.data_file = ROOT / "data" / "tasks" / f"{args.task}_pairs.jsonl"
    if args.method != "native-replay" and not args.data_file.is_file():
        parser.error("Local task data is required. Generate IOI pairs with scripts/ioi.py or supply --data-file.")
    from pie.behavior.runner import execute
    result = execute(args)
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
