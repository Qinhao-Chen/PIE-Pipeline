"""Command dispatch for the existing Prune → Interpret → Evaluate stages."""
from __future__ import annotations

import argparse
import sys


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    commands = {"prune": "Select a feature budget and measure behavioral fidelity",
                "interpret": "Describe retained feature identities",
                "evaluate": "Evaluate descriptions on held-out activations"}
    if argv and argv[0] in commands:
        command, rest = argv[0], argv[1:]
        if command == "prune":
            from pie.behavior.cli import main as run
        elif command == "interpret":
            from pie.interpretation.cli import main as run
        else:
            from pie.evaluation.cli import main as run
        return run(rest)
    parser = argparse.ArgumentParser(description="PIE: Prune, Interpret, Evaluate")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command, help_text in commands.items():
        subparsers.add_parser(command, help=help_text)
    parser.parse_args(argv)
