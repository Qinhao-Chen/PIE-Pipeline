#!/usr/bin/env python3
"""Run a behavioral budget sweep using the pruning CLI."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pie.behavior.cli import build_parser, k_values, main, positive

if __name__ == "__main__":
    main()
