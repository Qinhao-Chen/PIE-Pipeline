#!/usr/bin/env python3
"""Compatibility entry point; the maintained script is scripts/run_evaluation.py."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pie.evaluation.cli import main

if __name__ == "__main__":
    main()
