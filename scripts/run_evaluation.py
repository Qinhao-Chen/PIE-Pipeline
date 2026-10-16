#!/usr/bin/env python3
"""PIE Stage III: evaluate descriptions against held-out CLT activations."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor" / "circuit-tracer"))
from pie.evaluation.cli import main

if __name__ == "__main__":
    main()
