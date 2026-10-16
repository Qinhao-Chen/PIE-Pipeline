#!/usr/bin/env python3
"""PIE Stage I: select CLT occurrences and measure behavioral fidelity."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pie.behavior.cli import main

if __name__ == "__main__":
    main()
