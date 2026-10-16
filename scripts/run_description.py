#!/usr/bin/env python3
"""PIE Stage II: describe the selected feature identities."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pie.interpretation.cli import main

if __name__ == "__main__":
    main()
