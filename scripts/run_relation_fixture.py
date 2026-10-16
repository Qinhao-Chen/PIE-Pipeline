#!/usr/bin/env python3
"""Exercise relation questions and scoring using synthetic scripted answers."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pie.relations.fixture import main

if __name__ == "__main__":
    main()
