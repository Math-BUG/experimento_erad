"""Thin wrapper so the experiment can be run as: python run_experiment.py ..."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from erad_pipeline.run_experiment import main


if __name__ == "__main__":
    main(sys.argv[1:])
