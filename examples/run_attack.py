"""Canonical source-tree entry: python examples/run_attack.py [Hydra overrides]."""
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.experiment import main  # noqa: E402


if __name__ == "__main__":
    main()
