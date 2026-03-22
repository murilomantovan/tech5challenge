"""Compat shim for legacy package imports."""

import sys
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.passos_magicos_dt import __version__

__all__ = ["__version__"]
