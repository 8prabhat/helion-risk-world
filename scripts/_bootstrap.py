"""Small helpers for direct script execution."""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_src_path() -> Path:
    src = Path(__file__).resolve().parents[1] / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return src


__all__ = ["ensure_src_path"]
