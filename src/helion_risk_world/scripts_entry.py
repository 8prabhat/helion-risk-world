"""Console-script entry points declared in pyproject.toml.

These shims execute the canonical ``scripts/*.py`` entry points so there is exactly one runtime path
for CLI behaviour (DRY).
"""
from __future__ import annotations

from pathlib import Path
import runpy
import sys

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = _ROOT / "scripts"


def _run(script_name: str) -> None:
    script_path = _SCRIPTS / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"missing script entrypoint: {script_path}")
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    runpy.run_path(str(script_path), run_name="__main__")


def build_features() -> None:
    _run("build_features.py")


def train() -> None:
    _run("train.py")


def backtest() -> None:
    _run("backtest.py")


def paper_trade() -> None:
    _run("paper_trade.py")
