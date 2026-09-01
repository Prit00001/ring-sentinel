"""Configuration loading.

Nothing in src/ hardcodes a threshold, a seed, a path, or a rupee amount.
Everything comes from config/*.yaml so a reviewer can change one file and
re-run `make repro`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# Loaded once, on first import of this module (every entry point — serve,
# pipeline, Makefile scripts — imports config before touching GROQ_API_KEY).
# Walks up from the current working directory to find .env; a real
# environment variable already set always wins (override=False).
load_dotenv()


def repo_root() -> Path:
    """Locate the repo root by walking up from this file until config/ appears."""
    env = os.environ.get("RING_SENTINEL_ROOT")
    if env:
        return Path(env).resolve()
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "config" / "base.yaml").exists():
            return parent
    raise RuntimeError("Could not locate repo root (no config/base.yaml found)")


def _load_yaml(name: str) -> dict[str, Any]:
    path = repo_root() / "config" / name
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")
    with path.open() as fh:
        return yaml.safe_load(fh) or {}


@dataclass(frozen=True)
class Config:
    base: dict[str, Any]
    costs: dict[str, Any]
    features: dict[str, Any]
    llm: dict[str, Any]
    root: Path

    def path(self, key: str) -> Path:
        """Resolve a path from config.base.paths, relative to the repo root."""
        rel = self.base["paths"][key]
        p = self.root / rel
        return p

    def ensure_dirs(self) -> None:
        for key in ("interim", "processed", "artifacts", "reports", "figures"):
            self.path(key).mkdir(parents=True, exist_ok=True)

    @property
    def seed(self) -> int:
        return int(self.base["seed"])


@lru_cache(maxsize=1)
def load_config() -> Config:
    root = repo_root()
    cfg = Config(
        base=_load_yaml("base.yaml"),
        costs=_load_yaml("costs.yaml"),
        features=_load_yaml("features.yaml"),
        llm=_load_yaml("llm.yaml"),
        root=root,
    )
    return cfg
