"""Utility helpers for resolving project paths shared across modules."""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Optional

import yaml

# Resolve once so every module can share the same project root reference
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_config_from_disk() -> dict:
    config_path = PROJECT_ROOT / "config.yaml"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as config_file:
            return yaml.safe_load(config_file) or {}
    return {}


@lru_cache(maxsize=1)
def load_project_config() -> dict:
    """Return the parsed project config (cached)."""
    return _load_config_from_disk()


def _candidate_hatcat_paths(config: Optional[dict]) -> list[Path]:
    candidates = []

    env_path = os.environ.get("HATCAT_ROOT")
    if env_path:
        candidates.append(Path(env_path))

    if config:
        config_value = config.get("hatcat_path")
        if config_value:
            candidates.append(Path(config_value))

    candidates.append(PROJECT_ROOT.parent / "HatCat")
    return candidates


def resolve_hatcat_root(config: Optional[dict] = None) -> Optional[Path]:
    """Best-effort resolution of the HatCat repo path."""
    config = config or load_project_config()

    last_candidate: Optional[Path] = None
    for candidate in _candidate_hatcat_paths(config):
        expanded = candidate.expanduser()
        try:
            resolved = expanded.resolve()
        except FileNotFoundError:
            resolved = expanded
        last_candidate = resolved
        if resolved.exists():
            return resolved

    return last_candidate


def ensure_hatcat_on_sys_path(config: Optional[dict] = None) -> Optional[Path]:
    """Add HatCat to sys.path when available and return the resolved path."""
    hatcat_root = resolve_hatcat_root(config)
    if hatcat_root and hatcat_root.exists():
        if str(hatcat_root) not in sys.path:
            sys.path.insert(0, str(hatcat_root))
    return hatcat_root
