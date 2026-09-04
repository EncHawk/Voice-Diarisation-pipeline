"""Config loader with defaults from config/default.yaml."""
from __future__ import annotations

import pathlib

import yaml

DEFAULT_CONFIG_PATH = pathlib.Path(__file__).parent.parent.parent / "model_config" / "default.yaml"


def load_config(path: str | pathlib.Path | None = None) -> dict:
    cfg_path = pathlib.Path(path) if path else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        return {}
    with open(cfg_path) as f:
        return yaml.safe_load(f) or {}


def get(cfg: dict, dotted: str, default=None):
    cur = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur
