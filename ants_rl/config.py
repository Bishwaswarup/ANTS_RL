"""YAML configs with dotted-key overrides (``ppo.total_timesteps=2e5``)."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


def load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def deep_merge(base: dict, other: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in other.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def set_dotted(cfg: dict, key: str, value) -> None:
    node = cfg
    parts = key.split(".")
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = value


def get_dotted(cfg: dict, key: str, default=None):
    node = cfg
    for p in key.split("."):
        if not isinstance(node, dict) or p not in node:
            return default
        node = node[p]
    return node


def parse_overrides(items: list[str] | None) -> dict:
    out: dict = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got {item!r}")
        k, v = item.split("=", 1)
        val = yaml.safe_load(v)
        if isinstance(val, str):  # allow 1e6 / 2e5 style numbers
            try:
                f = float(val)
                val = int(f) if f.is_integer() else f
            except ValueError:
                pass
        set_dotted(out, k, val)
    return out


def load_config(path: str | Path | None = None, overrides: list[str] | dict | None = None) -> dict:
    cfg = load_yaml(DEFAULT_CONFIG)
    if path is not None and Path(path).resolve() != DEFAULT_CONFIG:
        cfg = deep_merge(cfg, load_yaml(path))
    if isinstance(overrides, dict):
        cfg = deep_merge(cfg, overrides)
    elif overrides:
        cfg = deep_merge(cfg, parse_overrides(overrides))
    return cfg
