"""Config loading utilities.

Configs are plain YAML dictionaries. We wrap them in a light attribute-access
object so experiment code can use ``cfg.training.epochs`` while still allowing
``cfg["training"]["epochs"]``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class Config(dict):
    """dict with attribute access, recursively applied to nested dicts."""

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(name) from exc
        if isinstance(value, dict) and not isinstance(value, Config):
            value = Config(value)
            self[name] = value
        return value

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def get_path(self, dotted: str, default: Any = None) -> Any:
        """Fetch a nested value by dotted path, e.g. ``training.epochs``."""
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node


def load_config(path: str | Path) -> Config:
    """Load a YAML config file into a :class:`Config`."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config {path} must be a mapping at the top level.")
    cfg = Config(data)
    cfg["_config_path"] = str(path)
    return cfg
