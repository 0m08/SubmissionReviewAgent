"""Load string variables from player_styles.yaml and player_config.py for the player style endpoints."""

from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType
from typing import Any, Dict
import yaml

_YAML_PATH = Path(__file__).resolve().parent / "player_styles.yaml"


def load_styles_schema() -> Dict[str, Any]:
    """Load the full player styles schema from player_styles.yaml."""
    if not _YAML_PATH.is_file():
        return {"primitives": {}}
    with open(_YAML_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {"primitives": {}}


def load_player_theme() -> Dict[str, str]:
    """Load player style variables, combining YAML defaults with player_config.py overrides."""
    out: Dict[str, str] = {}

    # 1. Base values from player_styles.yaml schema defaults
    schema = load_styles_schema()
    for _, p_data in schema.get("primitives", {}).items():
        if not isinstance(p_data, dict):
            continue
        for _, s_data in p_data.get("sections", {}).items():
            if not isinstance(s_data, dict):
                continue
            for prop_key, prop_data in s_data.get("properties", {}).items():
                if isinstance(prop_data, dict) and "default" in prop_data:
                    out[str(prop_key).lower()] = str(prop_data["default"])

    # 1b. Base values from animations schema defaults
    for anim_key, anim_data in schema.get("animations", {}).items():
        if isinstance(anim_data, dict) and "default" in anim_data:
            out[str(anim_key).lower()] = str(anim_data["default"])

    # 2. Overrides from player_config.py if present
    try:
        from human_feedback_app import player_config

        importlib.reload(player_config)
        overrides = _module_strings_as_dict(player_config)
        out.update(overrides)
    except Exception:
        pass

    return out


def _module_strings_as_dict(mod: ModuleType) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for name, value in mod.__dict__.items():
        if name.startswith("_"):
            continue
        if isinstance(value, str):
            out[name.lower()] = value
    return out

