"""Load string variables from player_config.py for the /api/player-theme endpoint."""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Dict


def load_player_theme() -> Dict[str, str]:
    from human_feedback_app import player_config

    importlib.reload(player_config)
    return _module_strings_as_dict(player_config)


def _module_strings_as_dict(mod: ModuleType) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for name, value in mod.__dict__.items():
        if name.startswith("_"):
            continue
        if isinstance(value, str):
            out[name.lower()] = value
    return out
