"""Validation helpers for per-scene Player overrides stored in sheet JSON."""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any, Dict, Set, Tuple

from human_feedback_app.player_config_loader import load_styles_schema


@lru_cache(maxsize=1)
def player_override_schema_keys() -> Tuple[Set[str], Set[str]]:
    """Return style keys and animation treatments supported by the Player schema."""
    style_keys: Set[str] = set()
    treatment_values: Set[str] = set()
    schema = load_styles_schema()

    for primitive in (schema.get("primitives") or {}).values():
        if not isinstance(primitive, dict):
            continue
        for section in (primitive.get("sections") or {}).values():
            if not isinstance(section, dict):
                continue
            style_keys.update(str(key) for key in (section.get("properties") or {}).keys())

    for animation in (schema.get("animations") or {}).values():
        if not isinstance(animation, dict):
            continue
        for option in animation.get("options") or []:
            if isinstance(option, dict) and option.get("value") is not None:
                treatment_values.add(str(option["value"]).strip())

    return style_keys, treatment_values


def clean_player_scene_override(raw: Any) -> Dict[str, Any]:
    """Normalize one scene override and discard unsupported keys or values."""
    if not isinstance(raw, dict):
        return {}

    style_keys, treatment_values = player_override_schema_keys()
    cleaned: Dict[str, Any] = {}

    if isinstance(raw.get("animationEnabled"), bool):
        cleaned["animationEnabled"] = raw["animationEnabled"]

    treatment = str(raw.get("animationTreatment") or "").strip()
    if treatment and treatment in treatment_values:
        cleaned["animationTreatment"] = treatment

    styles: Dict[str, str] = {}
    raw_styles = raw.get("styles")
    if isinstance(raw_styles, dict):
        for key, value in raw_styles.items():
            key_text = str(key).strip()
            value_text = "" if value is None else str(value).strip()
            if key_text in style_keys and value_text:
                styles[key_text] = value_text[:500]
    if styles:
        cleaned["styles"] = styles

    return cleaned


def parse_player_scene_overrides(raw: Any) -> Dict[str, Dict[str, Any]]:
    """Parse and validate the per-scene override map stored in a sheet cell."""
    if not raw or not str(raw).strip() or str(raw).strip().lower() == "nan":
        return {}
    try:
        data = json.loads(str(raw))
    except (TypeError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}

    out: Dict[str, Dict[str, Any]] = {}
    for scene_id, value in data.items():
        scene_key = str(scene_id).strip()
        if not scene_key:
            continue
        cleaned = clean_player_scene_override(value)
        if cleaned:
            out[scene_key] = cleaned
    return out
