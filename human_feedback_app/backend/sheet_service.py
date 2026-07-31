"""Sheet loading and UI payload transformation."""

from __future__ import annotations

import json
import pandas as pd
import re
import threading
import uuid
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple
import html

from services.sheets_service import save_to_sheet, get_sheet_data_and_df
from utils.decorator_helpers import try_n_times

from human_feedback_app.backend.constants import (
    ACTION_APPROVE,
    ACTION_NONE,
    ACTION_REJECT_AI,
    ACTION_REJECT_ALL,
    ACTION_REJECT_DRIVE_HVAC,
    AI_NO_FEEDBACK_MARKER,
    DEFAULT_REJECT_FEEDBACK,
    DEFAULT_WORKSHEET,
    FINAL_GRAPHICS_COLUMN,
    HUMAN_FEEDBACK_COLUMN,
    HUMAN_FEEDBACK_STATUS_COLUMN,
    HUMAN_FEEDBACK_TRACKING_COLUMN,
    HUMAN_REVIEW_ACTIONS_COLUMN,
    MODE_TO_ACTION,
    SEGMENTATION_FEEDBACK_COLUMN,
    SEGMENTATION_PLAN_COLUMN,
    LAYOUT_FEEDBACK_COLUMN,
    LAYOUT_PLAN_COLUMN,
)
from human_feedback_app.backend.sessions import UserSession

from agents.graphics_definition_v2.review_agent.human_feedback_based_review_and_revise import (
    _format_human_feedback_revision_tracking,
    _parse_tracking_column,
)
from agents.graphics_definition_v2.slideshow_manifest.slideshow_manifest import (
    apply_url_replacements_to_slideshow_manifest_inner_xml,
    generate_slideshow_manifest_for_row,
    parse_when_vo_assigned_pairs,
    urls_match_for_graphics_assignment,
)
from graphics_definition_v2_slideshow import (
    _apply_asset_overrides_to_raw,
    build_slides_from_df,
    detect_asset_type,
    detect_current_round,
    get_round_column_name,
    parse_graphics_definition,
    safe_str,
)
from agents.graphics_definition_v2.aggregation_agent.aggregation_agent import (
    parse_urls_from_image_pool,
    parse_urls_from_video_pool_filtered,
)


# This app pins the human-feedback round to a single set of "_1" columns and never advances rounds. Each visual's lifecycle (action, feedback, original -> after_revision -> after_regen_* history) lives inside the keyed JSON of the actions/tracking cells, so we never spawn _2/_3 column sets.
ROUND_INDEX = 0

# Serializes the read-modify-write of the worksheet. Because save_to_sheet writes the ENTIRE sheet, every writer must reload the latest data inside this lock and patch only the cells it owns; otherwise concurrent writers clobber each other.
_SHEET_WRITE_LOCK = threading.Lock()
_MANIFEST_SYNC_LOCK = threading.Lock()
_MANIFEST_SYNC_STATUS_LOCK = threading.Lock()
_MANIFEST_SYNC_RUNNING_BY_SESSION: Dict[str, bool] = {}
MANIFEST_SYNC_MAX_REGEN_ATTEMPTS = 3
MANIFEST_SYNC_MAX_WORKERS = 15
MANIFEST_SYNC_LOG_PREFIX = "[manifest_sync]"


def _manifest_sync_row_label(row: Any, row_index: int) -> str:
    """Human-readable row label for manifest sync terminal logs."""
    title = ""
    if isinstance(row, dict):
        title = safe_str(row.get("Slide Chunk Title", "")).strip()
    else:
        title = safe_str(row.get("Slide Chunk Title", "")).strip() if hasattr(row, "get") else ""
    if not title or title == "nan":
        title = f"Slide {row_index + 1}"
    return f"row {row_index + 1} ({title})"


def _ensure_sheet(session: UserSession):
    if session.sheet is None:
        session.sheet = session.gc.open_by_url(session.sheet_link)
    return session.sheet


def _pick_column(df, name):
    """
    Pick a column from a DataFrame by name.

    :param df: The DataFrame to search.
    :param name: The name of the column to pick.
    :return: The name of the picked column.
    """
    target = name.strip().lower()
    for col in df.columns:
        if str(col).strip().lower() == target:
            return col
    return name


def mutate_row_cells(session, row_index, mutate):
    """
    Reload the latest worksheet under the global write lock, run `mutate(df)` (which must patch only this row's owned cells), then persist.

    :param session: The session object.
    :param row_index: The index of the row to mutate.
    :param mutate: The function to mutate the row.
    """
    with _SHEET_WRITE_LOCK:
        sheet = _ensure_sheet(session)
        ws, latest_df = get_sheet_data_and_df(sheet, session.worksheet_name or DEFAULT_WORKSHEET)
        latest_df = _sanitize_df(latest_df)
        mutate(latest_df)
        save_to_sheet(ws, latest_df)


def save_row_cells(session, row_index, cell_updates):
    """
    Persist precomputed cell values for a single row, preserving every other cell in the worksheet.

    :param session: The session object.
    :param row_index: The index of the row to save.
    :param cell_updates: The cell updates to save.
    """
    if not cell_updates:
        return

    def mutate(df):
        for col, val in cell_updates.items():
            if col not in df.columns:
                df[col] = ""
            df.at[row_index, col] = "" if val is None else str(val)

    mutate_row_cells(session, row_index, mutate)


def extract_asset_url(raw_def, segment_index, step_index):
    """
    Extract the asset URL for one (segment, step) inside a final_graphics_definition cell.

    :param raw_def: The raw final_graphics_definition cell value.
    :param segment_index: The index of the segment.
    :param step_index: The index of the step.
    :return: The asset URL.
    """
    helpers = _import_slideshow_helpers()
    parse_graphics_definition = helpers["parse_graphics_definition"]
    safe_str = helpers["safe_str"]
    for seg in parse_graphics_definition(raw_def or ""):
        if seg.get("segment_index") == segment_index:
            for step in seg.get("steps") or []:
                if step.get("step_index") == step_index:
                    return safe_str(step.get("asset", "")).strip()
    return ""


def _effective_url_from_tracking_entry(entry: Optional[Dict[str, Any]]) -> str:
    """Return the latest assigned URL recorded in a tracking entry."""
    if not isinstance(entry, dict):
        return ""
    for key in ("after_regen_2", "after_regen_1", "after_revision", "manually_selected", "original"):
        val = entry.get(key)
        if val and str(val).strip():
            return str(val).strip()
    return ""


def _collect_manifest_old_url_candidates(
    current_fgd_url: str,
    tracking_entry: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Build ordered old-URL candidates for patching slideshow_manifest (newest first)."""
    seen = set()
    candidates: List[str] = []
    for url in (
        (current_fgd_url or "").strip(),
        _effective_url_from_tracking_entry(tracking_entry),
        str((tracking_entry or {}).get("original") or "").strip(),
    ):
        if not url or url in seen:
            continue
        seen.add(url)
        candidates.append(url)
    return candidates


def _patch_slideshow_manifest_urls(
    df,
    row_index: int,
    new_url: str,
    old_url_candidates: List[str],
) -> bool:
    """Replace the first matching old URL in slideshow_manifest with new_url."""
    new_url = (new_url or "").strip()
    if not new_url or "slideshow_manifest" not in df.columns:
        return False
    manifest = safe_str(df.at[row_index, "slideshow_manifest"])
    if not manifest or manifest == "nan" or manifest.startswith("ERROR:"):
        return False
    for old_url in old_url_candidates:
        if not old_url or old_url == new_url:
            continue
        try:
            updated_manifest, applied = apply_url_replacements_to_slideshow_manifest_inner_xml(
                manifest, [(old_url, new_url)]
            )
            if applied and updated_manifest != manifest:
                df.at[row_index, "slideshow_manifest"] = updated_manifest
                return True
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[human_feedback] manifest patch failed for row {row_index}: {exc}")
    return False


def merge_visual_revision(
    session: UserSession,
    *,
    row_index: int,
    segment_index: int,
    step_index: int,
    visual_id: str,
    final_url: str,
    original_url: str,
    tracking_entry: Optional[Dict[str, Any]],
    status_value: str = "",
) -> None:
    """Merge ONE visual's revision result into the latest sheet, touching only
    that visual's slice of each shared cell.

    This is what makes within-slide parallelism safe: two revisions on different
    visuals of the same row each call this with their own (segment, step) /
    visual_id, so they update disjoint parts of final_graphics_definition,
    tracking, and the manifest. The (fast) merge is serialized by the global
    write lock; the slow revision work already ran in parallel beforehand.
    """
    helpers = _import_slideshow_helpers()
    parse_graphics_definition = helpers["parse_graphics_definition"]
    safe_str = helpers["safe_str"]
    get_round_column_name = helpers["get_round_column_name"]

    tracking_col = get_round_column_name(HUMAN_FEEDBACK_TRACKING_COLUMN, ROUND_INDEX)
    status_col = get_round_column_name(HUMAN_FEEDBACK_STATUS_COLUMN, ROUND_INDEX)

    def mutate(df) -> None:
        _invalidate_manifest_repair_cache(session, row_index)
        final_col = _pick_column(df, FINAL_GRAPHICS_COLUMN)
        current_fgd_url = ""
        prior_tracking_entry = None
        if tracking_col in df.columns:
            prior_tracking_entry = _parse_tracking(safe_str(df.at[row_index, tracking_col])).get(visual_id)
        if final_col in df.columns:
            current_fgd_url = extract_asset_url(safe_str(df.at[row_index, final_col]), segment_index, step_index)

        # 1) final_graphics_definition: surgical override of just this visual.
        if final_url and final_col in df.columns:
            raw_def = safe_str(df.at[row_index, final_col])
            segments = parse_graphics_definition(raw_def)
            updated = _apply_asset_overrides_to_raw(
                raw_def, segments, {(segment_index, step_index): final_url}
            )
            df.at[row_index, final_col] = updated

        # 2) tracking: merge only this visual_id's entry.
        if tracking_entry is not None:
            if tracking_col not in df.columns:
                df[tracking_col] = ""
            tmap = _parse_tracking(safe_str(df.at[row_index, tracking_col]))
            tmap[visual_id] = tracking_entry
            df.at[row_index, tracking_col] = _format_human_feedback_revision_tracking(tmap)

        # 3) slideshow_manifest: replace the URL currently in use (not the first original).
        if final_url and final_url != current_fgd_url:
            old_candidates = _collect_manifest_old_url_candidates(current_fgd_url, prior_tracking_entry)
            if not _patch_slideshow_manifest_urls(df, row_index, final_url, old_candidates):
                print(
                    f"[human_feedback] manifest patch: no slot matched for row {row_index} "
                    f"visual {visual_id} (tried {len(old_candidates)} candidate URL(s))"
                )

        # 4) status: row-level, last-writer-wins (cosmetic; UI derives per-visual
        #    state from actions + tracking, not from this).
        if status_value:
            if status_col not in df.columns:
                df[status_col] = ""
            df.at[row_index, status_col] = status_value

        ensure_manifest_sync_for_row(
            session,
            df,
            row_index,
            reason="merge_visual_revision",
            force_regenerate=False,
        )

    mutate_row_cells(session, row_index, mutate)


def _first_http_url(text: str) -> str:
    if not text:
        return ""
    match = re.search(r"https?://[^\s)>\"]+", str(text).strip())
    return match.group(0).rstrip(".,);\"'") if match else str(text).strip()


def _import_slideshow_helpers():
    return {
        "build_slides_from_df": build_slides_from_df,
        "detect_asset_type": detect_asset_type,
        "detect_current_round": detect_current_round,
        "get_round_column_name": get_round_column_name,
        "parse_graphics_definition": parse_graphics_definition,
        "safe_str": safe_str,
    }


def _column_map(df) -> Dict[str, str]:
    cols = {c.lower(): c for c in df.columns}

    def pick(*names):
        for name in names:
            if name.lower() in cols:
                return cols[name.lower()]
        return names[0]

    return {
        "topic": pick("Topic"),
        "subtopic": pick("Subtopic"),
        "slide_title": pick("Slide Chunk Title", "Slide Title"),
        "slide_chunk": pick("Slide Chunk"),
        "final_def": pick(FINAL_GRAPHICS_COLUMN),
    }


def _infer_source(asset_url: str, asset_type: str) -> str:
    url = (asset_url or "").lower()
    if "drive.google.com" in url or "docs.google.com" in url:
        return "drive"
    if "youtube.com" in url or "youtu.be" in url:
        return "hvac_yt" if asset_type == "video" else "other_yt"
    if url.startswith("http"):
        return "web"
    return "drive"


def _label_from_asset(asset_url: str, asset_type: str) -> str:
    if not asset_url:
        return "no_asset.png"
    tail = asset_url.rstrip("/").split("/")[-1][:48]
    if asset_type == "video":
        return tail or "video_clip.mp4"
    return tail or "visual.png"


def _normalize_asset_url(url: str) -> str:
    return (url or "").strip().rstrip("/").lower()


def _alternatives_for_segment(
    row,
    *,
    segment_index: int,
    asset_type: str,
    exclude_url: str = "",
) -> List[Dict[str, Any]]:
    """Parse image_pool / video_pool_filtered candidates for one segment."""
    helpers = _import_slideshow_helpers()
    safe_str = helpers["safe_str"]

    exclude = _normalize_asset_url(exclude_url)
    seen: set[str] = set()
    out: List[Dict[str, Any]] = []

    def add_candidate(title: str, url: str, atype: str, duration: str = "") -> None:
        url = (url or "").strip()
        if not url:
            return
        dedupe_key = _normalize_asset_url(url)
        if dedupe_key in seen:
            return
        if exclude and dedupe_key == exclude:
            return
        seen.add(dedupe_key)
        out.append(
            {
                "title": (title or _label_from_asset(url, atype)).strip(),
                "url": url,
                "type": atype,
                "source": _infer_source(url, atype),
                "duration": (duration or "").strip(),
            }
        )

    # 1. Add all candidates from the image pool
    image_pool_text = safe_str(row.get("image_pool", "")).strip()
    if image_pool_text and image_pool_text != "nan":
        for item in parse_urls_from_image_pool(image_pool_text, segment_index):
            add_candidate(
                str(item.get("title") or ""),
                str(item.get("url") or ""),
                "image",
            )

    # 2. Add all candidates from the video pool
    video_pool_text = safe_str(row.get("video_pool_filtered", "")).strip()
    if video_pool_text and video_pool_text != "nan":
        for item in parse_urls_from_video_pool_filtered(video_pool_text, segment_index):
            meta = item.get("metadata") or {}
            add_candidate(
                str(meta.get("title") or "Video clip"),
                str(item.get("url") or ""),
                "video",
                str(meta.get("duration") or ""),
            )

    return out


def _status_for_visual(
    visual_id: str,
    actions_map: Dict[str, Any],
    tracking_parsed: Dict[str, Any],
) -> str:
    action = (actions_map.get(visual_id) or {}).get("action", ACTION_NONE)
    if action == ACTION_APPROVE:
        return "approved"
    tracking = tracking_parsed.get(visual_id) or {}
    if tracking.get("after_revision") or tracking.get("after_regen_1") or tracking.get("manually_selected"):
        if action != ACTION_APPROVE:
            return "revised"
    if action in (ACTION_REJECT_DRIVE_HVAC, ACTION_REJECT_ALL, ACTION_REJECT_AI):
        return "revising"
    return "pending"


def _parse_actions_payload(raw: str) -> Dict[str, Any]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except Exception:
        return {}
    if isinstance(parsed, dict) and isinstance(parsed.get("actions"), dict):
        return parsed["actions"]
    if isinstance(parsed, dict):
        return parsed
    return {}


def _parse_tracking(raw: str) -> Dict[str, Any]:
    return _parse_tracking_column(raw or "")


def _sanitize_df(df):
    """
    Prevent empty cells from being written back to Sheets as the literal
    string "nan".

    The shared ``save_to_sheet`` helper does ``df.astype(str)``, which turns
    pandas ``NaN`` into "nan". Since this app saves the whole worksheet on
    every revise/approve, any blank cell on untouched rows would otherwise be
    stamped with "nan". We normalize the dataframe the app holds (and hands to
    the agent) so this never happens — without modifying the shared service.
    """
    if df is None or df.empty:
        return df
    df = df.fillna("")
    # Clean any pre-existing literal artifacts (exact-cell matches only).
    return df.replace({"nan": "", "NaN": "", "<NA>": "", "None": ""})


def load_workbook(session: UserSession) -> Tuple[Any, Any, int]:
    helpers = _import_slideshow_helpers()
    safe_str = helpers["safe_str"]

    if session.sheet is None:
        session.sheet = session.gc.open_by_url(session.sheet_link)
    ws, df = get_sheet_data_and_df(session.sheet, session.worksheet_name or DEFAULT_WORKSHEET)
    df = _sanitize_df(df)

    session.current_round = ROUND_INDEX

    try:
        # find worksheet by loosely matching name
        sheet_titles = {ws.title.strip().lower(): ws for ws in session.sheet.worksheets()}
        if "course info" in sheet_titles:
            ws = sheet_titles["course info"]
            
            @try_n_times(n=3, wait=1, backoff='exponential')
            def fetch_values():
                return ws.get_all_values()
                
            values = fetch_values()
            if values and len(values) > 1:
                headers = [str(h).strip().lower() for h in values[0]]
                if "course name" in headers:
                    idx = headers.index("course name")
                    # Find the first non-empty value in this column
                    for row in values[1:]:
                        if len(row) > idx and str(row[idx]).strip():
                            session.course_name = str(row[idx]).strip()
                            break
    except Exception as e:
        print(f"Error fetching course info: {e}")

    return ws, df, session.current_round


# Human-readable names for the manifest layout templates the slideshow agent emits.
_SCENE_TEMPLATE_LABELS = {
    "single_visual_hero": "Single visual",
    "two_item_split_comparison": "Two-item split",
    "multi_panel_grid": "Multi-panel grid",
    "main_plus_supporting_inset": "Main + inset",
}

_SCENE_TEMPLATE_ALIASES = {
    "single_hero": "single_visual_hero",
    "hero": "single_visual_hero",
    "two_item_split": "two_item_split_comparison",
    "split_comparison": "two_item_split_comparison",
    "multi_panel": "multi_panel_grid",
    "grid": "multi_panel_grid",
    "main_plus_inset": "main_plus_supporting_inset",
    "main_plus_supporting": "main_plus_supporting_inset",
    "main_visual_plus_inset": "main_plus_supporting_inset",
}


def _normalize_scene_template(template: str) -> str:
    """Map manifest template strings to one of the four canonical layout ids."""
    t = re.sub(r"[\s\-]+", "_", (template or "").strip().lower())
    return _SCENE_TEMPLATE_ALIASES.get(t, t)


def _scene_template_label(template: str, slot_count: int) -> str:
    normalized = _normalize_scene_template(template)
    label = _SCENE_TEMPLATE_LABELS.get(normalized)
    if label:
        return label
    if slot_count <= 1:
        return "Single visual"
    return f"{slot_count} panels"


def _parse_manifest_scenes_fallback(manifest_xml: str) -> List[Dict[str, Any]]:

    text = (manifest_xml or "").strip()
    if not text or text == "nan" or text.startswith("ERROR:"):
        return []
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]
    # Normalize doubled attribute quotes exported by Sheets (id=""1"" -> id="1").
    text = re.sub(r'""([^"<>]*)""', r'"\1"', text)

    # Find all <scene ...> ... </scene> blocks
    scene_pattern = r"<scene\s+([^>]*?)>(.*?)</scene>"
    scene_matches = re.finditer(scene_pattern, text, re.DOTALL | re.IGNORECASE)
    
    scenes: List[Dict[str, Any]] = []
    for idx, match in enumerate(scene_matches, start=1):
        attrs_text = match.group(1)
        body_text = match.group(2)
        
        # Parse attributes from scene tag (e.g. id and template)
        attr_map = {}
        for attr_match in re.finditer(r'([a-zA-Z0-9_-]+)\s*=\s*["\']([^"\']*)["\']', attrs_text):
            attr_map[attr_match.group(1).lower()] = attr_match.group(2)
            
        scene_id = attr_map.get("id", str(idx)).strip()
        template = attr_map.get("template", "").strip()
        
        # Parse narration_span
        narr_match = re.search(r"<narration_span[^>]*?>(.*?)</narration_span>", body_text, re.DOTALL | re.IGNORECASE)
        narration = ""
        if narr_match:
            narration = " ".join(narr_match.group(1).split())
            
        # Parse slots inside the scene
        slots = []
        # Find anything starting with <slot and pull its attributes up to the closing tag character
        for slot_match in re.finditer(r"<slot\s+([^>]+)", body_text, re.IGNORECASE):
            slot_attrs_text = slot_match.group(1)
            slot_attr_map = {}
            for s_attr_match in re.finditer(r'([a-zA-Z0-9_-]+)\s*=\s*["\']([^"\']*)["\']', slot_attrs_text):
                slot_attr_map[s_attr_match.group(1).lower()] = s_attr_match.group(2)
                
            role = slot_attr_map.get("role", "").strip()
            asset = slot_attr_map.get("asset", "").strip()
            
            # Unescape entities like &amp; to & for proper internal URL processing
            role = html.unescape(role)
            asset = html.unescape(asset)
            
            if role or asset:
                slots.append({
                    "role": role,
                    "asset": asset,
                })
                
        scenes.append({
            "id": scene_id,
            "template": _normalize_scene_template(template),
            "narration": narration,
            "slots": slots
        })
    return scenes


def _parse_manifest_scenes(manifest_xml: str) -> List[Dict[str, Any]]:
    """Parse the slideshow_manifest cell into ordered scenes.

    Returns a list of dicts: ``{"id", "template", "narration", "slots": [{role, asset}]}``.
    Falls back to a relaxed regex-based parser if strict XML parsing fails (e.g. on unescaped &).
    """
    text = (manifest_xml or "").strip()
    if not text or text == "nan" or text.startswith("ERROR:"):
        return []
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]
    # Normalize doubled attribute quotes exported by Sheets (id=""1"" -> id="1").
    text = re.sub(r'""([^"<>]*)""', r'"\1"', text)
    wrapped = text
    if "<slideshow_manifest" not in wrapped.lower():
        wrapped = f"<slideshow_manifest>\n{wrapped}\n</slideshow_manifest>"
    # Escape bare ampersands that would otherwise break XML parsing.
    wrapped = re.sub(r"&(?!(amp|lt|gt|quot|apos|#\d+|#x[0-9A-Fa-f]+);)", "&amp;", wrapped)
    try:
        root = ET.fromstring(wrapped)
    except ET.ParseError:
        # Fall back to regex parsing so raw ampersands or malformed XML syntax in Google Sheets
        # never hides the layout view in the human review app UI.
        return _parse_manifest_scenes_fallback(manifest_xml)

    tag = (root.tag or "").lower()
    if tag.endswith("slideshow_manifest"):
        scene_els = root.findall("scene")
    elif tag.endswith("scene"):
        scene_els = [root]
    else:
        return _parse_manifest_scenes_fallback(manifest_xml)

    scenes: List[Dict[str, Any]] = []
    for idx, scene_el in enumerate(scene_els, start=1):
        narr_el = scene_el.find("narration_span")
        narration = ""
        if narr_el is not None and narr_el.text:
            narration = " ".join(narr_el.text.split())
        slots = [
            {
                "role": (slot_el.get("role") or "").strip(),
                "asset": (slot_el.get("asset") or "").strip(),
            }
            for slot_el in scene_el.findall("slot")
        ]
        scenes.append(
            {
                "id": (scene_el.get("id") or str(idx)).strip() or str(idx),
                "template": _normalize_scene_template(scene_el.get("template") or ""),
                "narration": narration,
                "slots": slots,
            }
        )
    return scenes


def _extract_fgd_urls(final_graphics_definition: str) -> List[str]:
    pairs = parse_when_vo_assigned_pairs(final_graphics_definition or "")
    return [str(url or "").strip() for _, url in pairs if str(url or "").strip()]


def _extract_manifest_slot_urls(manifest_xml: str) -> List[str]:
    urls: List[str] = []
    for scene in _parse_manifest_scenes(manifest_xml):
        for slot in scene.get("slots") or []:
            asset = str((slot or {}).get("asset") or "").strip()
            if asset:
                urls.append(asset)
    return urls


def _validate_manifest_sync(manifest_xml: str, final_graphics_definition: str) -> Tuple[bool, str]:
    fgd_urls = _extract_fgd_urls(final_graphics_definition)
    if not fgd_urls:
        return True, "No FGD URLs to validate."

    manifest_urls = _extract_manifest_slot_urls(manifest_xml)
    if not manifest_urls:
        return False, "Manifest is empty or unparsable."
    if len(manifest_urls) != len(fgd_urls):
        return (
            False,
            f"Mismatch count: FGD has {len(fgd_urls)} URLs; manifest has {len(manifest_urls)} slots.",
        )

    remaining = list(fgd_urls)
    for manifest_url in manifest_urls:
        match_idx = None
        for idx, fgd_url in enumerate(remaining):
            if _urls_match(manifest_url, fgd_url):
                match_idx = idx
                break
        if match_idx is None:
            return False, f"Manifest URL not present in FGD: {manifest_url}"
        remaining.pop(match_idx)

    if remaining:
        return False, f"FGD URL(s) missing in manifest: {remaining}"
    return True, ""


def _regenerate_manifest_for_row(
    session: UserSession,
    row,
    *,
    final_graphics_definition: str,
    llm: str = "gemini_3_flash_thinking",
) -> Tuple[Optional[str], str]:
    slide_title = safe_str(row.get("Slide Chunk Title", "")).strip()
    slide_chunk = safe_str(row.get("Slide Chunk", "")).strip()
    topic_name = safe_str(row.get("Topic", "")).strip()
    subtopic_name = safe_str(row.get("Subtopic", "")).strip()
    slide_type = safe_str(row.get("Slide Type", "")).strip()
    if slide_type == "nan":
        slide_type = ""
    layout_plan = safe_str(row.get("layout_plan", "")).strip()
    storyboard_planning = safe_str(row.get("storyboard_planning", "")).strip()
    course_name = safe_str(getattr(session, "course_name", "")).strip() or "Course"

    row_index = row.get("_manifest_sync_row_index") if isinstance(row, dict) else None
    label = _manifest_sync_row_label(row, int(row_index) if row_index is not None else 0)
    print(f"{MANIFEST_SYNC_LOG_PREFIX} Regenerating slideshow_manifest for {label}...")

    try:
        manifest_xml, _ = generate_slideshow_manifest_for_row(
            course_name=course_name,
            topic_name=topic_name,
            subtopic_name=subtopic_name,
            slide_type=slide_type,
            slide_title=slide_title,
            slide_content=slide_chunk,
            layout_plan="",
            storyboard_planning="",
            final_graphics_definition=final_graphics_definition,
            drive=session.drive,
            llm=llm,
        )
        return manifest_xml, ""
    except Exception as exc:  # pragma: no cover - defensive
        print(f"{MANIFEST_SYNC_LOG_PREFIX} Regeneration error for {label}: {exc}")
        return None, str(exc)


def _cache_repaired_manifest(session: UserSession, row_index: int, manifest_xml: str) -> None:
    """Keep the just-repaired manifest in session so UI reads are instant (Sheets can lag)."""
    if not manifest_xml:
        return
    cache = getattr(session, "manifest_repair_cache", None)
    if cache is None:
        session.manifest_repair_cache = {}
        cache = session.manifest_repair_cache
    cache[int(row_index)] = manifest_xml


def _invalidate_manifest_repair_cache(session: UserSession, row_index: int) -> None:
    """Drop cached manifest for a row after a user-initiated write so sheet edits win."""
    cache = getattr(session, "manifest_repair_cache", None)
    if cache:
        cache.pop(int(row_index), None)


def _manifest_for_ui_row(session: UserSession, row_index: int, row) -> str:
    """Prefer in-session repaired manifest over a potentially stale Sheets cell."""
    cache = getattr(session, "manifest_repair_cache", None) or {}
    cached = cache.get(int(row_index))
    if cached:
        return cached
    return safe_str(row.get("slideshow_manifest", ""))


def _set_manifest_sync_status(session: UserSession, **kwargs) -> None:
    with _MANIFEST_SYNC_STATUS_LOCK:
        status = dict(getattr(session, "manifest_sync_status", None) or {})
        status.update(kwargs)
        session.manifest_sync_status = status


def _append_manifest_sync_repaired_row(session: UserSession, row_index: int) -> None:
    with _MANIFEST_SYNC_STATUS_LOCK:
        status = dict(getattr(session, "manifest_sync_status", None) or {})
        repaired = list(status.get("repaired_row_indices") or [])
        if row_index not in repaired:
            repaired.append(int(row_index))
        status["repaired_row_indices"] = repaired
        status["repaired_rows"] = len(repaired)
        status["repair_version"] = int(status.get("repair_version") or 0) + 1
        session.manifest_sync_status = status


def _append_manifest_sync_failed_row(session: UserSession, row_index: int) -> None:
    with _MANIFEST_SYNC_STATUS_LOCK:
        status = dict(getattr(session, "manifest_sync_status", None) or {})
        failed = list(status.get("failed_row_indices") or [])
        if row_index not in failed:
            failed.append(int(row_index))
        status["failed_row_indices"] = failed
        status["failed_rows"] = len(failed)
        session.manifest_sync_status = status


def _persist_slideshow_manifest_cell(
    session: UserSession,
    row_index: int,
    manifest_xml: str,
) -> None:
    def mutate(df) -> None:
        if "slideshow_manifest" not in df.columns:
            df["slideshow_manifest"] = ""
        df.at[row_index, "slideshow_manifest"] = manifest_xml

    mutate_row_cells(session, row_index, mutate)
    _cache_repaired_manifest(session, row_index, manifest_xml)


def get_manifest_sync_status(session: UserSession) -> Dict[str, Any]:
    with _MANIFEST_SYNC_STATUS_LOCK:
        status = dict(getattr(session, "manifest_sync_status", None) or {})
    if not status:
        return {
            "status": "idle",
            "checked_rows": 0,
            "repaired_rows": 0,
            "failed_rows": 0,
            "repaired_row_indices": [],
            "failed_row_indices": [],
            "repair_version": 0,
        }
    status.setdefault("repaired_row_indices", [])
    status.setdefault("failed_row_indices", [])
    status.setdefault("repair_version", 0)
    return status


def maybe_trigger_manifest_sync_checker(session: UserSession) -> bool:
    if not session.sheet_link:
        return False
    if getattr(session, "manifest_sync_triggered", False):
        return False
    session.manifest_sync_triggered = True
    return trigger_manifest_sync_checker_background(session)


def ensure_manifest_sync_for_row(
    session: UserSession,
    df,
    row_index: int,
    *,
    reason: str = "",
    force_regenerate: bool = False,
) -> Dict[str, Any]:
    if "slideshow_manifest" not in df.columns:
        return {"checked": False, "synced": True, "reason": "missing_manifest_column"}

    final_col = _pick_column(df, FINAL_GRAPHICS_COLUMN)
    if final_col not in df.columns:
        return {"checked": False, "synced": True, "reason": "missing_fgd_column"}

    row = df.loc[row_index]
    final_graphics_definition = safe_str(row.get(final_col, "")).strip()
    if not final_graphics_definition or final_graphics_definition == "nan":
        return {"checked": False, "synced": True, "reason": "empty_fgd"}

    current_manifest = safe_str(row.get("slideshow_manifest", "")).strip()
    is_synced, err = _validate_manifest_sync(current_manifest, final_graphics_definition)
    if is_synced and not force_regenerate:
        return {"checked": True, "synced": True, "reason": "already_synced"}

    label = _manifest_sync_row_label(row, row_index)
    print(
        f"{MANIFEST_SYNC_LOG_PREFIX} Out of sync for {label} "
        f"(reason={reason or 'unspecified'}; issue={err or 'forced_regenerate'}) — "
        f"starting regeneration"
    )
    last_error = err or "forced_regenerate"
    for attempt in range(1, MANIFEST_SYNC_MAX_REGEN_ATTEMPTS + 1):
        print(
            f"{MANIFEST_SYNC_LOG_PREFIX} Regeneration attempt {attempt}/"
            f"{MANIFEST_SYNC_MAX_REGEN_ATTEMPTS} for {label}"
        )
        row_for_regen = row.to_dict() if hasattr(row, "to_dict") else row
        if isinstance(row_for_regen, dict):
            row_for_regen = dict(row_for_regen)
            row_for_regen["_manifest_sync_row_index"] = row_index
        regenerated_manifest, regen_error = _regenerate_manifest_for_row(
            session,
            row_for_regen,
            final_graphics_definition=final_graphics_definition,
        )
        if not regenerated_manifest:
            last_error = regen_error or "regeneration_failed"
            print(
                f"{MANIFEST_SYNC_LOG_PREFIX} Attempt {attempt} failed for {label}: {last_error}"
            )
            continue

        regenerated_ok, regenerated_err = _validate_manifest_sync(
            regenerated_manifest,
            final_graphics_definition,
        )
        if regenerated_ok:
            df.at[row_index, "slideshow_manifest"] = regenerated_manifest
            _cache_repaired_manifest(session, row_index, regenerated_manifest)
            print(
                f"{MANIFEST_SYNC_LOG_PREFIX} Regeneration succeeded for {label} "
                f"on attempt {attempt}"
            )
            return {
                "checked": True,
                "synced": True,
                "reason": "repaired_by_regeneration",
                "attempts": attempt,
            }

        last_error = regenerated_err or "regenerated_manifest_invalid"
        print(
            f"{MANIFEST_SYNC_LOG_PREFIX} Attempt {attempt} produced invalid manifest "
            f"for {label}: {last_error}"
        )

    print(
        f"{MANIFEST_SYNC_LOG_PREFIX} Regeneration exhausted for {label} after "
        f"{MANIFEST_SYNC_MAX_REGEN_ATTEMPTS} attempt(s): {last_error}"
    )
    return {
        "checked": True,
        "synced": False,
        "reason": "regeneration_exhausted",
        "error": last_error,
        "attempts": MANIFEST_SYNC_MAX_REGEN_ATTEMPTS,
    }


def _parallel_manifest_sync_check_task(
    row_index: int,
    manifest_xml: str,
    final_graphics_definition: str,
) -> Dict[str, Any]:
    fgd = (final_graphics_definition or "").strip()
    if not fgd or fgd == "nan":
        return {"row_index": row_index, "checked": False, "needs_repair": False}

    is_synced, sync_err = _validate_manifest_sync(manifest_xml, fgd)
    return {
        "row_index": row_index,
        "checked": True,
        "needs_repair": not is_synced,
        "sync_error": sync_err if not is_synced else "",
    }


def _parallel_manifest_sync_repair_task(
    session: UserSession,
    row_index: int,
    row_snapshot: Dict[str, Any],
    final_col: str,
    *,
    reason: str,
) -> Dict[str, Any]:
    final_graphics_definition = safe_str(row_snapshot.get(final_col, "")).strip()
    if not final_graphics_definition or final_graphics_definition == "nan":
        return {"row_index": row_index, "synced": False, "reason": "empty_fgd"}

    label = _manifest_sync_row_label(row_snapshot, row_index)
    print(
        f"{MANIFEST_SYNC_LOG_PREFIX} Out of sync for {label} "
        f"(reason={reason}) — starting regeneration"
    )
    last_error = "forced_regenerate"
    row_snapshot = dict(row_snapshot)
    row_snapshot["_manifest_sync_row_index"] = row_index
    for attempt in range(1, MANIFEST_SYNC_MAX_REGEN_ATTEMPTS + 1):
        print(
            f"{MANIFEST_SYNC_LOG_PREFIX} Regeneration attempt {attempt}/"
            f"{MANIFEST_SYNC_MAX_REGEN_ATTEMPTS} for {label}"
        )
        regenerated_manifest, regen_error = _regenerate_manifest_for_row(
            session,
            row_snapshot,
            final_graphics_definition=final_graphics_definition,
        )
        if not regenerated_manifest:
            last_error = regen_error or "regeneration_failed"
            print(
                f"{MANIFEST_SYNC_LOG_PREFIX} Attempt {attempt} failed for {label}: {last_error}"
            )
            continue

        regenerated_ok, regenerated_err = _validate_manifest_sync(
            regenerated_manifest,
            final_graphics_definition,
        )
        if regenerated_ok:
            print(f"{MANIFEST_SYNC_LOG_PREFIX} Saving regenerated manifest for {label}...")
            _persist_slideshow_manifest_cell(session, row_index, regenerated_manifest)
            _append_manifest_sync_repaired_row(session, row_index)
            print(
                f"{MANIFEST_SYNC_LOG_PREFIX} Regeneration succeeded for {label} "
                f"on attempt {attempt}"
            )
            return {
                "row_index": row_index,
                "synced": True,
                "reason": "repaired_by_regeneration",
                "attempts": attempt,
            }
        last_error = regenerated_err or "regenerated_manifest_invalid"
        print(
            f"{MANIFEST_SYNC_LOG_PREFIX} Attempt {attempt} produced invalid manifest "
            f"for {label}: {last_error}"
        )

    _append_manifest_sync_failed_row(session, row_index)
    print(
        f"{MANIFEST_SYNC_LOG_PREFIX} Regeneration exhausted for {label} after "
        f"{MANIFEST_SYNC_MAX_REGEN_ATTEMPTS} attempt(s): {last_error}"
    )
    return {
        "row_index": row_index,
        "synced": False,
        "reason": "regeneration_exhausted",
        "error": last_error,
        "attempts": MANIFEST_SYNC_MAX_REGEN_ATTEMPTS,
    }


def run_manifest_sync_checker_for_loaded_sheet(session: UserSession) -> Dict[str, Any]:
    worksheet = session.worksheet_name or DEFAULT_WORKSHEET
    print(
        f"{MANIFEST_SYNC_LOG_PREFIX} Starting sheet sync check "
        f"(worksheet={worksheet!r}, max_workers={MANIFEST_SYNC_MAX_WORKERS})"
    )

    _, df, _ = load_workbook(session)
    if "slideshow_manifest" not in df.columns:
        print(f"{MANIFEST_SYNC_LOG_PREFIX} No slideshow_manifest column — skipping sync")
        return {
            "checked_rows": 0,
            "repaired_rows": 0,
            "failed_rows": 0,
            "repaired_row_indices": [],
            "failed_row_indices": [],
        }

    final_col = _pick_column(df, FINAL_GRAPHICS_COLUMN)
    if final_col not in df.columns:
        print(f"{MANIFEST_SYNC_LOG_PREFIX} No final_graphics_definition column — skipping sync")
        return {
            "checked_rows": 0,
            "repaired_rows": 0,
            "failed_rows": 0,
            "repaired_row_indices": [],
            "failed_row_indices": [],
        }

    total_rows = len(df)
    print(
        f"{MANIFEST_SYNC_LOG_PREFIX} Checking {total_rows} row(s) in parallel "
        f"(max_workers={MANIFEST_SYNC_MAX_WORKERS})..."
    )

    check_jobs: List[Tuple[int, str, str]] = []
    repair_snapshots: Dict[int, Dict[str, Any]] = {}
    for row_index in range(len(df)):
        row = df.loc[row_index]
        final_graphics_definition = safe_str(row.get(final_col, "")).strip()
        manifest_xml = safe_str(row.get("slideshow_manifest", "")).strip()
        check_jobs.append((row_index, manifest_xml, final_graphics_definition))
        repair_snapshots[row_index] = row.to_dict()

    rows_needing_repair: List[int] = []
    repair_reasons: Dict[int, str] = {}
    checked_rows = 0
    with ThreadPoolExecutor(max_workers=MANIFEST_SYNC_MAX_WORKERS) as executor:
        futures = [
            executor.submit(_parallel_manifest_sync_check_task, row_index, manifest_xml, fgd)
            for row_index, manifest_xml, fgd in check_jobs
        ]
        for future in as_completed(futures):
            result = future.result()
            if not result.get("checked"):
                continue
            checked_rows += 1
            if result.get("needs_repair"):
                row_index = int(result["row_index"])
                rows_needing_repair.append(row_index)
                repair_reasons[row_index] = str(result.get("sync_error") or "out_of_sync")

    rows_needing_repair.sort()
    in_sync_count = checked_rows - len(rows_needing_repair)
    print(
        f"{MANIFEST_SYNC_LOG_PREFIX} Check complete: {checked_rows} row(s) checked, "
        f"{in_sync_count} in sync, {len(rows_needing_repair)} need regeneration"
    )
    if rows_needing_repair:
        for row_index in rows_needing_repair:
            label = _manifest_sync_row_label(repair_snapshots[row_index], row_index)
            issue = repair_reasons.get(row_index, "out_of_sync")
            print(f"{MANIFEST_SYNC_LOG_PREFIX}   NEEDS REPAIR: {label} — {issue}")
    else:
        print(f"{MANIFEST_SYNC_LOG_PREFIX} All checked rows are in sync — no regeneration needed")

    _set_manifest_sync_status(
        session,
        checked_rows=checked_rows,
        rows_needing_repair=len(rows_needing_repair),
    )

    repaired_rows = 0
    failed_rows = 0
    if rows_needing_repair:
        print(
            f"{MANIFEST_SYNC_LOG_PREFIX} Starting parallel regeneration for "
            f"{len(rows_needing_repair)} row(s) (max_workers={MANIFEST_SYNC_MAX_WORKERS})..."
        )
        with ThreadPoolExecutor(max_workers=MANIFEST_SYNC_MAX_WORKERS) as executor:
            futures = [
                executor.submit(
                    _parallel_manifest_sync_repair_task,
                    session,
                    row_index,
                    repair_snapshots[row_index],
                    final_col,
                    reason="sheet_load_checker",
                )
                for row_index in rows_needing_repair
            ]
            for future in as_completed(futures):
                result = future.result()
                row_index = int(result.get("row_index", -1))
                label = _manifest_sync_row_label(
                    repair_snapshots.get(row_index, {}),
                    row_index,
                )
                if result.get("synced"):
                    repaired_rows += 1
                    print(f"{MANIFEST_SYNC_LOG_PREFIX} Repair finished OK: {label}")
                else:
                    failed_rows += 1
                    err = result.get("error") or result.get("reason") or "unknown"
                    print(f"{MANIFEST_SYNC_LOG_PREFIX} Repair finished FAILED: {label} — {err}")

    status = get_manifest_sync_status(session)
    summary = {
        "checked_rows": checked_rows,
        "repaired_rows": repaired_rows,
        "failed_rows": failed_rows,
        "repaired_row_indices": list(status.get("repaired_row_indices") or []),
        "failed_row_indices": list(status.get("failed_row_indices") or []),
    }
    print(f"{MANIFEST_SYNC_LOG_PREFIX} Sheet sync finished: {summary}")
    return summary


def trigger_manifest_sync_checker_background(session: UserSession) -> bool:
    session_id = str(getattr(session, "session_id", "") or "global")
    with _MANIFEST_SYNC_LOCK:
        if _MANIFEST_SYNC_RUNNING_BY_SESSION.get(session_id):
            return False
        _MANIFEST_SYNC_RUNNING_BY_SESSION[session_id] = True

    _set_manifest_sync_status(
        session,
        status="running",
        checked_rows=0,
        repaired_rows=0,
        failed_rows=0,
        repaired_row_indices=[],
        failed_row_indices=[],
        rows_needing_repair=0,
        repair_version=0,
        sync_run_id=uuid.uuid4().hex[:8],
        error="",
    )
    print(f"{MANIFEST_SYNC_LOG_PREFIX} Background sync started for session {session_id}")

    def _runner() -> None:
        try:
            summary = run_manifest_sync_checker_for_loaded_sheet(session)
            _set_manifest_sync_status(session, status="done", error="", **summary)
        except Exception as exc:  # pragma: no cover - defensive
            _set_manifest_sync_status(
                session,
                status="error",
                error=str(exc),
            )
            print(f"{MANIFEST_SYNC_LOG_PREFIX} Background sync failed: {exc}")
        finally:
            with _MANIFEST_SYNC_LOCK:
                _MANIFEST_SYNC_RUNNING_BY_SESSION[session_id] = False

    threading.Thread(target=_runner, daemon=True).start()
    return True


def _url_token(u: str) -> str:
    # FGD entries can carry trailing free-text annotations after the URL
    # (e.g. "https://youtube.com/... (use the image at 3m16s)" or "... (AI Generated)").
    # A URL cannot contain whitespace, so the first whitespace-delimited token is the URL.
    s = (u or "").strip()
    return s.split()[0] if s else ""


def _urls_match(u1: str, u2: str) -> bool:
    if not u1 or not u2:
        return False
    t1 = _url_token(u1)
    t2 = _url_token(u2)
    if not t1 or not t2:
        return False
    if t1.lower() == t2.lower():
        return True
    try:
        return urls_match_for_graphics_assignment(t1, t2)
    except Exception:
        return t1.lower() == t2.lower()


def _parse_per_scene_layout_feedback(feedback_str: str) -> Dict[str, str]:
    if not feedback_str:
        return {}
    try:
        data = json.loads(feedback_str)
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
    except Exception:
        return {"1": feedback_str}
    return {}


def _build_scenes_payload(
    manifest_xml: str,
    flat_steps: List[Dict[str, Any]],
    layout_feedback_str: str = "",
) -> List[Dict[str, Any]]:
    """Map manifest scenes onto the slide's ordered visuals.

    We first try to match each slot asset URL to the step's assetUrl or afterUrl.
    If we can perfectly map all slots to steps this way, we use that mapping.
    Otherwise (e.g. if the manifest is stale or mismatching), we fall back
    to sequential positional matching. If the total slot count does not match,
    we return [] to fall back to flat list.
    """
    manifest_scenes = _parse_manifest_scenes(manifest_xml)
    if not manifest_scenes:
        return []

    total_slots = sum(len(sc["slots"]) for sc in manifest_scenes)
    if total_slots != len(flat_steps) or total_slots == 0:
        return []

    feedback_map = _parse_per_scene_layout_feedback(layout_feedback_str)

    # Try mapping by URL
    url_mapping_success = True
    assigned_vids_by_scene = {}
    used_step_indices = set()

    for sc in manifest_scenes:
        assigned_vids = []
        for slot in sc["slots"]:
            slot_asset = slot.get("asset") or ""
            matched_idx = -1
            # Find an unused step whose URL matches
            for idx, step in enumerate(flat_steps):
                if idx in used_step_indices:
                    continue
                step_url = step.get("afterUrl") or step.get("assetUrl") or ""
                if _urls_match(slot_asset, step_url):
                    matched_idx = idx
                    break
            if matched_idx != -1:
                used_step_indices.add(matched_idx)
                assigned_vids.append(flat_steps[matched_idx].get("visualId"))
            else:
                url_mapping_success = False
                break
        if not url_mapping_success:
            break
        assigned_vids_by_scene[sc["id"]] = assigned_vids

    out: List[Dict[str, Any]] = []
    if url_mapping_success and len(used_step_indices) == len(flat_steps):
        # Successful URL-based match
        for sc in manifest_scenes:
            n = len(sc["slots"])
            out.append(
                {
                    "id": sc["id"],
                    "template": sc["template"],
                    "templateLabel": _scene_template_label(sc["template"], n),
                    "narration": sc["narration"],
                    "slotCount": n,
                    "visualIds": assigned_vids_by_scene[sc["id"]],
                    "layoutFeedback": feedback_map.get(sc["id"], ""),
                }
            )
        return out

    # Fallback to positional mapping
    out = []
    cursor = 0
    for sc in manifest_scenes:
        n = len(sc["slots"])
        chunk = flat_steps[cursor : cursor + n]
        cursor += n
        out.append(
            {
                "id": sc["id"],
                "template": sc["template"],
                "templateLabel": _scene_template_label(sc["template"], n),
                "narration": sc["narration"],
                "slotCount": n,
                "visualIds": [step.get("visualId") for step in chunk],
                "layoutFeedback": feedback_map.get(sc["id"], ""),
            }
        )
    return out


def slides_to_ui_payload(session: UserSession) -> Dict[str, Any]:
    helpers = _import_slideshow_helpers()
    build_slides_from_df = helpers["build_slides_from_df"]
    detect_asset_type = helpers["detect_asset_type"]
    get_round_column_name = helpers["get_round_column_name"]
    safe_str = helpers["safe_str"]

    _, df, current_round = load_workbook(session)
    maybe_trigger_manifest_sync_checker(session)
    cmap = _column_map(df)
    
    # Forward-fill the topic column to handle merged or blank cells in the sheet
    topic_col = cmap.get("topic")
    if topic_col and topic_col in df.columns:
        df[topic_col] = df[topic_col].replace(r'^\s*$', None, regex=True).ffill().fillna("")
        
    slides = build_slides_from_df(df, cmap)

    actions_col = get_round_column_name(HUMAN_REVIEW_ACTIONS_COLUMN, current_round)
    tracking_col = get_round_column_name(HUMAN_FEEDBACK_TRACKING_COLUMN, current_round)
    seg_feedback_col = get_round_column_name(SEGMENTATION_FEEDBACK_COLUMN, current_round)
    layout_feedback_col = get_round_column_name(LAYOUT_FEEDBACK_COLUMN, current_round)

    ui_slides: List[Dict[str, Any]] = []
    for slide_idx, slide in enumerate(slides):
        row = df.iloc[slide["row_index"]]
        actions_map = _parse_actions_payload(safe_str(row.get(actions_col, "")))
        tracking_map = _parse_tracking(safe_str(row.get(tracking_col, "")))
        
        seg_feedback = safe_str(row.get(seg_feedback_col, "")).strip()
        layout_feedback = safe_str(row.get(layout_feedback_col, "")).strip()

        segments_out = []
        for segment in slide.get("segments") or []:
            steps_out = []
            for step in segment.get("steps") or []:
                asset = _first_http_url(safe_str(step.get("asset", "")).strip())
                asset_type = detect_asset_type(asset)
                seg_num = segment.get("segment_index", 1)
                step_num = step.get("step_index", 1)
                visual_id = f"S{seg_num}V{step_num}"
                status = _status_for_visual(visual_id, actions_map, tracking_map)

                tracking_entry = tracking_map.get(visual_id) or {}
                before_url = tracking_entry.get("original") or asset
                after_url = (
                    tracking_entry.get("after_revision")
                    or tracking_entry.get("after_regen_1")
                    or tracking_entry.get("manually_selected")
                )

                action_entry = actions_map.get(visual_id) or {}
                feedback = safe_str(action_entry.get("feedback", "")).strip()
                mode = ""
                for m, act in MODE_TO_ACTION.items():
                    if action_entry.get("action") == act:
                        mode = m
                        break

                alternatives = _alternatives_for_segment(
                    row,
                    segment_index=seg_num,
                    asset_type=asset_type,
                    exclude_url=asset,
                )

                steps_out.append(
                    {
                        "voiceover": safe_str(step.get("voiceover", "")),
                        "instruction": safe_str(step.get("instruction", "")),
                        "justification": safe_str(step.get("justification", "")),
                        "type": "video" if asset_type == "video" else "image",
                        "label": _label_from_asset(asset, asset_type),
                        "tone": "slate",
                        "status": status,
                        "feedback": feedback,
                        "assetUrl": asset,
                        "beforeUrl": before_url,
                        "afterUrl": after_url or "",
                        "source": _infer_source(asset, asset_type),
                        "visualId": visual_id,
                        "segmentIndex": seg_num,
                        "stepIndex": step_num,
                        "rowIndex": int(slide["row_index"]),
                        "mode": mode,
                        "alternatives": alternatives,
                    }
                )
            if steps_out:
                segments_out.append({"steps": steps_out})

        flat_steps = [step for seg in segments_out for step in seg["steps"]]
        row_index = int(slide["row_index"])
        scenes_payload = _build_scenes_payload(
            _manifest_for_ui_row(session, row_index, row),
            flat_steps,
            layout_feedback,
        )

        ui_slides.append(
            {
                "title": safe_str(slide.get("slide_title", "")) or f"Slide {slide_idx + 1}",
                "topic": safe_str(slide.get("topic", "")),
                "segments": segments_out,
                "segmentationFeedback": seg_feedback,
                "layoutFeedback": layout_feedback,
                "scenes": scenes_payload,
            }
        )

    course_name = session.course_name or "Course"
    topic_name = ui_slides[0].get("topic") if ui_slides else ""

    return {
        "slides": ui_slides,
        "currentRound": current_round,
        "moduleLabel": course_name,
        "courseName": course_name,
        "topicName": topic_name,
        "sheetLink": session.sheet_link,
        "worksheetName": session.worksheet_name,
    }


def _normalize_vo(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip().lower()


def _split_feedback_blocks(text: str) -> List[str]:
    if not text or not text.strip():
        return []
    parts = re.split(r"(?=^When VO:)", text, flags=re.MULTILINE)
    return [p.strip() for p in parts if p.strip()]


def _vo_of_block(block: str) -> str:
    match = re.match(r"When VO:\s*(.*?)\s*\nHuman Feedback:", block, flags=re.DOTALL)
    return match.group(1).strip() if match else ""


def filter_feedback_to_vo(feedback_text: str, target_vo: str) -> str:
    """Return only the feedback block(s) whose VO matches ``target_vo``.

    The feedback cell holds one block per voiceover part for the whole slide.
    A per-visual revision must only act on its own block, otherwise the worker
    would re-revise every visual in the slide that has feedback.

    Returns "" if no block matches (caller decides the fallback).
    """
    blocks = _split_feedback_blocks(feedback_text)
    if not blocks:
        return ""
    target = _normalize_vo(target_vo)
    if not target:
        return ""
    matched = [b for b in blocks if _normalize_vo(_vo_of_block(b)) == target]
    if not matched:
        # Looser fallback: VO stored with minor differences (punctuation, etc.).
        matched = [b for b in blocks if target in _normalize_vo(_vo_of_block(b))]
    return "\n\n".join(matched)


def _merge_feedback_block(existing: str, vo: str, feedback: str) -> str:
    block = f"When VO: {vo}\nHuman Feedback: {feedback}"
    if not existing.strip():
        return block
    if vo in existing:
        pattern = re.compile(
            rf"When VO:\s*{re.escape(vo)}\s*\nHuman Feedback:\s*[^\n]*(?:\n(?!When VO:).*)*",
            re.MULTILINE,
        )
        if pattern.search(existing):
            return pattern.sub(block, existing, count=1)
    return existing.rstrip() + "\n\n" + block


def write_visual_action(
    session: UserSession,
    *,
    row_index: int,
    segment_index: int,
    step_index: int,
    visual_id: str,
    vo: str,
    action: str,
    feedback: str = "",
    mode: str = "",
) -> None:
    helpers = _import_slideshow_helpers()
    get_round_column_name = helpers["get_round_column_name"]
    safe_str = helpers["safe_str"]

    actions_col = get_round_column_name(HUMAN_REVIEW_ACTIONS_COLUMN, ROUND_INDEX)
    feedback_col = get_round_column_name(HUMAN_FEEDBACK_COLUMN, ROUND_INDEX)

    def mutate(df) -> None:
        if actions_col not in df.columns:
            df[actions_col] = ""
        if feedback_col not in df.columns:
            df[feedback_col] = ""

        raw_actions = safe_str(df.at[row_index, actions_col])
        try:
            payload = json.loads(raw_actions) if raw_actions else {}
        except Exception:
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        actions_map = payload.get("actions") if isinstance(payload.get("actions"), dict) else {}
        segment_modes = payload.get("segment_modes") if isinstance(payload.get("segment_modes"), dict) else {}

        actions_map[visual_id] = {
            "action": action,
            "feedback": feedback,
            "vo": vo,
            "segment": segment_index,
        }

        if action in MODE_TO_ACTION.values():
            seg_key = str(segment_index)
            seg_mode = "all" if mode in ("all", "ai") or action in (MODE_TO_ACTION["all"], MODE_TO_ACTION["ai"]) else "drive_hvac"
            existing = segment_modes.get(seg_key, "drive_hvac")
            segment_modes[seg_key] = "all" if existing == "all" or seg_mode == "all" else "drive_hvac"

            effective_feedback = feedback
            if action == MODE_TO_ACTION["ai"] and not effective_feedback:
                effective_feedback = AI_NO_FEEDBACK_MARKER
            elif not effective_feedback:
                effective_feedback = DEFAULT_REJECT_FEEDBACK

            existing_fb = safe_str(df.at[row_index, feedback_col])
            df.at[row_index, feedback_col] = _merge_feedback_block(existing_fb, vo, effective_feedback)

        payload = {"actions": actions_map, "segment_modes": segment_modes, "round": ROUND_INDEX}
        df.at[row_index, actions_col] = json.dumps(payload, ensure_ascii=True)

    mutate_row_cells(session, row_index, mutate)


def select_pool_alternative(
    session: UserSession,
    *,
    row_index: int,
    segment_index: int,
    step_index: int,
    visual_id: str,
    vo: str,
    asset_url: str,
) -> None:
    """Assign a pool candidate to a visual, persist tracking + graphics definition, then approve."""
    helpers = _import_slideshow_helpers()
    get_round_column_name = helpers["get_round_column_name"]
    parse_graphics_definition = helpers["parse_graphics_definition"]
    safe_str = helpers["safe_str"]

    asset_url = (asset_url or "").strip()
    if not asset_url:
        raise ValueError("asset_url is required")

    tracking_col = get_round_column_name(HUMAN_FEEDBACK_TRACKING_COLUMN, ROUND_INDEX)

    def mutate(df) -> None:
        _invalidate_manifest_repair_cache(session, row_index)
        final_col = _pick_column(df, FINAL_GRAPHICS_COLUMN)
        if final_col not in df.columns:
            raise ValueError(f"Missing column: {final_col}")
        if tracking_col not in df.columns:
            df[tracking_col] = ""

        raw_def = safe_str(df.at[row_index, final_col])
        segments = parse_graphics_definition(raw_def)
        current_fgd_url = extract_asset_url(raw_def, segment_index, step_index)

        tracking_map = _parse_tracking(safe_str(df.at[row_index, tracking_col]))
        prior_tracking_entry = tracking_map.get(visual_id)

        if visual_id not in tracking_map:
            tracking_map[visual_id] = {
                "original": current_fgd_url or None,
                "manually_selected": None,
                "after_revision": None,
                "after_regen_1": None,
                "after_regen_2": None,
            }
        elif not tracking_map[visual_id].get("original"):
            tracking_map[visual_id]["original"] = current_fgd_url or None

        tracking_map[visual_id]["manually_selected"] = asset_url

        override_map = {(segment_index, step_index): asset_url}
        updated_def = _apply_asset_overrides_to_raw(raw_def, segments, override_map)

        df.at[row_index, final_col] = updated_def
        df.at[row_index, tracking_col] = _format_human_feedback_revision_tracking(tracking_map)

        if asset_url and asset_url != current_fgd_url:
            old_candidates = _collect_manifest_old_url_candidates(current_fgd_url, prior_tracking_entry)
            if not _patch_slideshow_manifest_urls(df, row_index, asset_url, old_candidates):
                print(
                    f"[human_feedback] manifest patch: no slot matched for row {row_index} "
                    f"visual {visual_id} (pool select; tried {len(old_candidates)} candidate URL(s))"
                )

        ensure_manifest_sync_for_row(
            session,
            df,
            row_index,
            reason="select_pool_alternative",
            force_regenerate=False,
        )

    # The mutate runs inside the global write lock against the freshest sheet and
    # touches only this visual's (segment, step) slice + its tracking key, so it
    # is safe to run alongside an in-flight revision of a DIFFERENT visual on the
    # same slide.
    mutate_row_cells(session, row_index, mutate)

    approve_visual(
        session,
        row_index=row_index,
        segment_index=segment_index,
        step_index=step_index,
        visual_id=visual_id,
        vo=vo,
    )


def extract_first_url(text: str) -> Optional[str]:
    if not text:
        return None
    # Matches http or https urls
    match = re.search(r'(https?://[^\s<>"]+|www\.[^\s<>"]+)', text)
    if match:
        url = match.group(0)
        # Clean trailing punctuation
        url = url.rstrip('.,!?;:)("')
        if url.startswith('www.'):
            url = 'https://' + url
        return url
    return None


def replace_visual_with_url(
    session: UserSession,
    *,
    row_index: int,
    segment_index: int,
    step_index: int,
    visual_id: str,
    vo: str,
    feedback: str,
) -> str:
    """Extract first URL from feedback, assign to visual, persist tracking + graphics definition, then approve."""
    url = extract_first_url(feedback)
    if not url:
        raise ValueError("No valid URL found in the feedback box. Please paste a link to an image or video.")

    # Normalize Google Drive links
    if "drive.google.com" in url or "docs.google.com" in url:
        from services.drive_service import extract_drive_id_from_url
        file_id = extract_drive_id_from_url(url)
        if file_id:
            url = f"https://drive.google.com/file/d/{file_id}/view"

    helpers = _import_slideshow_helpers()
    get_round_column_name = helpers["get_round_column_name"]
    parse_graphics_definition = helpers["parse_graphics_definition"]
    safe_str = helpers["safe_str"]

    tracking_col = get_round_column_name(HUMAN_FEEDBACK_TRACKING_COLUMN, ROUND_INDEX)

    def mutate(df) -> None:
        _invalidate_manifest_repair_cache(session, row_index)
        final_col = _pick_column(df, FINAL_GRAPHICS_COLUMN)
        if final_col not in df.columns:
            raise ValueError(f"Missing column: {final_col}")
        if tracking_col not in df.columns:
            df[tracking_col] = ""

        raw_def = safe_str(df.at[row_index, final_col])
        segments = parse_graphics_definition(raw_def)
        current_fgd_url = extract_asset_url(raw_def, segment_index, step_index)

        tracking_map = _parse_tracking(safe_str(df.at[row_index, tracking_col]))
        prior_tracking_entry = tracking_map.get(visual_id)

        if visual_id not in tracking_map:
            tracking_map[visual_id] = {
                "original": current_fgd_url or None,
                "manually_selected": None,
                "after_revision": None,
                "after_regen_1": None,
                "after_regen_2": None,
            }
        elif not tracking_map[visual_id].get("original"):
            tracking_map[visual_id]["original"] = current_fgd_url or None

        tracking_map[visual_id]["manually_selected"] = url

        override_map = {(segment_index, step_index): url}
        updated_def = _apply_asset_overrides_to_raw(raw_def, segments, override_map)

        df.at[row_index, final_col] = updated_def
        df.at[row_index, tracking_col] = _format_human_feedback_revision_tracking(tracking_map)

        if url and url != current_fgd_url:
            old_candidates = _collect_manifest_old_url_candidates(current_fgd_url, prior_tracking_entry)
            if not _patch_slideshow_manifest_urls(df, row_index, url, old_candidates):
                print(
                    f"[human_feedback] manifest patch: no slot matched for row {row_index} "
                    f"visual {visual_id} (manual replace; tried {len(old_candidates)} candidate URL(s))"
                )

        ensure_manifest_sync_for_row(
            session,
            df,
            row_index,
            reason="replace_visual_with_url",
            force_regenerate=False,
        )

    mutate_row_cells(session, row_index, mutate)

    approve_visual(
        session,
        row_index=row_index,
        segment_index=segment_index,
        step_index=step_index,
        visual_id=visual_id,
        vo=vo,
    )
    
    return url


def revert_visual(
    session: UserSession,
    *,
    row_index: int,
    segment_index: int,
    step_index: int,
    visual_id: str,
    vo: str,
) -> None:
    helpers = _import_slideshow_helpers()
    get_round_column_name = helpers["get_round_column_name"]
    safe_str = helpers["safe_str"]
    parse_graphics_definition = helpers["parse_graphics_definition"]

    actions_col = get_round_column_name(HUMAN_REVIEW_ACTIONS_COLUMN, ROUND_INDEX)
    tracking_col = get_round_column_name(HUMAN_FEEDBACK_TRACKING_COLUMN, ROUND_INDEX)

    def mutate(df) -> None:
        _invalidate_manifest_repair_cache(session, row_index)
        # 1. Clear the action so it becomes pending again
        if actions_col in df.columns:
            raw_actions = safe_str(df.at[row_index, actions_col])
            try:
                payload = json.loads(raw_actions) if raw_actions else {}
            except Exception:
                payload = {}
            if isinstance(payload, dict) and "actions" in payload and visual_id in payload["actions"]:
                del payload["actions"][visual_id]
                df.at[row_index, actions_col] = json.dumps(payload, ensure_ascii=True)

        # 2. Restore the original URL in the graphics definition
        if tracking_col in df.columns:
            tmap = _parse_tracking(safe_str(df.at[row_index, tracking_col]))
            tracking_entry = tmap.get(visual_id, {})
            original_url = tracking_entry.get("original")

            final_col = _pick_column(df, FINAL_GRAPHICS_COLUMN)
            if final_col in df.columns and original_url:
                raw_def = safe_str(df.at[row_index, final_col])
                current_url = extract_asset_url(raw_def, segment_index, step_index)
                segments = parse_graphics_definition(raw_def)
                updated = _apply_asset_overrides_to_raw(
                    raw_def, segments, {(segment_index, step_index): original_url}
                )
                df.at[row_index, final_col] = updated

                if current_url and current_url != original_url:
                    old_candidates = _collect_manifest_old_url_candidates(current_url, tracking_entry)
                    _patch_slideshow_manifest_urls(df, row_index, original_url, old_candidates)

            # 3. Wipe the revision history for this visual in tracking!
            if original_url:
                tmap[visual_id] = {
                    "original": original_url,
                    "manually_selected": None,
                    "after_revision": None,
                    "after_regen_1": None,
                    "after_regen_2": None,
                }
                df.at[row_index, tracking_col] = _format_human_feedback_revision_tracking(tmap)

        ensure_manifest_sync_for_row(
            session,
            df,
            row_index,
            reason="revert_visual",
            force_regenerate=False,
        )

    mutate_row_cells(session, row_index, mutate)


def approve_visual(session: UserSession, row_index: int, segment_index: int, step_index: int, visual_id: str, vo: str) -> None:
    write_visual_action(
        session,
        row_index=row_index,
        segment_index=segment_index,
        step_index=step_index,
        visual_id=visual_id,
        vo=vo,
        action=ACTION_APPROVE,
    )


def apply_segmentation_revision_to_sheet(
    session: UserSession,
    *,
    row_index: int,
    feedback: str,
    raw_plan: str,
    updated_final_graphics_definition: str,
    affected_visual_ids: Optional[List[str]] = None,
    search_tracking: Optional[Dict[str, Dict[str, str]]] = None,
    events: Optional[List[str]] = None,
    updated_slideshow_manifest: Optional[str] = None,
    id_mapping: Optional[Dict[str, str]] = None,
) -> None:
    """Persist a segmentation revision: write the new graphics definition + audit
    columns, and reconcile per-visual review state.

    - ``affected_visual_ids`` (structural targets): cleared, since their positional
      IDs may have been reindexed by merge/split.
    - ``search_tracking`` (search targets): review action cleared (so the new visual
      re-enters pending review) AND before/after recorded in the tracking column.
    - ``id_mapping``: maps old visual IDs to new reindexed visual IDs, ensuring
      unrelated approved/reviewed visual states shift seamlessly to their new positional indices.
    """
    helpers = _import_slideshow_helpers()
    get_round_column_name = helpers["get_round_column_name"]
    safe_str = helpers["safe_str"]

    seg_feedback_col = get_round_column_name(SEGMENTATION_FEEDBACK_COLUMN, ROUND_INDEX)
    seg_plan_col = get_round_column_name(SEGMENTATION_PLAN_COLUMN, ROUND_INDEX)
    actions_col = get_round_column_name(HUMAN_REVIEW_ACTIONS_COLUMN, ROUND_INDEX)
    tracking_col = get_round_column_name(HUMAN_FEEDBACK_TRACKING_COLUMN, ROUND_INDEX)
    affected = {v.strip().upper() for v in (affected_visual_ids or []) if v.strip()}
    search_map = {k.strip().upper(): v for k, v in (search_tracking or {}).items() if k.strip()}
    # Search targets also need their (stale) review action cleared.
    clear_actions = affected | set(search_map.keys())

    def mutate(df) -> None:
        _invalidate_manifest_repair_cache(session, row_index)
        final_col = _pick_column(df, FINAL_GRAPHICS_COLUMN)
        if final_col not in df.columns:
            df[final_col] = ""
        for col in (seg_feedback_col, seg_plan_col):
            if col not in df.columns:
                df[col] = ""

        df.at[row_index, final_col] = updated_final_graphics_definition
        df.at[row_index, seg_feedback_col] = feedback
        df.at[row_index, seg_plan_col] = raw_plan

        if updated_slideshow_manifest is not None and "slideshow_manifest" in df.columns:
            df.at[row_index, "slideshow_manifest"] = updated_slideshow_manifest

        # Shift / Clear review actions so unaffected visuals stay reviewed under their new IDs
        if actions_col in df.columns:
            raw_actions = safe_str(df.at[row_index, actions_col])
            try:
                payload = json.loads(raw_actions) if raw_actions else {}
            except Exception:
                payload = {}
            if isinstance(payload, dict) and isinstance(payload.get("actions"), dict):
                old_actions = payload["actions"]
                new_actions = {}
                for old_vid, action_data in old_actions.items():
                    old_vid_upper = old_vid.strip().upper()
                    if old_vid_upper in clear_actions:
                        # Merged or revised away - clear action completely
                        continue
                    new_vid = id_mapping.get(old_vid_upper) if id_mapping else None
                    if new_vid:
                        new_actions[new_vid] = action_data
                    else:
                        new_actions[old_vid] = action_data
                payload["actions"] = new_actions
                df.at[row_index, actions_col] = json.dumps(payload, ensure_ascii=True)

        # Tracking: drop structural-affected entries; shift unaffected; record search target before/after
        if tracking_col in df.columns:
            tmap = _parse_tracking(safe_str(df.at[row_index, tracking_col]))
            new_tmap = {}
            for vid, entry in tmap.items():
                vid_upper = vid.strip().upper()
                if vid_upper in affected and vid_upper not in search_map:
                    # Clear/drop structural affected
                    continue
                new_vid = id_mapping.get(vid_upper) if id_mapping else None
                if new_vid:
                    new_tmap[new_vid] = entry
                else:
                    new_tmap[vid] = entry
            for vid, data in search_map.items():
                new_tmap[vid] = {
                    "original": data.get("original") or None,
                    "manually_selected": None,
                    "after_revision": data.get("after_revision") or None,
                    "after_regen_1": None,
                    "after_regen_2": None,
                }
            df.at[row_index, tracking_col] = _format_human_feedback_revision_tracking(new_tmap)

        ensure_manifest_sync_for_row(
            session,
            df,
            row_index,
            reason="apply_segmentation_revision_to_sheet",
            force_regenerate=False,
        )

    mutate_row_cells(session, row_index, mutate)
    if events:
        print(f"[human_feedback] segmentation revision row {row_index}: " + "; ".join(events))


def prepare_visual_revision(
    session: UserSession,
    row_index: int,
    segment_index: int,
    step_index: int,
    visual_id: str,
    vo: str,
    mode: str,
    feedback: str,
) -> None:
    action = MODE_TO_ACTION.get(mode)
    if not action:
        raise ValueError(f"Unknown revision mode: {mode}")
    write_visual_action(
        session,
        row_index=row_index,
        segment_index=segment_index,
        step_index=step_index,
        visual_id=visual_id,
        vo=vo,
        action=action,
        feedback=feedback,
        mode=mode,
    )


def apply_layout_revision_to_sheet(
    session: UserSession,
    *,
    row_index: int,
    scene_id: str,
    feedback: str,
    raw_plan: str,
    updated_slideshow_manifest: str,
    events: Optional[List[str]] = None,
) -> None:
    """Persist a layout revision: write the new slideshow_manifest and layout feedback + plan columns."""
    helpers = _import_slideshow_helpers()
    get_round_column_name = helpers["get_round_column_name"]
    
    layout_feedback_col = get_round_column_name(LAYOUT_FEEDBACK_COLUMN, ROUND_INDEX)
    layout_plan_col = get_round_column_name(LAYOUT_PLAN_COLUMN, ROUND_INDEX)

    def mutate(df) -> None:
        _invalidate_manifest_repair_cache(session, row_index)
        for col in (layout_feedback_col, layout_plan_col):
            if col not in df.columns:
                df[col] = ""
                
        if "slideshow_manifest" in df.columns:
            df.at[row_index, "slideshow_manifest"] = updated_slideshow_manifest
            
        # Parse and merge layout feedback dict
        old_feedback_str = str(df.at[row_index, layout_feedback_col] or "").strip()
        feedback_dict = {}
        if old_feedback_str:
            try:
                feedback_dict = json.loads(old_feedback_str)
                if not isinstance(feedback_dict, dict):
                    feedback_dict = {"1": old_feedback_str}
            except Exception:
                feedback_dict = {"1": old_feedback_str}
        feedback_dict[str(scene_id)] = feedback
        df.at[row_index, layout_feedback_col] = json.dumps(feedback_dict)

        # Parse and merge layout plan dict
        old_plan_str = str(df.at[row_index, layout_plan_col] or "").strip()
        plan_dict = {}
        if old_plan_str:
            try:
                plan_dict = json.loads(old_plan_str)
                if not isinstance(plan_dict, dict):
                    plan_dict = {"1": old_plan_str}
            except Exception:
                plan_dict = {"1": old_plan_str}
        plan_dict[str(scene_id)] = raw_plan
        df.at[row_index, layout_plan_col] = json.dumps(plan_dict)

        ensure_manifest_sync_for_row(
            session,
            df,
            row_index,
            reason="apply_layout_revision_to_sheet",
            force_regenerate=False,
        )

    mutate_row_cells(session, row_index, mutate)
    if events:
        print(f"[human_feedback] layout revision row {row_index} scene {scene_id}: " + "; ".join(events))

