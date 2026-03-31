import base64
import hashlib
import imghdr
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import parse_qs, urlparse

import pandas as pd
import requests
import streamlit as st
from openai import OpenAI
from PIL import UnidentifiedImageError

from services.sheets_service import get_sheet_data_and_df, get_worksheet_names
from services.sheets_service import hide_columns_by_name
from agents.graphics_definition_v2.review_agent.human_feedback_based_review_and_revise import (
    run_human_feedback_review_revise_for_all_rows,
    _format_human_feedback_revision_tracking,
)
from agents.graphics_definition_v2.review_agent.visual_columns_for_human_feedback import (
    _parse_tracking_column,
)
from agents.graphics_asset_creation.automated.automated_voiceover_reviewer import run_automation
from agents.graphics_definition_v2.aggregation_agent.aggregation_agent import (
    parse_urls_from_image_pool,
)


DEFAULT_SHEET_NAME = "Slide Chunks"
FINAL_GRAPHICS_COLUMN = "final_graphics_definition"
HUMAN_FEEDBACK_COLUMN = "human_feedback"
HUMAN_FEEDBACK_STATUS_COLUMN = "human_feedback_status"
HUMAN_FEEDBACK_TRACKING_COLUMN = "human_feedback_revision_tracking"
HUMAN_REVIEW_ACTIONS_COLUMN = "human_review_actions"
HUMAN_REVIEW_FILTER_OPTIONS = ["all", "approved", "revised", "unreviewed"]
EDITED_FILTER_OPTION = "edited"
DEFAULT_REJECT_FEEDBACK = "I did not like the visual that is currently assigned. Find and assign a better visual that is relevant for this voiceover part"
TTS_MODEL = "gpt-4o-mini-tts"
TTS_VOICE = "alloy"

ACTION_NONE = "unreviewed"
ACTION_APPROVE = "approve"
ACTION_REJECT_DRIVE_HVAC = "reject_drive_hvac"
ACTION_REJECT_ALL = "reject_all"
ACTION_REJECT_AI = "reject_ai"
AI_NO_FEEDBACK_MARKER = "No Feedback"
VIDEO_POOL_COLUMN = "video_pool_filtered"
IMAGE_POOL_COLUMN = "image_pool"

ACTION_LABELS = {
    ACTION_APPROVE: "Approve",
    ACTION_REJECT_DRIVE_HVAC: "Reject, search again only in Drive and HVAC School videos",
    ACTION_REJECT_ALL: "Reject, search again everywhere",
    ACTION_REJECT_AI: "Reject, generate with AI",
}


def get_or_create_human_feedback_column(worksheet):
    """
    Return 1-based column index for human_feedback. Creates the column header if missing.
    """
    try:
        headers = worksheet.row_values(1)
    except Exception:
        return None
    target = HUMAN_FEEDBACK_COLUMN.strip().lower()
    for i, h in enumerate(headers):
        if safe_str(h).strip().lower() == target:
            return i + 1
    col_index = len(headers) + 1
    try:
        worksheet.update_cell(1, col_index, HUMAN_FEEDBACK_COLUMN)
    except Exception:
        return None
    return col_index


def get_round_column_name(base_name, round_index):

    return f"{base_name}_{round_index + 1}"


def get_or_create_column(worksheet, column_name):
    try:
        headers = worksheet.row_values(1)
    except Exception:
        return None
    target = column_name.strip().lower()
    for i, h in enumerate(headers):
        if safe_str(h).strip().lower() == target:
            return i + 1
    col_index = len(headers) + 1
    try:
        # Some sheets have a fixed column count; extend before writing new header.
        try:
            current_cols = int(getattr(worksheet, "col_count", 0) or 0)
        except Exception:
            current_cols = 0
        if current_cols and col_index > current_cols:
            worksheet.add_cols(col_index - current_cols)
        worksheet.update_cell(1, col_index, column_name)
    except Exception as e:
        print(f"[gdv2_slideshow] Failed to create column '{column_name}' at index {col_index}: {e}")
        return None
    return col_index


def detect_current_round(df):
    def _has_non_empty_values(col_name):
        if col_name not in df.columns:
            return False
        try:
            for v in df[col_name].tolist():
                text = safe_str(v).strip()
                if text and text.lower() != "nan":
                    return True
        except Exception:
            return False
        return False

    def _round_exists(round_index):
        return any(
            get_round_column_name(base, round_index) in df.columns
            for base in (
                HUMAN_FEEDBACK_COLUMN,
                HUMAN_FEEDBACK_STATUS_COLUMN,
                HUMAN_FEEDBACK_TRACKING_COLUMN,
                HUMAN_REVIEW_ACTIONS_COLUMN,
            )
        )

    def _round_has_data(round_index):
        return any(
            _has_non_empty_values(get_round_column_name(base, round_index))
            for base in (
                HUMAN_FEEDBACK_COLUMN,
                HUMAN_FEEDBACK_STATUS_COLUMN,
                HUMAN_FEEDBACK_TRACKING_COLUMN,
                HUMAN_REVIEW_ACTIONS_COLUMN,
            )
        )

    def _round_actions_complete(round_index):
        """
        A round is complete only when actions column exists and every row has
        a parsable actions map with no unreviewed/blank action entries.
        """
        actions_col = get_round_column_name(HUMAN_REVIEW_ACTIONS_COLUMN, round_index)
        if actions_col not in df.columns:
            return False
        try:
            for raw in df[actions_col].tolist():
                text = safe_str(raw).strip()
                if not text or text.lower() == "nan":
                    return False
                actions_map = _extract_actions_map(text)
                if not actions_map:
                    return False
                for item in actions_map.values():
                    if not isinstance(item, dict):
                        return False
                    action = safe_str(item.get("action", ACTION_NONE)).strip() or ACTION_NONE
                    if action == ACTION_NONE:
                        return False
        except Exception:
            return False
        return True

    # Determine highest round index represented by existing round columns.
    highest_existing_round = 0
    while _round_exists(highest_existing_round + 1):
        highest_existing_round += 1

    # Move to next round only if latest round exists, has data, and is complete.
    # This preserves resume behavior after timeouts/partial saves.
    if _round_has_data(highest_existing_round) and _round_actions_complete(highest_existing_round):
        return highest_existing_round + 1
    return highest_existing_round


def _parse_json_map(raw):
    text = safe_str(raw).strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _extract_actions_map(raw):
    parsed = _parse_json_map(raw)
    if not parsed:
        return {}
    # Current payload shape: {"actions": {...}, "segment_modes": {...}, "round": n}
    actions = parsed.get("actions")
    if isinstance(actions, dict):
        return actions
    # Backward compatibility: payload may itself already be the actions map.
    if all(isinstance(v, dict) for v in parsed.values()):
        return parsed
    return {}


def _set_review_action(action_key, action_value):
    st.session_state[action_key] = action_value


def _col_to_a1(col_index: int) -> str:
    """Convert 1-based column index to A1 column letters."""
    if col_index <= 0:
        return "A"
    letters = []
    n = int(col_index)
    while n > 0:
        n, rem = divmod(n - 1, 26)
        letters.append(chr(65 + rem))
    return "".join(reversed(letters))


def _chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def _build_row_review_payload(slide, row_actions, current_round):
    segments = slide.get("segments", [])
    parts = []
    row_action_map_by_visual_id = {}
    segment_mode_map = {}
    reject_count = 0

    for segment in segments:
        for step in segment.get("steps", []):
            action_scope_key = f"{slide.get('row_index', 0)}_{segment['segment_index']}_{step['step_index']}_r{current_round}"
            item = row_actions.get(action_scope_key, {})
            action = safe_str(item.get("action", ACTION_NONE)).strip() or ACTION_NONE
            fb_text = safe_str(item.get("feedback", "")).strip()
            visual_id = f"S{segment['segment_index']}V{step['step_index']}"
            vo = (step.get("voiceover") or "").strip()
            # ── Check for manual asset overrides from the image/video pool ──
            # Key must match _assign_candidate_to_step which uses slide_idx_1based
            override_key = f"asset_override_{slide.get('slide_idx_1based', slide.get('row_index', 0))}_{segment['segment_index']}_{step['step_index']}"
            if override_key in st.session_state:
                assigned_asset = st.session_state[override_key]
            else:
                assigned_asset = None
                
            row_action_map_by_visual_id[visual_id] = {
                "action": action,
                "feedback": fb_text,
                "vo": vo,
                "segment": segment["segment_index"],
                "assigned_asset": assigned_asset, # Include the manual selection
            }

            if action in (ACTION_REJECT_DRIVE_HVAC, ACTION_REJECT_ALL, ACTION_REJECT_AI):
                reject_count += 1
                if action == ACTION_REJECT_AI:
                    # Keep AI generation prompt feedback empty when user leaves it blank,
                    # but persist a marker so backend still treats this as actionable.
                    effective_feedback = fb_text if fb_text else AI_NO_FEEDBACK_MARKER
                else:
                    effective_feedback = fb_text if fb_text else DEFAULT_REJECT_FEEDBACK
                parts.append(f"When VO: {vo}\nHuman Feedback: {effective_feedback}")
                segment_mode = "all" if action in (ACTION_REJECT_ALL, ACTION_REJECT_AI) else "drive_hvac"
                existing = segment_mode_map.get(str(segment["segment_index"]), "drive_hvac")
                if existing == "all" or segment_mode == "all":
                    segment_mode_map[str(segment["segment_index"])] = "all"
                else:
                    segment_mode_map[str(segment["segment_index"])] = "drive_hvac"

    # ── Build override map: (seg_index, step_index) -> new_asset_url ────────
    slide_key_prefix = slide.get("slide_idx_1based", slide.get("row_index", 0))
    override_map = {}  # {(seg_idx, step_idx): new_url}
    for seg in segments:
        for st_obj in seg.get("steps", []):
            ov_key = f"asset_override_{slide_key_prefix}_{seg['segment_index']}_{st_obj['step_index']}"
            if ov_key in st.session_state:
                override_map[(seg["segment_index"], st_obj["step_index"])] = st.session_state[ov_key]

    # ── Map manual overrides for tracking ───────────────────────────────────
    # If a manual override is present, we update the tracking history.
    tracking_text = safe_str(slide.get(get_round_column_name(HUMAN_FEEDBACK_TRACKING_COLUMN, current_round), "")).strip()
    tracking_data = _parse_tracking_column(tracking_text)
    
    for (seg_idx, step_idx), new_url in override_map.items():
        vid = f"S{seg_idx}V{step_idx}"
        if vid not in tracking_data:
            # Initialize with original if not already tracked
            original_url = ""
            for seg in segments:
                if seg.get("segment_index") == seg_idx:
                    for step in seg.get("steps", []):
                        if step.get("step_index") == step_idx:
                            original_url = (step.get("asset") or "").strip()
                            break
            tracking_data[vid] = {
                "original": original_url or None,
                "manually_selected": None,
                "after_revision": None,
                "after_regen_1": None,
                "after_regen_2": None,
            }
        tracking_data[vid]["manually_selected"] = new_url

    updated_tracking_text = _format_human_feedback_revision_tracking(tracking_data) if tracking_data else ""
    feedback_value = "\n\n".join(parts) if parts else ""
    actions_payload = json.dumps(
        {
            "actions": row_action_map_by_visual_id,
            "segment_modes": segment_mode_map,
            "round": current_round,
        },
        ensure_ascii=True,
    )

    # ── Surgically update only overridden asset URLs in the raw source text ──
    # This preserves the original XML/JSON/text formatting of the column.
    raw_def = safe_str(slide.get("final_definition_raw", "")).strip()
    updated_graphics_text = _apply_asset_overrides_to_raw(raw_def, segments, override_map)

    return feedback_value, actions_payload, reject_count, row_action_map_by_visual_id, updated_graphics_text, updated_tracking_text


def _apply_asset_overrides_to_raw(raw_text, parsed_segments, override_map):
    """
    Surgically update asset URLs in the original raw column text WITHOUT
    reformatting or changing the overall structure.

    Strategy (tried in order):
    1. XML  — replace the <asset> tag content within the N-th <visual_step>
               of the M-th <segment> (or top-level visual_steps for single-segment).
    2. JSON — already-JSON column (from a previous save): parse, update, re-dump.
    3. Fallback — plain URL swap in the raw text (best-effort).

    :param raw_text: original column value (XML, JSON, or free-text)
    :param parsed_segments: list of segment dicts (already parsed) — used for step ordering
    :param override_map: dict of {(segment_index, step_index): new_url}
    :returns: updated raw text, or original if nothing to change
    """
    if not override_map:
        return raw_text  # Nothing changed

    # ── Strategy 1: XML surgical replacement ─────────────────────────────────
    if "<" in raw_text and ">" in raw_text:
        result = _xml_surgical_asset_replace(raw_text, parsed_segments, override_map)
        if result is not None:
            return result

    # ── Strategy 2: JSON column (from prior save) ────────────────────────────
    stripped = raw_text.strip()
    if stripped.startswith("[") or stripped.startswith("{"):
        try:
            data = json.loads(stripped)
            segments_list = data if isinstance(data, list) else [data]
            for seg in segments_list:
                seg_idx = seg.get("segment_index", 1)
                for step in seg.get("steps", []):
                    step_idx = step.get("step_index", 1)
                    if (seg_idx, step_idx) in override_map:
                        step["asset"] = override_map[(seg_idx, step_idx)]
            return json.dumps(segments_list, ensure_ascii=False)
        except Exception:
            pass

    # ── Strategy 3: Plain URL swap (best-effort, last resort) ────────────────
    updated = raw_text
    for (seg_idx, step_idx), new_url in override_map.items():
        # Find the original asset URL from parsed segments and swap it
        old_url = ""
        for seg in parsed_segments:
            if seg.get("segment_index") == seg_idx:
                for step in seg.get("steps", []):
                    if step.get("step_index") == step_idx:
                        old_url = (step.get("asset") or "").strip()
                        break
        if old_url and old_url in updated:
            updated = updated.replace(old_url, new_url, 1)
    return updated


def _xml_surgical_asset_replace(raw_text, parsed_segments, override_map):
    """
    Find and replace <asset>...</asset> content for specific visual_steps within
    the XML structure, returning the modified string or None on failure.
    """
    import xml.etree.ElementTree as ET

    # Build a flat list of (seg_idx, step_idx, old_asset) in document order
    # so we know the Nth visual_step globally and per-segment.
    ordered = []  # [(seg_idx, step_idx, old_asset)]
    for seg in parsed_segments:
        for step in seg.get("steps", []):
            ordered.append((seg["segment_index"], step["step_index"], (step.get("asset") or "").strip()))

    # Use regex to find all <asset>...</asset> occurrences in document order
    asset_pattern = re.compile(
        r"(<(?:asset|graphics_to_use|graphic)\b[^>]*>)(.*?)(</(?:asset|graphics_to_use|graphic)>)",
        re.DOTALL | re.IGNORECASE,
    )

    matches = list(asset_pattern.finditer(raw_text))
    if len(matches) != len(ordered):
        # Count mismatch — can't reliably map positions to steps. Give up.
        return None

    # Build replacement map: match index -> new content
    replacements = {}  # match_index -> new_url
    for i, (seg_idx, step_idx, _old) in enumerate(ordered):
        if (seg_idx, step_idx) in override_map:
            replacements[i] = override_map[(seg_idx, step_idx)]

    if not replacements:
        return raw_text  # Nothing to replace

    # Apply replacements from right to left to preserve string offsets
    result = raw_text
    for i in sorted(replacements.keys(), reverse=True):
        m = matches[i]
        new_url = replacements[i]
        open_tag = m.group(1)
        close_tag = m.group(3)
        result = result[:m.start()] + f"{open_tag}{new_url}{close_tag}" + result[m.end():]

    return result


def _write_single_row_review(
    worksheet,
    row_index,
    feedback_value,
    actions_payload,
    round_feedback_col_idx,
    round_status_col_idx,
    round_tracking_col_idx,
    round_actions_col_idx,
    updated_graphics_json=None,
    graphics_col_idx=None,
    updated_tracking_text=None,
):
    """
    Batch update multiple columns for a single row.
    """
    sheet_row = int(row_index) + 2
    feedback_col = _col_to_a1(round_feedback_col_idx)
    status_col = _col_to_a1(round_status_col_idx)
    tracking_col = _col_to_a1(round_tracking_col_idx)
    actions_col = _col_to_a1(round_actions_col_idx)
    
    data = [
        {"range": f"{feedback_col}{sheet_row}", "values": [[feedback_value]]},
        {"range": f"{status_col}{sheet_row}", "values": [[""]]},
        {"range": f"{tracking_col}{sheet_row}", "values": [[updated_tracking_text if updated_tracking_text else ""]]},
        {"range": f"{actions_col}{sheet_row}", "values": [[actions_payload]]},
    ]
    if updated_graphics_json and graphics_col_idx:
        g_col = _col_to_a1(graphics_col_idx)
        data.append({"range": f"{g_col}{sheet_row}", "values": [[updated_graphics_json]]})
    
    worksheet.batch_update(data, value_input_option="RAW")


def _inject_review_button_layout_css():
    st.markdown(
        """
        <style>
        /* Force image containment in the 3x10 grid cards */
        /* Targets the specific containers used for previews to prevent scrollbars */
        [data-testid="stVerticalBlockBorderWrapper"] > div[style*="height: 260px"] {
            overflow: hidden !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            background-color: #f8f9fa;
        }
        
        /* Ensure st.image inside these containers behaves like 'object-fit: contain' */
        [data-testid="stVerticalBlockBorderWrapper"] > div[style*="height: 260px"] img {
            max-height: 260px !important;
            width: auto !important;
            object-fit: contain !important;
            margin: auto !important;
        }

        button[aria-label*="Approve"],
        button[aria-label*="Reject (Drive + HVAC School Videos)"],
        button[aria-label*="Reject (Search all)"],
        button[aria-label*="Reject (Generate with AI)"] {
            padding: 0.28rem 0.50rem !important;
            min-height: 2.0rem !important;
            font-size: 0.90rem !important;
        }
        button[aria-label*="Approve"] p,
        button[aria-label*="Reject (Drive + HVAC School Videos)"] p,
        button[aria-label*="Reject (Search all)"] p,
        button[aria-label*="Reject (Generate with AI)"] p {
            font-size: 0.90rem !important;
            margin: 0 !important;
            line-height: 1.15 !important;
        }
        /* Make slide expander headings more prominent for review flow. */
        div[data-testid="stExpander"] summary p {
            font-size: 1.2rem !important;
            font-weight: 700 !important;
            line-height: 1.3 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def safe_str(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value)


def find_column(df, name):
    target = name.strip().lower()
    for col in df.columns:
        if str(col).strip().lower() == target:
            return col
    return None


def is_drive_url(url):
    if not url:
        return False
    lowered = url.lower()
    return "drive.google.com" in lowered or "docs.google.com" in lowered


def extract_drive_file_id(url):
    if not url:
        return None
    patterns = [
        r"/file/d/([a-zA-Z0-9-_]+)",
        r"id=([a-zA-Z0-9-_]+)",
        r"drive.google.com/open\?id=([a-zA-Z0-9-_]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def normalize_drive_image_url(url):
    file_id = extract_drive_file_id(url)
    if not file_id:
        return url
    return f"https://drive.google.com/uc?export=view&id={file_id}"


def is_youtube_embed(url):
    if not url:
        return False
    lowered = url.lower()
    return "youtube.com/embed" in lowered or "youtu.be/" in lowered


def _first_http_url_in_text(text):
    """First URL token; strips trailing junk e.g. '...end=20 (use the image at 19s)'."""
    if not text:
        return None
    m = re.search(r"https?://[^\s]+", str(text).strip())
    if not m:
        return None
    return m.group(0).rstrip(".,);\"'")


def parse_youtube_embed(url):
    if not url:
        return None
    url = _first_http_url_in_text(url) or url
    parsed = urlparse(url)
    video_id = ""
    if "youtube.com" in parsed.netloc and "/embed/" in parsed.path:
        video_id = parsed.path.split("/embed/")[-1].split("/")[0]
    elif "youtu.be" in parsed.netloc:
        video_id = parsed.path.lstrip("/").split("/")[0]
    if not video_id:
        return None
    query = parse_qs(parsed.query)
    start = query.get("start", [None])[0]
    end = query.get("end", [None])[0]
    try:
        start_val = float(start) if start is not None else 0.0
    except ValueError:
        start_val = 0.0
    try:
        end_val = float(end) if end is not None else None
    except ValueError:
        end_val = None
    return {
        "video_id": video_id,
        "start": start_val,
        "end": end_val,
    }


def to_youtube_watch_url(url):
    """Convert YouTube embed/short/watch links to canonical watch URL with start time."""
    meta = parse_youtube_embed(url)
    if not meta or not meta.get("video_id"):
        return url
    video_id = meta["video_id"]
    start_seconds = int(meta.get("start") or 0)
    watch_url = f"https://www.youtube.com/watch?v={video_id}"
    if start_seconds > 0:
        watch_url += f"&t={start_seconds}s"
    return watch_url


def detect_asset_type(asset):
    if not asset:
        return "unknown"
    if is_youtube_embed(asset):
        return "video"
    lowered = asset.lower()
    if is_drive_url(asset):
        return "image"
    if re.search(r"\.(png|jpe?g|gif|webp)(\?|$)", lowered):
        return "image"
    return "image"


def extract_text_from_tag(element, tag_names):
    if element is None:
        return ""
    for child in element.iter():
        tag = getattr(child, "tag", "")
        if not tag:
            continue
        if tag.lower() in tag_names:
            text = "".join(child.itertext()).strip()
            if text:
                return text
    return ""


def parse_xml_segments(text):
    if not text or "<" not in text:
        return []
    cleaned = text.strip()
    root = None
    try:
        root = _safe_xml_parse(cleaned)
    except Exception:
        root = None
    if root is None:
        return []

    segments = []
    segment_elements = [el for el in root.iter() if getattr(el, "tag", "").lower() == "segment"]
    if segment_elements:
        for seg_idx, seg in enumerate(segment_elements, start=1):
            steps = _extract_visual_steps(seg)
            if steps:
                segments.append(_build_segment(seg_idx, steps))
        return segments

    fgd_elements = [el for el in root.iter() if getattr(el, "tag", "").lower() == "final_graphics_definition"]
    if fgd_elements:
        for seg_idx, fgd in enumerate(fgd_elements, start=1):
            steps = _extract_visual_steps(fgd)
            if steps:
                segments.append(_build_segment(seg_idx, steps))
        return segments

    steps = _extract_visual_steps(root)
    if steps:
        segments.append(_build_segment(1, steps))
    return segments


def _safe_xml_parse(text):
    import xml.etree.ElementTree as ET

    try:
        return ET.fromstring(text)
    except ET.ParseError:
        pass
    try:
        return ET.fromstring(f"<root>{text}</root>")
    except ET.ParseError:
        pass
    sanitized = re.sub(r"&(?![a-zA-Z]+;|#\d+;|#x[0-9A-Fa-f]+;)", "&amp;", text)
    return ET.fromstring(f"<root>{sanitized}</root>")


def _extract_visual_steps(element):
    steps = []
    visual_steps = [el for el in element.iter() if getattr(el, "tag", "").lower() == "visual_step"]
    for idx, step_el in enumerate(visual_steps, start=1):
        voiceover = extract_text_from_tag(step_el, {"voiceover_part", "voiceover", "vo_text"})
        instruction = extract_text_from_tag(step_el, {"visual_instruction", "instruction"})
        asset = extract_text_from_tag(step_el, {"asset", "graphics_to_use", "graphic"})
        justification = extract_text_from_tag(step_el, {"selection_justification", "justification"})
        steps.append(
            {
                "step_index": idx,
                "voiceover": voiceover,
                "instruction": instruction,
                "asset": asset,
                "justification": justification,
            }
        )
    return steps


def parse_formatted_segments(text):
    if not text:
        return []
    segment_matches = list(re.finditer(r"SEGMENT\s+(\d+)", text, re.IGNORECASE))
    if not segment_matches:
        return []
    segments = []
    for idx, match in enumerate(segment_matches):
        seg_num = int(match.group(1))
        start = match.end()
        end = segment_matches[idx + 1].start() if idx + 1 < len(segment_matches) else len(text)
        segment_text = text[start:end]
        steps = _parse_formatted_steps(segment_text)
        if steps:
            segments.append(_build_segment(seg_num, steps))
    return segments


def _parse_formatted_steps(segment_text):
    if not segment_text:
        return []
    blocks = re.split(r"\n\s*-{2,}\s*\n", segment_text)
    steps = []
    for block in blocks:
        parsed = _parse_formatted_block(block)
        if parsed:
            steps.append(parsed)
    return steps


def _parse_formatted_block(block):
    labels = {
        "when vo": "voiceover",
        "visual instructions": "instruction",
        "graphics to use": "asset",
        "selection justification": "justification",
    }
    data = {"voiceover": "", "instruction": "", "asset": "", "justification": ""}
    current_key = None
    for raw_line in block.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        lowered = line.lower()
        matched = False
        for label, key in labels.items():
            prefix = f"{label}:"
            if lowered.startswith(prefix):
                value = line[len(prefix):].strip()
                if key == "voiceover":
                    value = value.strip().strip('"').strip("'")
                data[key] = value
                current_key = key
                matched = True
                break
        if matched:
            continue
        if current_key:
            if current_key == "asset" and re.match(
                r"^\(\s*use the image at\b", line, re.IGNORECASE
            ):
                continue
            data[current_key] = f"{data[current_key]} {line}".strip()
    if any(data.values()):
        return {
            "step_index": 0,
            "voiceover": data["voiceover"],
            "instruction": data["instruction"],
            "asset": data["asset"],
            "justification": data["justification"],
        }
    return None


def parse_graphics_definition(text):
    cleaned = safe_str(text).strip()
    if not cleaned:
        return []
    segments = parse_xml_segments(cleaned)
    if segments:
        return segments
    segments = parse_formatted_segments(cleaned)
    if segments:
        return segments
    fallback_step = _fallback_step_from_text(cleaned)
    if fallback_step:
        return [_build_segment(1, [fallback_step])]
    return []


def _primary_asset_url(asset_text: str) -> str:
    """
    Extract the stable 'primary' URL from an asset field.

    The asset text can accidentally include extra labels/lines after regen, so
    raw string equality produces false positives. We normalize comparison to
    only the first http(s) URL when present.
    """
    s = safe_str(asset_text).strip()
    s = re.sub(r"\s+", " ", s)
    m = re.search(r"https?://[^\s)>\"]+", s)
    return m.group(0).strip() if m else s


def _fallback_step_from_text(text):
    url_match = re.search(r"https?://[^\s)>\"]+", text)
    asset = url_match.group(0) if url_match else ""
    return {
        "step_index": 1,
        "voiceover": "",
        "instruction": text.strip(),
        "asset": asset,
        "justification": "",
    }


def _build_segment(segment_index, steps):
    normalized_steps = []
    for idx, step in enumerate(steps, start=1):
        step_copy = dict(step)
        step_copy["step_index"] = idx
        normalized_steps.append(step_copy)
    return {"segment_index": segment_index, "steps": normalized_steps}


INSPECTOR_IMAGE_CACHE_KEY = "gdv2_inspector_image_cache"
INSPECTOR_VIDEO_HTML_CACHE_KEY = "gdv2_inspector_video_html_cache"

# Streamlit >=1.33: fragment isolates reruns so feedback text_areas don't remount video iframes
try:
    _st_fragment = getattr(st, "fragment", None)
except Exception:
    _st_fragment = None

# If streamlit version is old and doesn't have fragment, make it a no-op decorator
if _st_fragment is None:
    def _st_fragment(func):
        return func


def _clear_inspector_image_cache():
    """Clear cached inspector images when sheet/worksheet/data changes."""
    st.session_state.pop(INSPECTOR_IMAGE_CACHE_KEY, None)


def _get_inspector_image_bytes(asset, drive):
    """
    Return image bytes for an asset URL, downloading at most once per URL per session.
    Reuses cache on Streamlit reruns (e.g. feedback text_area) so typing doesn't
    re-hit Drive/HTTP for every image again.
    """
    if not asset or not str(asset).strip():
        return None
    cache_key = str(asset).strip()
    cache = st.session_state.setdefault(INSPECTOR_IMAGE_CACHE_KEY, {})
    if cache_key in cache:
        return cache[cache_key]

    image_bytes = None
    if is_drive_url(asset) and drive is not None:
        image_bytes = download_image_bytes(asset, drive)
    if not image_bytes:
        display_url = normalize_drive_image_url(asset) if is_drive_url(asset) else asset
        if display_url:
            # Non-Drive or Drive fallback: fetch once and cache so reruns don't re-request
            image_bytes = download_image_bytes(display_url, drive)

    if image_bytes:
        cache[cache_key] = image_bytes
    return image_bytes


def download_image_bytes(url, drive):
    if not url:
        return None
    if is_drive_url(url):
        file_id = extract_drive_file_id(url)
        if file_id and drive is not None:
            try:
                drive_file = drive.CreateFile({"id": file_id})
                with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                    tmp_path = tmp_file.name
                drive_file.GetContentFile(tmp_path)
                with open(tmp_path, "rb") as handle:
                    data = handle.read()
                os.remove(tmp_path)
                return data
            except Exception:
                return None
        url = normalize_drive_image_url(url)
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.content
    except Exception:
        return None


def image_bytes_to_data_uri(image_bytes):
    if not image_bytes:
        return ""
    image_type = imghdr.what(None, h=image_bytes) or "jpeg"
    if image_type == "jpg":
        image_type = "jpeg"
    mime = f"image/{image_type}"
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _get_cached_data_uri(url, cache):
    """
    Get or create a data URI for a URL in the cache. 
    Caches the string itself to avoid re-encoding on every render.
    """
    uri_key = f"data_uri_{url}"
    if uri_key in cache:
        return cache[uri_key]
    
    img_bytes = cache.get(url)
    if img_bytes:
        uri = image_bytes_to_data_uri(img_bytes)
        cache[uri_key] = uri
        return uri
    return None



@st.cache_resource
def get_openai_client():
    return OpenAI()


def generate_tts_audio(text, client):
    payload = text.strip() if text else ""
    if not payload:
        payload = " "
    response = client.audio.speech.create(
        model=TTS_MODEL,
        voice=TTS_VOICE,
        input=payload,
        response_format="mp3",
    )
    return response.content


def audio_bytes_to_data_uri(audio_bytes):
    if not audio_bytes:
        return ""
    encoded = base64.b64encode(audio_bytes).decode("ascii")
    return f"data:audio/mpeg;base64,{encoded}"


def build_slides_from_df(df, column_map):
    slides = []
    for idx, row in df.iterrows():
        final_def = safe_str(row.get(column_map["final_def"], "")).strip()
        segments = parse_graphics_definition(final_def)
        slide = {
            "row_index": idx,
            "topic": safe_str(row.get(column_map["topic"], "")),
            "subtopic": safe_str(row.get(column_map["subtopic"], "")),
            "slide_title": safe_str(row.get(column_map["slide_title"], "")),
            "slide_chunk": safe_str(row.get(column_map["slide_chunk"], "")),
            "final_definition_raw": final_def,
            "segments": segments,
        }
        for col_name in df.columns:
            slide[col_name] = row.get(col_name, "")
        slides.append(slide)
    return slides


def flatten_steps(slides):
    flat_steps = []
    total_slides = len(slides)
    for slide_idx, slide in enumerate(slides, start=1):
        segments = slide.get("segments", [])
        for seg_idx, segment in enumerate(segments, start=1):
            steps = segment.get("steps", [])
            for step_idx, step in enumerate(steps, start=1):
                asset = safe_str(step.get("asset", "")).strip()
                asset_type = detect_asset_type(asset)
                flat_steps.append(
                    {
                        "flat_index": len(flat_steps),
                        "slide_index": slide_idx,
                        "slide_total": total_slides,
                        "segment_index": seg_idx,
                        "segment_total": len(segments),
                        "step_index": step_idx,
                        "step_total": len(steps),
                        "voiceover": safe_str(step.get("voiceover", "")),
                        "instruction": safe_str(step.get("instruction", "")),
                        "justification": safe_str(step.get("justification", "")),
                        "asset": asset,
                        "asset_type": asset_type,
                    }
                )
    return flat_steps


def compute_slideshow_key(steps):
    key_payload = [
        (step.get("voiceover", ""), step.get("asset", ""), step.get("asset_type", ""))
        for step in steps
    ]
    digest = hashlib.md5(json.dumps(key_payload, sort_keys=True).encode("utf-8")).hexdigest()
    return digest


def prepare_slideshow_assets(steps, drive, openai_client):
    """
    Prepare all slideshow assets with parallel TTS and image loading for faster preloading.
    """
    tts_cache = st.session_state.setdefault("gdv2_tts_cache", {})
    image_cache = st.session_state.setdefault("gdv2_image_cache", {})
    
    total = len(steps)
    progress = st.progress(0.0)
    status = st.empty()
    
    # Separate items that need TTS vs already cached
    tts_needed = []
    images_needed = []
    
    for idx, step in enumerate(steps):
        voiceover = step.get("voiceover", "").strip()
        if voiceover and voiceover not in tts_cache:
            tts_needed.append((idx, voiceover))
        
        asset = step.get("asset", "").strip()
        asset_type = step.get("asset_type", "unknown")
        if asset_type == "image" and asset and asset not in image_cache:
            images_needed.append((idx, asset))
    
    completed = 0
    total_tasks = len(tts_needed) + len(images_needed) + len(steps)
    
    # Generate TTS in parallel (up to 8 concurrent requests)
    if tts_needed:
        status.write(f"🎙️ Generating {len(tts_needed)} narration audio clips in parallel...")
        
        def generate_single_tts(item):
            idx, voiceover = item
            try:
                audio_bytes = generate_tts_audio(voiceover, openai_client)
                return voiceover, audio_bytes_to_data_uri(audio_bytes)
            except Exception as e:
                print(f"TTS error for step {idx}: {e}")
                return voiceover, ""
        
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(generate_single_tts, item): item for item in tts_needed}
            for future in as_completed(futures):
                try:
                    voiceover, audio_uri = future.result()
                    if audio_uri:
                        tts_cache[voiceover] = audio_uri
                except Exception as e:
                    print(f"TTS future error: {e}")
                completed += 1
                progress.progress(completed / total_tasks)
    
    # Download images in parallel (up to 6 concurrent requests)
    if images_needed:
        status.write(f"🖼️ Loading {len(images_needed)} images in parallel...")
        
        def download_single_image(item):
            idx, asset = item
            try:
                resolved = normalize_drive_image_url(asset) if is_drive_url(asset) else asset
                image_bytes = download_image_bytes(resolved, drive)
                return asset, image_bytes_to_data_uri(image_bytes) if image_bytes else resolved
            except Exception as e:
                print(f"Image error for step {idx}: {e}")
                return asset, ""
        
        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = {executor.submit(download_single_image, item): item for item in images_needed}
            for future in as_completed(futures):
                try:
                    asset, image_uri = future.result()
                    if image_uri:
                        image_cache[asset] = image_uri
                except Exception as e:
                    print(f"Image future error: {e}")
                completed += 1
                progress.progress(completed / total_tasks)
    
    # Now assemble the prepared steps (fast, just lookups)
    status.write("📦 Assembling slideshow data...")
    prepared_steps = []
    for idx, step in enumerate(steps):
        voiceover = step.get("voiceover", "").strip()
        audio_data_uri = tts_cache.get(voiceover, "")
        
        asset = step.get("asset", "").strip()
        asset_type = step.get("asset_type", "unknown")
        image_data_uri = ""
        video_meta = None
        
        if asset_type == "image":
            image_data_uri = image_cache.get(asset, "")
            if not image_data_uri and asset:
                resolved = normalize_drive_image_url(asset) if is_drive_url(asset) else asset
                image_data_uri = resolved
        elif asset_type == "video":
            video_meta = parse_youtube_embed(asset)
        
        prepared = dict(step)
        prepared["audio_data_uri"] = audio_data_uri
        prepared["image_data_uri"] = image_data_uri
        prepared["video_meta"] = video_meta
        prepared_steps.append(prepared)
        completed += 1
        progress.progress(completed / total_tasks)
    
    status.empty()
    progress.empty()
    return prepared_steps


def build_slideshow_html(prepared_steps):
    steps_payload = []
    for step in prepared_steps:
        video_meta = step.get("video_meta") or {}
        steps_payload.append(
            {
                "flatIndex": step.get("flat_index"),
                "slideIndex": step.get("slide_index"),
                "slideTotal": step.get("slide_total"),
                "segmentIndex": step.get("segment_index"),
                "segmentTotal": step.get("segment_total"),
                "stepIndex": step.get("step_index"),
                "stepTotal": step.get("step_total"),
                "voiceover": step.get("voiceover", ""),
                "instruction": step.get("instruction", ""),
                "justification": step.get("justification", ""),
                "assetType": step.get("asset_type", "unknown"),
                "assetUrl": step.get("asset", ""),
                "imageData": step.get("image_data_uri", ""),
                "audioData": step.get("audio_data_uri", ""),
                "videoId": video_meta.get("video_id"),
                "videoStart": video_meta.get("start", 0.0) if video_meta else 0.0,
                "videoEnd": video_meta.get("end", None) if video_meta else None,
            }
        )
    payload = json.dumps({"steps": steps_payload}, ensure_ascii=True)

    return f"""
    <div id="gdv2-slideshow">
      <div id="gdv2-preload">
        <div id="gdv2-preload-text">Preparing assets...</div>
        <div id="gdv2-preload-bar"><div id="gdv2-preload-bar-inner"></div></div>
      </div>
      <div id="gdv2-status"></div>
      <div id="gdv2-progress">
        <div id="gdv2-progress-bar"></div>
      </div>
      <div id="gdv2-voiceover"></div>
      <div id="gdv2-visual">
        <img id="gdv2-image" />
        <div id="gdv2-video-container"></div>
      </div>
      <div id="gdv2-controls">
        <button id="gdv2-start">Start</button>
        <button id="gdv2-pause" disabled>Pause</button>
        <button id="gdv2-resume" disabled>Resume</button>
        <button id="gdv2-prev-slide" disabled>Previous Slide</button>
        <button id="gdv2-next-slide" disabled>Next Slide</button>
        <button id="gdv2-prev-step" disabled>Previous VO Part</button>
        <button id="gdv2-next-step" disabled>Next VO Part</button>
      </div>
    </div>
    <style>
      #gdv2-slideshow {{
        font-family: Arial, sans-serif;
        color: #111;
        padding: 8px;
      }}
      #gdv2-preload {{
        border: 1px solid #ccc;
        padding: 8px;
        margin-bottom: 8px;
      }}
      #gdv2-preload-bar {{
        background: #eee;
        height: 8px;
        border-radius: 4px;
        overflow: hidden;
        margin-top: 6px;
      }}
      #gdv2-preload-bar-inner {{
        background: #1f77b4;
        height: 100%;
        width: 0%;
      }}
      #gdv2-status {{
        margin: 8px 0;
        font-weight: 600;
      }}
      #gdv2-progress {{
        width: 100%;
        max-width: 980px;
        height: 8px;
        background: #e6e9ee;
        border-radius: 999px;
        margin: 6px auto 10px;
        overflow: hidden;
      }}
      #gdv2-progress-bar {{
        height: 100%;
        width: 0%;
        background: #2a6fdf;
        transition: width 120ms linear;
      }}
      #gdv2-voiceover {{
        margin: 12px 0 8px;
        font-size: 18px;
        font-weight: 600;
        background: #ffffff;
        color: #222;
        border: 1px solid #d7dce2;
        border-radius: 10px;
        padding: 12px 16px;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.08);
      }}
      #gdv2-visual {{
        display: flex;
        align-items: center;
        justify-content: center;
        background: #f7f7f7;
        border: 1px solid #ddd;
        width: 100%;
        max-width: 980px;
        aspect-ratio: 16 / 9;
        margin: 0 auto;
        overflow: hidden;
        position: relative;
      }}
      #gdv2-image {{
        max-width: 100%;
        max-height: 100%;
        width: auto;
        height: auto;
        object-fit: contain;
        display: none;
      }}
      #gdv2-video-container {{
        position: absolute;
        inset: 0;
        width: 100%;
        height: 100%;
        display: none;
      }}
      #gdv2-video-container.active {{
        display: block;
      }}
      .gdv2-video-frame {{
        width: 100%;
        height: 100%;
        position: absolute;
        inset: 0;
        opacity: 0;
        visibility: hidden;
        transition: opacity 120ms linear;
      }}
      .gdv2-video-frame.active {{
        opacity: 1;
        visibility: visible;
        z-index: 1;
      }}
      #gdv2-controls {{
        margin-top: 10px;
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
      }}
      #gdv2-controls button {{
        padding: 6px 10px;
        font-size: 14px;
        cursor: pointer;
      }}
    </style>
    <script>
      const payload = {payload};
      const steps = payload.steps || [];
      const audioElements = [];
      const imageElements = [];
      const videoPlayers = {{}};
      let readyCount = 0;
      let totalToLoad = 0;
      let currentIndex = -1;
      let isPlaying = false;
      let isPaused = false;
      let narrationDone = false;
      let videoEnded = false;
      let currentAudio = null;
      let currentVideo = null;
      let videoCheckInterval = null;

      const preloadText = document.getElementById("gdv2-preload-text");
      const preloadBar = document.getElementById("gdv2-preload-bar-inner");
      const statusEl = document.getElementById("gdv2-status");
      const progressBarEl = document.getElementById("gdv2-progress-bar");
      const voiceoverEl = document.getElementById("gdv2-voiceover");
      const imageEl = document.getElementById("gdv2-image");
      const videoContainer = document.getElementById("gdv2-video-container");

      const startBtn = document.getElementById("gdv2-start");
      const pauseBtn = document.getElementById("gdv2-pause");
      const resumeBtn = document.getElementById("gdv2-resume");
      const prevSlideBtn = document.getElementById("gdv2-prev-slide");
      const nextSlideBtn = document.getElementById("gdv2-next-slide");
      const prevStepBtn = document.getElementById("gdv2-prev-step");
      const nextStepBtn = document.getElementById("gdv2-next-step");

      function markReady() {{
        readyCount += 1;
        if (totalToLoad > 0) {{
          preloadBar.style.width = `${{Math.min(100, (readyCount / totalToLoad) * 100)}}%`;
        }}
        if (readyCount >= totalToLoad) {{
          preloadText.textContent = "Assets ready.";
          startBtn.disabled = steps.length === 0;
        }}
      }}

      function setupAudio() {{
        steps.forEach((step, index) => {{
          const audio = new Audio(step.audioData || "");
          audio.preload = "auto";
          audio.addEventListener("loadedmetadata", () => {{
            step.audioDuration = audio.duration || 0;
            markReady();
          }});
          audio.addEventListener("error", () => {{
            step.audioDuration = 0;
            markReady();
          }});
          audioElements[index] = audio;
        }});
      }}

      function setupImages() {{
        steps.forEach((step) => {{
          if (step.assetType !== "image" || !step.imageData) {{
            return;
          }}
          const img = new Image();
          img.onload = markReady;
          img.onerror = markReady;
          img.src = step.imageData;
          imageElements.push(img);
        }});
      }}

      function buildVideoContainers() {{
        steps.forEach((step, index) => {{
          if (step.assetType !== "video" || !step.videoId) {{
            return;
          }}
          const frame = document.createElement("div");
          frame.id = `gdv2-video-${{index}}`;
          frame.className = "gdv2-video-frame";
          videoContainer.appendChild(frame);
        }});
      }}

      function onYouTubeIframeAPIReady() {{
        steps.forEach((step, index) => {{
          if (step.assetType !== "video" || !step.videoId) {{
            return;
          }}
          const player = new YT.Player(`gdv2-video-${{index}}`, {{
            videoId: step.videoId,
            playerVars: {{
              start: step.videoStart || 0,
              end: step.videoEnd || undefined,
              controls: 0,
              rel: 0,
              enablejsapi: 1,
              playsinline: 1,
              modestbranding: 1,
              mute: 1
            }},
            events: {{
              onReady: (event) => {{
                try {{
                  const cueArgs = {{
                    videoId: step.videoId,
                    startSeconds: step.videoStart || 0
                  }};
                  if (step.videoEnd !== null && step.videoEnd !== undefined) {{
                    cueArgs.endSeconds = step.videoEnd;
                  }}
                  event.target.cueVideoById(cueArgs);
                }} catch (e) {{}}
                markReady();
              }},
              onError: () => markReady()
            }}
          }});
          videoPlayers[index] = player;
        }});
      }}

      function loadYouTubeApi() {{
        if (!steps.some(step => step.assetType === "video")) {{
          return;
        }}
        const tag = document.createElement("script");
        tag.src = "https://www.youtube.com/iframe_api";
        document.body.appendChild(tag);
      }}

      function updateStatus(step) {{
        if (!step) {{
          statusEl.textContent = "";
          progressBarEl.style.width = "0%";
          return;
        }}
        statusEl.textContent =
          `Slide ${{step.slideIndex}} of ${{step.slideTotal}}, Segment ${{step.segmentIndex}} of ${{step.segmentTotal}}, VO Part ${{step.stepIndex}} of ${{step.stepTotal}}`;
        const totalSteps = steps.length || 1;
        const progressPercent = ((currentIndex + 1) / totalSteps) * 100;
        progressBarEl.style.width = `${{Math.min(100, Math.max(0, progressPercent))}}%`;
      }}

      function showVisual(step, index) {{
        imageEl.style.display = "none";
        videoContainer.classList.remove("active");
        Array.from(videoContainer.children).forEach((child) => {{
          child.classList.remove("active");
        }});
        if (step.assetType === "image" && step.imageData) {{
          imageEl.src = step.imageData;
          imageEl.style.display = "block";
        }} else if (step.assetType === "video") {{
          const frame = document.getElementById(`gdv2-video-${{index}}`);
          if (frame) {{
            frame.classList.add("active");
            videoContainer.classList.add("active");
          }}
        }}
      }}

      function clearTimers() {{
        if (videoCheckInterval) {{
          clearInterval(videoCheckInterval);
          videoCheckInterval = null;
        }}
      }}

      function setupVideoMonitoring(step, index) {{
        if (!step || step.assetType !== "video") {{
          return;
        }}
        const player = videoPlayers[index];
        if (!player || step.videoEnd === null || step.videoEnd === undefined) {{
          return;
        }}
        const clipLen = Number(step.videoEnd) - Number(step.videoStart || 0);
        const loopShortClip =
          clipLen > 0 &&
          clipLen <= 30;
        videoEnded = false;
        videoCheckInterval = setInterval(() => {{
          let currentTime = 0;
          try {{
            currentTime = player.getCurrentTime();
          }} catch (e) {{
            return;
          }}
          if (currentTime >= step.videoEnd - 0.12) {{
            if (loopShortClip) {{
              try {{
                player.seekTo(step.videoStart || 0, true);
                player.playVideo();
              }} catch (e) {{}}
            }} else {{
              try {{
                player.pauseVideo();
                player.seekTo(step.videoEnd, true);
              }} catch (e) {{}}
              videoEnded = true;
              clearTimers();
              if (narrationDone && clipLen > (step.audioDuration || 0)) {{
                advanceStep(1);
              }}
            }}
          }}
        }}, 100);
      }}

      function playStep(index) {{
        clearTimers();
        if (index < 0 || index >= steps.length) {{
          return;
        }}
        currentIndex = index;
        const step = steps[index];
        narrationDone = false;
        videoEnded = false;
        updateStatus(step);
        voiceoverEl.textContent = step.voiceover || "";
        showVisual(step, index);

        currentAudio = audioElements[index];
        if (currentAudio) {{
          currentAudio.pause();
          currentAudio.currentTime = 0;
          currentAudio.onended = () => {{
            narrationDone = true;
            if (step.assetType !== "video") {{
              advanceStep(1);
              return;
            }}
            const videoDuration = (step.videoEnd || 0) - (step.videoStart || 0);
            if (videoDuration <= (step.audioDuration || 0) || videoEnded) {{
              advanceStep(1);
            }}
          }};
          currentAudio.play();
        }} else {{
          narrationDone = true;
        }}

        if (step.assetType === "video") {{
          currentVideo = videoPlayers[index];
          if (currentVideo) {{
            try {{
              currentVideo.mute();
              if (
                step.videoEnd !== null &&
                step.videoEnd !== undefined &&
                step.videoId
              ) {{
                currentVideo.loadVideoById({{
                  videoId: step.videoId,
                  startSeconds: step.videoStart || 0,
                  endSeconds: step.videoEnd,
                }});
              }} else {{
                currentVideo.seekTo(step.videoStart || 0, true);
                currentVideo.playVideo();
              }}
            }} catch (e) {{}}
            setupVideoMonitoring(step, index);
          }}
        }} else {{
          currentVideo = null;
        }}
      }}

      function advanceStep(delta) {{
        if (!isPlaying) {{
          return;
        }}
        const nextIndex = currentIndex + delta;
        if (nextIndex < 0 || nextIndex >= steps.length) {{
          isPlaying = false;
          pauseBtn.disabled = true;
          resumeBtn.disabled = true;
          prevSlideBtn.disabled = false;
          nextSlideBtn.disabled = false;
          prevStepBtn.disabled = false;
          nextStepBtn.disabled = false;
          return;
        }}
        playStep(nextIndex);
      }}

      function pausePlayback() {{
        if (!isPlaying) {{
          return;
        }}
        isPaused = true;
        if (currentAudio) {{
          currentAudio.pause();
        }}
        if (currentVideo) {{
          try {{
            currentVideo.pauseVideo();
          }} catch (e) {{}}
        }}
        clearTimers();
        pauseBtn.disabled = true;
        resumeBtn.disabled = false;
      }}

      function resumePlayback() {{
        if (!isPlaying) {{
          return;
        }}
        isPaused = false;
        if (currentAudio) {{
          currentAudio.play();
        }}
        const step = steps[currentIndex];
        if (step && step.assetType === "video" && currentVideo) {{
          try {{
            currentVideo.playVideo();
          }} catch (e) {{}}
          setupVideoMonitoring(step, currentIndex);
        }}
        pauseBtn.disabled = false;
        resumeBtn.disabled = true;
      }}

      function getSlideStartIndex(targetSlide) {{
        let found = null;
        steps.forEach((step, index) => {{
          if (step.slideIndex === targetSlide && found === null) {{
            found = index;
          }}
        }});
        return found;
      }}

      function navigateSlide(delta) {{
        if (currentIndex < 0 || steps.length === 0) {{
          return;
        }}
        const currentSlide = steps[currentIndex].slideIndex;
        const targetSlide = currentSlide + delta;
        if (targetSlide < 1 || targetSlide > steps[currentIndex].slideTotal) {{
          return;
        }}
        const startIndex = getSlideStartIndex(targetSlide);
        if (startIndex !== null) {{
          playStep(startIndex);
        }}
      }}

      startBtn.addEventListener("click", () => {{
        if (steps.length === 0 || startBtn.disabled) {{
          return;
        }}
        isPlaying = true;
        isPaused = false;
        pauseBtn.disabled = false;
        resumeBtn.disabled = true;
        prevSlideBtn.disabled = false;
        nextSlideBtn.disabled = false;
        prevStepBtn.disabled = false;
        nextStepBtn.disabled = false;
        playStep(0);
      }});
      pauseBtn.addEventListener("click", pausePlayback);
      resumeBtn.addEventListener("click", resumePlayback);
      prevStepBtn.addEventListener("click", () => advanceStep(-1));
      nextStepBtn.addEventListener("click", () => advanceStep(1));
      prevSlideBtn.addEventListener("click", () => navigateSlide(-1));
      nextSlideBtn.addEventListener("click", () => navigateSlide(1));

      function init() {{
        totalToLoad = steps.length;
        const imageLoadCount = steps.filter(step => step.assetType === "image" && step.imageData).length;
        const videoLoadCount = steps.filter(step => step.assetType === "video" && step.videoId).length;
        totalToLoad = steps.length + imageLoadCount + videoLoadCount;
        if (steps.length === 0) {{
          preloadText.textContent = "No steps to play.";
          return;
        }}
        setupAudio();
        setupImages();
        buildVideoContainers();
        loadYouTubeApi();
      }}

      window.onYouTubeIframeAPIReady = onYouTubeIframeAPIReady;
      init();
    </script>
    """


def render_looping_youtube_embed(url, key, height=320):
    meta = parse_youtube_embed(url)
    if not meta:
        return "<div>Unsupported video URL</div>"
    
    video_id = meta["video_id"]
    start = meta.get("start") or 0
    end = meta.get("end")
    
    # Robust script that works with multiple players on the same page
    html = f"""
    <div id="yt-container-{key}" style="width:100%; height:{height}px; background:#000; overflow:hidden; border-radius:4px;">
      <div id="yt-player-{key}"></div>
    </div>
    <script>
      (function() {{
        function initPlayer() {{
          new YT.Player("yt-player-{key}", {{
            height: "{height}",
            width: "100%",
            videoId: "{video_id}",
            playerVars: {{
              start: {int(start)},
              end: {int(end) if end is not None else "undefined"},
              autoplay: 1,
              controls: 1,
              rel: 0,
              playsinline: 1,
              mute: 1
            }},
            events: {{
              onReady: function(event) {{
                try {{ event.target.playVideo(); }} catch (e) {{}}
              }},
              onStateChange: function(event) {{
                 // Backup check if the loop missed the end
                 const endT = {int(end) if end is not None else "null"};
                 if (event.data === YT.PlayerState.ENDED || (endT && event.target.getCurrentTime() >= endT - 0.2)) {{
                    event.target.seekTo({int(start)}, true);
                    event.target.playVideo();
                 }}
              }}
            }}
          }});
        }}

        // Independent polling instead of relying on a single global callback
        let retryCount = 0;
        function tryInit() {{
          if (window.YT && window.YT.Player) {{
            initPlayer();
          }} else if (retryCount < 50) {{
            retryCount++;
            setTimeout(tryInit, 200);
            
            // If we are the first to run, kick off the API load
            if (!document.getElementById("yt-iframe-api-script")) {{
               const tag = document.createElement("script");
               tag.id = "yt-iframe-api-script";
               tag.src = "https://www.youtube.com/iframe_api";
               document.body.appendChild(tag);
            }}
          }}
        }}

        tryInit();
      }})();
    </script>
    """
    return html


def _get_inspector_video_html(asset, embed_key):
    """
    Build YouTube embed HTML once per asset URL; reuse on reruns to avoid
    re-running render_looping_youtube_embed. Iframe may still remount without
    st.fragment, but string build + parse is avoided.
    """
    if not asset or not str(asset).strip():
        return ""
    cache_key = str(asset).strip()
    cache = st.session_state.setdefault(INSPECTOR_VIDEO_HTML_CACHE_KEY, {})
    if cache_key in cache:
        return cache[cache_key]
    html = render_looping_youtube_embed(asset, embed_key, height=360)
    if html:
        cache[cache_key] = html
    return html


def _render_inspector_step_visual(asset, asset_type, display_url, embed_key, drive):
    """Image/video only — no widgets. Safe to run inside st.fragment."""
    if asset_type == "image":
        if not display_url:
            display_url = normalize_drive_image_url(asset) if is_drive_url(asset) else asset
        image_bytes = _get_inspector_image_bytes(asset, drive)
        if image_bytes:
            try:
                st.image(image_bytes, use_container_width=True)
            except UnidentifiedImageError:
                # Image bytes were invalid; fall back to URL display below if available.
                image_bytes = None
            except Exception:
                # Any other image rendering issue: ignore bytes and try URL fallback.
                image_bytes = None
        elif display_url:
            st.image(display_url, use_container_width=True)
        else:
            st.warning("Image URL missing.")
        if display_url:
            st.markdown(f"[Open image]({display_url})")
    elif asset_type == "video":
        if asset:
            html = _get_inspector_video_html(asset, embed_key)
            if html:
                st.components.v1.html(html, height=360)
            st.markdown(f"[Open video]({to_youtube_watch_url(asset)})")
        else:
            st.warning("Video URL missing.")


@st.cache_resource
def _get_background_executor():
    """Shared executor for background tasks (e.g. preloading alternatives)."""
    return ThreadPoolExecutor(max_workers=5)


def _background_download_task(url, drive, cache):
    """Worker task to download image and put into cache."""
    try:
        img_bytes = download_image_bytes(url, drive)
        if img_bytes:
            cache[url] = img_bytes
            # Pre-generate data URI to save CPU cycles during swiping
            _get_cached_data_uri(url, cache)
    except Exception:
        pass


def _preload_candidate_images(slides, drive):
    """
    Asynchronously pre-warm the shared image cache for all candidate images.
    Submits tasks to a background executor and returns IMMEDIATELY to avoid blocking the UI.
    """
    if not slides or drive is None:
        return
        
    # Throttling: only scan for new candidates every 30 seconds to save CPU
    now = time.time()
    last_scan = st.session_state.get("last_alt_preload_time", 0)
    if now - last_scan < 30:
        return
    st.session_state["last_alt_preload_time"] = now

    cache = st.session_state.setdefault(INSPECTOR_IMAGE_CACHE_KEY, {})
    executor = _get_background_executor()
    seen_urls = set()
    tasks_count = 0
    
    # Identify unique Drive URLs that aren't already cached
    for slide in slides:
        image_pool_text = safe_str(slide.get("image_pool", "")).strip()
        if not image_pool_text or image_pool_text == "nan":
            continue
        segments = slide.get("segments", [])
        for segment in segments:
            seg_num = segment.get("segment_index", 1)
            try:
                candidates = parse_urls_from_image_pool(image_pool_text, seg_num)
            except Exception:
                continue
            for c in candidates:
                url = (c.get("url") or "").strip()
                if url and is_drive_url(url) and url not in cache and url not in seen_urls:
                    seen_urls.add(url)
                    # Use the shared executor to download in background - STAYS NON-BLOCKING
                    executor.submit(_background_download_task, url, drive, cache)
                    tasks_count += 1
    
    # We don't wait for completion. The main script continues and finishes quickly.



def _set_alt_start_index(key, new_val):
    st.session_state[key] = new_val

def _assign_candidate_to_step(slide_idx, seg_idx, step_idx, new_url):
    """Persist an asset override chosen from the candidate pool."""
    key = f"asset_override_{slide_idx}_{seg_idx}_{step_idx}"
    st.session_state[key] = new_url
    st.toast(f"Assigned to VO Step {step_idx}!", icon="🎯")


def _revert_asset_assignment(key):
    """Clear an asset override to return to the original."""
    if key in st.session_state:
        del st.session_state[key]
    st.toast("Reverted to original asset", icon="↩️")


def _fragment_candidate_images_expander(image_pool_text, segment_num, slide_idx, segment_steps, drive, embed_key_prefix):
    """
    Paginated UI for alternative images: shows 3 at a time, slides by 2 on click.
    Deduplicates URLs within the segment.
    """
    # ── Cache parsed candidates ──────────────────────────────────────────────
    cand_cache_key = f"cand_pool_{embed_key_prefix}"
    if cand_cache_key not in st.session_state:
        try:
            candidates = parse_urls_from_image_pool(image_pool_text, segment_num)
        except Exception:
            candidates = []
        st.session_state[cand_cache_key] = candidates
    else:
        candidates = st.session_state[cand_cache_key]

    if not candidates:
        return

    # ── Deduplicate by URL, preserving order (also cached) ─────────────────────
    unique_cache_key = f"cand_unique_{embed_key_prefix}"
    if unique_cache_key not in st.session_state:
        seen = set()
        unique = []
        for c in candidates:
            url = (c.get("url") or "").strip()
            if url and url not in seen:
                seen.add(url)
                unique.append(c)
        st.session_state[unique_cache_key] = unique
    unique = st.session_state[unique_cache_key]
    
    if not unique:
        return

    n = len(unique)
    state_key = f"alt_start_{embed_key_prefix}"
    if state_key not in st.session_state:
        st.session_state[state_key] = 0
    start_idx = st.session_state[state_key]

    # ── Bounds check and pagination logic (3x10 grid = 30 items) ────────────
    max_visible = 30
    if start_idx >= n:
        start_idx = max(0, n - max_visible)
        st.session_state[state_key] = start_idx

    with st.expander(
        f"🖼️ Alternative Images — {n} candidate{'s' if n != 1 else ''} considered",
        expanded=False,
    ):
        visible_indices = list(range(start_idx, min(start_idx + max_visible, n)))
        cache = st.session_state.get(INSPECTOR_IMAGE_CACHE_KEY, {})
        
        # Simple navigation row above the images to keep them clean
        nav_cols = st.columns([0.1, 0.8, 0.1])
        with nav_cols[0]:
            if start_idx > 0:
                if st.button("◀ Previous 30", key=f"{state_key}_prev", use_container_width=True):
                    st.session_state[state_key] = max(0, start_idx - max_visible)
                    st.rerun(scope="fragment")
        with nav_cols[2]:
            if start_idx + max_visible < n:
                if st.button("Next 30 ▶", key=f"{state_key}_next", use_container_width=True):
                    st.session_state[state_key] = min(n - 1, start_idx + max_visible)
                    st.rerun(scope="fragment")

        # ── Render 3x10 Grid ──────────────────────────────────────────────────
        for i, idx_in_list in enumerate(visible_indices):
            # Create a new row of columns for every 3 items
            if i % 3 == 0:
                grid_cols = st.columns(3, gap="small", vertical_alignment="top")
            
            curr_col = grid_cols[i % 3]
            
            cand = unique[idx_in_list]
            url = (cand.get("url") or "").strip()
            title = cand.get("title") or f"Candidate {idx_in_list + 1}"
            
            # (Fetching logic omitted for brevity in match, but included in replacement)
            img_src = _get_cached_data_uri(url, cache)
            if not img_src and is_drive_url(url):
                file_id = extract_drive_file_id(url)
                img_src = f"https://drive.google.com/thumbnail?id={file_id}&sz=w300" if file_id else ""
            
            with curr_col:
                # Wrap in a container for group logic, but use HTML for fixed-dimension image display
                # We use the Stable Thumbnail URL (not Base64) in the HTML block 
                # why? Because browsers cache these URLs perfectly, making swipes instant/cached.
                # ── Determine stable preview URL for browser caching ────────────
                if is_drive_url(url):
                    file_id = extract_drive_file_id(url)
                    display_url = f"https://drive.google.com/thumbnail?id={file_id}&sz=w400" if file_id else url
                elif is_youtube_embed(url):
                    m = re.search(r"youtube\.com/embed/([^?/]+)", url)
                    vid_id = m.group(1) if m else ""
                    display_url = f"https://img.youtube.com/vi/{vid_id}/mqdefault.jpg" if vid_id else url
                else:
                    # Regular web image link
                    display_url = url
                
            with curr_col:
                # ── Unified visual unit (Safety via st.image + Symmetry via CSS) ──
                # We use st.container(height=260) which we've globally styled to hide scrollbars
                with st.container(height=260, border=True):
                    # Use the cached binary if available (Base64 is no longer needed with st.image)
                    img_data = cache.get(url)
                    if not img_data:
                        img_data = display_url # st.image will proxy this safely
                    
                    st.image(img_data, use_container_width=True)
                
                # 2. Action popover placed directly below the image card
                with st.popover("🎯 Action", use_container_width=True):
                    st.markdown(f"**{title}**")
                    
                    # ── Option 1: Open high res link ──────────────────────────
                    if is_drive_url(url):
                        file_id = extract_drive_file_id(url)
                        open_url = f"https://drive.google.com/file/d/{file_id}/view" if file_id else url
                    elif is_youtube_embed(url):
                        open_url = to_youtube_watch_url(url)
                    else:
                        open_url = url
                    st.link_button("🔗 Open original link", open_url, use_container_width=True)
                    
                    st.divider()
                    st.caption("Assign to Voiceover Step:")
                    
                    # Unpack step tuple correctly to avoid AttributeError
                    if segment_steps:
                        for step_tuple in segment_steps:
                            # structure: (step_dict, action_scope_key, action_key, feedback_key, visual_id)
                            step_dict = step_tuple[0]
                            s_idx = step_dict.get("step_index", 1)
                            st.button(
                                f"Use for VO Part {s_idx}",
                                key=f"assign_{embed_key_prefix}_{idx_in_list}_{s_idx}",
                                on_click=_assign_candidate_to_step,
                                args=(slide_idx, segment_num, s_idx, url),
                                use_container_width=True
                            )
                    else:
                        st.info("No VO parts found.")

def _fragment_candidate_videos_expander(video_pool_text, segment_num, slide_idx, segment_steps, embed_key_prefix):
    """
    Shows alternative video candidates from the filtered video pool.
    Renders with looping players and assignment buttons.
    """
    if not video_pool_text or video_pool_text == "nan":
        return
    
    unique_cache_key = f"videocand_unique_{embed_key_prefix}"
    if unique_cache_key not in st.session_state:
        # Use existing image pool parser as the format is identical (Title | URL)
        candidates = parse_urls_from_image_pool(video_pool_text, segment_num)
        seen = set()
        unique = []
        for v in candidates:
            u = (v.get("url") or "").strip()
            if u and u not in seen:
                seen.add(u)
                unique.append(v)
        st.session_state[unique_cache_key] = unique
    
    unique = st.session_state[unique_cache_key]
    if not unique:
        return
    
    n = len(unique)
    state_key = f"vid_start_{embed_key_prefix}"
    if state_key not in st.session_state:
        st.session_state[state_key] = 0
    start_idx = st.session_state[state_key]
    
    max_visible = 30 # Upgraded to high-density 3x10 layout
    if start_idx >= n:
        start_idx = max(0, n - max_visible)
        st.session_state[state_key] = start_idx
        
    with st.expander(f"🎬 Alternative Videos — {n} segments found", expanded=False):
        visible_indices = list(range(start_idx, min(start_idx + max_visible, n)))
        
        # Navigation
        nav_cols = st.columns([0.1, 0.8, 0.1])
        with nav_cols[0]:
            if start_idx > 0:
                if st.button("◀ Prev", key=f"{state_key}_prev", use_container_width=True):
                    st.session_state[state_key] = max(0, start_idx - max_visible)
                    st.rerun(scope="fragment")
        with nav_cols[2]:
            if start_idx + max_visible < n:
                if st.button("Next ▶", key=f"{state_key}_next", use_container_width=True):
                    st.session_state[state_key] = min(n - 1, start_idx + max_visible)
                    st.rerun(scope="fragment")
        
        # Grid Rendering (3 columns)
        for i, idx_in_list in enumerate(visible_indices):
            if i % 3 == 0:
                grid_cols = st.columns(3, gap="small", vertical_alignment="top")
            
            curr_col = grid_cols[i % 3]
            v_cand = unique[idx_in_list]
            v_url = (v_cand.get("url") or "").strip()
            v_title = v_cand.get("title") or f"Video {idx_in_list + 1}"
            
            with curr_col:
                # ── Looping Player Unit ──────────────────────────────────────
                with st.container(height=320, border=True):
                    if is_youtube_embed(v_url):
                        v_html = render_looping_youtube_embed(v_url, f"altvid_{embed_key_prefix}_{idx_in_list}", height=240)
                        st.components.v1.html(v_html, height=240)
                    else:
                        st.caption("Unsupported video format")
                        st.write(v_url)
                    
                    # 🎯 Re-assignment Popover
                    with st.popover("🎯 Action", use_container_width=True):
                        st.markdown(f"**{v_title}**")
                        if is_youtube_embed(v_url):
                             st.link_button("🔗 Open on YouTube", to_youtube_watch_url(v_url), use_container_width=True)
                        
                        st.divider()
                        st.caption("Assign to Voiceover Step:")
                        for step_tuple in segment_steps:
                            s_dict = step_tuple[0]
                            s_idx = s_dict.get("step_index", 1)
                            st.button(
                                f"Use for VO Part {s_idx}",
                                key=f"assign_vid_{embed_key_prefix}_{idx_in_list}_{s_idx}",
                                on_click=_assign_candidate_to_step,
                                args=(slide_idx, segment_num, s_idx, v_url),
                                use_container_width=True
                            )


def _render_inspector_step_review_controls(step_key_prefix):
    action_key = f"{step_key_prefix}_action"
    feedback_key = f"{step_key_prefix}_feedback"

    if action_key not in st.session_state:
        st.session_state[action_key] = ACTION_NONE
    if feedback_key not in st.session_state:
        st.session_state[feedback_key] = ""

    st.markdown("**Review action**")
    current_action = st.session_state.get(action_key, ACTION_NONE)

    btn_cols = st.columns(4, gap="small")
    with btn_cols[0]:
        approve_label = "🟢 Approve" if current_action == ACTION_APPROVE else "Approve"
        st.button(
            approve_label,
            key=f"{step_key_prefix}_btn_approve",
            type="secondary",
            on_click=_set_review_action,
            args=(action_key, ACTION_APPROVE),
            use_container_width=True,
        )
    with btn_cols[1]:
        reject_dh_label = "🔴 Reject (Drive + HVAC School Videos)" if current_action == ACTION_REJECT_DRIVE_HVAC else "Reject (Drive + HVAC School Videos)"
        st.button(
            reject_dh_label,
            key=f"{step_key_prefix}_btn_reject_drive_hvac",
            type="secondary",
            on_click=_set_review_action,
            args=(action_key, ACTION_REJECT_DRIVE_HVAC),
            use_container_width=True,
        )
    with btn_cols[2]:
        reject_all_label = "🔴 Reject (Search all)" if current_action == ACTION_REJECT_ALL else "Reject (Search all)"
        st.button(
            reject_all_label,
            key=f"{step_key_prefix}_btn_reject_all",
            type="secondary",
            on_click=_set_review_action,
            args=(action_key, ACTION_REJECT_ALL),
            use_container_width=True,
        )
    with btn_cols[3]:
        reject_ai_label = "🔵 Reject (Generate with AI)" if current_action == ACTION_REJECT_AI else "Reject (Generate with AI)"
        st.button(
            reject_ai_label,
            key=f"{step_key_prefix}_btn_reject_ai",
            type="secondary",
            on_click=_set_review_action,
            args=(action_key, ACTION_REJECT_AI),
            use_container_width=True,
        )

    current_action = st.session_state.get(action_key, ACTION_NONE)
    if current_action == ACTION_APPROVE:
        st.markdown(":green[Selected: Approve]")
    elif current_action in (ACTION_REJECT_DRIVE_HVAC, ACTION_REJECT_ALL):
        st.markdown(f":red[Selected: {ACTION_LABELS.get(current_action, 'Reject')}]")
    elif current_action == ACTION_REJECT_AI:
        st.markdown(":blue[Selected: Reject, generate with AI]")
    else:
        st.caption("Unreviewed")

    st.text_area(
        "Human feedback (optional)",
        key=feedback_key,
        placeholder="e.g. Wrong image; need diagram of X",
        height=80,
    )


@_st_fragment
def _render_single_segment_block(
    slide_idx, seg_num, visible_steps, image_pool_text, video_pool_text, drive, slide_row_idx
):
    """
    Isolated fragment for a single segment's selection pool and VO visuals.
    Using arguments (not closures) avoids variable-leak bugs in loops.
    """
    st.markdown(f"### Segment {seg_num}")
    
    seg_embed_key = f"{slide_idx}-{seg_num}-segpool"
    
    # 1. Candidate Image Pool (Selection Pool)
    _fragment_candidate_images_expander(
        image_pool_text=image_pool_text,
        segment_num=seg_num,
        slide_idx=slide_idx,
        segment_steps=visible_steps,
        drive=drive,
        embed_key_prefix=seg_embed_key,
    )
    
    # 1.5 Candidate Video Pool (Selection Pool)
    _fragment_candidate_videos_expander(
        video_pool_text=video_pool_text,
        segment_num=seg_num,
        slide_idx=slide_idx,
        segment_steps=visible_steps,
        embed_key_prefix=seg_embed_key,
    )
    
    # 2. VO Step Visuals & Controls
    for step_tuple in visible_steps:
        # tuple: (step_dict, action_scope_key, action_key, feedback_key, visual_id)
        s, scope_key, a_key, f_key, v_id = step_tuple
        
        st.markdown(f"**VO Part {s['step_index']}**")
        if s.get("voiceover"):
            st.write(s["voiceover"])
        else:
            st.caption("Voiceover text not found.")
            
        ov_key = f"asset_override_{slide_idx}_{seg_num}_{s['step_index']}"
        is_ov = ov_key in st.session_state
        cur_asset = st.session_state[ov_key] if is_ov else s.get("asset", "")
        
        cur_type = detect_asset_type(cur_asset)
        cur_display = (
            normalize_drive_image_url(cur_asset) if is_drive_url(cur_asset) else cur_asset
        ) if cur_type == "image" else ""
        
        # Determine fixed embed key for consistency
        this_embed_key = f"{slide_idx}-{seg_num}-{s['step_index']}"
        
        _render_inspector_step_visual(
            cur_asset, cur_type, cur_display, this_embed_key, drive
        )
        
        if is_ov:
            st.button(
                "↩️ Undo Re-assignment",
                key=f"revert_{ov_key}",
                on_click=_revert_asset_assignment,
                args=(ov_key,),
                type="secondary",
                use_container_width=True
            )
        
        _render_inspector_step_review_controls(scope_key)
        
        sel_action = st.session_state.get(a_key, ACTION_NONE)
        if sel_action == ACTION_APPROVE:
            st.success("Approved")


def render_inspector(slides, column_map, drive, sheet=None, worksheet_name=None, current_round=0, action_filter="all", include_unreviewed_with_revised=False):
    if not slides:
        st.info("No slide data to display.")
        return

    can_save = sheet is not None and worksheet_name
    _inject_review_button_layout_css()
    round_feedback_col = get_round_column_name(HUMAN_FEEDBACK_COLUMN, current_round)
    round_status_col = get_round_column_name(HUMAN_FEEDBACK_STATUS_COLUMN, current_round)
    round_tracking_col = get_round_column_name(HUMAN_FEEDBACK_TRACKING_COLUMN, current_round)
    round_actions_col = get_round_column_name(HUMAN_REVIEW_ACTIONS_COLUMN, current_round)
    review_col_indices = {
        "feedback": None,
        "status": None,
        "tracking": None,
        "actions": None,
    }

    def _ensure_review_columns(worksheet):
        if all(v is not None for v in review_col_indices.values()):
            return True
        review_col_indices["feedback"] = get_or_create_column(worksheet, round_feedback_col)
        review_col_indices["status"] = get_or_create_column(worksheet, round_status_col)
        review_col_indices["tracking"] = get_or_create_column(worksheet, round_tracking_col)
        review_col_indices["actions"] = get_or_create_column(worksheet, round_actions_col)
        return all(v is not None for v in review_col_indices.values())

    # First-load behavior: show everything in round 0, then default to revised+unreviewed for later rounds.
    if current_round == 0 and action_filter == "all":
        show_revised = True
        show_unreviewed = True
        show_approved = True
    else:
        show_revised = action_filter in ("all", "revised")
        show_unreviewed = action_filter in ("all", "unreviewed") or (action_filter == "revised" and include_unreviewed_with_revised)
        show_approved = action_filter in ("all", "approved")

    any_visible_items = False
    edited_visual_ids_by_row = st.session_state.get("gdv2_edited_visual_ids_by_row", {}) if action_filter == EDITED_FILTER_OPTION else {}

    for slide_idx, slide in enumerate(slides, start=1):
        segments = slide.get("segments", [])
        if not segments:
            continue
        
        # Store 1-based slide_idx into the slide dict so that _build_row_review_payload
        # can construct asset_override keys that match those used during assignment.
        slide["slide_idx_1based"] = slide_idx

        row_index = slide.get("row_index", slide_idx - 1)
        row_actions_key = f"gdv2_row_actions_{row_index}_r{current_round}"
        row_actions = st.session_state.setdefault(row_actions_key, {})
        existing_saved_actions = safe_str(slide.get(round_actions_col, "")).strip()
        slide_has_saved_progress = bool(existing_saved_actions and existing_saved_actions.lower() != "nan")
        persisted_current_round_actions_by_visual = _extract_actions_map(
            slide.get(round_actions_col, "")
        )

        # Build latest non-none action per visual across all completed previous rounds.
        historical_action_by_visual = {}
        if current_round > 0:
            for round_idx in range(current_round):
                historical_actions_col = get_round_column_name(HUMAN_REVIEW_ACTIONS_COLUMN, round_idx)
                actions_map = _extract_actions_map(slide.get(historical_actions_col, ""))
                for historical_visual_id, historical_item in actions_map.items():
                    historical_action = safe_str(historical_item.get("action", ACTION_NONE)).strip() or ACTION_NONE
                    if historical_action != ACTION_NONE:
                        historical_action_by_visual[historical_visual_id] = historical_action

        visible_segments = []
        for segment in segments:
            visible_steps = []
            for step in segment.get("steps", []):
                visual_id = f"S{segment['segment_index']}V{step['step_index']}"
                is_edited = False
                if action_filter == EDITED_FILTER_OPTION:
                    is_edited = visual_id in edited_visual_ids_by_row.get(row_index, set())
                action_scope_key = f"{row_index}_{segment['segment_index']}_{step['step_index']}_r{current_round}"
                action_key = f"{action_scope_key}_action"
                feedback_key = f"{action_scope_key}_feedback"
                persisted_current_item = persisted_current_round_actions_by_visual.get(visual_id, {})
                if action_scope_key not in row_actions and isinstance(persisted_current_item, dict):
                    row_actions[action_scope_key] = {
                        "action": safe_str(
                            persisted_current_item.get("action", ACTION_NONE)
                        ).strip()
                        or ACTION_NONE,
                        "feedback": safe_str(
                            persisted_current_item.get("feedback", "")
                        ),
                        "visual_id": visual_id,
                        "voiceover": step.get("voiceover", ""),
                        "segment_index": segment["segment_index"],
                        "step_index": step["step_index"],
                    }

                persisted_action = ACTION_NONE
                if action_scope_key in row_actions:
                    persisted_action = safe_str(
                        row_actions[action_scope_key].get("action", ACTION_NONE)
                    ).strip() or ACTION_NONE
                historical_action = historical_action_by_visual.get(visual_id, ACTION_NONE)

                if action_key not in st.session_state:
                    # First-time hydration for this run.
                    if persisted_action != ACTION_NONE:
                        st.session_state[action_key] = persisted_action
                    elif historical_action == ACTION_APPROVE:
                        st.session_state[action_key] = ACTION_APPROVE
                    else:
                        st.session_state[action_key] = ACTION_NONE
                else:
                    # If stale session value is unreviewed but previous round was approved,
                    # keep approve preselected unless user has made a non-empty choice now.
                    current_state_action = safe_str(
                        st.session_state.get(action_key, ACTION_NONE)
                    ).strip() or ACTION_NONE
                    if (
                        current_state_action == ACTION_NONE
                        and persisted_action == ACTION_NONE
                        and historical_action == ACTION_APPROVE
                    ):
                        st.session_state[action_key] = ACTION_APPROVE
                if action_scope_key in row_actions and feedback_key not in st.session_state:
                    st.session_state[feedback_key] = row_actions[action_scope_key].get("feedback", "")

                current_action = st.session_state.get(action_key, ACTION_NONE)
                current_feedback = st.session_state.get(feedback_key, "")
                # Persist latest per-visual state before applying visibility filtering.
                # Otherwise actions can be lost when a visual becomes hidden by the active filter.
                row_actions[action_scope_key] = {
                    "action": current_action,
                    "feedback": current_feedback,
                    "visual_id": visual_id,
                    "voiceover": step.get("voiceover", ""),
                    "segment_index": segment["segment_index"],
                    "step_index": step["step_index"],
                }
                effective_action = current_action
                # For "unreviewed" filter, show true current-round unreviewed visuals.
                # Do not backfill from historical action, otherwise round 1+ can show empty
                # even when reviewer has not selected any action in this round yet.
                if effective_action == ACTION_NONE and action_filter != "unreviewed":
                    effective_action = historical_action_by_visual.get(visual_id, ACTION_NONE)

                is_unreviewed = effective_action == ACTION_NONE
                is_approved = effective_action == ACTION_APPROVE
                is_revised = effective_action in (ACTION_REJECT_DRIVE_HVAC, ACTION_REJECT_ALL, ACTION_REJECT_AI)
                if action_filter == EDITED_FILTER_OPTION:
                    visible = is_edited
                else:
                    visible = (
                        (show_approved and is_approved)
                        or (show_revised and is_revised)
                        or (show_unreviewed and is_unreviewed)
                    )
                if not visible:
                    continue

                visible_steps.append((step, action_scope_key, action_key, feedback_key, visual_id))
            if visible_steps:
                visible_segments.append((segment, visible_steps))

        if not visible_segments:
            continue

        any_visible_items = True
        title_parts = [f"Slide {slide_idx}"]
        if slide.get("slide_title"):
            title_parts.append(slide["slide_title"])
        if slide.get("topic") or slide.get("subtopic"):
            title_parts.append(f"{slide.get('topic', '')} / {slide.get('subtopic', '')}".strip(" /"))
        slide_title = " - ".join([part for part in title_parts if part])

        with st.expander(slide_title, expanded=False):
            if slide_has_saved_progress:
                st.caption("This slide already has saved feedback for the current round.")
            if slide.get("slide_chunk"):
                st.markdown("**Slide Content**")
                st.write(slide["slide_chunk"])

            for segment, visible_steps in visible_segments:
                seg_num = segment.get("segment_index", 1)
                img_pool_text = safe_str(slide.get(IMAGE_POOL_COLUMN, "")).strip()
                vid_pool_text = safe_str(slide.get(VIDEO_POOL_COLUMN, "")).strip()

                _render_single_segment_block(
                    slide_idx=slide_idx,
                    seg_num=seg_num,
                    visible_steps=visible_steps,
                    image_pool_text=img_pool_text,
                    video_pool_text=vid_pool_text,
                    drive=drive,
                    slide_row_idx=row_index
                )

            if can_save:
                if st.button(
                    "Save feedback for this slide",
                    key=f"gdv2_save_slide_{row_index}_r{current_round}",
                    use_container_width=True,
                ):
                    try:
                        worksheet = sheet.worksheet(worksheet_name)
                        if not _ensure_review_columns(worksheet):
                            st.error(
                                "Could not find or create one or more round review columns. "
                                "Please check if the sheet is protected or has restricted edit permissions."
                            )
                        else:
                            # 1. Build payload AND updated graphics JSON (includes manual candidates)
                            fb_val, act_payload, _, row_act_map, up_graphics_json, up_tracking_text = _build_row_review_payload(
                                slide=slide,
                                row_actions=row_actions,
                                current_round=current_round,
                            )
                            
                            # 2. Identify the main graphics column index for persistence
                            graphics_col_name = FINAL_GRAPHICS_COLUMN
                            graphics_col_idx = get_or_create_column(worksheet, graphics_col_name)
                            
                            # 3. Write all to sheet in one batch
                            current_action_round = current_round
                            _write_single_row_review(
                                worksheet=worksheet,
                                row_index=row_index,
                                feedback_value=fb_val,
                                actions_payload=act_payload,
                                round_feedback_col_idx=review_col_indices["feedback"],
                                round_status_col_idx=review_col_indices["status"],
                                round_tracking_col_idx=review_col_indices["tracking"],
                                round_actions_col_idx=review_col_indices["actions"],
                                updated_graphics_json=up_graphics_json,
                                graphics_col_idx=graphics_col_idx,
                                updated_tracking_text=up_tracking_text
                            )
                            # Keep in-memory data synced
                            slide[round_actions_col] = act_payload
                            slide[round_feedback_col] = fb_val
                            slide[FINAL_GRAPHICS_COLUMN] = up_graphics_json
                            if fb_val or row_act_map:
                                st.success(f"Saved feedback for Slide {slide_idx}.")
                            else:
                                st.success(f"Saved current state for Slide {slide_idx}.")
                    except Exception as e:
                        st.error(f"Failed to save Slide {slide_idx}: {e}")

    # Pre-warm cache for all candidate Drive images after visible slides are rendered.
    # This runs after the slide loop so it doesn't delay the main slide display.
    # Only Drive images need server-side fetching; web images load directly in the browser.
    _preload_candidate_images(slides, drive)

    if not any_visible_items:
        st.info("No visuals match the selected review filter.")

    if can_save:
        st.divider()
        save_col1, save_col2 = st.columns(2)
        with save_col1:
            save_and_revise_now = st.button(
                "Save all feedback and revise visuals",
                type="primary",
                key="gdv2_save_revise_now_btn",
            )
        with save_col2:
            save_and_revise_bg = st.button(
                "Save all feedback and revise visuals in Background",
                type="primary",
                key="gdv2_save_revise_bg_btn",
            )

        st.divider()
        regen_col1, regen_col2 = st.columns(2)
        with regen_col1:
            regen_web_now = st.button(
                "Regenerate web images with AI",
                type="primary",
                key="gdv2_regen_web_now_btn",
            )
        with regen_col2:
            regen_web_bg = st.button(
                "Regenerate web images with AI in Background",
                type="primary",
                key="gdv2_regen_web_bg_btn",
            )

        if regen_web_bg:
            sheet_url = st.session_state.get("gdv2_sheet_link", "") or ""
            if not sheet_url.strip():
                st.error("Missing sheet URL. Please re-load the sheet using the 'Load Sheet' button.")
            else:
                try:
                    worksheet = sheet.worksheet(worksheet_name)
                    headers = worksheet.row_values(1)

                    main_col = FINAL_GRAPHICS_COLUMN
                    if main_col not in headers:
                        st.error(f"Column '{main_col}' not found in worksheet. Cannot regenerate in background.")
                        return

                    archive_cols = []
                    for h in headers:
                        if isinstance(h, str) and h.startswith(f"{main_col}_"):
                            suffix = h[len(f"{main_col}_"):]
                            if suffix.isdigit():
                                archive_cols.append((int(suffix), h))
                    archive_cols.sort(key=lambda x: x[0])
                    latest_archive_col = archive_cols[-1][1] if archive_cols else None

                    _, df_before = get_sheet_data_and_df(sheet, worksheet_name)
                    output_values = []
                    if main_col in df_before.columns:
                        output_values = [
                            safe_str(v).strip()
                            for v in df_before[main_col].tolist()
                            if safe_str(v).strip() and safe_str(v).strip().lower() != "nan"
                        ]
                    total_rows = len(df_before)
                    filled_count = len(output_values)
                    has_partial_output = total_rows > 0 and 0 < filled_count < total_rows
                    has_any_output = filled_count > 0

                    # Resume mode: a previous regen already created an archive and output is partially filled.
                    if latest_archive_col and (has_partial_output or not has_any_output):
                        archive_col = latest_archive_col
                        skip_filled_rows_for_regen = True
                        st.info(
                            f"Resuming interrupted regeneration from '{archive_col}'. "
                            "Only remaining rows will be processed."
                        )
                    else:
                        max_n = archive_cols[-1][0] if archive_cols else 0
                        next_n = max_n + 1
                        archive_col = f"{main_col}_{next_n}"

                        # Fresh run: archive current main column and create a new output main column.
                        main_col_idx_1based = headers.index(main_col) + 1
                        worksheet.update_cell(1, main_col_idx_1based, archive_col)

                        _, df_after_archive = get_sheet_data_and_df(sheet, worksheet_name)
                        hide_columns_by_name(
                            worksheet,
                            [archive_col, FINAL_GRAPHICS_COLUMN],
                            df_after_archive,
                        )
                        get_or_create_column(worksheet, main_col)
                        skip_filled_rows_for_regen = False

                    user_email = st.session_state.get("user_email", "") or ""
                    cmd = [
                        sys.executable,
                        "launch_agents_via_sdk.py",
                        "--sheet_link",
                        sheet_url,
                        "--drive_folder_id",
                        st.session_state.get("root_folder_id", ""),
                        "--agent_name",
                        "web_image_regeneration_bg",
                        "--user_email",
                        user_email,
                        "--source_tab",
                        worksheet_name,
                        "--regen_input_column",
                        archive_col,
                        "--regen_output_column",
                        main_col,
                        "--regen_output_folder_name",
                        "Web Image Regeneration",
                        "--regen_write_final_graphics",
                        "false",
                        "--regen_skip_filled_rows",
                        "true" if skip_filled_rows_for_regen else "false",
                    ]

                    _rt = st.session_state.get("google_oauth_refresh_token")
                    if _rt:
                        cmd.extend(
                            [
                                "--google_oauth_refresh_token_b64",
                                base64.b64encode(_rt.encode("utf-8")).decode("ascii"),
                            ]
                        )
                    else:
                        st.info(
                            "No refresh token in session (re-login with Google to enable user Drive uploads "
                            "in the cloud job). Otherwise the job uses the service account for AI image uploads."
                        )

                    with st.spinner("Submitting background regeneration job..."):
                        process = subprocess.Popen(
                            cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True,
                        )
                        logs = ""
                        job_link = None
                        job_name = None
                        link_re = re.compile(r"^\[JOB_LINK\]\s+(?P<link>\S+)\s*$")
                        name_re = re.compile(r"^\[JOB_NAME\]\s+(?P<name>.+?)\s*$")
                        start = time.time()
                        while True:
                            if process.stdout is None:
                                break
                            line = process.stdout.readline()
                            if not line:
                                if process.poll() is not None:
                                    break
                                if time.time() - start > 10:
                                    break
                                time.sleep(0.1)
                                continue
                            logs += line
                            m = link_re.match(line.strip())
                            if m:
                                job_link = m.group("link")
                                break
                            m2 = name_re.match(line.strip())
                            if m2:
                                job_name = m2.group("name")
                        try:
                            if process.stdout is not None:
                                process.stdout.close()
                        except Exception:
                            pass
                        try:
                            process.wait(timeout=2)
                        except Exception:
                            pass

                    if job_link:
                        st.session_state["gdv2_regen_bg_link"] = job_link
                        st.session_state.pop("gdv2_regen_bg_name", None)
                        st.success("Background web-image regeneration job submitted.")
                    elif job_name:
                        st.session_state["gdv2_regen_bg_name"] = job_name
                        st.session_state.pop("gdv2_regen_bg_link", None)
                        st.success("Background web-image regeneration job submitted.")
                        with st.expander("Launcher output (no job link found)", expanded=False):
                            st.code(logs[-5000:] if len(logs) > 5000 else logs)
                    else:
                        st.error("Background regeneration submission did not return a job link.")
                        with st.expander("Launcher output", expanded=True):
                            st.code(logs[-5000:] if len(logs) > 5000 else logs)
                except Exception as e:
                    st.error(f"Background regeneration failed before launch: {e}")
                    raise

        if save_and_revise_now or save_and_revise_bg:
            worksheet = None
            had_save_error = False
            try:
                worksheet = sheet.worksheet(worksheet_name)
            except Exception as e:
                st.error(f"Failed to access worksheet: {e}")
                had_save_error = True

            if worksheet:
                if not _ensure_review_columns(worksheet):
                    st.error(
                        "Could not find or create one or more round review columns. "
                        "Please check if the sheet is protected or has restricted edit permissions."
                    )
                    had_save_error = True

            if worksheet and not had_save_error:
                saved_count = 0
                reject_count = 0
                skipped_existing_count = 0
                pending_row_writes = []
                # Get graphics column index once before the loop
                graphics_col_idx_global = get_or_create_column(worksheet, FINAL_GRAPHICS_COLUMN)

                for slide in slides:
                    segments = slide.get("segments", [])
                    if not segments:
                        continue
                    row_index = slide.get("row_index", 0)
                    row_actions = st.session_state.get(f"gdv2_row_actions_{row_index}_r{current_round}", {})
                    fb_val, actions_payload, row_reject_count, row_act_map, up_graphics_json, up_tracking_text = _build_row_review_payload(
                        slide=slide,
                        row_actions=row_actions,
                        current_round=current_round,
                    )
                    reject_count += row_reject_count

                    # Resume-safe behavior:
                    # use only round actions column as the checkpoint signal.
                    existing_actions = safe_str(slide.get(round_actions_col, "")).strip()
                    if existing_actions and existing_actions.lower() != "nan":
                        skipped_existing_count += 1
                        continue

                    pending_row_writes.append(
                        (row_index, fb_val, actions_payload, up_graphics_json, up_tracking_text)
                    )
                    slide[round_actions_col] = actions_payload
                    slide[round_feedback_col] = fb_val
                    slide[FINAL_GRAPHICS_COLUMN] = up_graphics_json
                    slide[round_tracking_col] = up_tracking_text
                    if fb_val or row_act_map:
                        saved_count += 1

                # Batch write row updates to avoid per-cell quota spikes (429).
                if pending_row_writes:
                    try:
                        feedback_col = _col_to_a1(review_col_indices["feedback"])
                        status_col = _col_to_a1(review_col_indices["status"])
                        tracking_col = _col_to_a1(review_col_indices["tracking"])
                        actions_col_letter = _col_to_a1(review_col_indices["actions"])
                        graphics_col_letter = _col_to_a1(graphics_col_idx_global) if graphics_col_idx_global else None

                        # 250 rows -> 1000 ranges per request; keep request size moderate.
                        for chunk in _chunked(pending_row_writes, 250):
                            batch_ranges = []
                            for row_index, fb_val, actions_payload, up_graphics_json, up_tracking_text in chunk:
                                sheet_row = int(row_index) + 2
                                batch_ranges.append({"range": f"{feedback_col}{sheet_row}", "values": [[fb_val]]})
                                batch_ranges.append({"range": f"{status_col}{sheet_row}", "values": [[""]]})
                                batch_ranges.append({"range": f"{tracking_col}{sheet_row}", "values": [[up_tracking_text if up_tracking_text else ""]]})
                                batch_ranges.append({"range": f"{actions_col_letter}{sheet_row}", "values": [[actions_payload]]})
                                # Write updated graphics JSON (includes image/video assignments)
                                if graphics_col_letter and up_graphics_json:
                                    batch_ranges.append({"range": f"{graphics_col_letter}{sheet_row}", "values": [[up_graphics_json]]})
                            worksheet.batch_update(batch_ranges, value_input_option="RAW")
                    except Exception as e:
                        st.error(f"Failed to save review batch update: {e}")
                        had_save_error = True

                if saved_count > 0:
                    st.success(f"Review saved for {saved_count} slide(s).")
                else:
                    st.warning("No review actions entered for any slide.")

            if not had_save_error and worksheet:
                if save_and_revise_now:
                    st.info("Starting revision process based on the given feedbacks...")
                    try:
                        if reject_count > 0:
                            run_human_feedback_review_revise_for_all_rows(
                                sheet=sheet,
                                llm="gemini_3_flash_thinking",
                                max_workers=50,
                                use_only_drive_and_hvac=False,
                                human_feedback_column=round_feedback_col,
                                human_feedback_status_column=round_status_col,
                                human_feedback_revision_tracking_column=round_tracking_col,
                                human_review_actions_column=round_actions_col,
                            )
                            st.success("Human-feedback revision completed.")
                            st.session_state["gdv2_round"] = current_round + 1
                            # Do not mutate widget state key after widget instantiation in same run.
                            # Apply this on next run before the selectbox is created.
                            st.session_state["gdv2_pending_visual_filter"] = "revised"
                            st.session_state["gdv2_revision_notice"] = True
                        else:
                            st.info("No rejected visuals found; skipped review-revise run.")

                        _, refreshed_df = get_sheet_data_and_df(sheet, worksheet_name)
                        st.session_state["gdv2_df"] = refreshed_df
                        st.rerun()
                    except Exception as e:
                        st.error(f"Human-feedback revision failed: {e}")
                else:
                    if reject_count <= 0:
                        st.info("No rejected visuals found; skipped review-revise run.")
                    else:
                        user_email = st.session_state.get("user_email", "") or ""
                        cmd = [
                            sys.executable,
                            "launch_agents_via_sdk.py",
                            "--sheet_link",
                            st.session_state.get("gdv2_sheet_link", ""),
                            "--drive_folder_id",
                            st.session_state.get("root_folder_id", ""),
                            "--agent_name",
                            "human_feedback_review_revise",
                            "--user_email",
                            user_email,
                            "--human_feedback_column",
                            round_feedback_col,
                            "--human_feedback_status_column",
                            round_status_col,
                            "--human_feedback_revision_tracking_column",
                            round_tracking_col,
                            "--human_review_actions_column",
                            round_actions_col,
                            "--llm",
                            "gemini_3_flash_thinking",
                            "--max_workers",
                            "50",
                        ]
                        _rt = st.session_state.get("google_oauth_refresh_token")
                        if _rt:
                            cmd.extend(
                                [
                                    "--google_oauth_refresh_token_b64",
                                    base64.b64encode(_rt.encode("utf-8")).decode("ascii"),
                                ]
                            )
                        else:
                            st.info(
                                "No refresh token in session (re-login with Google to enable user Drive uploads "
                                "in the cloud job). Otherwise the job uses the service account for AI image uploads."
                            )
                        with st.spinner("Submitting background job..."):
                            process = subprocess.Popen(
                                cmd,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                                text=True,
                            )
                            logs = ""
                            job_link = None
                            job_name = None
                            link_re = re.compile(r"^\[JOB_LINK\]\s+(?P<link>\S+)\s*$")
                            name_re = re.compile(r"^\[JOB_NAME\]\s+(?P<name>.+?)\s*$")
                            start = time.time()
                            while True:
                                if process.stdout is None:
                                    break
                                line = process.stdout.readline()
                                if not line:
                                    if process.poll() is not None:
                                        break
                                    if time.time() - start > 10:
                                        break
                                    time.sleep(0.1)
                                    continue
                                logs += line
                                m = link_re.match(line.strip())
                                if m:
                                    job_link = m.group("link")
                                    break
                                m2 = name_re.match(line.strip())
                                if m2:
                                    job_name = m2.group("name")
                            try:
                                if process.stdout is not None:
                                    process.stdout.close()
                            except Exception:
                                pass
                            try:
                                process.wait(timeout=2)
                            except Exception:
                                pass

                        if job_link:
                            st.session_state["gdv2_hf_bg_link"] = job_link
                            st.session_state.pop("gdv2_hf_bg_name", None)
                            st.success("Background human-feedback revise job submitted.")
                        elif job_name:
                            st.session_state["gdv2_hf_bg_name"] = job_name
                            st.session_state.pop("gdv2_hf_bg_link", None)
                            st.success("Background human-feedback revise job submitted.")
                            with st.expander("Launcher output (no job link found)", expanded=False):
                                st.code(logs[-5000:] if len(logs) > 5000 else logs)
                        else:
                            st.error("Background job submission did not return a job link.")
                            with st.expander("Launcher output", expanded=True):
                                st.code(logs[-5000:] if len(logs) > 5000 else logs)

        if regen_web_now:
            sheet_url = st.session_state.get("gdv2_sheet_link", "") or ""
            if not sheet_url.strip():
                st.error("Missing sheet URL. Please re-load the sheet using the 'Load Sheet' button.")
            else:
                try:
                    worksheet = sheet.worksheet(worksheet_name)
                    headers = worksheet.row_values(1)

                    main_col = FINAL_GRAPHICS_COLUMN
                    if main_col not in headers:
                        st.error(f"Column '{main_col}' not found in worksheet. Cannot regenerate.")
                        return

                    archive_cols = []
                    for h in headers:
                        if isinstance(h, str) and h.startswith(f"{main_col}_"):
                            suffix = h[len(f"{main_col}_"):]
                            if suffix.isdigit():
                                archive_cols.append((int(suffix), h))
                    archive_cols.sort(key=lambda x: x[0])
                    latest_archive_col = archive_cols[-1][1] if archive_cols else None

                    _, df_before = get_sheet_data_and_df(sheet, worksheet_name)
                    output_values = []
                    if main_col in df_before.columns:
                        output_values = [
                            safe_str(v).strip()
                            for v in df_before[main_col].tolist()
                            if safe_str(v).strip() and safe_str(v).strip().lower() != "nan"
                        ]
                    total_rows = len(df_before)
                    filled_count = len(output_values)
                    has_partial_output = total_rows > 0 and 0 < filled_count < total_rows
                    has_any_output = filled_count > 0

                    if latest_archive_col and (has_partial_output or not has_any_output):
                        archive_col = latest_archive_col
                        skip_filled_rows_for_regen = True
                        st.info(
                            f"Resuming interrupted regeneration from '{archive_col}'. "
                            "Only remaining rows will be processed."
                        )
                    else:
                        max_n = archive_cols[-1][0] if archive_cols else 0
                        next_n = max_n + 1
                        archive_col = f"{main_col}_{next_n}"

                        # Fresh run: archive current main column and create a new output main column.
                        main_col_idx_1based = headers.index(main_col) + 1
                        worksheet.update_cell(1, main_col_idx_1based, archive_col)

                        _, df_after_archive = get_sheet_data_and_df(sheet, worksheet_name)
                        # Hide archived history column(s) and also hide legacy final_graphics.
                        hide_columns_by_name(
                            worksheet,
                            [archive_col, FINAL_GRAPHICS_COLUMN],
                            df_after_archive,
                        )
                        get_or_create_column(worksheet, main_col)
                        skip_filled_rows_for_regen = False

                    st.info("Regenerating web images now. This can take a while...")
                    regen_status = st.empty()
                    regen_progress = st.progress(0.0, text="Regenerating web images: 0% | Just started...")
                    regen_start_time = time.time()

                    def _regen_progress_cb(completed: int, total: int):
                        # Called by run_automation after each subsegment completes.
                        pct = (completed / total) if total else 1.0
                        pct = max(0.0, min(1.0, pct))
                        elapsed_seconds = time.time() - regen_start_time

                        if total == 0:
                            text = (
                                "Regenerating web images: Nothing to regenerate "
                                "(0 eligible subsegments)."
                            )
                            regen_status.markdown(text)
                            regen_progress.progress(1.0, text=text)
                            return

                        if completed > 0 and total and total >= completed:
                            seconds_per_task = elapsed_seconds / completed
                            remaining_tasks = total - completed
                            estimated_remaining_seconds = max(0.0, seconds_per_task * remaining_tasks)

                            elapsed_time_str = str(datetime.timedelta(seconds=int(elapsed_seconds)))
                            remaining_time_str = str(datetime.timedelta(seconds=int(estimated_remaining_seconds)))
                            text = (
                                f"Regenerating web images: {int(pct * 100)}% "
                                f"({completed}/{total}) | Elapsed: {elapsed_time_str} | Remaining: {remaining_time_str}"
                            )
                        else:
                            text = (
                                f"Regenerating web images: {int(pct * 100)}% "
                                f"({completed}/{total}) | Elapsed: {str(datetime.timedelta(seconds=int(elapsed_seconds)))}"
                            )

                        # Update both the status line and the progress bar.
                        # (Depending on Streamlit timing, the status line tends to render more reliably.)
                        regen_status.markdown(text)
                        regen_progress.progress(pct, text=text)

                    with st.spinner("Running regeneration pipeline..."):
                        run_automation(
                            sheet_url=sheet_url,
                            source_tab=worksheet_name,
                            output_folder_name="Web Image Regeneration",
                            gc=st.session_state.get("gc"),
                            drive=st.session_state.get("drive"),
                            progress_callback=_regen_progress_cb,
                            skip_filled_rows=skip_filled_rows_for_regen,
                            input_column_name=archive_col,
                            output_column_name=main_col,
                            write_final_graphics=False,
                        )

                    # Refresh inspector state so regenerated links are visible immediately.
                    _clear_inspector_image_cache()
                    st.session_state.pop(INSPECTOR_VIDEO_HTML_CACHE_KEY, None)
                    _, refreshed_df = get_sheet_data_and_df(sheet, worksheet_name)
                    st.session_state["gdv2_df"] = refreshed_df
                    st.session_state["gdv2_round"] = detect_current_round(refreshed_df)
                    st.session_state["gdv2_pending_visual_filter"] = EDITED_FILTER_OPTION
                    st.session_state["gdv2_regen_notice"] = True
                    st.success("✅ Regeneration complete. Results are written back to the sheet.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Regeneration failed: {e}")
                    raise

        if st.session_state.get("gdv2_revision_notice", False):
            st.info(
                "Visuals have been revised based on all the feedbacks, you can now review the revised visuals and leave any new feedback if you want."
            )
        if st.session_state.get("gdv2_regen_notice", False):
            st.info(
                "Web-image regeneration completed. Review the edited visuals and leave additional feedback if needed."
            )
        if st.session_state.get("gdv2_hf_bg_link"):
            st.markdown(
                f"**Background job:** [{st.session_state['gdv2_hf_bg_link']}]({st.session_state['gdv2_hf_bg_link']})"
            )
        elif st.session_state.get("gdv2_hf_bg_name"):
            st.markdown(f"**Background job:** `{st.session_state['gdv2_hf_bg_name']}`")
        if st.session_state.get("gdv2_regen_bg_link"):
            st.markdown(
                f"**Background regen job:** [{st.session_state['gdv2_regen_bg_link']}]({st.session_state['gdv2_regen_bg_link']})"
            )
        elif st.session_state.get("gdv2_regen_bg_name"):
            st.markdown(f"**Background regen job:** `{st.session_state['gdv2_regen_bg_name']}`")


def main():
    try:
        st.set_page_config(
            page_title="Graphics Definition V2 Slideshow",
            page_icon="",
            layout="wide",
        )
    except Exception:
        pass

    st.title("Graphics Definition V2 Slideshow")
    st.caption("Review generated visuals and submit human feedback.")

    if "gc" not in st.session_state:
        st.error("Google Sheets client not found. Please log in first.")
        return

    gc = st.session_state["gc"]
    drive = st.session_state.get("drive")

    sheet_link_default = st.session_state.get("sheet_link", "")
    sheet_link = st.text_input("Google Sheet link", value=sheet_link_default)
    load_sheet = st.button("Load Sheet")
    if load_sheet and not (sheet_link and sheet_link.strip()):
        st.info("Please enter sheet link.")
    if load_sheet and sheet_link and sheet_link.strip():
        try:
            sheet = gc.open_by_url(sheet_link)
            st.session_state["sheet_link"] = sheet_link
            st.session_state["gdv2_sheet"] = sheet
            st.session_state["gdv2_sheet_link"] = sheet_link
            st.session_state["gdv2_df"] = None
            st.session_state["gdv2_df_sheet"] = None
            st.session_state["gdv2_df_sheet_link"] = sheet_link
            st.session_state.pop("gdv2_revision_notice", None)
            st.session_state.pop("gdv2_regen_notice", None)
            _clear_inspector_image_cache()
            st.session_state.pop(INSPECTOR_VIDEO_HTML_CACHE_KEY, None)
        except Exception as e:
            st.error(f"Failed to open sheet: {e}")
            return

    sheet = st.session_state.get("gdv2_sheet")
    if not sheet:
        return

    worksheet_name = DEFAULT_SHEET_NAME
    refresh = False
    active_sheet_link = st.session_state.get("gdv2_sheet_link", "")
    worksheet_or_sheet_changed = (
        st.session_state.get("gdv2_df_sheet") != worksheet_name
        or st.session_state.get("gdv2_df_sheet_link") != active_sheet_link
    )
    if (
        refresh
        or worksheet_or_sheet_changed
    ):
        try:
            _clear_inspector_image_cache()
            st.session_state.pop(INSPECTOR_VIDEO_HTML_CACHE_KEY, None)
            _, df = get_sheet_data_and_df(sheet, worksheet_name)
            st.session_state["gdv2_df"] = df
            st.session_state["gdv2_df_sheet"] = worksheet_name
            st.session_state["gdv2_df_sheet_link"] = active_sheet_link
            st.session_state["gdv2_round"] = detect_current_round(df)
        except Exception as e:
            st.error(f"Failed to load data: {e}")
            return

    df = st.session_state.get("gdv2_df")
    if df is None or df.empty:
        st.warning("No data found in the selected worksheet.")
        return

    # Determine whether an archived "previous version" column exists so we can show the edited filter.
    archive_cols = []
    for c in df.columns:
        if isinstance(c, str) and c.startswith(f"{FINAL_GRAPHICS_COLUMN}_"):
            suffix = c[len(f"{FINAL_GRAPHICS_COLUMN}_"):]
            if suffix.isdigit():
                archive_cols.append((int(suffix), c))
    archive_cols.sort(key=lambda x: x[0])
    latest_archive_col = archive_cols[-1][1] if archive_cols else None
    show_edited_filter = latest_archive_col is not None


    # If the sheet contains a human-feedback status column for any round (human_feedback_status_n),
    # treat that as meaning the human-feedback revise pipeline ran at least once.
    revise_happened_ever = any(
        isinstance(c, str)
        and c.startswith(f"{HUMAN_FEEDBACK_STATUS_COLUMN}_")
        and c[len(f"{HUMAN_FEEDBACK_STATUS_COLUMN}_") :].isdigit()
        for c in df.columns
    )

    def _latest_populated_actions_round_incomplete() -> bool:
        """
        Check the latest human_review_actions_<n> column that has any data.
        Return True when that latest populated round is incomplete.
        """
        action_rounds = []
        for c in df.columns:
            if not (isinstance(c, str) and c.startswith(f"{HUMAN_REVIEW_ACTIONS_COLUMN}_")):
                continue
            suffix = c[len(f"{HUMAN_REVIEW_ACTIONS_COLUMN}_"):]
            if suffix.isdigit():
                action_rounds.append((int(suffix), c))
        action_rounds.sort(key=lambda x: x[0])
        latest_populated_col = None
        for _, col in action_rounds:
            has_any_data = any(
                safe_str(v).strip() and safe_str(v).strip().lower() != "nan"
                for v in df[col].tolist()
            )
            if has_any_data:
                latest_populated_col = col
        if latest_populated_col is None:
            return False
        for raw in df[latest_populated_col].tolist():
            text = safe_str(raw).strip()
            if not text or text.lower() == "nan":
                return True
            actions_map = _extract_actions_map(text)
            if not actions_map:
                return True
            if any(
                safe_str(item.get("action", ACTION_NONE)).strip() in ("", ACTION_NONE)
                for item in actions_map.values()
                if isinstance(item, dict)
            ):
                return True
        return False

    column_map = {
        "topic": find_column(df, "Topic"),
        "subtopic": find_column(df, "Subtopic"),
        "slide_title": find_column(df, "Slide Chunk Title"),
        "slide_chunk": find_column(df, "Slide Chunk"),
        "final_def": find_column(df, FINAL_GRAPHICS_COLUMN),
    }

    if not column_map["final_def"]:
        st.error(f"Column '{FINAL_GRAPHICS_COLUMN}' not found in worksheet.")
        return

    topic_values = (
        sorted({safe_str(v) for v in df[column_map["topic"]].dropna()})
        if column_map["topic"]
        else []
    )
    subtopic_values = (
        sorted({safe_str(v) for v in df[column_map["subtopic"]].dropna()})
        if column_map["subtopic"]
        else []
    )

    if "gdv2_round" not in st.session_state:
        st.session_state["gdv2_round"] = detect_current_round(df)
    latest_actions_round_incomplete = _latest_populated_actions_round_incomplete()
    pending_visual_filter = st.session_state.pop("gdv2_pending_visual_filter", None)
    if pending_visual_filter == EDITED_FILTER_OPTION and show_edited_filter:
        st.session_state["gdv2_visual_filter"] = EDITED_FILTER_OPTION
    elif pending_visual_filter in HUMAN_REVIEW_FILTER_OPTIONS:
        st.session_state["gdv2_visual_filter"] = pending_visual_filter
    if "gdv2_visual_filter" not in st.session_state:
        # Default filter selection across sessions:
        # - If regen happened at least once, prefer `edited`.
        # - Else if latest populated actions round is incomplete, default to `unreviewed`.
        # - Else if revise happened before, default to `revised`.
        # - Else default to `all` (first time / nothing done yet).
        if show_edited_filter:
            st.session_state["gdv2_visual_filter"] = EDITED_FILTER_OPTION
        elif latest_actions_round_incomplete:
            st.session_state["gdv2_visual_filter"] = "unreviewed"
        elif revise_happened_ever:
            st.session_state["gdv2_visual_filter"] = "revised"
        else:
            st.session_state["gdv2_visual_filter"] = "all"
    filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)
    with filter_col1:
        topic_filter = st.selectbox("Topic", options=["All"] + topic_values)
    with filter_col2:
        subtopic_options = ["All"] + subtopic_values
        if topic_filter != "All" and column_map["topic"] and column_map["subtopic"]:
            subtopic_options = ["All"] + sorted(
                {
                    safe_str(v)
                    for v in df[df[column_map["topic"]] == topic_filter][
                        column_map["subtopic"]
                    ].dropna()
                }
            )
        subtopic_filter = st.selectbox("Subtopic", options=subtopic_options)
    with filter_col3:
        search_text = st.text_input("Search", value="")
    with filter_col4:
        if "gdv2_visual_filter" not in st.session_state:
            st.session_state["gdv2_visual_filter"] = (
                "all" if int(st.session_state.get("gdv2_round", 0)) == 0 else "revised"
            )
        current_visual_filter = st.session_state.get("gdv2_visual_filter", "all")
        visual_options = HUMAN_REVIEW_FILTER_OPTIONS + ([EDITED_FILTER_OPTION] if show_edited_filter else [])
        if current_visual_filter not in visual_options:
            st.session_state["gdv2_visual_filter"] = "all"
        visual_filter = st.selectbox(
            "Visual Review Filter",
            options=visual_options,
            key="gdv2_visual_filter",
            format_func=lambda x: x.capitalize(),
        )

    filtered_df = df.copy()
    if topic_filter != "All" and column_map["topic"]:
        filtered_df = filtered_df[filtered_df[column_map["topic"]] == topic_filter]
    if subtopic_filter != "All" and column_map["subtopic"]:
        filtered_df = filtered_df[filtered_df[column_map["subtopic"]] == subtopic_filter]
    if search_text:
        search_lower = search_text.lower()

        def row_matches(row):
            combined = " ".join(
                safe_str(row.get(col, "")) for col in column_map.values() if col
            ).lower()
            return search_lower in combined

        filtered_df = filtered_df[filtered_df.apply(row_matches, axis=1)]

    slides = build_slides_from_df(filtered_df, column_map)
    steps = flatten_steps(slides)

    if "gdv2_slideshow_key" in st.session_state:
        current_key = compute_slideshow_key(steps)
        if st.session_state.get("gdv2_slideshow_key") != current_key:
            st.session_state["gdv2_prepared_steps"] = None
            st.session_state["gdv2_slideshow_key"] = current_key
    else:
        st.session_state["gdv2_slideshow_key"] = compute_slideshow_key(steps)

    # If the user is filtering by "edited", compute which visuals changed between
    # the latest archived column and the current FINAL_GRAPHICS_COLUMN.
    if visual_filter == EDITED_FILTER_OPTION and latest_archive_col:
        edited_visual_ids_by_row = {}

        for row_idx, row in filtered_df.iterrows():
            curr_def = safe_str(row.get(FINAL_GRAPHICS_COLUMN, "")).strip()
            prev_def = safe_str(row.get(latest_archive_col, "")).strip()
            curr_segments = parse_graphics_definition(curr_def)
            prev_segments = parse_graphics_definition(prev_def)

            curr_assets = {}
            for seg in curr_segments:
                for stp in seg.get("steps", []):
                    vid = f"S{seg['segment_index']}V{stp['step_index']}"
                    curr_assets[vid] = _primary_asset_url(stp.get("asset", ""))

            prev_assets = {}
            for seg in prev_segments:
                for stp in seg.get("steps", []):
                    vid = f"S{seg['segment_index']}V{stp['step_index']}"
                    prev_assets[vid] = _primary_asset_url(stp.get("asset", ""))

            changed = set()
            for vid, asset in curr_assets.items():
                if prev_assets.get(vid, "") != asset:
                    changed.add(vid)
            # If a visual id exists in prev but not curr, it no longer renders; ignore.

            edited_visual_ids_by_row[row_idx] = changed

        st.session_state["gdv2_edited_visual_ids_by_row"] = edited_visual_ids_by_row
    else:
        st.session_state["gdv2_edited_visual_ids_by_row"] = {}

    render_inspector(
        slides,
        column_map,
        drive,
        sheet=sheet,
        worksheet_name=worksheet_name,
        current_round=int(st.session_state.get("gdv2_round", 0)),
        action_filter=visual_filter,
        include_unreviewed_with_revised=int(st.session_state.get("gdv2_round", 0)) > 0,
    )


if __name__ == "__main__":
    main()
else:
    main()