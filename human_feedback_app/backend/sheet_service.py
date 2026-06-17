"""Sheet loading and UI payload transformation."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from services.sheets_service import get_sheet_data_and_df, save_to_sheet

from human_feedback_app.backend.constants import (
    ACTION_APPROVE,
    ACTION_NONE,
    AI_NO_FEEDBACK_MARKER,
    DEFAULT_REJECT_FEEDBACK,
    DEFAULT_WORKSHEET,
    FINAL_GRAPHICS_COLUMN,
    HUMAN_FEEDBACK_COLUMN,
    HUMAN_FEEDBACK_STATUS_COLUMN,
    HUMAN_FEEDBACK_TRACKING_COLUMN,
    HUMAN_REVIEW_ACTIONS_COLUMN,
    MODE_TO_ACTION,
)
from human_feedback_app.backend.sessions import UserSession


def _first_http_url(text: str) -> str:
    if not text:
        return ""
    match = re.search(r"https?://[^\s)>\"]+", str(text).strip())
    return match.group(0).rstrip(".,);\"'") if match else str(text).strip()


def _import_slideshow_helpers():
    from graphics_definition_v2_slideshow import (
        build_slides_from_df,
        detect_asset_type,
        detect_current_round,
        get_round_column_name,
        parse_graphics_definition,
        safe_str,
    )

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
    from agents.graphics_definition_v2.aggregation_agent.aggregation_agent import (
        parse_urls_from_image_pool,
        parse_urls_from_video_pool_filtered,
    )

    helpers = _import_slideshow_helpers()
    safe_str = helpers["safe_str"]

    exclude = _normalize_asset_url(exclude_url)
    seen: set[str] = set()
    out: List[Dict[str, Any]] = []

    def add_candidate(title: str, url: str, duration: str = "") -> None:
        url = (url or "").strip()
        if not url:
            return
        dedupe_key = _normalize_asset_url(url)
        if dedupe_key in seen:
            return
        if exclude and dedupe_key == exclude:
            return
        seen.add(dedupe_key)
        atype = "video" if asset_type == "video" else "image"
        out.append(
            {
                "title": (title or _label_from_asset(url, atype)).strip(),
                "url": url,
                "type": atype,
                "source": _infer_source(url, atype),
                "duration": (duration or "").strip(),
            }
        )

    if asset_type == "video":
        pool_text = safe_str(row.get("video_pool_filtered", "")).strip()
        if pool_text and pool_text != "nan":
            for item in parse_urls_from_video_pool_filtered(pool_text, segment_index):
                meta = item.get("metadata") or {}
                add_candidate(
                    str(meta.get("title") or "Video clip"),
                    str(item.get("url") or ""),
                    str(meta.get("duration") or ""),
                )
            for item in parse_urls_from_image_pool(pool_text, segment_index):
                add_candidate(str(item.get("title") or ""), str(item.get("url") or ""))
    else:
        pool_text = safe_str(row.get("image_pool", "")).strip()
        if pool_text and pool_text != "nan":
            for item in parse_urls_from_image_pool(pool_text, segment_index):
                add_candidate(str(item.get("title") or ""), str(item.get("url") or ""))

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
    if action not in (ACTION_NONE, "", "unreviewed"):
        return "pending"
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
    from agents.graphics_definition_v2.review_agent.human_feedback_based_review_and_revise import (
        _parse_tracking_column,
    )

    return _parse_tracking_column(raw or "")


def load_workbook(session: UserSession) -> Tuple[Any, Any, int]:
    helpers = _import_slideshow_helpers()
    safe_str = helpers["safe_str"]
    detect_current_round = helpers["detect_current_round"]

    if session.sheet is None:
        session.sheet = session.gc.open_by_url(session.sheet_link)
    ws, df = get_sheet_data_and_df(session.sheet, session.worksheet_name or DEFAULT_WORKSHEET)
    session.current_round = detect_current_round(df)

    _, course_df = get_sheet_data_and_df(session.sheet, "Course info")
    if not course_df.empty:
        session.course_name = safe_str(course_df.iloc[0].get("Course Name", ""))

    return ws, df, session.current_round


def slides_to_ui_payload(session: UserSession) -> Dict[str, Any]:
    helpers = _import_slideshow_helpers()
    build_slides_from_df = helpers["build_slides_from_df"]
    detect_asset_type = helpers["detect_asset_type"]
    get_round_column_name = helpers["get_round_column_name"]
    safe_str = helpers["safe_str"]

    _, df, current_round = load_workbook(session)
    cmap = _column_map(df)
    slides = build_slides_from_df(df, cmap)

    actions_col = get_round_column_name(HUMAN_REVIEW_ACTIONS_COLUMN, current_round)
    tracking_col = get_round_column_name(HUMAN_FEEDBACK_TRACKING_COLUMN, current_round)

    ui_slides: List[Dict[str, Any]] = []
    for slide_idx, slide in enumerate(slides):
        row = df.iloc[slide["row_index"]]
        actions_map = _parse_actions_payload(safe_str(row.get(actions_col, "")))
        tracking_map = _parse_tracking(safe_str(row.get(tracking_col, "")))

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

        ui_slides.append(
            {
                "title": safe_str(slide.get("slide_title", "")) or f"Slide {slide_idx + 1}",
                "topic": safe_str(slide.get("topic", "")),
                "segments": segments_out,
            }
        )

    module_label = session.course_name or "Course"
    if ui_slides and ui_slides[0].get("topic"):
        module_label = f"{session.course_name or 'Course'} · {ui_slides[0]['topic']}"

    return {
        "slides": ui_slides,
        "currentRound": current_round,
        "moduleLabel": module_label,
        "sheetLink": session.sheet_link,
        "worksheetName": session.worksheet_name,
    }


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

    ws, df, current_round = load_workbook(session)
    actions_col = get_round_column_name(HUMAN_REVIEW_ACTIONS_COLUMN, current_round)
    feedback_col = get_round_column_name(HUMAN_FEEDBACK_COLUMN, current_round)

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

    payload = {"actions": actions_map, "segment_modes": segment_modes, "round": current_round}
    df.at[row_index, actions_col] = json.dumps(payload, ensure_ascii=True)

    save_to_sheet(ws, df)


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
    from agents.graphics_definition_v2.review_agent.human_feedback_based_review_and_revise import (
        _format_human_feedback_revision_tracking,
    )
    from graphics_definition_v2_slideshow import _apply_asset_overrides_to_raw

    helpers = _import_slideshow_helpers()
    get_round_column_name = helpers["get_round_column_name"]
    parse_graphics_definition = helpers["parse_graphics_definition"]
    safe_str = helpers["safe_str"]

    asset_url = (asset_url or "").strip()
    if not asset_url:
        raise ValueError("asset_url is required")

    ws, df, current_round = load_workbook(session)
    cmap = _column_map(df)
    final_col = cmap["final_def"]
    tracking_col = get_round_column_name(HUMAN_FEEDBACK_TRACKING_COLUMN, current_round)

    if final_col not in df.columns:
        raise ValueError(f"Missing column: {final_col}")
    if tracking_col not in df.columns:
        df[tracking_col] = ""

    raw_def = safe_str(df.at[row_index, final_col])
    segments = parse_graphics_definition(raw_def)
    original_url = ""
    for seg in segments:
        if seg.get("segment_index") == segment_index:
            for step in seg.get("steps") or []:
                if step.get("step_index") == step_index:
                    original_url = safe_str(step.get("asset", "")).strip()
                    break

    override_map = {(segment_index, step_index): asset_url}
    updated_def = _apply_asset_overrides_to_raw(raw_def, segments, override_map)

    tracking_map = _parse_tracking(safe_str(df.at[row_index, tracking_col]))

    if visual_id not in tracking_map:
        tracking_map[visual_id] = {
            "original": original_url or None,
            "manually_selected": None,
            "after_revision": None,
            "after_regen_1": None,
            "after_regen_2": None,
        }
    elif not tracking_map[visual_id].get("original"):
        tracking_map[visual_id]["original"] = original_url or None

    tracking_map[visual_id]["manually_selected"] = asset_url

    df.at[row_index, final_col] = updated_def
    df.at[row_index, tracking_col] = _format_human_feedback_revision_tracking(tracking_map)
    save_to_sheet(ws, df)

    approve_visual(
        session,
        row_index=row_index,
        segment_index=segment_index,
        step_index=step_index,
        visual_id=visual_id,
        vo=vo,
    )


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
