"""Re-plan instructional overlays after a visual URL change.

Runs outside the visual-save write lock. Patches only the scene block(s) that
use the new image. If planning fails, existing overlay cells are left unchanged.
Does not create overlay columns that are not already on the sheet.

Same-slide refreshes run one at a time so a later revise reloads the sheet
after the earlier overlay write (avoids a stale last-writer on a shared scene).
"""

from __future__ import annotations

import re
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

from agents.graphics_definition_v2.image_editing_for_layout.hero_scene_parallel import (
    GLOBAL_API_SEMAPHORE_LIMIT,
)
from human_feedback_app.backend.config import LLM_DEFAULT
from human_feedback_app.backend.sessions import UserSession

LOG_PREFIX = "[overlay_refresh]"

_OVERLAY_ROW_LOCKS_GUARD = threading.Lock()
_OVERLAY_ROW_LOCKS: Dict[Tuple[str, int], threading.Lock] = {}
_OVERLAY_SCENE_API_SEMAPHORE = threading.Semaphore(GLOBAL_API_SEMAPHORE_LIMIT)


def _overlay_row_lock(session: UserSession, row_index: int) -> threading.Lock:
    key = (str(getattr(session, "session_id", "") or ""), int(row_index))
    with _OVERLAY_ROW_LOCKS_GUARD:
        lock = _OVERLAY_ROW_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _OVERLAY_ROW_LOCKS[key] = lock
        return lock


def _log(icon: str, message: str) -> None:
    print(f"{LOG_PREFIX} {icon} {message}", flush=True)

HERO_PLAN_COL = "hero_animation_plan"
HERO_BBOX_COL = "hero_bbox_coordinates"
HERO_ICONS_COL = "hero_icon_overlays"
HERO_CALLOUTS_COL = "hero_callout_overlays"
MULTI_PLAN_COL = "multivisual_animation_plan"
OVERLAY_COLUMNS = (
    HERO_PLAN_COL,
    HERO_BBOX_COL,
    HERO_ICONS_COL,
    HERO_CALLOUTS_COL,
    MULTI_PLAN_COL,
)

_SCENE_HEADER_RE = re.compile(r"---Scene ID:\s*(\S+)---", re.IGNORECASE)
_ANIMATION_TYPE_RE = re.compile(
    r"<animation_type>\s*(.*?)\s*</animation_type>",
    re.DOTALL | re.IGNORECASE,
)
_HIGHLIGHT_TAG_RE = re.compile(r"<highlight\b", re.IGNORECASE)

_HERO_TYPES = frozenset(
    {"none", "text_label", "bbox_highlight", "icon_overlay", "callout_card"}
)
_MULTI_TYPES = frozenset({"none", "text_label"})

_HERO_NONE_PLAN = (
    "<animation_type>none</animation_type>\n"
    "<reason>No instructional overlay is needed for this visual.</reason>\n"
    "<label_text>N/A</label_text>\n"
    "<trigger_phrase>N/A</trigger_phrase>\n"
    "<highlights>N/A</highlights>\n"
    "<icons>N/A</icons>\n"
    "<callouts>N/A</callouts>"
)
_MULTI_NONE_PLAN = (
    "<multivisual_animation_plan>\n"
    "<animation_type>none</animation_type>\n"
    "<reason>No panel labels are needed for this scene.</reason>\n"
    "<label_text>N/A</label_text>\n"
    "</multivisual_animation_plan>"
)


def split_scene_blocks(plan_text: str) -> List[Tuple[str, str]]:
    text = str(plan_text or "")
    matches = list(_SCENE_HEADER_RE.finditer(text))
    if not matches:
        return []
    blocks: List[Tuple[str, str]] = []
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        scene_id = (match.group(1) or "").strip()
        blocks.append((scene_id, text[start:end].strip()))
    return blocks


def join_scene_blocks(blocks: List[Tuple[str, str]]) -> str:
    parts = []
    for sid, body in blocks:
        body_text = str(body or "").strip()
        if not sid or not body_text:
            continue
        parts.append(f"---Scene ID: {sid}---\n{body_text}".strip())
    return "\n\n".join(parts)


def replace_scene_block(cell_text: str, scene_id: str, new_body: Optional[str]) -> str:
    """Replace or remove one ---Scene ID--- block. Other scenes stay as-is."""
    raw = str(cell_text or "").strip()
    if raw.lower() == "nan":
        raw = ""
    wanted = str(scene_id or "").strip()
    if not wanted:
        return raw

    blocks = split_scene_blocks(raw)
    out: List[Tuple[str, str]] = []
    found = False
    incoming = str(new_body or "").strip() if new_body is not None else ""
    for sid, body in blocks:
        if str(sid).strip() == wanted:
            found = True
            if incoming:
                out.append((sid, incoming))
            continue
        if str(body or "").strip():
            out.append((sid, body))

    if not found:
        if incoming:
            out.append((wanted, incoming))
        else:
            return raw
    return join_scene_blocks(out)


def _animation_type(block_text: str) -> str:
    match = _ANIMATION_TYPE_RE.search(block_text or "")
    if not match:
        return ""
    return (match.group(1) or "").strip().lower()


def _scene_uses_url(scene: Dict[str, Any], visual_url: str) -> bool:
    if not visual_url:
        return False
    from agents.graphics_definition_v2.slideshow_manifest.slideshow_manifest import (
        urls_match_for_graphics_assignment,
    )

    for slot in scene.get("slots") or []:
        asset = str((slot or {}).get("asset") or "").strip()
        if asset and urls_match_for_graphics_assignment(asset, visual_url):
            return True
    return False


def _cell_ok(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    if text.lower() == "nan":
        return ""
    return text


def _load_course_context(session: UserSession) -> Tuple[str, str]:
    try:
        from services.sheets_service import get_sheet_data_and_df

        if session.sheet is None:
            return "", ""
        _, course_info_df = get_sheet_data_and_df(session.sheet, "Course info")
        if course_info_df is None or course_info_df.empty:
            return "", ""
        row0 = course_info_df.iloc[0]
        course_name = str(row0.get("Course Name", "") or "").strip()
        audience = str(row0.get("Target Audience & Industry", "") or "").strip()
        return course_name, audience
    except Exception as exc:
        print(f"{LOG_PREFIX} Course info unavailable ({exc}); continuing without it")
        return "", ""


def _run_with_overlay_row_lock(session: UserSession, row_index: int, work) -> None:
    if getattr(session, "aborted", False):
        _log("⏭️", f"Session aborted — skip overlay work (row {row_index})")
        return
    from human_feedback_app.backend.streamlit_shim import (
        agent_session_context,
        bind_user_session,
    )

    ctx = bind_user_session(
        drive=session.drive,
        gc=session.gc,
        user_email=session.user_email,
        role=session.role,
        sheet_link=session.sheet_link,
        root_folder_id=session.root_folder_id,
    )
    row_lock = _overlay_row_lock(session, row_index)
    if not row_lock.acquire(blocking=False):
        _log("⏳", f"Waiting for another overlay job on this slide (row {row_index})")
        row_lock.acquire()
    try:
        if getattr(session, "aborted", False):
            _log("⏭️", f"Session aborted — skip overlay work (row {row_index})")
            return
        with agent_session_context(ctx):
            work()
    finally:
        row_lock.release()


def refresh_overlays_after_visual_change(
    session: UserSession,
    row_index: int,
    visual_url: str,
) -> None:
    """Re-plan overlays for scenes that now show ``visual_url``. Never raises."""
    url = (visual_url or "").strip()
    if not url:
        return
    try:
        _run_with_overlay_row_lock(
            session,
            row_index,
            lambda: _refresh_overlays_locked_out(session, row_index, url),
        )
    except Exception as exc:
        _log("❌", f"Overlay refresh failed (row {row_index}): {exc}")
        traceback.print_exc()


def rebuild_overlays_for_row(session: UserSession, row_index: int) -> None:
    """Wipe this row's overlay columns and replan every scene. Never raises.

    Use after a full ``slideshow_manifest`` regeneration. A one-URL patch
    should keep using ``refresh_overlays_after_visual_change`` instead.
    """
    try:
        _run_with_overlay_row_lock(
            session,
            row_index,
            lambda: _rebuild_overlays_locked_out(session, row_index),
        )
    except Exception as exc:
        _log("❌", f"Overlay rebuild failed (row {row_index}): {exc}")
        traceback.print_exc()


def _refresh_overlays_locked_out(session: UserSession, row_index: int, visual_url: str) -> None:
    from agents.graphics_definition_v2.image_editing_for_layout.hero_animation_decision import (
        _hero_primary_asset_url,
        _is_single_visual_hero_scene,
        _is_video_asset_url,
        generate_hero_animation_decision_for_scene,
    )
    from agents.graphics_definition_v2.image_editing_for_layout.image_edit_planning import (
        parse_scenes_from_slideshow_manifest,
    )
    from agents.graphics_definition_v2.image_editing_for_layout.multivisual_animation_decision import (
        _is_multivisual_scene,
        generate_multivisual_animation_decision_for_scene,
    )
    from human_feedback_app.backend.sheet_service import load_workbook, mutate_row_cells

    _log("🎬", f"Starting overlay calculation (row {row_index})")
    _, df, _ = load_workbook(session)
    if row_index not in df.index:
        _log("⏭️", f"Row {row_index} missing — skip")
        return

    columns = set(df.columns)
    has_hero_plan = HERO_PLAN_COL in columns
    has_multi_plan = MULTI_PLAN_COL in columns
    if not has_hero_plan and not has_multi_plan:
        _log("⏭️", "No overlay plan columns on this sheet — skip")
        return

    row = df.loc[row_index]
    manifest = _cell_ok(row.get("slideshow_manifest", ""))
    if not manifest or manifest.startswith("ERROR:"):
        _log("⏭️", f"No usable slideshow_manifest (row {row_index}) — skip")
        return

    scenes = parse_scenes_from_slideshow_manifest(manifest)
    affected = []
    for fallback_idx, scene in enumerate(scenes or [], start=1):
        if not _scene_uses_url(scene, visual_url):
            continue
        scene_id = str(scene.get("id") or "").strip() or str(fallback_idx)
        affected.append((scene_id, scene))

    if not affected:
        _log("⏭️", f"No scene uses this revised visual (row {row_index}) — skip")
        return

    scene_ids = ", ".join(sid for sid, _ in affected)
    _log("🔎", f"Found {len(affected)} scene(s) to update: {scene_ids}")

    course_name, target_audience = _load_course_context(session)
    topic_name = _cell_ok(row.get("Topic", ""))
    subtopic_name = _cell_ok(row.get("Subtopic", ""))
    slide_title = _cell_ok(row.get("Slide Chunk Title", ""))
    slide_content = _cell_ok(row.get("Slide Chunk", ""))
    slide_type = _cell_ok(row.get("Slide Type", ""))
    if slide_type.lower() == "nan":
        slide_type = ""
    fgd = _cell_ok(row.get("final_graphics_definition", ""))
    is_transition = slide_type.lower() == "transition"

    patches: List[Tuple[str, str, Optional[str]]] = []
    for scene_id, scene in affected:
        if getattr(session, "aborted", False):
            _log("⏭️", f"Aborted before scene {scene_id} — keep existing overlays")
            return
        try:
            scene_patches = _plan_one_scene(
                session=session,
                scene_id=scene_id,
                scene=scene,
                has_hero_plan=has_hero_plan,
                has_multi_plan=has_multi_plan,
                columns=columns,
                is_transition=is_transition,
                course_name=course_name,
                target_audience=target_audience,
                topic_name=topic_name,
                subtopic_name=subtopic_name,
                slide_type=slide_type,
                slide_title=slide_title,
                slide_content=slide_content,
                fgd=fgd,
                generate_hero_animation_decision_for_scene=generate_hero_animation_decision_for_scene,
                generate_multivisual_animation_decision_for_scene=generate_multivisual_animation_decision_for_scene,
                is_hero=_is_single_visual_hero_scene,
                is_multi=_is_multivisual_scene,
                is_video=_is_video_asset_url,
                hero_primary=_hero_primary_asset_url,
            )
        except Exception as exc:
            _log("⚠️", f"Scene {scene_id} failed — leaving that scene's overlays unchanged ({exc})")
            traceback.print_exc()
            continue
        patches.extend(scene_patches)

    if not patches:
        _log("⏭️", "Nothing to write to the sheet")
        return
    if getattr(session, "aborted", False):
        _log("⏭️", f"Aborted before sheet write (row {row_index}) — keep existing overlays")
        return

    def mutate(latest_df) -> None:
        from graphics_definition_v2_slideshow import safe_str

        for col, sid, body in patches:
            if col not in latest_df.columns:
                continue
            current = safe_str(latest_df.at[row_index, col])
            latest_df.at[row_index, col] = replace_scene_block(current, sid, body)
            action = "cleared scene block in" if body is None else "replaced scene block in"
            _log("💾", f'Saving to sheet column "{col}" — {action} scene {sid}')

    mutate_row_cells(session, row_index, mutate)
    _log("✅", f"Overlay update complete (row {row_index})")


def _rebuild_overlays_locked_out(session: UserSession, row_index: int) -> None:
    from agents.graphics_definition_v2.image_editing_for_layout.hero_animation_decision import (
        _hero_primary_asset_url,
        _is_single_visual_hero_scene,
        _is_video_asset_url,
        generate_hero_animation_decision_for_scene,
    )
    from agents.graphics_definition_v2.image_editing_for_layout.image_edit_planning import (
        parse_scenes_from_slideshow_manifest,
    )
    from agents.graphics_definition_v2.image_editing_for_layout.multivisual_animation_decision import (
        _is_multivisual_scene,
        generate_multivisual_animation_decision_for_scene,
    )
    from human_feedback_app.backend.sheet_service import load_workbook, mutate_row_cells

    _log("🎬", f"Starting full overlay rebuild (row {row_index})")
    _, df, _ = load_workbook(session)
    if row_index not in df.index:
        _log("⏭️", f"Row {row_index} missing — skip rebuild")
        return

    columns = set(df.columns)
    overlay_cols = [col for col in OVERLAY_COLUMNS if col in columns]
    if not overlay_cols:
        _log("⏭️", "No overlay plan columns on this sheet — skip rebuild")
        return

    row = df.loc[row_index]
    manifest = _cell_ok(row.get("slideshow_manifest", ""))
    if not manifest or manifest.startswith("ERROR:"):
        _log("⏭️", f"No usable slideshow_manifest (row {row_index}) — skip rebuild")
        return

    scenes = parse_scenes_from_slideshow_manifest(manifest) or []
    has_hero_plan = HERO_PLAN_COL in columns
    has_multi_plan = MULTI_PLAN_COL in columns
    course_name, target_audience = _load_course_context(session)
    topic_name = _cell_ok(row.get("Topic", ""))
    subtopic_name = _cell_ok(row.get("Subtopic", ""))
    slide_title = _cell_ok(row.get("Slide Chunk Title", ""))
    slide_content = _cell_ok(row.get("Slide Chunk", ""))
    slide_type = _cell_ok(row.get("Slide Type", ""))
    if slide_type.lower() == "nan":
        slide_type = ""
    fgd = _cell_ok(row.get("final_graphics_definition", ""))
    is_transition = slide_type.lower() == "transition"

    if not scenes:
        _log("⚠️", f"No scenes in new manifest (row {row_index}) — clearing overlay columns")

    plan_kwargs = {
        "has_hero_plan": has_hero_plan,
        "has_multi_plan": has_multi_plan,
        "columns": columns,
        "is_transition": is_transition,
        "course_name": course_name,
        "target_audience": target_audience,
        "topic_name": topic_name,
        "subtopic_name": subtopic_name,
        "slide_type": slide_type,
        "slide_title": slide_title,
        "slide_content": slide_content,
        "fgd": fgd,
        "generate_hero_animation_decision_for_scene": generate_hero_animation_decision_for_scene,
        "generate_multivisual_animation_decision_for_scene": generate_multivisual_animation_decision_for_scene,
        "is_hero": _is_single_visual_hero_scene,
        "is_multi": _is_multivisual_scene,
        "is_video": _is_video_asset_url,
        "hero_primary": _hero_primary_asset_url,
    }
    scene_jobs = []
    for fallback_idx, scene in enumerate(scenes, start=1):
        scene_id = str(scene.get("id") or "").strip() or str(fallback_idx)
        scene_jobs.append((scene_id, scene))

    patches: List[Tuple[str, str, Optional[str]]] = []
    if len(scene_jobs) <= 1:
        for scene_id, scene in scene_jobs:
            if getattr(session, "aborted", False):
                _log("⏭️", f"Aborted during overlay rebuild (row {row_index}) — keep existing overlays")
                return
            patches.extend(
                _rebuild_one_scene_patches(session, scene_id, scene, plan_kwargs)
            )
    else:
        workers = len(scene_jobs)
        _log(
            "⚡",
            f"Planning {workers} scene(s) in parallel on this slide "
            f"(row {row_index}, api_cap={GLOBAL_API_SEMAPHORE_LIMIT})"
        )
        ordered_patches: List[Optional[List[Tuple[str, str, Optional[str]]]]] = [
            None
        ] * workers
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    _rebuild_one_scene_patches_worker,
                    session,
                    scene_id,
                    scene,
                    plan_kwargs,
                ): i
                for i, (scene_id, scene) in enumerate(scene_jobs)
            }
            for future in as_completed(futures):
                i = futures[future]
                scene_id, scene = scene_jobs[i]
                try:
                    ordered_patches[i] = future.result()
                except Exception as exc:
                    _log("⚠️", f"Scene {scene_id} failed during rebuild — writing none ({exc})")
                    traceback.print_exc()
                    ordered_patches[i] = _fallback_rebuild_patches(
                        scene_id, scene, plan_kwargs
                    )
                if getattr(session, "aborted", False):
                    _log(
                        "⏭️",
                        f"Aborted during overlay rebuild (row {row_index}) — keep existing overlays",
                    )
                    return
        for scene_patches in ordered_patches:
            patches.extend(scene_patches or [])

    if getattr(session, "aborted", False):
        _log("⏭️", f"Aborted before overlay rebuild write (row {row_index}) — keep existing overlays")
        return

    values = _full_overlay_column_values(patches, overlay_cols)

    def mutate(latest_df) -> None:
        for col in overlay_cols:
            if col not in latest_df.columns:
                continue
            latest_df.at[row_index, col] = values.get(col, "")
            _log("💾", f'Saving rebuilt overlay column "{col}" (row {row_index})')

    mutate_row_cells(session, row_index, mutate)
    _log("✅", f"Overlay rebuild complete (row {row_index})")


def _fallback_rebuild_patches(scene_id, scene, plan_kwargs):
    if plan_kwargs["has_hero_plan"] and plan_kwargs["is_hero"](scene):
        return _hero_none_patches(scene_id, plan_kwargs["columns"])
    if plan_kwargs["has_multi_plan"] and plan_kwargs["is_multi"](scene):
        return [(MULTI_PLAN_COL, scene_id, _MULTI_NONE_PLAN)]
    return []


def _rebuild_one_scene_patches(session, scene_id, scene, plan_kwargs):
    """Plan one rebuild scene. Same none-fallback as the old sequential loop."""
    try:
        return _plan_one_scene(
            session=session,
            scene_id=scene_id,
            scene=scene,
            **plan_kwargs,
        )
    except Exception as exc:
        _log("⚠️", f"Scene {scene_id} failed during rebuild — writing none ({exc})")
        traceback.print_exc()
        return _fallback_rebuild_patches(scene_id, scene, plan_kwargs)


def _rebuild_one_scene_patches_worker(session, scene_id, scene, plan_kwargs):
    """Inner-slide worker: bind agent context and take one shared API slot."""
    from human_feedback_app.backend.streamlit_shim import (
        agent_session_context,
        bind_user_session,
    )

    if getattr(session, "aborted", False):
        return []
    ctx = bind_user_session(
        drive=session.drive,
        gc=session.gc,
        user_email=session.user_email,
        role=session.role,
        sheet_link=session.sheet_link,
        root_folder_id=session.root_folder_id,
    )
    with agent_session_context(ctx):
        with _OVERLAY_SCENE_API_SEMAPHORE:
            if getattr(session, "aborted", False):
                return []
            return _rebuild_one_scene_patches(session, scene_id, scene, plan_kwargs)


def _full_overlay_column_values(
    patches: List[Tuple[str, str, Optional[str]]],
    overlay_cols: List[str],
) -> Dict[str, str]:
    by_col: Dict[str, List[Tuple[str, str]]] = {col: [] for col in overlay_cols}
    for col, sid, body in patches:
        if col not in by_col:
            continue
        incoming = str(body or "").strip()
        if incoming:
            by_col[col].append((sid, incoming))
    return {col: join_scene_blocks(blocks) for col, blocks in by_col.items()}


def _plan_one_scene(
    *,
    session: UserSession,
    scene_id: str,
    scene: Dict[str, Any],
    has_hero_plan: bool,
    has_multi_plan: bool,
    columns: set,
    is_transition: bool,
    course_name: str,
    target_audience: str,
    topic_name: str,
    subtopic_name: str,
    slide_type: str,
    slide_title: str,
    slide_content: str,
    fgd: str,
    generate_hero_animation_decision_for_scene,
    generate_multivisual_animation_decision_for_scene,
    is_hero,
    is_multi,
    is_video,
    hero_primary,
) -> List[Tuple[str, str, Optional[str]]]:
    if is_hero(scene):
        if not has_hero_plan:
            _log("⏭️", f"Scene {scene_id}: hero plan column missing — skip")
            return []
        primary = hero_primary(scene)
        if is_transition or is_video(primary):
            reason = "transition slide" if is_transition else "video scene"
            _log("📝", f"Scene {scene_id}: {reason} — overlay type none")
            return _hero_none_patches(scene_id, columns)
        _log("📝", f"Scene {scene_id}: generating overlay plan (hero layout)...")
        _, plan_inner = generate_hero_animation_decision_for_scene(
            course_name=course_name,
            target_audience=target_audience,
            topic_name=topic_name,
            subtopic_name=subtopic_name,
            slide_type=slide_type,
            slide_title=slide_title,
            slide_content=slide_content,
            scene=scene,
            drive=session.drive,
            llm=LLM_DEFAULT,
        )
        plan_inner = str(plan_inner or "").strip()
        if not plan_inner or plan_inner.startswith("ERROR:"):
            raise ValueError(f"hero plan empty or error for scene {scene_id}")
        atype = _animation_type(plan_inner)
        if atype not in _HERO_TYPES:
            raise ValueError(f"invalid hero animation_type {atype!r} for scene {scene_id}")
        _log("🎯", f"Scene {scene_id}: overlay type = {atype}")
        return _hero_followup_patches(
            session=session,
            scene_id=scene_id,
            scene=scene,
            plan_inner=plan_inner,
            atype=atype,
            columns=columns,
            hero_primary=hero_primary,
        )

    if is_multi(scene):
        if not has_multi_plan:
            _log("⏭️", f"Scene {scene_id}: multi-visual plan column missing — skip")
            return []
        if is_transition:
            _log("📝", f"Scene {scene_id}: transition slide — overlay type none")
            return [(MULTI_PLAN_COL, scene_id, _MULTI_NONE_PLAN)]
        _log("📝", f"Scene {scene_id}: generating overlay plan (split/grid/inset)...")
        _, plan_text = generate_multivisual_animation_decision_for_scene(
            course_name=course_name,
            target_audience=target_audience,
            topic_name=topic_name,
            subtopic_name=subtopic_name,
            slide_type=slide_type,
            slide_title=slide_title,
            slide_content=slide_content,
            scene=scene,
            final_graphics_definition=fgd,
            drive=session.drive,
            llm=LLM_DEFAULT,
        )
        plan_text = str(plan_text or "").strip()
        if not plan_text or plan_text.startswith("ERROR:"):
            raise ValueError(f"multivisual plan empty or error for scene {scene_id}")
        atype = _animation_type(plan_text)
        if atype not in _MULTI_TYPES:
            raise ValueError(
                f"invalid multivisual animation_type {atype!r} for scene {scene_id}"
            )
        _log("🎯", f"Scene {scene_id}: overlay type = {atype}")
        return [(MULTI_PLAN_COL, scene_id, plan_text)]

    return []


def _hero_none_patches(scene_id: str, columns: set) -> List[Tuple[str, str, Optional[str]]]:
    patches: List[Tuple[str, str, Optional[str]]] = []
    if HERO_PLAN_COL in columns:
        patches.append((HERO_PLAN_COL, scene_id, _HERO_NONE_PLAN))
    if HERO_BBOX_COL in columns:
        patches.append((HERO_BBOX_COL, scene_id, None))
    if HERO_ICONS_COL in columns:
        patches.append((HERO_ICONS_COL, scene_id, None))
    if HERO_CALLOUTS_COL in columns:
        patches.append((HERO_CALLOUTS_COL, scene_id, None))
    return patches


def _hero_followup_patches(
    *,
    session: UserSession,
    scene_id: str,
    scene: Dict[str, Any],
    plan_inner: str,
    atype: str,
    columns: set,
    hero_primary,
) -> List[Tuple[str, str, Optional[str]]]:
    from agents.graphics_definition_v2.image_editing_for_layout.hero_bbox_spatial import (
        generate_hero_bbox_coordinates_for_scene,
        parse_bbox_highlights_from_plan_block,
    )
    from agents.graphics_definition_v2.image_editing_for_layout.hero_icon_generation import (
        _generate_callout_scene_inner,
        _generate_icon_scene_inner,
        parse_callouts_from_plan_block,
        parse_icon_overlays_from_plan_block,
    )

    if atype == "none":
        return _hero_none_patches(scene_id, columns)

    if atype == "text_label":
        patches: List[Tuple[str, str, Optional[str]]] = []
        if HERO_PLAN_COL in columns:
            patches.append((HERO_PLAN_COL, scene_id, plan_inner))
        if HERO_BBOX_COL in columns:
            patches.append((HERO_BBOX_COL, scene_id, None))
        if HERO_ICONS_COL in columns:
            patches.append((HERO_ICONS_COL, scene_id, None))
        if HERO_CALLOUTS_COL in columns:
            patches.append((HERO_CALLOUTS_COL, scene_id, None))
        return patches

    if atype == "bbox_highlight":
        highlights = parse_bbox_highlights_from_plan_block(plan_inner)
        if not highlights:
            _log("⚠️", f"Scene {scene_id}: bbox chosen but no targets — using none")
            return _hero_none_patches(scene_id, columns)
        _log("📦", f"Scene {scene_id}: calculating bbox boxes on the new image...")
        asset_url = hero_primary(scene)
        bbox_inner = generate_hero_bbox_coordinates_for_scene(
            scene_id=scene_id,
            asset_url=asset_url,
            highlights=highlights,
            drive=session.drive,
            llm=LLM_DEFAULT,
        )
        bbox_inner = str(bbox_inner or "").strip()
        if not bbox_inner or bbox_inner.startswith("ERROR:") or not _HIGHLIGHT_TAG_RE.search(bbox_inner):
            _log("⚠️", f"Scene {scene_id}: bbox locate produced no boxes — using none")
            return _hero_none_patches(scene_id, columns)
        patches = []
        if HERO_PLAN_COL in columns:
            patches.append((HERO_PLAN_COL, scene_id, plan_inner))
        if HERO_BBOX_COL in columns:
            patches.append((HERO_BBOX_COL, scene_id, bbox_inner))
        if HERO_ICONS_COL in columns:
            patches.append((HERO_ICONS_COL, scene_id, None))
        if HERO_CALLOUTS_COL in columns:
            patches.append((HERO_CALLOUTS_COL, scene_id, None))
        return patches

    if atype == "icon_overlay":
        if not parse_icon_overlays_from_plan_block(plan_inner):
            _log("⚠️", f"Scene {scene_id}: icon chosen but no specs — using none")
            return _hero_none_patches(scene_id, columns)
        _log("🧩", f"Scene {scene_id}: generating icon overlays...")
        icons_inner = _generate_icon_scene_inner(scene_id, plan_inner, session.drive)
        icons_inner = str(icons_inner or "").strip()
        if not icons_inner or icons_inner.startswith("ERROR:"):
            raise ValueError(f"icon generation failed for scene {scene_id}")
        patches = []
        if HERO_PLAN_COL in columns:
            patches.append((HERO_PLAN_COL, scene_id, plan_inner))
        if HERO_BBOX_COL in columns:
            patches.append((HERO_BBOX_COL, scene_id, None))
        if HERO_ICONS_COL in columns:
            patches.append((HERO_ICONS_COL, scene_id, icons_inner))
        if HERO_CALLOUTS_COL in columns:
            patches.append((HERO_CALLOUTS_COL, scene_id, None))
        return patches

    if atype == "callout_card":
        if not parse_callouts_from_plan_block(plan_inner):
            _log("⚠️", f"Scene {scene_id}: callout chosen but no cards — using none")
            return _hero_none_patches(scene_id, columns)
        _log("💬", f"Scene {scene_id}: generating callout cards...")
        callouts_inner = _generate_callout_scene_inner(scene_id, plan_inner, session.drive)
        callouts_inner = str(callouts_inner or "").strip()
        if not callouts_inner or callouts_inner.startswith("ERROR:"):
            raise ValueError(f"callout generation failed for scene {scene_id}")
        patches = []
        if HERO_PLAN_COL in columns:
            patches.append((HERO_PLAN_COL, scene_id, plan_inner))
        if HERO_BBOX_COL in columns:
            patches.append((HERO_BBOX_COL, scene_id, None))
        if HERO_ICONS_COL in columns:
            patches.append((HERO_ICONS_COL, scene_id, None))
        if HERO_CALLOUTS_COL in columns:
            patches.append((HERO_CALLOUTS_COL, scene_id, callouts_inner))
        return patches

    return _hero_none_patches(scene_id, columns)
