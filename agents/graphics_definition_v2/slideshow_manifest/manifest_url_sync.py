"""Check slideshow_manifest URLs against final_graphics_definition before overlay planning.

Pipeline-only: uses the gspread sheet, not the human-feedback session checker.
Repaired rows get overlay plan columns cleared so Decide Overlay Animations
re-plans from the new manifest instead of skipping a stale plan.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

from agents.graphics_definition_v2.aggregation_agent.aggregation_agent import (
    get_drive_instance,
)
from agents.graphics_definition_v2.image_editing_for_layout.image_edit_planning import (
    parse_scenes_from_slideshow_manifest,
)
from agents.graphics_definition_v2.slideshow_manifest.slideshow_manifest import (
    generate_slideshow_manifest_for_row,
    parse_when_vo_assigned_pairs,
    urls_match_for_graphics_assignment,
)
from services.sheets_service import get_sheet_data_and_df, save_to_sheet

LOG_PREFIX = "[overlay_precheck]"
MANIFEST_COL = "slideshow_manifest"
FGD_COL = "final_graphics_definition"
MAX_REGEN_ATTEMPTS = 3
OVERLAY_COLUMNS = (
    "hero_animation_plan",
    "hero_bbox_coordinates",
    "hero_icon_overlays",
    "hero_callout_overlays",
    "multivisual_animation_plan",
)


def _cell(row, name):
    value = row.get(name, "") if hasattr(row, "get") else ""
    text = "" if value is None else str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    return text


def _pick_column(df, name):
    target = str(name).strip().lower()
    for col in df.columns:
        if str(col).strip().lower() == target:
            return col
    return name


def _url_token(url):
    text = (url or "").strip()
    return text.split()[0] if text else ""


def _urls_match(url_a, url_b):
    token_a = _url_token(url_a)
    token_b = _url_token(url_b)
    if not token_a or not token_b:
        return False
    if token_a.lower() == token_b.lower():
        return True
    try:
        return urls_match_for_graphics_assignment(token_a, token_b)
    except Exception:
        return token_a.lower() == token_b.lower()


def _fgd_urls(final_graphics_definition):
    pairs = parse_when_vo_assigned_pairs(final_graphics_definition or "")
    return [str(url or "").strip() for _, url in pairs if str(url or "").strip()]


def _manifest_slot_urls(manifest_xml):
    urls = []
    for scene in parse_scenes_from_slideshow_manifest(manifest_xml) or []:
        for slot in scene.get("slots") or []:
            asset = str((slot or {}).get("asset") or "").strip()
            if asset:
                urls.append(asset)
    return urls


def validate_slideshow_manifest_sync(manifest_xml, final_graphics_definition):
    """
    Validate the slideshow manifest against the final graphics definition.

    :param manifest_xml: The slideshow manifest XML.
    :param final_graphics_definition: The final graphics definition.
    :return: A tuple containing a boolean indicating success or failure and an error message.
    :rtype: tuple
    """
    fgd_urls = _fgd_urls(final_graphics_definition)
    if not fgd_urls:
        return True, "No FGD URLs to validate."

    manifest_urls = _manifest_slot_urls(manifest_xml)
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


def _row_label(row, row_index):
    title = _cell(row, "Slide Chunk Title")
    if not title:
        title = f"Slide {row_index + 1}"
    return f"row {row_index + 1} ({title})"


def _regen_manifest_for_row(row, course_name, drive, llm):
    fgd = _cell(row, FGD_COL)
    if not fgd:
        return None, "empty_fgd"
    try:
        manifest_xml, _ = generate_slideshow_manifest_for_row(
            course_name=course_name or "Course",
            topic_name=_cell(row, "Topic"),
            subtopic_name=_cell(row, "Subtopic"),
            slide_type=_cell(row, "Slide Type"),
            slide_title=_cell(row, "Slide Chunk Title"),
            slide_content=_cell(row, "Slide Chunk"),
            layout_plan="",
            storyboard_planning="",
            final_graphics_definition=fgd,
            drive=drive,
            llm=llm,
        )
        manifest_xml = str(manifest_xml or "").strip()
        if not manifest_xml or manifest_xml.startswith("ERROR:"):
            return None, manifest_xml or "regeneration_failed"
        ok, err = validate_slideshow_manifest_sync(manifest_xml, fgd)
        if not ok:
            return None, err or "regenerated_manifest_invalid"
        return manifest_xml, ""
    except Exception as exc:
        return None, str(exc)


def _repair_one_row(row_index, row_snapshot, course_name, drive, llm):
    label = _row_label(row_snapshot, row_index)
    last_error = "regeneration_failed"
    for attempt in range(1, MAX_REGEN_ATTEMPTS + 1):
        print(
            f"{LOG_PREFIX} Regeneration attempt {attempt}/{MAX_REGEN_ATTEMPTS} for {label}",
            flush=True,
        )
        manifest_xml, err = _regen_manifest_for_row(row_snapshot, course_name, drive, llm)
        if manifest_xml:
            print(f"{LOG_PREFIX} Regeneration succeeded for {label}", flush=True)
            return {"row_index": row_index, "ok": True, "manifest": manifest_xml}
        last_error = err or last_error
        print(f"{LOG_PREFIX} Attempt {attempt} failed for {label}: {last_error}", flush=True)
    print(
        f"{LOG_PREFIX} Regeneration exhausted for {label}: {last_error}",
        flush=True,
    )
    return {"row_index": row_index, "ok": False, "error": last_error}


def _clear_overlay_cells(df, row_index):
    cleared = []
    for col in OVERLAY_COLUMNS:
        if col not in df.columns:
            continue
        df.at[row_index, col] = ""
        cleared.append(col)
    return cleared


def ensure_slideshow_manifests_synced_for_overlay_step(sheet, llm="gemini_3_flash_thinking", max_workers=50):
    """
    Fix out-of-sync manifests, then return. Overlay planning runs after this.

    :param sheet: The worksheet to check.
    :param llm: The LLM to use for regeneration.
    :param max_workers: The maximum number of workers to use for regeneration.
    :return: None
    """
    worksheet_name = "Slide Chunks"
    print(
        f"{LOG_PREFIX} Checking slideshow_manifest vs assigned visuals "
        f"(max_workers={max_workers})...",
        flush=True,
    )

    ws, df = get_sheet_data_and_df(sheet, worksheet_name)
    if MANIFEST_COL not in df.columns:
        print(f"{LOG_PREFIX} No {MANIFEST_COL} column — skip precheck", flush=True)
        return {"checked": 0, "repaired": 0, "failed": 0}

    fgd_col = _pick_column(df, FGD_COL)
    if fgd_col not in df.columns:
        print(f"{LOG_PREFIX} No {FGD_COL} column — skip precheck", flush=True)
        return {"checked": 0, "repaired": 0, "failed": 0}

    _, course_info_df = get_sheet_data_and_df(sheet, "Course info")
    course_name = ""
    if course_info_df is not None and not course_info_df.empty:
        course_name = str(course_info_df.iloc[0].get("Course Name", "") or "").strip()

    drive = get_drive_instance()
    check_jobs = []
    snapshots = {}
    fgd_by_row = {}
    for raw_index, row in df.iterrows():
        row_index = int(raw_index)
        fgd = _cell(row, fgd_col)
        manifest_xml = _cell(row, MANIFEST_COL)
        check_jobs.append((row_index, manifest_xml, fgd))
        snapshots[row_index] = row.to_dict() if hasattr(row, "to_dict") else dict(row)
        if fgd_col != FGD_COL:
            snapshots[row_index][FGD_COL] = fgd
        fgd_by_row[row_index] = fgd

    needs_repair = []
    checked = 0
    workers = max(1, int(max_workers or 50))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(validate_slideshow_manifest_sync, manifest_xml, fgd): row_index
            for row_index, manifest_xml, fgd in check_jobs
        }
        for future in as_completed(futures):
            row_index = int(futures[future])
            if not fgd_by_row.get(row_index):
                continue
            checked += 1
            ok, err = future.result()
            if not ok:
                needs_repair.append(row_index)
                label = _row_label(snapshots[row_index], row_index)
                print(f"{LOG_PREFIX} NEEDS REPAIR: {label} — {err}", flush=True)

    print(
        f"{LOG_PREFIX} Check complete: {checked} row(s) checked, "
        f"{len(needs_repair)} need regeneration",
        flush=True,
    )
    if not needs_repair:
        print(f"{LOG_PREFIX} All checked rows are in sync — overlay can start", flush=True)
        return {"checked": checked, "repaired": 0, "failed": 0}

    repaired = 0
    failed = 0
    print(
        f"{LOG_PREFIX} Regenerating {len(needs_repair)} row(s) in parallel "
        f"(max_workers={workers})...",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                _repair_one_row,
                row_index,
                snapshots[row_index],
                course_name,
                drive,
                llm,
            )
            for row_index in needs_repair
        ]
        for future in as_completed(futures):
            result = future.result()
            row_index = int(result.get("row_index", -1))
            if result.get("ok") and result.get("manifest"):
                df.at[row_index, MANIFEST_COL] = result["manifest"]
                cleared = _clear_overlay_cells(df, row_index)
                repaired += 1
                extra = f"; cleared overlay columns {cleared}" if cleared else ""
                print(
                    f"{LOG_PREFIX} Saved new manifest for "
                    f"{_row_label(snapshots.get(row_index, {}), row_index)}{extra}",
                    flush=True,
                )
            else:
                failed += 1

    if repaired:
        save_to_sheet(ws, df)
    print(
        f"{LOG_PREFIX} Precheck finished: checked={checked}, repaired={repaired}, "
        f"failed={failed}",
        flush=True,
    )
    return {"checked": checked, "repaired": repaired, "failed": failed}
