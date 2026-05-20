"""
Build the "Image Edit results" worksheet from image_editing_tracking + scene_edit_plan.
"""
import re
import traceback

import streamlit as st
from langsmith import traceable

from agents.graphics_definition_v2.image_editing_for_layout.image_editing_based_on_edit_planning import (
    extract_slot_edit_blocks,
    split_scene_edit_plan_by_scene,
)
from agents.graphics_definition_v2.review_agent.visual_columns_for_human_feedback import (
    _cell_value_for_url,
    _drive_url_to_view_url,
)
from agents.graphics_definition_v2.slideshow_manifest.slideshow_manifest import (
    normalize_when_vo_line,
    parse_image_editing_tracking_edited_pairs,
    parse_when_vo_assigned_pairs,
    urls_match_for_graphics_assignment,
)
from services.sheets_service import (
    create_or_read_worksheet,
    delete_worksheet,
    format_worksheet,
    get_sheet_data_and_df,
    get_worksheet_names,
)
from services.smart_progress_bar import SmartProgressBar

IMAGE_EDIT_RESULTS_WS = "Image Edit results"

HEADERS = [
    "Whole Slide Content",
    "Voiceover",
    "Original Image",
    "Edit plan for the image",
    "Edited Image",
]


def _cell_text_literal(val):
    """
    Force plain text in Sheets when value could be interpreted as a formula.

    :param val: Cell value.
    :return: Safe string for non-formula columns.
    """
    if val is None:
        return ""
    t = str(val)
    if t in ("nan", "NaN", "None"):
        return ""
    if t.startswith("="):
        return "'" + t
    return t


def _extract_edits_inner_from_slot_xml(slot_xml):
    """
    Return inner XML of the first <edits>...</edits> block in a slot_edit fragment.

    :param slot_xml: Inner text of one <slot_edit> block.
    :return: Inner string or empty.
    """
    if not slot_xml:
        return ""
    m = re.search(r"<edits>(.*?)</edits>", slot_xml, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""


def _format_edit_plan_for_sheet(edits_inner):
    """
    Add a blank line between adjacent XML tags for readability in the sheet.

    :param edits_inner: Content inside <edits>...</edits>.
    :return: Formatted string.
    """
    if not edits_inner:
        return ""
    s = edits_inner.strip()
    return re.sub(r">\s*<", ">\n\n<", s)


def _vo_for_asset_url(fgd_text, asset_url):
    """
    Match When VO text to an asset URL using final_graphics_definition pairs.

    :param fgd_text: final_graphics_definition cell text.
    :param asset_url: Original image URL from tracking.
    :return: Normalized voiceover line or empty string.
    """
    if not asset_url:
        return ""
    for vo, pu in parse_when_vo_assigned_pairs(fgd_text or ""):
        if urls_match_for_graphics_assignment(asset_url, pu):
            return normalize_when_vo_line(vo) or ""
    return ""


def _slot_xml_for_scene_and_image_index(scene_edit_plan_text, scene_id, image_k):
    """
    Return the slot_edit inner XML for scene_id and 1-based image index k.

    :param scene_edit_plan_text: Full scene_edit_plan cell.
    :param scene_id: Scene id string matching ---Scene ID--- header.
    :param image_k: 1-based slot index within that scene in plan order.
    :return: slot_edit inner XML or None.
    """
    sid = str(scene_id).strip()
    for bid, body in split_scene_edit_plan_by_scene(scene_edit_plan_text or ""):
        if str(bid).strip() != sid:
            continue
        slots = extract_slot_edit_blocks(body)
        if 1 <= image_k <= len(slots):
            return slots[image_k - 1]
    return None


def _build_rows_from_slide_chunk_row(row):
    """
    Build output grid rows (values only, no header) for one Slide Chunks row.

    :param row: pandas Series for one slide chunk.
    :return: List of dicts with keys cells (5-list), original_url, edited_url for notes.
    """
    tracking = str(row.get("image_editing_tracking", "")).strip()
    plan = str(row.get("scene_edit_plan", "")).strip()
    chunk = str(row.get("Slide Chunk", "")).strip()
    fgd = str(row.get("final_graphics_definition", "")).strip()

    if not tracking or tracking == "nan":
        return []

    rows = []
    for rec in parse_image_editing_tracking_edited_pairs(tracking):
        slot_xml = _slot_xml_for_scene_and_image_index(plan, rec["scene_id"], rec["image_k"])
        edits_inner = _extract_edits_inner_from_slot_xml(slot_xml) if slot_xml else ""
        edit_plan_text = _format_edit_plan_for_sheet(edits_inner)

        vo = _vo_for_asset_url(fgd, rec["original_url"]) if fgd and fgd != "nan" else ""
        orig_cell = _cell_value_for_url(rec["original_url"])
        edit_cell = _cell_value_for_url(rec["edited_url"])
        rows.append({
            "cells": [
                _cell_text_literal(chunk),
                _cell_text_literal(vo),
                orig_cell,
                _cell_text_literal(edit_plan_text),
                edit_cell,
            ],
            "original_url": rec["original_url"],
            "edited_url": rec["edited_url"],
        })
    return rows


def _clear_values_and_notes(ws, max_row=4000, max_col=5):
    """
    Clear cell values and notes for the results tab.

    :param ws: gspread Worksheet.
    :param max_row: Max row index (exclusive, 0-based end).
    :param max_col: Max column index exclusive.
    :return: None
    """
    ws.clear()
    try:
        ws.spreadsheet.batch_update({
            "requests": [{
                "repeatCell": {
                    "range": {
                        "sheetId": ws.id,
                        "startRowIndex": 0,
                        "endRowIndex": max_row,
                        "startColumnIndex": 0,
                        "endColumnIndex": max_col,
                    },
                    "cell": {"note": ""},
                    "fields": "note",
                },
            }],
        })
    except Exception as e:
        print(f"Image edit results: note clear batch_update failed (values still cleared): {e}")


def _set_row_heights_for_images(ws, data_row_count, pixel_size=180):
    """
    Stretch data rows so in-cell IMAGE previews are visible.

    :param ws: Worksheet.
    :param data_row_count: Number of data rows excluding header.
    :param pixel_size: Row height in pixels.
    :return: None
    """
    if data_row_count <= 0:
        return
    try:
        ws.spreadsheet.batch_update({
            "requests": [{
                "updateDimensionProperties": {
                    "range": {
                        "sheetId": ws.id,
                        "dimension": "ROWS",
                        "startIndex": 1,
                        "endIndex": 1 + data_row_count,
                    },
                    "properties": {"pixelSize": pixel_size},
                    "fields": "pixelSize",
                },
            }],
        })
    except Exception as e:
        print(f"Image edit results: row height update skipped: {e}")


def _apply_url_notes_for_image_columns(ws, url_pairs):
    """
    Set cell notes on Original (C) and Edited (E) columns with Drive view URLs.

    :param ws: Results worksheet.
    :param url_pairs: List of dicts with original_url and edited_url keys (one per data row).
    :return: None
    """
    if not url_pairs:
        return
    requests = []
    for r0, pair in enumerate(url_pairs, start=1):
        ou = (pair.get("original_url") or "").strip()
        eu = (pair.get("edited_url") or "").strip()
        if ou:
            note_o = f"Image URL:\n{_drive_url_to_view_url(ou)}"
            requests.append({
                "repeatCell": {
                    "range": {
                        "sheetId": ws.id,
                        "startRowIndex": r0,
                        "endRowIndex": r0 + 1,
                        "startColumnIndex": 2,
                        "endColumnIndex": 3,
                    },
                    "cell": {"note": note_o},
                    "fields": "note",
                },
            })
        if eu:
            note_e = f"Image URL:\n{_drive_url_to_view_url(eu)}"
            requests.append({
                "repeatCell": {
                    "range": {
                        "sheetId": ws.id,
                        "startRowIndex": r0,
                        "endRowIndex": r0 + 1,
                        "startColumnIndex": 4,
                        "endColumnIndex": 5,
                    },
                    "cell": {"note": note_e},
                    "fields": "note",
                },
            })
    chunk = 400
    for i in range(0, len(requests), chunk):
        ws.spreadsheet.batch_update({"requests": requests[i:i + chunk]})


@traceable(
    metadata={
        "agent_name": "graphics_definition_v2",
        "step_name": "Image Edit Results Sheet",
        "function_name": "run_populate_image_edit_results_sheet",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous"),
    }
)
def run_populate_image_edit_results_sheet(sheet, save_interval=10):
    """
    Create or refresh the Image Edit results tab from Slide Chunks columns.

    One row per edited image (Original + Edited URLs present in image_editing_tracking).

    :param sheet: gspread spreadsheet.
    :param save_interval: Unused; reserved for parity with other runners.
    :return: None
    """
    _ = save_interval
    worksheet_name = "Slide Chunks"
    _, df = get_sheet_data_and_df(sheet, worksheet_name)

    results_ws, _ = create_or_read_worksheet(
        sheet,
        IMAGE_EDIT_RESULTS_WS,
        rows=3000,
        cols=len(HEADERS),
    )
    _clear_values_and_notes(results_ws)

    detail_rows = []
    progress = SmartProgressBar(
        total_tasks=max(1, len(df)),
        description="Image edit results sheet",
        save_interval=5,
    )

    for _, row in df.iterrows():
        try:
            detail_rows.extend(_build_rows_from_slide_chunk_row(row))
        except Exception as e:
            print(f"Image edit results: row build error: {e}")
            traceback.print_exc()
        progress.update()

    if not detail_rows:
        results_ws.update("A1:E1", [HEADERS], value_input_option="USER_ENTERED")
        format_worksheet(results_ws)
        print("Image edit results: no edited-image rows found; header only.")
        return

    output_rows = [HEADERS] + [d["cells"] for d in detail_rows]
    end = len(output_rows)
    results_ws.update(
        f"A1:E{end}",
        output_rows,
        value_input_option="USER_ENTERED",
    )
    _apply_url_notes_for_image_columns(results_ws, detail_rows)
    format_worksheet(results_ws)
    _set_row_heights_for_images(results_ws, end - 1)
    print(f"Image edit results: wrote {end - 1} data row(s) to '{IMAGE_EDIT_RESULTS_WS}'.")


def delete_image_edit_results_sheet_data(sheet):
    """
    Remove the Image Edit results tab if present, then add an empty tab with headers only.

    This clears values and notes (fresh worksheet).

    :param sheet: gspread spreadsheet.
    :return: None
    """
    names = get_worksheet_names(sheet)
    if IMAGE_EDIT_RESULTS_WS not in names:
        ws = sheet.add_worksheet(IMAGE_EDIT_RESULTS_WS, rows=2000, cols=len(HEADERS))
    else:
        delete_worksheet(sheet, IMAGE_EDIT_RESULTS_WS)
        ws = sheet.add_worksheet(IMAGE_EDIT_RESULTS_WS, rows=2000, cols=len(HEADERS))
    ws.update("A1:E1", [HEADERS], value_input_option="USER_ENTERED")
    format_worksheet(ws)
    print(f"Image edit results: reset tab '{IMAGE_EDIT_RESULTS_WS}' (empty data, headers kept).")
