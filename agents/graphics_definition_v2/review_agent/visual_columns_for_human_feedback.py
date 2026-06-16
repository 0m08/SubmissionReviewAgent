from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from gspread.utils import rowcol_to_a1

from services.sheets_service import get_sheet_data_and_df, save_to_sheet, format_worksheet
from agents.graphics_definition_v2.review_agent.review_and_revise import _safe_str
from agents.graphics_definition_v2.review_agent.human_feedback_based_review_and_revise import (
    parse_human_feedback_for_row,
    HUMAN_FEEDBACK_REVISION_TRACKING_COLUMN,
    _normalize_vo_for_match,
    ORIG_PREFIX,
    MANUAL_PREFIX,
    AFTER_REV_PREFIX,
    AFTER_REGEN1_PREFIX,
    AFTER_REGEN2_PREFIX,
    NO_REPLACEMENT,
    _parse_tracking_column,
)
from agents.graphics_definition_v2.review_agent.review_and_revise import (
    build_segment_visual_map,
)
from agents.graphics_definition_v2.aggregation_agent.aggregation_agent import (
    parse_video_url_timestamps,
    extract_video_id_from_url,
)

# Prefix for columns we own (delete on re-run)
HF_VIS_COLUMN_PREFIX = "hf_vis_"

# Pixel size for Original/Revised visual columns and for data rows that contain them
HF_VIS_DIMENSION_PX = 200

# (Prefixes imported from human_feedback_based_review_and_revise)


def _is_youtube_or_embed_url(url: str) -> bool:
    if not url or url == NO_REPLACEMENT or url.startswith("("):
        return False
    u = url.lower()
    return "youtube.com" in u or "youtu.be" in u


def _seconds_to_friendly_time(total_seconds: int) -> str:
    """e.g. 64 -> '1m 4s', 68 -> '1m 8s' (readable m/s)."""
    if total_seconds < 0:
        return "0s"
    m, s = divmod(int(total_seconds), 60)
    if m == 0:
        return f"{s}s"
    if s == 0:
        return f"{m}m"
    return f"{m}m {s}s"


def _format_youtube_embed_for_visual_column(url: str) -> Optional[str]:
    """
    Turn embed URL with start/end into watch URL + human-readable line for sheet cells.

    Input:  https://www.youtube.com/embed/D-MlNYq3lz4?start=64&end=68
    Output:
      https://www.youtube.com/watch?v=D-MlNYq3lz4&t=64s

      (play the clip from 1m 4s to 1m8s)
    Returns None if URL should not be transformed (no embed start, etc.).
    """
    if not url or not url.strip():
        return None
    url = url.strip()
    if "youtube.com/embed" not in url.lower():
        return None
    clip_url, start_seconds, end_seconds = parse_video_url_timestamps(url)
    if start_seconds is None:
        return None
    video_id = extract_video_id_from_url(url)
    if not video_id:
        return None
    watch = f"https://www.youtube.com/watch?v={video_id}&t={start_seconds}s"
    start_label = _seconds_to_friendly_time(start_seconds)
    if end_seconds is not None:
        end_label = _seconds_to_friendly_time(end_seconds)
        line2 = f"(play the clip from {start_label} to {end_label})"
    else:
        line2 = f"(play from {start_label})"
    return f"{watch}\n\n{line2}"


def _drive_file_id_from_url(url: str) -> Optional[str]:
    """Extract Google Drive file id from /file/d/ID/ or open?id=ID or uc?id=."""
    if not url:
        return None
    m = re.search(r"/file/d/([a-zA-Z0-9_-]+)", url)
    if m:
        return m.group(1)
    m = re.search(r"[?&]id=([a-zA-Z0-9_-]+)", url)
    if m:
        return m.group(1)
    return None


def _drive_url_to_view_url(url: str) -> str:
    """
    URL that opens the file in Drive/browser (full image/page), not thumbnail.
    """
    if not url or "drive.google.com" not in url.lower():
        return url
    file_id = _drive_file_id_from_url(url)
    if not file_id:
        return url
    return f"https://drive.google.com/file/d/{file_id}/view"


def _drive_url_to_image_friendly_url(url: str) -> str:
    """
    Drive /view and /open links are HTML pages; =IMAGE() needs a direct image response.
    Use thumbnail endpoint (often works for shared files) or uc?export=view.
    Fallback: return original (may still fail if file not public).
    """
    if not url or "drive.google.com" not in url.lower():
        return url
    file_id = _drive_file_id_from_url(url)
    if not file_id:
        return url
    # Thumbnail API — good for in-cell preview when file is accessible
    return f"https://drive.google.com/thumbnail?id={file_id}&sz=w400"


def _escape_formula_string(s: str) -> str:
    """Escape double quotes for use inside Sheets formula string literal."""
    if not s:
        return ""
    return s.replace("\\", "\\\\").replace('"', '""')


def _is_probably_direct_image_url(url: str) -> bool:
    """True if URL looks like a direct image (not Drive page, not YouTube)."""
    if not url or _is_youtube_or_embed_url(url):
        return False
    u = url.lower()
    if "drive.google.com" in u:
        return False
    # Common image extensions or image CDN paths
    if re.search(r"\.(png|jpe?g|gif|webp|bmp)(\?|$)", u):
        return True
    return False


def _cell_value_for_url(url: str) -> str:
    """
    Return cell content:
    - YouTube: watch + readable text, or HYPERLINK
    - Drive / direct image: =IMAGE(...) so the sheet shows a preview. Full-size / open in
      browser is via cell note (Open: …) — Sheets API cannot make IMAGE cells clickable.
    If url empty or not found, return empty string.
    """
    if not url or url.strip() == "" or url == NO_REPLACEMENT or "(not found)" in url:
        return ""
    url = url.strip()
    if _is_youtube_or_embed_url(url):
        # Timestamped embed: paste watch URL + readable range (multi-line; not a formula)
        formatted = _format_youtube_embed_for_visual_column(url)
        if formatted is not None:
            return formatted
        esc = _escape_formula_string(url)
        return f'=HYPERLINK("{esc}","Open YouTube / clip")'

    # Drive file links: thumbnail URL so =IMAGE can render in cell; note has /view link to open full
    if "drive.google.com" in url.lower() and "/file/d/" in url.lower():
        thumb = _drive_url_to_image_friendly_url(url)
        esc = _escape_formula_string(thumb)
        return f'=IMAGE("{esc}")'

    # Direct image URL (png/jpg/…): IMAGE loads it in cell; note has same URL to open in tab
    if url.lower().startswith("http") and _is_probably_direct_image_url(url):
        esc = _escape_formula_string(url)
        return f'=IMAGE("{esc}")'

    # Other Drive URLs without /file/d/: still try thumbnail helper
    if "drive.google.com" in url.lower():
        thumb = _drive_url_to_image_friendly_url(url)
        esc = _escape_formula_string(thumb)
        return f'=IMAGE("{esc}")'
    # Any other http(s): attempt IMAGE (works for many direct image URLs)
    esc = _escape_formula_string(url)
    return f'=IMAGE("{esc}")'


def _final_url_from_tracking(data: Dict[str, Optional[str]], original_fallback: Optional[str]) -> Optional[str]:
    """Last non-null stage wins; else original."""
    for key in ("after_regen_2", "after_regen_1", "after_revision", "manually_selected"):
        v = data.get(key)
        if v:
            return v
    return original_fallback


def _visual_id_to_vo_and_feedback(
    visual_id: str,
    segments_map: Dict,
    feedback_by_segment: Dict[int, List[Tuple[str, str]]],
) -> Tuple[str, str]:
    """Find voiceover_part and feedback text for this visual_id."""
    vo_part = ""
    feedback_text = ""
    for seg_num, segment in segments_map.items():
        for step in segment.get("visual_steps", []):
            if step.get("visual_id") == visual_id:
                vo_part = _safe_str(step.get("voiceover_part", ""))
                # Match feedback block by vo_part
                pairs = feedback_by_segment.get(seg_num, [])
                norm = _normalize_vo_for_match(vo_part)
                for vp, fb in pairs:
                    if _normalize_vo_for_match(vp) == norm or vp.strip() == vo_part.strip():
                        feedback_text = fb
                        break
                if not feedback_text and pairs:
                    feedback_text = pairs[0][1]
                return vo_part, feedback_text
    return vo_part, feedback_text


def _build_note_text(vo_part: str, feedback_text: str) -> str:

    if vo_part and feedback_text:
        return f"When VO: {vo_part}\n\nHuman Feedback: {feedback_text}"
    if vo_part:
        return f"When VO: {vo_part}"
    if feedback_text:
        return f"Human Feedback: {feedback_text}"
    return ""


def _open_url_for_note(url: str) -> Optional[str]:
    """
    URL to show in cell note so user can copy/open in browser.
    Drive file links -> /view; YouTube embed with start -> watch&t=; else http as-is.
    """
    if not url or not url.strip() or url == NO_REPLACEMENT or "(not found)" in url:
        return None
    url = url.strip()
    if not url.lower().startswith("http"):
        return None
    if _is_youtube_or_embed_url(url):
        formatted = _format_youtube_embed_for_visual_column(url)
        if formatted:
            # First line is the watch URL
            first_line = formatted.split("\n", 1)[0].strip()
            if first_line.startswith("http"):
                return first_line
        return url
    if "drive.google.com" in url.lower() and "/file/d/" in url.lower():
        return _drive_url_to_view_url(url)
    return url


def _append_open_url_to_note(note: str, url: str) -> str:
    """Append copyable open URL to note text."""
    open_url = _open_url_for_note(url)
    if not open_url:
        return note
    block = f"Image Link(copy or paste in browser):\n{open_url}"
    if note:
        return f"{note}\n\n{block}"
    return block


def _collect_pairs_for_row(row: pd.Series) -> List[Dict[str, Any]]:
    """
    Build ordered list of {visual_id, original_url, final_url, vo_part, feedback_text}
    for one dataframe row.
    """
    human_feedback_raw = _safe_str(row.get("human_feedback", ""))
    tracking_text = _safe_str(row.get(HUMAN_FEEDBACK_REVISION_TRACKING_COLUMN, ""))
    voiceover_text = _safe_str(row.get("voiceover_segment", ""))
    final_def = _safe_str(row.get("final_graphics_definition", ""))
    slide_chunk = _safe_str(row.get("Slide Chunk", ""))
    vas = str(row.get("Visual Assignment Strategy", "Flexible, let the agent decide")).strip()
    if not vas or vas == "nan":
        vas = "Flexible, let the agent decide"

    if not human_feedback_raw.strip() or human_feedback_raw == "nan":
        return []
    if not tracking_text.strip() or tracking_text == "nan":
        return []

    tracking_parsed = _parse_tracking_column(tracking_text)
    if not tracking_parsed:
        return []

    segments_map = build_segment_visual_map(voiceover_text, final_def, vas, slide_chunk)
    feedback_by_segment = parse_human_feedback_for_row(
        human_feedback_raw, voiceover_text, final_def, vas, slide_chunk
    )

    def sort_key(vid: str):
        m = re.match(r"S(\d+)V(\d+)", vid)
        if not m:
            return (0, 0)
        return (int(m.group(1)), int(m.group(2)))

    pairs: List[Dict[str, Any]] = []
    for visual_id in sorted(tracking_parsed.keys(), key=sort_key):
        data = tracking_parsed[visual_id]
        original = data.get("original")
        final_url = _final_url_from_tracking(data, original)
        vo_part, fb_text = _visual_id_to_vo_and_feedback(visual_id, segments_map, feedback_by_segment)
        pairs.append({
            "visual_id": visual_id,
            "original_url": original or "",
            "final_url": final_url or "",
            "vo_part": vo_part,
            "feedback_text": fb_text,
        })
    return pairs


def _is_orig_rev_visual_column(name: str) -> bool:
    """True if column is one of our Original Visual N / Revised Visual N headers."""
    s = str(name).strip()
    return bool(re.fullmatch(r"Original Visual \d+", s) or re.fullmatch(r"Revised Visual \d+", s))


def _drop_hf_vis_columns(df: pd.DataFrame) -> pd.DataFrame:
    to_drop = [
        c
        for c in df.columns
        if str(c).startswith(HF_VIS_COLUMN_PREFIX) or _is_orig_rev_visual_column(c)
    ]
    if to_drop:
        df = df.drop(columns=to_drop)
    return df


def _column_names_for_pair_index(i: int) -> Tuple[str, str, str]:
    """
    Display headers: Original Visual N, Revised Visual N.
    Gap column uses internal name only; row-1 header is cleared after save so it stays blank.
    """
    return (
        f"Original Visual {i}",
        f"Revised Visual {i}",
        f"{HF_VIS_COLUMN_PREFIX}gap_{i}",
    )


def run_populate_human_feedback_visual_columns(sheet, worksheet_name: str = "Slide Chunks") -> None:
    """
    Add dynamic columns after human_feedback_revision_tracking and fill formulas + notes.
    """
    ws, df = get_sheet_data_and_df(sheet, worksheet_name)

    if HUMAN_FEEDBACK_REVISION_TRACKING_COLUMN not in df.columns:
        print(f"Column {HUMAN_FEEDBACK_REVISION_TRACKING_COLUMN} not found; nothing to do.")
        return

    # Drop previous run columns
    df = _drop_hf_vis_columns(df)

    # Compute max pairs and per-row pair lists
    row_pairs: Dict[int, List[Dict[str, Any]]] = {}
    max_pairs = 0
    for idx in df.index:
        row = df.loc[idx]
        pairs = _collect_pairs_for_row(row)
        if pairs:
            row_pairs[idx] = pairs
            max_pairs = max(max_pairs, len(pairs))

    if max_pairs == 0:
        print("No rows with human feedback + revision tracking to visualize.")
        save_to_sheet(ws, df)
        format_worksheet(ws)
        return

    # Insert new columns after tracking column
    insert_at = df.columns.get_loc(HUMAN_FEEDBACK_REVISION_TRACKING_COLUMN) + 1
    for i in range(1, max_pairs + 1):
        orig_col, final_col, gap_col = _column_names_for_pair_index(i)
        df.insert(insert_at, orig_col, "")
        insert_at += 1
        df.insert(insert_at, final_col, "")
        insert_at += 1
        if i < max_pairs:
            df.insert(insert_at, gap_col, "")
            insert_at += 1

    # Fill formulas per row
    for idx, pairs in row_pairs.items():
        for i, p in enumerate(pairs, start=1):
            orig_col, final_col, _ = _column_names_for_pair_index(i)
            df.at[idx, orig_col] = _cell_value_for_url(p["original_url"])
            df.at[idx, final_col] = _cell_value_for_url(p["final_url"])

    # Save sheet — save_to_sheet astypes str which breaks formulas; write formulas with USER_ENTERED after
    save_to_sheet(ws, df)
    format_worksheet(ws)

    # Re-write hf_vis cells as formulas so Sheets parses =IMAGE / =HYPERLINK (not plain text)
    _write_hf_vis_formulas_user_entered(ws, df, row_pairs, max_pairs)

    # Gap columns: no header text in sheet (internal name is hf_vis_gap_* only in df until cleared)
    _clear_gap_column_headers(ws, df, max_pairs)

    # Notes via API (batch_update)
    _apply_notes_for_hf_vis_columns(ws, df, row_pairs, max_pairs)

    # Column width + row height for visual cells (after format_worksheet's default row height)
    _set_hf_vis_column_row_dimensions(ws, df, row_pairs, max_pairs, HF_VIS_DIMENSION_PX)

    print(f"Human feedback visual columns added (max {max_pairs} pair(s) per row).")


def _write_hf_vis_formulas_user_entered(
    ws, df: pd.DataFrame, row_pairs: Dict[int, List[Dict[str, Any]]], max_pairs: int
) -> None:
    """
    save_to_sheet uses astype(str) which makes Sheets treat =IMAGE as text.
    Overwrite hf_vis cells with valueInputOption USER_ENTERED so formulas execute.
    """
    headers = list(df.columns)
    num_rows = len(df)

    def col_letter_1based(col_idx: int) -> str:
        return rowcol_to_a1(1, col_idx + 1).rstrip("1").rstrip("$") if col_idx >= 0 else "A"

    for i in range(1, max_pairs + 1):
        orig_col, final_col, _ = _column_names_for_pair_index(i)
        for col_name in (orig_col, final_col):
            if col_name not in headers:
                continue
            cidx = headers.index(col_name)
            col_letter = col_letter_1based(cidx)
            # Build one column: row 1 is header (already str), rows 2.. are data
            column_values = [[df.columns[cidx]]]  # header row as single cell... no, update range should be data only
            # Update data rows only: from row 2 to num_rows+1
            data_rows: List[List[str]] = []
            for pos in range(num_rows):
                idx = df.index[pos]
                pairs = row_pairs.get(idx, [])
                formula = ""
                if i <= len(pairs):
                    p = pairs[i - 1]
                    if col_name == orig_col:
                        formula = _cell_value_for_url(p.get("original_url", ""))
                    else:
                        formula = _cell_value_for_url(p.get("final_url", ""))
                data_rows.append([formula] if formula else [""])
            if not data_rows:
                continue
            range_a1 = f"{col_letter}2:{col_letter}{num_rows + 1}"
            try:
                ws.update(range_a1, data_rows, value_input_option="USER_ENTERED")
            except TypeError:
                # Older gspread without value_input_option — batch_update fallback
                _batch_update_formulas(ws, col_letter, 2, data_rows)
            except Exception as e:
                print(f"  WARNING: Could not write formulas for {col_name}: {e}")


def _set_hf_vis_column_row_dimensions(
    ws,
    df: pd.DataFrame,
    row_pairs: Dict[int, List[Dict[str, Any]]],
    max_pairs: int,
    pixel_size: int = HF_VIS_DIMENSION_PX,
) -> None:
    """
    Set column width and row height to pixel_size for Original/Revised visual columns
    and for sheet rows that have at least one pair (so IMAGE/HYPERLINK/multiline fits).
    """
    headers = list(df.columns)
    sheet_id = ws.id
    requests: List[dict] = []

    # Column width for each Original Visual N / Revised Visual N (gap columns unchanged)
    for i in range(1, max_pairs + 1):
        orig_col, final_col, _ = _column_names_for_pair_index(i)
        for col_name in (orig_col, final_col):
            if col_name not in headers:
                continue
            cidx = headers.index(col_name)
            requests.append({
                "updateDimensionProperties": {
                    "range": {
                        "sheetId": sheet_id,
                        "dimension": "COLUMNS",
                        "startIndex": cidx,
                        "endIndex": cidx + 1,
                    },
                    "properties": {"pixelSize": pixel_size},
                    "fields": "pixelSize",
                }
            })

    # Row height only for rows that have pairs (row 1 stays default; data rows are 2-based)
    num_rows = len(df)
    for pos in range(num_rows):
        idx = df.index[pos]
        if idx not in row_pairs or not row_pairs[idx]:
            continue
        # Sheet row pos+2 -> 0-based row index pos+1
        row_index_0based = pos + 1
        requests.append({
            "updateDimensionProperties": {
                "range": {
                    "sheetId": sheet_id,
                    "dimension": "ROWS",
                    "startIndex": row_index_0based,
                    "endIndex": row_index_0based + 1,
                },
                "properties": {"pixelSize": pixel_size},
                "fields": "pixelSize",
            }
        })

    if requests:
        try:
            # Sheets API caps requests per batch (~100); chunk to stay safe
            chunk_size = 80
            for start in range(0, len(requests), chunk_size):
                ws.spreadsheet.batch_update({"requests": requests[start : start + chunk_size]})
            print(f"  hf_vis columns/rows set to {pixel_size}px where applicable.")
        except Exception as e:
            print(f"  WARNING: Could not set hf_vis dimensions: {e}")


def _clear_gap_column_headers(ws, df: pd.DataFrame, max_pairs: int) -> None:
    """Set row 1 for gap columns to empty so no column name shows."""
    headers = list(df.columns)
    sheet_id = ws.id
    requests = []
    for i in range(1, max_pairs):
        _, _, gap_col = _column_names_for_pair_index(i)
        if gap_col not in headers:
            continue
        cidx = headers.index(gap_col)
        requests.append({
            "updateCells": {
                "range": {
                    "sheetId": sheet_id,
                    "startRowIndex": 0,
                    "endRowIndex": 1,
                    "startColumnIndex": cidx,
                    "endColumnIndex": cidx + 1,
                },
                "rows": [{"values": [{"userEnteredValue": {"stringValue": ""}}]}],
                "fields": "userEnteredValue",
            }
        })
    if requests:
        ws.spreadsheet.batch_update({"requests": requests})


def _batch_update_formulas(ws, col_letter: str, start_row: int, data_rows: List[List[str]]) -> None:
    """Fallback: updateCells with formulaValue per cell."""
    from gspread.utils import a1_to_rowcol

    sheet_id = ws.id
    requests = []

    for r, row in enumerate(data_rows):
        if not row or not row[0] or not str(row[0]).startswith("="):
            continue
        formula = row[0]
        row_idx = start_row + r - 1  # 0-based
        # Resolve column index from letter (supports AA etc.)
        _, col_idx = a1_to_rowcol(f"{col_letter}{start_row + r}")
        requests.append({
            "updateCells": {
                "range": {
                    "sheetId": sheet_id,
                    "startRowIndex": row_idx,
                    "endRowIndex": row_idx + 1,
                    "startColumnIndex": col_idx - 1,
                    "endColumnIndex": col_idx,
                },
                "rows": [{"values": [{"userEnteredValue": {"formulaValue": formula}}]}],
                "fields": "userEnteredValue",
            }
        })
    if requests:
        for i in range(0, len(requests), 50):
            ws.spreadsheet.batch_update({"requests": requests[i : i + 50]})


def _apply_notes_for_hf_vis_columns(ws, df: pd.DataFrame, row_pairs: Dict[int, List[Dict[str, Any]]], max_pairs: int) -> None:
    """Set cell notes for original/final columns using Sheets API repeatCell."""
    try:
        headers = list(df.columns)
    except Exception:
        return

    def col_index(name: str) -> Optional[int]:
        if name not in headers:
            return None
        return headers.index(name)

    requests = []
    sheet_id = ws.id

    for df_idx, pairs in row_pairs.items():
        # DataFrame row -> sheet row: header row 1; data starts row 2
        try:
            pos = df.index.get_loc(df_idx)
            if isinstance(pos, slice):
                pos = pos.start
        except Exception:
            pos = int(df_idx) if str(df_idx).isdigit() else 0
        start_row_index = int(pos) + 1  # 0-based; row 2 in sheet -> index 1

        for i, p in enumerate(pairs, start=1):
            base_note = _build_note_text(p.get("vo_part", ""), p.get("feedback_text", ""))
            orig_col, final_col, _ = _column_names_for_pair_index(i)
            # Original column note: VO/feedback + open URL for original asset
            note_orig = _append_open_url_to_note(base_note, p.get("original_url", ""))
            # Revised column note: same VO/feedback + open URL for final asset
            note_final = _append_open_url_to_note(base_note, p.get("final_url", ""))

            for col_name, note in ((orig_col, note_orig), (final_col, note_final)):
                if not note:
                    continue
                cidx = col_index(col_name)
                if cidx is None:
                    continue
                requests.append({
                    "repeatCell": {
                        "range": {
                            "sheetId": sheet_id,
                            "startRowIndex": start_row_index,
                            "endRowIndex": start_row_index + 1,
                            "startColumnIndex": cidx,
                            "endColumnIndex": cidx + 1,
                        },
                        "cell": {"note": note},
                        "fields": "note",
                    }
                })

    if not requests:
        return

    # Batch in chunks to avoid request size limits
    chunk = 100
    for i in range(0, len(requests), chunk):
        ws.spreadsheet.batch_update({"requests": requests[i : i + chunk]})
    print(f"Applied notes to human feedback visual columns ({len(requests)} cell(s)).")


def delete_human_feedback_visual_columns(sheet, worksheet_name: str = "Slide Chunks") -> bool:
    """
    Remove Original Visual N, Revised Visual N, gap (hf_vis_gap_* or blank) columns and notes.
    deleteDimension removes columns from the sheet so notes are removed with them.
    """
    try:
        ws, df = get_sheet_data_and_df(sheet, worksheet_name)
        headers = ws.row_values(1)
        indices = []
        for i, h in enumerate(headers):
            s = str(h).strip()
            if s.startswith(HF_VIS_COLUMN_PREFIX):
                indices.append(i)
            elif _is_orig_rev_visual_column(s):
                indices.append(i)
        # Blank header columns that sit between our blocks are gap columns — if still present
        # after clear, they appear as ""; include consecutive empty headers between named cols
        # by scanning: any column after Revised Visual before next Original is gap
        indices = sorted(set(indices))
        # Gap columns have blank header after _clear_gap_column_headers — delete blank header
        # columns that sit right after a Revised Visual N column
        for i in range(len(headers) - 1):
            if i in indices or (i + 1) in indices:
                continue
            if re.match(r"Revised Visual \d+", str(headers[i]).strip()) and str(headers[i + 1]).strip() == "":
                indices.append(i + 1)
        indices = sorted(set(indices))
        indices.sort(reverse=True)
        if indices:
            requests = []
            for i in indices:
                requests.append({
                    "deleteDimension": {
                        "range": {
                            "sheetId": ws.id,
                            "dimension": "COLUMNS",
                            "startIndex": i,
                            "endIndex": i + 1,
                        }
                    }
                })
            ws.spreadsheet.batch_update({"requests": requests})
            print(f"Deleted {len(indices)} hf_vis column(s) from sheet (notes removed with columns).")

        df = _drop_hf_vis_columns(df)
        save_to_sheet(ws, df)
        format_worksheet(ws)
        print("Human feedback visual columns deleted; sheet synced.")
        return True
    except Exception as e:
        print(f"Error deleting human feedback visual columns: {e}")
        return False
