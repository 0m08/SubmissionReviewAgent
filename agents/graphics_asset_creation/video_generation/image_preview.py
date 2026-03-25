"""
image_preview.py
~~~~~~~~~~~~~~~~
Builds an 'Image Preview' Google Sheet tab that shows — per voiceover segment —
the INPUT image (external / Drive-snapshot link from final_graphics_definition)
and the OUTPUT image (Drive link written back by the voiceover-reviewer pipeline
in final_graphics), side-by-side as =IMAGE() formulas.

Public API
----------
    build_image_preview_sheet(sheet_link, gc, source_worksheet_name, preview_sheet_name)
        → (rows_written: int, preview_tab_url: str)

Design
------
* One output row per voiceover subsegment (not per sheet row).
* Rows are parsed in parallel across all source rows (ThreadPoolExecutor).
* A single batched gspread write is used — no per-row round-trips.
* Drive links → converted to /thumbnail?id=... so =IMAGE() renders without sign-in.
* YouTube links  → kept as plain text (cannot render in =IMAGE()).
* External links → wrapped in =IMAGE(url, 1) (fit-to-cell).
"""

from __future__ import annotations

import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import gspread

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so gac_utils can be imported
# ---------------------------------------------------------------------------
_THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_THIS_DIR, "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from agents.graphics_asset_creation.gac_utils import parse_subsegments  # noqa: E402

# ---------------------------------------------------------------------------
# Internal helpers — URL classification
# ---------------------------------------------------------------------------

def _is_youtube(url: str) -> bool:
    return "youtube.com" in url or "youtu.be" in url


def _is_drive(url: str) -> bool:
    return "drive.google.com" in url


def _extract_drive_file_id(url: str) -> Optional[str]:
    """Extract the Drive file ID from any standard Drive URL format."""
    patterns = [
        r"/file/d/([a-zA-Z0-9_-]+)",
        r"id=([a-zA-Z0-9_-]+)",
        r"/d/([a-zA-Z0-9_-]+)",
        r"drive\.google\.com/open\?id=([a-zA-Z0-9_-]+)",
    ]
    for pat in patterns:
        m = re.search(pat, url, re.IGNORECASE)
        if m:
            return m.group(1)
    return None


def _drive_thumbnail_url(url: str) -> str:
    """Convert a Drive share/view link to a thumbnail URL for =IMAGE()."""
    file_id = _extract_drive_file_id(url)
    if file_id:
        return f"https://drive.google.com/thumbnail?id={file_id}&sz=w400"
    return url


def _make_image_formula(url: str) -> str:
    """
    Return a Google Sheets =IMAGE() formula string for *url*.

    Rules
    -----
    - Empty             → ""
    - YouTube           → raw URL as text (cannot display via IMAGE)
    - Drive             → =IMAGE(thumbnail_url, 1)
    - All other URLs    → =IMAGE(url, 1)
    """
    url = (url or "").strip()
    if not url:
        return ""
    if _is_youtube(url):
        return url           # plain text fallback — Sheets can't render YouTube
    if _is_drive(url):
        return f'=IMAGE("{_drive_thumbnail_url(url)}",1)'
    return f'=IMAGE("{url}",1)'


# ---------------------------------------------------------------------------
# Per-segment parsing helpers
# ---------------------------------------------------------------------------

def _parse_input_entries(fgd_text: str) -> List[Dict[str, str]]:
    """
    Parse ``final_graphics_definition`` → list of per-voiceover dicts with keys:
        voiceover_focus, visual_instruction, input_url

    Input images are:
      (a) External web image links (non-Drive, non-YouTube)
      (b) Drive links that carry the ``(snapshot)`` tag
    """
    subsegments = parse_subsegments(fgd_text)   # exceptions propagate up intentionally

    results: List[Dict[str, str]] = []
    for seg in subsegments:
        ref_link    = (seg.get("reference_link") or "").strip()
        is_ai_generated = "(ai generated)" in ref_link.lower()
        has_snapshot = "(snapshot)" in ref_link.lower()
        clean_link  = re.sub(r"\s*\(snapshot\)\s*", "", ref_link, flags=re.IGNORECASE).strip()
        clean_link  = re.sub(r"\s*\(ai generated\)\s*", "", clean_link, flags=re.IGNORECASE).strip()

        # Determine validity as an INPUT image
        is_valid = False
        if (not is_ai_generated) and clean_link.startswith(("http://", "https://")):
            if has_snapshot and _is_drive(clean_link):
                is_valid = True          # (b) drive + snapshot tag
            elif not _is_drive(clean_link) and not _is_youtube(clean_link):
                is_valid = True          # (a) external non-drive, non-yt

        results.append({
            "voiceover_focus":    seg.get("voiceover_focus", ""),
            "visual_instruction": seg.get("visual_instruction", ""),
            "input_url":          clean_link if is_valid else "",
            "raw_url":            ref_link,
        })
    return results


def _parse_output_urls(fg_text: str) -> List[Dict[str, str]]:
    """
    Parse ``final_graphics`` → list of output URLs, one per subsegment.

    Output images are Drive links (the AI-generated results).
    Non-Drive entries (YouTube, external, empty) map to ``""``.
    """
    subsegments = parse_subsegments(fg_text)   # exceptions propagate up intentionally

    urls: List[Dict[str, str]] = []
    for seg in subsegments:
        ref_link   = (seg.get("reference_link") or "").strip()
        is_ai_generated = "(ai generated)" in ref_link.lower()
        clean_link = re.sub(r"\s*\(snapshot\)\s*", "", ref_link, flags=re.IGNORECASE).strip()
        clean_link = re.sub(r"\s*\(ai generated\)\s*", "", clean_link, flags=re.IGNORECASE).strip()
        if clean_link.startswith(("http://", "https://")) and _is_drive(clean_link):
            urls.append({
                "output_url": clean_link,
                "is_ai_generated": "yes" if is_ai_generated else "",
            })
        else:
            urls.append({
                "output_url": "",
                "is_ai_generated": "yes" if is_ai_generated else "",
            })   # YouTube / external / missing → no output image
    return urls


# ---------------------------------------------------------------------------
# Column-name resolver (case-insensitive + strip)
# ---------------------------------------------------------------------------

def _find_col(record: dict, *candidates: str) -> str:
    """
    Return the value for the first candidate key found in *record*,
    using case-insensitive, whitespace-stripped comparison.
    Returns "" if none of the candidates match.
    """
    # Build a normalised lookup once per call
    normalised = {k.strip().lower(): v for k, v in record.items()}
    for name in candidates:
        val = normalised.get(name.strip().lower())
        if val is not None:
            return str(val) if val else ""
    return ""


# ---------------------------------------------------------------------------
# Row-level worker (parallelised)
# ---------------------------------------------------------------------------

def _process_record(record: dict, row_idx: int = 0) -> List[List[str]]:
    """
    Process one source sheet record → list of output rows (one per voiceover).

    Uses case-insensitive column name lookup so header naming differences
    (capitalisation, extra spaces) don't silently produce empty results.

    Returns a (possibly empty) list of 8-element lists:
        [topic, subtopic, slide_chunk_title, slide_chunk,
         voiceover, visual_instruction, input_formula, output_formula]
    """
    topic             = _find_col(record, "Topic")
    subtopic          = _find_col(record, "Subtopic")
    slide_chunk_title = _find_col(record, "Slide Chunk Title")
    slide_chunk       = _find_col(record, "Slide Chunk")
    fgd_text          = _find_col(record, "final_graphics_definition",
                                           "final_graphics_definitionn",  # typo in sheet header
                                           "Final Graphics Definition",
                                           "final_graphics_def")
    fg_text           = _find_col(record, "final_graphics",
                                           "Final Graphics")

    if not fgd_text.strip():
        print(f"[ImagePreview] Row {row_idx + 1}: final_graphics_definition is empty — skipping")
        return []

    try:
        input_entries = _parse_input_entries(fgd_text)
    except Exception as exc:
        print(f"[ImagePreview] Row {row_idx + 1}: _parse_input_entries error — {exc}")
        input_entries = []

    try:
        output_urls = _parse_output_urls(fg_text)
    except Exception as exc:
        print(f"[ImagePreview] Row {row_idx + 1}: _parse_output_urls error — {exc}")
        output_urls = []

        print(f"[ImagePreview] Row {row_idx + 1} '{slide_chunk_title[:40]}': "
            f"{len(input_entries)} subseg(s), "
            f"{sum(1 for e in input_entries if e['input_url'])} input image(s), "
            f"{sum(1 for u in output_urls if u.get('output_url'))} output image(s)")

    rows: List[List[str]] = []
    for i, entry in enumerate(input_entries):
        input_url  = entry["input_url"]
        output_info = output_urls[i] if i < len(output_urls) else {"output_url": "", "is_ai_generated": ""}
        output_url = output_info.get("output_url", "")
        output_is_ai_generated = output_info.get("is_ai_generated", "") == "yes"

        # Include normal rows with valid input images, plus AI-generated output
        # rows where input is intentionally absent.
        include_row = bool(input_url) or (output_is_ai_generated and bool(output_url))
        if not include_row:
            continue

        voiceover  = entry["voiceover_focus"]
        input_cell = _make_image_formula(input_url) if input_url else ""
        if (not input_url) and output_is_ai_generated and output_url:
            input_cell = "(AI Generated)"

        output_cell = _make_image_formula(output_url) if output_url else ""

        rows.append([
            topic,
            subtopic,
            slide_chunk_title,
            slide_chunk,
            voiceover,
            entry["visual_instruction"],
            input_cell,
            output_cell,
        ])
    return rows


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------

HEADERS = [
    "Topic", "Subtopic", "Slide Chunk Title", "Slide Chunk",
    "Voiceover", "Visual Instruction", "Input Image", "Output Image",
]

ROW_HEIGHT_PX   = 150   # height for data rows so images are visible
MAX_PARSE_WORKERS = 16  # threads for parallel record parsing


def build_image_preview_sheet(
    sheet_link: str,
    gc: gspread.Client,
    source_worksheet_name: str = "Slide Chunks",
    preview_sheet_name: str = "Image Preview",
    max_workers: int = MAX_PARSE_WORKERS,
) -> Tuple[int, str]:
    """
    Read ``final_graphics_definition`` and ``final_graphics`` from every row of
    *source_worksheet_name*, then create / overwrite *preview_sheet_name* with:

        Topic | Subtopic | Slide Chunk Title | Slide Chunk |
        Voiceover | Input Image | Output Image

    One row per voiceover subsegment.  Images are embedded as =IMAGE() formulas
    (Drive links are converted to thumbnail URLs so they render without sign-in).

    Processing is **parallel** — all source records are parsed concurrently;
    a single batched API write is used for the final output.

    Parameters
    ----------
    sheet_link:             Full Google Sheets URL.
    gc:                     Authenticated gspread client.
    source_worksheet_name:  Tab containing final_graphics_definition / final_graphics.
    preview_sheet_name:     Output tab name (created if absent, cleared if present).
    max_workers:            Thread count for parallel record parsing.

    Returns
    -------
    (rows_written, preview_tab_url)
    """
    # ── 1. Open spreadsheet + read source tab ─────────────────────────────────
    m = re.search(r"/d/([a-zA-Z0-9_-]+)", sheet_link)
    sheet_id    = m.group(1) if m else sheet_link
    spreadsheet = gc.open_by_key(sheet_id)
    source_ws   = spreadsheet.worksheet(source_worksheet_name)
    all_records = source_ws.get_all_records()

    print(f"[ImagePreview] {len(all_records)} source rows to process "
          f"(up to {max_workers} parallel workers)")

    # ── Diagnostic: print actual column names from first record ──────────────
    if all_records:
        first_keys = list(all_records[0].keys())
        print(f"[ImagePreview] Sheet columns detected: {first_keys}")
        fgd_val = _find_col(all_records[0],
                            "final_graphics_definition",
                            "final_graphics_definitionn",  # typo in sheet header
                            "Final Graphics Definition",
                            "final_graphics_def")
        fg_val  = _find_col(all_records[0], "final_graphics", "Final Graphics")
        print(f"[ImagePreview] Row 1 — fgd length: {len(fgd_val)}, "
              f"fg length: {len(fg_val)}")
        if fgd_val:
            print(f"[ImagePreview] Row 1 — fgd preview: {fgd_val[:120]!r}")
        else:
            print(f"[ImagePreview] ⚠️  final_graphics_definition not found / empty in row 1!")

    # ── 2. Parse all records in parallel ──────────────────────────────────────
    # _process_record is pure CPU/regex — no network — so parallelism is safe.
    ordered_output: List[List[List[str]]] = [None] * len(all_records)  # type: ignore

    with ThreadPoolExecutor(max_workers=max_workers,
                            thread_name_prefix="ImagePreview") as pool:
        future_map = {
            pool.submit(_process_record, rec, idx): idx
            for idx, rec in enumerate(all_records)
        }
        for future in as_completed(future_map):
            idx = future_map[future]
            try:
                ordered_output[idx] = future.result()
            except Exception as exc:
                print(f"[ImagePreview] Row {idx + 1} unhandled error: {exc}")
                ordered_output[idx] = []

    # ── 3. Flatten results (preserving row order) ──────────────────────────────
    output_rows: List[List[str]] = [HEADERS]
    for row_group in ordered_output:
        if row_group:
            output_rows.extend(row_group)

    print(f"[ImagePreview] {len(output_rows) - 1} voiceover rows generated")

    # ── 4. Create / clear preview worksheet ───────────────────────────────────
    try:
        preview_ws = spreadsheet.worksheet(preview_sheet_name)
        preview_ws.clear()
        print(f"[ImagePreview] Cleared existing '{preview_sheet_name}' tab")
    except gspread.exceptions.WorksheetNotFound:
        preview_ws = spreadsheet.add_worksheet(
            title=preview_sheet_name,
            rows=max(len(output_rows) + 10, 50),
            cols=len(HEADERS),
        )
        print(f"[ImagePreview] Created new '{preview_sheet_name}' tab")

    # ── 5. Write all rows in one API call ─────────────────────────────────────
    #    USER_ENTERED so =IMAGE() is evaluated as a formula (not stored as text)
    if output_rows:
        preview_ws.update(
            range_name="A1",
            values=output_rows,
            value_input_option="USER_ENTERED",
        )
        print(f"[ImagePreview] Wrote {len(output_rows)} rows (incl. header)")

    # ── 6. Auto-resize data rows to ROW_HEIGHT_PX so images are visible ───────
    data_rows = len(output_rows) - 1   # exclude header
    if data_rows > 0:
        try:
            spreadsheet.batch_update({
                "requests": [
                    {
                        "updateDimensionProperties": {
                            "range": {
                                "sheetId":    preview_ws.id,
                                "dimension":  "ROWS",
                                "startIndex": 1,              # 0-based; skip header
                                "endIndex":   1 + data_rows,
                            },
                            "properties": {"pixelSize": ROW_HEIGHT_PX},
                            "fields":     "pixelSize",
                        }
                    }
                ]
            })
            print(f"[ImagePreview] Row height set to {ROW_HEIGHT_PX}px for {data_rows} rows")
        except Exception as exc:
            print(f"[ImagePreview] Row resize skipped: {exc}")

    # ── 7. Return stats + direct URL to the preview tab ───────────────────────
    preview_url = (
        f"https://docs.google.com/spreadsheets/d/{sheet_id}"
        f"/edit#gid={preview_ws.id}"
    )
    return data_rows, preview_url
