"""
reference_image_pipeline.py
============================
Pipeline for processing sheets where each row's Slide Chunk contains
*inline* reference image links in markdown format:

    [alt text](https://drive.google.com/file/d/.../view)

Each such marker splits the chunk into a (voiceover_segment, drive_link) pair.
All pairs within a row are processed in parallel through the full
Accuracy + Copyright review-and-edit pipeline.

Key differences from the standard automated_voiceover_reviewer pipeline:
  - There is NO dedicated "Reference Image" column — images are detected
    inline from the Slide Chunk text using [alt](url) markdown syntax.
  - The [alt text] label is stripped from the voiceover; only the URL is used.
  - Each inline marker becomes one independent work item (one processed image).
  - Voiceover ↔ image pairing is explicit — no Gemini extraction needed.
  - Results are saved to Drive sub-folders (one per segment per row).

Reuses:
  - review_and_edit_image()  — the full Accuracy + Copyright orchestration loop
  - _download_drive_image()  — robust Drive / web image download
  - _save_image_to_drive()   — thread-safe Drive upload
  - _create_drive_subfolder()— Drive folder management
  All imported directly from automated_voiceover_reviewer.

Usage (standalone):
    python reference_image_pipeline.py --url <SHEET_URL> --tab <TAB_NAME>

Usage (programmatic / Streamlit):
    from agents.graphics_asset_creation.automated.reference_image_pipeline import (
        run_reference_image_pipeline
    )
    run_reference_image_pipeline(
        sheet_url="https://...",
        source_tab="My Tab",
        gc=gc,
        drive=drive,
        progress_callback=my_callback,
    )
"""

import os
import re
import sys
import json
import time
import threading
import traceback
from typing import Optional, Callable, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from PIL import Image
import gspread
import pandas as pd

# ── Path resolution (same pattern as automated_voiceover_reviewer.py) ─────────
current_dir = os.path.dirname(os.path.abspath(__file__))   # .../automated
agents_dir  = os.path.dirname(current_dir)                  # .../graphics_asset_creation
root_dir    = os.path.dirname(os.path.dirname(agents_dir))  # workspace root
sys.path.insert(0, root_dir)

# ── Reuse the core pipeline components from automated_voiceover_reviewer ───────
from pydrive2.drive import GoogleDrive
from gspread.utils import rowcol_to_a1
from google.genai import types

from agents.graphics_asset_creation.automated.automated_voiceover_reviewer import (
    review_and_edit_image,
    _download_drive_image,
    _save_image_to_drive,
    _create_drive_subfolder,
)
from agents.graphics_asset_creation.automated.llm_call_tracker import tracker as llm_tracker
from agents.graphics_asset_creation.gac_utils import (
    get_or_create_drive_folder,
    is_valid_web_image_link,
    _get_client,
    prepare_image_for_gemini,
    DEFAULT_REVIEWER_MODEL,
)
from services.drive_service import login_with_service_account
from services.sheets_service import get_sheet_data_and_df

load_dotenv()

# =============================================================================
# COLUMN NAMES (update here if sheet headers change)
# =============================================================================
COL_SLIDE_CHUNK_TITLE  = "Slide Chunk Title"   # used as slide_title
COL_SLIDE_CHUNK        = "Slide Chunk"          # contains inline [alt](url) image markers
# NOTE: There is NO dedicated "Reference Image" column. Images are detected
# inline from COL_SLIDE_CHUNK using [alt text](drive_url) markdown syntax.
COL_EDITED_REFERENCE   = "Edited Reference"     # output — =IMAGE() formula rendered in sheet
COL_RELEVANT_VOICEOVER = "Relevant Voiceover"   # output — the specific VO segment for each image

# =============================================================================
# MAIN PIPELINE FUNCTION
# =============================================================================

def run_reference_image_pipeline(
    sheet_url: str,
    source_tab: str,
    output_folder_name: str = "Reference Image Pipeline",
    gc=None,
    drive=None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    skip_filled_rows: bool = False,
    execution_mode: str = "full",  # "full", "step1" (extraction), "step2" (review)
) -> None:
    """
    Batch-process a sheet where each row's Slide Chunk contains inline
    reference image markers of the form  [alt text](drive_url).

    Each marker is parsed into a (voiceover_segment, drive_url) pair.
    All pairs across all rows are processed in parallel.

    Phase 1 [Serial]   — Scan all rows, parse inline [alt](url) markers from
                         Slide Chunk, validate Drive links, pre-create folders.
    Phase 2 [Parallel] — Launch one worker thread per valid segment, running the
                         full Accuracy + Copyright review-and-edit pipeline.
    Phase 3 [Cleanup]  — Log any skipped rows/segments.

    Args:
        sheet_url:          Full Google Sheets URL.
        source_tab:         Name of the worksheet tab to read from and write to.
        output_folder_name: Name of the root Drive folder for saving edited images.
        gc:                 Authenticated gspread client (optional — will auth from env).
        drive:              Authenticated PyDrive2 GoogleDrive client (optional).
        progress_callback:  Optional callable(completed: int, total: int) for UI progress.
        skip_filled_rows:   If True, skip rows that already have content in "Edited Reference".
    """

    # Thread-safety lock (pydrive2 is not thread-safe)
    drive_lock  = threading.Lock()

    # Reset LLM usage tracker for this run
    llm_tracker.reset()

    try:
        # ── Authenticate if not provided ──────────────────────────────────────
        if gc is None or drive is None:
            service_account_json = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
            if not service_account_json:
                raise ValueError("Missing GOOGLE_SERVICE_ACCOUNT_JSON env var.")
            gauth = login_with_service_account(json_str=service_account_json)
            drive = GoogleDrive(gauth)
            gc    = gspread.service_account_from_dict(json.loads(service_account_json))

        # ── Phase 1: Setup ────────────────────────────────────────────────────
        output_folder_id = get_or_create_drive_folder(drive, output_folder_name)
        sheet            = gc.open_by_url(sheet_url)
        ws_source        = sheet.worksheet(source_tab)
        
        raw_records = ws_source.get_all_records()
        
        # Ensure output columns exist in the sheet
        source_headers = ws_source.row_values(1)
        columns_to_ensure = [COL_EDITED_REFERENCE, COL_RELEVANT_VOICEOVER]
        
        columns_added = False
        for col_name in columns_to_ensure:
            if col_name not in source_headers:
                next_col = len(source_headers) + 1
                ws_source.update(
                    range_name=rowcol_to_a1(1, next_col),
                    values=[[col_name]]
                )
                source_headers.append(col_name)
                print(f"📌 Added '{col_name}' column at position {next_col}")
                columns_added = True

        if columns_added:
            raw_records = ws_source.get_all_records()
        
        df = pd.DataFrame(raw_records)
        for col in columns_to_ensure:
            if col not in df.columns:
                df[col] = ""

        print(f"\n📊 Source tab '{source_tab}' has {len(df)} rows")
        print(f"📊 Columns: {list(df.columns)}")

        edited_ref_col_idx = source_headers.index(COL_EDITED_REFERENCE) + 1  # 1-based
        relevant_vo_col_idx = source_headers.index(COL_RELEVANT_VOICEOVER) + 1  # 1-based

        # ── Column Normalization (Case-Insensitive) ───────────────────────────
        df_cols = list(df.columns)
        def _find_col(target):
            for c in df_cols:
                if str(c).strip().lower() == target.lower():
                    return c
            return target

        col_slide_title = _find_col(COL_SLIDE_CHUNK_TITLE)
        col_slide_chunk = _find_col(COL_SLIDE_CHUNK)

        print(f"📌 Column mapping: ")
        print(f"   - '{COL_SLIDE_CHUNK_TITLE}' -> '{col_slide_title}'")
        print(f"   - '{COL_SLIDE_CHUNK}'       -> '{col_slide_chunk}'")
        print(f"   ℹ️  Reference images are parsed inline from '{col_slide_chunk}' using [alt](url) markers.")

        # ── Phase 1: Scan rows and parse inline reference image markers ────────
        print("\n🔍 Phase 1: Scanning all rows, parsing inline [alt](url) markers…")

        row_meta          = []           # one meta entry per source row
        global_work_items = []           # one work item per (row, segment) pair
        rows_written: set = set()        # (df_idx, seg_idx) tuples already handled

        for df_idx, (_, row) in enumerate(df.iterrows()):
            sheet_row_number = df_idx + 2  # +1 for 1-based, +1 for header

            slide_title = str(row.get(col_slide_title, "")).strip()
            slide_chunk = str(row.get(col_slide_chunk, "")).strip()

            if not slide_chunk:
                continue

            # ── Parse inline reference image markers ───────────────────────────
            # Format: [alt text](https://drive.google.com/file/d/.../view)
            # Each marker splits the chunk: text-before-marker → VO, url → ref image
            segments = parse_inline_references(slide_chunk)

            print(f"\n  Row {df_idx + 1}: {slide_title[:60] or '(no title)'}")

            if not segments:
                # ── NORMAL SLIDE: no inline markers — treat full chunk as one VO ──
                print(f"    📄 No inline markers → Normal slide (full chunk as single VO).")

                row_folder_name = f"Row_{df_idx + 1}_{slide_title[:30].replace('/', '_')}"
                row_folder_id = _create_drive_subfolder(
                    drive, output_folder_id, row_folder_name, drive_lock=drive_lock
                )

                work_item = {
                    "df_idx"         : df_idx,
                    "seg_idx"        : 0,
                    "sheet_row_number": sheet_row_number,
                    "slide_title"    : slide_title,
                    "slide_chunk"    : slide_chunk,
                    "voiceover_text" : slide_chunk,  # full chunk is the VO
                    "plain_link"     : None,           # no reference image
                    "row_folder_id"  : row_folder_id,
                    "slide_type"     : "normal",
                    "valid"          : True,
                }
                global_work_items.append(work_item)
                row_meta.append({
                    "df_idx"         : df_idx,
                    "sheet_row_number": sheet_row_number,
                    "slide_title"    : slide_title,
                    "num_segments"   : 1,
                    "slide_type"     : "normal",
                })
                continue  # move to next row

            # ── REFERENCE SLIDE: has inline markers → one work item per segment ─
            print(f"    🖼️  Found {len(segments)} inline reference image segment(s).")

            # Pre-create a parent Drive folder for this row (serial — not thread-safe)
            row_folder_name = f"Row_{df_idx + 1}_{slide_title[:30].replace('/', '_')}"
            row_folder_id = _create_drive_subfolder(
                drive, output_folder_id, row_folder_name, drive_lock=drive_lock
            )
            print(f"    📁 Created row folder: {row_folder_name}")

            for seg_idx, (vo_text, ref_image_url) in enumerate(segments):
                seg_label = f"Row {df_idx + 1} Seg {seg_idx + 1}"

                if not vo_text:
                    print(f"    ⚠️  {seg_label}: Empty voiceover segment — skipping.")
                    continue

                # ── Validate the inline drive link ─────────────────────────────
                plain_link = re.sub(
                    r'\s*\(snapshot\)\s*', '', ref_image_url, flags=re.IGNORECASE
                ).strip()

                link_is_valid = is_valid_web_image_link(plain_link)
                # Allow any Drive link (even without snapshot marker)
                if not link_is_valid:
                    if any(d in plain_link.lower() for d in ['drive.google.com', 'docs.google.com']):
                        link_is_valid = True

                if not link_is_valid:
                    print(f"    ⚠️  {seg_label}: Invalid drive link '{plain_link[:80]}' — skipping.")
                    continue

                # Pre-create a segment-level sub-folder
                seg_folder_name = f"Seg_{seg_idx + 1}_{vo_text[:25].replace('/', '_')}"
                seg_folder_id = _create_drive_subfolder(
                    drive, row_folder_id, seg_folder_name, drive_lock=drive_lock
                )
                print(f"      📁 {seg_label}: folder='{seg_folder_name}' | VO='{vo_text[:60]}...'")

                work_item = {
                    "df_idx"         : df_idx,
                    "seg_idx"        : seg_idx,
                    "sheet_row_number": sheet_row_number,
                    "slide_title"    : slide_title,
                    "slide_chunk"    : slide_chunk,   # full original chunk (for context)
                    "voiceover_text" : vo_text,       # the specific segment (pre-parsed)
                    "plain_link"     : plain_link,
                    "row_folder_id"  : seg_folder_id,
                    "slide_type"     : "reference",
                    "valid"          : True,
                }
                global_work_items.append(work_item)

            row_meta.append({
                "df_idx"         : df_idx,
                "sheet_row_number": sheet_row_number,
                "slide_title"    : slide_title,
                "num_segments"   : len(segments),
                "slide_type"     : "reference",
            })

        total_rows = len(global_work_items)
        print(f"\n📊 Phase 1 complete.")
        print(f"   Sheet rows scanned   : {len(df)}")
        print(f"   Rows with segments   : {len(row_meta)}")
        print(f"   Total work items     : {total_rows}  (1 per inline image marker)")
        print(f"   Workers to launch    : {total_rows}")

        if progress_callback:
            try:
                progress_callback(0, total_rows)
            except Exception:
                pass

        if total_rows == 0:
            print("⚠️  No valid rows to process. Nothing to do.")
            return


        # ── Worker function ──────────────────────────────────────────────────
        def _process_one_row(work_item: dict):
            """
            End-to-end processing for one work item. Branches on slide_type:

            - "reference": Has an inline Drive link → downloads the image and
              passes it as reference_image to review_and_edit_image().
            - "normal": No inline markers → passes reference_image=None 

            Returns (df_idx, seg_idx, drive_link_or_None, voiceover_text).
            """
            df_idx_w       = work_item["df_idx"]
            seg_idx_w      = work_item["seg_idx"]
            slide_title_w  = work_item["slide_title"]
            slide_chunk_w  = work_item["slide_chunk"]   # full chunk — used as slide_content
            voiceover_w    = work_item["voiceover_text"] # pre-parsed segment (or full chunk)
            plain_link_w   = work_item["plain_link"]     # None for normal slides
            seg_folder_id  = work_item["row_folder_id"]
            slide_type_w   = work_item.get("slide_type", "reference")

            thread_name      = threading.current_thread().name
            final_drive_link = None
            reviewer_counters = {"voiceover": 0, "copyright": 0}

            type_label = "📄 Normal" if slide_type_w == "normal" else "🖼️  Reference"
            seg_label  = f"Row {df_idx_w + 1} Seg {seg_idx_w + 1} [{type_label}]"

            print(f"\n[{thread_name}] ▶ {seg_label}: {slide_title_w[:40]} | VO='{voiceover_w[:60]}'")

            for retry in range(3):
                try:
                    if retry > 0:
                        time.sleep(retry * 2)
                        print(f"  [{thread_name}] Retry {retry}/2 for {seg_label}…")

                    # ── Step 1: Obtain reference image (type-dependent) ────────
                    if slide_type_w == "reference":
                        # Download the Drive image linked inline in the chunk
                        with drive_lock:
                            ref_image = _download_drive_image(plain_link_w, drive=drive)
                        print(f"    🖼️  [{thread_name}] Reference image downloaded from Drive.")
                    else:
                        # Normal slide — no reference image; pipeline generates from scratch
                        ref_image = None
                        print(f"    📄 [{thread_name}] Normal slide — no reference image.")

                    # ── Step 2: Log voiceover (already determined in Phase 1) ──
                    print(f"    📝 [{thread_name}] VO: '{voiceover_w[:100]}…'")

                    if execution_mode == "step1":
                        # Extraction-only mode — return the VO without running review
                        print(f"    ✅ [{thread_name}] Step 1 complete. VO ready.")
                        return df_idx_w, seg_idx_w, None, voiceover_w

                    # ── Step 3: Intermediate callback — save review rounds ─────
                    def _save_intermediate(
                        round_data: Dict[str, Any],
                        _folder_id=seg_folder_id,
                        _counters=reviewer_counters,
                    ):
                        agent_type = round_data["agent"]
                        img        = round_data["image"]
                        if "Voiceover" in agent_type:
                            _counters["voiceover"] += 1
                            fname = f"Reviewer_1_Round_{_counters['voiceover']}.png"
                        else:
                            _counters["copyright"] += 1
                            fname = f"Reviewer_2_Round_{_counters['copyright']}.png"
                        try:
                            _save_image_to_drive(img, fname, drive, _folder_id, drive_lock=drive_lock)
                            print(f"    💾 [{thread_name}] Saved intermediate: {fname}")
                        except Exception as cb_err:
                            print(f"    ⚠️  [{thread_name}] Failed to save {fname}: {cb_err}")

                    # ── Step 4: Run the full Accuracy + Copyright pipeline ─────
                    _, final_image, _ = review_and_edit_image(
                        reference_image    = ref_image,
                        slide_title        = slide_title_w,
                        slide_content      = slide_chunk_w,
                        voiceover          = voiceover_w,
                        visual_instruction = "",
                        target_stage       = "full",
                        callback           = _save_intermediate,
                    )
                    
                    # ── Step 5: Upload the final approved image ────────────────
                    if final_image:
                        upload_link = _save_image_to_drive(
                            final_image, "FINAL_Image.png", drive, seg_folder_id, drive_lock=drive_lock
                        )
                        if upload_link:
                            final_drive_link = upload_link
                            print(f"  [{thread_name}] ✅ {seg_label}: uploaded → {upload_link}")
                        else:
                            print(f"  [{thread_name}] ⚠️  {seg_label}: upload returned empty link")
                    else:
                        print(f"  [{thread_name}] ⚠️  {seg_label}: no final image produced")

                    break  # success — exit retry loop

                except Exception as exc:
                    if retry == 2:
                        print(
                            f"  [{thread_name}] ❌ {seg_label} failed after 2 retries: {exc}"
                        )
                        traceback.print_exc()

            return df_idx_w, seg_idx_w, final_drive_link, voiceover_w

        # ── Write helper ─────────────────────────────────────────────────────
        def _write_row_result(meta_entry: dict, drive_link: Optional[str], extracted_vo: str = "") -> None:
            """
            Write results back to the sheet:
              1. =IMAGE("LINK") formula to 'Edited Reference'
              2. Selected VO text to 'Relevant Voiceover'
            """
            nonlocal ws_source
            if not meta_entry:
                return

            sheet_row_w = meta_entry["sheet_row_number"]
            cell_addr_img = rowcol_to_a1(sheet_row_w, edited_ref_col_idx)
            cell_addr_vo = rowcol_to_a1(sheet_row_w, relevant_vo_col_idx)

            # Build =IMAGE() formula when a Drive link is available;
            # Use the "uc?export=view&id=..." format required for inline rendering.
            if drive_link:
                # Extract file ID to ensure the specific uc?export=view format
                file_id = None
                id_match = re.search(r'[/=]([a-zA-Z0-9_-]{25,})(?:[/?&]|$)', drive_link)
                if id_match:
                    file_id = id_match.group(1)
                
                if file_id:
                    final_url = f"https://drive.google.com/uc?export=view&id={file_id}"
                else:
                    final_url = drive_link
                    
                img_cell_value = f'=IMAGE("{final_url}")'
                print(f"  🖼️  Writing IMAGE formula for row {sheet_row_w}: {img_cell_value}")
            elif extracted_vo == "IRRELEVANT IMAGE":
                img_cell_value = "IRRELEVANT IMAGE"
            else:
                img_cell_value = ""

            for attempt in range(3):
                try:
                    if attempt > 0:
                        wait_s = 2 ** attempt
                        print(f"  ⏳ Retry {attempt}/2 writing row {sheet_row_w} (wait {wait_s}s)…")
                        time.sleep(wait_s)
                        try:
                            ws_source = sheet.worksheet(source_tab)
                        except Exception:
                            pass
                    
                    # 1. Update image column (only if we have a result or it's irrelevant)
                    if img_cell_value:
                        ws_source.update(
                            range_name=cell_addr_img,
                            values=[[img_cell_value]],
                            value_input_option="USER_ENTERED",
                        )
                    # 2. Update Relevant Voiceover column
                    if extracted_vo:
                        ws_source.update(
                            range_name=cell_addr_vo,
                            values=[[extracted_vo]],
                        )
                    print(f"  ✏️  Written results for row {sheet_row_w} → {cell_addr_img}")
                    break
                except Exception as write_err:
                    print(f"  ❌ Attempt {attempt + 1}/3 failed writing row {sheet_row_w}: {write_err}")
                    traceback.print_exc()
            else:
                print(f"  🆘 ALL WRITE ATTEMPTS FAILED for row {sheet_row_w}.")

        # ── Phase 2: Parallel execution — one worker per valid row ────────────
        print(f"\n🚀 Phase 2: Launching {total_rows} parallel worker(s)…")

        # Refresh worksheet handle before writes start
        try:
            ws_source = sheet.worksheet(source_tab)
        except Exception as ws_err:
            print(f"  ⚠️  Could not pre-refresh worksheet: {ws_err} — using original handle")

        completed = 0
        with ThreadPoolExecutor(
            max_workers=total_rows,
            thread_name_prefix="RefImgWorker"
        ) as pool:
            futures = {
                pool.submit(_process_one_row, wi): wi
                for wi in global_work_items
            }
            for future in as_completed(futures):
                wi = futures[future]
                try:
                    r_df_idx, r_seg_idx, link, vo_text = future.result()
                except Exception as fut_err:
                    r_df_idx  = wi["df_idx"]
                    r_seg_idx = wi["seg_idx"]
                    link      = None
                    vo_text   = ""
                    print(f"\n  ❌ Row {r_df_idx + 1} Seg {r_seg_idx + 1} future raised: {fut_err}")

                completed += 1
                print(
                    f"\n  ✔ [{completed}/{total_rows}] "
                    f"Row {r_df_idx + 1} Seg {r_seg_idx + 1} complete (Mode: {execution_mode})"
                )

                if progress_callback:
                    try:
                        progress_callback(completed, total_rows)
                    except Exception:
                        pass

                # Write result for this segment
                rows_written.add((r_df_idx, r_seg_idx))
                _write_row_result(wi, link, vo_text)

        # ── Phase 3: Log unprocessed segments ─────────────────────────────────
        print(f"\n📝 Phase 3: Logging any skipped segments…")
        for wi in global_work_items:
            key = (wi["df_idx"], wi["seg_idx"])
            if key not in rows_written:
                print(
                    f"  ⚠️  Row {wi['df_idx'] + 1} Seg {wi['seg_idx'] + 1}: "
                    f"was not completed — check logs above."
                )

        print(
            f"\n🎉 Reference Image Pipeline complete. "
            f"Processed {completed}/{total_rows} inline image segment(s). "
            f"Results saved to Drive folder '{output_folder_name}'."
        )

    except Exception as exc:
        print(f"Pipeline Error: {exc}")
        traceback.print_exc()
    finally:
        llm_tracker.mark_pipeline_end()


# =============================================================================
# INLINE REFERENCE IMAGE PARSING
# =============================================================================

# Matches: [any alt text](https://...)
_INLINE_REF_PATTERN = re.compile(
    r'\[([^\]]*)\]'           # [alt text]  — captured but ignored
    r'\((https?://[^\)]+)\)', # (url)       — the Drive link we want
    re.IGNORECASE,
)


def parse_inline_references(slide_chunk: str) -> list:
    """
    Parse a slide chunk that contains inline markdown image references:

    Returns:
        List of (voiceover_text: str, drive_url: str) tuples,
        one per inline marker found.  Empty list if no markers are present.
    """
    segments = []
    last_end = 0
    pending_text = ""

    for match in _INLINE_REF_PATTERN.finditer(slide_chunk):
        # Text from end of the previous marker up to the start of [alt]
        text_chunk = slide_chunk[last_end : match.start()]
        vo_text = (pending_text + text_chunk).strip()
        drive_url = match.group(2).strip()

        segments.append((vo_text, drive_url))

        last_end = match.end()
        pending_text = ""

    return segments


# =============================================================================
# VOICE OVER EXTRACTION UTILITIES (Legacy / Context-Based)
# =============================================================================

def _extract_best_voiceover(image: Image.Image, context_chunks: str, row_idx: int) -> str:
    """
    Calls Gemini to identify the specific part of the expanded 5-chunk context
    that best matches the provided reference image.
    """

    client = _get_client()
    if not client:
        return ""

    try:
        img_bytes = prepare_image_for_gemini(image)

        prompt = f"""You are an expert at aligning educational images with technical text.

Below are several 'Slide Chunks' representing the surrounding context of a specific slide (Row {row_idx}).
A 'Reference Image' is also provided.

TASK:
1. Determine if the 'Reference Image' is RELEVANT to the technical/educational context provided.
2. If relevant, identify the EXACT sentence, phrase, or paragraph from the provided chunks that is most accurately illustrated by the image.
3. Specify the CHUNK_ID (e.g. CHUNK_ID_5) that contains this text.

CHUNKS CONTEXT:
{context_chunks}

REQUIREMENTS:
1. IRRELEVANCE CHECK: Determine if the image is an unintended extraction error. 
   - MARK AS 'NOT_RELEVANT' ONLY IF the image is a UI component (like a padlock, gear, bell), a profile/user avatar, a generic UI button, or a non-scenic placeholder that contains no subject matter.
   - DO NOT mark scenic images, real-world photos, or technical diagrams as irrelevant. If it looks like an intended educational or scenic image, it is RELEVANT.
2. If relevant, return the selected CHUNK_ID followed by a colon and the extracted text.
   Example: CHUNK_ID_12: The blue inlet valve on the pump...
3. If the image matches the theme perfectly but not a specific sentence, return the [TARGET CHUNK] ID and its content.
4. Do not provide any explanation or conversational response. Just the 'CHUNK_ID: Text' or 'NOT_RELEVANT'.
"""

        parts = [
            types.Part.from_text(text=prompt),
            types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg")
        ]

        with llm_tracker.call(DEFAULT_REVIEWER_MODEL, "VO Extraction") as usage:
            response = client.models.generate_content(
                model=DEFAULT_REVIEWER_MODEL,
                contents=[types.Content(role="user", parts=parts)],
                config=types.GenerateContentConfig(
                    temperature=0.0,  # Strict extraction
                    max_output_tokens=500
                )
            )
            usage.set_response(response)

        return response.text.strip()

    except Exception as e:
        print(f"      ⚠️ Failed to extract best VO: {e}")
        return ""
 
 
# =============================================================================
# CLI ENTRYPOINT
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Reference Image Pipeline — runs the full Accuracy + Copyright review pipeline "
                    "on images from the 'Reference Image' column and writes results to 'Edited Reference'."
    )
    parser.add_argument("--url",    required=True, help="Google Sheets URL")
    parser.add_argument("--tab",    required=True, help="Source worksheet tab name")
    parser.add_argument("--folder", default="Reference Image Pipeline",
                        help="Output Drive folder name (default: 'Reference Image Pipeline')")
    parser.add_argument("--skip-filled", action="store_true",
                        help="Skip rows that already have a Drive link in 'Edited Reference'")
    args = parser.parse_args()

    run_reference_image_pipeline(
        sheet_url          = args.url,
        source_tab         = args.tab,
        output_folder_name = args.folder,
        skip_filled_rows   = args.skip_filled,
        execution_mode     = "full",
    )
