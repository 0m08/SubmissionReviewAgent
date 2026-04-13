"""
reference_image_pipeline.py
============================
Pipeline for processing sheets where each row contains a single *Reference Image*
that must pass through the full Accuracy + Copyright review-and-edit pipeline.

Key differences from the standard automated_voiceover_reviewer pipeline:
  - Input images come from the **"Reference Image"** column (one per row, no subsegment parsing).
  - slide_title = Slide Chunk Title.
  - slide_content + voiceover = Slide Chunk.
  - The final edited image Drive link is written to the **"Edited Reference"** column.

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
COL_SLIDE_CHUNK_TITLE = "Slide Chunk Title"   # used as slide_title
COL_SLIDE_CHUNK       = "Slide Chunk"          # used as BOTH slide_content AND voiceover
COL_REFERENCE_IMAGE   = "Reference Image"      # input image URL
COL_EDITED_REFERENCE  = "Edited Reference"     # output — =IMAGE() formula rendered in sheet
COL_RELEVANT_VOICEOVER = "Relevant Voiceover"  # output — the specific part of chunk illustrated

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
    Batch-process a sheet where each row has a single Reference Image.

    Phase 1 [Serial]   — Scan all rows, validate Reference Image links, pre-create
                         Drive folders.
    Phase 2 [Parallel] — Launch one worker thread per valid row, running the full
                         Accuracy + Copyright review-and-edit pipeline.
    Phase 3 [Cleanup]  — Write any remaining rows (e.g. skipped/empty) back.

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
        col_ref_image   = _find_col(COL_REFERENCE_IMAGE)

        print(f"📌 Column mapping: ")
        print(f"   - '{COL_SLIDE_CHUNK_TITLE}' -> '{col_slide_title}'")
        print(f"   - '{COL_SLIDE_CHUNK}'       -> '{col_slide_chunk}'")
        print(f"   - '{COL_REFERENCE_IMAGE}'   -> '{col_ref_image}'")

        if col_ref_image not in df.columns:
            print(f"❌ CRITICAL ERROR: Could not find '{COL_REFERENCE_IMAGE}' column in sheet.")
            print(f"   Available columns: {list(df.columns)}")

        # ── Phase 1: Scan rows ─────────────────────────────────────────────────
        print("\n🔍 Phase 1: Scanning all rows and pre-creating Drive folders…")

        row_meta          = []           # one entry per DataFrame row
        global_work_items = []           # valid rows to process in Phase 2
        rows_written: set = set()        # df_idx values already handled/skipped

        for df_idx, (_, row) in enumerate(df.iterrows()):
            sheet_row_number = df_idx + 2  # +1 for 1-based, +1 for header

            # ── Resume / skip already-processed rows ──────────────────────────
            if skip_filled_rows:
                existing = str(row.get(COL_EDITED_REFERENCE, "")).strip()
                if existing and (
                    "drive.google.com" in existing.lower()
                    or existing.upper().startswith("=IMAGE(")
                ):
                    print(f"  ⏭️  Row {df_idx + 1}: Already has '{COL_EDITED_REFERENCE}'. Skipping.")
                    rows_written.add(df_idx)
                    continue

            slide_title   = str(row.get(col_slide_title, "")).strip()
            slide_chunk   = str(row.get(col_slide_chunk, "")).strip()
            ref_image_url = str(row.get(col_ref_image, "")).strip()

            if not ref_image_url:
                continue

            # ── Collect Expanded Context (2 above, 2 below) ───────────
            context_list = []
            context_mapping = {}  # ID -> {title, chunk}
            
            for offset in range(-2, 3):
                neighbor_idx = df_idx + offset
                if 0 <= neighbor_idx < len(df):
                    n_row = df.iloc[neighbor_idx]
                    n_title = str(n_row.get(col_slide_title, "")).strip()
                    n_chunk = str(n_row.get(col_slide_chunk, "")).strip()
                    
                    if n_chunk:
                        marker = f"CHUNK_ID_{neighbor_idx+1}"
                        label = f"[TARGET CHUNK]" if offset == 0 else f"[NEIGHBOR {neighbor_idx+1}]"
                        context_list.append(f"{label} <{marker}>\n{n_chunk}\n</{marker}>")
                        context_mapping[marker] = {"title": n_title, "chunk": n_chunk}
            
            context_chunks_str = "\n\n---\n\n".join(context_list)
            relevant_vo_existing = str(row.get(COL_RELEVANT_VOICEOVER, "")).strip()

            meta = {
                "df_idx"          : df_idx,
                "sheet_row_number": sheet_row_number,
                "slide_title"     : slide_title,
                "slide_chunk"     : slide_chunk,
                "context_chunks"  : context_chunks_str,
                "context_mapping" : context_mapping,
                "ref_image_url"   : ref_image_url,
                "relevant_vo_existing": relevant_vo_existing,
                "valid"           : False,
            }
            row_meta.append(meta)

            print(f"\n  Row {df_idx + 1}: {slide_title[:60] or '(no title)'}")

            # ── Validate reference image link ──────────────────────────────────
            plain_link = re.sub(r'\s*\(snapshot\)\s*', '', ref_image_url, flags=re.IGNORECASE).strip()

            link_is_valid = is_valid_web_image_link(plain_link)
            
            # Local override: Reference Image Pipeline allows any Drive link even without (snapshot)
            if not link_is_valid:
                if any(domain in plain_link.lower() for domain in ['drive.google.com', 'docs.google.com']):
                    link_is_valid = True
            
            if not link_is_valid:
                print(f"    ⚠️  Row {df_idx + 1}: Invalid link '{plain_link[:80]}' (From col '{col_ref_image}')")
                continue

            if not slide_chunk:
                print(f"    ⚠️  Empty '{COL_SLIDE_CHUNK}' — skipping.")
                continue

            # Pre-create Drive folder for this row (serial — Drive API not thread-safe)
            folder_name = f"Row_{df_idx + 1}_{slide_title[:30].replace('/', '_')}"
            row_folder_id = _create_drive_subfolder(
                drive, output_folder_id, folder_name, drive_lock=drive_lock
            )
            print(f"    📁 Created folder: {folder_name}")

            meta["valid"]         = True
            meta["plain_link"]    = plain_link
            meta["row_folder_id"] = row_folder_id

            global_work_items.append(meta)

        total_rows = len(global_work_items)
        print(f"\n📊 Phase 1 complete.")
        print(f"   Rows scanned  : {len(df)}")
        print(f"   Valid rows    : {total_rows}")
        print(f"   Workers to launch: {total_rows}  (1 per valid row)")

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
            End-to-end processing for one row:
              1. Download the Reference Image.
              2. Run the full Accuracy + Copyright review-and-edit pipeline,
                 using the Slide Chunk as the voiceover.
              3. Upload the final approved image to Drive.
              4. Return (df_idx, drive_link_or_None).
            """
            df_idx_w        = work_item["df_idx"]
            slide_chunk_w   = work_item["slide_chunk"]    # fallback
            context_chunks_w = work_item["context_chunks"]
            context_mapping_w = work_item.get("context_mapping", {})
            plain_link_w     = work_item["plain_link"]
            row_folder_id_w  = work_item["row_folder_id"]
            relevant_vo_existing_w = work_item.get("relevant_vo_existing", "")
            slide_title_original = work_item["slide_title"]

            thread_name     = threading.current_thread().name
            final_drive_link = None
            reviewer_counters = {"voiceover": 0, "copyright": 0}

            print(f"\n[{thread_name}] ▶ Row {df_idx_w + 1}: {slide_title_original[:50]}")

            for retry in range(3):
                try:
                    if retry > 0:
                        time.sleep(retry * 2)
                        print(f"  [{thread_name}] Retry {retry}/2 for Row {df_idx_w + 1}…")

                    # 1. Download the reference image (Drive or plain web URL)
                    with drive_lock:
                        ref_image = _download_drive_image(plain_link_w, drive=drive)
                    
                    # 2. VO Extraction Logic
                    extracted_vo = ""
                    source_id = None
                    
                    if execution_mode == "step2" and relevant_vo_existing_w:
                        print(f"    📝 [{thread_name}] Using existing Voiceover from sheet.")
                        extracted_vo = relevant_vo_existing_w
                    else:
                        print(f"    🔍 [{thread_name}] Extracting best voiceover from 5-chunk context…")
                        extracted_vo_raw = _extract_best_voiceover(
                            ref_image, context_chunks_w, row_idx=df_idx_w + 1
                        )
                        
                        if ": " in extracted_vo_raw and "CHUNK_ID_" in extracted_vo_raw:
                            source_id, extracted_vo = extracted_vo_raw.split(": ", 1)
                            source_id = source_id.strip()
                            extracted_vo = extracted_vo.strip()
                        else:
                            extracted_vo = extracted_vo_raw
                    
                    # ── IRRELEVANCE FILTER ──────────────────────────────────────────
                    if extracted_vo.strip().upper() in ["NOT_RELEVANT", "IRRELEVANT IMAGE"]:
                        print(f"    🚫 [{thread_name}] Image identified as NOT RELEVANT. Skipping review.")
                        return df_idx_w, None, "IRRELEVANT IMAGE"
                    
                    if execution_mode == "step1":
                        print(f"    ✅ [{thread_name}] Step 1 Complete (Extraction Only).")
                        return df_idx_w, None, extracted_vo

                    # Identify the correct Slide Title and Content to use for review
                    source_info = context_mapping_w.get(source_id) if source_id else None
                    if source_info:
                        title_to_use   = source_info["title"]
                        content_to_use = source_info["chunk"]
                        print(f"    🎯 [{thread_name}] Switched context to {source_id}: {title_to_use[:40]}...")
                    else:
                        title_to_use   = slide_title_original
                        content_to_use = slide_chunk_w

                    voiceover_to_use = extracted_vo or content_to_use
                    
                    print(f"    📢 [{thread_name}] Selected Voiceover: {voiceover_to_use[:100]}...")

                    # 3. Intermediate callback — save every review round to Drive
                    def _save_intermediate(
                        round_data: Dict[str, Any],
                        _folder_id=row_folder_id_w,
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

                    # 4. Run the full Accuracy + Copyright pipeline.
                    _, final_image, _ = review_and_edit_image(
                        reference_image     = ref_image,
                        slide_title         = title_to_use,   # ← Refined
                        slide_content       = content_to_use, # ← Refined
                        voiceover           = voiceover_to_use,
                        visual_instruction  = "",
                        image_size          = "1K",
                        target_stage        = "full",
                        callback            = _save_intermediate,
                    )

                    # 4. Upload the final approved image
                    if final_image:
                        upload_link = _save_image_to_drive(
                            final_image, "FINAL_Image.png", drive, row_folder_id_w, drive_lock=drive_lock
                        )
                        if upload_link:
                            final_drive_link = upload_link
                            print(f"  [{thread_name}] ✅ Row {df_idx_w + 1}: uploaded → {upload_link}")
                        else:
                            print(f"  [{thread_name}] ⚠️  Row {df_idx_w + 1}: upload returned empty link")
                    else:
                        print(f"  [{thread_name}] ⚠️  Row {df_idx_w + 1}: no final image produced")

                    break  # success — exit retry loop

                except Exception as exc:
                    if retry == 2:
                        print(
                            f"  [{thread_name}] ❌ Row {df_idx_w + 1} failed after 2 retries: {exc}"
                        )
                        traceback.print_exc()

            return df_idx_w, final_drive_link, voiceover_to_use

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
                    r_df_idx, link, extracted_vo = future.result()
                except Exception as fut_err:
                    r_df_idx = wi["df_idx"]
                    link     = None
                    extracted_vo = ""
                    print(f"\n  ❌ Row {r_df_idx + 1} future raised: {fut_err}")

                completed += 1
                print(f"\n  ✔ [{completed}/{total_rows}] Row {r_df_idx + 1} complete (Mode: {execution_mode})")

                if progress_callback:
                    try:
                        progress_callback(completed, total_rows)
                    except Exception:
                        pass

                # Write result row immediately
                rows_written.add(r_df_idx)
                _write_row_result(wi, link, extracted_vo)

        # ── Phase 3: Handle rows that were skipped (invalid link / empty chunk) ─
        print(f"\n📝 Phase 3: Writing any remaining skipped rows…")
        for meta in row_meta:
            if meta["df_idx"] in rows_written:
                continue
            # Row was invalid — write an empty value to Edited Reference
            print(f"  Row {meta['df_idx'] + 1}: skipped (invalid/missing data) — writing empty cell")
            _write_row_result(meta, None)
            rows_written.add(meta["df_idx"])

        print(
            f"\n🎉 Reference Image Pipeline complete. "
            f"Processed {completed} row(s). "
            f"Results written to '{COL_EDITED_REFERENCE}' column."
        )

    except Exception as exc:
        print(f"Pipeline Error: {exc}")
        traceback.print_exc()
    finally:
        llm_tracker.mark_pipeline_end()
 
 
# =============================================================================
# VOICE OVER EXTRACTION UTILITIES
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
