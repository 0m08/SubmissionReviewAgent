import os
import re
import threading
import traceback
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from PIL import Image

from agents.graphics_asset_creation.generator.image_generator import run_analysis_stage, run_generation_stage
from agents.graphics_asset_creation.gac_utils import get_or_create_drive_folder, upload_image_to_drive

def _write_back_to_sheet(ws, data_df):
    """Refactored helper for writing DataFrame to Google Sheet."""
    try:
        temp_df = data_df.astype(str).replace("nan", "")
        ws.update([temp_df.columns.values.tolist()] + temp_df.values.tolist())
    except Exception as e:
        print(f"[Generation Automation] ❌ Error writing to sheet: {e}")
        traceback.print_exc()

def run_generation_automation(
    sheet_url: str,
    source_tab: str,
    output_folder_name: str,
    gc,
    drive,
    progress_callback=None
):
    print(f"\n[Generation Automation] Initializing...")
    
    # 1. Open Sheet and get DataFrame
    try:
        sheet = gc.open_by_url(sheet_url)
        worksheet = sheet.worksheet(source_tab)
        raw_rows = worksheet.get_all_values(value_render_option='FORMULA')
        if not raw_rows:
            raise RuntimeError(f"Sheet '{source_tab}' is empty.")
            
        headers = [h.strip() for h in raw_rows[0]]
        data = raw_rows[1:]
        df = pd.DataFrame(data, columns=headers)
    except Exception as e:
        raise RuntimeError(f"Failed to open Google Sheet: {e}")
        
    print(f"[Generation Automation] Loaded {len(df)} rows from '{source_tab}'.")

    # 2. Normalize and ensure columns exist
    required_cols = ["Slide Chunk Title", "Slide Chunk", "Voiceover"]
    for col in required_cols:
        if col not in df.columns:
            for actual in df.columns:
                if actual.lower() == col.lower():
                    df.rename(columns={actual: col}, inplace=True)
                    break
    
    # Ensure worksheet headers are synced for "on the spot" updates
    ws_headers = headers.copy()
    out_cols = ["Generated Brief", "Generated Asset Image"]
    headers_modified = False
    for oc in out_cols:
        if oc not in df.columns:
            df[oc] = ""
        if oc not in ws_headers:
            ws_headers.append(oc)
            headers_modified = True
    
    if headers_modified:
        print("[Generation Automation] Updating worksheet headers...")
        worksheet.update([ws_headers])
    
    # Map column indices for row-level updates (1-based)
    brief_idx = ws_headers.index("Generated Brief") + 1
    image_idx = ws_headers.index("Generated Asset Image") + 1

    work_items = []
    for idx, row in df.iterrows():
        vo = str(row.get("Voiceover", "")).strip()
        if not vo or vo.lower() == "nan":
            continue
        if str(row.get("Generated Asset Image", "")).strip():
            continue
        work_items.append({
            "idx": idx,
            "title": str(row.get("Slide Chunk Title", "")),
            "chunk": str(row.get("Slide Chunk", "")),
            "voiceover": vo,
        })

    total_tasks = len(work_items)
    if total_tasks == 0:
        print("[Generation Automation] No valid unprocessed rows found.")
        if progress_callback: progress_callback(0, 0)
        return

    output_folder_id = get_or_create_drive_folder(drive, output_folder_name)
    if not output_folder_id:
        raise RuntimeError(
            f"Could not create or find output Drive folder: '{output_folder_name}'"
        )

    print(f"[Generation Automation] Found {total_tasks} rows to process.")
    drive_lock = threading.Lock()
    
    def process_row(item):
        idx = item["idx"]
        title = item["title"]
        chunk = item["chunk"]
        vo = item["voiceover"]
        thread_name = threading.current_thread().name
        print(f"[{thread_name}] ▶ Generating Asset for Row {idx + 2}")
        
        brief_json = ""
        final_image_formula = ""
        
        try:
            # Stage 1: Analysis
            history_s1 = []
            brief_json = run_analysis_stage(
                slide_title=title,
                slide_content=chunk,
                voiceover_focus=vo,
                revision_history=history_s1
            )
            
            # Stage 2: Generation
            history_s2 = []
            generated_image, _ = run_generation_stage(
                approved_json_brief=brief_json,
                slide_title=title,
                slide_content=chunk,
                voiceover_focus=vo,
                aspect_ratio="16:9",
                image_size="1K",
                revision_history=history_s2
            )
            
            safe_title = re.sub(r'[^\w\s-]', '', title).strip().replace(' ', '_')
            if not safe_title:
                safe_title = f"Row_{idx + 2}"
            filename = f"{safe_title[:50]}_{idx + 2}.png"

            # Upload to Drive under the configured output folder.
            with drive_lock:
                img_url = upload_image_to_drive(
                    generated_image,
                    filename,
                    drive,
                    output_folder_id
                )

            if img_url:
                final_image_formula = f'=IMAGE("{img_url}")'
            else:
                # Fallback: store a note that upload failed.
                final_image_formula = "IMAGE_UPLOAD_FAILED"
            
            print(f"[{thread_name}] ✅ Row {idx + 2} Completed!")
            
        except Exception as e:
            print(f"[{thread_name}] ❌ Row {idx + 2} Error: {e}")
            traceback.print_exc()
            brief_json = f"ERROR: {str(e)}"
            
        return idx, brief_json, final_image_formula

    completed = 0
    sheet_lock = threading.Lock()
    
    # Use max_workers=5 to avoid hitting Sheets API rate limits too aggressively
    with ThreadPoolExecutor(max_workers=min(5, total_tasks)) as executor:
        futures = {executor.submit(process_row, item): item for item in work_items}
        for future in as_completed(futures):
            idx, brief, formula = future.result()
            
            # 1. Update local DF
            df.at[idx, "Generated Brief"] = brief
            df.at[idx, "Generated Asset Image"] = formula
            
            # 2. Update Worksheet "on the spot"
            row_num = idx + 2
            try:
                with sheet_lock:
                    # Update each cell immediately (USER_ENTERED is implicit in update_cell for some apps, 
                    # but we'll use basic update_cell for compatibility)
                    worksheet.update_cell(row_num, brief_idx, brief)
                    worksheet.update_cell(row_num, image_idx, formula)
                    time.sleep(0.5) # Safety throttle to prevent rate limit bursts
            except Exception as e:
                print(f"[Generation Automation] ⚠️ Failed on-the-spot update for Row {row_num}: {e}")

            completed += 1
            if progress_callback: progress_callback(completed, total_tasks)

    print(f"\n[Generation Automation] Batch processing complete. All results saved to sheet.")
    print(f"[Generation Automation] Finished!")
