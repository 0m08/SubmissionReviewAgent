import os
import re
import threading
import traceback
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from PIL import Image
import json

from agents.graphics_asset_creation.automated.automated_voiceover_reviewer import (
    _download_drive_image, _save_image_to_drive
)
from agents.graphics_asset_creation.illustrator.illustrator_agent import illustrator_agent
from agents.graphics_asset_creation.image_editing.image_editing import image_editing_with_review_loop
from services.sheets_service import get_sheet_data_and_df
from agents.graphics_asset_creation.gac_utils import get_or_create_drive_folder

def _write_back_to_sheet(ws, data_df):
    """Refactored helper for writing DataFrame to Google Sheet."""
    try:
        temp_df = data_df.astype(str).replace("nan", "")
        # For updates to work smoothly across columns, we must include the header row
        ws.update([temp_df.columns.values.tolist()] + temp_df.values.tolist())
    except Exception as e:
        print(f"[Illustrator Automation] ❌ Error writing to sheet: {e}")
        traceback.print_exc()

def run_illustrator_automation(
    sheet_url: str,
    source_tab: str,
    output_folder_name: str,
    gc,
    drive,
    progress_callback=None
):
    print(f"\n[Illustrator Automation] Initializing...")
    
    # 1. Open Sheet and get DataFrame
    try:
        sheet = gc.open_by_url(sheet_url)
        worksheet = sheet.worksheet(source_tab)
        # We use value_render_option='FORMULA' to catch =IMAGE() links
        raw_rows = worksheet.get_all_values(value_render_option='FORMULA')
        if not raw_rows:
            raise RuntimeError(f"Sheet '{source_tab}' is empty.")
            
        headers = [h.strip() for h in raw_rows[0]]
        data = raw_rows[1:]
        df = pd.DataFrame(data, columns=headers)
    except Exception as e:
        raise RuntimeError(f"Failed to open Google Sheet: {e}")
        
    print(f"[Illustrator Automation] Loaded {len(df)} rows from '{source_tab}'.")

    # Normalize column names (strip whitespace)
    print(f"[Illustrator Automation] Sheet Columns: {df.columns.tolist()}")

    # Ensure required columns exist
    required_cols = ["Slide Chunk Title", "Slide Chunk", "Voiceover", "Output Image"]
    for col in required_cols:
        if col not in df.columns:
            # Check for close matches or common variations
            found = False
            for actual in df.columns:
                if actual.lower() == col.lower():
                    df.rename(columns={actual: col}, inplace=True)
                    found = True
                    break
            if not found:
                raise KeyError(f"Sheet is missing required column: '{col}'. Found: {df.columns.tolist()}")

    # Add output columns if they don't exist
    cols_added = False
    if "Illustrator Instructions" not in df.columns:
        df["Illustrator Instructions"] = ""
        cols_added = True
    if "Illustrator Final Image" not in df.columns:
        df["Illustrator Final Image"] = ""
        cols_added = True

    # Pre-flight Setup: Find valid rows to process
    work_items = []
    
    # Pre-create root folder serially (Drive API is not thread-safe for folder creation)
    root_folder_id = get_or_create_drive_folder(drive, output_folder_name)
    if not root_folder_id:
        raise RuntimeError(f"Could not create or find root Drive folder: '{output_folder_name}'")

    for idx, row in df.iterrows():
        img_link = str(row.get("Output Image", "")).strip()
        
        # ── Extract Drive ID if wrapped in =IMAGE formula ──
        if img_link.startswith("=IMAGE"):
            # Matches anything between quotes following ?id=
            id_match = re.search(r'[?&]id=([a-zA-Z0-9_-]+)', img_link)
            if id_match:
                # Normalize to a standard view link that _download_drive_image understands
                img_link = f"https://drive.google.com/file/d/{id_match.group(1)}/view"

        # Skip if blank/empty
        if not img_link or img_link.lower() == "nan" or img_link == "":
            continue
            
        # Optional: Skip already processed
        if str(row.get("Illustrator Final Image", "")).strip():
            continue
            
        work_items.append({
            "idx": idx,
            "title": str(row.get("Slide Chunk Title", "")),
            "chunk": str(row.get("Slide Chunk", "")),
            "voiceover": str(row.get("Voiceover", "")),
            "image_link": img_link,
            "root_folder_id": root_folder_id
        })

    total_tasks = len(work_items)
    if total_tasks == 0:
        print("[Illustrator Automation] No valid unprocessed rows found.")
        # If we added columns, write them back anyway
        if cols_added:
            print("[Illustrator Automation] Adding missing output columns to the sheet...")
            _write_back_to_sheet(worksheet, df)
        if progress_callback: progress_callback(0, 0)
        return

    print(f"[Illustrator Automation] Found {total_tasks} rows to process in parallel.")

    drive_lock = threading.Lock()
    
    def process_row(item):
        idx = item["idx"]
        title = item["title"]
        chunk = item["chunk"]
        vo = item["voiceover"]
        img_link = item["image_link"]
        root_fid = item["root_folder_id"]
        
        thread_name = threading.current_thread().name
        print(f"[{thread_name}] ▶ Processing Row {idx + 2}: {title[:40]}")
        
        result_instructions = ""
        final_image_formula = ""
        
        try:
            # Step 1: Download Image
            with drive_lock:
                ref_image = _download_drive_image(img_link, drive=drive)
                
            # Step 2: Illustrator Agent (Analysis & Recommendation)
            ca_result = illustrator_agent(
                image=ref_image,
                voiceover=vo,
                slide_title=title,
                slide_content=chunk,
                visual_instruction=""
            )
            
            # Extract Instructions
            recs = ca_result.recommended_instructions
            instructions_dict = recs.model_dump() if hasattr(recs, "model_dump") else recs
            final_instructions_dict = {
                k: v for k, v in instructions_dict.items() 
                if v and isinstance(v, str) and v.strip()
            }
            
            result_instructions = str(final_instructions_dict)
            
            # Step 3: Image Editing Generation
            edited_image, _, _ = image_editing_with_review_loop(
                reference_image=ref_image,
                editing_instructions=final_instructions_dict,
                quick_mode=True, 
                image_size="1K",
                aspect_ratio="16:9"
            )
            
            # Step 4: Upload to Drive
            filename = f"Row_{idx + 2}_Illustrator_Image.png"
            uploaded_link = _save_image_to_drive(edited_image, filename, drive, root_fid, drive_lock=drive_lock)
            
            if uploaded_link:
                import re
                m = re.search(r'id=([a-zA-Z0-9_-]+)', uploaded_link) or re.search(r'/d/([a-zA-Z0-9_-]+)/', uploaded_link)
                if m:
                    file_id = m.group(1)
                    direct_url = f"https://drive.google.com/uc?export=view&id={file_id}"
                    final_image_formula = f'=IMAGE("{direct_url}")'
                else:
                    final_image_formula = uploaded_link
            
            print(f"[{thread_name}] ✅ Row {idx + 2} Completed!")
            
        except Exception as e:
            print(f"[{thread_name}] ❌ Row {idx + 2} Error: {e}")
            traceback.print_exc()
            result_instructions = f"ERROR: {str(e)}"
            final_image_formula = ""
            
        return idx, result_instructions, final_image_formula

    # Execute in Parallel
    completed = 0
    with ThreadPoolExecutor(max_workers=min(20, total_tasks)) as executor:
        futures = {executor.submit(process_row, item): item for item in work_items}
        
        for future in as_completed(futures):
            idx, instr, formula = future.result()
            
            # Update dataframe logic
            df.at[idx, "Illustrator Instructions"] = instr
            df.at[idx, "Illustrator Final Image"] = formula
            
            completed += 1
            if progress_callback:
                progress_callback(completed, total_tasks)

    # 3. Write Back to Sheet
    print(f"\n[Illustrator Automation] Writing results back to sheet...")
    _write_back_to_sheet(worksheet, df)
    print(f"[Illustrator Automation] Spreadsheet Process Finished!")
