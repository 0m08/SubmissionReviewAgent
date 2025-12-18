"""
Graphics Image Viewer - Populate images from Graphics Definition V2 into sheet
"""

import streamlit as st
import re
import pandas as pd
import gspread.exceptions
import time
from typing import List, Dict, Tuple


def parse_graphics_definition(text: str) -> list:
    """
    Parse the Graphics Definition V2 output to extract segments with VO and image URLs.
    
    Returns:
        List of dicts: [{"vo": "...", "images": [{"title": "...", "url": "..."}]}]
    """
    segments = []
    
    # Split by segment separator
    segment_blocks = re.split(r'={50,}', text)
    
    for block in segment_blocks:
        block = block.strip()
        if not block:
            continue
            
        # Check if this is a segment header
        if block.startswith("SEGMENT"):
            continue
            
        # Look for VO line
        vo_match = re.search(r'VO:\s*"([^"]+)"', block)
        if vo_match:
            current_segment = {
                "vo": vo_match.group(1),
                "images": []
            }
            
            
            if "Images to use for this segment:" in block:
                images_section = block.split("Images to use for this segment:")[1]
                
                # Parse each line individually since format is: "1. title | url" (one per line)
                # Use a pattern that matches: number. title | url (everything after | to end of line)
                image_pattern = r'^(\d+)\.\s+(.+?)\s+\|\s+(.+)$'
                
                for line in images_section.split('\n'):
                    line = line.strip()
                    if not line:
                        continue
                    
                    match = re.match(image_pattern, line)
                    if match:
                        # Strip brackets from title if present
                        title = match.group(2).strip()
                        if title.startswith('[') and title.endswith(']'):
                            title = title[1:-1].strip()
                        
                        # Strip brackets from URL if present, and strip any trailing whitespace
                        url = match.group(3).strip()
                        if url.startswith('[') and url.endswith(']'):
                            url = url[1:-1].strip()
                        
                        # Validate URL starts with http
                        if url and url.startswith('http'):
                            current_segment["images"].append({
                                "index": match.group(1),
                                "title": title,
                                "url": url
                            })
            
            segments.append(current_segment)
    
    return segments


def extract_file_id_from_url(url: str) -> str:
    """Extract Google Drive file ID from URL."""
    match = re.search(r'/d/([a-zA-Z0-9_-]+)', url)
    if match:
        return match.group(1)
    return None


def is_drive_url(url: str) -> bool:
    """Check if URL is a Google Drive URL."""
    return bool(re.search(r'drive\.google\.com', url, re.IGNORECASE))


def get_image_formula(file_id: str = None, web_url: str = None) -> str:
    """
    Create Google Sheets IMAGE formula for a Drive file or web URL.
    
    Args:
        file_id: Google Drive file ID (for Drive images)
        web_url: Direct web URL (for web images)
    
    Returns:
        Google Sheets IMAGE formula string
    """
    if file_id:
        # Drive image: use thumbnail URL
        thumbnail_url = f"https://drive.google.com/thumbnail?id={file_id}&sz=w400"
        return f'=IMAGE("{thumbnail_url}")'
    elif web_url:
        # Web image: use direct URL
        return f'=IMAGE("{web_url}")'
    else:
        raise ValueError("Either file_id or web_url must be provided")


def main():
    st.title("🖼️ Graphics Image Populator")
    st.markdown("Populate images from Graphics Definition V2 into the sheet")
    
    # Input: Google Sheet URL
    sheet_url = st.text_input(
        "Google Sheet URL",
        placeholder="https://docs.google.com/spreadsheets/d/..."
    )
    
    if not sheet_url:
        st.info("Enter a Google Sheet URL to load graphics definitions")
        return
    
    # Check for authenticated session
    gc = st.session_state.get("gc")
    if not gc:
        st.error("Please log in first to access Google Sheets")
        return
    
    # Load sheet - always use "Slide Chunks" worksheet
    try:
        sheet = gc.open_by_url(sheet_url)
        worksheet = sheet.worksheet("Slide Chunks")
        data = worksheet.get_all_records()
        if not data:
            st.warning("Worksheet is empty or has no data rows")
            return
        df = pd.DataFrame(data)
        
    except gspread.exceptions.APIError as e:
        st.error(f"Google Sheets API error: {e}")
        return
    except gspread.exceptions.SpreadsheetNotFound:
        st.error("Spreadsheet not found. Check the URL and make sure you have access.")
        return
    except Exception as e:
        st.error(f"Failed to load sheet: {e}")
        return
    
    # Check for Graphics Definition V2 column
    gd_column = None
    for col in df.columns:
        if "graphics definition v2" in col.lower():
            gd_column = col
            break
    
    if not gd_column:
        st.error("No 'Graphics Definition V2' column found in sheet")
        st.write("Available columns:", list(df.columns))
        return
    
    st.success(f"✓ Found {gd_column} column with {len(df)} rows")
    
    # Run button
    if st.button("🚀 Populate Images", type="primary"):
        with st.spinner("Processing..."):
            populate_images(worksheet, df, gd_column)


def retry_with_backoff(func, max_retries=5, initial_delay=1):
    """Retry a function with exponential backoff, handling rate limit errors."""
    for attempt in range(max_retries):
        try:
            return func()
        except gspread.exceptions.APIError as e:
            if e.response.status_code == 429:  # Rate limit
                if attempt < max_retries - 1:
                    delay = initial_delay * (2 ** attempt)
                    time.sleep(delay)
                    continue
            raise
        except Exception as e:
            raise


def populate_images(worksheet, df, gd_column):
    """Populate image formulas into the worksheet with VO text as column headers."""
    
    progress = st.progress(0, text="Processing rows...")
    
    # Get existing headers to find where to start adding columns
    existing_headers = worksheet.row_values(1)
    start_col = len(existing_headers) + 1
    
    # Get current sheet dimensions
    current_col_count = worksheet.col_count
    
    # Calculate maximum columns we might need
    # We need to scan all rows first to determine max columns needed
    max_col_offset = 0
    for idx, row in df.iterrows():
        gd_content = row[gd_column]
        if not gd_content or not str(gd_content).strip():
            continue
        
        segments = parse_graphics_definition(str(gd_content))
        if not segments:
            continue
        
        col_offset = 0
        for seg in segments:
            images = seg["images"]
            col_offset += len(images)
        max_col_offset = max(max_col_offset, col_offset)
    
    # Calculate total columns needed
    # max_col_offset is the total number of image columns we'll write
    # We write to columns: start_col, start_col+1, ..., start_col+max_col_offset-1
    # So the last column is start_col + max_col_offset - 1
    if max_col_offset > 0:
        total_cols_needed = start_col + max_col_offset - 1
    else:
        total_cols_needed = current_col_count  # No images to write, no columns needed
    
    # Add columns if needed (with a small buffer for safety)
    if total_cols_needed > current_col_count:
        cols_to_add = total_cols_needed - current_col_count + 2  # +2 buffer
        retry_with_backoff(lambda: worksheet.add_cols(cols_to_add))
        st.info(f"Added {cols_to_add} columns to accommodate images (needed {total_cols_needed - current_col_count}, added buffer)")
    
    # Collect all updates to batch them
    cell_updates: List[Tuple[int, int, str]] = []  # (row, col, value)
    note_updates: List[Dict] = []  # For batch note updates
    
    rows_processed = 0
    total_columns_created = 0
    
    progress.progress(0.1, text="Collecting image data...")
    
    for idx, row in df.iterrows():
        gd_content = row[gd_column]
        if not gd_content or not str(gd_content).strip():
            continue
        
        segments = parse_graphics_definition(str(gd_content))
        if not segments:
            continue
        
        row_num = idx + 2  # +1 for 0-index, +1 for header row
        col_offset = 0
        
        for seg in segments:
            vo_text = seg["vo"]
            
            # Get images for this segment
            images = seg["images"]
            
            for img_idx, img in enumerate(images):
                url = img["url"]
                if not url or not url.startswith('http'):
                    continue  # Skip invalid URLs
                
                col_num = start_col + col_offset
                
                # Set header as "Image 1", "Image 2", etc. (only on first row)
                if idx == 0:
                    cell_updates.append((1, col_num, f"Image {col_offset + 1}"))
                
                # Determine if it's a Drive URL or web URL and get appropriate formula
                if is_drive_url(url):
                    # Google Drive image: extract file ID and use Drive thumbnail
                    file_id = extract_file_id_from_url(url)
                    if file_id:
                        cell_updates.append((row_num, col_num, get_image_formula(file_id=file_id)))
                    else:
                        # Fallback: try direct URL if file ID extraction fails
                        cell_updates.append((row_num, col_num, get_image_formula(web_url=url)))
                else:
                    # Web image: use direct URL
                    cell_updates.append((row_num, col_num, get_image_formula(web_url=url)))
                
                # Collect note update (using 0-based indices for API)
                note_updates.append({
                    "row": row_num - 1,  # Convert to 0-based
                    "col": col_num - 1,  # Convert to 0-based
                    "note": f"VO: {vo_text}"
                })
                
                col_offset += 1
                total_columns_created = max(total_columns_created, col_offset)
        
        rows_processed += 1
    
    # Batch update cells using batch_update API for maximum efficiency
    # Process in chunks to stay under rate limits (60 writes per minute)
    progress.progress(0.3, text=f"Updating {len(cell_updates)} cells in batches...")
    
    # Group all updates by row
    updates_by_row: Dict[int, List[Tuple[int, str]]] = {}
    for row, col, value in cell_updates:
        if row not in updates_by_row:
            updates_by_row[row] = []
        updates_by_row[row].append((col, value))
    
    # Process in batches - each batch is one API call
    rows_list = sorted(updates_by_row.keys())
    batch_size = 50  # Process up to 50 rows per batch (one API call)
    
    for batch_start in range(0, len(rows_list), batch_size):
        batch_rows = rows_list[batch_start:batch_start + batch_size]
        progress.progress(0.3 + 0.4 * (batch_start / len(rows_list)), 
                         text=f"Updating rows {batch_start+1}-{min(batch_start+batch_size, len(rows_list))} of {len(rows_list)}...")
        
        # Build batch update requests for this batch
        requests = []
        for row in batch_rows:
            col_values = sorted(updates_by_row[row])  # Sort by column
            
            # Group contiguous columns into ranges
            if not col_values:
                continue
            
            # Find contiguous ranges
            ranges = []
            current_range_start = col_values[0][0]
            current_range_end = col_values[0][0]
            current_range_values = [col_values[0][1]]
            
            for i in range(1, len(col_values)):
                col, value = col_values[i]
                if col == current_range_end + 1:
                    # Contiguous, extend range
                    current_range_end = col
                    current_range_values.append(value)
                else:
                    # Gap found, save current range and start new one
                    ranges.append((current_range_start, current_range_end, current_range_values))
                    current_range_start = col
                    current_range_end = col
                    current_range_values = [value]
            
            # Add last range
            ranges.append((current_range_start, current_range_end, current_range_values))
            
            # Create updateCells request for each contiguous range
            for start_col, end_col, values in ranges:
                cell_data = []
                for value in values:
                    # Determine if it's a formula or string
                    if value.startswith("="):
                        cell_data.append({
                            "userEnteredValue": {"formulaValue": value}
                        })
                    else:
                        cell_data.append({
                            "userEnteredValue": {"stringValue": value}
                        })
                
                requests.append({
                    "updateCells": {
                        "range": {
                            "sheetId": worksheet.id,
                            "startRowIndex": row - 1,  # Convert to 0-based
                            "endRowIndex": row,  # Exclusive
                            "startColumnIndex": start_col - 1,  # Convert to 0-based
                            "endColumnIndex": end_col  # Convert to 0-based (exclusive)
                        },
                        "rows": [{
                            "values": cell_data
                        }],
                        "fields": "userEnteredValue"
                    }
                })
        
        # Execute batch update (one API call for all rows in batch)
        def update_batch():
            if requests:
                worksheet.spreadsheet.batch_update({"requests": requests})
        
        retry_with_backoff(update_batch)
        
        # Delay between batches to avoid rate limits (60 per minute = 1 per second)
        # We use 1.5 seconds to be safe
        if batch_start + batch_size < len(rows_list):
            time.sleep(1.5)
    
    # Batch update notes using batch_update API
    if note_updates:
        progress.progress(0.8, text=f"Adding {len(note_updates)} notes...")
        
        # Process notes in chunks to stay under rate limits
        note_chunk_size = 50  # Conservative chunk size
        for i in range(0, len(note_updates), note_chunk_size):
            chunk = note_updates[i:i + note_chunk_size]
            progress.progress(0.8 + 0.05 * (i / len(note_updates)), 
                             text=f"Adding notes {i+1}-{min(i+note_chunk_size, len(note_updates))} of {len(note_updates)}...")
            
            requests = []
            for note_update in chunk:
                requests.append({
                    "updateCells": {
                        "range": {
                            "sheetId": worksheet.id,
                            "startRowIndex": note_update["row"],
                            "endRowIndex": note_update["row"] + 1,
                            "startColumnIndex": note_update["col"],
                            "endColumnIndex": note_update["col"] + 1
                        },
                        "rows": [{
                            "values": [{
                                "note": note_update["note"]
                            }]
                        }],
                        "fields": "note"
                    }
                })
            
            def update_notes():
                if requests:
                    worksheet.spreadsheet.batch_update({"requests": requests})
            
            retry_with_backoff(update_notes)
            
            # Delay between chunks to avoid rate limits
            if i + note_chunk_size < len(note_updates):
                time.sleep(1.5)
    
    # Set row heights and column widths for better visibility
    progress.progress(0.9, text="Adjusting row heights and column widths...")
    try:
        # Set row heights for data rows (200 pixels for images)
        requests = []
        for idx in range(len(df)):
            requests.append({
                "updateDimensionProperties": {
                    "range": {
                        "sheetId": worksheet.id,
                        "dimension": "ROWS",
                        "startIndex": idx + 1,  # Skip header row
                        "endIndex": idx + 2
                    },
                    "properties": {"pixelSize": 200},
                    "fields": "pixelSize"
                }
            })
        
        # Set column widths for image columns (200 pixels)
        for col_idx in range(total_columns_created):
            requests.append({
                "updateDimensionProperties": {
                    "range": {
                        "sheetId": worksheet.id,
                        "dimension": "COLUMNS",
                        "startIndex": start_col - 1 + col_idx,
                        "endIndex": start_col + col_idx
                    },
                    "properties": {"pixelSize": 200},
                    "fields": "pixelSize"
                }
            })
        
        if requests:
            def update_dimensions():
                worksheet.spreadsheet.batch_update({"requests": requests})
            retry_with_backoff(update_dimensions)
    except Exception as e:
        st.warning(f"Could not adjust row/column sizes: {e}")
    
    progress.progress(1.0, text="Done!")
    st.success(f"✅ Populated images for {rows_processed} rows")
    st.balloons()


def col_letter(col_num):
    """Convert column number to letter (1=A, 2=B, ..., 27=AA, etc.)"""
    result = ""
    while col_num > 0:
        col_num, remainder = divmod(col_num - 1, 26)
        result = chr(65 + remainder) + result
    return result


# Run the main function
main()



# New Format

# """
# Graphics Image Viewer - Populate images from Graphics Definition V2 into sheet
# """

# import streamlit as st
# import re
# import pandas as pd
# import gspread.exceptions
# import time
# from typing import List, Dict, Tuple


# def parse_graphics_definition(text: str) -> list:
#     """
#     Parse the Graphics Definition V2 output (NEW FORMAT) to extract segments with VO and image URLs.
    
#     New Format Structure:
#     ================================================================================
#     SEGMENT 1
#     ================================================================================
#     VO: "..."
#     VISUALS:
#     ...
#     ACTION:
#     ...
#     IMAGES:
#     [1] Image Title
#         https://drive.google.com/file/d/.../view
#     [2] Another Title
#         https://drive.google.com/file/d/.../view
    
#     Returns:
#         List of dicts: [{"vo": "...", "images": [{"title": "...", "url": "..."}]}]
#     """
#     segments = []
    
#     # Split by segment separator (80 equals signs)
#     segment_blocks = re.split(r'={80,}', text)
    
#     for block in segment_blocks:
#         block = block.strip()
#         if not block:
#             continue
            
#         # Skip if this is just a segment header line
#         if block.startswith("SEGMENT") and len(block.split('\n')) <= 2:
#             continue
        
#         # Look for VO line: VO: "text"
#         vo_match = re.search(r'VO:\s*"([^"]+)"', block)
#         if not vo_match:
#             continue
        
#         current_segment = {
#             "vo": vo_match.group(1),
#             "images": []
#         }
        
#         # Look for IMAGES section
#         if "IMAGES:" in block:
#             images_section = block.split("IMAGES:")[1]
            
#             # Pattern to match: [N] Title (on one line) followed by URL (on next line, may be indented)
#             # Format: [1] Title\n    https://...
#             # The title is everything after [N] until the newline (same line)
#             # The URL is on the next line(s), possibly indented
#             image_pattern = r'\[(\d+)\]\s*([^\n]+?)\n+\s*(https?://[^\s\n]+)'
#             matches = re.finditer(image_pattern, images_section, re.MULTILINE)
            
#             for match in matches:
#                 index = match.group(1)
#                 title = match.group(2).strip()
#                 url = match.group(3).strip()
                
#                 # Validate URL starts with http
#                 if url and url.startswith('http'):
#                     current_segment["images"].append({
#                         "index": index,
#                         "title": title,
#                         "url": url
#                     })
        
#         segments.append(current_segment)
    
#     return segments


# def extract_file_id_from_url(url: str) -> str:
#     """Extract Google Drive file ID from URL."""
#     match = re.search(r'/d/([a-zA-Z0-9_-]+)', url)
#     if match:
#         return match.group(1)
#     return None


# def get_image_formula(file_id: str) -> str:
#     """Create Google Sheets IMAGE formula for a Drive file."""
#     thumbnail_url = f"https://drive.google.com/thumbnail?id={file_id}&sz=w400"
#     return f'=IMAGE("{thumbnail_url}")'


# def main():
#     st.title("🖼️ Graphics Image Populator")
#     st.markdown("Populate images from Graphics Definition V2 into the sheet")
    
#     # Input: Google Sheet URL
#     sheet_url = st.text_input(
#         "Google Sheet URL",
#         placeholder="https://docs.google.com/spreadsheets/d/..."
#     )
    
#     if not sheet_url:
#         st.info("Enter a Google Sheet URL to load graphics definitions")
#         return
    
#     # Check for authenticated session
#     gc = st.session_state.get("gc")
#     if not gc:
#         st.error("Please log in first to access Google Sheets")
#         return
    
#     # Load sheet - always use "Slide Chunks" worksheet
#     try:
#         sheet = gc.open_by_url(sheet_url)
#         worksheet = sheet.worksheet("Slide Chunks")
#         data = worksheet.get_all_records()
#         if not data:
#             st.warning("Worksheet is empty or has no data rows")
#             return
#         df = pd.DataFrame(data)
        
#     except gspread.exceptions.APIError as e:
#         st.error(f"Google Sheets API error: {e}")
#         return
#     except gspread.exceptions.SpreadsheetNotFound:
#         st.error("Spreadsheet not found. Check the URL and make sure you have access.")
#         return
#     except Exception as e:
#         st.error(f"Failed to load sheet: {e}")
#         return
    
#     # Check for Graphics Definition V2 column
#     gd_column = None
#     for col in df.columns:
#         if "graphics definition v2" in col.lower():
#             gd_column = col
#             break
    
#     if not gd_column:
#         st.error("No 'Graphics Definition V2' column found in sheet")
#         st.write("Available columns:", list(df.columns))
#         return
    
#     st.success(f"✓ Found {gd_column} column with {len(df)} rows")
    
#     # Run button
#     if st.button("🚀 Populate Images", type="primary"):
#         with st.spinner("Processing..."):
#             populate_images(worksheet, df, gd_column)


# def retry_with_backoff(func, max_retries=5, initial_delay=1):
#     """Retry a function with exponential backoff, handling rate limit errors."""
#     for attempt in range(max_retries):
#         try:
#             return func()
#         except gspread.exceptions.APIError as e:
#             if e.response.status_code == 429:  # Rate limit
#                 if attempt < max_retries - 1:
#                     delay = initial_delay * (2 ** attempt)
#                     time.sleep(delay)
#                     continue
#             raise
#         except Exception as e:
#             raise


# def populate_images(worksheet, df, gd_column):
#     """Populate image formulas into the worksheet with VO text as column headers."""
    
#     progress = st.progress(0, text="Processing rows...")
    
#     # Get existing headers to find where to start adding columns
#     existing_headers = worksheet.row_values(1)
#     start_col = len(existing_headers) + 1
    
#     # Get current sheet dimensions
#     current_col_count = worksheet.col_count
    
#     # Calculate maximum columns we might need
#     # We need to scan all rows first to determine max columns needed
#     max_col_offset = 0
#     for idx, row in df.iterrows():
#         gd_content = row[gd_column]
#         if not gd_content or not str(gd_content).strip():
#             continue
        
#         segments = parse_graphics_definition(str(gd_content))
#         if not segments:
#             continue
        
#         col_offset = 0
#         for seg in segments:
#             images = seg["images"]
#             col_offset += len(images)
#         max_col_offset = max(max_col_offset, col_offset)
    
#     # Calculate total columns needed
#     # max_col_offset is the total number of image columns we'll write
#     # We write to columns: start_col, start_col+1, ..., start_col+max_col_offset-1
#     # So the last column is start_col + max_col_offset - 1
#     if max_col_offset > 0:
#         total_cols_needed = start_col + max_col_offset - 1
#     else:
#         total_cols_needed = current_col_count  # No images to write, no columns needed
    
#     # Add columns if needed (with a small buffer for safety)
#     if total_cols_needed > current_col_count:
#         cols_to_add = total_cols_needed - current_col_count + 2  # +2 buffer
#         retry_with_backoff(lambda: worksheet.add_cols(cols_to_add))
#         st.info(f"Added {cols_to_add} columns to accommodate images (needed {total_cols_needed - current_col_count}, added buffer)")
    
#     # Collect all updates to batch them
#     cell_updates: List[Tuple[int, int, str]] = []  # (row, col, value)
#     note_updates: List[Dict] = []  # For batch note updates
    
#     rows_processed = 0
#     total_columns_created = 0
    
#     progress.progress(0.1, text="Collecting image data...")
    
#     for idx, row in df.iterrows():
#         gd_content = row[gd_column]
#         if not gd_content or not str(gd_content).strip():
#             continue
        
#         segments = parse_graphics_definition(str(gd_content))
#         if not segments:
#             continue
        
#         row_num = idx + 2  # +1 for 0-index, +1 for header row
#         col_offset = 0
        
#         for seg in segments:
#             vo_text = seg["vo"]
            
#             # Get images for this segment
#             images = seg["images"]
            
#             for img_idx, img in enumerate(images):
#                 file_id = extract_file_id_from_url(img["url"])
#                 if file_id:
#                     col_num = start_col + col_offset
                    
#                     # Set header as "Image 1", "Image 2", etc. (only on first row)
#                     if idx == 0:
#                         cell_updates.append((1, col_num, f"Image {col_offset + 1}"))
                    
#                     # Set image formula
#                     cell_updates.append((row_num, col_num, get_image_formula(file_id)))
                    
#                     # Collect note update (using 0-based indices for API)
#                     note_updates.append({
#                         "row": row_num - 1,  # Convert to 0-based
#                         "col": col_num - 1,  # Convert to 0-based
#                         "note": f"VO: {vo_text}"
#                     })
                    
#                     col_offset += 1
#                     total_columns_created = max(total_columns_created, col_offset)
        
#         rows_processed += 1
    
#     # Batch update cells using batch_update API for maximum efficiency
#     # Process in chunks to stay under rate limits (60 writes per minute)
#     progress.progress(0.3, text=f"Updating {len(cell_updates)} cells in batches...")
    
#     # Group all updates by row
#     updates_by_row: Dict[int, List[Tuple[int, str]]] = {}
#     for row, col, value in cell_updates:
#         if row not in updates_by_row:
#             updates_by_row[row] = []
#         updates_by_row[row].append((col, value))
    
#     # Process in batches - each batch is one API call
#     rows_list = sorted(updates_by_row.keys())
#     batch_size = 50  # Process up to 50 rows per batch (one API call)
    
#     for batch_start in range(0, len(rows_list), batch_size):
#         batch_rows = rows_list[batch_start:batch_start + batch_size]
#         progress.progress(0.3 + 0.4 * (batch_start / len(rows_list)), 
#                          text=f"Updating rows {batch_start+1}-{min(batch_start+batch_size, len(rows_list))} of {len(rows_list)}...")
        
#         # Build batch update requests for this batch
#         requests = []
#         for row in batch_rows:
#             col_values = sorted(updates_by_row[row])  # Sort by column
            
#             # Group contiguous columns into ranges
#             if not col_values:
#                 continue
            
#             # Find contiguous ranges
#             ranges = []
#             current_range_start = col_values[0][0]
#             current_range_end = col_values[0][0]
#             current_range_values = [col_values[0][1]]
            
#             for i in range(1, len(col_values)):
#                 col, value = col_values[i]
#                 if col == current_range_end + 1:
#                     # Contiguous, extend range
#                     current_range_end = col
#                     current_range_values.append(value)
#                 else:
#                     # Gap found, save current range and start new one
#                     ranges.append((current_range_start, current_range_end, current_range_values))
#                     current_range_start = col
#                     current_range_end = col
#                     current_range_values = [value]
            
#             # Add last range
#             ranges.append((current_range_start, current_range_end, current_range_values))
            
#             # Create updateCells request for each contiguous range
#             for start_col, end_col, values in ranges:
#                 cell_data = []
#                 for value in values:
#                     # Determine if it's a formula or string
#                     if value.startswith("="):
#                         cell_data.append({
#                             "userEnteredValue": {"formulaValue": value}
#                         })
#                     else:
#                         cell_data.append({
#                             "userEnteredValue": {"stringValue": value}
#                         })
                
#                 requests.append({
#                     "updateCells": {
#                         "range": {
#                             "sheetId": worksheet.id,
#                             "startRowIndex": row - 1,  # Convert to 0-based
#                             "endRowIndex": row,  # Exclusive
#                             "startColumnIndex": start_col - 1,  # Convert to 0-based
#                             "endColumnIndex": end_col  # Convert to 0-based (exclusive)
#                         },
#                         "rows": [{
#                             "values": cell_data
#                         }],
#                         "fields": "userEnteredValue"
#                     }
#                 })
        
#         # Execute batch update (one API call for all rows in batch)
#         def update_batch():
#             if requests:
#                 worksheet.spreadsheet.batch_update({"requests": requests})
        
#         retry_with_backoff(update_batch)
        
#         # Delay between batches to avoid rate limits (60 per minute = 1 per second)
#         # We use 1.5 seconds to be safe
#         if batch_start + batch_size < len(rows_list):
#             time.sleep(1.5)
    
#     # Batch update notes using batch_update API
#     if note_updates:
#         progress.progress(0.8, text=f"Adding {len(note_updates)} notes...")
        
#         # Process notes in chunks to stay under rate limits
#         note_chunk_size = 50  # Conservative chunk size
#         for i in range(0, len(note_updates), note_chunk_size):
#             chunk = note_updates[i:i + note_chunk_size]
#             progress.progress(0.8 + 0.05 * (i / len(note_updates)), 
#                              text=f"Adding notes {i+1}-{min(i+note_chunk_size, len(note_updates))} of {len(note_updates)}...")
            
#             requests = []
#             for note_update in chunk:
#                 requests.append({
#                     "updateCells": {
#                         "range": {
#                             "sheetId": worksheet.id,
#                             "startRowIndex": note_update["row"],
#                             "endRowIndex": note_update["row"] + 1,
#                             "startColumnIndex": note_update["col"],
#                             "endColumnIndex": note_update["col"] + 1
#                         },
#                         "rows": [{
#                             "values": [{
#                                 "note": note_update["note"]
#                             }]
#                         }],
#                         "fields": "note"
#                     }
#                 })
            
#             def update_notes():
#                 if requests:
#                     worksheet.spreadsheet.batch_update({"requests": requests})
            
#             retry_with_backoff(update_notes)
            
#             # Delay between chunks to avoid rate limits
#             if i + note_chunk_size < len(note_updates):
#                 time.sleep(1.5)
    
#     # Set row heights and column widths for better visibility
#     progress.progress(0.9, text="Adjusting row heights and column widths...")
#     try:
#         # Set row heights for data rows (200 pixels for images)
#         requests = []
#         for idx in range(len(df)):
#             requests.append({
#                 "updateDimensionProperties": {
#                     "range": {
#                         "sheetId": worksheet.id,
#                         "dimension": "ROWS",
#                         "startIndex": idx + 1,  # Skip header row
#                         "endIndex": idx + 2
#                     },
#                     "properties": {"pixelSize": 200},
#                     "fields": "pixelSize"
#                 }
#             })
        
#         # Set column widths for image columns (200 pixels)
#         for col_idx in range(total_columns_created):
#             requests.append({
#                 "updateDimensionProperties": {
#                     "range": {
#                         "sheetId": worksheet.id,
#                         "dimension": "COLUMNS",
#                         "startIndex": start_col - 1 + col_idx,
#                         "endIndex": start_col + col_idx
#                     },
#                     "properties": {"pixelSize": 200},
#                     "fields": "pixelSize"
#                 }
#             })
        
#         if requests:
#             def update_dimensions():
#                 worksheet.spreadsheet.batch_update({"requests": requests})
#             retry_with_backoff(update_dimensions)
#     except Exception as e:
#         st.warning(f"Could not adjust row/column sizes: {e}")
    
#     progress.progress(1.0, text="Done!")
#     st.success(f"✅ Populated images for {rows_processed} rows")
#     st.balloons()


# def col_letter(col_num):
#     """Convert column number to letter (1=A, 2=B, ..., 27=AA, etc.)"""
#     result = ""
#     while col_num > 0:
#         col_num, remainder = divmod(col_num - 1, 26)
#         result = chr(65 + remainder) + result
#     return result


# # Run the main function
# # main()
