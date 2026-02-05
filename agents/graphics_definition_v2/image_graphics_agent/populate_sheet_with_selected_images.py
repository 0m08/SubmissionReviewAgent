# from langsmith import traceable
# import streamlit as st
# from services.sheets_service import get_sheet_data_and_df, save_to_sheet, format_worksheet
# from services.smart_progress_bar import SmartProgressBar
# from dotenv import load_dotenv
# import re
# import time
# import gspread.exceptions
# from typing import List, Dict, Tuple

# load_dotenv()


# def parse_graphics_definition(text):
#     """
#     Parse the Graphics Definition V2 output to extract segments with VO and image URLs.
    
#     Returns:
#         List of dicts: [{"vo": "...", "images": [{"title": "...", "url": "..."}]}]
#     """
#     segments = []
    
#     # Split by segment separator
#     segment_blocks = re.split(r'={50,}', text)
    
#     for block in segment_blocks:
#         block = block.strip()
#         if not block:
#             continue
            
#         # Check if this is a segment header
#         if block.startswith("SEGMENT"):
#             continue
            
#         # Look for VO line
#         vo_match = re.search(r'VO:\s*"([^"]+)"', block)
#         if vo_match:
#             current_segment = {
#                 "vo": vo_match.group(1),
#                 "images": []
#             }
                       
#             if "Images to use for this segment:" in block:
#                 images_section = block.split("Images to use for this segment:")[1]
                

#                 image_pattern = r'^(\d+)\.\s+(.+?)\s+\|\s+(.+)$'
                
#                 for line in images_section.split('\n'):
#                     line = line.strip()
#                     if not line:
#                         continue
                    
#                     match = re.match(image_pattern, line)
#                     if match:
#                         # Strip brackets from title if present
#                         title = match.group(2).strip()
#                         if title.startswith('[') and title.endswith(']'):
#                             title = title[1:-1].strip()
                        
#                         # Strip brackets from URL if present, and strip any trailing whitespace
#                         url = match.group(3).strip()
#                         if url.startswith('[') and url.endswith(']'):
#                             url = url[1:-1].strip()
                        
#                         # Validate URL starts with http
#                         if url and url.startswith('http'):
#                             current_segment["images"].append({
#                                 "index": match.group(1),
#                                 "title": title,
#                                 "url": url
#                             })
            
#             segments.append(current_segment)
    
#     return segments


# def extract_file_id_from_url(url: str) -> str:
#     """Extract Google Drive file ID from URL."""
#     match = re.search(r'/d/([a-zA-Z0-9_-]+)', url)
#     if match:
#         return match.group(1)
#     return None


# def is_drive_url(url: str) -> bool:
#     """Check if URL is a Google Drive URL."""
#     return bool(re.search(r'drive\.google\.com', url, re.IGNORECASE))


# def get_image_formula(file_id: str = None, web_url: str = None) -> str:
#     """
#     Create Google Sheets IMAGE formula for a Drive file or web URL.
    
#     Args:
#         file_id: Google Drive file ID (for Drive images)
#         web_url: Direct web URL (for web images)
    
#     Returns:
#         Google Sheets IMAGE formula string
#     """
#     if file_id:
#         # Drive image: use thumbnail URL
#         thumbnail_url = f"https://drive.google.com/thumbnail?id={file_id}&sz=w400"
#         return f'=IMAGE("{thumbnail_url}")'
#     elif web_url:
#         # Web image: use direct URL
#         return f'=IMAGE("{web_url}")'
#     else:
#         raise ValueError("Either file_id or web_url must be provided")


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


# @traceable(
#     metadata={
#         "agent_name": "graphics_definition_v2",
#         "step_name": "Populate Sheet with Selected Images",
#         "function_name": "populate_images_for_row",
#         "user_id": st.session_state.get("role", "anonymous"),
#         "user_email": st.session_state.get("user_email", "anonymous")
#     }
# )
# def populate_images_for_row(index, row, worksheet, gd_column, start_col):
#     """
#     Populate images for a single row.
    
#     :param index: Row index in dataframe
#     :param row: Pandas Series with row data
#     :param worksheet: gspread worksheet object
#     :param gd_column: Column name containing graphics definition
#     :param start_col: Starting column number for images
#     :return: Tuple of (index, col_offset) where col_offset is the number of columns used
#     """
#     gd_content = str(row.get(gd_column, "")).strip()
#     if not gd_content or gd_content == "nan":
#         return index, 0
    
#     segments = parse_graphics_definition(gd_content)
#     if not segments:
#         return index, 0
    
#     row_num = index + 2  # +1 for 0-index, +1 for header row
#     col_offset = 0
    
#     # Collect updates for this row
#     cell_updates: List[Tuple[int, int, str]] = []  # (row, col, value)
#     note_updates: List[Dict] = []  # For batch note updates
    
#     for seg in segments:
#         vo_text = seg["vo"]
        
#         # Get images for this segment
#         images = seg["images"]
        
#         for img_idx, img in enumerate(images):
#             url = img["url"]
#             if not url or not url.startswith('http'):
#                 continue  # Skip invalid URLs
            
#             col_num = start_col + col_offset
            
#             # Determine if it's a Drive URL or web URL and get appropriate formula
#             if is_drive_url(url):
#                 # Google Drive image: extract file ID and use Drive thumbnail
#                 file_id = extract_file_id_from_url(url)
#                 if file_id:
#                     cell_updates.append((row_num, col_num, get_image_formula(file_id=file_id)))
#                 else:
#                     # Fallback: try direct URL if file ID extraction fails
#                     cell_updates.append((row_num, col_num, get_image_formula(web_url=url)))
#             else:
#                 # Web image: use direct URL
#                 cell_updates.append((row_num, col_num, get_image_formula(web_url=url)))
            
#             # Collect note update (using 0-based indices for API)
#             note_updates.append({
#                 "row": row_num - 1,  # Convert to 0-based
#                 "col": col_num - 1,  # Convert to 0-based
#                 "note": f"VO: {vo_text}"
#             })
            
#             col_offset += 1
    
#     # Apply updates for this row
#     if cell_updates:
#         # Group by row for batch update
#         updates_by_row: Dict[int, List[Tuple[int, str]]] = {}
#         for row, col, value in cell_updates:
#             if row not in updates_by_row:
#                 updates_by_row[row] = []
#             updates_by_row[row].append((col, value))
        
#         # Process each row
#         for row, col_values in updates_by_row.items():
#             col_values = sorted(col_values)  # Sort by column
            
#             # Find contiguous ranges
#             ranges = []
#             if col_values:
#                 current_range_start = col_values[0][0]
#                 current_range_end = col_values[0][0]
#                 current_range_values = [col_values[0][1]]
                
#                 for i in range(1, len(col_values)):
#                     col, value = col_values[i]
#                     if col == current_range_end + 1:
#                         # Contiguous, extend range
#                         current_range_end = col
#                         current_range_values.append(value)
#                     else:
#                         # Gap found, save current range and start new one
#                         ranges.append((current_range_start, current_range_end, current_range_values))
#                         current_range_start = col
#                         current_range_end = col
#                         current_range_values = [value]
                
#                 # Add last range
#                 ranges.append((current_range_start, current_range_end, current_range_values))
                
#                 # Create updateCells request for each contiguous range
#                 requests = []
#                 for start_col_idx, end_col_idx, values in ranges:
#                     cell_data = []
#                     for value in values:
#                         # Determine if it's a formula or string
#                         if value.startswith("="):
#                             cell_data.append({
#                                 "userEnteredValue": {"formulaValue": value}
#                             })
#                         else:
#                             cell_data.append({
#                                 "userEnteredValue": {"stringValue": value}
#                             })
                    
#                     requests.append({
#                         "updateCells": {
#                             "range": {
#                                 "sheetId": worksheet.id,
#                                 "startRowIndex": row - 1,  # Convert to 0-based
#                                 "endRowIndex": row,  # Exclusive
#                                 "startColumnIndex": start_col_idx - 1,  # Convert to 0-based
#                                 "endColumnIndex": end_col_idx  # Convert to 0-based (exclusive)
#                             },
#                             "rows": [{
#                                 "values": cell_data
#                             }],
#                             "fields": "userEnteredValue"
#                         }
#                     })
                
#                 # Execute batch update
#                 if requests:
#                     def update_batch():
#                         worksheet.spreadsheet.batch_update({"requests": requests})
#                     retry_with_backoff(update_batch)
    
#     # Update notes
#     if note_updates:
#         requests = []
#         for note_update in note_updates:
#             requests.append({
#                 "updateCells": {
#                     "range": {
#                         "sheetId": worksheet.id,
#                         "startRowIndex": note_update["row"],
#                         "endRowIndex": note_update["row"] + 1,
#                         "startColumnIndex": note_update["col"],
#                         "endColumnIndex": note_update["col"] + 1
#                     },
#                     "rows": [{
#                         "values": [{
#                             "note": note_update["note"]
#                         }]
#                     }],
#                     "fields": "note"
#                 }
#             })
        
#         if requests:
#             def update_notes():
#                 worksheet.spreadsheet.batch_update({"requests": requests})
#             retry_with_backoff(update_notes)
    
#     return index, col_offset


# @traceable(
#     metadata={
#         "agent_name": "graphics_definition_v2",
#         "step_name": "Populate Sheet with Selected Images",
#         "function_name": "run_populate_sheet_with_selected_images_for_all_rows",
#         "user_id": st.session_state.get("role", "anonymous"),
#         "user_email": st.session_state.get("user_email", "anonymous")
#     }
# )
# def run_populate_sheet_with_selected_images_for_all_rows(sheet):
#     """
#     Populate images from graphics definitions into the sheet.
    
#     :param sheet: The gspread sheet object.
#     :return: None
#     """
#     worksheet_name = "Slide Chunks"
    
#     # Load the worksheet and DataFrame
#     worksheet, df = get_sheet_data_and_df(sheet, worksheet_name)
    
#     # Find graphics definition column
#     gd_column = None
#     for col in df.columns:
#         if "graphics_definition" in col.lower() or "graphics definition" in col.lower():
#             gd_column = col
#             break
    
#     if not gd_column:
#         print("⚠️  No 'graphics_definition' column found in sheet")
#         print(f"Available columns: {list(df.columns)}")
#         return
    
#     print(f"✅ Found '{gd_column}' column with {len(df)} rows")
    
#     # Get existing headers to find where to start adding columns
#     existing_headers = worksheet.row_values(1)
#     start_col = len(existing_headers) + 1
    
#     # Get current sheet dimensions
#     current_col_count = worksheet.col_count
    
#     # Calculate maximum columns we might need
#     max_col_offset = 0
#     for idx, row in df.iterrows():
#         gd_content = str(row.get(gd_column, "")).strip()
#         if not gd_content or gd_content == "nan":
#             continue
        
#         segments = parse_graphics_definition(gd_content)
#         if not segments:
#             continue
        
#         col_offset = 0
#         for seg in segments:
#             images = seg["images"]
#             col_offset += len(images)
#         max_col_offset = max(max_col_offset, col_offset)
    
#     # Calculate total columns needed
#     if max_col_offset > 0:
#         total_cols_needed = start_col + max_col_offset - 1
#     else:
#         total_cols_needed = current_col_count
    
#     # Add columns if needed
#     if total_cols_needed > current_col_count:
#         cols_to_add = total_cols_needed - current_col_count + 2  # +2 buffer
#         try:
#             worksheet.add_cols(cols_to_add)
#             print(f"✅ Added {cols_to_add} columns to accommodate images")
#         except Exception as e:
#             print(f"⚠️  Could not add columns: {e}")
    
#     # Set headers for all image columns upfront (based on max_col_offset)
#     if max_col_offset > 0:
#         header_requests = []
#         for col_idx in range(max_col_offset):
#             header_requests.append({
#                 "updateCells": {
#                     "range": {
#                         "sheetId": worksheet.id,
#                         "startRowIndex": 0,  # Header row (0-based)
#                         "endRowIndex": 1,
#                         "startColumnIndex": start_col - 1 + col_idx,  # Convert to 0-based
#                         "endColumnIndex": start_col + col_idx
#                     },
#                     "rows": [{
#                         "values": [{
#                             "userEnteredValue": {"stringValue": f"Image {col_idx + 1}"}
#                         }]
#                     }],
#                     "fields": "userEnteredValue"
#                 }
#             })
        
#         if header_requests:
#             retry_with_backoff(lambda: worksheet.spreadsheet.batch_update({"requests": header_requests}))
#             print(f"✅ Set headers for {max_col_offset} image column(s)")
    
#     # Process rows
#     total_columns_created = 0
#     rows_processed = 0
    
#     # Initialize progress tracker
#     total_rows = len(df)
#     progress = SmartProgressBar(
#         total_tasks=total_rows,
#         description="Populating images",
#         save_interval=10  # Save every 10 rows
#     )
    
#     for idx, row in df.iterrows():
#         try:
#             _, col_offset = populate_images_for_row(idx, row, worksheet, gd_column, start_col)
#             total_columns_created = max(total_columns_created, col_offset)
#             rows_processed += 1
            
#             progress.update()
            
#             # Small delay to avoid rate limits
#             if idx % 10 == 0 and idx > 0:
#                 time.sleep(0.5)
#         except Exception as e:
#             print(f"Error processing row {idx}: {e}")
#             progress.update()
    
#     print(f"✅ Populated images for {rows_processed} rows")
#     format_worksheet(worksheet)
    
#     try:
#         requests = []
#         # Set row heights for data rows (120 pixels for images)
#         for idx in range(len(df)):
#             requests.append({
#                 "updateDimensionProperties": {
#                     "range": {
#                         "sheetId": worksheet.id,
#                         "dimension": "ROWS",
#                         "startIndex": idx + 1,  # Skip header row
#                         "endIndex": idx + 2
#                     },
#                     "properties": {"pixelSize": 120},
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
#             print("✅ Adjusted row heights and column widths")
#     except Exception as e:
#         print(f"⚠️  Could not adjust row/column sizes: {e}")


# def delete_populated_images(sheet):
#     """
#     Delete all images and notes populated by the populate sheet with selected images step.
    
#     :param sheet: The gspread sheet object.
#     :return: None
#     """
#     worksheet_name = "Slide Chunks"
    
#     # Load the worksheet and DataFrame
#     worksheet, df = get_sheet_data_and_df(sheet, worksheet_name)
    
#     # Get all headers to find image columns
#     headers = worksheet.row_values(1)
    
#     # Find all columns that start with "Image " (like "Image 1", "Image 2", etc.)
#     image_column_indices = []
#     for idx, header in enumerate(headers):
#         if isinstance(header, str) and header.strip().startswith("Image "):
#             image_column_indices.append(idx)
    
#     if not image_column_indices:
#         print("ℹ️  No image columns found to delete.")
#         return
    
#     print(f"✅ Found {len(image_column_indices)} image column(s) to delete")
    
#     # Get total number of rows (including header)
#     total_rows = len(df) + 1  # +1 for header row
    
#     # Create requests to clear cells and notes
#     requests = []
    
#     for col_idx in image_column_indices:

#         empty_rows = []
#         for _ in range(total_rows):
#             empty_rows.append({
#                 "values": [{}]  
#             })
        
#         requests.append({
#             "updateCells": {
#                 "range": {
#                     "sheetId": worksheet.id,
#                     "startRowIndex": 0,  # Start from header row (0-based)
#                     "endRowIndex": total_rows,  # End at last data row (exclusive)
#                     "startColumnIndex": col_idx,  # 0-based column index
#                     "endColumnIndex": col_idx + 1
#                 },
#                 "rows": empty_rows,
#                 "fields": "userEnteredValue"
#             }
#         })
        
#         # Clear notes for all rows in this column
#         empty_note_rows = []
#         for _ in range(total_rows):
#             empty_note_rows.append({
#                 "values": [{"note": ""}]  # Empty note to clear
#             })
        
#         requests.append({
#             "updateCells": {
#                 "range": {
#                     "sheetId": worksheet.id,
#                     "startRowIndex": 0,  # Start from header row (0-based)
#                     "endRowIndex": total_rows,  # End at last data row (exclusive)
#                     "startColumnIndex": col_idx,  # 0-based column index
#                     "endColumnIndex": col_idx + 1
#                 },
#                 "rows": empty_note_rows,
#                 "fields": "note"
#             }
#         })
    
#     # Reset row heights back to default (40px from format_worksheet)
#     for idx in range(len(df)):
#         requests.append({
#             "updateDimensionProperties": {
#                 "range": {
#                     "sheetId": worksheet.id,
#                     "dimension": "ROWS",
#                     "startIndex": idx + 1,  # Skip header row
#                     "endIndex": idx + 2
#                 },
#                 "properties": {"pixelSize": 40},
#                 "fields": "pixelSize"
#             }
#         })
    
#     # Execute batch update
#     if requests:
#         retry_with_backoff(lambda: worksheet.spreadsheet.batch_update({"requests": requests}))
#         print(f"✅ Cleared {len(image_column_indices)} image column(s) and all associated notes")
#     else:
#         print("⚠️  No requests to execute")

