from modules.chain import Chain
from langsmith import traceable
import streamlit as st
from services.sheets_service import get_sheet_data_and_df, save_to_sheet, format_worksheet, clear_worksheet, resize_column_by_name
from services.smart_progress_bar import SmartProgressBar
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
import re
import os
import json

load_dotenv()


def ensure_visual_assignment_strategy_column(sheet, worksheet_name="Slide Chunks"):
    """
    Adds the 'Visual Assignment Strategy' column if missing in Slide Chunks sheet.
    Fills all rows with default value and sets up dropdown with 3 options.
    This is designed to run as a pre_exec_func before the main agent starts.
    
    :param sheet: The gspread sheet object.
    :param worksheet_name: Name of the worksheet (default: "Slide Chunks").
    :return: None
    """
    worksheet, df = get_sheet_data_and_df(sheet, worksheet_name)
    
    column_name = "Visual Assignment Strategy"
    default_value = "Flexible, let the agent decide"
    transition_default = "1 Visual for the whole Slide"
    dropdown_options = [
        "1 Visual per Sentence",
        "1 Visual for the whole Slide",
        "Flexible, let the agent decide"
    ]
    slide_type_col = "Slide Type"

    def _default_for_row(row):
        """Default for a single row: 1 visual for whole slide if Slide Type is Transition."""
        if slide_type_col not in df.columns:
            return default_value
        st_val = str(row.get(slide_type_col, "")).strip().lower()
        if st_val in ("transition", "transition slide"):
            return transition_default
        return default_value

    # Check if column exists
    if column_name not in df.columns:
        # Add column: per-row default (Transition -> 1 visual for whole slide)
        df[column_name] = df.apply(_default_for_row, axis=1)
        save_to_sheet(worksheet=worksheet, df=df)
        print(f"✅ '{column_name}' column added.")
    else:
        print(f"ℹ️ '{column_name}' column already exists.")

    # Get updated data to find column index
    worksheet, df = get_sheet_data_and_df(sheet, worksheet_name)

    if column_name in df.columns:
        empty_mask = df[column_name].isna() | (df[column_name].astype(str).str.strip() == "")
        if empty_mask.any():
            df.loc[empty_mask, column_name] = df.loc[empty_mask].apply(_default_for_row, axis=1)
            save_to_sheet(worksheet=worksheet, df=df)
            print(f"✅ Filled empty cells in '{column_name}' with default value (Transition → 1 visual for whole slide).")
    
    # Set up data validation (dropdown) for the column
    headers = worksheet.row_values(1)
    if column_name in headers:
        col_index = headers.index(column_name)
        
        # Get total number of rows (including header)
        total_rows = len(df) + 1  # +1 for header row
        
        # Create data validation rule
        # Note: Google Sheets API uses 0-based indexing
        validation_request = {
            'setDataValidation': {
                'range': {
                    'sheetId': worksheet.id,
                    'startRowIndex': 1,  # Start from row 2 (skip header)
                    'endRowIndex': total_rows,
                    'startColumnIndex': col_index,
                    'endColumnIndex': col_index + 1
                },
                'rule': {
                    'condition': {
                        'type': 'ONE_OF_LIST',
                        'values': [{'userEnteredValue': option} for option in dropdown_options]
                    },
                    'showCustomUi': True,  # Shows dropdown arrow
                    'strict': True  # Only allows values from the list
                }
            }
        }
        
        # Apply the validation
        worksheet.spreadsheet.batch_update({'requests': [validation_request]})
        print(f"✅ Dropdown validation set for '{column_name}' column with {len(dropdown_options)} options.")
        
        # Resize column to 160 pixels
        resize_column_by_name(worksheet, column_name, pixel_size=160)
        print(f"✅ Column '{column_name}' resized to 160 pixels.")
    else:
        print(f"⚠️ Column '{column_name}' not found after creation attempt.")


segment_slide_prompt = """You are an expert instructional designer. Your task is to split the slide content into individual voiceover (VO) segments using sentence-based segmentation. Each complete sentence becomes one segment.

This is the slide content that needs to be segmented:

<slide_content>
Slide Content:
{slide_chunk}
</slide_content>

RULES:

1. Sentence-based splitting:
   - Split only at real sentence boundaries: ".", "!", "?"
   - One sentence = one <segment>
   - Do not split mid-sentence
   - Do not merge multiple sentences

2. Text must be copied exactly without any change:
   - No paraphrasing, rewording, or summarizing
   - Preserve all punctuation, capitalization, numbering, technical terms, etc.

3. Full coverage:
   - Include every sentence from the slide
   - Segments must appear in original order
 
EXAMPLES:

Example 1:
Slide Content: "The low-pressure gauge is on the left side of the manifold set and is typically blue. It measures the pressure on the low side, suction side, of the system."
Segment 1: "The low-pressure gauge is on the left side of the manifold set and is typically blue."
Segment 2: "It measures the pressure on the low side, suction side, of the system."

Example 2:
Slide Content: "First, connect the blue hose to the low-pressure port. Then, open the valve slowly. Finally, read the pressure gauge."
Segment 1: "First, connect the blue hose to the low-pressure port."
Segment 2: "Then, open the valve slowly."
Segment 3: "Finally, read the pressure gauge."

STRICT INSTRUCTION: While doing the segmentation, only include content inside the Slide Content that is given as input. Do not add or remove any content from the Slide Content no matter how big or small the content is.


OUTPUT FORMAT:
You must always output segments in the following XML format. Each segment must be wrapped in <segment> tags:

<segments>
<segment>First sentence here - copied exactly from the slide</segment>
<segment>Second sentence here - copied exactly from the slide</segment>
<segment>Third sentence here - copied exactly from the slide</segment>
...
</segments>
"""


@traceable(
    metadata={
        "agent_name": "graphics_definition_v2",
        "step_name": "Segment Slide",
        "function_name": "segment_slide_from_slide_chunk",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous")
    }
)
def segment_slide_from_slide_chunk(slide_chunk, llm="gemini_2_5_flash_lite"):
    """
    Segment slide content into individual voiceover (VO) segments using sentence-based segmentation.

    :param slide_chunk: The slide content to segment.
    :param llm: The language model to use.
    :return: The segmented output with segments in XML format.
    """
    # Initialize the segmentation agent
    segment_agent = Chain(llm=llm, tags=["segments"])

    # # Print the formatted prompt for debugging
    # print("\n🔍 Segment Slide Prompt Being Sent to LLM:\n")
    # print(segment_slide_prompt.format(slide_chunk=slide_chunk))
    # print("\n" + "=" * 100 + "\n")

    # Add the user message
    segment_agent.add_message(
        role="user",
        content=segment_slide_prompt.format(
            slide_chunk=slide_chunk
        )
    )

    # Run the segmentation agent
    response = segment_agent.run()

    # Print the response for debugging
    segments_output = response.get("segments", "")
    print(f"\nSlide Content: {slide_chunk}\n")
    print("📤 Segment Slide Response from LLM:\n")
    print(segments_output)
    print("\n" + "=" * 100 + "\n")

    return segments_output


@traceable(
    metadata={
        "agent_name": "graphics_definition_v2",
        "step_name": "Segment Slide",
        "function_name": "process_segment_row",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous")
    }
)
def process_segment_row(
    index,
    row,
    df,
    llm="gemini_2_5_flash_lite",
):
    """
    Process a single row for Step 1 (voiceover segmentation).

    - If a Reference Image URL is present: skip LLM segmentation entirely and set
      voiceover_segment to the full slide chunk text. The reference image pipeline
      will run later in Step 5 (Aggregation Agent) in parallel with normal GDv2 path.
    - Otherwise: run the normal LLM-based sentence-level segmentation.
    """
    try:
        slide_chunk = str(row.get("Slide Chunk", "")).strip()
        ref_image_url = str(row.get("Reference Image", "")).strip()

        result_payload = {
            "mode": "combined",
            "voiceover_segment": "",
        }

        has_ref_image = ref_image_url and ref_image_url.lower() != "nan"

        if has_ref_image:
            # Reference image detected — skip LLM segmentation.
            # voiceover_segment = full slide chunk so all downstream steps
            # (storyboard, search, pools) run with the complete slide text.
            # The reference image itself is processed in Step 5 (Aggregation).
            print(f"🔗 Row {index + 1}: Reference image detected — setting voiceover_segment to full slide chunk.")
            result_payload["voiceover_segment"] = slide_chunk
            result_payload["Visual Assignment Strategy"] = "1 Visual for the whole Slide"
        else:
            # Normal GDv2 path: LLM-based sentence segmentation
            segments_xml = segment_slide_from_slide_chunk(slide_chunk, llm=llm)
            segment_matches = re.findall(r'<segment>(.*?)</segment>', segments_xml, re.DOTALL)
            segment_texts = [seg.strip() for seg in segment_matches if seg.strip()]
            result_payload["voiceover_segment"] = '\n'.join(segment_texts)

        return index, result_payload

    except Exception as e:
        print(f"Error processing row {index}: {e}")
        return index, {"mode": "error", "message": f"ERROR: {str(e)}"}


@traceable(
    metadata={
        "agent_name": "graphics_definition_v2",
        "step_name": "Segment Slide",
        "function_name": "run_segment_slide_from_slide_chunk_for_all_rows",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous")
    }
)
def run_segment_slide_from_slide_chunk_for_all_rows(sheet, llm="gemini_2_5_flash_lite", max_workers=50):
    """
    Segment slide content into VO segments for all rows in the Slide Chunks sheet.

    :param sheet: The gspread sheet object.
    :param llm: The language model to use.
    :param max_workers: Number of parallel workers.
    :return: None
    """
    # Load the worksheet and DataFrame
    worksheet, df = get_sheet_data_and_df(sheet, "Slide Chunks")

    # Ensure voiceover_segment column exists
    if "voiceover_segment" not in df.columns:
        df["voiceover_segment"] = ""

    # Ensure all potential columns exist for clearing
    cols_to_clear = [
        "voiceover_segment", "storyboard_planning", "search_queries",
        "drive_results", "web_results", "video_pool",
        "video_pool_other_channels", "image_pool", "video_pool_filtered",
        "final_graphics_definition"
    ]
    for col in cols_to_clear:
        if col not in df.columns:
            df[col] = ""

    # Prepare for parallel processing - only process rows that need processing
    futures_map = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks first
        for index, row in df.iterrows():
            slide_chunk = str(row.get("Slide Chunk", "")).strip()
            vo_segment = str(row.get("voiceover_segment", "")).strip()

            # Skip if slide_chunk is empty
            if not slide_chunk or slide_chunk == "nan":
                continue

            # Skip if voiceover_segment is already filled
            if vo_segment and vo_segment != "nan":
                continue

            # Submit task for processing
            future = executor.submit(
                process_segment_row,
                index,
                row,
                df,
                llm,
            )
            futures_map[future] = index

        # If no rows to process, return early
        if not futures_map:
            print("All rows already processed or no valid slide chunks found.")
            return

        # Initialize progress tracker 
        total_tasks = len(futures_map)
        progress = SmartProgressBar(
            total_tasks=total_tasks,
            description="Segmenting slides",
            save_interval=5
        )

        # Collect results as they complete
        for future in as_completed(futures_map):
            index = futures_map[future]
            try:
                row_index, result_data = future.result()

                if result_data.get("mode") == "combined":
                    df.at[row_index, "voiceover_segment"] = result_data["voiceover_segment"]
                    # Persist Visual Assignment Strategy override for reference rows
                    if "Visual Assignment Strategy" in result_data:
                        df.at[row_index, "Visual Assignment Strategy"] = result_data["Visual Assignment Strategy"]

                elif result_data.get("mode") == "error":
                    df.at[row_index, "voiceover_segment"] = result_data["message"]

                # Update progress
                progress.update()

                # Save every 5 rows
                if progress.should_save():
                    print(f'Saving partial progress to sheet after {progress.completed_count} tasks completed.')
                    save_to_sheet(worksheet, df)
            except Exception as e:
                print(f"Error getting result for row {index}: {e}")
                # Update dataframe with error marker so row is marked as processed
                df.at[index, "voiceover_segment"] = f"ERROR: {str(e)}"
                progress.update()

    # Save final results before validation
    save_to_sheet(worksheet, df)
    format_worksheet(worksheet)

    # Validation and retry logic
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        # Reload dataframe to get latest state
        worksheet, df = get_sheet_data_and_df(sheet, "Slide Chunks")
        
        # Find rows that need processing (have Slide Chunk but empty voiceover_segment)
        invalid_rows = []
        for index, row in df.iterrows():
            slide_chunk = str(row.get("Slide Chunk", "")).strip()
            vo_segment = str(row.get("voiceover_segment", "")).strip()
            
            # Skip if Slide Chunk is empty
            if not slide_chunk or slide_chunk == "nan":
                continue
            
            # Check if voiceover_segment is empty or just whitespace/nan/error
            if not vo_segment or vo_segment == "nan" or vo_segment.strip() == "" or vo_segment.startswith("ERROR:"):
                invalid_rows.append((index, row))
        
        if not invalid_rows:
            break
        
        retry_count += 1
        print(f"\n⚠️ Found {len(invalid_rows)} rows with Slide Chunk but empty voiceover_segment. Retrying (attempt {retry_count}/{max_retries})...")
        
        # Clear voiceover_segment for invalid rows
        for index, row in invalid_rows:
            df.at[index, "voiceover_segment"] = ""
        
        # Save cleared state
        save_to_sheet(worksheet, df)
        format_worksheet(worksheet)
        
        # Retry processing invalid rows
        futures_map = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for index, row in invalid_rows:
                future = executor.submit(
                    process_segment_row,
                    index,
                    row,
                    df,
                    llm,
                )
                futures_map[future] = index

            # Collect results
            for future in as_completed(futures_map):
                index = futures_map[future]
                try:
                    row_index, result_data = future.result()
                    df.at[row_index, "voiceover_segment"] = result_data.get("voiceover_segment", "")
                    # Persist Visual Assignment Strategy override for reference rows
                    if "Visual Assignment Strategy" in result_data:
                        df.at[row_index, "Visual Assignment Strategy"] = result_data["Visual Assignment Strategy"]
                except Exception as e:
                    print(f"Error getting result for row {index} on retry: {e}")
                    df.at[index, "voiceover_segment"] = f"ERROR: {str(e)}"
        
        # Save after retry
        save_to_sheet(worksheet, df)
        format_worksheet(worksheet)
    
    if retry_count > 0:
        # Check final state
        worksheet, df = get_sheet_data_and_df(sheet, "Slide Chunks")
        final_invalid = []
        for index, row in df.iterrows():
            slide_chunk = str(row.get("Slide Chunk", "")).strip()
            vo_segment = str(row.get("voiceover_segment", "")).strip()
            if (slide_chunk and slide_chunk != "nan" and 
                (not vo_segment or vo_segment == "nan" or vo_segment.strip() == "" or vo_segment.startswith("ERROR:"))):
                final_invalid.append(index)
        
        if final_invalid:
            print(f"⚠️ After {retry_count} retry attempt(s), {len(final_invalid)} rows still have empty voiceover_segment.")
        else:
            print(f"✅ All rows filled after {retry_count} retry attempt(s).")
    
    # Final save to sheet
    print('All slides processed. Saving final DataFrame to sheet.')
    save_to_sheet(worksheet, df)
    format_worksheet(worksheet)
    print("✅ Slide segmentation complete and saved to sheet.")


def delete_segment_slide(sheet):
    """
    Remove the 'voiceover_segment' column from the Slide Chunks worksheet.
    
    :param sheet: The gspread sheet object.
    :return: None
    """
    ws, df = get_sheet_data_and_df(sheet, "Slide Chunks")
    if "voiceover_segment" in df.columns:
        df = df.drop(columns=["voiceover_segment"])
        clear_worksheet(ws)
        save_to_sheet(ws, df)
        print(f"🗑️ Deleted 'voiceover_segment' column from 'Slide Chunks' worksheet")
    else:
        print(f"ℹ️ 'voiceover_segment' column does not exist in 'Slide Chunks' worksheet")
