from modules.chain import Chain
from langsmith import traceable
import streamlit as st
from services.sheets_service import get_sheet_data_and_df, save_to_sheet, format_worksheet, clear_worksheet
from services.smart_progress_bar import SmartProgressBar
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
import re

load_dotenv()


segment_slide_prompt = """You are an expert instructional designer. Your task is to split the slide content into individual voiceover (VO) segments using sentence-based segmentation. Each complete sentence becomes one segment.

This is the slide content that needs to be segmented:

Slide Content:
{slide_chunk}

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
def segment_slide_from_slide_chunk(slide_chunk, llm="gemini_3_flash_thinking"):
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
def process_segment_row(index, slide_chunk, llm="gemini_3_flash_thinking"):
    """
    Process a single row and return the segmented output.

    :param index: The row index.
    :param slide_chunk: The slide content to segment.
    :param llm: The language model to use.
    :return: Tuple of (index, vo_segments_text) where vo_segments_text is newline-separated segments.
    """
    try:
        # Get the XML segments output
        segments_xml = segment_slide_from_slide_chunk(slide_chunk, llm=llm)
        
        # Extract segment text content (without XML tags)
        segment_matches = re.findall(r'<segment>(.*?)</segment>', segments_xml, re.DOTALL)
        segment_texts = [seg.strip() for seg in segment_matches if seg.strip()]
        
        # Join segments with newlines
        vo_segments_text = '\n'.join(segment_texts)
        
        return index, vo_segments_text
    except Exception as e:
        print(f"Error processing row {index}: {e}")
        return index, ""


@traceable(
    metadata={
        "agent_name": "graphics_definition_v2",
        "step_name": "Segment Slide",
        "function_name": "run_segment_slide_from_slide_chunk_for_all_rows",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous")
    }
)
def run_segment_slide_from_slide_chunk_for_all_rows(sheet, llm="gemini_3_flash_thinking", max_workers=5):
    """
    Segment slide content into VO segments for all rows in the Slide Chunks sheet.

    :param sheet: The gspread sheet object.
    :param llm: The language model to use.
    :param max_workers: Number of parallel workers (default 5).
    :return: None
    """
    # Load the worksheet and DataFrame
    worksheet, df = get_sheet_data_and_df(sheet, "Slide Chunks")

    # Ensure voiceover_segment column exists
    if "voiceover_segment" not in df.columns:
        df["voiceover_segment"] = ""

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
            future = executor.submit(process_segment_row, index, slide_chunk, llm)
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
                row_index, vo_segments_text = future.result()
                
                # Update dataframe
                df.at[row_index, "voiceover_segment"] = vo_segments_text
                
                # Update progress
                progress.update()
                
                # Save every 5 rows
                if progress.should_save():
                    print(f'Saving partial progress to sheet after {progress.completed_count} tasks completed.')
                    save_to_sheet(worksheet, df)
                    format_worksheet(worksheet)
            except Exception as e:
                print(f"Error getting result for row {index}: {e}")
                # Update dataframe with error marker so row is marked as processed
                df.at[index, "voiceover_segment"] = f"ERROR: {str(e)}"
                progress.update()

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
                slide_chunk = str(row.get("Slide Chunk", "")).strip()
                future = executor.submit(process_segment_row, index, slide_chunk, llm)
                futures_map[future] = index
            
            # Collect results
            for future in as_completed(futures_map):
                index = futures_map[future]
                try:
                    row_index, vo_segments_text = future.result()
                    df.at[row_index, "voiceover_segment"] = vo_segments_text
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

