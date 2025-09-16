import pandas as pd
from modules.chain import Chain
from pydantic import BaseModel, Field
from concurrent.futures import ThreadPoolExecutor, as_completed
from services.sheets_service import (
    get_sheet_data_and_df,
    create_or_read_worksheet,
    save_to_sheet,
    format_worksheet,
    delete_worksheet,
    get_worksheet_names,
)
from services.smart_progress_bar import SmartProgressBar
from langsmith import traceable
import streamlit as st
import re

class SlideChunk(BaseModel):
    topic: str = Field(description="The topic this slide belongs to.")
    subtopic: str = Field(description="The subtopic this slide belongs to.")
    slide_type: str = Field(description="The type of slide (Transition, Content, Video, Summary, etc.)")
    slide_chunk_title: str = Field(description="The title of the slide chunk.")
    slide_chunk: str = Field(description="The content/body of the slide chunk.")

slide_chunk_parsing_prompt = """You are an expert parser for E-learning slide content. Given a block of text representing a single slide, extract the following fields in a structured way:

- slide_type: The value after 'Slide Type:'
- slide_chunk_title: The value after 'Title:'
- slide_chunk: For blocks with a 'Content:' field, extract everything after 'Content:'. For blocks with 'Slide Type: Video', extract everything after the 'Slide Type:' line (including Video_Id, Start, End and Transcript.)

Return the output as a structured object with these fields. Do not add or infer any information. Only extract what is present in the block.

Block:
{block}
"""

@traceable(
    metadata={
        "agent_name": "slide_chunks",
        "step_name": "Slide Chunks Parsing",
        "function_name": "run_slide_chunks_parsing",
        "user_id": st.session_state.get("role", "anonymous")
    }
)
def run_slide_chunks_parsing(sheet, worksheet_name="Final Outline", output_sheet_name="Slide Chunks", max_workers=5):
    """
    Parses the slide_chunks column in the Final Outline sheet and outputs a structured Slide Chunks sheet.
    :param sheet: The gspread sheet object.
    :param worksheet_name: The worksheet name to read from.
    :param output_sheet_name: The worksheet name to write to.
    :param max_workers: Number of parallel workers (default 5).
    :return: The DataFrame written to the Slide Chunks sheet.
    """
    _, df = get_sheet_data_and_df(sheet, worksheet_name)

    # Prepare for parallel processing
    slide_chunks_data = []
    futures_map = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for index, row in df.iterrows():
            slide_chunks_cell = str(row.get("slide_chunks", "")).strip()
            if not slide_chunks_cell:
                continue  # Skip empty slide_chunks
            topic = row["Topic"]
            subtopic = row["Subtopic"]
            future = executor.submit(process_slide_chunks_row, topic, subtopic, slide_chunks_cell, index)
            futures_map[future] = index

        total_tasks = len(futures_map)
        progress = SmartProgressBar(total_tasks=total_tasks, description="Parsing slide chunks", save_interval=5)
        results = [None] * len(df)
        for future in as_completed(futures_map):
            index = futures_map[future]
            try:
                parsed_blocks = future.result()
                results[index] = parsed_blocks
            except Exception as e:
                print(f"Error parsing row {index}: {e}")
            progress.update()

    # Flatten and filter results, preserving order
    for parsed_blocks in results:
        if parsed_blocks:
            slide_chunks_data.extend(parsed_blocks)

    # Convert to DataFrame
    slide_chunks_df = pd.DataFrame(slide_chunks_data)
    column_order = ["Topic", "Subtopic", "Slide Type", "Slide Chunk Title", "Slide Chunk"]
    slide_chunks_df = slide_chunks_df.rename(columns={
        "topic": "Topic",
        "subtopic": "Subtopic",
        "slide_type": "Slide Type",
        "slide_chunk_title": "Slide Chunk Title",
        "slide_chunk": "Slide Chunk"
    })
    slide_chunks_df = slide_chunks_df[column_order]

    # Overwrite the Slide Chunks sheet
    slide_chunks_sheet, _ = create_or_read_worksheet(sheet, output_sheet_name)
    slide_chunks_sheet.clear()
    slide_chunks_sheet.append_row(column_order)
    save_to_sheet(slide_chunks_sheet, slide_chunks_df)
    format_worksheet(slide_chunks_sheet)
    print(f" Successfully updated the '{output_sheet_name}' tab with {len(slide_chunks_df)} rows.")
    return slide_chunks_df


@traceable(
    metadata={
        "agent_name": "slide_chunks",
        "step_name": "Slide Chunks Parsing",
        "function_name": "process_slide_chunks_row",
        "user_id": st.session_state.get("role", "anonymous")
    }
)
def process_slide_chunks_row(topic, subtopic, slide_chunks_cell, index):
    """
    Parses all blocks in a slide_chunks cell for a single row.
    :param topic: The topic for this row.
    :param subtopic: The subtopic for this row.
    :param slide_chunks_cell: The raw text from the slide_chunks column.
    :param index: The row index (for ordering).
    :return: List of dicts for each parsed slide chunk.
    """
    blocks = [b.strip() for b in re.split(r"\n\s*\n", slide_chunks_cell) if b.strip()]
    parsed_blocks = []
    for block in blocks:
        # Use LLM + Pydantic to parse and validate
        agent = Chain(llm="gemini_2_flash")
        agent.add_message(
            role="user",
            content=slide_chunk_parsing_prompt.format(block=block)
        )
        agent.structured_output = SlideChunk
        response = agent.run()
        parsed = response.model_dump()
        # Set topic and subtopic from the row, not from the LLM
        parsed['topic'] = topic
        parsed['subtopic'] = subtopic
        parsed_blocks.append(parsed)
    return parsed_blocks

def delete_slide_chunks_sheet(sheet, worksheet_name="Slide Chunks"):
    """
    Delete the Slide Chunks worksheet if present.
    :param sheet: The gspread sheet object.
    :param worksheet_name: The worksheet name (default 'Slide Chunks').
    :return: None
    """
    delete_worksheet(sheet, worksheet_name)
    
    # Delete the checklist backup sheet if it exists
    backup_name = "Backup Slide Chunks Sheet for Delete step of Slide Chunks Checklist"
    sheet_names = get_worksheet_names(sheet)
    if backup_name in sheet_names:
        delete_worksheet(sheet, backup_name)
        print(f"🗑️ Deleted backup sheet '{backup_name}' when deleting slide chunks sheet")
