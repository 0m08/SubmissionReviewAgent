import pandas as pd
from modules.chain import Chain
from tqdm import tqdm
from agents.slide_chunks.format_inputs import Subtopic
from concurrent.futures import ThreadPoolExecutor, as_completed
from services.sheets_service import get_sheet_data_and_df, create_or_read_worksheet, save_to_sheet, format_worksheet
from services.smart_progress_bar import SmartProgressBar
from langsmith import traceable
import streamlit as st 







extract_slide_info_prompt = """You are an expert in structuring and extracting slide information from XML-formatted content. Your task is to extract slide details for a given topic from the following content.

<content>
{content}
</content>

1. Extraction Rules:
    Subtopic Name: The subtopic name should be the same as the transition slide's title.
    Slides to Extract:
     - Transition Slide: Extract its title and content.
     - Content Slides: Extract a list of slides, each with its title and content.
     - Summary Slide: Extract its title and content.

2. Extraction Requirements:
    - Ensure all slides are extracted completely without skipping or paraphrasing.
    - Keep the original wording of slide content - do not rephrase or modify.
"""

@traceable(
    metadata={
        "agent_name": "slide_chunks",
        "step_name": "Research Notes Parsing",
        "function_name": "process_row",
        "user_id": st.session_state.get("role", "anonymous")
    }
)
def process_row(row, index):
    slide_chunks_data = []
    topic = row['Topic']
    section_notes = row['section_notes']

    # Split section_notes into individual subtopics
    subtopic_contents = section_notes.split("\n\n---\n\n")

    # Process each subtopic separately
    for subtopic_content in subtopic_contents:
        # Initialize the LLM agent with our prompt
        extract_slide_info_agent = Chain(llm="gemini_2_flash")
        extract_slide_info_agent.add_message(
            role="user",
            content=extract_slide_info_prompt.format(content=subtopic_content)
        )

        # Define structured output format using Pydantic
        extract_slide_info_agent.structured_output = Subtopic

        try:
            # Run the LLM extraction
            response = extract_slide_info_agent.run()
        except Exception as e:
            print(f"Error processing topic '{topic}': {e}")
            continue  # Move to the next subtopic if LLM fails

        # Append extracted slides to the list for Slide Chunks sheet
        slide_chunks_data.append({
            "Topic": topic,
            "Subtopic": response.subtopic_name,
            "Slide Type": "Transition Slide",
            "Slide Chunk Title": response.transition_slide.slide_title,
            "Slide Chunk": response.transition_slide.slide_content
        })

        for slide in response.content_slides:
            slide_chunks_data.append({
                "Topic": topic,
                "Subtopic": response.subtopic_name,
                "Slide Type": "Content Slide",
                "Slide Chunk Title": slide.slide_title,
                "Slide Chunk": slide.slide_content
            })

        slide_chunks_data.append({
            "Topic": topic,
            "Subtopic": response.subtopic_name,
            "Slide Type": "Summary Slide",
            "Slide Chunk Title": response.summary_slide.slide_title,
            "Slide Chunk": response.summary_slide.slide_content
        })

    return index, slide_chunks_data

@traceable(
    metadata={
        "agent_name": "slide_chunks",
        "step_name": "Research Notes Parsing",
        "function_name": "run_research_notes_parsing",
        "user_id": st.session_state.get("role", "anonymous")
    }
)
def run_research_notes_parsing(sheet, worksheet_name):
    _, research_notes_df = get_sheet_data_and_df(sheet, worksheet_name)

    # List to store processed slide chunks
    slide_chunks_data = []

    # Check if the slide chunks sheet is present and populated, skip processing
    try:
        slide_chunks_sheet, slide_chunks_df = create_or_read_worksheet(sheet, "Slide Chunks")
        if not slide_chunks_df.empty:
            print("Slide Chunks sheet is already populated. Skipping processing.")
            return slide_chunks_df
    except Exception as e:
        print(f"Slide Chunks sheet not found or empty. Proceeding with processing: {e}")

    # Prepare for parallel processing
    futures_map = {}
    with ThreadPoolExecutor(max_workers=5) as executor:
        # Submit tasks for each row
        for index, row in research_notes_df.iterrows():
            future = executor.submit(process_row, row, index)
            futures_map[future] = index
            
        
        total_tasks = len(futures_map) + 1
        progress = SmartProgressBar(total_tasks = total_tasks, description = "Percent complete:")

        # Collect the results as they complete
        results = []
        for future in tqdm(as_completed(futures_map)):
            try:
                index, data = future.result()
                results.append((index, data))
            except Exception as e:
                print(f"Error processing row {futures_map[future]}: {e}")
            progress.update()
        
     # Sort results by index
    results.sort(key=lambda x: x[0])

    # Flatten the results after sorting
    slide_chunks_data = []
    for _, data in results:
        slide_chunks_data.extend(data)

    # Convert slide_chunks_data into a DataFrame for easy insertion
    slide_chunks_df = pd.DataFrame(slide_chunks_data)

    # Define correct column order
    column_order = ["Topic", "Subtopic", "Slide Type", "Slide Chunk Title", "Slide Chunk"]
    slide_chunks_df = slide_chunks_df[column_order]  # Ensure correct column arrangement

    # Clear the sheet before writing new data
    slide_chunks_sheet.clear()

    # Write headers first
    slide_chunks_sheet.append_row(column_order)

    # Update the data to the slide chunks sheet
    save_to_sheet(slide_chunks_sheet, slide_chunks_df)
    progress.update()
    
    format_worksheet(slide_chunks_sheet)
    print(f"✅ Successfully updated the '{slide_chunks_sheet}' tab with {len(slide_chunks_df)} rows.")
    return slide_chunks_df