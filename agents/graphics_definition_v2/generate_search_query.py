
from modules.chain import Chain
from langsmith import traceable
import streamlit as st
from services.sheets_service import get_sheet_data_and_df, save_to_sheet, format_worksheet, clear_worksheet
from services.smart_progress_bar import SmartProgressBar
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
from services.sheets_service import get_sheet_data_and_df, save_to_sheet, format_worksheet, clear_worksheet
from services.smart_progress_bar import SmartProgressBar
from concurrent.futures import ThreadPoolExecutor, as_completed
import re


generate_search_query_prompt = """You are a Search Query Generator agent specializing in the field of HVAC. Your task is to generate concise, high-quality image search queries that can be used to retrieve relevant images from a vector store of image embeddings and web-based image search engines. The retrieved images will then be used as on-screen visuals in an educational e-learning slide while the given voiceover sentence is being narrated.

You will be given a single voiceover sentence from an educational slide, along with its course information and the whole slide content for context. First, reason internally about what visual elements would need to be shown on screen for the voiceover sentence to be clearly understood. Then, based on that reasoning, generate 3 image search queries that would retrieve the most relevant visuals.
 
These are the inputs:

Course Name: {course_name}

Topic Name: {topic_name}

Subtopic Name: {subtopic_name}

Voiceover sentence for which you need to generate search queries:
"{vo_text}"

Full slide content:
{slide_chunk}

Instructions:

1. Base all search queries strictly on the meaning of the voiceover sentence.
   - Do not generate queries based on general topic knowledge alone.
   - The queries should help retrieve images that are visually necessary to understand this specific sentence.

2. Generate concise, image-focused search queries.
   - Use concrete object names, components, or diagrams that are likely to appear in images.
   - Avoid abstract, instructional, or process-oriented wording.

3. Phrase queries the way images are commonly searched for or labeled.
   - Use short, keyword-based phrases.
   - Prefer phrasing typical of textbooks, technical diagrams, stock photos, or browser image searches.

4. Ensure each query adds value.
   - Queries should not be near-duplicates of each other.
   - Each query should represent a slightly different but relevant visual angle.

Always provide your output strictly in the following format:

<output>

<evaluation_breakdown>

1. Voiceover Meaning
- Briefly explain, in your own words, what the voiceover sentence is communicating.

2. Query Planning
- Reason about the kinds of image searches that would best retrieve visuals to support this sentence.
- Consider how such images are typically searched for or labeled.

</evaluation_breakdown>

(Based on your above evaluation, provide your final output of search queries)

<queries>
- Query 1 text
- Query 2 text
- Query 3 text
</queries>

</output>
"""


@traceable(
    metadata={
        "agent_name": "graphics_definition_v2",
        "step_name": "Generate Search Query",
        "function_name": "generate_search_query_from_segment",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous")
    }
)
def generate_search_query_from_segment(vo_text, slide_chunk, course_name, topic_name, subtopic_name, llm="gemini_2_5_pro"):
    """
    Generate search queries for a single voiceover segment.

    :param vo_text: The voiceover sentence/segment to generate queries for.
    :param slide_chunk: The full slide content for context.
    :param course_name: The course name.
    :param topic_name: The topic name.
    :param subtopic_name: The subtopic name.
    :param llm: The language model to use.
    :return: The search query output.
    """
    # Initialize the search query agent
    search_query_agent = Chain(llm=llm, tags=["queries", "evaluation_breakdown", "output"])

    # # Print the formatted prompt for debugging
    # print("\n🔍 Generate Search Query Prompt Being Sent to LLM:\n")
    # print(generate_search_query_prompt.format(
    #     course_name=course_name,
    #     topic_name=topic_name,
    #     subtopic_name=subtopic_name,
    #     vo_text=vo_text,
    #     slide_chunk=slide_chunk
    # ))
    # print("\n" + "=" * 100 + "\n")

    # Add the user message
    search_query_agent.add_message(
        role="user",
        content=generate_search_query_prompt.format(
            course_name=course_name,
            topic_name=topic_name,
            subtopic_name=subtopic_name,
            vo_text=vo_text,
            slide_chunk=slide_chunk
        )
    )

    # Run the search query agent
    response = search_query_agent.run()

    # Print the response for debugging
    queries_output = response.get("output", "")
    print(f"\nSegment: {vo_text}\n")
    print("📤 Generate Search Query Response from LLM:\n")
    print(queries_output)
    print("\n" + "=" * 100 + "\n")

    return queries_output


@traceable(
    metadata={
        "agent_name": "graphics_definition_v2",
        "step_name": "Generate Search Query",
        "function_name": "process_search_query_row",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous")
    }
)
def process_search_query_row(index, row, course_name, llm="gemini_2_5_pro"):
    """
    Process a single row and generate search queries for all segments.
    
    :param index: The row index.
    :param row: The row data.
    :param course_name: The course name.
    :param llm: The language model to use.
    :return: Tuple of (index, search_queries_text) where search_queries_text is formatted with segment markers.
    """
    try:
        # Get row data
        vo_segments = str(row.get("voiceover_segment", "")).strip()
        slide_chunk = str(row.get("Slide Chunk", "")).strip()
        topic_name = str(row.get("Topic", "")).strip()
        subtopic_name = str(row.get("Subtopic", "")).strip()
        
        # Skip if no segments
        if not vo_segments or vo_segments == "nan":
            return index, ""
        
        # Split segments by newline
        segments = [seg.strip() for seg in vo_segments.split('\n') if seg.strip()]
        
        if not segments:
            return index, ""
        
        # Generate queries for each segment
        all_queries_formatted = []
        for segment_idx, segment in enumerate(segments, start=1):
            # Generate queries for this segment
            queries_output = generate_search_query_from_segment(
                vo_text=segment,
                slide_chunk=slide_chunk,
                course_name=course_name,
                topic_name=topic_name,
                subtopic_name=subtopic_name,
                llm=llm
            )
            
            # Extract queries from XML format
            query_matches = re.findall(r'<queries>(.*?)</queries>', queries_output, re.DOTALL)
            if query_matches:
                queries_text = query_matches[0].strip()
                # Extract individual queries (lines starting with "- ")
                individual_queries = [q.strip()[2:] if q.strip().startswith("- ") else q.strip() 
                                     for q in queries_text.split('\n') if q.strip()]
            else:
                # Fallback: try to extract queries without XML tags
                individual_queries = [q.strip() for q in queries_output.split('\n') 
                                    if q.strip() and not q.strip().startswith('<')]
            
            # Format with segment marker
            segment_queries = [f"---SEGMENT_{segment_idx}---"] + individual_queries
            all_queries_formatted.append('\n'.join(segment_queries))
        
        # Join all segments with double newline
        search_queries_text = '\n\n'.join(all_queries_formatted)
        
        return index, search_queries_text
    except Exception as e:
        print(f"Error processing row {index}: {e}")
        return index, ""


@traceable(
    metadata={
        "agent_name": "graphics_definition_v2",
        "step_name": "Generate Search Query",
        "function_name": "run_generate_search_query_for_all_rows",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous")
    }
)
def run_generate_search_query_for_all_rows(sheet, llm="gemini_2_5_pro", max_workers=5):
    """
    Generate search queries for all segments in all rows in the Slide Chunks sheet.

    :param sheet: The gspread sheet object.
    :param llm: The language model to use.
    :param max_workers: Number of parallel workers (default 5).
    :return: None
    """
    worksheet_name = "Slide Chunks"
    
    # Fetch course info
    _, course_info_df = get_sheet_data_and_df(sheet, "Course info")
    course_name = course_info_df.loc[0, "Course Name"]
    
    # Load the worksheet and DataFrame
    worksheet, df = get_sheet_data_and_df(sheet, worksheet_name)

    # Ensure search_queries column exists
    if "search_queries" not in df.columns:
        df["search_queries"] = ""

    # Prepare for parallel processing - only process rows that need processing
    futures_map = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks first
        for index, row in df.iterrows():
            vo_segments = str(row.get("voiceover_segment", "")).strip()
            search_queries = str(row.get("search_queries", "")).strip()
            
            # Skip if voiceover_segment is empty
            if not vo_segments or vo_segments == "nan":
                continue
            
            # Skip if search_queries is already filled
            if search_queries and search_queries != "nan":
                continue
            
            # Submit task for processing
            future = executor.submit(process_search_query_row, index, row, course_name, llm)
            futures_map[future] = index

        # If no rows to process, return early
        if not futures_map:
            print("All rows already processed or no valid voiceover segments found.")
            return

        # Initialize progress tracker
        total_tasks = len(futures_map)
        progress = SmartProgressBar(
            total_tasks=total_tasks,
            description="Generating search queries",
            save_interval=5
        )

        # Collect results as they complete
        for future in as_completed(futures_map):
            index = futures_map[future]
            try:
                row_index, search_queries_text = future.result()
                
                # Update dataframe
                df.at[row_index, "search_queries"] = search_queries_text
                
                # Update progress
                progress.update()
                
                # Save every 5 rows
                if progress.should_save():
                    print(f'Saving partial progress to sheet after {progress.completed_count} tasks completed.')
                    save_to_sheet(worksheet, df)
                    format_worksheet(worksheet)
            except Exception as e:
                print(f"Error getting result for row {index}: {e}")
                progress.update()

    # Final save to sheet
    print('All search queries generated. Saving final DataFrame to sheet.')
    save_to_sheet(worksheet, df)
    format_worksheet(worksheet)
    print("✅ Search query generation complete and saved to sheet.")


def delete_search_queries(sheet):
    """
    Remove the 'search_queries' column from the specified worksheet.
    
    :param sheet: The gspread sheet object.
    :return: None
    """
    worksheet_name = "Slide Chunks"
    ws, df = get_sheet_data_and_df(sheet, worksheet_name)
    if "search_queries" in df.columns:
        df = df.drop(columns=["search_queries"])
        clear_worksheet(ws)
        save_to_sheet(ws, df)
        print(f"🗑️ Deleted 'search_queries' column from '{worksheet_name}' worksheet")
    else:
        print(f"ℹ️ 'search_queries' column does not exist in '{worksheet_name}' worksheet")