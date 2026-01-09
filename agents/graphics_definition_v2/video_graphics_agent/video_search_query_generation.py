
from modules.chain import Chain
from langsmith import traceable
import streamlit as st
from services.sheets_service import get_sheet_data_and_df, save_to_sheet, format_worksheet, clear_worksheet
from services.smart_progress_bar import SmartProgressBar
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
import re

load_dotenv()


video_search_query_generator_prompt = """You are a search query generation agent tasked with converting slide narration sentences into multiple optimized search queries that will be used for retrieving relevant HVAC instructional YouTube videos from a video-embedding vector database.

These are the inputs:

<course_information>
Course Name: {course_name}
</course_information>

<slide_context>
Topic: {topic_name}
Subtopic: {subtopic_name}
Slide Title: {slide_title}
Whole Slide Content: {whole_slide_content}
</slide_context>

This is the specific slide sentence for which you need to generate search queries:
<slide_sentence>
{slide_sentence}
</slide_sentence>

Follow the below guidelines while generating the search queries:

1. Sentence Understanding & Context Resolution
- Interpret the slide sentence using the topic, subtopic, slide title, and whole slide content.
- Resolve any pronouns or vague references into explicit HVAC terms.
- Treat the sentence as part of a cohesive slide, not isolated text.

2. Sentence Decomposition into Concepts
- Identify distinct conceptual or visualizable parts within the sentence.
- Each part should represent a component, action, process, condition, etc.
- Do not introduce concepts that are not implied by the sentence or slide context.

3. Search Query Generation
- Generate 2–4 search queries for the slide sentence.
- Include 1 query representing the full sentence meaning and 1–3 queries targeting individual parts or concepts within the sentence.
- Keep queries concise and focused on retrievable concepts.
- Each query must be suitable for independent video retrieval.

4. Query Diversity and Non-Overlap
- Ensure each query targets a different retrieval angle.
- Avoid overlapping phrasing or repeated intent across queries.
- Do not generate near-duplicate queries.

Always provide your output strictly in the following format:

<output>

<evaluation_breakdown>

Use this section to reason step by step before generating the final search queries:

1. Sentence Interpretation: Briefly state what the slide sentence is teaching or conveying. Indentify which HVAC component, action, process, condition, etc. is the primary focus. Resolve any pronouns or implicit references present in the slide sentence.
2. Identifying Conceptual Parts: Identify and list the distinct concepts or visualizable parts within the sentence.
3. Query Planning: Identify which query will represent the full sentence meaning and which queries will target individual concepts or parts within the sentence. Confirm that the total number of planned queries is 2–4 and that each serves a distinct retrieval purpose.

</evaluation_breakdown>

(Based on your above evaluation, provide your final output of search queries)

<queries>
List your final 2-4 search queries for the slide sentence here, one per line.
</queries>

</output>
"""


@traceable(
    metadata={
        "agent_name": "graphics_definition_v2",
        "step_name": "Video Search Query Generation",
        "function_name": "generate_video_search_query_from_segment",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous")
    }
)
def generate_video_search_query_from_segment(slide_sentence, course_name, topic_name, subtopic_name, slide_title, whole_slide_content, llm="gemini_3_flash_thinking"):
    """
    Generate video search queries for a single slide sentence.

    :param slide_sentence: The slide sentence to generate search queries for.
    :param course_name: The course name.
    :param topic_name: The topic name.
    :param subtopic_name: The subtopic name.
    :param slide_title: The slide title.
    :param whole_slide_content: The full slide content for context.
    :param llm: The language model to use.
    :return: The search query output.
    """
    # Initialize the search query agent
    search_query_agent = Chain(llm=llm, tags=["queries", "evaluation_breakdown", "output"])
    
    # # Print the prompt for debugging
    # print("\n🔍 Video Search Query Generator Prompt Being Sent to LLM:\n")
    # print(video_search_query_generator_prompt.format(
    #     course_name=course_name,
    #     topic_name=topic_name,
    #     subtopic_name=subtopic_name,
    #     slide_title=slide_title,
    #     whole_slide_content=whole_slide_content,
    #     slide_sentence=slide_sentence
    # ))
    # print("\n" + "=" * 100 + "\n")

    # Add the user message  
    search_query_agent.add_message(
        role="user",
        content=video_search_query_generator_prompt.format(
            course_name=course_name,
            topic_name=topic_name,
            subtopic_name=subtopic_name,
            slide_title=slide_title,
            whole_slide_content=whole_slide_content,
            slide_sentence=slide_sentence
        )
    )
    # Run the search query agent    
    response = search_query_agent.run()

    # Print the response for debugging
    queries_output = response.get("output", "")
    print(f"\nSegment: {slide_sentence}\n")
    print("📤 Generate Search Query Response from LLM:\n")
    print(queries_output)
    print("\n" + "=" * 100 + "\n")

    return queries_output


@traceable(
    metadata={
        "agent_name": "graphics_definition_v2",
        "step_name": "Video Search Query Generation",
        "function_name": "process_video_search_query_row",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous")
    }
)
def process_video_search_query_row(index, row, course_name, llm="gemini_3_flash_thinking"):
    """
    Process a single row and generate video search queries for all segments.
    
    :param index: The row index.
    :param row: The row data.
    :param course_name: The course name.
    :param llm: The language model to use.
    :return: Tuple of (index, video_search_queries_text) where video_search_queries_text is formatted with segment markers.
    """
    try:
        # Get row data
        vo_segments = str(row.get("voiceover_segment", "")).strip()
        slide_chunk = str(row.get("Slide Chunk", "")).strip()
        slide_title = str(row.get("Slide Chunk Title", "")).strip() or str(row.get("Topic", "")).strip() or "Slide"
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
        
        # Execute segments in parallel
        with ThreadPoolExecutor(max_workers=len(segments)) as executor:
            # Submit all segments
            futures = {
                executor.submit(
                    generate_video_search_query_from_segment,
                    segment,
                    course_name,
                    topic_name,
                    subtopic_name,
                    slide_title,
                    slide_chunk,  # whole_slide_content
                    llm
                ): (segment_idx, segment)
                for segment_idx, segment in enumerate(segments, start=1)
            }
            
            # Collect results as they complete
            segment_results = {}
            for future in as_completed(futures):
                segment_idx, segment = futures[future]
                try:
                    queries_output = future.result()
                    
                    # Extract queries from XML format
                    query_matches = re.findall(r'<queries>(.*?)</queries>', queries_output, re.DOTALL)
                    if query_matches:
                        queries_text = query_matches[0].strip()
                        # Extract individual queries (lines starting with "- " or just plain lines)
                        individual_queries = [q.strip()[2:] if q.strip().startswith("- ") else q.strip() 
                                             for q in queries_text.split('\n') if q.strip()]
                    else:
                        # Fallback: try to extract queries without XML tags
                        individual_queries = [q.strip() for q in queries_output.split('\n') 
                                            if q.strip() and not q.strip().startswith('<')]
                    
                    # Store result with segment index for ordering
                    segment_results[segment_idx] = (f"---SEGMENT_{segment_idx}---", individual_queries)
                except Exception as e:
                    print(f"❌ Error processing segment {segment_idx} (\"{segment[:50]}...\"): {e}")
            
            # Format results in order (by segment_idx)
            for segment_idx in sorted(segment_results.keys()):
                segment_header, individual_queries = segment_results[segment_idx]
                segment_queries = [segment_header] + individual_queries
                all_queries_formatted.append('\n'.join(segment_queries))
        
        # Join all segments with double newline
        video_search_queries_text = '\n\n'.join(all_queries_formatted)
        
        return index, video_search_queries_text
    except Exception as e:
        print(f"Error processing row {index}: {e}")
        return index, ""


def validate_video_search_query_row(row):
    """
    Validate that video_search_query matches voiceover_segment:
    - Count of segments matches
    - No gaps in segment numbering (must be sequential starting from 1)
    
    :param row: Pandas Series with row data
    :return: Tuple (is_valid, error_message)
    """
    vo_segments_text = str(row.get("voiceover_segment", "")).strip()
    video_search_queries_text = str(row.get("video_search_query", "")).strip()
    
    # Skip validation if voiceover_segment is empty
    if not vo_segments_text or vo_segments_text == "nan":
        return True, None
    
    # Count segments in voiceover_segment (split by newline)
    vo_segments = [seg.strip() for seg in vo_segments_text.split('\n') if seg.strip()]
    expected_count = len(vo_segments)
    
    # Skip validation if video_search_query is empty (will be caught by retry logic)
    if not video_search_queries_text or video_search_queries_text == "nan" or video_search_queries_text.strip() == "":
        return False, "video_search_query is empty"
    
    # Skip validation if it's an error marker
    if video_search_queries_text.startswith("ERROR:"):
        return False, "video_search_query contains error marker"
    
    # Parse segment numbers from video_search_query
    segment_pattern = r'---SEGMENT_(\d+)---'
    segment_numbers = [int(match) for match in re.findall(segment_pattern, video_search_queries_text)]
    
    if not segment_numbers:
        return False, "No segment markers found in video_search_query"
    
    # Check count matches
    actual_count = len(segment_numbers)
    if actual_count != expected_count:
        return False, f"Segment count mismatch: expected {expected_count}, found {actual_count}"
    
    # Check for gaps (must be sequential starting from 1)
    segment_numbers_sorted = sorted(segment_numbers)
    expected_sequence = list(range(1, expected_count + 1))
    
    if segment_numbers_sorted != expected_sequence:
        missing = set(expected_sequence) - set(segment_numbers_sorted)
        extra = set(segment_numbers_sorted) - set(expected_sequence)
        error_parts = []
        if missing:
            error_parts.append(f"missing segments: {sorted(missing)}")
        if extra:
            error_parts.append(f"extra segments: {sorted(extra)}")
        return False, f"Segment numbering gap: {', '.join(error_parts)}"
    
    return True, None


@traceable(
    metadata={
        "agent_name": "graphics_definition_v2",
        "step_name": "Video Search Query Generation",
        "function_name": "run_generate_video_search_query_for_all_rows",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous")
    }
)
def run_generate_video_search_query_for_all_rows(sheet, llm="gemini_3_flash_thinking", max_workers=5):
    """
    Generate video search queries for all segments in all rows in the Slide Chunks sheet.

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

    # Ensure video_search_query column exists
    if "video_search_query" not in df.columns:
        df["video_search_query"] = ""

    # Prepare for parallel processing - only process rows that need processing
    futures_map = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks first
        for index, row in df.iterrows():
            vo_segments = str(row.get("voiceover_segment", "")).strip()
            video_search_queries = str(row.get("video_search_query", "")).strip()
            
            # Skip if voiceover_segment is empty
            if not vo_segments or vo_segments == "nan":
                continue
            
            # Skip if video_search_query is already filled
            if video_search_queries and video_search_queries != "nan":
                continue
            
            # Submit task for processing
            future = executor.submit(process_video_search_query_row, index, row, course_name, llm)
            futures_map[future] = index

        # If no rows to process, return early
        if not futures_map:
            print("All rows already processed or no valid voiceover segments found.")
            return

        # Initialize progress tracker
        total_tasks = len(futures_map)
        progress = SmartProgressBar(
            total_tasks=total_tasks,
            description="Generating video search queries",
            save_interval=5
        )

        # Collect results as they complete
        for future in as_completed(futures_map):
            index = futures_map[future]
            try:
                row_index, video_search_queries_text = future.result()
                
                # Update dataframe
                df.at[row_index, "video_search_query"] = video_search_queries_text
                
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
                df.at[index, "video_search_query"] = f"ERROR: {str(e)}"
                progress.update()

    # Validation and retry logic
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        # Reload dataframe to get latest state
        worksheet, df = get_sheet_data_and_df(sheet, worksheet_name)
        
        # Validate all rows and find invalid ones
        invalid_rows = []
        for index, row in df.iterrows():
            is_valid, error_msg = validate_video_search_query_row(row)
            if not is_valid:
                invalid_rows.append((index, row, error_msg))
        
        if not invalid_rows:
            break
        
        retry_count += 1
        print(f"\n⚠️ Found {len(invalid_rows)} rows with invalid video_search_query. Retrying (attempt {retry_count}/{max_retries})...")
        for idx, row, error in invalid_rows[:3]:  # Show first 3 errors
            print(f"  Row {idx}: {error}")
        
        # Clear video_search_query for invalid rows
        for index, row, error_msg in invalid_rows:
            df.at[index, "video_search_query"] = ""
        
        # Save cleared state
        save_to_sheet(worksheet, df)
        format_worksheet(worksheet)
        
        # Retry processing invalid rows
        futures_map = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for index, row, error_msg in invalid_rows:
                future = executor.submit(process_video_search_query_row, index, row, course_name, llm)
                futures_map[future] = index
            
            # Collect results
            for future in as_completed(futures_map):
                index = futures_map[future]
                try:
                    row_index, video_search_queries_text = future.result()
                    df.at[row_index, "video_search_query"] = video_search_queries_text
                except Exception as e:
                    print(f"Error getting result for row {index} on retry: {e}")
                    df.at[index, "video_search_query"] = f"ERROR: {str(e)}"
        
        # Save after retry
        save_to_sheet(worksheet, df)
        format_worksheet(worksheet)
    
    if retry_count > 0:
        # Check final state
        worksheet, df = get_sheet_data_and_df(sheet, worksheet_name)
        final_invalid = []
        for index, row in df.iterrows():
            is_valid, error_msg = validate_video_search_query_row(row)
            if not is_valid:
                final_invalid.append((index, error_msg))
        
        if final_invalid:
            print(f"⚠️ After {retry_count} retry attempt(s), {len(final_invalid)} rows still have invalid video_search_query.")
            for idx, error in final_invalid[:5]:  # Show first 5
                print(f"  Row {idx}: {error}")
        else:
            print(f"✅ All rows validated after {retry_count} retry attempt(s).")
    
    # Final save to sheet
    print('All video search queries generated. Saving final DataFrame to sheet.')
    save_to_sheet(worksheet, df)
    format_worksheet(worksheet)
    print("✅ Video search query generation complete and saved to sheet.")


def delete_video_search_queries(sheet):
    """
    Remove the 'video_search_query' column from the specified worksheet.
    
    :param sheet: The gspread sheet object.
    :return: None
    """
    worksheet_name = "Slide Chunks"
    ws, df = get_sheet_data_and_df(sheet, worksheet_name)
    if "video_search_query" in df.columns:
        df = df.drop(columns=["video_search_query"])
        clear_worksheet(ws)
        save_to_sheet(ws, df)
        print(f"🗑️ Deleted 'video_search_query' column from '{worksheet_name}' worksheet")
    else:
        print(f"ℹ️ 'video_search_query' column does not exist in '{worksheet_name}' worksheet")