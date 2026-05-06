from modules.chain import Chain
from langsmith import traceable
import streamlit as st
from services.sheets_service import get_sheet_data_and_df, save_to_sheet, format_worksheet, clear_worksheet
from services.smart_progress_bar import SmartProgressBar
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
import re

load_dotenv()


# Prompt to use when we want the visual assingment to be flexible
generate_search_query_prompt = """You are a Search Query Generator agent specializing in the field of HVAC. Your task is to generate concise, high-quality image search queries that can be used to retrieve relevant images from a vector store of image embeddings and web-based image search engines. The retrieved images will then be used as on-screen visuals in an educational e-learning slide while the given voiceover sentence is being narrated.

You will be given a single voiceover sentence from an educational slide, along with its course information and the whole slide content for context. First, reason internally about what visual elements would need to be shown on screen for the voiceover sentence to be clearly understood. Then, based on that reasoning, generate 3 image search queries that would retrieve the most relevant visuals.
 
These are the inputs:

<course_information>
Course name: {course_name}
Topic name: {topic_name}
Subtopic name: {subtopic_name}
</course_information>

<slide_information>
Slide Type: {slide_type}
Slide title: {slide_title}

Voiceover sentence for which you need to generate search queries:
"{vo_text}"

Full slide content:
"{slide_chunk}"
</slide_information>

Visual storyboard for the full slide (for planning context and intent reference):
<visual_storyboard>
{visual_storyboard}
</visual_storyboard>

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

5. Use of Visual Storyboard Context
   - The visual storyboard represents the planned visual flow for the entire slide.
   - Refer to it to understand the intended visual ideas for the specific voiceover sentence you are processing.
   - Focus only on the storyboard planning that corresponds to this sentence.
   - Do not generate queries that are not supported by the visual storyboard.

6. Transition Slide Type:
   - Only in cases where the slide type is "Transition", you should take special care to assign a visual that is relevant to the topic and subtopic name as well.
   - The slide content of Transition slide may lack depth or details, so you should infer and plan the search queries for transition slides by taking into account the topic and subtopic name along with the storyboard.

Always provide your output strictly in the following format:

<output>

<evaluation_breakdown>

1. Voiceover Meaning
- Briefly explain, in your own words, what the voiceover sentence is communicating.

2. Query Planning
- Reason about the kinds of image searches that would best retrieve visuals to support this sentence.
- Refer to the visual storyboard to understand the intended visual ideas for the specific voiceover sentence you are processing and what visuals are required to support the sentence.
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

# Prompt to use when we want only one visual for the entire slide 
generate_search_query_prompt_for_entire_slide = """You are a Search Query Generator agent specializing in the field of HVAC. Your task is to generate concise, high-quality image search queries that can be used to retrieve relevant images from a vector store of image embeddings and web-based image search engines. The retrieved images will then be used as on-screen visuals in an educational e-learning slide as it is being narrated.

You will be given a educational slide, along with its course information for context. First, reason internally about what visual elements would need to be shown on screen for the entire slide to be clearly understood. Then, based on that reasoning, generate 4 image search queries that would retrieve the most relevant visuals.
 
These are the inputs:

<course_information>
Course name: {course_name}
Topic name: {topic_name}
Subtopic name: {subtopic_name}
</course_information>

<slide_content>
Slide type: {slide_type}
Slide title: {slide_title}
Full slide content: "{slide_chunk}"
</slide_content>

Visual storyboard for the full slide (for planning context and intent reference):
<visual_storyboard>
{visual_storyboard}
</visual_storyboard>

Instructions:

1. Base all search queries strictly on the meaning of the entire slide.
   - Do not generate queries based on general topic knowledge alone.
   - The queries should help retrieve images that are visually necessary to understand this specific slide content.

2. Generate concise, image-focused search queries.
   - Use concrete object names, components, or diagrams that are likely to appear in images.
   - Avoid abstract, instructional, or process-oriented wording.

3. Phrase queries the way images are commonly searched for or labeled.
   - Use short, keyword-based phrases.
   - Prefer phrasing typical of textbooks, technical diagrams, stock photos, or browser image searches.

4. Ensure each query adds value.
   - Queries should not be near-duplicates of each other.
   - Each query should represent a slightly different but relevant visual angle.

5. Use of Visual Storyboard Context
   - The visual storyboard represents the planned visual idea for the given slide.
   - Refer to it to understand the intended visual ideas for the entire slide.

6. Transition Slide Type:
   - Only in cases where the slide type is "Transition", you should take special care to assign a visual that is relevant to the topic and subtopic name as well.
   - The slide content of Transition slide may lack depth or details, so you should infer and plan the search queries for transition slides by taking into account the topic and subtopic name along with the storyboard.

Always provide your output strictly in the following format:

<output>

<evaluation_breakdown>

1. Slide Content Meaning: Briefly explain, in your own words, what the slide content is communicating.
2. Query Planning: Reason about the kinds of image searches that would best retrieve visuals to support this slide content.
3. Use of Visual Storyboard Context: Refer to the visual storyboard to understand the intended visual ideas for the entire slide.

</evaluation_breakdown>

(Based on your above evaluation, provide your final output of 4 search queries)

<queries>
- Query 1 text
- Query 2 text
- Query 3 text
- Query 4 text
</queries>
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
def generate_search_query_from_segment(vo_text, slide_chunk, course_name, topic_name, subtopic_name, slide_title="", visual_storyboard="", slide_type="", llm="gemini_2_5_flash_lite"):
    """
    Generate search queries for a single voiceover segment.

    :param vo_text: The voiceover sentence/segment to generate queries for.
    :param slide_chunk: The full slide content for context.
    :param course_name: The course name.
    :param topic_name: The topic name.
    :param subtopic_name: The subtopic name.
    :param slide_title: The slide title.
    :param visual_storyboard: The visual storyboard content.
    :param slide_type: The slide type from the Slide Type column (e.g. Transition, Content, Summary).
    :param llm: The language model to use.
    :return: The search query output.
    """
    # Initialize the search query agent
    search_query_agent = Chain(llm=llm, tags=["queries", "evaluation_breakdown", "output"])

    # Add the user message
    search_query_agent.add_message(
        role="user",
        content=generate_search_query_prompt.format(
            course_name=course_name,
            topic_name=topic_name,
            subtopic_name=subtopic_name,
            slide_title=slide_title,
            vo_text=vo_text,
            slide_chunk=slide_chunk,
            visual_storyboard=visual_storyboard,
            slide_type=slide_type or ""
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


def generate_search_query_for_entire_slide(slide_chunk, course_name, topic_name, subtopic_name, slide_title="", visual_storyboard="", slide_type="", llm="gemini_2_5_flash_lite"):
    """
    Generate search queries for the entire slide (not per segment).

    :param slide_chunk: The full slide content.
    :param course_name: The course name.
    :param topic_name: The topic name.
    :param subtopic_name: The subtopic name.
    :param slide_title: The slide title.
    :param visual_storyboard: The visual storyboard content.
    :param slide_type: The slide type from the Slide Type column (e.g. Transition, Content, Summary).
    :param llm: The language model to use.
    :return: The search query output.
    """
    # Initialize the search query agent
    search_query_agent = Chain(llm=llm, tags=["queries", "evaluation_breakdown", "output"])

    # # Print the formatted prompt for debugging
    # print("\n🔍 Generate Search Query Prompt Being Sent to LLM:\n")
    # print(generate_search_query_prompt_for_entire_slide.format(
    #     course_name=course_name,
    #     topic_name=topic_name,
    #     subtopic_name=subtopic_name,
    #     slide_title=slide_title,
    #     slide_chunk=slide_chunk,
    #     visual_storyboard=visual_storyboard,
    #     slide_type=slide_type or ""
    # ))
    # print("\n" + "=" * 100 + "\n")

    # Add the user message
    search_query_agent.add_message(
        role="user",
        content=generate_search_query_prompt_for_entire_slide.format(
            course_name=course_name,
            topic_name=topic_name,
            subtopic_name=subtopic_name,
            slide_title=slide_title,
            slide_chunk=slide_chunk,
            visual_storyboard=visual_storyboard,
            slide_type=slide_type or ""
        )
    )

    # Run the search query agent
    response = search_query_agent.run()

    # Print the response for debugging
    queries_output = response.get("output", "")
    print(f"\nSlide Content: {slide_chunk[:100]}...\n")
    print("📤 Generate Search Query Response from LLM (Entire Slide):\n")
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
def process_search_query_row(index, row, course_name, llm="gemini_2_5_flash_lite"):
    """
    Process a single row and generate search queries based on Visual Assignment Strategy.
    
    :param index: The row index.
    :param row: The row data.
    :param course_name: The course name.
    :param llm: The language model to use.
    :return: Tuple of (index, search_queries_text) where search_queries_text is formatted.
    """
    try:
        # Get row data
        vo_segments = str(row.get("voiceover_segment", "")).strip()
        slide_chunk = str(row.get("Slide Chunk", "")).strip()
        slide_title = str(row.get("Slide Chunk Title", "")).strip() or str(row.get("Topic", "")).strip() or "Slide"
        topic_name = str(row.get("Topic", "")).strip()
        subtopic_name = str(row.get("Subtopic", "")).strip()
        visual_storyboard = str(row.get("storyboard_planning", "")).strip()
        slide_type = str(row.get("Slide Type", "")).strip()
        if slide_type == "nan":
            slide_type = ""
        
        # Get visual assignment strategy (default to "Flexible, let the agent decide" if not found)
        visual_assignment_strategy = str(row.get("Visual Assignment Strategy", "Flexible, let the agent decide")).strip()
        if not visual_assignment_strategy or visual_assignment_strategy == "nan":
            visual_assignment_strategy = "Flexible, let the agent decide"
        
        # Handle "1 Visual for the whole Slide" - generate queries once for entire slide
        if visual_assignment_strategy == "1 Visual for the whole Slide":
            if not slide_chunk or slide_chunk == "nan":
                return index, ""
            
            # Generate queries for entire slide
            queries_output = generate_search_query_for_entire_slide(
                slide_chunk=slide_chunk,
                course_name=course_name,
                topic_name=topic_name,
                subtopic_name=subtopic_name,
                slide_title=slide_title,
                visual_storyboard=visual_storyboard,
                slide_type=slide_type,
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
            
            # Format with SEGMENT_1 marker (consistent with per-segment format)
            segment_header = "---SEGMENT_1---"
            segment_queries = [segment_header] + individual_queries
            search_queries_text = '\n'.join(segment_queries)
            return index, search_queries_text
        
        # Handle "Flexible" and "1 Visual per Sentence" - generate queries per segment
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
                    generate_search_query_from_segment,
                    segment,
                    slide_chunk,
                    course_name,
                    topic_name,
                    subtopic_name,
                    slide_title,
                    visual_storyboard,
                    slide_type,
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
                        # Extract individual queries (lines starting with "- ")
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
def run_generate_search_query_for_all_rows(sheet, llm="gemini_2_5_flash_lite", max_workers=50, selected_topics=None):
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
            topic_name = str(row.get("Topic", "")).strip()
            if selected_topics and topic_name not in selected_topics:
                continue
            vo_segments = str(row.get("voiceover_segment", "")).strip()
            slide_chunk = str(row.get("Slide Chunk", "")).strip()
            search_queries = str(row.get("search_queries", "")).strip()
            
            # Get visual assignment strategy
            visual_assignment_strategy = str(row.get("Visual Assignment Strategy", "Flexible, let the agent decide")).strip()
            if not visual_assignment_strategy or visual_assignment_strategy == "nan":
                visual_assignment_strategy = "Flexible, let the agent decide"
            
            # For "entire slide" strategy, check slide_chunk instead of voiceover_segment
            if visual_assignment_strategy == "1 Visual for the whole Slide":
                if not slide_chunk or slide_chunk == "nan":
                    continue
            else:
                # For per-segment strategies, check voiceover_segment
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
            except Exception as e:
                print(f"Error getting result for row {index}: {e}")
                # Update dataframe with error marker so row is marked as processed
                df.at[index, "search_queries"] = f"ERROR: {str(e)}"
                progress.update()

    # Save final results before validation
    save_to_sheet(worksheet, df)
    format_worksheet(worksheet)

    # Validation and retry logic
    max_retries = 3
    retry_count = 0
    
    def validate_search_queries_row(row):
        """
        Validate that search_queries matches voiceover_segment:
        - For "entire slide": just check that queries exist (no segment markers expected)
        - For per-segment: count of segments matches and no gaps in segment numbering
        
        :param row: Pandas Series with row data
        :return: Tuple (is_valid, error_message)
        """
        vo_segments_text = str(row.get("voiceover_segment", "")).strip()
        search_queries_text = str(row.get("search_queries", "")).strip()
        
        # Get visual assignment strategy
        visual_assignment_strategy = str(row.get("Visual Assignment Strategy", "Flexible, let the agent decide")).strip()
        if not visual_assignment_strategy or visual_assignment_strategy == "nan":
            visual_assignment_strategy = "Flexible, let the agent decide"
        
        # Skip validation if search_queries is empty (will be caught by retry logic)
        if not search_queries_text or search_queries_text == "nan" or search_queries_text.strip() == "":
            return False, "search_queries is empty"
        
        # Skip validation if it's an error marker
        if search_queries_text.startswith("ERROR:"):
            return False, "search_queries contains error marker"
        
        # For "entire slide" strategy, check that SEGMENT_1 exists
        if visual_assignment_strategy == "1 Visual for the whole Slide":
            # Check for SEGMENT_1 marker
            if "---SEGMENT_1---" not in search_queries_text:
                return False, "SEGMENT_1 marker not found for entire slide"
            # Check that there are some queries (at least one non-empty line after the marker)
            query_lines = [q.strip() for q in search_queries_text.split('\n') if q.strip() and not q.strip().startswith('---SEGMENT')]
            if not query_lines:
                return False, "No queries found for entire slide"
            return True, None
        
        # For per-segment strategies, validate segment markers
        # Skip validation if voiceover_segment is empty
        if not vo_segments_text or vo_segments_text == "nan":
            return True, None
        
        # Count segments in voiceover_segment (split by newline)
        vo_segments = [seg.strip() for seg in vo_segments_text.split('\n') if seg.strip()]
        expected_count = len(vo_segments)
        
        # Parse segment numbers from search_queries
        segment_pattern = r'---SEGMENT_(\d+)---'
        segment_numbers = [int(match) for match in re.findall(segment_pattern, search_queries_text)]
        
        if not segment_numbers:
            return False, "No segment markers found in search_queries"
        
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
    
    while retry_count < max_retries:
        # Reload dataframe to get latest state
        worksheet, df = get_sheet_data_and_df(sheet, worksheet_name)
        
        # Validate all rows and find invalid ones
        invalid_rows = []
        for index, row in df.iterrows():
            topic_name = str(row.get("Topic", "")).strip()
            if selected_topics and topic_name not in selected_topics:
                continue
            is_valid, error_msg = validate_search_queries_row(row)
            if not is_valid:
                invalid_rows.append((index, row, error_msg))
        
        if not invalid_rows:
            break
        
        retry_count += 1
        print(f"\n⚠️ Found {len(invalid_rows)} rows with invalid search_queries. Retrying (attempt {retry_count}/{max_retries})...")
        for idx, row, error in invalid_rows[:3]:  # Show first 3 errors
            print(f"  Row {idx}: {error}")
        
        # Clear search_queries for invalid rows
        for index, row, error_msg in invalid_rows:
            df.at[index, "search_queries"] = ""
        
        # Save cleared state
        save_to_sheet(worksheet, df)
        format_worksheet(worksheet)
        
        # Retry processing invalid rows
        futures_map = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for index, row, error_msg in invalid_rows:
                topic_name = str(row.get("Topic", "")).strip()
                if selected_topics and topic_name not in selected_topics:
                    continue
                future = executor.submit(process_search_query_row, index, row, course_name, llm)
                futures_map[future] = index
            
            # Collect results
            for future in as_completed(futures_map):
                index = futures_map[future]
                try:
                    row_index, search_queries_text = future.result()
                    df.at[row_index, "search_queries"] = search_queries_text
                except Exception as e:
                    print(f"Error getting result for row {index} on retry: {e}")
                    df.at[index, "search_queries"] = f"ERROR: {str(e)}"
        
        # Save after retry
        save_to_sheet(worksheet, df)
        format_worksheet(worksheet)
    
    if retry_count > 0:
        # Check final state
        worksheet, df = get_sheet_data_and_df(sheet, worksheet_name)
        final_invalid = []
        for index, row in df.iterrows():
            is_valid, error_msg = validate_search_queries_row(row)
            if not is_valid:
                final_invalid.append((index, error_msg))
        
        if final_invalid:
            print(f"⚠️ After {retry_count} retry attempt(s), {len(final_invalid)} rows still have invalid search_queries.")
            for idx, error in final_invalid[:5]:  # Show first 5
                print(f"  Row {idx}: {error}")
        else:
            print(f"✅ All rows validated after {retry_count} retry attempt(s).")
    
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