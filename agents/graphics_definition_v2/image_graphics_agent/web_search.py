from langsmith import traceable
import streamlit as st
from services.sheets_service import get_sheet_data_and_df, save_to_sheet, format_worksheet, clear_worksheet
from services.smart_progress_bar import SmartProgressBar
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
import re
from agents.vector_store_image_search.web_image_search_tool import web_image_search_tool

load_dotenv()

# Configuration constants
web_search_k = 5  # Number of results to retrieve per web image search query


@traceable(
    metadata={
        "agent_name": "graphics_definition_v2",
        "step_name": "Web Search",
        "function_name": "execute_web_search_for_query",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous")
    }
)
def execute_web_search_for_query(query, k=web_search_k):
    """
    Execute web search for a single query and return reference data with URLs.
    
    :param query: Search query string
    :param k: Number of results to retrieve
    :return: List of reference data dictionaries with url
    """
    print(f"🌐 Executing web search for query: \"{query}\"")
    
    # Execute web search
    try:
        web_results_raw = web_image_search_tool(
            query=query,
            k=k
        )
        print(f"Raw results from web search: {len(web_results_raw)} items")
    except Exception as e:
        print(f"❌ Web search error: {str(e)}")
        return []
    
    # Convert web search results to reference data format
    reference_data_list = []
    for result in web_results_raw:
        metadata = result.get("metadata", {})
        
        # Get URL from metadata (source_url field)
        url = metadata.get("source_url", "")
        
        # Get title from metadata (name field)
        title = metadata.get("name", "Untitled")
        
        if url and url.strip():
            reference_data = {
                "url": url,
                "title": title,
            }
            reference_data_list.append(reference_data)
    
    print(f"✅ Found {len(reference_data_list)} results with URLs")
    
    return reference_data_list


def parse_search_queries_column(search_queries_text):
    """
    Parse the search_queries column to extract segments and their queries.
    
    Format:
    ---SEGMENT_1---
    query1
    query2
    query3
    
    ---SEGMENT_2---
    query1
    query2
    query3
    
    :param search_queries_text: The search_queries column content
    :return: List of tuples (segment_number, list_of_queries)
    """
    if not search_queries_text or search_queries_text.strip() == "" or search_queries_text == "nan":
        return []
    
    segments = []
    # Split by segment markers
    segment_pattern = r'---SEGMENT_(\d+)---'
    parts = re.split(segment_pattern, search_queries_text)
    
    # parts[0] is text before first segment (usually empty)
    # Then alternating: segment_num, segment_content, segment_num, segment_content, ...
    for i in range(1, len(parts), 2):
        if i + 1 < len(parts):
            segment_num = int(parts[i])
            segment_content = parts[i + 1].strip()
            
            # Extract queries (each line is a query)
            queries = [q.strip() for q in segment_content.split('\n') if q.strip()]
            
            if queries:
                segments.append((segment_num, queries))
    
    return segments


@traceable(
    metadata={
        "agent_name": "graphics_definition_v2",
        "step_name": "Web Search",
        "function_name": "process_web_search_segment",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous")
    }
)
def process_web_search_segment(segment_num, queries, k=web_search_k):
    """
    Process a single segment: execute queries in parallel, deduplicate, format output.
    
    :param segment_num: Segment number
    :param queries: List of search queries for this segment
    :param k: Number of results per query
    :return: Tuple of (segment_num, segment_output_string) or (segment_num, None) if no results
    """
    print(f"\n{'─'*45}")
    print(f"📦 Processing SEGMENT_{segment_num} with {len(queries)} queries")
    print(f"{'─'*45}")
    
    # Execute all queries for this segment
    all_results_for_segment = []
    seen_urls = set()  # Deduplicate within segment by URL
    
    # Filter out empty queries
    valid_queries = [(idx, q.strip()) for idx, q in enumerate(queries, 1) if q.strip()]
    
    if not valid_queries:
        return segment_num, None
    
    # Execute queries in parallel
    with ThreadPoolExecutor(max_workers=len(valid_queries)) as executor:
        # Submit all queries
        futures = {
            executor.submit(execute_web_search_for_query, query, k): (query_idx, query)
            for query_idx, query in valid_queries
        }
        
        # Collect results as they complete
        for future in as_completed(futures):
            query_idx, query = futures[future]
            try:
                print(f"🔍 Query {query_idx}: \"{query}\"")
                results = future.result()
                print(f"✅ Query {query_idx} returned {len(results)} results")
                
                # Deduplicate by URL within this segment
                for ref in results:
                    url = ref.get("url", "").strip()
                    title = ref.get("title", "Untitled")
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        all_results_for_segment.append({"url": url, "title": title})
            except Exception as e:
                print(f"❌ Error executing query {query_idx} (\"{query}\"): {e}")
    
    # Format segment results with title and URL
    segment_items = [f"Title: {ref['title']} | URL: {ref['url']}" for ref in all_results_for_segment]
    
    print(f"✅ SEGMENT_{segment_num}: Found {len(segment_items)} unique results")
    
    # Format segment results
    if segment_items:
        segment_output = [f"---SEGMENT_{segment_num}---"] + segment_items
        return segment_num, '\n'.join(segment_output)
    return segment_num, None


@traceable(
    metadata={
        "agent_name": "graphics_definition_v2",
        "step_name": "Web Search",
        "function_name": "process_web_search_row",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous")
    }
)
def process_web_search_row(index, row, k=web_search_k):
    """
    Process a single row: parse search queries, execute web searches for each segment, deduplicate, format output.
    
    :param index: Row index
    :param row: Pandas Series with row data
    :param k: Number of results per query
    :return: Tuple of (index, web_results_text)
    """
    try:
        search_queries_text = str(row.get("search_queries", "")).strip()
        
        # Skip if search_queries is empty
        if not search_queries_text or search_queries_text == "nan":
            return index, ""
        
        # Parse segments and queries
        segments = parse_search_queries_column(search_queries_text)
        
        if not segments:
            return index, ""
        
        # Execute all segments in parallel
        with ThreadPoolExecutor(max_workers=len(segments)) as executor:
            # Submit all segments
            futures = {
                executor.submit(process_web_search_segment, segment_num, queries, k): segment_num
                for segment_num, queries in segments
            }
            
            # Collect results as they complete
            segment_results = {}
            for future in as_completed(futures):
                segment_num = futures[future]
                try:
                    seg_num, segment_output = future.result()
                    if segment_output:
                        segment_results[seg_num] = segment_output
                except Exception as e:
                    print(f"❌ Error processing segment {segment_num}: {e}")
        
        # Format results in order (by segment_num)
        all_segment_results = []
        for segment_num in sorted(segment_results.keys()):
            all_segment_results.append(segment_results[segment_num])
        
        # Join all segments with double newline
        web_results_text = '\n\n'.join(all_segment_results)
        
        return index, web_results_text
        
    except Exception as e:
        print(f"Error processing row {index}: {e}")
        return index, ""


def validate_web_search_row(row):
    """
    Validate that web_results matches search_queries:
    - Row is not empty
    - All segments from search_queries have results
    - No gaps in segment numbering
    
    :param row: Pandas Series with row data
    :return: Tuple (is_valid, error_message)
    """
    search_queries_text = str(row.get("search_queries", "")).strip()
    web_results_text = str(row.get("web_results", "")).strip()
    
    # Skip validation if search_queries is empty
    if not search_queries_text or search_queries_text == "nan":
        return True, None
    
    # Check if web_results is empty
    if not web_results_text or web_results_text == "nan" or web_results_text.strip() == "":
        return False, "web_results is empty"
    
    # Check if it's an error marker
    if web_results_text.startswith("ERROR:"):
        return False, "web_results contains error marker"
    
    # Parse expected segments from search_queries
    expected_segments = parse_search_queries_column(search_queries_text)
    if not expected_segments:
        return True, None  # No segments to validate
    
    expected_segment_nums = sorted([seg_num for seg_num, _ in expected_segments])
    expected_count = len(expected_segment_nums)
    
    # Parse actual segments from web_results
    segment_pattern = r'---SEGMENT_(\d+)---'
    actual_segment_nums = [int(match) for match in re.findall(segment_pattern, web_results_text)]
    
    if not actual_segment_nums:
        return False, "No segment markers found in web_results"
    
    # Check count matches
    actual_count = len(actual_segment_nums)
    if actual_count != expected_count:
        return False, f"Segment count mismatch: expected {expected_count}, found {actual_count}"
    
    # Check for gaps (must match expected segments)
    actual_segment_nums_sorted = sorted(actual_segment_nums)
    if actual_segment_nums_sorted != expected_segment_nums:
        missing = set(expected_segment_nums) - set(actual_segment_nums_sorted)
        extra = set(actual_segment_nums_sorted) - set(expected_segment_nums)
        error_parts = []
        if missing:
            error_parts.append(f"missing segments: {sorted(missing)}")
        if extra:
            error_parts.append(f"extra segments: {sorted(extra)}")
        return False, f"Segment mismatch: {', '.join(error_parts)}"
    
    return True, None


@traceable(
    metadata={
        "agent_name": "graphics_definition_v2",
        "step_name": "Web Search",
        "function_name": "run_web_search_for_all_rows",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous")
    }
)
def run_web_search_for_all_rows(sheet, k=web_search_k, max_workers=5):
    """
    Execute web search for all rows in the Slide Chunks sheet.
    
    :param sheet: The gspread sheet object.
    :param k: Number of results per query (default from settings).
    :param max_workers: Number of parallel workers (default 5).
    :return: None
    """
    worksheet_name = "Slide Chunks"
    
    # Load the worksheet and DataFrame
    worksheet, df = get_sheet_data_and_df(sheet, worksheet_name)
    
    # Ensure web_results column exists
    if "web_results" not in df.columns:
        df["web_results"] = ""
    
    # Prepare for parallel processing - only process rows that need processing
    futures_map = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks first
        for index, row in df.iterrows():
            search_queries = str(row.get("search_queries", "")).strip()
            web_results = str(row.get("web_results", "")).strip()
            
            # Skip if search_queries is empty
            if not search_queries or search_queries == "nan":
                continue
            
            # Skip if web_results is already filled
            if web_results and web_results != "nan":
                continue
            
            # Submit task for processing
            future = executor.submit(process_web_search_row, index, row, k)
            futures_map[future] = index
        
        # If no rows to process, return early
        if not futures_map:
            print("All rows already processed or no valid search queries found.")
            return
        
        # Initialize progress tracker
        total_tasks = len(futures_map)
        progress = SmartProgressBar(
            total_tasks=total_tasks,
            description="Executing web search",
            save_interval=5
        )
        
        # Collect results as they complete
        for future in as_completed(futures_map):
            index = futures_map[future]
            try:
                row_index, web_results_text = future.result()
                
                # Update dataframe
                df.at[row_index, "web_results"] = web_results_text
                
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
                df.at[index, "web_results"] = f"ERROR: {str(e)}"
                progress.update()
    
    # Validation and retry logic
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        # Reload dataframe to get latest state
        worksheet, df = get_sheet_data_and_df(sheet, worksheet_name)
        
        # Find rows that need processing (empty or invalid web_results)
        invalid_rows = []
        for index, row in df.iterrows():
            is_valid, error_msg = validate_web_search_row(row)
            if not is_valid:
                invalid_rows.append((index, row, error_msg))
        
        if not invalid_rows:
            break
        
        retry_count += 1
        print(f"\n⚠️ Found {len(invalid_rows)} rows with invalid web_results. Retrying (attempt {retry_count}/{max_retries})...")
        for idx, row, error in invalid_rows[:3]:  # Show first 3 errors
            print(f"  Row {idx}: {error}")
        
        # Clear web_results for invalid rows
        for index, row, error_msg in invalid_rows:
            df.at[index, "web_results"] = ""
        
        # Save cleared state
        save_to_sheet(worksheet, df)
        format_worksheet(worksheet)
        
        # Retry processing invalid rows
        futures_map = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for index, row, error_msg in invalid_rows:
                future = executor.submit(process_web_search_row, index, row, k)
                futures_map[future] = index
            
            # Collect results
            for future in as_completed(futures_map):
                index = futures_map[future]
                try:
                    row_index, web_results_text = future.result()
                    df.at[row_index, "web_results"] = web_results_text
                except Exception as e:
                    print(f"Error getting result for row {index} on retry: {e}")
                    df.at[index, "web_results"] = f"ERROR: {str(e)}"
        
        # Save after retry
        save_to_sheet(worksheet, df)
        format_worksheet(worksheet)
    
    if retry_count > 0:
        # Check final state
        worksheet, df = get_sheet_data_and_df(sheet, worksheet_name)
        final_invalid = []
        for index, row in df.iterrows():
            is_valid, error_msg = validate_web_search_row(row)
            if not is_valid:
                final_invalid.append((index, error_msg))
        
        if final_invalid:
            print(f"⚠️ After {retry_count} retry attempt(s), {len(final_invalid)} rows still have invalid web_results.")
            for idx, error in final_invalid[:5]:  # Show first 5
                print(f"  Row {idx}: {error}")
        else:
            print(f"✅ All rows validated after {retry_count} retry attempt(s).")
    
    # Final save to sheet
    print('All web searches completed. Saving final DataFrame to sheet.')
    save_to_sheet(worksheet, df)
    format_worksheet(worksheet)
    print("✅ Web search complete and saved to sheet.")


def delete_web_results(sheet):
    """
    Remove the 'web_results' column from the specified worksheet.
    
    :param sheet: The gspread sheet object.
    :return: None
    """
    worksheet_name = "Slide Chunks"
    ws, df = get_sheet_data_and_df(sheet, worksheet_name)
    if "web_results" in df.columns:
        df = df.drop(columns=["web_results"])
        clear_worksheet(ws)
        save_to_sheet(ws, df)
        print(f"🗑️ Deleted 'web_results' column from '{worksheet_name}' worksheet")
    else:
        print(f"ℹ️ 'web_results' column does not exist in '{worksheet_name}' worksheet")

