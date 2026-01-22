from langsmith import traceable
import streamlit as st
from services.sheets_service import get_sheet_data_and_df, save_to_sheet, format_worksheet, clear_worksheet
from services.smart_progress_bar import SmartProgressBar
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
from agents.course_outline.video_search_tool.video_retriever import load_new_video_embeddings_chroma_db
from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from dotenv import load_dotenv
import os
import base64
from services.drive_service import login_with_service_account
from pydrive2.drive import GoogleDrive

load_dotenv()

# Configuration constants
search_k = 10  # Number of results to retrieve per search query
use_reranking = True  # Whether to use Cohere reranking
#video_embeddings_folder_id = '1SSvxx1EJ3zgMPfvD8Zy5DaEgmI2prys7'  # Google Drive folder ID for video embeddings
video_embeddings_folder_id = '15H9thXq02JX3ldADSj1oD78mbV-fXfvu'  # Google Drive folder ID for video embeddings
video_embeddings_folder_name = 'Vectorstore for HVAC school video embeddings (Category - 3D Animations and Simulations, Hands-On Field Work, Equipment Demos & Teardowns)'  # Name of the vectorstore folder inside the parent folder

def get_drive_instance():
    """
    Get Google Drive instance from session state or initialize from environment.
    
    :return: Google Drive instance
    """
    # First try to get from session state
    if "drive" in st.session_state:
        return st.session_state["drive"]
    
    # Fallback: initialize from environment variables
    try:
        key_bytes = base64.b64decode(os.environ["GDRIVE_SA_B64"])
        sa_json = key_bytes.decode()
        gauth = login_with_service_account(json_str=sa_json)
        gauth.ServiceAuth()
        drive = GoogleDrive(gauth)
        return drive
    except Exception as e:
        print(f"⚠️ Could not initialize Drive from environment: {e}")
        return None


def retrieve_video_docs_for_query(query, drive, k=search_k, use_reranking=use_reranking):
    """
    Retrieve video documents from vectorstore for a single query.
    
    :param query: Search query string
    :param drive: Google Drive instance
    :param k: Number of results to retrieve
    :param use_reranking: Whether to use Cohere reranking
    :return: List of document objects with metadata
    """
    if not drive:
        print(f"⚠️  Drive instance not available for video search")
        return []
    
    try:
        # Load video embeddings retriever from Google Drive
        video_embeddings_chroma = load_new_video_embeddings_chroma_db(drive, video_embeddings_folder_id, video_embeddings_folder_name)
        video_embeddings_retriever = video_embeddings_chroma.as_retriever(
            search_kwargs={"k": k * 2 if use_reranking else k}  # Get more if reranking
        )
        
        # Optionally use Cohere reranking
        if use_reranking:
            compressor = CohereRerank(
                model="rerank-v3.5",                            
                top_n=k,
            )
            retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=video_embeddings_retriever
            )
        else:
            retriever = video_embeddings_retriever
        
        docs = retriever.invoke(query)
        
        return docs[:k] if len(docs) > k else docs
        
    except Exception as e:
        print(f"❌ Video embeddings retriever failed: {e}")
        return []


@traceable(
    metadata={
        "agent_name": "graphics_definition_v2",
        "step_name": "YouTube Video Search",
        "function_name": "execute_video_search_for_query",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous")
    }
)
def execute_video_search_for_query(query, drive, k=search_k):
    """
    Execute video search for a single query and return video URLs.
    
    :param query: Search query string
    :param drive: Google Drive instance
    :param k: Number of results to retrieve
    :return: List of YouTube embed URLs with timestamps
    """
    print(f"🔎 Executing video search for query: \"{query}\"")
    
    # Execute search
    try:
        docs = retrieve_video_docs_for_query(query, drive, k, use_reranking)
        print(f"Raw results: {len(docs)} video segments")
    except Exception as e:
        print(f"❌ Search error: {str(e)}")
        return []
    
    # Process results and extract URLs
    video_urls = []
    for doc in docs:
        metadata = getattr(doc, "metadata", {})
        
        vid_id = metadata.get("video_id")
        if not vid_id:
            continue
        
        # Get timestamps directly from metadata
        start_sec = metadata.get("start_time", 0)
        end_sec = metadata.get("end_time", None)
        
        # Use timestamps directly from metadata
        abs_start = int(start_sec) if start_sec is not None else 0
        abs_end = int(end_sec) if end_sec is not None else None
        
        if abs_end is not None:
            url = f"https://www.youtube.com/embed/{vid_id}?start={abs_start}&end={abs_end}"
        else:
            url = f"https://www.youtube.com/embed/{vid_id}?start={abs_start}"
        
        video_urls.append(url)
    
    print(f"✅ Extracted {len(video_urls)} video URLs")
    
    return video_urls


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
        "step_name": "YouTube Video Search",
        "function_name": "process_video_search_segment",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous")
    }
)
def process_video_search_segment(segment_num, queries, drive, k=search_k):
    """
    Process a single segment: execute queries in parallel, deduplicate, format output.
    
    :param segment_num: Segment number
    :param queries: List of search queries for this segment
    :param drive: Google Drive instance
    :param k: Number of results per query
    :return: Tuple of (segment_num, segment_output_string) or (segment_num, None) if no results
    """
    print(f"\n{'─'*45}")
    print(f"📦 Processing SEGMENT_{segment_num} with {len(queries)} queries")
    print(f"{'─'*45}")
    
    # Execute all queries for this segment
    all_urls_for_segment = []
    seen = set()  # Deduplicate by video_id_start_time within segment
    
    # Filter out empty queries
    valid_queries = [(idx, q.strip()) for idx, q in enumerate(queries, 1) if q.strip()]
    
    if not valid_queries:
        return segment_num, None
    
    # Execute queries in parallel
    with ThreadPoolExecutor(max_workers=len(valid_queries)) as executor:
        # Submit all queries
        futures = {
            executor.submit(
                execute_video_search_for_query,
                query,
                drive,
                k
            ): (query_idx, query)
            for query_idx, query in valid_queries
        }
        
        # Collect results as they complete
        for future in as_completed(futures):
            query_idx, query = futures[future]
            try:
                print(f"🔍 Query {query_idx}: \"{query}\"")
                video_urls = future.result()
                print(f"✅ Query {query_idx} returned {len(video_urls)} video URLs")
                
                # Deduplicate by video_id_start_time within this segment
                for url in video_urls:
                    # Extract video_id and start_time from URL for deduplication
                    # Format: https://www.youtube.com/embed/{vid_id}?start={start}&end={end}
                    match = re.search(r'/embed/([^?]+)\?start=(\d+)', url)
                    if match:
                        vid_id = match.group(1)
                        start_time = match.group(2)
                        segment_key = f"{vid_id}_{start_time}"
                        
                        if segment_key not in seen:
                            seen.add(segment_key)
                            all_urls_for_segment.append(url)
            except Exception as e:
                print(f"❌ Error executing query {query_idx} (\"{query}\"): {e}")
    
    print(f"✅ SEGMENT_{segment_num}: Found {len(all_urls_for_segment)} unique video URLs")
    
    # Format segment results
    if all_urls_for_segment:
        segment_output = [f"---SEGMENT_{segment_num}---"] + all_urls_for_segment
        return segment_num, '\n'.join(segment_output)
    return segment_num, None


@traceable(
    metadata={
        "agent_name": "graphics_definition_v2",
        "step_name": "YouTube Video Search",
        "function_name": "process_video_search_row",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous")
    }
)
def process_video_search_row(index, row, drive, k=search_k):
    """
    Process a single row: parse video search queries, execute searches for each segment, deduplicate, format output.
    
    :param index: Row index
    :param row: Pandas Series with row data
    :param drive: Google Drive instance
    :param k: Number of results per query
    :return: Tuple of (index, video_pool_text)
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
                executor.submit(process_video_search_segment, segment_num, queries, drive, k): segment_num
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
        video_pool_text = '\n\n'.join(all_segment_results)
        
        return index, video_pool_text
        
    except Exception as e:
        print(f"Error processing row {index}: {e}")
        return index, ""


def validate_video_pool_row(row):
    """
    Validate that video_pool matches search_queries:
    - Row is not empty
    - All segments from search_queries have results
    - No gaps in segment numbering
    
    :param row: Pandas Series with row data
    :return: Tuple (is_valid, error_message)
    """
    search_queries_text = str(row.get("search_queries", "")).strip()
    video_pool_text = str(row.get("video_pool", "")).strip()
    
    # Skip validation if search_queries is empty
    if not search_queries_text or search_queries_text == "nan":
        return True, None
    
    # Check if video_pool is empty
    if not video_pool_text or video_pool_text == "nan" or video_pool_text.strip() == "":
        return False, "video_pool is empty"
    
    # Check if it's an error marker
    if video_pool_text.startswith("ERROR:"):
        return False, "video_pool contains error marker"
    
    # Parse expected segments from search_queries
    expected_segments = parse_search_queries_column(search_queries_text)
    if not expected_segments:
        return True, None  # No segments to validate
    
    expected_segment_nums = sorted([seg_num for seg_num, _ in expected_segments])
    expected_count = len(expected_segment_nums)
    
    # Parse actual segments from video_pool
    segment_pattern = r'---SEGMENT_(\d+)---'
    actual_segment_nums = [int(match) for match in re.findall(segment_pattern, video_pool_text)]
    
    if not actual_segment_nums:
        return False, "No segment markers found in video_pool"
    
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
        "step_name": "YouTube Video Search",
        "function_name": "run_youtube_video_search_for_all_rows",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous")
    }
)
def run_youtube_video_search_for_all_rows(sheet, k=search_k, max_workers=5):
    """
    Execute YouTube video search for all rows in the Slide Chunks sheet.
    
    :param sheet: The gspread sheet object.
    :param k: Number of results per query (default from settings).
    :param max_workers: Number of parallel workers (default 5).
    :return: None
    """
    worksheet_name = "Slide Chunks"
    
    # Get drive instance
    drive = get_drive_instance()
    if not drive:
        print("❌ Drive instance not available. Cannot execute video search.")
        return
    
    # Load the worksheet and DataFrame
    worksheet, df = get_sheet_data_and_df(sheet, worksheet_name)
    
    # Ensure video_pool column exists
    if "video_pool" not in df.columns:
        df["video_pool"] = ""
    
    # Prepare for parallel processing - only process rows that need processing
    futures_map = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks first
        for index, row in df.iterrows():
            search_queries = str(row.get("search_queries", "")).strip()
            video_pool = str(row.get("video_pool", "")).strip()
            
            # Skip if search_queries is empty
            if not search_queries or search_queries == "nan":
                continue
            
            # Skip if video_pool is already filled
            if video_pool and video_pool != "nan":
                continue
            
            # Submit task for processing
            future = executor.submit(process_video_search_row, index, row, drive, k)
            futures_map[future] = index
        
        # If no rows to process, return early
        if not futures_map:
            print("All rows already processed or no valid search queries found.")
            return
        
        # Initialize progress tracker
        total_tasks = len(futures_map)
        progress = SmartProgressBar(
            total_tasks=total_tasks,
            description="Executing YouTube video search",
            save_interval=5
        )
        
        # Collect results as they complete
        for future in as_completed(futures_map):
            index = futures_map[future]
            try:
                row_index, video_pool_text = future.result()
                
                # Update dataframe
                df.at[row_index, "video_pool"] = video_pool_text
                
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
                df.at[index, "video_pool"] = f"ERROR: {str(e)}"
                progress.update()

    # Save final results before validation
    save_to_sheet(worksheet, df)
    format_worksheet(worksheet)

    # Validation and retry logic
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        # Reload dataframe to get latest state
        worksheet, df = get_sheet_data_and_df(sheet, worksheet_name)
        
        # Validate all rows and find invalid ones
        invalid_rows = []
        for index, row in df.iterrows():
            is_valid, error_msg = validate_video_pool_row(row)
            if not is_valid:
                invalid_rows.append((index, row, error_msg))
        
        if not invalid_rows:
            break
        
        retry_count += 1
        print(f"\n⚠️ Found {len(invalid_rows)} rows with invalid video_pool. Retrying (attempt {retry_count}/{max_retries})...")
        for idx, row, error in invalid_rows[:3]:  # Show first 3 errors
            print(f"  Row {idx}: {error}")
        
        # Clear video_pool for invalid rows
        for index, row, error_msg in invalid_rows:
            df.at[index, "video_pool"] = ""
        
        # Save cleared state
        save_to_sheet(worksheet, df)
        format_worksheet(worksheet)
        
        # Retry processing invalid rows
        futures_map = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for index, row, error_msg in invalid_rows:
                future = executor.submit(process_video_search_row, index, row, drive, k)
                futures_map[future] = index
            
            # Collect results
            for future in as_completed(futures_map):
                index = futures_map[future]
                try:
                    row_index, video_pool_text = future.result()
                    df.at[row_index, "video_pool"] = video_pool_text
                except Exception as e:
                    print(f"Error getting result for row {index} on retry: {e}")
                    df.at[index, "video_pool"] = f"ERROR: {str(e)}"
        
        # Save after retry
        save_to_sheet(worksheet, df)
        format_worksheet(worksheet)
    
    if retry_count > 0:
        # Check final state
        worksheet, df = get_sheet_data_and_df(sheet, worksheet_name)
        final_invalid = []
        for index, row in df.iterrows():
            is_valid, error_msg = validate_video_pool_row(row)
            if not is_valid:
                final_invalid.append((index, error_msg))
        
        if final_invalid:
            print(f"⚠️ After {retry_count} retry attempt(s), {len(final_invalid)} rows still have invalid video_pool.")
            for idx, error in final_invalid[:5]:  # Show first 5
                print(f"  Row {idx}: {error}")
        else:
            print(f"✅ All rows validated after {retry_count} retry attempt(s).")
    
    # Final save to sheet
    print('All YouTube video searches completed. Saving final DataFrame to sheet.')
    save_to_sheet(worksheet, df)
    format_worksheet(worksheet)
    print("✅ YouTube video search complete and saved to sheet.")


def delete_video_pool(sheet):
    """
    Remove the 'video_pool' column from the specified worksheet.
    
    :param sheet: The gspread sheet object.
    :return: None
    """
    worksheet_name = "Slide Chunks"
    ws, df = get_sheet_data_and_df(sheet, worksheet_name)
    if "video_pool" in df.columns:
        df = df.drop(columns=["video_pool"])
        clear_worksheet(ws)
        save_to_sheet(ws, df)
        print(f"🗑️ Deleted 'video_pool' column from '{worksheet_name}' worksheet")
    else:
        print(f"ℹ️ 'video_pool' column does not exist in '{worksheet_name}' worksheet")

