from langsmith import traceable
import streamlit as st
from services.sheets_service import get_sheet_data_and_df, save_to_sheet, merge_and_save_columns, format_worksheet, clear_worksheet
from services.smart_progress_bar import SmartProgressBar
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
import re
import os
import time
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from utils.decorator_helpers import cycle_api_keys_decorator
import isodate

load_dotenv()

# Configuration constants
search_k = 4  # Number of results to retrieve per search query (top 2)
max_video_duration_seconds = 15 * 60  # 15 minutes in seconds

# Channel IDs to exclude (the ones already vectorized)
EXCLUDED_CHANNEL_IDS = [
    "UCdLlUhD9LUGm0-NTYDASPFA",
    "UCIoD-SEdUWMA74tWXRMyCZQ"
]

# Collect all available API keys (filter out None values from missing env vars)
all_yt_api_keys = [
    os.environ.get("GCLOUD_YT_SEARCH_API_KEY_1"),
    os.environ.get("GCLOUD_YT_SEARCH_API_KEY_2"),
    os.environ.get("GCLOUD_YT_SEARCH_API_KEY_3"),
    os.environ.get("GCLOUD_YT_SEARCH_API_KEY_4"),
    os.environ.get("GCLOUD_YT_SEARCH_API_KEY_5"),
    os.environ.get("GCLOUD_YT_SEARCH_API_KEY_6"),
    os.environ.get("GCLOUD_YT_SEARCH_API_KEY_7"),
    os.environ.get("GCLOUD_YT_SEARCH_API_KEY_8")
]
gcloud_yt_search_api_keys = [key for key in all_yt_api_keys if key is not None]

# Log how many keys are available
#print(f"📺 YouTube API: {len(gcloud_yt_search_api_keys)} API key(s) configured")


def parse_duration_to_seconds(duration_str):
    """
    Parse ISO 8601 duration string (e.g., 'PT15M30S') to seconds.
    
    :param duration_str: ISO 8601 duration string
    :return: Duration in seconds or None if parsing fails
    """
    try:
        duration = isodate.parse_duration(duration_str)
        return int(duration.total_seconds())
    except Exception as e:
        print(f"Error parsing duration '{duration_str}': {e}")
        return None


@cycle_api_keys_decorator(gcloud_yt_search_api_keys)
def search_youtube_videos_with_filters(query, max_results=4, excluded_channel_ids=None, max_duration_seconds=None, developer_key=None):
    """
    Search YouTube for videos, filtering by duration and excluding specific channels.
    
    :param query: Search query string
    :param max_results: Number of results to return after filtering
    :param excluded_channel_ids: List of channel IDs to exclude
    :param max_duration_seconds: Maximum video duration in seconds
    :param developer_key: YouTube Data API key (injected by decorator)
    :return: List of video detail dictionaries
    """
    if excluded_channel_ids is None:
        excluded_channel_ids = EXCLUDED_CHANNEL_IDS
    
    youtube = build('youtube', 'v3', developerKey=developer_key)
    

    search_results = []
    
    try:

        response = youtube.search().list(
            q=query,
            part='snippet',
            type='video',
            maxResults=max_results * 5,  # Fetch extra to account for filtering
            videoEmbeddable='true',  # Only embeddable videos
            relevanceLanguage='en'  # Prefer English results
        ).execute()
        search_results = response.get('items', [])
    except HttpError as e:
        if e.resp.status == 403 and 'quotaExceeded' in str(e):
            raise  # Re-raise to trigger API key cycling
        print(f"HTTP Error searching videos: {e}")
        return []
    except Exception as e:
        print(f"Error searching videos: {e}")
        return []
    
    if not search_results:
        return []
    
    # Extract video IDs
    video_ids = [item['id']['videoId'] for item in search_results if 'videoId' in item.get('id', {})]
    
    if not video_ids:
        return []
    
    # Get video details including duration in a batch call
    video_details_list = []
    try:
        videos_response = youtube.videos().list(
            part='snippet,contentDetails',
            id=','.join(video_ids[:50])  # API limit is 50 IDs per call
        ).execute()
        
        for video in videos_response.get('items', []):
            video_id = video['id']
            duration_str = video['contentDetails'].get('duration', '')
            duration_seconds = parse_duration_to_seconds(duration_str)
            
            snippet = video['snippet']
            channel_id = snippet.get('channelId', '')
            
            # Apply filters
            should_exclude = False
            
            # Exclude specific channels
            if channel_id in excluded_channel_ids:
                should_exclude = True
            
            # Exclude videos longer than max duration
            if max_duration_seconds and duration_seconds and duration_seconds > max_duration_seconds:
                should_exclude = True
            
            if not should_exclude:
                # Format duration as MM:SS or HH:MM:SS
                if duration_seconds:
                    if duration_seconds >= 3600:
                        duration_formatted = f"{duration_seconds//3600}:{(duration_seconds%3600)//60:02d}:{duration_seconds%60:02d}"
                    else:
                        duration_formatted = f"{duration_seconds//60}:{duration_seconds%60:02d}"
                else:
                    duration_formatted = "Unknown"
                
                video_detail = {
                    'video_id': video_id,
                    'video_url': f"https://www.youtube.com/watch?v={video_id}",
                    'title': snippet.get('title', 'Untitled'),
                    'description': snippet.get('description', ''),
                    'channel_title': snippet.get('channelTitle', ''),
                    'channel_id': channel_id,
                    'published_at': snippet.get('publishedAt', ''),
                    'duration_seconds': duration_seconds,
                    'duration_formatted': duration_formatted
                }
                video_details_list.append(video_detail)
        
        # Return only up to max_results (search already returned in relevance order)
        return video_details_list[:max_results]
        
    except Exception as e:
        print(f"Error getting video details: {e}")
        return []


@traceable(
    metadata={
        "agent_name": "graphics_definition_v2",
        "step_name": "YouTube Video Search (Other Channels)",
        "function_name": "execute_youtube_search_for_query",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous")
    }
)
def execute_youtube_search_for_query(query, k=search_k, max_duration_seconds=max_video_duration_seconds):
    """
    Execute YouTube search for a single query and return video details.
    
    :param query: Search query string
    :param k: Number of results to retrieve (top k)
    :param max_duration_seconds: Maximum video duration in seconds
    :return: List of video detail dictionaries
    """
    print(f"🔍 YouTube search: \"{query[:60]}...\" (max {max_duration_seconds//60} min, top {k})")
    
    try:
        results = search_youtube_videos_with_filters(
            query=query,
            max_results=k,
            excluded_channel_ids=EXCLUDED_CHANNEL_IDS,
            max_duration_seconds=max_duration_seconds
        )
        print(f"✅ Found {len(results)} videos")
        return results
    except Exception as e:
        print(f"❌ YouTube search error: {str(e)}")
        return []


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


def parse_voiceover_segments(voiceover_text):
    """
    Parse voiceover_segment column (newline-separated sentences).
    
    :param voiceover_text: The voiceover_segment column content
    :return: List of segment sentences (1-indexed by position)
    """
    if not voiceover_text or voiceover_text.strip() == "" or voiceover_text == "nan":
        return []
    
    segments = [seg.strip() for seg in voiceover_text.split('\n') if seg.strip()]
    return segments


@traceable(
    metadata={
        "agent_name": "graphics_definition_v2",
        "step_name": "YouTube Video Search (Other Channels)",
        "function_name": "process_segment_other_channels",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous")
    }
)
def process_segment_other_channels(segment_num, queries, segment_sentence, k=search_k, max_duration_seconds=max_video_duration_seconds):
    """
    Process a single segment: execute n+1 queries (n queries + segment sentence), deduplicate, format output.
    
    :param segment_num: Segment number
    :param queries: List of search queries for this segment
    :param segment_sentence: The voiceover segment sentence (used as additional query)
    :param k: Number of results per query (top k)
    :param max_duration_seconds: Maximum video duration in seconds
    :return: Tuple of (segment_num, segment_output_string) or (segment_num, None) if no results
    """
    print(f"\n{'─'*50}")
    print(f"📦 SEGMENT_{segment_num}: {len(queries)} queries + 1 segment sentence = {len(queries) + 1} total queries")
    print(f"{'─'*50}")
    
    # Combine all queries: segment sentence + search queries
    all_queries = []
    
    # Add segment sentence as first query
    if segment_sentence and segment_sentence.strip():
        all_queries.append(("segment", segment_sentence.strip()))
    
    # Add search queries
    for idx, query in enumerate(queries, 1):
        if query.strip():
            all_queries.append((f"query_{idx}", query.strip()))
    
    if not all_queries:
        return segment_num, None
    
    # Execute all queries in parallel
    all_videos_for_segment = []
    seen_video_ids = set()  # Deduplicate by video_id within segment
    
    with ThreadPoolExecutor(max_workers=min(len(all_queries), 10)) as executor:
        futures = {
            executor.submit(
                execute_youtube_search_for_query,
                query,
                k,
                max_duration_seconds
            ): (query_label, query)
            for query_label, query in all_queries
        }
        
        for future in as_completed(futures):
            query_label, query = futures[future]
            try:
                videos = future.result()
                print(f"  [{query_label}] \"{query[:50]}...\" → {len(videos)} videos")
                
                # Deduplicate by video_id within this segment
                for video in videos:
                    video_id = video.get("video_id", "")
                    if video_id and video_id not in seen_video_ids:
                        seen_video_ids.add(video_id)
                        all_videos_for_segment.append(video)
            except Exception as e:
                print(f"  ❌ [{query_label}] Error: {e}")
    
    print(f"✅ SEGMENT_{segment_num}: {len(all_videos_for_segment)} unique videos after deduplication")
    
    # Format segment results
    if all_videos_for_segment:
        segment_lines = [f"---SEGMENT_{segment_num}---"]
        for video in all_videos_for_segment:
            title = video.get("title", "Untitled")
            url = video.get("video_url", "")
            duration = video.get("duration_formatted", "")
            channel = video.get("channel_title", "")
            segment_lines.append(f"Title: {title} | Duration: {duration} | Channel: {channel} | URL: {url}")
        
        return segment_num, '\n'.join(segment_lines)
    
    return segment_num, None


@traceable(
    metadata={
        "agent_name": "graphics_definition_v2",
        "step_name": "YouTube Video Search (Other Channels)",
        "function_name": "process_row_other_channels",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous")
    }
)
def process_row_other_channels(index, row, k=search_k, max_duration_seconds=max_video_duration_seconds):
    """
    Process a single row: parse search_queries and voiceover_segment,
    execute YouTube searches for each segment, deduplicate, format output.
    
    :param index: Row index
    :param row: Pandas Series with row data
    :param k: Number of results per query (top k)
    :param max_duration_seconds: Maximum video duration in seconds
    :return: Tuple of (index, video_pool_other_channels_text)
    """
    try:
        search_queries_text = str(row.get("search_queries", "")).strip()
        voiceover_segments_text = str(row.get("voiceover_segment", "")).strip()
        
        # Skip if search_queries is empty
        if not search_queries_text or search_queries_text == "nan":
            return index, ""
        
        # Parse segments and queries from search_queries
        query_segments = parse_search_queries_column(search_queries_text)
        
        if not query_segments:
            return index, ""
        
        # Parse voiceover segments (sentences)
        voiceover_sentences = parse_voiceover_segments(voiceover_segments_text)
        
        # Build segment data: match segment_num to voiceover sentence
        segment_data = []
        for segment_num, queries in query_segments:
            # voiceover_sentences is 0-indexed, segment_num is 1-indexed
            segment_sentence = ""
            if segment_num <= len(voiceover_sentences):
                segment_sentence = voiceover_sentences[segment_num - 1]
            
            segment_data.append((segment_num, queries, segment_sentence))
        
        # Execute all segments in parallel
        segment_results = {}
        with ThreadPoolExecutor(max_workers=min(len(segment_data), 5)) as executor:
            futures = {
                executor.submit(
                    process_segment_other_channels,
                    segment_num,
                    queries,
                    segment_sentence,
                    k,
                    max_duration_seconds
                ): segment_num
                for segment_num, queries, segment_sentence in segment_data
            }
            
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


def validate_video_pool_other_channels_row(row):
    """
    Validate that video_pool_other_channels matches search_queries:
    - Row is not empty
    - All segments from search_queries have results
    - No gaps in segment numbering
    
    :param row: Pandas Series with row data
    :return: Tuple (is_valid, error_message)
    """
    search_queries_text = str(row.get("search_queries", "")).strip()
    video_pool_text = str(row.get("video_pool_other_channels", "")).strip()
    slide_type = str(row.get("Slide Type", "")).strip().lower()
    
    # Transition slides have no video candidates by design; empty video_pool_other_channels is valid
    if slide_type in ("transition", "transition slide"):
        return True, None
    
    # Skip validation if search_queries is empty
    if not search_queries_text or search_queries_text == "nan":
        return True, None
    
    # Check if video_pool_other_channels is empty
    if not video_pool_text or video_pool_text == "nan" or video_pool_text.strip() == "":
        return False, "video_pool_other_channels is empty"
    
    # Check if it's an error marker
    if video_pool_text.startswith("ERROR:"):
        return False, "video_pool_other_channels contains error marker"
    
    # Parse expected segments from search_queries
    expected_segments = parse_search_queries_column(search_queries_text)
    if not expected_segments:
        return True, None  # No segments to validate
    
    expected_segment_nums = sorted([seg_num for seg_num, _ in expected_segments])
    expected_count = len(expected_segment_nums)
    
    # Parse actual segments from video_pool_other_channels
    segment_pattern = r'---SEGMENT_(\d+)---'
    actual_segment_nums = [int(match) for match in re.findall(segment_pattern, video_pool_text)]
    
    if not actual_segment_nums:
        return False, "No segment markers found in video_pool_other_channels"
    
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
        "step_name": "YouTube Video Search (Other Channels)",
        "function_name": "run_youtube_video_search_other_channels_for_all_rows",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous")
    }
)
def run_youtube_video_search_other_channels_for_all_rows(sheet, k=search_k, max_duration_seconds=max_video_duration_seconds, max_workers=50
):
    """
    Execute YouTube video search (other channels) for all rows in the Slide Chunks sheet.
    
    :param sheet: The gspread sheet object.
    :param k: Number of results per query (default 2 = top 2).
    :param max_duration_seconds: Maximum video duration in seconds (default 15 min).
    :param max_workers: Number of parallel workers (default 5).
    :return: None
    """
    worksheet_name = "Slide Chunks"
    
    print(f"\n🚀 Starting YouTube video search (other channels)")
    print(f" Config: top {k} per query, max duration {max_duration_seconds//60} min")
    
    # Load the worksheet and DataFrame
    worksheet, df = get_sheet_data_and_df(sheet, worksheet_name)
    
    # Ensure video_pool_other_channels column exists
    if "video_pool_other_channels" not in df.columns:
        df["video_pool_other_channels"] = ""
    
    # Prepare for parallel processing - only process rows that need processing
    futures_map = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks first
        for index, row in df.iterrows():
            search_queries = str(row.get("search_queries", "")).strip()
            video_pool_other = str(row.get("video_pool_other_channels", "")).strip()
            slide_type = str(row.get("Slide Type", "")).strip().lower()
            
            # Transition slides: no video candidates; leave video_pool_other_channels empty
            if slide_type in ("transition", "transition slide"):
                continue
            
            # Skip if search_queries is empty
            if not search_queries or search_queries == "nan":
                continue
            
            # Skip if video_pool_other_channels is already filled
            if video_pool_other and video_pool_other != "nan":
                continue
            
            # Submit task for processing
            future = executor.submit(process_row_other_channels, index, row, k, max_duration_seconds)
            futures_map[future] = index
        
        # If no rows to process, return early
        if not futures_map:
            print("✅ All rows already processed or no valid video search queries found.")
            return
        
        # Initialize progress tracker
        total_tasks = len(futures_map)
        progress = SmartProgressBar(
            total_tasks=total_tasks,
            description="YouTube search (other channels)",
            save_interval=5
        )
        
        # Collect results as they complete
        for future in as_completed(futures_map):
            index = futures_map[future]
            try:
                row_index, video_pool_text = future.result()
                
                # Update dataframe
                df.at[row_index, "video_pool_other_channels"] = video_pool_text
                
                # Update progress
                progress.update()
                
                # Save every 5 rows
                if progress.should_save():
                    print(f'Saving partial progress to sheet after {progress.completed_count} tasks completed.')
                    merge_and_save_columns(sheet, worksheet_name, df, ["video_pool_other_channels"])
                    format_worksheet(worksheet)
            except Exception as e:
                print(f"Error getting result for row {index}: {e}")
                # Update dataframe with error marker so row is marked as processed
                df.at[index, "video_pool_other_channels"] = f"ERROR: {str(e)}"
                progress.update()

    # Save final results before validation
    merge_and_save_columns(sheet, worksheet_name, df, ["video_pool_other_channels"])
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
            is_valid, error_msg = validate_video_pool_other_channels_row(row)
            if not is_valid:
                invalid_rows.append((index, row, error_msg))
        
        if not invalid_rows:
            break
        
        retry_count += 1
        print(f"\n⚠️ Found {len(invalid_rows)} rows with invalid video_pool_other_channels. Retrying (attempt {retry_count}/{max_retries})...")
        for idx, row, error in invalid_rows[:3]:  # Show first 3 errors
            print(f"  Row {idx}: {error}")
        
        # Clear video_pool_other_channels for invalid rows
        for index, row, error_msg in invalid_rows:
            df.at[index, "video_pool_other_channels"] = ""
        
        # Save cleared state
        merge_and_save_columns(sheet, worksheet_name, df, ["video_pool_other_channels"])
        format_worksheet(worksheet)
        
        # Retry processing invalid rows
        futures_map = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for index, row, error_msg in invalid_rows:
                future = executor.submit(process_row_other_channels, index, row, k, max_duration_seconds)
                futures_map[future] = index
            
            # Collect results
            for future in as_completed(futures_map):
                index = futures_map[future]
                try:
                    row_index, video_pool_text = future.result()
                    df.at[row_index, "video_pool_other_channels"] = video_pool_text
                except Exception as e:
                    print(f"Error getting result for row {index} on retry: {e}")
                    df.at[index, "video_pool_other_channels"] = f"ERROR: {str(e)}"
        
        # Save after retry
        merge_and_save_columns(sheet, worksheet_name, df, ["video_pool_other_channels"])
        format_worksheet(worksheet)
    
    if retry_count > 0:
        # Check final state
        worksheet, df = get_sheet_data_and_df(sheet, worksheet_name)
        final_invalid = []
        for index, row in df.iterrows():
            is_valid, error_msg = validate_video_pool_other_channels_row(row)
            if not is_valid:
                final_invalid.append((index, error_msg))
        
        if final_invalid:
            print(f"⚠️ After {retry_count} retry attempt(s), {len(final_invalid)} rows still have invalid video_pool_other_channels.")
            for idx, error in final_invalid[:5]:  # Show first 5
                print(f"  Row {idx}: {error}")
        else:
            print(f"✅ All rows validated after {retry_count} retry attempt(s).")
    
    # Final save to sheet
    print('All YouTube video searches (other channels) completed. Saving final DataFrame to sheet.')
    merge_and_save_columns(sheet, worksheet_name, df, ["video_pool_other_channels"])
    format_worksheet(worksheet)
    print("✅ YouTube video search (other channels) complete and saved to sheet.")


def delete_video_pool_other_channels(sheet):
    """
    Remove the 'video_pool_other_channels' column from the specified worksheet.
    
    :param sheet: The gspread sheet object.
    :return: None
    """
    worksheet_name = "Slide Chunks"
    ws, df = get_sheet_data_and_df(sheet, worksheet_name)
    if "video_pool_other_channels" in df.columns:
        df = df.drop(columns=["video_pool_other_channels"])
        clear_worksheet(ws)
        save_to_sheet(ws, df)
        print(f"🗑️ Deleted 'video_pool_other_channels' column from '{worksheet_name}' worksheet")
    else:
        print(f"ℹ️ 'video_pool_other_channels' column does not exist in '{worksheet_name}' worksheet")

