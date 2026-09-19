"""Google Drive video candidate search for Graphics Definition V2.

Reads search_queries per slide row, searches Google Drive video vector store, and writes segment-aligned results to the drive_video_pool column.
"""

from __future__ import annotations

import base64
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
from dotenv import load_dotenv
from langsmith import traceable
from pydrive2.drive import GoogleDrive

from agents.course_outline.video_search_tool.google_drive_video_search.google_drive_course_videos_search import (
    load_gemini_video_chroma_collection,
    search_gemini_drive_video_embeddings,
)
from agents.graphics_definition_v2.candidate_search.pool_registry import (
    DRIVE_VIDEO_MODE_ALL,
    DRIVE_VIDEO_MODE_NEXTECH,
    UI_KEY_DRIVE_VIDEO_MODE,
)
from agents.graphics_definition_v2.candidate_search.search_wrapper import run_pool_search_queries
from services.drive_service import login_with_service_account
from services.sheets_service import (
    clear_worksheet,
    format_worksheet,
    get_sheet_data_and_df,
    merge_and_save_columns,
    save_to_sheet,
)
from services.smart_progress_bar import SmartProgressBar

load_dotenv()

search_k = 5
DRIVE_VIDEO_POOL_COLUMN = "drive_video_pool"


def get_drive_instance():
    """Get Google Drive instance from session state or service-account env."""
    if "drive" in st.session_state:
        return st.session_state["drive"]

    try:
        key_bytes = base64.b64decode(os.environ["GDRIVE_SA_B64"])
        sa_json = key_bytes.decode()
        gauth = login_with_service_account(json_str=sa_json)
        gauth.ServiceAuth()
        return GoogleDrive(gauth)
    except Exception as e:
        print(f"⚠️ Could not initialize Drive from environment: {e}")
        return None


def parse_search_queries_column(search_queries_text: str) -> List[Tuple[int, List[str]]]:
    """Parse search_queries into (segment_number, queries) pairs."""
    if not search_queries_text or search_queries_text.strip() == "" or search_queries_text == "nan":
        return []

    segments = []
    segment_pattern = r"---SEGMENT_(\d+)---"
    parts = re.split(segment_pattern, search_queries_text)

    for i in range(1, len(parts), 2):
        if i + 1 < len(parts):
            segment_num = int(parts[i])
            segment_content = parts[i + 1].strip()
            queries = [q.strip() for q in segment_content.split("\n") if q.strip()]
            if queries:
                segments.append((segment_num, queries))

    return segments


def format_drive_video_result_line(item: Dict[str, Any]) -> Optional[str]:
    """Format one Drive video hit into a sheet line."""
    video_link = str(item.get("video_link") or item.get("video_url") or "").strip()
    video_id = str(item.get("video_id") or "").strip()
    if not video_link and video_id:
        video_link = f"https://drive.google.com/file/d/{video_id}/view?usp=drivesdk"
    if not video_link:
        return None
    
    # Ensure video_link has the drivesdk param
    if "usp=drivesdk" not in video_link:
        if "?" in video_link:
            video_link += "&usp=drivesdk"
        else:
            video_link += "?usp=drivesdk"

    title = str(item.get("video_name") or item.get("video_title") or item.get("title") or "Drive video").strip()
    start_time = item.get("start_time", 0)
    end_time = item.get("end_time", "")

    try:
        start_val = int(float(start_time)) if start_time is not None and str(start_time).strip() != "" else 0
    except (TypeError, ValueError):
        start_val = 0

    try:
        end_val = int(float(end_time)) if end_time is not None and str(end_time).strip() != "" else ""
    except (TypeError, ValueError):
        end_val = ""

    return f"Title: {title} | URL: {video_link} (start={start_val}&end={end_val})"


def result_dedupe_key(item: Dict[str, Any]) -> str:
    """Stable key for deduping clips within a segment."""
    video_id = str(item.get("video_id") or item.get("video_link") or item.get("video_url") or "").strip()
    start_time = item.get("start_time", 0)
    try:
        start_val = int(float(start_time)) if start_time is not None and str(start_time).strip() != "" else 0
    except (TypeError, ValueError):
        start_val = 0
    return f"{video_id}_{start_val}"


_NEXTECH_TAG_CACHE: list[str] | None = None


def get_discovered_nextech_tags(drive) -> list[str]:
    """
    Dynamically discover all unique source_tag values starting with 'nextech_' in Google Drive video vector store, to keep queries completely future-proof and avoid hardcoding domain-specific tag lists.
    """
    global _NEXTECH_TAG_CACHE
    if _NEXTECH_TAG_CACHE is not None:
        return _NEXTECH_TAG_CACHE

    try:
        # Load the warmed up collection
        collection = load_gemini_video_chroma_collection(drive=drive)
        get_results = collection.get(include=["metadatas"])
        metadatas = get_results.get("metadatas") or []
        
        tags = set()
        for meta in metadatas:
            if meta:
                tag = meta.get("source_tag")
                if isinstance(tag, str) and tag.lower().startswith("nextech_"):
                    tags.add(tag.strip().lower())
        
        discovered = sorted(list(tags))
        if discovered:
            _NEXTECH_TAG_CACHE = discovered
            print(f"🔍 Dynamically discovered NexTech tags in DB: {discovered}")
            return discovered
    except Exception as e:
        print(f"⚠️ Failed dynamic NexTech tag discovery: {e}")

    # Fallback to current known tags if discovery fails or database is empty
    return ["nextech_hvac", "nextech_electrical", "nextech_plumbing"]


@traceable(
    metadata={
        "agent_name": "graphics_definition_v2",
        "step_name": "Drive Video Search",
        "function_name": "execute_drive_video_search_for_query",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous"),
    }
)
def execute_drive_video_search_for_query(query, drive=None, k=search_k, drive_video_mode=DRIVE_VIDEO_MODE_ALL):
    """
    Search Google Drive video vector store for one text query.

    :param query: Search query string
    :param drive: Google Drive instance
    :param k: Number of results to retrieve
    :param drive_video_mode: "all" or "nextech" — must be passed explicitly from the main thread (worker threads cannot reliably read Streamlit session_state)
    :return: List of Drive video hit dictionaries
    """
    print(f'🔎 Executing Drive video search for query: "{query}" (mode={drive_video_mode})')
    if not drive:
        print("⚠️ Drive instance not available for Drive video search")
        return []

    try:
        if drive_video_mode == DRIVE_VIDEO_MODE_NEXTECH:
            # Dynamically discover all nextech_ tags in the DB, ensuring zero future code changes are needed
            tags = get_discovered_nextech_tags(drive)
            if not tags:
                print("⚠️ No NexTech source_tag values found in vector store")
                return []
            all_results = []

            print(f"🚀 Querying NexTech tags directly from DB in parallel: {tags}")
            with ThreadPoolExecutor(max_workers=len(tags)) as executor:
                futures = {
                    executor.submit(
                        search_gemini_drive_video_embeddings,
                        drive=drive,
                        query_type="text",
                        query=query,
                        k=k,
                        filters={"source_tag": tag},
                    ): tag
                    for tag in tags
                }
                for future in as_completed(futures):
                    tag = futures[future]
                    try:
                        tag_res = future.result() or []
                        print(f"Direct DB-level search for source_tag='{tag}' returned {len(tag_res)} results")
                        all_results.extend(tag_res)
                    except Exception as e:
                        print(f"❌ Failed DB-level search for tag {tag}: {e}")

            # Deduplicate results across parallel tag queries
            seen_ids = set()
            unique_results = []
            for item in all_results:
                item_id = item.get("id") or f"{item.get('video_id')}_{item.get('segment_index')}"
                if item_id not in seen_ids:
                    seen_ids.add(item_id)
                    unique_results.append(item)

            # Sort by distance (lower distance is a closer semantic match)
            unique_results.sort(key=lambda x: x.get("distance") if x.get("distance") is not None else 1.0)
            results = unique_results[:k]
            print(f"Total deduplicated NexTech results: {len(results)}")
        else:
            results = search_gemini_drive_video_embeddings(
                drive=drive,
                query_type="text",
                query=query,
                k=k,
                filters=None,
            )
            print(f"Raw results: {len(results)} Drive video segments")

        return results or []
    except Exception as e:
        print(f"❌ Drive video search error: {e}")
        return []


@traceable(
    metadata={
        "agent_name": "graphics_definition_v2",
        "step_name": "Drive Video Search",
        "function_name": "process_drive_video_search_segment",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous"),
    }
)
def process_drive_video_search_segment(segment_num, queries, drive=None, k=search_k, drive_video_mode=DRIVE_VIDEO_MODE_ALL):
    """
    Process a single segment: execute Drive video queries in parallel, deduplicate, format output.

    :param segment_num: Segment number
    :param queries: List of search queries for this segment
    :param drive: Google Drive instance
    :param k: Number of results per query
    :param drive_video_mode: "all" or "nextech"
    :return: Tuple of (segment_num, segment_output_string) or (segment_num, None) if no results
    """
    print(f"\n{'─' * 45}")
    print(f"📦 Processing Drive video SEGMENT_{segment_num} with {len(queries)} queries")
    print(f"{'─' * 45}")

    valid_queries = [q.strip() for q in queries if q and str(q).strip()]
    if not valid_queries:
        return segment_num, None

    all_lines = []
    seen = set()

    print(f"🚀 SEGMENT_{segment_num}: submitting {len(valid_queries)} Drive video query search task(s) in parallel")
    query_results = run_pool_search_queries(
        valid_queries,
        execute_drive_video_search_for_query,
        drive=drive,
        k=k,
        drive_video_mode=drive_video_mode,
    )

    for query_idx, (query, results) in enumerate(query_results, 1):
        print(f'🔍 Query {query_idx}: "{query}"')
        results = results or []
        print(f"✅ Query {query_idx} returned {len(results)} Drive video hits")
        for item in results:
            key = result_dedupe_key(item)
            if key in seen:
                continue
            line = format_drive_video_result_line(item)
            if not line:
                continue
            seen.add(key)
            all_lines.append(line)

    print(f"✅ SEGMENT_{segment_num}: Found {len(all_lines)} unique Drive video clips")

    if all_lines:
        segment_output = [f"---SEGMENT_{segment_num}---"] + all_lines
        return segment_num, "\n".join(segment_output)
    return segment_num, None


@traceable(
    metadata={
        "agent_name": "graphics_definition_v2",
        "step_name": "Drive Video Search",
        "function_name": "process_drive_video_search_row",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous"),
    }
)
def process_drive_video_search_row(index, row, drive=None, k=search_k, drive_video_mode=DRIVE_VIDEO_MODE_ALL):
    """Process one sheet row into drive_video_pool text."""
    try:
        search_queries_text = str(row.get("search_queries", "")).strip()
        if not search_queries_text or search_queries_text == "nan":
            return index, ""

        segments = parse_search_queries_column(search_queries_text)
        if not segments:
            return index, ""

        print(f"🚀 Row {index + 1}: submitting {len(segments)} Drive video segment task(s) in parallel")

        with ThreadPoolExecutor(max_workers=len(segments)) as executor:
            futures = {
                executor.submit(
                    process_drive_video_search_segment,
                    segment_num,
                    queries,
                    drive,
                    k,
                    drive_video_mode,
                ): segment_num
                for segment_num, queries in segments
            }

            segment_results = {}
            for future in as_completed(futures):
                segment_num = futures[future]
                try:
                    seg_num, segment_output = future.result()
                    if segment_output:
                        segment_results[seg_num] = segment_output
                except Exception as e:
                    print(f"❌ Error processing Drive video segment {segment_num}: {e}")

        all_segment_results = [segment_results[seg] for seg in sorted(segment_results.keys())]
        return index, "\n\n".join(all_segment_results)
    except Exception as e:
        print(f"Error processing Drive video row {index}: {e}")
        return index, ""


def validate_drive_video_pool_row(row):
    """Validate drive_video_pool against search_queries segment markers."""
    search_queries_text = str(row.get("search_queries", "")).strip()
    drive_video_pool_text = str(row.get(DRIVE_VIDEO_POOL_COLUMN, "")).strip()
    slide_type = str(row.get("Slide Type", "")).strip().lower()

    if slide_type == "transition":
        return True, None

    if not search_queries_text or search_queries_text == "nan":
        return True, None

    if not drive_video_pool_text or drive_video_pool_text == "nan":
        return False, f"{DRIVE_VIDEO_POOL_COLUMN} is empty"

    if drive_video_pool_text.startswith("ERROR:"):
        return False, f"{DRIVE_VIDEO_POOL_COLUMN} contains error marker"

    expected_segments = parse_search_queries_column(search_queries_text)
    if not expected_segments:
        return True, None

    expected_segment_nums = sorted([seg_num for seg_num, _ in expected_segments])
    actual_segment_nums = [int(match) for match in re.findall(r"---SEGMENT_(\d+)---", drive_video_pool_text)]

    if not actual_segment_nums:
        return False, f"No segment markers found in {DRIVE_VIDEO_POOL_COLUMN}"

    if len(actual_segment_nums) != len(expected_segment_nums):
        return (
            False,
            f"Segment count mismatch: expected {len(expected_segment_nums)}, found {len(actual_segment_nums)}",
        )

    actual_sorted = sorted(actual_segment_nums)
    if actual_sorted != expected_segment_nums:
        missing = set(expected_segment_nums) - set(actual_sorted)
        extra = set(actual_sorted) - set(expected_segment_nums)
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
        "step_name": "Drive Video Search",
        "function_name": "run_drive_video_search_for_all_rows",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous"),
    }
)
def run_drive_video_search_for_all_rows(sheet, k=search_k, max_workers=50, selected_topics=None, drive_video_mode=None):
    """
    Search Google Drive videos for all Slide Chunks rows and write drive_video_pool.

    Always-on companion to HVAC video_pool / Drive image drive_results.
    """
    worksheet_name = "Slide Chunks"

    # Resolve mode on the main thread. Worker threads cannot read Streamlit session_state.
    if drive_video_mode is None:
        drive_video_mode = st.session_state.get(UI_KEY_DRIVE_VIDEO_MODE, DRIVE_VIDEO_MODE_ALL)
    drive_video_mode = str(drive_video_mode or DRIVE_VIDEO_MODE_ALL).strip().lower()
    if drive_video_mode not in (DRIVE_VIDEO_MODE_ALL, DRIVE_VIDEO_MODE_NEXTECH):
        drive_video_mode = DRIVE_VIDEO_MODE_ALL
    print(f"🎯 Google Drive video search mode: {drive_video_mode}")

    drive = get_drive_instance()
    if not drive:
        print("❌ Drive instance not available. Cannot execute Drive video search.")
        return

    print("🚀 Preloading Google Drive video embeddings vectorstore before submitting row workers...")
    try:
        load_gemini_video_chroma_collection(drive=drive)
    except Exception as e:
        print(f"❌ Failed to preload Google Drive video embeddings vectorstore: {e}")
        return
    print("✅ Google Drive video embeddings vectorstore is warm and ready for parallel row processing")

    if drive_video_mode == DRIVE_VIDEO_MODE_NEXTECH:
        # Discover tags once on the main thread so workers reuse the cache.
        tags = get_discovered_nextech_tags(drive)
        print(f"🏷️ NexTech-only mode will filter to source_tag in: {tags}")

    worksheet, df = get_sheet_data_and_df(sheet, worksheet_name)

    if DRIVE_VIDEO_POOL_COLUMN not in df.columns:
        df[DRIVE_VIDEO_POOL_COLUMN] = ""

    futures_map = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for index, row in df.iterrows():
            topic_name = str(row.get("Topic", "")).strip()
            if selected_topics and topic_name not in selected_topics:
                continue

            search_queries = str(row.get("search_queries", "")).strip()
            drive_video_pool = str(row.get(DRIVE_VIDEO_POOL_COLUMN, "")).strip()
            slide_type = str(row.get("Slide Type", "")).strip().lower()

            if slide_type == "transition":
                continue
            if not search_queries or search_queries == "nan":
                continue
            if drive_video_pool and drive_video_pool != "nan":
                continue

            future = executor.submit(
                process_drive_video_search_row, index, row, drive, k, drive_video_mode
            )
            futures_map[future] = index

        if not futures_map:
            print("All rows already processed or no valid search queries found for Drive video search.")
            return

        total_tasks = len(futures_map)
        print(f"🚀 Submitted {total_tasks} row-level Drive video search task(s) with max_workers={max_workers}")
        progress = SmartProgressBar(
            total_tasks=total_tasks,
            description="Executing Google Drive video search",
            save_interval=5,
        )

        for future in as_completed(futures_map):
            index = futures_map[future]
            try:
                row_index, drive_video_pool_text = future.result()
                df.at[row_index, DRIVE_VIDEO_POOL_COLUMN] = drive_video_pool_text
                progress.update()
                if progress.should_save():
                    print(f"Saving partial progress to sheet after {progress.completed_count} tasks completed.")
                    merge_and_save_columns(sheet, worksheet_name, df, [DRIVE_VIDEO_POOL_COLUMN])
            except Exception as e:
                print(f"Error getting Drive video result for row {index}: {e}")
                df.at[index, DRIVE_VIDEO_POOL_COLUMN] = f"ERROR: {str(e)}"
                progress.update()

    merge_and_save_columns(sheet, worksheet_name, df, [DRIVE_VIDEO_POOL_COLUMN])
    format_worksheet(worksheet)

    max_retries = 3
    retry_count = 0

    while retry_count < max_retries:
        worksheet, df = get_sheet_data_and_df(sheet, worksheet_name)

        invalid_rows = []
        for index, row in df.iterrows():
            topic_name = str(row.get("Topic", "")).strip()
            if selected_topics and topic_name not in selected_topics:
                continue
            is_valid, error_msg = validate_drive_video_pool_row(row)
            if not is_valid:
                invalid_rows.append((index, row, error_msg))

        if not invalid_rows:
            break

        retry_count += 1
        print(
            f"\n⚠️ Found {len(invalid_rows)} rows with invalid {DRIVE_VIDEO_POOL_COLUMN}. "
            f"Retrying (attempt {retry_count}/{max_retries})..."
        )
        for idx, row, error in invalid_rows[:3]:
            print(f"  Row {idx}: {error}")

        for index, row, error_msg in invalid_rows:
            df.at[index, DRIVE_VIDEO_POOL_COLUMN] = ""

        merge_and_save_columns(sheet, worksheet_name, df, [DRIVE_VIDEO_POOL_COLUMN])
        format_worksheet(worksheet)

        futures_map = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for index, row, error_msg in invalid_rows:
                topic_name = str(row.get("Topic", "")).strip()
                if selected_topics and topic_name not in selected_topics:
                    continue
                future = executor.submit(process_drive_video_search_row, index, row, drive, k, drive_video_mode)
                futures_map[future] = index

            for future in as_completed(futures_map):
                index = futures_map[future]
                try:
                    row_index, drive_video_pool_text = future.result()
                    df.at[row_index, DRIVE_VIDEO_POOL_COLUMN] = drive_video_pool_text
                except Exception as e:
                    print(f"Error getting Drive video result for row {index} on retry: {e}")
                    df.at[index, DRIVE_VIDEO_POOL_COLUMN] = f"ERROR: {str(e)}"

        merge_and_save_columns(sheet, worksheet_name, df, [DRIVE_VIDEO_POOL_COLUMN])
        format_worksheet(worksheet)

    if retry_count > 0:
        worksheet, df = get_sheet_data_and_df(sheet, worksheet_name)
        final_invalid = []
        for index, row in df.iterrows():
            topic_name = str(row.get("Topic", "")).strip()
            if selected_topics and topic_name not in selected_topics:
                continue
            is_valid, error_msg = validate_drive_video_pool_row(row)
            if not is_valid:
                final_invalid.append((index, error_msg))

        if final_invalid:
            print(
                f"⚠️ After {retry_count} retry attempt(s), {len(final_invalid)} rows still have invalid "
                f"{DRIVE_VIDEO_POOL_COLUMN}."
            )
            for idx, error in final_invalid[:5]:
                print(f"  Row {idx}: {error}")
        else:
            print(f"✅ All rows validated after {retry_count} retry attempt(s).")

    print("All Google Drive video searches completed. Saving final DataFrame to sheet.")
    merge_and_save_columns(sheet, worksheet_name, df, [DRIVE_VIDEO_POOL_COLUMN])
    format_worksheet(worksheet)
    print("✅ Google Drive video search complete and saved to sheet.")


def delete_drive_video_pool(sheet):
    """Remove the drive_video_pool column from Slide Chunks."""
    worksheet_name = "Slide Chunks"
    ws, df = get_sheet_data_and_df(sheet, worksheet_name)
    if DRIVE_VIDEO_POOL_COLUMN in df.columns:
        df = df.drop(columns=[DRIVE_VIDEO_POOL_COLUMN])
        clear_worksheet(ws)
        save_to_sheet(ws, df)
        print(f"🗑️ Deleted '{DRIVE_VIDEO_POOL_COLUMN}' column from '{worksheet_name}' worksheet")
    else:
        print(f"ℹ️ '{DRIVE_VIDEO_POOL_COLUMN}' column does not exist in '{worksheet_name}' worksheet")
