from langsmith import traceable
import streamlit as st
from services.sheets_service import get_sheet_data_and_df, save_to_sheet, merge_and_save_columns, format_worksheet, clear_worksheet
from services.smart_progress_bar import SmartProgressBar
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import uuid
from agents.vector_store_image_search.graphics_retriever import graphics_retriever
from agents.graphics_definition_v2.candidate_search.search_wrapper import run_pool_search_queries
from dotenv import load_dotenv
import os
import json
import base64
from services.drive_service import login_with_service_account
from pydrive2.drive import GoogleDrive

load_dotenv()

# Configuration constants
search_k = 5  # Number of results to retrieve per search query
relevance_threshold = 1.5  # Minimum similarity/relevance score to consider a reference


DRIVE_IMAGE_VECTORSTORE_V1_ID = "1QS6PmCESfgFWNNEpRJDUB0E-t8iMAatH"
DRIVE_IMAGE_VECTORSTORE_V2_ID = "1IMGr4d8lwux5R_cAWfhVjBV0fTFdWvNi"
CENTRAL_DRIVE_IMAGE_VECTORSTORE_IDS = (
    DRIVE_IMAGE_VECTORSTORE_V1_ID,
    DRIVE_IMAGE_VECTORSTORE_V2_ID,
)


def _get_drive_instance():
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


def _normalize_drive_url(url):
    """
    Normalize a Drive URL for dedupe across vectorstore versions.

    :param url: Raw Drive URL string
    :return: Normalized URL key, or empty string
    """
    text = (url or "").strip()
    if not text:
        return ""
    match = re.search(r"/d/([a-zA-Z0-9_-]+)", text)
    if match:
        return match.group(1)
    return text


def _convert_retriever_hits_to_refs(results, source_label=""):
    """
    Convert graphics_retriever hits into the Drive-search reference shape.

    :param results: Raw retriever result list
    :param source_label: Optional label stored in metadata (e.g. v1 / v2)
    :return: List of reference dicts
    """
    reference_data_list = []
    for result in results or []:
        metadata = result.get("metadata", {}) or {}
        ref_id = metadata.get("image_id", str(uuid.uuid4()))
        url = metadata.get("drive_url", "")
        similarity_score = result.get("similarity", 1.0)
        relevance_score = max(0.0, min(1.0, 1.0 - similarity_score))
        title = metadata.get("image_title") or metadata.get("name", "Untitled")
        enriched_metadata = {
            **metadata,
            "similarity_distance": similarity_score,
        }
        if source_label:
            enriched_metadata["vectorstore_source"] = source_label
        reference_data_list.append({
            "reference_id": ref_id,
            "url": url,
            "title": title,
            "relevance_score": relevance_score,
            "metadata": enriched_metadata,
        })
    return reference_data_list


def _search_one_drive_image_vectorstore(query, drive, k, filters, root_folder_id, source_label=""):
    """
    Search a single Drive image vectorstore and return filtered reference dicts.

    :param query: Search query string
    :param drive: Google Drive instance
    :param k: Max results to request from this store
    :param filters: Optional filters for image search
    :param root_folder_id: Vectorstore parent folder ID
    :param source_label: Optional label for logging/metadata
    :return: List of reference dicts that pass the relevance threshold
    """
    label = source_label or root_folder_id
    try:
        results = graphics_retriever(
            query=query,
            drive=drive,
            k=k,
            filters=filters,
            load_images=False,
            root_folder_id=root_folder_id,
        )
        print(f"  📁 {label}: raw={len(results)} items")
    except Exception as e:
        print(f"  ❌ {label}: search failed: {e}")
        return []

    refs = _convert_retriever_hits_to_refs(results, source_label=source_label)
    filtered = [
        ref for ref in refs
        if ref["metadata"].get("similarity_distance", 999) <= relevance_threshold
    ]
    print(f"  🎯 {label}: after relevance filter (≤{relevance_threshold}): {len(filtered)}")
    return filtered


def _merge_and_rank_drive_image_refs(ref_lists, k=search_k):
    """
    Merge multi-store hits, dedupe, and keep the best overall top-k.

    Ranking uses similarity_distance ascending (lower distance = better match).

    :param ref_lists: Iterable of reference-dict lists
    :param k: Max results to keep after merge
    :return: Deduped, ranked list of up to k references
    """
    merged = []
    seen_ids = set()
    seen_urls = set()

    for refs in ref_lists:
        for ref in refs or []:
            ref_id = (ref.get("reference_id") or "").strip()
            url_key = _normalize_drive_url(ref.get("url", ""))
            if ref_id and ref_id in seen_ids:
                continue
            if url_key and url_key in seen_urls:
                continue
            if ref_id:
                seen_ids.add(ref_id)
            if url_key:
                seen_urls.add(url_key)
            merged.append(ref)

    merged.sort(
        key=lambda ref: float(ref.get("metadata", {}).get("similarity_distance", 999))
    )
    return merged[: max(0, int(k))]


def _count_refs_by_vectorstore_source(refs):
    """
    Count how many references came from each vectorstore source label.

    :param refs: Iterable of reference dicts
    :return: Dict like {"v1": n, "v2": n, "single": n, "unknown": n}
    """
    counts = {"v1": 0, "v2": 0, "single": 0, "unknown": 0}
    for ref in refs or []:
        source = str((ref.get("metadata") or {}).get("vectorstore_source", "")).strip().lower()
        if source in counts:
            counts[source] += 1
        else:
            counts["unknown"] += 1
    return counts


def _format_vectorstore_source_counts(counts):
    """
    Build a short log string for v1/v2 source counts.

    :param counts: Dict from _count_refs_by_vectorstore_source
    :return: Human-readable count string
    """
    counts = counts or {}
    parts = [
        f"v1={counts.get('v1', 0)}",
        f"v2={counts.get('v2', 0)}",
    ]
    single = counts.get("single", 0)
    unknown = counts.get("unknown", 0)
    if single:
        parts.append(f"single={single}")
    if unknown:
        parts.append(f"unknown={unknown}")
    return ", ".join(parts)


def _resolve_central_vectorstore_ids(root_folder_ids=None):
    """
    Resolve which central Drive image vectorstore folders to search.

    :param root_folder_ids: Optional explicit list/tuple of folder IDs
    :return: List of folder IDs
    """
    if root_folder_ids:
        resolved = []
        seen = set()
        for folder_id in root_folder_ids:
            fid = str(folder_id or "").strip()
            if not fid or fid in seen:
                continue
            seen.add(fid)
            resolved.append(fid)
        if resolved:
            return resolved
    return list(CENTRAL_DRIVE_IMAGE_VECTORSTORE_IDS)


@traceable(
    metadata={
        "agent_name": "graphics_definition_v2",
        "step_name": "Drive Search",
        "function_name": "execute_drive_search_for_query",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous")
    }
)
def execute_drive_search_for_query(query, drive, k=search_k, filters=None, root_folder_id=DRIVE_IMAGE_VECTORSTORE_V1_ID, use_both_central_vectorstores=False, root_folder_ids=None):
    """
    Execute drive search for a single query and return reference data.
    
    :param query: Search query string
    :param drive: Google Drive instance
    :param k: Number of results to retrieve (final top-k after merge)
    :param filters: Optional filters for image search
    :param root_folder_id: Single vectorstore folder ID (used when not searching both)
    :param use_both_central_vectorstores: If True, search central v1+v2 and return combined top-k
    :param root_folder_ids: Optional explicit multi-store folder list (overrides the default pair)
    :return: List of reference data dictionaries with url, reference_id, etc.
    """
    if not drive:
        print(f"⚠️  Drive instance not available for search")
        return []

    print(f"🔎 Executing drive search for query: \"{query}\"")

    if use_both_central_vectorstores or root_folder_ids:
        folder_ids = _resolve_central_vectorstore_ids(root_folder_ids)
        source_labels = {
            DRIVE_IMAGE_VECTORSTORE_V1_ID: "v1",
            DRIVE_IMAGE_VECTORSTORE_V2_ID: "v2",
        }
        print(
            f"📚 Searching {len(folder_ids)} Drive image vectorstore(s) "
            f"and keeping combined top-{k}"
        )

        ref_lists = []
        pre_merge_by_source = {"v1": 0, "v2": 0, "single": 0, "unknown": 0}
        # Search stores in parallel; each still returns up to k before merge.
        with ThreadPoolExecutor(max_workers=max(1, len(folder_ids))) as executor:
            futures = {
                executor.submit(
                    _search_one_drive_image_vectorstore,
                    query,
                    drive,
                    k,
                    filters,
                    folder_id,
                    source_labels.get(folder_id, folder_id),
                ): folder_id
                for folder_id in folder_ids
            }
            for future in as_completed(futures):
                folder_id = futures[future]
                try:
                    store_refs = future.result() or []
                    ref_lists.append(store_refs)
                    store_counts = _count_refs_by_vectorstore_source(store_refs)
                    for key, value in store_counts.items():
                        pre_merge_by_source[key] = pre_merge_by_source.get(key, 0) + value
                except Exception as e:
                    print(f"  ❌ {folder_id}: future failed: {e}")
                    ref_lists.append([])

        merged = _merge_and_rank_drive_image_refs(ref_lists, k=k)
        final_counts = _count_refs_by_vectorstore_source(merged)
        print(
            f"✅ Combined Drive image search: "
            f"{sum(len(x or []) for x in ref_lists)} pre-merge "
            f"({_format_vectorstore_source_counts(pre_merge_by_source)}) → "
            f"{len(merged)} top-{k} "
            f"({_format_vectorstore_source_counts(final_counts)})"
        )
        return merged

    # Single-store path (course reference pool / explicit one folder).
    return _search_one_drive_image_vectorstore(
        query=query,
        drive=drive,
        k=k,
        filters=filters,
        root_folder_id=root_folder_id,
        source_label="single",
    )


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
        "step_name": "Drive Search",
        "function_name": "process_drive_search_segment",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous")
    }
)
def process_drive_search_segment(segment_num, queries, drive, k=search_k, filters=None, root_folder_id=DRIVE_IMAGE_VECTORSTORE_V1_ID, use_both_central_vectorstores=True, root_folder_ids=None):
    """
    Process a single segment: execute queries in parallel, deduplicate, format output.
    
    :param segment_num: Segment number
    :param queries: List of search queries for this segment
    :param drive: Google Drive instance
    :param k: Number of results per query
    :param filters: Optional filters
    :param root_folder_id: Single-store folder ID when use_both_central_vectorstores is False
    :param use_both_central_vectorstores: Search central v1+v2 and keep combined top-k (GDv2 default)
    :param root_folder_ids: Optional explicit multi-store folder list
    :return: Tuple of (segment_num, segment_output_string) or (segment_num, None) if no results
    """
    print(f"\n{'─'*45}")
    print(f"📦 Processing SEGMENT_{segment_num} with {len(queries)} queries")
    print(f"{'─'*45}")
    
    all_results_for_segment = []
    seen_ids = set()  # Deduplicate within segment
    seen_urls = set()

    valid_queries = [q.strip() for q in queries if q and str(q).strip()]
    if not valid_queries:
        return segment_num, None

    print(f"🚀 SEGMENT_{segment_num}: submitting {len(valid_queries)} drive query search task(s) in parallel")
    query_results = run_pool_search_queries(
        valid_queries,
        execute_drive_search_for_query,
        drive=drive,
        k=k,
        filters=filters,
        root_folder_id=root_folder_id,
        use_both_central_vectorstores=use_both_central_vectorstores,
        root_folder_ids=root_folder_ids,
    )

    for query_idx, (query, results) in enumerate(query_results, 1):
        print(f"🔍 Query {query_idx}: \"{query}\"")
        results = results or []
        query_counts = _count_refs_by_vectorstore_source(results)
        print(
            f"✅ Query {query_idx} returned {len(results)} results "
            f"({_format_vectorstore_source_counts(query_counts)})"
        )
        for ref in results:
            ref_id = (ref.get("reference_id") or "").strip()
            url_key = _normalize_drive_url(ref.get("url", ""))
            if ref_id and ref_id in seen_ids:
                continue
            if url_key and url_key in seen_urls:
                continue
            if ref_id:
                seen_ids.add(ref_id)
            if url_key:
                seen_urls.add(url_key)
            all_results_for_segment.append(ref)

    segment_items = []
    for ref in all_results_for_segment:
        url = ref.get("url", "")
        title = ref.get("title", "Untitled")
        if url:
            segment_items.append(f"Title: {title} | URL: {url}")

    segment_counts = _count_refs_by_vectorstore_source(all_results_for_segment)
    print(
        f"✅ SEGMENT_{segment_num}: Found {len(segment_items)} unique results "
        f"({_format_vectorstore_source_counts(segment_counts)})"
    )

    if segment_items:
        segment_output = [f"---SEGMENT_{segment_num}---"] + segment_items
        return segment_num, '\n'.join(segment_output)
    return segment_num, None


@traceable(
    metadata={
        "agent_name": "graphics_definition_v2",
        "step_name": "Drive Search",
        "function_name": "process_drive_search_row",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous")
    }
)
def process_drive_search_row(index, row, drive, k=search_k, filters=None, root_folder_id=DRIVE_IMAGE_VECTORSTORE_V1_ID, use_both_central_vectorstores=True, root_folder_ids=None):
    """
    Process a single row: parse search queries, execute searches for each segment, deduplicate, format output.
    
    :param index: Row index
    :param row: Pandas Series with row data
    :param drive: Google Drive instance
    :param k: Number of results per query
    :param filters: Optional filters
    :param root_folder_id: Single-store folder ID when use_both_central_vectorstores is False
    :param use_both_central_vectorstores: Search central v1+v2 and keep combined top-k (GDv2 default)
    :param root_folder_ids: Optional explicit multi-store folder list
    :return: Tuple of (index, drive_results_text)
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
                executor.submit(
                    process_drive_search_segment,
                    segment_num,
                    queries,
                    drive,
                    k,
                    filters,
                    root_folder_id,
                    use_both_central_vectorstores,
                    root_folder_ids,
                ): segment_num
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
        drive_results_text = '\n\n'.join(all_segment_results)
        
        return index, drive_results_text
        
    except Exception as e:
        print(f"Error processing row {index}: {e}")
        return index, ""


def validate_drive_search_row(row):
    """
    Validate that drive_results matches search_queries:
    - Row is not empty
    - All segments from search_queries have results
    - No gaps in segment numbering
    
    :param row: Pandas Series with row data
    :return: Tuple (is_valid, error_message)
    """
    search_queries_text = str(row.get("search_queries", "")).strip()
    drive_results_text = str(row.get("drive_results", "")).strip()
    
    # Skip validation if search_queries is empty
    if not search_queries_text or search_queries_text == "nan":
        return True, None
    
    # Check if drive_results is empty
    if not drive_results_text or drive_results_text == "nan" or drive_results_text.strip() == "":
        return False, "drive_results is empty"
    
    # Check if it's an error marker
    if drive_results_text.startswith("ERROR:"):
        return False, "drive_results contains error marker"
    
    # Parse expected segments from search_queries
    expected_segments = parse_search_queries_column(search_queries_text)
    if not expected_segments:
        return True, None  # No segments to validate
    
    expected_segment_nums = sorted([seg_num for seg_num, _ in expected_segments])
    expected_count = len(expected_segment_nums)
    
    # Parse actual segments from drive_results
    segment_pattern = r'---SEGMENT_(\d+)---'
    actual_segment_nums = [int(match) for match in re.findall(segment_pattern, drive_results_text)]
    
    if not actual_segment_nums:
        return False, "No segment markers found in drive_results"
    
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
        "step_name": "Drive Search",
        "function_name": "run_drive_search_for_all_rows",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous")
    }
)
def run_drive_search_for_all_rows(sheet, k=search_k, filters=None, root_folder_id=DRIVE_IMAGE_VECTORSTORE_V1_ID, max_workers=50, selected_topics=None, use_both_central_vectorstores=True, root_folder_ids=None):
    """
    Execute drive search for all rows in the Slide Chunks sheet.
    
    :param sheet: The gspread sheet object.
    :param k: Number of results per query (default from settings).
    :param filters: Optional filters for image search.
    :param root_folder_id: Single-store folder ID when use_both_central_vectorstores is False.
    :param max_workers: Number of parallel workers (default 5).
    :param selected_topics: Optional topic filter.
    :param use_both_central_vectorstores: Search central v1+v2 and keep combined top-k (GDv2 default).
    :param root_folder_ids: Optional explicit multi-store folder list.
    :return: None
    """
    worksheet_name = "Slide Chunks"
    
    # Get drive instance
    drive = _get_drive_instance()
    if not drive:
        print("❌ Drive instance not available. Cannot execute drive search.")
        return

    if use_both_central_vectorstores or root_folder_ids:
        store_ids = _resolve_central_vectorstore_ids(root_folder_ids)
        print(
            f"📚 Drive image search mode: combined top-{k} from "
            f"{len(store_ids)} vectorstore(s): {store_ids}"
        )
    else:
        print(f"📚 Drive image search mode: single vectorstore {root_folder_id}")
    
    # Load the worksheet and DataFrame
    worksheet, df = get_sheet_data_and_df(sheet, worksheet_name)
    
    # Ensure drive_results column exists
    if "drive_results" not in df.columns:
        df["drive_results"] = ""
    
    # Prepare for parallel processing - only process rows that need processing
    futures_map = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks first
        for index, row in df.iterrows():
            topic_name = str(row.get("Topic", "")).strip()
            if selected_topics and topic_name not in selected_topics:
                continue
            search_queries = str(row.get("search_queries", "")).strip()
            drive_results = str(row.get("drive_results", "")).strip()
            
            # Skip if search_queries is empty
            if not search_queries or search_queries == "nan":
                continue
            
            # Skip if drive_results is already filled
            if drive_results and drive_results != "nan":
                continue
            
            # Submit task for processing
            future = executor.submit(
                process_drive_search_row,
                index,
                row,
                drive,
                k,
                filters,
                root_folder_id,
                use_both_central_vectorstores,
                root_folder_ids,
            )
            futures_map[future] = index
        
        # If no rows to process, return early
        if not futures_map:
            print("All rows already processed or no valid search queries found.")
            return
        
        # Initialize progress tracker
        total_tasks = len(futures_map)
        progress = SmartProgressBar(
            total_tasks=total_tasks,
            description="Executing drive search",
            save_interval=5
        )
        
        # Collect results as they complete
        for future in as_completed(futures_map):
            index = futures_map[future]
            try:
                row_index, drive_results_text = future.result()
                
                # Update dataframe
                df.at[row_index, "drive_results"] = drive_results_text
                
                # Update progress
                progress.update()
                
                # Save every 5 rows
                if progress.should_save():
                    print(f'Saving partial progress to sheet after {progress.completed_count} tasks completed.')
                    merge_and_save_columns(sheet, worksheet_name, df, ["drive_results"])
            except Exception as e:
                print(f"Error getting result for row {index}: {e}")
                # Update dataframe with error marker so row is marked as processed
                df.at[index, "drive_results"] = f"ERROR: {str(e)}"
                progress.update()
    
    # Save final results before validation
    merge_and_save_columns(sheet, worksheet_name, df, ["drive_results"])
    format_worksheet(worksheet)
    
    # Validation and retry logic
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        # Reload dataframe to get latest state
        worksheet, df = get_sheet_data_and_df(sheet, worksheet_name)
        
        # Find rows that need processing (empty or invalid drive_results)
        invalid_rows = []
        for index, row in df.iterrows():
            topic_name = str(row.get("Topic", "")).strip()
            if selected_topics and topic_name not in selected_topics:
                continue
            is_valid, error_msg = validate_drive_search_row(row)
            if not is_valid:
                invalid_rows.append((index, row, error_msg))
        
        if not invalid_rows:
            break
        
        retry_count += 1
        print(f"\n⚠️ Found {len(invalid_rows)} rows with invalid drive_results. Retrying (attempt {retry_count}/{max_retries})...")
        for idx, row, error in invalid_rows[:3]:  # Show first 3 errors
            print(f"  Row {idx}: {error}")
        
        # Clear drive_results for invalid rows
        for index, row, error_msg in invalid_rows:
            df.at[index, "drive_results"] = ""
        
        # Save cleared state
        merge_and_save_columns(sheet, worksheet_name, df, ["drive_results"])
        format_worksheet(worksheet)
        
        # Retry processing invalid rows
        futures_map = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for index, row, error_msg in invalid_rows:
                topic_name = str(row.get("Topic", "")).strip()
                if selected_topics and topic_name not in selected_topics:
                    continue
                future = executor.submit(
                    process_drive_search_row,
                    index,
                    row,
                    drive,
                    k,
                    filters,
                    root_folder_id,
                    use_both_central_vectorstores,
                    root_folder_ids,
                )
                futures_map[future] = index
            
            # Collect results
            for future in as_completed(futures_map):
                index = futures_map[future]
                try:
                    row_index, drive_results_text = future.result()
                    df.at[row_index, "drive_results"] = drive_results_text
                except Exception as e:
                    print(f"Error getting result for row {index} on retry: {e}")
                    df.at[index, "drive_results"] = f"ERROR: {str(e)}"
        
        # Save after retry
        merge_and_save_columns(sheet, worksheet_name, df, ["drive_results"])
        format_worksheet(worksheet)
    
    if retry_count > 0:
        # Check final state
        worksheet, df = get_sheet_data_and_df(sheet, worksheet_name)
        final_invalid = []
        for index, row in df.iterrows():
            is_valid, error_msg = validate_drive_search_row(row)
            if not is_valid:
                final_invalid.append((index, error_msg))
        
        if final_invalid:
            print(f"⚠️ After {retry_count} retry attempt(s), {len(final_invalid)} rows still have invalid drive_results.")
            for idx, error in final_invalid[:5]:  # Show first 5
                print(f"  Row {idx}: {error}")
        else:
            print(f"✅ All rows validated after {retry_count} retry attempt(s).")
    
    # Final save to sheet
    print('All drive searches completed. Saving final DataFrame to sheet.')
    merge_and_save_columns(sheet, worksheet_name, df, ["drive_results"])
    format_worksheet(worksheet)
    print("✅ Drive search complete and saved to sheet.")


def delete_drive_results(sheet):
    """
    Remove the 'drive_results' column from the specified worksheet.
    
    :param sheet: The gspread sheet object.
    :return: None
    """
    worksheet_name = "Slide Chunks"
    ws, df = get_sheet_data_and_df(sheet, worksheet_name)
    if "drive_results" in df.columns:
        df = df.drop(columns=["drive_results"])
        clear_worksheet(ws)
        save_to_sheet(ws, df)
        print(f"🗑️ Deleted 'drive_results' column from '{worksheet_name}' worksheet")
    else:
        print(f"ℹ️ 'drive_results' column does not exist in '{worksheet_name}' worksheet")

