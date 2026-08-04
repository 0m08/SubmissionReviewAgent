"""
External Reference candidate search for Graphics Definition V2.

Reads search_queries per slide row, searches Supabase external_ref_assets (filtered by sheet_id; images + video segments), writes segment-aligned results to external_ref_pool (one column for both modalities).
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse, parse_qs

import streamlit as st
from langsmith import traceable

from agents.graphics_definition_v2.candidate_search.search_wrapper import run_pool_search_queries
from agents.graphics_definition_v2.external_references.external_reference_extraction import (
    _resolve_sheet_id,
    parse_drive_file_id,
)
from agents.graphics_definition_v2.external_references.vectorstore import (
    search_external_ref_assets,
)
from agents.graphics_definition_v2.image_graphics_agent.drive_search import (
    parse_search_queries_column,
)
from services.sheets_service import (
    format_worksheet,
    get_sheet_data_and_df,
    merge_and_save_columns,
)
from services.smart_progress_bar import SmartProgressBar

search_k = 5
EXTERNAL_REF_POOL_COLUMN = "external_ref_pool"


def _as_int_seconds(value):
    """
    Coerce start/end times to non-negative int seconds.

    :param value: Raw start or end time value from a Supabase hit
    :return: Non-negative integer seconds, or None if invalid or missing
    """
    if value is None or value == "":
        return None
    try:
        return max(0, int(round(float(value))))
    except (TypeError, ValueError):
        return None


def _youtube_id_from_item(item):
    """
    Extract a YouTube video id from an external ref asset dict.

    :param item: Supabase hit dict with video_id and/or asset_url
    :return: YouTube video id string, or None if not found
    """
    vid = str(item.get("video_id") or "").strip()
    if vid:
        return vid
    url = str(item.get("asset_url") or "").strip()
    if not url:
        return None
    parsed = urlparse(url)
    host = (parsed.netloc or "").lower()
    if "youtu.be" in host:
        return (parsed.path or "").strip("/").split("/")[0] or None
    if "youtube.com" in host:
        qs = parse_qs(parsed.query or "")
        if qs.get("v") and qs["v"][0]:
            return qs["v"][0]
        parts = [p for p in (parsed.path or "").split("/") if p]
        if len(parts) >= 2 and parts[0] in {"embed", "shorts", "live", "v"}:
            return parts[1]
    return None


def is_external_ref_video_url(url):
    """
    Return True if a pool URL line is a video clip (not a still image).

    :param url: Formatted pool URL line or raw URL string
    :return: True when the URL represents a video clip
    """
    text = (url or "").strip().lower()
    if not text:
        return False
    if "(start=" in text and "&end=" in text:
        return True
    if "youtube.com/embed/" in text or "youtu.be/" in text:
        return True
    if "youtube.com/watch" in text:
        return True
    return False


def _normalize_drive_url_usp_drivesdk(url):
    """
    Normalize a Google Drive file URL to the canonical usp=drivesdk form.

    :param url: Drive URL (may include usp=sharing or no usp)
    :return: Canonical Drive view URL with usp=drivesdk, or the original URL if not a Drive file link
    """
    text = (url or "").strip()
    if not text or "drive.google.com" not in text.lower():
        return text
    base = text.split("(start=")[0].strip()
    file_id = parse_drive_file_id(base)
    if file_id:
        return f"https://drive.google.com/file/d/{file_id}/view?usp=drivesdk"
    # Fallback: force/replace usp when the id cannot be parsed.
    if "usp=drivesdk" in base:
        return base
    if "usp=" in base:
        return base.replace("usp=sharing", "usp=drivesdk").replace("usp=drive_link", "usp=drivesdk")
    if "?" in base:
        return f"{base}&usp=drivesdk"
    return f"{base}?usp=drivesdk"


def format_external_ref_result_line(item):
    """
    Format one Supabase hit into a sheet line for external_ref_pool.

    :param item: Supabase hit dict (image or video segment)
    :return: Formatted Title/URL line, or None if the hit cannot be formatted
    """
    asset_type = str(item.get("asset_type") or "").strip().lower()
    source_kind = str(item.get("source_kind") or "").strip().lower()
    url = str(item.get("asset_url") or "").strip()
    if not url:
        return None

    title = str(item.get("title") or "").strip()
    start = _as_int_seconds(item.get("start_time"))
    end = _as_int_seconds(item.get("end_time"))

    if asset_type == "video_segment" or source_kind in {"drive_video", "youtube_video"}:
        if source_kind == "youtube_video" or "youtube.com" in url.lower() or "youtu.be" in url.lower():
            video_id = _youtube_id_from_item(item)
            if not video_id or start is None or end is None or end <= start:
                return None
            title = title or f"YouTube {video_id}"
            clip_url = f"https://www.youtube.com/embed/{video_id}?start={start}&end={end}"
            return f"Title: {title} | URL: {clip_url}"

        # Drive (and any other non-YouTube) video segment
        if start is None or end is None or end <= start:
            return None
        title = title or "External ref video"
        base = _normalize_drive_url_usp_drivesdk(url)
        clip_url = f"{base} (start={start}&end={end})"
        return f"Title: {title} | URL: {clip_url}"

    # Image (default)
    title = title or "External ref image"
    return f"Title: {title} | URL: {_normalize_drive_url_usp_drivesdk(url)}"


def result_dedupe_key(item):
    """
    Build a stable deduplication key for an external ref hit.

    :param item: Supabase hit dict
    :return: Dedupe key string (asset_id or asset_url)
    """
    return str(item.get("asset_id") or item.get("asset_url") or "").strip()


@traceable(
    metadata={
        "agent_name": "graphics_definition_v2",
        "step_name": "External Reference Search",
        "function_name": "execute_external_ref_search_for_query",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous"),
    }
)
def execute_external_ref_search_for_query(query, sheet_id=None, k=search_k):
    """
    Search external_ref_assets for one text query (images + video segments, sheet-scoped).

    :param query: Search query string
    :param sheet_id: Course spreadsheet id (required for filter-before-retrieval)
    :param k: Max hits per modality
    :return: List of hit dicts from Supabase RPC
    """
    print(f'🔎 External ref search: "{query}" (sheet_id={sheet_id})')
    if not sheet_id:
        print("⚠️ sheet_id missing; skipping external ref search")
        return []
    try:
        sid = str(sheet_id).strip()
        images = search_external_ref_assets(
            query=query,
            sheet_id=sid,
            k=k,
            asset_type="image",
        ) or []
        videos = search_external_ref_assets(
            query=query,
            sheet_id=sid,
            k=k,
            asset_type="video_segment",
        ) or []
        results = list(images) + list(videos)
        print(
            f"✅ External ref search returned {len(images)} image + "
            f"{len(videos)} video hit(s)"
        )
        return results
    except Exception as exc:
        print(f"❌ External ref search error: {exc}")
        return []


@traceable(
    metadata={
        "agent_name": "graphics_definition_v2",
        "step_name": "External Reference Search",
        "function_name": "process_external_ref_search_segment",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous"),
    }
)
def process_external_ref_search_segment(segment_num, queries, sheet_id=None, k=search_k):
    """
    Run all queries for one segment and return a formatted segment block.

    :param segment_num: Segment index used in pool markers
    :param queries: List of search query strings for this segment
    :param sheet_id: Course spreadsheet id passed to Supabase search
    :param k: Max hits per modality per query
    :return: Tuple of (segment_num, formatted segment text or None)
    """
    print(f"\n{'─' * 45}")
    print(f"📦 External ref SEGMENT_{segment_num} with {len(queries)} queries")
    print(f"{'─' * 45}")

    valid_queries = [q.strip() for q in queries if q and str(q).strip()]
    if not valid_queries:
        return segment_num, None

    all_lines = []
    seen = set()
    query_results = run_pool_search_queries(
        valid_queries,
        execute_external_ref_search_for_query,
        sheet_id=sheet_id,
        k=k,
    )
    for query_idx, (query, results) in enumerate(query_results, 1):
        print(f'🔍 Query {query_idx}: "{query}"')
        for item in results or []:
            key = result_dedupe_key(item)
            if not key or key in seen:
                continue
            line = format_external_ref_result_line(item)
            if not line:
                continue
            seen.add(key)
            all_lines.append(line)

    print(f"✅ SEGMENT_{segment_num}: {len(all_lines)} unique external ref asset(s)")
    if all_lines:
        return segment_num, "\n".join([f"---SEGMENT_{segment_num}---"] + all_lines)
    return segment_num, None


@traceable(
    metadata={
        "agent_name": "graphics_definition_v2",
        "step_name": "External Reference Search",
        "function_name": "process_external_ref_search_row",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous"),
    }
)
def process_external_ref_search_row(index, row, sheet_id=None, k=search_k):
    """
    Process one Slide Chunks row into external_ref_pool text.

    :param index: DataFrame row index
    :param row: Slide Chunks row series with search_queries
    :param sheet_id: Course spreadsheet id passed to segment search
    :param k: Max hits per modality per query
    :return: Tuple of (index, formatted external_ref_pool cell text)
    """
    try:
        search_queries_text = str(row.get("search_queries", "")).strip()
        if not search_queries_text or search_queries_text == "nan":
            return index, ""
        segments = parse_search_queries_column(search_queries_text)
        if not segments:
            return index, ""

        segment_results = {}
        with ThreadPoolExecutor(max_workers=len(segments)) as executor:
            futures = {
                executor.submit(
                    process_external_ref_search_segment,
                    segment_num,
                    queries,
                    sheet_id,
                    k,
                ): segment_num
                for segment_num, queries in segments
            }
            for future in as_completed(futures):
                seg_num = futures[future]
                try:
                    out_num, text = future.result()
                    segment_results[out_num] = text
                except Exception as exc:
                    print(f"❌ External ref segment {seg_num} failed: {exc}")
                    segment_results[seg_num] = None

        blocks = []
        for segment_num, _queries in segments:
            text = segment_results.get(segment_num)
            if text:
                blocks.append(text)
        return index, "\n\n".join(blocks)
    except Exception as exc:
        print(f"❌ External ref row {index} failed: {exc}")
        return index, f"ERROR: {exc}"


def validate_external_ref_search_row(row):
    """
    Light validation: if search_queries exist, column should not be ERROR.

    :param row: Slide Chunks row series
    :return: Tuple of (is_valid, error_message)
    """
    search_queries_text = str(row.get("search_queries", "")).strip()
    if not search_queries_text or search_queries_text == "nan":
        return True, ""
    cell = str(row.get(EXTERNAL_REF_POOL_COLUMN, "")).strip()
    if cell.startswith("ERROR:"):
        return False, cell
    return True, ""


@traceable(
    metadata={
        "agent_name": "graphics_definition_v2",
        "step_name": "External Reference Search",
        "function_name": "run_external_ref_search_for_all_rows",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous"),
    }
)
def run_external_ref_search_for_all_rows(sheet, k=search_k, max_workers=50, selected_topics=None):
    """
    Execute external-ref search for Slide Chunks rows with search_queries and write external_ref_pool.

    :param sheet: gspread sheet object
    :param k: Max hits per modality per query
    :param max_workers: Row-level thread pool size
    :param selected_topics: Optional topic name filter; None processes all rows
    :return: None
    """
    worksheet_name = "Slide Chunks"
    sheet_id = _resolve_sheet_id(sheet)
    if not sheet_id:
        print("❌ Could not resolve sheet_id; cannot filter external_ref_assets")
        return

    worksheet, df = get_sheet_data_and_df(sheet, worksheet_name)
    if EXTERNAL_REF_POOL_COLUMN not in df.columns:
        df[EXTERNAL_REF_POOL_COLUMN] = ""

    futures_map = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for index, row in df.iterrows():
            topic_name = str(row.get("Topic", "")).strip()
            if selected_topics and topic_name not in selected_topics:
                continue
            search_queries = str(row.get("search_queries", "")).strip()
            existing = str(row.get(EXTERNAL_REF_POOL_COLUMN, "")).strip()
            if not search_queries or search_queries == "nan":
                continue
            if existing and existing != "nan" and not existing.startswith("ERROR:"):
                continue
            futures_map[
                executor.submit(process_external_ref_search_row, index, row, sheet_id, k)
            ] = index

        if not futures_map:
            print("All rows already have external_ref_pool or no search_queries.")
            merge_and_save_columns(sheet, worksheet_name, df, [EXTERNAL_REF_POOL_COLUMN])
            format_worksheet(worksheet)
            return

        progress = SmartProgressBar(
            total_tasks=len(futures_map),
            description="External reference search",
            save_interval=5,
        )
        for future in as_completed(futures_map):
            index = futures_map[future]
            try:
                row_index, text = future.result()
                df.at[row_index, EXTERNAL_REF_POOL_COLUMN] = text
            except Exception as exc:
                df.at[index, EXTERNAL_REF_POOL_COLUMN] = f"ERROR: {exc}"
            progress.update()
            if progress.should_save():
                merge_and_save_columns(sheet, worksheet_name, df, [EXTERNAL_REF_POOL_COLUMN])

    merge_and_save_columns(sheet, worksheet_name, df, [EXTERNAL_REF_POOL_COLUMN])
    format_worksheet(worksheet)

    # One retry pass for ERROR rows
    worksheet, df = get_sheet_data_and_df(sheet, worksheet_name)
    retry_rows = []
    for index, row in df.iterrows():
        topic_name = str(row.get("Topic", "")).strip()
        if selected_topics and topic_name not in selected_topics:
            continue
        ok, _ = validate_external_ref_search_row(row)
        if not ok:
            retry_rows.append((index, row))
    if retry_rows:
        print(f"⚠️ Retrying {len(retry_rows)} external_ref_pool ERROR row(s)...")
        for index, row in retry_rows:
            df.at[index, EXTERNAL_REF_POOL_COLUMN] = ""
        merge_and_save_columns(sheet, worksheet_name, df, [EXTERNAL_REF_POOL_COLUMN])
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures_map = {
                executor.submit(process_external_ref_search_row, index, row, sheet_id, k): index
                for index, row in retry_rows
            }
            for future in as_completed(futures_map):
                index = futures_map[future]
                try:
                    row_index, text = future.result()
                    df.at[row_index, EXTERNAL_REF_POOL_COLUMN] = text
                except Exception as exc:
                    df.at[index, EXTERNAL_REF_POOL_COLUMN] = f"ERROR: {exc}"
        merge_and_save_columns(sheet, worksheet_name, df, [EXTERNAL_REF_POOL_COLUMN])
        format_worksheet(worksheet)

    print(f"✅ External ref search complete (column={EXTERNAL_REF_POOL_COLUMN})")


def delete_external_ref_pool(sheet):
    """
    Clear external_ref_pool column on Slide Chunks (does not delete Supabase rows).

    :param sheet: gspread sheet object
    :return: None
    """
    worksheet_name = "Slide Chunks"
    try:
        worksheet, df = get_sheet_data_and_df(sheet, worksheet_name)
        if EXTERNAL_REF_POOL_COLUMN not in df.columns:
            print(f"ℹ️ '{EXTERNAL_REF_POOL_COLUMN}' column does not exist")
            return
        df[EXTERNAL_REF_POOL_COLUMN] = ""
        merge_and_save_columns(sheet, worksheet_name, df, [EXTERNAL_REF_POOL_COLUMN])
        print(f"🗑️ Cleared '{EXTERNAL_REF_POOL_COLUMN}' (Supabase index kept)")
    except Exception as exc:
        print(f"⚠️ Could not clear {EXTERNAL_REF_POOL_COLUMN}: {exc}")
