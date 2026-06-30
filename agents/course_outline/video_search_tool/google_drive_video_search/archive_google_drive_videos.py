"""
archive_videos.py
=================
Video archiving pipeline — VS Code / Streamlit edition.

Mirrors the Colab pipeline exactly:
  - Walks ASSETS_ARCHIVE_FOLDER_ID  →  course → topic → assets/  (structured)
  - Optionally walks any ad-hoc folder recursively              (Other Videos tab)
  - Detects duration via Drive metadata → partial byte-range → ffprobe fallback
  - Classifies Stock / Non-Stock in the same run, right after archiving
  - Writes / updates the Google Sheet (same schema as Colab)

Entry points
------------
run_archive_pipeline(drive_service, sheets_client, folder_id="")
    Call from VS Code main block or from Streamlit via
    render_gemini_drive_video_search_tab().

Required environment variables (set in .env or OS env):
    GDRIVE_SA_B64   – base-64-encoded service-account JSON  (preferred)
                      OR fall back to OAuth mycreds.txt / user prompt.
    GOOGLE_API_KEY  – only needed by the search/embedding tab, not archiving.

Dependencies:
    pip install google-api-python-client google-auth gspread gspread-dataframe
                pymediainfo requests python-dotenv
    apt-get / brew install mediainfo   (for pymediainfo)
    ffprobe must be on PATH            (for the final duration fallback)
"""

from __future__ import annotations

import base64
import io
import json
import os
import random
import re
import subprocess
import tempfile
import time
from typing import Optional

import requests
import streamlit as st
from dotenv import load_dotenv
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload
import gspread
from gspread.exceptions import APIError, WorksheetNotFound
from gspread_dataframe import set_with_dataframe
import pandas as pd

try:
    from pymediainfo import MediaInfo
    _MEDIAINFO_AVAILABLE = True
except ImportError:
    _MEDIAINFO_AVAILABLE = False

load_dotenv()

# ============================================================
# CONFIG  (mirrors Colab hard-codes)
# ============================================================

ASSETS_ARCHIVE_FOLDER_ID = "10dP4ysSfdzITI0b4OeBvJttIc_3lFFzm"
SHEET_LINK = "https://docs.google.com/spreadsheets/d/1CceOCu-AER_j3KBRZ66F0Er7z3LznF40a2c73xqr_u0/edit?usp=sharing"
OTHER_VIDEOS_TAB_NAME = "Other Videos"

VIDEO_MIME_TYPES = {
    "video/mp4", "video/quicktime", "video/webm",
    "video/x-msvideo", "video/x-matroska",
}
VIDEO_EXTENSIONS = (".mp4", ".mov", ".webm", ".avi", ".mkv")
ASSETS_FOLDER_NAME = "assets"

HEADERS = [
    "Video ID", "Video Name", "Course Name", "Topic Name",
    "MimeType", "Duration", "File Size", "Video Link", "Stock/Non Stock",
]

SHEETS_WRITE_DELAY_SECONDS = 1.2

STOCK_VIDEO_KEYWORDS = [
    "adobestock", "adobe_", "adobe-", "stock_", "stock-",
    "stockvideo", "licensed", "royaltyfree", "royalty-free",
    "dreamstime", "pond5", "shutterstock", "getty",
    "videoblocks", "storyblocks", "envato",
]


# ============================================================
# AUTH  (service-account preferred; OAuth fallback)
# ============================================================

def _build_services_from_sa(sa_json_str: str):
    """Build Drive + Sheets clients from a service-account JSON string."""
    import google.oauth2.service_account as sa_mod

    info = json.loads(sa_json_str)
    scopes = [
        "https://www.googleapis.com/auth/drive",
        "https://www.googleapis.com/auth/spreadsheets",
    ]
    creds = sa_mod.Credentials.from_service_account_info(info, scopes=scopes)
    drive_service = build("drive", "v3", credentials=creds)
    sheets_client = gspread.authorize(creds)
    return drive_service, sheets_client, creds


def _build_services_from_oauth():
    """Build Drive + Sheets clients via PyDrive2 OAuth (mycreds.txt)."""
    from pydrive2.auth import GoogleAuth
    from pydrive2.drive import GoogleDrive

    gauth = GoogleAuth()
    gauth.LoadCredentialsFile("mycreds.txt")
    if gauth.credentials is None:
        gauth.LocalWebserverAuth()
    elif gauth.access_token_expired:
        gauth.Refresh()
    else:
        gauth.Authorize()
    gauth.SaveCredentialsFile("mycreds.txt")

    raw_creds = gauth.credentials
    drive_service = build("drive", "v3", credentials=raw_creds)
    sheets_client = gspread.authorize(raw_creds)
    return drive_service, sheets_client, raw_creds


def get_archive_clients():
    """
    Return (drive_service, sheets_client, raw_creds).

    Priority:
      1. GDRIVE_SA_B64 env-var  (service account, best for production)
      2. mycreds.txt OAuth file  (local dev)
    """
    sa_b64 = os.environ.get("GDRIVE_SA_B64", "").strip()
    if sa_b64:
        sa_json = base64.b64decode(sa_b64).decode()
        return _build_services_from_sa(sa_json)
    return _build_services_from_oauth()


# ============================================================
# RETRY WRAPPER
# ============================================================

def retry_call(func, *args, max_retries: int = 7, base_sleep: float = 2.0, **kwargs):
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except APIError as exc:
            msg = str(exc)
            if not any(t in msg for t in ("429", "Quota exceeded", "RESOURCE_EXHAUSTED")):
                raise
            if attempt == max_retries - 1:
                raise
        except HttpError as exc:
            status = getattr(exc.resp, "status", None)
            if status not in [403, 429, 500, 502, 503, 504]:
                raise
            if attempt == max_retries - 1:
                raise

        sleep_for = base_sleep * (2 ** attempt) + random.uniform(0, 1.5)
        _log(f"  [retry {attempt + 1}] waiting {sleep_for:.1f}s…")
        time.sleep(sleep_for)


def _sheets_sleep():
    time.sleep(SHEETS_WRITE_DELAY_SECONDS)


# ============================================================
# LOGGING  (print + optional Streamlit placeholder)
# ============================================================

_st_log_placeholder = None  # set by Streamlit caller


def _log(msg: str):
    print(msg)
    if _st_log_placeholder is not None:
        try:
            # Append to a growing log in session_state
            log = st.session_state.get("_archive_log", "")
            log += msg + "\n"
            st.session_state["_archive_log"] = log
            # _st_log_placeholder.text_area(
            #     "Live log", value=log, height=300,
            #     key=f"_archive_log_box_{len(log)}",
            # )
        except Exception:
            pass


# ============================================================
# DURATION HELPERS
# ============================================================

def format_duration(ms_str) -> str:
    """Convert milliseconds (int or str) to hh:mm:ss / mm:ss."""
    if not ms_str:
        return ""
    try:
        ms_val = int(ms_str)
        if ms_val == 0:
            return "0:00"
        total_sec = ms_val / 1000.0
        if total_sec < 1.0:
            return f"0:{total_sec:.2f}"
        total_sec_int = int(total_sec)
        h, rem = divmod(total_sec_int, 3600)
        m, s = divmod(rem, 60)
        return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"
    except (TypeError, ValueError):
        return ""


def format_size(size_str) -> str:
    """Convert bytes (int or str) to MB / GB string."""
    if not size_str:
        return ""
    try:
        b = int(size_str)
        if b >= 1024 ** 3:
            return f"{b / 1024 ** 3:.2f} GB"
        return f"{b / 1024 ** 2:.2f} MB"
    except (TypeError, ValueError):
        return ""


def _get_fresh_token(creds) -> str:
    """Refresh OAuth / SA credentials and return a valid Bearer token."""
    if hasattr(creds, "token"):
        if not creds.valid:
            creds.refresh(Request())
        return creds.token
    # service_account.Credentials
    if not creds.valid:
        creds.refresh(Request())
    return creds.token


def _mediainfo_from_bytes(data: bytes) -> str:
    """Run pymediainfo on in-memory bytes. Returns formatted duration or ''."""
    if not _MEDIAINFO_AVAILABLE:
        return ""
    try:
        with io.BytesIO(data) as buf:
            mi = MediaInfo.parse(buf)
            for track in mi.tracks:
                if track.track_type == "Video" and track.duration:
                    return format_duration(track.duration)
    except Exception:
        pass
    return ""


def _probe_with_ffprobe(drive_service, file_id: str, video_name: str) -> str:
    """Download full file and run ffprobe. Last-resort fallback."""
    tmp_path = os.path.join(
        tempfile.gettempdir(),
        f"probe_{file_id[:12]}_{os.path.basename(video_name)[:40]}",
    )
    try:
        req = drive_service.files().get_media(fileId=file_id, supportsAllDrives=True)
        with open(tmp_path, "wb") as fh:
            dl = MediaIoBaseDownload(fh, req, chunksize=10 * 1024 * 1024)
            done = False
            while not done:
                _, done = dl.next_chunk()

        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                tmp_path,
            ],
            capture_output=True, text=True, timeout=120,
        )
        seconds = float(result.stdout.strip())
        return format_duration(str(int(seconds * 1000)))
    except Exception as exc:
        _log(f"    [ffprobe failed] {video_name}: {exc}")
        return ""
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def get_video_duration(file_metadata: dict, drive_service, creds) -> str:
    """
    Three-stage duration resolution — mirrors Colab exactly.

    Stage 1: Google Drive native videoMediaMetadata.durationMillis
    Stage 2: Partial byte-range via requests → pymediainfo in-memory parse
    Stage 3: Full download → ffprobe
    """
    file_name = file_metadata.get("name", "?")
    file_id   = file_metadata.get("id", "")

    # Stage 1 — native Drive metadata (instant, no download)
    vmd = file_metadata.get("videoMediaMetadata") or {}
    dur_ms = vmd.get("durationMillis")
    if dur_ms and int(dur_ms) > 0:
        return format_duration(dur_ms)

    if not file_id:
        return "0:00"

    # Stage 2 — partial byte-range via requests
    if _MEDIAINFO_AVAILABLE:
        try:
            token = _get_fresh_token(creds)
            stream_url = f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media"
            hdrs = {"Authorization": f"Bearer {token}"}
            file_size_str = file_metadata.get("size")

            if file_size_str:
                file_size = int(file_size_str)
                if file_size <= 1_500_000:
                    r = requests.get(stream_url, headers=hdrs, timeout=60)
                    r.raise_for_status()
                    result = _mediainfo_from_bytes(r.content)
                else:
                    chunk = 500 * 1024
                    hdrs["Range"] = f"bytes=0-{chunk}"
                    r_head = requests.get(stream_url, headers=hdrs, timeout=60)
                    hdrs["Range"] = f"bytes={file_size - chunk}-{file_size}"
                    r_foot = requests.get(stream_url, headers=hdrs, timeout=60)
                    result = _mediainfo_from_bytes(r_head.content + r_foot.content)

                if result:
                    return result
            else:
                # No size info — try 1 MB header only
                hdrs_ns = {"Authorization": f"Bearer {token}"}
                with requests.get(stream_url, headers=hdrs_ns, stream=True, timeout=60) as r:
                    r.raise_for_status()
                    chunk_data = next(r.iter_content(1024 * 1024), b"")
                result = _mediainfo_from_bytes(chunk_data)
                if result:
                    return result
        except Exception as exc:
            _log(f"    [Stage 2 warning] {file_name}: {exc}")

    # Stage 3 — full download + ffprobe
    _log(f"    [Stage 3] ffprobe fallback for '{file_name}'…")
    return _probe_with_ffprobe(drive_service, file_id, file_name)


# ============================================================
# STOCK CLASSIFICATION
# ============================================================

def classify_stock(video_name: str, mime_type: str = "") -> str:
    """Return 'Stock' or 'Non Stock' based on filename and MIME type."""
    name_lower = video_name.lower()
    if any(kw in name_lower for kw in STOCK_VIDEO_KEYWORDS):
        return "Stock"
    if mime_type == "video/quicktime" or name_lower.endswith(".mov"):
        return "Stock"
    return "Non Stock"


# ============================================================
# DRIVE HELPERS
# ============================================================

def _list_items(drive_service, parent_id: str, extra_q: str = "") -> list:
    """Page through Drive items in a folder and return all."""
    results, page_token = [], None
    base_q = f"'{parent_id}' in parents and trashed=false"
    q = f"{base_q} and {extra_q}" if extra_q else base_q

    while True:
        resp = retry_call(
            drive_service.files().list(
                q=q,
                fields="nextPageToken, files(id, name, mimeType, webViewLink, "
                       "size, videoMediaMetadata)",
                orderBy="name",
                pageToken=page_token,
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            ).execute
        )
        results.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return results


def list_subfolders(drive_service, parent_id: str) -> list:
    return _list_items(
        drive_service, parent_id,
        "mimeType='application/vnd.google-apps.folder'",
    )


def find_assets_folder(drive_service, topic_folder_id: str) -> Optional[str]:
    for sf in list_subfolders(drive_service, topic_folder_id):
        if sf["name"].strip().lower() == ASSETS_FOLDER_NAME:
            return sf["id"]
    return None


def list_videos_in_folder(
    drive_service,
    creds,
    folder_id: str,
    course_name: str,
    topic_name: str,
) -> list:
    """Return video dicts for all videos directly inside folder_id."""
    videos = []
    for f in _list_items(drive_service, folder_id):
        is_video = (
            f["mimeType"] in VIDEO_MIME_TYPES
            or f["name"].lower().endswith(VIDEO_EXTENSIONS)
        )
        if not is_video:
            continue

        duration = get_video_duration(f, drive_service, creds)
        videos.append({
            "video_id":    f["id"],
            "video_name":  f["name"],
            "course_name": course_name,
            "topic_name":  topic_name,
            "mime_type":   f["mimeType"],
            "duration":    duration,
            "file_size":   format_size(f.get("size")),
            "video_link":  f.get("webViewLink", ""),
            "stock":       classify_stock(f["name"], f["mimeType"]),
        })
    return videos


def find_videos_recursive(
    drive_service,
    creds,
    folder_id: str,
    depth: int = 0,
) -> list:
    """Recursively walk folder_id and return every video found."""
    videos = []
    for f in _list_items(drive_service, folder_id):
        indent = "  " * (depth + 1)
        if f["mimeType"] == "application/vnd.google-apps.folder":
            _log(f"{indent}↳ subfolder: {f['name']}")
            videos.extend(find_videos_recursive(drive_service, creds, f["id"], depth + 1))
            continue

        is_video = (
            f["mimeType"] in VIDEO_MIME_TYPES
            or f["name"].lower().endswith(VIDEO_EXTENSIONS)
        )
        if is_video:
            duration = get_video_duration(f, drive_service, creds)
            videos.append({
                "video_id":    f["id"],
                "video_name":  f["name"],
                "course_name": "",
                "topic_name":  "",
                "mime_type":   f["mimeType"],
                "duration":    duration,
                "file_size":   format_size(f.get("size")),
                "video_link":  f.get("webViewLink", ""),
                "stock":       classify_stock(f["name"], f["mimeType"]),
            })
    return videos


# ============================================================
# SHEETS HELPERS
# ============================================================

def _format_worksheet(spreadsheet, ws):
    """Apply the same formatting as the Colab version (frozen header, font 8, etc.)."""
    sheet_id   = ws.id
    total_rows = ws.row_count
    total_cols = ws.col_count

    reqs = [
        {
            "updateSheetProperties": {
                "properties": {
                    "sheetId": sheet_id,
                    "gridProperties": {"frozenRowCount": 1},
                },
                "fields": "gridProperties.frozenRowCount",
            }
        },
        {
            "repeatCell": {
                "range": {
                    "sheetId": sheet_id,
                    "startRowIndex": 0, "endRowIndex": total_rows,
                    "startColumnIndex": 0, "endColumnIndex": total_cols,
                },
                "cell": {
                    "userEnteredFormat": {
                        "textFormat": {"fontSize": 8},
                        "verticalAlignment": "MIDDLE",
                        "wrapStrategy": "CLIP",
                    }
                },
                "fields": "userEnteredFormat(textFormat.fontSize,verticalAlignment,wrapStrategy)",
            }
        },
        {
            "repeatCell": {
                "range": {
                    "sheetId": sheet_id,
                    "startRowIndex": 0, "endRowIndex": 1,
                    "startColumnIndex": 0, "endColumnIndex": total_cols,
                },
                "cell": {
                    "userEnteredFormat": {
                        "backgroundColor": {"red": 0.95, "green": 0.95, "blue": 0.95},
                        "textFormat": {"bold": True, "fontSize": 8},
                        "verticalAlignment": "MIDDLE",
                        "wrapStrategy": "CLIP",
                    }
                },
                "fields": "userEnteredFormat(backgroundColor,textFormat.bold,"
                           "textFormat.fontSize,verticalAlignment,wrapStrategy)",
            }
        },
        {
            "updateDimensionProperties": {
                "range": {
                    "sheetId": sheet_id,
                    "dimension": "ROWS",
                    "startIndex": 1, "endIndex": total_rows,
                },
                "properties": {"pixelSize": 40},
                "fields": "pixelSize",
            }
        },
    ]
    retry_call(spreadsheet.batch_update, {"requests": reqs})
    _sheets_sleep()


def _get_or_create_tab(spreadsheet, tab_name: str):
    """Return (worksheet, created_bool)."""
    safe_name = tab_name[:99]
    try:
        ws = retry_call(spreadsheet.worksheet, safe_name)
        return ws, False
    except WorksheetNotFound:
        ws = retry_call(
            spreadsheet.add_worksheet,
            title=safe_name,
            rows=1000,
            cols=len(HEADERS),
        )
        _sheets_sleep()
        retry_call(ws.append_row, HEADERS)
        _sheets_sleep()
        _log(f"  Created new tab: {safe_name}")
        return ws, True


def _get_existing_course_state(spreadsheet, course_name: str):
    """Return set of file IDs already in the sheet tab."""
    try:
        ws = retry_call(spreadsheet.worksheet, course_name[:99])
    except WorksheetNotFound:
        return set()
    rows = retry_call(ws.get_all_values)
    if len(rows) <= 1:
        return set()
    return {r[0] for r in rows[1:] if r and r[0]}


def _safe_write_sheet(ws, df: pd.DataFrame, retries: int = 5):
    for attempt in range(retries):
        try:
            set_with_dataframe(ws, df)
            _sheets_sleep()
            return
        except Exception as exc:
            wait = 2 ** attempt + random.uniform(0, 1.5)
            _log(f"  [sheet write retry {attempt + 1}] {exc} — waiting {wait:.1f}s")
            time.sleep(wait)
    _log("  WARNING: Could not save sheet after all retries.")


def log_videos_to_sheet(spreadsheet, tab_name: str, videos: list) -> int:
    """
    Upsert videos into sheet tab.

    - Appends new rows (by Drive file ID).
    - Patches any existing row where duration is '0:00' / blank.
    - Writes Stock/Non Stock column.
    """
    if not videos:
        return 0

    ws, created = _get_or_create_tab(spreadsheet, tab_name)
    existing_rows = ws.get_all_values()

    id_to_row_idx: dict[str, int] = {}
    id_to_duration: dict[str, str] = {}

    for row_idx, row in enumerate(existing_rows, 1):
        if row and row[0]:
            id_to_row_idx[row[0]] = row_idx
            id_to_duration[row[0]] = row[5] if len(row) > 5 else ""

    new_rows: list[list] = []
    cells_updated = 0

    for v in videos:
        vid_id = v["video_id"]

        if vid_id in id_to_row_idx:
            cur_dur = id_to_duration.get(vid_id, "").strip()
            if cur_dur in ("0:00", "0.00", "0:0", "0", "") and v["duration"] not in ("0:00", ""):
                ws.update_cell(id_to_row_idx[vid_id], 6, v["duration"])
                cells_updated += 1
                _sheets_sleep()
            continue

        new_rows.append([
            v["video_id"],
            v["video_name"],
            v["course_name"],
            v["topic_name"],
            v["mime_type"],
            v["duration"],
            v["file_size"],
            v["video_link"],
            v["stock"],
        ])

    if new_rows:
        retry_call(ws.append_rows, new_rows, value_input_option="RAW")
        _sheets_sleep()
        _format_worksheet(spreadsheet, ws)

    if cells_updated:
        _log(f"  Updated {cells_updated} duration(s) in '{tab_name[:99]}'.")
    if new_rows:
        _log(f"  Logged {len(new_rows)} new video(s) to '{tab_name[:99]}'.")

    return len(new_rows) + cells_updated


# ============================================================
# STOCK CLASSIFICATION PASS  (post-archive, same pipeline)
# ============================================================

def classify_stock_for_tab(ws) -> int:
    """
    Fill in the Stock/Non Stock column for any rows that are blank.
    Returns the number of rows updated.
    """
    df = pd.DataFrame(ws.get_all_records())
    if df.empty:
        _log(f"  [skip classify] '{ws.title}' is empty.")
        return 0

    col = "Stock/Non Stock"
    if col not in df.columns:
        df[col] = ""

    pending = df[col].astype(str).str.strip() == ""
    if not pending.any():
        _log(f"  [skip classify] '{ws.title}' — all rows already classified.")
        return 0

    count = int(pending.sum())
    _log(f"  Classifying {count} unclassified row(s) in '{ws.title}'…")

    for i in df[pending].index:
        name  = str(df.at[i, "Video Name"] if "Video Name" in df.columns else "")
        mime  = str(df.at[i, "MimeType"]   if "MimeType"   in df.columns else "")
        df.at[i, col] = classify_stock(name, mime)

    _safe_write_sheet(ws, df)
    _log(f"  Classification saved — '{ws.title}'.")
    return count

def run_stock_classification_pass(spreadsheet, tab_names: list[str]) -> dict:
    """Classify Stock/Non Stock only on the tabs that were just written to."""
    _log("\n📋 Stock / Non-Stock classification pass…")
    results: dict[str, int] = {}

    for tab_name in tab_names:
        try:
            ws = spreadsheet.worksheet(tab_name[:99])
            updated = classify_stock_for_tab(ws)
            results[tab_name] = updated
        except Exception as exc:
            _log(f"  [classify error] '{tab_name}': {exc}")

    return results


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_archive_pipeline(
    drive_service,
    sheets_client,
    creds,
    other_folder_id: str = "",
    mode: str = "course",
) -> dict:
    _log("=" * 60)
    _log("📦  VIDEO ARCHIVING PIPELINE")
    _log("=" * 60)

    pipeline_start = time.time()
    spreadsheet = sheets_client.open_by_url(SHEET_LINK)
    _log(f"Opened sheet: '{spreadsheet.title}'\n")

    total_course_videos = 0
    total_other_videos  = 0
    course_folders      = []
    touched_tabs: list[str] = []

    # ── PART A: Structured course → topic → assets ──────────────────────
    if mode == "course":
        _log("=== PART A: Course folders ===")
        course_folders = list_subfolders(drive_service, ASSETS_ARCHIVE_FOLDER_ID)
        _log(f"Found {len(course_folders)} course folder(s).\n")

        for course in course_folders:
            _log(f"Course: {course['name']}")

            existing_ids = _get_existing_course_state(spreadsheet, course["name"])

            topic_folders = list_subfolders(drive_service, course["id"])
            _log(f"  {len(topic_folders)} topic folder(s).")

            course_videos: list = []
            for topic in topic_folders:
                assets_id = find_assets_folder(drive_service, topic["id"])
                if not assets_id:
                    _log(f"    [skip] No 'assets' folder in: {topic['name']}")
                    continue

                videos = list_videos_in_folder(
                    drive_service, creds,
                    folder_id=assets_id,
                    course_name=course["name"],
                    topic_name=topic["name"],
                )
                # Filter out already-logged file IDs
                videos = [v for v in videos if v["video_id"] not in existing_ids]

                if videos:
                    _log(f"    Topic '{topic['name']}': {len(videos)} new video(s)")
                else:
                    _log(f"    Topic '{topic['name']}': no new videos")

                course_videos.extend(videos)

            if course_videos:
                logged = log_videos_to_sheet(spreadsheet, course["name"], course_videos)
                total_course_videos += logged
                touched_tabs.append(course["name"])
            else:
                _log("  No new videos to log.")
            _log("")
    else:
        _log("(mode=folder — Part A skipped.)\n")

    # ── PART B: Ad-hoc / Other Videos folder ────────────────────────────
    if mode == "folder":
        if other_folder_id.strip():
            _log(f"=== PART B: Other Videos folder ({other_folder_id}) ===")
            other_videos = find_videos_recursive(drive_service, creds, other_folder_id.strip())
            _log(f"  Found {len(other_videos)} video(s) total.")
            if other_videos:
                logged = log_videos_to_sheet(spreadsheet, OTHER_VIDEOS_TAB_NAME, other_videos)
                total_other_videos = logged
                touched_tabs.append(OTHER_VIDEOS_TAB_NAME)
        else:
            _log("(mode=folder but no folder ID provided — Part B skipped.)")
    else:
        _log("(mode=course — Part B skipped.)\n")

    # ── PART C: Stock classification — only touched tabs ─────────────────
    classify_results = run_stock_classification_pass(spreadsheet, touched_tabs)

    elapsed = time.time() - pipeline_start
    mins, sec = divmod(int(elapsed), 60)
    duration_str = f"{mins}m {sec}s" if mins else f"{sec}s"

    stats = {
        "course_folders_scanned": len(course_folders),
        "new_course_videos_logged": total_course_videos,
        "new_other_videos_logged": total_other_videos,
        "stock_rows_classified": classify_results,
        "sheet_link": SHEET_LINK,
        "duration": duration_str,
    }

    _log("\n" + "=" * 60)
    _log("✅  ARCHIVE PIPELINE COMPLETE")
    _log(f"    Course videos logged  : {total_course_videos}")
    _log(f"    Other videos logged   : {total_other_videos}")
    _log(f"    Duration              : {duration_str}")
    _log("=" * 60)

    return stats


# ============================================================
# STREAMLIT UI  — render function called from video_manager_app
# ============================================================

def render_archive_tab(
    drive_service=None,
    sheets_client=None,
    creds=None,
) -> None:
    """
    Render the Archiving Videos Streamlit tab.

    Intended to be called from inside render_gemini_drive_video_search_tab()
    when the user selects "📦  Archiving Videos" from the top selectbox.
    """
    global _st_log_placeholder

    # ── Auth: fall back to env-var init if callers didn't provide clients ──
    if drive_service is None or sheets_client is None:
        try:
            drive_service, sheets_client, creds = get_archive_clients()
        except Exception as exc:
            st.error(f"Could not initialise Drive / Sheets clients: {exc}")
            return

    st.subheader("📦 Archive Videos to Google Sheet")

    archive_mode = st.radio(
        "What would you like to do?",
        options=[
            "Process newly added course videos (automatic)",
            "Process a specific folder by ID",
        ],
        key="archive_mode_radio",
    )

    other_folder_id = ""

    if archive_mode == "Process a specific folder by ID":
        other_folder_id = st.text_input(
            "Google Drive folder ID",
            placeholder="Paste the folder ID (not the full URL) — e.g. 1e05r_3bjGzyec6XfpwwMBwwSdxCyM8Vp",
            key="archive_other_folder_id",
        )
        st.caption(
            "Videos found in this folder (and all its subfolders) will be logged "
            f"to the **{OTHER_VIDEOS_TAB_NAME}** tab of the sheet."
        )
    else:
        st.info(
            "This will automatically scan all course folders in the archive drive, "
            "skip topics already logged, and classify Stock / Non Stock — "
            "no input required."
        )

    if st.button("▶️  Run Archive Pipeline", type="primary", key="run_archive_btn"):
        # Validate folder mode before starting
        if archive_mode == "Process a specific folder by ID" and not other_folder_id.strip():
            st.warning("Please enter a folder ID before running.")
            return  # or st.stop()

        st.session_state["_archive_log"] = ""
        log_placeholder = st.empty()
        _st_log_placeholder = log_placeholder

        # Map UI choice to mode string
        pipeline_mode = "folder" if archive_mode == "Process a specific folder by ID" else "course"

        try:
            with st.spinner("Running archive pipeline…"):
                stats = run_archive_pipeline(
                    drive_service=drive_service,
                    sheets_client=sheets_client,
                    creds=creds,
                    other_folder_id=other_folder_id,
                    mode=pipeline_mode,
                )
            st.success("✅ Archive pipeline complete.")
            st.markdown(f"[Open Google Sheet]({SHEET_LINK})")

        except Exception as exc:
            import traceback
            st.error(f"Pipeline failed: {exc}")
            st.text(traceback.format_exc())
        finally:
            _st_log_placeholder = None


# # ============================================================
# # VS CODE  __main__  entry point
# # ============================================================

# if __name__ == "__main__":
#     import argparse

#     parser = argparse.ArgumentParser(description="Archive Drive videos to Google Sheet.")
#     parser.add_argument(
#         "--folder-id",
#         default="",
#         help="Optional Drive folder ID to scan for 'Other Videos'.",
#     )
#     args = parser.parse_args()

#     drive_svc, sheets_cli, raw_creds = get_archive_clients()
#     summary = run_archive_pipeline(
#         drive_service=drive_svc,
#         sheets_client=sheets_cli,
#         creds=raw_creds,
#         other_folder_id=args.folder_id,
#     )
#     print("\nSummary:", json.dumps(summary, indent=2))