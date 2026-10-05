# -*- coding: utf-8 -*-
"""Media service for fetching and previewing submission images, videos, audio from Google Drive via OAuth and local storage."""

from __future__ import annotations

import base64
import io
import logging
import mimetypes
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

from moodle_submission_feedback_app.backend.config import REPO_ROOT
from moodle_submission_feedback_app.backend.sessions import MentorSession
from services.drive_service import (
    build_service_account_drive_service,
    extract_drive_id_from_url,
)

logger = logging.getLogger("moodle_feedback_app.media")

# Supported media extensions (comprehensive list covering all media types)
IMAGE_EXTENSIONS = {
    "jpg", "jpeg", "png", "webp", "gif", "bmp", "svg", "heic", "heif",
    "tiff", "tif", "ico", "avif", "raw", "cr2", "nef", "arw", "dng",
}
VIDEO_EXTENSIONS = {
    "mp4", "mov", "webm", "avi", "mkv", "m4v", "ogv", "3gp", "3g2",
    "wmv", "flv", "mts", "m2ts", "ts", "mpg", "mpeg", "vob", "asf",
}
AUDIO_EXTENSIONS = {
    "mp3", "wav", "ogg", "m4a", "aac", "flac", "wma", "opus", "weba",
    "aiff", "aif", "mid", "midi", "amr", "alac", "pcm", "m4b",
}
ALL_MEDIA_EXTENSIONS = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS | AUDIO_EXTENSIONS

# Non-media extensions to strictly ignore
NON_MEDIA_EXTENSIONS = {
    "txt", "html", "htm", "json", "csv", "xml", "log", "pdf", "doc", "docx",
    "xls", "xlsx", "ppt", "pptx", "zip", "rar", "7z", "tar", "gz", "bz2",
}

# In-memory LRU-like cache for media bytes: file_id -> (bytes, mime_type)
_MEDIA_CACHE: Dict[str, Tuple[bytes, str]] = {}
_MAX_CACHE_ITEMS = 60

# In-memory TTL cache for folder listings: cache_key -> (timestamp, List[Dict[str, Any]])
_FOLDER_MEDIA_CACHE: Dict[str, Tuple[float, List[Dict[str, Any]]]] = {}
_FOLDER_CACHE_TTL_SECONDS = 900.0  # 15 minutes


def extract_all_drive_ids(text_input: str) -> List[Tuple[str, str]]:
    """Extracts (item_id, item_type) pairs from text containing Google Drive URLs."""
    if not text_input or not isinstance(text_input, str):
        return []

    results: List[Tuple[str, str]] = []
    seen = set()

    tokens = re.split(r"[\s,;\n]+", text_input.strip())
    for token in tokens:
        if not token:
            continue

        item_type = "unknown"
        if "/folders/" in token or "/drive/folders/" in token:
            item_type = "folder"
        elif "/file/d/" in token or "/d/" in token or "id=" in token:
            item_type = "file"

        item_id = extract_drive_id_from_url(token)
        if item_id and item_id not in seen:
            seen.add(item_id)
            results.append((item_id, item_type))

    return results


def classify_media(filename: str, mime_type: str = "") -> str:
    """
    Classifies a file as 'image', 'video', 'audio', or 'other'.
    Strictly flags text/docs (.txt, .html, .pdf, etc.) as 'other'.
    """
    fname_lower = (filename or "").lower().strip()
    ext = fname_lower.split(".")[-1] if "." in fname_lower else ""
    mime = (mime_type or "").lower().strip()

    # Reject non-media by extension or MIME
    if ext in NON_MEDIA_EXTENSIONS or fname_lower.endswith(".txt"):
        return "other"
    if mime.startswith("text/") or "html" in mime or "json" in mime or "pdf" in mime:
        return "other"

    if mime.startswith("image/") or ext in IMAGE_EXTENSIONS:
        return "image"
    if mime.startswith("video/") or ext in VIDEO_EXTENSIONS or "quicktime" in mime or "matroska" in mime or "mp4" in mime:
        return "video"
    if mime.startswith("audio/") or ext in AUDIO_EXTENSIONS or "audio" in mime:
        return "audio"

    return "other"


def guess_clean_mime(filename: str, raw_mime: str = "") -> str:
    """Ensures a clean MIME type for browser playback and display."""
    ext = filename.split(".")[-1].lower() if "." in filename else ""
    if ext == "mp4":
        return "video/mp4"
    if ext == "mov":
        return "video/quicktime"
    if ext == "webm":
        return "video/webm"
    if ext == "ogg" or ext == "ogv":
        return "video/ogg"
    if ext == "mp3":
        return "audio/mpeg"
    if ext in ("wav", "wave"):
        return "audio/wav"
    if ext in ("m4a", "aac"):
        return "audio/mp4"
    if ext in ("jpg", "jpeg"):
        return "image/jpeg"
    if ext == "png":
        return "image/png"
    if ext == "webp":
        return "image/webp"
    if ext == "gif":
        return "image/gif"
    if ext == "svg":
        return "image/svg+xml"

    if raw_mime and raw_mime != "application/octet-stream":
        return raw_mime

    guessed, _ = mimetypes.guess_type(filename)
    return guessed or "application/octet-stream"


def _get_drive_client_for_session(session: Optional[MentorSession]):
    """Returns an authenticated Google Drive v3 client from mentor session OAuth, or background service account."""
    if session:
        # 1. Primary: Session OAuth credentials from user login
        creds = getattr(session, "oauth_credentials", None)
        if creds:
            try:
                if hasattr(creds, "expired") and creds.expired and hasattr(creds, "refresh"):
                    from google.auth.transport.requests import Request
                    creds.refresh(Request())
                return build("drive", "v3", credentials=creds, cache_discovery=False)
            except Exception as e:
                logger.warning(f"Failed to build drive v3 from session.oauth_credentials: {e}")

        # 2. PyDrive session.drive auth credentials
        drive_obj = getattr(session, "drive", None)
        if drive_obj and hasattr(drive_obj, "auth") and hasattr(drive_obj.auth, "credentials"):
            try:
                return build("drive", "v3", credentials=drive_obj.auth.credentials, cache_discovery=False)
            except Exception as e:
                logger.warning(f"Failed to build drive v3 from session.drive.auth: {e}")

    # 3. Persistent OAuth token from environment (GOOGLE_OAUTH_REFRESH_TOKEN)
    try:
        from services.drive_service import get_authenticated_drive_service
        drive_client = get_authenticated_drive_service()
        if drive_client:
            return drive_client
    except Exception as e:
        logger.debug(f"get_authenticated_drive_service unavailable: {e}")

    # 4. Background service account fallback
    try:
        sa_client = build_service_account_drive_service()
        if sa_client:
            return sa_client
    except Exception as e:
        logger.debug(f"Service account client unavailable: {e}")

    return None


def fetch_submission_media_list(
    media_folder_url: str,
    student_name: str = "",
    session: Optional[MentorSession] = None,
    max_items: int = 35,
) -> List[Dict[str, Any]]:
    """
    Lists media files (images, videos, audio) belonging to a submission purely from Google Drive.
    Strictly filters out .txt, documents, and non-media files.
    Fully cloud-native with zero local filesystem downloads or storage.
    Uses in-memory TTL caching for sub-millisecond retrieval.
    """
    import time

    cache_key = f"{media_folder_url.strip() if media_folder_url else ''}__{max_items}"
    now = time.time()
    if cache_key in _FOLDER_MEDIA_CACHE:
        cached_time, cached_items = _FOLDER_MEDIA_CACHE[cache_key]
        if now - cached_time < _FOLDER_CACHE_TTL_SECONDS:
            return cached_items

    media_list: List[Dict[str, Any]] = []
    seen_ids = set()

    drive_client = _get_drive_client_for_session(session)
    drive_items = extract_all_drive_ids(media_folder_url) if media_folder_url else []

    if drive_client and drive_items:
        for item_id, item_type in drive_items:
            if len(media_list) >= max_items:
                break

            # If it's a folder, query for non-folder files inside it
            if item_type in ("folder", "unknown"):
                try:
                    # Query all non-folder files
                    query = f"'{item_id}' in parents and trashed = false and mimeType != 'application/vnd.google-apps.folder'"
                    resp = (
                        drive_client.files()
                        .list(
                            q=query,
                            fields="files(id, name, mimeType, webViewLink, thumbnailLink, size)",
                            supportsAllDrives=True,
                            includeItemsFromAllDrives=True,
                            pageSize=max_items,
                        )
                        .execute()
                    )
                    file_list = resp.get("files", [])
                    for f in file_list:
                        fid = f.get("id")
                        fname = f.get("name") or "media_file"
                        raw_mime = f.get("mimeType") or ""
                        mime = guess_clean_mime(fname, raw_mime)
                        cat = classify_media(fname, mime)

                        # STRICT FILTER: Only images, videos, audio. Exclude .txt, HTML, documents!
                        if cat not in ("image", "video", "audio"):
                            continue

                        if fid and fid not in seen_ids:
                            seen_ids.add(fid)
                            drive_url = f.get("webViewLink") or f"https://drive.google.com/file/d/{fid}/view?usp=drivesdk"
                            drive_thumb = f.get("thumbnailLink") or ""
                            if drive_thumb and "=s" in drive_thumb:
                                high_res_thumb = re.sub(r"=s\d+", "=s1000", drive_thumb)
                                low_res_thumb = re.sub(r"=s\d+", "=s160", drive_thumb)
                            elif drive_thumb:
                                high_res_thumb = drive_thumb + "=s1000"
                                low_res_thumb = drive_thumb + "=s160"
                            else:
                                high_res_thumb = ""
                                low_res_thumb = ""

                            # Images load directly from Google Drive image CDN with in-memory stream fallback
                            preview_url = high_res_thumb if (cat == "image" and high_res_thumb) else f"/api/media/file/{fid}"
                            thumb_url = low_res_thumb if (cat == "image" and low_res_thumb) else preview_url

                            media_list.append(
                                {
                                    "id": fid,
                                    "name": fname,
                                    "mime_type": mime,
                                    "media_type": cat,
                                    "preview_url": preview_url,
                                    "thumb_url": thumb_url,
                                    "api_stream_url": f"/api/media/file/{fid}",
                                    "drive_url": drive_url,  # Direct Drive file link
                                    "source": "google_drive",
                                }
                            )
                except Exception as err:
                    logger.warning(f"Error listing folder {item_id} from Drive: {err}")

            # If it's directly a file ID
            if item_type == "file":
                try:
                    meta = (
                        drive_client.files()
                        .get(fileId=item_id, fields="id, name, mimeType, webViewLink, thumbnailLink, size", supportsAllDrives=True)
                        .execute()
                    )
                    fid = meta.get("id")
                    fname = meta.get("name") or "media_file"
                    raw_mime = meta.get("mimeType") or ""
                    mime = guess_clean_mime(fname, raw_mime)
                    cat = classify_media(fname, mime)

                    # STRICT FILTER: Only images, videos, audio. Exclude .txt, HTML, documents!
                    if cat in ("image", "video", "audio") and fid and fid not in seen_ids:
                        seen_ids.add(fid)
                        drive_url = meta.get("webViewLink") or f"https://drive.google.com/file/d/{fid}/view?usp=drivesdk"
                        drive_thumb = meta.get("thumbnailLink") or ""
                        if drive_thumb and "=s" in drive_thumb:
                            high_res_thumb = re.sub(r"=s\d+", "=s1000", drive_thumb)
                            low_res_thumb = re.sub(r"=s\d+", "=s160", drive_thumb)
                        elif drive_thumb:
                            high_res_thumb = drive_thumb + "=s1000"
                            low_res_thumb = drive_thumb + "=s160"
                        else:
                            high_res_thumb = ""
                            low_res_thumb = ""

                        preview_url = high_res_thumb if (cat == "image" and high_res_thumb) else f"/api/media/file/{fid}"
                        thumb_url = low_res_thumb if (cat == "image" and low_res_thumb) else preview_url

                        media_list.append(
                            {
                                "id": fid,
                                "name": fname,
                                "mime_type": mime,
                                "media_type": cat,
                                "preview_url": preview_url,
                                "thumb_url": thumb_url,
                                "api_stream_url": f"/api/media/file/{fid}",
                                "drive_url": drive_url,  # Direct Drive file link
                                "source": "google_drive",
                            }
                        )
                except Exception as err:
                    logger.debug(f"Direct file get for {item_id} failed: {err}")

    if media_list and cache_key:
        _FOLDER_MEDIA_CACHE[cache_key] = (now, media_list)

    return media_list



def get_image_bytes(file_id: str, session: Optional[MentorSession] = None) -> Tuple[bytes, str]:
    """
    Retrieves media bytes and mime type for a Google Drive file in memory.
    Streams directly from Google Drive API without storing files on disk.
    """
    # Check cache first
    if file_id in _MEDIA_CACHE:
        return _MEDIA_CACHE[file_id]

    # Google Drive file streaming
    drive_client = _get_drive_client_for_session(session)
    if not drive_client:
        raise RuntimeError("No Google Drive client authenticated to retrieve file")

    try:
        meta = drive_client.files().get(fileId=file_id, fields="id, name, mimeType", supportsAllDrives=True).execute()
        fname = meta.get("name", "media_file")
        raw_mime = meta.get("mimeType") or ""
        mime_type = guess_clean_mime(fname, raw_mime)

        req = drive_client.files().get_media(fileId=file_id, supportsAllDrives=True)
        buf = io.BytesIO()
        downloader = MediaIoBaseDownload(buf, req)
        done = False
        while not done:
            _, done = downloader.next_chunk()

        data = buf.getvalue()
        if len(_MEDIA_CACHE) > _MAX_CACHE_ITEMS:
            _MEDIA_CACHE.pop(next(iter(_MEDIA_CACHE)))
        _MEDIA_CACHE[file_id] = (data, mime_type)
        return data, mime_type
    except Exception as e:
        logger.error(f"Failed to download Drive media {file_id}: {e}")
        raise
