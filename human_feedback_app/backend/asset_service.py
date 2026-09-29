"""Serve slide visuals through the authenticated Google session."""

from __future__ import annotations

import imghdr
import os
import re
import tempfile
import threading
import time
from html import unescape
from typing import Optional
from urllib.parse import quote

from fastapi import HTTPException
from fastapi.responses import Response

from human_feedback_app.backend.auth import ensure_google_clients
from human_feedback_app.backend.sessions import UserSession
from human_feedback_app.backend.youtube_clip_downloader import (
    bounded_youtube_clip_url,
    download_youtube_clip_detailed,
    is_youtube_bot_check_error,
    is_youtube_url,
)
from graphics_definition_v2_slideshow import download_image_bytes, is_drive_url, is_drive_video_url
from services.helper_functions import (
    _MIME_BY_EXT,
    _download_drive_video_bytes,
    _trim_video_bytes,
    parse_drive_clip_timestamps,
    parse_drive_file_id_from_url,
)

_IMAGE_CACHE = {}

# Trimmed Drive video clips keyed by (file_id, start, end) -> (bytes, mime_type).
_CLIP_CACHE = {}
_CLIP_CACHE_LOCK = threading.Lock()
_CLIP_INFLIGHT = {}
_CLIP_INFLIGHT_LOCK = threading.Lock()
_YOUTUBE_DOWNLOAD_SEMAPHORE = threading.Semaphore(2)
_YOUTUBE_CLIP_MAX_SECONDS = 90
_YOUTUBE_COOKIES_DRIVE_FOLDER_ID = os.environ.get(
    "YOUTUBE_COOKIES_DRIVE_FOLDER_ID",
    "1Y4u_0dEuXkdcEl_QjnqwS8MikdhMrJz0",
)
_YOUTUBE_COOKIES_FILENAME = os.environ.get("YOUTUBE_COOKIES_FILENAME", "youtube_cookies.txt")
_YOUTUBE_COOKIES_LOCK = threading.Lock()
_YOUTUBE_COOKIES_MAX_AGE_SECONDS = int(os.environ.get("YOUTUBE_COOKIES_MAX_AGE_SECONDS", str(4 * 3600)))

def _first_http_url(text: str) -> str:
    if not text:
        return ""
    normalized = unescape(str(text).strip()).replace("%26", "&").replace("%3D", "=").replace("%3d", "=")
    match = re.search(r"https?://[^\s)>\"]+", normalized)
    return match.group(0).rstrip(".,);\"'") if match else normalized


def _slideshow_helpers():
    return {"download_image_bytes": download_image_bytes, "is_drive_url": is_drive_url}


def fetch_image_bytes(session: UserSession, raw_url: str, use_thumbnail: bool = False) -> bytes:
    ensure_google_clients(session)
    helpers = _slideshow_helpers()
    url = _first_http_url(raw_url)
    if not url:
        raise HTTPException(status_code=400, detail="Missing image URL")

    if is_drive_video_url(raw_url):
        use_thumbnail = True

    cache_key = (session.user_email or "", url, use_thumbnail)
    if cache_key in _IMAGE_CACHE:
        return _IMAGE_CACHE[cache_key]

    data = helpers["download_image_bytes"](url, session.drive, use_thumbnail=use_thumbnail)
    if not data:
        raise HTTPException(status_code=404, detail="Could not load image")
        
    # Cache up to 1000 images to prevent memory leaks
    if len(_IMAGE_CACHE) > 1000:
        _IMAGE_CACHE.clear()
    _IMAGE_CACHE[cache_key] = data
    
    return data


def image_response(session: UserSession, raw_url: str, thumb: str = "") -> Response:
    use_thumbnail = thumb == "1" or thumb.lower() == "true"
    data = fetch_image_bytes(session, raw_url, use_thumbnail=use_thumbnail)
    image_type = imghdr.what(None, h=data) or "jpeg"
    if image_type == "jpg":
        image_type = "jpeg"
    return Response(
        content=data, 
        media_type=f"image/{image_type}",
        headers={"Cache-Control": "public, max-age=86400, immutable"}
    )


def needs_image_proxy(url: str) -> bool:
    if not url:
        return False
    lowered = url.lower()
    return (
        "drive.google.com" in lowered
        or "docs.google.com" in lowered
        or "googleusercontent.com" in lowered
    )


def proxied_image_url(raw_url: str) -> Optional[str]:
    url = _first_http_url(raw_url)
    if not url or not needs_image_proxy(url):
        return None
    return "/api/assets/image?url=" + quote(url, safe="")


def _fetch_drive_clip_bytes(session: UserSession, raw_url: str):
    """
    Download + trim a Drive video clip once and cache it by (file_id, start, end).

    :param session: Authenticated user session (provides the Drive client).
    :param raw_url: Drive video URL, optionally with a (start=..&end=..) suffix.
    :return: Tuple (clip_bytes, mime_type).
    """
    ensure_google_clients(session)
    file_id = parse_drive_file_id_from_url(raw_url)
    if not file_id:
        raise HTTPException(status_code=400, detail="Invalid Drive video URL")

    start_seconds, end_seconds = parse_drive_clip_timestamps(raw_url)
    cache_key = (file_id, start_seconds, end_seconds)
    with _CLIP_CACHE_LOCK:
        cached = _CLIP_CACHE.get(cache_key)
    if cached is not None:
        return cached

    try:
        raw_bytes, ext = _download_drive_video_bytes(session.drive, file_id)
    except Exception as exc:  # pragma: no cover - network/drive failure
        raise HTTPException(status_code=404, detail="Could not load Drive video") from exc

    mime_type = _MIME_BY_EXT.get(ext, "video/mp4")
    clip_bytes = raw_bytes
    if start_seconds is not None and end_seconds is not None and end_seconds > start_seconds:
        try:
            clip_bytes = _trim_video_bytes(raw_bytes, start_seconds, end_seconds, suffix=ext)
        except Exception as trim_err:  # pragma: no cover - ffmpeg failure
            print(f"[human_feedback] Drive clip trim failed for {file_id}, serving full video: {trim_err}")
            clip_bytes = raw_bytes

    result = (clip_bytes, mime_type)
    with _CLIP_CACHE_LOCK:
        if len(_CLIP_CACHE) > 200:
            _CLIP_CACHE.clear()
        _CLIP_CACHE[cache_key] = result
    return result


def _drive_query_value(value: str) -> str:
    return str(value or "").replace("\\", "\\\\").replace("'", "\\'")


def _youtube_cookies_path() -> str:
    configured = os.environ.get("YOUTUBE_COOKIES_PATH", "").strip()
    if configured:
        return configured
    return os.path.join(tempfile.gettempdir(), _YOUTUBE_COOKIES_FILENAME)


def _cookie_file_looks_valid(path: str) -> bool:
    try:
        if not path or not os.path.exists(path) or os.path.getsize(path) <= 0:
            return False
        with open(path, "r", encoding="utf-8", errors="ignore") as fp:
            text = fp.read(8000)
        head = text[:200].lower()
        if "<html" in head or "<!doctype" in head:
            return False
        return ("youtube.com" in text) or ("# netscape" in text.lower()) or ("\t" in text)
    except Exception:
        return False


def _local_cookies_are_fresh(path: str) -> bool:
    if not _cookie_file_looks_valid(path):
        return False
    try:
        age = time.time() - os.path.getmtime(path)
    except Exception:
        return False
    return age < max(60, _YOUTUBE_COOKIES_MAX_AGE_SECONDS)


def _download_youtube_cookies_from_drive(session: UserSession, local_path: str) -> Optional[str]:
    folder_id = os.environ.get("YOUTUBE_COOKIES_DRIVE_FOLDER_ID", _YOUTUBE_COOKIES_DRIVE_FOLDER_ID).strip()
    file_id = os.environ.get("YOUTUBE_COOKIES_DRIVE_FILE_ID", "").strip()
    filename = os.environ.get("YOUTUBE_COOKIES_FILENAME", _YOUTUBE_COOKIES_FILENAME).strip() or "youtube_cookies.txt"
    if not folder_id and not file_id:
        return None
    try:
        ensure_google_clients(session)
        if file_id:
            cookie_file = session.drive.CreateFile({"id": file_id})
        else:
            query = (
                f"title='{_drive_query_value(filename)}' "
                f"and '{_drive_query_value(folder_id)}' in parents "
                "and trashed=false"
            )
            params = {
                "q": query,
                "maxResults": 10,
                "supportsAllDrives": True,
                "includeItemsFromAllDrives": True,
                "corpora": "allDrives",
            }
            try:
                matches = session.drive.ListFile(params).GetList()
            except Exception:
                matches = session.drive.ListFile({"q": query, "maxResults": 10}).GetList()
            if not matches:
                print(f"[human_feedback] YouTube cookies file not found in Drive folder {folder_id}")
                return None
            cookie_file = matches[0]

        os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
        tmp_path = local_path + ".tmp"
        cookie_file.GetContentFile(tmp_path)
        if not _cookie_file_looks_valid(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
            print("[human_feedback] YouTube cookies downloaded from Drive are empty or invalid")
            return None
        os.replace(tmp_path, local_path)
        print(f"[human_feedback] YouTube cookies loaded from Drive into {local_path}")
        return local_path
    except Exception as exc:
        print(f"[human_feedback] Could not load YouTube cookies from Drive: {exc}")
        return None


def _ensure_youtube_cookies_file(session: UserSession, force_refresh: bool = False) -> Optional[str]:
    """
    Ensure youtube_cookies.txt exists on local disk for yt-dlp.

    Local and Lightning use the same path:
    1. If a fresh local cookies file already exists, use it.
    2. Otherwise download youtube_cookies.txt from the configured Drive folder.
    """
    local_path = _youtube_cookies_path()
    with _YOUTUBE_COOKIES_LOCK:
        if not force_refresh and _local_cookies_are_fresh(local_path):
            return local_path
        refreshed = _download_youtube_cookies_from_drive(session, local_path)
        if refreshed:
            return refreshed
        if _cookie_file_looks_valid(local_path):
            print(f"[human_feedback] Using existing YouTube cookies at {local_path}")
            return local_path
        print("[human_feedback] No usable YouTube cookies file; yt-dlp may be blocked")
        return None


def _youtube_download_error_detail(error: str, cookies_path: Optional[str]) -> str:
    if is_youtube_bot_check_error(error):
        if cookies_path:
            return (
                "YouTube blocked the clip download even with cookies. "
                "Export a fresh youtube_cookies.txt and replace the file in the cookies Drive folder."
            )
        return (
            "YouTube blocked the clip download (sign-in / bot check). "
            "Add a valid youtube_cookies.txt to the cookies Drive folder."
        )
    return error or "Could not load YouTube video"


def _fetch_youtube_clip_bytes(session: UserSession, raw_url: str):
    """
    Download + cache a bounded YouTube clip as MP4 bytes for native <video>.

    Player playback must not use a YouTube iframe because iframe chrome/title/
    subtitles/branding cannot be reliably hidden from the parent page.
    """
    bounded_url = bounded_youtube_clip_url(raw_url, max_seconds=_YOUTUBE_CLIP_MAX_SECONDS)
    if not bounded_url:
        raise HTTPException(status_code=400, detail="Invalid YouTube video URL")
    cache_key = ("youtube", bounded_url)
    with _CLIP_CACHE_LOCK:
        cached = _CLIP_CACHE.get(cache_key)
    if cached is not None:
        return cached

    with _CLIP_INFLIGHT_LOCK:
        waiter = _CLIP_INFLIGHT.get(cache_key)
        is_leader = waiter is None
        if is_leader:
            waiter = threading.Event()
            _CLIP_INFLIGHT[cache_key] = waiter

    if not is_leader:
        waiter.wait(timeout=320)
        with _CLIP_CACHE_LOCK:
            cached = _CLIP_CACHE.get(cache_key)
        if cached is not None:
            return cached
        raise HTTPException(status_code=404, detail="Could not load YouTube video")

    tmp_dir = tempfile.mkdtemp(prefix="hf_youtube_clip_")
    output_path = os.path.join(tmp_dir, "clip.mp4")
    data = None
    mime_type = "video/mp4"
    last_error = ""
    cookies_path = None
    try:
        cookies_path = _ensure_youtube_cookies_file(session)
        print(
            "[human_feedback] YouTube clip download "
            + ("using cookies" if cookies_path else "WITHOUT cookies")
            + f" for {bounded_url}"
        )
        with _YOUTUBE_DOWNLOAD_SEMAPHORE:
            path, last_error = download_youtube_clip_detailed(
                bounded_url,
                output_path,
                max_seconds=_YOUTUBE_CLIP_MAX_SECONDS,
                cookies_path=cookies_path,
            )
            if not path and is_youtube_bot_check_error(last_error):
                print("[human_feedback] YouTube bot-check; refreshing cookies and retrying once")
                cookies_path = _ensure_youtube_cookies_file(session, force_refresh=True)
                path, last_error = download_youtube_clip_detailed(
                    bounded_url,
                    output_path,
                    max_seconds=_YOUTUBE_CLIP_MAX_SECONDS,
                    cookies_path=cookies_path,
                )
        if not path or not os.path.exists(path):
            raise HTTPException(
                status_code=404,
                detail=_youtube_download_error_detail(last_error, cookies_path),
            )
        with open(path, "rb") as fp:
            data = fp.read()
        result = (data, mime_type)
        with _CLIP_CACHE_LOCK:
            if len(_CLIP_CACHE) > 200:
                _CLIP_CACHE.clear()
            _CLIP_CACHE[cache_key] = result
        return result
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - network/yt-dlp failure
        raise HTTPException(
            status_code=404,
            detail=_youtube_download_error_detail(str(exc), cookies_path),
        ) from exc
    finally:
        try:
            import shutil

            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass
        waiter.set()
        with _CLIP_INFLIGHT_LOCK:
            _CLIP_INFLIGHT.pop(cache_key, None)


def _parse_range_header(range_header: str, total: int):
    """
    Parse a single HTTP Range header into inclusive (start, end) byte offsets.

    :param range_header: Raw Range header value (e.g. "bytes=0-1023").
    :param total: Total content length in bytes.
    :return: Tuple (start, end) inclusive, or None when no valid range.
    """
    if not range_header:
        return None
    match = re.match(r"\s*bytes=(\d*)-(\d*)\s*$", range_header)
    if not match:
        return None
    g1, g2 = match.group(1), match.group(2)
    if g1 == "" and g2 == "":
        return None
    if g1 == "":
        length = int(g2)
        if length <= 0:
            return None
        start = max(0, total - length)
        end = total - 1
    else:
        start = int(g1)
        end = int(g2) if g2 != "" else total - 1
    end = min(end, total - 1)
    if start > end:
        return None
    return start, end


def video_clip_response(session: UserSession, raw_url: str, range_header: str = "") -> Response:
    """
    Serve a trimmed Drive/YouTube video clip with HTTP range support for <video>.

    :param session: Authenticated user session.
    :param raw_url: Drive or YouTube video URL with optional clip timestamps.
    :param range_header: Incoming Range request header, if any.
    :return: 200 full-body or 206 partial-content video Response.
    """
    url = _first_http_url(raw_url)

    if not url or not (is_drive_url(url) or is_youtube_url(url)):
        raise HTTPException(status_code=400, detail="Not a supported video URL")

    if is_youtube_url(url):
        data, mime_type = _fetch_youtube_clip_bytes(session, raw_url)
    else:
        data, mime_type = _fetch_drive_clip_bytes(session, raw_url)
    total = len(data)
    common_headers = {
        "Accept-Ranges": "bytes",
        "Cache-Control": "public, max-age=86400",
    }

    byte_range = _parse_range_header(range_header, total)
    if byte_range is None:
        headers = {**common_headers, "Content-Length": str(total)}
        return Response(content=data, media_type=mime_type, headers=headers)

    start, end = byte_range
    chunk = data[start : end + 1]
    headers = {
        **common_headers,
        "Content-Range": f"bytes {start}-{end}/{total}",
        "Content-Length": str(len(chunk)),
    }
    return Response(content=chunk, media_type=mime_type, headers=headers, status_code=206)


def proxied_video_url(raw_url: str) -> Optional[str]:
    """Return the app clip-proxy URL for a Drive video, else None."""
    url = _first_http_url(raw_url)
    if not url or not is_drive_video_url(raw_url):
        return None
    return "/api/assets/video?url=" + quote(raw_url, safe="")
