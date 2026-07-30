"""Serve slide visuals through the authenticated Google session."""

from __future__ import annotations

import imghdr
import re
import threading
from typing import Optional
from urllib.parse import quote

from fastapi import HTTPException
from fastapi.responses import Response

from human_feedback_app.backend.auth import ensure_google_clients
from human_feedback_app.backend.sessions import UserSession
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

def _first_http_url(text: str) -> str:
    if not text:
        return ""
    match = re.search(r"https?://[^\s)>\"]+", str(text).strip())
    return match.group(0).rstrip(".,);\"'") if match else str(text).strip()


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
    Serve a trimmed Drive video clip with HTTP range support for <video>.

    :param session: Authenticated user session.
    :param raw_url: Drive video URL, optionally with a (start=..&end=..) suffix.
    :param range_header: Incoming Range request header, if any.
    :return: 200 full-body or 206 partial-content video Response.
    """
    url = _first_http_url(raw_url)
    if not url or not is_drive_url(url):
        raise HTTPException(status_code=400, detail="Not a Drive video URL")

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
