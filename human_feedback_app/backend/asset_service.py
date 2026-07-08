"""Serve slide visuals through the authenticated Google session."""

from __future__ import annotations

import imghdr
import re
from typing import Optional
from urllib.parse import quote

from fastapi import HTTPException
from fastapi.responses import Response

from human_feedback_app.backend.auth import ensure_google_clients
from human_feedback_app.backend.sessions import UserSession
from graphics_definition_v2_slideshow import download_image_bytes, is_drive_url

_IMAGE_CACHE = {}

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
