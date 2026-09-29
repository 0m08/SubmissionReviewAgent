"""
Graphics Definition V2 step: index External Reference assets into Supabase.

Separate from extraction. Embeds with Gemini Embedding 2 into external_ref_assets only.
Never touches curriculum / courses tables.
"""

import json
import os
import re
import subprocess
import tempfile
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from urllib.parse import parse_qs, urlparse

from dotenv import load_dotenv
from langsmith import traceable

from agents.graphics_definition_v2.external_references.external_reference_extraction import (
    COURSE_INFO_WORKSHEET,
    EXTERNAL_REFERENCES_COLUMN,
    EXTERNAL_REF_ASSETS_FOLDER_ID,
    EXTRACTION_LOG_COLUMN,
    SUPPORTED_DOCUMENT_MIMES,
    _get_drive_instance,
    _read_course_info,
    _resolve_sheet_id,
    parse_drive_file_id,
    parse_external_reference_links,
    source_link_hash,
)
from agents.graphics_definition_v2.external_references.vectorstore import (
    count_assets_for_source,
    delete_assets_for_sheet,
    delete_assets_for_source,
    embed_image_bytes,
    embed_video_bytes,
    list_source_links_for_sheet,
    make_asset_id,
    upsert_asset_row,
)
from services.helper_functions import (
    _download_drive_video_bytes,
    _trim_video_bytes,
    parse_drive_file_id_from_url,
)
from services.sheets_service import format_worksheet
from services.smart_progress_bar import SmartProgressBar

load_dotenv()

INDEX_LOG_COLUMN = "external_ref_index_log"
INDEX_MAX_WORKERS = 50

# Clip lengths locked for External Refs videos.
DRIVE_VIDEO_CHUNK_SEC = 60
YOUTUBE_VIDEO_CHUNK_SEC = 30


def _classify_external_link(url):
    """
    Classify an External References URL.

    :param url: External reference URL to classify
    :return: 'youtube' | 'drive_video' | 'document' | 'unsupported'
    """
    lower = (url or "").lower().strip()
    if not lower:
        return "unsupported"
    if "youtube.com" in lower or "youtu.be" in lower:
        return "youtube"
    if "/folders/" in lower:
        return "unsupported"
    file_id = parse_drive_file_id(url)
    if not file_id:
        return "unsupported"
    # Mime decided after FetchMetadata for Drive links.
    if "docs.google.com/document" in lower or "docs.google.com/presentation" in lower:
        return "document"
    return "drive_unknown"


def _is_drive_video_mime(mime_type):
    """
    Return whether a Drive MIME type represents a video file.

    :param mime_type: MIME type string from Drive metadata
    :return: True if the MIME type is a video
    """
    mime = (mime_type or "").lower()
    return mime.startswith("video/") or mime in {
        "application/vnd.google-apps.video",
    }


def _read_extraction_log_assets(sheet):
    """
    Parse Course info external_ref_extraction_log into source_link to asset_urls mapping.

    :param sheet: gspread sheet object
    :return: Dict mapping source_link to list of asset URLs
    """
    try:
        worksheet = None
        for candidate in sheet.worksheets():
            if candidate.title.strip().lower() == COURSE_INFO_WORKSHEET.lower():
                worksheet = candidate
                break
        if worksheet is None:
            return {}
        headers = [str(h).strip() for h in worksheet.row_values(1)]
        col = None
        for idx, header in enumerate(headers):
            if header.lower() == EXTRACTION_LOG_COLUMN.lower():
                col = idx + 1
                break
        if col is None:
            return {}
        raw = worksheet.cell(2, col).value or ""
        if not str(raw).strip():
            return {}
        data = json.loads(raw)
        if not isinstance(data, dict):
            return {}
        out = {}
        for entry in data.get("per_source") or []:
            if not isinstance(entry, dict):
                continue
            link = (entry.get("source_link") or "").strip()
            urls = [u for u in (entry.get("asset_urls") or []) if u]
            if link and urls:
                out[link] = urls
        return out
    except json.JSONDecodeError:
        # Legacy plain-text empty-run logs are not JSON; treat as no assets.
        return {}
    except Exception as exc:
        print(f"⚠️ Could not read extraction log assets: {exc}")
        return {}


def _download_drive_file_bytes(drive, file_id):
    """
    Download a Drive file and return its bytes, MIME type, and title.

    :param drive: PyDrive drive instance
    :param file_id: Google Drive file id
    :return: Tuple of (file bytes, mime_type, title)
    """
    gfile = drive.CreateFile({"id": file_id})
    gfile.FetchMetadata(fields="title,mimeType")
    mime_type = str(gfile.get("mimeType") or "application/octet-stream")
    title = str(gfile.get("title") or "file.bin")
    ext = os.path.splitext(title)[1] or ".bin"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    path = tmp.name
    tmp.close()
    try:
        gfile.GetContentFile(path)
        with open(path, "rb") as fh:
            return fh.read(), mime_type, title
    finally:
        if os.path.exists(path):
            os.remove(path)


def _persist_index_log(sheet, log_text):
    """
    Write index summary to Course info, creating the column and formatting if needed.

    :param sheet: gspread sheet object
    :param log_text: JSON summary text to persist
    :return: None
    """
    try:
        worksheet = None
        for candidate in sheet.worksheets():
            if candidate.title.strip().lower() == COURSE_INFO_WORKSHEET.lower():
                worksheet = candidate
                break
        if worksheet is None:
            print(f"⚠️ '{COURSE_INFO_WORKSHEET}' not found; skipping index log")
            return
        headers = worksheet.row_values(1)
        col = None
        created = False
        for idx, header in enumerate(headers):
            if str(header).strip().lower() == INDEX_LOG_COLUMN.lower():
                col = idx + 1
                break
        if col is None:
            headers.append(INDEX_LOG_COLUMN)
            col = len(headers)
            if col > worksheet.col_count:
                worksheet.add_cols(col - worksheet.col_count)
            worksheet.update_cell(1, col, INDEX_LOG_COLUMN)
            created = True
        worksheet.update_cell(2, col, log_text)
        if created:
            format_worksheet(worksheet)
        print(f"📝 Persisted {INDEX_LOG_COLUMN}")
    except Exception as exc:
        print(f"⚠️ Could not persist index log: {exc}")


def delete_external_reference_index_log(sheet):
    """
    Delete indexed external reference assets for this sheet and clear the index log column.

    :param sheet: gspread sheet object
    :return: None
    """
    sheet_id = (_resolve_sheet_id(sheet) or "").strip()
    if not sheet_id:
        print(
            "❌ Index delete aborted: could not resolve spreadsheet id. "
            "Refusing to delete any Supabase rows without sheet_id."
        )
    else:
        try:
            deleted = delete_assets_for_sheet(sheet_id)
            print(
                f"🗑️ Deleted {deleted} row(s) from external_ref_assets "
                f"for sheet_id={sheet_id} only"
            )
        except Exception as exc:
            print(f"❌ Failed to delete external_ref_assets for sheet_id={sheet_id}: {exc}")
            traceback.print_exc()

    try:
        worksheet = None
        for candidate in sheet.worksheets():
            if candidate.title.strip().lower() == COURSE_INFO_WORKSHEET.lower():
                worksheet = candidate
                break
        if worksheet is None:
            print(f"ℹ️ '{COURSE_INFO_WORKSHEET}' tab not found")
            return
        headers = [str(h).strip() for h in worksheet.row_values(1)]
        for idx, header in enumerate(headers):
            if header.lower() == INDEX_LOG_COLUMN.lower():
                worksheet.update_cell(2, idx + 1, "")
                print(f"🗑️ Cleared '{INDEX_LOG_COLUMN}' in Course info")
                return
        print(f"ℹ️ '{INDEX_LOG_COLUMN}' column does not exist")
    except Exception as exc:
        print(f"⚠️ Could not clear index log: {exc}")


def _index_one_image(drive, sheet_id, course_name, source_link, asset_url):
    """
    Download, embed, and upsert one extracted image into external_ref_assets.

    :param drive: PyDrive drive instance
    :param sheet_id: Spreadsheet id for scoping Supabase rows
    :param course_name: Course display name for embedding prefix
    :param source_link: Original external reference URL
    :param asset_url: Drive URL of the extracted image asset
    :return: True on success, False otherwise
    """
    file_id = parse_drive_file_id_from_url(asset_url) or parse_drive_file_id(asset_url)
    if not file_id:
        print(f"⚠️ Skip image (no file id): {asset_url}")
        return False
    data, mime_type, title = _download_drive_file_bytes(drive, file_id)
    if not mime_type.startswith("image/") and "image" not in mime_type.lower():
        mime_type = "image/jpeg"
    # Short unique display title: 8-char source hash + Drive filename (e.g. a3f9c2e1_img_001.jpg).
    base_title = (title or file_id or "image").strip()
    display_title = f"{source_link_hash(source_link)[:8]}_{base_title}"
    text_prefix = f"Course: {course_name} | External reference image | Source: {source_link}"
    embedding = embed_image_bytes(data, mime_type=mime_type, text_prefix=text_prefix)
    asset_id = make_asset_id(sheet_id, source_link, asset_url)
    upsert_asset_row(
        {
            "asset_id": asset_id,
            "sheet_id": sheet_id,
            "course_name": course_name,
            "source_link": source_link,
            "asset_url": asset_url,
            "asset_type": "image",
            "source_kind": "pdf_extract",
            "mime_type": mime_type,
            "title": display_title,
            "drive_folder_id": EXTERNAL_REF_ASSETS_FOLDER_ID,
            "video_id": None,
            "segment_index": None,
            "start_time": None,
            "end_time": None,
            "embedding": embedding,
        }
    )
    print(f"✅ Indexed image: {asset_url}")
    return True


def _index_extracted_images(drive, sheet_id, course_name, source_link, asset_urls, max_workers=INDEX_MAX_WORKERS):
    """
    Download extracted Drive images, embed them, and upsert into external_ref_assets in parallel.

    :param drive: PyDrive drive instance
    :param sheet_id: Spreadsheet id for scoping Supabase rows
    :param course_name: Course display name for embedding prefix
    :param source_link: Original external reference URL
    :param asset_urls: List of Drive URLs for extracted images
    :param max_workers: Maximum parallel image indexing workers
    :return: Number of images successfully indexed
    """
    urls = [u for u in (asset_urls or []) if u]
    if not urls:
        return 0

    workers = max(1, min(int(max_workers or INDEX_MAX_WORKERS), len(urls)))
    print(f"📥 Indexing {len(urls)} image(s) in parallel (max_workers={workers})...")
    indexed = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _index_one_image,
                drive,
                sheet_id,
                course_name,
                source_link,
                asset_url,
            ): asset_url
            for asset_url in urls
        }
        for future in as_completed(futures):
            asset_url = futures[future]
            try:
                if future.result():
                    indexed += 1
            except Exception as exc:
                print(f"❌ Failed to index image {asset_url}: {exc}")
                traceback.print_exc()
    return indexed


def _probe_duration_seconds(video_bytes, suffix=".mp4"):
    """
    Return media duration in seconds via ffprobe.

    :param video_bytes: Raw video file bytes
    :param suffix: Temporary file suffix for ffprobe
    :return: Duration in seconds
    """
    if not video_bytes:
        raise ValueError("video_bytes is empty")
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix or ".mp4") as tmp:
        tmp.write(video_bytes)
        path = tmp.name
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                path,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return float((result.stdout or "").strip())
    finally:
        if os.path.exists(path):
            os.remove(path)


def _iter_time_segments(duration_sec, chunk_sec):
    """
    Split [0, duration) into non-overlapping time chunks.

    :param duration_sec: Total video duration in seconds
    :param chunk_sec: Target chunk length in seconds
    :return: List of (segment_index, start_sec, end_sec) tuples
    """
    duration = float(duration_sec or 0.0)
    chunk = float(chunk_sec or 0.0)
    if duration <= 0 or chunk <= 0:
        return []
    segments = []
    start = 0.0
    idx = 0
    while start < duration:
        end = min(start + chunk, duration)
        # Drop a tiny leftover tail (< 0.5s) unless it is the only segment.
        if end - start < 0.5 and idx > 0:
            break
        segments.append((idx, start, end))
        start = end
        idx += 1
    return segments


def parse_youtube_video_id(url):
    """
    Extract a YouTube video id from common URL shapes.

    :param url: YouTube URL
    :return: Video id string, or None if not parseable
    """
    text = (url or "").strip()
    if not text:
        return None
    parsed = urlparse(text)
    host = (parsed.netloc or "").lower()
    if "youtu.be" in host:
        vid = (parsed.path or "").strip("/").split("/")[0]
        return vid or None
    if "youtube.com" in host:
        qs = parse_qs(parsed.query or "")
        if qs.get("v") and qs["v"][0]:
            return qs["v"][0]
        parts = [p for p in (parsed.path or "").split("/") if p]
        if len(parts) >= 2 and parts[0] in {"embed", "shorts", "live", "v"}:
            return parts[1]
    match = re.search(r"(?:v=|/embed/|/shorts/|/live/|youtu\.be/)([A-Za-z0-9_-]{6,})", text)
    return match.group(1) if match else None


def _download_youtube_video_bytes(url):
    """
    Download a YouTube video with yt-dlp.

    :param url: YouTube video URL
    :return: Tuple of (video_bytes, mime_type, title)
    """
    import yt_dlp

    tmp_dir = tempfile.mkdtemp(prefix="extref_yt_")
    try:
        outtmpl = os.path.join(tmp_dir, "video.%(ext)s")
        ydl_opts = {
            "outtmpl": outtmpl,
            "quiet": True,
            "no_warnings": True,
            "format": "best[ext=mp4][vcodec^=avc]/best[ext=mp4]/best",
            "merge_output_format": "mp4",
        }
        title = "youtube_video"
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            if isinstance(info, dict):
                title = str(info.get("title") or title)
        video_path = None
        for name in os.listdir(tmp_dir):
            path = os.path.join(tmp_dir, name)
            if os.path.isfile(path) and os.path.getsize(path) > 0:
                video_path = path
                break
        if not video_path:
            raise RuntimeError(f"yt-dlp did not produce a video file for {url}")
        with open(video_path, "rb") as fh:
            data = fh.read()
        ext = os.path.splitext(video_path)[1].lower() or ".mp4"
        mime = "video/webm" if ext == ".webm" else "video/mp4"
        return data, mime, title
    finally:
        for name in os.listdir(tmp_dir):
            try:
                os.remove(os.path.join(tmp_dir, name))
            except OSError:
                pass
        try:
            os.rmdir(tmp_dir)
        except OSError:
            pass


def _index_one_video_segment(sheet_id, course_name, source_link, asset_url, source_kind, video_id, title, full_video_bytes, suffix, mime_type, segment_index, start_time, end_time):
    """
    Trim one video segment, embed it, and upsert without re-uploading the clip to Drive.

    :param sheet_id: Spreadsheet id for scoping Supabase rows
    :param course_name: Course display name for embedding prefix
    :param source_link: Original external reference URL
    :param asset_url: Canonical asset URL stored on the row
    :param source_kind: Source kind label (drive_video or youtube_video)
    :param video_id: Drive file id or YouTube video id
    :param title: Video title for the row
    :param full_video_bytes: Full downloaded video bytes to trim from
    :param suffix: File suffix used when trimming
    :param mime_type: MIME type of the video
    :param segment_index: Zero-based segment index
    :param start_time: Segment start time in seconds
    :param end_time: Segment end time in seconds
    :return: True on success
    """
    clip_bytes = _trim_video_bytes(
        full_video_bytes,
        start_time,
        end_time,
        suffix=suffix or ".mp4",
    )
    text_prefix = (
        f"Course: {course_name} | External reference video segment "
        f"| Source: {source_link} | {start_time:.1f}s-{end_time:.1f}s"
    )
    embedding = embed_video_bytes(
        clip_bytes,
        mime_type=mime_type or "video/mp4",
        text_prefix=text_prefix,
    )
    asset_id = make_asset_id(
        sheet_id,
        source_link,
        asset_url,
        start_time=float(start_time),
        end_time=float(end_time),
    )
    upsert_asset_row(
        {
            "asset_id": asset_id,
            "sheet_id": sheet_id,
            "course_name": course_name,
            "source_link": source_link,
            "asset_url": asset_url,
            "asset_type": "video_segment",
            "source_kind": source_kind,
            "mime_type": mime_type or "video/mp4",
            "title": title or video_id or "video",
            "drive_folder_id": EXTERNAL_REF_ASSETS_FOLDER_ID if source_kind == "drive_video" else None,
            "video_id": video_id,
            "segment_index": int(segment_index),
            "start_time": float(start_time),
            "end_time": float(end_time),
            "embedding": embedding,
        }
    )
    print(
        f"✅ Indexed {source_kind} segment {segment_index}: "
        f"{start_time:.1f}s-{end_time:.1f}s ({len(clip_bytes)} bytes)"
    )
    return True


def _index_video_segments(sheet_id, course_name, source_link, asset_url, source_kind, video_id, title, video_bytes, suffix, mime_type, chunk_sec, max_workers=INDEX_MAX_WORKERS):
    """
    Chunk a full video into segments, embed each in parallel, and upsert rows.

    :param sheet_id: Spreadsheet id for scoping Supabase rows
    :param course_name: Course display name for embedding prefix
    :param source_link: Original external reference URL
    :param asset_url: Canonical asset URL stored on each row
    :param source_kind: Source kind label (drive_video or youtube_video)
    :param video_id: Drive file id or YouTube video id
    :param title: Video title for the rows
    :param video_bytes: Full downloaded video bytes
    :param suffix: File suffix used when trimming segments
    :param mime_type: MIME type of the video
    :param chunk_sec: Segment length in seconds
    :param max_workers: Maximum parallel segment indexing workers
    :return: Number of segments successfully indexed
    """
    duration = _probe_duration_seconds(video_bytes, suffix=suffix or ".mp4")
    segments = _iter_time_segments(duration, chunk_sec)
    if not segments:
        print(f"⚠️ No video segments for {source_link} (duration={duration:.2f}s)")
        return 0

    workers = max(1, min(int(max_workers or INDEX_MAX_WORKERS), len(segments)))
    print(
        f"🎬 Indexing {len(segments)} {source_kind} segment(s) "
        f"(duration={duration:.1f}s, chunk={chunk_sec}s, max_workers={workers})..."
    )
    indexed = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _index_one_video_segment,
                sheet_id,
                course_name,
                source_link,
                asset_url,
                source_kind,
                video_id,
                title,
                video_bytes,
                suffix,
                mime_type,
                seg_idx,
                start,
                end,
            ): (seg_idx, start, end)
            for seg_idx, start, end in segments
        }
        for future in as_completed(futures):
            seg_idx, start, end = futures[future]
            try:
                if future.result():
                    indexed += 1
            except Exception as exc:
                print(
                    f"❌ Failed to index {source_kind} segment {seg_idx} "
                    f"({start:.1f}-{end:.1f}): {exc}"
                )
                traceback.print_exc()
    return indexed


def _index_drive_video(drive, sheet_id, course_name, source_link, file_id, title, mime_type, max_workers=INDEX_MAX_WORKERS):
    """
    Download a Drive video once, chunk at 60s, and embed segments without clip re-upload.

    :param drive: PyDrive drive instance
    :param sheet_id: Spreadsheet id for scoping Supabase rows
    :param course_name: Course display name for embedding prefix
    :param source_link: Original external reference URL
    :param file_id: Google Drive file id
    :param title: Video title from Drive metadata
    :param mime_type: MIME type from Drive metadata
    :param max_workers: Maximum parallel segment indexing workers
    :return: Number of segments successfully indexed
    """
    video_bytes, ext = _download_drive_video_bytes(drive, file_id)
    suffix = ext if str(ext).startswith(".") else f".{ext or 'mp4'}"
    mime = mime_type if (mime_type or "").startswith("video/") else "video/mp4"
    return _index_video_segments(
        sheet_id=sheet_id,
        course_name=course_name,
        source_link=source_link,
        asset_url=source_link,
        source_kind="drive_video",
        video_id=file_id,
        title=title or file_id,
        video_bytes=video_bytes,
        suffix=suffix,
        mime_type=mime,
        chunk_sec=DRIVE_VIDEO_CHUNK_SEC,
        max_workers=max_workers,
    )


def _index_youtube_video(sheet_id, course_name, source_link, max_workers=INDEX_MAX_WORKERS):
    """
    Download a YouTube video, chunk at 30s, and embed segments without clip re-upload.

    :param sheet_id: Spreadsheet id for scoping Supabase rows
    :param course_name: Course display name for embedding prefix
    :param source_link: YouTube URL
    :param max_workers: Maximum parallel segment indexing workers
    :return: Number of segments successfully indexed
    """
    video_id = parse_youtube_video_id(source_link)
    if not video_id:
        raise ValueError(f"Could not parse YouTube video id from: {source_link}")
    print(f"📥 Downloading YouTube video {video_id}...")
    video_bytes, mime_type, title = _download_youtube_video_bytes(source_link)
    asset_url = f"https://www.youtube.com/watch?v={video_id}"
    return _index_video_segments(
        sheet_id=sheet_id,
        course_name=course_name,
        source_link=source_link,
        asset_url=asset_url,
        source_kind="youtube_video",
        video_id=video_id,
        title=title or video_id,
        video_bytes=video_bytes,
        suffix=".mp4",
        mime_type=mime_type,
        chunk_sec=YOUTUBE_VIDEO_CHUNK_SEC,
        max_workers=max_workers,
    )


def _index_one_source(drive, sheet_id, course_name, source_link, extraction_assets, max_workers=INDEX_MAX_WORKERS):
    """
    Index one External References link into Supabase (images plus Drive/YouTube video segments).

    :param drive: PyDrive drive instance
    :param sheet_id: Spreadsheet id for scoping Supabase rows
    :param course_name: Course display name for embedding prefix
    :param source_link: External reference URL to index
    :param extraction_assets: Mapping of source_link to extracted image URLs
    :param max_workers: Maximum parallel workers for image/segment indexing
    :return: Per-source result dict with status, assets_indexed, error, and kind
    """
    entry = {
        "source_link": source_link,
        "status": "skipped",
        "assets_indexed": 0,
        "error": "",
        "kind": "",
    }
    kind = _classify_external_link(source_link)
    entry["kind"] = kind

    if kind == "unsupported":
        entry["status"] = "skipped_unsupported"
        entry["error"] = "Unsupported link type for indexing"
        return entry

    if kind == "youtube":
        entry["kind"] = "youtube_video"
        try:
            deleted = delete_assets_for_source(sheet_id, source_link)
            if deleted:
                print(f"🗑️ Removed {deleted} old row(s) for re-index: {source_link}")
            n = _index_youtube_video(
                sheet_id=sheet_id,
                course_name=course_name,
                source_link=source_link,
                max_workers=max_workers,
            )
            entry["assets_indexed"] = n
            entry["status"] = "ok" if n > 0 else "error"
            if n == 0:
                entry["error"] = "No YouTube segments indexed"
            return entry
        except Exception as exc:
            entry["status"] = "error"
            entry["error"] = str(exc)
            print(f"❌ YouTube index failed for {source_link}: {exc}")
            traceback.print_exc()
            return entry

    file_id = parse_drive_file_id(source_link)
    try:
        gfile = drive.CreateFile({"id": file_id})
        gfile.FetchMetadata(fields="title,mimeType")
        mime_type = str(gfile.get("mimeType") or "").strip()
        title = str(gfile.get("title") or "").strip()

        if _is_drive_video_mime(mime_type):
            entry["kind"] = "drive_video"
            deleted = delete_assets_for_source(sheet_id, source_link)
            if deleted:
                print(f"🗑️ Removed {deleted} old row(s) for re-index: {source_link}")
            n = _index_drive_video(
                drive=drive,
                sheet_id=sheet_id,
                course_name=course_name,
                source_link=source_link,
                file_id=file_id,
                title=title,
                mime_type=mime_type,
                max_workers=max_workers,
            )
            entry["assets_indexed"] = n
            entry["status"] = "ok" if n > 0 else "error"
            if n == 0:
                entry["error"] = "No Drive video segments indexed"
            return entry

        if mime_type not in SUPPORTED_DOCUMENT_MIMES and kind != "document":
            if mime_type not in SUPPORTED_DOCUMENT_MIMES:
                entry["status"] = "skipped_unsupported_mime"
                entry["error"] = f"Unsupported MIME '{mime_type}' for '{title}'"
                print(f"⏭️ {entry['error']}")
                return entry

        asset_urls = extraction_assets.get(source_link) or []
        if not asset_urls:
            entry["status"] = "needs_extraction"
            entry["error"] = (
                "No extracted image URLs in external_ref_extraction_log. "
                "Run 'Extract External Reference Images' first."
            )
            print(f"⏭️ {entry['error']}")
            return entry

        deleted = delete_assets_for_source(sheet_id, source_link)
        if deleted:
            print(f"🗑️ Removed {deleted} old row(s) for re-index: {source_link}")

        n = _index_extracted_images(
            drive,
            sheet_id,
            course_name,
            source_link,
            asset_urls,
            max_workers=max_workers,
        )
        entry["assets_indexed"] = n
        entry["status"] = "ok" if n > 0 else "error"
        if n == 0:
            entry["error"] = "No images indexed from extraction URLs"
        return entry
    except Exception as exc:
        entry["status"] = "error"
        entry["error"] = str(exc)
        print(f"❌ Index failed for {source_link}: {exc}")
        traceback.print_exc()
        return entry


@traceable(
    metadata={
        "agent_name": "graphics_definition_v2",
        "step_name": "Index External Reference Assets",
        "function_name": "run_external_reference_indexing",
    }
)
def run_external_reference_indexing(sheet, max_workers=INDEX_MAX_WORKERS):
    """
    Diff External References links vs Supabase, remove stale rows, and embed assets in parallel.

    :param sheet: gspread sheet object
    :param max_workers: Maximum parallel workers for source and asset indexing
    :return: None
    """
    drive = _get_drive_instance()
    if drive is None:
        raise RuntimeError("Google Drive is not available. Log in before running this step.")

    sheet_id = _resolve_sheet_id(sheet)
    if not sheet_id:
        raise RuntimeError("Could not resolve spreadsheet id from the open sheet.")

    _, course_name, links = _read_course_info(sheet)
    course_label = course_name or "course"
    current_links = list(links)
    current_set = set(current_links)
    workers = max(1, int(max_workers or INDEX_MAX_WORKERS))

    print("\n" + "=" * 80)
    print("🚀 External Reference indexing (Supabase external_ref_assets only)")
    print(f"   Course: {course_label}")
    print(f"   Sheet id: {sheet_id}")
    print(f"   Links in Course info: {len(current_links)}")
    print(f"   max_workers={workers}")
    print("=" * 80 + "\n")

    indexed_links = set(list_source_links_for_sheet(sheet_id))
    to_remove = sorted(indexed_links - current_set)
    to_index = current_links  # re-index current set (idempotent replace per source)

    removed_counts = []
    for link in to_remove:
        n = delete_assets_for_source(sheet_id, link)
        removed_counts.append({"source_link": link, "deleted": n})
        print(f"🗑️ Removed stale source ({n} row(s)): {link}")

    extraction_assets = _read_extraction_log_assets(sheet)
    per_source = []
    total_indexed = 0

    # Parallelize across sources; each source also parallelizes image/segment embeds.
    work_links = []
    for link in to_index:
        existing = count_assets_for_source(sheet_id, link)
        has_extract = bool(extraction_assets.get(link))
        kind_hint = _classify_external_link(link)
        # Always re-index YouTube. Always re-index Drive videos (peek mime when needed).
        # Skip only already-indexed documents that have no extraction URLs.
        if existing > 0 and not has_extract and link in indexed_links:
            if kind_hint == "youtube":
                work_links.append(link)
                continue
            if kind_hint == "drive_unknown":
                try:
                    fid = parse_drive_file_id(link)
                    gfile = drive.CreateFile({"id": fid})
                    gfile.FetchMetadata(fields="mimeType")
                    if _is_drive_video_mime(str(gfile.get("mimeType") or "")):
                        work_links.append(link)
                        continue
                except Exception:
                    pass
            print(f"ℹ️ Already indexed ({existing} row(s)); no new extraction URLs. Skipping.")
            per_source.append(
                {
                    "source_link": link,
                    "status": "unchanged",
                    "assets_indexed": existing,
                    "error": "",
                }
            )
            continue
        work_links.append(link)

    if work_links:
        source_workers = max(1, min(workers, len(work_links)))
        print(
            f"🚀 Indexing {len(work_links)} source(s) in parallel "
            f"(source_workers={source_workers}, workers={workers})"
        )
        progress = SmartProgressBar(
            total_tasks=len(work_links),
            description="Index External Reference Assets",
        )
        results_by_link = {}
        with ThreadPoolExecutor(max_workers=source_workers) as executor:
            futures = {
                executor.submit(
                    _index_one_source,
                    drive,
                    sheet_id,
                    course_label,
                    link,
                    extraction_assets,
                    workers,
                ): link
                for link in work_links
            }
            for future in as_completed(futures):
                link = futures[future]
                try:
                    results_by_link[link] = future.result()
                except Exception as exc:
                    print(f"❌ Unexpected index failure for {link}: {exc}")
                    traceback.print_exc()
                    results_by_link[link] = {
                        "source_link": link,
                        "status": "error",
                        "assets_indexed": 0,
                        "error": str(exc),
                    }
                progress.update()
        for link in work_links:
            entry = results_by_link.get(link) or {
                "source_link": link,
                "status": "error",
                "assets_indexed": 0,
                "error": "missing result",
            }
            per_source.append(entry)
            total_indexed += int(entry.get("assets_indexed") or 0)

    summary = {
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "course_name": course_name,
        "sheet_id": sheet_id,
        "table": "external_ref_assets",
        "links_current": len(current_links),
        "max_workers": workers,
        "stale_sources_removed": removed_counts,
        "assets_indexed_this_run": total_indexed,
        "per_source": per_source,
    }
    _persist_index_log(sheet, json.dumps(summary, indent=2))

    print("\n" + "=" * 80)
    print(
        f"✅ External Reference indexing complete: "
        f"{total_indexed} asset(s) upserted this run (images + video segments)"
    )
    print("=" * 80 + "\n")
