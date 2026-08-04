"""
External Reference document media extraction for Graphics Definition V2.

One module: read Course info → External References, extract images from PDF / Google Docs / Google Slides / PPT via LlamaParse, upload to Drive.

Requires LLAMA_CLOUD_API_KEY in the environment.
"""

import base64
import hashlib
import json
import os
import re
import tempfile
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from io import BytesIO

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv
from langsmith import traceable
from pydrive2.drive import GoogleDrive

from services.drive_service import login_with_service_account
from services.sheets_service import format_worksheet, get_sheet_data_and_df
from services.smart_progress_bar import SmartProgressBar

load_dotenv()

# Shared parent folder for extracted external-reference images.
EXTERNAL_REF_ASSETS_FOLDER_ID = "1-sS6py5FVDJDI85rMkMBOQH2qIhccs_z"

MIME_GOOGLE_DOC = "application/vnd.google-apps.document"
MIME_GOOGLE_SLIDES = "application/vnd.google-apps.presentation"
MIME_PDF = "application/pdf"
MIME_PPT = "application/vnd.ms-powerpoint"
MIME_PPTX = "application/vnd.openxmlformats-officedocument.presentationml.presentation"

SUPPORTED_DOCUMENT_MIMES = {
    MIME_GOOGLE_DOC,
    MIME_GOOGLE_SLIDES,
    MIME_PDF,
    MIME_PPT,
    MIME_PPTX,
}
EXPORT_TO_PDF_MIMES = {MIME_GOOGLE_DOC, MIME_GOOGLE_SLIDES}
LLAMA_IMAGES_TO_SAVE = ["embedded"]

COURSE_INFO_WORKSHEET = "Course info"
EXTERNAL_REFERENCES_COLUMN = "External References"
EXTRACTION_LOG_COLUMN = "external_ref_extraction_log"
EXTRACT_MAX_WORKERS = 50
# Hamming distance for pHash near-duplicate matching (same image, different size/crop).
PHASH_HAMMING_THRESHOLD = 8


# ---------------------------------------------------------------------------
# Drive helpers
# ---------------------------------------------------------------------------

def _get_drive_instance():
    """
    Get Google Drive from session state or service-account env fallback.

    :return: GoogleDrive instance, or None if initialization fails
    """
    if "drive" in st.session_state:
        return st.session_state["drive"]
    try:
        key_bytes = base64.b64decode(os.environ["GDRIVE_SA_B64"])
        sa_json = key_bytes.decode()
        gauth = login_with_service_account(json_str=sa_json)
        gauth.ServiceAuth()
        return GoogleDrive(gauth)
    except Exception as exc:
        print(f"⚠️ Could not initialize Drive from environment: {exc}")
        return None


def _resolve_sheet_id(sheet):
    """
    Best-effort spreadsheet id from a gspread Spreadsheet.

    :param sheet: gspread Spreadsheet object
    :return: Spreadsheet id string, or empty string if unavailable
    """
    for attr in ("id", "spreadsheetId"):
        value = getattr(sheet, attr, None)
        if value:
            return str(value).strip()
    try:
        return str(sheet._properties.get("spreadsheetId") or "").strip()
    except Exception:
        return ""


def parse_drive_file_id(url_or_id):
    """
    Extract a Google Drive file id from a URL, or return the string if it looks like an id.

    :param url_or_id: Drive URL or raw file id
    :return: Parsed file id, or None if not found
    """
    text = (url_or_id or "").strip()
    if not text:
        return None
    # Strip clip-window suffixes like "(start=12&end=34)" before parsing.
    text = text.split("(start=")[0].strip()
    for pattern in (
        r"/file/d/([a-zA-Z0-9_-]+)",
        r"/document/d/([a-zA-Z0-9_-]+)",
        r"/presentation/d/([a-zA-Z0-9_-]+)",
        r"/d/([a-zA-Z0-9_-]+)",
        r"[?&]id=([a-zA-Z0-9_-]+)",
    ):
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    if re.fullmatch(r"[a-zA-Z0-9_-]{20,}", text):
        return text
    return None


def is_drive_file_under_external_ref_assets(drive, url_or_id, parent_folder_id=EXTERNAL_REF_ASSETS_FOLDER_ID, cache=None, max_depth=12):
    """
    Return True if a Drive file lives under the shared external-ref assets folder tree.

    :param drive: GoogleDrive instance
    :param url_or_id: Drive URL or file id (clip suffixes allowed)
    :param parent_folder_id: Shared external-ref parent folder id
    :param cache: Optional dict file_id -> bool for reuse within one revise call
    :param max_depth: Max parent hops to walk
    :return: True when the file is under parent_folder_id; False if unknown / on error
    """
    if not drive or not parent_folder_id:
        return False
    file_id = parse_drive_file_id(url_or_id)
    if not file_id:
        return False
    if cache is not None and file_id in cache:
        return bool(cache[file_id])

    root = str(parent_folder_id).strip()
    if file_id == root:
        result = True
        if cache is not None:
            cache[file_id] = result
        return result

    try:
        current = file_id
        seen = set()
        depth = 0
        while current and current not in seen and depth < max_depth:
            seen.add(current)
            if current == root:
                result = True
                if cache is not None:
                    cache[file_id] = result
                return result
            gfile = drive.CreateFile({"id": current})
            gfile.FetchMetadata()
            parents = gfile.get("parents") or []
            parent_ids = []
            for parent in parents:
                pid = parent.get("id") if isinstance(parent, dict) else parent
                if pid:
                    parent_ids.append(str(pid).strip())
            if root in parent_ids:
                result = True
                if cache is not None:
                    cache[file_id] = result
                return result
            current = parent_ids[0] if parent_ids else None
            depth += 1
        result = False
    except Exception as exc:
        print(f"⚠️ Could not check external-ref folder ancestry for {file_id}: {exc}")
        result = False

    if cache is not None:
        cache[file_id] = result
    return result


def _suffix_for_mime(mime_type, title=""):
    """
    Choose a file suffix for a Drive document mime type.

    :param mime_type: Drive mimeType string
    :param title: Optional document title for extension fallback
    :return: File suffix including leading dot
    """
    mime = (mime_type or "").strip()
    if mime in EXPORT_TO_PDF_MIMES or mime == MIME_PDF:
        return ".pdf"
    if mime == MIME_PPTX:
        return ".pptx"
    if mime == MIME_PPT:
        return ".ppt"
    ext = os.path.splitext(title or "")[1].lower()
    return ext if ext else ".bin"


def download_document_source(drive, source_link):
    """
    Download or export a supported Drive document to a local temp file.

    :param drive: GoogleDrive instance
    :param source_link: Drive URL or file id for the source document
    :return: Tuple of (local_path, mime_type, title)
    """
    file_id = parse_drive_file_id(source_link)
    if not file_id:
        raise ValueError(f"Could not parse Drive file id from: {source_link}")

    gfile = drive.CreateFile({"id": file_id})
    gfile.FetchMetadata(fields="title,mimeType")
    title = str(gfile.get("title") or "document").strip() or "document"
    mime_type = str(gfile.get("mimeType") or "").strip()

    if mime_type not in SUPPORTED_DOCUMENT_MIMES:
        raise ValueError(
            f"Unsupported External References document type '{mime_type}' "
            f"for '{title}'. Supported: PDF, Google Docs, Google Slides, PPT/PPTX."
        )

    suffix = _suffix_for_mime(mime_type, title)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    local_path = tmp.name
    tmp.close()

    try:
        if mime_type in EXPORT_TO_PDF_MIMES:
            gfile.GetContentFile(local_path, mimetype=MIME_PDF)
            mime_type = MIME_PDF
        else:
            gfile.GetContentFile(local_path)
    except Exception:
        if os.path.exists(local_path):
            os.remove(local_path)
        raise

    return local_path, mime_type, title


def sanitize_folder_name(name, max_len=80):
    """
    Sanitize a string for use as a Drive folder name.

    :param name: Raw folder name
    :param max_len: Maximum length after sanitization
    :return: Safe folder name string
    """
    text = (name or "").strip()
    text = re.sub(r"[\\/:*?\"<>|]+", "_", text)
    text = re.sub(r"\s+", " ", text).strip(" ._")
    if not text:
        text = "course"
    return text[:max_len]


def source_link_hash(source_link):
    """
    Compute a short stable hash for an external reference source link.

    :param source_link: Source URL or Drive link
    :return: 12-character hex hash prefix
    """
    normalized = (source_link or "").strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:12]


def course_folder_name(course_name, sheet_id):
    """
    Build the course-level Drive folder name for extracted assets.

    :param course_name: Course display name
    :param sheet_id: Spreadsheet id
    :return: Folder name in the form `{course}_{sheet_id}`
    """
    return f"{sanitize_folder_name(course_name)}_{sanitize_folder_name(sheet_id or 'unknown_sheet', max_len=64)}"


def find_child_folder(drive, parent_folder_id, folder_name):
    """
    Find an existing child folder by exact title.

    :param drive: GoogleDrive instance
    :param parent_folder_id: Parent folder id
    :param folder_name: Exact child folder title to match
    :return: Child folder id, or None if not found
    """
    safe_name = (folder_name or "").strip()
    if not safe_name or not parent_folder_id:
        return None

    escaped = safe_name.replace("'", "\\'")
    query = (
        f"'{parent_folder_id}' in parents and title='{escaped}' "
        f"and mimeType='application/vnd.google-apps.folder' and trashed=false"
    )
    existing = drive.ListFile(
        {
            "q": query,
            "supportsAllDrives": True,
            "includeItemsFromAllDrives": True,
        }
    ).GetList()
    if existing:
        return existing[0]["id"]
    return None


def find_or_create_child_folder(drive, parent_folder_id, folder_name):
    """
    Reuse an existing child folder by exact title, or create it.

    :param drive: GoogleDrive instance
    :param parent_folder_id: Parent folder id
    :param folder_name: Child folder title
    :return: Child folder id
    """
    safe_name = (folder_name or "").strip()
    if not safe_name:
        raise ValueError("folder_name cannot be empty")
    if not parent_folder_id:
        raise ValueError("parent_folder_id is required")

    existing_id = find_child_folder(drive, parent_folder_id, safe_name)
    if existing_id:
        return existing_id

    folder = drive.CreateFile(
        {
            "title": safe_name,
            "mimeType": "application/vnd.google-apps.folder",
            "parents": [{"id": parent_folder_id}],
        }
    )
    folder.Upload(param={"supportsAllDrives": True})
    return folder["id"]


def delete_course_extract_folder(drive, course_name, sheet_id, parent_folder_id=EXTERNAL_REF_ASSETS_FOLDER_ID):
    """
    Delete the course extract folder and all uploaded images inside it.

    :param drive: GoogleDrive instance
    :param course_name: Course display name
    :param sheet_id: Spreadsheet id
    :param parent_folder_id: Shared parent folder id (defaults to EXTERNAL_REF_ASSETS_FOLDER_ID)
    :return: True if a folder was deleted, False otherwise
    """
    if drive is None:
        print("⚠️ Drive unavailable; cannot delete extract folder")
        return False
    folder_name = course_folder_name(course_name, sheet_id)
    parent_id = parent_folder_id or EXTERNAL_REF_ASSETS_FOLDER_ID
    folder_id = find_child_folder(drive, parent_id, folder_name)
    if not folder_id:
        print(f"ℹ️ No Drive extract folder found to delete: {folder_name}")
        return False
    try:
        drive.CreateFile({"id": folder_id}).Delete()
        print(f"🗑️ Deleted Drive extract folder: {folder_name} ({folder_id})")
        return True
    except Exception as exc:
        print(f"❌ Failed to delete Drive extract folder '{folder_name}': {exc}")
        traceback.print_exc()
        return False


def ensure_source_upload_folder(drive, course_name, sheet_id, source_link, parent_folder_id=EXTERNAL_REF_ASSETS_FOLDER_ID):
    """
    Ensure `{Course Name}_{sheet_id}/{source_hash}` exists under the parent folder.

    :param drive: GoogleDrive instance
    :param course_name: Course display name
    :param sheet_id: Spreadsheet id
    :param source_link: External reference source URL
    :param parent_folder_id: Shared parent folder id
    :return: Dict with course_folder_id, source_folder_id, source_hash, and course_folder_name
    """
    course_name_final = course_folder_name(course_name, sheet_id)
    src_hash = source_link_hash(source_link)
    course_folder_id = find_or_create_child_folder(drive, parent_folder_id, course_name_final)
    source_folder_id = find_or_create_child_folder(drive, course_folder_id, src_hash)
    return {
        "course_folder_id": course_folder_id,
        "source_folder_id": source_folder_id,
        "source_hash": src_hash,
        "course_folder_name": course_name_final,
    }


def _extension_for_mime(mime_type, fallback_name=""):
    """
    Map an image mime type to a file extension.

    :param mime_type: Image mime type string
    :param fallback_name: Optional filename for extension fallback
    :return: File extension including leading dot
    """
    mime = (mime_type or "").lower()
    if "png" in mime:
        return ".png"
    if "webp" in mime:
        return ".webp"
    if "gif" in mime:
        return ".gif"
    ext = os.path.splitext(fallback_name or "")[1].lower()
    if ext in {".jpg", ".jpeg", ".png", ".webp", ".gif"}:
        return ".jpg" if ext == ".jpeg" else ext
    return ".jpg"


def upload_image_bytes_to_drive(drive, folder_id, filename, data, mime_type="image/jpeg"):
    """
    Upload image bytes to Drive and return a view URL.

    :param drive: GoogleDrive instance
    :param folder_id: Destination folder id
    :param filename: Uploaded file title
    :param data: Raw image bytes
    :param mime_type: Image mime type for the upload
    :return: Drive view URL, or None on failure
    """
    if not data:
        return None
    try:
        metadata = {
            "title": filename,
            "mimeType": mime_type or "image/jpeg",
            "parents": [{"id": folder_id}],
        }
        gfile = drive.CreateFile(metadata)
        gfile.content = BytesIO(data)
        gfile.Upload(param={"supportsAllDrives": True})
        try:
            gfile.InsertPermission({"type": "anyone", "value": "anyone", "role": "reader"})
        except Exception as perm_err:
            print(f"⚠️ Could not set anyone-with-link on {filename}: {perm_err}")
        return f"https://drive.google.com/file/d/{gfile['id']}/view"
    except Exception as exc:
        print(f"❌ Failed to upload {filename}: {exc}")
        return None


def _upload_one_extracted_image(job, drive, source_folder_id, source_link, src_hash):
    """
    Upload one extracted image job to Drive.

    :param job: Tuple of (idx, item, filename, mime_type, data)
    :param drive: GoogleDrive instance
    :param source_folder_id: Destination Drive folder id
    :param source_link: External reference source URL
    :param src_hash: Short hash of the source link
    :return: Tuple of (idx, asset dict), or None on failure
    """
    idx, _item, filename, mime_type, data = job
    asset_url = upload_image_bytes_to_drive(
        drive=drive,
        folder_id=source_folder_id,
        filename=filename,
        data=data,
        mime_type=mime_type,
    )
    if not asset_url:
        return None
    print(f"☁️ Uploaded {filename} → {asset_url}")
    return (
        idx,
        {
            "asset_url": asset_url,
            "asset_type": "image",
            "source_link": source_link,
            "mime_type": mime_type,
            "source_hash": src_hash,
            "drive_folder_id": source_folder_id,
            "filename": filename,
        },
    )


def upload_extracted_images(drive, extracted_images, course_name, sheet_id, source_link, parent_folder_id=EXTERNAL_REF_ASSETS_FOLDER_ID, max_workers=EXTRACT_MAX_WORKERS):
    """
    Upload extracted local images into the course/source Drive folder (parallel).

    :param drive: GoogleDrive instance
    :param extracted_images: List of dicts with local_path, mime_type, and filename
    :param course_name: Course display name
    :param sheet_id: Spreadsheet id
    :param source_link: External reference source URL
    :param parent_folder_id: Shared parent folder id
    :param max_workers: Max parallel upload workers
    :return: List of uploaded asset metadata dicts
    """
    folder_info = ensure_source_upload_folder(
        drive=drive,
        course_name=course_name,
        sheet_id=sheet_id,
        source_link=source_link,
        parent_folder_id=parent_folder_id,
    )
    source_folder_id = folder_info["source_folder_id"]
    src_hash = folder_info["source_hash"]

    upload_jobs = []
    for idx, item in enumerate(extracted_images or [], start=1):
        local_path = item.get("local_path") or ""
        if not local_path or not os.path.isfile(local_path):
            continue
        mime_type = str(item.get("mime_type") or "image/jpeg")
        ext = _extension_for_mime(mime_type, item.get("filename") or "")
        filename = f"img_{idx:03d}{ext}"
        with open(local_path, "rb") as fh:
            data = fh.read()
        if not data:
            continue
        upload_jobs.append((idx, item, filename, mime_type, data))

    if not upload_jobs:
        return []

    workers = max(1, min(int(max_workers or EXTRACT_MAX_WORKERS), len(upload_jobs)))
    print(f"☁️ Uploading {len(upload_jobs)} image(s) to Drive in parallel (max_workers={workers})...")

    results_by_idx = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _upload_one_extracted_image,
                job,
                drive,
                source_folder_id,
                source_link,
                src_hash,
            ): job[0]
            for job in upload_jobs
        }
        for future in as_completed(futures):
            idx = futures[future]
            try:
                result = future.result()
                if result:
                    results_by_idx[result[0]] = result[1]
            except Exception as exc:
                print(f"❌ Failed Drive upload for image {idx}: {exc}")
                traceback.print_exc()

    return [results_by_idx[i] for i in sorted(results_by_idx)]


# ---------------------------------------------------------------------------
# LlamaParse extraction
# ---------------------------------------------------------------------------

def _get_llama_client():
    """
    Build a LlamaCloud client from LLAMA_CLOUD_API_KEY in the environment.

    :return: LlamaCloud client instance
    """
    api_key = (os.getenv("LLAMA_CLOUD_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError(
            "LLAMA_CLOUD_API_KEY is not set. Add it to .env before extracting external-reference images."
        )
    from llama_cloud import LlamaCloud

    return LlamaCloud(api_key=api_key)


def _iter_image_entries(result):
    """
    Yield image metadata entries from a LlamaParse parse result.

    :param result: LlamaParse parse result object
    :return: List of image entry objects or dicts
    """
    meta = getattr(result, "images_content_metadata", None)
    if meta is None:
        return []
    images = getattr(meta, "images", None)
    if images is None and isinstance(meta, dict):
        images = meta.get("images")
    return list(images or [])


def _download_one_llamaparse_image(idx, image):
    """
    Download one LlamaParse image entry to a temp file.

    :param idx: 1-based image index for logging
    :param image: LlamaParse image entry object or dict
    :return: Tuple of (idx, asset dict), or None if download fails
    """
    presigned_url = getattr(image, "presigned_url", None)
    if presigned_url is None and isinstance(image, dict):
        presigned_url = image.get("presigned_url")
    if not presigned_url:
        return None

    filename = getattr(image, "filename", None)
    if filename is None and isinstance(image, dict):
        filename = image.get("filename")
    filename = str(filename or f"image_{idx}.jpg").strip() or f"image_{idx}.jpg"

    content_type = getattr(image, "content_type", None)
    if content_type is None and isinstance(image, dict):
        content_type = image.get("content_type")
    content_type = str(content_type or "image/jpeg").strip() or "image/jpeg"

    ext = os.path.splitext(filename)[1].lower()
    if not ext:
        if "png" in content_type:
            ext = ".png"
        elif "webp" in content_type:
            ext = ".webp"
        else:
            ext = ".jpg"

    response = requests.get(presigned_url, timeout=120)
    response.raise_for_status()
    data = response.content
    if not data:
        return None

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    tmp.write(data)
    tmp.close()
    print(f"  ✅ Extracted image {idx}: {filename} ({len(data)} bytes)")
    return (idx, {"local_path": tmp.name, "filename": filename, "mime_type": content_type})


def _image_quality_key(local_path):
    """
    Score an image for dedupe keep/discard decisions (higher is better).

    :param local_path: Path to a local image file
    :return: Tuple of (pixel_count, file_bytes)
    """
    file_bytes = 0
    try:
        file_bytes = os.path.getsize(local_path)
    except OSError:
        pass
    pixels = 0
    try:
        from PIL import Image

        with Image.open(local_path) as img:
            width, height = img.size
            pixels = int(width) * int(height)
    except Exception:
        pass
    return (pixels, file_bytes)


def _compute_phash(local_path):
    """
    Compute a perceptual hash for near-duplicate detection.

    :param local_path: Path to a local image file
    :return: imagehash object, or None if unavailable
    """
    try:
        import imagehash
        from PIL import Image

        with Image.open(local_path) as img:
            return imagehash.phash(img.convert("RGB"))
    except Exception as exc:
        print(f"⚠️ Could not compute pHash for {local_path}: {exc}")
        return None


def _replace_kept_duplicate(kept, kept_md5, kept_phash, discarded_paths, idx, new_item, new_md5, new_phash):
    """
    Replace a kept duplicate with a higher-quality version and queue the old temp file.

    :param kept: List of kept image dicts
    :param kept_md5: Parallel list of MD5 digests for kept images
    :param kept_phash: Parallel list of pHash values for kept images
    :param discarded_paths: List collecting local paths to delete
    :param idx: Index in the kept list to replace
    :param new_item: Replacement image dict
    :param new_md5: MD5 digest of the replacement
    :param new_phash: pHash of the replacement
    :return: None
    """
    old_path = kept[idx].get("local_path") or ""
    if old_path:
        discarded_paths.append(old_path)
    kept[idx] = new_item
    kept_md5[idx] = new_md5
    kept_phash[idx] = new_phash


def _dedupe_extracted_images(extracted, phash_threshold=PHASH_HAMMING_THRESHOLD):
    """
    Drop exact and near-duplicate images from one LlamaParse result.

    :param extracted: List of extracted image dicts with local_path
    :param phash_threshold: Max Hamming distance for near-duplicate matching
    :return: Deduped list of image dicts
    """
    if not extracted:
        return []

    kept = []
    kept_md5 = []
    kept_phash = []
    discarded_paths = []

    for item in extracted:
        local_path = item.get("local_path") or ""
        if not local_path or not os.path.isfile(local_path):
            continue

        try:
            with open(local_path, "rb") as fh:
                digest = hashlib.md5(fh.read()).hexdigest()
        except Exception:
            digest = ""

        quality = _image_quality_key(local_path)
        filename = item.get("filename") or os.path.basename(local_path)
        phash = _compute_phash(local_path)

        # Exact byte duplicate
        exact_idx = next(
            (idx for idx, existing_md5 in enumerate(kept_md5) if digest and existing_md5 == digest),
            None,
        )
        if exact_idx is not None:
            existing_quality = _image_quality_key(kept[exact_idx].get("local_path") or "")
            if quality > existing_quality:
                _replace_kept_duplicate(kept, kept_md5, kept_phash, discarded_paths, exact_idx, item, digest, phash)
                print(f"♻️ Exact duplicate replaced with larger file: {filename}")
            else:
                discarded_paths.append(local_path)
                print(f"♻️ Dropped exact duplicate: {filename}")
            continue

        # Near-duplicate (same visual, different size/crop)
        near_idx = None
        if phash is not None:
            for idx, existing_phash in enumerate(kept_phash):
                if existing_phash is None:
                    continue
                try:
                    if (phash - existing_phash) <= phash_threshold:
                        near_idx = idx
                        break
                except Exception:
                    continue

        if near_idx is not None:
            existing_quality = _image_quality_key(kept[near_idx].get("local_path") or "")
            if quality > existing_quality:
                _replace_kept_duplicate(kept, kept_md5, kept_phash, discarded_paths, near_idx, item, digest, phash)
                print(f"♻️ Near-duplicate replaced with larger file: {filename}")
            else:
                discarded_paths.append(local_path)
                print(f"♻️ Dropped near-duplicate: {filename}")
            continue

        kept.append(item)
        kept_md5.append(digest)
        kept_phash.append(phash)

    for path in discarded_paths:
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass

    dropped = len(extracted) - len(kept)
    if dropped > 0:
        print(
            f"🧹 Deduped LlamaParse images: kept {len(kept)} / {len(extracted)} "
            f"(dropped {dropped})"
        )
    return kept


def extract_images_with_llamaparse(local_path, images_to_save=None, max_workers=EXTRACT_MAX_WORKERS):
    """
    Parse a local document with LlamaParse and download extracted images to temp files (parallel).

    :param local_path: Path to the local document file
    :param images_to_save: Optional LlamaParse image categories to save
    :param max_workers: Max parallel download workers
    :return: List of extracted image dicts with local_path, filename, and mime_type
    """
    if not local_path or not os.path.isfile(local_path):
        raise FileNotFoundError(f"Document not found for LlamaParse: {local_path}")

    categories = list(images_to_save or LLAMA_IMAGES_TO_SAVE)
    client = _get_llama_client()

    print(f"📄 Uploading to LlamaParse: {os.path.basename(local_path)}")
    uploaded = client.files.create(file=local_path, purpose="parse")
    file_id = getattr(uploaded, "id", None) or (uploaded.get("id") if isinstance(uploaded, dict) else None)
    if not file_id:
        raise RuntimeError("LlamaParse file upload did not return a file id")

    print(f"🔎 Parsing with LlamaParse (images={categories})...")
    result = client.parsing.parse(
        file_id=file_id,
        tier="agentic",
        version="latest",
        output_options={"images_to_save": categories},
        expand=["images_content_metadata"],
    )

    image_entries = list(enumerate(_iter_image_entries(result), start=1))
    if not image_entries:
        print("🖼️ LlamaParse extracted 0 image(s)")
        return []

    workers = max(1, min(int(max_workers or EXTRACT_MAX_WORKERS), len(image_entries)))
    print(f"📥 Downloading {len(image_entries)} LlamaParse image(s) in parallel (max_workers={workers})...")

    results_by_idx = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_download_one_llamaparse_image, idx, image): idx
            for idx, image in image_entries
        }
        for future in as_completed(futures):
            idx = futures[future]
            try:
                result_item = future.result()
                if result_item:
                    results_by_idx[result_item[0]] = result_item[1]
            except Exception as exc:
                print(f"❌ Failed to download LlamaParse image {idx}: {exc}")
                traceback.print_exc()

    extracted = [results_by_idx[i] for i in sorted(results_by_idx)]
    print(f"🖼️ LlamaParse downloaded {len(extracted)} image(s)")
    extracted = _dedupe_extracted_images(extracted)
    print(f"🖼️ LlamaParse unique images after dedupe: {len(extracted)}")
    return extracted


def extract_and_upload_from_source(source_link, drive, course_name, sheet_id, parent_folder_id=EXTERNAL_REF_ASSETS_FOLDER_ID, max_workers=EXTRACT_MAX_WORKERS):
    """
    Extract images from one supported document link and upload them to Drive.

    :param source_link: External reference document URL
    :param drive: GoogleDrive instance
    :param course_name: Course display name
    :param sheet_id: Spreadsheet id
    :param parent_folder_id: Shared parent folder id
    :param max_workers: Max parallel workers for LlamaParse downloads and Drive uploads
    :return: List of uploaded asset metadata dicts
    """
    if not source_link or not str(source_link).strip():
        raise ValueError("source_link is required")
    if drive is None:
        raise ValueError("drive instance is required")
    if not sheet_id or not str(sheet_id).strip():
        raise ValueError("sheet_id is required")

    source_link = str(source_link).strip()
    workers = max(1, int(max_workers or EXTRACT_MAX_WORKERS))
    local_doc_path = None
    extracted_paths = []

    try:
        local_doc_path, mime_type, title = download_document_source(drive, source_link)
        print(f"📥 Downloaded '{title}' ({mime_type}) → {local_doc_path}")

        extracted = extract_images_with_llamaparse(local_doc_path, max_workers=workers)
        extracted_paths = [item.get("local_path") for item in extracted if item.get("local_path")]
        if not extracted:
            print(f"⚠️ No images extracted from {source_link}")
            return []

        assets = upload_extracted_images(
            drive=drive,
            extracted_images=extracted,
            course_name=course_name or "course",
            sheet_id=str(sheet_id).strip(),
            source_link=source_link,
            parent_folder_id=parent_folder_id or EXTERNAL_REF_ASSETS_FOLDER_ID,
            max_workers=workers,
        )
        print(f"✅ External reference source done: {len(assets)} image(s) uploaded")
        return assets
    finally:
        if local_doc_path and os.path.exists(local_doc_path):
            try:
                os.remove(local_doc_path)
            except OSError:
                pass
        for path in extracted_paths:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass


# ---------------------------------------------------------------------------
# Graphics Definition V2 pipeline step
# ---------------------------------------------------------------------------

def parse_external_reference_links(raw):
    """
    Parse newline-separated External References cell into unique URLs.

    :param raw: Raw cell value from Course info
    :return: List of unique link strings
    """
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    text = str(raw).strip()
    if not text or text.lower() == "nan":
        return []
    links = []
    seen = set()
    for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        url = line.strip()
        if not url or url in seen:
            continue
        seen.add(url)
        links.append(url)
    return links


def _read_course_info(sheet):
    """
    Read course name and external reference links from the Course info worksheet.

    :param sheet: gspread Spreadsheet object
    :return: Tuple of (course_info_df, course_name, links)
    """
    _, course_info_df = get_sheet_data_and_df(sheet, COURSE_INFO_WORKSHEET)
    if course_info_df is None or course_info_df.empty:
        return course_info_df, "", []

    course_name = ""
    for col in ("Course Name", "course name", "Course name"):
        if col in course_info_df.columns:
            course_name = str(course_info_df.iloc[0].get(col, "") or "").strip()
            if course_name and course_name.lower() != "nan":
                break

    links = []
    for col in (EXTERNAL_REFERENCES_COLUMN, "external references", "External references"):
        if col in course_info_df.columns:
            links = parse_external_reference_links(course_info_df.iloc[0].get(col))
            break
    return course_info_df, course_name, links


def _is_probably_supported_document_link(url):
    """
    Heuristic check whether a URL likely points to a supported document type.

    :param url: External reference URL
    :return: True if the link is probably a supported document
    """
    lower = (url or "").lower()
    if "youtube.com" in lower or "youtu.be" in lower:
        return False
    if "/folders/" in lower:
        return False
    if "docs.google.com/document" in lower or "docs.google.com/presentation" in lower:
        return True
    return bool(parse_drive_file_id(url))


def _persist_extraction_log(sheet, log_text):
    """
    Write the extraction summary JSON to Course info → external_ref_extraction_log.

    :param sheet: gspread Spreadsheet object
    :param log_text: JSON log text to persist
    :return: None
    """
    try:
        worksheet = None
        for candidate in sheet.worksheets():
            if candidate.title.strip().lower() == COURSE_INFO_WORKSHEET.lower():
                worksheet = candidate
                break
        if worksheet is None:
            print(f"⚠️ '{COURSE_INFO_WORKSHEET}' tab not found; skipping extraction log persist")
            return

        headers = worksheet.row_values(1)
        col = None
        created_column = False
        for idx, header in enumerate(headers):
            if str(header).strip().lower() == EXTRACTION_LOG_COLUMN.lower():
                col = idx + 1
                break
        if col is None:
            headers.append(EXTRACTION_LOG_COLUMN)
            col = len(headers)
            if col > worksheet.col_count:
                worksheet.add_cols(col - worksheet.col_count)
            worksheet.update_cell(1, col, EXTRACTION_LOG_COLUMN)
            created_column = True

        worksheet.update_cell(2, col, log_text)
        if created_column:
            format_worksheet(worksheet)
        print(f"📝 Persisted {EXTRACTION_LOG_COLUMN} to Course info")
    except Exception as exc:
        print(f"⚠️ Could not persist extraction log: {exc}")


def delete_external_reference_extraction_log(sheet, parent_folder_id=EXTERNAL_REF_ASSETS_FOLDER_ID):
    """
    Delete extracted Drive assets and clear the extraction log for this course.

    :param sheet: gspread Spreadsheet object
    :param parent_folder_id: Shared parent folder id
    :return: None
    """
    drive = _get_drive_instance()
    sheet_id = _resolve_sheet_id(sheet)
    _, course_name, _ = _read_course_info(sheet)
    if sheet_id:
        delete_course_extract_folder(
            drive=drive,
            course_name=course_name or "course",
            sheet_id=sheet_id,
            parent_folder_id=parent_folder_id or EXTERNAL_REF_ASSETS_FOLDER_ID,
        )
    else:
        print("⚠️ Could not resolve sheet id; skipping Drive folder delete")

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
            if header.lower() == EXTRACTION_LOG_COLUMN.lower():
                worksheet.update_cell(2, idx + 1, "")
                print(f"🗑️ Cleared '{EXTRACTION_LOG_COLUMN}' in Course info")
                return
        print(f"ℹ️ '{EXTRACTION_LOG_COLUMN}' column does not exist")
    except Exception as exc:
        print(f"⚠️ Could not clear extraction log: {exc}")


def _process_one_extraction_source(idx, link, links_total, drive, course_label, sheet_id, parent_id, workers):
    """
    Extract and upload images for one external reference source link.

    :param idx: 1-based source index for logging
    :param link: External reference document URL
    :param links_total: Total number of source links in this run
    :param drive: GoogleDrive instance
    :param course_label: Course display name
    :param sheet_id: Spreadsheet id
    :param parent_id: Shared parent folder id for uploaded assets
    :param workers: Max parallel workers for within-source image work
    :return: Per-source result dict with status, assets, and optional error
    """
    print(f"\n—— Source {idx}/{links_total} ——")
    print(f"🔗 {link}")
    entry = {"source_link": link, "status": "skipped", "assets": 0, "error": ""}

    if not _is_probably_supported_document_link(link):
        entry["status"] = "skipped_non_document"
        entry["error"] = "Not a supported document link (PDF/Doc/Slides/PPT) for this step"
        print(f"⏭️ {entry['error']}")
        return entry

    file_id = parse_drive_file_id(link)
    try:
        gfile = drive.CreateFile({"id": file_id})
        gfile.FetchMetadata(fields="title,mimeType")
        mime_type = str(gfile.get("mimeType") or "").strip()
        title = str(gfile.get("title") or "").strip()
        if mime_type not in SUPPORTED_DOCUMENT_MIMES:
            entry["status"] = "skipped_unsupported_mime"
            entry["error"] = f"Unsupported MIME '{mime_type}' for '{title}'"
            print(f"⏭️ {entry['error']}")
            return entry

        assets = extract_and_upload_from_source(
            source_link=link,
            drive=drive,
            course_name=course_label,
            sheet_id=sheet_id,
            parent_folder_id=parent_id,
            max_workers=workers,
        )
        entry["status"] = "ok"
        entry["assets"] = len(assets)
        entry["asset_urls"] = [a.get("asset_url") for a in assets if a.get("asset_url")]
        print(f"✅ {len(assets)} image(s) uploaded for this source")
    except Exception as exc:
        entry["status"] = "error"
        entry["error"] = str(exc)
        print(f"❌ Extraction failed for {link}: {exc}")
        traceback.print_exc()
    return entry


@traceable(
    metadata={
        "agent_name": "graphics_definition_v2",
        "step_name": "Extract External Reference Media",
        "function_name": "run_external_reference_extraction",
    }
)
def run_external_reference_extraction(sheet, parent_folder_id=EXTERNAL_REF_ASSETS_FOLDER_ID, max_workers=EXTRACT_MAX_WORKERS):
    """
    Read External References from Course info, extract images, and upload them to Drive.

    :param sheet: gspread Spreadsheet object
    :param parent_folder_id: Shared parent folder id for uploaded assets
    :param max_workers: Max parallel workers across sources and within each source
    :return: None
    """
    drive = _get_drive_instance()
    if drive is None:
        raise RuntimeError("Google Drive is not available. Log in before running this step.")

    sheet_id = _resolve_sheet_id(sheet)
    if not sheet_id:
        raise RuntimeError("Could not resolve spreadsheet id from the open sheet.")

    _, course_name, links = _read_course_info(sheet)
    if not links:
        print("ℹ️ No links in Course info → External References. Nothing to extract.")
        _persist_extraction_log(
            sheet,
            f"[{datetime.now(timezone.utc).isoformat()}] No External References links found.",
        )
        return

    parent_id = parent_folder_id or EXTERNAL_REF_ASSETS_FOLDER_ID
    course_label = course_name or "course"
    workers = max(1, int(max_workers or EXTRACT_MAX_WORKERS))
    source_workers = max(1, min(workers, len(links)))

    print("\n" + "=" * 80)
    print(f"🚀 External Reference extraction ({len(links)} link(s), parallel)")
    print(f"   Course: {course_label}")
    print(f"   Sheet id: {sheet_id}")
    print(f"   Parent folder: {parent_id}")
    print(f"   source_workers={source_workers}, image_workers={workers}")
    print("=" * 80 + "\n")

    per_source = [None] * len(links)
    progress = SmartProgressBar(
        total_tasks=len(links),
        description="Extract External Reference Images",
    )
    with ThreadPoolExecutor(max_workers=source_workers) as executor:
        futures = {
            executor.submit(
                _process_one_extraction_source,
                idx,
                link,
                len(links),
                drive,
                course_label,
                sheet_id,
                parent_id,
                workers,
            ): idx - 1
            for idx, link in enumerate(links, start=1)
        }
        for future in as_completed(futures):
            out_idx = futures[future]
            try:
                per_source[out_idx] = future.result()
            except Exception as exc:
                link = links[out_idx]
                print(f"❌ Unexpected failure for {link}: {exc}")
                per_source[out_idx] = {
                    "source_link": link,
                    "status": "error",
                    "assets": 0,
                    "error": str(exc),
                }
            progress.update()

    results = [entry for entry in per_source if entry is not None]
    total_assets = sum(int(entry.get("assets") or 0) for entry in results)
    summary = {
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "course_name": course_name,
        "sheet_id": sheet_id,
        "parent_folder_id": parent_id,
        "sources_total": len(links),
        "max_workers": workers,
        "images_uploaded": total_assets,
        "per_source": results,
    }
    _persist_extraction_log(sheet, json.dumps(summary, indent=2))

    print("\n" + "=" * 80)
    print(f"✅ External Reference extraction complete: {total_assets} image(s) from {len(links)} link(s)")
    print("=" * 80 + "\n")
