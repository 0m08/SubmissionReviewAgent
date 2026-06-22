"""
Google Drive Video Search
=========================

Streamlit UI + backend helpers for searching a ChromaDB vectorstore created from
Google Drive videos using Gemini Embedding 2.

This file is designed as the Gemini Embedding 2 replacement for older
Vertex AI `multimodalembedding@001` YouTube video search code.

What it does:
- Downloads the ChromaDB video vectorstore from Google Drive using PyDrive2.
- Loads the Gemini video embeddings collection from ChromaDB.
- Creates query embeddings with Gemini Embedding 2.
- Supports text, image, and video query search.
- Returns timestamped segments from your original Google Drive videos.
- Renders a Streamlit tab/UI that can be added beside your existing search tabs.

Required environment variables:
- GEMINI_API_KEY
- GDRIVE_SA_B64, if you want this file to initialize Drive by itself.

Expected Chroma collection name:
- gemini_video_visual_embeddings

Expected metadata stored per vector:
- video_id
- video_name
- video_link
- course_name
- topic_name
- tab_name
- stock
- mime_type
- duration
- file_size
- segment_index
- start_time
- end_time
- optional transcript / transcript_text / text
"""

from __future__ import annotations

import base64
import json
import math
import os
import shutil

import tempfile
import threading
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple, Union

import chromadb
import subprocess
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydrive2.drive import GoogleDrive
from agents.course_outline.video_search_tool.google_drive_video_search.create_google_drive_video_vectorstore import create_video_embeddings
from agents.course_outline.video_search_tool.google_drive_video_search.archive_google_drive_videos import render_archive_tab

try:
    from services.drive_service import login_with_service_account
except Exception:
    login_with_service_account = None

try:
    from services.activity_tracking_service import track_tool_action
except Exception:
    def track_tool_action(*args, **kwargs):
        return None

load_dotenv()

# ============================================================
# Configuration
# ============================================================

# Parent Drive folder that contains your vectorstore folder.
# Change this to the parent folder ID used by your creation pipeline.
# GEMINI_VIDEO_VECTORSTORE_PARENT_FOLDER_ID = "15H9thXq02JX3ldADSj1oD78mbV-fXfvu"
GEMINI_VIDEO_VECTORSTORE_PARENT_FOLDER_ID = "1iv58CUkl-HXkukRTdcfDF1RhG9goYMXn"

VECTORSTORE_FOLDER_NAME = "Google Drive Videos Vectorstore"

CHROMA_BACKUP_FOLDER_NAME = "chroma_video_embeddings_db"

CHROMA_COLLECTION_NAME = os.getenv(
    "GEMINI_VIDEO_CHROMA_COLLECTION_NAME",
    "gemini_video_visual_embeddings",
)

GEMINI_EMBEDDING_MODEL = os.getenv(
    "GEMINI_EMBEDDING_MODEL",
    "gemini-embedding-2",
)

# Must match the output dimensionality used when the vectors were created.
# If your creation code used 3072, keep 3072. If it used 768, set 768.
OUTPUT_DIMENSIONALITY = 3072

LOCAL_VECTORSTORE_PATH = os.getenv(
    "GEMINI_VIDEO_LOCAL_SEARCH_DB_PATH",
    os.path.join(tempfile.gettempdir(), "gemini_drive_video_vectorstore_search"),
)

DRIVE_FOLDER_MIME = "application/vnd.google-apps.folder"
QUERY_TYPE_OPTIONS = ["text", "image", "video"]
SEARCH_MODE_OPTIONS = ["Visual / semantic"]

_VECTORSTORE_CACHE: Dict[str, Tuple[str, Any]] = {}
_VECTORSTORE_LOCK = threading.Lock()
_GEMINI_CLIENT = None
_GEMINI_CLIENT_LOCK = threading.Lock()


# ============================================================
# Auth helpers
# ============================================================

def get_drive_instance() -> Optional[GoogleDrive]:
    """
    Return a PyDrive2 GoogleDrive instance.

    If Streamlit session_state already contains `drive`, that is reused.
    Otherwise this tries to initialize from GDRIVE_SA_B64 using your existing
    services.drive_service.login_with_service_account helper.
    """
    if "drive" in st.session_state:
        return st.session_state["drive"]

    if login_with_service_account is None:
        return None

    try:
        key_bytes = base64.b64decode(os.environ["GDRIVE_SA_B64"])
        sa_json = key_bytes.decode()
        gauth = login_with_service_account(json_str=sa_json)
        gauth.ServiceAuth()
        drive = GoogleDrive(gauth)
        st.session_state["drive"] = drive
        return drive
    except Exception as exc:
        print(f"Could not initialize Drive from environment: {exc}")
        return None


def get_gemini_client() -> genai.Client:
    """Load Gemini client once per process."""
    global _GEMINI_CLIENT

    if _GEMINI_CLIENT is not None:
        return _GEMINI_CLIENT

    with _GEMINI_CLIENT_LOCK:
        if _GEMINI_CLIENT is not None:
            return _GEMINI_CLIENT

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("Missing GOOGLE_API_KEY environment variable.")

        _GEMINI_CLIENT = genai.Client(api_key=api_key)
        return _GEMINI_CLIENT


# ============================================================
# PyDrive folder helpers
# ============================================================

def find_drive_folder(
    drive: GoogleDrive,
    parent_folder_id: str,
    folder_name: str,
) -> Optional[str]:
    """Find a folder by name inside a parent folder. Returns folder ID or None."""
    safe_name = folder_name.replace("'", "\\'")
    query = (
        f"title='{safe_name}' and "
        f"'{parent_folder_id}' in parents and "
        f"mimeType='{DRIVE_FOLDER_MIME}' and "
        f"trashed=false"
    )
    folders = drive.ListFile({"q": query}).GetList()
    if not folders:
        return None
    return folders[0]["id"]


def download_folder_from_drive(
    drive: GoogleDrive,
    folder_id: str,
    local_path: str,
) -> None:
    """Recursively download a Drive folder into local_path using PyDrive2."""
    os.makedirs(local_path, exist_ok=True)

    items = drive.ListFile({
        "q": f"'{folder_id}' in parents and trashed=false"
    }).GetList()

    for item in items:
        title = item.get("title") or item.get("name")
        if not title:
            continue

        item_id = item["id"]
        mime_type = item.get("mimeType")
        destination = os.path.join(local_path, title)

        if mime_type == DRIVE_FOLDER_MIME:
            download_folder_from_drive(drive, item_id, destination)
        else:
            item.GetContentFile(destination)


def resolve_chroma_db_path(local_path: str) -> str:
    """
    Return the folder containing chroma.sqlite3.

    Handles both layouts:
    - local_path/chroma.sqlite3
    - local_path/chroma_video_embeddings_db/chroma.sqlite3
    """
    direct = os.path.join(local_path, "chroma.sqlite3")
    if os.path.exists(direct):
        return local_path

    if not os.path.exists(local_path):
        return local_path

    for root, dirs, files in os.walk(local_path):
        if "chroma.sqlite3" in files:
            return root

    return local_path

def download_drive_file_bytes(drive: GoogleDrive, video_id: str) -> tuple[bytes, str]:
    """
    Download a Drive file by file ID.
    Returns (raw_bytes, file_extension).
    """
    f = drive.CreateFile({"id": video_id})
    f.FetchMetadata(fields="title,mimeType")
    title = f.get("title", "video.mp4")
    ext = os.path.splitext(title)[-1].lower() or ".mp4"

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp_path = tmp.name

    f.GetContentFile(tmp_path)

    with open(tmp_path, "rb") as fh:
        data = fh.read()

    os.remove(tmp_path)
    return data, ext


def trim_video_ffmpeg(
    video_bytes: bytes,
    start: float,
    end: float,
    suffix: str = ".mp4",
) -> bytes:
    """
    Trim video_bytes to [start, end] seconds using ffmpeg fast seek.
    Uses -c copy so there is no re-encoding — very fast.
    Requires ffmpeg on PATH.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as src:
        src.write(video_bytes)
        src_path = src.name

    out_path = src_path.replace(suffix, f"_seg{suffix}")

    try:
        duration = max(0.1, float(end) - float(start))
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-ss", str(float(start)),
                "-i", src_path,
                "-t", str(duration),
                "-c", "copy",
                out_path,
            ],
            check=True,
            capture_output=True,
        )
        with open(out_path, "rb") as fh:
            return fh.read()
    finally:
        os.remove(src_path)
        if os.path.exists(out_path):
            os.remove(out_path)

def load_gemini_video_chroma_collection(
    drive: GoogleDrive,
    parent_folder_id: str = GEMINI_VIDEO_VECTORSTORE_PARENT_FOLDER_ID,
    vectorstore_folder_name: str = VECTORSTORE_FOLDER_NAME,
    backup_folder_name: str = CHROMA_BACKUP_FOLDER_NAME,
    collection_name: str = CHROMA_COLLECTION_NAME,
    force_download: bool = False,
):
    """
    Download/load your Gemini video Chroma vectorstore from Drive.

    Drive layout expected:
    parent_folder_id/
      Google Drive Videos Vectorstore/
        chroma_video_embeddings_db/
          chroma.sqlite3
          <uuid-folder>/

    :param drive: Authenticated PyDrive2 GoogleDrive instance.
    :param parent_folder_id: Drive folder ID that contains the vectorstore folder.
    :param vectorstore_folder_name: Name of the outer folder created by your embedding pipeline.
    :param backup_folder_name: Name of the inner Chroma backup folder created by your embedding pipeline.
    :param collection_name: Name of the Chroma collection to load.
    :param force_download: If True, forces redownloading the vectorstore from Drive even if cached locally.
    :return: A Chroma collection object ready for querying.
    """
    cache_key = f"{parent_folder_id}:{vectorstore_folder_name}:{backup_folder_name}:{collection_name}"

    if not force_download and cache_key in _VECTORSTORE_CACHE:
        _, collection = _VECTORSTORE_CACHE[cache_key]
        return collection

    with _VECTORSTORE_LOCK:
        if not force_download and cache_key in _VECTORSTORE_CACHE:
            _, collection = _VECTORSTORE_CACHE[cache_key]
            return collection

        print(f"Looking inside parent folder: {parent_folder_id}")
        print(f"Looking for outer folder: {vectorstore_folder_name}")

        vectorstore_folder_id = find_drive_folder(
            drive=drive,
            parent_folder_id=parent_folder_id,
            folder_name=vectorstore_folder_name,
        )

        if not vectorstore_folder_id:
            raise FileNotFoundError(
                f"Could not find outer folder '{vectorstore_folder_name}' "
                f"inside parent folder ID {parent_folder_id}."
            )

        print(f"Found outer folder ID: {vectorstore_folder_id}")
        print(f"Looking for Chroma backup folder: {backup_folder_name}")

        chroma_folder_id = find_drive_folder(
            drive=drive,
            parent_folder_id=vectorstore_folder_id,
            folder_name=backup_folder_name,
        )

        if not chroma_folder_id:
            raise FileNotFoundError(
                f"Could not find inner Chroma folder '{backup_folder_name}' "
                f"inside '{vectorstore_folder_name}'."
            )

        print(f"Found Chroma backup folder ID: {chroma_folder_id}")
        if not chroma_folder_id:
            raise FileNotFoundError(
                f"Could not find Chroma backup folder '{backup_folder_name}' "
                f"inside '{vectorstore_folder_name}'."
            )

        local_root = os.path.join(LOCAL_VECTORSTORE_PATH, chroma_folder_id)
        db_path = resolve_chroma_db_path(local_root)
        db_file = os.path.join(db_path, "chroma.sqlite3")

        if force_download or not os.path.exists(db_file):
            shutil.rmtree(local_root, ignore_errors=True)
            os.makedirs(local_root, exist_ok=True)
            print(f"Downloading Gemini video vectorstore to: {local_root}")
            download_folder_from_drive(drive, chroma_folder_id, local_root)
            db_path = resolve_chroma_db_path(local_root)
            db_file = os.path.join(db_path, "chroma.sqlite3")

        if not os.path.exists(db_file):
            raise RuntimeError(
                f"Downloaded vectorstore but chroma.sqlite3 was not found. Checked: {db_file}"
            )

        client = chromadb.PersistentClient(path=db_path)
        collections = client.list_collections()
        collection_names = [c.name for c in collections]
        print(f"Available Chroma collections: {collection_names}")

        if collection_name not in collection_names:
            raise RuntimeError(
                f"Collection '{collection_name}' not found. Available: {collection_names}. "
                "Check that your search code uses the same collection name as your creation pipeline."
            )

        collection = client.get_collection(collection_name)
        print(f"Loaded collection '{collection_name}' with {collection.count()} item(s).")

        _VECTORSTORE_CACHE[cache_key] = (db_path, collection)
        return collection


# ============================================================
# Gemini query embedding helpers
# ============================================================

def get_text_query_embedding(query: str, filters: Optional[Dict[str, Any]] = None) -> List[float]:
    """Create a Gemini Embedding 2 vector for a text query, matching ingestion text syntax."""
    client = get_gemini_client()
    
    # Mirror the ingestion structure exactly using available query filters
    if filters:
        text_anchor = (
            f"Course: {filters.get('course_name', 'All')} | "
            f"Topic: {filters.get('topic_name', 'All')} | "
            f"Video: Search Reference | Semantic search query: {query}"
        )
    else:
        text_anchor = f"Semantic search query: {query}"

    result = client.models.embed_content(
        model=GEMINI_EMBEDDING_MODEL,
        contents=text_anchor,  # The formatted text anchor acts as the complete query
        config=types.EmbedContentConfig(
            output_dimensionality=OUTPUT_DIMENSIONALITY,
            task_type="RETRIEVAL_QUERY",  
        ),
    )
    return list(result.embeddings[0].values)


def get_image_query_embedding(image_bytes: bytes, mime_type: str = "image/png") -> List[float]:
    """Create a pure visual Gemini Embedding 2 vector for an image query."""
    client = get_gemini_client()
    
    result = client.models.embed_content(
        model=GEMINI_EMBEDDING_MODEL,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
        ],
        config=types.EmbedContentConfig(
            output_dimensionality=OUTPUT_DIMENSIONALITY,
            task_type="RETRIEVAL_QUERY",  
        ),
    )
    return list(result.embeddings[0].values)


def get_video_query_embedding(video_bytes: bytes, mime_type: str = "video/mp4") -> List[float]:
    """Create a pure visual Gemini Embedding 2 vector for a video clip query."""
    client = get_gemini_client()
    
    result = client.models.embed_content(
        model=GEMINI_EMBEDDING_MODEL,
        contents=[
            types.Part.from_bytes(data=video_bytes, mime_type=mime_type),
        ],
        config=types.EmbedContentConfig(
            output_dimensionality=OUTPUT_DIMENSIONALITY,
            task_type="RETRIEVAL_QUERY",  
        ),
    )
    return list(result.embeddings[0].values)

def trim_query_video_bytes(video_bytes: bytes, mime_type: str = "video/mp4", max_seconds: int = 30) -> bytes:
    suffix = ".mp4"
    if "webm" in mime_type:
        suffix = ".webm"
    elif "quicktime" in mime_type or "mov" in mime_type:
        suffix = ".mov"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as src:
        src.write(video_bytes)
        src_path = src.name

    out_path = src_path + "_trimmed.mp4"

    try:
        # Added scaling parameters: -vf scale=320:-2 reduces resolution footprint
        # -r 1 reduces the frame rate to 1 frame per second to avoid over-tokenization drops
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", src_path,
                "-t", str(max_seconds),
                "-vf", "scale=320:-2,fps=1",
                "-c:v", "libx264",
                "-b:v", "150k",
                "-an",  # Strip audio track entirely to minimize footprint
                out_path,
            ],
            check=True,
            capture_output=True,
        )

        with open(out_path, "rb") as f:
            return f.read()

    finally:
        if os.path.exists(src_path):
            os.remove(src_path)
        if os.path.exists(out_path):
            os.remove(out_path)


# ============================================================
# Search helpers
# ============================================================

def build_where_filter(filters: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Build a Chroma where filter from exact-match metadata filters."""
    if not filters:
        return None

    clauses = []
    for key, value in filters.items():
        if value is None:
            continue
        value = str(value).strip()
        if not value or value.lower() == "all":
            continue
        clauses.append({key: {"$eq": value}})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def metadata_matches_filters(metadata: Dict[str, Any], filters: Optional[Dict[str, Any]]) -> bool:
    """Case-insensitive fallback metadata filtering."""
    if not filters:
        return True

    for key, expected in filters.items():
        if expected is None or str(expected).strip() == "" or str(expected).lower() == "all":
            continue
        actual = metadata.get(key)
        if str(actual).strip().lower() != str(expected).strip().lower():
            return False
    return True


def get_metadata_text(metadata: Dict[str, Any]) -> str:
    """Return transcript-ish text from metadata if present."""
    for key in ("transcript", "transcript_text", "text", "text_0", "caption", "description"):
        value = metadata.get(key)
        if value:
            return str(value)
    return ""


def chroma_results_to_list(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert Chroma query output into list of result dictionaries."""
    out: List[Dict[str, Any]] = []

    ids = (results.get("ids") or [[]])[0]
    metadatas = (results.get("metadatas") or [[]])[0]
    distances = (results.get("distances") or [[]])[0]
    documents = (results.get("documents") or [[]])[0]

    for i, metadata in enumerate(metadatas):
        metadata = dict(metadata or {})
        item = dict(metadata)
        item["id"] = ids[i] if i < len(ids) else None
        item["distance"] = distances[i] if i < len(distances) else None
        item["document"] = documents[i] if i < len(documents) else ""

        item.setdefault("video_title", item.get("video_name") or item.get("title") or "Unknown Video")
        item.setdefault("title", item.get("video_title"))
        item.setdefault("video_url", item.get("video_link") or item.get("video_url") or "")
        item.setdefault("transcript", get_metadata_text(item))

        out.append(item)

    return out


def dedupe_results_by_segment(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deduplicate by video ID + segment timestamp."""
    seen = set()
    out = []

    for item in results:
        key = (
            item.get("video_id"),
            item.get("segment_index"),
            item.get("start_time"),
            item.get("end_time"),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(item)

    return out


def apply_diversity_by_video(results: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
    """Avoid returning too many segments from one video."""
    if not results or k <= 0:
        return []

    max_per_video = max(1, math.ceil(k / 4))
    count_by_video: Dict[str, int] = {}
    out = []

    for item in results:
        if len(out) >= k:
            break

        video_id = str(item.get("video_id") or item.get("video_link") or item.get("video_name") or "")
        current = count_by_video.get(video_id, 0)
        if current < max_per_video:
            count_by_video[video_id] = current + 1
            out.append(item)

    return out


def search_gemini_drive_video_embeddings(
    drive: GoogleDrive,
    query_type: str,
    query: Union[str, bytes],
    k: int = 5,
    filters: Optional[Dict[str, Any]] = None,
    # search_mode: str = "Visual / semantic",
    parent_folder_id: str = GEMINI_VIDEO_VECTORSTORE_PARENT_FOLDER_ID,
    vectorstore_folder_name: str = VECTORSTORE_FOLDER_NAME,
    backup_folder_name: str = CHROMA_BACKUP_FOLDER_NAME,
    collection_name: str = CHROMA_COLLECTION_NAME,
    force_download: bool = False,
    query_mime_type: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Search your Gemini Embedding 2 video vectorstore.

    query_type:
    - "text": semantic text-to-video search.
    - "image": image-to-video visual search.
    - "video": video-clip-to-video visual search.

    search_mode:
    - "Visual / semantic": vector search only.
    """
    query_type = query_type.lower().strip()
    if query_type not in QUERY_TYPE_OPTIONS:
        raise ValueError(f"query_type must be one of {QUERY_TYPE_OPTIONS}")

    if query_type == "text" and not str(query).strip():
        return []
    if query_type in ("image", "video") and not query:
        return []

    collection = load_gemini_video_chroma_collection(
        drive=drive,
        parent_folder_id=parent_folder_id,
        vectorstore_folder_name=vectorstore_folder_name,
        backup_folder_name=backup_folder_name,
        collection_name=collection_name,
        force_download=force_download,
    )

    collection_count = int(collection.count())
    if collection_count == 0:
        return []

    where_filter = build_where_filter(filters)
    fetch_k = min(collection_count, max(k * 10, 200))

    # Vector search.
    if query_type == "text":
        query_embedding = get_text_query_embedding(
            query=str(query),
            filters=filters,
        )
    elif query_type == "image":
        query_embedding = get_image_query_embedding(
            image_bytes=query,  # type: ignore[arg-type]
            mime_type=query_mime_type or "image/png",
        )
    else:
        trimmed_query_video = trim_query_video_bytes(
            video_bytes=query,
            mime_type=query_mime_type or "video/mp4",
            max_seconds=30,
        )

        query_embedding = get_video_query_embedding(
            video_bytes=trimmed_query_video,
            mime_type="video/mp4",
        )


    query_kwargs = {
        "query_embeddings": [query_embedding],
        "n_results": fetch_k,
        "include": ["metadatas", "distances", "documents"],
    }
    if where_filter:
        query_kwargs["where"] = where_filter

    results = collection.query(**query_kwargs)
    raw = dedupe_results_by_segment(chroma_results_to_list(results))

    return apply_diversity_by_video(raw, k)


# ============================================================
# Drive video display helpers
# ============================================================

def build_drive_preview_url(video_link: str = "", video_id: str = "") -> str:
    """Build a Google Drive preview URL from a Drive file ID or link."""
    if video_id and "http" not in str(video_id):
        return f"https://drive.google.com/file/d/{video_id}/preview"

    link = video_link or ""
    if "/file/d/" in link:
        try:
            file_id = link.split("/file/d/")[1].split("/")[0]
            return f"https://drive.google.com/file/d/{file_id}/preview"
        except Exception:
            return link

    return link


def render_drive_video_result(
    item: Dict[str, Any],
    idx: int,
    drive: Optional[GoogleDrive] = None,
) -> None:
    """Render one Drive video result — downloads and trims to the segment."""
    title = item.get("video_name") or item.get("title") or f"Video {idx}"
    video_link = item.get("video_link") or item.get("video_url") or ""
    video_id = str(item.get("video_id") or "")
    start_time = float(item.get("start_time") or 0)
    end_time = item.get("end_time")

    st.markdown(f"### {idx}. {title}")
    st.caption(
        f"Course: {item.get('course_name', 'Unknown')} | "
        f"Topic: {item.get('topic_name', 'Unknown')} | "
        # f"Segment: {start_time}s – {end_time or 'end'}"
    )
    # if item.get("distance") is not None:
    #     try:
    #         st.caption(f"Distance: {float(item.get('distance')):.4f}")
    #     except Exception:
    #         pass

    # --- Video display ---
    can_trim = (
        drive is not None
        and video_id
        and "http" not in video_id
        and end_time is not None
    )

    if can_trim:
        cache_key = f"seg__{video_id}__{start_time}__{end_time}"

        if cache_key not in st.session_state:
            with st.spinner(f"Loading segment {start_time}s–{end_time}s for '{title}'..."):
                try:
                    raw_bytes, ext = download_drive_file_bytes(drive, video_id)
                    clip_bytes = trim_video_ffmpeg(raw_bytes, start_time, float(end_time), suffix=ext)
                    st.session_state[cache_key] = clip_bytes
                except subprocess.CalledProcessError as e:
                    err = e.stderr.decode(errors="replace") if e.stderr else str(e)
                    st.session_state[cache_key] = None
                    st.error(f"ffmpeg trim failed: {err}")
                except Exception as e:
                    st.session_state[cache_key] = None
                    st.error(f"Could not load segment: {e}")

        clip_bytes = st.session_state.get(cache_key)
        if clip_bytes:
            st.video(clip_bytes)
            if video_link:
                st.markdown(f"[Open full video in Drive]({video_link})")
        else:
            # Fallback if trim failed
            if video_link:
                st.markdown(f"[Open video in Drive]({video_link})")

    else:
        # No drive instance or no video_id — fall back to link only
        if video_link:
            st.markdown(f"[Open video in Drive]({video_link})")
        else:
            st.info("No video preview available.")

    st.markdown("---")


# ============================================================
# Streamlit UI
# ============================================================

def render_gemini_drive_video_search_tab(
    drive: Optional[GoogleDrive] = None,
    gc=None,
    parent_folder_id: str = GEMINI_VIDEO_VECTORSTORE_PARENT_FOLDER_ID,
    vectorstore_folder_name: str = VECTORSTORE_FOLDER_NAME,
    backup_folder_name: str = CHROMA_BACKUP_FOLDER_NAME,
) -> None:
    """
    Render a Streamlit UI for:
    1. Searching your Drive video vectorstore
    2. Creating/updating the vectorstore from a Google Sheet URL
    """

    if drive is None:
        drive = get_drive_instance()

    if drive is None:
        st.error("Google Drive is not available.")
        return

    # st.caption(
    #     "Search your Google Drive videos, or create/update the Gemini Embedding 2 vectorstore from a Google Sheet."
    # )

    task = st.selectbox(
    "Choose action",
    options=[
        "Archiving Videos",
        "Create / Update Vectorstore",
        "Searching Videos",
    ],
    key="gemini_drive_video_task",
    )

    if task == "Archiving Videos":
        render_archive_tab(drive_service=drive, sheets_client=gc)
        return

    # ============================================================
    # CREATE / UPDATE VECTORSTORE
    # ============================================================
    if task == "Create / Update Vectorstore":
        st.subheader("Create / Update Google Drive Video Vectorstore")

        sheet_url = st.text_input(
            "Google Sheet URL",
            placeholder="Paste the spreadsheet URL containing your Drive video rows",
            key="gemini_vectorstore_sheet_url",
        )

        st.info(
            "The sheet must contain the required columns: Video ID, Video Name, "
            "Course Name, Topic Name, MimeType, Duration, File Size, Video Link, Stock/Non Stock."
        )

        if st.button("Run Create / Update", type="primary", key="run_gemini_vectorstore_update"):
            if not sheet_url.strip():
                st.warning("Please enter a Google Sheet URL.")
                return

            started = time.perf_counter()

            try:
                with st.spinner("Creating/updating Google Drive video vectorstore. This may take a while..."):
                    gemini_client = get_gemini_client()
    
                    if gc is None:
                        if "gc" in st.session_state:
                            gc = st.session_state["gc"]
                        else:
                            st.error(
                                "Google Sheets client `gc` is not available. "
                                "Pass gc into render_gemini_drive_video_search_tab(drive, gc)."
                            )
                            return

                    local_path, stats = create_video_embeddings(
                        drive=drive,
                        gc=gc,
                        gemini_client=gemini_client,
                        sheet_url=sheet_url.strip(),
                        parent_folder_id=parent_folder_id,
                    )

                # Clear cached vectorstore so search reloads the updated Drive backup
                _VECTORSTORE_CACHE.clear()

                track_tool_action(
                    "Google Drive Video Search",
                    "create_update_vectorstore",
                    run_mode="tool",
                    course_name="",
                    sheet_link=sheet_url.strip(),
                    duration_seconds=time.perf_counter() - started,
                )

                st.success("Vectorstore created/updated successfully.")
                st.json(stats)

            except Exception as exc:
                track_tool_action(
                    "Google Drive Video Search",
                    "create_update_vectorstore",
                    run_mode="tool",
                    course_name="",
                    sheet_link=sheet_url.strip(),
                    duration_seconds=time.perf_counter() - started,
                    error_message=str(exc)[:500],
                )
                st.error(f"Vectorstore creation/update failed: {exc}")
                st.text(traceback.format_exc())

        return

    # ============================================================
    # SEARCH VIDEOS
    # ============================================================

    st.subheader("Search Google Drive Course Videos")

    query_type = st.radio(
        "Query type",
        options=QUERY_TYPE_OPTIONS,
        horizontal=True,
        format_func=lambda x: {"text": "Text", "image": "Image", "video": "Video clip"}[x],
        key="gemini_drive_video_query_type",
    )
    query_value: Union[str, bytes, None] = None
    query_mime_type = None

    if query_type == "text":
        query_value = st.text_input(
            "Search query",
            placeholder="e.g. compressor wiring, thermostat installation, blower motor teardown",
            key="gemini_drive_video_text_query",
        )

    elif query_type == "image":
        uploaded = st.file_uploader(
            "Upload an image to search by visual similarity",
            type=["png", "jpg", "jpeg", "webp"],
            key="gemini_drive_video_image_upload",
        )
        if uploaded:
            query_value = uploaded.read()
            query_mime_type = uploaded.type or "image/png"
            st.image(query_value, caption="Query image", use_container_width=True)

    else:
        uploaded = st.file_uploader(
            "Upload a short video clip to search by visual similarity",
            type=["mp4", "mov", "webm"],
            key="gemini_drive_video_clip_upload",
        )
        if uploaded:
            query_value = uploaded.read()
            query_mime_type = uploaded.type or "video/mp4"
            st.video(query_value)

    num_results = st.number_input(
        "Number of results",
        min_value=1,
        max_value=50,
        value=5,
        step=1,
        key="gemini_drive_video_num_results",
    )

    with st.expander("Optional filters"):
        course_filter = st.text_input("Course name", placeholder="All courses", key="gemini_course_filter")
        topic_filter = st.text_input("Topic name", placeholder="All topics", key="gemini_topic_filter")
        stock_filter = st.selectbox("Stock/Non Stock", ["All", "Stock", "Non Stock"], key="gemini_stock_filter")
    

    filters: Dict[str, Any] = {}
    if course_filter.strip():
        filters["course_name"] = course_filter.strip()
    if topic_filter.strip():
        filters["topic_name"] = topic_filter.strip()
    if stock_filter != "All":
        filters["stock"] = stock_filter

    if st.button("Search Google Drive Videos", type="primary", key="gemini_drive_video_search_btn"):
        if query_value is None or (query_type == "text" and not str(query_value).strip()):
            st.warning("Please provide a search query or upload a file.")
        else:
            started = time.perf_counter()
            try:
                with st.spinner("Searching relevant videos..."):
                    results = search_gemini_drive_video_embeddings(
                        drive=drive,
                        query_type=query_type,
                        query=query_value,
                        k=int(num_results),
                        filters=filters,
                        parent_folder_id=parent_folder_id,
                        vectorstore_folder_name=vectorstore_folder_name,
                        backup_folder_name=backup_folder_name,
                        query_mime_type=query_mime_type,
                    )

                track_tool_action(
                    "Google Drive Video Search",
                    "search_videos",
                    run_mode="tool",
                    course_name=course_filter or "",
                    sheet_link="",
                    duration_seconds=time.perf_counter() - started,
                )

            except Exception as exc:
                track_tool_action(
                    "Google Drive Video Search",
                    "search_videos",
                    run_mode="tool",
                    course_name=course_filter or "",
                    sheet_link="",
                    duration_seconds=time.perf_counter() - started,
                    error_message=str(exc)[:500],
                )
                st.error(f"Search failed: {exc}")
                st.text(traceback.format_exc())
                results = []

            st.session_state["gemini_drive_video_results"] = results

    results = st.session_state.get("gemini_drive_video_results", [])
    if results:
        st.subheader(f"Top {len(results)} result(s)")
        for idx, item in enumerate(results, start=1):
            render_drive_video_result(item, idx, drive=drive)
    elif "gemini_drive_video_results" in st.session_state:
        st.warning("No matching videos found.")