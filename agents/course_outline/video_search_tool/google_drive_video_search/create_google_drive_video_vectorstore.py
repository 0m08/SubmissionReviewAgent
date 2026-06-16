"""
Video Embeddings Creation Pipeline
Gemini Embedding 2 + ChromaDB + PyDrive + Google Sheets

Run in VS Code:
    python video_embeddings_gemini_pydrive.py

Before running:
    1. Install ffmpeg and make sure `ffmpeg` is on PATH.
    2. Put your Gemini API key in an environment variable:
           Windows PowerShell:
               setx GOOGLE_API_KEY "your-key"
           macOS/Linux:
               export GOOGLE_API_KEY="your-key"
    3. Create Google OAuth credentials for a Desktop app and save the file as:
           client_secrets.json
       in the same folder as this script.
    4. Install dependencies:
           pip install google-genai chromadb pandas numpy tqdm moviepy gspread PyDrive oauth2client
"""

import os
import json
import time
import shutil
import random
import tempfile
import threading
import subprocess
from pathlib import Path
from datetime import datetime
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm
from moviepy.editor import VideoFileClip

import gspread
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

from google import genai
from google.genai import types
from chromadb import PersistentClient


# ============================================================
# CONFIGURATION
# ============================================================

SHEET_URL = "https://docs.google.com/spreadsheets/d/1lV9_52Qicf6yHzn1NCbJ9M9lc5DnT29ODSOK5tXZHfk/edit?usp=sharing"

PARENT_FOLDER_ID = "1iv58CUkl-HXkukRTdcfDF1RhG9goYMXn"
VECTORSTORE_FOLDER_NAME = "Google Drive Videos Vectorstore"
CHROMA_BACKUP_FOLDER_NAME = "chroma_video_embeddings_db"

LOCAL_VECTORSTORE_PATH = os.getenv(
    "GEMINI_VIDEO_LOCAL_SEARCH_DB_PATH",
    os.path.join(tempfile.gettempdir(), "gemini_drive_video_vectorstore_search"),
)


COL_VIDEO_ID = "Video ID"
COL_VIDEO_NAME = "Video Name"
COL_COURSE_NAME = "Course Name"
COL_TOPIC_NAME = "Topic Name"
COL_MIME_TYPE = "MimeType"
COL_DURATION = "Duration"
COL_FILE_SIZE = "File Size"
COL_VIDEO_LINK = "Video Link"
COL_STOCK = "Stock/Non Stock"

REQUIRED_COLUMNS = [
    COL_VIDEO_ID,
    COL_VIDEO_NAME,
    COL_COURSE_NAME,
    COL_TOPIC_NAME,
    COL_MIME_TYPE,
    COL_DURATION,
    COL_FILE_SIZE,
    COL_VIDEO_LINK,
    COL_STOCK,
]

TRACKING_COLUMNS = [
    "vectorized",
    "embedding_ts",
    "segments_processed",
    "estimated_cost_usd",
    "error_message",
    "embedding_model",
    "embedding_dimensionality",
]

CHUNK_LENGTH_SEC = 60
CHUNK_OVERLAP_SEC = 10
MAX_WORKERS = 5
MAX_VIDEO_BYTES = 20_000_000

EMBEDDING_MAX_RETRIES = 6
EMBEDDING_BASE_BACKOFF_SEC = 2.0
EMBEDDING_MAX_BACKOFF_SEC = 60.0
EMBEDDING_RPM_LIMIT = 10
EMBEDDING_MAX_INFLIGHT = 2

GEMINI_MODEL_NAME = "gemini-embedding-2"
OUTPUT_DIMENSIONALITY = 3072
GEMINI_VIDEO_MAX_SEC = 120
CHROMA_COLLECTION_NAME = "gemini_video_visual_embeddings"

FRAMES_PER_CHUNK_MAX = 32
PRICE_PER_FRAME_STANDARD = 0.00079
PRICE_PER_FRAME_BATCH = 0.000395
USE_BATCH_PRICING_FOR_ESTIMATE = False

# Whether to backup ChromaDB to Drive after every video.
# True is safer but slower. False backs up once at the end.
BACKUP_AFTER_EACH_VIDEO = False


# ============================================================
# AUTH
# ============================================================

def initialize_clients():
    """
    Authenticates PyDrive, gspread, and Gemini.

    Requires client_secrets.json for Google OAuth.
    Requires GOOGLE_API_KEY environment variable.
    """
    google_auth = GoogleAuth()

    # Saves/loads OAuth token locally so you do not need to log in every run.
    google_auth.LoadCredentialsFile("mycreds.txt")

    if google_auth.credentials is None:
        google_auth.LocalWebserverAuth()
    elif google_auth.access_token_expired:
        google_auth.Refresh()
    else:
        google_auth.Authorize()

    google_auth.SaveCredentialsFile("mycreds.txt")

    drive = GoogleDrive(google_auth)

    try:
        gc = gspread.authorize(google_auth.credentials)
    except Exception as exc:
        raise RuntimeError(
            "Could not authorize gspread using PyDrive credentials. "
            "Make sure your OAuth app has Drive/Sheets scopes and try deleting mycreds.txt, "
            "then rerun to authorize again."
        ) from exc

    google_api_key = os.environ.get("GOOGLE_API_KEY")
    if not google_api_key:
        raise RuntimeError(
            "Missing GOOGLE_API_KEY. Set it before running this script."
        )

    gemini_client = genai.Client(api_key=google_api_key)

    print("✅ PyDrive ready.")
    print("✅ Google Sheets ready.")
    print("✅ Gemini client ready.")

    return drive, gc, gemini_client


# ============================================================
# UTILITY HELPERS
# ============================================================

def calculate_cost(num_chunks: int, use_batch: bool = USE_BATCH_PRICING_FOR_ESTIMATE) -> float:
    price_per_frame = PRICE_PER_FRAME_BATCH if use_batch else PRICE_PER_FRAME_STANDARD
    total_frames = num_chunks * FRAMES_PER_CHUNK_MAX
    return round(total_frames * price_per_frame, 6)


def cost_breakdown(num_chunks: int) -> dict:
    standard = calculate_cost(num_chunks, use_batch=False)
    batch = calculate_cost(num_chunks, use_batch=True)
    tier = "batch" if USE_BATCH_PRICING_FOR_ESTIMATE else "standard"
    used = batch if USE_BATCH_PRICING_FOR_ESTIMATE else standard
    return {
        "chunks_embedded": num_chunks,
        "frames_per_chunk_max": FRAMES_PER_CHUNK_MAX,
        "total_frames_estimated": num_chunks * FRAMES_PER_CHUNK_MAX,
        "estimated_cost_standard_usd": standard,
        "estimated_cost_batch_usd": batch,
        "pricing_tier_used": tier,
        "estimated_cost_usd": used,
    }


class RequestRateLimiter:
    """Thread-safe fixed-window rate limiter."""

    def __init__(self, max_calls: int, period_seconds: float):
        self.max_calls = max(1, int(max_calls))
        self.period_seconds = float(period_seconds)
        self._timestamps = deque()
        self._lock = threading.Lock()

    def acquire(self):
        while True:
            now = time.monotonic()
            with self._lock:
                while self._timestamps and (now - self._timestamps[0]) >= self.period_seconds:
                    self._timestamps.popleft()

                if len(self._timestamps) < self.max_calls:
                    self._timestamps.append(now)
                    return

                oldest = self._timestamps[0]
                wait_for = max(0.01, self.period_seconds - (now - oldest))

            time.sleep(wait_for)


def is_quota_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    signals = [
        "429",
        "quota exceeded",
        "resource_exhausted",
        "rate limit",
        "too many requests",
        "ratelimitexceeded",
        "requests per minute",
        "quota",
    ]
    return any(s in msg for s in signals)


# ============================================================
# SHEETS HELPERS
# ============================================================

def get_all_tabs(gc, sheet_url: str):
    spreadsheet = gc.open_by_url(sheet_url)
    return [(ws, ws.title) for ws in spreadsheet.worksheets()]


def get_tab_dataframe(worksheet) -> pd.DataFrame:
    records = worksheet.get_all_records()
    return pd.DataFrame(records)


def save_to_sheet(worksheet, df: pd.DataFrame, wait_seconds: float = 2.0):
    worksheet.clear()
    time.sleep(wait_seconds)
    worksheet.update([df.columns.tolist()] + df.fillna("").values.tolist())
    time.sleep(wait_seconds)


def clear_tracking_columns_all_tabs(gc, sheet_url=SHEET_URL, wait_seconds: float = 5.0):
    """
    Clears tracking column values across all tabs.
    Keeps the headers.
    """
    spreadsheet = gc.open_by_url(sheet_url)
    worksheets = spreadsheet.worksheets()

    for i, worksheet in enumerate(worksheets, start=1):
        print(f"[{i}/{len(worksheets)}] Clearing tab: {worksheet.title}")
        df = get_tab_dataframe(worksheet)

        if df.empty:
            print("  Skipped empty tab.")
            time.sleep(wait_seconds)
            continue

        cleared = []
        for col in TRACKING_COLUMNS:
            if col in df.columns:
                df[col] = ""
                cleared.append(col)

        if not cleared:
            print("  No tracking columns found.")
            time.sleep(wait_seconds)
            continue

        save_to_sheet(worksheet, df, wait_seconds=wait_seconds)
        print(f"  Cleared: {cleared}")


# ============================================================
# PYDRIVE HELPERS
# ============================================================

def _escape_drive_query_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'")


def create_or_get_drive_folder(drive, parent_folder_id: str, folder_name: str) -> str:
    safe_name = _escape_drive_query_value(folder_name)
    query = (
        f"title='{safe_name}' and "
        f"'{parent_folder_id}' in parents and "
        f"mimeType='application/vnd.google-apps.folder' and "
        f"trashed=false"
    )

    files = drive.ListFile({"q": query}).GetList()
    if files:
        folder_id = files[0]["id"]
        print(f"  📁 Found Drive folder '{folder_name}': {folder_id}")
        return folder_id

    folder = drive.CreateFile({
        "title": folder_name,
        "parents": [{"id": parent_folder_id}],
        "mimeType": "application/vnd.google-apps.folder",
    })
    folder.Upload()
    print(f"  📁 Created Drive folder '{folder_name}': {folder['id']}")
    return folder["id"]


def find_drive_folder(drive, parent_folder_id: str, folder_name: str):
    safe_name = _escape_drive_query_value(folder_name)
    query = (
        f"title='{safe_name}' and "
        f"'{parent_folder_id}' in parents and "
        f"mimeType='application/vnd.google-apps.folder' and "
        f"trashed=false"
    )
    files = drive.ListFile({"q": query}).GetList()
    if files:
        return files[0]["id"]
    return None


def delete_drive_file_or_folder(drive, file_id: str):
    item = drive.CreateFile({"id": file_id})
    item.Delete()


def upload_folder_to_drive(drive, local_folder_path: str, parent_folder_id: str):
    """
    Recursively upload a local folder's contents into an existing Drive folder.

    Important for ChromaDB:
    uploads both chroma.sqlite3 and the UUID subfolder.
    """
    local_folder_path = str(local_folder_path)

    for item_name in os.listdir(local_folder_path):
        item_path = os.path.join(local_folder_path, item_name)

        if os.path.isdir(item_path):
            folder = drive.CreateFile({
                "title": item_name,
                "parents": [{"id": parent_folder_id}],
                "mimeType": "application/vnd.google-apps.folder",
            })
            folder.Upload()
            print(f"    📁 Created Drive subfolder: {item_name}")
            upload_folder_to_drive(drive, item_path, folder["id"])

        elif os.path.isfile(item_path):
            file_obj = drive.CreateFile({
                "title": item_name,
                "parents": [{"id": parent_folder_id}],
            })
            file_obj.SetContentFile(item_path)
            file_obj.Upload()
            print(f"    ⬆️ Uploaded: {item_name}")


def download_folder_from_drive(drive, folder_id: str, local_path: str):
    """
    Recursively download a Drive folder to local disk.
    Restores both chroma.sqlite3 and Chroma UUID folders.
    """
    os.makedirs(local_path, exist_ok=True)
    items = drive.ListFile({"q": f"'{folder_id}' in parents and trashed=false"}).GetList()

    for item in items:
        title = item["title"]
        mime_type = item.get("mimeType")
        item_id = item["id"]
        destination = os.path.join(local_path, title)

        if mime_type == "application/vnd.google-apps.folder":
            print(f"    📁 Downloading Drive subfolder: {title}")
            download_folder_from_drive(drive, item_id, destination)
        else:
            item.GetContentFile(destination)
            print(f"    ⬇️ Downloaded: {title}")


def download_from_drive(drive, file_id: str, output_path: str) -> str:
    """
    Download a video from Google Drive using PyDrive.
    The file_id comes from the Sheet's 'Video ID' column.
    """
    file_obj = drive.CreateFile({"id": file_id})
    file_obj.GetContentFile(output_path)
    print(f"    📥 Downloaded video to {output_path}")
    return output_path


def backup_vectorstore_to_drive(drive, local_path: str, vectorstore_folder_id: str):
    """
    Replaces old chroma_video_embeddings_db backup with a fresh recursive upload.
    """
    print("  📤 Backing up vectorstore to Drive...")

    old_id = find_drive_folder(drive, vectorstore_folder_id, CHROMA_BACKUP_FOLDER_NAME)
    if old_id:
        delete_drive_file_or_folder(drive, old_id)
        print("  🗑️ Removed old Drive backup.")

    new_folder_id = create_or_get_drive_folder(
        drive,
        vectorstore_folder_id,
        CHROMA_BACKUP_FOLDER_NAME,
    )

    upload_folder_to_drive(drive, local_path, new_folder_id)
    print("  ✅ Drive backup complete.")


# ============================================================
# VIDEO + EMBEDDING HELPERS
# ============================================================

def segment_video(
    video_path: str,
    output_dir: str,
    chunk_length: int = CHUNK_LENGTH_SEC,
    overlap: int = CHUNK_OVERLAP_SEC,
):
    """
    Split video into chunks. Uses overlap if overlap > 0.

    The original video is not changed. Chunks are temporary copies used only
    for embedding.
    """
    if chunk_length > GEMINI_VIDEO_MAX_SEC:
        raise ValueError(f"chunk_length must be <= {GEMINI_VIDEO_MAX_SEC} seconds.")

    if overlap >= chunk_length:
        raise ValueError("CHUNK_OVERLAP_SEC must be smaller than CHUNK_LENGTH_SEC.")

    os.makedirs(output_dir, exist_ok=True)

    clip = VideoFileClip(video_path)
    duration = int(clip.duration)
    clip.close()

    segments = []
    start = 0
    step = chunk_length - overlap if overlap > 0 else chunk_length

    while start < duration:
        end = min(start + chunk_length, duration)
        segment_file = os.path.join(output_dir, f"segment_{start}_{end}.mp4")

        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            str(start),
            "-to",
            str(end),
            "-i",
            video_path,
            "-c",
            "copy",
            segment_file,
        ]

        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )

        if result.returncode != 0 or not os.path.exists(segment_file):
            raise RuntimeError(f"ffmpeg failed for segment {start}-{end}: {result.stderr[:500]}")

        segments.append((segment_file, start, end))

        if end >= duration:
            break
        start += step

    print(f"  ✂️ Created {len(segments)} segment(s) in {output_dir}")
    return segments


def generate_video_embedding(
    gemini_client,
    video_path: str,
    chunk_start_sec: float,
    chunk_end_sec: float,
    rate_limiter=None,
    request_semaphore=None,
):
    """
    Creates one Gemini Embedding 2 vector for one video chunk.

    Note: this embeds visual/video content. It does not preserve or store the
    original video file in Chroma, and it does not make audio speech searchable.
    The original video remains in Drive and is referenced by video_id/video_link.
    """
    with open(video_path, "rb") as f:
        video_bytes = f.read()

    if len(video_bytes) > MAX_VIDEO_BYTES:
        compressed_path = video_path.replace(".mp4", "_compressed.mp4")
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            video_path,
            "-vf",
            "scale=640:-1",
            "-b:v",
            "600k",
            "-b:a",
            "64k",
            compressed_path,
        ]
        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )

        if result.returncode != 0 or not os.path.exists(compressed_path):
            raise RuntimeError(f"ffmpeg compression failed: {result.stderr[:500]}")

        with open(compressed_path, "rb") as f:
            video_bytes = f.read()
        os.remove(compressed_path)
        print(f"    🗜️ Compressed chunk to {len(video_bytes) / 1e6:.1f} MB")

    text_prefix = (
        f"title: video segment | "
        f"text: segment from {int(chunk_start_sec)}s to {int(chunk_end_sec)}s"
    )

    last_error = None
    for attempt in range(EMBEDDING_MAX_RETRIES + 1):
        try:
            if rate_limiter:
                rate_limiter.acquire()

            def _call():
                return gemini_client.models.embed_content(
                    model=GEMINI_MODEL_NAME,
                    contents=[
                        text_prefix,
                        types.Part.from_bytes(
                            data=video_bytes,
                            mime_type="video/mp4",
                        ),
                    ],
                    config=types.EmbedContentConfig(
                        output_dimensionality=OUTPUT_DIMENSIONALITY,
                        task_type="RETRIEVAL_DOCUMENT",
                    ),
                )
    
            if request_semaphore:
                with request_semaphore:
                    result = _call()
            else:
                result = _call()

            vector = list(result.embeddings[0].values)
            if not vector:
                raise RuntimeError("Gemini returned an empty embedding vector.")

            return {
                "vector": vector,
                "start_time": chunk_start_sec,
                "end_time": chunk_end_sec,
            }

        except Exception as exc:
            last_error = exc

            if not is_quota_error(exc) or attempt >= EMBEDDING_MAX_RETRIES:
                raise

            backoff = min(
                EMBEDDING_MAX_BACKOFF_SEC,
                EMBEDDING_BASE_BACKOFF_SEC * (2 ** attempt),
            )
            jitter = random.uniform(0, 0.5 * EMBEDDING_BASE_BACKOFF_SEC)
            sleep_for = backoff + jitter

            print(
                f"    ⏳ Quota hit "
                f"(attempt {attempt + 1}/{EMBEDDING_MAX_RETRIES + 1}). "
                f"Retrying in {sleep_for:.1f}s..."
            )
            time.sleep(sleep_for)

    raise last_error


# ============================================================
# CHROMA HELPERS
# ============================================================

def get_or_create_collection(local_path: str, collection_name: str = CHROMA_COLLECTION_NAME):
    os.makedirs(local_path, exist_ok=True)
    client = PersistentClient(path=local_path)

    try:
        collection = client.get_collection(collection_name)
        print(f"  🧩 Loaded Chroma collection '{collection_name}' ({collection.count()} items).")
    except Exception:
        collection = client.create_collection(collection_name)
        print(f"  🆕 Created Chroma collection '{collection_name}'.")

    return client, collection


def upsert_embeddings(collection, embeddings_data: list):
    if not embeddings_data:
        raise RuntimeError("No embeddings were created, so nothing was saved to ChromaDB.")

    ids = []
    embeddings = []
    metadatas = []

    for item in embeddings_data:
        ids.append(f"{item['video_id']}_{item['segment_index']}")
        embeddings.append(item["embedding"])
        metadatas.append({
            "video_id": str(item["video_id"]),
            "video_name": str(item["video_name"]),
            "course_name": str(item["course_name"]),
            "topic_name": str(item["topic_name"]),
            "tab_name": str(item["tab_name"]),
            "mime_type": str(item["mime_type"]),
            "duration": str(item["duration"]),
            "file_size": str(item["file_size"]),
            "video_link": str(item["video_link"]),
            "stock": str(item["stock"]),
            "segment_index": int(item["segment_index"]),
            "start_time": float(item["start_time"]),
            "end_time": float(item["end_time"]),
        })

    collection.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas)

    print(f"  💾 Upserted {len(ids)} embedding(s) into ChromaDB.")
    print(f"  📊 Collection count is now: {collection.count()}")


# ============================================================
# MAIN PIPELINE
# ============================================================

def create_video_embeddings(
    drive,
    gc,
    gemini_client,
    sheet_url: str = SHEET_URL,
    parent_folder_id: str = PARENT_FOLDER_ID,
    chunk_length: int = CHUNK_LENGTH_SEC,
    max_workers: int = MAX_WORKERS,
):
    print("=" * 70)
    print("🚀 VIDEO EMBEDDINGS CREATION PIPELINE")
    print("   Gemini Embedding 2 + ChromaDB + PyDrive")
    print("=" * 70)

    pipeline_start = time.time()

    print("\n📁 Step 1: Setting up Drive folders...")
    vectorstore_folder_id = create_or_get_drive_folder(
        drive,
        parent_folder_id,
        VECTORSTORE_FOLDER_NAME,
    )

    print(f"\n💾 Step 2: Local vectorstore path: {LOCAL_VECTORSTORE_PATH}")
    os.makedirs(LOCAL_VECTORSTORE_PATH, exist_ok=True)

    print("\n🔍 Step 3: Checking for existing vectorstore backup...")
    local_db_file = os.path.join(LOCAL_VECTORSTORE_PATH, "chroma.sqlite3")

    if os.path.exists(local_db_file):
        print("  ✅ Local ChromaDB found.")
    else:
        chroma_folder_id = find_drive_folder(
            drive,
            vectorstore_folder_id,
            CHROMA_BACKUP_FOLDER_NAME,
        )
        if chroma_folder_id:
            print("  📥 Downloading existing ChromaDB backup from Drive...")
            download_folder_from_drive(drive, chroma_folder_id, LOCAL_VECTORSTORE_PATH)
            print("  ✅ Download complete.")
        else:
            print("  📝 No existing ChromaDB backup found. Starting fresh.")

    _, collection = get_or_create_collection(LOCAL_VECTORSTORE_PATH)

    rate_limiter = RequestRateLimiter(EMBEDDING_RPM_LIMIT, 60.0)
    request_semaphore = threading.Semaphore(EMBEDDING_MAX_INFLIGHT)

    print("\n📊 Step 4: Loading all Google Sheet tabs...")
    all_tabs = get_all_tabs(gc, sheet_url)
    print(f"  ✅ Found {len(all_tabs)} tab(s).")

    grand_total_rows = 0
    grand_already_done = 0
    grand_newly_processed = 0
    grand_newly_failed = 0
    grand_segments = 0
    grand_embeddings = 0
    grand_estimated_cost = 0.0
    tabs_skipped = []
    tabs_completed = []

    for tab_number, (worksheet, tab_name) in enumerate(all_tabs, start=1):
        print(f"\n{'=' * 70}")
        print(f"📑 Tab {tab_number}/{len(all_tabs)}: {tab_name}")
        print(f"{'=' * 70}")

        df = get_tab_dataframe(worksheet)

        if df.empty:
            print("  ⚠️ Empty tab. Skipping.")
            tabs_skipped.append(tab_name)
            continue

        missing_required = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing_required:
            print(f"  ⚠️ Missing required columns: {missing_required}. Skipping.")
            tabs_skipped.append(tab_name)
            continue

        for col in TRACKING_COLUMNS:
            if col not in df.columns:
                df[col] = ""
                print(f"  ➕ Added tracking column: {col}")

        mask_todo = (
            (df["vectorized"].astype(str).str.upper() != "TRUE") &
            (df["vectorized"].astype(str).str.upper() != "FAILED") &
            (df[COL_VIDEO_ID].notna()) &
            (df[COL_VIDEO_ID].astype(str).str.strip() != "")
        )

        videos_to_process = df[mask_todo]
        already_processed = int((df["vectorized"].astype(str).str.upper() == "TRUE").sum())
        failed_count = int((df["vectorized"].astype(str).str.upper() == "FAILED").sum())

        print(f"  📊 Total rows       : {len(df)}")
        print(f"  ✅ Already done     : {already_processed}")
        print(f"  ❌ Previously failed: {failed_count}")
        print(f"  ⏳ To process       : {len(videos_to_process)}")

        grand_total_rows += len(df)
        grand_already_done += already_processed

        if len(videos_to_process) == 0:
            tabs_completed.append(tab_name)
            continue

        tab_new_embeddings = 0
        tab_newly_processed = 0
        tab_newly_failed = 0
        tab_estimated_cost = 0.0

        for video_idx, (idx, row) in enumerate(
            tqdm(
                videos_to_process.iterrows(),
                total=len(videos_to_process),
                desc=f"Tab {tab_name}",
            ),
            start=1,
        ):
            file_id = str(row[COL_VIDEO_ID]).strip()
            video_name = str(row[COL_VIDEO_NAME]).strip()
            course_name = str(row.get(COL_COURSE_NAME, ""))
            topic_name = str(row.get(COL_TOPIC_NAME, ""))
            mime_type = str(row.get(COL_MIME_TYPE, "video/mp4"))
            duration = str(row.get(COL_DURATION, ""))
            file_size = str(row.get(COL_FILE_SIZE, ""))
            video_link = str(row.get(COL_VIDEO_LINK, ""))
            stock = str(row.get(COL_STOCK, ""))

            print(f"\n  🎥 [{video_idx}/{len(videos_to_process)}] {video_name}")
            print(f"     Tab: {tab_name} | Course: {course_name} | Topic: {topic_name}")

            temp_dir = tempfile.mkdtemp(prefix="video_embed_")

            try:
                video_path = os.path.join(temp_dir, "video.mp4")
                download_from_drive(drive, file_id, video_path)

                segments = segment_video(video_path, temp_dir, chunk_length, CHUNK_OVERLAP_SEC)
                if not segments:
                    raise RuntimeError("No video segments were created.")

                grand_segments += len(segments)

                video_cost = calculate_cost(len(segments))
                tab_estimated_cost += video_cost
                grand_estimated_cost += video_cost

                print(f"    💰 Est. cost: ${video_cost:.5f}")

                video_embeddings = []
                lock = threading.Lock()

                print(f"    🔄 Embedding {len(segments)} chunk(s)...")

                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_map = {
                        executor.submit(
                            generate_video_embedding,
                            gemini_client,
                            seg_path,
                            start_sec,
                            end_sec,
                            rate_limiter,
                            request_semaphore,
                        ): (j, seg_path, start_sec, end_sec)
                        for j, (seg_path, start_sec, end_sec) in enumerate(segments)
                    }

                    for future in as_completed(future_map):
                        j, seg_path, start_sec, end_sec = future_map[future]

                        try:
                            emb = future.result()
                            item = {
                                "video_id": file_id,
                                "video_name": video_name,
                                "course_name": course_name,
                                "topic_name": topic_name,
                                "mime_type": mime_type,
                                "duration": duration,
                                "file_size": file_size,
                                "video_link": video_link,
                                "stock": stock,
                                "tab_name": tab_name,
                                "segment_index": j,
                                "start_time": emb["start_time"],
                                "end_time": emb["end_time"],
                                "embedding": emb["vector"],
                            }

                            with lock:
                                video_embeddings.append(item)

                            print(f"      ✅ Chunk {j + 1}/{len(segments)}: {start_sec}s -> {end_sec}s")

                        except Exception as exc:
                            print(f"      ❌ Chunk {j + 1}/{len(segments)} failed: {exc}")

                if len(video_embeddings) == 0:
                    raise RuntimeError("No chunks were embedded. ChromaDB was not updated.")

                if len(video_embeddings) != len(segments):
                    raise RuntimeError(
                        f"Only embedded {len(video_embeddings)} of {len(segments)} chunks. "
                        "Stopping so the sheet is not incorrectly marked TRUE."
                    )

                upsert_embeddings(collection, video_embeddings)

                if BACKUP_AFTER_EACH_VIDEO:
                    backup_vectorstore_to_drive(drive, LOCAL_VECTORSTORE_PATH, vectorstore_folder_id)

                tab_new_embeddings += len(video_embeddings)
                grand_embeddings += len(video_embeddings)
                tab_newly_processed += 1
                grand_newly_processed += 1

                df.at[idx, "vectorized"] = "TRUE"
                df.at[idx, "embedding_ts"] = datetime.now().isoformat()
                df.at[idx, "segments_processed"] = str(len(segments))
                df.at[idx, "estimated_cost_usd"] = str(round(video_cost, 6))
                df.at[idx, "embedding_model"] = GEMINI_MODEL_NAME
                df.at[idx, "embedding_dimensionality"] = str(OUTPUT_DIMENSIONALITY)
                df.at[idx, "error_message"] = ""

                save_to_sheet(worksheet, df)
                print("    💾 Sheet saved and row marked TRUE.")

            except Exception as exc:
                tab_newly_failed += 1
                grand_newly_failed += 1

                print(f"    ❌ Video failed: {exc}")

                df.at[idx, "vectorized"] = "FAILED"
                df.at[idx, "embedding_ts"] = datetime.now().isoformat()
                df.at[idx, "embedding_model"] = GEMINI_MODEL_NAME
                df.at[idx, "embedding_dimensionality"] = str(OUTPUT_DIMENSIONALITY)
                df.at[idx, "error_message"] = str(exc)[:500]

                try:
                    save_to_sheet(worksheet, df)
                    print("    💾 Sheet saved and row marked FAILED.")
                except Exception as sheet_exc:
                    print(f"    ⚠️ Sheet save failed: {sheet_exc}")

            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)

        try:
            save_to_sheet(worksheet, df)
        except Exception as exc:
            print(f"  ⚠️ Final save failed for tab {tab_name}: {exc}")

        tabs_completed.append(tab_name)
        print(
            f"  ✅ Tab done: {tab_new_embeddings} new embeddings, "
            f"{tab_newly_processed} videos processed, {tab_newly_failed} failed, "
            f"est. cost ${tab_estimated_cost:.5f}"
        )

    print("\n📤 Step 5: Final Drive backup...")
    try:
        backup_vectorstore_to_drive(drive, LOCAL_VECTORSTORE_PATH, vectorstore_folder_id)
    except Exception as exc:
        print(f"  ⚠️ Drive backup failed: {exc}")
        print(f"  Local vectorstore is still at: {LOCAL_VECTORSTORE_PATH}")

    elapsed = time.time() - pipeline_start
    mins, sec = divmod(int(elapsed), 60)
    duration_str = f"{mins}m {sec}s" if mins else f"{sec}s"
    cb = cost_breakdown(grand_embeddings)

    stats = {
        "tabs_found": len(all_tabs),
        "tabs_completed": tabs_completed,
        "tabs_skipped": tabs_skipped,
        "total_rows_across_all_tabs": grand_total_rows,
        "already_processed_before_run": grand_already_done,
        "newly_processed": grand_newly_processed,
        "newly_failed": grand_newly_failed,
        "total_chunks_created": grand_segments,
        "total_chunks_embedded": grand_embeddings,
        "estimated_cost_standard_usd": cb["estimated_cost_standard_usd"],
        "estimated_cost_batch_usd": cb["estimated_cost_batch_usd"],
        "pricing_tier_used": cb["pricing_tier_used"],
        "estimated_total_cost_usd": cb["estimated_cost_usd"],
        "embedding_model": GEMINI_MODEL_NAME,
        "output_dimensionality": OUTPUT_DIMENSIONALITY,
        "collection_name": CHROMA_COLLECTION_NAME,
        "collection_count": collection.count(),
        "local_vectorstore_path": LOCAL_VECTORSTORE_PATH,
        "drive_parent_folder_id": parent_folder_id,
        "vectorstore_folder_name": VECTORSTORE_FOLDER_NAME,
        "duration": duration_str,
    }

    print("\n" + "=" * 70)
    print("🎉 PIPELINE COMPLETE")
    print(json.dumps(stats, indent=2))
    print("=" * 70)

    return LOCAL_VECTORSTORE_PATH, stats


def inspect_vectorstore(local_path: str = LOCAL_VECTORSTORE_PATH):
    client = PersistentClient(path=local_path)
    print("Collections:")
    for collection in client.list_collections():
        print(f"  - {collection.name}")
        c = client.get_collection(collection.name)
        print(f"    Count: {c.count()}")

    if os.path.exists(local_path):
        print("\nFiles:")
        for item in os.listdir(local_path):
            print(f"  - {item}")



