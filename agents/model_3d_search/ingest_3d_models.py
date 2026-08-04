import os
import re
import hashlib
import threading
import wordninja
import requests
import mimetypes
import imagehash
import pandas as pd
from PIL import Image
from io import BytesIO
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Optional, Tuple

from services.drive_service import upload_folder_to_drive
from agents.model_3d_search.vectorstore_3d import (
    load_3d_chroma_db,
    embed_paired_gemini,
    make_path_writable,
)

# ── Gemini multimodal embedding pricing (published rates) ──
_COST_PER_IMAGE_USD = 0.00012       # $0.00012 per image
_COST_PER_VIDEO_SEC_USD = 0.000790  # $0.000790 per second of video

# ── Parallelization limit (rate-limit safe) ──
MAX_INGEST_WORKERS = 16


# ── Gemini API rate limiter (simple token-bucket, thread-safe) ──
_gemini_semaphore = threading.Semaphore(MAX_INGEST_WORKERS)


def _video_duration_seconds(video_bytes: bytes) -> float:
    """
    Estimates video duration in seconds from raw bytes using mutagen (no temp file needed).
    Falls back to 0.0 if parsing fails (e.g. unsupported container).
    """
    try:
        from mutagen.mp4 import MP4
        tag = MP4(BytesIO(video_bytes))
        return float(tag.info.length)
    except Exception:
        pass
    try:
        from mutagen import File as MutagenFile
        tag = MutagenFile(BytesIO(video_bytes))
        if tag and hasattr(tag, 'info') and hasattr(tag.info, 'length'):
            return float(tag.info.length)
    except Exception:
        pass
    return 0.0


def clean_name_with_wordninja(text: str) -> str:
    """
    Uses wordninja to cleanly split concatenated compound words in 3D model names
    into space-separated words (e.g. 'Washingmachinesuspensionsystem' -> 'Washing Machine Suspension System').
    """
    if not text or not str(text).strip():
        return ""
    cleaned = re.sub(r'(\.blend|\.obj|\.fbx|\.glb|\.gltf|_v\d+)', '', str(text), flags=re.IGNORECASE).strip()
    tokens = wordninja.split(cleaned)
    if not tokens:
        return cleaned
    return " ".join(t.capitalize() for t in tokens if t.strip())


def extract_drive_file_id(url: str) -> str:
    """Extracts canonical Google Drive File ID or returns cleaned URL."""
    if not url:
        return ""
    url = str(url).strip()
    m = re.search(r"/file/d/([a-zA-Z0-9_-]+)", url)
    if m:
        return m.group(1)
    m = re.search(r"[?&]id=([a-zA-Z0-9_-]+)", url)
    if m:
        return m.group(1)
    return url.lower()


def _fetch_media_with_token(
    url: str,
    access_token: Optional[str],
) -> Optional[Tuple[bytes, str, str, Optional[Image.Image]]]:
    """
    Thread-safe media fetch using a pre-extracted OAuth Bearer token for Drive URLs.
    Avoids PyDrive shared-state issues by using requests directly.
    Returns (media_bytes, mime_type, media_kind, pil_image_or_None) or None on failure.
    """
    if not url or not str(url).strip():
        return None

    url = str(url).strip()
    media_bytes = None
    mime_type = None

    try:
        # Extract Drive file ID if it's a Drive URL
        file_id = None
        if "drive.google.com" in url:
            if "id=" in url:
                file_id = url.split("id=")[1].split("&")[0]
            elif "/file/d/" in url:
                file_id = url.split("/file/d/")[1].split("/")[0]

        if file_id and access_token:
            # Use Drive REST API v3 with Bearer token — fully thread-safe, with exponential backoff retry for 429/503
            dl_url = f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media"
            headers = {"Authorization": f"Bearer {access_token}"}
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    r = requests.get(dl_url, headers=headers, timeout=30)
                    if r.status_code == 200:
                        media_bytes = r.content
                        ct = r.headers.get("Content-Type", "").split(";")[0].strip()
                        mime_type = ct if ct else None
                        break
                    elif r.status_code in (429, 500, 502, 503, 504) and attempt < max_retries - 1:
                        import time, random
                        backoff = (2 ** attempt) + random.uniform(0.1, 1.0)
                        print(f"⚠️ Drive REST HTTP {r.status_code} for file_id={file_id} — retrying in {backoff:.2f}s (attempt {attempt+1}/{max_retries})")
                        time.sleep(backoff)
                    else:
                        print(f"⚠️ Drive REST download failed: HTTP {r.status_code} for file_id={file_id}")
                        return None
                except Exception as req_err:
                    if attempt < max_retries - 1:
                        import time, random
                        backoff = (2 ** attempt) + random.uniform(0.1, 1.0)
                        time.sleep(backoff)
                    else:
                        print(f"⚠️ Drive REST download exception for file_id={file_id}: {req_err}")
                        return None
        else:
            # Non-Drive URL — plain HTTP download with retry
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    r = requests.get(url, timeout=20)
                    if r.status_code == 200:
                        media_bytes = r.content
                        ct = r.headers.get("Content-Type", "").split(";")[0].strip()
                        mime_type = ct if ct else None
                        break
                    elif r.status_code in (429, 500, 502, 503, 504) and attempt < max_retries - 1:
                        import time, random
                        backoff = (2 ** attempt) + random.uniform(0.1, 1.0)
                        time.sleep(backoff)
                    else:
                        return None
                except Exception:
                    if attempt < max_retries - 1:
                        import time, random
                        time.sleep((2 ** attempt) + random.uniform(0.1, 1.0))
                    else:
                        return None

        if not media_bytes:
            return None

        # Fallback MIME type from URL extension
        if not mime_type or mime_type in ("application/octet-stream", ""):
            guessed, _ = mimetypes.guess_type(url)
            mime_type = guessed or "image/png"

        # Classify as video or image
        is_video = (
            (mime_type and "video" in mime_type)
            or url.lower().endswith((".mp4", ".mov", ".webm", ".avi", ".mkv"))
        )

        if is_video:
            if not mime_type or "video" not in mime_type:
                mime_type = "video/mp4"
            return (media_bytes, mime_type, "video", None)
        else:
            if "image" not in mime_type:
                mime_type = "image/png"
            try:
                pil_img = Image.open(BytesIO(media_bytes)).convert("RGB")
            except Exception:
                print(f"⚠️ Could not decode image bytes from {url} — falling back to text-only")
                return None
            return (media_bytes, mime_type, "image", pil_img)

    except Exception as e:
        print(f"⚠️ _fetch_media_with_token failed for {url}: {e}")
        return None


def flush_chroma_sqlite_wal(local_chroma_path: str):
    """
    Flushes SQLite Write-Ahead Log (WAL) into the main chroma.sqlite3 file
    so the SQLite database file on disk is 100% self-contained before Drive upload.
    """
    import sqlite3
    sqlite_file = os.path.join(local_chroma_path, "chroma.sqlite3")
    if os.path.exists(sqlite_file):
        try:
            conn = sqlite3.connect(sqlite_file)
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
            conn.close()
            print("  💾 SQLite WAL flushed to chroma.sqlite3 successfully.")
        except Exception as e:
            print(f"  ⚠️ Could not flush SQLite WAL: {e}")


def sync_chroma_to_drive(drive, parent_folder_id: str, local_chroma_path: str):
    """
    Upload updated local ChromaDB folder back to Google Drive under `Vectorstore files`.
    """
    if not drive:
        return

    # Ensure all WAL log records are committed into chroma.sqlite3 before copying
    flush_chroma_sqlite_wal(local_chroma_path)

    vstore_list = drive.ListFile({
        'q': f"title='Vectorstore files' and '{parent_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
    }).GetList()

    if vstore_list:
        vstore_folder_id = vstore_list[0]['id']
    else:
        vstore_folder = drive.CreateFile({
            'title': 'Vectorstore files',
            'parents': [{'id': parent_folder_id}],
            'mimeType': 'application/vnd.google-apps.folder'
        })
        vstore_folder.Upload()
        vstore_folder_id = vstore_folder['id']

    print("📤 Syncing 3D Vectorstore to Google Drive...")
    try:
        existing_items = drive.ListFile({
            'q': f"'{vstore_folder_id}' in parents and trashed=false"
        }).GetList()
        for item in existing_items:
            try:
                item.Trash()
            except Exception as e:
                print(f"  ⚠️ Could not trash old drive file {item.get('title', '?')}: {e}")
    except Exception as e:
        print(f"  ⚠️ Could not list old drive files: {e}")

    upload_folder_to_drive(local_chroma_path, vstore_folder_id, drive)
    print("✅ Sync to Google Drive complete!")


def get_3d_sheet_info_and_stats(gc, sheet_link: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Loads Google Sheet, checks/adds 'vectorized' and 'vectorized_ts' columns, and returns (df, stats_dict).
    """
    sheet = gc.open_by_url(sheet_link)
    worksheet = sheet.worksheet("Final Output of the migrated assets updated")
    df = pd.DataFrame(worksheet.get_all_records())

    added_cols = False
    for col in ["vectorized", "vectorized_ts"]:
        if col not in df.columns:
            df[col] = ""
            added_cols = True

    if added_cols:
        try:
            worksheet.update([df.columns.values.tolist()] + df.values.tolist())
            print("✅ Added 'vectorized' and 'vectorized_ts' tracking columns to Google Sheet!")
        except Exception as e:
            print(f"⚠️ Could not write updated header to Google Sheet: {e}")

    valid_mask = df["New File Name"].astype(str).str.strip().astype(bool)
    valid_df = df.loc[valid_mask].copy()

    processed_mask = valid_df["vectorized"].astype(str).str.upper() == "TRUE"
    processed_count = int(processed_mask.sum())
    total_count = int(len(valid_df))
    unprocessed_count = total_count - processed_count

    valid_df["Status"] = valid_df["vectorized"].apply(lambda v: "Processed" if str(v).upper() == "TRUE" else "Unprocessed")

    stats = {
        "total_rows": total_count,
        "processed_rows": processed_count,
        "unprocessed_rows": unprocessed_count,
    }

    return valid_df, stats


def _process_row_worker(args: Dict) -> Optional[Dict]:
    """
    Parallel worker: downloads media + computes Gemini embedding for one row.
    Pure I/O + compute — reads no shared mutable state, writes nothing.
    Returns a result dict for the main thread to write, or None on failure.
    """
    idx = args["idx"]
    row = args["row"]
    access_token = args["access_token"]

    new_file_name = str(row.get("New File Name", "")).strip()
    raw_cleaned = str(row.get("Cleaned Model", "")).strip()
    target_name = raw_cleaned or new_file_name
    cleaned_model = clean_name_with_wordninja(target_name)
    original_file_name = str(row.get("File Name", "")).strip()
    category = str(row.get("Category", "")).strip()
    extension = str(row.get("Extension", "")).strip()
    drive_url = str(row.get("Google Drive URL", "")).strip()

    _raw_preview_optional = row.get("Preview Media URL (Optional)")
    _raw_preview_short    = row.get("Preview Media URL")
    preview_url = str(_raw_preview_optional or _raw_preview_short or "").strip()

    print(
        f"[Worker {idx}] {new_file_name!r} ➜ wordninja: {cleaned_model!r} │ "
        f"preview_url={preview_url!r}"
    )

    try:
        size_mb = float(row.get("Size (MB)", 0.0) or 0.0)
    except (ValueError, TypeError):
        size_mb = 0.0
    owner = str(row.get("Owner", "")).strip()
    parent_folder = str(row.get("Parent Folder", "")).strip()

    if not new_file_name or not drive_url:
        return None

    canonical_key = extract_drive_file_id(drive_url)
    url_hash = hashlib.md5(canonical_key.encode("utf-8")).hexdigest()[:16]
    doc_id = f"3d_{url_hash}"

    # ── Step 1: Fetch media (thread-safe, no shared PyDrive state) ──
    media_bytes = None
    mime_type = None
    media_kind = "text_only"
    phash = ""

    if preview_url:
        print(f"[Worker {idx}] 🌐 Fetching media: {preview_url!r}")
        media_payload = _fetch_media_with_token(preview_url, access_token)
        if media_payload:
            media_bytes, mime_type, media_kind, pil_img = media_payload
            print(f"[Worker {idx}] ✅ Media ready │ kind={media_kind!r} │ bytes={len(media_bytes) if media_bytes else 0}")
            if pil_img:
                try:
                    phash = str(imagehash.phash(pil_img))
                except Exception:
                    pass
        else:
            print(f"[Worker {idx}] ❌ Media fetch returned None")
    else:
        print(f"[Worker {idx}] ⏭️  No preview_url — text-only embed")

    metadata = {
        "new_file_name": new_file_name,
        "cleaned_model": cleaned_model,
        "original_file_name": original_file_name,
        "category": category,
        "extension": extension,
        "3d_model_drive_url": drive_url,
        "preview_media_url": preview_url,
        "size_mb": size_mb,
        "owner": owner,
        "parent_folder": parent_folder,
        "phash": phash,
        "media_kind": media_kind,
    }

    # ── Step 2: Gemini embedding (thread-safe, pure HTTP call) ──
    try:
        with _gemini_semaphore:
            aggregated_vec = embed_paired_gemini(
                text=cleaned_model or new_file_name,
                media_bytes=media_bytes,
                mime_type=mime_type,
            )
    except Exception as e:
        print(f"[Worker {idx}] ⚠️ Embedding failed: {e}")
        return None

    doc_text = f"Cleaned Model: {cleaned_model} | New File Name: {new_file_name} | Category: {category}"

    # ── Step 3: Compute cost ──
    row_cost = 0.0
    if media_bytes and media_kind == "image":
        row_cost = _COST_PER_IMAGE_USD
    elif media_bytes and media_kind == "video":
        row_cost = _video_duration_seconds(media_bytes) * _COST_PER_VIDEO_SEC_USD

    print(f"[Worker {idx}] 💾 embed_type={'media+text' if media_bytes else 'text-only'} │ row_cost=${row_cost:.5f}")

    return {
        "idx": idx,
        "doc_id": doc_id,
        "aggregated_vec": aggregated_vec,
        "metadata": metadata,
        "doc_text": doc_text,
        "row_cost": row_cost,
        "new_file_name": new_file_name,
    }


def process_and_index_3d_models(
    gc,
    drive,
    sheet_link: str,
    parent_folder_id: str = "root",
    force_recreate: bool = False,
    batch_size: int = 5,
    progress_callback: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Core ingestion function. Parallelizes Drive download + Gemini embedding across
    MAX_INGEST_WORKERS workers, then serializes all ChromaDB/Sheet writes in the main thread.

    Architecture:
        Workers (parallel)  : _fetch_media_with_token + embed_paired_gemini  (I/O-bound)
        Main thread (serial): ChromaDB upsert + Sheet milestone saves          (fast writes)

    Rate-limit headroom at 6 workers / ~3s avg op time:
        Drive   → 2 req/s vs 10 req/s limit  (20%)
        Gemini  → 2 req/s vs 25 req/s limit   (8%)
        Sheets  → <0.2 req/s vs 1 req/s limit (20%)
    """
    dbs = load_3d_chroma_db(drive=drive, parent_folder_id=parent_folder_id, force_reload=force_recreate)
    unified_chroma = dbs["unified"]
    local_chroma_path = dbs["path"]
    make_path_writable(local_chroma_path)

    # Open sheet
    sheet = gc.open_by_url(sheet_link)
    worksheet = sheet.worksheet("Final Output of the migrated assets updated")
    df = pd.DataFrame(worksheet.get_all_records())

    # ── DIAGNOSTIC: verify column headers ──
    print(f"📋 Sheet columns ({len(df.columns)}): {list(df.columns)}")
    preview_col_present = "Preview Media URL (Optional)" in df.columns
    preview_col_short = "Preview Media URL" in df.columns
    print(f"   'Preview Media URL (Optional)' present: {preview_col_present}")
    print(f"   'Preview Media URL' present:            {preview_col_short}")
    if not preview_col_present and not preview_col_short:
        print("   ⚠️  NEITHER preview column found — all rows will embed text-only!")

    # Add tracking columns if missing
    added_cols = False
    for col in ["vectorized", "vectorized_ts"]:
        if col not in df.columns:
            df[col] = ""
            added_cols = True
    if added_cols:
        try:
            worksheet.update([df.columns.values.tolist()] + df.values.tolist())
        except Exception:
            pass

    rows_to_process = df
    if not force_recreate:
        mask = (df["vectorized"].astype(str).str.upper() != "TRUE") & df["New File Name"].astype(str).str.strip().astype(bool)
        rows_to_process = df.loc[mask]

    total_to_process = len(rows_to_process)
    print(f"📊 Total sheet rows: {len(df)}, Rows to index: {total_to_process} | Workers: {MAX_INGEST_WORKERS}")

    # Pre-extract OAuth token once (valid ~1 hour; avoids per-thread PyDrive auth races)
    access_token: Optional[str] = None
    if drive:
        try:
            access_token = drive.auth.credentials.access_token
            print(f"🔑 OAuth access token extracted (len={len(access_token) if access_token else 0})")
        except Exception as e:
            print(f"⚠️ Could not extract OAuth token — Drive downloads may fail: {e}")

    now_iso = datetime.now().isoformat()
    indexed_count = 0
    batch_indexed_count = 0
    total_cost_usd = 0.0

    # Build worker args list
    worker_args = [
        {"idx": idx, "row": row, "access_token": access_token}
        for idx, row in rows_to_process.iterrows()
    ]

    # ── Parallel producer + serial consumer ──
    with ThreadPoolExecutor(max_workers=MAX_INGEST_WORKERS) as executor:
        futures = {executor.submit(_process_row_worker, args): args["idx"] for args in worker_args}

        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as e:
                orig_idx = futures[future]
                print(f"⚠️ Worker exception for row {orig_idx}: {e}")
                continue

            if result is None:
                continue

            idx = result["idx"]

            # ── Serial: ChromaDB upsert (SQLite — one writer at a time, handled by as_completed loop) ──
            try:
                unified_chroma._collection.upsert(
                    embeddings=[result["aggregated_vec"]],
                    metadatas=[result["metadata"]],
                    documents=[result["doc_text"]],
                    ids=[result["doc_id"]],
                )
                df.at[idx, "vectorized"] = "TRUE"
                df.at[idx, "vectorized_ts"] = now_iso
                indexed_count += 1
                batch_indexed_count += 1
                total_cost_usd += result["row_cost"]

                if progress_callback:
                    try:
                        progress_callback(indexed_count, total_to_process, total_cost_usd)
                    except Exception:
                        pass

                print(f"✅ [{indexed_count}/{total_to_process}] Indexed: {result['new_file_name']} | cumulative cost=${total_cost_usd:.4f}")

                # ── Milestone: save sheet + sync Drive every batch_size items ──
                if batch_indexed_count >= batch_size:
                    print(f"🏁 Milestone ({indexed_count}/{total_to_process}). Saving Sheet & Drive...")
                    try:
                        worksheet.update([df.columns.values.tolist()] + df.values.tolist())
                        print("  ✅ Google Sheet milestone saved!")
                    except Exception as e:
                        print(f"  ⚠️ Milestone sheet update failed: {e}")

                    if drive:
                        try:
                            sync_chroma_to_drive(drive, parent_folder_id, local_chroma_path)
                            print("  ✅ Drive vectorstore milestone synced!")
                        except Exception as e:
                            print(f"  ⚠️ Milestone Drive sync failed: {e}")

                    batch_indexed_count = 0

            except Exception as e:
                print(f"⚠️ ChromaDB upsert failed for row {idx} ({result.get('new_file_name', '?')}): {e}")

    # ── Final save for remaining incomplete batch ──
    if batch_indexed_count > 0:
        try:
            worksheet.update([df.columns.values.tolist()] + df.values.tolist())
            print("✅ Final Google Sheet progress updated!")
        except Exception as e:
            print(f"⚠️ Final Google Sheet update failed: {e}")

        if drive and indexed_count > 0:
            try:
                sync_chroma_to_drive(drive, parent_folder_id, local_chroma_path)
                print("✅ Final Drive vectorstore synced!")
            except Exception as e:
                print(f"⚠️ Final Drive sync failed: {e}")

    return {
        "status": "success",
        "total_rows": len(df),
        "indexed_rows": indexed_count,
        "cost_usd": round(total_cost_usd, 6),
    }


def create_vectorstore_3d(
    gc,
    drive,
    sheet_link: str,
    parent_folder_id: str = "root",
    batch_size: int = 5,
    progress_callback: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Admin Action: Create/recreate 3D vectorstore from scratch.
    """
    return process_and_index_3d_models(
        gc=gc,
        drive=drive,
        sheet_link=sheet_link,
        parent_folder_id=parent_folder_id,
        force_recreate=True,
        batch_size=batch_size,
        progress_callback=progress_callback,
    )


def update_vectorstore_3d(
    gc,
    drive,
    sheet_link: str,
    parent_folder_id: str = "root",
    batch_size: int = 5,
    progress_callback: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Admin Action: Incremental update of 3D vectorstore.
    """
    return process_and_index_3d_models(
        gc=gc,
        drive=drive,
        sheet_link=sheet_link,
        parent_folder_id=parent_folder_id,
        force_recreate=False,
        batch_size=batch_size,
        progress_callback=progress_callback,
    )
