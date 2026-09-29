import re
import io
import streamlit as st
from typing import List, Tuple, Dict, Any, Optional
from PIL import Image
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from services.drive_service import (
    extract_drive_id_from_url,
    build_service_account_drive_service,
)


def extract_all_drive_ids(text_input: str) -> List[Tuple[str, str]]:
    """
    Extract (item_id, item_type) pairs from a text cell containing one or multiple Google Drive URLs.
    item_type can be 'file', 'folder', or 'unknown'.
    """
    if not text_input or not isinstance(text_input, str):
        return []

    results: List[Tuple[str, str]] = []
    seen = set()

    # Split by whitespace, comma, newline, semicolon
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


def _get_drive_v3_client(session_creds=None):
    """
    Get authenticated Google Drive v3 API client.
    First checks provided session_creds, then Streamlit session_state, then PyDrive, then service account.
    """
    creds = session_creds or st.session_state.get("google_credentials")
    if not creds and st.session_state.get("drive"):
        auth = getattr(st.session_state["drive"], "auth", None)
        creds = getattr(auth, "credentials", None) if auth else None

    if creds:
        try:
            return build("drive", "v3", credentials=creds, cache_discovery=False)
        except Exception:
            pass

    return build_service_account_drive_service()


def fetch_media_from_drive_links(
    links_cell_content: str,
    session_creds: Optional[Any] = None,
    max_items: int = 10
) -> Tuple[List[Image.Image], List[Dict[str, Any]], List[str]]:
    """
    Fetches PIL Images and Videos (raw bytes dicts) from a cell containing Google Drive URLs.
    Returns: (downloaded_images, downloaded_videos, media_descriptions)
    where downloaded_videos is a list of dicts: {'bytes': bytes, 'mime_type': str, 'filename': str}
    """
    if not links_cell_content:
        print("  [REVIEWER LOG] Drive Media Links cell is empty.")
        return [], [], ["No drive media links provided."]

    print(f"  [REVIEWER LOG] Fetching media for Drive links: {links_cell_content}")
    drive_client = _get_drive_v3_client(session_creds)
    if not drive_client:
        print("  [REVIEWER LOG] ❌ Error: Failed to initialize Google Drive API client.")
        return [], [], ["Failed to initialize Google Drive API client."]

    items = extract_all_drive_ids(links_cell_content)
    if not items:
        print(f"  [REVIEWER LOG] ⚠️ Warning: Could not parse valid Drive IDs from text.")
        return [], [], [f"Could not parse valid Drive IDs from: {links_cell_content}"]

    downloaded_images: List[Image.Image] = []
    downloaded_videos: List[Dict[str, Any]] = []
    media_descriptions: List[str] = []

    def process_file_id(file_id: str, name_hint: str = ""):
        nonlocal downloaded_images, downloaded_videos, media_descriptions
        if (len(downloaded_images) + len(downloaded_videos)) >= max_items:
            return

        try:
            meta = drive_client.files().get(
                fileId=file_id,
                fields="id, name, mimeType",
                supportsAllDrives=True
            ).execute()

            mime_type = meta.get("mimeType", "")
            file_name = meta.get("name") or name_hint or file_id
            ext = file_name.split(".")[-1].lower() if "." in file_name else ""

            is_image = mime_type.startswith("image/")
            is_video = mime_type.startswith("video/") or ext in ("mp4", "mov", "avi", "mkv", "webm", "m4v", "3gp")

            if not is_image and not is_video and mime_type == "application/octet-stream":
                # Guess based on extension if octet-stream
                if ext in ("jpg", "jpeg", "png", "webp", "heic"):
                    is_image = True
                    mime_type = f"image/{ext}"
                elif ext in ("mp4", "mov", "avi", "webm"):
                    is_video = True
                    mime_type = f"video/{'mp4' if ext == 'mov' else ext}"

            if not is_image and not is_video:
                print(f"  [REVIEWER LOG] Skipping unsupported non-media file '{file_name}' (mime: {mime_type})")
                return

            request = drive_client.files().get_media(fileId=file_id, supportsAllDrives=True)
            file_buffer = io.BytesIO()
            downloader = MediaIoBaseDownload(file_buffer, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()

            file_buffer.seek(0)
            raw_bytes = file_buffer.getvalue()

            if is_image:
                try:
                    img = Image.open(io.BytesIO(raw_bytes))
                    if img.mode not in ("RGB", "L"):
                        img = img.convert("RGB")
                    downloaded_images.append(img)
                    media_descriptions.append(f"Image: {file_name}")
                    print(f"  [REVIEWER LOG] 🖼️ Downloaded image: '{file_name}' ({img.size[0]}x{img.size[1]})")
                except Exception as img_err:
                    print(f"  [REVIEWER LOG] Could not parse image '{file_name}': {img_err}")
            elif is_video:
                if not mime_type or mime_type == "application/octet-stream":
                    mime_type = "video/mp4"
                downloaded_videos.append({
                    "bytes": raw_bytes,
                    "mime_type": mime_type,
                    "filename": file_name,
                })
                media_descriptions.append(f"Video: {file_name}")
                print(f"  [REVIEWER LOG] 🎥 Downloaded video: '{file_name}' ({len(raw_bytes)} bytes, mime: {mime_type})")

        except Exception as err:
            print(f"  [REVIEWER LOG] ❌ Error fetching Drive file '{file_id}': {err}")
            media_descriptions.append(f"Error fetching file {file_id}: {str(err)}")

    for item_id, item_type in items:
        if (len(downloaded_images) + len(downloaded_videos)) >= max_items:
            break

        if item_type == "folder" or item_type == "unknown":
            try:
                print(f"  [REVIEWER LOG] Querying folder ID '{item_id}' for media files...")
                query = f"'{item_id}' in parents and trashed = false and (mimeType starts with 'image/' or mimeType starts with 'video/')"
                response = drive_client.files().list(
                    q=query,
                    fields="files(id, name, mimeType)",
                    supportsAllDrives=True,
                    includeItemsFromAllDrives=True,
                    pageSize=max_items - (len(downloaded_images) + len(downloaded_videos))
                ).execute()

                files_in_folder = response.get("files", [])
                if files_in_folder:
                    print(f"  [REVIEWER LOG] Found {len(files_in_folder)} media file(s) in folder '{item_id}'")
                    for f in files_in_folder:
                        process_file_id(f["id"], f["name"])
                        if (len(downloaded_images) + len(downloaded_videos)) >= max_items:
                            break
                else:
                    process_file_id(item_id)
            except Exception:
                process_file_id(item_id)
        else:
            process_file_id(item_id)

    print(f"  [REVIEWER LOG] Finished media extraction: {len(downloaded_images)} image(s), {len(downloaded_videos)} video(s) loaded.")
    return downloaded_images, downloaded_videos, media_descriptions


def fetch_images_from_drive_links(
    links_cell_content: str,
    session_creds: Optional[Any] = None,
    max_images: int = 10
) -> Tuple[List[Image.Image], List[str]]:
    """Backward compatible wrapper returning (images, descriptions)."""
    imgs, vids, desc = fetch_media_from_drive_links(links_cell_content, session_creds, max_images)
    return imgs, desc

