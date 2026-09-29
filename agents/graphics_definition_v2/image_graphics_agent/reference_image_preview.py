import re
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from pydrive2.drive import GoogleDrive

from services.sheets_service import (
    get_sheet_data_and_df,
    create_or_read_worksheet,
    save_to_sheet,
    format_worksheet,
)
from agents.graphics_definition_v2.image_graphics_agent.drive_search import (
    _get_drive_instance,
    parse_search_queries_column,
)
from agents.graphics_definition_v2.image_graphics_agent.reference_pool_agent import (
    parse_reference_image_map,
    resolve_reference_pool_root_folder_id,
)

load_dotenv()


def _extract_drive_file_id(url: str) -> str:
    if not url:
        return ""
    patterns = [
        r"https://drive\.google\.com/file/d/([a-zA-Z0-9_-]+)",
        r"https://drive\.google\.com/open\?id=([a-zA-Z0-9_-]+)",
        r"https://drive\.google\.com/uc\?id=([a-zA-Z0-9_-]+)",
        r"https://drive\.google\.com/uc\?export=download&id=([a-zA-Z0-9_-]+)",
        r"https://drive\.google\.com/uc\?export=view&id=([a-zA-Z0-9_-]+)",
    ]
    for pattern in patterns:
        match = re.match(pattern, url)
        if match:
            return match.group(1)
    return ""


def _normalize_drive_url(url: str) -> str:
    file_id = _extract_drive_file_id(url)
    if file_id:
        return f"https://drive.google.com/uc?id={file_id}"
    return url or ""


def _list_images_in_folder(drive: GoogleDrive, folder_id: str):
    """Return a list of (file_id, name, url) for images under a folder (recursive)."""
    if not drive or not folder_id:
        return []

    results = []
    seen_ids = set()
    stack = [folder_id]

    while stack:
        current = stack.pop()
        file_list = drive.ListFile({
            "q": (
                f"'{current}' in parents and trashed=false"
            )
        }).GetList()

        for item in file_list:
            if item.get("mimeType") == "application/vnd.google-apps.folder":
                stack.append(item["id"])
                continue

            mime_type = item.get("mimeType") or ""
            if not mime_type.startswith("image/"):
                continue

            file_id = item.get("id")
            if not file_id or file_id in seen_ids:
                continue

            seen_ids.add(file_id)
            name = item.get("title") or item.get("name") or ""
            url = f"https://drive.google.com/uc?id={file_id}"
            results.append((file_id, name, url))

    return results


def _resolve_images_folder_id(drive: GoogleDrive, root_folder_id: str) -> str:
    """Find a likely images folder under Reference Image, fallback to root_folder_id."""
    if not drive or not root_folder_id:
        return root_folder_id

    try:
        ref_img_list = drive.ListFile({
            "q": (
                f"title='Reference Image' and '{root_folder_id}' in parents "
                "and mimeType='application/vnd.google-apps.folder' and trashed=false"
            )
        }).GetList()
        if ref_img_list:
            ref_img_id = ref_img_list[0]["id"]
            for name in ["Images", "Image Pool", "Reference Images", "Images Pool"]:
                img_list = drive.ListFile({
                    "q": (
                        f"title='{name}' and '{ref_img_id}' in parents "
                        "and mimeType='application/vnd.google-apps.folder' and trashed=false"
                    )
                }).GetList()
                if img_list:
                    return img_list[0]["id"]
            return ref_img_id
    except Exception:
        return root_folder_id

    return root_folder_id


def build_reference_image_preview(sheet, images_folder_id=None, worksheet_name="Reference Image Preview"):
    """
    Write a preview sheet with all images (deduped by link) from the Drive images folder.
    For assigned images, include VO/segment/slide/search queries and related context.
    """
    drive = _get_drive_instance()
    if not drive:
        raise ValueError("Google Drive authentication failed. Cannot build preview.")

    root_folder_id = st.session_state.get("root_folder_id")
    if not root_folder_id:
        root_folder_id = "1QS6PmCESfgFWNNEpRJDUB0E-t8iMAatH"

    resolved_root_id = resolve_reference_pool_root_folder_id(drive, root_folder_id)

    if not images_folder_id:
        images_folder_id = _resolve_images_folder_id(drive, resolved_root_id)

    images = _list_images_in_folder(drive, images_folder_id)

    _, df = get_sheet_data_and_df(sheet, "Slide Chunks")

    assigned_lookup = {}
    for index, row in df.iterrows():
        ref_map = parse_reference_image_map(row.get("reference_image_map", ""))
        vo_segments = [seg.strip() for seg in str(row.get("voiceover_segment", "")).splitlines() if seg.strip()]
        search_queries_text = str(row.get("search_queries", "")).strip()
        segments_queries_list = parse_search_queries_column(search_queries_text)
        segments_queries = {seg_num: queries for seg_num, queries in segments_queries_list}

        for seg_num, url in ref_map.items():
            norm_url = _normalize_drive_url(url)
            assigned_lookup.setdefault(norm_url, []).append({
                "Row": index + 2,
                "Segment": seg_num,
                "VO Segment": vo_segments[seg_num - 1] if seg_num - 1 < len(vo_segments) else "",
                "Slide Chunk Title": row.get("Slide Chunk Title", ""),
                "Topic": row.get("Topic", ""),
                "Subtopic": row.get("Subtopic", ""),
                "Search Queries": "\n".join(segments_queries.get(seg_num, [])),
            })

    seen_names = set()
    preview_rows = []
    for file_id, name, url in images:
        norm_url = _normalize_drive_url(url)
        name_key = (name or "").strip().lower()
        if name_key in seen_names:
            continue
        seen_names.add(name_key)

        assignments = assigned_lookup.get(norm_url, [])
        if assignments:
            for item in assignments:
                preview_rows.append({
                    "Image Name": name,
                    "Image URL": f"=IMAGE(\"{norm_url}\")",
                    "Assigned": "YES",
                    **item,
                })
        else:
            preview_rows.append({
                "Image Name": name,
                    "Image URL": f"=IMAGE(\"{norm_url}\")",
                "Assigned": "NO",
                "Row": "",
                "Segment": "",
                "VO Segment": "",
                "Slide Chunk Title": "",
                "Topic": "",
                "Subtopic": "",
                "Search Queries": "",
            })

    preview_df = pd.DataFrame(preview_rows)
    ws, _ = create_or_read_worksheet(sheet, worksheet_name, rows=max(len(preview_df) + 10, 100), cols=12)
    save_to_sheet(ws, preview_df)
    format_worksheet(ws)
    print(f"✅ Reference image preview written to worksheet: {worksheet_name}")
