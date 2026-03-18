"""
Download all assets from final_graphics_definition for a sheet into a Google Drive folder.

Creates a run folder: "CourseName, DD/MM/YYYY, HH:MM IST" under the configured parent folder,
and uploads one file per visual: "Slide X, S{n}V{m}.{ext}" (image or .mp4 for YouTube clips).
"""

import os
import re
import tempfile
import subprocess
import base64
from datetime import datetime, timezone, timedelta
from typing import Optional, Tuple, List, Dict, Any
from urllib.parse import urlparse, parse_qs

import requests
from tqdm import tqdm

from services.sheets_service import get_sheet_data_and_df
from services.drive_service import login_with_service_account
from services.smart_progress_bar import SmartProgressBar
from pydrive2.drive import GoogleDrive

from agents.graphics_definition_v2.review_agent.review_and_revise import (
    parse_final_graphics_definition,
    parse_visual_steps,
)

# Parent folder ID for "Downloadable Asset Folder"
DOWNLOAD_ASSETS_PARENT_FOLDER_ID = "1ZOtLwGXvOhxLY0eGksEWEgkYbRLOApSZ"


def _safe_str(value):
    if value is None:
        return ""
    try:
        import math
        if isinstance(value, float) and math.isnan(value):
            return ""
    except Exception:
        pass
    s = str(value).strip()
    if s == "nan" or s == "":
        return ""
    return s


def _is_drive_url(url: str) -> bool:
    if not url:
        return False
    return "drive.google.com" in url.lower() or "docs.google.com" in url.lower()


def _extract_drive_file_id(url: str) -> Optional[str]:
    if not url:
        return None
    for pattern in [
        r"/file/d/([a-zA-Z0-9-_]+)",
        r"id=([a-zA-Z0-9-_]+)",
        r"drive.google.com/open\?id=([a-zA-Z0-9-_]+)",
    ]:
        m = re.search(pattern, url)
        if m:
            return m.group(1)
    return None


def _parse_final_graphics_to_visuals(text: str) -> List[Dict[str, Any]]:
    """
    Parse final_graphics_definition text (formatted or XML) into a flat list of segments/steps with asset.
    Returns list of { "segment_num": int, "step_index": int, "asset": str }.
    """
    out = []
    cleaned = _safe_str(text)
    if not cleaned:
        return out

    # Try formatted style first (SEGMENT n headers + "Graphics to use:")
    segments = parse_final_graphics_definition(cleaned)
    if segments:
        for segment_num in sorted(segments.keys()):
            segment_text = segments[segment_num]
            steps = parse_visual_steps(segment_text)
            for idx, step in enumerate(steps, start=1):
                asset = _safe_str(step.get("asset", ""))
                if asset:
                    out.append({"segment_num": segment_num, "step_index": idx, "asset": asset})
        return out

    # Try XML: <final_graphics_definition> or <visual_steps> with <visual_step><asset>
    import xml.etree.ElementTree as ET
    try:
        root = ET.fromstring(f"<root>{cleaned}</root>")
    except ET.ParseError:
        try:
            root = ET.fromstring(cleaned)
        except ET.ParseError:
            return out
    step_list = []
    for el in root.iter():
        if el.tag.lower() == "visual_step":
            asset = ""
            for c in el:
                if c.tag.lower() in ("asset", "graphics_to_use", "graphic"):
                    asset = (c.text or "").strip()
                    break
            if asset:
                step_list.append(asset)
    if step_list:
        for idx, asset in enumerate(step_list, start=1):
            out.append({"segment_num": 1, "step_index": idx, "asset": asset})
    return out


def _get_drive_instance():
    """Get Google Drive instance from session state (if Streamlit) or from environment."""
    try:
        import streamlit as st
        if hasattr(st, "session_state") and "drive" in st.session_state:
            return st.session_state["drive"]
    except Exception:
        pass
    try:
        key_bytes = base64.b64decode(os.environ["GDRIVE_SA_B64"])
        sa_json = key_bytes.decode()
        gauth = login_with_service_account(json_str=sa_json)
        gauth.ServiceAuth()
        return GoogleDrive(gauth)
    except Exception as e:
        print(f"Could not initialize Drive from environment: {e}")
        return None


def _ist_now() -> datetime:
    """Current time in IST (UTC+5:30)."""
    return datetime.now(timezone(timedelta(hours=5, minutes=30)))


def _run_folder_name(course_name: str) -> str:
    """Build run folder name: 'CourseName, DD/MM/YYYY, HH:MM IST'."""
    t = _ist_now()
    safe_name = re.sub(r'[<>:"/\\|?*]', "_", str(course_name).strip() or "Course")
    return f"{safe_name}, {t.strftime('%d/%m/%Y, %H:%M')} IST"


def _is_youtube_url(url: str) -> bool:
    if not url or not isinstance(url, str):
        return False
    u = url.lower()
    return "youtube.com" in u or "youtu.be" in u


def _parse_youtube_url(url: str) -> Tuple[Optional[str], Optional[int], Optional[int]]:
    """Return (video_id, start_sec, end_sec). start/end may be None."""
    if not url:
        return None, None, None
    video_id = None
    if "youtu.be/" in url:
        video_id = url.split("youtu.be/")[-1].split("?")[0].split("/")[0]
    elif "youtube.com" in url:
        if "/embed/" in url:
            video_id = url.split("/embed/")[-1].split("?")[0].split("/")[0]
        elif "v=" in url:
            parsed = urlparse(url)
            params = parse_qs(parsed.query)
            video_id = (params.get("v") or [None])[0]
    if not video_id:
        return None, None, None
    parsed = urlparse(url)
    params = parse_qs(parsed.query)
    start = None
    end = None
    if "start" in params:
        try:
            start = int(params["start"][0])
        except (ValueError, TypeError):
            pass
    elif "t" in params:
        try:
            t_val = params["t"][0]
            start = int(t_val[:-1] if isinstance(t_val, str) and t_val.endswith("s") else t_val)
        except (ValueError, TypeError):
            pass
    if "end" in params:
        try:
            end = int(params["end"][0])
        except (ValueError, TypeError):
            pass
    return video_id, start, end


def _download_youtube_clip(url: str, output_path: str) -> Optional[str]:
    """Download YouTube clip (with start/end if present) to output_path. Returns path or None."""
    video_id, start, end = _parse_youtube_url(url)
    if not video_id:
        return None
    youtube_url = f"https://www.youtube.com/watch?v={video_id}"
    tmp_dir = tempfile.gettempdir()
    base = os.path.join(tmp_dir, f"yt_{video_id}_{os.getpid()}")
    out_tmpl = base + ".%(ext)s"

    # Prefer section download if we have both start and end
    if start is not None and end is not None and end > start:
        cmd = [
            "yt-dlp",
            "-f", "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best[height<=720]",
            "--extractor-args", "youtube:player_client=android",
            "--download-sections", f"*{start}-{end}",
            "--merge-output-format", "mp4",
            "-o", out_tmpl,
            "--no-playlist",
            youtube_url,
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if r.returncode == 0:
            for ext in ["mp4", "webm", "mkv"]:
                p = base + "." + ext
                if os.path.exists(p) and os.path.getsize(p) > 0:
                    if p != output_path:
                        try:
                            import shutil
                            shutil.move(p, output_path)
                        except Exception:
                            pass
                    if os.path.exists(output_path):
                        return output_path
                    return p

    # Fallback: full download then trim
    full_path = base + "_full.%(ext)s"
    cmd = [
        "yt-dlp",
        "-f", "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best[height<=720]",
        "--extractor-args", "youtube:player_client=android",
        "--merge-output-format", "mp4",
        "-o", full_path,
        "--no-playlist",
        youtube_url,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        return None
    actual_full = None
    for ext in ["mp4", "webm", "mkv"]:
        p = base + "_full." + ext
        if os.path.exists(p) and os.path.getsize(p) > 0:
            actual_full = p
            break
    if not actual_full:
        return None
    if start is not None and end is not None and end > start:
        duration = end - start
        trim_cmd = [
            "ffmpeg", "-y",
            "-ss", str(start),
            "-i", actual_full,
            "-t", str(duration),
            "-c", "copy",
            "-avoid_negative_ts", "make_zero",
            output_path,
        ]
        subprocess.run(trim_cmd, capture_output=True, text=True, timeout=120)
        try:
            os.remove(actual_full)
        except Exception:
            pass
        return output_path if os.path.exists(output_path) and os.path.getsize(output_path) > 0 else None
    else:
        try:
            import shutil
            shutil.move(actual_full, output_path)
        except Exception:
            pass
        return output_path if os.path.exists(output_path) else None


def _download_drive_image_bytes(drive, file_id: str) -> Optional[bytes]:
    """Download Drive file content as bytes. Works for images."""
    if not drive or not file_id:
        return None
    try:
        f = drive.CreateFile({"id": file_id})
        with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as tmp:
            tmp_path = tmp.name
        f.GetContentFile(tmp_path)
        with open(tmp_path, "rb") as h:
            data = h.read()
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        return data
    except Exception as e:
        print(f"  Warning: failed to download Drive file {file_id}: {e}")
        return None


def _download_web_image_bytes(url: str) -> Optional[bytes]:
    """Download image from HTTP URL."""
    try:
        r = requests.get(url, timeout=30, stream=True)
        r.raise_for_status()
        return r.content
    except Exception as e:
        print(f"  Warning: failed to download web image {url[:80]}: {e}")
        return None


def _image_extension_from_bytes(data: bytes) -> str:
    """Guess image extension from magic bytes."""
    if data.startswith(b"\xff\xd8\xff"):
        return "jpg"
    if data.startswith(b"\x89PNG"):
        return "png"
    if data.startswith(b"GIF"):
        return "gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    return "jpg"


def _create_drive_folder(drive, parent_id: str, title: str) -> Optional[str]:
    """Create a folder under parent_id. Returns new folder id or None."""
    try:
        meta = {
            "title": title,
            "parents": [{"id": parent_id}],
            "mimeType": "application/vnd.google-apps.folder",
        }
        folder = drive.CreateFile(meta)
        folder.Upload()
        return folder.get("id")
    except Exception as e:
        print(f"  Error creating folder '{title}': {e}")
        return None


def _upload_bytes_to_drive(drive, parent_folder_id: str, filename: str, data: bytes, mime_type: Optional[str] = None) -> bool:
    """Upload bytes as a file to the given Drive folder."""
    if not drive or not data:
        return False
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1] or "") as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        f = drive.CreateFile({"title": filename, "parents": [{"id": parent_folder_id}]})
        f.SetContentFile(tmp_path)
        f.Upload()
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        return True
    except Exception as e:
        print(f"  Error uploading '{filename}': {e}")
        return False


def _upload_file_to_drive(drive, parent_folder_id: str, local_path: str, drive_title: str) -> bool:
    """Upload a local file to the given Drive folder with the given title."""
    if not drive or not os.path.exists(local_path):
        return False
    try:
        f = drive.CreateFile({"title": drive_title, "parents": [{"id": parent_folder_id}]})
        f.SetContentFile(local_path)
        f.Upload()
        return True
    except Exception as e:
        print(f"  Error uploading '{drive_title}': {e}")
        return False


def _collect_visuals_from_sheet(sheet) -> List[Dict[str, Any]]:
    """
    For each row in Slide Chunks with final_graphics_definition, parse and return list of:
    { "slide_index_1based": int, "segment_index": int, "step_index": int, "visual_id": "S{n}V{m}", "asset": str }
    """
    _, df = get_sheet_data_and_df(sheet, "Slide Chunks")
    col = "final_graphics_definition"
    if col not in df.columns:
        return []
    out = []
    for idx, row in df.iterrows():
        raw = _safe_str(row.get(col, ""))
        if not raw or raw == "nan":
            continue
        slide_1based = int(idx) + 1
        steps = _parse_final_graphics_to_visuals(raw)
        for s in steps:
            seg_num = s["segment_num"]
            step_idx = s["step_index"]
            asset = _safe_str(s["asset"])
            if not asset:
                continue
            visual_id = f"S{seg_num}V{step_idx}"
            out.append({
                "slide_index_1based": slide_1based,
                "segment_index": seg_num,
                "step_index": step_idx,
                "visual_id": visual_id,
                "asset": asset,
            })
    return out


def run_download_assets_for_sheet(sheet, **kwargs) -> None:
    """
    Entry point for the pipeline: create a run folder under the downloadable-asset parent,
    then for each asset in final_graphics_definition download and upload as "Slide X, S{n}V{m}.ext".
    """
    worksheet_name = "Slide Chunks"
    ws, df = get_sheet_data_and_df(sheet, worksheet_name)
    _, course_info_df = get_sheet_data_and_df(sheet, "Course info")
    course_name = ""
    if not course_info_df.empty and "Course Name" in course_info_df.columns:
        course_name = _safe_str(course_info_df.loc[0, "Course Name"])
    run_folder_name = _run_folder_name(course_name)

    drive = _get_drive_instance()
    if not drive:
        print("Download assets: Could not initialize Google Drive. Skipping.")
        return

    parent_id = DOWNLOAD_ASSETS_PARENT_FOLDER_ID
    folder_id = _create_drive_folder(drive, parent_id, run_folder_name)
    if not folder_id:
        print("Download assets: Could not create run folder. Skipping.")
        return

    print(f"Download assets: created folder '{run_folder_name}'")
    visuals = _collect_visuals_from_sheet(sheet)
    total_assets = len(visuals)
    print(f"Download assets: total assets present in sheet: {total_assets}")
    if not visuals:
        print("Download assets: no visuals found in final_graphics_definition. Done.")
        return

    # Streamlit progress bar (SmartProgressBar) + terminal tqdm for observability
    smart_progress = SmartProgressBar(
        total_tasks=total_assets,
        description="Download assets",
        save_interval=0,
    )
    tqdm_bar = tqdm(total=total_assets, desc="Download assets", unit="asset")

    uploaded = 0
    try:
        for v in visuals:
            slide_x = v["slide_index_1based"]
            visual_id = v["visual_id"]
            asset = v["asset"]
            filename_base = f"Slide {slide_x}, {visual_id}"

            if _is_youtube_url(asset):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                    tmp_path = tmp.name
                try:
                    path = _download_youtube_clip(asset, tmp_path)
                    if path and os.path.exists(path) and os.path.getsize(path) > 0:
                        ok = _upload_file_to_drive(drive, folder_id, path, f"{filename_base}.mp4")
                        if ok:
                            uploaded += 1
                finally:
                    try:
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)
                    except Exception:
                        pass
            else:
                if _is_drive_url(asset):
                    file_id = _extract_drive_file_id(asset)
                    data = _download_drive_image_bytes(drive, file_id)
                else:
                    data = _download_web_image_bytes(asset)
                if data:
                    ext = _image_extension_from_bytes(data)
                    ok = _upload_bytes_to_drive(drive, folder_id, f"{filename_base}.{ext}", data)
                    if ok:
                        uploaded += 1

            # Advance both Streamlit and terminal progress bars
            smart_progress.update()
            tqdm_bar.update(1)
    finally:
        tqdm_bar.close()

    print(f"Download assets: total files created: {uploaded}")
    print(f"Download assets: uploaded to folder '{run_folder_name}'.")
