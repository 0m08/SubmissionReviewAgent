"""
Streamlit UI for Slideshow Generation
Optimized version with parallel processing for faster execution
"""

import streamlit as st
import os
import re
import json
import asyncio
import subprocess
import shutil
import sys
import importlib.util
from typing import List, Dict, Tuple, Optional, Any
from urllib.parse import urlparse, parse_qs
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO

import requests
import numpy as np
import edge_tts
import nest_asyncio
from PIL import Image, ImageDraw, ImageFont, ImageOps
from moviepy.editor import (
    VideoFileClip,
    ImageClip,
    AudioFileClip,
    CompositeVideoClip,
    ColorClip,
    CompositeAudioClip,
    concatenate_videoclips,
)
from pydrive2.drive import GoogleDrive
import gspread

nest_asyncio.apply()

# =============================================================================
# Configuration Constants
# =============================================================================

CANVAS_WIDTH = 1920
CANVAS_HEIGHT = 1080
FPS = 24
TRANSITION_DURATION = 0.3
BACKGROUND_COLOR = (0, 0, 0)
DEFAULT_PART_DURATION = 2.5

ENABLE_SUBTITLES = True
SUBTITLE_HEIGHT = 90
SUBTITLE_BG_COLOR = (0, 0, 0)
SUBTITLE_BG_OPACITY = 0.7
SUBTITLE_FONT_SIZE = 36
SUBTITLE_FONT_COLOR = (255, 255, 255)

MAX_WORKERS = 8

SLIDE_TRANSITIONS = {"slide_left", "slide_right"}

# Global caches (will be reset per run)
ASSET_CACHE: Dict[str, Dict[str, Any]] = {}
AUDIO_CACHE: Dict[str, str] = {}


# =============================================================================
# Utility Functions
# =============================================================================

def is_youtube_url(url: str) -> bool:
    return "youtube.com" in url or "youtu.be" in url


def is_drive_url(url: str) -> bool:
    return "drive.google.com" in url


def extract_drive_file_id(url: str) -> Optional[str]:
    patterns = [r"/file/d/([a-zA-Z0-9_-]+)", r"id=([a-zA-Z0-9_-]+)", r"/d/([a-zA-Z0-9_-]+)"]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def parse_youtube_url(url: str) -> Tuple[Optional[str], Optional[int], Optional[int]]:
    video_id = None
    start = None
    end = None

    if "youtu.be/" in url:
        video_id = url.split("youtu.be/")[-1].split("?")[0].split("/")[0]
    elif "youtube.com" in url:
        if "/embed/" in url:
            video_id = url.split("/embed/")[-1].split("?")[0].split("/")[0]
        elif "v=" in url:
            parsed = urlparse(url)
            params = parse_qs(parsed.query)
            video_id = params.get("v", [None])[0]

    parsed = urlparse(url)
    params = parse_qs(parsed.query)

    if "start" in params:
        start = int(params["start"][0])
    elif "t" in params:
        t_val = params["t"][0]
        start = int(t_val[:-1] if t_val.endswith("s") else t_val)

    if "end" in params:
        end = int(params["end"][0])

    return video_id, start, end


def find_downloaded_file(base_path: str) -> Optional[str]:
    if os.path.exists(base_path):
        return base_path
    for ext in [".mp4", ".webm", ".mkv", ".m4a", ""]:
        potential_path = base_path + ext
        if os.path.exists(potential_path):
            return potential_path
        if "." in base_path:
            base_no_ext = base_path.rsplit(".", 1)[0]
            potential_path = base_no_ext + ext
            if os.path.exists(potential_path):
                return potential_path
    directory = os.path.dirname(base_path)
    filename_base = os.path.basename(base_path).rsplit(".", 1)[0]
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            if filename.startswith(filename_base):
                return os.path.join(directory, filename)
    return None


def get_video_duration(video_path: str) -> float:
    """Get video duration, with caching to avoid repeated file operations."""
    if not video_path or not os.path.exists(video_path):
        return 0.0
    try:
        clip = VideoFileClip(video_path)
        duration = clip.duration
        clip.close()
        return duration
    except Exception:
        return 0.0


def get_video_durations_parallel(video_paths: List[str], max_workers: int = MAX_WORKERS) -> Dict[str, float]:
    """Get durations for multiple video files in parallel."""
    durations = {}
    
    def get_duration(path: str) -> Tuple[str, float]:
        try:
            return path, get_video_duration(path)
        except Exception:
            return path, 0.0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(get_duration, path): path for path in video_paths}
        for future in as_completed(futures):
            path, duration = future.result()
            durations[path] = duration
    
    return durations


def download_youtube_clip(url: str, output_path: str) -> Optional[str]:
    video_id, start, end = parse_youtube_url(url)
    if not video_id:
        return None
    youtube_url = f"https://www.youtube.com/watch?v={video_id}"
    temp_full_video = os.path.join(os.path.dirname(output_path), f"full_{video_id}.mp4")

    if start is not None and end is not None:
        cmd_direct = [
            "yt-dlp",
            "-f",
            "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best[height<=720]",
            "--no-playlist",
            "--download-sections",
            f"*{start}-{end}",
            "--force-keyframes-at-cuts",
            "-o",
            output_path,
            youtube_url,
        ]
        subprocess.run(cmd_direct, capture_output=True, text=True, check=False)
        found_path = find_downloaded_file(output_path)
        if found_path:
            return found_path

    cmd_full = [
        "yt-dlp",
        "-f",
        "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best[height<=720]",
        "--no-playlist",
        "--merge-output-format",
        "mp4",
        "-o",
        temp_full_video,
        youtube_url,
    ]
    subprocess.run(cmd_full, capture_output=True, text=True, check=False)
    full_video_path = find_downloaded_file(temp_full_video)
    if not full_video_path:
        return None

    if start is not None and end is not None:
        duration = end - start
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            str(start),
            "-i",
            full_video_path,
            "-t",
            str(duration),
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-avoid_negative_ts",
            "make_zero",
            output_path,
        ]
        subprocess.run(ffmpeg_cmd, capture_output=True, text=True, check=False)
        clip_path = output_path if os.path.exists(output_path) else None
    elif start is not None:
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            str(start),
            "-i",
            full_video_path,
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-avoid_negative_ts",
            "make_zero",
            output_path,
        ]
        subprocess.run(ffmpeg_cmd, capture_output=True, text=True, check=False)
        clip_path = output_path if os.path.exists(output_path) else None
    else:
        shutil.copy(full_video_path, output_path)
        clip_path = output_path if os.path.exists(output_path) else None

    try:
        os.remove(full_video_path)
    except Exception:
        pass

    return clip_path


def download_drive_image(url: str, drive_instance: GoogleDrive) -> Optional[Image.Image]:
    file_id = extract_drive_file_id(url)
    if not file_id:
        return None
    try:
        file = drive_instance.CreateFile({"id": file_id})
        temp_path = os.path.join(os.path.dirname(__file__), "temp", f"drive_{file_id}")
        os.makedirs(os.path.dirname(temp_path), exist_ok=True)
        file.GetContentFile(temp_path)
        img = Image.open(temp_path)
        img = ImageOps.exif_transpose(img).convert("RGB")
        return img
    except Exception:
        return None


def download_web_image(url: str) -> Optional[Image.Image]:
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        img = ImageOps.exif_transpose(img).convert("RGB")
        return img
    except Exception:
        return None


def load_asset(url: str, drive_instance: GoogleDrive, asset_index: int, temp_dir: str) -> Tuple[Optional[Any], str]:
    if url in ASSET_CACHE:
        cached = ASSET_CACHE[url]
        if cached.get("type") == "video" and cached.get("path") and os.path.exists(cached["path"]):
            return cached["path"], "video"
        if cached.get("type") == "image" and cached.get("data") is not None:
            return cached["data"], "image"

    if is_youtube_url(url):
        output_path = os.path.join(temp_dir, f"video_{asset_index}.mp4")
        video_path = download_youtube_clip(url, output_path)
        duration = get_video_duration(video_path) if video_path else 0.0
        if video_path:
            ASSET_CACHE[url] = {"type": "video", "path": video_path, "data": None, "duration": duration}
        return video_path, "video"

    if is_drive_url(url):
        img = download_drive_image(url, drive_instance)
        if img:
            ASSET_CACHE[url] = {"type": "image", "path": None, "data": img, "duration": 0.0}
        return img, "image"

    img = download_web_image(url)
    if img:
        ASSET_CACHE[url] = {"type": "image", "path": None, "data": img, "duration": 0.0}
    return img, "image"


def preload_assets_parallel(urls: List[str], drive_instance: GoogleDrive, temp_dir: str, max_workers: int = MAX_WORKERS) -> None:
    urls_to_download = [url for url in urls if url and url not in ASSET_CACHE]
    if not urls_to_download:
        print(f"   ✅ All {len(urls)} assets already cached")
        return

    print(f"   ⬇️  Downloading {len(urls_to_download)} new assets (cached: {len(urls) - len(urls_to_download)})")
    
    def download_single(args):
        url, index = args
        asset, asset_type = load_asset(url, drive_instance, index, temp_dir)
        return url, asset, asset_type

    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        args_list = [(url, i) for i, url in enumerate(urls_to_download)]
        futures = {executor.submit(download_single, args): args for args in args_list}
        for future in as_completed(futures):
            completed += 1
            if completed % 5 == 0 or completed == len(urls_to_download):
                print(f"   📥 Progress: {completed}/{len(urls_to_download)} assets downloaded")
            _ = future.result()
    print(f"   ✅ All assets downloaded")


# =============================================================================
# Audio Generation (Optimized with Parallel Processing)
# =============================================================================

async def _generate_narration_audio_async(text: str, output_path: str, voice: str) -> str:
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)
    return output_path


def generate_narration_audio(text: str, output_path: str, voice: str) -> str:
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(_generate_narration_audio_async(text, output_path, voice))


async def generate_narration_audio_batch(
    text_path_pairs: List[Tuple[str, str]], voice: str
) -> Dict[str, str]:
    """Generate multiple audio files in parallel using async."""
    tasks = [
        _generate_narration_audio_async(text, output_path, voice)
        for text, output_path in text_path_pairs
    ]
    results = await asyncio.gather(*tasks)
    return {output_path: output_path for output_path in results}


def generate_narration_audio_parallel(
    text_path_pairs: List[Tuple[str, str]], voice: str
) -> Dict[str, str]:
    """Generate multiple audio files in parallel."""
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(generate_narration_audio_batch(text_path_pairs, voice))


def get_audio_duration(audio_path: str) -> float:
    clip = AudioFileClip(audio_path)
    duration = clip.duration
    clip.close()
    return duration


def get_audio_durations_parallel(audio_paths: List[str], max_workers: int = MAX_WORKERS) -> Dict[str, float]:
    """Get durations for multiple audio files in parallel."""
    durations = {}
    
    def get_duration(path: str) -> Tuple[str, float]:
        try:
            return path, get_audio_duration(path)
        except Exception:
            return path, 0.0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(get_duration, path): path for path in audio_paths}
        for future in as_completed(futures):
            path, duration = future.result()
            durations[path] = duration
    
    return durations


# =============================================================================
# Parsing Functions
# =============================================================================

def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def parse_action_block(block: str) -> Optional[Dict[str, str]]:
    action_data: Dict[str, str] = {}
    for line in block.splitlines():
        line = line.strip()
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip().lower()
        value = value.strip()
        if key == "action":
            action_data["action"] = value.lower()
        elif key == "asset":
            action_data["asset"] = value
        elif key == "position":
            action_data["position"] = value.lower()
        elif key == "transition":
            transition = re.sub(r"\s+", "_", value.lower().strip()).replace("-", "_")
            action_data["transition"] = transition

    if "action" not in action_data:
        return None

    if "position" not in action_data or not action_data["position"]:
        action_data["position"] = "center"
    if "transition" not in action_data or not action_data["transition"]:
        action_data["transition"] = "none"
    return action_data


def parse_layout_instructions(layout_text: str) -> List[Dict[str, Any]]:
    if not layout_text or str(layout_text).strip().lower() == "nan":
        return []

    layout_text = str(layout_text).strip()
    if layout_text.startswith('"') and layout_text.endswith('"'):
        layout_text = layout_text[1:-1]
    if layout_text.startswith("'") and layout_text.endswith("'"):
        layout_text = layout_text[1:-1]

    wrapper_match = re.search(
        r"<layout_instructions>(.*?)</layout_instructions>",
        layout_text,
        re.DOTALL | re.IGNORECASE,
    )
    if wrapper_match:
        layout_text = wrapper_match.group(1)

    parts = re.findall(
        r"<narration_part>(.*?)</narration_part>",
        layout_text,
        re.DOTALL | re.IGNORECASE,
    )

    parsed_parts = []
    for part in parts:
        voiceover_match = re.search(r"<voiceover>(.*?)</voiceover>", part, re.DOTALL | re.IGNORECASE)
        voiceover = normalize_text(voiceover_match.group(1)) if voiceover_match else ""

        actions_blocks = re.findall(r"<action>(.*?)</action>", part, re.DOTALL | re.IGNORECASE)
        actions = []
        for block in actions_blocks:
            action = parse_action_block(block)
            if action:
                actions.append(action)

        parsed_parts.append({"voiceover": voiceover, "actions": actions})

    return parsed_parts


def extract_all_asset_urls(narration_parts: List[Dict[str, Any]]) -> List[str]:
    urls = set()
    for part in narration_parts:
        for action in part.get("actions", []):
            url = action.get("asset")
            if url:
                urls.add(url)
    return list(urls)


def get_sheet_records(sheet_link: str, gc: gspread.Client, worksheet_name: str = "Slide Chunks") -> List[Dict[str, Any]]:
    match = re.search(r"/d/([a-zA-Z0-9_-]+)", sheet_link)
    sheet_id = match.group(1) if match else sheet_link
    sheet = gc.open_by_key(sheet_id)
    worksheet = sheet.worksheet(worksheet_name)
    return worksheet.get_all_records()


def get_cached_video_duration(url: str) -> float:
    cached = ASSET_CACHE.get(url)
    if cached and cached.get("type") == "video":
        return float(cached.get("duration") or 0.0)
    if is_youtube_url(url):
        _, start, end = parse_youtube_url(url)
        if start is not None and end is not None:
            return float(max(0, end - start))
    return 0.0


# =============================================================================
# Video Composition Functions
# =============================================================================

def compute_target_geometry(clip_w: int, clip_h: int, position: str) -> Tuple[int, int, int, int]:
    canvas_w, canvas_h = CANVAS_WIDTH, CANVAS_HEIGHT

    if position == "full":
        scale = min(canvas_w / clip_w, canvas_h / clip_h)
        new_w = int(clip_w * scale)
        new_h = int(clip_h * scale)
        x = (canvas_w - new_w) // 2
        y = (canvas_h - new_h) // 2
        return new_w, new_h, x, y

    if position == "center":
        max_w = int(canvas_w * 0.8)
        max_h = int(canvas_h * 0.8)
        scale = min(max_w / clip_w, max_h / clip_h)
        new_w = int(clip_w * scale)
        new_h = int(clip_h * scale)
        x = (canvas_w - new_w) // 2
        y = (canvas_h - new_h) // 2
        return new_w, new_h, x, y

    if position == "left":
        max_w = int(canvas_w * 0.45)
        max_h = int(canvas_h * 0.8)
        scale = min(max_w / clip_w, max_h / clip_h)
        new_w = int(clip_w * scale)
        new_h = int(clip_h * scale)
        x = int(canvas_w * 0.05)
        y = (canvas_h - new_h) // 2
        return new_w, new_h, x, y

    if position == "right":
        max_w = int(canvas_w * 0.45)
        max_h = int(canvas_h * 0.8)
        scale = min(max_w / clip_w, max_h / clip_h)
        new_w = int(clip_w * scale)
        new_h = int(clip_h * scale)
        x = int(canvas_w * 0.5)
        y = (canvas_h - new_h) // 2
        return new_w, new_h, x, y

    return compute_target_geometry(clip_w, clip_h, "center")


def compute_position_for_size(clip_w: int, clip_h: int, position: str) -> Tuple[int, int]:
    canvas_w, canvas_h = CANVAS_WIDTH, CANVAS_HEIGHT
    if position in ("full", "center"):
        return (canvas_w - clip_w) // 2, (canvas_h - clip_h) // 2
    if position == "left":
        return int(canvas_w * 0.05), (canvas_h - clip_h) // 2
    if position == "right":
        return int(canvas_w * 0.5), (canvas_h - clip_h) // 2
    return compute_position_for_size(clip_w, clip_h, "center")


def position_clip_on_canvas(clip, position: str) -> Any:
    clip_w, clip_h = clip.size
    new_w, new_h, x, y = compute_target_geometry(clip_w, clip_h, position)
    clip = clip.resize((new_w, new_h))
    return clip.set_position((x, y))


def apply_slide_movement(
    clip,
    start_pos: Tuple[int, int],
    end_pos: Tuple[int, int],
    duration: float,
) -> Any:
    if start_pos == end_pos:
        return clip.set_position(end_pos)
    slide_duration = min(TRANSITION_DURATION, max(0.0, duration))
    if slide_duration <= 0:
        return clip.set_position(end_pos)

    start_x, start_y = start_pos
    end_x, end_y = end_pos

    def position_at_time(t: float) -> Tuple[float, float]:
        if t < slide_duration:
            progress = t / slide_duration
            x = start_x + (end_x - start_x) * progress
            y = start_y + (end_y - start_y) * progress
            return x, y
        return end_x, end_y

    return clip.set_position(position_at_time)


def create_image_clip(img: Any, duration: float) -> ImageClip:
    if isinstance(img, Image.Image):
        img_array = np.array(img)
    elif isinstance(img, str):
        img = Image.open(img).convert("RGB")
        img_array = np.array(img)
    else:
        img_array = np.array(img)
    return ImageClip(img_array).set_duration(duration)


def create_video_clip(video_path: str) -> Optional[VideoFileClip]:
    try:
        return VideoFileClip(video_path)
    except Exception:
        return None


def fit_video_to_duration(clip: VideoFileClip, duration: float) -> VideoFileClip:
    if clip.duration < duration:
        try:
            last_frame_time = max(clip.duration - 0.1, 0)
            last_frame = clip.to_ImageClip(t=last_frame_time).set_duration(duration - clip.duration)
            clip = concatenate_videoclips([clip, last_frame])
        except Exception:
            clip = clip.set_duration(duration)
    elif clip.duration > duration:
        clip = clip.subclip(0, duration)
    return clip


def create_subtitle_clip(text: str, duration: float, start_time: float) -> Optional[ImageClip]:
    if not ENABLE_SUBTITLES or not text:
        return None
    bar_height = SUBTITLE_HEIGHT
    subtitle_img = Image.new("RGBA", (CANVAS_WIDTH, bar_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(subtitle_img)
    bg_alpha = int(255 * SUBTITLE_BG_OPACITY)
    bg_color = (*SUBTITLE_BG_COLOR, bg_alpha)
    draw.rectangle([(0, 0), (CANVAS_WIDTH, bar_height)], fill=bg_color)

    font = None
    for font_path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        "C:/Windows/Fonts/arial.ttf",  # Windows fallback
    ]:
        if os.path.exists(font_path):
            try:
                font = ImageFont.truetype(font_path, SUBTITLE_FONT_SIZE)
                break
            except:
                continue
    if font is None:
        font = ImageFont.load_default()

    max_width = CANVAS_WIDTH - 100
    words = text.split()
    lines = []
    current_line = ""
    for word in words:
        test_line = f"{current_line} {word}".strip()
        try:
            bbox = draw.textbbox((0, 0), test_line, font=font)
            text_width = bbox[2] - bbox[0]
        except Exception:
            text_width = len(test_line) * (SUBTITLE_FONT_SIZE * 0.6)
        if text_width <= max_width:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)

    if len(lines) > 2:
        split = len(lines) // 2
        lines = [" ".join(lines[:split]), " ".join(lines[split:])]
    display_text = "\n".join(lines)

    try:
        bbox = draw.textbbox((0, 0), display_text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
    except Exception:
        text_width = len(display_text) * (SUBTITLE_FONT_SIZE * 0.6)
        text_height = SUBTITLE_FONT_SIZE * len(lines) * 1.2

    x = (CANVAS_WIDTH - text_width) // 2
    y = (bar_height - text_height) // 2
    draw.text((x, y), display_text, fill=SUBTITLE_FONT_COLOR + (255,), font=font)

    subtitle_array = np.array(subtitle_img)
    subtitle_clip = ImageClip(subtitle_array, ismask=False)
    subtitle_clip = subtitle_clip.set_duration(duration).set_start(start_time)
    subtitle_clip = subtitle_clip.set_position(("center", CANVAS_HEIGHT - bar_height))
    subtitle_clip = subtitle_clip.crossfadein(0.2).crossfadeout(0.2)
    return subtitle_clip


# =============================================================================
# Timeline and Lifecycle Functions
# =============================================================================

def build_timeline(
    narration_parts: List[Dict[str, Any]],
    voice: str,
    temp_dir: str,
    progress_callback=None,
) -> Tuple[List[Dict[str, Any]], List[AudioFileClip], float]:
    timeline = []
    audio_clips = []
    current_time = 0.0

    # PHASE 1: Generate all audio files in parallel (MAJOR OPTIMIZATION)
    print(f"   📢 Checking audio cache...")
    if progress_callback:
        progress_callback("Generating audio files in parallel...")
    text_path_pairs = []
    audio_paths_by_voiceover = {}
    
    for idx, part in enumerate(narration_parts):
        voiceover = part.get("voiceover", "")
        if voiceover:
            if voiceover in AUDIO_CACHE and os.path.exists(AUDIO_CACHE[voiceover]):
                audio_paths_by_voiceover[voiceover] = AUDIO_CACHE[voiceover]
            else:
                audio_path = os.path.join(temp_dir, f"audio_{idx}.mp3")
                text_path_pairs.append((voiceover, audio_path))
                audio_paths_by_voiceover[voiceover] = audio_path
    
    # Generate all missing audio files in parallel
    if text_path_pairs:
        print(f"   🎙️  Generating {len(text_path_pairs)} audio files in parallel...")
        generate_narration_audio_parallel(text_path_pairs, voice)
        print(f"   ✅ Audio generation complete")
        # Update cache
        for voiceover, audio_path in text_path_pairs:
            AUDIO_CACHE[voiceover] = audio_path
    else:
        print(f"   ✅ All audio files already cached")
    
    # PHASE 2: Get all audio durations in parallel
    print(f"   ⏱️  Getting audio durations...")
    if progress_callback:
        progress_callback("Getting audio durations...")
    unique_audio_paths = list(set(audio_paths_by_voiceover.values()))
    audio_durations = get_audio_durations_parallel(unique_audio_paths)
    print(f"   ✅ Retrieved durations for {len(audio_durations)} audio files")
    
    # PHASE 3: Build timeline with pre-computed durations
    if progress_callback:
        progress_callback("Building timeline...")
    for idx, part in enumerate(narration_parts):
        voiceover = part.get("voiceover", "")
        actions = part.get("actions", [])

        narration_duration = 0.0
        if voiceover:
            audio_path = audio_paths_by_voiceover[voiceover]
            narration_duration = audio_durations.get(audio_path, 0.0)
            audio_clips.append(AudioFileClip(audio_path).set_start(current_time))

        max_video_duration = 0.0
        for action in actions:
            if action.get("action") != "add":
                continue
            asset_url = action.get("asset", "")
            if asset_url and is_youtube_url(asset_url):
                max_video_duration = max(max_video_duration, get_cached_video_duration(asset_url))

        part_duration = max(narration_duration, max_video_duration)
        if part_duration <= 0:
            part_duration = DEFAULT_PART_DURATION

        timeline.append(
            {
                "voiceover": voiceover,
                "actions": actions,
                "start_time": current_time,
                "duration": part_duration,
                "end_time": current_time + part_duration,
                "narration_duration": narration_duration,
            }
        )
        current_time += part_duration

    return timeline, audio_clips, current_time


def build_asset_lifecycles(
    timeline: List[Dict[str, Any]], total_duration: float
) -> Dict[str, List[Dict[str, Any]]]:
    asset_lifecycles: Dict[str, List[Dict[str, Any]]] = {}
    current_canvas: Dict[str, Dict[str, Any]] = {}

    for part in timeline:
        part_start = part["start_time"]
        actions = part.get("actions", [])

        for action in actions:
            asset_url = action.get("asset", "")
            action_type = action.get("action", "")
            position = action.get("position", "center")
            transition = action.get("transition", "none")
            if not asset_url or not action_type:
                continue

            if action_type == "add":
                current_canvas[asset_url] = {
                    "position": position,
                    "start_time": part_start,
                    "transition_in": transition,
                    "move_from": None,
                }
            elif action_type == "keep":
                if asset_url not in current_canvas:
                    current_canvas[asset_url] = {
                        "position": position,
                        "start_time": part_start,
                        "transition_in": "none",
                        "move_from": None,
                    }
            elif action_type == "move":
                move_from = None
                if asset_url in current_canvas:
                    old_info = current_canvas[asset_url]
                    move_from = old_info.get("position")
                    lifecycle = {
                        "start": old_info["start_time"],
                        "end": part_start,
                        "position": old_info["position"],
                        "transition_in": old_info["transition_in"],
                        "transition_out": "none",
                        "move_from": old_info.get("move_from"),
                    }
                    asset_lifecycles.setdefault(asset_url, []).append(lifecycle)
                current_canvas[asset_url] = {
                    "position": position,
                    "start_time": part_start,
                    "transition_in": transition,
                    "move_from": move_from,
                }
            elif action_type == "remove":
                if asset_url in current_canvas:
                    old_info = current_canvas[asset_url]
                    lifecycle = {
                        "start": old_info["start_time"],
                        "end": part_start,
                        "position": old_info["position"],
                        "transition_in": old_info["transition_in"],
                        "transition_out": transition,
                        "move_from": old_info.get("move_from"),
                    }
                    asset_lifecycles.setdefault(asset_url, []).append(lifecycle)
                    del current_canvas[asset_url]

    for asset_url, info in current_canvas.items():
        lifecycle = {
            "start": info["start_time"],
            "end": total_duration,
            "position": info["position"],
            "transition_in": info["transition_in"],
            "transition_out": "fade_out",
            "move_from": info.get("move_from"),
        }
        asset_lifecycles.setdefault(asset_url, []).append(lifecycle)

    return asset_lifecycles


def create_visual_clips(
    asset_lifecycles: Dict[str, List[Dict[str, Any]]], drive_instance: GoogleDrive, temp_dir: str
) -> List[Any]:
    clips = []
    asset_index = 0
    for asset_url, lifecycles in asset_lifecycles.items():
        asset, asset_type = load_asset(asset_url, drive_instance, asset_index, temp_dir)
        asset_index += 1
        if asset is None:
            continue

        for lifecycle in lifecycles:
            duration = lifecycle["end"] - lifecycle["start"]
            if duration <= 0:
                continue

            if asset_type == "video":
                clip = create_video_clip(asset)
                if not clip:
                    continue
                clip = fit_video_to_duration(clip, duration)
            else:
                img = asset.copy() if isinstance(asset, Image.Image) else asset
                clip = create_image_clip(img, duration)

            clip_w, clip_h = clip.size
            target_position = lifecycle["position"]
            new_w, new_h, end_x, end_y = compute_target_geometry(clip_w, clip_h, target_position)
            clip = clip.resize((new_w, new_h))

            transition_in = lifecycle.get("transition_in")
            move_from = lifecycle.get("move_from")
            if transition_in in SLIDE_TRANSITIONS and move_from:
                start_x, start_y = compute_position_for_size(new_w, new_h, move_from)
                clip = apply_slide_movement(
                    clip,
                    (start_x, start_y),
                    (end_x, end_y),
                    duration,
                )
            else:
                clip = clip.set_position((end_x, end_y))

            if transition_in == "fade_in":
                clip = clip.crossfadein(TRANSITION_DURATION)
            if lifecycle.get("transition_out") == "fade_out":
                clip = clip.crossfadeout(TRANSITION_DURATION)

            clip = clip.set_start(lifecycle["start"])
            clips.append(clip)

    return clips


# =============================================================================
# Main Slideshow Generation Function
# =============================================================================

def generate_slideshow_from_sheet(
    sheet_link: str,
    drive_instance: GoogleDrive,
    gc: gspread.Client,
    worksheet_name: str = "Slide Chunks",
    voice: str = "en-US-GuyNeural",
    output_filename: str = "course_slideshow.mp4",
    output_dir: str = None,
    progress_callback=None,
    progress_bar=None,
) -> Optional[str]:
    """
    Generate slideshow video from Google Sheet with optimized parallel processing.
    
    Args:
        sheet_link: Google Sheets URL
        drive_instance: PyDrive GoogleDrive instance
        gc: gspread Client instance
        worksheet_name: Name of the worksheet to read
        voice: TTS voice to use
        output_filename: Name of output video file
        output_dir: Directory to save output (if None, uses temp directory)
        progress_callback: Optional function to call with progress updates
        
    Returns:
        Path to generated video file or None if failed
    """
    # Reset caches for this run
    global ASSET_CACHE, AUDIO_CACHE
    ASSET_CACHE = {}
    AUDIO_CACHE = {}
    
    # Setup temp directory
    if output_dir is None:
        import tempfile
        output_dir = tempfile.mkdtemp()
    temp_dir = os.path.join(output_dir, "temp_assets")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        # Phase 1: Read sheet data (5%)
        print("\n" + "="*60)
        print("🎬 Starting Slideshow Generation")
        print("="*60)
        if progress_callback:
            progress_callback("Reading sheet data...")
        if progress_bar:
            progress_bar.update_progress(0.05, "Reading sheet data...")
        print(f"📊 Reading sheet data from: {sheet_link}")
        print(f"📋 Worksheet: {worksheet_name}")
        records = get_sheet_records(sheet_link, gc, worksheet_name)
        print(f"✅ Found {len(records)} rows in sheet")
        narration_parts: List[Dict[str, Any]] = []

        for row in records:
            layout_instructions = row.get("layout_instructions", "")
            parts = parse_layout_instructions(layout_instructions)
            narration_parts.extend(parts)

        if not narration_parts:
            print("❌ No narration parts found in sheet.")
            if progress_callback:
                progress_callback("No narration parts found.")
            return None

        print(f"📝 Parsed {len(narration_parts)} narration parts from {len(records)} rows")
        if progress_callback:
            progress_callback(f"Processing {len(records)} rows with {len(narration_parts)} narration parts...")
        
        # Phase 2: Preload assets (10-30%)
        asset_urls = extract_all_asset_urls(narration_parts)
        print(f"\n📦 Extracted {len(asset_urls)} unique asset URLs")
        if progress_callback:
            progress_callback(f"Preloading {len(asset_urls)} assets in parallel...")
        if progress_bar:
            progress_bar.update_progress(0.10, f"Preloading {len(asset_urls)} assets...")
        print(f"⬇️  Downloading assets in parallel (max {MAX_WORKERS} workers)...")
        preload_assets_parallel(asset_urls, drive_instance, temp_dir)
        print(f"✅ Assets preloaded successfully")
        if progress_callback:
            progress_callback("Assets preloaded.")
        if progress_bar:
            progress_bar.update_progress(0.30, "Assets preloaded.")

        # Phase 3: Build timeline and generate audio (30-60%)
        print(f"\n🎙️  Generating audio files in parallel...")
        print(f"   Voice: {voice}")
        timeline, audio_clips, total_duration = build_timeline(narration_parts, voice, temp_dir, progress_callback)
        print(f"✅ Timeline built: {total_duration:.2f}s total duration")
        print(f"   Audio clips: {len(audio_clips)}")
        if progress_callback:
            progress_callback(f"Timeline built. Total duration: {total_duration:.2f}s")
        if progress_bar:
            progress_bar.update_progress(0.60, f"Timeline built ({total_duration:.1f}s total)")
        
        # Phase 4: Build asset lifecycles (60-65%)
        print(f"\n🔄 Building asset lifecycles...")
        if progress_callback:
            progress_callback("Building asset lifecycles...")
        if progress_bar:
            progress_bar.update_progress(0.65, "Building asset lifecycles...")
        asset_lifecycles = build_asset_lifecycles(timeline, total_duration)
        print(f"✅ Asset lifecycles created for {len(asset_lifecycles)} assets")

        # Phase 5: Create visual clips (65-75%)
        print(f"\n🎨 Creating visual clips...")
        if progress_callback:
            progress_callback("Creating visual clips...")
        if progress_bar:
            progress_bar.update_progress(0.70, "Creating visual clips...")
        visual_clips = create_visual_clips(asset_lifecycles, drive_instance, temp_dir)
        print(f"✅ Created {len(visual_clips)} visual clips")
        if progress_callback:
            progress_callback(f"Created {len(visual_clips)} visual clips.")
        if progress_bar:
            progress_bar.update_progress(0.75, f"Created {len(visual_clips)} visual clips")
        
        # Phase 6: Create subtitles (75-80%)
        print(f"\n📝 Creating subtitles...")
        subtitle_clips = []
        if ENABLE_SUBTITLES:
            if progress_bar:
                progress_bar.update_progress(0.77, "Creating subtitles...")
            for part in timeline:
                subtitle = create_subtitle_clip(
                    part.get("voiceover", ""),
                    part.get("narration_duration", 0),
                    part.get("start_time", 0),
                )
                if subtitle:
                    subtitle_clips.append(subtitle)
        print(f"✅ Created {len(subtitle_clips)} subtitle clips")
        if progress_bar:
            progress_bar.update_progress(0.80, "Compositing video layers...")

        # Phase 7: Composite video (80-85%)
        print(f"\n🎬 Compositing video layers...")
        print(f"   Background + {len(visual_clips)} visual clips + {len(subtitle_clips)} subtitle clips")
        if progress_callback:
            progress_callback("Compositing final video...")
        background = ColorClip(size=(CANVAS_WIDTH, CANVAS_HEIGHT), color=BACKGROUND_COLOR)
        background = background.set_duration(total_duration)

        all_layers = [background] + visual_clips + subtitle_clips
        final_video = CompositeVideoClip(all_layers, size=(CANVAS_WIDTH, CANVAS_HEIGHT))
        if audio_clips:
            final_audio = CompositeAudioClip(audio_clips)
            final_video = final_video.set_audio(final_audio)
        print(f"✅ Video composited: {CANVAS_WIDTH}x{CANVAS_HEIGHT} @ {FPS}fps")

        output_path = os.path.join(output_dir, output_filename)
        print(f"\n🎥 Rendering video to: {output_path}")
        print(f"   Codec: libx264 (preset: veryfast, threads: 4)")
        if progress_callback:
            progress_callback(f"Rendering video to {output_path}...")
        if progress_bar:
            progress_bar.update_progress(0.85, "Rendering video (this may take a while)...")
        
        # Phase 8: Render video (85-100%) - This is the longest step
        # Use 'veryfast' preset for faster encoding (trade-off: slightly larger file size)
        # logger=None disables MoviePy's internal progress bars to avoid multiple tqdm instances
        final_video.write_videofile(
            output_path,
            fps=FPS,
            codec="libx264",
            audio_codec="aac",
            preset="veryfast",
            temp_audiofile=os.path.join(temp_dir, "temp_audio.m4a"),
            remove_temp=True,
            verbose=False,
            logger=None,  # Prevents multiple progress bars
            threads=4,
        )

        if progress_bar:
            progress_bar.update_progress(0.98, "Finalizing video...")

        print(f"🧹 Cleaning up resources...")
        final_video.close()
        for clip in visual_clips:
            try:
                clip.close()
            except Exception:
                pass
        for clip in audio_clips:
            try:
                clip.close()
            except Exception:
                pass

        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"\n" + "="*60)
        print(f"✅ Video generated successfully!")
        print(f"   Path: {output_path}")
        print(f"   Size: {file_size_mb:.2f} MB")
        print(f"   Duration: {total_duration:.2f}s")
        print("="*60 + "\n")
        
        if progress_callback:
            progress_callback(f"✅ Video generated successfully: {output_path}")
        if progress_bar:
            progress_bar.update_progress(1.0, "✅ Complete!")
        
        return output_path
        
    except Exception as e:
        if progress_callback:
            progress_callback(f"❌ Error: {str(e)}")
        raise


# =============================================================================
# Streamlit UI
# =============================================================================

def main():
    """Streamlit UI for slideshow generation."""
    st.title("🎬 Slideshow Video Generator")
    st.markdown("Generate video slideshows from Google Sheets with optimized parallel processing")
    
    # Check if user is authenticated
    if "drive" not in st.session_state or "gc" not in st.session_state:
        st.error("Please log in first to access Google Drive and Sheets.")
        return
    
    drive_instance = st.session_state["drive"]
    gc = st.session_state["gc"]
    
    # Input form
    with st.form("slideshow_form"):
        st.subheader("Configuration")
        
        sheet_link = st.text_input(
            "Google Sheet Link",
            value="",
            help="Paste the full Google Sheets URL"
        )
        
        worksheet_name = st.text_input(
            "Worksheet Name",
            value="Slide Chunks",
            help="Name of the worksheet tab to read from"
        )
        
        voice = st.selectbox(
            "TTS Voice",
            options=[
                "en-US-GuyNeural",
                "en-US-JennyNeural",
                "en-US-AriaNeural",
                "en-GB-SoniaNeural",
                "en-AU-NatashaNeural",
            ],
            index=0,
            help="Text-to-speech voice for narration"
        )
        
        output_filename = st.text_input(
            "Output Filename",
            value="course_slideshow.mp4",
            help="Name for the generated video file"
        )
        
        submitted = st.form_submit_button("Generate Slideshow", type="primary")
    
    if submitted:
        if not sheet_link:
            st.error("Please provide a Google Sheet link.")
            return
        
        # Create progress tracking with time estimates
        import time
        import datetime
        
        progress_container = st.container()
        status_text = st.empty()
        progress_bar_placeholder = st.empty()
        
        # Initialize progress tracking
        start_time = time.time()
        progress_bar = progress_bar_placeholder.progress(0, text="Starting...")
        
        class ProgressTracker:
            def __init__(self, start_time, progress_bar, status_text):
                self.start_time = start_time
                self.progress_bar = progress_bar
                self.status_text = status_text
                self.progress_bar_placeholder = progress_bar_placeholder
            
            def update(self, fraction: float, message: str):
                """Update progress bar with fraction (0.0 to 1.0) and message."""
                elapsed = time.time() - self.start_time
                
                # Calculate time estimates
                if fraction > 0:
                    estimated_total = elapsed / fraction
                    remaining = estimated_total - elapsed
                    
                    elapsed_str = str(datetime.timedelta(seconds=int(elapsed))).split('.')[0]
                    remaining_str = str(datetime.timedelta(seconds=int(remaining))).split('.')[0]
                    
                    progress_text = f"{message} | {int(fraction * 100)}% | Elapsed: {elapsed_str} | Remaining: ~{remaining_str}"
                else:
                    progress_text = f"{message} | {int(fraction * 100)}% | Just started..."
                
                self.progress_bar_placeholder.progress(
                    min(1.0, max(0.0, fraction)),
                    text=progress_text
                )
                self.status_text.text(f"📌 {message}")
            
            def update_progress(self, fraction: float, message: str):
                """Alias for update() to match SmartProgressBar interface."""
                self.update(fraction, message)
        
        progress_tracker = ProgressTracker(start_time, progress_bar, status_text)
        
        def progress_callback(message: str):
            """Update progress in Streamlit UI."""
            if "progress_messages" not in st.session_state:
                st.session_state.progress_messages = []
            st.session_state.progress_messages.append(message)
        
        try:
            # Store progress messages
            if "progress_messages" not in st.session_state:
                st.session_state.progress_messages = []
            
            # Generate slideshow
            with progress_container:
                output_path = generate_slideshow_from_sheet(
                    sheet_link=sheet_link,
                    drive_instance=drive_instance,
                    gc=gc,
                    worksheet_name=worksheet_name,
                    voice=voice,
                    output_filename=output_filename,
                    progress_callback=progress_callback,
                    progress_bar=progress_tracker,
                )
            
            if output_path and os.path.exists(output_path):
                st.success("✅ Slideshow generated successfully!")
                
                # Display video
                st.subheader("Generated Video")
                with open(output_path, "rb") as video_file:
                    video_bytes = video_file.read()
                    st.video(video_bytes)
                
                # Download button
                st.download_button(
                    label="📥 Download Video",
                    data=video_bytes,
                    file_name=output_filename,
                    mime="video/mp4",
                )
                
                st.info(f"Video saved at: `{output_path}`")
            else:
                st.error("Failed to generate slideshow. Check the progress messages above.")
                
        except Exception as e:
            st.error(f"Error generating slideshow: {str(e)}")
            st.exception(e)
        
        # Show progress messages
        if st.session_state.get("progress_messages"):
            with st.expander("View Progress Log"):
                for msg in st.session_state.progress_messages:
                    st.text(msg)


# Call main() when loaded as a Streamlit page
main()

