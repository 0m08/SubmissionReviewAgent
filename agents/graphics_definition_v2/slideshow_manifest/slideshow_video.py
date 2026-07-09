"""
Manifest-driven slideshow video renderer for Graphics Definition V2.

Reads the `slideshow_manifest` column on the Slide Chunks worksheet (one
row per slide), composites the scenes per layout template with edge-tts
narration, and produces a downloadable MP4 per slide and an optional
stitched course video.

Visual goals: dark canvas, large clean media slots with rounded corners
and subtle shadow, slide title bar at top, narration subtitle bar at
bottom, smooth crossfade transitions.
"""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import subprocess
import tempfile
import textwrap
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse
from xml.etree import ElementTree as ET

import edge_tts
import nest_asyncio
import requests
import yt_dlp
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps
from moviepy.editor import (
    AudioFileClip,
    ColorClip,
    CompositeVideoClip,
    ImageClip,
    VideoFileClip,
    concatenate_videoclips,
)

try:
    nest_asyncio.apply()
except (ImportError, ValueError):
    pass


# ---------------------------------------------------------------------------
# Visual constants
# ---------------------------------------------------------------------------

CANVAS_WIDTH = 1920
CANVAS_HEIGHT = 1080
FPS = 24

BG_COLOR = (15, 20, 25)              # dark navy/charcoal canvas background
SURFACE_COLOR = (26, 32, 40)         # subtle panel color used as letterbox fill
ACCENT_COLOR = (255, 140, 0)         # orange accent (matches existing brand)
TITLE_TEXT_COLOR = (245, 246, 248)
SUBTITLE_TEXT_COLOR = (245, 246, 248)
SUBTITLE_BG_COLOR = (10, 14, 18)
SUBTITLE_BG_OPACITY = 0.78

TITLE_BAR_HEIGHT = 88
SUBTITLE_BAR_HEIGHT = 132
CONTENT_MARGIN = 56                  # gap between content area and edges
SLOT_GAP = 24                        # gap between adjacent slots
SLOT_CORNER_RADIUS = 22
SLOT_BORDER_COLOR = (60, 70, 82)
SLOT_BORDER_WIDTH = 2
SHADOW_OFFSET = 6
SHADOW_OPACITY = 110                 # 0-255

TITLE_FONT_SIZE = 36
SUBTITLE_FONT_SIZE = 30
SUBTITLE_LINE_SPACING = 8
SUBTITLE_HORIZONTAL_PAD = 80

CROSSFADE = 0.35
TAIL_PAUSE = 0.4                     # extra silence after narration before next scene
MIN_SCENE_DURATION = 1.8

DEFAULT_VOICE = "en-US-AvaNeural"

VALID_TEMPLATES = {
    "single_visual_hero": ["primary_visual"],
    "two_item_split_comparison": ["left_visual", "right_visual"],
    "multi_panel_grid": ["panel_1", "panel_2", "panel_3", "panel_4"],
    "main_plus_supporting_inset": ["main_visual", "inset_visual"],
}


# ---------------------------------------------------------------------------
# Render presets (speed vs quality)
# ---------------------------------------------------------------------------

# Mutable encoder/runtime knobs. Updated by apply_render_preset() before a run.
_ENCODE_PRESET = "ultrafast"
_ENCODE_CRF = 24
_ENCODE_THREADS = max(2, (os.cpu_count() or 4))
_AUDIO_BITRATE = "96k"
TTS_CONCURRENCY_PER_SLIDE = 40

RENDER_PRESETS = {
    "fast": {
        "width": 1280, "height": 720, "fps": 24,
        "preset": "ultrafast", "crf": 26, "threads": max(2, (os.cpu_count() or 4)),
        "audio_bitrate": "96k",
    },
    "balanced": {
        "width": 1920, "height": 1080, "fps": 24,
        "preset": "veryfast", "crf": 22, "threads": max(2, (os.cpu_count() or 4)),
        "audio_bitrate": "128k",
    },
    "high": {
        "width": 1920, "height": 1080, "fps": 30,
        "preset": "medium", "crf": 20, "threads": max(2, (os.cpu_count() or 4)),
        "audio_bitrate": "160k",
    },
}


def apply_render_preset(name):
    """
    Apply a render preset (mutates module-level encoder/canvas knobs).

    Note: This is process-global; do not invoke concurrent renders with different
    presets in the same process.

    :param name: One of "fast", "balanced", "high". Falls back to "fast" if unknown.
    :return: The selected preset dict for inspection/logging.
    """
    global CANVAS_WIDTH, CANVAS_HEIGHT, FPS
    global _ENCODE_PRESET, _ENCODE_CRF, _ENCODE_THREADS, _AUDIO_BITRATE
    cfg = RENDER_PRESETS.get(name, RENDER_PRESETS["fast"])
    CANVAS_WIDTH = int(cfg["width"])
    CANVAS_HEIGHT = int(cfg["height"])
    FPS = int(cfg["fps"])
    _ENCODE_PRESET = str(cfg["preset"])
    _ENCODE_CRF = int(cfg["crf"])
    _ENCODE_THREADS = int(cfg["threads"])
    _AUDIO_BITRATE = str(cfg["audio_bitrate"])
    return cfg


def _scale():
    """Linear scale factor relative to the 1080p design size for chrome elements."""
    return CANVAS_HEIGHT / 1080.0


def _scaled(v, minimum=1):
    """Round a base 1080p value to a sensible value at the current canvas size."""
    return max(int(minimum), int(round(v * _scale())))


# ---------------------------------------------------------------------------
# Font loading
# ---------------------------------------------------------------------------

_FONT_CANDIDATES = [
    "C:/Windows/Fonts/segoeuib.ttf",
    "C:/Windows/Fonts/segoeui.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "/Library/Fonts/HelveticaNeue.ttc",
    "/Library/Fonts/Helvetica.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def _load_font(size, bold=False):
    """
    Load a TrueType font with the given size, falling back across platforms.

    :param size: Pixel size of the font.
    :param bold: Prefer bold faces if available.
    :return: PIL ImageFont instance.
    """
    candidates = list(_FONT_CANDIDATES)
    if not bold:
        candidates = [c for c in candidates if "Bold" not in c and "segoeuib" not in c] + candidates
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


# ---------------------------------------------------------------------------
# URL helpers
# ---------------------------------------------------------------------------

def is_youtube_url(url):
    return "youtube.com" in (url or "") or "youtu.be" in (url or "")


def is_drive_url(url):
    return "drive.google.com" in (url or "")


def extract_drive_file_id(url):
    """Return the Drive file id from any common Drive URL form, or None."""
    if not url:
        return None
    patterns = [
        r"/file/d/([a-zA-Z0-9_-]+)",
        r"id=([a-zA-Z0-9_-]+)",
        r"/d/([a-zA-Z0-9_-]+)",
        r"drive\.google\.com/open\?id=([a-zA-Z0-9_-]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, url, re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def parse_youtube_url(url):
    """
    Parse a YouTube URL into (video_id, start_seconds, end_seconds).

    :param url: A youtu.be, youtube.com/watch, or youtube.com/embed URL with optional
        ?start= and ?end= query parameters.
    :return: Tuple where any element may be None when not available.
    """
    if not url:
        return None, None, None
    video_id = None
    if "youtu.be/" in url:
        video_id = url.split("youtu.be/")[-1].split("?")[0].split("/")[0]
    elif "youtube.com" in url:
        if "/embed/" in url:
            video_id = url.split("/embed/")[-1].split("?")[0].split("/")[0]
        elif "v=" in url:
            params = parse_qs(urlparse(url).query)
            video_id = params.get("v", [None])[0]
    params = parse_qs(urlparse(url).query)
    start = None
    end = None
    if "start" in params:
        try:
            start = int(params["start"][0])
        except Exception:
            start = None
    elif "t" in params:
        t_val = params["t"][0]
        try:
            start = int(t_val[:-1] if t_val.endswith("s") else t_val)
        except Exception:
            start = None
    if "end" in params:
        try:
            end = int(params["end"][0])
        except Exception:
            end = None
    return video_id, start, end


# ---------------------------------------------------------------------------
# Asset download
# ---------------------------------------------------------------------------

_REQ_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; rv:91.0) Gecko/20100101 Firefox/91.0"
}


def download_drive_image(url, drive=None):
    """
    Download a Google Drive-hosted image as a PIL Image.

    Tries the public export URL first (works for "anyone with link"), then
    falls back to PyDrive2 when an authenticated drive client is provided.

    :param url: Drive URL.
    :param drive: Optional PyDrive2 GoogleDrive client.
    :return: PIL.Image.RGB or None on failure.
    """
    file_id = extract_drive_file_id(url)
    if not file_id:
        return None
    export_url = f"https://drive.google.com/uc?export=view&id={file_id}"
    try:
        resp = requests.get(export_url, headers=_REQ_HEADERS, timeout=30, allow_redirects=True)
        resp.raise_for_status()
        ctype = (resp.headers.get("Content-Type") or "").lower()
        if ctype.startswith("image/"):
            img = Image.open(BytesIO(resp.content))
            return ImageOps.exif_transpose(img).convert("RGB")
    except Exception:
        pass
    if drive is not None:
        tmp_path = None
        try:
            f = drive.CreateFile({"id": file_id})
            with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as tmp:
                tmp_path = tmp.name
            f.GetContentFile(tmp_path)
            with open(tmp_path, "rb") as fp:
                data = fp.read()
            img = Image.open(BytesIO(data))
            return ImageOps.exif_transpose(img).convert("RGB")
        except Exception as e:
            print(f"  [drive] PyDrive download failed for {file_id[:14]}...: {e}")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
    return None


def download_web_image(url):
    """Download an image from a public HTTPS URL into a PIL Image, or None."""
    try:
        resp = requests.get(url, headers=_REQ_HEADERS, timeout=30)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content))
        return ImageOps.exif_transpose(img).convert("RGB")
    except Exception:
        return None


def _ffmpeg_get_duration(path):
    if not path or not os.path.exists(path):
        return 0.0
    try:
        clip = VideoFileClip(path)
        d = float(clip.duration or 0.0)
        clip.close()
        return d
    except Exception:
        return 0.0


def download_youtube_clip(url, output_path, max_seconds=60):
    """
    Download a YouTube video clip with optional start/end seconds via yt-dlp.

    :param url: YouTube URL (watch / embed / youtu.be), optionally with ?start=&end=.
    :param output_path: Target mp4 path.
    :param max_seconds: Hard cap on requested clip length when end is missing.
    :return: Path to mp4 on disk, or None on failure.
    """
    video_id, start, end = parse_youtube_url(url)
    if not video_id:
        return None
    out_dir = os.path.dirname(output_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(output_path))[0]

    # Bound the clip to keep download fast and predictable
    if start is not None and end is None:
        end = start + max_seconds
    if start is not None and end is not None and (end - start) > max_seconds:
        end = start + max_seconds

    yt_url = f"https://www.youtube.com/watch?v={video_id}"
    base_opts = {
        "quiet": True,
        "no_warnings": True,
        "ignoreerrors": False,
        "extractor_args": {"youtube": {"player_client": ["android", "web"], "skip": ["dash"]}},
        "merge_output_format": "mp4",
        "format": "bestvideo[ext=mp4][height<=1080]+bestaudio[ext=m4a]/best[ext=mp4]/best",
    }
    section = None
    if start is not None and end is not None:
        section = f"*{start}-{end}"
    full_template = os.path.join(out_dir, f"full_{base}.%(ext)s")
    section_template = os.path.join(out_dir, f"{base}.%(ext)s")

    if section:
        try:
            opts = {
                **base_opts,
                "outtmpl": section_template,
                "download_sections": [section],
                "force_keyframes_at_cuts": True,
            }
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([yt_url])
        except Exception as e:
            print(f"  [yt-dlp] section extraction failed for {video_id}: {e}")

    candidate = output_path if os.path.exists(output_path) else None
    if not candidate:
        for ext in (".mp4", ".webm", ".mkv"):
            p = os.path.join(out_dir, f"{base}{ext}")
            if os.path.exists(p):
                candidate = p
                break

    if candidate and _ffmpeg_get_duration(candidate) > 0:
        if not candidate.endswith(".mp4"):
            remuxed = output_path
            cmd = ["ffmpeg", "-y", "-i", candidate, "-c", "copy", "-movflags", "+faststart", remuxed]
            subprocess.run(cmd, capture_output=True, check=False)
            if os.path.exists(remuxed) and _ffmpeg_get_duration(remuxed) > 0:
                return remuxed
        return candidate

    # Fallback: full download then trim
    try:
        opts = {**base_opts, "outtmpl": full_template}
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([yt_url])
    except Exception as e:
        print(f"  [yt-dlp] full download failed for {video_id}: {e}")
        return None

    full_path = None
    for ext in (".mp4", ".webm", ".mkv"):
        p = os.path.join(out_dir, f"full_{base}{ext}")
        if os.path.exists(p):
            full_path = p
            break
    if not full_path:
        return None

    if start is None and end is None:
        if full_path != output_path:
            shutil.copy(full_path, output_path)
        return output_path

    duration = (end - start) if end is not None else max_seconds
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start or 0),
        "-i", full_path,
        "-t", str(duration),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "aac",
        "-movflags", "+faststart",
        output_path,
    ]
    res = subprocess.run(cmd, capture_output=True, check=False)
    try:
        os.remove(full_path)
    except Exception:
        pass
    if res.returncode == 0 and os.path.exists(output_path) and _ffmpeg_get_duration(output_path) > 0:
        return output_path
    return None


# ---------------------------------------------------------------------------
# Manifest parsing and normalization
# ---------------------------------------------------------------------------

def _normalize_attribute_quotes(text):
    """
    Replace doubled quotes in attribute positions (sheet CSV escape) with single quotes.

    Cells exported from Sheets/Excel sometimes look like `id=""1""`. ElementTree
    will reject that. We rewrite to `id="1"` only inside attribute regions.
    """
    if not text:
        return text
    return re.sub(r'""([^"<>]*)""', r'"\1"', text)


def _escape_bare_ampersands(text):
    """Escape ``&`` characters that are not already part of a valid XML entity."""
    if not text:
        return text
    return re.sub(r"&(?!(amp|lt|gt|quot|apos|#\d+|#x[0-9A-Fa-f]+);)", "&amp;", text)


def parse_slideshow_manifest(xml_text):
    """
    Parse the inner slideshow_manifest XML into a list of scene dicts.

    :param xml_text: Either an outer `<slideshow_manifest>...</slideshow_manifest>` block
        or just the inner scenes (the agent stores the inner content).
    :return: List of dicts with keys: id, template, narration, slots (list of {role, asset}).
    """
    if not xml_text or not str(xml_text).strip():
        return []
    text = str(xml_text).strip()
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]
    text = _normalize_attribute_quotes(text)
    if "<slideshow_manifest" not in text:
        text = f"<slideshow_manifest>\n{text}\n</slideshow_manifest>"
    safe_text = _escape_bare_ampersands(text)
    try:
        root = ET.fromstring(safe_text)
    except ET.ParseError:
        scenes = []
        for match in re.finditer(r"<scene\b[^>]*>.*?</scene>", text, re.DOTALL | re.IGNORECASE):
            scene_block = match.group(0)
            parsed = _parse_scene_via_regex(scene_block)
            if parsed:
                scenes.append(parsed)
        return scenes

    scenes = []
    for idx, scene_el in enumerate(root.findall("scene"), start=1):
        scene_id = (scene_el.get("id") or str(idx)).strip()
        template = (scene_el.get("template") or "").strip()
        narration_el = scene_el.find("narration_span")
        narration = (narration_el.text or "").strip() if narration_el is not None else ""
        slots = []
        for slot_el in scene_el.findall("slot"):
            role = (slot_el.get("role") or "").strip()
            asset = (slot_el.get("asset") or "").strip()
            if role and asset:
                slots.append({"role": role, "asset": asset})
        if template and slots:
            scenes.append({
                "id": scene_id,
                "template": template,
                "narration": narration,
                "slots": slots,
            })
    return scenes


def _parse_scene_via_regex(scene_block):
    template_match = re.search(r'template\s*=\s*"([^"]+)"', scene_block, re.IGNORECASE)
    id_match = re.search(r'\bid\s*=\s*"([^"]+)"', scene_block, re.IGNORECASE)
    narration_match = re.search(r"<narration_span>(.*?)</narration_span>", scene_block, re.DOTALL | re.IGNORECASE)
    if not template_match:
        return None
    slots = []
    slot_pattern = re.compile(r"<slot\b\s+([^>]*?)\s*/?>", re.IGNORECASE | re.DOTALL)
    for slot_match in slot_pattern.finditer(scene_block):
        attrs = slot_match.group(1)
        role_match = re.search(r'role\s*=\s*"([^"]+)"', attrs, re.IGNORECASE)
        asset_match = re.search(r'asset\s*=\s*"([^"]+)"', attrs, re.IGNORECASE)
        if role_match and asset_match:
            slots.append({"role": role_match.group(1).strip(), "asset": asset_match.group(1).strip()})
    if not slots:
        return None
    return {
        "id": (id_match.group(1) if id_match else "1").strip(),
        "template": template_match.group(1).strip(),
        "narration": (narration_match.group(1).strip() if narration_match else ""),
        "slots": slots,
    }


# ---------------------------------------------------------------------------
# Layout: slot rectangles per template
# ---------------------------------------------------------------------------

def _content_rect():
    """Return (x, y, w, h) of the area available for slots between title and subtitle bars."""
    title_h = _scaled(TITLE_BAR_HEIGHT)
    sub_h = _scaled(SUBTITLE_BAR_HEIGHT)
    margin = _scaled(CONTENT_MARGIN)
    x = margin
    y = title_h + margin
    w = CANVAS_WIDTH - 2 * margin
    h = CANVAS_HEIGHT - title_h - sub_h - 2 * margin
    return x, y, w, h


def compute_slot_rectangles(template, slots):
    """
    Return ordered list of (x, y, w, h) rectangles aligned with `slots` for the template.

    For unknown templates, returns a single hero-sized rect repeated for the slot count.

    :param template: Template id from VALID_TEMPLATES.
    :param slots: List of slot dicts in order.
    :return: List of (x, y, w, h) tuples, one per slot.
    """
    cx, cy, cw, ch = _content_rect()
    n = len(slots)

    gap = _scaled(SLOT_GAP)

    if template == "two_item_split_comparison" and n == 2:
        half = (cw - gap) // 2
        return [
            (cx, cy, half, ch),
            (cx + half + gap, cy, half, ch),
        ]

    if template == "multi_panel_grid" and n in (3, 4):
        if n == 3:
            third = (cw - 2 * gap) // 3
            return [
                (cx + i * (third + gap), cy, third, ch)
                for i in range(3)
            ]
        cell_w = (cw - gap) // 2
        cell_h = (ch - gap) // 2
        return [
            (cx, cy, cell_w, cell_h),
            (cx + cell_w + gap, cy, cell_w, cell_h),
            (cx, cy + cell_h + gap, cell_w, cell_h),
            (cx + cell_w + gap, cy + cell_h + gap, cell_w, cell_h),
        ]

    if template == "main_plus_supporting_inset" and n == 2:
        main_w = int(cw * 0.66)
        inset_w = cw - main_w - gap
        return [
            (cx, cy, main_w, ch),
            (cx + main_w + gap, cy, inset_w, ch),
        ]

    # single_visual_hero / fallback
    return [(cx, cy, cw, ch) for _ in range(max(1, n))]


# ---------------------------------------------------------------------------
# PIL panel rendering (background, title bar, subtitle, slot frames)
# ---------------------------------------------------------------------------

def _draw_title_bar(img, slide_title):
    """Paint the title bar onto an RGBA image in-place."""
    title_h = _scaled(TITLE_BAR_HEIGHT)
    accent_w = _scaled(8, minimum=4)
    title_font_size = max(20, _scaled(TITLE_FONT_SIZE))

    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, accent_w, title_h), fill=ACCENT_COLOR)
    draw.rectangle((0, title_h - 2, CANVAS_WIDTH, title_h), fill=(35, 42, 50))

    title_text = (slide_title or "").strip()
    if not title_text:
        return
    if len(title_text) > 90:
        title_text = title_text[:89] + "…"
    title_font = _load_font(title_font_size, bold=True)
    pad_x = _scaled(28, minimum=12)
    draw.text(
        (pad_x, (title_h - title_font_size) // 2 - 4),
        title_text,
        font=title_font,
        fill=TITLE_TEXT_COLOR,
    )


def _draw_panel(img, x, y, w, h):
    """Draw a rounded surface panel with a soft drop shadow at (x, y, w, h) onto img."""
    radius = _scaled(SLOT_CORNER_RADIUS, minimum=8)
    shadow_off = _scaled(SHADOW_OFFSET, minimum=2)
    border_w = max(1, _scaled(SLOT_BORDER_WIDTH))

    shadow = Image.new("RGBA", (w + shadow_off * 4, h + shadow_off * 4), (0, 0, 0, 0))
    sdraw = ImageDraw.Draw(shadow)
    sdraw.rounded_rectangle(
        (shadow_off * 2, shadow_off * 2, shadow_off * 2 + w, shadow_off * 2 + h),
        radius=radius,
        fill=(0, 0, 0, SHADOW_OPACITY),
    )
    shadow = shadow.filter(ImageFilter.GaussianBlur(shadow_off * 1.5))
    img.alpha_composite(shadow, (x - shadow_off * 2 + shadow_off, y - shadow_off * 2 + shadow_off))

    panel = Image.new("RGBA", (w, h), SURFACE_COLOR + (255,))
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, w, h), radius=radius, fill=255)
    panel.putalpha(mask)
    img.alpha_composite(panel, (x, y))

    border = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(border).rounded_rectangle(
        (0, 0, w - 1, h - 1),
        radius=radius,
        outline=SLOT_BORDER_COLOR + (255,),
        width=border_w,
    )
    img.alpha_composite(border, (x, y))


def _wrap_subtitle(text, font, max_width):
    if not text:
        return []
    words = text.split()
    lines = []
    current = ""
    for w in words:
        candidate = (current + " " + w).strip()
        bbox = font.getbbox(candidate)
        if bbox[2] - bbox[0] <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = w
    if current:
        lines.append(current)
    return lines


def _draw_subtitle_bar(img, narration):
    """Paint the bottom narration subtitle bar onto an RGBA image in-place."""
    sub_h = _scaled(SUBTITLE_BAR_HEIGHT)
    sub_font_size = max(18, _scaled(SUBTITLE_FONT_SIZE))
    line_spacing = _scaled(SUBTITLE_LINE_SPACING, minimum=4)
    horizontal_pad = _scaled(SUBTITLE_HORIZONTAL_PAD, minimum=24)

    bar_top = CANVAS_HEIGHT - sub_h
    bg = Image.new(
        "RGBA",
        (CANVAS_WIDTH, sub_h),
        SUBTITLE_BG_COLOR + (int(255 * SUBTITLE_BG_OPACITY),),
    )
    img.alpha_composite(bg, (0, bar_top))

    if not narration:
        return

    font = _load_font(sub_font_size, bold=False)
    max_width = CANVAS_WIDTH - 2 * horizontal_pad
    lines = _wrap_subtitle(narration.replace("\n", " ").strip(), font, max_width)
    if len(lines) > 3:
        lines = lines[:3]
        lines[-1] = lines[-1].rstrip(".") + "…"

    line_h = sub_font_size + line_spacing
    total_h = len(lines) * line_h - line_spacing
    y = bar_top + (sub_h - total_h) // 2

    draw = ImageDraw.Draw(img)
    for line in lines:
        bbox = font.getbbox(line)
        line_w = bbox[2] - bbox[0]
        x = (CANVAS_WIDTH - line_w) // 2
        draw.text((x, y), line, font=font, fill=SUBTITLE_TEXT_COLOR)
        y += line_h


def _draw_image_into_panel(img, asset_image, x, y, w, h):
    """Paste a fitted+rounded image into a panel rect on the RGBA canvas in-place."""
    radius = _scaled(SLOT_CORNER_RADIUS, minimum=8)
    fitted = _fit_image_into_rect(asset_image, w, h)
    rounded = _round_pil_image(fitted, radius)
    img.alpha_composite(rounded, (x, y))


def _draw_missing_asset(img, x, y, w, h):
    """Stamp an 'Asset unavailable' message inside a panel rect."""
    radius = _scaled(SLOT_CORNER_RADIUS, minimum=8)
    placeholder = Image.new("RGB", (w, h), SURFACE_COLOR)
    draw = ImageDraw.Draw(placeholder)
    font = _load_font(max(16, _scaled(28)))
    msg = "Asset unavailable"
    bbox = font.getbbox(msg)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    draw.text(((w - tw) // 2, (h - th) // 2), msg, font=font, fill=(180, 185, 195))
    rounded = _round_pil_image(placeholder, radius)
    img.alpha_composite(rounded, (x, y))


def _make_baked_scene_image(slide_title, slot_rects, asset_results, narration):
    """
    Bake a single static RGB image for a scene: background + title + slot panels +
    image-slot content + subtitle. Video slots are left as the panel surface so a
    VideoFileClip can be overlaid on top later.

    Returns a numpy RGB array suitable for ImageClip(transparent=False).
    """
    img = Image.new("RGBA", (CANVAS_WIDTH, CANVAS_HEIGHT), BG_COLOR + (255,))
    _draw_title_bar(img, slide_title)
    for (x, y, w, h) in slot_rects:
        _draw_panel(img, x, y, w, h)
    for (rect, asset) in zip(slot_rects, asset_results):
        if not rect:
            continue
        x, y, w, h = rect
        if asset is None:
            _draw_missing_asset(img, x, y, w, h)
            continue
        kind = asset.get("kind")
        if kind == "image" and asset.get("image") is not None:
            _draw_image_into_panel(img, asset["image"], x, y, w, h)
        elif kind == "video" and asset.get("video_path"):
            # Leave the panel surface; the moving VideoFileClip will overlay it.
            pass
        else:
            _draw_missing_asset(img, x, y, w, h)
    _draw_subtitle_bar(img, narration)
    return _pil_rgb_to_array(img.convert("RGB"))


# ---------------------------------------------------------------------------
# Image fitting and slot clip construction
# ---------------------------------------------------------------------------

def _fit_image_into_rect(pil_image, rect_w, rect_h):
    """Resize a PIL image to fit inside (rect_w, rect_h) preserving aspect ratio (letterboxed)."""
    if pil_image is None:
        return Image.new("RGB", (rect_w, rect_h), SURFACE_COLOR)
    src = pil_image.copy()
    src.thumbnail((rect_w, rect_h), Image.LANCZOS)
    canvas = Image.new("RGB", (rect_w, rect_h), SURFACE_COLOR)
    px = (rect_w - src.width) // 2
    py = (rect_h - src.height) // 2
    canvas.paste(src, (px, py))
    return canvas


def _round_pil_image(pil_image, radius):
    """Apply a rounded-corner alpha mask to a PIL RGB image, returning RGBA."""
    rgba = pil_image.convert("RGBA")
    mask = Image.new("L", rgba.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, rgba.size[0], rgba.size[1]), radius=radius, fill=255)
    rgba.putalpha(mask)
    return rgba


# ---------------------------------------------------------------------------
# TTS
# ---------------------------------------------------------------------------

async def _edge_tts_save(text, voice, output_path):
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)
    return output_path


def _audio_duration_seconds(path):
    """Cheap audio-duration probe via ffprobe with a moviepy fallback."""
    if not path or not os.path.exists(path):
        return 0.0
    try:
        res = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                path,
            ],
            capture_output=True, check=False, text=True,
        )
        out = (res.stdout or "").strip()
        if out:
            return float(out)
    except Exception:
        pass
    try:
        clip = AudioFileClip(path)
        d = float(clip.duration or 0.0)
        clip.close()
        return d
    except Exception:
        return 0.0


def synthesize_narration(text, output_path, voice=DEFAULT_VOICE):
    """
    Generate an mp3 narration file from text using edge-tts.

    :param text: Narration text. Falls back to silent placeholder when empty.
    :param output_path: Target mp3 path.
    :param voice: edge-tts voice id (e.g. "en-US-AvaNeural").
    :return: Tuple (audio_path, duration_seconds). Returns (None, 0.0) on failure.
    """
    text = (text or "").strip()
    if not text:
        return None, 0.0
    try:
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(_edge_tts_save(text, voice, output_path))
        finally:
            loop.close()
    except Exception as e:
        print(f"  [tts] edge-tts failed: {e}")
        return None, 0.0
    if not os.path.exists(output_path):
        return None, 0.0
    return output_path, _audio_duration_seconds(output_path)


def synthesize_narrations_batch(jobs, voice=DEFAULT_VOICE):
    """
    Run many edge-tts requests concurrently in a single asyncio event loop.

    :param jobs: List of (text, output_path) tuples.
    :param voice: edge-tts voice id.
    :return: Dict mapping output_path -> (audio_path or None, duration_seconds).
    """
    results = {}
    if not jobs:
        return results

    semaphore = asyncio.Semaphore(max(1, int(TTS_CONCURRENCY_PER_SLIDE)))

    async def _run_one(text, out_path):
        text = (text or "").strip()
        if not text:
            return out_path, None
        try:
            async with semaphore:
                await _edge_tts_save(text, voice, out_path)
        except Exception as e:
            print(f"  [tts-batch] edge-tts failed for {out_path}: {e}")
            return out_path, None
        if not os.path.exists(out_path):
            return out_path, None
        return out_path, out_path

    async def _runner():
        return await asyncio.gather(*[_run_one(t, p) for (t, p) in jobs])

    try:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                raise RuntimeError("loop closed")
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        outputs = loop.run_until_complete(_runner())
    except Exception as e:
        print(f"  [tts-batch] batch failed, falling back to serial: {e}")
        outputs = []
        for text, out_path in jobs:
            audio_path, _ = synthesize_narration(text, out_path, voice=voice)
            outputs.append((out_path, audio_path))

    for out_path, audio_path in outputs:
        if audio_path:
            results[out_path] = (audio_path, _audio_duration_seconds(audio_path))
        else:
            results[out_path] = (None, 0.0)
    return results


# ---------------------------------------------------------------------------
# Scene composition
# ---------------------------------------------------------------------------

class AssetCache:
    """Thread-safe in-memory cache keyed by URL for a single render run."""

    def __init__(self):
        self._lock = threading.Lock()
        self._store = {}

    def get(self, url):
        with self._lock:
            return self._store.get(url)

    def set(self, url, value):
        with self._lock:
            self._store[url] = value

    def __contains__(self, url):
        with self._lock:
            return url in self._store


def _load_asset_for_slot(asset_url, drive, temp_dir, slot_index, cache=None):
    """
    Resolve an asset URL into a usable artifact for compositing.

    :param cache: Optional AssetCache shared across scenes/slides in a run.
    :return: dict { "kind": "image"|"video"|"missing", "image": PIL or None, "video_path": str or None }
    """
    if not asset_url:
        return {"kind": "missing", "image": None, "video_path": None}

    if cache is not None:
        cached = cache.get(asset_url)
        if cached is not None:
            if cached.get("kind") == "video" and cached.get("video_path") and os.path.exists(cached["video_path"]):
                return cached
            if cached.get("kind") == "image":
                return cached
            if cached.get("kind") == "missing":
                return cached

    if is_youtube_url(asset_url):
        out_path = os.path.join(temp_dir, f"slot_{slot_index}.mp4")
        path = download_youtube_clip(asset_url, out_path)
        result = (
            {"kind": "video", "image": None, "video_path": path}
            if path else
            {"kind": "missing", "image": None, "video_path": None}
        )
    elif is_drive_url(asset_url):
        img = download_drive_image(asset_url, drive)
        result = {"kind": "image" if img else "missing", "image": img, "video_path": None}
    else:
        img = download_web_image(asset_url)
        result = {"kind": "image" if img else "missing", "image": img, "video_path": None}

    if cache is not None:
        cache.set(asset_url, result)
    return result


def _build_video_overlay_clip(asset, rect, duration):
    """
    Build a moviepy video clip for a video slot, sized + positioned to overlay the
    panel surface that has already been baked into the scene's static image.

    Timing rules:
      - clip_duration >= duration: trim to `duration`.
      - clip_duration <  duration: play full clip then freeze last frame to fill `duration`.
    """
    x, y, w, h = rect
    path = asset["video_path"]
    try:
        vc = VideoFileClip(path).without_audio()
    except Exception as e:
        print(f"  [slot] failed to open video {path}: {e}")
        return None

    src_w, src_h = vc.size
    if not src_w or not src_h:
        try:
            vc.close()
        except Exception:
            pass
        return None

    scale = min(w / src_w, h / src_h)
    new_w = max(2, int(src_w * scale))
    new_h = max(2, int(src_h * scale))
    px = x + (w - new_w) // 2
    py = y + (h - new_h) // 2

    try:
        resized = vc.resize(newsize=(new_w, new_h))
    except Exception:
        resized = vc

    vd = float(resized.duration or 0.0)
    if vd <= 0:
        try:
            vc.close()
        except Exception:
            pass
        return None

    if vd >= duration:
        composed = resized.subclip(0, duration)
    else:
        try:
            last_frame = resized.to_ImageClip(t=max(0.0, vd - 0.05)).set_duration(duration - vd)
        except Exception:
            last_frame = resized.to_ImageClip(t=0).set_duration(duration - vd)
        composed = concatenate_videoclips([resized, last_frame], method="compose")

    return composed.set_position((px, py))


def _pil_rgba_to_array(pil_image):
    """Convert a PIL RGBA image to a numpy array suitable for ImageClip(transparent=True)."""
    import numpy as np
    return np.array(pil_image.convert("RGBA"))


def _pil_rgb_to_array(pil_image):
    import numpy as np
    return np.array(pil_image.convert("RGB"))


def build_scene_clip(
    scene,
    slide_title,
    drive,
    voice,
    temp_dir,
    scene_index,
    cache=None,
    presynth_audio=None,
):
    """
    Build a single scene clip. Static visuals (background, title, slot panels,
    image content, subtitle) are baked into ONE ImageClip. Only video slots are
    overlaid as moviepy clips. This dramatically cuts per-frame compositing cost.

    :param presynth_audio: Optional dict produced by synthesize_narrations_batch
        keyed by the same audio_path used here. When present, TTS is not rerun.
    :param cache: Optional AssetCache shared across scenes/slides in a run.
    """
    slots = scene.get("slots", [])
    template = scene.get("template", "single_visual_hero")
    rects = compute_slot_rectangles(template, slots)
    narration = scene.get("narration", "")

    audio_path_target = os.path.join(temp_dir, f"scene_{scene_index}.mp3")
    if presynth_audio is not None and audio_path_target in presynth_audio:
        audio_path, audio_duration = presynth_audio[audio_path_target]
    else:
        audio_path, audio_duration = synthesize_narration(narration, audio_path_target, voice=voice)

    asset_results = [None] * len(slots)
    if slots:
        with ThreadPoolExecutor(max_workers=min(8, max(1, len(slots)))) as ex:
            futures = {
                ex.submit(
                    _load_asset_for_slot,
                    slot["asset"],
                    drive,
                    temp_dir,
                    f"{scene_index}_{i}",
                    cache,
                ): i
                for i, slot in enumerate(slots)
            }
            for fut in as_completed(futures):
                i = futures[fut]
                try:
                    asset_results[i] = fut.result()
                except Exception as e:
                    print(f"  [scene] slot {i} asset load failed: {e}")
                    asset_results[i] = {"kind": "missing", "image": None, "video_path": None}

    has_video_slot = any(a and a.get("kind") == "video" for a in asset_results)
    longest_video = 0.0
    if has_video_slot:
        for a in asset_results:
            if a and a.get("kind") == "video" and a.get("video_path"):
                longest_video = max(longest_video, _ffmpeg_get_duration(a["video_path"]))

    if audio_duration > 0:
        scene_duration = max(audio_duration + TAIL_PAUSE, MIN_SCENE_DURATION)
    else:
        scene_duration = max(longest_video, MIN_SCENE_DURATION)

    base_arr = _make_baked_scene_image(slide_title, rects, asset_results, narration)
    base = ImageClip(base_arr, transparent=False).set_duration(scene_duration).set_position((0, 0))

    video_overlays = []
    for i, slot in enumerate(slots):
        a = asset_results[i] or {"kind": "missing"}
        if a.get("kind") == "video" and a.get("video_path"):
            ov = _build_video_overlay_clip(a, rects[i], scene_duration)
            if ov is not None:
                video_overlays.append(ov.set_duration(scene_duration))

    if video_overlays:
        final = CompositeVideoClip(
            [base] + video_overlays,
            size=(CANVAS_WIDTH, CANVAS_HEIGHT),
        ).set_duration(scene_duration)
    else:
        final = base

    if audio_path:
        try:
            final = final.set_audio(AudioFileClip(audio_path))
        except Exception as e:
            print(f"  [scene] failed to attach narration: {e}")

    return final


# ---------------------------------------------------------------------------
# Slide rendering (concatenate scenes) and full course rendering
# ---------------------------------------------------------------------------

def render_slide_video(
    slide_index,
    slide_title,
    manifest_xml,
    output_path,
    drive=None,
    voice=DEFAULT_VOICE,
    temp_root=None,
    cache=None,
):
    """
    Render the manifest for one slide row to an MP4.

    :param slide_index: 1-based row index used in default file names.
    :param slide_title: Slide title shown in the top bar.
    :param manifest_xml: Inner slideshow_manifest XML for this slide.
    :param output_path: Destination MP4 path.
    :param drive: Optional PyDrive2 client used to authenticate Drive image fetches.
    :param voice: edge-tts voice id.
    :param temp_root: Optional directory for intermediate files; auto-created when None.
    :param cache: Optional AssetCache shared across slides in a run.
    :return: Path to MP4 on success, or None on failure (e.g. invalid manifest).
    """
    scenes = parse_slideshow_manifest(manifest_xml)
    if not scenes:
        print(f"  [slide {slide_index}] no scenes parsed from manifest; skipping.")
        return None

    own_temp = False
    if temp_root is None:
        temp_root = tempfile.mkdtemp(prefix=f"slide_{slide_index}_")
        own_temp = True
    os.makedirs(temp_root, exist_ok=True)

    try:
        # Pre-synthesize ALL narrations for this slide concurrently so TTS latency
        # overlaps and we don't pay it serially per scene.
        tts_jobs = []
        for i, scene in enumerate(scenes):
            text = (scene.get("narration") or "").strip()
            if text:
                tts_jobs.append((text, os.path.join(temp_root, f"scene_{i}.mp3")))
        presynth_audio = synthesize_narrations_batch(tts_jobs, voice=voice) if tts_jobs else {}

        scene_clips = []
        for i, scene in enumerate(scenes):
            try:
                clip = build_scene_clip(
                    scene,
                    slide_title,
                    drive,
                    voice,
                    temp_root,
                    i,
                    cache=cache,
                    presynth_audio=presynth_audio,
                )
                scene_clips.append(clip)
            except Exception as e:
                print(f"  [slide {slide_index}] scene {i + 1} render failed: {e}")
                traceback.print_exc()

        if not scene_clips:
            return None

        if len(scene_clips) == 1:
            final = scene_clips[0]
        else:
            final = concatenate_videoclips(
                [c.crossfadein(CROSSFADE) if i > 0 else c for i, c in enumerate(scene_clips)],
                method="compose",
                padding=-CROSSFADE,
            )

        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
        final.write_videofile(
            output_path,
            fps=FPS,
            codec="libx264",
            audio_codec="aac",
            preset=_ENCODE_PRESET,
            threads=_ENCODE_THREADS,
            audio_bitrate=_AUDIO_BITRATE,
            ffmpeg_params=["-crf", str(_ENCODE_CRF), "-pix_fmt", "yuv420p", "-movflags", "+faststart"],
            verbose=False,
            logger=None,
        )

        for c in scene_clips:
            try:
                c.close()
            except Exception:
                pass
        return output_path
    finally:
        if own_temp:
            shutil.rmtree(temp_root, ignore_errors=True)


def render_course_video(slide_video_paths, output_path, crossfade=0.0):
    """
    Stitch per-slide MP4s into a single course video.

    Fast path: ffmpeg `concat` demuxer with stream copy (no re-encode). Requires
    that all slide MP4s share codec, profile, resolution, and fps — which they do
    when produced by render_slide_video in the same run.

    Fallback: ffmpeg concat with re-encode (used if stream-copy concat fails).

    :param slide_video_paths: Ordered list of MP4 paths.
    :param output_path: Final stitched MP4 path.
    :param crossfade: Ignored in the stream-copy path. Kept for API compatibility.
    :return: Path to MP4 on success, or None.
    """
    valid_paths = [p for p in slide_video_paths if p and os.path.exists(p) and _ffmpeg_get_duration(p) > 0]
    if not valid_paths:
        return None
    if len(valid_paths) == 1:
        shutil.copy(valid_paths[0], output_path)
        return output_path

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    list_path = output_path + ".concat.txt"
    with open(list_path, "w", encoding="utf-8") as f:
        for p in valid_paths:
            ap = os.path.abspath(p).replace("\\", "/").replace("'", "'\\''")
            f.write(f"file '{ap}'\n")

    fast_cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", list_path,
        "-c", "copy",
        "-movflags", "+faststart",
        output_path,
    ]
    try:
        res = subprocess.run(fast_cmd, capture_output=True, check=False)
        if res.returncode == 0 and os.path.exists(output_path) and _ffmpeg_get_duration(output_path) > 0:
            try:
                os.remove(list_path)
            except Exception:
                pass
            return output_path
        print(f"  [stitch] concat stream-copy failed (rc={res.returncode}); falling back to re-encode.")
    except Exception as e:
        print(f"  [stitch] concat stream-copy errored, falling back to re-encode: {e}")

    fallback_cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", list_path,
        "-c:v", "libx264", "-preset", _ENCODE_PRESET, "-crf", str(_ENCODE_CRF),
        "-c:a", "aac", "-b:a", _AUDIO_BITRATE,
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        output_path,
    ]
    try:
        subprocess.run(fallback_cmd, capture_output=True, check=False)
    finally:
        try:
            os.remove(list_path)
        except Exception:
            pass

    if os.path.exists(output_path) and _ffmpeg_get_duration(output_path) > 0:
        return output_path
    return None


# ---------------------------------------------------------------------------
# Sheet helpers
# ---------------------------------------------------------------------------

def extract_sheet_id_from_link(link):
    """Return the document id portion of a Google Sheets URL, or the input itself if already an id."""
    if not link:
        return None
    match = re.search(r"/d/([a-zA-Z0-9_-]+)", link)
    return match.group(1) if match else link.strip()


def load_slide_rows(gc, sheet_link, worksheet_name="Slide Chunks"):
    """
    Load the Slide Chunks worksheet rows needed for rendering.

    :param gc: Authenticated gspread client.
    :param sheet_link: Sheet URL or id.
    :param worksheet_name: Worksheet/tab name; defaults to "Slide Chunks".
    :return: Tuple of (course_name, list_of_row_dicts). Each row dict carries
        slide_index, slide_title, slide_chunk, slide_type, manifest, narration_text.
    """
    sheet_id = extract_sheet_id_from_link(sheet_link)
    sheet = gc.open_by_key(sheet_id)
    worksheet = sheet.worksheet(worksheet_name)
    records = worksheet.get_all_records()

    course_name = ""
    try:
        ci = sheet.worksheet("Course info")
        ci_records = ci.get_all_records()
        if ci_records:
            course_name = str(ci_records[0].get("Course Name", "")).strip()
    except Exception:
        course_name = ""

    rows = []
    for idx, rec in enumerate(records, start=1):
        manifest = str(rec.get("slideshow_manifest", "")).strip()
        if not manifest or manifest.lower() == "nan" or manifest.startswith("ERROR:"):
            continue
        rows.append({
            "slide_index": idx,
            "slide_title": str(rec.get("Slide Chunk Title", "")).strip(),
            "slide_chunk": str(rec.get("Slide Chunk", "")).strip(),
            "slide_type": str(rec.get("Slide Type", "")).strip(),
            "manifest": manifest,
            "final_graphics_definition": str(rec.get("final_graphics_definition", "")).strip(),
        })
    return course_name, rows


# ---------------------------------------------------------------------------
# Sanitization
# ---------------------------------------------------------------------------

def sanitize_filename(name, fallback="slide"):
    """Return a filesystem-safe filename derived from `name`."""
    if not name:
        return fallback
    safe = re.sub(r"[^A-Za-z0-9._\-]+", "_", name).strip("._")
    return safe or fallback
