"""
Graphics Definition V2 Slideshow Preview
Preview combined image/video outputs from final_graphics_definition.
"""

import base64
import json
import os
import re
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import requests
import streamlit as st
from openai import OpenAI
from PIL import Image

from services.sheets_service import get_sheet_data_and_df


# =============================================================================
# AUTHENTICATION CHECK
# =============================================================================

if "gc" not in st.session_state:
    st.error("You are not authenticated yet. Please use the Login in the main app, then return here.")
    st.stop()

gc = st.session_state["gc"]
drive = st.session_state.get("drive")


# =============================================================================
# OPENAI TTS CLIENT
# =============================================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY is not set in the environment. Please load it from your .env before running this app.")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)


@lru_cache(maxsize=2000)
def generate_tts_audio_b64(vo_text: str) -> str:
    """
    Generate audio (MP3) for a VO segment using OpenAI TTS.
    Returns a base64-encoded string. Uses LRU cache to avoid repeated calls.
    """
    vo_text = (vo_text or "").strip()
    if not vo_text:
        return ""

    try:
        resp = client.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice="alloy",
            input=vo_text,
        )
        audio_bytes = resp.read()
        return base64.b64encode(audio_bytes).decode("utf-8")
    except Exception as exc:
        print(f"[TTS ERROR] Failed for text '{vo_text[:80]}...': {exc}")
        return ""


# =============================================================================
# IMAGE LOADING & CACHING
# =============================================================================

@st.cache_resource
def get_requests_session():
    """Create and cache a requests session for connection pooling."""
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    })
    return session


def _resample_filter():
    try:
        return Image.Resampling.LANCZOS
    except AttributeError:
        return Image.LANCZOS


def extract_drive_file_id(url: str) -> Optional[str]:
    """Extract Google Drive file ID from URL."""
    if not url:
        return None

    match = re.search(r"/file/d/([a-zA-Z0-9_-]+)", url)
    if match:
        return match.group(1)

    match = re.search(r"[?&]id=([a-zA-Z0-9_-]+)", url)
    if match:
        return match.group(1)

    return None


def normalize_drive_view_url(url: str) -> str:
    """Normalize Drive URLs to a consistent view URL when possible."""
    file_id = extract_drive_file_id(url)
    if not file_id:
        return url
    return f"https://drive.google.com/file/d/{file_id}/view"


def is_drive_url(url: str) -> bool:
    """Check if URL is a Google Drive URL."""
    return bool(re.search(r"drive\.google\.com", url or "", re.IGNORECASE))


def download_image_from_web_url(web_url: str) -> Optional[Image.Image]:
    """Download image from a direct web URL."""
    try:
        session = get_requests_session()
        response = session.get(web_url, timeout=12, stream=True, allow_redirects=True)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            img = img.convert("RGB")
            img.thumbnail((800, 800), _resample_filter())
            return img
        return None
    except Exception as exc:
        print(f"Failed to download image from web URL {web_url}: {exc}")
        return None


def download_image_from_drive_url(drive_url: str, drive=None) -> Optional[Image.Image]:
    """Download image from Google Drive using authenticated API when available."""
    try:
        file_id = extract_drive_file_id(drive_url)
        if not file_id:
            return None

        if drive:
            try:
                import os as _os
                import tempfile

                file = drive.CreateFile({"id": file_id})
                file.FetchMetadata(fields="title, mimeType")

                temp_path = _os.path.join(tempfile.gettempdir(), f"{file_id}.img")
                try:
                    file.GetContentFile(temp_path)
                    with Image.open(temp_path) as img:
                        pil_img = img.convert("RGB").copy()
                        pil_img.thumbnail((800, 800), _resample_filter())
                        return pil_img
                finally:
                    if _os.path.exists(temp_path):
                        try:
                            _os.remove(temp_path)
                        except Exception:
                            pass
            except Exception as exc:
                print(f"Drive API failed for {file_id}, trying public URL: {exc}")

        download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        session = get_requests_session()
        response = session.get(download_url, timeout=12, stream=True, allow_redirects=True)

        if response.status_code == 200:
            content = response.content
            if b"<html" in content[:1000].lower() or b"virus scan" in content.lower():
                confirm_match = re.search(
                    r'href="(/uc\?export=download[^"]+)"',
                    content.decode("utf-8", errors="ignore"),
                )
                if confirm_match:
                    confirm_url = "https://drive.google.com" + confirm_match.group(1)
                    response = session.get(confirm_url, timeout=12, stream=True)
                    if response.status_code == 200:
                        content = response.content

            if content and not (b"<html" in content[:1000].lower()):
                img = Image.open(BytesIO(content))
                img = img.convert("RGB")
                img.thumbnail((800, 800), _resample_filter())
                return img
        return None
    except Exception as exc:
        print(f"Failed to download image from {drive_url}: {exc}")
        return None


def download_image_from_url(url: str, drive=None) -> Optional[Image.Image]:
    """Download image from either Google Drive URL or web URL."""
    if not url or not url.startswith("http"):
        return None
    if is_drive_url(url):
        return download_image_from_drive_url(url, drive=drive)
    return download_image_from_web_url(url)


def image_to_base64(img: Image.Image) -> str:
    """Convert PIL Image to base64 string for HTML embedding."""
    buffered = BytesIO()
    if img.mode in ("RGBA", "LA", "P"):
        rgb_img = Image.new("RGB", img.size, (255, 255, 255))
        rgb_img.paste(img, mask=img.split()[-1] if img.mode in ("RGBA", "LA") else None)
        rgb_img.save(buffered, format="JPEG", quality=85, optimize=True)
    else:
        img.save(buffered, format="JPEG", quality=85, optimize=True)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def get_image_cache() -> Dict[str, Dict[str, object]]:
    """Session-level cache for downloaded images and base64 strings."""
    if "graphics_definition_v2_image_cache" not in st.session_state:
        st.session_state["graphics_definition_v2_image_cache"] = {}
    return st.session_state["graphics_definition_v2_image_cache"]


def preload_images(urls: List[str], drive=None, require_base64: bool = False) -> None:
    """Download images in parallel with a progress indicator."""
    urls = [u for u in urls if u and u.startswith("http")]
    if not urls:
        return

    cache = get_image_cache()
    unique_urls = list(dict.fromkeys(urls))
    to_fetch = []
    preexisting = {}
    for url in unique_urls:
        entry = cache.get(url, {})
        if entry.get("error"):
            continue
        if entry.get("pil") and (not require_base64 or entry.get("b64")):
            continue
        preexisting[url] = entry
        to_fetch.append(url)

    if not to_fetch:
        return

    progress_container = st.container()
    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()

    completed = 0
    total = len(to_fetch)

    def download_single(url: str):
        entry = preexisting.get(url, {})
        img = entry.get("pil")
        if not img:
            img = download_image_from_url(url, drive=drive)
        b64 = None
        if img and require_base64:
            b64 = image_to_base64(img)
        return url, img, b64

    with ThreadPoolExecutor(max_workers=24) as executor:
        futures = [executor.submit(download_single, url) for url in to_fetch]
        for future in as_completed(futures):
            url, img, b64 = future.result()
            entry = cache.get(url, {})
            if img:
                entry["pil"] = img
                if b64:
                    entry["b64"] = b64
            else:
                entry["error"] = True
            cache[url] = entry

            completed += 1
            progress_bar.progress(completed / total)
            status_text.markdown(f"Loading images... {completed}/{total}")

    progress_container.empty()


# =============================================================================
# PARSING HELPERS
# =============================================================================

def extract_tag_content(text: str, tag: str) -> str:
    pattern = rf"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    if not match:
        return ""
    return " ".join(match.group(1).strip().split())


def parse_graphics_definition_xml(text: str) -> List[Dict]:
    segments = []
    blocks = re.findall(
        r"<final_graphics_definition>(.*?)</final_graphics_definition>",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if not blocks and re.search(r"<visual_step>", text, re.IGNORECASE):
        blocks = [text]

    for block in blocks:
        steps = []
        for step_xml in re.findall(r"<visual_step>(.*?)</visual_step>", block, re.IGNORECASE | re.DOTALL):
            step = {
                "voiceover_part": extract_tag_content(step_xml, "voiceover_part"),
                "visual_instruction": extract_tag_content(step_xml, "visual_instruction"),
                "asset_url": extract_tag_content(step_xml, "asset"),
                "selection_justification": extract_tag_content(step_xml, "selection_justification"),
            }
            if any(step.values()):
                steps.append(step)
        if steps:
            segments.append({"steps": steps})
    return segments


def parse_formatted_steps(segment_text: str) -> List[Dict]:
    steps = []
    parts = re.split(r"\n-{2,}\s*\n", segment_text.strip(), flags=re.MULTILINE)
    for part in parts:
        if not part.strip():
            continue
        step = {
            "voiceover_part": "",
            "visual_instruction": "",
            "asset_url": "",
            "selection_justification": "",
        }
        current_key = None
        for raw_line in part.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            if line.lower().startswith("when vo"):
                current_key = "voiceover_part"
                value = line.split(":", 1)[1].strip() if ":" in line else line
                step[current_key] = value.strip().strip('"')
                continue
            if line.lower().startswith("visual instructions"):
                current_key = "visual_instruction"
                value = line.split(":", 1)[1].strip() if ":" in line else ""
                step[current_key] = value
                continue
            if line.lower().startswith("graphics to use"):
                current_key = "asset_url"
                value = line.split(":", 1)[1].strip() if ":" in line else ""
                step[current_key] = value
                continue
            if line.lower().startswith("selection justification"):
                current_key = "selection_justification"
                value = line.split(":", 1)[1].strip() if ":" in line else ""
                step[current_key] = value
                continue

            if current_key:
                if step[current_key]:
                    step[current_key] += " " + line
                else:
                    step[current_key] = line

        if any(step.values()):
            steps.append(step)
    return steps


def parse_graphics_definition_formatted(text: str) -> List[Dict]:
    segments = []
    header_pattern = re.compile(r"=+\s*\nSEGMENT\s+\d+.*?\n=+\s*\n", re.IGNORECASE)
    matches = list(header_pattern.finditer(text))
    if matches:
        for idx, match in enumerate(matches):
            start = match.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
            segment_text = text[start:end]
            steps = parse_formatted_steps(segment_text)
            if steps:
                segments.append({"steps": steps})
    else:
        steps = parse_formatted_steps(text)
        if steps:
            segments.append({"steps": steps})
    return segments


def parse_final_graphics_definition(text: str) -> List[Dict]:
    if not text or text.strip() == "" or text.strip().lower() == "nan":
        return []
    text = text.strip()
    segments = parse_graphics_definition_xml(text)
    if segments:
        return segments
    return parse_graphics_definition_formatted(text)


def parse_youtube_clip_params(url: str) -> Tuple[str, int, Optional[int]]:
    """Extract video_id, start, and end timestamps from YouTube URL shapes."""
    try:
        parsed = urllib.parse.urlparse(url)
        query = urllib.parse.parse_qs(parsed.query or "")

        video_id = ""
        if "youtube.com" in (parsed.netloc or "") and parsed.path.startswith("/embed/"):
            video_id = parsed.path.split("/embed/")[-1].split("/")[0]
        elif "youtube.com" in (parsed.netloc or "") and parsed.path.startswith("/watch"):
            video_id = (query.get("v", [""])[0] or "").strip()
        elif "youtu.be" in (parsed.netloc or ""):
            video_id = (parsed.path or "").lstrip("/").split("/")[0]

        start = 0
        if "start" in query:
            try:
                start = int(query["start"][0])
            except Exception:
                start = 0
        elif "t" in query:
            t_val = query["t"][0]
            try:
                start = int(t_val[:-1]) if t_val.endswith("s") else int(t_val)
            except Exception:
                start = 0

        end = None
        if "end" in query and query["end"][0] != "":
            try:
                end = int(query["end"][0])
            except Exception:
                end = None

        return video_id, start, end
    except Exception:
        return "", 0, None


def is_youtube_url(url: str) -> bool:
    return "youtube.com" in (url or "") or "youtu.be" in (url or "")


def classify_asset(asset_url: str) -> Dict[str, object]:
    if not asset_url:
        return {"type": "missing"}
    if is_youtube_url(asset_url):
        video_id, start, end = parse_youtube_clip_params(asset_url)
        if not video_id:
            return {"type": "unknown"}
        if end is None:
            return {"type": "video_still", "video_id": video_id, "start": start, "end": None}
        return {"type": "video", "video_id": video_id, "start": start, "end": end}
    return {"type": "image"}


# =============================================================================
# SHEET LOADING
# =============================================================================

def ensure_sheet_loaded() -> bool:
    if "graphics_definition_v2_sheet" in st.session_state:
        return True
    if "sheet" in st.session_state:
        st.session_state["graphics_definition_v2_sheet"] = st.session_state["sheet"]
        return True

    default_link = st.session_state.get("sheet_link", "")
    sheet_link = st.text_input("Enter Google Sheet link", value=default_link, key="gdv2_sheet_link")

    if st.button("Load Data", type="primary", key="gdv2_load_button"):
        if not sheet_link.strip():
            st.error("Please paste a valid Google Sheet link.")
        else:
            try:
                sheet = gc.open_by_url(sheet_link.strip())
                st.session_state["graphics_definition_v2_sheet"] = sheet
                st.session_state["graphics_definition_v2_sheet_link"] = sheet_link.strip()
                st.success("Data loaded successfully!")
                st.rerun()
            except Exception as exc:
                st.error(f"Failed to open sheet: {exc}")
    return False


# =============================================================================
# YOUTUBE EMBED (INSPECTOR LOOPING)
# =============================================================================

def render_youtube_clip(url: str, dom_id: str, mute: bool, loop: bool) -> None:
    video_id, start, end = parse_youtube_clip_params(url)
    if not video_id:
        st.video(url)
        return

    params = []
    should_autoplay = loop or end is not None
    if start:
        params.append(f"start={start}")
    if end is not None:
        params.append(f"end={end}")
    if mute:
        params.append("mute=1")
    params += [f"autoplay={1 if should_autoplay else 0}", "controls=1", "rel=0", "modestbranding=1", "enablejsapi=1"]
    param_str = "&".join(params)
    embed_url = f"https://www.youtube.com/embed/{video_id}?{param_str}" if param_str else f"https://www.youtube.com/embed/{video_id}"

    end_js = "null" if end is None else str(end)
    loop_flag_js = "true" if loop else "false"
    start_val_js = str(start or 0)
    mute_js = "try{e.target.mute(); e.target.setVolume(0);}catch(_){ }" if mute else ""
    play_js = "try { e.target.playVideo(); } catch(_){ }" if should_autoplay else ""

    html = f"""
<div style="position:relative;padding-top:56.25%;">
    <iframe id="{dom_id}" src="{embed_url}" style="position:absolute;top:0;left:0;width:100%;height:100%;" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>
<script>
(function() {{
    var tag = document.createElement('script');
    tag.src = "https://www.youtube.com/iframe_api";
    var firstScriptTag = document.getElementsByTagName('script')[0];
    if (firstScriptTag && firstScriptTag.parentNode) {{
        firstScriptTag.parentNode.insertBefore(tag, firstScriptTag);
    }}

    var oldReady = window.onYouTubeIframeAPIReady;
    window.onYouTubeIframeAPIReady = function() {{
        if (oldReady) try {{ oldReady(); }} catch(_) {{}}
        try {{
            var player = new YT.Player('{dom_id}', {{
                events: {{
                    'onReady': function(e) {{
                        {mute_js}
                        try {{ e.target.seekTo({start_val_js}, true); }} catch(_) {{}}
                        {play_js}
                    }},
                    'onStateChange': function(e) {{
                        {mute_js}
                        if (e.data === YT.PlayerState.PLAYING) {{
                            var endTime = {end_js};
                            var shouldLoop = {loop_flag_js};
                            var loopStart = {start_val_js};
                            if (endTime) {{
                                var iv = setInterval(function() {{
                                    try {{
                                        var t = player.getCurrentTime();
                                        if (t >= endTime) {{
                                            if (shouldLoop) {{
                                                player.seekTo(loopStart, true);
                                                player.playVideo();
                                            }} else {{
                                                player.pauseVideo();
                                                clearInterval(iv);
                                            }}
                                        }}
                                    }} catch(err) {{}}
                                }}, 250);
                            }}
                        }}
                        if (e.data === YT.PlayerState.ENDED) {{
                            var shouldLoop = {loop_flag_js};
                            var loopStart = {start_val_js};
                            if (shouldLoop) {{
                                try {{
                                    player.seekTo(loopStart, true);
                                    player.playVideo();
                                }} catch(err) {{}}
                            }}
                        }}
                    }}
                }}
            }});
        }} catch(err) {{}}
    }};
}})();
</script>
"""
    st.components.v1.html(html, height=360)


# =============================================================================
# DATA BUILDING
# =============================================================================

def build_slides_data(filtered_df, gd_column: str) -> Tuple[List[Dict], List[str]]:
    slides = []
    image_urls = []

    for slide_pos, (_, row) in enumerate(filtered_df.iterrows(), start=1):
        gd_content = str(row.get(gd_column, "")).strip()
        if not gd_content:
            continue

        segments = parse_final_graphics_definition(gd_content)
        if not segments:
            continue

        slide_title = str(row.get("Slide Chunk Title") or row.get("Topic") or f"Slide {slide_pos}")
        slide = {
            "slideTitle": slide_title,
            "topic": str(row.get("Topic") or ""),
            "subtopic": str(row.get("Subtopic") or ""),
            "slideChunk": str(row.get("Slide Chunk") or ""),
            "segments": [],
        }

        for seg_idx, segment in enumerate(segments, start=1):
            steps_out = []
            for step_idx, step in enumerate(segment.get("steps", []), start=1):
                asset_url = (step.get("asset_url") or "").strip()
                asset_info = classify_asset(asset_url)

                if asset_info.get("type") == "image" and asset_url:
                    image_urls.append(asset_url)

                steps_out.append({
                    "stepId": f"slide_{slide_pos}_seg_{seg_idx}_step_{step_idx}",
                    "vo": step.get("voiceover_part", "").strip(),
                    "instruction": step.get("visual_instruction", "").strip(),
                    "justification": step.get("selection_justification", "").strip(),
                    "assetUrl": asset_url,
                    "assetType": asset_info.get("type"),
                    "videoId": asset_info.get("video_id", ""),
                    "start": asset_info.get("start", 0),
                    "end": asset_info.get("end", None),
                })

            if steps_out:
                slide["segments"].append({
                    "segmentIndex": seg_idx,
                    "steps": steps_out,
                })

        if slide["segments"]:
            slides.append(slide)

    return slides, image_urls


# =============================================================================
# INSPECTOR MODE
# =============================================================================

def render_inspector_mode(slides: List[Dict], image_urls: List[str]) -> None:
    st.markdown("#### Slides Inspector")

    if not slides:
        st.info("No slides with parseable graphics definitions.")
        return

    preload_images(image_urls, drive=drive, require_base64=False)
    cache = get_image_cache()

    for slide_idx, slide in enumerate(slides, start=1):
        with st.expander(slide.get("slideTitle", "Slide"), expanded=False):
            st.caption(f"Topic: {slide.get('topic', '')} | Subtopic: {slide.get('subtopic', '')}")
            with st.container(border=True):
                st.markdown("**Slide Content**")
                st.write(slide.get("slideChunk", "").strip() or "(empty)")

            for segment in slide.get("segments", []):
                with st.container(border=True):
                    st.markdown(f"**Segment {segment.get('segmentIndex')}**")
                    for idx, step in enumerate(segment.get("steps", []), start=1):
                        with st.container(border=True):
                            vo_text = step.get("vo") or ""
                            st.markdown(f"**VO Part {idx}:** {vo_text}")

                            if step.get("instruction"):
                                st.caption(f"Visual Instructions: {step.get('instruction')}")
                            if step.get("justification"):
                                st.caption(f"Selection Justification: {step.get('justification')}")

                            asset_url = step.get("assetUrl") or ""
                            asset_type = step.get("assetType")

                            if asset_type == "image" and asset_url:
                                entry = cache.get(asset_url, {})
                                img = entry.get("pil")
                                if img:
                                    st.image(img, use_container_width=True)
                                else:
                                    st.warning("Image failed to load.")
                            elif asset_type in ("video", "video_still") and asset_url:
                                dom_id = f"gdv2_inspector_{slide_idx}_{segment.get('segmentIndex')}_{idx}"
                                render_youtube_clip(asset_url, dom_id, mute=True, loop=(asset_type == "video"))
                            else:
                                st.warning("No asset available for this step.")

                            if asset_url:
                                asset_link = normalize_drive_view_url(asset_url) if is_drive_url(asset_url) else asset_url
                                st.markdown(f"[Open asset]({asset_link})")


# =============================================================================
# SLIDESHOW MODE
# =============================================================================

def render_slideshow_mode(slides: List[Dict], image_urls: List[str]) -> None:
    st.markdown("#### Slideshow Preview")
    st.info("Click Start to begin automatic narration. Visuals will advance with the VO parts.")

    if not slides:
        st.info("No slides with parseable graphics definitions.")
        return

    preload_images(image_urls, drive=drive, require_base64=True)
    cache = get_image_cache()

    slides_payload = []
    audio_map = {}

    with st.spinner("Generating narration audio..."):
        for slide in slides:
            slide_out = {
                "slideTitle": slide.get("slideTitle", ""),
                "segments": [],
            }
            for segment in slide.get("segments", []):
                segment_out = {
                    "segmentIndex": segment.get("segmentIndex"),
                    "steps": [],
                }
                for step in segment.get("steps", []):
                    asset_url = step.get("assetUrl") or ""
                    image_src = ""
                    if step.get("assetType") == "image" and asset_url:
                        entry = cache.get(asset_url, {})
                        if entry.get("b64"):
                            image_src = f"data:image/jpeg;base64,{entry['b64']}"

                    step_out = {
                        "stepId": step.get("stepId"),
                        "vo": step.get("vo"),
                        "assetUrl": asset_url,
                        "assetType": step.get("assetType"),
                        "imageSrc": image_src,
                        "videoId": step.get("videoId"),
                        "start": step.get("start", 0),
                        "end": step.get("end"),
                    }
                    segment_out["steps"].append(step_out)

                    audio_b64 = generate_tts_audio_b64(step.get("vo", ""))
                    audio_map[step_out["stepId"]] = audio_b64

                if segment_out["steps"]:
                    slide_out["segments"].append(segment_out)

            if slide_out["segments"]:
                slides_payload.append(slide_out)

    if not slides_payload:
        st.info("No steps found to play.")
        return

    slides_json = json.dumps(slides_payload)
    audio_json = json.dumps(audio_map)

    html = f"""
<div id="gdv2-slideshow" style="display:flex;flex-direction:column;gap:12px;font-family:system-ui,-apple-system,sans-serif;">
  <div id="controls" style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;">
    <button id="btnStart" style="padding:8px 16px;background:#2563eb;color:white;border:none;border-radius:4px;cursor:pointer;font-weight:600;" disabled>Preparing...</button>
    <button id="btnPause" style="padding:8px 16px;background:#6b7280;color:white;border:none;border-radius:4px;cursor:pointer;">Pause</button>
    <button id="btnResume" style="padding:8px 16px;background:#2563eb;color:white;border:none;border-radius:4px;cursor:pointer;display:none;">Resume</button>
    <button id="btnPrevSlide" style="padding:8px 16px;background:#374151;color:white;border:none;border-radius:4px;cursor:pointer;">Previous Slide</button>
    <button id="btnNextSlide" style="padding:8px 16px;background:#374151;color:white;border:none;border-radius:4px;cursor:pointer;">Next Slide</button>
    <button id="btnPrevPart" style="padding:8px 16px;background:#9ca3af;color:white;border:none;border-radius:4px;cursor:pointer;">Previous VO Part</button>
    <button id="btnNextPart" style="padding:8px 16px;background:#9ca3af;color:white;border:none;border-radius:4px;cursor:pointer;">Next VO Part</button>
    <span id="status" style="margin-left:8px;color:#374151;font-size:14px;"></span>
  </div>

  <div id="voBox" style="background:#eff6ff;padding:18px;border-radius:8px;border:2px solid #2563eb;font-size:18px;line-height:1.5;min-height:72px;display:flex;align-items:center;justify-content:center;text-align:center;color:#111827;">
    Loading...
  </div>

  <div id="visualStage" style="position:relative;width:100%;padding-top:56.25%;background:#111827;border-radius:8px;overflow:hidden;">
    <div id="videoLayer" style="position:absolute;top:0;left:0;width:100%;height:100%;display:none;">
      <div id="ytPlayer" style="width:100%;height:100%;"></div>
    </div>
    <img id="imageLayer" alt="Visual" style="position:absolute;top:0;left:0;width:100%;height:100%;object-fit:contain;display:none;background:#000;" />
    <div id="placeholderLayer" style="position:absolute;top:0;left:0;width:100%;height:100%;display:flex;align-items:center;justify-content:center;color:#9ca3af;font-size:16px;">No asset available</div>
  </div>

  <div id="assetLinkWrap" style="font-size:14px;color:#1d4ed8;">
    <a id="assetLink" href="#" target="_blank" rel="noreferrer" style="display:none;">Open asset</a>
  </div>

  <div id="progressBar" style="width:100%;height:8px;background:#e5e7eb;border-radius:4px;overflow:hidden;">
    <div id="progressFill" style="width:0%;height:100%;background:#2563eb;transition:width 0.2s;"></div>
  </div>
</div>

<div id="preloadContainer" style="width:1px;height:1px;overflow:hidden;position:absolute;left:-9999px;top:-9999px;">
  <div id="ytPreloader"></div>
</div>

<script>
(function() {{
  var slides = {slides_json};
  var audioData = {audio_json};
  var flatSteps = [];
  var slideFirstStep = [];
  var slideStepCounts = [];

  slides.forEach(function(slide, sIdx) {{
    slideFirstStep[sIdx] = flatSteps.length;
    var slideStepCounter = 0;
    (slide.segments || []).forEach(function(seg, segIdx) {{
      var segSteps = seg.steps || [];
      var segStepCount = segSteps.length;
      segSteps.forEach(function(step, stepIdx) {{
        slideStepCounter += 1;
        flatSteps.push({{
          slideIndex: sIdx,
          slideTitle: slide.slideTitle || ("Slide " + (sIdx + 1)),
          segmentIndex: segIdx,
          segmentStepIndex: stepIdx + 1,
          segmentStepCount: segStepCount,
          stepInSlide: slideStepCounter,
          stepId: step.stepId,
          vo: step.vo || "",
          assetType: step.assetType || "missing",
          assetUrl: step.assetUrl || "",
          imageSrc: step.imageSrc || "",
          videoId: step.videoId || "",
          start: step.start || 0,
          end: step.end
        }});
      }});
    }});
    slideStepCounts[sIdx] = slideStepCounter;
  }});

  flatSteps.forEach(function(step) {{
    step.slideStepCount = slideStepCounts[step.slideIndex] || 0;
  }});

  var statusEl = document.getElementById('status');
  var voBox = document.getElementById('voBox');
  var videoLayer = document.getElementById('videoLayer');
  var imageLayer = document.getElementById('imageLayer');
  var placeholderLayer = document.getElementById('placeholderLayer');
  var assetLink = document.getElementById('assetLink');
  var progressFill = document.getElementById('progressFill');

  var btnStart = document.getElementById('btnStart');
  var btnPause = document.getElementById('btnPause');
  var btnResume = document.getElementById('btnResume');
  var btnPrevSlide = document.getElementById('btnPrevSlide');
  var btnNextSlide = document.getElementById('btnNextSlide');
  var btnPrevPart = document.getElementById('btnPrevPart');
  var btnNextPart = document.getElementById('btnNextPart');

  if (!flatSteps.length) {{
    if (statusEl) statusEl.textContent = "No steps available.";
    if (voBox) voBox.textContent = "No steps available.";
    btnStart.disabled = true;
    return;
  }}

  var currentIndex = 0;
  var isPlaying = false;
  var paused = false;
  var mainPlayer = null;
  var preloadPlayer = null;
  var currentAudio = null;
  var narrationDone = false;
  var videoDone = false;
  var advanceInterval = null;
  var videoInterval = null;

  function cleanupTimers() {{
    if (advanceInterval) {{
      clearInterval(advanceInterval);
      advanceInterval = null;
    }}
    if (videoInterval) {{
      clearInterval(videoInterval);
      videoInterval = null;
    }}
  }}

  function stopAudio() {{
    if (currentAudio) {{
      try {{ currentAudio.pause(); }} catch (e) {{}}
      currentAudio = null;
    }}
  }}

  function updateStatus(extra) {{
    if (!statusEl) return;
    var step = flatSteps[currentIndex];
    var text = "Slide " + (step.slideIndex + 1) + " of " + slides.length;
    text += ", Step " + step.stepInSlide + " of " + (step.slideStepCount || 0);
    if (step.slideTitle) {{
      text += " | " + step.slideTitle;
    }}
    if (extra) {{
      text += " | " + extra;
    }}
    statusEl.textContent = text;
  }}

  function updateProgress() {{
    if (!progressFill) return;
    var pct = ((currentIndex + 1) / flatSteps.length) * 100;
    progressFill.style.width = pct + "%";
  }}

  function updateVO(text) {{
    if (!voBox) return;
    voBox.textContent = text || "No voiceover text available.";
  }}

  function updateAssetLink(url) {{
    if (!assetLink) return;
    if (url) {{
      assetLink.href = url;
      assetLink.style.display = "inline-block";
    }} else {{
      assetLink.style.display = "none";
    }}
  }}

  function showPlaceholder(message) {{
    if (placeholderLayer) {{
      placeholderLayer.style.display = "flex";
      placeholderLayer.textContent = message || "No asset available";
    }}
    if (imageLayer) imageLayer.style.display = "none";
    if (videoLayer) videoLayer.style.display = "none";
  }}

  function showImage(src) {{
    if (imageLayer) {{
      imageLayer.src = src;
      imageLayer.style.display = "block";
    }}
    if (videoLayer) videoLayer.style.display = "none";
    if (placeholderLayer) placeholderLayer.style.display = "none";
  }}

  function showVideo() {{
    if (videoLayer) videoLayer.style.display = "block";
    if (imageLayer) imageLayer.style.display = "none";
    if (placeholderLayer) placeholderLayer.style.display = "none";
  }}

  function cueVideo(step) {{
    if (!mainPlayer || !step.videoId) return;
    try {{
      mainPlayer.cueVideoById({{ videoId: step.videoId, startSeconds: step.start, endSeconds: step.end || undefined }});
      mainPlayer.mute();
    }} catch (e) {{}}
  }}

  function playVideo(step) {{
    if (!mainPlayer || !step.videoId) return;
    try {{
      mainPlayer.loadVideoById({{ videoId: step.videoId, startSeconds: step.start, endSeconds: step.end || undefined }});
      mainPlayer.mute();
      mainPlayer.playVideo();
    }} catch (e) {{}}
  }}

  function playAudioForStep(step, onend) {{
    stopAudio();
    var b64 = audioData[step.stepId];
    if (!b64) {{
      if (onend) {{
        setTimeout(onend, 1200);
      }}
      return;
    }}
    try {{
      var audio = new Audio("data:audio/mp3;base64," + b64);
      currentAudio = audio;
      audio.onended = function() {{
        currentAudio = null;
        if (onend) onend();
      }};
      audio.onerror = function() {{
        currentAudio = null;
        if (onend) onend();
      }};
      audio.play();
    }} catch (err) {{
      currentAudio = null;
      if (onend) onend();
    }}
  }}

  function startWatchers(step) {{
    cleanupTimers();
    if (step.assetType === "video" && step.end !== null && step.end !== undefined) {{
      videoInterval = setInterval(function() {{
        if (!mainPlayer) return;
        try {{
          var t = mainPlayer.getCurrentTime ? mainPlayer.getCurrentTime() : 0;
          if (t >= (step.end - 0.05)) {{
            videoDone = true;
            mainPlayer.pauseVideo();
            mainPlayer.seekTo(Math.max(step.end - 0.05, step.start), true);
            clearInterval(videoInterval);
          }}
        }} catch (err) {{}}
      }}, 200);
    }} else {{
      videoDone = true;
    }}

    advanceInterval = setInterval(function() {{
      if (!paused && narrationDone && videoDone) {{
        clearInterval(advanceInterval);
        nextStep();
      }}
    }}, 200);
  }}

  function displayStep(index) {{
    currentIndex = index;
    var step = flatSteps[currentIndex];
    updateStatus();
    updateVO(step.vo);
    updateAssetLink(step.assetUrl);
    updateProgress();

    if (step.assetType === "image" && step.imageSrc) {{
      showImage(step.imageSrc);
    }} else if (step.assetType === "video" && step.videoId) {{
      showVideo();
      cueVideo(step);
    }} else {{
      showPlaceholder("No asset available for this step");
    }}
  }}

  function playStep(index) {{
    cleanupTimers();
    stopAudio();
    currentIndex = index;
    var step = flatSteps[currentIndex];

    narrationDone = false;
    videoDone = (step.assetType !== "video");

    updateStatus("Playing");
    updateVO(step.vo);
    updateAssetLink(step.assetUrl);
    updateProgress();

    if (step.assetType === "image" && step.imageSrc) {{
      showImage(step.imageSrc);
    }} else if (step.assetType === "video" && step.videoId) {{
      showVideo();
      playVideo(step);
    }} else {{
      showPlaceholder("No asset available for this step");
    }}

    playAudioForStep(step, function() {{
      narrationDone = true;
    }});

    startWatchers(step);
  }}

  function nextStep() {{
    cleanupTimers();
    stopAudio();
    if (currentIndex + 1 < flatSteps.length) {{
      currentIndex += 1;
      if (isPlaying && !paused) {{
        playStep(currentIndex);
      }} else {{
        displayStep(currentIndex);
      }}
    }} else {{
      isPlaying = false;
      updateStatus("End of slideshow");
      if (voBox) voBox.textContent = "Slideshow complete.";
    }}
  }}

  function prevStep() {{
    cleanupTimers();
    stopAudio();
    if (currentIndex - 1 >= 0) {{
      currentIndex -= 1;
      if (isPlaying && !paused) {{
        playStep(currentIndex);
      }} else {{
        displayStep(currentIndex);
      }}
    }} else {{
      displayStep(0);
    }}
  }}

  function gotoSlide(slideIndex) {{
    if (slideIndex < 0 || slideIndex >= slides.length) return;
    var target = slideFirstStep[slideIndex];
    if (target === undefined) return;
    cleanupTimers();
    stopAudio();
    currentIndex = target;
    if (isPlaying && !paused) {{
      playStep(currentIndex);
    }} else {{
      displayStep(currentIndex);
    }}
  }}

  function nextSlide() {{
    var slideIndex = flatSteps[currentIndex].slideIndex;
    gotoSlide(slideIndex + 1);
  }}

  function prevSlide() {{
    var slideIndex = flatSteps[currentIndex].slideIndex;
    gotoSlide(slideIndex - 1);
  }}

  function pausePlayback() {{
    if (!isPlaying || paused) return;
    paused = true;
    cleanupTimers();
    if (currentAudio) {{
      try {{ currentAudio.pause(); }} catch (e) {{}}
    }}
    if (mainPlayer) {{
      try {{ mainPlayer.pauseVideo(); }} catch (e) {{}}
    }}
    btnResume.style.display = "inline-block";
    updateStatus("Paused");
  }}

  function resumePlayback() {{
    if (!paused) return;
    paused = false;
    btnResume.style.display = "none";
    var step = flatSteps[currentIndex];
    if (currentAudio && !narrationDone) {{
      try {{ currentAudio.play(); }} catch (e) {{}}
    }}
    if (mainPlayer && step.assetType === "video" && !videoDone) {{
      try {{ mainPlayer.playVideo(); }} catch (e) {{}}
    }}
    startWatchers(step);
    updateStatus("Playing");
  }}

  btnStart.onclick = function() {{
    isPlaying = true;
    paused = false;
    btnResume.style.display = "none";
    playStep(currentIndex);
  }};

  btnPause.onclick = pausePlayback;
  btnResume.onclick = resumePlayback;
  btnNextPart.onclick = nextStep;
  btnPrevPart.onclick = prevStep;
  btnNextSlide.onclick = nextSlide;
  btnPrevSlide.onclick = prevSlide;

  displayStep(0);

  var preloadQueue = [];
  var seen = {{}};
  flatSteps.forEach(function(step) {{
    if (step.assetType === "video" && step.videoId && step.end !== null && step.end !== undefined) {{
      var key = step.videoId + "_" + step.start + "_" + step.end;
      if (!seen[key]) {{
        seen[key] = true;
        preloadQueue.push({{ videoId: step.videoId, start: step.start, end: step.end }});
      }}
    }}
  }});

  function setStartEnabled(enabled) {{
    btnStart.disabled = !enabled;
    btnStart.textContent = enabled ? "Start" : "Preparing...";
  }}

  function finishPreload() {{
    setStartEnabled(true);
    updateStatus("Ready. Click Start to begin.");
  }}

  var preloadIndex = 0;
  var preloadBusy = false;
  var preloadActive = false;

  function loadNextPreload() {{
    if (!preloadActive) return;
    if (preloadIndex >= preloadQueue.length) {{
      preloadActive = false;
      finishPreload();
      return;
    }}
    var item = preloadQueue[preloadIndex];
    updateStatus("Preloading videos " + (preloadIndex + 1) + "/" + preloadQueue.length);
    preloadBusy = true;
    try {{
      preloadPlayer.loadVideoById({{ videoId: item.videoId, startSeconds: item.start, endSeconds: item.end }});
    }} catch (e) {{
      preloadBusy = false;
      preloadIndex += 1;
      loadNextPreload();
    }}
  }}

  function handlePreloadState(event) {{
    if (!preloadActive || !preloadBusy) return;
    if (event.data === YT.PlayerState.PLAYING) {{
      preloadBusy = false;
      setTimeout(function() {{
        try {{ preloadPlayer.pauseVideo(); }} catch (e) {{}}
        preloadIndex += 1;
        setTimeout(loadNextPreload, 120);
      }}, 350);
    }}
  }}

  function handlePreloadError() {{
    if (!preloadActive) return;
    preloadBusy = false;
    preloadIndex += 1;
    setTimeout(loadNextPreload, 120);
  }}

  function startPreload() {{
    if (!preloadQueue.length) {{
      finishPreload();
      return;
    }}
    preloadActive = true;
    preloadIndex = 0;
    loadNextPreload();
  }}

  function initPlayers() {{
    var firstVideoId = preloadQueue.length ? preloadQueue[0].videoId : "M7lc1UVf-VE";
    mainPlayer = new YT.Player('ytPlayer', {{
      videoId: firstVideoId,
      playerVars: {{
        'rel': 0,
        'modestbranding': 1,
        'controls': 1
      }},
      events: {{
        'onReady': function(e) {{
          try {{ e.target.mute(); }} catch (err) {{}}
          displayStep(0);
        }}
      }}
    }});

    preloadPlayer = new YT.Player('ytPreloader', {{
      videoId: firstVideoId,
      playerVars: {{
        'rel': 0,
        'modestbranding': 1,
        'controls': 0
      }},
      events: {{
        'onReady': function(e) {{
          try {{ e.target.mute(); }} catch (err) {{}}
          startPreload();
        }},
        'onStateChange': handlePreloadState,
        'onError': handlePreloadError
      }}
    }});
  }}

  if (preloadQueue.length) {{
    setStartEnabled(false);
    var tag = document.createElement('script');
    tag.src = "https://www.youtube.com/iframe_api";
    var firstScriptTag = document.getElementsByTagName('script')[0];
    if (firstScriptTag && firstScriptTag.parentNode) {{
      firstScriptTag.parentNode.insertBefore(tag, firstScriptTag);
    }}
    var oldReady = window.onYouTubeIframeAPIReady;
    window.onYouTubeIframeAPIReady = function() {{
      if (oldReady) try {{ oldReady(); }} catch (e) {{}}
      initPlayers();
    }};
  }} else {{
    finishPreload();
  }}
}})();
</script>
"""
    st.components.v1.html(html, height=720)


# =============================================================================
# MAIN UI
# =============================================================================

def main() -> None:
    st.title("Graphics Definition V2 Slideshow Preview")
    st.markdown("Preview combined image and video outputs from the Aggregation Agent.")

    if not ensure_sheet_loaded():
        return

    sheet = st.session_state["graphics_definition_v2_sheet"]

    try:
        worksheet, df = get_sheet_data_and_df(sheet, "Slide Chunks")
    except Exception as exc:
        st.error(f"Failed to load 'Slide Chunks' worksheet: {exc}")
        return

    if df.empty:
        st.info("Slide Chunks worksheet is empty.")
        return

    gd_column = None
    for col in df.columns:
        if "final_graphics_definition" in col.lower():
            gd_column = col
            break

    if not gd_column:
        st.error("No 'final_graphics_definition' column found. Run the aggregation agent first.")
        st.write("Available columns:", list(df.columns))
        return

    df[gd_column] = df[gd_column].astype(str)
    preview_rows = df[df[gd_column].str.strip() != ""].copy()

    if preview_rows.empty:
        st.info("No slides with final_graphics_definition found.")
        return

    st.markdown("---")
    st.markdown("### Filters")

    def unique_values(series):
        values = []
        for item in series.astype(str).tolist():
            item = item.strip()
            if not item or item.lower() == "nan":
                continue
            values.append(item)
        return ["All"] + sorted(set(values))

    topics = unique_values(preview_rows.get("Topic", ""))
    subtopics = unique_values(preview_rows.get("Subtopic", ""))

    col1, col2, col3 = st.columns([2, 2, 3])
    with col1:
        sel_topic = st.selectbox("Filter by Topic", topics, index=0)
    with col2:
        sel_subtopic = st.selectbox("Filter by Subtopic", subtopics, index=0)
    with col3:
        search = st.text_input("Search slide title/content")

    filtered = preview_rows
    if sel_topic != "All":
        filtered = filtered[filtered.get("Topic", "").astype(str) == sel_topic]
    if sel_subtopic != "All":
        filtered = filtered[filtered.get("Subtopic", "").astype(str) == sel_subtopic]
    if search:
        s = search.lower()
        filtered = filtered[
            filtered.get("Slide Chunk Title", "").astype(str).str.lower().str.contains(s, na=False)
            | filtered.get("Slide Chunk", "").astype(str).str.lower().str.contains(s, na=False)
        ]

    if filtered.empty:
        st.warning("No slides match current filters.")
        return

    slides, image_urls = build_slides_data(filtered, gd_column)

    if not slides:
        st.info("No parseable graphics definitions found in filtered rows.")
        return

    st.success(f"Found {len(slides)} slide(s) with graphics definitions.")

    st.markdown("---")
    inspector_tab, slideshow_tab = st.tabs(["Inspector", "Slideshow"])

    with inspector_tab:
        render_inspector_mode(slides, image_urls)

    with slideshow_tab:
        render_slideshow_mode(slides, image_urls)


if __name__ == "__main__":
    main()
else:
    main()
