import base64
import hashlib
import imghdr
import json
import os
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import parse_qs, urlparse

import pandas as pd
import requests
import streamlit as st
from openai import OpenAI

from services.sheets_service import get_sheet_data_and_df, get_worksheet_names


DEFAULT_SHEET_NAME = "Slide Chunks"
FINAL_GRAPHICS_COLUMN = "final_graphics_definition"
TTS_MODEL = "gpt-4o-mini-tts"
TTS_VOICE = "alloy"


def safe_str(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value)


def find_column(df, name):
    target = name.strip().lower()
    for col in df.columns:
        if str(col).strip().lower() == target:
            return col
    return None


def is_drive_url(url):
    if not url:
        return False
    lowered = url.lower()
    return "drive.google.com" in lowered or "docs.google.com" in lowered


def extract_drive_file_id(url):
    if not url:
        return None
    patterns = [
        r"/file/d/([a-zA-Z0-9-_]+)",
        r"id=([a-zA-Z0-9-_]+)",
        r"drive.google.com/open\?id=([a-zA-Z0-9-_]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def normalize_drive_image_url(url):
    file_id = extract_drive_file_id(url)
    if not file_id:
        return url
    return f"https://drive.google.com/uc?export=view&id={file_id}"


def is_youtube_embed(url):
    if not url:
        return False
    lowered = url.lower()
    return "youtube.com/embed" in lowered or "youtu.be/" in lowered


def parse_youtube_embed(url):
    if not url:
        return None
    parsed = urlparse(url)
    video_id = ""
    if "youtube.com" in parsed.netloc and "/embed/" in parsed.path:
        video_id = parsed.path.split("/embed/")[-1].split("/")[0]
    elif "youtu.be" in parsed.netloc:
        video_id = parsed.path.lstrip("/").split("/")[0]
    if not video_id:
        return None
    query = parse_qs(parsed.query)
    start = query.get("start", [None])[0]
    end = query.get("end", [None])[0]
    try:
        start_val = float(start) if start is not None else 0.0
    except ValueError:
        start_val = 0.0
    try:
        end_val = float(end) if end is not None else None
    except ValueError:
        end_val = None
    return {
        "video_id": video_id,
        "start": start_val,
        "end": end_val,
    }


def detect_asset_type(asset):
    if not asset:
        return "unknown"
    if is_youtube_embed(asset):
        return "video"
    lowered = asset.lower()
    if is_drive_url(asset):
        return "image"
    if re.search(r"\.(png|jpe?g|gif|webp)(\?|$)", lowered):
        return "image"
    return "image"


def extract_text_from_tag(element, tag_names):
    if element is None:
        return ""
    for child in element.iter():
        tag = getattr(child, "tag", "")
        if not tag:
            continue
        if tag.lower() in tag_names:
            text = "".join(child.itertext()).strip()
            if text:
                return text
    return ""


def parse_xml_segments(text):
    if not text or "<" not in text:
        return []
    cleaned = text.strip()
    root = None
    try:
        root = _safe_xml_parse(cleaned)
    except Exception:
        root = None
    if root is None:
        return []

    segments = []
    segment_elements = [el for el in root.iter() if getattr(el, "tag", "").lower() == "segment"]
    if segment_elements:
        for seg_idx, seg in enumerate(segment_elements, start=1):
            steps = _extract_visual_steps(seg)
            if steps:
                segments.append(_build_segment(seg_idx, steps))
        return segments

    fgd_elements = [el for el in root.iter() if getattr(el, "tag", "").lower() == "final_graphics_definition"]
    if fgd_elements:
        for seg_idx, fgd in enumerate(fgd_elements, start=1):
            steps = _extract_visual_steps(fgd)
            if steps:
                segments.append(_build_segment(seg_idx, steps))
        return segments

    steps = _extract_visual_steps(root)
    if steps:
        segments.append(_build_segment(1, steps))
    return segments


def _safe_xml_parse(text):
    import xml.etree.ElementTree as ET

    try:
        return ET.fromstring(text)
    except ET.ParseError:
        pass
    try:
        return ET.fromstring(f"<root>{text}</root>")
    except ET.ParseError:
        pass
    sanitized = re.sub(r"&(?![a-zA-Z]+;|#\d+;|#x[0-9A-Fa-f]+;)", "&amp;", text)
    return ET.fromstring(f"<root>{sanitized}</root>")


def _extract_visual_steps(element):
    steps = []
    visual_steps = [el for el in element.iter() if getattr(el, "tag", "").lower() == "visual_step"]
    for idx, step_el in enumerate(visual_steps, start=1):
        voiceover = extract_text_from_tag(step_el, {"voiceover_part", "voiceover", "vo_text"})
        instruction = extract_text_from_tag(step_el, {"visual_instruction", "instruction"})
        asset = extract_text_from_tag(step_el, {"asset", "graphics_to_use", "graphic"})
        justification = extract_text_from_tag(step_el, {"selection_justification", "justification"})
        steps.append(
            {
                "step_index": idx,
                "voiceover": voiceover,
                "instruction": instruction,
                "asset": asset,
                "justification": justification,
            }
        )
    return steps


def parse_formatted_segments(text):
    if not text:
        return []
    segment_matches = list(re.finditer(r"SEGMENT\s+(\d+)", text, re.IGNORECASE))
    if not segment_matches:
        return []
    segments = []
    for idx, match in enumerate(segment_matches):
        seg_num = int(match.group(1))
        start = match.end()
        end = segment_matches[idx + 1].start() if idx + 1 < len(segment_matches) else len(text)
        segment_text = text[start:end]
        steps = _parse_formatted_steps(segment_text)
        if steps:
            segments.append(_build_segment(seg_num, steps))
    return segments


def _parse_formatted_steps(segment_text):
    if not segment_text:
        return []
    blocks = re.split(r"\n\s*-{2,}\s*\n", segment_text)
    steps = []
    for block in blocks:
        parsed = _parse_formatted_block(block)
        if parsed:
            steps.append(parsed)
    return steps


def _parse_formatted_block(block):
    labels = {
        "when vo": "voiceover",
        "visual instructions": "instruction",
        "graphics to use": "asset",
        "selection justification": "justification",
    }
    data = {"voiceover": "", "instruction": "", "asset": "", "justification": ""}
    current_key = None
    for raw_line in block.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        lowered = line.lower()
        matched = False
        for label, key in labels.items():
            prefix = f"{label}:"
            if lowered.startswith(prefix):
                value = line[len(prefix):].strip()
                if key == "voiceover":
                    value = value.strip().strip('"').strip("'")
                data[key] = value
                current_key = key
                matched = True
                break
        if matched:
            continue
        if current_key:
            data[current_key] = f"{data[current_key]} {line}".strip()
    if any(data.values()):
        return {
            "step_index": 0,
            "voiceover": data["voiceover"],
            "instruction": data["instruction"],
            "asset": data["asset"],
            "justification": data["justification"],
        }
    return None


def parse_graphics_definition(text):
    cleaned = safe_str(text).strip()
    if not cleaned:
        return []
    segments = parse_xml_segments(cleaned)
    if segments:
        return segments
    segments = parse_formatted_segments(cleaned)
    if segments:
        return segments
    fallback_step = _fallback_step_from_text(cleaned)
    if fallback_step:
        return [_build_segment(1, [fallback_step])]
    return []


def _fallback_step_from_text(text):
    url_match = re.search(r"https?://[^\s)>\"]+", text)
    asset = url_match.group(0) if url_match else ""
    return {
        "step_index": 1,
        "voiceover": "",
        "instruction": text.strip(),
        "asset": asset,
        "justification": "",
    }


def _build_segment(segment_index, steps):
    normalized_steps = []
    for idx, step in enumerate(steps, start=1):
        step_copy = dict(step)
        step_copy["step_index"] = idx
        normalized_steps.append(step_copy)
    return {"segment_index": segment_index, "steps": normalized_steps}


def download_image_bytes(url, drive):
    if not url:
        return None
    if is_drive_url(url):
        file_id = extract_drive_file_id(url)
        if file_id and drive is not None:
            try:
                drive_file = drive.CreateFile({"id": file_id})
                with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                    tmp_path = tmp_file.name
                drive_file.GetContentFile(tmp_path)
                with open(tmp_path, "rb") as handle:
                    data = handle.read()
                os.remove(tmp_path)
                return data
            except Exception:
                return None
        url = normalize_drive_image_url(url)
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.content
    except Exception:
        return None


def image_bytes_to_data_uri(image_bytes):
    if not image_bytes:
        return ""
    image_type = imghdr.what(None, h=image_bytes) or "jpeg"
    if image_type == "jpg":
        image_type = "jpeg"
    mime = f"image/{image_type}"
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime};base64,{encoded}"


@st.cache_resource
def get_openai_client():
    return OpenAI()


def generate_tts_audio(text, client):
    payload = text.strip() if text else ""
    if not payload:
        payload = " "
    response = client.audio.speech.create(
        model=TTS_MODEL,
        voice=TTS_VOICE,
        input=payload,
        response_format="mp3",
    )
    return response.content


def audio_bytes_to_data_uri(audio_bytes):
    if not audio_bytes:
        return ""
    encoded = base64.b64encode(audio_bytes).decode("ascii")
    return f"data:audio/mpeg;base64,{encoded}"


def build_slides_from_df(df, column_map):
    slides = []
    for idx, row in df.iterrows():
        final_def = safe_str(row.get(column_map["final_def"], "")).strip()
        segments = parse_graphics_definition(final_def)
        slide = {
            "row_index": idx,
            "topic": safe_str(row.get(column_map["topic"], "")),
            "subtopic": safe_str(row.get(column_map["subtopic"], "")),
            "slide_title": safe_str(row.get(column_map["slide_title"], "")),
            "slide_chunk": safe_str(row.get(column_map["slide_chunk"], "")),
            "final_definition_raw": final_def,
            "segments": segments,
        }
        slides.append(slide)
    return slides


def flatten_steps(slides):
    flat_steps = []
    total_slides = len(slides)
    for slide_idx, slide in enumerate(slides, start=1):
        segments = slide.get("segments", [])
        for seg_idx, segment in enumerate(segments, start=1):
            steps = segment.get("steps", [])
            for step_idx, step in enumerate(steps, start=1):
                asset = safe_str(step.get("asset", "")).strip()
                asset_type = detect_asset_type(asset)
                flat_steps.append(
                    {
                        "flat_index": len(flat_steps),
                        "slide_index": slide_idx,
                        "slide_total": total_slides,
                        "segment_index": seg_idx,
                        "segment_total": len(segments),
                        "step_index": step_idx,
                        "step_total": len(steps),
                        "voiceover": safe_str(step.get("voiceover", "")),
                        "instruction": safe_str(step.get("instruction", "")),
                        "justification": safe_str(step.get("justification", "")),
                        "asset": asset,
                        "asset_type": asset_type,
                    }
                )
    return flat_steps


def compute_slideshow_key(steps):
    key_payload = [
        (step.get("voiceover", ""), step.get("asset", ""), step.get("asset_type", ""))
        for step in steps
    ]
    digest = hashlib.md5(json.dumps(key_payload, sort_keys=True).encode("utf-8")).hexdigest()
    return digest


def prepare_slideshow_assets(steps, drive, openai_client):
    """
    Prepare all slideshow assets with parallel TTS and image loading for faster preloading.
    """
    tts_cache = st.session_state.setdefault("gdv2_tts_cache", {})
    image_cache = st.session_state.setdefault("gdv2_image_cache", {})
    
    total = len(steps)
    progress = st.progress(0.0)
    status = st.empty()
    
    # Separate items that need TTS vs already cached
    tts_needed = []
    images_needed = []
    
    for idx, step in enumerate(steps):
        voiceover = step.get("voiceover", "").strip()
        if voiceover and voiceover not in tts_cache:
            tts_needed.append((idx, voiceover))
        
        asset = step.get("asset", "").strip()
        asset_type = step.get("asset_type", "unknown")
        if asset_type == "image" and asset and asset not in image_cache:
            images_needed.append((idx, asset))
    
    completed = 0
    total_tasks = len(tts_needed) + len(images_needed) + len(steps)
    
    # Generate TTS in parallel (up to 8 concurrent requests)
    if tts_needed:
        status.write(f"🎙️ Generating {len(tts_needed)} narration audio clips in parallel...")
        
        def generate_single_tts(item):
            idx, voiceover = item
            try:
                audio_bytes = generate_tts_audio(voiceover, openai_client)
                return voiceover, audio_bytes_to_data_uri(audio_bytes)
            except Exception as e:
                print(f"TTS error for step {idx}: {e}")
                return voiceover, ""
        
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(generate_single_tts, item): item for item in tts_needed}
            for future in as_completed(futures):
                try:
                    voiceover, audio_uri = future.result()
                    if audio_uri:
                        tts_cache[voiceover] = audio_uri
                except Exception as e:
                    print(f"TTS future error: {e}")
                completed += 1
                progress.progress(completed / total_tasks)
    
    # Download images in parallel (up to 6 concurrent requests)
    if images_needed:
        status.write(f"🖼️ Loading {len(images_needed)} images in parallel...")
        
        def download_single_image(item):
            idx, asset = item
            try:
                resolved = normalize_drive_image_url(asset) if is_drive_url(asset) else asset
                image_bytes = download_image_bytes(resolved, drive)
                return asset, image_bytes_to_data_uri(image_bytes) if image_bytes else resolved
            except Exception as e:
                print(f"Image error for step {idx}: {e}")
                return asset, ""
        
        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = {executor.submit(download_single_image, item): item for item in images_needed}
            for future in as_completed(futures):
                try:
                    asset, image_uri = future.result()
                    if image_uri:
                        image_cache[asset] = image_uri
                except Exception as e:
                    print(f"Image future error: {e}")
                completed += 1
                progress.progress(completed / total_tasks)
    
    # Now assemble the prepared steps (fast, just lookups)
    status.write("📦 Assembling slideshow data...")
    prepared_steps = []
    for idx, step in enumerate(steps):
        voiceover = step.get("voiceover", "").strip()
        audio_data_uri = tts_cache.get(voiceover, "")
        
        asset = step.get("asset", "").strip()
        asset_type = step.get("asset_type", "unknown")
        image_data_uri = ""
        video_meta = None
        
        if asset_type == "image":
            image_data_uri = image_cache.get(asset, "")
            if not image_data_uri and asset:
                resolved = normalize_drive_image_url(asset) if is_drive_url(asset) else asset
                image_data_uri = resolved
        elif asset_type == "video":
            video_meta = parse_youtube_embed(asset)
        
        prepared = dict(step)
        prepared["audio_data_uri"] = audio_data_uri
        prepared["image_data_uri"] = image_data_uri
        prepared["video_meta"] = video_meta
        prepared_steps.append(prepared)
        completed += 1
        progress.progress(completed / total_tasks)
    
    status.empty()
    progress.empty()
    return prepared_steps


def build_slideshow_html(prepared_steps):
    steps_payload = []
    for step in prepared_steps:
        video_meta = step.get("video_meta") or {}
        steps_payload.append(
            {
                "flatIndex": step.get("flat_index"),
                "slideIndex": step.get("slide_index"),
                "slideTotal": step.get("slide_total"),
                "segmentIndex": step.get("segment_index"),
                "segmentTotal": step.get("segment_total"),
                "stepIndex": step.get("step_index"),
                "stepTotal": step.get("step_total"),
                "voiceover": step.get("voiceover", ""),
                "instruction": step.get("instruction", ""),
                "justification": step.get("justification", ""),
                "assetType": step.get("asset_type", "unknown"),
                "assetUrl": step.get("asset", ""),
                "imageData": step.get("image_data_uri", ""),
                "audioData": step.get("audio_data_uri", ""),
                "videoId": video_meta.get("video_id"),
                "videoStart": video_meta.get("start", 0.0) if video_meta else 0.0,
                "videoEnd": video_meta.get("end", None) if video_meta else None,
            }
        )
    payload = json.dumps({"steps": steps_payload}, ensure_ascii=True)

    return f"""
    <div id="gdv2-slideshow">
      <div id="gdv2-preload">
        <div id="gdv2-preload-text">Preparing assets...</div>
        <div id="gdv2-preload-bar"><div id="gdv2-preload-bar-inner"></div></div>
      </div>
      <div id="gdv2-status"></div>
      <div id="gdv2-progress">
        <div id="gdv2-progress-bar"></div>
      </div>
      <div id="gdv2-voiceover"></div>
      <div id="gdv2-visual">
        <img id="gdv2-image" />
        <div id="gdv2-video-container"></div>
      </div>
      <div id="gdv2-controls">
        <button id="gdv2-start">Start</button>
        <button id="gdv2-pause" disabled>Pause</button>
        <button id="gdv2-resume" disabled>Resume</button>
        <button id="gdv2-prev-slide" disabled>Previous Slide</button>
        <button id="gdv2-next-slide" disabled>Next Slide</button>
        <button id="gdv2-prev-step" disabled>Previous VO Part</button>
        <button id="gdv2-next-step" disabled>Next VO Part</button>
      </div>
    </div>
    <style>
      #gdv2-slideshow {{
        font-family: Arial, sans-serif;
        color: #111;
        padding: 8px;
      }}
      #gdv2-preload {{
        border: 1px solid #ccc;
        padding: 8px;
        margin-bottom: 8px;
      }}
      #gdv2-preload-bar {{
        background: #eee;
        height: 8px;
        border-radius: 4px;
        overflow: hidden;
        margin-top: 6px;
      }}
      #gdv2-preload-bar-inner {{
        background: #1f77b4;
        height: 100%;
        width: 0%;
      }}
      #gdv2-status {{
        margin: 8px 0;
        font-weight: 600;
      }}
      #gdv2-progress {{
        width: 100%;
        max-width: 980px;
        height: 8px;
        background: #e6e9ee;
        border-radius: 999px;
        margin: 6px auto 10px;
        overflow: hidden;
      }}
      #gdv2-progress-bar {{
        height: 100%;
        width: 0%;
        background: #2a6fdf;
        transition: width 120ms linear;
      }}
      #gdv2-voiceover {{
        margin: 12px 0 8px;
        font-size: 18px;
        font-weight: 600;
        background: #ffffff;
        color: #222;
        border: 1px solid #d7dce2;
        border-radius: 10px;
        padding: 12px 16px;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.08);
      }}
      #gdv2-visual {{
        display: flex;
        align-items: center;
        justify-content: center;
        background: #f7f7f7;
        border: 1px solid #ddd;
        width: 100%;
        max-width: 980px;
        aspect-ratio: 16 / 9;
        margin: 0 auto;
        overflow: hidden;
        position: relative;
      }}
      #gdv2-image {{
        max-width: 100%;
        max-height: 100%;
        width: auto;
        height: auto;
        object-fit: contain;
        display: none;
      }}
      #gdv2-video-container {{
        position: absolute;
        inset: 0;
        width: 100%;
        height: 100%;
        display: none;
      }}
      #gdv2-video-container.active {{
        display: block;
      }}
      .gdv2-video-frame {{
        width: 100%;
        height: 100%;
        position: absolute;
        inset: 0;
        opacity: 0;
        visibility: hidden;
        transition: opacity 120ms linear;
      }}
      .gdv2-video-frame.active {{
        opacity: 1;
        visibility: visible;
        z-index: 1;
      }}
      #gdv2-controls {{
        margin-top: 10px;
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
      }}
      #gdv2-controls button {{
        padding: 6px 10px;
        font-size: 14px;
        cursor: pointer;
      }}
    </style>
    <script>
      const payload = {payload};
      const steps = payload.steps || [];
      const audioElements = [];
      const imageElements = [];
      const videoPlayers = {{}};
      let readyCount = 0;
      let totalToLoad = 0;
      let currentIndex = -1;
      let isPlaying = false;
      let isPaused = false;
      let narrationDone = false;
      let videoEnded = false;
      let currentAudio = null;
      let currentVideo = null;
      let videoCheckInterval = null;

      const preloadText = document.getElementById("gdv2-preload-text");
      const preloadBar = document.getElementById("gdv2-preload-bar-inner");
      const statusEl = document.getElementById("gdv2-status");
      const progressBarEl = document.getElementById("gdv2-progress-bar");
      const voiceoverEl = document.getElementById("gdv2-voiceover");
      const imageEl = document.getElementById("gdv2-image");
      const videoContainer = document.getElementById("gdv2-video-container");

      const startBtn = document.getElementById("gdv2-start");
      const pauseBtn = document.getElementById("gdv2-pause");
      const resumeBtn = document.getElementById("gdv2-resume");
      const prevSlideBtn = document.getElementById("gdv2-prev-slide");
      const nextSlideBtn = document.getElementById("gdv2-next-slide");
      const prevStepBtn = document.getElementById("gdv2-prev-step");
      const nextStepBtn = document.getElementById("gdv2-next-step");

      function markReady() {{
        readyCount += 1;
        if (totalToLoad > 0) {{
          preloadBar.style.width = `${{Math.min(100, (readyCount / totalToLoad) * 100)}}%`;
        }}
        if (readyCount >= totalToLoad) {{
          preloadText.textContent = "Assets ready.";
          startBtn.disabled = steps.length === 0;
        }}
      }}

      function setupAudio() {{
        steps.forEach((step, index) => {{
          const audio = new Audio(step.audioData || "");
          audio.preload = "auto";
          audio.addEventListener("loadedmetadata", () => {{
            step.audioDuration = audio.duration || 0;
            markReady();
          }});
          audio.addEventListener("error", () => {{
            step.audioDuration = 0;
            markReady();
          }});
          audioElements[index] = audio;
        }});
      }}

      function setupImages() {{
        steps.forEach((step) => {{
          if (step.assetType !== "image" || !step.imageData) {{
            return;
          }}
          const img = new Image();
          img.onload = markReady;
          img.onerror = markReady;
          img.src = step.imageData;
          imageElements.push(img);
        }});
      }}

      function buildVideoContainers() {{
        steps.forEach((step, index) => {{
          if (step.assetType !== "video" || !step.videoId) {{
            return;
          }}
          const frame = document.createElement("div");
          frame.id = `gdv2-video-${{index}}`;
          frame.className = "gdv2-video-frame";
          videoContainer.appendChild(frame);
        }});
      }}

      function onYouTubeIframeAPIReady() {{
        steps.forEach((step, index) => {{
          if (step.assetType !== "video" || !step.videoId) {{
            return;
          }}
          const player = new YT.Player(`gdv2-video-${{index}}`, {{
            videoId: step.videoId,
            playerVars: {{
              start: step.videoStart || 0,
              end: step.videoEnd || undefined,
              controls: 0,
              rel: 0,
              enablejsapi: 1,
              playsinline: 1,
              modestbranding: 1,
              mute: 1
            }},
            events: {{
              onReady: (event) => {{
                try {{
                  const cueArgs = {{
                    videoId: step.videoId,
                    startSeconds: step.videoStart || 0
                  }};
                  if (step.videoEnd !== null && step.videoEnd !== undefined) {{
                    cueArgs.endSeconds = step.videoEnd;
                  }}
                  event.target.cueVideoById(cueArgs);
                }} catch (e) {{}}
                markReady();
              }},
              onError: () => markReady()
            }}
          }});
          videoPlayers[index] = player;
        }});
      }}

      function loadYouTubeApi() {{
        if (!steps.some(step => step.assetType === "video")) {{
          return;
        }}
        const tag = document.createElement("script");
        tag.src = "https://www.youtube.com/iframe_api";
        document.body.appendChild(tag);
      }}

      function updateStatus(step) {{
        if (!step) {{
          statusEl.textContent = "";
          progressBarEl.style.width = "0%";
          return;
        }}
        statusEl.textContent =
          `Slide ${{step.slideIndex}} of ${{step.slideTotal}}, Segment ${{step.segmentIndex}} of ${{step.segmentTotal}}, VO Part ${{step.stepIndex}} of ${{step.stepTotal}}`;
        const totalSteps = steps.length || 1;
        const progressPercent = ((currentIndex + 1) / totalSteps) * 100;
        progressBarEl.style.width = `${{Math.min(100, Math.max(0, progressPercent))}}%`;
      }}

      function showVisual(step, index) {{
        imageEl.style.display = "none";
        videoContainer.classList.remove("active");
        Array.from(videoContainer.children).forEach((child) => {{
          child.classList.remove("active");
        }});
        if (step.assetType === "image" && step.imageData) {{
          imageEl.src = step.imageData;
          imageEl.style.display = "block";
        }} else if (step.assetType === "video") {{
          const frame = document.getElementById(`gdv2-video-${{index}}`);
          if (frame) {{
            frame.classList.add("active");
            videoContainer.classList.add("active");
          }}
        }}
      }}

      function clearTimers() {{
        if (videoCheckInterval) {{
          clearInterval(videoCheckInterval);
          videoCheckInterval = null;
        }}
      }}

      function setupVideoMonitoring(step, index) {{
        if (!step || step.assetType !== "video") {{
          return;
        }}
        const player = videoPlayers[index];
        if (!player || step.videoEnd === null || step.videoEnd === undefined) {{
          return;
        }}
        videoEnded = false;
        videoCheckInterval = setInterval(() => {{
          let currentTime = 0;
          try {{
            currentTime = player.getCurrentTime();
          }} catch (e) {{
            return;
          }}
          if (currentTime >= step.videoEnd - 0.05) {{
            try {{
              player.pauseVideo();
              player.seekTo(step.videoEnd, true);
            }} catch (e) {{}}
            videoEnded = true;
            clearTimers();
            if (narrationDone && (step.videoEnd - (step.videoStart || 0)) > (step.audioDuration || 0)) {{
              advanceStep(1);
            }}
          }}
        }}, 200);
      }}

      function playStep(index) {{
        clearTimers();
        if (index < 0 || index >= steps.length) {{
          return;
        }}
        currentIndex = index;
        const step = steps[index];
        narrationDone = false;
        videoEnded = false;
        updateStatus(step);
        voiceoverEl.textContent = step.voiceover || "";
        showVisual(step, index);

        currentAudio = audioElements[index];
        if (currentAudio) {{
          currentAudio.pause();
          currentAudio.currentTime = 0;
          currentAudio.onended = () => {{
            narrationDone = true;
            if (step.assetType !== "video") {{
              advanceStep(1);
              return;
            }}
            const videoDuration = (step.videoEnd || 0) - (step.videoStart || 0);
            if (videoDuration <= (step.audioDuration || 0) || videoEnded) {{
              advanceStep(1);
            }}
          }};
          currentAudio.play();
        }} else {{
          narrationDone = true;
        }}

        if (step.assetType === "video") {{
          currentVideo = videoPlayers[index];
          if (currentVideo) {{
            try {{
              currentVideo.mute();
              currentVideo.seekTo(step.videoStart || 0, true);
              currentVideo.playVideo();
            }} catch (e) {{}}
            setupVideoMonitoring(step, index);
          }}
        }} else {{
          currentVideo = null;
        }}
      }}

      function advanceStep(delta) {{
        if (!isPlaying) {{
          return;
        }}
        const nextIndex = currentIndex + delta;
        if (nextIndex < 0 || nextIndex >= steps.length) {{
          isPlaying = false;
          pauseBtn.disabled = true;
          resumeBtn.disabled = true;
          prevSlideBtn.disabled = false;
          nextSlideBtn.disabled = false;
          prevStepBtn.disabled = false;
          nextStepBtn.disabled = false;
          return;
        }}
        playStep(nextIndex);
      }}

      function pausePlayback() {{
        if (!isPlaying) {{
          return;
        }}
        isPaused = true;
        if (currentAudio) {{
          currentAudio.pause();
        }}
        if (currentVideo) {{
          try {{
            currentVideo.pauseVideo();
          }} catch (e) {{}}
        }}
        clearTimers();
        pauseBtn.disabled = true;
        resumeBtn.disabled = false;
      }}

      function resumePlayback() {{
        if (!isPlaying) {{
          return;
        }}
        isPaused = false;
        if (currentAudio) {{
          currentAudio.play();
        }}
        const step = steps[currentIndex];
        if (step && step.assetType === "video" && currentVideo) {{
          try {{
            currentVideo.playVideo();
          }} catch (e) {{}}
          setupVideoMonitoring(step, currentIndex);
        }}
        pauseBtn.disabled = false;
        resumeBtn.disabled = true;
      }}

      function getSlideStartIndex(targetSlide) {{
        let found = null;
        steps.forEach((step, index) => {{
          if (step.slideIndex === targetSlide && found === null) {{
            found = index;
          }}
        }});
        return found;
      }}

      function navigateSlide(delta) {{
        if (currentIndex < 0 || steps.length === 0) {{
          return;
        }}
        const currentSlide = steps[currentIndex].slideIndex;
        const targetSlide = currentSlide + delta;
        if (targetSlide < 1 || targetSlide > steps[currentIndex].slideTotal) {{
          return;
        }}
        const startIndex = getSlideStartIndex(targetSlide);
        if (startIndex !== null) {{
          playStep(startIndex);
        }}
      }}

      startBtn.addEventListener("click", () => {{
        if (steps.length === 0 || startBtn.disabled) {{
          return;
        }}
        isPlaying = true;
        isPaused = false;
        pauseBtn.disabled = false;
        resumeBtn.disabled = true;
        prevSlideBtn.disabled = false;
        nextSlideBtn.disabled = false;
        prevStepBtn.disabled = false;
        nextStepBtn.disabled = false;
        playStep(0);
      }});
      pauseBtn.addEventListener("click", pausePlayback);
      resumeBtn.addEventListener("click", resumePlayback);
      prevStepBtn.addEventListener("click", () => advanceStep(-1));
      nextStepBtn.addEventListener("click", () => advanceStep(1));
      prevSlideBtn.addEventListener("click", () => navigateSlide(-1));
      nextSlideBtn.addEventListener("click", () => navigateSlide(1));

      function init() {{
        totalToLoad = steps.length;
        const imageLoadCount = steps.filter(step => step.assetType === "image" && step.imageData).length;
        const videoLoadCount = steps.filter(step => step.assetType === "video" && step.videoId).length;
        totalToLoad = steps.length + imageLoadCount + videoLoadCount;
        if (steps.length === 0) {{
          preloadText.textContent = "No steps to play.";
          return;
        }}
        setupAudio();
        setupImages();
        buildVideoContainers();
        loadYouTubeApi();
      }}

      window.onYouTubeIframeAPIReady = onYouTubeIframeAPIReady;
      init();
    </script>
    """


def render_looping_youtube_embed(url, key, height=320):
    meta = parse_youtube_embed(url)
    if not meta:
        return ""
    video_id = meta["video_id"]
    start = meta.get("start") or 0
    end = meta.get("end")
    html = f"""
    <div id="yt-container-{key}">
      <div id="yt-player-{key}"></div>
    </div>
    <script>
      (function() {{
        let player;
        window.onYouTubeIframeAPIReady = function() {{
          player = new YT.Player("yt-player-{key}", {{
            height: "{height}",
            width: "100%",
            videoId: "{video_id}",
            playerVars: {{
              start: {int(start)},
              end: {int(end) if end is not None else "undefined"},
              autoplay: 1,
              controls: 1,
              rel: 0,
              playsinline: 1,
              mute: 1
            }},
            events: {{
              onReady: function(event) {{
                try {{
                  event.target.playVideo();
                }} catch (e) {{}}
              }}
            }}
          }});
          setInterval(function() {{
            if (!player || !player.getCurrentTime) {{
              return;
            }}
            const current = player.getCurrentTime();
            const endTime = {int(end) if end is not None else "null"};
            if (endTime !== null && current >= endTime - 0.1) {{
              player.seekTo({int(start)}, true);
            }}
          }}, 500);
        }};
        const tag = document.createElement("script");
        tag.src = "https://www.youtube.com/iframe_api";
        document.body.appendChild(tag);
      }})();
    </script>
    """
    return html


def render_inspector(slides, column_map, drive):
    if not slides:
        st.info("No slide data to display.")
        return

    for slide_idx, slide in enumerate(slides, start=1):
        title_parts = [f"Slide {slide_idx}"]
        if slide.get("slide_title"):
            title_parts.append(slide["slide_title"])
        if slide.get("topic") or slide.get("subtopic"):
            title_parts.append(f"{slide.get('topic', '')} / {slide.get('subtopic', '')}".strip(" /"))
        slide_title = " - ".join([part for part in title_parts if part])
        with st.expander(slide_title, expanded=False):
            if slide.get("slide_chunk"):
                st.markdown("**Slide Content**")
                st.write(slide["slide_chunk"])

            segments = slide.get("segments", [])
            if not segments:
                st.warning("No visual steps parsed for this slide.")
                if slide.get("final_definition_raw"):
                    st.markdown("**Raw Definition**")
                    st.text(slide["final_definition_raw"])
                continue

            for segment in segments:
                st.markdown(f"### Segment {segment['segment_index']}")
                for step in segment.get("steps", []):
                    st.markdown(f"**VO Part {step['step_index']}**")
                    if step.get("voiceover"):
                        st.write(step["voiceover"])
                    else:
                        st.caption("Voiceover text not found.")

                    asset = step.get("asset", "")
                    asset_type = detect_asset_type(asset)
                    if asset_type == "image":
                        display_url = normalize_drive_image_url(asset) if is_drive_url(asset) else asset
                        image_bytes = None
                        if is_drive_url(asset) and drive is not None:
                            image_bytes = download_image_bytes(asset, drive)
                        if image_bytes:
                            st.image(image_bytes, use_container_width=True)
                        elif display_url:
                            st.image(display_url, use_container_width=True)
                        else:
                            st.warning("Image URL missing.")
                        if display_url:
                            st.markdown(f"[Open image]({display_url})")
                    elif asset_type == "video":
                        if asset:
                            html = render_looping_youtube_embed(
                                asset,
                                f"{slide_idx}-{segment['segment_index']}-{step['step_index']}",
                            )
                            if html:
                                st.components.v1.html(html, height=360)
                            st.markdown(f"[Open video]({asset})")
                        else:
                            st.warning("Video URL missing.")

                    if step.get("instruction"):
                        st.markdown("**Instruction**")
                        st.write(step["instruction"])
                    if step.get("justification"):
                        st.markdown("**Justification**")
                        st.write(step["justification"])


def main():
    try:
        st.set_page_config(
            page_title="Graphics Definition V2 Slideshow",
            page_icon="",
            layout="wide",
        )
    except Exception:
        pass

    st.title("Graphics Definition V2 Slideshow")
    st.caption("Preview and playback for final graphics definitions with narration sync.")

    if "gc" not in st.session_state:
        st.error("Google Sheets client not found. Please log in first.")
        return

    gc = st.session_state["gc"]
    drive = st.session_state.get("drive")

    sheet_link_default = st.session_state.get("sheet_link", "")
    sheet_link = st.text_input("Google Sheet link", value=sheet_link_default)
    if not sheet_link:
        st.info("Enter a Google Sheet link to load Slide Chunks data.")
        return

    load_sheet = st.button("Load Sheet")
    if load_sheet or st.session_state.get("gdv2_sheet_link") != sheet_link:
        try:
            sheet = gc.open_by_url(sheet_link)
            st.session_state["gdv2_sheet"] = sheet
            st.session_state["gdv2_sheet_link"] = sheet_link
            st.session_state["gdv2_df"] = None
            st.session_state["gdv2_df_sheet"] = None
            st.session_state["gdv2_df_sheet_link"] = sheet_link
        except Exception as e:
            st.error(f"Failed to open sheet: {e}")
            return

    sheet = st.session_state.get("gdv2_sheet")
    if not sheet:
        st.info("Load a sheet to continue.")
        return

    worksheet_names = get_worksheet_names(sheet)
    worksheet_name = st.selectbox(
        "Worksheet",
        options=worksheet_names,
        index=worksheet_names.index(DEFAULT_SHEET_NAME)
        if DEFAULT_SHEET_NAME in worksheet_names
        else 0,
    )

    refresh = st.button("Refresh Data")
    if (
        refresh
        or st.session_state.get("gdv2_df_sheet") != worksheet_name
        or st.session_state.get("gdv2_df_sheet_link") != sheet_link
    ):
        try:
            _, df = get_sheet_data_and_df(sheet, worksheet_name)
            st.session_state["gdv2_df"] = df
            st.session_state["gdv2_df_sheet"] = worksheet_name
            st.session_state["gdv2_df_sheet_link"] = sheet_link
        except Exception as e:
            st.error(f"Failed to load data: {e}")
            return

    df = st.session_state.get("gdv2_df")
    if df is None or df.empty:
        st.warning("No data found in the selected worksheet.")
        return

    column_map = {
        "topic": find_column(df, "Topic"),
        "subtopic": find_column(df, "Subtopic"),
        "slide_title": find_column(df, "Slide Chunk Title"),
        "slide_chunk": find_column(df, "Slide Chunk"),
        "final_def": find_column(df, FINAL_GRAPHICS_COLUMN),
    }

    if not column_map["final_def"]:
        st.error(f"Column '{FINAL_GRAPHICS_COLUMN}' not found in worksheet.")
        return

    topic_values = (
        sorted({safe_str(v) for v in df[column_map["topic"]].dropna()})
        if column_map["topic"]
        else []
    )
    subtopic_values = (
        sorted({safe_str(v) for v in df[column_map["subtopic"]].dropna()})
        if column_map["subtopic"]
        else []
    )

    filter_col1, filter_col2, filter_col3 = st.columns(3)
    with filter_col1:
        topic_filter = st.selectbox("Topic", options=["All"] + topic_values)
    with filter_col2:
        subtopic_options = ["All"] + subtopic_values
        if topic_filter != "All" and column_map["topic"] and column_map["subtopic"]:
            subtopic_options = ["All"] + sorted(
                {
                    safe_str(v)
                    for v in df[df[column_map["topic"]] == topic_filter][
                        column_map["subtopic"]
                    ].dropna()
                }
            )
        subtopic_filter = st.selectbox("Subtopic", options=subtopic_options)
    with filter_col3:
        search_text = st.text_input("Search", value="")

    filtered_df = df.copy()
    if topic_filter != "All" and column_map["topic"]:
        filtered_df = filtered_df[filtered_df[column_map["topic"]] == topic_filter]
    if subtopic_filter != "All" and column_map["subtopic"]:
        filtered_df = filtered_df[filtered_df[column_map["subtopic"]] == subtopic_filter]
    if search_text:
        search_lower = search_text.lower()

        def row_matches(row):
            combined = " ".join(
                safe_str(row.get(col, "")) for col in column_map.values() if col
            ).lower()
            return search_lower in combined

        filtered_df = filtered_df[filtered_df.apply(row_matches, axis=1)]

    slides = build_slides_from_df(filtered_df, column_map)
    steps = flatten_steps(slides)

    if "gdv2_slideshow_key" in st.session_state:
        current_key = compute_slideshow_key(steps)
        if st.session_state.get("gdv2_slideshow_key") != current_key:
            st.session_state["gdv2_prepared_steps"] = None
            st.session_state["gdv2_slideshow_key"] = current_key
    else:
        st.session_state["gdv2_slideshow_key"] = compute_slideshow_key(steps)

    tabs = st.tabs(["Inspector Mode", "Slideshow Mode"])
    with tabs[0]:
        render_inspector(slides, column_map, drive)

    with tabs[1]:
        if not steps:
            st.info("No visual steps available for slideshow.")
            return

        if "OPENAI_API_KEY" not in os.environ:
            st.warning("OPENAI_API_KEY not set. Slideshow requires OpenAI TTS.")
            return

        prep_col1, prep_col2 = st.columns([1, 3])
        with prep_col1:
            prepare = st.button("Prepare Slideshow Assets")
        with prep_col2:
            st.caption("Preloads images, audio, and video players before playback.")

        if prepare:
            client = get_openai_client()
            prepared_steps = prepare_slideshow_assets(steps, drive, client)
            st.session_state["gdv2_prepared_steps"] = prepared_steps
            st.success("Assets prepared.")

        prepared_steps = st.session_state.get("gdv2_prepared_steps")
        if not prepared_steps:
            st.info("Prepare assets to enable playback.")
            return

        slideshow_html = build_slideshow_html(prepared_steps)
        st.components.v1.html(slideshow_html, height=760, scrolling=True)


if __name__ == "__main__":
    main()
else:
    main()
