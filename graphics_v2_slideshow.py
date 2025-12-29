"""
Graphics Definition V2 Slideshow Preview
Preview images from Graphics Definition V2 output in a slideshow format
"""

import os
import streamlit as st
import gspread
import pandas as pd
import json
import re
import base64
from io import BytesIO
import requests
from PIL import Image
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from openai import OpenAI

from services.sheets_service import get_sheet_data_and_df

# Import parsing function from populate_sheet_with_selected_images
from agents.graphics_definition_v2.populate_sheet_with_selected_images import parse_graphics_definition

# =============================================================================
# AUTHENTICATION CHECK
# =============================================================================

if "gc" not in st.session_state:
    st.error("❌ You are not authenticated yet. Please use the Login in the main app, then return here.")
    st.stop()

gc = st.session_state["gc"]
drive = st.session_state.get("drive")  # Get Drive instance for authenticated downloads

# =============================================================================
# OPENAI TTS CLIENT
# =============================================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("❌ OPENAI_API_KEY is not set in the environment. Please load it from your .env before running this app.")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)


@lru_cache(maxsize=2000)
def generate_tts_audio_b64(vo_text: str) -> str:
    """
    Generate natural-sounding audio (MP3) for a VO segment using OpenAI TTS.
    Returns a base64-encoded string. Uses LRU cache to avoid repeated calls for the same text.
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
    except Exception as e:
        # Log for debugging, fail silently for the UI
        print(f"[TTS ERROR] Failed for text '{vo_text[:80]}...': {e}")
        return ""


# =============================================================================
# IMAGE LOADING & CACHING
# =============================================================================

@st.cache_resource
def get_requests_session():
    """Create and cache a requests session for connection pooling."""
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    })
    return session


def extract_file_id_from_url(url: str) -> Optional[str]:
    """Extract Google Drive file ID from URL."""
    if not url:
        return None

    # Pattern for /file/d/FILE_ID/
    match = re.search(r'/file/d/([a-zA-Z0-9_-]+)', url)
    if match:
        return match.group(1)

    # Pattern for ?id=FILE_ID or &id=FILE_ID
    match = re.search(r'[?&]id=([a-zA-Z0-9_-]+)', url)
    if match:
        return match.group(1)

    return None


def is_drive_url(url: str) -> bool:
    """Check if URL is a Google Drive URL."""
    return bool(re.search(r'drive\.google\.com', url, re.IGNORECASE))


def download_image_from_web_url(web_url: str) -> Optional[Image.Image]:
    """Download image from a direct web URL."""
    try:
        session = get_requests_session()
        response = session.get(web_url, timeout=10, stream=True, allow_redirects=True)
        
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            img = img.convert("RGB")
            # Smaller thumbnail for faster processing
            img.thumbnail((600, 600), Image.Resampling.LANCZOS)
            return img
        
        return None
    except Exception as e:
        # Log error for debugging (but don't expose to user)
        print(f"Failed to download image from web URL {web_url}: {e}")
        return None


def download_image_from_drive_url(drive_url: str, drive=None) -> Optional[Image.Image]:
    """Download and create thumbnail for faster display using authenticated Drive API."""
    try:
        file_id = extract_file_id_from_url(drive_url)
        if not file_id:
            return None

        # Try authenticated Drive API first (more reliable)
        if drive:
            try:
                import tempfile
                import os

                file = drive.CreateFile({'id': file_id})
                file.FetchMetadata(fields='title, mimeType')

                # Download to temporary file (Drive API requires file path)
                temp_path = None
                try:
                    temp_path = os.path.join(tempfile.gettempdir(), f"{file_id}.jpg")
                    file.GetContentFile(temp_path)

                    # Open with PIL and convert to RGB
                    with Image.open(temp_path) as img:
                        pil_img = img.convert("RGB").copy()
                        # Smaller thumbnail for faster processing
                        pil_img.thumbnail((600, 600), Image.Resampling.LANCZOS)
                        return pil_img
                finally:
                    # Clean up temp file
                    if temp_path and os.path.exists(temp_path):
                        try:
                            os.remove(temp_path)
                        except:
                            pass
            except Exception as e:
                # Fall back to public URL if Drive API fails
                print(f"Drive API failed for {file_id}, trying public URL: {e}")
                pass

        # Fallback to public download URL (for publicly shared files)
        download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        session = get_requests_session()
        response = session.get(download_url, timeout=10, stream=True, allow_redirects=True)

        # Handle virus scan warning page
        if response.status_code == 200:
            content = response.content
            # Check if we got an HTML page (virus scan warning)
            if b'<html' in content[:1000].lower() or b'virus scan' in content.lower():
                # Try the confirm download link
                confirm_match = re.search(r'href="(/uc\?export=download[^"]+)"', content.decode('utf-8', errors='ignore'))
                if confirm_match:
                    confirm_url = "https://drive.google.com" + confirm_match.group(1)
                    response = session.get(confirm_url, timeout=10, stream=True)
                    if response.status_code == 200:
                        content = response.content

            if content and not (b'<html' in content[:1000].lower()):
                img = Image.open(BytesIO(content))
                img.thumbnail((600, 600), Image.Resampling.LANCZOS)
                return img

        return None
    except Exception as e:
        # Log error for debugging (but don't expose to user)
        print(f"Failed to download image from {drive_url}: {e}")
        return None


def download_image_from_url(url: str, drive=None) -> Optional[Image.Image]:
    """
    Download image from either Google Drive URL or web URL.
    
    Args:
        url: Image URL (can be Google Drive or web URL)
        drive: Google Drive instance (optional, for Drive images)
    
    Returns:
        PIL Image or None if download fails
    """
    if not url or not url.startswith('http'):
        return None
    
    if is_drive_url(url):
        return download_image_from_drive_url(url, drive=drive)
    else:
        return download_image_from_web_url(url)


def image_to_base64(img: Image.Image) -> str:
    """Convert PIL Image to base64 string for HTML embedding."""
    buffered = BytesIO()
    # Convert to RGB if necessary and use JPEG with quality setting for smaller size
    if img.mode in ('RGBA', 'LA', 'P'):
        # Convert RGBA to RGB with white background
        rgb_img = Image.new('RGB', img.size, (255, 255, 255))
        rgb_img.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
        rgb_img.save(buffered, format="JPEG", quality=85, optimize=True)
    else:
        img.save(buffered, format="JPEG", quality=85, optimize=True)
    return base64.b64encode(buffered.getvalue()).decode()


# =============================================================================
# SHEET LOADING
# =============================================================================

def ensure_sheet_loaded():
    """Ensure the Google Sheet is loaded in session state."""
    if "graphics_v2_sheet" in st.session_state:
        return True

    # Single input section - prevent duplicates
    default_link = st.session_state.get("graphics_v2_sheet_link", "")
    
    sheet_link = st.text_input(
        "📋 Enter Google Sheet link", 
        value=default_link,
        placeholder="https://docs.google.com/spreadsheets/d/...",
        key="graphics_slideshow_sheet_input_single"
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        load_button = st.button("Load Data", type="primary", key="graphics_slideshow_load_button_single")
    
    if load_button:
        if not sheet_link.strip():
            st.error("Please paste a valid Google Sheet link.")
        else:
            try:
                sheet = gc.open_by_url(sheet_link.strip())
                st.session_state["graphics_v2_sheet"] = sheet
                st.session_state["graphics_v2_sheet_link"] = sheet_link.strip()
                st.success("✅ Data loaded successfully!")
                st.rerun()
            except Exception as exc:
                st.error(f"Failed to open sheet: {exc}")

    st.caption("💡 Tip: Load the sheet containing the 'Slide Chunks' worksheet with 'graphics_definition' column.")
    
    return False


# =============================================================================
# INSPECTOR MODE
# =============================================================================

def render_inspector_mode(filtered_df):
    """Render Inspector mode for browsing slides and images."""
    st.markdown("#### 🔍 Slides Inspector")

    # Find graphics_definition column
    gd_column = None
    for col in filtered_df.columns:
        if "graphics_definition" in col.lower() or "graphics definition" in col.lower():
            gd_column = col
            break
    
    if not gd_column:
        st.error("❌ No 'graphics_definition' column found in filtered data.")
        return

    # Pre-load all images with progress bar
    all_urls = []
    url_to_location = {}  # Map URL to (row_index, segment_idx, image_idx)

    for row_index, row in filtered_df.iterrows():
        gd_content = str(row.get(gd_column, "")).strip()
        if not gd_content:
            continue

        segments = parse_graphics_definition(gd_content)
        for seg_idx, segment in enumerate(segments):
            for img_idx, img in enumerate(segment.get("images", [])):
                url = img.get("url", "")
                if url:
                    url_to_location[url] = (row_index, seg_idx, img_idx)
                    all_urls.append(url)

    # Download all images in parallel with progress bar
    image_cache = {}
    if all_urls:
        progress_container = st.container()
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()

        completed = 0
        total = len(all_urls)

        def download_single(url):
            try:
                img = download_image_from_url(url, drive=drive)
                return (url, img)
            except Exception:
                return (url, None)

        with ThreadPoolExecutor(max_workers=30) as executor:
            futures = [executor.submit(download_single, url) for url in all_urls]

            for future in as_completed(futures):
                url, img = future.result()
                image_cache[url] = img

                completed += 1
                percentage = int((completed / total) * 100)
                progress_bar.progress(completed / total)
                status_text.markdown(f"**Loading images... {percentage}%** ({completed}/{total})")

        progress_container.empty()

    # Display slides with cached images
    for row_index, row in filtered_df.iterrows():
        title = str(row.get("Slide Chunk Title") or row.get("Topic") or "Slide")

        with st.expander(f"🎯 {title}", expanded=False):
            st.caption(f"📌 Topic: {row.get('Topic', '')} | Subtopic: {row.get('Subtopic', '')}")

            with st.container(border=True):
                st.markdown("**📄 Slide Content**")
                st.write(str(row.get("Slide Chunk", "")).strip() or "(empty)")

            gd_content = str(row.get(gd_column, "")).strip()
            if not gd_content:
                st.info("No graphics_definition found for this slide.")
                continue

            segments = parse_graphics_definition(gd_content)
            if not segments:
                st.info("No segments found in graphics_definition.")
                continue

            for seg_idx, segment in enumerate(segments, 1):
                with st.container(border=True):
                    vo_text = segment.get("vo", "")
                    st.markdown(f"**💬 Segment {seg_idx} VO**: {vo_text}")

                    images = segment.get("images", [])
                    if images:
                        cols = st.columns(min(3, len(images)))

                        for col_idx, img_data in enumerate(images):
                            with cols[col_idx % len(cols)]:
                                url = img_data.get("url", "")
                                title = img_data.get("title", f"Image {col_idx + 1}")

                                img = image_cache.get(url)
                                if img:
                                    st.image(img, use_container_width=True, caption=title)
                                else:
                                    st.error("Failed to load")
                    else:
                        st.warning("⚠️ No images for this segment")


# =============================================================================
# SLIDESHOW MODE
# =============================================================================

def render_slideshow_mode(filtered_df):
    """Render Slideshow mode with automatic narration and image transitions."""
    st.markdown("#### 🎬 Slideshow Preview")
    st.info("📢 Click **Start** to begin automatic narration. Images will appear for each segment with its VO text.")

    # Find graphics_definition column
    gd_column = None
    for col in filtered_df.columns:
        if "graphics_definition" in col.lower() or "graphics definition" in col.lower():
            gd_column = col
            break
    
    if not gd_column:
        st.error("❌ No 'graphics_definition' column found in filtered data.")
        return

    # Build playlist grouped by slides and segments
    slides_data = []  # List of slides with their segments
    image_data = {}   # Store base64 images for each segment

    # First pass: collect all URLs and build slide structure
    all_urls = []
    url_to_title = {}  # Map URL to title
    url_map = {}  # Map (slide_idx, segment_idx, image_idx) -> global_url_index

    for slide_idx, row in filtered_df.iterrows():
        slide_title = str(row.get("Slide Chunk Title") or row.get("Topic") or "Slide")
        topic = str(row.get("Topic") or "")

        gd_content = str(row.get(gd_column, "")).strip()
        if not gd_content:
            continue

        segments_list = parse_graphics_definition(gd_content)
        if not segments_list:
            continue

        segments = []
        for seg_idx, segment in enumerate(segments_list):
            vo_text = segment.get("vo", "")
            images = segment.get("images", [])

            segment_id = f"{slide_idx}_{seg_idx}"

            # Collect URLs for this segment
            segment_url_indices = []
            if images:
                for img_idx, img_data in enumerate(images):
                    url = img_data.get("url", "")
                    if url:
                        global_idx = len(all_urls)
                        all_urls.append(url)
                        # Store title for this URL
                        title = img_data.get("title", f"Image {img_idx + 1}")
                        url_to_title[url] = title
                        url_map[(slide_idx, seg_idx, img_idx)] = global_idx
                        segment_url_indices.append(global_idx)

            image_data[segment_id] = []  # Will be populated later

            segments.append({
                "vo": vo_text,
                "segmentId": segment_id,
                "hasImages": len(segment_url_indices) > 0,
                "urlIndices": segment_url_indices
            })

        slides_data.append({
            "slideTitle": slide_title,
            "topic": topic,
            "segments": segments
        })

    # Second pass: Download all images in parallel with progress bar
    if all_urls:
        progress_container = st.container()
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()

        downloaded_images = [None] * len(all_urls)
        completed = 0
        total = len(all_urls)

        def download_single(idx_url):
            idx, url = idx_url
            try:
                img = download_image_from_url(url, drive=drive)
                if img:
                    return (idx, image_to_base64(img))
                return (idx, None)
            except Exception:
                return (idx, None)

        # Use 30 workers for faster parallel downloads
        with ThreadPoolExecutor(max_workers=30) as executor:
            futures = [executor.submit(download_single, (i, url)) for i, url in enumerate(all_urls)]

            for future in as_completed(futures):
                idx, img_b64 = future.result()
                downloaded_images[idx] = img_b64

                completed += 1
                percentage = int((completed / total) * 100)
                progress_bar.progress(completed / total)
                status_text.markdown(f"**Loading images... {percentage}%**")

        progress_container.empty()

        # Third pass: Assign images and titles to segments
        for slide_idx, slide in enumerate(slides_data):
            for seg_idx, segment in enumerate(slide["segments"]):
                segment_id = segment["segmentId"]
                url_indices = segment.get("urlIndices", [])
                # Store both image and title for each image
                image_items = []
                for url_idx in url_indices:
                    if downloaded_images[url_idx] is not None:
                        url = all_urls[url_idx]
                        title = url_to_title.get(url, f"Image {len(image_items) + 1}")
                        image_items.append({
                            "src": downloaded_images[url_idx],
                            "title": title
                        })
                image_data[segment_id] = image_items
                segment["imageCount"] = len(image_items)
                segment["hasImages"] = len(image_items) > 0

    if not slides_data:
        st.info("No slides found to present.")
        return

    # Generate TTS audio for each segment
    audio_map = {}
    for slide in slides_data:
        for segment in slide["segments"]:
            seg_id = segment["segmentId"]
            vo_text = segment.get("vo", "")
            audio_b64 = generate_tts_audio_b64(vo_text)
            audio_map[seg_id] = audio_b64

    slides_json = json.dumps(slides_data)

    # Embed images as data URIs in HTML with titles
    # Convert image_data to proper format for JSON serialization
    images_dict = {}
    for seg_id, img_items in image_data.items():
        if img_items:
            images_dict[seg_id] = [
                {
                    "src": f"data:image/jpeg;base64,{item['src']}",
                    "title": item["title"]
                }
                for item in img_items
            ]
        else:
            images_dict[seg_id] = []
    
    images_json = json.dumps(images_dict)
    audio_json = json.dumps(audio_map)

    html = f"""
<div style="display:flex;flex-direction:column;gap:12px;font-family:system-ui,-apple-system,sans-serif;">
    <div id="controls" style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;">
        <button id="btnStart" style="padding:8px 16px;background:#0066ff;color:white;border:none;border-radius:4px;cursor:pointer;font-weight:500;">▶️ Start</button>
        <button id="btnPause" style="padding:8px 16px;background:#666;color:white;border:none;border-radius:4px;cursor:pointer;">⏸️ Pause</button>
        <button id="btnResume" style="padding:8px 16px;background:#0066ff;color:white;border:none;border-radius:4px;cursor:pointer;display:none;">▶️ Resume</button>
        <button id="btnPrev" style="padding:8px 16px;background:#444;color:white;border:none;border-radius:4px;cursor:pointer;">⏮️ Previous Slide</button>
        <button id="btnNext" style="padding:8px 16px;background:#444;color:white;border:none;border-radius:4px;cursor:pointer;">⏭️ Next Slide</button>
        <button id="btnPrevSeg" style="padding:8px 16px;background:#888;color:white;border:none;border-radius:4px;cursor:pointer;">◀️ Prev Segment</button>
        <button id="btnNextSeg" style="padding:8px 16px;background:#888;color:white;border:none;border-radius:4px;cursor:pointer;">▶️ Next Segment</button>
        <span id="status" style="margin-left:8px;color:#555;font-size:14px;"></span>
    </div>
    
    <div id="currentVO" style="background:#f0f8ff;padding:20px;border-radius:8px;border:2px solid #0066ff;font-size:18px;line-height:1.6;min-height:80px;display:flex;align-items:center;justify-content:center;text-align:center;transition:opacity 0.3s ease;">
        <i style="color:#999;">Click Start to begin...</i>
    </div>
    
    <div id="imageGallery" style="display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px;min-height:400px;max-height:500px;background:#f5f5f5;padding:20px;border-radius:8px;transition:opacity 0.3s ease;overflow:auto;">
        <div style="grid-column:1/-1;text-align:center;color:#999;padding:40px;">No images to display</div>
    </div>
    
    <div id="progressBar" style="width:100%;height:8px;background:#e0e0e0;border-radius:4px;overflow:hidden;">
        <div id="progressFill" style="width:0%;height:100%;background:#0066ff;transition:width 0.3s;"></div>
    </div>
</div>

<script>
(function() {{
    var slides = {slides_json};
    var imageData = {images_json};
    var audioData = {audio_json};
    var currentSlide = 0;
    var currentSegment = 0;
    var paused = false;
    var segmentTimers = [];
    var currentAudio = null;

    function totalSlides() {{
        return slides.length;
    }}

    function updateStatus(extra) {{
        var el = document.getElementById('status');
        if (!el) return;
        var slide = slides[currentSlide];
        var topic = slide.topic ? '[' + slide.topic + '] ' : '';
        el.textContent = topic + "Slide " + (currentSlide + 1) + "/" + totalSlides() + " — Segment " + (currentSegment + 1) + "/" + slide.segments.length + (extra ? " — " + extra : "");
    }}

    function updateProgress() {{
        var totalSegs = 0;
        var currentPos = 0;
        for (var i = 0; i < slides.length; i++) {{
            if (i < currentSlide) currentPos += slides[i].segments.length;
            totalSegs += slides[i].segments.length;
        }}
        currentPos += currentSegment + 1;
        var pct = (currentPos / totalSegs) * 100;
        var fill = document.getElementById('progressFill');
        if (fill) fill.style.width = pct + '%';
    }}

    function clearSegmentTimers() {{
        segmentTimers.forEach(function(timer) {{ clearTimeout(timer); }});
        segmentTimers = [];
    }}

    function stopAudio() {{
        if (currentAudio) {{
            try {{ currentAudio.pause(); }} catch (e) {{}}
            currentAudio = null;
        }}
    }}

    function fadeOut(el, callback) {{
        if (!el) return callback();
        el.style.opacity = '0';
        setTimeout(callback, 300);
    }}

    function fadeIn(el) {{
        if (!el) return;
        setTimeout(function() {{ el.style.opacity = '1'; }}, 50);
    }}

    function displayImages(segmentId) {{
        var gallery = document.getElementById('imageGallery');
        if (!gallery) return;
        
        fadeOut(gallery, function() {{
            var imgs = imageData[segmentId] || [];
            
            if (imgs.length === 0) {{
                gallery.innerHTML = '<div style="grid-column:1/-1;text-align:center;color:#999;padding:40px;font-size:16px;">⚠️ No relevant images for this segment</div>';
            }} else {{
                gallery.innerHTML = imgs.map(function(item, idx) {{
                    var src = typeof item === 'string' ? item : item.src;
                    var title = typeof item === 'string' ? ('Image ' + (idx+1)) : (item.title || 'Image ' + (idx+1));
                    return '<div style="background:white;border-radius:8px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,0.1);animation:fadeInScale 0.4s ease forwards;animation-delay:' + (idx * 0.1) + 's;opacity:0;display:flex;flex-direction:column;"><div style="height:280px;display:flex;align-items:center;justify-content:center;"><img src="' + src + '" style="width:100%;height:100%;object-fit:contain;display:block;" alt="' + title.replace(/'/g, "\\'") + '"/></div><div style="padding:12px;text-align:center;font-size:14px;font-weight:500;color:#333;border-top:1px solid #eee;">' + title.replace(/</g, '&lt;').replace(/>/g, '&gt;') + '</div></div>';
                }}).join('');
            }}
            fadeIn(gallery);
        }});
    }}

    function displayVO(text) {{
        var el = document.getElementById('currentVO');
        if (!el) return;
        fadeOut(el, function() {{
            el.innerHTML = '<span style="color:#333;">' + text + '</span>';
            fadeIn(el);
        }});
    }}

    function playAudioForSegment(segmentId, onend) {{
        stopAudio();
        var b64 = audioData[segmentId];
        if (!b64) {{
            if (onend) onend();
            return;
        }}
        try {{
            var audio = new Audio("data:audio/mp3;base64," + b64);
            currentAudio = audio;
            audio.onended = function() {{
                currentAudio = null;
                if (onend) onend();
            }};
            audio.play();
        }} catch (err) {{
            currentAudio = null;
            if (onend) onend();
        }}
    }}

    function showSegment() {{
        var slide = slides[currentSlide];
        if (!slide || currentSegment >= slide.segments.length) return;
        
        var seg = slide.segments[currentSegment];
        updateStatus("Playing...");
        updateProgress();
        displayVO(seg.vo);
        displayImages(seg.segmentId);
    }}

    function speakSegmentSequence(slide, startIdx) {{
        if (paused || currentSlide !== slides.indexOf(slide)) return;
        if (startIdx >= slide.segments.length) {{
            // All segments done, move to next slide
            setTimeout(function() {{
                if (!paused) nextSlide();
            }}, 500);
            return;
        }}
        
        currentSegment = startIdx;
        showSegment();
        
        var seg = slide.segments[startIdx];
        
        // Play this segment's audio and wait for it to finish before moving to next
        playAudioForSegment(seg.segmentId, function() {{
            if (!paused && currentSlide === slides.indexOf(slide)) {{
                // Add a small pause between segments
                var timer = setTimeout(function() {{
                    if (!paused && currentSlide === slides.indexOf(slide)) {{
                        speakSegmentSequence(slide, startIdx + 1);
                    }}
                }}, 300);
                segmentTimers.push(timer);
            }}
        }});
    }}

    function playSlide(resumeFromSegment) {{
        if (currentSlide < 0 || currentSlide >= slides.length) return;
        if (paused) return;
        
        clearSegmentTimers();
        stopAudio();
        var slide = slides[currentSlide];
        
        if (!resumeFromSegment) {{
            currentSegment = 0;
        }}
        
        // Show first segment & start sequence
        showSegment();
        speakSegmentSequence(slide, resumeFromSegment ? currentSegment : 0);
    }}

    function nextSlide() {{
        clearSegmentTimers();
        stopAudio();
        if (currentSlide + 1 < slides.length) {{
            currentSlide++;
            currentSegment = 0;
            if (!paused) playSlide(false);
        }} else {{
            updateStatus("End of slideshow ✓");
            displayVO("🎉 Slideshow Complete!");
        }}
    }}

    function prevSlide() {{
        clearSegmentTimers();
        stopAudio();
        if (currentSlide - 1 >= 0) {{
            currentSlide--;
            currentSegment = 0;
            playSlide(false);
        }} else {{
            currentSlide = 0;
            currentSegment = 0;
            playSlide(false);
        }}
    }}

    function nextSegment() {{
        clearSegmentTimers();
        stopAudio();
        var slide = slides[currentSlide];
        if (currentSegment + 1 < slide.segments.length) {{
            currentSegment++;
            showSegment();
            // Optional: don't auto-play audio on manual segment skip, or do:
            playAudioForSegment(slide.segments[currentSegment].segmentId, null);
        }}
    }}

    function prevSegment() {{
        clearSegmentTimers();
        stopAudio();
        var slide = slides[currentSlide];
        if (currentSegment - 1 >= 0) {{
            currentSegment--;
            showSegment();
            playAudioForSegment(slide.segments[currentSegment].segmentId, null);
        }}
    }}

    // Controls
    document.getElementById('btnStart').onclick = function() {{
        paused = false;
        document.getElementById('btnResume').style.display = 'none';
        playSlide(false);
    }};
    
    document.getElementById('btnPause').onclick = function() {{
        paused = true;
        clearSegmentTimers();
        stopAudio();
        updateStatus("Paused");
        document.getElementById('btnResume').style.display = 'inline-block';
    }};
    
    document.getElementById('btnResume').onclick = function() {{
        paused = false;
        document.getElementById('btnResume').style.display = 'none';
        playSlide(true);
    }};
    
    document.getElementById('btnNext').onclick = function() {{
        paused = false;
        document.getElementById('btnResume').style.display = 'none';
        nextSlide();
    }};
    
    document.getElementById('btnPrev').onclick = function() {{
        paused = false;
        document.getElementById('btnResume').style.display = 'none';
        prevSlide();
    }};
    
    document.getElementById('btnNextSeg').onclick = function() {{
        nextSegment();
    }};
    
    document.getElementById('btnPrevSeg').onclick = function() {{
        prevSegment();
    }};

    // Add CSS animation
    var style = document.createElement('style');
    style.textContent = '@keyframes fadeInScale {{ from {{ opacity:0; transform:scale(0.95); }} to {{ opacity:1; transform:scale(1); }} }}';
    document.head.appendChild(style);

    // Initial state
    updateStatus("Ready. Click Start to begin.");
    updateProgress();
}})();
</script>
"""
    
    st.components.v1.html(html, height=720)


# =============================================================================
# MAIN UI
# =============================================================================

def main():
    st.title("🖼️ Graphics Definition V2 Slideshow Preview")
    st.markdown("View and preview images from Graphics Definition V2 output in a slideshow format.")

    if not ensure_sheet_loaded():
        return

    sheet = st.session_state["graphics_v2_sheet"]

    # Load data
    try:
        worksheet, df = get_sheet_data_and_df(sheet, "Slide Chunks")
    except Exception as e:
        st.error(f"Failed to load 'Slide Chunks' worksheet: {e}")
        return

    if df.empty:
        st.info("📭 Slide Chunks worksheet is empty.")
        return

    # Check for graphics_definition column
    gd_column = None
    for col in df.columns:
        if "graphics_definition" in col.lower() or "graphics definition" in col.lower():
            gd_column = col
            break

    if not gd_column:
        st.error("❌ No 'graphics_definition' column found. Please generate graphics definitions first.")
        st.write("Available columns:", list(df.columns))
        return

    # Filter to rows with graphics definitions
    df[gd_column] = df[gd_column].astype(str)
    preview_rows = df[df[gd_column].str.strip() != ""].copy()

    if preview_rows.empty:
        st.info("📭 No slides with graphics_definition found. Generate graphics definitions first.")
        return

    # Filters
    st.markdown("---")
    st.markdown("### 🔍 Filters")

    topics = ["All"] + sorted(set(preview_rows.get("Topic", "").astype(str).unique()))
    subtopics = ["All"] + sorted(set(preview_rows.get("Subtopic", "").astype(str).unique()))

    col1, col2, col3 = st.columns([2, 2, 3])

    with col1:
        sel_topic = st.selectbox("📚 Filter by Topic", topics, index=0)
    with col2:
        sel_subtopic = st.selectbox("📑 Filter by Subtopic", subtopics, index=0)
    with col3:
        search = st.text_input("🔎 Search slide title/content")

    # Apply filters
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
        st.warning("⚠️ No slides match current filters.")
        return

    st.success(f"✅ Found {len(filtered)} slide(s) with graphics_definition")

    # Tabs
    st.markdown("---")
    inspector_tab, slideshow_tab = st.tabs(["🔍 Inspector", "🎬 Slideshow"])

    with inspector_tab:
        render_inspector_mode(filtered)

    with slideshow_tab:
        render_slideshow_mode(filtered)


if __name__ == "__main__":
    main()
else:
    main()
