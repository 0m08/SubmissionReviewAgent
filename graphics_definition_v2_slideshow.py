# ============================================================================
# GRAPHICS DEFINITION V2 - SLIDESHOW GENERATOR
# ============================================================================
# This notebook generates video slideshows from the layout agent output.
# It reads the Google Sheet, processes each row's layout instructions,
# and creates synchronized video with narration.
# ============================================================================

# =============================================================================
# CELL 1: Install Dependencies
# =============================================================================
# Run this cell first to install required packages

"""
!pip install moviepy==1.0.3
!pip install edge-tts
!pip install yt-dlp
!pip install Pillow
!pip install gspread
!pip install pydrive2
!pip install nest_asyncio
# tqdm not needed - using MoviePy's built-in progress
!apt-get install -y ffmpeg
!apt-get install -y imagemagick

# Fix ImageMagick policy for MoviePy TextClip
!sed -i 's/rights="none" pattern="PDF"/rights="read|write" pattern="PDF"/' /etc/ImageMagick-6/policy.xml 2>/dev/null || true
!cat /etc/ImageMagick-6/policy.xml | sed 's/none/read|write/g' > /tmp/policy.xml && mv /tmp/policy.xml /etc/ImageMagick-6/policy.xml 2>/dev/null || true
"""

# =============================================================================
# CELL 2: Setup and Authentication (User's existing setup)
# =============================================================================

"""
# @title Enter the sheet link { display-mode: "form" }
sheet_link = "https://docs.google.com/spreadsheets/d/1Qw9g6pGKUvZTwsF0rsF8TBnfksoJRxmRJg24uJvTGPQ/edit?usp=sharing" # @param {type:"string"}

## Mounting google drive to read data
from google.colab import drive
drive.mount('/content/drive')

import gspread
import json

gc = gspread.service_account(filename = '/content/drive/MyDrive/Troubleshooting flowcharts/Collab files/service-credentials.json')

from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

def login_with_service_account(path):
    settings = {
                "client_config_backend": "service",
                "service_config": {
                    "client_json_file_path": path,
                }
            }
    gauth = GoogleAuth(settings = settings)
    gauth.ServiceAuth()
    return gauth

gauth = login_with_service_account("/content/drive/MyDrive/Troubleshooting flowcharts/Collab files/service-credentials.json")
drive_instance = GoogleDrive(gauth)

def load_env_variables(project_name, tracing = "true", credentials_file_id = '1rWT3pQnZlvDQETtavAIUQG1B85zRj9P0'):
    credentials_file = drive_instance.CreateFile({'id': credentials_file_id})
    credentials_file.GetContentFile('credentials.json')
    with open('credentials.json', 'r') as file:
        config = json.load(file)

load_env_variables(project_name = "graphics-definition-agent", tracing = "false", credentials_file_id = '1rWT3pQnZlvDQETtavAIUQG1B85zRj9P0')
"""

# =============================================================================
# CELL 3: Imports and Configuration
# =============================================================================

import os
import re
import json
import asyncio
import tempfile
import subprocess
import base64
from io import BytesIO
from typing import List, Dict, Tuple, Optional, Any
from urllib.parse import urlparse, parse_qs
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from PIL import Image
import numpy as np

# Note: tqdm removed - using MoviePy's built-in progress bar for encoding

# MoviePy imports
from moviepy.editor import (
    VideoFileClip, ImageClip, AudioFileClip, CompositeVideoClip,
    concatenate_videoclips, ColorClip, CompositeAudioClip, concatenate_audioclips
)
from moviepy.video.fx.all import fadein, fadeout, resize

# For async TTS
import nest_asyncio
nest_asyncio.apply()

# Configuration
CANVAS_WIDTH = 1920
CANVAS_HEIGHT = 1080
FPS = 24
TRANSITION_DURATION = 0.3  # seconds for fade transitions
BACKGROUND_COLOR = (0, 0, 0)  # Black background
OUTPUT_DIR = "/content/slideshow_output"
TEMP_DIR = "/content/temp_assets"

# Subtitle settings
ENABLE_SUBTITLES = True
SUBTITLE_HEIGHT = 80  # Height of subtitle bar in pixels
SUBTITLE_BG_COLOR = (0, 0, 0)  # Black background for subtitles
SUBTITLE_BG_OPACITY = 0.7  # Semi-transparent background
SUBTITLE_FONT_SIZE = 36
SUBTITLE_FONT_COLOR = 'white'
SUBTITLE_FONT = 'Arial'  # Will fallback to default if not available

# Performance settings
MAX_WORKERS = 8  # Parallel download threads
ENABLE_ASSET_CACHE = True  # Cache downloaded assets to avoid re-downloading

# Global asset cache (persists across rows)
ASSET_CACHE = {}  # {url: (asset_data, asset_type, file_path)}

# Create directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

print("✅ Configuration loaded")
print(f"   Canvas: {CANVAS_WIDTH}x{CANVAS_HEIGHT}")
print(f"   FPS: {FPS}")
print(f"   Subtitles: {'Enabled' if ENABLE_SUBTITLES else 'Disabled'}")
print(f"   Max parallel workers: {MAX_WORKERS}")
print(f"   Output directory: {OUTPUT_DIR}")

# =============================================================================
# CELL 4: Text-to-Speech (Narration) Functions
# =============================================================================

import edge_tts

async def generate_narration_audio_async(text: str, output_path: str, voice: str = "en-US-GuyNeural") -> str:
    """
    Generate narration audio using Edge TTS.
    
    :param text: Text to convert to speech
    :param output_path: Path to save the audio file
    :param voice: Voice to use (default: en-US-GuyNeural - male voice)
    :return: Path to the generated audio file
    """
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)
    return output_path


def generate_narration_audio(text: str, output_path: str, voice: str = "en-US-GuyNeural") -> str:
    """
    Synchronous wrapper for generating narration audio.
    """
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(generate_narration_audio_async(text, output_path, voice))


def get_audio_duration(audio_path: str) -> float:
    """
    Get the duration of an audio file in seconds.
    """
    audio_clip = AudioFileClip(audio_path)
    duration = audio_clip.duration
    audio_clip.close()
    return duration


print("✅ TTS functions loaded")

# =============================================================================
# CELL 5: Asset Loading Functions
# =============================================================================

def is_youtube_url(url: str) -> bool:
    """Check if URL is a YouTube video URL."""
    youtube_patterns = [
        r'youtube\.com/watch',
        r'youtube\.com/embed',
        r'youtu\.be/',
        r'youtube\.com/v/'
    ]
    return any(re.search(pattern, url) for pattern in youtube_patterns)


def is_drive_url(url: str) -> bool:
    """Check if URL is a Google Drive URL."""
    return 'drive.google.com' in url


def extract_drive_file_id(url: str) -> Optional[str]:
    """Extract file ID from Google Drive URL."""
    patterns = [
        r'/file/d/([a-zA-Z0-9_-]+)',
        r'id=([a-zA-Z0-9_-]+)',
        r'/d/([a-zA-Z0-9_-]+)'
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def parse_youtube_url(url: str) -> Tuple[Optional[str], Optional[int], Optional[int]]:
    """
    Parse YouTube URL to extract video ID and timestamps.
    
    :return: (video_id, start_seconds, end_seconds)
    """
    video_id = None
    start = None
    end = None
    
    # Extract video ID
    if 'youtu.be/' in url:
        video_id = url.split('youtu.be/')[-1].split('?')[0].split('/')[0]
    elif 'youtube.com' in url:
        if '/embed/' in url:
            video_id = url.split('/embed/')[-1].split('?')[0].split('/')[0]
        elif 'v=' in url:
            parsed = urlparse(url)
            params = parse_qs(parsed.query)
            video_id = params.get('v', [None])[0]
    
    # Extract timestamps
    parsed = urlparse(url)
    params = parse_qs(parsed.query)
    
    if 'start' in params:
        start = int(params['start'][0])
    elif 't' in params:
        t_val = params['t'][0]
        if t_val.endswith('s'):
            start = int(t_val[:-1])
        else:
            start = int(t_val)
    
    if 'end' in params:
        end = int(params['end'][0])
    
    return video_id, start, end


def download_youtube_clip(url: str, output_path: str) -> Optional[str]:
    """
    Download a YouTube video clip between start and end timestamps.
    Uses a two-step approach: download full video, then extract segment with ffmpeg.
    
    :param url: YouTube URL with start/end parameters
    :param output_path: Path to save the video clip
    :return: Path to the downloaded clip or None if failed
    """
    video_id, start, end = parse_youtube_url(url)
    
    if not video_id:
        print(f"⚠️ Could not extract video ID from: {url}")
        return None
    
    youtube_url = f"https://www.youtube.com/watch?v={video_id}"
    
    # First, try direct download with --download-sections (works for most videos)
    temp_full_video = os.path.join(TEMP_DIR, f"full_{video_id}.mp4")
    
    print(f"📥 Downloading YouTube clip: {video_id} ({start}s - {end}s)")
    
    expected_duration = (end - start) if (start is not None and end is not None) else None
    
    # Try Method 1: Direct section download (faster when it works)
    if start is not None and end is not None:
        cmd_direct = [
            'yt-dlp',
            '-f', 'bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best[height<=720]',
            '--no-playlist',
            '--download-sections', f'*{start}-{end}',
            '--force-keyframes-at-cuts',
            '-o', output_path,
            youtube_url
        ]
        
        try:
            result = subprocess.run(cmd_direct, capture_output=True, text=True, timeout=120)
            
            # Check if file was created
            found_path = find_downloaded_file(output_path)
            if found_path:
                # VERIFY the downloaded duration matches expected!
                actual_duration = get_video_duration(found_path)
                if actual_duration > 0 and expected_duration:
                    # Allow 20% tolerance for keyframe alignment
                    if actual_duration >= expected_duration * 0.8:
                        print(f"   ✅ Direct download successful ({actual_duration:.1f}s)")
                        return found_path
                    else:
                        print(f"   ⚠️ Direct download gave wrong duration: {actual_duration:.1f}s vs expected {expected_duration:.1f}s")
                        # Delete the bad file and try fallback
                        try:
                            os.remove(found_path)
                        except:
                            pass
                else:
                    # Can't verify, assume it's okay
                    print(f"   ✅ Direct download successful")
                    return found_path
        except Exception as e:
            print(f"   ⚠️ Direct download failed: {e}")
    
    # Method 2: Download full video then extract with ffmpeg
    print(f"   ⏳ Trying alternative download method...")
    
    cmd_full = [
        'yt-dlp',
        '-f', 'bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best[height<=720]',
        '--no-playlist',
        '--merge-output-format', 'mp4',
        '-o', temp_full_video,
        youtube_url
    ]
    
    try:
        result = subprocess.run(cmd_full, capture_output=True, text=True, timeout=300)
        
        # Find the downloaded file
        full_video_path = find_downloaded_file(temp_full_video)
        
        if not full_video_path:
            print(f"⚠️ Failed to download full video")
            print(f"   stderr: {result.stderr[-300:] if result.stderr else 'None'}")
            return None
        
        # Now extract the segment using ffmpeg
        if start is not None and end is not None:
            duration = end - start
            ffmpeg_cmd = [
                'ffmpeg', '-y',
                '-ss', str(start),
                '-i', full_video_path,
                '-t', str(duration),
                '-c:v', 'libx264',
                '-c:a', 'aac',
                '-avoid_negative_ts', 'make_zero',
                output_path
            ]
        elif start is not None:
            ffmpeg_cmd = [
                'ffmpeg', '-y',
                '-ss', str(start),
                '-i', full_video_path,
                '-c:v', 'libx264',
                '-c:a', 'aac',
                '-avoid_negative_ts', 'make_zero',
                output_path
            ]
        else:
            # No timestamps, just copy the file
            import shutil
            shutil.copy(full_video_path, output_path)
            return output_path
        
        ffmpeg_result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, timeout=120)
        
        if os.path.exists(output_path):
            # Cleanup full video
            try:
                os.remove(full_video_path)
            except:
                pass
            print(f"   ✅ Extracted clip successfully")
            return output_path
        else:
            print(f"⚠️ FFmpeg extraction failed")
            print(f"   stderr: {ffmpeg_result.stderr[-300:] if ffmpeg_result.stderr else 'None'}")
            return None
        
    except subprocess.TimeoutExpired:
        print(f"⚠️ Download timed out for: {url}")
        return None
    except Exception as e:
        print(f"⚠️ Error downloading YouTube clip: {e}")
        return None


def find_downloaded_file(base_path: str) -> Optional[str]:
    """
    Find a downloaded file that may have a different extension than expected.
    yt-dlp sometimes adds extensions or changes them.
    """
    if os.path.exists(base_path):
        return base_path
    
    # Try common extensions
    for ext in ['.mp4', '.webm', '.mkv', '.m4a', '']:
        # Check with extension added
        potential_path = base_path + ext
        if os.path.exists(potential_path):
            return potential_path
        
        # Check with extension replaced
        if '.' in base_path:
            base_no_ext = base_path.rsplit('.', 1)[0]
            potential_path = base_no_ext + ext
            if os.path.exists(potential_path):
                return potential_path
    
    # Check directory for any file matching the base name
    directory = os.path.dirname(base_path)
    filename_base = os.path.basename(base_path).rsplit('.', 1)[0]
    
    if os.path.exists(directory):
        for f in os.listdir(directory):
            if f.startswith(filename_base):
                return os.path.join(directory, f)
    
    return None


def download_drive_image(url: str, drive_instance) -> Optional[Image.Image]:
    """
    Download image from Google Drive.
    
    :param url: Google Drive URL
    :param drive_instance: PyDrive GoogleDrive instance
    :return: PIL Image or None if failed
    """
    file_id = extract_drive_file_id(url)
    if not file_id:
        print(f"⚠️ Could not extract file ID from: {url}")
        return None
    
    try:
        file = drive_instance.CreateFile({'id': file_id})
        
        # Get file metadata
        file.FetchMetadata(fields='title,mimeType')
        
        # Download to temp file
        temp_path = os.path.join(TEMP_DIR, f"drive_{file_id}")
        file.GetContentFile(temp_path)
        
        # Open as PIL Image
        img = Image.open(temp_path)
        img = img.convert('RGB')  # Ensure RGB format
        
        print(f"✅ Loaded Drive image: {file['title']}")
        return img
        
    except Exception as e:
        print(f"⚠️ Error downloading Drive image: {e}")
        return None


def download_web_image(url: str) -> Optional[Image.Image]:
    """
    Download image from web URL.
    
    :param url: Web URL to image
    :return: PIL Image or None if failed
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        img = Image.open(BytesIO(response.content))
        img = img.convert('RGB')  # Ensure RGB format
        
        print(f"✅ Loaded web image: {url[:60]}...")
        return img
        
    except Exception as e:
        print(f"⚠️ Error downloading web image {url[:50]}...: {e}")
        return None


def load_asset(url: str, drive_instance, asset_index: int) -> Tuple[Optional[Any], str]:
    """
    Load an asset (image or video) from URL with caching.
    Also stores actual video duration in cache for proper timeline calculation.
    
    :param url: Asset URL (web, Drive, or YouTube)
    :param drive_instance: PyDrive GoogleDrive instance
    :param asset_index: Index for unique temp file naming
    :return: (asset, asset_type) where asset is PIL Image or video path, type is 'image' or 'video'
    """
    global ASSET_CACHE
    
    # Check cache first
    if ENABLE_ASSET_CACHE and url in ASSET_CACHE:
        cached = ASSET_CACHE[url]
        # For images, return the cached PIL image
        # For videos, return the cached file path
        if cached['type'] == 'video':
            if cached['path'] and os.path.exists(cached['path']):
                return cached['path'], 'video'
        else:
            if cached['data'] is not None:
                return cached['data'], 'image'
    
    # Not cached, download
    if is_youtube_url(url):
        # Download YouTube clip
        output_path = os.path.join(TEMP_DIR, f"video_{asset_index}.mp4")
        video_path = download_youtube_clip(url, output_path)
        if ENABLE_ASSET_CACHE and video_path:
            # Get actual video duration for proper timeline calculation
            video_duration = get_video_duration(video_path) if video_path else 0.0
            ASSET_CACHE[url] = {'type': 'video', 'path': video_path, 'data': None, 'duration': video_duration}
        return video_path, 'video'
    
    elif is_drive_url(url):
        # Download from Google Drive
        img = download_drive_image(url, drive_instance)
        if ENABLE_ASSET_CACHE and img:
            ASSET_CACHE[url] = {'type': 'image', 'path': None, 'data': img, 'duration': 0.0}
        return img, 'image'
    
    else:
        # Assume web image
        img = download_web_image(url)
        if ENABLE_ASSET_CACHE and img:
            ASSET_CACHE[url] = {'type': 'image', 'path': None, 'data': img, 'duration': 0.0}
        return img, 'image'


def extract_all_asset_urls(layout_json: Dict) -> List[str]:
    """Extract all unique asset URLs from layout JSON."""
    urls = set()
    for segment in layout_json.get("segments", []):
        for part in segment.get("narration_parts", []):
            for action in part.get("actions", []):
                url = action.get("asset", "")
                if url:
                    urls.add(url)
    return list(urls)


def get_video_duration(video_path: str) -> float:
    """Get the actual duration of a video file in seconds."""
    try:
        clip = VideoFileClip(video_path)
        duration = clip.duration
        clip.close()
        return duration
    except Exception as e:
        print(f"   ⚠️ Could not get video duration: {e}")
        return 0.0


def preload_assets_parallel(urls: List[str], drive_instance, max_workers: int = MAX_WORKERS):
    """
    Pre-download all assets in parallel for faster processing.
    Also stores actual video durations in the cache.
    
    :param urls: List of asset URLs to download
    :param drive_instance: PyDrive GoogleDrive instance
    :param max_workers: Number of parallel download threads
    """
    global ASSET_CACHE
    
    # Filter out already cached URLs
    urls_to_download = [url for url in urls if url not in ASSET_CACHE]
    
    if not urls_to_download:
        print(f"   ✅ All {len(urls)} assets already cached")
        return
    
    def download_single(args):
        url, index = args
        try:
            asset, asset_type = load_asset(url, drive_instance, index)
            return url, asset, asset_type
        except Exception as e:
            return url, None, None
    
    # Download in parallel
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        args_list = [(url, i) for i, url in enumerate(urls_to_download)]
        futures = {executor.submit(download_single, args): args for args in args_list}
        
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
    
    # After all downloads complete, get video durations and update cache
    # (load_asset already stored in cache, but we need to ensure duration is set)
    video_count = 0
    for url, asset, asset_type in results:
        if asset is not None and asset_type == 'video':
            # Get actual video duration
            if os.path.exists(asset):
                duration = get_video_duration(asset)
                # Update cache with duration
                ASSET_CACHE[url] = {'type': 'video', 'path': asset, 'data': None, 'duration': duration}
                if duration > 0:
                    print(f"   🎬 Video duration: {duration:.1f}s - ...{url[-40:]}")
                    video_count += 1
                else:
                    print(f"   ⚠️ Video has 0 duration: ...{url[-40:]}")
    
    # Count successes
    success_count = sum(1 for _, asset, _ in results if asset is not None)
    print(f"   ✅ Pre-downloaded {success_count} assets ({video_count} videos with duration)")


def generate_all_audio_parallel(timeline: List[Dict], row_index: int, voice: str, max_workers: int = MAX_WORKERS) -> Dict[str, str]:
    """
    Generate all narration audio files in parallel.
    
    :param timeline: List of timeline entries with voiceover text
    :param row_index: Row index for file naming
    :param voice: TTS voice to use
    :param max_workers: Number of parallel threads
    :return: Dict mapping voiceover text to audio file path
    """
    audio_paths = {}
    
    def generate_single(args):
        segment_id, part_idx, voiceover_text = args
        audio_path = os.path.join(TEMP_DIR, f"audio_{row_index}_{segment_id}_{part_idx}.mp3")
        try:
            generate_narration_audio(voiceover_text, audio_path, voice)
            return voiceover_text, audio_path
        except Exception as e:
            return voiceover_text, None
    
    # Prepare args
    args_list = [(entry['segment_id'], entry['part_idx'], entry['voiceover']) 
                 for entry in timeline if entry.get('voiceover')]
    
    if not args_list:
        return audio_paths
    
    # Generate in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(generate_single, args): args for args in args_list}
        
        for future in as_completed(futures):
            text, path = future.result()
            if path:
                audio_paths[text] = path
    
    return audio_paths


print("✅ Asset loading functions loaded (with caching & parallel download)")

# =============================================================================
# CELL 6: Video Composition Functions
# =============================================================================

def position_clip_on_canvas(clip, position: str, canvas_size: Tuple[int, int] = (CANVAS_WIDTH, CANVAS_HEIGHT)):
    """
    Position a clip on the canvas according to the position parameter.
    
    :param clip: MoviePy clip (ImageClip or VideoFileClip)
    :param position: Position string ('left', 'right', 'center', 'full')
    :param canvas_size: Canvas dimensions (width, height)
    :return: Positioned clip
    """
    canvas_w, canvas_h = canvas_size
    clip_w, clip_h = clip.size
    
    if position == 'full':
        # Scale to FIT entire image within canvas (no cropping, black bars if needed)
        scale_w = canvas_w / clip_w
        scale_h = canvas_h / clip_h
        scale = min(scale_w, scale_h)  # Use min to fit entire image (no cropping)
        
        new_w = int(clip_w * scale)
        new_h = int(clip_h * scale)
        
        clip = clip.resize((new_w, new_h))
        
        # Center the clip (black bars on sides if aspect ratio differs)
        x = (canvas_w - new_w) // 2
        y = (canvas_h - new_h) // 2
        
        return clip.set_position((x, y))
    
    elif position == 'center':
        # Scale to fit within center area (80% of canvas)
        max_w = int(canvas_w * 0.8)
        max_h = int(canvas_h * 0.8)
        
        scale_w = max_w / clip_w
        scale_h = max_h / clip_h
        scale = min(scale_w, scale_h)  # Use min to fit
        
        new_w = int(clip_w * scale)
        new_h = int(clip_h * scale)
        
        clip = clip.resize((new_w, new_h))
        
        # Center on canvas
        x = (canvas_w - new_w) // 2
        y = (canvas_h - new_h) // 2
        
        return clip.set_position((x, y))
    
    elif position == 'left':
        # Scale to fit left half
        max_w = int(canvas_w * 0.45)
        max_h = int(canvas_h * 0.8)
        
        scale_w = max_w / clip_w
        scale_h = max_h / clip_h
        scale = min(scale_w, scale_h)
        
        new_w = int(clip_w * scale)
        new_h = int(clip_h * scale)
        
        clip = clip.resize((new_w, new_h))
        
        # Position on left side
        x = int(canvas_w * 0.05)
        y = (canvas_h - new_h) // 2
        
        return clip.set_position((x, y))
    
    elif position == 'right':
        # Scale to fit right half
        max_w = int(canvas_w * 0.45)
        max_h = int(canvas_h * 0.8)
        
        scale_w = max_w / clip_w
        scale_h = max_h / clip_h
        scale = min(scale_w, scale_h)
        
        new_w = int(clip_w * scale)
        new_h = int(clip_h * scale)
        
        clip = clip.resize((new_w, new_h))
        
        # Position on right side
        x = int(canvas_w * 0.5)
        y = (canvas_h - new_h) // 2
        
        return clip.set_position((x, y))
    
    else:
        # Default to center
        return position_clip_on_canvas(clip, 'center', canvas_size)


def apply_transition(clip, transition: str, duration: float = TRANSITION_DURATION):
    """
    Apply transition effect to a clip.
    
    :param clip: MoviePy clip
    :param transition: Transition type ('fade_in', 'fade_out', 'none')
    :param duration: Transition duration in seconds
    :return: Clip with transition applied
    """
    if transition == 'fade_in':
        return clip.crossfadein(duration)
    elif transition == 'fade_out':
        return clip.crossfadeout(duration)
    else:
        return clip


def create_subtitle_clip(text: str, duration: float, start_time: float) -> Optional[ImageClip]:
    """
    Create a subtitle clip with semi-transparent background at the bottom of the canvas.
    Uses PIL for text rendering (no ImageMagick dependency).
    
    :param text: Subtitle text to display
    :param duration: Duration to show the subtitle
    :param start_time: When the subtitle should appear
    :return: ImageClip with subtitle or None if subtitles disabled
    """
    if not ENABLE_SUBTITLES or not text:
        return None
    
    try:
        from PIL import Image, ImageDraw, ImageFont
        
        # Create subtitle bar image with transparency
        bar_height = SUBTITLE_HEIGHT
        subtitle_img = Image.new('RGBA', (CANVAS_WIDTH, bar_height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(subtitle_img)
        
        # Draw semi-transparent background
        bg_alpha = int(255 * SUBTITLE_BG_OPACITY)
        bg_color = (*SUBTITLE_BG_COLOR, bg_alpha)
        draw.rectangle([(0, 0), (CANVAS_WIDTH, bar_height)], fill=bg_color)
        
        # Try to load a font, fallback to default
        try:
            # Try common fonts available in Colab/Linux
            font_paths = [
                '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
                '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
                '/usr/share/fonts/truetype/freefont/FreeSans.ttf',
            ]
            font = None
            for font_path in font_paths:
                if os.path.exists(font_path):
                    font = ImageFont.truetype(font_path, SUBTITLE_FONT_SIZE)
                    break
            if font is None:
                font = ImageFont.load_default()
        except:
            font = ImageFont.load_default()
        
        # Wrap text if too long
        max_width = CANVAS_WIDTH - 100  # Padding
        words = text.split()
        lines = []
        current_line = ""
        
        for word in words:
            test_line = current_line + " " + word if current_line else word
            # Get text width
            try:
                bbox = draw.textbbox((0, 0), test_line, font=font)
                text_width = bbox[2] - bbox[0]
            except:
                text_width = len(test_line) * (SUBTITLE_FONT_SIZE * 0.6)  # Rough estimate
            
            if text_width <= max_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        
        if current_line:
            lines.append(current_line)
        
        # Join lines (max 2 lines for subtitles)
        if len(lines) > 2:
            lines = [' '.join(lines[:len(lines)//2]), ' '.join(lines[len(lines)//2:])]
        
        display_text = '\n'.join(lines)
        
        # Calculate text position (centered)
        try:
            bbox = draw.textbbox((0, 0), display_text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        except:
            text_width = len(display_text) * (SUBTITLE_FONT_SIZE * 0.6)
            text_height = SUBTITLE_FONT_SIZE * len(lines) * 1.2
        
        x = (CANVAS_WIDTH - text_width) // 2
        y = (bar_height - text_height) // 2
        
        # Draw text
        draw.text((x, y), display_text, fill=(255, 255, 255, 255), font=font)
        
        # Convert to numpy array for MoviePy
        subtitle_array = np.array(subtitle_img)
        
        # Create ImageClip
        subtitle_clip = ImageClip(subtitle_array, ismask=False)
        subtitle_clip = subtitle_clip.set_duration(duration)
        subtitle_clip = subtitle_clip.set_start(start_time)
        
        # Position at bottom of canvas
        subtitle_clip = subtitle_clip.set_position(('center', CANVAS_HEIGHT - bar_height))
        
        # Add fade in/out
        subtitle_clip = subtitle_clip.crossfadein(0.2).crossfadeout(0.2)
        
        return subtitle_clip
        
    except Exception as e:
        print(f"   ⚠️ Could not create subtitle: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_image_clip(image: Image.Image, duration: float) -> ImageClip:
    """
    Create a MoviePy ImageClip from a PIL Image.
    
    :param image: PIL Image
    :param duration: Duration in seconds
    :return: ImageClip
    """
    # Convert PIL Image to numpy array
    img_array = np.array(image)
    
    # Create ImageClip
    clip = ImageClip(img_array).set_duration(duration)
    
    return clip


def create_video_clip(video_path: str, target_duration: Optional[float] = None) -> Optional[VideoFileClip]:
    """
    Create a MoviePy VideoFileClip from a video file.
    
    :param video_path: Path to video file
    :param target_duration: Optional target duration (will loop or cut if needed)
    :return: VideoFileClip or None if failed
    """
    try:
        clip = VideoFileClip(video_path)
        
        if target_duration:
            if clip.duration < target_duration:
                # Loop the clip if it's shorter than target
                # For simplicity, we'll just extend with last frame
                pass  # Keep original duration
            elif clip.duration > target_duration:
                # Cut the clip if it's longer
                clip = clip.subclip(0, target_duration)
        
        return clip
        
    except Exception as e:
        print(f"⚠️ Error creating video clip from {video_path}: {e}")
        return None


print("✅ Video composition functions loaded")

# =============================================================================
# CELL 7: Main Slideshow Generation Function
# =============================================================================

def generate_slideshow_for_row(
    layout_json: Dict,
    row_index: int,
    slide_title: str,
    drive_instance,
    voice: str = "en-US-GuyNeural"
) -> Optional[str]:
    """
    Generate a slideshow video for a single row based on layout instructions.
    
    Properly handles canvas state across narration parts:
    - 'add': Asset appears on canvas (starts lifecycle)
    - 'keep': Asset persists from previous part (continues lifecycle)
    - 'remove': Asset disappears from canvas (ends lifecycle)
    - 'move': Asset changes position (ends old lifecycle, starts new one)
    
    :param layout_json: Parsed JSON layout instructions
    :param row_index: Row index for output naming
    :param slide_title: Slide title for output naming
    :param drive_instance: PyDrive GoogleDrive instance
    :param voice: TTS voice to use
    :return: Path to generated video or None if failed
    """
    import time
    start_time_total = time.time()
    
    print(f"\n{'='*60}")
    print(f"📽️ Generating slideshow for: {slide_title}")
    print(f"{'='*60}")
    
    segments = layout_json.get("segments", [])
    
    # =========================================================================
    # PHASE 0: Pre-download all assets in parallel (FAST!)
    # =========================================================================
    
    print("\n📐 Phase 0: Pre-downloading assets...")
    phase0_start = time.time()
    
    asset_urls = extract_all_asset_urls(layout_json)
    print(f"   Found {len(asset_urls)} unique assets")
    preload_assets_parallel(asset_urls, drive_instance)
    
    print(f"   ⏱️ Phase 0 completed in {time.time() - phase0_start:.1f}s")
    
    # =========================================================================
    # PHASE 1: Build timeline structure (no audio yet - just structure)
    # =========================================================================
    
    print("\n📐 Phase 1: Building timeline structure...")
    phase1_start = time.time()
    
    # First pass: collect all voiceover texts for parallel TTS
    timeline_structure = []
    for segment in segments:
        segment_id = segment.get("segment_id", "?")
        narration_parts = segment.get("narration_parts", [])
        
        for part_idx, narration_part in enumerate(narration_parts):
            voiceover_text = narration_part.get("voiceover_part", "")
            actions = narration_part.get("actions", [])
            
            if not voiceover_text:
                continue
            
            timeline_structure.append({
                'segment_id': segment_id,
                'part_idx': part_idx,
                'voiceover': voiceover_text,
                'actions': actions,
            })
    
    # Generate all audio in parallel
    audio_paths = generate_all_audio_parallel(timeline_structure, row_index, voice)
    
    # Now build the actual timeline with durations
    timeline = []
    all_audio_clips = []
    current_time = 0.0
    
    for entry in timeline_structure:
        voiceover_text = entry['voiceover']
        actions = entry['actions']
        segment_id = entry['segment_id']
        part_idx = entry['part_idx']
        
        # Get audio duration
        audio_path = audio_paths.get(voiceover_text)
        if audio_path and os.path.exists(audio_path):
            narration_duration = get_audio_duration(audio_path)
            audio_clip = AudioFileClip(audio_path).set_start(current_time)
            all_audio_clips.append(audio_clip)
        else:
            narration_duration = 3.0
        
        # Check for video clips that might extend duration - use ACTUAL video duration from cache
        max_video_duration = 0.0
        for action in actions:
            if action.get("action") == "add":
                asset_url = action.get("asset", "")
                
                # Debug: check what's in cache for this URL
                if asset_url in ASSET_CACHE:
                    cached = ASSET_CACHE[asset_url]
                    cached_type = cached.get('type', 'unknown')
                    cached_duration = cached.get('duration', 0.0)
                    
                    if cached_type == 'video' and cached_duration > 0:
                        max_video_duration = max(max_video_duration, cached_duration)
                        print(f"      🎬 Using cached video duration: {cached_duration:.2f}s")
                        continue
                    elif cached_type == 'video':
                        print(f"      ⚠️ Video in cache but duration=0")
                
                # Fallback: estimate from URL timestamps
                if is_youtube_url(asset_url):
                    video_id, start, end = parse_youtube_url(asset_url)
                    if start is not None and end is not None:
                        estimated = end - start
                        max_video_duration = max(max_video_duration, estimated)
                        print(f"      🎬 Using URL timestamp estimate: {estimated:.2f}s")
        
        part_duration = max(narration_duration, max_video_duration)
        print(f"      📊 Part {segment_id}.{part_idx}: narration={narration_duration:.2f}s, video={max_video_duration:.2f}s → duration={part_duration:.2f}s")
        
        timeline.append({
            'segment_id': segment_id,
            'part_idx': part_idx,
            'voiceover': voiceover_text,
            'actions': actions,
            'start_time': current_time,
            'duration': part_duration,
            'end_time': current_time + part_duration
        })
        
        current_time += part_duration
    
    total_duration = current_time
    print(f"   📊 Total timeline: {total_duration:.2f}s across {len(timeline)} parts")
    print(f"   ⏱️ Phase 1 completed in {time.time() - phase1_start:.1f}s")
    
    # =========================================================================
    # PHASE 2: Track asset lifecycles (when each asset appears/disappears)
    # =========================================================================
    
    print("\n📐 Phase 2: Tracking asset lifecycles...")
    phase2_start = time.time()
    
    # asset_lifecycles: {url: [{start, end, position, transition_in, transition_out}, ...]}
    asset_lifecycles = {}
    
    # Track current canvas state: {url: {position, start_time, transition_in}}
    current_canvas = {}
    
    for part in timeline:
        part_start = part['start_time']
        part_end = part['end_time']
        actions = part['actions']
        
        for action in actions:
            asset_url = action.get("asset", "")
            action_type = action.get("action", "")
            position = action.get("position", "center")
            transition = action.get("transition", "none")
            
            if action_type == "add":
                # Asset starts appearing
                current_canvas[asset_url] = {
                    'position': position,
                    'start_time': part_start,
                    'transition_in': transition
                }
            
            elif action_type == "keep":
                # Asset continues - no change needed
                if asset_url not in current_canvas:
                    # Asset should be on canvas but isn't - recover by treating as add
                    current_canvas[asset_url] = {
                        'position': position if position else 'center',
                        'start_time': part_start,
                        'transition_in': 'none'
                    }
                # else: asset continues, nothing to do
            
            elif action_type == "move":
                # Asset moves - finalize old lifecycle, start new one
                if asset_url in current_canvas:
                    old_info = current_canvas[asset_url]
                    lifecycle = {
                        'start': old_info['start_time'],
                        'end': part_start,
                        'position': old_info['position'],
                        'transition_in': old_info['transition_in'],
                        'transition_out': 'none'
                    }
                    if asset_url not in asset_lifecycles:
                        asset_lifecycles[asset_url] = []
                    asset_lifecycles[asset_url].append(lifecycle)
                
                # Start new lifecycle at new position
                current_canvas[asset_url] = {
                    'position': position,
                    'start_time': part_start,
                    'transition_in': transition
                }
            
            elif action_type == "remove":
                # Asset stops appearing at the START of this part (not the end)
                # This ensures the old image is gone before the new one appears
                if asset_url in current_canvas:
                    old_info = current_canvas[asset_url]
                    lifecycle = {
                        'start': old_info['start_time'],
                        'end': part_start,  # Ends when this part STARTS (so new image can take over)
                        'position': old_info['position'],
                        'transition_in': old_info['transition_in'],
                        'transition_out': transition
                    }
                    if asset_url not in asset_lifecycles:
                        asset_lifecycles[asset_url] = []
                    asset_lifecycles[asset_url].append(lifecycle)
                    del current_canvas[asset_url]
    
    # Finalize any assets still on canvas at the end
    for asset_url, info in current_canvas.items():
        lifecycle = {
            'start': info['start_time'],
            'end': total_duration,
            'position': info['position'],
            'transition_in': info['transition_in'],
            'transition_out': 'fade_out'
        }
        if asset_url not in asset_lifecycles:
            asset_lifecycles[asset_url] = []
        asset_lifecycles[asset_url].append(lifecycle)
    
    total_lifecycles = sum(len(lc) for lc in asset_lifecycles.values())
    print(f"   📊 Tracked {len(asset_lifecycles)} assets with {total_lifecycles} lifecycles")
    print(f"   ⏱️ Phase 2 completed in {time.time() - phase2_start:.1f}s")
    
    # =========================================================================
    # PHASE 3: Create video clips from cached assets
    # =========================================================================
    
    print("\n📐 Phase 3: Creating video clips from cache...")
    phase3_start = time.time()
    
    all_clips = []
    clip_count = 0
    
    for asset_url, lifecycles in asset_lifecycles.items():
        # Get asset from cache (should already be loaded in Phase 0)
        if asset_url in ASSET_CACHE:
            cached = ASSET_CACHE[asset_url]
            if cached['type'] == 'video':
                asset = cached['path']
                asset_type = 'video'
            else:
                asset = cached['data']
                asset_type = 'image'
        else:
            # Fallback: load if not in cache
            asset, asset_type = load_asset(asset_url, drive_instance, clip_count)
            if asset is None:
                print(f"   ⚠️ Asset not in cache and failed to load: ...{asset_url[-40:]}")
                continue
        
        for lifecycle in lifecycles:
            start = lifecycle['start']
            end = lifecycle['end']
            duration = end - start
            position = lifecycle['position']
            transition_in = lifecycle['transition_in']
            transition_out = lifecycle['transition_out']
            
            if duration <= 0:
                continue
            
            if asset_type == 'video':
                clip = create_video_clip(asset)
                if clip:
                    print(f"      🎬 Video clip: actual={clip.duration:.2f}s, lifecycle={duration:.2f}s")
                    # Handle video duration vs lifecycle duration
                    if clip.duration < duration:
                        # Video shorter than lifecycle - extend with last frame
                        print(f"         → Extending with last frame (+{duration - clip.duration:.2f}s)")
                        try:
                            last_frame = clip.to_ImageClip(t=clip.duration - 0.1)
                            last_frame = last_frame.set_duration(duration - clip.duration)
                            clip = concatenate_videoclips([clip, last_frame])
                        except:
                            clip = clip.set_duration(duration)
                    elif clip.duration > duration:
                        # Video longer than lifecycle - this should NOT happen if Phase 1 calculated correctly
                        print(f"         ⚠️ Video LONGER than lifecycle! Cutting to {duration:.2f}s (losing {clip.duration - duration:.2f}s)")
                        clip = clip.subclip(0, duration)
                    else:
                        # Video matches lifecycle - perfect!
                        print(f"         → Video duration matches lifecycle ✓")
            else:
                # Image asset
                if isinstance(asset, str):
                    img = Image.open(asset).convert('RGB')
                    clip = create_image_clip(img, duration)
                else:
                    clip = create_image_clip(asset.copy() if hasattr(asset, 'copy') else asset, duration)
            
            if clip is None:
                continue
            
            # Position on canvas
            clip = position_clip_on_canvas(clip, position)
            
            # Apply transitions
            if transition_in == 'fade_in':
                clip = clip.crossfadein(TRANSITION_DURATION)
            if transition_out == 'fade_out':
                clip = clip.crossfadeout(TRANSITION_DURATION)
            
            # Set start time
            clip = clip.set_start(start)
            
            all_clips.append(clip)
            clip_count += 1
    
    print(f"   📊 Created {clip_count} video clips")
    print(f"   ⏱️ Phase 3 completed in {time.time() - phase3_start:.1f}s")
    
    if not all_clips:
        print("⚠️ No clips generated")
        return None
    
    # =========================================================================
    # PHASE 3.5: Create subtitle clips for each narration part
    # =========================================================================
    
    subtitle_clips = []
    if ENABLE_SUBTITLES:
        parts_with_vo = [p for p in timeline if p.get('voiceover') and p.get('duration', 0) > 0]
        
        for part in parts_with_vo:
            voiceover = part.get('voiceover', '')
            start_time = part.get('start_time', 0)
            duration = part.get('duration', 0)
            
            subtitle = create_subtitle_clip(voiceover, duration, start_time)
            if subtitle:
                subtitle_clips.append(subtitle)
    
    # =========================================================================
    # PHASE 4: Composite and export
    # =========================================================================
    
    print(f"\n📐 Phase 4: Compositing & exporting...")
    phase4_start = time.time()
    
    # Create background
    background = ColorClip(size=(CANVAS_WIDTH, CANVAS_HEIGHT), color=BACKGROUND_COLOR)
    background = background.set_duration(total_duration)
    
    # Composite all clips (background + visuals + subtitles on top)
    all_layers = [background] + all_clips + subtitle_clips
    print(f"   🎬 Compositing {len(all_clips)} visual clips + {len(subtitle_clips)} subtitles + {len(all_audio_clips)} audio...")
    final_video = CompositeVideoClip(all_layers, size=(CANVAS_WIDTH, CANVAS_HEIGHT))
    
    # Add audio
    if all_audio_clips:
        final_audio = CompositeAudioClip(all_audio_clips)
        final_video = final_video.set_audio(final_audio)
    
    # Generate output filename
    safe_title = re.sub(r'[^\w\s-]', '', slide_title)[:50].strip()
    output_path = os.path.join(OUTPUT_DIR, f"row_{row_index}_{safe_title}.mp4")
    
    # Write video
    print(f"   💾 Encoding video ({total_duration:.1f}s @ {FPS}fps)...")
    
    final_video.write_videofile(
        output_path,
        fps=FPS,
        codec='libx264',
        audio_codec='aac',
        preset='fast',  # Faster encoding
        temp_audiofile=os.path.join(TEMP_DIR, f"temp_audio_{row_index}.m4a"),
        remove_temp=True,
        verbose=False,
        logger='bar'  # MoviePy's built-in progress bar
    )
    
    # Cleanup
    final_video.close()
    for clip in all_clips:
        try:
            clip.close()
        except:
            pass
    
    print(f"   ⏱️ Phase 4 completed in {time.time() - phase4_start:.1f}s")
    
    total_time = time.time() - start_time_total
    print(f"\n✅ Video saved: {output_path}")
    print(f"🎉 Total processing time: {total_time:.1f}s ({total_time/60:.1f} min)")
    return output_path


print("✅ Main slideshow generation function loaded")

# =============================================================================
# CELL 8: Sheet Reading and Processing Functions
# =============================================================================

def get_sheet_data(gc, sheet_link: str, worksheet_name: str = "Slide Chunks"):
    """
    Read data from Google Sheet.
    
    :param gc: gspread client
    :param sheet_link: Google Sheet URL
    :param worksheet_name: Name of the worksheet to read
    :return: List of row dictionaries
    """
    # Extract sheet ID from link
    match = re.search(r'/d/([a-zA-Z0-9_-]+)', sheet_link)
    if not match:
        raise ValueError(f"Invalid sheet link: {sheet_link}")
    
    sheet_id = match.group(1)
    
    # Open sheet
    sheet = gc.open_by_key(sheet_id)
    worksheet = sheet.worksheet(worksheet_name)
    
    # Get all records
    records = worksheet.get_all_records()
    
    return records


def concatenate_videos(video_paths: List[str], output_path: str, add_transitions: bool = True) -> Optional[str]:
    """
    Concatenate multiple videos into a single video.
    
    :param video_paths: List of video file paths to concatenate
    :param output_path: Path to save the combined video
    :param add_transitions: Whether to add fade transitions between slides
    :return: Path to the combined video or None if failed
    """
    if not video_paths:
        print("⚠️ No videos to concatenate")
        return None
    
    if len(video_paths) == 1:
        # Only one video, just copy it
        import shutil
        shutil.copy(video_paths[0], output_path)
        return output_path
    
    print(f"\n🎬 Concatenating {len(video_paths)} videos...")
    
    clips = []
    for i, video_path in enumerate(video_paths):
        try:
            clip = VideoFileClip(video_path)
            
            if add_transitions and i > 0:
                # Add fade in for all clips except the first
                clip = clip.crossfadein(0.5)
            
            if add_transitions and i < len(video_paths) - 1:
                # Add fade out for all clips except the last
                clip = clip.crossfadeout(0.5)
            
            clips.append(clip)
            print(f"   ✅ Added: {os.path.basename(video_path)} ({clip.duration:.1f}s)")
        except Exception as e:
            print(f"   ⚠️ Error loading {video_path}: {e}")
    
    if not clips:
        print("⚠️ No clips could be loaded")
        return None
    
    # Concatenate all clips
    if add_transitions:
        # Use crossfade method for smoother transitions
        final_clip = concatenate_videoclips(clips, method="compose", padding=-0.5)
    else:
        final_clip = concatenate_videoclips(clips, method="compose")
    
    print(f"   📊 Total duration: {final_clip.duration:.1f}s")
    
    # Write the final video
    print(f"   💾 Writing combined video...")
    final_clip.write_videofile(
        output_path,
        fps=FPS,
        codec='libx264',
        audio_codec='aac',
        temp_audiofile=os.path.join(TEMP_DIR, "temp_concat_audio.m4a"),
        remove_temp=True,
        verbose=False,
        logger=None
    )
    
    # Cleanup
    final_clip.close()
    for clip in clips:
        try:
            clip.close()
        except:
            pass
    
    print(f"   ✅ Combined video saved: {output_path}")
    return output_path


def process_sheet(gc, sheet_link: str, drive_instance, worksheet_name: str = "Slide Chunks", 
                  row_indices: Optional[List[int]] = None, voice: str = "en-US-GuyNeural",
                  combine_videos: bool = True, output_filename: str = "course_slideshow.mp4"):
    """
    Process all rows (or specified rows) from the sheet and generate slideshows.
    
    :param gc: gspread client
    :param sheet_link: Google Sheet URL
    :param drive_instance: PyDrive GoogleDrive instance
    :param worksheet_name: Name of the worksheet
    :param row_indices: Optional list of row indices to process (0-based). If None, process all.
    :param voice: TTS voice to use
    :param combine_videos: Whether to combine all slide videos into one (default: True)
    :param output_filename: Filename for the combined video
    :return: Path to combined video (if combine_videos=True) or list of individual video paths
    """
    print("\n" + "="*60)
    print("📊 READING SHEET DATA")
    print("="*60)
    
    records = get_sheet_data(gc, sheet_link, worksheet_name)
    print(f"📋 Found {len(records)} rows in sheet")
    
    generated_videos = []
    
    # Determine which rows to process
    if row_indices is not None:
        rows_to_process = [(i, records[i]) for i in row_indices if i < len(records)]
    else:
        rows_to_process = list(enumerate(records))
    
    print(f"🎯 Processing {len(rows_to_process)} row(s)\n")
    
    for slide_num, (idx, row) in enumerate(rows_to_process, 1):
        slide_title = row.get("Slide Chunk Title", f"Row {idx}")
        layout_instructions = row.get("layout_instructions", "")
        
        # Show overall progress
        print(f"\n{'='*60}")
        print(f"📊 SLIDE {slide_num}/{len(rows_to_process)}: {slide_title[:50]}")
        print(f"{'='*60}")
        
        if not layout_instructions or layout_instructions.strip() == "":
            print(f"⚠️ No layout_instructions found, skipping")
            continue
        
        # Parse JSON
        try:
            # Try to parse directly
            layout_json = json.loads(layout_instructions)
        except json.JSONDecodeError:
            # Try to extract from <output> tags
            output_match = re.search(r'<output>\s*(.*?)\s*</output>', layout_instructions, re.DOTALL)
            if output_match:
                try:
                    layout_json = json.loads(output_match.group(1))
                except json.JSONDecodeError as e:
                    print(f"⚠️ Invalid JSON in layout_instructions: {e}")
                    continue
            else:
                print(f"⚠️ Could not parse layout_instructions")
                continue
        
        # Generate slideshow
        try:
            video_path = generate_slideshow_for_row(
                layout_json=layout_json,
                row_index=idx,
                slide_title=slide_title,
                drive_instance=drive_instance,
                voice=voice
            )
            
            if video_path:
                generated_videos.append(video_path)
                print(f"✅ Slide {slide_num}/{len(rows_to_process)} complete")
                
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("📊 PROCESSING COMPLETE")
    print("="*60)
    print(f"✅ Generated {len(generated_videos)} slide video(s)")
    
    # Combine videos if requested
    if combine_videos and len(generated_videos) > 0:
        combined_path = os.path.join(OUTPUT_DIR, output_filename)
        result = concatenate_videos(generated_videos, combined_path, add_transitions=True)
        
        if result:
            print(f"\n🎉 FINAL VIDEO: {result}")
            return result
        else:
            print("⚠️ Failed to combine videos, returning individual videos")
            return generated_videos
    else:
        for video in generated_videos:
            print(f"   📹 {video}")
        return generated_videos


print("✅ Sheet processing functions loaded")

# =============================================================================
# CELL 9: Video Playback Functions (for Colab)
# =============================================================================

def play_video(video_path: str, width: int = 800, show_download: bool = True):
    """
    Play a video directly in Google Colab notebook with download button.
    
    :param video_path: Path to the video file
    :param width: Width of the video player in pixels
    :param show_download: Whether to show download button
    """
    if video_path is None:
        print("⚠️ No video path provided")
        return
    
    # Handle case where video_path is a list
    if isinstance(video_path, list):
        if len(video_path) == 0:
            print("⚠️ No videos to play")
            return
        video_path = video_path[0]  # Play first video
        print(f"ℹ️ Multiple videos found, playing first one")
    
    if not os.path.exists(video_path):
        print(f"⚠️ Video not found: {video_path}")
        return
    
    try:
        from IPython.display import Video, display, HTML
        from google.colab import files
        import base64
        
        filename = os.path.basename(video_path)
        file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
        
        print(f"\n🎬 Playing: {filename}")
        print(f"   Size: {file_size_mb:.1f} MB")
        
        # Get video duration for info
        try:
            clip = VideoFileClip(video_path)
            duration = clip.duration
            clip.close()
            print(f"   Duration: {duration:.1f}s ({duration/60:.1f} min)")
        except:
            duration = None
        
        # Display the video player
        display(Video(video_path, width=width, embed=True))
        
        # Add download button
        if show_download:
            # Create a styled download button using HTML
            download_html = f'''
            <div style="margin: 10px 0;">
                <button onclick="
                    var link = document.createElement('a');
                    link.href = 'data:video/mp4;base64,{_get_video_base64(video_path)}';
                    link.download = '{filename}';
                    link.click();
                " style="
                    background-color: #4CAF50;
                    border: none;
                    color: white;
                    padding: 12px 24px;
                    text-align: center;
                    text-decoration: none;
                    display: inline-block;
                    font-size: 16px;
                    margin: 4px 2px;
                    cursor: pointer;
                    border-radius: 8px;
                ">
                    📥 Download Video ({file_size_mb:.1f} MB)
                </button>
                <span style="margin-left: 10px; color: #666;">
                    or use: <code>files.download('{video_path}')</code>
                </span>
            </div>
            '''
            
            # For large files, use Colab's files.download instead of base64
            if file_size_mb > 50:
                print(f"\n📥 Large file detected. Click below to download:")
                # Create download button that triggers Colab download
                display(HTML(f'''
                <div style="margin: 10px 0;">
                    <p style="color: #666;">File is large ({file_size_mb:.1f} MB). Use the button below:</p>
                </div>
                '''))
                # Trigger Colab's native download
                try:
                    files.download(video_path)
                except:
                    print(f"   Run this to download: files.download('{video_path}')")
            else:
                # For smaller files, create inline download button
                display(HTML(f'''
                <div style="margin: 15px 0; padding: 10px; background-color: #f0f0f0; border-radius: 8px;">
                    <p style="margin: 0 0 10px 0; font-weight: bold;">📥 Download Options:</p>
                    <button onclick="
                        (function() {{
                            var element = document.createElement('a');
                            element.setAttribute('href', URL.createObjectURL(new Blob([Uint8Array.from(atob('{_get_video_base64_safe(video_path)}'), c => c.charCodeAt(0))], {{type: 'video/mp4'}})));
                            element.setAttribute('download', '{filename}');
                            element.style.display = 'none';
                            document.body.appendChild(element);
                            element.click();
                            document.body.removeChild(element);
                        }})();
                    " style="
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        border: none;
                        color: white;
                        padding: 12px 24px;
                        font-size: 14px;
                        cursor: pointer;
                        border-radius: 6px;
                        margin-right: 10px;
                    ">
                        ⬇️ Download Video
                    </button>
                    <span style="color: #666; font-size: 12px;">
                        Alternative: Run <code>from google.colab import files; files.download('{video_path}')</code>
                    </span>
                </div>
                '''))
        
    except ImportError:
        print("⚠️ IPython.display not available. Run this in Google Colab.")
        print(f"   Video saved at: {video_path}")
    except Exception as e:
        print(f"⚠️ Error playing video: {e}")
        print(f"   Video saved at: {video_path}")
        # Fallback: just show the path
        print(f"\n📥 To download, run: from google.colab import files; files.download('{video_path}')")


def _get_video_base64_safe(video_path: str) -> str:
    """Get base64 encoded video for download button (with size limit)."""
    try:
        file_size = os.path.getsize(video_path)
        # Limit to 50MB for base64 encoding
        if file_size > 50 * 1024 * 1024:
            return ""
        
        with open(video_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    except:
        return ""


def _get_video_base64(video_path: str) -> str:
    """Get base64 encoded video."""
    try:
        with open(video_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    except:
        return ""


def play_video_html5(video_path: str, width: int = 800):
    """
    Alternative method to play video using HTML5 player with base64 encoding.
    Use this if play_video() doesn't work (e.g., for larger files).
    
    :param video_path: Path to the video file
    :param width: Width of the video player in pixels
    """
    if not video_path or not os.path.exists(video_path):
        print(f"⚠️ Video not found: {video_path}")
        return
    
    try:
        from IPython.display import HTML, display
        import base64
        
        print(f"\n🎬 Loading video for playback: {os.path.basename(video_path)}")
        print("   (This may take a moment for large files...)")
        
        # Read video file and encode to base64
        with open(video_path, 'rb') as f:
            video_data = f.read()
        
        video_b64 = base64.b64encode(video_data).decode('utf-8')
        
        # Create HTML5 video player
        html = f'''
        <video width="{width}" controls>
            <source src="data:video/mp4;base64,{video_b64}" type="video/mp4">
            Your browser does not support the video tag.
        </video>
        '''
        
        display(HTML(html))
        print("✅ Video loaded. Use the controls to play.")
        
    except Exception as e:
        print(f"⚠️ Error: {e}")
        print(f"   Video saved at: {video_path}")


print("✅ Video playback functions loaded")

# =============================================================================
# CELL 10: Run the Slideshow Generator
# =============================================================================

# Run this cell to generate the combined slideshow

# Generate and combine all slides into one video
final_video = process_sheet(
    gc=gc,
    sheet_link=sheet_link,
    drive_instance=drive_instance,
    worksheet_name="Slide Chunks",
    voice="en-US-GuyNeural",  # Male voice, or use "en-US-JennyNeural" for female
    combine_videos=True,  # Combine all slides into one video
    output_filename="course_slideshow.mp4"  # Name for the final video
)

# Play the video directly in Colab
play_video(final_video)


