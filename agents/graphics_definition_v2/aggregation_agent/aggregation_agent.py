from langsmith import traceable
import streamlit as st
from services.sheets_service import get_sheet_data_and_df, save_to_sheet, format_worksheet, clear_worksheet
from services.smart_progress_bar import SmartProgressBar
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import requests
from io import BytesIO
from PIL import Image
from agents.vector_store_image_search.graphics_retriever_agent import pil_to_base64_data_uri
from agents.vector_store_image_search.create_vectorstore import download_image_from_drive
from services.video_clip_tools import build_video_part
from dotenv import load_dotenv
import os
import base64
from services.drive_service import login_with_service_account
from pydrive2.drive import GoogleDrive
from google import genai
from google.genai import types
from typing import List, Dict, Optional, Tuple, Any
import tempfile
import subprocess
import urllib.parse

load_dotenv()

# Drive folder ID for storing extracted video frames
video_frames_drive_folder_id = "1sab6wSDPLj54q7KMumGwB1oZHRzXmVf-"


aggregation_agent_prompt = """You are a senior expert graphics designer specializing in the field of HVAC. Your role is to assemble the final, production-ready graphics definition for a single voiceover sentence so that it can be directly used by a graphics team to design the corresponding visuals for an educational e-learning slide. 
You will be given the voiceover sentence, the full course and slide context for reference, and a set of image and video candidates that have already been identified for this sentence by upstream agents, some of which may be only partially relevant or not ultimately suitable for use. Your task is to decide which visuals should be selected from the provided image and video candidates to best support the entire voiceover sentence visually, and whether those visuals should be images, video clips, still frames extracted from videos, or a combination of these.

These are the inputs:

<course_information>
Course name: {course_name}
Topic name: {topic_name}
Subtopic name: {subtopic_name}
</course_information>

<voiceover_sentence>
{vo_text}
</voiceover_sentence>

<whole_slide_context>
Slide Title: {slide_title}
Slide Content: {slide_chunk}
</whole_slide_context>

<image_candidates>
{image_candidates}
</image_candidates>

<video_candidates>
{video_candidates}
</video_candidates>

Instructions:

1. Scope and Decision Responsibility

- Your task is to assemble a final graphics definition for the given voiceover sentence only.
- Use the full slide context strictly for reference and continuity awareness, not to design visuals for other sentences.
- Select visuals only from the provided image and video candidates.

2. Visual Coverage of the Entire Sentence  
   
- First, understand the full meaning and instructional intent of the entire voiceover sentence.
- Identify the key visual ideas that must be shown on screen for the sentence to be clearly understood
- Select visuals so that the chosen visual or visuals, taken together, fully support the complete meaning of the voiceover sentence.

3. Allowed Visual Selection Forms

- You may select one or more still images from the provided image candidates.
- You may select one or more segments from the provided video candidates, including short portions of a video clip that are most relevant to the voiceover sentence.
- You may select a specific still frame from a provided video clip and use it as a static image.
- You may use a combination of still images, video clips, and still frames extracted from video clips, as long as the selected visuals collectively support the entire voiceover sentence.

4. Time-Constrained Visual Design

- Visuals are displayed only during the narration of the voiceover sentence.
- Select the minimum number of visuals required to clearly support the sentence within this limited time.
- Do not select lots of visuals or long video clips that cannot be realistically shown during the narration of the given sentence.

5. Alignment of Visuals to the Voiceover Sentence

- For each selected visual, indicate which part of the voiceover sentence it should appear with during narration.
- Align visuals to the natural progression of the sentence so that each visual appears when the corresponding idea is being spoken.
- If multiple visuals are selected, ensure they are ordered and aligned in a way that makes the sentence easy to follow visually within the narration time.
- Do not assign visuals to parts of the sentence that they do not clearly support.

6. Instructional Clarity Priority

- Prioritize instructional clarity over visual richness or variety.
- When you find both a video clip and a still image that are equally clear, directly relevant, and instructionally effective for any part of the voiceover sentence, prefer using the video clip, since motion can add useful context. This is a guiding preference, not a strict rule—do not prioritize a video clip over an image if the video is only partially relevant, loosely related, or less effective than the still image at supporting the narration.


Strictly provide your output in the following format:

<output>

<evaluation_breakdown>

This section is your reasoning scratchpad used to analyze the voiceover sentence and the available visual candidates before producing the final graphics definition output. Use it to document your observations, reasoning, and decision process. Provide the following sections:

<image_candidates_scan>
Create a numbered list of all provided image candidates and briefly describe what you see in each of the image candidate.
</image_candidates_scan>

<video_candidate_scan>
Create a numbered list of all provided video candidates and briefly describe what you see in each of the video candidate. Focus on explaining the visual content of the video, and not what is being spoken in the video.
</video_candidate_scan>

<voiceover_sentence_understanding>
Briefly explain, in your own words, what the voiceover sentence is communicating. Use the full slide context to resolve any references, pronouns, or implied meaning if needed.
</voiceover_sentence_understanding>

<detailed_overall_analysis>
Use this section to reason through how to assemble the final graphics definition for the given voiceover sentence.

Apply the instruction guidelines to:
- Carefully review each of the provided image and video candidate in detail before making any selection decisions.
- For video candidates, pay close attention to the visual content within the video to identify whether a video segment or a specific still frame from the video can be used as a suitable visual for the corresponding part of the voiceover sentence. Consider whether any visually clear, frame-worthy moments within the videos could be used as static images.
- Decide whether to use still images, video segments, still frames from videos, or a combination of these. While considering the use of any specific video clip for a particular part of the voiceover sentence, determine exactly which portion of the video is visually relevant to help assign correct start and end timestamps for the video clip that you will select.
- Ensure the selected visuals, taken together, fully support the entire meaning of the voiceover sentence.
- Account for the narration time of the voiceover sentence and how all selected visuals should realistically fit within that timeframe.
- Plan how the selected visuals align with the progression of the voiceover sentence.
- Address any other reasoning considerations needed to arrive at a clear and instructionally useful final decision.

Document your reasoning, tradeoffs, and decision process as you work toward the final selection. It is ok for this evaluation breakdown section to be quite long to fit all your reasoning and intermediate thinking needed to arrive at the best final decision.
</detailed_overall_analysis>

</evaluation_breakdown>

Based on your above evaluation, provide the final graphics definition for the voiceover sentence in the following format:

<final_graphics_definition>

<visual_steps>

<visual_step>

<voiceover_part>
(Exact phrase or clause from the voiceover sentence that this visual aligns with)
</voiceover_part>

<visual_instruction>
Concise description of what appears on screen for this part of the narration, using only the selected visual asset.
</visual_instruction>

<asset>
The visual asset selected to use for this step, in one of the following forms:
- Image URL (if a still image is selected for this part of the voiceover sentence)
- Video URL with start and end timestamps (if a portion of a video clip is selected for this part of the voiceover sentence. e.g. "https://www.youtube.com/embed/dQw4w9WgXc?start=10&end=20")
- Video URL with a single start timestamp (if a still frame extracted from a video clip is selected for this part of the voiceover sentence. e.g. "https://www.youtube.com/embed/dQw4w9WgXc?start=10")
</asset>

<selection_justification>
Briefly explain how the selected visual clearly supports this specific part of the voiceover sentence, based on what is visibly shown in the asset. Do not refer to the visual number/index while giving the justification (eg. don't say "Image 9 supports the voiceover sentence because it shows...", instead say "the selectedimage supports the voiceover sentence because it shows...")
</selection_justification>

</visual_step>

<!-- Repeat <visual_step> as needed, in the sentence narration order -->

</visual_steps>

</final_graphics_definition>

</output>
(Ensure that you follow this exact XML format in your output)
"""


def get_drive_instance():
    """
    Get Google Drive instance from session state or initialize from environment.
    
    :return: Google Drive instance
    """
    if "drive" in st.session_state:
        return st.session_state["drive"]
    
    try:
        key_bytes = base64.b64decode(os.environ["GDRIVE_SA_B64"])
        sa_json = key_bytes.decode()
        gauth = login_with_service_account(json_str=sa_json)
        gauth.ServiceAuth()
        drive = GoogleDrive(gauth)
        return drive
    except Exception as e:
        print(f"⚠️ Could not initialize Drive from environment: {e}")
        return None


def extract_drive_file_id(url):
    """
    Extract the Google Drive file ID from a URL.

    :param url: Google Drive URL
    :return: Google Drive file ID or None if not found
    """
    if not url:
        return None
    
    # Pattern for Google Drive URLs
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


def is_drive_url(url):
    """Check if URL is a Google Drive URL."""
    if not url:
        return False
    return "drive.google.com" in url.lower() or "docs.google.com" in url.lower()


def download_image_from_web_url(url):
    """
    Download an image from a web URL and return as PIL Image.

    :param url: Image URL
    :return: PIL Image object or None if download fails
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        return img.convert("RGB")
    except Exception as e:
        print(f"⚠️ Error downloading image from {url}: {e}")
        return None


def load_image_from_drive_url(url, drive, title=""):
    """
    Load an image from Google Drive URL and return as PIL Image.

    :param url: Google Drive URL
    :param drive: Google Drive instance
    :param title: Image title (for logging)
    :return: PIL Image object or None if load fails
    """
    if not drive:
        print(f"⚠️ Drive instance not available for loading image: {title}")
        return None
    
    try:
        file_id = extract_drive_file_id(url)
        if not file_id:
            print(f"⚠️ Could not extract file ID from URL: {url}")
            return None
        
        pil_image = download_image_from_drive(drive, file_id)
        if pil_image:
            return pil_image
        return None
    except Exception as e:
        print(f"⚠️ Error loading image from Drive ({title}): {e}")
        return None


def load_image_from_url(url, drive, title=""):
    """
    Load an image from URL (Drive or web) and return as PIL Image.

    :param url: Image URL
    :param drive: Google Drive instance
    :param title: Image title (for logging)
    :return: PIL Image object or None if load fails
    """
    if is_drive_url(url):
        return load_image_from_drive_url(url, drive, title)
    else:
        return download_image_from_web_url(url)


def parse_urls_from_image_pool(image_pool_text, segment_num):
    """
    Parse image URLs from image_pool column for a specific segment.
    
    :param image_pool_text: The image_pool column content
    :param segment_num: Segment number to extract URLs for
    :return: List of dictionaries with 'title' and 'url' keys or empty list if no URLs found
    """
    if not image_pool_text or image_pool_text.strip() == "" or image_pool_text == "nan":
        return []
    
    # Find the segment section
    segment_pattern = rf'---SEGMENT_{segment_num}---\s*\n(.*?)(?=\n---SEGMENT_|\Z)'
    match = re.search(segment_pattern, image_pool_text, re.DOTALL)
    
    if not match:
        return []
    
    segment_content = match.group(1).strip()
    items = []
    
    # Parse each line: "Title | URL: ..." or "Title | URL"
    for line in segment_content.split('\n'):
        line = line.strip()
        if not line:
            continue
        
        # Try to parse "Title | URL" or "Title | `URL`" format
        if " | " in line:
            parts = line.split(" | ", 1)
            if len(parts) == 2:
                title = parts[0].strip()
                url_part = parts[1].strip()
                
                # Remove backticks if present (format: "Title | `URL`")
                if url_part.startswith("`") and url_part.endswith("`"):
                    url = url_part[1:-1].strip()
                # Handle "Title | URL: ..." format (if URL: is present)
                elif url_part.startswith("URL: "):
                    url = url_part[5:].strip()
                    # Remove backticks if present
                    if url.startswith("`") and url.endswith("`"):
                        url = url[1:-1].strip()
                elif url_part.startswith("URL:"):
                    url = url_part[4:].strip()
                    # Remove backticks if present
                    if url.startswith("`") and url.endswith("`"):
                        url = url[1:-1].strip()
                else:
                    url = url_part
                
                if url:
                    items.append({"title": title, "url": url})
        # Fallback: if line is just a URL
        elif line.startswith('http') or (line.startswith('`http') and line.endswith('`')):
            url = line
            # Remove backticks if present
            if url.startswith("`") and url.endswith("`"):
                url = url[1:-1].strip()
            items.append({"title": "Untitled", "url": url})
    
    return items


def parse_urls_from_video_pool(video_pool_text, segment_num):
    """
    Parse video URLs from video_pool column for a specific segment.
    
    :param video_pool_text: The video_pool column content
    :param segment_num: Segment number to extract URLs for
    :return: List of video URLs (YouTube embed URLs with timestamps) or empty list if no URLs found
    """
    if not video_pool_text or video_pool_text.strip() == "" or video_pool_text == "nan":
        return []
    
    # Find the segment section
    segment_pattern = rf'---SEGMENT_{segment_num}---\s*\n(.*?)(?=\n---SEGMENT_|\Z)'
    match = re.search(segment_pattern, video_pool_text, re.DOTALL)
    
    if not match:
        return []
    
    segment_content = match.group(1).strip()
    urls = []
    
    # Each line is a video URL
    for line in segment_content.split('\n'):
        line = line.strip()
        if line and line.startswith('http'):
            urls.append(line)
    
    return urls


def parse_video_url_timestamps(video_url):
    """
    Parse start and end timestamps from YouTube embed URL.
    
    :param video_url: YouTube embed URL like "https://www.youtube.com/embed/{vid_id}?start=10&end=20"
    :return: Tuple of (clip_url, start_seconds, end_seconds) or (None, None, None) if parsing fails
    """
    if not video_url:
        return None, None, None
    
    # Extract start and end timestamps
    start_match = re.search(r'[?&]start=(\d+)', video_url)
    end_match = re.search(r'[?&]end=(\d+)', video_url)
    
    start_seconds = int(start_match.group(1)) if start_match else None
    end_seconds = int(end_match.group(1)) if end_match else None
    
    # Extract base URL without query params for file_uri
    base_url_match = re.search(r'(https://www\.youtube\.com/embed/[^?]+)', video_url)
    if not base_url_match:
        return None, None, None
    
    clip_url = base_url_match.group(1)
    
    return clip_url, start_seconds, end_seconds


def extract_video_id_from_url(youtube_url):
    """
    Extract video ID from YouTube embed URL.
    
    :param youtube_url: YouTube URL (embed or watch format)
    :return: Video ID string or None if extraction fails
    """
    if not youtube_url:
        return None
    
    # Handle embed URLs: https://www.youtube.com/embed/VIDEO_ID
    embed_match = re.search(r'youtube\.com/embed/([a-zA-Z0-9_-]+)', youtube_url)
    if embed_match:
        return embed_match.group(1)
    
    # Handle watch URLs: https://www.youtube.com/watch?v=VIDEO_ID
    watch_match = re.search(r'youtube\.com/watch\?v=([a-zA-Z0-9_-]+)', youtube_url)
    if watch_match:
        return watch_match.group(1)
    
    # Handle youtu.be URLs: https://youtu.be/VIDEO_ID
    short_match = re.search(r'youtu\.be/([a-zA-Z0-9_-]+)', youtube_url)
    if short_match:
        return short_match.group(1)
    
    return None


def extract_frame_from_youtube_video(video_id, timestamp_seconds, output_path):
    """
    Extract a single frame from a YouTube video at a specific timestamp.
    Uses yt-dlp to download the video and ffmpeg to extract the frame.
    
    :param video_id: YouTube video ID
    :param timestamp_seconds: Timestamp in seconds to extract frame at
    :param output_path: Path where the extracted frame will be saved
    :return: True if successful, False otherwise
    """
    try:
        youtube_url = f"https://www.youtube.com/watch?v={video_id}"
        temp_video_path = os.path.join(tempfile.gettempdir(), f"{video_id}_{timestamp_seconds}_temp.%(ext)s")
        
        try:
            # Use yt-dlp to download best quality video
            # We'll try to download just a segment around the timestamp using --download-sections
            # Format: download-sections "*start-end" (in seconds)
            segment_start = max(0, timestamp_seconds - 10)  # 10 seconds before
            segment_end = timestamp_seconds + 10  # 10 seconds after
            
            yt_dlp_cmd = [
                "yt-dlp",
                "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",  # Best quality
                "--download-sections", f"*{segment_start}-{segment_end}",
                "-o", temp_video_path,
                "--no-playlist",
                youtube_url
            ]
            
            result = subprocess.run(
                yt_dlp_cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            # If --download-sections fails, fall back to downloading full video
            if result.returncode != 0:
                print(f"⚠️ Segment download failed, trying full video download: {result.stderr[:200]}")
                temp_video_path = os.path.join(tempfile.gettempdir(), f"{video_id}_{timestamp_seconds}_temp.%(ext)s")
                yt_dlp_cmd = [
                    "yt-dlp",
                    "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
                    "-o", temp_video_path,
                    "--no-playlist",
                    youtube_url
                ]
                
                result = subprocess.run(
                    yt_dlp_cmd,
                    capture_output=True,
                    text=True,
                    timeout=600  # 10 minute timeout for full video
                )
            
            if result.returncode != 0:
                print(f"⚠️ yt-dlp failed for video {video_id} at {timestamp_seconds}s: {result.stderr[:500]}")
                return False
            
            # Find the actual downloaded file (yt-dlp adds extension)
            actual_video_path = None
            for ext in ['mp4', 'webm', 'mkv']:
                candidate = temp_video_path.replace('%(ext)s', ext)
                if os.path.exists(candidate):
                    actual_video_path = candidate
                    break
            
            if not actual_video_path:
                # Try to find any file starting with the temp name
                temp_dir = os.path.dirname(temp_video_path)
                base_name = os.path.basename(temp_video_path).replace('%(ext)s', '')
                for file in os.listdir(temp_dir):
                    if file.startswith(base_name.split('.')[0]):
                        actual_video_path = os.path.join(temp_dir, file)
                        break
            
            if not actual_video_path or not os.path.exists(actual_video_path):
                print(f"⚠️  Could not find downloaded video file for {video_id}")
                return False
            
            # Calculate relative timestamp if we downloaded a segment
            if '--download-sections' in str(yt_dlp_cmd):
                relative_timestamp = timestamp_seconds - segment_start
            else:
                relative_timestamp = timestamp_seconds
            
            # Extract frame using ffmpeg
            ffmpeg_cmd = [
                "ffmpeg",
                "-i", actual_video_path,
                "-ss", str(relative_timestamp),
                "-vframes", "1",
                "-q:v", "2",  # High quality JPEG (2 is very high quality, lower number = higher quality)
                "-y",  # Overwrite output file
                output_path
            ]
            
            result = subprocess.run(
                ffmpeg_cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode != 0:
                print(f"⚠️ ffmpeg failed to extract frame: {result.stderr[:500]}")
                return False
            
            return os.path.exists(output_path) and os.path.getsize(output_path) > 0
            
        finally:
            # Clean up temp video file(s)
            temp_dir = os.path.dirname(temp_video_path) if '%' not in temp_video_path else tempfile.gettempdir()
            base_name = os.path.basename(temp_video_path).replace('%(ext)s', '').split('.')[0]
            for file in os.listdir(temp_dir):
                if file.startswith(base_name) and (file.endswith('.mp4') or file.endswith('.webm') or file.endswith('.mkv')):
                    try:
                        os.remove(os.path.join(temp_dir, file))
                    except:
                        pass
        
    except subprocess.TimeoutExpired:
        print(f"⚠️ Timeout extracting frame from video {video_id} at {timestamp_seconds}s")
        return False
    except Exception as e:
        print(f"⚠️ Error extracting frame from video {video_id} at {timestamp_seconds}s: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_file_exists_in_drive(drive, folder_id, filename):
    """
    Check if a file with the given filename exists in the specified Drive folder.
    
    :param drive: Google Drive instance
    :param folder_id: Drive folder ID
    :param filename: Name of the file to check
    :return: Drive file object if exists, None otherwise
    """
    try:
        query = f"title='{filename}' and '{folder_id}' in parents and trashed=false"
        file_list = drive.ListFile({'q': query}).GetList()
        
        if file_list:
            return file_list[0]  # Return first matching file
        return None
    except Exception as e:
        print(f"⚠️ Error checking for existing file {filename} in Drive: {e}")
        return None


def upload_image_to_drive(image_path, filename, folder_id, drive):
    """
    Upload an image file to Google Drive folder.
    
    :param image_path: Local path to the image file
    :param filename: Name to use for the file in Drive
    :param folder_id: Drive folder ID to upload to
    :param drive: Google Drive instance
    :return: Shareable Drive URL or None if upload fails
    """
    try:
        # Check if file already exists
        existing_file = check_file_exists_in_drive(drive, folder_id, filename)
        if existing_file:
            file_id = existing_file['id']
            shareable_url = f"https://drive.google.com/file/d/{file_id}/view"
            print(f"✅ File {filename} already exists in Drive, reusing: {shareable_url}")
            return shareable_url
        
        # Upload new file
        file_drive = drive.CreateFile({
            'title': filename,
            'parents': [{'id': folder_id}]
        })
        file_drive.SetContentFile(image_path)
        file_drive.Upload()
        
        # Get shareable URL
        file_id = file_drive['id']
        shareable_url = f"https://drive.google.com/file/d/{file_id}/view"
        
        print(f"✅ Uploaded {filename} to Drive: {shareable_url}")
        return shareable_url
        
    except Exception as e:
        print(f"⚠️ Error uploading {filename} to Drive: {e}")
        return None


def process_video_frames_in_xml(graphics_definition_xml, drive, drive_folder_id=video_frames_drive_folder_id):
    """
    Find all YouTube URLs with only start timestamps (video frames) in the XML,
    extract frames, upload to Drive, and replace URLs in XML.
    
    :param graphics_definition_xml: XML string containing graphics definition
    :param drive: Google Drive instance
    :param drive_folder_id: Drive folder ID to store extracted frames
    :return: Modified XML with Drive URLs instead of YouTube frame URLs, or original XML if processing fails
    """
    if not graphics_definition_xml or not graphics_definition_xml.strip():
        return graphics_definition_xml
    
    if not drive:
        print("⚠️ Drive instance not available, skipping video frame extraction")
        return graphics_definition_xml
    
    try:
        # Find all <asset> tags
        asset_pattern = r'<asset>(.*?)</asset>'
        assets = re.findall(asset_pattern, graphics_definition_xml, re.DOTALL | re.IGNORECASE)
        
        # Dictionary to cache processed frames: {youtube_url: drive_url}
        processed_frames = {}
        
        # Process each asset URL
        for asset_content in assets:
            asset_url = asset_content.strip()
            
            # Check if it's a YouTube URL with only start parameter (no end)
            if 'youtube.com/embed' in asset_url or 'youtube.com/watch' in asset_url:
                # Parse URL to check if it has only start, no end
                parsed_url = urllib.parse.urlparse(asset_url)
                query_params = urllib.parse.parse_qs(parsed_url.query)
                
                has_start = 'start' in query_params or 'start' in asset_url
                has_end = 'end' in query_params or 'end' in asset_url
                
                # If it has start but no end, it's a frame to extract
                if has_start and not has_end:
                    # Check if we've already processed this URL
                    if asset_url in processed_frames:
                        continue
                    
                    # Extract video ID and timestamp
                    video_id = extract_video_id_from_url(asset_url)
                    start_match = re.search(r'[?&]start=(\d+)', asset_url)
                    timestamp = int(start_match.group(1)) if start_match else None
                    
                    if not video_id or timestamp is None:
                        print(f"⚠️ Could not parse video ID or timestamp from URL: {asset_url}")
                        continue
                    
                    # Generate filename
                    filename = f"{video_id}_frame_{timestamp}s.jpg"
                    
                    # Create temp file for extracted frame
                    temp_frame_path = os.path.join(tempfile.gettempdir(), filename)
                    
                    try:
                        print(f"🎬 Extracting frame from video {video_id} at {timestamp}s...")
                        # Extract frame
                        success = extract_frame_from_youtube_video(video_id, timestamp, temp_frame_path)
                        
                        if success and os.path.exists(temp_frame_path):
                            # Upload to Drive
                            drive_url = upload_image_to_drive(
                                temp_frame_path,
                                filename,
                                drive_folder_id,
                                drive
                            )
                            
                            if drive_url:
                                processed_frames[asset_url] = drive_url
                                print(f"✅ Successfully processed frame: {asset_url} -> {drive_url}")
                            else:
                                print(f"⚠️ Failed to upload frame for {asset_url}, keeping original URL")
                        else:
                            print(f"⚠️ Failed to extract frame for {asset_url}, keeping original URL")
                    finally:
                        # Clean up temp frame file
                        if os.path.exists(temp_frame_path):
                            try:
                                os.remove(temp_frame_path)
                            except:
                                pass
        
        # Replace all processed URLs in XML
        if processed_frames:
            modified_xml = graphics_definition_xml
            for youtube_url, drive_url in processed_frames.items():
                # Replace all occurrences of this YouTube URL in the XML
                modified_xml = modified_xml.replace(youtube_url, drive_url)
            
            print(f"✅ Replaced {len(processed_frames)} video frame URLs with Drive URLs")
            return modified_xml
        
        return graphics_definition_xml
        
    except Exception as e:
        print(f"⚠️ Error processing video frames in XML: {e}")
        import traceback
        traceback.print_exc()
        # Return original XML on error
        return graphics_definition_xml


def invoke_gemini_multimodal(parts, llm="gemini_3_flash_thinking", temperature=0.7):
    """
    Invoke the Gemini API with multimodal parts (images, videos, text).

    :param parts: List of Gemini Part objects (text, video, image, etc.)
    :param llm: Model identifier to use
    :param temperature: Temperature setting for generation
    :return: Text response from the model
    """
    # Map model identifier to actual model name
    model_mapping = {
        "gemini_3_flash_thinking": "gemini-3-flash-preview",
        "gemini_3_flash": "gemini-3-flash-preview",
        "gemini_2_5_flash": "gemini-2.5-flash",
        "gemini-2.5-flash": "gemini-2.5-flash",
        "gemini-3-flash-preview": "gemini-3-flash-preview",
    }
    actual_model = model_mapping.get(llm, llm)  # Default to llm if not in mapping
    

    client = genai.Client()
    response = client.models.generate_content(
        model=actual_model,
        contents=types.Content(parts=parts),
        config=types.GenerateContentConfig(
            temperature=temperature,
        ),
    )
    if hasattr(response, "text") and response.text:
        return response.text
    if getattr(response, "candidates", None):
        first_candidate = response.candidates[0]
        if getattr(first_candidate, "content", None) and first_candidate.content.parts:
            part = first_candidate.content.parts[0]
            if hasattr(part, "text"):
                return part.text
    return str(response)


def parse_segments_from_voiceover(voiceover_text):
    """
    Parse segments from voiceover_segment column (newline-separated).
    
    :param voiceover_text: The voiceover_segment column content
    :return: List of tuples (segment_index, segment_text) where segment_index is 1-based or empty list if no segments found
    """
    if not voiceover_text or voiceover_text.strip() == "" or voiceover_text == "nan":
        return []
    
    segments = voiceover_text.strip().split('\n')
    return [(i + 1, seg.strip()) for i, seg in enumerate(segments) if seg.strip()]


@traceable(
    metadata={
        "agent_name": "graphics_definition_v2",
        "step_name": "Aggregation Agent",
        "function_name": "aggregate_graphics_definition_for_segment",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous")
    }
)
def aggregate_graphics_definition_for_segment(vo_text, slide_title, slide_chunk, image_items, video_urls, course_name, topic_name, subtopic_name, drive, llm="gemini_3_flash_thinking"):
    """
    Aggregate graphics definition for a single segment using images and videos.
    
    :param vo_text: Voiceover text for the segment
    :param slide_title: Slide title
    :param slide_chunk: Full slide content
    :param image_items: List of dicts with 'title' and 'url' keys
    :param video_urls: List of YouTube embed URLs with timestamps
    :param course_name: Course name
    :param topic_name: Topic name
    :param subtopic_name: Subtopic name
    :param drive: Google Drive instance
    :param llm: Language model to use
    :return: Graphics definition XML text or None if generation fails
    """
    print(f"📝 Generating aggregation for segment: \"{vo_text[:60]}...\"")
    
    # Build image candidates text
    image_candidates_text = ""
    if image_items:
        image_candidates_text = "\n".join([
            f"{idx + 1}. {item.get('title', 'Untitled')} | URL: {item.get('url', '')}"
            for idx, item in enumerate(image_items)
        ])
    else:
        image_candidates_text = "No image candidates provided."
    
    # Build video candidates text
    video_candidates_text = ""
    if video_urls:
        video_candidates_text = "\n".join([
            f"{idx + 1}. {url}"
            for idx, url in enumerate(video_urls)
        ])
    else:
        video_candidates_text = "No video candidates provided."
    
    # Format prompt
    prompt_text = aggregation_agent_prompt.format(
        course_name=course_name,
        topic_name=topic_name,
        subtopic_name=subtopic_name,
        vo_text=vo_text,
        slide_title=slide_title,
        slide_chunk=slide_chunk,
        image_candidates=image_candidates_text,
        video_candidates=video_candidates_text
    )
    
    # Build multimodal parts: images + videos + text
    parts: List[types.Part] = []
    
    # Add images
    for idx, item in enumerate(image_items, start=1):
        image_url = item.get("url", "")
        image_title = item.get("title", f"Image {idx}")
        
        if image_url:
            # Load image
            pil_image = load_image_from_url(image_url, drive, image_title)
            if pil_image:
                # Convert PIL image to bytes
                buffered = BytesIO()
                pil_image.convert("RGB").save(buffered, format="JPEG")
                image_bytes = buffered.getvalue()
                # Add text label
                parts.append(types.Part(text=f"Image {idx}: {image_title}\nURL: {image_url}"))
                # Add image part
                parts.append(types.Part(inline_data=types.Blob(mime_type="image/jpeg", data=image_bytes)))
                print(f"✅ Loaded image {idx}: {image_title}")
            else:
                print(f"⚠️ Failed to load image {idx}: {image_title}")
    
    # Add videos
    for idx, video_url in enumerate(video_urls, start=1):
        clip_url, start_seconds, end_seconds = parse_video_url_timestamps(video_url)
        if clip_url:
            # Add text label
            parts.append(types.Part(text=f"Video {idx}:\nURL: {video_url}"))
            # Add video part
            video_part = build_video_part(clip_url, start_seconds, end_seconds)
            parts.append(video_part)
            print(f"✅ Added video {idx}: start={start_seconds}s, end={end_seconds}s")
        else:
            print(f"⚠️ Failed to parse video URL {idx}: {video_url}")
    
    # Add prompt text at the end
    parts.append(types.Part(text=prompt_text))
        
    # Call LLM
    try:
        print(f" 🤖 Calling {llm} with {len(image_items)} images and {len(video_urls)} videos...")
        response_text = invoke_gemini_multimodal(parts, llm=llm, temperature=0.7)
        
        # Print the full response for debugging
        print(f"\n{'─'*80}")
        print(f"📤 Aggregation Agent Response from LLM for segment: \"{vo_text[:60]}...\"")
        print(f"{'─'*80}")
        print(response_text)
        print(f"{'─'*80}\n")
        
        # Extract <final_graphics_definition> content
        final_def_match = re.search(
            r'<final_graphics_definition>(.*?)</final_graphics_definition>',
            response_text,
            re.DOTALL | re.IGNORECASE
        )
        
        if final_def_match:
            final_graphics_definition = final_def_match.group(1).strip()
            print(f" ✅ Successfully generated graphics definition")
            return final_graphics_definition
        else:
            print(f" ⚠️  Could not extract <final_graphics_definition> from response")
            return None
            
    except Exception as e:
        print(f" ❌ Error calling LLM: {e}")
        return None


def format_aggregation_definition_for_sheet(vo_text, graphics_definition_xml, segment_num):
    """
    Format the aggregated graphics definition XML into the readable format for the sheet.
    
    :param vo_text: Voiceover text for the segment
    :param graphics_definition_xml: Graphics definition XML from <final_graphics_definition>
    :param segment_num: Segment number
    :return: Formatted graphics definition text or empty string if graphics definition XML is empty
    """
    if not graphics_definition_xml or graphics_definition_xml.strip() == "":
        return ""
    
    # Extract all visual steps
    visual_steps_pattern = r'<visual_step>(.*?)</visual_step>'
    visual_steps = re.findall(visual_steps_pattern, graphics_definition_xml, re.DOTALL | re.IGNORECASE)
    
    if not visual_steps:
        # If no visual steps found, return empty
        return ""
    
    # Build formatted output
    formatted_parts = []
    
    # Segment header
    formatted_parts.append("=" * 80)
    formatted_parts.append(f"SEGMENT {segment_num}")
    formatted_parts.append("=" * 80)
    formatted_parts.append("")  # Empty line after header
    
    # Process each visual step
    for step_idx, step_xml in enumerate(visual_steps):
        # Extract components from each visual step
        voiceover_match = re.search(r'<voiceover_part>(.*?)</voiceover_part>', step_xml, re.DOTALL | re.IGNORECASE)
        instruction_match = re.search(r'<visual_instruction>(.*?)</visual_instruction>', step_xml, re.DOTALL | re.IGNORECASE)
        asset_match = re.search(r'<asset>(.*?)</asset>', step_xml, re.DOTALL | re.IGNORECASE)
        justification_match = re.search(r'<selection_justification>(.*?)</selection_justification>', step_xml, re.DOTALL | re.IGNORECASE)
        
        # Add separator between multiple visual steps (except before the first one)
        if step_idx > 0:
            formatted_parts.append("----")
            formatted_parts.append("")
        
        # When VO: (with quotes)
        if voiceover_match:
            vo_part = voiceover_match.group(1).strip()
            formatted_parts.append(f'When VO: "{vo_part}"')
            formatted_parts.append("")  # One line gap
        
        # Visual Instructions:
        if instruction_match:
            instruction = instruction_match.group(1).strip()
            formatted_parts.append(f"Visual Instructions: {instruction}")
            formatted_parts.append("")  # One line gap
        
        # Graphics to use:
        if asset_match:
            asset_url = asset_match.group(1).strip()
            formatted_parts.append(f"Graphics to use: {asset_url}")
            formatted_parts.append("")  # One line gap
        
        # Selection Justification:
        if justification_match:
            justification = justification_match.group(1).strip()
            formatted_parts.append(f"Selection Justification: {justification}")
            formatted_parts.append("")  # One line gap
    
    # Join all parts with newlines
    formatted_text = "\n".join(formatted_parts)
    
    return formatted_text


@traceable(
    metadata={
        "agent_name": "graphics_definition_v2",
        "step_name": "Aggregation Agent",
        "function_name": "process_aggregation_segment",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous")
    }
)
def process_aggregation_segment(segment_idx, vo_text, slide_title, slide_chunk, image_pool_text, video_pool_text, course_name, topic_name, subtopic_name, drive, llm="gemini_3_flash_thinking"):
    """
    Process a single segment: aggregate graphics definition from images and videos.
    
    :param segment_idx: Segment index (1-based)
    :param vo_text: Voiceover text for the segment
    :param slide_title: Slide title
    :param slide_chunk: Full slide content
    :param image_pool_text: Image pool column content
    :param video_pool_text: Video pool column content
    :param course_name: Course name
    :param topic_name: Topic name
    :param subtopic_name: Subtopic name
    :param drive: Google Drive instance
    :param llm: Language model to use
    :return: Tuple of (segment_idx, formatted_segment_text) or (segment_idx, None) if no definition generated
    """
    print(f"\n📦 Processing SEGMENT_{segment_idx}")
    
    # Parse image and video items for this segment
    image_items = parse_urls_from_image_pool(image_pool_text, segment_idx)
    video_urls = parse_urls_from_video_pool(video_pool_text, segment_idx)
    
    print(f" 🖼️  Found {len(image_items)} image candidates")
    print(f" 🎥 Found {len(video_urls)} video candidates")
    
    if not image_items and not video_urls:
        print(f" ⚠️  No image or video candidates available for segment {segment_idx}, skipping")
        return segment_idx, None
    
    # Generate aggregated graphics definition
    graphics_definition_xml = aggregate_graphics_definition_for_segment(
        vo_text=vo_text,
        slide_title=slide_title,
        slide_chunk=slide_chunk,
        image_items=image_items,
        video_urls=video_urls,
        course_name=course_name,
        topic_name=topic_name,
        subtopic_name=subtopic_name,
        drive=drive,
        llm=llm
    )
    
    if graphics_definition_xml:
        # Post-processing: Extract video frames and replace with Drive URLs
        print(f"🔄 Processing video frames in graphics definition...")
        graphics_definition_xml = process_video_frames_in_xml(
            graphics_definition_xml,
            drive,
            video_frames_drive_folder_id
        )
        
        # Format the definition for the sheet
        formatted_segment = format_aggregation_definition_for_sheet(
            vo_text,
            graphics_definition_xml,
            segment_idx
        )
        if formatted_segment:
            return segment_idx, formatted_segment
    
    return segment_idx, None


@traceable(
    metadata={
        "agent_name": "graphics_definition_v2",
        "step_name": "Aggregation Agent",
        "function_name": "process_aggregation_row",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous")
    }
)
def process_aggregation_row(index, row, course_name, drive, llm="gemini_3_flash_thinking", max_workers=3):
    """
    Process a single row: aggregate graphics for all segments and combine into final definition.
    
    :param index: Row index
    :param row: Pandas Series with row data
    :param course_name: Course name
    :param drive: Google Drive instance
    :param llm: Language model to use
    :param max_workers: Max parallel workers for segment processing
    :return: Tuple of (index, final_graphics_definition_text) or (index, empty string) if no segments found
    """
    try:
        voiceover_text = str(row.get("voiceover_segment", "")).strip()
        image_pool_text = str(row.get("image_pool", "")).strip()
        video_pool_text = str(row.get("video_pool", "")).strip()
        
        # Skip if voiceover_segment is empty
        if not voiceover_text or voiceover_text == "nan":
            return index, ""
        
        # Parse segments
        segments = parse_segments_from_voiceover(voiceover_text)
        if not segments:
            return index, ""
        
        # Get row data
        topic_name = str(row.get("Topic", "")).strip()
        subtopic_name = str(row.get("Subtopic", "")).strip()
        slide_title = str(row.get("Slide Chunk Title", "")).strip()
        slide_chunk = str(row.get("Slide Chunk", "")).strip()
        
        print(f"\n{'='*80}")
        print(f"📋 Row {index + 2}: Processing {len(segments)} segment(s)")
        print(f"{'='*80}")
        
        # Process all segments in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            
            # Submit all segments
            futures = {
                executor.submit(
                    process_aggregation_segment,
                    segment_idx,
                    vo_text,
                    slide_title,
                    slide_chunk,
                    image_pool_text,
                    video_pool_text,
                    course_name,
                    topic_name,
                    subtopic_name,
                    drive,
                    llm
                ): (segment_idx, vo_text)
                for segment_idx, vo_text in segments
            }
            
            # Collect results as they complete
            segment_results = {}
            for future in as_completed(futures):
                segment_idx, vo_text = futures[future]
                try:
                    result_idx, formatted_segment = future.result()
                    if formatted_segment:
                        segment_results[result_idx] = formatted_segment
                except Exception as e:
                    print(f"❌ Error processing segment {segment_idx} (\"{vo_text[:50]}...\"): {e}")
        
        # Combine all segments in order
        if segment_results:
            all_segment_results = [
                segment_results[seg_idx]
                for seg_idx in sorted(segment_results.keys())
            ]
            final_graphics_definition_text = '\n\n'.join(all_segment_results)
            return index, final_graphics_definition_text
        
        return index, ""
        
    except Exception as e:
        print(f"Error processing row {index}: {e}")
        return index, ""


def validate_final_graphics_definition_row(row):
    """
    Validate that final_graphics_definition matches voiceover_segment:
    - Row is not empty
    - All segments from voiceover_segment have results
    - No gaps in segment numbering (must be sequential starting from 1)
    
    :param row: Pandas Series with row data
    :return: Tuple (is_valid, error_message)
    """
    vo_segments_text = str(row.get("voiceover_segment", "")).strip()
    final_graphics_def_text = str(row.get("final_graphics_definition", "")).strip()
    
    # Skip validation if voiceover_segment is empty
    if not vo_segments_text or vo_segments_text == "nan":
        return True, None
    
    # Count segments in voiceover_segment (split by newline)
    vo_segments = [seg.strip() for seg in vo_segments_text.split('\n') if seg.strip()]
    expected_count = len(vo_segments)
    
    # Skip validation if final_graphics_definition is empty (will be caught by retry logic)
    if not final_graphics_def_text or final_graphics_def_text == "nan" or final_graphics_def_text.strip() == "":
        return False, "final_graphics_definition is empty"
    
    # Skip validation if it's an error marker
    if final_graphics_def_text.startswith("ERROR:"):
        return False, "final_graphics_definition contains error marker"
    
    # Parse segment numbers from final_graphics_definition
    # Format: "SEGMENT X" (not "---SEGMENT_X---")
    segment_pattern = r'SEGMENT\s+(\d+)'
    segment_numbers = [int(match) for match in re.findall(segment_pattern, final_graphics_def_text, re.IGNORECASE)]
    
    if not segment_numbers:
        return False, "No segment markers found in final_graphics_definition"
    
    # Check count matches
    actual_count = len(segment_numbers)
    if actual_count != expected_count:
        return False, f"Segment count mismatch: expected {expected_count}, found {actual_count}"
    
    # Check for gaps (must be sequential starting from 1)
    segment_numbers_sorted = sorted(segment_numbers)
    expected_sequence = list(range(1, expected_count + 1))
    
    if segment_numbers_sorted != expected_sequence:
        missing = set(expected_sequence) - set(segment_numbers_sorted)
        extra = set(segment_numbers_sorted) - set(expected_sequence)
        error_parts = []
        if missing:
            error_parts.append(f"missing segments: {sorted(missing)}")
        if extra:
            error_parts.append(f"extra segments: {sorted(extra)}")
        return False, f"Segment numbering gap: {', '.join(error_parts)}"
    
    return True, None


@traceable(
    metadata={
        "agent_name": "graphics_definition_v2",
        "step_name": "Aggregation Agent",
        "function_name": "run_aggregation_agent_for_all_rows",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous")
    }
)
def run_aggregation_agent_for_all_rows(sheet, llm="gemini_3_flash_thinking", max_workers=5):
    """
    Run aggregation agent for all rows in the Slide Chunks sheet.

    :param sheet: The gspread sheet object.
    :param llm: The language model to use.
    :param max_workers: Number of parallel workers (default 5).
    :return: None
    """
    worksheet_name = "Slide Chunks"
    ws, df = get_sheet_data_and_df(sheet, worksheet_name)
    
    # Get course name from Course info tab
    _, course_info_df = get_sheet_data_and_df(sheet, "Course info")
    course_name = course_info_df.loc[0, "Course Name"]
    
    print(f"\n{'='*80}")
    print(f"🚀 Starting Aggregation Agent")
    print(f"📚 Course: {course_name}")
    print(f"📊 Processing {len(df)} row(s)")
    print(f"{'='*80}\n")
    
    # Get Drive instance
    drive = get_drive_instance()
    if not drive:
        print("❌ Could not initialize Google Drive. Aborting.")
        return
    
    # Ensure final_graphics_definition column exists
    if "final_graphics_definition" not in df.columns:
        df["final_graphics_definition"] = ""
    
    # Filter rows that have voiceover_segment but missing final_graphics_definition
    rows_to_process = []
    for index, row in df.iterrows():
        voiceover_segment = str(row.get("voiceover_segment", "")).strip()
        final_graphics_def = str(row.get("final_graphics_definition", "")).strip()
        
        # Skip if no voiceover_segment
        if not voiceover_segment or voiceover_segment == "nan":
            continue
        
        # Skip if final_graphics_definition is already filled
        if final_graphics_def and final_graphics_def != "nan":
            continue
        
        rows_to_process.append((index, row))
    
    if not rows_to_process:
        print("✅ No rows to process. All rows already have final_graphics_definition or missing voiceover_segment.")
        return
    
    print(f"📝 Processing {len(rows_to_process)} row(s) with missing final_graphics_definition\n")
    
    # Initialize progress bar
    total_tasks = len(rows_to_process)
    progress = SmartProgressBar(total_tasks, "Aggregation Agent")
    
    # Process rows in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all rows
        futures = {
            executor.submit(
                process_aggregation_row,
                index,
                row,
                course_name,
                drive,
                llm
            ): index
            for index, row in rows_to_process
        }
        
        # Collect results as they complete
        for future in as_completed(futures):
            index = futures[future]
            try:
                row_index, final_graphics_def_text = future.result()
                
                # Update dataframe
                df.at[row_index, "final_graphics_definition"] = final_graphics_def_text
                
                # Update progress
                progress.update()
                
                # Save every 5 rows
                if progress.should_save():
                    print(f'Saving partial progress to sheet after {progress.completed_count} tasks completed.')
                    save_to_sheet(ws, df)
                    format_worksheet(ws)
            except Exception as e:
                print(f"Error getting result for row {index}: {e}")
                # Update dataframe with error marker so row is marked as processed
                df.at[index, "final_graphics_definition"] = f"ERROR: {str(e)}"
                progress.update()

    # Validation and retry logic
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        # Reload dataframe to get latest state
        ws, df = get_sheet_data_and_df(sheet, worksheet_name)
        
        # Validate all rows and find invalid ones
        invalid_rows = []
        for index, row in df.iterrows():
            is_valid, error_msg = validate_final_graphics_definition_row(row)
            if not is_valid:
                invalid_rows.append((index, row, error_msg))
        
        if not invalid_rows:
            break
        
        retry_count += 1
        print(f"\n⚠️ Found {len(invalid_rows)} rows with invalid final_graphics_definition. Retrying (attempt {retry_count}/{max_retries})...")
        for idx, row, error in invalid_rows[:3]:  # Show first 3 errors
            print(f"  Row {idx}: {error}")
        
        # Clear final_graphics_definition for invalid rows
        for index, row, error_msg in invalid_rows:
            df.at[index, "final_graphics_definition"] = ""
        
        # Save cleared state
        save_to_sheet(ws, df)
        format_worksheet(ws)
        
        # Retry processing invalid rows
        futures = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for index, row, error_msg in invalid_rows:
                future = executor.submit(
                    process_aggregation_row,
                    index,
                    row,
                    course_name,
                    drive,
                    llm
                )
                futures[future] = index
            
            # Collect results
            for future in as_completed(futures):
                index = futures[future]
                try:
                    row_index, final_graphics_def_text = future.result()
                    df.at[row_index, "final_graphics_definition"] = final_graphics_def_text
                except Exception as e:
                    print(f"Error getting result for row {index} on retry: {e}")
                    df.at[index, "final_graphics_definition"] = f"ERROR: {str(e)}"
        
        # Save after retry
        save_to_sheet(ws, df)
        format_worksheet(ws)
    
    if retry_count > 0:
        # Check final state
        ws, df = get_sheet_data_and_df(sheet, worksheet_name)
        final_invalid = []
        for index, row in df.iterrows():
            is_valid, error_msg = validate_final_graphics_definition_row(row)
            if not is_valid:
                final_invalid.append((index, error_msg))
        
        if final_invalid:
            print(f"⚠️ After {retry_count} retry attempt(s), {len(final_invalid)} rows still have invalid final_graphics_definition.")
            for idx, error in final_invalid[:5]:  # Show first 5
                print(f"  Row {idx}: {error}")
        else:
            print(f"✅ All rows validated after {retry_count} retry attempt(s).")
    
    # Final save to sheet
    print('All aggregation agent tasks completed. Saving final DataFrame to sheet.')
    save_to_sheet(ws, df)
    format_worksheet(ws)
    print("✅ Aggregation agent complete and saved to sheet.")


def delete_final_graphics_definition(sheet):
    """
    Remove the 'final_graphics_definition' column from the specified worksheet.
    
    :param sheet: The gspread sheet object.
    :return: None
    """
    worksheet_name = "Slide Chunks"
    ws, df = get_sheet_data_and_df(sheet, worksheet_name)
    if "final_graphics_definition" in df.columns:
        df = df.drop(columns=["final_graphics_definition"])
        clear_worksheet(ws)
        save_to_sheet(ws, df)
        print(f"🗑️ Deleted 'final_graphics_definition' column from '{worksheet_name}' worksheet")
    else:
        print(f"ℹ️ 'final_graphics_definition' column does not exist in '{worksheet_name}' worksheet")