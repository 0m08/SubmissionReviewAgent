"""
Video Embeddings Creation Pipeline
==================================

This module creates video embeddings from YouTube videos using Vertex AI's MultiModal Embedding Model.
It processes videos in 30-second chunks and stores them in a Chroma vectorstore for semantic search.

Features:
- YouTube video download and segmentation
- Vertex AI video embedding generation
- Local + Google Drive vectorstore storage
- Incremental processing with progress tracking
- Sheet-based logging and resume capability
- Parallel processing for embedding generation

Author: Graphics Definition Pipeline
"""

import os
import tempfile
import shutil
import subprocess
import json
import time
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple

import pandas as pd
import numpy as np
from tqdm import tqdm
import yt_dlp
from moviepy.editor import VideoFileClip

# Vertex AI imports
import vertexai
from vertexai.vision_models import MultiModalEmbeddingModel, Video, VideoSegmentConfig
from google.oauth2 import service_account

# ChromaDB imports
from chromadb import PersistentClient

# Local service imports
from services.sheets_service import get_sheet_data_and_df, save_to_sheet
from services.drive_service import upload_folder_to_drive, download_folder_from_drive
from services.smart_progress_bar import SmartProgressBar


# ============================================================
# CONFIGURATION
# ============================================================

# Hardcoded configuration
SHEET_URL = "https://docs.google.com/spreadsheets/d/1p9G7SeBRzfl1KGcEzqzF6zG5TO7B__kuSVlY3tQhB2U/edit?gid=687427813#gid=687427813"
# SHEET_TAB = "3D Animations and Simulations"  # Old value - commented out
SHEET_TAB = "3D, Hands-On,Equipment Demos"
# PARENT_FOLDER_ID = "1SSvxx1EJ3zgMPfvD8Zy5DaEgmI2prys7"  # Old value - commented out
PARENT_FOLDER_ID = "15H9thXq02JX3ldADSj1oD78mbV-fXfvu"
VECTORSTORE_FOLDER_NAME = "Vectorstore for HVAC school video embeddings (Category - 3D Animations and Simulations, Hands-On Field Work, Equipment Demos & Teardowns)"
# LOCAL_VECTORSTORE_PATH = "/tmp/HVAC Video Embeddings"  # Old value - commented out
LOCAL_VECTORSTORE_PATH = "/tmp/Youtube Video Vectorstore for HVAC Channel (Category - 3D Animations and Simulations, Hands-On Field Work, Equipment Demos & Teardowns)"

# Processing configuration
CHUNK_LENGTH_SEC = 30
MAX_WORKERS = 15
MAX_ENCODED_BYTES = 27_000_000
SAFE_THRESHOLD = 24_000_000
PRICE_PER_M_TOKEN = 0.30

# Required columns
REQUIRED_COLUMNS = ["video_id", "video_url"]
TRACKING_COLUMNS = ["vectorized", "embedding_ts", "segments_processed"]


# ============================================================
# VERTEX AI SETUP
# ============================================================

def initialize_vertex_ai():
    """Initialize Vertex AI with service account credentials."""
    try:
        # Use the same approach as the working Colab code
        # Try to get from environment variables first (like VERTEX_AI_SA_B64)
        import base64
        import json
        
        if "VERTEX_AI_SA_B64" in os.environ:
            # Decode base64 service account
            key_bytes = base64.b64decode(os.environ["VERTEX_AI_SA_B64"])
            sa_json = key_bytes.decode()
            sa_dict = json.loads(sa_json)
            creds = service_account.Credentials.from_service_account_info(sa_dict)
        else:
            # Fallback to file path (like in Colab)
            creds_path = os.getenv("VERTEX_CREDS_PATH", "/path/to/service-account.json")
            
            if os.path.exists(creds_path):
                creds = service_account.Credentials.from_service_account_file(creds_path)
            else:
                # Try to get from environment variables
                creds_json = os.getenv("VERTEX_CREDS_JSON")
                if creds_json:
                    creds = service_account.Credentials.from_service_account_info(json.loads(creds_json))
                else:
                    raise Exception("Vertex AI credentials not found. Please set VERTEX_AI_SA_B64, VERTEX_CREDS_PATH, or VERTEX_CREDS_JSON environment variable.")
        
        vertexai.init(
            # project="dam-images-tagging",  # Old project - commented out
            #project="handy-reference-478614-e2",  # Niket's project
            project="beaming-theorem-485115-k5",
            location="us-central1",
            credentials=creds
        )
        
        model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding@001")
        print("✅ Vertex AI video embedding model loaded successfully!")
        return model
        
    except Exception as e:
        print(f"❌ Failed to initialize Vertex AI: {e}")
        raise e


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def create_or_get_drive_folder(drive, parent_folder_id, folder_name):
    """
    Create or get a folder in Google Drive.
    
    Args:
        drive: PyDrive GoogleDrive object
        parent_folder_id: Parent folder ID
        folder_name: Name of folder to create/get
    
    Returns:
        folder_id: ID of the folder
    """
    # Check if folder exists
    folder_list = drive.ListFile({
        'q': f"title='{folder_name}' and '{parent_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
    }).GetList()
    
    if folder_list:
        folder_id = folder_list[0]['id']
        print(f"✅ Found existing '{folder_name}' folder: {folder_id}")
    else:
        print(f"📁 Creating '{folder_name}' folder...")
        folder_metadata = {
            'title': folder_name,
            'parents': [{'id': parent_folder_id}],
            'mimeType': 'application/vnd.google-apps.folder'
        }
        new_folder = drive.CreateFile(folder_metadata)
        new_folder.Upload()
        folder_id = new_folder['id']
        print(f"✅ Created folder with ID: {folder_id}")
    
    return folder_id


def check_vectorstore_exists_in_drive(drive, parent_folder_id):
    """
    Check if vectorstore exists in Google Drive.
    
    Args:
        drive: PyDrive GoogleDrive object
        parent_folder_id: ID of the vectorstore folder
    
    Returns:
        tuple: (exists: bool, chroma_folder_id: str or None)
    """
    chroma_folder_list = drive.ListFile({
        'q': f"title='chroma_video_embeddings_db' and '{parent_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
    }).GetList()
    
    if chroma_folder_list:
        return True, chroma_folder_list[0]['id']
    return False, None


def calculate_cost(tokens_used):
    """Calculate estimated cost based on tokens used."""
    return round((tokens_used / 1_000_000) * PRICE_PER_M_TOKEN, 6)


# ============================================================
# VIDEO PROCESSING FUNCTIONS
# ============================================================

def download_video(url, output_dir):
    """
    Download YouTube video using yt-dlp with robust format selection.
    
    Handles YouTube's SABR streaming changes by using format selectors that
    avoid problematic formats and includes fallback options.
    
    Args:
        url: YouTube video URL
        output_dir: Directory to save the video
    
    Returns:
        str: Path to downloaded video file
    """
    output_path = os.path.join(output_dir, "video.mp4")
    
    # Format selection strategies - try in order until one works
    # These avoid SABR-affected formats that cause empty downloads
    format_options = [
        # Option 1: Best pre-merged mp4 format (avoids SABR issues)
        "best[ext=mp4][vcodec^=avc]/best[ext=mp4]/best",
        # Option 2: Specific resolution with merging
        "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height<=720]+bestaudio/best[height<=720]/best",
        # Option 3: Any available format, let yt-dlp figure it out
        "bestvideo+bestaudio/best",
    ]
    
    base_opts = {
        "outtmpl": f"{output_dir}/video.%(ext)s",
        "quiet": True,
        "no_warnings": False,  # Keep warnings visible for debugging
        "ignoreerrors": False,
        # Force IPv4 to avoid some network issues
        "source_address": "0.0.0.0",
        # Extractor arguments to work around SABR
        "extractor_args": {
            "youtube": {
                # Use android client which isn't affected by SABR
                "player_client": ["android", "web"],
                # Skip dash formats that are affected by SABR
                "skip": ["dash"],
            }
        },
        # Merge to mp4 if needed
        "merge_output_format": "mp4",
        # Postprocessor to ensure mp4 output
        "postprocessors": [{
            "key": "FFmpegVideoConvertor",
            "preferedformat": "mp4",
        }],
    }
    
    last_error = None
    
    for i, fmt in enumerate(format_options):
        try:
            print(f"  📥 Download attempt {i+1}/{len(format_options)} with format: {fmt[:50]}...")
            
            ydl_opts = {**base_opts, "format": fmt}
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            
            # Check if file was downloaded (look for any video file)
            for ext in ["mp4", "webm", "mkv", "avi"]:
                potential_path = os.path.join(output_dir, f"video.{ext}")
                if os.path.exists(potential_path) and os.path.getsize(potential_path) > 0:
                    # If not mp4, rename/convert
                    if ext != "mp4":
                        # Use ffmpeg to convert to mp4
                        subprocess.run([
                            "ffmpeg", "-y", "-i", potential_path,
                            "-c", "copy", output_path
                        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        os.remove(potential_path)
                    else:
                        output_path = potential_path
                    
                    print(f"  ✅ Download successful!")
                    return output_path
            
            # If we get here, file wasn't found or was empty
            raise Exception("Downloaded file is empty or not found")
            
        except Exception as e:
            last_error = e
            print(f"  ⚠️ Format attempt {i+1} failed: {str(e)[:100]}")
            # Clean up any partial downloads
            for ext in ["mp4", "webm", "mkv", "avi", "part"]:
                potential_path = os.path.join(output_dir, f"video.{ext}")
                if os.path.exists(potential_path):
                    try:
                        os.remove(potential_path)
                    except:
                        pass
            continue
    
    # All attempts failed
    print(f"❌ Download failed for {url}: {last_error}")
    raise Exception(f"All download attempts failed: {last_error}")


def segment_video(video_path, output_dir, chunk_length=CHUNK_LENGTH_SEC):
    """
    Split video into chunks using ffmpeg.
    
    Args:
        video_path: Path to input video
        output_dir: Directory to save segments
        chunk_length: Length of each chunk in seconds
    
    Returns:
        List[str]: Paths to segment files
    """
    os.makedirs(output_dir, exist_ok=True)

    # Get total duration
    clip = VideoFileClip(video_path)
    duration = int(clip.duration)
    clip.close()

    segment_paths = []

    for start in range(0, duration, chunk_length):
        end = min(start + chunk_length, duration)
        segment_file = os.path.join(output_dir, f"segment_{start}_{end}.mp4")

        # ffmpeg trim with stream copy (very fast)
        cmd = [
            "ffmpeg",
            "-y",                      # overwrite
            "-ss", str(start),         # start time
            "-to", str(end),           # end time
            "-i", video_path,
            "-c", "copy",              # copy both audio & video
            segment_file
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        segment_paths.append(segment_file)

    print(f"✅ Created {len(segment_paths)} segments in {output_dir}")
    return segment_paths


def generate_video_embedding(video_path, model):
    """
    Generate video embeddings using Vertex AI.
    
    Args:
        video_path: Path to video file
        model: Vertex AI MultiModalEmbeddingModel
    
    Returns:
        tuple: (embeddings_list, estimated_tokens, estimated_cost)
    """
    try:
        import base64
        
        def get_encoded_size(path):
            with open(path, "rb") as f:
                data = f.read()
            return len(base64.b64encode(data))

        print(f"  📹 Loading video: {os.path.basename(video_path)}")
        raw_size = os.path.getsize(video_path)
        print(f"  📏 Raw file size: {raw_size / (1024 * 1024):.2f} MB")

        encoded_size = get_encoded_size(video_path)
        print(f"  🧮 Encoded size: {encoded_size / (1024 * 1024):.2f} MB")

        # Recompress iteratively until below safe threshold
        compressed_path = None
        target_bitrate = 800  # start bitrate in kbps
        
        while encoded_size > SAFE_THRESHOLD:
            compressed_path = video_path.replace(".mp4", f"reduced{target_bitrate}k.mp4")
            print(f"  ⚙️ Compressing -> target bitrate {target_bitrate}k...")
            cmd = [
                "ffmpeg", "-y",
                "-i", video_path,
                "-vf", "scale=640:-1",  # shrink resolution
                "-b:v", f"{target_bitrate}k",
                "-b:a", "64k",
                compressed_path
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            encoded_size = get_encoded_size(compressed_path)
            print(f"  📦 New encoded size: {encoded_size / (1024 * 1024):.2f} MB")

            # Lower bitrate progressively if still too large
            if encoded_size > SAFE_THRESHOLD:
                target_bitrate = int(target_bitrate * 0.75)
                if target_bitrate < 150:
                    print("  ⚠️ Minimum bitrate reached, stopping recompression loop.")
                    break
            else:
                video_path = compressed_path
                break

        # Generate embeddings
        print(f"  🤖 Calling Vertex AI to generate embeddings...")
        video = Video.load_from_file(video_path)
        result = model.get_embeddings(
            video=video,
            video_segment_config=VideoSegmentConfig(end_offset_sec=CHUNK_LENGTH_SEC)
        )

        print(f"  ✅ Received embeddings from Vertex AI")
        embeddings = []
        for seg in result.video_embeddings:
            embeddings.append({
                "start": seg.start_offset_sec,
                "end": seg.end_offset_sec,
                "vector": seg.embedding
            })

        print(f"  📊 Generated {len(embeddings)} embedding(s)")
        estimated_tokens = len(embeddings) * 50_000
        estimated_cost = calculate_cost(estimated_tokens)
        print(f"  💰 Tokens: {estimated_tokens}, cost: ${estimated_cost}")

        # Cleanup any temporary compressed file
        if compressed_path and os.path.exists(compressed_path):
            os.remove(compressed_path)

        return embeddings, estimated_tokens, estimated_cost

    except Exception as e:
        print(f"  ❌ Embedding generation failed: {e}")
        raise RuntimeError(f"Embedding generation failed: {e}")


# ============================================================
# VECTORSTORE MANAGEMENT
# ============================================================

def create_or_update_vector_store(embeddings_data, local_path, drive, parent_folder_id):
    """
    Create or update vectorstore with new embeddings.
    
    Args:
        embeddings_data: List of embedding data
        local_path: Local path for vectorstore
        drive: Google Drive object
        parent_folder_id: Drive folder ID for vectorstore
    
    Returns:
        str: Path to local vectorstore
    """
    os.makedirs(local_path, exist_ok=True)
    
    print(f"\n💾 Working in local folder: {local_path}")
    
    # Initialize Chroma collection
    client_chroma = PersistentClient(path=local_path)
    
    try:
        collection = client_chroma.get_collection("video_embeddings")
        print("🧩 Existing local Chroma collection found — appending new embeddings...")
    except:
        collection = client_chroma.create_collection("video_embeddings")
        print("🆕 Created new local Chroma collection.")
    
    # Add new embeddings
    for item in embeddings_data:
        collection.add(
            ids=[f"{item['video_id']}_{item['segment_index']}"],
            embeddings=[item["embedding"]],
            metadatas=[{
                "video_id": item["video_id"],
                "video_url": item["video_url"],
                "title": item["title"],
                "segment_index": item["segment_index"],
                "start_time": item["start_time"],
                "end_time": item["end_time"]
            }]
        )
    
    print(f"✅ Appended {len(embeddings_data)} new embeddings to local collection.")
    
    # Upload to Drive
    print("📤 Uploading vectorstore to Drive...")
    try:
        # Check if chroma folder already exists in Drive
        exists_in_drive, chroma_folder_id = check_vectorstore_exists_in_drive(drive, parent_folder_id)
        
        if exists_in_drive:
            # Delete old version
            print("🗑️ Removing old vectorstore from Drive...")
            drive.CreateFile({'id': chroma_folder_id}).Delete()
        
        # Create new chroma folder in Drive
        chroma_folder_metadata = {
            'title': 'chroma_video_embeddings_db',
            'parents': [{'id': parent_folder_id}],
            'mimeType': 'application/vnd.google-apps.folder'
        }
        chroma_folder = drive.CreateFile(chroma_folder_metadata)
        chroma_folder.Upload()
        new_chroma_folder_id = chroma_folder['id']
        
        # Upload local folder contents
        upload_folder_to_drive(local_path, new_chroma_folder_id, drive)
        print("✅ Upload complete!")
        
    except Exception as e:
        print(f"⚠️ Drive upload failed: {e}")
        print(f"Vectorstore is available locally at: {local_path}")
    
    return local_path


# ============================================================
# MAIN PROCESSING FUNCTION
# ============================================================

def create_video_embeddings(
    sheet,
    drive,
    parent_folder_id=PARENT_FOLDER_ID,
    sheet_name=SHEET_TAB,
    chunk_length=CHUNK_LENGTH_SEC,
    max_workers=MAX_WORKERS
):
    """
    Create video embeddings from YouTube videos with incremental updates.
    
    This function:
    1. Creates/loads vectorstore from local/Drive
    2. Reads sheet and adds tracking columns
    3. Filters videos that need processing
    4. Processes videos in parallel with progress tracking
    5. Uploads to Drive
    
    Args:
        sheet: gspread sheet object
        drive: PyDrive GoogleDrive object
        parent_folder_id: Google Drive folder ID where vectorstore will be stored
        sheet_name: Name of the worksheet to read from
        chunk_length: Length of video chunks in seconds
        max_workers: Number of parallel workers for embedding generation
    
    Returns:
        tuple: (vectorstore_path, stats)
            vectorstore_path: Path to local vectorstore
            stats: Dictionary with processing statistics
    """
    
    print("=" * 80)
    print("🚀 VIDEO EMBEDDINGS CREATION PIPELINE")
    print("=" * 80)
    
    start_time = time.time()
    
    # ============================================================
    # STEP 1: Setup Drive folders
    # ============================================================
    print("\n📁 Step 1: Setting up Google Drive folders...")
    
    vectorstore_folder_id = create_or_get_drive_folder(
        drive, 
        parent_folder_id, 
        VECTORSTORE_FOLDER_NAME
    )
    
    # ============================================================
    # STEP 2: Setup local paths and Vertex AI model
    # ============================================================
    print("\n💾 Step 2: Setting up local storage and Vertex AI model...")
    
    os.makedirs(LOCAL_VECTORSTORE_PATH, exist_ok=True)
    print(f"✅ Local storage path: {LOCAL_VECTORSTORE_PATH}")
    
    model = initialize_vertex_ai()
    
    # ============================================================
    # STEP 3: Check for existing vectorstore and download if exists
    # ============================================================
    print("\n🔍 Step 3: Checking for existing vectorstore...")
    
    exists_in_drive, chroma_folder_id = check_vectorstore_exists_in_drive(
        drive, 
        vectorstore_folder_id
    )
    
    # Check if local vectorstore already exists
    local_exists = os.path.exists(os.path.join(LOCAL_VECTORSTORE_PATH, "chroma.sqlite3"))
    
    if local_exists:
        print(f"✅ Found existing local vectorstore at: {LOCAL_VECTORSTORE_PATH}")
        print("📌 Using local copy (preserves any unsaved progress)")
    elif exists_in_drive:
        print("📥 No local copy found. Downloading from Drive...")
        try:
            download_folder_from_drive(chroma_folder_id, LOCAL_VECTORSTORE_PATH, drive)
            print("✅ Downloaded successfully")
        except Exception as e:
            print(f"⚠️ Download failed: {e}")
            print("Will create fresh vectorstore")
    else:
        print("📝 No existing vectorstore found (local or Drive). Will create new one.")
    
    # ============================================================
    # STEP 4: Load sheet data and add tracking columns
    # ============================================================
    print("\n📊 Step 4: Loading sheet data...")
    
    worksheet, df = get_sheet_data_and_df(sheet, sheet_name)
    print(f"✅ Loaded {len(df)} rows from '{sheet_name}'")
    
    # Add tracking columns if missing
    for col in TRACKING_COLUMNS:
        if col not in df.columns:
            df[col] = ''
            print(f"✅ Added '{col}' column")
    
    # ============================================================
    # STEP 5: Validate required columns
    # ============================================================
    print("\n🔍 Step 5: Validating required columns...")
    
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise Exception(f"❌ Missing required columns: {missing_columns}")
    
    print(f"✅ All required columns found")
    
    # ============================================================
    # STEP 6: Filter videos that need processing
    # ============================================================
    print("\n🧹 Step 6: Filtering videos to process...")
    
    videos_to_process = df[
        (df['vectorized'] != 'TRUE') &  # Not already done
        (df['vectorized'] != 'FAILED') &  # Not failed (skip failed videos)
        (df['video_url'].notna()) &     # Has URL
        (df['video_url'].astype(str).str.strip() != '')
    ]
    
    already_processed = len(df[df['vectorized'] == 'TRUE'])
    failed_videos = len(df[df['vectorized'] == 'FAILED'])
    
    print(f"📊 Total videos: {len(df)}")
    print(f"✅ Already processed: {already_processed}")
    print(f"❌ Failed (skipped): {failed_videos}")
    print(f"⏳ Remaining to process: {len(videos_to_process)}")
    
    if len(videos_to_process) == 0:
        print("\n🎉 All videos are already processed! Nothing to do.")
        
        # Get collection count
        try:
            client_chroma = PersistentClient(path=LOCAL_VECTORSTORE_PATH)
            collection = client_chroma.get_collection("video_embeddings")
            collection_count = collection.count()
        except:
            collection_count = already_processed
        
        stats = {
            'total_videos': len(df),
            'already_processed': already_processed,
            'newly_processed': 0,
            'total_processed': already_processed,
            'duration': '0s',
            'drive_folder_id': vectorstore_folder_id,
            'collection_count': collection_count
        }
        
        return LOCAL_VECTORSTORE_PATH, stats
    
    # ============================================================
    # STEP 7: Process videos
    # ============================================================
    print(f"\n⚡ Step 7: Processing {len(videos_to_process)} videos...")
    print("=" * 80)
    
    # Initialize SmartProgressBar for Streamlit UI
    progress = SmartProgressBar(
        total_tasks=len(videos_to_process),
        description="Processing videos",
        save_interval=1
    )
    
    total_tokens = 0
    total_cost = 0.0
    total_segments = 0
    
    # Use tqdm for console/log progress
    for video_idx, (idx, row) in enumerate(tqdm(videos_to_process.iterrows(), total=len(videos_to_process), desc="Processing videos")):
        try:
            video_url = str(row["video_url"]).strip()
            video_id = str(row["video_id"]).strip()
            title = str(row.get("title", "")).strip()
            
            print(f"\n🎥 Processing video {video_idx+1}/{len(videos_to_process)}: {video_url}")
            
            # Create temporary directory for this video
            temp_dir = tempfile.mkdtemp()
            
            try:
                # 1. Download video
                video_path = download_video(video_url, temp_dir)
                
                # 2. Segment video
                segments = segment_video(video_path, temp_dir, chunk_length)
                total_segments += len(segments)
                
                # 3. Generate embeddings (parallel processing)
                current_video_embeddings = []
                print(f"🔄 Generating embeddings for {len(segments)} segments in parallel...")
                
                # Thread-safe lock for appending to lists
                lock = threading.Lock()
                
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # Submit all segments for processing with original timestamps
                    future_to_segment = {
                        executor.submit(generate_video_embedding, seg_path, model): (j, seg_path, j * chunk_length)
                        for j, seg_path in enumerate(segments)
                    }
                    
                    # Process results as they complete
                    for future in as_completed(future_to_segment):
                        j, seg_path, original_start_time = future_to_segment[future]
                        try:
                            print(f"  🎬 Completed segment {j+1}/{len(segments)}")
                            emb_list, tokens, cost = future.result()
                            total_tokens += tokens
                            total_cost += cost
                            
                            # Thread-safe list operations
                            with lock:
                                for e in emb_list:
                                    embedding_item = {
                                        "video_id": video_id,
                                        "video_url": video_url,
                                        "title": title,
                                        "segment_index": j,
                                        "embedding": e["vector"],
                                        "start_time": original_start_time + e["start"],  # Original video timestamp
                                        "end_time": original_start_time + e["end"]        # Original video timestamp
                                    }
                                    current_video_embeddings.append(embedding_item)
                        except Exception as e:
                            print(f"  ❌ Failed to process segment {j+1}: {e}")
                            # Continue with other segments instead of stopping
                            continue
                
                # 4. Update vectorstore
                print(f"📦 Updating vectorstore with {len(current_video_embeddings)} embeddings...")
                create_or_update_vector_store(current_video_embeddings, LOCAL_VECTORSTORE_PATH, drive, vectorstore_folder_id)
                
                # 5. Update sheet
                df.at[idx, 'vectorized'] = 'TRUE'
                df.at[idx, 'embedding_ts'] = datetime.now().isoformat()
                df.at[idx, 'segments_processed'] = len(segments)
                
                # Update progress bar (Streamlit UI)
                progress.update()
                
                # Save progress to sheet after every successful video
                try:
                    save_to_sheet(worksheet, df)
                    print(f"💾 Sheet saved (video {video_idx+1} marked as TRUE)")
                except Exception as e:
                    print(f"\n⚠️ Sheet save failed: {e}. Continuing...")
            
            finally:
                # Cleanup temporary directory
                shutil.rmtree(temp_dir, ignore_errors=True)
        
        except Exception as e:
            print(f"\n❌ Error processing video {video_idx+1}: {e}")
            # Skip failed video and continue with others
            print(f"⏭️ Skipping video {video_idx+1} and continuing...")
            df.at[idx, 'vectorized'] = 'FAILED'
            df.at[idx, 'embedding_ts'] = datetime.now().isoformat()
            
            # Save progress to sheet after every failed video
            try:
                save_to_sheet(worksheet, df)
                print(f"💾 Sheet saved (video {video_idx+1} marked as FAILED)")
            except Exception as e:
                print(f"\n⚠️ Sheet save failed: {e}. Continuing...")
            
            continue
    
    # ============================================================
    # STEP 8: Final save
    # ============================================================
    print("\n💾 Step 8: Performing final save...")
    
    # Final sheet save
    try:
        save_to_sheet(worksheet, df)
        print("✅ Sheet saved successfully")
    except Exception as e:
        print(f"⚠️ Final sheet save failed: {e}")
    
    # ============================================================
    # STEP 9: Calculate and return stats
    # ============================================================
    elapsed_time = time.time() - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    duration_str = f"{minutes}m {seconds}s" if minutes > 0 else f"{seconds}s"
    
    try:
        client_chroma = PersistentClient(path=LOCAL_VECTORSTORE_PATH)
        collection = client_chroma.get_collection("video_embeddings")
        collection_count = collection.count()
    except:
        collection_count = len(df[df['vectorized'] == 'TRUE'])
    
    stats = {
        'total_videos': len(df),
        'already_processed': already_processed,
        'newly_processed': len(videos_to_process),
        'total_processed': len(df[df['vectorized'] == 'TRUE']),
        'total_segments': total_segments,
        'total_tokens': total_tokens,
        'total_cost': total_cost,
        'duration': duration_str,
        'drive_folder_id': vectorstore_folder_id,
        'collection_count': collection_count
    }
    
    print("\n" + "=" * 80)
    print("🎉 VIDEO EMBEDDINGS CREATION COMPLETE!")
    print("=" * 80)
    print(f"📊 Statistics:")
    print(f"   • Total videos in sheet: {stats['total_videos']}")
    print(f"   • Already processed: {stats['already_processed']}")
    print(f"   • Newly processed: {stats['newly_processed']}")
    print(f"   • Total processed: {stats['total_processed']}")
    print(f"   • Total segments: {stats['total_segments']}")
    print(f"   • Total tokens: {stats['total_tokens']}")
    print(f"   • Total cost: ${stats['total_cost']}")
    print(f"   • Collection count: {stats['collection_count']}")
    print(f"   • Duration: {stats['duration']}")
    print(f"   • Drive folder ID: {stats['drive_folder_id']}")
    print("=" * 80)
    
    return LOCAL_VECTORSTORE_PATH, stats


def check_video_embeddings_status(sheet, drive, parent_folder_id=PARENT_FOLDER_ID, sheet_name=SHEET_TAB):
    """
    Check current video processing progress.
    
    Args:
        sheet: gspread sheet object
        drive: PyDrive GoogleDrive object
        parent_folder_id: Google Drive folder ID
        sheet_name: Name of the worksheet
    
    Returns:
        dict: Status information
    """
    print("🔍 Checking video embeddings status...")
    
    # Load sheet
    worksheet, df = get_sheet_data_and_df(sheet, sheet_name)
    
    # Check columns
    has_vectorized_col = 'vectorized' in df.columns
    has_ts_col = 'embedding_ts' in df.columns
    has_segments_col = 'segments_processed' in df.columns
    
    if not has_vectorized_col:
        total_processed = 0
        last_updated = None
    else:
        total_processed = len(df[df['vectorized'] == 'TRUE'])
        
        if has_ts_col and total_processed > 0:
            # Get last timestamp
            timestamps = df[df['embedding_ts'] != '']['embedding_ts'].tolist()
            last_updated = max(timestamps) if timestamps else None
        else:
            last_updated = None
    
    total_videos = len(df)
    remaining_videos = total_videos - total_processed
    progress_percentage = (total_processed / total_videos * 100) if total_videos > 0 else 0
    
    # Check Drive
    vectorstore_folder_list = drive.ListFile({
        'q': f"title='{VECTORSTORE_FOLDER_NAME}' and '{parent_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
    }).GetList()
    
    vectorstore_exists_in_drive = False
    if vectorstore_folder_list:
        vectorstore_folder_id = vectorstore_folder_list[0]['id']
        exists, _ = check_vectorstore_exists_in_drive(drive, vectorstore_folder_id)
        vectorstore_exists_in_drive = exists
    
    status = {
        'total_videos': total_videos,
        'processed_videos': total_processed,
        'remaining_videos': remaining_videos,
        'progress_percentage': round(progress_percentage, 2),
        'vectorstore_exists_in_drive': vectorstore_exists_in_drive,
        'last_updated': last_updated,
        'has_tracking_columns': has_vectorized_col and has_ts_col and has_segments_col
    }
    
    print(f"✅ Status checked:")
    print(f"   • Total videos: {status['total_videos']}")
    print(f"   • Processed: {status['processed_videos']}")
    print(f"   • Remaining: {status['remaining_videos']}")
    print(f"   • Progress: {status['progress_percentage']}%")
    print(f"   • Drive backup: {'✅ Exists' if status['vectorstore_exists_in_drive'] else '❌ Not Found'}")
    
    return status