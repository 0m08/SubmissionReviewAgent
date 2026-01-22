"""
Video Embeddings Streamlit UI
=============================

Streamlit UI for creating and managing video embeddings from YouTube videos.
This pipeline creates embeddings from video segments using Vertex AI's MultiModal Embedding Model.

This pipeline processes videos from the "3D Animations and Simulations" sheet
and creates a searchable vector database for video content analysis.
"""

import streamlit as st
from create_video_embeddings import (
    create_video_embeddings,
    check_video_embeddings_status
)

# ============================================================
# Authentication Check
# ============================================================
if "drive" in st.session_state and "gc" in st.session_state:
    drive = st.session_state["drive"]
    gc = st.session_state["gc"]
else:
    st.error("❌ Authentication not found. Please log out and log in again.")
    st.stop()

# ============================================================
# UI Header
# ============================================================
st.markdown("## 🎥 Video Embeddings Creation")

# Hardcoded configuration display
st.info(f"""
*📊 Sheet:* [3D Animations and Simulations](https://docs.google.com/spreadsheets/d/1p9G7SeBRzfl1KGcEzqzF6zG5TO7B__kuSVlY3tQhB2U/edit?gid=687427813#gid=687427813)

*📁 Drive Folder:* 1SSvxx1EJ3zgMPfvD8Zy5DaEgmI2prys7

*⚙️ Configuration:*
- Video chunks: 30 seconds
- Parallel workers: 5
- Embedding model: Vertex AI MultiModal (multimodalembedding@001)
""")

st.markdown("""
Create video embeddings from YouTube videos for semantic search and content analysis.

*What this does:*
- Reads "3D Animations and Simulations" sheet with video URLs
- Downloads YouTube videos and segments them into 30-second chunks
- Generates embeddings using Vertex AI's MultiModal Embedding Model
- Stores in local ChromaDB + Google Drive backup
- Tracks progress in sheet with 'vectorized', 'embedding_ts', and 'segments_processed' columns
- Supports resume capability (skips already-processed videos)
""")

st.markdown("---")

# ============================================================
# Load Sheet
# ============================================================
st.markdown("### 📋 Load Google Sheet")

# Hardcoded sheet URL
sheet_url = "https://docs.google.com/spreadsheets/d/1p9G7SeBRzfl1KGcEzqzF6zG5TO7B__kuSVlY3tQhB2U/edit?gid=687427813#gid=687427813"
# sheet_tab = "3D Animations and Simulations"  # Old value - commented out
sheet_tab = "3D, Hands-On,Equipment Demos"

if st.button("📥 Load Sheet", type="primary"):
    try:
        sheet = gc.open_by_url(sheet_url)
        st.session_state["video_sheet"] = sheet
        st.success(f"✅ Sheet loaded: *{sheet.title}*")
        st.rerun()
    except Exception as e:
        st.error(f"❌ Failed to load sheet: {str(e)}")

# Check if sheet is loaded
if "video_sheet" not in st.session_state:
    st.warning("⚠️ Please load the sheet first")
    st.stop()

sheet = st.session_state["video_sheet"]

# ============================================================
# Configuration Display
# ============================================================
st.markdown("### ⚙️ Configuration")

# parent_folder_id = "1SSvxx1EJ3zgMPfvD8Zy5DaEgmI2prys7"  # Old value - commented out
parent_folder_id = "15H9thXq02JX3ldADSj1oD78mbV-fXfvu"
# sheet_name = "3D Animations and Simulations"  # Old value - commented out
sheet_name = "3D, Hands-On,Equipment Demos"

col1, col2 = st.columns(2)

with col1:
    st.info(f"📁 *Drive Folder ID:* {parent_folder_id}")
    st.info(f"📊 *Sheet Name:* {sheet_name}")

with col2:
    st.info(f"🎬 *Video Chunks:* 30 seconds")
    st.info(f"⚡ *Parallel Workers:* 5")

st.markdown("---")

# ============================================================
# Status Check
# ============================================================
st.markdown("### 📊 Current Status")

if st.button("🔍 Check Video Embeddings Status", use_container_width=True):
    with st.spinner("Checking status..."):
        try:
            status = check_video_embeddings_status(
                sheet=sheet,
                drive=drive,
                parent_folder_id=parent_folder_id,
                sheet_name=sheet_name
            )
            
            # Display status in metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Videos", status['total_videos'])
            with col2:
                st.metric("Processed", status['processed_videos'])
            with col3:
                st.metric("Remaining", status['remaining_videos'])
            with col4:
                st.metric("Progress", f"{status['progress_percentage']}%")
            
            # Additional info
            st.info(f"""
            *Drive Backup:* {'✅ Exists' if status['vectorstore_exists_in_drive'] else '❌ Not Found'}  
            *Last Updated:* {status['last_updated'] if status['last_updated'] else 'Never'}  
            *Tracking Columns:* {'✅ Present' if status['has_tracking_columns'] else '❌ Missing'}
            """)
            
            # Show full status
            with st.expander("📋 View Full Status Details"):
                st.json(status)
                
        except Exception as e:
            st.error(f"❌ Error checking status: {str(e)}")

st.markdown("---")

# ============================================================
# Create/Update Video Embeddings
# ============================================================
st.markdown("### 🚀 Create/Update Video Embeddings")

st.info("""
*What will happen:*
1. ✅ Check for existing vectorstore in Drive (downloads if exists)
2. ✅ Load sheet and add tracking columns if missing
3. ✅ Skip videos that are already processed
4. ✅ Process remaining videos one by one
5. ✅ Download, segment, and generate embeddings for each video
6. ✅ Update vectorstore after each video (local + Drive)
7. ✅ Save progress to sheet after each video

*Time estimate:* 
- First run (all videos): Depends on number of videos and their length
- Resume run (few videos): ~30 seconds - 10 minutes per video
""")

# Warning about Vertex AI credentials
st.warning("""
*⚠️ Prerequisites:*
- Vertex AI credentials must be configured
- Ensure you have access to the 'dam-images-tagging' project
- Make sure ffmpeg is installed for video processing
""")

if st.button("🚀 Start Video Embeddings Creation", type="primary", use_container_width=True):
    with st.spinner("Creating video embeddings... This may take a while..."):
        try:
            # Create progress placeholder
            progress_placeholder = st.empty()
            stats_placeholder = st.empty()
            
            progress_placeholder.info("🔄 Starting video embeddings creation...")
            
            # Run creation
            vectorstore_path, stats = create_video_embeddings(
                sheet=sheet,
                drive=drive,
                parent_folder_id=parent_folder_id,
                sheet_name=sheet_name,
                chunk_length=30,
                max_workers=5
            )
            
            progress_placeholder.empty()
            
            # Show success message
            st.success("✅ Video embeddings created successfully!")
            
            # Display statistics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Videos", stats['total_videos'])
                st.metric("Already Processed", stats['already_processed'])
            with col2:
                st.metric("Newly Processed", stats['newly_processed'])
                st.metric("Total Processed", stats['total_processed'])
            with col3:
                st.metric("Total Segments", stats['total_segments'])
                st.metric("Duration", stats['duration'])
            
            # Cost and technical metrics
            col4, col5, col6 = st.columns(3)
            
            with col4:
                st.metric("Total Tokens", f"{stats['total_tokens']:,}")
            with col5:
                st.metric("Total Cost", f"${stats['total_cost']:.4f}")
            with col6:
                st.metric("Collection Count", stats['collection_count'])
            
            # Show full stats
            with st.expander("📊 View Full Statistics"):
                st.json(stats)
            
            # Show vectorstore location
            st.info(f"""
            *📁 Vectorstore Location:*
            - Local: {vectorstore_path}
            - Drive Folder ID: {stats['drive_folder_id']}
            """)
            
            st.balloons()
            
        except Exception as e:
            st.error(f"❌ Error during video embeddings creation: {str(e)}")
            st.exception(e)

st.markdown("---")

# ============================================================
# Help & Documentation
# ============================================================
with st.expander("ℹ️ Help & Documentation"):
    st.markdown("""
    ### What are video embeddings?
    
    Video embeddings are numerical representations of video content that capture semantic meaning.
    This allows for similarity search across video content, even when the exact words or visuals differ.
    
    ### How does it work?
    
    1. *Video Download:* Downloads YouTube videos using yt-dlp
    2. *Segmentation:* Splits videos into 30-second chunks for processing
    3. *Embedding Generation:* Uses Vertex AI's MultiModal Embedding Model to create embeddings
    4. *Storage:* Stores embeddings in ChromaDB with metadata (video_id, URL, timestamps, etc.)
    5. *Search:* Enables semantic search across all processed video content
    
    ### What gets embedded?
    
    - *Video Content:* Visual and audio information from each 30-second segment
    - *Metadata:* Video ID, URL, title, segment timestamps (stored alongside, not embedded)
    
    ### Processing Details
    
    - *Chunk Size:* 30 seconds per segment
    - *Parallel Processing:* 5 segments processed simultaneously
    - *Compression:* Videos are automatically compressed if they exceed size limits
    - *Resume Capability:* Can stop and restart without losing progress
    
    ### Troubleshooting
    
    *Problem:* "Vertex AI credentials not found"  
    *Solution:* Ensure Vertex AI service account credentials are properly configured
    
    *Problem:* "Download failed"  
    *Solution:* Check video URL accessibility and internet connection
    
    *Problem:* "FFmpeg not found"  
    *Solution:* Install ffmpeg: apt-get install ffmpeg or brew install ffmpeg
    
    *Problem:* Process is slow  
    *Solution:* This is normal! Video processing is computationally intensive. 
    Progress is saved after each video, so you can stop and resume anytime.
    
    ### Cost Information
    
    - *Pricing:* ~$0.30 per million tokens
    - *Token Usage:* Approximately 50,000 tokens per 30-second segment
    - *Cost Tracking:* Real-time cost estimation and tracking
    
    ### Need Help?
    
    Contact the Graphics Definition team for assistance.
    """)

st.markdown("---")

# ============================================================
# Footer
# ============================================================
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.8em; margin-top: 2rem;'>
    Video Embeddings Creation Pipeline | Graphics Definition Team
</div>
""", unsafe_allow_html=True)