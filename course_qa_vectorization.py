"""
Course Q&A App - Slide Chunks Vectorization
Single file containing both UI and vectorization logic
"""

import streamlit as st
import os
import re
import pandas as pd
from typing import Optional, Tuple
from tqdm import tqdm
from datetime import datetime
from langchain_chroma import Chroma
from langchain.schema import Document
from services.embedding_service import get_embedding_model
from services.sheets_service import get_sheet_data_and_df, save_to_sheet
from services.drive_service import download_folder_from_drive, upload_folder_to_drive
from pydrive2.drive import GoogleDrive
from pydrive2.auth import GoogleAuth
import gspread
from google.oauth2.credentials import Credentials
from services.drive_service import init_clients_from_credentials

# ============================================================================
# Configuration
# ============================================================================

ROOT_FOLDER_ID = "1UkZGl3HeCFq48zEl3HTkKdvty91gUse_"
VECTORSTORE_FOLDER_NAME = "Vectorstore files"
CHROMA_FOLDER_NAME = "chroma_slide_chunks_db"
COLLECTION_NAME = "slide_chunks"
LOCAL_CHROMA_ROOT = "/tmp/temp_chroma_folder"

# ============================================================================
# Helper Functions
# ============================================================================

def extract_sheet_id(sheet_link: str) -> str:
    """
    Extract Google Sheet ID from a link or return as-is if already an ID.
    
    Supports:
    - https://docs.google.com/spreadsheets/d/SHEET_ID/edit
    - https://docs.google.com/spreadsheets/d/SHEET_ID/edit#gid=0
    - SHEET_ID (direct ID)
    """
    if not sheet_link:
        raise ValueError("Sheet link cannot be empty")
    
    # Try to extract from URL
    match = re.search(r'/spreadsheets/d/([a-zA-Z0-9_-]+)', sheet_link)
    if match:
        return match.group(1)
    
    # If no match, assume it's already an ID
    if re.match(r'^[a-zA-Z0-9_-]+$', sheet_link):
        return sheet_link
    
    raise ValueError(f"Could not extract sheet ID from: {sheet_link}")


def get_course_name_from_sheet(sheet, worksheet_name: str = "Course info") -> str:
    """Extract course name from Course info tab."""
    try:
        _, course_info_df = get_sheet_data_and_df(sheet, worksheet_name)
        if 'Course Name' not in course_info_df.columns:
            raise ValueError(f"'Course Name' column not found in '{worksheet_name}' tab")
        
        course_name = course_info_df['Course Name'].iloc[0] if len(course_info_df) > 0 else None
        if not course_name or pd.isna(course_name):
            raise ValueError(f"Course Name is empty in '{worksheet_name}' tab")
        
        return str(course_name).strip()
    except Exception as e:
        raise ValueError(f"Failed to get course name: {e}")


def ensure_vectorized_column(sheet, worksheet, df: pd.DataFrame) -> pd.DataFrame:
    """Ensure 'vectorized' column exists in both DataFrame and Sheet."""
    if 'vectorized' not in df.columns:
        df['vectorized'] = 'FALSE'
        # Add column to sheet header
        try:
            headers = worksheet.row_values(1)
            if 'vectorized' not in headers:
                headers.append('vectorized')
                worksheet.update('A1', [headers])
        except Exception as e:
            st.warning(f"Could not update sheet header: {e}")
    else:
        # Ensure all values are strings and handle NaN
        df['vectorized'] = df['vectorized'].fillna('FALSE').astype(str).str.upper()
    
    return df


def initialize_vectorstore(drive, root_folder_id: str) -> Tuple[Chroma, str, str, str]:
    """
    Initialize or load existing vectorstore from Drive.
    Returns: (chroma_db, local_path, chroma_folder_id, vectorstore_folder_id)
    """
    embedding_function = get_embedding_model()
    
    # Find or create 'Vectorstore files' folder
    vectorstore_query = (
        f"title='{VECTORSTORE_FOLDER_NAME}' and '{root_folder_id}' in parents "
        f"and mimeType='application/vnd.google-apps.folder' and trashed=false"
    )
    vectorstore_folders = drive.ListFile({'q': vectorstore_query}).GetList()
    
    if vectorstore_folders:
        vectorstore_folder_id = vectorstore_folders[0]['id']
    else:
        # Create 'Vectorstore files' folder
        folder_metadata = {
            'title': VECTORSTORE_FOLDER_NAME,
            'parents': [{'id': root_folder_id}],
            'mimeType': 'application/vnd.google-apps.folder'
        }
        vectorstore_folder = drive.CreateFile(folder_metadata)
        vectorstore_folder.Upload()
        vectorstore_folder_id = vectorstore_folder['id']
    
    # Find or create chroma folder
    chroma_query = (
        f"title='{CHROMA_FOLDER_NAME}' and '{vectorstore_folder_id}' in parents "
        f"and mimeType='application/vnd.google-apps.folder' and trashed=false"
    )
    chroma_folders = drive.ListFile({'q': chroma_query}).GetList()
    
    local_path = os.path.join(LOCAL_CHROMA_ROOT, CHROMA_FOLDER_NAME)
    os.makedirs(LOCAL_CHROMA_ROOT, exist_ok=True)
    
    if chroma_folders:
        # Download existing vectorstore
        chroma_folder_id = chroma_folders[0]['id']
        sqlite_path = os.path.join(local_path, "chroma.sqlite3")
        
        if not os.path.exists(sqlite_path):
            st.info("Downloading existing vectorstore from Drive...")
            download_folder_from_drive(chroma_folder_id, local_path, drive)
        
        chroma_db = Chroma(
            embedding_function=embedding_function,
            collection_name=COLLECTION_NAME,
            persist_directory=local_path
        )
    else:
        # Create new vectorstore
        chroma_db = Chroma(
            embedding_function=embedding_function,
            collection_name=COLLECTION_NAME,
            persist_directory=local_path
        )
        
        # Create folder in Drive
        folder_metadata = {
            'title': CHROMA_FOLDER_NAME,
            'parents': [{'id': vectorstore_folder_id}],
            'mimeType': 'application/vnd.google-apps.folder'
        }
        chroma_folder = drive.CreateFile(folder_metadata)
        chroma_folder.Upload()
        chroma_folder_id = chroma_folder['id']
    
    # Get vectorstore_folder_id to return it for later use
    if not vectorstore_folders:
        # This shouldn't happen, but just in case
        vectorstore_query = (
            f"title='{VECTORSTORE_FOLDER_NAME}' and '{root_folder_id}' in parents "
            f"and mimeType='application/vnd.google-apps.folder' and trashed=false"
        )
        vectorstore_folders = drive.ListFile({'q': vectorstore_query}).GetList()
        if vectorstore_folders:
            vectorstore_folder_id = vectorstore_folders[0]['id']
        else:
            raise ValueError("Could not find Vectorstore files folder")
    else:
        vectorstore_folder_id = vectorstore_folders[0]['id']
    
    return chroma_db, local_path, chroma_folder_id, vectorstore_folder_id


def vectorize_slide_chunks(
    sheet,
    slide_chunks_ws,
    slide_chunks_df: pd.DataFrame,
    course_name: str,
    sheet_id: str,
    chroma_db: Chroma,
    progress_bar
) -> Tuple[int, int]:
    """
    Vectorize all unvectorized slide chunks.
    Returns: (success_count, error_count)
    """
    # Filter unvectorized rows
    unvectorized_df = slide_chunks_df[
        slide_chunks_df['vectorized'].str.upper() != 'TRUE'
    ].copy()
    
    total_rows = len(unvectorized_df)
    if total_rows == 0:
        return 0, 0
    
    success_count = 0
    error_count = 0
    
    required_columns = ['Topic', 'Subtopic', 'Slide Type', 'Slide Chunk Title', 'Slide Chunk']
    missing_cols = [col for col in required_columns if col not in slide_chunks_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    print(f"\n🚀 Starting vectorization for course: {course_name}")
    print(f"📊 Processing {total_rows} slides...")
    
    # Batch update settings
    BATCH_SIZE = 20  # Update sheet every 20 slides to avoid quota limits
    
    for idx, row in unvectorized_df.iterrows():
        try:
            # Get slide title for logging
            slide_title = str(row.get('Slide Chunk Title', 'Untitled')).strip()
            
            # Get slide content
            slide_content = str(row['Slide Chunk']).strip()
            
            if not slide_content or slide_content.lower() in ['nan', 'none', '']:
                slide_chunks_df.at[idx, 'vectorized'] = 'ERROR'
                if 'error_message' not in slide_chunks_df.columns:
                    slide_chunks_df['error_message'] = ''
                slide_chunks_df.at[idx, 'error_message'] = 'Empty slide content'
                error_count += 1
                continue
            
            # Log starting vectorization
            print(f"  🔄 Starting vectorization for slide: {slide_title}")
            
            # Build metadata
            metadata = {
                'course_name': course_name,
                'topic': str(row.get('Topic', '')).strip(),
                'subtopic': str(row.get('Subtopic', '')).strip(),
                'slide_type': str(row.get('Slide Type', '')).strip(),
                'slide_chunk_title': slide_title,
                'sheet_id': sheet_id,
                'row_index': int(idx) + 2,  # +2 for 0-indexed + header row
            }
            
            # Create unique ID
            unique_id = f"{sheet_id}_{idx}"
            
            # Add to vectorstore
            chroma_db.add_texts(
                texts=[slide_content],
                metadatas=[metadata],
                ids=[unique_id]
            )
            
            # Update DataFrame
            slide_chunks_df.at[idx, 'vectorized'] = 'TRUE'
            success_count += 1
            
            # Log successful vectorization
            print(f"  ✓ Vectorization successful for slide: {slide_title}")
            
            # Print progress every 10 slides
            if success_count % 10 == 0:
                print(f"  📊 Progress: {success_count}/{total_rows} slides processed...")
            
            # Batch update sheet every BATCH_SIZE slides to avoid quota limits
            if success_count % BATCH_SIZE == 0:
                try:
                    print(f"  💾 Saving progress to sheet (batch update)...")
                    save_to_sheet(slide_chunks_ws, slide_chunks_df)
                except Exception as e:
                    st.warning(f"Could not batch update sheet: {e}")
            
            # Update progress bar
            if progress_bar:
                progress_bar.progress((success_count + error_count) / total_rows)
            
        except Exception as e:
            error_msg = str(e)
            slide_chunks_df.at[idx, 'vectorized'] = 'ERROR'
            if 'error_message' not in slide_chunks_df.columns:
                slide_chunks_df['error_message'] = ''
            slide_chunks_df.at[idx, 'error_message'] = error_msg
            error_count += 1
            
            print(f"  ❌ Error on row {idx + 2}: {error_msg}")
            
            # Update progress bar
            if progress_bar:
                progress_bar.progress((success_count + error_count) / total_rows)
            
            st.warning(f"Error vectorizing row {idx + 2}: {error_msg}")
            continue
    
    # Final save of dataframe
    try:
        save_to_sheet(slide_chunks_ws, slide_chunks_df)
    except Exception as e:
        st.error(f"Failed to save final updates to sheet: {e}")
    
    print(f"✅ Vectorization complete: {success_count} successful, {error_count} errors")
    
    return success_count, error_count


# ============================================================================
# Streamlit UI
# ============================================================================

def main():
    st.set_page_config(
        page_title="Course Q&A - Vectorization",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("📚 Course Q&A App - Slide Chunks Vectorization")
    
    # Check authentication
    if 'drive' not in st.session_state or 'gc' not in st.session_state:
        st.error("❌ Please authenticate first by logging in through the main app")
        return
    
    drive = st.session_state.drive
    gc = st.session_state.gc
    
    # Main UI - Two Sections
    tab1, tab2 = st.tabs(["🔧 Vectorization", "❓ Q&A (Coming Soon)"])
    
    with tab1:
        st.header("Vectorize Slide Chunks")
        st.markdown("""
        **Instructions:**
        1. Paste the Google Sheet link containing course slides
        2. Click "Vectorize" to process all unvectorized slides
        3. Progress will be shown in real-time
        """)
        
        sheet_link = st.text_input(
            "Google Sheet Link",
            placeholder="https://docs.google.com/spreadsheets/d/SHEET_ID/edit",
            help="Paste the full Google Sheets URL or just the Sheet ID"
        )
        
        if st.button("🚀 Vectorize", type="primary"):
            if not sheet_link:
                st.error("Please provide a Google Sheet link")
                return
            
            try:
                # Extract sheet ID
                sheet_id = extract_sheet_id(sheet_link)
                print(f"\n📄 Processing sheet: {sheet_id}")
                st.info(f"Processing sheet: {sheet_id}")
                
                # Open sheet
                sheet = gc.open_by_key(sheet_id)
                
                # Get course name
                with st.spinner("Reading course information..."):
                    course_name = get_course_name_from_sheet(sheet)
                    st.success(f"Course: **{course_name}**")
                
                # Read Slide Chunks tab
                with st.spinner("Reading slide chunks..."):
                    slide_chunks_ws, slide_chunks_df = get_sheet_data_and_df(sheet, "Slide Chunks")
                    
                    # Ensure vectorized column exists
                    slide_chunks_df = ensure_vectorized_column(sheet, slide_chunks_ws, slide_chunks_df)
                    
                    total_slides = len(slide_chunks_df)
                    vectorized_count = len(slide_chunks_df[slide_chunks_df['vectorized'].str.upper() == 'TRUE'])
                    unvectorized_count = total_slides - vectorized_count
                    
                    st.info(f"Total slides: {total_slides} | Already vectorized: {vectorized_count} | To process: {unvectorized_count}")
                
                if unvectorized_count == 0:
                    st.success("✅ All slides are already vectorized!")
                    return
                
                # Initialize vectorstore
                with st.spinner("Initializing vectorstore..."):
                    print("📦 Initializing vectorstore...")
                    chroma_db, local_path, chroma_folder_id, vectorstore_folder_id = initialize_vectorstore(
                        drive, ROOT_FOLDER_ID
                    )
                    print("✓ Vectorstore initialized")
                    st.success("Vectorstore initialized")
                
                # Progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Vectorize slides
                status_text.text(f"Vectorizing {unvectorized_count} slides...")
                
                success_count, error_count = vectorize_slide_chunks(
                    sheet,
                    slide_chunks_ws,
                    slide_chunks_df,
                    course_name,
                    sheet_id,
                    chroma_db,
                    progress_bar
                )
                
                progress_bar.progress(1.0)
                
                # Upload to Drive (Chroma auto-saves with persist_directory)
                with st.spinner("Saving vectorstore to Drive..."):
                    print("📤 Uploading vectorstore to Drive...")
                    # Delete old folder to prevent duplicates, then upload fresh
                    try:
                        print("🗑️ Removing old vectorstore from Drive to prevent duplicates...")
                        drive.CreateFile({'id': chroma_folder_id}).Delete()
                        
                        # Recreate the folder
                        folder_metadata = {
                            'title': CHROMA_FOLDER_NAME,
                            'parents': [{'id': vectorstore_folder_id}],
                            'mimeType': 'application/vnd.google-apps.folder'
                        }
                        chroma_folder = drive.CreateFile(folder_metadata)
                        chroma_folder.Upload()
                        chroma_folder_id = chroma_folder['id']
                    except Exception as e:
                        print(f"⚠️ Could not delete old folder (may not exist): {e}")
                        # If delete failed, try to find existing folder
                        chroma_query = (
                            f"title='{CHROMA_FOLDER_NAME}' and '{vectorstore_folder_id}' in parents "
                            f"and mimeType='application/vnd.google-apps.folder' and trashed=false"
                        )
                        chroma_folders = drive.ListFile({'q': chroma_query}).GetList()
                        if chroma_folders:
                            chroma_folder_id = chroma_folders[0]['id']
                    
                    # Upload fresh vectorstore
                    upload_folder_to_drive(local_path, chroma_folder_id, drive)
                    print("✓ Upload complete")
                
                # Final status
                status_text.empty()
                st.success(f"""
                ✅ **Vectorization Complete!**
                
                - ✅ Successfully vectorized: **{success_count}** slides
                - ❌ Errors: **{error_count}** slides
                - 📊 Total processed: **{success_count + error_count}** / {unvectorized_count}
                """)
                
                if error_count > 0:
                    st.warning("Some slides had errors. Check the 'vectorized' and 'error_message' columns in your sheet.")
                
            except Exception as e:
                st.error(f"❌ Error: {e}")
                import traceback
                st.code(traceback.format_exc())
    
    with tab2:
        st.header("Q&A Interface")
        st.info("🚧 Coming soon! This section will allow you to ask questions about vectorized courses.")


#Make the app visible on Streamlit UI

main()
