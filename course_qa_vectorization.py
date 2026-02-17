"""
Course Q&A App - Slide Chunks Vectorization
Single file containing both UI and vectorization logic
"""

import streamlit as st
import os
import re
import pandas as pd
from typing import Optional, Tuple, List, Dict, Any
from tqdm import tqdm
from datetime import datetime
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain_cohere import CohereRerank
from langchain_classic.retrievers import ContextualCompressionRetriever
from services.embedding_service import get_embedding_model
from services.sheets_service import get_sheet_data_and_df, save_to_sheet
from services.drive_service import download_folder_from_drive, upload_folder_to_drive
from services.llm_service import llm_with_retry
from pydrive2.drive import GoogleDrive
from pydrive2.auth import GoogleAuth
import gspread
from google.oauth2.credentials import Credentials
from services.drive_service import init_clients_from_credentials

# ============================================================================
# Configuration
# ============================================================================

ROOT_FOLDER_ID = "1UkZGl3HeCFq48zEl3HTkKdvty91gUse_"
COURSE_SHEETS_FOLDER_ID = "1SYzI0NDng49lM_F9HBruW8EJCxlp6IQy"
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
# Q&A Functions
# ============================================================================

def get_available_courses_from_folder(drive, gc, folder_id: str) -> List[Tuple[str, str]]:
    """
    Get list of available courses by reading Course Name from sheets in a Drive folder.
    
    Returns: List of tuples (course_name, sheet_id)
    """
    courses = []
    
    # Query for all Google Sheets in the folder
    query = (
        f"'{folder_id}' in parents and "
        f"mimeType='application/vnd.google-apps.spreadsheet' and "
        f"trashed=false"
    )
    
    try:
        file_list = drive.ListFile({'q': query}).GetList()
        print(f"📋 Found {len(file_list)} sheets in folder")
        
        for file_item in file_list:
            sheet_id = file_item['id']
            sheet_name = file_item.get('title', 'Unknown')
            
            try:
                # Open the sheet
                sheet = gc.open_by_key(sheet_id)
                
                # Get course name
                course_name = get_course_name_from_sheet(sheet, "Course info")
                
                courses.append((course_name, sheet_id))
                print(f"  ✓ Found course: {course_name}")
                
            except Exception as e:
                # Skip sheets that don't have Course info tab or Course Name
                print(f"  ⚠️ Skipping sheet '{sheet_name}': {e}")
                continue
                
    except Exception as e:
        print(f"❌ Error listing sheets from folder: {e}")
    
    # Sort by course name
    courses.sort(key=lambda x: x[0])
    return courses


def load_qa_retriever(drive, root_folder_id: str, use_reranking: bool = True, k: int = 15) -> ContextualCompressionRetriever:
    """
    Load vectorstore and create retriever for Q&A.
    Returns: ContextualCompressionRetriever with optional Cohere reranking
    """
    embedding_function = get_embedding_model()
    
    # Find vectorstore folder
    vectorstore_query = (
        f"title='{VECTORSTORE_FOLDER_NAME}' and '{root_folder_id}' in parents "
        f"and mimeType='application/vnd.google-apps.folder' and trashed=false"
    )
    vectorstore_folders = drive.ListFile({'q': vectorstore_query}).GetList()
    
    if not vectorstore_folders:
        raise FileNotFoundError("Vectorstore not found. Please vectorize some courses first.")
    
    # Find chroma folder
    vectorstore_folder_id = vectorstore_folders[0]['id']
    chroma_query = (
        f"title='{CHROMA_FOLDER_NAME}' and '{vectorstore_folder_id}' in parents "
        f"and mimeType='application/vnd.google-apps.folder' and trashed=false"
    )
    chroma_folders = drive.ListFile({'q': chroma_query}).GetList()
    
    if not chroma_folders:
        raise FileNotFoundError("Chroma vectorstore not found. Please vectorize some courses first.")
    
    # Download and load vectorstore
    local_path = os.path.join(LOCAL_CHROMA_ROOT, CHROMA_FOLDER_NAME)
    os.makedirs(LOCAL_CHROMA_ROOT, exist_ok=True)
    
    chroma_folder_id = chroma_folders[0]['id']
    sqlite_path = os.path.join(local_path, "chroma.sqlite3")
    
    if not os.path.exists(sqlite_path):
        print("📥 Downloading vectorstore from Drive...")
        download_folder_from_drive(chroma_folder_id, local_path, drive)
    
    chroma_db = Chroma(
        embedding_function=embedding_function,
        collection_name=COLLECTION_NAME,
        persist_directory=local_path
    )
    
    # Create base retriever - reduce initial retrieval for faster reranking
    # Retrieve k+5 instead of k*2 to reduce latency
    base_retriever = chroma_db.as_retriever(
        search_kwargs={"k": k + 5 if use_reranking else k}
    )
    
    # Add reranking if enabled
    if use_reranking:
        compressor = CohereRerank(
            model="rerank-v3.5",
            top_n=k,
        )
        retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever
        )
    else:
        retriever = base_retriever
    
    return retriever


def retrieve_relevant_slides(
    retriever: ContextualCompressionRetriever,
    question: str,
    course_name: Optional[str] = None,
    k: int = 10
) -> List[Document]:
    """
    Retrieve relevant slides for a question, optionally filtered by course.
    
    Args:
        retriever: The retriever to use
        question: User's question
        course_name: Optional course name to filter by (None = all courses)
        k: Number of results to return
    
    Returns:
        List of Document objects with metadata
    """
    # Retrieve documents
    docs = retriever.invoke(question)
    
    # Filter by course if specified
    if course_name:
        filtered_docs = []
        for doc in docs:
            metadata = getattr(doc, 'metadata', {}) or {}
            doc_course_name = metadata.get('course_name', '')
            if doc_course_name and course_name.lower() in doc_course_name.lower():
                filtered_docs.append(doc)
        
        # If filtering removed all docs, return original (fallback)
        if not filtered_docs:
            print(f"⚠️ No results for course '{course_name}', returning all courses")
            return docs[:k]
        
        return filtered_docs[:k]
    
    return docs[:k]


def format_citations(docs: List[Document]) -> List[Dict[str, str]]:
    """
    Format citations from retrieved documents.
    
    Returns:
        List of citation dicts with topic, subtopic, slide_content
    """
    citations = []
    seen = set()
    
    for doc in docs:
        metadata = getattr(doc, 'metadata', {}) or {}
        slide_content = getattr(doc, 'page_content', '')
        
        # Create unique key to avoid duplicates
        citation_key = (
            metadata.get('topic', ''),
            metadata.get('subtopic', ''),
            slide_content[:100]  # First 100 chars for uniqueness
        )
        
        if citation_key in seen:
            continue
        seen.add(citation_key)
        
        citation = {
            'topic': metadata.get('topic', 'Unknown Topic'),
            'subtopic': metadata.get('subtopic', 'Unknown Subtopic'),
            'slide_content': slide_content,
        }
        citations.append(citation)
    
    return citations


def generate_answer_with_rag(
    question: str,
    docs: List[Document],
    course_name: Optional[str] = None,
    llm: str = "gemini_3_flash"
) -> Tuple[str, List[Dict[str, str]]]:
    """
    Generate answer using RAG with strict grounding in retrieved content.
    
    Returns:
        Tuple of (answer_text, citations)
    """
    if not docs:
        return "I couldn't find any relevant information in the course content to answer this question.", []
    
    # Format retrieved context
    context_parts = []
    for i, doc in enumerate(docs, 1):
        metadata = getattr(doc, 'metadata', {}) or {}
        content = getattr(doc, 'page_content', '')
        
        context_parts.append(f"""
[Source {i}]
Course: {metadata.get('course_name', 'Unknown')}
Topic: {metadata.get('topic', 'Unknown')}
Subtopic: {metadata.get('subtopic', 'Unknown')}
Slide: {metadata.get('slide_chunk_title', 'Untitled')}
Content: {content}
""")
    
    context = "\n".join(context_parts)
    
    # Build RAG prompt
    course_filter_note = f"\nNote: The user is asking about the course '{course_name}'. Focus on information from that course." if course_name else ""
    
    rag_prompt = f"""You are a helpful assistant specializing in the field of HVAC that answers questions about SkillCat courses using ONLY the provided course content.

Your task is to answer the user's question using STRICTLY the information provided in the context below. You must NOT use any external knowledge or make assumptions beyond what is explicitly stated in the course content.

{course_filter_note}

<context>
{context}
</context>

<instructions>
1. Answer the question using ONLY information from the provided context
2. If the context doesn't contain enough information to answer the question, say so explicitly
3. Do NOT make up information or use knowledge outside the provided context
4. Cite specific sources when referencing information (e.g., "According to [Source 1]...")
5. If multiple sources provide relevant information, synthesize them clearly
6. Be concise but complete
7. Use clear, instructional language appropriate for course learners
8. Don't use words like "Based on the provided course content" while starting your answers
</instructions>

<question>
{question}
</question>
"""
    
    # Generate answer
    try:
        response = llm_with_retry(rag_prompt, llm_name=llm, max_retries=3)
        
        # Extract text from response
        if hasattr(response, 'content'):
            answer = response.content
        elif isinstance(response, dict):
            answer = response.get('content', str(response))
        else:
            answer = str(response)
        
        # Format citations
        citations = format_citations(docs)
        
        return answer.strip(), citations
        
    except Exception as e:
        error_msg = f"Error generating answer: {e}"
        print(f"❌ {error_msg}")
        return error_msg, []


# ============================================================================
# Streamlit UI
# ============================================================================

def main():
    st.set_page_config(
        page_title="Course Q&A - Vectorization",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("📚 Course Q&A App")
    
    # Check authentication
    if 'drive' not in st.session_state or 'gc' not in st.session_state:
        st.error("❌ Please authenticate first by logging in through the main app")
        return
    
    drive = st.session_state.drive
    gc = st.session_state.gc
    
    # Main UI - Two Sections
    tab1, tab2 = st.tabs(["🔧 Vectorization", "❓ Q&A"])
    
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

        st.markdown("""
        **Ask questions about courses:**
        - Select a course (or "All Courses" for cross-course search)
        - Ask your question in natural language
        - Get answers with citations to exact slide sources
        """)
        
        # Load available courses
        if 'available_courses' not in st.session_state:
            st.session_state.available_courses = []
        
        # Get courses
        if not st.session_state.available_courses:
            with st.spinner("Loading available courses..."):
                try:
                    courses = get_available_courses_from_folder(
                        drive, gc, COURSE_SHEETS_FOLDER_ID
                    )
                    st.session_state.available_courses = courses
                    if courses:
                        st.success(f"✅ Found {len(courses)} course(s)")
                    else:
                        st.warning("⚠️ No courses found. Please vectorize some courses first.")
                except Exception as e:
                    st.error(f"❌ Error loading courses: {e}")
                    st.session_state.available_courses = []
        
        # Preload retriever when tab opens (cache in session state)
        if 'qa_retriever' not in st.session_state:
            with st.spinner("Loading vectorstore (one-time setup)..."):
                try:
                    print("📥 Preloading vectorstore...")
                    st.session_state.qa_retriever = load_qa_retriever(drive, ROOT_FOLDER_ID, use_reranking=True, k=10)
                    print("✅ Vectorstore preloaded and cached")
                except Exception as e:
                    print(f"❌ Error preloading vectorstore: {e}")
                    st.error(f"❌ Error loading vectorstore: {e}")
                    st.session_state.qa_retriever = None
        
        # Course selection
        if st.session_state.available_courses:
            course_options = ["All Courses"] + [name for name, _ in st.session_state.available_courses]
            selected_course = st.selectbox(
                "Select Course",
                options=course_options,
                help="Choose a specific course or 'All Courses' to search across all"
            )
            
            # Get selected course's sheet_id if not "All Courses"
            selected_sheet_id = None
            selected_course_name = None
            if selected_course != "All Courses":
                for name, sheet_id in st.session_state.available_courses:
                    if name == selected_course:
                        selected_sheet_id = sheet_id
                        selected_course_name = name
                        break
            
            st.divider()
            
            # Question input
            question = st.text_area(
                "Ask a question:",
                placeholder="e.g., What are the key safety precautions when working with HVAC systems?",
                height=100,
                help="Enter your question about the course"
            )
            
            # Ask button 
            if st.button("🔍 Ask Question", type="primary"):
                if not question.strip():
                    st.warning("Please enter a question")
                    return
                
                try:
                    # Use preloaded retriever (should already be cached from tab load)
                    if 'qa_retriever' not in st.session_state or st.session_state.qa_retriever is None:
                        st.error("❌ Vectorstore not loaded. Please refresh the page.")
                        return
                    
                    retriever = st.session_state.qa_retriever
                    print(f"🔍 Question: {question}")
                    print(f"📚 Course filter: {selected_course_name or 'All Courses'}")
                    
                    # Retrieve relevant slides
                    with st.spinner("Searching for relevant slides..."):
                        docs = retrieve_relevant_slides(
                            retriever,
                            question,
                            course_name=selected_course_name,
                            k=10
                        )
                        print(f"📄 Retrieved {len(docs)} relevant slides")
                    
                    if not docs:
                        st.warning("No relevant slides found. Try rephrasing your question or selecting a different course.")
                        return
                    
                    # Generate answer (use faster model without thinking)
                    with st.spinner("Generating answer..."):
                        answer, citations = generate_answer_with_rag(
                            question,
                            docs,
                            course_name=selected_course_name,
                            llm="gemini_3_flash"  # Faster than thinking model
                        )
                    
                    # Display answer
                    st.subheader("💡 Answer")
                    st.markdown(answer)
                    
                    # Display citations
                    if citations:
                        st.subheader("📚 Sources")
                        st.markdown(f"*Found {len(citations)} relevant slide(s):*")
                        
                        for i, citation in enumerate(citations, 1):
                            with st.expander(f"Source {i}: {citation['topic']} - {citation['subtopic']}"):
                                st.markdown(f"""
                                **Topic:** {citation['topic']}  
                                **Subtopic:** {citation['subtopic']}  
                                
                                **Slide Content:**
                                {citation['slide_content']}
                                """)
                    
                    print(f"✅ Answer generated with {len(citations)} citations")
                    
                except Exception as e:
                    st.error(f"❌ Error: {e}")
                    import traceback
                    st.code(traceback.format_exc())
                    print(f"❌ Q&A Error: {e}")
        else:
            st.info("⚠️ No courses found. Please vectorize some courses first using the Vectorization tab.")


#Make the app visible on Streamlit UI

main()
