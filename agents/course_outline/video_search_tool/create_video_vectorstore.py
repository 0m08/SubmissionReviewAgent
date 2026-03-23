from langchain_community.vectorstores import Chroma
from tqdm import tqdm
from services.embedding_service import get_embedding_model
from agents.vector_store_image_search.create_vectorstore import upload_folder_to_drive
import os
import json
from services.sheets_service import (
    get_sheet_data_and_df,
    save_to_sheet,
)
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

def chroma_db_exists(drive, parent_folder_id):
    """
    Checks whether chroma_research_db exists inside Vectorstore files in the specified parent folder.
    Returns True if both folders exist.
    """
    vectorstore_folder_list = drive.ListFile({
        'q': f"title='Vectorstore files' and '{parent_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
    }).GetList()

    if not vectorstore_folder_list:
        return False  # No 'Vectorstore files' folder

    vectorstore_folder_id = vectorstore_folder_list[0]['id']

    chroma_folder_list = drive.ListFile({
        'q': f"title='chroma_video_db' and '{vectorstore_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
    }).GetList()

    return bool(chroma_folder_list)

def create_video_vectorstore(sheet, drive):
    """
    Combine all chunks per video into a single embedding.
    Parallelize across video IDs.
    """
    # Read data from the sheet
    video_chunks_sheet, video_chunks_df = get_sheet_data_and_df(sheet, 'HVAC School Video Chunks')
    print(f"Processing {len(video_chunks_df)} video chunks from the sheet...")

    # Ensure 'vectorized' and 'embedding_ts' columns exist
    if 'vectorized' not in video_chunks_df.columns:
        video_chunks_df['vectorized'] = 'FALSE'
    if 'embedding_ts' not in video_chunks_df.columns:
        video_chunks_df['embedding_ts'] = ''

    # Initialize embedding model
    embedding_function = get_embedding_model()
    print("Cohere embedding model initialized.")
    parent_folder_id='1SoJDL08Wa7sQq9bCsHcB1bNzSzKj3Z1y'


    # Ensure 'Vectorstore files' folder exists
    file_list = drive.ListFile({
        'q': f"title='Vectorstore files' and '{parent_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
    }).GetList()

    if file_list:
        vectorstore_folder_id = file_list[0]['id']
        print("Found 'Vectorstore files' folder inside the specified parent folder.\n")
    else:
        print("Creating 'Vectorstore files' folder inside the specified parent folder...")
        folder_metadata = {
            'title': 'Vectorstore files',
            'parents': [{'id': parent_folder_id}],
            'mimeType': 'application/vnd.google-apps.folder'
        }
        new_folder = drive.CreateFile(folder_metadata)
        new_folder.Upload()
        vectorstore_folder_id = new_folder['id']
        print("Created 'Vectorstore files' folder inside the specified parent folder.\n")

    # Initialize Chroma vectorstore
    local_chroma_root = "/tmp/temp_chroma_folder"
    local_chroma_path = os.path.join(local_chroma_root, "chroma_video_db")
    os.makedirs(local_chroma_path, exist_ok=True)

    chroma_db = Chroma(
        embedding_function=embedding_function,
        collection_name="video_embeddings",
        persist_directory=local_chroma_path
    )
    print(f"Chroma vectorstore initialized at {local_chroma_path}.")

    # Prepare tasks for each row (chunk)
    tasks = [(idx, row) for idx, row in video_chunks_df.iterrows() if row['vectorized'] != 'TRUE']
    print(f"{len(tasks)} chunks to vectorize individually.")

    # Embed a single chunk
    def embed_chunk(row_tuple):
        idx, row = row_tuple
        try:
            # Combine fields for text
            combined_text = " ".join([
                str(row.get('video_title', '')),
                str(row.get('chapter_title', '')),
                str(row.get('text_0', ''))
            ])

            # Parse metadata
            row_metadata = row.get('metadata', {})
            if isinstance(row_metadata, str):
                try:
                    row_metadata = json.loads(row_metadata)
                except Exception:
                    row_metadata = {}

            # Unique chunk ID (video_id + row index)
            chunk_id = f"{row.get('video_id', 'unknown')}_{idx}"

            # Build metadata dict
            metadata = {
                'chunk_id': chunk_id,
                'video_id': row.get('video_id', 'unknown'),
                'video_title': row.get('video_title', 'Untitled'),
                'chapter_title': row.get('chapter_title', ''),
                'text_0': row.get('text_0', ''),
                **row_metadata  # include all metadata keys like source, channel, etc.
            }

            # Add to vectorstore
            chroma_db.add_texts([combined_text], [metadata])

            # Update DataFrame
            video_chunks_df.at[idx, 'vectorized'] = 'TRUE'
            video_chunks_df.at[idx, 'embedding_ts'] = datetime.now().isoformat()

            return f"Chunk {chunk_id} vectorized successfully."

        except Exception as e:
            return f"Error vectorizing chunk {idx}: {e}"

    # Run in parallel
    max_workers = 5
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(embed_chunk, t) for t in tasks]
        for future in tqdm(as_completed(futures), total=len(futures)):
            print(future.result())

    # Persist and save
    chroma_db.persist()
    print("Vectorstore persisted to disk.")
    print("Total embeddings stored:", chroma_db._collection.count())

    save_to_sheet(video_chunks_sheet, video_chunks_df)
    print("Google Sheet updated with vectorization status.")

    print("Uploading vectorstore to Drive...")
    upload_folder_to_drive(local_chroma_path, vectorstore_folder_id, drive)
    
    print("Upload complete. All embeddings processed successfully.")