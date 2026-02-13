from langchain_classic.vectorstores import Chroma
from tqdm import tqdm
from services.embedding_service import get_embedding_model
from agents.vector_store_image_search.create_vectorstore import upload_folder_to_drive, download_folder_from_drive
import os
from services.sheets_service import (
    get_sheet_data_and_df,
    save_to_sheet,
)
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime


def update_video_vectorstore(sheet, drive):
    """
    Incrementally update Chroma DB with new video embeddings from Google Sheets and upload to Drive.
    :param sheet: Google Sheet instance containing video chunks.
    :param drive: Google Drive instance.
    """
    print("Starting video vectorstore update...")

    parent_folder_id = '1cUBmd1H1hBHSLohnAF68VJTEU9XcFXFK'
    local_chroma_root = "/tmp/temp_chroma_folder"
    local_chroma_path = os.path.join(local_chroma_root, "chroma_video_db")
    os.makedirs(local_chroma_root, exist_ok=True)

    # ------------------ Download existing vectorstore ------------------
    print("Downloading existing video vectorstore from Drive...")
    file_list = drive.ListFile({
        'q': f"title='chroma_video_db' and '{parent_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
    }).GetList()

    if file_list:
        chroma_folder_id = file_list[0]['id']
        download_folder_from_drive(chroma_folder_id, local_chroma_path, drive)
        print("Existing vectorstore downloaded.")
    else:
        print("No existing vectorstore found. A new one will be created.")

    # ------------------ Initialize Chroma ------------------
    embedding_function = get_embedding_model()
    chroma_db = Chroma(
        embedding_function=embedding_function,
        collection_name="video_embeddings",
        persist_directory=local_chroma_path
    )
    print(f"Chroma initialized at {local_chroma_path}.")

    # ------------------ Load video chunks ------------------
    video_chunks_sheet, video_chunks_df = get_sheet_data_and_df(sheet, 'HVAC School Video Chunks')
    print(f"Loaded {len(video_chunks_df)} video chunks from sheet.")

    # Ensure required columns exist
    for col in ['vectorized', 'embedding_ts']:
        if col not in video_chunks_df.columns:
            video_chunks_df[col] = 'FALSE' if col == 'vectorized' else ''

    # ------------------ Filter new videos ------------------
    grouped = video_chunks_df.groupby('video_id')
    tasks = [(video_id, group) for video_id, group in grouped if not all(group['vectorized'] == 'TRUE')]
    print(f"{len(tasks)} new videos to vectorize.")

    # ------------------ Embed each video ------------------
    def embed_video(video_tuple):
        video_id, group = video_tuple
        try:
            combined_texts = []
            for _, row in group.iterrows():
                chunk_text = " ".join([
                    str(row.get('video_title', '')),
                    str(row.get('chapter_title', '')),
                    str(row.get('source', '')),
                    str(row.get('channel', 'Unknown')),
                    str(row.get('text_0', ''))
                ])
                combined_texts.append(chunk_text)

            combined_text = " ".join(combined_texts)
            metadata = {
                'video_id': video_id,
                'video_title': group['video_title'].iloc[0],
                'chapter_titles': "; ".join(group['chapter_title'].tolist()),
                'source': group.get('source', [''])[0],
                'channel': group.get('channel', ['Unknown'])[0],
                'text_0': group.get('text_0', [''])[0],
            }

            chroma_db.add_texts([combined_text], [metadata])

            # Update DataFrame
            now = datetime.now().isoformat()
            for idx in group.index:
                video_chunks_df.at[idx, 'vectorized'] = 'TRUE'
                video_chunks_df.at[idx, 'embedding_ts'] = now

            return f"Video '{metadata['video_title']}' ({video_id}) vectorized successfully."

        except Exception as e:
            return f"Error vectorizing video {video_id}: {e}"

    # ------------------ Parallel embedding ------------------
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tqdm import tqdm

    max_workers = 5
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(embed_video, t) for t in tasks]
        for future in tqdm(as_completed(futures), total=len(futures)):
            print(future.result())

    # ------------------ Persist and update sheet ------------------
    chroma_db.persist()
    save_to_sheet(video_chunks_sheet, video_chunks_df)
    print("Vectorstore persisted and Google Sheet updated.")

    # ------------------ Upload updated vectorstore to Drive ------------------
    if file_list:
        print("Removing old vectorstore folder from Drive...")
        drive.CreateFile({'id': chroma_folder_id}).Delete()

    print("Uploading updated vectorstore to Drive...")
    upload_folder_to_drive(local_chroma_path, parent_folder_id, drive)
    print("Update complete. All new video embeddings processed successfully.")
