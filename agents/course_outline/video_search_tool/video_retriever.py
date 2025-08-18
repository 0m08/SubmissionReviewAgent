import os
from typing import List, Dict, Optional
from langchain.vectorstores import Chroma
from langchain.vectorstores import Chroma
from services.embedding_service import get_embedding_model
from agents.vector_store_image_search.create_vectorstore import download_folder_from_drive
from concurrent.futures import ThreadPoolExecutor, as_completed

def load_video_chroma_db(embedding_function, drive, central_folder_id):
    """
    Load the shared Chroma DB collection for HVAC videos from Google Drive.
    :param embedding_function: Function to compute text embeddings.
    :param drive: Authenticated PyDrive2 instance.
    :param central_folder_id: Drive folder containing 'chroma_video_db'.
    :return: Chroma vectorstore instance.
    """

    # Local path for Chroma DB
    local_chroma_root = "/tmp/temp_chroma_folder"
    local_chroma_path = os.path.join(local_chroma_root, "chroma_video_db")
    os.makedirs(local_chroma_root, exist_ok=True)

    # Search for the Chroma DB folder in Drive
    print("Searching for 'chroma_video_db' in Drive...")
    file_list = drive.ListFile({
        'q': f"title='chroma_video_db' and '{central_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
    }).GetList()

    if not file_list:
        raise FileNotFoundError("'chroma_video_db' not found in Drive.")
    chroma_folder_id = file_list[0]['id']
    print(f"Found Chroma folder ID: {chroma_folder_id}")

    # Download folder if not already present
    if not os.path.exists(os.path.join(local_chroma_path, "chroma.sqlite3")):
        print("⬇ Downloading Chroma DB from Drive...")
        download_folder_from_drive(chroma_folder_id, local_chroma_path, drive)
        print(f"Downloaded to local path: {local_chroma_path}")
    else:
        print("Using existing local Chroma DB copy.")

    # Load video embeddings collection
    video_chroma = Chroma(
        embedding_function=embedding_function,
        collection_name="video_embeddings",
        persist_directory=local_chroma_path
    )
    print("Loaded 'video_embeddings' collection.")
    return video_chroma


def video_retriever(query: str, drive=None, k: int = 10, filters: Optional[Dict] = None) -> List[Dict]:
    """
    Retrieve relevant HVAC video chunks based on a text query.
    Returns results per chunk, sorted from most relevant to least relevant.
    Constructs YouTube embed URLs with start and end times from metadata,
    and includes the transcript from text_0.
    """
    central_folder_id = '1cUBmd1H1hBHSLohnAF68VJTEU9XcFXFK'
    if not query:
        raise ValueError("Please provide a text query.")

    # Load embedding model and video Chroma DB
    embedding_function = get_embedding_model()
    video_db = load_video_chroma_db(embedding_function, drive, central_folder_id)

    # Helper to filter metadata
    def match_filters(metadata):
        if not filters:
            return True
        for key, val in filters.items():
            if key in metadata and val.lower() not in str(metadata[key]).lower():
                return False
        return True

    print("Searching video embeddings for query:", query)
    try:
        # Search top k*5 chunks to ensure enough results after filtering
        raw_results = video_db.similarity_search_with_score(query, k=k*5)

        final_results = []
        for doc, score in raw_results:
            metadata = doc.metadata or {}
            if not match_filters(metadata):
                continue

            chunk_id = metadata.get("chunk_id") or f"{metadata.get('video_id', 'unknown')}_0"
            vid_id = metadata.get("video_id")
            start_sec = metadata.get("start_time") or 0
            end_sec = metadata.get("end_time") or None

            if vid_id:
                if end_sec is not None:
                    video_url = f"https://www.youtube.com/embed/{vid_id}?start={start_sec}&end={end_sec}"
                else:
                    video_url = f"https://www.youtube.com/embed/{vid_id}?start={start_sec}"
            else:
                video_url = "#"

            transcript = metadata.get("text_0", "Transcript not available.")

            final_results.append({
                "similarity": score,
                "chunk_id": chunk_id,
                "video_id": vid_id,
                "video_title": metadata.get("video_title") or "Untitled",
                "chapter_title": metadata.get("chapter_title") or "",
                "start_time": start_sec,
                "end_time": end_sec,
                "video_url": video_url,
                "channel": metadata.get("channel"),
                "transcript": transcript
            })

        # Sort by similarity descending and return top k chunks
        final_results = sorted(final_results, key=lambda x: x["similarity"], reverse=True)[:k]

        print(f"Found {len(final_results)} matching video chunks.")
        return final_results

    except Exception as e:
        print(f"Error during video similarity search: {e}")
        return []