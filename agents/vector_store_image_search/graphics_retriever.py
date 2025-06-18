import torch
import clip
import os
from services.embedding_service import get_embedding_model
from agents.vector_store_image_search.create_vectorstore import download_image_from_drive
import os
from langchain_chroma import Chroma
from services.drive_service import download_folder_from_drive

def load_central_chroma_db(embedding_function, drive, central_folder_id):
    """
    Load the shared Chroma DB collections from Google Drive.
    :param embedding_function: Function to compute text embeddings.
    :param drive: Google Drive instance.
    :param central_folder_id: ID of the central folder containing the Chroma DB.
    :return: Dictionary with 'text' and 'image' collections.
    """


    # Local DB folder (shared by both collections)
    local_chroma_root = "/tmp/temp_chroma_folder"
    local_chroma_path = os.path.join(local_chroma_root, "chroma_graphics_db")
    os.makedirs(local_chroma_root, exist_ok=True)

    # Search for the shared Chroma DB folder
    print("Searching for 'chroma_graphics_db' in Drive...")
    file_list = drive.ListFile({
        'q': f"title='chroma_graphics_db' and '{central_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
    }).GetList()

    if not file_list:
        raise FileNotFoundError("'chroma_graphics_db' not found in Drive.")

    chroma_folder_id = file_list[0]['id']
    print(f"Found Chroma folder ID: {chroma_folder_id}")

    if not os.path.exists(os.path.join(local_chroma_path, "chroma.sqlite3")):
        print("⬇Downloading Chroma DB...")
        download_folder_from_drive(chroma_folder_id, local_chroma_path, drive)
        print(f"Downloaded to: {local_chroma_path}")
    else:
        print("Using existing local copy of Chroma DB.")

    # Load text_embeddings collection
    text_chroma = Chroma(
        embedding_function=embedding_function,
        collection_name="text_embeddings",
        persist_directory=local_chroma_path
    )
    print("Loaded 'text_embeddings' collection.")

    # Load image_embeddings collection
    # image_chroma = Chroma(
    #     embedding_function=None,  # CLIP vectors are precomputed
    #     collection_name="image_embeddings",
    #     persist_directory=local_chroma_path
    # )
    # print("Loaded 'image_embeddings' collection.")

    return {
        "text": text_chroma,
        # "image": image_chroma
    }

    

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)
clip_model.eval()

def get_text_embedding_clip(text: str):
    """
    Compute CLIP embedding for a given text query.
    Returns a normalized list of floats.
    """
    with torch.no_grad():
        tokens = clip.tokenize([text]).to(device)
        text_features = clip_model.encode_text(tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features[0].cpu().numpy().tolist()


def graphics_retriever(query, drive, k=5, filters=None):
    """
    Search for similar images using a text query and return them as PIL images.
    :param query: Text query to search for.
    :param drive: Google Drive instance for image retrieval.
    :param k: Number of top results to return.
    :param filters: Optional filters to apply on the results.
    :return: List of dictionaries with "image", "similarity", and "metadata".
    """

    print("🔍 Loading central Chroma DB...")
    embedding_function = get_embedding_model()
    central_folder_id = '1BACIAhOG-c2659kl2d_gJuzulopNGIrP'
    dbs = load_central_chroma_db(embedding_function, drive, central_folder_id)

    print(f"🔤 Performing text similarity search: '{query}'")
    chroma_db = dbs["text"]
    all_docs = chroma_db.get()
    num_docs = len(all_docs['documents'])
    metadata_list = all_docs['metadatas']

    if num_docs == 0:
        print("⚠️ Chroma DB is empty. No documents found.")
        return []

    print(f"✅ Text Chroma DB loaded with {num_docs} documents.")
    results_docs = chroma_db.similarity_search_with_score(query, k=50)

    # 🔍 In-memory filtering and image download
    filtered_results = []
    for doc, score in results_docs:
        if len(filtered_results) >= k:
            break  # ✅ Stop once top k images are collected

        metadata = doc['metadata'] if isinstance(doc, dict) else doc.metadata

        if 'image_id' not in metadata:
            continue

        # 🧪 Apply filters
        if filters:
            if filters.get("mime_type") and metadata.get("mime_type") not in filters["mime_type"]:
                continue
            if filters.get("image_title") and metadata.get("image_title") and filters["image_title"].lower() not in metadata["image_title"].lower():
                continue
            if filters.get("image_type") and metadata.get("image_type") and filters["image_type"].lower() not in metadata["image_type"].lower():
                continue

        try:
            image_file_id = metadata['image_id']
            print(f"⬇️ Attempting to download image: {image_file_id}")
            import time

            try:
                start_time = time.time()
                pil_image = download_image_from_drive(drive, image_file_id)
                print("✅ Image downloaded:", image_file_id)

                if not pil_image:
                    print(f"⚠️ Image {image_file_id} download returned None.")
                    continue

                elapsed = time.time() - start_time
                if elapsed > 10:
                    print(f"⏱️ Warning: Downloading image {image_file_id} took {elapsed:.2f} seconds.")

                filtered_results.append({
                    "similarity": score,
                    "image": pil_image,
                    "metadata": metadata
                })

            except Exception as e:
                print(f"❌ Error downloading image {image_file_id}: {e}")

        except Exception as e:
            print(f"❌ Error downloading or processing image ID {metadata.get('image_id', 'N/A')}: {e}")

    if not filtered_results:
        print("⚠️ No results returned after filtering or downloading.")

    filtered_results.sort(key=lambda x: x['similarity'])
    print(f"✅ Returning {len(filtered_results)} images.")
    return filtered_results[:k]  # List of dicts with "image", "similarity", etc.
