import torch
import clip
import os
import time
import requests
from io import BytesIO
from PIL import Image
from services.embedding_service import get_embedding_model
from agents.vector_store_image_search.create_vectorstore import download_image_from_drive
from langchain_chroma import Chroma
from services.drive_service import download_folder_from_drive
# from typing import List, Set, Dict

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
    image_chroma = Chroma(
        embedding_function=None,  # CLIP vectors are precomputed
        collection_name="image_embeddings",
        persist_directory=local_chroma_path
    )
    print("Loaded 'image_embeddings' collection.")

    return {
        "text": text_chroma,
        "image": image_chroma
    }
    
    
# Load model globally
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)
clip_model.eval()

def get_image_embedding_from_pil(image: Image.Image):
    """
    Compute CLIP embedding for a given PIL Image.

    Returns a list of floats (embedding vector).
    """
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = clip_model.encode_image(image_tensor)
        image_features /= image_features.norm(dim=-1, keepdim=True)
    return image_features[0].cpu().numpy().tolist()



def graphics_retriever(query=None, query_image=None, drive=None, k=5, filters=None):
    """
    Search for similar images using text, a local image path, an image URL, or a PIL image.
    :param query: Text string or image path or image URL.
    :param query_image: Optional direct PIL image (overrides query).
    :param drive: Authenticated GoogleDrive instance.
    :param k: Number of results to return.
    :param filters: Optional metadata filters.
    :return: List of dicts with 'similarity', 'image' (PIL), and 'metadata'.
    """

    assert query or query_image, "Please provide a query or an image."

    def match_filters(metadata):
        if not filters:
            return True
        if filters.get("mime_type") and metadata.get("mime_type") not in filters["mime_type"]:
            return False
        if filters.get("image_title") and metadata.get("image_title") and \
                filters["image_title"].lower() not in metadata["image_title"].lower():
            return False
        if filters.get("image_type") and metadata.get("image_type"):
            metadata_types = [t.strip().lower() for t in metadata["image_type"].split(",")]
            selected_types = [t.lower() for t in filters["image_type"]]
            if not any(sel in metadata_types for sel in selected_types):
                return False
        return True

    # Load Chroma DBs
    print("Loading central Chroma DB...")
    embedding_function = get_embedding_model()
    central_folder_id = '1BACIAhOG-c2659kl2d_gJuzulopNGIrP'
    dbs = load_central_chroma_db(embedding_function, drive, central_folder_id)

    results = []
    seen_phashes = set()

    # === Image Query Handling ===
    if query_image is None and query:
        if query.startswith("http") and any(query.endswith(ext) for ext in [".jpg", ".jpeg", ".png, .gif,.webp, .bmp, .tiff, .svg, .ico, .avif, .heic,, .heif, .jfif, .exif, .jxl, .j2c, .j2k, .jpf, .jp2, .jpx, .jpm, .jxr, .wdp, .hdp"]):
            print("Detected image URL. Downloading...")
            try:
                response = requests.get(query)
                if response.status_code == 200:
                    query_image = Image.open(BytesIO(response.content)).convert("RGB")
                    print("Image loaded from URL.")
                else:
                    print("Failed to fetch image from URL.")
                    return []
            except Exception as e:
                print(f"Error loading image from URL: {e}")
                return []

        elif os.path.exists(query) and query.lower().endswith((".jpg", ".jpeg", ".png, .gif,.webp, .bmp, .tiff, .svg, .ico, .avif, .heic,, .heif, .jfif, .exif, .jxl, .j2c, .j2k, .jpf, .jp2, .jpx, .jpm, .jxr, .wdp, .hdp")):
            print("Detected local image path. Loading...")
            try:
                query_image = Image.open(query).convert("RGB")
                print("Image loaded from file.")
            except Exception as e:
                print(f"Error loading local image: {e}")
                return []

    # === Image-Based Search ===
    if query_image:
        print("Using image embedding for similarity search.")
        try:
            query_vector = get_image_embedding_from_pil(query_image)
        except Exception as e:
            print(f"Failed to embed image: {e}")
            return []

        image_db = dbs["image"]
        raw_results = image_db._collection.query(
            query_embeddings=[query_vector],
            n_results=50,
            include=["metadatas", "distances"]
        )
        matches = zip(raw_results["metadatas"][0], raw_results["distances"][0])

    # === Text-Based Search ===
    else:
        print(f"🔍 Performing text search: '{query}'")
        text_db = dbs["text"]
        raw_results = text_db.similarity_search_with_score(query, k=50)
        matches = ((doc.metadata, score) for doc, score in raw_results)

    # === Process Results ===
    for metadata, score in matches:
        if len(results) >= k:
            break

        if 'image_id' not in metadata or not match_filters(metadata):
            continue

        phash = metadata.get("phash")
        if phash and phash in seen_phashes:
            print(f"Skipping visually duplicate image (pHash: {phash})")
            continue
        if phash:
            seen_phashes.add(phash)

        try:
            image_file_id = metadata['image_id']
            print(f"⬇ Downloading image: {image_file_id}")
            start_time = time.time()

            pil_image = download_image_from_drive(drive, image_file_id)

            if not pil_image:
                print(f"Image {image_file_id} returned None.")
                continue

            elapsed = time.time() - start_time
            if elapsed > 10:
                print(f"Download took {elapsed:.2f} seconds.")

            results.append({
                "similarity": score,
                "image": pil_image,
                "metadata": metadata
            })

        except Exception as e:
            print(f"Error downloading image ID {metadata.get('image_id')}: {e}")

    results.sort(key=lambda x: x["similarity"])
    print(f"Returning {len(results)} results.")
    return results[:k]
