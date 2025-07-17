import cohere
import os
import requests
from io import BytesIO
from PIL import Image
from services.embedding_service import get_embedding_model
from agents.vector_store_image_search.create_vectorstore import download_image_from_drive
from langchain_chroma import Chroma
from services.drive_service import download_folder_from_drive
from typing import Optional, List, Dict, Union
import base64
import tempfile


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
    
    
co = cohere.ClientV2(api_key=os.getenv('COHERE_API_KEY'))

def get_image_embedding(query: Union[str, Image.Image]) -> List[float]:
    """
    Generate Cohere embed-v4.0 vector for a text or image query.
    Accepts either a string (text) or PIL.Image.
    """
    if isinstance(query, str):
        response = co.embed(
            model="embed-v4.0",
            texts=[query],
            input_type="search_query",
            embedding_types=["float"],
        )
        return response.embeddings.float[0]

    elif isinstance(query, Image.Image):
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
            query.save(tmp_file.name, format="JPEG")
            with open(tmp_file.name, "rb") as f:
                enc_img = base64.b64encode(f.read()).decode("utf-8")
                enc_img = f"data:image/jpeg;base64,{enc_img}"

        response = co.embed(
            model="embed-v4.0",
            images=[enc_img],
            input_type="image",
            embedding_types=["float"],
        )
        return response.embeddings.float[0]

    else:
        raise ValueError("Unsupported query type for embedding: must be str or PIL.Image")



def graphics_retriever(query: Optional[str] = None, query_image: Optional[Image.Image] = None,
                       drive=None, k: int = 5, filters=None) -> List[Dict[str, any]]:
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

    print("Loading Chroma DBs...")
    embedding_function = get_embedding_model()
    central_folder_id = '1w5gJD_ALnqbRwl9XH0xTI0wr66IZmGL2'
    dbs = load_central_chroma_db(embedding_function, drive, central_folder_id)

    seen_phashes = set()
    combined_results = []

    # === Image Input ===
    if not query_image and query:
        if query.startswith("http") and query.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp")):
            print("Detected image URL, downloading...")
            try:
                response = requests.get(query)
                if response.status_code == 200:
                    query_image = Image.open(BytesIO(response.content)).convert("RGB")
                    print("Loaded image from URL.")
                else:
                    print("Failed to download image from URL.")
                    return []
            except Exception as e:
                print(f"Error downloading image: {e}")
                return []

        elif os.path.exists(query):
            print("Detected local image path, loading...")
            try:
                query_image = Image.open(query).convert("RGB")
                print("Loaded image from local file.")
            except Exception as e:
                print(f"Error loading image file: {e}")
                return []

    # === Search via Image Embedding ===
    if query_image:
        print("Searching image collection using image embedding...")
        try:
            query_vector = get_image_embedding(query_image)
        except Exception as e:
            print(f"Failed to embed image: {e}")
            return []

        image_db = dbs["image"]
        raw_image_results = image_db._collection.query(
            query_embeddings=[query_vector],
            n_results=50,
            include=["metadatas", "distances"]
        )
        
        for metadata, distance in zip(raw_image_results["metadatas"][0], raw_image_results["distances"][0]):
            if 'image_id' not in metadata or not match_filters(metadata):
                continue
            combined_results.append({
                "similarity": distance,
                "metadata": metadata,
                "source": "image"
            })

    # === Search via Text Embedding ===
    if query:
        print("Searching both collections using text query...")
        try:
            # === Search text DB ===
            text_db = dbs["text"]
            text_results = text_db.similarity_search_with_score(query, k=50)
            print(f"[INFO] Text DB returned {len(text_results)} results")
            for doc, score in text_results:
                if 'image_id' not in doc.metadata or not match_filters(doc.metadata):
                    continue
                combined_results.append({
                    "similarity": score,
                    "metadata": doc.metadata,
                    "source": "text"
                })

            # === Search image DB with CLIP text embedding ===
            print("Searching image collection with CLIP text embedding...")
            clip_vector = get_image_embedding(query)
            image_db = dbs["image"]
            clip_results = image_db._collection.query(
                query_embeddings=[clip_vector],
                n_results=50,
                include=["metadatas", "distances"]
            )
            for metadata, distance in zip(clip_results["metadatas"][0], clip_results["distances"][0]):
                if 'image_id' not in metadata or not match_filters(metadata):
                    continue
                combined_results.append({
                    "similarity": distance,
                    "metadata": metadata,
                    "source": "image"
                })

        except Exception as e:
            print(f"Error during text embedding search: {e}")
            return []

    # === Sort and Deduplicate ===
    print("Ranking and filtering results...")
    results = []
    for result in sorted(combined_results, key=lambda x: x["similarity"]):
        metadata = result["metadata"]
        phash = metadata.get("phash")
        if phash and phash in seen_phashes:
            continue
        if phash:
            seen_phashes.add(phash)

        try:
            image_file_id = metadata["image_id"]
            pil_image = download_image_from_drive(drive, image_file_id)
            if not pil_image:
                continue

            results.append({
                "similarity": result["similarity"],
                "image": pil_image,
                "metadata": {
                    **metadata,
                    "source": result["source"]
                }
            })

            if len(results) >= k:
                break

        except Exception as e:
            print(f"Failed to load image {metadata.get('image_id')}: {e}")
            continue

    print(f"Returning {len(results)} results.")
    return results

