import cohere
import os
import requests
from io import BytesIO
from PIL import Image
import imagehash
from services.embedding_service import get_embedding_model
from agents.vector_store_image_search.create_vectorstore import download_image_from_drive
from langchain_chroma import Chroma
from services.drive_service import download_folder_from_drive
from typing import Optional, List, Dict, Union
import base64
import hashlib
import tempfile


def load_central_chroma_db(embedding_function, drive, root_folder_id):
    """Load the shared Chroma DB collections from Google Drive."""

    # Create version-specific local path to avoid cache conflicts
    local_chroma_root = "/tmp/temp_chroma_folder"
    local_chroma_path = os.path.join(local_chroma_root, f"chroma_graphics_db_{root_folder_id}")
    os.makedirs(local_chroma_root, exist_ok=True)

    # Locate Vectorstore files folder
    vectorstore_list = drive.ListFile({
        'q': f"title='Vectorstore files' and '{root_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
    }).GetList()
    if not vectorstore_list:
        raise FileNotFoundError("'Vectorstore files' folder not found in Drive.")
    vectorstore_folder_id = vectorstore_list[0]['id']

    # Search for chroma db folder
    print("Searching for 'chroma_graphics_db' in Drive...")
    file_list = drive.ListFile({
        'q': f"title='chroma_graphics_db' and '{vectorstore_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
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

def _safe_iter_clip_results(clip_results):
    """Return an iterator over (metadata, distance) or an empty list safely."""
    if not clip_results:
        print("⚠️ clip_results is None/empty")
        return []
    metadatas = clip_results.get("metadatas")
    distances = clip_results.get("distances")

    if not metadatas or not distances:
        print(f"⚠️ clip_results missing metadatas/distances. "
              f"metadatas={type(metadatas)}, distances={type(distances)}")
        return []

    if not isinstance(metadatas, list) or not metadatas or not isinstance(metadatas[0], list):
        print("⚠️ clip_results['metadatas'] malformed:", type(metadatas), metadatas)
        return []
    if not isinstance(distances, list) or not distances or not isinstance(distances[0], list):
        print("⚠️ clip_results['distances'] malformed:", type(distances), distances)
        return []

    return zip(metadatas[0], distances[0])


def graphics_retriever(query: Optional[str] = None, query_image: Optional[Image.Image] = None,
                       drive=None, k: int = 5, filters=None,
                       root_folder_id: str = '1QS6PmCESfgFWNNEpRJDUB0E-t8iMAatH') -> List[Dict[str, any]]:
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
        if filters.get("course_name") and metadata.get("course_name") and \
                filters["course_name"].lower() not in metadata["course_name"].lower():
            return False
        if filters.get("topic_name") and metadata.get("topic_name") and \
                filters["topic_name"].lower() not in metadata["topic_name"].lower():
            return False
        if filters.get("stock_type") and metadata.get("stock_type") and \
                metadata["stock_type"].lower() != filters["stock_type"].lower():
            return False
        return True

    print("Loading Chroma DBs...")
    embedding_function = get_embedding_model()
    dbs = load_central_chroma_db(embedding_function, drive, root_folder_id)

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
            image_db = dbs["image"]
            try:
                print("Sample vector from image DB:")
                sample_doc = image_db._collection.peek()
                print("✅ Vector length in DB:", len(sample_doc["embeddings"][0]))
            except Exception as e:
                print("❌ Could not inspect vector size in DB:", e)

            raw_image_results = image_db._collection.query(
                query_embeddings=[query_vector],
                n_results=50,
                include=["metadatas", "distances"]
            )
           

            for metadata, distance in _safe_iter_clip_results(raw_image_results):
                if 'image_id' not in metadata or not match_filters(metadata):
                    continue
                combined_results.append({
                    "similarity": distance,
                    "metadata": metadata,
                    "source": "image"
                })
        except Exception as e:
            print(f"Failed image embedding search: {e}")
            return []

    # === Search via Text Embedding ===
    if query:
        print("Searching both collections using text query...")

        try:
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
        except Exception as e:
            print(f"❌ Text DB similarity_search_with_score failed: {e}")

        try:
            print("Searching image collection with CLIP text embedding...")
            clip_vector = get_image_embedding(query)

            if not clip_vector or not isinstance(clip_vector, list):
                print("❌ clip_vector is None/invalid. Skipping CLIP search.")
            else:
                print(f"✅ clip_vector length: {len(clip_vector)} (first 3: {clip_vector[:3]})")
                image_db = dbs["image"]

                # === Safe query with internal error catching ===
                clip_results = None
                try:
                    clip_results = image_db._collection.query(
                        query_embeddings=[clip_vector],
                        n_results=50,
                        include=["metadatas", "distances"]
                    )
                except Exception as query_error:
                    print(f"❌ Chroma .query() failed: {query_error}")

                if clip_results is None:
                    print("⚠️ clip_results is None — Chroma query failed or returned nothing.")
                else:
                    for metadata, distance in _safe_iter_clip_results(clip_results):
                        if 'image_id' not in metadata or not match_filters(metadata):
                            continue
                        combined_results.append({
                            "similarity": distance,
                            "metadata": metadata,
                            "source": "image"
                        })

        except Exception as e:
            print(f"❌ CLIP text embedding→image search failed: {e}")

    # === Sort and Deduplicate Results ===
    print("Ranking and filtering results...")
    results = []
    seen_phashes = set()
    seen_content_hashes = set()
    seen_image_ids = set()
    seen_drive_urls = set()
    seen_filenames = set()
    seen_topics = set()
    
    # Sort by similarity first
    sorted_results = sorted(combined_results, key=lambda x: x['similarity'])
    
    for i, result in enumerate(sorted_results):
        metadata = result['metadata']
        
        image_id = metadata.get('image_id')
        if image_id and image_id in seen_image_ids:
            continue
        
        drive_url = metadata.get('drive_url')
        if drive_url and drive_url in seen_drive_urls:
            continue
        
        filename = metadata.get('name') or metadata.get('image_title')
        if filename and filename in seen_filenames:
            print(f"Skipping duplicate by filename: {filename}")
            continue
        
        existing_phash = metadata.get('phash')
        if existing_phash and existing_phash in seen_phashes:
            continue
        
        # Check topic diversity for v2 (when topic_name is available)
        # topic_name = metadata.get("topic_name")
        # if topic_name:
        #     # Count how many results we already have from this topic
        #     current_topic_count = sum(1 for r in results if r["metadata"].get("topic_name") == topic_name)
        #     
        #     # Allow maximum 1 result per topic to ensure maximum diversity
        #     if current_topic_count >= 1:
        #         continue
        #     
        #     seen_topics.add(topic_name)
        
        try:
            if not image_id:
                continue
        
            pil_image = download_image_from_drive(drive, image_id)
            if not pil_image:
                continue
        
            phash_value = existing_phash
            if not phash_value:
                try:
                    phash_value = str(imagehash.phash(pil_image))
                except Exception as hash_error:
                    print(f"Failed to compute pHash for {image_id}: {hash_error}")
                    phash_value = None
        
            if phash_value:
                if phash_value in seen_phashes:
                    continue
                seen_phashes.add(phash_value)
                metadata.setdefault('phash', phash_value)
            else:
                try:
                    content_signature = hashlib.md5(pil_image.tobytes()).hexdigest()
                except Exception as digest_error:
                    print(f"Failed to fingerprint image {image_id}: {digest_error}")
                    content_signature = None
        
                if content_signature:
                    if content_signature in seen_content_hashes:
                        continue
                    seen_content_hashes.add(content_signature)
        
            results.append({
                "similarity": result["similarity"],
                "image": pil_image,
                "metadata": {
                    **metadata,
                    "source": result["source"]
                }
            })
        
            if image_id:
                seen_image_ids.add(image_id)
            if drive_url:
                seen_drive_urls.add(drive_url)
            if filename:
                seen_filenames.add(filename)
        
            if len(results) >= k:
                break
        
        except Exception as e:
            print(f"Failed to load image {metadata.get('image_id')}: {e}")
            continue

    # Debug: Show topic distribution
    # if results:
    #     topic_counts = {}
    #     for result in results:
    #         topic = result["metadata"].get("topic_name", "Unknown")
    #         topic_counts[topic] = topic_counts.get(topic, 0) + 1
    #     print(f"Topic distribution: {topic_counts}")
    
    print(f"Returning {len(results)} results.")
    return results

