from google_images_search import GoogleImagesSearch
from typing import List, Dict, Any, Optional
import os
from PIL import Image
import requests
from io import BytesIO
import requests
from PIL import Image
from io import BytesIO
from typing import Optional, List, Dict, Any

API_KEY = os.getenv("GOOGLE_API_KEY")
CSE_ID = os.getenv("GOOGLE_CSE_ID")

def web_image_search_tool(
    query: Optional[str] = None,
    query_image: Optional[Image.Image] = None,
    k: int = 5
) -> List[Dict[str, Any]]:
    if not query:
        raise ValueError("A text query is required for web image search.")

    gis = GoogleImagesSearch(API_KEY, CSE_ID)

    search_params = {
        'q': query, 
        'query_image':query_image,
        'num': k,
        'fileType': 'jpg|png',
        'safe': 'high',
        'imgType': 'photo',
        'imgSize': 'medium',
        'rights': 'cc_publicdomain|cc_attribute',
    }

    try:
        gis.search(search_params=search_params)
    except Exception as e:
        print(f"[ERROR] Google Image Search failed: {e}")
        return []

    results = []
    for idx, image in enumerate(gis.results()):
        try:
            print(f"Downloading image {idx + 1}: {image.url}")
            response = requests.get(image.url, timeout=10)
            response.raise_for_status()
            pil_image = Image.open(BytesIO(response.content)).convert("RGB")

            results.append({
                "image": pil_image,
                "metadata": {
                    "name": f"{query.title()} Image {idx+1}",
                    "drive_url": image.referrer_url or image.url,
                    "source_url": image.url
                }
            })
        except Exception as e:
            print(f"[WARNING] Failed to load image {image.url}: {e}")

    print(f"[INFO] Fetched {len(results)} valid image(s) for query: '{query}'")

    # Optionally post-filter using `query_image` similarity (if needed in future)
    return results