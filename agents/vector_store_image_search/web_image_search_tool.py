from google_images_search import GoogleImagesSearch
from typing import List, Dict, Any, Optional
import os
from PIL import Image
import requests
from io import BytesIO
import uuid
from bs4 import BeautifulSoup

API_KEY = os.getenv("GOOGLE_CSE_API_KEY")
CSE_ID = os.getenv("GOOGLE_CSE_ID")


def fetch_page_title(url: Optional[str], timeout: int = 5) -> Optional[str]:
    if not url:
        return None

    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0 Safari/537.36"
            )
        }

        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        if soup.title and soup.title.string:
            return soup.title.string.strip()

    except Exception:
        pass

    return None

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
        #'imgType': 'photo',
        #'imgSize': 'medium',
        #'rights': 'cc_publicdomain|cc_attribute',
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

            # 1. Try to fetch title from referrer page
            actual_title = fetch_page_title(image.referrer_url)

            # 2. Fallback to filename from image URL
            if not actual_title:
                actual_title = image.url.split("/")[-1].split("?")[0]
 
            # 3. Final fallback to UUID
            if not actual_title or not actual_title.strip():
                actual_title = f"image_{uuid.uuid4().hex[:8]}"
            
            # Get description if available
            description = getattr(image, 'description', None) or getattr(image, 'snippet', None) or ""

            results.append({
                "image": pil_image,
                "metadata": {
                    "name": actual_title,
                    "description": description,
                    "drive_url": image.referrer_url or image.url,
                    "source_url": image.url
                }
            })
        except Exception as e:
            print(f"[WARNING] Failed to load image {image.url}: {e}")

    print(f"[INFO] Fetched {len(results)} valid image(s) for query: '{query}'")

    # Optionally post-filter using `query_image` similarity (if needed in future)
    return results