from google_images_search import GoogleImagesSearch
from typing import List, Dict, Any

from dotenv import load_dotenv
import os

from PIL import Image
import requests
from io import BytesIO

load_dotenv()

API_KEY = os.getenv("GOOGLE_CSE_API_KEY")
CSE_ID = os.getenv("GOOGLE_CSE_ID")


def web_image_search_tool(query: str, k: int = 5) -> List[Dict[str, Any]]:
    gis = GoogleImagesSearch(API_KEY, CSE_ID)

    search_params = {
        'q': query,
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
                "image": pil_image,  # ✅ PIL image object
                "metadata": {
                    "name": f"{query.title()} Image {idx+1}",
                    "drive_url": image.referrer_url or image.url,
                    "source_url": image.url
                }
            })
        except Exception as e:
            print(f"[WARNING] Failed to load image {image.url}: {e}")

    print(f"[INFO] Fetched {len(results)} valid image(s) for query: '{query}'")
    return results



# UNSPLASH_ACCESS_KEY = os.getenv("UNSPLASH_ACCESS_KEY")

# def web_image_search_tool(query: str, k: int = 5):
#     url = 'https://api.unsplash.com/search/photos'
#     headers = {
#         'Accept-Version': 'v1',
#         'Authorization': f'Client-ID {UNSPLASH_ACCESS_KEY}'
#     }
#     params = {
#         'query': query,
#         'per_page': k
#     }

#     response = requests.get(url, headers=headers, params=params)
#     if response.status_code != 200:
#         print(f"[ERROR] Failed to fetch images: {response.status_code}")
#         return []

#     data = response.json()
#     results = []
#     for i, result in enumerate(data.get('results', [])):
#         image_url = result['urls']['regular']
#         image_title = result['alt_description'] or f"{query.title()} Image {i+1}"
#         results.append({
#             "image": image_url,
#             "metadata": {
#                 "name": image_title,
#                 "author": result['user']['name'],
#                 "link": result['links']['html']
#             }
#         })

#     return results