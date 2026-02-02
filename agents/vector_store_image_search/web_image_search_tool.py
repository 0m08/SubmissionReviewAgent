from google_images_search import GoogleImagesSearch
from typing import List, Dict, Any, Optional
import os
from PIL import Image
import requests
from io import BytesIO
import uuid
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import time
import threading
import base64
import re

# Load environment variables
load_dotenv()

# Rate limiting for Brave Search API (free tier: 1 request per second)
_brave_rate_limiter_lock = threading.Lock()
_brave_last_request_time = 0.0

API_KEY = os.getenv("GOOGLE_CSE_API_KEY")
CSE_ID = os.getenv("GOOGLE_CSE_ID")
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY")
SERP_API_KEY = os.getenv("SERP_API_KEY")


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

def _search_with_google_custom_search(query: str, query_image: Optional[Image.Image], k: int) -> List[Dict[str, Any]]:
    """Try Google Custom Search API first."""
    api_key = os.getenv("GOOGLE_CSE_API_KEY")
    cse_id = os.getenv("GOOGLE_CSE_ID")
    
    if not api_key or not cse_id:
        raise ValueError("Google Custom Search API credentials not configured.")
    
    gis = GoogleImagesSearch(api_key, cse_id)
    search_params = {
        'q': query,
        'query_image': query_image,
        'num': k,
        'fileType': 'jpg|png',
        'safe': 'high',
    }
    
    gis.search(search_params=search_params)
    return list(gis.results())


def _extract_original_url_from_brave_proxy(proxy_url: str) -> Optional[str]:
    """
    Extract the original source URL from Brave's image proxy URL.
    Brave proxy URLs format: https://imgs.search.brave.com/ID/rs:fit:.../g:ce/BASE64_ENCODED_URL
    The original URL is Base64 encoded after /g:ce/
    
    Returns the original URL to get full quality, or None if extraction fails (will use proxy URL = 500px quality).
    """
    if not proxy_url or 'imgs.search.brave.com' not in proxy_url:
        return None
    
    try:
        # Find the /g:ce/ part and extract everything after it
        match = re.search(r'/g:ce/(.+)', proxy_url)
        if not match:
            return None
        
        # Get the Base64 encoded part (may be split by /)
        encoded_parts = match.group(1).split('/')
        
        # Try to decode - the URL might be split across multiple path segments
        decoded_urls = []
        for part in encoded_parts:
            try:
                # Add padding if needed (Base64 requires length to be multiple of 4)
                padding = 4 - (len(part) % 4)
                if padding != 4:
                    part += '=' * padding
                decoded = base64.urlsafe_b64decode(part).decode('utf-8')
                decoded_urls.append(decoded)
            except:
                continue
        
        # If we got multiple decoded parts, try to combine them
        if decoded_urls:
            # Usually the full URL is in one part, but if split, combine
            original_url = ''.join(decoded_urls)
            # Validate it's a proper URL
            if original_url.startswith('http://') or original_url.startswith('https://'):
                return original_url
    except Exception as e:
        # If decoding fails, return None to use proxy URL as fallback
        # Note: This means we'll get 500px quality instead of original
        pass
    
    return None


def _sanitize_query_for_brave(query: str, max_length: int = 200) -> str:
    """Sanitize and truncate query for Brave Search API to avoid 422 errors."""
    # Remove or replace problematic characters
    sanitized = query.strip()
    
    # Remove any control characters
    sanitized = ''.join(char for char in sanitized if ord(char) >= 32 or char in '\n\r\t')
    
    # Truncate if too long (Brave API may have query length limits)
    if len(sanitized) > max_length:
        # Try to truncate at word boundary
        truncated = sanitized[:max_length].rsplit(' ', 1)[0]
        if len(truncated) < max_length * 0.8:  # If truncation removed too much, just cut at max_length
            truncated = sanitized[:max_length]
        sanitized = truncated
        print(f"[WARNING] Query truncated from {len(query)} to {len(sanitized)} characters: '{query[:50]}...' -> '{sanitized[:50]}...'")
    
    return sanitized


def _search_with_brave(query: str, k: int, max_retries: int = 2) -> List[Dict[str, Any]]:
    """Fallback to Brave Search API when Google Custom Search fails."""
    global _brave_last_request_time
    
    brave_api_key = os.getenv("BRAVE_API_KEY")
    
    if not brave_api_key:
        raise ValueError("BRAVE_API_KEY environment variable is not set. Please check your .env file.")
    
    # Sanitize query to avoid 422 errors
    sanitized_query = _sanitize_query_for_brave(query)
    
    # Retry logic for rate limiting and transient errors
    for attempt in range(max_retries + 1):
        # Rate limiting: Free tier allows 1 request per second
        with _brave_rate_limiter_lock:
            current_time = time.time()
            time_since_last_request = current_time - _brave_last_request_time
            if time_since_last_request < 1.0:
                sleep_time = 1.0 - time_since_last_request
                print(f"[INFO] Rate limiting: waiting {sleep_time:.2f}s before next Brave API request...")
                time.sleep(sleep_time)
            _brave_last_request_time = time.time()
        
        # Brave Search API endpoint for image search
        url = "https://api.search.brave.com/res/v1/images/search"
        headers = {
            "X-Subscription-Token": brave_api_key,
            "Accept": "application/json"
        }
        params = {
            "q": sanitized_query,
            "count": k,
            "safesearch": "strict"  # Brave API only accepts 'off' or 'strict', not 'moderate'
        }
        
        try:
            response = requests.get(url, headers=headers, params=params, timeout=30)
            
            # Handle 429 (Too Many Requests) with exponential backoff
            if response.status_code == 429:
                if attempt < max_retries:
                    wait_time = (2 ** attempt) * 2  # 2s, 4s, 8s...
                    print(f"[WARNING] Rate limit hit (429), waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"[ERROR] Rate limit exceeded after {max_retries} retries")
                    response.raise_for_status()
            
            # Handle 422 (Unprocessable Entity) - query might be invalid
            if response.status_code == 422:
                error_msg = f"422 Unprocessable Entity - Query may be invalid or too long"
                try:
                    error_data = response.json()
                    error_msg += f": {error_data}"
                except:
                    error_msg += f" (original query length: {len(query)}, sanitized length: {len(sanitized_query)})"
                print(f"[ERROR] {error_msg}")
                # Don't retry 422 errors - they're likely due to query format
                raise requests.exceptions.HTTPError(error_msg, response=response)
            
            # For other errors, raise immediately
            response.raise_for_status()
            data = response.json()
            break  # Success, exit retry loop
            
        except requests.exceptions.HTTPError as e:
            if attempt < max_retries and e.response and e.response.status_code == 429:
                continue  # Will retry in next iteration
            raise  # Re-raise if not retryable or out of retries
    
    # Extract image results from Brave API response
    images = data.get("results", [])
    results = []
    seen_urls = set()  # Deduplicate by URL within Brave results
    
    for img in images[:k]:
        try:
            # Brave API image search returns:
            # - thumbnail.src: actual image URL (prioritize this)
            # - url: page URL where image is hosted
            # - properties.url: may contain image or page URL
            
            # Prioritize thumbnail.src as it's the actual image URL
            thumbnail = img.get("thumbnail", {})
            if isinstance(thumbnail, dict):
                image_url = thumbnail.get("src")
            else:
                image_url = None
            
            # Fallback to other fields if thumbnail.src not available
            if not image_url:
                # Check if url field is actually an image (has image extension)
                url_field = img.get("url", "")
                if url_field and any(url_field.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.svg']):
                    image_url = url_field
                else:
                    # url is likely a page URL, try properties.url
                    properties = img.get("properties", {})
                    if isinstance(properties, dict):
                        prop_url = properties.get("url", "")
                        if prop_url and any(prop_url.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.svg']):
                            image_url = prop_url
                        else:
                            # Last resort: use url field even if it might be a page
                            image_url = url_field
            
            if not image_url:
                continue
            
            # Try to extract original URL from Brave proxy URL for cleaner, shorter URLs and original quality
            original_url = _extract_original_url_from_brave_proxy(image_url)
            if original_url:
                image_url = original_url
                print(f"[INFO] Extracted original URL from Brave proxy (original quality): {original_url[:80]}...")
            elif 'imgs.search.brave.com' in image_url:
                # Extraction failed - we'll use proxy URL which is resized to 500px (lower quality)
                print(f"[WARNING] Could not extract original URL from Brave proxy, using proxy URL (500px quality): {image_url[:80]}...")
            
            # Deduplicate by URL (normalize URL by stripping)
            normalized_url = image_url.strip()
            if normalized_url in seen_urls:
                continue
            seen_urls.add(normalized_url)
            
            # Get referrer URL (source page) - this is the page URL, not the image URL
            referrer_url = (
                img.get("url") or  # Page URL where image is hosted
                img.get("properties", {}).get("url") if isinstance(img.get("properties"), dict) else None or
                image_url  # Fallback
            )
            
            # Get title/description
            title = (
                img.get("title") or 
                img.get("properties", {}).get("title") if isinstance(img.get("properties"), dict) else None or
                ""
            )
            description = (
                img.get("description") or 
                img.get("properties", {}).get("description") if isinstance(img.get("properties"), dict) else None or
                ""
            )
            
            # Create a mock object similar to GoogleImagesSearch result
            class BraveImage:
                def __init__(self, img_data):
                    self.url = image_url
                    self.referrer_url = referrer_url
                    self.description = title or description
                    self.snippet = description
            
            results.append(BraveImage(img))
        except Exception as e:
            print(f"[WARNING] Failed to parse Brave image result: {e}")
    
    return results


def _search_with_serpapi(query: str, k: int) -> List[Dict[str, Any]]:
    """Fallback to SerpApi when Google Custom Search fails."""
    serp_api_key = os.getenv("SERP_API_KEY")
    
    if not serp_api_key:
        raise ValueError("SERP_API_KEY environment variable is not set. Please check your .env file.")
    
    # SerpApi endpoint for Google Images
    url = "https://serpapi.com/search.json"
    params = {
        "engine": "google_images",
        "q": query,
        "api_key": serp_api_key,
        "num": k,
        "safe": "active",
        "ijn": 0  # Image result page number
    }
    
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()
    
    # Extract image results from SerpApi response
    images = data.get("images_results", [])
    results = []
    seen_urls = set()  # Deduplicate by URL within SerpApi results
    
    for img in images[:k]:
        try:
            # SerpApi returns different structure
            image_url = img.get("original") or img.get("link") or img.get("url")
            if not image_url:
                continue
            
            # Deduplicate by URL (normalize URL by stripping)
            normalized_url = image_url.strip()
            if normalized_url in seen_urls:
                continue
            seen_urls.add(normalized_url)
                
            # Create a mock object similar to GoogleImagesSearch result
            class SerpApiImage:
                def __init__(self, img_data):
                    self.url = image_url
                    self.referrer_url = img_data.get("link", "")
                    self.description = img_data.get("title", "") or img_data.get("snippet", "")
                    self.snippet = img_data.get("snippet", "")
            
            results.append(SerpApiImage(img))
        except Exception as e:
            print(f"[WARNING] Failed to parse SerpApi image result: {e}")
    
    return results


def _is_image_url(url: str) -> bool:
    """Check if URL appears to be a direct image URL."""
    if not url:
        return False
    url_lower = url.lower()
    # Check for image extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.svg']
    if any(url_lower.endswith(ext) for ext in image_extensions):
        return True
    # Check for common image URL patterns
    if '/image' in url_lower or '/img' in url_lower or '/photo' in url_lower:
        # But exclude page URLs that contain these words
        if '/blog/' in url_lower or '/article/' in url_lower or '/post/' in url_lower:
            return False
        return True
    return False


def _process_image_results(image_results, query: str) -> List[Dict[str, Any]]:
    """Process image results (from Google Custom Search, Brave Search API, or SerpApi) into standard format."""
    results = []
    for idx, image in enumerate(image_results):
        try:
            # Skip if URL doesn't look like a direct image URL
            if not _is_image_url(image.url):
                print(f"[WARNING] Skipping non-image URL: {image.url}")
                continue
                
            print(f"Downloading image {idx + 1}: {image.url}")
            response = requests.get(image.url, timeout=10, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            })
            response.raise_for_status()
            
            # Check if response is actually an image
            content_type = response.headers.get('content-type', '').lower()
            if not content_type.startswith('image/'):
                print(f"[WARNING] URL returned non-image content type: {content_type}")
                continue
                
            pil_image = Image.open(BytesIO(response.content)).convert("RGB")

            # 1. Try to fetch title from referrer page
            actual_title = fetch_page_title(getattr(image, 'referrer_url', None) or "")

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
                    "drive_url": getattr(image, 'referrer_url', None) or image.url,
                    "source_url": image.url
                }
            })
        except Exception as e:
            print(f"[WARNING] Failed to load image {image.url}: {e}")

    return results


def web_image_search_tool(
    query: Optional[str] = None,
    query_image: Optional[Image.Image] = None,
    k: int = 5
) -> List[Dict[str, Any]]:
    if not query:
        raise ValueError("A text query is required for web image search.")

    # TEMPORARILY DISABLED: Google Custom Search - using only Brave Search API
    # # Try Google Custom Search first (primary)
    # try:
    #     print(f"[INFO] Attempting Google Custom Search for query: '{query}'")
    #     image_results = _search_with_google_custom_search(query, query_image, k)
    #     if image_results:
    #         print(f"[INFO] Google Custom Search succeeded, found {len(image_results)} result(s)")
    #         results = _process_image_results(image_results, query)
    #         print(f"[INFO] Fetched {len(results)} valid image(s) for query: '{query}'")
    #         return results
    # except Exception as e:
    #     print(f"[WARNING] Google Custom Search failed: {e}")
    #     print(f"[INFO] Falling back to Brave Search API...")
    
    # Use Brave Search API (primary - temporarily)
    try:
        brave_api_key = os.getenv("BRAVE_API_KEY")
        if not brave_api_key:
            print(f"[ERROR] Brave Search API requested but BRAVE_API_KEY not configured.")
            return []
        
        print(f"[INFO] Using Brave Search API for query: '{query}'")
        image_results = _search_with_brave(query, k)
        if image_results:
            print(f"[INFO] Brave Search API succeeded, found {len(image_results)} result(s)")
            results = _process_image_results(image_results, query)
            print(f"[INFO] Fetched {len(results)} valid image(s) for query: '{query}'")
            return results
        else:
            print(f"[WARNING] Brave Search API returned no results")
            return []
    except Exception as e:
        print(f"[ERROR] Brave Search API failed: {e}")
        return []
    
    # TEMPORARILY DISABLED: SerpApi - using only Brave Search API
    # # Fallback 2: SerpApi if both Google Custom Search and Brave Search fail
    # try:
    #     serp_api_key = os.getenv("SERP_API_KEY")
    #     if not serp_api_key:
    #         print(f"[ERROR] SerpApi fallback requested but SERP_API_KEY not configured.")
    #         return []
    #     
    #     print(f"[INFO] Using SerpApi for query: '{query}'")
    #     image_results = _search_with_serpapi(query, k)
    #     if image_results:
    #         print(f"[INFO] SerpApi succeeded, found {len(image_results)} result(s)")
    #         results = _process_image_results(image_results, query)
    #         print(f"[INFO] Fetched {len(results)} valid image(s) for query: '{query}'")
    #         return results
    # except Exception as e:
    #     print(f"[ERROR] SerpApi fallback also failed: {e}")
    #     return []
    
    print(f"[WARNING] Brave Search API failed for query: '{query}'")
    return []