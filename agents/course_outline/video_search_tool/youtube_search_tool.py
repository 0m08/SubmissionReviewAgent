import os
from utils.decorator_helpers import cycle_api_keys_decorator


gcloud_yt_search_api_keys = [
    os.environ.get("GCLOUD_YT_SEARCH_API_KEY_1"),
    os.environ.get("GCLOUD_YT_SEARCH_API_KEY_2"),
    os.environ.get("GCLOUD_YT_SEARCH_API_KEY_3")
]

@cycle_api_keys_decorator(gcloud_yt_search_api_keys)
def search_youtube_videos(query, max_results=5, channel_id=None, developer_key=None, force_hvac=True):
    """
    Searches YouTube for HVAC-related video links based on a query.

    Args:
        query (str): The search query (will be augmented with 'HVAC' if force_hvac=True).
        max_results (int): Number of video links to return.
        channel_id (str, optional): Restrict results to a specific channel ID.
        developer_key (str, optional): YouTube Data API key.
        force_hvac (bool): If True, appends 'HVAC' to the query for better relevance.

    Returns:
        list of dict: Video details with id, url, title, description, channel, published_at.
    """
    from googleapiclient.discovery import build

    youtube = build("youtube", "v3", developerKey=developer_key)

    # 🔹 Ensure HVAC focus by appending keywords
    hvac_keywords = ["HVAC", "air conditioning", "heating", "ventilation", "refrigeration"]
    if force_hvac and not channel_id:
        # Append HVAC to query if not already present
        hvac_query = query
        if not any(k.lower() in query.lower() for k in hvac_keywords):
            hvac_query = f"{query} HVAC"
    else:
        hvac_query = query

    # Prepare search parameters
    search_params = {
        "q": hvac_query,
        "part": "snippet",
        "type": "video",
        "maxResults": max_results,
    }
    if channel_id:
        search_params["channelId"] = channel_id

    # Perform the search
    search_response = youtube.search().list(**search_params).execute()

    video_details = []
    for item in search_response.get("items", []):
        video_id = item["id"]["videoId"]
        snippet = item["snippet"]
        video_detail = {
            "search_query": hvac_query,
            "video_id": video_id,
            "video_url": f"https://www.youtube.com/watch?v={video_id}",
            "title": snippet.get("title"),
            "description": snippet.get("description"),
            "channel_title": snippet.get("channelTitle"),
            "published_at": snippet.get("publishedAt"),
        }
        video_details.append(video_detail)

    return video_details
