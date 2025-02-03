from googleapiclient.discovery import build
import os

gcloud_yt_search_api_key = os.environ.get("GCLOUD_YT_SEARCH_API_KEY")

def search_youtube_videos(query, max_results=5, channel_id=None):
    """
    Searches YouTube for video links based on a query.

    Args:
        query (str): The search query.
        max_results (int): Number of video links to return.
        channel_id (str, optional): Filter results to a specific channel ID.

    Returns:
        list: A list of dictionaries containing video details.
    """
    youtube = build('youtube', 'v3', developerKey = gcloud_yt_search_api_key)

    # Prepare search parameters
    search_params = {
        'q': query,
        'part': 'snippet',
        'type': 'video',
        'maxResults': max_results
    }

    if channel_id:
        search_params['channelId'] = channel_id

    # Perform the search
    search_response = youtube.search().list(**search_params).execute()

    video_details = []

    # Extract video details from the response
    for item in search_response.get('items', []):
        video_id = item['id']['videoId']
        snippet = item['snippet']
        video_detail = {
            'search_query': query,
            'video_id': video_id,
            'video_url': f"https://www.youtube.com/watch?v={video_id}",
            'title': snippet.get('title'),
            'description': snippet.get('description'),
            'channel_title': snippet.get('channelTitle'),
            'published_at': snippet.get('publishedAt')
        }
        video_details.append(video_detail)

    return video_details


