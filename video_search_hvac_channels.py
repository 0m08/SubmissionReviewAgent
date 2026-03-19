"""
Video Search inside "HVAC School" and "Love2HVAC with Ty" YouTube channels
===========================================================================

Streamlit UI and backend for searching the multimodal video embeddings vectorstore
using text, image, or video as the query. Retrieves timestamped YouTube clips and
displays them with optional loop playback.

Requires: GDRIVE_SA_B64, VERTEX_AI_SA_B64 in environment. Uses the same vectorstore
as Graphics Definition V2 (HVAC School / Love2HVAC with Ty categories).
"""

import math
import os
import base64
import json
import tempfile
import time
import streamlit as st
from dotenv import load_dotenv
from pydrive2.drive import GoogleDrive

from services.drive_service import login_with_service_account
from agents.course_outline.video_search_tool.video_retriever import load_new_video_embeddings_chroma_db

load_dotenv()

# Vectorstore config (same as youtube_video_search_from_queries.py)
VIDEO_EMBEDDINGS_FOLDER_ID = "15H9thXq02JX3ldADSj1oD78mbV-fXfvu"
VIDEO_EMBEDDINGS_FOLDER_NAME = (
    "Vectorstore for HVAC school video embeddings (Category - 3D Animations and Simulations, "
    "Hands-On Field Work, Equipment Demos & Teardowns)"
)
VIDEO_EMBEDDING_DIM = 1408


def get_drive_instance():
    """
    Get Google Drive instance from session state or initialize from environment.

    :return: Google Drive instance or None if unavailable.
    """
    if "drive" in st.session_state:
        return st.session_state["drive"]
    try:
        key_bytes = base64.b64decode(os.environ["GDRIVE_SA_B64"])
        sa_json = key_bytes.decode()
        gauth = login_with_service_account(json_str=sa_json)
        gauth.ServiceAuth()
        drive = GoogleDrive(gauth)
        return drive
    except Exception as e:
        print(f"⚠️ Could not initialize Drive from environment: {e}")
        return None


def get_vertex_embedding_model():
    """
    Initialize Vertex AI and return the multimodal embedding model.

    :return: MultiModalEmbeddingModel instance.
    :raises Exception: If VERTEX_AI_SA_B64 is not set or init fails.
    """
    if "VERTEX_AI_SA_B64" not in os.environ:
        raise Exception("VERTEX_AI_SA_B64 environment variable not found. Add it to .env for image/video search.")
    key_bytes = base64.b64decode(os.environ["VERTEX_AI_SA_B64"])
    sa_json = key_bytes.decode()
    sa_dict = json.loads(sa_json)
    from google.oauth2 import service_account
    import vertexai
    from vertexai.vision_models import MultiModalEmbeddingModel

    creds = service_account.Credentials.from_service_account_info(sa_dict)
    vertexai.init(
        project="dam-images-tagging",
        location="us-central1",
        credentials=creds,
    )
    return MultiModalEmbeddingModel.from_pretrained("multimodalembedding@001")


def get_query_embedding_text(model, query):
    """
    Get 1408-dim embedding for a text query.

    :param model: Vertex MultiModalEmbeddingModel.
    :param query: Search query string.
    :return: List of floats (embedding vector).
    """
    result = model.get_embeddings(contextual_text=query, dimension=VIDEO_EMBEDDING_DIM)
    return result.text_embedding


def get_query_embedding_image(model, image_path_or_bytes):
    """
    Get 1408-dim embedding for an image (file path or bytes).

    :param model: Vertex MultiModalEmbeddingModel.
    :param image_path_or_bytes: Path to image file (str) or image bytes (bytes).
    :return: List of floats (embedding vector).
    """
    from vertexai.vision_models import Image

    if isinstance(image_path_or_bytes, bytes):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(image_path_or_bytes)
            path = f.name
        try:
            image = Image.load_from_file(path)
            result = model.get_embeddings(image=image, dimension=VIDEO_EMBEDDING_DIM)
            return result.image_embedding
        finally:
            try:
                os.unlink(path)
            except Exception:
                pass
    else:
        image = Image.load_from_file(image_path_or_bytes)
        result = model.get_embeddings(image=image, dimension=VIDEO_EMBEDDING_DIM)
        return result.image_embedding


def get_query_embedding_video(model, video_path_or_bytes):
    """
    Get 1408-dim embedding for a video (file path or bytes), using first segment.

    :param model: Vertex MultiModalEmbeddingModel.
    :param video_path_or_bytes: Path to video file (str) or video bytes (bytes).
    :return: List of floats (embedding vector).
    """
    from vertexai.vision_models import Video, VideoSegmentConfig

    if isinstance(video_path_or_bytes, bytes):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(video_path_or_bytes)
            path = f.name
        try:
            video = Video.load_from_file(path)
            result = model.get_embeddings(
                video=video,
                video_segment_config=VideoSegmentConfig(end_offset_sec=30),
            )
            if not result.video_embeddings:
                raise ValueError("No video embeddings returned")
            return result.video_embeddings[0].embedding
        finally:
            try:
                os.unlink(path)
            except Exception:
                pass
    else:
        video = Video.load_from_file(video_path_or_bytes)
        result = model.get_embeddings(
            video=video,
            video_segment_config=VideoSegmentConfig(end_offset_sec=30),
        )
        if not result.video_embeddings:
            raise ValueError("No video embeddings returned")
        return result.video_embeddings[0].embedding


def search_hvac_video_embeddings(drive, query_type, query, k, folder_id=None, folder_name=None):
    """
    Search the HVAC / Love2HVAC video embeddings vectorstore by text, image, or video.

    :param drive: Google Drive instance (for loading vectorstore from Drive).
    :param query_type: One of "text", "image", "video".
    :param query: For text: string. For image/video: file path (str) or bytes.
    :param k: Number of results to return.
    :param folder_id: Drive folder ID containing the vectorstore (default: VIDEO_EMBEDDINGS_FOLDER_ID).
    :param folder_name: Name of vectorstore folder in Drive (default: VIDEO_EMBEDDINGS_FOLDER_NAME).
    :return: List of dicts with keys: title, video_id, start_time, end_time, video_url, segment_index.
    """
    folder_id = folder_id or VIDEO_EMBEDDINGS_FOLDER_ID
    folder_name = folder_name or VIDEO_EMBEDDINGS_FOLDER_NAME

    chroma = load_new_video_embeddings_chroma_db(drive, folder_id, folder_name)
    collection = chroma.collection

    # Oversample so we have enough segments from many videos for diversity
    fetch_k = min(k * 5, 500)

    if query_type == "text":
        docs = chroma.similarity_search(query, k=fetch_k)
        raw = [_doc_to_result(doc) for doc in docs]
    else:
        model = get_vertex_embedding_model()
        if query_type == "image":
            query_embedding = get_query_embedding_image(model, query)
        else:
            query_embedding = get_query_embedding_video(model, query)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=fetch_k,
        )
        raw = _chroma_results_to_list(results)

    raw = _dedupe_results_by_segment(raw)
    return _apply_diversity_by_video(raw, k)


def _doc_to_result(doc):
    """
    Convert a LangChain Document from similarity_search to a result dict.

    :param doc: Document with metadata (video_id, start_time, end_time, title, video_url, etc.).
    :return: Dict with title, video_id, start_time, end_time, video_url, segment_index.
    """
    meta = getattr(doc, "metadata", {})
    return {
        "title": meta.get("title", "Unknown Video"),
        "video_id": meta.get("video_id"),
        "start_time": meta.get("start_time", 0),
        "end_time": meta.get("end_time"),
        "video_url": meta.get("video_url", ""),
        "segment_index": meta.get("segment_index", 0),
    }


def _dedupe_results_by_segment(results):
    """
    Deduplicate results by (video_id, start_time, end_time) so each segment appears at most once.
    Preserves order (first occurrence kept).

    :param results: List of result dicts with video_id, start_time, end_time.
    :return: Deduplicated list.
    """
    seen = set()
    out = []
    for r in results:
        key = (r.get("video_id"), r.get("start_time"), r.get("end_time"))
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def _apply_diversity_by_video(results, k):
    """
    Limit how many segments come from the same video so results span more distinct videos.
    max_per_video = max(1, ceil(k/4)). Walk results in order, add segment only if that
    video_id has fewer than max_per_video so far, until we have k segments.

    :param results: List of result dicts (ordered by relevance), each with video_id.
    :param k: Desired number of results to return.
    :return: List of at most k results, with at most max(1, ceil(k/4)) per video_id.
    """
    if not results or k <= 0:
        return []
    max_per_video = max(1, math.ceil(k / 4))
    count_by_video = {}
    out = []
    for r in results:
        if len(out) >= k:
            break
        vid = r.get("video_id")
        if vid is None:
            out.append(r)
            continue
        n = count_by_video.get(vid, 0)
        if n < max_per_video:
            count_by_video[vid] = n + 1
            out.append(r)
    return out


def _chroma_results_to_list(results):
    """
    Convert Chroma query results to list of result dicts for the UI.

    :param results: Dict from collection.query (ids, metadatas, documents).
    :return: List of dicts with title, video_id, start_time, end_time, video_url, segment_index.
    """
    out = []
    metadatas = (results.get("metadatas") or [None])[0]
    if not metadatas:
        return out
    for meta in metadatas:
        if not meta:
            continue
        out.append({
            "title": meta.get("title", "Unknown Video"),
            "video_id": meta.get("video_id"),
            "start_time": meta.get("start_time", 0),
            "end_time": meta.get("end_time"),
            "video_url": meta.get("video_url", ""),
            "segment_index": meta.get("segment_index", 0),
        })
    return out


def build_embed_url(video_id, start_time, end_time):
    """
    Build YouTube embed URL with start/end (plays once; user can use Play again to replay).

    :param video_id: YouTube video ID.
    :param start_time: Start time in seconds (int or float).
    :param end_time: End time in seconds (int, float, or None).
    :return: URL string for iframe src.
    """
    start_sec = int(start_time) if start_time is not None else 0
    end_sec = int(end_time) if end_time is not None else None
    base = f"https://www.youtube.com/embed/{video_id}"
    params = [f"start={start_sec}"]
    if end_sec is not None:
        params.append(f"end={end_sec}")
    params += ["modestbranding=1", "rel=0", "playsinline=1"]
    return base + "?" + "&".join(params)


# ============================================================
# Streamlit UI
# ============================================================


def render_hvac_visual_search(drive):
    """
    Render the HVAC School / Love2HVAC visual search UI (text, image, or video query).
    Can be used standalone (video_search_hvac_channels.py) or inside run_video_search_tool tabs.
    """
    st.markdown(
        "Search for relevant videos using **text**, **image**, or **video** as input query. "
    )
    st.markdown("---")

    query_type = st.radio(
        "Query type",
        options=["text", "image", "video"],
        horizontal=True,
        format_func=lambda x: {"text": "Text", "image": "Image", "video": "Video"}[x],
        key="hvac_tab_query_type",
    )

    # Clear previous results when user switches query type
    if st.session_state.get("hvac_search_query_type") != query_type:
        st.session_state.pop("hvac_search_results", None)
        st.session_state["hvac_search_query_type"] = query_type

    query_value = None
    if query_type == "text":
        query_value = st.text_input(
            "Search query",
            placeholder="e.g. refrigerant flow through evaporator",
            key="hvac_tab_text_query",
        )
    elif query_type == "image":
        uploaded = st.file_uploader(
            "Upload an image to search by visual content",
            type=["png", "jpg", "jpeg"],
            key="hvac_tab_image_upload",
        )
        if uploaded:
            query_value = uploaded.read()
            st.image(query_value, caption="Query image", use_container_width=True)
    else:
        uploaded = st.file_uploader(
            "Upload a video clip to search by visual content",
            type=["mp4", "webm"],
            key="hvac_tab_video_upload",
        )
        if uploaded:
            query_value = uploaded.read()
            st.video(query_value)

    num_results = st.number_input(
        "Number of videos to retrieve",
        min_value=1,
        max_value=100,
        value=5,
        step=1,
        key="hvac_tab_num_results",
    )

    if st.button("Search", type="primary", key="hvac_tab_search_btn"):
        if query_value is None or (query_type == "text" and not str(query_value).strip()):
            st.warning("Please provide a search query or upload an image/video.")
        else:
            with st.spinner("Searching for relevant videos..."):
                try:
                    results = search_hvac_video_embeddings(
                        drive=drive,
                        query_type=query_type,
                        query=query_value,
                        k=num_results,
                    )
                except Exception as e:
                    st.error(f"Search failed: {str(e)}")
                    results = []
            st.session_state["hvac_search_results"] = results

    results = st.session_state.get("hvac_search_results", [])
    reload_idx = st.session_state.pop("hvac_reload_iframe_idx", None)

    if results:
        st.subheader(f"Top {len(results)} results")
        for idx, item in enumerate(results, start=1):
            video_id = item.get("video_id")
            if not video_id:
                continue
            title = item.get("title", f"Video {idx}")
            start_time = item.get("start_time", 0)
            end_time = item.get("end_time")
            embed_url = build_embed_url(video_id, start_time, end_time)
            if reload_idx == idx:
                embed_url = embed_url + "&_=" + str(time.time()) + "&autoplay=1"
            embed_url_escaped = embed_url.replace("&", "&amp;").replace('"', "&quot;")

            st.markdown(
                f'<iframe src="{embed_url_escaped}" width="640" height="360" frameborder="0" '
                f'allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" '
                f'allowfullscreen></iframe>',
                unsafe_allow_html=True,
            )
            if st.button("Play again", key=f"hvac_play_again_{idx}"):
                st.session_state["hvac_reload_iframe_idx"] = idx
                st.rerun()
            st.markdown(f"**{title}**")
            start_sec = int(start_time) if start_time is not None else 0
            watch_url = f"https://www.youtube.com/watch?v={video_id}&t={start_sec}"
            st.caption(f"Segment: {start_time}s – {end_time or 'end'} | [Open on YouTube]({watch_url})")
            st.markdown("---")
    elif "hvac_search_results" in st.session_state:
        st.warning("No videos found.")


if __name__ == "__main__":
    if "drive" in st.session_state:
        drive = st.session_state["drive"]
    else:
        drive = get_drive_instance()
        if drive:
            st.session_state["drive"] = drive

    if not drive:
        st.error("❌ Drive not available. Please log in or set GDRIVE_SA_B64.")
        st.stop()

    st.markdown("## Video Search inside \"HVAC School\" and \"Love2HVAC with Ty\" YouTube channels")
    render_hvac_visual_search(drive)