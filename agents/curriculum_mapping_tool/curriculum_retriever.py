import json
import logging
import os
import pickle
import pandas as pd
import streamlit as st
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from langchain_classic.schema import BaseRetriever, Document
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from pydrive2.files import ApiRequestError
from gspread.utils import rowcol_to_a1
from services.embedding_service import get_embedding_model
from agents.curriculum_mapping_tool.supabase_vectorstore_service import (
    vector_search as supabase_vector_search,
    fulltext_search as supabase_fulltext_search,
)
from agents.course_outline.video_search_tool.video_retriever import video_retriever, warmup_video_retriever
from agents.vector_store_image_search.create_vectorstore import download_folder_from_drive
from agents.curriculum_mapping_tool.curriculum_mapping_agent import select_best_resources_unified
from agents.curriculum_mapping_tool.consolidation_agent import consolidate_category

from services.sheets_service import get_sheet_data_and_df, save_to_sheet, hide_columns_by_name, resize_column_by_name
from services.smart_progress_bar import SmartProgressBar
from langsmith import traceable
import re

logger = logging.getLogger(__name__)


def normalize_duration(duration_value: Any) -> str:
    """
    Normalize duration from various formats to a consistent 'Xh Ym' format.

    Input formats:
    - SkillCat: number (e.g., 1.5 means 1.5 hours)
    - Videos: "X seconds" format
    - NexTech: "1h 15m" or "1 hour 15 minutes" format

    Output format: "1h 30m" or "45m" (if less than an hour)
    """
    if duration_value is None or duration_value == "" or duration_value == "None":
        return ""

    duration_str = str(duration_value).strip().lower()

    if not duration_str:
        return ""

    total_minutes = 0

    # Handle "X seconds" format (videos)
    if "second" in duration_str:
        match = re.search(r'(\d+(?:\.\d+)?)\s*second', duration_str)
        if match:
            seconds = float(match.group(1))
            total_minutes = seconds / 60

    # Handle "Xh Ym" or "X hour(s) Y minute(s)" format (NexTech)
    elif "h" in duration_str or "hour" in duration_str or "m" in duration_str or "minute" in duration_str:
        hours = 0
        minutes = 0

        # Match hours: "1h", "1 h", "1hour", "1 hour", "1hours", "1 hours"
        hour_match = re.search(r'(\d+(?:\.\d+)?)\s*h(?:our)?s?', duration_str)
        if hour_match:
            hours = float(hour_match.group(1))

        # Match minutes: "15m", "15 m", "15min", "15 min", "15minute", "15 minute", "15minutes", "15 minutes"
        min_match = re.search(r'(\d+(?:\.\d+)?)\s*m(?:in(?:ute)?s?)?', duration_str)
        if min_match:
            # Avoid matching the 'm' from 'hour' accidentally
            min_str = min_match.group(0)
            if 'h' not in min_str:
                minutes = float(min_match.group(1))

        total_minutes = hours * 60 + minutes

    # Handle plain number (SkillCat - hours as decimal)
    else:
        try:
            hours = float(duration_str)
            total_minutes = hours * 60
        except ValueError:
            return duration_str  # Return original if can't parse

    # Format output
    if total_minutes <= 0:
        return ""

    hours = int(total_minutes // 60)
    minutes = int(round(total_minutes % 60))

    if hours > 0 and minutes > 0:
        return f"{hours}h {minutes}m"
    elif hours > 0:
        return f"{hours}h"
    elif minutes > 0:
        return f"{minutes}m"
    else:
        return ""



# ------------------ SOURCE CONFIG ------------------ #
VIDEO_CFG = {
    "name": "video",
    "central_folder_id": "1cUBmd1H1hBHSLohnAF68VJTEU9XcFXFK",
    "chroma_folder_name": "chroma_video_db",
    "collection_name": "video_embeddings",
    "bm25_pickle_name": "bm25_video_db.pkl",   
}

NEXTECH_CFG = {
    "name": "nextech",
    "central_folder_id": "1d5A-XwT_pZA78_vj2Tb2mfvWmgTOdi65",
    "chroma_folder_name": "chroma_nextech_db",
    "collection_name": "nextech_courses",  
    "bm25_pickle_name": "bm25_nextech_db.pkl",
}


SKILLCAT_CFG = {
    "name": "skillcat",
    "backend": "supabase",
    "central_folder_id": "14Xeg7riEhPOL_zaVQ2eo7bVRa9lTr-W1",
    "chroma_folder_name": "chroma_skillcat_db",
    "collection_name": "skillcat_courses",
    "bm25_pickle_name": "bm25_skillcat_db.pkl",
}

# module-level caches so we only initialize heavy retrievers once
_VECTOR_RETRIEVER_CACHE: Dict[str, Tuple[Chroma, Any]] = {}
_BM25_CACHE: Dict[str, BM25Retriever] = {}
_COMPRESSION_CACHE: Dict[str, ContextualCompressionRetriever] = {}


class SupabaseVectorRetriever(BaseRetriever):
    """LangChain BaseRetriever wrapping Supabase pgvector search."""

    k: int = 20

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        embedding_model = get_embedding_model()
        query_embedding = embedding_model.embed_query(query)
        return supabase_vector_search(query_embedding, k=self.k)


class SupabaseFTSRetriever(BaseRetriever):
    """LangChain BaseRetriever wrapping Supabase full-text search."""

    k: int = 20

    def _get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        return supabase_fulltext_search(query, k=self.k)


def _is_drive_quota_error(err: Exception) -> bool:
    return isinstance(err, ApiRequestError) and "quota" in str(err).lower()


def _warn_drive_quota(context: str) -> None:
    msg = (
        f"Google Drive storage quota exceeded while {context}. "
        "Continuing locally without uploading cache files."
    )
    # try:
        # st.warning(msg)
    # except Exception:
    print(msg)


def _get_default_llm_name() -> str:
    try:
        return st.session_state.get("llm_model", "gemini_2_5_flash") or "gemini_2_5_flash"
    except Exception:
        return "gemini_2_flash"


def load_chroma_db(cfg: Dict[str, str], embedding_function, drive) -> Chroma:
    """
    Loads a Chroma DB from Drive -> local temp folder, then opens Chroma collection.
    """
    central_folder_id = cfg["central_folder_id"]
    chroma_folder_name = cfg["chroma_folder_name"]
    collection_name = cfg["collection_name"]

    local_root = "/tmp/temp_chroma_folder"
    local_path = os.path.join(local_root, chroma_folder_name)
    os.makedirs(local_root, exist_ok=True)

    file_list = drive.ListFile({
        "q": (
            f"title='{chroma_folder_name}' and '{central_folder_id}' in parents "
            f"and mimeType='application/vnd.google-apps.folder' and trashed=false"
        )
    }).GetList()

    if not file_list:
        raise FileNotFoundError(f"'{chroma_folder_name}' not found in Drive folder {central_folder_id}.")

    chroma_folder_id = file_list[0]["id"]

    if not os.path.exists(os.path.join(local_path, "chroma.sqlite3")):
        download_folder_from_drive(chroma_folder_id, local_path, drive)

    return Chroma(
        embedding_function=embedding_function,
        collection_name=collection_name,
        persist_directory=local_path
    )


def load_vector_db_retriever(cfg: Dict[str, str], drive, k: int = 20) -> Tuple[Chroma, Any]:
    cache_key = cfg["name"]
    if cache_key in _VECTOR_RETRIEVER_CACHE:
        return _VECTOR_RETRIEVER_CACHE[cache_key]

    if cfg.get("backend") == "supabase":
        retriever = SupabaseVectorRetriever(k=k)
        _VECTOR_RETRIEVER_CACHE[cache_key] = (None, retriever)
        return None, retriever

    embedding_model = get_embedding_model()
    chroma_db = load_chroma_db(cfg, embedding_model, drive)
    retriever = chroma_db.as_retriever(search_kwargs={"k": k})
    _VECTOR_RETRIEVER_CACHE[cache_key] = (chroma_db, retriever)
    return chroma_db, retriever


def load_bm25_retriever_with_pydrive(cfg: Dict[str, str], drive, k: int = 20) -> BM25Retriever:
    """
    Loads or creates a BM25 retriever pickle stored in Drive under 'Pickle files' folder.
    Mirrors your pattern, but parameterized per source.
    """
    cache_key = cfg["name"]
    if cache_key in _BM25_CACHE:
        return _BM25_CACHE[cache_key]

    if cfg.get("backend") == "supabase":
        retriever = SupabaseFTSRetriever(k=k)
        _BM25_CACHE[cache_key] = retriever
        return retriever

    central_folder_id = cfg["central_folder_id"]
    pickle_name = cfg["bm25_pickle_name"]

    local_pickle_path = f"/tmp/{pickle_name}"

    # find/create 'Pickle files' folder
    pickle_folder_q = (
        f"title='Pickle files' and '{central_folder_id}' in parents "
        f"and mimeType='application/vnd.google-apps.folder' and trashed=false"
    )
    pickle_folders = drive.ListFile({"q": pickle_folder_q}).GetList()

    pickle_folder_id = None
    if pickle_folders:
        pickle_folder_id = pickle_folders[0]["id"]
    else:
        try:
            pf = drive.CreateFile({
                "title": "Pickle files",
                "parents": [{"id": central_folder_id}],
                "mimeType": "application/vnd.google-apps.folder"
            })
            pf.Upload()
            pickle_folder_id = pf["id"]
        except Exception as exc:
            if _is_drive_quota_error(exc):
                _warn_drive_quota("creating Drive cache folder")
            else:
                raise

    # locate pickle
    bm25_files = []
    if pickle_folder_id:
        bm25_q = f"title='{pickle_name}' and '{pickle_folder_id}' in parents and trashed=false"
        bm25_files = drive.ListFile({"q": bm25_q}).GetList()

    def _build_and_upload() -> BM25Retriever:
        embedding_fn = get_embedding_model()
        db = load_chroma_db(cfg, embedding_fn, drive)
        all_docs = db.get()

        documents = [
            Document(page_content=all_docs["documents"][i],
                     metadata=(all_docs["metadatas"][i] or {}))
            for i in range(len(all_docs["ids"]))
        ]

        bm25 = BM25Retriever.from_documents(documents, k=k)

        with open(local_pickle_path, "wb") as f:
            pickle.dump(bm25, f)

        if pickle_folder_id:
            try:
                pf = drive.CreateFile({"title": pickle_name, "parents": [{"id": pickle_folder_id}]})
                pf.SetContentFile(local_pickle_path)
                pf.Upload()
            except Exception as exc:
                if _is_drive_quota_error(exc):
                    _warn_drive_quota("uploading BM25 cache to Drive")
                else:
                    raise
        return bm25

    # if missing on drive, create
    if not bm25_files:
        bm25_retriever = _build_and_upload()
        _BM25_CACHE[cache_key] = bm25_retriever
        return bm25_retriever

    # try load local first
    if os.path.exists(local_pickle_path):
        try:
            with open(local_pickle_path, "rb") as f:
                bm25_retriever = pickle.load(f)
                _BM25_CACHE[cache_key] = bm25_retriever
                return bm25_retriever
        except Exception:
            try:
                os.remove(local_pickle_path)
            except Exception:
                pass

    # download and load
    bm25_files[0].GetContentFile(local_pickle_path)
    try:
        with open(local_pickle_path, "rb") as f:
            bm25_retriever = pickle.load(f)
            _BM25_CACHE[cache_key] = bm25_retriever
            return bm25_retriever
    except Exception:
        # rebuild if corrupted
        try:
            os.remove(local_pickle_path)
        except Exception:
            pass
        bm25_retriever = _build_and_upload()
        _BM25_CACHE[cache_key] = bm25_retriever
        return bm25_retriever


def get_ensemble_retriever(
    cfg: Dict[str, str],
    drive,
    w_bm25: float = 0.5,
    w_vec: float = 0.5,
    show_spinner: bool = False,
):
    def _load():
        _, vec = load_vector_db_retriever(cfg, drive, k=20)
        bm25 = load_bm25_retriever_with_pydrive(cfg, drive, k=20)
        return EnsembleRetriever(
            retrievers=[bm25, vec],
            weights=[w_bm25, w_vec],
        )

    if show_spinner:
        with st.spinner("Loading hybrid retriever..."):
            return _load()
    return _load()


def get_compression_retriever(
    cfg: Dict[str, str],
    drive,
    w_bm25: float = 0.5,
    w_vec: float = 0.5,
    compressor_top_n: int = 15,
    show_spinner: bool = False,
):
    cache_key = cfg["name"]
    if cache_key in _COMPRESSION_CACHE:
        return _COMPRESSION_CACHE[cache_key]

    def _load():
        ensemble = get_ensemble_retriever(
            cfg, drive, w_bm25=w_bm25, w_vec=w_vec, show_spinner=False
        )
        compressor = CohereRerank(
            model="rerank-v3.5",
            top_n=compressor_top_n,
        )
        compression = ContextualCompressionRetriever(
            base_retriever=ensemble,
            base_compressor=compressor
        )
        _COMPRESSION_CACHE[cache_key] = compression
        return compression

    if show_spinner:
        with st.spinner("Loading reranker..."):
            return _load()
    return _load()


def retrieve_course_docs(cfg: Dict[str, str], drive, query: str, k: int = 15) -> List[Document]:
    retriever = get_compression_retriever(cfg, drive, w_bm25=0.5, w_vec=0.5)
    docs = retriever.invoke(query)
    if not docs:
        return []
    
    # Filter out documents without course_link for skillcat and nextech
    if cfg["name"] in ("skillcat", "nextech"):
        docs = [
            doc for doc in docs
            if (getattr(doc, "metadata", {}) or {}).get("course_link")
        ]
    
    return list(docs[: max(0, k)])


def best_course_result_from_docs(docs, k: int = 1) -> List[Dict[str, Any]]:
    """
    From retrieved documents, pick best k unique by (name + link).
    :param docs: List of Document objects
    :param k: Number of results to return
    :return: List of dicts with course_name, course_link, metadata, text
    
    """
    results = []
    seen = set()

    for doc in docs:
        if len(results) >= k:
            break

        md = getattr(doc, "metadata", {}) or {}
        name = md.get("course_name") or md.get("title") or md.get("name")
        link = md.get("course_link") or md.get("url") or md.get("link")

        # fallback: allow something even if link missing
        dedupe_key = (name or "") + "|" + (link or "")
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        results.append({
            "course_name": name or "Untitled",
            "course_link": link or "",
            "metadata": md,
            "text": getattr(doc, "page_content", "") or "",
        })

    return results


def build_row_query(category: str, course: str) -> str:
    category = (category or "").strip()
    course = (course or "").strip()
    return f"{category} - {course}".strip(" -")


def format_course_result(res_list: List[Dict[str, Any]]) -> str:
    if not res_list:
        return "No relevant match found"
    r = res_list[0]
    name = r.get("course_name", "Untitled")
    return name


def _apply_video_hyperlinks(worksheet, df: pd.DataFrame) -> None:
    """
    Rewrites the YT Videos column using USER_ENTERED semantics so Google Sheets
    treats =HYPERLINK() strings as formulas instead of plain text.
    """
    if "YT Videos" not in df.columns or df.empty:
        return

    col_idx = df.columns.get_loc("YT Videos") + 1  # 1-based column index
    start_row = 2  # header occupies row 1
    end_row = len(df) + 1
    if end_row < start_row:
        return

    start_cell = rowcol_to_a1(start_row, col_idx)
    end_cell = rowcol_to_a1(end_row, col_idx)
    rng = f"{start_cell}:{end_cell}"

    values = [[("" if pd.isna(v) else str(v))] for v in df["YT Videos"].tolist()]
    worksheet.update(rng, values, value_input_option="USER_ENTERED")


def _extract_duration_from_resource(resource_json: str) -> str:
    """
    Extract and normalize duration from a resource JSON string.
    Returns normalized duration string or empty string if not available.
    """
    if not resource_json or not resource_json.strip():
        return ""

    try:
        resource = json.loads(resource_json)
    except json.JSONDecodeError:
        return ""

    duration = resource.get("duration")
    return normalize_duration(duration)


def _format_consolidated_hyperlink(resource_json: str) -> str:
    """
    Format consolidated resource as a HYPERLINK with source prefix.
    Example: =HYPERLINK("url", "SkillCat: Course Name")
    """
    if not resource_json or not resource_json.strip():
        return "No match found"

    try:
        resource = json.loads(resource_json)
    except json.JSONDecodeError:
        return "Invalid resource"

    name = resource.get("name", "")
    link = resource.get("link", "")
    source = resource.get("source", "").lower()

    if not name:
        return "No match found"

    # Add source prefix
    if source == "skillcat":
        display_name = f"SkillCat: {name}"
    elif source == "nextech":
        display_name = f"NexTech: {name}"
    elif source == "video":
        display_name = f"Video: {name}"
    else:
        display_name = name

    if not link:
        return display_name

    # Format as HYPERLINK formula
    safe_name = str(display_name).replace('"', '""')
    safe_url = str(link).replace('"', '""')
    return f'=HYPERLINK("{safe_url}", "{safe_name}")'


def _apply_consolidated_hyperlinks(worksheet, df: pd.DataFrame) -> None:
    """
    Rewrites the Consolidated Resource Name column using USER_ENTERED semantics
    so Google Sheets treats =HYPERLINK() strings as formulas.
    """
    if "Consolidated Resource Name" not in df.columns or df.empty:
        return

    col_idx = df.columns.get_loc("Consolidated Resource Name") + 1  # 1-based
    start_row = 2  # header occupies row 1
    end_row = len(df) + 1
    if end_row < start_row:
        return

    start_cell = rowcol_to_a1(start_row, col_idx)
    end_cell = rowcol_to_a1(end_row, col_idx)
    rng = f"{start_cell}:{end_cell}"

    values = [[("" if pd.isna(v) else str(v))] for v in df["Consolidated Resource Name"].tolist()]
    worksheet.update(rng, values, value_input_option="USER_ENTERED")


# def _seconds_to_timestamp(value: Optional[Any]) -> Optional[str]:
#     if value in (None, "", "None"):
#         return None
#     try:
#         seconds = int(float(value))
#     except Exception:
#         return None

#     minutes, sec = divmod(seconds, 60)
#     hours, minutes = divmod(minutes, 60)
#     if hours:
#         return f"{hours:d}:{minutes:02d}:{sec:02d}"
#     return f"{minutes:02d}:{sec:02d}"


def _format_selected_course(docs: List[Document], selected_idx: Optional[int]) -> str:
    if not docs:
        return "No relevant match found"

    if selected_idx is None or selected_idx < 0 or selected_idx >= len(docs):
        chosen = docs[0]
    else:
        chosen = docs[selected_idx]

    return format_course_result(best_course_result_from_docs([chosen], k=1))


def _format_selected_video(videos: List[Dict[str, Any]], selected_idx: Optional[int]) -> str:
    if not videos:
        return "No relevant match found"

    if selected_idx is None or selected_idx < 0 or selected_idx >= len(videos):
        chosen = videos[0]
    else:
        chosen = videos[selected_idx]

    url = chosen.get("url") or chosen.get("video_link") or ""
    title = chosen.get("video_title") or chosen.get("title") or "No relevant match found"
    channel_name = chosen.get("channel_name") or chosen.get("channel") or ""

    if not url:
        return "No relevant match found"

    # Limit channel name to 20 characters
    if channel_name:
        channel_name = str(channel_name)[:20]
        display_title = f"{channel_name}: {title}"
    else:
        display_title = str(title)

    safe_title = str(display_title).replace('"', '""')
    safe_url = str(url).replace('"', '""')
    return f'=HYPERLINK("{safe_url}", "{safe_title}")'


def _extract_best_resource_metadata(
    selection: Dict[str, Any],
    skillcat_docs: List[Document],
    nextech_docs: List[Document],
    video_candidates: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """
    Extract full metadata for the overall best resource.
    Returns a dict with standardized fields for PDF export.
    """
    best_source = selection.get("overall_best_source")
    if not best_source:
        return None

    best_metadata = {
        "source": best_source,
        "reason": selection.get("overall_best_reason", ""),
    }

    if best_source == "skillcat" and skillcat_docs:
        idx = selection.get("skillcat_idx")
        if idx is None:
            idx = 0
        idx = min(idx, len(skillcat_docs) - 1)
        doc = skillcat_docs[idx]
        md = getattr(doc, "metadata", {}) or {}

        best_metadata.update({
            "name": md.get("course_name") or "Untitled",
            "description": md.get("course_description") or "",
            "duration": md.get("course_duration_hours"),
            "topics": md.get("course_topics") or "",
            "category": md.get("course_category") or "",
            "link": md.get("course_link") or "",
            "learning_objectives": md.get("course_learning_objectives") or "",
            "prerequisites": md.get("course_prerequisites") or "",
            "nps_score": md.get("nps_score"),
        })

    elif best_source == "nextech" and nextech_docs:
        idx = selection.get("nextech_idx")
        if idx is None:
            idx = 0
        idx = min(idx, len(nextech_docs) - 1)
        doc = nextech_docs[idx]
        md = getattr(doc, "metadata", {}) or {}

        best_metadata.update({
            "name": md.get("course_name") or md.get("title") or "Untitled",
            "description": md.get("description") or "",
            "duration": md.get("duration"),
            "topics": md.get("tags") or "",
            "category": md.get("category") or md.get("trade") or "",
            "link": md.get("course_link") or "",
            "level": md.get("level") or "",
        })

    elif best_source == "video" and video_candidates:
        idx = selection.get("video_idx")
        if idx is None:
            idx = 0
        idx = min(idx, len(video_candidates) - 1)
        video = video_candidates[idx]

        # Calculate duration from timestamps if available
        start_time = video.get("start_time")
        end_time = video.get("end_time")
        duration = None
        if start_time is not None and end_time is not None:
            try:
                duration = f"{int(end_time) - int(start_time)} seconds"
            except (ValueError, TypeError):
                pass

        best_metadata.update({
            "name": video.get("video_title") or video.get("title") or "Untitled",
            "description": video.get("transcript") or video.get("text_0") or "",
            "duration": duration,
            "category": video.get("channel_name") or video.get("channel") or "",
            "link": video.get("url") or video.get("video_link") or "",
            "published": video.get("published") or "",
        })

    return best_metadata


@traceable(metadata={
    "agent_name": "curriculum_mapping",
    "step_name": "Map Row",
    "function_name": "map_one_row",
    "user_id": st.session_state.get("role", "anonymous"),
    "user_email": st.session_state.get("user_email", "anonymous")
})
def map_one_row(category: str, course: str, drive, k_each: int = 5) -> Dict[str, str]:
    query = build_row_query(category, course)
    k_each = max(1, k_each)
    logger.info("Mapping query='%s'", query)

    # ---- Retrieval (same logic, no nesting) ----
    skillcat_docs = retrieve_course_docs(SKILLCAT_CFG, drive, query, k_each) or []
    nextech_docs = retrieve_course_docs(NEXTECH_CFG, drive, query, k_each) or []
    video_candidates = video_retriever(query, drive, k_each, None) or []

    logger.debug(
        "Retrieved | SkillCat=%d | NexTech=%d | Video=%d",
        len(skillcat_docs),
        len(nextech_docs),
        len(video_candidates),
    )

    llm_name = _get_default_llm_name()

    try:
        selection = select_best_resources_unified(
            query=query,
            skillcat_docs=skillcat_docs,
            nextech_docs=nextech_docs,
            video_docs=video_candidates,
            llm=llm_name,
        )
    except Exception as exc:
        logger.exception("Unified selection failed for query='%s'", query)
        selection = {
            "skillcat_idx": None,
            "nextech_idx": None,
            "video_idx": None,
            "overall_best_source": None,
            "overall_best_reason": "",
        }

    skillcat_display = _format_selected_course(skillcat_docs, selection.get("skillcat_idx"))
    nextech_display = _format_selected_course(nextech_docs, selection.get("nextech_idx"))
    video_display = _format_selected_video(video_candidates, selection.get("video_idx"))

    # Extract best resource metadata for PDF export
    best_metadata = _extract_best_resource_metadata(
        selection=selection,
        skillcat_docs=skillcat_docs,
        nextech_docs=nextech_docs,
        video_candidates=video_candidates,
    )

    logger.info(
        "Selected | SkillCat=%s | NexTech=%s | Video=%s | Overall=%s",
        selection.get("skillcat_idx"),
        selection.get("nextech_idx"),
        selection.get("video_idx"),
        selection.get("overall_best_source"),
    )

    # Extract best resource name for clear display
    best_resource_name = ""
    if best_metadata:
        best_resource_name = best_metadata.get("name", "")

    return {
        "Category": category,
        "Course": course,
        "SkillCat Resource": skillcat_display,
        "NexTech Resource": nextech_display,
        "YT Videos": video_display,
        "Best Resource Name": best_resource_name,
        "Best Resource": json.dumps(best_metadata) if best_metadata else "",
    }



@traceable(metadata={
    "agent_name": "curriculum_mapping",
    "step_name": "Run Curriculum Mapping",
    "function_name": "run_curriculum_mapping",
    "user_id": st.session_state.get("role", "anonymous"),
    "user_email": st.session_state.get("user_email", "anonymous")
})
def run_curriculum_mapping(
    sheet,
    drive,
    input_worksheet_name: Optional[str] = None,
    k_each: int = 5,
):
    """
    Runs curriculum mapping on the given sheet, writing results back to it.
    """

    # -------- Load worksheet --------
    if input_worksheet_name:
        ws_in, df = get_sheet_data_and_df(sheet, input_worksheet_name)
    else:
        first_ws = sheet.get_worksheet(0)
        ws_in, df = get_sheet_data_and_df(sheet, first_ws.title)

    required = {"Category", "Course"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input sheet missing columns: {missing}")

    for col in ["SkillCat Resource", "NexTech Resource", "YT Videos", "Best Resource Name", "Best Resource"]:
        if col not in df.columns:
            df[col] = ""

    # Check if mapping has already been completed
    total_rows_to_map = 0
    mapped_rows = 0

    for idx, row in df.iterrows():
        category = str(row.get("Category", "") or "").strip()
        course = str(row.get("Course", "") or "").strip()

        if not category and not course:
            continue

        total_rows_to_map += 1

        # Check if this row has all mapping results
        if (
            str(row.get("SkillCat Resource", "")).strip()
            and str(row.get("NexTech Resource", "")).strip()
            and str(row.get("YT Videos", "")).strip()
        ):
            mapped_rows += 1

    if total_rows_to_map > 0:
        completion_rate = mapped_rows / total_rows_to_map

        if completion_rate >= 0.95:  # 95% or more already mapped
            st.info(f"Mapping already completed ({mapped_rows}/{total_rows_to_map} rows). Skipping mapping step.")
            logger.info(f"Skipping mapping - already {completion_rate*100:.1f}% complete ({mapped_rows}/{total_rows_to_map} rows)")
            return df

    # -------- Warm up retrievers once --------
    with st.spinner("Preparing SkillCat and NexTech retrievers..."):
        get_compression_retriever(SKILLCAT_CFG, drive)
        get_compression_retriever(NEXTECH_CFG, drive)

    with st.spinner("Preparing video retriever..."):
        warmup_video_retriever(drive)

    # -------- Build task list --------
    tasks = []
    for idx, row in df.iterrows():
        category = str(row.get("Category", "") or "").strip()
        course = str(row.get("Course", "") or "").strip()

        if not category and not course:
            continue

        if (
            str(row.get("SkillCat Resource", "")).strip()
            and str(row.get("NexTech Resource", "")).strip()
            and str(row.get("YT Videos", "")).strip()
        ):
            continue

        tasks.append((idx, category, course))

    if not tasks:
        st.info("No rows to process.")
        return df

    total = len(tasks)

    # -------- Parallel row execution (CORRECTED) --------
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(map_one_row, category, course, drive, k_each): idx
            for idx, category, course in tasks
        }

        progress = SmartProgressBar(
            total_tasks=total,
            description="Percent complete",
            save_interval=15,
        )

        for future in tqdm(as_completed(futures), total=total):
            idx = futures[future]

            try:
                mapped = future.result()
            except Exception as exc:
                logger.exception("Row %s failed", idx)
                continue

            df.at[idx, "SkillCat Resource"] = mapped["SkillCat Resource"]
            df.at[idx, "NexTech Resource"] = mapped["NexTech Resource"]
            df.at[idx, "YT Videos"] = mapped["YT Videos"]
            df.at[idx, "Best Resource Name"] = mapped.get("Best Resource Name", "")
            df.at[idx, "Best Resource"] = mapped.get("Best Resource", "")

            progress.update()

            if progress.should_save():
                save_to_sheet(ws_in, df)
                _apply_video_hyperlinks(ws_in, df)

    # -------- Final save --------
    save_to_sheet(ws_in, df)
    _apply_video_hyperlinks(ws_in, df)

    return df


@traceable(metadata={
    "agent_name": "curriculum_mapping",
    "step_name": "Run Curriculum Consolidation",
    "function_name": "run_curriculum_consolidation",
    "user_id": st.session_state.get("role", "anonymous"),
    "user_email": st.session_state.get("user_email", "anonymous")
})
def run_curriculum_consolidation(
    sheet,
    drive,
    input_worksheet_name: Optional[str] = None,
) -> pd.DataFrame:
    """
    Runs consolidation on existing curriculum mapping results.

    This function groups mapping results by Category and uses an LLM agent
    to intelligently consolidate minority courses into majority courses
    when the majority course adequately covers the concept.

    Prerequisites: run_curriculum_mapping() must have been completed first.

    Returns DataFrame with added columns:
    - Consolidated Resource: The consolidated course assignment (JSON)
    - Consolidation Reason: Explanation for the consolidation decision
    """

    # -------- Load worksheet --------
    if input_worksheet_name:
        ws_in, df = get_sheet_data_and_df(sheet, input_worksheet_name)
    else:
        first_ws = sheet.get_worksheet(0)
        ws_in, df = get_sheet_data_and_df(sheet, first_ws.title)

    # Verify required columns exist
    required = {"Category", "Course", "Best Resource"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Consolidation requires columns: {missing}. Run mapping first.")

    # Check if Best Resource has data
    best_resource_filled = df["Best Resource"].notna() & (df["Best Resource"].str.strip() != "")
    if not best_resource_filled.any():
        raise ValueError("No 'Best Resource' data found. Run mapping first.")

    # Initialize consolidation columns
    if "Consolidated Resource Name" not in df.columns:
        df["Consolidated Resource Name"] = ""
    if "Time Duration" not in df.columns:
        df["Time Duration"] = ""
    if "Consolidated Resource" not in df.columns:
        df["Consolidated Resource"] = ""
    if "Consolidation Reason" not in df.columns:
        df["Consolidation Reason"] = ""

    # Check if consolidation has already been completed
    consolidated_filled = df["Consolidated Resource"].notna() & (df["Consolidated Resource"].str.strip() != "")
    total_rows_with_best_resource = (df["Best Resource"].notna() & (df["Best Resource"].str.strip() != "")).sum()

    if total_rows_with_best_resource > 0:
        completed_rows = consolidated_filled.sum()
        completion_rate = completed_rows / total_rows_with_best_resource

        if completion_rate >= 0.95:  # 95% or more already consolidated
            st.info(f"Consolidation already completed ({completed_rows}/{total_rows_with_best_resource} rows). Skipping consolidation step.")
            logger.info(f"Skipping consolidation - already {completion_rate*100:.1f}% complete ({completed_rows}/{total_rows_with_best_resource} rows)")
            return df

    # -------- Group by Category --------
    categories = df["Category"].dropna().unique()
    categories = [c for c in categories if str(c).strip()]

    if not categories:
        st.info("No categories found to consolidate.")
        return df

    total_categories = len(categories)
    llm_name = _get_default_llm_name()

    logger.info(f"Starting consolidation for {total_categories} categories")

    # -------- Build category tasks --------
    category_tasks = []
    for category in categories:
        category = str(category).strip()
        if not category:
            continue

        # Get rows for this category
        category_mask = df["Category"] == category
        category_indices = df.index[category_mask].tolist()

        if not category_indices:
            continue

        # Build concepts_data for this category
        concepts_data = []
        for idx in category_indices:
            row = df.loc[idx]
            concept = str(row.get("Course", "")).strip()
            best_resource = str(row.get("Best Resource", "")).strip()

            if concept:
                concepts_data.append({
                    "concept": concept,
                    "best_resource": best_resource,
                    "row_idx": idx
                })

        if concepts_data:
            category_tasks.append((category, concepts_data))

    if not category_tasks:
        st.info("No categories with valid data found to consolidate.")
        return df

    total_tasks = len(category_tasks)

    # Helper function to process a single category
    def _consolidate_one_category(category: str, concepts_data: List[Dict]) -> Dict[str, Any]:
        logger.info(f"Processing category '{category}' with {len(concepts_data)} concepts")
        try:
            consolidation_results = consolidate_category(
                category_name=category,
                concepts_data=concepts_data,
                llm=llm_name
            )
            return {
                "category": category,
                "concepts_data": concepts_data,
                "results": consolidation_results,
                "error": None
            }
        except Exception as exc:
            logger.exception(f"Consolidation failed for category '{category}'")
            return {
                "category": category,
                "concepts_data": concepts_data,
                "results": None,
                "error": str(exc)[:100]
            }

    # -------- Process categories in parallel --------
    with st.spinner(f"Consolidating {total_tasks} categories..."):
        progress_bar = st.progress(0, text="Consolidating categories...")

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {
                executor.submit(_consolidate_one_category, category, concepts_data): category
                for category, concepts_data in category_tasks
            }

            completed = 0
            for future in as_completed(futures):
                result = future.result()
                completed += 1

                category = result["category"]
                concepts_data = result["concepts_data"]
                consolidation_results = result["results"]
                error = result["error"]

                if error:
                    # On error, copy original Best Resource to Consolidated Resource
                    for item in concepts_data:
                        idx = item["row_idx"]
                        original_json = item["best_resource"]
                        # Format as hyperlink with source prefix
                        df.at[idx, "Consolidated Resource Name"] = _format_consolidated_hyperlink(original_json)
                        df.at[idx, "Time Duration"] = _extract_duration_from_resource(original_json)
                        df.at[idx, "Consolidated Resource"] = original_json
                        df.at[idx, "Consolidation Reason"] = f"Error during consolidation: {error}"
                else:
                    # Map results back to DataFrame
                    results_by_concept = {
                        r["concept"].lower().strip(): r
                        for r in consolidation_results
                    }

                    for item in concepts_data:
                        concept = item["concept"]
                        idx = item["row_idx"]

                        res = results_by_concept.get(concept.lower().strip())

                        if res:
                            consolidated_json = res.get("consolidated_resource_json", "")
                            # Format as hyperlink with source prefix
                            df.at[idx, "Consolidated Resource Name"] = _format_consolidated_hyperlink(consolidated_json)
                            df.at[idx, "Time Duration"] = _extract_duration_from_resource(consolidated_json)
                            df.at[idx, "Consolidated Resource"] = consolidated_json
                            df.at[idx, "Consolidation Reason"] = res.get("reason", "")
                        else:
                            # Fallback: keep original
                            original_json = item["best_resource"]
                            # Format as hyperlink with source prefix
                            df.at[idx, "Consolidated Resource Name"] = _format_consolidated_hyperlink(original_json)
                            df.at[idx, "Time Duration"] = _extract_duration_from_resource(original_json)
                            df.at[idx, "Consolidated Resource"] = original_json
                            df.at[idx, "Consolidation Reason"] = "No consolidation decision"

                # Update progress
                progress = completed / total_tasks
                progress_bar.progress(progress, text=f"Consolidated {completed}/{total_tasks} categories")

        progress_bar.progress(1.0, text="Consolidation complete!")

    # -------- Calculate and log stats --------
    changes = 0
    for idx, row in df.iterrows():
        original = str(row.get("Best Resource", "")).strip()
        consolidated = str(row.get("Consolidated Resource", "")).strip()

        if original and consolidated and original != consolidated:
            try:
                orig_json = json.loads(original) if original else {}
                cons_json = json.loads(consolidated) if consolidated else {}
                if orig_json.get("name") != cons_json.get("name"):
                    changes += 1
            except json.JSONDecodeError:
                pass

    total_rows = len(df[df["Best Resource"].str.strip() != ""])
    logger.info(f"Consolidation complete: {changes} of {total_rows} assignments changed")

    # -------- Save to sheet --------
    save_to_sheet(ws_in, df)

    # -------- Apply hyperlinks for Consolidated Resource Name --------
    _apply_consolidated_hyperlinks(ws_in, df)

    # -------- Hide all columns except Category, Course, Consolidated Resource Name, and Time Duration --------
    columns_to_hide = [
        col for col in df.columns
        if col not in ["Category", "Course", "Consolidated Resource Name", "Time Duration"]
    ]
    if columns_to_hide:
        hide_columns_by_name(ws_in, columns_to_hide, df)

    # resize_column_by_name(worksheet=ws_in, column_name="Consolidated Resource Name", pixel_size=250, wrap="OVERFLOW")

    return df