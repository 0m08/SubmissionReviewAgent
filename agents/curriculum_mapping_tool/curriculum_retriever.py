import logging
import os
import pickle
import pandas as pd
import streamlit as st
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain.retrievers import BM25Retriever, EnsembleRetriever, ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from pydrive2.files import ApiRequestError
from services.embedding_service import get_embedding_model
from agents.course_outline.video_search_tool.video_retriever import video_retriever
from agents.vector_store_image_search.create_vectorstore import download_folder_from_drive
from agents.curriculum_mapping_tool.curriculum_mapping_agent import (
    select_best_course_resource,
    select_best_video_resource,
)
from services.sheets_service import get_sheet_data_and_df, save_to_sheet
from services.smart_progress_bar import SmartProgressBar
from langsmith import traceable

logger = logging.getLogger(__name__)



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
    "central_folder_id": "14Xeg7riEhPOL_zaVQ2eo7bVRa9lTr-W1",
    "chroma_folder_name": "chroma_skillcat_db",
    "collection_name": "skillcat_courses",
    "bm25_pickle_name": "bm25_skillcat_db.pkl",
}

# module-level caches so we only initialize heavy retrievers once
_VECTOR_RETRIEVER_CACHE: Dict[str, Tuple[Chroma, Any]] = {}
_BM25_CACHE: Dict[str, BM25Retriever] = {}
_COMPRESSION_CACHE: Dict[str, ContextualCompressionRetriever] = {}


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
        return st.session_state.get("llm_model", "gemini_2_flash") or "gemini_2_flash"
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
    return url or "No relevant match found"


@traceable(metadata={
    "agent_name": "curriculum_mapping",
    "step_name": "Map Row",
    "function_name": "map_one_row",
    "user_id": st.session_state.get("role", "anonymous"),
    "user_email": st.session_state.get("user_email", "anonymous")
})
def map_one_row(category: str, course: str, drive, k_each: int = 15) -> Dict[str, str]:
    """
    Maps one row (category + course) to best resources from SkillCat, NexTech, and YouTube videos.
    :param category: Course category
    :param course: Course name
    :param drive: PyDrive2 Drive object
    :param k_each: Number of top documents to retrieve from each source.
    :return: Dict with mapped resources.
    """
    query = build_row_query(category, course)
    k_each = max(1, k_each)
    logger.info("Mapping query='%s' (Category=%s, Course=%s) with k_each=%d", query, category, course, k_each)

    # run 3 sources in parallel
    with ThreadPoolExecutor(max_workers=3) as ex:
        futures = {
            ex.submit(retrieve_course_docs, SKILLCAT_CFG, drive, query, k_each): "skillcat",
            ex.submit(retrieve_course_docs, NEXTECH_CFG, drive, query, k_each): "nextech",
            ex.submit(video_retriever, query, drive, k_each, None): "video",
        }

        results = {}
        for fut in as_completed(futures):
            source = futures[fut]
            try:
                results[source] = fut.result()
            except Exception as e:
                results[source] = []
                # don't crash per-row
                # in Streamlit you can log this if desired:
                # st.error(f"{source} search failed: {e}")

    skillcat_docs = results.get("skillcat", []) or []
    nextech_docs = results.get("nextech", []) or []
    video_candidates = results.get("video", []) or []
    logger.debug(
        "Retrieved candidates | query='%s' | skillcat=%d | nextech=%d | video=%d",
        query,
        len(skillcat_docs),
        len(nextech_docs),
        len(video_candidates),
    )

    llm_name = _get_default_llm_name()

    def _warn_agent_failure(source: str, exc: Exception) -> None:
        warning = f"{source} agent failed for '{query}': {exc}"
        logger.exception("%s agent failed for query='%s'.", source, query)
        # try:
            # st.warning(warning)
        # except Exception:
        print(warning)

    try:
        skillcat_selection = select_best_course_resource(
            "SkillCat", query, skillcat_docs, llm=llm_name
        )
    except Exception as exc:
        _warn_agent_failure("SkillCat", exc)
        skillcat_selection = {"index": None, "reason": f"SkillCat selection failed: {exc}"}

    try:
        nextech_selection = select_best_course_resource(
            "NexTech", query, nextech_docs, llm=llm_name
        )
    except Exception as exc:
        _warn_agent_failure("NexTech", exc)
        nextech_selection = {"index": None, "reason": f"NexTech selection failed: {exc}"}

    try:
        video_selection = select_best_video_resource(
            query, video_candidates, llm=llm_name
        )
    except Exception as exc:
        _warn_agent_failure("Video", exc)
        video_selection = {"index": None, "reason": f"Video selection failed: {exc}"}

    skillcat_idx = (skillcat_selection or {}).get("index")
    nextech_idx = (nextech_selection or {}).get("index")
    video_idx = (video_selection or {}).get("index")

    skillcat_display = _format_selected_course(skillcat_docs, skillcat_idx)
    nextech_display = _format_selected_course(nextech_docs, nextech_idx)
    video_display = _format_selected_video(video_candidates, video_idx)
    logger.debug("SkillCat reasoning: %s", (skillcat_selection or {}).get("reason"))
    logger.debug("NexTech reasoning: %s", (nextech_selection or {}).get("reason"))
    logger.debug("Video reasoning: %s", (video_selection or {}).get("reason"))
    logger.info(
        "Agent selection | query='%s' | SkillCat idx=%s | NexTech idx=%s | Video idx=%s",
        query,
        skillcat_idx,
        nextech_idx,
        video_idx,
    )

    return {
        "Category": category,
        "Course": course,
        "SkillCat Resource": skillcat_display,
        "NexTech Resource": nextech_display,
        "YT Videos": video_display,
    }


@traceable(metadata={
    "agent_name": "curriculum_mapping",
    "step_name": "Run Curriculum Mapping",
    "function_name": "run_curriculum_mapping",
    "user_id": st.session_state.get("role", "anonymous"),
    "user_email": st.session_state.get("user_email", "anonymous")
})
def run_curriculum_mapping(sheet, drive, input_worksheet_name: Optional[str] = None, k_each: int = 15):
    """
    Runs curriculum mapping on the given sheet, writing results back to it.
    :param sheet: Google Sheet object
    :param drive: PyDrive2 Drive object
    :param input_worksheet_name: Optional name of the worksheet to read from. If None uses the first worksheet.
    :param k_each: Number of top documents to retrieve from each source per query.
    :return: DataFrame with results.
    """
    
    # choose input worksheet
    if input_worksheet_name:
        ws_in, df = get_sheet_data_and_df(sheet, input_worksheet_name)
    else:
        first_ws = sheet.get_worksheet(0)
        ws_in, df = get_sheet_data_and_df(sheet, first_ws.title)

    required = {"Category", "Course"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input sheet missing columns: {missing}")

    for col in ["SkillCat Resource", "NexTech Resource", "YT Videos"]:
        if col not in df.columns:
            df[col] = ""

    # Warm up retrievers once on the main thread so downstream threaded calls avoid Streamlit errors.
    with st.spinner("Preparing SkillCat and NexTech retrievers..."):
        get_compression_retriever(SKILLCAT_CFG, drive)
        get_compression_retriever(NEXTECH_CFG, drive)

    tasks = []
    for idx, row in df.iterrows():
        category = str(row.get("Category", "") or "").strip()
        course = str(row.get("Course", "") or "").strip()

        # skip empty rows
        if not category and not course:
            continue

        # skip rows already processed
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
    futures = {}
    with ThreadPoolExecutor(max_workers=5) as executor:
        for idx, category, course in tasks:
            futures[executor.submit(map_one_row, category, course, drive, k_each)] = (idx, category, course)

        progress = SmartProgressBar(total_tasks=total, description="Percent complete", save_interval=5)
        completed = 0

        for future in tqdm(as_completed(futures), total=total):
            idx, category, course = futures[future]
            mapped = future.result()

            df.at[idx, "SkillCat Resource"] = mapped["SkillCat Resource"]
            df.at[idx, "NexTech Resource"] = mapped["NexTech Resource"]
            df.at[idx, "YT Videos"] = mapped["YT Videos"]

            completed += 1
            progress.update()

            if progress.should_save():
                save_to_sheet(ws_in, df)

    save_to_sheet(ws_in, df)
    return df
