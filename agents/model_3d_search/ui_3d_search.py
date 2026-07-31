import os
import re
import time
import traceback
import pandas as pd
import requests
from io import BytesIO
import streamlit as st
from PIL import Image
from typing import Dict, Any, List, Optional, Tuple

from agents.model_3d_search.vectorstore_3d import load_3d_chroma_db
from agents.model_3d_search.ingest_3d_models import (
    create_vectorstore_3d,
    update_vectorstore_3d,
    get_3d_sheet_info_and_stats,
)
from agents.model_3d_search.retriever_3d import search_vectorstore_3d
from services.activity_tracking_service import track_tool_action

DEFAULT_CATALOG_SHEET_URL = "https://docs.google.com/spreadsheets/d/1-iPNvmmsUW1ZbHANd0a3foeycSP3t6k8bn5DmF3aIoI/edit?usp=sharing"

# Central Google Drive folder that stores the 3D Models vectorstore files.
# https://drive.google.com/drive/u/0/folders/1tJRuj-kbNeKA8TH1PGZTegVjm6cIa6ok
CENTRAL_3D_MODELS_FOLDER_ID = "1tJRuj-kbNeKA8TH1PGZTegVjm6cIa6ok"


@st.cache_data(ttl=120, show_spinner=False)
def fetch_sheet_filter_options(sheet_link: str) -> Tuple[List[str], List[str]]:
    """
    Dynamically fetch unique Category and Extension values from Google Sheet tab 'Final Output of the migrated assets '.
    """
    if "gc" not in st.session_state or not st.session_state["gc"]:
        return [], []
    try:
        gc = st.session_state["gc"]
        sheet = gc.open_by_url(sheet_link)
        ws = sheet.worksheet("Final Output of the migrated assets updated")
        df = pd.DataFrame(ws.get_all_records())
        categories = set()
        extensions = set()

        if "Category" in df.columns:
            for cat in df["Category"].dropna().unique():
                val = str(cat).strip()
                if val:
                    categories.add(val)

        if "Extension" in df.columns:
            for ext in df["Extension"].dropna().unique():
                val = str(ext).strip()
                if val:
                    extensions.add(val)

        return sorted(list(categories)), sorted(list(extensions))
    except Exception as e:
        print(f"⚠️ Could not fetch filter options from Google Sheet: {e}")
        return [], []


def get_unique_categories_from_vectorstore(dbs) -> List[str]:
    """
    Fallback: Extract unique category strings from indexed metadata.
    """
    try:
        col = dbs["text"]._collection
        get_res = col.get(include=["metadatas"])
        categories = set()
        for meta in get_res.get("metadatas", []):
            if meta and "category" in meta and str(meta["category"]).strip():
                categories.add(str(meta["category"]).strip())
        return sorted(list(categories))
    except Exception:
        return []


def _extract_drive_file_id(url: str) -> Optional[str]:
    """Extract a Google Drive file ID from various Drive URL formats."""
    if not url:
        return None
    # /file/d/<ID>/  or  id=<ID>
    m = re.search(r"/file/d/([a-zA-Z0-9_-]+)", url)
    if m:
        return m.group(1)
    m = re.search(r"[?&]id=([a-zA-Z0-9_-]+)", url)
    if m:
        return m.group(1)
    return None


def fetch_drive_media_for_display(
    url: str,
    drive=None,
) -> Optional[Tuple[bytes, str]]:
    """
    Fetches image or video bytes from Google Drive using the OAuth-authenticated
    drive object (st.session_state['drive'] — the logged-in user's account).

    Returns (media_bytes, kind) where kind is 'image' or 'video', or None on failure.
    Results are cached in st.session_state by URL to avoid re-fetching on every rerender.

    Access strategy (OAuth only — no public fallback):
      Path 1: PyDrive2 drive.CreateFile().GetContentBytes()   ← primary
      Path 2: HTTPS GET with Bearer token from drive.auth     ← fallback
    """
    if not url or not str(url).strip():
        return None

    cache_key = f"_3d_media_cache_{url}"
    if cache_key in st.session_state:
        return st.session_state[cache_key]

    # Prefer the drive from session state (OAuth-authenticated user)
    _drive = drive or st.session_state.get("drive")
    if not _drive:
        return None

    file_id = _extract_drive_file_id(url)
    media_bytes = None
    mime_type = None

    # --- Path 1: PyDrive2 authenticated download ---
    if file_id:
        tmp_path = None
        try:
            import tempfile
            gfile = _drive.CreateFile({"id": file_id})
            gfile.FetchMetadata(fields="mimeType")
            mime_type = gfile.get("mimeType", "")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as tmp:
                tmp_path = tmp.name
            gfile.GetContentFile(tmp_path)
            with open(tmp_path, "rb") as f:
                media_bytes = f.read()
        except Exception as e:
            print(f"⚠️ PyDrive2 GetContentFile failed for {file_id}: {e}")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

    # --- Path 2: Authenticated HTTPS with Bearer token ---
    if not media_bytes and file_id:
        try:
            auth_instance = _drive.auth
            token = None
            if hasattr(auth_instance, "credentials") and auth_instance.credentials:
                token = auth_instance.credentials.access_token
            if token:
                export_url = f"https://drive.google.com/uc?export=download&id={file_id}"
                resp = requests.get(
                    export_url,
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=20,
                )
                if resp.status_code == 200 and resp.content:
                    media_bytes = resp.content
                    ct = resp.headers.get("Content-Type", "")
                    if ct and "octet-stream" not in ct:
                        mime_type = ct.split(";")[0].strip()
        except Exception as e:
            print(f"⚠️ Bearer-token HTTPS fetch failed for {file_id}: {e}")

    if not media_bytes:
        st.session_state[cache_key] = None
        return None

    # Determine kind: video or image
    is_video = (
        (mime_type and "video" in mime_type)
        or any(url.lower().endswith(ext) for ext in (".mp4", ".mov", ".webm", ".avi", ".mkv"))
    )
    kind = "video" if is_video else "image"

    result = (media_bytes, kind)
    st.session_state[cache_key] = result
    return result



def render_3d_model_search_tab(drive=None):  # noqa: C901
    """
    Renders 3D Models Search Tool UI matching Image Search Tool layout with Search by: dropdown.
    """
    st.markdown("## 3D Models Search Tool")

    if "gc" in st.session_state:
        gc = st.session_state["gc"]
    else:
        gc = None

    # Always use the central Drive folder for 3D model vectorstore storage/retrieval.
    parent_folder_id = CENTRAL_3D_MODELS_FOLDER_ID
    user_role = st.session_state.get("impersonated_role", st.session_state.get("role"))
    is_admin = (user_role == "Admin")

    # Fetch dynamic categories and extensions
    sheet_categories, sheet_extensions = fetch_sheet_filter_options(DEFAULT_CATALOG_SHEET_URL)

    if not sheet_categories:
        try:
            dbs = load_3d_chroma_db(drive=drive, parent_folder_id=parent_folder_id)
            sheet_categories = get_unique_categories_from_vectorstore(dbs)
        except Exception:
            sheet_categories = []

    if not sheet_extensions:
        sheet_extensions = [".glb", ".blend", ".obj", ".fbx", ".gltf", ".stl"]

    # Exactly 2 Tabs if Admin, 1 Tab if non-Admin
    tab_labels = ["Search Model"]
    if is_admin:
        tab_labels.append("Create/Update vectorstore")

    tabs = st.tabs(tab_labels)
    tab_search = tabs[0]
    tab_admin = tabs[1] if is_admin else None

    # =========================================================================
    # TAB 1: Search Model (Text or Image via Search by: dropdown)
    # =========================================================================
    with tab_search:
        st.markdown("#### Choose Query Type")
        st.info("""
        **Search for 3D models using text queries or by uploading an image.** 

        - **Text Search**: Enter descriptive terms (e.g., "pipe wrench", "industrial generator", "HVAC unit")
        - **Image Search**: Upload an image or paste a URL to find visually similar 3D models
        - **Note**: A search query is required - filters work as additional refinements on top of your search
        """)

        search_mode = st.selectbox("Search by:", ["Text", "Image"], key="3d_search_by_mode")

        query_text = None
        query_image = None

        if search_mode == "Text":
            query_text = st.text_input(
                "Enter your search query",
                placeholder="e.g., pipe wrench, industrial generator, HVAC unit",
                help="Required: Enter descriptive terms for the 3D models you want to find.",
                key="3d_text_query_input",
            )
        else:
            image_source = st.selectbox("Select image input method", ["Upload", "Paste URL"], key="3d_image_source_mode")

            if image_source == "Upload":
                uploaded_file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg", "webp"], key="3d_uploaded_file")
                if uploaded_file:
                    try:
                        query_image = Image.open(uploaded_file).convert("RGB")
                        st.image(query_image, caption="Uploaded Query Image", use_container_width=True)
                    except Exception as e:
                        st.error(f"Couldn't read image: {e}")
            else:
                image_url = st.text_input("Paste Image URL & Press Enter to Continue", key="3d_image_url_input")
                if image_url:
                    try:
                        # Detect Google Drive URLs and extract file ID
                        _gdrive_match = re.search(r"(?:/d/|id=)([a-zA-Z0-9_-]{25,})", image_url)
                        _drive = st.session_state.get("drive")
                        if _gdrive_match and _drive:
                            _file_id = _gdrive_match.group(1)
                            _token = _drive.auth.credentials.access_token
                            _dl_url = f"https://www.googleapis.com/drive/v3/files/{_file_id}?alt=media"
                            response = requests.get(_dl_url, headers={"Authorization": f"Bearer {_token}"}, timeout=20)
                        else:
                            response = requests.get(image_url, timeout=10)

                        if response.status_code == 200:
                            query_image = Image.open(BytesIO(response.content)).convert("RGB")
                            st.image(query_image, caption="Image from URL", use_container_width=True)
                        elif response.status_code == 403 and _gdrive_match:
                            st.warning("Google Drive file is not accessible with your account. Check sharing permissions.")
                        else:
                            st.warning(f"Failed to fetch image from URL (HTTP {response.status_code}).")
                    except Exception as e:
                        st.error(f"Error loading image from URL: {e}")

        st.markdown("#### Number of Models to Retrieve")
        k_results = st.number_input("How many models?", min_value=1, max_value=50, value=5, step=1, key="3d_num_models_k")

        category_filter = []
        extension_filter = []
        with st.expander("Apply Filters (Optional)", expanded=False):
            st.caption("Narrow your search by Category or File Extension.")
            category_filter = st.multiselect(
                "Category",
                options=sheet_categories,
                default=[],
                key="3d_cat_filter_multiselect",
            )
            extension_filter = st.multiselect(
                "Extension",
                options=sheet_extensions,
                default=[],
                key="3d_ext_filter_multiselect",
            )

        if st.button("Search 3D Models", type="primary", key="3d_search_models_btn"):
            if search_mode == "Text" and not (query_text and query_text.strip()):
                st.warning("Please enter a search query.")
            elif search_mode == "Image" and query_image is None:
                st.warning("Please upload an image or provide a valid image URL.")
            else:
                started = time.perf_counter()
                try:
                    with st.spinner("Searching relevant 3D models..."):
                        results = search_vectorstore_3d(
                            search_type="text" if search_mode == "Text" else "image",
                            query_text=query_text.strip() if query_text else None,
                            query_image=query_image if search_mode == "Image" else None,
                            category_filter=category_filter if category_filter else None,
                            extension_filter=extension_filter if extension_filter else None,
                            top_n=int(k_results),
                            drive=drive,
                            parent_folder_id=parent_folder_id,
                        )
                    track_tool_action(
                        "3D Models Search",
                        "search_models",
                        run_mode="tool",
                        duration_seconds=time.perf_counter() - started,
                    )
                except Exception as exc:
                    track_tool_action(
                        "3D Models Search",
                        "search_models",
                        run_mode="tool",
                        duration_seconds=time.perf_counter() - started,
                        error_message=str(exc)[:500],
                    )
                    st.error(f"Search failed: {exc}")
                    st.text(traceback.format_exc())
                    results = []

                st.session_state["3d_model_search_results"] = results

        # ---------------------------------------------------------------------
        # Results Display — 2-column uniform thumbnail card grid (Search Model tab only)
        # ---------------------------------------------------------------------
        results = st.session_state.get("3d_model_search_results", [])
        if results:
            st.markdown("---")
            st.markdown(
                f"<p style='color:#6b7280;font-size:0.82rem;margin-bottom:0.75rem'>"
                f"{len(results)} result(s)</p>",
                unsafe_allow_html=True,
            )

            COLS_PER_ROW = 2
            THUMB_H = 200  # px — fixed height for all thumbnails

            # Placeholder HTML blocks (same fixed height, no image)
            def _placeholder_html(icon: str, label: str) -> str:
                return (
                    f"<div style='height:{THUMB_H}px;background:#1a1d2e;border-radius:8px 8px 0 0;"
                    f"display:flex;flex-direction:column;align-items:center;justify-content:center;"
                    f"color:#374151;gap:6px'>"
                    f"<span style='font-size:2rem'>{icon}</span>"
                    f"<span style='font-size:0.72rem'>{label}</span>"
                    f"</div>"
                )

            def _image_html(b64: str) -> str:
                return (
                    f"<div style='height:{THUMB_H}px;background:#0e1117;border-radius:8px 8px 0 0;"
                    f"display:flex;align-items:center;justify-content:center;overflow:hidden'>"
                    f"<img src='data:image/png;base64,{b64}' "
                    f"style='max-width:100%;max-height:100%;object-fit:contain;display:block'/>"
                    f"</div>"
                )

            def _video_html(b64: str) -> str:
                return (
                    f"<div style='height:{THUMB_H}px;background:#0e1117;border-radius:8px 8px 0 0;"
                    f"display:flex;align-items:center;justify-content:center;overflow:hidden'>"
                    f"<video src='data:video/mp4;base64,{b64}' "
                    f"autoplay loop muted playsinline "
                    f"style='max-width:100%;max-height:100%;object-fit:contain;display:block'></video>"
                    f"</div>"
                )

            rows = [results[i : i + COLS_PER_ROW] for i in range(0, len(results), COLS_PER_ROW)]

            for row_items in rows:
                cols = st.columns(COLS_PER_ROW, gap="small")
                for col, item in zip(cols, row_items):
                    new_file_name = item.get("new_file_name", "3D Model")
                    cleaned_model = item.get("cleaned_model", "")
                    preview_url   = item.get("preview_media_url", "")
                    drive_url     = item.get("3d_model_drive_url", "#")

                    display_name = cleaned_model or new_file_name

                    with col:
                        with st.container(border=True):
                            # ── Thumbnail (fixed height, object-fit:cover) ─────
                            if preview_url:
                                media_result = fetch_drive_media_for_display(preview_url, drive=drive)
                                if media_result:
                                    media_bytes, media_kind = media_result
                                    if media_kind == "video":
                                        try:
                                            import base64 as _b64
                                            v_b64 = _b64.b64encode(media_bytes).decode()
                                            st.markdown(_video_html(v_b64), unsafe_allow_html=True)
                                        except Exception:
                                            st.video(media_bytes, autoplay=True, loop=True, muted=True)
                                    else:
                                        try:
                                            import base64 as _b64
                                            pil_img = Image.open(BytesIO(media_bytes)).convert("RGB")
                                            buf = BytesIO()
                                            pil_img.save(buf, format="JPEG", quality=85)
                                            b64str = _b64.b64encode(buf.getvalue()).decode()
                                            st.markdown(_image_html(b64str), unsafe_allow_html=True)
                                        except Exception:
                                            st.markdown(
                                                _placeholder_html("⚠️", "render error"),
                                                unsafe_allow_html=True,
                                            )
                                else:
                                    st.markdown(
                                        _placeholder_html("🔒", "preview restricted"),
                                        unsafe_allow_html=True,
                                    )
                            else:
                                st.markdown(
                                    _placeholder_html("📦", "no preview"),
                                    unsafe_allow_html=True,
                                )

                            # ── Hyperlinked Name (bold electric blue with link symbol) ──
                            if drive_url and drive_url != "#":
                                st.markdown(
                                    f"<a href='{drive_url}' target='_blank' rel='noopener noreferrer' "
                                    f"style='color:#2563eb;font-size:0.95rem;font-weight:700;width:100%;"
                                    f"text-decoration:none;display:block;margin:0.6rem 0 0.3rem;line-height:1.3;"
                                    f"white-space:nowrap;overflow:hidden;text-overflow:ellipsis;' "
                                    f"onmouseover=\"this.style.color='#1d4ed8';this.style.textDecoration='underline'\" "
                                    f"onmouseout=\"this.style.color='#2563eb';this.style.textDecoration='none'\" "
                                    f"title='Open {display_name} in Google Drive'>🔗 {display_name}</a>",
                                    unsafe_allow_html=True,
                                )
                            else:
                                st.markdown(
                                    f"<p style='color:#2563eb;font-size:0.95rem;font-weight:700;width:100%;"
                                    f"margin:0.6rem 0 0.3rem;line-height:1.3;"
                                    f"white-space:nowrap;overflow:hidden;text-overflow:ellipsis;' "
                                    f"title='{display_name}'>{display_name}</p>",
                                    unsafe_allow_html=True,
                                )

            # pad remaining empty columns in last row
            last_row = rows[-1] if rows else []
            empty_slots = COLS_PER_ROW - len(last_row)
            if empty_slots and rows:
                for _ in range(empty_slots):
                    cols[COLS_PER_ROW - empty_slots].empty()

        elif "3d_model_search_results" in st.session_state:
            st.warning("No matching 3D models found.")

    # =========================================================================
    # TAB 2 (Admin Only): Create/Update vectorstore
    # =========================================================================
    if is_admin and tab_admin:
        with tab_admin:
            st.caption("Vectorstore Management — Load Google Sheet data to check vectorization status, then Create or Update the unified 3D vectorstore (3d_models_unified).")

            admin_sheet_link = st.text_input(
                "Google Sheet Link (3D Models Catalog)",
                value=DEFAULT_CATALOG_SHEET_URL,
                key="admin_3d_sheet_link_tab",
            )

            # Step 1: Load Sheet Data
            if st.button("Load Sheet", key="btn_load_3d_sheet_stats"):
                if not gc or not admin_sheet_link:
                    st.error("Google Service Account credentials (gc) and Sheet Link are required.")
                else:
                    with st.spinner("Loading sheet data and inspecting vectorized status..."):
                        try:
                            sheet_df, stats = get_3d_sheet_info_and_stats(gc, admin_sheet_link)
                            st.session_state["3d_sheet_loaded_df"] = sheet_df
                            st.session_state["3d_sheet_loaded_stats"] = stats
                            st.success("Google Sheet loaded successfully!")
                        except Exception as e:
                            st.error(f"Failed to load sheet: {e}")

            # Display Metric Stats if loaded
            if "3d_sheet_loaded_stats" in st.session_state:
                stats = st.session_state["3d_sheet_loaded_stats"]

                col_m1, col_m2, col_m3 = st.columns(3)
                with col_m1:
                    st.metric("Total 3D Models", stats["total_rows"])
                with col_m2:
                    st.metric("Processed (Vectorized = TRUE)", stats["processed_rows"])
                with col_m3:
                    st.metric("Unprocessed Rows", stats["unprocessed_rows"])

            if "3d_last_run_summary" in st.session_state:
                summary = st.session_state["3d_last_run_summary"]
                st.success(f"✅ {summary['title']}")
                st.info(f"💰 Estimated embedding cost: **${summary['cost']:.4f} USD** — ⏱️ Time taken: **{summary['elapsed']}** ({summary['count']} models processed)")

            st.markdown("---")

            # Step 2: Ingestion Action Buttons
            col_btn1, col_btn2 = st.columns(2)

            with col_btn1:
                if st.button("🚀 Create Vectorstore (Process All)", key="btn_create_3d_vstore_tab"):
                    if not gc or not admin_sheet_link:
                        st.error("Google Service Account credentials (gc) and Sheet Link are required.")
                    else:
                        status_placeholder = st.status("Initializing vectorstore creation...", expanded=True)
                        prog_bar = st.progress(0.0, text="Preparing models for indexing...")
                        _t0_create = time.perf_counter()
                        def _prog_cb_create(curr, tot, cost=0.0):
                            pct = min(1.0, max(0.0, curr / max(tot, 1)))
                            elapsed = time.perf_counter() - _t0_create
                            elapsed_str = f"{int(elapsed // 60)}m {int(elapsed % 60)}s"
                            prog_bar.progress(pct, text=f"Indexing 3D models: {curr}/{tot} ({int(pct * 100)}%) — {elapsed_str} — Est. cost: ${cost:.4f}")
                            status_placeholder.update(label=f"Indexed {curr}/{tot} ({int(pct * 100)}%) — {elapsed_str} elapsed — Est. cost: ${cost:.4f}", state="running")
                        try:
                            res = create_vectorstore_3d(gc, drive, admin_sheet_link, parent_folder_id, batch_size=50, progress_callback=_prog_cb_create)
                            cost = res.get('cost_usd', 0.0)
                            elapsed = time.perf_counter() - _t0_create
                            elapsed_str = f"{int(elapsed // 60)}m {int(elapsed % 60)}s"
                            st.session_state["3d_last_run_summary"] = {
                                "title": f"Vectorstore Created! {res['indexed_rows']} of {res['total_rows']} models indexed.",
                                "cost": cost,
                                "elapsed": elapsed_str,
                                "count": res['indexed_rows']
                            }
                            track_tool_action("3D Models Search", "create_vectorstore", res)
                            st.cache_data.clear()
                            st.session_state.pop("3d_sheet_loaded_stats", None)
                            st.rerun()
                        except Exception as e:
                            status_placeholder.update(label="Failed to create vectorstore", state="error")
                            st.error(f"Failed to create vectorstore: {e}")

            with col_btn2:
                if st.button("🔄 Update Vectorstore (Process Unprocessed)", key="btn_update_3d_vstore_tab"):
                    if not gc or not admin_sheet_link:
                        st.error("Google Service Account credentials (gc) and Sheet Link are required.")
                    else:
                        status_placeholder = st.status("Scanning for unprocessed entries...", expanded=True)
                        prog_bar = st.progress(0.0, text="Preparing models for update...")
                        _t0_update = time.perf_counter()
                        def _prog_cb_update(curr, tot, cost=0.0):
                            pct = min(1.0, max(0.0, curr / max(tot, 1)))
                            elapsed = time.perf_counter() - _t0_update
                            elapsed_str = f"{int(elapsed // 60)}m {int(elapsed % 60)}s"
                            prog_bar.progress(pct, text=f"Indexing unprocessed models: {curr}/{tot} ({int(pct * 100)}%) — {elapsed_str} — Est. cost: ${cost:.4f}")
                            status_placeholder.update(label=f"Indexed {curr}/{tot} ({int(pct * 100)}%) — {elapsed_str} elapsed — Est. cost: ${cost:.4f}", state="running")
                        try:
                            res = update_vectorstore_3d(gc, drive, admin_sheet_link, parent_folder_id, batch_size=50, progress_callback=_prog_cb_update)
                            cost = res.get('cost_usd', 0.0)
                            elapsed = time.perf_counter() - _t0_update
                            elapsed_str = f"{int(elapsed // 60)}m {int(elapsed % 60)}s"
                            st.session_state["3d_last_run_summary"] = {
                                "title": f"Vectorstore Updated! {res['indexed_rows']} newly indexed models.",
                                "cost": cost,
                                "elapsed": elapsed_str,
                                "count": res['indexed_rows']
                            }
                            track_tool_action("3D Models Search", "update_vectorstore", res)
                            st.cache_data.clear()
                            st.session_state.pop("3d_sheet_loaded_stats", None)
                            st.rerun()
                        except Exception as e:
                            status_placeholder.update(label="Failed to update vectorstore", state="error")
                            st.error(f"Failed to update vectorstore: {e}")





