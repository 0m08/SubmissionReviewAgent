import streamlit as st
from agents.vector_store_image_search.graphics_retriever import graphics_retriever
from agents.vector_store_image_search.create_vectorstore import build_vectorstore_and_upload, update_vectorstore, is_valid_folderid, chroma_db_exists, extract_drive_file_id
from agents.vector_store_image_search.create_vectorstore import download_image_from_drive
from agents.vector_store_image_search.langgraph_agent_with_tools import run_graphics_search_graph
from agents.vector_store_image_search.web_image_search_tool import web_image_search_tool
from services.drive_service import login_with_service_account, login_with_oauth2
from services.sheets_service import get_worksheet_names, get_sheet_data_and_df
from pydrive2.drive import GoogleDrive
from dotenv import load_dotenv
import gspread
import base64
import os
import json
import requests
from io import BytesIO
from PIL import Image

# Load Google credentials
load_dotenv()

# Check if we should use OAuth 2.0 or service account
# If OAuth credentials are provided, use OAuth 2.0; otherwise use service account
oauth_client_id = os.getenv("OAUTH_CLIENT_ID")
oauth_client_secret = os.getenv("OAUTH_CLIENT_SECRET")
use_oauth2 = bool(oauth_client_id and oauth_client_secret)

# ----------------- Streamlit App ----------------- #

# Use pre-authenticated credentials from login
if "drive" in st.session_state and "gc" in st.session_state:
    # Use existing authentication from login
    drive = st.session_state["drive"]
    gc = st.session_state["gc"]
    
else:
    # Fallback: authenticate here if not done during login
    st.error("❌ Authentication not found. Please log out and log in again.")
    st.stop()

sheet = st.session_state.get("sheet")

# Define task visibility by role
st.markdown("## SkillCat Graphics Search Tool")
st.markdown("Use this tool to search and retrieve relevant images based on text queries.")

# Version toggle
version = st.radio("Version", ["v1", "v2"], index=0)
if version == "v2":
    root_folder_id = "1IMGr4d8lwux5R_cAWfhVjBV0fTFdWvNi"
else:
    root_folder_id = "1QS6PmCESfgFWNNEpRJDUB0E-t8iMAatH"


# --- Define task visibility by role ---
# Initialize session state for role if not already set
if "role" not in st.session_state:
    st.session_state.role = None

role = st.session_state.role

# Determine tasks based on role
task_options = []

if role in ["Admin Team", "Innovation Team"]:
    task_options = ["Create Vectorstore", "Update Vectorstore", "Search Images"]
elif role == "Content Team":  # Content Team
    task_options = ["Update Vectorstore", "Search Images"]
elif role == "Visual Designer Team":  # Visual Designer Team
    task_options = ["Search Images"]
else:
    st.warning("Your role does not have access to any tasks.")

task = None
if task_options:
    task = st.selectbox("Choose a task:", task_options)


# Only require Google Sheet for Step 1 and Step 2
sheet = None
if task in ["Create Vectorstore", "Update Vectorstore"]:
    st.subheader("Provide Google Sheet")
    sheet_url = st.text_input("Enter your Google Sheet URL:")

    if sheet_url:
        try:
            # Use the already initialized gc from the authentication setup above
            sheet = gc.open_by_url(sheet_url)
            st.session_state["sheet"] = sheet
            st.success("Sheet loaded successfully.")
        except Exception as e:
            st.error(f"Failed to load sheet: {e}")

    sheet = st.session_state.get("sheet")

central_folder_id = root_folder_id


# -------------------- Task: Create Vectorstore -------------------- #
if task == "Create Vectorstore":
    if not sheet:
        st.info("Load a Google Sheet above to continue.")
    else:
        db_exists = chroma_db_exists(drive, central_folder_id)
        if db_exists:
            st.success("Vectorstore already exists in Drive.")
        else:
            if st.button("Create Vectorstore"):
                with st.spinner("⏳ Building and uploading Vectorstore..."):
                    build_vectorstore_and_upload(sheet, drive, root_folder_id=root_folder_id, version=version)
                st.success("Vectorstore built and uploaded successfully!")
                st.session_state["chroma_created"] = True

# -------------------- Task: Update Vectorstore -------------------- #
elif task == "Update Vectorstore":
    if not sheet:
        st.info("Load a Google Sheet above to continue.")
    else:
        if st.button("Check and Update Vectorstore"):
            worksheet_names = get_worksheet_names(sheet)
            update_required = False
            for name in worksheet_names:
                if not is_valid_folderid(name):
                    continue
                _, df = get_sheet_data_and_df(sheet, name)
                if 'Image Description' not in df.columns:
                    continue
                if 'vectorized' not in df.columns or 'embedding_ts' not in df.columns:
                    update_required = True
                    break
                mask = (df['vectorized'] != 'TRUE') & df['Image Description'].str.strip().astype(bool)
                if mask.any():
                    update_required = True
                    break

            if update_required:
                with st.spinner("Updating vectorstore... This may take a few minutes."):
                    if version == "v2":
                        build_vectorstore_and_upload(sheet, drive, root_folder_id=root_folder_id, version=version)
                    else:
                        update_vectorstore(sheet, drive, root_folder_id=root_folder_id)
                st.success("Vectorstore updated.")
            else:
                st.info("No updates needed. Vectorstore is up to date.")


# -------------------- Task: Search Images -------------------- #


elif task == "Search Images":
    st.markdown("#### Choose Query Type")
    st.info("Search for images using text queries or by uploading an image. If you are using an image, you can either upload it from your device or paste a URL to an image from the web or Google Drive.")
    search_mode = st.selectbox("Search by:", ["Text", "Image"])

    query = None
    query_image = None
    

    if search_mode == "Text":
        query = st.text_input("Enter your search query", placeholder="e.g., ventilation duct")

    else:
        image_source = st.selectbox("Select image input method", ["Upload", "Paste URL"])
        # Session state to track and clear image
        if "active_image_source" not in st.session_state:
            st.session_state.active_image_source = image_source

        if st.session_state.active_image_source != image_source:
            # Reset previous input if user switches method
            st.session_state.pop("uploaded_file", None)
            st.session_state.pop("image_url", None)
            query_image = None
            st.session_state.active_image_source = image_source
            
        # === Upload Method ===
        if image_source == "Upload":
            uploaded_file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"], key="uploaded_file")
            if uploaded_file:
                try:
                    query_image = Image.open(uploaded_file).convert("RGB")
                    st.image(query_image, caption="Uploaded Image", use_container_width=True)
                except Exception as e:
                    st.error(f"Couldn't read image: {e}")
                    
        # === URL Method ===
        elif image_source == "Paste URL":
            image_url = st.text_input("Paste Image URL & Press Enter to Continue", key="image_url")
            
            if image_url:
                if "drive.google.com" in image_url:
                    file_id = extract_drive_file_id(image_url)
                    if file_id:
                        try:
                            query_image = download_image_from_drive(drive, file_id)
                            if query_image:
                                st.image(query_image, caption="Image from Google Drive", use_container_width=True)
                            else:
                                st.warning("Could not load image from Drive.")
                        except Exception as e:
                            st.error(f"Error downloading from Drive: {e}")
                    else:
                        st.warning("Invalid Google Drive URL.")
                else:
                    try:
                        response = requests.get(image_url)
                        if response.status_code == 200:
                            query_image = Image.open(BytesIO(response.content)).convert("RGB")
                            st.image(query_image, caption="Image from URL", use_container_width=True)
                        else:
                            st.warning("Failed to fetch image from URL.")
                    except Exception as e:
                        st.error(f"Error loading image from URL: {e}")

    # === Retrieval Options ===
    st.markdown("#### Number of Images to Retrieve")
    k = st.number_input("How many images?", min_value=1, max_value=20, value=5, step=1)

    filters = {}
    with st.expander("Apply Filters (Optional)", expanded=False):
        st.caption("Narrow your search by file type or title.")
        selected_mime_types = st.multiselect("Mime type", options=["image/png", "image/jpeg", "image/webp", "image/gif"])
        if selected_mime_types:
            filters["mime_type"] = selected_mime_types

        image_title_keyword = st.text_input("Image Title Keyword")
        if image_title_keyword:
            filters["image_title"] = image_title_keyword

        if version == "v2":
            course_name = st.text_input("Course Name")
            if course_name:
                filters["course_name"] = course_name
            topic_name = st.text_input("Topic Name")
            if topic_name:
                filters["topic_name"] = topic_name
            stock_type = st.selectbox("Stock Type", ["All", "Stock", "Non Stock"], index=0)
            if stock_type != "All":
                filters["stock_type"] = stock_type

    # === Search Mode Toggles ===
    use_graph = st.toggle("Agent Mode", value=False)

    run_search = st.button("Run Search")

    results = None
    if run_search:
        if not query and not query_image:
            st.warning("Please enter a query or upload an image.")
        else:
            mode = (
                'Agent Mode' if use_graph else 'Graphics Retriever')
            
            with st.spinner(f"Searching images using {mode}..."):
                if use_graph:
                    results = run_graphics_search_graph(
                        query=query,
                        query_image=query_image,
                        drive=drive,
                        k=k,
                        llm="gemini_2_flash",
                        max_turns=3,
                        filters=filters,
                        root_folder_id=root_folder_id
                    )
                else:
                    results = graphics_retriever(
                        query=query,
                        query_image=query_image,
                        drive=drive,
                        k=k,
                        filters=filters,
                        root_folder_id=root_folder_id
                    )
                    # results =web_image_search_tool(query=query, k=k)

    # === Display Results ===
    if results:
        st.subheader(f"Top {len(results)} Results")
        cols = st.columns(2)
        for idx, img_data in enumerate(results):
            with cols[idx % 2]:
                image = img_data["image"]
                metadata = img_data["metadata"]
                name = metadata.get("name", f"Image {idx+1}")
                url = metadata.get("source_url") or metadata.get("drive_url", "#")
                short_name = name if len(name) <= 60 else name[:57] + "..."
                source = img_data.get("metadata", {}).get("source", "web").capitalize()

                st.image(image, use_container_width=True)
                st.markdown(
                    f"""
                    <div style='text-align: center; margin-top: 10px; margin-bottom: 5px;'>
                        <a href='{url}' target='_blank' style='text-decoration: none; font-size: 18px; font-weight: bold; color: #1a73e8;'>
                            {idx + 1}. {short_name}
                        </a>
                    </div>
                    <div style='text-align: center; color: gray; font-size: 14px; margin-bottom: 30px;'>
                        Source: <b>{source} collection</b>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
    elif run_search:
        st.warning("No images found.")

            
            
            

