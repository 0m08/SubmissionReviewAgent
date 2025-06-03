import streamlit as st
from agents.vector_store_image_search.vector_store import (
    build_vectorstore_and_upload,
    update_vectorstore,
    search_similar_images_across_all,
    download_image_from_drive,
    chroma_db_exists, is_valid_folderid
)
from services.drive_service import login_with_service_account
from pydrive2.drive import GoogleDrive
from dotenv import load_dotenv
import gspread
import base64
import os
import json
import pandas as pd

# -------------------- Auth -------------------- #
load_dotenv()
key_bytes = base64.b64decode(os.environ["GDRIVE_SA_B64"])
sa_json = key_bytes.decode()
sa_dict = json.loads(sa_json)
gauth = login_with_service_account(json_str=sa_json)
gauth.ServiceAuth()
drive = GoogleDrive(gauth)

# -------------------- Helpers -------------------- #
def extract_spreadsheet_id(sheet_url):
    try:
        return sheet_url.split("/d/")[1].split("/")[0]
    except:
        return None

def get_spreadsheet(sheet_url):
    sheet_id = extract_spreadsheet_id(sheet_url)
    if not sheet_id:
        st.error("Invalid Google Sheet URL.")
        return None
    try:
        client = gspread.authorize(gauth)
        return client.open_by_key(sheet_id)
    except Exception as e:
        st.error(f"Failed to open Google Sheet: {e}")
        return None

def check_vectorstore_update_required(spreadsheet):
    for sheet in spreadsheet.worksheets():
        if not is_valid_folderid(sheet.title):
            continue
        df = pd.DataFrame(sheet.get_all_records())
        if 'Image Description' not in df.columns:
            continue
        if 'vectorized' not in df.columns or 'embedding_ts' not in df.columns:
            return True
        mask = (df['vectorized'] != 'TRUE') & df['Image Description'].str.strip().astype(bool)
        if mask.any():
            return True
    return False

# -------------------- UI -------------------- #
st.title("Vector Store for Image Search")

menu_option = st.radio("What do you want to do?", ["Create Chroma DB", "Update Chroma DB", "Search Images"])

# Central Google Drive folder ID (for checking DB presence)
central_folder_id = '1QS6PmCESfgFWNNEpRJDUB0E-t8iMAatH'

# --- Option 1: Build Chroma DB --- #
if menu_option == "Create Chroma DB":
    if chroma_db_exists(drive, central_folder_id):
        st.info("Chroma DB already exists in Drive. No need to build.")
    else:
        sheet_url = st.text_input("Enter Google Sheet URL to build Chroma DB:")
        if sheet_url and st.button("Build Now"):
            spreadsheet = get_spreadsheet(sheet_url)
            if spreadsheet:
                build_vectorstore_and_upload(spreadsheet, drive)
                st.success("Chroma DB built and uploaded.")

# --- Option 2: Update Chroma DB --- #
elif menu_option == "Update Chroma DB":
    sheet_url = st.text_input("Enter Google Sheet URL to check for updates:")
    if sheet_url and st.button("Check and Update"):
        spreadsheet = get_spreadsheet(sheet_url)
        if spreadsheet:
            if check_vectorstore_update_required(spreadsheet):
                update_vectorstore(spreadsheet, drive)
                st.success("Chroma DB updated.")
            else:
                st.info("No updates needed. Chroma DB is up to date.")

# --- Option 3: Search Images --- #
elif menu_option == "Search Images":
    query_text = st.text_input("Enter your search query:", placeholder="Search images...")
    st.markdown("### Filter Options")
    mime_type_options = ["image/gif", "image/png", "image/jpeg", "image/webp"]
    selected_mime_types = st.multiselect("Select Mimetypes:", mime_type_options)
    image_title_keyword = st.text_input("Filter by Image Title (contains):", "")
    image_type_keyword = st.text_input("Filter by Image Type (contains):", "")
    k = st.number_input("Number of top results:", min_value=1, max_value=10, value=5, step=1)

    if st.button("Search"):
        filters = {}
        if selected_mime_types:
            filters["mime_type"] = selected_mime_types
        if image_title_keyword:
            filters["image_title"] = image_title_keyword
        if image_type_keyword:
            filters["image_type"] = image_type_keyword

        similar_images = search_similar_images_across_all(query_text, k=k, filters=filters)

        if similar_images:
            st.subheader(f"Top {len(similar_images)} Similar Images")
            cols = st.columns(2)
            for idx, img_data in enumerate(similar_images):
                col = cols[idx % 2]
                with col:
                    st.markdown(f"**Name:** {img_data['name']}")
                    st.markdown(f"**Description:** {img_data['description']}")
                    st.markdown(f"**Folder ID:** {img_data['folder_id']}")
                    try:
                        pil_image = download_image_from_drive(img_data['image_id'], drive)
                        st.image(pil_image, width=300)
                    except Exception as e:
                        st.warning(f"Could not display image: {e}")
                    st.markdown(f"[View in Drive]({img_data['drive_url']})")
                    st.markdown("---")
        else:
            st.warning("No similar images found.")
