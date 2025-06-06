import streamlit as st
from agents.vector_store_image_search.vector_store import (
    build_vectorstore_and_upload,
    update_vectorstore,
    search_similar_images_across_all,
    download_image_from_drive,
    chroma_db_exists, is_valid_folderid
)
from services.drive_service import login_with_service_account
from services.sheets_service import get_worksheet_names, get_sheet_data_and_df
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
gc = gspread.service_account_from_dict(sa_dict)

st.session_state["drive"] = drive
st.session_state["gc"] = gc


# -------------------- UI -------------------- #
st.title("Vector Store for Image Search")

menu_option = st.radio("What do you want to do?", ["Create Chroma DB", "Update Chroma DB", "Search Images"])

# Central Google Drive folder ID
central_folder_id = '1QS6PmCESfgFWNNEpRJDUB0E-t8iMAatH'

# --- Option 1: Build Chroma DB --- #
if menu_option == "Create Chroma DB":
    if chroma_db_exists(drive, central_folder_id):
        st.info("Chroma DB already exists in Drive. No need to build.")
    else:
        sheet_link = st.text_input("Enter Google Sheet URL to build Chroma DB:")
        if sheet_link:
            sheet = gc.open_by_url(sheet_link)
            st.session_state["sheet"] = sheet
            if st.button("Build Now"):
                build_vectorstore_and_upload(sheet, drive)
                st.success("Chroma DB built and uploaded.")

# --- Option 2: Update Chroma DB --- #
elif menu_option == "Update Chroma DB":
    sheet_link = st.text_input("Enter Google Sheet URL to check for updates:")
    if sheet_link:
        sheet = gc.open_by_url(sheet_link)
        st.session_state["sheet"] = sheet
        if st.button("Check and Update"):
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
                update_vectorstore(sheet, drive)
                st.success("Chroma DB updated.")
            else:
                st.info("No updates needed. Chroma DB is up to date.")

# --- Option 3: Search Images --- #
elif menu_option == "Search Images":
    query_text = st.text_input("Enter your search query:", placeholder="Search images...")

    # In-page filters
    st.markdown("### Filter Options")

    # Mimetype filter (multiple select)
    mime_type_options = ["image/gif", "image/png", "image/jpeg", "image/webp"]
    selected_mime_types = st.multiselect("Select Mimetypes:", mime_type_options)

    # Image Title filter (contains search)
    image_title_keyword = st.text_input("Filter by Image Title (contains):", "")

    # Image Type filter (contains search)
    image_type_keyword = st.text_input("Filter by Image Type (contains):", "")

    # Number of top results
    k = st.number_input("Number of top results:", min_value=1, max_value=10, value=5, step=1)

    # Search button
    if st.button("Search"):
        # Gather filter conditions
        filters = {}
        if selected_mime_types:
            filters["mime_type"] = selected_mime_types
        if image_title_keyword:
            filters["image_title"] = image_title_keyword
        if image_type_keyword:
            filters["image_type"] = image_type_keyword
        
        with st.spinner("Searching for similar images..."):
            # Search for similar images with filters
            similar_images = search_similar_images_across_all(query_text, k=k, filters=filters)

            if similar_images:
                st.subheader(f"Top {len(similar_images)} Similar Images")

                # Display in a grid layout
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