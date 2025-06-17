import streamlit as st
from agents.vector_store_image_search.vector_store import graphics_retriever, build_vectorstore_and_upload, update_vectorstore, download_image_from_drive, chroma_db_exists, is_valid_folderid, graphics_retriever
from services.drive_service import login_with_service_account
from services.sheets_service import get_worksheet_names, get_sheet_data_and_df
from pydrive2.drive import GoogleDrive
from dotenv import load_dotenv
import gspread
import base64
import os
import json


# Load Google Service Account credentials
load_dotenv()
key_bytes = base64.b64decode(os.environ["GDRIVE_SA_B64"])
sa_json = key_bytes.decode()
sa_dict = json.loads(sa_json)
gauth = login_with_service_account(json_str=sa_json)
gauth.ServiceAuth()
drive = GoogleDrive(gauth)

# ----------------- Streamlit App ----------------- #

# ----------------- Streamlit App ----------------- #
st.title("🔎 Image Search")

# Task selector
st.subheader("Select Task")
task = st.radio("Choose an action:", ["Create Chroma DB", "Update Chroma DB", "Search Images"])

import gspread

# Only require Google Sheet for Step 1 and Step 2
sheet = None
if task in ["Create Chroma DB", "Update Chroma DB"]:
    st.subheader("Step 0: Provide Google Sheet")
    sheet_url = st.text_input("Enter your Google Sheet URL:")

    if sheet_url:
        try:
            gc = gspread.service_account_from_dict(sa_dict)
            sheet = gc.open_by_url(sheet_url)
            st.session_state["sheet"] = sheet
            st.success("✅ Sheet loaded successfully.")
        except Exception as e:
            st.error(f"❌ Failed to load sheet: {e}")

    sheet = st.session_state.get("sheet")

central_folder_id = '1ujM1OkJRUcQlg2_ZhRIE-kOZa1m-Qgnc'

# -------------------- Task: Create Chroma DB -------------------- #
if task == "Create Chroma DB":
    if not sheet:
        st.info("📄 Load a Google Sheet above to continue.")
    else:
        db_exists = chroma_db_exists(drive, central_folder_id)
        if db_exists:
            st.success("✅ Chroma DB already exists in Drive.")
        else:
            if st.button("Create Chroma DB"):
                build_vectorstore_and_upload(sheet, drive)
                st.success("✅ Chroma DB built and uploaded.")
                st.session_state["chroma_created"] = True

# -------------------- Task: Update Chroma DB -------------------- #
elif task == "Update Chroma DB":
    if not sheet:
        st.info("📄 Load a Google Sheet above to continue.")
    else:
        if st.button("Check and Update Chroma DB"):
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

# -------------------- Task: Search Images -------------------- #
elif task == "Search Images":
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

        similar_images = graphics_retriever(query_text, drive = drive, k=k, filters=filters)

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
