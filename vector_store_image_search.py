import streamlit as st
from agents.vector_store_image_search.vector_store import (graphics_retriever_agent
    # build_vectorstore_and_upload,
    # update_vectorstore,
    # download_image_from_drive,
    # chroma_db_exists, is_valid_folderid, graphics_retriever
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
from agents.research_notes.retriever import load_vector_db_retriever


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

sheet_link = st.text_input("Enter Google Sheet link")

sheet = st.session_state.get(sheet_link)



import streamlit as st



st.markdown("Enter your slide content to search for the most visually relevant image(s).")

# -------------------- User Input -------------------- #
slide_text = st.text_area("Enter slide text or topic:", height=100, placeholder="e.g., A technician fixing an AC unit")

# Optional filters
st.markdown("**Filter Options:**")
selected_mime_types = st.multiselect(
    "Select allowed image types:", ["image/png", "image/jpeg"], default=["image/png", "image/jpeg"]
)

top_k = st.slider("Number of top images to consider (k):", min_value=5, max_value=20, value=10)

# -------------------- Action Button -------------------- #
if st.button("Find Best Image(s)"):
    if not slide_text.strip():
        st.warning("Please enter a valid slide description.")
    else:
        with st.spinner("🔍 Searching and asking the LLM..."):

            # Load retriever only once
            if "retriever" not in st.session_state:
                retriever, _ = load_vector_db_retriever(
                    course_name="text embeddings",
                    course_drive_folder_id="1QS6PmCESfgFWNNEpRJDUB0E-t8iMAatH",
                    drive=drive,
                    sheet=sheet
                )
                st.session_state["retriever"] = retriever

            filters = {"mime_type": selected_mime_types} if selected_mime_types else None

            # Run the agent
            urls = graphics_retriever_agent(
                slide_text=slide_text,
                drive=drive,
                llm_name="gemini_2_flash",
                k=top_k,
                filters=filters,
                verbose=False
            )

        # -------------------- Display Results -------------------- #
        if urls and urls != ["NONE"]:
            st.success(f"🎯 Top {len(urls)} image(s) selected:")
            for i, url in enumerate(urls):
                st.markdown(f"**Image {i+1}:** [Open in Drive]({url})")
                st.markdown("---")
        else:
            st.warning("🤷‍♀️ No suitable images found by the LLM.")
