import streamlit as st
from agents.vector_store_image_search.graphics_retriever import graphics_retriever
from agents.vector_store_image_search.graphics_retriever_agent import graphics_retriever_agent
from agents.vector_store_image_search.create_vectorstore import build_vectorstore_and_upload, update_vectorstore, is_valid_folderid, chroma_db_exists
from agents.vector_store_image_search.graphics_search_graph import run_graphics_search_graph
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
gc = gspread.service_account_from_dict(sa_dict)


# ----------------- Streamlit App ----------------- #

st.session_state["drive"] = drive
st.session_state["gc"] = gc

sheet = st.session_state.get("sheet")

# Define task visibility by role
st.markdown("## Image Search Tool")
st.markdown("Use this tool to search and retrieve relevant images based on text queries.")


# --- Define task visibility by role ---
authenticated_roles = {
    "Editor": "Editor",
    "Admin": "Admin",
    "Content Head": "ch", 
    "Instructional Designer": "id", 
    "Visual Designer": "vd",
}

# Initialize session state for role if not already set
if "role" not in st.session_state:
    st.session_state.role = None

role = st.session_state.role

# Determine tasks based on role
task_options = []

if role in ["Editor", "Admin"]:
    task_options = ["Create Vectorstore", "Update Vectorstore", "Search Images"]
elif role == "Content Head":  # Content Head
    task_options = ["Update Vectorstore", "Search Images"]
elif role in ["Instructional Designer", "Visual Designer"]:  # Instructional or Visual Designer
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
            gc = gspread.service_account_from_dict(sa_dict)
            sheet = gc.open_by_url(sheet_url)
            st.session_state["sheet"] = sheet
            st.success("Sheet loaded successfully.")
        except Exception as e:
            st.error(f"Failed to load sheet: {e}")

    sheet = st.session_state.get("sheet")

central_folder_id = '1QS6PmCESfgFWNNEpRJDUB0E-t8iMAatH'


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
                    build_vectorstore_and_upload(sheet, drive)
                st.success("✅ Vectorstore built and uploaded successfully!")
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
                update_vectorstore(sheet, drive)
                st.success("Vectorstore updated.")
            else:
                st.info("No updates needed. Vectorstore is up to date.")

# -------------------- Task: Search Images -------------------- #
elif task == "Search Images":
    

    # User inputs
    st.markdown("#### Enter Search Query")
    query = st.text_input("What are you looking for?(Press enter to continue)", placeholder="e.g., ventilation duct")

    st.markdown("#### Number of images to retrieve")
    k = st.number_input("How many images?", min_value=1, max_value=20, value=5, step=1)

    # Filters in expandable section
    filters = {}
    selected_mime_types = []
    image_title_keyword = ""
    # selected_image_types = []
    # unique_image_types = []

    with st.expander("Apply Filters (Optional)", expanded=False):
        st.caption("Narrow your search by file type, title, or visual category.")

        selected_mime_types = st.multiselect(
            "Mime type",
            options=["image/png", "image/jpeg", "image/webp", "image/gif"],
            help="Filter by file format (e.g. PNG or JPEG)"
        )

        image_title_keyword = st.text_input(
            "Image Title Keyword",
            help="Enter a word or phrase that appears in the image's *title* — the descriptive name assigned to an image, like 'Electrical Tools' or 'High Voltage Sign'. This helps narrow results by topic or concept."
        )

        # unique_image_types will be set after first search, so keep it empty for now
        # selected_image_types = st.multiselect(
        #     "Image Type",
        #     options=unique_image_types,
        #     help="Select one or more image types (e.g., Diagram, Icon, Logo)"
        # )

    if selected_mime_types:
        filters["mime_type"] = selected_mime_types
    if image_title_keyword:
        filters["image_title"] = image_title_keyword
    # if selected_image_types:
    #     filters["image_type"] = selected_image_types

    # results, image_types = graphics_retriever(query, drive, k, filters)
    # unique_image_types = sorted(set(image_types)) if image_types else []

    # Toggle to choose search mode
    # use_agent = st.toggle("Use Graphics Search Agent", value=False)
    # use_graph = st.toggle("Use LangGraph Search", value=False)

    # Button to execute search
    run_search = st.button("Run Search")

    results = None
    if run_search:
    #     if not query:
    #         st.warning("Please enter a search query.")
    #     else:
    #         mode = (
    #             'LangGraph Search' if use_graph else
    #             ('Graphics Search Agent' if use_agent else 'Graphics Retriever')
    #         )
    #         with st.spinner(f"Searching images using {mode}..."):
    #             if use_graph:
    #                 results = run_graphics_search_graph(
    #                     query=query,
    #                     drive=drive,
    #                     k=k,
    #                     llm="gemini_2_flash",
    #                     max_turns=3,
    #                     filters=filters,
    #                 )
    #             elif use_agent:
    #                 results = graphics_retriever_agent(
    #                     query=query,
    #                     drive=drive,
    #                     llm="gemini_2_flash",
    #                     k=k,
    #                     max_turns=3,
    #                     filters=filters,
    #                     verbose=False
    #                 )
    #             else:
        results = graphics_retriever(query=query, drive=drive, k=k, filters=filters)

        # Display results
        if results:
            st.subheader(f"Top {len(results)} Results")
            cols = st.columns(2)
            for idx, img_data in enumerate(results):
                with cols[idx % 2]:
                    image = img_data["image"]
                    metadata = img_data["metadata"]

                    name = metadata.get("name", f"Image {idx+1}")
                    url = metadata.get("drive_url", "#")
                    short_name = name if len(name) <= 60 else name[:57] + "..."

                    st.image(image, use_container_width=True)
                    st.markdown(
                        f"""
                        <div style='text-align: center; margin-top: 10px; margin-bottom: 30px;'>
                            <a href='{url}' target='_blank' style='text-decoration: none; font-size: 18px; font-weight: bold; color: #1a73e8;'>
                                {idx + 1}. {short_name}
                            </a>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
        else:
            st.warning("No images found.")