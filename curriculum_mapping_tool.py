import os
import json
import base64
import gspread
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from pydrive2.drive import GoogleDrive
from services.sheets_service import get_sheet_data_and_df
from services.drive_service import login_with_service_account
from agents.curriculum_mapping_tool.create_skillcat_vectorstore import create_skillcat_vectorstore, update_skillcat_vectorstore
from agents.curriculum_mapping_tool.create_nextech_vectorstore import create_nextech_vectorstore, update_nextech_vectorstore
from agents.course_outline.video_search_tool.update_video_vectorstore import update_video_vectorstore
# from agents.course_outline.video_search_tool.video_retriever import video_retriever
from agents.course_outline.video_search_tool.create_video_vectorstore import create_video_vectorstore
from agents.curriculum_mapping_tool.curriculum_retriever import run_curriculum_mapping



llm_model = st.session_state.get("llm_model", "gemini_2_flash") or "gemini_2_flash"
# --------------------- Auth --------------------- #
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


def chroma_db_exists(drive, central_folder_id, vectorstore_folder_name, chroma_db_folder_name):
    """
    Checks whether a specific Chroma DB folder exists inside a named
    Vectorstore folder within a central parent folder.

    :param drive: GoogleDrive instance
    :param central_folder_id: ID of the central folder containing all vectorstores
    :param vectorstore_folder_name: Name of the vectorstore folder
        (e.g. 'Vectorstore files', 'Vectorstore files - NexTech', 'Vectorstore files - SkillCat')
    :param chroma_db_folder_name: Name of the chroma db folder
        (e.g. 'chroma_video_db', 'chroma_nextech_db', 'chroma_skillcat_db')
    :return: True if the chroma db folder exists, False otherwise
    """

    # 1️⃣ Find the vectorstore folder inside the central folder
    vectorstore_folders = drive.ListFile({
        'q': (
            f"title='{vectorstore_folder_name}' and "
            f"'{central_folder_id}' in parents and "
            "mimeType='application/vnd.google-apps.folder' and trashed=false"
        )
    }).GetList()

    if not vectorstore_folders:
        return False

    vectorstore_folder_id = vectorstore_folders[0]['id']

    # 2️⃣ Check for the chroma db folder inside the vectorstore folder
    chroma_folders = drive.ListFile({
        'q': (
            f"title='{chroma_db_folder_name}' and "
            f"'{vectorstore_folder_id}' in parents and "
            "mimeType='application/vnd.google-apps.folder' and trashed=false"
        )
    }).GetList()

    return bool(chroma_folders)


import streamlit as st
import pandas as pd

st.markdown("## Curriculum Mapping Tool")

st.markdown(
    """
    ### What this tool does
    This agent **automatically maps your curriculum courses** to the **best matching learning resources**
    across **YouTube (HVAC School / Ty Videos)**, **NexTech**, and **SkillCat** using **semantic search**.

    ### How to use it
    1. Prepare a Google Sheet, and on a new tab, name it **Mapped Resources** with two columns:
       - **Category**
       - **Course**
    2. Paste the Google Sheet link below
    3. Click **Run Curriculum Mapping Agent**
    4. Wait for completion  
    5. Results will be written back to your sheet.

    ---
    ⚠️ **Important**
    - This tool assumes vectorstores already exist.
    - Use the tabs below **only if you need to create or update vectorstores**.\
    
    ---
    """
)


# ==========================================================
# 🔹 PRIMARY ACTION: CURRICULUM MAPPING (NOT A TAB)
# ==========================================================

mapping_sheet = None

st.markdown("### Provide Google Sheet for Curriculum Mapping")

sheet_url = st.text_input(
    "Paste Google Sheet URL (must include Category & Course columns)",
    key="mapping_sheet_url"
)


if sheet_url:
    try:
        mapping_sheet = gc.open_by_url(sheet_url)
        st.session_state["mapping_sheet"] = mapping_sheet
        st.success("Sheet loaded successfully.")
    except Exception as e:
        st.error(f"Could not open sheet: {e}")

mapping_sheet = st.session_state.get("mapping_sheet")

if mapping_sheet:
    if st.button("Run Curriculum Mapping Agent", type="primary"):
        run_curriculum_mapping(
            sheet=mapping_sheet,
            drive=drive,
            input_worksheet_name=None, 
            )
    st.success("Curriculum mapping completed. Results written to the sheet.")

# ==========================================================
# 🔹 SECONDARY TOOLS: VECTORSTORE MANAGEMENT
# ==========================================================

st.markdown("---")
st.markdown("## Vectorstore Management")

tabs = st.tabs([
    "🎥 HVAC School / Ty Videos",
    "🏗️ NexTech",
    "📘 SkillCat"
])

central_folder_id = "1PiAHiA15CROr9haNh61fDOv9R-7OBOj-"

# ==========================================================
# HVAC SCHOOL / TY VIDEOS
# ==========================================================

with tabs[0]:
    st.header("HVAC School / Ty Videos Vectorstore")

    video_task = st.selectbox(
        "Choose task",
        ["Create Vectorstore", "Update Vectorstore"],
        key="video_task"
    )

    video_sheet = None
    st.subheader("Provide Google Sheet")
    sheet_url = st.text_input(
        "Enter your Google Sheet URL:",
        key="video_sheet_url"
    )

    if sheet_url:
        try:
            video_sheet = gc.open_by_url(sheet_url)
            st.session_state["video_sheet"] = video_sheet
            st.success("Sheet loaded successfully.")
        except Exception as e:
            st.error(f"Failed to load sheet: {e}")

    video_sheet = st.session_state.get("video_sheet")

    if video_task == "Create Vectorstore":
        if video_sheet and st.button("Create Video Vectorstore"):
            with st.spinner("Building and uploading video vectorstore..."):
                create_video_vectorstore(video_sheet, drive)
            st.success("Video vectorstore built successfully.")

    elif video_task == "Update Vectorstore":
        if video_sheet and st.button("Update Video Vectorstore"):
            with st.spinner("Updating video vectorstore..."):
                update_video_vectorstore(video_sheet, drive)
            st.success("Video vectorstore updated.")

# ==========================================================
# NEXTECH
# ==========================================================

with tabs[1]:
    st.header("NexTech Vectorstore")

    nextech_task = st.selectbox(
        "Choose task",
        ["Create Vectorstore", "Update Vectorstore"],
        key="nextech_task"
    )

    nextech_sheet = None
    st.subheader("Provide Google Sheet")
    sheet_url = st.text_input(
        "Enter your Google Sheet URL:",
        key="nextech_sheet_url"
    )

    if sheet_url:
        try:
            nextech_sheet = gc.open_by_url(sheet_url)
            st.session_state["nextech_sheet"] = nextech_sheet
            st.success("Sheet loaded successfully.")
        except Exception as e:
            st.error(f"Failed to load sheet: {e}")

    nextech_sheet = st.session_state.get("nextech_sheet")

    if nextech_task == "Create Vectorstore":
        if nextech_sheet and st.button("Create NexTech Vectorstore"):
            with st.spinner("Creating NexTech vectorstore..."):
                create_nextech_vectorstore(nextech_sheet, drive)
            st.success("NexTech vectorstore created.")

    elif nextech_task == "Update Vectorstore":
        if nextech_sheet and st.button("Update NexTech Vectorstore"):
            with st.spinner("Updating NexTech vectorstore..."):
                update_nextech_vectorstore(nextech_sheet, drive)
            st.success("NexTech vectorstore updated.")

# ==========================================================
# SKILLCAT
# ==========================================================

with tabs[2]:
    st.header("SkillCat Vectorstore")

    skillcat_task = st.selectbox(
        "Choose task",
        ["Create Vectorstore", "Update Vectorstore"],
        key="skillcat_task"
    )

    skillcat_sheet = None
    st.subheader("Provide Google Sheet")
    sheet_url = st.text_input(
        "Enter your Google Sheet URL:",
        key="skillcat_sheet_url"
    )

    if sheet_url:
        try:
            skillcat_sheet = gc.open_by_url(sheet_url)
            st.session_state["skillcat_sheet"] = skillcat_sheet
            st.success("Sheet loaded successfully.")
        except Exception as e:
            st.error(f"Failed to load sheet: {e}")

    skillcat_sheet = st.session_state.get("skillcat_sheet")

    if skillcat_task == "Create Vectorstore":
        if skillcat_sheet and st.button("Create SkillCat Vectorstore"):
            with st.spinner("Creating SkillCat vectorstore..."):
                create_skillcat_vectorstore(skillcat_sheet, drive)
            st.success("SkillCat vectorstore created.")

    elif skillcat_task == "Update Vectorstore":
        if skillcat_sheet and st.button("Update SkillCat Vectorstore"):
            with st.spinner("Updating SkillCat vectorstore..."):
                update_skillcat_vectorstore(skillcat_sheet, drive)
            st.success("SkillCat vectorstore updated.")
