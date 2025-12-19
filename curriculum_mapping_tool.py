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



llm_model = st.session_state.get("llm_model", "gemini_2_5_flash") or "gemini_2_5_flash"
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

    # 1️Find the vectorstore folder inside the central folder
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

    # 2️Check for the chroma db folder inside the vectorstore folder
    chroma_folders = drive.ListFile({
        'q': (
            f"title='{chroma_db_folder_name}' and "
            f"'{vectorstore_folder_id}' in parents and "
            "mimeType='application/vnd.google-apps.folder' and trashed=false"
        )
    }).GetList()

    return bool(chroma_folders)

import pandas as pd
import streamlit as st

REQUIRED_SHEET_NAME = "Mapped Resources"
REQUIRED_COLUMNS = ["Category", "Course"]


def validate_mapping_sheet(gc, sheet_url):
    try:
        sh = gc.open_by_url(sheet_url)
    except Exception:
        return None, "Invalid or inaccessible Google Sheet URL."

    if REQUIRED_SHEET_NAME not in [ws.title for ws in sh.worksheets()]:
        return None, f"Missing worksheet: '{REQUIRED_SHEET_NAME}'"

    ws = sh.worksheet(REQUIRED_SHEET_NAME)
    df = pd.DataFrame(ws.get_all_records())

    if df.empty:
        return None, "Worksheet exists but is empty."

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        return None, f"Missing columns: {missing_cols}"

    return sh, None


def sample_format_df():
    return pd.DataFrame({
        "Category": [
            "HVAC Industry Overview",
            "HVAC Industry Overview",
            "Electricity",
            "Electricity"
        ],
        "Course": [
            "What is HVAC?",
            "Industry outlook",
            "Magnets and Electricity",
            "Ohm's law"
        ]
    })


st.title("Curriculum Mapping Tool")
st.caption("Map curriculum courses to the best learning resources automatically.")

sheet_url = st.text_input(
    "Google Sheet URL",
    placeholder="https://docs.google.com/spreadsheets/..."
)

mapping_sheet = None
validation_error = None

if sheet_url:
    mapping_sheet, validation_error = validate_mapping_sheet(gc, sheet_url)

    if validation_error:
        st.error(validation_error)
        st.markdown("**Required sheet format:**")
        st.dataframe(sample_format_df(), use_container_width=True)
    else:
        st.success("Sheet validated successfully.")

run_disabled = (not sheet_url) or (validation_error is not None)

if st.button(
    "Run Curriculum Mapping",
    type="primary",
    disabled=run_disabled
):
    run_curriculum_mapping(
        sheet=mapping_sheet,
        drive=drive,
        input_worksheet_name=REQUIRED_SHEET_NAME,
    )
    st.success("Curriculum mapping completed. Results written to the sheet.")

st.markdown("---")

# ==========================================================
# TOGGLE
# ==========================================================
show_advanced = st.toggle("Advanced: Vectorstore Management", value=False)

if show_advanced:
    tabs = st.tabs([
        "HVAC School / Ty Videos",
        "NexTech",
        "SkillCat"
    ])

    # ==========================================================
    # HVAC / TY VIDEOS
    # ==========================================================
    with tabs[0]:
        st.header("HVAC School / Ty Videos Vectorstore")

        video_task = st.selectbox(
            "Choose task",
            ["Create Vectorstore", "Update Vectorstore"],
            key="video_task"
        )

        st.subheader("Provide Google Sheet")
        video_sheet_url = st.text_input(
            "Enter your Google Sheet URL:",
            key="video_sheet_url"
        )

        if video_sheet_url:
            try:
                video_sheet = gc.open_by_url(video_sheet_url)
                st.session_state["video_sheet"] = video_sheet
                st.success("Sheet loaded successfully.")
            except Exception as e:
                st.error(f"Failed to load sheet: {e}")

        video_sheet = st.session_state.get("video_sheet")

        if video_task == "Create Vectorstore":
            if video_sheet and st.button("Create Video Vectorstore", key="btn_create_video"):
                with st.spinner("Building and uploading video vectorstore..."):
                    create_video_vectorstore(video_sheet, drive)
                st.success("Video vectorstore built successfully.")

        elif video_task == "Update Vectorstore":
            if video_sheet and st.button("Update Video Vectorstore", key="btn_update_video"):
                with st.spinner("Updating video vectorstore..."):
                    update_video_vectorstore(video_sheet, drive)
                st.success("Video vectorstore updated successfully.")

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

        st.subheader("Provide Google Sheet")
        nextech_sheet_url = st.text_input(
            "Enter your Google Sheet URL:",
            key="nextech_sheet_url"
        )

        if nextech_sheet_url:
            try:
                nextech_sheet = gc.open_by_url(nextech_sheet_url)
                st.session_state["nextech_sheet"] = nextech_sheet
                st.success("Sheet loaded successfully.")
            except Exception as e:
                st.error(f"Failed to load sheet: {e}")

        nextech_sheet = st.session_state.get("nextech_sheet")

        if nextech_task == "Create Vectorstore":
            if nextech_sheet and st.button("Create NexTech Vectorstore", key="btn_create_nextech"):
                with st.spinner("Creating NexTech vectorstore..."):
                    create_nextech_vectorstore(nextech_sheet, drive)
                st.success("NexTech vectorstore created.")

        elif nextech_task == "Update Vectorstore":
            if nextech_sheet and st.button("Update NexTech Vectorstore", key="btn_update_nextech"):
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

        st.subheader("Provide Google Sheet")
        skillcat_sheet_url = st.text_input(
            "Enter your Google Sheet URL:",
            key="skillcat_sheet_url"
        )

        if skillcat_sheet_url:
            try:
                skillcat_sheet = gc.open_by_url(skillcat_sheet_url)
                st.session_state["skillcat_sheet"] = skillcat_sheet
                st.success("Sheet loaded successfully.")
            except Exception as e:
                st.error(f"Failed to load sheet: {e}")

        skillcat_sheet = st.session_state.get("skillcat_sheet")

        if skillcat_task == "Create Vectorstore":
            if skillcat_sheet and st.button("Create SkillCat Vectorstore", key="btn_create_skillcat"):
                with st.spinner("Creating SkillCat vectorstore..."):
                    create_skillcat_vectorstore(skillcat_sheet, drive)
                st.success("SkillCat vectorstore created.")

        elif skillcat_task == "Update Vectorstore":
            if skillcat_sheet and st.button("Update SkillCat Vectorstore", key="btn_update_skillcat"):
                with st.spinner("Updating SkillCat vectorstore..."):
                    update_skillcat_vectorstore(skillcat_sheet, drive)
                st.success("SkillCat vectorstore updated.")
