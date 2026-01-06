import os
import json
import base64
from datetime import datetime
import gspread
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from pydrive2.drive import GoogleDrive
from services.sheets_service import get_sheet_data_and_df
from services.drive_service import login_with_service_account
from services.pdf_export_service import generate_curriculum_mapping_pdf
from agents.curriculum_mapping_tool.create_skillcat_vectorstore import create_skillcat_vectorstore, update_skillcat_vectorstore
from agents.curriculum_mapping_tool.create_nextech_vectorstore import create_nextech_vectorstore, update_nextech_vectorstore
from agents.course_outline.video_search_tool.update_video_vectorstore import update_video_vectorstore
# from agents.course_outline.video_search_tool.video_retriever import video_retriever
from agents.course_outline.video_search_tool.create_video_vectorstore import create_video_vectorstore
from agents.curriculum_mapping_tool.curriculum_retriever import run_curriculum_mapping, run_curriculum_consolidation
from agents.curriculum_mapping_tool.supabase_export import transform_df_for_supabase
from agents.curriculum_mapping_tool.supabase_service import save_curriculum_to_supabase



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
st.caption("Automatically map your curriculum to the best learning resources from SkillCat, NexTech, and YouTube.")

# Input Section
st.markdown("#### Enter your Google Sheet URL")
sheet_url = st.text_input(
    "Google Sheet URL",
    placeholder="https://docs.google.com/spreadsheets/...",
    label_visibility="collapsed"
)

mapping_sheet = None
validation_error = None

if sheet_url:
    mapping_sheet, validation_error = validate_mapping_sheet(gc, sheet_url)

    if validation_error:
        st.error(validation_error)
        with st.expander("View required sheet format"):
            st.dataframe(sample_format_df(), use_container_width=True)
            st.caption(f"Your sheet must have a worksheet named **'{REQUIRED_SHEET_NAME}'** with columns: **Category** and **Course**")
    else:
        st.success("Sheet validated successfully.")

# Check if mapping is already complete
mapping_complete = st.session_state.get("mapping_results_df") is not None

run_disabled = (not sheet_url) or (validation_error is not None)

# Single button to run both mapping and consolidation
if st.button(
    "Map Curriculum",
    type="primary",
    disabled=run_disabled,
    use_container_width=True
):
    # Step 1: Run curriculum mapping
    st.info("Step 1/2: Mapping concepts to resources...")
    result_df = run_curriculum_mapping(
        sheet=mapping_sheet,
        drive=drive,
        input_worksheet_name=REQUIRED_SHEET_NAME,
    )

    # Store intermediate results
    st.session_state["mapping_results_df"] = result_df
    st.session_state["mapping_sheet_title"] = mapping_sheet.title if mapping_sheet else "Curriculum Mapping"
    st.session_state["mapping_sheet"] = mapping_sheet

    # Step 2: Run consolidation
    st.info("Step 2/2: Consolidating resources by category...")
    consolidated_df = run_curriculum_consolidation(
        sheet=mapping_sheet,
        drive=drive,
        input_worksheet_name=REQUIRED_SHEET_NAME,
    )

    # Store final results
    st.session_state["mapping_results_df"] = consolidated_df

    # Calculate stats for success message
    total_concepts = len(consolidated_df)
    unique_resources = set()
    changes = 0

    for _, row in consolidated_df.iterrows():
        original = str(row.get("Best Resource", "")).strip()
        consolidated = str(row.get("Consolidated Resource", "")).strip()

        # Count unique resources
        if consolidated:
            try:
                res = json.loads(consolidated)
                name = res.get("name", "")
                if name:
                    unique_resources.add(name)
            except:
                pass

        # Count consolidation changes
        if original and consolidated and original != consolidated:
            try:
                orig_json = json.loads(original)
                cons_json = json.loads(consolidated)
                if orig_json.get("name") != cons_json.get("name"):
                    changes += 1
            except:
                pass

    st.success(f"Mapping complete! {total_concepts} concepts mapped to {len(unique_resources)} unique resources. Consolidation optimized {changes} assignments.")
    st.rerun()

# Results Section - show after mapping is complete
if st.session_state.get("mapping_results_df") is not None:
    st.markdown("---")

    df = st.session_state["mapping_results_df"]

    # Calculate summary stats
    total_concepts = len(df)
    unique_resources = set()
    for _, row in df.iterrows():
        consolidated = str(row.get("Consolidated Resource", "")).strip()
        best = str(row.get("Best Resource", "")).strip()
        resource_json = consolidated if consolidated else best
        if resource_json:
            try:
                res = json.loads(resource_json)
                name = res.get("name", "")
                if name:
                    unique_resources.add(name)
            except:
                pass

    # Display summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Concepts", total_concepts)
    with col2:
        st.metric("Unique Resources", len(unique_resources))
    with col3:
        categories = df["Category"].nunique()
        st.metric("Categories", categories)

    st.markdown("---")

    # Export Section
    st.markdown("#### Export Results")

    with st.spinner("Generating PDF..."):
        pdf_bytes = generate_curriculum_mapping_pdf(
            df=st.session_state["mapping_results_df"],
            sheet_title=st.session_state.get("mapping_sheet_title", "Curriculum Mapping"),
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"curriculum_mapping_{timestamp}.pdf"

    st.download_button(
        label="Download PDF Report",
        data=pdf_bytes,
        file_name=filename,
        mime="application/pdf",
        type="primary",
        use_container_width=True
    )

    st.caption("The PDF groups resources by category and shows which concepts each resource covers.")

    # Save to SkillCat Platform section
    st.markdown("---")
    st.markdown("#### Save to SkillCat Platform")

    if st.button("Save to SkillCat", type="secondary", use_container_width=True):
        with st.spinner("Saving curriculum to SkillCat..."):
            try:
                # Transform data for Supabase
                export_data = transform_df_for_supabase(
                    df=st.session_state["mapping_results_df"],
                    curriculum_name=st.session_state.get("mapping_sheet_title", "Untitled Curriculum")
                )

                if not export_data["courses"]:
                    st.warning("No SkillCat courses found to save. Ensure the mapping includes SkillCat resources.")
                else:
                    # Save to Supabase
                    result = save_curriculum_to_supabase(
                        curriculum_name=export_data["curriculum_name"],
                        courses=export_data["courses"]
                    )

                    if result["success"]:
                        st.success(f"Saved '{export_data['curriculum_name']}' with {result['courses_count']} SkillCat courses to the platform!")
                    else:
                        st.error(f"Failed to save: {result.get('error', 'Unknown error')}")
            except ValueError as e:
                st.error(f"Data validation error: {str(e)}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")

    st.caption("This saves only SkillCat courses to the SkillCat platform for display in the course catalog.")

st.markdown("---")

# ==========================================================
# TOGGLE
# ==========================================================
# print("Current role:", st.session_state.get("role"))
if st.session_state.get("role") == "Admin":
    show_advanced = st.toggle("Advanced: Vectorstore Management", value=False)
else:
    show_advanced = False

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
