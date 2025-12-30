from langchain.vectorstores import Chroma
from tqdm import tqdm
from services.embedding_service import get_embedding_model
from agents.vector_store_image_search.create_vectorstore import upload_folder_to_drive, download_folder_from_drive
import os
import pandas as pd
import json
from services.sheets_service import (
    get_sheet_data_and_df,
    save_to_sheet,
)
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from config.logging_config import contextual_logger, log_step


logger = contextual_logger(__name__, agent="course_outline")
step = lambda name: log_step(logger, name)

def create_skillcat_vectorstore(sheet, drive):
    """
    Create a vectorstore from the SkillCat course catalog.
    ONLY vectorizes the tab named '1 Dec'.
    """

    TARGET_TAB_NAME = "1 Dec"

    embedding_function = get_embedding_model()
    logger.info("Embedding model initialized for SkillCat vectorization.")

    parent_folder_id = "1PiAHiA15CROr9haNh61fDOv9R-7OBOj-"

    # Ensure vectorstore folder exists
    file_list = drive.ListFile({
        "q": f"title='Vectorstore files - SkillCat' and '{parent_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
    }).GetList()

    if file_list:
        vectorstore_folder_id = file_list[0]["id"]
    else:
        folder = drive.CreateFile({
            "title": "Vectorstore files - SkillCat",
            "parents": [{"id": parent_folder_id}],
            "mimeType": "application/vnd.google-apps.folder"
        })
        folder.Upload()
        vectorstore_folder_id = folder["id"]

    # Initialize Chroma
    local_chroma_root = "/tmp/temp_chroma_folder"
    local_chroma_path = os.path.join(local_chroma_root, "chroma_skillcat_db")
    os.makedirs(local_chroma_path, exist_ok=True)

    chroma_db = Chroma(
        embedding_function=embedding_function,
        collection_name="skillcat_courses",
        persist_directory=local_chroma_path
    )

    # Get the specific worksheet
    worksheet = sheet.worksheet(TARGET_TAB_NAME)
    df = pd.DataFrame(worksheet.get_all_records())

    # Ensure tracking columns
    if "vectorized" not in df.columns:
        df["vectorized"] = "FALSE"
    if "embedding_ts" not in df.columns:
        df["embedding_ts"] = ""

    # Allowed metadata fields (STRICT)
    allowed_metadata_fields = {
        "id",
        "skillmap_id",
        "course_id",
        "course_week_count",
        "course_name",
        "course_category",
        "course_learning_plan",
        "course_status",
        "course_prerequisites",
        "course_image",
        "course_screenshots",
        "course_description",
        "course_link",
        "course_competencies_level1",
        "course_competencies_level2",
        "course_topics",
        "course_topics_count",
        "course_learning_objectives",
        "course_processes_covered",
        "course_equipment_covered",
        "course_keywords",
        "course_practice_scenarios",
        "created_at",
        "updated_at",
        "course_duration_hours",
        "course_release_date",
        "course_proposed_date",
        "nps_score",
    }

    for idx, row in df.iterrows():
        if row["vectorized"] == "TRUE":
            continue

        try:
            # Page content for embedding
            page_content = " ".join(filter(None, [
                str(row.get("course_name", "")),
                str(row.get("course_description", "")),
                str(row.get("course_topics", "")),
                str(row.get("course_learning_objectives", ""))
            ])).strip()

            if not page_content:
                continue

            # Metadata (only allowed fields that exist)
            metadata = {
                col: row.get(col)
                for col in allowed_metadata_fields
                if col in df.columns and row.get(col)
            }

            doc_id = f"skillcat_{row.get('course_id', idx)}"

            chroma_db.add_texts(
                texts=[page_content],
                metadatas=[metadata],
                ids=[doc_id]
            )

            df.at[idx, "vectorized"] = "TRUE"
            df.at[idx, "embedding_ts"] = datetime.now().isoformat()

        except Exception as e:
            logger.error("Error vectorizing SkillCat row %s: %s", idx, e)

    # Write updates back to sheet
    worksheet.update([df.columns.values.tolist()] + df.values.tolist())

    # Persist and upload
    chroma_db.persist()
    upload_folder_to_drive(local_chroma_path, vectorstore_folder_id, drive)

    logger.info("SkillCat vectorstore created successfully.")


def update_skillcat_vectorstore(sheet, drive):
    """
    Incrementally update SkillCat vectorstore.
    Only processes new rows in tab '1 Dec'.
    """

    logger.info("Starting SkillCat vectorstore update.")

    TARGET_TAB = "1 Dec"
    parent_folder_id = "1SoJDL08Wa7sQq9bCsHcB1bNzSzKj3Z1y"

    local_chroma_root = "/tmp/temp_chroma_folder"
    local_chroma_path = os.path.join(local_chroma_root, "chroma_skillcat_db")
    os.makedirs(local_chroma_root, exist_ok=True)

    # ------------------ Download existing vectorstore ------------------
    file_list = drive.ListFile({
        "q": f"title='chroma_skillcat_db' and '{parent_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
    }).GetList()

    if file_list:
        chroma_folder_id = file_list[0]["id"]
        download_folder_from_drive(chroma_folder_id, local_chroma_path, drive)
    else:
        chroma_folder_id = None

    # ------------------ Initialize Chroma ------------------
    embedding_function = get_embedding_model()
    chroma_db = Chroma(
        embedding_function=embedding_function,
        collection_name="skillcat_courses",
        persist_directory=local_chroma_path
    )

    worksheet = sheet.worksheet(TARGET_TAB)
    df = pd.DataFrame(worksheet.get_all_records())

    # Ensure tracking columns
    if "vectorized" not in df.columns:
        df["vectorized"] = "FALSE"
    if "embedding_ts" not in df.columns:
        df["embedding_ts"] = ""

    allowed_metadata_fields = {
        "id", "skillmap_id", "course_id", "course_week_count",
        "course_name", "course_category", "course_learning_plan",
        "course_status", "course_prerequisites", "course_image",
        "course_screenshots", "course_description", "course_link",
        "course_competencies_level1", "course_competencies_level2",
        "course_topics", "course_topics_count",
        "course_learning_objectives", "course_processes_covered",
        "course_equipment_covered", "course_keywords",
        "course_practice_scenarios", "created_at", "updated_at",
        "course_duration_hours", "course_release_date",
        "course_proposed_date", "nps_score"
    }

    for idx, row in df.iterrows():
        if row["vectorized"] == "TRUE":
            continue

        try:
            page_content = " ".join(filter(None, [
                str(row.get("course_name", "")),
                str(row.get("course_description", "")),
                str(row.get("course_topics", "")),
                str(row.get("course_learning_objectives", ""))
            ])).strip()

            if not page_content:
                continue

            metadata = {
                col: row.get(col)
                for col in allowed_metadata_fields
                if col in df.columns and row.get(col)
            }

            doc_id = f"skillcat_{row.get('course_id', idx)}"

            chroma_db.add_texts(
                texts=[page_content],
                metadatas=[metadata],
                ids=[doc_id]
            )

            df.at[idx, "vectorized"] = "TRUE"
            df.at[idx, "embedding_ts"] = datetime.now().isoformat()

        except Exception as e:
            logger.error("Error vectorizing SkillCat row %s: %s", idx, e)

    worksheet.update([df.columns.tolist()] + df.values.tolist())

    # ------------------ Persist & Upload ------------------
    chroma_db.persist()

    if chroma_folder_id:
        drive.CreateFile({"id": chroma_folder_id}).Delete()

    upload_folder_to_drive(local_chroma_path, parent_folder_id, drive)

    logger.info("SkillCat vectorstore update complete.")
