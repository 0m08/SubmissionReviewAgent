from langchain.vectorstores import Chroma
from tqdm import tqdm
import pandas as pd
from services.embedding_service import get_embedding_model
from agents.vector_store_image_search.create_vectorstore import upload_folder_to_drive, download_folder_from_drive
import os
import json
from services.sheets_service import (
    get_sheet_data_and_df,
    save_to_sheet,
)
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from config.logging_config import contextual_logger, log_step
logger = contextual_logger(__name__, agent="curriculum_mapping")
step = lambda name: log_step(logger, name)


def create_nextech_vectorstore(sheet, drive):
    """
    Create a vectorstore from NexTech curriculum sheets.
    Each viable tab must contain Course Name and Course Link.
    """

    embedding_function = get_embedding_model()
    logger.info("Embedding model initialized for NexTech vectorization.")

    parent_folder_id = "1PiAHiA15CROr9haNh61fDOv9R-7OBOj-"

    # Ensure vectorstore folder exists
    file_list = drive.ListFile({
        "q": f"title='Vectorstore files - NexTech' and '{parent_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
    }).GetList()

    if file_list:
        vectorstore_folder_id = file_list[0]["id"]
    else:
        folder = drive.CreateFile({
            "title": "Vectorstore files - NexTech",
            "parents": [{"id": parent_folder_id}],
            "mimeType": "application/vnd.google-apps.folder"
        })
        folder.Upload()
        vectorstore_folder_id = folder["id"]

    # Initialize Chroma
    local_chroma_root = "/tmp/temp_chroma_folder"
    local_chroma_path = os.path.join(local_chroma_root, "chroma_nextech_db")
    os.makedirs(local_chroma_path, exist_ok=True)

    chroma_db = Chroma(
        embedding_function=embedding_function,
        collection_name="nextech_courses",
        persist_directory=local_chroma_path
    )

    # Iterate through all tabs
    for worksheet in sheet.worksheets():
        tab_name = worksheet.title
        logger.info("Processing tab: %s", tab_name)

        rows = worksheet.get_all_values()
        if not rows or len(rows) < 2:
            continue

        headers = [h.strip() for h in rows[0]]
        data = rows[1:]

        df = pd.DataFrame(data, columns=headers)

        # Viability check
        if "Course Name" not in df.columns:
            logger.info("Skipping tab %s — missing required columns.", tab_name)
            continue

        # Ensure tracking columns
        if "vectorized" not in df.columns:
            df["vectorized"] = "FALSE"
        if "embedding_ts" not in df.columns:
            df["embedding_ts"] = ""

        for idx, row in df.iterrows():
            if row["vectorized"] == "TRUE":
                continue

            try:
                # Page content
                page_content = " ".join(filter(None, [
                    str(row.get("Course Name", "")),
                    str(row.get("Description", ""))
                ])).strip()

                if not page_content:
                    continue

                # Metadata (strictly limited)
                metadata = {
                    "course_name": row.get("Course Name"),
                    "description": row.get("Description"),
                    "trade": row.get("Trade"),
                    "category": row.get("Category"),
                    "duration": row.get("Duration"),
                    "tags": row.get("Tags"),
                    "level": row.get("Level"),
                    "course_link": row.get("Course Link"),
                    "source_content": row.get("Source Content"),
                    "tab_name": tab_name
                }

                # Remove None values
                metadata = {k: v for k, v in metadata.items() if v}

                doc_id = f"{tab_name}_{idx}"

                chroma_db.add_texts(
                    texts=[page_content],
                    metadatas=[metadata],
                    ids=[doc_id]
                )

                df.at[idx, "vectorized"] = "TRUE"
                df.at[idx, "embedding_ts"] = datetime.now().isoformat()

            except Exception as e:
                logger.error("Error vectorizing row %s in %s: %s", idx, tab_name, e)

        # Write back to sheet
        worksheet.update([df.columns.values.tolist()] + df.values.tolist())

    # Persist and upload
    chroma_db.persist()
    upload_folder_to_drive(local_chroma_path, vectorstore_folder_id, drive)

    logger.info("NexTech vectorstore created successfully.")


def update_nextech_vectorstore(sheet, drive):
    """
    Incrementally update NexTech vectorstore with newly added tabs or rows.
    """

    logger.info("Starting NexTech vectorstore update.")

    parent_folder_id = "1SoJDL08Wa7sQq9bCsHcB1bNzSzKj3Z1y"
    local_chroma_root = "/tmp/temp_chroma_folder"
    local_chroma_path = os.path.join(local_chroma_root, "chroma_nextech_db")
    os.makedirs(local_chroma_root, exist_ok=True)

    # ------------------ Download existing vectorstore ------------------
    file_list = drive.ListFile({
        "q": f"title='chroma_nextech_db' and '{parent_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
    }).GetList()

    if file_list:
        chroma_folder_id = file_list[0]["id"]
        download_folder_from_drive(chroma_folder_id, local_chroma_path, drive)
        logger.info("Existing NexTech vectorstore downloaded.")
    else:
        chroma_folder_id = None
        logger.info("No existing NexTech vectorstore found.")

    # ------------------ Initialize Chroma ------------------
    embedding_function = get_embedding_model()
    chroma_db = Chroma(
        embedding_function=embedding_function,
        collection_name="nextech_courses",
        persist_directory=local_chroma_path
    )

    # ------------------ Process tabs ------------------
    for worksheet in sheet.worksheets():
        tab_name = worksheet.title
        df = pd.DataFrame(worksheet.get_all_records())

        # Viability check
        if "Course Name" not in df.columns or "Course Link" not in df.columns:
            continue

        # Ensure tracking columns
        if "vectorized" not in df.columns:
            df["vectorized"] = "FALSE"
        if "embedding_ts" not in df.columns:
            df["embedding_ts"] = ""

        for idx, row in df.iterrows():
            if row["vectorized"] == "TRUE":
                continue

            try:
                page_content = " ".join(filter(None, [
                    str(row.get("Course Name", "")),
                    str(row.get("Description", ""))
                ])).strip()

                if not page_content:
                    continue

                metadata = {
                    "course_name": row.get("Course Name"),
                    "description": row.get("Description"),
                    "trade": row.get("Trade"),
                    "category": row.get("Category"),
                    "duration": row.get("Duration"),
                    "tags": row.get("Tags"),
                    "level": row.get("Level"),
                    "course_link": row.get("Course Link"),
                    "source_content": row.get("Source Content"),
                    "tab_name": tab_name
                }

                metadata = {k: v for k, v in metadata.items() if v}

                doc_id = f"{tab_name}_{idx}"

                chroma_db.add_texts(
                    texts=[page_content],
                    metadatas=[metadata],
                    ids=[doc_id]
                )

                df.at[idx, "vectorized"] = "TRUE"
                df.at[idx, "embedding_ts"] = datetime.now().isoformat()

            except Exception as e:
                logger.error("Error vectorizing NexTech row %s (%s): %s", idx, tab_name, e)

        worksheet.update([df.columns.tolist()] + df.values.tolist())

    # ------------------ Persist & Upload ------------------
    chroma_db.persist()

    if chroma_folder_id:
        drive.CreateFile({"id": chroma_folder_id}).Delete()

    upload_folder_to_drive(local_chroma_path, parent_folder_id, drive)

    logger.info("NexTech vectorstore update complete.")
