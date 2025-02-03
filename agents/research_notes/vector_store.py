from langchain_chroma import Chroma
import os
from services.helper_functions import get_short_name
#from services.drive_service import drive
#from services.embedding_service import get_embedding_model

# Vectorstore


# Example helper function to recursively download a folder from Google Drive
# This ensures that the local directory structure mirrors what we have on Drive.
def download_folder_from_drive(folder_id: str, local_path: str, drive) -> None:
    """
    Recursively download all files and subfolders from a given Google Drive folder ID
    into the specified local path.

    :param folder_id: ID of the Google Drive folder to download.
    :param local_path: Path to the local folder where files will be saved.
    :param drive: Authenticated GoogleDrive instance.
    """
    os.makedirs(local_path, exist_ok=True)
    # List all items in the folder
    file_list = drive.ListFile({ 'q': f"'{folder_id}' in parents" }).GetList()
    for item in file_list:
        mime_type = item.get('mimeType')
        title = item.get('title')
        item_id = item.get('id')
        if mime_type == 'application/vnd.google-apps.folder':
            # Create a matching subfolder locally and recurse
            subfolder_path = os.path.join(local_path, title)
            download_folder_from_drive(item_id, subfolder_path, drive)
        else:
            # Download the file into local_path
            local_file_path = os.path.join(local_path, title)
            item.GetContentFile(local_file_path)


def load_vector_db_with_pydrive(course_name: str,
                                root_folder_id: str,
                                embedding_function,
                                drive) -> Chroma:
    """
    Load a Chroma vector database from a nested folder structure on Google Drive.

    The hierarchy is assumed to be:
        root_folder (root_folder_id)
            -> Vectorstore files
                -> chroma_research_db
                    -> chroma.sqlite3 (file)
                    -> <random_subfolder> (folder containing other files)

    Steps:
      1. Locates the folder named 'Vectorstore files' under root_folder_id.
      2. Locates the folder named 'chroma_research_db' inside that folder.
      3. Recursively downloads all files from 'chroma_research_db' into a local
         temp directory.
      4. Initializes a Chroma instance with the local directory.

    :param course_name: The name of the course (used for the Chroma collection name).
    :param root_folder_id: The ID of the root folder containing 'Vectorstore files'.
    :param embedding_function: The embedding function used by Chroma.
    :param drive: Authenticated GoogleDrive instance (PyDrive2).
    :return: A loaded Chroma vector database.
    """

    short_course_name = get_short_name(course_name)

    local_chroma_root = os.path.join("/tmp", "temp_chroma_folder")
    local_chroma_path = os.path.join(local_chroma_root, f"{short_course_name}_chroma_research_db")
    sqlite_db_path = os.path.join(local_chroma_path, "chroma.sqlite3")

    # Check if the database already exists locally
    if os.path.exists(local_chroma_path) and os.path.exists(sqlite_db_path):
        print("Database already exists locally. Skipping download.")
    else:
        print("Database not found locally. Downloading from Google Drive...")
    
        # 1. Find the subfolder named 'Vectorstore files' within root_folder_id
        query_vectorstore = (
            f"title='Vectorstore files' and '{root_folder_id}' in parents "
            f"and mimeType='application/vnd.google-apps.folder'"
        )
        vectorstore_folders = drive.ListFile({'q': query_vectorstore}).GetList()
        if not vectorstore_folders:
            raise FileNotFoundError(
                f"No folder named 'Vectorstore files' found in folder ID {root_folder_id}."
            )
        vectorstore_folder = vectorstore_folders[0]
        vectorstore_folder_id = vectorstore_folder['id']

        # 2. Find the subfolder named 'chroma_research_db' inside the 'Vectorstore files' folder
        query_chroma = (
            f"title='chroma_research_db' and '{vectorstore_folder_id}' in parents "
            f"and mimeType='application/vnd.google-apps.folder'"
        )
        chroma_folders = drive.ListFile({'q': query_chroma}).GetList()
        if not chroma_folders:
            raise FileNotFoundError(
                "No folder named 'chroma_research_db' found under 'Vectorstore files'."
            )
        chroma_folder = chroma_folders[0]
        chroma_folder_id = chroma_folder['id']

        # 3. Recursively download everything in 'chroma_research_db' to a local path
        os.makedirs(local_chroma_path, exist_ok=True)

        download_folder_from_drive(chroma_folder_id, local_chroma_path, drive)

        # Make sure chroma.sqlite3 is present
        if not os.path.exists(sqlite_db_path):
            raise FileNotFoundError(
                "chroma.sqlite3 was not found in the downloaded 'chroma_research_db' folder."
            )

    # 4. Load the database from the downloaded directory
    chroma_db = Chroma(
        embedding_function=embedding_function,
        collection_name=short_course_name,
        persist_directory=local_chroma_path
    )
    print("Successfully loaded the Chroma DB from Google Drive.")

    return chroma_db


#bge_large = get_embedding_model()
#course_name = "Basic Airflow Principles"

#chroma_db = load_vector_db_with_pydrive(
#    course_name = course_name,
#    root_folder_id = "1r0GJBKsu_X-2sbFkbItpymUfdVOBxEXt",
#    embedding_function = bge_large,
#    drive = drive
#)