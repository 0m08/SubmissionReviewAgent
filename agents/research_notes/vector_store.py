from langchain_chroma import Chroma
import os
from services.helper_functions import get_short_name
from agents.research_notes.load_references import get_all_chunks_as_docs
from services.drive_service import upload_folder_to_drive, download_folder_from_drive
from langsmith import traceable


@traceable
# Vectorstore
def load_vector_db_with_pydrive(course_name: str,
                                root_folder_id: str,
                                sheet,
                                embedding_function,
                                drive) -> Chroma:
    """
    Load a Chroma vector database from a nested folder structure on Google Drive.

    Hierarchy:
        root_folder (root_folder_id)
            -> Vectorstore files
                -> chroma_research_db
                    -> chroma.sqlite3 (file)
                    -> <random_subfolder> (folder)
    """
    short_course_name = get_short_name(course_name)

    local_chroma_root = os.path.join("/tmp", "temp_chroma_folder")
    local_chroma_path = os.path.join(local_chroma_root, f"{short_course_name}_chroma_research_db")
    sqlite_db_path = os.path.join(local_chroma_path, "chroma.sqlite3")
    
    # Initialize all_doc_chunk_list to None
    all_doc_chunk_list = None

    # 1. Check local existence
    if os.path.exists(local_chroma_path) and os.path.exists(sqlite_db_path):
        print("Database already exists locally. Skipping download.")
    else:
        print("Database not found locally. Checking / creating Drive folders...")

        # Locate or create 'Vectorstore files'
        query_vectorstore = (
            f"title='Vectorstore files' and '{root_folder_id}' in parents "
            f"and mimeType='application/vnd.google-apps.folder'"
        )
        vectorstore_folders = drive.ListFile({'q': query_vectorstore}).GetList()

        if not vectorstore_folders:
            print(f"No folder named 'Vectorstore files' found. Creating it in ID {root_folder_id}...")
            file_metadata = {
                'title': 'Vectorstore files',
                'parents': [{'id': root_folder_id}],
                'mimeType': 'application/vnd.google-apps.folder'
            }
            vectorstore_folder = drive.CreateFile(file_metadata)
            vectorstore_folder.Upload()
            vectorstore_folder_id = vectorstore_folder['id']
        else:
            vectorstore_folder = vectorstore_folders[0]
            vectorstore_folder_id = vectorstore_folder['id']

        # Locate or create 'chroma_research_db'
        query_chroma = (
            f"title='chroma_research_db' and '{vectorstore_folder_id}' in parents "
            f"and mimeType='application/vnd.google-apps.folder'"
        )
        chroma_folders = drive.ListFile({'q': query_chroma}).GetList()

        # If no chroma_research_db exists, create one and also create a local Chroma DB
        if not chroma_folders:
            print("No folder named 'chroma_research_db' found. Creating it...")

            os.makedirs(local_chroma_path, exist_ok=True)

            # Load the chunks
            print("Loading the chunks...")
            all_doc_chunk_list = get_all_chunks_as_docs(sheet = sheet)

            # (Example) Create a brand-new Chroma DB locally
            print("Creating new local Chroma DB...")
            chroma_db = Chroma.from_documents(
                documents=all_doc_chunk_list,  # Your list of documents
                embedding=embedding_function,           # Your chosen embedding function
                collection_name=short_course_name,
                persist_directory=local_chroma_path
            )
            print("New local Chroma DB created at:", local_chroma_path)

            # Create empty folder in drive
            file_metadata = {
                'title': 'chroma_research_db',
                'parents': [{'id': vectorstore_folder_id}],
                'mimeType': 'application/vnd.google-apps.folder'
            }
            chroma_folder = drive.CreateFile(file_metadata)
            chroma_folder.Upload()
            chroma_folder_id = chroma_folder['id']

            # Upload the newly created folder/files to Drive
            print("Uploading new local Chroma DB to Google Drive...")
            upload_folder_to_drive(local_chroma_path, chroma_folder_id, drive)
            print("Upload complete.")
            return chroma_db, all_doc_chunk_list

        else:
            # If chroma_research_db folder exists, download it
            chroma_folder = chroma_folders[0]
            chroma_folder_id = chroma_folder['id']
            os.makedirs(local_chroma_path, exist_ok=True)
            download_folder_from_drive(chroma_folder_id, local_chroma_path, drive)
            print('Downloaded the chroma db from drive.')

            # Ensure chroma.sqlite3 is present
            if not os.path.exists(sqlite_db_path):
                print("chroma.sqlite3 was not found in the downloaded 'chroma_research_db' folder.")
                print("Recreating database...")
                # Delete corrupted folder and recreate
                drive.CreateFile({'id': chroma_folder_id}).Delete()
                # Clear local path and recreate
                import shutil
                if os.path.exists(local_chroma_path):
                    shutil.rmtree(local_chroma_path)
                # Recreate the database
                os.makedirs(local_chroma_path, exist_ok=True)
                # Load the chunks
                print("Loading the chunks...")
                all_doc_chunk_list = get_all_chunks_as_docs(sheet=sheet)
                # Create a brand-new Chroma DB locally
                print("Creating new local Chroma DB...")
                chroma_db = Chroma.from_documents(
                    documents=all_doc_chunk_list,
                    embedding=embedding_function,
                    collection_name=short_course_name,
                    persist_directory=local_chroma_path
                )
                print("New local Chroma DB created at:", local_chroma_path)
                # Create new folder in drive
                file_metadata = {
                    'title': 'chroma_research_db',
                    'parents': [{'id': vectorstore_folder_id}],
                    'mimeType': 'application/vnd.google-apps.folder'
                }
                chroma_folder = drive.CreateFile(file_metadata)
                chroma_folder.Upload()
                chroma_folder_id = chroma_folder['id']
                # Upload the newly created folder/files to Drive
                print("Uploading new local Chroma DB to Google Drive...")
                upload_folder_to_drive(local_chroma_path, chroma_folder_id, drive)
                print("Upload complete.")
                return chroma_db, all_doc_chunk_list

    # 2. Load the DB from the local path
    chroma_db = Chroma(
        embedding_function=embedding_function,
        collection_name=short_course_name,
        persist_directory=local_chroma_path
    )
    print("Successfully loaded the Chroma DB from local path.")

    # Load chunks for BM25 retriever if not already loaded
    if all_doc_chunk_list is None:
        print("Loading chunks for BM25 retriever...")
        all_doc_chunk_list = get_all_chunks_as_docs(sheet=sheet)

    return chroma_db, all_doc_chunk_list