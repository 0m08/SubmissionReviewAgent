import os
import json
import base64
import re
from io import BytesIO
from PIL import Image
import pandas as pd
import datetime
from services.drive_service import download_folder_from_drive
from langchain_chroma import Chroma
from services.embedding_service import get_embedding_model
from dotenv import load_dotenv
from services.drive_service import login_with_service_account
from pydrive2.drive import GoogleDrive
from services.sheets_service import save_to_sheet




load_dotenv()
# Decode and load Google Service Account credentials
key_bytes = base64.b64decode(os.environ["GDRIVE_SA_B64"])
sa_json = key_bytes.decode()
sa_dict = json.loads(sa_json)
gauth = login_with_service_account(json_str=sa_json)
gauth.ServiceAuth()
drive = GoogleDrive(gauth)



def is_valid_folderid(title):
    title = title.strip()
    if len(title) < 15:
        return False
    for c in title:
        if not (c.isalnum() or c in '-_'):
            return False
    return True


def upload_folder_to_drive(local_chroma_path: str, parent_folder_id: str, drive) -> None:
    folder_name = os.path.basename(local_chroma_path.rstrip('/\\'))

    # Create the top-level folder in Google Drive
    folder_metadata = {
        'title': folder_name,
        'parents': [{'id': parent_folder_id}],
        'mimeType': 'application/vnd.google-apps.folder'
    }
    new_folder = drive.CreateFile(folder_metadata)
    new_folder.Upload()
    new_folder_id = new_folder['id']
    print(f"Created folder: {folder_name}")

    # Upload contents of local_folder_path to the newly created folder
    for item in os.listdir(local_chroma_path):
        item_path = os.path.join(local_chroma_path, item)
        if os.path.isdir(item_path):
            # Recursive upload for subfolders
            upload_folder_to_drive(item_path, new_folder_id, drive)
        else:
            f = drive.CreateFile({'title': item, 'parents': [{'id': new_folder_id}]})
            f.SetContentFile(item_path)
            f.Upload()
            print(f"Uploaded: {item} to {folder_name}")



def build_vectorstore_and_upload(spreadsheet, drive):
    """
    Build a single vector store for all valid sheets in the spreadsheet and upload to Google Drive.

    All embeddings are stored in a single local chroma_graphics_db folder and uploaded to
    a single 'Vectorstore files' folder inside the specified parent folder in Google Drive.
    """
    valid_sheets = [ws.title for ws in spreadsheet.worksheets() if is_valid_folderid(ws.title)]
    print("📝 Valid sheets to process:", valid_sheets, "\n")

    parent_folder_id = '1QS6PmCESfgFWNNEpRJDUB0E-t8iMAatH'  # <-- your specified parent folder ID

    # ✅ Ensure 'Vectorstore files' folder exists inside the given parent folder
    file_list = drive.ListFile({
        'q': f"title='Vectorstore files' and '{parent_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
    }).GetList()

    print("DEBUG: type of file_list =", type(file_list))  # Should be <class 'list'>
    print("DEBUG: first element type =", type(file_list[0]) if file_list else "Empty list")

    if file_list:
        vectorstore_folder_id = file_list[0]['id']
        print("✅ Found 'Vectorstore files' folder inside the specified parent folder.\n")
    else:
        print("📂 Creating 'Vectorstore files' folder inside the specified parent folder...")
        folder_metadata = {
            'title': 'Vectorstore files',
            'parents': [{'id': parent_folder_id}],
            'mimeType': 'application/vnd.google-apps.folder'
        }
        new_folder = drive.CreateFile(folder_metadata)
        new_folder.Upload()
        vectorstore_folder_id = new_folder['id']
        print("✅ Created 'Vectorstore files' folder inside the specified parent folder.\n")

    # Prepare local Chroma DB path (ONE database for all sheets!)
    local_chroma_root = "/tmp/temp_chroma_folder"
    local_chroma_path = os.path.join(local_chroma_root, "chroma_graphics_db")
    os.makedirs(local_chroma_path, exist_ok=True)

    # Initialize embedding model and Chroma DB once
    embedding_function = get_embedding_model()
    chroma_db = Chroma(
        embedding_function=embedding_function,
        collection_name="text_embeddings",
        persist_directory=local_chroma_path
    )
    print("🧠 Embedding model and Chroma DB initialized (single DB for all sheets).\n")

    # Loop through valid sheets and add their embeddings to the single Chroma DB
    for sheet in spreadsheet.worksheets():
        if not is_valid_folderid(sheet.title):
            continue

        folder_id = sheet.title
        print(f"🔍 Processing sheet '{sheet.title}' (folder_id: {folder_id})...")

        records = sheet.get_all_records()
        df = pd.DataFrame(records)

        # ✅ Skip if 'Image Description' column not present
        if 'Image Description' not in df.columns:
            print(f"⚠️ Sheet '{sheet.title}' does not have 'Image Description' column. Skipping.\n")
            continue

        if 'vectorized' in df.columns and 'embedding_ts' in df.columns:
            mask_to_vectorize = (df['vectorized'] != 'TRUE') & df['Image Description'].str.strip().astype(bool)
        else:
            df['vectorized'] = ''
            df['embedding_ts'] = ''
            mask_to_vectorize = df['Image Description'].str.strip().astype(bool)

        rows_to_vectorize = df.loc[mask_to_vectorize]
        print(f"📦 Rows to vectorize for this folder: {len(rows_to_vectorize)}")

        if rows_to_vectorize.empty:
            print(f"⚠️ No new rows to process for folder {folder_id}. Skipping.\n")
            continue

        # Prepare new data to add
        new_texts = rows_to_vectorize['Image Description'].tolist()
        new_metadatas = rows_to_vectorize.apply(lambda row: {
            'image_id': row['Image ID'],
            'name': row['Image Name'],
            'drive_url': row['Image Link'],
            'mime_type': row['MimeType'],
            'description': row['Image Description'],
            'image_type': row['Image Type'],
            'image_title': row['Image Title'],
            'folder_id': folder_id
        }, axis=1).tolist()

        # 🚀 Add in smaller batches
        BATCH_SIZE = 5000
        for i in range(0, len(new_texts), BATCH_SIZE):
            batch_texts = new_texts[i:i + BATCH_SIZE]   
            batch_metadatas = new_metadatas[i:i + BATCH_SIZE]
            print(f"🔢 Adding batch {i//BATCH_SIZE + 1} ({len(batch_texts)} items)...")
            chroma_db.add_texts(texts=batch_texts, metadatas=batch_metadatas)

        # Mark vectorized rows in the sheet
        now = datetime.datetime.now().isoformat()
        df.loc[mask_to_vectorize, 'vectorized'] = 'TRUE'
        df.loc[mask_to_vectorize, 'embedding_ts'] = now

        # Update sheet with updated status
        print("🔄 Updating sheet with vectorization status...")
        save_to_sheet(sheet, df)
        print(f"✅ Sheet '{folder_id}' updated.\n")

    # After processing all sheets, persist the single Chroma DB
    chroma_db.persist()
    print("✅ Chroma DB persisted locally at:", local_chroma_path)

    # Upload the single local chroma_graphics_db to Drive
    print("☁️ Uploading final 'chroma_graphics_db' to 'Vectorstore files' in Drive...")
    upload_folder_to_drive(local_chroma_path, vectorstore_folder_id, drive)
    print("✅ Uploaded final Chroma DB to: Vectorstore files/chroma_graphics_db/\n")

    print("🎉 All embeddings stored in a single Chroma DB, uploaded, and sheets updated!")


def update_vectorstore(spreadsheet, drive):
    """
    Update an existing Chroma DB by processing:
    - New sheets added to the spreadsheet (new folder IDs)
    - New rows in existing sheets where vectorized != TRUE

    Only appends new data to an already existing local Chroma DB (chroma_graphics_db),
    and re-uploads it to Drive.
    """

    parent_folder_id = '1QS6PmCESfgFWNNEpRJDUB0E-t8iMAatH'
    local_chroma_root = "/tmp/temp_chroma_folder"
    local_chroma_path = os.path.join(local_chroma_root, "chroma_graphics_db")
    os.makedirs(local_chroma_root, exist_ok=True)

    # Find the existing Chroma DB on Drive and download it
    print("Downloading existing 'chroma_graphics_db' from Drive...")
    file_list = drive.ListFile({
        'q': f"title='chroma_graphics_db' and '{parent_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
    }).GetList()

    if not file_list:
        raise FileNotFoundError("Existing 'chroma_graphics_db' not found in Google Drive.")

    chroma_folder_id = file_list[0]['id']
    download_folder_from_drive(chroma_folder_id, local_chroma_path, drive)
    print(f"Downloaded to local path: {local_chroma_path}")

    # Initialize embedding model and load existing Chroma DB
    embedding_function = get_embedding_model()
    chroma_db = Chroma(
        embedding_function=embedding_function,
        collection_name="text_embeddings",
        persist_directory=local_chroma_path
    )
    print("Chroma DB loaded locally and ready to update.\n")

    valid_sheets = [ws.title for ws in spreadsheet.worksheets() if is_valid_folderid(ws.title)]
    print("Valid sheets to scan for updates:", valid_sheets, "\n")

    for sheet in spreadsheet.worksheets():
        if not is_valid_folderid(sheet.title):
            continue

        folder_id = sheet.title
        print(f"🔍 Scanning sheet '{sheet.title}' (folder_id: {folder_id})...")

        records = sheet.get_all_records()
        df = pd.DataFrame(records)

        if 'Image Description' not in df.columns:
            print(f"⚠️ Sheet '{sheet.title}' does not have 'Image Description'. Skipping.\n")
            continue

        if 'vectorized' in df.columns and 'embedding_ts' in df.columns:
            mask_to_vectorize = (df['vectorized'] != 'TRUE') & df['Image Description'].str.strip().astype(bool)
        else:
            df['vectorized'] = ''
            df['embedding_ts'] = ''
            mask_to_vectorize = df['Image Description'].str.strip().astype(bool)

        rows_to_vectorize = df.loc[mask_to_vectorize]
        print(f"New rows to vectorize: {len(rows_to_vectorize)}")

        if rows_to_vectorize.empty:
            print(f"No new rows to update for folder {folder_id}.\n")
            continue

        new_texts = rows_to_vectorize['Image Description'].tolist()
        new_metadatas = rows_to_vectorize.apply(lambda row: {
            'image_id': row['Image ID'],
            'name': row['Image Name'],
            'drive_url': row['Image Link'],
            'mime_type': row['MimeType'],
            'description': row['Image Description'],
            'image_type': row['Image Type'],
            'image_title': row['Image Title'],
            'folder_id': folder_id
        }, axis=1).tolist()

        BATCH_SIZE = 5000
        for i in range(0, len(new_texts), BATCH_SIZE):
            batch_texts = new_texts[i:i + BATCH_SIZE]
            batch_metadatas = new_metadatas[i:i + BATCH_SIZE]
            print(f"➕ Adding batch {i//BATCH_SIZE + 1} ({len(batch_texts)} items)...")
            chroma_db.add_texts(texts=batch_texts, metadatas=batch_metadatas)

        # Mark vectorized rows in the sheet
        now = datetime.datetime.now().isoformat()
        df.loc[mask_to_vectorize, 'vectorized'] = 'TRUE'
        df.loc[mask_to_vectorize, 'embedding_ts'] = now

        save_to_sheet(sheet, df)
        print(f"✅ Sheet '{folder_id}' updated with vectorization status.\n")

    # Persist and upload the updated DB
    chroma_db.persist()
    print("💾 Chroma DB updated and persisted locally.")

    # Upload back to Drive
    print("☁️ Uploading updated 'chroma_graphics_db' to Google Drive...")
    upload_folder_to_drive(local_chroma_path, parent_folder_id, drive)
    print("✅ Update complete. Vectorstore is now synced with spreadsheet.\n")


def sanitize_filename(filename):
    # Remove or replace problematic characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    return filename.strip()


def chroma_db_exists(drive, parent_folder_id):
    """
    Checks whether chroma_graphics_db exists inside Vectorstore files in the specified parent folder.
    Returns True if both folders exist.
    """
    vectorstore_folder_list = drive.ListFile({
        'q': f"title='Vectorstore files' and '{parent_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
    }).GetList()

    if not vectorstore_folder_list:
        return False  # No 'Vectorstore files' folder

    vectorstore_folder_id = vectorstore_folder_list[0]['id']

    chroma_folder_list = drive.ListFile({
        'q': f"title='chroma_graphics_db' and '{vectorstore_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
    }).GetList()

    return bool(chroma_folder_list)


def download_folder_from_drive(folder_id, local_path, drive):
    os.makedirs(local_path, exist_ok=True)
    file_list = drive.ListFile({
        'q': f"'{folder_id}' in parents and trashed=false"
    }).GetList()

    for item in file_list:
        item_id = item["id"]
        item_title = sanitize_filename(item["title"])  # SANITIZE!
        item_mimeType = item["mimeType"]

        local_item_path = os.path.join(local_path, item_title)

        if item_mimeType == "application/vnd.google-apps.folder":
            print(f"Creating folder: {local_item_path}")
            os.makedirs(local_item_path, exist_ok=True)
            download_folder_from_drive(item_id, local_item_path, drive)
        else:
            print(f"Downloading file: {local_item_path}")
            item.GetContentFile(local_item_path)


def download_image_from_drive(file_id, drive):


    file = drive.CreateFile({'id': file_id})
    file.FetchMetadata(fields='title, mimeType')

    # Download to a temporary file
    temp_file = 'temp_image'
    file.GetContentFile(temp_file)
    with open(temp_file, 'rb') as f:
        img = Image.open(BytesIO(f.read()))
    return img


def load_central_chroma_db(drive, central_folder_id):
    """
    Load the single central Chroma DB stored in Google Drive (in 'chroma_graphics_db' inside the central_folder_id).

    Parameters:
    - embedding_function: Function to generate embeddings.
    - drive: Authenticated Google Drive instance.
    - central_folder_id (str): ID of the parent folder containing the single Chroma DB.

    Returns:
    - chroma_db: The loaded Chroma DB.
    """
    
    embedding_function = get_embedding_model()

    # Check embedding model
    embedding_fn = get_embedding_model()
    print("🔧 Testing embedding...")
    _ = embedding_fn.embed_query("wire shieling")
    print("✅ Embedding works.")

        # Check for the 'vectorstore files' folder inside the central folder
    vectorstore_files_list = drive.ListFile({
        'q': f"title='vectorstore files' and '{central_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
    }).GetList()

    if not vectorstore_files_list:
        raise FileNotFoundError("❌ 'vectorstore files' folder not found in the central folder.")

    vectorstore_folder_id = vectorstore_files_list[0]['id']
    print(f"✅ Found 'vectorstore files' folder in central folder (id: {vectorstore_folder_id}).")

    # Check for the 'chroma_graphics_db' folder inside the vectorstore files folder
    file_list = drive.ListFile({
        'q': f"title='chroma_graphics_db' and '{vectorstore_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
    }).GetList()

    if not file_list:
        raise FileNotFoundError("❌ 'chroma_graphics_db' folder not found in the vectorstore files folder.")

    chroma_folder_id = file_list[0]['id']
    print(f"✅ Found 'chroma_graphics_db' folder in vectorstore files folder (id: {chroma_folder_id}).")

    # Download the Chroma DB folder to a local directory
    local_chroma_root = "/tmp/central_chroma_folder"
    local_chroma_path = os.path.join(local_chroma_root, "chroma_graphics_db")
    os.makedirs(local_chroma_root, exist_ok=True)

    # Skip download if the database has already been downloaded
    if os.path.exists(local_chroma_path) and os.listdir(local_chroma_path):
        print(f"✅ Chroma DB already exists locally at: {local_chroma_path}, skipping download.")
    else:
        print("☁️ Downloading 'chroma_graphics_db' from Drive...")
        download_folder_from_drive(chroma_folder_id, local_chroma_path, drive)
        print(f"✅ Downloaded 'chroma_graphics_db' to: {local_chroma_path}")

    try:

        # Initialize Chroma DB from the local folder
        chroma_db = Chroma(
            embedding_function=embedding_function,
            collection_name="text_embeddings",
            persist_directory=local_chroma_path
        )
    except Exception as e:
        raise Exception(f"❌ Error loading Chroma DB: {e}")

    print("✅ Chroma DB loaded from the central folder.")
    return chroma_db

def search_similar_images_across_all(query_text, k=5, filters=None):
    """
    Search for similar images across the central Chroma DB with optional metadata filters (in-memory).
    """
    print("🔍 Loading central Chroma DB...")

    central_folder_id = '1QS6PmCESfgFWNNEpRJDUB0E-t8iMAatH'  # Replace with your actual central folder ID
    chroma_db = load_central_chroma_db(drive, central_folder_id)
    print("Document count:", chroma_db._collection.count())

    # Initial similarity search (broad pool)
    print("🔎 Performing similarity search...")
    results_docs = chroma_db.similarity_search_with_score(query_text, k=5)
    print(f"✅ Retrieved {len(results_docs)} results.")

    # In-memory filtering
    filtered_results = []
    for doc, score in results_docs:
        metadata = doc.metadata
        if filters:
            if filters.get("mime_type") and metadata.get("mime_type") not in filters["mime_type"]:
                continue
            if filters.get("image_title") and filters["image_title"].lower() not in metadata.get("image_title", "").lower():
                continue
            if filters.get("image_type") and filters["image_type"].lower() not in metadata.get("image_type", "").lower():
                continue

        filtered_results.append({
            "similarity": score,
            "image_id": metadata['image_id'],
            "name": metadata['name'],
            "drive_url": metadata['drive_url'],
            "description": metadata['description'],
            "folder_id": metadata['folder_id'],
        })

    filtered_results.sort(key=lambda x: x['similarity'])
    return filtered_results[:k]



    

