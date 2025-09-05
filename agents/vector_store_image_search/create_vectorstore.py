import os
from PIL import Image
import pandas as pd
from datetime import datetime
import requests
import imagehash
from io import BytesIO
from langchain.vectorstores import Chroma
from services.embedding_service import get_embedding_model
from services.sheets_service import save_to_sheet
from services.drive_service import download_folder_from_drive
import re
from concurrent.futures import ThreadPoolExecutor
import tempfile
from typing import List
import cohere
import base64
from dotenv import load_dotenv

load_dotenv()

def is_valid_folderid(title):
    """
    Check if the title is a valid folder ID:
    :param title: String to validate.
    :return: True if valid, False otherwise.
    """
    title = title.strip()
    if len(title) < 15:
        return False
    for c in title:
        if not (c.isalnum() or c in '-_'):
            return False
    return True


def chroma_db_exists(drive, parent_folder_id):
    """
    Checks whether a Chroma DB exists inside ``Vectorstore files`` in the specified
    parent folder.

    This function mirrors the folder structure used by the vectorstore builder:
    ``Vectorstore files`` → ``chroma_graphics_db``.
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


def upload_folder_to_drive(local_folder_path, parent_drive_folder_id, drive):
    folder_name = os.path.basename(local_folder_path)

    # Step 1: Create a folder on Drive
    folder_metadata = {
        'title': folder_name,
        'parents': [{'id': parent_drive_folder_id}],
        'mimeType': 'application/vnd.google-apps.folder'
    }
    drive_folder = drive.CreateFile(folder_metadata)
    drive_folder.Upload()
    drive_folder_id = drive_folder['id']

    # Step 2: Recursively upload contents
    for root, dirs, files in os.walk(local_folder_path):
        rel_path = os.path.relpath(root, local_folder_path)
        current_folder_id = drive_folder_id

        # Maintain folder structure
        if rel_path != '.':
            path_parts = rel_path.split(os.sep)
            for part in path_parts:
                folder_list = drive.ListFile({
                    'q': f"title='{part}' and '{current_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
                }).GetList()
                if folder_list:
                    current_folder_id = folder_list[0]['id']
                else:
                    subfolder_metadata = {
                        'title': part,
                        'parents': [{'id': current_folder_id}],
                        'mimeType': 'application/vnd.google-apps.folder'
                    }
                    subfolder = drive.CreateFile(subfolder_metadata)
                    subfolder.Upload()
                    current_folder_id = subfolder['id']

        # Upload files in current directory
        for file_name in files:
            file_path = os.path.join(root, file_name)
            f = drive.CreateFile({'title': file_name, 'parents': [{'id': current_folder_id}]})
            f.SetContentFile(file_path)
            f.Upload()

def compute_phash_from_drive_url(url):
    """
    Compute perceptual hash (pHash) for an image from a URL.
    :param url: URL of the image.
    :return: pHash as a string or None if an error occurs.
    """
    try:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content)).convert("RGB")
        return str(imagehash.phash(img))
    except Exception as e:
        print(f"⚠️ Could not compute pHash for {url}: {e}")
        return None


def extract_drive_file_id(url):
    patterns = [
        r"https://drive\.google\.com/file/d/([a-zA-Z0-9_-]+)",
        r"https://drive\.google\.com/open\?id=([a-zA-Z0-9_-]+)",
        r"https://drive\.google\.com/uc\?id=([a-zA-Z0-9_-]+)",
        r"https://drive\.google\.com/uc\?export=download&id=([a-zA-Z0-9_-]+)"
    ]
    for pattern in patterns:
        match = re.match(pattern, url)
        if match:
            return match.group(1)
    return None


co = cohere.ClientV2(api_key=os.getenv('COHERE_API_KEY'))
def get_image_embedding_from_file(image: Image.Image) -> List[float]:
    """
    Save a PIL image to a temporary file and extract the embedding using Cohere's embed-v4.0.
    """
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
        image.save(tmp_file.name, format="JPEG")
        with open(tmp_file.name, "rb") as f:
            enc_img = base64.b64encode(f.read()).decode("utf-8")
            enc_img = f"data:image/jpeg;base64,{enc_img}"

    response = co.embed(
        model="embed-v4.0",
        images=[enc_img],
        input_type="image",
        embedding_types=["float"],
    )
    return response.embeddings.float[0]


def download_image_from_drive(drive, file_id):
    
    """
    Download an image from Google Drive by file ID and return it as a PIL Image.
    :param drive: Google Drive instance.
    :param file_id: ID of the file to download.
    :return: PIL Image object or None if download fails.
    """
    try:
        file = drive.CreateFile({'id': file_id})
        file.FetchMetadata(fields='title, mimeType')

        # Download content to a temporary file
        temp_path = f"/tmp/{file_id}.jpg"
        file.GetContentFile(temp_path)

        # Open with PIL and convert to RGB
        with Image.open(temp_path) as img:
            pil_img = img.convert("RGB").copy()  # ensure it's fully loaded before file is closed

        # Optionally delete temp file
        os.remove(temp_path)

        return pil_img  # Return actual PIL Image

    except Exception as e:
        print(f"Failed to download image {file_id}: {e}")
        return None



def process_image_embedding(row, folder_id, drive):
    try:
        local_img = download_image_from_drive(drive, row['Image ID'])
        if not isinstance(local_img, Image.Image):
            return None, None, None
        image_vector = get_image_embedding_from_file(local_img)
        metadata = {
            'image_id': row['Image ID'],
            'name': row['Image Name'],
            'drive_url': row['Image Link'],
            'mime_type': row['MimeType'],
            'description': row['Image Description'],
            'image_type': row['Image Type'],
            'image_title': row['Image Title'],
            'folder_id': folder_id,
            'embedding_type': 'image'
        }

        # Optional v2 metadata
        if 'Course Name' in row:
            metadata['course_name'] = row['Course Name']
        if 'Topic Name' in row:
            metadata['topic_name'] = row['Topic Name']
        if 'Stock/Non Stock' in row:
            metadata['stock_type'] = row['Stock/Non Stock']

        return row.name, image_vector, metadata
    except Exception as e:
        print(f"Error embedding image ID {row['Image ID']}: {e}")
        return None, None, None


def build_vectorstore_and_upload(spreadsheet, drive, root_folder_id='1QS6PmCESfgFWNNEpRJDUB0E-t8iMAatH', version='v1'):
    valid_sheets = [ws.title for ws in spreadsheet.worksheets() if is_valid_folderid(ws.title)]
    print("Valid sheets to process:", valid_sheets, "\n")

    file_list = drive.ListFile({
        'q': f"title='Vectorstore files' and '{root_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
    }).GetList()

    if file_list:
        vectorstore_folder_id = file_list[0]['id']
        print("Found 'Vectorstore files' folder.\n")
    else:
        folder_metadata = {
            'title': 'Vectorstore files',
            'parents': [{'id': root_folder_id}],
            'mimeType': 'application/vnd.google-apps.folder'
        }
        new_folder = drive.CreateFile(folder_metadata)
        new_folder.Upload()
        vectorstore_folder_id = new_folder['id']
        print("Created 'Vectorstore files' folder.\n")

    local_chroma_root = "/tmp/temp_chroma_folder"
    local_chroma_path = os.path.join(local_chroma_root, "chroma_graphics_db")
    os.makedirs(local_chroma_path, exist_ok=True)

    embedding_function = get_embedding_model()
    chroma_db = Chroma(
        embedding_function=embedding_function,
        collection_name="text_embeddings",
        persist_directory=local_chroma_path
    )
    image_chroma_db = Chroma(
        embedding_function=None,
        collection_name="image_embeddings",
        persist_directory=local_chroma_path
    )

    print("Embedding model and Chroma DB initialized.\n")

    required_cols_v2 = {'Image Description', 'Course Name', 'Topic Name', 'Stock/Non Stock'}

    for sheet in spreadsheet.worksheets():
        if not is_valid_folderid(sheet.title):
            continue

        folder_id = sheet.title
        print(f"🔍 Processing sheet '{folder_id}'...")

        df = pd.DataFrame(sheet.get_all_records())

        required_cols = {'Image Description'}
        if version == 'v2':
            required_cols = required_cols_v2

        if any(col not in df.columns for col in required_cols):
            print(f"Sheet '{folder_id}' missing required columns. Skipping.\n")
            continue

        for col in ['vectorized', 'embedding_ts', 'image_vectorized', 'image_embedding_ts', 'fully_vectorized']:
            if col not in df.columns:
                df[col] = ''

        # ----------- TEXT EMBEDDING SECTION -----------
        mask_to_vectorize = (df['vectorized'] != 'TRUE') & df['Image Description'].str.strip().astype(bool)
        rows_to_vectorize = df.loc[mask_to_vectorize]
        print(f"Rows to vectorize (text): {len(rows_to_vectorize)}")

        if not rows_to_vectorize.empty:
            new_texts = rows_to_vectorize['Image Description'].tolist()
            def build_metadata(row):
                meta = {
                    'image_id': row['Image ID'],
                    'name': row['Image Name'],
                    'drive_url': row['Image Link'],
                    'mime_type': row['MimeType'],
                    'description': row['Image Description'],
                    'image_type': row['Image Type'],
                    'image_title': row['Image Title'],
                    'folder_id': folder_id,
                    'embedding_type': 'text'
                }
                if 'Course Name' in row:
                    meta['course_name'] = row['Course Name']
                if 'Topic Name' in row:
                    meta['topic_name'] = row['Topic Name']
                if 'Stock/Non Stock' in row:
                    meta['stock_type'] = row['Stock/Non Stock']
                return meta

            new_metadatas = rows_to_vectorize.apply(build_metadata, axis=1).tolist()

            BATCH_SIZE = 5000
            for i in range(0, len(new_texts), BATCH_SIZE):
                batch_texts = new_texts[i:i + BATCH_SIZE]
                batch_metadatas = new_metadatas[i:i + BATCH_SIZE]
                print(f"Adding text batch {i//BATCH_SIZE + 1} ({len(batch_texts)} items)...")
                chroma_db.add_texts(texts=batch_texts, metadatas=batch_metadatas)

            # Update text vectorized flags
            now = datetime.now().isoformat()
            df.loc[mask_to_vectorize, 'vectorized'] = 'TRUE'
            df.loc[mask_to_vectorize, 'embedding_ts'] = now

        # ---- IMAGE EMBEDDING ----
        image_mask = df['image_vectorized'] != 'TRUE'
        image_rows = df.loc[image_mask]
        print(f"Rows to vectorize (image): {len(image_rows)}")

        if not image_rows.empty:
            print("Image rows to embed:")
            print(image_rows[['Image ID']].reset_index())

            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {
                    executor.submit(process_image_embedding, row, folder_id, drive): idx
                    for idx, row in image_rows.iterrows()
                }
                for future in futures:
                    idx, image_vector, metadata = future.result()
                    if image_vector is not None:
                        print(f"Image ID processed: {metadata['image_id']} (row {idx})")
                        image_chroma_db._collection.add(
                            embeddings=[image_vector],
                            metadatas=[metadata],
                            documents=["image_embedding_only"],
                            ids=[f"img_{metadata['image_id']}"]
                        )
                        df.at[idx, 'image_vectorized'] = 'TRUE'
                        df.at[idx, 'image_embedding_ts'] = datetime.now().isoformat()
                    else:
                        print(f"Skipping image at row {idx} due to processing failure.")


        # ---- COMBINED STATUS ----
        df['fully_vectorized'] = df.apply(
            lambda row: 'TRUE' if row['vectorized'] == 'TRUE' and row['image_vectorized'] == 'TRUE' else '',
            axis=1
        )

        save_to_sheet(sheet, df)
        print(f"Sheet '{folder_id}' updated.\n")

    chroma_db.persist()
    print("Text collection count before persist:", chroma_db._collection.count())
    print("Image collection count before persist:", image_chroma_db._collection.count())
    image_chroma_db.persist()
    print("Chroma DB persisted locally.")

    print("Uploading to Drive...")
    upload_folder_to_drive(local_chroma_path, vectorstore_folder_id, drive)
    print("Upload complete.\n All embeddings processed successfully.")
    
    
def update_vectorstore(spreadsheet, drive):
    """
    Update Chroma DB with new text and image embeddings from Google Sheets and upload to Drive.
    :param spreadsheet: Google Sheets instance.
    :param drive: Google Drive instance.
    """
    print("Starting vectorstore update...")
    parent_folder_id = '1w5gJD_ALnqbRwl9XH0xTI0wr66IZmGL2'
    local_chroma_root = "/tmp/temp_chroma_folder"
    local_chroma_path = os.path.join(local_chroma_root, "chroma_graphics_db")
    os.makedirs(local_chroma_root, exist_ok=True)

    print("Downloading existing Chroma DB from Drive...")
    file_list = drive.ListFile({
        'q': f"title='chroma_graphics_db' and '{parent_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
    }).GetList()

    if not file_list:
        raise FileNotFoundError("'chroma_graphics_db' folder not found in Drive.")
    
    chroma_folder_id = file_list[0]['id']
    download_folder_from_drive(chroma_folder_id, local_chroma_path, drive)

    print("DB downloaded. Initializing Chroma...")
    embedding_function = get_embedding_model()
    chroma_db = Chroma(
        embedding_function=embedding_function,
        collection_name="text_embeddings",
        persist_directory=local_chroma_path
    )
    image_chroma_db = Chroma(
        embedding_function=None,
        collection_name="image_embeddings",
        persist_directory=local_chroma_path
    )

    for sheet in spreadsheet.worksheets():
        if not is_valid_folderid(sheet.title):
            continue

        folder_id = sheet.title
        print(f"\nScanning sheet '{folder_id}'...")

        df = pd.DataFrame(sheet.get_all_records())
        if 'Image Description' not in df.columns:
            print(f"Sheet '{folder_id}' missing 'Image Description'. Skipping.")
            continue

        # Ensure necessary columns
        for col in ['vectorized', 'embedding_ts', 'image_vectorized', 'image_embedding_ts', 'fully_vectorized']:
            if col not in df.columns:
                df[col] = ''

        # ----------- TEXT EMBEDDING -----------
        mask_to_vectorize = (df['vectorized'] != 'TRUE') & df['Image Description'].str.strip().astype(bool)
        rows_to_vectorize = df.loc[mask_to_vectorize]
        print(f"New text rows: {len(rows_to_vectorize)}")

        if not rows_to_vectorize.empty:
            new_texts = rows_to_vectorize['Image Description'].tolist()
            new_metadatas = rows_to_vectorize.apply(lambda row: {
                'image_id': row['Image ID'],
                'name': row['Image Name'],
                'drive_url': row['Image Link'],
                'mime_type': row['MimeType'],
                'description': row['Image Description'],
                'image_type': row['Image Type'],
                'image_title': row['Image Title'],
                'folder_id': folder_id,
                'embedding_type': 'text'
            }, axis=1).tolist()

            BATCH_SIZE = 5000
            for i in range(0, len(new_texts), BATCH_SIZE):
                chroma_db.add_texts(
                    texts=new_texts[i:i + BATCH_SIZE],
                    metadatas=new_metadatas[i:i + BATCH_SIZE]
                )
            now = datetime.now().isoformat()
            df.loc[mask_to_vectorize, 'vectorized'] = 'TRUE'
            df.loc[mask_to_vectorize, 'embedding_ts'] = now

        # ----------- IMAGE EMBEDDING -----------
        image_mask = df['image_vectorized'] != 'TRUE'
        image_rows = df.loc[image_mask]
        print(f"New image rows: {len(image_rows)}")

        if not image_rows.empty:
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {
                    executor.submit(process_image_embedding, row, folder_id, drive): idx
                    for idx, row in image_rows.iterrows()
                }
                for future in futures:
                    idx, image_vector, metadata = future.result()
                    if image_vector is not None:
                        image_chroma_db._collection.add(
                            embeddings=[image_vector],
                            metadatas=[metadata],
                            documents=["image_embedding_only"],
                            ids=[f"img_{metadata['image_id']}"]
                        )
                        df.at[idx, 'image_vectorized'] = 'TRUE'
                        df.at[idx, 'image_embedding_ts'] = datetime.now().isoformat()
                    else:
                        print(f"Skipping image at row {idx} due to processing failure.")

        # ----------- COMBINED STATUS -----------
        df['fully_vectorized'] = df.apply(
            lambda row: 'TRUE' if row['vectorized'] == 'TRUE' and row['image_vectorized'] == 'TRUE' else '',
            axis=1
        )

        save_to_sheet(sheet, df)
        print(f"Sheet '{folder_id}' updated.")

    # ----------- DEDUPLICATION: ID + URL -----------
    print("Removing duplicates by image_id or drive_url...")
    all_docs = chroma_db.get(include=["metadatas"])
    ids = all_docs["ids"]
    metadatas = all_docs["metadatas"]
    seen_keys = set()
    id_url_duplicates = []

    for doc_id, metadata in zip(ids, metadatas):
        key = (metadata.get("image_id"), metadata.get("drive_url"))
        if key in seen_keys:
            id_url_duplicates.append(doc_id)
        else:
            seen_keys.add(key)

    if id_url_duplicates:
        chroma_db.delete(ids=id_url_duplicates)
        print(f"Removed {len(id_url_duplicates)} duplicates by image_id/URL.")

    # ----------- DEDUPLICATION: pHash -----------
    print("Removing visual duplicates using pHash...")
    all_docs = chroma_db.get(include=["metadatas"])
    ids = all_docs["ids"]
    metadatas = all_docs["metadatas"]
    seen_phashes = set()
    phash_duplicates = []

    for doc_id, metadata in zip(ids, metadatas):
        phash = metadata.get("phash")
        if phash:
            if phash in seen_phashes:
                phash_duplicates.append(doc_id)
            else:
                seen_phashes.add(phash)

    if phash_duplicates:
        chroma_db.delete(ids=phash_duplicates)
        print(f"Removed {len(phash_duplicates)} visually duplicate entries.")

    # ----------- PERSIST + UPLOAD -----------
    print("\nPersisting local DBs...")
    chroma_db.persist()
    image_chroma_db.persist()

    print("Removing old DB folder from Drive...")
    drive.CreateFile({'id': chroma_folder_id}).Delete()

    print("Uploading updated vectorstore to Drive...")
    upload_folder_to_drive(local_chroma_path, parent_folder_id, drive)

    print("Update complete. Both collections are synced.")

