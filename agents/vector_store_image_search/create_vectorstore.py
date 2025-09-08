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
import time
import shutil
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
    # Always use 'chroma_graphics_db' as the folder name in Google Drive for consistency
    folder_name = "chroma_graphics_db"

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
    temp_file_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
            temp_file_path = tmp_file.name
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
    finally:
        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception as cleanup_error:
                print(f"Warning: Could not clean up temp file {temp_file_path}: {cleanup_error}")


def download_image_from_drive(drive, file_id):
    
    """
    Download an image from Google Drive by file ID and return it as a PIL Image.
    :param drive: Google Drive instance.
    :param file_id: ID of the file to download.
    :return: PIL Image object or None if download fails.
    """
    temp_path = None
    try:
        file = drive.CreateFile({'id': file_id})
        file.FetchMetadata(fields='title, mimeType')

        # Download content to a temporary file
        temp_path = f"/tmp/{file_id}.jpg"
        file.GetContentFile(temp_path)

        # Open with PIL and convert to RGB
        with Image.open(temp_path) as img:
            pil_img = img.convert("RGB").copy()  # ensure it's fully loaded before file is closed

        return pil_img  # Return actual PIL Image

    except Exception as e:
        print(f"Failed to download image {file_id}: {e}")
        return None
    finally:
        # Always clean up temp file
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as cleanup_error:
                print(f"Warning: Could not clean up temp file {temp_path}: {cleanup_error}")



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

    # Create version-specific local path to avoid cache conflicts
    local_chroma_root = "/tmp/temp_chroma_folder"
    local_chroma_path = os.path.join(local_chroma_root, f"chroma_graphics_db_{root_folder_id}")
    os.makedirs(local_chroma_path, exist_ok=True)

    # Check if existing DB exists and download it for continuation
    existing_db_downloaded = False
    chroma_folder_list = drive.ListFile({
        'q': f"title='chroma_graphics_db' and '{vectorstore_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
    }).GetList()
    
    # Try to download existing DB with error handling for version compatibility
    if chroma_folder_list:
        print("Found existing Chroma DB on Drive. Attempting to download for continuation...")
        chroma_folder_id = chroma_folder_list[0]['id']
        try:
            download_folder_from_drive(chroma_folder_id, local_chroma_path, drive)
            existing_db_downloaded = True
            print("Existing DB downloaded successfully.\n")
        except Exception as e:
            print(f"⚠️  Failed to download existing DB due to version compatibility: {e}")
            print("   Starting fresh - existing DB will be overwritten.")
            # Clear the local directory to ensure clean start
            if os.path.exists(local_chroma_path):
                shutil.rmtree(local_chroma_path)
            os.makedirs(local_chroma_path, exist_ok=True)
            existing_db_downloaded = True

    embedding_function = get_embedding_model()
    
    # Try to initialize Chroma DB with error handling for version compatibility
    try:
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
    except Exception as e:
        print(f"⚠️  ChromaDB initialization failed due to version compatibility: {e}")
        print("   Starting fresh with clean database...")
        
        # Clear the local directory completely with retry logic
        if os.path.exists(local_chroma_path):
            # Try multiple times with delays to handle file locks
            for attempt in range(3):
                try:
                    shutil.rmtree(local_chroma_path)
                    break
                except PermissionError:
                    if attempt < 2:
                        print(f"   File locked, retrying in 1 second... (attempt {attempt + 1}/3)")
                        time.sleep(1)
                    else:
                        print("   Could not clear directory, creating new path...")
                        local_chroma_path = os.path.join(local_chroma_root, f"chroma_graphics_db_{root_folder_id}_{int(time.time())}")
                        break
        os.makedirs(local_chroma_path, exist_ok=True)
        
        # Initialize fresh Chroma DB instances with consistent collection names
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
        print("Fresh Chroma DB instances created with consistent collection names.\n")

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

        # ----------- QUICK SHEET-LEVEL CHECK -----------
        # Check if entire sheet is already processed (fast check)
        if 'fully_vectorized' in df.columns:
            fully_processed_count = (df['fully_vectorized'] == 'TRUE').sum()
            total_rows = len(df[df['Image Description'].str.strip().astype(bool)])
            
            if fully_processed_count == total_rows and total_rows > 0:
                print(f"✅ Sheet '{folder_id}' already fully processed ({fully_processed_count}/{total_rows} rows). Skipping.\n")
                continue

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
                try:
                    chroma_db.add_texts(texts=batch_texts, metadatas=batch_metadatas)
                except Exception as e:
                    print(f"⚠️  ChromaDB add_texts failed due to version compatibility: {e}")
                    print("   Preserving data and recreating Chroma DB...")
                    
                    # Get existing data before recreation to preserve it
                    existing_text_data = []
                    existing_image_data = []
                    
                    try:
                        # Get all existing text embeddings
                        existing_text_docs = chroma_db.get(include=["metadatas", "documents"])
                        existing_text_data = list(zip(existing_text_docs["documents"], existing_text_docs["metadatas"]))
                        
                        # Get all existing image embeddings
                        existing_image_docs = image_chroma_db.get(include=["metadatas", "embeddings"])
                        existing_image_data = list(zip(existing_image_docs["embeddings"], existing_image_docs["metadatas"]))
                        
                        print(f"   Preserving {len(existing_text_data)} text embeddings and {len(existing_image_data)} image embeddings...")
                    except:
                        print("   No existing data to preserve.")
                    
                    # Close existing Chroma instances to release file locks
                    try:
                        del chroma_db
                        del image_chroma_db
                    except:
                        pass
                    
                    # Clear the local directory completely
                    if os.path.exists(local_chroma_path):
                        # Try multiple times with delays to handle file locks
                        for attempt in range(3):
                            try:
                                shutil.rmtree(local_chroma_path)
                                break
                            except PermissionError:
                                if attempt < 2:
                                    print(f"   File locked, retrying in 1 second... (attempt {attempt + 1}/3)")
                                    time.sleep(1)
                                else:
                                    print("   Could not clear directory, creating new path...")
                                    local_chroma_path = os.path.join(local_chroma_root, f"chroma_graphics_db_{root_folder_id}_{int(time.time())}")
                                    break
                    os.makedirs(local_chroma_path, exist_ok=True)
                    
                    # Recreate Chroma DB instances with SAME collection names
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
                    
                    # Restore existing data
                    if existing_text_data:
                        print("   Restoring existing text embeddings...")
                        for doc, metadata in existing_text_data:
                            chroma_db.add_texts(texts=[doc], metadatas=[metadata])
                    
                    if existing_image_data:
                        print("   Restoring existing image embeddings...")
                        for embedding, metadata in existing_image_data:
                            image_chroma_db._collection.add(
                                embeddings=[embedding],
                                metadatas=[metadata],
                                documents=["image_embedding_only"],
                                ids=[f"img_{metadata['image_id']}"]
                            )
                    
                    print("   Data preserved and Chroma DB recreated. Retrying add_texts...")
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
                        try:
                            image_chroma_db._collection.add(
                                embeddings=[image_vector],
                                metadatas=[metadata],
                                documents=["image_embedding_only"],
                                ids=[f"img_{metadata['image_id']}"]
                            )
                            df.at[idx, 'image_vectorized'] = 'TRUE'
                            df.at[idx, 'image_embedding_ts'] = datetime.now().isoformat()
                        except Exception as e:
                            print(f"⚠️  ChromaDB image add failed due to version compatibility: {e}")
                            print("   Preserving data and recreating Chroma DB...")
                            
                            # Get existing data before recreation to preserve it
                            existing_text_data = []
                            existing_image_data = []
                            
                            try:
                                # Get all existing text embeddings
                                existing_text_docs = chroma_db.get(include=["metadatas", "documents"])
                                existing_text_data = list(zip(existing_text_docs["documents"], existing_text_docs["metadatas"]))
                                
                                # Get all existing image embeddings
                                existing_image_docs = image_chroma_db.get(include=["metadatas", "embeddings"])
                                existing_image_data = list(zip(existing_image_docs["embeddings"], existing_image_docs["metadatas"]))
                                
                                print(f"   Preserving {len(existing_text_data)} text embeddings and {len(existing_image_data)} image embeddings...")
                            except:
                                print("   No existing data to preserve.")
                            
                            # Close existing Chroma instances to release file locks
                            try:
                                del chroma_db
                                del image_chroma_db
                            except:
                                pass
                            
                            # Clear the local directory completely
                            if os.path.exists(local_chroma_path):
                                # Try multiple times with delays to handle file locks
                                for attempt in range(3):
                                    try:
                                        shutil.rmtree(local_chroma_path)
                                        break
                                    except PermissionError:
                                        if attempt < 2:
                                            print(f"   File locked, retrying in 1 second... (attempt {attempt + 1}/3)")
                                            time.sleep(1)
                                        else:
                                            print("   Could not clear directory, creating new path...")
                                            local_chroma_path = os.path.join(local_chroma_root, f"chroma_graphics_db_{root_folder_id}_{int(time.time())}")
                                            break
                            os.makedirs(local_chroma_path, exist_ok=True)
                            
                            # Recreate Chroma DB instances with SAME collection names
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
                            
                            # Restore existing data
                            if existing_text_data:
                                print("   Restoring existing text embeddings...")
                                for doc, metadata in existing_text_data:
                                    chroma_db.add_texts(texts=[doc], metadatas=[metadata])
                            
                            if existing_image_data:
                                print("   Restoring existing image embeddings...")
                                for embedding, metadata in existing_image_data:
                                    image_chroma_db._collection.add(
                                        embeddings=[embedding],
                                        metadatas=[metadata],
                                        documents=["image_embedding_only"],
                                        ids=[f"img_{metadata['image_id']}"]
                                    )
                            
                            print("   Data preserved and Chroma DB recreated. Retrying image add...")
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

        # Persist locally after each sheet for safety (no Drive upload)
        print(f"💾 Persisting ChromaDB locally after sheet '{folder_id}'...")
        chroma_db.persist()
        image_chroma_db.persist()
        print(f"✅ Local persistence complete for sheet '{folder_id}'")

    print("🎉 All embeddings processed successfully! Now uploading final ChromaDB to Drive...")
    
    # Final upload of complete ChromaDB to Drive
    try:
        print("📤 Uploading final ChromaDB to Drive...")
        
        # Final persist before upload
        chroma_db.persist()
        image_chroma_db.persist()
        
        # If this was an incremental update, remove old vectorstore first
        if existing_db_downloaded:
            print("Removing old vectorstore before uploading updated version...")
            chroma_folder_list = drive.ListFile({
                'q': f"title='chroma_graphics_db' and '{vectorstore_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
            }).GetList()
            if chroma_folder_list:
                drive.CreateFile({'id': chroma_folder_list[0]['id']}).Delete()
                print("Old vectorstore removed.")
        
        upload_folder_to_drive(local_chroma_path, vectorstore_folder_id, drive)
        print("✅ Final ChromaDB uploaded successfully to Drive!")
    except Exception as e:
        print(f"❌ Final upload failed: {e}")
        print("💾 Your complete ChromaDB is safely stored locally at:")
        print(f"   {local_chroma_path}")
        print("💡 You can manually upload it later or use a different service account.")
    
    # Clean up temporary files to free up space (only this version's directory)
    try:
        shutil.rmtree(local_chroma_path)
        print(f"Cleaned up temporary directory: {local_chroma_path}")
    except Exception as e:
        print(f"Warning: Could not clean up temporary directory: {e}")
    
    
def update_vectorstore(spreadsheet, drive, root_folder_id='1w5gJD_ALnqbRwl9XH0xTI0wr66IZmGL2'):
    """
    Update Chroma DB with new text and image embeddings from Google Sheets and upload to Drive.
    :param spreadsheet: Google Sheets instance.
    :param drive: Google Drive instance.
    :param root_folder_id: Root folder ID for version-specific operations.
    """
    print("Starting vectorstore update...")
    # Create version-specific local path to avoid cache conflicts
    local_chroma_root = "/tmp/temp_chroma_folder"
    local_chroma_path = os.path.join(local_chroma_root, f"chroma_graphics_db_{root_folder_id}")
    os.makedirs(local_chroma_root, exist_ok=True)

    print("Downloading existing Chroma DB from Drive...")
    file_list = drive.ListFile({
        'q': f"title='chroma_graphics_db' and '{root_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
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
        
        # Persist locally after each sheet for safety (no Drive upload)
        print(f"💾 Persisting ChromaDB locally after sheet '{folder_id}'...")
        chroma_db.persist()
        image_chroma_db.persist()
        print(f"✅ Local persistence complete for sheet '{folder_id}'")

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

    # ----------- FINAL PERSIST + UPLOAD -----------
    print("\nFinal persist and upload...")
    chroma_db.persist()
    image_chroma_db.persist()

    # Final upload of complete ChromaDB to Drive
    try:
        print("📤 Uploading final ChromaDB to Drive...")
        print("Removing old DB folder from Drive...")
        drive.CreateFile({'id': chroma_folder_id}).Delete()
        upload_folder_to_drive(local_chroma_path, root_folder_id, drive)
        print("✅ Final ChromaDB uploaded successfully to Drive!")
    except Exception as e:
        print(f"❌ Final upload failed: {e}")
        print("💾 Your complete ChromaDB is safely stored locally at:")
        print(f"   {local_chroma_path}")
        print("💡 You can manually upload it later or use a different service account.")

    # Clean up temporary files to free up space (only this version's directory)
    try:
        shutil.rmtree(local_chroma_path)
        print(f"Cleaned up temporary directory: {local_chroma_path}")
    except Exception as e:
        print(f"Warning: Could not clean up temporary directory: {e}")

    print("Update complete. Both collections are synced.")

