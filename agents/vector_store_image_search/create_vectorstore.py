import os
from PIL import Image
import pandas as pd
import datetime
import requests
import imagehash
from io import BytesIO
import torch
import clip
from langchain_chroma import Chroma
from services.embedding_service import get_embedding_model
from services.sheets_service import save_to_sheet
from services.drive_service import upload_folder_to_drive, download_folder_from_drive




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
    Checks whether chroma_research_db exists inside Vectorstore files in the specified parent folder.
    Returns True if both folders exist.
    """
    vectorstore_folder_list = drive.ListFile({
        'q': f"title='Vectorstore files' and '{parent_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
    }).GetList()

    if not vectorstore_folder_list:
        return False  # No 'Vectorstore files' folder

    vectorstore_folder_id = vectorstore_folder_list[0]['id']

    chroma_folder_list = drive.ListFile({
        'q': f"title='chroma_research_db' and '{vectorstore_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
    }).GetList()

    return bool(chroma_folder_list)



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

# device = "cuda" if torch.cuda.is_available() else "cpu"
# clip_model, preprocess = clip.load("ViT-B/32", device=device)  # ← BAD: runs on import
# clip_model.eval()

_clip_model = None
_preprocess = None

def get_clip_model_and_preprocess():
    global _clip_model, _preprocess
    if _clip_model is None or _preprocess is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _clip_model, _preprocess = clip.load("ViT-B/32", device=device)
        _clip_model.eval()
    return _clip_model, _preprocess



def get_image_embedding_from_pil(image: Image.Image):
    """
    Get image embedding from a PIL image using CLIP.
    """
    model, preprocess = get_clip_model_and_preprocess()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image_tensor)
        image_features /= image_features.norm(dim=-1, keepdim=True)
    return image_features[0].cpu().numpy().tolist()


# def get_image_embedding_from_pil(image: Image.Image):
#     """
#     Get image embedding from a PIL image using CLIP.
#     :param image: PIL Image object.
#     :return: Normalized image embedding as a list.
#     """
#     image_tensor = preprocess(image).unsqueeze(0).to(device)
#     with torch.no_grad():
#         image_features = clip_model.encode_image(image_tensor)
#         image_features /= image_features.norm(dim=-1, keepdim=True)
#     return image_features[0].cpu().numpy().tolist()


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


def build_vectorstore_and_upload(sheet, drive):
    """
    Build a new Chroma DB from Google Sheets and upload it to Google Drive.
    :param sheet: Google Sheets instance.
    :param drive: Google Drive instance.
    :return: None
    """

    valid_sheets = [ws.title for ws in sheet.worksheets() if is_valid_folderid(ws.title)]
    print("Valid sheets to process:", valid_sheets, "\n")

    parent_folder_id = '1QS6PmCESfgFWNNEpRJDUB0E-t8iMAatH'

    file_list = drive.ListFile({
        'q': f"title='Vectorstore files' and '{parent_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
    }).GetList()

    if file_list:
        vectorstore_folder_id = file_list[0]['id']
        print("Found 'Vectorstore files' folder inside the specified parent folder.\n")
    else:
        print("Creating 'Vectorstore files' folder inside the specified parent folder...")
        folder_metadata = {
            'title': 'Vectorstore files',
            'parents': [{'id': parent_folder_id}],
            'mimeType': 'application/vnd.google-apps.folder'
        }
        new_folder = drive.CreateFile(folder_metadata)
        new_folder.Upload()
        vectorstore_folder_id = new_folder['id']
        print("Created 'Vectorstore files' folder.\n")

    local_chroma_root = "/tmp/temp_chroma_folder"
    local_chroma_path = os.path.join(local_chroma_root, "chroma_graphics_db")
    os.makedirs(local_chroma_root, exist_ok=True)
    print("Local Chroma DB path:", local_chroma_path)

    embedding_function = get_embedding_model()
    chroma_db = Chroma(
        embedding_function=embedding_function,
        collection_name="text_embeddings",
        persist_directory=local_chroma_path
    )
    print("Embedding model and Chroma DB initialized.\n")

    for ws in sheet.worksheets():
        if not is_valid_folderid(ws.title):
            continue

        folder_id = ws.title
        print(f"Processing sheet '{folder_id}'...")

        records = ws.get_all_records()
        df = pd.DataFrame(records)

        if 'Image Description' not in df.columns:
            print(f"Sheet '{folder_id}' lacks 'Image Description'. Skipping.\n")
            continue

        for col in ['vectorized', 'embedding_ts', 'image_vectorized', 'image_embedding_ts', 'fully_vectorized']:
            if col not in df.columns:
                df[col] = ''

        mask_to_vectorize = (df['vectorized'] != 'TRUE') & df['Image Description'].str.strip().astype(bool)
        rows_to_vectorize = df.loc[mask_to_vectorize]
        print(f" Rows to vectorize (text): {len(rows_to_vectorize)}")

        if not rows_to_vectorize.empty:
            new_texts = rows_to_vectorize['Image Description'].tolist()

            new_metadatas = []
            for _, row in rows_to_vectorize.iterrows():
                phash = compute_phash_from_drive_url(row['Image Link'])

                meta = {
                    'image_id': row['Image ID'],
                    'name': row['Image Name'],
                    'drive_url': row['Image Link'],
                    'mime_type': row['MimeType'],
                    'description': row['Image Description'],
                    'image_type': row['Image Type'],
                    'image_title': row['Image Title'],
                    'folder_id': folder_id,
                    'embedding_type': 'text',
                    'phash': phash
                }
                new_metadatas.append(meta)

            BATCH_SIZE = 5000
            for i in range(0, len(new_texts), BATCH_SIZE):
                batch_texts = new_texts[i:i + BATCH_SIZE]
                batch_metadatas = new_metadatas[i:i + BATCH_SIZE]
                print(f"Adding text batch {i // BATCH_SIZE + 1} ({len(batch_texts)} items)...")
                chroma_db.add_texts(texts=batch_texts, metadatas=batch_metadatas)

            now = datetime.now().isoformat()
            df.loc[mask_to_vectorize, 'vectorized'] = 'TRUE'
            df.loc[mask_to_vectorize, 'embedding_ts'] = now

        save_to_sheet(ws, df)
        print(f"Sheet '{folder_id}' updated with all embedding statuses.\n")

    # ---- Deduplication ----
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
        print(f"Removed {len(id_url_duplicates)} duplicates by image_id or drive_url.")

    # ---- Visual Deduplication via pHash ----
    all_docs = chroma_db.get(include=["metadatas"])  # Fetch updated list
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
        print(f"🧠 Removed {len(phash_duplicates)} visually duplicate entries using pHash.")

    # Show remaining entries after all deduplication
    final_docs = chroma_db.get()
    remaining_count = len(final_docs["ids"])
    print(f"📦 Remaining entries in vectorstore after deduplication: {remaining_count}")

    chroma_db.persist()
    print("Chroma DB persisted locally at:", local_chroma_path)

    print("Uploading final 'chroma_graphics_db' to Drive...")
    upload_folder_to_drive(local_chroma_path, vectorstore_folder_id, drive)
    print("Upload complete: Vectorstore files/chroma_graphics_db/\n")

    print("✅ All embeddings stored, deduplicated, and uploaded successfully!")
    
def update_vectorstore(sheet, drive):
    """
    Update existing Chroma DB with new embeddings from Google Sheets and upload to Drive.
    :param sheet: Google Sheets instance.
    :param drive: Google Drive instance.
    :return: None
    """


    parent_folder_id = '1QS6PmCESfgFWNNEpRJDUB0E-t8iMAatH'
    local_chroma_root = "/tmp/temp_chroma_folder"
    local_chroma_path = os.path.join(local_chroma_root, "chroma_graphics_db")
    os.makedirs(local_chroma_root, exist_ok=True)
    print("Local Chroma DB path:", local_chroma_path)

    # ---------- TEXT EMBEDDINGS DB ----------
    print("Downloading existing 'chroma_graphics_db'...")
    file_list_text = drive.ListFile({
        'q': f"title='chroma_graphics_db' and '{parent_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
    }).GetList()

    if not file_list_text:
        raise FileNotFoundError("'chroma_graphics_db' (text) not found in Drive.")
    chroma_folder_id = file_list_text[0]['id']
    download_folder_from_drive(chroma_folder_id, local_chroma_path, drive)

    chroma_db = Chroma(
        embedding_function=get_embedding_model(),
        collection_name="text_embeddings",
        persist_directory=local_chroma_path
    )

    # # ---------- IMAGE EMBEDDINGS DB ----------
    # print("Downloading existing 'chroma_graphics_db' (image)...")
    # file_list_img = drive.ListFile({
    #     'q': f"title='chroma_graphics_db' and '{parent_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
    # }).GetList()

    # if not file_list_img:
    #     raise FileNotFoundError("'chroma_graphics_db' not found in Drive.")
    # chroma_folder_id_img = file_list_img[0]['id']
    # download_folder_from_drive(chroma_folder_id_img, local_chroma_path_image, drive)

    # image_chroma_db = Chroma(
    #     embedding_function=None,
    #     collection_name="image_embeddings",
    #     persist_directory=local_chroma_path_image
    # )

    # print("Both Chroma DBs loaded locally.\n")

    valid_sheets = [ws.title for ws in sheet.worksheets() if is_valid_folderid(ws.title)]
    print("Valid sheets to scan for updates:", valid_sheets, "\n")

    for sheet in sheet.worksheets():
        if not is_valid_folderid(sheet.title):
            continue

        folder_id = sheet.title
        print(f"Scanning sheet '{sheet.title}' (folder_id: {folder_id})...")

        records = sheet.get_all_records()
        df = pd.DataFrame(records)

        if 'Image Description' not in df.columns:
            print(f"Sheet '{sheet.title}' has no 'Image Description'. Skipping.\n")
            continue

        for col in ['vectorized', 'embedding_ts', 'image_vectorized', 'image_embedding_ts', 'fully_vectorized']:
            if col not in df.columns:
                df[col] = ''

        # ---------- TEXT ----------
        mask_to_vectorize = (df['vectorized'] != 'TRUE') & df['Image Description'].str.strip().astype(bool)
        rows_to_vectorize = df.loc[mask_to_vectorize]
        print(f" New rows to vectorize (text): {len(rows_to_vectorize)}")

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

        # # ---------- IMAGE ----------
        # mask_to_img = df['image_vectorized'] != 'TRUE'
        # rows_to_img = df.loc[mask_to_img]
        # print(f"New rows to vectorize (image): {len(rows_to_img)}")

        # for index, row in rows_to_img.iterrows():
        #     try:
        #         local_img_path = download_image_from_drive(drive, row['Image ID'])
        #         if not local_img_path or not os.path.exists(local_img_path):
        #             print(f"Could not download image: {row['Image ID']}")
        #             continue

        #         image_vector = get_image_embedding_from_pil(local_img_path)
        #         metadata = {
        #             'image_id': row['Image ID'],
        #             'name': row['Image Name'],
        #             'drive_url': row['Image Link'],
        #             'mime_type': row['MimeType'],
        #             'description': row['Image Description'],
        #             'image_type': row['Image Type'],
        #             'image_title': row['Image Title'],
        #             'folder_id': folder_id,
        #             'embedding_type': 'image'
        #         }

        #         image_chroma_db._collection.add(
        #             embeddings=[image_vector],
        #             metadatas=[metadata],
        #             documents=["image_embedding_only"],
        #             ids=[f"img_{row['Image ID']}"]
        #         )

        #         now = datetime.now().isoformat()
        #         df.at[index, 'image_vectorized'] = 'TRUE'
        #         df.at[index, 'image_embedding_ts'] = now

        #     except Exception as e:
        #         print(f"Error vectorizing image {row['Image ID']}: {e}")

        # # ---------- COMBINED STATUS ----------
        # df['fully_vectorized'] = df.apply(
        #     lambda r: 'TRUE' if r['vectorized'] == 'TRUE' and r['image_vectorized'] == 'TRUE' else '',
        #     axis=1
        # )

        save_to_sheet(sheet, df)
        print(f"Sheet '{folder_id}' updated.\n")


    
   # ---- Deduplication ----
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
        print(f"Removed {len(id_url_duplicates)} duplicates by image_id or drive_url.")

    # ---- Visual Deduplication via pHash ----
    all_docs = chroma_db.get(include=["metadatas"])  # Fetch updated list
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
        print(f"🧠 Removed {len(phash_duplicates)} visually duplicate entries using pHash.")

    # Show remaining entries after all deduplication
    final_docs = chroma_db.get()
    remaining_count = len(final_docs["ids"])
    print(f"📦 Remaining entries in vectorstore after deduplication: {remaining_count}")

    # ---------- PERSIST + UPLOAD ----------
    chroma_db.persist()
    # image_chroma_db.persist()
    print("Both DBs persisted locally.\n")

    print("removing old DB folders from Drive...")
    drive.CreateFile({'id': chroma_folder_id}).Delete()
    # drive.CreateFile({'id': chroma_folder_id_img}).Delete()

    print("Uploading updated DBs...")
    upload_folder_to_drive(local_chroma_path, parent_folder_id, drive)
    # upload_folder_to_drive(local_chroma_path_image, parent_folder_id, drive)

    print("Update complete. Both vectorstores are synced.")
