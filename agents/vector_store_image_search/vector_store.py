import os
import json
import base64
import re
from io import BytesIO
from PIL import Image
import pandas as pd
import datetime
from langchain_chroma import Chroma
from services.embedding_service import get_embedding_model
from dotenv import load_dotenv
from services.drive_service import login_with_service_account
from pydrive2.drive import GoogleDrive
from services.sheets_service import save_to_sheet
from services.drive_service import upload_folder_to_drive, download_folder_from_drive
import gspread
import streamlit as st
from modules.chain import Chain


load_dotenv()
# Decode and load Google Service Account credentials
key_bytes = base64.b64decode(os.environ["GDRIVE_SA_B64"])
sa_json = key_bytes.decode()
sa_dict = json.loads(sa_json)
gauth = login_with_service_account(json_str=sa_json)
gauth.ServiceAuth()
drive = GoogleDrive(gauth)

gc = gspread.service_account_from_dict(sa_dict)


st.session_state["drive"] = drive
st.session_state["gc"] = gc

sheet = st.session_state.get("sheet")


def is_valid_folderid(title):
    title = title.strip()
    if len(title) < 15:
        return False
    for c in title:
        if not (c.isalnum() or c in '-_'):
            return False
    return True


def build_vectorstore_and_upload(spreadsheet, drive):
    """
    Build a single vector store for all valid sheets in the spreadsheet and upload to Google Drive.

    All embeddings are stored in a single local chroma_research_db folder and uploaded to
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
    local_chroma_path = os.path.join(local_chroma_root, "chroma_research_db")
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

    # Upload the single local chroma_research_db to Drive
    print("☁️ Uploading final 'chroma_research_db' to 'Vectorstore files' in Drive...")
    upload_folder_to_drive(local_chroma_path, vectorstore_folder_id, drive)
    print("✅ Uploaded final Chroma DB to: Vectorstore files/chroma_research_db/\n")

    print("🎉 All embeddings stored in a single Chroma DB, uploaded, and sheets updated!")


def update_vectorstore(spreadsheet, drive):
    """
    Update an existing Chroma DB by processing:
    - New sheets added to the spreadsheet (new folder IDs)
    - New rows in existing sheets where vectorized != TRUE

    Only appends new data to an already existing local Chroma DB (chroma_research_db),
    and re-uploads it to Drive.
    """

    parent_folder_id = '1QS6PmCESfgFWNNEpRJDUB0E-t8iMAatH'
    local_chroma_root = "/tmp/temp_chroma_folder"
    local_chroma_path = os.path.join(local_chroma_root, "chroma_research_db")
    os.makedirs(local_chroma_root, exist_ok=True)

    # Find the existing Chroma DB on Drive and download it
    print("Downloading existing 'chroma_research_db' from Drive...")
    file_list = drive.ListFile({
        'q': f"title='chroma_research_db' and '{parent_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
    }).GetList()

    if not file_list:
        raise FileNotFoundError("Existing 'chroma_research_db' not found in Google Drive.")

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
            print(f"Adding batch {i//BATCH_SIZE + 1} ({len(batch_texts)} items)...")
            chroma_db.add_texts(texts=batch_texts, metadatas=batch_metadatas)

        # Mark vectorized rows in the sheet
        now = datetime.datetime.now().isoformat()
        df.loc[mask_to_vectorize, 'vectorized'] = 'TRUE'
        df.loc[mask_to_vectorize, 'embedding_ts'] = now

        save_to_sheet(sheet, df)
        print(f"Sheet '{folder_id}' updated with vectorization status.\n")

    # Persist and upload the updated DB
    chroma_db.persist()
    print("💾 Chroma DB updated and persisted locally.")

    # Upload back to Drive
    print("Uploading updated 'chroma_research_db' to Google Drive...")
    upload_folder_to_drive(local_chroma_path, parent_folder_id, drive)
    print("Update complete. Vectorstore is now synced with spreadsheet.\n")



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


def download_image_from_drive(file_id, drive):


    file = drive.CreateFile({'id': file_id})
    file.FetchMetadata(fields='title, mimeType')

    # Download to a temporary file
    temp_file = 'temp_image'
    file.GetContentFile(temp_file)
    with open(temp_file, 'rb') as f:
        img = Image.open(BytesIO(f.read()))
    return img


def load_central_chroma_db(embedding_function, drive, central_folder_id):
    """
    Load the single central Chroma DB stored in Google Drive (in 'chroma_research_db' inside the central_folder_id).

    Parameters:
    - embedding_function: Function to generate embeddings.
    - drive: Authenticated Google Drive instance.
    - central_folder_id (str): ID of the parent folder containing the single Chroma DB.

    Returns:
    - chroma_db: The loaded Chroma DB.
    """
    # Check for the 'chroma_research_db' folder inside the central folder
    file_list = drive.ListFile({
        'q': f"title='chroma_research_db' and '{central_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
    }).GetList()

    if not file_list:
        raise FileNotFoundError("❌ 'chroma_research_db' folder not found in the central folder.")

    chroma_folder_id = file_list[0]['id']
    print(f"✅ Found 'chroma_research_db' folder in central folder (id: {chroma_folder_id}).")

    # Download the Chroma DB folder to a local directory
    local_chroma_root = "/tmp/central_chroma_folder"
    local_chroma_path = os.path.join(local_chroma_root, "chroma_research_db")
    os.makedirs(local_chroma_root, exist_ok=True)
    sqlite_db_path = os.path.join(local_chroma_path, "chroma.sqlite3")

    # 1. Check local existence
    if os.path.exists(local_chroma_path) and os.path.exists(sqlite_db_path):
        print("Database already exists locally. Skipping download.")
    else:
        print("Database not found locally. Checking / creating Drive folders...")
        
        download_folder_from_drive(chroma_folder_id, local_chroma_path, drive)
        print(f"✅ Downloaded 'chroma_research_db' to: {local_chroma_path}")

    # Initialize Chroma DB from the local folder
    chroma_db = Chroma(
        embedding_function=embedding_function,
        collection_name="text_embeddings",
        persist_directory=local_chroma_path
    )

    print("✅ Chroma DB loaded from the central folder.")
    return chroma_db


def graphics_retriever(query, drive, k=5, filters=None):
    """
    Search for similar images using a text query and download them from Drive.
    """
    print("🔍 Loading central Chroma DB...")
    embedding_function = get_embedding_model()
    central_folder_id = '1ujM1OkJRUcQlg2_ZhRIE-kOZa1m-Qgnc'  
    
    try:
        # Load DB with verification
        chroma_db = load_central_chroma_db(embedding_function, drive, central_folder_id)
        
        # Verify DB is loaded and accessible
        count = chroma_db._collection.count()
        print(f"✓ DB loaded with {count} items")
        
        # Verify query is valid
        if not query or not isinstance(query, str):
            print("⚠️ Invalid query")
            return []
            
        print(f"🔎 Starting similarity search for: '{query}'")
        
        # Perform similarity search with explicit verification
        try:
            # First try a small search to verify functionality
            test_results = chroma_db.similarity_search_with_score(query, k=1)
            print("✓ Search functionality verified")
            
            # If test passes, do full search
            print(f"🔍 Performing full search (k={k})")
            results_docs = chroma_db.similarity_search_with_score(query, k=50)
            
            if not results_docs:
                print("ℹ️ No results found")
                return []
                
            print(f"✓ Found {len(results_docs)} initial results")
            
            # Filter results
            filtered_results = []
            print("🔍 Applying filters...")
            
            for doc, score in results_docs:
                metadata = doc.metadata
                
                # Verify metadata structure
                if not all(key in metadata for key in ['image_id', 'name', 'drive_url', 'description', 'folder_id']):
                    print(f"⚠️ Skipping result with incomplete metadata")
                    continue
                    
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
            
            print(f"✓ Filtering complete - {len(filtered_results)} results remain")
            
            # Sort and return results
            filtered_results.sort(key=lambda x: x['similarity'])
            final_results = filtered_results[:k]
            
            print(f"✓ Returning top {len(final_results)} results")
            return final_results
            
        except Exception as e:
            print(f"❌ Search error: {str(e)}")
            import traceback
            traceback.print_exc()
            return []
            
    except Exception as e:
        print(f"❌ DB error: {str(e)}")
        import traceback
        traceback.print_exc()
        return []


graphics_retriever_agent_prompt = """
You are a graphics search agent that selects the most visually relevant image(s) for a presentation slide. You have access to a vector database of image captions and metadata.

Your goal is to select up to 2 Google Drive image URLs that best match the slide description visually.

If no good images are found, you may refine the query and re-search up to 3 times. If nothing fits after all attempts, return "NONE".

You'll be able to receive, for each search turn, you are given:
- The slide description (what the user wants to visualize).
- The current search query used.
- A list of image search results, each with:
  - Title
  - Description
  - Similarity Score
  - Drive URL

What you are expected do do is:
1. Carefully review the image results.
   - Consider the title, description, and similarity score.
   - Decide if any of the results clearly match the visual idea of the slide description.

2. If 1-2 good images are found:
   - Select the most visually relevant 1 or 2 image URLs.
   - End the process by setting the verdict to "TERMINATE".

3. If none of the results are a good match:
   - Decide whether it's worth refining the query.
   - Suggest a clear and specific refinement strategy (e.g., add or change keywords).
   - Provide a new query that is more likely to get relevant results.
   - Set the verdict to "CONTINUE".

4. If after 3 attempts, you still find no useful images:
   - Set the verdict to "TERMINATE" and return "NONE".

The output format should be as follows:
Respond using the following exact tags:

<observations>
[Summarize what the results contained. Say whether any images visually match the slide description. Comment on strengths/weaknesses of the results.]
</observations>

<verdict>
["TERMINATE" if you're confident in your image selections or no good images exist, otherwise "CONTINUE"]
</verdict>

<selected_urls>
[List 1 or 2 Google Drive image URLs if you found any suitable matches. Leave blank if verdict is CONTINUE.]
</selected_urls>

<action>
[If CONTINUE: explain how you want to change the search query (e.g., “focus on technician using tools”)]
</action>

<query>
[If CONTINUE: provide the revised text query you want to use.]
</query>
"""


def graphics_retriever_agent(
    slide_text: str,
    drive,
    llm_name: str = "gemini_2_flash",
    k: int = 10,
    max_turns: int = 3,
    filters: dict = None,
    verbose: bool = True
):
    """
    LLM-driven agent that finds the best image(s) for a slide from a vector store,
    refining its query over multiple turns based on LLM feedback.

    :param slide_text: Text of the slide needing a visual.
    :param retriever: Vector retriever instance for image metadata.
    :param drive: Authenticated PyDrive object for image download.
    :param llm_name: LLM backend to use (e.g., 'gpt-4o').
    :param k: Number of top results to show per query.
    :param max_turns: Max number of re-query attempts.
    :param filters: Optional filter criteria (e.g., mime_type).
    :param verbose: If True, prints debugging info.
    :return: List of Drive URLs, or ["NONE"] if none found.
    """

    # Initialize the LLM chain and add system prompt
    graphics_retriever_agent = Chain(
        llm=llm_name,
        tags=["observations", "verdict", "selected_urls", "action", "query"]
    )
    graphics_retriever_agent.add_message(role="system", content=graphics_retriever_agent_prompt)

     # Initial query is the slide text
    query = slide_text

    for turn in range(max_turns):
        if verbose:
            print(f"\n🔎 Turn {turn + 1}: Querying with — {query}")

        # Call the retriever to get matching images
        results = graphics_retriever(query, drive=drive, k=k, filters=filters)

        # Format the retrieved results into a string for the LLM
        if results:
            results_str = ""
            for i, res in enumerate(results):
                results_str += f"""
                ======== Result {i} ========
                Title: {res.get('name', 'N/A')}
                Description: {res.get('description', 'N/A')}
                Drive URL: {res.get('drive_url', 'N/A')}
                Similarity Score: {res.get('score', 'N/A'):.4f}
                """
        else:
            results_str = "No search results found for this query."

        # Construct the user prompt for the LLM
        user_msg = f"""
        Slide description:
        "{slide_text}"

        Query:
        "{query}"

        Search Results:
        {results_str}
        """

        if verbose:
            print("Sending user message to LLM:\n", user_msg)

        # Add message to the agent chain
        graphics_retriever_agent.add_message(role="user", content=user_msg)

        try:
            response = graphics_retriever_agent.run()
        except Exception as e:
            print(f"LLM call failed: {e}")
            return ["NONE"]

        if verbose:
            print("\nLLM Response:")
            print(response.get('observations', 'No observations tag found'))
            print("Verdict:", response.get('verdict', 'No verdict tag found'))
            print("Selected URLs:", response.get('selected_urls', 'No selected_urls tag found'))
            print("Action:", response.get('action', 'No action tag found'))
            print("New Query:", response.get('query', 'No query tag found'))

        # Check whether LLM wants to terminate or continue
        verdict = response.get('verdict', '').strip().upper()
        if verdict == "TERMINATE":
            urls_content = response.get('selected_urls', '')
            urls = [url.strip() for url in urls_content.splitlines() if "http" in url or "drive.google.com" in url]
            if verbose:
                print(f"\nTERMINATING. Selected URLs: {urls}")
            return urls if urls else ["NONE"]

        # Continue with refined query
        query = response.get('query', '').strip()
        if not query:
            print("Warning: LLM returned 'CONTINUE' verdict but no new query. Terminating.")
            return ["NONE"]
        if verbose:
            print(f"CONTINUING with new query: {query}")

    print("\nMax turns reached without finding suitable images.")
    return ["NONE"]





    

