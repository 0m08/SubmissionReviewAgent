import os
import pickle
import json
import streamlit as st
from modules.chain import Chain
from langchain_core.documents import Document
from langchain_cohere import CohereRerank
from langchain_community.vectorstores import Chroma
from typing import Any, Dict, List, Optional, Tuple
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from services.embedding_service import get_embedding_model
from langchain_classic.retrievers import ContextualCompressionRetriever
from agents.vector_store_image_search.create_vectorstore import download_folder_from_drive
import base64
import vertexai
from chromadb import PersistentClient
from vertexai.vision_models import MultiModalEmbeddingModel
from vertexai.generative_models import Part
from langchain_core.runnables import Runnable

VIDEO_CENTRAL_FOLDER_ID = '1kovlkUd3pN5IGDB16LC2H8grvmQOhXHy'

_VIDEO_VECTOR_DB_CACHE: Dict[str, Tuple[Chroma, Any]] = {}
_VIDEO_BM25_CACHE: Dict[str, BM25Retriever] = {}
_VIDEO_COMPRESSION_CACHE: Dict[str, ContextualCompressionRetriever] = {}


def load_video_chroma_db(embedding_function, drive, central_folder_id):
    """
    Load the shared Chroma DB collection for HVAC videos from Google Drive.
    :param embedding_function: Function to compute text embeddings.
    :param drive: Authenticated PyDrive2 instance.
    :param central_folder_id: Drive folder containing 'chroma_video_db'.
    :return: Chroma vectorstore instance.
    """

    # Local path for Chroma DB
    local_chroma_root = "/tmp/temp_chroma_folder"
    local_chroma_path = os.path.join(local_chroma_root, "chroma_video_db")
    os.makedirs(local_chroma_root, exist_ok=True)

    # Search for the Chroma DB folder in Drive
    print("Searching for 'chroma_video_db' in Drive...")
    file_list = drive.ListFile({
        'q': f"title='chroma_video_db' and '{central_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
    }).GetList()

    if not file_list:
        raise FileNotFoundError("'chroma_video_db' not found in Drive.")

    chroma_folder_id = file_list[0]['id']
    print(f"Found Chroma folder ID: {chroma_folder_id}")

    # Download folder if not already present
    if not os.path.exists(os.path.join(local_chroma_path, "chroma.sqlite3")):
        print("⬇ Downloading Chroma DB from Drive...")
        download_folder_from_drive(chroma_folder_id, local_chroma_path, drive)
        print(f"Downloaded to local path: {local_chroma_path}")
    else:
        print("Using existing local Chroma DB.")

    # Load video embeddings collection
    video_chroma = Chroma(
        embedding_function=embedding_function,
        collection_name="video_embeddings",
        persist_directory=local_chroma_path
    )
    print("Loaded 'video_embeddings' collection.")
    return video_chroma


def load_video_vector_db_retriever(drive, central_folder_id: str = VIDEO_CENTRAL_FOLDER_ID):
    """
    Load the vector database retriever.
    :param course_name: The name of the course.
    :param course_drive_folder_id: The ID of the folder containing the vector database.
    :param drive: Authenticated GoogleDrive instance (PyDrive2).
    :param sheet: The Google Sheets object.
    :return: Loaded vector database retriever object.
    """

    cache_key = central_folder_id
    if cache_key in _VIDEO_VECTOR_DB_CACHE:
        return _VIDEO_VECTOR_DB_CACHE[cache_key]

    embedding_model = get_embedding_model()

    chroma_db = load_video_chroma_db(embedding_model, drive, central_folder_id)
    vector_db_retriever = chroma_db.as_retriever(
        search_kwargs = {
            "k": 20,
        }
    )
    print("Loaded vector database retriever.")

    _VIDEO_VECTOR_DB_CACHE[cache_key] = (chroma_db, vector_db_retriever)
    return chroma_db, vector_db_retriever


def load_new_video_embeddings_chroma_db(drive, video_embeddings_folder_id):
    """
    Load the new multimodal video embeddings Chroma DB from Google Drive.
    
    :param drive: Authenticated PyDrive2 instance.
    :param video_embeddings_folder_id: Drive folder ID containing 'Vectorstore for HVAC school video embeddings'.
    :return: Chroma vectorstore instance.
    """

    # Local path for new video embeddings Chroma DB
    local_chroma_path = "/tmp/HVAC Video Embeddings"
    os.makedirs(local_chroma_path, exist_ok=True)

    # Search for the 'Vectorstore for HVAC school video embeddings' folder in Drive
    print("Searching for 'Vectorstore for HVAC school video embeddings' in Drive...")
    file_list = drive.ListFile({
        'q': f"title='Vectorstore for HVAC school video embeddings' and '{video_embeddings_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
    }).GetList()

    if not file_list:
        raise FileNotFoundError("'Vectorstore for HVAC school video embeddings' not found in Drive.")

    chroma_folder_id = file_list[0]['id']
    print(f"Found Video Embeddings folder ID: {chroma_folder_id}")

    # Download folder if not already present
    lock_file = os.path.join(local_chroma_path, ".download_lock")
    
    if not os.path.exists(os.path.join(local_chroma_path, "chroma.sqlite3")):
        
        # Check if another process is already downloading
        if os.path.exists(lock_file):
            print("⏳ Another process is downloading the vectorstore. Waiting...")
            while os.path.exists(lock_file):
                time.sleep(1)
            # Check again after waiting
            if not os.path.exists(os.path.join(local_chroma_path, "chroma.sqlite3")):
                print("⬇ Downloading Video Embeddings Chroma DB from Drive...")
                with open(lock_file, 'w') as f:
                    f.write(str(os.getpid()))
                try:
                    download_folder_from_drive(chroma_folder_id, local_chroma_path, drive)
                    print(f"Downloaded to local path: {local_chroma_path}")
                finally:
                    if os.path.exists(lock_file):
                        os.remove(lock_file)
            else:
                print("Using existing local Video Embeddings Chroma DB (downloaded by another process).")
        else:
            print("⬇ Downloading Video Embeddings Chroma DB from Drive...")
            with open(lock_file, 'w') as f:
                f.write(str(os.getpid()))
            try:
                download_folder_from_drive(chroma_folder_id, local_chroma_path, drive)
                print(f"Downloaded to local path: {local_chroma_path}")
            finally:
                if os.path.exists(lock_file):
                    os.remove(lock_file)
    else:
        print("Using existing local Video Embeddings Chroma DB.")

    # Load video embeddings collection 
    client = PersistentClient(path=local_chroma_path)
    
    try:
        collections = client.list_collections()
        print(f"Available collections in vectorstore: {[c.name for c in collections]}")
        
        if not collections:
            print("❌ No collections found in vectorstore. The vectorstore is empty.")
            raise Exception("Vectorstore is empty - no video embeddings have been created yet.")
        
        collection = client.get_collection("video_embeddings")
        print("Loaded 'video_embeddings' collection from new vectorstore.")
        
        # Create a wrapper to make it compatible with LangChain
        class VideoEmbeddingsChroma:
            def __init__(self, collection):
                self.collection = collection
                
            def similarity_search(self, query, k=10, **kwargs):
                # Use Vertex AI multimodal embedding model to generate query embeddings
                print(f"🔍 Searching video embeddings for query: '{query[:50]}...'")
                
                try:
                    # Get Vertex AI service account credentials from environment
                    if "VERTEX_AI_SA_B64" in os.environ:
                        key_bytes = base64.b64decode(os.environ["VERTEX_AI_SA_B64"])
                        sa_json = key_bytes.decode()
                        sa_dict = json.loads(sa_json)
                        
                        # Initialize Vertex AI with service account
                        from google.oauth2 import service_account
                        creds = service_account.Credentials.from_service_account_info(sa_dict)
                        
                        # Initialize Vertex AI with the paid project (dam-images-tagging)
                        vertexai.init(
                            project="dam-images-tagging",  # Paid project with Vertex AI API enabled
                            location="us-central1",
                            credentials=creds
                        )
                        
                        print(f"✅ Vertex AI initialized with project: dam-images-tagging")
                        
                        # Load the multimodal embedding model
                        model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding@001")
                        
                        # Generate query embedding for text queries
                        result = model.get_embeddings(contextual_text=query)
                        query_embedding = result.text_embedding
                        
                        print(f"✅ Using Vertex AI multimodal embeddings for query")
                        print(f"✅ Query embedding dimension: {len(query_embedding)}")
                       
                    else:
                        raise Exception("VERTEX_AI_SA_B64 environment variable not found. Please add it to your .env file.")
                        
                except Exception as e:
                    print(f"❌ Error initializing Vertex AI: {str(e)}")
                    raise 
                
                # Query the vector store with the embedding
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=k
                )
                
                # Convert to LangChain Document format
                docs = []
                if results['documents'] and results['documents'][0]:
                    for i, doc_content in enumerate(results['documents'][0]):
                        metadata = results['metadatas'][0][i] if results['metadatas'] and results['metadatas'][0] else {}
                        
                        if doc_content and doc_content.strip():
                            page_content = doc_content
                        else:
                            title = metadata.get('title', 'Unknown Video')
                            start_time = metadata.get('start_time', 0)
                            end_time = metadata.get('end_time', 'end')
                            video_url = metadata.get('video_url', '')
                            segment_index = metadata.get('segment_index', 0)
                            page_content = f"Video: {title} | Segment: {segment_index} | Time: {start_time}s-{end_time}s | URL: {video_url}"
                        
                        docs.append(Document(page_content=page_content, metadata=metadata))
                
                return docs
                
            def as_retriever(self, search_kwargs=None):
                if search_kwargs is None:
                    search_kwargs = {"k": 10}
                return VideoEmbeddingsRetriever(self, search_kwargs)
        
        class VideoEmbeddingsRetriever(Runnable):
            def __init__(self, chroma_db, search_kwargs):
                self.chroma_db = chroma_db
                self.search_kwargs = search_kwargs
                
            def invoke(self, input: Any, config: Any = None) -> List[Any]:
                return self.chroma_db.similarity_search(input, **self.search_kwargs)
            
            def _invoke(self, input: Any, config: Any = None) -> List[Any]:
                return self.invoke(input, config)
        
        return VideoEmbeddingsChroma(collection)
        
    except Exception as e:
        print(f"Error loading video embeddings collection: {e}")
        raise


def load_bm25_retriever_with_pydrive(central_folder_id: str, drive):
    """
    Load the BM25 retriever from a pickle file stored in Google Drive.

    Optimization:
    - If the file is already downloaded, skip the download process.
    - If missing, download the file from Google Drive.

    :param root_folder_id: The ID of the root folder containing 'Pickle files'.
    :param drive: Authenticated GoogleDrive instance (PyDrive2).
    :param all_doc_chunk_list: The list of all document chunks.
    :return: Loaded BM25 retriever object.
    """
    if central_folder_id in _VIDEO_BM25_CACHE:
        return _VIDEO_BM25_CACHE[central_folder_id]

    local_pickle_path = "/tmp/bm25_retriever.pkl"

    # Check if already downloaded
    if os.path.exists(local_pickle_path):
        print("BM25 retriever already exists locally. Skipping download.")
    else:
        print("BM25 retriever not found locally. Downloading from Google Drive...")

        # Locate 'Pickle files' folder
        query_pickle_folder = (
            f"title='Pickle files' and '{central_folder_id}' in parents "
            f"and mimeType='application/vnd.google-apps.folder'"
        )
        pickle_folders = drive.ListFile({'q': query_pickle_folder}).GetList()

        if not pickle_folders:
            print(f"No folder named 'Pickle files' found in folder ID {central_folder_id}.")
            file_metadata = {
                'title': 'Pickle files',
                'parents': [{'id': central_folder_id}],
                'mimeType': 'application/vnd.google-apps.folder'
            }
            pickle_folder = drive.CreateFile(file_metadata)
            pickle_folder.Upload()
            pickle_folder_id = pickle_folder['id']
            print(f"Created 'Pickle files' folder with ID {pickle_folder_id}.")
        else:
            pickle_folder = pickle_folders[0]
            pickle_folder_id = pickle_folder['id']

        # Locate 'bm25_research_db.pkl' inside 'Pickle files'
        query_bm25 = (
            f"title='bm25_research_db.pkl' and '{pickle_folder_id}' in parents"
        )
        bm25_files = drive.ListFile({'q': query_bm25}).GetList()
        if not bm25_files:
            print("bm25_research_db.pkl not found under 'Pickle files'.")
            
            embedding_function = get_embedding_model()
            video_db = load_video_chroma_db(embedding_function, drive, central_folder_id)
            all_docs = video_db.get()
            documents = [
                Document(page_content=all_docs['documents'][i], metadata=all_docs['metadatas'][i] or {})
                for i in range(len(all_docs['ids']))
                ]
            # Create the bm_25 retriever
            bm_25_retriever = BM25Retriever.from_documents(documents, k = 20, )

            # Save as pickle file locally
            with open(local_pickle_path, 'wb') as file:
                pickle.dump(bm_25_retriever, file)

            # Upload the pickle file to Google Drive
            file_metadata = {
                'title': 'bm25_research_db.pkl',
                'parents': [{'id': pickle_folder_id}]
            }
            bm25_file = drive.CreateFile(file_metadata)
            bm25_file.SetContentFile(local_pickle_path)
            bm25_file.Upload()
            print("bm25_research_db.pkl uploaded to Google Drive.")
            _VIDEO_BM25_CACHE[central_folder_id] = bm_25_retriever
            return bm_25_retriever

        else:
            print("bm25_research_db.pkl found under 'Pickle files'")

            bm25_file = bm25_files[0]
            bm25_file.GetContentFile(local_pickle_path)

    # Load the retriever from the pickle file
    with open(local_pickle_path, 'rb') as file:
        bm_25_retriever = pickle.load(file)
    print("Successfully loaded the BM25 retriever from Google Drive.")

    _VIDEO_BM25_CACHE[central_folder_id] = bm_25_retriever
    return bm_25_retriever


def get_ensemble_retriever(central_folder_id, drive, retriever_1_weight = 0.5, retriever_2_weight = 0.5):
    """
    Get the ensemble retriever.

    :param course_name: The name of the course.
    :param root_folder_id: The ID of the root folder containing 'Pickle files' and 'Vectorstore files'.
    :param drive: Authenticated GoogleDrive instance (PyDrive2).
    :param sheet: The Google Sheets object.
    :param retriever_1_weight: The weight of the BM25 retriever.
    :param retriever_2_weight: The weight of the vector database retriever.
    :return: Ensemble retriever object.
    """
    with st.spinner(text = "Loading the embedding vectorstore...", show_time = True):
        _, vector_db_retriever = load_video_vector_db_retriever(drive, central_folder_id)

    with st.spinner(text = "Loading the bm25 vectorstore...", show_time = True):
        bm_25_retriever = load_bm25_retriever_with_pydrive(central_folder_id, drive)

    ensemble_retriever = EnsembleRetriever(
        retrievers = [bm_25_retriever, vector_db_retriever],
        weights = [retriever_1_weight, retriever_2_weight],
    )
    return ensemble_retriever


def get_compression_retriever(central_folder_id, drive, retriever_1_weight = 0.5, retriever_2_weight = 0.5):
    """
    Get the compression retriever.

    :param: course_name: The name of the course.
    :param: root_folder_id: The ID of the root folder containing 'Pickle files' and 'Vectorstore files'.
    :param: drive: Authenticated GoogleDrive instance (PyDrive2).
    :param: sheet: The Google Sheets object.
    :param: retriever_1_weight: The weight of the BM25 retriever.
    :param: retriever_2_weight: The weight of the vector database retriever.
    :return: Compression retriever object.
    """

    cache_key = central_folder_id
    if cache_key in _VIDEO_COMPRESSION_CACHE:
        return _VIDEO_COMPRESSION_CACHE[cache_key]

    ensemble_retriever = get_ensemble_retriever(central_folder_id, drive, retriever_1_weight, retriever_2_weight)

    compressor = CohereRerank(
        model="rerank-v3.5",
        top_n=15,
    )

    compression_retriever = ContextualCompressionRetriever(
        base_compressor = compressor, base_retriever = ensemble_retriever
    )

    _VIDEO_COMPRESSION_CACHE[cache_key] = compression_retriever
    return compression_retriever


def doc_matches_filters(metadata: dict, filters: Dict[str, Any]) -> bool:
    for key, expected in filters.items():
        actual = metadata.get(key)

        # Handle published year
        if key == "published" and isinstance(expected, int):
            try:
                actual_year = int(str(actual)[:4])
                if actual_year != expected:
                    return False
            except Exception:
                return False

        elif isinstance(expected, str) and isinstance(actual, str):
            if actual.strip().lower() != expected.strip().lower():
                return False

        else:
            if actual != expected:
                return False

    return True


def video_retriever(
    query: str,
    drive,
    k: int = 10,
    filters: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Returns up to `k` video metadata dicts, including:
    - 'url', 'video_id', 'start_time', 'end_time'
    - 'channel', 'published', etc.
    - 'transcript' 
    """

    central_folder_id = VIDEO_CENTRAL_FOLDER_ID
    compression_retriever = get_compression_retriever(
        central_folder_id=central_folder_id,
        drive=drive,
        retriever_1_weight=0.5,
        retriever_2_weight=0.5,
    )

    docs = compression_retriever.invoke(query)

    results: List[Dict[str, Any]] = []
    seen: set = set()

    for doc in docs:
        if len(results) >= k:
            break

        metadata = getattr(doc, "metadata", {}) or {}
        vid_id = metadata.get("video_id")
        if not vid_id or vid_id in seen:
            continue

        if filters and not doc_matches_filters(metadata, filters):
            continue

        def _to_int_or_none(v):
            try:
                return int(float(v)) if v is not None and str(v) != "" else None
            except Exception:
                return None

        start_sec = _to_int_or_none(metadata.get("start_time")) or 0
        end_sec = _to_int_or_none(metadata.get("end_time"))
        transcript = metadata.get("text_0", "Transcript not available.")


        if end_sec is not None:
            url = f"https://www.youtube.com/embed/{vid_id}?start={start_sec}&end={end_sec}"
        else:
            url = f"https://www.youtube.com/embed/{vid_id}?start={start_sec}"

        seen.add(vid_id)

        result = {
            "url": url,
            "video_id": vid_id,
            "start_time": start_sec,
            "end_time": end_sec,
            "transcript": transcript,
        }
        result.update(metadata)  # Add channel, published, video_title, etc.
        results.append(result)

    return results


def warmup_video_retriever(drive) -> None:
    """Preload caches so later queries avoid repeated initialization."""
    get_compression_retriever(
        central_folder_id=VIDEO_CENTRAL_FOLDER_ID,
        drive=drive,
        retriever_1_weight=0.5,
        retriever_2_weight=0.5,
    )


video_search_retriever_agent_prompt = """
You are an expert in HVAC video evaluation, tasked with selecting the most relevant video(s) to match a given query.

Your goal is to identify videos that best match the intent of the search query based on their content, not just metadata or similarity scores.
If the retrieved videos do not adequately cover the query, you should refine the search query to improve results.

For each search turn, you will receive:
- The search query used to retrieve results
- A list of video search results with metadata (title, description, source)

Your responsibilities are as follows:

---

1. VISUAL AND CONTEXTUAL INSPECTION
   - Evaluate whether each video clearly demonstrates or explains the concept in the query.
   - Use video metadata (title, description, channel) only as supporting information.
   - Ignore similarity score if the content does not actually match the instructional intent.
  

---

2. IF RELEVANT VIDEOS ARE FOUND

If you found any clearly relevant and matching videos, mark them and set verdict = TERMINATE.
Do not suggest refinements if a usable set of results has already been found.
Videos that are close to the search query with the same intent are acceptable.

---

3. IF RESULTS ARE POOR OR IRRELEVANT

   - Determine why:
       * Is the query vague or too general?
       * Does the content mismatch the instructional intent?
       * Are domain terms missing or ambiguous?
   - Suggest a refined query keeping intent intact.

- When refining:
  - Disambiguate vague terms
  - Add context (e.g., "commercial electrical panel wiring tutorial")
  - Clarify content type (e.g., "training video", "troubleshooting guide", "installation walkthrough")
  - Preserve the **intent and scope** of the original query
  - Since most of the queries used are learning objective for a certain course, when requerying, make sure to follow the format of the original query

---

4. IF AFTER 3 REFINED ATTEMPTS NO MATCH IS FOUND

- Set verdict to "TERMINATE" and return the closely found and retrieved videos only return "NONE" if:
  - The query is fundamentally unrelated to video content
  - All refinements have failed to yield even loosely related results
  - The subject cannot be represented in video form

---

Important Considerations:
When evaluating videos, consider the following guidelines:
1. Focus on the instructional intent of the query, not just keyword matches.
2. Use metadata only to support your content evaluation.
3. Be decisive: either select relevant videos or refine the query.
4. Maintain the original intent when refining queries.
5. Limit to 3 refinement attempts before terminating.
6. If no relevant videos are found after 3 attempts, return "NONE".
7. If the initial query yields relevant results, do not refine further.

RESPONSE FORMAT:

Respond using these exact tags:

<observations>
Summarize what you observed in the video results and their relevance.
</observations>

<verdict>
["TERMINATE" if you're confident in your video selection(s) or none are good at all, otherwise "CONTINUE" only if the videos are totally irrelevant]
</verdict>

<selected_indexes>
[If TERMINATE: return the indexes of selected videos from the list above, e.g., "0", "1". Leave blank if CONTINUE.]
</selected_indexes>

<action>
[If CONTINUE: explain what is wrong with the current query and how you'll refine it.
 If TERMINATE: briefly state that the query was sufficient, or that no relevant videos were found after exhaustive attempts.]
</action>

<query>
[If CONTINUE: provide your improved, more precise search query.
 If TERMINATE: repeat the original query to confirm no refinement was needed or indicate failure after 3 attempts.]
</query>
"""


def parse_selected_indexes(index_string):
    """
    Safely extract index list from string like '0, 2' or '1'
    :param index_string: String to parse.
    :return: List of indexes.
    """
    try:
        return [int(i.strip()) for i in index_string.split(",") if i.strip().isdigit()]
    except:
        return []

    
def video_search_retriever_agent(
    query: str,
    drive,
    llm,
    k: int = 10,
    max_turns: int = 3,
    filters: Optional[Dict] = None,
    verbose: bool = True,
) -> List[Dict]:
    """
    Agent that uses LLM to iteratively refine video search queries and select relevant videos.
        :param query: Initial search query.
        :param drive: Authenticated PyDrive2 instance.
        :param llm: Language model instance with chat/completion interface.
        :param k: Number of top videos to retrieve each turn.
        :param max_turns: Maximum number of refinement turns.
        :param filters: Optional metadata filters to apply to retrieved videos.
        :param verbose: Whether to print debug info.
        :return: List of selected video metadata dicts.
    """
    

    chain = Chain(
        llm=llm,
        tags=["observations", "verdict", "selected_indexes", "action", "query"],
        use_xml_checker=False
    )

    # Add system prompt
    chain.add_message(role="system", content=video_search_retriever_agent_prompt)

    original_query = query

    for turn in range(max_turns):
        if verbose:
            print(f"\nTurn {turn + 1}: Query = '{query}'")

        # Retrieve top-k videos
        results = video_retriever(query=query, drive=drive, k=k, filters=filters)
        if not results:
            print("No videos retrieved.")
            return []

        # Build candidate list prompt
        video_list_str = "\n".join(
            [f"{i}: {v['video_title']} ({v.get('channel', 'Unknown Channel')})" for i, v in enumerate(results)]
        )
        user_prompt = f"""
Original query: {original_query}
Candidate videos:
{video_list_str}
"""
        chain.add_message(role="user", content=user_prompt)

        # Run the chain
        response = chain.run(query=user_prompt)

        # Debug: inspect the raw response
        print(f"Debug: Type of response = {type(response)}")
        print(f"Debug: Response content = {response}")

        # Normalize the response
        if isinstance(response, dict):
            if "output" in response:
                content = response["output"]
            elif "text" in response:
                content = response["text"]
            else:
                raise ValueError(f"Unexpected response format: {response}")
        else:
            content = response

        # Join list outputs into a string
        if isinstance(content, list):
            content = "\n".join(content)

        chain.add_message(role="ai", content=content)

        # Extract tags
        try:
            llm_response = chain.extract_text_in_tags(content)
        except Exception as e:
            print("Failed to parse LLM response.")
            print("Raw content:\n", content)
            raise e

        verdict = llm_response.get("verdict", "").strip().upper()
        if verbose:
            print(f"Verdict: {verdict}")

        if verdict == "TERMINATE":
            selected_indexes = parse_selected_indexes(llm_response.get("selected_indexes", ""))
            selected_videos = [results[i] for i in selected_indexes if 0 <= i < len(results)]
            return selected_videos

        # Otherwise refine query
        query = llm_response.get("query", "").strip()
        if not query:
            print("LLM returned CONTINUE but no refined query. Ending.")
            return []

    print("Max turns reached. No videos selected.")
    return []

