from agents.research_notes.vector_store import load_vector_db_with_pydrive
from services.embedding_service import get_embedding_model
import os
import pickle
from langchain.retrievers import EnsembleRetriever
from langchain_cohere import CohereRerank
from langchain.retrievers import ContextualCompressionRetriever
from services.helper_functions import get_short_name
from langchain_community.retrievers import BM25Retriever
from langchain_exa import ExaSearchRetriever


def load_vector_db_retriever(course_name, course_drive_folder_id, drive, sheet):
    """
    Load the vector database retriever.
    :param course_name: The name of the course.
    :param course_drive_folder_id: The ID of the folder containing the vector database.
    :param drive: Authenticated GoogleDrive instance (PyDrive2).
    :param sheet: The Google Sheets object.
    :return: Loaded vector database retriever object.
    """

    embedding_model = get_embedding_model()

    chroma_db, all_doc_chunk_list = load_vector_db_with_pydrive(
        course_name = course_name,
        root_folder_id = course_drive_folder_id,
        embedding_function = embedding_model,
        sheet = sheet,
        drive = drive
    )

    vector_db_retriever = chroma_db.as_retriever(
        search_kwargs = {
            "k": 20,
        }
    )

    return vector_db_retriever, all_doc_chunk_list


def load_bm25_retriever_with_pydrive(root_folder_id: str, drive, all_doc_chunk_list):
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
    local_pickle_path = "/tmp/bm25_retriever.pkl"

    # Check if already downloaded
    if os.path.exists(local_pickle_path):
        print("BM25 retriever already exists locally. Skipping download.")
    else:
        print("BM25 retriever not found locally. Downloading from Google Drive...")

        # Locate 'Pickle files' folder
        query_pickle_folder = (
            f"title='Pickle files' and '{root_folder_id}' in parents "
            f"and mimeType='application/vnd.google-apps.folder'"
        )
        pickle_folders = drive.ListFile({'q': query_pickle_folder}).GetList()

        if not pickle_folders:
            print(f"No folder named 'Pickle files' found in folder ID {root_folder_id}.")
            file_metadata = {
                'title': 'Pickle files',
                'parents': [{'id': root_folder_id}],
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

            # Create the bm_25 retriever
            bm_25_retriever = BM25Retriever.from_documents(all_doc_chunk_list, k = 20, )

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
            return bm_25_retriever

        else:
            print("bm25_research_db.pkl found under 'Pickle files'")

            bm25_file = bm25_files[0]
            bm25_file.GetContentFile(local_pickle_path)

    # Load the retriever from the pickle file
    with open(local_pickle_path, 'rb') as file:
        bm_25_retriever = pickle.load(file)
    print("Successfully loaded the BM25 retriever from Google Drive.")

    return bm_25_retriever


# initialize the ensemble retriever
def get_ensemble_retriever(course_name, root_folder_id, drive, sheet, retriever_1_weight = 0.5, retriever_2_weight = 0.5):
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
    vector_db_retriever, all_doc_chunk_list = load_vector_db_retriever(course_name, root_folder_id, drive, sheet)

    bm_25_retriever = load_bm25_retriever_with_pydrive(root_folder_id, drive, all_doc_chunk_list)

    ensemble_retriever = EnsembleRetriever(
        retrievers = [bm_25_retriever, vector_db_retriever],
        weights = [retriever_1_weight, retriever_2_weight],
    )
    return ensemble_retriever



def get_compression_retriever(course_name, root_folder_id, drive, sheet, retriever_1_weight = 0.5, retriever_2_weight = 0.5):
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

    ensemble_retriever = get_ensemble_retriever(course_name, root_folder_id, drive, sheet, retriever_1_weight, retriever_2_weight)

    compressor = CohereRerank(
        model="rerank-english-v2.0",
        top_n=15,
    )

    compression_retriever = ContextualCompressionRetriever(
        base_compressor = compressor, base_retriever = ensemble_retriever
    )

    return compression_retriever


def get_web_search_retriever():
    """
    Get the web search retriever.
    :return: Web search retriever object.
    """
    web_search_retriever = ExaSearchRetriever(
        k = 5,
    )
    return web_search_retriever

