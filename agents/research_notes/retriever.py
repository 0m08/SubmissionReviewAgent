from agents.research_notes.vector_store import load_vector_db_with_pydrive
from services.drive_service import drive
from services.embedding_service import get_embedding_model
import os
import pickle
from langchain.retrievers import EnsembleRetriever
from langchain_cohere import CohereRerank
from langchain.retrievers import ContextualCompressionRetriever
from services.helper_functions import get_short_name

def load_vector_db_retriever(course_name, root_folder_id, drive):
    """
    Load the vector database retriever.
    :param course_name: The name of the course.
    :param root_folder_id: The ID of the folder containing the vector database.
    :param drive: Authenticated GoogleDrive instance (PyDrive2).
    :return: Loaded vector database retriever object.
    """

    bge_large = get_embedding_model()

    chroma_db = load_vector_db_with_pydrive(
        course_name = course_name,
        root_folder_id = root_folder_id,
        embedding_function = bge_large,
        drive = drive
    )

    vector_db_retriever = chroma_db.as_retriever(
        search_kwargs = {
            "k": 20,
        }
    )

    return vector_db_retriever


def load_bm25_retriever_with_pydrive(root_folder_id: str, drive, course_name):
    """
    Load the BM25 retriever from a pickle file stored in Google Drive.

    Optimization:
    - If the file is already downloaded, skip the download process.
    - If missing, download the file from Google Drive.

    :param root_folder_id: The ID of the root folder containing 'Pickle files'.
    :param drive: Authenticated GoogleDrive instance (PyDrive2).
    :param course_name: The name of the course (used for the local path name).
    :return: Loaded BM25 retriever object.
    """

    short_course_name = get_short_name(course_name)

    local_pickle_path = f"/tmp/{short_course_name}_bm25_retriever.pkl"

    # Locate 'Pickle files' folder
    query_pickle_folder = (
        f"title='Pickle files' and '{root_folder_id}' in parents "
        f"and mimeType='application/vnd.google-apps.folder'"
    )
    pickle_folders = drive.ListFile({'q': query_pickle_folder}).GetList()
    if not pickle_folders:
        raise FileNotFoundError(f"No folder named 'Pickle files' found in folder ID {root_folder_id}.")

    pickle_folder = pickle_folders[0]
    pickle_folder_id = pickle_folder['id']

    # Locate 'bm25_research_db.pkl' inside 'Pickle files'
    query_bm25 = (
        f"title='bm25_research_db.pkl' and '{pickle_folder_id}' in parents"
    )
    bm25_files = drive.ListFile({'q': query_bm25}).GetList()
    if not bm25_files:
        raise FileNotFoundError("bm25_research_db.pkl not found under 'Pickle files'.")

    bm25_file = bm25_files[0]

    # Check if already downloaded
    if not os.path.exists(local_pickle_path):
        print("BM25 retriever not found locally. Downloading from Google Drive...")
        bm25_file.GetContentFile(local_pickle_path)
    else:
        print("BM25 retriever already exists locally. Skipping download.")

    # Load the retriever from the pickle file
    with open(local_pickle_path, 'rb') as file:
        bm_25_retriever = pickle.load(file)
    print("Successfully loaded the BM25 retriever from Google Drive.")

    return bm_25_retriever


# initialize the ensemble retriever
def get_ensemble_retriever(course_name, root_folder_id, drive, retriever_1_weight = 0.5, retriever_2_weight = 0.5):
    """
    Get the ensemble retriever.

    :param course_name: The name of the course.
    :param root_folder_id: The ID of the root folder containing 'Pickle files' and 'Vectorstore files'.
    :param drive: Authenticated GoogleDrive instance (PyDrive2).
    :param retriever_1_weight: The weight of the BM25 retriever.
    :param retriever_2_weight: The weight of the vector database retriever.
    :return: Ensemble retriever object.
    """
    vector_db_retriever = load_vector_db_retriever(course_name, root_folder_id, drive)

    bm_25_retriever = load_bm25_retriever_with_pydrive(root_folder_id, drive, course_name)

    ensemble_retriever = EnsembleRetriever(
        retrievers = [bm_25_retriever, vector_db_retriever],
        weights = [retriever_1_weight, retriever_2_weight],
    )
    return ensemble_retriever



def get_compression_retriever(course_name, root_folder_id, drive, retriever_1_weight = 0.5, retriever_2_weight = 0.5):
    """
    Get the compression retriever.
    
    :param: course_name: The name of the course.
    :param: root_folder_id: The ID of the root folder containing 'Pickle files' and 'Vectorstore files'.
    :param: drive: Authenticated GoogleDrive instance (PyDrive2).
    :param: retriever_1_weight: The weight of the BM25 retriever.
    :param: retriever_2_weight: The weight of the vector database retriever.
    :return: Compression retriever object.
    """

    ensemble_retriever = get_ensemble_retriever(course_name, root_folder_id, drive, retriever_1_weight, retriever_2_weight)

    compressor = CohereRerank(
        model="rerank-english-v2.0",
        top_n=15,
    )

    compression_retriever = ContextualCompressionRetriever(
        base_compressor = compressor, base_retriever = ensemble_retriever
    )

    return compression_retriever


# compression_retriever = get_compression_retriever(
#    course_name = course_name,
#    root_folder_id = "1r0GJBKsu_X-2sbFkbItpymUfdVOBxEXt",
#    drive = drive,
#    retriever_1_weight = 0.5,
#    retriever_2_weight = 0.5
# )