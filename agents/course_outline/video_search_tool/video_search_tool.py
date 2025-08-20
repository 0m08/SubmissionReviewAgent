import os
from services.sheets_service import (
    get_sheet_data_and_df,
    save_to_sheet,
)
from services.smart_progress_bar import SmartProgressBar
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import os
from langchain.vectorstores import Chroma
from tqdm import tqdm
from services.embedding_service import get_embedding_model
from agents.vector_store_image_search.create_vectorstore import upload_folder_to_drive, download_folder_from_drive


    

