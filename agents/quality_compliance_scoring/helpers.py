
import os
import re
import json
import shutil
import tempfile
import pandas as pd
from tqdm import tqdm
import streamlit as st
from datetime import datetime
from modules.chain import Chain
from typing import List, Optional
from collections import defaultdict


def extract_course_name(title: str) -> str:
    """
    Extracts the course name from a given title string.
    The course name is everything after the first colon.
    """
    parts = title.split(":", 1)
    return parts[1].strip() if len(parts) > 1 else title.strip()



def get_stage_sheets(spreadsheet):
    """
    Retrieves all stage sheets from the provided spreadsheet.
    :param spreadsheet: The spreadsheet object containing multiple worksheets.
    :return: A list of stage sheets that contain 'Checklist Criteria' and 'Task' in their headers.
    """
    stage_sheets = []
    for sheet in spreadsheet.worksheets():
        headers = sheet.row_values(1)
        if 'Checklist Criteria' in headers and 'Task' in headers:
            stage_sheets.append(sheet)
    print(f"Found {len(stage_sheets)} stage sheets.")
    return stage_sheets


def sanitize_filename(name: str) -> str:
    """Remove or replace invalid characters for Windows file/folder names."""
    return re.sub(r'[<>:"/\\|?*]', '_', name)


def derive_category_name(header: str) -> str:
    """Create a readable category label from any comments column header."""
    cleaned = re.sub(r'(?i)comments', '', header)
    cleaned = cleaned.replace('\n', ' ')
    cleaned = re.sub(r'\s+', ' ', cleaned).strip(" :-")
    return cleaned or "Comments"
