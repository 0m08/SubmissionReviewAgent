import os
import json
import base64
import gspread
import streamlit as st
from dotenv import load_dotenv
from pydrive2.drive import GoogleDrive

from services.drive_service import login_with_service_account
from agents.model_3d_search.ui_3d_search import render_3d_model_search_tab

load_dotenv()

# Auth & Service Account initialization
if "drive" in st.session_state and "gc" in st.session_state:
    drive = st.session_state["drive"]
    gc = st.session_state["gc"]
else:
    if "GDRIVE_SA_B64" in os.environ:
        key_bytes = base64.b64decode(os.environ["GDRIVE_SA_B64"])
        sa_json = key_bytes.decode()
        sa_dict = json.loads(sa_json)

        gauth = login_with_service_account(json_str=sa_json)
        gauth.ServiceAuth()
        drive = GoogleDrive(gauth)
        gc = gspread.service_account_from_dict(sa_dict)

        st.session_state["drive"] = drive
        st.session_state["gc"] = gc
    else:
        drive = None
        gc = None

# Render 3D Model Search Tool UI
render_3d_model_search_tab(drive=drive)
