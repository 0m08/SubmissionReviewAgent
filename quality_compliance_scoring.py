import streamlit as st
from agents.quality_compliance_scoring.quality_scoring import run_update_quality_scores
import streamlit as st


import os
import re
import json
import base64
import gspread
import streamlit as st
from dotenv import load_dotenv
from pydrive2.drive import GoogleDrive
from services.drive_service import login_with_service_account

load_dotenv()
key_bytes = base64.b64decode(os.environ["GDRIVE_SA_B64"])
sa_json = key_bytes.decode()
sa_dict = json.loads(sa_json)
gauth = login_with_service_account(json_str=sa_json)
gauth.ServiceAuth()
drive = GoogleDrive(gauth)
gc = gspread.service_account_from_dict(sa_dict)


# Extract spreadsheet ID from URL

def extract_spreadsheet_id(url):
    try:
        return url.split("/d/")[1].split("/")[0]
    except IndexError:
        return None


# === Streamlit App UI ===
st.title("Quality Compliance Scoring Automation")

st.info(
    """
    ### 📝 Instructions
    1. Please paste the **Checklist Sheet Link** of the course checklist reviewed below.  
    2. Make sure the sheet contains the reviewed stages with properly labeled columns.
    """
)

spreadsheet_url = st.text_input("Enter Google Spreadsheet URL")

if spreadsheet_url:
    spreadsheet_id = extract_spreadsheet_id(spreadsheet_url)
    if not spreadsheet_id:
        st.error("Invalid Google Sheets URL. Please check the format.")
    else:
        try:
            spreadsheet = gc.open_by_key(spreadsheet_id)
            st.success(f"Loaded Spreadsheet: {spreadsheet.title}")

            if st.button("Run Quality Scoring"):
                run_update_quality_scores(spreadsheet)
                st.success("Task Logs updated successfully!")

        except Exception as e:
            st.error(f"Error loading spreadsheet: {e}")