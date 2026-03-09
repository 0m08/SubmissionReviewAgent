import time
import traceback
import streamlit as st
from agents.quality_compliance_scoring.quality_scoring import run_update_quality_scores
from services.activity_tracking_service import track_tool_action
import gspread


def _get_gspread_client():
    gc_client = st.session_state.get("gspread_client") or st.session_state.get("gc")
    if gc_client:
        st.session_state.setdefault("gspread_client", gc_client)
        return gc_client

    creds = st.session_state.get("google_credentials")
    if not creds:
        raise RuntimeError("Google OAuth credentials missing. Please sign in from the agent UI.")

    gc_client = gspread.authorize(creds)
    st.session_state["gspread_client"] = gc_client
    return gc_client


try:
    gc = _get_gspread_client()
except RuntimeError as auth_err:
    gc = None
    st.error(str(auth_err))
    st.stop()

# === Helper Function ===
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
    1. Enter the **Course Folder ID** (the Google Drive folder containing all course files).
    2. Paste the **Checklist Sheet Link** of the reviewed course checklist.
    3. Make sure the sheet contains the reviewed stages with properly labeled columns.
    """
)

# --- Step 1: Input for Course Folder ID ---
course_folder_id = st.text_input("Enter Course Folder ID")

# --- Step 2: Input for Google Spreadsheet URL ---
spreadsheet_url = st.text_input("Enter Google Spreadsheet URL")

# --- Step 3: Process Inputs ---
if course_folder_id and spreadsheet_url:
    spreadsheet_id = extract_spreadsheet_id(spreadsheet_url)
    if not spreadsheet_id:
        st.error("Invalid Google Sheets URL. Please check the format.")
    else:
        try:
            # Open the spreadsheet
            spreadsheet = gc.open_by_key(spreadsheet_id)
            st.success(f"Loaded Spreadsheet: {spreadsheet.title}")

            # Run scoring logic
            if st.button("Run Quality Scoring"):
                _t = time.perf_counter()
                try:
                    run_update_quality_scores(spreadsheet, course_folder_id, spreadsheet_url)
                    track_tool_action("Quality Compliance Scoring", "run_quality_scoring", run_mode="tool", duration_seconds=time.perf_counter() - _t, course_name="", sheet_link=spreadsheet_url)
                    st.success("Task Logs updated successfully!")
                except Exception as e:
                    track_tool_action("Quality Compliance Scoring", "run_quality_scoring", run_mode="tool", error_message=str(e)[:500], course_name="", sheet_link=spreadsheet_url)
                    st.error(f"Error running quality scoring: {e}")
                    st.text(traceback.format_exc())

        except Exception as e:
            st.error(f"Error loading spreadsheet: {e}")

elif not course_folder_id and spreadsheet_url:
    st.warning(" Please enter the Course Folder ID before continuing.")
