import argparse
import importlib
import sys
import os
import streamlit as st
import json
import base64
from agent_ui_template import log_completed_step, load_completed_steps
from dotenv import load_dotenv


# Google Sheets / Drive
import gspread
from pydrive2.drive import GoogleDrive
from services.drive_service import login_with_service_account


# Set up argument parser
parser = argparse.ArgumentParser(description="Run agent pipeline in CLI mode.")
parser.add_argument("--sheet_link", required=True, help="Google Sheet URL")
parser.add_argument("--drive_folder_id", required=True, help="Google Drive folder ID")
parser.add_argument(
    "--agent_name",
    required=True,
    choices=[
        "course_outline",
        "research_notes",
        "slide_chunks",
        "graphics_definition",
        "assessment",
    ],
    help="Agent/pipeline to run",
)
args = parser.parse_args()

AGENT_DISPLAY_NAMES = {
    "course_outline": "Course Outline",
    "research_notes": "Research Notes",
    "slide_chunks": "Slide Chunks",
    "graphics_definition": "Graphics Definition",
    "assessment": "Assessment",
}
ui_agent_name = AGENT_DISPLAY_NAMES.get(args.agent_name, args.agent_name)


load_dotenv()

# Authenticate Google Sheets
try:
    key_bytes = base64.b64decode(os.environ["GDRIVE_SA_B64"])
    sa_json = key_bytes.decode()
    sa_dict = json.loads(sa_json)        


    sa_dict = json.loads(sa_json)
    gc = gspread.service_account_from_dict(sa_dict)
    sheet = gc.open_by_url(args.sheet_link)
    load_completed_steps(sheet, ui_agent_name)

except Exception as e:
    print(f"[ERROR] Failed to authenticate or open Google Sheet: {e}")
    sys.exit(1)

# Authenticate Google Drive
try:
    gauth = login_with_service_account(json_str=sa_json)
    gauth.ServiceAuth()
    drive = GoogleDrive(gauth)
except Exception as e:
    print(f"[ERROR] Failed to authenticate Google Drive: {e}")
    sys.exit(1)

# Map agent_name to pipeline module
AGENT_PIPELINES = {
    "course_outline": "course_outline",
    "research_notes": "research_notes",
    "slide_chunks": "slide_chunks",
    "graphics_definition": "graphics_definition",
    "assessment": "assessment",
}
pipeline_module_name = AGENT_PIPELINES.get(args.agent_name)
if not pipeline_module_name:
    print(f"[ERROR] Unknown agent: {args.agent_name}")
    sys.exit(1)

try:
    pipeline_module = importlib.import_module(pipeline_module_name)
except ImportError as e:
    print(f"[ERROR] Could not import pipeline module '{pipeline_module_name}': {e}")
    sys.exit(1)

# Prepare pipeline sections
pipeline_sections = getattr(pipeline_module, "pipeline_sections", None)
if pipeline_sections is None:
    print(f"[ERROR] Pipeline sections not found in module '{pipeline_module_name}'.")
    sys.exit(1)

# Set up session state
session_state = {
    "sheet": sheet,
    "agent_name": ui_agent_name,
    "drive": drive,
    "skip_manual_step": True,
    "root_folder_id": args.drive_folder_id,
    "gc": gc,
}

# Extract course info
try:
    info_ws = sheet.worksheet("Course info")
    info_rows = info_ws.get_all_records()
    if info_rows:
        row0 = info_rows[0]
        session_state["course_name"] = row0.get("Course Name", "")
        session_state["target_audience"] = row0.get("Target Audience & Industry", "")
        session_state["course_background"] = row0.get("Course Background", "")
        session_state["course_objective_guidelines"] = row0.get("Course Objective Guidelines", "")
except Exception as e:
    print(f"[WARNING] Could not extract course info: {e}")

# Helper: Run all automated steps
def run_all_automated_steps_for_cli(sections, state):
    progress = True
    while progress:
        progress = False
        for sec in sections:
            for step in sec["steps"]:
                step_key = f"{step['name']}_done"
                if state.get(step_key):
                    continue
                if not all(state.get(f"{d}_done") for d in step.get("depends_on", [])):
                    continue
                if step.get("is_manual_step", False) or "instructions" in step:
                    if not state.get("skip_manual_step"):
                        print(f"[INFO] Paused at manual step: {step['name']}")
                        continue
                kwargs = {a: state.get(k, k) for a, k in step.get("args", {}).items()}
                print(f"[START] {step['name']}")
                try:
                    step["func"](**kwargs)
                    state[step_key] = True
                    log_completed_step(state["sheet"], state["agent_name"], step["name"])
                    print(f"[DONE] {step['name']}")
                except Exception as err:
                    print(f"[ERROR] Step '{step['name']}' failed: {err}")
                    sys.exit(1)
                progress = True
    print("[SUCCESS] All steps completed.")

# Run the pipeline
if args.agent_name == 'graphics_definition':
    from agents.graphics_definition.define_graphics.run_generate_revise_graphics_definition import ensure_reference_description_column
    # Determine worksheet name 
    worksheet_name = 'Slide Chunks'
    ensure_reference_description_column(sheet, worksheet_name)

for k, v in session_state.items():
    st.session_state[k] = v

run_all_automated_steps_for_cli(pipeline_sections, session_state)
