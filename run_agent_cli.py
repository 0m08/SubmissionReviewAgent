import argparse
import importlib
import sys
import os
from datetime import datetime, timedelta
import streamlit as st
import json
import base64
from agent_ui_template import log_completed_step, load_completed_steps
from dotenv import load_dotenv
from services.email_service import send_notification_email, is_email_configured
from services.sheets_service import get_sheet_data_and_df

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
        "graphics_definition_v2",
        "assessment",
    ],
    help="Agent/pipeline to run",
)
parser.add_argument("--user_email", default="", help="User email for job notifications (or set USER_EMAIL env)")
parser.add_argument("--toggles", default="", help="JSON string of UI toggle values")
args = parser.parse_args()

AGENT_DISPLAY_NAMES = {
    "course_outline": "Course Outline",
    "research_notes": "Research Notes",
    "slide_chunks": "Slide Chunks",
    "graphics_definition": "Graphics Definition",
    "graphics_definition_v2": "Graphics Definition V2",
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
    "graphics_definition_v2": "graphics_definition_v2",
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
    "use_only_drive_and_hvac": False,
}

# Copy completed-step flags from Agent logs (loaded into st.session_state) into session_state
# so run_all_automated_steps_for_cli actually skips them and does not re-run or re-log.
for key, value in st.session_state.items():
    if key.endswith("_done") and value is True:
        session_state[key] = value

# Merge UI toggle values forwarded from the Streamlit app (e.g. use_only_drive_and_hvac)
if args.toggles:
    try:
        toggle_values = json.loads(args.toggles)
        session_state.update(toggle_values)
        print(f"[INFO] Loaded UI toggles: {toggle_values}")
    except json.JSONDecodeError as e:
        print(f"[WARNING] Could not parse --toggles JSON: {e}")

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
        session_state["checklist_sheet_link"] = str(row0.get("Checklist Link") or "").strip()

        # Determine if outline is finalized
        outline_stage_raw = row0.get("Outline Stage")
        if isinstance(outline_stage_raw, str):
            session_state["outline_finalized"] = outline_stage_raw.strip().lower() == "final"
        else:
            session_state["outline_finalized"] = False
except Exception as e:
    print(f"[WARNING] Could not extract course info: {e}")


def _format_time_utc_and_ist():
    """Return a string like '2026-02-23 13:18 UTC (18:48 IST)'."""
    utc_now = datetime.utcnow()
    ist_now = utc_now + timedelta(hours=5, minutes=30)
    return f"{utc_now.strftime('%Y-%m-%d %H:%M')} UTC ({ist_now.strftime('%Y-%m-%d %H:%M')} IST)"


def _get_agent_work_count(sheet, agent_name):
    """Return (count, label) e.g. (42, 'rows') or (10, 'topics'), or (None, None)."""
    try:
        if agent_name == "graphics_definition_v2":
            _, df = get_sheet_data_and_df(sheet, "Slide Chunks")
            return len(df), "rows"
        if agent_name == "assessment":
            _, df = get_sheet_data_and_df(sheet, "Slide Chunks")
            return len(df), "topics"
        if agent_name == "research_notes":
            for ws_name in ("Final Outline", "Course Outline with LOs"):
                try:
                    _, df = get_sheet_data_and_df(sheet, ws_name)
                    if "Topic" in df.columns:
                        unique_topics = df["Topic"].dropna().astype(str).str.strip()
                        unique_topics = unique_topics[unique_topics != ""]
                        count = unique_topics.nunique()
                    else:
                        count = len(df)
                    return count, "topics"
                except Exception:
                    continue
            return 0, "topics"
    except Exception:
        pass
    return None, None


# Helper: Run all automated steps
def run_all_automated_steps_for_cli(sections, state):
    progress = True
    while progress:
        progress = False
        for sec in sections:
            for step in sec["steps"]:
                # Skip step completely if outline is finalized and this step is marked to hide
                if state.get("outline_finalized", False) and step.get("hide_if_final_outline", False):
                    continue

                step_key = f"{step['name']}_done"
                if state.get(step_key):
                    continue

                if not all(
                    state.get(f"{d}_done", False)
                    or (
                        state.get("outline_finalized", False)
                        and any(
                            d == s["name"] and s.get("hide_if_final_outline", False)
                            for sec_all in sections
                            for s in sec_all["steps"]
                        )
                    )
                    for d in step.get("depends_on", [])
                ):
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
                    raise
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

# Notifications: send "agent is running" only when Execute has started 
user_email = (args.user_email or os.environ.get("USER_EMAIL", "")).strip()
course_name = session_state.get("course_name", "Unknown course")
first_message_id = None
if not user_email:
    print("[INFO] No USER_EMAIL set; skipping notification emails.")
elif not is_email_configured():
    print("[INFO] SMTP_USER or SMTP_APP_PASSWORD not set in job environment; skipping notification emails.")
# Base subject used for threading completion emails
thread_subject_base = f"Course generation: {ui_agent_name} AI Agent – Running"

if user_email and is_email_configured():
    count, count_label = _get_agent_work_count(sheet, args.agent_name)
    started_at = _format_time_utc_and_ist()
    subject = thread_subject_base
    work_scope = ""
    if count is not None:
        work_scope = f"Work scope: {count} {count_label} to process.\n\n"
    body = (
        f"Your {ui_agent_name} AI agent has started and is now running in the background.\n\n"
        f"————————————————————————————\n"
        f"AGENT DETAILS\n"
        f"————————————————————————————\n"
        f"Agent Name:   {ui_agent_name}\n"
        f"Course Name:  {course_name}\n"
        f"Started At:   {started_at}\n\n"
        f"{work_scope}"
        f"————————————————————————————\n"
        f"NEXT STEPS\n"
        f"————————————————————————————\n"
        f"You may close the browser tab or window where you started this agent, or even shut down your computer. The agent is running in the cloud and will continue on its own. You will receive a follow-up email when it completes successfully or if an error occurs-no need to keep the app open.\n\n"
        f"Thank you for using Course Generation AI Agents."
    )
    work_scope_html = f"<p><b>Work scope:</b> {count} {count_label} to process.</p>" if count is not None else ""
    body_html = (
        f"<p>Your {ui_agent_name} AI agent has started and is now running in the background.</p>"
        f"<hr><p><b>AGENT DETAILS</b></p><hr>"
        f"<p><b>Agent Name:</b> {ui_agent_name}<br>"
        f"<b>Course Name:</b> {course_name}<br>"
        f"<b>Started At:</b> {started_at}</p>"
        f"{work_scope_html}"
        f"<hr><p><b>NEXT STEPS</b></p><hr>"
        f"<p>You may close the browser tab or window where you started this agent, or even shut down your computer. The agent is running in the cloud and will continue on its own. You will receive a follow-up email when it completes successfully or if an error occurs—no need to keep the app open.</p>"
        f"<p>Thank you for using Course Generation AI Agents.</p>"
    )
    first_message_id = send_notification_email(to_email=user_email, subject=subject, body_plain=body, body_html=body_html)
    if first_message_id:
        print(f"[INFO] Sent 'agent is running' notification to {user_email}")

try:
    run_all_automated_steps_for_cli(pipeline_sections, session_state)
    if user_email and is_email_configured() and first_message_id:
        completed_at = _format_time_utc_and_ist()
        subject = f"Re: {thread_subject_base}"
        body = (
            f"Your {ui_agent_name} AI agent has completed successfully in the background.\n\n"
            f"————————————————————————————\n"
            f"AGENT DETAILS\n"
            f"————————————————————————————\n"
            f"Agent Name:   {ui_agent_name}\n"
            f"Course Name:  {course_name}\n"
            f"Completed At: {completed_at}\n"
            f"————————————————————————————\n"
            f"NEXT STEPS\n"
            f"————————————————————————————\n"
            f"You can return to the Google Sheet that you are using as input for this agent, to review the outputs and continue with the next steps in your course workflow.\n\n"
            f"Thank you for using Course Generation AI Agents."
        )
        body_html = (
            f"<p>Your {ui_agent_name} AI agent has completed successfully in the background.</p>"
            f"<hr><p><b>AGENT DETAILS</b></p><hr>"
            f"<p><b>Agent Name:</b> {ui_agent_name}<br>"
            f"<b>Course Name:</b> {course_name}<br>"
            f"<b>Completed At:</b> {completed_at}</p>"
            f"<hr><p><b>NEXT STEPS</b></p><hr>"
            f"<p>You can return to the Google Sheet that you are using as input for this agent, to review the outputs and continue with the next steps in your course workflow.</p>"
            f"<p>Thank you for using Course Generation AI Agents.</p>"
        )
        send_notification_email(
            to_email=user_email,
            subject=subject,
            body_plain=body,
            reply_to_message_id=first_message_id,
            body_html=body_html,
        )
        print(f"[INFO] Sent completion notification to {user_email}")
except Exception as err:
    if user_email and is_email_configured() and first_message_id:
        failed_at = _format_time_utc_and_ist()
        err_escaped = str(err).replace("<", "&lt;").replace(">", "&gt;")
        subject = f"Course generation: {ui_agent_name} AI Agent – Error"
        body = (
            f"Your {ui_agent_name} AI agent encountered an error and did not complete successfully.\n\n"
            f"————————————————————————————\n"
            f"AGENT DETAILS\n"
            f"————————————————————————————\n"
            f"Agent:      {ui_agent_name}\n"
            f"Course:     {course_name}\n"
            f"Failed at:  {failed_at}\n"
            f"Status:     Error\n\n"
            f"————————————————————————————\n"
            f"ERROR MESSAGE\n"
            f"————————————————————————————\n"
            f"{err}\n\n"
            f"————————————————————————————\n"
            f"RECOMMENDED ACTIONS\n"
            f"————————————————————————————\n"
            f"Share this email with Dilip (dilip@skillcatapp.com) and he will help you troubleshoot the issue and next steps.\n\n"
            f"Thank you for using Course Generation AI Agents."
        )
        body_html = (
            f"<p>Your {ui_agent_name} AI agent encountered an error and did not complete successfully.</p>"
            f"<hr><p><b>AGENT DETAILS</b></p><hr>"
            f"<p><b>Agent:</b> {ui_agent_name}<br>"
            f"<b>Course:</b> {course_name}<br>"
            f"<b>Failed at:</b> {failed_at}<br>"
            f"<b>Status:</b> Error</p>"
            f"<hr><p><b>ERROR MESSAGE</b></p><hr>"
            f"<p>{err_escaped}</p>"
            f"<hr><p><b>RECOMMENDED ACTIONS</b></p><hr>"
            f"<p>Share this email with Dilip (dilip@skillcatapp.com) and he will help you troubleshoot the issue and next steps.</p>"
            f"<p>Thank you for using Course Generation AI Agents.</p>"
        )
        send_notification_email(
            to_email=user_email,
            subject=subject,
            body_plain=body,
            reply_to_message_id=first_message_id,
            body_html=body_html,
        )
        print(f"[INFO] Sent error notification to {user_email}")
    sys.exit(1)