import argparse
import importlib
import sys
import os
import signal
import inspect
from datetime import datetime, timedelta
import streamlit as st
import json
import base64
from agent_ui_template import log_completed_step, load_completed_steps
from dotenv import load_dotenv
from services.email_service import send_notification_email, is_email_configured
from services.sheets_service import get_sheet_data_and_df
from services.background_job_status_service import append_background_job_status

# Google Sheets / Drive
import gspread
from pydrive2.drive import GoogleDrive
from services.drive_service import login_with_service_account, try_build_user_drive_for_background_jobs


# Set up argument parser
parser = argparse.ArgumentParser(description="Run agent pipeline in CLI mode.")
parser.add_argument("--sheet_link", required=True, help="Google Sheet URL")
parser.add_argument("--drive_folder_id", default="", help="Google Drive folder ID")
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
        "human_feedback_review_revise",
        "web_image_regeneration_bg",
    ],
    help="Agent/pipeline to run",
)
parser.add_argument("--user_email", default="", help="User email for job notifications (or set USER_EMAIL env)")
parser.add_argument("--toggles", default="", help="JSON string of UI toggle values")
parser.add_argument("--human_feedback_column", default="human_feedback")
parser.add_argument("--human_feedback_status_column", default="human_feedback_status")
parser.add_argument("--human_feedback_revision_tracking_column", default="human_feedback_revision_tracking")
parser.add_argument("--human_review_actions_column", default="human_review_actions")
parser.add_argument("--llm", default="gemini_3_flash_thinking")
parser.add_argument("--max_workers", type=int, default=50)
parser.add_argument("--use_only_drive_and_hvac", default="false")
parser.add_argument("--source_tab", default="Slide Chunks")
parser.add_argument("--regen_input_column", default="")
parser.add_argument("--regen_output_column", default="final_graphics_definition")
parser.add_argument("--regen_output_folder_name", default="Web Image Regeneration")
parser.add_argument("--regen_write_final_graphics", default="false")
parser.add_argument("--regen_skip_filled_rows", default="false")
parser.add_argument("--run_id", default="", help="Launcher-generated background run identifier")
args = parser.parse_args()

AGENT_DISPLAY_NAMES = {
    "course_outline": "Course Outline",
    "research_notes": "Research Notes",
    "slide_chunks": "Slide Chunks",
    "graphics_definition": "Graphics Definition",
    "graphics_definition_v2": "Graphics Definition V2",
    "assessment": "Assessment",
    "human_feedback_review_revise": "Human Feedback Review & Revise",
    "web_image_regeneration_bg": "Web Images to AI Images Regeneration",
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

drive_for_session = drive
oauth_drive = try_build_user_drive_for_background_jobs()
if oauth_drive is not None:
    drive_for_session = oauth_drive
    print(
        "[INFO] Background job: using OAuth user Google Drive. Sheets access remains service-account."
    )
else:
    print(
        "[INFO] Background job: using service-account Google Drive. "
        "Use 'run in background' from the app while logged in with Google (or set GOOGLE_OAUTH_REFRESH_TOKEN "
        "on the job) to match in-browser Drive uploads."
    )

# Set up session state
session_state = {
    "sheet": sheet,
    "agent_name": ui_agent_name,
    "drive": drive_for_session,
    "skip_manual_step": True,
    "root_folder_id": args.drive_folder_id,
    "gc": gc,
    "use_only_drive_and_hvac": False,
    "llm_model": args.llm,
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

        _RESEARCH_ALIASES = {"video": "video", "web": "web", "deep": "deep"}
        _research_raw = str(row0.get("Research Sources", "") or "").strip()
        if _research_raw:
            import re as _re
            _tokens = _re.split(r"[\n,]", _research_raw)
            _enabled = {_RESEARCH_ALIASES[t.strip().lower()] for t in _tokens if t.strip().lower() in _RESEARCH_ALIASES}
        else:
            _enabled = {"video", "web", "deep"}
        session_state["video_research_enabled"] = "video" in _enabled
        session_state["web_research_enabled"]   = "web"   in _enabled
        session_state["deep_research_enabled"]  = "deep"  in _enabled
        print(f"[INFO] Research sources: {_enabled}")
except Exception as e:
    print(f"[WARNING] Could not extract course info: {e}")

# Populate st.session_state before importing pipeline module
for k, v in session_state.items():
    st.session_state[k] = v
# Map agent_name to pipeline module
AGENT_PIPELINES = {
    "course_outline": "course_outline",
    "research_notes": "research_notes",
    "slide_chunks": "slide_chunks",
    "graphics_definition": "graphics_definition",
    "graphics_definition_v2": "graphics_definition_v2",
    "assessment": "assessment",
}
pipeline_sections = None
if args.agent_name not in ("human_feedback_review_revise", "web_image_regeneration_bg"):
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


def _format_time_utc_and_ist():
    """Return a string like '2026-02-23 13:18 UTC (18:48 IST)'."""
    utc_now = datetime.utcnow()
    ist_now = utc_now + timedelta(hours=5, minutes=30)
    return f"{utc_now.strftime('%Y-%m-%d %H:%M')} UTC ({ist_now.strftime('%Y-%m-%d %H:%M')} IST)"


def _to_bool(v):
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")


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


def _step_is_hidden_cli(step, state):
    if state.get("outline_finalized", False) and step.get("hide_if_final_outline", False):
        return True
    if not state.get("video_research_enabled", True) and step.get("hide_if_video_disabled", False):
        return True
    if not state.get("graphics_v2_web_fallback_enabled", True) and step.get("hide_if_web_disabled", False):
        return True
    return False


# Helper: Run all automated steps
def run_all_automated_steps_for_cli(sections, state):
    progress = True
    while progress:
        progress = False
        for sec in sections:
            for step in sec["steps"]:
                # Skip step completely if it is hidden/disabled
                if _step_is_hidden_cli(step, state):
                    continue

                step_key = f"{step['name']}_done"
                if state.get(step_key):
                    continue

                if not all(
                    state.get(f"{d}_done", False)
                    or any(
                        d == s["name"] and _step_is_hidden_cli(s, state)
                        for sec_all in sections
                        for s in sec_all["steps"]
                    )
                    for d in step.get("depends_on", [])
                ):
                    continue
                if step.get("is_manual_step", False) or "instructions" in step:
                    if not state.get("skip_manual_step"):
                        print(f"[INFO] Paused at manual step: {step['name']}")
                        continue
                kwargs = {a: state.get(k, k) for a, k in step.get("args", {}).items()}
                fn_params = inspect.signature(step["func"]).parameters
                if "selected_topics" in fn_params and "selected_topics" not in kwargs:
                    kwargs["selected_topics"] = state.get("selected_topics", [])
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

if args.agent_name == "human_feedback_review_revise":
    from agents.graphics_definition_v2.review_agent.human_feedback_based_review_and_revise import (
        run_human_feedback_review_revise_for_all_rows,
    )

    def _run_target():
        run_human_feedback_review_revise_for_all_rows(
            sheet=sheet,
            llm=args.llm,
            max_workers=args.max_workers,
            use_only_drive_and_hvac=_to_bool(args.use_only_drive_and_hvac),
            human_feedback_column=args.human_feedback_column,
            human_feedback_status_column=args.human_feedback_status_column,
            human_feedback_revision_tracking_column=args.human_feedback_revision_tracking_column,
            human_review_actions_column=args.human_review_actions_column,
        )
elif args.agent_name == "web_image_regeneration_bg":
    from agents.graphics_asset_creation.automated.automated_voiceover_reviewer import run_automation

    def _run_target():
        input_col = (args.regen_input_column or "").strip()
        if not input_col:
            raise ValueError("Missing --regen_input_column for web_image_regeneration_bg.")
        output_col = (args.regen_output_column or "final_graphics_definition").strip()
        run_automation(
            sheet_url=args.sheet_link,
            source_tab=(args.source_tab or "Slide Chunks").strip(),
            output_folder_name=(args.regen_output_folder_name or "Web Image Regeneration").strip(),
            gc=gc,
            drive=drive_for_session,
            progress_callback=None,
            skip_filled_rows=_to_bool(args.regen_skip_filled_rows),
            input_column_name=input_col,
            output_column_name=output_col,
            write_final_graphics=_to_bool(args.regen_write_final_graphics),
        )
else:
    def _run_target():
        run_all_automated_steps_for_cli(pipeline_sections, session_state)

for k, v in session_state.items():
    st.session_state[k] = v

# Notifications: each message is standalone (no In-Reply-To / thread with other mails for this run).
user_email = (args.user_email or os.environ.get("USER_EMAIL", "")).strip()
course_name = session_state.get("course_name", "Unknown course")
notify = False
run_id = (args.run_id or "").strip()
terminal_notification_sent = False
if not user_email:
    print("[INFO] No USER_EMAIL set; skipping notification emails.")
elif not is_email_configured():
    print("[INFO] SMTP_USER or SMTP_APP_PASSWORD not set in job environment; skipping notification emails.")
else:
    notify = True

running_subject = f"Course generation: {ui_agent_name} AI Agent – Running"
sheet_link = (args.sheet_link or "").strip()
sheet_link_html = (
    f'<a href="{sheet_link}" target="_blank" rel="noopener noreferrer">Link</a>'
    if sheet_link else "N/A"
)
sheet_link_plain = f"Link ({sheet_link})" if sheet_link else "N/A"

if notify:
    count, count_label = _get_agent_work_count(sheet, args.agent_name)
    started_at = _format_time_utc_and_ist()
    subject = running_subject
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
        f"Input Sheet:  {sheet_link_plain}\n\n"
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
        f"<b>Started At:</b> {started_at}<br>"
        f"<b>Input Sheet:</b> {sheet_link_html}</p>"
        f"{work_scope_html}"
        f"<hr><p><b>NEXT STEPS</b></p><hr>"
        f"<p>You may close the browser tab or window where you started this agent, or even shut down your computer. The agent is running in the cloud and will continue on its own. You will receive a follow-up email when it completes successfully or if an error occurs—no need to keep the app open.</p>"
        f"<p>Thank you for using Course Generation AI Agents.</p>"
    )
    _started_mid = send_notification_email(to_email=user_email, subject=subject, body_plain=body, body_html=body_html)
    if _started_mid:
        print(f"[INFO] Sent 'agent is running' notification to {user_email}")

try:
    append_background_job_status(
        gc=gc,
        run_id=run_id,
        user_email=user_email,
        agent_name=ui_agent_name,
        status="running",
        sheet_link=sheet_link,
        message="Background job started",
    )
except Exception as _track_err:
    print(f"[WARNING] Could not write background status 'running': {_track_err}")


def _send_stopped_email_and_track(reason: str):
    global terminal_notification_sent
    if terminal_notification_sent:
        return
    terminal_notification_sent = True
    stopped_at = _format_time_utc_and_ist()
    if notify:
        subject = f"Course generation: {ui_agent_name} AI Agent – Stopped"
        body = (
            f"Your {ui_agent_name} AI agent stopped before completion.\n\n"
            f"————————————————————————————\n"
            f"AGENT DETAILS\n"
            f"————————————————————————————\n"
            f"Agent:      {ui_agent_name}\n"
            f"Course:     {course_name}\n"
            f"Stopped at: {stopped_at}\n"
            f"Status:     Stopped\n\n"
            f"Input Sheet:  {sheet_link_plain}\n\n"
            f"Reason: {reason}\n\n"
            f"————————————————————————————\n"
            f"RECOMMENDED ACTIONS\n"
            f"————————————————————————————\n"
            f"Please re-run the agent again by clicking the 'Run the Agent in Background' button in the app.\n"
        )
        body_html = (
            f"<p>Your {ui_agent_name} AI agent stopped before completion.</p>"
            f"<hr><p><b>AGENT DETAILS</b></p><hr>"
            f"<p><b>Agent:</b> {ui_agent_name}<br>"
            f"<b>Course:</b> {course_name}<br>"
            f"<b>Stopped at:</b> {stopped_at}<br>"
            f"<b>Status:</b> Stopped<br>"
            f"<b>Input Sheet:</b> {sheet_link_html}<br>"
            f"<b>Reason:</b> {reason}</p>"
            f"<hr><p><b>RECOMMENDED ACTIONS</b></p><hr>"
            f"<p>Please re-run the agent again by clicking the 'Run the Agent in Background' button in the app.</p>"
        )
        send_notification_email(
            to_email=user_email,
            subject=subject,
            body_plain=body,
            body_html=body_html,
        )
    try:
        append_background_job_status(
            gc=gc,
            run_id=run_id,
            user_email=user_email,
            agent_name=ui_agent_name,
            status="stopped",
            sheet_link=sheet_link,
            message=reason,
        )
    except Exception as _track_err:
        print(f"[WARNING] Could not write background status 'stopped': {_track_err}")


def _handle_stop_signal(signum, _frame):
    signame = "SIGTERM" if signum == signal.SIGTERM else "SIGINT"
    print(f"[WARNING] Received {signame}; marking run as stopped.")
    _send_stopped_email_and_track(f"Received {signame} in background job")
    sys.exit(1)


signal.signal(signal.SIGTERM, _handle_stop_signal)
signal.signal(signal.SIGINT, _handle_stop_signal)

try:
    _run_target()
    if notify:
        completed_at = _format_time_utc_and_ist()
        subject = f"Course generation: {ui_agent_name} AI Agent – Completed"
        body = (
            f"Your {ui_agent_name} AI agent has completed successfully in the background.\n\n"
            f"————————————————————————————\n"
            f"AGENT DETAILS\n"
            f"————————————————————————————\n"
            f"Agent Name:   {ui_agent_name}\n"
            f"Course Name:  {course_name}\n"
            f"Completed At: {completed_at}\n"
            f"Input Sheet:  {sheet_link_plain}\n"
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
            f"<b>Completed At:</b> {completed_at}<br>"
            f"<b>Input Sheet:</b> {sheet_link_html}</p>"
            f"<hr><p><b>NEXT STEPS</b></p><hr>"
            f"<p>You can return to the Google Sheet that you are using as input for this agent, to review the outputs and continue with the next steps in your course workflow.</p>"
            f"<p>Thank you for using Course Generation AI Agents.</p>"
        )
        send_notification_email(
            to_email=user_email,
            subject=subject,
            body_plain=body,
            body_html=body_html,
        )
        print(f"[INFO] Sent completion notification to {user_email}")
    terminal_notification_sent = True
    try:
        append_background_job_status(
            gc=gc,
            run_id=run_id,
            user_email=user_email,
            agent_name=ui_agent_name,
            status="completed",
            sheet_link=sheet_link,
            message="Background job completed successfully",
        )
    except Exception as _track_err:
        print(f"[WARNING] Could not write background status 'completed': {_track_err}")
except Exception as err:
    if notify:
        failed_at = _format_time_utc_and_ist()
        err_escaped = str(err).replace("<", "&lt;").replace(">", "&gt;")
        subject = f"Course generation: {ui_agent_name} AI Agent – Failed"
        body = (
            f"Your {ui_agent_name} AI agent encountered an error and did not complete successfully.\n\n"
            f"————————————————————————————\n"
            f"AGENT DETAILS\n"
            f"————————————————————————————\n"
            f"Agent:      {ui_agent_name}\n"
            f"Course:     {course_name}\n"
            f"Failed at:  {failed_at}\n"
            f"Status:     Failed\n\n"
            f"Input Sheet:  {sheet_link_plain}\n\n"
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
            f"<b>Status:</b> Failed<br>"
            f"<b>Input Sheet:</b> {sheet_link_html}</p>"
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
            body_html=body_html,
        )
        print(f"[INFO] Sent error notification to {user_email}")
    terminal_notification_sent = True
    try:
        append_background_job_status(
            gc=gc,
            run_id=run_id,
            user_email=user_email,
            agent_name=ui_agent_name,
            status="failed",
            sheet_link=sheet_link,
            message=str(err),
        )
    except Exception as _track_err:
        print(f"[WARNING] Could not write background status 'failed': {_track_err}")
    sys.exit(1)