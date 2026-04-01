import argparse
import os
import sys
import base64
import uuid
import subprocess
from lightning_sdk import Machine, Studio
from dotenv import load_dotenv

load_dotenv()

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

def main():
    parser = argparse.ArgumentParser(description="Launch a Lightning AI job for a single agent via SDK.")
    parser.add_argument('--sheet_link', required=True, help='Google Sheet URL')
    parser.add_argument('--drive_folder_id', default='', help='Google Drive folder ID')
    parser.add_argument('--agent_name', required=True, help='Name of the agent to run')
    parser.add_argument('--user_email', default='', help='User email for job notifications')
    parser.add_argument('--toggles', default='', help='JSON string of UI toggle values to forward')
    parser.add_argument('--human_feedback_column', default='', help='Round-specific human feedback column')
    parser.add_argument('--human_feedback_status_column', default='', help='Round-specific human feedback status column')
    parser.add_argument('--human_feedback_revision_tracking_column', default='', help='Round-specific tracking column')
    parser.add_argument('--human_review_actions_column', default='', help='Round-specific review actions column')
    parser.add_argument('--llm', default='', help='LLM name override')
    parser.add_argument('--max_workers', default='', help='Max workers override')
    parser.add_argument('--use_only_drive_and_hvac', default='', help='true/false override')
    parser.add_argument('--source_tab', default='', help='Worksheet tab name for regen background job')
    parser.add_argument('--regen_input_column', default='', help='Input graphics definition column for regen')
    parser.add_argument('--regen_output_column', default='', help='Output graphics definition column for regen')
    parser.add_argument('--regen_output_folder_name', default='', help='Drive output folder name for regen')
    parser.add_argument('--regen_write_final_graphics', default='', help='true/false for writing legacy final_graphics')
    parser.add_argument('--regen_skip_filled_rows', default='', help='true/false resume flag for regen')
    parser.add_argument('--machine', default='CPU', help='Lightning machine type (e.g., CPU, CPU_X_8). Defaults to CPU.')
    parser.add_argument('--google_oauth_refresh_token_b64', default='', help='Base64-encoded UTF-8 Google OAuth refresh token (browser session) for human-feedback Drive uploads on the job')
    args = parser.parse_args()

    # Use GDRIVE_SA_B64 directly if available, otherwise encode GDRIVE_SA_JSON
    gdrive_sa_b64 = os.environ.get("GDRIVE_SA_B64")
    if not gdrive_sa_b64:
        gdrive_sa_json = os.environ.get("GDRIVE_SA_JSON")
        if gdrive_sa_json:
            gdrive_sa_b64 = base64.b64encode(gdrive_sa_json.encode()).decode()
        else:
            gdrive_sa_b64 = ""

    # VERTEX_AI_SA_B64 is required for Graphics Definition V2 video search (multimodal embeddings)
    vertex_ai_sa_b64 = os.environ.get("VERTEX_AI_SA_B64", "")

    # # Main ones
    studio_name = "course-generation-agents"
    #studio_name= "current"
    teamspace = "Vision-model"
    user = "dilip"

    print(f"[INFO] Initializing Studio '{studio_name}' in teamspace '{teamspace}'...")
    studio = Studio(name=studio_name, teamspace=teamspace, user=user, create_ok=True)

    print("[INFO] Installing 'jobs' plugin...")
    studio.install_plugin("jobs")
    jobs_plugin = studio.installed_plugins["jobs"]

    agent = args.agent_name
    ui_agent_name = AGENT_DISPLAY_NAMES.get(agent, agent)
    run_id = uuid.uuid4().hex[:12]
    
    # Export env vars for the job: GDRIVE_SA_B64 (required); VERTEX_AI_SA_B64 (for graphics_definition_v2); USER_EMAIL + SMTP (for notifications)
    def _shell_escape(s):
        return (s or "").replace("'", "'\"'\"'")
    user_email_escaped = _shell_escape(args.user_email)
    smtp_user = _shell_escape(os.environ.get("SMTP_USER", ""))
    smtp_pass = _shell_escape(os.environ.get("SMTP_APP_PASSWORD", ""))
    oauth_id = _shell_escape(os.environ.get("OAUTH_CLIENT_ID", ""))
    oauth_secret = _shell_escape(os.environ.get("OAUTH_CLIENT_SECRET", ""))
    refresh_plain = ""
    if (args.google_oauth_refresh_token_b64 or "").strip():
        try:
            refresh_plain = base64.b64decode(args.google_oauth_refresh_token_b64.strip()).decode("utf-8")
        except Exception as e:
            print(f"[WARN] Could not decode --google_oauth_refresh_token_b64: {e}")
    refresh_escaped = _shell_escape(refresh_plain)
    export_env = (
        f"export GDRIVE_SA_B64='{gdrive_sa_b64}' && "
        f"export VERTEX_AI_SA_B64='{vertex_ai_sa_b64}' && "
        f"export USER_EMAIL='{user_email_escaped}' && "
        f"export SMTP_USER='{smtp_user}' && "
        f"export SMTP_APP_PASSWORD='{smtp_pass}' && "
        f"export OAUTH_CLIENT_ID='{oauth_id}' && "
        f"export OAUTH_CLIENT_SECRET='{oauth_secret}' && "
    )
    if refresh_plain:
        export_env += f"export GOOGLE_OAUTH_REFRESH_TOKEN='{refresh_escaped}' && "
        print("[INFO] Forwarding user OAuth refresh token to job (human-feedback Drive uploads).")
    toggles_arg = ""
    if args.toggles:
        escaped_toggles = args.toggles.replace("'", "'\"'\"'")
        toggles_arg = f" --toggles '{escaped_toggles}'"
    hf_args = ""
    if args.human_feedback_column:
        hf_args += f" --human_feedback_column '{_shell_escape(args.human_feedback_column)}'"
    if args.human_feedback_status_column:
        hf_args += f" --human_feedback_status_column '{_shell_escape(args.human_feedback_status_column)}'"
    if args.human_feedback_revision_tracking_column:
        hf_args += f" --human_feedback_revision_tracking_column '{_shell_escape(args.human_feedback_revision_tracking_column)}'"
    if args.human_review_actions_column:
        hf_args += f" --human_review_actions_column '{_shell_escape(args.human_review_actions_column)}'"
    if args.llm:
        hf_args += f" --llm '{_shell_escape(args.llm)}'"
    if args.max_workers:
        hf_args += f" --max_workers '{_shell_escape(args.max_workers)}'"
    if args.use_only_drive_and_hvac:
        hf_args += f" --use_only_drive_and_hvac '{_shell_escape(args.use_only_drive_and_hvac)}'"
    if args.source_tab:
        hf_args += f" --source_tab '{_shell_escape(args.source_tab)}'"
    if args.regen_input_column:
        hf_args += f" --regen_input_column '{_shell_escape(args.regen_input_column)}'"
    if args.regen_output_column:
        hf_args += f" --regen_output_column '{_shell_escape(args.regen_output_column)}'"
    if args.regen_output_folder_name:
        hf_args += f" --regen_output_folder_name '{_shell_escape(args.regen_output_folder_name)}'"
    if args.regen_write_final_graphics:
        hf_args += f" --regen_write_final_graphics '{_shell_escape(args.regen_write_final_graphics)}'"
    if args.regen_skip_filled_rows:
        hf_args += f" --regen_skip_filled_rows '{_shell_escape(args.regen_skip_filled_rows)}'"
    command = (
        f"echo 'numpy<2' > /tmp/constraints.txt && "
        f"pip install 'numpy<2' 'matplotlib>=3.9' 'scikit-learn>=1.5' google-cloud-aiplatform && "
        f"pip install -c /tmp/constraints.txt -r requirements.txt && "
        f"{export_env}"
        f"python run_agent_cli.py "
        f"--sheet_link '{args.sheet_link}' "
        f"--drive_folder_id '{args.drive_folder_id}' "
        f"--agent_name '{agent}'"
        f" --run_id '{run_id}'"
        f"{toggles_arg}"
        f"{hf_args}"
    )

    machine_name = (args.machine or "CPU").strip().upper()
    machine = getattr(Machine, machine_name, None)
    if machine is None:
        print(f"[WARN] Unknown machine '{args.machine}'. Falling back to Machine.CPU.")
        machine = Machine.CPU

    gdv2_agent_markers = (
        "graphics_definition_v2",
        "human_feedback_review_revise",
    )
    is_gdv2_related_job = any(marker in (agent or "") for marker in gdv2_agent_markers)
    interruptible = not is_gdv2_related_job

    print(f"\n[INFO] Submitting job for agent: {agent}")
    print(f"[RUN_ID] {run_id}")
    print(f"[INFO] interruptible={interruptible}")
    job = jobs_plugin.run(
        command,
        name = f"{agent}-job",
        machine = machine,
        interruptible = interruptible
    )


    print(f"[INFO] Job '{job.name}' submitted.")
    job_link = getattr(job, "link", "") or ""

    # Launch an external monitor process so terminal-status emails still send even if the job process is hard-killed and cannot run its own handlers.
    try:
        monitor_cmd = [
            sys.executable,
            "monitor_background_job.py",
            "--job_name",
            job.name,
            "--teamspace",
            teamspace,
            "--user",
            user,
            "--run_id",
            run_id,
            "--user_email",
            args.user_email or "",
            "--agent_name",
            ui_agent_name,
            "--sheet_link",
            args.sheet_link or "",
            "--job_link",
            job_link,
        ]
        subprocess.Popen(monitor_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        print("[INFO] Background monitor process started.")
    except Exception as e:
        print(f"[WARN] Could not start background monitor process: {e}")

    if job_link:
        print(f"[JOB_LINK] {job_link}")
    else:
        print(f"[JOB_NAME] {job.name}")

if __name__ == "__main__":
    main()