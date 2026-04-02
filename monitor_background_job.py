import argparse
import base64
import json
import os
import sys
import time
from datetime import datetime, timedelta

import gspread
from dotenv import load_dotenv
from lightning_sdk.job.job import Job

from services.background_job_status_service import (
    append_background_job_status,
    has_terminal_status_for_run,
    should_append_status,
)
from services.email_service import is_email_configured, send_notification_email


def _format_time_utc_and_ist():
    utc_now = datetime.utcnow()
    ist_now = utc_now + timedelta(hours=5, minutes=30)
    return f"{utc_now.strftime('%Y-%m-%d %H:%M')} UTC ({ist_now.strftime('%Y-%m-%d %H:%M')} IST)"


def _normalize_status(status_text: str):
    return (status_text or "").strip().lower()


def _status_to_terminal(status_text: str):
    s = _normalize_status(status_text)
    if s in {"completed", "failed", "stopped"}:
        return s
    return ""


def _send_terminal_email(status, user_email, ui_agent_name, course_name, sheet_link, job_link):
    if not user_email or not is_email_configured():
        return
    happened_at = _format_time_utc_and_ist()
    status_title = status.capitalize()
    subject = f"Course generation: {ui_agent_name} AI Agent – {status_title}"
    sheet_link_html = (
        f'<a href="{sheet_link}" target="_blank" rel="noopener noreferrer">Link</a>'
        if sheet_link else "N/A"
    )
    sheet_link_plain = f"Link ({sheet_link})" if sheet_link else "N/A"
    job_link_html = (
        f'<a href="{job_link}" target="_blank" rel="noopener noreferrer">Lightning Job</a>'
        if job_link else "N/A"
    )
    job_link_plain = job_link or "N/A"

    body = (
        f"Your {ui_agent_name} AI agent reached terminal status: {status_title}.\n\n"
        f"Agent:      {ui_agent_name}\n"
        f"Course:     {course_name}\n"
        f"Time:       {happened_at}\n"
        f"Status:     {status_title}\n\n"
        f"Input Sheet:  {sheet_link_plain}\n"
        f"Job Link:     {job_link_plain}\n"
    )
    body_html = (
        f"<p>Your {ui_agent_name} AI agent reached terminal status: <b>{status_title}</b>.</p>"
        f"<p><b>Agent:</b> {ui_agent_name}<br>"
        f"<b>Course:</b> {course_name}<br>"
        f"<b>Time:</b> {happened_at}<br>"
        f"<b>Status:</b> {status_title}<br>"
        f"<b>Input Sheet:</b> {sheet_link_html}<br>"
        f"<b>Job Link:</b> {job_link_html}</p>"
    )
    send_notification_email(
        to_email=user_email,
        subject=subject,
        body_plain=body,
        body_html=body_html,
    )


def _build_gc():
    load_dotenv()
    key_b64 = os.environ.get("GDRIVE_SA_B64", "")
    if not key_b64:
        raise RuntimeError("GDRIVE_SA_B64 not set for monitor_background_job.py")
    sa_json = base64.b64decode(key_b64).decode()
    sa_dict = json.loads(sa_json)
    return gspread.service_account_from_dict(sa_dict)


def main():
    parser = argparse.ArgumentParser(description="Monitor Lightning job terminal status and notify.")
    parser.add_argument("--job_name", required=True)
    parser.add_argument("--teamspace", default="Vision-model")
    parser.add_argument("--user", default="dilip")
    parser.add_argument("--run_id", default="")
    parser.add_argument("--user_email", default="")
    parser.add_argument("--agent_name", default="")
    parser.add_argument("--course_name", default="")
    parser.add_argument("--sheet_link", default="")
    parser.add_argument("--job_link", default="")
    parser.add_argument("--poll_seconds", type=int, default=30)
    parser.add_argument("--timeout_seconds", type=int, default=172800)  # 48h
    args = parser.parse_args()

    try:
        gc = _build_gc()
    except Exception as e:
        print(f"[MONITOR] Failed to build gspread client: {e}")
        sys.exit(1)

    start = time.time()
    while True:
        if args.run_id and has_terminal_status_for_run(gc, args.run_id, args.sheet_link, args.agent_name):
            print("[MONITOR] Terminal status already recorded for run; exiting.")
            return

        try:
            job = Job(name=args.job_name, teamspace=args.teamspace, user=args.user)
            status_raw = str(getattr(job, "status", "")).strip()
            status_norm = _normalize_status(status_raw)
            terminal = _status_to_terminal(status_raw)
            print(f"[MONITOR] job={args.job_name} status={status_raw}")

            if status_norm and should_append_status(
                gc,
                args.run_id,
                status_norm,
                sheet_link=args.sheet_link,
                agent_name=args.agent_name,
            ):
                append_background_job_status(
                    gc=gc,
                    run_id=args.run_id,
                    user_email=args.user_email,
                    agent_name=args.agent_name,
                    status=status_norm,
                    sheet_link=args.sheet_link,
                    job_link=args.job_link or getattr(job, "link", ""),
                    message=f"Lightning status update: {status_raw}",
                )

            if terminal:
                _send_terminal_email(
                    status=terminal,
                    user_email=args.user_email,
                    ui_agent_name=args.agent_name,
                    course_name=args.course_name,
                    sheet_link=args.sheet_link,
                    job_link=args.job_link or getattr(job, "link", ""),
                )
                print(f"[MONITOR] Terminal status reached: {terminal}. Exiting.")
                return
        except Exception as e:
            print(f"[MONITOR] Status poll error: {e}")

        if time.time() - start > args.timeout_seconds:
            print("[MONITOR] Timeout reached; exiting monitor.")
            return
        time.sleep(max(10, args.poll_seconds))


if __name__ == "__main__":
    main()
