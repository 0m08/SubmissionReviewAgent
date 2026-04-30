import os
import sys
import time

from lightning_sdk import Studio


#
# import smtplib
# from datetime import datetime, timedelta
# from email.mime.multipart import MIMEMultipart
# from email.mime.text import MIMEText
#
# RECIPIENT_EMAILS = [
#     "niket@skillcatapp.com",
#     "dilip@skillcatapp.com",
# ]


def _required_env(name: str) -> str:
    value = (os.environ.get(name) or "").strip()
    if not value:
        raise RuntimeError(f"Missing required env var: {name}")
    return value


def _status_text(status_obj) -> str:
    if status_obj is None:
        return ""
    text = str(status_obj)
    if "." in text:
        text = text.split(".")[-1]
    return text.strip().lower()


def _is_running(status_obj) -> bool:
    return _status_text(status_obj) in {"running"}


# def _format_time_utc_and_ist() -> str:
#     utc_now = datetime.utcnow()
#     ist_now = utc_now + timedelta(hours=5, minutes=30)
#     return f"{utc_now.strftime('%Y-%m-%d %H:%M:%S')} UTC ({ist_now.strftime('%Y-%m-%d %H:%M:%S')} IST)"
#
#
# def _send_smtp_email(subject: str, body_plain: str, body_html: str | None = None) -> bool:
#     smtp_user = (os.environ.get("SMTP_USER") or "").strip()
#     smtp_password = (os.environ.get("SMTP_APP_PASSWORD") or "").strip()
#     if not smtp_user or not smtp_password:
#         print("[WARN] SMTP is not fully configured; skipping status email.")
#         return False
#     recipients = [email.strip() for email in RECIPIENT_EMAILS if email.strip()]
#     if not recipients:
#         print("[WARN] No recipient emails configured; skipping status email.")
#         return False
#
#     if body_html:
#         msg = MIMEMultipart("alternative")
#         msg.attach(MIMEText(body_plain, "plain", "utf-8"))
#         msg.attach(MIMEText(body_html, "html", "utf-8"))
#     else:
#         msg = MIMEText(body_plain, "plain", "utf-8")
#
#     msg["Subject"] = subject
#     msg["From"] = smtp_user
#     msg["To"] = ", ".join(recipients)
#     try:
#         with smtplib.SMTP("smtp.gmail.com", 587) as server:
#             server.starttls()
#             server.login(smtp_user, smtp_password)
#             server.sendmail(smtp_user, recipients, msg.as_string())
#         print(f"[INFO] Status email sent to: {', '.join(recipients)}")
#         return True
#     except Exception as exc:
#         print(f"[WARN] Failed to send status email: {exc}")
#         return False


def main() -> int:
    started_at = time.time()
    action_taken = "none"
    status_before = "unknown"
    status_after = "unknown"
    changed = "no"
    outcome = "failure"
    error_text = ""

    studio_name = _required_env("LIGHTNING_STUDIO_NAME")
    teamspace = _required_env("LIGHTNING_TEAMSPACE")
    username = _required_env("LIGHTNING_USERNAME")

    max_wait_seconds = int(os.environ.get("LIGHTNING_WAKE_MAX_WAIT_SECONDS", "300"))
    poll_seconds = int(os.environ.get("LIGHTNING_WAKE_POLL_SECONDS", "10"))

    exit_code = 1
    try:
        print(
            f"[INFO] Checking studio status for '{studio_name}' "
            f"(teamspace='{teamspace}', user='{username}')"
        )
        studio = Studio(name=studio_name, teamspace=teamspace, user=username, create_ok=False)
        status_before = _status_text(getattr(studio, "status", "")) or "unknown"
        print(f"[INFO] Current status: {status_before}")

        if _is_running(getattr(studio, "status", "")):
            status_after = status_before
            outcome = "success"
            exit_code = 0
            print("[INFO] Studio already running. No action needed.")
        else:
            action_taken = "start"
            print("[INFO] Studio not running. Sending start request...")
            studio.start()

            wake_started_at = time.time()
            while time.time() - wake_started_at < max_wait_seconds:
                status_now = getattr(studio, "status", "")
                status_text = _status_text(status_now) or "unknown"
                print(f"[INFO] Waiting for running state; current status: {status_text}")
                if _is_running(status_now):
                    status_after = status_text
                    outcome = "success"
                    exit_code = 0
                    print("[INFO] Studio is now running.")
                    break
                time.sleep(max(1, poll_seconds))

            if exit_code != 0:
                status_after = _status_text(getattr(studio, "status", "")) or "unknown"
                error_text = (
                    "Timed out waiting for studio to reach running state "
                    f"(waited {max_wait_seconds}s)."
                )
                print(f"[ERROR] {error_text}", file=sys.stderr)
    except Exception as exc:
        error_text = str(exc)
        print(f"[ERROR] Wake check failed: {error_text}", file=sys.stderr)

    if status_before != status_after:
        changed = "yes"

    elapsed = round(time.time() - started_at, 2)
    print(
        f"[INFO] Wake check summary: status_before={status_before}, "
        f"action_taken={action_taken}, status_after={status_after}, "
        f"status_changed={changed}, outcome={outcome}, elapsed_seconds={elapsed}, "
        f"error={error_text or 'none'}"
    )

    # timestamp = _format_time_utc_and_ist()
    # subject = (
    #     f"[Lightning Wake Check] {studio_name} - "
    #     f"{'SUCCESS' if outcome == 'success' else 'FAILED'}"
    # )
    # body_plain = (
    #     "Lightning Studio wake-check result\n\n"
    #     f"Timestamp: {timestamp}\n"
    #     f"Studio: {studio_name}\n"
    #     f"Teamspace: {teamspace}\n"
    #     f"User: {username}\n"
    #     f"Status before: {status_before}\n"
    #     f"Action taken: {action_taken}\n"
    #     f"Status after: {status_after}\n"
    #     f"Status changed: {changed}\n"
    #     f"Outcome: {outcome}\n"
    #     f"Elapsed seconds: {elapsed}\n"
    #     f"Error: {error_text or 'none'}\n"
    # )
    # _send_smtp_email(subject=subject, body_plain=body_plain)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
