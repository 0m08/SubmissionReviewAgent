import os
import sys
import time

from lightning_sdk import Studio


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
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
