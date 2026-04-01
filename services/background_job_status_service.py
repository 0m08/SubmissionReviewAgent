from datetime import datetime
import gspread

from services.sheets_service import format_worksheet


BACKGROUND_STATUS_HEADERS = [
    "Timestamp",
    "Run ID",
    "User Email",
    "Agent Name",
    "Status",
    "Sheet Link",
    "Job Link",
    "Message",
]

TERMINAL_STATUSES = {"completed", "failed", "stopped"}
IN_PROGRESS_STATUSES = {"pending", "running"}


def _tracking_tab_name(agent_name: str) -> str:
    raw = (agent_name or "Agent").strip()
    safe = "".join(ch for ch in raw if ch not in "[]:*?/\\")
    title = f"{safe} Background Job Tracking"
    return title[:100]  # Google Sheets worksheet title limit.


def _get_status_ws(gc: gspread.Client, sheet_link: str, agent_name: str, create_if_missing: bool):
    sheet = gc.open_by_url(sheet_link)
    ws_name = _tracking_tab_name(agent_name)
    try:
        return sheet.worksheet(ws_name)
    except gspread.exceptions.WorksheetNotFound:
        if not create_if_missing:
            return None
        try:
            ws = sheet.add_worksheet(ws_name, rows=5000, cols=len(BACKGROUND_STATUS_HEADERS))
            ws.append_row(BACKGROUND_STATUS_HEADERS, value_input_option="USER_ENTERED")
            try:
                format_worksheet(ws)
            except Exception:
                pass
            return ws
        except Exception:
            return sheet.worksheet(ws_name)


def append_background_job_status(
    gc: gspread.Client,
    run_id: str,
    user_email: str,
    agent_name: str,
    status: str,
    sheet_link: str = "",
    job_link: str = "",
    message: str = "",
):
    if run_id and not should_append_status(
        gc=gc,
        run_id=run_id,
        new_status=status,
        sheet_link=sheet_link,
        agent_name=agent_name,
    ):
        return

    ws = _get_status_ws(gc, sheet_link, agent_name, create_if_missing=True)
    if ws is None:
        return
    row = [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        run_id or "",
        user_email or "",
        agent_name or "",
        (status or "").strip().lower(),
        sheet_link or "",
        job_link or "",
        message or "",
    ]
    ws.append_row(row, value_input_option="USER_ENTERED")


def get_latest_background_status(
    gc: gspread.Client,
    sheet_link: str = "",
    agent_name: str = "",
    run_id: str = "",
    user_email: str = "",
):
    ws = _get_status_ws(gc, sheet_link, agent_name, create_if_missing=False)
    if ws is None:
        return None
    records = ws.get_all_records()
    if not records:
        return None

    run_id = (run_id or "").strip()
    user_email = (user_email or "").strip().lower()
    agent_name = (agent_name or "").strip()
    sheet_link = (sheet_link or "").strip()

    for rec in reversed(records):
        if run_id and str(rec.get("Run ID", "")).strip() != run_id:
            continue
        if user_email and str(rec.get("User Email", "")).strip().lower() != user_email:
            continue
        if agent_name and str(rec.get("Agent Name", "")).strip() != agent_name:
            continue
        if sheet_link and str(rec.get("Sheet Link", "")).strip() != sheet_link:
            continue
        return rec
    return None


def has_terminal_status_for_run(gc: gspread.Client, run_id: str, sheet_link: str, agent_name: str) -> bool:
    rec = get_latest_background_status(gc=gc, run_id=run_id, sheet_link=sheet_link, agent_name=agent_name)
    if not rec:
        return False
    status = str(rec.get("Status", "")).strip().lower()
    return status in TERMINAL_STATUSES


def should_append_status(gc: gspread.Client, run_id: str, new_status: str, sheet_link: str, agent_name: str) -> bool:
    """
    Return True if this status should be appended as a new transition.
    Avoids writing duplicate rows for unchanged status.
    """
    rec = get_latest_background_status(gc=gc, run_id=run_id, sheet_link=sheet_link, agent_name=agent_name)
    if not rec:
        return True
    prev = str(rec.get("Status", "")).strip().lower()
    curr = str(new_status or "").strip().lower()
    return prev != curr


def clear_background_status_rows(gc: gspread.Client, sheet_link: str, agent_name: str) -> None:
    ws = _get_status_ws(gc, sheet_link, agent_name, create_if_missing=True)
    if ws is None:
        return
    total_rows = ws.row_count
    if total_rows <= 1:
        return
    ws.batch_clear([f"A2:H{total_rows}"])
