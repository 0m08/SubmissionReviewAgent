# -*- coding: utf-8 -*-
"""
Central Google Sheet Review Registry Manager.

Manages the central Google Sheet tracking all Moodle submissions:
- One tab per Activity (e.g. 'System Identification')
- Specific schema: Name, Status, Grade, Online text, Media folder, Attempt number, Last modified, Feedback comment, Agent's checklist
- Preserves unique rows per attempt; increments Attempt number if a student submits a new attempt
- Full integration with Google Drive and user OAuth credentials (om@skillcatapp.com)
"""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import gspread
import pandas as pd
from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from gspread_formatting import (
    CellFormat,
    Color,
    TextFormat,
    format_cell_range,
    set_column_width,
)

load_dotenv()

logger = logging.getLogger("central_sheet_manager")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] [SheetManager] %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Default configuration constants
DEFAULT_DRIVE_FOLDER_ID = "1jR5JP3aTKt8u_X57Z31IS71ZGBX-zfeu"
DEFAULT_REGISTRY_SHEET_NAME = "Moodle Submission Review Registry"
REGISTRY_SHEET_ID_KNOWN = "1aP7Xdvoi4TTIT8K7X9j2wD0G6-YCnA7sEn-BiLDWWQU"

CANONICAL_COLUMNS = [
    "Name",
    "Status",
    "Grade",
    "Online text",
    "Media folder",
    "Attempt number",
    "Last modified",
    "Feedback comment",
    "Agent's checklist",
    "Agent Grade",
    "Review status",
    "Mentor reviewer",
    "Mentor reviewed at",
]


def sanitize_sheet_cell_value(val: Any) -> Any:
    """
    Guards against CSV/Formula Injection in Google Sheets.
    If string starts with =, +, -, @, \\t, or \\r, prefix with a single quote '
    unless it is a legitimate signed number.
    """
    if val is None:
        return ""
    if isinstance(val, (int, float)):
        return val
    s = str(val)
    if not s:
        return ""
    if s[0] in ("=", "+", "-", "@", "\t", "\r"):
        try:
            float(s)
            return val
        except ValueError:
            return "'" + s
    return s


def get_oauth_credentials() -> Credentials:
    """Instantiates Google OAuth credentials using user refresh token."""
    refresh = (os.environ.get("GOOGLE_OAUTH_REFRESH_TOKEN") or os.environ.get("OAUTH_REFRESH_TOKEN", "")).strip()
    client_id = (os.environ.get("GOOGLE_OAUTH_CLIENT_ID") or os.environ.get("OAUTH_CLIENT_ID", "")).strip()
    client_secret = (os.environ.get("GOOGLE_OAUTH_CLIENT_SECRET") or os.environ.get("OAUTH_CLIENT_SECRET", "")).strip()

    if not refresh:
        raise ValueError("Missing GOOGLE_OAUTH_REFRESH_TOKEN in environment.")
    if not client_id or not client_secret:
        raise ValueError("Missing GOOGLE_OAUTH_CLIENT_ID or GOOGLE_OAUTH_CLIENT_SECRET in environment.")

    creds = Credentials(
        None,
        refresh_token=refresh,
        token_uri="https://oauth2.googleapis.com/token",
        client_id=client_id,
        client_secret=client_secret,
        scopes=[
            "https://www.googleapis.com/auth/drive",
            "https://www.googleapis.com/auth/spreadsheets",
        ],
    )
    creds.refresh(Request())
    return creds


def get_gspread_client() -> gspread.Client:
    """Returns an authenticated gspread Client."""
    creds = get_oauth_credentials()
    return gspread.authorize(creds)


def get_or_create_registry_spreadsheet(
    drive_folder_id: Optional[str] = None,
    sheet_name: str = DEFAULT_REGISTRY_SHEET_NAME,
) -> Tuple[gspread.Spreadsheet, str]:
    """
    Locates or creates the central 'Moodle Submission Review Registry' spreadsheet
    in the specified Google Drive folder.
    Returns: (Spreadsheet, webViewLink)
    """
    target_folder_id = drive_folder_id or os.environ.get("MOODLE_DRIVE_FOLDER_ID") or DEFAULT_DRIVE_FOLDER_ID
    if "/folders/" in target_folder_id:
        m = re.search(r'/folders/([a-zA-Z0-9_-]+)', target_folder_id)
        if m:
            target_folder_id = m.group(1)

    gc = get_gspread_client()

    # 1. Try explicit env var or known ID first
    explicit_id = os.environ.get("CENTRAL_REVIEW_SHEET_ID") or REGISTRY_SHEET_ID_KNOWN
    if explicit_id:
        try:
            sh = gc.open_by_key(explicit_id)
            link = f"https://docs.google.com/spreadsheets/d/{explicit_id}/edit"
            logger.info(f"Connected to central review spreadsheet by ID: {sh.title} ({link})")
            return sh, link
        except Exception as e:
            logger.warning(f"Could not open spreadsheet by ID '{explicit_id}': {e}. Searching Drive...")

    # 2. Search target Google Drive folder
    creds = get_oauth_credentials()
    drive_service = build("drive", "v3", credentials=creds, cache_discovery=False)
    q = f"name = '{sheet_name}' and '{target_folder_id}' in parents and mimeType = 'application/vnd.google-apps.spreadsheet' and trashed = false"
    res = drive_service.files().list(
        q=q, spaces="drive", fields="files(id, name, webViewLink)",
        supportsAllDrives=True, includeItemsFromAllDrives=True
    ).execute()
    files = res.get("files", [])

    if files:
        sheet_id = files[0]["id"]
        link = files[0].get("webViewLink") or f"https://docs.google.com/spreadsheets/d/{sheet_id}/edit"
        sh = gc.open_by_key(sheet_id)
        logger.info(f"Found existing registry spreadsheet in Drive: '{sh.title}' ({link})")
        return sh, link

    # 3. Create if not found
    file_metadata = {
        "name": sheet_name,
        "parents": [target_folder_id],
        "mimeType": "application/vnd.google-apps.spreadsheet",
    }
    new_file = drive_service.files().create(
        body=file_metadata, fields="id, name, webViewLink", supportsAllDrives=True
    ).execute()
    sheet_id = new_file["id"]
    link = new_file.get("webViewLink") or f"https://docs.google.com/spreadsheets/d/{sheet_id}/edit"
    sh = gc.open_by_key(sheet_id)
    logger.info(f"Created new registry spreadsheet in Drive folder: '{sh.title}' ({link})")
    return sh, link


def format_worksheet_headers(worksheet: gspread.Worksheet) -> None:
    """Applies professional formatting to row 1 (headers) and column widths for all 13 columns."""
    try:
        # Freeze row 1
        worksheet.freeze(rows=1, cols=0)

        # Style header row (columns A to M)
        header_format = CellFormat(
            backgroundColor=Color(0.12, 0.22, 0.35),  # Deep Navy #1F3859
            textFormat=TextFormat(bold=True, foregroundColor=Color(1.0, 1.0, 1.0), fontSize=10),
        )
        format_cell_range(worksheet, "A1:M1", header_format)

        # Set column widths
        widths = {
            1: 180,  # Name
            2: 140,  # Status
            3: 110,  # Grade
            4: 250,  # Online text
            5: 220,  # Media folder
            6: 100,  # Attempt number
            7: 170,  # Last modified
            8: 300,  # Feedback comment
            9: 380,  # Agent's checklist
            10: 130, # Agent Grade
            11: 170, # Review status
            12: 150, # Mentor reviewer
            13: 180, # Mentor reviewed at
        }
        for col_idx, width in widths.items():
            try:
                set_column_width(worksheet, str(col_idx), width)
            except Exception:
                pass
    except Exception as fmt_err:
        logger.warning(f"Could not apply worksheet formatting: {fmt_err}")


def ensure_worksheet_schema(worksheet: gspread.Worksheet) -> List[str]:
    """
    Checks worksheet headers. If columns are missing or if it has the older 9-column
    schema, upgrades row 1 to all 13 CANONICAL_COLUMNS and applies formatting.
    """
    first_row = worksheet.row_values(1)
    if not first_row:
        worksheet.update("A1:M1", [CANONICAL_COLUMNS])
        format_worksheet_headers(worksheet)
        return CANONICAL_COLUMNS

    existing_lower = [h.strip().lower() for h in first_row]
    canonical_lower = [h.strip().lower() for h in CANONICAL_COLUMNS]

    if existing_lower != canonical_lower:
        worksheet.update("A1:M1", [CANONICAL_COLUMNS])
        format_worksheet_headers(worksheet)
        logger.info(f"Upgraded worksheet '{worksheet.title}' headers to 13 canonical columns.")
        return CANONICAL_COLUMNS

    return first_row


def get_or_create_activity_worksheet(
    spreadsheet: gspread.Spreadsheet,
    activity_name: str,
) -> gspread.Worksheet:
    """
    Finds or creates a tab in the spreadsheet for the given activity name.
    Ensures canonical headers exist on row 1.
    """
    clean_title = re.sub(r'[:/\\?*\[\]]', '_', activity_name.strip())[:90] or "General Activity"

    worksheets = {ws.title.lower(): ws for ws in spreadsheet.worksheets()}
    if clean_title.lower() in worksheets:
        ws = worksheets[clean_title.lower()]
    else:
        existing = spreadsheet.worksheets()
        if len(existing) == 1 and existing[0].title.lower() in ("sheet1", "sheet 1"):
            first_row = existing[0].row_values(1)
            if not first_row:
                existing[0].update_title(clean_title)
                ws = existing[0]
            else:
                ws = spreadsheet.add_worksheet(title=clean_title, rows=500, cols=15)
        else:
            ws = spreadsheet.add_worksheet(title=clean_title, rows=500, cols=15)

    ensure_worksheet_schema(ws)
    return ws


def format_checklist_for_sheet(checklist_evaluations: List[Any]) -> str:
    """Formats checklist result items into human-readable multi-line summary for Sheet cells."""
    if not checklist_evaluations:
        return ""

    lines = []
    for item in checklist_evaluations:
        if hasattr(item, "followed"):
            followed = item.followed
            text = item.instruction_or_condition
            comment = getattr(item, "comment", "")
        elif isinstance(item, dict):
            followed = item.get("followed", False)
            text = item.get("instruction_or_condition", "")
            comment = item.get("comment", "")
        else:
            continue

        symbol = "✓ PASS" if followed else "✗ FAIL"
        entry = f"[{symbol}] {text}"
        if comment:
            entry += f" — {comment}"
        lines.append(entry)

    return "\n".join(lines)


def sync_evaluations_to_registry(
    activity_name: str,
    evaluated_records: List[Dict[str, Any]],
    drive_folder_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Synchronizes a batch of student evaluation records into the central Google Sheet.

    Args:
        activity_name: Name of activity tab (e.g. 'System Identification')
        evaluated_records: List of dicts, each containing:
            - name: Student name
            - status: Moodle submission status
            - grade: AI qualified grade ('Pass', 'Fail', 'Pass (Unsure)', etc.)
            - online_text: Student's online text
            - media_folder: Google Drive folder link for student
            - last_modified: Moodle submission last modified timestamp
            - feedback_comment: Agent mentor comment
            - checklist_evaluations: Raw checklist items list
        drive_folder_id: Optional Drive folder ID containing the sheet.

    Returns:
        Summary dict of rows added, updated, and spreadsheet link.
    """
    if not evaluated_records:
        logger.info("No student evaluation records provided for sheet sync.")
        return {"updated": 0, "added": 0, "sheet_url": None}

    spreadsheet, sheet_url = get_or_create_registry_spreadsheet(drive_folder_id=drive_folder_id)
    ws = get_or_create_activity_worksheet(spreadsheet, activity_name)

    all_values = ws.get_all_values()
    if not all_values:
        ws.update("A1:I1", [CANONICAL_COLUMNS])
        all_values = [CANONICAL_COLUMNS]

    headers = [h.strip().lower() for h in all_values[0]]
    col_map = {h: idx for idx, h in enumerate(headers)}
    name_idx = col_map.get("name", 0)
    attempt_idx = col_map.get("attempt number", 5)
    last_mod_idx = col_map.get("last modified", 6)

    student_history: Dict[str, List[Tuple[int, str, int]]] = {}
    for r_idx, row_vals in enumerate(all_values[1:], start=2):
        if not row_vals:
            continue
        s_name = row_vals[name_idx].strip().lower() if len(row_vals) > name_idx else ""
        if not s_name:
            continue

        l_mod = row_vals[last_mod_idx].strip() if len(row_vals) > last_mod_idx else ""
        att_str = row_vals[attempt_idx].strip() if len(row_vals) > attempt_idx else "1"
        try:
            att_num = int(att_str)
        except ValueError:
            att_num = 1

        if s_name not in student_history:
            student_history[s_name] = []
        student_history[s_name].append((r_idx, l_mod, att_num))

    rows_to_append = []
    updates_to_perform = []
    added_count = 0
    updated_count = 0

    for rec in evaluated_records:
        name = rec.get("name", "").strip()
        if not name:
            continue

        status = rec.get("status", "")
        grade = rec.get("grade", "")
        online_text = rec.get("online_text", "")
        media_folder = rec.get("media_folder", "") or ""
        last_modified = rec.get("last_modified", "").strip()
        feedback_comment = rec.get("feedback_comment", "")
        checklist_str = format_checklist_for_sheet(rec.get("checklist_evaluations", []))

        s_key = name.lower()
        history = student_history.get(s_key, [])

        matched_row_idx = None
        current_attempt = 1

        if history:
            for existing_row, existing_mod, ex_att in history:
                if last_modified and existing_mod and (last_modified == existing_mod):
                    matched_row_idx = existing_row
                    current_attempt = ex_att
                    break

            if not matched_row_idx:
                max_att = max(h[2] for h in history)
                current_attempt = max_att + 1

        raw_agent_grade = rec.get("agent_grade") or rec.get("grade", "")
        # Active grade for Moodle
        if "pass" in str(raw_agent_grade).lower():
            active_grade = "Pass"
        elif "fail" in str(raw_agent_grade).lower():
            active_grade = "Fail"
        else:
            active_grade = rec.get("grade", "")

        review_status = rec.get("review_status") or "Pending Review"
        mentor_reviewer = rec.get("mentor_reviewer") or ""
        mentor_reviewed_at = rec.get("mentor_reviewed_at") or ""

        # If updating an existing row that already had a mentor review, preserve mentor's verdict
        if matched_row_idx and matched_row_idx - 1 < len(all_values):
            ex_row = all_values[matched_row_idx - 1]
            if len(ex_row) >= 11 and ex_row[10].strip() in ("Approved by Mentor", "Overridden by Mentor"):
                review_status = ex_row[10].strip()
                if len(ex_row) >= 3 and ex_row[2].strip():
                    active_grade = ex_row[2].strip()
                if len(ex_row) >= 8 and ex_row[7].strip():
                    feedback_comment = ex_row[7].strip()
                if len(ex_row) >= 12:
                    mentor_reviewer = ex_row[11].strip()
                if len(ex_row) >= 13:
                    mentor_reviewed_at = ex_row[12].strip()

        row_data = [
            sanitize_sheet_cell_value(name),
            sanitize_sheet_cell_value(status),
            sanitize_sheet_cell_value(active_grade),
            sanitize_sheet_cell_value(online_text),
            sanitize_sheet_cell_value(media_folder),
            current_attempt,
            sanitize_sheet_cell_value(last_modified),
            sanitize_sheet_cell_value(feedback_comment),
            sanitize_sheet_cell_value(checklist_str),
            sanitize_sheet_cell_value(raw_agent_grade),
            sanitize_sheet_cell_value(review_status),
            sanitize_sheet_cell_value(mentor_reviewer),
            sanitize_sheet_cell_value(mentor_reviewed_at),
        ]

        if matched_row_idx:
            range_name = f"A{matched_row_idx}:M{matched_row_idx}"
            updates_to_perform.append({"range": range_name, "values": [row_data]})
            updated_count += 1
            logger.info(f"   🔄 [Registry]: Updating existing attempt {current_attempt} for '{name}' at row {matched_row_idx}.")
        else:
            rows_to_append.append(row_data)
            added_count += 1
            if s_key not in student_history:
                student_history[s_key] = []
            next_row = len(all_values) + len(rows_to_append)
            student_history[s_key].append((next_row, last_modified, current_attempt))
            logger.info(f"   ➕ [Registry]: Appending new attempt {current_attempt} for '{name}'.")

    if updates_to_perform:
        try:
            ws.batch_update(updates_to_perform)
            logger.info(f"Updated {len(updates_to_perform)} existing row(s) in tab '{ws.title}'.")
        except Exception as up_err:
            logger.error(f"Failed to execute batch update: {up_err}")
            for u in updates_to_perform:
                try:
                    ws.update(u["range"], u["values"])
                except Exception:
                    pass

    if rows_to_append:
        try:
            ws.append_rows(rows_to_append, value_input_option="USER_ENTERED")
            logger.info(f"Appended {len(rows_to_append)} new row(s) in tab '{ws.title}'.")
        except Exception as ap_err:
            logger.error(f"Failed to append rows: {ap_err}")

    return {
        "activity": activity_name,
        "tab_name": ws.title,
        "sheet_url": sheet_url,
        "added": added_count,
        "updated": updated_count,
        "total_synced": added_count + updated_count,
    }


def fetch_all_registry_activities(
    drive_folder_id: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Fetches all activity worksheets from the registry and returns a mapping of
    {activity_name: DataFrame}.
    Used by Streamlit UI for instant interactive filtering and inspection.
    """
    spreadsheet, _ = get_or_create_registry_spreadsheet(drive_folder_id=drive_folder_id)
    worksheets = spreadsheet.worksheets()

    results: Dict[str, pd.DataFrame] = {}
    reserved_tabs = {"sheet1", "sheet 1", "activity details", "activity instructions", "guardrails"}
    for ws in worksheets:
        clean_title = ws.title.strip().lower()
        if clean_title in reserved_tabs:
            continue
        if clean_title in ("sheet1", "sheet 1") and ws.row_count <= 1:
            continue

        try:
            # Upgrade schema if older
            ensure_worksheet_schema(ws)
            data = ws.get_all_records()
            df = pd.DataFrame(data)
            if not df.empty:
                # Ensure all 13 canonical columns exist in df
                for c in CANONICAL_COLUMNS:
                    if c not in df.columns:
                        df[c] = ""
                # Backfill defaults if empty
                if "Agent Grade" in df.columns:
                    df["Agent Grade"] = df["Agent Grade"].replace("", pd.NA).fillna(df["Grade"]).fillna("")
                if "Review status" in df.columns:
                    df["Review status"] = df["Review status"].replace("", "Pending Review")
                results[ws.title] = df
            else:
                vals = ws.get_all_values()
                if vals:
                    df = pd.DataFrame(columns=vals[0])
                    results[ws.title] = df
        except Exception as err:
            logger.warning(f"Could not load records from worksheet '{ws.title}': {err}")

    return results


def fetch_activity_details_from_sheet(
    sheet_id_or_url: Optional[str] = None,
    drive_folder_id: Optional[str] = None,
) -> Dict[str, Dict[str, str]]:
    """
    Reads the 'Activity Details' (or 'Activity Instructions') tab from the Moodle Submission Review sheet.

    Expected columns in 'Activity Details' tab:
      - 'Activity Name' or 'Tab Name'
      - 'Activity Instructions' or 'Instructions'
      - 'Edge Cases'
      - 'Guidelines'

    Returns:
        Dict mapping activity names (both exact title and lowercase title) to:
        {
            "activity_name": str,
            "instructions": str,
            "edge_cases": str,
            "guidelines": str,
        }
    """
    spreadsheet = None
    target_id = None
    if sheet_id_or_url:
        m = re.search(r'/spreadsheets/d/([a-zA-Z0-9_-]+)', sheet_id_or_url)
        if m:
            target_id = m.group(1)
        else:
            m2 = re.search(r'id=([a-zA-Z0-9_-]+)', sheet_id_or_url)
            if m2:
                target_id = m2.group(1)
            elif re.match(r'^[a-zA-Z0-9_-]{20,}$', sheet_id_or_url.strip()):
                target_id = sheet_id_or_url.strip()

    if target_id:
        try:
            gc = get_gspread_client()
            spreadsheet = gc.open_by_key(target_id)
        except Exception as err:
            logger.warning(f"[SheetManager] Could not open sheet by key '{target_id}': {err}")

    if spreadsheet is None:
        try:
            spreadsheet, _ = get_or_create_registry_spreadsheet(drive_folder_id=drive_folder_id)
        except Exception as err:
            logger.error(f"[SheetManager] Failed to connect to registry spreadsheet: {err}")
            return {}

    ws = None
    for tab_candidate in ["Activity Details", "activity details", "Activity Instructions", "activity instructions"]:
        try:
            ws = spreadsheet.worksheet(tab_candidate)
            break
        except Exception:
            continue

    if ws is None:
        for w in spreadsheet.worksheets():
            if w.title.strip().lower() in ("activity details", "activity instructions"):
                ws = w
                break

    if ws is None:
        logger.warning(f"[SheetManager] 'Activity Details' tab not found in spreadsheet '{spreadsheet.title}'.")
        return {}

    try:
        records = ws.get_all_records()
    except Exception as read_err:
        logger.error(f"[SheetManager] Failed to read records from tab '{ws.title}': {read_err}")
        return {}

    results: Dict[str, Dict[str, str]] = {}
    for r in records:
        if not isinstance(r, dict):
            continue

        act_name = str(
            r.get("Activity Name")
            or r.get("Tab Name")
            or r.get("activity name")
            or r.get("tab name")
            or r.get("Activity")
            or r.get("Name")
            or ""
        ).strip()

        if not act_name:
            continue

        instructions = str(
            r.get("Activity Instructions")
            or r.get("Instructions")
            or r.get("activity instructions")
            or r.get("instructions")
            or r.get("Activity Instruction")
            or ""
        ).strip()

        edge_cases = str(
            r.get("Edge Cases")
            or r.get("edge cases")
            or r.get("Edge Case")
            or r.get("edge case")
            or r.get("Activity Edge Cases")
            or ""
        ).strip()

        guidelines = str(
            r.get("Guidelines")
            or r.get("guidelines")
            or r.get("Guideline")
            or r.get("guideline")
            or r.get("Activity Guidelines")
            or ""
        ).strip()

        entry = {
            "activity_name": act_name,
            "instructions": instructions,
            "edge_cases": edge_cases,
            "guidelines": guidelines,
        }
        results[act_name] = entry
        results[act_name.lower()] = entry

    logger.info(f"[SheetManager] Successfully loaded activity details for {len(results) // 2 if results else 0} activities from '{ws.title}'.")
    return results


def get_activity_instructions(
    activity_name: str,
    sheet_id_or_url: Optional[str] = None,
    drive_folder_id: Optional[str] = None,
) -> Dict[str, str]:
    """
    Fetches instructions, edge cases, and guidelines for a specific activity
    from the 'Activity Details' tab of the Moodle Submission Review sheet.
    """
    empty_result = {
        "activity_name": activity_name,
        "instructions": "",
        "edge_cases": "",
        "guidelines": "",
    }
    if not activity_name:
        return empty_result

    all_details = fetch_activity_details_from_sheet(
        sheet_id_or_url=sheet_id_or_url,
        drive_folder_id=drive_folder_id,
    )
    if not all_details:
        return empty_result

    # 1. Exact match
    if activity_name in all_details:
        return all_details[activity_name]

    # 2. Case-insensitive stripped match
    key_lower = activity_name.strip().lower()
    if key_lower in all_details:
        return all_details[key_lower]

    # 3. Substring match
    for k, val in all_details.items():
        if k in key_lower or key_lower in k:
            return val

    return empty_result


def append_activity_details_rule(
    activity_name: str,
    edge_case: Optional[str] = None,
    guideline: Optional[str] = None,
    sheet_id_or_url: Optional[str] = None,
    drive_folder_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Appends an edge case and/or guideline to the 'Activity Details' tab
    for the specific activity.

    - If edge_case is provided, writes/appends to 'Edge Cases' column.
    - If guideline is provided, writes/appends to 'Guidelines' column.
    - If the row for activity_name doesn't exist, appends a new row.

    Returns:
        Dict with status, updated values, and sheet URL.
    """
    if not activity_name:
        raise ValueError("activity_name is required.")

    clean_edge = (edge_case or "").strip()
    clean_guide = (guideline or "").strip()
    if not clean_edge and not clean_guide:
        return {"success": False, "message": "Neither edge case nor guideline was provided."}

    spreadsheet = None
    target_id = None
    if sheet_id_or_url:
        m = re.search(r'/spreadsheets/d/([a-zA-Z0-9_-]+)', sheet_id_or_url)
        if m:
            target_id = m.group(1)
        else:
            m2 = re.search(r'id=([a-zA-Z0-9_-]+)', sheet_id_or_url)
            if m2:
                target_id = m2.group(1)
            elif re.match(r'^[a-zA-Z0-9_-]{20,}$', sheet_id_or_url.strip()):
                target_id = sheet_id_or_url.strip()

    if target_id:
        try:
            gc = get_gspread_client()
            spreadsheet = gc.open_by_key(target_id)
        except Exception as err:
            logger.warning(f"[SheetManager] Could not open sheet by key '{target_id}': {err}")

    if spreadsheet is None:
        spreadsheet, _ = get_or_create_registry_spreadsheet(drive_folder_id=drive_folder_id)

    ws = None
    for tab_candidate in ["Activity Details", "activity details", "Activity Instructions", "activity instructions"]:
        try:
            ws = spreadsheet.worksheet(tab_candidate)
            break
        except Exception:
            continue

    if ws is None:
        for w in spreadsheet.worksheets():
            if w.title.strip().lower() in ("activity details", "activity instructions"):
                ws = w
                break

    if ws is None:
        ws = spreadsheet.add_worksheet(title="Activity Details", rows=50, cols=10)
        ws.append_row(["Activity Name", "Activity Instructions", "Edge Cases", "Guidelines"], value_input_option="USER_ENTERED")

    all_values = ws.get_all_values()
    if not all_values:
        headers = ["Activity Name", "Activity Instructions", "Edge Cases", "Guidelines"]
        ws.append_row(headers, value_input_option="USER_ENTERED")
        all_values = [headers]

    headers = [str(h).strip() for h in all_values[0]]
    header_lower = [h.lower() for h in headers]

    # Find column indices (1-indexed)
    act_col = None
    for i, h in enumerate(header_lower, 1):
        if h in ("activity name", "tab name", "name", "activity"):
            act_col = i
            break
    if not act_col:
        act_col = 1

    edge_col = None
    for i, h in enumerate(header_lower, 1):
        if h in ("edge cases", "edge case", "activity edge cases"):
            edge_col = i
            break

    guide_col = None
    for i, h in enumerate(header_lower, 1):
        if h in ("guidelines", "guideline", "activity guidelines"):
            guide_col = i
            break

    # If columns missing from header row, add them
    if clean_edge and not edge_col:
        edge_col = len(headers) + 1
        headers.append("Edge Cases")
        ws.update_cell(1, edge_col, "Edge Cases")

    if clean_guide and not guide_col:
        guide_col = len(headers) + 1
        headers.append("Guidelines")
        ws.update_cell(1, guide_col, "Guidelines")

    # Locate activity row (1-indexed)
    matched_row_idx = None
    target_name_clean = activity_name.strip().lower()

    for r_idx, row in enumerate(all_values[1:], start=2):
        row_act = (row[act_col - 1] if len(row) >= act_col else "").strip().lower()
        if row_act == target_name_clean:
            matched_row_idx = r_idx
            break

    if not matched_row_idx:
        for r_idx, row in enumerate(all_values[1:], start=2):
            row_act = (row[act_col - 1] if len(row) >= act_col else "").strip().lower()
            if row_act and (row_act in target_name_clean or target_name_clean in row_act):
                matched_row_idx = r_idx
                break

    final_edge_val = ""
    final_guide_val = ""

    def _format_rule_entry(text: str) -> str:
        text = text.strip()
        if not text:
            return ""
        if text.startswith("- ") or text.startswith("• ") or text.startswith("* "):
            return text
        return f"• {text}"

    if matched_row_idx:
        matched_row = all_values[matched_row_idx - 1]

        # Update Edge Cases
        if clean_edge and edge_col:
            curr_edge = (matched_row[edge_col - 1] if len(matched_row) >= edge_col else "").strip()
            formatted_edge = _format_rule_entry(clean_edge)
            if curr_edge:
                final_edge_val = f"{curr_edge}\n\n{formatted_edge}"
            else:
                final_edge_val = formatted_edge
            ws.update_cell(matched_row_idx, edge_col, final_edge_val)
            logger.info(f"[SheetManager] Updated Edge Cases for '{activity_name}' at row {matched_row_idx}.")

        # Update Guidelines
        if clean_guide and guide_col:
            curr_guide = (matched_row[guide_col - 1] if len(matched_row) >= guide_col else "").strip()
            formatted_guide = _format_rule_entry(clean_guide)
            if curr_guide:
                final_guide_val = f"{curr_guide}\n\n{formatted_guide}"
            else:
                final_guide_val = formatted_guide
            ws.update_cell(matched_row_idx, guide_col, final_guide_val)
            logger.info(f"[SheetManager] Updated Guidelines for '{activity_name}' at row {matched_row_idx}.")

    else:
        # Append new row
        max_cols = max(len(headers), act_col, edge_col or 0, guide_col or 0)
        new_row = [""] * max_cols
        new_row[act_col - 1] = activity_name
        if clean_edge and edge_col:
            final_edge_val = _format_rule_entry(clean_edge)
            new_row[edge_col - 1] = final_edge_val
        if clean_guide and guide_col:
            final_guide_val = _format_rule_entry(clean_guide)
            new_row[guide_col - 1] = final_guide_val

        ws.append_row(new_row, value_input_option="USER_ENTERED")
        logger.info(f"[SheetManager] Appended new row for '{activity_name}' with Edge Cases/Guidelines to '{ws.title}'.")

    sheet_url = f"https://docs.google.com/spreadsheets/d/{spreadsheet.id}/edit"
    return {
        "success": True,
        "activity_name": activity_name,
        "edge_case_added": bool(clean_edge),
        "guideline_added": bool(clean_guide),
        "edge_cases": final_edge_val,
        "guidelines": final_guide_val,
        "sheet_url": sheet_url,
    }


def sync_assignment_folder_to_registry(
    assignment_folder: str,
    activity_name: str = "System Identification",
    drive_folder_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Convenience function that reads an assignment run folder (grading_worksheet.csv and
    ingestion_manifest.json) and synchronizes all graded/submitted student records
    into the Central Google Sheet Registry tab for the activity.
    """
    import csv
    import json

    csv_path = os.path.join(assignment_folder, "grading_worksheet.csv")
    manifest_path = os.path.join(assignment_folder, "ingestion_manifest.json")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"grading_worksheet.csv not found in {assignment_folder}")

    # Load manifest data for media folders and evaluations
    student_drive_map: Dict[str, str] = {}
    manifest_eval_map: Dict[str, Dict[str, Any]] = {}
    run_drive_folder = ""

    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as mf:
                mdata = json.load(mf)
                run_drive_folder = mdata.get("drive_folder_url", "")
                for s in mdata.get("students", []):
                    s_id = (s.get("identifier") or "").strip().lower()
                    s_name = (s.get("full_name") or "").strip().lower()
                    df_url = s.get("drive_folder_url") or run_drive_folder
                    if s_id:
                        student_drive_map[s_id] = df_url
                    if s_name:
                        student_drive_map[s_name] = df_url

                for ev in mdata.get("evaluations", []):
                    ev_id = (ev.get("identifier") or "").strip().lower()
                    ev_name = (ev.get("full_name") or "").strip().lower()
                    if ev_id:
                        manifest_eval_map[ev_id] = ev
                    if ev_name:
                        manifest_eval_map[ev_name] = ev
        except Exception as man_err:
            logger.warning(f"Could not load manifest from {manifest_path}: {man_err}")

    records_to_sync: List[Dict[str, Any]] = []

    with open(csv_path, mode="r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            full_name = (row.get("Full name") or "").strip()
            identifier = (row.get("Identifier") or "").strip()
            status = (row.get("Status") or "").strip()
            grade = (row.get("Grade") or "").strip()
            online_text = (row.get("Online text") or "").strip()
            last_modified = (row.get("Last modified (submission)") or row.get("Last modified") or "").strip()
            feedback = (row.get("Feedback comments") or "").strip()

            # Skip students with no submissions and no grade
            if not grade and status.lower() in ("no submission", "reopened", "reopned"):
                continue

            s_key = full_name.lower()
            id_key = identifier.lower()

            media_folder = student_drive_map.get(id_key) or student_drive_map.get(s_key) or run_drive_folder

            ev_info = manifest_eval_map.get(id_key) or manifest_eval_map.get(s_key) or {}
            raw_grade = ev_info.get("raw_agent_grade") or grade
            checklist = ev_info.get("checklist_evaluations") or []

            records_to_sync.append({
                "name": full_name,
                "status": status,
                "grade": raw_grade,
                "online_text": online_text,
                "media_folder": media_folder,
                "last_modified": last_modified,
                "feedback_comment": feedback,
                "checklist_evaluations": checklist,
            })

    logger.info(f"Parsed {len(records_to_sync)} student record(s) from {assignment_folder} for registry sync.")
    return sync_evaluations_to_registry(
        activity_name=activity_name,
        evaluated_records=records_to_sync,
        drive_folder_id=drive_folder_id,
    )


def update_mentor_review_in_sheet(
    activity_name: str,
    student_name: str,
    attempt_number: int,
    new_grade: str,
    new_feedback: str,
    review_status: str,
    mentor_name: str,
    drive_folder_id: Optional[str] = None,
    assignment_folder: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Updates a student submission row with the human mentor's verdict and feedback.

    Args:
        activity_name: Target activity tab
        student_name: Name of the student
        attempt_number: The attempt number to update
        new_grade: 'Pass' or 'Fail' (or custom grade override)
        new_feedback: Human mentor's edited/reviewed feedback comment
        review_status: 'Approved by Mentor' or 'Overridden by Mentor'
        mentor_name: Name of human mentor
        drive_folder_id: Optional Drive folder ID
        assignment_folder: Optional local assignment run folder to mirror changes to grading_worksheet.csv
    """
    spreadsheet, sheet_url = get_or_create_registry_spreadsheet(drive_folder_id=drive_folder_id)
    ws = get_or_create_activity_worksheet(spreadsheet, activity_name)
    all_values = ws.get_all_values()

    if not all_values:
        raise ValueError(f"Worksheet '{activity_name}' is empty.")

    headers = [h.strip().lower() for h in all_values[0]]
    col_map = {h: idx + 1 for idx, h in enumerate(headers)}  # 1-indexed

    name_col = col_map.get("name", 1)
    attempt_col = col_map.get("attempt number", 6)
    grade_col = col_map.get("grade", 3)
    feedback_col = col_map.get("feedback comment", 8)
    review_status_col = col_map.get("review status", 11)
    mentor_reviewer_col = col_map.get("mentor reviewer", 12)
    mentor_reviewed_at_col = col_map.get("mentor reviewed at", 13)

    target_row_idx = None
    target_student_norm = student_name.strip().lower()
    target_att_str = str(attempt_number).strip()

    for r_idx, row_vals in enumerate(all_values[1:], start=2):
        if not row_vals:
            continue
        row_name = row_vals[name_col - 1].strip().lower() if len(row_vals) >= name_col else ""
        row_att = row_vals[attempt_col - 1].strip() if len(row_vals) >= attempt_col else "1"
        if row_name == target_student_norm and (row_att == target_att_str or not target_att_str):
            target_row_idx = r_idx
            break

    if not target_row_idx:
        raise ValueError(f"Student '{student_name}' attempt {attempt_number} not found in tab '{activity_name}'.")

    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    # Update cells in Google Sheet with formula sanitization
    cells_to_update = [
        {"range": gspread.utils.rowcol_to_a1(target_row_idx, grade_col), "values": [[sanitize_sheet_cell_value(new_grade)]]},
        {"range": gspread.utils.rowcol_to_a1(target_row_idx, feedback_col), "values": [[sanitize_sheet_cell_value(new_feedback)]]},
        {"range": gspread.utils.rowcol_to_a1(target_row_idx, review_status_col), "values": [[sanitize_sheet_cell_value(review_status)]]},
        {"range": gspread.utils.rowcol_to_a1(target_row_idx, mentor_reviewer_col), "values": [[sanitize_sheet_cell_value(mentor_name)]]},
        {"range": gspread.utils.rowcol_to_a1(target_row_idx, mentor_reviewed_at_col), "values": [[sanitize_sheet_cell_value(now_iso)]]},
    ]

    ws.batch_update(cells_to_update)
    logger.info(
        f"✅ [Mentor Review]: Updated row {target_row_idx} for '{student_name}' in tab '{activity_name}'. "
        f"Grade: '{new_grade}', Status: '{review_status}', Mentor: '{mentor_name}'."
    )

    # Mirror to local grading_worksheet.csv if available
    local_csv_updated = False
    if not assignment_folder:
        try:
            from agents.submission_reviewer.moodle_review_runner import find_latest_assignment_folder
            assignment_folder = find_latest_assignment_folder()
        except Exception:
            pass

    if assignment_folder and os.path.exists(os.path.join(assignment_folder, "grading_worksheet.csv")):
        try:
            import csv
            import unicodedata
            csv_path = os.path.join(assignment_folder, "grading_worksheet.csv")
            with open(csv_path, mode="r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                fnames = reader.fieldnames or []
                csv_rows = list(reader)

            target_norm = unicodedata.normalize("NFKD", target_student_norm)
            for crow in csv_rows:
                row_sname = unicodedata.normalize("NFKD", (crow.get("Full name") or "").strip().lower())
                if row_sname == target_norm:
                    crow["Grade"] = new_grade
                    crow["Feedback comments"] = new_feedback
                    local_csv_updated = True

            if local_csv_updated:
                with open(csv_path, mode="w", encoding="utf-8-sig", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fnames, quoting=csv.QUOTE_MINIMAL)
                    writer.writeheader()
                    writer.writerows(csv_rows)
                logger.info(f"📄 Mirrored mentor review to local CSV: {csv_path}")
        except Exception as local_err:
            logger.warning(f"Could not mirror review to local CSV: {local_err}")

    return {
        "success": True,
        "activity": activity_name,
        "row": target_row_idx,
        "student": student_name,
        "grade": new_grade,
        "status": review_status,
        "mentor": mentor_name,
        "reviewed_at": now_iso,
        "sheet_url": sheet_url,
        "local_csv_updated": local_csv_updated,
    }


def batch_approve_pending_reviews(
    activity_name: str,
    mentor_name: str,
    drive_folder_id: Optional[str] = None,
    assignment_folder: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Bulk approves all currently 'Pending Review' submissions that have Grade == 'Pass'
    and do not have 'unsure' in their Agent Grade. Also mirrors approval to local CSV if present.
    """
    spreadsheet, sheet_url = get_or_create_registry_spreadsheet(drive_folder_id=drive_folder_id)
    ws = get_or_create_activity_worksheet(spreadsheet, activity_name)
    all_values = ws.get_all_values()

    if not all_values or len(all_values) <= 1:
        return {"approved_count": 0, "sheet_url": sheet_url}

    headers = [h.strip().lower() for h in all_values[0]]
    col_map = {h: idx for idx, h in enumerate(headers)}

    name_idx = col_map.get("name", 0)
    grade_idx = col_map.get("grade", 2)
    agent_grade_idx = col_map.get("agent grade", 9)
    review_status_idx = col_map.get("review status", 10)
    mentor_col = col_map.get("mentor reviewer", 11) + 1  # 1-indexed
    status_col = col_map.get("review status", 10) + 1
    reviewed_at_col = col_map.get("mentor reviewed at", 12) + 1

    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    updates = []
    approved_count = 0
    approved_student_names = set()

    for r_idx, row in enumerate(all_values[1:], start=2):
        if not row:
            continue
        g = row[grade_idx].strip() if len(row) > grade_idx else ""
        ag = row[agent_grade_idx].strip() if len(row) > agent_grade_idx else g
        status = row[review_status_idx].strip() if len(row) > review_status_idx else ""
        sname = row[name_idx].strip() if len(row) > name_idx else ""

        is_pending = not status or status.lower() == "pending review"
        is_confident_pass = "pass" in g.lower() and "unsure" not in ag.lower()

        if is_pending and is_confident_pass:
            # Batch update contiguous columns K to M (Review status, Mentor reviewer, Mentor reviewed at)
            range_k_to_m = f"K{r_idx}:M{r_idx}"
            updates.append({
                "range": range_k_to_m,
                "values": [["Approved by Mentor", sanitize_sheet_cell_value(mentor_name), now_iso]]
            })
            approved_count += 1
            if sname:
                approved_student_names.add(sname.lower())

    if updates:
        ws.batch_update(updates)
        logger.info(f"⚡ Batch approved {approved_count} pending pass submissions in tab '{activity_name}'.")

    # Mirror approved statuses to local CSV if available
    if approved_student_names:
        if not assignment_folder:
            try:
                from agents.submission_reviewer.moodle_review_runner import find_latest_assignment_folder
                assignment_folder = find_latest_assignment_folder()
            except Exception:
                pass

        if assignment_folder and os.path.exists(os.path.join(assignment_folder, "grading_worksheet.csv")):
            try:
                import csv
                import unicodedata
                csv_path = os.path.join(assignment_folder, "grading_worksheet.csv")
                with open(csv_path, mode="r", encoding="utf-8-sig") as f:
                    reader = csv.DictReader(f)
                    fnames = reader.fieldnames or []
                    csv_rows = list(reader)

                csv_changed = False
                for crow in csv_rows:
                    r_norm = unicodedata.normalize("NFKD", (crow.get("Full name") or "").strip().lower())
                    if r_norm in approved_student_names:
                        # Confirmed pass in CSV
                        crow["Grade"] = "Pass"
                        csv_changed = True

                if csv_changed:
                    with open(csv_path, mode="w", encoding="utf-8-sig", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=fnames, quoting=csv.QUOTE_MINIMAL)
                        writer.writeheader()
                        writer.writerows(csv_rows)
                    logger.info(f"📄 Mirrored batch approvals to local CSV: {csv_path}")
            except Exception as batch_csv_err:
                logger.warning(f"Could not mirror batch approvals to local CSV: {batch_csv_err}")

    return {
        "approved_count": approved_count,
        "sheet_url": sheet_url,
    }
