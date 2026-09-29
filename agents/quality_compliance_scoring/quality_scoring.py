
import os
import re
import shutil
import pandas as pd
import gspread
from tqdm import tqdm
import streamlit as st
from typing import List, Tuple
from datetime import datetime
from collections import defaultdict
from googleapiclient.discovery import build
from services.smart_progress_bar import SmartProgressBar
from concurrent.futures import ThreadPoolExecutor, as_completed
from googleapiclient.http import MediaFileUpload
from services.sheets_service import create_or_read_worksheet, save_to_sheet, format_worksheet, clear_worksheet, get_sheet_data_and_df
from agents.quality_compliance_scoring.generate_llm_feedback import generate_llm_feedback_from_issues, generate_rephrased_flagged_items
from agents.quality_compliance_scoring.helpers import extract_course_name, get_stage_sheets
from agents.quality_compliance_scoring.create_review_report_doc import create_course_review_doc
GOOGLE_API_SCOPES = [
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/documents",
]

GOOGLE_DOC_MIME = "application/vnd.google-apps.document"
DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


def _resolve_credentials():
    """Fetch ready-to-use OAuth credentials from session or PyDrive auth."""
    creds = st.session_state.get("google_credentials")
    if creds:
        return creds

    py_drive = st.session_state.get("drive")
    if py_drive:
        auth = getattr(py_drive, "auth", None)
        creds = getattr(auth, "credentials", None) if auth else None
        if creds:
            st.session_state["google_credentials"] = creds
            return creds
    return None


def _initialize_google_clients_from_session():
    """
    Ensure Drive/Docs/Sheets/gspread clients exist in Streamlit session_state.
    Falls back to building clients from stored google_credentials.
    """
    credentials = _resolve_credentials()

    drive_client = st.session_state.get("drive_v3")
    if not drive_client and credentials:
        drive_client = build("drive", "v3", credentials=credentials)
        st.session_state["drive_v3"] = drive_client

    docs_client = st.session_state.get("docs")
    if not docs_client and credentials:
        docs_client = build("docs", "v1", credentials=credentials)
        st.session_state["docs"] = docs_client

    sheets_client = st.session_state.get("sheets")
    if not sheets_client and credentials:
        sheets_client = build("sheets", "v4", credentials=credentials)
        st.session_state["sheets"] = sheets_client

    gspread_client = (
        st.session_state.get("gspread_client")
        or st.session_state.get("gc")
    )
    if not gspread_client and credentials:
        gspread_client = gspread.authorize(credentials)
        st.session_state["gspread_client"] = gspread_client

    if not gspread_client:
        raise RuntimeError(
            "Google clients not available. Please authenticate in the agent UI first."
        )

    return gspread_client, drive_client, docs_client, sheets_client


try:
    gc, drive_service, docs_service, sheets_service = _initialize_google_clients_from_session()
except RuntimeError as auth_err:
    gc = drive_service = docs_service = sheets_service = None
    print(f"⚠️ {auth_err}")


def normalize_header_text(value: str) -> str:
    """Collapse whitespace in header labels so regex patterns are easier to match."""
    if not value:
        return ""
    return re.sub(r"\s+", " ", value).strip()


def extract_entity_name(header: str, label: str) -> str:
    """Extract the entity name that follows a label (Reviewer/Reporter/etc.) in a header."""
    if not header:
        return ""

    bracket_match = re.search(rf"{label}[^\[]*\[\s*(.+?)\s*\]", header, re.I)
    if bracket_match:
        return bracket_match.group(1).strip()

    header_flat = normalize_header_text(header)

    colon_match = re.search(rf"{label}\s*[:\-]\s*(.+)", header_flat, re.I)
    if colon_match:
        return colon_match.group(1).strip(" :-")

    idx = header_flat.lower().find(label.lower())
    if idx != -1:
        tail = header_flat[idx + len(label):].strip(" :-")
        if tail:
            return tail

    return ""


def fetch_row_value(row: dict, header: str):
    """
    Safely read a value from a gspread row dict, handling duplicated header names
    where gspread appends suffixes such as '.1' or '_2'.
    """
    if not header:
        return None

    if header in row:
        return row[header]

    normalized_target = normalize_header_text(header).lower()
    fallback = None
    for key, value in row.items():
        normalized_key = normalize_header_text(key).lower()
        if normalized_key == normalized_target:
            return value
        if fallback is None and key.startswith(header):
            suffix_index = len(header)
            if len(key) > suffix_index and key[suffix_index] in {'.', '_', ' '}:
                fallback = value
    return fallback


def map_stage_columns(header_row: List[str]):
    """
    Build lookups for reviewer, reporter, and comment columns per topic so we don't
    miss any data due to slightly different header formats.
    """
    topic_columns = []
    reporter_lookup = {}
    comment_columns = defaultdict(list)

    for header in header_row:
        if not header:
            continue

        normalized = normalize_header_text(header)
        topic_match = re.search(r"Topic\s+(\d+)", normalized, re.I)
        if not topic_match:
            continue

        topic_number = topic_match.group(1).strip()
        lowered = normalized.lower()

        if "reviewer" in lowered:
            reviewer_name = extract_entity_name(header, "Reviewer") or "Unknown Reviewer"
            topic_columns.append({
                "header": header,
                "topic": topic_number,
                "reviewer": reviewer_name
            })

        if "reporter" in lowered:
            reporter_name = extract_entity_name(header, "Reporter")
            if reporter_name:
                reporter_lookup.setdefault(topic_number, reporter_name)

        if "comment" in lowered:
            comment_columns[topic_number].append(header)

    return topic_columns, reporter_lookup, dict(comment_columns)


def collect_topic_comments(rows: List[dict], comment_column_pairs: List[Tuple[str, str]]) -> dict:
    """
    Gather every comment per topic (across all reviewers) so Quality Summary prompts
    always receive the full context.
    """
    topic_comments = defaultdict(list)
    invalid_tokens = {"", "-", "na", "n/a", "none"}

    for row in rows:
        for topic_number, header in comment_column_pairs:
            comment_cell = fetch_row_value(row, header)
            if comment_cell is None:
                continue
            comment_text = str(comment_cell).strip()
            if not comment_text:
                continue
            if comment_text.strip().lower() in invalid_tokens:
                continue
            topic_comments[topic_number].append(comment_text)

    return dict(topic_comments)
    
    
def parse_quality_summary_sections(text: str) -> List[dict]:
    """Parse the LLM quality summary into structured sections."""
    sections = []
    if text:
        pattern = re.compile(
            r"\d+\.\s+(?P<header>[^\n]+)\nComments Grouped:\s*(?P<comments>.*?)(?:\nActionable Summary:\s*(?P<action>.*?))?(?=\n\d+\.\s+|\Z)",
            re.S
        )
        for idx, match in enumerate(pattern.finditer(text.strip()), 1):
            header = match.group("header").strip()
            topics_match = re.search(r"\[(.*?)\]", header)
            topics = topics_match.group(1).strip() if topics_match else ""
            category = re.sub(r"\[.*?\]", "", header).strip() or f"Category {idx}"
            comments_block = (match.group("comments") or "").strip()
            comments = [ln.strip() for ln in comments_block.splitlines() if ln.strip()]
            actionable = (match.group("action") or "").strip() or "No actionable summary provided."
            sections.append({
                "category": category,
                "topics": topics,
                "comments": comments,
                "actionable": actionable
            })
    return sections


def format_quality_summary_for_sheet(sections: List[dict]) -> str:
    """Convert structured sections back into template text for the Issue Log."""
    if not sections:
        return ""
    lines = []
    for idx, section in enumerate(sections, 1):
        header = f"{idx}. {section.get('category', f'Category {idx}')}"
        if section.get("topics"):
            header += f" [{section['topics']}]"
        lines.append(header)
        lines.append("Comments Grouped:")
        if section.get("comments"):
            lines.extend(section["comments"])
        else:
            lines.append("No comments provided.")
        lines.append("Actionable Summary:")
        lines.append(section.get("actionable") or "No actionable summary provided.")
        lines.append("")
    return "\n".join(lines).strip()


def derive_category_name(header: str) -> str:
    """Create a readable category from any comments column header."""
    cleaned = re.sub(r'(?i)comments', '', header)
    cleaned = cleaned.replace('\n', ' ')
    cleaned = re.sub(r'\s+', ' ', cleaned).strip(" :-")
    return cleaned or "Comments"



def format_quality_summary_for_sheet(sections: List[dict]) -> str:
    """Convert structured sections back into template text for the sheet."""
    if not sections:
        return ""
    lines = []
    for idx, section in enumerate(sections, 1):
        header = f"{idx}. {section.get('category', f'Category {idx}')}"
        if section.get("topics"):
            header += f" [{section['topics']}]"
        lines.append(header)
        lines.append("Comments Grouped:")
        if section.get("comments"):
            lines.extend(section["comments"])
        else:
            lines.append("No comments provided.")
        lines.append("Actionable Summary:")
        lines.append(section.get("actionable") or "No actionable summary provided.")
        lines.append("")
    return "\n".join(lines).strip()



def process_and_save_stage(sheet, spreadsheet):
    print(f"\nProcessing stage: {sheet.title}")
    try:
        print("Reading all records from sheet...")
        data = sheet.get_all_records()
        print(f"Loaded {len(data)} rows successfully.")
    except Exception as e:
        print(f"Error loading data from stage '{sheet.title}': {e}")
        try:
            header_row = sheet.row_values(1)
            print(f"Header row retrieved ({len(header_row)} columns): {header_row}")
        except Exception as header_error:
            print(f"Failed to read header row from '{sheet.title}': {header_error}")
        return None

    try:
        print("Fetching header row...")
        header_row = sheet.row_values(1)
        print(f"Header row contains {len(header_row)} columns.")
    except Exception as e:
        print(f"Error reading header row in '{sheet.title}': {e}")
        return None

    # Identify topic/reviewer/comment columns with flexible parsing
    topic_columns, reporter_lookup, comment_columns = map_stage_columns(header_row)
    print(f"Found {len(topic_columns)} reviewer topic columns: {[col['header'] for col in topic_columns]}")

    if len(topic_columns) == 0:
        print(f"Skipping stage '{sheet.title}' — No reviewer columns found.")
        return None

    comment_column_pairs = [
        (topic, header)
        for topic, headers in comment_columns.items()
        for header in headers
    ]
    topic_comment_cache = collect_topic_comments(data, comment_column_pairs) if comment_column_pairs else {}

    # ───────────────────────────────────────────────
    # CREATOR → DATA BUCKET
    # ───────────────────────────────────────────────
    creators = {}  # creator_name → bucket

    try:
        for column_info in topic_columns:
            col_header = column_info["header"]
            topic_number = column_info["topic"]
            reviewer = column_info["reviewer"]

            creator = reporter_lookup.get(topic_number)
            if not creator:
                creator = f"Unassigned Topic {topic_number}"
                print(f"⚠️ No reporter column found for Topic {topic_number} in '{sheet.title}'. Using placeholder '{creator}'.")

            # Initialize bucket
            if creator not in creators:
                creators[creator] = {
                    "reviewers": set(),
                    "topics": set(),
                    "basic_total": 0,
                    "critical_total": 0,
                    "basic_issues": [],
                    "critical_issues": [],
                    "comments": {},  # topic → [comments]
                }

            bucket = creators[creator]
            bucket["reviewers"].add(reviewer)
            bucket["topics"].add(topic_number)

            if topic_number not in bucket["comments"] and topic_number in topic_comment_cache:
                bucket["comments"][topic_number] = list(topic_comment_cache[topic_number])

            # ───────────────────────────────────────────────
            # MAIN CHECKLIST ITEM PROCESSING
            # ───────────────────────────────────────────────
            for row in data:

                # Checklist values
                criteria = row.get("Checklist Criteria", "").strip()
                task_name = row.get("Task", "").strip() or "Uncategorized"
                item_type = str(row.get("Checklist Item Type", "")).strip().lower()
                reviewer_raw = fetch_row_value(row, col_header)
                reviewer_status = str(reviewer_raw).strip().lower() if reviewer_raw is not None else ""

                # Count items, issues — SAME LOGIC AS ORIGINAL
                if criteria and reviewer_status in ["true", "false"]:
                    is_checked = reviewer_status == "true"

                    if item_type == "basic":
                        bucket["basic_total"] += 1
                        if not is_checked:
                            # now include task name so compliance category = task name
                            bucket["basic_issues"].append(
                                f"{task_name} :: {criteria} (Topic {topic_number})"
                            )

                    elif item_type == "critical":
                        bucket["critical_total"] += 1
                        if not is_checked:
                            # now include task name so compliance category = task name
                            bucket["critical_issues"].append(
                                f"{task_name} :: {criteria} (Topic {topic_number})"
                            )

    except Exception as e:
        print(f"Error while processing topics or rows in '{sheet.title}': {e}")
        import traceback
        traceback.print_exc()
        return None

    # ───────────────────────────────────────────────
    # BUILD FINAL OUTPUT (ONE ROW PER CREATOR)
    # ───────────────────────────────────────────────
    results = []
    course_name = extract_course_name(spreadsheet.title)

    for creator_name, bucket in creators.items():

        reviewer_name = ", ".join(sorted(bucket["reviewers"]))

        # Build grouped comments EXACTLY like LLM expects
        combined_comments = ""
        def _topic_sort_key(value):
            try:
                return (0, int(value))
            except (TypeError, ValueError):
                return (1, str(value))

        for topic_number in sorted(bucket["comments"].keys(), key=_topic_sort_key):
            clist = bucket["comments"][topic_number]
            if clist:
                combined_comments += (
                    f"\n\n### Topic {topic_number}\n" +
                    "\n".join(f"- {c}" for c in clist)
                )

        combined_comments = combined_comments.strip()

        # SAME SCORING LOGIC AS ORIGINAL
        total_basic = bucket["basic_total"]
        total_critical = bucket["critical_total"]
        total_basic_issues = len(bucket["basic_issues"])
        total_critical_issues = len(bucket["critical_issues"])

        basic_quality = 100.0 if total_basic == 0 else round((1 - total_basic_issues / total_basic) * 100, 2)
        critical_quality = 100.0 if total_critical == 0 else round((1 - total_critical_issues / total_critical) * 100, 2)

        print("Generating LLM feedback summaries...")

        basic_feedback = generate_llm_feedback_from_issues(
            bucket["basic_issues"], "basic", sheet.title, creator_name, course_name
        ) or {}

        critical_feedback = generate_llm_feedback_from_issues(
            bucket["critical_issues"], "critical", sheet.title, creator_name, course_name
        ) or {}

        quality_feedback = generate_llm_feedback_from_issues(
            [], "quality_overall", sheet.title, creator_name, course_name,
            comments=combined_comments
        ) or {}

        results.append({
            "Creator Name": creator_name,
            "Course Name": course_name,
            "Topics Reviewed": len(bucket["topics"]),
            "Stage": sheet.title,
            "Reviewer Name": reviewer_name,

            "Basic Issues": "\n".join(bucket["basic_issues"]),
            "Total Basic Checklist Items": total_basic,

            "Critical Issues": "\n".join(bucket["critical_issues"]),
            "Total Critical Checklist Items": total_critical,

            "Basic Quality %": f"{basic_quality}%",
            "Critical Quality %": f"{critical_quality}%",

            "Compliance Summary - Basic": basic_feedback.get("Compliance Summary - Basic", ""),
            "Compliance Summary - Critical": critical_feedback.get("Compliance Summary - Critical", ""),
            "Quality Summary": quality_feedback.get("Quality Summary", ""),

            "Comments": combined_comments
        })

    print(f"Finished processing stage '{sheet.title}' successfully.")
    return results





def create_or_get_course_folder(drive, course_name, course_folder_id):
    """Ensure 'Course Review Report - {course_name}' folder exists under given parent."""
    folder_name = f"Course Review Report - {course_name}"

    # Search if folder already exists under the parent
    query = (
        f"name = '{folder_name}' and mimeType = 'application/vnd.google-apps.folder' "
        f"and '{course_folder_id}' in parents and trashed = false"
    )
    response = drive.files().list(q=query, fields="files(id, name)").execute()
    files = response.get("files", [])

    if files:
        folder_id = files[0]["id"]
        print(f"📁 Using existing folder: {folder_name} ({folder_id})")
    else:
        metadata = {
            "name": folder_name,
            "mimeType": "application/vnd.google-apps.folder",
            "parents": [course_folder_id]
        }
        file = drive.files().create(body=metadata, fields="id").execute()
        folder_id = file.get("id")
        print(f"✅ Created folder: {folder_name} ({folder_id})")

    return folder_id


def create_review_report_from_template(
    result_dict,
    spreadsheet_url,
    drive,
    course_folder_id
):
    """
    Build a DOCX report locally, upload it to Drive,
    and overwrite an existing report if present.
    """

    docx_path, metadata = create_course_review_doc(result_dict, spreadsheet_url)
    if not docx_path:
        return None

    folder_id = create_or_get_course_folder(
        drive,
        result_dict.get("Course Name", "Unnamed Course"),
        course_folder_id
    )

    # ----------------------------------------------------
    # 🔍 CHECK IF THE FILE ALREADY EXISTS IN THIS FOLDER
    # ----------------------------------------------------
    drive_name = metadata.get("drive_name", "Course Review Report")
    query = (
        f"name = '{drive_name}' "
        f"and '{folder_id}' in parents "
        f"and mimeType = '{GOOGLE_DOC_MIME}' "
        f"and trashed = false"
    )
    existing_files = drive.files().list(
        q=query,
        fields="files(id, name)"
    ).execute().get("files", [])

    media = MediaFileUpload(docx_path, mimetype=DOCX_MIME, resumable=False)

    try:
        if existing_files:
            # ----------------------------------------------------
            # ✏️ OVERWRITE THE EXISTING Google Doc
            # ----------------------------------------------------
            file_id = existing_files[0]["id"]
            print(f"✏️ Overwriting existing report: {drive_name} ({file_id})")

            updated = drive.files().update(
                fileId=file_id,
                media_body=media
            ).execute()

            return f"https://docs.google.com/document/d/{file_id}/edit"

        else:
            # ----------------------------------------------------
            # 🆕 CREATE A NEW DOC
            # ----------------------------------------------------
            body = {
                "name": drive_name,
                "parents": [folder_id],
                "mimeType": GOOGLE_DOC_MIME,
                "description": f"Generated from {spreadsheet_url}",
            }

            created = drive.files().create(
                body=body,
                media_body=media,
                fields="id, webViewLink"
            ).execute()
            doc_id = created.get("id")
            link = created.get("webViewLink") or (f"https://docs.google.com/document/d/{doc_id}/edit")

            print(f"📄 Uploaded new Google Doc for {drive_name}: {link}")
            return link

    finally:
        temp_dir = metadata.get("temp_dir")
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)



def run_update_quality_scores(spreadsheet, course_folder_id, spreadsheet_url):
    if gc is None:
        raise RuntimeError(
            "Google clients not initialized. Please sign in with Google OAuth before running quality scoring."
        )

    print("Syncing to Task Logs...")

    source_sheet_id = spreadsheet.id
    course_name = extract_course_name(spreadsheet.title)
    task_logs_sheet_id = "1LSdFMKnRCr6sdukD-12bIJtYTigFVZfNzNxib0Yqebc"

    stage_sheets = get_stage_sheets(spreadsheet)
    stage_order = [sheet.title for sheet in stage_sheets]
    results = []

    def process_stage(sheet):
        try:
            print(f"\nStarting processing for stage: {sheet.title}")
            result = process_and_save_stage(sheet, spreadsheet)
            return sheet.title, result
        except Exception as e:
            print(f"Error processing stage '{sheet.title}': {e}")
            import traceback
            traceback.print_exc()
            return sheet.title, None

    print("Processing stages in parallel...")
    futures_map = {}
    with ThreadPoolExecutor(max_workers=5) as executor:
        for sheet in stage_sheets:
            future = executor.submit(process_stage, sheet)
            futures_map[future] = sheet.title

        progress = SmartProgressBar(total_tasks=len(futures_map), description="Percent Complete", save_interval=0)
        for future in tqdm(as_completed(futures_map), total=len(futures_map)):
            sheet_title, result = future.result()
            if result:
                # ✅ result is now a list (one per reporter)
                if isinstance(result, list):
                    for r in result:
                        results.append((sheet_title, r))
                else:
                    results.append((sheet_title, result))
            else:
                print(f"Skipped stage due to errors: {sheet_title}")
            progress.update()

    print("Stage processing complete. Writing results to sheets...")

    # ✅ Sort and flatten results
    results.sort(key=lambda x: stage_order.index(x[0]) if x[0] in stage_order else 999)
    issue_data = [r[1] for r in results]
    issue_df = pd.DataFrame(issue_data)

    # --- Clean data before writing to avoid APIError ---
    issue_df = issue_df.fillna("").astype(str)

    # --- Write clean table to Issue Log sheet ---
    issue_ws, _ = create_or_read_worksheet(spreadsheet, "Issue Log")
    clear_worksheet(issue_ws)
    values = [issue_df.columns.tolist()] + issue_df.values.tolist()
    issue_ws.update(values)  # ✅ correctly writes rows/columns, not dicts
    format_worksheet(issue_ws)

    print("✅ Issue Log updated successfully.")

    # --- Update Task Logs ---
    try:
        tracker = gc.open_by_key(task_logs_sheet_id)
        print("Opened Task Logs tracker successfully.")
    except Exception as e:
        raise RuntimeError(
            f"Failed to open Task Logs sheet {task_logs_sheet_id}") from e


    
    task_ws, task_df = get_sheet_data_and_df(tracker, "Task Logs")

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_rows = []

    headers = [
        "sheet id", "Creator Name", "Course Name", "Topics Reviewed", "Stage",
        "Basic Quality %", "Critical Quality %", "Compliance Summary - Basic",
        "Compliance Summary - Critical", "Quality Summary", "timestamp"
    ]

    if task_df.empty or list(task_df.columns) != headers:
        print("Writing headers to Task Logs sheet...")
        task_df = pd.DataFrame(columns=headers)

    for _, row in issue_df.iterrows():
        new_entry = {
            "sheet id": source_sheet_id,
            "Creator Name": row["Creator Name"],
            "Course Name": course_name,
            "Topics Reviewed": row["Topics Reviewed"],
            "Stage": row["Stage"],
            "Basic Quality %": row["Basic Quality %"],
            "Critical Quality %": row["Critical Quality %"],
            "Compliance Summary - Basic": row["Compliance Summary - Basic"],
            "Compliance Summary - Critical": row["Compliance Summary - Critical"],
            "Quality Summary": row["Quality Summary"],
            "timestamp": now
        }

        if not task_df.empty:
            match = (
                (task_df["sheet id"] == source_sheet_id) &
                (task_df["Creator Name"] == row["Creator Name"]) &
                (task_df["Course Name"] == course_name) &
                (task_df["Stage"] == row["Stage"])
            )
        else:
            match = pd.Series([False] * len(task_df))

        if match.any():
            task_df.loc[match, list(new_entry.keys())[2:]] = list(new_entry.values())[2:]
            print(f"Updated task log for {row['Creator Name']}")
        else:
            new_rows.append(new_entry)
            print(f"Added task log for {row['Creator Name']}")

    if new_rows:
        task_df = pd.concat([task_df, pd.DataFrame(new_rows)], ignore_index=True)

    task_df = task_df.fillna("").astype(str)
    task_values = [task_df.columns.tolist()] + task_df.values.tolist()
    clear_worksheet(task_ws)
    task_ws.update(task_values)
    print("✅ Task Logs sheet updated.")

    # --- Create Google Docs reports in Drive ---
    print("📂 Creating Google Docs reports")
    
    # --- Generate Review Reports ---
    print("\nGenerating Course Review Reports...")
    issue_df = generate_rephrased_flagged_items(issue_df)

    drive = st.session_state.get("drive_v3") or drive_service
    print(f"Drive object: {drive}")
    if not drive:
        print("❌ Google Drive client unavailable. Skipping Doc creation.")
        return

    for _, row in issue_df.iterrows():
        result_dict = row.to_dict()
        try:
            link = create_review_report_from_template(
                result_dict=result_dict,
                spreadsheet_url=spreadsheet_url,
                drive=drive,
                course_folder_id=course_folder_id
            )
            if link:
                print(f"✅ Created report for {result_dict['Creator Name']}: {link}")
        except Exception as e:
            print(f"⚠️ Failed to create report for {result_dict['Creator Name']}: {e}")