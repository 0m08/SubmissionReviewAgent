import os
import re
import json
import base64
import gspread
import pandas as pd
from tqdm import tqdm
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv
from modules.chain import Chain
from typing import List, Optional
from pydrive2.drive import GoogleDrive
from services.smart_progress_bar import SmartProgressBar
from services.drive_service import login_with_service_account
from concurrent.futures import ThreadPoolExecutor, as_completed
from services.sheets_service import create_or_read_worksheet, save_to_sheet, format_worksheet, clear_worksheet, get_sheet_data_and_df

load_dotenv()
key_bytes = base64.b64decode(os.environ["GDRIVE_SA_B64"])
sa_json = key_bytes.decode()
sa_dict = json.loads(sa_json)
gauth = login_with_service_account(json_str=sa_json)
gauth.ServiceAuth()
drive = GoogleDrive(gauth)
gc = gspread.service_account_from_dict(sa_dict)


generate_llm_feedback_prompt_template = """
You are an instructional design reviewer helping a course creator improve their course materials.

The following issues were found in the **{issue_type}** checklist items for the **{stage}** stage of the course **{course}**, created by **{creator}**.

Here is the list of problematic checklist items:
{issues}

For each checklist item above, do the following:
- Extract a short 2–4 word heading that summarizes the main idea
- Write helpful, friendly, and clear feedback under that heading
- Use this exact format for every item:

<Short Heading>:
<Specific, constructive feedback in paragraph form>

Write feedback in plain English that directly helps the creator improve this checklist item.
Be direct but supportive. Avoid repeating the full checklist item.
"""


def generate_llm_feedback_from_issues(
    issues: List[str],
    issue_type: str,
    stage: str,
    creator: str,
    course: str,
    llm: str = "gemini_2_flash"
) -> Optional[str]:
    if not issues:
        print(f"No {issue_type} issues to generate feedback.")
        return None

    agent = Chain(llm=llm) 

    formatted_prompt = generate_llm_feedback_prompt_template.format(
    creator=creator,
    course=course,
    stage=stage,
    issue_type=issue_type,
    issues="\n".join(f"- {i}" for i in issues)
)


    agent.add_message(role='user', content=formatted_prompt)

    try:
        response = agent.run()
        print(f"LLM raw response:\n{response}")

        if isinstance(response, str):
            return response.strip()
        else:
            print("Unexpected LLM response format.")
            return None

    except Exception as e:
        print(f"LLM failed to generate feedback: {e}")
        return None


def extract_course_name(title: str) -> str:
    """    
    Extracts the course name from a given title string.
    The course name is assumed to be the part of the title after the last colon.
    :param title: The title string from which to extract the course name.
    :return: The extracted course name, stripped of leading and trailing whitespace.
    """
    parts = title.split(":")
    return parts[-1].strip() if len(parts) > 1 else title.strip()


def get_stage_sheets(spreadsheet):
    """
    Retrieves all stage sheets from the provided spreadsheet.
    :param spreadsheet: The spreadsheet object containing multiple worksheets.
    :return: A list of stage sheets that contain 'Checklist Criteria' and 'Task' in their headers.
    """
    stage_sheets = []
    for sheet in spreadsheet.worksheets():
        headers = sheet.row_values(1)
        if 'Checklist Criteria' in headers and 'Task' in headers:
            stage_sheets.append(sheet)
    print(f"Found {len(stage_sheets)} stage sheets.")
    return stage_sheets


def process_and_save_stage(sheet, spreadsheet):
    print(f"\nProcessing stage: {sheet.title}")
    data = sheet.get_all_records()
    header_row = sheet.row_values(1)

    topic_columns = [
        col for col in header_row
        if re.search(r'Topic\s+\d+\s*\nReviewer\s*\n\[\s*.+?\s*\]', col.strip())
    ]
    print(f"Found {len(topic_columns)} reviewer topic columns.")

    if len(topic_columns) == 0:
        print(f"Skipping stage '{sheet.title}' — No reviewer columns found.")
        return None

    total_basic = total_critical = total_basic_issues = total_critical_issues = 0
    all_basic_issue_texts, all_critical_issue_texts = [], []
    topic_count = 0
    creator_name = None

    for col in topic_columns:
        topic_match = re.search(r'Topic\s+(\d+)\s*\nReviewer\s*\n\[\s*(.+?)\s*\]', col.strip())
        if not topic_match:
            continue

        topic_number, reviewer = topic_match.groups()
        creator = None
        for header in header_row:
            creator_match = re.search(fr'Topic\s+{topic_number}\s*\nReporter\s*\n\[\s*(.+?)\s*\]', header.strip())
            if creator_match:
                creator = creator_match.group(1)
                break

        if not creator:
            print(f"Skipping topic {topic_number} — no creator found.")
            continue

        if not creator_name:
            creator_name = creator

        topic_count += 1

        for row in data:
            item_type = str(row.get('Checklist Item Type', '')).strip().lower()
            criteria = row.get('Checklist Criteria', '')
            reviewer_status = row.get(col)
            is_checked = str(reviewer_status).strip().lower() == 'true'

            if item_type == 'basic':
                total_basic += 1
                if not is_checked:
                    total_basic_issues += 1
                    all_basic_issue_texts.append(criteria)
            elif item_type == 'critical':
                total_critical += 1
                if not is_checked:
                    total_critical_issues += 1
                    all_critical_issue_texts.append(criteria)

    basic_quality = 100.0 if total_basic == 0 else round((1 - total_basic_issues / total_basic) * 100, 2)
    critical_quality = 100.0 if total_critical == 0 else round((1 - total_critical_issues / total_critical) * 100, 2)

    course_name = extract_course_name(spreadsheet.title)

    basic_feedback = generate_llm_feedback_from_issues(
        all_basic_issue_texts, "basic", sheet.title, creator_name, course_name
    ) or ""
    critical_feedback = generate_llm_feedback_from_issues(
        all_critical_issue_texts, "critical", sheet.title, creator_name, course_name
    ) or ""

    return {
        "Creator Name": creator_name,
        "Course Name": course_name,
        "Topics Reviewed": topic_count,
        "Stage": sheet.title,
        "Basic Issues": "\n".join(all_basic_issue_texts),
        "Total Basic Checklist Items": total_basic,
        "Critical Issues": "\n".join(all_critical_issue_texts),
        "Total Critical Checklist Items": total_critical,
        "Basic Quality %": f"{basic_quality}%",
        "Critical Quality %": f"{critical_quality}%",
        "LLM Feedback - Basic": basic_feedback.strip(),
        "LLM Feedback - Critical": critical_feedback.strip()
    }


def run_update_quality_scores(spreadsheet):
    print("Syncing to Task Logs...")

    source_sheet_id = spreadsheet.id
    course_name = extract_course_name(spreadsheet.title)
    task_logs_sheet_id = "1LSdFMKnRCr6sdukD-12bIJtYTigFVZfNzNxib0Yqebc"

    # STEP 1: Get all stage sheets
    stage_sheets = get_stage_sheets(spreadsheet)
    stage_order = [sheet.title for sheet in stage_sheets]

    # STEP 2: Process stages in parallel and collect data
    results = []

    def process_stage(sheet):
        return sheet.title, process_and_save_stage(sheet, spreadsheet)

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
                results.append((sheet_title, result))
            progress.update()

    # STEP 3: Sort results by stage order and write once
    results.sort(key=lambda x: stage_order.index(x[0]))
    issue_data = [r[1] for r in results]
    issue_df = pd.DataFrame(issue_data)

    issue_ws, _ = create_or_read_worksheet(spreadsheet, "Issue Log")
    clear_worksheet(issue_ws)
    issue_ws.append_row(issue_df.columns.tolist())
    issue_ws.append_rows(issue_df.values.tolist())
    format_worksheet(issue_ws)

    # STEP 4: Sync to Task Logs
    tracker = gc.open_by_key(task_logs_sheet_id)
    task_ws, task_df = get_sheet_data_and_df(tracker, "Task Logs")

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_rows = []

    headers = [
        "sheet id", "Creator Name", "Course Name", "Topics Reviewed", "Stage",
        "Basic Quality %", "Critical Quality %", "LLM Feedback - Basic",
        "LLM Feedback - Critical", "timestamp"
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
            "LLM Feedback - Basic": row["LLM Feedback - Basic"],
            "LLM Feedback - Critical": row["LLM Feedback - Critical"],
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

    save_to_sheet(task_ws, task_df)
    print("Task Logs sheet updated.")


