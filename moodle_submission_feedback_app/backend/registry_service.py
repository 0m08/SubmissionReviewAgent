# -*- coding: utf-8 -*-
"""Service layer for Moodle Submission Review Registry operations."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional
import pandas as pd

from agents.submission_reviewer.central_sheet_manager import (
    DEFAULT_DRIVE_FOLDER_ID,
    REGISTRY_SHEET_ID_KNOWN,
    batch_approve_pending_reviews as sheet_batch_approve,
    fetch_all_registry_activities,
    get_or_create_registry_spreadsheet,
    sync_assignment_folder_to_registry as sheet_sync_folder,
    update_mentor_review_in_sheet as sheet_update_review,
)
from moodle_submission_feedback_app.backend.schemas import (
    ActivitySummary,
    ChecklistItem,
    SubmissionItem,
)


def parse_checklist_items(checklist_str: str) -> List[ChecklistItem]:
    """
    Parses multi-line checklist text into structured ChecklistItem objects.
    Derives pass/fail from [...] brackets (e.g. [PASSED], [FAILED], [PASS], [FAIL], [✓ PASS], [✗ FAIL]),
    removes the [...] bracket and bullets from instruction text, and extracts comments cleanly.
    """
    if not checklist_str or not isinstance(checklist_str, str):
        return []

    items = []
    lines = checklist_str.strip().split("\n")
    for line in lines:
        line = line.strip()
        if not line:
            continue

        # 1. Detect [STATUS] bracket anywhere in the line
        status_match = re.search(r'\[(.*?)\]', line)
        passed = True
        if status_match:
            tag = status_match.group(1).lower().strip()
            if any(w in tag for w in ("fail", "failed", "✗", "false", "no")):
                passed = False
            elif any(w in tag for w in ("pass", "passed", "✓", "true", "yes")):
                passed = True

        # 2. Strip leading bullets, numbers, and the [STATUS] tag completely
        clean_line = re.sub(r'^[•\-\*\d\.\)\s]*', '', line).strip()
        clean_line = re.sub(r'\[.*?\]\s*', '', clean_line).strip()
        clean_line = re.sub(r'^[•\-\*\s]+', '', clean_line).strip()

        # 3. Extract comment (either separated by dash " — " or trailing parenthesis "(...)")
        instruction = clean_line
        comment = ""

        dash_split = re.split(r'\s+[—–]\s+|\s+-\s+', clean_line, maxsplit=1)
        if len(dash_split) == 2:
            instruction = dash_split[0].strip()
            comment = dash_split[1].strip()
        else:
            paren_match = re.search(r'^(.*?)\s*\(([^)]+)\)\s*$', clean_line)
            if paren_match and paren_match.group(1).strip():
                instruction = paren_match.group(1).strip()
                comment = paren_match.group(2).strip()

        if instruction or comment:
            items.append(ChecklistItem(
                passed=passed,
                instruction=instruction or clean_line,
                comment=comment,
            ))
    return items


def get_activities_list() -> List[str]:
    """Returns list of activity tab names available in the registry."""
    data = fetch_all_registry_activities()
    return list(data.keys())


def get_activities_overview() -> Dict[str, Any]:
    """
    Returns an overview of all activities in the registry sheet with full metrics:
    submission counts, pending grading, pending approval, pushed/completed status, pass rates.
    """
    data = fetch_all_registry_activities()
    activities_list: List[Dict[str, Any]] = []

    global_total_submissions = 0
    global_pending_count = 0
    global_approved_count = 0
    global_overridden_count = 0
    global_pushed_count = 0
    global_pass_count = 0
    global_fail_count = 0

    for act_name, df in data.items():
        if df.empty:
            continue

        # Normalize and synchronize renamed columns: Grade -> Final Grade (Moodle), Agent Grade -> AI Initial Verdict
        if "Final Grade (Moodle)" not in df.columns and "Grade" in df.columns:
            df["Final Grade (Moodle)"] = df["Grade"]
        elif "Grade" not in df.columns and "Final Grade (Moodle)" in df.columns:
            df["Grade"] = df["Final Grade (Moodle)"]
        elif "Final Grade (Moodle)" in df.columns and "Grade" in df.columns:
            df["Final Grade (Moodle)"] = df["Final Grade (Moodle)"].replace("", pd.NA).fillna(df["Grade"]).fillna("")
            df["Grade"] = df["Final Grade (Moodle)"]

        if "AI Initial Verdict" not in df.columns and "Agent Grade" in df.columns:
            df["AI Initial Verdict"] = df["Agent Grade"]
        elif "Agent Grade" not in df.columns and "AI Initial Verdict" in df.columns:
            df["Agent Grade"] = df["AI Initial Verdict"]
        elif "AI Initial Verdict" in df.columns and "Agent Grade" in df.columns:
            df["AI Initial Verdict"] = df["AI Initial Verdict"].replace("", pd.NA).fillna(df["Agent Grade"]).fillna("")
            df["Agent Grade"] = df["AI Initial Verdict"]

        # Ensure required columns
        for col in [
            "Name", "Status", "Final Grade (Moodle)", "Grade", "Online text", "Media folder",
            "Attempt number", "Last modified", "Feedback comment",
            "Agent's checklist", "AI Initial Verdict", "Agent Grade", "Review status",
            "Mentor reviewer", "Mentor reviewed at"
        ]:
            if col not in df.columns:
                df[col] = ""

        df["Review status"] = df["Review status"].replace("", "Pending Review")
        df["AI Initial Verdict"] = df["AI Initial Verdict"].replace("", pd.NA).fillna(df["Final Grade (Moodle)"]).fillna("")
        df["Agent Grade"] = df["AI Initial Verdict"]

        total_sub = len(df)
        rev_status_lower = df["Review status"].astype(str).str.lower()
        status_lower = df["Status"].astype(str).str.lower()
        grade_lower = df["Grade"].astype(str).str.lower()
        agent_grade_lower = df["Agent Grade"].astype(str).str.lower()

        pending_count = int(sum(rev_status_lower == "pending review"))
        approved_count = int(sum(rev_status_lower == "approved by mentor"))
        overridden_count = int(sum(rev_status_lower == "overridden by mentor"))

        # Pushed/Completed: marked as 'Graded' in Moodle Status or mentor review finished
        pushed_count = int(sum((status_lower.str.contains("graded", na=False)) | (rev_status_lower.str.contains("approved|overridden", na=False))))

        pass_count = int(sum(grade_lower.str.contains("pass", na=False)))
        fail_count = int(sum(grade_lower.str.contains("fail", na=False)))
        unsure_count = int(sum(agent_grade_lower.str.contains("unsure", na=False)))
        pass_rate = round((pass_count / total_sub * 100), 1) if total_sub > 0 else 0.0

        # Pending approval: specifically submissions where agent passed but mentor hasn't confirmed yet
        pending_approval = int(sum((rev_status_lower == "pending review") & (agent_grade_lower.str.contains("pass", na=False))))
        pending_grading = pending_count

        # Latest activity timestamp
        last_mod = ""
        if "Last modified" in df.columns:
            non_empty_dates = df["Last modified"].dropna().astype(str)
            non_empty_dates = non_empty_dates[non_empty_dates.str.strip() != ""]
            if not non_empty_dates.empty:
                last_mod = str(non_empty_dates.iloc[-1])

        activities_list.append({
            "name": act_name,
            "total_submissions": total_sub,
            "pending_count": pending_count,
            "pending_grading": pending_grading,
            "pending_approval": pending_approval,
            "approved_count": approved_count,
            "overridden_count": overridden_count,
            "pushed_count": pushed_count,
            "pass_count": pass_count,
            "fail_count": fail_count,
            "unsure_count": unsure_count,
            "pass_rate": pass_rate,
            "last_modified": last_mod,
        })

        global_total_submissions += total_sub
        global_pending_count += pending_count
        global_approved_count += approved_count
        global_overridden_count += overridden_count
        global_pushed_count += pushed_count
        global_pass_count += pass_count
        global_fail_count += fail_count

    global_summary = {
        "total_activities": len(activities_list),
        "total_submissions": global_total_submissions,
        "pending_count": global_pending_count,
        "approved_count": global_approved_count,
        "overridden_count": global_overridden_count,
        "pushed_count": global_pushed_count,
        "pass_count": global_pass_count,
        "fail_count": global_fail_count,
        "overall_pass_rate": round((global_pass_count / global_total_submissions * 100), 1) if global_total_submissions > 0 else 0.0,
    }

    return {
        "summary": global_summary,
        "activities": activities_list,
        "activity_names": [a["name"] for a in activities_list],
    }



def get_activity_submissions(activity_name: str) -> Dict[str, Any]:
    """
    Returns the submissions list and summary metrics for a specific activity.
    """
    data = fetch_all_registry_activities()
    if activity_name not in data:
        # If not present, try case-insensitive match
        for k, v in data.items():
            if k.lower() == activity_name.lower():
                activity_name = k
                break

    df = data.get(activity_name, pd.DataFrame())
    if df.empty:
        return {
            "activity_name": activity_name,
            "submissions": [],
            "summary": ActivitySummary(
                activity_name=activity_name,
                total_submissions=0,
                pending_count=0,
                approved_count=0,
                overridden_count=0,
                pass_count=0,
                fail_count=0,
                unsure_count=0,
                pass_rate=0.0,
            ).model_dump(),
        }

    # Normalize and synchronize renamed columns: Grade -> Final Grade (Moodle), Agent Grade -> AI Initial Verdict
    if "Final Grade (Moodle)" not in df.columns and "Grade" in df.columns:
        df["Final Grade (Moodle)"] = df["Grade"]
    elif "Grade" not in df.columns and "Final Grade (Moodle)" in df.columns:
        df["Grade"] = df["Final Grade (Moodle)"]
    elif "Final Grade (Moodle)" in df.columns and "Grade" in df.columns:
        df["Final Grade (Moodle)"] = df["Final Grade (Moodle)"].replace("", pd.NA).fillna(df["Grade"]).fillna("")
        df["Grade"] = df["Final Grade (Moodle)"]

    if "AI Initial Verdict" not in df.columns and "Agent Grade" in df.columns:
        df["AI Initial Verdict"] = df["Agent Grade"]
    elif "Agent Grade" not in df.columns and "AI Initial Verdict" in df.columns:
        df["Agent Grade"] = df["AI Initial Verdict"]
    elif "AI Initial Verdict" in df.columns and "Agent Grade" in df.columns:
        df["AI Initial Verdict"] = df["AI Initial Verdict"].replace("", pd.NA).fillna(df["Agent Grade"]).fillna("")
        df["Agent Grade"] = df["AI Initial Verdict"]

    # Ensure required columns
    for col in [
        "Name", "Status", "Final Grade (Moodle)", "Grade", "Online text", "Media folder",
        "Attempt number", "Last modified", "Feedback comment",
        "Agent's checklist", "AI Initial Verdict", "Agent Grade", "Review status",
        "Mentor reviewer", "Mentor reviewed at"
    ]:
        if col not in df.columns:
            df[col] = ""

    df["Attempt_Num_Int"] = pd.to_numeric(df["Attempt number"], errors="coerce").fillna(1).astype(int)
    df["Review status"] = df["Review status"].replace("", "Pending Review")
    df["AI Initial Verdict"] = df["AI Initial Verdict"].replace("", pd.NA).fillna(df["Final Grade (Moodle)"]).fillna("")
    df["Agent Grade"] = df["AI Initial Verdict"]

    total_submissions = len(df)
    pending_count = int(sum(df["Review status"].str.lower() == "pending review"))
    approved_count = int(sum(df["Review status"].str.lower() == "approved by mentor"))
    overridden_count = int(sum(df["Review status"].str.lower() == "overridden by mentor"))
    pass_count = int(sum(df["Grade"].str.lower().str.contains("pass", na=False)))
    fail_count = int(sum(df["Grade"].str.lower().str.contains("fail", na=False)))
    unsure_count = int(sum(df["Agent Grade"].str.lower().str.contains("unsure", na=False)))
    pass_rate = round((pass_count / total_submissions * 100), 1) if total_submissions > 0 else 0.0

    summary = ActivitySummary(
        activity_name=activity_name,
        total_submissions=total_submissions,
        pending_count=pending_count,
        approved_count=approved_count,
        overridden_count=overridden_count,
        pass_count=pass_count,
        fail_count=fail_count,
        unsure_count=unsure_count,
        pass_rate=pass_rate,
    )

    submissions: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        raw_checklist = str(row["Agent's checklist"]).strip()
        sub = SubmissionItem(
            name=str(row["Name"]).strip(),
            status=str(row["Status"]).strip(),
            grade=str(row["Grade"]).strip(),
            agent_grade=str(row["Agent Grade"]).strip(),
            online_text=str(row["Online text"]).strip(),
            media_folder=str(row["Media folder"]).strip(),
            attempt_number=int(row["Attempt_Num_Int"]),
            last_modified=str(row["Last modified"]).strip(),
            feedback_comment=str(row["Feedback comment"]).strip(),
            agent_checklist=raw_checklist,
            checklist_items=parse_checklist_items(raw_checklist),
            review_status=str(row["Review status"]).strip(),
            mentor_reviewer=str(row.get("Mentor reviewer", "")).strip(),
            mentor_reviewed_at=str(row.get("Mentor reviewed at", "")).strip(),
        )
        submissions.append(sub.model_dump())

    from agents.submission_reviewer.central_sheet_manager import get_activity_instructions
    activity_details = get_activity_instructions(activity_name)

    return {
        "activity_name": activity_name,
        "instructions": activity_details.get("instructions", ""),
        "edge_cases": activity_details.get("edge_cases", ""),
        "guidelines": activity_details.get("guidelines", ""),
        "submissions": submissions,
        "summary": summary.model_dump(),
        "sheet_id": REGISTRY_SHEET_ID_KNOWN,
        "sheet_url": f"https://docs.google.com/spreadsheets/d/{REGISTRY_SHEET_ID_KNOWN}/edit",
    }


def add_activity_rule(
    activity_name: str,
    edge_case: Optional[str] = None,
    guideline: Optional[str] = None,
) -> Dict[str, Any]:
    """Adds an edge case and/or guideline to 'Activity Details' tab for the activity."""
    from agents.submission_reviewer.central_sheet_manager import append_activity_details_rule
    return append_activity_details_rule(
        activity_name=activity_name,
        edge_case=edge_case,
        guideline=guideline,
    )


def update_submission_review(
    activity_name: str,
    student_name: str,
    attempt_number: int,
    new_grade: str,
    new_feedback: str,
    review_status: str,
    mentor_name: str,
    edge_case: Optional[str] = None,
    guideline: Optional[str] = None,
) -> Dict[str, Any]:
    """Applies mentor review/override to Google Sheet and local CSV, and adds rule if provided."""
    res = sheet_update_review(
        activity_name=activity_name,
        student_name=student_name,
        attempt_number=attempt_number,
        new_grade=new_grade,
        new_feedback=new_feedback,
        review_status=review_status,
        mentor_name=mentor_name,
    )

    if (edge_case and edge_case.strip()) or (guideline and guideline.strip()):
        try:
            rule_res = add_activity_rule(
                activity_name=activity_name,
                edge_case=edge_case,
                guideline=guideline,
            )
            res["activity_rule_sync"] = rule_res
        except Exception as r_err:
            logger.warning(f"Could not append activity rule for {activity_name}: {r_err}")

    return res


def batch_approve_activity(activity_name: str, mentor_name: str) -> Dict[str, Any]:
    """Approves all pending confident passes in the activity."""
    return sheet_batch_approve(activity_name=activity_name, mentor_name=mentor_name)


def sync_local_folder(assignment_folder: Optional[str], activity_name: str) -> Dict[str, Any]:
    """Syncs an assignment run folder into the sheet."""
    if not assignment_folder:
        from agents.submission_reviewer.moodle_review_runner import find_latest_assignment_folder
        assignment_folder = find_latest_assignment_folder()

    if not assignment_folder:
        raise ValueError("No assignment folder specified or found locally.")

    return sheet_sync_folder(assignment_folder=assignment_folder, activity_name=activity_name)


def export_moodle_gradebook_csv(activity_name: str) -> str:
    """
    Generates a CSV string formatted for direct Moodle gradebook upload.
    Deduplicates multiple attempts per student by keeping only the latest attempt.
    """
    import os
    import csv
    data = fetch_all_registry_activities()
    df = data.get(activity_name, pd.DataFrame())
    if df.empty:
        return "Identifier,Full name,Grade,Feedback comments\n"

    # Ensure required columns
    for col in ["Name", "Grade", "Feedback comment", "Attempt number"]:
        if col not in df.columns:
            df[col] = ""

    # Deduplicate: sort by Attempt number descending and keep latest attempt per student
    df_clean = df[df["Name"].str.strip() != ""].copy()
    df_clean["Attempt_Int"] = pd.to_numeric(df_clean["Attempt number"], errors="coerce").fillna(1).astype(int)
    df_clean["Name_Norm"] = df_clean["Name"].str.strip().str.lower()
    df_latest = df_clean.sort_values(by="Attempt_Int", ascending=False).drop_duplicates(subset=["Name_Norm"], keep="first")

    # Try mapping to Moodle Identifier from local grading_worksheet.csv if available
    identifier_map = {}
    try:
        from agents.submission_reviewer.moodle_review_runner import find_latest_assignment_folder
        latest_folder = find_latest_assignment_folder()
        if latest_folder:
            local_csv = os.path.join(latest_folder, "grading_worksheet.csv")
            if os.path.exists(local_csv):
                with open(local_csv, mode="r", encoding="utf-8-sig") as f:
                    for row in csv.DictReader(f):
                        fn = (row.get("Full name") or "").strip().lower()
                        ident = (row.get("Identifier") or "").strip()
                        if fn and ident:
                            identifier_map[fn] = ident
    except Exception:
        pass

    moodle_df = pd.DataFrame()
    if identifier_map:
        moodle_df["Identifier"] = df_latest["Name_Norm"].map(lambda n: identifier_map.get(n, ""))
    moodle_df["Full name"] = df_latest["Name"]
    moodle_grade_val = df_latest["Final Grade (Moodle)"] if "Final Grade (Moodle)" in df_latest.columns else df_latest["Grade"]
    moodle_df["Grade"] = moodle_grade_val
    moodle_df["Feedback comments"] = df_latest["Feedback comment"]

    return moodle_df.to_csv(index=False)
