import time
import gspread
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Optional, Callable, Tuple

from langsmith import traceable

from agents.submission_reviewer.drive_image_helper import fetch_media_from_drive_links
from agents.submission_reviewer.reviewer import review_single_submission
from agents.submission_reviewer.schemas import SubmissionReviewOutput
from services.sheets_service import get_sheet_data_and_df

# Number of parallel worker threads for image fetch + LLM review
MAX_WORKERS = 10


def parse_activity_instructions_tab(sheet: gspread.Spreadsheet) -> Dict[str, Dict[str, str]]:
    """
    Reads the 'Activity Instructions' tab from the given Google Sheet.
    Returns dict mapping tab_name -> {'instructions': ..., 'checklist': ..., 'edge_cases': ..., 'guidelines': ...}
    """
    print("[REVIEWER LOG] Fetching 'Activity Instructions' tab from spreadsheet...")
    try:
        ws, df = get_sheet_data_and_df(sheet, "Activity Instructions")
    except Exception as err:
        print(f"[REVIEWER LOG] ❌ Error reading 'Activity Instructions' tab: {err}")
        raise ValueError(f"Could not open 'Activity Instructions' tab: {err}")

    cols = {str(c).strip().lower(): c for c in df.columns}
    tab_col = cols.get("tab name") or cols.get("tab_name") or cols.get("activity name")
    instr_col = cols.get("activity instructions") or cols.get("activity_instructions") or cols.get("instructions")
    chk_col = cols.get("reviewer checklist") or cols.get("reviewer_checklist") or cols.get("checklist")
    edge_col = (
        cols.get("edge cases")
        or cols.get("edge_cases")
        or cols.get("edge case")
        or cols.get("edge_case")
        or cols.get("activity edge cases")
        or cols.get("activity_edge_cases")
    )
    guide_col = (
        cols.get("guidelines")
        or cols.get("guideline")
        or cols.get("activity guidelines")
        or cols.get("activity_guidelines")
        or cols.get("reviewer guidelines")
    )

    if not tab_col:
        raise ValueError(f"'Activity Instructions' tab is missing 'Tab Name' column. Columns found: {list(df.columns)}")

    result: Dict[str, Dict[str, str]] = {}
    for _, row in df.iterrows():
        name = str(row.get(tab_col, "")).strip()
        if not name:
            continue
        instructions = str(row.get(instr_col, "")) if instr_col else ""
        edge_cases = str(row.get(edge_col, "")) if edge_col and row.get(edge_col) is not None else ""
        guidelines = str(row.get(guide_col, "")) if guide_col and row.get(guide_col) is not None else ""
        # reviewer_checklist is completely disabled — evaluation is done purely on activity_instructions
        result[name] = {
            "instructions": instructions.strip(),
            "checklist": "",
            "edge_cases": edge_cases.strip(),
            "guidelines": guidelines.strip(),
        }

    edge_cases_count = sum(1 for v in result.values() if v.get("edge_cases"))
    guidelines_count = sum(1 for v in result.values() if v.get("guidelines"))
    print(f"[REVIEWER LOG] Parsed instructions for {len(result)} activity tab(s) "
          f"({edge_cases_count} with Edge Cases, {guidelines_count} with Guidelines): {list(result.keys())}")
    return result



def parse_guardrails_tab(sheet: gspread.Spreadsheet) -> List[dict]:
    """
    Reads the 'Guardrails' tab from the Google Sheet.
    Returns a list of dicts: [{'name': ..., 'description': ...}, ...]
    Only rows with both name and description are included.
    Returns an empty list if the tab doesn't exist (guardrails are optional).
    """
    print("[REVIEWER LOG] Fetching 'Guardrails' tab from spreadsheet...")
    try:
        ws, df = get_sheet_data_and_df(sheet, "Guardrails")
    except Exception as err:
        print(f"[REVIEWER LOG] ⚠️ 'Guardrails' tab not found or unreadable: {err}. Proceeding without guardrails.")
        return []

    cols = {str(c).strip().lower(): c for c in df.columns}
    name_col = cols.get("name")
    desc_col = cols.get("description")

    if not name_col or not desc_col:
        print(f"[REVIEWER LOG] ⚠️ 'Guardrails' tab missing 'Name' or 'Description' column. Columns found: {list(df.columns)}")
        return []

    guardrails = []
    for _, row in df.iterrows():
        name = str(row.get(name_col, "")).strip()
        desc = str(row.get(desc_col, "")).strip()
        if name and desc:
            guardrails.append({"name": name, "description": desc})

    print(f"[REVIEWER LOG] Loaded {len(guardrails)} guardrail(s): {[g['name'] for g in guardrails]}")
    return guardrails


def _ensure_output_columns_exist(worksheet: gspread.Worksheet) -> Tuple[int, int, int]:
    """
    Ensures 'Agent_Grade', 'Agent_Comment', and 'Agent_Checklist' headers exist in Row 1.
    Appends any missing headers to Row 1 and returns 1-based column indexes.
    """
    headers = [str(h).strip() for h in worksheet.row_values(1)]
    target_headers = ["Agent_Grade", "Agent_Comment", "Agent_Checklist"]
    col_map = {h.lower(): idx + 1 for idx, h in enumerate(headers)}

    for th in target_headers:
        if th.lower() not in col_map:
            headers.append(th)
            new_col_idx = len(headers)
            print(f"[REVIEWER LOG] Adding missing header '{th}' to column {new_col_idx} in sheet '{worksheet.title}'")
            worksheet.update_cell(1, new_col_idx, th)
            col_map[th.lower()] = new_col_idx
            time.sleep(0.5)

    return col_map["agent_grade"], col_map["agent_comment"], col_map["agent_checklist"]


def format_agent_checklist_text(review_output: SubmissionReviewOutput) -> str:
    """Formats checklist evaluations into clean informal bullet points for the sheet cell."""
    if not review_output.checklist_evaluations:
        return review_output.agent_comment or "No specific checklist evaluations logged."

    lines = []
    for item in review_output.checklist_evaluations:
        status_tag = "PASSED" if item.followed else "FAILED"
        if item.is_fail_if_condition and not item.followed:
            status_tag = "FAIL-IF TRIGGERED"

        rule_desc = item.instruction_or_condition or item.item_id
        note = f" ({item.comment})" if item.comment else ""
        lines.append(f"• [{status_tag}] {rule_desc}{note}")

    return "\n".join(lines)


@traceable(
    name="Submission Reviewer - Review Row Task",
    run_type="chain",
    metadata={
        "agent_name": "submission_reviewer",
        "step_name": "Review Row Task",
        "function_name": "_review_row_task",
    }
)
def _review_row_task(
    row_idx: int,
    user_name: str,
    tab_name: str,
    activity_instructions: str,
    reviewer_checklist: str,
    user_comment: str,
    drive_links: str,
    guardrails: List[dict],
    activity_edge_cases: str = "",
    activity_guidelines: str = "",
    lenient_mode: Optional[bool] = None,
    ultra_lenient_mode: Optional[bool] = None,
    primary_model_choice: Optional[str] = None,
) -> Tuple[int, str, SubmissionReviewOutput, str]:
    """
    Worker function executed in a thread pool.
    Handles media fetching (images + videos) + LLM review for a single row.

    Returns: (row_idx, user_name, review_output, formatted_checklist)
    """
    print(f"\n  [WORKER row={row_idx}] Starting review for {user_name}...")

    # Fetch media (images + videos) from Drive
    images, videos, media_logs = fetch_media_from_drive_links(drive_links)
    print(f"  [WORKER row={row_idx}] Fetched {len(images)} image(s), {len(videos)} video(s). Calling LLM...")

    # Run LLM review
    review_output = review_single_submission(
        activity_name=tab_name,
        activity_instructions=activity_instructions,
        reviewer_checklist=reviewer_checklist,
        user_comment=user_comment,
        images=images,
        videos=videos,
        guardrails=guardrails,
        activity_edge_cases=activity_edge_cases,
        activity_guidelines=activity_guidelines,
        lenient_mode=lenient_mode,
        ultra_lenient_mode=ultra_lenient_mode,
        primary_model_choice=primary_model_choice,
    )

    formatted_checklist = format_agent_checklist_text(review_output)
    print(f"  [WORKER row={row_idx}] Done. Grade: '{review_output.agent_grade}'")
    return row_idx, user_name, review_output, formatted_checklist


@traceable(
    name="Submission Reviewer - Process Activity Sheet",
    run_type="chain",
    metadata={
        "agent_name": "submission_reviewer",
        "step_name": "Process Activity Sheet",
        "function_name": "process_activity_sheet",
    }
)
def process_activity_sheet(
    sheet_url: str,
    gc_client: Optional[gspread.Client] = None,
    target_tab_names: Optional[List[str]] = None,
    overwrite_existing: bool = True,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    max_workers: int = MAX_WORKERS,
    lenient_mode: Optional[bool] = None,
    ultra_lenient_mode: Optional[bool] = None,
    primary_model_choice: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Main processor function with parallel row evaluation.

    Architecture:
    - Images + LLM review run in parallel across up to `max_workers` threads.
    - Sheet writes happen serially on the main thread after all workers complete
      (gspread is not thread-safe; batch serial writes also respect API rate limits).

    Steps:
    1. Reads Activity Instructions map and Guardrails from the sheet.
    2. Filters target activity tabs.
    3. For each tab: submits all eligible rows to the thread pool concurrently.
    4. Collects completed results and writes them back to the sheet one by one.
    """
    print("\n==================================================")
    print(f"[REVIEWER LOG] Starting Submission Reviewer Run | Workers: {max_workers}")
    print(f"[REVIEWER LOG] Sheet URL: {sheet_url}")
    print("==================================================")

    if gc_client is None:
        gc_client = st.session_state.get("gc") or st.session_state.get("gspread_client")
    if not gc_client:
        print("[REVIEWER LOG] ❌ Error: gspread client is not authenticated.")
        raise RuntimeError("gspread client is not authenticated. Please log in via Streamlit OAuth.")

    sheet = gc_client.open_by_url(sheet_url)
    instructions_map = parse_activity_instructions_tab(sheet)
    guardrails = parse_guardrails_tab(sheet)

    all_worksheets = [ws.title for ws in sheet.worksheets()]
    activity_tabs_to_run = []

    if target_tab_names:
        if isinstance(target_tab_names, str):
            target_tab_names = [target_tab_names]
        for t in target_tab_names:
            t_clean = t.strip()
            if t_clean in all_worksheets:
                activity_tabs_to_run.append(t_clean)
            else:
                print(f"[REVIEWER LOG] ⚠️ Requested tab '{t_clean}' not found in workbook: {all_worksheets}")
    else:
        activity_tabs_to_run = [t for t in instructions_map.keys() if t in all_worksheets]

    print(f"[REVIEWER LOG] Selected {len(activity_tabs_to_run)} tab(s): {activity_tabs_to_run}")

    summary_results = {
        "processed_rows": 0,
        "skipped_rows": 0,
        "evaluated_rows": 0,
        "tab_summaries": {},
    }

    instructions_map_lower = {k.lower(): v for k, v in instructions_map.items()}

    total_tabs = len(activity_tabs_to_run)
    for tab_idx, tab_name in enumerate(activity_tabs_to_run):
        print(f"\n[REVIEWER LOG] --- Tab [{tab_idx + 1}/{total_tabs}]: '{tab_name}' ---")
        if progress_callback:
            progress_callback(tab_idx + 1, total_tabs, f"Submitting jobs for tab: {tab_name}")

        worksheet = sheet.worksheet(tab_name)
        grade_col_idx, comment_col_idx, chk_col_idx = _ensure_output_columns_exist(worksheet)
        rows_data = worksheet.get_all_records()

        tab_info = instructions_map.get(tab_name) or instructions_map_lower.get(tab_name.strip().lower(), {})

        default_instructions = tab_info.get("instructions", "")
        default_checklist = tab_info.get("checklist", "")
        default_edge_cases = tab_info.get("edge_cases", "")
        default_guidelines = tab_info.get("guidelines", "")

        parts = []
        if default_edge_cases:
            parts.append("Edge Cases: ✅")
        if default_guidelines:
            parts.append("Guidelines: ✅")
        print(f"[REVIEWER LOG] 📌 '{tab_name}' → {', '.join(parts) if parts else 'No Edge Cases or Guidelines defined'}")




        tab_stats = {"total": len(rows_data), "evaluated": 0, "skipped": 0, "pass": 0, "fail": 0, "unsure": 0}
        print(f"[REVIEWER LOG] {len(rows_data)} row(s) found in '{tab_name}'")

        # ------------------------------------------------------------------ #
        # Phase 1: Filter rows & resolve per-row activity instructions
        # (Handles mixed-activity tabs like 'Test Set' where each row specifies
        #  its activity in the 'Course / Activity' column).
        # ------------------------------------------------------------------ #
        eligible_rows = []  # (row_idx, user_name, user_comment, drive_links, act_name, instructions, checklist, edge_cases, guidelines)
        for row_offset, row in enumerate(rows_data):
            row_idx = row_offset + 2  # 1-based, row 1 is header
            user_name = str(row.get("User Name", "") or row.get("user name", "") or f"User #{row_idx}").strip()

            status_val = str(row.get("Status", "") or row.get("status", "")).strip()
            status_clean = status_val.lower()
            if status_clean == "no submission" or "reopen" in status_clean or status_clean in ("reopened", "reopned"):
                print(f"[REVIEWER LOG] ⏩ SKIP Row {row_idx} ({user_name}) — Status: '{status_val}'")
                tab_stats["skipped"] += 1
                summary_results["skipped_rows"] += 1
                continue

            existing_grade = str(row.get("Agent_Grade", "")).strip()
            if existing_grade and not overwrite_existing:
                print(f"[REVIEWER LOG] ⏩ SKIP Row {row_idx} ({user_name}) — Agent_Grade already set: '{existing_grade}'")
                tab_stats["skipped"] += 1
                summary_results["skipped_rows"] += 1
                continue

            user_comment = str(row.get("User Comment", "") or row.get("user comment", "")).strip()
            drive_links = str(row.get("Drive Image Links", "") or row.get("drive image links", "")).strip()

            # Dynamic resolution of row-specific activity instructions/checklist/edge_cases/guidelines:
            course_activity_raw = str(
                row.get("Course / Activity") or
                row.get("course / activity") or
                row.get("Course/Activity") or
                row.get("course/activity") or
                row.get("Activity") or
                row.get("activity") or
                ""
            ).strip()

            if course_activity_raw and course_activity_raw in instructions_map:
                row_act_name = course_activity_raw
                row_instr = instructions_map[course_activity_raw]["instructions"]
                row_chk = instructions_map[course_activity_raw]["checklist"]
                row_edge = instructions_map[course_activity_raw]["edge_cases"]
                row_guide = instructions_map[course_activity_raw]["guidelines"]
                print(f"[REVIEWER LOG] Row {row_idx} ({user_name}) → Mapped activity: '{row_act_name}' (from Course / Activity column)")
            elif course_activity_raw and course_activity_raw.lower() in instructions_map_lower:
                matched_info = instructions_map_lower[course_activity_raw.lower()]
                row_act_name = course_activity_raw
                row_instr = matched_info["instructions"]
                row_chk = matched_info["checklist"]
                row_edge = matched_info["edge_cases"]
                row_guide = matched_info["guidelines"]
                print(f"[REVIEWER LOG] Row {row_idx} ({user_name}) → Mapped activity: '{row_act_name}' (case-insensitive match)")
            else:
                row_act_name = tab_name
                row_instr = default_instructions
                row_chk = default_checklist
                row_edge = default_edge_cases
                row_guide = default_guidelines

            eligible_rows.append((row_idx, user_name, user_comment, drive_links, row_act_name, row_instr, row_chk, row_edge, row_guide))

        print(f"[REVIEWER LOG] {len(eligible_rows)} row(s) eligible for review, {tab_stats['skipped']} skipped.")
        if not eligible_rows:
            summary_results["tab_summaries"][tab_name] = tab_stats
            continue

        # ------------------------------------------------------------------ #
        # Phase 2 + 3 (interleaved): Parallel review with incremental sheet writes
        # Workers run concurrently; results are flushed to the sheet every
        # FLUSH_EVERY rows as they complete — no waiting for all workers to finish.
        # ------------------------------------------------------------------ #
        FLUSH_EVERY = 5
        print(f"[REVIEWER LOG] 🚀 Launching {min(max_workers, len(eligible_rows))} parallel worker(s) "
              f"for {len(eligible_rows)} row(s). Flushing to sheet every {FLUSH_EVERY} completed rows...")

        future_to_meta = {}
        write_buffer: Dict[int, Tuple[str, SubmissionReviewOutput, str]] = {}

        def _flush_buffer():
            """Write all buffered results to the sheet in ascending row order, then clear buffer."""
            if not write_buffer:
                return
            print(f"\n[REVIEWER LOG] ✍️  Flushing {len(write_buffer)} row(s) to sheet...")
            for r_idx, (r_name, r_output, r_checklist) in sorted(write_buffer.items()):
                print(f"[REVIEWER LOG]   → Row {r_idx} ({r_name}) | Grade: '{r_output.agent_grade}'")
                worksheet.update_cell(r_idx, grade_col_idx, r_output.agent_grade)
                worksheet.update_cell(r_idx, comment_col_idx, r_output.agent_comment)
                worksheet.update_cell(r_idx, chk_col_idx, r_checklist)
                time.sleep(0.3)  # Respect Sheets API rate limit between cell writes

                tab_stats["evaluated"] += 1
                summary_results["evaluated_rows"] += 1
                grade_str = str(r_output.agent_grade)
                if "Pass" in grade_str:
                    tab_stats["pass"] += 1
                elif "Fail" in grade_str:
                    tab_stats["fail"] += 1

                if "Unsure" in grade_str:
                    tab_stats["unsure"] += 1

            write_buffer.clear()
            print(f"[REVIEWER LOG] ✅ Flush complete. Running totals — "
                  f"Pass: {tab_stats['pass']}, Fail: {tab_stats['fail']}, Unsure (Qualified): {tab_stats['unsure']}\n")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for row_idx, user_name, user_comment, drive_links, r_act_name, r_instr, r_chk, r_edge, r_guide in eligible_rows:
                future = executor.submit(
                    _review_row_task,
                    row_idx,
                    user_name,
                    r_act_name,
                    r_instr,
                    r_chk,
                    user_comment,
                    drive_links,
                    guardrails,
                    r_edge,
                    r_guide,
                    lenient_mode,
                    ultra_lenient_mode,
                    primary_model_choice,
                )
                future_to_meta[future] = (row_idx, user_name)


            completed = 0
            total_eligible = len(eligible_rows)

            for future in as_completed(future_to_meta):
                row_idx_f, user_name_f = future_to_meta[future]
                try:
                    r_idx, r_name, r_output, r_checklist = future.result()
                    write_buffer[r_idx] = (r_name, r_output, r_checklist)
                except Exception as exc:
                    print(f"[REVIEWER LOG] ❌ Worker failed for Row {row_idx_f} ({user_name_f}): {exc}")
                    fallback = SubmissionReviewOutput(
                        checklist_evaluations=[],
                        agent_comment=f"Worker hit an error during review: {exc}. Marking Unsure for manual check.",
                        agent_grade="Unsure",
                    )
                    write_buffer[row_idx_f] = (
                        user_name_f,
                        fallback,
                        "• [ERROR] Worker failed — see Agent_Comment",
                    )

                completed += 1
                print(f"[REVIEWER LOG] ✅ {completed}/{total_eligible} reviews complete "
                      f"(buffer: {len(write_buffer)} pending write)...")
                if progress_callback:
                    progress_callback(tab_idx + 1, total_tabs, f"{tab_name}: {completed}/{total_eligible} reviews done")

                # Flush to sheet every FLUSH_EVERY completed results
                if len(write_buffer) >= FLUSH_EVERY:
                    _flush_buffer()

        # Final flush — any remaining results that didn't fill a full batch
        if write_buffer:
            print(f"[REVIEWER LOG] Final flush for {len(write_buffer)} remaining row(s)...")
            _flush_buffer()

        summary_results["tab_summaries"][tab_name] = tab_stats
        print(f"[REVIEWER LOG] Tab '{tab_name}' Complete! Summary: {tab_stats}")

    print("\n==================================================")
    print(f"[REVIEWER LOG] All tabs done. Evaluated: {summary_results['evaluated_rows']}, Skipped: {summary_results['skipped_rows']}")
    print("==================================================\n")

    return summary_results
