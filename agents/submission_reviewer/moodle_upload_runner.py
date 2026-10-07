# -*- coding: utf-8 -*-
"""
Moodle Upload Runner.

Action 2 Runner:
1. Connects to the Central Google Sheet (Moodle Submission Review Registry).
2. Reads the reviewed/approved grades and feedback comments for the target activity tab.
3. Obtains the Moodle grading worksheet template (either downloading fresh from Moodle or using local CSV).
4. Populates 'Grade' and 'Feedback comments' from the Google Sheet records.
5. Uploads the populated worksheet back into the Moodle gradebook via Playwright automation.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import logging
import os
import re
import sys
import unicodedata
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from dotenv import load_dotenv

from agents.submission_reviewer.Ingestion.ingestion import (
    MoodleConfig,
    TARGET_URL,
    get_active_or_fresh_otp,
)
from agents.submission_reviewer.Upload.upload import (
    build_upload_worksheet_url,
    upload_grading_worksheet_to_moodle,
)
from agents.submission_reviewer.central_sheet_manager import (
    DEFAULT_DRIVE_FOLDER_ID,
    REGISTRY_SHEET_ID_KNOWN,
    fetch_all_registry_activities,
    get_or_create_registry_spreadsheet,
)

load_dotenv()

logger = logging.getLogger("moodle_upload_runner")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] [MoodleUpload] %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

DEFAULT_REVIEW_SHEET_URL = (
    f"https://docs.google.com/spreadsheets/d/{REGISTRY_SHEET_ID_KNOWN}/edit?usp=sharing"
)


def normalize_student_name(name: str) -> str:
    """Normalizes student names for reliable cross-system matching."""
    if not name:
        return ""
    clean = unicodedata.normalize("NFKD", str(name)).strip().lower()
    clean = re.sub(r"\s+", " ", clean)
    return clean


def fetch_activity_grades_from_sheet(
    activity_name: str,
    sheet_id_or_url: str = DEFAULT_REVIEW_SHEET_URL,
    only_mentor_reviewed: bool = False,
) -> Dict[str, Dict[str, str]]:
    """
    Reads all student evaluation records for an activity tab from the central Google Sheet.

    Returns:
        Dict mapping normalized student name -> {
            "name": str,
            "grade": str,
            "feedback": str,
            "review_status": str,
            "attempt_number": int,
        }
    """
    logger.info(f"📊 Connecting to Google Sheet for activity '{activity_name}'...")
    all_data = fetch_all_registry_activities()

    # Find matching tab (exact or case-insensitive)
    df = all_data.get(activity_name)
    if df is None or df.empty:
        for tab_name, tab_df in all_data.items():
            if tab_name.strip().lower() == activity_name.strip().lower():
                df = tab_df
                break

    if df is None or df.empty:
        raise ValueError(
            f"No records found for activity tab '{activity_name}' in sheet {sheet_id_or_url}."
        )

    # Ensure required columns
    if "Final Grade (Moodle)" not in df.columns and "Grade" in df.columns:
        df["Final Grade (Moodle)"] = df["Grade"]
    elif "Grade" not in df.columns and "Final Grade (Moodle)" in df.columns:
        df["Grade"] = df["Final Grade (Moodle)"]

    for col in ["Name", "Final Grade (Moodle)", "Grade", "Feedback comment", "Review status", "Attempt number"]:
        if col not in df.columns:
            df[col] = ""

    student_map: Dict[str, Dict[str, str]] = {}

    # Sort so that highest attempt numbers take precedence
    df_clean = df.copy()
    df_clean["att_int"] = df_clean["Attempt number"].astype(str).str.extract(r"(\d+)").fillna(1).astype(int)
    df_sorted = df_clean.sort_values(by="att_int", ascending=True)

    for _, row in df_sorted.iterrows():
        raw_name = str(row.get("Name") or "").strip()
        if not raw_name:
            continue

        raw_grade = str(row.get("Final Grade (Moodle)") or row.get("Grade") or "").strip()
        raw_feedback = str(row.get("Feedback comment") or "").strip()
        review_status = str(row.get("Review status") or "").strip()
        attempt_num = int(row.get("att_int", 1))

        if only_mentor_reviewed:
            status_lower = review_status.lower()
            if not ("approved" in status_lower or "overridden" in status_lower):
                continue

        # Format grade appropriately for Moodle (Pass / Fail or numeric)
        if "pass" in raw_grade.lower():
            moodle_grade = "Pass"
        elif "fail" in raw_grade.lower():
            moodle_grade = "Fail"
        else:
            moodle_grade = raw_grade

        norm_name = normalize_student_name(raw_name)
        student_map[norm_name] = {
            "name": raw_name,
            "grade": moodle_grade,
            "feedback": raw_feedback,
            "review_status": review_status,
            "attempt_number": attempt_num,
        }

    logger.info(f"   ✅ Loaded {len(student_map)} student record(s) from tab '{activity_name}'.")
    return student_map


def populate_worksheet_from_sheet_records(
    template_csv_path: str,
    output_csv_path: str,
    student_records: Dict[str, Dict[str, str]],
) -> Tuple[int, int]:
    """
    Populates an existing Moodle grading worksheet with grades and feedback
    from the Google Sheet records.

    Returns:
        (total_students_in_worksheet, matched_and_updated_count)
    """
    if not os.path.exists(template_csv_path):
        raise FileNotFoundError(f"Template CSV not found at {template_csv_path}")

    with open(template_csv_path, mode="r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if "Grade" not in fieldnames or "Feedback comments" not in fieldnames:
        raise ValueError(
            f"CSV file at {template_csv_path} does not match Moodle grading worksheet format. "
            f"Columns: {fieldnames}"
        )

    matched_count = 0
    total_students = len(rows)

    for row in rows:
        student_name = row.get("Full name") or row.get("Name") or ""
        norm_name = normalize_student_name(student_name)

        record = student_records.get(norm_name)
        if not record:
            # Fallback substring match
            for s_key, s_data in student_records.items():
                if s_key in norm_name or norm_name in s_key:
                    record = s_data
                    break

        if record:
            row["Grade"] = record["grade"]
            row["Feedback comments"] = record["feedback"]
            matched_count += 1

    with open(output_csv_path, mode="w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        writer.writerows(rows)

    logger.info(
        f"📝 Populated grading worksheet: {output_csv_path} "
        f"({matched_count}/{total_students} students mapped)"
    )
    return total_students, matched_count


async def run_moodle_upload_pipeline(
    activity_name: str,
    assignment_url: Optional[str] = None,
    review_sheet_url: str = DEFAULT_REVIEW_SHEET_URL,
    existing_csv_path: Optional[str] = None,
    only_mentor_reviewed: bool = False,
    config: Optional[MoodleConfig] = None,
) -> bool:
    """
    Executes the complete Moodle Gradebook Upload flow:
    1. Fetches grades from Google Sheet tab for `activity_name`.
    2. Downloads or locates Moodle grading worksheet template.
    3. Populates grades and feedback comments.
    4. Uploads to Moodle via Playwright.
    """
    cfg = config or MoodleConfig(
        target_url=assignment_url or os.getenv("MOODLE_ASSIGNMENT_URL") or TARGET_URL
    )
    cfg.validate()

    url = cfg.target_url
    logger.info(f"🚀 Starting Moodle Gradebook Upload Pipeline for '{activity_name}'...")
    logger.info(f"   Target URL: {url}")

    # 1. Fetch records from Google Sheet tab
    student_records = fetch_activity_grades_from_sheet(
        activity_name=activity_name,
        sheet_id_or_url=review_sheet_url,
        only_mentor_reviewed=only_mentor_reviewed,
    )

    if not student_records:
        logger.warning(f"No student grades found for '{activity_name}'. Nothing to upload.")
        return False

    # 2. Prepare worksheet CSV
    os.makedirs(cfg.base_dir, exist_ok=True)
    temp_dir = os.path.join(cfg.base_dir, f"upload_{int(datetime.now().timestamp())}")
    os.makedirs(temp_dir, exist_ok=True)

    template_csv = existing_csv_path
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=cfg.browser_headless,
            args=["--no-sandbox", "--disable-setuid-sandbox"],
        )

        try:
            context = await browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
            page = await context.new_page()

            # Navigate to grading page to check auth / download worksheet if needed
            logger.info("🔑 Checking Moodle session & authentication...")
            await page.goto(url, wait_until="networkidle", timeout=cfg.browser_timeout_ms)

            # Perform login if required
            if "login" in page.url or "auth.php" in page.url or await page.locator("#username, input[name='username']").count() > 0:
                logger.info("🔐 Entering Moodle credentials...")
                parsed = urlparse(url)
                login_url = f"{parsed.scheme}://{parsed.netloc}/login/index.php"
                if "login" not in page.url:
                    await page.goto(login_url, wait_until="networkidle", timeout=cfg.browser_timeout_ms)

                await page.fill("#username, input[name='username']", cfg.moodle_user or "")
                await page.fill("#password, input[name='password']", cfg.moodle_pass or "")
                await page.click("#loginbtn, button[type='submit']")
                await page.wait_for_load_state("networkidle")
                await page.wait_for_timeout(2000)

                # MFA check
                if "tool/mfa" in page.url or "auth.php" in page.url or await page.locator("#id_verificationcode").count() > 0:
                    logger.info("🔒 MFA detected! Retrieving verification OTP from Gmail...")
                    otp = await asyncio.to_thread(
                        get_active_or_fresh_otp,
                        cfg.email_user,
                        cfg.email_pass,
                        cfg.imap_server,
                        cfg.imap_folder,
                        cfg.otp_max_age_minutes,
                        cfg.otp_timeout_seconds,
                    )
                    await page.fill("#id_verificationcode, input[name*='code'], #id_code", otp)
                    continue_btn = page.locator("button:has-text('Continue'), input[value='Continue']").first
                    if await continue_btn.is_visible(timeout=2000):
                        await continue_btn.click()
                    for _ in range(15):
                        if "auth.php" not in page.url and "login" not in page.url and "tool/mfa" not in page.url:
                            break
                        await page.wait_for_timeout(1000)

                logger.info("🎉 Authenticated into Moodle successfully!")
                await page.goto(url, wait_until="networkidle", timeout=cfg.browser_timeout_ms)

            # If template CSV not provided locally, download fresh template directly from Moodle
            if not template_csv or not os.path.exists(template_csv):
                logger.info("📥 Downloading fresh grading worksheet template from Moodle...")
                downloaded_template = os.path.join(temp_dir, "template_grading_worksheet.csv")

                download_url = None
                opt_elem = page.locator("option:has-text('Download grading worksheet')").first
                if await opt_elem.count() > 0:
                    download_url = await opt_elem.get_attribute("value")

                if download_url:
                    if download_url.startswith("/"):
                        parsed_t = urlparse(url)
                        download_url = f"{parsed_t.scheme}://{parsed_t.netloc}{download_url}"
                    async with page.expect_download(timeout=30000) as dl_info:
                        await page.evaluate(f"window.location.href = '{download_url}'")
                    dl = await dl_info.value
                    await dl.save_as(downloaded_template)
                else:
                    # Direct action pattern
                    direct_dl_url = f"{url.split('&action')[0]}&action=downloadgradingworksheet"
                    async with page.expect_download(timeout=30000) as dl_info:
                        await page.evaluate(f"window.location.href = '{direct_dl_url}'")
                    dl = await dl_info.value
                    await dl.save_as(downloaded_template)

                template_csv = downloaded_template
                logger.info(f"   ✅ Saved template worksheet to: {template_csv}")

            # Populate template with Google Sheet records
            final_csv_to_upload = os.path.join(temp_dir, "populated_grading_worksheet.csv")
            populate_worksheet_from_sheet_records(
                template_csv_path=template_csv,
                output_csv_path=final_csv_to_upload,
                student_records=student_records,
            )

            # Upload back to Moodle
            logger.info("📤 Uploading populated worksheet back to Moodle...")
            success = await upload_grading_worksheet_to_moodle(
                page=page,
                csv_path=final_csv_to_upload,
                target_url=url,
                timeout_ms=cfg.browser_timeout_ms,
            )

            if success:
                logger.info(f"🎉 SUCCESS! Approved grades for '{activity_name}' published to Moodle Gradebook!")
            else:
                logger.error("❌ Moodle worksheet upload failed or confirmation was not verified.")

            return success

        finally:
            await browser.close()


def main():
    parser = argparse.ArgumentParser(
        description="Moodle Gradebook Upload Runner — publishes reviewed grades from Google Sheet to Moodle."
    )
    parser.add_argument(
        "--activity-name",
        type=str,
        required=True,
        help="Name of the activity tab in the Google Sheet (e.g. 'System Identification').",
    )
    parser.add_argument(
        "--assignment-url",
        type=str,
        default=None,
        help="Moodle assignment URL (defaults to MOODLE_ASSIGNMENT_URL).",
    )
    parser.add_argument(
        "--review-sheet",
        type=str,
        default=DEFAULT_REVIEW_SHEET_URL,
        help="Google Sheet URL containing the activity tab.",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Optional path to existing grading_worksheet.csv template.",
    )
    parser.add_argument(
        "--only-reviewed",
        action="store_true",
        help="Upload only submissions that were explicitly approved or overridden by mentor.",
    )
    parser.add_argument(
        "--headed",
        action="store_true",
        help="Run Playwright browser in visible window.",
    )

    args = parser.parse_args()

    cfg = MoodleConfig(
        target_url=args.assignment_url or os.getenv("MOODLE_ASSIGNMENT_URL") or TARGET_URL,
        browser_headless=not args.headed,
    )

    success = asyncio.run(
        run_moodle_upload_pipeline(
            activity_name=args.activity_name,
            assignment_url=args.assignment_url,
            review_sheet_url=args.review_sheet,
            existing_csv_path=args.csv,
            only_mentor_reviewed=args.only_reviewed,
            config=cfg,
        )
    )

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
