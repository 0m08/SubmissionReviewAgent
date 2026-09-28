# -*- coding: utf-8 -*-
"""
Moodle Submission Review Runner.

Connects Moodle Ingestion with the Submission Reviewer Agent:
1. Fetches Activity Instructions and Guardrails from Google Docs via Drive OAuth.
2. Locates student media (images, videos, audio) in Submissions/{Student_Name}_{ID}/.
3. Evaluates each candidate submission using review_single_submission.
4. Writes results STRICTLY back to grading_worksheet.csv (updating ONLY 'Grade' and 'Feedback comments').
5. Syncs the updated grading worksheet back to Google Drive.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import mimetypes
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from PIL import Image

from agents.submission_reviewer.Ingestion.ingestion import (
    BASE_DIR,
    get_user_oauth_drive_service,
    run_moodle_ingestion_sync,
    MoodleConfig,
)
from agents.submission_reviewer.reviewer import review_single_submission
from agents.submission_reviewer.schemas import SubmissionReviewOutput

load_dotenv()

logger = logging.getLogger("moodle_review_runner")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# Default Document IDs in Moodle Downloads folder
DEFAULT_INSTRUCTIONS_DOC_ID = os.getenv(
    "MOODLE_INSTRUCTIONS_DOC_ID",
    "1whICRKo83X7djnfZOwTe9QoRxGmfrsKV_KnBHv-1Na8"
)
DEFAULT_GUARDRAILS_DOC_ID = os.getenv(
    "MOODLE_GUARDRAILS_DOC_ID",
    "1_XTnq0HmCfLFNjJy_tUzj9a6-S5uEbaNGS0SupXDaeY"
)


# ==============================================================================
# 1. KNOWLEDGE INGESTION: GOOGLE DOCS FETCHER
# ==============================================================================

def extract_google_doc_id(input_str: str) -> str:
    """Extracts a Google Doc/File ID from a URL or raw ID string."""
    if not input_str:
        return ""
    m = re.search(r'/document/d/([a-zA-Z0-9_-]+)', input_str)
    if m:
        return m.group(1)
    m = re.search(r'/file/d/([a-zA-Z0-9_-]+)', input_str)
    if m:
        return m.group(1)
    m = re.search(r'id=([a-zA-Z0-9_-]+)', input_str)
    if m:
        return m.group(1)
    return input_str.strip()


def fetch_google_doc_text(doc_id_or_url: str, drive_service=None) -> str:
    """
    Exports a Google Doc as plain text using the authenticated user Drive client.
    """
    doc_id = extract_google_doc_id(doc_id_or_url)
    if not doc_id:
        return ""

    if drive_service is None:
        drive_service = get_user_oauth_drive_service()

    try:
        content = drive_service.files().export(
            fileId=doc_id,
            mimeType="text/plain"
        ).execute()
        text = content.decode("utf-8", errors="replace")
        # Strip UTF-8 Byte Order Mark if present
        if text.startswith("\ufeff"):
            text = text[1:]
        return text.strip()
    except Exception as err:
        logger.error(f"❌ Failed to export Google Doc '{doc_id}': {err}")
        raise RuntimeError(f"Could not fetch Google Doc '{doc_id}': {err}") from err


# ==============================================================================
# 2. MEDIA LOADER FOR STUDENT SUBMISSIONS
# ==============================================================================

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".webm", ".mkv", ".m4v"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac"}


def load_student_submission_assets(
    student_dir: str
) -> Tuple[List[Image.Image], List[Dict[str, Any]], str]:
    """
    Loads images, videos, and text from a student submission directory.
    Returns: (images_list, videos_list, online_text_str)
    """
    images: List[Image.Image] = []
    videos: List[Dict[str, Any]] = []
    online_text = ""

    if not student_dir or not os.path.exists(student_dir):
        return images, videos, online_text

    # 1. Read online_text.txt if present
    text_path = os.path.join(student_dir, "online_text.txt")
    if os.path.exists(text_path):
        try:
            with open(text_path, "r", encoding="utf-8", errors="replace") as f:
                online_text = f.read().strip()
        except Exception as e:
            logger.warning(f"Could not read {text_path}: {e}")

    # 2. Read media files
    for root, _, filenames in os.walk(student_dir):
        for fname in sorted(filenames):
            if fname.startswith(".") or fname == "online_text.txt":
                continue
            file_path = os.path.join(root, fname)
            ext = Path(fname).suffix.lower()

            if ext in IMAGE_EXTENSIONS:
                try:
                    img = Image.open(file_path)
                    if img.mode not in ("RGB", "L"):
                        img = img.convert("RGB")
                    images.append(img)
                    logger.debug(f"Loaded image: {fname} ({img.size[0]}x{img.size[1]})")
                except Exception as img_err:
                    logger.warning(f"Could not parse image '{fname}': {img_err}")

            elif ext in VIDEO_EXTENSIONS:
                try:
                    with open(file_path, "rb") as vf:
                        raw_bytes = vf.read()
                    mime_type, _ = mimetypes.guess_type(file_path)
                    if not mime_type:
                        mime_type = "video/mp4"
                    videos.append({
                        "bytes": raw_bytes,
                        "mime_type": mime_type,
                        "filename": fname,
                    })
                    logger.debug(f"Loaded video: {fname} ({len(raw_bytes)} bytes)")
                except Exception as vid_err:
                    logger.warning(f"Could not read video '{fname}': {vid_err}")

    return images, videos, online_text


# ==============================================================================
# 3. WORKSHEET EVALUATION & STRICT CSV WRITEBACK
# ==============================================================================

def find_student_folder(submissions_dir: str, full_name: str, identifier: str) -> Optional[str]:
    """
    Locates the student's subfolder under Submissions/ matching either:
    1. '{Full Name}_{Identifier_Slug}'
    2. Any folder containing both full name and identifier digits
    """
    if not submissions_dir or not os.path.exists(submissions_dir):
        return None

    id_slug = identifier.replace(" ", "_")
    exact_name = f"{full_name}_{id_slug}"
    exact_path = os.path.join(submissions_dir, exact_name)
    if os.path.exists(exact_path):
        return exact_path

    # Extract digits from identifier
    id_digits_match = re.search(r'\d+', identifier)
    id_digits = id_digits_match.group(0) if id_digits_match else ""

    norm_name = full_name.strip().lower()

    for item in os.listdir(submissions_dir):
        item_path = os.path.join(submissions_dir, item)
        if os.path.isdir(item_path):
            item_lower = item.lower()
            if (norm_name and norm_name in item_lower) or (id_digits and id_digits in item_lower):
                return item_path

    return None


def evaluate_and_update_worksheet(
    assignment_folder: str,
    instructions_text: str,
    guardrails_text: str,
    activity_name: str = "System Identification",
    primary_model: Optional[str] = None,
    only_unmarked: bool = False,
) -> Dict[str, Any]:
    """
    Evaluates submissions found in the assignment run folder and updates
    grading_worksheet.csv STRICTLY in-place for Grade and Feedback comments.
    """
    csv_path = os.path.join(assignment_folder, "grading_worksheet.csv")
    submissions_dir = os.path.join(assignment_folder, "Submissions")
    manifest_path = os.path.join(assignment_folder, "ingestion_manifest.json")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"grading_worksheet.csv not found in {assignment_folder}")

    # Read existing CSV preserving exact field order
    with open(csv_path, mode="r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    if not fieldnames:
        raise ValueError(f"grading_worksheet.csv is empty or has no header row.")

    # Guardrails structured format for reviewer
    guardrail_list = [{"name": "Standard Guardrails", "description": guardrails_text}] if guardrails_text else []

    evaluated_count = 0
    evaluation_summaries = []

    for row in rows:
        identifier = row.get("Identifier", "")
        full_name = row.get("Full name", "")
        status = row.get("Status", "").strip()
        existing_grade = (row.get("Grade") or "").strip()
        online_text_col = (row.get("Online text") or "").strip()

        status_lower = status.lower()

        # Skip if status is 'No submission' or 'Reopened'
        if status_lower in ("no submission", "reopened", "reopned") or "reopen" in status_lower:
            logger.info(f"⏭️ Skipping {full_name} ({identifier}) — Status: '{status}'.")
            continue

        # Check if student has submission folder
        student_folder = find_student_folder(submissions_dir, full_name, identifier)
        has_submitted_status = status_lower == "submitted for grading"
        has_folder = bool(student_folder and os.path.exists(student_folder))

        if not has_submitted_status and not has_folder:
            logger.info(f"⏭️ Skipping {full_name} ({identifier}) — Status: '{status}' (No submission files found).")
            continue

        if only_unmarked and existing_grade:
            logger.info(f"⏭️ Skipping {full_name} ({identifier}) — already graded ({existing_grade}).")
            continue

        logger.info(f"\n========================================================")
        logger.info(f"🧑‍🎓 Evaluating Submission: {full_name} ({identifier})")
        logger.info(f"   Status: {status}")
        logger.info(f"========================================================")

        # Load media & online text
        images, videos, local_online_text = load_student_submission_assets(student_folder) if student_folder else ([], [], "")
        combined_text = online_text_col or local_online_text

        logger.info(f"   🖼️ Images: {len(images)} | 🎥 Videos: {len(videos)} | 📝 Text: '{combined_text[:50]}...'")

        # Run AI Evaluation
        review_result: SubmissionReviewOutput = review_single_submission(
            activity_name=activity_name,
            activity_instructions=instructions_text,
            guardrails=guardrail_list,
            user_comment=combined_text,
            images=images,
            videos=videos,
            primary_model_choice=primary_model,
        )

        # Normalize grade for Moodle "Fail   Pass" scale
        # ("Pass" or "Fail")
        grade_str = review_result.agent_grade
        if "pass" in grade_str.lower():
            final_grade = "Pass"
        else:
            final_grade = "Fail"

        feedback_comment = review_result.agent_comment.strip()

        logger.info(f"   🎯 Final Grade: {final_grade} (Model raw: {grade_str})")
        logger.info(f"   💬 Feedback Comment: {feedback_comment}")

        # STRICT CSV COLUMN UPDATE: ONLY 'Grade' and 'Feedback comments'
        row["Grade"] = final_grade
        row["Feedback comments"] = feedback_comment

        evaluated_count += 1
        evaluation_summaries.append({
            "identifier": identifier,
            "full_name": full_name,
            "final_grade": final_grade,
            "raw_agent_grade": grade_str,
            "feedback_comments": feedback_comment,
            "checklist_count": len(review_result.checklist_evaluations),
        })

    # Write back strictly to grading_worksheet.csv preserving existing schema
    with open(csv_path, mode="w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"\n✅ Successfully updated {evaluated_count} student record(s) in {csv_path}")

    # Update manifest if present
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as mf:
                manifest_data = json.load(mf)

            manifest_data["reviewed_at"] = datetime.now(timezone.utc).isoformat()
            manifest_data["evaluated_students_count"] = evaluated_count
            manifest_data["evaluations"] = evaluation_summaries

            with open(manifest_path, "w", encoding="utf-8") as mf:
                json.dump(manifest_data, mf, indent=2)
            logger.info(f"📋 Updated ingestion manifest with review metrics: {manifest_path}")
        except Exception as man_err:
            logger.warning(f"Could not update manifest: {man_err}")

    return {
        "assignment_folder": assignment_folder,
        "csv_path": csv_path,
        "evaluated_count": evaluated_count,
        "evaluations": evaluation_summaries,
    }


# ==============================================================================
# 4. DRIVE SYNCHRONIZATION
# ==============================================================================

def sync_worksheet_to_drive(
    assignment_folder: str,
    drive_service=None,
) -> Optional[str]:
    """
    Uploads or updates the modified grading_worksheet.csv to Google Drive
    under the corresponding Assignment run folder.
    """
    from googleapiclient.http import MediaFileUpload

    csv_path = os.path.join(assignment_folder, "grading_worksheet.csv")
    manifest_path = os.path.join(assignment_folder, "ingestion_manifest.json")

    if not os.path.exists(csv_path):
        return None

    if drive_service is None:
        drive_service = get_user_oauth_drive_service()

    # Look for drive folder URL or ID in manifest
    run_folder_id = None
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as mf:
                mdata = json.load(mf)
                folder_url = mdata.get("drive_folder_url", "")
                if folder_url:
                    match = re.search(r'/folders/([a-zA-Z0-9_-]+)', folder_url)
                    if match:
                        run_folder_id = match.group(1)
        except Exception:
            pass

    # If run_folder_id not found in manifest, search Drive for folder matching run_folder_name
    folder_name = os.path.basename(assignment_folder)
    if not run_folder_id:
        query = f"name = '{folder_name}' and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
        res = drive_service.files().list(q=query, spaces='drive', fields='files(id, name)', supportsAllDrives=True).execute()
        files = res.get('files', [])
        if files:
            run_folder_id = files[0]['id']

    if not run_folder_id:
        logger.warning(f"⚠️ Could not locate Google Drive folder for '{folder_name}'. Skipping Drive sync.")
        return None

    # Check if grading_worksheet.csv already exists in that Drive folder
    check_query = f"name = 'grading_worksheet.csv' and '{run_folder_id}' in parents and trashed = false"
    res = drive_service.files().list(q=check_query, spaces='drive', fields='files(id, webViewLink)', supportsAllDrives=True).execute()
    existing_files = res.get('files', [])

    media = MediaFileUpload(csv_path, mimetype='text/csv', resumable=True)

    if existing_files:
        file_id = existing_files[0]['id']
        up = drive_service.files().update(fileId=file_id, media_body=media, fields='id, webViewLink', supportsAllDrives=True).execute()
        link = up.get('webViewLink') or f"https://drive.google.com/file/d/{file_id}/view"
        logger.info(f"☁️ [Drive]: Successfully updated existing grading_worksheet.csv (ID: {file_id})")
    else:
        file_meta = {'name': 'grading_worksheet.csv', 'parents': [run_folder_id]}
        up = drive_service.files().create(body=file_meta, media_body=media, fields='id, webViewLink', supportsAllDrives=True).execute()
        file_id = up.get('id', '')
        link = up.get('webViewLink') or f"https://drive.google.com/file/d/{file_id}/view"
        logger.info(f"☁️ [Drive]: Uploaded new grading_worksheet.csv (ID: {file_id})")

    return link


# ==============================================================================
# 5. HIGH-LEVEL PIPELINE EXECUTION
# ==============================================================================

def find_latest_assignment_folder(base_dir: Optional[str] = None) -> Optional[str]:
    """Finds the most recent Assignment_* directory across possible download locations."""
    search_dirs = []
    if base_dir:
        search_dirs.append(base_dir)
    else:
        if BASE_DIR and BASE_DIR not in search_dirs:
            search_dirs.append(BASE_DIR)
        for fallback in ("downloads/moodle_submissions", "./MyDrive/Moodle_Automated_Downloads"):
            if fallback not in search_dirs:
                search_dirs.append(fallback)

    candidates = []
    for d in search_dirs:
        if not os.path.exists(d):
            continue
        for item in os.listdir(d):
            item_path = os.path.join(d, item)
            if os.path.isdir(item_path) and item.startswith("Assignment_"):
                candidates.append(item_path)
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def run_review_pipeline(
    assignment_folder: Optional[str] = None,
    instructions_doc: str = DEFAULT_INSTRUCTIONS_DOC_ID,
    guardrails_doc: str = DEFAULT_GUARDRAILS_DOC_ID,
    activity_name: str = "System Identification",
    primary_model: Optional[str] = None,
    sync_to_drive: bool = True,
    upload_to_moodle: bool = False,
) -> Dict[str, Any]:
    """
    Executes the review pipeline on an assignment run folder.
    If no assignment folder exists yet, automatically runs Moodle ingestion to create one.
    """
    target_folder = assignment_folder or find_latest_assignment_folder()
    if not target_folder:
        logger.info("ℹ️ No assignment folder found. Automatically running Moodle ingestion to create one...")
        ingestion_result = run_moodle_ingestion_sync()
        if not ingestion_result.success or not ingestion_result.assignment_folder:
            raise RuntimeError(f"Failed to automatically ingest and create assignment folder: {ingestion_result.error}")
        target_folder = ingestion_result.assignment_folder
        logger.info(f"✅ Ingestion created folder: {target_folder}")

    logger.info(f"🚀 Starting Review Pipeline on: {target_folder}")

    # 1. Fetch Knowledge from Google Docs
    drive_service = get_user_oauth_drive_service()
    logger.info(f"📖 Fetching Activity Instructions from Google Doc: {instructions_doc}...")
    instructions_text = fetch_google_doc_text(instructions_doc, drive_service=drive_service)
    logger.info(f"   ✅ Instructions loaded ({len(instructions_text)} characters)")

    logger.info(f"🛡️ Fetching Guardrails from Google Doc: {guardrails_doc}...")
    guardrails_text = fetch_google_doc_text(guardrails_doc, drive_service=drive_service)
    logger.info(f"   ✅ Guardrails loaded ({len(guardrails_text)} characters)")

    # 2. Evaluate Submissions & Update CSV in-place
    review_summary = evaluate_and_update_worksheet(
        assignment_folder=target_folder,
        instructions_text=instructions_text,
        guardrails_text=guardrails_text,
        activity_name=activity_name,
        primary_model=primary_model,
    )

    # 3. Synchronize updated CSV back to Google Drive
    drive_csv_url = None
    if sync_to_drive:
        try:
            drive_csv_url = sync_worksheet_to_drive(target_folder, drive_service=drive_service)
            review_summary["drive_csv_url"] = drive_csv_url
        except Exception as drive_err:
            logger.error(f"❌ Failed to sync updated CSV to Google Drive: {drive_err}")

    # 4. Optional: Upload Graded CSV back to Moodle
    if upload_to_moodle:
        from agents.submission_reviewer.Upload.upload import run_standalone_worksheet_upload_sync
        logger.info(f"📤 Uploading updated grading worksheet back to Moodle...")
        try:
            target_url = None
            manifest_file = os.path.join(target_folder, "ingestion_manifest.json")
            if os.path.exists(manifest_file):
                with open(manifest_file, "r", encoding="utf-8") as mf:
                    target_url = json.load(mf).get("assignment_url")

            moodle_uploaded = run_standalone_worksheet_upload_sync(
                csv_path=review_summary["csv_path"],
                target_url=target_url,
            )
            review_summary["moodle_uploaded"] = moodle_uploaded
        except Exception as m_err:
            logger.error(f"❌ Failed to upload worksheet to Moodle: {m_err}")
            review_summary["moodle_uploaded"] = False

    logger.info(f"\n🎉 Review Pipeline finished successfully!")
    logger.info(f"   Students evaluated: {review_summary['evaluated_count']}")
    logger.info(f"   Updated CSV: {review_summary['csv_path']}")
    if drive_csv_url:
        logger.info(f"   ☁️ Drive Worksheet URL: {drive_csv_url}")
    if review_summary.get("moodle_uploaded"):
        logger.info(f"   🏫 Moodle Upload Status: Successfully Applied to Moodle Gradebook!")

    return review_summary


def run_full_ingest_and_review(
    moodle_config: Optional[MoodleConfig] = None,
    instructions_doc: str = DEFAULT_INSTRUCTIONS_DOC_ID,
    guardrails_doc: str = DEFAULT_GUARDRAILS_DOC_ID,
    activity_name: str = "System Identification",
    primary_model: Optional[str] = None,
    upload_to_moodle: bool = False,
) -> Dict[str, Any]:
    """
    Combined end-to-end: Ingests from Moodle -> Evaluates submissions -> Updates CSV -> Syncs to Drive -> Uploads to Moodle.
    """
    logger.info("🚀 Step 1: Ingesting latest submissions from Moodle...")
    ingestion_result = run_moodle_ingestion_sync(moodle_config)

    if not ingestion_result.success:
        raise RuntimeError(f"Moodle ingestion failed: {ingestion_result.error}")

    logger.info(f"✅ Ingestion successful! Folder: {ingestion_result.assignment_folder}")

    logger.info("🚀 Step 2: Running Submission Reviewer Agent on ingested files...")
    return run_review_pipeline(
        assignment_folder=ingestion_result.assignment_folder,
        instructions_doc=instructions_doc,
        guardrails_doc=guardrails_doc,
        activity_name=activity_name,
        primary_model=primary_model,
        sync_to_drive=True,
        upload_to_moodle=upload_to_moodle,
    )


# ==============================================================================
# 6. CLI
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Moodle Submission Review Runner — evaluates student submissions and updates grading_worksheet.csv."
    )
    parser.add_argument(
        "--folder",
        type=str,
        default=None,
        help="Path to the Assignment_{id}_{timestamp} folder (defaults to most recent run folder).",
    )
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Run fresh Moodle ingestion first before reviewing.",
    )
    parser.add_argument(
        "--upload-moodle",
        action="store_true",
        help="Upload the graded CSV back to Moodle gradebook after review.",
    )
    parser.add_argument(
        "--instructions-doc",
        type=str,
        default=DEFAULT_INSTRUCTIONS_DOC_ID,
        help="Google Doc ID or URL for Activity Instructions.",
    )
    parser.add_argument(
        "--guardrails-doc",
        type=str,
        default=DEFAULT_GUARDRAILS_DOC_ID,
        help="Google Doc ID or URL for Guardrails.",
    )
    parser.add_argument(
        "--activity-name",
        type=str,
        default="System Identification",
        help="Name of the activity being evaluated.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="LLM to use (defaults to reviewer.py configuration: gpt-5.6-luna or gemini for video).",
    )
    parser.add_argument(
        "--no-drive-sync",
        action="store_true",
        help="Skip syncing the updated CSV back to Google Drive.",
    )

    args = parser.parse_args()

    try:
        if args.ingest:
            run_full_ingest_and_review(
                instructions_doc=args.instructions_doc,
                guardrails_doc=args.guardrails_doc,
                activity_name=args.activity_name,
                primary_model=args.model,
                upload_to_moodle=args.upload_moodle,
            )
        else:
            run_review_pipeline(
                assignment_folder=args.folder,
                instructions_doc=args.instructions_doc,
                guardrails_doc=args.guardrails_doc,
                activity_name=args.activity_name,
                primary_model=args.model,
                sync_to_drive=not args.no_drive_sync,
                upload_to_moodle=args.upload_moodle,
            )
    except Exception as exc:
        logger.error(f"❌ Review runner failed: {exc}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
