# -*- coding: utf-8 -*-
"""
Moodle Assignment Ingestion Pipeline.

Automates:
1. Moodle authentication with adaptive Gmail IMAP OTP / MFA verification.
2. Navigating to the assignment grading page.
3. Downloading the student submissions ZIP archive and extracting files.
4. Downloading the grading worksheet CSV and mapping student records to extracted files.
5. Returning structured ingestion results for downstream reviewer pipeline execution.
"""

from __future__ import annotations

import argparse
import asyncio
import email
import imaplib
import json
import logging
import os
import re
import sys
import time
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from bs4 import BeautifulSoup
from dotenv import load_dotenv

# Ensure environment variables are loaded
load_dotenv()

logger = logging.getLogger("moodle_ingestion")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# Module-level defaults & backwards compatibility
TARGET_URL = os.getenv(
    "MOODLE_ASSIGNMENT_URL",
    "https://planning.guroo.app/mod/assign/view.php?id=11922&action=grading"
)
_DEFAULT_BASE_DIR = (
    "downloads/moodle_submissions"
    if os.path.exists("downloads/moodle_submissions")
    else "./MyDrive/Moodle_Automated_Downloads"
)
BASE_DIR = os.getenv("MOODLE_DOWNLOAD_DIR", _DEFAULT_BASE_DIR)

MOODLE_USER = os.getenv("MOODLE_USER")
MOODLE_PASS = os.getenv("MOODLE_PASS")
GMAIL_USER = os.getenv("GMAIL_USER") or os.getenv("SMTP_USER")
GMAIL_APP_PASS = os.getenv("GMAIL_APP_PASS") or os.getenv("SMTP_APP_PASSWORD")


# ==============================================================================
# CONFIGURATION & DATA STRUCTURES
# ==============================================================================

@dataclass
class MoodleConfig:
    """Configuration settings for Moodle submission ingestion."""
    target_url: str = TARGET_URL
    moodle_user: Optional[str] = MOODLE_USER
    moodle_pass: Optional[str] = MOODLE_PASS
    email_user: Optional[str] = GMAIL_USER
    email_pass: Optional[str] = GMAIL_APP_PASS
    base_dir: str = BASE_DIR
    extract_dirname: str = "extracted_submissions"
    imap_server: str = os.getenv("IMAP_SERVER", "imap.gmail.com")
    imap_folder: str = '"[Gmail]/All Mail"'
    otp_max_age_minutes: int = 30
    otp_timeout_seconds: int = 90
    browser_headless: bool = True
    browser_timeout_ms: int = 30000
    upload_to_drive: bool = os.getenv("MOODLE_UPLOAD_TO_DRIVE", "true").lower() not in ("false", "0", "no")
    drive_folder_id: Optional[str] = os.getenv("MOODLE_DRIVE_FOLDER_ID")
    drive_folder_name: str = "Moodle_Automated_Downloads"

    @property
    def extract_dir(self) -> str:
        return os.path.join(self.base_dir, self.extract_dirname)

    @property
    def zip_save_path(self) -> str:
        return os.path.join(self.base_dir, "all_submissions.zip")

    @property
    def csv_save_path(self) -> str:
        return os.path.join(self.base_dir, "grading_worksheet.csv")

    def validate(self) -> None:
        """Validate essential credentials."""
        missing = []
        if not self.moodle_user:
            missing.append("MOODLE_USER")
        if not self.moodle_pass:
            missing.append("MOODLE_PASS")
        if missing:
            raise ValueError(
                f"Missing required Moodle credential(s): {', '.join(missing)}. "
                "Please configure them in your .env file or pass them via MoodleConfig."
            )


@dataclass
class StudentSubmissionRecord:
    """Represents a student entry parsed from the grading worksheet paired with their files."""
    identifier: str
    full_name: str
    email_address: str = ""
    status: str = ""
    grade: Optional[str] = None
    online_text: str = ""
    student_folder: Optional[str] = None
    submission_files: List[str] = field(default_factory=list)
    drive_folder_url: Optional[str] = None
    drive_file_urls: List[str] = field(default_factory=list)
    raw_data: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_submission(self) -> bool:
        """True if the student submitted any file or online text."""
        return bool(self.submission_files or (self.online_text and self.online_text.strip()))

    @property
    def image_files(self) -> List[str]:
        """List of submission file paths that are images."""
        valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
        return [f for f in self.submission_files if Path(f).suffix.lower() in valid_exts]

    @property
    def video_files(self) -> List[str]:
        """List of submission file paths that are videos."""
        valid_exts = {".mp4", ".mov", ".avi", ".webm", ".mkv"}
        return [f for f in self.submission_files if Path(f).suffix.lower() in valid_exts]

    def load_images(self) -> List[Any]:
        """Loads and returns PIL Images for this student's image submissions."""
        from PIL import Image
        loaded = []
        for img_path in self.image_files:
            try:
                img = Image.open(img_path)
                if img.mode not in ("RGB", "L"):
                    img = img.convert("RGB")
                loaded.append(img)
            except Exception as err:
                logger.warning(f"Could not load image {img_path}: {err}")
        return loaded


@dataclass
class MoodleIngestionResult:
    """Structured result returned by the Moodle ingestion pipeline."""
    success: bool
    assignment_id: str
    assignment_url: str
    base_dir: str
    assignment_folder: str
    manifest_path: Optional[str] = None
    zip_path: Optional[str] = None
    csv_path: Optional[str] = None
    submissions_dir: Optional[str] = None
    extracted_files: List[str] = field(default_factory=list)
    student_records: List[StudentSubmissionRecord] = field(default_factory=list)
    has_submissions: bool = False
    has_worksheet: bool = False
    drive_folder_id: Optional[str] = None
    drive_folder_url: Optional[str] = None
    drive_uploaded_files: Dict[str, str] = field(default_factory=dict)
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)



# ==============================================================================
# SMART OTP INBOX READER (IMAP)
# ==============================================================================

def parse_email_for_otp(msg: email.message.Message) -> Tuple[Optional[str], datetime]:
    """
    Extracts a 6-digit OTP code and timestamp from an email message.
    """
    date_header = msg.get("Date")
    email_dt = parsedate_to_datetime(date_header) if date_header else datetime.min.replace(tzinfo=timezone.utc)
    
    raw_text = ""
    for part in (msg.walk() if msg.is_multipart() else [msg]):
        content_type = part.get_content_type()
        if content_type in ["text/plain", "text/html"]:
            payload = part.get_payload(decode=True)
            if payload:
                raw_text += payload.decode(errors="ignore") + " "

    soup = BeautifulSoup(raw_text, "html.parser")
    clean_text = f"{msg.get('Subject', '')} {soup.get_text(' ', strip=True)}"
    
    # Matches a standalone 6-digit number
    match = re.search(r'\b([0-9]{6})\b', clean_text)
    return (match.group(1) if match else None), email_dt


def get_active_or_fresh_otp(
    user_email: str,
    app_password: str,
    imap_server: str = "imap.gmail.com",
    imap_folder: str = '"[Gmail]/All Mail"',
    max_age_minutes: int = 30,
    timeout_seconds: int = 90,
) -> str:
    """
    Connects to IMAP, searches for recent or incoming OTP verification emails,
    and returns the 6-digit code.
    """
    clean_pass = app_password.replace(" ", "")
    logger.info(f"⏳ [IMAP]: Connecting to {imap_server} for {user_email}...")

    mail = imaplib.IMAP4_SSL(imap_server)
    try:
        mail.login(user_email, clean_pass)
        
        # Try configured folder with fallback to INBOX
        select_status, _ = mail.select(imap_folder)
        if select_status != "OK":
            logger.warning(f"⚠️ [IMAP]: Could not select {imap_folder}. Falling back to 'INBOX'.")
            mail.select("INBOX")

        status, data = mail.search(None, "ALL")
        mail_ids = data[0].split() if (status == "OK" and data and data[0]) else []
        now = datetime.now(timezone.utc)

        # Check existing recent emails
        if mail_ids:
            logger.info(f"⏳ [IMAP]: Checking for active OTP within last {max_age_minutes} minutes...")
            for mid in reversed(mail_ids[-5:]):
                _, msg_data = mail.fetch(mid, "(RFC822)")
                if not msg_data or not msg_data[0]:
                    continue
                msg = email.message_from_bytes(msg_data[0][1])
                otp, email_dt = parse_email_for_otp(msg)

                if otp:
                    age_minutes = (now - email_dt).total_seconds() / 60
                    if age_minutes <= max_age_minutes:
                        logger.info(f"⚡ [IMAP]: Found valid OTP from {age_minutes:.1f} min ago -> {otp}")
                        return otp

        # Poll for new incoming email
        logger.info(f"⏳ [IMAP]: Waiting for new incoming OTP email (timeout {timeout_seconds}s)...")
        baseline_count = len(mail_ids)
        start_time = time.time()

        while time.time() - start_time < timeout_seconds:
            time.sleep(3)
            try:
                # Re-select to refresh mailbox state
                mail.select(imap_folder)
                _, fresh_data = mail.search(None, "ALL")
                current_ids = fresh_data[0].split() if fresh_data and fresh_data[0] else []

                if len(current_ids) > baseline_count:
                    latest_id = current_ids[-1]
                    _, msg_data = mail.fetch(latest_id, "(RFC822)")
                    if msg_data and msg_data[0]:
                        msg = email.message_from_bytes(msg_data[0][1])
                        otp, _ = parse_email_for_otp(msg)
                        if otp:
                            logger.info(f"✅ [IMAP]: Captured fresh OTP -> {otp}")
                            return otp
            except Exception as loop_err:
                logger.debug(f"[IMAP] Polling attempt error: {loop_err}")

        raise TimeoutError("❌ [IMAP]: Timed out waiting for Moodle verification email.")
    finally:
        try:
            mail.logout()
        except Exception:
            pass


# ==============================================================================
# WORKSHEET & SUBMISSION PARSING HELPER
# ==============================================================================

def map_worksheet_to_submissions(
    csv_path: str,
    extracted_dir: str
) -> List[StudentSubmissionRecord]:
    """
    Parses grading_worksheet.csv and maps each student to extracted submission files and online text.
    """
    import csv

    if not os.path.exists(csv_path):
        return []

    # Get list of all extracted files
    all_files: List[str] = []
    if os.path.exists(extracted_dir):
        for root, _, filenames in os.walk(extracted_dir):
            for fn in filenames:
                if not fn.startswith(".") and fn != "__MACOSX":
                    all_files.append(os.path.join(root, fn))

    records: List[StudentSubmissionRecord] = []
    try:
        with open(csv_path, mode="r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                norm_row = {k.strip().lower(): v for k, v in row.items() if k}
                identifier = norm_row.get("identifier") or norm_row.get("id", "")
                full_name = (norm_row.get("full name") or norm_row.get("fullname") or norm_row.get("name", "")).strip()
                email_addr = norm_row.get("email address") or norm_row.get("email", "")
                status = norm_row.get("status", "")
                grade = norm_row.get("grade", "")
                online_text = norm_row.get("online text") or ""

                # Extract digits from identifier, e.g. "Participant 2788085" -> "2788085"
                id_digits_match = re.search(r'\d+', identifier)
                id_digits = id_digits_match.group(0) if id_digits_match else ""

                # Associate student files: Moodle naming format:
                # "Full Name_id_assignsubmission_file/...filename.ext" or "Full Name_id_assignsubmission_onlinetext/onlinetext.html"
                student_files: List[str] = []
                for fpath in all_files:
                    path_str = fpath.lower()
                    name_match = full_name and (full_name.lower() in path_str)
                    id_match = id_digits and (id_digits in path_str)

                    if name_match or id_match:
                        # If file is onlinetext.html, extract student's online text
                        if os.path.basename(fpath).lower() == "onlinetext.html":
                            try:
                                with open(fpath, "r", encoding="utf-8", errors="ignore") as tf:
                                    html_content = tf.read()
                                    soup = BeautifulSoup(html_content, "html.parser")
                                    extracted_text = soup.get_text(separator=" ", strip=True)
                                    if extracted_text and not online_text:
                                        online_text = extracted_text
                            except Exception as text_err:
                                logger.debug(f"Could not read onlinetext.html: {text_err}")
                        else:
                            student_files.append(fpath)

                records.append(
                    StudentSubmissionRecord(
                        identifier=identifier,
                        full_name=full_name,
                        email_address=email_addr,
                        status=status,
                        grade=grade,
                        online_text=online_text,
                        submission_files=student_files,
                        raw_data=dict(row),
                    )
                )
    except Exception as e:
        logger.warning(f"⚠️ Error parsing grading worksheet CSV: {e}")

    return records


# ==============================================================================
# USER ACCOUNT GOOGLE DRIVE UPLOAD (NO SERVICE ACCOUNT)
# ==============================================================================

def get_user_oauth_drive_service():
    """
    Returns an authenticated Google Drive v3 client strictly for the user's
    account (e.g. om@skillcatapp.com) using OAuth credentials.
    Explicitly refuses to use a service account.
    """
    refresh = (os.environ.get("GOOGLE_OAUTH_REFRESH_TOKEN") or os.environ.get("OAUTH_REFRESH_TOKEN", "")).strip()
    client_id = (os.environ.get("GOOGLE_OAUTH_CLIENT_ID") or os.environ.get("OAUTH_CLIENT_ID", "")).strip()
    client_secret = (os.environ.get("GOOGLE_OAUTH_CLIENT_SECRET") or os.environ.get("OAUTH_CLIENT_SECRET", "")).strip()

    if not refresh:
        raise ValueError(
            "User Google Drive OAuth token (GOOGLE_OAUTH_REFRESH_TOKEN) is not configured in .env.\n"
            "Per instructions, service accounts are NOT used.\n"
            "To authorize your account (om@skillcatapp.com), run:\n"
            "    python scripts/authenticate_user_drive.py"
        )
    if not client_id or not client_secret:
        raise ValueError("Missing GOOGLE_OAUTH_CLIENT_ID / OAUTH_CLIENT_ID or GOOGLE_OAUTH_CLIENT_SECRET / OAUTH_CLIENT_SECRET in .env.")

    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build

    creds = Credentials(
        None,
        refresh_token=refresh,
        token_uri="https://oauth2.googleapis.com/token",
        client_id=client_id,
        client_secret=client_secret,
        scopes=["https://www.googleapis.com/auth/drive"],
    )
    creds.refresh(Request())
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def extract_assignment_id(url: str) -> str:
    m = re.search(r'[?&]id=(\d+)', url)
    return m.group(1) if m else "unknown"


def organize_assignment_run(
    base_dir: str,
    target_url: str,
    raw_zip_path: Optional[str],
    raw_csv_path: Optional[str],
    student_records: List[StudentSubmissionRecord],
) -> Dict[str, Any]:
    """
    Organizes the downloaded assignment files into the clean structure:
    Assignment_{id}_{YYYY-MM-DD_HH-MM-SS}/
      all_submissions.zip
      grading_worksheet.csv
      Submissions/
        {Student_Name}_Participant_{ID}/
          online_text.txt
          student files...
      ingestion_manifest.json
    """
    import shutil

    assignment_id = extract_assignment_id(target_url)
    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_folder_name = f"Assignment_{assignment_id}_{run_timestamp}"
    run_folder_path = os.path.join(base_dir, run_folder_name)
    submissions_dir = os.path.join(run_folder_path, "Submissions")

    os.makedirs(run_folder_path, exist_ok=True)
    os.makedirs(submissions_dir, exist_ok=True)

    # 1. Copy all_submissions.zip into run folder
    final_zip_path = None
    if raw_zip_path and os.path.exists(raw_zip_path):
        final_zip_path = os.path.join(run_folder_path, "all_submissions.zip")
        shutil.copy2(raw_zip_path, final_zip_path)

    # 2. Copy grading_worksheet.csv into run folder
    final_csv_path = None
    if raw_csv_path and os.path.exists(raw_csv_path):
        final_csv_path = os.path.join(run_folder_path, "grading_worksheet.csv")
        shutil.copy2(raw_csv_path, final_csv_path)

    # 3. Organize each student's submission files and online text
    for student in student_records:
        if student.has_submission:
            id_slug = student.identifier.replace(" ", "_")
            student_folder_name = f"{student.full_name}_{id_slug}"
            student_dir = os.path.join(submissions_dir, student_folder_name)
            os.makedirs(student_dir, exist_ok=True)
            student.student_folder = student_dir

            # Write online_text.txt if text is present
            if student.online_text:
                text_file = os.path.join(student_dir, "online_text.txt")
                with open(text_file, "w", encoding="utf-8") as tf:
                    tf.write(student.online_text.strip())

            # Move/copy student's media files
            new_files = []
            for src_file in student.submission_files:
                if os.path.exists(src_file):
                    dest_file = os.path.join(student_dir, os.path.basename(src_file))
                    shutil.copy2(src_file, dest_file)
                    new_files.append(dest_file)
            student.submission_files = new_files
        else:
            student.student_folder = None

    # 4. Generate ingestion_manifest.json
    manifest_path = os.path.join(run_folder_path, "ingestion_manifest.json")
    manifest_data = {
        "assignment_id": assignment_id,
        "assignment_url": target_url,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "assignment_folder": run_folder_path,
        "grading_worksheet_csv": final_csv_path,
        "all_submissions_zip": final_zip_path,
        "total_students": len(student_records),
        "students_with_submissions": sum(1 for s in student_records if s.has_submission),
        "students": [
            {
                "identifier": s.identifier,
                "full_name": s.full_name,
                "email": s.email_address,
                "status": s.status,
                "online_text": s.online_text,
                "has_submission": s.has_submission,
                "student_folder": s.student_folder,
                "submission_files": s.submission_files,
                "drive_folder_url": s.drive_folder_url,
                "drive_file_urls": s.drive_file_urls,
            }
            for s in student_records
        ],
    }

    with open(manifest_path, "w", encoding="utf-8") as mf:
        json.dump(manifest_data, mf, indent=2)

    return {
        "assignment_id": assignment_id,
        "run_folder_name": run_folder_name,
        "run_folder_path": run_folder_path,
        "submissions_dir": submissions_dir,
        "zip_path": final_zip_path,
        "csv_path": final_csv_path,
        "manifest_path": manifest_path,
        "manifest_data": manifest_data,
    }


def upload_submissions_to_user_drive(
    run_info: Dict[str, Any],
    student_records: List[StudentSubmissionRecord],
    target_folder_id: Optional[str] = None,
    target_folder_name: str = "Moodle_Automated_Downloads",
) -> Dict[str, Any]:
    """
    Uploads the organized assignment run to Google Drive under om@skillcatapp.com:
    [Target Base Folder]/
      Assignment_{id}_{timestamp}/
        all_submissions.zip
        grading_worksheet.csv
        Submissions/
          {Student_Name}_Participant_{ID}/
            online_text.txt
            submission files...
        ingestion_manifest.json
    """
    from googleapiclient.http import MediaFileUpload

    drive_service = get_user_oauth_drive_service()
    logger.info("☁️ [Drive]: Authenticated with om@skillcatapp.com for Drive uploads.")

    # Base folder
    base_folder_id = target_folder_id.strip() if target_folder_id else None
    if base_folder_id:
        url_match = re.search(r'/folders/([a-zA-Z0-9_-]+)', base_folder_id)
        if url_match:
            base_folder_id = url_match.group(1)

    if not base_folder_id:
        query = f"name = '{target_folder_name}' and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
        res = drive_service.files().list(
            q=query, spaces='drive', fields='files(id, name, webViewLink)',
            supportsAllDrives=True, includeItemsFromAllDrives=True
        ).execute()
        files = res.get('files', [])
        if files:
            base_folder_id = files[0]['id']
        else:
            meta = {'name': target_folder_name, 'mimeType': 'application/vnd.google-apps.folder'}
            new_f = drive_service.files().create(body=meta, fields='id, webViewLink', supportsAllDrives=True).execute()
            base_folder_id = new_f['id']

    # Create Assignment run folder
    run_folder_name = run_info["run_folder_name"]
    run_folder_meta = {
        'name': run_folder_name,
        'parents': [base_folder_id],
        'mimeType': 'application/vnd.google-apps.folder'
    }
    run_f = drive_service.files().create(body=run_folder_meta, fields='id, webViewLink', supportsAllDrives=True).execute()
    run_folder_id = run_f['id']
    run_folder_url = run_f.get('webViewLink') or f"https://drive.google.com/drive/folders/{run_folder_id}"
    logger.info(f"📁 [Drive]: Created assignment folder: {run_folder_name} (ID: {run_folder_id})")

    uploaded_files: Dict[str, str] = {}

    def _upload_file(local_path: str, parent_id: str, mime: Optional[str] = None) -> Optional[str]:
        if not local_path or not os.path.exists(local_path):
            return None
        fname = os.path.basename(local_path)
        media = MediaFileUpload(local_path, mimetype=mime, resumable=True)
        file_meta = {'name': fname, 'parents': [parent_id]}
        up_file = drive_service.files().create(body=file_meta, media_body=media, fields='id, webViewLink', supportsAllDrives=True).execute()
        f_id = up_file.get('id', '')
        uploaded_files[fname] = f_id
        logger.info(f"   ☁️ Uploaded: {fname} (ID: {f_id})")
        return up_file.get('webViewLink') or f"https://drive.google.com/file/d/{f_id}/view"

    # Upload all_submissions.zip
    if run_info.get("zip_path"):
        _upload_file(run_info["zip_path"], run_folder_id, mime="application/zip")

    # Upload grading_worksheet.csv
    if run_info.get("csv_path"):
        _upload_file(run_info["csv_path"], run_folder_id, mime="text/csv")

    # Create Submissions/ folder
    submissions_meta = {
        'name': 'Submissions',
        'parents': [run_folder_id],
        'mimeType': 'application/vnd.google-apps.folder'
    }
    sub_f = drive_service.files().create(body=submissions_meta, fields='id, webViewLink', supportsAllDrives=True).execute()
    sub_folder_id = sub_f['id']

    # For each student with submissions
    for student in student_records:
        if student.has_submission and student.student_folder and os.path.exists(student.student_folder):
            sf_name = os.path.basename(student.student_folder)
            sf_meta = {
                'name': sf_name,
                'parents': [sub_folder_id],
                'mimeType': 'application/vnd.google-apps.folder'
            }
            sf_obj = drive_service.files().create(body=sf_meta, fields='id, webViewLink', supportsAllDrives=True).execute()
            sf_id = sf_obj['id']
            student.drive_folder_url = sf_obj.get('webViewLink') or f"https://drive.google.com/drive/folders/{sf_id}"

            for item in os.listdir(student.student_folder):
                item_path = os.path.join(student.student_folder, item)
                if os.path.isfile(item_path):
                    link = _upload_file(item_path, sf_id)
                    if link:
                        student.drive_file_urls.append(link)

    # Re-save manifest with drive URLs and upload
    manifest_data = run_info["manifest_data"]
    manifest_data["drive_folder_url"] = run_folder_url
    for s_dict in manifest_data["students"]:
        matched = next((s for s in student_records if s.identifier == s_dict["identifier"]), None)
        if matched:
            s_dict["drive_folder_url"] = matched.drive_folder_url
            s_dict["drive_file_urls"] = matched.drive_file_urls

    manifest_path = run_info["manifest_path"]
    with open(manifest_path, "w", encoding="utf-8") as mf:
        json.dump(manifest_data, mf, indent=2)

    _upload_file(manifest_path, run_folder_id, mime="application/json")

    return {
        "folder_id": run_folder_id,
        "folder_url": run_folder_url,
        "uploaded_files": uploaded_files,
    }




# ==============================================================================
# MAIN BROWSER ORCHESTRATION PIPELINE
# ==============================================================================

class MoodleSubmissionDownloader:
    """
    Encapsulates Playwright automation for Moodle login, MFA resolution,
    and downloading submission ZIPs and grading worksheets.
    """

    def __init__(self, config: Optional[MoodleConfig] = None):
        self.config = config or MoodleConfig()

    async def execute(self) -> MoodleIngestionResult:
        """
        Runs the end-to-end Moodle ingestion automation.
        """
        self.config.validate()

        os.makedirs(self.config.base_dir, exist_ok=True)
        os.makedirs(self.config.extract_dir, exist_ok=True)

        parsed_target = urlparse(self.config.target_url)
        base_origin = f"{parsed_target.scheme}://{parsed_target.netloc}"
        moodle_login_url = f"{base_origin}/login/index.php"

        has_submissions = False
        has_worksheet = False
        extracted_files: List[str] = []
        student_records: List[StudentSubmissionRecord] = []

        try:
            from playwright.async_api import async_playwright
        except ImportError:
            raise ImportError(
                "playwright is required for Moodle ingestion. Run: pip install playwright && playwright install chromium"
            )

        async with async_playwright() as p:
            logger.info("🚀 [1/4] Launching Playwright browser...")
            browser = await p.chromium.launch(
                headless=self.config.browser_headless,
                args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-blink-features=AutomationControlled"],
            )

            session_file = os.path.join(self.config.base_dir, ".moodle_session.json")
            candidate_session_files = [
                session_file,
                os.path.join("downloads/moodle_submissions", ".moodle_session.json"),
                "./MyDrive/Moodle_Automated_Downloads/.moodle_session.json",
            ]
            storage_state = next((f for f in candidate_session_files if os.path.exists(f)), None)

            try:
                context_kwargs = {
                    "user_agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                    ),
                    "accept_downloads": True,
                }
                if storage_state:
                    context_kwargs["storage_state"] = storage_state
                    logger.info(f"🔄 Reusing saved session from: {storage_state}")

                context = await browser.new_context(**context_kwargs)
                page = await context.new_page()

                # --------------------------------------------------------------
                # STEP 1: NAVIGATE TO TARGET ASSIGNMENT / CHECK AUTH
                # --------------------------------------------------------------
                logger.info(f"🌐 [2/4] Navigating to: {self.config.target_url}...")
                await page.goto(self.config.target_url, wait_until="networkidle", timeout=self.config.browser_timeout_ms)

                needs_login = (
                    "login" in page.url
                    or "auth.php" in page.url
                    or "tool/mfa" in page.url
                    or await page.locator("#username, input[name='username']").count() > 0
                )

                if needs_login:
                    logger.info(f"🔑 Authentication required. Submitting credentials for {self.config.moodle_user}...")
                    if "login" not in page.url:
                        await page.goto(moodle_login_url, wait_until="networkidle", timeout=self.config.browser_timeout_ms)

                    user_input = page.locator("#username, input[name='username']").first
                    if await user_input.is_visible(timeout=3000):
                        await user_input.fill(self.config.moodle_user or "")
                        await page.fill("#password, input[name='password']", self.config.moodle_pass or "")
                        await page.click("#loginbtn, button[type='submit']")
                        await page.wait_for_load_state("networkidle")
                        await page.wait_for_timeout(2000)

                    # --------------------------------------------------------------
                    # STEP 2: ADAPTIVE MFA / OTP RESOLUTION
                    # --------------------------------------------------------------
                    mfa_needed = (
                        "tool/mfa" in page.url
                        or "auth.php" in page.url
                        or await page.locator("#id_verificationcode, input[name*='code'], #id_code").count() > 0
                    )

                    if mfa_needed:
                        logger.info("🔒 [3/4] MFA verification detected!")
                        if not self.config.email_user or not self.config.email_pass:
                            raise ValueError(
                                "MFA verification is required by Moodle, but GMAIL_USER / GMAIL_APP_PASS "
                                "(or SMTP credentials) are not configured."
                            )

                        otp_code = await asyncio.to_thread(
                            get_active_or_fresh_otp,
                            self.config.email_user,
                            self.config.email_pass,
                            self.config.imap_server,
                            self.config.imap_folder,
                            self.config.otp_max_age_minutes,
                            self.config.otp_timeout_seconds,
                        )

                        code_input_selector = "#id_verificationcode, input[type='text'], input[name*='code'], #id_code"
                        await page.wait_for_selector(code_input_selector, timeout=10000)
                        logger.info(f"✍️ Submitting OTP code: {otp_code}...")
                        await page.fill(code_input_selector, otp_code)

                        try:
                            continue_btn = page.locator("button:has-text('Continue'), input[value='Continue'], #id_submitbutton").first
                            if await continue_btn.is_visible(timeout=2000):
                                await continue_btn.click(timeout=3000, no_wait_after=True)
                        except Exception as btn_err:
                            logger.debug(f"Continue button click handled: {btn_err}")

                        for _ in range(15):
                            if "auth.php" not in page.url and "login" not in page.url and "tool/mfa" not in page.url:
                                break
                            await page.wait_for_timeout(1000)

                    logger.info(f"🎉 Login verified successfully! Current page: {page.url}")
                    if "action=" not in page.url:
                        await page.goto(self.config.target_url, wait_until="networkidle", timeout=self.config.browser_timeout_ms)
                else:
                    logger.info(f"🎉 Active Moodle session verified! Reused session without re-authenticating.")

                # Save authenticated browser storage state for reuse in Upload and future runs
                try:
                    await context.storage_state(path=session_file)
                    logger.info(f"💾 Saved authenticated session to {session_file}")
                except Exception as s_err:
                    logger.debug(f"Could not save session state: {s_err}")

                # --------------------------------------------------------------
                # TASK 1: DOWNLOAD ALL SUBMISSIONS (ZIP) & EXTRACT
                # --------------------------------------------------------------
                logger.info("📦 Checking for All Submissions ZIP download...")
                download_btn = page.locator(
                    "a:has-text('Download all submissions'), "
                    "button:has-text('Download all submissions'), "
                    "a[href*='action=downloadall']"
                ).first

                try:
                    if await download_btn.is_visible(timeout=5000):
                        async with page.expect_download(timeout=30000) as download_info:
                            await download_btn.click()
                        download_zip = await download_info.value
                        await download_zip.save_as(self.config.zip_save_path)
                        has_submissions = True
                        logger.info(f"   ✅ Saved submissions ZIP: {self.config.zip_save_path}")
                    else:
                        # Fallback direct URL
                        base_grading_url = page.url.split("&action")[0]
                        direct_zip_url = f"{base_grading_url}&action=downloadall"
                        logger.info(f"   Navigating directly to zip download: {direct_zip_url}")
                        async with page.expect_download(timeout=30000) as download_info:
                            try:
                                await page.goto(direct_zip_url)
                            except Exception as err:
                                if "Download is starting" not in str(err):
                                    raise err
                        download_zip = await download_info.value
                        await download_zip.save_as(self.config.zip_save_path)
                        has_submissions = True
                        logger.info(f"   ✅ Saved submissions ZIP via fallback: {self.config.zip_save_path}")

                    # Extract the zip
                    if has_submissions and os.path.exists(self.config.zip_save_path):
                        logger.info(f"   📂 Unpacking files to: {self.config.extract_dir} ...")
                        with zipfile.ZipFile(self.config.zip_save_path, 'r') as zip_ref:
                            zip_ref.extractall(self.config.extract_dir)

                        for root, _, fnames in os.walk(self.config.extract_dir):
                            for fn in fnames:
                                if not fn.startswith("."):
                                    extracted_files.append(os.path.join(root, fn))
                        logger.info(f"   ✅ Successfully extracted {len(extracted_files)} files!")

                except Exception as zip_err:
                    if "Timeout" in str(zip_err):
                        logger.info("   ℹ️ No student files downloaded (the assignment currently has 0 submissions or timed out).")
                    else:
                        logger.warning(f"   ⚠️ Submissions ZIP download warning: {zip_err}")

                # Ensure we are back on the grading view before worksheet download
                if page.url != self.config.target_url:
                    await page.goto(self.config.target_url, wait_until="networkidle", timeout=self.config.browser_timeout_ms)

                # --------------------------------------------------------------
                # TASK 2: DOWNLOAD GRADING WORKSHEET (CSV)
                # --------------------------------------------------------------
                logger.info("📑 Checking for Grading Worksheet CSV download...")
                try:
                    opt_elem = page.locator("option:has-text('Download grading worksheet')").first
                    worksheet_url = None
                    if await opt_elem.count() > 0:
                        worksheet_url = await opt_elem.get_attribute("value")

                    if worksheet_url:
                        if worksheet_url.startswith("/"):
                            worksheet_url = f"{base_origin}{worksheet_url}"

                        async with page.expect_download(timeout=30000) as download_info:
                            await page.evaluate(f"window.location.href = '{worksheet_url}'")
                        download_csv = await download_info.value
                        await download_csv.save_as(self.config.csv_save_path)
                        has_worksheet = True
                        logger.info(f"   ✅ Saved grading worksheet CSV: {self.config.csv_save_path}")
                    else:
                        # Fallback direct worksheet URL pattern
                        worksheet_fallback_url = f"{self.config.target_url.split('&action')[0]}&action=downloadgradingworksheet"
                        try:
                            async with page.expect_download(timeout=10000) as download_info:
                                await page.evaluate(f"window.location.href = '{worksheet_fallback_url}'")
                            download_csv = await download_info.value
                            await download_csv.save_as(self.config.csv_save_path)
                            has_worksheet = True
                            logger.info(f"   ✅ Saved grading worksheet via fallback: {self.config.csv_save_path}")
                        except Exception:
                            logger.info("   ℹ️ 'Download grading worksheet' option not found on page.")

                except Exception as csv_err:
                    logger.warning(f"   ⚠️ Worksheet CSV download error: {csv_err}")

                # If worksheet is downloaded, map students to submissions
                if has_worksheet and os.path.exists(self.config.csv_save_path):
                    student_records = map_worksheet_to_submissions(
                        self.config.csv_save_path,
                        self.config.extract_dir
                    )
                    logger.info(f"   📋 Mapped {len(student_records)} student records from worksheet.")

            finally:
                await browser.close()

        # Organize assignment run into the clean hierarchy:
        # Assignment_{id}_{timestamp}/
        #   all_submissions.zip
        #   grading_worksheet.csv
        #   Submissions/
        #     {Student_Name}_Participant_{ID}/
        #       online_text.txt
        #       submission files...
        #   ingestion_manifest.json
        run_info = organize_assignment_run(
            base_dir=self.config.base_dir,
            target_url=self.config.target_url,
            raw_zip_path=self.config.zip_save_path if has_submissions else None,
            raw_csv_path=self.config.csv_save_path if has_worksheet else None,
            student_records=student_records,
        )

        # Copy session state into run folder for upload reuse
        if os.path.exists(session_file):
            try:
                import shutil
                shutil.copy2(session_file, os.path.join(run_info["run_folder_path"], ".moodle_session.json"))
            except Exception:
                pass
        drive_folder_id = None
        drive_folder_url = None
        drive_uploaded_files = {}

        if self.config.upload_to_drive:
            try:
                logger.info("☁️ Uploading submissions to Google Drive (om@skillcatapp.com)...")
                drive_info = upload_submissions_to_user_drive(
                    run_info=run_info,
                    student_records=student_records,
                    target_folder_id=self.config.drive_folder_id,
                    target_folder_name=self.config.drive_folder_name,
                )
                drive_folder_id = drive_info.get("folder_id")
                drive_folder_url = drive_info.get("folder_url")
                drive_uploaded_files = drive_info.get("uploaded_files", {})
                logger.info(f"   🎉 Drive upload complete: {drive_folder_url}")
            except Exception as drive_err:
                logger.error(f"❌ Google Drive user account upload failed: {drive_err}")
                raise

        return MoodleIngestionResult(
            success=True,
            assignment_id=run_info["assignment_id"],
            assignment_url=self.config.target_url,
            base_dir=self.config.base_dir,
            assignment_folder=run_info["run_folder_path"],
            manifest_path=run_info["manifest_path"],
            zip_path=run_info["zip_path"],
            csv_path=run_info["csv_path"],
            submissions_dir=run_info["submissions_dir"],
            extracted_files=extracted_files,
            student_records=student_records,
            has_submissions=has_submissions,
            has_worksheet=has_worksheet,
            drive_folder_id=drive_folder_id,
            drive_folder_url=drive_folder_url,
            drive_uploaded_files=drive_uploaded_files,
            metadata={"timestamp": datetime.now(timezone.utc).isoformat()},
        )



# ==============================================================================
# PIPELINE ENTRYPOINTS (ASYNC & SYNC)
# ==============================================================================

async def run_moodle_ingestion(config: Optional[MoodleConfig] = None) -> MoodleIngestionResult:
    """
    Asynchronous pipeline entry point.
    """
    downloader = MoodleSubmissionDownloader(config=config)
    return await downloader.execute()


def run_moodle_ingestion_sync(config: Optional[MoodleConfig] = None) -> MoodleIngestionResult:
    """
    Synchronous pipeline entry point safe for usage in scripts, Streamlit, or worker threads.
    Handles existing running event loops gracefully.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(lambda: asyncio.run(run_moodle_ingestion(config)))
            return future.result()
    else:
        return asyncio.run(run_moodle_ingestion(config))


# ==============================================================================
# CLI EXECUTION
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Moodle Assignment Submissions Ingestion & Downloader Pipeline."
    )
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="Target Moodle assignment grading URL (defaults to MOODLE_ASSIGNMENT_URL env var).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Base directory for downloaded files (defaults to MOODLE_DOWNLOAD_DIR env var or ./downloads/moodle_submissions).",
    )
    parser.add_argument(
        "--moodle-user",
        type=str,
        default=None,
        help="Moodle username (defaults to MOODLE_USER env var).",
    )
    parser.add_argument(
        "--moodle-pass",
        type=str,
        default=None,
        help="Moodle password (defaults to MOODLE_PASS env var).",
    )
    parser.add_argument(
        "--email-user",
        type=str,
        default=None,
        help="Gmail address for OTP retrieval (defaults to GMAIL_USER or SMTP_USER env var).",
    )
    parser.add_argument(
        "--email-pass",
        type=str,
        default=None,
        help="Gmail App password for OTP retrieval (defaults to GMAIL_APP_PASS or SMTP_APP_PASSWORD env var).",
    )
    parser.add_argument(
        "--no-drive",
        action="store_true",
        help="Skip uploading to Google Drive (uploading to Drive is enabled by default).",
    )
    parser.add_argument(
        "--upload-to-drive",
        action="store_true",
        help="Explicitly enable Google Drive upload (already enabled by default).",
    )
    parser.add_argument(
        "--drive-folder-id",
        type=str,
        default=None,
        help="Target Google Drive folder ID (defaults to MOODLE_DRIVE_FOLDER_ID from .env).",
    )
    parser.add_argument(
        "--drive-folder-name",
        type=str,
        default="Moodle_Automated_Downloads",
        help="Folder name in My Drive to create/use for submissions (default: Moodle_Automated_Downloads).",
    )
    parser.add_argument(
        "--headed",
        action="store_true",
        help="Run browser in visible headed mode (useful for debugging).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30000,
        help="Browser navigation timeout in milliseconds.",
    )

    args = parser.parse_args()

    # Upload to drive is enabled by default unless --no-drive or MOODLE_UPLOAD_TO_DRIVE=false
    env_upload = os.getenv("MOODLE_UPLOAD_TO_DRIVE", "true").lower() not in ("false", "0", "no")
    upload_to_drive = False if args.no_drive else (True if args.upload_to_drive else env_upload)

    config = MoodleConfig(
        target_url=args.url or os.getenv("MOODLE_ASSIGNMENT_URL", "https://planning.guroo.app/mod/assign/view.php?id=11922&action=grading"),
        moodle_user=args.moodle_user or os.getenv("MOODLE_USER"),
        moodle_pass=args.moodle_pass or os.getenv("MOODLE_PASS"),
        email_user=args.email_user or os.getenv("GMAIL_USER") or os.getenv("SMTP_USER"),
        email_pass=args.email_pass or os.getenv("GMAIL_APP_PASS") or os.getenv("SMTP_APP_PASSWORD"),
        base_dir=args.output_dir or BASE_DIR,
        upload_to_drive=upload_to_drive,
        drive_folder_id=args.drive_folder_id or os.getenv("MOODLE_DRIVE_FOLDER_ID"),
        drive_folder_name=args.drive_folder_name,
        browser_headless=not args.headed,
        browser_timeout_ms=args.timeout,
    )

    try:
        result = run_moodle_ingestion_sync(config)
        logger.info(f"\n🎉 Ingestion finished with success={result.success}")
        logger.info(f"   Submissions found: {result.has_submissions} ({len(result.extracted_files)} files)")
        logger.info(f"   Worksheet found: {result.has_worksheet} ({len(result.student_records)} students)")
        if result.zip_path:
            logger.info(f"   ZIP path: {result.zip_path}")
        if result.csv_path:
            logger.info(f"   CSV path: {result.csv_path}")
        if result.assignment_folder:
            logger.info(f"   Assignment folder: {result.assignment_folder}")
        if result.submissions_dir:
            logger.info(f"   Submissions dir: {result.submissions_dir}")
        if result.drive_folder_url:
            logger.info(f"   ☁️ Drive Folder URL: {result.drive_folder_url}")
    except Exception as exc:
        logger.error(f"❌ Ingestion failed: {exc}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()