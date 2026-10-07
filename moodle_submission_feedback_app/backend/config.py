# -*- coding: utf-8 -*-
"""Configuration for Moodle Submission Feedback App."""

from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
REPO_ROOT = Path(__file__).resolve().parents[2]
BRAND_ASSETS_DIR = REPO_ROOT / "assets"

DEFAULT_DRIVE_FOLDER_ID = os.environ.get("MOODLE_DRIVE_FOLDER_ID", "1jR5JP3aTKt8u_X57Z31IS71ZGBX-zfeu")
REGISTRY_SHEET_ID = os.environ.get("CENTRAL_REVIEW_SHEET_ID", "1aP7Xdvoi4TTIT8K7X9j2wD0G6-YCnA7sEn-BiLDWWQU")

# Google OAuth Configuration (mirrors human_feedback_app)
OAUTH_CLIENT_ID = os.getenv("OAUTH_CLIENT_ID", "").strip()
OAUTH_CLIENT_SECRET = os.getenv("OAUTH_CLIENT_SECRET", "").strip()
OAUTH_REDIRECT_URI = (
    os.getenv("OAUTH_REDIRECT_URI_MOODLE_REVIEW", "").strip()
    or os.getenv("OAUTH_REDIRECT_URI_MOODLE_FEEDBACK", "").strip()
    or os.getenv("OAUTH_REDIRECT_URI_LIGHTNING", "").strip()
    or os.getenv("OAUTH_REDIRECT_URI", "").strip()
    or "http://localhost:8000/auth/callback"
)
SESSION_COOKIE = "moodle_review_session_id"
DEV_BYPASS_AUTH = os.getenv("DEV_BYPASS_AUTH", "false").lower() in ("true", "1", "yes")

PORT = int(os.environ.get("PORT", "8000"))
HOST = os.environ.get("HOST", "0.0.0.0")
