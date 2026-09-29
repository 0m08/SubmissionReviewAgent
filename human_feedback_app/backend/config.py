from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"
BRAND_ASSETS_DIR = REPO_ROOT / "assets"
BRAND_FALLBACK_DIR = FRONTEND_DIR / "brand"

OAUTH_CLIENT_ID = os.getenv("OAUTH_CLIENT_ID", "").strip()
OAUTH_CLIENT_SECRET = os.getenv("OAUTH_CLIENT_SECRET", "").strip()
OAUTH_REDIRECT_URI = (
    os.getenv("OAUTH_REDIRECT_URI_HUMAN_FEEDBACK", "").strip()
    or os.getenv("OAUTH_REDIRECT_URI_LIGHTNING", "").strip()
    or os.getenv("OAUTH_REDIRECT_URI", "").strip()
    or "http://localhost:8000/auth/callback"
)

SESSION_SECRET = os.getenv("HUMAN_FEEDBACK_SESSION_SECRET", "change-me-in-production")
SESSION_COOKIE = "hf_session_id"

LLM_DEFAULT = os.getenv("HUMAN_FEEDBACK_LLM", "gemini_3_8_flash_thinking")
REVISE_MAX_WORKERS = 2
