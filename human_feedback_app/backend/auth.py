"""Google OAuth routes."""

from __future__ import annotations

import secrets
import traceback
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from human_feedback_app.backend.config import (
    OAUTH_CLIENT_ID,
    OAUTH_CLIENT_SECRET,
    OAUTH_REDIRECT_URI,
    SESSION_COOKIE,
)
from human_feedback_app.backend.jobs import revision_queue
from human_feedback_app.backend.sessions import session_store
from human_feedback_app.backend.sheet_service import clear_visual_revision_action
from services.drive_service import (
    exchange_code_for_credentials,
    get_google_oauth_authorization_url,
    init_clients_from_credentials,
)
from utils.role_utils import get_user_info

router = APIRouter(prefix="/auth", tags=["auth"])


def _session_id_from_request(request: Request) -> Optional[str]:
    return request.cookies.get(SESSION_COOKIE)


def get_current_session(request: Request):
    session = session_store.get(_session_id_from_request(request))
    if session is None or not session.user_email:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return session


def ensure_google_clients(session):
    """Ensure the authenticated session has Drive and Sheets clients attached."""
    if session.drive is not None and session.gc is not None:
        return
    if session.oauth_credentials is None:
        raise HTTPException(
            status_code=401,
            detail="Session is missing Google clients. Please sign in again.",
        )
    _, drive, gc = init_clients_from_credentials(
        session.oauth_credentials,
        client_id=OAUTH_CLIENT_ID,
        client_secret=OAUTH_CLIENT_SECRET,
    )
    session.drive = drive
    session.gc = gc


@router.get("/login")
def login(request: Request):
    if not OAUTH_CLIENT_ID or not OAUTH_CLIENT_SECRET:
        raise HTTPException(status_code=500, detail="OAuth is not configured")
    auth_url, state = get_google_oauth_authorization_url(
        OAUTH_CLIENT_ID, OAUTH_CLIENT_SECRET, OAUTH_REDIRECT_URI
    )
    pending = session_store.create("", "", [])
    pending.oauth_state = state
    response = RedirectResponse(auth_url, status_code=302)
    response.set_cookie(SESSION_COOKIE, pending.session_id, httponly=True, samesite="lax")
    return response


@router.get("/callback")
def callback(request: Request, code: str = "", state: str = ""):
    session = session_store.get(_session_id_from_request(request))
    if session is None:
        raise HTTPException(status_code=400, detail="Missing login session")
    if not session.oauth_state or not state or state != session.oauth_state:
        raise HTTPException(status_code=400, detail="OAuth state mismatch")

    creds = exchange_code_for_credentials(
        OAUTH_CLIENT_ID, OAUTH_CLIENT_SECRET, OAUTH_REDIRECT_URI, code
    )
    _, drive, gc = init_clients_from_credentials(
        creds, client_id=OAUTH_CLIENT_ID, client_secret=OAUTH_CLIENT_SECRET
    )
    about = drive.GetAbout()
    user_email = (about.get("user") or {}).get("emailAddress") or ""
    user_info = get_user_info(user_email)
    if not user_info["is_authorized"]:
        session_store.delete(session.session_id)
        return HTMLResponse(
            f"<h2>Access denied</h2><p>{user_email} is not authorized.</p>",
            status_code=403,
        )
    if "aggregation_agent_page" not in user_info["pages"]:
        session_store.delete(session.session_id)
        return HTMLResponse(
            f"<h2>Access denied</h2><p>{user_email} does not have access to Human Feedback review.</p>",
            status_code=403,
        )

    session.user_email = user_email
    session.role = user_info["role"]
    session.pages = user_info["pages"]
    session.drive = drive
    session.gc = gc
    session.oauth_credentials = creds
    session.refresh_token = creds.refresh_token
    session.oauth_state = None

    response = RedirectResponse("/setup", status_code=302)
    response.set_cookie(SESSION_COOKIE, session.session_id, httponly=True, samesite="lax")
    return response


@router.post("/logout")
def logout_post(request: Request):
    return _logout_response(request)


@router.get("/logout")
def logout_get(request: Request):
    return _logout_response(request)


def _logout_response(request: Request):
    session = session_store.get(_session_id_from_request(request))
    if session:
        # 1) Cancel in-flight jobs so workers will not mark results completed.
        cancelled = revision_queue.cancel_session_jobs(session.session_id)

        # 2) Block further sheet writes immediately (workers check this under the write lock).
        session.aborted = True

        # 3) Clear durable "revising" reject actions for unfinished visual jobs while Google clients are still usable.
        if session.sheet_link and cancelled:
            try:
                ensure_google_clients(session)
                for job in cancelled:
                    if job.kind != "visual" or job.row_index is None or not job.visual_id:
                        continue
                    try:
                        clear_visual_revision_action(
                            session,
                            int(job.row_index),
                            job.visual_id,
                            allow_when_aborted=True,
                        )
                    except Exception:
                        traceback.print_exc()
            except Exception:
                traceback.print_exc()

        session_store.delete(session.session_id)

    response = RedirectResponse("/login-page", status_code=302)
    response.delete_cookie(SESSION_COOKIE)
    return response
