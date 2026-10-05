# -*- coding: utf-8 -*-
"""Google OAuth authentication routes and session management."""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from moodle_submission_feedback_app.backend.config import (
    DEV_BYPASS_AUTH,
    OAUTH_CLIENT_ID,
    OAUTH_CLIENT_SECRET,
    OAUTH_REDIRECT_URI,
    SESSION_COOKIE,
)
from moodle_submission_feedback_app.backend.sessions import MentorSession, session_store
from services.drive_service import (
    exchange_code_for_credentials,
    get_google_oauth_authorization_url,
    init_clients_from_credentials,
)
from utils.role_utils import get_user_info

logger = logging.getLogger("moodle_feedback_app.auth")
router = APIRouter(prefix="/auth", tags=["auth"])


def _session_id_from_request(request: Request) -> Optional[str]:
    return request.cookies.get(SESSION_COOKIE)


def get_current_session(request: Request) -> MentorSession:
    """Dependency to enforce authenticated mentor session."""
    if DEV_BYPASS_AUTH:
        bypass = session_store.get("dev-bypass-session")
        if not bypass:
            bypass = MentorSession(
                session_id="dev-bypass-session",
                user_email="om@skillcatapp.com",
                mentor_name="Om Aryan",
                role="Admin",
                avatar_initials="OA",
            )
            session_store._sessions["dev-bypass-session"] = bypass
        return bypass

    session = session_store.get(_session_id_from_request(request))
    if session is None or not session.user_email:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return session


def get_optional_session(request: Request) -> Optional[MentorSession]:
    """Dependency to retrieve session if authenticated, or None."""
    if DEV_BYPASS_AUTH:
        return get_current_session(request)
    session_id = _session_id_from_request(request)
    if not session_id:
        return None
    session = session_store.get(session_id)
    if session and session.user_email:
        return session
    return None


@router.get("/login")
def login(request: Request):
    """Initiates Google OAuth sign-in flow."""
    if not OAUTH_CLIENT_ID or not OAUTH_CLIENT_SECRET:
        logger.error("OAuth client credentials not configured")
        raise HTTPException(status_code=500, detail="OAuth credentials not configured in .env")

    auth_url, state = get_google_oauth_authorization_url(
        OAUTH_CLIENT_ID, OAUTH_CLIENT_SECRET, OAUTH_REDIRECT_URI
    )
    pending = session_store.create(user_email="", mentor_name="", role="")
    pending.oauth_state = state

    response = RedirectResponse(auth_url, status_code=302)
    response.set_cookie(
        SESSION_COOKIE,
        pending.session_id,
        httponly=True,
        samesite="lax",
        max_age=86400 * 7,
    )
    return response


@router.get("/callback")
def callback(request: Request, code: str = "", state: str = ""):
    """Handles OAuth callback and verifies user credentials."""
    session = session_store.get(_session_id_from_request(request))
    if session is None:
        raise HTTPException(status_code=400, detail="Missing login session")
    if not session.oauth_state or not state or state != session.oauth_state:
        raise HTTPException(status_code=400, detail="OAuth state mismatch or expired")

    try:
        creds = exchange_code_for_credentials(
            OAUTH_CLIENT_ID, OAUTH_CLIENT_SECRET, OAUTH_REDIRECT_URI, code
        )
        _, drive, gc = init_clients_from_credentials(
            creds, client_id=OAUTH_CLIENT_ID, client_secret=OAUTH_CLIENT_SECRET
        )
        about = drive.GetAbout()
        user_obj = about.get("user") or {}
        user_email = (user_obj.get("emailAddress") or "").strip()
        display_name = (user_obj.get("displayName") or "").strip()
    except Exception as e:
        logger.error(f"Failed to exchange OAuth code: {e}")
        session_store.delete(session.session_id)
        return HTMLResponse(
            f"<h2>Authentication failed</h2><p>{str(e)}</p><p><a href='/login-page'>Back to login</a></p>",
            status_code=400,
        )

    user_info = get_user_info(user_email)
    is_authorized = user_info.get("is_authorized") or user_email.lower().endswith("@skillcatapp.com")
    if not is_authorized:
        session_store.delete(session.session_id)
        return HTMLResponse(
            f"<h2>Access denied</h2><p><b>{user_email}</b> is not authorized to access Moodle Review Studio.</p><p><a href='/login-page'>Try another account</a></p>",
            status_code=403,
        )

    # Derive mentor friendly name and avatar initials
    mentor_name = display_name
    if not mentor_name:
        if user_email.lower().startswith("om@"):
            mentor_name = "Om Aryan"
        else:
            mentor_name = user_email.split("@")[0].replace(".", " ").title()

    initials = "".join([part[0].upper() for part in mentor_name.split()[:2]]) or "ME"

    session.user_email = user_email
    session.mentor_name = mentor_name
    session.role = user_info.get("role") or "Mentor"
    session.avatar_initials = initials
    session.drive = drive
    session.gc = gc
    session.oauth_credentials = creds
    session.oauth_state = None

    response = RedirectResponse("/", status_code=302)
    response.set_cookie(
        SESSION_COOKIE,
        session.session_id,
        httponly=True,
        samesite="lax",
        max_age=86400 * 7,
    )
    return response


@router.get("/logout")
def logout_get(request: Request):
    return _logout_response(request)


@router.post("/logout")
def logout_post(request: Request):
    return _logout_response(request)


def _logout_response(request: Request) -> RedirectResponse:
    session_id = _session_id_from_request(request)
    if session_id:
        session_store.delete(session_id)

    response = RedirectResponse("/login-page", status_code=302)
    response.delete_cookie(SESSION_COOKIE)
    return response
