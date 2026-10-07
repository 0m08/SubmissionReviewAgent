# -*- coding: utf-8 -*-
"""Google OAuth authentication routes and session management."""

from __future__ import annotations

import logging
import os
from typing import Optional

import requests

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


def _is_request_https(request: Request) -> bool:
    return request.url.scheme == "https" or request.headers.get("x-forwarded-proto") == "https"


def _resolve_redirect_uri(request: Request) -> str:
    """
    Resolves OAuth redirect URI:
    - If explicit OAUTH_REDIRECT_URI_MOODLE_REVIEW is set, honors it.
    - If explicit OAUTH_REDIRECT_URI_MOODLE_FEEDBACK or OAUTH_REDIRECT_URI_LIGHTNING is set, honors it.
    - Otherwise falls back to configured OAUTH_REDIRECT_URI.
    Does not dynamically construct unwhitelisted hostnames to avoid Google redirect_uri_mismatch.
    """
    explicit = (
        os.getenv("OAUTH_REDIRECT_URI_MOODLE_REVIEW", "").strip()
        or os.getenv("OAUTH_REDIRECT_URI_MOODLE_FEEDBACK", "").strip()
        or os.getenv("OAUTH_REDIRECT_URI_LIGHTNING", "").strip()
    )
    if explicit:
        return explicit

    return OAUTH_REDIRECT_URI


@router.get("/login")
def login(request: Request):
    """Initiates Google OAuth sign-in flow."""
    if not OAUTH_CLIENT_ID or not OAUTH_CLIENT_SECRET:
        logger.error("OAuth client credentials not configured")
        raise HTTPException(status_code=500, detail="OAuth credentials not configured in .env")

    redirect_uri = _resolve_redirect_uri(request)
    auth_url, state = get_google_oauth_authorization_url(
        OAUTH_CLIENT_ID, OAUTH_CLIENT_SECRET, redirect_uri
    )
    pending = session_store.create(user_email="", mentor_name="", role="")
    pending.oauth_state = state
    pending.redirect_uri = redirect_uri

    is_https = _is_request_https(request)
    response = RedirectResponse(auth_url, status_code=302)
    response.set_cookie(
        SESSION_COOKIE,
        pending.session_id,
        httponly=True,
        samesite="lax",
        secure=is_https,
        max_age=86400 * 7,
    )
    return response


@router.get("/callback")
def callback(request: Request, code: str = "", state: str = ""):
    """Handles OAuth callback and verifies user credentials."""
    session = session_store.get(_session_id_from_request(request))
    if session is None and state:
        session = session_store.get_by_state(state)
    if session is None:
        raise HTTPException(status_code=400, detail="Missing login session")
    if not session.oauth_state or not state or state != session.oauth_state:
        raise HTTPException(status_code=400, detail="OAuth state mismatch or expired")

    redirect_uri = getattr(session, "redirect_uri", None) or _resolve_redirect_uri(request)

    try:
        creds = exchange_code_for_credentials(
            OAUTH_CLIENT_ID, OAUTH_CLIENT_SECRET, redirect_uri, code
        )
        _, drive, gc = init_clients_from_credentials(
            creds, client_id=OAUTH_CLIENT_ID, client_secret=OAUTH_CLIENT_SECRET
        )
        user_email = ""
        display_name = ""

        # 1. Primary: fetch user info directly via Google OAuth2 userinfo API
        if hasattr(creds, "token") and creds.token:
            try:
                info_resp = requests.get(
                    "https://www.googleapis.com/oauth2/v2/userinfo",
                    headers={"Authorization": f"Bearer {creds.token}"},
                    timeout=5,
                )
                if info_resp.status_code == 200:
                    uinfo = info_resp.json()
                    user_email = (uinfo.get("email") or "").strip()
                    display_name = (uinfo.get("name") or "").strip()
            except Exception as token_err:
                logger.warning(f"Could not fetch userinfo via bearer token: {token_err}")

        # 2. Fallback to drive.GetAbout() if drive is available
        if not user_email and drive is not None and hasattr(drive, "GetAbout"):
            try:
                about = drive.GetAbout()
                user_obj = about.get("user") or {}
                user_email = (user_obj.get("emailAddress") or "").strip()
                display_name = (user_obj.get("displayName") or "").strip()
            except Exception as drive_err:
                logger.warning(f"Could not fetch userinfo via drive.GetAbout(): {drive_err}")
    except Exception as e:
        logger.error(f"Failed to exchange OAuth code: {e}")
        session_store.delete(session.session_id)
        return HTMLResponse(
            f"<h2>Authentication failed</h2><p>{str(e)}</p><p><a href='/'>Back to login</a></p>",
            status_code=400,
        )

    user_info = get_user_info(user_email)
    is_authorized = user_info.get("is_authorized") or user_email.lower().endswith("@skillcatapp.com")
    if not is_authorized:
        session_store.delete(session.session_id)
        return HTMLResponse(
            f"<h2>Access denied</h2><p><b>{user_email}</b> is not authorized to access Moodle Review Studio.</p><p><a href='/'>Try another account</a></p>",
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

    is_https = _is_request_https(request)
    response = RedirectResponse("/review", status_code=302)
    response.set_cookie(
        SESSION_COOKIE,
        session.session_id,
        httponly=True,
        samesite="lax",
        secure=is_https,
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

    response = RedirectResponse("/", status_code=302)
    response.delete_cookie(SESSION_COOKIE)
    return response
