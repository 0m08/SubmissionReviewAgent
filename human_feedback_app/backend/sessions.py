"""Server-side session store for OAuth clients and loaded sheet context."""

from __future__ import annotations

import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class UserSession:
    session_id: str
    user_email: str
    role: str
    pages: list
    drive: Any = None
    gc: Any = None
    oauth_credentials: Any = None
    refresh_token: Optional[str] = None
    sheet_link: str = ""
    worksheet_name: str = "Slide Chunks"
    root_folder_id: str = ""
    course_name: str = ""
    current_round: int = 0
    enabled_sources: Optional[list] = None
    drive_video_mode: str = ""
    oauth_state: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    sheet: Any = None
    manifest_sync_status: Dict[str, Any] = field(default_factory=dict)
    manifest_sync_triggered: bool = False
    manifest_repair_cache: Dict[int, str] = field(default_factory=dict)
    # Set on logout so in-flight workers cannot write sheet results after the session ends.
    aborted: bool = False


class SessionStore:
    def __init__(self) -> None:
        self._sessions: Dict[str, UserSession] = {}
        self._lock = threading.Lock()

    def create(self, user_email: str, role: str, pages: list) -> UserSession:
        session_id = secrets.token_urlsafe(32)
        session = UserSession(
            session_id=session_id,
            user_email=user_email,
            role=role,
            pages=pages,
        )
        with self._lock:
            self._sessions[session_id] = session
        return session

    def get(self, session_id: Optional[str]) -> Optional[UserSession]:
        if not session_id:
            return None
        with self._lock:
            return self._sessions.get(session_id)

    def delete(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)


session_store = SessionStore()
