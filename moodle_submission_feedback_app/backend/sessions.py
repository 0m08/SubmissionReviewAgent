# -*- coding: utf-8 -*-
"""Server-side session store for authenticated mentor sessions."""

from __future__ import annotations

import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class MentorSession:
    session_id: str
    user_email: str
    mentor_name: str = ""
    role: str = ""
    avatar_initials: str = ""
    drive: Any = None
    gc: Any = None
    oauth_credentials: Any = None
    oauth_state: Optional[str] = None
    created_at: float = field(default_factory=time.time)


class SessionStore:
    def __init__(self) -> None:
        self._sessions: Dict[str, MentorSession] = {}
        self._lock = threading.Lock()

    def create(self, user_email: str = "", mentor_name: str = "", role: str = "") -> MentorSession:
        session_id = secrets.token_urlsafe(32)
        initials = ""
        if mentor_name:
            parts = mentor_name.strip().split()
            initials = "".join([p[0].upper() for p in parts[:2]])
        elif user_email:
            initials = user_email[:2].upper()

        session = MentorSession(
            session_id=session_id,
            user_email=user_email,
            mentor_name=mentor_name,
            role=role,
            avatar_initials=initials,
        )
        with self._lock:
            self._sessions[session_id] = session
        return session

    def get(self, session_id: Optional[str]) -> Optional[MentorSession]:
        if not session_id:
            return None
        with self._lock:
            return self._sessions.get(session_id)

    def delete(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)


session_store = SessionStore()
