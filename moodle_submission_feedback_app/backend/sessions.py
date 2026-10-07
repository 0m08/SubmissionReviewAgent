# -*- coding: utf-8 -*-
"""Server-side session store for authenticated mentor sessions."""

from __future__ import annotations

import json
import os
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
    redirect_uri: Optional[str] = None
    created_at: float = field(default_factory=time.time)


class SessionStore:
    def __init__(self) -> None:
        self._sessions: Dict[str, MentorSession] = {}
        self._lock = threading.Lock()
        self._load_from_disk()

    def _save_to_disk(self) -> None:
        try:
            cache_data = {}
            for sid, s in self._sessions.items():
                cache_data[sid] = {
                    "session_id": s.session_id,
                    "user_email": s.user_email,
                    "mentor_name": s.mentor_name,
                    "role": s.role,
                    "avatar_initials": s.avatar_initials,
                    "oauth_state": s.oauth_state,
                    "redirect_uri": getattr(s, "redirect_uri", None),
                    "created_at": s.created_at,
                }
            with open("/tmp/moodle_review_sessions.json", "w", encoding="utf-8") as f:
                json.dump(cache_data, f)
        except Exception:
            pass

    def _load_from_disk(self) -> None:
        try:
            path = "/tmp/moodle_review_sessions.json"
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    cache_data = json.load(f)
                now = time.time()
                for sid, data in cache_data.items():
                    if now - data.get("created_at", 0) < 86400 * 7:
                        self._sessions[sid] = MentorSession(
                            session_id=data["session_id"],
                            user_email=data.get("user_email", ""),
                            mentor_name=data.get("mentor_name", ""),
                            role=data.get("role", ""),
                            avatar_initials=data.get("avatar_initials", ""),
                            oauth_state=data.get("oauth_state"),
                            redirect_uri=data.get("redirect_uri"),
                            created_at=data.get("created_at", now),
                        )
        except Exception:
            pass

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
            self._save_to_disk()
        return session

    def get(self, session_id: Optional[str]) -> Optional[MentorSession]:
        if not session_id:
            return None
        with self._lock:
            s = self._sessions.get(session_id)
            if not s:
                self._load_from_disk()
                s = self._sessions.get(session_id)
            return s

    def get_by_state(self, state: Optional[str]) -> Optional[MentorSession]:
        if not state:
            return None
        with self._lock:
            for s in self._sessions.values():
                if s.oauth_state and s.oauth_state == state:
                    return s
            self._load_from_disk()
            for s in self._sessions.values():
                if s.oauth_state and s.oauth_state == state:
                    return s
        return None

    def delete(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)
            self._save_to_disk()


session_store = SessionStore()
