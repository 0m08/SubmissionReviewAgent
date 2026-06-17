"""Inject Streamlit session state so existing agent code can run headless."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, Optional

import streamlit as st


@contextmanager
def agent_session_context(session_data: Dict[str, Any]):
    """Temporarily populate ``st.session_state`` for agent / slideshow helpers."""
    backup: Dict[str, Any] = {}
    keys = set(session_data.keys()) | set(getattr(st.session_state, "_keys", lambda: [])())
    for key in list(st.session_state.keys()):
        backup[key] = st.session_state[key]
    try:
        for key, value in session_data.items():
            st.session_state[key] = value
        yield
    finally:
        for key in list(st.session_state.keys()):
            if key not in backup:
                try:
                    del st.session_state[key]
                except Exception:
                    pass
        for key, value in backup.items():
            st.session_state[key] = value


def bind_user_session(
    *,
    drive,
    gc,
    user_email: str,
    role: str,
    sheet_link: str = "",
    root_folder_id: str = "",
) -> Dict[str, Any]:
    return {
        "drive": drive,
        "gc": gc,
        "user_email": user_email,
        "role": role or "anonymous",
        "sheet_link": sheet_link,
        "root_folder_id": root_folder_id,
    }
