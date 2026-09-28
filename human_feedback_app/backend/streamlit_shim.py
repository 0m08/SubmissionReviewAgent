"""Inject Streamlit session state so existing agent code can run headless."""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from typing import Any, Dict, Optional

import streamlit as st

logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(
    logging.ERROR
)

try:
    from streamlit.runtime.scriptrunner import get_script_run_ctx
except ImportError:
    get_script_run_ctx = None


def _script_run_ctx():
    """Return ScriptRunContext without logging the headless-mode warning."""
    if get_script_run_ctx is None:
        return None
    try:
        return get_script_run_ctx(suppress_warning=True)
    except TypeError:
        return get_script_run_ctx()


class ThreadLocalSessionState:
    def __init__(self, original_state):
        self._local = threading.local()
        self._original_state = original_state

    def _get_state(self) -> Any:
        if _script_run_ctx() is not None:
            return self._original_state
        
        if not hasattr(self._local, "state"):
            self._local.state = {}
        return self._local.state

    def __getitem__(self, key: str) -> Any:
        state = self._get_state()
        if state is self._original_state:
            return self._original_state[key]
        if key not in state:
            raise KeyError(key)
        return state[key]

    def __setitem__(self, key: str, value: Any) -> None:
        state = self._get_state()
        if state is self._original_state:
            self._original_state[key] = value
        else:
            state[key] = value

    def __delitem__(self, key: str) -> None:
        state = self._get_state()
        if state is self._original_state:
            del self._original_state[key]
        else:
            del state[key]

    def __contains__(self, key: str) -> bool:
        state = self._get_state()
        if state is self._original_state:
            return key in self._original_state
        return key in state

    def __getattr__(self, name: str) -> Any:
        if name in ("_local", "_original_state", "_get_state", "_keys"):
            return object.__getattribute__(self, name)
        state = self._get_state()
        if state is self._original_state:
            return getattr(self._original_state, name)
        if name not in state:
            raise AttributeError(f"'ThreadLocalSessionState' object has no attribute '{name}'")
        return state[name]

    def __setattr__(self, name: str, value: Any) -> None:
        if name in ("_local", "_original_state"):
            object.__setattr__(self, name, value)
            return
        state = self._get_state()
        if state is self._original_state:
            setattr(self._original_state, name, value)
        else:
            state[name] = value

    def __delattr__(self, name: str) -> None:
        if name in ("_local", "_original_state"):
            object.__delattr__(self, name)
            return
        state = self._get_state()
        if state is self._original_state:
            delattr(self._original_state, name)
        else:
            if name in state:
                del state[name]
            else:
                raise AttributeError(f"'ThreadLocalSessionState' object has no attribute '{name}'")

    def __iter__(self):
        state = self._get_state()
        if state is self._original_state:
            return iter(self._original_state)
        return iter(state)

    def __len__(self) -> int:
        state = self._get_state()
        if state is self._original_state:
            return len(self._original_state)
        return len(state)

    def _keys(self) -> list[str]:
        state = self._get_state()
        if state is self._original_state:
            return getattr(self._original_state, "_keys", lambda: [])()
        return list(state.keys())

    def get(self, key: str, default: Any = None) -> Any:
        state = self._get_state()
        if state is self._original_state:
            return self._original_state.get(key, default)
        return state.get(key, default)

    def setdefault(self, key: str, default: Any = None) -> Any:
        state = self._get_state()
        if state is self._original_state:
            return self._original_state.setdefault(key, default)
        return state.setdefault(key, default)

    def keys(self):
        state = self._get_state()
        if state is self._original_state:
            return self._original_state.keys()
        return state.keys()

    def values(self):
        state = self._get_state()
        if state is self._original_state:
            return self._original_state.values()
        return state.values()

    def items(self):
        state = self._get_state()
        if state is self._original_state:
            return self._original_state.items()
        return state.items()

    def update(self, other: Dict[str, Any]) -> None:
        state = self._get_state()
        if state is self._original_state:
            self._original_state.update(other)
        else:
            state.update(other)

    def clear(self) -> None:
        state = self._get_state()
        if state is self._original_state:
            self._original_state.clear()
        else:
            state.clear()

    def pop(self, key: str, default: Any = None) -> Any:
        state = self._get_state()
        if state is self._original_state:
            return self._original_state.pop(key, default)
        return state.pop(key, default)


# Apply the patch globally upon importing this module
if not isinstance(st.session_state, ThreadLocalSessionState):
    st.session_state = ThreadLocalSessionState(st.session_state)


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
