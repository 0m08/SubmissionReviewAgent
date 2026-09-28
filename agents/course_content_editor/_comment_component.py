"""Declaration site for the select-and-comment custom component.

This exists as its own importable module for one non-obvious reason:
`streamlit.components.v1.declare_component` walks back to its *caller's*
stack frame and asserts that `inspect.getmodule(frame)` is not None, to
derive a module name for the component's registry key.

course_content_editor.py cannot satisfy that. It is a page under
`st.navigation`, which Streamlit runs with `exec(code, module.__dict__)` —
a frame that `inspect.getmodule()` cannot resolve, so the assert fires.
It raises a bare `AssertionError` with no message, which surfaces as an
empty error string and explains nothing:

    File "streamlit/components/v1/component_registry.py", line 36
        assert module is not None
    AssertionError

Declaring it here instead gives `declare_component` an ordinary imported
module to attribute the component to. The page imports the result.

Declared at import time, which is also once per process — the module cache
does the memoizing that st.cache_resource was doing badly (it re-raises a
cached exception with the message stripped, which is what made the original
failure so opaque).
"""

from __future__ import annotations

from pathlib import Path

import streamlit.components.v1 as components

COMPONENT_DIR = Path(__file__).resolve().parent / "components" / "comment_selector"

#: The declared component, or None if its static files are missing. The page
#: reports the reason rather than silently falling back to the static view.
comment_selector = None
load_error: str | None = None

if not (COMPONENT_DIR / "index.html").exists():
    load_error = f"missing {COMPONENT_DIR / 'index.html'}"
else:
    try:
        comment_selector = components.declare_component(
            "cce_comment_selector", path=str(COMPONENT_DIR)
        )
    except Exception as exc:  # noqa: BLE001 — the page degrades, but says why
        load_error = f"{type(exc).__name__}: {str(exc).strip() or '(no message)'}"
