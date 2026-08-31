"""Shared helpers for the working-with-google-sheets skill scripts.

Kept small on purpose — scripts in skills are most reliable when they're
single-purpose and read-only on shared state. Anything that touches more
than one script lives here.
"""

from __future__ import annotations

import base64
import json
import os
import re
import sys
import tempfile
from pathlib import Path

import gspread


# Lookup order for the service account credential, in priority order:
#   1. GDRIVE_SA_JSON env var holding the JSON content as a string. Matches
#      services/drive_service.py:520 — the convention already used elsewhere
#      in this repo.
#   2. GDRIVE_SA_B64 env var holding base64-encoded JSON. Also matches
#      services/drive_service.py:526. Many .env workflows prefer base64
#      because it avoids escaping the embedded newlines in the private key.
#   3. SLIDE_CHUNKS_SA_KEY env var holding a *path* to a JSON file. Useful
#      for local dev where you'd rather not paste a multi-line secret.
#   4. Conventional file paths in the hosted container. When you attach a
#      Files-API resource to a Managed Agents session with Mount Path
#      "/uploads/service_account.json", the actual filesystem location is
#      /mnt/session/uploads/service_account.json — the console-facing path
#      is a logical prefix, the real mount is under /mnt/session/. The
#      other paths are kept as fallbacks in case the mount convention
#      changes or differs across hosting tiers, and /mnt/secrets/ is the
#      manual-mount fallback for local dev.
DEFAULT_SA_FILE_PATHS = [
    "/mnt/session/uploads/service_account.json",
    "/uploads/service_account.json",
    "/mnt/user-data/uploads/service_account.json",
    "/mnt/secrets/service_account.json",
]


def _write_temp_sa(content: bytes) -> str:
    """Write JSON content to a private temp file and return its path.

    gspread.service_account() needs a path, not bytes, so when the credential
    comes in via an env var we materialize it to disk with 0600 permissions.
    """
    fd, tmp_path = tempfile.mkstemp(prefix="sa-", suffix=".json")
    with os.fdopen(fd, "wb") as f:
        f.write(content)
    os.chmod(tmp_path, 0o600)
    return tmp_path


def resolve_sa_path() -> str:
    """Find the service account JSON or exit with a clear, actionable error.

    Returns a *file path* — callers pass it to gspread.service_account(filename=...).
    If only an env-var variant is set, the content is decoded (if needed) and
    written to a private temp file.

    The eprint-and-exit pattern is deliberate: skill scripts are invoked
    via bash by the agent, and stderr is what the agent will read to
    decide how to recover. A raised exception with a stack trace is
    less actionable than a one-line "do X" message.
    """
    # 1. Env var holding raw JSON content.
    sa_json = os.environ.get("GDRIVE_SA_JSON")
    if sa_json:
        try:
            json.loads(sa_json)
        except json.JSONDecodeError as exc:
            eprint(f"ERROR: GDRIVE_SA_JSON is set but is not valid JSON: {exc}")
            sys.exit(2)
        return _write_temp_sa(sa_json.encode("utf-8"))

    # 2. Env var holding base64-encoded JSON.
    sa_b64 = os.environ.get("GDRIVE_SA_B64")
    if sa_b64:
        try:
            decoded = base64.b64decode(sa_b64, validate=True)
            json.loads(decoded)  # sanity check
        except Exception as exc:
            eprint(f"ERROR: GDRIVE_SA_B64 is set but did not decode to valid JSON: {exc}")
            sys.exit(2)
        return _write_temp_sa(decoded)

    # 3. Env var holding a path.
    sa_path_env = os.environ.get("SLIDE_CHUNKS_SA_KEY")
    if sa_path_env and Path(sa_path_env).exists():
        return sa_path_env

    # 4. Conventional file paths.
    for candidate in DEFAULT_SA_FILE_PATHS:
        if Path(candidate).exists():
            return candidate

    eprint(
        "ERROR: service account credentials not found. Provide one of:\n"
        "  - GDRIVE_SA_JSON env var containing the service account JSON content, OR\n"
        "  - GDRIVE_SA_B64 env var containing the base64-encoded JSON, OR\n"
        "  - SLIDE_CHUNKS_SA_KEY env var containing a path to a JSON file, OR\n"
        "  - in the session config, attach a Files-API resource with Mount Path "
        "/uploads/service_account.json (this lands at /mnt/session/uploads/ in the container), OR\n"
        "  - place the file at /mnt/secrets/service_account.json (local dev).\n"
        "The service account must have edit access to the target Google Sheet."
    )
    sys.exit(2)


def open_sheet(sheet_url: str):
    """Open a sheet by URL, with a remediation-focused error on failure."""
    sa_path = resolve_sa_path()
    try:
        client = gspread.service_account(filename=sa_path)
    except Exception as exc:
        eprint(f"ERROR: failed to authenticate with service account at {sa_path}: {exc}")
        sys.exit(2)

    try:
        return client.open_by_url(sheet_url)
    except gspread.exceptions.APIError as exc:
        # 403 here is almost always "sheet not shared with the SA".
        sa_email = _peek_sa_email(sa_path)
        eprint(
            f"ERROR: could not open sheet {sheet_url}. "
            f"The most common cause is that the sheet is not shared with the "
            f"service account ({sa_email}). Share the sheet with that email "
            f"as Editor and retry. Underlying error: {exc}"
        )
        sys.exit(2)
    except Exception as exc:
        eprint(f"ERROR: could not open sheet {sheet_url}: {exc}")
        sys.exit(2)


def _peek_sa_email(sa_path: str) -> str:
    """Best-effort SA email lookup for error messages."""
    import json
    try:
        with open(sa_path) as f:
            return json.load(f).get("client_email", "<unknown>")
    except Exception:
        return "<unknown — could not read service_account.json>"


# Column aliases for the Final Outline tab — Title-cased or lowercase headers,
# matching what's actually been seen on a live course sheet.
OUTLINE_COL_ALIASES = {
    "topic": ("Topic", "topic"),
    "subtopic": ("Subtopic", "subtopic"),
    "learning_objectives": ("Learning Objectives", "learning_objectives"),
    "research_notes": ("research_notes", "Research Notes"),
}


def get_outline_field(row, key: str) -> str:
    """Look up an outline row's field by alias, '' if missing/NaN/blank."""
    import pandas as pd

    for col in OUTLINE_COL_ALIASES[key]:
        if col in row and pd.notna(row[col]) and str(row[col]).strip():
            return str(row[col]).strip()
    return ""


def format_context_block(lo_id: int, topic_name: str, row) -> str:
    """One LO's research-notes block, in the ###LO ID: marker format.

    Shared by prepare_workspace.py and prepare_context_workspace.py so both
    producers of context/*.md emit exactly the format commit_context.py's
    parser expects — kept in one place instead of duplicated to avoid drift.
    See working-with-google-sheets/references/parsing-rules.md for why this
    format replaced the old "\\n\\n---\\n\\n" join.
    """
    subtopic = get_outline_field(row, "subtopic")
    lo = get_outline_field(row, "learning_objectives")
    rn = get_outline_field(row, "research_notes")

    parts = [
        f"###LO ID: {lo_id}",
        "####**Topic:**",
        topic_name,
        "####**Subtopic:**",
        subtopic,
        "####**Learning Objective:**",
        lo,
        "####**Research Notes:**",
    ]
    if rn:
        parts.append(rn)

    return "\n".join(parts)


def build_context_for_topic(topic_name: str, outline_df) -> str:
    """Assemble all LO/research-notes blocks for one topic.

    Returns "Context unavailable." when the outline is empty or has no
    recognizable Topic column.
    """
    if outline_df.empty:
        return "Context unavailable."

    topic_col = next((c for c in OUTLINE_COL_ALIASES["topic"] if c in outline_df.columns), None)
    if topic_col is None:
        return "Context unavailable."

    matches = outline_df[outline_df[topic_col].astype(str).str.strip() == topic_name.strip()]
    if matches.empty:
        return f"No matching context found in Final Outline for topic: {topic_name}"

    blocks = [
        format_context_block(lo_id, topic_name, row)
        for lo_id, (_, row) in enumerate(matches.iterrows())
    ]
    return "\n\n".join(blocks) + "\n"


def slugify(name: str) -> str:
    """Convert a topic name to a stable, filesystem-safe slug.

    Lowercase, alphanumerics + hyphens. The slug is for human readability
    only; the leading topic number in the filename is what's load-bearing.
    """
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", name.strip().lower()).strip("-")
    return slug or "unnamed"


def eprint(*args, **kwargs) -> None:
    """Print to stderr. The agent reads stderr for error remediation."""
    print(*args, file=sys.stderr, **kwargs)
