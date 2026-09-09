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


def resolve_sa_path(*, required: bool = True) -> str | None:
    """Find the service account JSON or exit with a clear, actionable error.

    Returns a *file path* — callers pass it to gspread.service_account(filename=...).
    If only an env-var variant is set, the content is decoded (if needed) and
    written to a private temp file.

    The eprint-and-exit pattern is deliberate: skill scripts are invoked
    via bash by the agent, and stderr is what the agent will read to
    decide how to recover. A raised exception with a stack trace is
    less actionable than a one-line "do X" message.

    With ``required=False`` the not-found case returns ``None`` instead of
    exiting, which is how the sandbox path asks "is there a key here?" without
    treating its absence as a failure. See ``open_sheet``.
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

    if not required:
        return None

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


def authorize():
    """A gspread client, however this environment supplies credentials.

    Two worlds, one file. Where a service account key is reachable — the local
    Deep Agents build, the Anthropic Managed Agents container — we use it, and
    behaviour is unchanged.

    Inside a LangSmith sandbox there is deliberately no key: thread sandboxes do
    not inherit deploy secrets, and baking one into the snapshot would put it on
    a disk the model can read, which is exactly what an injected instruction in
    a course sheet would go looking for. Instead the sandbox auth proxy
    intercepts egress and fills in the ``Authorization`` header *outside* the
    box. So the right move when no key is found is to send a bare request, not
    to fail.

    ``AnonymousCredentials`` reports ``valid=True``, never refreshes, and its
    ``before_request`` is a no-op, so ``AuthorizedSession`` adds no header of its
    own and leaves the slot free for the proxy.

    Absence of a key is therefore not an error here. It is only an error if the
    request then comes back unauthorized, which ``open_sheet`` reports with both
    possibilities named, since from inside the box the two are indistinguishable.
    """
    sa_path = resolve_sa_path(required=False)
    if sa_path is not None:
        try:
            return gspread.service_account(filename=sa_path), sa_path
        except Exception as exc:
            eprint(f"ERROR: failed to authenticate with service account at {sa_path}: {exc}")
            sys.exit(2)

    from google.auth.credentials import AnonymousCredentials

    return gspread.authorize(AnonymousCredentials()), None


def open_sheet(sheet_url: str):
    """Open a sheet by URL, with a remediation-focused error on failure."""
    client, sa_path = authorize()

    try:
        return client.open_by_url(sheet_url)
    except gspread.exceptions.APIError as exc:
        # 403 here is almost always "sheet not shared with the SA".
        sa_email = _peek_sa_email(sa_path) if sa_path else "the deployment's service account"
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
    # Only the Research Notes body is normalized. Topic/Subtopic/LO are the
    # write-back identity key and are left byte-exact; commit_context.py
    # normalizes both sides when matching, so either form resolves.
    rn = normalize_punctuation(get_outline_field(row, "research_notes"))

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


# Curly quotes in source content cannot be reliably reproduced by a model
# writing an edit tool's old_string — it emits the straight ASCII equivalent,
# which then matches nothing, making those spans effectively uneditable. The
# workspace is a working copy, so we normalize quotes on the way in. Applied
# symmetrically in commit_context.py's identity matching so a normalized
# workspace block still matches its unnormalized sheet row.
_PUNCT_MAP = {
    "‘": "'", "’": "'",   # single curly quotes
    "‚": "'", "‛": "'",
    "“": '"', "”": '"',   # double curly quotes
    "„": '"', "‟": '"',
    "′": "'", "″": '"',   # prime / double prime
}


def normalize_punctuation(text: str) -> str:
    """Fold curly quote characters to their straight ASCII equivalents.

    Deliberately leaves em/en dashes, ellipses, degree signs and accented
    letters alone — those round-trip through an edit tool without trouble.
    """
    if not text:
        return text
    for src, dst in _PUNCT_MAP.items():
        text = text.replace(src, dst)
    return text


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


# ---------------------------------------------------------------------------
# Baseline snapshots
# ---------------------------------------------------------------------------

BASELINE_DIRNAME = ".baseline"
WORKSPACE_KINDS = ("topics", "context")


OUTLINE_FILENAME = "outline.md"


def build_outline(ordered_topics, outline_df, topics_manifest) -> str:
    """The whole course's learning objectives, LOs only, in one file.

    Why this exists as its own file rather than reusing `context/*.md`: those
    carry each LO *and* its research notes, and the notes are the bulk of the
    bytes. An editor that only needs the scope of a topic — which is every
    editor, every session, before its first edit — should not have to read a
    research-notes corpus to find it, and a delegated editor holding one topic
    file should still be able to see where its topic sits in the course.

    Read-only by construction: it is regenerated on every prepare, nothing
    parses it back, and it is not part of any write-back identity key.
    """
    lines = [
        "# Course Learning Objectives",
        "",
        "Every learning objective in this course, with the topic file whose slides",
        "carry it. Read the objectives for a topic before editing its slides: the",
        "objective is the boundary of what the learner needs, and any parenthetical",
        "beside it is the instructional designer's scope note.",
        "",
        "Generated by the prepare script from the Final Outline tab, and rewritten",
        "on every prepare. Nothing reads it back, so editing it changes nothing;",
        "research notes are deliberately not here (see `context/*.md`).",
        "",
    ]

    by_index = {t["name"]: t for t in topics_manifest}

    topic_col = None
    if not outline_df.empty:
        topic_col = next(
            (c for c in OUTLINE_COL_ALIASES["topic"] if c in outline_df.columns), None
        )

    for topic_name in ordered_topics:
        entry = by_index.get(topic_name, {})
        lines.append(f"## Topic {entry.get('index', 0):02d} — {topic_name}")
        slide_file = entry.get("file")
        if slide_file:
            lines.append(f"Slides: `{slide_file}`")
        lines.append("")

        if topic_col is None:
            lines.append("_No Final Outline tab available; objectives unknown._")
            lines.append("")
            continue

        matches = outline_df[
            outline_df[topic_col].astype(str).str.strip() == topic_name.strip()
        ]
        if matches.empty:
            lines.append("_No matching rows in the Final Outline tab._")
            lines.append("")
            continue

        current_subtopic = None
        for _, row in matches.iterrows():
            subtopic = get_outline_field(row, "subtopic")
            lo = get_outline_field(row, "learning_objectives")
            if not lo:
                continue
            if subtopic != current_subtopic:
                lines.append(f"### {subtopic or '(no subtopic)'}")
                current_subtopic = subtopic
            # The LO cell is reproduced as written, including any parenthetical
            # the ID added under it. That note is the scope instruction and is
            # routinely more specific than the objective sentence itself.
            lines.append(lo)
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def snapshot_baseline(workspace) -> int:
    """Copy the pristine workspace aside as the "before" for later diffs.

    Called by the prepare scripts as their last act. It lives here, and is
    called by them rather than by a caller afterwards, because of what happened
    when it wasn't: the Managed Deep Agents build dropped the middleware that
    used to invoke it, nothing else did, and `.baseline/` silently stopped
    existing. Nothing failed loudly — `present.py` just had no "before", and the
    agent burned fourteen turns reading script source trying to work out why,
    then rebuilt a baseline by re-running prepare into a temp directory. A
    workspace and its pristine copy are one artifact; whatever creates the first
    should create the second.

    Overwrites any previous snapshot on purpose: a fresh prepare *is* a new
    starting point.
    """
    import shutil  # noqa: PLC0415
    from pathlib import Path as _Path  # noqa: PLC0415

    workspace = _Path(workspace)
    baseline = workspace / BASELINE_DIRNAME
    if baseline.exists():
        shutil.rmtree(baseline, ignore_errors=True)
    n = 0
    for kind in WORKSPACE_KINDS:
        directory = workspace / kind
        if not directory.is_dir():
            continue
        for f in sorted(directory.glob("*.md")):
            dest = baseline / kind / f.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dest)
            n += 1
    return n
