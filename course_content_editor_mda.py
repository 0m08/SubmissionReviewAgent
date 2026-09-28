"""Course Content Editor (MDA) — chat front end for the Managed Deep Agent.

Same page as `course_content_editor.py`, driven by a different engine. That one
runs an Anthropic Managed Agents session and reconstructs diffs from a Files
API; this one talks to a LangGraph Agent Server, and the diffs are computed
inside the deployment by the same `_presentation.py` the other builds use.

Everything a user can see is deliberately the same: the palette and type from
`services/cce_diff.py`, the woven turn (agent prose and tool activity in
the order they happened, subagents in their own panels), the Chat / Review
tabs, and the chat box pinned to the bottom of the page. The backend is an
implementation detail and should not be legible as a change of product.

## The content never passes through the model

`present.py` prints a structured payload; a middleware in the deployment routes
it to this page on LangGraph's **custom stream** and hands the agent counts and
an acknowledgement instead. Custom-stream data never enters message state, so
what you see below came off the workspace files and was not re-typed by a model
on its way here — the same guarantee the Managed Agents build gets from a
client-side `present_files` tool.

## Why the thread id lives in the URL

Runs execute on the Agent Server. If this session times out or the tab closes
mid-edit, the agent keeps working and the thread keeps its workspace, history
and files — verified in `agents/course_content_editor_mda/disconnecttest.py`.
What is easy to lose is the *pointer*: a thread id held only in session state
dies with the session, and intact work you cannot address looks exactly like
lost work. So it goes in the query string, where a refresh, a reconnect or a
pasted link all recover the same conversation.

## No approval gate

The deployed agent has no `interrupt_on`. The rule that a commit needs your
explicit words lives in its instructions. Both commit scripts write to a **new**
tab and never modify a source tab, so the worst case is a spare tab to delete —
but do not read the absence of a confirm dialog as the absence of a write.
"""

from __future__ import annotations

import html
import json
import os
import re
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st
from dotenv import dotenv_values

from services.activity_tracking_service import track_tool_action
from services.agent_identity import editor_id, run_context, run_metadata
from services.cce_diff import (
    PAGE_CSS,
    counts_line,
    render_comment_surface,
    rendered_cards,
)

TOOL_NAME = "Course Content Editor (MDA)"
ASSISTANT = "course-content-editor-mda"
#: `course-content-editor-mda-free`. The deployment it replaced was created
#: before LangSmith's free Development tier existed and was billed for its
#: uptime; a tier cannot be changed in place, so the fix was a new deployment.
#: Overridable with `CCE_MDA_URL`, which is how you point at the old one.
#:
#: Memory and threads do not migrate. Memory is `scope="agent"`, so the tree
#: was re-seeded by hand here; sessions from the old deployment stay there.
DEFAULT_URL = (
    "https://course-content-editor-mda-f-21b23d6376b85b3986b2ad89ac86b7de.us.langgraph.app"
)
MDA_PROJECT = Path(__file__).resolve().parent / "agents" / "course_content_editor_mda"

# Commands worth a plain-language line in the turn. Anything else is shown
# verbatim, which is the honest default: a command nobody wrote a label for is
# exactly the one worth reading.
FRIENDLY = {
    "list_tabs.py": "Reading the sheet's tab names",
    "prepare_workspace.py": "Loading slide chunks and research notes",
    "prepare_context_workspace.py": "Loading research notes",
    "present.py --workspace": "Reading the files back",
    "--measure": "Counting blocks and words",
    "commit_context.py": "Writing research notes to a new tab",
    "commit_workspace.py": "Writing slide chunks to a new tab",
    "check_auth.py": "Checking sheet access",
}


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------


def _api_key() -> str | None:
    """LangSmith key from the app env, falling back to the MDA project's .env.

    Parsed with `dotenv_values` rather than by splitting on "=", because the
    value in that file is quoted and a hand-rolled reader kept the quotes —
    which the API rejects as `PermissionDeniedError: API key is forbidden`, an
    error that reads like a permissions problem and is two stray characters.
    """
    key = os.environ.get("LANGSMITH_API_KEY")
    if key:
        return key.strip()
    env_file = MDA_PROJECT / ".env"
    if env_file.exists():
        value = dotenv_values(env_file).get("LANGSMITH_API_KEY")
        if value:
            return value.strip()
    return None


@st.cache_resource(show_spinner=False)
def _client(url: str, key: str):
    from langgraph_sdk import get_sync_client  # noqa: PLC0415

    return get_sync_client(url=url, api_key=key)


#: A sandbox is deleted 14 days after it stops (``delete_after_stop_seconds``
#: = 1209600, read off the live sandboxes). Its thread outlives it, so a
#: conversation older than this comes back with its messages and an empty
#: workspace — which reads as the agent having lost the files. Warn before that
#: happens rather than after.
SANDBOX_LIFE_DAYS = 14
SANDBOX_WARN_DAYS = 12

#: Models a session can run on, as provider:model -> label. The keys must match
#: `_ALLOWED` in the agent's `middleware/model_select.py`, which refuses any
#: other. The first is the default and should be the deployment's CCE_MODEL.
MODELS = {
    "google_genai:gemini-3.8-flash": "Gemini 3.8 Flash",
    "openai:gpt-6-sol": "GPT-6 Sol",
    "anthropic:claude-sonnet-5": "Claude Sonnet 5",
}
MODEL_PICK = "cce_mda_model"

#: Placeholder for a thread with no human message in it.
EMPTY_THREAD = "(no messages yet)"


def _current_editor() -> str | None:
    return editor_id(st.session_state.get("user_email"))


def _new_thread(client, model: str) -> str:
    """Create a thread tagged with who it belongs to.

    The tag is what makes the session picker possible: `threads.search` filters
    on metadata, and without it the only listing available is every thread in
    the deployment — one editor's work shown to another. Threads created before
    this carry no tag and will not be listed; they are still reachable by id.

    The model is tagged too, which is what fixes it for the thread's life: a
    reopened session reads it back rather than taking whatever the picker says.
    """
    meta = {"surface": "course-content-editor-mda", "model": model}
    who = _current_editor()
    if who:
        meta["editor_id"] = who
    thread_id = client.threads.create(metadata=meta)["thread_id"]
    st.session_state.setdefault("cce_mda_thread_models", {})[thread_id] = model
    st.query_params["thread"] = thread_id
    return thread_id


def _thread_model(client, thread_id: str) -> str | None:
    """The model a thread was started on, or None for threads that predate the
    picker (they run on the deployment's default).

    Cached per browser session, so it costs one call per thread opened rather
    than one per rerun.
    """
    known = st.session_state.setdefault("cce_mda_thread_models", {})
    if thread_id not in known:
        try:
            meta = client.threads.get(thread_id).get("metadata") or {}
        except Exception:
            return None  # not cached: the next rerun tries again
        known[thread_id] = meta.get("model")
    return known[thread_id]


def _touch_thread(client, thread_id: str) -> None:
    """Record that this session was just used, for ordering the past list.

    `updated_at` is not a usable substitute. It moves on *any* write, so the
    metadata backfill set thirteen threads to the same instant, and it is what
    the ordering used to fall back to. This stamp moves only when someone
    actually takes a turn, which is what "last used" means to a reader.

    Metadata writes merge rather than replace — verified against the
    deployment — so `editor_id`, `surface` and the platform's own `owner` all
    survive this. Best-effort: a session that ran is not worth failing over a
    bookkeeping write.
    """
    try:
        client.threads.update(
            thread_id,
            metadata={"last_active": datetime.now(timezone.utc).isoformat()},
        )
    except Exception:  # noqa: BLE001
        pass


def _recent_threads(client, fetch: int = 100) -> list[dict]:
    """This editor's threads, newest first. Unfiltered and uncapped.

    Deliberately returns everything it fetched rather than a page. The caller
    discards threads nobody ever spoke in, and capping before that discards the
    wrong ones: a page load with no thread in the URL creates an empty thread,
    those are the newest rows, so a cap of eight returned eight empties and the
    list rendered four real sessions out of a dozen.

    Returns [] when nobody is signed in rather than falling back to an
    unfiltered search: a listing that cannot be scoped to one person should not
    be shown at all.
    """
    who = _current_editor()
    if not who:
        return []
    try:
        rows = client.threads.search(
            metadata={"surface": "course-content-editor-mda", "editor_id": who},
            sort_by="updated_at",
            sort_order="desc",
            limit=fetch,
        )
    except Exception:  # noqa: BLE001 — a picker that errors is worse than none
        return []
    # Sorted again here, on the timestamp this app trusts. `updated_at` moves
    # whenever anything writes to the thread — including a metadata-only write,
    # which is not activity the person would recognise. `_last_active` prefers
    # a stamp we set deliberately and falls back to `updated_at`.
    rows.sort(key=lambda t: _last_active(t) or "", reverse=True)
    return rows


def _last_active(thread: dict) -> str | None:
    """When this thread last did work, newest-first sortable.

    Metadata wins over `updated_at` because the latter is not a record of
    activity: backfilling a tag onto thirteen old threads set all thirteen to
    the same instant and made the picker list 2026-09-11 test threads above
    yesterday's real work.
    """
    stamped = (thread.get("metadata") or {}).get("last_active")
    return str(stamped) if stamped else (
        str(thread["updated_at"]) if thread.get("updated_at") else None)


def _age_days(stamp: str | None) -> float | None:
    """Days since an ISO timestamp, or None when it cannot be read."""
    if not stamp:
        return None
    try:
        when = datetime.fromisoformat(str(stamp).replace("Z", "+00:00"))
    except ValueError:
        return None
    return (datetime.now(timezone.utc) - when).total_seconds() / 86400


def _when(age: float | None) -> str:
    """Age as a short phrase, with the expiry said plainly when it is close.

    A workspace is reclaimed 14 days after its last turn, and by then the
    conversation still comes back while the files do not. Saying so on the row
    is the only warning anyone gets.
    """
    if age is None:
        return "Just now"
    if age < 1:
        hours = max(round(age * 24), 1)
        stamp = f"{hours} hour{'s' if hours != 1 else ''} ago"
    else:
        days = round(age)
        stamp = f"{days} day{'s' if days != 1 else ''} ago"
    if age >= SANDBOX_WARN_DAYS:
        left = max(round(SANDBOX_LIFE_DAYS - age), 0)
        stamp += " · files expire today" if left == 0 else (
            f" · files expire in {left} day{'s' if left != 1 else ''}")
    return stamp


def _thread_summary(client, thread: dict) -> str:
    """First thing the person said in that thread, for the picker row.

    Prefers the messages `search` already returned; only falls back to fetching
    state when they are absent, so a picker of eight rows is normally one API
    call rather than nine.
    """
    messages = (thread.get("values") or {}).get("messages") or []
    for message in messages:
        if message.get("type") == "human":
            text = _text(message).strip()
            if text:
                return text
    for turn in _server_turns(client, thread["thread_id"]):
        text = (turn.get("user") or "").strip()
        if text:
            return text
    return EMPTY_THREAD


# ---------------------------------------------------------------------------
# Reading a thread back
# ---------------------------------------------------------------------------


def _text(message) -> str:
    content = message.get("content") if isinstance(message, dict) else getattr(message, "content", "")
    if isinstance(content, list):
        return "".join(b.get("text", "") for b in content if isinstance(b, dict))
    return str(content or "")


def _ack_presentations(body: str, call_id: str = "") -> list[dict]:
    """Presentation entries rebuilt from a `present.py` acknowledgement.

    The ack is what the middleware leaves in message state: which files were
    shown and the counts it measured, deliberately without the file content —
    that went to the browser on the custom stream and never entered the model's
    context. So a rebuilt card carries every number and no prose, and
    `render_presentation` offers to fetch the current content.
    """
    if '"status": "presented"' not in body and '"status":"presented"' not in body:
        return []
    try:
        ack = json.loads(body)
    except (ValueError, TypeError):
        return []
    out = []
    for index, shown in enumerate(ack.get("shown") or []):
        path = shown.get("path")
        if not path:
            continue
        counts = {k: shown[k] for k in ("blocks", "words", "slide_types", "per_block")
                  if k in shown}
        # Keyed off the acknowledgement's own tool_call_id, for the same reason
        # the live path keeps a uid on the entry: position is not unique. Two
        # turns each presenting one file both sit at index 0, and Streamlit
        # rejects the second widget with that key. A call id is unique within
        # the thread and stable across reruns, so widget state survives.
        out.append({"path": path, "kind": "topics" if "/topics/" in f"/{path}" else "context",
                    "counts": counts, "before": None, "after": None, "restored": True,
                    "uid": f"r{call_id or 'x'}_{index}"})
    return out


def _server_turns(client, thread_id: str) -> list[dict]:
    """Prior turns rebuilt from the thread's own message history.

    The thread is the source of truth: a browser reconnecting after a timeout
    has no local transcript, and inventing one from session state would show a
    different conversation from the one the agent is continuing.

    Rebuilt in the same woven shape a live turn has, because the thread keeps
    more than it looked like it did: 46 of the 109 messages in one real thread
    carry `tool_calls` with their names and arguments, and 51 are the results.
    Only the presented file *content* is genuinely absent, by design.
    """
    try:
        state = client.threads.get_state(thread_id)
    except Exception:  # noqa: BLE001 — brand-new or unreachable thread
        return []

    turns: list[dict] = []
    task_of: dict[str, dict] = {}  # tool_call_id -> the subagent record it answers
    for message in (state.get("values") or {}).get("messages") or []:
        kind = message.get("type")
        body = _text(message).strip()

        if kind == "human":
            if body:
                turns.append({"user": body, "timeline": [], "presentations": [],
                              "threads": {}, "reply": "", "error": None})
            continue
        if not turns:
            continue  # tool chatter before the first human message
        turn = turns[-1]

        if kind == "ai":
            if body:
                turn["timeline"].append({"kind": "text", "text": body})
                turn["reply"] = body
            for call in message.get("tool_calls") or []:
                if call.get("name") == "task":
                    label, brief = _brief_of(call)
                    call_id = str(call.get("id"))
                    rec = {"id": call_id, "file": label, "agent": "editor",
                           "status": "done", "task": brief, "report": "", "lines": []}
                    turn["threads"][call_id] = rec
                    task_of[call_id] = rec
                    turn["timeline"].append({"kind": "thread", "id": call_id})
                else:
                    turn["timeline"].append({"kind": "tool", "label": _label_for(call)})

        elif kind == "tool":
            rec = task_of.get(str(message.get("tool_call_id") or ""))
            if rec is not None:
                # A subagent's report. Its own tool calls ran in a separate
                # graph and are not in this thread's messages, so the panel
                # shows the brief and the report without the step list.
                rec["report"] = body
                continue
            for entry in _ack_presentations(body, str(message.get("tool_call_id") or "")):
                turn["timeline"].append(
                    {"kind": "presentation", "index": len(turn["presentations"])})
                turn["presentations"].append(entry)
    return turns


def _label_for(call: dict) -> str:
    """One line of tool activity, in `course_content_editor.py`'s vocabulary.

    Same verbs (`running:` / `reading` / `editing`) and the same 90-character
    truncation, so a command reads identically on both pages. A command with no
    friendly label is shown verbatim, which is the honest default: the one
    nobody wrote a label for is exactly the one worth reading.
    """
    name = call.get("name")
    args = call.get("args") or {}
    if name == "execute":
        command = str(args.get("command") or "")
        for needle, label in FRIENDLY.items():
            if all(part in command for part in needle.split()):
                return label
        return f"running: {command[:90]}{'…' if len(command) > 90 else ''}"
    if name == "read_file":
        return f"reading {Path(str(args.get('file_path', ''))).name}"
    if name in ("edit_file", "write_file"):
        return f"editing {Path(str(args.get('file_path', ''))).name}"
    return f"using {name}"


# ---------------------------------------------------------------------------
# Per-thread state
#
# Keyed by thread id rather than held flat in session state, so switching
# threads (or starting a new one) shows that thread's history instead of the
# previous conversation's leftovers.
# ---------------------------------------------------------------------------


def _presentations() -> dict[str, dict]:
    """Latest payload per file path, for this thread.

    Keyed by path and overwritten, so the Review tab always shows the current
    state of a file rather than a pile of every intermediate version.
    """
    thread = st.query_params.get("thread") or "none"
    store = st.session_state.setdefault("cce_mda_presentations", {})
    return store.setdefault(thread, {})


def _transcript(client, thread_id: str | None) -> list[dict]:
    """This thread's turns, seeded once from the server then appended to live.

    `None` means the session has not started — the thread is created by the
    first message, not by opening the page — so there is nothing to seed from
    and nowhere to cache it against.
    """
    if not thread_id:
        return []
    store = st.session_state.setdefault("cce_mda_transcript", {})
    if thread_id not in store:
        store[thread_id] = _server_turns(client, thread_id)
    return store[thread_id]


def _resync(client, thread_id: str) -> None:
    """Re-read the thread from the server, discarding the local transcript.

    For when this browser missed part of a turn — a timeout, a dropped
    connection, a run that finished while the tab was closed. The woven detail
    of those turns is genuinely gone (it was never stored server-side), but the
    conversation itself is intact and that is what needs recovering.
    """
    st.session_state.setdefault("cce_mda_transcript", {})[thread_id] = _server_turns(
        client, thread_id)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def set_progress(slot, text: str) -> None:
    """Spinner + the current activity, below the turn while it runs.

    Transient by contract: this never enters the transcript. The spinner is the
    point — a delegated sweep runs a minute or more with nothing to print, and
    a motionless line reads as a frozen page.
    """
    slot.markdown(
        f'<div class="cce-working"><span class="cce-spin"></span>{html.escape(text)}</div>',
        unsafe_allow_html=True,
    )


def render_command(label: str) -> None:
    st.markdown(f'<div class="cce-cmd">{html.escape(label)}</div>', unsafe_allow_html=True)


def render_thread(rec: dict) -> None:
    """One editor subagent: its brief, what it did, and its report.

    The activity list is why this panel exists rather than a single status
    line. A delegated topic edit is the longest-running thing on the page and,
    without its own tool calls showing, a working editor and a hung one look
    identical from out here.
    """
    name = rec.get("file") or rec.get("agent") or "editor"
    status = rec.get("status") or "working"
    icon = {"done": "✓", "working": "⋯"}.get(status, "⋯")
    lines = rec.get("lines") or []
    tail = f" · {len(lines)} step{'s' if len(lines) != 1 else ''}" if lines else ""
    with st.expander(f"{icon} {name} · {rec.get('agent', 'editor')} · {status}{tail}",
                     expanded=False):
        if rec.get("task"):
            st.markdown('<div class="cce-thread-meta">Brief it was given</div>',
                        unsafe_allow_html=True)
            render_command(rec["task"][:1200])
        if lines:
            st.markdown('<div class="cce-thread-meta">What it did</div>',
                        unsafe_allow_html=True)
            for line in lines:
                render_command(line)
        if rec.get("report"):
            st.markdown('<div class="cce-thread-meta">What it reported back</div>',
                        unsafe_allow_html=True)
            st.markdown(rec["report"])
        elif status != "done":
            st.caption("Still working — its report appears here when it finishes.")


def render_presentation(entry: dict, *, uid: str) -> None:
    """One presented file, in its own expander — the Managed Agents page's view.

    The cards, both views and the select-to-comment surface come from
    `services/cce_diff.py`, so this build shows the same thing that one does.
    It is the same edit to the same file; only the engine that made it differs.

    Diffs are built here from the payload's whole-file `before`/`after` rather
    than from its `blocks` list, because `blocks` is a selection of what
    changed and both views need more than that: the side-by-side needs the
    unchanged paragraphs around an edit as context, and a comment is often
    about a block nobody touched. The counts stay as the deployment measured
    them.
    """
    path = entry.get("path") or "(unknown file)"
    name = path.rsplit("/", 1)[-1]
    kind = entry.get("kind") or "topics"
    before, after = entry.get("before"), entry.get("after")

    if entry.get("restored"):
        # Rebuilt from the acknowledgement in message state. Every number the
        # deployment measured is here; the prose is not, because it went to the
        # browser on the custom stream and never entered message state — the
        # same split that guarantees what you saw came off disk. The workspace
        # outlives the session by 14 days, so the content is one turn away.
        with st.expander(f"{name} — counts only", expanded=False):
            summary = counts_line(entry.get("counts") or {})
            if summary:
                st.markdown(f'<div class="cce-counts">{summary}</div>',
                            unsafe_allow_html=True)
            per_block = (entry.get("counts") or {}).get("per_block") or []
            if per_block:
                st.caption(", ".join(
                    f"{b.get('label', b.get('id'))} · {b.get('words')}w"
                    for b in per_block[:12]))
            st.caption(
                "Content isn't replayed — it goes straight to the screen and is "
                "never stored in the conversation. Fetch it from the workspace:"
            )
            if st.button("Show current content", key=f"repres_{uid}",
                         use_container_width=True):
                st.session_state["cce_mda_pending_feedback"] = (
                    f"Run present.py on {path} so I can see it as it stands now. "
                    "Do not edit anything."
                )
                st.rerun()
        return

    if before is None or after is None:
        # A payload from before present.py shipped whole-file content. Session
        # state survives a hot reload, so these turn up on the very next rerun
        # after a deploy; say so rather than rendering an empty diff.
        with st.expander(f"{name} — diff unavailable", expanded=False):
            st.caption(
                "This was presented by an older build that sent only the changed "
                "blocks. Ask the agent to show the file again."
            )
        return

    # Cached on the file's own bytes — see rendered_cards. Without that this
    # re-diffed every presented file on every rerun, which is what made the
    # page slow down as the conversation grew.
    cards = rendered_cards(before, after, kind)
    n = len(cards)
    header = f"{name} — {n} change{'' if n == 1 else 's'}" if n else f"{name} — no changes"
    with st.expander(header, expanded=False):
        if entry.get("note"):
            st.caption(entry["note"])
        summary = counts_line(entry.get("counts") or {})
        if summary:
            st.markdown(f'<div class="cce-counts">{summary}</div>', unsafe_allow_html=True)
        if not cards:
            st.caption("Nothing differs from the original in this file.")
            return

        # Purely a rendering choice — flipping it re-renders from the same
        # payload and sends nothing to the agent.
        show_diff = st.toggle(
            "Show changes", value=True, key=f"presdiff_{uid}",
            help="On: redline against the original. Off: the content as it reads now.",
        )
        if not show_diff and render_comment_surface(
                entry, before, after, kind, uid,
                pending_key="cce_mda_pending_feedback"):
            return
        for label, meta, extra, body, plain in cards:
            if not show_diff:
                # Keep the same label/meta so the card doesn't jump around when
                # toggled — but strip the redline markup out of the label,
                # which is itself a word diff of the topic name.
                label = re.sub(r"<[^>]+>", "", label)
                body = plain
            with st.container(border=True):
                st.markdown(
                    f'<div class="cce-card-label">{label}</div>'
                    f'<div class="cce-card-meta">{html.escape(meta)}</div>'
                    + (f'<div class="cce-card-lo">{extra}</div>' if extra else "")
                    + f'<div class="cce-diff-text">{body}</div>',
                    unsafe_allow_html=True,
                )


def render_timeline(turn: dict) -> None:
    """The woven view: prose and tool activity in the order they happened.

    Not one status box with every command collected inside it — that detaches
    the work from the sentence explaining it, and the sentence is the part
    worth reading.
    """
    threads = turn.get("threads") or {}
    for item in turn.get("timeline") or []:
        kind = item.get("kind")
        if kind == "text":
            st.markdown(item["text"])
        elif kind == "tool":
            render_command(item["label"])
        elif kind == "thread":
            rec = threads.get(item["id"])
            if rec:
                render_thread(rec)
        elif kind == "presentation":
            shown = turn.get("presentations") or []
            if item["index"] < len(shown):
                entry = shown[item["index"]]
                render_presentation(entry, uid=entry.get("uid") or f"t{item['index']}")


def render_turn(turn: dict) -> None:
    """Replay one completed turn, in whichever shape it was recorded.

    Turns this browser watched carry a `timeline` and render woven. Turns
    rebuilt from the server have only the user's message and the final reply.
    Both paths have to work, because a transcript outlives the connection that
    produced it.
    """
    with st.chat_message("user"):
        st.markdown(turn["user"])
    with st.chat_message("assistant"):
        if turn.get("timeline"):
            render_timeline(turn)
        elif turn.get("reply"):
            st.markdown(turn["reply"])
        if turn.get("error"):
            st.error(f"Request failed: {turn['error']}")


# ---------------------------------------------------------------------------
# One turn
# ---------------------------------------------------------------------------

_TOPIC_FILE = re.compile(r"(?:topics|context)/[\w.\-]+\.md")
_MD_FILE = re.compile(r"[\w.\-]+\.md")


def _brief_of(call: dict) -> tuple[str, str]:
    """(label, brief) for a `task` call — the file it owns, if it names one."""
    args = call.get("args") or {}
    brief = str(args.get("description") or args.get("prompt") or "")
    found = _TOPIC_FILE.search(brief)
    return (found.group(0) if found else "one topic"), brief


def stream_turn(client, thread_id: str, text: str, live, progress,
                model: str | None) -> dict:
    """Run one turn, drawing it into `live` in event order as it happens.

    The live shape is the same shape `render_turn` replays afterwards, which is
    what lets this page skip a rerun when the turn ends — see the note at the
    bottom of the file.

    The signed-in email is read here rather than passed in, because it is read
    twice for two destinations that must not be confused: an opaque id goes to
    the agent as run context, the email goes to the LangSmith trace. See
    `services/agent_identity.py` for why that split exists.
    """
    email = st.session_state.get("user_email")
    timeline: list[dict] = []
    presentations: list[dict] = []
    threads: dict[str, dict] = {}
    thread_slots: dict[str, object] = {}
    store = _presentations()
    seen: set[str] = set()
    reply = ""

    # namespace -> the editor panel its activity belongs to, and a buffer for
    # lines that arrived before that could be worked out. See _attach.
    bound: dict[str, dict] = {}
    pending: dict[str, list[str]] = {}

    def _redraw(rec: dict) -> None:
        slot = thread_slots.get(rec["id"])
        if slot is not None:
            with slot.container():
                render_thread(rec)

    def _attach(namespace: str, line: str) -> None:
        """Route one line of subagent activity to the editor panel it came from.

        Subgraph events are tagged with a checkpoint namespace, not with the
        `task` tool call id that opened the panel, so the two have to be
        correlated. This does it on evidence rather than on ordering: an editor
        is given exactly one file and reads it first, so the first line naming a
        topic/context file identifies which panel this namespace is. A single
        editor working alone is unambiguous anyway.

        Until one of those holds, lines are buffered rather than shown. Guessing
        by dispatch order would attribute one topic's edits to another topic's
        panel, and a confidently mislabelled edit is worse than a late one.
        """
        rec = bound.get(namespace)
        if rec is None:
            working = [r for r in threads.values() if r["status"] == "working"]
            # Compared as basenames: the brief names `topics/topic_02_x.md`
            # while the editor's own line says `reading topic_02_x.md`, and the
            # file is the same file either way.
            named = set(_MD_FILE.findall(line))
            if named:
                rec = next((r for r in working
                            if Path(r["file"]).name in named), None)
            if rec is None and len(working) == 1:
                rec = working[0]
            if rec is None:
                pending.setdefault(namespace, []).append(line)
                return
            bound[namespace] = rec
            rec.setdefault("lines", []).extend(pending.pop(namespace, []))
        rec.setdefault("lines", []).append(line)
        _redraw(rec)
        set_progress(progress, f"{rec['file']}: {line[:90]}")

    stream = client.runs.stream(
        thread_id,
        ASSISTANT,
        input={"messages": [{"role": "user", "content": text}]},
        # Who this run is for. `context` reaches the agent — an opaque editor
        # id and nothing else, because it names a path in memory that every
        # caller of the deployment can read. `metadata` reaches the LangSmith
        # trace, which is workspace-private and is where the email belongs.
        # Both are empty for a session with no signed-in user, and the agent
        # then works from team memory alone.
        # `model` is read by `select_model` in both the coordinator and the
        # editors, so the whole session runs on it.
        context=run_context(email, model),
        metadata=run_metadata(email),
        # "custom" is how present.py content reaches this page without going
        # through the agent's context. Without it the review cards never arrive.
        stream_mode=["updates", "custom"],
        # Without this the editor subagents are opaque: the only events that
        # reach here are the `task` call going out and its report coming back,
        # so a topic edit that takes two minutes shows as two minutes of
        # nothing. With it, each editor's own tool calls stream too, tagged
        # with the subgraph namespace that _attach resolves to a panel.
        stream_subgraphs=True,
        # The run belongs to the server: if this browser goes away mid-turn the
        # work continues and the thread keeps it.
        on_disconnect="continue",
    )

    for part in stream:
        # Subgraph events arrive as "updates|<namespace>"; the parent graph's
        # keep the bare mode name.
        mode, _, namespace = str(part.event).partition("|")

        if mode == "custom" and isinstance(part.data, dict):
            payload = part.data.get("cce_present")
            if not isinstance(payload, dict):
                continue
            set_progress(progress, "Showing you the edited content…")
            for entry in payload.get("files") or []:
                # A uid that survives reruns, because the toggle and the
                # comment iframe are keyed by it and Streamlit rebuilds both on
                # every run. Stored on the entry itself, which lives in the
                # transcript, rather than derived from position — two turns
                # presenting the same file would otherwise collide.
                entry["uid"] = f"p{st.session_state.get('cce_mda_uid_seq', 0)}"
                st.session_state["cce_mda_uid_seq"] = (
                    st.session_state.get("cce_mda_uid_seq", 0) + 1)
                store[entry.get("path", "?")] = entry
                # Index into presentations, so the woven view can place each
                # diff where the agent actually showed it rather than in a pile
                # at the end.
                timeline.append({"kind": "presentation", "index": len(presentations)})
                presentations.append(entry)
                with live:
                    render_presentation(entry, uid=entry["uid"])
            continue

        if mode != "updates" or not isinstance(part.data, dict):
            continue

        if namespace:
            # An editor subagent's own work. It never joins the main timeline:
            # the parent conversation is the agent talking to you, and an
            # editor's internal steps belong inside its panel.
            for node_update in part.data.values():
                if not isinstance(node_update, dict):
                    continue
                for msg in node_update.get("messages") or []:
                    if not isinstance(msg, dict) or msg.get("type") != "ai":
                        continue
                    msg_id = msg.get("id")
                    if msg_id and msg_id in seen:
                        continue
                    if msg_id:
                        seen.add(msg_id)
                    for call in msg.get("tool_calls") or []:
                        _attach(namespace, _label_for(call))
            continue

        for node_update in part.data.values():
            if not isinstance(node_update, dict):
                continue
            for msg in node_update.get("messages") or []:
                if not isinstance(msg, dict):
                    continue
                # The same message can surface under more than one node
                # update; without this its text renders twice in the turn.
                msg_id = msg.get("id")
                if msg_id and msg_id in seen:
                    continue
                if msg_id:
                    seen.add(msg_id)

                if msg.get("type") == "tool":
                    # A subagent's report coming back. Its panel is already on
                    # screen showing "working"; fill it in place rather than
                    # printing the report somewhere else.
                    rec = threads.get(str(msg.get("tool_call_id")))
                    if rec:
                        rec["report"] = _text(msg).strip()
                        rec["status"] = "done"
                        _redraw(rec)
                        set_progress(progress, f"{rec['file']} — done")
                    continue

                if msg.get("type") != "ai":
                    continue

                for call in msg.get("tool_calls") or []:
                    name = call.get("name")
                    if name == "task":
                        label, brief = _brief_of(call)
                        call_id = str(call.get("id"))
                        rec = {"id": call_id, "file": label, "agent": "editor",
                               "status": "working", "task": brief, "report": "",
                               "lines": []}
                        threads[call_id] = rec
                        timeline.append({"kind": "thread", "id": call_id})
                        set_progress(progress, f"Editing {label}…")
                        with live:
                            thread_slots[call_id] = st.empty()
                        _redraw(rec)
                        continue

                    line = _label_for(call)

                    timeline.append({"kind": "tool", "label": line})
                    set_progress(progress, line[:110])
                    with live:
                        render_command(line)

                body = _text(msg).strip()
                if body:
                    reply = body
                    timeline.append({"kind": "text", "text": body})
                    set_progress(progress, "working…")
                    with live:
                        st.markdown(body)

    progress.empty()
    return {"user": text, "timeline": timeline, "presentations": presentations,
            "threads": threads, "reply": reply, "error": None}


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------


SHOW_SESSIONS = "cce_mda_show_sessions"


def _render_session_picker(client) -> None:
    """The past-sessions view. Reached from the toolbar, never automatically.

    Ends the run with `st.stop()` so this view *is* the page while it is open.
    Streamlit renders as it executes, so it can never return having drawn
    something — the caller would carry on in the same run and draw the chat
    underneath whatever this left on screen.

    Only loaded when asked for. Listing threads on every run costs an API round
    trip per rerun, which is what made the toolbar version of this slow.
    """
    st.subheader("Past sessions")

    back, _spacer = st.columns([1, 3])
    with back:
        if st.button("Back to current session", use_container_width=True):
            st.session_state.pop(SHOW_SESSIONS, None)
            st.rerun()

    # Discard the ones nobody spoke in first, then cap. Capping first shows a
    # short list padded with nothing, which is how a dozen sessions rendered
    # as four.
    rows: list[tuple[str, str, float | None]] = []
    for thread in _recent_threads(client):
        age = _age_days(_last_active(thread))
        if age is not None and age >= SANDBOX_LIFE_DAYS:
            continue  # its workspace is gone; the messages alone would mislead
        summary = _thread_summary(client, thread)
        if summary == EMPTY_THREAD:
            continue
        rows.append((thread["thread_id"], summary, age))
        if len(rows) >= 15:
            break

    if not rows:
        st.caption("No earlier sessions yet.")
        st.stop()

    st.caption("Each session keeps its files for 14 days.")
    current = st.query_params.get("thread")
    for tid, summary, age in rows:
        with st.container(border=True):
            body, action = st.columns([6, 1], vertical_alignment="center")
            body.write(summary[:110] + ("…" if len(summary) > 110 else ""))
            body.caption(_when(age) + (" · current" if tid == current else ""))
            if action.button("Open", key=f"open_{tid}", use_container_width=True):
                st.query_params["thread"] = tid
                st.session_state.pop(SHOW_SESSIONS, None)
                for name in ("cce_mda_transcript", "cce_mda_presentations"):
                    st.session_state.pop(name, None)
                st.rerun()

    st.stop()


def _connection_error(url: str, error: Exception) -> None:
    """Explain a failed call to the deployment, distinguishing auth from reach.

    Lives here because it is now raised from the first message rather than from
    page load: with lazy session creation nothing touches the deployment until
    someone types, so this is the first moment a wrong key or URL can show.
    """
    if "forbidden" in str(error).lower() or "denied" in str(error).lower():
        st.error(
            f"The deployment rejected the API key ({error}). The key reached "
            f"{url}, so this is not a connectivity problem. Check that "
            "LANGSMITH_API_KEY belongs to the same LangSmith workspace as "
            "the deployment."
        )
    else:
        st.error(f"Could not reach the deployment at {url}: {error}")
    st.text(traceback.format_exc())


def main() -> None:
    st.markdown(PAGE_CSS, unsafe_allow_html=True)
    st.markdown('<div class="cce-title">Course Content Editor</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="cce-subtitle">Managed Deep Agent build — edits run in a sandbox on '
        "LangGraph, and keep running if this tab closes.</div>",
        unsafe_allow_html=True,
    )

    key = _api_key()
    url = os.environ.get("CCE_MDA_URL", DEFAULT_URL)

    with st.sidebar:
        st.subheader("Deployment")
        url = st.text_input("Agent Server URL", value=url)
        st.caption(
            "The thread id is in this page's URL. Copy the link and the same "
            "conversation comes back — including after a timeout, since the "
            "agent keeps running on the server. Lost the link? Start a new "
            "session and your recent ones are listed to pick from."
        )

    if not key:
        st.error(
            "LANGSMITH_API_KEY is not set. Set it in the environment, or in "
            "`agents/course_content_editor_mda/.env` for local dev."
        )
        return

    client = _client(url, key)
    # May be None: a session is created by its first message, not by opening
    # the page. Visiting used to mint a thread every time, and those empty
    # threads are the newest rows — they crowded real sessions out of the list
    # and accumulated on the deployment forever.
    #
    # Nothing is called on the deployment here any more, so there is no startup
    # probe to catch a bad URL or key. Adding one back would mean an API call on
    # every rerun, which is the cost this page has been trimming. The first real
    # call reports it instead, through `_connection_error`.
    thread_id = st.query_params.get("thread")

    with st.sidebar:
        st.subheader("Model")
        if thread_id:
            # Fixed for the thread's life. Switching providers mid-thread can
            # fail on the previous model's reasoning blocks in the history.
            model = _thread_model(client, thread_id)
            st.markdown(f"**{MODELS.get(model, model) if model else 'Deployment default'}**")
            st.caption("Fixed for this session. Start a new session to use another model.")
        else:
            model = st.selectbox(
                "Model for this session", list(MODELS), format_func=MODELS.get,
                key=MODEL_PICK, label_visibility="collapsed",
            )

    if st.session_state.get(SHOW_SESSIONS):
        _render_session_picker(client)  # ends the run

    transcript = _transcript(client, thread_id)

    _bar_left, _bar_mid, _bar_prev, _bar_right = st.columns([1, 1, 1, 1])
    with _bar_left:
        if transcript and st.button("Resync this session", use_container_width=True):
            _resync(client, thread_id)
            st.rerun()
    with _bar_mid:
        if st.button("Start new session", disabled=not thread_id,
                     use_container_width=True):
            # Drops the pointer rather than making a thread. The next message
            # makes one; until then there is nothing to leave behind.
            st.query_params.pop("thread", None)
            for name in ("cce_mda_transcript", "cce_mda_presentations"):
                st.session_state.pop(name, None)
            st.rerun()
    with _bar_prev:
        if st.button("View past sessions", use_container_width=True):
            # Sets a flag and reruns rather than listing inline. The list costs
            # an API call, and paying for it on every rerun is what made the
            # toolbar version slow; this pays once, when it is asked for.
            st.session_state[SHOW_SESSIONS] = True
            st.rerun()
    with _bar_right:
        st.caption(f"Thread `{thread_id}`" if thread_id else "New session")

    tab_chat, tab_review = st.tabs(["💬 Chat", "📝 Review changes"])

    # -----------------------------------------------------------------------
    # Chat tab — the conversation, woven.
    # -----------------------------------------------------------------------
    with tab_chat:
        if not transcript:
            with st.expander("How to use this", expanded=True):
                st.markdown(
                    "1. Paste a Google Sheet URL and say what to change — e.g. *Here is a "
                    "course sheet: `<url>`. The outline tab is 'Final Outline'. Tighten the "
                    "research notes for topic 2, cut padding but keep every specification "
                    "and number.*\n"
                    "2. The agent loads the course, edits it, and shows you before/after "
                    "cards. **Nothing reaches the sheet at this point.**\n"
                    "3. Ask for more changes as many times as you like.\n"
                    "4. When happy: *commit that to a new tab called X*. It writes a new "
                    "tab and never modifies a source tab.\n\n"
                    "The service account must already have edit access to the sheet."
                )
        for turn in transcript:
            render_turn(turn)

    # -----------------------------------------------------------------------
    # Review tab — every file the agent has shown, newest state per file.
    # -----------------------------------------------------------------------
    with tab_review:
        store = _presentations()
        if not store:
            st.caption(
                "Nothing shown yet. Ask for an edit — every file the agent "
                "presents appears here as a diff."
            )
        else:
            st.caption(
                f"{len(store)} file(s). These are the real bytes off the "
                "workspace, diffed against the copy taken when the course was "
                "opened."
            )
            for path, entry in store.items():
                render_presentation(entry, uid=f"review-{path}")
            st.divider()
            st.caption(
                "To write changes back to the sheet, just ask for it in the chat — e.g. "
                '"write the research notes back" or "commit the slide chunks to a tab '
                'called Reviewed". It always writes to a **new** tab; the source tabs '
                "are never touched."
            )

    # -----------------------------------------------------------------------
    # Chat input — declared here, genuinely after both tab blocks have
    # finished. st.chat_input pins itself to the bottom of the page only when
    # it is a top-level element; declared inside a `with tab_x:` block it
    # renders inline instead, which is what put the box above the
    # conversation. Both tab panels also exist in the DOM at once (Streamlit
    # toggles visibility rather than unmounting), so a declaration site
    # between the two blocks leaves a stray empty slot in whichever tab
    # renders next.
    # -----------------------------------------------------------------------
    prompt = st.chat_input("Paste a course Google Sheet URL, or say what to change…")

    # Comments written on a passage arrive as a queued message rather than
    # being sent from inside the expander they were written in — a turn started
    # there would render into the middle of an earlier turn.
    if not prompt and st.session_state.get("cce_mda_pending_feedback"):
        prompt = st.session_state.pop("cce_mda_pending_feedback")

    if not prompt:
        return

    sheet_link = ""
    for text in [t["user"] for t in transcript] + [prompt]:
        if "https://docs.google.com/spreadsheets" in text:
            sheet_link = text[text.index("https://docs.google.com/spreadsheets"):].split()[0]

    started = time.perf_counter()
    with tab_chat:
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            # The turn renders into `live` in event order — the same shape
            # render_turn replays afterwards, so nothing rearranges on screen
            # when it finishes. `progress` sits below it for transient
            # "still working" text that never enters the transcript.
            live = st.container()
            progress = st.empty()
            set_progress(progress, "Working…")
            try:
                if not thread_id:
                    # First message of a new session. Creating it here rather
                    # than on page load is what keeps unused threads from
                    # existing at all. `_new_thread` also puts the id in the
                    # URL, so a reload or a shared link finds this session.
                    thread_id = _new_thread(client, model)
                turn = stream_turn(client, thread_id, prompt, live, progress, model)
                _touch_thread(client, thread_id)
                if not turn["reply"] and not turn["presentations"]:
                    st.info("The agent finished without a closing message.")
                track_tool_action(
                    TOOL_NAME, "chat_turn", run_mode="agent",
                    duration_seconds=time.perf_counter() - started,
                    course_name="", sheet_link=sheet_link,
                )
            except Exception as e:  # noqa: BLE001
                progress.empty()
                track_tool_action(
                    TOOL_NAME, "chat_turn", run_mode="agent",
                    duration_seconds=time.perf_counter() - started,
                    error_message=str(e)[:500],
                    course_name="", sheet_link=sheet_link,
                )
                _connection_error(url, e)
                st.caption(
                    "The run may still be going on the server — it was started with "
                    "on_disconnect=continue. Press \"Resync this session\" once it has "
                    "had time to finish, or reload: the thread id is in the URL."
                )
                turn = {"user": prompt, "timeline": [], "presentations": [],
                        "threads": {}, "reply": "", "error": str(e)}
        transcript.append(turn)

    # Deliberately no st.rerun(). The turn is already on screen in the exact
    # shape render_turn would replay it in, and it is already in the
    # transcript, so a rerun would re-render the whole conversation to produce
    # a pixel-identical result — visible as the page flickering the moment the
    # agent finishes. The cost is that the Review tab, rendered earlier in this
    # script run, shows its pre-turn state until the next interaction: a
    # one-interaction lag on a secondary panel, against a full re-render every
    # single turn.


main()
