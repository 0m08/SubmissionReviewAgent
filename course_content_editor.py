"""
Course Content Editor — Streamlit front end for the Course Content Editor
Managed Agent (agents/course_content_editor/).

Chat with the agent about a course's slide chunks and research notes; it
edits a workspace inside its own hosted sandbox. This page polls the
session-scoped Files API for the actual, current content of every file the
agent touches — not the tool-call stream — and renders diffs against the
original content read the first time each file was seen. That design choice
(and why event-stream parsing was rejected) is documented in
agents/course_content_editor/README.md.

There's no separate "start a session" step: the session is created lazily on
your first chat message — paste a sheet URL, or just say what you want
changed and the agent will ask for one. This mirrors chat.py's free-form
mode exactly.

The Managed Agents events API has no token-delta event — agent.message
always carries a complete text block, not partial chunks (checked the SDK's
type directly: BetaManagedAgentsAgentMessageEvent has no delta variant). So
"streaming" here means rendering each event the moment it arrives — status
updates, tool activity, and each completed message chunk appear live as the
turn progresses — rather than the old behavior of collecting the whole turn
silently and only showing it after the fact.

Accept is a local checkbox; nothing is sent for it. Reject and "request
changes on this row/block" both work by sending a normal chat turn that
quotes the exact original content back at the agent, addressed by the same
identity fields commit_context.py / commit_workspace.py use. Writing to the
sheet is a final, explicit turn that runs commit_context.py / commit_workspace.py.
"""

from __future__ import annotations

import difflib
import html
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Reuse the exact block parsers the deployed skill scripts use, rather than
# re-implementing the ###LO ID: / ###Block ID: contract a second time here.
# Same sys.path convention agents/slide_chunk_editor/_paths.py established
# for cross-referencing a skill's scripts/ directory from outside the sandbox.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent
_CCE_DIR = _REPO_ROOT / "agents" / "course_content_editor"
_SCRIPTS_DIR = _CCE_DIR / "skills" / "working-with-google-sheets" / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
if str(_CCE_DIR) not in sys.path:
    sys.path.insert(0, str(_CCE_DIR))

import commit_context as cc_context  # noqa: E402  (parse_lo_block, LO_BLOCK_START)
import commit_workspace as cc_topics  # noqa: E402  (parse_block, BLOCK_START)

# Shared with chat.py so the CLI and this page agree on what the agent's
# present_files call means — which file a path resolves to, which blocks a
# request selects. Two independent readings of that would drift.
import _presentation as pres  # noqa: E402
import _comment_component as _cmt  # noqa: E402  (see its docstring: exec'd pages can't declare)

from services.activity_tracking_service import track_tool_action  # noqa: E402

# ---------------------------------------------------------------------------
# The review layer — diff building, card HTML and the select-and-comment
# surface — lives in services/cce_diff.py, because the MDA build's page shows
# the same cards from the same content and had drifted into a second, poorer
# implementation of them. See that module's docstring.
# ---------------------------------------------------------------------------
from services.cce_diff import (  # noqa: E402
    PAGE_CSS,
    card_plain_html,
    compose_feedback_message,
    diff_card_html,
    diff_context_file,
    diff_topic_file,
    flatten_topic_group,
    format_topic_block_for_message,
    paragraph_diff_html,
    render_comment_surface,
    rendered_cards,
    text_diff_html,
    word_diff_html,
    _plain_paragraph_html,
    _split_paragraphs,
)


try:
    from dotenv import load_dotenv

    load_dotenv(_REPO_ROOT / ".env")
except ImportError:
    pass

import anthropic  # noqa: E402

DEPLOY_STATE_PATH = _CCE_DIR / "deploy_state.json"
SA_MOUNT_PATH = "/uploads/service_account.json"
FILES_BETA = ["managed-agents-2026-04-01"]

TOOL_NAME = "Course Content Editor"

# No st.set_page_config here — this page is loaded through streamlit_app.py's
# st.navigation, which already owns page config for the whole app; calling it
# again here raises "set_page_config can only be called once".


# ---------------------------------------------------------------------------
# Page chrome — an editorial "redline" aesthetic: this is a proofreading
# tool, not a chatbot, so it leans on real diff coloring and a serif display
# face instead of a generic chat-app look.
# ---------------------------------------------------------------------------
st.markdown(PAGE_CSS, unsafe_allow_html=True)

st.markdown('<div class="cce-title">Course Content Editor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="cce-subtitle">Paste a course Google Sheet URL to begin, then say what to fix. '
    "Every change shows up as a diff to review before it's written back.</div>",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Deployment state
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _load_deploy_state() -> dict | None:
    if not DEPLOY_STATE_PATH.exists():
        return None
    return json.loads(DEPLOY_STATE_PATH.read_text(encoding="utf-8"))


deploy_state = _load_deploy_state()
if not deploy_state or not deploy_state.get("agent_id"):
    st.error(
        "The Course Content Editor agent hasn't been deployed yet. "
        f"Run `python {DEPLOY_STATE_PATH.parent.relative_to(_REPO_ROOT)}/deploy.py` first."
    )
    st.stop()
if not deploy_state.get("sa_file_id"):
    st.error(
        "No Google service account is configured for this agent (deploy_state.json has no "
        "sa_file_id). Set GDRIVE_SA_B64 and re-run deploy.py."
    )
    st.stop()


def get_client() -> anthropic.Anthropic:
    if "cce_client" not in st.session_state:
        st.session_state.cce_client = anthropic.Anthropic()
    return st.session_state.cce_client


# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
_DEFAULTS = {
    "cce_session_id": None,
    # Each entry is one full turn: {"user": str, "tool_lines": [str], "reply": str,
    # "presentations": [dict], "error": str | None}. Rendered with the exact same
    # st.status shape whether it's the turn currently streaming in or a past turn
    # replayed from history — that's deliberate, see render_turn().
    # Entries here outlive the code that wrote them (session_state survives a hot
    # reload), so every reader must tolerate older shapes — render_presentation
    # documents what that costs when it isn't done.
    "cce_transcript": [],
    "cce_files": {},             # "topics/<name>" | "context/<name>" -> {"before", "after", "file_id"}
    "cce_approvals": {},         # card_id -> "accepted" | "rejected"
    "cce_manifest": None,
    # uid -> list of pending selection comments, handed back to the
    # component on every rerun because the iframe is rebuilt each time.
    "cce_comments": {},
    "cce_comment_seq": {},
    # uid -> the free-text "anything else" note typed alongside the
    # comments. Scope belongs to the user, never to the composer.
    "cce_comment_notes": {},
    # Set by the comment component; consumed where chat turns are run, so
    # the resulting turn lands at the bottom of the conversation instead of
    # inside the expander the button was clicked in.
    "cce_pending_feedback": None,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


def get_user_email() -> str:
    """The logged-in user's email, set by streamlit_app.py's login flow
    before this page ever runs (same key agent_ui_template.py reads).
    Falls back to a fixed string outside a real login (e.g. local dev
    without OAuth) rather than crashing — session ownership just won't be
    meaningfully scoped in that case."""
    return st.session_state.get("user_email") or "unknown-user"


USER_MEMORY_STORE_PREFIX = "cce-user-"


def _slugify_email(email: str) -> str:
    """Memory store names become the mount directory name
    (/mnt/memory/<store-name>/), so keep it filesystem-safe — no @, no dots
    run together, no spaces."""
    return re.sub(r"[^a-zA-Z0-9]+", "-", email.strip().lower()).strip("-")


def get_or_create_user_memory_store(user_email: str) -> str | None:
    """Lazily find-or-create this user's personal memory store.

    Unlike the shared store (created once at deploy time — see deploy.py's
    ensure_shared_memory_store, which avoids a multi-process race on a
    workspace-wide singleton), a per-user store has no such race: only one
    person's browser session ever creates their own store. Memory stores
    have no metadata field to tag with the owner's email (unlike sessions),
    so ownership is encoded in the store's `name` itself and discovered by
    listing + filtering — same shape as find_user_sessions().

    Cached in st.session_state so this costs at most one list() call per
    browser tab, not one per rerun. Returns None (and lets the caller
    proceed without per-user memory) if the store can't be resolved —
    memory is a nice-to-have, not a hard dependency for editing to work.
    """
    cache_key = f"cce_user_memory_store_id::{user_email}"
    if cache_key in st.session_state:
        return st.session_state[cache_key]

    client = get_client()
    store_name = f"{USER_MEMORY_STORE_PREFIX}{_slugify_email(user_email)}"
    store_id = None
    try:
        for store in client.beta.memory_stores.list():
            if store.name == store_name:
                store_id = store.id
                break
        if store_id is None:
            store = client.beta.memory_stores.create(
                name=store_name,
                description=(
                    f"Personal editing preferences for {user_email}: durable, recurring "
                    "preferences this person has expressed about how their slide chunks "
                    "and research notes should be written or reviewed — not one-off "
                    "corrections, and not a course's writing style (that lives in the "
                    "editing-course-content skill's writing-styles reference instead)."
                ),
            )
            store_id = store.id
    except Exception:
        store_id = None  # memory is best-effort; don't block session creation on it
    st.session_state[cache_key] = store_id
    return store_id


def _describe_thread_event(event) -> str | None:
    """Human-readable label for a multiagent delegation event on the
    session-level (primary) stream. Returns None for event types this
    function doesn't cover, so callers can skip them.

    These only appear once the agent's `multiagent` roster includes a
    `self` entry (see managed_agent_config.yaml) and the coordinator
    actually delegates — a session that never delegates simply never emits
    them, so this is purely additive and safe to always check for.
    """
    et = event.type
    name = getattr(event, "agent_name", None) or getattr(event, "to_agent_name", None) \
        or getattr(event, "from_agent_name", None) or "a copy of itself"
    if et == "session.thread_created":
        return f"delegating to {name}…"
    if et == "session.thread_status_terminated":
        return f"{name} finished"
    if et == "agent.thread_message_sent":
        return f"→ task sent to {name}"
    if et == "agent.thread_message_received":
        return f"← report received from {name}"
    return None


def _describe_tool_use(event) -> str:
    name = getattr(event, "name", "?")
    inp = getattr(event, "input", {}) or {}
    if name == "bash":
        cmd = str(inp.get("command", ""))
        return f"running: {cmd[:90]}{'…' if len(cmd) > 90 else ''}"
    if name == "read":
        return f"reading {Path(str(inp.get('file_path', ''))).name}"
    if name == "edit":
        return f"editing {Path(str(inp.get('file_path', ''))).name}"
    return f"using {name}"


def ensure_session() -> str:
    """Create the session on first use. There's no separate "start" step —
    the first chat message doubles as the kickoff, same as chat.py's
    free-form mode (paste a sheet URL, or just say what you want and the
    agent will ask for one).

    Tagged with the current user's email in `metadata` — that's what makes
    a session resumable later: Managed Agents sessions persist server-side
    independent of this Streamlit process, but nothing links one back to a
    person unless we tag it ourselves. See find_user_sessions().
    """
    if st.session_state.cce_session_id:
        return st.session_state.cce_session_id
    client = get_client()
    user_email = get_user_email()
    resources = [
        {"type": "file", "file_id": deploy_state["sa_file_id"], "mount_path": SA_MOUNT_PATH},
    ]
    # Memory stores only attach at session-create time (sessions.resources.add()
    # rejects memory_store) — this is why store resolution has to happen here,
    # before create(), rather than added onto an already-running session. A
    # session created before this shipped will simply run without memory
    # mounted for its lifetime; resuming it later doesn't retroactively add
    # it — only a fresh session picks up whatever stores exist at the time.
    shared_store_id = deploy_state.get("shared_memory_store_id")
    if shared_store_id:
        resources.append({
            "type": "memory_store",
            "memory_store_id": shared_store_id,
            "access": "read_write",
            "instructions": (
                "Team-wide editorial standards — recurring preferences that apply "
                "across all users and courses. See the editing-course-content skill "
                "for what belongs here vs. the per-user store vs. the writing-styles "
                "reference."
            ),
        })
    user_store_id = get_or_create_user_memory_store(user_email)
    if user_store_id:
        resources.append({
            "type": "memory_store",
            "memory_store_id": user_store_id,
            "access": "read_write",
            "instructions": (
                f"Personal editing preferences for {user_email} only — recurring "
                "preferences this specific person has expressed, not team-wide "
                "standards and not a course's writing style."
            ),
        })
    session = client.beta.sessions.create(
        agent=deploy_state["agent_id"],
        environment_id=deploy_state["environment_id"],
        title=f"Course content editor — {user_email}",
        metadata={"user_email": user_email},
        resources=resources,
    )
    st.session_state.cce_session_id = session.id
    track_tool_action(TOOL_NAME, "session_started", run_mode="agent")
    return session.id


@st.cache_data(ttl=30, show_spinner=False)
def find_user_sessions(user_email: str, *, limit: int = 20) -> list:
    """Sessions belonging to this user, most recent first.

    Managed Agents sessions don't support filtering `list()` by metadata
    server-side, so this lists the agent's recent sessions (excluding
    terminated/archived ones — those can't be sent new turns) and filters
    client-side on the `user_email` tag `ensure_session()` sets. `limit`
    bounds how far back we look, not how many results come back.

    Cached briefly (30s) — this gets called on every script rerun to
    populate the "Switch session" popover, and there's no need to hit the
    API that often for a list that only changes when a session is created.
    """
    client = get_client()
    try:
        sessions = client.beta.sessions.list(
            agent_id=deploy_state["agent_id"],
            statuses=["idle", "running", "rescheduling"],
            order="desc",
            limit=limit,
            betas=FILES_BETA,
        )
        return [s for s in sessions if s.metadata.get("user_email") == user_email]
    except Exception:
        return []


def load_session_transcript(session_id: str) -> list[dict]:
    """Reconstruct the turn-dict transcript (see render_turn) from a
    session's actual event history, so a resumed session shows the same
    conversation instead of opening on a blank chat with the work already
    done invisibly in the background."""
    client = get_client()
    try:
        events = list(client.beta.sessions.events.list(
            session_id, order="asc",
            types=["user.message", "agent.message", "agent.tool_use", "agent.custom_tool_use",
                   "session.thread_created", "agent.thread_message_sent",
                   "agent.thread_message_received", "session.thread_status_idle",
                   "session.thread_status_terminated"],
        ))
    except Exception:
        return []

    turns: list[dict] = []
    current: dict | None = None
    for event in events:
        if event.type == "user.message":
            if current is not None:
                turns.append(current)
            user_text = "".join(
                block.text for block in event.content if getattr(block, "type", None) == "text"
            )
            current = {"user": user_text, "tool_lines": [], "reply": "",
                       "presentations": [], "timeline": [], "threads": {},
                       "error": None}
        elif current is None:
            continue  # stray event before any user.message — shouldn't happen, skip defensively
        elif event.type == "agent.message":
            for block in event.content:
                if getattr(block, "type", None) == "text" and block.text.strip():
                    current["reply"] += ("\n\n" if current["reply"] else "") + block.text
                    current["timeline"].append({"kind": "text", "text": block.text})
        elif event.type == "agent.tool_use":
            label = _describe_tool_use(event)
            current["tool_lines"].append(label)
            current["timeline"].append({"kind": "tool", "label": label})
        elif event.type.startswith(("session.thread", "agent.thread")):
            # Same folding the live path uses, so a resumed conversation shows
            # its subagent panels instead of collapsing them into one line.
            if event.type == "session.thread_created":
                tid = getattr(event, "session_thread_id", None)
                if tid and tid not in current["threads"]:
                    current["timeline"].append({"kind": "thread", "id": tid})
            label = track_thread_event(event, current["threads"])
            if label:
                current["tool_lines"].append(label)
        elif event.type == "agent.custom_tool_use":
            # Rebuilt without content: the payload carries the path and an
            # id, and render_presentation resolves before/after from the live
            # file cache (resume_session repopulates it, originals included).
            # So a resumed conversation gets its diffs back — and with them
            # the select-and-comment surface, which is otherwise unreachable
            # after any restart, since session_state does not survive one.
            #
            # The tradeoff, deliberately taken: the diff shown is the file's
            # current state against the session baseline, not a snapshot of
            # what that turn displayed at the time. For reading and commenting
            # on where the file stands now, current is the right answer.
            for _n, _item in enumerate(pres.normalize_items(getattr(event, "input", None))):
                _key = pres.resolve_key(st.session_state.cce_files, _item["path"]) or _item["path"]
                current["tool_lines"].append(f"Presented {_key}")
                current["timeline"].append(
                    {"kind": "presentation", "index": len(current["presentations"])})
                current["presentations"].append({
                    "path": _key,
                    "note": _item.get("note", ""),
                    "uid": f"{event.id}-{_n}",
                    "missing": False,
                })
    if current is not None:
        turns.append(current)
    return turns


_LINE_NUM_PREFIX = re.compile(r"^\s*\d+\t", re.MULTILINE)


def _strip_line_numbers(text: str) -> str:
    """The text-editor 'read' tool returns content with cat -n-style line
    number prefixes ("1\\t...\\n2\\t...") — strip them to get the actual
    file text, same as what Files API downloads (which are raw) contain."""
    return _LINE_NUM_PREFIX.sub("", text)


def _collect_originals(events, originals: dict[str, str]) -> None:
    """Pair each `read` tool_use with its result and keep the EARLIEST
    content seen per file, into `originals` (mutated in place).

    Shared by the session stream and each subagent thread stream, which
    carry the same event shapes. `setdefault` is what makes "earliest wins"
    work across several streams processed in sequence.
    """
    pending_reads: dict[str, str] = {}  # tool_use event id -> file_path
    for event in events:
        if event.type == "agent.tool_use" and getattr(event, "name", None) == "read":
            path = str((event.input or {}).get("file_path", ""))
            if path:
                pending_reads[event.id] = path
        elif event.type == "agent.tool_result" and not getattr(event, "is_error", False):
            path = pending_reads.pop(getattr(event, "tool_use_id", None), None)
            if not path:
                continue
            text_blocks = [b.text for b in (event.content or []) if getattr(b, "type", None) == "text"]
            if not text_blocks:
                continue
            # Strip BEFORE classifying, not after: the read tool prefixes every
            # line with "<n>\t", so the block markers are not at line start
            # until the numbers come off and every file would classify as
            # neither topics nor context.
            content = _strip_line_numbers("".join(text_blocks))
            if cc_topics.BLOCK_START.search(content):
                kind = "topics"
            elif cc_context.LO_BLOCK_START.search(content):
                kind = "context"
            else:
                continue
            originals.setdefault(f"{kind}/{Path(path).name}", content)


def reconstruct_original_content(session_id: str) -> dict[str, str]:
    """Best-effort recovery of each file's ORIGINAL (pre-edit) content, for
    when a session is resumed cold.

    The Files API only ever exposes a file's CURRENT content — confirmed
    earlier there's no version history, one object per path — so
    refresh_files() alone cannot distinguish "original" from "already
    edited" the first time it sees a file after a resume; both come out
    identical, which is why diffs vanish on resume even though real,
    uncommitted edits are still sitting there.

    This replays the session's full event history AND every subagent
    thread's, keeping the EARLIEST successful `read` of each file as its
    original — a real snapshot from before any edit happened, not the
    current file's content, using exactly the identity scheme
    (content-sniffed kind + basename) refresh_files() already uses so
    results line up under the same keys.

    The thread half is not an optimisation: with per-topic delegation the
    coordinator never reads the topic files at all, so the session stream
    alone recovers nothing and every diff comes back empty.

    Only recovers files the agent actually viewed with its native read
    tool before editing. A file edited via bash without ever being read
    first has no possible original from event history — that file's diff
    just won't reappear on resume, same as today, not worse.
    """
    client = get_client()
    try:
        events = list(client.beta.sessions.events.list(
            session_id, order="asc", types=["agent.tool_use", "agent.tool_result"],
        ))
    except Exception:
        return {}

    originals: dict[str, str] = {}      # store_key -> earliest content seen
    _collect_originals(events, originals)

    # The coordinator's own stream is not enough, and this is the whole reason
    # resumed diffs came back empty. Editing is delegated per topic, and a
    # subagent's tool calls never surface on the session stream — every
    # agent.tool_use there carries session_thread_id None. Measured on a real
    # session: the 6 reads visible at session level were skills, the manifest
    # and memory files, and not one topic file, because all three topic files
    # were read and rewritten inside editor threads. So every topic resolved
    # to "no original", before fell back to the current (already-edited)
    # content, and present_files answered 0.0% change on substantive edits.
    #
    # Each thread's events are listable separately, which is where the real
    # pre-edit reads live. Session stream first, threads after, earliest-wins:
    # the coordinator reads before it delegates, so its copy is the older one
    # where both exist.
    try:
        threads = list(client.beta.sessions.threads.list(session_id))
    except Exception:  # noqa: BLE001 — a resume without thread history is still useful
        threads = []
    for thread in threads:
        try:
            thread_events = list(client.beta.sessions.threads.events.list(
                thread_id=thread.id, session_id=session_id, limit=1000,
            ))
        except Exception:  # noqa: BLE001 — one unreadable thread must not sink the rest
            continue
        # No `types` filter on this endpoint (nor `order`), so it comes back
        # whole and gets filtered here instead.
        _collect_originals(thread_events, originals)
    return originals


def _load_files_with_baselines(session_id: str, *, attempts: int, delay: float) -> None:
    """Refresh the file cache and restore each file's true pre-edit original.

    These two always belong together: refresh_files() sets "before" to
    whatever the file holds the first time it sees it, which on a resumed
    session is the already-edited content, so every diff and every measured
    count would come out empty without the reconstruct pass behind it.
    """
    refresh_files(attempts=attempts, delay=delay)
    for store_key, original_content in reconstruct_original_content(session_id).items():
        if store_key in st.session_state.cce_files:
            st.session_state.cce_files[store_key]["before"] = original_content


def answer_pending_custom_tools(session_id: str) -> list[str]:
    """Answer every custom-tool call this session is still parked on.

    A custom tool is client-side by definition: the session parks at
    idle/requires_action and NOTHING moves server-side until a
    user.custom_tool_result goes back. So if the browser tab driving the
    turn disappears between the call and the answer — a dropped websocket,
    a blank page, a refresh, a closed laptop — nobody ever sends it and the
    session stays parked indefinitely.

    Reloading the transcript does not help and is the trap worth naming:
    events keep listing fine, so a resync looks like it worked ("it pulled
    up to the latest event") while no *new* events ever arrive, because the
    turn is still waiting on us. The result is the only thing that unblocks
    it. That is why this runs from every entry point into an existing
    session, not just from the live turn's disconnect handler.

    Scans full history rather than a time window, so it cannot miss a call
    that was parked before whatever slice the caller happens to be looking
    at; already-answered calls are skipped by id.
    """
    client = get_client()
    try:
        events = list(client.beta.sessions.events.list(
            session_id, order="asc",
            types=["agent.custom_tool_use", "user.custom_tool_result"],
        ))
    except Exception:  # noqa: BLE001 — resume must not hard-fail on a list error
        return []

    answered = {
        getattr(e, "custom_tool_use_id", None)
        for e in events if e.type == "user.custom_tool_result"
    }
    lines: list[str] = []
    for event in events:
        if (event.type == "agent.custom_tool_use"
                and event.id not in answered
                and getattr(event, "name", None) == pres.TOOL_NAME):
            handle_present_files(event)
            lines.append("Answered a pending present_files call")
    return lines


def drain_parked_turn(session_id: str, timeout: float = 180.0) -> list[str]:
    """Answer parked tool calls and let the resumed turn run itself out.

    Answering restarts the turn server-side, and the agent may well call
    present_files again before it finishes — so this loops rather than
    answering once: answer, let it work, answer whatever it asks next, stop
    when the session is no longer running and nothing is left parked.

    Costs nothing on a healthy session: the first pass finds no pending call
    and a non-running status, and returns immediately without sleeping.
    """
    client = get_client()
    lines: list[str] = []
    deadline = time.monotonic() + timeout
    while True:
        answered_now = answer_pending_custom_tools(session_id)
        lines.extend(answered_now)
        if not answered_now:
            try:
                status = client.beta.sessions.retrieve(session_id).status
            except Exception:  # noqa: BLE001
                break
            if status not in ("running", "rescheduling"):
                break
        if time.monotonic() > deadline:
            lines.append("Stopped waiting for the resumed turn (still running)")
            break
        time.sleep(2.0)
    return lines


def resume_session(session_id: str) -> None:
    """Point this browser session at an existing Managed Agents session —
    reloads its conversation history and re-syncs the file cache so the
    Review tab's diffs reappear, not just the chat."""
    st.session_state.cce_session_id = session_id
    st.session_state.cce_files = {}
    st.session_state.cce_manifest = None
    st.session_state.cce_approvals = {}
    # Baselines BEFORE draining, and this order is load-bearing. A parked
    # present_files call is answered with counts measured from
    # cce_files[...]["before"], and on a cold resume refresh_files() alone
    # sets "before" to the file's CURRENT (already-edited) content — the very
    # gap reconstruct_original_content exists to close. Draining first would
    # therefore answer the agent with before == after: 0.0% change and equal
    # block counts, which is exactly the fabricated-numbers failure the
    # measured facts were added to prevent, handed over with a straight face.
    _load_files_with_baselines(session_id, attempts=1, delay=0)

    # Now unblock. A session parked on an unanswered call still serves its
    # whole event history, so loading the transcript first would render a
    # complete-looking conversation that then never advances — the turn is
    # waiting on a result nobody sent.
    with st.spinner("Checking for an unfinished turn…"):
        drained = drain_parked_turn(session_id)
    if drained:
        st.toast(f"Resumed a turn that was waiting on this tab ({len(drained)} call(s) answered).")
        # The resumed turn may have edited more files and read new ones, so
        # re-read both halves. "before" survives: refresh_files only sets it
        # the first time a file is seen.
        _load_files_with_baselines(session_id, attempts=2, delay=2.0)

    # Transcript last: rebuilding a presentation resolves the agent's path
    # against cce_files, so loading it against an empty cache would leave
    # every presented file unresolved and its diff (and comment surface) missing.
    st.session_state.cce_transcript = load_session_transcript(session_id)
    track_tool_action(TOOL_NAME, "session_resumed", run_mode="agent")


def _reconcile_after_disconnect(session_id: str, since) -> tuple[list[str], str, bool]:
    """Catch up on whatever happened after our SSE connection to the
    session dropped.

    The event stream connecting Streamlit to the session is not the same
    thing as the turn itself — the agent runs server-side and keeps going
    even if our connection drops mid-stream (confirmed: a turn that hit a
    dropped connection here showed as completed in the platform.claude.com
    console). Session events are independently listable after the fact via
    `sessions.events.list`, so re-fetch everything since the last event we
    actually saw and look for the turn's true end state, instead of just
    declaring failure because our socket happened to close.
    """
    client = get_client()
    kwargs = {"created_at_gt": since} if since is not None else {}
    try:
        events = list(client.beta.sessions.events.list(
            session_id, order="asc",
            types=[
                "agent.message", "agent.tool_use", "session.status_idle", "session.status_terminated",
                "session.thread_created", "session.thread_status_terminated",
                "agent.thread_message_sent", "agent.thread_message_received",
                # Not used for the parked-call check below (that scans full
                # history via answer_pending_custom_tools) — kept so a dropped
                # connection's own events still read consistently here.
                "agent.custom_tool_use", "user.custom_tool_result",
            ],
            **kwargs,
        ))
    except Exception:
        return [], "", False

    extra_tool_lines: list[str] = []
    extra_reply = ""
    completed = False

    # A present_files call that went out while our socket was down left the
    # turn parked at requires_action with nobody to answer it. Answering it
    # now is what actually resumes the turn — without this, "resync" would
    # keep reporting the same unfinished turn forever, because nothing on
    # the server can move until the result lands. Shared with resume_session
    # rather than duplicated: these are the same failure reached two ways,
    # and a fix applied to one copy would silently miss the other.
    extra_tool_lines.extend(answer_pending_custom_tools(session_id))

    for event in events:
        if event.type == "agent.message":
            for block in event.content:
                if getattr(block, "type", None) == "text" and block.text.strip():
                    extra_reply += ("\n\n" if extra_reply else "") + block.text
        elif event.type == "agent.tool_use":
            extra_tool_lines.append(_describe_tool_use(event))
        elif (thread_label := _describe_thread_event(event)) is not None:
            extra_tool_lines.append(thread_label)
        elif event.type == "session.status_idle":
            # An idle carrying requires_action is the opposite of completed —
            # it means the turn stopped *on us*, waiting for a custom tool
            # result we never sent because the connection had dropped.
            # Counting it as completion would report a half-finished turn as
            # done and leave the session parked indefinitely.
            stop = getattr(event, "stop_reason", None)
            if getattr(stop, "type", None) != "requires_action":
                completed = True
        elif event.type == "session.status_terminated":
            completed = True
    return extra_tool_lines, extra_reply, completed


def _run_turn(text: str, live, progress, *, log_action: str) -> dict:
    """Drive one turn, rendering into `live` (a container) as events arrive.

    Elements are appended to `live` in the order the events land, which is
    the same order and the same widgets render_timeline replays afterwards —
    so the turn does not visibly rearrange itself the moment it finishes.
    That was the old shape's real cost: commands went into a status box and
    the prose underneath it, then the rerun re-drew the whole thing woven,
    and the answer moved on screen just as you started reading it.

    `progress` is an st.empty() below the container, used for transient
    "still working" text that should NOT survive into the transcript —
    including the "Syncing changed files…" wait, which is a couple of real
    seconds of Files API polling and would otherwise look like a hang.

    Always returns a plain turn dict; the caller appends it to
    cce_transcript, and render_turn replays it into the identical shape.
    """
    client = get_client()
    session_id = ensure_session()
    tool_lines: list[str] = []
    presentations: list[dict] = []
    # Ordered record of what happened. It drives both the live render below
    # and render_timeline's replay, which is what keeps them identical.
    # tool_lines/reply are still filled in: _reconcile_after_disconnect and
    # load_session_transcript both produce turns in that older shape, and
    # render_turn still has to be able to draw one.
    timeline: list[dict] = []
    threads: dict[str, dict] = {}
    # tid -> an st.empty() placed where the delegation happened, so the
    # panel can be re-rendered in place as that subagent progresses.
    thread_slots: dict[str, object] = {}
    pending_seed = False  # see the deferred seed in the event loop below
    reply = ""
    # Seeded to "now", not left None, before we even send — if the
    # connection drops before a single event of this turn arrives,
    # reconciliation must still start from just-before-this-turn, not from
    # the beginning of the session (which would re-pull every prior turn's
    # content into this one).
    last_event_at = datetime.now(timezone.utc)
    started = time.time()
    try:
        with client.beta.sessions.events.stream(session_id=session_id) as stream:
            client.beta.sessions.events.send(
                session_id=session_id,
                events=[{"type": "user.message", "content": [{"type": "text", "text": text}]}],
            )
            for event in stream:
                et = event.type
                last_event_at = getattr(event, "processed_at", last_event_at)
                # Deferred baseline seed: a tool_use event fires when the
                # command *starts*, and prepare_workspace.py takes longer to
                # pull a sheet than any sane retry window, so seeding on the
                # event itself finds an empty workspace. The agent acting
                # again is proof the script returned.
                if pending_seed and et in ("agent.message", "agent.tool_use"):
                    refresh_files(attempts=3, delay=2.0)
                    pending_seed = False
                if et == "agent.message":
                    for block in event.content:
                        if getattr(block, "type", None) == "text" and block.text.strip():
                            reply += ("\n\n" if reply else "") + block.text
                            timeline.append({"kind": "text", "text": block.text})
                            set_progress(progress, "working…")
                            with live:
                                st.markdown(block.text)
                elif et == "agent.tool_use":
                    label = _describe_tool_use(event)
                    tool_lines.append(label)
                    timeline.append({"kind": "tool", "label": label})
                    set_progress(progress, label)
                    with live:
                        st.markdown(f'<div class="cce-cmd">{html.escape(label)}</div>',
                                    unsafe_allow_html=True)
                    # Seed cce_files' "before" the moment the workspace is
                    # pulled. refresh_files otherwise runs only at the end of
                    # the turn, and prepare-then-edit routinely happens inside
                    # one turn — in which case first sighting is already the
                    # edited file and every diff comes out empty.
                    _inp = getattr(event, "input", None)
                    _cmd = _inp.get("command", "") if isinstance(_inp, dict) else ""
                    if pres.PREPARE_CMD.search(str(_cmd)):
                        pending_seed = True
                elif et == "agent.custom_tool_use":
                    if getattr(event, "name", None) == pres.TOOL_NAME:
                        set_progress(progress, "Showing you the edited content…")
                        payloads = handle_present_files(event)
                        for p in payloads:
                            # Index into presentations, so the woven view can
                            # place each diff where the agent actually showed
                            # it rather than in a pile at the end.
                            timeline.append({"kind": "presentation", "index": len(presentations)})
                            presentations.append(p)
                            tool_lines.append(f"Presented {p['path']}")
                            with live:
                                render_presentation(p)
                        progress.empty()
                    else:
                        # Unhandled custom tool still has to be answered, or
                        # the session stays parked at requires_action forever.
                        client.beta.sessions.events.send(
                            session_id=session_id,
                            events=[{
                                "type": "user.custom_tool_result",
                                "custom_tool_use_id": event.id,
                                "content": [{"type": "text", "text": json.dumps(
                                    {"status": "failed",
                                     "reason": f"no client handler for {getattr(event, 'name', '?')}"}
                                )}],
                            }],
                        )
                elif et.startswith(("session.thread", "agent.thread")):
                    if et == "session.thread_created":
                        tid = getattr(event, "session_thread_id", None)
                        if tid and tid not in threads:
                            timeline.append({"kind": "thread", "id": tid})
                            with live:
                                thread_slots[tid] = st.empty()
                    thread_label = track_thread_event(event, threads)
                    if thread_label:
                        tool_lines.append(thread_label)
                        set_progress(progress, thread_label)
                    # Re-draw every subagent panel this event moved, so a
                    # delegation shows as working and then fills in with its
                    # report, instead of staying blank until the turn ends.
                    for _tid, _slot in thread_slots.items():
                        if _tid in threads:
                            with _slot.container():
                                render_thread(threads[_tid])
                elif et == "session.error":
                    err = f"⚠ session error: {getattr(event, 'message', event)}"
                    tool_lines.append(err)
                    with live:
                        st.warning(err)
                elif et == "session.status_idle":
                    stop = getattr(event, "stop_reason", None)
                    if getattr(stop, "type", None) == "requires_action":
                        # Not the end of the turn: the agent is parked on a
                        # client-side event (a present_files call, answered
                        # above) and resumes once the result lands. Breaking
                        # here would end the turn mid-thought and leave the
                        # session waiting on a result that never comes.
                        continue
                    break
                elif et == "session.status_terminated":
                    break
    except Exception as exc:  # noqa: BLE001 — the connection dropped; the turn may not have.
        set_progress(progress, f"connection dropped ({exc}) — checking whether the turn finished…")
        extra_tools, extra_reply, completed = _reconcile_after_disconnect(session_id, last_event_at)
        tool_lines.extend(extra_tools)
        for line in extra_tools:
            timeline.append({"kind": "tool", "label": line})
            with live:
                st.markdown(f'<div class="cce-cmd">{html.escape(line)}</div>',
                            unsafe_allow_html=True)
        if extra_reply:
            reply += ("\n\n" if reply else "") + extra_reply
            timeline.append({"kind": "text", "text": extra_reply})
            with live:
                st.markdown(extra_reply)

        if completed:
            set_progress(progress, "Syncing changed files…")
            refresh_files(attempts=2, delay=2.0)
            progress.empty()
            track_tool_action(TOOL_NAME, log_action, duration_seconds=time.time() - started)
            return {"user": text, "tool_lines": tool_lines, "reply": reply,
                    "presentations": presentations, "timeline": timeline,
                    "threads": threads, "error": None}

        # Genuinely unresolved — still worth a defensive refresh, since tool
        # calls that ran before the drop may already have written files.
        progress.empty()
        refresh_files(attempts=1, delay=0)
        track_tool_action(TOOL_NAME, log_action, error_message=str(exc), duration_seconds=time.time() - started)
        return {
            "user": text, "tool_lines": tool_lines, "reply": reply,
            "presentations": presentations, "timeline": timeline, "threads": threads,
            "error": (
                f"Connection lost mid-turn ({exc}), and the session hadn't reached a stopping point yet as of "
                "the last check. It may still be running server-side — use \"Resync this session\" above to "
                "check again without resending anything."
            ),
        }

    set_progress(progress, "Syncing changed files…")
    track_tool_action(TOOL_NAME, log_action, duration_seconds=time.time() - started)
    refresh_files(attempts=2, delay=2.0)
    progress.empty()
    return {"user": text, "tool_lines": tool_lines, "reply": reply,
            "presentations": presentations, "timeline": timeline,
            "threads": threads, "error": None}


def stream_turn(text: str, live, progress, *, log_action: str = "turn_sent") -> dict:
    """Send one turn from the main chat box, rendering live as events arrive."""
    return _run_turn(text, live, progress, log_action=log_action)


def send_turn_blocking(text: str, *, log_action: str) -> None:
    """Used by the Review tab's Accept/Reject/Request-changes actions —
    short, administrative instructions. Renders its own st.status inline
    (same shape render_turn uses for history) rather than a plain spinner,
    so these turns look identical to chat-driven ones once the page reruns.
    """
    live = st.container()
    progress = st.empty()
    turn = _run_turn(text, live, progress, log_action=log_action)
    st.session_state.cce_transcript.append(turn)


_TASK_FILE_RE = re.compile(r"^File:\s*(\S+)", re.MULTILINE)


def track_thread_event(event, threads: dict[str, dict]) -> str | None:
    """Fold one delegation event into per-subagent records.

    A subagent's *internal* tool calls never reach this stream — measured
    against a real delegating session: every `agent.tool_use` there carried
    `session_thread_id: None`, i.e. they were all the coordinator's own. So
    there is no way to show "Editor 2 is editing block 5", at any price.

    What the stream does carry, per thread, is the brief the coordinator
    sent, the status transitions, and the editor's full report — and the
    report is the valuable part, since it is where an editor states its
    measured before/after counts. Today that text is thrown away and
    summarized as "← report received", which is why delegation reads as a
    black box.

    Returns a short status line for the live status box, or None.
    """
    et = event.type
    if et == "session.thread_created":
        tid = getattr(event, "session_thread_id", None)
        if tid:
            threads.setdefault(tid, {}).update(
                {"agent": getattr(event, "agent_name", None) or "editor",
                 "status": "working", "task": "", "file": "", "report": ""}
            )
            return f"delegating to {threads[tid]['agent']}…"
        return None

    if et == "agent.thread_message_sent":
        tid = getattr(event, "to_session_thread_id", None)
        if not tid:
            return None
        text = "".join(
            b.text for b in (event.content or []) if getattr(b, "type", None) == "text"
        )
        rec = threads.setdefault(tid, {"agent": getattr(event, "to_agent_name", None) or "editor",
                                       "status": "working", "task": "", "file": "", "report": ""})
        rec["task"] = text
        m = _TASK_FILE_RE.search(text)
        if m:
            rec["file"] = m.group(1).rsplit("/", 1)[-1]
        return f"→ {rec['file'] or rec['agent']} handed off"

    if et == "agent.thread_message_received":
        tid = getattr(event, "from_session_thread_id", None)
        if not tid:
            return None
        text = "".join(
            b.text for b in (event.content or []) if getattr(b, "type", None) == "text"
        )
        rec = threads.setdefault(tid, {"agent": getattr(event, "from_agent_name", None) or "editor",
                                       "status": "done", "task": "", "file": "", "report": ""})
        rec["report"] = text
        rec["status"] = "done"
        return f"← {rec['file'] or rec['agent']} reported back"

    if et in ("session.thread_status_idle", "session.thread_status_terminated"):
        # Fires for the coordinator's own thread too, which is not a
        # delegation — only touch threads we actually created.
        tid = getattr(event, "session_thread_id", None)
        if tid in threads and threads[tid].get("status") != "done":
            threads[tid]["status"] = "finished" if et.endswith("terminated") else "idle"
    return None


def set_progress(slot, text: str) -> None:
    """Spinner + the current activity, in the placeholder below the turn.

    Transient by contract: this never enters the transcript, so it can say
    things a replayed turn shouldn't ("Syncing changed files…"). The spinner
    is the point — the gaps here are long (a delegated sweep runs a minute
    or more with nothing to print), and a motionless line reads as a frozen
    page.
    """
    slot.markdown(
        f'<div class="cce-working"><span class="cce-spin"></span>{html.escape(text)}</div>',
        unsafe_allow_html=True,
    )


def render_timeline(turn: dict) -> None:
    """The woven view: agent prose and tool activity in the order they
    happened, rather than every command collected into one status box
    detached from the sentence that explains it.

    Commands are set in mono (`.cce-cmd`) because that is what they are —
    it keeps them legible as machine output without letting them compete
    with the agent's own writing.
    """
    threads = turn.get("threads") or {}
    for item in turn.get("timeline") or []:
        kind = item.get("kind")
        if kind == "text":
            st.markdown(item["text"])
        elif kind == "tool":
            st.markdown(f'<div class="cce-cmd">{html.escape(item["label"])}</div>',
                        unsafe_allow_html=True)
        elif kind == "thread":
            rec = threads.get(item["id"])
            if rec:
                render_thread(rec)
        elif kind == "presentation":
            shown = (turn.get("presentations") or [])
            if item["index"] < len(shown):
                render_presentation(shown[item["index"]])


def render_thread(rec: dict) -> None:
    """One subagent, its brief and its report, in its own expander."""
    name = rec.get("file") or rec.get("agent") or "editor"
    status = rec.get("status") or "working"
    icon = {"done": "✓", "working": "⋯", "idle": "⋯", "finished": "✓"}.get(status, "⋯")
    with st.expander(f"{icon} {name} · {rec.get('agent', 'editor')} · {status}", expanded=False):
        if rec.get("task"):
            st.markdown('<div class="cce-thread-meta">Brief it was given</div>',
                        unsafe_allow_html=True)
            st.markdown(f'<div class="cce-cmd">{html.escape(rec["task"][:1200])}</div>',
                        unsafe_allow_html=True)
        if rec.get("report"):
            st.markdown('<div class="cce-thread-meta">What it reported back</div>',
                        unsafe_allow_html=True)
            st.markdown(rec["report"])
        elif status != "done":
            st.caption("Still working — its report appears here when it finishes.")


def render_turn(turn: dict) -> None:
    """Replay one completed turn.

    Two shapes are supported on purpose. Turns carrying a `timeline` render
    woven — prose and commands interleaved in the order they happened, with
    a per-subagent expander where each delegation occurred. Older turns
    (and those rebuilt by load_session_transcript) have only `tool_lines`
    and `reply`, and keep the original status-box shape; transcript entries
    outlive the code that wrote them, so both paths have to work.
    """
    with st.chat_message("user"):
        st.markdown(turn["user"])
    with st.chat_message("assistant"):
        if turn.get("timeline"):
            render_timeline(turn)
        else:
            if turn["tool_lines"]:
                label = "Failed" if turn["error"] else "Done"
                state = "error" if turn["error"] else "complete"
                status = st.status(label, state=state, expanded=False)
                for line in turn["tool_lines"]:
                    status.write(line)
            if turn["reply"]:
                st.markdown(turn["reply"])
        if turn["error"]:
            st.error(f"Request failed: {turn['error']}")
        if not turn.get("timeline"):
            # Legacy turns: the timeline already placed these inline.
            for presentation in turn.get("presentations") or []:
                render_presentation(presentation)


def refresh_files(attempts: int = 3, delay: float = 2.0) -> None:
    """Poll the session-scoped Files API and update the before/after cache.

    'before' is set only the first time a file is seen (the original
    content); 'after' is overwritten on every call. Files written under
    /mnt/session/outputs/ (which the skill's SKILL.md now mandates as the
    workspace path) surface here — sometimes a few seconds after the turn
    goes idle, hence the small retry loop.

    The Files API exposes only a bare `filename`, not the sandbox path
    (confirmed against the SDK's FileMetadata type — no path field exists).
    That's a real collision: `topics/topic_01_<slug>.md` and
    `context/topic_01_<slug>.md` share an identical basename. So every
    candidate is downloaded and classified by its actual content
    (`###Block ID:` -> topics, `###LO ID:` -> context) rather than trusted by
    name, and cached under a `topics/<name>` / `context/<name>` key that
    matches the manifest's own relative-path values exactly.

    `files.list()` returns a paginated SyncPage — every write the agent
    makes appears to create a new file object, so the session accumulates
    one per edit across a multi-turn conversation. Reading only `.data`
    (the first page) meant that once enough turns had passed, the file
    holding an actual edit could fall off that first page while an older
    (pre-edit) version of the same logical file was still visible, making
    an already-shown diff silently disappear on a later, unrelated turn.
    Iterating the page object directly auto-paginates through everything.
    """
    client = get_client()
    session_id = st.session_state.cce_session_id
    if not session_id:
        return
    for attempt in range(attempts):
        try:
            listed = list(client.beta.files.list(scope_id=session_id, betas=FILES_BETA))
        except Exception:
            break
        candidates = [f for f in listed if f.filename.endswith(".md") or f.filename == "manifest.json"]

        latest_by_key: dict[tuple[str, str], tuple] = {}  # (kind, filename) -> (created_at, content, file_id)
        for f in candidates:
            try:
                content = client.beta.files.download(f.id).read().decode("utf-8")
            except Exception:
                continue
            if f.filename == "manifest.json":
                kind = "manifest"
            elif cc_topics.BLOCK_START.search(content):
                kind = "topics"
            elif cc_context.LO_BLOCK_START.search(content):
                kind = "context"
            else:
                continue  # not a workspace file we know how to diff
            key = (kind, f.filename)
            existing = latest_by_key.get(key)
            if existing is None or f.created_at > existing[0]:
                latest_by_key[key] = (f.created_at, content, f.id)

        for (kind, filename), (_, content, file_id) in latest_by_key.items():
            store_key = filename if kind == "manifest" else f"{kind}/{filename}"
            entry = st.session_state.cce_files.setdefault(
                store_key, {"before": content, "after": content, "file_id": file_id, "kind": kind}
            )
            entry["after"] = content
            entry["file_id"] = file_id
            entry["kind"] = kind
            if kind == "manifest":
                try:
                    st.session_state.cce_manifest = json.loads(content)
                except json.JSONDecodeError:
                    pass
        if attempt < attempts - 1:
            time.sleep(delay)


# ---------------------------------------------------------------------------
# present_files — the agent's only sanctioned way to show content
# ---------------------------------------------------------------------------
# A custom tool is client-side by definition: Anthropic emits
# agent.custom_tool_use, parks the session at idle/requires_action, and the
# turn does not move until a user.custom_tool_result goes back. So the
# handler's contract is "always answer" — every branch, including failures.
#
# The Review tab already renders every changed file on its own. This is not
# that: it's the agent pointing at specific blocks at the moment it's
# talking about them, in the chat, which is the difference between a diff
# the user has to go find and one put in front of them.


def handle_present_files(event) -> list[dict]:
    """Answer one present_files call; return payloads for render_turn.

    Returns plain dicts (paths, block ids, before/after strings) rather than
    rendered HTML because they get stored on the turn and replayed by
    render_turn on every rerun — the same reason turns carry tool_lines
    instead of a finished st.status.
    """
    client = get_client()
    session_id = st.session_state.cce_session_id
    items = pres.normalize_items(getattr(event, "input", None))
    payloads: list[dict] = []
    result_text = None

    try:
        if not items:
            result_text = json.dumps({"status": "failed", "reason": "no items in input"})
        else:
            # The agent calls this straight after editing, and a just-written
            # file can lag a beat behind in the Files API — the same lag
            # refresh_files' retry loop exists for. Refresh before resolving,
            # and once more if something the agent named isn't there yet.
            refresh_files(attempts=2, delay=1.5)
            if any(pres.resolve_key(st.session_state.cce_files, i["path"]) is None for i in items):
                refresh_files(attempts=2, delay=2.0)

            missing: list[str] = []
            facts: dict[str, dict] = {}
            # event.id is unique per tool call, so "<id>-<n>" is a stable
            # identity for this card across every later rerun — which is what
            # the view toggle needs for its widget key. A key derived from the
            # path alone would collide the moment the same file is presented
            # in two turns.
            for n, item in enumerate(items):
                uid = f"{event.id}-{n}"
                key = pres.resolve_key(st.session_state.cce_files, item["path"])
                if key is None:
                    missing.append(item["path"])
                    payloads.append({"path": item["path"], "note": item["note"],
                                     "uid": uid, "missing": True})
                    continue
                entry = st.session_state.cce_files[key]
                # Carry the raw before/after, not a pre-computed selection:
                # the cards are built by the same diff_topic_file /
                # diff_context_file the Review tab uses, so the two views are
                # the same view. Storing the text also means render_turn can
                # rebuild them on every rerun without re-fetching.
                payloads.append({
                    "path": key,
                    "note": item["note"],
                    "uid": uid,
                    "kind": entry.get("kind") or "topics",
                    "before": entry["before"],
                    "after": entry["after"],
                    "missing": False,
                })
                # Keyed by the path the AGENT used — that's what the result
                # echoes back to it, not the resolved cache key.
                facts[item["path"]] = pres.measure(
                    entry["before"], entry["after"], entry.get("kind") or "topics")
            result_text = (
                pres.unresolved_result(items, missing, facts) if missing
                else pres.ack_result(items, facts)
            )
    except Exception as exc:  # noqa: BLE001 — see above: the turn is blocked on us.
        result_text = json.dumps({"status": "failed", "reason": str(exc)})

    client.beta.sessions.events.send(
        session_id=session_id,
        events=[{
            "type": "user.custom_tool_result",
            "custom_tool_use_id": event.id,
            "content": [{"type": "text", "text": result_text}],
        }],
    )
    return payloads




def render_presentation(p: dict) -> None:
    """One presented file, in its own expander.

    Reads the payload defensively. Transcript entries outlive the code that
    wrote them — st.session_state survives a hot reload, so after any change
    to what handle_present_files stores, the transcript still holds turns in
    the previous shape and re-renders them on the very next rerun. (That
    surfaced as a KeyError: 'before' the first time this payload changed.)
    Anything missing is re-derived from the live file cache, which is keyed
    by the same path and is the same content the payload was a snapshot of.
    """
    path = p.get("path") or "(unknown file)"
    name = path.rsplit("/", 1)[-1]
    if p.get("missing"):
        with st.expander(f"⚠ {path} — not found in the workspace", expanded=False):
            st.caption("The agent named a file that isn't in the current workspace.")
        return

    before, after = p.get("before"), p.get("after")
    kind = p.get("kind")
    if before is None or after is None:
        entry = st.session_state.cce_files.get(path)
        if entry is None:
            with st.expander(f"{name} — diff unavailable", expanded=False):
                st.caption(
                    "This was presented in an earlier turn and its content is no longer "
                    "cached. The Review tab has the current diff."
                )
            return
        before, after = entry["before"], entry["after"]
        kind = kind or entry.get("kind")

    # Cached on the file's own bytes — see services/cce_diff.rendered_cards.
    # Every rerun re-rendered every presented file from scratch, so the page
    # got slower with each turn while producing identical HTML.
    cards = rendered_cards(before, after, kind or "topics")
    n = len(cards)
    header = f"{name} — {n} change{'' if n == 1 else 's'}" if n else f"{name} — no changes"
    with st.expander(header, expanded=False):
        if p.get("note"):
            st.caption(p["note"])
        if not cards:
            st.caption("Nothing differs from the original in this file.")
            return

        # Purely a rendering choice — flipping it re-renders from the same
        # stored before/after and sends nothing to the agent.
        # `uid` keeps the key stable across reruns and distinct from the same
        # file presented in another turn; the hash fallback covers transcript
        # entries written before uid existed.
        uid = p.get("uid") or f"legacy{abs(hash((path, p.get('note', ''))))}"
        show_diff = st.toggle(
            "Show changes", value=True, key=f"presdiff_{uid}",
            help="On: redline against the original. Off: the content as it reads now.",
        )
        if not show_diff and render_comment_surface(p, before, after, kind or "topics", uid):
            return
        for label, meta, extra, body, plain in cards:
            if not show_diff:
                # Keep the same label/meta so the card doesn't jump around
                # when toggled — but strip the redline markup out of the
                # label, which is itself a word diff of the topic name.
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




# ---------------------------------------------------------------------------
# Resume-on-load: if this browser tab has no session yet (fresh page load,
# server restart, a different device — cce_session_id lives only in
# Streamlit's in-memory session_state, which the Managed Agents session
# itself does not depend on), look for this user's most recent still-alive
# session and reattach to it instead of silently starting over. Guarded by
# cce_resume_checked so this only ever costs one API call per browser tab,
# not one per rerun.
# ---------------------------------------------------------------------------
if not st.session_state.cce_session_id and not st.session_state.get("cce_resume_checked"):
    st.session_state.cce_resume_checked = True
    _candidates = find_user_sessions(get_user_email())
    if _candidates:
        resume_session(_candidates[0].id)
        st.session_state.cce_just_resumed = True

# ---------------------------------------------------------------------------
# Session bar — no sidebar, just a slim row: which session this is, a link
# to watch it on platform.claude.com, and controls to start fresh or switch
# to a different one of this user's sessions.
# ---------------------------------------------------------------------------
_bar_left, _bar_resync, _bar_mid, _bar_right = st.columns([3, 1, 1, 1])
with _bar_left:
    if st.session_state.cce_session_id:
        resumed_note = " · resumed" if st.session_state.get("cce_just_resumed") else ""
        st.caption(
            f"Session `{st.session_state.cce_session_id}`{resumed_note} · "
            f"[watch on platform.claude.com](https://platform.claude.com/workspaces/default/sessions/"
            f"{st.session_state.cce_session_id})"
        )
    else:
        st.caption("No session yet — it starts as soon as you send a message.")
with _bar_resync:
    # For when a turn's connection dropped and _run_turn's own reconciliation
    # couldn't confirm completion yet (e.g. the server hadn't caught up at
    # that moment) — re-pull the session's actual history and files fresh
    # from the server without resending anything.
    if st.session_state.cce_session_id and st.button("Resync this session", use_container_width=True):
        resume_session(st.session_state.cce_session_id)
        st.rerun()
with _bar_mid:
    if st.session_state.cce_session_id and st.button("Start new session", use_container_width=True):
        for _k, _v in _DEFAULTS.items():
            st.session_state[_k] = _v
        st.session_state.cce_resume_checked = True  # we're deliberately starting fresh, don't auto-resume again
        st.session_state.cce_just_resumed = False
        st.rerun()
with _bar_right:
    with st.popover("Switch session", use_container_width=True):
        _others = [s for s in find_user_sessions(get_user_email()) if s.id != st.session_state.cce_session_id]
        if not _others:
            st.caption("No other sessions for you yet.")
        for _s in _others[:10]:
            _label = (_s.title or _s.id).removeprefix("Course content editor — ")
            if st.button(f"{_label} · {_s.updated_at:%b %d, %H:%M}", key=f"switch_{_s.id}", use_container_width=True):
                resume_session(_s.id)
                st.session_state.cce_just_resumed = True
                st.rerun()

tab_chat, tab_review = st.tabs(["💬 Chat", "📝 Review changes"])


# ---------------------------------------------------------------------------
# Chat tab — full-width native chat UI, live-updating as events arrive.
# ---------------------------------------------------------------------------
with tab_chat:
    for turn in st.session_state.cce_transcript:
        render_turn(turn)


# ---------------------------------------------------------------------------
# Review tab — diff cards with accept / reject / request-changes.
# ---------------------------------------------------------------------------
with tab_review:
    if not st.session_state.cce_session_id:
        st.caption("Start chatting to begin a session — changes will show up here as diffs.")
    else:
        # Sort by the manifest's own topic order, not whatever order the
        # Files API happens to return files in — confirmed that's unrelated
        # to topic sequence (it's what caused a later topic's content to
        # render before an earlier one's). Files not in the current
        # manifest sort last rather than vanishing, same reasoning as
        # dispatching on content-sniffed `kind` instead of the manifest.
        _manifest = st.session_state.cce_manifest or {}
        _topic_order: dict[str, int] = {}
        for _t in _manifest.get("topics", []):
            _idx = _t.get("index", 0)
            if _t.get("file"):
                _topic_order[_t["file"]] = _idx
            if _t.get("context_file"):
                _topic_order[_t["context_file"]] = _idx
        _sorted_files = sorted(
            st.session_state.cce_files.items(),
            key=lambda kv: (_topic_order.get(kv[0], float("inf")), kv[0]),
        )

        any_cards = False

        for filename, entry in _sorted_files:
            if filename == "manifest.json":
                continue
            if entry["before"] == entry["after"]:
                continue

            # Dispatch on the content-sniffed `kind` recorded when this file
            # was downloaded (refresh_files), NOT by re-checking against the
            # current manifest.json. manifest.json can get rewritten
            # mid-conversation (e.g. the agent re-runs a prepare script to
            # answer an unrelated question) with different topic ordering or
            # slugs — cross-referencing against it lets an already-shown
            # diff silently vanish the moment its old path no longer matches
            # a fresh manifest, even though the edit itself is still there.
            if entry.get("kind") == "context":
                cards = diff_context_file(entry["before"], entry["after"])
            elif entry.get("kind") == "topics":
                cards = diff_topic_file(entry["before"], entry["after"])
            else:
                continue

            for idx, card in enumerate(cards):
                any_cards = True
                card_id = f"{filename}:{card['kind']}:{idx}:{card.get('key', '')}"
                status = st.session_state.cce_approvals.get(card_id)

                with st.container(border=True):
                    # Not a hand-rolled <div class="cce-card">...</div> spanning
                    # multiple st.markdown() calls — that can't actually work:
                    # every st.markdown() call is its own independent block, so
                    # an opening tag in one call and a closing tag in a later
                    # one never nest; the browser just auto-closes the empty
                    # opening div on the spot. That's exactly what produced the
                    # empty boxes between (and before) every card. st.container
                    # is a real box that genuinely wraps everything inside it —
                    # text, columns, buttons — no HTML-splitting trick needed.
                    if card["kind"] == "context_row":
                        b, a = card["before"], card["after"]
                        badge = (f'<span class="cce-badge cce-badge-{status}">{status}</span>' if status
                                 else '<span class="cce-badge cce-badge-pending">pending</span>')
                        label, meta, extra, body = diff_card_html(card)
                        st.markdown(
                            f'<div class="cce-card-label">{label}{badge}</div>'
                            f'<div class="cce-card-meta">{html.escape(meta)}</div>'
                            f'<div class="cce-card-lo">{extra}</div>'
                            f'<div class="cce-diff-text">{body}</div>',
                            unsafe_allow_html=True,
                        )
                        c1, c2, c3 = st.columns([1, 1, 2])
                        if c1.button("Accept", key=f"acc_{card_id}"):
                            st.session_state.cce_approvals[card_id] = "accepted"
                            st.rerun()
                        if c2.button("Reject", key=f"rej_{card_id}"):
                            msg = (
                                f'Revert the research notes for Topic "{b["Topic"]}", Subtopic "{b["Subtopic"]}", '
                                f'Learning Objective "{b["Learning Objective"]}" back to exactly this original text '
                                f'(keep everything else in the workspace exactly as it currently is):\n\n"""\n'
                                f'{b["Research Notes"]}\n"""'
                            )
                            st.session_state.cce_approvals[card_id] = "rejected"
                            with st.spinner("Reverting…"):
                                send_turn_blocking(msg, log_action="row_rejected")
                            st.rerun()
                        with c3.popover("Request changes"):
                            note = st.text_area("What should change about this row?", key=f"note_{card_id}")
                            if st.button("Send", key=f"sendnote_{card_id}"):
                                msg = (
                                    f'For the research notes on Topic "{b["Topic"]}", Subtopic "{b["Subtopic"]}", '
                                    f'Learning Objective "{b["Learning Objective"]}": {note}'
                                )
                                with st.spinner("Working…"):
                                    send_turn_blocking(msg, log_action="row_change_requested")
                                st.rerun()

                    elif card["kind"] == "context_new":
                        a = card["after"]
                        label, meta, extra, body = diff_card_html(card)
                        st.markdown(
                            f'<div class="cce-card-label">{label}'
                            f'<span class="cce-badge cce-badge-pending">new row</span></div>'
                            f'<div class="cce-card-meta">{html.escape(meta)}</div>'
                            f'<div class="cce-card-lo">{extra}</div>'
                            f'<div class="cce-diff-text">{body}</div>',
                            unsafe_allow_html=True,
                        )

                    elif card["kind"] == "topic_group":
                        before_g, after_g = card["before"], card["after"]
                        before_name = before_g["topic"] if before_g else ""
                        after_name = after_g["topic"] if after_g else ""
                        badge = (f'<span class="cce-badge cce-badge-{status}">{status}</span>' if status
                                 else '<span class="cce-badge cce-badge-pending">pending</span>')
                        label, meta, _extra, body = diff_card_html(card)
                        st.markdown(
                            f'<div class="cce-card-label">{label}{badge}</div>'
                            f'<div class="cce-card-meta">{html.escape(meta)}</div>'
                            f'<div class="cce-diff-text">{body}</div>',
                            unsafe_allow_html=True,
                        )
                        c1, c2, c3 = st.columns([1, 1, 2])
                        if c1.button("Accept", key=f"acc_{card_id}"):
                            st.session_state.cce_approvals[card_id] = "accepted"
                            st.rerun()
                        if c2.button("Reject", key=f"rej_{card_id}"):
                            # Quote the original blocks verbatim rather than
                            # naming the topic: after a rename the agent has
                            # no other handle on what the old version was.
                            original = "\n\n".join(
                                format_topic_block_for_message(b) for b in (before_g["blocks"] if before_g else [])
                            )
                            if original:
                                msg = (
                                    f'Revert Topic "{after_name or before_name}" back to exactly these original '
                                    f"blocks, including its original topic name (keep every other topic "
                                    f"as-is):\n\n{original}"
                                )
                            else:
                                msg = (
                                    f'Remove the topic "{after_name}" you just added — it did not exist in the '
                                    f"original content. Keep every other topic as-is."
                                )
                            st.session_state.cce_approvals[card_id] = "rejected"
                            with st.spinner("Reverting…"):
                                send_turn_blocking(msg, log_action="topic_rejected")
                            st.rerun()
                        with c3.popover("Request changes"):
                            note = st.text_area("What should change in this topic?", key=f"note_{card_id}")
                            if st.button("Send", key=f"sendnote_{card_id}"):
                                msg = f'For Topic "{after_name or before_name}": {note}'
                                with st.spinner("Working…"):
                                    send_turn_blocking(msg, log_action="topic_change_requested")
                                st.rerun()

        if not any_cards:
            st.caption("No pending edits yet. Ask for a change in the chat and it'll show up here as a diff.")

        st.divider()
        st.caption(
            "To write changes back to the sheet, just ask for it in the chat — e.g. \"write the "
            "research notes back\" or \"commit the slide chunks\". It always writes to a **new** "
            "tab; the source tabs are never touched."
        )


# ---------------------------------------------------------------------------
# Chat input — declared here, genuinely after both tab blocks have finished
# rendering, not textually sandwiched between them. st.chat_input still
# pins itself to the bottom of the page regardless of where in the script
# it's called; what matters is NOT leaving its declaration site between two
# `with tab_x:` blocks, which left an empty placeholder rectangle at the
# top of whichever tab rendered next (both tab panels exist in the DOM
# simultaneously — Streamlit only toggles visibility, it doesn't unmount —
# so that stray slot was landing inside the live tab every time).
# ---------------------------------------------------------------------------
chat_prompt = st.chat_input("Paste a course Google Sheet URL, or say what to change…")

# Selection comments arrive as a queued message rather than being sent from
# inside the expander they were written in — a turn started there would
# render into the middle of an earlier turn.
if not chat_prompt and st.session_state.cce_pending_feedback:
    chat_prompt = st.session_state.cce_pending_feedback
    st.session_state.cce_pending_feedback = None

if chat_prompt:
    with tab_chat:
        with st.chat_message("user"):
            st.markdown(chat_prompt)
        with st.chat_message("assistant"):
            # The turn renders into `live` in event order — the same shape
            # render_timeline replays afterwards, so nothing rearranges on
            # screen when the turn finishes. `progress` sits below it for
            # transient "still working" text that never enters the transcript.
            live_area = st.container()
            progress_slot = st.empty()
            set_progress(progress_slot, "Working…")
        completed_turn = stream_turn(chat_prompt, live_area, progress_slot)
        st.session_state.cce_transcript.append(completed_turn)
    # Deliberately no st.rerun() here. The turn is already on screen in the
    # exact shape render_turn would replay it in, and it is already in the
    # transcript, so a rerun would re-render the entire conversation to
    # produce a pixel-identical result — visible as the app going "Running"
    # and the page flickering the moment the agent finishes. It only existed
    # to normalize a live shape that no longer differs from the replayed one.
    #
    # The cost: tabs rendered earlier in this script run (the Review tab's
    # diff cards) still show their pre-turn state until the next interaction.
    # That is a one-interaction lag on a secondary panel, against a full
    # re-render on every single turn.
