"""Client-side handling for the agent's `present_files` custom tool.

The agent cannot show the user file content by writing it into a message —
it demonstrably makes text up when it tries (an editing turn that printed
nine "rewritten" slides while the files on disk were untouched is the
motivating case). So it doesn't get to: the system prompt forbids pasting
content, and instead the agent calls `present_files` with *paths*, and the
client renders the real bytes it fetches from the session-scoped Files API.

`present_files` is a custom tool, which in Managed Agents means client-side
by definition: Anthropic never executes it. The session emits
`agent.custom_tool_use`, goes idle with `stop_reason.type ==
"requires_action"`, and waits for a `user.custom_tool_result` before the
turn can continue. Both of this repo's clients (chat.py and the Streamlit
page) share the logic here so their two renderings can't drift apart on
what "block 4 of topic 2" means.

Deliberately free of Streamlit and of any Anthropic client construction —
callers pass in their own `client`. That's what lets chat.py use it.
"""

from __future__ import annotations

import json
import re
import sys
import textwrap
import time
from pathlib import Path

# Same sys.path convention course_content_editor.py uses to reach the skill's
# scripts/ directory: the block-format contract (`###Block ID:` / `###LO ID:`)
# is defined by the commit scripts, and re-implementing it here would be a
# second, silently-diverging copy of it.
_SCRIPTS_DIR = Path(__file__).resolve().parent / "skills" / "working-with-google-sheets" / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import commit_context as cc_context  # noqa: E402  (parse_lo_block, LO_BLOCK_START)
import commit_workspace as cc_topics  # noqa: E402  (parse_block, BLOCK_START)

TOOL_NAME = "present_files"
FILES_BETA = ["managed-agents-2026-04-01"]


# ---------------------------------------------------------------------------
# Parsing — block id preserved
# ---------------------------------------------------------------------------
# The page's own _parse_topic_blocks/_parse_lo_blocks drop the id (they feed
# a diff that matches on content, not identity). `present_files` addresses
# blocks *by* id, so it needs it kept.


def classify(content: str) -> str | None:
    """topics / context / None, by content rather than filename.

    The Files API exposes only a bare basename, and `topics/topic_01_x.md`
    and `context/topic_01_x.md` share one — so the name genuinely cannot
    disambiguate them. Same content-sniffing rule as refresh_files().
    """
    if cc_topics.BLOCK_START.search(content):
        return "topics"
    if cc_context.LO_BLOCK_START.search(content):
        return "context"
    return None


def parse_blocks(content: str, kind: str) -> list[dict]:
    """Blocks in file order, each with its declared id under "_id"."""
    if kind == "topics":
        pattern, parse = cc_topics.BLOCK_START, cc_topics.parse_block
    else:
        pattern, parse = cc_context.LO_BLOCK_START, cc_context.parse_lo_block

    starts = list(pattern.finditer(content))
    out: list[dict] = []
    for i, m in enumerate(starts):
        end = starts[i + 1].start() if i + 1 < len(starts) else len(content)
        block = parse(content[m.end():end])
        if block:
            out.append({"_id": m.group(1).strip(), **block})
    return out


def block_label(block: dict, kind: str) -> str:
    if kind == "topics":
        return f"[{block.get('Slide Type', '?')}] {block.get('Slide Chunk Title', '')}".strip()
    return block.get("Learning Objective", "") or block.get("Subtopic", "")


def block_body(block: dict, kind: str) -> str:
    return block.get("Slide Chunk", "") if kind == "topics" else block.get("Research Notes", "")


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def resolve_key(store_keys, path: str) -> str | None:
    """Map an agent-supplied path onto a `<kind>/<basename>` cache key.

    The agent works in the sandbox and will refer to files however it
    happens to have them in context — an absolute `/mnt/session/outputs/
    workspace/topics/topic_01_x.md`, a workspace-relative `topics/
    topic_01_x.md`, or a bare `topic_01_x.md`. Cache keys are always
    `<kind>/<basename>`. Rejecting anything but the middle form would turn
    a cosmetic mismatch into "the agent showed you nothing", so all three
    resolve; a bare basename resolves only when it is unambiguous, since
    topics/ and context/ share basenames by design.
    """
    keys = list(store_keys)
    norm = str(path).replace("\\", "/").strip().strip("/")
    if not norm:
        return None

    parts = [p for p in norm.split("/") if p]
    base = parts[-1]
    parent = parts[-2] if len(parts) >= 2 else None

    if parent in ("topics", "context"):
        candidate = f"{parent}/{base}"
        if candidate in keys:
            return candidate

    matches = [k for k in keys if k.rsplit("/", 1)[-1] == base]
    if len(matches) == 1:
        return matches[0]
    return None  # ambiguous bare basename, or no such file


# ---------------------------------------------------------------------------
# Tool input / output
# ---------------------------------------------------------------------------


def normalize_items(tool_input) -> list[dict]:
    """Coerce the tool input into a list of {path, block_ids, note}.

    Tool inputs are model-generated JSON, so this stays tolerant: a bare
    string, a single object instead of a list, `block_ids` given as ints or
    as a comma-joined string. None of those are worth failing a turn over.
    """
    if isinstance(tool_input, str):
        try:
            tool_input = json.loads(tool_input)
        except (ValueError, TypeError):
            return []
    if not isinstance(tool_input, dict):
        return []

    raw = tool_input.get("items")
    if raw is None:
        raw = [tool_input] if tool_input.get("path") else []
    if isinstance(raw, (str, dict)):
        raw = [raw]
    if not isinstance(raw, list):
        return []

    items: list[dict] = []
    for entry in raw:
        if isinstance(entry, str):
            entry = {"path": entry}
        if not isinstance(entry, dict):
            continue
        path = entry.get("path") or entry.get("file") or entry.get("filename")
        if not path:
            continue

        block_ids = entry.get("block_ids")
        if block_ids is None:
            block_ids = []
        elif isinstance(block_ids, (str, int)):
            block_ids = re.split(r"[,\s]+", str(block_ids).strip())
        block_ids = [str(b).strip() for b in block_ids if str(b).strip() != ""]

        items.append({
            "path": str(path),
            "block_ids": block_ids,
            "note": str(entry.get("note") or "").strip(),
        })
    return items


def measure(before: str, after: str, kind: str) -> dict:
    """Counted facts about one file's edit, for the tool result.

    Deliberately only three, and only ones that are exact: how many blocks,
    how many words, and (for slide chunks) how many of each slide type.
    Each replaces a number the agent stated wrongly while reporting an edit
    as done — "35 -> 34 blocks" for a 34 -> 33 change, "~12% reduction" on
    a course that grew 14-15%, and nine deleted transition slides that went
    unmentioned. Measuring costs nothing and removes the need to estimate.

    An earlier version also reported which block IDs were added, removed,
    and changed, plus per-block word counts. Those are gone: they key off
    block IDs, and nothing stops an editor from renumbering blocks — after
    which "removed_block_ids: [11]" means only "there is no longer an ID
    11", not that any slide was removed. A fact that needs a caveat
    explaining when it lies is worse than no fact, because it will be
    quoted without the caveat. Every count below survives renumbering.
    """
    before_blocks = parse_blocks(before, kind)
    after_blocks = parse_blocks(after, kind)

    def words(blocks) -> int:
        return sum(len(block_body(b, kind).split()) for b in blocks)

    wb, wa = words(before_blocks), words(after_blocks)
    facts = {
        "blocks": {"before": len(before_blocks), "after": len(after_blocks)},
        "words": {"before": wb, "after": wa,
                  "change_pct": round((wa - wb) / wb * 100, 1) if wb else None},
    }
    if kind == "topics":
        # The slide-type census is what makes "I deleted every transition
        # slide" impossible to overlook — the largest unreported change
        # observed so far — and it needs no block identity to compute.
        def census(blocks):
            out: dict[str, int] = {}
            for b in blocks:
                key = b.get("Slide Type") or "?"
                out[key] = out.get(key, 0) + 1
            return out
        facts["slide_types"] = {"before": census(before_blocks), "after": census(after_blocks)}
    return facts


_COUNTS_NOTE = (
    "These counts are measured from the files. Any number you state must come "
    "from here — do not estimate one."
)


def ack_result(items: list[dict], facts: dict[str, dict] | None = None) -> str:
    """The tool result sent back as `user.custom_tool_result`.

    Carries counted facts per presented file (see measure). The agent has
    no other reliable source for them: its own reading of a long file
    produces numbers that have been wrong every time they were checked, and
    a wrong count reads exactly like a right one to whoever is reviewing.
    """
    facts = facts or {}
    return json.dumps({
        "status": "presented",
        "files": [
            {"path": i["path"], **(facts.get(i["path"]) or {})} for i in items
        ],
        "note": _COUNTS_NOTE,
    })


def unresolved_result(items: list[dict], missing: list[str], facts: dict[str, dict] | None = None) -> str:
    """Result when some paths matched nothing.

    Reported rather than swallowed: if the agent named a file that isn't in
    the workspace it should find that out and correct itself, not carry on
    believing the user is looking at something. The files that *did* resolve
    still carry their counts, so a partial failure doesn't cost the agent
    the facts for the rest.
    """
    facts = facts or {}
    return json.dumps({
        "status": "partial" if len(missing) < len(items) else "failed",
        "files": [
            {"path": i["path"], **(facts.get(i["path"]) or {})}
            for i in items if i["path"] not in missing
        ],
        "not_found": missing,
        "hint": "Paths are workspace-relative, e.g. topics/topic_01_<slug>.md",
        "note": _COUNTS_NOTE,
    })


# ---------------------------------------------------------------------------
# Files API
# ---------------------------------------------------------------------------


# A bash command running either prepare script is the moment the pristine
# workspace lands — the last moment a client can still capture a real
# "before".
PREPARE_CMD = re.compile(r"prepare_(?:context_)?workspace(?:_xlsx)?\.py")


def fetch_workspace_files(client, session_id: str) -> dict[str, dict]:
    """Current content of every workspace .md file, keyed `<kind>/<basename>`.

    There is exactly one file object per logical file and downloading it
    always returns the file's *current* bytes — measured, not assumed: a
    session whose three topic files had been edited exposed one object each,
    holding only the edited content, with the pre-edit version nowhere in the
    listing. (`created_at` looks like version history at first glance, but
    the duplicate timestamps are `topics/x.md` and `context/x.md` sharing a
    basename, not two versions of one file.)

    So a "before" cannot be recovered after the fact — it has to be captured
    while it is still current, which is what seed_baselines() is for.
    """
    latest: dict[str, tuple] = {}
    for f in client.beta.files.list(scope_id=session_id, betas=FILES_BETA):
        if not f.filename.endswith(".md"):
            continue
        try:
            content = client.beta.files.download(f.id).read().decode("utf-8")
        except Exception:  # noqa: BLE001 — one unreadable file shouldn't sink the rest
            continue
        kind = classify(content)
        if kind is None:
            continue
        key = f"{kind}/{f.filename}"
        prev = latest.get(key)
        if prev is None or f.created_at > prev[0]:
            latest[key] = (f.created_at, content, kind)
    return {k: {"content": c, "kind": kind} for k, (_, c, kind) in latest.items()}


def seed_baselines(client, session_id: str, baseline: dict[str, str],
                   attempts: int = 3, delay: float = 2.0) -> dict[str, str]:
    """Record each file's content as its "before", first sighting only.

    Called at the start of every turn and again the moment a prepare script
    runs. Both are needed: prepare-and-edit often happen inside a single
    turn, so a per-turn seed alone would first look after the damage is done
    — which is exactly the empty-diff bug this replaced ("(no changed
    blocks)" under a presentation of three genuinely rewritten topics).

    setdefault, never overwrite: "before" means the state this client first
    saw, so a later turn's seed must not quietly become the new baseline and
    erase an edit from the diff.

    A prepare script takes a few seconds and the tool-use event fires when
    it *starts*, so the first look can legitimately find nothing; retry a
    couple of times, then give up rather than block the stream forever.
    """
    for attempt in range(attempts):
        try:
            files = fetch_workspace_files(client, session_id)
        except Exception:  # noqa: BLE001 — a failed seed degrades the diff, never the turn
            return baseline
        for key, entry in files.items():
            baseline.setdefault(key, entry["content"])
        if files:
            break
        if attempt < attempts - 1:
            time.sleep(delay)
    return baseline


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def select(before: str, after: str, kind: str, block_ids: list[str]) -> list[dict]:
    """Blocks to show, as {id, label, before, after}.

    With no ids, shows every block whose body changed — the agent asking to
    present a file it just edited means "show my edit", not "show 12 slides
    of which 3 moved". With ids, shows exactly those, changed or not, since
    the agent may be pointing at something deliberately left alone.
    """
    after_blocks = parse_blocks(after, kind)
    before_by_id = {b["_id"]: b for b in parse_blocks(before, kind)}

    wanted = [b for b in after_blocks if b["_id"] in set(block_ids)] if block_ids else after_blocks

    out: list[dict] = []
    for b in wanted:
        old = before_by_id.get(b["_id"])
        old_body = block_body(old, kind) if old else None
        new_body = block_body(b, kind)
        new_label = block_label(b, kind)
        # Title and slide type count as changes, not just the body — a
        # retitled slide with untouched prose is a real edit, and dropping it
        # here would mean the agent presents a file and the user is shown
        # nothing while believing they've seen everything.
        changed = old is None or old_body != new_body or block_label(old, kind) != new_label
        if not block_ids and not changed:
            continue
        out.append({
            "id": b["_id"],
            "label": new_label,
            "old_label": block_label(old, kind) if old else None,
            "subtopic": b.get("Subtopic", ""),
            "before": old_body,
            "after": new_body,
        })
    return out


# ---------------------------------------------------------------------------
# Terminal rendering (chat.py)
# ---------------------------------------------------------------------------


def _wrap(text: str, indent: str) -> str:
    return "\n".join(
        textwrap.fill(line, width=96, initial_indent=indent, subsequent_indent=indent) or indent
        for line in (text or "").split("\n")
    )


def render_cli(path: str, note: str, rows: list[dict], out=None) -> None:
    """Plain ASCII on purpose.

    A Windows console defaults to cp1252, where box-drawing characters and
    even an arrow raise UnicodeEncodeError mid-write — which would crash the
    REPL while the session sits blocked waiting for the tool result this
    function is supposed to lead to. Nothing here goes outside ASCII.
    """
    out = out or sys.stdout
    print(f"\n=== {path} " + "=" * max(0, 60 - len(path)), file=out)
    if note:
        print(f"    {note}", file=out)
    if not rows:
        print("    (no changed blocks)", file=out)
    for r in rows:
        head = f"--- Block {r['id']} | {r['label']}"
        if r["subtopic"]:
            head += f"  ({r['subtopic']})"
        print(f"\n{head}", file=out)
        if r.get("old_label") and r["old_label"] != r["label"]:
            print(f"    title: {r['old_label']}  ->  {r['label']}", file=out)
        if r["before"] is None:
            print(f"    NEW ({len(r['after'].split())}w)", file=out)
            print(_wrap(r["after"], "      "), file=out)
        elif r["before"] == r["after"]:
            print(f"    body unchanged ({len(r['after'].split())}w)", file=out)
            print(_wrap(r["after"], "      "), file=out)
        else:
            print(f"    {len(r['before'].split())}w -> {len(r['after'].split())}w", file=out)
            print(_wrap(r["before"], "    - "), file=out)
            print(_wrap(r["after"], "    + "), file=out)
    print("=" * 66, file=out)
