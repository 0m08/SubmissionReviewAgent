"""Open an interactive chat session with the Course Content Editor agent.

Per-run data-plane script — creates a SESSION against the already-deployed
agent (see deploy.py for the one-time setup), mounts the service account
credential, then drops into a REPL. Each line you type is sent as a user
message; agent output streams back until the session is idle, then you're
prompted again.

Usage:
    # Free-form chat:
    python agents/course_content_editor/chat.py

    # Shortcut: send an "open this sheet" kickoff as the first message:
    python agents/course_content_editor/chat.py \\
        --sheet-url "https://docs.google.com/spreadsheets/d/.../edit"

    # Override the service account file id:
    python agents/course_content_editor/chat.py --sa-file-id file_011C...

Commands inside the REPL:
    /quit  or  Ctrl-D    end the session and exit

--sa-file-id defaults to state['sa_file_id'] from deploy.py, or the
COURSE_CONTENT_EDITOR_SA_FILE_ID env var.

Expects ANTHROPIC_API_KEY in the environment (or a repo-root .env).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

try:
    from dotenv import load_dotenv

    _here = Path(__file__).resolve()
    for _parent in _here.parents:
        if (_parent / ".env").exists():
            load_dotenv(_parent / ".env")
            break
except ImportError:
    pass

import anthropic

import _presentation as pres

# Each workspace file as this process first saw it. The Files API exposes
# only current content, so a "before" that isn't captured while it's still
# current is gone for good — hence seeding rather than reading it back later.
_BASELINE: dict[str, str] = {}

STATE_PATH = Path(__file__).parent / "deploy_state.json"
SA_MOUNT_PATH = "/uploads/service_account.json"


def load_state() -> dict:
    if not STATE_PATH.exists():
        print(
            f"ERROR: {STATE_PATH.name} not found. Run deploy.py first to create "
            "the agent and environment.",
            file=sys.stderr,
        )
        sys.exit(2)
    return json.loads(STATE_PATH.read_text(encoding="utf-8"))


def handle_present_files(client: anthropic.Anthropic, session_id: str, event) -> None:
    """Render a `present_files` call, then unblock the turn.

    The turn is *stopped* until the result goes back, so every path out of
    this function must send one — including the failure paths. A raised
    exception here would strand the session waiting forever on a client
    that has moved on, which is worse than showing nothing.
    """
    items = pres.normalize_items(getattr(event, "input", None))
    result_text = None
    try:
        if not items:
            result_text = json.dumps({"status": "failed", "reason": "no items in input"})
        else:
            files = pres.fetch_workspace_files(client, session_id)
            # A file written moments ago can take a beat to appear in the
            # Files API. One retry, rather than telling the agent a file it
            # just edited doesn't exist.
            if any(pres.resolve_key(files, i["path"]) is None for i in items):
                time.sleep(2.0)
                files = pres.fetch_workspace_files(client, session_id)

            missing: list[str] = []
            facts: dict[str, dict] = {}
            for item in items:
                key = pres.resolve_key(files, item["path"])
                if key is None:
                    missing.append(item["path"])
                    print(f"\n[present_files: no workspace file matching {item['path']!r}]")
                    continue
                entry = files[key]
                # baseline = the file as prepare_workspace.py first wrote it,
                # taken from the Files API's own version history rather than
                # from when this process first looked — see fetch_workspace_files.
                baseline = _BASELINE.get(key, entry["content"])
                rows = pres.select(baseline, entry["content"], entry["kind"], item["block_ids"])
                pres.render_cli(key, item["note"], rows)
                # Keyed by the path the AGENT used, since that is what the
                # result echoes back to it — not the resolved cache key.
                facts[item["path"]] = pres.measure(baseline, entry["content"], entry["kind"])

            result_text = (
                pres.unresolved_result(items, missing, facts) if missing
                else pres.ack_result(items, facts)
            )
    except Exception as exc:  # noqa: BLE001 — see docstring: always answer.
        print(f"\n[present_files failed locally: {exc}]")
        result_text = json.dumps({"status": "failed", "reason": str(exc)})

    client.beta.sessions.events.send(
        session_id=session_id,
        events=[{
            "type": "user.custom_tool_result",
            "custom_tool_use_id": event.id,
            "content": [{"type": "text", "text": result_text}],
        }],
    )


def send_user_text(client: anthropic.Anthropic, session_id: str, text: str) -> None:
    """Stream-first send: open stream, send, drain to idle/terminated."""
    # Before anything in this turn can edit a file, record what the files
    # look like now — a diff needs a "before" and the Files API won't hand
    # one back after the fact.
    pres.seed_baselines(client, session_id, _BASELINE, attempts=1)
    # Set when a prepare script is launched, acted on when the agent next
    # does anything — see the comment at the deferred seed below.
    pending_seed = False
    with client.beta.sessions.events.stream(session_id=session_id) as stream:
        client.beta.sessions.events.send(
            session_id=session_id,
            events=[{"type": "user.message", "content": [{"type": "text", "text": text}]}],
        )
        for event in stream:
            # Deferred baseline seed. A tool_use event fires when the command
            # *starts*, and prepare_workspace.py takes longer to pull a sheet
            # than any sane retry window — seeding on the event itself found
            # an empty workspace every time and produced empty diffs. The
            # agent acting again is proof the script returned.
            if pending_seed and event.type in ("agent.message", "agent.tool_use"):
                pres.seed_baselines(client, session_id, _BASELINE)
                pending_seed = False

            if event.type == "agent.message":
                for block in event.content:
                    if getattr(block, "type", None) == "text":
                        print(block.text, end="", flush=True)
                print()
            elif event.type == "agent.tool_use":
                name = getattr(event, "name", "?")
                print(f"[tool: {name}]", flush=True)
                # prepare_workspace.py just laid down the pristine workspace.
                # This is the only moment its content is still the current
                # content, so it is the only moment a baseline can be taken —
                # and prepare + edit routinely happen in one turn.
                inp = getattr(event, "input", None)
                cmd = inp.get("command", "") if isinstance(inp, dict) else ""
                if pres.PREPARE_CMD.search(str(cmd)):
                    pending_seed = True
            elif event.type == "agent.custom_tool_use":
                if getattr(event, "name", None) == pres.TOOL_NAME:
                    handle_present_files(client, session_id, event)
                else:
                    # Unknown custom tool — still has to be answered, or the
                    # session sits idle-requires_action forever.
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
            elif event.type == "session.error":
                print(f"[session error: {getattr(event, 'message', event)}]", flush=True)
            elif event.type == "session.status_terminated":
                print("[session terminated]")
                raise SessionEnded()
            elif event.type == "session.status_idle":
                stop = getattr(event, "stop_reason", None)
                stop_type = getattr(stop, "type", None) if stop is not None else None
                if stop_type == "requires_action":
                    # NOT the end of the turn — the agent is blocked on a
                    # client-side event (a present_files call, handled above)
                    # and will carry on once the result lands. Returning here
                    # would hand the prompt back mid-turn and strand it.
                    continue
                return


class SessionEnded(Exception):
    pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sheet-url", default=None,
                        help="Optional. If set, send an 'open this sheet' kickoff as the first message.")
    parser.add_argument("--sa-file-id", default=None,
                        help="Override the service account file id.")
    args = parser.parse_args()

    state = load_state()
    sa_file_id = (
        args.sa_file_id
        or state.get("sa_file_id")
        or os.environ.get("COURSE_CONTENT_EDITOR_SA_FILE_ID")
    )
    if not sa_file_id:
        print(
            "ERROR: no service account file id. Set GDRIVE_SA_B64 and re-run "
            "deploy.py, or pass --sa-file-id.",
            file=sys.stderr,
        )
        return 2

    client = anthropic.Anthropic()

    session = client.beta.sessions.create(
        agent=state["agent_id"],
        environment_id=state["environment_id"],
        title="Course content editor chat",
        resources=[
            {"type": "file", "file_id": sa_file_id, "mount_path": SA_MOUNT_PATH},
        ],
    )

    print(f"Session: {session.id}")
    print(f"Watch:   https://platform.claude.com/workspaces/default/sessions/{session.id}")
    print("Type a message and press Enter. Type /quit or hit Ctrl-D to exit.\n")

    first_turn = (
        f"Here is a course sheet to edit: {args.sheet_url}"
        if args.sheet_url
        else None
    )

    try:
        while True:
            if first_turn is not None:
                user_text = first_turn
                first_turn = None
                print(f"> {user_text}\n")
            else:
                try:
                    user_text = input("> ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    break
                if not user_text:
                    continue
                if user_text in {"/quit", "/exit"}:
                    break
                print()

            send_user_text(client, session.id, user_text)
            print()

    except SessionEnded:
        pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
