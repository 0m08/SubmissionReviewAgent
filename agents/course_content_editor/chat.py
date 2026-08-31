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


def send_user_text(client: anthropic.Anthropic, session_id: str, text: str) -> None:
    """Stream-first send: open stream, send, drain to idle/terminated."""
    with client.beta.sessions.events.stream(session_id=session_id) as stream:
        client.beta.sessions.events.send(
            session_id=session_id,
            events=[{"type": "user.message", "content": [{"type": "text", "text": text}]}],
        )
        for event in stream:
            if event.type == "agent.message":
                for block in event.content:
                    if getattr(block, "type", None) == "text":
                        print(block.text, end="", flush=True)
                print()
            elif event.type == "agent.tool_use":
                name = getattr(event, "name", "?")
                print(f"[tool: {name}]", flush=True)
            elif event.type == "session.error":
                print(f"[session error: {getattr(event, 'message', event)}]", flush=True)
            elif event.type == "session.status_terminated":
                print("[session terminated]")
                raise SessionEnded()
            elif event.type == "session.status_idle":
                stop = getattr(event, "stop_reason", None)
                stop_type = getattr(stop, "type", None) if stop is not None else None
                if stop_type == "requires_action":
                    print("[agent is waiting on an external action]")
                    return
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
