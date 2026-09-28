"""Does a run survive the client going away? (e.g. a Streamlit session timing out)

    python disconnecttest.py --url https://<deployment>.us.langgraph.app \
                             --sheet-url "https://docs.google.com/spreadsheets/d/.../edit"

`deployedtest.py` drives the deployment with `runs.create()` followed by
`runs.join()`, which looks like a blocking call and proves nothing about who is
doing the work. This asks the question that a hosted Streamlit page actually
depends on: if the browser tab closes mid-edit, does the agent keep going, and
is the thread still there to re-attach to?

The dispatch happens in a **subprocess that exits** before the run finishes, so
"the client disconnected" is real rather than simulated by dropping a variable.
Everything after that runs through a client constructed from scratch.

Two properties, because they fail separately:

1. A run in flight completes server-side with no client attached.
2. A run parked on an approval interrupt stays parked, and a *later* client can
   still resume it.

Read-only against the sheet: it opens a course and edits the workspace, then
rejects the commit. No tab is created.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

PROJECT = Path(__file__).resolve().parent
load_dotenv(PROJECT / ".env")
sys.path.insert(0, str(PROJECT))

from langgraph_sdk import get_client  # noqa: E402

ASSISTANT = "course-content-editor-mda"
TERMINAL = {"success", "error", "timeout", "interrupted"}
failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if ok else 'FAIL'}  {label}" + (f"  ({detail})" if detail else ""))
    if not ok:
        failures.append(label)


# --------------------------------------------------------------------------
# The dispatching subprocess
# --------------------------------------------------------------------------

# Runs as `python disconnecttest.py --dispatch`, prints one JSON line, exits. Its
# exit is the disconnect: no join, no polling, no open socket left behind.
DISPATCH_FLAG = "--dispatch"


async def dispatch(url: str, key: str, message: str, thread_id: str | None) -> None:
    client = get_client(url=url, api_key=key)
    tid = thread_id or (await client.threads.create())["thread_id"]
    run = await client.runs.create(
        tid, ASSISTANT, input={"messages": [{"role": "user", "content": message}]}
    )
    print(json.dumps({"thread_id": tid, "run_id": run["run_id"]}))


def dispatch_and_die(url: str, key: str, message: str, thread_id: str | None = None) -> dict:
    """Fire a run from a process that then exits. Returns {thread_id, run_id}."""
    cmd = [sys.executable, str(Path(__file__).resolve()), DISPATCH_FLAG,
           "--url", url, "--message", message]
    if thread_id:
        cmd += ["--thread-id", thread_id]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if out.returncode != 0:
        raise RuntimeError(f"dispatch subprocess failed: {out.stderr.strip()[:400]}")
    return json.loads(out.stdout.strip().splitlines()[-1])


# --------------------------------------------------------------------------
# Re-attachment, always through a freshly built client
# --------------------------------------------------------------------------


async def await_terminal(url: str, key: str, tid: str, rid: str,
                         *, timeout: float = 900.0) -> tuple[str, float]:
    """Poll a run to a terminal status with a client built after the dispatcher died."""
    client = get_client(url=url, api_key=key)
    started = time.monotonic()
    while time.monotonic() - started < timeout:
        status = (await client.runs.get(tid, rid)).get("status")
        if status in TERMINAL:
            return str(status), time.monotonic() - started
        await asyncio.sleep(5)
    return "TIMED-OUT-LOCALLY", time.monotonic() - started


async def state_of(url: str, key: str, tid: str) -> dict:
    return await get_client(url=url, api_key=key).threads.get_state(tid)


def workspace_files(state: dict) -> list[str]:
    files = (state.get("values") or {}).get("files") or {}
    return [p for p in files if str(p).startswith("/workspace/")]


def interrupts_of(state: dict) -> list[dict]:
    return [it for task in (state.get("tasks") or []) for it in (task.get("interrupts") or [])]


def tool_names(state: dict) -> list[str]:
    out: list[str] = []
    for m in (state.get("values") or {}).get("messages") or []:
        for tc in m.get("tool_calls") or []:
            out.append(str(tc.get("name")))
    return out


def tab_names(url: str) -> list[str]:
    import gspread  # noqa: PLC0415

    # The sheet scripts moved into the skill when the tools were removed.
    sys.path.insert(0, str(PROJECT / "skills" / "working-with-google-sheets" / "scripts"))
    import _common  # noqa: PLC0415

    return [w.title for w in
            gspread.service_account(filename=_common.resolve_sa_path()).open_by_url(url).worksheets()]


# --------------------------------------------------------------------------


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", required=True)
    ap.add_argument("--sheet-url")
    ap.add_argument(DISPATCH_FLAG, action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--message", help=argparse.SUPPRESS)
    ap.add_argument("--thread-id", help=argparse.SUPPRESS)
    args = ap.parse_args()

    key = os.environ.get("LANGSMITH_API_KEY")
    if not key:
        print("ERROR: LANGSMITH_API_KEY not set in .env", file=sys.stderr)
        return 2

    if args.dispatch:
        await dispatch(args.url, key, args.message or "", args.thread_id)
        return 0

    if not args.sheet_url:
        print("ERROR: --sheet-url is required", file=sys.stderr)
        return 2

    before_tabs = tab_names(args.sheet_url)
    print(f"tabs before: {len(before_tabs)}")

    # --- 1. Dispatch real work, then die -----------------------------------
    print("\n--- PHASE 1: dispatch from a process that exits " + "-" * 12)
    handle = dispatch_and_die(args.url, key, (
        f"Open the research notes for this course: {args.sheet_url}\n"
        "Call list_course_tabs first to find the real tab names, then open it "
        "research-notes-only. Once it is open, tighten the research notes for the "
        "first two topics: cut the padding and keep every specification, number and "
        "trade term. Do not write anything back to the sheet yet."
    ))
    tid, rid = handle["thread_id"], handle["run_id"]
    print(f"  dispatcher exited; thread={tid} run={rid[:8]}...")
    check("dispatcher process exited while the run was still live", True)

    status, secs = await await_terminal(args.url, key, tid, rid)
    print(f"  run reached {status!r} after {secs:.0f}s with no client attached")
    check("run completed server-side after the client vanished", status == "success", status)

    state = await state_of(args.url, key, tid)
    files = workspace_files(state)
    check("workspace survived in thread state", len(files) > 0, f"{len(files)} files")
    called = tool_names(state)
    check("the agent really did the work", "open_course" in called, str(sorted(set(called))))

    # --- 2. Park on an interrupt, then die ---------------------------------
    print("\n--- PHASE 2: abandon a run parked on the approval gate " + "-" * 6)
    handle2 = dispatch_and_die(args.url, key,
                               "Commit those research notes to the sheet now.", thread_id=tid)
    rid2 = handle2["run_id"]
    status2, secs2 = await await_terminal(args.url, key, tid, rid2)
    print(f"  run reached {status2!r} after {secs2:.0f}s")
    state2 = await state_of(args.url, key, tid)
    pending = interrupts_of(state2)
    # A run halted on an interrupt reports "success", not "interrupted": the run
    # did finish, the *graph* is what is paused. So the run status proves nothing
    # here and the pending interrupt in thread state is the whole assertion.
    check("run ended without committing unattended",
          status2 in {"success", "interrupted"} and len(pending) > 0,
          f"{status2}, {len(pending)} pending")
    check("interrupt still pending for a later client", len(pending) > 0, f"{len(pending)} pending")

    # --- 3. A brand-new client resumes it ----------------------------------
    print("\n--- PHASE 3: reject the commit from a fresh client " + "-" * 10)
    client = get_client(url=args.url, api_key=key)
    run3 = await client.runs.create(tid, ASSISTANT, command={"resume": {"decisions": [{"type": "reject"}]}})
    await client.runs.join(tid, run3["run_id"])
    final = await client.runs.get(tid, run3["run_id"])
    check("a client that never saw the original run resumed it",
          final.get("status") == "success", str(final.get("status")))

    after_tabs = tab_names(args.sheet_url)
    check("rejected commit wrote nothing",
          len(after_tabs) == len(before_tabs), f"{len(before_tabs)} -> {len(after_tabs)}")

    print()
    if failures:
        print(f"{len(failures)} FAILED: " + ", ".join(failures))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
