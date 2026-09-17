"""End-to-end against the **deployed** agent, over the Agent Server API.

    python deployedtest.py --url https://<deployment>.us.langgraph.app \
                           --sheet-url "https://docs.google.com/spreadsheets/d/.../edit"

`scriptstest.py` proves the scripts work when *this file* drives them. That is a
different claim from the one that matters, which is whether the agent — given a
shell, a skill and no tools — finds the scripts and runs them itself. A build
whose instructions describe a workflow the model never adopts passes every
offline check and is still broken.

So the assertions here are about behaviour, not plumbing:

- it reaches for `execute` and the skill's scripts, rather than hunting for the
  tools the previous build had
- it presents the edit instead of retyping the content into its message
- it does not commit until asked, and there is no interrupt backstop any more —
  the restraint itself is what is under test
- when asked, the commit writes a new tab and leaves every source tab alone

Creates one tab on the sheet, reads it back, and deletes it — pass `--keep`
to leave it there. The default is a clean sheet after every run, which is
right for repeated automated runs and wrong the first time somebody wants
to see the output with their own eyes.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT = Path(__file__).resolve().parent
load_dotenv(PROJECT / ".env")
sys.path.insert(0, str(PROJECT))

from langgraph_sdk import get_client  # noqa: E402

ASSISTANT = "course-content-editor-mda"
SCRATCH_TAB = "MDA Deployed Test (delete me)"
REMOVED_TOOLS = {"open_course", "list_course_tabs", "present_files", "measure_file",
                 "commit_research_notes", "commit_slide_chunks"}
failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if ok else 'FAIL'}  {label}" + (f"  ({detail})" if detail else ""))
    if not ok:
        failures.append(label)


def sheet_handle(url: str):
    import gspread  # noqa: PLC0415

    # The sheet scripts moved into the skill when the tools were removed.
    sys.path.insert(0, str(PROJECT / "skills" / "working-with-google-sheets" / "scripts"))
    import _common  # noqa: PLC0415

    return gspread.service_account(filename=_common.resolve_sa_path()).open_by_url(url)


def tab_names(url: str) -> list[str]:
    return [w.title for w in sheet_handle(url).worksheets()]


def text_of(message: dict) -> str:
    content = message.get("content")
    if isinstance(content, list):
        return " ".join(b.get("text", "") for b in content if isinstance(b, dict))
    return str(content or "")


async def run_turn(client, thread_id: str, message: str, label: str) -> dict:
    print(f"\n--- {label} " + "-" * max(0, 52 - len(label)))
    run = await client.runs.create(
        thread_id, ASSISTANT, input={"messages": [{"role": "user", "content": message}]}
    )
    await client.runs.join(thread_id, run["run_id"])
    record = await client.runs.get(thread_id, run["run_id"])
    print(f"  run status: {record.get('status')}")
    return record


async def state_of(client, thread_id: str) -> dict:
    return await client.threads.get_state(thread_id)


def tool_calls(state: dict) -> list[dict]:
    out: list[dict] = []
    for m in (state.get("values") or {}).get("messages") or []:
        for tc in m.get("tool_calls") or []:
            out.append(tc)
    return out


def commands_run(state: dict) -> list[str]:
    """Every shell command the agent issued, across the whole thread."""
    return [str((tc.get("args") or {}).get("command") or "")
            for tc in tool_calls(state) if tc.get("name") == "execute"]


def last_text(state: dict) -> str:
    for m in reversed((state.get("values") or {}).get("messages") or []):
        if m.get("type") == "ai" and text_of(m).strip():
            return text_of(m)
    return ""


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", required=True)
    ap.add_argument("--sheet-url", required=True)
    ap.add_argument("--outline-tab", default="Final Outline")
    ap.add_argument("--keep", action="store_true",
                    help="Leave the committed tab on the sheet so you can look at it. "
                         "Off by default so repeated runs do not litter the sheet.")
    args = ap.parse_args()

    key = os.environ.get("LANGSMITH_API_KEY")
    if not key:
        print("ERROR: LANGSMITH_API_KEY not set in .env", file=sys.stderr)
        return 2

    client = get_client(url=args.url, api_key=key)
    assistants = await client.assistants.search()
    check("deployment reachable and assistant registered",
          any(a.get("graph_id") == ASSISTANT for a in assistants),
          str([a.get("graph_id") for a in assistants]))

    before_tabs = tab_names(args.sheet_url)
    print(f"tabs before: {len(before_tabs)}")

    thread = await client.threads.create()
    tid = thread["thread_id"]

    # --- turn 1: open and edit, entirely inside the deployment ---------------
    await run_turn(client, tid, (
        f"Here is a course sheet: {args.sheet_url}\n"
        f"The outline tab is '{args.outline_tab}'. Open the research notes only — no slide "
        "chunks. Then tighten the research notes for topic 1: cut the padding, but keep every "
        "specification, number and trade term. Don't write anything back to the sheet yet."
    ), "TURN 1: open + edit")

    state = await state_of(client, tid)
    cmds = commands_run(state)
    names = sorted({str(tc.get("name")) for tc in tool_calls(state)})
    print(f"  tools used: {names}")
    print(f"  shell commands: {len(cmds)}")
    for c in cmds[:8]:
        print(f"    $ {c[:150]}")

    check("the agent used the shell at all", len(cmds) > 0, str(names))
    # The point of the rebuild: it should reach for the skill's scripts.
    check("it ran a prepare script from the skill",
          any("prepare_context_workspace.py" in c or "prepare_workspace.py" in c for c in cmds),
          str(cmds[:2]))
    check("it did not go looking for the removed tools",
          not (REMOVED_TOOLS & set(names)), str(sorted(REMOVED_TOOLS & set(names))))
    check("it presented the edit rather than retyping it",
          any("present.py" in c for c in cmds),
          str([c for c in cmds if "present" in c][:2]))

    # A commit needs the user's own words, and there is no interrupt to catch a
    # mistake now. This assertion is what replaced the approval gate.
    check("it did not commit unasked",
          not any("commit_context.py" in c or "commit_workspace.py" in c for c in cmds),
          str([c for c in cmds if "commit" in c]))
    check("tabs untouched before any commit",
          len(tab_names(args.sheet_url)) == len(before_tabs))

    # --- turn 2: now ask for the write ---------------------------------------
    await run_turn(client, tid,
                   f"Good. Now commit those research notes to a new tab called '{SCRATCH_TAB}'.",
                   "TURN 2: commit when asked")

    state = await state_of(client, tid)
    cmds = commands_run(state)
    check("it ran the commit script when asked",
          any("commit_context.py" in c for c in cmds),
          str([c for c in cmds if "commit" in c][:2]))

    after_tabs = tab_names(args.sheet_url)
    made = SCRATCH_TAB in after_tabs
    check("commit created the new tab", made, f"{len(before_tabs)} -> {len(after_tabs)}")
    check("no source tab was disturbed", set(before_tabs) <= set(after_tabs),
          str(sorted(set(before_tabs) - set(after_tabs))))

    if made:
        sh = sheet_handle(args.sheet_url)
        ws = sh.worksheet(SCRATCH_TAB)
        rows = ws.get_all_values()
        source = sh.worksheet(args.outline_tab).get_all_values()
        source_cols = len(source[0])
        check("committed tab has content", len(rows) > 1, f"{len(rows)} rows")
        check("every original column survived", len(rows[0]) >= source_cols,
              f"{len(rows[0])} of {source_cols} cols")

        # The assertion that separates "the edit was committed" from "the edit
        # was silently discarded and the original re-committed". Re-running a
        # prepare script overwrites the workspace with what is still on the
        # sheet, so a lost edit produces a tab that is byte-identical to the
        # source — which satisfies every row and column check above. Turn 1
        # asked for the notes to be *tightened*, so the committed notes must be
        # materially shorter than the source's.
        def notes_words(table: list[list[str]]) -> int:
            header = [h.strip().lower() for h in table[0]]
            try:
                col = header.index("research notes")
            except ValueError:
                col = next((i for i, h in enumerate(header) if "research" in h), -1)
            if col < 0:
                return -1
            return sum(len(r[col].split()) for r in table[1:] if len(r) > col)

        before_words, after_words = notes_words(source), notes_words(rows)
        check("the research notes column was located in both tabs",
              before_words > 0 and after_words >= 0, f"{before_words} / {after_words}")
        check("the committed notes are the EDITED ones, not a re-pull of the source",
              0 <= after_words < before_words * 0.95,
              f"{before_words} -> {after_words} words")
        if args.keep:
            print(f"  scratch tab '{SCRATCH_TAB}' KEPT for inspection: {ws.url}")
        else:
            sh.del_worksheet(ws)
            print(f"  scratch tab '{SCRATCH_TAB}' deleted (pass --keep to inspect it)")

    print(f"\n  final summary: {last_text(state)[:400]}")

    print()
    if failures:
        print(f"{len(failures)} FAILED: " + ", ".join(failures))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
