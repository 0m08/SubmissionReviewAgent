"""Drive the compiled MDA agent with a real model against a real course sheet.

    python livetest.py --sheet-url "https://docs.google.com/spreadsheets/d/.../edit"

Costs tokens and touches the sheet. What it touches: it creates one new tab,
reads it back, and deletes it. The commit tools never modify a source tab.

Three scripted turns, because the interesting behaviour is in the seams:

1. open the course and tighten one topic's research notes;
2. ask to commit, **reject** the approval — nothing may reach the sheet;
3. ask to commit, **approve** — the tab must appear with the edited text in it.

Run `mda build .` first; this imports the compiled graph out of `.mda/build`.

One harness-only liberty: the managed runtime leaves the checkpointer to the
LangGraph server, so a graph imported directly has none and cannot interrupt.
An `InMemorySaver` is attached here to stand in for it. Everything else — the
tools, the middleware, the permissions, the skills, the instructions — is
exactly what `mda deploy` would run.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PROJECT = Path(__file__).resolve().parent
BUILD = PROJECT / ".mda" / "build"
REPO = PROJECT.parent.parent

load_dotenv(REPO / ".env")
load_dotenv(PROJECT / ".env", override=True)

sys.path.insert(0, str(BUILD))
sys.path.insert(0, str(BUILD / "__runtime__"))
sys.path.insert(0, str(PROJECT))

from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.types import Command  # noqa: E402

SCRATCH_TAB = "MDA Smoke Test (delete me)"
failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if ok else 'FAIL'}  {label}" + (f"  ({detail})" if detail else ""))
    if not ok:
        failures.append(label)


def sheet_handle(url: str):
    # The sheet scripts moved into the skill when the tools were removed.
    sys.path.insert(0, str(PROJECT / "skills" / "working-with-google-sheets" / "scripts"))
    import _common  # noqa: PLC0415
    import gspread  # noqa: PLC0415

    return gspread.service_account(filename=_common.resolve_sa_path()).open_by_url(url)


def tab_names(url: str) -> list[str]:
    return [w.title for w in sheet_handle(url).worksheets()]


class Run:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple, str, dict]] = []
        self.text: list[str] = []

    def names(self) -> list[str]:
        return [n for _, n, _ in self.calls]


async def stream(graph, config, payload, label: str, run: Run) -> None:
    # MDA drives the graph with `astream`, and this build's middleware is async
    # only — the same code run through sync `stream()` raises NotImplementedError.
    # Testing through the sync path would be testing something the deployment
    # never does.
    print(f"\n--- {label} " + "-" * (56 - len(label)))
    async for chunk in graph.astream(payload, config=config, stream_mode=["updates"],
                                     subgraphs=True, version="v2"):
        if not isinstance(chunk, dict) or chunk.get("type") != "updates":
            continue
        ns, data = chunk.get("ns") or (), chunk.get("data")
        if not isinstance(data, dict):
            continue
        for update in data.values():
            if not isinstance(update, dict):
                continue
            for m in update.get("messages") or []:
                for c in getattr(m, "tool_calls", None) or []:
                    if isinstance(c, dict):
                        args = dict(c.get("args") or {})
                        run.calls.append((ns, str(c.get("name")), args))
                        hint = str(args.get("subagent_type") or args.get("target_tab")
                                   or args.get("sheet_url") or "")[:70]
                        print(f"  [tool{'/' + ns[-1] if ns else ''}: {c.get('name')}] {hint}")
                content = getattr(m, "content", "")
                if isinstance(content, str) and content.strip() and getattr(m, "type", None) == "ai":
                    run.text.append(content)
                    print(f"  > {content.strip()[:300]}")


async def pending(graph, config) -> list:
    state = await graph.aget_state(config)
    out = list(getattr(state, "interrupts", None) or [])
    if out:
        return out
    for task in getattr(state, "tasks", None) or []:
        out.extend(getattr(task, "interrupts", None) or [])
    return out


async def turn(graph, config, text: str, label: str, decision: str, run: Run) -> list[str]:
    """One turn, answering every approval request with `decision`."""
    payload: Any = {"messages": [{"role": "user", "content": text}]}
    gated: list[str] = []
    while True:
        await stream(graph, config, payload, label, run)
        ints = await pending(graph, config)
        if not ints:
            return gated
        value = getattr(ints[0], "value", ints[0]) or {}
        requests = value.get("action_requests") or []
        for a in requests:
            gated.append(str(a.get("name")))
            print(f"  !! APPROVAL REQUESTED: {a.get('name')} args={json.dumps(a.get('args', {}))[:100]}")
        print(f"  !! answering: {decision}")
        decisions = [
            {"type": "approve"} if decision == "approve"
            else {"type": "reject", "message": "The user did not approve this. Do not retry it. "
                                               "Say the workspace is still uncommitted."}
            for _ in requests
        ]
        payload = Command(resume={"decisions": decisions})


async def files_of(graph, config) -> dict[str, str]:
    values = (await graph.aget_state(config)).values or {}
    return {p: (e.get("content", "") if isinstance(e, dict) else str(e))
            for p, e in (values.get("files") or {}).items()}


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sheet-url", required=True)
    ap.add_argument("--outline-tab", default="Final Outline")
    ap.add_argument("--keep-tab", action="store_true", help="Don't delete the scratch tab.")
    ap.add_argument("--all-topics", action="store_true",
                    help="Edit every topic instead of the first, which should make the "
                         "coordinator delegate one editor subagent per topic.")
    args = ap.parse_args()

    if not BUILD.exists():
        print("ERROR: .mda/build missing — run `mda build .` first.", file=sys.stderr)
        return 2

    from _mda_entry import agent as build_graph  # noqa: PLC0415

    config = {"configurable": {"thread_id": "mda-livetest"}, "recursion_limit": 150}
    graph = build_graph(config)
    graph.checkpointer = InMemorySaver()  # see module docstring

    import agent as agent_module  # noqa: PLC0415
    print(f"model: {agent_module.MODEL}")

    before_tabs = tab_names(args.sheet_url)
    print(f"tabs before: {len(before_tabs)}")

    run = Run()
    t0 = time.monotonic()

    scope = "every topic" if args.all_topics else "just the FIRST topic"
    await turn(graph, config,
               f"Here is a course sheet: {args.sheet_url}\n"
               f"The outline tab is called '{args.outline_tab}'. I only want to touch research "
               f"notes - nothing to do with slide chunks. Open it, then take {scope} "
               "and tighten its research notes to their learning objectives. Do all of that "
               "now, in this turn. Do not write anything back to the sheet yet.",
               "TURN 1: open + edit", "reject", run)

    files = await files_of(graph, config)
    workspace = {p: c for p, c in files.items()
                 if p.startswith("/workspace/") and not p.startswith("/workspace/.baseline")}
    baseline = {p: c for p, c in files.items() if p.startswith("/workspace/.baseline")}
    context_files = [p for p in workspace if "/context/" in p]

    print()
    check("open_course ran", "open_course" in run.names(), str(sorted(set(run.names()))))
    check("manifest written to the workspace", "/workspace/manifest.json" in workspace)
    check("research notes pulled into the workspace", len(context_files) >= 1,
          f"{len(context_files)} context file(s)")
    check("baseline snapshot taken alongside", len(baseline) >= 1, f"{len(baseline)} file(s)")
    check("no slide-chunks files (research-notes-only path)",
          not any("/topics/" in p for p in workspace),
          str([p for p in workspace if "/topics/" in p]))

    edited = [p for p in context_files
              if workspace[p] != baseline.get(f"/workspace/.baseline/{p[len('/workspace/'):]}")]
    # Per-file, so "2 of 3 edited" names which one and by how much rather than
    # leaving it to be guessed at.
    from tools import presentation as _pres  # noqa: PLC0415
    print("  per-file:")
    for p_ in sorted(context_files):
        b = baseline.get(f"/workspace/.baseline/{p_[len('/workspace/'):]}", "")
        f_ = _pres.measure(b, workspace[p_], "context")
        print(f"    {p_.rsplit('/', 1)[-1]:44} "
              f"{f_['words']['before']:>5} -> {f_['words']['after']:<5} words, "
              f"{f_['blocks']['before']} -> {f_['blocks']['after']} blocks"
              + ("" if workspace[p_] != b else "   [UNCHANGED]"))
    check("at least one note file actually changed", len(edited) >= 1, f"{len(edited)} edited")
    if edited:
        from tools import presentation as pres  # noqa: PLC0415
        p = edited[0]
        facts = pres.measure(baseline[f"/workspace/.baseline/{p[len('/workspace/'):]}"],
                             workspace[p], "context")
        check("notes got shorter, as asked",
              facts["words"]["after"] < facts["words"]["before"], str(facts["words"]))
        check("no LO block was dropped",
              facts["blocks"]["before"] == facts["blocks"]["after"], str(facts["blocks"]))
    check("present_files called without being asked", "present_files" in run.names())

    if args.all_topics:
        # The instructions say 2+ topic files is one editor subagent per topic.
        tasks = [a for ns, n, a in run.calls if n == "task" and not ns]
        editors = [a for a in tasks if a.get("subagent_type") == "editor"]
        check("delegated to the editor subagent", len(editors) >= 1,
              f"{len(tasks)} task call(s), {len(editors)} to 'editor'")
        check("one editor per topic", len(editors) >= len(context_files),
              f"{len(editors)} editor(s) for {len(context_files)} topics")
        check("every topic file was edited", len(edited) == len(context_files),
              f"{len(edited)}/{len(context_files)} edited")
        nested = sum(1 for ns, _, _ in run.calls if ns)
        check("subagent work happened in a nested namespace", nested > 0,
              f"{nested} namespaced tool call(s)")
        # An editor that reads its file and never edits it has under-worked.
        # The coordinator quietly picking up the slack still produces edited
        # files, so this is checked separately from the file diff.
        editing_editors = {ns for ns, name, _ in run.calls if ns and name == "edit_file"}
        check("every editor edited its own file", len(editing_editors) >= len(editors),
              f"{len(editing_editors)} of {len(editors)} editors called edit_file")

    gated = await turn(graph, config,
                 f"Now commit that to the sheet, in a new tab called '{SCRATCH_TAB}'.",
                 "TURN 2: commit, REJECTED", "reject", run)
    mid_tabs = tab_names(args.sheet_url)
    print()
    check("commit raised an approval request", "commit_research_notes" in gated, str(gated))
    check("rejected commit wrote nothing", len(mid_tabs) == len(before_tabs)
          and SCRATCH_TAB not in mid_tabs, f"{len(before_tabs)} -> {len(mid_tabs)} tabs")

    gated = await turn(graph, config,
                 f"I've changed my mind — go ahead and commit it to '{SCRATCH_TAB}' now.",
                 "TURN 3: commit, APPROVED", "approve", run)
    after_tabs = tab_names(args.sheet_url)
    new_tabs = [t for t in after_tabs if t not in before_tabs]
    print()
    check("commit raised an approval request again", "commit_research_notes" in gated, str(gated))
    check("approved commit created exactly one new tab", len(new_tabs) == 1, str(new_tabs))

    if new_tabs:
        sh = sheet_handle(args.sheet_url)
        ws = sh.worksheet(new_tabs[0])
        rows = ws.get_all_records()
        check("committed tab has rows and all original columns",
              bool(rows) and len(rows[0]) > 4, f"{len(rows)} rows, {len(rows[0]) if rows else 0} cols")
        # The edited text must actually be in the sheet, not just claimed.
        from tools import presentation as pres  # noqa: PLC0415
        edited_blocks = pres.parse_blocks(workspace[edited[0]], "context") if edited else []
        landed = 0
        for b in edited_blocks:
            body = (b.get("Research Notes") or "").strip()
            if body and any(body[:60] in str(r.get("research_notes", "")) for r in rows):
                landed += 1
        # A vacuous pass is worse than a failure: with nothing edited there is
        # nothing to find, and "0/0 found" must not read as success.
        check("edited note text is present in the committed tab",
              bool(edited_blocks) and landed >= 1,
              f"{landed}/{len(edited_blocks)} edited blocks found"
              + ("  [nothing was edited, so this cannot pass]" if not edited_blocks else ""))
        untouched = [r for r in rows if str(r.get("research_notes", "")).strip()]
        check("other rows kept their notes", len(untouched) >= len(rows) - len(edited_blocks),
              f"{len(untouched)}/{len(rows)} rows still have notes")

        if args.keep_tab:
            print(f"  kept scratch tab: {new_tabs[0]}")
        else:
            sh.del_worksheet(ws)
            print(f"  deleted scratch tab: {new_tabs[0]}")

    print(f"\nelapsed {time.monotonic() - t0:.0f}s, {len(run.calls)} tool calls")
    print("tools used:", ", ".join(sorted(set(run.names()))))
    if failures:
        print(f"\n{len(failures)} FAILED: " + ", ".join(failures))
        return 1
    print("\nall live checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
