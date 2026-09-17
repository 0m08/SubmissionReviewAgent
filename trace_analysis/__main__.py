"""Pull MDA traces from LangSmith and write readable session logs.

    python -m trace_analysis list                 # recent sessions, newest first
    python -m trace_analysis pull --last 5        # write logs for the 5 newest
    python -m trace_analysis pull --trace <id>    # one specific session
    python -m trace_analysis pull --since 2026-09-01

Logs land in `trace_analysis/sessions/` as markdown, named by start time so
chronological order is also filename order — the order you want to read them in.
Raw runs are cached beside them in `.cache/` so re-rendering after a change to
the renderer costs nothing and does not re-hit a rate-limited API.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys

from . import langsmith_client as ls
from .transcript import render, span_stats, build_spans

HERE = os.path.dirname(os.path.abspath(__file__))
SESSIONS = os.path.join(HERE, "sessions")
CACHE = os.path.join(HERE, ".cache")


def _dirs(project: str) -> tuple[str, str]:
    """Where logs and cached runs go for a project.

    The default project keeps the flat layout the findings ledger cites. Any
    other project — a second deployment testing a different model, say — is
    namespaced under its own name, so a throwaway comparison never mixes into
    the production session logs or gets read by `spans`. A test run landing in
    the ledger's evidence would be worse than no test at all.
    """
    if project == ls.PROJECT:
        return SESSIONS, CACHE
    return os.path.join(SESSIONS, project), os.path.join(CACHE, project)


def _slug(run) -> str:
    stamp = (run.get("start_time") or "")[:19].replace(":", "").replace("-", "").replace("T", "-")
    return f"{stamp}_{run.get('trace_id', run.get('id', 'unknown'))[:8]}"


def cmd_list(args):
    pid = ls.project_id(args.project)
    roots = ls.list_traces(pid, limit=args.limit)
    if not roots:
        print("No traces found.")
        return
    print(f"{'started':20} {'trace_id':38} {'status':9} name")
    for r in roots:
        print(
            f"{(r.get('start_time') or '')[:19]:20} {r.get('trace_id', ''):38} "
            f"{str(r.get('status')):9} {r.get('name')}"
        )
    print(f"\n{len(roots)} sessions.")


def _load_runs(cache, trace_id, refresh=False):
    os.makedirs(cache, exist_ok=True)
    path = os.path.join(cache, f"{trace_id}.json")
    if os.path.exists(path) and not refresh:
        return json.load(io.open(path, encoding="utf-8"))
    runs = ls.trace_runs(trace_id)
    json.dump(runs, io.open(path, "w", encoding="utf-8"))
    return runs


def cmd_pull(args):
    sessions, cache = _dirs(args.project)
    pid = ls.project_id(args.project)
    if args.trace:
        targets = [{"trace_id": args.trace, "start_time": ""}]
    else:
        roots = ls.list_traces(pid, limit=max(args.last, 100))
        if args.since:
            roots = [r for r in roots if (r.get("start_time") or "") >= args.since]
        targets = roots[: args.last] if args.last else roots

    os.makedirs(sessions, exist_ok=True)
    written = []
    for root in targets:
        tid = root["trace_id"]
        try:
            runs = _load_runs(cache, tid, refresh=args.refresh)
        except Exception as exc:  # one bad trace should not sink the batch
            print(f"  !! {tid[:8]}: {exc}", file=sys.stderr)
            continue
        if not root.get("start_time"):
            starts = [r.get("start_time") or "" for r in runs if r.get("start_time")]
            root["start_time"] = min(starts) if starts else ""
        text = render(runs)
        path = os.path.join(sessions, _slug(root) + ".md")
        io.open(path, "w", encoding="utf-8").write(text)
        _, spans, content = build_spans(runs)
        s = span_stats(content)
        written.append(path)
        print(
            f"  {os.path.basename(path):34} {s['steps']:3} steps  {s['tools']:3} tools  "
            f"{len(spans)} subagents  ${s['cost']:.2f}"
        )
    print(f"\n{len(written)} session log(s) in {os.path.relpath(SESSIONS)}")


def cmd_spans(args):
    _, CACHE_DIR = _dirs(args.project)
    """Every subagent fan-out, one row per span, across all cached sessions.

    This is the view that motivates a close read: siblings get near-identical
    briefs on near-identical files, so an outlier here is the cheapest possible
    pointer to the session worth reading line by line.
    """
    rows = []
    for name in sorted(os.listdir(CACHE_DIR)) if os.path.isdir(CACHE_DIR) else []:
        if not name.endswith(".json"):
            continue
        runs = json.load(io.open(os.path.join(CACHE_DIR, name), encoding="utf-8"))
        _, spans, content = build_spans(runs)
        if not spans:
            continue
        started = min((r.get("start_time") or "") for r in content)[:19]
        stats = [span_stats(s["runs"]) for s in spans]
        steps = [s["steps"] for s in stats]
        for i, s in enumerate(stats, 1):
            flag = ""
            if len(steps) > 1 and min(steps) > 0 and s["steps"] == max(steps) and max(steps) >= 1.5 * min(steps):
                flag = "  <-- outlier"
            rows.append(
                f"{started:20} {name[:8]:9} span {i}  {s['steps']:3} steps  "
                f"{s['tools']:3} tools  {s['failures']:2} failed  ${s['cost']:.2f}{flag}"
            )
        rows.append(
            f"{'':20} {'':9} spread: {min(steps)}-{max(steps)} steps"
            + (f" ({max(steps) / min(steps):.1f}x)" if min(steps) else "")
        )
        rows.append("")
    print("\n".join(rows) if rows else "No cached sessions with subagents. Run `pull` first.")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="trace_analysis", description=__doc__)
    ap.add_argument("--project", default=ls.PROJECT)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="show recent sessions")
    p_list.add_argument("--limit", type=int, default=25)
    p_list.set_defaults(func=cmd_list)

    p_pull = sub.add_parser("pull", help="write session logs")
    p_pull.add_argument("--trace", help="one trace id")
    p_pull.add_argument("--last", type=int, default=5)
    p_pull.add_argument("--since", help="ISO date, e.g. 2026-09-01")
    p_pull.add_argument("--refresh", action="store_true", help="ignore the run cache")
    p_pull.set_defaults(func=cmd_pull)

    p_spans = sub.add_parser("spans", help="step spread across subagent fan-outs")
    p_spans.set_defaults(func=cmd_spans)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
