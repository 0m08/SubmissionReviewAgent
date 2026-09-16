"""Assemble a trace's runs into a chronological transcript with subagents in place.

Kept separate from `render.py` so the primitives (message flattening, argument
formatting, loud truncation) stay testable without dragging in layout choices.
"""

from __future__ import annotations

from .render import (
    _dur,
    _tool_output,
    _ts,
    content_text,
    fmt_args,
    parse_messages,
    trim,
)


def build_spans(runs):
    """Split runs into the coordinator span and one span per subagent.

    A subagent is spawned by the `task` tool, and every run it performs is a
    descendant of that tool run — which `dotted_order` makes checkable by string
    prefix, with no need to walk parent pointers. This is also what makes the
    step counts trustworthy: a span is exactly the work done under one brief.
    """
    content = [r for r in runs if r.get("run_type") in ("llm", "tool")]
    content.sort(key=lambda r: (r.get("dotted_order") or "", _ts(r)))
    tasks = [r for r in content if r.get("run_type") == "tool" and r["name"] == "task"]

    spans = []
    for task in tasks:
        prefix = (task.get("dotted_order") or "") + "."
        members = [r for r in content if (r.get("dotted_order") or "").startswith(prefix)]
        args = task.get("inputs") or {}
        label = args.get("subagent_type") or args.get("name") or "subagent"
        spans.append({"task": task, "label": label, "runs": members, "prefix": prefix})

    claimed = set()
    for span in spans:
        claimed.update(id(r) for r in span["runs"])
        claimed.add(id(span["task"]))
    coordinator = [r for r in content if id(r) not in claimed]
    return coordinator, spans, content


def span_stats(runs) -> dict:
    llm = [r for r in runs if r["run_type"] == "llm"]
    tool = [r for r in runs if r["run_type"] == "tool"]
    counts = {}
    for r in tool:
        counts[r["name"]] = counts.get(r["name"], 0) + 1
    failures = sum(1 for r in tool if _tool_output(r)[1])
    return {
        "steps": len(llm),
        "tools": len(tool),
        "tool_counts": counts,
        "tokens": sum(r.get("total_tokens") or 0 for r in llm),
        "cost": sum(r.get("total_cost") or 0.0 for r in llm),
        "failures": failures,
        "seconds": round(sum(_dur(r) for r in llm) + sum(_dur(r) for r in tool), 1),
    }


def _emit_run(run, out, indent="", step=None):
    """Render one content-bearing run as transcript lines.

    `step` is the model-step ordinal within this span. It is printed rather than
    the bare word "assistant" because the question these logs exist to answer is
    how many turns a piece of work took — and because the tool calls that follow
    a step all issued from it, which is how you tell three parallel calls in one
    turn from three sequential turns.
    """
    stamp = (run.get("start_time") or "")[11:19]
    fence = "```"
    if run["run_type"] == "llm":
        gens = (run.get("outputs") or {}).get("generations") or [[]]
        text, calls = "", []
        if gens and gens[0]:
            gen = gens[0][0]
            text = gen.get("text") or ""
            msg = (gen.get("message") or {}).get("kwargs") or {}
            calls = msg.get("tool_calls") or []
            if not text:
                text = content_text(msg.get("content"))
        label = f"step {step}" if step else "assistant"
        names = ", ".join(c.get("name", "?") for c in calls if isinstance(c, dict))
        if text.strip():
            out.append(f"{indent}**[{stamp}] {label}**")
            out.append("")
            body = trim(text, max_lines=20, max_chars=2000)
            out.append(indent + body.replace("\n", "\n" + indent))
            out.append("")
        elif names:
            out.append(f"{indent}**[{stamp}] {label}** -> {names}")
            out.append("")
        else:
            out.append(f"{indent}**[{stamp}] {label}**")
            out.append("")
        return

    args = fmt_args(run.get("inputs") or {})
    text, failed = _tool_output(run)
    mark = "  **FAILED**" if failed else ""
    out.append(f"{indent}`[{stamp}] -> {run['name']}({args})`{mark}")
    body = trim(text)
    if body:
        out.append(indent + fence)
        out.append(indent + body.replace("\n", "\n" + indent))
        out.append(indent + fence)
    out.append("")


def render(trace_runs_list) -> str:
    """Full session log for one trace."""
    coordinator, spans, content = build_spans(trace_runs_list)
    if not content:
        return "_(no content-bearing runs in this trace)_\n"

    root = next((r for r in trace_runs_list if not r.get("parent_run_id")), None)
    started = min(_ts(r) for r in content)
    overall = span_stats(content)
    trace_id = content[0].get("trace_id", "?")

    # editor_id is stamped by the identity middleware and so appears on the runs
    # it wraps, not necessarily on the root — scan rather than assume.
    meta = {}
    for r in trace_runs_list:
        candidate = (r.get("extra") or {}).get("metadata") or {}
        if candidate.get("editor_id"):
            meta = candidate
            break

    out = [f"# Session {trace_id}", ""]
    out.append(f"- **Started** {started[:19].replace('T', ' ')} UTC")
    out.append(
        f"- **Editor** `{meta.get('editor_id', 'unknown')}`"
        f"  ·  **Thread** `{(root or {}).get('thread_id', '?')}`"
    )
    out.append(
        f"- **Totals** {overall['steps']} model steps · {overall['tools']} tool calls · "
        f"{overall['tokens']:,} tokens · ${overall['cost']:.2f} · {overall['seconds']}s"
    )
    if overall["failures"]:
        out.append(f"- **Tool failures** {overall['failures']} (grep no-match excluded)")
    out.append("")

    # The comparison the run tree cannot give you: sibling subagents side by side.
    if spans:
        out += ["## Subagent spans", ""]
        out.append("| # | subagent | steps | tools | failed | tokens | cost | seconds |")
        out.append("|---|---|---|---|---|---|---|---|")
        for i, span in enumerate(spans, 1):
            s = span_stats(span["runs"])
            out.append(
                f"| {i} | {span['label']} | **{s['steps']}** | {s['tools']} | {s['failures']} | "
                f"{s['tokens']:,} | ${s['cost']:.2f} | {s['seconds']} |"
            )
        out.append("")
        steps = [span_stats(s["runs"])["steps"] for s in spans]
        if len(steps) > 1 and min(steps) > 0:
            out.append(
                f"_Spread: fewest {min(steps)} steps, most {max(steps)} "
                f"({max(steps) / min(steps):.1f}x)._"
            )
            out.append("")

    # The request that *started this session* is the last human message in the
    # first model call's inputs, not the first: a thread carries its whole
    # history forward, so the earliest HumanMessage is an older turn entirely.
    first_llm = next((r for r in content if r["run_type"] == "llm"), None)
    if first_llm:
        humans = [
            m for m in parse_messages((first_llm.get("inputs") or {}).get("messages"))
            if m["role"] == "HumanMessage" and m["text"].strip()
        ]
        if humans:
            out += ["## Request", ""]
            if len(humans) > 1:
                out.append(f"_(turn {len(humans)} of this thread; {len(humans) - 1} earlier turn(s) in history)_")
                out.append("")
            quoted = trim(humans[-1]["text"], max_lines=12, max_chars=1200)
            out.append("> " + quoted.replace("\n", "\n> "))
            out.append("")

    out += ["## Transcript", ""]
    # Step ordinals restart inside each subagent, so "step 24" in span 3 means
    # that editor's 24th turn — the number you compare against its siblings.
    counters = {"": 0}
    for i, run in enumerate(content):
        order = run.get("dotted_order") or ""
        span = next((s for s in spans if order.startswith(s["prefix"])), None)
        key = span["prefix"] if span else ""
        if run["run_type"] == "llm":
            counters[key] = counters.get(key, 0) + 1
        step = counters.get(key, 0) if run["run_type"] == "llm" else None
        if span is None:
            if run.get("run_type") == "tool" and run["name"] == "task":
                own = next(s for s in spans if s["task"] is run)
                idx = spans.index(own) + 1
                brief = (run.get("inputs") or {}).get("description") or fmt_args(
                    run.get("inputs") or {}
                )
                out.append(f"### Subagent {idx} starts — {own['label']}")
                out += ["", "_Brief:_", ""]
                out.append("> " + trim(str(brief), 10, 900).replace("\n", "\n> "))
                out.append("")
            else:
                _emit_run(run, out, step=step)
            continue

        _emit_run(run, out, indent="> ", step=step)
        nxt = content[i + 1] if i + 1 < len(content) else None
        if nxt is None or not (nxt.get("dotted_order") or "").startswith(span["prefix"]):
            s = span_stats(span["runs"])
            idx = spans.index(span) + 1
            out.append(
                f"### Subagent {idx} ends — {s['steps']} steps, "
                f"{s['tools']} tool calls, ${s['cost']:.2f}"
            )
            out.append("")

    return "\n".join(out) + "\n"
