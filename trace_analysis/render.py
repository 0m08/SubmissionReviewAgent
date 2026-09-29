"""Turn a LangSmith trace into a flat, chronological, readable session log.

The LangSmith UI shows a run *tree*, which is the wrong shape for judging an
agent's behaviour: a subagent appears as a collapsed node you open separately,
so the one thing you want to see — what the whole system did, in the order it
did it — is the one thing the tree hides.

This renders the same trace as a transcript. Three rules:

1. **Chronological, with subagents in place.** `dotted_order` is LangSmith's
   canonical hierarchical-chronological key, and a child's value is prefixed by
   its parent's, so sorting by it interleaves a subagent's work exactly where it
   happened rather than in an appendix.
2. **Only content-bearing runs.** A big trace is ~1000 runs, of which ~90% are
   middleware `chain` wrappers (`SummarizationMiddleware.awrap_model_call` and
   friends). They carry no conversation and are dropped.
3. **No JSON scaffolding.** Tool-call ids, `lc`/`type`/`kwargs` envelopes and
   message-schema noise are stripped. Long values are truncated with the amount
   elided stated, so a cut is visible rather than silent.
"""

from __future__ import annotations

import json
from datetime import datetime

# Arguments whose value is the *point* of the call and must never be trimmed
# to a preview: a command you cannot read is a step you cannot judge.
VERBATIM_ARGS = {"command", "file_path", "path", "pattern", "glob", "query"}


def _msg_class(raw) -> str:
    ident = raw.get("id") or []
    return ident[-1] if ident else "Message"


def content_text(content) -> str:
    """Flatten message content to text.

    Gemini returns a list of typed blocks; Anthropic and OpenAI may return a
    bare string. Non-text blocks are named rather than dropped, so a reasoning
    or image block shows up as a marker instead of vanishing.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if block.get("type") == "text" or "text" in block:
                    parts.append(block.get("text", ""))
                else:
                    parts.append(f"[{block.get('type', 'block')}]")
        return "".join(parts)
    return str(content)


def parse_messages(raw_list):
    """Normalise LangChain's serialised message envelopes to (role, text, tool_calls)."""
    out = []
    if not raw_list:
        return out
    # inputs.messages is a list holding a single list of messages.
    if len(raw_list) == 1 and isinstance(raw_list[0], list):
        raw_list = raw_list[0]
    for raw in raw_list:
        if not isinstance(raw, dict):
            continue
        kwargs = raw.get("kwargs") or {}
        out.append({
            "role": _msg_class(raw),
            "text": content_text(kwargs.get("content")),
            "tool_calls": kwargs.get("tool_calls") or [],
            "name": kwargs.get("name"),
        })
    return out


def trim(text: str, max_lines: int = 14, max_chars: int = 1400) -> str:
    """Truncate loudly: every cut states what it removed."""
    text = (text or "").replace("\r\n", "\n").rstrip()
    if not text:
        return ""
    lines = text.split("\n")
    cut_lines = False
    if len(lines) > max_lines:
        head, tail = lines[: max_lines - 3], lines[-2:]
        hidden = len(lines) - len(head) - len(tail)
        lines = head + [f"    … {hidden} more lines …"] + tail
        cut_lines = True
    text = "\n".join(lines)
    if len(text) > max_chars and not cut_lines:
        text = text[:max_chars] + f"\n    … {len(text) - max_chars} more characters …"
    return text


def fmt_args(args) -> str:
    """One-line rendering of tool arguments.

    Long free-text arguments (a file body being written) collapse to a size,
    because their content is not what you are reading a session log to judge.
    The arguments that decide whether a call was *necessary* — the command, the
    path, the pattern — are always shown whole.
    """
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            return trim(args, max_lines=6, max_chars=400)
    if not isinstance(args, dict):
        return str(args)
    parts = []
    for key, val in args.items():
        if isinstance(val, str) and key not in VERBATIM_ARGS and len(val) > 160:
            parts.append(f"{key}=<{len(val)} chars, {val.count(chr(10)) + 1} lines>")
        else:
            rendered = val if isinstance(val, str) else json.dumps(val)
            if len(str(rendered)) > 300:
                rendered = str(rendered)[:300] + "…"
            parts.append(f"{key}={rendered}")
    return ", ".join(parts)


def _ts(run):
    return run.get("start_time") or ""


def _dur(run) -> float:
    try:
        start = datetime.fromisoformat(run["start_time"].replace("Z", "+00:00"))
        end = datetime.fromisoformat(run["end_time"].replace("Z", "+00:00"))
        return (end - start).total_seconds()
    except Exception:
        return 0.0


FAIL_MARKERS = ("Traceback", "[Command failed", "No such file", "Permission denied")


def _tool_output(run) -> tuple[str, bool]:
    """Tool result text, plus whether it looks like a failure.

    Two failure channels have to be merged here. A tool that raises comes back
    as a ToolMessage with `status == "error"`. A *script* that fails inside the
    sandbox shell does not: the shell exits cleanly having printed the error, so
    the run is `success` and the only evidence is in the text. Reading just one
    channel misses half the failures.

    `grep` exiting 1 on no-match is not a failure and is excluded deliberately —
    it is how you ask whether something is absent.
    """
    node = run.get("outputs") or {}
    status = None
    # The result is wrapped to an inconsistent depth: sometimes the serialised
    # ToolMessage itself, sometimes under `output`, sometimes under an `lc`
    # constructor envelope's `kwargs`. Unwrap until the text is in hand, and drop
    # everything alongside it (tool_call_id, additional_kwargs, response_metadata)
    # — that scaffolding is precisely the bloat these logs exist to remove.
    for _ in range(6):
        if not isinstance(node, dict):
            break
        if isinstance(node.get("status"), str):
            status = node["status"]
        for key in ("content", "output", "kwargs"):
            if key in node:
                node = node[key]
                break
        else:
            break
    text = content_text(node)
    if not isinstance(text, str):
        text = json.dumps(text, default=str)
    failed = status == "error" or any(m in text for m in FAIL_MARKERS)
    if "[Command failed with exit code 1]" in text and not text.strip().replace(
        "[Command failed with exit code 1]", ""
    ).strip():
        cmd = str((run.get("inputs") or {}).get("command", ""))
        if cmd.strip().startswith(("grep", "rg")):
            failed = False
    return text, failed
