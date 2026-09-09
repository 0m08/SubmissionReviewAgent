"""Route `present.py` output to the user's screen instead of the agent's context.

## The contract this restores

In the Anthropic Managed Agents build, `present_files` is a *client-side* tool:
the client reads the files, renders before/after cards, and answers the agent
with `ack_result(...)` — counts and an acknowledgement, no content. The agent
never sees the file text it just showed someone.

That is not a detail. Content routed through a model's context is content that
can come back paraphrased, truncated, or subtly rewritten, and a reader cannot
tell a faithful quote from a confident approximation. Sending bytes from disk
to screen without passing through the model removes the opportunity entirely.

Moving sheet access into `scripts/` lost that property by accident: a script's
stdout *is* the tool result, so `present.py` was handing the model every file it
displayed — expensive, and exactly the failure the design existed to prevent.

## How it is restored

`execute` is wrapped. When the command is `present.py`, the payload it printed
is pulled out of the result and:

- **sent to the client** on LangGraph's custom stream, which the UI renders as
  diff cards. Custom-stream data never enters message state, so it is not in the
  agent's context now and not in the thread's history later.
- **replaced, for the model,** with counts and a one-line acknowledgement.

Everything else `execute` does passes through untouched.

## Deliberate failure behaviour

If the payload cannot be parsed — the script errored, someone passed `--text`,
a future edit changes the format — the original result is returned unchanged.
A present that shows the agent too much is a bad turn; a present that silently
shows the *user* nothing is a broken review loop, and the second is worse. So
the fallback is noisy rather than quiet: the agent gets the raw output and can
see for itself that something is wrong.
"""

from __future__ import annotations

import json
from typing import Any

from langchain.agents.middleware import wrap_tool_call

# Must match `PAYLOAD_KEY` in skills/working-with-google-sheets/scripts/present.py.
PAYLOAD_KEY = "cce_present_v1"

_ACK_NOTE = (
    "Shown to the user as before/after cards. The content is on their screen; "
    "do not repeat it in your message. These counts are measured from the files "
    "— any number you state must come from here."
)


def _result_text(result: Any) -> str | None:
    """The stdout text of a tool result, whatever shape it arrives in."""
    content = getattr(result, "content", None)
    if content is None and isinstance(result, dict):
        content = result.get("content")
    if isinstance(content, list):
        return "".join(
            block.get("text", "") for block in content if isinstance(block, dict)
        )
    return content if isinstance(content, str) else None


def _extract_payload(text: str) -> dict | None:
    """The present payload from a line of stdout, or None if it is not there.

    Scans lines rather than parsing the whole blob: the sandbox appends its own
    "[Command succeeded with exit code 0]" footer, so the output is not valid
    JSON even when the script emitted valid JSON.
    """
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("{") or PAYLOAD_KEY not in line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        payload = parsed.get(PAYLOAD_KEY)
        if isinstance(payload, dict):
            return payload
    return None


def _set_content(result: Any, text: str) -> Any:
    """Return `result` carrying `text`, without assuming its concrete type."""
    if isinstance(result, dict):
        return {**result, "content": text}
    try:
        return result.model_copy(update={"content": text})
    except AttributeError:
        try:
            result.content = text
        except Exception:  # noqa: BLE001 — frozen or exotic; leave it alone
            return result
        return result


def _ack(payload: dict) -> str:
    files = payload.get("files") or []
    ack: dict[str, Any] = {
        "status": "presented" if not payload.get("missing") else "partial",
        "shown": [
            {"path": f.get("path"), "blocks_shown": len(f.get("blocks") or []),
             **(f.get("counts") or {})}
            for f in files
        ],
        "note": _ACK_NOTE,
    }
    if payload.get("missing"):
        ack["not_found"] = payload["missing"]
        ack["hint"] = "Paths are workspace-relative, e.g. context/topic_01_<slug>.md"
    return json.dumps(ack)


@wrap_tool_call
async def present_to_client(request, handler):
    """Divert `present.py` content to the UI; give the model counts only."""
    result = await handler(request)

    call = getattr(request, "tool_call", None) or {}
    if call.get("name") != "execute":
        return result
    command = str((call.get("args") or {}).get("command") or "")
    if "present.py" not in command or "--measure" in command:
        # --measure is counts-only and is deliberately *for* the agent.
        return result

    text = _result_text(result)
    if not text:
        return result
    payload = _extract_payload(text)
    if payload is None:
        # See "Deliberate failure behaviour" above: pass it through rather than
        # swallow it, so a broken present is visible instead of silent.
        return result

    try:
        from langgraph.config import get_stream_writer  # noqa: PLC0415

        writer = get_stream_writer()
        if writer is not None:
            writer({"cce_present": payload})
    except Exception:  # noqa: BLE001 — no writer outside a streaming run
        # The agent still gets a correct ack; only the live rendering is lost,
        # and the client can rebuild cards from the workspace if it needs to.
        pass

    return _set_content(result, _ack(payload))
