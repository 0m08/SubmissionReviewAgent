"""Minimal LangSmith REST client for pulling MDA traces.

Deliberately not the `langsmith` SDK: this runs from the repo's app venv, which
does not carry the agent's dependencies, and the only thing needed here is two
endpoints. The SDK would drag in a version of langchain-core that has to agree
with whatever the agent build pins, for no gain.

Rate limits are real — the workspace returns 429 readily on a long pull — so
every request goes through the same exponential backoff.
"""

from __future__ import annotations

import io
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

BASE = "https://api.smith.langchain.com"
PROJECT = "course-content-editor-mda"

# Fields worth pulling. `inputs`/`outputs` are the expensive ones and are only
# requested for the run types that carry conversation content.
LIGHT = [
    "id", "name", "run_type", "status", "error", "start_time", "end_time",
    "total_tokens", "total_cost", "trace_id", "parent_run_id", "dotted_order",
    "extra", "thread_id",
]
FULL = LIGHT + ["inputs", "outputs"]


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def api_key() -> str:
    """Read the LangSmith key from env, else the MDA .env.

    The key is never logged or echoed; only its presence is ever reported.
    """
    key = os.environ.get("LANGSMITH_API_KEY") or os.environ.get("LANGCHAIN_API_KEY")
    if key:
        return key
    path = os.path.join(_repo_root(), "agents", "course_content_editor_mda", ".env")
    if os.path.exists(path):
        for line in io.open(path, encoding="utf-8", errors="replace"):
            line = line.strip()
            if line.startswith("LANGSMITH_API_KEY=") or line.startswith("LANGCHAIN_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit(
        "No LangSmith API key. Set LANGSMITH_API_KEY, or put it in "
        "agents/course_content_editor_mda/.env"
    )


def _request(method: str, path: str, *, body=None, params=None):
    url = BASE + path + ("?" + urllib.parse.urlencode(params) if params else "")
    delay = 2.0
    for attempt in range(7):
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode() if body is not None else None,
            headers={"x-api-key": api_key(), "Content-Type": "application/json"},
            method=method,
        )
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            # 429 is routine on long pulls, not an error worth surfacing.
            if exc.code in (429, 502, 503, 504) and attempt < 6:
                time.sleep(delay)
                delay = min(delay * 2, 30)
                continue
            raise
    raise RuntimeError("retries exhausted")


def project_id(name: str = PROJECT) -> str:
    for sess in _request("GET", "/api/v1/sessions", params={"limit": 100}):
        if sess["name"] == name:
            return sess["id"]
    raise SystemExit(f"Project {name!r} not found in this workspace.")


def _page(body):
    """Yield every run for a query body, following cursors."""
    cursor = None
    while True:
        payload = dict(body)
        if cursor:
            payload["cursor"] = cursor
        res = _request("POST", "/api/v1/runs/query", body=payload)
        for run in res.get("runs", []):
            yield run
        cursor = (res.get("cursors") or {}).get("next")
        if not cursor:
            return
        time.sleep(0.4)


def list_traces(pid: str, limit: int = 40):
    """Root runs, newest first — one per session."""
    runs = list(_page({
        "session": [pid], "limit": 100, "select": LIGHT,
        "filter": "eq(is_root, true)",
    }))
    runs.sort(key=lambda r: r.get("start_time") or "", reverse=True)
    return runs[:limit]


def trace_runs(trace_id: str):
    """Every run in one trace, with full inputs/outputs on llm and tool runs.

    Pulled in two passes: the content-bearing run types with payloads, and the
    rest light. Pulling `inputs` for all 1000+ chain runs in a big trace moves
    tens of megabytes to render nothing — the chain wrappers are exactly the
    bloat this whole tool exists to strip.
    """
    heavy = list(_page({
        "trace": trace_id, "limit": 40, "select": FULL,
        "filter": 'in(run_type, ["llm", "tool"])',
    }))
    light = list(_page({
        "trace": trace_id, "limit": 100, "select": LIGHT,
        "filter": 'eq(run_type, "chain")',
    }))
    return heavy + light
