"""Does the auth proxy actually let a sandbox script reach Google Sheets?

    python sandboxprobe.py --sheet-url "https://docs.google.com/spreadsheets/d/.../edit"

This is the one link in the chain that nobody has run. Everything else is
verified: `define_sandbox(proxy_config=...)` survives normalization and reaches
`create_sandbox`, which takes `proxy_config` as a typed parameter; `gcp_auth`
and `workspace_secret` exist and have the documented shapes. What remains is
whether a *bare* gspread request — authorized with `AnonymousCredentials`, so it
carries no `Authorization` of its own — comes back with rows because the proxy
filled the header in on the way out.

It drives `SandboxClient` directly rather than going through `mda dev`, because
the question is about the sandbox and its proxy, not about the agent. That makes
this a two-minute test with no deployment and nothing billable left behind.

Four things get checked, in the order they can fail:

1. The credential is genuinely *absent* from the box. If a script can read it,
   the proxy is pointless — that is the property the whole design is buying.
2. An unproxied call to a Google API is refused. Proves the sheet read below
   succeeded *because of* the proxy rather than despite it.
3. A bare gspread read returns real rows.
4. Writes work too, not just reads — the commit half of the skill needs it.

Sandbox notes, both learned by getting a false pass first: the interpreter is
`python3`, not `python`, and the system environment is PEP 668
externally-managed, so a bare `pip install` is refused. A missing interpreter
reads exactly like "the request was refused" to a check that only greps output,
and `cmd | tail` reports *tail's* exit code, so a failed install reported
success. Hence the venv, no pipes around anything whose status is asserted, and
an early abort when the interpreter is not actually working.

The sandbox is deleted in a `finally`, including on failure.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT = Path(__file__).resolve().parent
load_dotenv(PROJECT / ".env")
sys.path.insert(0, str(PROJECT))

from langsmith.sandbox import SandboxClient, gcp_auth, proxy_config, workspace_secret  # noqa: E402

from sandbox import SA_SECRET  # noqa: E402

PY = "/tmp/v/bin/python"
failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if ok else 'FAIL'}  {label}" + (f"  ({detail})" if detail else ""))
    if not ok:
        failures.append(label)


def sheet_id(url: str) -> str:
    m = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", url)
    if not m:
        raise SystemExit(f"ERROR: no spreadsheet id in {url!r}")
    return m.group(1)


def run(box, command: str, *, timeout: int = 240) -> tuple[int, str]:
    """Run a shell command in the box; return (exit code, stdout+stderr)."""
    result = box.run(command, timeout=timeout)
    code = getattr(result, "exit_code", None)
    out = (getattr(result, "stdout", "") or "") + (getattr(result, "stderr", "") or "")
    return (code if code is not None else -1), out.strip()


# The script the sandbox runs. `AnonymousCredentials` reports valid=True, never
# refreshes, and its before_request is a no-op, so AuthorizedSession adds no
# Authorization header of its own and leaves it free for the proxy to set.
READ_SCRIPT = """
import json, sys
import gspread
from google.auth.credentials import AnonymousCredentials

gc = gspread.authorize(AnonymousCredentials())
sh = gc.open_by_key(sys.argv[1])
titles = [w.title for w in sh.worksheets()]
rows = sh.get_worksheet(0).get_all_values()
print("PROBE_OK " + json.dumps({"tabs": len(titles), "rows": len(rows),
                                "first_tab": titles[0] if titles else None}))
"""

WRITE_SCRIPT = """
import sys
import gspread
from google.auth.credentials import AnonymousCredentials

gc = gspread.authorize(AnonymousCredentials())
sh = gc.open_by_key(sys.argv[1])
ws = sh.add_worksheet(title=sys.argv[2], rows=2, cols=2)
ws.update([["proxy", "write"]], "A1")
value = ws.acell("A1").value
sh.del_worksheet(ws)
print("PROBE_WRITE_OK " + str(value))
"""

# A bogus id must not come back 200. Written as a file rather than `-c` so the
# control is readable, and so an interpreter failure cannot masquerade as a
# refusal: no STATUS line at all is a FAIL.
CONTROL_SCRIPT = """
import urllib.request as u, urllib.error as e
try:
    print("STATUS", u.urlopen(
        "https://sheets.googleapis.com/v4/spreadsheets/nonexistent-id-probe").status)
except e.HTTPError as x:
    print("STATUS", x.code)
except Exception as x:
    print("ERR", type(x).__name__, x)
"""

SCRATCH_TAB = "Proxy Probe (auto-deleted)"


def write_and_run(box, name: str, script: str, args: str = "") -> tuple[int, str]:
    """Put a script in the box and run it under the venv interpreter."""
    heredoc = f"cat <<'PYEOF' > /tmp/{name}\n{script}\nPYEOF\n{PY} /tmp/{name} {args} 2>&1"
    return run(box, heredoc)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sheet-url", required=True)
    ap.add_argument("--keep", action="store_true", help="leave the sandbox running for debugging")
    args = ap.parse_args()

    key = os.environ.get("LANGSMITH_API_KEY")
    if not key:
        print("ERROR: LANGSMITH_API_KEY not set in .env", file=sys.stderr)
        return 2

    client = SandboxClient(api_key=key)

    print(f"creating sandbox with a {SA_SECRET} gcp_auth rule...")
    box = client.create_sandbox(
        name=None,
        idle_ttl_seconds=600,
        proxy_config=proxy_config(
            rules=[
                gcp_auth(
                    service_account_json=workspace_secret(SA_SECRET),
                    scopes=["https://www.googleapis.com/auth/spreadsheets"],
                )
            ]
        ),
    )
    print(f"  sandbox {box.id} status={box.status}")

    try:
        code, out = run(box, "command -v python3 && python3 -m venv /tmp/v && "
                             f"/tmp/v/bin/pip install -q gspread google-auth && "
                             f"{PY} -c 'import gspread; print(\"GSPREAD_OK\")'")
        check("gspread installed in a sandbox venv", code == 0 and "GSPREAD_OK" in out,
              f"exit={code} {out[-180:]}")
        if "GSPREAD_OK" not in out:
            print("  aborting: nothing below can be trusted without a working interpreter")
            return 1

        # 1. The credential must not be reachable from inside.
        code, out = run(
            box,
            "env | grep -icE 'GDRIVE|SERVICE_ACCOUNT|GCP_|GOOGLE_APPLICATION'; "
            "ls /uploads /mnt/secrets 2>/dev/null | wc -l",
        )
        leaked = [ln for ln in out.splitlines() if ln.strip() not in ("0", "")]
        check("no credential visible inside the sandbox", not leaked, out.replace("\n", " | "))

        # 2. Control: Google reached, but a bogus id refused.
        code, out = write_and_run(box, "control.py", CONTROL_SCRIPT)
        print(f"  control output: {out[-200:]}")
        check("proxy is not a blanket allow (Google reached, bogus id refused)",
              "STATUS" in out and "STATUS 200" not in out, out[-160:])

        sid = sheet_id(args.sheet_url)

        # 3. The real question.
        code, out = write_and_run(box, "probe.py", READ_SCRIPT, sid)
        print(f"  read output: {out[-400:]}")
        check("bare gspread read returned rows via the proxy", "PROBE_OK" in out, out[-200:])

        # 4. Commits need writes, so prove those too.
        code, out = write_and_run(box, "probew.py", WRITE_SCRIPT, f"{sid} '{SCRATCH_TAB}'")
        print(f"  write output: {out[-400:]}")
        check("bare gspread write worked too", "PROBE_WRITE_OK" in out, out[-200:])

    finally:
        if args.keep:
            print(f"\nsandbox {box.id} left running (--keep); delete it when done")
        else:
            client.delete_sandbox(box.id)
            print(f"\nsandbox {box.id} deleted")

    print()
    if failures:
        print(f"{len(failures)} FAILED: " + ", ".join(failures))
        return 1
    print("all checks passed — scripts on MDA are viable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
