"""The real skill scripts, in a real sandbox, against a real sheet.

    python scriptstest.py --sheet-url "https://docs.google.com/spreadsheets/d/.../edit"

`sandboxprobe.py` proved the auth proxy fills in the header for a two-line
gspread call. That is necessary but not sufficient: it says nothing about
whether *these* scripts — the ones the skill actually ships, shared byte for
byte with the Deep Agents build — work when the credential they were written
around is absent.

So this runs the workflow end to end the way the agent will:

    list_tabs -> prepare_context_workspace -> present -> commit_context

and then deletes the tab it made. It uploads the scripts rather than baking an
image, so it tests the current working tree, not the last deploy.

Nothing is left behind: the scratch tab is removed, and the sandbox is deleted
in a `finally` even when a check fails.
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

SCRIPTS = PROJECT / "skills" / "working-with-google-sheets" / "scripts"
REMOTE = "/skills/working-with-google-sheets/scripts"
PY = "/tmp/v/bin/python"
SCRATCH_TAB = "Scripts Test (auto-deleted)"

EDIT_SCRIPT = """import os
d = 'workspace/context'
p = os.path.join(d, sorted(os.listdir(d))[0])
s = open(p, encoding='utf-8').read()
open(p, 'w', encoding='utf-8').write(s.replace(' the ', ' ', 40))
print('EDITED', p)"""

failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if ok else 'FAIL'}  {label}" + (f"  ({detail})" if detail else ""))
    if not ok:
        failures.append(label)


def run(box, command: str, *, timeout: int = 300) -> tuple[int, str]:
    result = box.run(command, timeout=timeout)
    code = getattr(result, "exit_code", None)
    out = (getattr(result, "stdout", "") or "") + (getattr(result, "stderr", "") or "")
    return (code if code is not None else -1), out.strip()


def tab_names(sheet_url: str) -> list[str]:
    """Read tabs from *here*, using the local key — an independent observer.

    Deliberately not asked of the sandbox: a test that checks its own work
    through the same path it just exercised cannot tell "the commit worked" from
    "the read is broken in a way that agrees with it".
    """
    import gspread  # noqa: PLC0415

    sys.path.insert(0, str(SCRIPTS))
    import _common  # noqa: PLC0415

    path = _common.resolve_sa_path()
    return [w.title for w in gspread.service_account(filename=path).open_by_url(sheet_url).worksheets()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sheet-url", required=True)
    ap.add_argument("--outline-tab", default="Final Outline")
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    key = os.environ.get("LANGSMITH_API_KEY")
    if not key:
        print("ERROR: LANGSMITH_API_KEY not set in .env", file=sys.stderr)
        return 2

    before_tabs = tab_names(args.sheet_url)
    print(f"tabs before: {len(before_tabs)}")

    client = SandboxClient(api_key=key)
    box = client.create_sandbox(
        name=None,
        idle_ttl_seconds=600,
        proxy_config=proxy_config(rules=[
            gcp_auth(service_account_json=workspace_secret(SA_SECRET),
                     scopes=["https://www.googleapis.com/auth/spreadsheets"])
        ]),
    )
    print(f"sandbox {box.id} status={box.status}")

    try:
        code, out = run(box, "python3 -m venv /tmp/v && "
                             f"/tmp/v/bin/pip install -q gspread google-auth pandas && "
                             f"{PY} -c 'import gspread, pandas; print(\"DEPS_OK\")'")
        check("sandbox deps installed", code == 0 and "DEPS_OK" in out, out[-180:])
        if "DEPS_OK" not in out:
            return 1

        run(box, f"mkdir -p {REMOTE}")
        uploaded = 0
        for script in sorted(SCRIPTS.glob("*.py")):
            box.write(f"{REMOTE}/{script.name}", script.read_text(encoding="utf-8"))
            uploaded += 1
        check("skill scripts uploaded", uploaded >= 9, f"{uploaded} files")

        # 1. list_tabs — the cheapest real call through the proxy.
        code, out = run(box, f"{PY} {REMOTE}/list_tabs.py --sheet-url '{args.sheet_url}' 2>&1")
        check("list_tabs.py reached the sheet", code == 0 and args.outline_tab in out,
              out[-200:])

        # 2. prepare — pulls the course into a workspace and snapshots a baseline.
        code, out = run(box, f"cd /tmp && {PY} {REMOTE}/prepare_context_workspace.py "
                             f"--sheet-url '{args.sheet_url}' --outline-tab '{args.outline_tab}' "
                             f"--workspace /tmp/workspace 2>&1 | tail -20")
        print(f"  prepare: {out[-300:]}")
        code2, listing = run(box, "ls /tmp/workspace/context/ 2>/dev/null | head -5; "
                                  "test -f /tmp/workspace/manifest.json && echo MANIFEST_OK; "
                                  "ls /tmp/workspace/.baseline/context/ 2>/dev/null | wc -l")
        check("prepare_context_workspace.py built a workspace", "MANIFEST_OK" in listing,
              listing[-200:])
        # A missing baseline fails nothing loudly: present.py just reports no
        # change, and the agent burns a dozen turns working out why. Assert it.
        baseline_count = listing.strip().splitlines()[-1].strip()
        check("prepare snapshotted a baseline", baseline_count.isdigit() and int(baseline_count) > 0,
              f"{baseline_count} baseline files")

        # 3. present — the review path, and the only source of counts.
        code, out = run(box, "cd /tmp && FIRST=$(ls workspace/context/ | head -1) && "
                             f"{PY} {REMOTE}/present.py --workspace /tmp/workspace "
                             "--measure context/$FIRST 2>&1 | tail -5")
        print(f"  present: {out[-250:]}")
        check("present.py measured a real file", '"blocks"' in out and '"words"' in out,
              out[-200:])

        # Edit a file, then confirm present.py reports the change against the
        # baseline. This is the check that would have caught the missing
        # snapshot: without a baseline, present.py reports before == after, so a
        # broken review path looks perfectly healthy.
        box.write("/tmp/edit.py", EDIT_SCRIPT)
        code, out = run(box, f"cd /tmp && {PY} /tmp/edit.py")
        check("test edit applied", "EDITED" in out, out[-160:])

        code, out = run(box, "cd /tmp && FIRST=$(ls workspace/context/ | head -1) && "
                             f"{PY} {REMOTE}/present.py --workspace /tmp/workspace "
                             "--measure context/$FIRST 2>&1 | tail -3")
        print(f"  after edit: {out[-250:]}")
        check("present.py sees the edit against the baseline",
              '"before"' in out and '"change_pct": 0.0' not in out,
              out[-200:])

        # 4. commit — writes a new tab; the source tab is never touched.
        code, out = run(box, f"cd /tmp && {PY} {REMOTE}/commit_context.py "
                             f"--workspace /tmp/workspace "
                             f"--outline-target-tab '{SCRATCH_TAB}' 2>&1 | tail -20")
        print(f"  commit: {out[-300:]}")

        after_tabs = tab_names(args.sheet_url)
        made = SCRATCH_TAB in after_tabs
        check("commit_context.py created the new tab", made,
              f"{len(before_tabs)} -> {len(after_tabs)}")
        check("no source tab was disturbed",
              set(before_tabs) <= set(after_tabs),
              str(sorted(set(before_tabs) - set(after_tabs))))

        if made:
            import gspread  # noqa: PLC0415

            sys.path.insert(0, str(SCRIPTS))
            import _common  # noqa: PLC0415

            sh = gspread.service_account(filename=_common.resolve_sa_path()).open_by_url(
                args.sheet_url)
            ws = sh.worksheet(SCRATCH_TAB)
            rows = ws.get_all_values()
            check("committed tab has content", len(rows) > 1, f"{len(rows)} rows")
            sh.del_worksheet(ws)
            print(f"  scratch tab '{SCRATCH_TAB}' deleted")

    finally:
        if args.keep:
            print(f"\nsandbox {box.id} left running (--keep)")
        else:
            client.delete_sandbox(box.id)
            print(f"\nsandbox {box.id} deleted")

    print()
    if failures:
        print(f"{len(failures)} FAILED: " + ", ".join(failures))
        return 1
    print("all checks passed — the scripts work with no credential in the box")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
