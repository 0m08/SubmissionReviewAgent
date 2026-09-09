"""Show the user the real current content of files you have edited.

    python /opt/cce/scripts/present.py --workspace workspace context/topic_01_x.md
    python /opt/cce/scripts/present.py --workspace workspace --measure topics/topic_02_y.md
    python /opt/cce/scripts/present.py --workspace workspace --text context/topic_01_x.md

## Who this output is for

**The user, not you.** This prints a JSON payload that the client renders as
before/after cards on screen; a middleware intercepts it, shows it to the
person, and hands you back counts and an acknowledgement only.

That split is the whole point. Content that passes through your context is
content you might paraphrase, truncate, or retype slightly differently — and a
reader cannot tell a faithful quote from a confident approximation. Sending the
bytes straight from disk to the screen removes the opportunity.

So: **do not repeat what this prints.** You will not usually see it. Say what
you changed and why in your own words; the content itself is already shown.

`--measure` gives counts alone, no content — that one *is* for you, when you
need to check a direction (tighten, trim, expand) before and after an edit.

`--text` renders for a terminal instead of a client. Useful when running this
by hand; the agent should not need it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _presentation as pres  # noqa: E402

# The middleware keys off this so it can tell a present payload from any other
# JSON a script might print.
PAYLOAD_KEY = "cce_present_v1"


def _collect(workspace: Path, paths: list[str], block_ids: list[str], note: str) -> dict:
    files: list[dict] = []
    missing: list[str] = []
    for path in paths:
        key = pres.resolve_key(workspace, path)
        if key is None:
            missing.append(path)
            continue
        before, after, kind = pres.read_pair(workspace, key)
        files.append({
            "path": key,
            "note": note,
            "kind": kind,
            "counts": pres.measure(before, after, kind),
            "blocks": pres.select(before, after, kind, block_ids),
            # The whole file, both versions. `blocks` is a selection — the
            # blocks that changed — and a selection cannot support the views
            # the client actually offers: a side-by-side diff needs the
            # unchanged paragraphs around an edit to read as context, and
            # select-to-comment needs every block, including the ones nobody
            # touched, because "this doesn't follow from the previous slide"
            # is a comment about a block that did not change.
            #
            # This does not reach the model. The middleware replaces the tool
            # result with counts before it returns (see present_bridge._ack),
            # so these bytes go to the screen and nowhere else — which is the
            # entire reason the content is worth sending in full.
            "before": before,
            "after": after,
        })
    return {"files": files, "missing": missing}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="Workspace-relative file paths.")
    ap.add_argument("--workspace", default="workspace")
    ap.add_argument("--blocks", default="",
                    help="Comma-separated block IDs. Omit to show every block that changed.")
    ap.add_argument("--note", default="", help="One short line saying what to look at.")
    ap.add_argument("--measure", action="store_true",
                    help="Counts only, no content. This output IS for the agent.")
    ap.add_argument("--text", action="store_true",
                    help="Render for a terminal instead of emitting a client payload.")
    args = ap.parse_args()

    workspace = Path(args.workspace)
    if not workspace.exists():
        print(f"ERROR: no workspace at {workspace}. Run a prepare script first.",
              file=sys.stderr)
        return 2

    block_ids = [b.strip() for b in args.blocks.split(",") if b.strip()]

    # --- counts only: this is the one mode written for the agent to read ----
    if args.measure:
        missing: list[str] = []
        for path in args.paths:
            key = pres.resolve_key(workspace, path)
            if key is None:
                missing.append(path)
                continue
            before, after, kind = pres.read_pair(workspace, key)
            print(f"{key}: " + json.dumps(pres.measure(before, after, kind)))
        if missing:
            print(f"NOT FOUND: {missing}", file=sys.stderr)
            print("Paths are workspace-relative, e.g. topics/topic_01_<slug>.md",
                  file=sys.stderr)
            return 1
        return 0

    payload = _collect(workspace, args.paths, block_ids, args.note)

    # --- human-readable, for running this by hand ---------------------------
    if args.text:
        for entry in payload["files"]:
            pres.render_cli(entry["path"], entry["note"], entry["blocks"])
            print("counts: " + json.dumps(entry["counts"]))
            print()
        if payload["missing"]:
            print(f"NOT FOUND: {payload['missing']}", file=sys.stderr)
            return 1
        return 0

    # --- the client payload -------------------------------------------------
    # One line, one key, so the middleware can recognise it without guessing.
    print(json.dumps({PAYLOAD_KEY: payload}))
    if payload["missing"]:
        print(f"NOT FOUND: {payload['missing']}", file=sys.stderr)
        print("Paths are workspace-relative, e.g. context/topic_01_<slug>.md",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
