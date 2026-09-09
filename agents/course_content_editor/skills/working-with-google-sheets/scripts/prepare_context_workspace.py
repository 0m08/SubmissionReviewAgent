"""Load a course's research notes ONLY into a workspace of markdown files.

Usage:
    python prepare_context_workspace.py \\
        --sheet-url "<URL>" \\
        --workspace /mnt/session/outputs/workspace \\
        [--outline-tab "Final Outline"] \\
        [--outline-target-tab "Final Outline (Revised)"]

Produces:
    <workspace>/
        manifest.json
        context/topic_NN_<slug>.md     # research notes — same ###LO ID: format
                                        # prepare_workspace.py produces

No topics/ directory, no slide-chunks tab involved at all. Use this instead
of prepare_workspace.py when the session is only ever going to touch research
notes — prepare_workspace.py requires a valid slide-chunks tab to run
(topic segmentation comes from it) and errors out if that tab is missing or
empty, which is unnecessary friction for a research-notes-only edit.

Topic order here comes from the Final Outline tab's own row order (first
appearance of each Topic value), since there's no slide-chunks tab to derive
it from instead.

commit_context.py works unchanged against a workspace built by either script
— both write the same context/*.md block format and the same manifest shape
(sheet_url, outline_tab, outline_target_tab, topics[].context_file). It never
reads topics/ or the slide-chunks-specific manifest fields.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from _common import OUTLINE_COL_ALIASES, build_context_for_topic, eprint, open_sheet, slugify


DEFAULT_OUTLINE_TAB = "Final Outline"
DEFAULT_OUTLINE_TARGET_TAB = "Final Outline (Revised)"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sheet-url", required=True)
    parser.add_argument("--workspace", required=True, help="Output directory")
    parser.add_argument("--outline-tab", default=DEFAULT_OUTLINE_TAB)
    parser.add_argument("--outline-target-tab", default=DEFAULT_OUTLINE_TARGET_TAB,
                        help="Tab name commit_context.py will write research notes to")
    args = parser.parse_args()

    workspace = Path(args.workspace)
    (workspace / "context").mkdir(parents=True, exist_ok=True)

    sheet = open_sheet(args.sheet_url)

    try:
        ws = sheet.worksheet(args.outline_tab)
    except Exception:
        available = [w.title for w in sheet.worksheets()]
        eprint(
            f"ERROR: outline tab '{args.outline_tab}' not found. "
            f"Available tabs: {available}. "
            f"Re-run with --outline-tab pointing at the right tab."
        )
        return 2
    outline_df = pd.DataFrame(ws.get_all_records())
    if outline_df.empty:
        eprint(f"ERROR: outline tab '{args.outline_tab}' is empty.")
        return 2

    topic_col = next((c for c in OUTLINE_COL_ALIASES["topic"] if c in outline_df.columns), None)
    if topic_col is None:
        eprint(
            f"ERROR: outline tab '{args.outline_tab}' has no Topic / topic column. "
            f"Columns present: {list(outline_df.columns)}"
        )
        return 2

    seen, ordered_topics = set(), []
    for t in outline_df[topic_col].astype(str):
        if t and t not in seen:
            seen.add(t)
            ordered_topics.append(t)

    topics_manifest = []
    for idx, topic_name in enumerate(ordered_topics, start=1):
        slug = slugify(topic_name)
        context_file = workspace / "context" / f"topic_{idx:02d}_{slug}.md"
        context_file.write_text(build_context_for_topic(topic_name, outline_df), encoding="utf-8")

        topics_manifest.append({
            "index": idx,
            "name": topic_name,
            "slug": slug,
            "context_file": str(context_file.relative_to(workspace).as_posix()),
        })

    manifest = {
        "mode": "context_only",
        "sheet_url": args.sheet_url,
        "outline_tab": args.outline_tab,
        "outline_target_tab": args.outline_target_tab,
        "topics": topics_manifest,
    }
    (workspace / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"OK — wrote {len(topics_manifest)} topics to {workspace} (context only, no slide chunks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
