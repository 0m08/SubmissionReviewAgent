"""Load a course Google Sheet into a workspace of editable markdown files.

Usage:
    python prepare_workspace.py \\
        --sheet-url "<URL>" \\
        --workspace /mnt/session/outputs/workspace \\
        [--source-tab "Slide Chunks"] \\
        [--outline-tab "Final Outline"]

Produces:
    <workspace>/
        manifest.json
        topics/topic_NN_<slug>.md      # slide chunks — unchanged format/logic
        context/topic_NN_<slug>.md     # research notes — NEW block format, see below

Forked from agents/slide_chunks_skills/skills/working-with-google-sheets. The
topics/ half (format_block, slide-chunk parsing) is unchanged from that trusted
script. The context/ half is rewritten: the original joined each subtopic/LO's
research notes with a bare "\\n\\n---\\n\\n" separator, which is ambiguous —
real research_notes content contains its own "---" section dividers (verified
against a live course sheet: 'Shutoff and Control Valves' has one). A reverse
parser built on that separator would silently mis-split content.

Instead each row gets an explicit, unlikely-to-collide block marker, the same
convention topics/*.md already uses successfully:

    ###LO ID: 0
    ####**Topic:**
    Water Service and Distribution Pipe Materials
    ####**Subtopic:**
    Copper Tube in Water Systems
    ####**Learning Objective:**
    Recognize copper tube by appearance, common markings, and typical...
    ####**Research Notes:**
    <content, greedy to end of block>

Topic / Subtopic / Learning Objective are the write-back identity key for
commit_context.py — see that script and SKILL.md for why they must not be
edited. Confirmed against a live sheet: (Topic, Subtopic) is NOT unique
(multiple LOs per subtopic), but (Topic, Subtopic, Learning Objectives) is.

This script always requires a valid slide-chunks tab — topic segmentation is
derived from it, and it exits with an error if that tab is missing or empty,
even if all you actually want is a research-notes edit. For a session that
touches only research notes, with no dependency on the slide-chunks tab, use
prepare_context_workspace.py instead — same context/*.md format, same
manifest shape (so commit_context.py works unchanged against either), but
topic order comes from the Final Outline tab and no topics/ directory is
produced.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from _common import (OUTLINE_FILENAME, build_context_for_topic, build_outline,
                     eprint, normalize_punctuation, open_sheet, refuse_if_edited,
                     slugify, snapshot_baseline)


DEFAULT_SOURCE_TAB = "Slide Chunks"
DEFAULT_OUTLINE_TAB = "Final Outline"
DEFAULT_TARGET_TAB = "Slide Chunks (Revised)"
DEFAULT_OUTLINE_TARGET_TAB = "Final Outline (Revised)"


# Column aliases — the sheet may use Title-cased or lowercase headers.
COL_ALIASES = {
    "topic": ("topic", "Topic"),
    "subtopic": ("subtopic", "Subtopic"),
    "slide_type": ("Slide Type", "slide_type"),
    "slide_chunk_title": ("Slide Chunk Title", "slide_chunk_title"),
    "slide_chunk": ("Slide Chunk", "slide_chunk"),
}


def _get(row, key):
    for col in COL_ALIASES[key]:
        if col in row and pd.notna(row[col]) and str(row[col]).strip():
            return str(row[col]).strip()
    return ""


def format_block(block_id: int, row) -> str:
    """Slide-chunk block text. Unchanged from the trusted checklist skill."""
    # normalize_punctuation: curly quotes make a span uneditable by the edit
    # tool (see _common.normalize_punctuation). commit_workspace.py writes a
    # fresh tab rather than matching rows, so folding them here is lossless.
    topic = normalize_punctuation(_get(row, "topic"))
    subtopic = normalize_punctuation(_get(row, "subtopic"))
    slide_type = normalize_punctuation(_get(row, "slide_type"))
    title = normalize_punctuation(_get(row, "slide_chunk_title"))
    content = normalize_punctuation(_get(row, "slide_chunk"))

    parts = [
        f"###Block ID: {block_id}",
        "####**Topic:**",
        topic,
        "####**Subtopic:**",
        subtopic,
        "####**Slide Chunk:**",
    ]
    if slide_type:
        parts.append(f"Slide Type: {slide_type}")
    if title:
        parts.append(f"Title: {title}")
    if content:
        parts.append(f"Content: {content}")

    return "\n".join(parts)


def _read_tab(sheet, tab_name: str, required: bool):
    try:
        ws = sheet.worksheet(tab_name)
    except Exception:
        if required:
            available = [w.title for w in sheet.worksheets()]
            eprint(
                f"ERROR: required tab '{tab_name}' not found. "
                f"Available tabs: {available}. "
                f"Re-run with --source-tab pointing at the right tab."
            )
            sys.exit(2)
        return pd.DataFrame()
    return pd.DataFrame(ws.get_all_records())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sheet-url", required=True)
    parser.add_argument("--workspace", required=True, help="Output directory")
    parser.add_argument("--source-tab", default=DEFAULT_SOURCE_TAB,
                        help="Slide chunks tab. Varies per course sheet — check with list_tabs.py.")
    parser.add_argument("--outline-tab", default=DEFAULT_OUTLINE_TAB)
    parser.add_argument("--target-tab", default=DEFAULT_TARGET_TAB,
                        help="Tab name commit_workspace.py will write slide chunks to")
    parser.add_argument("--outline-target-tab", default=DEFAULT_OUTLINE_TARGET_TAB,
                        help="Tab name commit_context.py will write research notes to")
    parser.add_argument("--discard-local", action="store_true",
                        help="Overwrite a workspace that has uncommitted edits.")
    args = parser.parse_args()

    workspace = Path(args.workspace)
    refused = refuse_if_edited(workspace, args.discard_local)
    if refused is not None:
        return refused
    (workspace / "topics").mkdir(parents=True, exist_ok=True)
    (workspace / "context").mkdir(parents=True, exist_ok=True)

    sheet = open_sheet(args.sheet_url)

    slide_chunks_df = _read_tab(sheet, args.source_tab, required=True)
    if slide_chunks_df.empty:
        eprint(f"ERROR: source tab '{args.source_tab}' is empty.")
        return 2
    outline_df = _read_tab(sheet, args.outline_tab, required=False)

    topic_col = next((c for c in COL_ALIASES["topic"] if c in slide_chunks_df.columns), None)
    if topic_col is None:
        eprint(
            f"ERROR: source tab '{args.source_tab}' has no Topic / topic column. "
            f"Columns present: {list(slide_chunks_df.columns)}"
        )
        return 2

    seen, ordered_topics = set(), []
    for t in slide_chunks_df[topic_col].astype(str):
        if t and t not in seen:
            seen.add(t)
            ordered_topics.append(t)

    topics_manifest = []
    for idx, topic_name in enumerate(ordered_topics, start=1):
        slug = slugify(topic_name)
        topic_file = workspace / "topics" / f"topic_{idx:02d}_{slug}.md"
        context_file = workspace / "context" / f"topic_{idx:02d}_{slug}.md"

        topic_rows = slide_chunks_df[slide_chunks_df[topic_col].astype(str) == topic_name]
        blocks = [
            format_block(block_id, row)
            for block_id, (_, row) in enumerate(topic_rows.iterrows())
        ]
        topic_file.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
        context_file.write_text(build_context_for_topic(topic_name, outline_df), encoding="utf-8")

        topics_manifest.append({
            "index": idx,
            "name": topic_name,
            "slug": slug,
            "file": str(topic_file.relative_to(workspace).as_posix()),
            "context_file": str(context_file.relative_to(workspace).as_posix()),
            "row_count": len(topic_rows),
        })

    outline_file = workspace / OUTLINE_FILENAME
    outline_file.write_text(
        build_outline(ordered_topics, outline_df, topics_manifest), encoding="utf-8")

    manifest = {
        "mode": "slide_chunks_and_context",
        "outline_file": OUTLINE_FILENAME,
        "sheet_url": args.sheet_url,
        "source_tab": args.source_tab,
        "outline_tab": args.outline_tab,
        "target_tab": args.target_tab,
        "outline_target_tab": args.outline_target_tab,
        "topics": topics_manifest,
        "output_columns": ["Topic", "Subtopic", "Slide Type", "Slide Chunk Title", "Slide Chunk"],
    }
    (workspace / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    n = snapshot_baseline(workspace)
    print(f"OK — wrote {len(topics_manifest)} topics to {workspace}")
    print(f"Baseline snapshot: {n} file(s) copied to {workspace}/.baseline")
    return 0


if __name__ == "__main__":
    sys.exit(main())
