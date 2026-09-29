"""Parse edited markdown files in a workspace and write them back to the sheet.

Usage:
    python commit_workspace.py \\
        --workspace /mnt/session/outputs/workspace \\
        [--target-tab "Slide Chunks (Revised)"]

Reads manifest.json for the sheet URL and target tab name. Parses each
topics/*.md file in order (deterministic regex — no LLM). Assembles a
DataFrame with the same column structure as the original sheet, and
writes to the target tab. The original source tab is never modified.

See references/parsing-rules.md for the exact contract this script
follows when interpreting each block.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

from _common import eprint, open_sheet


BLOCK_START = re.compile(r"^###Block ID:\s*(.+)$", re.MULTILINE)


def parse_block(block_text: str) -> dict | None:
    """Parse one block's body (everything after the ###Block ID: line).

    Returns a dict with keys Topic, Subtopic, Slide Type, Slide Chunk Title,
    Slide Chunk — or None if the block is malformed.

    Single-line fields (Topic, Subtopic, Slide Type, Title) take the
    immediately-following line. Content is greedy: from "Content:" to the
    end of the block. This greediness is intentional — slide bodies can
    have arbitrary multiline content including markdown, video transcripts,
    and image links.
    """
    lines = block_text.split("\n")
    result = {
        "Topic": "",
        "Subtopic": "",
        "Slide Type": "",
        "Slide Chunk Title": "",
        "Slide Chunk": "",
    }

    i = 0
    n = len(lines)
    in_slide_chunk = False
    capturing_content = False
    content_lines: list[str] = []

    while i < n:
        line = lines[i]
        stripped = line.strip()

        if not in_slide_chunk:
            if stripped == "####**Topic:**":
                i += 1
                if i < n:
                    result["Topic"] = lines[i].strip()
            elif stripped == "####**Subtopic:**":
                i += 1
                if i < n:
                    result["Subtopic"] = lines[i].strip()
            elif stripped == "####**Slide Chunk:**":
                in_slide_chunk = True
        else:
            if capturing_content:
                content_lines.append(line)
            elif stripped.startswith("Slide Type:"):
                result["Slide Type"] = stripped[len("Slide Type:"):].strip()
            elif stripped.startswith("Title:"):
                result["Slide Chunk Title"] = stripped[len("Title:"):].strip()
            elif stripped.startswith("Content:"):
                first = stripped[len("Content:"):].strip()
                if first:
                    content_lines.append(first)
                capturing_content = True
        i += 1

    result["Slide Chunk"] = "\n".join(content_lines).rstrip()

    # Malformed check: must have a Slide Chunk section AND at least one of
    # Title / Content. Empty Subtopic and Slide Type are allowed (matches
    # the original parser's lenient behavior at slide_chunks_parsing.py:147).
    if not in_slide_chunk:
        return None
    if not (result["Slide Chunk Title"] or result["Slide Chunk"]):
        return None
    if not result["Topic"]:
        return None
    return result


def parse_topic_file(path: Path) -> list[dict]:
    """Parse all blocks in a single topic file."""
    text = path.read_text(encoding="utf-8")

    starts = list(BLOCK_START.finditer(text))
    if not starts:
        eprint(f"WARN: {path.name} contains no ###Block ID: blocks; skipping file")
        return []

    rows: list[dict] = []
    for i, match in enumerate(starts):
        body_start = match.end()
        body_end = starts[i + 1].start() if i + 1 < len(starts) else len(text)
        body = text[body_start:body_end]

        parsed = parse_block(body)
        if parsed is None:
            line_no = text.count("\n", 0, match.start()) + 1
            eprint(
                f"WARN: skipping malformed block in {path.name} near line {line_no} "
                f"(Block ID label: {match.group(1).strip()})"
            )
            continue
        rows.append(parsed)
    return rows


def _write_tab(sheet, tab_name: str, df: pd.DataFrame, columns: list[str]) -> str:
    """Overwrite or create the target tab with df. Returns tab URL."""
    try:
        ws = sheet.worksheet(tab_name)
        ws.clear()
    except Exception:
        ws = sheet.add_worksheet(title=tab_name, rows=str(max(len(df) + 10, 100)), cols="26")

    df = df[columns].astype(str).fillna("")
    ws.update([columns] + df.values.tolist())
    return ws.url


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--target-tab", default=None,
                        help="Overrides manifest.target_tab if set.")
    args = parser.parse_args()

    workspace = Path(args.workspace)
    manifest_path = workspace / "manifest.json"
    if not manifest_path.exists():
        eprint(f"ERROR: {manifest_path} not found. Run prepare_workspace.py first.")
        return 2

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sheet_url = manifest["sheet_url"]
    target_tab = args.target_tab or manifest.get("target_tab", "Slide Chunks (Revised)")
    columns = manifest.get("output_columns",
                           ["Topic", "Subtopic", "Slide Type", "Slide Chunk Title", "Slide Chunk"])

    all_rows: list[dict] = []
    for topic_meta in manifest["topics"]:
        topic_path = workspace / topic_meta["file"]
        if not topic_path.exists():
            eprint(f"WARN: {topic_path} missing — entire topic '{topic_meta['name']}' will be dropped")
            continue
        all_rows.extend(parse_topic_file(topic_path))

    if not all_rows:
        eprint("ERROR: no parseable blocks found across workspace. Nothing to write.")
        return 2

    df = pd.DataFrame(all_rows)
    # Ensure every output column exists even if empty in the data.
    for col in columns:
        if col not in df.columns:
            df[col] = ""

    sheet = open_sheet(sheet_url)
    tab_url = _write_tab(sheet, target_tab, df, columns)

    print(f"OK — wrote {len(df)} rows to tab '{target_tab}'")
    print(tab_url)
    return 0


if __name__ == "__main__":
    sys.exit(main())
