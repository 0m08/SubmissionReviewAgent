"""Parse edited context/*.md files and write research notes back to the sheet.

Usage:
    python commit_context.py \\
        --workspace /mnt/session/outputs/workspace \\
        [--outline-target-tab "Final Outline (Revised)"]

Unlike commit_workspace.py (which reassembles a whole new DataFrame from
scratch), this script re-reads the ENTIRE Final Outline tab fresh, then
overwrites only the research_notes cell of rows it can match — every other
column (References, context_0..3, rn_order, slide_chunks, inline image
status, etc.) passes through untouched. A live course sheet was inspected to
design this: Final Outline carries ~20 columns beyond the four this skill
reads, so reconstructing the frame from the .md files (which only ever
carried Topic/Subtopic/LO/Research Notes) would silently drop the rest.

Row identity for the match is (Topic, Subtopic, Learning Objectives) —
verified against a live sheet that this triple is unique even though
(Topic, Subtopic) is not (one subtopic can carry several LOs, each with its
own research-notes row). This means the editing skill must never change the
Topic:, Subtopic:, or Learning Objective: fields inside a context block —
only the Research Notes: body. A block whose identity fields were edited (or
that doesn't match any row) is reported and skipped, never guessed at.

Deleting an entire ###LO ID: block from a context file is NOT the same as
blanking that row's research notes — a missing block simply means "no change
requested for this row," and its research_notes cell is left exactly as it
was in the source sheet. To actually blank a row's notes, keep the block but
empty the Research Notes: body.

See references/parsing-rules.md for the exact block-marker contract.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

from _common import eprint, normalize_punctuation, open_sheet


LO_BLOCK_START = re.compile(r"^###LO ID:\s*(.+)$", re.MULTILINE)

OUTLINE_COL_ALIASES = {
    "topic": ("Topic", "topic"),
    "subtopic": ("Subtopic", "subtopic"),
    "learning_objectives": ("Learning Objectives", "learning_objectives"),
    "research_notes": ("research_notes", "Research Notes"),
}


def _resolve_col(df: pd.DataFrame, key: str) -> str | None:
    for col in OUTLINE_COL_ALIASES[key]:
        if col in df.columns:
            return col
    return None


def parse_lo_block(block_text: str) -> dict | None:
    """Parse one ###LO ID: block body into Topic/Subtopic/LO/Research Notes.

    Topic, Subtopic, Learning Objective are single-line fields (verified: no
    embedded newlines in a live sheet). Research Notes is greedy to the end
    of the block, mirroring Content:'s behavior in the slide-chunk parser —
    same intentional greediness, same hazard: a body line before the
    Research Notes: header is silently dropped, so the editing skill must
    always write full blocks in field order.
    """
    lines = block_text.split("\n")
    result = {"Topic": "", "Subtopic": "", "Learning Objective": "", "Research Notes": ""}

    i, n = 0, len(lines)
    capturing_notes = False
    notes_lines: list[str] = []

    while i < n:
        stripped = lines[i].strip()
        if capturing_notes:
            notes_lines.append(lines[i])
        elif stripped == "####**Topic:**":
            i += 1
            if i < n:
                result["Topic"] = lines[i].strip()
        elif stripped == "####**Subtopic:**":
            i += 1
            if i < n:
                result["Subtopic"] = lines[i].strip()
        elif stripped == "####**Learning Objective:**":
            i += 1
            if i < n:
                result["Learning Objective"] = lines[i].strip()
        elif stripped == "####**Research Notes:**":
            capturing_notes = True
        i += 1

    result["Research Notes"] = "\n".join(notes_lines).strip("\n")

    if not result["Topic"] or not result["Subtopic"] or not result["Learning Objective"]:
        return None
    return result


def parse_context_file(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    starts = list(LO_BLOCK_START.finditer(text))
    if not starts:
        eprint(f"WARN: {path.name} contains no ###LO ID: blocks; skipping file")
        return []

    parsed: list[dict] = []
    for i, match in enumerate(starts):
        body_start = match.end()
        body_end = starts[i + 1].start() if i + 1 < len(starts) else len(text)
        block = parse_lo_block(text[body_start:body_end])
        if block is None:
            line_no = text.count("\n", 0, match.start()) + 1
            eprint(
                f"WARN: skipping malformed context block in {path.name} near line {line_no} "
                f"(LO ID label: {match.group(1).strip()}) — missing Topic/Subtopic/Learning Objective"
            )
            continue
        parsed.append(block)
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--outline-target-tab", default=None,
                        help="Overrides manifest.outline_target_tab if set.")
    args = parser.parse_args()

    workspace = Path(args.workspace)
    manifest_path = workspace / "manifest.json"
    if not manifest_path.exists():
        eprint(f"ERROR: {manifest_path} not found. Run prepare_workspace.py first.")
        return 2

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sheet_url = manifest["sheet_url"]
    outline_tab = manifest.get("outline_tab", "Final Outline")
    target_tab = args.outline_target_tab or manifest.get("outline_target_tab", "Final Outline (Revised)")

    edits: list[dict] = []
    for topic_meta in manifest["topics"]:
        context_path = workspace / topic_meta["context_file"]
        if not context_path.exists():
            eprint(f"WARN: {context_path} missing — skipping topic '{topic_meta['name']}'")
            continue
        edits.extend(parse_context_file(context_path))

    if not edits:
        eprint("ERROR: no parseable LO blocks found across workspace context files. Nothing to write.")
        return 2

    sheet = open_sheet(sheet_url)
    try:
        ws = sheet.worksheet(outline_tab)
    except Exception:
        eprint(f"ERROR: outline tab '{outline_tab}' not found in the sheet.")
        return 2
    df = pd.DataFrame(ws.get_all_records())

    topic_col = _resolve_col(df, "topic")
    subtopic_col = _resolve_col(df, "subtopic")
    lo_col = _resolve_col(df, "learning_objectives")
    rn_col = _resolve_col(df, "research_notes")
    missing = [name for name, col in [
        ("Topic", topic_col), ("Subtopic", subtopic_col),
        ("Learning Objectives", lo_col), ("research_notes", rn_col),
    ] if col is None]
    if missing:
        eprint(f"ERROR: '{outline_tab}' is missing expected column(s): {missing}. "
               f"Columns present: {list(df.columns)}")
        return 2

    # Build the identity index. Verified unique on a live sheet even though
    # (Topic, Subtopic) alone is not.
    # Identity is compared with punctuation normalized on BOTH sides: the
    # workspace may carry straight quotes where the sheet has curly ones (see
    # _common.normalize_punctuation), and an identity field must still match.
    def _k(*parts) -> tuple:
        return tuple(normalize_punctuation(str(p).strip()) for p in parts)

    key = [_k(t, s_, l) for t, s_, l in zip(
        df[topic_col].astype(str),
        df[subtopic_col].astype(str),
        df[lo_col].astype(str),
    )]
    index_by_key: dict[tuple, list[int]] = {}
    for i, k in enumerate(key):
        index_by_key.setdefault(k, []).append(i)

    matched, unmatched, ambiguous, blanked = 0, [], [], []
    for edit in edits:
        k = _k(edit["Topic"], edit["Subtopic"], edit["Learning Objective"])
        rows = index_by_key.get(k)
        if not rows:
            unmatched.append(edit)
            continue
        if len(rows) > 1:
            ambiguous.append(edit)
            continue
        row_idx = rows[0]
        original_notes = str(df.at[row_idx, rn_col]).strip()
        new_notes = edit["Research Notes"].strip()
        # Emptying a row's notes is legitimate — the skill documents it as
        # the way to deliberately blank one. But it's also exactly what a
        # dropped "####**Research Notes:**" header produces (parse_lo_block
        # returns "" when the header is missing, not an error), which is
        # silent data loss wearing the same shape as an intentional edit.
        # Can't tell those apart here, so flag every non-trivial case loudly
        # instead of guessing — the agent relays this, the user confirms.
        if original_notes and not new_notes:
            blanked.append(edit)
        df.at[row_idx, rn_col] = edit["Research Notes"]
        matched += 1

    for edit in unmatched:
        eprint(f"WARN: no matching row for Topic={edit['Topic']!r} Subtopic={edit['Subtopic']!r} "
               f"Learning Objective={edit['Learning Objective']!r} — identity fields may have been "
               f"edited or the row no longer exists. This row's research notes were NOT written.")
    for edit in ambiguous:
        eprint(f"WARN: multiple rows match Topic={edit['Topic']!r} Subtopic={edit['Subtopic']!r} "
               f"Learning Objective={edit['Learning Objective']!r} — ambiguous, skipped.")
    for edit in blanked:
        eprint(f"WARN: writing an EMPTY research_notes for Topic={edit['Topic']!r} "
               f"Subtopic={edit['Subtopic']!r} Learning Objective={edit['Learning Objective']!r}, "
               f"which previously had content. If this wasn't a deliberate blank-out, the block's "
               f"####**Research Notes:** header was likely dropped during editing (a known hazard of "
               f"hand-editing outside the text-editor tool) — check the source .md file before trusting "
               f"this write.")

    if matched == 0:
        eprint("ERROR: none of the edited context blocks matched a row in the sheet. Nothing written.")
        return 2

    try:
        target_ws = sheet.worksheet(target_tab)
        target_ws.clear()
    except Exception:
        target_ws = sheet.add_worksheet(title=target_tab, rows=str(max(len(df) + 10, 100)), cols="26")

    columns = list(df.columns)
    out = df[columns].astype(str).fillna("")
    target_ws.update([columns] + out.values.tolist())

    print(f"OK — wrote {len(df)} rows to tab '{target_tab}' ({matched} research-notes cell(s) updated, "
          f"{len(unmatched)} unmatched, {len(ambiguous)} ambiguous, {len(blanked)} blanked-with-prior-content "
          f"— see stderr for details on any nonzero count)")
    print(target_ws.url)
    return 0


if __name__ == "__main__":
    sys.exit(main())
