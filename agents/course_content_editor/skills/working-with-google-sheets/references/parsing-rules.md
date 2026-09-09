# Parsing Rules

How `commit_workspace.py` and `commit_context.py` turn markdown back into
sheet rows. The slide-chunk half is unchanged from the trusted checklist
skill; the context half is new.

## Slide chunks (`topics/*.md` → `commit_workspace.py`)

A block starts at a line matching `^###Block ID:\s*(.+)$` and ends just before
the next such line (or EOF).

| Header line | What follows | DataFrame column |
|-------------|--------------|------------------|
| `####**Topic:**` | next line, stripped | `Topic` |
| `####**Subtopic:**` | next line, stripped | `Subtopic` |
| `####**Slide Chunk:**` | one or more `Key: value` lines | see below |

Inside `Slide Chunk:`:

| Line prefix | DataFrame column |
|-------------|------------------|
| `Slide Type:` | `Slide Type` (single line) |
| `Title:` | `Slide Chunk Title` (single line) |
| `Content:` | `Slide Chunk` (multiline — greedy to the next `###Block ID:` or EOF) |

A block is malformed and skipped if it has no `####**Topic:**` line, no
`####**Slide Chunk:**` line, or both `Title:` and `Content:` are missing.
Missing `Subtopic` / `Slide Type` fall back to `""`.

Row order in the output is file order (`topic_01` before `topic_02` …) then
block-appearance order within a file. `Block ID:` values are descriptive
labels, not row keys — they are not used for ordering or matching.

**Round-trip invariant:** `prepare_workspace.py` then `commit_workspace.py`
with zero edits should reproduce the source `Slide Chunks` tab row-for-row.

## Research notes (`context/*.md` → `commit_context.py`)

A block starts at `^###LO ID:\s*(.+)$` and ends just before the next such line
(or EOF).

| Header line | What follows | Role |
|-------------|--------------|------|
| `####**Topic:**` | next line, stripped | identity key |
| `####**Subtopic:**` | next line, stripped | identity key |
| `####**Learning Objective:**` | next line, stripped | identity key |
| `####**Research Notes:**` | multiline — greedy to next `###LO ID:` or EOF | the editable content |

A block is malformed and skipped if any of Topic / Subtopic / Learning
Objective is missing. Research Notes may legitimately be empty.

**Why this differs from the `\n\n---\n\n` join `prepare_workspace.py` used to
use:** verified against a live course sheet that real `research_notes` content
contains its own bare `---` line (a markdown section divider). Joining blocks
on that same separator makes a reverse parser ambiguous — it cannot tell a
block boundary from content. The `###LO ID:` marker is a full line
(`^###LO ID:\s*(.+)$`) that legitimate research-notes prose will not
reproduce, the same way `###Block ID:` already works safely for slide chunks.

### Identity key, not free text

`commit_context.py` never reconstructs the sheet from these files. It
re-reads the live `Final Outline` tab fresh, matches each parsed block to a
sheet row by **(Topic, Subtopic, Learning Objective)** — verified unique on a
live sheet, even though (Topic, Subtopic) alone is not, because one subtopic
can carry several LOs — and overwrites only that row's `research_notes` cell.
Every other column on the sheet (references, source links, `context_0..3`,
`rn_order`, `inline_image_placement_status`, `slide_chunks`, …) passes through
completely untouched.

**This means the editing skill must treat Topic:, Subtopic:, and Learning
Objective: as read-only.** Editing them breaks the match:

- No matching row → the block's notes are **not written**, and `commit_context.py`
  warns with the exact Topic/Subtopic/LO it couldn't find. It does not guess.
- A **deleted** `###LO ID:` block is not a request to blank that row — it means
  no change was requested, and the row's research notes are left exactly as
  they were in the source sheet. To blank notes, keep the block and empty the
  `Research Notes:` body.

### Round-trip invariant

`prepare_workspace.py` then `commit_context.py` with zero edits should
reproduce every column of the source `Final Outline` tab row-for-row — because
unmatched/unchanged rows are never touched, this holds even for columns this
skill never reads.
