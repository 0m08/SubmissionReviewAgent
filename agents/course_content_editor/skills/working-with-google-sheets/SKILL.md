---
name: working-with-google-sheets
description: Load a course's slide chunks and research notes from a Google Sheet into local markdown files for editing, then write edits back to new tabs. Use whenever the user pastes a Google Sheet URL, or asks to edit slide chunks or research notes for a course.
---

# Working with Google Sheets

Round-trips a course between a Google Sheet and a local workspace of markdown
files: slide chunks (`topics/*.md`) and research notes (`context/*.md`). The
editing-course-content skill edits those files; this skill handles I/O on
both ends.

Forked from the checklist agent's `working-with-google-sheets` skill. The
slide-chunk half is unchanged. The research-notes half is new — the checklist
agent never had to write research notes back, only read them for context.

## Where this skill's scripts actually live

Once this skill is loaded, its scripts run from
`/workspace/skills/working-with-google-sheets/scripts/` — that's an absolute
path, not something to rediscover by exploring. `cd` there (or reference
scripts with that full path) before running anything.

## Never touch the sheet except through these scripts

Don't call `gspread` yourself, and don't try to fetch a Google Sheet URL with
a generic web-fetch tool — it isn't API-authenticated and will fail. Every
script here already handles credential resolution internally (see
Authentication below), including checking more than one possible mount path
for the service account file, precisely so you never have to work that out
yourself. If you need to inspect the sheet — confirm access, see what tabs
exist — that's `check_auth.py` / `list_tabs.py`, not a one-off script you
write.

## Always use this workspace path

**Always pass `--workspace /mnt/session/outputs/workspace`** to whichever
prepare script you run — not a relative path, not anywhere else under
`/mnt/session/`. Files written under `/mnt/session/outputs/` are the only
ones that become visible outside the session through the Files API
(`GET /v1/files?scope_id=<session_id>`), which is how the review UI reads
your edits to build its diff view. A workspace built anywhere else is
invisible to it, even though every script here works identically regardless
of path.

## When to use

- **Start of a session, and the user wants to edit slide chunks (with or
  without also touching research notes):** run `prepare_workspace.py`. It
  always builds both `topics/*.md` and `context/*.md`, and it **requires** a
  valid slide-chunks tab — it errors out if that tab is missing or empty.
- **Start of a session, and the user only wants to edit research notes —
  slide chunks never come up:** run `prepare_context_workspace.py` instead.
  It only touches the Final Outline tab and has no dependency on a
  slide-chunks tab existing at all. Don't reach for `prepare_workspace.py`
  here — it would fail (or silently require a tab the user doesn't care
  about) for no reason.
- **After editing slide chunks:** run `commit_workspace.py` to write a
  `Slide Chunks (Revised)` tab.
- **After editing research notes** (from either kind of workspace): run
  `commit_context.py` to write a `Final Outline (Revised)` tab.
- **Sanity check:** `check_auth.py` or `list_tabs.py` before a long session if
  there's doubt about access or tab names. Tab names vary per course sheet —
  don't assume `Slide Chunks` is the literal tab name; confirm with
  `list_tabs.py` first if the user hasn't told you.

## Workspace layout

```
<workspace>/
├── manifest.json                    # sheet URL, tab names, topic→file map
├── topics/topic_NN_<slug>.md        # slide chunks — one or more ###Block ID: blocks
└── context/topic_NN_<slug>.md       # research notes — one or more ###LO ID: blocks
```

### `topics/topic_NN_<slug>.md`

```
###Block ID: 0
####**Topic:**
HVAC Fundamentals
####**Subtopic:**
Refrigeration Cycle
####**Slide Chunk:**
Slide Type: Transition
Title: Understanding the Basic Refrigeration Cycle
Content: The refrigeration cycle has four main components...
```

`Block ID:` is a descriptive label, not a key — row order comes from file
position. Blocks may be deleted, merged, inserted, or reordered freely.

### `context/topic_NN_<slug>.md`

```
###LO ID: 0
####**Topic:**
HVAC Fundamentals
####**Subtopic:**
Refrigeration Cycle
####**Learning Objective:**
Trace how refrigerant state changes drive heat transfer through the cycle.
####**Research Notes:**
The refrigeration cycle moves heat, not creates cold...
```

**Topic:, Subtopic:, and Learning Objective: are the write-back identity
key — never edit them.** `commit_context.py` matches each block back to a
sheet row by those three fields, verified unique on a live course sheet even
though (Topic, Subtopic) alone is not (a subtopic can carry several LOs).
Edit only the `Research Notes:` body. Deleting a whole block does not blank
that row's notes — it means no change was requested for it. See
`references/parsing-rules.md` for the full contract and why the old
`---`-joined format was unsafe.

## Scripts

### `prepare_workspace.py`

```bash
python scripts/prepare_workspace.py \
  --sheet-url "<URL>" \
  --workspace /mnt/session/outputs/workspace \
  [--source-tab "Slide Chunks"] \
  [--outline-tab "Final Outline"] \
  [--target-tab "Slide Chunks (Revised)"] \
  [--outline-target-tab "Final Outline (Revised)"]
```

Writes `topics/*.md`, `context/*.md`, and `manifest.json`. **Requires a valid
slide-chunks tab** — exits with an error if `--source-tab` doesn't exist or is
empty, even if the session never ends up editing a slide chunk. If the outline
tab is empty or missing (but the slide-chunks tab is fine), context files
contain `"Context unavailable."` instead of failing — tell the user you're
editing without research-notes context before making scope judgements on
slide edits.

### `prepare_context_workspace.py`

```bash
python scripts/prepare_context_workspace.py \
  --sheet-url "<URL>" \
  --workspace /mnt/session/outputs/workspace \
  [--outline-tab "Final Outline"] \
  [--outline-target-tab "Final Outline (Revised)"]
```

For research-notes-only sessions. Writes only `context/*.md` and
`manifest.json` (`"mode": "context_only"`) — no `topics/` directory, no
dependency on any slide-chunks tab. Same `context/*.md` block format as
`prepare_workspace.py` produces, so `commit_context.py` works identically
against either workspace.

### `commit_workspace.py`

```bash
python scripts/commit_workspace.py --workspace /mnt/session/outputs/workspace [--target-tab "Slide Chunks (Revised)"]
```

Parses `topics/*.md`, writes a **new** tab. The source slide-chunks tab is
never modified.

### `commit_context.py`

```bash
python scripts/commit_context.py --workspace /mnt/session/outputs/workspace [--outline-target-tab "Final Outline (Revised)"]
```

Re-reads the *live* `Final Outline` tab fresh, overwrites only the matched
rows' `research_notes` cell, writes the full result (all ~20+ original
columns, only the touched cells changed) to a **new** tab. The source outline
tab is never modified. Reports unmatched/ambiguous blocks to stderr — relay
those to the user; they mean an identity field drifted or a row no longer
exists, and the corresponding edit was **not** written.

It also warns (not blocks) whenever it's about to write an **empty**
`research_notes` value for a row that previously had content — that's the
legitimate way to deliberately blank a row's notes, but it's also exactly
the shape a dropped `####**Research Notes:**` header produces (see "When
your edit tool's old_string doesn't match" in the editing-course-content
skill). The script can't tell those apart, so it always flags it; relay the
warning and let the user confirm the row is supposed to be empty before
trusting the write.

### `list_tabs.py` / `check_auth.py`

Same as the checklist agent's — see their own output for usage.

## Authentication

Same credential resolution as the checklist skill: Files API mount at
`/uploads/service_account.json` (Managed Agents), or `GDRIVE_SA_JSON` /
`GDRIVE_SA_KEY_PATH` / `/mnt/secrets/service_account.json` for local
development. Relay any `ERROR: service account credentials not found...`
verbatim — it's a deployment issue, not something to retry.

## Dependencies

`gspread`, `pandas`. No LLM parsing — everything here is deterministic regex.

## Failure modes worth knowing

- **Sheet not shared with the service account.** `check_auth.py` reports this
  directly with the SA email to share with.
- **Tab missing.** `prepare_workspace.py` lists available tabs and asks for
  the right `--source-tab` / `--outline-tab`.
- **commit_context.py: unmatched block.** The block's Topic/Subtopic/Learning
  Objective don't match any row in the live sheet — usually means an identity
  field was edited by mistake, or the row was deleted/renamed upstream since
  `prepare_workspace.py` ran. That row's notes are left untouched; tell the
  user which block didn't land.
- **commit_workspace.py: malformed block.** Skipped with a stderr line
  (file + line number). The final tab is missing that row; warn the user.

See `references/parsing-rules.md` for the exact contract both commit scripts
use.
