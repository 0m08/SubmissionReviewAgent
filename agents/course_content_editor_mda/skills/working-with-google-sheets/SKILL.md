---
name: working-with-google-sheets
description: Load a course's slide chunks and research notes from a Google Sheet into local markdown files for editing, then write edits back to new tabs. Use whenever the user pastes a Google Sheet URL, or asks to edit slide chunks or research notes for a course.
---

# Working with Google Sheets

Round-trips a course between a Google Sheet and a local workspace of markdown
files: slide chunks (`topics/*.md`) and research notes (`context/*.md`). The
editing-slide-chunks and editing-research-notes skills edit those files;
this skill handles I/O on both ends.

Forked from the checklist agent's `working-with-google-sheets` skill. The
slide-chunk half is unchanged. The research-notes half is new — the checklist
agent never had to write research notes back, only read them for context.

## One filesystem — use absolute paths everywhere

Your file tools (`read_file`, `edit_file`, `write_file`, `ls`, `glob`, `grep`)
and your `execute` shell see the same sandbox disk, so `/workspace/topics/topic_01.md`
is the same file in both. Always write the absolute form, in `execute` too.

Never rely on the shell's working directory. It is not `/` — it is whatever the
sandbox image sets (currently `/root`), so a relative `workspace` in a command
lands somewhere your file tools will not find it.

This skill's scripts live at `/opt/cce/scripts/` and are called with that path
from `execute`.

They are deliberately **not** under `/skills/`. That path is a Context Hub
mount, which `read_file` can see and the shell cannot — a script you can read is
not a script you can run.

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

**Always pass `--workspace /workspace`** — absolute, with the leading slash —
to every script here, and never any other path. It is where your file tools
expect the files and the directory `present.py` reads by default, so a workspace
built somewhere else cannot be read, shown to the user, or measured. Every
script here works identically regardless of path, so nothing warns you when
this is wrong; the files simply are not where you look.

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
- **Ignore the `gid` in a pasted URL.** A sheet link copied from a browser
  carries `?gid=…#gid=…` for whatever tab happened to be open. It is an artifact
  of copying, not a request to work on that tab, and the scripts take a tab
  *name* — `--source-tab`, `--outline-tab`, `--target-tab` — never a gid. There
  is no supported way to resolve one, and you do not need one: take the tab name
  from what the user said, or from `list_tabs.py`, or leave the default. If the
  gid and the user's words seem to disagree, that is not a discrepancy to
  investigate — the words win. Reconciling a gid against the sheet is the exact
  situation that tempts a hand-rolled `gspread` call, which is forbidden above.

## Workspace layout

```
<workspace>/
├── manifest.json                    # sheet URL, tab names, topic→file map
├── outline.md                       # every learning objective, LOs only, whole course
├── topics/topic_NN_<slug>.md        # slide chunks — one or more ###Block ID: blocks
└── context/topic_NN_<slug>.md       # research notes — one or more ###LO ID: blocks
```

### `outline.md`

Generated, read-only, one per workspace. Every learning objective in the
course, grouped by topic and subtopic, each topic naming the slide file its
objectives scope — and each objective reproduced as written, including the
parenthetical scope note the instructional designer put beside it.

It carries no research notes, which is the point: an editor needs a topic's
scope before its first edit, and should not read a research-notes corpus to
find it. The same objectives do appear in `context/*.md`, but there they are
part of the write-back identity key; read them from here, where there is
nothing to break. Nothing parses this file back, so nothing you change in it
reaches the sheet.

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
python /opt/cce/scripts/prepare_workspace.py \
  --sheet-url "<URL>" \
  --workspace /workspace \
  [--source-tab "Slide Chunks"] \
  [--outline-tab "Final Outline"] \
  [--target-tab "Slide Chunks (Revised)"] \
  [--outline-target-tab "Final Outline (Revised)"]
```

Writes `topics/*.md`, `context/*.md`, `outline.md`, and `manifest.json`. **Requires a valid
slide-chunks tab** — exits with an error if `--source-tab` doesn't exist or is
empty, even if the session never ends up editing a slide chunk. If the outline
tab is empty or missing (but the slide-chunks tab is fine), context files
contain `"Context unavailable."` instead of failing — tell the user you're
editing without research-notes context before making scope judgements on
slide edits.

**Re-preparing into a workspace that has edits in it is refused.** Both prepare
scripts compare the workspace against `.baseline` first. If anything differs,
they write nothing, exit 3, and name the drifted files with their word counts.
This matters because a prepare replaces the baseline as well as the files, so an
overwrite would destroy the edit *and* the only record that there was one.

The error names three ways forward: prepare into a second workspace
(`--workspace /workspace_2`) and keep both, commit the edits first, or discard
them on purpose with `--discard-local`. Relay it and let the user pick. A second
workspace is self-contained — pass the same `--workspace` to `present.py` and
the commit scripts and it behaves exactly like the first.

Reaching for `--discard-local` because the script said no is not a fix; it is
the destructive option, and it is there to be chosen deliberately.

### `prepare_context_workspace.py`

```bash
python /opt/cce/scripts/prepare_context_workspace.py \
  --sheet-url "<URL>" \
  --workspace /workspace \
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
python /opt/cce/scripts/commit_workspace.py --workspace /workspace [--target-tab "Slide Chunks (Revised)"]
```

Parses `topics/*.md`, writes a **new** tab. The source slide-chunks tab is
never modified.

### `commit_context.py`

```bash
python /opt/cce/scripts/commit_context.py --workspace /workspace [--outline-target-tab "Final Outline (Revised)"]
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
your edit tool's old_string doesn't match" in the editing-research-notes
skill). The script can't tell those apart, so it always flags it; relay the
warning and let the user confirm the row is supposed to be empty before
trusting the write.

### `list_tabs.py` / `check_auth.py`

Same as the checklist agent's — see their own output for usage.

`list_tabs.py` marks each tab with what its header row says it holds: `slide
chunks`, `outline`, `slide chunks (backup/output)` for backups and `(Revised)`
tabs, or nothing. It also marks which tabs are the defaults. Read the marks;
they are read off the sheet, not guessed from the name.

**When it says the source tab is ambiguous, ask.** Several tabs can hold slide
chunks at once — the generator's `Slide Chunks`, plus per-topic tabs somebody
split out by hand. They are all valid sources and the names do not say which one
is meant. The script prints the candidates when there is more than one; name
them back to the user and let them choose. Taking the default in that situation
is how a whole turn gets spent editing content nobody asked about — which has
happened: the user meant `Topic 1: Pipe Preparation` and got `Slide Chunks`,
silently.

One candidate, or none, means no question to ask. Do not ask anyway.

## The commands, in full

Copy these. Square brackets mark optional arguments, with their defaults shown —
so anything unbracketed is required and anything bracketed can be left out.

```bash
# 1. Tab names vary per course sheet. Check rather than assume.
python /opt/cce/scripts/list_tabs.py --sheet-url "<SHEET_URL>"

# 2a. Research notes only. No slide-chunks tab needed.
python /opt/cce/scripts/prepare_context_workspace.py \
    --sheet-url "<SHEET_URL>" --workspace /workspace \
    [--outline-tab "Final Outline"] \
    [--outline-target-tab "Final Outline (Revised)"]

# 2b. Slide chunks, and the research notes alongside them.
python /opt/cce/scripts/prepare_workspace.py \
    --sheet-url "<SHEET_URL>" --workspace /workspace \
    [--source-tab "Slide Chunks"] [--outline-tab "Final Outline"] \
    [--target-tab "Slide Chunks (Revised)"] \
    [--outline-target-tab "Final Outline (Revised)"]

# 3. Show the user what you changed. Run this after editing, unprompted.
python /opt/cce/scripts/present.py --workspace /workspace \
    context/topic_01_<slug>.md [--blocks 3,7] [--note "what to look at"]

# 4. Counts only — this output is for you, before and after a
#    tighten / trim / expand request.
python /opt/cce/scripts/present.py --workspace /workspace --measure \
    topics/topic_02_<slug>.md

# 5a. Write research notes to a new tab.
python /opt/cce/scripts/commit_context.py --workspace /workspace \
    [--outline-target-tab "<NEW_TAB_NAME>"]

# 5b. Write slide chunks to a new tab.
python /opt/cce/scripts/commit_workspace.py --workspace /workspace \
    [--target-tab "<NEW_TAB_NAME>"]

# 6. If a sheet call fails, confirm access before retrying anything.
python /opt/cce/scripts/check_auth.py --sheet-url "<SHEET_URL>"
```

**Name the target tab when you commit.** Both commit scripts fall back to the
tab recorded when the workspace was prepared — `Slide Chunks (Revised)` or
`Final Outline (Revised)` — so leaving the flag off does not mean "no tab", it
means "that one", and a second commit overwrites the first. Pass the name the
user asked for.

## Showing your work: `present.py`

**Never write file content into your message.** Not a rewritten slide, not a
before/after pair, not a block quote of a research note. Run `present.py`
instead:

```
python /opt/cce/scripts/present.py     --workspace /workspace context/topic_01_<slug>.md     [--blocks 3,7] [--note "what to look at"]
```

**Its output goes to the user's screen, not to you.** The client renders it as
before/after cards; you get back counts and an acknowledgement. So do not repeat
what it showed — you will not usually see it, and content that passes through
your context is content that can come back paraphrased. Say what you changed and
why in your own words; the content itself is already on screen.

`--measure` is the exception: counts only, no content, and that output *is* for
you.

Call it without being asked once you finish editing, and again whenever the user
asks to see, read, check or confirm anything.

It also prints counted facts — blocks before/after, words before/after with the
percentage change, a `per_block` list giving each block's id, title and word
count, and for slide chunks a census of slide types. **Every number you state
must come from that output.**

`per_block` is what answers a per-slide length rule. A limit like "keep slides
under 60 words" is a limit on each block, and the file total cannot tell you
whether any single block breaks it — read `per_block` and look. Do not write a
script to count words; this is that script, and one already ran.

`--measure` gives the counts alone, with no diff:

```
python /opt/cce/scripts/present.py     --workspace /workspace --measure topics/topic_02_<slug>.md
```

Run it before and after any request naming a direction — tighten, trim, expand,
fewer slides — because a rewrite that reads tighter is routinely longer than
what it replaced. You have a shell, so `wc -w` also works for a quick check, but
only `present.py` counts *blocks*, which is what catches an accidental deletion.

## Authentication — there is no key here, and that is deliberate

Do not go looking for a service account file or a credential environment
variable. There is none in this sandbox, on purpose: an instruction injected
through a course sheet could otherwise ask you to read it out.

Sheet access works anyway. A proxy outside the sandbox intercepts calls to
Google APIs and fills in the `Authorization` header on the way out, resolving
the credential from a workspace secret you never see. The scripts send a bare
request — `gspread.authorize(AnonymousCredentials())` — and the header appears
in flight. Nothing for you to configure or pass.

The one consequence worth knowing: a `401`/`403` from a script is *not*
necessarily a sharing problem here. It could equally be a proxy or workspace
secret misconfiguration, and from inside the sandbox those look identical.
Relay the error verbatim with both possibilities named rather than guessing, and
do not retry — neither cause resolves on a second attempt.

## Dependencies

`gspread`, `pandas`, `google-auth` — installed into the sandbox image by
`sandbox/setup.sh` at deploy, so they are already present. No LLM parsing — everything here is deterministic regex.

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
