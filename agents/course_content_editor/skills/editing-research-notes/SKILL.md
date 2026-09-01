---
name: editing-research-notes
description: Edit the research notes of a course workspace (context/*.md) — rewrite, tighten, trim to the learning objective, correct, expand, or clear a note. Use for any request about research notes, notes, or learning objectives, even when the user doesn't say "edit".
---

# Editing Research Notes

You edit `context/*.md` in a workspace produced by the working-with-google-sheets
skill. Read that skill first if no workspace is prepared yet.

Slide chunks (`topics/*.md`) are a different file type with different rules —
use the editing-slide-chunks skill for those. Writing-style guides do not apply
here; research notes are working material, not learner-facing prose.

## Scope of your mandate

The user names a problem; you fix that problem. Do not scan for issues that
weren't raised, and do not bundle in a second change because you noticed it
while you were in there. If you spot something genuinely wrong outside the
request, say so in one sentence at the end and let the user decide.

When the request *is* a sweep — "tighten every note to its objective" — then
judging every note against that request is the assignment; do it thoroughly.
What stays off-limits is widening the criteria to things the request didn't
name. This holds equally when the request reaches you as a task from a
coordinating copy of yourself: a brief that names a goal rather than specific
notes is asking you to do the reviewing, on the file you were given, within the
goal you were given.

## File format

```
###LO ID: 0
####**Topic:**
...
####**Subtopic:**
...
####**Learning Objective:**
...
####**Research Notes:**
The refrigeration cycle moves heat, not creates cold...
```

**`Topic:`, `Subtopic:`, and `Learning Objective:` are read-only.** They are the
identity `commit_context.py` uses to find the matching row in the live sheet.
Edit only the text under `Research Notes:`. Change an identity field and that
block silently fails to write back — the tool warns, but don't rely on it.

`Research Notes:` is greedy to the end of the block — every body line must sit
under it. Full contract:
`/workspace/skills/working-with-google-sheets/references/parsing-rules.md`.

Deleting a whole `###LO ID:` block does **not** blank that row on commit; it
means "no change requested." To clear a row's notes, keep the block and empty
the `Research Notes:` body.

## Making edits

Use your native text editor tool for every change. Do not fall back to
hand-rolled Python (`open()`/`write()`, regex, line-slicing) to make an edit
you could make with the editor: reconstructing a block by hand is how a
required header line like `####**Research Notes:**` gets dropped, which then parses as
the field being absent rather than edited.

The workspace is normalized to straight quotes when it is prepared, so an
`old_string` you copy from a block you just read will match. If one doesn't,
re-read the block and work from the exact text you see — don't guess at the
punctuation.

## Priors: what these requests usually mean

Use these to resolve a directional request — *"tighten this," "it rambles,"
"make it match the objective"* — the way this team resolves it. They are not a
checklist and license nothing the user didn't ask for. In particular, do not
trim scope on your own initiative: these tell you *how* a requested trim should
go, not that one is due.

**The Learning Objective is the scope boundary.** Including its parenthetical
qualifiers. Notes overrun it rather than getting it wrong — reviewers struck
roughly 45% of a typical note, almost none of it for inaccuracy.

**Trailing bolt-on sections are the highest-frequency deletion.** Notes
reflexively end with a section on maintenance, installation practice, design
principles, troubleshooting, tips, best practices, or "why it matters" — struck
in full six times out of nine notes. Watch for headings shaped like
*Maintaining…, Installation…, Tips, Best Practices, Why It Matters, Types of….*

**Quantification the objective doesn't call for.** PSI ranges, pipe diameters,
fractions, temperatures. The objective verbs in this corpus are recognition-level
— *trace, point out, connect, relate, state, recognize* — and default to zero
quantification. The fix is usually trimming a **modifier**, not deleting the
sentence: cut "¾-inch", keep the trunk.

**Terminology inflation.** Trade synonyms, formal classifications, and "you
might hear this called…" asides are near-always cut. Every new term is load the
objective has to justify.

**Cross-lesson bleed.** Notes get written as standalone articles and teach the
neighboring subtopic's material too. Before keeping an explanation, check the
manifest and the other context files — if a concept is another subtopic's job,
it doesn't belong here.

**Register drift into field training.** "As a technician, you can…", "Tech
Tip:", callback and reputation framing, advice to give customers, field
workarounds. The objectives are knowledge-level; the register stays third-person
and conceptual.

**Source-shaped output.** Content mirroring the structure of a retrieved source
rather than the objective — reproducing a diagram's full numbered legend when
the objective names four items.

## Every turn

1. Say in one sentence what you're about to change. For anything sweeping —
   more than a few notes, or a whole file — get a yes before you write.
2. Make the edit.
3. If you touched more than two notes or more than one file, re-read each
   edited file and confirm item by item that every change landed. Fix what
   didn't before reporting.
4. Report: notes edited, what was removed, anything you chose not to do.

Keep status updates short — the user is watching a chat, not reading a report.

## Writing back

Only when the user explicitly asks you to write, commit, or push to the sheet.
Run `commit_context.py`. It writes to a **new** tab; source tabs are never
modified. Relay any unmatched or malformed-block warnings verbatim — those rows
were not written.
