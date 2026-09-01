---
name: editing-slide-chunks
description: Edit the slide chunks of a course workspace (topics/*.md) — change, fix, trim, expand, merge, split, reorder, retitle, or delete slides, and rewrite slide prose in a named course writing style. Use for any request about slides, decks, or topics, even when the user doesn't say "edit".
---

# Editing Slide Chunks

You edit `topics/*.md` in a workspace produced by the working-with-google-sheets
skill. Read that skill first if no workspace is prepared yet.

Research notes (`context/*.md`) are a different file type with different rules —
use the editing-research-notes skill for those. Never apply the priors below to
research notes.

## Scope of your mandate

The user names a problem; you fix that problem. Do not scan for issues that
weren't raised, and do not bundle in a second change because you noticed it
while you were in there. If you spot something genuinely wrong outside the
request, say so in one sentence at the end and let the user decide.

When the request *is* a sweep — "review this topic and tighten it," "give the
transitions real hooks" — then judging every block against that request is the
assignment; do it thoroughly. What stays off-limits is widening the criteria to
things the request didn't name. This holds equally when the request reaches you
as a task from a coordinating copy of yourself: a brief that names a goal rather
than specific blocks is asking you to do the reviewing, on the file you were
given, within the goal you were given.

## Preserve what you were not asked to change

Editing is subtraction only where subtraction was requested. Unless the user
asked for it:

- **Never drop an image, diagram, or link.** If a block carrying
  `![...](...)` is merged or rewritten, the markdown travels with it.
- **Never remove a component, term, or concept entirely from the topic.**
  Merging two slides means combining their content, not discarding one side.
- **Never strip specifications, units, or trade terms** on your own initiative.

If an edit you were asked to make genuinely forces something out, say which
thing and why, in the same message that reports the edit.

## File format

```
###Block ID: 0
####**Topic:**
...
####**Subtopic:**
...
####**Slide Chunk:**
Slide Type: Transition
Title: The Push Side
Content: Every time you open a tap...
```

Field order inside `Slide Chunk:` must be `Slide Type:`, `Title:`, `Content:`.
`Content:` is greedy to the end of the block — every body line must sit under
it, and a line without that prefix is silently dropped. Full contract:
`/workspace/skills/working-with-google-sheets/references/parsing-rules.md`.

Blocks may be freely deleted, merged, inserted, and reordered. `Block ID:` is a
label, not a key — do not renumber. Do not preserve block count out of caution;
reducing it is often the correct edit.

## Making edits

Use your native text editor tool for every change. Do not fall back to
hand-rolled Python (`open()`/`write()`, regex, line-slicing) to make an edit
you could make with the editor: reconstructing a block by hand is how a
required header line like `####**Slide Chunk:**` gets dropped, which then parses as
the field being absent rather than edited.

The workspace is normalized to straight quotes when it is prepared, so an
`old_string` you copy from a block you just read will match. If one doesn't,
re-read the block and work from the exact text you see — don't guess at the
punctuation.

## Priors: what these requests usually mean

Use these to resolve a directional request — *"tighten this," "too many
slides," "this feels off"* — the way this team resolves it. They are not a
checklist and license nothing the user didn't ask for.

**The generator's problem is scope and packaging, not accuracy.** Across six
reviewed courses, reviewers disputed exactly one fact. When a user says
something is wrong, they want fewer containers or less scope, not corrections.
Do not go hunting for factual errors unless asked.

**Merging beats shortening.** Whole-course reviews ran 34 → 21 and 69 → 36
slides while total text fell only ~12%. Slides get sliced too thin — one idea
split across three slides carrying a sentence each. Reach for merge before cut.
When you merge, write the bridge sentence; don't just concatenate two bodies.

**Transition and Content slides fail in opposite directions.** Check
`Slide Type:` before deciding which way to push:

| Slide type | Its job | Typical defect | Default direction |
|---|---|---|---|
| Transition / hook | Situate the learner, pose the question | Thin, flat, no curiosity | **Expand** |
| Content | Carry one LO-bearing idea | Over-detailed, over-split | **Restrain** |
| Summary | Synthesize the takeaway | Re-lists every term covered | **Synthesize, don't itemize** |

"Make this better" on a Transition slide usually means give it *more*.

**Titles churn.** In one whole-course pass, 1 of 34 titles survived verbatim.
Reviewers rewrite titles to name the idea the slide carries rather than to be
catchy. After a merge, expect to write a new title rather than keeping the
first slide's.

**Filler slides.** A slide with no LO-bearing content of its own is a merge or
delete candidate — but flag it and let the user decide, since slide count is
sometimes a hard requirement on their side.

**Only when the user asks for scope work**, these also apply: trailing bonus
slides (*Maintaining…, Tips, Best Practices, Why It Matters*) are usually cut;
quantification the learning objective doesn't call for is usually trimmed as a
modifier rather than by deleting the sentence; and a slide teaching a
neighboring subtopic's material belongs to that subtopic. Do not act on these
unprompted — they are reasons a *requested* trim goes one way rather than
another.

## Writing style

Six style guides are available, each extracted from one course's
manually-corrected slides and backed by verbatim quotes. **Read the full guide
before rewriting prose in that voice** — don't work from the label alone.

Always refer to a style by its label, never by the source course it came from.
A user may point at a style by naming its course; answer back using the label.

| Style label | File | One-line description |
|---|---|---|
| Guided Walkthrough | `references/writing-styles/guided-walkthrough.md` | Second-person narration following one concrete scenario step by step through a system; short declarative sentences, almost no formatting. |
| Direct Trade Explainer | `references/writing-styles/direct-trade-explainer.md` | Matter-of-fact component-by-component walkthrough, trade terms named plainly, most slides closing with a forward-pointing sentence. |
| Trade Mentor | `references/writing-styles/trade-mentor.md` | A working plumber explaining tools to an apprentice; third-person "plumbers do X", each mini-topic opening on an everyday scenario and closing by teeing up the next. |
| Scenario-Driven Technician Voice | `references/writing-styles/scenario-driven-technician-voice.md` | Drops a plumber into a specific job scenario, then walks tool-by-tool through what to grab and why, addressed as "you." |
| Direct-Address Field Guide | `references/writing-styles/direct-address-field-guide.md` | Speaks to the technician as "you" in the moment of the job, moving from a real-world trigger to the plain fact they need. |
| Plain Sequential Descriptive | `references/writing-styles/plain-sequential-descriptive.md` | Third-person walkthrough one connection at a time, stitched with simple sequencing words ("Next," "Then," "Finally"). |

**Which style applies, in order:**

1. **The user names one** — by label or by source course. This wins outright.
2. **A `/preferred_style.md` memory** in the per-user memory store, if one is
   mounted and the user hasn't overridden it this session.
3. **Otherwise, ask.** Name the labels and ask which to write in, before
   rewriting prose. Inferring silently produces a different answer on different
   runs of the same sheet, and a wrong guess means rewriting twice.
4. **If the user says "you decide":** infer from slides you are *not* editing —
   compare their orthography, voice, and formatting against the guides — then
   say which you picked and why in one sentence. If nothing matches
   confidently, say so and match the sheet's own tone without forcing it into a
   labeled style.

Fastest tells when inferring: numbers spelled out even for measurements →
Guided Walkthrough. (The style guides mention curly apostrophes as a tell for
Plain Sequential Descriptive — ignore that one. The workspace is normalized to
straight quotes when it is prepared, so quote shape carries no signal here.
Judge by voice, sentence shape, and hook pattern instead.) "You" as the default voice → Scenario-Driven
(opens on a job scenario) or Direct-Address (opens on a servicing trigger).
Third-person with occasional semicolons → Direct Trade Explainer or Trade
Mentor; Trade Mentor leans on trade-wisdom asides, Direct Trade Explainer on
plain component walkthroughs.

Style governs prose only. It never overrides the format contract above or an
explicit instruction from the user.

## Every turn

1. Say in one sentence what you're about to change, including before → after
   block count when it changes. For anything sweeping — more than a few blocks,
   or a whole topic — get a yes before you write.
2. Make the edit.
3. If you touched more than two blocks or more than one file, re-read each
   edited file and confirm item by item that every change landed. Fix what
   didn't before reporting.
4. Report: blocks edited, removed, added; anything you chose not to do; anything
   that had to be dropped and why.

Keep status updates short — the user is watching a chat, not reading a report.

## Writing back

Only when the user explicitly asks you to write, commit, or push to the sheet.
Run `commit_workspace.py`. It writes to a **new** tab; source tabs are never
modified. Relay any unmatched or malformed-block warnings verbatim — those rows
were not written.
