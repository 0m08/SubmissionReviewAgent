---
name: editing-course-content
description: Apply a user's plain-language edit request to a course workspace — slide chunks (topics/*.md) or research notes (context/*.md). Use whenever the user asks to change, fix, trim, expand, merge, split, reorder, retitle, or delete slides, or to rewrite, tighten, correct, or remove research notes — even if they don't say "edit".
---

# Editing Course Content

You are making the exact change the user asked for, to a workspace produced
by the working-with-google-sheets skill. Read that skill first if you haven't
prepared a workspace yet.

## You are a reactive editor, not a reviewer

The user names a problem; you fix that problem. Do not scan for issues that
weren't raised. Do not bundle a second change into the same edit because you
noticed it while you were in there. If you see something genuinely wrong
outside the request, say so in one sentence at the end and let the user
decide.

## Two file types, two formats, one shared rule

Both `topics/*.md` (slide chunks) and `context/*.md` (research notes) are
block-marker files edited with your native text editor tools. In both, field
order is load-bearing and the last field is greedy to the end of the block —
get this wrong and content silently vanishes or gets swallowed into the wrong
field. Full contract: `/workspace/skills/working-with-google-sheets/references/parsing-rules.md`.

### Slide chunks (`topics/*.md`)

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
`Content:` swallows everything after it. Every body line must live under
`Content:` — a line without that prefix is silently dropped.

Blocks may be freely deleted, merged, inserted, and reordered. `Block ID:` is
a label, not a key — do not renumber. Do not preserve block count out of
caution; reducing it is very often the correct edit (see editing-priors.md).

### Research notes (`context/*.md`)

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

**Topic:, Subtopic:, and Learning Objective: are read-only.** They are the
identity `commit_context.py` uses to find the matching row in the live sheet.
Edit only the text under `Research Notes:`. If you change an identity field,
that block silently fails to write back — the tool will warn, but don't rely
on that; just don't touch those three fields.

Deleting a whole `###LO ID:` block does **not** blank that row's notes on
commit — it means "no change requested." If the user wants a row's notes
cleared, keep the block and empty the `Research Notes:` body instead of
deleting the block.

## When your edit tool's old_string doesn't match

Real course content routinely contains smart/curly quotes (’ " ") and other
Unicode punctuation, not the plain ASCII versions — expect this, it's normal
for this data, not a sign anything is corrupted. If your text-editor tool
reports the old text wasn't found, that mismatch is the most likely cause,
not a typo in your line numbers.

**Don't fall back to hand-rolled Python (line-slicing, regex substitution,
etc.) to work around it.** That path is how a required block header
(`####**Research Notes:**`, `####**Slide Chunk:**`) gets silently dropped —
the replacement text you write in has to reconstruct the *entire* block
structure by hand, and it's easy to reconstruct everything except the header
line, which then parses as if the field were simply absent rather than
edited. Instead, re-read the block to see the exact characters actually
there, and do a full-block replacement through your native text editor —
that keeps the header lines mechanically intact because you're replacing a
block you just read, not retyping one from memory.

## Where the editing priors come from

`references/editing-priors.md` is distilled from a review of six SkillCat
plumbing courses (9 research-note rows and 64 of 331 slide rows carrying
instructional-designer feedback). Read it when a request is directional but
underspecified — *"tighten this," "too many slides," "this feels off"* — to
resolve the ambiguity the way this team historically has.

**It is not a checklist.** Nothing in it licenses an edit the user didn't
ask for. Two findings worth knowing before your first edit of a session:

- The dominant slide edit is **merging**, not shortening (34→21, 69→36 in
  whole-course reviews). When someone says a topic is bloated, reach for
  merge before cut.
- Transition/hook slides and Content slides fail in *opposite* directions —
  Transitions run thin and want expanding, Content slides run over-detailed
  and want restraining. Check `Slide Type:` before deciding which way to push.
- At the research-notes stage, the highest-frequency edit is deleting
  trailing bolt-on sections (maintenance, installation practice, "why it
  matters," best practices) that aren't in scope for the Learning Objective.

## Matching a course's writing style

Different SkillCat courses were written (and corrected) in genuinely
different voices — narrative vs. clinical vs. procedural-imperative, British
vs. American spelling, some allow meta-narration openers ("In this topic we
will look at…"), some ban it. Six style guides are available, each extracted
from one course's manually-corrected slide chunks and backed by verbatim
quotes (orthography, sentence/paragraph shape, voice, hooks, transitions,
titles, formatting, plus 3 full worked examples). **Read the full guide
before rewriting prose in that voice — don't work from the label alone:**

**Always refer to a style by its label** ("Direct Trade Explainer," "Trade
Mentor," etc.) when talking to the user or reasoning out loud about which one
applies — never by the name of the course it happened to be extracted from.
The style is what's being reused; the source course is just provenance
(recorded inside each file for traceability, not a name to speak in). A user
may still name the source course to point at a style ("write it like the
Fundamentals course") — recognize that, but answer back using the label.

| Style label | File | One-line description |
|---|---|---|
| Guided Walkthrough | `references/writing-styles/guided-walkthrough.md` | Plain, second-person narration that follows a single concrete scenario (a drop of water, a flushed toilet) step by step through a system, using short declarative sentences and almost no formatting. |
| Direct Trade Explainer | `references/writing-styles/direct-trade-explainer.md` | Plain, matter-of-fact prose that walks a technician component-by-component through a fixture, naming trade terms plainly and closing most slides with a forward-pointing "here's what's next" sentence. |
| Trade Mentor | `references/writing-styles/trade-mentor.md` | A working plumber explaining tools to an apprentice — short, plain, third-person "plumbers do X" sentences that open each mini-topic with an everyday scenario or rhetorical question and close it by teeing up the next. |
| Scenario-Driven Technician Voice | `references/writing-styles/scenario-driven-technician-voice.md` | Short, plain-spoken paragraphs that drop a working plumber into a specific job scenario, then walk them tool-by-tool through what to grab and why, addressed directly as "you." |
| Direct-Address Field Guide | `references/writing-styles/direct-address-field-guide.md` | Short, single-topic-per-slide prose that speaks to the technician as "you" in the moment of doing the job, moving from a real-world trigger ("when you're servicing...") to the plain fact the technician needs to know. |
| Plain Sequential Descriptive | `references/writing-styles/plain-sequential-descriptive.md` | Short, plain-spoken, third-person prose that walks through a fixture or component one connection at a time, stitched together with simple sequencing words ("Next," "Then," "Finally," "Because"). |

**Quick-differentiate cheat sheet** — fastest tells if you're inferring a
style from a sheet's own untouched slides (step 4 below), cheapest to check
first:

- **Curly apostrophes (’) instead of straight (')** → almost certainly
  **Plain Sequential Descriptive** — the only one of the six that isn't
  straight-quote by default.
- **Numbers spelled out, never digits, even for measurements** →
  **Guided Walkthrough**. Every other style uses digits for specs/measurements.
- **No numbers/measurements at all in the sample** → could be **Plain
  Sequential Descriptive** (0 numerals found in its whole corpus) — check for
  curly apostrophes too before committing.
- **Direct "you" address is the *default* voice, not occasional** →
  **Scenario-Driven Technician Voice** or **Direct-Address Field Guide**.
  Distinguish by hook style: Scenario-Driven drops the reader into a job
  scenario ("You're installing a new fixture..."); Direct-Address opens with
  a servicing-moment trigger ("When you're servicing a residential plumbing
  system...").
- **Third-person, semicolons used occasionally to join two independent
  clauses** → **Direct Trade Explainer** or **Trade Mentor**. Distinguish by
  whether tool names get a plain-language alias clause on first mention
  ("a basin wrench, also called a sink or faucet wrench") — that's
  Scenario-Driven Technician Voice, not these two. Trade Mentor leans more
  toward trade-wisdom asides/rhetorical questions; Direct Trade Explainer
  leans more toward plain component-by-component walkthroughs.

**A note on isolated British-spelling slips:** three of the six guides
(Guided Walkthrough, Plain Sequential Descriptive, Scenario-Driven Technician
Voice) each independently flag exactly *one* stray British spelling
("equalise," "Recognising," "colouring") in an otherwise all-American corpus.
That repeated single-slip pattern across otherwise-unrelated courses is more
consistent with a residual quirk from the *original AI-generated draft*
leaking through one uncaught sentence than a trait of the human editors who
corrected the rest — don't read it as house style in any of the three
guides; default to American spelling regardless.

**Picking which style applies, in this order:**

1. **The user names one.** If they say "write it like the Fundamentals
   course" or name a style label directly ("Direct Trade Explainer"), use
   that — it overrides everything below.
2. **Check the per-user memory store** (see Using memory, below) for a
   `/preferred_style.md` memory. If present and the user hasn't overridden it
   this session, use it.
3. **For slide-chunk work, ask before you start reviewing or editing** if
   neither 1 nor 2 resolved a style: name the styles table above (labels
   only) in one short message and ask which one to write in, before you read
   slide content for the purpose of matching voice. Don't silently infer and
   proceed — a wrong guess means rewriting later. For research notes this
   matters far less (notes are rarely rewritten in a specific "voice"), so
   asking is optional there.
4. **If the user says "you decide" (or equivalent), or for research-notes
   work where asking felt unnecessary): infer from the sheet's own untouched
   slides.** Read a few slides you are *not* editing this turn and compare
   their orthography/voice/formatting against each style guide's
   fingerprint. If one is a clear match, use it, and say which one you picked
   and why in one sentence.
5. **No confident match:** say so in one sentence, proceed in a neutral
   default voice (match the sheet's own existing tone as closely as you can
   without forcing it into one of the labeled styles), and don't fabricate a
   match — a wrong style guess is worse than no style guess.

This only governs the *voice/prose* of rewritten text — it never overrides
the block-format contract above, the identity-field rule for research notes,
or an explicit instruction from the user that conflicts with the style
guide's usual pattern (the user's literal request always wins).

## Using memory

Two memory stores may be mounted (see their `instructions` in your system
context for exact paths — typically `/mnt/memory/cce-shared-editorial-standards/`
and `/mnt/memory/cce-user-<slug>/`): a **shared** store (team-wide editorial
standards, read/write by everyone) and a **per-user** store (this person's
own preferences). Either may be absent — check before assuming a mount
exists, and don't fail an edit just because memory isn't there.

**Division of labor — keep these in disjoint domains, don't duplicate:**
- The **writing-styles reference** (above) owns prose/format conventions for
  a specific *course*.
- The **shared memory store** owns durable, recurring standards that apply
  across *all* courses and users — e.g. "always spell out PSI on first use."
- The **per-user memory store** owns one person's recurring personal
  preferences, including `/preferred_style.md` for their default style label.

**When to write a memory** — this is a judgment call, not a mechanical log:
only write something a person would reasonably expect to still apply next
session. A one-off correction to a specific slide is not memory-worthy —
that's just the edit you already made. Write to memory when the user
expresses a preference in general terms ("I always want you to cut PSI
numbers unless the LO asks for them", "for my courses, never use the word
'technician'") or repeats the same kind of correction more than once in a
session. Say out loud, briefly, when you write one ("Noting that as a
standing preference"), so the user can correct or remove it if wrong.

**Keep memories small and specific** — many short files, not one growing
document. Use a clear path per topic, e.g. `/preferred_style.md`,
`/scope-and-length.md`, `/register.md`, `/terminology.md`. Never write
credentials, sheet URLs, or anything user-identifying beyond what's already
implied by the store being per-user.

## Every turn

1. State in one sentence what you're about to change, including the
   before → after block count if it changes. For anything sweeping — more
   than a few blocks, or a whole topic/file — get a yes before you write.
2. Make the edit.
3. **If this turn touched more than 2 blocks/notes, or spanned more than one
   file** — re-read each edited file and check, item by item, that every
   change you said you'd make is actually present. This is a real failure
   mode, not a hypothetical: measured testing found multi-row requests
   silently landing only partially (a topic needing an 8-block cut ending up
   basically untouched, despite the agent's own stated plan to fix it) with
   no self-check to catch it. For 1-2 block edits this step is unnecessary
   overhead — skip it and just make the edit.
4. Report: blocks/notes edited, removed, added; anything you chose not to do.
   If step 3's check found something that didn't land, fix it before
   reporting done — don't report success on an edit you haven't confirmed.

Keep status updates short — the user is watching a chat, not reading a report.

## Writing back

Only when asked. `commit_workspace.py` for slide-chunk edits,
`commit_context.py` for research-notes edits — run whichever (or both) match
what was actually edited this session. Both write to a **new** tab; the
source tabs are never modified. Relay any unmatched/malformed-block warnings
from the commit scripts to the user — those rows were not written.
