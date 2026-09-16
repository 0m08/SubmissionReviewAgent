---
name: editing-slide-chunks
description: Edit the slide chunks of a course workspace (topics/*.md) — change, fix, trim, expand, merge, split, reorder, retitle, or delete slides, and rewrite slide prose in a course writing style. Use when you are about to change a slide-chunk file, or to inspect one you are about to change, even when the user doesn't say "edit". Not for general discussion about slides, teaching, or writing where no file will be touched — answer those from the standards you already carry.
---

# Editing Slide Chunks

You edit `/workspace/topics/*.md` in a workspace produced by a prepare script
(see working-with-google-sheets). If no workspace is open yet, say so rather than guessing at content.

Research notes (`/workspace/context/*.md`) are a different file type with different rules —
use the editing-research-notes skill for those. Never apply the priors below to
research notes.

## Load the standards before you edit

Before your first edit in a session, read **every** memory file — not only the
one you expect to matter:

```
read_file /memories/agent/AGENTS.md
```

This deployment has one shared memory tree. `/memories/agent/AGENTS.md` holds
team-wide editorial standards that apply across courses and users, and it is
loaded into every run — so as the coordinator you already have it. A delegated
editor should read it anyway rather than assume the brief carried it.

This is not background reading, and skipping it is the most expensive mistake
available here. Measured on a real run: of three delegated editors working the
same sheet, one read both stores and two read none. The two that read none
produced prose contradicting standards the team had already written down — a
component name the shared store explicitly rules out, and a summary opening it
names as an anti-pattern. Nobody was told the rules and disagreed with them;
they were never loaded.

Do this whether you are working alone or as a delegated copy that owns one file.
A brief is not a substitute: it passes on the user's request, not the team's
accumulated standards.

What you load is binding. If a standard names a term to prefer, an anti-pattern
to avoid, or a target to hit, it governs the prose you write, and you check your
output against it before reporting (see *Verify the direction you were asked to
move in*). Loading a rule and then breaking it in the same file has already
happened — that check is what catches it.

## Read the learning objective first

**`/workspace/outline.md`** holds every learning objective in the course, and
nothing else — one file, grouped by topic and subtopic, each topic naming the
slide file its objectives scope:

```
## Topic 03 — Fittings, Valves, and Transitions
Slides: `topics/topic_03_fittings-valves-and-transitions.md`

### Valves
Identify fixture stops, ball valves, gate valves, check valves, and
pressure-control valves by function.
(Frame each valve around its job in the system. The learner should understand
where a valve would be found and what control it gives the technician)
```

The prepare script writes it from the Final Outline tab, reproducing each
objective as written — **including the parenthetical the instructional designer
put beside it**. That parenthetical is the scope instruction and is routinely
more specific than the objective sentence: *"Keep this visual and
field-based"*, *"This should feel like a layout recognition lesson, not a
catalog"*. Read it as closely as the objective itself.

Read `outline.md` before your first edit. It is a generated view — nothing
parses it back, so editing it changes nothing. (The same objectives also appear
inside `context/*.md` next to their research notes, where they are part of the
sheet write-back key. Read them from `outline.md` instead: it is the whole
course, it is a fraction of the bytes, and there is nothing there to break.)

The LO is the boundary of what the learner needs, and it is the one standard
always in force whether or not the request mentions it. A request to fix the
transitions in a topic is a request to fix them *within* that topic's
objectives. Content the LO does not call for is a legitimate cut; content it
does call for is not yours to remove.

If `outline.md` is absent, or your topic's section reads *"No matching rows in
the Final Outline tab"*, say so and edit to the request alone rather than
inventing a scope.

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

The learning objective is not a second request you went looking for. It is the
frame the request is already stated inside, so applying it is not overreach —
it is the one thing you read that the user did not have to say.

## Preserve what you were not asked to change

Editing is subtraction only where subtraction was requested — with one
standing exception, the learning objective (see *Read the learning objective
first*). Content the LO does not call for is always in scope for a cut. These
rules govern everything else:

- **Never drop an image, diagram, or link.** No exception, the LO included.
  If a block carrying `![...](...)` is merged or rewritten, the markdown
  travels with it; if the block it illustrated is gone, move it to the block
  that now carries that idea.
- **Never remove a component, term, or concept the LO asks for.** Merging two
  slides means combining their content, not discarding one side. Cutting a
  component the LO does not name is a different act, and it is allowed.
- **Never strip specifications, units, or trade terms the LO asks for**, and
  never strip one for style alone. A specification the LO does not call for is
  trimmed as a modifier where the sentence still reads without it, and dropped
  outright where it does not.
- **Never invent to fill the space a cut left.** A slide that got shorter is
  finished. Do not add a rationale, a mechanism, a field cue, or an example
  that was not in the source. A plausible sentence you wrote reads exactly
  like one the SME wrote, and only one of them is true.

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
the working-with-google-sheets skill.

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
| Transition / hook | Situate the learner, hand off to what follows | Thin and flat, *or* padded with throat-clearing | **Sharpen, not lengthen** |
| Content | Carry one LO-bearing idea | Over-detailed, over-split | **Restrain** |
| Summary | Synthesize the takeaway | Re-lists every term covered | **Synthesize, don't itemize** |

"Make this better" on a Transition slide means a sharper hook, not a longer
one: two or three purposeful sentences that situate the learner and hand off
to what follows. A flat transition needs a better first sentence, not another
paragraph. Padding one to reach a length is the defect, not the fix.

**Titles churn.** In one whole-course pass, 1 of 34 titles survived verbatim.
Reviewers rewrite titles to name the idea the slide carries rather than to be
catchy. After a merge, expect to write a new title rather than keeping the
first slide's.

**Filler slides.** A slide with no LO-bearing content of its own is a merge or
delete candidate — but slide count is sometimes a hard requirement on their
side, so always flag it. If you can reach the user, let them decide. If you
are a delegated editor you cannot, so make the call the LO supports and report
it as a call you made.

**Only when the user asks for scope work**, these also apply: trailing bonus
slides (*Maintaining…, Tips, Best Practices, Why It Matters*) are usually cut,
and a slide teaching a neighboring subtopic's material belongs to that
subtopic. Do not act on these unprompted — they are reasons a *requested* trim
goes one way rather than another.

Quantification the learning objective doesn't call for is the exception: that
one is always live, because the LO is always live. Trim it as a modifier
rather than by deleting the sentence that carries it.

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
2. **The voice in `/memories/agent/AGENTS.md` is this deployment's default**, and
   it is the normal case. You already have it: shared memory is loaded into every
   run. It describes the house voice directly rather than naming one of the
   labels below, and that is deliberate — it is the most evolved statement of how
   this team writes, and the labelled guides have fallen behind it. Do not go
   looking for a label to match it to, and do not ask which style to use when the
   user has not raised the question. Note it is *shared* — the team's default, not
   one person's preference — so a conflict with what this user just said is the
   user winning.
3. **Ask only when `AGENTS.md` does not cover what was asked for.** A request for
   a voice the house default does not describe — a different register, a named
   course's feel, something the user calls "not our usual" — is the case for
   naming the labels below and asking which. Inferring silently *there* produces a
   different answer on different runs of the same sheet. A delegated editor cannot
   ask, so the coordinator settles this before delegating and names the choice in
   the brief; a brief that names none means the house default.
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

**When a memory standard describes a voice without naming a label**, match it
to the label it describes, and say which you matched. A described voice and a
labeled one are the same thing, and working from the description alone throws
away the guide's worked examples. A voice described as direct second-person
address in short paragraphs of one or two sentences is **Direct-Address Field
Guide**.

**A memory standard outranks a style guide wherever they collide.** Each guide
is an observation of one course, and a formula that was right for that course
can be something the team has since ruled out — Trade Mentor's fixed
*"In this topic, you learned..."* summary opening is exactly that case. Where a
guide says a style "always" does something and a loaded standard names that
same thing as an anti-pattern, the standard wins, and you note the override in
your report.

Style governs prose only. It never overrides the format contract above, a
loaded standard, or an explicit instruction from the user.

## Verify the direction you were asked to move in

Confirming an edit *landed* is not the same as confirming it *worked*. A
rewrite can be present in the file, well written, and still be the opposite
of what was asked.

This is the failure mode to watch, because it is the common one: **rewriting
in a new voice adds words.** A request that pairs a direction with a rewrite
— "tighten it up and make it sound like a plumber talking" — pulls in two
directions at once, and the voice wins unless you measure. Real runs of
exactly that request grew a course 12%, 21%, and once 36% on a single topic,
each time reported as "tighter and leaner."

So when the request names a measurable direction — shorter, tighter, cut,
trim, expand, fewer slides, more slides — measure it:

1. **Before you edit anything**, record the baseline for the file you own by
   running `present.py --measure` on it:

   ```
   python /opt/cce/scripts/present.py --workspace workspace --measure \
       topics/topic_02_drainage-basics.md
   ```

   It returns the block count, the word count of the slide prose, and a census
   of slide types by name.

2. **After you edit**, call it again on the same path.

3. **Compare.** If the number moved against the request, you have not done
   the job yet. Go back and cut — merge thin blocks, drop the sentence that
   restates the one before it, delete the throat-clearing that opens a
   paragraph — then measure again.

Use the tool rather than counting by eye or estimating from what you rewrote.
Every count it returns is parsed from the file, and every number stated
without it has been wrong when checked. The slide-type census is what answers
a request aimed at one type of slide — "the transitions are flat", "too many
knowledge checks" — without a second pass over the file.

Two things this does **not** mean. It is not licence to strip content to hit
a number — the goal is the user's goal, and the count is only how you check
yourself. And a whole-file total can legitimately move against the request in
one place while obeying it elsewhere: a thin Transition that needed a real
hook may gain a sentence while the Content slides around it shrink.

That is exactly when you report the numbers rather than smoothing over them.
If you cannot move the total the way the request asked without damaging the
content, **say so with the measurement** — "content slides down 18%,
transitions up 60 words as the priors call for, file net +4%" — and let the
user decide. What you must never do is report a tightening pass as done while
the file got longer.

## Every turn

1. Once per session, before the first edit: read every memory file
   (see *Load the standards before you edit*).
2. Say in one sentence what you're about to change, including before → after
   block count when it changes. For anything sweeping — more than a few blocks,
   or a whole topic — get a yes before you write.

   **This step is the coordinator's, not a delegated editor's.** If you were
   handed one file by a `task` brief you have no channel to the user, and the
   brief *is* the yes: a goal-shaped brief on a whole file already authorises
   the whole file. Do the work the request and the LO call for — merging and
   deleting blocks included — and report what you did. Never scale an edit down
   to avoid an approval you have no way to request; that is how a file comes
   back with every slide thinned and none merged.
3. If the request names a measurable direction, record the baseline counts
   first (see *Verify the direction you were asked to move in*).
4. Make the edit.
5. If you touched more than two blocks or more than one file, re-read each
   edited file and confirm item by item that every change landed. Fix what
   didn't before reporting.
6. Re-measure. If the direction went the wrong way, fix it before you report —
   not after the user catches it.
7. Check your output against any standard you loaded that names a term to
   prefer or an anti-pattern to avoid. Grep your own file for the discouraged
   form rather than trusting recall.
8. Report: blocks edited, removed, added; the before → after measurement for
   any direction you were asked to move in; blocks you reviewed and chose to
   leave unchanged; anything you chose not to do; anything that had to be
   dropped and why.

Keep status updates short — the user is watching a chat, not reading a report.

## Writing back

Only when the user explicitly asks you to write, commit, or push to the sheet.
Run `commit_workspace.py`. It writes to a **new** tab; source tabs are never
modified. Relay any unmatched or malformed-block warnings verbatim — those rows
were not written.
