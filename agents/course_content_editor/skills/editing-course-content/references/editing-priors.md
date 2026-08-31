# Editing priors — how SkillCat IDs actually edit this content

Read this when the user's request is directional but underspecified: *"tighten
this"*, *"too many slides"*, *"this feels off"*, *"make it match the objective"*.
It tells you what those phrases have historically meant to this team, so you
resolve the ambiguity the way they would.

**This is not a checklist.** Nothing here licenses an edit the user did not ask
for. Frequencies are given so you can weight a guess, not so you can audit.

Source: review of six SkillCat plumbing courses — 9 research-note rows and 64 of
331 slide rows carrying instructional-designer feedback, plus before/after slide
pairs. The reviewed slice is front-loaded (early rows of each deck), so "rare"
often means "not looked at" rather than "does not happen".

---

## Contents
- The one-line summary
- Structural priors (slides): dominant edit is merging; Transition vs. Content
  slides fail in opposite directions; missing connective tissue; slide-count
  padding
- Scope priors (content within a slide or note)
- Title priors
- One factual note

---

## The one-line summary

The generator's problem is **scope and packaging, not accuracy.** Across the
whole corpus, reviewers disputed exactly one fact. Everything else was *true,
but not needed here* or *right content, wrong number of slides*.

So when a user says something is wrong, the prior is: they want **less scope**
or **fewer containers**, not corrections.

---

## Structural priors (slides)

### The dominant edit is merging, not shortening
Whole-course reviews: 34 → 21 slides, 69 → 36 slides. Characters fell ~12%
against a 38% drop in slide count. When someone says a topic is bloated, reach
for **merge** before **cut**. Slides are sliced too thin: a single idea gets
split across three slides that each carry one sentence.

### Transition and Content slides fail in *opposite* directions
This is the most useful single distinction in this document, and the reason two
courses got contradictory-looking feedback (+27.8% words in one, −67% in
another) in the same review programme.

| Slide type | Its job | Typical defect | Default direction |
|---|---|---|---|
| Transition / hook | Situate the learner, pose the question | Thin, flat, no curiosity | **Expand** |
| Content | Carry one LO-bearing idea | Over-detailed, over-split | **Restrain** |
| Summary | Synthesise the takeaway | Re-lists every term covered | **Synthesise, don't itemise** |

Check `Slide Type:` before deciding which way to push. "Make this better" on a
Transition slide usually means *give it more*, not less.

### Missing connective tissue
Reviewers frequently hand-wrote bridges between slides. This is the same defect
as over-fragmentation seen from the other side: thin slices leave no room for
the sentence that links one to the next. If you merge slides, write the bridge —
don't just concatenate two bodies.

### Slide-count padding
In one course, nearly half the flagged slides existed only to satisfy a
"more than 3 topics per course" configuration. If a slide has no LO-bearing
content of its own and reads like filler, it is a merge or delete candidate —
but say so and let the user decide, since the count may be a hard requirement
on their side.

---

## Scope priors (content within a slide or note)

These came from the research-notes stage, where reviewers struck roughly 45% of
every note. They carry over to slide bodies.

**Trailing bolt-on sections.** The single highest-frequency deletion. Notes and
decks reflexively end with a section on maintenance, installation practice,
design principles, troubleshooting, tips, best practices, or "why it matters" —
struck in full, six times out of nine notes, and reappearing at the slide stage
as *bonus slides* (`The 5S Routine`, `Managing Swarf and Materials` — 13 of 13
checked slides of this shape were cut). Signature to watch for: a section or
slide headed *Maintaining…, Installation…, Tips, Best Practices, Why It Matters,
Keeping … Open, Types of ….*

**Unrequested numbers.** The most reliable carryover pattern — 5 of 6 courses at
the slide stage, ~10 flags at the notes stage. PSI ranges, pipe diameters,
fractions, temperature figures. The LO verbs in this corpus are *trace, point
out, connect, relate, state, recognise* — recognition-level. Recognition-level
verbs default to **zero** quantification.

Nuance that matters: the fix is often trimming a **modifier**, not deleting the
sentence. Reviewer, verbatim: *"The trunk concept is useful, but the size
specification is unnecessary"*; *"The branching concept is relevant; the size
detail is not."* Cut `¾-inch`, keep the trunk.

**Jargon and terminology inflation.** 4 of 6 courses. Trade synonyms, formal
classifications, and *"you might hear this called…"* asides are near-always cut.
Every new term is cognitive load that the LO has to justify.

**Cross-lesson bleed.** 3 of 6 courses. Each slide gets written as a standalone
article, so it teaches the neighbouring subtopic's material too. Before adding
an explanation, check the manifest and the other context files: if a concept is
another subtopic's job, it does not belong here.

**Register drift into field training.** *"As a technician, you can…"*,
*"Tech Tip:"*, callback/reputation framing, advice to give customers, field
workarounds. The LOs are knowledge-level; the register should stay third-person
and conceptual.

**Source-shaped output.** Content that mirrors the structure of a retrieved
source rather than the LO — e.g. reproducing a diagram's numbered legend and
teaching all of its items when the LO names four.

---

## Title priors

Titles churn heavily under review — in one whole-course pass only 1 of 34
survived verbatim. Reviewers rewrite titles to name the idea the slide carries
rather than to be catchy. If you merge slides, expect to write a new title; do
not keep the first slide's title by default.

---

## One factual note

The only accuracy defect found across six courses: domestic hot water given as
*140–180 °F*, corrected to *around 120 °F*. It is a safety-relevant figure. Its
rarity is the point — do not go looking for factual errors unless asked.
