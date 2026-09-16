# MDA editor — session findings

The durable ledger of behaviour worth changing in the Course Content Editor,
built by reading `trace_analysis/sessions/` end to end. Written by the
`analyze-mda-sessions` skill; acted on separately and deliberately.

Every entry separates **what the transcript shows** from **why it might have
happened**. The first is fact and is cited to a session and step. The second is
a hypothesis with its alternatives named, because a transcript cannot tell a
wrong instruction from an unclear file format from plain model error — and a
finding that asserts the wrong cause sends the fix the wrong way.

Entries are never deleted. A fixed finding is marked fixed and kept, so a
recurrence can be recognised as one.

**Status key:** `open` · `fixed` · `wontfix` · `needs-evidence`
**Confidence key:** `verified` (re-checked against raw run data) · `reported`
(one analyst, cited but not independently re-checked)

---

## Pass 1 — 2026-09-15

15 sessions across 5 threads, 2026-09-11 to 2026-09-15. 511 model steps,
~16.9M tokens, ~$4.42. One analyst per thread, reading each thread whole and in
order; findings merged and deduplicated here, with cross-thread recurrence noted.
Six entries were re-verified against the raw runs before being written down.

---

### F-001 — A search that correctly finds nothing is read as a broken command
**Status:** open · **Confidence:** verified · **Sessions:** 3 · **Priority: highest**

**Observed.** `01a09034` runs four `rg` searches for first-person drift (SA1
steps 13–16), each returning `[Command failed with exit code 1]`. The agent
reformulates the pattern each time, then abandons the question and moves on.
The same shape recurs in `01a09034` SA3 step 26 and `01a09044` SA1 steps 18–22
(five consecutive non-productive commands).

**Verified.** All four `rg` calls exited **1, not 127**. `rg` is installed and
ran correctly; exit 1 means *no matches*, which was the right answer. The
agent's final report — "First-person plural drift: 0 detected across all
blocks" — was **correct**. It did not believe its own evidence.

**Cost.** ~11 steps across the thread, plus one user-facing claim that reads as
unsupported but is in fact sound.

**Cause — not a hypothesis.** The sandbox renders any non-zero exit as
`[Command failed with exit code N]`. For search tools exit 1 is a meaningful
negative result, not a failure. The harness is telling the agent that a
successful "not found" is a broken command.

**Fix.** Special-case exit 1 from `grep`/`rg` in the shell tool's result
formatting. Corroboration: this tool's own failure detector needed exactly the
same special case (`render.py:_tool_output`) or it flagged every clean negative
search as a failure. The agent is making the identical mistake against the
identical signal, with no special case available to it.

---

### F-002 — Headline count in a user-facing report is wrong and unsourced
**Status:** open · **Confidence:** verified · **Sessions:** 1

**Observed.** `20260915-133239_01a0a545` step 16 states "**34** of the 57 blocks
exceed your target limit of 60 words." Its own step-10 script output marks
19 + 7 + 6 + 4 blocks as `(>60!)`.

**Verified.** Independently recounted from the cached tool output: **36**. The
denominator 57 is correct. 34 is topic_01's *block count*, which appears both in
step 5's `--measure` and in step 10's own header line `=== …topic_01… (34 blocks) ===`.

**Cost.** No steps. One wrong number in the only user-facing message of the session.

**Hypothesis.** A nearby salient number was substituted for one that was never
tallied. Not ruled out: it counted and miscounted; or it applied an unstated
exclusion (dropping the 4 Summary blocks gives 32, which matches nothing).

**Would confirm.** Have whatever prints per-block counts also print a total, and
see whether the stated figure tracks it. This is the cheapest fix in the ledger —
`instructions.md` already forbids estimating a number, but supplies no total for
the agent to quote.

---

### F-003 — WITHDRAWN. The agent had the standards. The finding was wrong.
**Status:** withdrawn · **Confidence:** verified wrong · **Sessions:** 3

**The original claim.** That `01a0a521` step 26 cited "our team editorial
standards" without reading any standards file, because thread 4 makes zero reads
under `/memories/`.

**Why it is wrong.** `memory.py` defines `scope="agent"`, and
`/memories/agent/AGENTS.md` is **hot memory — loaded into every run**. I checked
the system prompt of all 15 traces: the full text of `AGENTS.md` is present in
every one. The agent held the team standards at all times. It did not need to
read the file, and the citation was accurate.

**What this invalidates.** The zero memory reads in thread 4 are correct
behaviour, not an omission. The 7–11 memory reads per session on Sep 11 are the
unusual case, not the baseline — and they are instructed (see F-019).

**Root cause of the error.** Every analyst read the transcripts against
`instructions.md` alone. None read `memory.py`, and none checked what the system
prompt already carried. A transcript shows what the agent *did*; it does not show
what the agent was *given*. Both are needed. See the methodology note below.

---

### F-004 — Same rule, opposite behaviour: reading a named editor's private memory with no editor resolved
**Status:** open · **Confidence:** verified · **Sessions:** 2 · **Contradiction**

**Observed.** `instructions.md`: *"If no editor was named for this session, work
from `AGENTS.md` alone. Do not guess whose file to read."*

- `01a0904a` / `01a0904c` (Editor `unknown`): never lists `/memories/agent/editors/`,
  never reads an editor file. **Correct.**
- `01a09034` (Editor `unknown`): reads `/memories/agent/editors/ed_f84a65233902.md`
  directly. **Violation.**

**Verified.** Only two traces in the corpus read that path: `01a09034` and
`01a0a545`. `01a0a545` is the one trace that *does* carry `editor_id=ed_f84a65233902`,
so its read is correct. `01a09034` carries no `editor_id` and read it anyway.

**Note.** Neither analyst could see this — one praised the compliance, the other
recorded the read. It only appears holding two threads side by side, which is
the argument for merging findings centrally rather than per-thread.

---

### F-005 — Editor identity resolves intermittently, and not by build age
**Status:** open · **Confidence:** verified · **Sessions:** 15

**Observed.** 14 of 15 traces carry no `editor_id`. I had assumed this was the
identity middleware landing after Sep 11.

**Verified.** It isn't. On Sep 15: `01a0a521` (12:53), `01a0a53a` (13:21) and
`01a0a53c` (13:23) resolve no editor; `01a0a545` (13:32) — nine minutes later,
same build — does. Same day, same deployment, different outcome.

**Verified further.** The identity is absent from the system prompt itself, not
only from the run metadata: searching all 15 system prompts for `ed_[0-9a-f]{12}`
returns a match in `01a0a545` alone. So `editor_identity` genuinely named nobody
in the other 14 runs.

**Why it matters — corrected.** An earlier version of this entry said no
criterion was loaded in those runs. That was wrong: `AGENTS.md` is hot memory and
was present in all 15 (see F-003). Team standards were always available. What is
lost when identity fails is only the *per-person* layer — in `01a0a545` that
layer supplied the 60-word threshold the whole session worked to, and without it
the request "identify the issues" carries no specific numeric criterion.

**Would confirm.** Check how identity is resolved and whether a failure is
logged server-side for `01a0a521`.

---

### F-006 — Full skill load, ~40k tokens, that provably never reached the answer
**Status:** fixed 2026-09-15 — SKILL.md `description` now gates on an imminent file change and explicitly excludes general discussion.
**Was:** open · **Confidence:** verified · **Sessions:** 1

**Observed.** `20260911-114832_01a0904c`: user asks to *discuss* storytelling
principles. No workspace open, no file named, no edit requested. The agent reads
the entire 376-line `editing-slide-chunks/SKILL.md` across steps 1–2, lists
`references/writing-styles/` at step 3 (6 files), then answers from general
pedagogy at step 4.

**Verified.** The full 4,725-character answer contains zero mentions of any of
the six style filenames, zero instances of the word "style", and zero references
to the skill or house guidance. The listing and the skill text did not reach the
answer.

**Cost.** ~40k tokens, 3 of 4 steps, on a turn that changed nothing.

**Hypothesis.** Two texts disagree about the trigger. `instructions.md` scopes
the skill read to *"before your first edit"*; the skill's own frontmatter says
*"Use for any request about slides, decks, or topics, even when the user doesn't
say 'edit'."* The agent followed the broader one. Not ruled out: deliberate
grounding of a domain answer, or plain over-preparation. The transcript cannot
say which text it was acting on.

**Fix if it recurs.** One-line frontmatter change. Did not recur in the other
four threads, so this is currently a single observation.

---

### F-007 — `python -c` quoting re-derived from scratch, and failed identically on retry
**Status:** open · **Confidence:** reported · **Threads:** 1, 2, 4 · **Recurs**

**Observed.**
- Thread 1: `01a0901f` step 13 (`unexpected EOF`), fixed step 14. `01a09022`
  SA4 steps 19 **and** 20 — same `SyntaxError: unexpected character after line
  continuation character` twice, fixed step 21. Step 20 is a true recovery loop.
- Thread 2: `01a09034` SA2 steps 18–19, identical error twice, fixed step 20.
- Thread 4: `01a0a521` step 11 FAILED, fixed step 12; step 14 FAILED with the
  **identical** error, fixed step 15 — after the lesson had already been paid for.

**Cost.** ~9 steps across three threads.

**Hypothesis.** Ad-hoc `python -c` is the default measuring instrument and its
quoting is regenerated each time. Not ruled out: intervening context compaction;
plain generation error.

**Fix.** Document a heredoc or temp-script idiom, or supply a first-class
counting tool (see F-013).

---

### F-008 — `glob` without `path` fails against the sandbox root, repeatedly
**Status:** open · **Confidence:** reported · **Threads:** 1, 2, 5 · **Recurs**

**Observed.** Identical `cannot glob the sandbox root` error at least seven
times: `01a09022` SA2/SA3/SA4 step 3–4; `01a09031` step 25; `01a09034` SA1 step 3,
SA3 step 6; `01a0903b` SA1 step 8; `01a0a545` step 11. Every occurrence recovers
on the next call with an explicit `path`.

**Cost.** ~7 wasted steps + 7 retries.

**Hypothesis.** `path` is not effectively required by the tool schema, so a
path-less call is the natural first guess. Not ruled out: the agent knows the
rule and is guessing at a location because the target path was never given.

**Would confirm.** Check whether `glob`'s schema marks `path` required or
defaults it to `/workspace`. This is a schema fix, not a prompt fix.

---

### F-009 — The same measurement taken three to four times per file
**Status:** open · **Confidence:** reported · **Threads:** 1, 4 · **Recurs**

**Observed.** Thread 1, `01a09022`: parent measures all four files at step 4;
every span then re-measures the *unedited* file before writing (SA1 9, SA2 3,
SA3 2, SA4 11), all returning identical before==after, 0.0%. Post-write measures
follow, then the parent measures all four again at step 9. Thread 4,
`01a0a521` step 25: `--measure` run before any edit existed; `01a0a53a` step 2's
`--measure` is then superseded by step 3's full `present.py`, which returns the
same `counts` block.

**Cost.** ~8 steps across both threads.

**Hypothesis.** The parent's measurements are never passed into the worker
briefs, so each worker re-establishes its own baseline. Not ruled out: the
post-write measure is correct and only the pre-write one is habit.

**Would confirm.** Put the parent's per-file counts into the brief and see
whether the pre-edit measure drops.

---

### F-010 — Script source read instead of `--help`, three times over per fan-out
**Status:** open · **Confidence:** reported · **Threads:** 1, 2, 4 · **Recurs**

**Observed.** Thread 1, `01a09022`: SA2, SA3 and SA4 each independently read
`present.py`, `_presentation.py` and `commit_workspace.py` to rediscover the
block-format contract — ~21 steps, roughly a quarter of all subagent steps in
that session. SA1 did none of it. Thread 2, `01a09034`: same pattern in SA1, SA2
and SA3; SA4 (the 17-step span) read `parsing-rules.md` once and opened no `.py`
file at all, reaching the same outcome. Thread 4: `01a0a521` step 7 ran
`prepare_workspace.py --help`, then read the source anyway at step 10;
`01a0a53c` read 189 lines of `commit_workspace.py` with no `--help` first.

**Cost.** ~25 steps across three threads; the reads are large (`_common.py` is
426 lines).

**Hypothesis.** The block schema is documented in
`skills/working-with-google-sheets/references/parsing-rules.md`, but the worker
brief does not point at it, so each worker reaches for the implementation. Not
ruled out: workers distrust the doc; or the SKILL.md block section is buried
(one span found it at offset 100 of a 344-line file).

**Would confirm.** Check whether `parsing-rules.md` plus the SKILL.md schema
actually state what the spans went to source for. If yes this is discoverability;
if no, the docs are missing a rule.

---

### F-011 — Two-call grep: files-with-matches, then the same search for content
**Status:** open · **Confidence:** reported · **Threads:** 1, 2 · **Recurs**

**Observed.** At least eleven times, always on a single already-known file path,
so the first call returns only the filename the agent supplied: `01a09031` 16→17;
`01a09034` SA1 9→10, SA2 5→6, SA3 24→25, SA4 7→8; `01a0903b` SA1 18→19, SA3 14→15;
`01a09044` SA2 5→6, SA3 10→11 and 12→13.

**Cost.** ~10 steps.

**Hypothesis.** Default `output_mode` is `files_with_matches`, so the first call
is the agent discovering it asked the wrong question. Recurs in the final fan-out
after two user corrections, so it is habitual rather than situational.

---

### F-012 — Post-write verification performed by every available instrument
**Status:** open · **Confidence:** reported · **Threads:** 1, 2 · **Recurs**

**Observed.** Thread 1, `01a09022` SA1: after one `write_file`, seven
verification steps — grep, `--measure`, full `present.py`, then the whole
415-line file re-read in four paged calls. SA3: `--measure`, full re-read, then
`parse_topic_file` twice with overlapping output. Thread 2, `01a0903b` SA1:
nine consecutive single-pattern greps (seven returning no matches) where sibling
SA4 ran one `python -c` with four assertions covering the same ground in a
single step.

**Cost.** ~25–30 steps across both threads. This is the largest single
contributor to sibling step spread.

**Hypothesis.** "Check your work" names no mechanism, so each span invents one,
and greps are the cheapest thing to reach for one at a time. Not ruled out: the
`_presentation.py` docstring (read by several spans) recounts an episode of the
agent reporting edits it never made, which may read as a mandate for maximal
verification.

**Would confirm.** Name one canonical post-write check in the brief and measure
the step delta. The short siblings already demonstrate the target shape.

---

### F-013 — No tool exposes per-block word counts, so the agent writes a second parser
**Status:** open · **Confidence:** reported · **Threads:** 2, 5

**Observed.** `present.py --measure` returns file-level totals only. To find
which individual slides breach a word limit, `01a0a545` step 10 writes a 20-line
Python script re-implementing the block parser that `parsing-rules.md` says the
commit scripts already contain. `--blocks 3,7` exists but selects blocks to
*display*, not to count.

**Cost.** 1 step, plus a second parser whose agreement with the canonical one is
unverified — and it is the source of F-002's wrong number.

**Note.** The same script computed a paragraph count it never printed, while the
report asserts "nearly all slides are a single wall of text" — a claim whose
measurement was taken and discarded.

**Fix.** A per-block count mode on `present.py` would close F-002, F-013 and part
of F-007 at once.

---

### F-014 — The coordinator does the delegatee's reading, against an explicit instruction
**Status:** partly fixed 2026-09-15 — the five words that forced it are gone: "Every turn" item 1 now asks for the block count only when already known, and says never to read every topic to fill it in. "Do not pre-diagnose" is unchanged and correct. Whether the coordinator still over-reads is now an open question for the next pass.
**Was:** open · **Confidence:** reported · **Sessions:** 1 · **Highest cost in its thread**

**Observed.** `instructions.md` §"Delegating large sweeps": *"any request
spanning 2+ topic files. Spawn one editor per topic… Do not pre-diagnose a topic
you are delegating: don't build its edit list… Read only as far as you need to
confirm which topics are in scope."* `01a0a545` step 4 read `manifest.json`,
which already confirmed scope. The agent then read all four topic files to the
last line (steps 6–9, ~790 lines), ran a per-block diagnosis across all four
(step 10), and handed back a finished per-topic edit list at step 16.

**Cost.** 5 of 16 steps, 8 of 24 tool calls, reading content four parallel
editors would each have read anyway.

**Hypothesis.** Two instructions pull opposite ways. §"Every turn" requires the
before→after block count and a yes *before* delegating — which cannot be stated
without knowing what each topic needs — while §"Delegating large sweeps" forbids
building exactly that. The agent resolved toward the approval requirement. Not
ruled out: the session ends before any delegation decision is observable, so it
may have intended to delegate next; but the forbidden pre-diagnosis has already
happened either way.

**This is the only finding that looks structural rather than incidental** — two
instructions that cannot both be satisfied.

---

### F-015 — Writing style chosen silently where the skill says to ask
**Status:** open · **Confidence:** reported · **Sessions:** 1

**Observed.** `editing-slide-chunks/SKILL.md` "Which style applies, in order":
user names one → a default in `AGENTS.md` → *"Otherwise, ask. Name the labels and
ask which to write in, before rewriting prose. Inferring silently produces a
different answer on different runs of the same sheet."* In `01a0a545` the user
named no style and no `AGENTS.md` default was loaded, so rule 3 applied. The
agent read one of six guides (step 15) and asserted it at step 16, asking only
the composite "Shall I proceed with this sweep?" The other five labels were never
shown.

**Cost.** Risk, not steps: a wrong voice means the whole 4-topic sweep is redone.

**Related.** `01a09039` surveyed 3 of 6 available styles before recommending one,
and that recommendation drove the next 99-step session.

---

### F-016 — Subagent step count tracks neither file size nor work required
**Status:** open · **Confidence:** reported · **Sessions:** 1

**Observed.** `01a09022` spans: 20 / 25 / 22 / 23 steps (spread 1.2x) against
jobs differing ~5.7x. SA1's topic_01 has 34 blocks and 8 of the 10 em dashes;
SA4's topic_04 has 6 blocks and — as SA4 itself measured — **zero** em dashes,
i.e. nothing to do for half its brief. It spent 23 steps anyway. Tokens *do*
track size (788k/687k/572k/447k), so the flat step count is not a context artifact.

**Hypothesis.** The brief is a fixed 5-point checklist executed at full ceremony
regardless of what the first measurement shows. Not ruled out: a minimum
verification ritual at model level; or the brief's per-file content mandates it.

**Would confirm.** Re-run with a brief that says "if the first scan finds nothing
to change, stop and report that."

---

### F-017 — A self-run check flagged missing terms; the report claimed 100% integrity
**Status:** needs-evidence · **Confidence:** reported · **Sessions:** 1

**Observed.** `01a09042` step 1's own term-presence script printed
`topic_01: 71/72 … Missing: ['TPI']`, `topic_02: 10/12 … Missing: ['retracted',
'waterproof base']`, `topic_03: 15/16 … Missing: ['box beam']`. Step 2 resolved
TPI. Step 5 answered "**Yes, all informational integrity is 100% maintained**…
No technical specifications… were dropped." The three remaining flags are never
mentioned.

**Hypothesis.** Probably false positives from exact-string matching (topic_02
does contain "Always retract razor knives"). Not ruled out: resolved in
non-visible reasoning; or dropped.

**Regardless of the answer, the behaviour to flag is:** a measurement that
contradicted the headline claim was not surfaced.

---

### F-018 — Duplicate `present.py` for the same file — do not treat as waste yet
**Status:** needs-evidence · **Confidence:** reported · **Threads:** 1, 2

**Observed.** `01a09034` step 6 chains present for topic_01 `&&` topic_02, then
step 7 presents topic_02 again alone. `01a09044` step 5 chains 02 `&&` 03 `&&` 04,
then steps 6–7 present 03 and 04 again. ~47k chars of duplicated JSON.

**Why this is not yet a finding.** `present.py` output is intercepted by
middleware that renders cards to the user. If that middleware handles only the
first JSON payload per `execute`, the chained call showed the user one file and
**the re-runs were necessary**.

**Would confirm.** Test whether chained `present.py` calls in one `execute`
render multiple card sets. If they do not, the skill should say "one
`present.py` per `execute`" — and this becomes a documentation fix, not an agent
fix.

---

## Working well — do not regress these

Recorded because a later change that breaks one of these should be recognisable
as a regression.

1. **No fabricated counts, with two exceptions.** Across 511 steps, every number
   in the user-facing reports traces to a measurement — except F-002 and F-017.
   Block censuses are invariant across sessions and across two independent
   instruments (raw split and parser both give 34/8/9/6 = 57).
2. **No block loss.** Every write in thread 1 and thread 2 preserved block count
   and slide-type census exactly; the commit wrote exactly 57 rows, 0 malformed.
3. **Read-before-judge held everywhere.** No span edited a file it had not read
   first, in 511 steps.
4. **Write gates held on open-ended requests.** `01a0901f` stopped and asked
   rather than editing on a review request. `01a0a545` made zero edits on
   "identify and fix the issues" and ended with a stated block delta and an
   approval request. `01a0a521` proposed before acting, then showed the diff
   before committing.
5. **Measure-then-correct works.** `01a0903b` SA1/SA2/SA4 each caught word
   inflation (+27.1%, +21.1%, +29.1%) in `--measure` and rewrote to +4.7%,
   +15.5%, +3.3%. The second write is not waste.
6. **An honest zero.** `01a09022` SA4 reported "0 em dashes were present" rather
   than manufacturing replacements to match the brief's expectation.
7. **Corrections do sometimes stick.** After a user complaint about formulaic
   openings, the next fan-out measured the defect *before* dispatching, put
   per-file counts into each brief, and re-ran the identical census after. That
   fan-out is the cheapest of three (88 steps vs 109 and 99) with the tightest
   siblings — the best-calibrated behaviour in the corpus.
8. **Sensible unprompted judgement.** `01a0a53c` overrode a `manifest.json`
   target tab that would have clobbered an unrelated tab, and committed to the
   correct one instead.
9. **Exemplary calibration exists.** `01a09048`: 2 steps, 1 tool call, $0.04 for
   "write to the sheet" — one commit, then report, nothing re-verified.
10. **`write_file` for whole-file overhauls** per `AGENTS.md`, with zero failed
    string replacements in 511 steps.

---

## Ranked by what to do next

| | Finding | Why first |
|---|---|---|
| 1 | **F-001** | Verified, one-line fix in tool-result formatting, explains behaviour across 3 sessions |
| 2 | **F-008** | Verified pattern, schema fix not prompt fix, 7 occurrences |
| 3 | **F-013 + F-002** | One per-block count mode closes a wrong user-facing number and a duplicate parser |
| 4 | **F-014** | The only structural one: two instructions that cannot both be satisfied |
| 5 | **F-005** | If identity is silently optional, per-editor memory is dead weight for most runs |
| 6 | **F-012 + F-010** | Largest recoverable step cost (~50 steps), but needs a brief redesign, not a one-liner |

Bringing each fan-out's outlier down to its shortest sibling in thread 2 alone
would save ~42 steps and ~2.6M tokens (~19% of that thread, ~$0.70).

**Caveat on the whole pass.** One editor, four days, 15 sessions, all of it
authoring-and-testing rather than production use. Recurrence counts here mean
"this agent did it repeatedly", not "this happens at rate X in production".
Re-run this pass once real editors are using the tool before treating any
frequency as stable.

---

### F-019 — Repeated AGENTS.md reads are instructed, not agent error
**Status:** open · **Confidence:** verified · **Sessions:** 4

**Observed.** Sessions on Sep 11 read `/memories/` 7 to 11 times each. An earlier
draft of this ledger counted these as redundancy.

**Verified.** They are instructed. `editing-slide-chunks/SKILL.md` says: *"Before
your first edit in a session, read every memory file"*, then `read_file
/memories/agent/AGENTS.md`, then — in the same paragraph — *"it is loaded into
every run — so as the coordinator you already have it. A delegated editor should
read it anyway rather than assume the brief carried it."*

So the skill states the file is already in context and directs a read anyway. The
agent complied. The behaviour is correct against the instruction.

**The real question is whether the instruction is correct.** The skill gives a
reason: a measured run where two of three delegated editors read no memory and
produced prose contradicting team standards. That is a real failure the rule
prevents. But a delegated editor inherits the same system prompt, so `AGENTS.md`
is hot for it too — which would make the read redundant for subagents as well.

**Would confirm.** Check a subagent span's system prompt for `AGENTS.md` text. If
it is present, the instruction costs a step per span for nothing, and the earlier
failure it cites had some other cause. If it is absent, the instruction is right
and this entry closes.

---

## Methodology note — added after pass 1 was reviewed

Pass 1 was run against `instructions.md` only. That was not enough, and it
produced at least one inverted finding (F-003) and one wrong rationale (F-005).

**A transcript shows what the agent did. It does not show what the agent was
given.** Before calling any behaviour redundant, unsourced, or excessive, check:

1. **`memory.py`** — what is hot memory, loaded into every run without a read.
2. **The system prompt itself** — it is in the raw runs at
   `inputs.messages[0]` as the `SystemMessage`, ~30k characters here, and it
   already contains `AGENTS.md`, the skill index and the identity block. An agent
   that does not read a file may already hold it.
3. **The skill files**, not only `instructions.md` — `SKILL.md` and
   `instructions.md` sometimes disagree, and the agent may be following the other
   one (see F-006, F-019).
4. **What the user actually asked for in that turn.** Work that looks excessive
   may be the literal request. `01a0a545` was asked to "identify and fix"; the
   identify half may legitimately belong to the coordinator (see F-014).

The `analyze-mda-sessions` skill has been updated to require these four checks
before any finding is written.

---

## Pass 2 — 2026-09-15 · warrant, not efficiency

Pass 1 asked whether a step was efficient. Pass 2 asks whether the user or the
instructions asked for it. Four analysts, each required to read `REQUESTS.md`,
its system-prompt variant, `instructions.md` and the skill files before writing
anything. Every verdict is one of: warranted by request · warranted by
instruction · agent's own invention · skipped.

**Headline result: nothing the user asked for was left undone, in any of the 15
sessions.** Every gap is against an instruction. Most behaviour pass 1 called
waste turns out to be instructed. The defects are in the documents.

---

### F-020 — Two memory policies in the same prompt give opposite orders
**Status:** fixed 2026-09-15 — instructions.md Memory now names the harness policy and overrides it, and tells an unidentified session not to write to AGENTS.md by default.
**Was:** open · **Confidence:** verified · **Priority: highest of pass 2**

**Observed.** `01a09044` step 10 wrote a new team-wide standard ("Avoid
Formulaic Slide Openings") into `/memories/agent/AGENTS.md`, mid-turn, without
asking. The correction had been made once, by an unidentified editor, about
prose the agent itself wrote two turns earlier.

`instructions.md` Memory section: *"Write here only for something the whole team
would expect to still apply next session … or the same correction made more than
once"*, *"A one-off fix is not memory-worthy"*, and *"When a correction could go
in either place, ask which."*

**Verified.** The system prompt carries a `memory_guidelines` block at lines
383-442 which says the opposite:

- line 392 — *"Learning from your interactions with the user is a top priority."*
- line 395 — *"Each correction is a chance to improve permanently - don't just
  fix the immediate issue, update your instructions."*
- line 408 — *"When the user gives feedback on your work - capture what was wrong
  and how to improve"*

The agent obeyed the injected block over `instructions.md`. Both were in front of
it. The two cannot both be followed.

**Cost.** Plus 302 characters on the system prompt of every run of this
deployment, for every caller, forever — recorded as a team standard from one
person's single comment.

**Second problem in the same rule.** "Ask which — team or yours?" assumes an
identified editor. This session had `Editor unknown`, so there was no second
branch to offer. The rule has no guidance for an anonymous session, which is
14 of 15 sessions in this corpus (see F-005).

**Note.** The write itself is visible in the prompt diff:
`system_prompt_3f3b4af7` to `3d54e0b5` differs by exactly this rule. The memory
path works correctly. The question is whether it should have fired.

---

### F-021 — The style picking order was skipped entirely, then a topic was rewritten and committed
**Status:** fixed 2026-09-15 — REGRADED. The user confirms the silent default was wanted: AGENTS.md is the evolved house voice and the labelled guides are a reserve. The skill's picking order now says so, and asking is reserved for a voice AGENTS.md does not cover. The agent's behaviour was right; the document was stale.
**Was:** open · **Confidence:** verified · **Sessions:** 3

**Observed.** `editing-slide-chunks/SKILL.md` gives an order: the user names a
style, or `AGENTS.md` states a default, or *"Otherwise, ask. Name the labels and
ask which to write in, before rewriting prose. Inferring silently produces a
different answer on different runs of the same sheet."*

**Verified.** Across all 31 tool calls of thread 4's three sessions there are
**zero** reads under `references/writing-styles/`. No style label appears in any
message. The user was never asked. `01a0a53a` then rewrote a whole topic —
15 blocks to 10, minus 29.7% words — and `01a0a53c` committed it.

Rung 2 does not apply: the deployed `AGENTS.md` describes a voice ("On-the-Job
Coach", short paragraphs) but names none of the six labels. Rung 3 was live.

**The instruction problem.** Rung 2 asks the agent to tell "a stated default
style" from "editorial principles about voice" with no test for doing so. The
`AGENTS.md` actually deployed reads like the former and is the latter.

**Contrast.** Thread 2 did this correctly. `01a09039` read three style guides
and recommended one, which let every later brief name the label.

---

### F-022 — Raw gspread calls against an explicit prohibition
**Status:** fixed 2026-09-15 — REFRAMED. The user was pasting a sheet link, not naming a tab; the gid was copy-paste noise. The sheets skill now says to ignore the gid outright. The earlier suggestion to print gids from list_tabs.py was REJECTED — it would teach the agent the gid matters.
**Was:** open · **Confidence:** verified · **Sessions:** 1

**Observed.** `working-with-google-sheets/SKILL.md`: *"Never touch the sheet
except through these scripts. Don't call gspread yourself … If you need to
inspect the sheet … that's `check_auth.py` / `list_tabs.py`, not a one-off script
you write."*

**Verified.** `01a0a521` made **9** hand-rolled gspread calls against 7
sanctioned script calls. Two failed on shell quoting. Their result ("Total
diffs: 7") is never used again in any of the three sessions.

**This raises pass 1's verdict.** Pass 1 called these steps unproductive. They
are a rule violation.

**The instruction problem — and the real fix.** The user's URL ended in
`#gid=1672887275`. No sanctioned script converts a gid to a tab name;
`list_tabs.py` prints names only. The agent had a question the toolkit cannot
answer, and a prohibition alone did not stop it improvising. Either
`list_tabs.py` should print gids, or the skill should say plainly that the gid is
not needed because `--source-tab` takes a name.

---

### F-023 — The coordinator pre-diagnosed files it was delegating, in the briefs
**Status:** open · **Confidence:** reported · **Sessions:** 2 · **Recurs**

**Observed.** `instructions.md` Delegating section: *"Delegate the judgment, not
the typing … Do not pre-diagnose a topic you are delegating: don't build its edit
list … Read only as far as you need to confirm which topics are in scope."*

`01a09034` briefs name specific blocks: *"notably in Block 7 (Summary) which
opens with 'We've covered how to choose the right carrier'"*. `01a09044` step 1
scanned the first 8 words of every `Content:` line across all four files before
fanning out, then put per-file defect counts in each brief.

**The measurable symptom.** In `01a09034`, subagent 4 got a file the coordinator
had already cleared. It ran 17 steps and reported point 3 as "Confirmed… no
drift" — it verified the coordinator's finding instead of reviewing
independently.

**But the instruction fights the user here.** The same briefs pin "verify block
count remains exactly 34 / 8 / 9 / 6", and that pin is what correctly enforced
the user's "hold on the 4th point" exclusion (F-024). Pre-diagnosis and
constraint-passing are hard to separate in practice.

---

### F-024 — Point 4 was correctly excluded
**Status:** working as intended · **Confidence:** reported

User said *"hold on the 4th point but implement 1,2,3"*. Point 4 was a
consolidation and merge proposal. `01a09034` step 2 measured 34/8/9/6 blocks
after the sweep — identical to the baseline. No brief mentions merging. Recorded
so a later regression is recognisable.

---

### F-025 — Two slide titles replaced with "Intro", unreported, under a prose-only request
**Status:** open · **Confidence:** verified · **Sessions:** 1

**Diagnosis superseded by F-046 (2026-09-16):** the fixed labels are instructed by the writing-style reference, not an agent defect. The unreported-change half of this entry stands.

**Observed.** User said *"Okay works. Implement this style"*.
`editing-slide-chunks`: *"Style governs prose only. It never overrides the format
contract above, a loaded standard, or an explicit instruction from the user."*

**Verified** from `present.py` output in `01a0903b`:

```
"[Transition] The Power of Hand Tools"  ->  "[Transition] Intro"
"[Transition] Carrying Your Gear"       ->  "[Transition] Intro"
```

Topics 03 and 04 kept their titles. The coordinator's step-12 summary lists five
bullets for topic 01 and mentions no title change, against Every turn item 3
which requires reporting what was edited. The change reached the sheet.

**Cost.** Two informative opening titles replaced with the least informative
possible string.

---

### F-026 — Transition slides expanded about 3x against a loaded standard
**Status:** open · **Confidence:** reported · **Sessions:** 1

**Observed.** `AGENTS.md` section 3, in the system prompt on every turn: *"No
Forced Expansion on Transitions: Never force-expand transition slides or pad them
with artificial throat-clearing."* The skill agrees: *"Sharpen, not lengthen …
Padding one to reach a length is the defect, not the fix."*

`01a0903b` step 3: topic 02 block 0 went from one 22-word sentence to three
paragraphs closing *"Let's look at how different toolboxes and bags keep your
gear protected…"*. The same "Let's look at / Let's see how" closer was added to
all four topic openers. Topic 02 plus 15.5% words, topic 03 plus 9.9%.

**Hypothesis.** Four parallel editors each reached for the same three-beat
template from the style guide, and no single context saw all four to notice they
were identical. Not ruled out: the guide's worked examples model that shape.

**Note the likely chain.** This is plausibly what produced the "every slide opens
the same way" monotony the user complained about two turns later — which in turn
triggered the unasked memory write in F-020.

---

### F-027 — Verification questions answered without the instructed tool
**Status:** open · **Confidence:** verified · **Threads:** 1, 2 · **Recurs**

**Observed.** `instructions.md`: *"Run it again whenever they ask to see, read,
check, or confirm anything"* and *"do not restate a number from memory a turn
later."*

**Verified.** `01a0901f` — a "review and check formatting" turn — made 15 tool
calls. Exactly one mentions `present.py`, and it is `cat /opt/cce/scripts/present.py`:
the agent read the source and never ran the tool. It then reported "57 blocks,
9 Transition, 44 Content, 4 Summary" from hand-written scripts.

`01a09042` — user asked *"all informational integrity is maintained as per LO?"*
— made four tool calls, none of them `present.py`. Its own checker printed gaps
(`Missing: ['TPI']`, `['retracted', 'waterproof base']`, `['box beam']`). The
answer was *"Yes, all informational integrity is 100% maintained."* Two gaps were
chased; `'waterproof base'` was never resolved in the transcript; none were
mentioned to the user.

**Why it matters.** Both sets of numbers were later shown correct. That is
exactly the failure the instruction names: *"An unverified count reads exactly
like a verified one to the person trusting it."*

---

### F-028 — The fourth editor ran serially, against the parallel-spawn rule
**Status:** open · **Confidence:** verified · **Sessions:** 1

**Verified** from timestamps in `01a09022`:

```
task 1   start 11:02:49.540   end 11:04:26
task 2   start 11:02:49.542   end 11:04:10
task 3   start 11:02:49.546   end 11:04:07
task 4   start 11:04:32.669   end 11:05:34
```

Three spawn within 6 milliseconds. The fourth starts 103 seconds later, after all
three finish. `instructions.md`: *"Spawn one editor per topic, in parallel, in a
single turn."*

**Hypothesis.** A three-way parallel cap in the harness. Not ruled out: the model
emitted three and re-planned. The step-6 brief uses the same template as the
others, which argues against a deliberate follow-up.

**Cost.** About 100 seconds of a 681-second session, serialised for no stated
reason.

---

### F-029 — manifest.json "every session" is skipped whenever a workspace persists
**Status:** fixed 2026-09-15 — now scoped to "whenever a workspace is opened or reopened, and before your first edit in a thread".
**Was:** open · **Confidence:** reported · **Threads:** 1, 2, 4 · **Recurs**

**Observed.** `instructions.md` Getting oriented item 3: *"Read
`/workspace/manifest.json` every session."* Thread 2 read it once, in session 1,
and in none of sessions 2 to 7. Thread 1 read it in neither session. Thread 4
read it in session 1 only.

**But on a bare greeting the rule fires and guarantees a failure.** `01a0904a`
step 1 read it, got `file_not_found`, and correctly asked for a sheet URL. The
probe is instructed and its failure is designed in.

**The instruction problem.** "Every session" is both over- and under-applied. The
suggested repair from two analysts: *"every session that opens or re-opens a
workspace"*, or gate it on an imminent edit.

---

### Scope discipline — what pass 2 found to be clean

- **No edit was ever made without authorisation.** `01a0901f` reviewed without
  touching a file and asked. `01a0a521` proposed and asked. `01a0a545` made zero
  edits on "identify and fix" and asked. Every commit traces to the user's own
  words.
- **The deletion pass 1 flagged was correct.** `01a0a53a` removed a transition
  block; it was named in advance (`01a0a521` step 26), licensed by `AGENTS.md`
  section 3 "Avoid Subtopic Transition Bloat", answered with *"yes please"*, and
  reported after with the verified 15 to 10 count. The five missing blocks
  reconcile exactly: four merges plus one deletion.
- **Briefs transmitted the user's words verbatim**, typo included, to all four
  editors in `01a09022`.
- **Edit scope did not widen.** Editors audited en dashes, double hyphens and
  hyphen-minus and changed none. Titles, slide types and block counts held —
  except F-025.
- **A subagent's own report was more honest than the coordinator's summary.**
  `01a09044` subagent 1 disclosed "After: exactly 2 slides open with 'When…'
  (5.9%)"; the coordinator relayed "replaced" without the caveat.

---

## Pass 2 conclusion

The agent follows its documents. The documents disagree with each other.

| Conflict | Sides |
|---|---|
| **F-020** memory | `instructions.md` "a one-off is not memory-worthy" against injected `memory_guidelines` "each correction is a chance to improve permanently" |
| **F-021** style | Rung 2 points at `AGENTS.md`, which describes a voice without naming a label |
| **F-006** skill trigger | `SKILL.md` description fires on topic; `instructions.md` fires on imminent edit |
| **F-022** sheets | The rule forbids raw gspread but the toolkit cannot answer the gid question |
| **F-029** manifest | "Every session" guarantees a failed probe on greetings and is ignored on continuations |
| **F-023** delegation | "Do not pre-diagnose" against the need to pass the user's constraints into a brief |

Fix the documents before you tune the agent. Five of these six are text edits.

---

## Pass 3 — 2026-09-16

6 sessions across 3 threads, all 2026-09-16 10:40–11:15. 177 model steps,
~4.9M tokens, ~$1.58. One session (`01a0a9ed`) is 121 steps and $1.15 on its
own — 73% of the pass. Read whole and in order; every claim below re-checked
against the raw runs.

---

### F-030 — A per-block word limit with no per-block measurement tool
**Status:** fixed · **Confidence:** verified · **Sessions:** 1 · **Priority: highest**

**Fixed 2026-09-16.** `measure()` keeps per-block counts as `per_block` (id, words, label); both skills point at it.

**Observed.** In `01a0a9ed`, 25 of the 42 `execute` calls are ad-hoc
`python -c` scripts. Every one of them computes the same two things: the word
count of each block, and whether an em dash is present. Four subagents each
wrote their own counter, and each wrote it differently: topic_01 splits the raw
text on `###Block ID:`; topic_02 and topic_03 import `_presentation.parse_blocks`;
topic_04 imports `commit_workspace.parse_topic_file`. Subagent 3 additionally
ran its own draft sentences through the counter one at a time before writing
them (steps 14–17, 19, 20).

**Verified.** `_presentation.measure()` computes per-block word counts and then
discards them: `def words(blocks): return sum(len(block_body(b, kind).split())
for b in blocks)`. `present.py --measure` reports a file total only. The brief
the orchestrator wrote demands "Strictly keep every slide under 60 words" — a
per-block constraint. No sanctioned tool reports a per-block number.

**Why.** This is a missing tool, not model error. The agents were given a
constraint they were required to verify and no instrument that measures it. The
only remaining move is to build one, and a fresh subagent with no shared context
builds it from scratch every time. The alternative explanation — that the agents
distrust `present.py` — is not supported: they call `present.py --measure` as
well, and use it for exactly the totals it does report.

**Fix.** Add per-block words to the `blocks` array that `present.py` already
emits, and note in editing-slide-chunks that per-slide counts come from there.
One `sum()` unrolled removes the single largest source of steps in the pass.

---

### F-031 — An editor's preference hardened into a gate, against the memory contract
**Status:** fixed · **Confidence:** verified · **Sessions:** 1 · **Priority: high**

**Fixed 2026-09-16.** instructions.md delegation now requires a standard be carried in its own words, with no force added.

**Observed.** `/memories/agent/editors/ed_f84a65233902.md` reads, in full:
"Target Audience: Apprentice-level HVAC learners. Slide Length: Keep slides
under 60 words." All four briefs in `01a0a9ed` render this as "**Strictly** keep
every slide under 60 words", and subagents 2 and 3 turned it into a literal
`assert wc < 60` in their verification scripts. Measured effect: topic_01
−27.6%, topic_02 −35.9%, topic_03 −36.4%, topic_04 −19.5% words.

**Verified.** `AGENTS.md` §4 says the opposite in the same breath: "Don't pad a
slide with unnecessary background just to hit an arbitrary word count; if a
concept is clearly explained in 35–50 words, let it breathe." Its opening line
asks that all of it "guide editorial reasoning rather than [be treated] as
rigid, dogmatic checklists." `instructions.md` Memory says "Memory is notes, not
instructions… It never widens what you are allowed to do."

**Why.** The orchestrator restated a preference as a threshold when it wrote the
brief, and a threshold is the one form a subagent cannot soften — it has no
channel back to ask. The word "Strictly" is the orchestrator's own; it is in no
source file. Contrast with a plausible alternative — that the user asked for
tight slides — which the request does not support: the user said "find issues
and fix them".

**Fix.** State in the delegation section that a brief must carry a preference in
the words the preference was written in, and must not add force to it.

---

### F-032 — Pre-diagnosis in every brief, against the explicit rule
**Status:** open · **Confidence:** verified · **Sessions:** 1 · **Priority: high**

**Observed.** All four briefs in `01a0a9ed` carry a per-block edit list:
topic_04 "Tighten wordy slides (e.g. Block 2 at 83 words and Block 5 at 81
words)"; topic_03 "testing horizontal and vertical vials use the identical
180-degree reversal test — consider merging them"; topic_02 "Rewrite the summary
slide (Block 7)"; topic_01 "Eliminate subtopic transition bloat."

**Verified.** `instructions.md` Delegating: "Do not pre-diagnose a topic you are
delegating: don't build its edit list, don't decide which blocks merge… Read
only as far as you need to confirm which topics are in scope." The topic_03
brief decides which blocks merge, in those words. To produce these numbers the
orchestrator read all four topic files in the prior turn (`01a0a9eb` steps
6, 9, 10, 11) — the pre-reading the rule exists to prevent.

**Why.** Two instructions pull apart here and the agent obeyed the wrong one.
The previous turn's user request was "find issues… and then fix them" — finding
is diagnosis. The agent diagnosed, reported, got a "Proceed", and then had no
way to un-know what it had found. This is a sequencing conflict in the
instructions, not disobedience.

**Note.** Three of the four briefs contain an em dash while instructing
"Absolutely zero em dashes".

---

### F-033 — Re-preparing a workspace silently discarded the previous turn's edits
**Status:** fixed · **Confidence:** verified · **Sessions:** 1 · **Priority: high**

**Fixed 2026-09-16.** Both prepare scripts refuse to overwrite a workspace that differs from `.baseline` (exit 3), naming the drifted files and three ways forward: a second workspace, commit first, or `--discard-local`.

**Observed.** In `01a0a9d2` (turn 3) the agent ran `prepare_workspace.py …
--source-tab "Topic 1: Pipe Preparation"` at step 7. The previous turn
(`01a0a9cf`) had rewritten `/workspace/topics/topic_01_pipe-preparation.md`,
raising it from 411 to 456 words. After step 7 the same path measures 548 words
and 10 blocks — the sheet's content, not the edit. The edit is gone. The agent
did not mention this in its reply.

**Verified.** Step 7 output reads "Baseline snapshot: 10 file(s) copied to
workspace/.baseline", so the baseline was overwritten too: the before/after
comparison that would have exposed the loss was destroyed in the same command.

**Why.** `prepare_workspace.py` overwrites without warning and the skill does
not say that re-preparing destroys local edits. The agent had no reason to
expect it. This is a missing guard, not a judgment error.

**Fix.** Make `prepare_workspace.py` refuse to overwrite a workspace whose files
differ from `.baseline` unless given an explicit flag, and name the drifted
files in the error.

---

### F-034 — Four ways to fail at a shell one-liner, each one repeated
**Status:** open · **Confidence:** verified · **Sessions:** 2 · **Priority: medium**

**Observed.** 15 tool failures in `01a0a9ed`, in four families:

| family | count | example |
|---|---|---|
| backslash-escaped quotes inside a single-quoted `python -c` | 4 | `f"Block {b[\"_id\"]}"` gives `SyntaxError: unexpected character after line continuation character` |
| `present.py` at the `/skills/` path, which the shell cannot see | 2 | `python3 /skills/working-with-google-sheets/scripts/present.py` gives `No such file or directory` |
| `glob` with no `path`, walking the sandbox root | 3 | `glob(pattern=*direct-address*)` gives `cannot glob the sandbox root` |
| `git` in a sandbox that is not a repository | 3 | `git status` gives `fatal: not a git repository` |

**Verified.** The `/skills/` path failure happens although `instructions.md`
states the rule plainly: "/skills/ is a mount the shell cannot see. The sheet
skill's scripts are baked into the image at `/opt/cce/scripts/`." Both offending
subagents had read the file that says so.

**Why.** Each family has a different cause and they should not be fixed
together. The escaping errors are model error and unfixable by instruction. The
`/skills/` path error is a stated rule that did not survive into a subagent's
working set, which argues for putting the path in the brief rather than
restating the rule. The `glob` and `git` failures are the tool teaching the
agent its own shape, and cost 2 steps each to learn — acceptable, and cheaper to
leave than to document.

---

### F-035 — Chained present.py works; the agent re-ran it anyway
**Status:** open · **Confidence:** verified · **Sessions:** 1 · **Resolves:** F-018

**Observed.** `01a0a9ed` step 4 chains three `present.py` calls with `&&`. Its
output contains `cce_present_v1` three times, once per topic: the chain renders
all three card sets. The agent then re-ran topic_03 alone (step 5) and topic_04
alone (step 6). Both were duplicates. The user saw those two topics twice.

**Verified.** Counted on the raw output: 44,276 characters, three
`cce_present_v1` keys, one per topic file.

**Note.** This answers F-018, which was open pending evidence. Chaining is safe.

---

### F-036 — Present called twice for one look, in five of six sessions
**Status:** open · **Confidence:** verified · **Sessions:** 5 · **Priority: medium**

**Observed.** The pattern is `present.py --measure <file>` followed immediately
by `present.py <file>`: `01a0a9cd` steps 10–11, `01a0a9cf` steps 2 and 6,
`01a0a9d2` steps 9–10, `01a0a9d7` steps 11–12, `01a0a9ed` steps 2–3.

**Verified.** The full `present.py` output already contains a `counts` object
identical to what `--measure` prints. The first call is never needed.

**Why.** `instructions.md` names the two separately — "Every number you state
must come from that output, or from `--measure`" — which reads as two sources
rather than one that subsumes the other.

**Fix.** Say that the full output already carries the counts, and that
`--measure` is for when the content is not wanted.

---

### F-037 — A real title replaced with "Intro", unreported
**Status:** open · **Confidence:** verified · **Sessions:** 1 · **Recurrence of:** F-025

**Diagnosis superseded by F-046 (2026-09-16):** the fixed labels are instructed by the writing-style reference, not an agent defect. The unreported-change half of this entry stands.

**Observed.** In `01a0a9ed`, topic_01 block 0 went from
`[Transition] The Power of Hand Tools` to `[Transition] Intro`. Seven titles
changed in the session; six are reasonable rewrites. The orchestrator's report
mentions no retitling at all.

**Verified.** Read from the `label` / `old_label` pair in the `present.py`
output, which reports the change correctly. The agent had the evidence on screen
and did not read it.

**Why.** This is the second occurrence of the same specific failure — a titled
slide flattened to the word "Intro" — recorded in F-025. Two occurrences of one
word is unlikely to be chance. Worth checking whether a style reference uses
"Intro" as a placeholder.

---

### F-038 — Probing for a workspace that cannot exist yet
**Status:** open · **Confidence:** verified · **Sessions:** 1 · **Recurrence of:** F-029

**Observed.** `01a0a9d7` step 4 calls `ls(path=/workspace)` in a fresh thread,
before any prepare has run, and gets `path_not_found`.

**Verified.** `instructions.md` Getting oriented anticipates this in as many
words: "probing for it before any workspace exists just produces a
`file_not_found` you already expected." The sentence is about `manifest.json`;
the agent probed the directory instead.

**Why.** The rule was written against one path and the agent used another. A
rule that names a file does not generalise to its directory.

---

### F-039 — Per-block word counts stated without measuring them
**Status:** fixed · **Confidence:** verified · **Sessions:** 1 · **Priority: medium**

**Fixed 2026-09-16.** Same fix as F-030.

**Observed.** `01a0a9eb` step 12 reports to the user: "in Topic 2, several
slides reach 75–114 words (Block 1 is 114 words, Block 5 is 84 words)."

**Verified.** The only measurement in that session is
`present.py --measure`, which reported topic_02 at 607 words for the whole file.
No per-block number was computed. The figures were counted by eye.

**Why.** Same root as F-030: the constraint is per-block, the instrument is
per-file. Here it produced an unverified number in front of the user, which
`instructions.md` names as the specific thing to avoid: "An unverified count
reads exactly like a verified one to the person trusting it." Fixing F-030 fixes
this.

---

### F-040 — Four calls spent rediscovering a flag already on screen
**Status:** open · **Confidence:** verified · **Sessions:** 1 · **Priority: low**

**Observed.** In `01a0a9d2` the user asked to check one named tab. Step 2 ran
`prepare_workspace.py --help`, whose output includes `--source-tab SOURCE_TAB`.
Steps 3–6 then read `_common.py` twice and wrote two ad-hoc scripts that import
`open_sheet` to read the tab directly. Step 7 finally used `--source-tab`.

**Verified.** `--source-tab` is the second line of the step 2 output.

**Why.** The flag was visible and not used for four steps. The `--help` text
lists it in a usage block without saying what it is for, and the skill's own
prose describes source tabs in terms of mode selection rather than "read a
different tab". Plain model error is also possible; one session is not enough to
separate them.

---

### F-041 — The default source tab taken silently, on a sheet split into per-topic tabs
**Status:** fixed · **Confidence:** verified · **Sessions:** 3 · **Priority: high**

**Fixed 2026-09-16.** `list_tabs.py` classifies every tab by its header row and
names the candidates when more than one real source tab exists; the sheets skill
requires the agent to ask rather than take the default.

**Observed.** In `01a0a9cd` the user pasted a sheet and asked about "Topic 1:
Pipe Preparation". Step 2 ran `list_tabs.py`, whose output included `Slide
Chunks`, `Topic 1: Pipe Preparation`, `Topic 2: Threaded Joints` and `Topic 2:
Threaded Joints (Revised)`. Step 3 ran `prepare_workspace.py` with no
`--source-tab`, taking the default `Slide Chunks`. The agent then reviewed and
(in `01a0a9cf`) edited content from the wrong tab. Two turns were spent before
`01a0a9d2` reached the tab the user meant.

**Verified.** Reported by the user, who confirmed the intent was the topic tab:
the team had split the slide chunks into per-topic tabs. Re-checked against the
live sheet: five tabs carry the slide-chunk header signature, three of them real
sources. The second sheet in the same session (`01a0a9eb`) has exactly one, so
the condition distinguishes the two cases rather than firing everywhere.

**Why.** Not model error and not a missing instruction — the skill already said
tab names vary and to confirm with `list_tabs.py`, and the agent *did* run it.
The output was a bare list of 21 names in which nothing marked `Slide Chunks` as
one of several equally valid choices. A list that does not distinguish its
entries reads as one obvious answer plus noise. The fix belongs where the agent
already looks, which is the same shape as F-030.

**Note.** Detection is exact rather than a name heuristic: a tab holds slide
chunks if its header row carries a Topic column and a Slide Chunk column, the
same test `prepare_workspace.py` applies when reading it. Backups and
`(Revised)` tabs carry that header too and are excluded from the count by name,
because counting them made the warning fire on a sheet with one real source.
Cost is one batched API call, measured at 0.42s for 21 tabs.

---

### F-042 — The editor subagent was told to "count with the shell"
**Status:** fixed · **Confidence:** verified · **Sessions:** 1 · **Priority: highest**

**Fixed 2026-09-16.** The editor system prompt in `agent.py` now names
`present.py --measure` and `per_block` as the only source of counts, and says
not to write a counter.

**Observed.** Subagent 4 in `01a0a9ed` made 37 tool calls against the smallest
file in the session — 6 blocks, 399 words — and 2 of them changed it. It was
the largest span in the fan-out; the largest file, at 34 blocks, took 23.

**Verified.** The editor's own `system_prompt` said, in Editing mechanics:

> Count with the shell before and after any request that names a direction

and in the report spec:

> What you changed — block by block, with before -> after counts you measured.

The two together require a per-block count and point at the shell to get it.
The subagents complied exactly. Calls 13–21 are nine consecutive calls reading
`commit_workspace.py`, `present.py` and `_presentation.py` — twice each for the
last — to find `block_body()` and copy how it counts words. Calls 24, 25, 36 and
37 then build the counter, two of them failing on backslash-escaped quotes.

**Why.** This is the stronger half of F-030. The missing `per_block` left the
agent without the number; this line told it where to go instead, and it went
there. A tool gap and an instruction pointing away from the tool compound: the
agent read the measuring tool's source rather than its output.

**Note.** F-030 was recorded as "the single largest source of steps in the
pass", which over-credited it. On this span the two fixes together account for
13 of 37 calls. The rest is F-043 and F-044.

---

### F-043 — Six greps to re-check an edit already read back
**Status:** fixed · **Confidence:** verified · **Sessions:** 2 · **Priority: medium**

**Fixed 2026-09-16.** Step 5 of the editor prompt now says to re-read once and
not to grep the file for what was just written.

**Observed.** Subagent 4 calls 26–31 are six greps of its own file, four
returning "No matches found", followed by a full re-read at 34. The same shape
appears in the orchestrator's own turn: `01a0a9cf` steps 3–5 grep the file it
had just written for an em dash and for `**`.

**Verified.** Call 22 wrote the file; call 23 measured it; call 34 read all 71
lines back. Every grep between them searched text the agent had authored in
that same turn.

**Why.** Step 5 of the editor prompt requires confirming "every change you
intend to report is actually present", and does not say that one read satisfies
it. A per-item confirmation loop is a reasonable reading of it. Compounded by
F-001: a grep that correctly finds nothing is rendered as a failed command, so
a negative result reads as something to retry rather than an answer.

---

### F-044 — An editor reading outside its own brief
**Status:** fixed · **Confidence:** verified · **Sessions:** 1 · **Priority: medium**

**Fixed 2026-09-16.** The editor prompt now names what it may open, and forbids
looking in `/memories/`.

**Observed.** Subagent 4, owner of topic 04, opened: `/memories/agent/AGENTS.md`
(call 4), `ls /memories/agent` (5), `ls /memories/agent/editors` (6),
`/memories/agent/editors/ed_f84a65233902.md` (7), the sheets SKILL.md (12), and
`/workspace/topics/topic_02_toolboxes-bags.md` (11) — a file belonging to
subagent 2, which was editing it concurrently.

**Verified.** Calls 5–7 re-derived something the brief already carried: "Slide
Length: Strictly keep every slide under 60 words (editor ed_f84a65233902
preference)". Call 4 re-read AGENTS.md, which is hot memory and already in the
system prompt of that very run. `instructions.md` forbids listing the editors
directory — "Do not list the editors directory, do not read or write another
editor's file" — but that rule is in the *coordinator's* instructions, and the
editor subagent has its own system prompt, which said nothing about memory at
all.

**Why.** A boundary stated in one agent's instructions does not reach another
agent. The editor prompt had a "Your file only" rule that governed *edits* and
explicitly allowed reading `outline.md`, which reads as permission to read
widely. Reading a sibling topic mid-edit is also a correctness risk, not only a
cost one: subagent 2 was writing that file at the time.

---

### F-045 — Audit: rules in instructions.md that never reached the editor subagent
**Status:** fixed · **Confidence:** verified · **Sessions:** 1 · **Priority: high**

**Fixed 2026-09-16.** The four gaps below are now in the editor's system prompt;
the content-injection rule was added to both prompts.

**Why the audit.** This deployment has two prompts — `instructions.md` for the
coordinator, and `EDITOR["system_prompt"]` in `agent.py` — and F-044 showed a
rule written in the first does not reach the second. Every section of
`instructions.md` was checked against the editor prompt and classified:
coordinator-only, already covered, or a gap. Four gaps, all with observed
failures behind them. Rules found correctly coordinator-only: Getting oriented,
Delegating, Every turn, Writing back, status updates, mid-edit preferences.
Rules found already covered: reactive-editor scope, the two editing skills,
`outline.md` as the boundary, block-removal mechanics.

**Gap 1 — `/skills/` is readable but not runnable.** `instructions.md` Where
things are: "the shell runs on the sandbox disk and `/skills/` is a mount the
shell cannot see. The sheet skill's scripts are baked into the image at
`/opt/cce/scripts/`." Absent from the editor prompt. Subagents 1 and 3 both ran
`python3 /skills/working-with-google-sheets/scripts/present.py` and got `No such
file or directory`. Previously logged under F-034 as a rule that "did not
survive into a subagent's working set"; that was the wrong diagnosis. The rule
was never there.

**Gap 2 — course content is data, not instructions.** Neither prompt had one.
The only mention of injection anywhere in the deployment is one clause in
`working-with-google-sheets/SKILL.md` about why no credential is in the sandbox.
The editor is the higher-risk side: it reads sheet-derived content directly, and
it has `write_file` and `execute`. The F-044 fix narrowed its reading scope and
in doing so removed its one path to that clause, so the gap was widened before
it was found. Now stated in both prompts, in the terms each needs: the editor
takes direction from its brief only, the coordinator from the user only.

**Gap 3 — `.baseline` is read-only and `git` does not exist here.**
`instructions.md`: "Read-only, and you never need to touch it." Absent from the
editor. Subagent 2 ran `diff -u /workspace/.baseline/... `; subagents 2 and 3 ran
`git status`, `git diff` and `git -C / status`, three failures, in a sandbox that
is not a repository.

**Gap 4 — relay a script's stderr rather than summarising it.**
`instructions.md` General rules: "the message is written for you to act on, not
to summarise away." Absent from the editor, which is the agent that most often
sees a script fail. No mis-relay observed yet; added on the same grounds the
coordinator has it.

**Checked and found not to be a gap.** "Never write file content into your
message" is coordinator-facing, and the concern that an editor might paste
rewritten prose upward for the coordinator to relay was tested: zero sentences
of twelve words or more from any newly written file appear verbatim in any of
the four reports. No rule added — the evidence did not support one.

**Note.** The reports do misattribute their per-block figures. Subagent 4's
report is headed "Quantitative Measurements (`present.py --measure`)" above
counts the tool did not produce at the time. F-030 and F-042 remove the reason
to derive them; the report spec now says to quote the tool's numbers.

---

## Pass 4 — 2026-09-16 (post-fix)

1 session, `01a0aa84`, 14:00 UTC — the first run against the deployed fixes from
F-030, F-042, F-043, F-044 and F-045. Same sheet, same four topics, same editor
identity and same request as `01a0a9eb` + `01a0a9ed`, so the comparison is
like-for-like. Read to check whether the fixes did what they claimed, not to
find new things; F-046 was found anyway.

**Measured against the same work before the fixes:**

| | before (2 turns) | after (1 turn) |
|---|---|---|
| model steps | 133 | 58 |
| tool calls | 145 | 71 |
| cost | $1.25 | $0.78 |
| subagent steps | 22 / 31 / 26 / 35 = 114 | 13 / 7 / 9 / 7 = 36 |
| subagent tool failures | 15 | **0** |
| ad-hoc `python -c` counters | 25 | **0** |
| `/skills/` path in shell | 2 | **0** |
| `git` in the sandbox | 3 | **0** |
| hand-diffing `.baseline` | 1 | **0** |
| `ls /memories/...` | 6 | **0** |

The editor loop is now read, measure, write, measure, verify. `per_block`
appears in 21 tool outputs and the coordinator's final report quotes it as a
per-block range for every topic — a per-block claim it can now substantiate,
which F-039 was about.

---

### F-046 — The placeholder titles are the style guide's instruction, not a defect
**Status:** open · **Confidence:** verified · **Sessions:** 3 · **Supersedes the
diagnosis in:** F-025, F-037 · **Priority: high — needs a decision, not a fix**

**Observed.** In `01a0aa84`, five of the six title changes replace a written
title with a fixed generic one, across three subagents independently:

| file | before | after |
|---|---|---|
| topic_03 | `Precision Alignment` | `Intro` |
| topic_04 | `Matching the Blade to the Job` | `Intro` |
| topic_02 | `Organizing for Success` | `Topic Summary` |
| topic_03 | `Level Basics Summary` | `Topic Summary` |
| topic_04 | `Hacksaw Blade Selection Summary` | `Topic Summary` |

The sixth, `Testing Horizontal Accuracy` -> `Testing Level Accuracy`, is a
correct consequence of merging two slides.

**Verified.** `skills/editing-slide-chunks/references/writing-styles/direct-address-field-guide.md`
line 98 says, in full:

> "Intro" is the fixed title for a topic's opening Transition slide; "Topic
> Summary" (or "Summary") is the fixed title for the closing Summary slide —
> reuse these exact labels rather than inventing new ones.

Lines 89–90 repeat it, and `plain-sequential-descriptive.md` line 88 says the
same for Summary slides. The coordinator selected Direct-Address Field Guide as
the style and named it in all four briefs. The editors read the style file and
complied exactly.

**Why this entry corrects two earlier ones.** F-025 recorded two titles
"silently changed to Intro" and F-037 recorded a third as a regression the agent
should have caught. Both framed it as agent error. It is not: it is the
documented house style being applied. F-037 guessed the cause correctly ("worth
checking whether a style reference uses 'Intro' as a placeholder") and that
guess is now confirmed. Three sessions of consistent behaviour across at least
five different subagents is compliance, not drift.

**What is still a real problem.** The retitling is never reported. The word
"title" does not appear anywhere in the `01a0aa84` final report, which otherwise
runs to per-block word ranges for all four topics. `present.py` reports the
change correctly in `old_label` / `label`; nobody reads it. A user who did not
want `Intro` would have no way to notice from the report.

**The decision.** Whether these fixed labels are wanted is the user's call, not
a defect to fix. Relevant: the user has said the `AGENTS.md` house style is now
the most evolved one and the skill style files are "kind of obsolete", kept in
case something in them is not yet captured. `AGENTS.md` says nothing about
titles, so nothing currently contradicts the style file. Three options, none
taken yet:

1. Drop the fixed-label rule from the style files, and let titles be written.
2. Keep it, and require the report to state retitles so the user sees them.
3. Keep it and say so in `AGENTS.md`, making it the house rule rather than one
   style's rule.

Option 2 is worth doing whichever of 1 and 3 is chosen: a title change reaching
the sheet unmentioned is the same class of problem as an unreported deletion.
