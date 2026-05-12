# End-to-End Course Generation

> One pipeline, five agents, one Google Sheet. The pipeline takes a course idea (a name, a target audience, and either a tentative outline, a set of curated references, or both) and produces a complete, source-grounded SkillCat course: structured outline, research notes per Learning Objective, narrated slide sequences, ready-to-place graphics, and a typed assessment. The content team's main tuning surfaces are the course's input Sheet and the per-course AI Checklist Sheet — no code changes needed to steer style, depth, or quality per course.

This datasheet is the umbrella overview. Each stage has its own detailed datasheet in this folder.

---

## At a glance

| | |
| :--- | :--- |
| **Owners** | Dilip, Niket, Om |
| **Status** | ~90% (No Sources mode); Internal-Sources and External-Sources targeting Q2 closeout |
| **Stages** | 6 agents in a single codebase (Streamlit pages, shared sheet-driven orchestration) |
| **Central artifact** | One Google Sheet per course (the "Course Sheet") that evolves stage-by-stage |
| **User tuning surface** | The Course Sheet's `Course info` tab + the per-course AI Checklist Sheet (one tab per agent) |
| **Primary user** | Content team |
| **Q2 goal** | 3+ approved courses generated end-to-end |
| **Requirements** | R4 (Workflow Improvement), R8 (Content & Curriculum) |
| **Impact rating** | Very High across all three modes |

---

## The three modes are not three pipelines

The briefing names three projects — **Auto Course Gen (No Sources / Internal Sources / External Sources)**. These are **not three pipelines**. They are three common ways of configuring the same pipeline by what you put into the Course Info tab. Every mode runs the same agents in the same order; the difference is which features each agent activates.

| Mode | Course Info preset | What changes |
| :--- | :--- | :--- |
| **No Sources** | `Research Sources` blank or limited; `External References` blank | Course Outline skips external research streams; downstream agents work from the LLM's own grounding + the manual Base Outline. |
| **Internal Sources** | `Research Sources = video` (or `video, deep`); `External References` blank | Course Outline pulls from SkillCat's internal video sources (HVAC School / Ty Branaman channels), embeds them, and grounds research notes / slides / assessment in them. |
| **External Sources** | `Research Sources = video, web, deep`; `External References` populated with user URLs | Full external sweep — agent-discovered web + video + deep research, blended with the user's curated URLs. |

For any course, you can also pick **Outline Stage = `initial`** (let the AI improve a rough outline) or `final` (trust the user's detailed outline; just collect references for it). Most courses run in **Final** stage.

---

## End-to-end block diagram

```
                       ┌────────────────────────┐
                       │  Content team enters:  │
                       │   • Course Name        │
                       │   • Drive folder       │
                       └────────────┬───────────┘
                                    ▼
   ┌──────────────────────────────────────────────────────────────────┐
   │ 0. Template Sheet Setup                                          │
   │  Copies master templates → creates the two artifacts that every  │
   │  later agent uses:                                               │
   │    • "AI course: {name}"     ← the Course Sheet                  │
   │    • "AI checklist: {name}"  ← the AI Checklist Sheet            │
   │  Validates user has filled Course Info + Base Outline.           │
   └──────────────────────────────────┬───────────────────────────────┘
                                      ▼
   ┌──────────────────────────────────────────────────────────────────┐
   │ 1. Course Outline                                                │
   │  Reads Course Info flags (Research Sources, Outline Stage,       │
   │  Topic Deep Research, External References).                      │
   │   • Runs enabled research sub-pipelines: Videos, Web, Deep       │
   │   • Consolidates partial outlines (Initial stage only)           │
   │   • Enhances outline with Learning Objectives (Initial + flag)   │
   │   • Loads all references into per-course Chroma vectorstore      │
   │   • Per-LO RAG retrieval → populates References on Final Outline │
   │  → Final Outline tab (Topic / Subtopic / LO + References)        │
   └──────────────────────────────────┬───────────────────────────────┘
                                      ▼
   ┌──────────────────────────────────────────────────────────────────┐
   │ 2. Research Notes                                                │
   │  For each LO row on Final Outline:                               │
   │   • Loads underlying content for each reference                  │
   │     (web body, video transcript, Drive video transcript)         │
   │   • Two generation paths driven by `Reference usage`:            │
   │       Content → paraphrase into narrative notes                  │
   │       Video   → preserve transcript verbatim + embed timestamps  │
   │   • Internal 3-iteration reviewer/reviser quality gate           │
   │   • Checklist-driven parallel-reviewer revise loop               │
   │     (Research Notes Checklist tab)                               │
   │   • Inline image placement — agent previews each image and       │
   │     positions it next to the paragraph it illustrates            │
   │  → Final Outline → `research_notes` column                       │
   └──────────────────────────────────┬───────────────────────────────┘
                                      ▼
   ┌──────────────────────────────────────────────────────────────────┐
   │ 3. Slide Chunks                                                  │
   │  Per Topic, one LLM call generates a coherent slide sequence:    │
   │     Transition (per subtopic) → Content (20-30s narration) →     │
   │     Summary (when ≥3 Content slides) → Video (where notes        │
   │     pointed at a timestamped clip)                               │
   │   • Internal 3-iteration reviewer/reviser quality gate           │
   │   • Parsing: LLM expands the <slides> blob to one row per slide  │
   │   • Checklist-driven parallel-reviewer revise loop               │
   │     (Slide Chunks Checklist tab)                                 │
   │  → Slide Chunks tab (one row per slide)                          │
   └──────────────────────────────────┬───────────────────────────────┘
                                      ▼
   ┌──────────────────────────────────────────────────────────────────┐
   │ 4. Graphics                                                      │
   │  For each voiceover (sub-unit of a slide chunk):                 │
   │   • Storyboard + segmentation by visual-density preference       │
   │   • Internal search (SkillCat Drive image library, HVAC School,  │
   │     Ty Branaman YouTube) with review-and-revise pass             │
   │   • External search (web + non-internal YouTube channels)        │
   │   • Final comparison → pick the best image / video link          │
   │   • Feedback UI: approve / re-search / AI-generate / AI-edit     │
   │  → Slide Chunks → `final_graphics_definition` column             │
   └──────────────────────────────────┬───────────────────────────────┘
                                      ▼
   ┌──────────────────────────────────────────────────────────────────┐
   │ 5. Assessment                                                    │
   │  Per Topic, one LLM call generates a mixed question set:         │
   │     Multiple Choice (knowledge + scenario), True/False,          │
   │     Select All That Apply, Matching (conditional)                │
   │   • Per-question × per-Task review-and-revise loop               │
   │     (Review Agent Checklist tab; 2 iters per Task)               │
   │   • Aggregate verdicts pass (Assessment Checklist tab)           │
   │   • Per-question verdict matrix (Review Agent Checklist tab)     │
   │  → Final Assessment tab + populated checklist verdict columns    │
   └──────────────────────────────────┬───────────────────────────────┘
                                      ▼
                          Course ready for QA / publishing
```

---

## How the Course Sheet evolves

The Course Sheet is the **central artifact**. Every agent reads from and writes to it. Tabs accumulate as the pipeline progresses.

```
After stage 0 (Template Setup):
  ┌────────────────────────────────────────────────────────┐
  │ Course info • Base Outline • (empty downstream tabs)   │
  └────────────────────────────────────────────────────────┘

After stage 1 (Course Outline):
  + Videos Research, Video Chunks
  + Preliminary Research, Article Content, Relevant Info, Research Summaries
  + Deep Research
  + Outline Consolidation, Outline Review        ← Initial stage only
  + Topic Outline, Topic Deep Research, …        ← Initial stage + Topic Deep
                                                   Research flag only
  + All References (chunked, source-tagged)
  + Final Outline (Topic / Subtopic / LO / References / Reference type /
                   Reference usage)
  + Chroma vectorstore (persisted to Drive, not a tab)

After stage 2 (Research Notes):
  Final Outline gains:
    + context_0, context_1, …                    ← per-row reference content
    + source_links, as_is_sources, content_sources, web_links, video_links
    + research_notes                             ← the main deliverable
    + inline_image_placement_status
  + Backup tabs (hidden)

After stage 3 (Slide Chunks):
  Final Outline → slide_chunks (per-Topic blob)
  + Slide Chunks tab (one row per slide:
      Topic / Subtopic / Slide Type / Slide Chunk Title / Slide Chunk)
  + Backup tab (hidden)

After stage 4 (Graphics):
  Slide Chunks gains:
    + final_graphics_definition                  ← graphics link per voiceover
    + alternate visuals, feedback columns

After stage 5 (Assessment):
  + Assessment questions (hidden) — Step 1 raw XML blob
  + Final Assessment (one row per typed question)
  AI Checklist Sheet gains:
    + LLM Based Output column on Assessment Checklist tab
    + Question 1, Question 2, … columns on Review Agent Checklist tab
```

---

## The AI Checklist Sheet — one tuning surface, five agents

Every agent that does revisions reads its rubric from a dedicated tab on the per-course AI Checklist Sheet. Editing those tabs is how the content team changes the agent's behaviour per course without touching code.

| Tab | Read by | What changes when you edit it |
| :--- | :--- | :--- |
| `Course Outline Checklist` | Course Outline (Initial stage only) | Criteria for outline structure, topic coverage, LO quality. |
| `Research Notes Checklist` | Research Notes | Style, voice, citation rules, technical accuracy bar. |
| `Slide Chunks Checklist` | Slide Chunks | Slide tone, density, transitions, summary rules. |
| `Review Agent Checklist` | Assessment (Step 2 revisions + Step 4 per-question verdicts) | Question quality criteria — distractor plausibility, ambiguity, alignment with LO. |
| `Assessment Checklist` | Assessment (Step 3 aggregate verdicts) | Coverage criteria for the whole question bank. |

Most agents use the **same v2 review pattern**: parallel reviewers per `Task × Scope` slice → aggregator LLM → single reviser with block-level CRUD tools, re-reviewed with diffs until criteria pass or `Iteration Count` is hit. The Assessment agent uses a different (sequential per-question, per-Task) loop because the output is a typed question rather than free-form text.

---

## Per-stage summary

For full detail on each, see the agent-specific datasheet alongside this file.

| # | Stage | Generative? | Primary input → output | Tuning surface |
| :- | :--- | :--- | :--- | :--- |
| 0 | **Template Sheet Setup** | No (deterministic) | Drive folder + Course name → Course Sheet + AI Checklist Sheet | — |
| 1 | **Course Outline** | Yes (multi-LLM) | Course Info + Base Outline → Final Outline + per-course vectorstore | Course Info flags, External References, Course Outline Checklist (Initial only) |
| 2 | **Research Notes** | Yes (RAG + LLM) | Final Outline References → `research_notes` per LO | `References` / `Reference type` / `Reference usage` columns; External References; Research Notes Checklist |
| 3 | **Slide Chunks** | Yes (LLM) | `research_notes` → per-slide rows on Slide Chunks tab | Slide Chunks Checklist |
| 4 | **Graphics** | Yes (LLM + search) | Slide Chunks → graphics link per voiceover | Visual-density preference; Reference Image Pool (Q2 closeout); Layout Agent (Q2 closeout) |
| 5 | **Assessment** | Yes (LLM) | Slide Chunks → Final Assessment + verdict matrices | Review Agent Checklist + Assessment Checklist |

---

## Where the human stays in the loop

The pipeline is human-supervised by design. Every stage has at least one explicit checkpoint where the user reads the output before the next stage runs:

- **Setup**: user fills Course Info + Base Outline, agent validates.
- **Course Outline (Initial stage)**: user approves the consolidated outline (`Approved` / `Rejected` verdict).
- **Research Notes**: manual review of `research_notes`; pre/post-checklist diff view.
- **Slide Chunks**: pre/post-checklist diff view.
- **Graphics**: per-voiceover approve / re-search / AI-generate / AI-edit UI.
- **Assessment**: read the two verdict matrices; tune checklist criteria; re-run.

The content team can also override the agents at any stage by hand-editing the Sheet — most notably, the per-row `References` / `Reference type` / `Reference usage` columns on Final Outline let them direct exactly which sources should ground a Learning Objective's notes.

---

## What this means for SkillCat

- **Static / certification courses at volume without dedicated visual designers.** The pipeline can take a course concept all the way to a publishable Sheet with slides, narration, graphics, and assessment.
- **A single tuning surface (the AI Checklist Sheet) for course-specific quality control.** Content team owns the rubric; engineering doesn't need to redeploy.
- **A single source of truth (the Course Sheet) for the whole course lifecycle.** Every artefact — outline, references, notes, slides, graphics, questions, QA verdicts — lives on one Sheet, addressable by Topic / Subtopic / Learning Objective.
- **A path to scaling internal sources.** Once an internal source (HVAC School, Ty Branaman, future archives) is wired into the Course Outline agent's research loader, every subsequent course can pull from it automatically.

---

## Status, open work, and quality bar (as of 2026-05)

- **No Sources mode:** ~90% ready.
- **Internal Sources mode:** in progress, targeting Q2 close.
- **External Sources mode:** in progress, targeting Q2 close.
- **Graphics agent:** ~75%, polishing. Two incremental improvements in flight — Layout Agent (composite/multi-pane images, Niket) and Reference Image Pool (user-supplied reference pool, Om).
- **Q2 goal:** 3+ approved courses generated end-to-end.
- **Proposed quality bar (Graphics):** ≥80 % of generated images need no human swap. Minimum bar: agent produces a working static informational course with minimal human intervention.
- **Course-level quality bar:** still being defined per stage; the AI Checklist Sheet tabs are the current vehicle for encoding it.

---

## Related datasheets

- `template_sheet_setup_agent.md`
- `course_outline_agent.md`
- `research_notes_agent.md`
- `slide_chunks_agent.md`
- `assessment_agent.md`
- *(Graphics — covered in detail in the existing Graphics Agent term sheet.)*
