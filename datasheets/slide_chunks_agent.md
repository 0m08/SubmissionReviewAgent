# Slide Chunks Agent

> Stage 3 of the Auto Course Gen pipeline. Reads the Final Outline (Topics, Subtopics, Learning Objectives, and the `research_notes` the previous agent produced) and turns each Topic into an ordered sequence of narration-ready slides — Transition, Content, Summary, and Video slides. The slide sequence is then parsed into structured rows on a `Slide Chunks` tab and run through a checklist-driven review-and-revise loop that the content team can tune per course.

---

## At a glance

| | |
| :--- | :--- |
| **Owner** | Dilip |
| **Status** | Done / stable |
| **Pipeline stage** | 3 — slide-chunk authoring |
| **Generative?** | Yes — Gemini 3 Flash by default for generation, parsing, reviewers, and reviser |
| **Interface** | Streamlit page (sidebar → Agentic Workflows → *Slide Chunks*) |
| **Granularity** | Generation is **per Topic**; parsed output is **per slide** (one row per slide on the Slide Chunks tab) |
| **Typical runtime** | ~25–45 minutes total (15–30 min dominated by the checklist loop) |
| **Run frequency** | Once per course, with human-in-the-loop iteration |
| **Primary user** | Content team |
| **External APIs** | Google Gemini, Gemini Google-Search grounding (reviser's web lookups), Google Sheets / Drive |
| **Requirements** | R4 (Workflow Improvement), R8 (Content & Curriculum) |

---

## What it does

For every Topic on the `Final Outline` tab:

1. Groups all the subtopics, Learning Objectives, and per-LO `research_notes` under that Topic.
2. Asks an LLM to author an ordered slide sequence covering the entire Topic — one Transition slide per subtopic, multiple Content slides, an optional Summary slide, and any Video slides where the research notes pointed at a YouTube timestamp.
3. Runs an internal 3-iteration reviewer/reviser quality gate on that sequence before writing it back to the sheet.
4. Parses the LLM blob into structured rows on a separate `Slide Chunks` tab — one row per slide.
5. Runs a tunable, checklist-driven parallel-reviewer loop that revises the parsed slide chunks until they pass the user's quality criteria.
6. Surfaces a side-by-side diff (pre vs post checklist) for the user to confirm.

The `Slide Chunks` tab is the contract for the Graphics agent.

---

## Block diagram

```
   Final Outline (Topic / Subtopic / LO / research_notes)
                              │
                              ▼
   ┌─────────────────────────────────────────────────────┐
   │ 1. Generate Slide Chunks from the Research Notes    │
   │  • Group rows by Topic                              │
   │  • Build one LLM prompt per Topic with all          │
   │    subtopics + LOs + research notes                 │
   │  • LLM emits an XML <slides> blob:                  │
   │      Transition / Content / Summary / Video slides  │
   │  • Internal 3-iteration review→revise loop          │
   │    (10 quality criteria, str_replace/full_overwrite │
   │    edits) runs before the blob is saved             │
   │  → write to Final Outline → `slide_chunks` column   │
   │    (first row of each Topic group)                  │
   └─────────────────────────────────────────────────────┘
                              │
                              ▼
   ┌─────────────────────────────────────────────────────┐
   │ 2. Slide Chunks Parsing                             │
   │  • Split blob on `---` (fallback: blank lines)      │
   │  • LLM parses each block into a Pydantic            │
   │    SlideChunk struct                                │
   │  • Two levels of parallelism                        │
   │    (rows × blocks; max_workers=5 each)              │
   │  → write to `Slide Chunks` tab:                     │
   │      Topic / Subtopic / Slide Type /                │
   │      Slide Chunk Title / Slide Chunk                │
   │    (one row per slide)                              │
   └─────────────────────────────────────────────────────┘
                              │
                              ▼
   ┌─────────────────────────────────────────────────────┐
   │ 3. Slide Chunks Checklist Review & Revise (v2)      │
   │  • Back up Slide Chunks tab                         │
   │  • Parallel reviewers per Task × Scope slice        │
   │    (Global / Topic / Subtopic / Learning Objective) │
   │  • Aggregator LLM merges feedback                   │
   │  • Single reviser with CRUD tools                   │
   │    (Create/Read/Update/Delete/StrReplace/Search)    │
   │  • Topic-level parallelization across the loop      │
   │  • Re-review until criteria pass or                 │
   │    `Iteration Count` is hit                         │
   └─────────────────────────────────────────────────────┘
                              │
                              ▼
   ┌─────────────────────────────────────────────────────┐
   │ 4. Visualize Slide Chunks Diff (manual)             │
   │  • Pre vs post-checklist diff per row,              │
   │    red/green highlighting                           │
   │  • User confirms                                    │
   └─────────────────────────────────────────────────────┘
                              │
                              ▼
              `Slide Chunks` tab (handoff to Graphics agent)
```

---

## Inputs

| Source | Field / Tab | Role |
| :--- | :--- | :--- |
| Course Sheet → `Final Outline` | Topic, Subtopic, Learning Objectives, `research_notes` | Generator groups rows by Topic and feeds the full set of LOs + notes for each Topic into a single LLM prompt |
| Course Sheet → `Course info` | Course Name, Target Audience & Industry | Threaded into every LLM prompt as context |
| AI Checklist Sheet → `Slide Chunks Checklist` tab | Tasks + scope + criteria + examples + corrective operations + (optional) Iteration Count + Tools | Drives the post-parsing checklist loop |

---

## Outputs

| Tab / Artifact | Description |
| :--- | :--- |
| `Final Outline` → `slide_chunks` column | The raw XML-style slide sequence produced by the generator (written to the first row of each Topic group). |
| **`Slide Chunks` tab** | The main deliverable. One row per slide, with columns: `Topic`, `Subtopic`, `Slide Type` (Transition / Content / Summary / Video), `Slide Chunk Title`, `Slide Chunk` (narration text or, for Video slides, the embedded video metadata + transcript). Consumed directly by the Graphics agent. |
| `Backup Slide Chunks Sheet for Delete step of Slide Chunks Checklist` (hidden worksheet) | Pre-checklist snapshot used both by the diff view and to safely restore state if the checklist step is deleted/re-run. |

---

## Functional description

The pipeline runs as a DAG of named steps; the user runs each step from the Streamlit page.

**1. Generate Slide Chunks from the Research Notes** *(automatic, per Topic)*

- Reads `Final Outline` and groups rows by `Topic`. Within each Topic, deduplicates per-subtopic research notes and assembles a structured prompt block (`<subtopic>` blocks containing Subtopic name, Learning Objectives, and Research Notes).
- For each Topic, runs a single LLM call with detailed rules: plain technician voice, 20–30 s narration per Content slide, one Transition slide per Subtopic (teaser hook, no teaching), one Summary slide per Topic (only if there are ≥3 Content slides), Video slides created where research notes point at a timestamped YouTube chunk.
- The LLM is required to preserve every markdown image link (`![alt](url)` and `[alt](url)`) verbatim and inline with the relevant narration.
- After the initial emit, runs an **internal review-revise loop** (max 3 iterations): a reviewer LLM checks 10 quality criteria (subtopic coverage, inter-slide redundancy, flow, attribution, summary completeness, missing/invented data, title quality, tone, etc.) and a reviser LLM uses `str_replace` / `full_overwrite` to fix flagged issues. Loops exit early if the reviewer reports no failures.
- Topics are processed in parallel (`ThreadPoolExecutor`, default 5 workers).
- Writes the final `<slides>` blob into the `slide_chunks` column on the first row of each Topic group.

**2. Slide Chunks Parsing** *(automatic, per slide block)*

- Reads the `slide_chunks` column on `Final Outline`. Splits each cell on `---` separators (falling back to blank-line splits if the model didn't emit separators).
- For each block, calls an LLM with a strict parsing prompt that returns a Pydantic `SlideChunk` struct — `subtopic`, `slide_type`, `slide_chunk_title`, `slide_chunk` (the narration body, or `Video_Id` + `Start` + `End` + `Transcript` for Video slides). Inline image links are preserved.
- Two levels of parallelism: rows × blocks, each with `max_workers=5`.
- Writes the structured rows to the `Slide Chunks` tab with columns `Topic`, `Subtopic`, `Slide Type`, `Slide Chunk Title`, `Slide Chunk`.

**3. Slide Chunks Checklist Review and Revise** *(automatic, iterative)*

- Backs up the current `Slide Chunks` tab to `Backup Slide Chunks Sheet for Delete step of Slide Chunks Checklist` (hidden), so the diff view and a clean re-run are both safe.
- Reads the `Slide Chunks Checklist` tab from the AI Checklist Sheet (see *Checklist* section below).
- Runs the **parallel-reviewer → aggregator → single-reviser** pattern, with topic-level parallelization across the loop. Iteration cap is the max `Iteration Count` across all checklist rows (default 1).
- Reviser tools: `Create`, `Read`, `Update`, `Delete`, `StrReplace`, `Search` (Gemini Google-Search grounding for fact lookups), `Stop`.
- The aggregator strips reviewer feedback that targets image links to keep them safe across revisions.
- After each revision, the updated block text is re-parsed back into the structured `Slide Chunks` columns.

**4. Visualize Slide Chunks Diff** *(manual)*

- Reads the live `Slide Chunks` tab and the pre-checklist backup, reconstructs both as concatenated `Topic / Subtopic / Slide Type / Title / Content / ---` strings, and shows a side-by-side red/green diff.
- The user reviews and clicks Confirm; nothing is mutated by the step itself.

---

## The Slide Chunks Checklist — the user's tuning lever

The agent reads its review-and-revise rubric from the **`Slide Chunks Checklist`** tab inside the AI Checklist Sheet (URL stored in Course info → `Checklist Link`). **Editing this tab is how the content team changes slide-chunk style and quality bar per course without code changes.**

The checklist has one row per criterion, with the same columns as the Research Notes Checklist (so authors only need to learn one schema):

| Column | Purpose |
| :--- | :--- |
| **Task** | A named review dimension. Criteria sharing a Task are reviewed together by the same reviewer agent. |
| **Scope** | The slice each reviewer sees per pass: `Global (full output)`, `Topic`, `Subtopic`, or `Learning Objective`. |
| **Criteria Name** | Short identifier (used in feedback and audit trails). |
| **Review Criteria** | The rubric statement itself. |
| **Review Agent Examples** | Worked passing / failing examples shown to the reviewer LLM. |
| **Corrective Operations** | Instructions the reviser LLM follows when this criterion fails. |
| **Reviser Agent Examples** | Worked examples for the reviser LLM, paired with the corrective operations. |
| **Iteration Count** *(optional)* | Per-task maximum number of review-revise rounds; the agent uses the max across all rows as its global cap (default 1). |
| **Review Agent Tools** *(optional)* | Per-task override of which reviser tools are available. |

Behaviour:

- **Parallel review.** All `(Task, Scope)` groups run in parallel. Within each group, scope slices (e.g., one per Topic) also fan out in parallel.
- **Aggregation.** An aggregator LLM merges reviewer outputs into one consolidated feedback document; short-circuits if everything passes or if only one task ran.
- **Single revision pass.** A reviser agent armed with block-level CRUD tools applies all corrective operations in one pass. The `Search` tool is **Gemini Google-Search grounding** — used when a checklist failure requires a fact lookup.
- **Topic-level parallelism.** The whole review-aggregate-revise cycle is parallelized across Topics, so a 10-Topic course doesn't pay a 10× wall-clock cost.
- **Re-review with diffs.** On iterations after the first, reviewers see blockwise diffs of what changed (not the full prior context) so they can focus on whether the changes actually resolved the failure.
- **Image-link guardrails.** The aggregator drops any feedback asking the reviser to remove or rewrite `![](url)` / `[alt](url)` links.

In short: **changing a row in the Slide Chunks Checklist tab changes how the agent writes the slides for that course.** Loosening or tightening a criterion, swapping examples, or rewriting the corrective operation are all valid ways to steer output without touching code.

---

## Operating procedure

1. Confirm the Research Notes agent has populated the `research_notes` column on `Final Outline`.
2. Open *Slide Chunks* in the Streamlit sidebar and load the Course Sheet.
3. *(Optional)* Open the AI Checklist Sheet's `Slide Chunks Checklist` tab and tune any rows that need to behave differently for this course.
4. Run the four steps in order:
   - **Generate Slide Chunks from the Research Notes** — emits one `<slides>` blob per Topic into `Final Outline → slide_chunks`.
   - **Slide Chunks Parsing** — expands the blob into per-slide rows on the `Slide Chunks` tab.
   - **Slide Chunks Checklist Review and Revise** — iterative auto-loop; progress surfaces in the UI.
   - **Visualize Slide Chunks Diff** — eyeball the pre/post diff and confirm.
5. Handoff: the `Slide Chunks` tab is the contract for the Graphics agent.

---

## Role in the pipeline

| Stage | Agent | Reads from | Writes to |
| :--- | :--- | :--- | :--- |
| 0 | Template Sheet Setup | — | Course Sheet, Checklist Sheet |
| 1 | Course Outline | Course Sheet, Checklist Sheet | Research tabs, vectorstore, `Final Outline` (incl. `References`) |
| 2 | Research Notes | `Final Outline` (`References`), upstream worksheets, Checklist Sheet | `Final Outline` (`context_n`, `research_notes`) |
| 3 | **Slide Chunks** *(this agent)* | `Final Outline` (Topic / Subtopic / LO / `research_notes`), `Slide Chunks Checklist` | `Final Outline → slide_chunks`, **`Slide Chunks` tab** |
| 4 | Graphics | `Slide Chunks`, Checklist Sheet | `Slide Chunks → final_graphics_definition` |
| 5 | Assessment | Course Sheet, Checklist Sheet | `Assessment` tab |

---

## Impact

- **Topic-coherent slides.** Because generation is per-Topic (not per-LO or per-Subtopic), the LLM can decide where Transition slides land, where to add a Summary, and how Content slides should chunk a multi-LO subtopic — producing a sequence that reads as one teaching unit rather than a stitched-together row dump.
- **Two quality gates.** An internal 3-iteration reviewer/reviser runs *before* the slide blob is even saved, and the user-tunable Checklist loop runs *after* parsing. The internal loop catches mechanical failures (missing slides, redundancy); the checklist loop catches style / policy issues.
- **Per-course tunability.** The Slide Chunks Checklist tab is the single user-facing surface for changing slide style, tone, density, or content rules — no code redeploy.
- **Image-link aware throughout.** Image links carry through generation, parsing, and revision; the aggregator actively protects them from being edited or stripped.

---

## Dependencies

- **LLMs:** Gemini 3 Flash by default for generation, internal reviewer/reviser, parsing, checklist reviewers, aggregator, and reviser. Configurable per step via `services/llm_service.py`.
- **Web search (reviser only):** Gemini with Google Search grounding, used as the reviser's `Search` tool.
- **Concurrency:** `ThreadPoolExecutor` (max 5 workers) at the topic, row, and block levels. Rate-limited LLM calls via `InMemoryRateLimiter` (1 req/s, bucket 10).
- **Activity tracking:** LangSmith.
- **Env vars:** `GOOGLE_API_KEY`, `LANGCHAIN_API_KEY`, `LANGCHAIN_PROJECT` (plus the standard LLM keys for failover providers).

---

## Source

| File | Role |
| :--- | :--- |
| `slide_chunks.py` | Streamlit page — pipeline DAG and UI orchestration |
| `agents/slide_chunks/generate_slide_chunks_v2.py` | Per-Topic slide-chunk generator + internal 3-iteration reviewer/reviser |
| `agents/slide_chunks/slide_chunks_parsing.py` | LLM-driven parsing of the `<slides>` blob into structured rows on the `Slide Chunks` tab |
| `agents/slide_chunks/slide_chunks_checklist_v2.py` | Parallel reviewers + aggregator + reviser with CRUD tools; topic-level parallelization |
| `agents/slide_chunks/visualize_slide_chunks_diff.py` | Pre/post-checklist diff view |
| `services/crud_text_block_tools.py` — `search_web()` | Gemini-grounded web search exposed as the reviser's `Search` tool |
