# Assessment Agent

> Stage 5 of the Auto Course Gen pipeline. Reads the parsed `Slide Chunks` tab and generates a full course assessment — multiple-choice, true/false, select-all-that-apply, and (conditionally) matching questions — one batch per Topic. Each question is reviewed and revised against a tunable checklist sourced from the AI Checklist Sheet, then the final question set is run through two pass/fail evaluation passes (aggregate-level and per-question) that surface a quality matrix the team can read back to spot weak coverage or weak questions.

---

## At a glance

| | |
| :--- | :--- |
| **Owner** | Dilip |
| **Status** | Done / stable |
| **Pipeline stage** | 5 — assessment authoring |
| **Generative?** | Yes — Gemini 3 Flash by default for generation, reviewer, reviser, and verdict evaluators |
| **Interface** | Streamlit page (sidebar → Agentic Workflows → *Assessment*) |
| **Granularity** | Generation is **per Topic** (one LLM call per Topic); review-and-revise is **per question × per checklist Task** |
| **Typical runtime** | ~35–60 minutes total (Step 2 review-revise dominates ~15–20 min; Step 4 per-question verdicts ~20–40 min) |
| **Run frequency** | Once per course, with iteration via the checklist tabs |
| **Primary user** | Content team |
| **External APIs** | Google Gemini, Google Sheets / Drive |
| **Requirements** | R4 (Workflow Improvement), R8 (Content & Curriculum) |

---

## What it does

Once the Slide Chunks agent has finalized the `Slide Chunks` tab:

1. For each Topic, the agent feeds **all slide chunks under that Topic** into one LLM call and generates a mixed-type question set sized to the content depth.
2. Each question is then run through a per-question, per-Task review-and-revise loop that reads the **`Review Agent Checklist`** tab from the AI Checklist Sheet. The loop applies up to 2 review→revise iterations per Task and writes the final, schema-clean question set to a `Final Assessment` worksheet.
3. The full assessment is evaluated holistically against the **`Assessment Checklist`** tab — a coarse "did the whole assessment hit this?" pass that writes Yes/No verdicts in an `LLM Based Output` column on that tab.
4. Each individual question is then evaluated against the same Review Agent Checklist criteria — a per-question matrix that lets the team see exactly which questions miss which criteria.

The `Final Assessment` tab is the deliverable; the two checklist tabs (with verdicts written in) are the QA artifacts the team uses to decide whether the course is ready to ship.

---

## The user's tuning surface — two checklist tabs

Unlike the other agents, the Assessment agent reads from **two separate tabs** on the AI Checklist Sheet (URL stored in Course info → `Checklist Link`). Both are populated with Tasks + Review Criteria at template setup and edited by the content team to steer the agent per course.

| Tab | Used by | Role |
| :--- | :--- | :--- |
| **`Review Agent Checklist`** | Step 2 (Review and Revise) and Step 4 (per-question verdicts) | The drive-and-evaluate checklist. Tasks here actively shape question revisions in Step 2 and produce per-question Yes/No matrices in Step 4. |
| **`Assessment Checklist`** | Step 3 (aggregate verdicts) | A holistic checklist run over the whole assessment set after revisions, producing a single `LLM Based Output` Yes/No per criterion. Use it for "course-wide" coverage criteria. |

Both tabs share the same schema:

| Column | Purpose |
| :--- | :--- |
| **Task** | A named review dimension (e.g., a quality theme). In Step 2, all criteria sharing a Task are evaluated together by the same reviewer call. |
| **Review Criteria** | The rubric statement itself — what the LLM is checking for. |
| *(written by the agent)* `LLM Based Output` *(Assessment Checklist)* | Yes / No verdict for the whole assessment against this row's criterion. Written by Step 3. |
| *(written by the agent)* `Question 1`, `Question 2`, …  *(Review Agent Checklist)* | One dynamic column per question, each carrying a Yes / No verdict for that question against this row's criterion. Written by Step 4. |

**Editing a row in these tabs changes how the agent behaves for that course** — for the Review Agent Checklist it changes what questions get revised in Step 2 and how Step 4 grades them; for the Assessment Checklist it changes the rubric for the aggregate Step 3 pass.

---

## Block diagram

```
   Slide Chunks tab (Topic / Subtopic / Slide Type / Title / Slide Chunk)
                              │
                              ▼
   ┌──────────────────────────────────────────────────────────┐
   │ 1. Generate Assessment Questions                         │
   │  • Group Slide Chunks by Topic                           │
   │  • One LLM call per Topic with all slide chunks for it   │
   │    concatenated as one input                             │
   │  • Generates Multiple Choice (knowledge + scenario),     │
   │    True/False, Select All That Apply, and (if topic      │
   │    has system components / categories) Matching          │
   │  • Quantity rules: depth-driven; Select All = 1 if <9    │
   │    questions total else 2+; Scenario MC = 20-60% of MCs  │
   │  • 5 parallel workers across topics                      │
   │  → hidden "Assessment questions" worksheet               │
   │    (columns: topic, questions[XML blob])                 │
   └──────────────────────────────────────────────────────────┘
                              │
                              ▼
   ┌──────────────────────────────────────────────────────────┐
   │ 2. Review and Revise all Assessment Questions            │
   │  • Read AI Checklist Sheet → `Review Agent Checklist`    │
   │  • Group rows by Task                                    │
   │  • For each (topic, question):                           │
   │      for each Task:                                      │
   │          up to 2 iterations:                             │
   │            reviewer LLM evaluates Q against all of       │
   │              that Task's criteria; emits <evaluation>    │
   │              blocks with PASS/FAIL + feedback            │
   │            if any FAIL → reviser LLM rewrites Q          │
   │              (cannot change question type)               │
   │  • Parse each revised Q to typed Pydantic models         │
   │    (MultiChoice / TrueFalse / SelectAll / Matching)      │
   │  • 5 parallel workers across (topic, question) pairs     │
   │  → "Final Assessment" worksheet                          │
   │    (#, Course Name, Topic, Question type, Question,      │
   │     Option A/B/C/D, Correct Answer,                      │
   │     Correct feedback, Incorrect feedback)                │
   └──────────────────────────────────────────────────────────┘
                              │
                              ▼
   ┌──────────────────────────────────────────────────────────┐
   │ 3. Update Assessment Checklist (~1 min)                  │
   │  • Read AI Checklist Sheet → `Assessment Checklist`      │
   │  • For each Task, one LLM call with all Slide Chunks +   │
   │    the entire Final Assessment as context                │
   │  • Emits a Yes/No verdict per criterion in that task     │
   │  • 5 parallel workers across tasks                       │
   │  → write `LLM Based Output` column on the                │
   │    `Assessment Checklist` tab                            │
   └──────────────────────────────────────────────────────────┘
                              │
                              ▼
   ┌──────────────────────────────────────────────────────────┐
   │ 4. Update Review Agent Checklist (~20-40 min)            │
   │  • Read AI Checklist Sheet → `Review Agent Checklist`    │
   │  • Per Task, per Question:                               │
   │      LLM call with that one question + Slide Chunks      │
   │      → Yes/No verdict for each criterion in that task    │
   │  • 5 parallel workers across questions within each Task  │
   │  • Tasks looped sequentially in the outer loop           │
   │  → write dynamic columns `Question 1`, `Question 2`, …   │
   │    on the `Review Agent Checklist` tab; existing rows    │
   │    preserved (only cell verdicts overwritten)            │
   └──────────────────────────────────────────────────────────┘
                              │
                              ▼
              Final Assessment tab + populated checklist tabs
              (course's assessment ready for QA / publishing)
```

---

## Inputs

| Source | Field / Tab | Role |
| :--- | :--- | :--- |
| Course Sheet → `Slide Chunks` | `Topic`, `Subtopic`, `Slide Chunk Title`, `Slide Chunk` | The source content for question generation and the grounding evidence the reviewers/evaluators check against. |
| Course Sheet → `Course info` | Course Name, Target Audience, **Checklist Link** | LLM prompt context + the pointer to the AI Checklist Sheet. |
| AI Checklist Sheet → **`Review Agent Checklist`** tab | Tasks + Review Criteria | Drives Step 2 review-and-revise; populated with per-question Yes/No verdicts by Step 4. |
| AI Checklist Sheet → **`Assessment Checklist`** tab | Tasks + Review Criteria | Drives Step 3 aggregate verdicts; populated with `LLM Based Output` by Step 3. |
| Course Sheet → `Assessment questions` (hidden) | `topic`, `questions` (XML blob) | Internal handoff from Step 1 to Step 2. |
| Course Sheet → `Final Assessment` | structured rows | Read by Steps 3 and 4 as the corpus to evaluate. |

---

## Outputs

| Tab / Artifact | Description |
| :--- | :--- |
| `Assessment questions` (hidden worksheet) | Step 1 output. Two columns: `topic`, `questions`. The `questions` cell holds the raw XML blob the generator emitted for that Topic; consumed by Step 2 and then ignored. |
| **`Final Assessment` worksheet** | Step 2 output and the main deliverable. One row per question. Columns: `#`, `Course Name`, `Topic`, `Question type`, `Question`, `Option A`, `Option B`, `Option C`, `Option D`, `Correct Answer`, `Correct feedback`, `Incorrect feedback`. Question type ∈ `multichoice` / `truefalse` / `select_all` / `matching`. For Matching questions, the options columns hold the match pairs and `Correct Answer` is the canonical mapping. |
| `Assessment Checklist` tab (AI Checklist Sheet) → `LLM Based Output` column | Step 3 output. One Yes/No verdict per row, judging the entire assessment against that criterion. |
| `Review Agent Checklist` tab (AI Checklist Sheet) → dynamic `Question 1`, `Question 2`, … columns | Step 4 output. A 2D matrix of Yes/No verdicts: rows are (Task × Criterion); columns are individual questions. Lets the team see exactly which question failed which criterion. |

---

## Functional description

The pipeline runs as a DAG of named steps; the user runs each step from the Streamlit page.

**1. Generate Assessment Questions** *(automatic, per Topic, 5 parallel workers)*

- Reads the `Slide Chunks` tab and groups by `Topic`.
- For each Topic, concatenates every slide chunk under it (`Slide Chunk Title: …` + content) into one long input string.
- Calls the generator LLM with a prompt that enforces:
  - Mandatory question types: Multiple Choice (knowledge + scenario), True/False, Select All That Apply.
  - Conditional: Matching questions only when the topic contains system components, categories, or closely related items.
  - **Quantity rules**: total question count scaled to content depth; Select All = exactly 1 if total <9 else 2+; Scenario MC = 20–60 % of MCs depending on practical level.
  - **Quality rules**: scenario MCs in second person ("You are…", "You notice…"); Select All has 2–3 correct options of 4; feedback never references option letters (uses the actual content text); Matching has unique strings with no letter labels.
  - XML output format: each question wrapped in `<question>…</question>` with `Question no`, `Question Type`, `Question`, options, `Correct Answer`, `Correct feedback`, `Incorrect feedback`.
- Writes the per-Topic XML blob to a hidden `Assessment questions` worksheet (`topic`, `questions` columns).

**2. Review and Revise all Assessment Questions** *(automatic, iterative, 5 parallel workers)*

- Reads `Assessment questions`, `Slide Chunks` (for grounding context), and the `Review Agent Checklist` tab from the AI Checklist Sheet. Groups checklist rows by `Task`.
- For each `(topic, question)` pair, runs a **nested sequential review-and-revise loop**:
  - For each Task in the checklist:
    - Up to 2 iterations of:
      - **Reviewer LLM** evaluates the question against all of that Task's criteria. Output is XML `<evaluation>` blocks with `<item_name>`, `<analysis>`, `<verdict>` (`PASS` / `FAIL`), `<feedback_summary>`, `<improvement_suggestions>`.
      - If any criterion in that Task fails, a **reviser LLM** rewrites the question, taking the reviewer feedback as input. The reviser is explicitly **not allowed to change the question type**.
    - Loop exits early once every criterion in that Task passes.
- After all questions have been through every Task, each revised question is parsed into a typed Pydantic model (`MultiChoiceQuestion`, `TrueFalseQuestion`, `SelectAllQuestion`, `MatchingQuestion`) and serialized into the flat `Final Assessment` schema.
- This is the longest single-LLM-call step in the pipeline (~15–20 min) because the loop is sequential per question over all Tasks.

**3. Update Assessment Checklist** *(automatic, aggregate, ~1 min, 5 parallel workers across tasks)*

- Reads the `Assessment Checklist` tab. For each Task, makes a single LLM call with the entire `Slide Chunks` corpus and the entire `Final Assessment` as evidence.
- LLM emits, per criterion in that Task, a structured response: `Scratchpad` (reasoning), `Verdict` (Yes/No), `Why no` (only if No).
- Writes each verdict into the `LLM Based Output` column on the same row of the checklist tab. This is the *aggregate* QA pass — "does the whole assessment as a whole satisfy this?"

**4. Update Review Agent Checklist** *(automatic, per-question, ~20–40 min)*

- Reads the `Review Agent Checklist` tab. For each Task (outer loop, sequential), spawns 5 parallel workers over questions in `Final Assessment`. Each worker makes one LLM call per (Task, Question).
- LLM emits a Yes/No verdict per criterion against that single question, grounded against the full Slide Chunks corpus.
- Verdicts are parsed and written to dynamic per-question columns (`Question 1`, `Question 2`, …). The `_preserve` part of the function name means the checklist tab's row structure and any prior verdicts are kept intact — only the targeted cells are overwritten — so re-runs don't blow away previous columns.
- This step's runtime is dominated by the per-question count: ~N_tasks × N_questions calls (typically 5+ tasks × 20–30 questions = 100–150 LLM calls).

---

## Operating procedure

1. Confirm the Slide Chunks agent has populated the `Slide Chunks` tab.
2. Open the AI Checklist Sheet's `Review Agent Checklist` and `Assessment Checklist` tabs and tune the Tasks / Review Criteria as needed for this course. The Review Agent Checklist drives revisions, so changes there are higher-impact.
3. Open *Assessment* in the Streamlit sidebar and load the Course Sheet.
4. Run the four steps in order:
   - **Generate Assessment Questions** — emits a per-Topic XML blob to a hidden worksheet.
   - **Review and Revise all Assessment Questions** — runs the per-question / per-Task loop and writes the typed `Final Assessment` rows.
   - **Update Assessment Checklist** — populates `LLM Based Output` for an aggregate QA read.
   - **Update Review Agent Checklist** — populates per-question columns for a granular QA read.
5. Read the two checklist tabs to spot weak questions or weak coverage. Iterate by editing the criteria and re-running the relevant steps.

---

## Role in the pipeline

| Stage | Agent | Reads from | Writes to |
| :--- | :--- | :--- | :--- |
| 0 | Template Sheet Setup | — | Course Sheet, Checklist Sheet |
| 1 | Course Outline | Course Sheet, Checklist Sheet | Research tabs, vectorstore, `Final Outline` (incl. `References`) |
| 2 | Research Notes | `Final Outline` + upstream worksheets + Checklist Sheet | `Final Outline → research_notes` |
| 3 | Slide Chunks | `Final Outline → research_notes` + Checklist Sheet | `Slide Chunks` tab |
| 4 | Graphics | `Slide Chunks` + Checklist Sheet | `Slide Chunks → final_graphics_definition` |
| 5 | **Assessment** *(this agent)* | `Slide Chunks` + `Review Agent Checklist` + `Assessment Checklist` | `Assessment questions` (hidden), **`Final Assessment` tab**, `LLM Based Output` on `Assessment Checklist`, dynamic `Question N` columns on `Review Agent Checklist` |

---

## Impact

- **End-to-end assessment in one run.** From slide content to a typed, schema-clean question bank — including revision against custom criteria and per-question quality verdicts — without anyone hand-writing questions.
- **Two-checklist tuning surface.** The Review Agent Checklist tab is where the team decides what "good" looks like at the question level; the Assessment Checklist tab is where they encode course-wide coverage rules. Both are editable per course.
- **Schema-clean output.** Pydantic typing on the final write means downstream systems get question type, options, correct answer, and feedback as discrete fields — not free-form text that needs re-parsing.
- **Built-in QA matrix.** Step 4's per-question verdict matrix lets the content team scan a single tab and see exactly which questions failed which criteria — a much faster QA loop than reading each question individually.

---

## Dependencies

- **LLMs:** Gemini 3 Flash by default for the generator, reviewer, reviser, aggregate evaluator (Step 3), and per-question evaluator (Step 4). Configurable via `services/llm_service.py`.
- **Concurrency:** `ThreadPoolExecutor` (max 5 workers) at every parallel boundary — topics in Step 1, (topic, question) pairs in Step 2, tasks in Step 3, questions per task in Step 4.
- **Structured output:** Pydantic models (`MultiChoiceQuestion`, `TrueFalseQuestion`, `SelectAllQuestion`, `MatchingQuestion`) to enforce schema on Step 2 output.
- **Activity tracking:** LangSmith.
- **Env vars:** `GOOGLE_API_KEY`, `LANGCHAIN_API_KEY`, `LANGCHAIN_PROJECT` (plus standard LLM keys for failover providers).

---

## Source

| File | Role |
| :--- | :--- |
| `assessment.py` | Streamlit page — pipeline DAG and UI orchestration |
| `agents/generate_assessments/generate_assessment_questions.py` | Step 1: per-Topic XML generator + hidden `Assessment questions` worksheet |
| `agents/generate_assessments/run_review_and_reviser.py` | Step 2: nested per-question / per-Task review-and-revise loop + Pydantic serialization to `Final Assessment` |
| `agents/generate_assessments/review_assessment.py` | Reviewer prompt for Step 2 |
| `agents/generate_assessments/revise_assessment.py` | Reviser prompt for Step 2 |
| `agents/generate_assessments/generate_assessment_checklist.py` | Step 3: aggregate-level Yes/No verdicts against `Assessment Checklist` |
| `agents/generate_assessments/generate_review_agent_checklist.py` | Step 4: per-question Yes/No verdict matrix against `Review Agent Checklist`, with row-preservation semantics |
