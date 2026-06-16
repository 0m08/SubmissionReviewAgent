# Research Notes Agent

> Stage 2 of the Auto Course Gen pipeline. Turns each Learning Objective row in the `Final Outline` tab into a self-contained block of source-backed research notes. The content team picks the references each LO should be grounded in — either by leaning on what the Course Outline agent retrieved, by hand-editing the row's reference list, or by feeding their own URLs through the `External References` column on Course Info — and this agent loads each reference's actual content, generates the notes (paraphrased for documents, preserved verbatim for video transcripts), and runs a checklist-driven multi-reviewer loop the team can tune per course.

---

## At a glance

| | |
| :--- | :--- |
| **Owner** | Dilip |
| **Status** | Done / stable |
| **Pipeline stage** | 2 — research notes generation |
| **Generative?** | Yes — Gemini 3 Flash by default for generation, internal reviewer/reviser, parallel checklist reviewers, aggregator, and reviser |
| **Interface** | Streamlit page (sidebar → Agentic Workflows → *Research Notes*) |
| **Granularity** | One row of `Final Outline` (one Learning Objective) → one block of `research_notes` |
| **Typical runtime** | ~15–25 minutes total (checklist loop dominates) |
| **Run frequency** | Once per course, with human-in-the-loop iteration |
| **Primary user** | Content team |
| **External APIs** | Google Gemini, Gemini Google-Search grounding (reviser's web lookups), YouTube transcript API, web article fetch, Google Sheets / Drive |
| **Requirements** | R4 (Workflow Improvement), R8 (Content & Curriculum) |

---

## The two reference-control surfaces (key concept)

The content team has **two complementary ways** to specify which references should ground a given research-notes block. Both end up driving this agent.

### 1. Per-row reference control: `References`, `Reference type`, `Reference usage` on `Final Outline`

Each row on the `Final Outline` tab carries these three columns. The Course Outline agent's retriever populates them automatically by pulling from the per-course vectorstore — **but the content team can edit them by hand** before running this agent, to override what the agent retrieved or to point the LLM at a specific source.

| Column | Format | Effect |
| :--- | :--- | :--- |
| **References** | Newline-separated URLs (one URL per line) | The actual references for this LO. If left blank, the Researcher falls back to a fresh RAG retrieval over the course vectorstore + web search. |
| **Reference type** | `Web Article`, `Youtube Video`, or `Google Drive Video` | Tells the loader which fetching strategy to use. |
| **Reference usage** | `Content` or `Video` | Tells the Researcher which generation strategy to use. `Content` → paraphrase the source into narrative notes. `Video` → preserve the underlying transcript verbatim, with timestamps. |

The triple `(Reference type, Reference usage)` determines which branch this agent takes — see *Reference processing matrix* below.

### 2. Course-level user URLs: `External References` on `Course info`

A Course-Info-level field where the user pastes a newline-separated list of URLs they want available to the pipeline for *any* LO. These URLs flow in upstream — the Course Outline agent's reference loader picks them up with `source_origin='External References'` (typed as `Youtube Video` or `Web Article` from the URL pattern), embeds them into the per-course vectorstore alongside everything else, and the Course Outline agent's retriever then surfaces them on the `References` column of whichever LO rows they're most relevant to.

So **External References** isn't read directly by this agent — it shows up here only via the `References` column that the upstream retriever populated. It's documented here because it's the most common way a user adds their own curated URLs to a course.

---

## Reference processing matrix

The agent's first step branches per row on `Reference type` × `Reference usage`:

| Reference type | Reference usage | What Step 1 (context gen) does | What Step 2 (Researcher) does |
| :--- | :--- | :--- | :--- |
| `Web Article` | `Content` | Fetches each URL's article body (via `get_docs_from_url`), writes content into `context_0`, `context_1`, … (49 KB per cell) | LLM **paraphrases** into narrative `research_notes` with internal 3-iteration reviewer/reviser quality gate |
| `Youtube Video` | `Content` | Fetches the transcript, formats it (without the video ID in the header), writes into `context_n` | LLM **paraphrases** into narrative `research_notes` |
| `Youtube Video` | `Video` | Fetches the transcript, includes the video ID + start/end timestamps in the header, writes into `context_n` | LLM **preserves the transcript verbatim** and emits chunked `Video_id` / `Start` / `End` / Transcript blocks. Post-processing wraps them into YouTube embed links (`https://www.youtube.com/embed/{id}?start=X&end=Y`) separated by `---`. |
| `Google Drive Video` | `Video` | Looks up the file ID in the `MAIN GRIT VIDEOS` sheet, pulls pre-computed transcript columns | Same verbatim-transcript path as YouTube Video. |
| *(empty)* | *(empty)* | Falls back to RAG: invokes the course vectorstore via `ContextualCompressionRetriever` (Chroma + BM25 + Cohere rerank); if recall is thin, an LLM-driven retriever can switch to **Exa web search** (max 5 turns). | Standard paraphrase path. |

---

## Block diagram

```
   Final Outline rows
   (Topic / Subtopic / LO / References / Reference type / Reference usage)
                              │
                              ▼
   ┌──────────────────────────────────────────────────────────┐
   │ 1. Generate Context from Provided References             │
   │  • For each row with a non-empty `References`:           │
   │      branch on Reference type:                           │
   │        - Web Article → fetch article body                │
   │        - Youtube Video → fetch transcript (filtered to   │
   │          ?start= / ?end= range if present in URL)        │
   │        - Google Drive Video → look up transcript in      │
   │          MAIN GRIT VIDEOS sheet                          │
   │  • For rows with empty `References`: fall back to        │
   │      RAG over the per-course vectorstore + Exa web       │
   │      search (LLM-driven retriever, max 5 turns)          │
   │  • Chunk content into `context_0`, `context_1`, …        │
   │    (49 KB per cell)                                      │
   │  • Also writes `source_links`, `as_is_sources`,          │
   │    `content_sources`, `web_links`, `video_links`         │
   │  • Backs up previous state to "Context Column Backup"    │
   │    hidden worksheet                                      │
   └──────────────────────────────────────────────────────────┘
                              │
                              ▼
   ┌──────────────────────────────────────────────────────────┐
   │ 2. Researcher (LLM)                                      │
   │  • For each row, read the `context_n` columns            │
   │    (joined into one context string)                      │
   │  • Branch on (Reference type, Reference usage):          │
   │      - Video usage → generate_transcript_chunks          │
   │        (preserve transcript verbatim; emit               │
   │        Video_id/Start/End/Transcript blocks)             │
   │      - else → generate_research_notes                    │
   │        (paraphrase; internal 3-iteration reviewer/       │
   │         reviser quality gate; preserves image links;     │
   │         50 K-char cap with auto-summarize fallback)      │
   │  • Video-usage rows: post-process into YouTube embed     │
   │    links + transcript code blocks                        │
   │  → writes `research_notes` column                        │
   └──────────────────────────────────────────────────────────┘
                              │
                              ▼
                  Manual review of research notes
                              │
                              ▼
   ┌──────────────────────────────────────────────────────────┐
   │ 3. Checklist-driven review & revise (v2)                 │
   │  • Back up Final Outline to hidden worksheet             │
   │  • Parallel reviewers per Task × Scope slice             │
   │  • Aggregator LLM merges feedback (drops feedback        │
   │    targeting image links to keep them safe)              │
   │  • Single reviser with CRUD tools                        │
   │    (Create/Read/Update/Delete/StrReplace/Search/         │
   │    PreviewImage/Stop)                                    │
   │  • Re-review with blockwise diffs until criteria pass    │
   │    or `Iteration Count` is hit                           │
   │  • restore_dropped_image_links() re-appends any          │
   │    `![](url)` the reviser dropped                        │
   └──────────────────────────────────────────────────────────┘
                              │
                              ▼
   ┌──────────────────────────────────────────────────────────┐
   │ 4. Inline Image Placement                                │
   │  • LangGraph agent reads each row's research_notes       │
   │  • For each image URL, calls preview_image(url) to see   │
   │    the actual image, then uses str_replace to move the   │
   │    `![](url)` link next to the paragraph it illustrates  │
   │  • URL preserved byte-for-byte; alt text may be improved │
   │  • If image-link set changed (URLs added or removed),    │
   │    reverts that row                                      │
   │  • Status: `inline_image_placement_status = "done"`      │
   │  • Skips video-usage rows                                │
   └──────────────────────────────────────────────────────────┘
                              │
                              ▼
   ┌──────────────────────────────────────────────────────────┐
   │ 5. Visualize Research Notes Diff (manual)                │
   │  • Pre vs post-checklist diff per row,                   │
   │    red/green highlighting                                │
   │  • User confirms                                         │
   └──────────────────────────────────────────────────────────┘
                              │
                              ▼
              `Final Outline → research_notes` column
                  (handoff to Slide Chunks agent)
```

---

## Inputs

| Source | Field / Tab | Role |
| :--- | :--- | :--- |
| Course Sheet → `Final Outline` | Topic, Subtopic, Learning Objectives | Per-row generation targets (one LO per row). |
| Course Sheet → `Final Outline` | **`References`** | Newline-separated URLs for this LO. Populated by the Course Outline agent's retriever; the content team can hand-edit before running this agent. |
| Course Sheet → `Final Outline` | **`Reference type`** | `Web Article` / `Youtube Video` / `Google Drive Video`. Picks the loader. |
| Course Sheet → `Final Outline` | **`Reference usage`** | `Content` (paraphrase) / `Video` (preserve verbatim). Picks the Researcher branch. |
| Course Sheet → `Course info` | **`External References`** | User-supplied newline-separated URLs. *Consumed upstream* by the Course Outline agent and routed into the vectorstore + `References` column — not read directly here. |
| Course Sheet → upstream source worksheets | `MAIN GRIT VIDEOS` (for Google Drive Videos), preliminary research / videos research (for fallback path) | Where Step 1 looks up reference content. |
| Per-course Chroma vectorstore (on Drive) | — | Built by the Course Outline agent. Used by Step 1 only as the **fallback retriever** when a row's `References` is empty. |
| Course Sheet → `Course info` | Course Name, Course Background, Course Objective Guidelines, Target Audience & Industry | Threaded into every LLM prompt as context. |
| AI Checklist Sheet → `Research Notes Checklist` tab | Tasks + scope + criteria + examples + corrective operations + (optional) Iteration Count + Tools | Drives the post-generation checklist loop. |

---

## Outputs

| Tab / Artifact | Description |
| :--- | :--- |
| `Final Outline` → `context_0`, `context_1`, … columns | The reference content loaded for each row, chunked into 49 KB cells. This is what the Researcher LLM reads to write the notes. |
| `Final Outline` → `source_links`, `as_is_sources`, `content_sources`, `web_links`, `video_links` columns | Per-row metadata produced by Step 1: numbered source links (`[1] URL`), HVAC-School-flagged "as-is" sources, content (paraphrase-able) sources, and YouTube embed links with start/end timestamps. |
| `Context Column Backup` (hidden worksheet) | Pre-overwrite backup of `context_n` for every row. Restored if Step 1 is deleted/re-run. |
| **`Final Outline` → `research_notes` column** | The main deliverable. Either narrative markdown (paraphrase path) or YouTube-embed-link + verbatim-transcript blocks (video-usage path). With preserved inline image links. Consumed directly by the Slide Chunks agent. |
| `Final Outline` → `inline_image_placement_status` column | `"done"` once Step 4 has repositioned the row's images. Used to resume safely if Step 4 is re-run. |
| `Backup Final Outline Sheet for Delete step of Research Notes Checklist` (hidden worksheet) | Pre-checklist snapshot of `Final Outline`. Used by the diff view and to safely restore state if the checklist step is deleted/re-run. |

---

## Functional description

The pipeline runs as a DAG of named steps; the user runs each step from the Streamlit page.

**1. Generate Context from Provided References** *(automatic, 5 parallel workers)*

- Reads each row's `References`, `Reference type`, and `Reference usage` columns from `Final Outline`.
- Backs up the entire current `context_n` state to a hidden `Context Column Backup` worksheet.
- For each row with non-empty `References`, fans out one task per row that processes every URL in the list:
  - **Web Article + Content** — fetches the article body via `get_docs_from_url`. Articles are prefixed `"Article {n}:"` and joined with `---`.
  - **Youtube Video + Content / Video** — fetches the transcript via the YouTube transcript API (with fallback chain), respects `?start=` / `?end=` URL params to filter the transcript window, formats as `'(timestamp)': (text)` lines. Includes the video ID in the header only for `Video` usage.
  - **Google Drive Video + Video** — extracts the file ID, looks it up in the `MAIN GRIT VIDEOS` sheet, pulls the pre-computed `Transcript *` columns.
- For each row with empty `References`, falls back to the **course vectorstore retriever** (Chroma + BM25 + Cohere rerank) with an LLM-driven multi-turn loop (max 5 turns) that can switch to **Exa web search** if vectorstore recall is thin.
- Writes the assembled content into `context_0`, `context_1`, … (49 KB per cell) on the same row, plus the per-row metadata columns (`source_links`, etc.).

**2. Researcher** *(automatic, 5 parallel workers)*

- For each row with a non-empty Subtopic & LO, joins the `context_n` columns into a single context string.
- Branches on `Reference type` × `Reference usage`:
  - **`Video` usage** → `generate_transcript_chunks`. Prompt instructs: "Do not paraphrase, rephrase, or summarize the transcripts in the output." Returns `<extract_relevant_chunks>` blocks. Post-processing extracts each `Video_id` / `Start` / `End` triple and wraps them into `https://www.youtube.com/embed/{id}?start=X&end=Y` headers plus the verbatim transcript as code blocks, separated by `---`.
  - **Otherwise** → `generate_research_notes`. Prompt enforces SkillCat voice rules: plain language, active voice, no corporate jargon ("optimal", "leverage", "utilize", "facilitate"), short sentences, technician tone, trade terminology explained when first used. **All facts must be preserved; nothing invented.** All markdown image links (`![alt](url)`) must be carried verbatim. Output wrapped as `<output><analysis>...</analysis><research_notes>...</research_notes></output>`. The body of `<research_notes>` is what's saved.
  - Within `generate_research_notes`, an **internal 3-iteration reviewer/reviser quality gate** runs before returning — catches mechanical failures (missing LOs, invented data, redundancy) and self-corrects via a tool-using reviser. Exits early if the reviewer reports no failures.
  - **50 K-character cap**: if output exceeds 50,000 chars, a follow-up LLM call summarizes; ultimate fallback truncates at 49,990 chars.
- Writes the result to the `research_notes` column. Saves incrementally every 5 completed rows.

**3. Manually Review the Research Notes** *(manual)*

- User opens the `Final Outline` tab in Google Sheets and edits the `research_notes` column directly (delete, refine, add).
- Confirm in the UI to advance.

**4. Checklist Based Review and Revise Agents** *(automatic, iterative)*

- The agent's main quality lever (see *Checklist* section below).
- Reads the `Research Notes Checklist` tab; runs parallel reviewers → aggregator → reviser; loops until criteria pass or `Iteration Count` is hit.
- Backs up `Final Outline` to a hidden worksheet so the diff view and a clean re-run are both safe.

**5. Inline Image Placement** *(automatic, 10 parallel workers)*

- A LangGraph agent (Gemini 3 Flash with thinking enabled) reads each row's `research_notes`.
- For each image URL in the cell, it calls `preview_image(url)` to actually see what the image shows, then uses `str_replace` to move the `![alt](url)` link next to the paragraph that talks about that thing.
- The URL must be preserved byte-for-byte; the alt text can be rewritten to better describe what was seen.
- Safety check: if the set of image URLs in the post-edit `research_notes` doesn't match the set in the input (i.e. a URL was added or dropped), the row's edits are reverted.
- Marks `inline_image_placement_status = "done"` to allow safe resumption.
- Skips rows on the `Video` usage path (no images expected in raw transcript output).

**6. Visualize Research Notes Diff** *(manual)*

- Concatenates each row of the live `Final Outline` and the pre-checklist backup as `Topic / Subtopic / Learning Objective / Research Notes` strings and shows a side-by-side red/green diff.
- The user reviews and clicks Confirm; nothing is mutated by the step itself.

---

## The Research Notes Checklist — the user's tuning lever

The agent reads its review-and-revise rubric from the **`Research Notes Checklist`** tab inside the AI Checklist Sheet (URL stored in Course info → `Checklist Link`). **Editing this tab is how the content team changes the agent's output style and quality bar per course without code changes.**

The checklist has one row per criterion:

| Column | Purpose |
| :--- | :--- |
| **Task** | A named review dimension. Criteria sharing a Task are reviewed together by the same reviewer agent. |
| **Scope** | The slice each reviewer sees per pass: `Global (full output)`, `Topic`, `Subtopic`, or `Learning Objective`. Smaller scopes mean more focused but more numerous review passes. |
| **Criteria Name** | Short identifier (used in feedback and audit trails). |
| **Review Criteria** | The rubric statement itself. |
| **Review Agent Examples** | Worked passing / failing examples shown to the reviewer LLM. |
| **Corrective Operations** | Instructions the reviser LLM follows when this criterion fails. |
| **Reviser Agent Examples** | Worked examples for the reviser LLM, paired with the corrective operations. |
| **Iteration Count** *(optional)* | Per-task maximum number of review-revise rounds; the agent uses the max across all rows as its global cap (default 1). |
| **Review Agent Tools** *(optional)* | Per-task override of which reviser tools are available. |

Behaviour:

- **Parallel review.** All `(Task, Scope)` groups run in parallel. Within each group, scope slices (e.g., one per Topic) also fan out in parallel.
- **Aggregation.** An aggregator LLM merges reviewer outputs into one consolidated feedback document; short-circuits if everything passes. Reviewer feedback that targets image links is dropped here, so the reviser never tries to edit them.
- **Single revision pass.** A reviser agent armed with block-level CRUD tools applies all corrective operations in one pass:
  - `Create`, `Read`, `Update`, `Delete` — block-level edits keyed by `Block ID`.
  - `StrReplace` — find-and-replace within a block.
  - `Search` — **Gemini Google-Search grounding** (web search, not RAG over the vectorstore). Used when a checklist failure requires a fact lookup.
  - `PreviewImage` — load and visually inspect an image URL.
  - `Stop` — required end marker.
- **Re-review with diffs.** On iterations after the first, reviewers see blockwise diffs of what changed (not the full prior context) so they can focus on whether the changes actually resolved the failure.
- **Image-link safety net.** After each revision, `restore_dropped_image_links()` compares the old and new block image-link sets; any `![](url)` the reviser silently dropped is re-appended at the end of the block.

In short: **changing a row in the Research Notes Checklist tab changes how the agent writes the notes for that course.** Loosening or tightening a criterion, swapping examples, or rewriting the corrective operation are all valid ways to steer output without touching code.

---

## Operating procedure

1. Confirm the Course Outline agent has populated the `References`, `Reference type`, and `Reference usage` columns on `Final Outline`. Hand-edit any rows where you want to override the retrieved references — paste your own URLs, change the `Reference usage` to `Video` if you want a video kept verbatim, etc.
2. *(Optional)* Add per-course curated URLs to the `External References` field on Course Info before the Course Outline agent runs (they get pulled into the vectorstore upstream).
3. Open *Research Notes* in the Streamlit sidebar and load the Course Sheet.
4. *(Optional)* Open the AI Checklist Sheet's `Research Notes Checklist` tab and tune any rows that need to behave differently for this course.
5. Run the steps in order:
   - **Generate Context from Provided References** — loads each row's reference content into `context_0..N`.
   - **Researcher** — reads `context_n` and writes `research_notes`.
   - **Manually Review the Research Notes** — edit `research_notes` directly in the Sheet; confirm.
   - **Checklist Based Review and Revise Agents** — iterative auto-loop; progress surfaces in the UI.
   - **Inline Image Placement** — auto.
   - **Visualize Research Notes Diff** — eyeball the pre/post diff and confirm.
6. Handoff: the `research_notes` column on `Final Outline` is the contract for the Slide Chunks agent.

---

## Role in the pipeline

| Stage | Agent | Reads from | Writes to |
| :--- | :--- | :--- | :--- |
| 0 | Template Sheet Setup | — | Course Sheet, Checklist Sheet |
| 1 | Course Outline | Course Sheet, Checklist Sheet | Research tabs, per-course vectorstore, `Final Outline` (incl. `References` / `Reference type` / `Reference usage`) |
| 2 | **Research Notes** *(this agent)* | `Final Outline` (`References` / `Reference type` / `Reference usage`), upstream source worksheets, per-course vectorstore (fallback), `Research Notes Checklist` | `Final Outline` (`context_0..N`, `source_links`, `as_is_sources`, `content_sources`, `web_links`, `video_links`, `research_notes`, `inline_image_placement_status`) |
| 3 | Slide Chunks | `Final Outline` (`research_notes`), Slide Chunks Checklist | `Slide Chunks` tab |
| 4 | Graphics | `Slide Chunks`, Checklist Sheet | `Slide Chunks → final_graphics_definition` |
| 5 | Assessment | Course Sheet, Checklist Sheet | `Assessment` tab |

---

## Impact

- **Two reference-control surfaces for the content team.** Per-row `References` / `Reference type` / `Reference usage` on `Final Outline` for surgical, LO-level control; `External References` on Course Info for course-wide curated URLs that the upstream retriever weaves into the relevant rows.
- **Source-faithful for video, narrative for documents.** The `Reference usage = Video` path preserves the underlying transcript verbatim and emits ready-to-embed YouTube links — perfect when the reference *is* the teaching content. The `Content` path paraphrases into SkillCat voice — perfect when the reference is reading material that needs rewriting.
- **Two quality gates.** An internal 3-iteration reviewer/reviser runs *inside the Researcher* before notes are ever written, and the user-tunable Checklist loop runs *after*. The internal loop catches mechanical failures (missing LOs, invented facts); the checklist loop catches style / policy issues.
- **Image links survive every transformation.** Preserved by the generation prompt, protected by the aggregator (which strips reviewer feedback targeting them), restored by the post-revision safety net, and finally repositioned by an agent that actually *sees* each image before placing it.

---

## Dependencies

- **LLMs:** Gemini 3 Flash by default for the Researcher, internal reviewer/reviser, parallel checklist reviewers, aggregator, and reviser. Gemini 2 Flash available; Groq / OpenAI / Anthropic available as failovers via `services/llm_service.py`.
- **Reranker (fallback path):** Cohere `rerank` (used only when `References` is empty and the agent falls back to the vectorstore).
- **Web search (reviser only):** Gemini with Google Search grounding, used as the reviser's `Search` tool.
- **Article fetching:** `get_docs_from_url` (web article body).
- **Transcript fetching:** YouTube transcript API with fallback chain.
- **Concurrency:** `ThreadPoolExecutor` with 5 workers (Step 1, Step 2) and 10 workers (Step 4). Rate-limited LLM calls via `InMemoryRateLimiter`.
- **Activity tracking:** LangSmith.
- **Env vars:** `GOOGLE_API_KEY`, `COHERE_API_KEY`, `LANGCHAIN_API_KEY`, `LANGCHAIN_PROJECT` (plus standard LLM keys for failover providers).

*(The Chroma vectorstore, Cohere embeddings, BM25 retriever, and Exa fallback are dependencies of the Course Outline agent — see that datasheet. They're only touched by this agent through Step 1's fallback path for rows with empty `References`.)*

---

## Source

| File | Role |
| :--- | :--- |
| `research_notes.py` | Streamlit page — pipeline DAG and UI orchestration |
| `agents/research_notes/retriever_agent.py` — `run_reference_based_context_generator_for_all_rows`, `process_single_row` | Step 1: per-row reference loading, branching on Reference type/usage; chunking into `context_n`; vectorstore + Exa fallback for empty-reference rows |
| `agents/research_notes/generate_notes.py` — `generate_research_notes`, `generate_transcript_chunks`, `run_research_notes_agent_for_all_rows` | Step 2: Researcher (paraphrase path + verbatim-transcript path), internal 3-iteration reviewer/reviser quality gate |
| `agents/research_notes/research_notes_checklist_v2.py` | Step 4: parallel reviewers + aggregator + reviser with CRUD tools; image-link safety net |
| `agents/research_notes/inline_image_placement.py` | Step 5: `preview_image`-aware image repositioning per row |
| `agents/research_notes/visualize_research_notes_diff.py` | Step 6: pre/post-checklist diff view |
| `services/crud_text_block_tools.py` — `search_web()` | Gemini-grounded web search exposed as the reviser's `Search` tool |
