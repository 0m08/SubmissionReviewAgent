# Course Outline Agent

> Stage 1 of the Auto Course Gen pipeline. The agent runs in one of two stages — **Initial** (AI helps build the outline) or **Final** (trust the user's outline, build the reference base for the Research Notes agent). In day-to-day use the **Final stage** is the workhorse: it executes whichever research sub-pipelines the user opts into (Videos / Web / Deep) plus any user-supplied external URLs, and hands a populated set of research tabs to the Research Notes agent.

---

## At a glance

| | |
| :--- | :--- |
| **Owner** | Dilip |
| **Status** | Done / stable |
| **Pipeline stage** | 1 — outline + reference collection |
| **Generative?** | Yes — multi-LLM (Gemini family, with grounding) |
| **Interface** | Streamlit page (sidebar → Agentic Workflows → *Course Outline*) |
| **Typical runtime** | Minutes to ~1 hour depending on enabled features (video transcripts dominate) |
| **Run frequency** | Once per course, with human-in-the-loop iteration |
| **Primary user** | Content team |
| **External APIs** | Google Gemini (incl. grounding), YouTube Data API, web search (Google CSE / Tavily-class), Firecrawl, Cohere (embeddings + rerank), Chroma (per-course vector DB persisted to Drive), Exa Search, Google Sheets / Drive |
| **Requirements** | R4 (Workflow Improvement), R8 (Content & Curriculum) |

---

## The two stages

The agent's behaviour is controlled by the `Outline Stage` field in the Course Info tab.

| Stage | When to use it | What the agent does | Output |
| :--- | :--- | :--- | :--- |
| **Initial** | The user has a rough, tentative outline and wants the agent to **improve it** with research-derived structure and Learning Objectives. | Runs the enabled research sub-pipelines (per `Research Sources`), consolidates partial outlines into one, optionally runs per-topic deep research + LO generation, and goes through a human review-and-revise loop. | A **more detailed outline** with references attached, written to the `Final Outline` tab. |
| **Final** *(common case)* | The user already has a detailed outline they don't want changed. The goal is to **collect useful references** for each topic so the Research Notes agent has a strong evidence base to draw from. | Runs the enabled research sub-pipelines and pulls in any user-supplied `External References`. Skips outline consolidation, review, and enhancement entirely. The Base Outline is copied straight to the Final Outline. | A **reference base** in the research tabs + a `Final Outline` tab that mirrors the Base Outline. |

> Most courses run in **Final** stage. The rest of this datasheet focuses there. *Initial* is summarized at the end.

---

## Final stage — Course Info settings that matter

| Field | Type | Effect |
| :--- | :--- | :--- |
| **Outline Stage** | `final` / `initial` | Must be `final` to use this stage. Short-circuits all outline-modifying steps; the Base Outline is trusted as-is and copied straight to the `Final Outline` tab. |
| **Research Sources** | Tokens from `{video, web, deep}`, newline- or comma-separated. Blank = skip all. | Independently toggles the three research sub-pipelines: **Videos Research**, **Web Research**, **Deep Research**. Mix freely. |
| **External References** | Newline-separated URLs | User-supplied references. Loaded into the Research Notes agent's vectorstore with `source_origin='External References'`; YouTube URLs are typed as `Youtube Video`, the rest as `Web Article`. |
| **Course Background**, **Course Objective Guidelines**, **Target Audience & Industry** | Free text | Threaded into every LLM prompt as context (search-query generation, relevance classification, deep research). |
| **Checklist Link** | URL (auto-populated by Template Setup) | Points to the AI Checklist Sheet. Not heavily used in Final stage (since the outline isn't being revised). |

> `Outline Topic Deep Research` and `Required Topic Count` are **Initial-stage only** — they have no effect when `Outline Stage = final`.

### How Research Sources combinations map to common usage

| Research Sources | Behaviour |
| :--- | :--- |
| (blank) | No research sub-pipeline runs. Useful only if the user is relying entirely on `External References` or has no research need. |
| `video` | Pulls relevant videos from HVAC School and Ty Branaman YouTube channels, transcribes, classifies, and chunks them. |
| `web` | Runs web search per subtopic, screens results, scrapes article content via Firecrawl, and extracts relevant info. |
| `deep` | Runs Gemini-with-grounding (or Perplexity) deep research per subtopic, with sources. |
| `video, web, deep` | Full sweep. Highest-quality reference base; longest runtime. |

---

## Block diagram (Final stage)

```
            Course Sheet (Course info + Base Outline)
                              │
                              ▼
       ┌────────────────────────────────────────────┐
       │ Read Course info → Outline Stage = final?  │
       │ Read research_sources, external_references │
       └────────────────────────────────────────────┘
                              │
       ┌──────────────┬───────┴───────┬──────────────┐
       ▼              ▼               ▼              ▼
  Section 1      Section 2       Section 4     (External
  Videos         Web Research    Deep          References
  Research       (if web ∈       Research      collected
  (if video ∈    sources)        (if deep ∈    from
   sources)                       sources)     Course info)
       │              │               │              │
       └──────────────┴───────┬───────┴──────────────┘
                              ▼
       ┌────────────────────────────────────────────┐
       │ Get relevant references for LOs            │
       │  • Consolidate research tabs + External    │
       │    References → All References tab         │
       │    (chunked, deduped, source_origin-tagged)│
       │  • Build / load per-course Chroma          │
       │    vectorstore (Cohere embed-english-v3.0, │
       │    persisted to Drive)                     │
       └────────────────────────────────────────────┘
                              │
                              ▼
       ┌────────────────────────────────────────────┐
       │ Retrieve relevant references for LOs       │
       │  • Per-LO ensemble retrieval               │
       │    (Chroma semantic + BM25)                │
       │  • Cohere Rerank on top-k                  │
       │  • Exa web search as fallback              │
       │  → populate `References` / `Reference type`│
       │    / `Reference usage` on Final Outline    │
       └────────────────────────────────────────────┘
                              │
                              ▼
                  Create Final Outline Sheet
                  (copy Base Outline →
                   Final Outline, flatten LOs)
                              │
                              ▼
       Final Outline (with References per row) +
       per-course vectorstore on Drive
                              │
                              ▼
                  Research Notes agent
   (consumes the References column + the existing
    vectorstore — no rebuild)
```

---

## Inputs (Final stage)

| Source | Field / Tab | Role |
| :--- | :--- | :--- |
| Course Sheet → `Course info` | Outline Stage, Research Sources, External References, Course Background, Course Objective Guidelines, Target Audience & Industry, Checklist Link | Stage + feature flags + LLM context |
| Course Sheet → `Base Outline` | Topic / Subtopic rows (and existing Learning Objectives, if any) | Seed for research-query generation; copied verbatim to the `Final Outline` tab |

---

## Outputs (Final stage)

| Tab / Artifact | Written by | Description |
| :--- | :--- | :--- |
| `Videos Research`, `Video Chunks` | Section 1 (if `video ∈ Research Sources`) | YouTube videos from HVAC School & Ty Branaman channels, transcripts, relevance-classified and chunked |
| `Preliminary Research`, `Article Content`, `Relevant Info`, `Research Summaries` | Section 2 (if `web ∈ Research Sources`) | Web search results, scraped article content (Firecrawl), LLM-extracted relevant info |
| `Deep Research` | Section 4 (if `deep ∈ Research Sources`) | Per-subtopic Gemini-grounded research with sources |
| `All References` | Reference loader | Consolidated reference table — every entry from the research tabs and from `External References`, chunked and stored as JSON in `chunks_0`, `chunks_1`, … columns (49 KB per cell). Each row tagged with `source_origin` ∈ `{Video Research, Web Research, Deep Research, External References, Client Reference, Topic Outline, References}`. |
| Per-course Chroma vectorstore (on Drive) | Vectorstore builder | Persistent per-course vector DB built from the `All References` chunks using Cohere `embed-english-v3.0`. Stays on Drive for the Research Notes agent to reuse. |
| **`Final Outline`** | Final stage | A flattened copy of `Base Outline` (one LO per row). Columns: `Topic`, `Subtopic`, `Learning Objectives`, `References`, `Reference type`, `Reference usage`. The `References` columns are populated by the per-LO retrieval step (see below). |

---

## Functional description (Final stage)

The agent runs as a DAG of named steps; sections turn on/off based on `Research Sources`. Each section ends in user-approval checkpoints where appropriate.

**Section 1 — Videos Research** *(if `video ∈ Research Sources`)*

1. Generate per-subtopic YouTube search queries (LLM).
2. Manual review of queries.
3. Retrieve videos from configured HVAC School / Ty Branaman channels (YouTube Data API).
4. Fetch transcripts (slowest step — 10–20 min batch).
5. LLM relevance classification on each transcript.
6. Chunk relevant transcripts into meaningful segments.
7. Identify relevant chunks; manual `Yes/No` review (drives which videos make it into the references downstream).

**Section 2 — Web Research** *(if `web ∈ Research Sources`)*

1. Generate per-subtopic web search queries (LLM).
2. Run web search; screen for relevance.
3. Fetch full article content (Firecrawl).
4. LLM extracts learning-relevant info.
5. Produce research summaries; user approves.

**Section 4 — Deep Research** *(if `deep ∈ Research Sources`)*

1. Grounded LLM research per subtopic (Gemini with Google Search grounding, or Perplexity deep research as alternative).
2. Sources captured alongside the research content.

**Get relevant references for Learning Objectives** *(automatic — reference loading + vectorstore build)*

- Consolidates everything from the research tabs that ran (Videos, Web, Deep Research) plus the user-supplied `External References` from Course Info into a single `All References` tab.
- Each row carries a `source_origin` tag — `Video Research`, `Web Research`, `Deep Research`, or `External References` — so downstream agents can tell internal from external content.
- Content is chunked (web articles via a general chunker; YouTube via chapter-aware chunking) and stored as JSON across `chunks_0`, `chunks_1`, … columns (49 KB per cell to stay inside Sheets limits).
- Builds the per-course **Chroma vectorstore** using Cohere `embed-english-v3.0`, then persists it to a `Vectorstore files/chroma_research_db` folder under the course's Drive folder. If a vectorstore already exists for the course, it's reused (idempotent).

**Retrieve relevant references for Learning Objectives** *(automatic — RAG retrieval)*

- For each row in `Final Outline`, an agentic retriever queries the vectorstore with the row's Learning Objective and surrounding context.
- Retrieval is hybrid: `EnsembleRetriever` combining Chroma semantic search with BM25, wrapped by `ContextualCompressionRetriever` and reranked by Cohere `rerank` on top-k=20.
- If recall is thin, the retriever can escalate to **Exa web search** as a fallback (up to a small number of agent turns before terminating).
- Selected references are written into the `References`, `Reference type`, and `Reference usage` columns of the row — that's the contract the Research Notes agent reads.

**Create Final Outline Sheet**

- Because `Outline Stage = final`, the source is `Base Outline` (no consolidation, no enhancement).
- Multi-LO cells are flattened to one LO per row.
- Combined with the previous two steps, the resulting `Final Outline` row carries: Topic, Subtopic, Learning Objective, and a curated set of references per LO.

---

## Operating procedure (Final stage)

1. Confirm Course Info has `Outline Stage = final`, `Research Sources` set to the desired subset (or blank), and any URLs in `External References`.
2. Open *Course Outline* in the Streamlit sidebar and load the Course Sheet.
3. The page reads Course Info and shows only the sections enabled by `Research Sources`.
4. Run the agent with **Run all automated steps** or the **Run in background** button.
5. The `Final Outline` tab is the structural contract for the next agent - research notes.

---

## Role in the pipeline

| Stage | Agent | Reads from | Writes to |
| :--- | :--- | :--- | :--- |
| 0 | Template Sheet Setup | — | Course Sheet, Checklist Sheet |
| 1 | **Course Outline** *(this agent)* | Course Sheet (`Course info`, `Base Outline`), Checklist Sheet | Course Sheet (research tabs, `Final Outline` tab) |
| 2 | Research Notes | Course Sheet (research tabs, `External References`, `Final Outline`), Checklist Sheet | Course Sheet (`Final Outline` — `research_notes` column) |
| 3 | Slide Chunks | Course Sheet, Checklist Sheet | Course Sheet (`Slide Chunks` tab) |
| 4 | Graphics | Course Sheet, Checklist Sheet | Course Sheet (`Slide Chunks` — `final_graphics_definition`) |
| 5 | Assessment | Course Sheet, Checklist Sheet | Course Sheet (`Assessment` tab) |

---

## Impact

- **Reference-base as a first-class deliverable.** Final stage treats reference collection — not outline generation — as the agent's primary job. The Research Notes agent gets a richer, source-tagged evidence base to draw from.
- **Mix-and-match research streams.** Independent toggles per source mean the same agent serves "internal only" courses (videos), "web only" courses, and full-sweep courses without code branching.
- **User-supplied references blended automatically.** `External References` lets content authors inject curated URLs into the vectorstore alongside agent-collected material.
- **Trust the user's outline.** Final stage doesn't second-guess a detailed Base Outline — it just goes and finds material for it.

---

## Initial stage *(brief)*

When `Outline Stage = initial`, the agent additionally runs:

- **Outline Consolidation** — an LLM merges the partial outlines from the enabled research sub-pipelines into a single outline, enforcing `Required Topic Count` (±1 tolerance). Followed by a human `Approved` / `Rejected` review-and-revise loop.
- **Enhance Outline** *(only if `Outline Topic Deep Research = TRUE`, default `FALSE`)* — per-topic grounded deep research, then LLM-generated Learning Objectives, categorized and labeled, with a final human review and a diff view against the pre-enhancement outline.
- **Checklist-driven Review & Revise** — reads the AI Checklist Sheet and uses it as a rubric for an LLM self-review pass over the Final Outline, with the option to revise.

`Outline Topic Deep Research` and `Required Topic Count` are **only consulted in Initial stage** and have no effect in Final stage. In practice Initial stage is used infrequently; most courses run in Final.

---

## Dependencies

- **LLMs:** Gemini 3 Flash (default), Gemini 2 Flash, Gemini 2.5 Flash (checklist), Gemini with grounding (deep research). Perplexity deep-research available as an alternative.
- **Embeddings + reranking:** Cohere `embed-english-v3.0` (vectorstore build), Cohere `rerank` (retrieval).
- **Vector DB:** Chroma (SQLite-backed), persisted per course to Google Drive.
- **Retrieval stack:** `EnsembleRetriever` (Chroma + BM25) wrapped by `ContextualCompressionRetriever`; **Exa Search** as a web-search fallback for thin recall.
- **External APIs:** YouTube Data API (3 rotating keys), Google Custom Search or Tavily-class web search, Firecrawl (article extraction), Google Sheets / Drive, LangSmith (tracing).
- **Internal sources:** HVAC School and Ty Branaman YouTube channels (hardcoded channel IDs).
- **Env vars:** `GCLOUD_YT_SEARCH_API_KEY_1/2/3`, `GOOGLE_CSE_API_KEY`, `GOOGLE_CSE_ID`, `FIRECRAWL_API_KEY`, `GOOGLE_API_KEY`, `COHERE_API_KEY`, `EXA_API_KEY`, `GDRIVE_SA_B64`, `LANGCHAIN_API_KEY`, `LANGCHAIN_PROJECT`.

---

## Source

| File | Role |
| :--- | :--- |
| `course_outline.py` | Streamlit page — reads Course Info flags, builds the pipeline DAG, orchestrates the UI |
| `agent_ui_template.py` | Shared UI engine (step dependencies, run/delete, manual-step handling) |
| `agents/course_outline/video_research/*` | Section 1 sub-agents |
| `agents/course_outline/web_research/*` | Section 2 sub-agents |
| `agents/course_outline/deep_research/*` | Section 4 sub-agents |
| `agents/course_outline/outline_consolidation/outline_consolidation.py` | Initial-stage consolidation LLM logic |
| `agents/course_outline/enhance_outline/*` | Initial-stage enhancement sub-agents (LOs etc.) |
| `agents/course_outline/course_outline_checklist/course_outline_checklist.py` | Initial-stage checklist-driven review-and-revise |
| `agents/research_notes/load_references.py` | Consolidates research tabs + `External References` into the `All References` tab *(module lives under `agents/research_notes/` historically, but is invoked from `course_outline.py`)* |
| `agents/research_notes/vector_store.py` | Builds the per-course Chroma vectorstore and persists it to Drive |
| `agents/research_notes/retriever.py` | Ensemble retriever (Chroma + BM25) with Cohere rerank |
| `agents/research_notes/retriever_agent.py` | Agentic per-LO retrieval with Exa web-search fallback |
| `services/helper_functions.py` — `create_final_outline_sheet()` | Flatten Base Outline (Final stage) or enhanced outline (Initial stage) into the Final Outline tab |
