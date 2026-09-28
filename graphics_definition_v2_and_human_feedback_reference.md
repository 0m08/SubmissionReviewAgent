# Graphics Definition V2 and Human Feedback App

Reference for future chats. Grounded in the code as of 2026-09-23.

This file describes only what is actually wired up: steps registered in the Streamlit pipeline, and controls a reviewer can reach in the Human Feedback UI. Code that still exists but is commented out of the pipeline, or whose buttons were removed from the review page, is not part of the product. Do not document it, and do not treat it as a live flow.

If this file and the code disagree, the code wins. Update this file when pipeline behavior, sheet columns, or review-app routes change.

Do not re-scan every agent file to answer "how does Graphics Definition V2 work?" or "how does the human feedback app work?" Read this first, then open only the function you are about to change.

---

## 1. What these two systems are

**Graphics Definition V2** is a Streamlit pipeline (`graphics_definition_v2.py` → `agent_ui()` in `agent_ui_template.py`). It turns each row of a course Google Sheet tab named `Slide Chunks` into a final visual assignment, a slideshow manifest, and overlay-animation plans.

**Human Feedback** is a FastAPI app (`human_feedback_app/backend/main.py`) plus a review UI. A reviewer loads the same sheet, watches the slideshow, and approves or replaces individual visuals. A visual revision calls `process_human_feedback_row`. Overlay plans are refreshed after that URL change. The app does not re-run the Streamlit pipeline.

The two systems share:

- Google Sheet tabs `Slide Chunks` and `Course info`
- Asset-library selection stored on Course info
- `final_graphics_definition` as the source of truth for which asset is assigned to which voiceover
- `slideshow_manifest` as the source of truth for how those assets are composed into scenes

---

## 2. How a run is hosted

Entry: `graphics_definition_v2.py` builds `pipeline_sections` and calls:

```python
agent_ui(step_name="Graphics Definition V2", pipeline_sections=pipeline_sections, ...)
```

`agent_ui_template.py` renders steps, enforces `depends_on`, hides steps, and injects session args.

### 2.1 Before running

The page shows `TOP_INSTRUCTIONS`. Two things must be set:

1. **Visual Assignment Strategy** on each `Slide Chunks` row (dropdown). The layout-plan step's pre-exec creates this column and fills blanks if it is missing.
2. **Asset libraries for this run.** Checkboxes in the Streamlit UI. Source of truth is the Course info cell `Allowed Asset Search Libraries for the Graphics Agent` (row 2). The UI loads that cell on sheet open and writes it back when toggles change.

Also: **Topics to run.** Multiselect over the `Topic` column. `selected_topics = []` means all topics. Most later steps skip rows whose `Topic` is outside the selection. Layout planning does **not** filter by topic.

### 2.2 Dependency and hide rules

A step runs only when every name in `depends_on` is either marked done or is a hidden step whose own parents are satisfied (`_dependency_is_satisfied`).

A step is hidden when:

| Flag on the step | Hidden when |
|---|---|
| `hide_if_web_disabled` | `graphics_v2_web_fallback_enabled` is false (the "Web Images and Other YouTube Channel Videos" checkbox is off) |
| `hide_if_external_references_disabled` | External References checkbox is off |

Hidden steps do not block dependents. So if External References is off, "Index External Reference Assets" is hidden and layout planning can still run. If Web is off, all three Section 9 steps are hidden and "Generate Slideshow Manifest" (which depends on "Decide which visual to Use") can still run.

### 2.3 What "enabled sources" actually means

Defined in `agents/graphics_definition_v2/candidate_search/pool_registry.py`.

| Source id | UI / Course info label | Used for |
|---|---|---|
| `drive_images` | Drive Images | Section 5 Drive image search |
| `hvac_youtube` | HVAC YouTube Videos | Section 5 HVAC YouTube embed search |
| `drive_videos` | Google Drive Videos (All) or (NexTech Only) | Section 5 Drive video search |
| `external_references` | External References | Section 5 Supabase search of this sheet's indexed refs |
| `web_images` | Web Images and Videos | Section 9 fallback, and human-feedback "search the web". **Not** a Section 5 runner |
| `youtube_other_channels` | (same Web checkbox) | Same as web images |

Drive video mode: `all` or `nextech`. If both Course info labels are selected, `all` wins.

**Product default when the Course info cell is blank** (`default_enabled_sources_for_course_info`): Drive Images, HVAC YouTube, Drive Videos (All), and Web. External References is off.

**Section 5 runner list** is the session key `graphics_v2_enabled_sources`, which is the full selection **minus** `web_images` and `youtube_other_channels`. Web is stored separately as `graphics_v2_web_fallback_enabled` and only gates Section 9.

The UI refuses to run if every primary library is off and only Web is on: "Web asset library alone is not enough to run the agent."

`resolve_enabled_sources` (used when a caller passes an explicit list or the legacy boolean):

1. Explicit list: keep known ids, drop unknowns.
2. Else `use_only_drive_and_hvac is False`: all known sources, including external refs and web.
3. Else (`True` or `None`): `drive_images` + `hvac_youtube` + `drive_videos` only.

### 2.4 LLMs and parallelism used by the live pipeline

| Step | LLM arg | max_workers |
|---|---|---|
| Extract / index external refs | none (LlamaParse + Gemini embeddings) | 50 inside extraction |
| Layout plan | `gemini_3_flash_thinking` | 50 |
| Segment slide | `gemini_2_5_flash_lite` | 50 |
| Storyboard | `gemini_3_flash_thinking` | 50 |
| Search queries | `gemini_2_5_flash_lite` | 50 |
| Reference image pool | `gemini_3_flash` | 50 |
| Candidates | no LLM (retrieval) | 50 per runner; runners themselves run in parallel |
| Pools | `gemini_3_flash_thinking` for image and video | 50 |
| Aggregation | `gemini_3_flash_thinking` | 50 |
| Review and revise | `gemini_3_flash_thinking` | 50 |
| Section 9 (three steps) | `gemini_3_flash_thinking` | 50 |
| Slideshow manifest | `gemini_3_flash_thinking` | 50 |
| Overlay animations | `gemini_3_flash_thinking` | 50 |

Pricing table in `graphics_definition_v2.py` is for the Streamlit cost display only.

### 2.5 The usual skip pattern

Almost every row runner:

- Skips rows already filled (non-empty, not `nan`, not starting with `ERROR:`).
- Skips rows with missing required input (empty slide body, empty voiceover, empty search queries, and so on).
- Writes `ERROR: ...` into the output cell on failure.
- Retries empty / `ERROR:` cells, usually up to 3 times.
- Respects `selected_topics` except layout planning.

Delete functions drop the columns that step wrote (and sometimes restore earlier cells). They do not roll back downstream columns.

---

## 3. Sheet contract

### 3.1 Worksheets

| Tab | Role |
|---|---|
| `Slide Chunks` | One row per slide. Almost every agent reads and writes here. |
| `Course info` | Row 2 holds course-level settings and external-ref logs. |

Course info headers these systems read or write:

| Header | Who |
|---|---|
| `Course Name` | Almost every LLM step |
| `Target Audience & Industry` | Review, aggregation-adjacent review, human feedback, overlays |
| `External References` | Newline-separated source URLs for extraction/indexing |
| `Allowed Asset Search Libraries for the Graphics Agent` | Comma-separated labels. Shared by Streamlit UI and the FastAPI app |
| `external_ref_extraction_log` | JSON log, extraction step |
| `external_ref_index_log` | JSON log, indexing step |

### 3.2 Slide Chunks columns the pipeline expects to already exist

These are course-content columns. The graphics agent does not create them (except the title-transition rows inserted before layout planning):

- `Topic`
- `Subtopic`
- `Slide Type` (`Transition` / `transition` is special)
- `Slide Chunk Title`
- `Slide Chunk` — this is the narration body. There is no column named `voiceover` or `Slide content`.

### 3.3 Columns the live pipeline creates, in order

| Column | Written by | Notes |
|---|---|---|
| `Visual Assignment Strategy` | pre-exec of layout plan | Dropdown. See §4 |
| `layout_plan` | layout plan | Inner scene XML, tags stripped |
| `voiceover_segment` | segment slide | One sentence per line |
| `slide_chunk_raw` | segment slide, inline-image path only | Original `Slide Chunk` before markers are stripped. Hidden |
| `reference_image_map` | segment slide (inline markers) and reference-pool step | `SEGMENT_n:url` lines |
| `storyboard_planning` | storyboard | `When VO` / `Visual Idea` blocks |
| `search_queries` | search-query step; later patched by Section 9 and review regen | `---SEGMENT_n---` blocks |
| `drive_results` | Drive image search | `Title: … \| URL: …` |
| `video_pool` | HVAC YouTube search | raw embed URLs |
| `drive_video_pool` | Drive video search | `Title \| URL (start=&end=)` |
| `external_ref_pool` | external-ref search, only if that source is on | images and video segments |
| `image_score` | image scoring | per-set `Score: n/10` |
| `video_score` | video scoring | same |
| `image_pool` | image selection | shortlisted then LLM-filtered |
| `video_pool_filtered` | video selection | same; external clips marked `FramesOnly: yes` |
| `final_graphics_definition` | aggregation; later mutated by review, Section 9 decide, and a human-feedback visual revision | The assignment |
| `graphics_review_v2_notes` | review and revise | `alignment=PASS\|FAIL; segmentation=SKIPPED` |
| `review_complete` | review and revise | `TRUE` / `FALSE`. Rows with `TRUE` are skipped on rerun |
| `revision_tracking` | review and revise | per visual id, alignment loops |
| `web_results` | Section 9 web fallback (column may be created there) | not filled in Section 5 |
| `video_pool_other_channels` | Section 9 (column may be created there) | not filled in Section 5 |
| `final_visuals_review` | Section 9 review | `S#V#: PASS` or `FAIL \| Reason \| Feedback` plus optional `Replacement visual:` |
| `decision_of_final_visual` | Section 9 decide | original vs replacement vs selection |
| `slideshow_manifest` | slideshow manifest; regenerated by the overlay pre-check and by manifest sync after a visual URL change | scene XML |
| `hero_animation_plan` | overlay | per scene |
| `hero_bbox_coordinates` | overlay, bbox type only | |
| `hero_icon_overlays` | overlay, icon type only | |
| `hero_callout_overlays` | overlay, callout type only | |
| `multivisual_animation_plan` | overlay | per non-hero scene |

`layout_plan_evaluation` is listed in the layout-plan delete function and is parsed from the LLM, then **discarded**. It is never saved. Slideshow manifest evaluation is also discarded.

### 3.4 Columns the human-feedback app uses (round suffix)

`ROUND_INDEX = 0` in `human_feedback_app/backend/sheet_service.py`. `get_round_column_name(base, 0)` in `graphics_definition_v2_slideshow.py` returns `{base}_1`. The app never advances the round.

So the app reads and writes:

| Constant | Actual header |
|---|---|
| `human_feedback` | `human_feedback_1` |
| `human_feedback_status` | `human_feedback_status_1` |
| `human_feedback_revision_tracking` | `human_feedback_revision_tracking_1` |
| `human_review_actions` | `human_review_actions_1` |

These have **no** round suffix: `final_graphics_definition`, `slideshow_manifest`, all overlay columns, `player_scene_overrides`, and every pool column.

The app calls `process_human_feedback_row` with the `_1` names above. It does not call the batch runner `run_human_feedback_review_revise_for_all_rows`.

### 3.5 `final_graphics_definition` shape

Written by `format_aggregation_definition_for_sheet`. This is the format every later parser expects:

```
================================================================================
SEGMENT N
================================================================================

When VO: "{sentence or full slide}"

Visual Instructions: {instruction}

Graphics to use: {url}

Selection Justification: {justification}
```

Multiple visuals inside one segment are separated by a line of dashes (`----`). Visual ids are **not stored in this cell**. They are assigned when parsed: segment N, step M (1-based) becomes `S{N}V{M}`.

`Graphics to use:` is the asset key. `Assigned Asset:` is only an alias accepted later by slideshow-manifest parsing.

YouTube timestamp links are normalized (`t=` → `start=`), single-timestamp clips are expanded, frame snapshots can be uploaded to Drive and labeled `(snapshot)`.

### 3.6 `slideshow_manifest` shape

The cell stores the inner scene tree (the model is asked for a `<slideshow_manifest>` wrapper; the saved cell is the inner XML). Parser in `image_edit_planning.py` wraps it if the root tag is missing.

```xml
<scene id="1" template="single_visual_hero">
  <narration_span>...</narration_span>
  <slot role="primary_visual" asset="https://..."/>
</scene>
```

| `template` | Required slots |
|---|---|
| `single_visual_hero` | exactly 1: `primary_visual` |
| `two_item_split_comparison` | exactly 2: `left_visual`, `right_visual` |
| `multi_panel_grid` | exactly 3 or 4: `panel_1` … `panel_4` |
| `main_plus_supporting_inset` | exactly 2: `main_visual`, `inset_visual` |

Scene ids are `"1","2","3"` with no gaps. Every `Graphics to use:` URL must land in exactly one slot. The model must not invent URLs.

---

## 4. Visual Assignment Strategy

Column: `Visual Assignment Strategy`. Dropdown values, in this order:

1. `1 Visual per Sentence`
2. `1 Visual for the whole Slide`
3. `Flexible, let the agent decide`

Filled by `ensure_visual_assignment_strategy_column` (pre-exec of layout planning):

- Missing column: add it.
- Blank / NaN cells: fill them.
- Default: `Flexible, let the agent decide`.
- If `Slide Type` lowercased is `transition`: `1 Visual for the whole Slide`.

`ensure_title_transition_rows` runs in the same pre-exec, and only if **no** row already has a `layout_plan`:

- For each `Topic` group, if the first row is not already a transition whose `Slide Chunk` equals the topic name, insert a row:
  - `Slide Type` = `Transition`
  - `Slide Chunk Title` and `Slide Chunk` = topic name
  - `Visual Assignment Strategy` = `1 Visual for the whole Slide`
- Aborts the insert if `Topic` or `Slide Type` is missing.

How each strategy changes later steps:

| Strategy | Search queries | Storyboard | Aggregation | Pools |
|---|---|---|---|---|
| Flexible | 3 queries per voiceover line | per-step storyboard | may emit **multiple** visual steps per sentence; default is one | one score/select block per voiceover line |
| 1 Visual per Sentence | same as Flexible (3 per line) | one visual idea per sentence | **exactly one** visual for the whole sentence | same as Flexible |
| 1 Visual for the whole Slide | **4** queries, all under `---SEGMENT_1---`, from the full `Slide Chunk` | one idea for the whole slide | candidates from **segment 1 only**; exactly one visual; formatted as `SEGMENT 1` | scoring/selection uses segment 1 only, `entire_slide=True` |

Empty / `nan` strategy is treated as Flexible everywhere it is read.

Layout planning itself does **not** read the strategy. Segmentation does **not** read it either. The three strategies do not change how sentences are split.

If a segment has an inline reference URL (`reference_image_map`), aggregation **forces** that segment onto the 1-visual-per-sentence path.

---

## 5. Live pipeline, step by step

Order below is the order in `pipeline_sections`. Commented-out sections are in §6.

### 5.0 External Reference Media Extraction

Hidden unless External References is checked.

**Step: Extract External Reference Images** — `run_external_reference_extraction`

- Reads Course info `External References` (newline-separated unique URLs) and `Course Name` from row 0.
- Drive parent folder id: `1-sS6py5FVDJDI85rMkMBOQH2qIhccs_z`.
- Supports Google Docs, Google Slides, PDF, PPT, PPTX. Docs/Slides are exported to PDF. LlamaParse (`tier="agentic"`, embedded images) extracts images. Keys rotate across `LLAMA_CLOUD_API_KEY_1` … `_10`.
- YouTube links and Drive folder links are skipped (`skipped_non_document`). Unsupported MIME is `skipped_unsupported_mime`.
- Images are deduped (MD5, and pHash Hamming ≤ 8). Uploaded under `{course}_{sheet_id}/{sha256(source)[:12]}/img_00N.ext`, shared `anyone` / `reader`.
- Writes Course info `external_ref_extraction_log` (row 2) as JSON: `ran_at`, `course_name`, `sheet_id`, `parent_folder_id`, `sources_total`, `images_uploaded`, `per_source[]` with `source_link`, `status`, `assets`, `error`, and `asset_urls` on success.
- No links: writes an empty log and returns. Does not raise.
- Drive missing or no spreadsheet id: raises.
- Delete: removes the Drive folder `{course}_{sheet_id}` and clears the log cell. Does not drop the column.

**Step: Index External Reference Assets** — `run_external_reference_indexing`. Depends on extraction.

- Embeds with Gemini Embedding 2 (`gemini-embedding-2`, 3072-d, `RETRIEVAL_DOCUMENT`) into Supabase table `external_ref_assets` only. Match RPC later is `match_external_ref_assets`. Never writes curriculum/course tables.
- `asset_id` = SHA-256 of `{sheet_id}|{source_link}|{asset_url}|{start_time}|{end_time}` truncated to 32 hex. Every row is scoped by `sheet_id`.
- Images: from `asset_urls` in the extraction log. `asset_type=image`, `source_kind=pdf_extract`. If a document link has no extraction URLs, status is `needs_extraction` ("Run Extract External Reference Images first").
- YouTube videos: yt-dlp, 30-second chunks, `source_kind=youtube_video`, `asset_type=video_segment`. Always re-indexed.
- Drive videos: 60-second chunks, `source_kind=drive_video`. Always re-indexed. Clips are embedded in memory; they are not re-uploaded.
- Other already-indexed links with `existing > 0` and not video: status `unchanged`, skipped.
- Stale sources (in Supabase for this `sheet_id` but no longer in Course info) are deleted.
- Writes `external_ref_index_log` JSON on Course info row 2.
- Delete: `delete_assets_for_sheet(sheet_id)` then clears the log. Refuses to delete Supabase rows if `sheet_id` cannot be resolved.

### 5.1 Section 1 — Generate Layout Plan for each Slide

`run_layout_planning_agent_for_all_rows`. Pre-exec: `prepare_slide_chunks_for_layout_plan` (title-transition rows + VAS column, §4). Depends on indexing, which is skippable when external refs are off.

Purpose: split one slide's narration into coarse **scenes** and pick exactly one layout from a 4-item library, with role-based slots. This happens **before** any asset exists. It is not the later layout-revision agent.

Layouts: `single_visual_hero` (1 slot), `two_item_split_comparison` (2), `multi_panel_grid` (3–4), `main_plus_supporting_inset` (2). Default for a Transition slide in the prompt: one `single_visual_hero` scene.

Reads: Course Name; `Topic`, `Subtopic`, `Slide Chunk Title`, `Slide Chunk`, `Slide Type`.

Writes: `layout_plan` only (inner `<scene>` blocks, outer tags stripped). Evaluation XML is thrown away.

Skips: empty `Slide Chunk`; `layout_plan` already filled. No topic filter.

Scene block shape:

```xml
<scene>
<scene_id>SCENE_1</scene_id>
<narration_span>...</narration_span>
<layout_category>single_visual_hero | two_item_split_comparison | multi_panel_grid | main_plus_supporting_inset</layout_category>
<layout_rationale>...</layout_rationale>
<required_slots>
- slot_name: role and what must be visible
</required_slots>
</scene>
```

Delete drops `layout_plan` and `layout_plan_evaluation` if present.

### 5.2 Section 2 — Segment Slide into Voiceover Segments

`run_segment_slide_from_slide_chunk_for_all_rows`. Depends on layout plan.

Not a strategy switch. Two paths:

**Inline markers.** If `Slide Chunk` contains markdown `[alt](http...)`:

- Split at each marker. Alt text is stripped. Trailing punctuation stays with the preceding text.
- A chunk longer than 10 characters that contains `.!?` or a newline is sent through the sentence LLM. Otherwise the chunk is one segment.
- `reference_image_map` gets `SEGMENT_n:{url}` on the **last** sub-segment of that chunk.
- `slide_chunk_raw` stores the original. `Slide Chunk` is rewritten with markers removed.
- Does **not** change Visual Assignment Strategy. Log line says "Strategy unchanged (Flexible)."

**No markers.** LLM sentence split (`segment_slide_prompt`). Split only at `.` `!` `?`. Copy text exactly. XML `<segments><segment>…</segment></segments>`. Inners are joined with newlines into `voiceover_segment`.

Skips: topic filter; empty `Slide Chunk`; `voiceover_segment` already filled. Retries empty / `ERROR:` up to 3 times.

Delete: restores `Slide Chunk` from `slide_chunk_raw`, drops `voiceover_segment`, `slide_chunk_raw`, `reference_image_map`. Does not drop VAS.

### 5.3 Section 3 — Generate Storyboard for each Slide

`run_storyboard_agent_for_all_rows`. Depends on segmentation.

Reads layout plan, VAS, voiceover, `reference_image_map`, slide fields. Missing `layout_plan` warns and continues with `""`.

Branching in `process_storyboard_row`:

| Condition | Prompt path |
|---|---|
| `reference_image_map` is non-empty | Locked-segment path. Uses the voiceover lines as fixed narration. **Ignores VAS. Does not pass `layout_plan`.** |
| `1 Visual for the whole Slide` | entire-slide prompt |
| `1 Visual per Sentence` | per-sentence prompt |
| else | flexible prompt |

Sheet format after `format_storyboard_for_sheet`:

```
When VO: "{narration}"

Visual Idea: {idea}

-----------------------------------------------
```

Entire-slide stores one block whose When VO is the full `Slide Chunk`. Locked path copies the exact voiceover line (not the model's narration tag). Missing visual idea becomes `(See reference image for this segment)`.

Writes `storyboard_planning` only. Skips filled cells, empty slides, off-topic rows. Retry ×3. Delete drops `storyboard_planning`.

### 5.4 Section 4 — Search queries, then reference image pool

**Generate Search Queries** — `run_generate_search_query_for_all_rows`. Depends on storyboard.

| VAS | Unit | Queries |
|---|---|---|
| whole slide | full `Slide Chunk` | 4, under `---SEGMENT_1---` |
| Flexible or per sentence (and any other value) | each line of `voiceover_segment` | 3 per segment |

Cell format:

```
---SEGMENT_1---
query one
query two
query three
```

Blocks separated by a blank line. Parser (`parse_search_queries_column` in `drive_search.py`) splits on `---SEGMENT_(\d+)---` and treats each non-empty line as a query.

Per-segment validation: marker count must equal voiceover line count, sequential from 1. Entire-slide: must contain `---SEGMENT_1---` and at least one query. Retry ×3. Delete drops `search_queries`.

**Map Reference Image Pool** — `run_reference_image_pool_mapping`. Depends on search queries.

This is **not** the Supabase external-ref index. It searches the course Drive image vector store.

Root folder: `st.session_state["root_folder_id"]`, else `1QS6PmCESfgFWNNEpRJDUB0E-t8iMAatH`. `resolve_reference_pool_root_folder_id` walks for a `Vectorstore files` folder, possibly under `Reference Image` / `Vector Store`.

Per segment that is not already in `reference_image_map`: search k=5, keep top 8 by relevance, LLM (`gemini_3_flash`) picks one URL or `NONE`. Accept only `http…`.

Existing inline maps are kept. The pool only fills unmapped segment indexes. Format: `SEGMENT_1:https://…`.

Row skipped when voiceover or search queries are missing, or every segment is already mapped. Delete does **not** drop the column. It rebuilds it from inline markers in `slide_chunk_raw` (fallback `Slide Chunk`), or clears it.

### 5.5 Section 5 — Generate Image and Video Candidates

`run_generate_image_and_video_candidates`. Depends on reference-pool mapping.

Resolves `graphics_v2_enabled_sources` (primary libraries only). Empty list: skip. Runs the matching runners **in parallel** (one thread per source). Each runner then parallelizes rows (`max_workers`).

Also persists the selection to Course info, and if the Web checkbox is on, the persisted labels include Web even though Web runners are **not** started here.

| Source id | Function | Column | Backend | k | Line format | Transition rows |
|---|---|---|---|---|---|---|
| `drive_images` | `run_drive_search_for_all_rows` | `drive_results` | `graphics_retriever` on two central Drive image vectorstores (`1QS6PmCESfgFWNNEpRJDUB0E-t8iMAatH` and `1IMGr4d8lwux5R_cAWfhVjBV0fTFdWvNi`), merged by reference id / file id, distance ≤ 1.5 | 5 | `Title: {title} \| URL: {url}` | not specially skipped |
| `hvac_youtube` | `run_youtube_video_search_for_all_rows` | `video_pool` | Chroma video embeddings, folder `15H9thXq02JX3ldADSj1oD78mbV-fXfvu`. Reranking is off | 5 | raw `https://www.youtube.com/embed/{id}?start=&end=` | skipped (`Slide Type` == `transition`) |
| `drive_videos` | `run_drive_video_search_for_all_rows` | `drive_video_pool` | Gemini Drive-video Chroma. Mode `nextech` filters `source_tag` (discovered tags, else `nextech_hvac` / `nextech_electrical` / `nextech_plumbing`) | 5 | `Title \| URL (start=&end=)`, URL forced to include `usp=drivesdk` | skipped |
| `external_references` | `run_external_ref_search_for_all_rows` | `external_ref_pool` | Supabase `match_external_ref_assets` filtered by this `sheet_id`. Two calls per query: `image` and `video_segment` | 5 per modality | YouTube embed with start/end, or Drive URL with start/end, or image `Title \| URL` | not specially skipped |
| `web_images` | `run_web_search_for_all_rows` | `web_results` | Brave image search (`web_image_search_tool`). Registered, but Section 5 does not enable it | 5 | `Title \| URL` | |
| `youtube_other_channels` | `run_youtube_video_search_other_channels_for_all_rows` | `video_pool_other_channels` | YouTube Data API v3, keys `GCLOUD_YT_SEARCH_API_KEY_1..56`. Embeddable, English, excludes a channel denylist, duration ≤ 15 min. Also searches the voiceover sentence itself | 4 | `Title \| Duration \| Channel \| URL` (`watch?v=`) | skipped |

All query execution goes through `run_pool_search_queries` in `candidate_search/search_wrapper.py` (parallel per query; empty query returns `[]`). The wrapper does not merge sources.

Row skip (typical): off-topic, empty `search_queries`, output column already filled. Drive / HVAC / other-YouTube / Drive-video validate that segment markers match the query segments. External-ref validation is lighter: empty results are valid; only `ERROR:` fails, and it retries once instead of 3.

Delete (`delete_all_candidate_results`) calls each enabled source's delete helper.

**How later steps read candidates** (`candidate_wrapper.py`):

Image columns, in order: `drive_results`, `web_results`, `external_ref_pool` (video lines in the external pool are skipped for images).

Video columns: `video_pool` (type `pool`), `video_pool_other_channels` (`other`), `drive_video_pool` (`drive`), `external_ref_pool` video lines (`external_frames`). A URL is an external-ref video if it has `(start=` and `&end=)` or is a YouTube URL.

`enabled_sources is None` means "read every column that is present."

### 5.6 Section 6 — Generate Image and Video Pools

`run_generate_image_and_video_pools`. Depends on candidates.

Resolves sources on the main thread (`resolve_section6_enabled_sources`): session primary sources, plus `web_images` and `youtube_other_channels` if the Web checkbox is on. At this point those web columns are usually empty, so this only matters if something filled them earlier.

Two phases, each phase runs image and video in parallel:

1. Score: `run_image_scoring_for_all_rows` → `image_score`; `run_video_scoring_for_all_rows` → `video_score`.
2. Select: `run_image_selection_from_all_images_for_all_rows` → `image_pool`; `run_video_selection_from_all_videos_for_all_rows` → `video_pool_filtered`.

Scoring is multimodal Gemini, temperature 0.1. Candidates are batched with `MIN_BATCH_SIZE=3`, `MAX_BATCH_SIZE=5`.

Shortlist per set: `top_n=2`, `min_score=4.0`. Sort by score. If more than 2, keep everyone tied with the item at index 1, then drop scores under 4.0. If that empties the set, force the top item. Score lines are `n/10` clamped to 0–10.

"Filtered" means a second LLM pass over the shortlist: keep everything relevant, drop only pure redundancy. Output is `Title \| URL` (videos keep their source-specific line, and external clips get `FramesOnly: yes`).

Whole-slide strategy scores and selects segment 1 only. Transition slides: video scoring/selection treat an empty cell as valid and skip the row. Image side does not special-case transition.

Skip scoring when the row has no candidates for that modality (and clear a stale score). Skip selection output when the pool cell is already filled. Image selection's all-rows skip does **not** re-check `image_row_has_candidates`; a segment with no URLs is skipped inside the row.

Delete drops `image_score`, `video_score`, `image_pool`, `video_pool_filtered`.

### 5.7 Section 7 — Aggregation Agent

`run_aggregation_agent_for_all_rows`. Depends on pools.

Selects the asset(s) for each voiceover and writes `final_graphics_definition`. `evaluation_breakdown` is computed and discarded. Delete would also drop `evaluation_breakdown` if that column existed.

Skip: off-topic, empty `voiceover_segment`, `final_graphics_definition` already filled. Retry ×3. A row must pass both `validate_final_graphics_definition_row` and `validate_youtube_clip_links_for_row`.

Validation:

- Markers are `SEGMENT N` (the banner), not `---SEGMENT_N---`.
- Whole-slide: zero segments or exactly `SEGMENT 1`. If a reference image exists, a `Graphics to use:` line is required.
- Otherwise segment count must match voiceover lines, sequential from 1.
- When-VO texts must cover `Slide Chunk` in order, with no drops or repeats.
- YouTube start/end must fit the real duration (YouTube Data API, keys `GCLOUD_YT_SEARCH_API_KEY_1/2/3`).

Candidate source for a segment:

- Images from `image_pool`. If that segment's pool is empty, fall back to `get_image_candidates` (drive + web + external ref), filtered by `enabled_sources`.
- Videos from `video_pool_filtered`. If empty and any raw video column is non-empty, fall back through `get_video_candidates`, remapped to `embed` / `full_video` / `drive_clip` / `frames_only`.
- If both lists are empty, skip that segment.

Multimodal prompt (temperature 0.7): images as JPEG parts, HVAC embeds via `build_video_part`, Drive clips in parallel, full videos / frames-only YouTube via `build_video_part`. Transition slides are told that only images are available.

`reference_image_map` (`SEGMENT_n:url`): after the model picks, the first `Graphics to use:` URL is compared multimodally to the reference. If the graphics-definition URL loses, or there is no URL, the reference is injected with justification `Utilised pre-specified and processed reference image.` The reference URL is **not** added to the candidate list before the model call. Forcing per-sentence VAS when a reference exists is the only pre-call effect.

Post-process: expand single-timestamp YouTube clips, normalize timestamp params, extract frames to Drive, add `(snapshot)` labels on those Drive links.

### 5.8 Section 8 — Review and Revise Graphics Definitions

`run_review_and_revise_graphics_definition_v2_for_all_rows`. Depends on aggregation.

Pipeline args that matter:

```python
use_only_drive_and_hvac=True
enabled_sources="graphics_v2_enabled_sources"  # resolved from session
```

`use_only_drive_and_hvac=True` does **not** rebuild the Section 5 pools. It only affects `regenerate_failed_segments`: web image search and other-channel YouTube are forced off, even if `enabled_sources` contains them. Drive / HVAC / Drive-video / external refs stay gated by `enabled_sources` (if that list is missing, all four default to on).

**What actually runs today**

| Criterion | Status |
|---|---|
| Alignment | Runs. One review pass, then revise-from-pool, then a follow-up review, then up to 2 regeneration attempts |
| Segmentation quality | Hardcoded `SKIPPED`. The function `run_segmentation_quality_loop_for_slide` still exists and is commented out at the call site. It would have run only for Flexible |
| Specificity | Hardcoded `SKIPPED`. Comment in code: alignment already includes a light specificity check, and a second pass was overwriting good replacements |
| Topic-level redundancy | Commented out. `find_repeated_urls(..., min_occurrences=4)` is unused |

Alignment FAIL means a visual does not show what that voiceover says (wrong subject, generic, or unusable). Slide fails if any segment fails. Whole-slide strategy uses `ALIGNMENT_REVIEW_PROMPT_FOR_ONE_VISUAL_PER_SLIDE`.

Loop constants: `MAX_REVIEW_ATTEMPTS = 1`, `MAX_REGEN_ATTEMPTS = 2`. Tracking still has slots for loops 1–3.

Per slide (`run_review_loop_for_slide`):

1. Review attempt 1, multimodal, targets formatted as `S#V# | When VO | Asset`.
2. PASS → mark remaining loops "No replacement", return.
3. FAIL → `revise_segment_visuals` from `image_pool` + `video_pool_filtered`, with raw-column fallback filtered by `enabled_sources`. Merge via `update_final_graphics_definition_with_replacements`.
4. If the revise produced no updates, break into regeneration.
5. Follow-up review. PASS returns. FAIL goes to regen.
6. Regen attempts 1 and 2: `generate_search_queries_with_feedback`, then search only the sources regen is allowed to use (no web, no other-YouTube, because the flag is True). k is `REGEN_IMAGE_SEARCH_K = 4`, `REGEN_VIDEO_SEARCH_K = 3`. External ref regen k is 4. Then re-aggregate replacements.

Transition rows: write `graphics_review_v2_notes="-"`, `review_complete="TRUE"`, `revision_tracking="-"` and return. No LLM.

`review_complete` is `TRUE` when alignment and segmentation are both in `{PASS, SKIPPED}`. Because segmentation is always SKIPPED, `review_complete` tracks alignment. Rows with `TRUE` are skipped on the next run.

Columns written: updated `final_graphics_definition`, `graphics_review_v2_notes`, `review_complete`, `revision_tracking`.

`revision_tracking` shape:

```
S1V1

Alignment
Original visual - {url}
Visual after loop 1 - {url|No replacement}
Visual after loop 2 - …
Visual after loop 3 - …

Segmentation
Original visual - {url}
Visual after loop N - {url|No replacement|N/A (merged out)}
```

There is no persisted review-breakdown column. Delete drops `review_complete`, `graphics_review_v2_notes`, `revision_tracking`.

After the loops, the row's definition is normalized again (YouTube params, frame extraction, snapshot labels).

### 5.9 Section 9 — Review finalized visuals, web fallback, decide

All three steps have `hide_if_web_disabled: True`. If Web is off they disappear and the slideshow step is unblocked.

**Review finalized visuals** — `run_review_finalized_visuals_for_all_rows`. Depends on Section 8.

Writes `final_visuals_review`. Transition → `"-"`. Skip if definition empty or review already filled.

PASS: the visual clearly shows the spoken meaning, is specific, and is usable. FAIL: wrong subject, generic, contradictory, or unusable. Slide PASS only if every visual PASSes.

Cell format:

```
---SEGMENT_1---

S1V1: PASS
S1V2: FAIL | Reason: {reason} | Feedback: {needed_visual}
```

Parser only collects `<failure>` blocks that sit inside `<failures>`. The whole-slide prompt emits a single `<failure>` **without** that wrapper. If the model follows that prompt, `failures` is empty and every visual is marked FAIL with reason `Model returned FAIL without visual-level failure blocks`.

**Generate alternative visuals from Web** — `run_generate_alternative_visuals_from_web_for_all_rows`.

Runs only when `final_visuals_review` contains `FAIL` and that FAIL line does not already have `| Replacement visual:`.

For each failed segment:

1. New queries via `generate_search_queries_with_feedback`; that segment's block in `search_queries` is replaced.
2. `process_web_search_segment` → `web_results`.
3. `process_segment_other_channels` → `video_pool_other_channels` (not on transition; the row is already skipped for transition).
4. `revise_segment_visuals` with image pool, filtered video pool, drive results, and HVAC video pool **forced empty**. Only web + other-channel results are candidates.

The replacement URL is appended on the FAIL line: `| Replacement visual: {url}`. **`final_graphics_definition` is not changed here.**

Ensures columns `web_results` (after `drive_results` if that exists) and `video_pool_other_channels`.

Delete replacements strips the `| Replacement visual: …` suffix. It does not drop the column.

**Decide which visual to Use** — `run_decide_final_visuals_for_all_rows`.

Writes `decision_of_final_visual` (created immediately after `final_visuals_review`) and may update `final_graphics_definition`.

Only FAIL lines that already have a replacement are candidates.

- Option A / FIRST = current `Graphics to use` URL.
- Option B / SECOND = replacement URL.
- Same URL: keep A, no LLM.
- Empty current URL: take B, no LLM.
- Unparseable model output: keep A.
- Visual id missing from the definition: skip, keep current.

Winner is merged with `update_final_graphics_definition_with_replacements`, then the usual YouTube / frame / snapshot post-process.

Decision cell:

```
---SEGMENT_1---

S1V1
Original Visual: {url}
Replacement Visual: {url}
Selection: {url}
```

No candidates → `No Failed visuals` and the definition is unchanged.

Delete parses the decision column and reverts the definition only where selection equals the replacement and they differ, then drops `decision_of_final_visual`.

### 5.10 Section 10 — Generate Slideshow Manifest

`run_slideshow_manifest_for_all_rows`. Depends on "Decide which visual to Use" (satisfied automatically when Section 9 is hidden).

Reads `final_graphics_definition`, topic/subtopic/title/chunk/type, `layout_plan`, `storyboard_planning`.

Writes `slideshow_manifest` only. Evaluation is discarded. The pipeline description string that mentions an evaluation column is stale.

Skip if definition is empty, or manifest is already filled and not `ERROR:`. Up to 2 retry passes for empty / `ERROR:` cells.

The model must 1:1-map every assigned URL into exactly one slot, prefer fewer scenes when one template covers several When-VO beats, and split when the template or the asset set changes. Transition may be a single hero scene.

Delete drops `slideshow_manifest`.

### 5.11 Section 14 — Decide Overlay Animations

`run_overlay_animation_decisions_for_all_rows`. Depends on the slideshow manifest. Overlays are planned on the assets already in the manifest.

Order inside the function:

1. `ensure_slideshow_manifests_synced_for_overlay_step` (`manifest_url_sync.py`). Compare definition URLs to manifest slot URLs (count, membership, leftovers). Match is exact token or Drive file id / trimmed URL (`urls_match_for_graphics_assignment`). On mismatch, regenerate the manifest up to 3 times. On success, write `slideshow_manifest` and **clear** `hero_animation_plan`, `hero_bbox_coordinates`, `hero_icon_overlays`, `hero_callout_overlays`, `multivisual_animation_plan`.
2. Ensure those five columns exist.
3. Plan hero scenes and multi-visual scenes in parallel.
4. `run_hero_spatial_and_icon_phases_parallel`: bbox, icons, and callouts together.

Classification is by manifest template only:

| Template | What runs |
|---|---|
| `single_visual_hero` | hero animation decision, then bbox / icons / callouts if that type was chosen |
| `two_item_split_comparison`, `multi_panel_grid`, `main_plus_supporting_inset` | multi-visual decision (`none` or `text_label` only) |
| anything else | no overlay for that scene |

Hero animation types (exactly one): `none | text_label | bbox_highlight | icon_overlay | callout_card`. Default `none`. These are player-runtime overlays. They are not baked into the image.

Hero skips:

- Transition: all four hero columns set to `"-"`.
- Missing `primary_visual` URL: `ERROR: Missing primary_visual URL`.
- Video URL (YouTube, Drive video, or `.mp4` / `.webm` / `.mov`): scene omitted from the plan entirely.
- Existing valid `<animation_type>`: reused.

If a row has no remaining hero stills, the plan cell is `"-"`.

After the type is chosen:

- `bbox_highlight` → `hero_bbox_coordinates`. Gemini `box_2d` `[ymin, xmin, ymax, xmax]` on a 0–1000 grid. Up to 3 spatial review rounds.
- `icon_overlay` → `hero_icon_overlays`, max 3 icons per scene.
- `callout_card` → `hero_callout_overlays`, max 3 per scene.

Multi-visual: `none | text_label` only, stored in `multivisual_animation_plan`. If **any** slot is video, skip the LLM and force `none`. If any non-video panel needs a label, **every** non-video slot must get one (symmetry). Labels are semicolon-separated in slot order. The planner sees a composed preview from `make_agent_scene_preview_image`.

Delete calls both the hero and the multi-visual delete helpers.

---

## 6. Not active. Ignore these.

Files and routes below still exist. They are not in the live Streamlit pipeline, and a reviewer cannot start them. Do not describe them as product behavior.

Commented out of `pipeline_sections`, or imported and never registered:

- Scene edit planning, scene edit execution, apply-edited-URLs, and the Image Edit results sheet
- Batch human-feedback revise (`run_human_feedback_review_revise_for_all_rows`)
- Original/Revised visual columns (`run_populate_human_feedback_visual_columns`)
- Download assets into a Drive folder (`run_download_assets_for_sheet`)
- `layout_agent.py`, which would write `layout_instructions` and `layout_evaluation`. That is not Section 1 layout planning.

Present in the review app's HTML, JS, and API, but the buttons that open them were removed:

- Segmentation feedback and revise
- Layout feedback and revise

"Hide Layout Panels" / "Show Layout Panels" is a different control. It shows or hides the scene composition view. That view is active. It does not revise the layout.

The review app's MP4 download is the player render job. It is not `run_download_assets_for_sheet`.

---

## 7. Human Feedback FastAPI app

### 7.1 What it is

A review loop over one Google Sheet. Run from the **repo root** so `agents/`, `services/`, and `graphics_definition_v2_slideshow.py` import:

```bash
python -m uvicorn human_feedback_app.backend.main:app --host 0.0.0.0 --port 8000 --reload
```

Open `http://localhost:8000`.

Env (same OAuth app as Streamlit):

- `OAUTH_CLIENT_ID`, `OAUTH_CLIENT_SECRET`
- `OAUTH_REDIRECT_URI_HUMAN_FEEDBACK` (falls back to `OAUTH_REDIRECT_URI_LIGHTNING`, then `OAUTH_REDIRECT_URI`, then `http://localhost:8000/auth/callback`)
- `HUMAN_FEEDBACK_LLM` — default `gemini_3_flash_thinking`

`REVISE_MAX_WORKERS = 2` in `config.py` is **unused**. The job queue is constructed with `max_workers=10`.

### 7.2 Page flow

| Path | Behavior |
|---|---|
| `GET /` | If query has both `code` and `state`, 307 to `/auth/callback`. Else session + loaded sheet → `/review`. Else session → `/setup`. Else `/login-page` |
| `GET /login-page` | `frontend/login.html` |
| `GET /setup` | `frontend/setup.html`. Requires session |
| `GET /review` | `frontend/Slide Review.dc.html`. Requires session. Cache-bust: if `?v` is not the file mtime, redirect to `/review?v={mtime}`. `Cache-Control: no-store` |
| `GET /health` | `{"status":"ok"}` |
| `/brand-assets` | `assets/` if it exists, else `frontend/brand` |
| `/static` | entire `frontend/` |

Setup posts `/api/session/load` with the sheet URL and `root_folder_id: ""`, worksheet default `Slide Chunks`, then navigates to `/review`.

### 7.3 Auth and sessions

Cookie `hf_session_id` (`httponly`, `samesite=lax`). No `secure`, no max-age. `SESSION_SECRET` exists in config and is never used. Sessions live in process memory (`SessionStore`). A restart logs everyone out.

`GET /auth/login` → Google. `GET /auth/callback`:

- Cookie session must exist and `state` must match `oauth_state`.
- Drive + Sheets clients from the code exchange.
- Email from Drive `GetAbout`.
- `get_user_info(email)` must say `is_authorized`.
- `pages` must include `aggregation_agent_page` (granted by several roles in `config/user_roles.py`). Otherwise 403 and the session is deleted.

`UserSession` fields that matter: `user_email`, `role`, `pages`, `drive`, `gc`, `oauth_credentials`, `sheet_link`, `worksheet_name` (default `Slide Chunks`), `root_folder_id`, `course_name`, `current_round` (always 0), `enabled_sources`, `drive_video_mode`, `sheet`, `manifest_sync_status`, `manifest_sync_triggered`, `manifest_repair_cache`, `aborted`.

`enabled_sources` and `drive_video_mode` are filled in `load_workbook` from Course info, not at login. Blank cell → the product defaults in §2.3.

Logout (`POST` or `GET /auth/logout`):

1. Cancel every pending/running job for the session (`Cancelled on logout`).
2. Set `session.aborted = True`, which makes sheet writers raise `JobCancelled` unless `allow_when_aborted=True`.
3. For each cancelled **visual** job, clear that visual's reject action (allowed even though aborted).
4. Delete the session and the cookie. Redirect to `/login-page`.

### 7.4 Why `streamlit_shim.py` exists

Graphics agents read `st.session_state` (`drive`, `gc`, `user_email`, `role`, `sheet_link`, `root_folder_id`). FastAPI has no Streamlit script context. `main.py` imports the shim first. It patches `st.session_state` with a thread-local dict when no Streamlit context exists. Workers call `bind_user_session` + `agent_session_context` around agent calls.

### 7.5 Sheet write model

`save_to_sheet` writes the **entire** worksheet. Every mutation goes through `mutate_row_cells` under one global `_SHEET_WRITE_LOCK`: reload the latest dataframe, patch only the cells this operation owns, save. Parallel revisions of two visuals on the same slide are safe only because of this. Do not write a dataframe from a worker outside the lock.

### 7.6 API

All `/api/*` routes except `/api/player-theme` and `/api/styles-schema` require a session. 401 if the cookie is missing or `user_email` is empty.

| Method | Path | What it does |
|---|---|---|
| GET | `/api/me` | email, role, sheetLink, worksheetName |
| POST | `/api/session/load` | Body: `sheet_link`, optional `worksheet_name`, `root_folder_id`. Sheet must be inside the Skillcat Shared Drive. Tab name match is case-insensitive. On success: reconcile stale revisions, return the slides payload, kick the manifest checker. On failure: clear `sheet_link` |
| GET | `/api/slides` | Reconcile, then the same payload |
| GET | `/api/manifest-sync` | Start the checker if it has not run; return status |
| POST | `/api/visuals/approve` | Mark one visual approved |
| POST | `/api/visuals/revise` | Queue a revision. `mode` must be `drive`, `all`, or `ai` |
| POST | `/api/visuals/select-pool` | Assign a pool alternative URL now |
| POST | `/api/visuals/replace` | Assign the first URL found in `feedback` now. Returns the new payload |
| POST | `/api/visuals/revert` | Restore the original asset |
| POST | `/api/scenes/player-overrides` | Save animation/style overrides for one scene |
| GET | `/api/jobs` | All jobs for this session, newest first |
| GET | `/api/jobs/{id}` | One job. 404 if it belongs to another session |
| GET | `/api/assets/image?url=&thumb=` | Proxied image. `thumb=1` or `true` requests a thumbnail |
| GET | `/api/assets/video?url=` | Proxied clip. Honors `Range` |
| GET | `/api/tts?voiceover=` | MP3. Optional `X-TTS-Words` header |
| GET | `/api/tts/words?voiceover=` | `{words: [...]}` |
| GET | `/api/player-theme` | CSS variables from `player_styles.yaml`. No auth |
| GET | `/api/styles-schema` | Style schema. No auth |
| POST | `/api/download-slide-videos` | Start an MP4 render job |
| GET | `/api/download-slide-videos/{id}` | Render status |
| GET | `/api/download-slide-videos/{id}/file` | The file. 409 if not completed |

Load rejects a sheet that is not in the Skillcat Shared Drive, a bad URL, or a missing tab. The error names the tab the user asked for.

### 7.7 Slides payload (what the UI renders)

`slides_to_ui_payload` returns:

```text
slides[]
currentRound          always 0
moduleLabel / courseName
topicName             first slide's topic
sheetLink, worksheetName
```

Each slide: `title`, `topic` (blank topic cells are forward-filled), `segments`, `scenes`, `slideType`, `slideChunk`.

Each segment is `{steps: [...]}` and is omitted when it has no steps. Steps come from parsing `final_graphics_definition`.

Each step:

| Field | Meaning |
|---|---|
| `visualId` | `S{segmentIndex}V{stepIndex}` |
| `voiceover`, `instruction`, `justification` | from the definition |
| `type` | `video` if `detect_asset_type` says video, else `image` |
| `assetUrl` | current definition URL |
| `beforeUrl` | tracking `original`, else current asset |
| `afterUrl` | first of `after_revision`, `after_regen_1`, `manually_selected` |
| `source` | resolved from the display URL (`afterUrl` or `assetUrl`). See below |
| `status` | `pending` / `revising` / `revised` / `approved` |
| `mode` | `drive` / `all` / `ai` if a reject action is set, else `""` |
| `feedback` | from the actions JSON |
| `alternatives` | pool candidates for this segment, excluding the current URL |
| `rowIndex`, `segmentIndex`, `stepIndex` | ints. `rowIndex` is the dataframe index used by every write |

Scenes are attached only when the manifest slot count equals the flat step count and both are > 0. Otherwise `scenes` is `[]` and the UI falls back to a flat visual list.

Scene fields the review and player use: `id`, `template`, `templateLabel`, `narration`, `slotCount`, `visualIds`, `animationType`, `labelText`, `triggerPhrase`, `bboxHighlights`, `iconOverlays`, `calloutCards`, `playerOverrides`.

Slot-to-step mapping: match by URL first (`afterUrl` or `assetUrl`). If any slot misses, fall back to position.

**Status** (`_status_for_visual`):

| Condition | Status |
|---|---|
| action `approve` | `approved` |
| tracking has `after_revision` or `after_regen_1` or `manually_selected`, and action is not approve | `revised` |
| action is `reject_drive_hvac`, `reject_all`, or `reject_ai` | `revising` |
| else (`unreviewed` or missing) | `pending` |

A successful revise does **not** write `approve`. It leaves the reject action in place. The chip shows `revised` because tracking now has an after-revision URL.

**Source labels:**

- Drive host + video → `drive_video`, else `drive`
- YouTube + video → `hvac_yt`, else `other_yt`
- Other http → `web`
- Override to `external_ref_video` if the video is `frames_only`, or its Drive id appears in a `FramesOnly: yes` line in `video_pool_filtered`, or the URL/id is in the external-ref key set (from `external_ref_pool` columns plus the Course info extraction log)
- Override to `external_ref_image` for external-ref images

**Alternatives** for a segment, from this row only: `image_pool` URLs, then `video_pool_filtered` items (title and duration kept). Deduped. Current asset excluded. A Drive id that is frames-only is typed as video.

### 7.8 Actions, feedback text, and tracking

Stored in `human_review_actions_1` as JSON:

```json
{
  "actions": {
    "S1V1": {"action": "reject_drive_hvac", "feedback": "...", "vo": "...", "segment": 1}
  },
  "segment_modes": {"1": "drive_hvac"},
  "round": 0
```

| UI mode | `action` | Feedback cell |
|---|---|---|
| approve | `approve` | not merged |
| drive ("search the asset library") | `reject_drive_hvac` | block appended to `human_feedback_1`. Empty feedback becomes the default "I did not like the visual…" sentence |
| all ("find a better visual from the web") | `reject_all` | same |
| ai ("generate with AI") | `reject_ai` | empty feedback stored as the marker `No Feedback`, which the agent treats as actionable |
| pool select or paste-URL replace | `approve` after the URL is written | feedback cleared |
| revert | action key **deleted** | tracking reset; definition restored to `original` |

`segment_modes[segment]`: `all` if this visual's mode is `all` or `ai`, else `drive_hvac`. Once a segment is `all`, it stays `all`.

Feedback cell blocks:

```
When VO: {vo}

Human Feedback: {feedback}
```

Before a per-visual revise, `filter_feedback_to_vo` keeps only the block for that voiceover so sibling visuals on the row are not revised again.

Tracking cell `human_feedback_revision_tracking_1`:

```
S1V1

Original visual - {url}
Manual Selection - {url}          # omitted if empty
After revision - {url|No replacement}
After regeneration loop 1 - {url|No replacement}
After regeneration loop 2 - {url|No replacement}
```

Display URL, newest first: `after_regen_2`, `after_regen_1`, `after_revision`, `manually_selected`, `original`.

`human_feedback_status_1` is written (`PASS` / `FAIL` / `ERROR: …`) but the UI chips do not read it.

### 7.9 Visual operations, in order

**Approve.** `approve_visual` → `write_visual_action(approve)`. Does not touch the definition, tracking, or manifest.

**Revise.** Order in `api_revise` is deliberate:

1. `revision_queue.register` **before** any sheet write, so a concurrent `/slides` reconcile cannot see a reject with no job and delete it.
2. `prepare_visual_revision` writes the reject action and the feedback block.
3. Worker calls `run_row_revision`.
4. On start failure or worker failure: `clear_visual_revision_action` (only if the action is still one of the three rejects). Logout sets `aborted` and the worker must not clear after that; logout already cleared it.

`run_row_revision` loads the workbook, narrows feedback to the target voiceover, binds a fake Streamlit session, and calls `process_human_feedback_row` with `ws=None` (mutate the dataframe in memory, write nothing yet) and `use_only_drive_and_hvac=False`.

That boolean is **not** the UI mode. Modes come from the actions JSON:

| Action | What the agent does |
|---|---|
| `reject_ai` | Classify EDIT vs SCRATCH, generate or edit an image, upload to Drive folder `1c3rmYhF8kCrJVv3ui1-362mr90OCkEqB`. If the row's only actions are AI, skip the satisfaction review and the regen loop |
| `reject_drive_hvac` | `revise_segment_with_human_feedback` with `candidate_mode="drive_hvac"`: Course info sources **minus** web and other-YouTube. External refs stay if they were enabled |
| `reject_all` | `candidate_mode="all"`: force-include web images and other-YouTube. **Drop** external refs |
| Satisfaction FAIL | Up to `MAX_HUMAN_FEEDBACK_REGEN_ATTEMPTS = 2`. Segments still in `drive_hvac` regen with web off. Segments in `all` regen on the web path and may create `web_results` / `video_pool_other_channels` if missing |

`_enabled_sources_for_hf_candidate_mode` is the exact filter. If `enabled_sources` is None, the base list is the three primary defaults plus web and other-YouTube.

Then the worker calls `merge_visual_revision`, which under the write lock:

1. Drops this row from `manifest_repair_cache`.
2. Patches only this visual's URL in `final_graphics_definition`.
3. Merges only this visual's tracking entry.
4. Patches `slideshow_manifest` at the Nth occurrence of the old URL among definition matches, so two slots sharing one Drive file do not both change.
5. Writes `human_feedback_status_1` (last writer wins).
6. `ensure_manifest_sync_for_row`.
7. Overlay: full rebuild if the sync regenerated the manifest; otherwise a surgical refresh of scenes that use the new URL.

If the worker was called without visual ids (it is not, from the API), it falls back to saving every changed cell except the feedback cell.

**Select pool / paste URL.** Both write the new URL into the definition, set tracking `manually_selected`, set action `approve`, patch the manifest, sync, and refresh overlays. Paste URL takes the first `http(s)` or `www.` in the feedback text. Drive URLs are normalized to `https://drive.google.com/file/d/{id}/view`. These are synchronous. They are not jobs.

**Revert.** Delete the action, restore definition and manifest to tracking `original`, null the later tracking stages, sync, refresh overlays if the URL changed.

**Reconcile** (`reconcile_stale_visual_revisions`), on sheet load and on every `GET /slides`: a reject action with no after-revision / regen / manual URL, and no active visual job, is cleared. That is how a crashed process does not leave chips stuck on "revising".

### 7.10 Job queue

`RevisionJobQueue` in `jobs.py`. Statuses: `pending | running | completed | failed | cancelled`. The live kind is `visual`. Ten workers. No per-session cap.

`result` on success is a full `slides_to_ui_payload`. If the job was cancelled before the worker returned, the result is discarded and the job stays `cancelled`.

The UI polls `GET /api/jobs/{id}` about every 2.5s. `GET /api/jobs` exists; the review page does not use it for the poll loop.

### 7.11 Manifest sync

Two triggers:

- Once per loaded session, background: `maybe_trigger_manifest_sync_checker` from load, from the slides payload, and from `GET /api/manifest-sync`.
- Inline, under the write lock, after a visual revise, pool pick, paste URL, or revert: `ensure_manifest_sync_for_row`. Approve does not change the URL, so it does not sync.

Mismatch (`_validate_manifest_sync`): no definition URLs → synced. Empty or unparsable manifest → not synced. Slot count must equal definition URL count. Every slot URL must match a remaining definition URL. Leftover definition URLs fail the check.

Background repair: check all rows (up to 50 workers), then regenerate mismatches with `generate_slideshow_manifest_for_row`, passing **empty** `layout_plan` and `storyboard_planning`, up to 3 attempts. Persist the manifest, cache it on the session, full overlay rebuild.

`review_ready` becomes true when the first 10 rows that needed repair are resolved (repaired or failed). `player_ready` becomes true when every such row is resolved. The review UI waits on `review_ready`. The player waits on `player_ready`.

If the checker throws, status is `error` and **both** ready flags are forced true so the UI is not stuck.

`_manifest_for_ui_row` prefers `manifest_repair_cache` over the sheet cell. Any user write invalidates that row's cache.

Status fields: `status` (`idle|running|done|error`), `checked_rows`, `repaired_rows`, `failed_rows`, index lists, `repair_version`, `check_complete`, `review_ready`, `player_ready`, `prefix_row_count`, `rows_needing_repair_indices`, `total_rows`.

### 7.12 Overlay refresh after a human edit

`overlay_refresh.py`. Does not create overlay columns. If they are absent, it no-ops.

| Function | When | What |
|---|---|---|
| `refresh_overlays_after_visual_change` | URL patch, manifest was not regenerated | Replace the `---Scene ID: X---` block only for scenes whose slots use the new URL |
| `rebuild_overlays_for_row` | Manifest sync regenerated the manifest after a visual URL change | Wipe the row's overlay cells and replan every scene |

Same agents as Section 14: hero decision, bbox, icons, callouts, multi-visual labels. LLM is `HUMAN_FEEDBACK_LLM`. A per-row lock stops two overlay jobs on the same slide from interleaving. Failures are logged and do not roll back the visual save. Aborted sessions skip the work.

### 7.13 Player overrides

Column `player_scene_overrides` (no `_1`). JSON map `{sceneId: {animationEnabled, animationTreatment, styles}}`.

`save_scene_player_override` cleans against `player_styles.yaml` via `player_overrides.py`: `animationEnabled` only if bool; `animationTreatment` only if it is a known option; style keys must be schema properties; values stripped, max 500 chars. An empty cleaned object deletes that scene's key. An empty map writes `""`.

### 7.14 Assets, TTS, and MP4 download

**Images** (`asset_service.py`): downloaded with the user's Drive client. Drive video URLs are forced to a thumbnail. Cache key `(email, url, thumb)`, cap 1000. `needs_image_proxy` is true for `drive.google.com`, `docs.google.com`, `googleusercontent.com`.

**Videos:** only Drive or YouTube, else 400.

- Drive: download, optional ffmpeg trim from `(start=&end=)`. Cache by file id + range, cap 200.
- YouTube: clip bounded to 90 seconds. Download semaphore 2. Cookies from `YOUTUBE_COOKIES_PATH` or a temp file, refreshed from a Drive folder when older than `YOUTUBE_COOKIES_MAX_AGE_SECONDS` (default 4 hours). Bot-check refreshes cookies once and retries.

Range requests return 206.

**TTS** (`tts_service.py`): voice is hardcoded `en-CA-LiamNeural`. `/api/tts` does not accept a voice parameter. edge-tts word boundaries become `{text, offset, duration}` in seconds. Cache cap 500. `X-TTS-Words` is base64 JSON, omitted if the encoded header would exceed 7000 characters.

**Download slide videos** (`player_video_jobs.py`): Ken Burns MP4 from the player. Statuses `pending | running | completed | failed`. No cancel. Two render workers.

`quality` is accepted by the API and sent by the UI as `fast`, and then **ignored**. Encode settings come from `slideshow_video` constants.

`hero_motion`: `zoom_in | drift_x | diagonal | pull_out`; anything else becomes `off`. Applied only to a `single_visual_hero` scene with one image slot. Direction alternates by slide index + scene index.

Rows with an empty or `ERROR:` manifest are skipped. Assets are pre-downloaded. Multiple slides are stitched with `render_course_video`. The download-video voice default is `en-US-AvaNeural`, which is **not** the player TTS voice.

### 7.15 Frontend files

| File | Role |
|---|---|
| `login.html` | Link to `/auth/login` |
| `setup.html` | Sheet URL form |
| `Slide Review.dc.html` | The review and player UI |
| `review-api.js` | `window.HFApi` |
| `player.js` | `window.HFPlayer`: cues from `slide.scenes`, TTS prefetch, overlays, style overrides. Drive and YouTube play through `/api/assets/video`, not an iframe |
| `load-config.js` | Theme + style schema |
| `youtube-player.js` | Embed helper for review tiles |
| `animations/*.js` | GSAP treatments: hero stills, split comparison, multi-panel grid, main+inset, treatment rotator, overlay timing, scene transitions |
| `player_styles.yaml`, `player_config.py`, `player_config_loader.py` | Theme and the override schema |

`review-api.js` calls that the page actually uses: `/api/slides`, approve, select-pool, revise, replace, player-overrides, revert, `/api/jobs/{id}`, `/api/manifest-sync`, and `/api/tts?voiceover=`.

The HTML also calls download-slide-videos, styles-schema, `/api/assets/image`, `/api/assets/video`, and `/api/tts/words`.

Approve and revise buttons are offered for `pending` and `revised`. Pool select uses `step.alternatives`. Paste-URL replace requires an http(s) or www URL in the text and returns immediately. Revert goes back to `pending`.

The review page shows each scene's template and panels. "Hide Layout Panels" / "Show Layout Panels" only toggles that view. There is no control that submits segmentation feedback or layout feedback.

---

## 8. End-to-end data flow

```text
Course info
  Course Name, Target Audience, External References, Allowed Asset Search Libraries
        │
        ▼
Slide Chunks row
  Topic, Slide Type, Slide Chunk Title, Slide Chunk
        │
        │  pre-exec: maybe insert Transition row, fill Visual Assignment Strategy
        ▼
layout_plan                          (scenes + layout category, no assets yet)
        ▼
voiceover_segment                    (one sentence per line; inline [alt](url) → reference_image_map)
        ▼
storyboard_planning                  (When VO + Visual Idea; locked if a reference map exists)
        ▼
search_queries                       (---SEGMENT_n--- + N queries)
        ▼
reference_image_map                  (fill gaps from the course Drive image vector store)
        ▼
candidate columns                    (drive_results, video_pool, drive_video_pool, external_ref_pool)
        │                            web_results and video_pool_other_channels stay empty here
        ▼
image_score / video_score            (multimodal, batches of 3–5, shortlist top 2 at ≥ 4/10)
        ▼
image_pool / video_pool_filtered
        ▼
final_graphics_definition            (When VO + Graphics to use; VAS controls how many)
        ▼
review_complete + revision_tracking  (alignment only; regen cannot use web)
        ▼
final_visuals_review                (only if Web checkbox is on)
  + Replacement visual URLs          (web + other YouTube only)
        ▼
decision_of_final_visual             (A vs B merged back into final_graphics_definition)
        ▼
slideshow_manifest                   (1:1 URL → slot, one template per scene)
        ▼
hero_* and multivisual_animation_plan
        │
        │  Human Feedback app loads the same row
        ▼
human_review_actions_1 + human_feedback_1
        │
        ├─ approve / pool pick / paste URL / revert     (sync sheet writes)
        └─ revise drive|all|ai                          (process_human_feedback_row, one visual)
        │
        ▼
manifest sync (repair if URLs diverged) + overlay refresh or full rebuild
        ▼
player (TTS en-CA-LiamNeural, GSAP overlays, proxied Drive/YouTube)
```

---

## 9. Facts that are easy to get wrong

1. Narration lives in `Slide Chunk` and `voiceover_segment`. There is no `voiceover` column.
2. The asset line in `final_graphics_definition` is `Graphics to use:`, not `Assigned Asset:`.
3. Visual ids (`S1V1`) are computed at parse time. They are not a column.
4. Web and other-YouTube are Section 9 and human-feedback "search the web". They are not Section 5 runners, even when the checkbox is on.
5. Section 8 hardcodes `use_only_drive_and_hvac=True`, which turns web off **only during regeneration**. The first revise-from-pool pass can still see whatever is already in `web_results`.
6. Alignment is the only review criterion that runs. Specificity, redundancy, and segmentation quality are skipped or commented out. `review_complete=TRUE` means alignment passed (or was skipped) because segmentation is always `SKIPPED`.
7. `layout_plan_evaluation` and slideshow evaluation are not saved.
8. Layout planning does not read Visual Assignment Strategy. Segmentation does not either.
9. A non-empty `reference_image_map` locks the storyboard to the existing voiceover lines and ignores VAS. Aggregation then forces that segment to one visual.
10. Transition slides skip HVAC, Drive-video, and other-channel search, and short-circuit review to `review_complete=TRUE`.
11. The whole-slide finalized-visuals prompt can mark every visual FAIL if the model omits the `<failures>` wrapper. That is a parser limitation, not a policy.
12. Human-feedback columns are `*_1`. `ROUND_INDEX` is 0 and is not incremented.
13. `use_only_drive_and_hvac=False` in the revise worker does not mean "search everything." The reject action and `segment_modes` do. `all` drops external refs. `drive` drops web.
14. Register the revision job before writing the reject action. Reconcile deletes reject actions that have no job and no replacement URL.
15. Successful revise leaves the reject action in the sheet. The UI shows `revised` from tracking, not from a status column.
16. Sheet saves rewrite the whole worksheet. All writers must go through `mutate_row_cells`.
17. Overlay columns are not created by the review app. If Section 14 never ran, overlay refresh no-ops.
18. Manifest repair regenerates with empty layout plan and storyboard, so a repaired manifest can differ from the Section 10 manifest.
19. Player TTS voice is `en-CA-LiamNeural`. Download-video voice defaults to `en-US-AvaNeural`. Download `quality` is ignored.
20. Sessions are RAM-only. Logout aborts jobs and clears in-flight reject actions.
21. `REVISE_MAX_WORKERS` in config is unused. The queue size is 10.
22. Segmentation feedback and layout feedback are not in the review UI. The routes and agents still exist. Ignore them. Section 6 lists the other dead paths.
23. "Hide Layout Panels" toggles the scene composition view. It does not revise layouts. Section 1 layout planning and Section 2 voiceover segmentation are pipeline steps. They are not the removed review-app feedback forms.
24. External reference images and the reference image pool are different systems. The first is Supabase `external_ref_assets` scoped by `sheet_id`. The second is the course Drive image vector store written into `reference_image_map`.

---

## 10. Where to edit what

| Change | Open |
|---|---|
| Add, remove, or reorder a Streamlit step | `graphics_definition_v2.py` `pipeline_sections` |
| Asset-library ids, defaults, Course info labels | `agents/graphics_definition_v2/candidate_search/pool_registry.py` |
| Checkbox behavior, hide rules, topic selector | `agent_ui_template.py` (`_render_graphics_v2_asset_library_controls`, `_step_is_hidden`) |
| Which column a source writes, and how a segment's candidates are merged | the runner in §5.5, then `candidate_search/candidate_wrapper.py` |
| How many visuals get assigned | VAS handling in `aggregation_agent.py` and the storyboard/search-query branches |
| Alignment loop limits or which criteria run | `review_and_revise.py` (`MAX_REVIEW_ATTEMPTS`, `process_review_revise_row` around the SKIPPED statuses) |
| Web fallback policy | `review_finalized_visuals.py` and `decide_final_visuals.py` |
| Scene templates | `slideshow_manifest.py` prompt + `parse_scenes` in `image_edit_planning.py` |
| Overlay types | `hero_animation_decision.py`, `multivisual_animation_decision.py`, then bbox/icon files |
| A review button or route | `human_feedback_app/backend/main.py` and `sheet_service.py` |
| What "search library" vs "search web" vs "AI" is allowed to retrieve | `_enabled_sources_for_hf_candidate_mode` and the `reject_ai` branch in `human_feedback_based_review_and_revise.py` |
| Chip status rules | `_status_for_visual` in `sheet_service.py` |
| Player look | `player_styles.yaml`, `frontend/player.js`, `frontend/animations/` |
