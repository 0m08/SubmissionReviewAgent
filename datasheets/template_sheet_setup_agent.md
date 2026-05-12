# Template Sheet Setup Agent

> Bootstraps a new course in the Auto Course Gen pipeline. Creates the course's Google Sheet (the central pipeline artifact) and the AI checklist sheet (a tunable quality-control input read by every downstream agent), then validates that the user has filled in course background and base outline before handing off to the Course Outline agent.

---

## At a glance

| | |
| :--- | :--- |
| **Owner** | Dilip |
| **Status** | Done / stable |
| **Pipeline stage** | 0 — entry point |
| **Generative?** | No — fully deterministic (no LLM calls) |
| **Interface** | Streamlit page (sidebar → Agentic Workflows → *Template Sheet Setup Agent*) |
| **Typical runtime** | Seconds (two Drive copies + one Sheets write) |
| **Run frequency** | Once per new course |
| **Primary user** | Content team |
| **External APIs** | Google Drive (PyDrive2), Google Sheets (gspread) |
| **Requirements** | R4 (Workflow Improvement), R8 (Content & Curriculum) |

---

## Block diagram

```
                       ┌──────────────────────────────┐
   Drive Folder ID ───▶│                              │
                       │   Template Sheet Setup       │──▶ AI course: {name}      (Sheet)
   Course Name    ───▶│   Agent                      │──▶ AI checklist: {name}   (Sheet)
                       │                              │
                       │   ┌───────────────────────┐  │     • Checklist URL written
                       │   │ 1. Copy templates     │  │       into Course Sheet's
                       │   │ 2. Cross-link sheets  │  │       Course info tab
                       │   │ 3. (user edits sheet) │  │
                       │   │ 4. Validate           │  │
                       │   └───────────────────────┘  │
                       └──────────────────────────────┘
                                      │
                                      ▼
                          Course Outline Agent
                              (next stage)
```

---

## Inputs

### Provided up-front (in the UI)

| Field | Required | Format | Used for |
| :--- | :--- | :--- | :--- |
| Drive Folder ID | Yes | Google Drive folder ID | Destination for both Sheets |
| Course Name | Yes | Free text | Sheet naming (`AI course: {name}`, `AI checklist: {name}`) |

### Provided after the Sheets are created (edited in Google Sheets, then validated)

| Field | Required | Location | Used for |
| :--- | :--- | :--- | :--- |
| Course Name | Yes | Course Sheet → `Course info` tab | Downstream context for every agent |
| Course Background | Yes | Course Sheet → `Course info` tab | Topic framing for outline + research |
| Course Objective Guidelines | Yes | Course Sheet → `Course info` tab | Steering signal for outline + assessment agents |
| Base Outline | Yes | Course Sheet → `Base Outline` tab | Seed structure for the Course Outline agent |
| Checklist content | Optional | Checklist Sheet | Per-course tuning of downstream review-and-revise loops |

> **Note on modes.** Mode selection (No-Sources / Internal-Sources / External-Sources) is not exposed at this stage. The same template is used for all three modes; mode-specific behavior is configured downstream.

---

## Outputs

| Artifact | Description |
| :--- | :--- |
| **`AI course: {name}`** (Google Sheet) | The central pipeline artifact. Pre-populated with every tab downstream agents need: `Course info`, `Base Outline`. Every later agent reads from and writes to this Sheet. |
| **`AI checklist: {name}`** (Google Sheet) | Tunable checklist used by downstream agents' **internal review-and-revise** passes. Content team can edit per course to steer agent behavior (e.g., tighten outline-agent verification, adjust slide-chunk quality bar) without code changes. |
| **Cross-link** | Checklist URL is written into the Course Sheet's `Course info` tab so downstream agents can locate the checklist from the Course Sheet alone. |
| **Validated-ready state** | Setup is only marked complete when the agent confirms every required `Course info` field has been changed from its template default and the `Base Outline` tab has been populated. |

---

## Functional description

The agent runs in two phases separated by a manual edit step.

**Phase A — Create (automatic, on click "Setup Template Sheets")**

1. List spreadsheets already in the target Drive folder; reuse any matching `AI course: {name}` / `AI checklist: {name}` if present (idempotent).
2. Otherwise, copy the master Course Sheet template and master Checklist Sheet template into the folder, named accordingly.
3. Write the Checklist Sheet URL into the Course Sheet's `Course info` tab (`Checklist Link` column).
4. Return both Sheet URLs to the UI.

**Phase B — Validate (automatic, on click "Validate Changes")**

5. Re-read the Course Sheet's `Course info` and `Base Outline` tabs.
6. Compare `Course Name`, `Course Background`, `Course Objective Guidelines` against the master template's defaults — any field that is empty or unchanged is flagged.
7. Compare the `Base Outline` tab against the master template — if it's unchanged or effectively empty, flag it.
8. If any flags, return the list of unchanged fields; otherwise mark setup complete.

The agent does no generative work. It is templating + cross-linking + diff-based validation.

---

## Operating procedure

1. Authenticate with Google Drive + Sheets through the main Streamlit app.
2. Open *Template Sheet Setup Agent* in the sidebar.
3. Enter the Drive Folder ID and Course Name. Click **Setup Template Sheets**.
4. Open the Course Sheet via the surfaced link. Fill in `Course info` fields and populate `Base Outline`.
5. *(Optional)* Edit the Checklist Sheet if you want to tune downstream agent behavior for this course.
6. Return to the agent page. Click **Validate Changes**.
7. If any required fields are flagged as unchanged, edit them and re-validate. Otherwise, setup is complete and the Course Sheet is ready for the Course Outline agent.

---

## Role in the pipeline

| Stage | Agent | Reads from | Writes to |
| :--- | :--- | :--- | :--- |
| 0 | **Template Sheet Setup** *(this agent)* | — | Course Sheet, Checklist Sheet |
| 1 | Course Outline | Course Sheet (`Course info`, `Base Outline`), Checklist Sheet | Course Sheet (Final Outline tab) |
| 2 | Research Notes | Course Sheet, Checklist Sheet | Course Sheet (Final Outline tab - research_notes column) |
| 3 | Slide Chunks | Course Sheet, Checklist Sheet | Course Sheet (Slide Chunks tab) |
| 4 | Graphics | Course Sheet, Checklist Sheet | Course Sheet (Slide Chunks tab - final_graphics_definition column) |
| 5 | Assessment | Course Sheet, Checklist Sheet | Course Sheet (Assessment tab) |

Every downstream stage operates on the artifacts this agent creates.

---

## Impact

- **Single source of truth.** Guarantees every course in the pipeline has the same Sheet structure, eliminating per-course schema drift.
- **Tunable quality control.** The Checklist Sheet gives the content team a per-course knob to steer agent output without code changes.
- **Friction-free entry.** Two inputs and a guided manual-edit step are all that stand between a content-team member and an in-flight course.

---

## Dependencies

- Google Drive API (PyDrive2) and Google Sheets API (gspread), via the main app's authenticated session.
- Master template Sheets, referenced by URL via env vars `TEMPLATE_COURSE_SHEET_LINK` and `TEMPLATE_CHECKLIST_SHEET_LINK`.
- No blocking dependencies. Agent is shipped and stable.

---

## Source

| File | Role |
| :--- | :--- |
| `template_sheet_agent.py` | Streamlit page — UI, 2-step flow, validation |
| `agents/template_sheet_setup/setup_template_sheets.py` | Backend templating logic |
| `services/drive_service.py` | `copy_sheet_from_link` helper |
| `services/sheets_service.py` | `get_sheet_data_and_df`, `save_to_sheet` helpers |
