# Quality Compliance Scoring

Tools for scoring checklist compliance, generating LLM summaries, and producing course review reports.

## What this does
- Reads stage worksheets from a Google Sheet and computes basic/critical compliance scores per creator.
- Generates LLM-based compliance summaries and quality summaries.
- Writes results to the `Issue Log` and `Task Logs` sheets.
- Produces a formatted DOCX review report and uploads it to Google Drive.

## Files
- `quality_scoring.py`:
  - Orchestrates the end-to-end workflow: read sheets, compute scores, call LLM summarization, update logs, and create Drive reports.
  - Manages Google client initialization from Streamlit session state.
  - Parallelizes per-stage processing and formats Issue Log/Task Logs output.
- `generate_llm_feedback.py`:
  - Builds the LLM prompts and parses XML output for compliance and quality summaries.
  - Rephrases compliance checklist items into report-ready statements.
- `create_review_report_doc.py`:
  - Creates a DOCX “Course Review Report” with tables, headings, and hyperlinks.
  - Parses quality summary sections and builds comment anchors/links.
- `helpers.py`:
  - Shared utilities for parsing course names, stage sheets, filenames, and comment headers.

## Primary entry point
- `run_update_quality_scores(spreadsheet, course_folder_id, spreadsheet_url)` in `quality_scoring.py`.

## Key inputs
- Google Sheet with stage tabs that include `Checklist Criteria` and `Task` headers.
- Streamlit session state with authenticated Google clients.
- Reviewer columns labeled with `Topic N` and “Reviewer/Reporter/Comments” headers.

## Outputs
- Issue Log sheet populated with per-creator results.
- Task Logs sheet updated with latest scores and summaries.
- A Google Docs review report per creator (uploaded to Drive) based on a generated DOCX.

## Notes
- LLM output is expected to be strict XML inside `<output>` tags; parsing relies on that contract.
- Compliance summaries preserve task names exactly as provided in the checklist to align categories.
