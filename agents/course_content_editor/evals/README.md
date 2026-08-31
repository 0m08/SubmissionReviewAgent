# Evals — editing-course-content skill

Three saved evaluation scenarios, per the Skills best-practices guide's
"Build evaluations first" section. Each is grounded in a real session already
run against the deployed Managed Agent (not synthetic) — the query text and
expected behavior are drawn directly from the two live quality-test rounds
run against `Course - Plumbing System Fundamentals` (real course sheet, real
reviewer-corrected ground truth in
`agents/slide_chunk_editor/feedback_sheets/Course - Plumbing System
Fundamentals.xlsx`).

The JSON schema here adapts the best-practices doc's `{skills, query, files,
expected_behavior}` shape to a Managed Agents session (no local `files` input
— the workspace comes from a Google Sheet URL named in the query instead).
There's no built-in runner for these; re-run them by hand (or script against
`client.beta.sessions`) the same way the two quality-test rounds were driven,
and check the transcript/resulting files against `expected_behavior`.

| File | What it exercises | Status as of `agent_version: 4` |
|---|---|---|
| `eval_research_notes_directive.json` | Single-row research-notes edit, reviewer-level specificity | **Passed** — near-exact match to ground truth, see quality-test round 1 |
| `eval_slide_chunks_directive.json` | Single-topic slide-chunk structural edit (merge/cut/reorder/hook), reviewer-level specificity | **Passed** — closest of 4 personas tested to ground truth, see quality-test round 2 |
| `eval_multi_topic_batch_regression.json` | One request sweeping multiple topics/rows in a single turn | **Failed** on `agent_version: 3/4` — confirmed independently on both the research-notes and slide-chunks rounds (a targeted row left virtually untouched despite the agent's own stated plan; one run also committed to the sheet without being asked). This is the regression case the new "Every turn" step 3 (re-read and confirm before reporting done) is meant to fix — re-run this one after any future skill change that touches multi-block editing to check whether it now passes. |
