# Moodle Submission Review & Mentor Feedback App

Dedicated FastAPI backend + modern interactive web UI for Human-in-the-Loop Moodle student submission grading, review, and overrides.

Built on the same architecture as `human_feedback_app`.

---

## Features

- **Activity Multi-Tab Organization**: Synchronizes dynamically with Central Google Sheet activities (e.g. `System Identification`).
- **Human Mentor Controls**:
  - Direct 1-click grade overrides (`Pass` / `Fail`).
  - Rich feedback comment editor to refine AI mentor notes before students see them.
  - Review status tracking (`Pending Review`, `Approved by Mentor`, `Overridden by Mentor`).
- **Evidence & Reason Inspection**:
  - One-click access to student's Google Drive media folder.
  - Formatted display of student's online text submission.
  - Full checklist breakdown with pass/fail badges and reasoning.
- **Batch Operations**:
  - `⚡ Batch Approve Confident Passes`: Instantly sign-off all confident pass submissions.
  - `📤 Export Moodle CSV`: Export gradebook-ready CSV with mentor-approved grades and comments.
  - `📥 Sync Local Folder`: Import submissions directly from any local Moodle run folder.
- **Deduplication & Attempt Tracking**: Preserves multiple attempts per student and increments attempt numbers automatically.

---

## Running the App

### Prerequisites
Run from the repository root:
```bash
pip install -r moodle_submission_feedback_app/requirements.txt
```

Environment variables (in `.env`):
- `GOOGLE_OAUTH_REFRESH_TOKEN`
- `GOOGLE_OAUTH_CLIENT_ID`
- `GOOGLE_OAUTH_CLIENT_SECRET`
- `MOODLE_DRIVE_FOLDER_ID` (Defaults to `1jR5JP3aTKt8u_X57Z31IS71ZGBX-zfeu`)
- `CENTRAL_REVIEW_SHEET_ID` (Defaults to `1aP7Xdvoi4TTIT8K7X9j2wD0G6-YCnA7sEn-BiLDWWQU`)

### Launch
```bash
python -m uvicorn moodle_submission_feedback_app.backend.main:app --host 0.0.0.0 --port 8000 --reload
```
Open **http://localhost:8000** in your browser.

---

## API Reference

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| GET | `/api/me` | Current mentor info & sheet links |
| GET | `/api/activities` | List all activity tabs in Central Registry |
| GET | `/api/activity/{name}` | Submissions & KPI metrics for an activity |
| POST | `/api/activity/{name}/review` | Save mentor review/override |
| POST | `/api/activity/{name}/batch-approve` | Batch approve confident pass submissions |
| POST | `/api/activity/{name}/sync-local` | Sync an ingested assignment run folder |
| GET | `/api/activity/{name}/export` | Download Moodle gradebook CSV |
