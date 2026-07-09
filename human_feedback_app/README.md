# Human Feedback App

FastAPI backend + slide review UI, wired to the real Graphics Definition V2 sheet and revision agents.

## What it does

- Google OAuth login (same accounts as the Streamlit app)
- Load a course Google Sheet from the Skillcat Shared Drive
- Review slide visuals with approve / revise (Drive, Web, AI)
- Run revisions in a background job queue (non-blocking UI)
- Poll job status and refresh visuals when a revision completes

## Prerequisites

Run from the **repository root** so existing `agents/`, `services/`, and `graphics_definition_v2_slideshow.py` imports resolve.

Environment variables (same OAuth app as Streamlit):

- `OAUTH_CLIENT_ID`
- `OAUTH_CLIENT_SECRET`
- `OAUTH_REDIRECT_URI_HUMAN_FEEDBACK` — e.g. `https://<your-lightning-public-url>/auth/callback`

Optional:

- `HUMAN_FEEDBACK_LLM` — defaults to `gemini_3_flash_thinking`

Install repo dependencies (recommended) plus app extras:

```bash
pip install -r requirements.txt
pip install -r human_feedback_app/requirements.txt
```

## Local run

From the repo root:

```bash
python -m uvicorn human_feedback_app.backend.main:app --host 0.0.0.0 --port 8000 --reload
```

Open: http://localhost:8000

## Lightning AI Studio

1. Copy/sync the full repo (or at minimum: `human_feedback_app/`, `agents/`, `services/`, `utils/`, `config/`, `graphics_definition_v2_slideshow.py`, and dependencies)
2. Set OAuth env vars; set `OAUTH_REDIRECT_URI_HUMAN_FEEDBACK` to your Lightning public URL + `/auth/callback`
3. From repo root:

```bash
pip install -r requirements.txt
pip install -r human_feedback_app/requirements.txt
python -m uvicorn human_feedback_app.backend.main:app --host 0.0.0.0 --port 8000
```

4. Expose port **8000** in Lightning Studio (Ports panel)
5. Open the **Public** URL Lightning provides

## API

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| GET | `/auth/login` | Start Google OAuth |
| GET | `/auth/callback` | OAuth callback |
| GET/POST | `/auth/logout` | Log out |
| GET | `/api/me` | Current user + loaded sheet |
| POST | `/api/session/load` | Load sheet by URL |
| GET | `/api/slides` | Slides payload for review UI |
| POST | `/api/visuals/approve` | Approve one visual |
| POST | `/api/visuals/revise` | Queue revision job |
| GET | `/api/jobs/{id}` | Job status + result |

## Folder structure

```
human_feedback_app/
  requirements.txt
  backend/
    main.py
    auth.py
    jobs.py
    sheet_service.py
    revise_worker.py
    ...
  frontend/
    login.html
    setup.html
    Slide Review.dc.html
    review-api.js
    support.js
```
