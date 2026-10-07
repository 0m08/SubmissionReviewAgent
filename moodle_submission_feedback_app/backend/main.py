# -*- coding: utf-8 -*-
"""FastAPI application for Moodle Submission Review & Mentor Feedback."""

from __future__ import annotations

import logging
import asyncio
import os
import threading
import time
from typing import Any, Dict, Optional
from urllib.parse import unquote

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from moodle_submission_feedback_app.backend.auth import (
    get_current_session,
    get_optional_session,
    router as auth_router,
)
from moodle_submission_feedback_app.backend.config import (
    BRAND_ASSETS_DIR,
    FRONTEND_DIR,
    REGISTRY_SHEET_ID,
)
from moodle_submission_feedback_app.backend.schemas import (
    ActivityRuleRequest,
    BatchApproveRequest,
    SyncLocalRequest,
    UpdateReviewRequest,
)
from moodle_submission_feedback_app.backend.registry_service import (
    add_activity_rule,
    batch_approve_activity,
    export_moodle_gradebook_csv,
    get_activities_list,
    get_activities_overview,
    get_activity_submissions,
    sync_local_folder,
    update_submission_review,
)
from moodle_submission_feedback_app.backend.media_service import (
    fetch_submission_media_list,
    get_image_bytes,
)
from moodle_submission_feedback_app.backend.sessions import MentorSession

logger = logging.getLogger("moodle_feedback_app")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")

app = FastAPI(
    title="Moodle Submission Review & Mentor Feedback API",
    description="Dedicated Human-in-the-Loop review and override backend for student submissions.",
    version="1.0.0",
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Auth router (/auth/login, /auth/callback, /auth/logout)
app.include_router(auth_router)

# Protected API router
api_router = APIRouter(prefix="/api", tags=["registry"], dependencies=[Depends(get_current_session)])


@app.get("/health")
def health_check():
    return {"status": "ok", "app": "moodle_submission_feedback_app"}


@app.get("/login-page")
def login_page():
    return FileResponse(FRONTEND_DIR / "login.html")


@app.get("/")
def index_page(request: Request):
    if "code" in request.query_params and "state" in request.query_params:
        return RedirectResponse(f"/auth/callback?{request.url.query}", status_code=307)
    session = get_optional_session(request)
    if not session:
        return RedirectResponse("/login-page", status_code=302)
    return FileResponse(FRONTEND_DIR / "index.html")


@api_router.get("/me")
def get_current_mentor(session: MentorSession = Depends(get_current_session)):
    """Returns active mentor information and sheet configuration."""
    return {
        "mentor_name": session.mentor_name or "Om Aryan",
        "email": session.user_email or "om@skillcatapp.com",
        "role": session.role or "Mentor",
        "initials": session.avatar_initials or "OA",
        "sheet_id": REGISTRY_SHEET_ID,
        "sheet_url": f"https://docs.google.com/spreadsheets/d/{REGISTRY_SHEET_ID}/edit",
    }


@api_router.get("/activities")
def list_activities():
    """Returns all available activity tabs in the registry sheet with full metric summaries."""
    try:
        overview = get_activities_overview()
        return overview
    except Exception as e:
        logger.error(f"Failed to fetch activities: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/activity/{activity_name}")
def get_activity_details(activity_name: str):
    """Returns submissions and summary KPIs for an activity tab."""
    try:
        clean_name = unquote(activity_name)
        data = get_activity_submissions(clean_name)
        return data
    except Exception as e:
        logger.error(f"Failed to load activity '{activity_name}': {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.post("/activity/{activity_name}/review")
def update_review(
    activity_name: str,
    payload: UpdateReviewRequest,
    session: MentorSession = Depends(get_current_session),
):
    """Saves a mentor's review or override for a student submission."""
    try:
        clean_name = unquote(activity_name)
        mentor = payload.mentor_name or session.mentor_name or "Om Aryan"
        result = update_submission_review(
            activity_name=clean_name,
            student_name=payload.student_name,
            attempt_number=payload.attempt_number,
            new_grade=payload.grade,
            new_feedback=payload.feedback_comment,
            review_status=payload.review_status,
            mentor_name=mentor,
            edge_case=payload.edge_case,
            guideline=payload.guideline,
        )
        return result
    except Exception as e:
        logger.error(f"Failed to update review for '{payload.student_name}': {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.post("/activity/{activity_name}/rule")
def add_activity_rule_endpoint(
    activity_name: str,
    payload: ActivityRuleRequest,
    session: MentorSession = Depends(get_current_session),
):
    """Adds an edge case or guideline for the activity to the 'Activity Details' sheet."""
    try:
        clean_name = unquote(activity_name)
        result = add_activity_rule(
            activity_name=clean_name,
            edge_case=payload.edge_case,
            guideline=payload.guideline,
        )
        return result
    except Exception as e:
        logger.error(f"Failed to add activity rule for '{activity_name}': {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.post("/activity/{activity_name}/batch-approve")
def batch_approve(
    activity_name: str,
    payload: BatchApproveRequest,
    session: MentorSession = Depends(get_current_session),
):
    """Batch approves all pending confident pass submissions."""
    try:
        clean_name = unquote(activity_name)
        mentor = payload.mentor_name or session.mentor_name or "Om Aryan"
        result = batch_approve_activity(clean_name, mentor_name=mentor)
        return result
    except Exception as e:
        logger.error(f"Failed to batch approve activity '{activity_name}': {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.post("/activity/{activity_name}/sync-local")
def sync_local(activity_name: str, payload: SyncLocalRequest):
    """Synchronizes a local Moodle assignment run folder into the sheet."""
    try:
        clean_name = unquote(activity_name or payload.activity_name)
        result = sync_local_folder(payload.assignment_folder, activity_name=clean_name)
        return result
    except Exception as e:
        logger.error(f"Failed to sync local folder: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/activity/{activity_name}/export")
def export_csv(activity_name: str):
    """Exports Moodle-compatible CSV with approved grades and feedback."""
    try:
        clean_name = unquote(activity_name)
        csv_content = export_moodle_gradebook_csv(clean_name)
        filename = f"moodle_grades_{clean_name.lower().replace(' ', '_')}.csv"
        return Response(
            content=csv_content,
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )
    except Exception as e:
        logger.error(f"Failed to export CSV for '{activity_name}': {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Background Job State Tracking for Moodle Pipelines
pipeline_jobs: Dict[str, Dict[str, Any]] = {
    "pull": {
        "status": "idle",
        "message": "Ready to pull submissions.",
        "activity_name": "",
        "started_at": 0,
        "finished_at": 0,
        "error": None,
    },
    "push": {
        "status": "idle",
        "message": "Ready to push grades.",
        "activity_name": "",
        "started_at": 0,
        "finished_at": 0,
        "error": None,
    },
}


def _run_pull_job(activity_name: str, mentor_email: str):
    logger.info(f"🚀 [Action 1 Backend] Starting Moodle Ingestion & AI Review for '{activity_name}' (mentor: {mentor_email})...")
    pipeline_jobs["pull"]["status"] = "running"
    pipeline_jobs["pull"]["activity_name"] = activity_name
    pipeline_jobs["pull"]["message"] = f"Pulling submissions from Moodle & evaluating with AI for '{activity_name}'..."
    pipeline_jobs["pull"]["started_at"] = time.time()
    pipeline_jobs["pull"]["finished_at"] = 0
    pipeline_jobs["pull"]["error"] = None

    try:
        from agents.submission_reviewer.moodle_review_runner import run_full_ingest_and_review
        result = run_full_ingest_and_review(
            activity_name=activity_name,
            upload_to_moodle=False,
        )
        evaluated_count = result.get("evaluated_count", 0)
        pipeline_jobs["pull"]["status"] = "completed"
        pipeline_jobs["pull"]["finished_at"] = time.time()
        pipeline_jobs["pull"]["message"] = f"Done! Evaluated and synced {evaluated_count} submission(s) to Google Sheet tab '{activity_name}'."
        logger.info(f"✅ [Action 1 Backend] Finished successfully for '{activity_name}' ({evaluated_count} evaluated).")
    except Exception as e:
        logger.error(f"❌ [Action 1 Backend] Failed for '{activity_name}': {e}", exc_info=True)
        pipeline_jobs["pull"]["status"] = "failed"
        pipeline_jobs["pull"]["finished_at"] = time.time()
        pipeline_jobs["pull"]["error"] = str(e)
        pipeline_jobs["pull"]["message"] = f"Ingestion error: {str(e)}"


def _run_push_job(activity_name: str, only_reviewed: bool, mentor_email: str):
    logger.info(f"🚀 [Action 2 Backend] Starting Moodle Gradebook Upload for '{activity_name}' (mentor: {mentor_email})...")
    pipeline_jobs["push"]["status"] = "running"
    pipeline_jobs["push"]["activity_name"] = activity_name
    pipeline_jobs["push"]["message"] = f"Publishing reviewed grades from Google Sheet to Moodle for '{activity_name}'..."
    pipeline_jobs["push"]["started_at"] = time.time()
    pipeline_jobs["push"]["finished_at"] = 0
    pipeline_jobs["push"]["error"] = None

    try:
        from agents.submission_reviewer.moodle_upload_runner import run_moodle_upload_pipeline
        success = asyncio.run(run_moodle_upload_pipeline(
            activity_name=activity_name,
            only_mentor_reviewed=only_reviewed,
        ))
        if success:
            pipeline_jobs["push"]["status"] = "completed"
            pipeline_jobs["push"]["finished_at"] = time.time()
            pipeline_jobs["push"]["message"] = f"Success! Approved grades and feedback for '{activity_name}' published to Moodle Gradebook!"
            logger.info(f"✅ [Action 2 Backend] Finished successfully for '{activity_name}'.")
        else:
            raise RuntimeError("Moodle worksheet upload failed or confirmation could not be verified.")
    except Exception as e:
        logger.error(f"❌ [Action 2 Backend] Failed for '{activity_name}': {e}", exc_info=True)
        pipeline_jobs["push"]["status"] = "failed"
        pipeline_jobs["push"]["finished_at"] = time.time()
        pipeline_jobs["push"]["error"] = str(e)
        pipeline_jobs["push"]["message"] = f"Upload error: {str(e)}"


@api_router.post("/moodle/pull")
def pull_moodle_grades(
    request: Optional[Dict[str, Any]] = None,
    session: MentorSession = Depends(get_current_session),
):
    """
    Action 1 Backend: Ingests from Moodle + reviews via AI agent + writes to Central Google Sheet.
    Runs asynchronously in a background worker thread.
    """
    activity_name = (request or {}).get("activity_name") or "System Identification"
    logger.info(f"Moodle pull triggered for '{activity_name}' by {session.user_email}")

    if pipeline_jobs["pull"]["status"] == "running":
        return {
            "status": "already_running",
            "action": "pull",
            "message": f"Ingestion pipeline is already running for '{pipeline_jobs['pull']['activity_name']}'...",
            "activity_name": pipeline_jobs["pull"]["activity_name"],
            "started_at": pipeline_jobs["pull"]["started_at"],
        }

    thread = threading.Thread(
        target=_run_pull_job,
        args=(activity_name, session.user_email),
        daemon=True,
    )
    thread.start()

    return {
        "status": "started",
        "action": "pull",
        "activity_name": activity_name,
        "message": f"Moodle ingestion and AI review pipeline initiated for '{activity_name}'.",
        "initiated_by": session.mentor_name or session.user_email,
        "timestamp": time.time(),
    }


@api_router.post("/moodle/push")
def push_moodle_grades(
    request: Optional[Dict[str, Any]] = None,
    session: MentorSession = Depends(get_current_session),
):
    """
    Action 2 Backend: Uploads approved grades and feedback comments from Google Sheet to Moodle Gradebook.
    Runs asynchronously in a background worker thread.
    """
    req = request or {}
    activity_name = req.get("activity_name") or "System Identification"
    only_reviewed = bool(req.get("only_reviewed", False))
    logger.info(f"Moodle push triggered for '{activity_name}' by {session.user_email}")

    if pipeline_jobs["push"]["status"] == "running":
        return {
            "status": "already_running",
            "action": "push",
            "message": f"Moodle upload is already running for '{pipeline_jobs['push']['activity_name']}'...",
            "activity_name": pipeline_jobs["push"]["activity_name"],
            "started_at": pipeline_jobs["push"]["started_at"],
        }

    thread = threading.Thread(
        target=_run_push_job,
        args=(activity_name, only_reviewed, session.user_email),
        daemon=True,
    )
    thread.start()

    return {
        "status": "started",
        "action": "push",
        "activity_name": activity_name,
        "message": f"Moodle gradebook upload initiated for '{activity_name}'.",
        "initiated_by": session.mentor_name or session.user_email,
        "timestamp": time.time(),
    }


@api_router.get("/moodle/status")
def get_moodle_job_status(session: Optional[MentorSession] = Depends(get_optional_session)):
    """Returns the current execution status of Action 1 (pull) and Action 2 (push) jobs."""
    return pipeline_jobs


media_router = APIRouter(prefix="/api", tags=["media"])


@media_router.get("/submission-media")
def get_submission_media(
    media_folder: str = "",
    student_name: str = "",
    session: Optional[MentorSession] = Depends(get_optional_session),
):
    """Fetches media image list for a submission using the mentor's OAuth credentials or fallback."""
    try:
        images = fetch_submission_media_list(
            media_folder_url=media_folder,
            student_name=student_name,
            session=session,
        )
        return {
            "media_folder": media_folder,
            "student_name": student_name,
            "images": images,
            "total": len(images),
        }
    except Exception as e:
        logger.error(f"Failed to fetch submission media for {student_name}: {e}")
        return {"media_folder": media_folder, "student_name": student_name, "images": [], "total": 0, "error": str(e)}


@media_router.get("/media/file/{file_id:path}")
@media_router.get("/media/image/{file_id:path}")
@media_router.head("/media/file/{file_id:path}")
@media_router.head("/media/image/{file_id:path}")
def serve_media_file(
    file_id: str,
    session: Optional[MentorSession] = Depends(get_optional_session),
):
    """Streams image, video, or audio content using mentor's OAuth credentials or local cache."""
    try:
        data, mime_type = get_image_bytes(file_id, session=session)
        return Response(
            content=data,
            media_type=mime_type,
            headers={
                "Cache-Control": "private, max-age=3600",
                "Content-Disposition": "inline",
                "Accept-Ranges": "bytes",
            },
        )
    except Exception as e:
        logger.error(f"Error serving media file {file_id}: {e}")
        raise HTTPException(status_code=404, detail="Media file not found or inaccessible")


app.include_router(api_router)
app.include_router(media_router)

# Mount static asset routes
if BRAND_ASSETS_DIR.exists():
    app.mount("/brand-assets", StaticFiles(directory=str(BRAND_ASSETS_DIR)), name="brand-assets")

if (FRONTEND_DIR / "css").exists():
    app.mount("/css", StaticFiles(directory=str(FRONTEND_DIR / "css")), name="css")

if (FRONTEND_DIR / "js").exists():
    app.mount("/js", StaticFiles(directory=str(FRONTEND_DIR / "js")), name="js")

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
