"""FastAPI application for Human Feedback review."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from human_feedback_app.backend.auth import ensure_google_clients, get_current_session, router as auth_router
from human_feedback_app.backend.asset_service import image_response
from human_feedback_app.backend.config import BRAND_ASSETS_DIR, BRAND_FALLBACK_DIR, FRONTEND_DIR
from human_feedback_app.backend.jobs import revision_queue
from human_feedback_app.backend.revise_worker import run_row_revision
from human_feedback_app.backend.segmentation_worker import run_row_segmentation_revision
from human_feedback_app.backend.sheet_service import (
    approve_visual,
    load_workbook,
    prepare_visual_revision,
    select_pool_alternative,
    slides_to_ui_payload,
)
from human_feedback_app.backend.sessions import UserSession
from services.drive_service import extract_drive_id_from_url, is_inside_skillcat_shared_drive

api_router = APIRouter(prefix="/api", tags=["api"])


class LoadSheetRequest(BaseModel):
    sheet_link: str = Field(min_length=10)
    worksheet_name: str = Field(default="Slide Chunks")
    root_folder_id: str = ""


class ReviseRequest(BaseModel):
    row_index: int
    segment_index: int
    step_index: int
    visual_id: str
    vo: str
    mode: str
    feedback: str = ""


class ApproveRequest(BaseModel):
    row_index: int
    segment_index: int
    step_index: int
    visual_id: str
    vo: str


class SelectPoolRequest(BaseModel):
    row_index: int
    segment_index: int
    step_index: int
    visual_id: str
    vo: str
    asset_url: str = Field(min_length=8)


class RevertRequest(BaseModel):
    row_index: int
    segment_index: int
    step_index: int
    visual_id: str
    vo: str


class SegmentationReviseRequest(BaseModel):
    row_index: int
    feedback: str = Field(min_length=3)


def session_dep(request: Request) -> UserSession:
    return get_current_session(request)


@api_router.get("/me")
def api_me(session: UserSession = Depends(session_dep)) -> Dict[str, Any]:
    return {
        "email": session.user_email,
        "role": session.role,
        "sheetLink": session.sheet_link,
        "worksheetName": session.worksheet_name,
    }


@api_router.post("/session/load")
def api_load_sheet(body: LoadSheetRequest, session: UserSession = Depends(session_dep)) -> Dict[str, Any]:
    ensure_google_clients(session)

    sheet_id = extract_drive_id_from_url(body.sheet_link)
    if not sheet_id:
        raise HTTPException(status_code=400, detail="Invalid Google Sheet link")
    if not is_inside_skillcat_shared_drive(sheet_id):
        raise HTTPException(status_code=400, detail="Sheet must be inside the Skillcat Shared Drive")

    session.sheet_link = body.sheet_link.strip()
    session.worksheet_name = body.worksheet_name.strip() or "Slide Chunks"
    session.root_folder_id = (body.root_folder_id or "").strip()
    session.sheet = session.gc.open_by_url(session.sheet_link)
    return slides_to_ui_payload(session)


@api_router.get("/slides")
def api_slides(session: UserSession = Depends(session_dep)) -> Dict[str, Any]:
    if not session.sheet_link:
        raise HTTPException(status_code=400, detail="No sheet loaded")
    return slides_to_ui_payload(session)


@api_router.get("/assets/image")
def api_asset_image(url: str = Query(min_length=8), thumb: str = Query(""), session: UserSession = Depends(session_dep)):
    return image_response(session, url, thumb)


@api_router.post("/visuals/revert")
def api_revert(body: RevertRequest, session: UserSession = Depends(session_dep)) -> Dict[str, str]:
    from human_feedback_app.backend.sheet_service import revert_visual
    revert_visual(
        session,
        row_index=body.row_index,
        segment_index=body.segment_index,
        step_index=body.step_index,
        visual_id=body.visual_id,
        vo=body.vo,
    )
    return {"status": "ok"}


@api_router.post("/visuals/approve")
def api_approve(body: ApproveRequest, session: UserSession = Depends(session_dep)) -> Dict[str, str]:
    approve_visual(
        session,
        row_index=body.row_index,
        segment_index=body.segment_index,
        step_index=body.step_index,
        visual_id=body.visual_id,
        vo=body.vo,
    )
    return {"status": "ok"}


@api_router.post("/visuals/select-pool")
def api_select_pool(body: SelectPoolRequest, session: UserSession = Depends(session_dep)) -> Dict[str, str]:
    select_pool_alternative(
        session,
        row_index=body.row_index,
        segment_index=body.segment_index,
        step_index=body.step_index,
        visual_id=body.visual_id,
        vo=body.vo,
        asset_url=body.asset_url,
    )
    return {"status": "ok"}


@api_router.post("/visuals/revise")
def api_revise(body: ReviseRequest, session: UserSession = Depends(session_dep)) -> Dict[str, Any]:
    if body.mode not in ("drive", "all", "ai"):
        raise HTTPException(status_code=400, detail="Invalid mode")

    prepare_visual_revision(
        session,
        row_index=body.row_index,
        segment_index=body.segment_index,
        step_index=body.step_index,
        visual_id=body.visual_id,
        vo=body.vo,
        mode=body.mode,
        feedback=body.feedback,
    )

    slide_idx = None
    payload = slides_to_ui_payload(session)
    for idx, slide in enumerate(payload["slides"]):
        for seg in slide.get("segments", []):
            for step in seg.get("steps", []):
                if step.get("rowIndex") == body.row_index and step.get("visualId") == body.visual_id:
                    slide_idx = idx
                    break

    label = f"Slide {(slide_idx or 0) + 1} / {body.visual_id}"

    def worker() -> Dict[str, Any]:
        run_row_revision(
            session,
            body.row_index,
            body.vo,
            segment_index=body.segment_index,
            step_index=body.step_index,
            visual_id=body.visual_id,
        )
        refreshed = slides_to_ui_payload(session)
        return refreshed

    job = revision_queue.submit(
        session_id=session.session_id,
        label=label,
        worker=worker,
        slide_index=slide_idx or 0,
        segment_index=body.segment_index,
        step_index=body.step_index,
    )
    return job.to_dict()


@api_router.post("/slides/revise-segmentation")
def api_revise_segmentation(
    body: SegmentationReviseRequest,
    session: UserSession = Depends(session_dep),
) -> Dict[str, Any]:
    slide_idx = None
    payload = slides_to_ui_payload(session)
    for idx, slide in enumerate(payload["slides"]):
        for seg in slide.get("segments", []):
            for step in seg.get("steps", []):
                if step.get("rowIndex") == body.row_index:
                    slide_idx = idx
                    break

    label = f"Slide {(slide_idx or 0) + 1} / segmentation"

    def worker() -> Dict[str, Any]:
        run_row_segmentation_revision(session, body.row_index, body.feedback.strip())
        return slides_to_ui_payload(session)

    job = revision_queue.submit(
        session_id=session.session_id,
        label=label,
        worker=worker,
        slide_index=slide_idx or 0,
        segment_index=0,
        step_index=0,
    )
    return job.to_dict()


@api_router.get("/jobs")
def api_jobs(session: UserSession = Depends(session_dep)) -> Any:
    return [job.to_dict() for job in revision_queue.list_jobs(session.session_id)]


@api_router.get("/jobs/{job_id}")
def api_job(job_id: str, session: UserSession = Depends(session_dep)) -> Dict[str, Any]:
    job = revision_queue.get(job_id, session.session_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job.to_dict()


def create_app() -> FastAPI:
    app = FastAPI(title="Human Feedback App", version="1.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(auth_router)
    app.include_router(api_router)

    @app.get("/health")
    def health() -> Dict[str, str]:
        return {"status": "ok"}

    @app.get("/")
    def root(request: Request):
        if "code" in request.query_params and "state" in request.query_params:
            return RedirectResponse(f"/auth/callback?{request.url.query}", status_code=307)
        try:
            session = get_current_session(request)
            if session.sheet_link:
                return RedirectResponse("/review")
            return RedirectResponse("/setup")
        except HTTPException:
            return RedirectResponse("/login-page")

    @app.get("/login-page")
    def login_page() -> FileResponse:
        return FileResponse(FRONTEND_DIR / "login.html")

    @app.get("/setup")
    def setup_page(request: Request):
        try:
            get_current_session(request)
        except HTTPException:
            return RedirectResponse("/login-page", status_code=302)
        return FileResponse(FRONTEND_DIR / "setup.html")

    @app.get("/review")
    def review_page(request: Request):
        try:
            get_current_session(request)
        except HTTPException:
            return RedirectResponse("/login-page", status_code=302)
        path = FRONTEND_DIR / "Slide Review.dc.html"
        if not path.exists():
            raise HTTPException(status_code=404, detail="Review UI not found")
        return FileResponse(path)

    brand_dir = BRAND_ASSETS_DIR if BRAND_ASSETS_DIR.exists() else BRAND_FALLBACK_DIR
    if brand_dir.exists():
        app.mount("/brand-assets", StaticFiles(directory=str(brand_dir)), name="brand-assets")

    if FRONTEND_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    return app


app = create_app()
