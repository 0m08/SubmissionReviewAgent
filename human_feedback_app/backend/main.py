"""FastAPI application for Human Feedback review."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Dict, Optional, List

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from human_feedback_app.backend.auth import ensure_google_clients, get_current_session, router as auth_router
from human_feedback_app.backend.asset_service import image_response, video_clip_response
from human_feedback_app.backend.config import BRAND_ASSETS_DIR, BRAND_FALLBACK_DIR, FRONTEND_DIR
from human_feedback_app.backend.jobs import JobCancelled, revision_queue
from human_feedback_app.backend.tts_service import encode_tts_words_header, synthesize_tts_with_words
from human_feedback_app.backend.revise_worker import run_row_revision
from human_feedback_app.backend.segmentation_worker import run_row_segmentation_revision
from human_feedback_app.backend.layout_worker import run_row_layout_revision
from human_feedback_app.backend.sheet_service import (
    approve_visual,
    clear_visual_revision_action,
    get_manifest_sync_status,
    load_workbook,
    maybe_trigger_manifest_sync_checker,
    prepare_visual_revision,
    reconcile_stale_visual_revisions,
    replace_visual_with_url,
    revert_visual,
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


class ReplaceRequest(BaseModel):
    row_index: int
    segment_index: int
    step_index: int
    visual_id: str
    vo: str
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


class LayoutReviseRequest(BaseModel):
    row_index: int
    scene_id: str
    feedback: str = Field(min_length=3)


class DownloadVideosRequest(BaseModel):
    row_indexes: List[int]
    voice: Optional[str] = "en-US-AvaNeural"
    quality: Optional[str] = "fast"
    hero_motion: Optional[str] = "zoom_in"


def session_dep(request: Request) -> UserSession:
    return get_current_session(request)


@api_router.get("/player-theme")
def api_player_theme() -> Dict[str, str]:
    """Course player style variables from human_feedback_app/player_styles.yaml."""
    from human_feedback_app.player_config_loader import load_player_theme

    return load_player_theme()


@api_router.get("/styles-schema")
def api_styles_schema() -> Dict[str, Any]:
    """Course player styles schema and groupings from human_feedback_app/player_styles.yaml."""
    from human_feedback_app.player_config_loader import load_styles_schema

    return load_styles_schema()


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
    session.manifest_repair_cache = {}
    session.manifest_sync_triggered = False
    session.manifest_sync_status = {}
    reconcile_stale_visual_revisions(session)
    payload = slides_to_ui_payload(session)
    maybe_trigger_manifest_sync_checker(session)
    return payload


@api_router.get("/slides")
def api_slides(session: UserSession = Depends(session_dep)) -> Dict[str, Any]:
    if not session.sheet_link:
        raise HTTPException(status_code=400, detail="No sheet loaded")
    reconcile_stale_visual_revisions(session)
    return slides_to_ui_payload(session)


@api_router.get("/manifest-sync")
def api_manifest_sync(session: UserSession = Depends(session_dep)) -> Dict[str, Any]:
    if not session.sheet_link:
        raise HTTPException(status_code=400, detail="No sheet loaded")
    maybe_trigger_manifest_sync_checker(session)
    return get_manifest_sync_status(session)


@api_router.get("/assets/image")
def api_asset_image(url: str = Query(min_length=8), thumb: str = Query(""), session: UserSession = Depends(session_dep)):
    return image_response(session, url, thumb)


@api_router.get("/assets/video")
def api_asset_video(request: Request, url: str = Query(min_length=8), session: UserSession = Depends(session_dep)):
    range_header = request.headers.get("range", "")
    return video_clip_response(session, url, range_header)


@api_router.get("/tts")
def api_tts(voiceover: str = Query(min_length=1), session: UserSession = Depends(session_dep)) -> Response:
    try:
        data, words = synthesize_tts_with_words(voiceover)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"TTS failed: {exc}") from exc
    headers = {"Cache-Control": "public, max-age=86400"}
    encoded = encode_tts_words_header(words)
    if encoded:
        headers["X-TTS-Words"] = encoded
    return Response(
        content=data,
        media_type="audio/mpeg",
        headers=headers,
    )


@api_router.get("/tts/words")
def api_tts_words(voiceover: str = Query(min_length=1), session: UserSession = Depends(session_dep)) -> Dict[str, Any]:
    try:
        _, words = synthesize_tts_with_words(voiceover)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"TTS failed: {exc}") from exc
    return {"words": words}


@api_router.post("/download-slide-videos")
def api_start_download_slide_videos(
    body: DownloadVideosRequest,
    session: UserSession = Depends(session_dep),
) -> Dict[str, Any]:
    ensure_google_clients(session)
    if not body.row_indexes:
        raise HTTPException(status_code=400, detail="No slides selected.")

    from human_feedback_app.backend.player_video_jobs import video_render_jobs

    job = video_render_jobs.start(
        session=session,
        row_indexes=body.row_indexes,
        voice=body.voice or "en-US-AvaNeural",
        hero_motion=body.hero_motion or "zoom_in",
    )
    return {"job_id": job.id, **job.to_dict()}


@api_router.get("/download-slide-videos/{job_id}")
def api_download_slide_videos_status(
    job_id: str,
    session: UserSession = Depends(session_dep),
) -> Dict[str, Any]:
    from human_feedback_app.backend.player_video_jobs import video_render_jobs

    job = video_render_jobs.get(job_id, session.session_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Render job not found")
    return job.to_dict()


@api_router.get("/download-slide-videos/{job_id}/file")
def api_download_slide_videos_file(
    job_id: str,
    session: UserSession = Depends(session_dep),
):
    from human_feedback_app.backend.player_video_jobs import video_render_jobs

    job = video_render_jobs.get(job_id, session.session_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Render job not found")
    if job.status.value != "completed" or not job.output_path:
        raise HTTPException(status_code=409, detail=job.error or "Render is not ready yet")
    if not Path(job.output_path).exists():
        raise HTTPException(status_code=404, detail="Rendered file is missing on server")
    return FileResponse(
        job.output_path,
        media_type=job.output_media_type or "application/octet-stream",
        filename=job.output_name or "slide_video",
    )


@api_router.post("/visuals/revert")
def api_revert(body: RevertRequest, session: UserSession = Depends(session_dep)) -> Dict[str, str]:
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


@api_router.post("/visuals/replace")
def api_replace(body: ReplaceRequest, session: UserSession = Depends(session_dep)) -> Dict[str, Any]:
    try:
        replaced_url = replace_visual_with_url(
            session,
            row_index=body.row_index,
            segment_index=body.segment_index,
            step_index=body.step_index,
            visual_id=body.visual_id,
            vo=body.vo,
            feedback=body.feedback,
        )
        payload = slides_to_ui_payload(session)
        return {"status": "ok", "replaced_url": replaced_url, "payload": payload}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


def _job_aborted(session: UserSession, job_id: str) -> bool:
    return bool(getattr(session, "aborted", False)) or revision_queue.is_cancelled(job_id)


@api_router.post("/visuals/revise")
def api_revise(body: ReviseRequest, session: UserSession = Depends(session_dep)) -> Dict[str, Any]:
    if body.mode not in ("drive", "all", "ai"):
        raise HTTPException(status_code=400, detail="Invalid mode")

    # Register the job BEFORE writing the reject action so a concurrent /slides reconcile cannot treat this visual as an orphan and clear it.
    job_id = uuid.uuid4().hex[:12]
    job = revision_queue.register(
        session_id=session.session_id,
        label=f"visual / {body.visual_id}",
        slide_index=0,
        segment_index=body.segment_index,
        step_index=body.step_index,
        kind="visual",
        row_index=body.row_index,
        visual_id=body.visual_id,
        job_id=job_id,
    )

    try:
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

        job.label = f"Slide {(slide_idx or 0) + 1} / {body.visual_id}"
        job.slide_index = slide_idx or 0
    except Exception:
        revision_queue.cancel_job(job_id, error="Failed to start revision")
        try:
            clear_visual_revision_action(session, body.row_index, body.visual_id)
        except Exception:
            pass
        raise

    def worker() -> Dict[str, Any]:
        try:
            if _job_aborted(session, job_id):
                raise JobCancelled("Cancelled on logout")
            run_row_revision(
                session,
                body.row_index,
                body.vo,
                segment_index=body.segment_index,
                step_index=body.step_index,
                visual_id=body.visual_id,
            )
            if _job_aborted(session, job_id):
                raise JobCancelled("Cancelled on logout")
            return slides_to_ui_payload(session)
        except JobCancelled:
            raise
        except Exception:
            # Failed revisions must not leave the sheet stuck in "revising".
            try:
                if not getattr(session, "aborted", False):
                    clear_visual_revision_action(session, body.row_index, body.visual_id)
            except Exception:
                pass
            raise

    revision_queue.start(job_id, worker)
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
    job_id = uuid.uuid4().hex[:12]

    def worker() -> Dict[str, Any]:
        if _job_aborted(session, job_id):
            raise JobCancelled("Cancelled on logout")
        run_row_segmentation_revision(session, body.row_index, body.feedback.strip())
        if _job_aborted(session, job_id):
            raise JobCancelled("Cancelled on logout")
        return slides_to_ui_payload(session)

    job = revision_queue.submit(
        session_id=session.session_id,
        label=label,
        worker=worker,
        slide_index=slide_idx or 0,
        segment_index=0,
        step_index=0,
        kind="segmentation",
        row_index=body.row_index,
        job_id=job_id,
    )
    return job.to_dict()


@api_router.post("/slides/revise-layout")
def api_revise_layout(
    body: LayoutReviseRequest,
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

    label = f"Slide {(slide_idx or 0) + 1} / layout / Scene {body.scene_id}"
    job_id = uuid.uuid4().hex[:12]

    def worker() -> Dict[str, Any]:
        if _job_aborted(session, job_id):
            raise JobCancelled("Cancelled on logout")
        layout_result = run_row_layout_revision(session, body.row_index, body.scene_id, body.feedback.strip())
        if _job_aborted(session, job_id):
            raise JobCancelled("Cancelled on logout")
        payload = slides_to_ui_payload(session)
        payload["layoutRevisionResult"] = layout_result
        return payload

    job = revision_queue.submit(
        session_id=session.session_id,
        label=label,
        worker=worker,
        slide_index=slide_idx or 0,
        segment_index=0,
        step_index=0,
        kind="layout",
        row_index=body.row_index,
        scene_id=body.scene_id,
        job_id=job_id,
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

        mtime = str(int(path.stat().st_mtime))
        if request.query_params.get("v") != mtime:
            return RedirectResponse("/review?v=" + mtime, status_code=302)
        return FileResponse(
            path,
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )

    brand_dir = BRAND_ASSETS_DIR if BRAND_ASSETS_DIR.exists() else BRAND_FALLBACK_DIR
    if brand_dir.exists():
        app.mount("/brand-assets", StaticFiles(directory=str(brand_dir)), name="brand-assets")

    if FRONTEND_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    return app


app = create_app()
