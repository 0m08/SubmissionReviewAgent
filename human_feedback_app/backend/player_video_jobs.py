"""Background slide-video render jobs with progress for the experimental Player."""

from __future__ import annotations

import os
import shutil
import tempfile
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image, ImageFilter, ImageOps, ImageDraw
from moviepy.editor import (
    AudioFileClip,
    CompositeVideoClip,
    ImageClip,
    VideoClip,
    concatenate_videoclips,
)

from human_feedback_app.backend.sessions import UserSession
from agents.graphics_definition_v2.slideshow_manifest.slideshow_video import (
    CANVAS_WIDTH,
    CANVAS_HEIGHT,
    BG_COLOR,
    SLOT_CORNER_RADIUS,
    TAIL_PAUSE,
    MIN_SCENE_DURATION,
    CROSSFADE,
    FPS,
    _scaled,
    _draw_title_bar,
    _draw_panel,
    _draw_missing_asset,
    _draw_image_into_panel,
    _draw_subtitle_bar,
    _pil_rgb_to_array,
    compute_slot_rectangles,
    parse_slideshow_manifest,
    synthesize_narrations_batch,
    synthesize_narration,
    _load_asset_for_slot,
    _ffmpeg_get_duration,
    _build_video_overlay_clip,
    _round_pil_image,
    _ENCODE_PRESET,
    _ENCODE_THREADS,
    _AUDIO_BITRATE,
    _ENCODE_CRF,
)


class VideoJobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class VideoRenderJob:
    id: str
    session_id: str
    status: VideoJobStatus = VideoJobStatus.PENDING
    message: str = "Queued"
    error: Optional[str] = None
    created_at: str = field(default_factory=_now)
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    total: int = 0
    completed: int = 0
    current_title: str = ""
    output_path: Optional[str] = None
    output_name: Optional[str] = None
    output_media_type: Optional[str] = None
    slide_durations: List[float] = field(default_factory=list)
    started_monotonic: float = 0.0
    current_slide_started: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        pct = 0
        if self.total > 0:
            pct = int(round(100.0 * self.completed / self.total))
            if self.status == VideoJobStatus.RUNNING and self.completed < self.total:
                # Partial credit for the slide currently rendering.
                pct = min(99, pct + int(round(50.0 / self.total)))
        if self.status == VideoJobStatus.COMPLETED:
            pct = 100

        eta_seconds = None
        if self.status == VideoJobStatus.RUNNING and self.total > 0:
            remaining = self.total - self.completed
            if remaining > 0:
                if self.slide_durations:
                    avg = sum(self.slide_durations) / len(self.slide_durations)
                    eta_seconds = max(1, int(round(avg * remaining)))
                elif self.started_monotonic:
                    elapsed = time.monotonic() - self.started_monotonic
                    if self.completed > 0:
                        avg = elapsed / self.completed
                        eta_seconds = max(1, int(round(avg * remaining)))
                    else:
                        # First slide still going — rough floor so UI isn't blank.
                        eta_seconds = max(15, int(round(elapsed * 1.5)))

        return {
            "id": self.id,
            "status": self.status.value,
            "message": self.message,
            "error": self.error,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "total": self.total,
            "completed": self.completed,
            "current_title": self.current_title,
            "percent": pct,
            "eta_seconds": eta_seconds,
            "ready": self.status == VideoJobStatus.COMPLETED and bool(self.output_path),
            "filename": self.output_name,
        }


# ---------------------------------------------------------------------------
# Custom Experimental Video Rendering with Animation Support
# ---------------------------------------------------------------------------

def _experimental_cover_image_into_rect(pil_image, rect_w, rect_h):
    """Resize a PIL image to cover (rect_w, rect_h), cropping overflow (object-fit: cover)."""
    if pil_image is None:
        return Image.new("RGB", (rect_w, rect_h), (13, 15, 18))
    src = pil_image.convert("RGB")
    scale = max(rect_w / max(1, src.width), rect_h / max(1, src.height))
    nw = max(1, int(round(src.width * scale)))
    nh = max(1, int(round(src.height * scale)))
    resized = src.resize((nw, nh), Image.LANCZOS)
    left = max(0, (nw - rect_w) // 2)
    top = max(0, (nh - rect_h) // 2)
    return resized.crop((left, top, left + rect_w, top + rect_h))


def _experimental_normalize_hero_motion(treatment: Optional[str]) -> str:
    value = (treatment or "off").strip().lower()
    if value in ("zoom_in", "drift_x", "diagonal", "pull_out"):
        return value
    return "off"


def _experimental_normalize_scene_template(template: str) -> str:
    """Normalize slide templates to check for single hero layouts cleanly."""
    t = (template or "").strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "single_hero": "single_visual_hero",
        "hero": "single_visual_hero",
        "single_visual": "single_visual_hero",
        "two_item_split": "two_item_split_comparison",
        "split_comparison": "two_item_split_comparison",
        "multi_panel": "multi_panel_grid",
        "grid": "multi_panel_grid",
        "main_plus_inset": "main_plus_supporting_inset",
        "main_plus_supporting": "main_plus_supporting_inset",
        "main_visual_plus_inset": "main_plus_supporting_inset",
    }
    return aliases.get(t, t)


def _experimental_ease_sine_inout(u: float) -> float:
    import math
    u = max(0.0, min(1.0, float(u)))
    return 0.5 - 0.5 * math.cos(math.pi * u)


def _experimental_hero_motion_transform(treatment: str, t: float, duration: float, direction: int = 1):
    """
    Match experimental Player GSAP treatments.
    Returns (scale, x_percent, y_percent).
    """
    e = _experimental_ease_sine_inout(0.0 if duration <= 0 else (t / duration))
    d = 1 if direction >= 0 else -1
    if treatment == "zoom_in":
        return 1.0 + 0.10 * e, 0.0, 0.0
    if treatment == "pull_out":
        return 1.10 - 0.10 * e, 0.0, 0.0
    if treatment == "drift_x":
        return 1.05, (-2.0 + 4.0 * e) * d, 0.0
    if treatment == "diagonal":
        return 1.02 + 0.04 * e, (-1.5 + 3.0 * e) * d, (-1.0 + 2.0 * e) * d
    return 1.0, 0.0, 0.0


def _experimental_build_hero_motion_overlay(pil_image, rect, duration, treatment, direction=1):
    """
    Build an animated single-hero still clip matching the experimental Player:
    blurred cover backdrop + sharp contain image with Ken Burns / pan motion.
    """
    x, y, w, h = rect
    radius = _scaled(SLOT_CORNER_RADIUS, minimum=8)
    dark = (13, 15, 18)

    # Precompute expensive still layers once.
    # 1. Blurred Cover Backdrop
    blur_base = _experimental_cover_image_into_rect(pil_image, w, h)
    blur_w, blur_h = int(w * 1.08), int(h * 1.08)
    blur_scaled = blur_base.resize((blur_w, blur_h), Image.LANCZOS).filter(
        ImageFilter.GaussianBlur(radius=18)
    )
    darken = Image.new("RGB", blur_scaled.size, (0, 0, 0))
    blur_scaled = Image.blend(blur_scaled, darken, 0.35)

    # Precompute blurred backdrop RGBA
    blur_rgba = blur_scaled.convert("RGBA")
    blur_rgba.putalpha(int(255 * 0.85))
    bx = (w - blur_w) // 2
    by = (h - blur_h) // 2

    # 2. Transparent Contain Fit Foreground
    # Calculate scale factor to contain src inside (w, h) preserving its original aspect ratio
    src = pil_image.convert("RGBA")
    scale_fit = min(w / src.width, h / src.height)
    fit_w = max(1, int(round(src.width * scale_fit)))
    fit_h = max(1, int(round(src.height * scale_fit)))
    src_fit = src.resize((fit_w, fit_h), Image.LANCZOS)

    # Precompute rounded corner mask once for this scene
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, w, h), radius=radius, fill=255)

    def make_frame(t):
        scale, x_pct, y_pct = _experimental_hero_motion_transform(treatment, t, duration, direction)
        
        # Create a dark background canvas (matching the player's hf-hero-motion-wrapper background color #0d0f12)
        canvas = Image.new("RGBA", (w, h), (13, 15, 18, 255))

        # Composite the blurred backdrop onto the canvas
        canvas.alpha_composite(blur_rgba, (bx, by))

        # Resize the contained sharp image using BILINEAR (fast!)
        sw = max(1, int(round(fit_w * scale)))
        sh = max(1, int(round(fit_h * scale)))
        sharp = src_fit.resize((sw, sh), Image.BILINEAR)

        # Center + apply motion percentages
        ox = int(round((w - sw) / 2.0 + (x_pct / 100.0) * w))
        oy = int(round((h - sh) / 2.0 + (y_pct / 100.0) * h))
        
        # Composite sharp foreground onto canvas (preserves transparency around image!)
        canvas.alpha_composite(sharp, (ox, oy))

        # Apply rounded corner mask to panel composite
        canvas.putalpha(mask)
        return np.array(canvas.convert("RGB"))

    return VideoClip(make_frame, duration=duration).set_position((x, y)).set_fps(FPS)


def _experimental_make_baked_scene_image(slide_title, slot_rects, asset_results, narration, skip_image_slots=None):
    """
    Bake a single static RGB image for a scene: background + title + slot panels +
    image-slot content + subtitle. Video slots are left as the panel surface so a
    VideoFileClip can be overlaid on top later.
    """
    skip_image_slots = skip_image_slots or set()
    img = Image.new("RGBA", (CANVAS_WIDTH, CANVAS_HEIGHT), BG_COLOR + (255,))
    _draw_title_bar(img, slide_title)
    for (x, y, w, h) in slot_rects:
        _draw_panel(img, x, y, w, h)
    for i, (rect, asset) in enumerate(zip(slot_rects, asset_results)):
        if not rect:
            continue
        x, y, w, h = rect
        if i in skip_image_slots:
            # Leave dark panel; animated hero clip fills this rect.
            continue
        if asset is None:
            _draw_missing_asset(img, x, y, w, h)
            continue
        kind = asset.get("kind")
        if kind == "image" and asset.get("image") is not None:
            _draw_image_into_panel(img, asset["image"], x, y, w, h)
        elif kind == "video" and asset.get("video_path"):
            # Leave the panel surface; the moving VideoFileClip will overlay it.
            pass
        else:
            _draw_missing_asset(img, x, y, w, h)
    _draw_subtitle_bar(img, narration)
    return _pil_rgb_to_array(img.convert("RGB"))


def _local_load_asset_for_slot(asset_url: str, local_map: Dict[str, Any]):
    if not asset_url or not local_map or asset_url not in local_map:
        return {"kind": "missing", "image": None, "video_path": None}
    
    asset_info = local_map[asset_url]
    kind = asset_info["kind"]
    path = asset_info["path"]
    
    if kind == "image":
        try:
            img = Image.open(path)
            img.load()  # Force load image bytes into RAM
            return {"kind": "image", "image": img, "video_path": None}
        except Exception as e:
            print(f"Error loading local image {path}: {e}")
            return {"kind": "missing", "image": None, "video_path": None}
    elif kind == "video":
        return {"kind": "video", "image": None, "video_path": path}
    
    return {"kind": "missing", "image": None, "video_path": None}


def _experimental_load_asset_for_slot(asset_url, drive, temp_dir, slot_index, cache=None, local_map=None):
    if local_map and asset_url in local_map:
        return _local_load_asset_for_slot(asset_url, local_map)
    return _load_asset_for_slot(asset_url, drive, temp_dir, slot_index, cache)


def experimental_build_scene_clip(
    scene,
    slide_title,
    drive,
    voice,
    temp_dir,
    scene_index,
    cache=None,
    presynth_audio=None,
    hero_motion="off",
    motion_direction=1,
    local_map=None,
):
    """
    Build a single scene clip experimentally with optional hero Ken Burns / panning motion.
    """
    slots = scene.get("slots", [])
    template = scene.get("template", "single_visual_hero")
    rects = compute_slot_rectangles(template, slots)
    narration = scene.get("narration", "")
    motion = _experimental_normalize_hero_motion(hero_motion)
    norm_template = _experimental_normalize_scene_template(template)

    audio_path_target = os.path.join(temp_dir, f"scene_{scene_index}.mp3")
    if presynth_audio is not None and audio_path_target in presynth_audio:
        audio_path, audio_duration = presynth_audio[audio_path_target]
    else:
        audio_path, audio_duration = synthesize_narration(narration, audio_path_target, voice=voice)

    asset_results = [None] * len(slots)
    if slots:
        with ThreadPoolExecutor(max_workers=min(8, max(1, len(slots)))) as ex:
            futures = {
                ex.submit(
                    _experimental_load_asset_for_slot,
                    slot["asset"],
                    drive,
                    temp_dir,
                    f"{scene_index}_{i}",
                    cache,
                    local_map,
                ): i
                for i, slot in enumerate(slots)
            }
            for fut in as_completed(futures):
                i = futures[fut]
                try:
                    asset_results[i] = fut.result()
                except Exception as e:
                    print(f"  [scene] slot {i} asset load failed: {e}")
                    asset_results[i] = {"kind": "missing", "image": None, "video_path": None}

    has_video_slot = any(a and a.get("kind") == "video" for a in asset_results)
    longest_video = 0.0
    if has_video_slot:
        for a in asset_results:
            if a and a.get("kind") == "video" and a.get("video_path"):
                longest_video = max(longest_video, _ffmpeg_get_duration(a["video_path"]))

    if audio_duration > 0:
        scene_duration = max(audio_duration + TAIL_PAUSE, MIN_SCENE_DURATION)
    else:
        scene_duration = max(longest_video, MIN_SCENE_DURATION)

    # Experimental single-hero still motion (matches Player GSAP treatments).
    use_hero_motion = (
        motion != "off"
        and len(slots) == 1
        and norm_template == "single_visual_hero"
        and asset_results
        and asset_results[0]
        and asset_results[0].get("kind") == "image"
        and asset_results[0].get("image") is not None
        and rects
        and rects[0]
    )

    skip_image_slots = {0} if use_hero_motion else set()
    base_arr = _experimental_make_baked_scene_image(
        slide_title,
        rects,
        asset_results,
        narration,
        skip_image_slots=skip_image_slots,
    )
    base = ImageClip(base_arr, transparent=False).set_duration(scene_duration).set_position((0, 0))

    overlays = []
    if use_hero_motion:
        hero = _experimental_build_hero_motion_overlay(
            asset_results[0]["image"],
            rects[0],
            scene_duration,
            motion,
            direction=motion_direction,
        )
        if hero is not None:
            overlays.append(hero.set_duration(scene_duration))

    for i, slot in enumerate(slots):
        a = asset_results[i] or {"kind": "missing"}
        if a.get("kind") == "video" and a.get("video_path"):
            ov = _build_video_overlay_clip(a, rects[i], scene_duration)
            if ov is not None:
                overlays.append(ov.set_duration(scene_duration))

    if overlays:
        final = CompositeVideoClip(
            [base] + overlays,
            size=(CANVAS_WIDTH, CANVAS_HEIGHT),
        ).set_duration(scene_duration)
    else:
        final = base

    if audio_path:
        try:
            final = final.set_audio(AudioFileClip(audio_path))
        except Exception as e:
            print(f"  [scene] failed to attach narration: {e}")

    return final


def experimental_render_slide_video(
    slide_index,
    slide_title,
    manifest_xml,
    output_path,
    drive=None,
    voice="en-US-AvaNeural",
    temp_root=None,
    cache=None,
    hero_motion="off",
    local_map=None,
):
    """
    Experimental-only slide render using Ken Burns and pan animations.
    """
    scenes = parse_slideshow_manifest(manifest_xml)
    if not scenes:
        print(f"  [slide {slide_index}] no scenes parsed from manifest; skipping.")
        return None

    own_temp = False
    if temp_root is None:
        temp_root = tempfile.mkdtemp(prefix=f"slide_{slide_index}_")
        own_temp = True
    os.makedirs(temp_root, exist_ok=True)
    motion = _experimental_normalize_hero_motion(hero_motion)

    try:
        # Pre-synthesize ALL narrations concurrently.
        tts_jobs = []
        for i, scene in enumerate(scenes):
            text = (scene.get("narration") or "").strip()
            if text:
                tts_jobs.append((text, os.path.join(temp_root, f"scene_{i}.mp3")))
        presynth_audio = synthesize_narrations_batch(tts_jobs, voice=voice) if tts_jobs else {}

        scene_clips = []
        for i, scene in enumerate(scenes):
            try:
                direction = 1 if ((int(slide_index) + i) % 2 == 0) else -1
                clip = experimental_build_scene_clip(
                    scene,
                    slide_title,
                    drive,
                    voice,
                    temp_root,
                    i,
                    cache=cache,
                    presynth_audio=presynth_audio,
                    hero_motion=motion,
                    motion_direction=direction,
                    local_map=local_map,
                )
                scene_clips.append(clip)
            except Exception as e:
                print(f"  [slide {slide_index}] scene {i + 1} render failed: {e}")
                traceback.print_exc()

        if not scene_clips:
            return None

        if len(scene_clips) == 1:
            final = scene_clips[0]
        else:
            final = concatenate_videoclips(
                [c.crossfadein(CROSSFADE) if i > 0 else c for i, c in enumerate(scene_clips)],
                method="compose",
                padding=-CROSSFADE,
            )

        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
        final.write_videofile(
            output_path,
            fps=FPS,
            codec="libx264",
            audio_codec="aac",
            preset=_ENCODE_PRESET,
            threads=_ENCODE_THREADS,
            audio_bitrate=_AUDIO_BITRATE,
            ffmpeg_params=["-crf", str(_ENCODE_CRF), "-pix_fmt", "yuv420p", "-movflags", "+faststart"],
            verbose=False,
            logger=None,
        )

        for c in scene_clips:
            try:
                c.close()
            except Exception:
                pass
        return output_path
    finally:
        if own_temp:
            shutil.rmtree(temp_root, ignore_errors=True)


class VideoRenderJobStore:
    def __init__(self, max_workers: int = 2) -> None:
        self._jobs: Dict[str, VideoRenderJob] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    def get(self, job_id: str, session_id: str) -> Optional[VideoRenderJob]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.session_id != session_id:
                return None
            return job

    def start(
        self,
        session: UserSession,
        row_indexes: List[int],
        voice: str,
        hero_motion: str = "off",
    ) -> VideoRenderJob:
        job_id = uuid.uuid4().hex[:12]
        job = VideoRenderJob(
            id=job_id,
            session_id=session.session_id,
            total=len(row_indexes),
            message="Queued for render",
        )
        with self._lock:
            self._jobs[job_id] = job
        self._executor.submit(
            self._run,
            job_id,
            session,
            list(row_indexes),
            voice,
            hero_motion or "off",
        )
        return job

    def _update(self, job_id: str, **kwargs: Any) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            for key, value in kwargs.items():
                setattr(job, key, value)

    def _run(
        self,
        job_id: str,
        session: UserSession,
        row_indexes: List[int],
        voice: str,
        hero_motion: str = "off",
    ) -> None:
        from human_feedback_app.backend.sheet_service import (
            _column_map,
            _manifest_for_ui_row,
            load_workbook,
        )
        from agents.graphics_definition_v2.slideshow_manifest.slideshow_video import (
            AssetCache,
            render_course_video,
            sanitize_filename,
        )

        def _as_text(value) -> str:
            if value is None:
                return ""
            text = str(value).strip()
            return "" if text.lower() == "nan" else text

        self._update(
            job_id,
            status=VideoJobStatus.RUNNING,
            started_at=_now(),
            started_monotonic=time.monotonic(),
            message="Loading sheet...",
        )

        try:
            _, df, _ = load_workbook(session)
            if "slideshow_manifest" not in df.columns:
                raise RuntimeError("The sheet is missing the slideshow_manifest column.")

            cmap = _column_map(df)
            title_col = cmap.get("slide_title") or cmap.get("title")
            temp_dir = tempfile.mkdtemp(prefix=f"slide_videos_{job_id}_")
            asset_cache = AssetCache()
            rendered_files = []

            valid_rows = []
            for row_idx in row_indexes:
                if not isinstance(row_idx, int) or row_idx < 0 or row_idx >= len(df):
                    continue
                row = df.iloc[row_idx]
                manifest_xml = _as_text(_manifest_for_ui_row(session, row_idx, row))
                if not manifest_xml or manifest_xml.startswith("ERROR:"):
                    continue
                slide_title = _as_text(row.get(title_col, "")) if title_col else ""
                if not slide_title:
                    slide_title = f"Slide {row_idx + 1}"
                valid_rows.append((row_idx, slide_title, manifest_xml))

            if not valid_rows:
                raise RuntimeError(
                    "No valid slide videos to render. Check slideshow_manifest for selected slides."
                )

            self._update(job_id, total=len(valid_rows), message="Pre-downloading course assets in parallel...")

            # 1. Gather all unique assets across selected slides
            unique_assets = set()
            for _, _, manifest_xml in valid_rows:
                scenes = parse_slideshow_manifest(manifest_xml)
                for scene in scenes:
                    for slot in scene.get("slots", []):
                        url = slot.get("asset")
                        if url:
                            unique_assets.add(url)

            # 2. Pre-download all unique assets in parallel to local files
            local_map = {}
            local_map_lock = threading.Lock()

            def download_one(url):
                try:
                    from agents.graphics_definition_v2.slideshow_manifest.slideshow_video import (
                        is_youtube_url,
                        is_drive_url,
                        download_youtube_clip,
                        download_drive_image,
                        download_web_image,
                    )
                    import uuid
                    asset_id = uuid.uuid4().hex[:10]
                    if is_youtube_url(url):
                        out_path = os.path.join(temp_dir, f"yt_{asset_id}.mp4")
                        path = download_youtube_clip(url, out_path)
                        if path and os.path.exists(path):
                            with local_map_lock:
                                local_map[url] = {"kind": "video", "path": path}
                    elif is_drive_url(url):
                        img = download_drive_image(url, session.drive)
                        if img:
                            out_path = os.path.join(temp_dir, f"drive_{asset_id}.png")
                            img.save(out_path, "PNG")
                            with local_map_lock:
                                local_map[url] = {"kind": "image", "path": out_path}
                    else:
                        img = download_web_image(url)
                        if img:
                            out_path = os.path.join(temp_dir, f"web_{asset_id}.png")
                            img.save(out_path, "PNG")
                            with local_map_lock:
                                local_map[url] = {"kind": "image", "path": out_path}
                except Exception as e:
                    print(f"Error pre-downloading asset {url}: {e}")

            if unique_assets:
                with ThreadPoolExecutor(max_workers=8) as ex:
                    ex.map(download_one, list(unique_assets))

            self._update(job_id, total=len(valid_rows), message="Starting parallel render...")

            # Render slides in parallel using a ThreadPoolExecutor
            max_slide_workers = min(2, len(valid_rows))
            completed_count = 0
            completed_lock = threading.Lock()
            rendered_files_list = [None] * len(valid_rows)

            def render_single_slide(idx, row_idx, slide_title, manifest_xml):
                nonlocal completed_count
                slide_started = time.monotonic()
                out_name = (
                    f"Slide_{row_idx:03d}_{sanitize_filename(slide_title, fallback='slide')[:60]}.mp4"
                )
                out_path = os.path.join(temp_dir, out_name)

                self._update(
                    job_id,
                    message=f"Rendering slide: {slide_title}",
                )

                try:
                    result_path = experimental_render_slide_video(
                        slide_index=row_idx + 1,
                        slide_title=slide_title,
                        manifest_xml=manifest_xml,
                        output_path=out_path,
                        drive=session.drive,
                        voice=voice,
                        cache=asset_cache,
                        hero_motion=hero_motion,
                        local_map=local_map,
                    )
                except Exception as exc:
                    print(f"Error rendering slide {slide_title}: {exc}")
                    traceback.print_exc()
                    result_path = None

                duration = time.monotonic() - slide_started
                with completed_lock:
                    completed_count += 1
                    job = self._jobs.get(job_id)
                    if job is not None:
                        job.slide_durations.append(duration)
                    self._update(
                        job_id,
                        completed=completed_count,
                        message=f"Rendered {completed_count} of {len(valid_rows)} slides: {slide_title}",
                    )

                return result_path, out_name

            with ThreadPoolExecutor(max_workers=max_slide_workers) as executor:
                futures = {
                    executor.submit(render_single_slide, i, row_idx, slide_title, manifest_xml): i
                    for i, (row_idx, slide_title, manifest_xml) in enumerate(valid_rows)
                }
                for fut in as_completed(futures):
                    i = futures[fut]
                    try:
                        res_path, out_name = fut.result()
                        if res_path and os.path.exists(res_path):
                            rendered_files_list[i] = (res_path, out_name)
                    except Exception as exc:
                        print(f"Slide render future failed: {exc}")
                        traceback.print_exc()

            rendered_files = [f for f in rendered_files_list if f is not None]

            if not rendered_files:
                raise RuntimeError("Render finished but produced no MP4 files.")

            if len(rendered_files) > 1:
                self._update(job_id, message="Stitching slides into one video...")
                course_name = sanitize_filename(
                    getattr(session, "course_name", None) or "course_video",
                    fallback="course_video",
                )
                output_name = f"{course_name[:60]}_selected_slides.mp4"
                output_path = os.path.join(temp_dir, output_name)
                stitched = render_course_video(
                    [path for path, _ in rendered_files],
                    output_path,
                )
                if not stitched or not os.path.exists(stitched):
                    raise RuntimeError("Failed to stitch selected slides into one video.")
                output_path = stitched
                media_type = "video/mp4"
            else:
                output_path, output_name = rendered_files[0]
                media_type = "video/mp4"

            self._update(
                job_id,
                status=VideoJobStatus.COMPLETED,
                finished_at=_now(),
                completed=len(valid_rows),
                current_title="",
                message="Ready to download",
                output_path=output_path,
                output_name=output_name,
                output_media_type=media_type,
            )
        except Exception as exc:
            traceback.print_exc()
            self._update(
                job_id,
                status=VideoJobStatus.FAILED,
                finished_at=_now(),
                error=str(exc),
                message="Render failed",
            )


video_render_jobs = VideoRenderJobStore()
