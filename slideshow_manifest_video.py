"""
Streamlit page: Slideshow Video Generator (manifest-driven).

Reads the `slideshow_manifest` column on the Slide Chunks worksheet of a
Google Sheet, generates per-slide MP4s with edge-tts narration and the
layout templates declared in the manifest, and lets the user download
either individual slide videos or a single stitched course video.
"""

import os
import re
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import streamlit as st

from agents.graphics_definition_v2.slideshow_manifest.slideshow_video import (
    AssetCache,
    DEFAULT_VOICE,
    RENDER_PRESETS,
    apply_render_preset,
    extract_sheet_id_from_link,
    load_slide_rows,
    parse_slideshow_manifest,
    render_course_video,
    render_slide_video,
    sanitize_filename,
)


VOICE_OPTIONS = [
    ("Ava (US, female, warm)", "en-US-AvaNeural"),
    ("Andrew (US, male, neutral)", "en-US-AndrewNeural"),
    ("Emma (US, female, energetic)", "en-US-EmmaNeural"),
    ("Brian (US, male, calm)", "en-US-BrianNeural"),
    ("Ryan (UK, male)", "en-GB-RyanNeural"),
    ("Sonia (UK, female)", "en-GB-SoniaNeural"),
]

SLIDE_CONCURRENCY = 40


def _ensure_session_keys():
    defaults = {
        "manifest_video_outputs": {},        # slide_index -> {"path", "title"}
        "manifest_video_course_path": None,
        "manifest_video_run_id": None,
        "manifest_video_logs": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _log(msg):
    st.session_state.setdefault("manifest_video_logs", []).append(
        f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    )


def _format_seconds(seconds):
    """Format seconds as HH:MM:SS."""
    seconds = max(0, int(seconds or 0))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _summarize_manifest(manifest_xml):
    scenes = parse_slideshow_manifest(manifest_xml)
    total_slots = sum(len(s.get("slots", [])) for s in scenes)
    templates = sorted({s.get("template", "?") for s in scenes})
    return {
        "scenes": len(scenes),
        "slots": total_slots,
        "templates": templates,
    }


def render_slideshow_manifest_video_page():
    """Streamlit entry point for the manifest-driven slideshow video generator."""
    st.title(":material/slideshow: Slideshow Video Generator")
    st.caption(
        "Render layout-aware slideshow videos from the Graphics Definition V2 "
        "`slideshow_manifest` column. Narration uses edge-tts; visuals follow the manifest "
        "templates and slot roles."
    )

    if "drive" not in st.session_state or "gc" not in st.session_state:
        st.error("Please log in first so the app can read your Google Sheet.")
        st.stop()

    _ensure_session_keys()

    drive = st.session_state["drive"]
    gc = st.session_state["gc"]

    with st.form("slideshow_manifest_video_form"):
        col_a, col_b = st.columns([3, 2])
        with col_a:
            sheet_link = st.text_input(
                "Google Sheet link",
                value=st.session_state.get("manifest_video_sheet_link", ""),
                placeholder="https://docs.google.com/spreadsheets/d/.../edit",
            )
            worksheet_name = st.text_input(
                "Worksheet (tab) name",
                value=st.session_state.get("manifest_video_worksheet", "Slide Chunks"),
            )
        with col_b:
            voice_label = st.selectbox(
                "Narration voice",
                options=[v[0] for v in VOICE_OPTIONS],
                index=0,
            )
            voice_id = next(v[1] for v in VOICE_OPTIONS if v[0] == voice_label)
            output_dir = st.text_input(
                "Output folder (server-side)",
                value=st.session_state.get(
                    "manifest_video_output_dir",
                    os.path.join(tempfile.gettempdir(), "slideshow_manifest_videos"),
                ),
                help="Per-slide MP4s and the optional stitched course MP4 are written here.",
            )

        st.markdown("**Render speed vs quality**")
        with st.container():
            quality_label = st.selectbox(
                "Render preset",
                options=[
                    "Fast (720p, ultrafast x264) — recommended",
                    "Balanced (1080p, veryfast x264)",
                    "High (1080p 30fps, medium x264)",
                ],
                index=0,
                help="Fast is ~3-5x quicker than High at the cost of slightly larger files / softer detail.",
            )
            quality_key = (
                "fast" if "Fast" in quality_label
                else "balanced" if "Balanced" in quality_label
                else "high"
            )

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            preview_submit = st.form_submit_button(":material/preview: Preview manifest")
        with col_s2:
            render_submit = st.form_submit_button(":material/movie: Render slides", type="primary")

    st.session_state["manifest_video_sheet_link"] = sheet_link
    st.session_state["manifest_video_worksheet"] = worksheet_name
    st.session_state["manifest_video_output_dir"] = output_dir

    if not sheet_link:
        return

    # Load rows once for both preview and render paths
    try:
        course_name, rows = load_slide_rows(gc, sheet_link, worksheet_name=worksheet_name)
    except Exception as e:
        st.error(f"Failed to load worksheet '{worksheet_name}': {e}")
        return

    if not rows:
        st.warning(
            f"No rows on '{worksheet_name}' have a `slideshow_manifest` filled in yet. "
            "Run the Slideshow Manifest step (Section 10 of Graphics Definition V2) first."
        )
        return

    selected_rows = rows

    # ----- Preview -----
    if preview_submit:
        st.subheader("Manifest preview")
        st.caption(f"Course: {course_name or '(unknown)'} — {len(selected_rows)} slide(s)")
        for r in selected_rows[:30]:
            summary = _summarize_manifest(r["manifest"])
            with st.expander(
                f"Slide {r['slide_index']} — {r['slide_title'] or '(untitled)'}  "
                f"[{summary['scenes']} scenes, {summary['slots']} slots]"
            ):
                st.markdown(
                    f"- **Templates:** {', '.join(summary['templates']) or '—'}  \n"
                    f"- **Slide type:** {r['slide_type'] or '—'}"
                )
                st.code(r["manifest"][:4000], language="xml")
        if len(selected_rows) > 30:
            st.info(f"Preview limited to first 30 of {len(selected_rows)} slides.")
        return

    # ----- Render -----
    if not render_submit:
        _render_existing_outputs()
        return

    os.makedirs(output_dir, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(output_dir, f"{sanitize_filename(course_name) or 'course'}_{run_id}")
    os.makedirs(run_dir, exist_ok=True)

    st.session_state["manifest_video_run_id"] = run_id
    st.session_state["manifest_video_outputs"] = {}
    st.session_state["manifest_video_course_path"] = None
    st.session_state["manifest_video_logs"] = []

    cfg = apply_render_preset(quality_key)
    asset_cache = AssetCache()

    st.subheader("Rendering slides")
    st.caption(
        f"Output folder: `{run_dir}` — preset: **{quality_key}** "
        f"({cfg['width']}x{cfg['height']} @ {cfg['fps']}fps, x264 {cfg['preset']} crf {cfg['crf']}) "
        f"— {SLIDE_CONCURRENCY} slide(s) in parallel"
    )
    progress = st.progress(0.0, text="Starting...")
    status = st.empty()
    log_area = st.expander("Render log", expanded=False)

    total = len(selected_rows)
    completed = 0
    started_at = datetime.now()

    def _render_one(row):
        slide_idx = row["slide_index"]
        title = row["slide_title"] or f"Slide {slide_idx}"
        out_name = f"Slide_{slide_idx:03d}_{sanitize_filename(title, fallback='slide')[:60]}.mp4"
        out_path = os.path.join(run_dir, out_name)
        try:
            result = render_slide_video(
                slide_index=slide_idx,
                slide_title=title,
                manifest_xml=row["manifest"],
                output_path=out_path,
                drive=drive,
                voice=voice_id,
                cache=asset_cache,
            )
            return slide_idx, title, result, None
        except Exception as e:
            return slide_idx, title, None, str(e)

    workers = max(1, min(SLIDE_CONCURRENCY, total))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_render_one, row): row for row in selected_rows}
        for fut in as_completed(futures):
            slide_idx, title, result, err = fut.result()
            completed += 1
            if err:
                _log(f"Slide {slide_idx} ({title}): FAILED — {err}")
                status.error(f"Slide {slide_idx} failed: {err}")
            elif result:
                _log(f"Slide {slide_idx} ({title}): rendered → {result}")
                st.session_state["manifest_video_outputs"][slide_idx] = {
                    "path": result,
                    "title": title,
                }
                status.success(f"Slide {slide_idx} done.")
            else:
                _log(f"Slide {slide_idx} ({title}): no output produced (manifest empty?)")
                status.warning(f"Slide {slide_idx} produced no output.")
            elapsed_s = (datetime.now() - started_at).total_seconds()
            avg_per_slide = (elapsed_s / completed) if completed else 0.0
            remaining = max(0, total - completed)
            eta_s = int(avg_per_slide * remaining)
            progress.progress(
                completed / total,
                text=(
                    f"{completed}/{total} slide(s) processed | "
                    f"elapsed {_format_seconds(elapsed_s)} | "
                    f"ETA {_format_seconds(eta_s)}"
                ),
            )

    progress.empty()
    status.empty()

    # Always stitch all rendered slides into one course video
    if st.session_state["manifest_video_outputs"]:
        st.info("Stitching course video...")
        ordered_paths = [
            st.session_state["manifest_video_outputs"][k]["path"]
            for k in sorted(st.session_state["manifest_video_outputs"].keys())
        ]
        course_path = os.path.join(
            run_dir,
            f"{sanitize_filename(course_name) or 'course'}_full.mp4",
        )
        try:
            stitched = render_course_video(ordered_paths, course_path)
            if stitched:
                st.session_state["manifest_video_course_path"] = stitched
                _log(f"Stitched course video → {stitched}")
            else:
                _log("Course stitching produced no output.")
        except Exception as e:
            _log(f"Course stitching failed: {e}")
            st.warning(f"Course stitching failed: {e}")

    with log_area:
        for line in st.session_state.get("manifest_video_logs", []):
            st.text(line)

    _render_existing_outputs()


def _render_existing_outputs():
    outputs = st.session_state.get("manifest_video_outputs", {})
    course_path = st.session_state.get("manifest_video_course_path")
    if not outputs and not course_path:
        return

    st.divider()
    st.subheader("Downloads")

    if course_path and os.path.exists(course_path):
        st.markdown("**Full course video**")
        try:
            with open(course_path, "rb") as f:
                st.video(f.read())
        except Exception:
            pass
        with open(course_path, "rb") as f:
            st.download_button(
                ":material/download: Download full course video",
                data=f.read(),
                file_name=os.path.basename(course_path),
                mime="video/mp4",
                use_container_width=True,
                key=f"download_course_{os.path.basename(course_path)}",
            )

    if outputs:
        st.markdown("**Per-slide videos**")
        for slide_idx in sorted(outputs.keys()):
            entry = outputs[slide_idx]
            path = entry.get("path")
            title = entry.get("title", f"Slide {slide_idx}")
            if not path or not os.path.exists(path):
                continue
            with st.expander(f"Slide {slide_idx} — {title}", expanded=False):
                try:
                    with open(path, "rb") as f:
                        st.video(f.read())
                except Exception:
                    pass
                with open(path, "rb") as f:
                    st.download_button(
                        f":material/download: Download Slide {slide_idx}",
                        data=f.read(),
                        file_name=os.path.basename(path),
                        mime="video/mp4",
                        key=f"download_slide_{slide_idx}_{os.path.basename(path)}",
                    )


render_slideshow_manifest_video_page()
