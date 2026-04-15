from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue, Empty
import time
from agents.graphics_definition_v2.image_graphics_agent.image_selection_from_all_images import (
    run_image_selection_from_all_images_for_all_rows,
    run_image_scoring_for_all_rows,
)
from agents.graphics_definition_v2.video_graphics_agent.video_selection_from_all_videos import (
    run_video_selection_from_all_videos_for_all_rows,
    run_video_scoring_for_all_rows,
)
import streamlit as st
from langsmith import traceable
from services.smart_progress_bar import SmartProgressBar
from services.sheets_service import get_sheet_data_and_df, clear_worksheet, save_to_sheet


def run_parallel_pair_with_progress(sheet, max_workers, progress, progress_events, emit_progress, completed, errors, label_a, fn_a, llm_a, label_b, fn_b, llm_b):
    """
    Run two row-parallel pipeline functions concurrently and stream progress events.

    :param sheet: gspread sheet object
    :param max_workers: Row-level workers passed to both functions
    :param progress: SmartProgressBar instance
    :param progress_events: Queue receiving progress increments
    :param emit_progress: Callback passed into child functions
    :param completed: Mutable list to collect completed labels
    :param errors: Mutable list to collect (label, error) tuples
    :param label_a: Display label for function A
    :param fn_a: Function A
    :param llm_a: Model for function A
    :param label_b: Display label for function B
    :param fn_b: Function B
    :param llm_b: Model for function B
    :return: None
    """
    
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures_map = {
            executor.submit(
                fn_a,
                sheet=sheet,
                llm=llm_a,
                max_workers=max_workers,
                progress_callback=emit_progress,
                show_progress=False,
            ): label_a,
            executor.submit(
                fn_b,
                sheet=sheet,
                llm=llm_b,
                max_workers=max_workers,
                progress_callback=emit_progress,
                show_progress=False,
            ): label_b,
        }
        while True:
            try:
                inc = progress_events.get(timeout=0.2)
                progress.update(increment=inc)
            except Empty:
                pass
            all_done = all(f.done() for f in futures_map.keys())
            if all_done and progress_events.empty():
                break
        for future in as_completed(futures_map):
            name = futures_map[future]
            try:
                future.result()
                completed.append(name)
                print(f"✅ {name} completed successfully")
            except Exception as e:
                errors.append((name, str(e)))
                print(f"❌ {name} failed: {e}")


@traceable(
    metadata={
        "agent_name": "graphics_definition_v2",
        "step_name": "Generate Image and Video Pools",
        "function_name": "run_generate_image_and_video_pools",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous")
    }
)
def run_generate_image_and_video_pools(
    sheet,
    image_pool_llm="gemini_3_flash_thinking",
    video_pool_llm="gemini_3_flash_thinking",
    max_workers=50,
):
    """
    Run both Image and Video scoring/filtering with two phases:
    1) Score candidates into image_score and video_score
    2) Filter shortlisted candidates into image_pool and video_pool_filtered
    
    Each step writes to its own column independently (image_pool, video_pool_filtered),
    so they can run in parallel.
    
    :param sheet: The gspread sheet object.
    :param image_pool_llm: The LLM model to use for Image Pool generation.
    :param video_pool_llm: The LLM model to use for Video Pool generation.
    :param max_workers: Passed to image/video pool runners for row-level parallelism.
    :return: None
    """
    print("\n" + "="*80)
    print(f"🚀 Starting parallel scoring + pool generation")
    print(f"🤖 Image Pool LLM: {image_pool_llm}")
    print(f"🤖 Video Pool LLM: {video_pool_llm}")
    print("="*80 + "\n")
    
    # Compute how many rows each phase will process
    worksheet_name = "Slide Chunks"
    _, df = get_sheet_data_and_df(sheet, worksheet_name)

    def _count_image_tasks(_df) -> int:
        tasks = 0
        for _, row in _df.iterrows():
            voiceover_segments = str(row.get("voiceover_segment", "")).strip()
            image_pool = str(row.get("image_pool", "")).strip()
            if not voiceover_segments or voiceover_segments == "nan":
                continue
            if image_pool and image_pool != "nan" and not str(image_pool).startswith("ERROR:"):
                continue
            tasks += 1
        return tasks

    def _count_video_tasks(_df) -> int:
        tasks = 0
        for _, row in _df.iterrows():
            video_pool = str(row.get("video_pool", "")).strip()
            video_pool_other_channels = str(row.get("video_pool_other_channels", "")).strip()
            video_pool_filtered = str(row.get("video_pool_filtered", "")).strip()
            if (not video_pool or video_pool == "nan") and (not video_pool_other_channels or video_pool_other_channels == "nan"):
                continue
            if video_pool_filtered and video_pool_filtered != "nan" and not str(video_pool_filtered).startswith("ERROR:"):
                continue
            tasks += 1
        return tasks

    image_tasks = _count_image_tasks(df)
    video_tasks = _count_video_tasks(df)

    total_ticks = (image_tasks + video_tasks) * 2
    if image_tasks > 0:
        total_ticks += 2
    if video_tasks > 0:
        total_ticks += 2

    progress = SmartProgressBar(
        total_tasks=total_ticks,
        description="Generate Image & Video Scores/Pools (row-level)",
        save_interval=0,
    )

    # Thread-safe event channel for worker threads -> main thread progress updates
    progress_events: "Queue[int]" = Queue()

    def _emit_progress(inc: int = 1):
        try:
            progress_events.put(int(inc))
        except Exception:
            pass
    
    completed = []
    errors = []

    print("🔹 Phase 1: scoring candidates (image_score + video_score)")
    run_parallel_pair_with_progress(
        sheet=sheet,
        max_workers=max_workers,
        progress=progress,
        progress_events=progress_events,
        emit_progress=_emit_progress,
        completed=completed,
        errors=errors,
        label_a="Image Scoring",
        fn_a=run_image_scoring_for_all_rows,
        llm_a=image_pool_llm,
        label_b="Video Scoring",
        fn_b=run_video_scoring_for_all_rows,
        llm_b=video_pool_llm,
    )

    print("🔹 Phase 2: filtering shortlisted candidates (image_pool + video_pool_filtered)")
    run_parallel_pair_with_progress(
        sheet=sheet,
        max_workers=max_workers,
        progress=progress,
        progress_events=progress_events,
        emit_progress=_emit_progress,
        completed=completed,
        errors=errors,
        label_a="Image Pool",
        fn_a=run_image_selection_from_all_images_for_all_rows,
        llm_a=image_pool_llm,
        label_b="Video Pool",
        fn_b=run_video_selection_from_all_videos_for_all_rows,
        llm_b=video_pool_llm,
    )
    
    # Summary
    print("\n" + "="*80)
    print("📊 Parallel Pool Generation Summary:")
    print(f"✅ Completed: {len(completed)}/4")
    if completed:
        for name in completed:
            print(f"   - {name}")
    if errors:
        print(f"❌ Errors: {len(errors)}")
        for name, error in errors:
            print(f"   - {name}: {error}")
    print("="*80 + "\n")


def delete_all_pool_results(sheet):
    """
    Delete all pool generation results by dropping the pool columns from Slide Chunks.
    
    :param sheet: The gspread sheet object.
    :return: None
    """
    worksheet_name = "Slide Chunks"
    columns_to_delete = ["image_score", "video_score", "image_pool", "video_pool_filtered"]
    print("🗑️ Deleting all pool generation results...")
    try:
        ws, df = get_sheet_data_and_df(sheet, worksheet_name)
        to_drop = [c for c in columns_to_delete if c in df.columns]
        if not to_drop:
            print(f"ℹ️ None of {columns_to_delete} exist in '{worksheet_name}'")
            return
        df = df.drop(columns=to_drop)
        clear_worksheet(ws)
        save_to_sheet(ws, df)
        print(f"✅ Deleted columns from '{worksheet_name}': {to_drop}")
    except Exception as e:
        print(f"⚠️ Error during deletion: {e}")
