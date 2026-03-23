from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue, Empty
import time
from agents.graphics_definition_v2.image_graphics_agent.image_selection_from_all_images import (
    run_image_selection_from_all_images_for_all_rows,
)
from agents.graphics_definition_v2.video_graphics_agent.video_selection_from_all_videos import (
    run_video_selection_from_all_videos_for_all_rows,
)
import streamlit as st
from langsmith import traceable
from services.smart_progress_bar import SmartProgressBar
from services.sheets_service import get_sheet_data_and_df, clear_worksheet, save_to_sheet


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
    image_pool_llm="gemini_2_5_flash_lite",
    video_pool_llm="gemini_3_flash_thinking",
    max_workers=50,
):
    """
    Run both Image Pool and Video Pool generation steps in parallel:
    1. Image Pool - selects relevant images from drive_results and web_results
    2. Video Pool - selects relevant videos from video_pool and video_pool_other_channels
    
    Each step writes to its own column independently (image_pool, video_pool_filtered),
    so they can run in parallel.
    
    :param sheet: The gspread sheet object.
    :param image_pool_llm: The LLM model to use for Image Pool generation.
    :param video_pool_llm: The LLM model to use for Video Pool generation.
    :param max_workers: Passed to image/video pool runners for row-level parallelism.
    :return: None
    """
    print("\n" + "="*80)
    print(f"🚀 Starting parallel pool generation (Image Pool, Video Pool)")
    print(f"🤖 Image Pool LLM: {image_pool_llm}")
    print(f"🤖 Video Pool LLM: {video_pool_llm}")
    print("="*80 + "\n")
    
    # Compute how many rows each pool will process
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

    total_ticks = image_tasks + video_tasks
    if image_tasks > 0:
        total_ticks += 1
    if video_tasks > 0:
        total_ticks += 1

    progress = SmartProgressBar(
        total_tasks=total_ticks,
        description="Generate Image & Video Pools (row-level)",
        save_interval=0,
    )

    # Thread-safe event channel for worker threads -> main thread progress updates
    progress_events: "Queue[int]" = Queue()

    def _emit_progress(inc: int = 1):
        try:
            progress_events.put(int(inc))
        except Exception:
            pass
    
    # Run both in parallel, but disable internal Streamlit progress bars in the sub-steps (not thread-safe).
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures_map = {
            executor.submit(
                run_image_selection_from_all_images_for_all_rows,
                sheet=sheet,
                llm=image_pool_llm,
                max_workers=max_workers,
                progress_callback=_emit_progress,
                show_progress=False,
            ): "Image Pool",
            executor.submit(
                run_video_selection_from_all_videos_for_all_rows,
                sheet=sheet,
                llm=video_pool_llm,
                max_workers=max_workers,
                progress_callback=_emit_progress,
                show_progress=False,
            ): "Video Pool",
        }

        completed = []
        errors = []

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
    
    # Summary
    print("\n" + "="*80)
    print("📊 Parallel Pool Generation Summary:")
    print(f"✅ Completed: {len(completed)}/2")
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
    columns_to_delete = ["image_pool", "video_pool_filtered"]
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
