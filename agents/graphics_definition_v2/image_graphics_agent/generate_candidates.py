from concurrent.futures import ThreadPoolExecutor, as_completed
from agents.graphics_definition_v2.image_graphics_agent.drive_search import run_drive_search_for_all_rows
from agents.graphics_definition_v2.image_graphics_agent.web_search import run_web_search_for_all_rows
from agents.graphics_definition_v2.video_graphics_agent.youtube_video_search_from_queries import run_youtube_video_search_for_all_rows
from agents.graphics_definition_v2.video_graphics_agent.youtube_video_search_in_other_channels import run_youtube_video_search_other_channels_for_all_rows
from agents.graphics_definition_v2.image_graphics_agent.drive_search import delete_drive_results
from agents.graphics_definition_v2.image_graphics_agent.web_search import delete_web_results
from agents.graphics_definition_v2.video_graphics_agent.youtube_video_search_from_queries import delete_video_pool
from agents.graphics_definition_v2.video_graphics_agent.youtube_video_search_in_other_channels import delete_video_pool_other_channels
import streamlit as st
from langsmith import traceable
from services.smart_progress_bar import SmartProgressBar


@traceable(
    metadata={
        "agent_name": "graphics_definition_v2",
        "step_name": "Generate Image and Video Candidates",
        "function_name": "run_generate_image_and_video_candidates",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous")
    }
)
def run_generate_image_and_video_candidates(sheet, max_workers=50, use_only_drive_and_hvac=False):
    """
    Run candidate generation steps in parallel.
    When use_only_drive_and_hvac is False (default), runs all 4:
      1. Drive Search  2. Web Search  3. Video Search (HVAC)  4. Video Search (Other channels)
    When use_only_drive_and_hvac is True, runs only:
      1. Drive Search  2. Video Search (HVAC channels)

    :param sheet: The gspread sheet object.
    :param max_workers: Passed to each sub-step for row-level parallelism.
    :param use_only_drive_and_hvac: If True, skip Web Search and Video Search (Other Channels).
    :return: None
    """
    mode_label = "Drive + HVAC only" if use_only_drive_and_hvac else "Drive, Web, Video searches"
    print("\n" + "="*80)
    print(f"🚀 Starting parallel candidate generation ({mode_label})")
    print("="*80 + "\n")
    
    search_functions = [
        ("Drive Search", run_drive_search_for_all_rows, {"sheet": sheet, "max_workers": max_workers}),
        ("Video Search (HVAC Channels)", run_youtube_video_search_for_all_rows, {"sheet": sheet, "max_workers": max_workers}),
    ]
    if not use_only_drive_and_hvac:
        search_functions.append(("Web Search", run_web_search_for_all_rows, {"sheet": sheet, "max_workers": max_workers}))
        search_functions.append(("Video Search (Other Channels)", run_youtube_video_search_other_channels_for_all_rows, {"sheet": sheet, "max_workers": max_workers}))
    else:
        print("ℹ️  Toggle ON: Skipping Web Search and Video Search (Other Channels)")

    # One top-level progress bar for the merged step (advances as each sub-step completes).
    progress = SmartProgressBar(
        total_tasks=len(search_functions),
        description="Generate Image & Video Candidates",
        save_interval=0,
    )
    
    # Run all 4 in parallel
    with ThreadPoolExecutor(max_workers=4) as executor:
        # Submit all tasks
        futures_map = {}
        for name, func, kwargs in search_functions:
            future = executor.submit(func, **kwargs)
            futures_map[future] = name
        
        # Collect results as they complete
        completed = []
        errors = []
        
        for future in as_completed(futures_map):
            name = futures_map[future]
            try:
                future.result()  # Wait for completion
                completed.append(name)
                print(f"✅ {name} completed successfully")
            except Exception as e:
                errors.append((name, str(e)))
                print(f"❌ {name} failed: {e}")
            finally:
                # Always advance the top-level progress bar so it doesn't get stuck.
                progress.update()
    
    # Summary
    print("\n" + "="*80)
    print("📊 Parallel Candidate Generation Summary:")
    print(f"✅ Completed: {len(completed)}/{len(search_functions)}")
    if completed:
        for name in completed:
            print(f"   - {name}")
    if errors:
        print(f"❌ Errors: {len(errors)}")
        for name, error in errors:
            print(f"   - {name}: {error}")
    print("="*80 + "\n")


def delete_all_candidate_results(sheet):
    """
    Delete all candidate generation results by calling all 4 delete functions.
    
    :param sheet: The gspread sheet object.
    :return: None
    """
    print("🗑️ Deleting all candidate generation results...")
    try:
        delete_drive_results(sheet)
        delete_web_results(sheet)
        delete_video_pool(sheet)
        delete_video_pool_other_channels(sheet)
        print("✅ All candidate results deleted")
    except Exception as e:
        print(f"⚠️ Error during deletion: {e}")
