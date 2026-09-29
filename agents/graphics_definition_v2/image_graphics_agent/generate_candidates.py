from concurrent.futures import ThreadPoolExecutor, as_completed

from agents.graphics_definition_v2.candidate_search.pool_registry import (
    COURSE_INFO_ENABLED_SOURCES_KEY,
    DRIVE_VIDEO_MODE_ALL,
    SOURCE_DISPLAY_NAMES,
    encode_enabled_sources_for_course_info,
    SOURCE_DRIVE_IMAGES,
    SOURCE_DRIVE_VIDEOS,
    SOURCE_EXTERNAL_REFERENCES,
    SOURCE_HVAC_YOUTUBE,
    SOURCE_WEB_IMAGES,
    SOURCE_YOUTUBE_OTHER_CHANNELS,
    UI_KEY_DRIVE_VIDEO_MODE,
    UI_KEY_WEB_FALLBACK_ENABLED,
    resolve_enabled_sources,
)
from agents.graphics_definition_v2.external_references.external_ref_search_from_queries import (
    delete_external_ref_pool,
    run_external_ref_search_for_all_rows,
)
from agents.graphics_definition_v2.image_graphics_agent.drive_search import (
    delete_drive_results,
    run_drive_search_for_all_rows,
)
from agents.graphics_definition_v2.image_graphics_agent.web_search import (
    delete_web_results,
    run_web_search_for_all_rows,
)
from agents.graphics_definition_v2.video_graphics_agent.drive_video_search_from_queries import (
    delete_drive_video_pool,
    run_drive_video_search_for_all_rows,
)
from agents.graphics_definition_v2.video_graphics_agent.youtube_video_search_from_queries import (
    delete_video_pool,
    run_youtube_video_search_for_all_rows,
)
from agents.graphics_definition_v2.video_graphics_agent.youtube_video_search_in_other_channels import (
    delete_video_pool_other_channels,
    run_youtube_video_search_other_channels_for_all_rows,
)
import streamlit as st
from langsmith import traceable
from services.smart_progress_bar import SmartProgressBar
from services.sheets_service import (
    clear_worksheet,
    format_worksheet,
    get_sheet_data_and_df,
    save_to_sheet,
)


def _course_info_worksheet(sheet):
    """
    Return the Course info worksheet, or None if missing.

    :param sheet: gspread Spreadsheet object.
    :return: Worksheet or None
    """
    for candidate in sheet.worksheets():
        if candidate.title.strip().lower() == "course info":
            return candidate
    return None


def _read_enabled_sources_raw_from_course_info(sheet):
    """
    Read the raw Allowed Asset Search Libraries cell from Course info.

    :param sheet: gspread Spreadsheet object.
    :return: Raw cell string ("" if missing/blank).
    """
    try:
        worksheet = _course_info_worksheet(sheet)
        if worksheet is None:
            return ""
        headers = worksheet.row_values(1)
        key = COURSE_INFO_ENABLED_SOURCES_KEY
        col = None
        for idx, header in enumerate(headers):
            if str(header).strip().lower() == key.lower():
                col = idx + 1
                break
        if col is None:
            return ""
        return str(worksheet.cell(2, col).value or "").strip()
    except Exception as exc:
        print(f"⚠️ Could not read enabled_sources from Course info: {exc}")
        return ""


def _persist_run_sources_to_course_info(sheet, sources, drive_video_mode):
    """
    Record enabled sources (+ Drive video mode) on Course info so Graphics UI and HF share one setting.

    :param sheet: gspread Spreadsheet object.
    :param sources: Resolved list of enabled source ids.
    :param drive_video_mode: Drive video mode string, or "" when Drive videos off.
    :return: None
    """
    try:
        worksheet = _course_info_worksheet(sheet)
        if worksheet is None:
            print("⚠️ 'Course info' tab not found; skipping enabled_sources persistence")
            return

        headers = worksheet.row_values(1)
        key = COURSE_INFO_ENABLED_SOURCES_KEY
        col = None
        created_column = False
        for idx, header in enumerate(headers):
            if str(header).strip().lower() == key.lower():
                col = idx + 1
                break
        if col is None:
            headers.append(key)
            col = len(headers)
            if col > worksheet.col_count:
                worksheet.add_cols(col - worksheet.col_count)
            worksheet.update_cell(1, col, key)
            created_column = True

        cell_value = encode_enabled_sources_for_course_info(sources, drive_video_mode)
        worksheet.update_cell(2, col, cell_value)
        if created_column:
            format_worksheet(worksheet)
        print(f"📝 Persisted enabled_sources to Course info: {cell_value}")
    except Exception as exc:  # never break the pipeline over metadata persistence
        print(f"⚠️ Could not persist enabled_sources to Course info: {exc}")


def _delete_enabled_sources_from_course_info(sheet):
    """
    Remove the Allowed Asset Search Libraries column from Course info.

    :param sheet: gspread Spreadsheet object.
    :return: None
    """
    try:
        worksheet = None
        for candidate in sheet.worksheets():
            if candidate.title.strip().lower() == "course info":
                worksheet = candidate
                break
        if worksheet is None:
            print("ℹ️ 'Course info' tab not found; nothing to clear for enabled_sources")
            return

        ws, df = get_sheet_data_and_df(sheet, worksheet.title)
        key = COURSE_INFO_ENABLED_SOURCES_KEY
        match_cols = [c for c in df.columns if str(c).strip().lower() == key.lower()]
        if not match_cols:
            print(f"ℹ️ No '{key}' column on Course info")
            return
        df = df.drop(columns=match_cols)
        clear_worksheet(ws)
        save_to_sheet(ws, df)
        format_worksheet(ws)
        print(f"🗑️ Deleted '{key}' from Course info")
    except Exception as exc:
        print(f"⚠️ Could not delete enabled_sources from Course info: {exc}")


# Sheet-level runners + delete helpers keyed by source id.
POOL_RUNNERS = {
    SOURCE_DRIVE_IMAGES: (run_drive_search_for_all_rows, delete_drive_results),
    SOURCE_HVAC_YOUTUBE: (run_youtube_video_search_for_all_rows, delete_video_pool),
    SOURCE_DRIVE_VIDEOS: (run_drive_video_search_for_all_rows, delete_drive_video_pool),
    SOURCE_EXTERNAL_REFERENCES: (
        run_external_ref_search_for_all_rows,
        delete_external_ref_pool,
    ),
    SOURCE_WEB_IMAGES: (run_web_search_for_all_rows, delete_web_results),
    SOURCE_YOUTUBE_OTHER_CHANNELS: (
        run_youtube_video_search_other_channels_for_all_rows,
        delete_video_pool_other_channels,
    ),
}


@traceable(
    metadata={
        "agent_name": "graphics_definition_v2",
        "step_name": "Generate Image and Video Candidates",
        "function_name": "run_generate_image_and_video_candidates",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous"),
    }
)
def run_generate_image_and_video_candidates(sheet, max_workers=50, use_only_drive_and_hvac=None, enabled_sources=None, selected_topics=None):
    """
    Run enabled candidate-search pool runners in parallel.

    Pool selection:
      - enabled_sources: explicit list of source ids, if provided
      - else use_only_drive_and_hvac:
          True / None -> drive_images + hvac_youtube + drive_videos
          False       -> all known pools (adds web + other YouTube)

    Sheet-level runners still own column write/validate/retry. Query execution inside each pool goes through the shared search wrapper.

    :param sheet: The gspread sheet object.
    :param max_workers: Passed to each sub-step for row-level parallelism.
    :param use_only_drive_and_hvac: Legacy boolean mapped to enabled sources.
    :param enabled_sources: Optional explicit source id list.
    :param selected_topics: Optional topic filter passed to runners.
    :return: None
    """
    if enabled_sources is None or isinstance(enabled_sources, str):

        session_sources = st.session_state.get("graphics_v2_enabled_sources")
        if isinstance(session_sources, (list, tuple)):
            enabled_sources = list(session_sources)
        else:
            enabled_sources = None

    sources = resolve_enabled_sources(
        enabled_sources=enabled_sources,
        use_only_drive_and_hvac=use_only_drive_and_hvac,
    )
    if not sources:
        print("⚠️ No candidate sources enabled; skipping candidate generation")
        return

    _drive_video_mode = (
        st.session_state.get(UI_KEY_DRIVE_VIDEO_MODE, DRIVE_VIDEO_MODE_ALL)
        if SOURCE_DRIVE_VIDEOS in sources
        else ""
    )
    # Persist primary Section 5 sources plus optional web/other when that UI toggle is on. Do not add web/other into `sources` itself — those are Section 9 fallbacks, not Section 5 runners.
    persisted_sources = list(sources)
    if st.session_state.get(UI_KEY_WEB_FALLBACK_ENABLED, False):
        for optional_id in (SOURCE_WEB_IMAGES, SOURCE_YOUTUBE_OTHER_CHANNELS):
            if optional_id not in persisted_sources:
                persisted_sources.append(optional_id)
    _persist_run_sources_to_course_info(sheet, persisted_sources, _drive_video_mode)

    mode_label = " + ".join(SOURCE_DISPLAY_NAMES.get(s, s) for s in sources)
    print("\n" + "=" * 80)
    print(f"🚀 Starting parallel candidate generation ({mode_label})")
    print("=" * 80 + "\n")

    search_functions = []
    for source_id in sources:
        runner_pair = POOL_RUNNERS.get(source_id)
        if not runner_pair:
            print(f"⚠️ No sheet runner registered for source: {source_id}")
            continue
        run_fn, _delete_fn = runner_pair
        display_name = SOURCE_DISPLAY_NAMES.get(source_id, source_id)
        kwargs = {
            "sheet": sheet,
            "max_workers": max_workers,
            "selected_topics": selected_topics,
        }
        # Resolve Drive video mode on the main thread; worker threads cannot read session_state.
        if source_id == SOURCE_DRIVE_VIDEOS:
            kwargs["drive_video_mode"] = st.session_state.get(
                UI_KEY_DRIVE_VIDEO_MODE, DRIVE_VIDEO_MODE_ALL
            )
        search_functions.append((display_name, run_fn, kwargs))

    if not search_functions:
        print("⚠️ No runnable candidate sources after registry lookup")
        return

    progress = SmartProgressBar(
        total_tasks=len(search_functions),
        description="Generate Image & Video Candidates",
        save_interval=0,
    )

    print(f"🚀 Submitting {len(search_functions)} candidate search in parallel")
    for name, _, _ in search_functions:
        print(f"   - {name}")

    with ThreadPoolExecutor(max_workers=len(search_functions)) as executor:
        futures_map = {}
        for name, func, kwargs in search_functions:
            future = executor.submit(func, **kwargs)
            futures_map[future] = name

        completed = []
        errors = []

        for future in as_completed(futures_map):
            name = futures_map[future]
            try:
                future.result()
                completed.append(name)
                print(f"✅ {name} completed successfully")
            except Exception as e:
                errors.append((name, str(e)))
                print(f"❌ {name} failed: {e}")
            finally:
                progress.update()

    print("\n" + "=" * 80)
    print("📊 Parallel Candidate Generation Summary:")
    print(f"✅ Completed: {len(completed)}/{len(search_functions)}")
    if completed:
        for name in completed:
            print(f"   - {name}")
    if errors:
        print(f"❌ Errors: {len(errors)}")
        for name, error in errors:
            print(f"   - {name}: {error}")
    print("=" * 80 + "\n")


def delete_all_candidate_results(sheet):
    """
    Delete all known candidate generation result columns (pools/search results).

    :param sheet: The gspread sheet object.
    :return: None
    """
    print("🗑️ Deleting all candidate generation results...")
    try:
        for source_id, (_run_fn, delete_fn) in POOL_RUNNERS.items():
            delete_fn(sheet)
        print("✅ All candidate results deleted")
    except Exception as e:
        print(f"⚠️ Error during deletion: {e}")
