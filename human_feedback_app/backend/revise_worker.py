"""Run human-feedback revision for a single sheet row / visual."""

from __future__ import annotations

from typing import Optional

import pandas as pd

from human_feedback_app.backend.config import LLM_DEFAULT
from human_feedback_app.backend.sheet_service import (
    ROUND_INDEX,
    extract_asset_url,
    filter_feedback_to_vo,
    load_workbook,
    merge_visual_revision,
    save_row_cells,
)
from human_feedback_app.backend.sessions import UserSession
from human_feedback_app.backend.streamlit_shim import agent_session_context, bind_user_session
from agents.graphics_definition_v2.review_agent.human_feedback_based_review_and_revise import (
    _parse_tracking_column,
    process_human_feedback_row,
)
from agents.graphics_definition_v2.external_references.external_reference_extraction import (
    _resolve_sheet_id,
)
from human_feedback_app.backend.constants import (
    HUMAN_FEEDBACK_COLUMN,
    HUMAN_FEEDBACK_STATUS_COLUMN,
    HUMAN_FEEDBACK_TRACKING_COLUMN,
    HUMAN_REVIEW_ACTIONS_COLUMN,
)
from graphics_definition_v2_slideshow import get_round_column_name
from services.sheets_service import get_sheet_data_and_df


def _cell_to_str(value) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def run_row_revision(
    session: UserSession,
    row_index: int,
    target_vo: str = "",
    segment_index: Optional[int] = None,
    step_index: Optional[int] = None,
    visual_id: str = "",
) -> None:
    hf_col = get_round_column_name(HUMAN_FEEDBACK_COLUMN, ROUND_INDEX)
    status_col = get_round_column_name(HUMAN_FEEDBACK_STATUS_COLUMN, ROUND_INDEX)
    tracking_col = get_round_column_name(HUMAN_FEEDBACK_TRACKING_COLUMN, ROUND_INDEX)
    actions_col = get_round_column_name(HUMAN_REVIEW_ACTIONS_COLUMN, ROUND_INDEX)

    _, df, _ = load_workbook(session)

    # Backward-compat fallback for sheets created before the graphics_v2_enabled_sources column existed (and any empty/malformed value): treat as None so the agent behaves exactly as it did before the column — all present sources are searchable during revise.
    enabled_sources = session.enabled_sources or None
    drive_video_mode = session.drive_video_mode or None
    sheet_id = _resolve_sheet_id(session.sheet) if session.sheet is not None else ""

    _, course_info_df = get_sheet_data_and_df(session.sheet, "Course info")
    course_name = str(course_info_df.iloc[0].get("Course Name", "")).strip()
    target_audience = str(course_info_df.iloc[0].get("Target Audience & Industry", "")).strip()

    before_row = df.loc[row_index].copy()

    scoped_feedback = filter_feedback_to_vo(_cell_to_str(before_row.get(hf_col)), target_vo)
    if scoped_feedback:
        df.at[row_index, hf_col] = scoped_feedback

    ctx = bind_user_session(
        drive=session.drive,
        gc=session.gc,
        user_email=session.user_email,
        role=session.role,
        sheet_link=session.sheet_link,
        root_folder_id=session.root_folder_id,
    )
    with agent_session_context(ctx):
        # ws=None: the agent mutates df in memory but writes nothing to the sheet.
        process_human_feedback_row(
            row_index=row_index,
            df=df,
            course_name=course_name,
            target_audience=target_audience,
            drive=session.drive,
            llm=LLM_DEFAULT,
            ws=None,
            use_only_drive_and_hvac=False,
            enabled_sources=enabled_sources,
            drive_video_mode=drive_video_mode,
            sheet_id=sheet_id or None,
            human_feedback_column=hf_col,
            human_feedback_status_column=status_col,
            human_feedback_revision_tracking_column=tracking_col,
            human_review_actions_column=actions_col,
        )

    after_row = df.loc[row_index]
    status_value = _cell_to_str(after_row.get(status_col))

    # Preferred path: surgically merge ONLY this visual's result into the latest sheet so a parallel same-slide revision is never clobbered.
    if segment_index is not None and step_index is not None and visual_id:
        final_url = extract_asset_url(
            _cell_to_str(after_row.get("final_graphics_definition")), segment_index, step_index
        )
        tracking_map = _parse_tracking_column(_cell_to_str(after_row.get(tracking_col)))
        tracking_entry = tracking_map.get(visual_id)
        original_url = ""
        if isinstance(tracking_entry, dict):
            original_url = _cell_to_str(tracking_entry.get("original"))
        merge_visual_revision(
            session,
            row_index=row_index,
            segment_index=segment_index,
            step_index=step_index,
            visual_id=visual_id,
            final_url=final_url,
            original_url=original_url,
            tracking_entry=tracking_entry,
            status_value=status_value,
        )
        return

    # Fallback (no visual identifiers): persist the row-level cells the agent changed, excluding the feedback cell we narrowed in memory.
    changed = {
        col: _cell_to_str(after_row.get(col))
        for col in df.columns
        if col != hf_col
        and _cell_to_str(before_row.get(col)) != _cell_to_str(after_row.get(col))
    }
    save_row_cells(session, row_index, changed)
