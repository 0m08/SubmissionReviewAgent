"""Run human-feedback revision for a single sheet row."""

from __future__ import annotations

from human_feedback_app.backend.config import LLM_DEFAULT
from human_feedback_app.backend.sheet_service import load_workbook
from human_feedback_app.backend.sessions import UserSession
from human_feedback_app.backend.streamlit_shim import agent_session_context, bind_user_session


def run_row_revision(session: UserSession, row_index: int) -> None:
    from agents.graphics_definition_v2.review_agent.human_feedback_based_review_and_revise import (
        process_human_feedback_row,
    )
    from human_feedback_app.backend.constants import (
        HUMAN_FEEDBACK_COLUMN,
        HUMAN_FEEDBACK_STATUS_COLUMN,
        HUMAN_FEEDBACK_TRACKING_COLUMN,
        HUMAN_REVIEW_ACTIONS_COLUMN,
    )
    from graphics_definition_v2_slideshow import get_round_column_name
    from services.sheets_service import get_sheet_data_and_df

    ws, df, current_round = load_workbook(session)
    _, course_info_df = get_sheet_data_and_df(session.sheet, "Course info")
    course_name = str(course_info_df.iloc[0].get("Course Name", "")).strip()
    target_audience = str(course_info_df.iloc[0].get("Target Audience & Industry", "")).strip()

    hf_col = get_round_column_name(HUMAN_FEEDBACK_COLUMN, current_round)
    status_col = get_round_column_name(HUMAN_FEEDBACK_STATUS_COLUMN, current_round)
    tracking_col = get_round_column_name(HUMAN_FEEDBACK_TRACKING_COLUMN, current_round)
    actions_col = get_round_column_name(HUMAN_REVIEW_ACTIONS_COLUMN, current_round)

    ctx = bind_user_session(
        drive=session.drive,
        gc=session.gc,
        user_email=session.user_email,
        role=session.role,
        sheet_link=session.sheet_link,
        root_folder_id=session.root_folder_id,
    )
    with agent_session_context(ctx):
        process_human_feedback_row(
            row_index=row_index,
            df=df,
            course_name=course_name,
            target_audience=target_audience,
            drive=session.drive,
            llm=LLM_DEFAULT,
            ws=ws,
            use_only_drive_and_hvac=False,
            human_feedback_column=hf_col,
            human_feedback_status_column=status_col,
            human_feedback_revision_tracking_column=tracking_col,
            human_review_actions_column=actions_col,
        )
