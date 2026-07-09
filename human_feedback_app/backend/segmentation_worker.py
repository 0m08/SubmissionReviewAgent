"""Run human-driven segmentation revision for a single sheet row."""

from __future__ import annotations

from human_feedback_app.backend.sessions import UserSession
from human_feedback_app.backend.streamlit_shim import agent_session_context, bind_user_session
from agents.graphics_definition_v2.review_agent.segmentation_based_reviser import (
    run_segmentation_revision_for_row,
)


def run_row_segmentation_revision(
    session: UserSession,
    row_index: int,
    feedback: str,
) -> None:
    ctx = bind_user_session(
        drive=session.drive,
        gc=session.gc,
        user_email=session.user_email,
        role=session.role,
        sheet_link=session.sheet_link,
        root_folder_id=session.root_folder_id,
    )
    with agent_session_context(ctx):
        run_segmentation_revision_for_row(session, row_index, feedback)
