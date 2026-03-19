"""
User Activity Tracking Service

Logs login, page navigation, and agent step events to Google Sheets in
real-time.  Events are logged asynchronously to avoid blocking the UI.
"""

import threading
from datetime import datetime
import gspread
from config.logging_config import get_logger

logger = get_logger(__name__)

# Event type constants — existing
EVENT_LOGIN = "login"
EVENT_PAGE_VIEW = "page_view"

# Event type constants — step tracking
EVENT_STEP_START = "step_start"
EVENT_STEP_COMPLETE = "step_complete"
EVENT_STEP_ERROR = "step_error"
EVENT_RUN_ALL_START = "run_all_start"
EVENT_RUN_IN_BACKGROUND_START = "run_in_background_start"

# Tracking configuration
TRACKING_SHEET_URL = "https://docs.google.com/spreadsheets/d/1dWfoOpnwQXwElDoW2Dz_MMRDqLZGDgnltSLjOces6Ls/edit"
TRACKING_WORKSHEET_NAME = "Records"
STEP_TRACKING_WORKSHEET_NAME = "Step Tracking"

STEP_TRACKING_HEADERS = [
    "Timestamp", "User Email", "Event Type", "Agent Name",
    "Step Name", "Course Name", "Sheet Link",
    "Duration Seconds", "Error Message", "Run Mode",
]


def _get_tracking_worksheet(gc: gspread.Client, sheet_url: str) -> gspread.Worksheet:
    """
    Get the tracking worksheet from the specified Google Sheet.

    Args:
        gc: Authenticated gspread client
        sheet_url: URL of the Google Sheet for tracking

    Returns:
        gspread Worksheet object
    """
    sheet = gc.open_by_url(sheet_url)
    return sheet.worksheet(TRACKING_WORKSHEET_NAME)


def _append_tracking_row(worksheet: gspread.Worksheet, row_data: list) -> None:
    """
    Append a single row to the tracking worksheet.

    Args:
        worksheet: gspread Worksheet object
        row_data: List of values to append [Timestamp, User Email, Event Type, Page Name]
    """
    worksheet.append_row(row_data, value_input_option='USER_ENTERED')


def log_activity_async(
    gc: gspread.Client,
    tracking_sheet_url: str,
    user_email: str,
    event_type: str,
    page_name: str = ""
) -> None:
    """
    Log an activity event asynchronously (non-blocking).

    Writes to Google Sheets in a background thread so the main app
    is not blocked by network latency.

    Args:
        gc: Authenticated gspread client
        tracking_sheet_url: URL of the Google Sheet for tracking
        user_email: User's email address
        event_type: Either "login" or "page_view"
        page_name: Name of the page (for page_view events)
    """
    def _log_task():
        try:
            worksheet = _get_tracking_worksheet(gc, tracking_sheet_url)

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            row_data = [timestamp, user_email, event_type, page_name]

            _append_tracking_row(worksheet, row_data)
            logger.debug(f"Logged activity: {event_type} for {user_email} on {page_name}")

        except Exception as e:
            # Log error but don't crash the app
            logger.error(f"Failed to log activity: {e}")

    # Run in background thread
    thread = threading.Thread(target=_log_task, daemon=True)
    thread.start()


def track_page_view(
    gc: gspread.Client,
    user_email: str,
    current_page_title: str
) -> None:
    """
    Track a page view if the page has changed since last check.

    This function checks st.session_state to avoid duplicate logging
    when Streamlit reruns the script.

    Args:
        gc: Authenticated gspread client
        user_email: User's email address
        current_page_title: Title of the current page
    """
    import streamlit as st

    last_tracked_page = st.session_state.get("_last_tracked_page", None)

    if current_page_title != last_tracked_page:
        st.session_state["_last_tracked_page"] = current_page_title
        log_activity_async(
            gc=gc,
            tracking_sheet_url=TRACKING_SHEET_URL,
            user_email=user_email,
            event_type=EVENT_PAGE_VIEW,
            page_name=current_page_title
        )


##############################################################################
# Step-level tracking — writes to the "Step Tracking" tab
##############################################################################

def _get_step_tracking_worksheet(gc: gspread.Client, sheet_url: str) -> gspread.Worksheet:
    """
    Return the Step Tracking worksheet, creating it (with headers) on first use.
    """
    sheet = gc.open_by_url(sheet_url)
    try:
        return sheet.worksheet(STEP_TRACKING_WORKSHEET_NAME)
    except gspread.exceptions.WorksheetNotFound:
        try:
            ws = sheet.add_worksheet(STEP_TRACKING_WORKSHEET_NAME, rows=2000, cols=len(STEP_TRACKING_HEADERS))
            ws.append_row(STEP_TRACKING_HEADERS, value_input_option="USER_ENTERED")
            return ws
        except Exception:
            # Race condition: another thread created it first
            return sheet.worksheet(STEP_TRACKING_WORKSHEET_NAME)


def log_step_event_async(
    gc: gspread.Client,
    user_email: str,
    event_type: str,
    agent_name: str,
    step_name: str = "",
    course_name: str = "",
    sheet_link: str = "",
    duration_seconds: float = None,
    error_message: str = "",
    run_mode: str = "",
) -> None:
    """
    Log a step-level event asynchronously to the central tracking sheet.
    """
    def _log_task():
        try:
            worksheet = _get_step_tracking_worksheet(gc, TRACKING_SHEET_URL)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            row_data = [
                timestamp,
                user_email,
                event_type,
                agent_name,
                step_name,
                course_name,
                sheet_link,
                str(round(duration_seconds, 2)) if duration_seconds is not None else "",
                error_message,
                run_mode,
            ]
            worksheet.append_row(row_data, value_input_option="USER_ENTERED")
            logger.debug(f"Logged step event: {event_type} for {step_name} ({agent_name})")
        except Exception as e:
            logger.error(f"Failed to log step event: {e}")

    thread = threading.Thread(target=_log_task, daemon=True)
    thread.start()


# ── Public convenience functions for step tracking ──────────────────────────

def track_step_start(gc, user_email, agent_name, step_name,
                     course_name="", sheet_link="", run_mode=""):
    log_step_event_async(gc, user_email, EVENT_STEP_START,
                         agent_name, step_name, course_name,
                         sheet_link, run_mode=run_mode)


def track_step_complete(gc, user_email, agent_name, step_name,
                        course_name="", sheet_link="",
                        duration_seconds=None, run_mode=""):
    log_step_event_async(gc, user_email, EVENT_STEP_COMPLETE,
                         agent_name, step_name, course_name, sheet_link,
                         duration_seconds=duration_seconds, run_mode=run_mode)


def track_step_error(gc, user_email, agent_name, step_name,
                     course_name="", sheet_link="",
                     error_message="", run_mode=""):
    log_step_event_async(gc, user_email, EVENT_STEP_ERROR,
                         agent_name, step_name, course_name, sheet_link,
                         error_message=error_message, run_mode=run_mode)


def track_run_all_start(gc, user_email, agent_name,
                        course_name="", sheet_link=""):
    log_step_event_async(gc, user_email, EVENT_RUN_ALL_START,
                         agent_name, step_name="",
                         course_name=course_name, sheet_link=sheet_link,
                         run_mode="automated")


def track_run_in_background_start(gc, user_email, agent_name,
                                  course_name="", sheet_link=""):
    log_step_event_async(gc, user_email, EVENT_RUN_IN_BACKGROUND_START,
                         agent_name, step_name="",
                         course_name=course_name, sheet_link=sheet_link,
                         run_mode="background")


##############################################################################
# High-level convenience — auto-reads context from Streamlit session state
##############################################################################

EVENT_TOOL_ACTION = "tool_action"

def track_tool_action(
    tool_name: str,
    action_name: str,
    run_mode: str = "tool",
    duration_seconds: float = None,
    error_message: str = "",
    course_name: str = None,
    sheet_link: str = None,
):
    """
    Track a standalone tool/agent action.  Reads gc / user_email from
    st.session_state automatically so callers only need tool + action names.

    Args:
        tool_name: Display name of the tool or agent (e.g. "Paraphraser",
                   "Template Sheet Setup").
        action_name: Specific action taken (e.g. "paraphrase_text",
                     "create_vectorstore").
        run_mode: "tool" for standalone tools, "agent" for standalone agents.
        duration_seconds: Optional elapsed time for the action.
        error_message: If non-empty, event is logged as step_error instead.
        course_name: Override course name. Pass "" to force blank.
                     None (default) reads from session state.
        sheet_link: Override sheet link. Pass "" to force blank.
                    None (default) reads from session state.

    Silently no-ops if not logged in.
    """
    import streamlit as st

    gc = st.session_state.get("gc")
    if gc is None:
        return
    user_email = st.session_state.get("user_email", "")
    course_name = st.session_state.get("course_name", "") if course_name is None else course_name
    sheet_link = st.session_state.get("sheet_link", "") if sheet_link is None else sheet_link

    event_type = EVENT_TOOL_ACTION if not error_message else EVENT_STEP_ERROR
    log_step_event_async(
        gc, user_email, event_type,
        agent_name=tool_name,
        step_name=action_name,
        course_name=course_name,
        sheet_link=sheet_link,
        duration_seconds=duration_seconds,
        error_message=error_message,
        run_mode=run_mode,
    )


##############################################################################
# Login / page-view tracking — existing (writes to "Records" tab)
##############################################################################

def track_login(
    gc: gspread.Client,
    user_email: str
) -> None:
    """
    Track a login event.

    Args:
        gc: Authenticated gspread client
        user_email: User's email address
    """
    log_activity_async(
        gc=gc,
        tracking_sheet_url=TRACKING_SHEET_URL,
        user_email=user_email,
        event_type=EVENT_LOGIN,
        page_name="Login"
    )
