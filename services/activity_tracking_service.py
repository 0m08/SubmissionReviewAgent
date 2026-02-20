"""
User Activity Tracking Service

Logs login and page navigation events to Google Sheets in real-time.
Events are logged asynchronously to avoid blocking the UI.
"""

import threading
from datetime import datetime
import gspread
from config.logging_config import get_logger

logger = get_logger(__name__)

# Event type constants
EVENT_LOGIN = "login"
EVENT_PAGE_VIEW = "page_view"

# Tracking configuration
TRACKING_SHEET_URL = "https://docs.google.com/spreadsheets/d/1dWfoOpnwQXwElDoW2Dz_MMRDqLZGDgnltSLjOces6Ls/edit"
TRACKING_WORKSHEET_NAME = "Records"


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
