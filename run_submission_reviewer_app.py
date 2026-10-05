# -*- coding: utf-8 -*-
"""
Standalone Streamlit Runner for Submission Reviewer & Central Review Registry.

Run via:
    .venv/bin/streamlit run run_submission_reviewer_app.py
"""

import streamlit as st
from agents.submission_reviewer.submission_reviewer_ui import render_submission_reviewer_ui

st.set_page_config(
    page_title="Moodle Submission Reviewer & Registry",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded",
)

render_submission_reviewer_ui()
