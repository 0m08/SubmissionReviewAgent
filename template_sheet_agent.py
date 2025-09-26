import streamlit as st
from dotenv import load_dotenv
import os
from services.drive_service import login_with_oauth2
from services.helper_functions import get_short_name
from agents.template_sheet_setup.setup_template_sheets import setup_template_sheets

def template_sheet_setup_agent():
    """Agent for setting up template sheets for a course"""
    st.title("Template Sheet Setup Agent")
    
    # Input fields
    drive_folder_id = st.text_input("Enter course Drive folder ID:")
    course_name = st.text_input("Enter Course Name:")
    
    if st.button("Setup Template Sheets"):
        if not drive_folder_id or not course_name:
            st.error("Please provide both Drive folder ID and course name.")
            return
        
        try:
            load_dotenv()  # Load environment variables
            
            # Get authenticated clients from session state
            if "drive" in st.session_state and "gc" in st.session_state:
                drive = st.session_state["drive"]
                gc = st.session_state["gc"]
            else:
                st.error("Please authenticate with Google Drive and Sheets first.")
                return
            
            # Setup template sheets
            with st.spinner("Setting up template sheets..."):
                result = setup_template_sheets(drive, gc, drive_folder_id, course_name)
            
            # Display results
            st.success("Template sheets setup completed!")
            
        except Exception as e:
            st.error(f"Error setting up template sheets: {e}")

template_sheet_setup_agent()