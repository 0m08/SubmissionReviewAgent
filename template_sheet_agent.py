import time
import streamlit as st
from dotenv import load_dotenv
from agents.template_sheet_setup.setup_template_sheets import setup_template_sheets, TEMPLATE_COURSE_SHEET_LINK
from services.sheets_service import get_sheet_data_and_df
from services.activity_tracking_service import track_tool_action
import re

def get_template_values(gc, template_link):
    """Get original values from template sheet"""
    match = re.search(r'/d/([a-zA-Z0-9_-]+)', template_link)
    if match:
        template_id = match.group(1)
        template_sheet = gc.open_by_key(template_id)
        course_info_sheet, course_info_df = get_sheet_data_and_df(template_sheet, 'Course info')
        
        template_values = {}
        for field in ['Course Name', 'Course Background', 'Course Objective Guidelines']:
            if field in course_info_df.columns:
                template_values[field] = course_info_df[field][0] if len(course_info_df[field]) > 0 else ''
        
        try:
            base_outline_sheet, base_outline_df = get_sheet_data_and_df(template_sheet, 'Base Outline')
            template_values['base_outline_df'] = base_outline_df
        except:
            template_values['base_outline_df'] = None
            
        return template_values
    return {}

def validate_manual_review(gc, course_sheet_id):
    """Validate that user has made required changes to template sheets"""
    try:
        template_values = get_template_values(gc, TEMPLATE_COURSE_SHEET_LINK)
        
        course_sheet = gc.open_by_key(course_sheet_id)
        course_info_sheet, course_info_df = get_sheet_data_and_df(course_sheet, 'Course info')
        
        unchanged_fields = []
        
        for field in ['Course Name', 'Course Background', 'Course Objective Guidelines']:
            current_value = course_info_df.get(field, [''])[0] if field in course_info_df.columns else ''
            template_value = template_values.get(field, '')
            
            if not current_value or current_value == template_value:
                unchanged_fields.append(field)
        
        try:
            base_outline_sheet, base_outline_df = get_sheet_data_and_df(course_sheet, 'Base Outline')
            template_base_outline_df = template_values.get('base_outline_df')
            
            # Check if Base Outline content has changed
            if template_base_outline_df is not None:
                # Compare dataframes
                if base_outline_df.equals(template_base_outline_df):
                    unchanged_fields.append('Base Outline')
            else:
                # If no template Base Outline, check if current has content
                if base_outline_df.empty or len(base_outline_df) <= 1:
                    unchanged_fields.append('Base Outline')
        except:
            unchanged_fields.append('Base Outline')
        
        return unchanged_fields
        
    except Exception as e:
        raise Exception(f"Error validating manual review: {str(e)}")

def template_sheet_setup_agent():
    """Clean, minimal template sheet setup interface"""
    st.title("Template Sheet Setup Agent")
    
    # Step 1: Template Setup
    st.subheader("Step 1: Template Setup")
    drive_folder_id = st.text_input("Enter course Drive folder ID:", key="template_drive_folder_id")
    course_name = st.text_input("Enter Course Name:", key="template_course_name")
    
    if st.button("Setup Template Sheets", type="primary"):
        if not drive_folder_id or not course_name:
            st.error("Please provide both Drive folder ID and course name.")
            return

        _t = time.perf_counter()
        try:
            load_dotenv()

            # Check for authenticated clients
            if "drive" in st.session_state and "gc" in st.session_state:
                drive = st.session_state["drive"]
                gc = st.session_state["gc"]
            else:
                st.error("Please authenticate with Google Drive and Sheets first through the main app.")
                return

            with st.spinner("Setting up template sheets..."):
                result = setup_template_sheets(drive, gc, drive_folder_id, course_name)

            track_tool_action("Template Sheet Setup", "setup_template_sheets", run_mode="agent", duration_seconds=time.perf_counter() - _t, course_name=course_name, sheet_link=result.get("course_sheet_url", ""))
            st.success("✅ Template sheets setup completed!")

            # Store sheet info for persistent display below
            course_status = "Created" if result['course_sheet_created'] else "Already existed"
            checklist_status = "Created" if result['checklist_sheet_created'] else "Already existed"
            
            st.session_state["template_setup_result"] = result
            st.session_state["course_sheet_url"] = result['course_sheet_url']
            st.session_state["checklist_sheet_url"] = result['checklist_sheet_url']
            st.session_state["course_sheet_status"] = course_status
            st.session_state["checklist_sheet_status"] = checklist_status
            
        except Exception as e:
            track_tool_action("Template Sheet Setup", "setup_template_sheets", run_mode="agent", error_message=str(e)[:500], course_name=course_name, sheet_link="")
            st.error(f"❌ Error setting up template sheets: {e}")
    
    # Always show sheet links if they exist (keeps consistent layout)
    if "course_sheet_url" in st.session_state:
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Course Sheet:** {st.session_state['course_sheet_status']}")
            st.markdown(f"🔗 [Open Course Sheet]({st.session_state['course_sheet_url']})")
        with col2:
            st.info(f"**Checklist Sheet:** {st.session_state['checklist_sheet_status']}")
            st.markdown(f"🔗 [Open Checklist Sheet]({st.session_state['checklist_sheet_url']})")
    
    # Step 2: Manual Review (always visible, but dependent on step 1)
    st.divider()
    st.subheader("Step 2: Manual Review")
    
    if "template_setup_result" in st.session_state:
        # Check if setup is completed
        if st.session_state.get("template_setup_completed", False):
            # Show completion state
            st.success("✅ Template setup process completed successfully!")
            st.info("Both steps have been completed. You can now proceed with your course development.")
        else:
            # Step 1 completed - show active instructions and validation
            st.info("""
            **Please edit the created sheets:**
            1. Open the Course Sheet above
            2. Update Course Info and Base Outline sheet
            3. Click Validate below when done
            """)
        
        if not st.session_state.get("template_setup_completed", False) and st.button("Validate Changes", type="primary"):
            try:
                result = st.session_state["template_setup_result"]
                gc = st.session_state["gc"]
                
                with st.spinner("Validating changes..."):
                    unchanged_fields = validate_manual_review(gc, result['course_sheet_id'])
                
                if unchanged_fields:
                    st.error(f"Please update these fields: **{', '.join(unchanged_fields)}**")
                    st.info("Make sure all fields are different from template defaults.")
                else:
                    track_tool_action("Template Sheet Setup", "validate_changes", run_mode="agent", course_name=course_name, sheet_link=st.session_state.get("course_sheet_url", ""))
                    st.success("✅ All required changes completed! Template setup finished.")
                    st.balloons()
                    # Mark as completed but keep the session state to maintain layout
                    st.session_state["template_setup_completed"] = True
                    
            except Exception as e:
                st.error(f"❌ Error validating changes: {e}")
    else:
        # Step 1 not completed - show disabled state
        st.warning("⚠️ Please complete Step 1 first to enable manual review validation.")
        st.info("""
        **You will need to edit the sheets after Step 1:**
        1. Update Course Info and Base Outline sheet
        2. Validate the changes
        """)
        st.button("Validate Changes", disabled=True, help="Complete Step 1 first")

template_sheet_setup_agent()