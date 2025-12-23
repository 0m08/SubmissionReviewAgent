# Befor Parallelization

# """
# Graphics Workflow V2 - Streamlit UI

# A hierarchical multi-agent system for creating graphics definitions for educational video slides.
# """

# import streamlit as st
# import logging
# import time
# from agents.graphics_workflow_v2.config.settings import SEARCH_K
# # # Suppress the "missing ScriptRunContext" warnings from LangGraph's background threads
# # # These warnings don't affect functionality, just terminal aesthetics
# # logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(logging.ERROR)
# from agents.graphics_workflow_v2 import (
#     run_graphics_workflow,
#     get_final_definition,
#     get_workflow_summary,
# )
# from services.sheets_service import get_sheet_data_and_df, save_to_sheet
# from services.smart_progress_bar import SmartProgressBar
# import pandas as pd
# from io import StringIO

# # --- Authentication Check ---
# if "drive" in st.session_state and "gc" in st.session_state:
#     drive = st.session_state["drive"]
#     gc = st.session_state["gc"]
# else:
#     st.error("❌ Authentication not found. Please log out and log in again.")
#     st.stop()

# # --- Page Header ---
# st.title("🎨 Graphics Workflow V2")
# st.markdown("""
# This is a **next-generation** graphics definition system using hierarchical ReAct agents powered by LangGraph.

# ### Architecture
# - **Level 1:** Slide Supervisor - Orchestrates the entire workflow
# - **Level 2:** Segment Processor - Processes individual VO segments
# - **Level 3:** Search Agent - Finds and validates visual references

# ### Features
# ✨ Automatic VO segmentation
# ✨ Sequential segment processing with context
# ✨ Reference reuse across segments
# ✨ Quality validation with revision loops
# ✨ Cross-segment conflict detection
# ✨ Graceful error handling
# """)

# st.divider()

# # --- Google Sheet Input Method ---
# st.subheader("📝 Load from Google Sheet")

# # Sheet URL input
# # ============================================================================
# # TEMPORARY: Hardcoded URL for testing - REMOVE AFTER TESTING
# # ============================================================================
# sheet_url = "https://docs.google.com/spreadsheets/d/1d9Zdfp1T3iZRGIsw3e3DnLtDSwwsEAKOKZdr0J-6MVg/edit?gid=496582708#gid=496582708"

# # ============================================================================
# # ORIGINAL CODE (uncomment to restore):
# # ============================================================================
# # sheet_url = st.text_input(
# #     "Enter your Google Sheet URL:",
# #     help="The sheet should have 'Slide Chunks' worksheet and 'Course info' worksheet"
# # )

# if sheet_url:
#     try:
#         sheet = gc.open_by_url(sheet_url)
#         st.session_state["graphics_v2_sheet"] = sheet
#         st.success("✓ Sheet loaded successfully")

#         # Fixed worksheet name: "Slide Chunks"
#         worksheet_name = "Slide Chunks"
        
#         # Check if worksheet exists
#         worksheet_names = [ws.title for ws in sheet.worksheets()]
#         if worksheet_name not in worksheet_names:
#             st.error(f"❌ Worksheet '{worksheet_name}' not found. Available worksheets: {', '.join(worksheet_names)}")
#         else:
#             worksheet, df = get_sheet_data_and_df(sheet, worksheet_name)

#             st.write(f"**Found {len(df)} rows in '{worksheet_name}' worksheet**")
#             st.dataframe(df.head(5), use_container_width=True)

#             # Fixed column names
#             content_column = "Slide Chunk"  # Fixed column name
#             topic_column = "Topic"
#             subtopic_column = "Subtopic"
#             slide_title_column = "Slide Chunk Title"

#             # Validate columns exist
#             missing_columns = []
#             if content_column not in df.columns:
#                 missing_columns.append(content_column)
#             if topic_column not in df.columns:
#                 missing_columns.append(topic_column)
#             if subtopic_column not in df.columns:
#                 missing_columns.append(subtopic_column)
#             if slide_title_column not in df.columns:
#                 missing_columns.append(slide_title_column)

#             if missing_columns:
#                 st.error(f"❌ Missing required columns: {', '.join(missing_columns)}")
#                 st.info(f"Available columns: {', '.join(df.columns.tolist())}")
#             else:
#                 # Extract course context from "Course info" tab (once, for all rows)
#                 course_name = None
#                 try:
#                     _, course_info_df = get_sheet_data_and_df(sheet, "Course info")
#                     if "Course Name" in course_info_df.columns:
#                         course_name = str(course_info_df.iloc[0]["Course Name"]).strip()
#                         if course_name and course_name != "nan":
#                             st.success(f"✓ Course Name: {course_name}")
#                         else:
#                             st.warning("⚠️ Course name is empty in 'Course info' worksheet")
#                     else:
#                         st.warning("⚠️ 'Course name' column not found in 'Course info' worksheet")
#                 except Exception as e:
#                     st.warning(f"⚠️ Could not load 'Course info' worksheet: {e}")

#                 # Save configuration for batch processing
#                 st.session_state["graphics_v2_config"] = {
#                     "sheet": sheet,
#                     "worksheet_name": worksheet_name,
#                     "worksheet": worksheet,
#                     "content_column": content_column,
#                     "topic_column": topic_column,
#                     "subtopic_column": subtopic_column,
#                     "slide_title_column": slide_title_column,
#                     "course_name": course_name,
#                     "df": df,  # Store dataframe for processing
#                 }
                
#                 st.success(f"✓ Ready to process {len(df)} slides")

#     except Exception as e:
#         st.error(f"❌ Failed to load sheet: {e}")

# # --- Advanced Settings ---
# with st.expander("⚙️ Advanced Settings"):
#     st.markdown("##### Model Configuration")
#     col1, col2 = st.columns(2)

#     with col1:
#         max_iterations = st.number_input(
#             "Max iterations per segment:",
#             min_value=1,
#             max_value=10,
#             value=3,
#             help="Maximum revision attempts for each segment"
#         )

#     with col2:
#         recursion_limit = st.number_input(
#             "Recursion limit:",
#             min_value=50,
#             max_value=200,
#             value=100,
#             help="Total tool calls allowed for the entire workflow"
#         )

#     st.markdown("##### Search Settings")
#     col3, col4 = st.columns(2)

#     with col3:
#         search_k = SEARCH_K

#     with col4:
#         root_folder_id = st.text_input(
#             "Vector store folder ID:",
#             value="1QS6PmCESfgFWNNEpRJDUB0E-t8iMAatH",
#             help="Google Drive folder ID for the vector store"
#         )

# st.divider()

# # --- Run Workflow ---
# st.subheader("🚀 Run Workflow")

# # Validation
# can_run = "graphics_v2_config" in st.session_state

# if not can_run:
#     st.warning("⚠️ Please load Google Sheet above to continue.")
# else:
#     config = st.session_state["graphics_v2_config"]
#     df = config["df"]
#     total_rows = len(df)
    
#     st.success(f"✓ Ready to process {total_rows} slides")

#     # Display what will be processed
#     with st.expander("📋 Processing Preview"):
#         st.markdown(f"**Total Slides:** {total_rows}")
#         st.markdown(f"**Course Name:** {config.get('course_name', 'Not found')}")
#         st.markdown(f"**Output Column:** Will be added to sheet")

#     # Output column selection
#     output_column = st.text_input(
#         "Output column name:",
#         value="Graphics Definition V2",
#         help="Column where the graphics definitions will be saved"
#     )

#     # Run button
#     if st.button("▶️ Generate Graphics Definitions for All Slides", type="primary", disabled=not can_run):
#         status_text = st.empty()
#         results_container = st.container()

#         try:
#             # Prepare helper function to process a single slide
#             def process_single_slide(row_index):
#                 """Process a single slide and return the result."""
#                 try:
#                     # Extract slide content for this row
#                     slide_content = str(df.iloc[row_index][config["content_column"]]).strip()
                    
#                     # Skip empty rows
#                     if not slide_content or slide_content == "nan" or len(slide_content) < 10:
#                         return {
#                             "row_index": row_index,
#                             "status": "skipped",
#                             "error": "Empty or invalid slide content"
#                         }
                    
#                     # Extract course context for this row
#                     course_context = {}
#                     if config.get("course_name"):
#                         course_context["course_name"] = config["course_name"]
                    
#                     topic_value = str(df.iloc[row_index][config["topic_column"]]).strip()
#                     if topic_value and topic_value != "nan":
#                         course_context["topic"] = topic_value
                    
#                     subtopic_value = str(df.iloc[row_index][config["subtopic_column"]]).strip()
#                     if subtopic_value and subtopic_value != "nan":
#                         course_context["subtopic"] = subtopic_value
                    
#                     slide_title_value = str(df.iloc[row_index][config["slide_title_column"]]).strip()
#                     if slide_title_value and slide_title_value != "nan":
#                         course_context["slide_title"] = slide_title_value
                    
#                     # Run the workflow
#                     result = run_graphics_workflow(
#                         slide_chunk=slide_content,
#                         drive=drive,
#                         course_context=course_context if course_context else None,
#                         max_iterations_per_segment=max_iterations,
#                         recursion_limit=recursion_limit,
#                         root_folder_id=root_folder_id,
#                         search_k=search_k,
#                     )
                    
#                     # Get final definition
#                     from agents.graphics_workflow_v2.agents.slide_supervisor import get_final_definition
#                     final_def = get_final_definition(result)
                    
#                     if final_def:
#                         return {
#                             "row_index": row_index,
#                             "status": "success",
#                             "final_def": final_def,
#                             "slide_title": course_context.get("slide_title", f"Row {row_index + 1}"),
#                         }
#                     else:
#                         return {
#                             "row_index": row_index,
#                             "status": "failed",
#                             "error": "No definition generated"
#                         }
                        
#                 except Exception as e:
#                     return {
#                         "row_index": row_index,
#                         "status": "failed",
#                         "error": str(e)
#                     }
            
#             # Process all rows sequentially
#             results = []
#             failed_rows = []
#             skipped_rows = []
            
#             # Initialize progress tracking
#             valid_rows = []
#             for row_index in range(total_rows):
#                 slide_content = str(df.iloc[row_index][config["content_column"]]).strip()
#                 if slide_content and slide_content != "nan" and len(slide_content) >= 10:
#                     valid_rows.append(row_index)
            
#             if not valid_rows:
#                 st.warning("⚠️ No valid slides to process")
#                 st.stop()
            
#             progress = SmartProgressBar(
#                 total_tasks=len(valid_rows),
#                 description="Processing slides",
#                 save_interval=1  # Save after each completion
#             )
            
#             # Process rows sequentially
#             for row_index in valid_rows:
#                 # Update status text
#                 completed_count = len(results) + len(failed_rows) + len(skipped_rows)
#                 status_text.text(f"🔄 Processing slide {completed_count + 1} of {len(valid_rows)}...")
                
#                 # Process the slide
#                 result_data = process_single_slide(row_index)
                
#                 # Process result
#                 if result_data["status"] == "success":
#                     # Update dataframe
#                     if output_column not in df.columns:
#                         df[output_column] = ""
#                     df.at[result_data["row_index"], output_column] = result_data["final_def"]
                    
#                     results.append({
#                         "row": result_data["row_index"] + 1,
#                         "status": "success",
#                         "slide_title": result_data.get("slide_title", f"Row {result_data['row_index'] + 1}"),
#                     })
                    
#                     # ✅ SAVE IMMEDIATELY after each row completes
#                     try:
#                         save_to_sheet(config["worksheet"], df)
#                         st.info(f"💾 Saved progress after row {result_data['row_index'] + 1}")
#                     except Exception as save_error:
#                         st.warning(f"⚠️ Could not save progress for row {result_data['row_index'] + 1}: {save_error}")
                        
#                 elif result_data["status"] == "skipped":
#                     skipped_rows.append({
#                         "row": result_data["row_index"] + 1,
#                         "reason": result_data.get("error", "Skipped")
#                     })
#                     st.warning(f"⚠️ Row {result_data['row_index'] + 1}: {result_data.get('error', 'Skipped')}")
#                 else:
#                     failed_rows.append({
#                         "row": result_data["row_index"] + 1,
#                         "error": result_data.get("error", "Unknown error")
#                     })
#                     st.error(f"❌ Row {result_data['row_index'] + 1} failed: {result_data.get('error', 'Unknown error')}")
                
#                 # Update progress
#                 progress.update()
                
#             # Final save (redundant but ensures everything is saved)
#             try:
#                 save_to_sheet(config["worksheet"], df)
#                 st.success("💾 Final results saved to sheet")
#             except Exception as save_error:
#                 st.error(f"❌ Failed to save final results: {save_error}")

#             status_text.text(f"✅ Processing complete! {len(results)} succeeded, {len(failed_rows)} failed, {len(skipped_rows)} skipped")

#             # Display summary
#             st.success(f"🎉 Processed {len(results)} slides successfully!")
            
#             if skipped_rows:
#                 st.info(f"ℹ️ {len(skipped_rows)} slides skipped (empty/invalid content)")
            
#             if failed_rows:
#                 st.warning(f"⚠️ {len(failed_rows)} slides failed:")
#                 for failed in failed_rows[:10]:  # Show first 10 failures
#                     st.text(f"  Row {failed['row']}: {failed['error']}")

#         except Exception as e:
#             st.error(f"❌ Batch processing failed: {str(e)}")
#             st.exception(e)

# # --- Display Batch Processing Summary ---
# # Results are automatically saved to the Google Sheet
# # Check the sheet to view all generated graphics definitions


# # --- Help & Documentation ---
# st.divider()
# with st.expander("❓ Help & Documentation"):
#     st.markdown("""
#     ### How to Use

#     1. **Choose input method:**
#        - **Direct Input:** Paste slide content directly
#        - **From Google Sheet:** Load from an existing sheet

#     2. **Provide slide content:**
#        - Paste or select the slide content to process
#        - Optionally provide course context

#     3. **Configure settings (optional):**
#        - Adjust max iterations, recursion limits, search parameters

#     4. **Run the workflow:**
#        - Click "Generate Graphics Definition"
#        - Wait for processing (typically 3-5 minutes per slide)

#     5. **Review results:**
#        - View the formatted definition
#        - Check individual segments
#        - Review any flags
#        - Download or save to sheet

#     ### What It Does

#     The workflow:
#     1. **Segments** the slide into VO (voiceover) moments
#     2. **Processes each segment** sequentially:
#        - Creates graphics definition
#        - Searches for visual references
#        - Reviews quality
#        - Revises if needed (up to 3 times)
#     3. **Assembles** final definition with all segments

#     ### Tips

#     - Provide detailed slide content for better results
#     - Add course context for more relevant searches
#     - Check flags for any issues that need attention
#     - Review individual segments to understand the breakdown
#     - Use the download button to save locally
#     - Save to sheet to integrate with existing workflow

#     ### For More Information

#     See: `agents/graphics_workflow_v2/README.md`
#     """)

# # --- Footer ---
# st.divider()
# st.caption("Graphics Workflow V2 | Powered by LangGraph & ReAct Agents")




# After Parallelization

"""
Graphics Workflow V2 - Streamlit UI

A hierarchical multi-agent system for creating graphics definitions for educational video slides.
"""

import streamlit as st
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# # Suppress the "missing ScriptRunContext" warnings from LangGraph's background threads
# # These warnings don't affect functionality, just terminal aesthetics
# logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(logging.ERROR)
from agents.graphics_workflow_v2 import (
    run_graphics_workflow,
    get_final_definition,
    get_workflow_summary,
)
from services.sheets_service import get_sheet_data_and_df, save_to_sheet, format_worksheet
from services.smart_progress_bar import SmartProgressBar
import pandas as pd
from io import StringIO

# --- Authentication Check ---
if "drive" in st.session_state and "gc" in st.session_state:
    drive = st.session_state["drive"]
    gc = st.session_state["gc"]
else:
    st.error("❌ Authentication not found. Please log out and log in again.")
    st.stop()

# --- Page Header ---
st.title("🎨 Graphics Workflow V2")
st.markdown("""
This is a **next-generation** graphics definition system using hierarchical ReAct agents powered by LangGraph.

### Architecture
- **Level 1:** Slide Supervisor - Orchestrates the entire workflow
- **Level 2:** Segment Processor - Processes individual VO segments
- **Level 3:** Search Agent - Finds and validates visual references

### Features
✨ Automatic VO segmentation
✨ Sequential segment processing with context
✨ Reference reuse across segments
✨ Quality validation with revision loops
✨ Cross-segment conflict detection
✨ Graceful error handling
""")

st.divider()

# --- Google Sheet Input Method ---
st.subheader("📝 Load from Google Sheet")

# Sheet URL input
# ============================================================================
# TEMPORARY: Hardcoded URL for testing - REMOVE AFTER TESTING
# ============================================================================
#sheet_url = "https://docs.google.com/spreadsheets/d/1vbFfqnxYP-Y30w0XbTRMaknitz9pTOCRglNq5wxRSMw/edit?gid=2123556641#gid=2123556641"

#============================================================================
#ORIGINAL CODE (uncomment to restore):
#============================================================================
sheet_url = st.text_input(
    "Enter your Google Sheet URL:",
    help="The sheet should have 'Slide Chunks' worksheet and 'Course info' worksheet"
)

if sheet_url:
    try:
        sheet = gc.open_by_url(sheet_url)
        st.session_state["graphics_v2_sheet"] = sheet
        st.success("✓ Sheet loaded successfully")

        # Fixed worksheet name: "Slide Chunks"
        worksheet_name = "Slide Chunks"
        
        # Check if worksheet exists
        worksheet_names = [ws.title for ws in sheet.worksheets()]
        if worksheet_name not in worksheet_names:
            st.error(f"❌ Worksheet '{worksheet_name}' not found. Available worksheets: {', '.join(worksheet_names)}")
        else:
            worksheet, df = get_sheet_data_and_df(sheet, worksheet_name)

            st.write(f"**Found {len(df)} rows in '{worksheet_name}' worksheet**")
            st.dataframe(df.head(5), use_container_width=True)

            # Fixed column names
            content_column = "Slide Chunk"  # Fixed column name
            topic_column = "Topic"
            subtopic_column = "Subtopic"
            slide_title_column = "Slide Chunk Title"

            # Validate columns exist
            missing_columns = []
            if content_column not in df.columns:
                missing_columns.append(content_column)
            if topic_column not in df.columns:
                missing_columns.append(topic_column)
            if subtopic_column not in df.columns:
                missing_columns.append(subtopic_column)
            if slide_title_column not in df.columns:
                missing_columns.append(slide_title_column)

            if missing_columns:
                st.error(f"❌ Missing required columns: {', '.join(missing_columns)}")
                st.info(f"Available columns: {', '.join(df.columns.tolist())}")
            else:
                # Extract course context from "Course info" tab (once, for all rows)
                course_name = None
                try:
                    _, course_info_df = get_sheet_data_and_df(sheet, "Course info")
                    if "Course Name" in course_info_df.columns:
                        course_name = str(course_info_df.iloc[0]["Course Name"]).strip()
                        if course_name and course_name != "nan":
                            st.success(f"✓ Course Name: {course_name}")
                        else:
                            st.warning("⚠️ Course name is empty in 'Course info' worksheet")
                    else:
                        st.warning("⚠️ 'Course name' column not found in 'Course info' worksheet")
                except Exception as e:
                    st.warning(f"⚠️ Could not load 'Course info' worksheet: {e}")

                # Save configuration for batch processing
                st.session_state["graphics_v2_config"] = {
                    "sheet": sheet,
                    "worksheet_name": worksheet_name,
                    "worksheet": worksheet,
                    "content_column": content_column,
                    "topic_column": topic_column,
                    "subtopic_column": subtopic_column,
                    "slide_title_column": slide_title_column,
                    "course_name": course_name,
                    "df": df,  # Store dataframe for processing
                }
                
                st.success(f"✓ Ready to process {len(df)} slides")

    except Exception as e:
        st.error(f"❌ Failed to load sheet: {e}")

# --- Advanced Settings ---
with st.expander("⚙️ Advanced Settings"):
    st.markdown("##### Model Configuration")
    col1, col2 = st.columns(2)

    with col1:
        max_iterations = st.number_input(
            "Max iterations per segment:",
            min_value=1,
            max_value=10,
            value=3,
            help="Maximum revision attempts for each segment"
        )

    with col2:
        recursion_limit = st.number_input(
            "Recursion limit:",
            min_value=50,
            max_value=200,
            value=100,
            help="Total tool calls allowed for the entire workflow"
        )

    st.markdown("##### Search Settings")
    col3, col4 = st.columns(2)

    with col3:
        search_k = st.number_input(
            "Search results per query:",
            min_value=5,
            max_value=20,
            value=5,
            help="Number of results to retrieve per search query"
        )

    with col4:
        root_folder_id = st.text_input(
            "Vector store folder ID:",
            value="1QS6PmCESfgFWNNEpRJDUB0E-t8iMAatH",
            help="Google Drive folder ID for the vector store"
        )

st.divider()

# --- Run Workflow ---
st.subheader("🚀 Run Workflow")

# Validation
can_run = "graphics_v2_config" in st.session_state

if not can_run:
    st.warning("⚠️ Please load Google Sheet above to continue.")
else:
    config = st.session_state["graphics_v2_config"]
    df = config["df"]
    total_rows = len(df)
    
    # Output column selection
    output_column = st.text_input(
        "Output column name:",
        value="Graphics Definition V2",
        help="Column where the graphics definitions will be saved"
    )
    
    # Count rows that are already filled and rows that will be processed
    if output_column not in df.columns:
        df[output_column] = ""
    
    filled_count = 0
    to_process_count = 0
    
    for row_index in range(total_rows):
        slide_content = str(df.iloc[row_index][config["content_column"]]).strip()
        # Skip if slide content is empty or invalid
        if not slide_content or slide_content == "nan" or len(slide_content) < 10:
            continue
        
        existing_def = str(df.iloc[row_index][output_column]).strip()
        if existing_def and existing_def != "nan":
            filled_count += 1
        else:
            to_process_count += 1
    
    # Display status
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"📝 Already filled: {filled_count} slides")
    with col2:
        st.success(f"✓ Ready to process: {to_process_count} slides")

    # Display what will be processed
    with st.expander("📋 Processing Preview"):
        st.markdown(f"**Total Slides:** {total_rows}")
        st.markdown(f"**Already Filled:** {filled_count}")
        st.markdown(f"**Will Process:** {to_process_count}")
        st.markdown(f"**Course Name:** {config.get('course_name', 'Not found')}")
        st.markdown(f"**Output Column:** {output_column}")

    # Run button
    if st.button("▶️ Generate Graphics Definitions for All Slides", type="primary", disabled=not can_run):
        status_text = st.empty()
        save_status_container = st.empty()  # Container for save status messages
        results_container = st.container()

        try:
            # Prepare helper function to process a single slide
            def process_single_slide(row_index):
                """Process a single slide and return the result."""
                try:
                    # Extract slide content for this row
                    slide_content = str(df.iloc[row_index][config["content_column"]]).strip()
                    
                    # Skip empty rows
                    if not slide_content or slide_content == "nan" or len(slide_content) < 10:
                        return {
                            "row_index": row_index,
                            "status": "skipped",
                            "error": "Empty or invalid slide content"
                        }
                    
                    # Extract course context for this row
                    course_context = {}
                    if config.get("course_name"):
                        course_context["course_name"] = config["course_name"]
                    
                    topic_value = str(df.iloc[row_index][config["topic_column"]]).strip()
                    if topic_value and topic_value != "nan":
                        course_context["topic"] = topic_value
                    
                    subtopic_value = str(df.iloc[row_index][config["subtopic_column"]]).strip()
                    if subtopic_value and subtopic_value != "nan":
                        course_context["subtopic"] = subtopic_value
                    
                    slide_title_value = str(df.iloc[row_index][config["slide_title_column"]]).strip()
                    if slide_title_value and slide_title_value != "nan":
                        course_context["slide_title"] = slide_title_value
                    
                    # Run the workflow
                    result = run_graphics_workflow(
                        slide_chunk=slide_content,
                        drive=drive,
                        course_context=course_context if course_context else None,
                        max_iterations_per_segment=max_iterations,
                        recursion_limit=recursion_limit,
                        root_folder_id=root_folder_id,
                        search_k=search_k,
                    )
                    
                    # Get final definition
                    from agents.graphics_workflow_v2.agents.slide_supervisor import get_final_definition
                    final_def = get_final_definition(result)
                    
                    if final_def:
                        return {
                            "row_index": row_index,
                            "status": "success",
                            "final_def": final_def,
                            "slide_title": course_context.get("slide_title", f"Row {row_index + 1}"),
                        }
                    else:
                        return {
                            "row_index": row_index,
                            "status": "failed",
                            "error": "No definition generated"
                        }
                        
                except Exception as e:
                    return {
                        "row_index": row_index,
                        "status": "failed",
                        "error": str(e)
                    }
            
            # Process all rows in parallel
            results = []
            failed_rows = []
            skipped_rows = []
            
            # Initialize progress tracking
            valid_rows = []
            # Ensure output column exists in dataframe
            if output_column not in df.columns:
                df[output_column] = ""
            
            for row_index in range(total_rows):
                slide_content = str(df.iloc[row_index][config["content_column"]]).strip()
                # Skip if slide content is empty or invalid
                if not slide_content or slide_content == "nan" or len(slide_content) < 10:
                    continue
                
                # Skip if Graphics Definition V2 already exists (not empty)
                existing_def = str(df.iloc[row_index][output_column]).strip()
                if existing_def and existing_def != "nan":
                    continue
                
                valid_rows.append(row_index)
            
            if not valid_rows:
                st.warning("⚠️ No valid slides to process")
                st.stop()
            
            progress = SmartProgressBar(
                total_tasks=len(valid_rows),
                description="Processing slides",
                save_interval=1  # Save after each completion
            )
            
            # Lock for thread-safe sheet operations (even though we're in main thread, this prevents API conflicts)
            sheet_lock = Lock()
            
            # Submit all tasks to ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=6) as executor:
                futures_map = {}
                
                for row_index in valid_rows:
                    future = executor.submit(process_single_slide, row_index)
                    futures_map[future] = row_index
                
                # Collect results as they complete
                for future in as_completed(futures_map):
                    row_index = futures_map[future]
                    result_data = future.result()
                    
                    # Update status text
                    completed_count = len(results) + len(failed_rows) + len(skipped_rows) + 1
                    status_text.text(f"🔄 Completed {completed_count} of {len(valid_rows)} slides...")
                    
                    # Process result
                    if result_data["status"] == "success":
                        # Update dataframe in MAIN THREAD (safe - different rows)
                        if output_column not in df.columns:
                            df[output_column] = ""
                        df.at[result_data["row_index"], output_column] = result_data["final_def"]
                        
                        results.append({
                            "row": result_data["row_index"] + 1,
                            "status": "success",
                            "slide_title": result_data.get("slide_title", f"Row {result_data['row_index'] + 1}"),
                        })
                        
                        # ✅ SAVE IMMEDIATELY after each row completes (thread-safe)
                        try:
                            with sheet_lock:
                                save_to_sheet(config["worksheet"], df)
                            # Print to terminal for visibility
                            print(f"\n💾 [SAVE] Row {result_data['row_index'] + 1} saved to sheet successfully")
                            # Update UI with save status
                            save_status_container.info(f"💾 **Saved to sheet:** Rows {', '.join([str(r['row']) for r in results])} completed and saved")
                            st.info(f"💾 Saved progress after row {result_data['row_index'] + 1}")
                        except Exception as save_error:
                            print(f"\n⚠️ [SAVE ERROR] Row {result_data['row_index'] + 1} failed to save: {save_error}")
                            st.warning(f"⚠️ Could not save progress for row {result_data['row_index'] + 1}: {save_error}")
                            
                    elif result_data["status"] == "skipped":
                        skipped_rows.append({
                            "row": result_data["row_index"] + 1,
                            "reason": result_data.get("error", "Skipped")
                        })
                        st.warning(f"⚠️ Row {result_data['row_index'] + 1}: {result_data.get('error', 'Skipped')}")
                    else:
                        failed_rows.append({
                            "row": result_data["row_index"] + 1,
                            "error": result_data.get("error", "Unknown error")
                        })
                        st.error(f"❌ Row {result_data['row_index'] + 1} failed: {result_data.get('error', 'Unknown error')}")
                    
                    # Update progress after processing each result
                    progress.update()
                
            # Final save (redundant but ensures everything is saved)
            try:
                save_to_sheet(config["worksheet"], df)
                format_worksheet(config["worksheet"])
                st.success("💾 Final results saved to sheet")
            except Exception as save_error:
                st.error(f"❌ Failed to save final results: {save_error}")

            status_text.text(f"✅ Processing complete! {len(results)} succeeded, {len(failed_rows)} failed, {len(skipped_rows)} skipped")

            # Display summary
            st.success(f"🎉 Processed {len(results)} slides successfully!")
            
            if skipped_rows:
                st.info(f"ℹ️ {len(skipped_rows)} slides skipped (empty/invalid content)")
            
            if failed_rows:
                st.warning(f"⚠️ {len(failed_rows)} slides failed:")
                for failed in failed_rows[:10]:  # Show first 10 failures
                    st.text(f"  Row {failed['row']}: {failed['error']}")

        except Exception as e:
            st.error(f"❌ Batch processing failed: {str(e)}")
            st.exception(e)

# --- Display Batch Processing Summary ---
# Results are automatically saved to the Google Sheet
# Check the sheet to view all generated graphics definitions


# --- Help & Documentation ---
st.divider()
with st.expander("❓ Help & Documentation"):
    st.markdown("""
    ### How to Use

    1. **Choose input method:**
       - **Direct Input:** Paste slide content directly
       - **From Google Sheet:** Load from an existing sheet

    2. **Provide slide content:**
       - Paste or select the slide content to process
       - Optionally provide course context

    3. **Configure settings (optional):**
       - Adjust max iterations, recursion limits, search parameters

    4. **Run the workflow:**
       - Click "Generate Graphics Definition"
       - Wait for processing (typically 3-5 minutes per slide)

    5. **Review results:**
       - View the formatted definition
       - Check individual segments
       - Review any flags
       - Download or save to sheet

    ### What It Does

    The workflow:
    1. **Segments** the slide into VO (voiceover) moments
    2. **Processes each segment** sequentially:
       - Creates graphics definition
       - Searches for visual references
       - Reviews quality
       - Revises if needed (up to 3 times)
    3. **Assembles** final definition with all segments

    ### Tips

    - Provide detailed slide content for better results
    - Add course context for more relevant searches
    - Check flags for any issues that need attention
    - Review individual segments to understand the breakdown
    - Use the download button to save locally
    - Save to sheet to integrate with existing workflow

    ### For More Information

    See: `agents/graphics_workflow_v2/README.md`
    """)

# --- Footer ---
st.divider()
st.caption("Graphics Workflow V2 | Powered by LangGraph & ReAct Agents")