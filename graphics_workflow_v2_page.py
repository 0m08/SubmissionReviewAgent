"""
Graphics Workflow V2 - Streamlit UI

A hierarchical multi-agent system for creating graphics definitions for educational video slides.
"""

import streamlit as st
import time
from agents.graphics_workflow_v2 import (
    run_graphics_workflow,
    get_final_definition,
    get_workflow_summary,
)
from services.sheets_service import get_sheet_data_and_df, save_to_sheet
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

# --- Input Method Selection ---
st.subheader("📝 Input Method")
input_method = st.radio(
    "How would you like to provide the slide content?",
    ["Direct Input", "From Google Sheet"],
    help="Choose whether to paste slide content directly or load from a Google Sheet"
)

slide_content = None
course_context = {}

# --- Direct Input Method ---
if input_method == "Direct Input":
    st.markdown("#### Enter Slide Content")
    slide_content = st.text_area(
        "Paste your slide content here:",
        height=200,
        placeholder="Example:\nSuperheat happens after the refrigerant has fully evaporated into a vapor in the evaporator. At this point, it continues to absorb heat...",
        help="Paste the complete slide content that you want to create graphics for"
    )

    with st.expander("📚 Optional: Course Context"):
        course_name = st.text_input("Course Name", placeholder="HVAC Fundamentals")
        module_name = st.text_input("Module Name", placeholder="Refrigeration Cycle")
        topic_name = st.text_input("Topic Name", placeholder="Superheat")

        if course_name or module_name or topic_name:
            course_context = {
                "course_name": course_name if course_name else None,
                "module": module_name if module_name else None,
                "topic": topic_name if topic_name else None,
            }

# --- Google Sheet Input Method ---
elif input_method == "From Google Sheet":
    st.markdown("#### Load from Google Sheet")

    # Sheet URL input
    sheet_url = st.text_input(
        "Enter your Google Sheet URL:",
        help="The sheet should have a worksheet with slide content"
    )

    if sheet_url:
        try:
            sheet = gc.open_by_url(sheet_url)
            st.session_state["graphics_v2_sheet"] = sheet
            st.success("✓ Sheet loaded successfully")

            # Worksheet selection
            worksheet_names = [ws.title for ws in sheet.worksheets()]
            worksheet_name = st.selectbox(
                "Select worksheet:",
                worksheet_names,
                help="Choose the worksheet containing slide content"
            )

            # Column selection
            if worksheet_name:
                _, df = get_sheet_data_and_df(sheet, worksheet_name)

                st.write(f"**Preview of '{worksheet_name}':**")
                st.dataframe(df.head(3), use_container_width=True)

                # Row and column selection
                col1, col2 = st.columns(2)
                with col1:
                    content_column = st.selectbox(
                        "Slide content column:",
                        df.columns.tolist(),
                        help="Column containing the slide content"
                    )
                with col2:
                    row_index = st.number_input(
                        "Row number (0-indexed):",
                        min_value=0,
                        max_value=len(df) - 1,
                        value=0,
                        help="Which row to process"
                    )

                # Extract slide content
                if content_column and row_index is not None:
                    slide_content = str(df.iloc[row_index][content_column])

                    st.info(f"**Selected slide content (Row {row_index}):**")
                    st.text_area("Content preview:", slide_content, height=150, disabled=True)

                    # Optional: Extract course context from other columns
                    with st.expander("📚 Auto-extract Course Context"):
                        context_col1, context_col2 = st.columns(2)

                        with context_col1:
                            course_col = st.selectbox(
                                "Course name column (optional):",
                                ["None"] + df.columns.tolist(),
                                help="Column with course name"
                            )
                        with context_col2:
                            topic_col = st.selectbox(
                                "Topic column (optional):",
                                ["None"] + df.columns.tolist(),
                                help="Column with topic/module name"
                            )

                        if course_col != "None":
                            course_context["course_name"] = str(df.iloc[row_index][course_col])
                        if topic_col != "None":
                            course_context["topic"] = str(df.iloc[row_index][topic_col])

                    # Save configuration
                    st.session_state["graphics_v2_config"] = {
                        "sheet": sheet,
                        "worksheet_name": worksheet_name,
                        "content_column": content_column,
                        "row_index": row_index,
                    }

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
            value=10,
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
can_run = bool(slide_content and slide_content.strip())

if not can_run:
    st.warning("⚠️ Please provide slide content above to continue.")
else:
    st.success("✓ Ready to run")

    # Display what will be processed
    with st.expander("📋 Workflow Preview"):
        st.markdown("**Slide Content:**")
        st.code(slide_content[:300] + "..." if len(slide_content) > 300 else slide_content)

        if course_context:
            st.markdown("**Course Context:**")
            st.json(course_context)

    # Run button
    if st.button("▶️ Generate Graphics Definition", type="primary", disabled=not can_run):
        # Initialize progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            status_text.text("🔄 Initializing workflow...")
            progress_bar.progress(10)

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

            progress_bar.progress(100)
            status_text.text("✅ Workflow complete!")

            # Store results in session state
            st.session_state["graphics_v2_result"] = result
            st.session_state["graphics_v2_timestamp"] = time.time()

            # Success message
            st.success("🎉 Graphics definition generated successfully!")

        except Exception as e:
            st.error(f"❌ Workflow failed: {str(e)}")
            st.exception(e)

# --- Display Results ---
if "graphics_v2_result" in st.session_state:
    st.divider()
    st.subheader("📊 Results")

    result = st.session_state["graphics_v2_result"]

    # Summary
    summary = get_workflow_summary(result)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Status", summary["status"])
    with col2:
        st.metric("Segments", f"{summary['completed_segments']}/{summary['total_segments']}")
    with col3:
        st.metric("References", summary["total_references"])
    with col4:
        st.metric("Flags", summary["total_flags"])

    # Detailed metrics
    with st.expander("📈 Detailed Metrics"):
        st.json(summary)

    # Final Definition
    final_def = get_final_definition(result)

    if final_def:
        st.markdown("### 📝 Final Graphics Definition")

        # Display with tabs
        tab1, tab2, tab3 = st.tabs(["📄 Formatted", "📋 Raw Markdown", "💾 Download"])

        with tab1:
            st.markdown(final_def)

        with tab2:
            st.code(final_def, language="markdown")

        with tab3:
            st.download_button(
                label="💾 Download as Markdown",
                data=final_def,
                file_name="graphics_definition.md",
                mime="text/markdown"
            )

            st.download_button(
                label="💾 Download as Text",
                data=final_def,
                file_name="graphics_definition.txt",
                mime="text/plain"
            )

    # Segment Details
    if result.get("segments"):
        with st.expander("🔍 View Individual Segments"):
            for seg in result["segments"]:
                st.markdown(f"### Segment {seg['segment_index']}")
                st.markdown(f"**VO:** {seg['vo_text']}")
                st.markdown(f"**Status:** {seg['status']} | **Iterations:** {seg['iteration_count']}")

                if seg.get("graphics_definition"):
                    with st.expander("Graphics Definition"):
                        st.text(seg["graphics_definition"])

                if seg.get("references"):
                    with st.expander(f"References ({len(seg['references'])})"):
                        for i, ref in enumerate(seg["references"]):
                            st.markdown(f"**{i+1}. {ref['title']}** ({ref['type']})")
                            st.caption(f"URL: {ref['url']}")
                            st.caption(f"Relevance: {ref['relevance_score']:.2f}")

                st.divider()

    # Flags (if any)
    if summary["flags"]:
        st.warning("⚠️ **Flags for Human Review:**")
        for i, flag in enumerate(summary["flags"]):
            st.markdown(f"{i+1}. {flag}")

    # Save to Google Sheet (if loaded from sheet)
    if input_method == "From Google Sheet" and "graphics_v2_config" in st.session_state:
        st.divider()
        st.subheader("💾 Save to Google Sheet")

        config = st.session_state["graphics_v2_config"]

        output_column = st.text_input(
            "Output column name:",
            value="Graphics Definition V2",
            help="Column where the graphics definition will be saved"
        )

        if st.button("💾 Save to Sheet"):
            try:
                sheet = config["sheet"]
                worksheet_name = config["worksheet_name"]
                row_index = config["row_index"]

                # Get current data
                _, df = get_sheet_data_and_df(sheet, worksheet_name)

                # Add or update column
                if output_column not in df.columns:
                    df[output_column] = ""

                df.at[row_index, output_column] = final_def

                # Save back to sheet
                save_to_sheet(sheet, worksheet_name, df)

                st.success(f"✓ Saved to '{worksheet_name}' sheet, column '{output_column}', row {row_index}")

            except Exception as e:
                st.error(f"❌ Failed to save: {e}")

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
