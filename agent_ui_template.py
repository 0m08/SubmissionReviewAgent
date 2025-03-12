import streamlit as st
import gspread
from dotenv import load_dotenv
from pydrive2.drive import GoogleDrive
import traceback

from services.sheets_service import get_sheet_data_and_df, create_or_read_worksheet, format_worksheet
from services.drive_service import login_with_service_account
from datetime import datetime
import pandas as pd


def agent_ui(step_name: str, pipeline_sections: list[dict]):
    st.title(f"{step_name} Agent")

    # --- 1) Define pipeline as sections, each with its own steps ---

    # --- 2) Initialize session states for each step ---
    for section in pipeline_sections:
        for step in section["steps"]:
            step_key = f"{step['name']}_done"
            if step_key not in st.session_state:
                st.session_state[step_key] = False

    # --- 3) Hide "Load Data" inputs once data is loaded ---
    if "sheet" not in st.session_state:
        root_folder_id = st.text_input("Enter course Drive folder ID")
        sheet_link = st.text_input("Enter Google Sheet link")

        # Button to load data
        if st.button("Load Data"):
            load_dotenv()  # Load env variables from .env
            try:
                gauth = login_with_service_account("content/service-credentials.json")
                drive = GoogleDrive(gauth)

                gc = gspread.service_account(filename='content/service-credentials.json')
                sheet = gc.open_by_url(sheet_link)
                course_info_sheet, course_info_df = get_sheet_data_and_df(sheet, 'Course info')

                st.session_state["root_folder_id"] = root_folder_id
                st.session_state["sheet"] = sheet
                st.session_state["course_name"] = course_info_df['Course Name'][0]
                st.session_state["target_audience"] = course_info_df['Target Audience & Industry'][0]
                st.session_state["course_background"] = course_info_df['Course Background'][0]
                st.session_state["drive"] = drive
                st.session_state["agent_name"] = step_name
                
                # Load previously completed steps from Agent logs
                load_completed_steps(sheet, step_name)

                st.success("Data loaded successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Error loading data: {e}")
                # Display the full stack trace
                st.text(traceback.format_exc())
    else:
        # If data is already loaded, simply confirm it to the user
        st.success(f"Data already loaded for course: **{st.session_state['course_name']}**. Proceed below.")

        # Admin exclusive features
        if 'role' in st.session_state and st.session_state['role'] == 'Admin':
            # Skip Manual Steps
            st.checkbox(label = "Skip Manual Steps", value = False, key = "skip_manual_step")
            
            # # Reset completed steps option
            # if st.button("Reset All Completed Steps"):
            #     for section in pipeline_sections:
            #         for step in section["steps"]:
            #             step_key = f"{step['name']}_done"
            #             st.session_state[step_key] = False
            #     # Clear the sheet log
            #     clear_agent_logs(st.session_state["sheet"], st.session_state["agent_name"])
            #     # st.rerun()
            
            # Add "Run All Automated Steps" button
            if st.button("Run All Automated Steps", type="primary"):
                run_all_automated_steps(pipeline_sections)
                # st.rerun()

    # --- 4) Display pipeline steps in nested sections ---
    if "sheet" in st.session_state:
        step_global_count = 1  # So we can label steps 1,2,3 across sections
        for section_idx, section in enumerate(pipeline_sections, start=1):
            with st.container(border=True):
                st.header(section["section_name"], divider = True)
                for step in section["steps"]:
                    step_key = f"{step['name']}_done"
                    
                    # Check if dependencies are satisfied
                    dependencies_satisfied = all(
                        st.session_state.get(f"{dep}_done", False)
                        for dep in step["depends_on"]
                    )

                    # # Code to hide if step not ready.
                    # if not dependencies_satisfied:
                    #     continue

                    # Expander
                    with st.expander(label = f"Step {step_global_count}: {step['name']}", expanded = (not st.session_state[step_key] and dependencies_satisfied)):
                        st.subheader(f"Step {step_global_count}: {step['name']}")
                        step_global_count += 1

                        # Always show estimated time
                        st.write(f"**Estimated Time:** {step.get('estimated_time', 'N/A')}")

                        if not dependencies_satisfied:
                            # If dependencies are not done, show a message & skip
                            missing_steps = [
                                dep for dep in step["depends_on"]
                                if not st.session_state.get(f"{dep}_done", False)
                            ]
                            missing_list = ", ".join(missing_steps)
                            st.warning(f"Waiting on these steps to be done first: {missing_list}")
                            continue

                        if not st.session_state[step_key]:
                            # This step is not done yet

                            # If we have a description, only show it while the user can act on the step
                            if "description" in step:
                                st.info(step["description"])

                            if "instructions" in step:
                                for instruction in step["instructions"]:
                                    st.write(instruction)
                                button_name = f"Confirm {step['name']}"
                                button_type = "primary"
                            else:
                                button_name = f"Run {step['name']}"
                                button_type = "secondary"

                            if st.button(button_name, type=button_type, key=f"btn_{step['name']}"):
                                try:
                                    # Gather actual arguments from session_state
                                    kwargs = {}
                                    for arg_name, session_key in step["args"].items():
                                        # If the session_key is a string that matches a valid session_state key, retrieve it
                                        if isinstance(session_key, str) and session_key in st.session_state:
                                            kwargs[arg_name] = st.session_state[session_key]
                                        else:
                                            # or if it's a literal / direct value, pass it through
                                            kwargs[arg_name] = session_key

                                    if "instructions" in step:
                                        # If it is a manual input type function
                                        response = step["func"](**kwargs)
                                        if response:
                                            st.session_state[step_key] = True
                                            # Log the completed step
                                            log_completed_step(st.session_state["sheet"], st.session_state["agent_name"], step["name"])
                                            st.success(f"{step['name']} completed!")
                                            st.rerun()
                                        else:
                                            st.warning(f"{step['name']} not completed!")
                                    else:
                                        # Run the actual function
                                        step["func"](**kwargs)
                                        st.session_state[step_key] = True
                                        # Log the completed step
                                        log_completed_step(st.session_state["sheet"], st.session_state["agent_name"], step["name"])
                                        st.success(f"{step['name']} completed!")
                                        st.rerun()
                                except Exception as e:
                                    st.error(f"Error running {step['name']}: {e}")
                                    st.text(traceback.format_exc())
                        else:
                            # st.write(f"{step['name']}: **Done**")
                            st.write("**Status:** Done")

                            # If this is the very last step in the entire pipeline, celebrate
                            if (
                                section_idx == len(pipeline_sections)
                                and step == section["steps"][-1]
                            ):
                                st.balloons()
                                st.toast(
                                    f"You have successfully generated the {step_name}!",
                                    icon=":material/done_all:",
                                )

    # Debug
    # st.write(st.session_state)


def run_all_automated_steps(pipeline_sections):
    """Run all non-manual steps in the pipeline that have their dependencies satisfied."""
    progress_made = True
    
    # Keep iterating as long as we're making progress
    while progress_made:
        progress_made = False
        
        step_global_count = 0
        # Go through all steps in all sections
        for section in pipeline_sections:
            for step in section["steps"]:
                step_key = f"{step['name']}_done"
                step_global_count += 1

                # Skip if already done
                if st.session_state[step_key]:
                    continue

                # # Skip manual steps (those with instructions)
                # if "instructions" in step:
                #     continue

                # Check if dependencies are satisfied
                dependencies_satisfied = all(
                    st.session_state.get(f"{dep}_done", False)
                    for dep in step["depends_on"]
                )

                if dependencies_satisfied:
                    try:
                        with st.spinner(text = f"Running: Step {step_global_count}. {step['name']}...", show_time = True):
                            # Gather actual arguments from session_state
                            kwargs = {}
                            for arg_name, session_key in step["args"].items():
                                # If the session_key is a string that matches a valid session_state key, retrieve it
                                if isinstance(session_key, str) and session_key in st.session_state:
                                    kwargs[arg_name] = st.session_state[session_key]
                                else:
                                    # or if it's a literal / direct value, pass it through
                                    kwargs[arg_name] = session_key

                            # Run the function
                            step["func"](**kwargs)
                            st.session_state[step_key] = True
                            # Log the completed step
                            log_completed_step(st.session_state["sheet"], st.session_state["agent_name"], step["name"])
                            progress_made = True
                            st.success(f"Auto-run: Step {step_global_count}. {step['name']} completed!")
                    except Exception as e:
                        st.error(f"Error auto-running Step {step_global_count}. {step['name']}: {e}")
                        st.text(traceback.format_exc())


def log_completed_step(sheet, agent_name, step_name):
    """Log a completed step to the Agent logs worksheet."""
    try:
        worksheet, df = create_or_read_worksheet(sheet, "Agent logs")
        
        # Create a timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Create a new log entry
        log_data = [agent_name, step_name, timestamp]
        
        # Add the log entry to the worksheet
        worksheet.append_row(log_data)
    except Exception as e:
        st.error(f"Error logging completed step: {e}")


def load_completed_steps(sheet, agent_name):
    """Load previously completed steps from the Agent logs worksheet."""
    try:
        worksheet, df = create_or_read_worksheet(sheet, "Agent logs")
        
        # If the dataframe is empty (only has headers), return
        if df.empty:
            # Add header row
            worksheet.append_row(["Agent Name", "Step Name", "Timestamp"])
            format_worksheet(worksheet = worksheet)
            return
        
        # Filter logs for the current agent
        agent_logs = df[df["Agent Name"] == agent_name]
        
        # Mark each logged step as completed in the session state
        for _, row in agent_logs.iterrows():
            step_key = f"{row['Step Name']}_done"
            st.session_state[step_key] = True
                
    except Exception as e:
        st.error(f"Error loading completed steps: {e}")


# def clear_agent_logs(sheet, agent_name):
#     """Clear logs for a specific agent from the Agent logs worksheet."""
#     try:
#         # Get the Agent logs worksheet
#         worksheet, df = get_sheet_data_and_df(sheet, "Agent logs")
        
#         # Filter out logs for the current agent
#         new_df = df[df["Agent Name"] != agent_name]
        
#         # Clear the worksheet
#         worksheet.clear()
        
#         # Add header row
#         worksheet.append_row(["Agent Name", "Step Name", "Timestamp"])
        
#         # Add remaining logs back
#         if not new_df.empty:
#             for _, row in new_df.iterrows():
#                 worksheet.append_row(row.tolist())
        
#         st.success("All steps have been reset!")

#     except Exception as e:
#         st.error(f"Error clearing agent logs: {e}")

