import streamlit as st
import gspread
from dotenv import load_dotenv
from pydrive2.drive import GoogleDrive
import traceback

from services.sheets_service import get_sheet_data_and_df, create_or_read_worksheet, format_worksheet, save_to_sheet
from services.smart_progress_bar import SmartProgressBar
from services.drive_service import login_with_service_account
from datetime import datetime
import os
from langtrace_python_sdk import langtrace # Must precede any llm module imports
import tempfile, json, base64


def agent_ui(step_name: str, pipeline_sections: list[dict]):
    st.title(f"{step_name} Agent")

    if "agent_name" not in st.session_state:
        st.session_state["agent_name"] = ""
    if st.session_state["agent_name"] != step_name:
        st.session_state["agent_name"] = step_name

    if "sheet" in st.session_state:
        load_completed_steps(st.session_state["sheet"], step_name)

    if "google_api_key" not in st.session_state:
        st.session_state["google_api_key"] = ""

    with st.sidebar:
        st.text_input(label="Google API Key", type = "password", key = "google_api_key", value = st.session_state["google_api_key"])
        # st.selectbox(label="LLM Model", options= ["gemini_2_flash", "gpt4_1"], index = None, key = "llm_model")

    if st.session_state["google_api_key"] != "":
        os.environ["GOOGLE_API_KEY"] = st.session_state["google_api_key"]

    if "current_step" not in st.session_state:
        st.session_state["current_step"] = None

    # --- 1) Define pipeline as sections, each with its own steps ---

    # --- 2) Initialize session states for each step ---
    for section in pipeline_sections:
        for step in section["steps"]:
            step_key = f"{step['name']}_done"
            if step_key not in st.session_state:
                st.session_state[step_key] = False

            # Only initialize pre-execution state if the step has a pre_exec_func
            if "pre_exec_func" in step:
                pre_exec_key = f"{step['name']}_pre_executed"
                if pre_exec_key not in st.session_state:
                    st.session_state[pre_exec_key] = False

    # --- 3) Hide "Load Data" inputs once data is loaded ---
    if "sheet" not in st.session_state:
        root_folder_id = st.text_input("Enter course Drive folder ID")
        sheet_link = st.text_input("Enter Google Sheet link")

        # Button to load data
        if st.button("Load Data"):
            load_dotenv()  # Load env variables from .env
            # Check if the Google API key is provided
            if st.session_state["google_api_key"] != "":
                os.environ["GOOGLE_API_KEY"] = st.session_state["google_api_key"]
            try:
                if os.environ.get('LANGTRACE_ON', 'false') == "true":
                    langtrace.init(api_key = os.environ.get('LANGTRACE_API_KEY'))

                # --- decode the secret ---
                key_bytes = base64.b64decode(os.environ["GDRIVE_SA_B64"])
                sa_json = key_bytes.decode()
                sa_dict   = json.loads(sa_json)        # <‑ real newlines intact
                # sa_dict = json.loads(os.environ["GDRIVE_SA_JSON"])   # injected secret
                # print("Service Account JSON: ", sa_dict)
                # with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
                #     json.dump(sa_dict, tmp)
                #     os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = tmp.name
                #     # print(os.environ["GOOGLE_APPLICATION_CREDENTIALS"])

                # with open(os.environ["GOOGLE_APPLICATION_CREDENTIALS"], "r") as f:
                #     st.write(f"Service Account JSON: {f.read()}")
                gauth = login_with_service_account(json_str = sa_json)
                gauth.ServiceAuth()
                drive = GoogleDrive(gauth)

                gc = gspread.service_account_from_dict(sa_dict)
                # gc = gspread.service_account(filename=os.environ["GOOGLE_APPLICATION_CREDENTIALS"])
                sheet = gc.open_by_url(sheet_link)
                course_info_sheet, course_info_df = get_sheet_data_and_df(sheet, 'Course info')

                st.session_state["root_folder_id"] = root_folder_id
                st.session_state["sheet"] = sheet
                st.session_state["course_name"] = course_info_df['Course Name'][0]
                st.session_state["target_audience"] = course_info_df['Target Audience & Industry'][0]
                st.session_state["course_background"] = course_info_df['Course Background'][0]
                st.session_state["drive"] = drive
                st.session_state["gc"] = gc
                
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

                            # NEW CODE: Run pre-execution function if it exists and hasn't been run yet
                            if "pre_exec_func" in step:
                                pre_exec_key = f"{step['name']}_pre_executed"
                                # Check if we should always run the pre_exec function or only if it hasn't been run yet
                                always_run = step.get("pre_exec_always_run", False)
                                if always_run or not st.session_state.get(pre_exec_key, False):
                                    try:
                                        # Gather pre-execution arguments from session_state
                                        pre_kwargs = {}
                                        if "pre_exec_args" in step:
                                            for arg_name, session_key in step["pre_exec_args"].items():
                                                if isinstance(session_key, str) and session_key in st.session_state:
                                                    pre_kwargs[arg_name] = st.session_state[session_key]
                                                else:
                                                    pre_kwargs[arg_name] = session_key
                                        
                                        # Run the pre-execution function
                                        with st.spinner(f"Loading preview data for {step['name']}..."):
                                            step["pre_exec_func"](**pre_kwargs)
                                        
                                        # Mark pre-execution as done
                                        st.session_state[pre_exec_key] = True
                                    except Exception as e:
                                        st.error(f"Error in pre-execution for {step['name']}: {e}")
                                        st.text(traceback.format_exc())

                            if st.button(button_name, type=button_type, key=f"btn_{step['name']}"):
                                try:
                                    # Add current step to session state
                                    st.session_state["current_step"] = step["name"]

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

                            # Add delete button for completed steps
                            if st.button("Delete Step", type="secondary", key=f"delete_{step['name']}"):
                                try:
                                    # Get all dependent steps
                                    affected_steps = get_dependent_steps(pipeline_sections, step["name"])
                                    
                                    affected_list = ", ".join(affected_steps)
                                    st.warning(f"Deleting this step will also delete these dependent steps: {affected_list}")
                                    # if st.button("Confirm Delete", type="secondary", key=f"confirm_delete_{step['name']}"):
                                    delete_steps(st.session_state["sheet"], st.session_state["agent_name"], 
                                                affected_steps, pipeline_sections)
                                    st.success(f"Deleted step {step['name']} and its dependencies!")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error deleting {step['name']}: {e}")
                                    st.text(traceback.format_exc())

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
    st.write(st.session_state)


def get_dependent_steps(pipeline_sections, step_name):
    """
    Find all steps that depend on the given step (directly or indirectly).
    Returns a list of step names including the starting step.
    Only includes steps that are marked as done in the session state.
    """
    all_steps = {}
    # First, build a dictionary of all steps and their dependencies
    for section in pipeline_sections:
        for step in section["steps"]:
            all_steps[step["name"]] = step["depends_on"]
    
    # Initialize with the starting step
    dependent_steps = [step_name]
    
    # Function to recursively find dependent steps
    def find_dependents(step_to_check):
        for current_step, dependencies in all_steps.items():
            # Only include steps that are marked as done
            if step_to_check in dependencies and current_step not in dependent_steps and st.session_state.get(f"{current_step}_done", False):
                dependent_steps.append(current_step)
                find_dependents(current_step)
    
    # Start the recursive search
    find_dependents(step_name)
    
    return dependent_steps


def delete_steps(sheet, agent_name, step_names, pipeline_sections):
    """
    Delete the specified steps from the session state and the logs.
    Also executes any delete functions associated with the steps.
    
    Note: Only steps that are marked as done will be included in the deletion process.
    This is ensured by the get_dependent_steps function which filters out steps
    that are not marked as done.
    """
    # Get the Agent logs worksheet
    worksheet, df = get_sheet_data_and_df(sheet, "Agent logs")
    
    # Filter logs to remove the specified steps for this agent
    new_df = df[~((df["Agent Name"] == agent_name) & (df["Step Name"].isin(step_names)))]
    
    # Clear the worksheet
    worksheet.clear()
    
    # Add header row
    # worksheet.append_row(["Agent Name", "Step Name", "Timestamp"])
    
    # Add remaining logs back
    if not new_df.empty:
        save_to_sheet(worksheet = worksheet, df = new_df)
    
    # Initialize the progress tracker
    progress = SmartProgressBar(total_tasks = len(step_names), description = "Percent complete")

    # Reset session state for the deleted steps
    for step_name in step_names:
        st.session_state[f"{step_name}_done"] = False
        pre_exec_key = f"{step_name}_pre_executed"
        if pre_exec_key in st.session_state:
            st.session_state[pre_exec_key] = False
        
        # Execute the delete function if it exists
        for section in pipeline_sections:
            for step in section["steps"]:
                if step["name"] == step_name and "delete_func" in step:
                    # try:
                    # Gather delete function arguments
                    delete_kwargs = {}
                    if "delete_args" in step:
                        for arg_name, session_key in step["delete_args"].items():
                            if isinstance(session_key, str) and session_key in st.session_state:
                                delete_kwargs[arg_name] = st.session_state[session_key]
                            else:
                                delete_kwargs[arg_name] = session_key
                    
                    # Execute the delete function
                    with st.spinner(text = f"Deleting {step_name}...", show_time = True):
                        step["delete_func"](**delete_kwargs)
                    # except Exception as e:
                    #     st.error(f"Error in delete function for {step_name}: {e}")
                    #     st.text(traceback.format_exc())
        
        # Update progress
        progress.update()



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

                # Skip manual steps unless "skip_manual_step" is checked
                # if "instructions" in step and not st.session_state.get("skip_manual_step", False):
                    # continue

                # Check if dependencies are satisfied
                dependencies_satisfied = all(
                    st.session_state.get(f"{dep}_done", False)
                    for dep in step["depends_on"]
                )

                if dependencies_satisfied:
                    try:
                        with st.spinner(text = f"Running: Step {step_global_count}. {step['name']}...", show_time = True):
                            # Add current step to session state
                            st.session_state["current_step"] = step["name"]

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
        worksheet, df = get_sheet_data_and_df(sheet, "Agent logs")
        
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
