import streamlit as st
import gspread
from dotenv import load_dotenv
from pydrive2.drive import GoogleDrive
import traceback

from services.sheets_service import get_sheet_data_and_df
from services.drive_service import login_with_service_account


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
        if st.session_state['role'] == 'Admin':
            # Skip Manual Steps
            st.checkbox(label = "Skip Manual Steps", value = False, key = "skip_manual_step")
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
                                            st.success(f"{step['name']} completed!")
                                            st.rerun()
                                        else:
                                            st.warning(f"{step['name']} not completed!")
                                    else:
                                        # Run the actual function
                                        step["func"](**kwargs)
                                        st.session_state[step_key] = True
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
        
        # Go through all steps in all sections
        for section in pipeline_sections:
            for step in section["steps"]:
                step_key = f"{step['name']}_done"
                
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
                        progress_made = True
                        st.success(f"Auto-run: {step['name']} completed!")
                    except Exception as e:
                        st.error(f"Error auto-running {step['name']}: {e}")
                        st.text(traceback.format_exc())

# if __name__ == "__main__":
# main()
