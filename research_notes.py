import streamlit as st
import gspread
from dotenv import load_dotenv
from pydrive2.drive import GoogleDrive
import traceback

from agents.research_notes.retriever_agent import run_retriever_agent_for_all_rows
from agents.research_notes.generate_notes import run_research_notes_agent_for_all_rows
from agents.research_notes.retriever_agent import manual_input_review_context
from agents.research_notes.review_subtopic_notes import run_reviewer_agent_for_all_rows, manual_input_review_research_notes
from agents.research_notes.revise_subtopic_notes import run_reviser_agent_for_all_rows
from agents.research_notes.create_research_notes_sheet import create_research_notes_sheet
from agents.research_notes.review_topic_notes import run_review_topic_notes_agent_for_all_rows, manual_input_review_topic_notes
from agents.research_notes.revise_topic_notes import run_revise_topic_notes_for_all_rows

from services.sheets_service import get_sheet_data_and_df
from services.drive_service import login_with_service_account


def main():
    st.title("Research Notes Agent")

    # --- 1) Define pipeline steps as a list of dicts ---
    pipeline_steps = [
        {
            "name": "Retriever",
            "func": run_retriever_agent_for_all_rows,
            "depends_on": [],  # no dependencies
            "args": {
                "root_folder_id": "root_folder_id",
                "drive": "drive",
                "sheet": "sheet",
                "worksheet_name": "Course Outline with LOs",
                "course_name": "course_name",
                "target_audience": "target_audience",
                "llm": "gemini_2_flash",
            },
        },
        {
            "name": "Retriever Manual Input",
            "func": manual_input_review_context,
            "depends_on": ["Retriever"],
            "args": {
                "sheet": "sheet",
                "worksheet_name": "Course Outline with LOs",
            },
            "instructions": ["""Manually add the following column `Context Review`
in the tab and mark it as Done."""]
        },
        {
            "name": "Researcher",
            "func": run_research_notes_agent_for_all_rows,
            "depends_on": ["Retriever"],  # can only run after "Retriever"
            "args": {
                "sheet": "sheet",
                "worksheet_name": "Course Outline with LOs",
                "course_name": "course_name",
                "target_audience": "target_audience",
                "llm": "gemini_2_flash",
            },
        },
        {
            "name": "Reviewer",
            "func": run_reviewer_agent_for_all_rows,
            "depends_on": ["Researcher"],
            "args": {
                "sheet": "sheet",
                "worksheet_name": "Course Outline with LOs",
                "course_name": "course_name",
                "target_audience": "target_audience",
                "llm": "gemini_2_flash",
            },
        },
        {
            "name": "Manual Review",
            "func": manual_input_review_research_notes,
            "depends_on": ["Reviewer"],
            "args": {
                "sheet": "sheet",
                "worksheet_name": "Course Outline with LOs",
            },
            "instructions": ["Review the research notes, and the ai generated review.",
                            "Column names:\n `research_notes`,\n `analysis`,\n `not_covered_at_all`,\n `not_covered_enough`,\n `perfectly_covered`,\n `covered_too_much`,\n `verdict`",
                            "---",
                            "Edit any of the category columns to incorporate your final review.",
                            "You can put any additional comments in the `Manual Comments` column.",
                            "---"]
        },
        {
            "name": "Reviser",
            "func": run_reviser_agent_for_all_rows,
            "depends_on": ["Manual Review"],
            "args": {
                "root_folder_id": "root_folder_id",
                "drive": "drive",
                "sheet": "sheet",
                "worksheet_name": "Course Outline with LOs",
                "course_name": "course_name",
                "target_audience": "target_audience",
                "llm": "gemini_2_flash",
            },
        },
        {
            "name": "Create Research Notes Sheet",
            "func": create_research_notes_sheet,
            "depends_on": ["Reviser"],
            "args": {
                "sheet": "sheet",
                "source_worksheet_name": "Course Outline with LOs",
                "target_worksheet_name": "Research Notes",
            }
        },
        {
            "name": "Review Topic Notes",
            "func": run_review_topic_notes_agent_for_all_rows,
            "depends_on": ["Create Research Notes Sheet"],
            "args": {
                "sheet": "sheet",
                "worksheet_name": "Research Notes",
                "course_name": "course_name",
                "target_audience": "target_audience",
                "llm": "gemini_2_flash",
            },
        },
        {
            "name": "Manual Review Topic Notes",
            "func": manual_input_review_topic_notes,
            "depends_on": ["Review Topic Notes"],
            "args": {
                "sheet": "sheet",
                "worksheet_name": "Research Notes",
            },
            "instructions": [
                "Review the research notes, and the ai generated review.",
                "Column names:\n `research_notes`,\n `final_review`",
                "---",
                "You can put any additional comments in the `Manual Comments` column.",
                "---",
            ]
        },
        {
            "name": "Revise Topic Notes",
            "func": run_revise_topic_notes_for_all_rows,
            "depends_on": ["Manual Review Topic Notes"],
            "args": {
                "sheet": "sheet",
                "worksheet_name": "Research Notes",
                "course_name": "course_name",
                "target_audience": "target_audience",
                "llm": "gemini_2_flash",
            },
        },
        # Add more steps here as needed...
    ]

    # --- 2) Initialize session states for each step ---
    for step in pipeline_steps:
        step_key = f"{step['name']}_done"
        if step_key not in st.session_state:
            st.session_state[step_key] = False

    # --- 3) UI: Ask for user inputs (shared across steps) ---
    root_folder_id = st.text_input("Enter course Drive folder ID")
    sheet_link = st.text_input("Enter Google Sheet link")

    # Button to load data
    if st.button("Load Data"):
        load_dotenv()  # Load env variables from .env
        try:
            gauth = login_with_service_account("content/service-credentials.json")
            drive = GoogleDrive(gauth)

            gc = gspread.service_account(filename='content/service-credentials.json')
            # gc = gspread.oauth(
            #     credentials_filename='content/oauth-credentials.json',
            #     authorized_user_filename='content/authorized_user.json',
            # )
            sheet = gc.open_by_url(sheet_link)
            course_info_sheet, course_info_df = get_sheet_data_and_df(sheet, 'Course info')

            st.session_state["root_folder_id"] = root_folder_id
            st.session_state["sheet"] = sheet
            st.session_state["course_name"] = course_info_df['Course Name'][0]
            st.session_state["target_audience"] = course_info_df['Target Audience & Industry'][0]

            # If you have a 'drive' object defined:
            st.session_state["drive"] = drive

            st.success("Data loaded successfully!")
        except Exception as e:
            st.error(f"Error loading data: {e}")
            # Display the full stack trace
            st.text(traceback.format_exc())

    # --- 4) Display pipeline steps in order ---
    # Only proceed if data is loaded (i.e. "sheet" in st.session_state).
    if "sheet" in st.session_state:
        for idx, step in enumerate(pipeline_steps, start=1):
            step_key = f"{step['name']}_done"
            st.subheader(f"{idx}. Run the {step['name']}")

            # Check if dependencies are satisfied:
            dependencies_satisfied = all(
                st.session_state.get(f"{dep}_done", False) for dep in step["depends_on"]
            )

            if not dependencies_satisfied:
                # If dependencies are not done, show a message & skip
                missing_steps = [
                    dep for dep in step["depends_on"] 
                    if not st.session_state.get(f"{dep}_done", False)
                ]
                missing_list = ", ".join(missing_steps)
                st.write(
                    f"Waiting on these steps to be done: {missing_list}"
                )
                continue

            # If step is not done yet, show the button; else show "Done"
            if not st.session_state[step_key]:
                # Check if step is manual input or not
                if "instructions" in step:
                    for instruction in step["instructions"]:
                        st.write(instruction)
                    button_name = f"Confirm {step['name']}"
                    button_type = "primary"
                else:
                    button_name = f"Run {step['name']}"
                    button_type = "secondary"
                if st.button(label = button_name, type = button_type):
                    try:
                        # Collect real arguments from session_state
                        kwargs = {}
                        for arg_name, session_key in step["args"].items():
                            # e.g. arg_name="root_folder_id", session_key="root_folder_id"
                            # If the session_key is a string that matches a valid session_state key, retrieve it
                            if isinstance(session_key, str) and session_key in st.session_state:
                                kwargs[arg_name] = st.session_state[session_key]
                            else:
                                # or if it's a literal value, put that
                                kwargs[arg_name] = session_key

                        if "instructions" in step: # If it is a manual input type function
                            # st.write(step["instructions"])
                            response = step["func"](**kwargs)
                            if response:
                                st.session_state[step_key] = True
                                st.success(f"{step['name']} completed!")
                            else:
                                st.warning(f"{step['name']} not completed!")
                        else:
                            # Run the actual function
                            step["func"](**kwargs)
                            st.session_state[step_key] = True
                            st.success(f"{step['name']} completed!")
                    except Exception as e:
                        st.error(f"Error running {step['name']}: {e}")
                        # Display the full stack trace
                        st.text(traceback.format_exc())
            else:
                st.write(f"{step['name']}: **Done**")
                if idx == len(pipeline_steps):
                    st.balloons()
                    st.toast("You have successfully generated the Research Notes", icon = ":material/done_all:")

    st.session_state

# if __name__ == "__main__":
main()
