import streamlit as st
import gspread
from dotenv import load_dotenv

from agents.research_notes.retriever_agent import run_retriever_agent_for_all_rows
from agents.research_notes.research_notes_agent import run_research_notes_agent_for_all_rows
from agents.research_notes.retriever_agent import manual_input_review_context

from services.sheets_service import get_sheet_data_and_df
from services.drive_service import drive


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
            }
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
            gc = gspread.service_account(filename='content/service-credentials.json')
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
                if st.button(f"Run {step['name']}"):
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

                        # Run the actual function
                        step["func"](**kwargs)

                        st.session_state[step_key] = True
                        st.success(f"{step['name']} completed!")
                    except Exception as e:
                        st.error(f"Error running {step['name']}: {e}")
            else:
                st.write(f"{step['name']}: **Done**")


# if __name__ == "__main__":
main()
