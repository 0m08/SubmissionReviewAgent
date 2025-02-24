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

    # --- 1) Define pipeline as sections, each with its own steps ---
    pipeline_sections = [
        {
            "section_name": "Section 1: Subtopic Research",
            "steps": [
                {
                    "name": "Retriever",
                    "func": run_retriever_agent_for_all_rows,
                    "depends_on": [],
                    "args": {
                        "root_folder_id": "root_folder_id",
                        "drive": "drive",
                        "sheet": "sheet",
                        "worksheet_name": "Course Outline with LOs",
                        "course_name": "course_name",
                        "target_audience": "target_audience",
                        "llm": "gemini_2_flash",
                    },
                    "estimated_time": "~1 minute",
                    "description": "Gathers relevant drive files, topics, or references needed for the research.",
                },
                {
                    "name": "Retriever Manual Input",
                    "func": manual_input_review_context,
                    "depends_on": ["Retriever"],
                    "args": {
                        "sheet": "sheet",
                        "worksheet_name": "Course Outline with LOs",
                    },
                    "instructions": [
                        """Manually add the following column `Context Review`
in the tab and mark it as Done."""
                    ],
                    "estimated_time": "Manual step",
                    "description": "Review the retriever’s output and add manual notes if needed.",
                },
                {
                    "name": "Researcher",
                    "func": run_research_notes_agent_for_all_rows,
                    "depends_on": ["Retriever"],
                    "args": {
                        "sheet": "sheet",
                        "worksheet_name": "Course Outline with LOs",
                        "course_name": "course_name",
                        "target_audience": "target_audience",
                        "llm": "gemini_2_flash",
                    },
                    "estimated_time": "~2-3 minutes",
                    "description": "Uses the retriever data to produce initial research notes for each subtopic."
                },
            ],
        },
        {
            "section_name": "Section 2: Subtopic Review & Revision",
            "steps": [
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
                    "estimated_time": "~1 minute",
                    "description": "Reviews the subtopic research notes and suggests changes or improvements.",
                },
                {
                    "name": "Manual Review",
                    "func": manual_input_review_research_notes,
                    "depends_on": ["Reviewer"],
                    "args": {
                        "sheet": "sheet",
                        "worksheet_name": "Course Outline with LOs",
                    },
                    "instructions": [
                        "Review the research notes, and the ai generated review.",
                        "Column names:\n `research_notes`,\n `analysis`,\n `not_covered_at_all`,\n `not_covered_enough`,\n `perfectly_covered`,\n `covered_too_much`,\n `verdict`",
                        "---",
                        "Edit any of the category columns to incorporate your final review.",
                        "You can put any additional comments in the `Manual Comments` column.",
                        "---"
                    ],
                    "estimated_time": "Manual step",
                    "description": "Manually verify and refine the AI review for each subtopic note.",
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
                    "estimated_time": "~1 minute",
                    "description": "AI-driven revision of the subtopic research notes based on your manual inputs."
                },
            ],
        },
        {
            "section_name": "Section 3: Topic-Level Research Notes",
            "steps": [
                {
                    "name": "Create Research Notes Sheet",
                    "func": create_research_notes_sheet,
                    "depends_on": ["Reviser"],
                    "args": {
                        "sheet": "sheet",
                        "source_worksheet_name": "Course Outline with LOs",
                        "target_worksheet_name": "Research Notes",
                    },
                    "estimated_time": "A few seconds",
                    "description": "Creates a new sheet with aggregated research notes for broader topics.",
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
                    "estimated_time": "~1 minute",
                    "description": "AI-driven review of the newly created topic-level research notes.",
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
                    ],
                    "estimated_time": "Manual step",
                    "description": "Manually verify and refine AI’s review for each topic-level note."
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
                    "estimated_time": "~1 minute",
                    "description": "AI-driven revision of the topic-level notes, completing the pipeline."
                },
            ]
        }
    ]

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
                st.session_state["drive"] = drive

                st.success("Data loaded successfully!")
            except Exception as e:
                st.error(f"Error loading data: {e}")
                # Display the full stack trace
                st.text(traceback.format_exc())
    else:
        # If data is already loaded, simply confirm it to the user
        st.success("Data already loaded. Proceed below.")

    # --- 4) Display pipeline steps in nested sections ---
    if "sheet" in st.session_state:
        step_global_count = 1  # So we can label steps 1,2,3 across sections
        for section_idx, section in enumerate(pipeline_sections, start=1):
            with st.container(border=True):
                st.header(section["section_name"])
                for step in section["steps"]:
                    step_key = f"{step['name']}_done"
                    
                    # Check if dependencies are satisfied
                    dependencies_satisfied = all(
                        st.session_state.get(f"{dep}_done", False)
                        for dep in step["depends_on"]
                    )

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
                        st.write(f"{step['name']}: **Done**")

                        # If this is the very last step in the entire pipeline, celebrate
                        if (
                            section_idx == len(pipeline_sections)
                            and step == section["steps"][-1]
                        ):
                            st.balloons()
                            st.toast(
                                "You have successfully generated the Research Notes",
                                icon=":material/done_all:",
                            )

    # Debug
    # st.session_state

# if __name__ == "__main__":
main()
