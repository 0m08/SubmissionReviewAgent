from agent_ui_template import agent_ui
import streamlit as st
#from services.sheets_service import get_worksheet_names


def noop(*args, **kwargs):
    """Placeholder delete function for manual steps."""
    return

#from agents.research_notes.retriever_agent import run_retriever_agent_for_all_rows
from agents.research_notes.generate_notes import (
    run_research_notes_agent_for_all_rows,
    delete_research_notes,
)
from agents.research_notes.review_subtopic_notes import (
    run_reviewer_agent_for_all_rows,
    manual_input_review_research_notes,
    delete_subtopic_review,
)
from agents.research_notes.revise_subtopic_notes import (
    run_reviser_agent_for_all_rows,
    manual_input_review_revised_research_notes,
    delete_revised_research_notes,
)
from agents.research_notes.create_research_notes_sheet import (
    create_research_notes_sheet,
    delete_research_notes_sheet,
)
from agents.research_notes.review_topic_notes import (
    run_review_topic_notes_agent_for_all_rows,
    manual_input_review_topic_notes,
    delete_review_topic_notes,
)
from agents.research_notes.revise_topic_notes import (
    run_revise_topic_notes_for_all_rows,
    delete_revised_topic_notes,
)
#from agents.research_notes.load_references import load_references  # New import

# Determine which outline sheet to use
# outline_worksheet = "Enhanced Outline with LOs"
# if "sheet" in st.session_state:
#     worksheet_names = get_worksheet_names(st.session_state["sheet"])
#     if "Enhanced Outline with LOs" not in worksheet_names:
#         outline_worksheet = "Course Outline with LOs"

# --- 1) Define pipeline as sections, each with its own steps ---
pipeline_sections = [
    {
        "section_name": "Section 1: Subtopic Research",
        "steps": [
            # {
            #     "name": "Load References",
            #     "func": load_references,
            #     "depends_on": [],
            #     "args": {
            #         "sheet": "sheet",
            #     },
            #     "estimated_time": "~ 2-5 minutes",
            #     "description": "Loads all reference documents into the vectorstore for faster retrieval.",
            # },
            # {
            #     "name": "Retriever",
            #     "func": run_retriever_agent_for_all_rows,
            #     "depends_on": [],
            #     "args": {
            #         "root_folder_id": "root_folder_id",
            #         "drive": "drive",
            #         "sheet": "sheet",
            #         "worksheet_name": outline_worksheet,
            #         "course_name": "course_name",
            #         "target_audience": "target_audience",
            #         "llm": "gemini_2_flash",
            #     },
            #     "estimated_time": "~ 10 - 20 minutes",
            #     "description": "Gathers relevant context needed for the research.",
            # },
            {
                "name": "Researcher",
                "func": run_research_notes_agent_for_all_rows,
                "depends_on": [],
                "args": {
                    "sheet": "sheet",
                    "worksheet_name": "Final Outline",
                    "course_name": "course_name",
                    "target_audience": "target_audience",
                    "llm": "gemini_2_flash",
                },
                "estimated_time": "~ 5 minutes",
                "description": "Uses the context retrived earlier to produce initial research notes for each subtopic.",
                "delete_func": delete_research_notes,
                "delete_args": {
                    "sheet": "sheet",
                    "worksheet_name": "Final Outline",
                }
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
                    "worksheet_name": "Final Outline",
                    "course_name": "course_name",
                    "target_audience": "target_audience",
                    "llm": "gemini_2_flash",
                },
                "estimated_time": "~ 2 minutes",
                "description": "Reviews the subtopic research notes and suggests changes or improvements.",
                "delete_func": delete_subtopic_review,
                "delete_args": {
                    "sheet": "sheet",
                    "worksheet_name": "Final Outline",
                }
            },
            {
                "name": "Manual Review",
                "func": manual_input_review_research_notes,
                "depends_on": ["Reviewer"],
                "args": {
                    "sheet": "sheet",
                    "worksheet_name": "Final Outline",
                },
                "instructions": [
                    f"Review the research notes, and the AI-generated review within the `Final Outline` sheet.",
                    "Column names:\n `research_notes`,\n `analysis`,\n `not_covered_at_all`,\n `not_covered_enough`,\n `perfectly_covered`,\n `covered_too_much`,\n `verdict`",
                    "---",
                    "1. **Review and edit** the category columns (`not_covered_at_all`, `not_covered_enough`, etc.) as needed to incorporate your final thoughts.",
                    "2. If something stands out or the outline needs more changes, add your thoughts in the **`Manual Comments`** column. This is optional—but strongly recommended for rows where the verdict is **Fail**.",
                    "---",
                    "Example",
                    "- In `not_covered_at_all`, add explanation of all types of manometers: U-tube, digital, inclined.",
                    "- In `covered_too_much`, Remove detailed measurement methods — will be covered in another course.",
                    "- In `Manual Comments` consider rephrasing the learning objective to focus on application instead of theory.",

                    "---",
                    "All of the above are just examples of what can be added / edited.",
                    "If you **agree** with the AI's review, you can **leave the row as is**.",
                    "If you **disagree**, update the appropriate columns accordingly.",
                    "Make sure all rows are reviewed."
                ],
                "is_manual_step": True,
                "estimated_time": "Manual step",
                "description": "Manually verify and refine the AI review for each subtopic note.",
                "video_link": "https://drive.google.com/file/d/1o5LabDPkV8LjFx-yA_XK28P6YlZChXIq/view?usp=drive_link",
                "delete_func": noop,
            },
            {
                "name": "Reviser",
                "func": run_reviser_agent_for_all_rows,
                "depends_on": ["Manual Review"],
                "args": {
                    "root_folder_id": "root_folder_id",
                    "drive": "drive",
                    "sheet": "sheet",
                    "worksheet_name": "Final Outline",
                    "course_name": "course_name",
                    "target_audience": "target_audience",
                    "llm": "gemini_2_flash",
                },
                "estimated_time": "~ 10 minutes",
                "description": "AI-driven revision of the subtopic research notes based on your manual inputs.",
                "delete_func": delete_revised_research_notes,
                "delete_args": {
                    "sheet": "sheet",
                    "worksheet_name": "Final Outline",
                }
                
            },
            {
                "name": "Manual Review Revised Notes",
                "func": manual_input_review_revised_research_notes,
                "depends_on": ["Reviser"],
                "args": {
                    "sheet": "sheet",
                    "worksheet_name": "Final Outline",
                    "course_name": "course_name",
                    "target_audience": "target_audience",
                    "skip_manual_step": "skip_manual_step",
                },
                "instructions": [
                    "Review the `revised_research_notes` column within the `Final Outline` sheet.",
                    "Make any edits—such as deleting specific or extra information—directly in that column.",
                    "Once finished, proceed to the next step.",
                ],
                "is_manual_step": True,
                "estimated_time": "Manual step",
                "description": "Manually refine the revised research notes before continuing.",
                "delete_func": noop,
            },
        ],
    },
    {
        "section_name": "Section 3: Topic-Level Research Notes",
        "steps": [
            {
                "name": "Create Research Notes Sheet",
                "func": create_research_notes_sheet,
                "depends_on": ["Manual Review Revised Notes"],
                "args": {
                    "sheet": "sheet",
                    "source_worksheet_name": "Final Outline",
                    "target_worksheet_name": "Research Notes",
                },
                "estimated_time": "A few seconds",
                "description": "Creates a new sheet with aggregated research notes for broader topics.",
                "delete_func": delete_research_notes_sheet,
                "delete_args": {
                    "sheet": "sheet",
                    "worksheet_name": "Research Notes",
                },
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
                "estimated_time": "~ 2 minutes",
                "description": "AI-driven review of the newly created topic-level research notes.",
                "delete_func": delete_review_topic_notes,
                "delete_args": {
                    "sheet": "sheet",
                    "worksheet_name": "Research Notes",
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
                    "Review the research notes, and the ai generated review within the `Research Notes` sheet.",
                    "Column names:\n `research_notes`,\n `final_review`",
                    "---",
                    "- Go through each row and review the AI's `final_review` in context of the `research_notes`.",
                    "- If you have anything to add, clarify, or correct, use the **`Manual Comments`** column to leave your notes",
                    "---",
                    "Manual comments are optional but helpful—especially if the review needs changes or extra clarity.",
                    "If everything looks good, you can leave the row as is."
                    
                ],
                "is_manual_step": True,
                "estimated_time": "Manual step",
                "description": "Manually verify and refine AI's review for each topic-level note.",
                "video_link": "https://drive.google.com/file/d/1IwSqwFzI3vxNHTsmfpQUw62hyTzjyUPx/view?usp=drive_link",
                "delete_func": noop,

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
                "estimated_time": "~ 4 minutes",
                "description": "AI-driven revision of the topic-level notes, completing the pipeline.",
                "delete_func": delete_revised_topic_notes,
                "delete_args": {
                    "sheet": "sheet",
                    "worksheet_name": "Research Notes",
                }
            },
        ]
    }
]


agent_ui(step_name = "Research Notes", pipeline_sections = pipeline_sections)
