from agent_ui_template import agent_ui

from agents.research_notes.retriever_agent import run_retriever_agent_for_all_rows
from agents.research_notes.generate_notes import run_research_notes_agent_for_all_rows
from agents.research_notes.review_subtopic_notes import run_reviewer_agent_for_all_rows, manual_input_review_research_notes
from agents.research_notes.revise_subtopic_notes import run_reviser_agent_for_all_rows
from agents.research_notes.create_research_notes_sheet import create_research_notes_sheet
from agents.research_notes.review_topic_notes import run_review_topic_notes_agent_for_all_rows, manual_input_review_topic_notes
from agents.research_notes.revise_topic_notes import run_revise_topic_notes_for_all_rows
from agents.research_notes.load_references import load_references  # New import

# --- 1) Define pipeline as sections, each with its own steps ---
pipeline_sections = [
    {
        "section_name": "Section 1: Subtopic Research",
        "steps": [
            {
                "name": "Load References",
                "func": load_references,
                "depends_on": [],
                "args": {
                    "sheet": "sheet",
                },
                "estimated_time": "~ 2-5 minutes",
                "description": "Loads all reference documents into the vectorstore for faster retrieval.",
            },
            {
                "name": "Retriever",
                "func": run_retriever_agent_for_all_rows,
                "depends_on": [],
                "args": {
                    "root_folder_id": "root_folder_id",
                    "drive": "drive",
                    "sheet": "sheet",
                    "worksheet_name": "Enhanced Outline with LOs",
                    "course_name": "course_name",
                    "target_audience": "target_audience",
                    "llm": "gemini_2_flash",
                },
                "estimated_time": "~ 10 - 20 minutes",
                "description": "Gathers relevant context needed for the research.",
            },
            {
                "name": "Researcher",
                "func": run_research_notes_agent_for_all_rows,
                "depends_on": ["Retriever"],
                "args": {
                    "sheet": "sheet",
                    "worksheet_name": "Enhanced Outline with LOs",
                    "course_name": "course_name",
                    "target_audience": "target_audience",
                    "llm": "gemini_2_flash",
                },
                "estimated_time": "~ 5 minutes",
                "description": "Uses the context retrived earlier to produce initial research notes for each subtopic."
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
                    "worksheet_name": "Enhanced Outline with LOs",
                    "course_name": "course_name",
                    "target_audience": "target_audience",
                    "llm": "gemini_2_flash",
                },
                "estimated_time": "~ 2 minutes",
                "description": "Reviews the subtopic research notes and suggests changes or improvements.",
            },
            {
                "name": "Manual Review",
                "func": manual_input_review_research_notes,
                "depends_on": ["Reviewer"],
                "args": {
                    "sheet": "sheet",
                    "worksheet_name": "Enhanced Outline with LOs",
                },
                "instructions": [
                    "Review the research notes, and the ai generated review within the `Enhanced Outline with LOs` sheet.",
                    "Column names:\n `research_notes`,\n `analysis`,\n `not_covered_at_all`,\n `not_covered_enough`,\n `perfectly_covered`,\n `covered_too_much`,\n `verdict`",
                    "---",
                    "1) Edit any of the category columns to incorporate your final review.",
                    "2) You can put any additional comments in the `Manual Comments` column. This is optional, but recommended for all the Failed rows.",
                    "3) For example, you can add items such as 'Cover all types of manometer include A, B, C' within the not covered columns.",
                    "4) Or say something like 'Measurement will be covered in more detail in a separate course, thus remove it from here' within the covered to much column.",
                    "5) Similarly, you can add any addition comments such as 'restructure / change the phrasing of this part' in the Manual comments column",
                    "---",
                    "All of the above are just examples of what can be added / edited.",
                    "You are expected to review all the rows. If you agree with the AI's review, leave it as is. If you disagree, add them in appropriate columns as illustrated with the above examples."
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
                    "worksheet_name": "Enhanced Outline with LOs",
                    "course_name": "course_name",
                    "target_audience": "target_audience",
                    "llm": "gemini_2_flash",
                },
                "estimated_time": "~ 10 minutes",
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
                    "source_worksheet_name": "Enhanced Outline with LOs",
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
                "estimated_time": "~ 2 minutes",
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
                    "Review the research notes, and the ai generated review within the `Research Notes` sheet.",
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
                "estimated_time": "~ 4 minutes",
                "description": "AI-driven revision of the topic-level notes, completing the pipeline."
            },
        ]
    }
]


agent_ui(step_name = "Research Notes", pipeline_sections = pipeline_sections)
