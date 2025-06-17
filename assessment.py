from agent_ui_template import agent_ui

from agents.generate_assessments.generate_assessment_questions import (
    run_generate_assessment_question,
    delete_assessment_questions,
)
from agents.generate_assessments.run_review_and_reviser import (
    run_review_and_revise_all_questions,
    delete_final_assessment,
)
from agents.generate_assessments.generate_assessment_checklist import (
    run_generate_assessment_checklist,
    delete_assessment_checklist,
)
from agents.generate_assessments.generate_review_agent_checklist import (
    run_update_checklist_with_verdicts_preserve,
    delete_review_agent_checklist,
)


pipeline_sections = [
    {
        "section_name": "Section 1: Assessment",
        "steps": [
            
            {
                "name": "Generate Assessment Questions",
                "func": run_generate_assessment_question,
                "depends_on": [],
                "args": {
                    "sheet": "sheet",
                    "worksheet_name": "Slide Chunks",
                    "course_name": "course_name",
                    "target_audience": "target_audience",
                    "llm": "gemini_2_flash"
                },
                "delete_func": delete_assessment_questions,
                "delete_args": {
                    "sheet": "sheet",
                    "worksheet_name": "Assessment questions",
                },
            
                "estimated_time": "> 1 minute",
                "description": "This function generates assessment questions for a given course."
            },
        
            
            {
                "name": "Review and Revise all Assessment Questions",
                "func": run_review_and_revise_all_questions,
                "depends_on": ['Generate Assessment Questions'],
                "args": {
                    "sheet": "sheet",
                    "worksheet_name": "Slide Chunks",
                    "course_name": "course_name",
                    "target_audience": "target_audience",
                    "llm": "gemini_2_flash"
                },
                "delete_func": delete_final_assessment,
                "delete_args": {
                    "sheet": "sheet",
                    "worksheet_name": "Final Assessment",
                },
            
                "estimated_time": " 15-20 minutes",
                "description": "This function reviews and revises all assessment questions for a given course."
            },
        
            
            {
                "name": "Update Assessment Checklist",
                "func": run_generate_assessment_checklist,
                "depends_on": ['Review and Revise all Assessment Questions'],
                "args": {
                    "sheet": "sheet",
                    "worksheet_name": 'Slide Chunks',
                    "course_name": "course_name",
                    "target_audience": "target_audience",
                    "llm": "gemini_2_flash"
                },
                "delete_func": delete_assessment_checklist,
                "delete_args": {
                    "sheet": "sheet",
                    "worksheet_name": "Assessment Checklist",
                },
            
                "estimated_time": "~ 1 minute",
                "description": "This function updates the assessment checklist for a given course."
            },
            
            {
                "name": "Update Review Agent Checklist",
                "func": run_update_checklist_with_verdicts_preserve,
                "depends_on": ['Update Assessment Checklist'],
                "args": {
                    "sheet": "sheet",
                    "worksheet_name": 'Review Agent Checklist',
                    "course_name": "course_name",
                    "target_audience": "target_audience",
                    "llm": "gemini_2_flash"
                },
                "delete_func": delete_review_agent_checklist,
                "delete_args": {
                    "sheet": "sheet",
                    "worksheet_name": "Review Agent Checklist",
                },
            
                "estimated_time": "~ 20 - 40 minutes",
                "description": "This function updates the review agent checklist for a given course."
            },
        ],
    },
]

agent_ui(step_name = "Assessment", pipeline_sections = pipeline_sections)
