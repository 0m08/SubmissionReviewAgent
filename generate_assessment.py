from agent_ui_template import agent_ui

from agents.generate_assessments.run_review_and_reviser import run_update_assessment_questions
from agents.generate_assessments.generate_assessment_checklist import run_update_assessment_checklist
from agents.generate_assessments.generate_review_agent_checklist import run_update_checklist_with_verdicts_preserve


pipeline_sections = [
    {
        "section_name": "Section 1: Assessement Questions Generation",
        "steps": [
            
            {
                "name": "Generate and Update Assessment Questions",
                "func": run_update_assessment_questions,
                "depends_on": [],
                "args": {
                    "sheet": "sheet",
                    "worksheet_name": "Slide Chunks",
                    "course_name": "course_name",
                    "target_audience": "target_audience",
                    "llm": "gemini_2_flash"
                },
            
                "estimated_time": "~ 30 - 50 seconds",
                "description": "This function generates assessment questions for a given course."
            },
            
            {
                "name": "Update Assessment Checklist",
                "func": run_update_assessment_checklist,
                "depends_on": ['Generate and Update Assessment Questions'],
                "args": {
                    "sheet": "sheet",
                    "worksheet_name": 'Assessment Checklist',
                    "course_name": "course_name",
                    "target_audience": "target_audience"
                },
            
                "estimated_time": "~ 5-10 minutes",
                "description": "This function updates the assessment checklist for a given course."
            },
            
            {
                "name": "Update Review Agent Checklist",
                "func": run_update_checklist_with_verdicts_preserve,
                "depends_on": ['Update Assessment Checklist'],
                "args": {
                    "sheet": "sheet",
                    "worksheet_name": 'Review Agent Checklist'
                },
            
                "estimated_time": "~ 30 - 50 seconds",
                "description": "This function updates the review agent checklist for a given course."
            },
        ],
    },
]

agent_ui(step_name = "Assessments Generation", pipeline_sections = pipeline_sections)