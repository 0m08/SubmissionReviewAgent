from agent_ui_template import agent_ui

from agents.graphics_definition.define_graphics.revise_graphics_definition import run_generate_and_revise_graphics
from agents.graphics_definition.define_graphics.revise_graphics_definition import run_revise_generated_graphics_definition

from agents.graphics_definition.graphics_definition_checklist.checklist_reviser import run_checklist_evaluation_for_all_slides


pipeline_sections = [
    {
        "section_name": "Section 1: Graphics Definition Generation",
        "steps": [
            {
                "name": "Generate Graphics Definition",
                "func": run_generate_and_revise_graphics,
                "depends_on": [],
                "args": {
                    "sheet": "sheet",
                    "worksheet_name": "Slide Chunks",
                    "course_name": "course_name",
                    "target_audience": "target_audience",
                    "llm": "gemini_2_flash"
                },
                "instructions": [
                    "1. This step will generate the graphics definition for each row in the `Slide Chunks` sheet.",
                    "2. The generated graphics definition will be saved in the `graphics_definition` sheet.",
                    "3. Open the Google Sheet and navigate to the `Slide Chunks` worksheet.",
                    "4. Locate the `graphics_definition` column the current slide.",
                    "5. **Manually enter feedback**(Optional) in the `human_review` column.",
                ],
                "estimated_time": "~ Semi-Automated Step",
                "description": "Review the generated graphics definition in the `graphics_definition` sheet. Add comments in the `human_review` column (Optional)."
            },
        #     {
        #         "name": "Revise Graphics Definition",
        #         "func": run_revise_generated_graphics_definition,
        #         "depends_on": [],
        #         "args": {
        #             "sheet": "sheet",
        #             "worksheet_name": "Slide Chunks",
        #             "course_name": "course_name",
        #             "target_audience": "target_audience",
        #             "llm": "gemini_2_flash"
        #         },
        #         "estimated_time": "~ Semi-Automated Step",
        #         "description": "Generates revised graphics definition for all the rows in the `Slide Chunks` sheet.",
        #     },
        ],
    },
        
            
        
        
    {
        "section_name": "Section 2: Graphics Definition Checklist",
        "steps": [
            {
                "name": "Checklist Evaluation",
                "func": run_checklist_evaluation_for_all_slides,
                "depends_on": ["Generate Graphics Definition"],
                "args": {
                    "sheet": "sheet",
                    # "checklist_sheet": "Graphics Definition Checklist",
                    "worksheet_name": "Slide Chunks",
                    "course_name": "course_name",
                    "target_audience": "target_audience",
                    "llm": "gemini_2_flash"
                },
                "estimated_time": "~ 5 - 10 minutes",
                "description": "Generates client reference based outlines for all the rows in the `Graphics Definition Checklist` sheet.",
            },
        ],
    }
]

agent_ui(step_name = "Graphics Definition", pipeline_sections = pipeline_sections)