from agent_ui_template import agent_ui

from agents.graphics_definition.define_graphics.run_generate_revise_graphics_definition import run_generate_and_revise_graphics, ensure_reference_description_column
from agents.graphics_definition.graphics_definition_checklist.checklist_reviser import run_checklist_evaluation_for_all_slides


pipeline_sections = [
    {
        "section_name": "Section 1: Graphics Definition Generation",
        "steps": [
            {
                "name": "Generate Graphics Definition",
                "func": run_generate_and_revise_graphics,
                "pre_exec_func": ensure_reference_description_column, 
                "pre_exec_args": {
                    "sheet": "sheet",
                    "worksheet_name": "Slide Chunks"
                },
                "depends_on": [],
                "args": {
                    "sheet": "sheet",
                    "worksheet_name": "Slide Chunks",
                    "course_name": "course_name",
                    "target_audience": "target_audience",
                    "skip_manual_step":"skip_manual_step",
                    "llm": "gemini_2_flash"
                },
                "instructions": [
                    "1. This step will generate the graphics definition for each slide in the `Slide Chunks` sheet.",
                    "2. The generated graphics definition will be saved in the `graphics_definition` column of the `Slide Chunks` sheet.",
                    "3. Open the Google Sheet and navigate to the `Slide Chunks` sheet.",
                    "4. Locate the `graphics_definition` column of the current slide.",
                    "5. **Manually enter feedback** (Optional) in the `human_review` column. This feedback should be regarding the changes you want in the generated graphics definition in the `graphics_definition` column.",
                ],
                "estimated_time": "~ Semi-Automated Step",
                "description": "Review the generated graphics definition in the `graphics_definition` column of the `Slide Chunks` sheet. Add comments in the human_review column (Optional)"
            },
      
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
                "description": "Generates revised Graphics Definition based on the review of the generated graphics definitions on the checklist criteria. The final Graphics Definitions are populated in the `checklist_revised_graphics_definition` column.",
            },
        ],
    }
]

agent_ui(step_name = "Graphics Definition", pipeline_sections = pipeline_sections)
