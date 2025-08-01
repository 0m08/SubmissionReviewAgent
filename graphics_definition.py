from agent_ui_template import agent_ui

from agents.graphics_definition.define_graphics.run_generate_revise_graphics_definition import (
    run_generate_and_revise_graphics,
    ensure_reference_description_column,
    delete_graphics_definition_generation,
)
from agents.graphics_definition.graphics_definition_checklist.checklist_reviser import (
    run_checklist_evaluation_for_all_slides,
    delete_graphics_definition_checklist,
)
from agents.graphics_definition.short_grahphics_definition.generate_short_graphic_definitions import (
    run_generate_short_graphic_definitions,
    delete_short_graphic_definitions,
)

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
                    "1. Before running this step, open the `Slide Chunks` sheet, review all the Slide content in the `Content` column, and populate the `Reference Description` column for any slides where you have visual suggestions, references, or instructions — or leave it blank if you don’t have input for a particular slide. These suggestions will be used by the agent as references while generating the graphics definition for each slide.",
                    "2. Click on **Confirm Generate Graphics Definition** button to start processing the first slide.",
                    "3. The agent will generate a scene-by-scene graphics definition and 4 AI reviews: complexity, missing sentences, accuracy, and reuse previous graphics.",
                    "4. Navigate to the `Slide Chunks` sheet and review the generated graphics definition in the `graphics_definition` column.",
                    "5. You can optionally provide feedback in the `human_review` column to suggest improvements for the generated graphics definition.",
                    "6. After entering your feedback (if any), click on **Confirm Generate Graphics Definition** to generate the revised graphics definition based on your feedback and the AI reviews in the `revised_graphics_definition` column.",
                    "7. The agent will automatically move to the next slide and repeat the same process."   
                ],
                "is_manual_step": True,
                "estimated_time": "~ Semi-Automated Step",
                "description": "Review the generated graphics definition in the `graphics_definition` column of the `Slide Chunks` sheet. Add comments in the human_review column (Optional)",
                "video_link": "https://drive.google.com/file/d/14pis5XLbK30_u23Amyek-uLw5bFr8Cm4/view?usp=sharing",
                "delete_func": delete_graphics_definition_generation,
                "delete_args": {
                    "sheet": "sheet",
                    "worksheet_name": "Slide Chunks",
                }
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
                "description": "Generates revised Graphics Definition based on the review of the generated graphics definitions on the checklist criteria. The final Graphics Definitions are populated in the `Graphics Definition` column.",
                "delete_func": delete_graphics_definition_checklist,
                "delete_args": {
                    "sheet": "sheet",
                    "worksheet_name": "Slide Chunks",
                },
            },
        ],
    },

    {
        "section_name": "Section 3: Short Graphics Definition",
        "steps": [
            {
                "name": "Short Graphics Definition",
                "func": run_generate_short_graphic_definitions,
                "depends_on": ["Checklist Evaluation"],
                "args": {
                    "sheet": "sheet",
                    "worksheet_name": "Slide Chunks",
                    "llm": "gemini_2_flash"
                },
                "is_manual_step": False,
                "estimated_time": "~ 5 - 10 minutes",
                "description": "Generates short graphics definitions for each slide based on the revised graphics definitions.",
                "delete_func": delete_short_graphic_definitions,
                "delete_args": {
                    "sheet": "sheet",
                    "worksheet_name": "Slide Chunks",
                },
            },
        ],
    }
]

agent_ui(step_name = "Graphics Definition", pipeline_sections = pipeline_sections)
