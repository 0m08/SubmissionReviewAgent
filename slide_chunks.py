from agent_ui_template import agent_ui

from agents.slide_chunks.research_notes_parsing import (
    run_research_notes_parsing,
    delete_slide_chunks_sheet,
)
from agents.slide_chunks.generate_los import (
    run_learning_objectives_agent,
    delete_learning_objectives_slide_chunks,
)
from agents.slide_chunks.review_and_revise_checklist import (
    run_checklist_review_and_revise,
    delete_checklist_review_and_revise,
)
from agents.slide_chunks.generate_slide_chunks import generate_slide_chunks_from_research_notes_for_all_subtopics, delete_slide_chunks_generation
from agents.slide_chunks.slide_chunks_checklist import run_slide_chunks_checklist_and_reviser
from agents.slide_chunks.slide_chunks_parsing import run_slide_chunks_parsing, delete_slide_chunks_sheet
#from agents.slide_chunks.ai_detection_review_revise import run_ai_detection_review_revise
#from agents.slide_chunks.winston_ai_plagiarism_detection import run_plagiarism_detection
#from agents.slide_chunks.winston_ai_detection_with_readability import run_ai_detection_with_readability

pipeline_sections = [
    {
        "section_name": "Section 1: Generate Slide Chunks from Research Notes and Parse it",
        "steps": [
            {
                "name": "Generate Slide Chunks from the Research Notes",
                "func": generate_slide_chunks_from_research_notes_for_all_subtopics,
                "depends_on": [],
                "args": {
                    "sheet": "sheet",
                    "sheet_name": "Final Outline",
                    "llm": "gemini_2_flash"
                },
                "estimated_time": "5-10 minutes",
                "description": "This function generates slide chunks for each unique subtopic in the Final Outline sheet using the research notes and fills the slide_chunks column.",
                "delete_func": delete_slide_chunks_generation,
                "delete_args": {
                    "sheet": "sheet",
                    "worksheet_name": "Final Outline"
                }
            },            
            {
                "name": "Slide Chunks Parsing",
                "func": run_slide_chunks_parsing,
                "depends_on": ["Generate Slide Chunks from the Research Notes"],
                "args": {
                    "sheet": "sheet",
                    "worksheet_name": "Final Outline",
                    "output_sheet_name": "Slide Chunks"
                },
                "delete_func": delete_slide_chunks_sheet,
                "delete_args": {
                    "sheet": "sheet",
                    "worksheet_name": "Slide Chunks"
                },
                "estimated_time": "5-10 minutes",
                "description": "This function parses the slide_chunks column in the Final Outline sheet, extracts structured slide data, and writes it to the Slide Chunks sheet."
            },
            {
                "name": "Slide Chunks Checklist Review and Revise",
                "func": run_slide_chunks_checklist_and_reviser,
                "depends_on": ["Slide Chunks Parsing"],
                "args": {
                    "sheet": "sheet",
                    "course_name": "course_name",
                    "target_audience": "target_audience",
                    "checklist_sheet_link": "checklist_sheet_link",
                    "gc": "gc",
                    "llm": "gemini_2_flash"
                },
                "estimated_time": "15-30 minutes",
                "description": "This function reviews and revises the parsed slide chunks based on a checklist to ensure quality and compliance."
            },
            # {
            #     "name": "Research Notes Parsing",
            #     "func": run_research_notes_parsing,
            #     "depends_on": ["Slide Chunks Parsing"],
            #     "args": {
            #         "sheet": "sheet",
            #         "worksheet_name": "Research Notes",
            #     },
            #
            #     "delete_func": delete_slide_chunks_sheet,
            #     "delete_args": {
            #         "sheet": "sheet",
            #         "worksheet_name": "Slide Chunks",
            #     },
            #
            #     "estimated_time": "10-20 minutes",
            #     "description": "This function parses the research notes from the `Research Notes` sheet and extracts slide chunks in the `Slide Chunks` sheet.",
            # },
        ]
    },

    {
        "section_name": "Section 2: Learning Objectives Slide Generation",
        "steps": [

            {
                "name": "Generate Learning Objectives",
                "func": run_learning_objectives_agent,
                "depends_on": ['Slide Chunks Checklist Review and Revise'],
                "args": {
                    "sheet": "sheet",
                    "worksheet_name": "Slide Chunks",
                    "course_name": "course_name",
                    "target_audience": "target_audience",
                    "llm":"gemini_2_flash"
                },

                "delete_func": delete_learning_objectives_slide_chunks,
                "delete_args": {
                    "sheet": "sheet",
                    "worksheet_name": "Slide Chunks",
                },

                "estimated_time": "< 1 minute",
                "description": "This function generates slide content based on the reviewed and revised slide chunks.",
            },
        ]
    }
    # {
    #     "section_name": "Section 3: Generate Final Slides by doing Checklist based Review and Revise",
    #     "steps": [               
    #         {
    #             "name": "Review and Revise Checklist",
    #             "func": run_checklist_review_and_revise,
    #             "depends_on": ['Generate Learning Objectives'],
    #             "args": {
    #                 "sheet": "sheet",
    #                 "worksheet_name": "Slide Chunks",
    #                 "course_name": "course_name",
    #                 "target_audience": "target_audience",
    #                 "llm": "gemini_2_flash"
    #             },

    #             "delete_func": delete_checklist_review_and_revise,
    #             "delete_args": {
    #                 "sheet": "sheet",
    #                 "worksheet_name": "Slide Chunks",
    #             },

    #             "estimated_time": "40-60 minutes",
    #             "description": "This function reviews and revises the slide chunks based on the review checklist to ensure quality and coherence.",
    #         },
            
            # {
            #     "name": "AI Detection based Review and Revise Agents",
            #     "func": run_ai_detection_review_revise,
            #     "depends_on": ['Review and Revise Checklist'],
            #     "args": {
            #         "sheet": "sheet",
            #         "worksheet_name": "Slide Chunks",
            #         "course_name": "course_name",
            #         "target_audience": "target_audience",
            #         "llm": "gemini_2_flash"
            #     },
            
            #     "estimated_time": "30-50 minutes",
            #     "description": "This function reviews and revises the slide chunks based on AI detection to ensure that the content looks Human-generated and not AI-generated.",
            # },
            
            # {
            #     "name": "Winston.ai Plagiarism Check",
            #     "func": run_plagiarism_detection,
            #     "depends_on": ['AI Detection based Review and Revise Agents'],
            #     "args": {
            #         "sheet": "sheet",
            #         "worksheet_name": "Slide Chunks"
            #     },
            
            #     "estimated_time": "1-2 minutes",
            #     "description": "This function checks for plagiarism in the slide chunks using Winston.ai.",
            # },
            
            # {
            #     "name": "Winston.ai AI Detection Check with Readability Score based Reviser Agent",
            #     "func": run_ai_detection_with_readability,
            #     "depends_on": ['Winston.ai Plagiarism Check'],
            #     "args": {
            #         "sheet": "sheet",
            #         "worksheet_name": "Slide Chunks",
            #         "course_name": "course_name",
            #         "target_audience": "target_audience"
            #     },
            
            #     "estimated_time": "5-10 minutes",
            #     "description": "This function checks for AI detection and readability in the slide chunks using Winston.ai.",
            # },

]

agent_ui(step_name = "Slide Chunks", pipeline_sections = pipeline_sections)
