from agent_ui_template import agent_ui

from agents.graphics_definition_v2.segment_slide import (
    run_segment_slide_from_slide_chunk_for_all_rows,
    delete_segment_slide,
)
from agents.graphics_definition_v2.generate_search_query import (
    run_generate_search_query_for_all_rows,
    delete_search_queries,
)
from agents.graphics_definition_v2.drive_search import (
    run_drive_search_for_all_rows,
    delete_drive_results,
)
from agents.graphics_definition_v2.web_search import (
    run_web_search_for_all_rows,
    delete_web_results,
)
from agents.graphics_definition_v2.finalize_graphics_definition import (
    run_finalize_graphics_definition_for_all_rows,
    delete_graphics_definition,
)
from agents.graphics_definition_v2.populate_sheet_with_selected_images import (
    run_populate_sheet_with_selected_images_for_all_rows,
    delete_populated_images,
)

pipeline_sections = [
    {
        "section_name": "Section 1: Segment Slide into Voiceover Segments",
        "steps": [
            {
                "name": "Segment Slide into Voiceover Segments",
                "func": run_segment_slide_from_slide_chunk_for_all_rows,
                "depends_on": [],
                "args": {
                    "sheet": "sheet",
                    "llm": "gemini_3_flash"
                },
                "estimated_time": "2-5 minutes",
                "description": "This function segments each slide chunk into individual voiceover (VO) segments using sentence-based segmentation and saves them to the voiceover_segment column.",
                "delete_func": delete_segment_slide,
                "delete_args": {
                    "sheet": "sheet"
                }
            },
        ]
    },
    {
        "section_name": "Section 2: Generate Search Queries",
        "steps": [
            {
                "name": "Generate Search Queries",
                "func": run_generate_search_query_for_all_rows,
                "depends_on": ["Segment Slide into Voiceover Segments"],
                "args": {
                    "sheet": "sheet",
                    "llm": "gemini_3_flash_thinking"
                },
                "estimated_time": "5-10 minutes",
                "description": "This function generates 3 search queries for each voiceover segment and saves them to the search_queries column",
                "delete_func": delete_search_queries,
                "delete_args": {
                    "sheet": "sheet"
                }
            },
        ]
    },
    {
        "section_name": "Section 3: Image Search in Drive and Web",
        "steps": [
            {
                "name": "Execute Drive Search",
                "func": run_drive_search_for_all_rows,
                "depends_on": ["Generate Search Queries"],
                "args": {
                    "sheet": "sheet"
                },
                "estimated_time": "10-20 minutes",
                "description": "This function executes drive search for all queries in each segment",
                "delete_func": delete_drive_results,
                "delete_args": {
                    "sheet": "sheet"
                }
            },
            {
                "name": "Execute Web Search",
                "func": run_web_search_for_all_rows,
                "depends_on": ["Execute Drive Search"],
                "args": {
                    "sheet": "sheet"
                },
                "estimated_time": "10-20 minutes",
                "description": "This function executes web search for all queries in each segment",
                "delete_func": delete_web_results,
                "delete_args": {
                    "sheet": "sheet"
                }
            },
        ]
    },
    {
        "section_name": "Section 4: Generate Graphics Definitions with Images",
        "steps": [
            {
                "name": "Generate Graphics Definitions with Images",
                "func": run_finalize_graphics_definition_for_all_rows,
                "depends_on": ["Execute Web Search"],
                "args": {
                    "sheet": "sheet",
                    "llm": "gemini_3_flash_thinking"
                },
                "estimated_time": "20-40 minutes",
                "description": "This step uses vision model to select best images and create fianl graphics definitions for each segment, then combines them into a final definition",
                "delete_func": delete_graphics_definition,
                "delete_args": {
                    "sheet": "sheet"
                }
            },
        ]
    },
    {
        "section_name": "Section 5: Populate Images in Sheet",
        "steps": [
            {
                "name": "Populate Sheet with Selected Images",
                "func": run_populate_sheet_with_selected_images_for_all_rows,
                "depends_on": ["Generate Graphics Definitions with Images"],
                "args": {
                    "sheet": "sheet"
                },
                "estimated_time": "5-15 minutes",
                "description": "This step populates the selected images from graphics definitions into the sheet",
                "delete_func": delete_populated_images,
                "delete_args": {
                    "sheet": "sheet"
                }
            },
        ]
    },
]

agent_ui(step_name="Graphics Definition V2", pipeline_sections=pipeline_sections)

