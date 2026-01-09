from agent_ui_template import agent_ui

from agents.graphics_definition_v2.image_graphics_agent.segment_slide import (
    run_segment_slide_from_slide_chunk_for_all_rows,
    delete_segment_slide,
)
from agents.graphics_definition_v2.image_graphics_agent.generate_search_query import (
    run_generate_search_query_for_all_rows,
    delete_search_queries,
)
from agents.graphics_definition_v2.image_graphics_agent.drive_search import (
    run_drive_search_for_all_rows,
    delete_drive_results,
)
from agents.graphics_definition_v2.image_graphics_agent.web_search import (
    run_web_search_for_all_rows,
    delete_web_results,
)
from agents.graphics_definition_v2.image_graphics_agent.finalize_graphics_definition import (
    run_finalize_graphics_definition_for_all_rows,
    delete_graphics_definition,
)
from agents.graphics_definition_v2.image_graphics_agent.populate_sheet_with_selected_images import (
    run_populate_sheet_with_selected_images_for_all_rows,
    delete_populated_images,
)
from agents.graphics_definition_v2.video_graphics_agent.video_search_query_generation import (
    run_generate_video_search_query_for_all_rows,
    delete_video_search_queries,
)
from agents.graphics_definition_v2.video_graphics_agent.youtube_video_search_from_queries import (
    run_youtube_video_search_for_all_rows,
    delete_video_pool,
)
from agents.graphics_definition_v2.image_graphics_agent.image_selection_from_all_images import (
    run_image_selection_from_all_images_for_all_rows,
    delete_image_pool,
)
from agents.graphics_definition_v2.aggregation_agent.aggregation_agent import (
    run_aggregation_agent_for_all_rows,
    delete_final_graphics_definition,
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
        "section_name": "Section 2: Generate Search Queries for Image Retrieval",
        "steps": [
            {
                "name": "Generate Search Queries for Image Retrieval",
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
                "depends_on": ["Generate Search Queries for Image Retrieval"],
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
    # {
    #     "section_name": "Section 4: Generate Graphics Definitions with Images",
    #     "steps": [
    #         {
    #             "name": "Generate Graphics Definitions with Images",
    #             "func": run_finalize_graphics_definition_for_all_rows,
    #             "depends_on": ["Execute Web Search"],
    #             "args": {
    #                 "sheet": "sheet",
    #                 "llm": "gemini_3_flash_thinking"
    #             },
    #             "estimated_time": "20-40 minutes",
    #             "description": "This step uses vision model to select best images and create fianl graphics definitions for each segment, then combines them into a final definition",
    #             "delete_func": delete_graphics_definition,
    #             "delete_args": {
    #                 "sheet": "sheet"
    #             }
    #         },
    #     ]
    # },
    # {
    #     "section_name": "Section 5: Populate Images in Sheet",
    #     "steps": [
    #         {
    #             "name": "Populate Sheet with Selected Images",
    #             "func": run_populate_sheet_with_selected_images_for_all_rows,
    #             "depends_on": ["Generate Graphics Definitions with Images"],
    #             "args": {
    #                 "sheet": "sheet"
    #             },
    #             "estimated_time": "5-15 minutes",
    #             "description": "This step populates the selected images from graphics definitions into the sheet",
    #             "delete_func": delete_populated_images,
    #             "delete_args": {
    #                 "sheet": "sheet"
    #             }
    #         },
    #     ]
    # },
    {
        "section_name": "Section 6: Generate Search Queries for Video Retrieval",
        "steps": [
            {
                "name": "Generate Search Queries for Video Retrieval",
                "func": run_generate_video_search_query_for_all_rows,
                "depends_on": ["Execute Web Search"],
                "args": {
                    "sheet": "sheet",
                    "llm": "gemini_3_flash_thinking"
                },
                "estimated_time": "5-10 minutes",
                "description": "This function generates 2-4 video search queries for each voiceover segment and saves them to the video_search_query column",
                "delete_func": delete_video_search_queries,
                "delete_args": {
                    "sheet": "sheet"
                }
            },
        ]
    },
    {
        "section_name": "Section 7: Generate Asset Pool for the Aggregation Agent",
        "steps": [
            {
                "name": "Generate Video Pool",
                "func": run_youtube_video_search_for_all_rows,
                "depends_on": ["Generate Search Queries for Video Retrieval"],
                "args": {
                    "sheet": "sheet"
                },
                "estimated_time": "10-20 minutes",
                "description": "This function executes YouTube video search in vectorstore for all queries in each segment and saves results to video_pool column",
                "delete_func": delete_video_pool,
                "delete_args": {
                    "sheet": "sheet"
                }
            },
            {
                "name": "Generate Image Pool",
                "func": run_image_selection_from_all_images_for_all_rows,
                "depends_on": ["Generate Video Pool"],
                "args": {
                    "sheet": "sheet",
                    "llm": "gemini_3_flash_thinking"
                },
                "estimated_time": "20-40 minutes",
                "description": "This function selects all relevant images from drive_results and web_results for each segment and saves them to image_pool column",
                "delete_func": delete_image_pool,
                "delete_args": {
                    "sheet": "sheet"
                }
            },
        ]
    },
    {
        "section_name": "Section 8: Final Aggregation Agent",
        "steps": [
            {
                "name": "Run Aggregation Agent",
                "func": run_aggregation_agent_for_all_rows,
                "depends_on": ["Generate Image Pool"],
                "args": {
                    "sheet": "sheet",
                    "llm": "gemini_3_flash_thinking"
                },
                "estimated_time": "30-60 minutes",
                "description": "This function combines images and videos from image_pool and video_pool to create final graphics definitions that can use images, video segments, or still frames from videos",
                "delete_func": delete_final_graphics_definition,
                "delete_args": {
                    "sheet": "sheet"
                }
            },
        ]
    },
]

agent_ui(step_name="Graphics Definition V2", pipeline_sections=pipeline_sections)

