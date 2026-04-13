from agent_ui_template import agent_ui

from agents.graphics_definition_v2.image_graphics_agent.segment_slide import (
    run_segment_slide_from_slide_chunk_for_all_rows,
    delete_segment_slide,
    ensure_visual_assignment_strategy_column,
)
from agents.graphics_definition_v2.image_graphics_agent.storyboard_agent import (
    run_storyboard_agent_for_all_rows,
    delete_storyboard_planning,
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
#from agents.graphics_definition_v2.image_graphics_agent.finalize_graphics_definition import (
#    run_finalize_graphics_definition_for_all_rows,
#    delete_graphics_definition,
#)
from agents.graphics_definition_v2.video_graphics_agent.video_search_query_generation import (
    run_generate_video_search_query_for_all_rows,
    delete_video_search_queries,
)
from agents.graphics_definition_v2.video_graphics_agent.youtube_video_search_from_queries import (
    run_youtube_video_search_for_all_rows,
    delete_video_pool,
)
from agents.graphics_definition_v2.image_graphics_agent.generate_pools import (
    run_generate_image_and_video_pools,
    delete_all_pool_results,
)
from agents.graphics_definition_v2.aggregation_agent.aggregation_agent import (
    run_aggregation_agent_for_all_rows,
    delete_final_graphics_definition,
)
from agents.graphics_definition_v2.review_agent.review_and_revise import (
    run_review_and_revise_graphics_definition_v2_for_all_rows,
    delete_review_and_revise_graphics_definition_v2,
)
from agents.graphics_definition_v2.video_graphics_agent.youtube_video_search_in_other_channels import (
    run_youtube_video_search_other_channels_for_all_rows,
    delete_video_pool_other_channels,
)
from agents.graphics_definition_v2.layout_agent.layout_agent import (
    run_layout_agent_for_all_rows,
    delete_layout_columns,
)
from agents.graphics_definition_v2.image_graphics_agent.generate_candidates import (
    run_generate_image_and_video_candidates,
    delete_all_candidate_results,
)
# from agents.graphics_definition_v2.review_agent.human_feedback_based_review_and_revise import (
#     run_human_feedback_review_revise_for_all_rows,
#     delete_human_feedback_based_review_and_revise,
# )
from agents.graphics_definition_v2.review_agent.visual_columns_for_human_feedback import (
    run_populate_human_feedback_visual_columns,
    delete_human_feedback_visual_columns,
)
from agents.graphics_definition_v2.download_assets.download_assets_for_sheet import (
    run_download_assets_for_sheet,
)

# Shown at top of page (before Section 1) so users set Visual Assignment Strategy before running.
TOP_INSTRUCTIONS = (
                    "**Before running:** Set **Visual Assignment Strategy** in the Slide Chunks sheet for each row (dropdown per row). For each row, you can select one of the following options:\n\n"
                    "- **Flexible, let the agent decide**: Agent picks how many visuals get assigned for the slide.\n"
                    "- **1 Visual per Sentence**: One visual gets assigned per sentence.\n"
                    "- **1 Visual for the whole Slide**: One visual gets assigned for the entire slide.\n\n"
)



pipeline_sections = [
    {
        "section_name": "Section 1: Segment Slide into Voiceover Segments",
        "steps": [
            {
                "name": "Segment Slide into Voiceover Segments",
                "func": run_segment_slide_from_slide_chunk_for_all_rows,
                "pre_exec_func": ensure_visual_assignment_strategy_column,
                "pre_exec_args": {
                    "sheet": "sheet",
                    "worksheet_name": "Slide Chunks"
                },
                "depends_on": [],
                "args": {
                    "sheet": "sheet",
                    "llm": "gemini_2_5_flash_lite",
                    "max_workers": 50,
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
        "section_name": "Section 2: Generate Storyboard Planning for each Slide",
        "steps": [
            {
                "name": "Generate Storyboard for each Slide",
                "func": run_storyboard_agent_for_all_rows,
                "depends_on": ["Segment Slide into Voiceover Segments"],
                "args": {
                    "sheet": "sheet",
                    "llm": "gemini_3_flash_thinking",
                    "max_workers": 50,
                },
                "estimated_time": "5-10 minutes",
                "description": "This function generates a storyboard for each slide and saves them to the storyboard_planning column",
                "delete_func": delete_storyboard_planning,
                "delete_args": {
                    "sheet": "sheet"
                }
            },
        ]
    },
    {
        "section_name": "Section 2: Generate Search Queries for Image and Video Retrieval",
        "steps": [
            {
                "name": "Generate Search Queries for Image and Video Retrieval",
                "func": run_generate_search_query_for_all_rows,
                "depends_on": ["Generate Storyboard for each Slide"],
                "args": {
                    "sheet": "sheet",
                    "llm": "gemini_2_5_flash_lite",
                    "max_workers": 50,
                },
                "estimated_time": "5-10 minutes",
                "description": "This function generates search queries for image and video retrieval and saves them to the search_queries column",
                "delete_func": delete_search_queries,
                "delete_args": {
                    "sheet": "sheet"
                }
            },
        ]
    },
    {
        "section_name": "Section 3: Generate Image and Video Candidates",
        "steps": [
            {
                "name": "Generate Image and Video Candidates",
                "func": run_generate_image_and_video_candidates,
                "depends_on": ["Generate Search Queries for Image and Video Retrieval"],
                "args": {
                    "sheet": "sheet",
                    "max_workers": 50,
                    "use_only_drive_and_hvac": "use_only_drive_and_hvac",
                },
                "estimated_time": "10-20 minutes",
                "description": "This step runs 4 parallel searches: Drive Search, Web Search, Video Search (HVAC channels), and Video Search (Other channels). Each writes to its respective column (drive_results, web_results, video_pool, video_pool_other_channels).",
                "delete_func": delete_all_candidate_results,
                "delete_args": {
                    "sheet": "sheet"
                }
            },
        ]
    },
    {
        "section_name": "Section 4: Generate Image and Video Pools",
        "steps": [
            {
                "name": "Generate Image and Video Pools",
                "func": run_generate_image_and_video_pools,
                "depends_on": ["Generate Image and Video Candidates"],
                "args": {
                    "sheet": "sheet",
                    "image_pool_llm": "gemini_2_5_flash_lite",
                    "video_pool_llm": "gemini_3_flash_thinking",
                    "max_workers": 50,
                },
                "is_llm_step": True, 
                "estimated_time": "20-40 minutes",
                "description": "This step runs Image Pool and Video Pool generation in parallel. Image Pool selects relevant images from drive_results and web_results. Video Pool selects relevant videos from video_pool and video_pool_other_channels. Both write to their respective columns (image_pool, video_pool_filtered).",
                "delete_func": delete_all_pool_results,
                "delete_args": {
                    "sheet": "sheet"
                }
            },
        ]
    },
    {
        "section_name": "Section 5: Aggregation Agent",
        "steps": [
            {
                "name": "Aggregation Agent",
                "func": run_aggregation_agent_for_all_rows,
                "depends_on": ["Generate Image and Video Pools"],
                "args": {
                    "sheet": "sheet",
                    "llm": "gemini_3_flash_thinking",
                    "max_workers": 50,
                },
                "estimated_time": "30-60 minutes",
                "description": "This step selects the best visuals from the Image and Video pool for the voiceover segments to create the final graphics definitions for the slide content",
                "delete_func": delete_final_graphics_definition,
                "delete_args": {
                    "sheet": "sheet"
                }
            },
        ]
    },
    {
        "section_name": "Section 6: Review and Revise Graphics Definitions",
        "steps": [
            {
                "name": "Review and Revise Graphics Definitions",
                "func": run_review_and_revise_graphics_definition_v2_for_all_rows,
                "depends_on": ["Aggregation Agent"],
                "args": {
                    "sheet": "sheet",
                    "llm": "gemini_3_flash_thinking",
                    "max_workers": 50,
                    "use_only_drive_and_hvac": "use_only_drive_and_hvac",
                },
                "estimated_time": "30-90 minutes",
                "description": "This step reviews assigned visuals against alignment, specificity, and redundancy criteria, revises using existing pools, and regenerates only when necessary.",
                "delete_func": delete_review_and_revise_graphics_definition_v2,
                "delete_args": {
                    "sheet": "sheet"
                }
            },
        ]
    },
    # {
    #     "section_name": "Section 7: Human Feedback Revisions",
    #     "steps": [
    #         {
    #             "name": "Review and Revise Graphics Definitions based on Human Feedback",
    #             "func": run_human_feedback_review_revise_for_all_rows,
    #             "depends_on": ["Review and Revise Graphics Definitions"],
    #             "args": {
    #                 "sheet": "sheet",
    #                 "llm": "gemini_3_flash_thinking",
    #                 "max_workers": 50,
    #                 "use_only_drive_and_hvac": "use_only_drive_and_hvac",
    #             },
    #             "estimated_time": "20-40 minutes",
    #             "description": "This step revises the graphics definitions based on the human feedback provided in the human_feedback column.",
    #             "delete_func": delete_human_feedback_based_review_and_revise,
    #             "delete_args": {
    #                 "sheet": "sheet"
    #             }
    #         },
    #         # },
    #         # {
    #         #     "name": "Populate Human Feedback Original/Final Visual Columns",
    #         #     "func": run_populate_human_feedback_visual_columns,
    #         #     "depends_on": [],
    #         #     "args": {
    #         #         "sheet": "sheet",
    #         #     },
    #         #     "estimated_time": "1-5 minutes",
    #         #     "description": "Adds dynamic columns after human_feedback_revision_tracking: Original Visual 1, Revised Visual 1, (blank gap), Original Visual 2, … per VO with feedback; IMAGE() for image URLs, HYPERLINK for YouTube; cell notes with When VO and Human Feedback.",
    #         #     "delete_func": delete_human_feedback_visual_columns,
    #         #     "delete_args": {
    #         #         "sheet": "sheet"
    #         #     }
    #         # },
    #     ]
    # },
    # {
    #     "section_name": "Section 8: Layout Agent",
    #     "steps": [
    #         {
    #             "name": "Run Layout Agent",
    #             "func": run_layout_agent_for_all_rows,
    #             "depends_on": ["Review and Revise Graphics Definitions based on Human Feedback"],
    #             "args": {
    #                 "sheet": "sheet",
    #                 "llm": "gemini_3_flash_thinking",
    #                 "max_workers": 50,
    #             },
    #             "estimated_time": "15-30 minutes",
    #             "description": "This function generates presentation-ready layout instructions for each slide based on the final graphics definition. It determines how assets are arranged on the canvas, how they transition, and how visual continuity is maintained.",
    #             "delete_func": delete_layout_columns,
    #             "delete_args": {
    #                 "sheet": "sheet"
    #             }
    #         },
    #     ]
    # },
    # {
    #     "section_name": "Section 8: Download Assets to Drive",
    #     "steps": [
    #         {
    #             "name": "Download Assets to Drive",
    #             "func": run_download_assets_for_sheet,
    #             "depends_on": [],
    #             "args": {
    #                 "sheet": "sheet",
    #             },
    #             "estimated_time": "2-10 minutes",
    #             "description": "Downloads all assets from final_graphics_definition (Drive images, web images, YouTube clips with start/end) into a new folder under 'Downloadable Asset Folder' on Drive. Folder name: 'CourseName, DD/MM/YYYY, HH:MM IST'. Files are named 'Slide X, S{n}V{m}.ext' (e.g. Slide 1, S1V1.jpg, Slide 2, S2V1.mp4).",
    #         },
    #     ]
    # },
]

llm_pricing = {
    "currency": "$",
    "default": {
        "input_per_million": 0.50,
        "output_per_million": 3.00,
    },
    "models": {
        "gemini_2_5_flash_lite": {
            "input_per_million": 0.10,  
            "output_per_million": 0.40, 
        },
        "gemini_3_flash_thinking": {
            "input_per_million": 0.50,
            "output_per_million": 3.00,
        },
    },
}

TOP_TOGGLES = [
    {
        "key": "use_only_drive_and_hvac",
        "label": "Use only Drive images and videos from 'HVAC School' and 'Love2HVAC with Ty' youtube channel",
        "default": True,
    },
]

agent_ui(step_name="Graphics Definition V2", pipeline_sections=pipeline_sections, llm_pricing=llm_pricing, top_instructions=TOP_INSTRUCTIONS, top_toggles=TOP_TOGGLES)
