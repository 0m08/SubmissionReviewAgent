import streamlit as st
from agent_ui_template import agent_ui

from agents.graphics_definition_v2.image_graphics_agent.segment_slide import (
    run_segment_slide_from_slide_chunk_for_all_rows,
    delete_segment_slide,
    prepare_slide_chunks_for_layout_plan,
)
from agents.graphics_definition_v2.planning_layout.layout_plan import (
    run_layout_planning_agent_for_all_rows,
    delete_layout_plan_columns,
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
from agents.graphics_definition_v2.image_graphics_agent.reference_pool_agent import (
    run_reference_image_pool_mapping,
    delete_reference_image_pool_mappings,
)
# from agents.graphics_definition_v2.review_agent.human_feedback_based_review_and_revise import (
#     run_human_feedback_review_revise_for_all_rows,
#     delete_human_feedback_based_review_and_revise,
# )
from agents.graphics_definition_v2.review_agent.visual_columns_for_human_feedback import (
    run_populate_human_feedback_visual_columns,
    delete_human_feedback_visual_columns,
)
from agents.graphics_definition_v2.review_agent.review_finalized_visuals import (
    run_review_finalized_visuals_for_all_rows,
    delete_final_visuals_review,
    run_generate_alternative_visuals_from_web_for_all_rows,
    delete_replacement_visuals_from_final_visuals_review,
)
from agents.graphics_definition_v2.review_agent.decide_final_visuals import (
    run_decide_final_visuals_for_all_rows,
    delete_final_visual_decisions,
)
from agents.graphics_definition_v2.slideshow_manifest.slideshow_manifest import (
    run_slideshow_manifest_for_all_rows,
    delete_slideshow_manifest_columns,
    run_apply_edited_urls_to_slideshow_manifest_for_all_rows,
    delete_apply_edited_urls_to_slideshow_manifest,
)
from agents.graphics_definition_v2.image_editing_for_layout.image_edit_planning import (
    run_scene_edit_planning_for_all_rows,
    delete_scene_edit_plan_columns,
)
from agents.graphics_definition_v2.image_editing_for_layout.image_editing_based_on_edit_planning import (
    run_image_editing_execution_for_all_rows,
    delete_image_editing_execution_columns,
)
from agents.graphics_definition_v2.image_editing_for_layout.hero_animation_decision import (
    run_hero_animation_decision_for_all_rows,
    delete_hero_animation_decision_columns,
)
from agents.graphics_definition_v2.image_editing_for_layout.image_edit_results_sheet import (
    run_populate_image_edit_results_sheet,
    delete_image_edit_results_sheet_data,
)
from agents.graphics_definition_v2.download_assets.download_assets_for_sheet import (
    run_download_assets_for_sheet,
)
from agents.graphics_definition_v2.external_references.external_reference_extraction import (
    run_external_reference_extraction,
    delete_external_reference_extraction_log,
)
from agents.graphics_definition_v2.external_references.run_indexing_step import (
    run_external_reference_indexing,
    delete_external_reference_index_log,
)

# Shown at top of page (before Section 1) so users set Visual Assignment Strategy before running.
TOP_INSTRUCTIONS = (
                    "**Before running:** Set **Visual Assignment Strategy** in the Slide Chunks sheet for each row (dropdown per row). For each row, you can select one of the following options:\n\n"
                    "- **Flexible, let the agent decide**: Agent picks how many visuals get assigned for the slide.\n"
                    "- **1 Visual per Sentence**: One visual gets assigned per sentence.\n"
                    "- **1 Visual for the whole Slide**: One visual gets assigned for the entire slide.\n\n"
                    "Also choose **Asset libraries for this run** below (Drive Images, HVAC YouTube, Google Drive Videos, External References, and Web Images and Videos)."
)



pipeline_sections = [
    {
        "section_name": "External Reference Media Extraction",
        "steps": [
            {
                "name": "Extract External Reference Images",
                "func": run_external_reference_extraction,
                "depends_on": [],
                "args": {
                    "sheet": "sheet",
                },
                "estimated_time": "5-20 minutes",
                "description": "Extracts images from PDF / Google Docs / Google Slides / PPT via LlamaParse and uploads them under the shared External Reference Assets Drive folder.",
                "delete_func": delete_external_reference_extraction_log,
                "delete_args": {
                    "sheet": "sheet",
                },
                "hide_if_external_references_disabled": True,
            },
            {
                "name": "Index External Reference Assets",
                "func": run_external_reference_indexing,
                "depends_on": ["Extract External Reference Images"],
                "args": {
                    "sheet": "sheet",
                },
                "estimated_time": "5-30 minutes",
                "description": "Indexes the external reference images and videos and store them into Supabase.",
                "delete_func": delete_external_reference_index_log,
                "delete_args": {
                    "sheet": "sheet",
                },
                "hide_if_external_references_disabled": True,
            },
        ],
    },
    {
        "section_name": "Section 1: Layout Planning",
        "steps": [
            {
                "name": "Generate Layout Plan for each Slide",
                "func": run_layout_planning_agent_for_all_rows,
                "pre_exec_func": prepare_slide_chunks_for_layout_plan,
                "pre_exec_args": {
                    "sheet": "sheet",
                    "worksheet_name": "Slide Chunks"
                },
                "depends_on": ["Index External Reference Assets"],
                "args": {
                    "sheet": "sheet",
                    "llm": "gemini_3_flash_thinking",
                    "max_workers": 50,
                },
                "estimated_time": "5-10 minutes",
                "description": "This function identifies scene-wise layout strategy for each slide row and saves the output to the layout_plan column.",
                "delete_func": delete_layout_plan_columns,
                "delete_args": {
                    "sheet": "sheet"
                }
            },
        ]
    },
    {
        "section_name": "Section 2: Segment Slide into Voiceover Segments",
        "steps": [
            {
                "name": "Segment Slide into Voiceover Segments",
                "func": run_segment_slide_from_slide_chunk_for_all_rows,
                "depends_on": ["Generate Layout Plan for each Slide"],
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
        "section_name": "Section 3: Generate Storyboard Planning for each Slide",
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
        "section_name": "Section 4: Generate Search Queries for Image and Video Retrieval",
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
            {
                "name": "Map Reference Image Pool",
                "func": run_reference_image_pool_mapping,
                "depends_on": ["Generate Search Queries for Image and Video Retrieval"],
                "args": {
                    "sheet": "sheet",
                    "llm": "gemini_3_flash",
                    "max_workers": 50,
                },
                "is_llm_step": True,
                "estimated_time": "5-10 minutes",
                "description": "This sub-agent finds the best matching images from the reference image pool for each segment and stores them in reference_image_map.",
                "delete_func": delete_reference_image_pool_mappings,
                "delete_args": {
                    "sheet": "sheet"
                }
            },
        ]
    },
    {
        "section_name": "Section 5: Generate Image and Video Candidates",
        "steps": [
            {
                "name": "Generate Image and Video Candidates",
                "func": run_generate_image_and_video_candidates,
                "depends_on": ["Map Reference Image Pool"],
                "args": {
                    "sheet": "sheet",
                    "max_workers": 50,
                    "enabled_sources": "graphics_v2_enabled_sources",
                },
                "estimated_time": "10-20 minutes",
                "description": "This step runs enabled candidate searches in parallel via the pool registry.",
                "delete_func": delete_all_candidate_results,
                "delete_args": {
                    "sheet": "sheet"
                }
            },
        ]
    },
    {
        "section_name": "Section 6: Generate Image and Video Pools",
        "steps": [
            {
                "name": "Generate Image and Video Pools",
                "func": run_generate_image_and_video_pools,
                "depends_on": ["Generate Image and Video Candidates"],
                "args": {
                    "sheet": "sheet",
                    "image_pool_llm": "gemini_3_flash_thinking",
                    "video_pool_llm": "gemini_3_flash_thinking",
                    "max_workers": 50,
                },
                "is_llm_step": True, 
                "estimated_time": "20-40 minutes",
                "description": "This step runs Image Pool and Video Pool scoring/filtering in parallel from enabled candidate columns.",
                "delete_func": delete_all_pool_results,
                "delete_args": {
                    "sheet": "sheet"
                }
            },
        ]
    },
    {
        "section_name": "Section 7: Aggregation Agent",
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
        "section_name": "Section 8: Review and Revise Graphics Definitions",
        "steps": [
            {
                "name": "Review and Revise Graphics Definitions",
                "func": run_review_and_revise_graphics_definition_v2_for_all_rows,
                "depends_on": ["Aggregation Agent"],
                "args": {
                    "sheet": "sheet",
                    "llm": "gemini_3_flash_thinking",
                    "max_workers": 50,
                    "use_only_drive_and_hvac": True,
                    "enabled_sources": "graphics_v2_enabled_sources",
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
    {
        "section_name": "Section 9: Review Finalized Visuals and do a Targeted Web Fallback and Selection.",
        "steps": [
            {
                "name": "Review all the Finalized Visuals to decide if Web Search is needed for some visuals",
                "func": run_review_finalized_visuals_for_all_rows,
                "depends_on": ["Review and Revise Graphics Definitions"],
                "hide_if_web_disabled": True,
                "args": {
                    "sheet": "sheet",
                    "llm": "gemini_3_flash_thinking",
                    "max_workers": 50,
                },
                "estimated_time": "10-30 minutes",
                "description": "Reviews all the finalized visuals and decides if Web Search is needed for some visuals",
                "delete_func": delete_final_visuals_review,
                "delete_args": {
                    "sheet": "sheet"
                }
            },
            {
                "name": "Generate alternative visuals from Web",
                "func": run_generate_alternative_visuals_from_web_for_all_rows,
                "depends_on": ["Review all the Finalized Visuals to decide if Web Search is needed for some visuals"],
                "hide_if_web_disabled": True,
                "args": {
                    "sheet": "sheet",
                    "llm": "gemini_3_flash_thinking",
                    "max_workers": 50,
                },
                "estimated_time": "5-20 minutes",
                "description": "Search and select the best visual from the web for the failed visuals",
                "delete_func": delete_replacement_visuals_from_final_visuals_review,
                "delete_args": {
                    "sheet": "sheet"
                }
            },
            {
                "name": "Decide which visual to Use",
                "func": run_decide_final_visuals_for_all_rows,
                "depends_on": ["Generate alternative visuals from Web"],
                "hide_if_web_disabled": True,
                "args": {
                    "sheet": "sheet",
                    "llm": "gemini_3_flash_thinking",
                    "max_workers": 50,
                },
                "estimated_time": "5-25 minutes",
                "description": "Decide which visual to use between the currently assigned visual and the newly retrieved visual from web",
                "delete_func": delete_final_visual_decisions,
                "delete_args": {
                    "sheet": "sheet"
                }
            },
        ]
    },
    {
        "section_name": "Section 10: Slideshow Manifest",
        "steps": [
            {
                "name": "Generate Slideshow Manifest",
                "func": run_slideshow_manifest_for_all_rows,
                "depends_on": ["Decide which visual to Use"],
                "args": {
                    "sheet": "sheet",
                    "llm": "gemini_3_flash_thinking",
                    "max_workers": 50,
                },
                "estimated_time": "5-20 minutes",
                "description": "Builds a machine-readable slideshow_manifest (and evaluation) per slide from final_graphics_definition (When VO / Assigned Asset pairs), with multimodal asset parts for layout decisions. Runs after final visual decisions are merged into the sheet.",
                "delete_func": delete_slideshow_manifest_columns,
                "delete_args": {
                    "sheet": "sheet"
                },
            },
        ]
    },
    {
        "section_name": "Section 11: Scene Edit Planning",
        "steps": [
            {
                "name": "Generate Scene Edit Plan",
                "func": run_scene_edit_planning_for_all_rows,
                "depends_on": ["Generate Slideshow Manifest"],
                "args": {
                    "sheet": "sheet",
                    "llm": "gemini_3_flash_thinking",
                    "max_workers": 50,
                },
                "estimated_time": "10-30 minutes",
                "description": "Parses slideshow_manifest scene-by-scene and generates scene_edit_plan output by running the image edit planning agent once per scene with multimodal slot assets.",
                "delete_func": delete_scene_edit_plan_columns,
                "delete_args": {
                    "sheet": "sheet"
                },
            },
        ]
    },
    {
        "section_name": "Section 12: Scene Edit Execution",
        "steps": [
            {
                "name": "Run Scene Edit Execution",
                "func": run_image_editing_execution_for_all_rows,
                "depends_on": ["Generate Scene Edit Plan"],
                "args": {
                    "sheet": "sheet",
                    "max_workers": 50,
                },
                "estimated_time": "20-120 minutes",
                "description": "Per instructional slot, runs an edit -> review -> regenerate loop.",
                "delete_func": delete_image_editing_execution_columns,
                "delete_args": {
                    "sheet": "sheet"
                },
            },
        ]
    },
    {
        "section_name": "Section 13: Apply edited images to slideshow manifest",
        "steps": [
            {
                "name": "Apply edited asset URLs to slideshow manifest",
                "func": run_apply_edited_urls_to_slideshow_manifest_for_all_rows,
                "depends_on": ["Run Scene Edit Execution"],
                "args": {
                    "sheet": "sheet",
                    "max_workers": 30,
                },
                "estimated_time": "1-5 minutes",
                "description": "Replace the Original Image URLs with the Edited Image URLs in the slideshow_manifest and final_graphics_definition columns",
                "delete_func": delete_apply_edited_urls_to_slideshow_manifest,
                "delete_args": {
                    "sheet": "sheet"
                },
            },
        ]
    },
    {
        "section_name": "Section 14: Hero Overlay Animation Decision",
        "steps": [
            {
                "name": "Decide Hero Overlay Animation",
                "func": run_hero_animation_decision_for_all_rows,
                "depends_on": ["Apply edited asset URLs to slideshow manifest"],
                "args": {
                    "sheet": "sheet",
                    "llm": "gemini_3_flash_thinking",
                    "max_workers": 50,
                },
                "estimated_time": "5-20 minutes",
                "description": "Decides whether the video player should add a short instructional overlay animation on top of the hero images, locates bbox coordinates, and generates icon overlay images",
                "delete_func": delete_hero_animation_decision_columns,
                "delete_args": {
                    "sheet": "sheet"
                },
            },
        ]
    },
    # {
    #     "section_name": "Section 13: Image Edit Results Sheet",
    #     "steps": [
    #         {
    #             "name": "Populate Image Edit Results Sheet",
    #             "func": run_populate_image_edit_results_sheet,
    #             "depends_on": ["Apply edited asset URLs to slideshow manifest"],
    #             "args": {
    #                 "sheet": "sheet",
    #             },
    #             "estimated_time": "2-10 minutes",
    #             "description": "Creates or refreshes the 'Image Edit results' tab: one row per edited image with slide content, matched When VO, =IMAGE previews for original/edited assets (notes with view URLs), and formatted <edits> XML from scene_edit_plan.",
    #             "delete_func": delete_image_edit_results_sheet_data,
    #             "delete_args": {
    #                 "sheet": "sheet"
    #             },
    #         },
    #     ]
    # },
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

agent_ui(step_name="Graphics Definition V2", pipeline_sections=pipeline_sections, llm_pricing=llm_pricing, top_instructions=TOP_INSTRUCTIONS)
