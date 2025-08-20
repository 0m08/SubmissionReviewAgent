from agent_ui_template import agent_ui
import streamlit as st
import pandas as pd
from services.sheets_service import get_sheet_data_and_df
from services.helper_functions import create_final_outline_sheet


def noop(*args, **kwargs):
    """Placeholder delete function for manual steps."""
    return

from agents.course_outline.video_based_outline.video_search_queries import run_construct_video_search_queries, manual_input_review_video_search_queries
from agents.course_outline.video_based_outline.get_hvac_school_videos import run_get_hvac_school_videos
from agents.course_outline.video_based_outline.classify_and_chunk_videos import run_get_transcripts, run_classify_video, run_chunk_videos
from agents.course_outline.video_based_outline.get_relevant_chunks import run_get_relevant_chunks, manual_input_mark_relevant_videos
from agents.course_outline.video_based_outline.video_based_outlines import run_generate_video_based_outline, manual_input_video_outline_consolidation_comments
from agents.course_outline.video_based_outline.consolidate_video_based_outline import run_propose_consolidated_video_outlines

from agents.course_outline.course_outline_checklist.checklist_based_review_and_reviser_agent import (
    run_outline_checklist_review_and_revise,
    delete_outline_checklist_review_and_revise,
)

#from agents.course_outline.client_reference_based_outline.client_reference_based_outlines import run_generate_outline_from_client_reference, manual_input_client_reference_consolidation_comments
#from agents.course_outline.client_reference_based_outline.consolidate_client_reference_based_outline import run_propose_consolidated_reference_outlines

from agents.course_outline.web_research_based_outline.web_search_queries import run_construct_web_search_queries
from agents.course_outline.web_research_based_outline.web_search_screening import run_web_search_screening
from agents.course_outline.web_research_based_outline.get_article_content import run_fetch_article_content
from agents.course_outline.web_research_based_outline.extract_relevant_info import run_get_relevant_info_from_article
from agents.course_outline.web_research_based_outline.get_research_summary import run_get_research_summary, manual_input_review_research_summary
from agents.course_outline.web_research_based_outline.web_research_based_outline import run_generate_web_research_outline

from agents.course_outline.deep_research.deep_research import run_deep_research
from agents.course_outline.deep_research.deep_research_based_outline import run_generate_deep_research_outline

from agents.course_outline.outline_consolidation.outline_consolidation import run_all_outlines_consolidated_and_review, delete_all_outlines_consolidated_and_review
from agents.course_outline.outline_consolidation.review_and_revise_outline import run_review_and_revise_outline, print_course_outline_before_review, delete_review_and_revise_outline

from agents.course_outline.enhance_outline.create_input_sheets import pre_topic_deep_research, run_create_topic_outline_sheet
from agents.course_outline.enhance_outline.topic_deep_research import run_topic_deep_research
from agents.course_outline.enhance_outline.generate_learning_objectives import run_generate_learning_objectives
from agents.course_outline.enhance_outline.categorize_learning_objectives import run_categorize_learning_objectives
from agents.course_outline.enhance_outline.label_learning_objectives import run_label_learning_objectives
from agents.course_outline.enhance_outline.review_revise_topic_outline import show_outline_diff, run_review_and_revise_topic_outline
from agents.course_outline.video_based_outline.video_search_queries import delete_video_search_queries
from agents.course_outline.video_based_outline.get_hvac_school_videos import delete_videos_research_sheet
from agents.course_outline.video_based_outline.classify_and_chunk_videos import (
    delete_video_transcripts,
    delete_video_relevance,
    delete_video_chunks,
)
from agents.course_outline.video_based_outline.get_relevant_chunks import (
    delete_relevant_chunks,
    delete_mark_relevant_videos,
)
from agents.course_outline.video_based_outline.video_based_outlines import (
    delete_video_based_outlines,
    clear_video_outline_comments,
)
from agents.course_outline.video_based_outline.consolidate_video_based_outline import delete_video_outline_consolidation
from agents.course_outline.web_research_based_outline.web_search_queries import delete_web_search_queries
from agents.course_outline.web_research_based_outline.web_search_screening import delete_preliminary_research_sheet
from agents.course_outline.web_research_based_outline.get_article_content import delete_article_content
from agents.course_outline.web_research_based_outline.extract_relevant_info import delete_relevant_info
from agents.course_outline.web_research_based_outline.get_research_summary import (
    delete_research_summaries,
    clear_manual_extract,
)
from agents.course_outline.web_research_based_outline.web_research_based_outline import delete_web_research_outline
from agents.course_outline.deep_research.deep_research import delete_deep_research_sheet
from agents.course_outline.deep_research.deep_research_based_outline import delete_deep_research_outline
from agents.course_outline.enhance_outline.create_input_sheets import delete_create_topic_outline
from agents.course_outline.enhance_outline.topic_deep_research import delete_topic_deep_research
from agents.course_outline.enhance_outline.generate_learning_objectives import delete_learning_objectives
from agents.course_outline.enhance_outline.categorize_learning_objectives import delete_lo_categorization
from agents.course_outline.enhance_outline.label_learning_objectives import delete_label_learning_objectives
from agents.course_outline.enhance_outline.review_revise_topic_outline import delete_review_and_revise_topic_outline
from agents.course_outline.enhance_outline.map_original_outline_to_revised_outline import delete_topic_outline_mapping
from services.helper_functions import delete_final_outline
from agents.research_notes.load_references import delete_all_references
from agents.research_notes.retriever_agent import delete_retriever_context
from agents.course_outline.enhance_outline.map_original_outline_to_revised_outline import map_original_outline_to_revised_outline_for_all_topics
from agents.course_outline.video_search_tool.video_retriever_agent import run_video_search_for_los

from agents.research_notes.load_references import load_references
from agents.research_notes.retriever_agent import run_retriever_agent_for_all_rows

llm_model = st.session_state.get("llm_model", "gemini_2_flash") or "gemini_2_flash" # Or is set incase llm_model is None

# Read value from Course Info sheet
topic_deep_research_enabled = False 
outline_finalized = False

if "sheet" in st.session_state:
    _, course_info_df = get_sheet_data_and_df(st.session_state["sheet"], "Course info")
    flag_raw = course_info_df.loc[0, "Outline Topic Deep Research"]
    topic_deep_research_enabled = (
        str(flag_raw).strip().lower() != "false"
        if pd.notna(flag_raw)
        else True
    )

    # Check the status of the oultine
    if "Outline Stage" in course_info_df.columns:
        status = course_info_df.loc[0, "Outline Stage"]
        outline_finalized = isinstance(status, str) and status.strip().lower() == "final"

# --- 1) Define pipeline as sections, each with its own steps ---
pipeline_sections = [
    {
        "section_name": "Section 1: Videos Research",
        "steps": [
            {
                "name": "Video Search Query Generator",
                "func": run_construct_video_search_queries,
                "depends_on": [],
                "args": {
                    "sheet": "sheet",
                    "worksheet_name": "Base Outline",
                    "course_name": "course_name",
                    "target_audience": "target_audience",
                    "llm": llm_model,
                },
                "estimated_time": "~ 1 minute",
                "description": "Generates search queries to identify relevant HVAC school videos on YouTube.",
                "delete_func": delete_video_search_queries,
                "delete_args": {
                    "sheet": "sheet",
                    "worksheet_name": "Base Outline",
                }
            },
            {
                "name": "Manual Review - Video Search Queries",
                "func": manual_input_review_video_search_queries,
                "depends_on": ["Video Search Query Generator"],
                "args": {
                    "sheet": "sheet",
                    "worksheet_name": "Base Outline",
                },
                "instructions": [
                    "**Instructions:**",
                    "- Open the `Base Outline` tab.",
                    "- Check the `video_search_queries` column.",
                    "- Review each query to see if it clearly matches the topic and would make sense to your intended audience",
                    "- You can:",
                    "1. **Keep** the query if it looks good as-is.",
                    "2. **Edit** to make it clearer or more specific.",
                    "3. **Add** a new query if one is missing.",
                    "4.**Delete** a query if it doesn't fit the topic.",
                    "- If the queries are strong and accurate, you can simply click on **Confirm Manual Review - Video Search Queries** button below. This means the step is done.",
                    "---",
                    "**Best Practices:**",
                    "- Use keywords your audience would actually type into a search bar",
                        "- Keep it short and clear. No long sentences.",
                    "- Make sure the query matches the topic's goal.",
                    "---",
                    "**Example:**",
                    """- Change **"Basic HVAC troubleshooting"** to **"Troubleshooting residential HVAC systems"** for better clarity and focus."""
                ],
                "is_manual_step": True,
                "estimated_time": "Manual step",
                "description": "Review and adjust the AI-generated video search queries as needed.",
                "video_link": "https://drive.google.com/file/d/1A4AILmEAXkKCKUQY_GgZoXmCPjtnnOfz/view?usp=sharing",
                "delete_func": noop,
            },
            {
                "name": "Retrieve HVAC School Videos",
                "func": run_get_hvac_school_videos,
                "depends_on": ["Manual Review - Video Search Queries"],
                "args": {
                    "sheet": "sheet",
                    "worksheet_name": "Videos Research",
                },
                "estimated_time": "~ 1 minute",
                "description": "Fetches relevant HVAC school YouTube videos based on the approved search queries and lists them in the `Videos Research` sheet.",
                "delete_func": delete_videos_research_sheet,
                "delete_args": {
                    "sheet": "sheet",
                    "worksheet_name": "Videos Research",
                }
            },
            {
                "name": "Retrieve Video Transcripts",
                "func": run_get_transcripts,
                "depends_on": ["Retrieve HVAC School Videos"],
                "args": {
                    "sheet": "sheet",
                    "worksheet": "Videos Research",
                },
                "estimated_time": "~ 10 - 20 minutes",
                "description": "Fetches the complete transcripts for the listed videos and stores them in the `Videos Research` sheet.",
                "delete_func": delete_video_transcripts,
                "delete_args": {
                    "sheet": "sheet",
                    "worksheet_name": "Videos Research",
                }
            },
            {
                "name": "Check Video Relevance",
                "func": run_classify_video,
                "depends_on": ["Retrieve Video Transcripts"],
                "args": {
                    "sheet": "sheet",
                    "worksheet_name": "Videos Research",
                    "course_name": "course_name",
                    "target_audience": "target_audience",
                    "llm": llm_model,
                },
                "estimated_time": "~ 10 - 20 minutes",
                "description": "Evaluates each video transcript to classify videos as either relevant or irrelevant based on the course context and audience.",
                "delete_func": delete_video_relevance,
                "delete_args": {
                    "sheet": "sheet",
                    "worksheet_name": "Videos Research",
                }
            },
            {
                "name": "Chunk Videos",
                "func": run_chunk_videos,
                "depends_on": ["Check Video Relevance"],
                "args": {
                    "sheet": "sheet",
                    "videos_research_worksheet_name": "Videos Research",
                    "video_chunks_worksheet_name": "Video Chunks",
                    "llm": llm_model,
                },
                "estimated_time": "~ 10 - 20 minutes",
                "description": "Segments transcripts of all videos marked as relevant into meaningful chunks and stores them in the `Video Chunks` sheet.",
                "delete_func": delete_video_chunks,
                "delete_args": {
                    "sheet": "sheet",
                    "videos_research_worksheet": "Videos Research",
                    "video_chunks_worksheet": "Video Chunks",
                }
            },
            {
                "name": "Identify Relevant Chunks",
                "func": run_get_relevant_chunks,
                "depends_on": ["Chunk Videos"],
                "args": {
                    "sheet": "sheet",
                    "worksheet_name": "Videos Research",
                    "course_name": "course_name",
                    "target_audience": "target_audience",
                    "llm": llm_model,
                },
                "estimated_time": "~ 10 - 20 minutes",
                "description": "Analyzes segmented chunks from the previous step and identifies the most relevant ones based on course content and audience criteria.",
                "delete_func": delete_relevant_chunks,
                "delete_args": {
                    "sheet": "sheet",
                    "worksheet_name": "Videos Research",
                }
            },
            {
                "name": "Manual Review - Mark relevant videos",
                "func": manual_input_mark_relevant_videos,
                "depends_on": ["Identify Relevant Chunks"],
                "args": {
                    "sheet": "sheet",
                    "worksheet_name": "Videos Research",
                    "skip_manual_step": "skip_manual_step"
                },
                "instructions": [
                    "**Instructions:**",
                    "- Go to the `Videos Research` sheet.",
                    "- Use the AI-generated filter view (shows only videos AI marked as relevant).",
                    "- In the `Manual Review` column, enter `Yes` if the video is indeed relevant, or `No` if irrelevant.",
                    "- In the `Used for` column, enter `Just Content` if only the transcript is helpful, `As a video` if the full video should be used, or leave blank if you entered `No` in the previous column.",
                    "---",
                    "**Best Practices:**",
                    "1. Based on the columns `chapter_summaries`, `video_relevance` and `proposed_chapters_to_include`, decide whether video is actually relevant.",
                    "2. For all relevant videos, decide whether to use just content or the clips from videos can be inserted directly.",
                    "---",
                    "**Example:**",
                    """- Video about **HVAC thermostat calibration** might be marked `Yes` and used `Just Content` if the transcript is valuable but video format isn't needed.""",
                    "- A random **product review** should be marked **No**—it's not helpful.",
                ],
                "is_manual_step": True,
                "estimated_time": "Manual step",
                "description": "Verify and confirm the relevance classification suggested by AI for videos.",
                "video_link": "https://drive.google.com/file/d/1JckbG8T34Baa2vZc5Gc6RXFZ-3sgRdff/view?usp=drive_link",
                "delete_func": delete_mark_relevant_videos,
                "delete_args": {
                    "sheet": "sheet",
                    "worksheet_name": "Videos Research",
                }
            },
            {
                "name": "Generate Video Based Outlines",
                "func": run_generate_video_based_outline,
                "depends_on": ["Manual Review - Mark relevant videos"],
                "args": {
                    "sheet": "sheet",
                    "worksheet_name": "Videos Research",
                    "course_name": "course_name",
                    "target_audience": "target_audience",
                    "llm": llm_model,
                },
                "hide_if_final_outline": True,
                "estimated_time": "~ 5 - 10 minutes",
                "description": "Produces outlines from transcripts of videos manually confirmed as relevant.",
                "delete_func": delete_video_based_outlines,
                "delete_args": {
                    "sheet": "sheet",
                    "worksheet_name": "Videos Research",
                }
            },
            {
                "name": "Manual Review - Video Outline Consolidation Comments",
                "func": manual_input_video_outline_consolidation_comments,
                "depends_on": ["Generate Video Based Outlines"],
                "args": {
                    "sheet": "sheet",
                    "worksheet_name": "Videos Research",
                    "course_name": "course_name",
                    "target_audience": "target_audience",
                    "skip_manual_step": "skip_manual_step",
                    "llm": "gemini_2_5_flash",
                },
                "instructions": [
                    "- Open the **Videos Research** sheet.",
                    "- Look at the **video-based outlines** generated by AI in the `outline` column. ",
                    "- For each row with an outline, write your notes in the `consolidation_comments` column.",
                    "- Your comments should guide the AI on how to merge all outlines into one clear and coherent version.",
                    "---",
                    "**Best Practices**",
                    "1. Say **what to keep**, **what to remove**, or **how to rephrase** something.",
                    "2. Point out if topics are **repetitive**, **missing**, or **out of order**.",
                    "3. Suggest combining similar points or simplifying where needed.",
                    "4. If something is **confusing or incorrect**, explain why.",
                    "---",
                    ""
                    "**Example:**",
                    "- Remove this topic. It doesn't match the course goal.",
                    "- Combine this point with the one above. It's the same idea.",
                    "- Reword this section to make it simpler and clearer.",
                    "- Add a section on safety tips/ This was missing from all videos.",
                    "---",
                    "NOTE: You must fill out the **consolidation_comments** column for **every row** that contains an outline. Leave it blank only if the row has no outline at all."
                ],
                "is_manual_step": True,
                "hide_if_final_outline": True,
                "estimated_time": "Manual step",
                "description": "Provide detailed feedback to guide consolidation of individual video outlines into a unified course outline.",
                "video_link": "https://drive.google.com/file/d/1vQttQADL78qwf_qQq5RVblt8RVUWfIOW/view?usp=drive_link",
                "delete_func": clear_video_outline_comments,
                "delete_args": {
                    "sheet": "sheet",
                    "worksheet_name": "Videos Research",
                }
            },
            {
                "name": "Consolidate Video Based Outlines",
                "func": run_propose_consolidated_video_outlines,
                "depends_on": ["Manual Review - Video Outline Consolidation Comments"],
                "args": {
                    "sheet": "sheet",
                    "worksheet_name": "Outline Consolidation",
                    "course_name": "course_name",
                    "target_audience": "target_audience",
                    # "llm": llm_model,
                },
                "hide_if_final_outline": True,
                "estimated_time": "~ 2 minutes",
                "description": "Combines multiple individual outlines into four proposed consolidated outline variations in the `Outline Consolidation` sheet.",
                "delete_func": delete_video_outline_consolidation,
                "delete_args": {
                    "sheet": "sheet",
                    "worksheet_name": "Outline Consolidation"
                }
            },
        ],
    },
    # {
    #     "section_name": "Section 2: Client References",
    #     "steps": [
    #         {
    #             "name": "Generate Client Reference Based Outlines",
    #             "func": run_generate_outline_from_client_reference,
    #             "depends_on": ["Consolidate Video Based Outlines"],
    #             "args": {
    #                 "sheet": "sheet",
    #                 "worksheet_name": "Client References",
    #                 "course_name": "course_name",
    #                 "target_audience": "target_audience",
    #                 "llm": llm_model,
    #             },
    #             "hide_if_final_outline": True,
    #             "estimated_time": "~ 5 - 10 minutes",
    #             "description": "Creates outlines based on client-provided references listed in the `Client References` sheet.",
    #         },
    #         {
    #             "name": "Manual Review - Client References Outline Consolidation Comments",
    #             "func": manual_input_client_reference_consolidation_comments,
    #             "depends_on": ["Generate Client Reference Based Outlines"],
    #             "args": {
    #                 "sheet": "sheet",
    #                 "worksheet_name": "Client References",
    #                 "course_name": "course_name",
    #                 "target_audience": "target_audience",
    #                 "skip_manual_step": "skip_manual_step",
    #                 "llm": llm_model,
    #             },
    #             "instructions": [
    #                 "- Open the **Client References** sheet.",
    #                 "- Please review the client references based outlines generated by AI in the `Client References` sheet",
    #                 "- For every row with an outline, write comments in the **consolidation_comments** column.",
    #                 "- Your goal: guide the AI on how to turn all this info into one clear and cohesive outline.",
    #                 "---",
    #                 "**Best Practices**",
    #                 "1. Suggest **what to keep**, **what to cut**, or **how to rewrite** parts of the outline.",
    #                 "2. Flag any **off-topic**, **unclear**, or **repetitive** content.",
    #                 "3. Recommend combining similar points or reorganizing ideas for flow.",
    #                 "4. Call out **missing points** that should be included based on your judgment.",
    #                 "---",
    #                 "**Example:**",
    #                 "- This point doesn't apply to our use case. Remove it.",
    #                 "- Merge this with the previous idea. They're very similar.",
    #                 "- This is too complex. Simplify the wording.",
    #                 "- Missing reference to installation process. Please add.",
    #                 "---",
    #                 "NOTE: Fill in the **consolidation_comments** column for **every row** that has an outline. Leave it blank only if the row does **not** have an outline."
    #             ],
    #             "is_manual_step": True,
    #             "hide_if_final_outline": True,
    #             "estimated_time": "Manual step",
    #             "description": "Provide detailed comments to guide the consolidation of client reference-based outlines.",
    #             "video_link": "https://drive.google.com/file/d/1Xm1ZCp92GiAyBXs_CAkv82JnInfGy02m/view?usp=drive_link",
    #         },
    #         {
    #             "name": "Consolidate Client Reference Based Outline",
    #             "func": run_propose_consolidated_reference_outlines,
    #             "depends_on": ["Manual Review - Client References Outline Consolidation Comments"],
    #             "args": {
    #                 "sheet": "sheet",
    #                 "worksheet_name": "Outline Consolidation",
    #                 "course_name": "course_name",
    #                 "target_audience": "target_audience",
    #                 # "llm": llm_model,
    #             },
    #             "hide_if_final_outline": True,
    #             "estimated_time": "~ 2 minutes",
    #             "description": "Combines individual client reference outlines into four cohesive outline options stored in the `Outline Consolidation` sheet.",
    #         },
    #     ],
    # },
    {
        "section_name": "Section 3: Web Research",
        "steps": [
            {
                "name": "Web Search Queries Generator",
                "func": run_construct_web_search_queries,
                "depends_on": ["Manual Review - Mark relevant videos"] if outline_finalized else ["Consolidate Video Based Outlines"],
                "args": {
                    "sheet": "sheet",
                    "worksheet_name": "Base Outline",
                    "course_name": "course_name",
                    "target_audience": "target_audience",
                    "llm": llm_model,
                },
                "estimated_time": "~ 2 minutes",
                "description": "Generates web search queries in the `Base Outline` sheet to use for searching the web.",
                "delete_func": delete_web_search_queries,
                "delete_args": {
                    "sheet": "sheet",
                    "worksheet_name": "Base Outline",
                }
            },
            {
                "name": "Obtain Web Article Links",
                "func": run_web_search_screening,
                "depends_on": ["Web Search Queries Generator"],
                "args": {
                    "sheet": "sheet",
                    "worksheet_name": "Preliminary Research",
                    "course_name": "course_name",
                    "llm": llm_model,
                },
                "estimated_time": "~ 5 - 10 minutes",
                "description": "Executes web searches using generated queries and lists relevant article links in the `Preliminary Research` sheet.",
                "delete_func": delete_preliminary_research_sheet,
                "delete_args": {
                    "sheet": "sheet",
                    "worksheet_name": "Preliminary Research",
                }
            },
            {
                "name": "Fetch Web Article Content",
                "func": run_fetch_article_content,
                "depends_on": ["Obtain Web Article Links"],
                "args": {
                    "sheet": "sheet",
                    "worksheet_name": "Preliminary Research",
                },
                "estimated_time": "~ 10 - 20 minutes",
                "description": "Retrieves full content of web articles listed in the Preliminary Research sheet.",
                "delete_func": delete_article_content,
                "delete_args": {
                    "sheet": "sheet",
                    "worksheet_name": "Preliminary Research",
                }
            },
            {
                "name": "Extract Relevant Information from Articles",
                "func": run_get_relevant_info_from_article,
                "depends_on": ["Fetch Web Article Content"],
                "args": {
                    "sheet": "sheet",
                    "worksheet_name": "Preliminary Research",
                    "course_name": "course_name",
                    "target_audience": "target_audience",
                    "course_background": "course_background",
                    "llm": llm_model,
                },
                "estimated_time": "~ 40 - 60 minutes",
                "description": "Identifies and extracts pertinent content from each article to align with course context and audience.",
                "delete_func": delete_relevant_info,
                "delete_args": {
                    "sheet": "sheet",
                    "worksheet_name": "Preliminary Research",
                }
            },
            {
                "name": "Generate Research Summaries",
                "func": run_get_research_summary,
                "depends_on": ["Extract Relevant Information from Articles"],
                "args": {
                    "sheet": "sheet",
                    "worksheet_name": "Base Outline",
                    "course_name": "course_name",
                    "llm": llm_model,
                },
                "hide_if_final_outline": True,
                "estimated_time": "~ 5 - 10 minutes",
                "description": "Summarizes extracted information from web articles into concise, relevant summaries stored in the `Base Outline` sheet.",
                "delete_func": delete_research_summaries,
                "delete_args": {
                    "sheet": "sheet",
                    "worksheet_name": "Base Outline",
                }
            },
            {
                "name": "Manual Review - Web Research Summary",
                "func": manual_input_review_research_summary,
                "depends_on": ["Generate Research Summaries"],
                "args": {
                    "sheet": "sheet",
                    "worksheet_name": "Base Outline",
                    "course_name": "course_name",
                    "target_audience": "target_audience",
                    "skip_manual_step": "skip_manual_step",
                    "llm": "gemini_2_5_flash",
                },
                "instructions": [
                    "Your task is to populate the `Manual Extract` column in the `Base Outline` sheet.",
                    "- Open the `Base Outline` sheet.",
                    "- Look at the `research_summary` column.",
                    "- Copy the summary from **research_summary** into the **Manual Extract** column.",
                    "- Edit the text as needed—add, remove, or rephrase parts to make the summary more accurate and useful.",
                    "---",
                    "**Best Practices**",
                    "1. Remove anything that doesn't fit the topic.",
                    "2. Clarify vague or confusing points.",
                    "3. Add important missing info if needed.",
                    "4. Keep it short, clear, and focused on the topic.",
                    "---",
                    "**Example:**",
                    "- If the summary includes an off-topic idea, delete it.",
                    "- If a key detail is missing, add it.",
                    "- Make sure the final text reads smoothly and makes sense.",
                    "---",
                    "NOTE: Fill in the `Manual Extract` column for `every row` that has a `research_summary`. Leave it blank only if there's no research summary in that row.",
                ],
                "is_manual_step": True,
                "hide_if_final_outline": True,
                "estimated_time": "Manual step",
                "description": "Review, edit, and confirm AI-generated summaries, ensuring accuracy and relevancy.",
                "video_link": "https://drive.google.com/file/d/1Wjb6lO0zwKPV8577EcRyct9Tq6J6XDfG/view?usp=drive_link",
                "delete_func": clear_manual_extract,
                "delete_args": {
                    "sheet": "sheet",
                    "worksheet_name": "Base Outline",
                }
            },
            {
                "name": "Generate Web Research Based Outline",
                "func": run_generate_web_research_outline,
                "depends_on": ["Manual Review - Web Research Summary"],
                "args": {
                    "sheet": "sheet",
                    "worksheet_name": "Outline Consolidation",
                    "course_name": "course_name",
                    "course_background": "course_background",
                    # "llm": llm_model,
                },
                "hide_if_final_outline": True,
                "estimated_time": "~ 2 - 4 minutes",
                "description": "Creates a detailed course outline based on refined summaries, storing the output in the `Outline Consolidation` sheet.",
                "delete_func": delete_web_research_outline,
                "delete_args": {
                    "sheet": "sheet",
                    "worksheet_name": "Outline Consolidation",
                }
            },
        ],
    },
    {
        "section_name": "Section 4: Deep Research",
        "steps": [
            {
                "name": "Deep Research",
                "func": run_deep_research,
                "depends_on": ["Extract Relevant Information from Articles"] if outline_finalized else ["Generate Web Research Based Outline"],
                "args": {
                    "sheet": "sheet",
                    "worksheet_name": "Deep Research",
                    "course_name": "course_name",
                    "target_audience": "target_audience",
                    "llm": "gemini_with_grounding",
                },
                "estimated_time": "~ 5 - 10 minutes",
                "description": "Creates a new sheet `Deep Research`, performs agentic deep research on the Base outline subtopics and pastes the research into the sheet.",
                "delete_func": delete_deep_research_sheet,
                "delete_args": {
                    "sheet": "sheet",
                    "worksheet_name": "Deep Research",
                }
            },
            {
                "name": "Generate Deep Research Based Outline",
                "func": run_generate_deep_research_outline,
                "depends_on": ["Deep Research"],
                "args": {
                    "sheet": "sheet",
                    "worksheet_name": "Outline Consolidation",
                    "course_name": "course_name",
                    "course_background": "course_background",
                    "llm": llm_model,
                },
                "hide_if_final_outline": True,
                "estimated_time": "~ 5 - 10 minutes",
                "description": "Generates a course outline based on the deep research findings and saves it to a `Outline Consolidation` sheet.",
                "delete_func": delete_deep_research_outline,
                "delete_args": {
                    "sheet": "sheet",
                    "worksheet_name": "Outline Consolidation",
                }
            },
        ],
    },
    {
        "section_name": "Section 5: Outline Consolidation",
        "steps": [
            {
                "name": "Generate Consolidated Outline",
                "func": run_all_outlines_consolidated_and_review,
                "depends_on": ["Generate Deep Research Based Outline"],
                "args": {
                    "sheet": "sheet",
                    "worksheet_name": "Outline Consolidation",
                    "course_name": "course_name",
                    "target_audience": "target_audience",
                    "llm": llm_model,
                },
                "hide_if_final_outline": True,
                "estimated_time": "~ 2 - 4 minutes",
                "description": "Merges all previously created outlines from various sources into one comprehensive consolidated outline in the `Outline Review` sheet.",
                "delete_func": delete_all_outlines_consolidated_and_review,
                "delete_args": {
                    "sheet": "sheet",
                    "worksheet_name": "Outline Consolidation"
                }
            },
            {
                "name": "Review and Revise Outline",
                "func": run_review_and_revise_outline,
                "depends_on": ["Generate Consolidated Outline"],
                "args": {
                    "sheet": "sheet",
                    "course_name": "course_name",
                    "target_audience": "target_audience",
                    "llm": llm_model,
                    "skip_manual_step": "skip_manual_step"
                },
                "instructions": [
                    "In this step, you can collaborate with the AI to review and revise the course outline pasted in the `Outline Review` sheet. Follow the below steps:",
                    "- Open the `Outline Review` sheet.",
                    "- Please review the outline generated by AI in the `Outline` column",
                    "- You need to populate these two columns before running this step - `Verdict` and `Manual Feedback`.",
                    "- In the **Verdict** enter:",
                            "1. `Approved` — if the outline looks good and you can include that it looks good in the **Manual Feedback**column",
                            "2. `Rejected` — if changes are needed.",
                    "- If `Rejected`, populate the `Manual Feedback` column with specific comments on what should be changed in the outline.",
                    "---",
                    "**Best Practices**",
                    "1. Be specific about what needs to be removed, added, or improved.",
                    "2. Focus on clarity, relevance, and completeness.",
                    "3. Use simple language. Aim for helpful and actionable feedback.",
                    "---",
                    "**Example Feedback:**",
                    "- Remove section 3 on blower motors. It's not relevant to this course.",
                    "- Add an intro section to explain basic HVAC terms.",
                    "- Reorder sections 2 and 4 for better flow.",
                    "---",
                    "NOTE: If `Rejected`, add clear notes in the `Manual Feedback` column on what should be changed."
                ],
                "is_manual_step": True,
                "hide_if_final_outline": True,
                "estimated_time": "~ Semi-Automated Step",
                "description": "Review the AI outline in the `Outline Review` sheet. Add comments in the `Verdict` column (valid options are Approved / Rejected) and in the `Manual Feedback` column.",
                "video_link":"https://drive.google.com/file/d/1c062MbZbwes69Oq64QNB7wA8JLiQfk-Z/view?usp=drive_link",
                "pre_exec_func": print_course_outline_before_review,
                "pre_exec_always_run": False,
                "pre_exec_args": {
                    "sheet": "sheet",
                    "worksheet_name": "Outline Review",
                },
                "delete_func": delete_review_and_revise_outline,
                "delete_args": {
                    "sheet": "sheet",
                }
            },
        ],
    },
]

# Conditionally add Section 6: Enhance Outline
if topic_deep_research_enabled:
    pipeline_sections.append({
        "section_name": "Section 6: Enhance Outline",
        "steps": [
            {
                "name": "Manual Step - Create Topic Outline Sheet",
                "func": run_create_topic_outline_sheet,
                "depends_on": ["Generate Consolidated Outline"],
                "args": {
                    "sheet": "sheet",
                    "worksheet_name": "Topic Outline",
                    "use_existing_outline": "use_existing_outline",
                    "skip_manual_step": "skip_manual_step",
                },
                "instructions": [],
                "is_manual_step": True,
                "hide_if_final_outline": True,
                "estimated_time": "~ Manual Step",
                "description": "Create the `Topic Outline` sheet with following two columns: `Topic` and `Learning Objective`.",
                "video_link": "https://drive.google.com/file/d/1csVwfr6Vhi5ZPJjw3WhKxxZWFs_XwJ24/view?usp=drive_link",
                "pre_exec_func": pre_topic_deep_research,
                "pre_exec_always_run": True,
                "pre_exec_args": {
                    "sheet": "sheet",
                    "worksheet_name": "Topic Outline",
                },
                "delete_func": delete_create_topic_outline,
                "delete_args": {
                    "sheet": "sheet",
                    "topic_outline_ws": "Topic Outline",
                    "topic_deep_research_ws": "Topic Deep Research",
                }
            },
            {
                "name": "Topic Deep Research",
                "func": run_topic_deep_research,
                "depends_on": ["Manual Step - Create Topic Outline Sheet"],
                "args": {
                    "sheet": "sheet",
                    "worksheet_name": "Topic Deep Research",
                    "course_name": "course_name",
                    "target_audience": "target_audience",
                    "llm": "gemini_with_grounding",
                },
                "hide_if_final_outline": True,
                "estimated_time": "~ 5 - 10 minutes",
                "description": "Performs deep research on each topic in the Topic Deep Research sheet and populates the research and sources columns.",
                "delete_func": delete_topic_deep_research,
                "delete_args": {
                    "sheet": "sheet",
                    "worksheet_name": "Topic Deep Research",
                }
            },
            {
                "name": "Generate Learning Objectives",
                "func": run_generate_learning_objectives,
                "depends_on": ["Topic Deep Research"],
                "args": {
                    "sheet": "sheet",
                    "worksheet_name": "Topic Deep Research",
                    "course_name": "course_name",
                    "target_audience": "target_audience",
                    "llm": llm_model,
                },
                "hide_if_final_outline": True,
                "estimated_time": "~ 2 - 5 minutes",
                "description": "Generates learning objectives for each topic based on the research data and populates the learning objectives column in the Topic Deep Research sheet.",
                "delete_func": delete_learning_objectives,
                "delete_args": {
                    "sheet": "sheet",
                    "worksheet_name": "Topic Deep Research",
                }
            },
            {
                "name": "Categorize Learning Objectives",
                "func": run_categorize_learning_objectives,
                "depends_on": ["Generate Learning Objectives"],
                "args": {
                    "sheet": "sheet",
                    "worksheet_name": "Topic Deep Research",
                    "course_name": "course_name",
                    "target_audience": "target_audience",
                    "llm": llm_model,
                },
                "hide_if_final_outline": True,
                "estimated_time": "~ 2 - 5 minutes",
                "description": "Categorizes learning objectives for each topic in the `Topic Deep Research` sheet based on their relevance to the course outline.",
                "delete_func": delete_lo_categorization,
                "delete_args": {
                    "sheet": "sheet",
                    "worksheet_name": "Topic Deep Research",
                }
            },
            {
                "name": "Label Learning Objectives",
                "func": run_label_learning_objectives,
                "depends_on": ["Categorize Learning Objectives"],
                "args": {
                    "sheet": "sheet",
                    "worksheet_name": "Missing Learning Objectives",
                    "course_name": "course_name",
                    "target_audience": "target_audience",
                    "llm": llm_model,
                },
                "hide_if_final_outline": True,
                "estimated_time": "~ 2 - 5 minutes",
                "description": "Labels learning objectives for each topic in the `Topic Deep Research` sheet based on their relevance to the course outline.",
                "delete_func": delete_label_learning_objectives,
                "delete_args": {
                    "sheet": "sheet",
                    "topic_deep_research_ws": "Topic Deep Research",
                }
            },
            {
                "name": "Review and Revise Topic Outline",
                "func": run_review_and_revise_topic_outline,
                "depends_on": ["Label Learning Objectives"],
                "args": {
                    "sheet": "sheet",
                    "course_name": "course_name",
                    "target_audience": "target_audience",
                    "llm": llm_model,
                    "skip_manual_step": "skip_manual_step"
                },
                "instructions": [
                    "In this step, you can collaborate with the AI to review and revise the course outline pasted in the `Enhanced Outline Review` sheet. Follow the below steps:",
                    "- Open the `Enhanced Outline Review` sheet.",
                    "- Please review the outline generated by AI in the `Outline` column",
                    "- You need to populate these two columns before running this step - `Verdict` and `Manual Feedback`.",
                    "- In the **Verdict** enter:",
                            "1. `Approved` — if the outline looks good and you can include that it looks good in the **Manual Feedback**column",
                            "2. `Rejected` — if changes are needed.",
                    "- If `Rejected`, populate the `Manual Feedback` column with specific comments on what should be changed in the outline.",
                    "---",
                    "**Best Practices**",
                    "1. Be specific about what needs to be removed, added, or improved.",
                    "2. Focus on clarity, relevance, and completeness.",
                    "3. Use simple language. Aim for helpful and actionable feedback.",
                    "---",
                    "**Example Feedback:**",
                    "- Remove section 3 on blower motors. It's not relevant to this course.",
                    "- Add an intro section to explain basic HVAC terms.",
                    "- Reorder sections 2 and 4 for better flow.",
                    "---",
                    "NOTE: If `Rejected`, add clear notes in the `Manual Feedback` column on what should be changed."
                ],
                "is_manual_step": True,
                "hide_if_final_outline": True,
                "estimated_time": "~ Semi-Automated Step",
                "description": "Review and revise the outline in the `Enhanced Outline Review` sheet.",
                "video_link": "https://drive.google.com/file/d/1hwvyfFfRPvTnDLnQzfx7L1rmR7cS5g4q/view?usp=drive_link",
                "pre_exec_func": show_outline_diff,
                "pre_exec_always_run": st.session_state.get("pre_exec_show_topic_outline_diff", True),
                "pre_exec_args": {
                    "sheet": "sheet",
                },
                "delete_func": delete_review_and_revise_topic_outline,
                "delete_args": {
                    "sheet": "sheet",
                }
            },
            {
                "name": "Map Topic Outline to Enhanced Outline",
                "func": map_original_outline_to_revised_outline_for_all_topics,
                "depends_on": ["Review and Revise Topic Outline"],
                "args": {
                    "sheet": "sheet",
                    "worksheet_name": "Enhanced Outline with LOs",
                    "llm": llm_model,
                },
                "hide_if_final_outline": True,
                "estimated_time": "~ 2 - 5 minutes",
                "description": "Maps additional columns from the topic outline to the enhanced outline.",
                "delete_func": delete_topic_outline_mapping,
                "delete_args": {
                    "sheet": "sheet",
                    "mapping_ws": "Topic - Revised Outline Mapping",
                    "topic_outline_ws": "Topic Outline",
                    "enhanced_outline_ws": "Enhanced Outline with LOs",
                }
            },
        ]
    })

pipeline_sections.append({
    "section_name": "Section: Final Outline Sheet Creation",
    "steps": [
        {
            "name": "Create the Final Outline Sheet",
            "func": create_final_outline_sheet,
            "depends_on": ["Map Topic Outline to Enhanced Outline"] if topic_deep_research_enabled and not outline_finalized else ["Deep Research"] if outline_finalized else ["Review and Revise Outline"],
            "args": {
                "sheet": "sheet",
            },
            "estimated_time": "~ 1 minute",
            "description": "Creates the 'Final Outline' sheet by flattening multiple LOs into one-per-row format.",
            "delete_func": delete_final_outline,
            "delete_args": {
                "sheet": "sheet",
                "worksheet_name": "Final Outline",
            }
        }
    ]
})

# pipeline_sections.append({
#     "section_name": "Section: Checklist based Review-Revise Agents",
#     "steps": [
#         {
#             "name": "Checklist Based Review and Revise Agents",
#             "func": run_outline_checklist_review_and_revise,
#             "depends_on": ["Create the Final Outline Sheet"],
#             "args": {
#                 "sheet": "sheet",
#                 "worksheet_name": "Final Outline",
#                 "llm": "gemini_2_5_flash",
#             },
#             "delete_func": delete_outline_checklist_review_and_revise,
#             "delete_args": {
#                 "sheet": "sheet",
#                 "worksheet_name": "Final Outline",
#             },
#             "hide_if_final_outline": True,
#             "estimated_time": "~ 5 - 10 minutes",
#             "description": "Reviews and Revises the Final Outline based on predefined checklist criteria, ensuring quality and consistency.",
#         }
#     ]
# })

pipeline_sections.append({
    "section_name": "Section: Get References for the Final Outline",
    "steps": [
        {
            "name": "Get relevant references for Learning Objectives",
            "func": load_references,
            "depends_on": ["Create the Final Outline Sheet"], #["Checklist Based Review and Revise Agents"] if not outline_finalized else ["Create the Final Outline Sheet"],
            "args": {
                "sheet": "sheet",
            },
            "estimated_time": "~ 2-5 minutes",
            "description": "Loads all reference documents into the vectorstore for faster retrieval.",
            "delete_func": delete_all_references,
            "delete_args": {
                "sheet": "sheet",
                "worksheet_name": "All References",
            }
        },
        {
            "name": "Retrieve relevant references for Learning Objectives",
            "func": run_retriever_agent_for_all_rows,
            "depends_on": ["Get relevant references for Learning Objectives"],
            "args": {
                "root_folder_id": "root_folder_id",
                "drive": "drive",
                "sheet": "sheet",
                "worksheet_name": "Final Outline",
                "course_name": "course_name",
                "target_audience": "target_audience",
                "llm": llm_model,
            },
            "estimated_time": "~ 10 - 20 minutes",
            "description": "Gathers relevant context needed for the research.",
            "delete_func": delete_retriever_context,
            "delete_args": {
                "sheet": "sheet",
                "worksheet_name": "Final Outline",
            }
        },
        
        {
            "name": "Retrieve relevant HVAC Videos for Learning Objectives",
            "func": run_video_search_for_los,
            "depends_on": ["Get relevant references for Learning Objectives"],
            "args": {
                "sheet": "sheet",
                "worksheet_name": "Final Outline",
                "drive": "drive",
                "llm": llm_model,
            },
            "estimated_time": "~ 10 - 20 minutes",
            "description": "Searches for relevant HVAC videos based on the learning objectives in the Final Outline sheet.",
            # "delete_func": delete_retriever_context,
            # "delete_args": {
            #     "sheet": "sheet",
            #     "worksheet_name": "Final Outline",
            # }
        },
    ]
})

agent_ui(step_name="Course Outline", pipeline_sections=pipeline_sections, outline_finalized=outline_finalized)


