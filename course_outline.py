from agent_ui_template import agent_ui

from agents.course_outline.video_based_outline.video_search_queries import run_construct_video_search_queries, manual_input_review_video_search_queries
from agents.course_outline.video_based_outline.get_hvac_school_videos import run_get_hvac_school_videos
from agents.course_outline.video_based_outline.classify_and_chunk_videos import run_get_transcripts, run_classify_video, run_chunk_videos
from agents.course_outline.video_based_outline.get_relevant_chunks import run_get_relevant_chunks, manual_input_mark_relevant_videos
from agents.course_outline.video_based_outline.video_based_outlines import run_generate_video_based_outline, manual_input_video_outline_consolidation_comments
from agents.course_outline.video_based_outline.consolidate_video_based_outline import run_propose_consolidated_video_outlines

from agents.course_outline.client_reference_based_outline.client_reference_based_outlines import run_generate_outline_from_client_reference, manual_input_client_reference_consolidation_comments
from agents.course_outline.client_reference_based_outline.consolidate_client_reference_based_outline import run_propose_consolidated_reference_outlines

from agents.course_outline.web_research_based_outline.web_search_queries import run_construct_web_search_queries
from agents.course_outline.web_research_based_outline.web_search_screening import run_web_search_screening
from agents.course_outline.web_research_based_outline.get_article_content import run_fetch_article_content
from agents.course_outline.web_research_based_outline.extract_relevant_info import run_get_relevant_info_from_article
from agents.course_outline.web_research_based_outline.get_research_summary import run_get_research_summary, manual_input_review_research_summary
from agents.course_outline.web_research_based_outline.web_research_based_outline import run_generate_web_research_outline

from agents.course_outline.outline_consolidation.outline_consolidation import run_all_outlines_consolidated_and_review
from agents.course_outline.outline_consolidation.review_and_revise_outline import run_review_and_revise_outline


# --- 1) Define pipeline as sections, each with its own steps ---
pipeline_sections = [
    {
        "section_name": "Section 1: Videos Research",
        "steps": [
            {
                "name": "Video Search Queries Generator",
                "func": run_construct_video_search_queries,
                "depends_on": [],
                "args": {
                    "sheet": "sheet",
                    "worksheet_name": "Rough Outline",
                    "course_name": "course_name",
                    "target_audience": "target_audience",
                    "llm": "gemini_2_flash",
                },
                "estimated_time": "~ 1 minute",
                "description": "Generates search queries to find relevant HVAC school youtube videos.",
            },
            {
                "name": "Manual Review - Video Search Queries",
                "func": manual_input_review_video_search_queries,
                "depends_on": ["Video Search Queries Generator"],
                "args": {
                    "sheet": "sheet",
                    "worksheet_name": "Rough Outline",
                },
                "instructions": [
                    "Check the video search queries in the Rough Outline sheet.",
                    "Add / Delete / Modify the search queries if needed.",
                    "Tab Name - `Rough Outline`",
                    "Column(s) to check - `video_search_queries`"
                ],
                "estimated_time": "Manual step",
                "description": "Manually review the generated video search queries and make optional edits if needed.",
            },
            {
                "name": "Get HVAC School Videos",
                "func": run_get_hvac_school_videos,
                "depends_on": ["Manual Review - Video Search Queries"],
                "args": {
                    "sheet": "sheet",
                    "worksheet_name": "Videos Research",
                },
                "estimated_time": "~ 1 minute",
                "description": "Retrives the HVAC school YouTube videos and lists them in the `Videos Research` sheet"
            },
            {
                "name": "Get Video Transcripts",
                "func": run_get_transcripts,
                "depends_on": ["Get HVAC School Videos"],
                "args": {
                    "sheet": "sheet",
                    "worksheet": "Videos Research",
                },
                "estimated_time": "~ 10 - 20 minutes",
                "description": "Gets Video Transcript for all videos and saves them to sheet."
            },
            {
                "name": "Check Video Relevance",
                "func": run_classify_video,
                "depends_on": ["Get Video Transcripts"],
                "args": {
                    "sheet": "sheet",
                    "worksheet_name": "Videos Research",
                    "course_name": "course_name",
                    "target_audience": "target_audience",
                    "llm": "gemini_2_flash"
                },
                "estimated_time": "~ 10 - 20 minutes",
                "description": "Analyze each videos in the `Videos Research` sheet to mark them as Relevant / Irrelevant."
            },
            {
                "name": "Chunk Videos",
                "func": run_chunk_videos,
                "depends_on": ["Check Video Relevance"],
                "args": {
                    "sheet": "sheet",
                    "videos_research_worksheet_name": "Videos Research",
                    "video_chunks_worksheet_name": "Video Chunks",
                    "llm": "gemini_2_flash"
                },
                "estimated_time": "~ 10 - 20 minutes",
                "description": "Chunk and add all relevant videos in `Video Chunks` sheet."
            },
            {
                "name": "Get Relevant Chunks",
                "func": run_get_relevant_chunks,
                "depends_on": ["Chunk Videos"],
                "args": {
                    "sheet": "sheet",
                    "worksheet_name": "Videos Research",
                    "course_name": "course_name",
                    "target_audience": "target_audience",
                    "llm": "gemini_2_flash"
                },
                "estimated_time": "~ 10 - 20 minutes",
                "description": "Identifies relevant chunks for all the videos marked as relevant in previous step."
            },
            {
                "name": "Manual Review - Mark relevant videos",
                "func": manual_input_mark_relevant_videos,
                "depends_on": ["Get Relevant Chunks"],
                "args": {
                    "sheet": "sheet",
                    "worksheet_name": "Videos Research",
                    "skip_manual_step": "skip_manual_step"
                },
                "instructions": [
                    "Your task is to mark videos that are actually relevant to this course. To do so, follow these steps:",
                    "1) Please review the videos marked as relevant by AI in the `Videos Research` sheet",
                    "2) The automation has applied a filter to narrow down the list of rows you need to review.",
                    "3) To complete the review, you must manually populate the following columns: `Manual Review`, `Used for`",
                    "4) `Manual Review` column should be populated with a `Yes` or a `No`. Yes means video is relevant.",
                    "5) `Used for` column should be populated with a `Just Content` or `As a video`. Can be left blank for `No` rows",
                    "---",
                    "NOTE: You need to populate all the rows visible in the filter view to complete this step."
                ],
                "estimated_time": "Manual step",
                "description": "Manually review the overall AI analysis so far and mark relevant videos.",
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
                    "llm": "gemini_2_flash"
                },
                "estimated_time": "~ 5 - 10 minutes",
                "description": "Generates video based outlines from the video transcripts for all the videos marked as Yes in Manual Review."
            },
            {
                "name": "Manual Review - Video Outline Consolidation Comments",
                "func": manual_input_video_outline_consolidation_comments,
                "depends_on": ["Generate Video Based Outlines"],
                "args": {
                    "sheet": "sheet",
                    "worksheet_name": "Videos Research",
                    "skip_manual_step": "skip_manual_step"
                },
                "instructions": [
                    "Your task is to add comments to tell the AI how you'd like to consolidate the video based outlines into a single outline.",
                    "1) Please review the videos based outlines generated by AI in the `Videos Research` sheet",
                    "2) Add comments in the `consolidation_comments` column.",
                    "3) For example, if you disagree with some topic / concept included in the outline for a given row, simply put your comments telling the AI why do you disagree.",
                    "4) The above is just an example of what can be put in the comments.",
                    "5) Think of these comments as telling the AI how to incorporate all this information present into a single coherent outline.",
                    "---",
                    "NOTE: You need to populate the `consolidation_comments` column for all the rows populated with `outline`."
                ],
                "estimated_time": "Manual step",
                "description": "Manually add consolidation comments for the generated outlines telling the AI how to combine the outlines into one.",
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
                    # "llm": "gemini_2_flash"
                },
                "estimated_time": "~ 2 minutes",
                "description": "Consolidates the multiple video based outline into a single outline. Technically it generates 4 variations of the outline in the `Outline Consolidation` sheet"
            },
        ],
    },
    {
        "section_name": "Section 2: Client References",
        "steps": [
            {
                "name": "Generate Client Reference Based Outlines",
                "func": run_generate_outline_from_client_reference,
                "depends_on": [],
                "args": {
                    "sheet": "sheet",
                    "worksheet_name": "Client References",
                    "course_name": "course_name",
                    "target_audience": "target_audience",
                    "llm": "gemini_2_flash",
                },
                "estimated_time": "~ 5 - 10 minutes",
                "description": "Generates client reference based outlines for all the rows in the `Client Reference` sheet.",
            },
            {
                "name": "Manual Review - Client References Outline Consolidation Comments",
                "func": manual_input_client_reference_consolidation_comments,
                "depends_on": ["Generate Client Reference Based Outlines"],
                "args": {
                    "sheet": "sheet",
                    "worksheet_name": "Client References",
                    "skip_manual_step": "skip_manual_step"
                },
                "instructions": [
                    "Your task is to add comments to tell the AI how you'd like to consolidate the client references based outlines into a single outline.",
                    "1) Please review the client references based outlines generated by AI in the `Client References` sheet",
                    "2) Add comments in the `consolidation_comments` column.",
                    "3) For example, if you disagree with some topic / concept included in the outline for a given row, simply put your comments telling the AI why do you disagree.",
                    "4) The above is just an example of what can be put in the comments.",
                    "5) Think of these comments as telling the AI how to incorporate all this information present into a single coherent outline.",
                    "---",
                    "NOTE: You need to populate the `consolidation_comments` column for all the rows populated with `outline`."
                ],
                "estimated_time": "Manual step",
                "description": "Manually add consolidation comments for the generated outlines telling the AI how to combine the outlines into one.",
            },
            {
                "name": "Consolidate Client Reference Based Outline",
                "func": run_propose_consolidated_reference_outlines,
                "depends_on": ["Manual Review - Client References Outline Consolidation Comments"],
                "args": {
                    "sheet": "sheet",
                    "worksheet_name": "Outline Consolidation",
                    "course_name": "course_name",
                    "target_audience": "target_audience",
                    # "llm": "gemini_2_flash",
                },
                "estimated_time": "~ 2 minutes",
                "description": "Consolidates the multiple client reference based outline into a single outline. Technically it generates 4 variations of the outline in the `Outline Consolidation` sheet",
            },
        ],
    },
    {
        "section_name": "Section 3: Web Research",
        "steps": [
            {
                "name": "Web Search Queries Generator",
                "func": run_construct_web_search_queries,
                "depends_on": [],
                "args": {
                    "sheet": "sheet",
                    "worksheet_name": "Rough Outline",
                    "course_name": "course_name",
                    "target_audience": "target_audience",
                    "llm": "gemini_2_flash",
                },
                "estimated_time": "~ 2 minutes",
                "description": "Generates web search queries in the `Rough Outline` sheet to use for searching the web.",
            },
            {
                "name": "Get Web Article Links",
                "func": run_web_search_screening,
                "depends_on": ["Web Search Queries Generator"],
                "args": {
                    "sheet": "sheet",
                    "worksheet_name": "Preliminary Research",
                    "course_name": "course_name",
                    "llm": "gemini_2_flash",
                },
                "estimated_time": "~ 5 - 10 minutes",
                "description": "Generates web search queries in the `Rough Outline` sheet to use for searching the web.",
            },
            {
                "name": "Get Web Article Content",
                "func": run_fetch_article_content,
                "depends_on": ["Get Web Article Links"],
                "args": {
                    "sheet": "sheet",
                    "worksheet_name": "Preliminary Research",
                },
                "estimated_time": "~ 10 - 20 minutes",
                "description": "Gets the web article content for all the rows in the `Preliminary Research` sheet.",
            },
            {
                "name": "Get Relevant Info from Articles",
                "func": run_get_relevant_info_from_article,
                "depends_on": ["Get Web Article Content"],
                "args": {
                    "sheet": "sheet",
                    "worksheet_name": "Preliminary Research",
                    "course_name": "course_name",
                    "target_audience": "target_audience",
                    "course_background": "course_background",
                    "llm": "gemini_2_flash",
                },
                "estimated_time": "~ 40 - 60 minutes",
                "description": "Get relevant information from the web articles for all the rows in the `Preliminary Research` Sheet.",
            },
            {
                "name": "Get Research Summary",
                "func": run_get_research_summary,
                "depends_on": ["Get Relevant Info from Articles"],
                "args": {
                    "sheet": "sheet",
                    "worksheet_name": "Rough Outline",
                    "course_name": "course_name",
                    "llm": "gemini_2_flash",
                },
                "estimated_time": "~ 5 - 10 minutes",
                "description": "Get the research summary for the search queries inside the `Rough Outline` sheet.",
            },
            {
                "name": "Manual Review - Web Research Summary",
                "func": manual_input_review_research_summary,
                "depends_on": ["Get Research Summary"],
                "args": {
                    "sheet": "sheet",
                    "worksheet_name": "Rough Outline",
                    "course_name": "course_name",
                    "target_audience": "target_audience",
                    "skip_manual_step": "skip_manual_step",
                    "llm": "gemini_2_flash",
                },
                "instructions": [
                    "Your task is to populate the `Manual Extract` column in the `Rough Outline` sheet.",
                    "1) Please review the `research_summary` column in the `Rough Outline` sheet.",
                    "2) Copy-paste the text in `research_summary` into the `Manual Extract` and make edits (eg. add, delete, modify)",
                    "3) For example, if you disagree with some topic / concept included in the summary for a given row, simply delete that part.",
                    "4) The above is just an example of the kind of edits to make.",
                    "5) The info present in `Manual Extract` column will be used in the next step to produce web research notes based outline.",
                    "---",
                    "NOTE: You need to populate the `Manual Extract` column for all the rows populated with `research_summary`."
                ],
                "estimated_time": "Manual step",
                "description": "Review and refine the `research_summary` and paste cleaned text in the `Manual Extract` column within the `Rough Outline` sheet.",
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
                    # "llm": "gemini_2_flash",
                },
                "estimated_time": "~ 2 - 4 minutes",
                "description": "Generates web research based outline in the `Outline Consolidation` sheet.",
            },
        ],
    },
    {
        "section_name": "Section 4: Outline Consolidation",
        "steps": [
            {
                "name": "Generate Consolidated Outline",
                "func": run_all_outlines_consolidated_and_review,
                "depends_on": ["Consolidate Video Based Outlines", "Consolidate Client Reference Based Outline", "Generate Web Research Based Outline"],
                "args": {
                    "sheet": "sheet",
                    "worksheet_name": "Outline Consolidation",
                    "course_name": "course_name",
                    "target_audience": "target_audience",
                    "llm": "gemini_2_flash",
                },
                "estimated_time": "~ 2 - 4 minutes",
                "description": "Consolidates all the outlines from the various sources and pastes the consolidated outline in `Outline Review` sheet.",
            },
            {
                "name": "Review and Revise Outline",
                "func": run_review_and_revise_outline,
                "depends_on": ["Generate Consolidated Outline"],
                "args": {
                    "sheet": "sheet",
                    "course_name": "course_name",
                    "target_audience": "target_audience",
                    "llm": "gemini_2_flash",
                    "skip_manual_step": "skip_manual_step"
                },
                "instructions": [
                    "In this step, you can collaborate with the AI to review and revise the course outline pasted in the `Outline Review` sheet. Follow the below steps.",
                    "1) Please review the outline generated by AI in the `Outline` column",
                    "2) You need to populate these two columns before running this step - `Verdict` and `Manual Feedback`.",
                    "3) In the `Verdict` column, say `Approved` if to approve the outline. Else say `Rejected`.",
                    "3) If `Rejected`, populate the `Manual Feedback` column with specific comments on what should be changed in the outline.",
                    "4) For example, I can say 'Remove section 3 on blower motors as it is not relevant...' or something along these lines.",
                    "5) After entering the inputs, click the button to get revised outline.",
                    "---",
                    "NOTE: You need to populate the `Verdict` column and `Manual Feedback` (if rejected) column."
                ],
                "estimated_time": "~ Semi-Automated Step",
                "description": "Review the AI outline in the `Outline Review` sheet. Add comments in the `Verdict` (valid options are Approved / Rejected) and in the `Manual Feedback` column.",
            },
        ],
    },

]


agent_ui(step_name = "Course Outline", pipeline_sections = pipeline_sections)
