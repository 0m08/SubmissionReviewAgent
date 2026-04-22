from __future__ import annotations

import os
import re
import json
import threading
import tempfile
from datetime import datetime
from urllib.parse import parse_qs, urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Set, Tuple

import streamlit as st
from dotenv import load_dotenv
from langsmith import traceable
from google.genai import types

from services.sheets_service import (
    get_sheet_data_and_df,
    save_to_sheet,
    format_worksheet,
    clear_worksheet,
)
from services.smart_progress_bar import SmartProgressBar
from services.helper_functions import build_video_part

from agents.graphics_definition_v2.aggregation_agent.aggregation_agent import (
    get_drive_instance,
    parse_video_url_timestamps,
    convert_watch_url_to_embed_url,
    expand_youtube_single_timestamp_clips_in_xml,
    upload_image_to_drive,
)
from agents.graphics_asset_creation.generator.image_generator import generate_asset

from agents.graphics_definition_v2.review_agent.review_and_revise import (
    _safe_str,
    _extract_tag,
    _parse_segment_marker,
    _sheet_lock,
    build_segment_visual_map,
    build_candidates_for_segment,
    build_candidate_text,
    build_video_candidates_text,
    build_current_visual_parts,
    build_visual_part_only,
    build_review_targets,
    build_review_targets_for_whole_slide,
    build_assets_for_segments,
    resolve_asset_urls_in_definition,
    update_final_graphics_definition_with_replacements,
    process_video_frames_in_text_format,
    normalize_youtube_timestamp_urls,
    add_snapshot_label_to_drive_links,
    invoke_gemini_multimodal,
    parse_review_response,
    regenerate_failed_segments,
)

load_dotenv()

MAX_HUMAN_FEEDBACK_REGEN_ATTEMPTS = 2
AI_GENERATED_IMAGES_FOLDER_ID = "1c3rmYhF8kCrJVv3ui1-362mr90OCkEqB"
AI_NO_FEEDBACK_MARKER = "No Feedback"

# Tracking line prefixes (must match _format_human_feedback_revision_tracking)
ORIG_PREFIX = "Original visual - "
MANUAL_PREFIX = "Manual Selection - "
AFTER_REV_PREFIX = "After revision - "
AFTER_REGEN1_PREFIX = "After regeneration loop 1 - "
AFTER_REGEN2_PREFIX = "After regeneration loop 2 - "
NO_REPLACEMENT = "No replacement"
HUMAN_FEEDBACK_REVISION_TRACKING_COLUMN = "human_feedback_revision_tracking"


# ===========================================================================
# PROMPTS
# ===========================================================================

# Revision prompt to use when we have flexible or 1 visual per sentence visual assingment strategy
HUMAN_FEEDBACK_REVISION_PROMPT = """You are a Graphics Definition Revision Agent specializing in the field of HVAC. Your task is to revise and correct the graphics definition for a voiceover (VO) segment when a human reviewer has requested changes, by selecting the most appropriate visual from the provided candidate image and video pools.
You will be given the voiceover segment, the slide and course context for reference, the visuals currently assigned to this segment, the human-provided feedback describing what they want changed, and a pool of candidate visuals from which to select the most appropriate visual(s) to replace the current visual(s) based on the feedback.

Inputs:
These are the inputs for your revision:

<course_information>
Course name: {course_name}
Target audience: {target_audience}
Topic name: {topic_name}
Subtopic name: {subtopic_name}
</course_information>

<slide_information>
Slide title: {slide_title}
Slide content: {slide_chunk}
</slide_information>

This is the voiceover segment from the slide content for which the revision is needed:
<voiceover_segment>
{vo_text}
</voiceover_segment>

These are the visuals currently assigned to the voiceover segment:
<current_visuals_assigned>
{current_visuals}
</current_visuals_assigned>

This is the human-provided feedback requesting the following change(s) for the visual(s) assigned to this voiceover segment:
<human_feedback>
{human_feedback}
</human_feedback>

These are the candidate images and videos from which to select the most appropriate visual to replace the current visual(s) based on the human feedback:

<candidates>

<image_candidates>
{image_candidates}
</image_candidates>

<video_candidates>
{video_candidates}
</video_candidates>

</candidates>

Instructions:

1. Scope and Revision Responsibility
   - Your task is to revise the graphics definition for this single voiceover (VO) segment only.
   - Use the slide content and the specific voiceover segment to understand the context and the instructional intent.
   - The assigned visuals are displayed on screen as the voiceover text for that segment is played/narrated.
   - When interpreting any part of the voiceover sentences, you must use the surrounding text in the slide content to understand the context and the instructional intent, before selecting visuals for the respective parts of the slide. Do not interpret a sentence, phrase or a clause literally in isolation. Always derive the meaning of a sentence, phrase or a clause from the surrounding text in the slide content.
   - Identify the specific visual requirement implied by the human feedback.
   - If the human feedback asks for a particular visual subject, object, scene, component, diagram, process, or other concrete visual requirement, treat that requirement as mandatory. You must actively look for a candidate that actually shows that requested visual, not just something broadly related to the same topic.
   - Apply changes only to the visual(s) that the human feedback refers to.
   - Select replacement visuals only from the provided candidate image and video pools.

2. Candidate Evaluation and Visual Replacement
   - Carefully review all provided image and video candidates for this voiceover segment.
   - Evaluate each candidate only against the specific visual requirement described in the human feedback.
   - Select the candidate that most directly and clearly satisfies the human feedback while remaining aligned with the voiceover segment.
   - If the feedback explicitly requests a specific kind of visual, do not choose a generic, adjacent, or loosely related candidate when another candidate actually shows the requested thing more directly.
   - If no candidate fully satisfies the human feedback, select the closest acceptable alternative.
   - Ensure the replacement visual(s) are instructionally clear and effective for the voiceover segment.

3. Visual Form and Usage Constraints
   - You may select images, video clips with timestamps, or still frames extracted from videos, using only the provided candidate visuals.
   - Choose the visual form (image, video clip, or still frame) that most clearly satisfies the human feedback while fitting within the narration timing of the relevant part(s) of the voiceover segment.
   - When you find both a video clip and a still image that equally satisfies the human feedback for any part of the voiceover sentence, prefer using the video clip, since motion can add useful context. This is a guiding preference, not a strict rule - do not prioritize a video clip over an image if the video is only partially relevant, loosely related, or less effective than the still image at supporting the narration and addressing the feedback.
   - When selecting a video clip, identify the exact portion of the video that visually supports the required detail and assign appropriate start and end timestamps. Strictly use such example format of video URL with start and end timestamps: e.g. "https://www.youtube.com/embed/dQw4w9WgXc?start=10&end=20"
   - When selecting a still frame from a video so that it can be used as a static image, output the video URL with a single start timestamp only (no end timestamp). Strictly use such example format of video URL with a single start timestamp indicating the frame timestamp: e.g. "https://www.youtube.com/embed/dQw4w9WgXc?start=10"
   - Respect source-specific constraints when selecting video candidates from the provided candidate video pools.
     - Video candidates listed under:
        "Videos from which you can use video clips (with timestamps) or still frames as images" may be used in any of the following ways:
        - short video clips with start and end timestamps
        - still frames extracted from the video
     - Video candidates listed under:
        "Videos from which you can ONLY use still frames as images (NOT playable video clips with timestamps)" have the following strict constraints:
        - You MUST NOT select them as playable video clips
        - You MUST NOT assign start–end timestamps
        - You MAY ONLY extract still frames and use them as static images
        - When using these videos, the asset MUST be represented as a video URL with a single start timestamp only

Output:

Always provide your output strictly in the following format:

<output>

<evaluation_breakdown>

Use this section as a structured reasoning and scratchpad space for you to address the human feedback and select the most appropriate visual(s) to replace the current visual(s). Use it to document your observations, reasoning, and decision process. Provide the following sections:

1. Voiceover Sentence Understanding
   - Briefly explain, in your own words, what the voiceover sentence is communicating and what part of the sentence the human feedback is addressing. Use the slide content to resolve any references, pronouns, or implied meaning if needed.

2. Human Feedback Interpretation
   - Briefly summarize what the human feedback indicates should be changed.
   - Identify how many visual(s) are currently assigned to the voiceover sentence and how many need to be replaced.
   - Clearly state the specific visual requirement that must be satisfied by the revision.
   - If the feedback contains any explicit visual requirement, name that exact requirement and treat it as a hard constraint for a fully satisfactory replacement.

3. Image Candidates Scan
   - Create a numbered list of all provided image candidates and briefly describe what you see in each of the image candidate.

4. Video Candidates Scan
   - Create a numbered list of all provided video candidates and briefly describe what you see in each of the video candidate.
   - Focus on explaining the visual content of the video, and not what is being spoken in the video. Split the list into two sections: one for videos from which video clips (with timestamps) or still frames as images can be used, and one for videos from which ONLY still frames as images can be used (NOT playable video clips with timestamps).

5. Candidate Fit Analysis
   - Compare the candidates against the human feedback requirement(s).
   - Identify which candidate(s) most directly satisfy the requirement and why.
   - Explicitly check whether each strong candidate visibly contains the exact thing the human feedback asked for. Topic similarity alone is not enough.
   - If multiple candidates partially match, reason about which one is the closest acceptable match.
   - If no candidate fully satisfies the human feedback, determine the closest acceptable alternative.

6. Video Timestamp / Frame Selection Thinking (only if selecting video to replace any of the visual(s))
   - If selecting a playable video clip if you find a relevant one that best satisfies the human feedback requirement:
   - Determine exactly which portion of the video is visually relevant and will satisfy the feedback.
   - Plan the correct start and end timestamps for that portion, ensuring the selected video clip can realistically fit within the narration timing of the specific part of the voiceover segment being addressed by the feedback.
   - If selecting a still frame from a video, determine the moment (single timestamp) that captures the required visual so that it can be used as a static image.

7. Additional Analysis:
   - Note any additional observations, thoughts or analysis that can help you arrive at the correct output and decision that addresses all of the given human feedback for all the visual(s) of this voiceover segment.

(It is ok for this section to be quite verbose, long and detailed as long as it helps you arrive at the correct output.)

</evaluation_breakdown>

(Based on your above evaluation, provide your output in the following format. Remember, only provide the replacement visuals for the visual(s) that the human feedback refers to for this voiceover segment. Do not include visuals that are not being replaced. Meaning if for example you are revising segment 2, and there are 3 visuals assigned to that segment, and the human feedback applies to 2 of them, you should only provide the replacement visuals for those 2 visuals. Do not include the unchanged visual in your output.)

<replacement_visuals>

<visual>

<visual_id>
(Provide the Visual ID of the current visual that we are replacing, e.g. S2V3)
</visual_id>

<voiceover_part>
(Provide the exact portion of the voiceover text to which this visual was assigned)
</voiceover_part>

<current_visual_url>
(Provide the URL of the current visual that we are replacing, e.g. https://www.youtube.com/embed/dQw4w9WgXc?start=10&end=20)
</current_visual_url>

<replacement_visual_url>
(Provide the URL of the replacement visual that we are selecting for this voiceover part, in one of the following forms:
- Image URL (Exact URL as provided in the image candidates if an image is selected for this part of the voiceover sentence)
- Video URL with start and end timestamps (if a portion of a video clip is selected for this part of the voiceover sentence. Strictly use such example format of video URL with start and end timestamps: e.g. "https://www.youtube.com/embed/dQw4w9WgXc?start=10&end=20")
- Video URL with a single start timestamp (if a still frame extracted from a video clip is selected for this part of the voiceover sentence. Strictly use such example format of video URL with a single start timestamp indicating the frame timestamp: e.g. "https://www.youtube.com/embed/dQw4w9WgXc?start=10"))
</replacement_visual_url>

<visual_instruction>
(Provide a concise description of what is visibly shown in the replacement visual for this voiceover part)
</visual_instruction>

<selection_justification>
(Provide a concise justification for why this replacement visual is the most appropriate for this voiceover part and how it addresses the human feedback)
</selection_justification>

</visual>

Repeat one <visual> block per visual that needs to be replaced based on the human feedback for this voiceover segment.

</replacement_visuals>

</output>

(Remember to strictly use this exact format for the output with the XML tags and structure, and nothing else.)
"""


# Revision prompt to use when we have 1 visual for the whole slide visual assingment strategy
HUMAN_FEEDBACK_REVISION_PROMPT_FOR_ONE_VISUAL_PER_SLIDE = """You are a Graphics Definition Revision Agent specializing in the field of HVAC. Your task is to revise and correct the graphics definition for the given slide when a human reviewer has requested changes to its currently assigned visual, by selecting the most appropriate visual from the provided candidate image and video pools.
You will be given the slide, the course context for reference, the visual currently assigned to this slide, the human-provided feedback describing what they want changed, and a pool of candidate visuals from which to select the most appropriate visual to replace the current visual based on the feedback.

Inputs:
These are the inputs for your revision:

<course_information>
Course name: {course_name}
Target audience: {target_audience}
Topic name: {topic_name}
Subtopic name: {subtopic_name}
</course_information>

<slide_information>
Slide ID: {slide_id}
Slide title: {slide_title}
Slide content: "{slide_chunk}"
</slide_information>

This is the assigned visual for this whole slide:
<current_visual_assigned>
{current_visuals}
</current_visual_assigned>

This is the human-provided feedback requesting the following change for the visual assigned to this slide:
<human_feedback>
{human_feedback}
</human_feedback>

These are the candidate images and videos from which to select the most appropriate visual to replace the failed visual based on the human feedback:

<candidates>

<image_candidates>
{image_candidates}
</image_candidates>

<video_candidates>
{video_candidates}
</video_candidates>

</candidates>

Instructions:
Follow the below evaluation rules to guide your evaluation:

1. Scope and Revision Responsibility
   - Your task is to revise the graphics definition for this slide.
   - Use the whole slide content to understand the intended meaning of the slide.
   - The assigned visual is displayed on screen as the entire slide content is narrated.
   - Identify the specific visual requirement implied by the human feedback.
   - If the human feedback asks for a particular visual subject, object, scene, component, diagram, process, or other concrete visual requirement, treat that requirement as mandatory. You must actively look for a candidate that actually shows that requested visual, not just something broadly related to the same topic.
   - Select replacement visual only from the provided candidate image and video pools.
   - IMPORTANT: Know that we have been allowed to assign only one visual asset for this particular slide. So keep that in mind as you select the replacement visual.

2. Candidate Evaluation and Visual Replacement
   - Carefully review all provided image and video candidates for this slide.
   - Evaluate each candidate against the specific visual requirement described in the human feedback.
   - Select the candidate that most directly and clearly satisfies the human feedback while remaining aligned with the slide content.
   - If the feedback explicitly requests a specific kind of visual, do not choose a generic, adjacent, or loosely related candidate when another candidate actually shows the requested thing more directly.
   - If no candidate fully satisfies the human feedback, select the closest acceptable alternative.
   - Ensure the replacement visual is instructionally clear and effective for the slide.

3. Visual Form and Usage Constraints
   - You may select an image, video clip with timestamps, or a still frame extracted from video, using only the provided candidate visuals.
   - Choose the visual form (image, video clip, or still frame) that most clearly satisfies the human feedback while fitting within the narration timing of the slide content.
   - When you find both a video clip and a still image that equally satisfies the human feedback, prefer using the video clip, since motion can add useful context. This is a guiding preference, not a strict rule - do not prioritize a video clip over an image if the video is only partially relevant, loosely related, or less effective than the still image at supporting the narration and addressing the feedback.
   - When selecting a video clip, identify the exact portion of the video that visually supports the required detail and assign appropriate start and end timestamps. Strictly use such example format of video URL with start and end timestamps: e.g. "https://www.youtube.com/embed/dQw4w9WgXc?start=10&end=20"
   - When selecting a still frame from a video so that it can be used as a static image, output the video URL with a single start timestamp only (no end timestamp). Strictly use such example format of video URL with a single start timestamp indicating the frame timestamp: e.g. "https://www.youtube.com/embed/dQw4w9WgXc?start=10"
   - Respect source-specific constraints when selecting video candidates from the provided candidate video pools.
     - Video candidates listed under:
        "Videos from which you can use video clips (with timestamps) or still frames as images" may be used in any of the following ways:
        - short video clip with start and end timestamps
        - still frame extracted from the video
     - Video candidates listed under:
        "Videos from which you can ONLY use still frames as images (NOT playable video clips with timestamps)" have the following strict constraints:
        - You MUST NOT select them as playable video clips
        - You MUST NOT assign start–end timestamps
        - You MAY ONLY extract still frames and use them as static images
        - When using these videos, the asset MUST be represented as a video URL with a single start timestamp only

Output:

Always provide your output strictly in the following format:

<output>

<evaluation_breakdown>

Use this section as a structured reasoning and scratchpad space for you to address the human feedback and select the most appropriate visual to replace the current visual. Use it to document your observations, reasoning, and decision process. Provide the following sections:

1. Slide Understanding
   - Briefly explain, in your own words, what the slide is about and what it is trying to convey.

2. Human Feedback Interpretation
   - Briefly summarize what the human feedback indicates should be changed about the current visual.
   - Clearly state the specific visual requirement that must be satisfied by the revision.
   - If the feedback contains any explicit visual requirement, name that exact requirement and treat it as a hard constraint for a fully satisfactory replacement.

3. Image Candidates Scan
   - Create a numbered list of all provided image candidates and briefly describe what you see in each of the image candidate.

4. Video Candidates Scan
   - Create a numbered list of all provided video candidates and briefly describe what you see in each of the video candidate.
   - Focus on explaining the visual content of the video, and not what is being spoken in the video. Split the list into two sections: one for videos from which video clips (with timestamps) or still frames as images can be used, and one for videos from which ONLY still frames as images can be used (NOT playable video clips with timestamps).

5. Candidate Fit Analysis
   - Compare the candidates against the human feedback requirement.
   - Identify which candidate most directly satisfies the requirement and why.
   - Explicitly check whether each strong candidate visibly contains the exact thing the human feedback asked for. Topic similarity alone is not enough.
   - If multiple candidates partially match, reason about which one is the closest acceptable match.
   - If no candidate fully satisfies the human feedback, determine the closest acceptable alternative.

6. Video Timestamp / Frame Selection Thinking (only if selecting video as replacement visual)
   - If selecting a playable video clip if you find a relevant one that best satisfies the human feedback requirement:
   - Determine exactly which portion of the video is visually relevant and will satisfy the feedback.
   - Plan the correct start and end timestamps for that portion, ensuring the selected video clip can realistically fit within the narration timing of the slide content.
   - If selecting a still frame from a video, determine the moment (single timestamp) that captures the required visual so that it can be used as a static image.

7. Additional Analysis:
   - Note any additional observations, thoughts or analysis that can help you arrive at the correct output and decision that addresses the given human feedback for the slide.

(It is ok for this section to be quite verbose, long and detailed as long as it helps you arrive at the correct output.)

</evaluation_breakdown>

(Based on your above evaluation, provide your output in the following format)

<replacement_visual>

<visual>

<visual_id>
(Provide the Visual ID of the current visual that we are replacing, e.g. S2V1)
</visual_id>

<voiceover_part>
(Provide the entire slide content text as it is.)
</voiceover_part>

<current_visual_url>
(Provide the URL of the current visual assigned to the slide that we are replacing, e.g. https://www.youtube.com/embed/dQw4w9WgXc?start=10&end=20)
</current_visual_url>

<replacement_visual_url>
(Provide the URL of the replacement visual that we are selecting for this slide, in one of the following forms:
- Image URL (Exact URL as provided in the image candidates if an image is selected for this slide)
- Video URL with start and end timestamps (if a portion of a video clip is selected for this slide. Strictly use such example format of video URL with start and end timestamps: e.g. "https://www.youtube.com/embed/dQw4w9WgXc?start=10&end=20")
- Video URL with a single start timestamp (if a still frame extracted from a video clip is selected for this slide. Strictly use such example format of video URL with a single start timestamp indicating the frame timestamp: e.g. "https://www.youtube.com/embed/dQw4w9WgXc?start=10"))
</replacement_visual_url>

<visual_instruction>
(Provide a concise description of what is visibly shown in the replacement visual for this slide)
</visual_instruction>

<selection_justification>
(Provide a concise justification for why this replacement visual is the most appropriate for this slide and how it addresses the human feedback)
</selection_justification>

</visual>

</replacement_visual>

</output>

(Remember to strictly use this exact format for the output with the XML tags and structure, and nothing else.)
"""


# Review prompt to use when we have flexible or 1 visual per sentence visual assingment strategy
HUMAN_FEEDBACK_SATISFACTION_REVIEW_PROMPT = """You are a Graphics Definition Review Agent specializing in the field of HVAC.

Criterion: Human Feedback Satisfaction

Definition: For the given slide, verify that the replacement visual(s) that were applied in response to the human reviewer's feedback now satisfy what the human reviewer asked for. Your job is to determine whether the replacement(s) adequately address the human-provided feedback.

Inputs:
These are the inputs for your evaluation:

<course_information>
Course name: {course_name}
Target audience: {target_audience}
Topic name: {topic_name}
Subtopic name: {subtopic_name}
</course_information>

<slide_information>
Slide ID: {slide_id}
Slide title: {slide_title}
Slide content: "{slide_chunk}"
</slide_information>

This is the human-provided feedback that was used to generate the replacement(s). The reviewer requested the following change(s):
<human_feedback>
{human_feedback}
</human_feedback>

These are the voiceover segments and the replacement visual(s) that were applied in response to the human feedback (i.e. the current state after revision):
<review_targets>
{review_targets}
</review_targets>

Note: Visual IDs follow the format "S(segment_number)V(visual_number)". For example, "S1V3" means Segment 1, Visual 3, "S2V1" means Segment 2, Visual 1, and so on.

Instructions:
Follow the below evaluation rules to guide your evaluation:

1) Scope
   - Review only the replacement visual(s) that were applied in response to the human feedback.
   - The question to answer is: Does the replacement visual (or set of replacements) satisfy what the human reviewer asked for?
   - You may use the full slide content and the sequence of voiceover segments to understand the whole context.
   - When interpreting the human feedback, consider it in the context of the voiceover and slide content. Do not interpret the feedback in isolation.
   - Each visual asset has a Visual ID (for example, S2V1). Use these IDs when listing any failures.
   - Be fair but strict: PASS only if the replacement(s) clearly and adequately address the human's request. FAIL if the replacement still does not match what was asked, is off-topic, or introduces new issues relative to the feedback.
   - If the human feedback includes any explicit visual requirement, visual subject, object, scene, component, diagram, process, or other concrete thing that should appear, that exact requirement must be visibly satisfied in the replacement. Being generally related to the topic is not enough.

2) What counts as PASS for a segment
   - The replacement visual(s) clearly address what the human feedback requested.
   - A reasonable reviewer would agree that the change requested has been satisfied.
   - The replacement does not contradict the human feedback or introduce a different problem.
   - If the feedback requested a particular visual, that requested visual is clearly and unambiguously visible in the replacement.

3) What counts as FAIL for a segment
   A segment FAILS if any one of the following is true:
   - The replacement does not address what the human asked for.
   - The replacement is still wrong, generic, or off-topic relative to the feedback.
   - The replacement only partially satisfies the feedback and key aspects are still missing.
   - The human feedback asked for a specific visual requirement and the replacement does not clearly show that requirement, only implies it, or shows something merely adjacent or related.
   - The replacement visual is unusable (missing, broken, or non-loadable).

4) Slide-Level Verdict
   - The slide receives a PASS only if all replacement(s) that were made in response to the human feedback satisfy that feedback.
   - If any single replacement FAILS to satisfy the human feedback, the entire slide verdict must be FAIL.
   - Be strict enough to ensure that the human's request is actually met before passing.

5) Failure Reporting Requirements
   For every replacement that still does not satisfy the human feedback, you MUST:
   - Identify the voiceover segment ID
   - Quote the exact voiceover text
   - List the failing Visual ID (the replacement that was applied)
   - Clearly state why the replacement does not satisfy the human feedback
   - Describe what would be required for the replacement to satisfy the human feedback

Output Format:
Always provide your output strictly in the following format:

<evaluation_breakdown>

Use this section as a structured reasoning and scratchpad space to evaluate whether the replacement(s) satisfy the human feedback.

- Human Feedback Summary: State in your own words what the human requested.
- Explicit Visual Requirements: List any concrete visual requirements that the human feedback explicitly asks to see. If none are explicit, say so.
- Review of the replacement visuals: For each segment that was revised, list the Visual IDs and briefly describe what is visibly shown in each replacement visual.
- Satisfaction Analysis: For each replacement, analyze whether it adequately addresses the human feedback. Does it do what the reviewer asked? If the human feedback asked for a specific visual, explicitly state whether that exact requested visual is visibly present.
- Additional Analysis: Note any additional observations that can help you arrive at the correct verdict.

(It is ok for this section to be quite verbose, long and detailed as long as it helps you arrive at the correct output.)

</evaluation_breakdown>

(Based on your above evaluation, provide your output in the following format)

<review>

<verdict>
PASS|FAIL 
</verdict>

(If the slide verdict is FAIL, provide the details of the failed segment(s) in the following format)
<failures>

<failure>
<segment_id>
(Provide the segment number of the failed segment. e.g. SEGMENT 1)
</segment_id>

<vo_text>
(Provide the exact portion of voiceover text for which the replacement does not satisfy the human feedback.)
</vo_text>

<failing_visual_id>
(Provide the Visual ID of the replacement visual that does not satisfy the human feedback. e.g. S1V3)
</failing_visual_id>

<reason>
(Provide the reason why the replacement does not satisfy the human feedback.)
</reason>

<needed_visual>
(Describe what is needed for the replacement to satisfy the human feedback. Prefer words like "visual" rather than "image" since replacements will be selected from image or video pools.)
</needed_visual>

</failure>

Repeat the <failure> block for each failed segment and its corresponding visual id.

</failures>

</review>

(Use this exact XML format given above while providing your output)
"""


# Review prompt to use when we have 1 visual for the whole slide visual assingment strategy
HUMAN_FEEDBACK_SATISFACTION_REVIEW_PROMPT_FOR_ONE_VISUAL_PER_SLIDE = """You are a Graphics Definition Review Agent specializing in the field of HVAC.

Criterion: Human Feedback Satisfaction

Definition: For the given slide, verify that the replacement visual that was applied in response to the human reviewer's feedback now satisfies what the reviewer asked for. Your job is to determine whether the replacement adequately address the human-provided feedback.

Inputs:
These are the inputs for your evaluation:

<course_information>
Course name: {course_name}
Target audience: {target_audience}
Topic name: {topic_name}
Subtopic name: {subtopic_name}
</course_information>

<slide_information>
Slide ID: {slide_id}
Slide title: {slide_title}
Slide content: "{slide_chunk}"
</slide_information>

This is the human-provided feedback that was used to generate the replacement. The reviewer requested the following change(s):
<human_feedback>
{human_feedback}
</human_feedback>

This is the replacement visual that was applied for this whole slide in response to the human feedback (i.e. the current state after revision):
<review_targets>
{review_targets}
</review_targets>

Note: Visual IDs follow the format "S(segment_number)V(visual_number)".

Instructions:
Follow the below evaluation rules to guide your evaluation:

1) Scope
   - Review the single replacement visual that was applied for this slide in response to the human feedback.
   - The question to answer is: Does the replacement visual satisfy what the human reviewer asked for?
   - Use the full slide content to understand context.
   - When interpreting the human feedback, consider it in the context of the slide content. Do not interpret the feedback in isolation.
   - The visual asset has a Visual ID (for example, S1V1). Use this ID when listing any failure.
   - Be fair but strict: PASS only if the replacement clearly and adequately addresses the human's request. FAIL if the replacement still does not match what was asked, is off-topic, or introduces new issues relative to the feedback.
   - IMPORTANT: Know that only one visual asset is assigned for this particular slide. Evaluate whether that single replacement satisfies the human feedback.
   - If the human feedback includes any explicit visual requirement, visual subject, object, scene, component, diagram, process, or other concrete thing that should appear, that exact requirement must be visibly satisfied in the replacement. Being generally related to the topic is not enough.

2) What counts as PASS for the slide
   - The replacement visual clearly addresses what the human feedback requested.
   - A reasonable reviewer would agree that the change requested has been satisfied.
   - The replacement does not contradict the human feedback or introduce a different problem.
   - If the feedback requested a particular visual, that requested visual is clearly and unambiguously visible in the replacement.

3) What counts as FAIL for the slide
   The slide FAILS if any one of the following is true:
   - The replacement does not address what the human asked for.
   - The replacement is still wrong, generic, or off-topic relative to the feedback.
   - The replacement only partially satisfies the feedback and key aspects are still missing.
   - The human feedback asked for a specific visual requirement and the replacement does not clearly show that requirement, only implies it, or shows something merely adjacent or related.
   - The replacement visual is unusable (missing, broken, or non-loadable).

4) Slide-Level Verdict
   - The slide receives a PASS only if the replacement satisfies the human feedback.
   - If the replacement FAILS to satisfy the human feedback, the slide verdict must be FAIL.
   - Be strict enough to ensure that the human's request is actually met before passing.

5) Failure Reporting Requirements
   If your verdict for the slide is FAIL, you MUST:
   - List the Segment ID (e.g. SEGMENT 1 for whole-slide visual)
   - List the failing Visual ID
   - Clearly state why the replacement does not satisfy the human feedback
   - Describe what would be required for the replacement to satisfy the human feedback

Output Format:
Always provide your output strictly in the following format:

<evaluation_breakdown>

Use this section as a structured reasoning and scratchpad space to evaluate whether the replacement satisfies the human feedback.

- Human Feedback Summary: State in your own words what the human requested.
- Explicit Visual Requirements: List any concrete visual requirements that the human feedback explicitly asks to see. If none are explicit, say so.
- Review of the replacement visual: Briefly describe what is visibly shown in the replacement visual.
- Satisfaction Analysis: Analyze whether the replacement adequately addresses the human feedback. If the feedback asked for a specific visual, explicitly state whether that exact requested visual is visibly present.
- Additional Analysis: Note any additional observations that can help you arrive at the correct verdict.

(It is ok for this section to be quite verbose, long and detailed as long as it helps you arrive at the correct output.)

</evaluation_breakdown>

(Based on your above evaluation, provide your output in the following format)

<review>

<verdict>
PASS|FAIL
</verdict>

(If the slide verdict is FAIL, provide the details of the failed segment in the following format)

<failure>

<segment_id>
(Provide the segment number. e.g. SEGMENT 1)
</segment_id>

<vo_text>
(Provide the entire slide content text as it is.)
</vo_text>

<failing_visual_id>
(Provide the Visual ID of the replacement visual. e.g. S1V1)
</failing_visual_id>

<reason>
(Provide the reason why the replacement does not satisfy the human feedback.)
</reason>

<needed_visual>
(Describe what is needed for the replacement to satisfy the human feedback. Prefer words like "visual" rather than "image" since replacements will be selected from image or video pools.)
</needed_visual>

</failure>

</review>

(Use this exact XML format given above while providing your output)
"""


# ---------------------------------------------------------------------------
# A. Parse human feedback from the sheet
# ---------------------------------------------------------------------------


def _format_human_feedback_for_prompt(raw):
    """
    Normalize human feedback so prompts always show 'When VO:' and 'Human Feedback:' labels.

    :param raw: Raw human feedback text
    :return: Normalized feedback string
    """
    
    if not raw or not raw.strip():
        return raw or ""
    normalized = re.sub(r"(?m)^\s*vo:\s*", "When VO: ", raw, flags=re.IGNORECASE)
    normalized = re.sub(r"(?m)^\s*feedback:\s*", "Human Feedback: ", normalized, flags=re.IGNORECASE)
    return normalized


def parse_human_feedback_for_row(human_feedback_text, voiceover_text, final_graphics_definition, visual_assignment_strategy, slide_chunk):
    """
    Parse the human_feedback column into a mapping of segment_num to list of (vo_part, feedback_text).

    :param human_feedback_text: Raw content of the human_feedback column
    :param voiceover_text: The voiceover_segment column content
    :param final_graphics_definition: Current final_graphics_definition
    :param visual_assignment_strategy: Visual assignment strategy
    :param slide_chunk: Full slide content
    :return: Dict mapping segment_num to list of (vo_part, feedback_text)
    """
    
    if not human_feedback_text or human_feedback_text.strip() in ("", "nan"):
        return {}

    segments_map = build_segment_visual_map(
        voiceover_text, final_graphics_definition, visual_assignment_strategy, slide_chunk)

    vo_to_segment: Dict[str, int] = {}
    for segment_num, segment_data in segments_map.items():
        vo = _safe_str(segment_data.get("vo_text", "")).strip().lower()
        if vo:
            vo_to_segment[vo] = segment_num
        for step in segment_data.get("visual_steps", []):
            part = _safe_str(step.get("voiceover_part", "")).strip().lower()
            if part:
                vo_to_segment[part] = segment_num

    blocks = re.split(r"\n\s*\n", human_feedback_text.strip())
    feedback_by_segment: Dict[int, List[Tuple[str, str]]] = {}

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        vo_match = re.search(r"(?:^|\n)\s*(?:When VO:|vo:)\s*(.+?)(?:\n|$)", block, re.IGNORECASE)
        fb_match = re.search(r"(?:^|\n)\s*(?:Human Feedback:|feedback:)\s*(.+)", block, re.IGNORECASE | re.DOTALL)

        vo_part = vo_match.group(1).strip() if vo_match else ""
        fb_text = fb_match.group(1).strip() if fb_match else ""

        # Preserve explicit AI-empty-feedback marker so Reject (Generate with AI)
        # remains actionable without injecting default feedback text.
        if not fb_text:
            continue

        segment_num = _match_vo_to_segment(vo_part, vo_to_segment, segments_map)

        if segment_num is None:
            if len(segments_map) == 1:
                segment_num = next(iter(segments_map))
            else:
                print(f"  WARNING: Could not match human feedback vo line to a segment: \"{vo_part[:80]}\"")
                continue

        if segment_num not in feedback_by_segment:
            feedback_by_segment[segment_num] = []
        feedback_by_segment[segment_num].append((vo_part, fb_text))

    return feedback_by_segment


def _match_vo_to_segment(vo_line, vo_to_segment, segments_map):
    """
    Best-effort match a vo line from human feedback to a segment number.

    :param vo_line: The voiceover line from human feedback
    :param vo_to_segment: A dictionary mapping voiceover lines to segment numbers
    :param segments_map: A dictionary mapping segment numbers to segment data
    :return: The segment number if found, None otherwise
    """
    
    if not vo_line:
        return None
    vo_lower = vo_line.strip().lower()

    if vo_lower in vo_to_segment:
        return vo_to_segment[vo_lower]

    for key, seg_num in vo_to_segment.items():
        if vo_lower in key or key in vo_lower:
            return seg_num

    for segment_num, segment_data in segments_map.items():
        for step in segment_data.get("visual_steps", []):
            part = _safe_str(step.get("voiceover_part", "")).strip().lower()
            if part and (vo_lower in part or part in vo_lower):
                return segment_num

    return None


def _normalize_vo_for_match(vo_text):
    """
    Normalize VO text for matching so feedback and definition variants match.

    :param vo_text: Voiceover text to normalize
    :return: Normalized string (strip, remove trailing punctuation, collapse spaces)
    """
    
    if not vo_text:
        return ""
    s = vo_text.strip()
    while s and s[-1] in ".?!":
        s = s[:-1].rstrip()
    s = re.sub(r"[,;:]+", " ", s)
    return " ".join(s.split())


# ---------------------------------------------------------------------------
# B. Human-feedback revision function
# ---------------------------------------------------------------------------

def revise_segment_with_human_feedback(course_name, target_audience, topic_name, subtopic_name, slide_title, slide_chunk, vo_text, current_visuals, human_feedback, image_pool_text, video_pool_filtered_text, drive_results_text, web_results_text, segment_num, drive, llm, visual_assignment_strategy="Flexible, let the agent decide", video_pool_text="", video_pool_other_channels_text="", candidate_mode="all"):
    """
    Revise segment visuals based on human feedback using human-feedback revision prompts.

    :param course_name: Course name
    :param target_audience: Target audience
    :param topic_name: Topic name
    :param subtopic_name: Subtopic name
    :param slide_title: Slide title
    :param slide_chunk: Slide chunk text
    :param vo_text: Voiceover text for the segment
    :param current_visuals: Current visuals description
    :param human_feedback: Human feedback text
    :param image_pool_text: Image pool text
    :param video_pool_filtered_text: Video pool filtered text
    :param drive_results_text: Drive results text
    :param web_results_text: Web results text
    :param segment_num: Segment number
    :param drive: Google Drive instance
    :param llm: LLM model name
    :param visual_assignment_strategy: Visual assignment strategy
    :param video_pool_text: Video pool text
    :param video_pool_other_channels_text: Video pool other channels text
    :param candidate_mode: "all" or "drive_hvac" (drive images + timestamped embed videos only)
    :return: Replacement XML string or None if revision fails
    """
    
    print(f"  Revising segment {segment_num} with human feedback...")
    print(f"    Human feedback: {human_feedback[:120]}...")

    images, videos, frame_videos = build_candidates_for_segment(
        image_pool_text,
        video_pool_filtered_text,
        drive_results_text,
        web_results_text,
        segment_num,
        video_pool_text=video_pool_text,
        video_pool_other_channels_text=video_pool_other_channels_text,
    )

    if str(candidate_mode).strip().lower() == "drive_hvac":
        images = [c for c in images if _is_drive_url(c.get("url", ""))]
        videos = [c for c in videos if _is_embed_with_start_end(c.get("url", ""))]
        frame_videos = []
        print(
            f"  Restricted candidate mode for segment {segment_num}: "
            f"{len(images)} drive image(s), {len(videos)} timestamped video clip(s)"
        )
    candidate_map = {
        c["id"]: c["url"]
        for c in images + videos + frame_videos
        if c.get("id") and c.get("url")
    }
    image_candidates_text = build_candidate_text(images)
    video_candidates_text = build_video_candidates_text(videos, frame_videos)

    current_visual_steps = []
    for line in current_visuals.split("\n"):
        line = line.strip()
        if not line:
            continue
        if " | Visual assigned: " in line:
            left, asset_url = line.split(" | Visual assigned: ", 1)
            left, asset_url = left.strip(), asset_url.strip()
            vid_part = left.split(" | ")[0].strip() if " | " in left else left
            vo_match = re.search(r'When VO:\s*"([^"]*)"', left) or re.search(r'When VO:\s*(.+)', left)
            voiceover_part = (vo_match.group(1).strip() if vo_match else "").strip()
            current_visual_steps.append({
                "visual_id": vid_part,
                "asset_url": asset_url,
                "voiceover_part": voiceover_part,
            })

    visual_assignment_strategy = str(visual_assignment_strategy).strip()
    if visual_assignment_strategy == "1 Visual for the whole Slide":
        prompt_template = HUMAN_FEEDBACK_REVISION_PROMPT_FOR_ONE_VISUAL_PER_SLIDE
        actual_vo_text = slide_chunk
        split_tag = "</current_visual_assigned>"
    else:
        prompt_template = HUMAN_FEEDBACK_REVISION_PROMPT
        actual_vo_text = vo_text
        split_tag = "</current_visuals_assigned>"

    full_prompt = prompt_template.format(
        course_name=course_name,
        target_audience=target_audience,
        topic_name=topic_name,
        subtopic_name=subtopic_name,
        slide_id=f"SLIDE_{segment_num}",
        slide_title=slide_title,
        slide_chunk=slide_chunk,
        vo_text=actual_vo_text,
        current_visuals=current_visuals,
        human_feedback=human_feedback,
        image_candidates=image_candidates_text,
        video_candidates=video_candidates_text,
    )

    # # Print the fully formatted text prompt for debugging/reproducibility.
    # print(f"\n{'=' * 80}")
    # print(f"HUMAN FEEDBACK REVISION (Segment {segment_num}) [Strategy: {visual_assignment_strategy}]")
    # print(f"{'=' * 80}")
    # print(f"\n--- TEXT PROMPT (REVISION) ---")
    # print(full_prompt)
    # print(f"--- END TEXT PROMPT ---\n")

    parts: List[types.Part] = []

    split1 = full_prompt.split(split_tag, 1)
    if len(split1) == 2:
        parts.append(types.Part(text=split1[0] + split_tag))
        for step in current_visual_steps:
            vo_part = step.get("voiceover_part", "") or ""
            label = f"When VO: {vo_part}\nCurrent visual assigned({step['visual_id']}): {step['asset_url']}"
            parts.append(types.Part(text=label))
            visual_part = build_visual_part_only(step["asset_url"], drive)
            if visual_part:
                parts.append(visual_part)
            else:
                parts.append(types.Part(text=f"[Current visual {step['visual_id']} could not be loaded]"))
        remaining = split1[1]
    else:
        remaining = full_prompt

    candidate_num = 1
    split2 = remaining.split("</image_candidates>", 1)
    if len(split2) == 2:
        parts.append(types.Part(text=split2[0] + "</image_candidates>"))
        for candidate in images:
            label = f"Image Candidate {candidate_num} ({candidate['id']}): {candidate['title']} | URL: {candidate['url']}"
            parts.append(types.Part(text=label))
            visual_part = build_visual_part_only(candidate["url"], drive)
            if visual_part:
                parts.append(visual_part)
            candidate_num += 1
        remaining = split2[1]
    else:
        remaining = split2[0] if split2 else remaining

    split3 = remaining.split("</video_candidates>", 1)
    if len(split3) == 2:
        parts.append(types.Part(text=split3[0] + "</video_candidates>"))
        total_video_candidates = len(videos) + len(frame_videos)
        for candidate in videos:
            video_url = candidate.get("url", "")
            clip_url, start_seconds, end_seconds = parse_video_url_timestamps(video_url)
            if clip_url:
                label = (
                    f"\n--- Video {candidate_num} of {total_video_candidates} ---\n"
                    f"ID: {candidate['id']}\n"
                    f"Usage: Can be used as video clip (any part with start/end) OR as still frame\n"
                    f"URL: {video_url}\n"
                )
                parts.append(types.Part(text=label))
                video_part = build_video_part(clip_url, start_seconds, end_seconds)
                if video_part:
                    parts.append(video_part)
                candidate_num += 1
            else:
                label = (
                    f"\n--- Video {candidate_num} of {total_video_candidates} ---\n"
                    f"ID: {candidate['id']}\n"
                    f"Usage: Can be used as video clip OR as still frame\n"
                    f"URL: {video_url}\n"
                )
                parts.append(types.Part(text=label))
                visual_part = build_visual_part_only(video_url, drive)
                if visual_part:
                    parts.append(visual_part)
                candidate_num += 1

        for candidate in frame_videos:
            video_url = candidate.get("url", "")
            embed_url = convert_watch_url_to_embed_url(video_url)
            if embed_url:
                label = (
                    f"\n--- Video {candidate_num} of {total_video_candidates} ---\n"
                    f"ID: {candidate['id']}\n"
                    f"Usage: Can be used ONLY as still frames (NOT playable clips)\n"
                    f"URL: {video_url}\n"
                )
                parts.append(types.Part(text=label))
                video_part = build_video_part(embed_url, start_seconds=None, end_seconds=None)
                if video_part:
                    parts.append(video_part)
                candidate_num += 1
            else:
                label = (
                    f"\n--- Video {candidate_num} of {total_video_candidates} ---\n"
                    f"ID: {candidate['id']}\n"
                    f"Usage: Can be used ONLY as still frames (NOT playable clips)\n"
                    f"URL: {video_url}\n"
                )
                parts.append(types.Part(text=label))
                visual_part = build_visual_part_only(video_url, drive)
                if visual_part:
                    parts.append(visual_part)
                candidate_num += 1

        parts.append(types.Part(text=split3[1]))
    else:
        parts.append(types.Part(text=remaining))

    print(
        f"  Loaded {len(parts)} part(s) for human-feedback revision "
        f"({len(current_visual_steps)} current visual(s), {len(images)} image(s), "
        f"{len(videos) + len(frame_videos)} video(s))"
    )

    # --- Print MULTIMODAL PARTS only (labels + inline media; keep separate from text prompt) ---
    # print(f"\n--- MULTIMODAL PARTS (REVISION) — labels and inline media only ---")
    # for i, part in enumerate(parts, 1):
    #     text = getattr(part, "text", None) if part else None
    #     if text:
    #         # Only print label-style parts here (skip long prompt chunks)
    #         is_label = (
    #             text.strip().startswith("When VO:")
    #             or text.strip().startswith("Current Assigned Visual ")
    #             or "Image Candidate " in text and "): " in text
    #             or "Video Candidate " in text and "): " in text
    #         )
    #         if is_label:
    #             print(f"[Part {i}] {text.strip()}")
    #         # (prompt chunks are not reprinted; they are in TEXT PROMPT section above)
    #     else:
    #         print(f"[Part {i}] <inline media (image or video)>")
    # print("--- END MULTIMODAL PARTS ---\n")

    print(f"  Invoking Gemini multimodal human-feedback revision (model: {llm})...")
    response_text, _ = invoke_gemini_multimodal(parts, llm=llm)
    print(f"\nHuman-feedback revision response (SEGMENT {segment_num}):\n{response_text}\n")

    replacement_visuals = _extract_tag(response_text, "replacement_visuals")
    if not replacement_visuals:
        replacement_visuals = _extract_tag(response_text, "replacement_visual")
    if not replacement_visuals:
        print(f"  WARNING: No replacement visuals found in response for segment {segment_num}")
        return None

    resolved = resolve_asset_urls_in_definition(replacement_visuals, candidate_map)
    if resolved != replacement_visuals:
        print(f"  Resolved asset IDs to URLs for segment {segment_num}")

    print(f"  Successfully revised segment {segment_num} with human feedback")
    return resolved


# ---------------------------------------------------------------------------
# C. Human-feedback satisfaction review function
# ---------------------------------------------------------------------------

def _build_review_targets_with_original_and_replacement(segments_map, original_visuals_by_segment, segment_nums, slide_id, slide_chunk, visual_assignment_strategy, vo_parts_with_feedback_by_segment):
    """
    Build review_targets string with original and replacement visual only for VO parts that had feedback and were replaced.

    :param segments_map: Map of segment_num to segment data
    :param original_visuals_by_segment: Map of segment_num to list of original visual steps
    :param segment_nums: List of segment numbers
    :param slide_id: Slide identifier
    :param slide_chunk: Full slide content
    :param visual_assignment_strategy: Visual assignment strategy
    :param vo_parts_with_feedback_by_segment: Map of segment_num to set of normalized VO parts with feedback
    :return: Formatted review_targets string
    """
    
    lines: List[str] = []
    for segment_num in segment_nums:
        segment = segments_map.get(segment_num, {})
        replacement_steps = segment.get("visual_steps", [])
        original_steps = original_visuals_by_segment.get(segment_num, [])
        vo_parts_with_feedback = vo_parts_with_feedback_by_segment.get(segment_num, set())

        if not replacement_steps:
            continue

        segment_lines: List[str] = []
        for i, rep_step in enumerate(replacement_steps):
            visual_id = rep_step.get("visual_id", "")
            vo_part = rep_step.get("voiceover_part", "")
            rep_asset = rep_step.get("asset", "")
            orig_step = original_steps[i] if i < len(original_steps) else None
            orig_asset = orig_step.get("asset", "") if orig_step else ""

            if _normalize_vo_for_match(vo_part) not in vo_parts_with_feedback or orig_asset == rep_asset:
                continue

            segment_lines.append(f"When VO: \"{vo_part}\"")
            segment_lines.append(f"Original Visual Assigned for which the feedback was given: {visual_id} | Asset: {orig_asset}")
            segment_lines.append(f"Replacement Visual Assigned based on the feedback: {visual_id} | Asset: {rep_asset}")
            segment_lines.append("")

        if segment_lines:
            lines.append(f"{slide_id} SEGMENT {segment_num}")
            lines.extend(segment_lines)
    return "\n".join(lines).strip() if lines else ""


def _build_assets_for_satisfaction_review(segments_map, original_visuals_by_segment, segment_nums, drive, slide_chunk, visual_assignment_strategy, vo_parts_with_feedback_by_segment):
    """
    Build multimodal parts for satisfaction review (original and replacement visuals for VO parts that had feedback and were replaced).

    :param segments_map: Map of segment_num to segment data
    :param original_visuals_by_segment: Map of segment_num to list of original visual steps
    :param segment_nums: List of segment numbers
    :param drive: Google Drive instance
    :param slide_chunk: Full slide content
    :param visual_assignment_strategy: Visual assignment strategy
    :param vo_parts_with_feedback_by_segment: Map of segment_num to set of normalized VO parts with feedback
    :return: List of multimodal parts
    """
    
    parts: List = []
    for segment_num in segment_nums:
        segment = segments_map.get(segment_num, {})
        replacement_steps = segment.get("visual_steps", [])
        original_steps = original_visuals_by_segment.get(segment_num, [])
        vo_parts_with_feedback = vo_parts_with_feedback_by_segment.get(segment_num, set())

        for i, rep_step in enumerate(replacement_steps):
            visual_id = rep_step.get("visual_id", "")
            vo_part = rep_step.get("voiceover_part", "")
            rep_asset = rep_step.get("asset", "")
            orig_step = original_steps[i] if i < len(original_steps) else None
            orig_asset = orig_step.get("asset", "") if orig_step else ""

            # Only include where feedback was given for this VO part and a replacement was actually done.
            # Match using normalized VO (so "text." and "text" from feedback both match).
            if _normalize_vo_for_match(vo_part) not in vo_parts_with_feedback or orig_asset == rep_asset:
                continue

            parts.append(types.Part(text=f"When VO: \"{vo_part}\""))
            parts.append(types.Part(text=f"Original Visual Assigned for which the feedback was given: {visual_id} | Asset: {orig_asset}"))
            if orig_asset:
                visual_part = build_visual_part_only(orig_asset, drive)
                if visual_part:
                    parts.append(visual_part)
                else:
                    parts.append(types.Part(text=f"[Original visual {visual_id} could not be loaded]"))
            else:
                parts.append(types.Part(text=f"[Original visual {visual_id}: no asset]"))

            parts.append(types.Part(text=f"Replacement Visual Assigned based on the feedback: {visual_id} | Asset: {rep_asset}"))
            if rep_asset:
                visual_part = build_visual_part_only(rep_asset, drive)
                if visual_part:
                    parts.append(visual_part)
                else:
                    parts.append(types.Part(text=f"[Replacement visual {visual_id} could not be loaded]"))
            else:
                parts.append(types.Part(text=f"[Replacement visual {visual_id}: no asset]"))
    return parts


def review_human_feedback_satisfaction(course_name, target_audience, topic_name, subtopic_name, slide_id, slide_title, slide_chunk, human_feedback, segments_map, segment_nums, drive, llm, visual_assignment_strategy="Flexible, let the agent decide", original_visuals_by_segment=None, vo_parts_with_feedback_by_segment=None):
    """
    Review whether replacement visuals satisfy the human feedback.

    :param course_name: Course name
    :param target_audience: Target audience
    :param topic_name: Topic name
    :param subtopic_name: Subtopic name
    :param slide_id: Slide identifier
    :param slide_title: Slide title
    :param slide_chunk: Slide chunk text
    :param human_feedback: Human feedback text
    :param segments_map: Map of segment_num to segment data
    :param segment_nums: List of segment numbers to review
    :param drive: Google Drive instance
    :param llm: LLM model name
    :param visual_assignment_strategy: Visual assignment strategy
    :param original_visuals_by_segment: Optional; when set, review shows original and replacement for VO parts with feedback
    :param vo_parts_with_feedback_by_segment: Optional; set of normalized VO parts with feedback per segment
    :return: Tuple of (verdict, failures, conversation_history)
    """
    
    print(f"  Reviewing human-feedback satisfaction for {slide_id}...")

    if visual_assignment_strategy == "1 Visual for the whole Slide":
        prompt_template = HUMAN_FEEDBACK_SATISFACTION_REVIEW_PROMPT_FOR_ONE_VISUAL_PER_SLIDE
        if original_visuals_by_segment and vo_parts_with_feedback_by_segment:
            review_targets = _build_review_targets_with_original_and_replacement(
                segments_map, original_visuals_by_segment, segment_nums, slide_id, slide_chunk, visual_assignment_strategy,
                vo_parts_with_feedback_by_segment,
            )
        else:
            review_targets = build_review_targets_for_whole_slide(segments_map, slide_id, slide_chunk)
    else:
        prompt_template = HUMAN_FEEDBACK_SATISFACTION_REVIEW_PROMPT
        if original_visuals_by_segment and vo_parts_with_feedback_by_segment:
            review_targets = _build_review_targets_with_original_and_replacement(
                segments_map, original_visuals_by_segment, segment_nums, slide_id, slide_chunk, visual_assignment_strategy,
                vo_parts_with_feedback_by_segment,
            )
        else:
            review_targets = build_review_targets(segments_map, segment_nums, slide_id)

    if original_visuals_by_segment and vo_parts_with_feedback_by_segment:
        asset_parts = _build_assets_for_satisfaction_review(
            segments_map, original_visuals_by_segment, segment_nums, drive, slide_chunk, visual_assignment_strategy,
            vo_parts_with_feedback_by_segment,
        )
    else:
        asset_parts = build_assets_for_segments(segments_map, segment_nums, drive)
    print(f"  Loaded {len(asset_parts)} asset part(s) for satisfaction review")

    prompt = prompt_template.format(
        course_name=course_name,
        target_audience=target_audience,
        topic_name=topic_name,
        subtopic_name=subtopic_name,
        slide_id=slide_id,
        slide_title=slide_title,
        slide_chunk=slide_chunk,
        human_feedback=human_feedback,
        review_targets=review_targets,
    )

    # --- Print TEXT PROMPT only (separate from multimodal parts) ---
    # print(f"\n{'=' * 80}")
    # print(f"HUMAN FEEDBACK SATISFACTION REVIEW ({slide_id}) [Strategy: {visual_assignment_strategy}]")
    # print(f"{'=' * 80}")
    # print(f"\n--- TEXT PROMPT (SATISFACTION REVIEW) ---")
    # print(prompt)
    # print(f"--- END TEXT PROMPT ---\n")

    # --- Print MULTIMODAL PARTS only (asset labels + inline media; keep separate from text prompt) ---
    # print(f"--- MULTIMODAL PARTS (SATISFACTION REVIEW) — asset labels and inline media only ---")
    # for i, part in enumerate(asset_parts, 1):
    #     text = getattr(part, "text", None) if part else None
    #     if text:
    #         print(f"[Part {i}] {text.strip()}")
    #     else:
    #         print(f"[Part {i}] <inline media (image or video)>")
    # print("--- END MULTIMODAL PARTS ---\n")

    print(f"  Invoking Gemini multimodal satisfaction review (model: {llm})...")
    response_text, conversation_history = invoke_gemini_multimodal(
        asset_parts + [types.Part(text=prompt)],
        llm=llm,
    )
    print(f"\nSatisfaction review response ({slide_id}):\n{response_text}\n")

    verdict, failures = parse_review_response(response_text)
    print(f"  Verdict: {verdict}, Failures: {len(failures)}")
    if failures:
        for f in failures:
            print(f"    - {f.get('segment_id', '?')}: {f.get('reason', '')[:100]}")
    return verdict, failures, conversation_history


def review_human_feedback_satisfaction_followup(slide_id, human_feedback, previous_failures_text, visual_ids_by_segment, old_asset_urls_by_visual_id, segments_map, segment_nums, drive, llm, conversation_history, visual_assignment_strategy="Flexible, let the agent decide", slide_chunk=""):
    """
    Follow-up satisfaction review in the same conversation after regeneration.

    :param slide_id: Slide identifier
    :param human_feedback: Human feedback text
    :param previous_failures_text: Formatted text of previous failures for the prompt
    :param visual_ids_by_segment: Map of segment_num to list of replaced visual IDs
    :param old_asset_urls_by_visual_id: Map of visual_id to old asset URL before replacement
    :param segments_map: Map of segment_num to segment data
    :param segment_nums: List of segment numbers
    :param drive: Google Drive instance
    :param llm: LLM model name
    :param conversation_history: Previous conversation history
    :param visual_assignment_strategy: Visual assignment strategy
    :param slide_chunk: Full slide content
    :return: Tuple of (verdict, failures, updated_conversation_history)
    """
    
    print(f"  Follow-up satisfaction review for {slide_id} (regeneration)...")

    if visual_assignment_strategy == "1 Visual for the whole Slide":
        review_targets = build_review_targets_for_whole_slide(segments_map, slide_id, slide_chunk)
    else:
        review_targets = build_review_targets(segments_map, segment_nums, slide_id)

    # Build multimodal parts that show BOTH old failed visuals and new replacement visuals.
    asset_parts: List[types.Part] = []

    # First, add old failed visuals with clear labeling.
    for segment_num, visual_ids in visual_ids_by_segment.items():
        for visual_id in visual_ids:
            old_asset_url = old_asset_urls_by_visual_id.get(visual_id)
            if not old_asset_url:
                continue
            print(f"    Loading OLD failed visual {visual_id}: {old_asset_url[:80]}...")
            asset_parts.append(
                types.Part(
                    text=(
                        f"OLD FAILED VISUAL - Visual ID: {visual_id} | "
                        f"Original URL: {old_asset_url}"
                    )
                )
            )
            old_visual_part = build_visual_part_only(old_asset_url, drive)
            if old_visual_part:
                asset_parts.append(old_visual_part)
            else:
                asset_parts.append(
                    types.Part(
                        text=f"[OLD FAILED VISUAL {visual_id} could not be loaded]"
                    )
                )

    # Then, add new replacement visuals with clear labeling.
    for segment_num, visual_ids in visual_ids_by_segment.items():
        segment = segments_map.get(segment_num)
        if not segment:
            continue
        for step in segment.get("visual_steps", []):
            visual_id = step.get("visual_id", "")
            if visual_id not in visual_ids:
                continue
            asset_url = step.get("asset")
            if not asset_url:
                continue
            print(f"    Loading NEW replacement visual {visual_id}: {asset_url[:80]}...")
            asset_parts.append(
                types.Part(
                    text=(
                        f"NEW REPLACEMENT VISUAL - Visual ID: {visual_id} | "
                        f"Replacement URL: {asset_url}"
                    )
                )
            )
            new_visual_part = build_visual_part_only(asset_url, drive)
            if new_visual_part:
                asset_parts.append(new_visual_part)
            else:
                asset_parts.append(
                    types.Part(
                        text=(
                            f"[NEW REPLACEMENT VISUAL {visual_id} could not be loaded]"
                        )
                    )
                )

    old_count = len(
        [
            p
            for p in asset_parts
            if getattr(p, "text", None) and "OLD FAILED VISUAL" in p.text
        ]
    )
    new_count = len(
        [
            p
            for p in asset_parts
            if getattr(p, "text", None) and "NEW REPLACEMENT VISUAL" in p.text
        ]
    )
    print(
        "  Loaded "
        f"{len(asset_parts)} asset part(s) for follow-up satisfaction review "
        f"({old_count} old visuals, {new_count} new visuals)"
    )

    followup_prompt = f"""Based on your previous satisfaction review feedback, I have revised the visuals for the following segments. Please review the revised visuals again to determine if they now satisfy the human feedback.

This is the human feedback that must be satisfied:
<human_feedback>
{human_feedback}
</human_feedback>

This was your previous feedback based on the human feedback and the previous visuals:
<previous_feedback>
{previous_failures_text}
</previous_feedback>

These are the voiceover segments and their updated visuals that were revised in response to the previous feedback:
<review_targets>
{review_targets}
</review_targets>

You will see both the OLD FAILED visuals and the NEW REPLACEMENT visuals for each visual that was regenerated.

Please review the NEW REPLACEMENT visuals using the same criteria as before. Compare them to what the human feedback requested. Output your review in the same format as before:
- <verdict>PASS|FAIL</verdict>
- If FAIL, provide <failure> blocks for any visuals that still fail to satisfy the human feedback

Be very strict in your evaluation. Do not give a PASS verdict if any visual still fails to satisfy the human feedback even after multiple rounds of revisions. The verdict of "PASS" should only be assigned after you are completely satisfied that the revised visual(s) satisfy the human reviewer's request.
If the human feedback asked for a specific visual subject, object, scene, component, diagram, process, or other concrete thing to appear, PASS only if that exact requested thing is clearly visible in the NEW REPLACEMENT visual. A visual that is merely topically related, vaguely suggestive, or only partially aligned must still be marked FAIL.

Remember: Strictly use the same output format as before while reviewing the NEW REPLACEMENT visuals and providing your output. Do not provide any additional text or commentary.
"""

    # print(f"\n{'=' * 80}")
    # print(f"HUMAN FEEDBACK SATISFACTION FOLLOW-UP REVIEW ({slide_id}) [Strategy: {visual_assignment_strategy}]")
    # print(f"{'=' * 80}")
    # print(f"\n--- TEXT PROMPT (FOLLOW-UP SATISFACTION REVIEW) ---")
    # print(followup_prompt)
    # print(f"--- END TEXT PROMPT ---\n")

    # print(f"--- MULTIMODAL PARTS (FOLLOW-UP SATISFACTION REVIEW) ---")
    # for i, part in enumerate(asset_parts, 1):
    #     text = getattr(part, "text", None) if part else None
    #     if text:
    #         print(f"[Part {i}] {text.strip()}")
    #     else:
    #         print(f"[Part {i}] <inline media (image or video)>")
    # print("--- END MULTIMODAL PARTS ---\n")

    print(f"  Invoking Gemini multimodal follow-up satisfaction review (model: {llm})...")
    response_text, updated_history = invoke_gemini_multimodal(
        asset_parts + [types.Part(text=followup_prompt)],
        llm=llm,
        conversation_history=conversation_history,
    )
    print(f"\nFollow-up satisfaction review response ({slide_id}):\n{response_text}\n")

    verdict, failures = parse_review_response(response_text)
    print(f"  Verdict: {verdict}, Failures: {len(failures)}")
    if failures:
        for f in failures:
            print(f"    - {f.get('segment_id', '?')}: {f.get('reason', '')[:100]}")
    return verdict, failures, updated_history


# ---------------------------------------------------------------------------
# D. Per-row orchestration
# ---------------------------------------------------------------------------

def _hf_status_head_for_skip(status: str) -> str:
    """Leading token before '|' so PASS | AI_UPLOAD_FAILED: ... still skips batch re-runs."""
    s = _safe_str(status).strip()
    if "|" in s:
        return s.split("|", 1)[0].strip().upper()
    return s.strip().upper()


def _hf_row_should_skip_in_batch(status: str) -> bool:
    # Rows saved as "PASS | AI_UPLOAD_FAILED: ..." still skip re-runs; clear the status cell to retry the row.
    return _hf_status_head_for_skip(status) in ("DONE", "PASS")


def _compose_hf_status_with_ai_errors(base: str, ai_errors: List[str]) -> str:
    if not ai_errors:
        return _safe_str(base).strip()
    detail = "; ".join(ai_errors)
    if len(detail) > 4500:
        detail = detail[:4497] + "..."
    base_s = _safe_str(base).strip()
    if base_s.upper() == "ERROR":
        return f"ERROR: {detail}"
    return f"{base_s} | AI_UPLOAD_FAILED: {detail}"


@traceable(
    metadata={
        "agent_name": "graphics_definition_v2",
        "step_name": "Human Feedback Review Agent",
        "function_name": "process_human_feedback_row",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous"),
    }
)
def process_human_feedback_row(
    row_index,
    df,
    course_name,
    target_audience,
    drive,
    llm,
    ws,
    use_only_drive_and_hvac=False,
    human_feedback_column="human_feedback",
    human_feedback_status_column="human_feedback_status",
    human_feedback_revision_tracking_column="human_feedback_revision_tracking",
    human_review_actions_column="human_review_actions",
):
    """
    Process a single row's full human-feedback workflow: revise, review, optional regen loop, then write status and tracking to df.

    :param row_index: Row index in the dataframe
    :param df: Dataframe (shared; each thread only modifies its own row)
    :param course_name: Course name
    :param target_audience: Target audience
    :param drive: Google Drive instance
    :param llm: LLM model name
    :param ws: Worksheet object or None
    :param use_only_drive_and_hvac: If True, skip web search and other-channels video search during regeneration.
    :return: None
    """
    
    try:
        row = df.loc[row_index]
        slide_title = _safe_str(row.get("Slide Chunk Title", ""))
        slide_chunk = _safe_str(row.get("Slide Chunk", ""))
        topic_name = _safe_str(row.get("Topic", ""))
        subtopic_name = _safe_str(row.get("Subtopic", ""))
        voiceover_text = _safe_str(row.get("voiceover_segment", ""))
        final_graphics_definition = _safe_str(row.get("final_graphics_definition", ""))
        human_feedback_raw = _safe_str(row.get(human_feedback_column, ""))
        tracking_raw = _safe_str(row.get(human_feedback_revision_tracking_column, ""))
        actions_map, segment_mode_map = _parse_actions_payload(row.get(human_review_actions_column, ""))
        round_index = 0
        m_round = re.search(r"_(\d+)$", str(human_feedback_column))
        if m_round:
            try:
                round_index = int(m_round.group(1))
            except Exception:
                round_index = 0
        slide_id = f"SLIDE_{row_index + 1}"

        visual_assignment_strategy = str(row.get("Visual Assignment Strategy", "Flexible, let the agent decide")).strip()
        if not visual_assignment_strategy or visual_assignment_strategy == "nan":
            visual_assignment_strategy = "Flexible, let the agent decide"

        print("\n" + "=" * 80)
        print(f"[HUMAN FEEDBACK] Processing row {row_index + 1} — {slide_title}")
        print("=" * 80)

        segments_map = build_segment_visual_map(
            voiceover_text, final_graphics_definition, visual_assignment_strategy, slide_chunk
        )
        feedback_by_segment = parse_human_feedback_for_row(
            human_feedback_raw, voiceover_text, final_graphics_definition,
            visual_assignment_strategy, slide_chunk,
        )
        if not feedback_by_segment:
            print("  No actionable feedback parsed; skipping row.")
            df.at[row_index, human_feedback_status_column] = "PASS"
            if ws is not None:
                with _sheet_lock:
                    save_to_sheet(ws, df)
            return

        print(f"  Parsed human feedback for {len(feedback_by_segment)} segment(s): {list(feedback_by_segment.keys())}")

        reviewed_segment_nums = list(feedback_by_segment.keys())
        # Only track visuals whose VO had human feedback (not every visual in the segment)
        hf_revision_tracking = _initialize_human_feedback_revision_tracking(
            segments_map, feedback_by_segment, existing_tracking_text=tracking_raw
        )

        original_visuals_by_segment: Dict[int, List[dict]] = {}
        vo_parts_with_feedback_by_segment: Dict[int, Set[str]] = {}
        for seg_num, vo_fb_pairs in feedback_by_segment.items():
            seg = segments_map.get(seg_num, {})
            original_visuals_by_segment[seg_num] = [
                {"visual_id": s.get("visual_id"), "voiceover_part": s.get("voiceover_part"), "asset": s.get("asset")}
                for s in seg.get("visual_steps", [])
            ]
            vo_parts_with_feedback_by_segment[seg_num] = {_normalize_vo_for_match(vo_part) for vo_part, _ in vo_fb_pairs}

        # --- REVISE (single loop) ---
        print("\n[STEP 1] Revising visuals based on human feedback...")
        updated_segments: Dict[int, str] = {}
        ai_generated_urls: Set[str] = set()
        ai_generation_errors: List[str] = []
        # Segments/VO parts that should go through satisfaction review.
        # AI-generated replacements are intentionally excluded from this review path.
        review_segment_nums: List[int] = []
        review_vo_parts_with_feedback_by_segment: Dict[int, Set[str]] = {}

        for segment_num, vo_fb_pairs in feedback_by_segment.items():
            segment = segments_map.get(segment_num, {})
            vo_text = _safe_str(segment.get("vo_text", ""))
            if visual_assignment_strategy == "1 Visual for the whole Slide" and segment_num == 1:
                vo_text = slide_chunk

            current_visuals = "\n".join([
                f"{step.get('visual_id')} | When VO: \"{step.get('voiceover_part', '')}\" | Visual assigned: {step.get('asset', '')}"
                for step in segment.get("visual_steps", [])
            ])

            ai_visual_blocks: List[str] = []
            non_ai_vo_fb_pairs: List[Tuple[str, str]] = []
            segment_steps = segment.get("visual_steps", [])

            for vo_part, fb_text in vo_fb_pairs:
                matched_step = None
                norm_vo = _normalize_vo_for_match(vo_part)
                for step in segment_steps:
                    if _normalize_vo_for_match(step.get("voiceover_part", "")) == norm_vo:
                        matched_step = step
                        break
                if matched_step is None:
                    non_ai_vo_fb_pairs.append((vo_part, fb_text))
                    continue

                visual_id = _safe_str(matched_step.get("visual_id", "")).strip()
                action_entry = actions_map.get(visual_id, {}) if visual_id else {}
                action_value = _safe_str(action_entry.get("action", "")).strip().lower()

                if action_value == "reject_ai":
                    raw_feedback = _safe_str(fb_text).strip()
                    effective_feedback = "" if raw_feedback == AI_NO_FEEDBACK_MARKER else raw_feedback
                    ai_url = None
                    ai_err = None
                    is_option2_regen = _safe_str(action_entry.get("selected_visual_option", "")).strip().lower() == "option2"
                    reference_asset_for_ai = _infer_reference_asset_for_ai(action_entry, matched_step)

                    if reference_asset_for_ai:
                        ai_url, ai_err = _generate_ai_visual_replacement_from_reference(
                            slide_title=slide_title,
                            slide_chunk=slide_chunk,
                            vo_part=_safe_str(matched_step.get("voiceover_part", "")),
                            feedback_text=effective_feedback,
                            visual_id=visual_id,
                            segment_num=segment_num,
                            round_index=round_index,
                            reference_url=reference_asset_for_ai,
                            drive=drive,
                            skip_accuracy_validation=is_option2_regen,
                        )
                        if ai_err:
                            print(
                                f"  WARNING: Reference pipeline failed for {visual_id}; "
                                "falling back to standard AI generation."
                            )

                    if not ai_url:
                        ai_url, ai_err = _generate_ai_visual_replacement(
                            slide_title=slide_title,
                            slide_chunk=slide_chunk,
                            vo_part=_safe_str(matched_step.get("voiceover_part", "")),
                            feedback_text=effective_feedback,
                            visual_id=visual_id,
                            segment_num=segment_num,
                            round_index=round_index,
                            drive=drive,
                        )
                    if ai_err:
                        print(f"  WARNING: {ai_err}")
                        ai_generation_errors.append(ai_err)
                        continue
                    ai_generated_urls.add(ai_url)
                    justification = _safe_str(matched_step.get("selection_justification", "")).strip() or "AI-generated visual selected based on reviewer feedback."
                    ai_visual_blocks.append(
                        (
                            f"<visual>\n"
                            f"<visual_id>{visual_id}</visual_id>\n"
                            f"<voiceover_part>{_safe_str(matched_step.get('voiceover_part', ''))}</voiceover_part>\n"
                            f"<current_visual_url>{_safe_str(matched_step.get('asset', ''))}</current_visual_url>\n"
                            f"<replacement_visual_url>{ai_url}</replacement_visual_url>\n"
                            f"<visual_instruction>{_safe_str(matched_step.get('visual_instruction', ''))}</visual_instruction>\n"
                            f"<selection_justification>{justification}</selection_justification>\n"
                            f"</visual>"
                        )
                    )
                else:
                    non_ai_vo_fb_pairs.append((vo_part, fb_text))

            revised_xml = ""
            if non_ai_vo_fb_pairs:
                review_segment_nums.append(segment_num)
                review_vo_parts_with_feedback_by_segment[segment_num] = {
                    _normalize_vo_for_match(vo_part) for vo_part, _ in non_ai_vo_fb_pairs
                }
                human_feedback_formatted = "\n\n".join(
                    f"When VO: {vo_part}\nHuman Feedback: {fb_text}"
                    for vo_part, fb_text in non_ai_vo_fb_pairs
                )
                seg_mode = _safe_str(segment_mode_map.get(str(segment_num), "drive_hvac")).strip().lower()
                candidate_mode = "all" if seg_mode == "all" else "drive_hvac"
                revised_xml = revise_segment_with_human_feedback(
                    course_name=course_name,
                    target_audience=target_audience,
                    topic_name=topic_name,
                    subtopic_name=subtopic_name,
                    slide_title=slide_title,
                    slide_chunk=slide_chunk,
                    vo_text=vo_text,
                    current_visuals=current_visuals,
                    human_feedback=human_feedback_formatted,
                    image_pool_text=_safe_str(row.get("image_pool", "")),
                    video_pool_filtered_text=_safe_str(row.get("video_pool_filtered", "")),
                    drive_results_text=_safe_str(row.get("drive_results", "")),
                    web_results_text=_safe_str(row.get("web_results", "")),
                    segment_num=segment_num,
                    drive=drive,
                    llm=llm,
                    visual_assignment_strategy=visual_assignment_strategy,
                    video_pool_text=_safe_str(row.get("video_pool", "")),
                    video_pool_other_channels_text=_safe_str(row.get("video_pool_other_channels", "")),
                    candidate_mode=candidate_mode,
                )

            combined_blocks: List[str] = []
            if revised_xml:
                stripped = revised_xml.strip()
                if not stripped.startswith("<replacement_visuals>") and not stripped.startswith("<replacement_visual>"):
                    revised_xml = f"<replacement_visuals>\n{revised_xml}\n</replacement_visuals>"
                combined_blocks.extend(_extract_visual_blocks(revised_xml))
            combined_blocks.extend([re.sub(r"^\s*<visual>\s*|\s*</visual>\s*$", "", b.strip(), flags=re.DOTALL) for b in ai_visual_blocks])
            revised_xml = _compose_replacement_xml_from_visual_blocks(combined_blocks)

            if revised_xml:
                processed_xml = expand_youtube_single_timestamp_clips_in_xml(revised_xml)
                if processed_xml != revised_xml:
                    print(f"  Processed video frames for segment {segment_num}")
                    revised_xml = processed_xml
                updated_segments[segment_num] = revised_xml
            else:
                print(f"  WARNING: Revision produced no output for segment {segment_num}")

        if updated_segments:
            for segment_num, replacement_xml in updated_segments.items():
                final_graphics_definition = update_final_graphics_definition_with_replacements(
                    final_graphics_definition,
                    segment_num,
                    replacement_xml,
                )
            df.at[row_index, "final_graphics_definition"] = final_graphics_definition

            processed_def = process_video_frames_in_text_format(final_graphics_definition, drive)
            if processed_def != final_graphics_definition:
                final_graphics_definition = processed_def
                df.at[row_index, "final_graphics_definition"] = final_graphics_definition
                print("  Processed video frames in final graphics definition")

            if ai_generated_urls:
                labeled_def = _add_ai_generated_label_for_urls(final_graphics_definition, ai_generated_urls)
                if labeled_def != final_graphics_definition:
                    final_graphics_definition = labeled_def
                    df.at[row_index, "final_graphics_definition"] = final_graphics_definition
                    print(f"  Added (AI Generated) label for {len(ai_generated_urls)} visual(s)")

            if ws is not None:
                print("  Saving revision to sheet...")
                with _sheet_lock:
                    save_to_sheet(ws, df)
                print("  Revision saved")
        else:
            print("  WARNING: No segments were updated by revision")

        # --- Rebuild segments_map and update tracking with after_revision URLs ---
        replaced_visual_ids_by_segment = {}
        if updated_segments:
            for segment_num, replacement_xml in updated_segments.items():
                replaced_visual_ids_by_segment[segment_num] = _extract_replaced_visual_ids_from_replacement_xml(
                    replacement_xml
                )
        row = df.loc[row_index]
        final_graphics_definition = _safe_str(row.get("final_graphics_definition", ""))
        segments_map = build_segment_visual_map(
            voiceover_text, final_graphics_definition, visual_assignment_strategy, slide_chunk
        )
        _update_human_feedback_revision_tracking(
            hf_revision_tracking, "after_revision", segments_map, replaced_visual_ids_by_segment
        )

        # AI-only rows: skip review + regeneration by design.
        if not review_segment_nums:
            print("\n[STEP 2] Skipped: only AI-generation actions were requested.")
            _finalize_row(row_index, df, drive, ws)
            if ai_generation_errors and not updated_segments:
                df.at[row_index, human_feedback_status_column] = _compose_hf_status_with_ai_errors(
                    "ERROR", ai_generation_errors
                )
            else:
                df.at[row_index, human_feedback_status_column] = _compose_hf_status_with_ai_errors(
                    "PASS", ai_generation_errors
                )
            df.at[row_index, human_feedback_revision_tracking_column] = _format_human_feedback_revision_tracking(hf_revision_tracking)
            if ws is not None:
                with _sheet_lock:
                    save_to_sheet(ws, df)
            if ai_generation_errors and not updated_segments:
                print(f"Row {row_index + 1}: ERROR (AI upload failed; no revisions saved)")
            else:
                print(f"Row {row_index + 1}: PASS (AI generation only, review skipped)")
            return

        # --- REVIEW (single) ---
        print("\n[STEP 2] Reviewing replacements against human feedback...")
        verdict, failures, satisfaction_conversation_history = review_human_feedback_satisfaction(
            course_name=course_name,
            target_audience=target_audience,
            topic_name=topic_name,
            subtopic_name=subtopic_name,
            slide_id=slide_id,
            slide_title=slide_title,
            slide_chunk=slide_chunk,
            human_feedback=_format_human_feedback_for_prompt(human_feedback_raw),
            segments_map=segments_map,
            segment_nums=review_segment_nums,
            drive=drive,
            llm=llm,
            visual_assignment_strategy=visual_assignment_strategy,
            original_visuals_by_segment={
                k: v for k, v in original_visuals_by_segment.items() if k in set(review_segment_nums)
            },
            vo_parts_with_feedback_by_segment=review_vo_parts_with_feedback_by_segment,
        )

        if verdict == "PASS":
            print(f"\n  Human feedback satisfaction PASSED for row {row_index + 1}")
            _finalize_row(row_index, df, drive, ws)
            df.at[row_index, human_feedback_status_column] = _compose_hf_status_with_ai_errors(
                "PASS", ai_generation_errors
            )
            df.at[row_index, human_feedback_revision_tracking_column] = _format_human_feedback_revision_tracking(hf_revision_tracking)
            if ws is not None:
                with _sheet_lock:
                    save_to_sheet(ws, df)
            print(f"Row {row_index + 1}: PASS")
            return

        # --- REGENERATION LOOP ---
        print(f"\n  Human feedback satisfaction FAILED for row {row_index + 1} — preparing for regeneration")
        failed_segments, regen_feedback = _build_regen_feedback(failures, segments_map, visual_assignment_strategy)
        _finalize_row(row_index, df, drive, ws)

        print(f"\n  Entering regeneration for row {row_index + 1} ({len(failed_segments)} failed segment(s))...")
        conversation_history = satisfaction_conversation_history
        for attempt in range(1, MAX_HUMAN_FEEDBACK_REGEN_ATTEMPTS + 1):
            print(f"\n  Regeneration attempt {attempt}/{MAX_HUMAN_FEEDBACK_REGEN_ATTEMPTS}")
            row = df.loc[row_index]
            restricted_segments = [s for s in failed_segments if _safe_str(segment_mode_map.get(str(s), "drive_hvac")).strip().lower() != "all"]
            all_segments = [s for s in failed_segments if _safe_str(segment_mode_map.get(str(s), "drive_hvac")).strip().lower() == "all"]
            replaced_visual_ids_by_segment = {}
            old_asset_urls_by_visual_id = {}

            if restricted_segments:
                restricted_feedback = {k: v for k, v in regen_feedback.items() if k in restricted_segments}
                r_rep, r_old = regenerate_failed_segments(
                    row_index=row_index,
                    row=row,
                    df=df,
                    course_name=course_name,
                    target_audience=target_audience,
                    drive=drive,
                    llm=llm,
                    failed_segments=restricted_segments,
                    feedback_by_segment=restricted_feedback,
                    ws=ws,
                    use_only_drive_and_hvac=True,
                )
                replaced_visual_ids_by_segment.update(r_rep or {})
                old_asset_urls_by_visual_id.update(r_old or {})

            if all_segments:
                all_feedback = {k: v for k, v in regen_feedback.items() if k in all_segments}
                a_rep, a_old = regenerate_failed_segments(
                    row_index=row_index,
                    row=row,
                    df=df,
                    course_name=course_name,
                    target_audience=target_audience,
                    drive=drive,
                    llm=llm,
                    failed_segments=all_segments,
                    feedback_by_segment=all_feedback,
                    ws=ws,
                    use_only_drive_and_hvac=False,
                    create_aux_search_columns_if_missing=True,
                )
                replaced_visual_ids_by_segment.update(a_rep or {})
                old_asset_urls_by_visual_id.update(a_old or {})
            _finalize_row(row_index, df, drive, ws)

            row = df.loc[row_index]
            voiceover_text = _safe_str(row.get("voiceover_segment", ""))
            final_def = _safe_str(row.get("final_graphics_definition", ""))
            slide_chunk = _safe_str(row.get("Slide Chunk", ""))
            vas = str(row.get("Visual Assignment Strategy", "Flexible, let the agent decide")).strip()
            if not vas or vas == "nan":
                vas = "Flexible, let the agent decide"
            segments_map = build_segment_visual_map(voiceover_text, final_def, vas, slide_chunk)

            stage = "after_regen_1" if attempt == 1 else "after_regen_2"
            _update_human_feedback_revision_tracking(
                hf_revision_tracking, stage, segments_map, replaced_visual_ids_by_segment
            )

            # Build previous failures text
            previous_failures_lines = []
            for seg_num in failed_segments:
                if seg_num not in regen_feedback:
                    continue
                feedback_str = regen_feedback[seg_num]
                enriched_lines = []
                for line in feedback_str.split("\n"):
                    enriched_lines.append(line)
                    stripped_line = line.strip()
                    if stripped_line.startswith("Failing Visual:"):
                        visual_id = stripped_line.replace("Failing Visual:", "").strip()
                        if visual_id:
                            old_url = old_asset_urls_by_visual_id.get(visual_id, "")
                            if old_url:
                                enriched_lines.append(f"Original URL: {old_url}")
                block = "\n".join(enriched_lines)
                previous_failures_lines.append(f"Segment {seg_num}:\n{block}")
            previous_failures_text = "\n\n".join(previous_failures_lines)

            human_feedback_formatted = _format_human_feedback_for_prompt(_safe_str(row.get(human_feedback_column, "")))

            print(f"\n  [REGENERATION] Follow-up satisfaction review (attempt {attempt}):")
            if conversation_history and replaced_visual_ids_by_segment:
                regen_verdict, regen_failures, conversation_history = review_human_feedback_satisfaction_followup(
                    slide_id=f"SLIDE_{row_index + 1}",
                    human_feedback=human_feedback_formatted,
                    previous_failures_text=previous_failures_text,
                    visual_ids_by_segment=replaced_visual_ids_by_segment,
                    old_asset_urls_by_visual_id=old_asset_urls_by_visual_id,
                    segments_map=segments_map,
                    segment_nums=failed_segments,
                    drive=drive,
                    llm=llm,
                    conversation_history=conversation_history,
                    visual_assignment_strategy=vas,
                    slide_chunk=slide_chunk,
                )
            else:
                print(f"  No conversation history available; using new chat for satisfaction review")
                regen_verdict, regen_failures, conversation_history = review_human_feedback_satisfaction(
                    course_name=course_name,
                    target_audience=target_audience,
                    topic_name=_safe_str(row.get("Topic", "")),
                    subtopic_name=_safe_str(row.get("Subtopic", "")),
                    slide_id=f"SLIDE_{row_index + 1}",
                    slide_title=_safe_str(row.get("Slide Chunk Title", "")),
                    slide_chunk=slide_chunk,
                    human_feedback=human_feedback_formatted,
                    segments_map=segments_map,
                    segment_nums=failed_segments,
                    drive=drive,
                    llm=llm,
                    visual_assignment_strategy=vas,
                )

            if regen_verdict == "PASS":
                print(f"  Regeneration PASSED on attempt {attempt}")
                _finalize_row(row_index, df, drive, ws)
                df.at[row_index, human_feedback_status_column] = _compose_hf_status_with_ai_errors(
                    "PASS", ai_generation_errors
                )
                df.at[row_index, human_feedback_revision_tracking_column] = _format_human_feedback_revision_tracking(hf_revision_tracking)
                if ws is not None:
                    with _sheet_lock:
                        save_to_sheet(ws, df)
                print(f"Row {row_index + 1}: PASS (after regen attempt {attempt})")
                return
            else:
                print(f"  Regeneration attempt {attempt} FAILED")
                failed_segments, regen_feedback = _build_regen_feedback(
                    regen_failures, segments_map, vas
                )

        # All regen attempts exhausted
        _finalize_row(row_index, df, drive, ws)
        df.at[row_index, human_feedback_status_column] = _compose_hf_status_with_ai_errors(
            "FAIL", ai_generation_errors
        )
        df.at[row_index, human_feedback_revision_tracking_column] = _format_human_feedback_revision_tracking(hf_revision_tracking)
        if ws is not None:
            with _sheet_lock:
                save_to_sheet(ws, df)
        print(f"  Row {row_index + 1}: FAIL after {MAX_HUMAN_FEEDBACK_REGEN_ATTEMPTS} regen attempts")

    except Exception as e:
        print(f"  ERROR processing human-feedback row {row_index + 1}: {e}")
        df.at[row_index, human_feedback_status_column] = f"ERROR: {str(e)}"
        if ws is not None:
            with _sheet_lock:
                save_to_sheet(ws, df)
        raise


def _finalize_row(row_index, df, drive, ws):
    """
    Normalize YouTube URLs, process video frames, add snapshot labels, and save the row to the sheet.

    :param row_index: Row index in the dataframe
    :param df: Dataframe (shared)
    :param drive: Google Drive instance
    :param ws: Worksheet object or None
    :return: None
    """
    
    row = df.loc[row_index]
    final_def = _safe_str(row.get("final_graphics_definition", ""))
    if not final_def:
        return

    normalized = normalize_youtube_timestamp_urls(final_def)
    processed = process_video_frames_in_text_format(normalized, drive)
    if processed != final_def:
        df.at[row_index, "final_graphics_definition"] = processed
        print(f"  Final graphics definition updated with Drive image URLs")

    row = df.loc[row_index]
    current_def = _safe_str(row.get("final_graphics_definition", ""))
    if current_def:
        labeled = add_snapshot_label_to_drive_links(current_def, drive)
        if labeled != current_def:
            df.at[row_index, "final_graphics_definition"] = labeled
            print(f"  Added (snapshot) labels to Drive links")

    if ws is not None:
        print(f"  Saving finalized row {row_index + 1} to sheet...")
        with _sheet_lock:
            save_to_sheet(ws, df)
        print(f"  Row {row_index + 1} saved")


def _build_regen_feedback(failures, segments_map, visual_assignment_strategy):
    """
    Build failed_segments list and feedback_by_segment dict from satisfaction review failures.

    :param failures: List of failure dicts from parse_review_response
    :param segments_map: Map of segment_num to segment data
    :param visual_assignment_strategy: Visual assignment strategy
    :return: Tuple of (failed_segments, feedback_by_segment)
    """
    
    failed_segments: List[int] = []
    feedback_by_segment: Dict[int, str] = {}

    for failure in failures:
        segment_id_text = failure.get("segment_id", "")
        segment_num = _parse_segment_marker(segment_id_text)
        if not segment_num and visual_assignment_strategy == "1 Visual for the whole Slide":
            if re.search(r"SLIDE_", segment_id_text, re.IGNORECASE):
                segment_num = 1
        if not segment_num:
            continue

        if segment_num not in failed_segments:
            failed_segments.append(segment_num)

        failing_visual_id = failure.get("failing_visual_ids", "")

        voiceover_part = ""
        segment = segments_map.get(segment_num, {})
        for step in segment.get("visual_steps", []):
            if step.get("visual_id") == failing_visual_id:
                voiceover_part = step.get("voiceover_part", "")
                break

        failure_text = f"Failing Visual: {failing_visual_id}"
        if voiceover_part:
            failure_text += f"\nVoiceover sentence to which this visual was assigned: {voiceover_part}"
        failure_text += f"\nReason: {failure.get('reason', '')}\nNeeded: {failure.get('needed_visual', '')}"

        if segment_num in feedback_by_segment:
            feedback_by_segment[segment_num] += "\n\n" + failure_text
        else:
            feedback_by_segment[segment_num] = failure_text

    return failed_segments, feedback_by_segment


HUMAN_FEEDBACK_REVISION_TRACKING_COLUMN = "human_feedback_revision_tracking"


def _parse_actions_payload(raw):
    text = _safe_str(raw).strip()
    if not text:
        return {}, {}
    try:
        payload = json.loads(text)
        if not isinstance(payload, dict):
            return {}, {}
        actions = payload.get("actions", {})
        segment_modes = payload.get("segment_modes", {})
        if not isinstance(actions, dict):
            actions = {}
        if not isinstance(segment_modes, dict):
            segment_modes = {}
        return actions, segment_modes
    except Exception:
        return {}, {}


def _is_drive_url(url: str) -> bool:
    if not url:
        return False
    lowered = str(url).lower()
    return "drive.google.com" in lowered or "docs.google.com" in lowered


def _is_embed_with_start_end(url: str) -> bool:
    if not url:
        return False
    try:
        parsed = urlparse(str(url))
        if "youtube.com" not in parsed.netloc or "/embed/" not in parsed.path:
            return False
        qs = parse_qs(parsed.query or "")
        return bool(qs.get("start")) and bool(qs.get("end"))
    except Exception:
        return False


def _extract_visual_blocks(replacement_xml: str) -> List[str]:
    if not replacement_xml or not replacement_xml.strip():
        return []
    content = _extract_tag(replacement_xml, "replacement_visuals") or _extract_tag(replacement_xml, "replacement_visual")
    if not content:
        return []
    return re.findall(r"<visual>(.*?)</visual>", content, re.DOTALL | re.IGNORECASE)


def _compose_replacement_xml_from_visual_blocks(visual_blocks: List[str]) -> str:
    if not visual_blocks:
        return ""
    parts = ["<replacement_visuals>"]
    for block in visual_blocks:
        parts.append("<visual>")
        parts.append(block.strip())
        parts.append("</visual>")
    parts.append("</replacement_visuals>")
    return "\n".join(parts)


def _add_ai_generated_label_for_urls(graphics_definition_text: str, ai_urls: Set[str]) -> str:
    """
    Add '(AI Generated)' on a new line right after 'Graphics to use: <url>' for ai-generated URLs.
    """
    if not graphics_definition_text or not ai_urls:
        return graphics_definition_text

    lines = graphics_definition_text.splitlines()
    output_lines: List[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        output_lines.append(line)
        m = re.match(r"^(\s*)Graphics to use:\s*(\S+)\s*$", line, re.IGNORECASE)
        if m:
            url = m.group(2).strip()
            if url in ai_urls:
                next_line = lines[i + 1].strip() if i + 1 < len(lines) else ""
                if next_line != "(AI Generated)":
                    output_lines.append("(AI Generated)")
        i += 1
    return "\n".join(output_lines)


def _generate_ai_visual_replacement(
    slide_title: str,
    slide_chunk: str,
    vo_part: str,
    feedback_text: str,
    visual_id: str,
    segment_num: int,
    round_index: int,
    drive,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Generate an AI image, upload to Drive, and return shareable URL.
    Returns (url, error_message).
    """
    feedback_text = _safe_str(feedback_text).strip()
    if feedback_text and feedback_text != AI_NO_FEEDBACK_MARKER:
        voiceover_focus = f'When VO: "{vo_part}"\nHuman Feedback: {feedback_text}'
    else:
        voiceover_focus = f'When VO: "{vo_part}"'
    result = generate_asset(
        slide_title=slide_title,
        slide_content=slide_chunk,
        voiceover_focus=voiceover_focus,
        aspect_ratio="16:9",
        image_size="1K",
    )
    if result.get("error"):
        return None, f"AI generation failed for {visual_id}: {result['error']}"
    image = result.get("image")
    if image is None:
        return None, f"AI generation returned no image for {visual_id}"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{visual_id}_seg{segment_num}_ai_round{round_index}_{timestamp}.jpg"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
        image.convert("RGB").save(tmp_path, format="JPEG", quality=92)
        drive_url, upload_err = upload_image_to_drive(
            tmp_path, filename, AI_GENERATED_IMAGES_FOLDER_ID, drive
        )
        if not drive_url:
            detail = upload_err or "unknown error"
            return None, f"Drive upload failed for {visual_id}: {detail}"
        return drive_url, None
    except Exception as e:
        return None, f"AI image upload failed for {visual_id}: {e}"
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def _infer_reference_asset_for_ai(action_entry: Dict[str, object], matched_step: Dict[str, object]) -> str:
    """Infer the best reference URL for AI regeneration when available."""
    action_entry = action_entry or {}
    matched_step = matched_step or {}

    reference_asset = _safe_str(action_entry.get("reference_asset", "")).strip()
    if not reference_asset:
        reference_asset = _safe_str(matched_step.get("asset_reference", "")).strip()

    selected_option = _safe_str(action_entry.get("selected_visual_option", "")).strip().lower()
    assigned_asset = _safe_str(action_entry.get("assigned_asset", "")).strip()
    current_asset = _safe_str(matched_step.get("asset", "")).strip()

    if reference_asset:
        if selected_option == "option2":
            return reference_asset
        if assigned_asset and assigned_asset == reference_asset:
            return reference_asset
        if current_asset and current_asset == reference_asset:
            return reference_asset

    if selected_option == "option2":
        return assigned_asset or current_asset

    return ""


def _generate_ai_visual_replacement_from_reference(
    slide_title: str,
    slide_chunk: str,
    vo_part: str,
    feedback_text: str,
    visual_id: str,
    segment_num: int,
    round_index: int,
    reference_url: str,
    drive,
    skip_accuracy_validation: bool = False,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Generate replacement by editing the provided reference image, then upload to Drive.
    Returns (url, error_message).
    """
    reference_url = _safe_str(reference_url).strip()
    if not reference_url:
        return None, f"Reference image URL missing for {visual_id}"

    try:
        from agents.graphics_asset_creation.automated.automated_voiceover_reviewer import (
            _download_drive_image,
            review_and_edit_image,
        )
    except Exception as e:
        return None, f"Reference pipeline import failed for {visual_id}: {e}"

    try:
        reference_image = _download_drive_image(reference_url, drive=drive)
    except Exception as e:
        return None, f"Reference image download failed for {visual_id}: {e}"

    try:
        _, final_image, _ = review_and_edit_image(
            reference_image=reference_image,
            slide_title=slide_title,
            slide_content=slide_chunk,
            voiceover=_safe_str(vo_part).strip() or slide_chunk,
            visual_instruction=_safe_str(feedback_text).strip(),
            image_size="1K",
            target_stage="full",
            skip_accuracy_validation=bool(skip_accuracy_validation),
        )
    except Exception as e:
        return None, f"Reference pipeline generation failed for {visual_id}: {e}"

    if final_image is None:
        return None, f"Reference pipeline produced no image for {visual_id}"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{visual_id}_seg{segment_num}_ai_ref_round{round_index}_{timestamp}.jpg"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
        final_image.convert("RGB").save(tmp_path, format="JPEG", quality=92)
        drive_url, upload_err = upload_image_to_drive(
            tmp_path, filename, AI_GENERATED_IMAGES_FOLDER_ID, drive
        )
        if not drive_url:
            detail = upload_err or "unknown error"
            return None, f"Drive upload failed for {visual_id}: {detail}"
        return drive_url, None
    except Exception as e:
        return None, f"Reference pipeline upload failed for {visual_id}: {e}"
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def _get_asset_url_for_visual(segments_map, visual_id):
    """
    Return current asset URL for visual_id from segments_map, or None if not found.

    :param segments_map: Map of segment_num to segment data
    :param visual_id: Visual ID to look up
    :return: Asset URL string or None
    """
    
    for segment_data in segments_map.values():
        for step in segment_data.get("visual_steps", []):
            if step.get("visual_id") == visual_id:
                return step.get("asset") or None
    return None


def _initialize_human_feedback_revision_tracking(segments_map, feedback_by_segment, existing_tracking_text=None):
    """
    Initialize revision tracking by merging existing tracking data with new revision targets.
    This preserves "Manual Selection" and "Original" URLs even during AI revision cycles.

    :param segments_map: Map of segment_num to segment data
    :param feedback_by_segment: segment_num -> list of (vo_part, fb_text)
    :param existing_tracking_text: Raw string from the sheet's tracking column
    :return: Tracking dict keyed by visual_id with merged state
    """
    # 1. Parse existing tracking text if provided
    tracking = _parse_tracking_column(existing_tracking_text) if existing_tracking_text else {}
    
    # 2. Ensure all visuals with feedback are represented
    for seg_num, vo_fb_pairs in feedback_by_segment.items():
        vo_parts_with_feedback = {_normalize_vo_for_match(vo_part) for vo_part, _ in vo_fb_pairs}
        segment = segments_map.get(seg_num, {})
        for step in segment.get("visual_steps", []):
            vo_part = step.get("voiceover_part", "")
            if _normalize_vo_for_match(vo_part) not in vo_parts_with_feedback:
                continue
            visual_id = step.get("visual_id", "")
            asset = step.get("asset", "")
            if visual_id and visual_id not in tracking:
                tracking[visual_id] = {
                    "original": asset or None,
                    "manually_selected": None,
                    "after_revision": None,
                    "after_regen_1": None,
                    "after_regen_2": None,
                }
    return tracking


def _extract_replaced_visual_ids_from_replacement_xml(replacement_xml):
    """
    Extract visual IDs that were replaced from a segment's replacement XML.

    :param replacement_xml: XML string with <replacement_visuals> or <replacement_visual> and <visual> blocks
    :return: List of visual_id strings that appear in the replacement
    """
    
    if not replacement_xml or not replacement_xml.strip():
        return []
    content = _extract_tag(replacement_xml, "replacement_visuals") or _extract_tag(replacement_xml, "replacement_visual")
    if not content:
        return []
    visual_blocks = re.findall(r"<visual>(.*?)</visual>", content, re.DOTALL | re.IGNORECASE)
    ids = []
    for block in visual_blocks:
        vid = _extract_tag(block, "visual_id").strip()
        if vid:
            ids.append(vid)
    return ids


def _update_human_feedback_revision_tracking(tracking, stage, segments_map, replaced_visual_ids_by_segment=None):
    """
    Update tracking with current URLs from segments_map for the given stage.

    :param tracking: Revision tracking dict to update
    :param stage: "after_revision", "after_regen_1", or "after_regen_2"
    :param segments_map: Current segments map with latest visuals
    :param replaced_visual_ids_by_segment: Optional; only these visuals get the new URL; others get No replacement
    :return: None
    """
    replaced_set: Set[str] = set()
    if replaced_visual_ids_by_segment:
        for vids in replaced_visual_ids_by_segment.values():
            replaced_set.update(vids)

    for visual_id in list(tracking.keys()):
        if visual_id in replaced_set:
            url = _get_asset_url_for_visual(segments_map, visual_id)
            tracking[visual_id][stage] = url
        else:
            tracking[visual_id][stage] = None


def _parse_tracking_column(text: str) -> Dict[str, Dict[str, Optional[str]]]:
    """
    Parse human_feedback_revision_tracking column text into visual_id -> stages.
    Returns dict: visual_id -> {original, manually_selected, after_revision, after_regen_1, after_regen_2}
    """
    out: Dict[str, Dict[str, Optional[str]]] = {}
    if not text or not text.strip() or text.strip() == "nan":
        return out

    current_id: Optional[str] = None
    data: Optional[Dict[str, Optional[str]]] = None
    for line in text.splitlines():
        line_stripped = line.strip()
        if re.match(r"^S\d+V\d+$", line_stripped):
            if current_id and data:
                out[current_id] = data
            current_id = line_stripped
            data = {
                "original": None,
                "manually_selected": None,
                "after_revision": None,
                "after_regen_1": None,
                "after_regen_2": None,
            }
            continue
        if not current_id or data is None:
            continue
        if line_stripped.startswith(ORIG_PREFIX):
            v = line_stripped[len(ORIG_PREFIX) :].strip()
            data["original"] = None if v == "(not found)" else v
        elif line_stripped.startswith(MANUAL_PREFIX):
            v = line_stripped[len(MANUAL_PREFIX) :].strip()
            data["manually_selected"] = None if v == NO_REPLACEMENT else v
        elif line_stripped.startswith(AFTER_REV_PREFIX):
            v = line_stripped[len(AFTER_REV_PREFIX) :].strip()
            data["after_revision"] = None if v == NO_REPLACEMENT else v
        elif line_stripped.startswith(AFTER_REGEN1_PREFIX):
            v = line_stripped[len(AFTER_REGEN1_PREFIX) :].strip()
            data["after_regen_1"] = None if v == NO_REPLACEMENT else v
        elif line_stripped.startswith(AFTER_REGEN2_PREFIX):
            v = line_stripped[len(AFTER_REGEN2_PREFIX) :].strip()
            data["after_regen_2"] = None if v == NO_REPLACEMENT else v
    if current_id and data:
        out[current_id] = data
    return out


def _format_human_feedback_revision_tracking(tracking):
    """
    Format human-feedback revision tracking into a readable string for the sheet column.

    :param tracking: Revision tracking dict keyed by visual_id
    :return: Formatted string for the sheet column
    """
    
    if not tracking:
        return ""

    def _sort_key(vid):
        s_match = re.search(r"S(\d+)", vid)
        v_match = re.search(r"V(\d+)", vid)
        return (
            int(s_match.group(1)) if s_match else 0,
            int(v_match.group(1)) if v_match else 0,
        )

    lines: List[str] = []
    for visual_id in sorted(tracking.keys(), key=_sort_key):
        lines.append(visual_id)
        lines.append("")
        data = tracking[visual_id]
        orig = data.get("original")
        lines.append(f"Original visual - {orig if orig else '(not found)'}")
        ms = data.get("manually_selected")
        if ms:
            lines.append(f"Manual Selection - {ms}")
        ar = data.get("after_revision")
        lines.append(f"After revision - {ar if ar else 'No replacement'}")
        r1 = data.get("after_regen_1")
        lines.append(f"After regeneration loop 1 - {r1 if r1 else 'No replacement'}")
        r2 = data.get("after_regen_2")
        lines.append(f"After regeneration loop 2 - {r2 if r2 else 'No replacement'}")
        lines.append("")
    return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# E. Entry point
# ---------------------------------------------------------------------------

@traceable(
    metadata={
        "agent_name": "graphics_definition_v2",
        "step_name": "Human Feedback Review and Revise",
        "function_name": "run_human_feedback_review_revise_for_all_rows",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous"),
    }
)
def run_human_feedback_review_revise_for_all_rows(
    sheet,
    llm="gemini_3_flash_thinking",
    max_workers=50,
    use_only_drive_and_hvac=False,
    human_feedback_column="human_feedback",
    human_feedback_status_column="human_feedback_status",
    human_feedback_revision_tracking_column=HUMAN_FEEDBACK_REVISION_TRACKING_COLUMN,
    human_review_actions_column="human_review_actions",
):
    """
    Entry point: process all rows that have human feedback in the Slide Chunks sheet (revise, review, optional regeneration per row).

    :param sheet: gspread sheet object
    :param llm: LLM model name
    :param max_workers: Number of parallel workers
    :param use_only_drive_and_hvac: If True, skip web search and other-channels video search during regeneration.
    :return: None
    """
    worksheet_name = "Slide Chunks"
    ws, df = get_sheet_data_and_df(sheet, worksheet_name)

    _, course_info_df = get_sheet_data_and_df(sheet, "Course info")
    course_name = _safe_str(course_info_df.loc[0, "Course Name"])
    target_audience = _safe_str(course_info_df.loc[0, "Target Audience & Industry"])

    if human_feedback_column not in df.columns:
        print(f"No {human_feedback_column} column found. Nothing to process.")
        return
    if human_feedback_status_column not in df.columns:
        df[human_feedback_status_column] = ""
    if human_feedback_revision_tracking_column not in df.columns:
        df[human_feedback_revision_tracking_column] = ""
    if human_review_actions_column not in df.columns:
        df[human_review_actions_column] = ""

    print("Initializing Google Drive instance...")
    drive = get_drive_instance()
    if not drive:
        print("ERROR: Drive instance unavailable. Aborting.")
        return
    try:
        about = drive.GetAbout()
        email = (about.get("user") or {}).get("emailAddress") or ""
        if email.endswith(".gserviceaccount.com"):
            print(f"Drive instance initialized (service account: {email})")
        else:
            print(f"Drive instance initialized (OAuth user: {email})")
    except Exception as e:
        print(f"Drive instance initialized (could not read identity: {e})")

    rows_to_process = []
    for index, row in df.iterrows():
        hf = _safe_str(row.get(human_feedback_column, "")).strip()
        if not hf or hf == "nan":
            continue
        status_raw = _safe_str(row.get(human_feedback_status_column, ""))
        if _hf_row_should_skip_in_batch(status_raw):
            print(f"  Skipping row {index + 1}: {human_feedback_status_column}={status_raw}")
            continue
        rows_to_process.append(index)

    if not rows_to_process:
        print("No rows with pending human feedback found.")
        return

    print(f"\n{'=' * 80}")
    print(f"Starting Human Feedback Review & Revise for {len(rows_to_process)} row(s)")
    print(f"{'=' * 80}\n")

    progress = SmartProgressBar(
        total_tasks=len(rows_to_process),
        description="Human feedback review-revise",
    )
    _progress_lock = threading.Lock()

    def _safe_progress_update():
        with _progress_lock:
            progress.update()

    if max_workers > 1:
        print(f"Processing {len(rows_to_process)} row(s) in parallel with {max_workers} worker(s)...\n")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    process_human_feedback_row,
                    row_index,
                    df,
                    course_name,
                    target_audience,
                    drive,
                    llm,
                    ws,
                    use_only_drive_and_hvac,
                    human_feedback_column,
                    human_feedback_status_column,
                    human_feedback_revision_tracking_column,
                    human_review_actions_column,
                ): row_index
                for row_index in rows_to_process
            }

            for future in as_completed(futures):
                row_index = futures[future]
                try:
                    future.result()
                    _safe_progress_update()
                    print(f"Row {row_index + 1} completed and saved")
                except Exception as e:
                    print(f"Error processing row {row_index + 1}: {e}")
                    _safe_progress_update()
    else:
        print(f"Processing {len(rows_to_process)} row(s) sequentially...\n")
        for row_index in rows_to_process:
            try:
                process_human_feedback_row(
                    row_index,
                    df,
                    course_name,
                    target_audience,
                    drive,
                    llm,
                    ws,
                    use_only_drive_and_hvac,
                    human_feedback_column,
                    human_feedback_status_column,
                    human_feedback_revision_tracking_column,
                    human_review_actions_column,
                )
                _safe_progress_update()
            except Exception as e:
                print(f"Error processing row {row_index + 1}: {e}")
                _safe_progress_update()

    if ws is not None:
        with _sheet_lock:
            save_to_sheet(ws, df)
            format_worksheet(ws)

    print(f"\n{'=' * 80}")
    print("Human Feedback Review & Revise complete.")
    print(f"{'=' * 80}")


def delete_human_feedback_based_review_and_revise(sheet):
    """
    Delete all columns that are generated while running this step (human_feedback_status, human_feedback_revision_tracking)

    :param sheet: gspread sheet object
    :return: None
    """
    try:
        worksheet_name = "Slide Chunks"
        ws, df = get_sheet_data_and_df(sheet, worksheet_name)
        cols = ["human_feedback_status", "human_feedback_revision_tracking"]
        cols = [col for col in cols if col in df.columns]
        if cols:
            df = df.drop(columns=cols)
            clear_worksheet(ws)
            save_to_sheet(ws, df)
            print("Human feedback based review and revise results deleted.")
        else:
            print("Human feedback columns not found in worksheet.")
        return True
    except Exception as e:
        print(f"Error deleting human feedback based review and revise results: {e}")
        return False