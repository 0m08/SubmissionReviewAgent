from __future__ import annotations

import re
import time
import traceback
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from typing import Dict, List, Optional, Tuple, Any
import re as regex_module
import streamlit as st
from dotenv import load_dotenv
from langsmith import traceable
from google import genai
from google.genai import types
import urllib.parse
import tempfile
import os
from modules.chain import Chain

from services.sheets_service import (
    get_sheet_data_and_df,
    save_to_sheet,
    format_worksheet,
    clear_worksheet,
    hide_columns_by_name,
)
from services.smart_progress_bar import SmartProgressBar
from services.helper_functions import build_video_part
from services.llm_service import extract_token_usage, log_token_usage

from agents.graphics_definition_v2.aggregation_agent.aggregation_agent import (
    get_drive_instance,
    load_image_from_url,
    parse_urls_from_results,
    parse_urls_from_video_pool,
    parse_urls_from_video_pool_other_channels,
    parse_urls_from_image_pool,
    parse_urls_from_video_pool_filtered,
    get_video_items_fallback_from_pools,
    parse_segments_from_voiceover,
    parse_video_url_timestamps,
    convert_watch_url_to_embed_url,
    expand_youtube_single_timestamp_clips_in_xml,
    process_video_frames_in_text_format,
    normalize_youtube_timestamp_urls,
    format_aggregation_definition_for_sheet,
    aggregate_graphics_definition_for_segment,
    video_frames_drive_folder_id,
)
from agents.graphics_definition_v2.image_graphics_agent.storyboard_agent import (
    format_storyboard_for_sheet,
)
from agents.graphics_definition_v2.image_graphics_agent.drive_search import (
    process_drive_search_segment,
)
from agents.graphics_definition_v2.image_graphics_agent.web_search import (
    process_web_search_segment,
)
from agents.graphics_definition_v2.video_graphics_agent.youtube_video_search_from_queries import (
    process_video_search_segment,
)
from agents.graphics_definition_v2.video_graphics_agent.youtube_video_search_in_other_channels import (
    process_segment_other_channels,
)
from agents.graphics_definition_v2.image_graphics_agent.image_selection_from_all_images import (
    select_images_from_all_for_segment,
    select_images_from_all_for_entire_slide,
    format_selected_images_for_segment,
)
from agents.graphics_definition_v2.video_graphics_agent.video_selection_from_all_videos import (
    select_videos_from_all_for_segment,
    select_videos_from_all_for_entire_slide,
    format_selected_videos_for_segment,
    parse_video_items_from_pool_other_channels,
)

load_dotenv()

MAX_REVIEW_ATTEMPTS = 1
MAX_REGEN_ATTEMPTS = 2
REGEN_IMAGE_SEARCH_K = 4
REGEN_VIDEO_SEARCH_K = 3


# Alingment prompt to use when we have flexible or 1 visual per sentence visual assingment strategy
ALIGNMENT_REVIEW_PROMPT = """You are a Graphics Definition Review Agent specializing in the field of HVAC.

Criterion: Visual–Voiceover Alignment Accuracy

Definition: For the given slide, verify that the assigned visual(s) for the given voiceover segment(s) clearly and directly show what the voiceover is saying at that moment.

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

These are the voiceover segments of this slide and the assigned visuals for each segment:
<review_targets>
{review_targets}
</review_targets>

Note: Visual IDs follow the format "S(segment_number)V(visual_number)". For example, "S1V3" means Segment 1, Visual 3 (the third visual assigned to segment 1 for its respecitve voiceover text), "S2V1" means Segment 2, Visual 1 (the first visual assigned to segment 2 for its respecitve voiceover text) and so on.

Instructions:
Follow the below evaluation rules to guide your evaluation:

1) Scope
   - Review each voiceover segment independently as the primary evaluation unit.
   - The assigned visuals are displayed on screen as the voiceover text for that segment is played/narrated.
   - You may use the full slide content and the sequence of voiceover segments to understand the intended meaning of a segment (for example, split sentences, pronouns, or continuation phrases).
   - When interpreting any part of the voiceover sentences, you must use the surrounding text in the slide content to understand the context and the instructional intent. Do not interpret a sentence, phrase or a clause literally in isolation. Always derive the meaning of a sentence, phrase or a clause from the surrounding text in the slide content.
   - Judge alignment only between that segment's intended meaning and the visuals explicitly assigned to it.
   - Use only the provided assets and voiceover text; do not assume missing context beyond what is present in the slide.
   - Each visual asset has a Visual ID (for example, S2V1). Use these IDs when listing any failures.

2) What counts as PASS for a segment
   - The assigned visual(s) clearly show what is described by its respective voiceover sentence.
   - The visual(s) match the specific meaning of the segment as spoken, not just the general topic of the slide.
   - The visual should be specific and clear enough that the exact object/action/detail in the segment is easy to identify without guesswork.
   - For segments that are part of a split sentence (e.g., lists or continuations), the visual must correctly represent the specific item or clause being spoken in that segment.
   - If multiple visuals are assigned to a segment, together they must fully support the segment’s meaning without introducing confusion or contradiction.
   - A learner should be able to understand what the voiceover segment is referring to by looking at the assigned visual(s) at that moment.

3) What counts as FAIL for a segment
   A segment FAILS if any one of the following is true:
   - The visual shows something different than what the voiceover segment describes.
   - The visual is generic, symbolic, or only loosely related, and does not clearly illustrate the specific meaning of the segment.
   - The visual is too vague, distant, cluttered, or unclear to confidently identify the exact detail being referenced.
   - The visual represents the general topic but not the specific clause or item being spoken in that segment.
   - The visual contradicts the voiceover or implies a different instructional idea.
   - The assigned visual(s) do not provide enough visual evidence for a learner to understand the segment at that moment.
   - The visual asset is unusable (missing, broken, or non-loadable).

4) Slide-Level Verdict
   - The slide receives a PASS only if all voiceover segments PASS.
   - If any single segment FAILS, the entire slide verdict must be FAIL.
   - Be extremely strict and critical in your evaluation to ensure that the visuals are correctly aligned with the voiceover segments.

5) Failure Reporting Requirements
   For every failed visual, you MUST:
   - Identify the voiceover segment ID
   - Quote the exact voiceover text
   - List the failing Visual ID
   - Clearly state why the visual does not align with the voiceover
   - Describe the specific visual requirements that would be required for the segment to PASS

Output Format:
Always provide your output strictly in the following format:

<evaluation_breakdown>

Use this section as a structured reasoning and scratchpad space for you to evaluate visual–voiceover alignment for the slide.

- Slide Understanding: State in your own words what the slide is about and what the voiceover segments are trying to convey.
- Review of the assigned visuals: For each segment, list the assigned Visual IDs and briefly describe what is visibly shown in each visual (image or video).
- Visual Alignment Analysis: For each segment, analyze whether the assigned visuals correctly support the respective part of the voiceover segment. 
- Additional Analysis: Note any additional observations, thoughts or analysis that can help you arrive at the correct output and verdict.

(It is ok for this section to be quite verbose, long and detailed as long as it helps you arrive at the correct output.)

</evaluation_breakdown>

(Based on your above evaluation, provide your output in the following format)

<review>

<verdict>
PASS|FAIL
</verdict>

(If the slide verdict is FAIL, provide the details of the failed segments in the following format)
<failures>

<failure>
<segment_id>
(Provide the segment number of the failed segment. e.g. SEGMENT 1)
</segment_id>

<vo_text>
(Provide the exact portion of voiceover text that is not correctly supported by this visual.)
</vo_text>

<failing_visual_id>
(Provide the Visual ID of the assigned visual that does not correctly support the voiceover segment. e.g. S1V3)
</failing_visual_id>

<reason>
(Provide the reason why the assigned visual does not correctly support the voiceover segment.)
</reason>

<needed_visual>
(Describe the visual requirements that is needed to correctly support the failed voiceover segment for this criteria. Don't use words like "image" in this section since we are going to replace the faulty visual from a pool of image as well as video candidates. So prefer words like "visual" instead.)
</needed_visual>

</failure>

Repeat the <failure> block for each failed segment and its corresponding visual id. (Even if multiple visuals within the same segment fail, repeat the <failure> block separately for each failing visual.)

</failures>

</review>

(Use this exact XML format given above while providing your output)
"""


# Alingment prompt to use when we have 1 visual for the whole slide visual assingment strategy
ALIGNMENT_REVIEW_PROMPT_FOR_ONE_VISUAL_PER_SLIDE = """You are a Graphics Definition Review Agent specializing in the field of HVAC.

Criterion: Visual–Voiceover Alignment Accuracy

Definition: For the given slide, verify that the assigned visual clearly and directly shows what the slide is trying to convey.

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

This is the assigned visual for this whole slide:
<review_targets>
{review_targets}
</review_targets>

Note: Visual IDs follow the format "S(segment_number)V(visual_number)".

Instructions:
Follow the below evaluation rules to guide your evaluation:

1) Scope
   - Review the assigned visual for the whole slide.
   - The assigned visual is displayed on screen as the entire slide content is narrated.
   - Use the full slide content to understand the intended meaning of the slide.
   - Judge alignment only between that slide's intended meaning and the visual explicitly assigned to it.
   - Use only the provided slide content and the assigned visual; do not assume missing context beyond what is present in the slide.
   - The visual asset has a Visual ID assigned (for example, S1V1). Use this ID when listing any failures.
   - IMPORTANT: Know that we have been allowed to assign only one visual asset for this particular slide. So keep that in mind as you evaluate the alignment of the visual to the slide.

2) What counts as PASS for the slide
   - The assigned visual clearly shows what is described by the slide content.
   - The visual matches the specific meaning of the slide as spoken, not just the general topic of the slide.
   - The visual should be specific and clear enough that the exact object/action/detail in the slide is identifiable without guesswork.
   - The visual does not contradict the slide content or implies a different instructional idea.
   - The assigned visual provides enough visual evidence for a learner to understand the intended meaning of the slide.
   - The visual asset is usable (not missing, broken, or non-loadable).

3) What counts as FAIL for the slide
   The slide FAILS if any one of the following is true:
   - The visual shows something different than what the slide content describes.
   - The visual is generic, symbolic, or only loosely related, and does not clearly illustrate the main intended meaning for the slide.
   - The visual is too vague, distant, cluttered, or unclear to confidently identify the key detail being referenced.
   - The visual contradicts the slide content or implies a different instructional idea.
   - The assigned visual does not provide enough visual evidence for a learner to understand the slide.
   - The visual asset is unusable (missing, broken, or non-loadable).

4) Slide-Level Verdict
   - The slide receives a PASS only if the assigned visual is PASS for this criteria.
   - If the assigned visual FAILS, then you must assign a FAIL verdict to the slide.
   - Be extremely strict and critical in your evaluation to ensure that the visual is correctly aligned with the slide content.

5) Failure Reporting Requirements
   If your verdict for the slide is FAIL, you MUST:
   - List the Segment ID 
   - List the failing Visual ID
   - Clearly state why the visual does not align with the voiceover content of the slide
   - Describe the specific visual requirement that will be required for the slide to PASS.

Output Format:
Always provide your output strictly in the following format:

<evaluation_breakdown>

Use this section as a structured reasoning and scratchpad space for you to evaluate visual–voiceover alignment for the slide.

- Slide Understanding: State in your own words what the slide is about and what it is trying to convey.
- Review of the assigned visual: Briefly describe what is visibly shown in the assigned visual.
- Visual Alignment Analysis: Analyze whether the assigned visual correctly supports the main intended meaning of the slide.
- Additional Analysis: Note any additional observations, thoughts or analysis that can help you arrive at the correct output and verdict.

(It is ok for this section to be quite verbose, long and detailed as long as it helps you arrive at the correct output.)

</evaluation_breakdown>

(Based on your above evaluation, provide your output in the following format)

<review>

<verdict>
PASS|FAIL 
</verdict>

(If the slide verdict is FAIL, provide the details of the failed visual in the following format)

<failure>

<segment_id>
(Provide the segment number of the failed segment. e.g. SEGMENT 1)
</segment_id>

<vo_text>
(Provide the entire slide content text as it is.)
</vo_text>

<failing_visual_id>
(Provide the Visual ID of the assigned visual. e.g. S1V1)
</failing_visual_id>

<reason>
(Provide the reason why the assigned visual does not correctly support the slide content.)
</reason>

<needed_visual>
(Describe the visual requirement that is needed to correctly support the slide content for this criteria. Don't use words like "image" in this section since we are going to replace the faulty visual from a pool of image as well as video candidates. So prefer words like "visual" instead.)
</needed_visual>

</failure>

</review>

(Use this exact XML format given above while providing your output)
"""


# # Specificity review prompt to use when we have flexible or 1 visual per sentence visual assingment strategy
# SPECIFICITY_REVIEW_PROMPT = """You are a Graphics Definition Review Agent specializing in the field of HVAC.

# Criterion: Visual Specificity and Clarity

# Definition:
# For the given slide, verify that the assigned visual(s) for the given voiceover segment(s) show the correct object, component, action, condition, etc. with enough visual detail, focus, and clarity for a learner to easily identify exactly what the voiceover is referring to at that moment.

# Inputs:
# These are the inputs for your evaluation:

# <course_information>
# Course name: {course_name}
# Target audience: {target_audience}
# Topic name: {topic_name}
# Subtopic name: {subtopic_name}
# </course_information>

# <slide_information>
# Slide ID: {slide_id}
# Slide title: {slide_title}
# Slide content: "{slide_chunk}"
# </slide_information>

# These are the voiceover segments of this slide and the assigned visuals for each segment:
# <review_targets>
# {review_targets}
# </review_targets>

# Note: Visual IDs follow the format "S(segment_number)V(visual_number)". For example, "S1V3" means Segment 1, Visual 3 (the third visual assigned to segment 1 for its respecitve voiceover text), "S2V1" means Segment 2, Visual 1 (the first visual assigned to segment 2 for its respecitve voiceover text) and so on.

# Instructions:
# Follow the below evaluation rules to guide your evaluation:

# 1) Scope
#    - Review each voiceover segment independently as the primary evaluation unit.
#    - The assigned visuals are displayed on screen as the voiceover text for that segment is played/narrated.
#    - You may use the full slide content and the sequence of voiceover segments to understand the intended meaning of a segment.
#    - When interpreting any part of the voiceover sentences, you must use the surrounding text in the slide content to understand the context and the instructional intent. Do not interpret a sentence, phrase or a clause literally in isolation. Always derive the meaning of a sentence, phrase or a clause from the surrounding text in the slide content.
#    - Judge specificity and clarity only between that segment's intended meaning and the visuals explicitly assigned to it.
#    - Use only the provided assets and voiceover text; do not assume missing context beyond what is present in the slide.
#    - Each visual asset has a Visual ID (for example, S2V1). Use these IDs when listing any failures.

# 2) What counts as PASS for a segment
#    - The visual clearly shows the exact component, part, action, condition, or detail referenced in the voiceover segment.
#    - The visual removes ambiguity and does not require guesswork from the learner.
#    - If multiple visuals are assigned, together they provide sufficient clarity to identify the exact thing being described for the voiceover segment.
#    - A learner should be able to confidently point to the relevant detail in the visual while the voiceover is playing.

# 3) What counts as FAIL for a segment
#    A segment FAILS if any one of the following is true:
#    - The visual is too generic or vague.
#    - The visual does not clearly show the specific part, action, condition, detail, etc. mentioned in the segment.
#    - The framing is too distant, obstructed, cluttered, or unfocused to identify the required detail.
#    - The visual asset is unusable (missing, broken, or non-loadable).

# 4) Slide-Level Verdict
#    - The slide receives a PASS only if all voiceover segments PASS.
#    - If any single segment FAILS, the entire slide verdict must be FAIL.
#    - Be extremely strict and critical in your evaluation to ensure that the visuals are correctly specific and clear.
   
# 5) Failure Reporting Requirements
#    For every failed visual, you MUST:
#    - Identify the voiceover segment ID
#    - Quote the exact voiceover text
#    - List the failing Visual ID
#    - Clearly state why the visual lacks sufficient specificity or clarity for the voiceover
#    - Describe the specific visual requirements that would be required for the segment to PASS

# Output Format:
# Always provide your output strictly in the following format:

# <evaluation_breakdown>

# Use this section as a structured reasoning and scratchpad space for you to evaluate visual specificity and clarity for the slide.

# - Slide Understanding: State in your own words what the slide is about and what the voiceover segments are trying to convey.
# - Review of the assigned visuals: For each segment, list the assigned Visual IDs and briefly describe what is visibly shown in each visual (image or video).
# - Visual Specificity Analysis: For each segment, analyze whether the assigned visuals show the required specific detail clearly and unambiguously. Consider whether a learner can easily identify the exact thing being referenced without guessing, inference, or prior knowledge.
# - Additional Analysis: Note any additional observations, thoughts or analysis that can help you arrive at the correct output and verdict.

# (It is ok for this section to be quite verbose, long and detailed as long as it helps you arrive at the correct output.)

# </evaluation_breakdown>

# (Based on your above evaluation, provide your output in the following format)

# <review>

# <verdict>
# PASS|FAIL
# </verdict> 

# (If the slide verdict is FAIL, provide the details of the failed segments in the following format)
# <failures>

# <failure>
# <segment_id>
# (Provide the segment number of the failed segment. e.g. SEGMENT 1)
# </segment_id>

# <vo_text>
# (Provide the exact portion of voiceover text that is not clearly or specifically supported by this visual.)
# </vo_text>

# <failing_visual_id>
# (Provide the Visual ID of the assigned visual that lacks sufficient specificity or clarity. e.g. S1V3)
# </failing_visual_id>

# <reason>
# (Provide the reason why the assigned visual lacks sufficient specificity or clarity for the voiceover segment.)
# </reason>

# <needed_visual>
# (Describe the visual requirements that is needed to correctly support the failed voiceover segment for this criteria. Don't use words like "image" in this section since we are going to replace the faulty visual from a pool of image as well as video candidates. So prefer words like "visual" instead.)
# </needed_visual>

# </failure>

# Repeat the <failure> block for each failed segment and its corresponding visual id. (Even if multiple visuals within the same segment fail, repeat the <failure> block separately for each failing visual.)

# </failures>

# </review>

# (Use this exact XML format given above while providing your output)
# """


# # Specificity review prompt to use when we have 1 visual for the whole slide visual assingment strategy
# SPECIFICITY_REVIEW_PROMPT_FOR_ONE_VISUAL_PER_SLIDE = """You are a Graphics Definition Review Agent specializing in the field of HVAC.

# Criterion: Visual Specificity and Clarity

# Definition:
# For the given slide, verify that the assigned visual clearly shows the correct object, component, action, condition, etc. with enough visual detail, focus, and clarity for a learner to easily identify exactly what the slide is trying to convey.

# Inputs:
# These are the inputs for your evaluation:

# <course_information>
# Course name: {course_name}
# Target audience: {target_audience}
# Topic name: {topic_name}
# Subtopic name: {subtopic_name}
# </course_information>

# <slide_information>
# Slide ID: {slide_id}
# Slide title: {slide_title}
# Slide content: "{slide_chunk}"
# </slide_information>

# This is the assigned visual for this whole slide:
# <review_targets>
# {review_targets}
# </review_targets>

# Note: Visual IDs follow the format "S(segment_number)V(visual_number)".

# Instructions:
# Follow the below evaluation rules to guide your evaluation:

# 1) Scope
#    - Review the assigned visual for the whole slide.
#    - The assigned visual is displayed on screen as the entire slide content is narrated.
#    - Use the full slide content to understand the intended meaning of the slide.
#    - Judge specificity and clarity only between that slide's intended meaning and the visual explicitly assigned to it.
#    - Use only the provided slide content and the assigned visual; do not assume missing context beyond what is present in the slide.
#    - The visual asset has a Visual ID assigned (for example, S1V1). Use this ID when listing any failures.
#    - IMPORTANT: Know that we have been allowed to assign only one visual asset for this particular slide. So keep that in mind as you evaluate the specificity and clarity of the visual to the slide.

# 2) What counts as PASS for the slide
#   - The assigned visual clearly shows the correct object, component, action, condition, etc. with enough visual detail, focus, and clarity for a learner to easily identify exactly what the slide is trying to convey.
#   - The visual removes ambiguity and does not require guesswork from the learner.
#   - The assigned visual provides enough visual evidence for a learner to understand the intended meaning of the slide.
#   - The visual asset is usable (not missing, broken, or non-loadable).

# 3) What counts as FAIL for the slide
#    The slide FAILS if any one of the following is true:
#    - The visual is too generic or vague.
#    - The visual does not clearly show the specific part, action, condition, detail, etc. mentioned in the slide content.
#    - The framing is too distant, obstructed, cluttered, or unfocused to identify the required detail.
#    - The visual asset is unusable (missing, broken, or non-loadable).
   
# 4) Slide-Level Verdict
#    - The slide receives a PASS only if the assigned visual is PASS for this criteria.
#    - If the assigned visual FAILS, then you must assign a FAIL verdict to the slide.

# 5) Failure Reporting Requirements
#    If your verdict for the slide is FAIL, you MUST:
#    - List the Segment ID
#    - List the failing Visual ID
#    - Clearly state why the visual does not clearly show the specific part, action, condition, detail, etc. mentioned in the slide content.
#    - Describe the specific visual requirement that will be required for the slide to PASS.

# Output Format:
# Always provide your output strictly in the following format:

# <evaluation_breakdown>

# Use this section as a structured reasoning and scratchpad space for you to evaluate visual specificity and clarity for the slide.

# - Slide Understanding: State in your own words what the slide is about and what it is trying to convey.
# - Review of the assigned visual: Briefly describe what is visibly shown in the assigned visual.
# - Visual Specificity Analysis: Analyze whether the assigned visual correctly shows the correct object, component, action, condition, etc. with enough visual detail, focus, and clarity for a learner to easily identify exactly what the slide is trying to convey.
# - Additional Analysis: Note any additional observations, thoughts or analysis that can help you arrive at the correct output and verdict.

# (It is ok for this section to be quite verbose, long and detailed as long as it helps you arrive at the correct output.)

# </evaluation_breakdown>

# (Based on your above evaluation, provide your output in the following format)

# <review>

# <verdict>
# PASS|FAIL
# </verdict>

# (If the slide verdict is FAIL, provide the details of the failed segment in the following format)

# <failure>

# <segment_id>
# (Provide the segment number of the failed segment. e.g. SEGMENT 1)
# </segment_id>

# <vo_text>
# (Provide the entire slide content text as it is.)
# </vo_text>

# <failing_visual_id>
# (Provide the Visual ID of the assigned visual. e.g. S1V1)
# </failing_visual_id>

# <reason>
# (Provide the reason why the assigned visual lacks sufficient specificity or clarity for the slide content.)
# </reason>

# <needed_visual>
# (Describe the visual requirements that is needed to correctly support the slide content for this criteria. Don't use words like "image" in this section since we are going to replace the faulty visual from a pool of image as well as video candidates. So prefer words like "visual" instead.)
# </needed_visual>

# </failure>

# </review>

# (Use this exact XML format given above while providing your output)
# """


# REDUNDANCY_REVIEW_PROMPT = """You are a Graphics Definition Review Agent specializing in the field of HVAC.

# Criterion: Visual Redundancy and Variety

# Definition:
# For the given set of slides, verify that a specific visual asset (identified by its URL) is not overly repetitive across voiceover segments and slides without clear instructional reason. Visual reuse is allowed when it supports continuity or learning, but unnecessary or excessive repetition that reduces instructional value or visual engagement should be flagged.

# Inputs:
# These are the inputs for your evaluation:

# <course_information>
# Course name: {course_name}
# Target audience: {target_audience}
# Topic name: {topic_name}
# </course_information>

# This is the visual that has been repeated multiple times:
# <repeated_visual_url>
# {repeated_visual_url}
# </repeated_visual_url>

# IMPORTANT: The URL above has been identified as appearing more than 3 times across the slides in this topic. Your task is to evaluate whether this specific visual asset is being used redundantly and whether it should be replaced in some or all of its occurrences.

# These are the voiceover segments and assigned visuals across the topic slide group being reviewed.
# <review_targets>
# {review_targets}
# </review_targets>

# Note: Visual IDs follow the format "S(segment_number)V(visual_number)". For example, "S1V3" means Segment 1, Visual 3 (the third visual assigned to segment 1 for its respecitve voiceover text), "S2V1" means Segment 2, Visual 1 (the first visual assigned to segment 2 for its respecitve voiceover text) and so on.

# Instructions:
# Follow the below evaluation rules to guide your evaluation:

# 1) Scope
#    - Review all voiceover segments listed in <review_targets> together as a group.
#    - Identify all the Visual IDs where the repeated visual URL appears. Examine the respective voiceover texts to which the same repeated visuals are assigned.
#    - The assigned visuals are displayed on screen as the voiceover text for that segment is played/narrated.
#    - The scope of review is across multiple slides within a topic and its segments.
#    - You may consider slide order and proximity when judging redundancy (for example, repetition across consecutive or nearby slides).
#    - Use only the provided voiceover text and assigned visuals; do not assume missing context beyond what is provided.
#    - Each visual asset has a Visual ID (for example, S2V1). Use these IDs when listing any failures.
#    - Your evaluation should focus specifically on whether the repeated visual URL is being used redundantly and whether it should be replaced in some or all occurrences.

# 2) What counts as ACCEPTABLE reuse (PASS)
#    The repeated visual asset identified in <repeated_visual_url> is acceptable when:
#    - The same component or object must be shown again for instructional continuity across the segments where it appears.
#    - Reuse reinforces understanding of a key concept that remains the focus across slides.
#    - Reuse clearly serves a learning purpose and does not make the slides feel visually repetitive or lazy.
#    - The repetition is instructionally justified and adds value to the learning experience.

# 3) What counts as REDUNDANCY (FAIL)
#    The repeated visual asset identified in <repeated_visual_url> FAILS and must be flagged for replacement in some or all occurrences if any one of the following is true:
#    - The same visual asset is reused across multiple slides or segments without clear instructional need.
#    - Consecutive or nearby slides feel visually identical when a different example or view would reasonably improve clarity or engagement.
#    - The repetition does not add new instructional value and could confuse, bore, or disengage the learner.
#    - A different example, view, visual or variation would reasonably improve clarity, engagement, or instructional quality, but the same visual is reused instead.
#    - The visual appears in contexts where different visuals would be more appropriate, even if the repetition serves some instructional purpose.

# 4) Group-Level Verdict
#    - The group receives a PASS only if no visuals are flagged as needing replacement due to unnecessary or excessive repetition.
#    - If one or more visuals are identified as redundant and requiring replacement, the overall verdict must be FAIL.

# 4) When to FAIL
# - One or more voiceover part(s) uses a visual that is clearly repetitive with no instructional need.
# - The repetition reduces clarity or engagement when a different visual example should be used.

# 5) Failure Reporting Requirements
#   For every occurrence of the repeated visual asset (identified in <repeated_visual_url>) that is identified as unnecessarily repetitive and should be replaced, you MUST:
#    - Create a separate <failure> block
#    - Identify the slide and segment where this specific visual asset appears
#    - Quote the exact voiceover text where the visual is used
#    - List the Visual ID that needs replacement (this Visual ID must correspond to the repeated visual URL)
#    - Identify where else the same visual asset (the repeated URL) is being reused and needs to be replaced as well
#    - Clearly explain why the repetition of this specific visual asset is not instructionally justified
#    - Describe what kind of alternative or varied visual should be used instead
#    - Note: You should evaluate each occurrence of the repeated visual individually. Some occurrences may be instructionally justified and acceptable, while others may be redundant and need replacement. However, the overall verdict will be FAIL if ANY occurrence needs replacement, and PASS only if ALL occurrences are acceptable.

# Output Format:
# Always provide your output strictly in the following format:

# <evaluation_breakdown>

# Use this section as a structured reasoning and scratchpad space for you to evaluate visual redundancy and variety for the specific repeated visual asset across all its occurrences.

# - Repeated Visual Asset: Confirm the visual asset URL you are evaluating (from <repeated_visual_url>) and describe what is shown in this visual.
# - Slide Group Understanding: Briefly explain what the group of slides and all its voiceover segments is covering instructionally, and then note down what visuals are assigned for each of the voiceover segments by looking at the assigned visuals for each of the voiceover segments.
# - Occurrence Mapping: List all locations (Slide/segment/Visual ID) where the repeated visual asset appears. For each occurrence, note the assigned voiceover text and context.
# - Reuse Analysis: Analyze whether the repetition of this specific visual asset across these locations is instructionally justified or unnecessarily repetitive. Consider:
#   * Whether the same visual is needed for continuity or learning reinforcement
#   * Whether different visuals would improve clarity, engagement, or instructional quality
#   * Whether the repetition makes slides feel visually identical or lazy
#   * The proximity and order of slides where the visual appears
# - Redundancy Judgment: For each occurrence of the repeated visual asset, determine whether it is acceptable (instructionally justified) or needs replacement (redundant). Explain your judgment clearly for each occurrence. Based on these individual judgments, determine the overall verdict: PASS if ALL occurrences are acceptable, FAIL if ANY of the occurrence(s) needs replacement.
# - Additional Analysis: Note any additional observations, thoughts or analysis that can help you arrive at the correct output and verdict.

# (It is ok for this section to be quite verbose, long and detailed as long as it helps you arrive at the correct output.)

# </evaluation_breakdown>

# (Based on your above evaluation, provide your output in the following format)

# <review>

# <verdict>
# PASS|FAIL
# </verdict>

# (If the slide group verdict is FAIL, provide the details of the failed visuals in the following format)

# <failures>

# <failure>

# <slide_segment_id>
# (Provide the slide and segment identifier where this visual appears, e.g. SLIDE_2_SEGMENT_1)
# </slide_segment_id>

# <vo_text>
# (Provide the exact portion of the voiceover text where this visual is used.)
# </vo_text>

# <failing_visual_id>
# (Provide the Visual ID that needs to be replaced, e.g. S2V3)
# </failing_visual_id>

# <reason>
# Explain why this specific occurrence of the repeated visual asset is unnecessarily repetitive or lacks sufficient variety. Explicitly mention where else this same visual asset appears (for example: another slide ID, segment, or nearby visual) along with the Voiceover text to which it is assigned, and explain why replacing this occurrence is instructionally appropriate. 
# </reason>

# <needed_visual>
# (Describe what kind of alternative or varied visual should be used instead. Don't use words like "image" in this section since we are going to replace the faulty visual from a pool of image as well as video candidates. So prefer words like "visual" instead.)
# </needed_visual>

# </failure>

# Repeat one <failure> block per visual that needs to be replaced.

# </failures>

# </review>
# (Use this exact XML format given above while providing your output)
# """


# Revision prompt to use when we have flexible or 1 visual per sentence visual assingment strategy
REVISION_PROMPT = """You are a Graphics Definition Revision Agent specializing in the field of HVAC. Your task is to revise and correct the graphics definition for a SINGLE voiceover (VO) segment when one or more of its currently assigned visuals have failed review checks, by selecting the most appropriate visual from the provided candidate image and video pools.
You will be given the voiceover segment, the slide and course context for reference, the visuals currently assigned to this segment, explicit review feedback describing what is wrong and what is required instead, and a pool of candidate visuals from which to select the most appropriate visual to replace the failed visual(s) based on the feedback.

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

This is the explicit review feedback describing what is wrong and what is required instead:
<feedback>
{feedback}
</feedback>

These are the candidate images and videos from which to select the most appropriate visual to replace the failed visuals based on the feedback:

<candidates>

<image_candidates>
{image_candidates}
</image_candidates>

<video_candidates>
{video_candidates}
</video_candidates>

</candidates>

Instructions:

1) Scope and Revision Responsibility
   - Your task is to revise the graphics definition for this single voiceover (VO) segment only.
   - Use the slide content and the specific voiceover segment to understand the context and the instructional intent.
   - The assigned visuals are displayed on screen as the voiceover text for that segment is played/narrated.
   - When interpreting any part of the voiceover sentences, you must use the surrounding text in the slide content to understand the context and the instructional intent, before selecting visuals for the respective parts of the slide. Do not interpret a sentence, phrase or a clause literally in isolation. Always derive the meaning of a sentence, phrase or a clause from the surrounding text in the slide content.
   - Identify the specific visual requirement implied by the feedback.
   - Apply changes only to the specific visual(s) identified as failing in the provided feedback.
   - Select replacement visuals only from the provided candidate image and video pools.

2) Candidate Evaluation and Visual Replacement
   - Carefully review all provided image and video candidates for this voiceover segment.
   - Evaluate each candidate only against the specific visual requirement described in the feedback.
   - Select the candidate that most directly and clearly satisfies the feedback while remaining aligned with the voiceover segment.
   - Replace only the visual(s) identified as failing; do not modify other visuals if they are not identified as failing.
   - If no candidate fully satisfies the feedback, select the closest acceptable alternative
   - Ensure the replacement visual(s) are instructionally clear and effective for the voiceover segment.

3) Visual Form and Usage Constraints
   - You may select images, video clips with timestamps, or still frames extracted from videos, using only the provided candidate visuals.
   - Choose the visual form (image, video clip, or still frame) that most clearly satisfies the feedback while fitting within the narration timing of the relevant part(s) of the voiceover segment.
   - When you find both a video clip and a still image that equally satisfies the feedback for any part of the voiceover sentence, prefer using the video clip, since motion can add useful context. This is a guiding preference, not a strict rule - do not prioritize a video clip over an image if the video is only partially relevant, loosely related, or less effective than the still image at supporting the narration and addressing the feedback.
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

Use this section as a structured reasoning and scratchpad space for you to address the feedback and select the most appropriate visual(s) to replace the failed visual(s). Use it to document your observations, reasoning, and decision process. Provide the following sections:

1) Voiceover Sentence Understanding
   - Briefly explain, in your own words, what the voiceover sentence is communicating and what part of the sentence the feedback is addressing. Use the slide content to resolve any references, pronouns, or implied meaning if needed.

2) Feedback Interpretation
    - Briefly summarize what the feedback indicates is wrong with the current visual(s). 
    - Identify how many visual(s) are currently assigned to the voiceover segment and how many need to be replaced.
    - Clearly state the specific visual requirement that must be satisfied by the revision.

3) Image Candidates Scan
   - Create a numbered list of all provided image candidates and briefly describe what you see in each of the image candidate.

4) Video Candidates Scan
   - Create a numbered list of all provided video candidates and briefly describe what you see in each of the video candidate. 
   - Focus on explaining the visual content of the video, and not what is being spoken in the video. Split the list into two sections: one for videos from which video clips (with timestamps) or still frames as images can be used, and one for videos from which ONLY still frames as images can be used (NOT playable video clips with timestamps).

5) Candidate Fit Analysis
   - Compare the candidates against the feedback requirement(s). 
   - Identify which candidate(s) most directly satisfy the requirement and why. 
   - If multiple candidates partially match, reason about which one is the closest acceptable match.
   - If no candidate fully satisfies the feedback, determine the closest acceptable alternative.

6) Video Timestamp / Frame Selection Thinking (only if selecting video to replace any of the failed visual(s))
   - If selecting a playable video clip if you find a relevant one that best satisfies the feedback requirement:
   - Determine exactly which portion of the video is visually relevant and will satisfy the feedback.
   - Plan the correct start and end timestamps for that portion, ensuring the selected video clip can realistically fit within the narration timing of the specific part of the voiceover segment being addressed by the feedback.
   - If selecting a still frame from a video, determine the moment (single timestamp) that captures the required visual so that it can be used as a static image.

7) Additional Analysis:
   - Note any additional observations, thoughts or analysis that can help you arrive at the correct output and decision that addresses all the given feedback for all the failed visual(s) of this voiceover segment.

(It is ok for this section to be quite verbose, long and detailed as long as it helps you arrive at the correct output.)

</evaluation_breakdown>

(Based on your above evaluation, provide your output in the following format. Remember, only provide the replacement visuals for the failed visual(s) based on the feedback for this voiceover segment. Dont include the visuals that are not failed. Meaning if for example you are revising segment 2, and there are 3 visuals assigned to that segment, and 2 of them are failed, and 1 is not failed, you should only provide the replacement visuals for the 2 failed visuals. Dont include the non-failed visual in your output.)

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
(Provide a concise justification for why this replacement visual is the most appropriate for this voiceover part)
</selection_justification>

</visual>

Repeat one <visual> block per visual that needs to be replaced based on the feedback for this voiceover segment.

</replacement_visuals>

</output>

(Remember to strictly use this exact format for the output with the XML tags and structure, and nothing else.)
"""


# Revision prompt to use when we have 1 visual for the whole slide visual assingment strategy
REVISION_PROMPT_FOR_ONE_VISUAL_PER_SLIDE = """You are a Graphics Definition Revision Agent specializing in the field of HVAC. Your task is to revise and correct the graphics definition for the given slide when its currently assigned visual has failed review checks, by selecting the most appropriate visual from the provided candidate image and video pools.
You will be given the slide, the course context for reference, the visual currently assigned to this slide, explicit review feedback describing what is wrong and what is required instead, and a pool of candidate visuals from which to select the most appropriate visual to replace the failed visual based on the feedback.

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

This is the explicit review feedback describing what is wrong and what is required instead:
<feedback>
{feedback}
</feedback>

These are the candidate images and videos from which to select the most appropriate visual to replace the failed visual based on the feedback:

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

1) Scope and Revision Responsibility
    - Your task is to revise the graphics definition for this slide.
    - Use the whole slide content to understand the intended meaning of the slide.
    - The assigned visual is displayed on screen as the entire slide content is narrated.
    - Identify the specific visual requirement implied by the feedback.
    - Select replacement visual only from the provided candidate image and video pools.
    - IMPORTANT: Know that we have been allowed to assign only one visual asset for this particular slide. So keep that in mind as you select the replacement visual.

2) Candidate Evaluation and Visual Replacement
   - Carefully review all provided image and video candidates for this slide.
   - Evaluate each candidate against the specific visual requirement described in the feedback.
   - Select the candidate that most directly and clearly satisfies the feedback while remaining aligned with the slide content.
   - If no candidate fully satisfies the feedback, select the closest acceptable alternative.
   - Ensure the replacement visual is instructionally clear and effective for the slide.

3) Visual Form and Usage Constraints
   - You may select an image, video clip with timestamps, or a still frame extracted from video, using only the provided candidate visuals.
   - Choose the visual form (image, video clip, or still frame) that most clearly satisfies the feedback while fitting within the narration timing of the slide content.
   - When you find both a video clip and a still image that equally satisfies the feedback, prefer using the video clip, since motion can add useful context. This is a guiding preference, not a strict rule - do not prioritize a video clip over an image if the video is only partially relevant, loosely related, or less effective than the still image at supporting the narration and addressing the feedback.
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

Use this section as a structured reasoning and scratchpad space for you to address the feedback and select the most appropriate visual to replace the failed visual. Use it to document your observations, reasoning, and decision process. Provide the following sections:

1) Slide Understanding
   - Briefly explain, in your own words, what the slide is about and what it is trying to convey.

2) Feedback Interpretation
   - Briefly summarize what the feedback indicates is wrong with the current visual.
   - Clearly state the specific visual requirement that must be satisfied by the revision.

3) Image Candidates Scan
   - Create a numbered list of all provided image candidates and briefly describe what you see in each of the image candidate.

4) Video Candidates Scan
   - Create a numbered list of all provided video candidates and briefly describe what you see in each of the video candidate.
   - Focus on explaining the visual content of the video, and not what is being spoken in the video. Split the list into two sections: one for videos from which video clips (with timestamps) or still frames as images can be used, and one for videos from which ONLY still frames as images can be used (NOT playable video clips with timestamps).

5) Candidate Fit Analysis
   - Compare the candidates against the feedback requirement. 
   - Identify which candidate most directly satisfy the requirement and why. 
   - If multiple candidates partially match, reason about which one is the closest acceptable match.
   - If no candidate fully satisfies the feedback, determine the closest acceptable alternative.

6) Video Timestamp / Frame Selection Thinking (only if selecting video as replacement visual)
   - If selecting a playable video clip if you find a relevant one that best satisfies the feedback requirement:
   - Determine exactly which portion of the video is visually relevant and will satisfy the feedback.
   - Plan the correct start and end timestamps for that portion, ensuring the selected video clip can realistically fit within the narration timing of the slide content.
   - If selecting a still frame from a video, determine the moment (single timestamp) that captures the required visual so that it can be used as a static image.

7) Additional Analysis:
   - Note any additional observations, thoughts or analysis that can help you arrive at the correct output and decision that addresses the given feedback for the slide.

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
(Provide a concise justification for why this replacement visual is the most appropriate for this slide)
</selection_justification>

</visual>

</replacement_visual>

</output>

(Remember to strictly use this exact format for the output with the XML tags and structure, and nothing else.)
"""


# Search query revision prompt to use when we have flexible or 1 visual per sentence visual assingment strategy 
SEARCH_QUERY_REVISION_PROMPT = """You are a Search Query Generator Agent specializing in HVAC instructional visuals.
Your task is to generate search queries for a specific parts of the given voiceover segment based on the feedback and the visual needs it describes. The goal of these queries is to retrieve visual assets(image or video) that accurately and clearly support the instructional intent of the specific voiceover part(s), based on the provided context and feedback needs.

Inputs:

These are the inputs:

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

<voiceover_segment>
{vo_text}
</voiceover_segment>

<feedback>
{feedback}
</feedback>

Instructions:

1. Scope and Query Responsibility
   - Your task is to generate search queries only for the specific part(s) of the voiceover segment implicated by the feedback.
   - Use the full voiceover segment and slide context to understand the instructional intent of the specific voiceover part(s) and resolve any references, pronouns, or implied meaning if needed.
   - Do not generate queries for parts of the voiceover segment that are not associated with failed visuals.

2. Feedback-Driven Targeting
   - Carefully analyze the feedback to identify what is missing, incorrect, unclear, or insufficient in the failed visual(s).
   - Translate the described visual requirement into concrete, searchable visual concepts (object, component, action, condition, orientation, state, context, etc.).
   - Ensure each query directly reflects what the feedback says is needed, not the general topic of the slide.

3. Query Generation Criteria
   - Generate search queries that are directly relevant to the specific visual needs described in the feedback.
   - Focus on visual elements that are necessary to support the instructional intent of the specific voiceover part(s).
   - Use concrete object names, components, or diagrams that are likely to appear in images.
   - Phrase queries the way images and videos are commonly searched for or labeled.
   - Use short, keyword-based phrases.
   - Prefer phrasing typical of textbooks, technical diagrams, stock photos and videos, or browser image and video searches.
   - Ensure each query adds value.
   - Queries should not be near-duplicates of each other.
   - Each query should represent a slightly different but relevant visual angle.
   - Generate exactly 3 search queries and ensure that collectively, all the queries cover all distinct visual requirements implied by all the given feedback.

Output Format:

Always provide your output strictly in the following format:

<output>

<evaluation_breakdown>

1. Voiceover Meaning
- Briefly explain, in your own words, what the voiceover sentence is communicating. 

2. Feedback Interpretation
- Explain which part(s) of the voiceover sentence the feedback is addressing.

3. Query Planning
- Reason about the kinds of search queries that would best retrieve visuals to support this sentence based on the feedback.
- Consider how such queries are typically phrased.

</evaluation_breakdown>

(Based on your above evaluation, provide your final output of search queries)

<queries>
- Query 1
- Query 2
...
(Provide exactly 3 search queries to address all the given feedback)
</queries>

</output>

(Remember to strictly use this exact format for the output with the XML tags and structure, and nothing else.)
"""


# Search query revision prompt to use when we have 1 visual per slide visual assingment strategy
SEARCH_QUERY_REVISION_PROMPT_FOR_ONE_VISUAL_PER_SLIDE = """You are a Search Query Generator Agent specializing in HVAC instructional visuals.
Your task is to generate search queries for the given slide based on the feedback and the visual needs it describes. The goal of these queries is to retrieve visual assets(image or video) that accurately and clearly support the instructional intent of the slide, based on the provided context and feedback needs.

Inputs:

These are the inputs:

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

<feedback>
{feedback}
</feedback>

Instructions:

1. Scope and Query Responsibility
   - Your task is to generate search queries for the given slide based on the feedback and the visual needs it describes.
   - Use the slide content to understand the instructional intent of the slide.

2. Feedback-Driven Targeting
   - Carefully analyze the feedback to identify what is missing, incorrect, unclear, or insufficient in the failed visual.
   - Translate the described visual requirement into concrete, searchable visual concepts (object, component, action, condition, orientation, state, context, etc.).
   - Ensure each query directly reflects what the feedback says is needed, not the general topic of the slide.

3. Query Generation Criteria
   - Generate search queries that are directly relevant to the specific visual needs described in the feedback.
   - Focus on visual element that is necessary to support the instructional intent of the slide.
   - Use concrete object names, components, or diagrams that are likely to appear in images.
   - Phrase queries the way images and videos are commonly searched for or labeled.
   - Use short, keyword-based phrases.
   - Prefer phrasing typical of textbooks, technical diagrams, stock photos and videos, or browser image and video searches.
   - Ensure each query adds value.
   - Queries should not be near-duplicates of each other.
   - Each query should represent a slightly different but relevant visual angle.
   - Generate 1-4 search queries and ensure that collectively, all the queries cover all the visual requirements implied by the given feedback.

Output Format:

Always provide your output strictly in the following format:

<output>

<evaluation_breakdown>

1. Voiceover Meaning
- Briefly explain, in your own words, what the slide is about and what it is trying to convey. 

2. Feedback Interpretation
- Explain what the feedback is saying is wrong with the current visual.

3. Query Planning
- Reason about the kinds of search queries that would best retrieve visuals to support the slide based on the feedback.
- Consider how such queries are typically phrased.

</evaluation_breakdown>

(Based on your above evaluation, provide your final output of search queries)

<queries>
- Query 1
- Query 2
...
(Provide 1-4 search queries to address the given feedback)
</queries>

</output>

(Remember to strictly use this exact format for the output with the XML tags and structure, and nothing else.)
"""

SEGMENT_HEADER_RE = re.compile(r'(={5,}\s*\nSEGMENT\s+(\d+)\s*\n={5,}\s*\n)', re.IGNORECASE)


def _safe_str(value):
    """
    Safely convert a value to string, returning empty string if None.
    
    :param value: Value to convert to string
    :return: String representation of value or empty string if None
    """
    
    if value is None:
        return ""
    return str(value)


def _extract_tag(text, tag):
    """
    Extract content from XML tag in text.
    
    :param text: Text containing XML tags
    :param tag: Tag name to extract content from
    :return: Extracted content or empty string if not found
    """
    
    match = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", text or "", re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""


def _parse_segment_marker(text):
    """
    Parse segment number from segment marker text.
    
    :param text: Text containing segment marker (e.g., "SEGMENT 1")
    :return: Segment number as integer or None if not found
    """
    
    match = re.search(r"SEGMENT\s*(\d+)", text or "", re.IGNORECASE)
    return int(match.group(1)) if match else None


def parse_segmented_text(text):
    """
    Parse segmented text into dictionary mapping segment numbers to lines.
    
    :param text: Text with segment markers (---SEGMENT_N---)
    :return: Dictionary mapping segment numbers to lists of lines
    """
    
    if not text or text.strip() == "" or text == "nan":
        return {}
    segments: Dict[int, List[str]] = {}
    parts = re.split(r"---SEGMENT_(\d+)---", text)
    for i in range(1, len(parts), 2):
        if i + 1 >= len(parts):
            continue
        segment_num = int(parts[i])
        content = parts[i + 1].strip()
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        segments[segment_num] = lines
    return segments


def build_segmented_text(segments):
    """
    Build segmented text from dictionary of segments.
    
    :param segments: Dictionary mapping segment numbers to lists of lines
    :return: Formatted text with segment markers
    """
    
    blocks = []
    for segment_num in sorted(segments.keys()):
        lines = segments[segment_num]
        block_lines = [f"---SEGMENT_{segment_num}---"] + lines
        blocks.append("\n".join(block_lines))
    return "\n\n".join(blocks)


def replace_segment_block(text, segment_num, new_lines):
    """
    Replace a specific segment block in segmented text.
    
    :param text: Original segmented text
    :param segment_num: Segment number to replace
    :param new_lines: New lines to replace the segment with
    :return: Updated segmented text
    """
    
    segments = parse_segmented_text(text)
    segments[segment_num] = new_lines
    return build_segmented_text(segments)


def parse_final_graphics_definition(text):
    """
    Parse final graphics definition text into dictionary of segments.
    
    :param text: Final graphics definition text with segment headers
    :return: Dictionary mapping segment numbers to segment text
    """
    
    if not text or text.strip() == "" or text == "nan":
        return {}
    matches = list(SEGMENT_HEADER_RE.finditer(text))
    if not matches:
        return {}
    segments: Dict[int, str] = {}
    for idx, match in enumerate(matches):
        segment_num = int(match.group(2))
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        segments[segment_num] = text[start:end].strip()
    return segments


def build_final_graphics_definition(segments):
    """
    Build final graphics definition text from dictionary of segments.
    
    :param segments: Dictionary mapping segment numbers to segment text
    :return: Formatted final graphics definition text
    """
    
    blocks = [segments[segment_num] for segment_num in sorted(segments.keys())]
    return "\n\n".join(blocks)


def parse_visual_steps(segment_text):
    """
    Parse visual steps from segment text.
    
    :param segment_text: Segment text containing visual step information
    :return: List of dictionaries with visual step data
    """
    
    if not segment_text:
        return []
    cleaned = SEGMENT_HEADER_RE.sub("", segment_text, count=1).strip()
    if not cleaned:
        return []
    chunks = re.split(r"(?m)^\s*-{2,}\s*$", cleaned)
    steps: List[Dict[str, str]] = []
    for chunk in chunks:
        lines = [line.strip() for line in chunk.splitlines() if line.strip()]
        if not lines:
            continue
        data: Dict[str, str] = {}
        for line in lines:
            if line.startswith("When VO:"):
                data["voiceover_part"] = line.split("When VO:", 1)[1].strip().strip('"')
            elif line.startswith("Visual Instructions:"):
                data["visual_instruction"] = line.split("Visual Instructions:", 1)[1].strip()
            elif line.startswith("Graphics to use:"):
                data["asset"] = line.split("Graphics to use:", 1)[1].strip()
            elif line.startswith("Selection Justification:"):
                data["selection_justification"] = line.split("Selection Justification:", 1)[1].strip()
        if data:
            steps.append(data)
    return steps


def check_file_in_folder(drive, file_id, folder_id):
    """
    Check if a Drive file belongs to a specific folder.
    
    :param drive: Google Drive instance
    :param file_id: Drive file ID to check
    :param folder_id: Target folder ID
    :return: True if file is in the folder, False otherwise
    """
    
    if not drive or not file_id or not folder_id:
        return False
    
    try:
        # Get file metadata
        file_obj = drive.CreateFile({'id': file_id})
        file_obj.FetchMetadata()
        
        # Check if the folder_id is in the file's parents
        # Parents is a list of dicts like [{'id': 'folder_id_1'}, {'id': 'folder_id_2'}]
        parents = file_obj.get('parents', [])
        parent_ids = [p.get('id') if isinstance(p, dict) else p for p in parents]
        is_in_folder = folder_id in parent_ids
        
        return is_in_folder
    except Exception as e:
        print(f"⚠️ Error checking if file {file_id} is in folder {folder_id}: {e}")
        import traceback
        traceback.print_exc()
        return False


def add_snapshot_label_to_drive_links(graphics_definition_text, drive, target_folder_id=video_frames_drive_folder_id):
    """
    Add "(snapshot)" label to Drive links that belong to the target folder. Finds all "Graphics to use:" lines with Drive URLs, checks if the file is in the target folder, and adds "(snapshot)" on a new line after the URL if it is.
    
    :param graphics_definition_text: Text string containing graphics definition
    :param drive: Google Drive instance
    :param target_folder_id: Target folder ID to check (default: video_frames_drive_folder_id)
    :return: Modified text with "(snapshot)" labels added, or original text if no changes
    """
    
    if not graphics_definition_text or not graphics_definition_text.strip():
        return graphics_definition_text
    
    if not drive:
        print("⚠️ Drive instance not available, skipping snapshot label check")
        return graphics_definition_text
    
    try:
        import re
        
        drive_url_pattern = re.compile(
            r'https?://drive\.google\.com/file/d/([a-zA-Z0-9_-]+)\S*',
            re.IGNORECASE,
        )
        
        lines = graphics_definition_text.split('\n')
        result_lines: list = []
        matches_found = 0
        labels_added = 0
        
        for i, line in enumerate(lines):
            result_lines.append(line)
            
            if not re.match(r'\s*Graphics to use:', line, re.IGNORECASE):
                continue
            
            m = drive_url_pattern.search(line)
            if not m:
                continue
            
            matches_found += 1
            file_id = m.group(1)
            
            # Already labeled on the next line — skip
            next_line = lines[i + 1].strip() if i + 1 < len(lines) else ""
            if next_line == "(snapshot)":
                continue
            
            if check_file_in_folder(drive, file_id, target_folder_id):
                labels_added += 1
                result_lines.append("(snapshot)")
        
        if matches_found > 0:
            if labels_added > 0:
                print(f"Added (snapshot) label to {labels_added} Drive link(s) from target folder")
            else:
                print(f"Found {matches_found} Drive link(s) but none needed (snapshot) labels")
        
        modified_text = '\n'.join(result_lines)
        return modified_text if modified_text != graphics_definition_text else graphics_definition_text
        
    except Exception as e:
        print(f"⚠️ Error adding snapshot labels to Drive links: {e}")
        import traceback
        traceback.print_exc()
        return graphics_definition_text


def build_segment_visual_map(voiceover_text, final_graphics_definition, visual_assignment_strategy=None, slide_chunk=None):
    """
    Build a map of voiceover segments to their corresponding visual steps.
    
    :param voiceover_text: The voiceover text to parse.
    :param final_graphics_definition: The final graphics definition to parse.
    :param visual_assignment_strategy: Visual assignment strategy (optional, for handling entire slide case)
    :param slide_chunk: Full slide content (optional, for handling entire slide case)
    :return: A dictionary mapping segment numbers to their corresponding visual steps.
    """

    segments = parse_segments_from_voiceover(voiceover_text)
    
    # Handle "1 Visual for the whole Slide" case where voiceover_text might be empty
    if visual_assignment_strategy == "1 Visual for the whole Slide" and not segments:
        # Use slide_chunk as voiceover for SEGMENT 1
        if slide_chunk and slide_chunk.strip() and slide_chunk != "nan":
            segments = [(1, slide_chunk.strip())]
    
    graphics_segments = parse_final_graphics_definition(final_graphics_definition)
    result: Dict[int, Dict[str, object]] = {}
    print(f"  Building segment visual map: {len(segments)} voiceover segments, {len(graphics_segments)} graphics segments")
    for segment_num, vo_text in segments:
        segment_text = graphics_segments.get(segment_num, "")
        visual_steps = parse_visual_steps(segment_text)
        for idx, step in enumerate(visual_steps, start=1):
            step["visual_id"] = f"S{segment_num}V{idx}"
        result[segment_num] = {
            "vo_text": vo_text,
            "visual_steps": visual_steps,
            "segment_text": segment_text,
        }
        print(f"Segment {segment_num}: {len(visual_steps)} visual step(s)")
    return result


def initialize_revision_tracking(segments_map):
    """
    Initialize revision tracking structure for all visuals in the slide.

    :param segments_map: Map of segment_num -> segment data with visual_steps
    :return: Initialized tracking structure with original URLs
    """
    
    tracking: Dict[str, Dict[str, Dict[int, str]]] = {}
    
    for segment_num, segment_data in segments_map.items():
        visual_steps = segment_data.get("visual_steps", [])
        for step in visual_steps:
            visual_id = step.get("visual_id", "")
            asset_url = step.get("asset", "")
            if visual_id and asset_url:
                if visual_id not in tracking:
                    tracking[visual_id] = {
                        "alignment": {0: asset_url},  # 0 = original
                        "specificity": {}
                    }
                else:
                    # Update original if not set
                    if 0 not in tracking[visual_id]["alignment"]:
                        tracking[visual_id]["alignment"][0] = asset_url
    
    return tracking


def update_revision_tracking(tracking, criterion_name, loop_num, replaced_visual_ids, segments_map):
    """
    Update revision tracking with replacements from a specific loop.

    :param tracking: Current tracking structure
    :param criterion_name: "alignment" or "specificity"
    :param loop_num: Loop number (1 to MAX_REVIEW_ATTEMPTS + MAX_REGEN_ATTEMPTS)
    :param replaced_visual_ids: Map of segment_num -> list of visual_ids that were replaced
    :param segments_map: Current segments map with latest visuals
    :return: Updated tracking structure
    """
    
    # Get all replaced visual IDs across all segments
    all_replaced_ids = set()
    for visual_ids in replaced_visual_ids.values():
        all_replaced_ids.update(visual_ids)
    
    # Update tracking for all visuals
    for segment_num, segment_data in segments_map.items():
        visual_steps = segment_data.get("visual_steps", [])
        for step in visual_steps:
            visual_id = step.get("visual_id", "")
            asset_url = step.get("asset", "")
            if not visual_id:
                continue
            
            # Initialize if not exists
            if visual_id not in tracking:
                tracking[visual_id] = {
                    "alignment": {0: asset_url},
                    "specificity": {}
                }
            
            # If this visual was replaced in this loop, record the new URL
            if visual_id in all_replaced_ids:
                if criterion_name not in tracking[visual_id]:
                    tracking[visual_id][criterion_name] = {}
                tracking[visual_id][criterion_name][loop_num] = asset_url
            # If not replaced, mark as "No replacement" (we'll handle formatting later)
            else:
                if criterion_name not in tracking[visual_id]:
                    tracking[visual_id][criterion_name] = {}
                # Don't overwrite if already set
                if loop_num not in tracking[visual_id][criterion_name]:
                    tracking[visual_id][criterion_name][loop_num] = None  # None = No replacement
    
    return tracking


def format_revision_tracking(tracking):
    """
    Format revision tracking structure into the readable string format.

    :param tracking: Tracking structure
    :return: Formatted string
    """
    
    if not tracking:
        return ""
    
    # Calculate total number of loops dynamically
    total_loops = MAX_REVIEW_ATTEMPTS + MAX_REGEN_ATTEMPTS
    
    lines = []
    # Sort visual IDs (S1V1, S1V2, S2V1, etc.)
    sorted_visual_ids = sorted(tracking.keys(), key=lambda x: (int(re.search(r'S(\d+)', x).group(1)) if re.search(r'S(\d+)', x) else 0, 
                                                               int(re.search(r'V(\d+)', x).group(1)) if re.search(r'V(\d+)', x) else 0))
    
    for visual_id in sorted_visual_ids:
        lines.append(visual_id)
        lines.append("")
        
        # Alignment section
        lines.append("Alignment")
        alignment_data = tracking[visual_id].get("alignment", {})
        original_url = alignment_data.get(0, "")
        if original_url:
            lines.append(f"Original visual - {original_url}")
        else:
            lines.append("Original visual - (not found)")
        
        # Loops 1 to total_loops (dynamic)
        for loop_num in range(1, total_loops + 1):
            url = alignment_data.get(loop_num)
            if url:
                lines.append(f"Visual after loop {loop_num} - {url}")
            else:
                lines.append(f"Visual after loop {loop_num} - No replacement")
        
        lines.append("")
        
        # Specificity section intentionally hidden from sheet output because
        # specificity loop execution is currently disabled.
        # lines.append("Specificity")
        # specificity_data = tracking[visual_id].get("specificity", {})
        # specificity_original = original_url
        # if alignment_data:
        #     alignment_loops = [k for k in alignment_data.keys() if k > 0 and alignment_data[k] is not None]
        #     if alignment_loops:
        #         last_alignment_loop = max(alignment_loops)
        #         specificity_original = alignment_data[last_alignment_loop]
        # if specificity_original:
        #     lines.append(f"Original - {specificity_original}")
        # else:
        #     lines.append("Original - (not found)")
        # for loop_num in range(1, total_loops + 1):
        #     url = specificity_data.get(loop_num)
        #     if url:
        #         lines.append(f"Visual after loop {loop_num} - {url}")
        #     else:
        #         lines.append(f"Visual after loop {loop_num} - No replacement")
        # lines.append("")
        lines.append("")
    
    return "\n".join(lines)


def is_youtube_url(url):
    """
    Check if URL is a YouTube URL.
    
    :param url: URL to check
    :return: True if URL is a YouTube URL, False otherwise
    """
    
    if not url:
        return False
    url_lower = url.lower()
    return "youtube.com" in url_lower or "youtu.be" in url_lower


def build_asset_parts(visual_id, asset_url, drive):
    """
    Build multimodal parts for a visual asset (image or video).
    
    :param visual_id: Visual ID identifier
    :param asset_url: URL of the asset
    :param drive: Google Drive instance
    :return: List of multimodal parts
    """
    
    parts: List[types.Part] = [types.Part(text=f"Asset {visual_id}: {asset_url}")]
    if is_youtube_url(asset_url):
        clip_url, start_seconds, end_seconds = parse_video_url_timestamps(asset_url)
        if not clip_url:
            clip_url = convert_watch_url_to_embed_url(asset_url)
        parts.append(build_video_part(clip_url or asset_url, start_seconds, end_seconds))
        return parts

    pil_image = load_image_from_url(asset_url, drive, visual_id)
    if pil_image:
        buffered = BytesIO()
        pil_image.convert("RGB").save(buffered, format="JPEG")
        parts.append(types.Part(inline_data=types.Blob(mime_type="image/jpeg", data=buffered.getvalue())))
    else:
        parts.append(types.Part(text=f"[Asset {visual_id} could not be loaded]"))
        print(f"WARNING: Failed to load image asset {visual_id} from {asset_url[:80]}...")
    return parts


def build_visual_part_only(asset_url, drive):
    """
    Build only the visual part (image or video) without text label.
    
    :param asset_url: The URL of the asset to build the visual part for.
    :param drive: The drive to use to load the asset.
    :return: The visual part.
    """
    
    if is_youtube_url(asset_url):
        clip_url, start_seconds, end_seconds = parse_video_url_timestamps(asset_url)
        if not clip_url:
            clip_url = convert_watch_url_to_embed_url(asset_url)
        return build_video_part(clip_url or asset_url, start_seconds, end_seconds)

    pil_image = load_image_from_url(asset_url, drive, "candidate")
    if pil_image:
        buffered = BytesIO()
        pil_image.convert("RGB").save(buffered, format="JPEG")
        return types.Part(inline_data=types.Blob(mime_type="image/jpeg", data=buffered.getvalue()))
    else:
        print(f"WARNING: Failed to load image asset from {asset_url[:80]}...")
        return None


def build_current_visual_parts(visual_id, asset_url, drive):
    """
    Build multimodal parts for current assigned visual.
    
    :param visual_id: The ID of the visual to build the parts for.
    :param asset_url: The URL of the asset to build the parts for.
    :param drive: The drive to use to load the asset.
    :return: The parts.
    """
    
    parts: List[types.Part] = []
    parts.append(types.Part(text=f"Current Assigned Visual {visual_id}: {asset_url}"))
    
    visual_part = build_visual_part_only(asset_url, drive)
    if visual_part:
        parts.append(visual_part)
    else:
        parts.append(types.Part(text=f"[Current visual {visual_id} could not be loaded]"))
    
    return parts


def build_review_targets(segments_map, segment_nums, slide_id):
    """
    Build review targets text for specified segments.
    
    :param segments_map: Map of segment_num -> segment data
    :param segment_nums: List of segment numbers to include
    :param slide_id: Slide identifier
    :return: Formatted review targets string
    """
    
    lines: List[str] = []
    for segment_num in segment_nums:
        segment = segments_map.get(segment_num)
        if not segment:
            continue
        lines.append(f"{slide_id} SEGMENT {segment_num}")
        lines.append(f"VO: {segment.get('vo_text', '')}")
        visual_steps = segment.get("visual_steps", [])
        if not visual_steps:
            lines.append("No visuals assigned.")
        else:
            for step in visual_steps:
                lines.append(
                    f"{step.get('visual_id')} | When VO: {step.get('voiceover_part', '')} | "
                    f"Asset: {step.get('asset', '')}"
                )
        lines.append("")
    return "\n".join(lines).strip()


def build_review_targets_for_whole_slide(segments_map, slide_id, slide_chunk):
    """
    Build review targets for "1 Visual for the whole Slide" strategy.
    Formats as a single visual for the entire slide content.
    
    :param segments_map: Map of segment_num -> segment data
    :param slide_id: Slide identifier
    :param slide_chunk: Full slide content
    :return: Formatted review targets string
    """
    lines: List[str] = []
    # For "1 Visual for the whole Slide", there should be only segment 1
    segment = segments_map.get(1)
    if not segment:
        return "No visuals assigned."
    
    visual_steps = segment.get("visual_steps", [])
    if not visual_steps:
        return "No visuals assigned."
    
    # Format as single visual for entire slide
    for step in visual_steps:
        lines.append(
            f"{step.get('visual_id')} | When VO: {slide_chunk} | "
            f"Asset: {step.get('asset', '')}"
        )
    
    return "\n".join(lines).strip()


def build_review_targets_for_visuals(segments_map, visual_ids_by_segment, slide_id):
    """
    Build review targets including only specific visual IDs for each segment.
    
    :param segments_map: Map of segment_num -> segment data
    :param visual_ids_by_segment: Map of segment_num -> list of visual_ids to include
    :param slide_id: Slide identifier
    :return: Formatted review targets string
    """
    lines: List[str] = []
    for segment_num, visual_ids in visual_ids_by_segment.items():
        segment = segments_map.get(segment_num)
        if not segment:
            continue
        lines.append(f"{slide_id} SEGMENT {segment_num}")
        lines.append(f"VO: {segment.get('vo_text', '')}")
        visual_steps = segment.get("visual_steps", [])
        if not visual_steps:
            lines.append("No visuals assigned.")
        else:
            # Only include visuals that are in the visual_ids list
            for step in visual_steps:
                visual_id = step.get('visual_id', '')
                if visual_id in visual_ids:
                    lines.append(
                        f"{visual_id} | When VO: {step.get('voiceover_part', '')} | "
                        f"Asset: {step.get('asset', '')}"
                    )
        lines.append("")
    return "\n".join(lines).strip()


def build_assets_for_segments(segments_map, segment_nums, drive):
    """
    Build asset parts for all visuals in specified segments.
    
    :param segments_map: Map of segment_num -> segment data
    :param segment_nums: List of segment numbers to process
    :param drive: Google Drive instance
    :return: List of multimodal parts
    """
    
    parts: List[types.Part] = []
    for segment_num in segment_nums:
        segment = segments_map.get(segment_num)
        if not segment:
            continue
        for step in segment.get("visual_steps", []):
            asset_url = step.get("asset")
            if not asset_url:
                continue
            visual_id = step.get("visual_id", "")
            asset_parts = build_asset_parts(visual_id, asset_url, drive)
            parts.extend(asset_parts)
            if len(asset_parts) > 1:  # More than just the text part means asset was loaded
                print(f"Loaded asset {visual_id}: {asset_url[:80]}...")
    return parts


def build_assets_for_visuals(segments_map, visual_ids_by_segment, drive):
    """
    Build asset parts for only specific visual IDs in each segment.
    
    :param segments_map: Map of segment_num -> segment data
    :param visual_ids_by_segment: Map of segment_num -> list of visual_ids to include
    :param drive: Drive instance for loading assets
    :return: List of types.Part objects
    """
    
    parts: List[types.Part] = []
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
            asset_parts = build_asset_parts(visual_id, asset_url, drive)
            parts.extend(asset_parts)
            if len(asset_parts) > 1:  # More than just the text part means asset was loaded
                print(f" Loaded asset {visual_id}: {asset_url[:80]}...")
    return parts


def invoke_gemini_multimodal(parts, llm, temperature=0.1, conversation_history=None, max_retries=5):
    """
    Invoke Gemini multimodal API with optional conversation history and retry logic for quota errors.
    
    :param parts: List of parts (text, images, videos) for the current message
    :param llm: LLM model name
    :param temperature: Temperature setting
    :param conversation_history: Optional list of previous Content objects in the conversation
    :param max_retries: Maximum number of retry attempts for quota errors
    :return: Tuple of (response_text, updated_conversation_history)
    """
    
    model_mapping = {
        "gemini_3_flash_thinking": "gemini-3-flash-preview",
        "gemini_3_flash": "gemini-3-flash-preview",
        "gemini_3_pro": "gemini-3-pro",  
        "gemini_2.5_flash": "gemini-2.5-flash",
        "gemini-3-flash-preview": "gemini-3-flash-preview",
    }
    actual_model = model_mapping.get(llm, llm)
    client = genai.Client()
    
    # Build conversation contents
    contents = []
    if conversation_history:
        contents.extend(conversation_history)
    
    # Add current user message
    contents.append(types.Content(role="user", parts=parts))
    
    # Enable thinking mode for gemini_3_flash_thinking using ThinkingConfig
    if llm == "gemini_3_flash_thinking" and "gemini-3" in actual_model:
        # Use ThinkingConfig to enable thinking mode for complex review tasks
        config = types.GenerateContentConfig(
            temperature=temperature,
            thinking_config=types.ThinkingConfig(
                include_thoughts=True
            )
        )
    else:
        config = types.GenerateContentConfig(temperature=temperature)
    
    # Retry logic for quota errors
    retries = 0
    while retries < max_retries:
        try:
            response = client.models.generate_content(
                model=actual_model,
                contents=contents,
                config=config,
            )
            try:
                token_usage = extract_token_usage(response)
                input_tokens = token_usage["input_tokens"]
                output_tokens = token_usage["output_tokens"]
                log_token_usage(
                    llm=llm,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    log_file="token_usage_log.csv",
                )
            except Exception as e:
                print(f"Token usage logging failed: {e}")
            
            # Extract response text
            response_text = ""
            if hasattr(response, "text") and response.text:
                response_text = response.text
            elif getattr(response, "candidates", None):
                first_candidate = response.candidates[0]
                if getattr(first_candidate, "content", None) and first_candidate.content.parts:
                    part = first_candidate.content.parts[0]
                    if hasattr(part, "text"):
                        response_text = part.text
            
            # Build updated conversation history
            updated_history = []
            if conversation_history:
                updated_history.extend(conversation_history)
            
            # Add user message and model response to history
            updated_history.append(types.Content(role="user", parts=parts))
            if response_text:
                updated_history.append(types.Content(role="model", parts=[types.Part(text=response_text)]))
            
            return response_text or str(response), updated_history
            
        except Exception as e:
            error_str = str(e)
            # Check if it's a 429 quota error
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "quota" in error_str.lower():
                # Try to extract retry delay from error message
                retry_delay = None
                # First, try to access error details if available (for Google Genai SDK exceptions)
                try:
                    if hasattr(e, 'error') and isinstance(e.error, dict):
                        error_dict = e.error
                        # Check for retryDelay in details
                        if 'details' in error_dict:
                            for detail in error_dict.get('details', []):
                                if isinstance(detail, dict) and 'retryDelay' in detail:
                                    retry_delay = float(detail['retryDelay'])
                                    break
                except (AttributeError, KeyError, ValueError, TypeError):
                    pass
                
                # If not found in structured format, parse from string
                if retry_delay is None and ("retry in" in error_str.lower() or "retrydelay" in error_str.lower()):
                   
                    delay_match = re.search(r'retry\s+in\s+([\d.]+)\s*s', error_str, re.IGNORECASE)
                    if delay_match:
                        retry_delay = float(delay_match.group(1))
                    else:
                        # Try to find retryDelay in JSON-like structure
                        delay_match = re.search(r'["\']retryDelay["\']\s*:\s*["\']?([\d.]+)', error_str, re.IGNORECASE)
                        if delay_match:
                            retry_delay = float(delay_match.group(1))
                        else:
                            # Try to find it in the error message format: "retryDelay": "32s"
                            delay_match = re.search(r'retryDelay["\']?\s*[:=]\s*["\']?([\d.]+)s?', error_str, re.IGNORECASE)
                            if delay_match:
                                retry_delay = float(delay_match.group(1))
                
                if retry_delay is None:
                    # Fallback to exponential backoff
                    retry_delay = min(2 ** retries * 5, 60)  # Cap at 60 seconds
                else:
                    # Add a small buffer to the API-specified delay to be safe
                    retry_delay = retry_delay + 1.0
                
                if retries < max_retries - 1:
                    print(f"  ⚠️  Quota exceeded (429). Waiting {retry_delay:.1f}s before retry {retries + 1}/{max_retries}...")
                    time.sleep(retry_delay)
                    retries += 1
                    continue
                else:
                    print(f"  ❌ Max retries ({max_retries}) exceeded for quota error.")
                    raise
            else:
                # For non-quota errors, raise immediately
                raise
    
    # Should never reach here, but just in case
    raise Exception(f"Failed after {max_retries} retries")


def parse_review_response(text):
    """
    Parse review response text to extract verdict and failures.
    
    Handles both formats:
    - Original: <failures><failure>...</failure></failures> (multiple failures)
    - Entire slide: <failure>...</failure> (single failure, no wrapper)
    
    :param text: Review response text with XML tags
    :return: Tuple of (verdict, failures) where failures is a list of failure dictionaries
    """
    
    verdict = _extract_tag(text, "verdict").upper() or "FAIL"
    failures: List[Dict[str, str]] = []
    
    # Check if <failures> wrapper exists (original format)
    if re.search(r'<failures>', text, re.IGNORECASE):
        # Original format: extract failures from within <failures> wrapper
        failures_block = _extract_tag(text, "failures")
        if failures_block:
            failure_blocks = re.findall(r"<failure>(.*?)</failure>", failures_block, re.DOTALL | re.IGNORECASE)
        else:
            failure_blocks = []
    else:
        # Entire slide format: single <failure> without wrapper
        failure_blocks = re.findall(r"<failure>(.*?)</failure>", text or "", re.DOTALL | re.IGNORECASE)
    
    for block in failure_blocks:
        # Extract segment_key with fallback: try slide_segment_id first (for redundancy), then segment_id (for alignment/specificity)
        segment_key = _extract_tag(block, "slide_segment_id") or _extract_tag(block, "segment_id")
        failures.append({
            "segment_id": _extract_tag(block, "segment_id"),
            "segment_key": segment_key,
            "vo_text": _extract_tag(block, "vo_text"),
            "failing_visual_ids": _extract_tag(block, "failing_visual_id"),
            "reason": _extract_tag(block, "reason"),
            "needed_visual": _extract_tag(block, "needed_visual"),
            "reused_with": _extract_tag(block, "reused_with"),
        })
    return verdict, failures


@traceable(
    metadata={
        "agent_name": "graphics_definition_v2",
        "step_name": "Review Agent",
        "function_name": "review_slide_segments",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous")
    }
)
def review_slide_segments(prompt_template, course_name, target_audience, topic_name, subtopic_name, slide_id, slide_title, slide_chunk, segments_map, segment_nums, drive, llm, conversation_history=None, criterion_name="review", visual_assignment_strategy="Flexible, let the agent decide"):
    """
    Review slide segments using multimodal LLM.
    
    :param prompt_template: Prompt template string
    :param course_name: Course name
    :param target_audience: Target audience
    :param topic_name: Topic name
    :param subtopic_name: Subtopic name
    :param slide_id: Slide identifier
    :param slide_title: Slide title
    :param slide_chunk: Full slide content
    :param segments_map: Map of segment_num -> segment data
    :param segment_nums: List of segment numbers to review
    :param drive: Google Drive instance
    :param llm: Language model to use
    :param conversation_history: Optional conversation history
    :param criterion_name: Name of the review criterion (e.g., "alignment", "specificity")
    :param visual_assignment_strategy: Visual assignment strategy
    :return: Tuple of (verdict, failures, response_text, updated_conversation_history)
    """
    
    print(f"  Reviewing {len(segment_nums)} segment(s): {segment_nums}")
    # Use different format for "1 Visual for the whole Slide" strategy
    if visual_assignment_strategy == "1 Visual for the whole Slide":
        review_targets = build_review_targets_for_whole_slide(segments_map, slide_id, slide_chunk)
    else:
        review_targets = build_review_targets(segments_map, segment_nums, slide_id)
    asset_parts = build_assets_for_segments(segments_map, segment_nums, drive)
    print(f"Loaded {len(asset_parts)} asset part(s) for review")
    prompt = prompt_template.format(
        course_name=course_name,
        target_audience=target_audience,
        topic_name=topic_name,
        subtopic_name=subtopic_name,
        slide_id=slide_id,
        slide_title=slide_title,
        slide_chunk=slide_chunk,
        review_targets=review_targets,
    )
    strategy_label = f" [Strategy: {visual_assignment_strategy}]" if visual_assignment_strategy else ""
    # print(f"\n{'='*80}")
    # print(f"📝 FORMATTED {criterion_name.upper()} REVIEW PROMPT ({slide_id}){strategy_label}:")
    # print(f"{'='*80}")
    # print(prompt)
    # print(f"{'='*80}\n")
    print(f"Starting {criterion_name} review for {slide_id}...")
    print(f"Multimodal parts to be sent:")
    for idx, part in enumerate(asset_parts, 1):
        if hasattr(part, 'text') and part.text:
            print(f"Part {idx} (text): {part.text}...")
        elif hasattr(part, 'inline_data'):
            print(f"Part {idx} (image/video data)")
    print(f"Invoking Gemini multimodal review (model: {llm})...")
    response_text, updated_history = invoke_gemini_multimodal(
        asset_parts + [types.Part(text=prompt)], 
        llm=llm,
        conversation_history=conversation_history
    )
    print(f"\nReview response ({slide_id}):\n{response_text}\n")
    verdict, failures = parse_review_response(response_text)
    print(f"Verdict: {verdict}, Failures: {len(failures)}")
    if failures:
        for failure in failures:
            print(f"    - {failure.get('segment_id', 'Unknown')}: {failure.get('reason', 'No reason provided')[:100]}")
    return verdict, failures, response_text, updated_history


@traceable(
    metadata={
        "agent_name": "graphics_definition_v2",
        "step_name": "Review Agent",
        "function_name": "review_slide_segments_followup",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous")
    }
)
def review_slide_segments_followup(course_name, target_audience, topic_name, subtopic_name, slide_id, slide_title, slide_chunk, segments_map, visual_ids_by_segment, old_asset_urls_by_visual_id, previous_feedback, drive, llm, conversation_history, criterion_name="review", visual_assignment_strategy="Flexible, let the agent decide"):
    """
    Follow-up review in the same conversation, reviewing only revised visuals.
    
    :param visual_ids_by_segment: Map of segment_num -> list of visual_ids that were replaced
    :param old_asset_urls_by_visual_id: Map of visual_id -> old_asset_url (the failed visual before replacement)
    :param previous_feedback: The feedback from the previous review
    :param conversation_history: The conversation history from previous review
    :param criterion_name: Name of the review criterion (e.g., "alignment", "specificity")
    :param visual_assignment_strategy: Visual assignment strategy
    :return: Tuple of (verdict, failures, response_text, updated_conversation_history)
    """
    
    segment_nums = list(visual_ids_by_segment.keys())
    print(f"Follow-up review for {len(segment_nums)} segment(s): {segment_nums}")
    print(f"Reviewing only replaced visuals: {visual_ids_by_segment}")
    # Use different format for "1 Visual for the whole Slide" strategy
    if visual_assignment_strategy == "1 Visual for the whole Slide":
        review_targets = build_review_targets_for_whole_slide(segments_map, slide_id, slide_chunk)
    else:
        review_targets = build_review_targets_for_visuals(segments_map, visual_ids_by_segment, slide_id)
    
    # Build asset parts for both old (failed) and new (replacement) visuals
    asset_parts: List[types.Part] = []
    
    # First, add old failed visuals with clear labeling
    for segment_num, visual_ids in visual_ids_by_segment.items():
        for visual_id in visual_ids:
            old_asset_url = old_asset_urls_by_visual_id.get(visual_id)
            if old_asset_url:
                print(f"Loading OLD failed visual {visual_id}: {old_asset_url[:80]}...")
                # Create clear label showing this is the old failed visual
                old_label = types.Part(text=f"OLD FAILED VISUAL - Visual ID: {visual_id} | Original URL: {old_asset_url}")
                asset_parts.append(old_label)
                # Load only the visual part (no duplicate label)
                old_visual_part = build_visual_part_only(old_asset_url, drive)
                if old_visual_part:
                    asset_parts.append(old_visual_part)
                else:
                    asset_parts.append(types.Part(text=f"[OLD FAILED VISUAL {visual_id} could not be loaded]"))
    
    # Then, add new replacement visuals with clear labeling
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
            print(f"Loading NEW replacement visual {visual_id}: {asset_url[:80]}...")
            # Create clear label showing this is the new replacement visual
            new_label = types.Part(text=f"NEW REPLACEMENT VISUAL - Visual ID: {visual_id} | Replacement URL: {asset_url}")
            asset_parts.append(new_label)
            # Load only the visual part (no duplicate label)
            new_visual_part = build_visual_part_only(asset_url, drive)
            if new_visual_part:
                asset_parts.append(new_visual_part)
            else:
                asset_parts.append(types.Part(text=f"[NEW REPLACEMENT VISUAL {visual_id} could not be loaded]"))
    
    old_count = len([p for p in asset_parts if hasattr(p, 'text') and p.text and 'OLD FAILED VISUAL' in p.text])
    new_count = len([p for p in asset_parts if hasattr(p, 'text') and p.text and 'NEW REPLACEMENT VISUAL' in p.text])
    print(f"Loaded {len(asset_parts)} asset part(s) for follow-up review ({old_count} old visuals, {new_count} new visuals)")
    
    followup_prompt = f"""Based on your previous review feedback, I have revised the visuals for the following segments. Please review the revised visuals again to determine if they now pass the review criteria.

Previous Feedback:
{previous_feedback}

These are the revised voiceover segments and their updated visuals:
{review_targets}

You will see both the OLD FAILED visuals and the NEW REPLACEMENT visuals for each visual that was replaced. The labels clearly indicate which visual ID each asset belongs to and whether it is the old failed version or the new replacement version.

Please review the NEW REPLACEMENT visuals using the same criteria as before. Compare them to the corresponding OLD FAILED visuals to ensure the issues have been addressed. Output your review in the same format as before:
- <verdict>PASS|FAIL</verdict>
- If FAIL, provide <failure> blocks for any visuals that still fail

Be very strict in your evaluation. Do not give a PASS verdict if any visual still fails the review criteria even after multiple rounds of revisions. The verdict of "PASS" should only be asssigned after you are completely satisfied that the revised visual(s) pass the review criteria.

Remember: Striclty use the same output format as before while reviewing the NEW REPLACEMENT visuals and providing your output. Do not provide any additional text or commentary."""
    
    strategy_label = f" [Strategy: {visual_assignment_strategy}]" if visual_assignment_strategy else ""
    # print(f"\n{'='*80}")
    # print(f"📝 FORMATTED {criterion_name.upper()} FOLLOW-UP REVIEW PROMPT ({slide_id}){strategy_label}:")
    # print(f"{'='*80}")
    # print(followup_prompt)
    # print(f"{'='*80}\n")
    print(f"Starting {criterion_name} follow-up review for {slide_id}...")
    print(f"Multimodal parts to be sent:")
    for idx, part in enumerate(asset_parts, 1):
        if hasattr(part, 'text') and part.text:
            print(f"    Part {idx} (text): {part.text}...")
        elif hasattr(part, 'inline_data'):
            print(f"    Part {idx} (image/video data)")
    print(f"Invoking Gemini multimodal follow-up review (model: {llm})...")
    response_text, updated_history = invoke_gemini_multimodal(
        asset_parts + [types.Part(text=followup_prompt)], 
        llm=llm,
        conversation_history=conversation_history
    )
    print(f"\nFollow-up review response ({slide_id}):\n{response_text}\n")
    verdict, failures = parse_review_response(response_text)
    print(f"Verdict: {verdict}, Failures: {len(failures)}")
    if failures:
        for failure in failures:
            print(f"    - {failure.get('segment_id', 'Unknown')}: {failure.get('reason', 'No reason provided')[:100]}")
    return verdict, failures, response_text, updated_history


@traceable(
    metadata={
        "agent_name": "graphics_definition_v2",
        "step_name": "Review Agent",
        "function_name": "review_topic_segments",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous")
    }
)
def review_topic_segments(course_name, target_audience, topic_name, subtopic_name, group_segments, target_keys, drive, llm, repeated_visual_url, conversation_history=None):
    """
    Review topic segments for visual redundancy across multiple slides.
    
    :param course_name: Course name
    :param target_audience: Target audience
    :param topic_name: Topic name
    :param subtopic_name: Subtopic name
    :param group_segments: Dictionary mapping slide_id -> segments_map
    :param target_keys: List of segment keys to evaluate
    :param drive: Google Drive instance
    :param llm: Language model to use
    :param repeated_visual_url: The specific URL that is being repeated and reviewed
    :param conversation_history: Optional conversation history
    :return: Tuple of (verdict, failures, response_text, updated_conversation_history)
    """
    
    total_segments = sum(len(segments_map) for segments_map in group_segments.values())
    print(f"Reviewing redundancy across {len(group_segments)} slide(s), {total_segments} total segment(s)")
    if target_keys:
        print(f"Target segments: {len(target_keys)}")
    lines: List[str] = []
    for slide_id, segments_map in group_segments.items():
        for segment_num, segment in segments_map.items():
            segment_key = f"{slide_id}_SEGMENT_{segment_num}"
            lines.append(f"{segment_key}")
            lines.append(f"VO: {segment.get('vo_text', '')}")
            for step in segment.get("visual_steps", []):
                lines.append(
                    f"{step.get('visual_id')} | When VO: {step.get('voiceover_part', '')} | "
                    f"Asset: {step.get('asset', '')}"
                )
            lines.append("")
    review_targets = "\n".join(lines).strip()
    asset_parts: List[types.Part] = []
    for segments_map in group_segments.values():
        for segment_num in segments_map.keys():
            asset_parts.extend(build_assets_for_segments(segments_map, [segment_num], drive))
    print(f"Loaded {len(asset_parts)} asset part(s) for redundancy review")
    if target_keys:
        target_block = "TARGET SEGMENTS TO EVALUATE (These are the segments that have the same redundant visual assigned to their respective voiceover parts): \n" + "\n".join(target_keys) + "\n\n"
        review_targets = target_block + review_targets
    prompt = REDUNDANCY_REVIEW_PROMPT.format(
        course_name=course_name,
        target_audience=target_audience,
        topic_name=topic_name,
        repeated_visual_url=repeated_visual_url,  # The specific URL that is being repeated
        review_targets=review_targets,
    )
    print(f"Starting redundancy review for {topic_name}...")
    # print(f"\n{'='*80}")
    # print(f"\n{'='*80}")
    # print(f"📝 FORMATTED REDUNDANCY REVIEW PROMPT (Topic: {topic_name}):")
    # print(f"{'='*80}")
    # print(prompt)
    # print(f"{'='*80}\n")
    print(f"Multimodal parts to be sent:")
    for idx, part in enumerate(asset_parts, 1):
        if hasattr(part, 'text') and part.text:
            print(f"    Part {idx} (text): {part.text}...")
        elif hasattr(part, 'inline_data'):
            print(f"    Part {idx} (image/video data)")
    print(f"Invoking Gemini multimodal redundancy review (model: {llm})...")
    response_text, updated_history = invoke_gemini_multimodal(
        asset_parts + [types.Part(text=prompt)], 
        llm=llm,
        conversation_history=conversation_history
    )
    print(f"\nRedundancy review response (Topic: {topic_name}):\n{response_text}\n")
    verdict, failures = parse_review_response(response_text)
    if target_keys:
        failures = [f for f in failures if f.get("segment_key") in target_keys]
        if failures:
            verdict = "FAIL"
    print(f"Verdict: {verdict}, Failures: {len(failures)}")
    if failures:
        for failure in failures:
            print(f"- {failure.get('segment_key', 'Unknown')}: {failure.get('reason', 'No reason provided')[:100]}")
    return verdict, failures, response_text, updated_history


@traceable(
    metadata={
        "agent_name": "graphics_definition_v2",
        "step_name": "Review Agent",
        "function_name": "review_topic_segments_followup",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous")
    }
)
def review_topic_segments_followup(course_name, target_audience, topic_name, subtopic_name, group_segments, target_keys, previous_feedback, drive, llm, repeated_visual_url, conversation_history):
    """
    Follow-up redundancy review in the same conversation, reviewing only revised visuals.
    
    :param course_name: Course name
    :param target_audience: Target audience
    :param topic_name: Topic name
    :param subtopic_name: Subtopic name
    :param group_segments: Dictionary mapping slide_id -> segments_map
    :param target_keys: List of segment keys to evaluate
    :param previous_feedback: The feedback from the previous review
    :param drive: Google Drive instance
    :param llm: Language model to use
    :param repeated_visual_url: The specific URL that is being repeated and reviewed
    :param conversation_history: The conversation history from previous review
    :return: Tuple of (verdict, failures, response_text, updated_conversation_history)
    """
    
    total_segments = sum(len(segments_map) for segments_map in group_segments.values())
    print(f"Follow-up redundancy review across {len(group_segments)} slide(s), {total_segments} total segment(s)")
    if target_keys:
        print(f"Target segments: {len(target_keys)}")
    lines: List[str] = []
    for slide_id, segments_map in group_segments.items():
        for segment_num, segment in segments_map.items():
            segment_key = f"{slide_id}_SEGMENT_{segment_num}"
            lines.append(f"{segment_key}")
            lines.append(f"VO: {segment.get('vo_text', '')}")
            for step in segment.get("visual_steps", []):
                lines.append(
                    f"{step.get('visual_id')} | When VO: {step.get('voiceover_part', '')} | "
                    f"Asset: {step.get('asset', '')}"
                )
            lines.append("")
    review_targets = "\n".join(lines).strip()
    asset_parts: List[types.Part] = []
    for segments_map in group_segments.values():
        for segment_num in segments_map.keys():
            asset_parts.extend(build_assets_for_segments(segments_map, [segment_num], drive))
    print(f"Loaded {len(asset_parts)} asset part(s) for follow-up redundancy review")
    if target_keys:
        target_block = "TARGET SEGMENTS TO EVALUATE:\n" + "\n".join(target_keys) + "\n\n"
        review_targets = target_block + review_targets
    
    followup_prompt = f"""Based on your previous review feedback, I have revised the visuals for the following segments. Please review the revised visuals again to determine if they now pass your redundancy review criteria.

Repeated Visual URL: {repeated_visual_url}

Previous Feedback:
{previous_feedback}

These are the revised voiceover segments and their updated visuals:
{review_targets}

Please review these revised visuals using the same redundancy criteria as before. Focus on whether the repeated visual asset (URL: {repeated_visual_url}) is still being used redundantly or if the revisions have addressed the redundancy issues. Output your review in the same format as before:
- <verdict>PASS|FAIL</verdict>
- If FAIL, provide <failure> blocks for any visuals that still fail

Be very strict in your evaluation. Do not give a PASS verdict if any visual still fails the review criteria even after multiple rounds of revisions. The verdict of "PASS" should only be asssigned after you are completely satisfied that all the revised visual(s) pass the review criteria.

Remember: Only review the segments listed above. Focus on the repeated visual asset (URL: {repeated_visual_url}). If all revised visuals now meet the criteria, output PASS. If any visual still fails, provide detailed feedback in the same format as before."""
    
    # print(f"\n{'='*80}")
    # print(f"📝 FORMATTED FOLLOW-UP REDUNDANCY REVIEW PROMPT (Topic: {topic_name}):")
    # print(f"{'='*80}")
    # print(followup_prompt)
    # print(f"{'='*80}\n")
    print(f"Invoking Gemini multimodal follow-up redundancy review (model: {llm})...")
    response_text, updated_history = invoke_gemini_multimodal(
        asset_parts + [types.Part(text=followup_prompt)], 
        llm=llm,
        conversation_history=conversation_history
    )
    print(f"\nFollow-up redundancy review response (Topic: {topic_name}):\n{response_text}\n")
    verdict, failures = parse_review_response(response_text)
    if target_keys:
        failures = [f for f in failures if f.get("segment_key") in target_keys]
        if failures:
            verdict = "FAIL"
    print(f"Verdict: {verdict}, Failures: {len(failures)}")
    if failures:
        for failure in failures:
            print(f"- {failure.get('segment_key', 'Unknown')}: {failure.get('reason', 'No reason provided')[:100]}")
    return verdict, failures, response_text, updated_history


def build_candidate_text(candidates):
    """
    Build formatted text from candidate list.
    
    :param candidates: List of candidate dictionaries with 'id', 'title', and 'url' keys
    :return: Formatted candidate text string
    """
    
    return "\n".join([f"{c['id']} | {c['title']} | URL: {c['url']}" for c in candidates]) or "None"


def build_video_candidates_text(videos, frame_videos):
    """
    Build video candidates text with two sections: 1. Videos from video pool (can use clips or frames), 2. Videos from other channels (frames only)
    
    :param videos: The list of videos to build the text for.
    :param frame_videos: The list of frame videos to build the text for.
    :return: The video candidates text.
    """
    video_candidates_text = ""
    
    # First section: video_pool (clips or frames)
    if videos:
        video_candidates_text += "Videos from which you can use video clips (any part of this video with start and end timestamps) or still frames (extracted from any point in the video) as images:\n"
        video_candidates_text += "\n".join([
            f"{c['id']} | URL: {c['url']}"
            for c in videos
        ])
        video_candidates_text += "\n\n"
    
    # Second section: video_pool_other_channels (frames only)
    if frame_videos:
        video_candidates_text += "Videos from which you can ONLY use still frames as images (extracted from any point in the video) and NOT playable video clips with timestamps:\n"
        video_candidates_text += "\n".join([
            f"{c['id']} | URL: {c['url']}"
            for c in frame_videos
        ])
    
    if not video_candidates_text.strip():
        video_candidates_text = "No video candidates provided."
    
    return video_candidates_text


def build_candidates_for_segment(image_pool_text, video_pool_filtered_text, drive_results_text, web_results_text, segment_num, video_pool_text="", video_pool_other_channels_text=""):
    """
    Build candidate images and videos for a specific segment.
    
    :param image_pool_text: Image pool column content (filtered images)
    :param video_pool_filtered_text: Video pool filtered column content (filtered videos)
    :param drive_results_text: Drive results column content (fallback for images)
    :param web_results_text: Web results column content (fallback for images)
    :param segment_num: Segment number to extract candidates for
    :param video_pool_text: Optional; used as fallback when video_pool_filtered is empty
    :param video_pool_other_channels_text: Optional; used as fallback when video_pool_filtered is empty
    :return: Tuple of (images, videos, frame_videos) where each is a list of candidate dictionaries
    """
    # Parse images from image_pool first, fallback to drive_results + web_results if empty
    image_items = parse_urls_from_image_pool(image_pool_text, segment_num)
    if not image_items:
        # Fallback to original search results
        image_items = parse_urls_from_results(drive_results_text, segment_num)
        image_items += parse_urls_from_results(web_results_text, segment_num)
        print(f"Segment {segment_num}: Using fallback image sources (drive_results + web_results)")
    else:
        print(f"Segment {segment_num}: Using image_pool")
    
    images: List[Dict[str, str]] = []
    for idx, item in enumerate(image_items, start=1):
        images.append({
            "id": f"IMG_{idx}",
            "title": item.get("title", "Untitled"),
            "url": item.get("url", ""),
        })

    # Parse videos from video_pool_filtered
    video_items_filtered = parse_urls_from_video_pool_filtered(video_pool_filtered_text, segment_num)
    # Fallback: if video_pool_filtered is empty, use video_pool + video_pool_other_channels combined
    if not video_items_filtered and (video_pool_text or video_pool_other_channels_text):
        video_items_filtered = get_video_items_fallback_from_pools(video_pool_text or "", video_pool_other_channels_text or "", segment_num)
        if video_items_filtered:
            print(f"Segment {segment_num}: video_pool_filtered empty, using fallback video_pool + video_pool_other_channels ({len(video_items_filtered)} video(s))")
    
    # Separate embed videos (can use clips or frames) from full videos (frames only)
    embed_videos = [item for item in video_items_filtered if item.get("type") == "embed"]
    full_video_items = [item for item in video_items_filtered if item.get("type") == "full_video"]
    
    videos: List[Dict[str, Any]] = []
    for idx, item in enumerate(embed_videos, start=1):
        videos.append({
            "id": f"VID_1.{idx}",
            "title": "Video clip (timestamps allowed)",
            "url": item.get("url", ""),
            "type": "embed",
        })

    frame_videos: List[Dict[str, Any]] = []
    for idx, item in enumerate(full_video_items, start=1):
        frame_videos.append({
            "id": f"VID_2.{idx}",
            "title": "Frames only (no clip timestamps)",
            "url": item.get("url", ""),
            "type": "full_video",
        })

    print(f"Segment {segment_num} candidates: {len(images)} image(s), {len(videos)} video clip(s), {len(frame_videos)} frame video(s)")
    return images, videos, frame_videos


def resolve_asset_urls_in_definition(graphics_definition_xml, candidate_map):
    """
    Replace candidate IDs inside <replacement_visual_url> tags with their actual URLs.

    :param graphics_definition_xml: XML string containing <replacement_visual_url> tags.
    :param candidate_map: Mapping of candidate IDs (IMG_1, VID_2, etc.) to URLs.
    :return: XML string with resolved asset URLs.
    """
    if not graphics_definition_xml or not candidate_map:
        return graphics_definition_xml

    def replace_asset(match):
        raw_value = match.group(1).strip()
        resolved = candidate_map.get(raw_value)
        if resolved:
            return f"<replacement_visual_url>{resolved}</replacement_visual_url>"
        return match.group(0)

    return re.sub(
        r"<replacement_visual_url>\s*(.*?)\s*</replacement_visual_url>",
        replace_asset,
        graphics_definition_xml,
        flags=re.DOTALL | re.IGNORECASE,
    )


@traceable(
    metadata={
        "agent_name": "graphics_definition_v2",
        "step_name": "Review Agent",
        "function_name": "revise_segment_visuals",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous")
    }
)
def revise_segment_visuals(course_name, target_audience, topic_name, subtopic_name, slide_title, slide_chunk, vo_text, current_visuals, feedback, image_pool_text, video_pool_filtered_text, drive_results_text, web_results_text, segment_num, drive, llm, visual_assignment_strategy="Flexible, let the agent decide", video_pool_text="", video_pool_other_channels_text=""):
    """
    Revise segment visuals based on review feedback.
    
    :param course_name: Course name
    :param target_audience: Target audience
    :param topic_name: Topic name
    :param subtopic_name: Subtopic name
    :param slide_title: Slide title
    :param slide_chunk: Full slide content
    :param vo_text: Voiceover text for the segment (or slide_chunk for entire slide case)
    :param current_visuals: Current visual assignments text
    :param feedback: Review feedback describing what needs to be changed
    :param image_pool_text: Image pool column content (filtered images)
    :param video_pool_filtered_text: Video pool filtered column content (filtered videos)
    :param drive_results_text: Drive results column content (fallback for images)
    :param web_results_text: Web results column content (fallback for images)
    :param segment_num: Segment number
    :param drive: Google Drive instance
    :param llm: Language model to use
    :param visual_assignment_strategy: Visual assignment strategy (default: "Flexible, let the agent decide")
    :return: Graphics definition XML with replacement visuals or None if revision fails
    """
    
    print(f"Revising segment {segment_num} visuals...")
    print(f"Feedback: {feedback}...")
    images, videos, frame_videos = build_candidates_for_segment(
        image_pool_text,
        video_pool_filtered_text,
        drive_results_text,
        web_results_text,
        segment_num,
        video_pool_text=video_pool_text,
        video_pool_other_channels_text=video_pool_other_channels_text,
    )
    candidate_map = {
        candidate["id"]: candidate["url"]
        for candidate in images + videos + frame_videos
        if candidate.get("id") and candidate.get("url")
    }
    image_candidates_text = build_candidate_text(images)
    video_candidates_text = build_video_candidates_text(videos, frame_videos)

    # Parse current visuals to extract visual steps for multimodal loading
    current_visual_steps = []
    for line in current_visuals.split('\n'):
        line = line.strip()
        if not line:
            continue
        # Format: "S1V1 | When VO: \"...\" | Visual assigned: https://..."
        if " | Visual assigned: " in line:
            parts = line.split(" | Visual assigned: ", 1)
            if len(parts) == 2:
                visual_id_part = parts[0].split(" | ")[0].strip()
                asset_url = parts[1].strip()
                current_visual_steps.append({
                    "visual_id": visual_id_part,
                    "asset_url": asset_url
                })

    # Select prompt based on visual_assignment_strategy
    visual_assignment_strategy = str(visual_assignment_strategy).strip()
    if visual_assignment_strategy == "1 Visual for the whole Slide":
        prompt_template = REVISION_PROMPT_FOR_ONE_VISUAL_PER_SLIDE
        print(f"Using REVISION_PROMPT_FOR_ONE_VISUAL_PER_SLIDE (strategy: {visual_assignment_strategy})")
        # For entire slide, use slide_chunk as vo_text
        actual_vo_text = slide_chunk
        split_tag = "</current_visual_assigned>"  # Different tag for entire slide prompt
        include_one_visual_note = False
    else:
        prompt_template = REVISION_PROMPT
        print(f"Using REVISION_PROMPT (strategy: {visual_assignment_strategy})")
        actual_vo_text = vo_text
        split_tag = "</current_visuals_assigned>"  # Original tag for segment-based prompts
        # Only include the "1 visual" note for "1 Visual per Sentence" strategy, not for "Flexible"
        include_one_visual_note = (visual_assignment_strategy == "1 Visual per Sentence")

    # Format the full prompt
    full_prompt = prompt_template.format(
        course_name=course_name,
        target_audience=target_audience,
        topic_name=topic_name,
        subtopic_name=subtopic_name,
        slide_id=f"SLIDE_{segment_num}",  # For entire slide, segment_num is 1
        slide_title=slide_title,
        slide_chunk=slide_chunk,
        vo_text=actual_vo_text,
        current_visuals=current_visuals,
        feedback=feedback,
        image_candidates=image_candidates_text,
        video_candidates=video_candidates_text,
    )
    
    # Add the "1 visual" note only for "1 Visual per Sentence" strategy
    if include_one_visual_note:
        # Insert the note after </candidates> and before Instructions:
        candidates_end = full_prompt.find("</candidates>")
        if candidates_end != -1:
            insert_pos = full_prompt.find("\n\nInstructions:", candidates_end)
            if insert_pos != -1:
                note = "\n\nIMPORTANT: Remember, you have been allowed to use only 1 visual for replacing the failed visual for this voiceover segment. So keep that in mind as you select the replacement visual."
                full_prompt = full_prompt[:insert_pos] + note + full_prompt[insert_pos:]
    
    print(f"Starting segment {segment_num} revision...")
    # print(f"\n{'='*80}")
    # strategy_label = f" [Strategy: {visual_assignment_strategy}]" if visual_assignment_strategy else ""
    # print(f"📝 FORMATTED REVISION PROMPT (Segment {segment_num}){strategy_label}:")
    # print(f"{'='*80}")
    # print(full_prompt)
    # print(f"{'='*80}\n")

    # Split prompt at specific points and insert multimodal parts
    parts: List[types.Part] = []
    
    # Split after current visuals tag (different for entire slide vs segment-based)
    split1 = full_prompt.split(split_tag, 1)
    if len(split1) == 2:
        parts.append(types.Part(text=split1[0] + split_tag))
        
        # Insert current visuals as multimodal
        for idx, step in enumerate(current_visual_steps, start=1):
            visual_parts = build_current_visual_parts(f"{idx} ({step['visual_id']})", step["asset_url"], drive)
            parts.extend(visual_parts)
        
        remaining = split1[1]
    else:
        remaining = full_prompt
    
    # Split after </image_candidates>
    split2 = remaining.split("</image_candidates>", 1)
    candidate_num = 1  # Initialize candidate numbering
    if len(split2) == 2:
        parts.append(types.Part(text=split2[0] + "</image_candidates>"))
        
        # Insert image candidates as multimodal 
        for candidate in images:
            label_text = f"Image Candidate {candidate_num} ({candidate['id']}): {candidate['title']} | URL: {candidate['url']}"
            parts.append(types.Part(text=label_text))
            visual_part = build_visual_part_only(candidate["url"], drive)
            if visual_part:
                parts.append(visual_part)
            candidate_num += 1
        
        remaining = split2[1]
    else:
        remaining = split2[0] if split2 else remaining
    
    # Split after </video_candidates>
    split3 = remaining.split("</video_candidates>", 1)
    if len(split3) == 2:
        parts.append(types.Part(text=split3[0] + "</video_candidates>"))
        total_video_candidates = len(videos) + len(frame_videos)
        
        # Insert video candidates as multimodal
        # Handle embed videos (can use clips or frames)
        for candidate in videos:
            video_type = candidate.get("type", "embed")
            video_url = candidate.get("url", "")
            
            if video_type == "embed":
                # Embed URL with timestamps - can be used as clips or frames
                clip_url, start_seconds, end_seconds = parse_video_url_timestamps(video_url)
                if clip_url:
                    label_text = (
                        f"\n--- Video {candidate_num} of {total_video_candidates} ---\n"
                        f"ID: {candidate['id']}\n"
                        f"Usage: Can be used as video clip (any part of this video with start and end timestamps) OR as still frame (extracted from any point in the video)\n"
                        f"URL: {video_url}\n"
                    )
                    parts.append(types.Part(text=label_text))
                    video_part = build_video_part(clip_url, start_seconds, end_seconds)
                    if video_part:
                        parts.append(video_part)
                    candidate_num += 1
                else:
                    print(f"WARNING: Failed to parse embed video URL: {video_url}")
            else:
                # Fallback: treat as regular video
                label_text = (
                    f"\n--- Video {candidate_num} of {total_video_candidates} ---\n"
                    f"ID: {candidate['id']}\n"
                    f"Usage: Can be used as video clip (any part of this video with start and end timestamps) OR as still frame (extracted from any point in the video)\n"
                    f"URL: {video_url}\n"
                )
                parts.append(types.Part(text=label_text))
                visual_part = build_visual_part_only(video_url, drive)
                if visual_part:
                    parts.append(visual_part)
                candidate_num += 1
        
        # Handle full videos (frames only)
        for candidate in frame_videos:
            video_type = candidate.get("type", "full_video")
            video_url = candidate.get("url", "")
            
            if video_type == "full_video":
                # Full video - frames only
                embed_url = convert_watch_url_to_embed_url(video_url)
                if embed_url:
                    label_text = (
                        f"\n--- Video {candidate_num} of {total_video_candidates} ---\n"
                        f"ID: {candidate['id']}\n"
                        f"Usage: Can be used ONLY as still frames (extracted from any point in the video) as images (NOT playable video clips with timestamps)\n"
                        f"URL: {video_url}\n"
                    )
                    parts.append(types.Part(text=label_text))
                    # Add video part without timestamps (full video)
                    video_part = build_video_part(embed_url, start_seconds=None, end_seconds=None)
                    if video_part:
                        parts.append(video_part)
                    candidate_num += 1
                else:
                    print(f"WARNING: Failed to convert video URL: {video_url}")
            else:
                # Fallback: treat as regular video
                label_text = (
                    f"\n--- Video {candidate_num} of {total_video_candidates} ---\n"
                    f"ID: {candidate['id']}\n"
                    f"Usage: Can be used ONLY as still frames (extracted from any point in the video) as images (NOT playable video clips with timestamps)\n"
                    f"URL: {video_url}\n"
                )
                parts.append(types.Part(text=label_text))
                visual_part = build_visual_part_only(video_url, drive)
                if visual_part:
                    parts.append(visual_part)
                candidate_num += 1
        
        # Add the rest of the prompt
        parts.append(types.Part(text=split3[1]))
    else:
        # If splitting failed, just add remaining text
        parts.append(types.Part(text=remaining))
    
    print(f"Loaded {len(parts)} part(s) for revision (including {len(current_visual_steps)} current visual(s), {len(images)} image candidate(s), {len(videos) + len(frame_videos)} video candidate(s))")
    
    print(f"Invoking Gemini multimodal revision (model: {llm})...")
    response_text, _ = invoke_gemini_multimodal(parts, llm=llm)
    print(f"\nRevision response (SEGMENT {segment_num}):\n{response_text}\n")
    
    # Extract replacement visuals - handle both plural and singular wrappers
    replacement_visuals = _extract_tag(response_text, "replacement_visuals")
    if not replacement_visuals:
        # Try singular wrapper (for entire slide case)
        replacement_visuals = _extract_tag(response_text, "replacement_visual")
    
    if not replacement_visuals:
        print(f"WARNING: No replacement_visuals or replacement_visual found in revision response for segment {segment_num}")
        return None
    
    # Resolve asset URLs 
    resolved_def = resolve_asset_urls_in_definition(replacement_visuals, candidate_map)
    if resolved_def != replacement_visuals:
        print(f"Resolved asset IDs to URLs for segment {segment_num}")
    
    # Return the XML directly for merging, not formatted text
    print(f"Successfully revised segment {segment_num}")
    return resolved_def


def generate_search_queries_with_feedback(course_name, target_audience, topic_name, subtopic_name, slide_title, slide_chunk, vo_text, feedback, llm, segment_num, segments_map, drive, visual_assignment_strategy="Flexible, let the agent decide"):
    """
    Generate revised search queries for a failed segment using human/feedback context.

    :param course_name: Course name
    :param target_audience: Target audience
    :param topic_name: Topic name
    :param subtopic_name: Subtopic name
    :param slide_title: Slide title
    :param slide_chunk: Slide chunk text
    :param vo_text: Voiceover text for the segment
    :param feedback: Feedback string 
    :param llm: LLM model name
    :param segment_num: Segment number
    :param segments_map: Map of segment_num -> segment data
    :param drive: Google Drive instance
    :param visual_assignment_strategy: Visual assignment strategy
    :return: List of search query strings
    """
    
    print(f"Generating revised search queries (model: {llm})...")
    
    # Extract failed visual IDs from feedback
    failed_visual_ids = []
    for line in feedback.splitlines():
        if line.startswith("Failing Visual:"):
            visual_id = line.replace("Failing Visual:", "").strip()
            if visual_id:
                failed_visual_ids.append(visual_id)
    
    # Load failed visuals as multimodal inputs
    asset_parts: List[types.Part] = []
    segment = segments_map.get(segment_num, {})
    visual_steps = segment.get("visual_steps", [])
    
    for step in visual_steps:
        visual_id = step.get("visual_id", "")
        if visual_id in failed_visual_ids:
            asset_url = step.get("asset", "")
            if asset_url:
                print(f"Loading failed visual {visual_id}: {asset_url[:80]}...")
                # Create clear label for the failed visual
                label = types.Part(text=f"FAILED VISUAL - Visual ID: {visual_id} | URL: {asset_url}")
                asset_parts.append(label)
                # Load the actual visual asset
                visual_parts = build_asset_parts(visual_id, asset_url, drive)
                # Skip the text label from build_asset_parts (we already added our own), keep the visual parts
                if len(visual_parts) > 1:
                    asset_parts.extend(visual_parts[1:])
                elif len(visual_parts) == 1:
                    asset_parts.extend(visual_parts)
    
    # Select prompt based on visual_assignment_strategy
    visual_assignment_strategy = str(visual_assignment_strategy).strip()
    if visual_assignment_strategy == "1 Visual for the whole Slide":
        prompt_template = SEARCH_QUERY_REVISION_PROMPT_FOR_ONE_VISUAL_PER_SLIDE
        # print(f"  Using SEARCH_QUERY_REVISION_PROMPT_FOR_ONE_VISUAL_PER_SLIDE (strategy: {visual_assignment_strategy})")
        # For entire slide, use slide_chunk instead of vo_text
        actual_vo_text = slide_chunk
    else:
        prompt_template = SEARCH_QUERY_REVISION_PROMPT
        # print(f"  Using SEARCH_QUERY_REVISION_PROMPT (strategy: {visual_assignment_strategy})")
        actual_vo_text = vo_text

    # Format the prompt
    prompt_text = prompt_template.format(
        course_name=course_name,
        target_audience=target_audience,
        topic_name=topic_name,
        subtopic_name=subtopic_name,
        slide_title=slide_title,
        slide_chunk=slide_chunk,
        vo_text=actual_vo_text,
        feedback=feedback,
    )
    print(f"Generating search queries for segment {segment_num}...")
    # strategy_label = f" [Strategy: {visual_assignment_strategy}]" if visual_assignment_strategy else ""
    # print(f"\n{'='*80}")
    # print(f"📝 FORMATTED SEARCH QUERY REVISION PROMPT (Segment {segment_num}){strategy_label}:")
    # print(f"{'='*80}")
    # print(prompt_text)
    # print(f"{'='*80}\n")

    # Use multimodal API if we have visuals, otherwise fallback to text-only
    if asset_parts:
        # print(f"    Using multimodal input with {len([p for p in asset_parts if hasattr(p, 'text') and p.text and 'FAILED VISUAL' in p.text])} failed visual(s)")
        # print(f"    Multimodal parts to be sent:")
        # for idx, part in enumerate(asset_parts, 1):
        #     if hasattr(part, 'text') and part.text:
        #         print(f"      Part {idx} (text): {part.text}...")
        #     elif hasattr(part, 'inline_data'):
        #         print(f"      Part {idx} (image/video data)")
        response_text, _ = invoke_gemini_multimodal(
            asset_parts + [types.Part(text=prompt_text)],
            llm=llm,
            temperature=0.7,
        )
        queries_block = _extract_tag(response_text, "queries")
        if response_text:
            print(f"\nSearch query revision response ({slide_title}):\n{response_text}\n")
    else:
        # Fallback to text-only if no visuals loaded
        print(f"No failed visuals to load, using text-only mode")
        agent = Chain(llm=llm, tags=["output"])
        agent.add_message(role="user", content=prompt_text)
        response = agent.run()
        response_text = response.get("output", "")
        queries_block = _extract_tag(response_text, "queries")
        if response_text:
            print(f"\nSearch query revision response ({slide_title}):\n{response_text}\n")
    
    if not queries_block:
        queries_block = response_text
    
    queries = []
    for line in queries_block.splitlines():
        cleaned = line.strip()
        if not cleaned:
            continue
        if cleaned.startswith("-"):
            cleaned = cleaned[1:].strip()
        if cleaned:
            queries.append(cleaned)

    # Enforce fixed query count for regeneration:
    # - 3 for Flexible/1 Visual per Sentence
    # - 4 for 1 Visual for the whole Slide
    target_query_count = 4 if visual_assignment_strategy == "1 Visual for the whole Slide" else 3

    # Deduplicate while preserving order
    deduped_queries = []
    seen_queries = set()
    for q in queries:
        key = q.strip().lower()
        if key and key not in seen_queries:
            seen_queries.add(key)
            deduped_queries.append(q.strip())

    # If model under-produces, backfill with deterministic context queries.
    if len(deduped_queries) < target_query_count:
        fallback_candidates = [
            _safe_str(vo_text),
            _safe_str(slide_title),
            f"{_safe_str(topic_name)} {_safe_str(subtopic_name)}".strip(),
            _safe_str(slide_chunk),
        ]
        for candidate in fallback_candidates:
            cleaned = " ".join(candidate.split()).strip()
            if not cleaned:
                continue
            cleaned = cleaned[:120]
            key = cleaned.lower()
            if key in seen_queries:
                continue
            deduped_queries.append(cleaned)
            seen_queries.add(key)
            if len(deduped_queries) >= target_query_count:
                break

    final_queries = deduped_queries[:target_query_count]
    print(
        f"Generated {len(queries)} raw query/queries; using {len(final_queries)} "
        f"query/queries (target={target_query_count})"
    )
    return final_queries


def merge_replacements_into_segment(existing_segment_text, replacement_visuals_xml, segment_num):
    """
    Merge replacement visuals into existing segment, keeping unchanged visuals.

    :param existing_segment_text: The existing formatted segment text from final_graphics_definition
    :param replacement_visuals_xml: The XML string containing <replacement_visuals> with <visual> blocks
    :param segment_num: Segment number
    :return: Merged segment text with replacements applied
    """
    
    if not replacement_visuals_xml or not replacement_visuals_xml.strip():
        return existing_segment_text
    
    # Parse existing segment to get all visual steps
    existing_steps = parse_visual_steps(existing_segment_text)
    if not existing_steps:
        return existing_segment_text
    
    # Assign visual_ids to existing steps if not present
    for idx, step in enumerate(existing_steps, start=1):
        if "visual_id" not in step:
            step["visual_id"] = f"S{segment_num}V{idx}"
    
    # Parse replacement visuals from XML - handle both plural and singular wrappers
    replacement_visuals_match = re.search(
        r'<replacement_visuals>(.*?)</replacement_visuals>',
        replacement_visuals_xml,
        re.DOTALL | re.IGNORECASE
    )
    if not replacement_visuals_match:
        # Try singular wrapper (for entire slide case)
        replacement_visuals_match = re.search(
            r'<replacement_visual>(.*?)</replacement_visual>',
            replacement_visuals_xml,
            re.DOTALL | re.IGNORECASE
        )
    
    if not replacement_visuals_match:
        return existing_segment_text
    
    visual_blocks = re.findall(
        r'<visual>(.*?)</visual>',
        replacement_visuals_match.group(1),
        re.DOTALL | re.IGNORECASE
    )
    
    if not visual_blocks:
        return existing_segment_text
    
    # Build a map of visual_id -> replacement data
    replacements_map = {}
    for visual_xml in visual_blocks:
        visual_id = _extract_tag(visual_xml, "visual_id").strip()
        if not visual_id:
            continue
        
        replacement_url = _extract_tag(visual_xml, "replacement_visual_url")
        
        replacements_map[visual_id] = {
            "voiceover_part": _extract_tag(visual_xml, "voiceover_part"),
            "replacement_visual_url": replacement_url,
            "visual_instruction": _extract_tag(visual_xml, "visual_instruction"),
            "selection_justification": _extract_tag(visual_xml, "selection_justification"),
        }
    
    # Apply replacements to existing steps
    for step in existing_steps:
        visual_id = step.get("visual_id", "")
        if visual_id in replacements_map:
            replacement = replacements_map[visual_id]
            step["voiceover_part"] = replacement["voiceover_part"]
            step["asset"] = replacement["replacement_visual_url"]
            step["visual_instruction"] = replacement["visual_instruction"]
            step["selection_justification"] = replacement["selection_justification"]
    
    # Rebuild segment text from merged steps
    formatted_parts = []
    formatted_parts.append("=" * 80)
    formatted_parts.append(f"SEGMENT {segment_num}")
    formatted_parts.append("=" * 80)
    formatted_parts.append("")
    
    for step_idx, step in enumerate(existing_steps):
        if step_idx > 0:
            formatted_parts.append("----")
            formatted_parts.append("")
        
        if step.get("voiceover_part"):
            formatted_parts.append(f'When VO: "{step["voiceover_part"]}"')
            formatted_parts.append("")
        
        if step.get("visual_instruction"):
            formatted_parts.append(f"Visual Instructions: {step['visual_instruction']}")
            formatted_parts.append("")
        
        if step.get("asset"):
            formatted_parts.append(f"Graphics to use: {step['asset']}")
            formatted_parts.append("")
        
        if step.get("selection_justification"):
            formatted_parts.append(f"Selection Justification: {step['selection_justification']}")
            formatted_parts.append("")
    
    return "\n".join(formatted_parts)


def update_final_graphics_definition_with_replacements(original_text, segment_num, replacement_visuals_xml):
    """
    Update a single segment in final_graphics_definition by merging replacements.

    :param original_text: The full final_graphics_definition text
    :param segment_num: Segment number to update
    :param replacement_visuals_xml: The XML string containing <replacement_visuals>
    :return: Updated final_graphics_definition text
    """
    
    segments = parse_final_graphics_definition(original_text)
    existing_segment_text = segments.get(segment_num, "")
    
    if not existing_segment_text:
        return original_text
    
    merged_segment = merge_replacements_into_segment(
        existing_segment_text,
        replacement_visuals_xml,
        segment_num,
    )
    
    segments[segment_num] = merged_segment
    return build_final_graphics_definition(segments)


def update_final_graphics_definition(original_text, updated_segments):
    """
    Replace specified segments in final_graphics_definition with updated segment text.

    :param original_text: The full final_graphics_definition text
    :param updated_segments: Map of segment_num -> new segment text
    :return: Updated final_graphics_definition text
    """
    segments = parse_final_graphics_definition(original_text)
    for segment_num, segment_text in updated_segments.items():
        if segment_text:
            segments[segment_num] = segment_text
    if not segments:
        return original_text
    return build_final_graphics_definition(segments)


@traceable(
    metadata={
        "agent_name": "graphics_definition_v2",
        "step_name": "Review Agent",
        "function_name": "regenerate_failed_segments",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous")
    }
)
def regenerate_failed_segments(row_index, row, df, course_name, target_audience, drive, llm, failed_segments, feedback_by_segment, ws=None, use_only_drive_and_hvac=False, create_aux_search_columns_if_missing=False):
    """
    Regenerate visuals for failed segments: new search queries, search execution, and revision.

    :param row_index: Row index in the dataframe
    :param row: Row data (series)
    :param df: Dataframe (shared)
    :param course_name: Course name
    :param target_audience: Target audience
    :param drive: Google Drive instance
    :param llm: LLM model name
    :param failed_segments: List of segment numbers that failed review
    :param feedback_by_segment: Map of segment_num -> feedback string
    :param ws: Worksheet object or None
    :param use_only_drive_and_hvac: If True, skip web search and other-channels video search during regeneration.
    :param create_aux_search_columns_if_missing: If True, create web_results and video_pool_other_channels when missing.
    :return: Tuple of (replaced_visual_ids_by_segment, old_asset_urls_by_visual_id)
    """
    if create_aux_search_columns_if_missing:
        if "web_results" not in df.columns:
            if "drive_results" in df.columns:
                df.insert(int(df.columns.get_loc("drive_results")) + 1, "web_results", "")
            else:
                df["web_results"] = ""
        if "video_pool_other_channels" not in df.columns:
            if "web_results" in df.columns:
                df.insert(int(df.columns.get_loc("web_results")) + 1, "video_pool_other_channels", "")
            else:
                df["video_pool_other_channels"] = ""

    slide_title = _safe_str(row.get("Slide Chunk Title", ""))
    slide_chunk = _safe_str(row.get("Slide Chunk", ""))
    topic_name = _safe_str(row.get("Topic", ""))
    subtopic_name = _safe_str(row.get("Subtopic", ""))
    voiceover_text = _safe_str(row.get("voiceover_segment", ""))
    final_graphics_definition = _safe_str(row.get("final_graphics_definition", ""))
    
    # Get visual assignment strategy
    visual_assignment_strategy = str(row.get("Visual Assignment Strategy", "Flexible, let the agent decide")).strip()
    if not visual_assignment_strategy or visual_assignment_strategy == "nan":
        visual_assignment_strategy = "Flexible, let the agent decide"

    slide_type = str(row.get("Slide Type", "")).strip().lower()
    skip_video_candidates = slide_type in ("transition", "transition slide")
    if skip_video_candidates:
        print("Transition slide: skipping video candidate generation and filtering in regeneration loop")

    voiceover_segments = parse_segments_from_voiceover(voiceover_text)
    segments_map = build_segment_visual_map(voiceover_text, final_graphics_definition, visual_assignment_strategy, slide_chunk)

    print(f"Generating search queries for {len(failed_segments)} failed segment(s)...")
    for segment_num in failed_segments:
        try:
            vo_text = ""
            for seg_idx, seg_text in voiceover_segments:
                if seg_idx == segment_num:
                    vo_text = seg_text
                    break
            feedback = feedback_by_segment.get(segment_num, "")
            queries = generate_search_queries_with_feedback(
                course_name=course_name,
                target_audience=target_audience,
                topic_name=topic_name,
                subtopic_name=subtopic_name,
                slide_title=slide_title,
                slide_chunk=slide_chunk,
                vo_text=vo_text,
                feedback=feedback,
                llm=llm,
                segment_num=segment_num,
                segments_map=segments_map,
                drive=drive,
                visual_assignment_strategy=visual_assignment_strategy,
            )
            if queries:
                df.at[row_index, "search_queries"] = replace_segment_block(
                    _safe_str(df.at[row_index, "search_queries"]),
                    segment_num,
                    queries,
                )
                print(f"Segment {segment_num}: Updated with {len(queries)} query/queries")
                if ws is not None:
                    with _sheet_lock:
                        save_to_sheet(ws, df)
                    print(f"Segment {segment_num}: Saved search_queries to sheet")
            else:
                print(f"Segment {segment_num}: WARNING - No queries generated")
        except Exception as e:
            print(f"ERROR: Query generation failed for segment {segment_num} during regeneration (skipping): {e}")
            traceback.print_exc()

    row = df.loc[row_index]

    print(f"Executing searches for {len(failed_segments)} segment(s)...")
    for segment_num in failed_segments:
        queries_map = parse_segmented_text(_safe_str(row.get("search_queries", "")))
        queries = queries_map.get(segment_num, [])
        if not queries:
            print(f"Segment {segment_num}: Skipping - no queries found")
            continue

        try:
            print(f"Segment {segment_num}: Searching drive...")
            seg_num, drive_results = process_drive_search_segment(
                segment_num,
                queries,
                drive,
                k=REGEN_IMAGE_SEARCH_K,
            )
            if drive_results:
                lines = [line.strip() for line in drive_results.splitlines() if line.strip()]
                df.at[row_index, "drive_results"] = replace_segment_block(
                    _safe_str(df.at[row_index, "drive_results"]),
                    segment_num,
                    lines[1:] if lines and lines[0].startswith("---SEGMENT_") else lines,
                )
                print(f"      Drive search: {len(lines)} result(s)")
                if ws is not None:
                    with _sheet_lock:
                        save_to_sheet(ws, df)
                    print(f"      Segment {segment_num}: Saved drive_results to sheet")
        except Exception as e:
            print(f"  ERROR: Drive search failed for segment {segment_num} (continuing with other searches): {e}")
            traceback.print_exc()

        if use_only_drive_and_hvac:
            print(f"    Segment {segment_num}: Skipping web search (drive + HVAC only mode)")
        elif "web_results" in df.columns:
            try:
                print(f"    Segment {segment_num}: Searching web...")
                seg_num, web_results = process_web_search_segment(
                    segment_num,
                    queries,
                    k=REGEN_IMAGE_SEARCH_K,
                )
                if web_results:
                    lines = [line.strip() for line in web_results.splitlines() if line.strip()]
                    df.at[row_index, "web_results"] = replace_segment_block(
                        _safe_str(df.at[row_index, "web_results"]),
                        segment_num,
                        lines[1:] if lines and lines[0].startswith("---SEGMENT_") else lines,
                    )
                    print(f"      Web search: {len(lines)} result(s)")
                    if ws is not None:
                        with _sheet_lock:
                            save_to_sheet(ws, df)
                        print(f"      Segment {segment_num}: Saved web_results to sheet")
            except Exception as e:
                print(f"  ERROR: Web search failed for segment {segment_num} (continuing with video searches): {e}")
                traceback.print_exc()
        else:
            pass

        if not skip_video_candidates:
            try:
                print(f"    Segment {segment_num}: Searching video pool...")
                seg_num, video_pool = process_video_search_segment(
                    segment_num,
                    queries,
                    drive,
                    k=REGEN_VIDEO_SEARCH_K,
                )
                if video_pool:
                    lines = [line.strip() for line in video_pool.splitlines() if line.strip()]
                    df.at[row_index, "video_pool"] = replace_segment_block(
                        _safe_str(df.at[row_index, "video_pool"]),
                        segment_num,
                        lines[1:] if lines and lines[0].startswith("---SEGMENT_") else lines,
                    )
                    print(f"      Video pool: {len(lines)} result(s)")
                    if ws is not None:
                        with _sheet_lock:
                            save_to_sheet(ws, df)
                        print(f"      Segment {segment_num}: Saved video_pool to sheet")
            except Exception as e:
                print(f"  ERROR: Video pool search failed for segment {segment_num} (continuing with other channels): {e}")
                traceback.print_exc()

            if use_only_drive_and_hvac:
                print(f"    Segment {segment_num}: Skipping other-channels video search (drive + HVAC only mode)")
            elif "video_pool_other_channels" in df.columns:
                segment_sentence = ""
                for seg_idx, seg_text in voiceover_segments:
                    if seg_idx == segment_num:
                        segment_sentence = seg_text
                        break
                try:
                    print(f"    Segment {segment_num}: Searching other channels...")
                    seg_num, video_other = process_segment_other_channels(
                        segment_num,
                        queries,
                        segment_sentence,
                        k=REGEN_VIDEO_SEARCH_K,
                    )
                    if video_other:
                        lines = [line.strip() for line in video_other.splitlines() if line.strip()]
                        df.at[row_index, "video_pool_other_channels"] = replace_segment_block(
                            _safe_str(df.at[row_index, "video_pool_other_channels"]),
                            segment_num,
                            lines[1:] if lines and lines[0].startswith("---SEGMENT_") else lines,
                        )
                        print(f"      Other channels: {len(lines)} result(s)")
                        if ws is not None:
                            with _sheet_lock:
                                save_to_sheet(ws, df)
                            print(f"      Segment {segment_num}: Saved video_pool_other_channels to sheet")
                except Exception as e:
                    print(f"  ERROR: Other-channels search failed for segment {segment_num}: {e}")
                    traceback.print_exc()
            else:
                pass
        else:
            print(f"Segment {segment_num}: Skipping video pool and other-channels search (transition slide)")

    row = df.loc[row_index]
    image_pool_text = _safe_str(row.get("image_pool", ""))
    video_pool_filtered_text = _safe_str(row.get("video_pool_filtered", ""))
    drive_results_text = _safe_str(row.get("drive_results", ""))
    web_results_text = _safe_str(row.get("web_results", ""))

    # Select images from new search results using feedback prompts
    print(f"  Selecting images from new search results for {len(failed_segments)} failed segment(s)...")
    for segment_num in failed_segments:
        try:
            vo_text = ""
            for seg_idx, seg_text in voiceover_segments:
                if seg_idx == segment_num:
                    vo_text = seg_text
                    break

            feedback = feedback_by_segment.get(segment_num, "")

            drive_items = parse_urls_from_results(drive_results_text, segment_num)
            web_items = parse_urls_from_results(web_results_text, segment_num)

            seen_urls = set()
            all_items = []
            for item in drive_items + web_items:
                url = item.get("url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    all_items.append(item)

            all_urls = [item["url"] for item in all_items]
            url_to_title = {item["url"]: item.get("title", "Untitled") for item in all_items}

            if not all_urls:
                print(f"    Segment {segment_num}: No images found in new search results, skipping image selection")
                continue

            print(f"    Segment {segment_num}: Selecting from {len(all_urls)} image(s) with feedback...")

            if visual_assignment_strategy == "1 Visual for the whole Slide":
                selected_images_text = select_images_from_all_for_entire_slide(
                    slide_title=slide_title,
                    slide_chunk=slide_chunk,
                    image_urls=all_urls,
                    url_to_title=url_to_title,
                    course_name=course_name,
                    topic_name=topic_name,
                    subtopic_name=subtopic_name,
                    drive=drive,
                    llm=llm,
                    feedback=feedback
                )
            else:
                selected_images_text = select_images_from_all_for_segment(
                    vo_text=vo_text,
                    slide_title=slide_title,
                    slide_chunk=slide_chunk,
                    image_urls=all_urls,
                    url_to_title=url_to_title,
                    course_name=course_name,
                    topic_name=topic_name,
                    subtopic_name=subtopic_name,
                    drive=drive,
                    llm=llm,
                    feedback=feedback
                )

            if selected_images_text:
                formatted_images = format_selected_images_for_segment(selected_images_text)
                if formatted_images:
                    image_pool_segment_num = 1 if visual_assignment_strategy == "1 Visual for the whole Slide" else segment_num
                    image_pool_text = replace_segment_block(
                        image_pool_text,
                        image_pool_segment_num,
                        formatted_images
                    )
                    df.at[row_index, "image_pool"] = image_pool_text
                    print(f"      Segment {segment_num}: Updated image_pool (segment {image_pool_segment_num}) with {len(formatted_images)} selected image(s)")
                    if ws is not None:
                        with _sheet_lock:
                            save_to_sheet(ws, df)
                        print(f"      Segment {segment_num}: Saved image_pool to sheet")
                else:
                    print(f"      Segment {segment_num}: WARNING - No formatted images generated")
            else:
                print(f"      Segment {segment_num}: WARNING - No images selected")
        except Exception as e:
            print(f"  ERROR: Image selection failed for segment {segment_num} during regeneration (skipping this segment's image_pool update): {e}")
            traceback.print_exc()
            continue
    
    # Reload row to get updated image_pool
    row = df.loc[row_index]
    image_pool_text = _safe_str(row.get("image_pool", ""))
    video_pool_text = _safe_str(row.get("video_pool", ""))
    video_pool_other_channels_text = _safe_str(row.get("video_pool_other_channels", ""))
    video_pool_filtered_text = _safe_str(row.get("video_pool_filtered", ""))

    # Select videos from new search results using feedback prompts (skipped for transition slides)
    if skip_video_candidates:
        print(f"  Skipping video selection / video_pool_filtered update (transition slide)")
    else:
        print(f"  Selecting videos from new search results for {len(failed_segments)} failed segment(s)...")
        for segment_num in failed_segments:
            try:
                vo_text = ""
                for seg_idx, seg_text in voiceover_segments:
                    if seg_idx == segment_num:
                        vo_text = seg_text
                        break

                feedback = feedback_by_segment.get(segment_num, "")

                # Get new video URLs from video_pool and video_pool_other_channels for this segment
                video_urls_pool = parse_urls_from_video_pool(video_pool_text, segment_num)
                video_items_other_channels = parse_video_items_from_pool_other_channels(video_pool_other_channels_text, segment_num)

                if not video_urls_pool and not video_items_other_channels:
                    print(f"    Segment {segment_num}: No videos found in new search results, skipping video selection")
                    continue

                print(f"    Segment {segment_num}: Selecting from {len(video_urls_pool)} video(s) from pool and {len(video_items_other_channels)} video(s) from other channels with feedback...")

                # Select prompt and function based on visual_assignment_strategy
                if visual_assignment_strategy == "1 Visual for the whole Slide":
                    selected_videos_text = select_videos_from_all_for_entire_slide(
                        slide_title=slide_title,
                        slide_chunk=slide_chunk,
                        video_urls_pool=video_urls_pool,
                        video_items_other_channels=video_items_other_channels,
                        course_name=course_name,
                        topic_name=topic_name,
                        subtopic_name=subtopic_name,
                        drive=drive,
                        llm=llm,
                        feedback=feedback
                    )
                else:
                    selected_videos_text = select_videos_from_all_for_segment(
                        vo_text=vo_text,
                        slide_title=slide_title,
                        slide_chunk=slide_chunk,
                        video_urls_pool=video_urls_pool,
                        video_items_other_channels=video_items_other_channels,
                        course_name=course_name,
                        topic_name=topic_name,
                        subtopic_name=subtopic_name,
                        drive=drive,
                        llm=llm,
                        feedback=feedback
                    )

                if selected_videos_text:
                    formatted_videos = format_selected_videos_for_segment(selected_videos_text, video_urls_pool, video_items_other_channels)
                    if formatted_videos:
                        video_pool_filtered_segment_num = 1 if visual_assignment_strategy == "1 Visual for the whole Slide" else segment_num
                        video_pool_filtered_text = replace_segment_block(
                            video_pool_filtered_text,
                            video_pool_filtered_segment_num,
                            formatted_videos
                        )
                        df.at[row_index, "video_pool_filtered"] = video_pool_filtered_text
                        print(f"      Segment {segment_num}: Updated video_pool_filtered (segment {video_pool_filtered_segment_num}) with {len(formatted_videos)} selected video(s)")
                        if ws is not None:
                            with _sheet_lock:
                                save_to_sheet(ws, df)
                            print(f"Segment {segment_num}: Saved video_pool_filtered to sheet")
                    else:
                        print(f"Segment {segment_num}: WARNING - No formatted videos generated")
                else:
                    print(f"Segment {segment_num}: WARNING - No videos selected")
            except Exception as e:
                print(f"  ERROR: Video filtering failed for segment {segment_num} during regeneration (skipping this segment's video_pool_filtered update): {e}")
                traceback.print_exc()
                continue
    
    # Reload row to get updated video_pool_filtered
    row = df.loc[row_index]
    video_pool_filtered_text = _safe_str(row.get("video_pool_filtered", ""))
    video_pool_text = _safe_str(row.get("video_pool", ""))
    video_pool_other_channels_text = _safe_str(row.get("video_pool_other_channels", ""))

    print(f"Aggregating graphics definitions for {len(failed_segments)} segment(s)...")
    updated_segments: Dict[int, str] = {}
    for segment_num in failed_segments:
        try:
            vo_text = ""
            for seg_idx, seg_text in voiceover_segments:
                if seg_idx == segment_num:
                    vo_text = seg_text
                    break

            # Parse images from image_pool first, fallback to drive_results + web_results if empty
            image_items = parse_urls_from_image_pool(image_pool_text, segment_num)
            if not image_items:
                image_items = parse_urls_from_results(drive_results_text, segment_num)
                image_items += parse_urls_from_results(web_results_text, segment_num)
                print(f"    Segment {segment_num}: Using fallback image sources (drive_results + web_results)")
            else:
                print(f"    Segment {segment_num}: Using image_pool")

            # Parse videos from video_pool_filtered
            if skip_video_candidates:
                video_items_filtered = []
            else:
                video_items_filtered = parse_urls_from_video_pool_filtered(video_pool_filtered_text, segment_num)
                if not video_items_filtered and (video_pool_text or video_pool_other_channels_text):
                    video_items_filtered = get_video_items_fallback_from_pools(video_pool_text or "", video_pool_other_channels_text or "", segment_num)
                    if video_items_filtered:
                        print(f"    Segment {segment_num}: video_pool_filtered empty, using fallback video_pool + video_pool_other_channels ({len(video_items_filtered)} video(s))")

            if not image_items and not video_items_filtered:
                print(f"    Segment {segment_num}: WARNING - No candidates found, skipping aggregation")
                continue

            embed_count = len([item for item in video_items_filtered if item.get("type") == "embed"])
            full_video_count = len([item for item in video_items_filtered if item.get("type") == "full_video"])
            print(f"    Segment {segment_num}: Aggregating ({len(image_items)} image(s), {embed_count} embed video(s), {full_video_count} full video(s))...")

            # Extract failed visuals for this segment
            failed_visuals = []
            feedback = feedback_by_segment.get(segment_num, "")
            if feedback:
                failed_visual_ids = []
                for line in feedback.splitlines():
                    if line.startswith("Failing Visual:"):
                        visual_id = line.replace("Failing Visual:", "").strip()
                        if visual_id:
                            failed_visual_ids.append(visual_id)
                segment = segments_map.get(segment_num, {})
                visual_steps = segment.get("visual_steps", [])
                for step in visual_steps:
                    visual_id = step.get("visual_id", "")
                    if visual_id in failed_visual_ids:
                        asset_url = step.get("asset", "")
                        if asset_url:
                            failed_visuals.append({"visual_id": visual_id, "asset_url": asset_url})

            actual_vo_text = vo_text
            if visual_assignment_strategy == "1 Visual for the whole Slide" and segment_num == 1:
                actual_vo_text = slide_chunk

            graphics_definition_xml, evaluation_breakdown = aggregate_graphics_definition_for_segment(
                vo_text=actual_vo_text,
                slide_title=slide_title,
                slide_chunk=slide_chunk,
                image_items=image_items,
                video_items_filtered=video_items_filtered,
                course_name=course_name,
                topic_name=topic_name,
                subtopic_name=subtopic_name,
                storyboard="",
                drive=drive,
                llm=llm,
                feedback=feedback,
                target_audience=target_audience,
                failed_visuals=failed_visuals if failed_visuals else None,
                visual_assignment_strategy=visual_assignment_strategy,
            )
            if graphics_definition_xml:
                graphics_definition_xml = expand_youtube_single_timestamp_clips_in_xml(graphics_definition_xml)
                updated_segments[segment_num] = graphics_definition_xml
                print(f"      Segment {segment_num}: Graphics definition generated")
            else:
                print(f"      Segment {segment_num}: WARNING - No graphics definition generated")
        except Exception as e:
            print(f"  ERROR: Aggregation failed for segment {segment_num} during regeneration (skipping): {e}")
            traceback.print_exc()
            continue

    # Track replaced visuals and old URLs for follow-up review
    replaced_visual_ids_by_segment: Dict[int, List[str]] = {}
    old_asset_urls_by_visual_id: Dict[str, str] = {}

    if updated_segments:
        # Before merging, capture old asset URLs for replaced visuals
        final_graphics_definition_before = _safe_str(row.get("final_graphics_definition", ""))
        segments_before = parse_final_graphics_definition(final_graphics_definition_before)
        
        # Merge replacements instead of replacing entire segments
        final_graphics_definition = final_graphics_definition_before
        for segment_num, replacement_xml in updated_segments.items():
            # Extract replaced visual IDs and old URLs before merging
            existing_segment_text = segments_before.get(segment_num, "")
            existing_steps = parse_visual_steps(existing_segment_text)
            
            # Assign visual_ids to existing steps if not present (same logic as merge_replacements_into_segment)
            for idx, step in enumerate(existing_steps, start=1):
                if "visual_id" not in step:
                    step["visual_id"] = f"S{segment_num}V{idx}"
            
            # Parse replacement visuals to get visual IDs - handle both plural and singular wrappers
            replacement_visuals_match = re.search(
                r'<replacement_visuals>(.*?)</replacement_visuals>',
                replacement_xml,
                re.DOTALL | re.IGNORECASE
            )
            if not replacement_visuals_match:
                # Try singular wrapper (for entire slide case)
                replacement_visuals_match = re.search(
                    r'<replacement_visual>(.*?)</replacement_visual>',
                    replacement_xml,
                    re.DOTALL | re.IGNORECASE
                )
            
            if replacement_visuals_match:
                visual_blocks = re.findall(
                    r'<visual>(.*?)</visual>',
                    replacement_visuals_match.group(1),
                    re.DOTALL | re.IGNORECASE
                )
                for visual_xml in visual_blocks:
                    visual_id = _extract_tag(visual_xml, "visual_id").strip()
                    if visual_id:
                        if segment_num not in replaced_visual_ids_by_segment:
                            replaced_visual_ids_by_segment[segment_num] = []
                        replaced_visual_ids_by_segment[segment_num].append(visual_id)
                        
                        # Find old asset URL from existing steps
                        for step in existing_steps:
                            step_visual_id = step.get("visual_id", "")
                            if step_visual_id == visual_id:
                                old_url = step.get("asset", "")
                                if old_url:
                                    old_asset_urls_by_visual_id[visual_id] = old_url
                                break
            
            final_graphics_definition = update_final_graphics_definition_with_replacements(
                final_graphics_definition,
                segment_num,
                replacement_xml,
            )
        df.at[row_index, "final_graphics_definition"] = final_graphics_definition
        print(f"  Merged {len(updated_segments)} segment(s) in final_graphics_definition")
        
        # Process video frames in the final merged graphics definition
        # This converts any YouTube URLs with only start timestamps (still frames) to Drive images
        # This ensures that the final_graphics_definition always has Drive images instead of YouTube frame URLs
        print(f"  Checking for video frames to process in final graphics definition after regeneration...")
        processed_final_def = process_video_frames_in_text_format(final_graphics_definition, drive)
        if processed_final_def != final_graphics_definition:
            final_graphics_definition = processed_final_def
            df.at[row_index, "final_graphics_definition"] = final_graphics_definition
            print(f"  ✅ Processed video frames in final graphics definition (converted YouTube still frames to Drive images)")
        else:
            print(f"  No video frames found to process in final graphics definition")
    else:
        print(f"  WARNING: No segments updated in final_graphics_definition")
    
    return replaced_visual_ids_by_segment, old_asset_urls_by_visual_id


@traceable(
    metadata={
        "agent_name": "graphics_definition_v2",
        "step_name": "Review Agent",
        "function_name": "run_review_loop_for_slide",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous")
    }
)
def run_review_loop_for_slide(criterion_name, prompt_template, row_index, row, df, course_name, target_audience, drive, llm, ws=None, revision_tracking=None, use_only_drive_and_hvac=False):
    """
    Run review-revise loop for one slide (alignment or specificity): review, revise failed segments, repeat until PASS or max attempts.

    :param criterion_name: "alignment" or "specificity"
    :param prompt_template: Prompt template string for the review
    :param row_index: Row index in the dataframe
    :param row: Row data (series)
    :param df: Dataframe (shared)
    :param course_name: Course name
    :param target_audience: Target audience
    :param drive: Google Drive instance
    :param llm: LLM model name
    :param ws: Worksheet object or None
    :param revision_tracking: Optional initial revision tracking structure
    :param use_only_drive_and_hvac: If True, skip web search and other-channels video search during regeneration.
    :return: Tuple of (outcome "PASS"/"FAIL", revision_tracking)
    """
    
    slide_id = f"SLIDE_{row_index + 1}"
    slide_title = _safe_str(row.get("Slide Chunk Title", ""))
    slide_chunk = _safe_str(row.get("Slide Chunk", ""))
    topic_name = _safe_str(row.get("Topic", ""))
    subtopic_name = _safe_str(row.get("Subtopic", ""))
    voiceover_text = _safe_str(row.get("voiceover_segment", ""))
    final_graphics_definition = _safe_str(row.get("final_graphics_definition", ""))
    segment_nums_to_review: Optional[List[int]] = None
    
    # Get visual assignment strategy and select appropriate prompt
    visual_assignment_strategy = str(row.get("Visual Assignment Strategy", "Flexible, let the agent decide")).strip()
    if not visual_assignment_strategy or visual_assignment_strategy == "nan":
        visual_assignment_strategy = "Flexible, let the agent decide"
    
    # Override prompt_template based on visual_assignment_strategy if needed
    if visual_assignment_strategy == "1 Visual for the whole Slide":
        if criterion_name == "alignment":
            prompt_template = ALIGNMENT_REVIEW_PROMPT_FOR_ONE_VISUAL_PER_SLIDE
        elif criterion_name == "specificity":
            prompt_template = SPECIFICITY_REVIEW_PROMPT_FOR_ONE_VISUAL_PER_SLIDE
        # For other criteria (e.g., redundancy), use the provided prompt_template as-is

    # Initialize revision tracking if not provided
    if revision_tracking is None:
        segments_map_initial = build_segment_visual_map(voiceover_text, final_graphics_definition, visual_assignment_strategy, slide_chunk)
        revision_tracking = initialize_revision_tracking(segments_map_initial)
    
    conversation_history: Optional[List[types.Content]] = None
    replaced_visual_ids_by_segment: Dict[int, List[str]] = {}  # Track which visual IDs were replaced across iterations
    old_asset_urls_by_visual_id: Dict[str, str] = {}  # Track old asset URLs before replacement (visual_id -> old_asset_url)
    # Persist feedback across attempts for follow-up reviews
    last_feedback_by_segment: Dict[int, str] = {}
    last_failed_segments: List[int] = []
    for attempt in range(1, MAX_REVIEW_ATTEMPTS + 1):
        # CRITICAL: Read fresh from dataframe at start of each attempt to ensure we have latest updates
        # This ensures that follow-up reviews (attempt > 1) use the LATEST visuals from final_graphics_definition
        # which includes all replacements merged from previous revision iterations
        # The final_graphics_definition is in text format, while revision agent outputs XML format
        # After merging replacements, the text format contains the replacement visuals, not the original stale ones
        row = df.loc[row_index]  # Refresh row reference to get latest data
        final_graphics_definition = _safe_str(row.get("final_graphics_definition", ""))
        
        segments_map = build_segment_visual_map(voiceover_text, final_graphics_definition, visual_assignment_strategy, slide_chunk)
        segment_nums = segment_nums_to_review or list(segments_map.keys())
        print(f"\n{'='*60}")
        print(f"Review {criterion_name}: Row {row_index + 1} attempt {attempt}/{MAX_REVIEW_ATTEMPTS}")
        print(f"{'='*60}")
        
        # Use follow-up review if we have conversation history (after first attempt with revisions)
        if conversation_history and attempt > 1:
            # Build feedback string from previous failures (use persisted feedback from last attempt)
            previous_feedback_lines = []
            # Use replaced_visual_ids_by_segment if available, otherwise fall back to segment_nums
            if replaced_visual_ids_by_segment:
                for segment_num in replaced_visual_ids_by_segment.keys():
                    if segment_num in last_feedback_by_segment:
                        previous_feedback_lines.append(f"Segment {segment_num}:\n{last_feedback_by_segment[segment_num]}")
            else:
                for segment_num in segment_nums:
                    if segment_num in last_feedback_by_segment:
                        previous_feedback_lines.append(f"Segment {segment_num}:\n{last_feedback_by_segment[segment_num]}")
            previous_feedback = "\n\n".join(previous_feedback_lines)
            
            # CRITICAL: Only use follow-up review if we have replaced visuals to review
            # Follow-up review should ONLY review the failed visuals that were replaced, not all visuals
            if replaced_visual_ids_by_segment:
                # Follow-up review: Only pass the failed visuals that were replaced
                # visual_ids_by_segment contains ONLY the visual IDs that were replaced in the previous revision
                # build_review_targets_for_visuals and review_slide_segments_followup filter to only these visuals
                print(f"  Using follow-up review with {sum(len(vids) for vids in replaced_visual_ids_by_segment.values())} replaced visual(s) only")
                verdict, failures, _, conversation_history = review_slide_segments_followup(
                    course_name=course_name,
                    target_audience=target_audience,
                    topic_name=topic_name,
                    subtopic_name=subtopic_name,
                    slide_id=slide_id,
                    slide_title=slide_title,
                    slide_chunk=slide_chunk,
                    segments_map=segments_map,
                    visual_ids_by_segment=replaced_visual_ids_by_segment,  # ONLY replaced visuals
                    old_asset_urls_by_visual_id=old_asset_urls_by_visual_id,
                    previous_feedback=previous_feedback,
                    drive=drive,
                    llm=llm,
                    conversation_history=conversation_history,
                    criterion_name=criterion_name,
                    visual_assignment_strategy=visual_assignment_strategy,
                )
            else:
                # Edge case: No replaced visuals but we're on attempt > 1
                # This shouldn't happen in normal flow (revisions should produce replacements)
                # Fall back to regular review instead of follow-up review
                print(f"  WARNING: Follow-up review requested but no replaced visuals found. Using regular review instead.")
                verdict, failures, _, conversation_history = review_slide_segments(
                    prompt_template=prompt_template,
                    course_name=course_name,
                    target_audience=target_audience,
                    topic_name=topic_name,
                    subtopic_name=subtopic_name,
                    slide_id=slide_id,
                    slide_title=slide_title,
                    slide_chunk=slide_chunk,
                    segments_map=segments_map,
                    segment_nums=segment_nums,
                    drive=drive,
                    llm=llm,
                    conversation_history=conversation_history,
                    criterion_name=criterion_name,
                    visual_assignment_strategy=visual_assignment_strategy,
                )
        else:
            # First review or fresh start
            verdict, failures, _, conversation_history = review_slide_segments(
                prompt_template=prompt_template,
                course_name=course_name,
                target_audience=target_audience,
                topic_name=topic_name,
                subtopic_name=subtopic_name,
                slide_id=slide_id,
                slide_title=slide_title,
                slide_chunk=slide_chunk,
                segments_map=segments_map,
                segment_nums=segment_nums,
                drive=drive,
                llm=llm,
                conversation_history=conversation_history,
                criterion_name=criterion_name,
                visual_assignment_strategy=visual_assignment_strategy,
            )
        
        if verdict == "PASS":
            print(f"✓ {criterion_name} review PASSED for row {row_index + 1}")
            # Update tracking to mark remaining loops as "No replacement"
            segments_map = build_segment_visual_map(voiceover_text, final_graphics_definition, visual_assignment_strategy, slide_chunk)
            total_loops = MAX_REVIEW_ATTEMPTS + MAX_REGEN_ATTEMPTS
            # Mark remaining review loops
            for loop_num in range(attempt + 1, MAX_REVIEW_ATTEMPTS + 1):
                revision_tracking = update_revision_tracking(
                    revision_tracking, criterion_name, loop_num, {}, segments_map
                )
            # Mark all regeneration loops (didn't run)
            for loop_num in range(MAX_REVIEW_ATTEMPTS + 1, total_loops + 1):
                revision_tracking = update_revision_tracking(
                    revision_tracking, criterion_name, loop_num, {}, segments_map
                )
            return "PASS", revision_tracking

        failed_segments: List[int] = []
        feedback_by_segment: Dict[int, str] = {}
        for failure in failures:
            segment_id_text = failure.get("segment_id", "")
            segment_num = _parse_segment_marker(segment_id_text)
            # Handle "1 Visual for the whole Slide" case where LLM might return "SLIDE_1" instead of "SEGMENT 1"
            if not segment_num and visual_assignment_strategy == "1 Visual for the whole Slide":
                # If it's "SLIDE_X" format, map to segment 1
                if re.search(r"SLIDE_", segment_id_text, re.IGNORECASE):
                    segment_num = 1
            if not segment_num:
                continue
            if segment_num not in failed_segments:
                failed_segments.append(segment_num)
            failing_visual_id = failure.get('failing_visual_ids', '')
            
            # Extract voiceover_part for the failing visual
            voiceover_part = ""
            segment = segments_map.get(segment_num, {})
            visual_steps = segment.get("visual_steps", [])
            for step in visual_steps:
                if step.get('visual_id') == failing_visual_id:
                    voiceover_part = step.get('voiceover_part', '')
                    break
            
            # Build failure text with voiceover part
            failure_text = f"Failing Visual: {failing_visual_id}"
            if voiceover_part:
                failure_text += f"\nVoiceover sentence to which this visual was assigned: {voiceover_part}"
            failure_text += f"\nReason: {failure.get('reason', '')}\nNeeded: {failure.get('needed_visual', '')}"
            
            if segment_num in feedback_by_segment:
                feedback_by_segment[segment_num] += "\n\n" + failure_text
            else:
                feedback_by_segment[segment_num] = failure_text

        if not failed_segments:
            print("  WARNING: No failed segments parsed; stopping review loop.")
            # Mark all remaining loops as "No replacement"
            row = df.loc[row_index]
            final_graphics_definition = _safe_str(row.get("final_graphics_definition", ""))
            segments_map = build_segment_visual_map(voiceover_text, final_graphics_definition, visual_assignment_strategy, slide_chunk)
            total_loops = MAX_REVIEW_ATTEMPTS + MAX_REGEN_ATTEMPTS
            # Mark remaining review loops
            for loop_num in range(attempt, MAX_REVIEW_ATTEMPTS + 1):
                revision_tracking = update_revision_tracking(
                    revision_tracking, criterion_name, loop_num, {}, segments_map
                )
            # Mark all regeneration loops (didn't run)
            for loop_num in range(MAX_REVIEW_ATTEMPTS + 1, total_loops + 1):
                revision_tracking = update_revision_tracking(
                    revision_tracking, criterion_name, loop_num, {}, segments_map
                )
            return "FAIL", revision_tracking
        
        # Persist feedback for next attempt's follow-up review
        last_failed_segments = failed_segments
        last_feedback_by_segment = feedback_by_segment.copy()

        print(f"  Attempting revision for {len(failed_segments)} failed segment(s): {failed_segments}")
        segment_nums_to_review = failed_segments
        updated_segments: Dict[int, str] = {}
        # Track which visual IDs were replaced in this iteration
        iteration_replaced_visual_ids: Dict[int, List[str]] = {}
        # Track old asset URLs before replacement
        iteration_old_asset_urls: Dict[str, str] = {}
        for segment_num in failed_segments:
            segment = segments_map.get(segment_num, {})
            # Store old asset URLs before replacement
            for step in segment.get("visual_steps", []):
                visual_id = step.get('visual_id', '')
                asset_url = step.get('asset', '')
                if visual_id and asset_url:
                    iteration_old_asset_urls[visual_id] = asset_url
            
            current_visuals = "\n".join([
                f"{step.get('visual_id')} | When VO: \"{step.get('voiceover_part', '')}\" | Visual assigned: {step.get('asset', '')}"
                for step in segment.get("visual_steps", [])
            ])
            # For "1 Visual for the whole Slide", use slide_chunk as vo_text when segment_num == 1
            actual_vo_text = _safe_str(segment.get("vo_text", ""))
            if visual_assignment_strategy == "1 Visual for the whole Slide" and segment_num == 1:
                actual_vo_text = slide_chunk
            
            revised = revise_segment_visuals(
                course_name=course_name,
                target_audience=target_audience,
                topic_name=topic_name,
                subtopic_name=subtopic_name,
                slide_title=slide_title,
                slide_chunk=slide_chunk,
                vo_text=actual_vo_text,
                current_visuals=current_visuals,
                feedback=feedback_by_segment.get(segment_num, ""),
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
            )
            if revised:
                # CRITICAL: revise_segment_visuals returns only the content inside <replacement_visuals> or <replacement_visual> tags
                # We need to wrap it in the tags for the merge function and visual ID extraction
                # Wrap the revised content in <replacement_visuals> tags if not already wrapped
                revised_stripped = revised.strip()
                if not revised_stripped.startswith('<replacement_visuals>') and not revised_stripped.startswith('<replacement_visual>'):
                    revised = f"<replacement_visuals>\n{revised}\n</replacement_visuals>"
                
                updated_segments[segment_num] = revised
                # Extract visual IDs that were replaced - handle both plural and singular wrappers
                visual_ids = []
                replacement_visuals_match = re.search(
                    r'<replacement_visuals>(.*?)</replacement_visuals>',
                    revised,
                    re.DOTALL | re.IGNORECASE
                )
                if not replacement_visuals_match:
                    # Try singular wrapper (for entire slide case)
                    replacement_visuals_match = re.search(
                        r'<replacement_visual>(.*?)</replacement_visual>',
                    revised,
                    re.DOTALL | re.IGNORECASE
                )
                if replacement_visuals_match:
                    visual_blocks = re.findall(
                        r'<visual>(.*?)</visual>',
                        replacement_visuals_match.group(1),
                        re.DOTALL | re.IGNORECASE
                    )
                    for visual_xml in visual_blocks:
                        visual_id = _extract_tag(visual_xml, "visual_id").strip()
                        if visual_id:
                            visual_ids.append(visual_id)
                if visual_ids:
                    iteration_replaced_visual_ids[segment_num] = visual_ids

        if updated_segments:
            # Process video frames before merging (convert YouTube frames to Drive images)
            for segment_num, replacement_xml in updated_segments.items():
                processed_xml = expand_youtube_single_timestamp_clips_in_xml(replacement_xml)
                if processed_xml != replacement_xml:
                    updated_segments[segment_num] = processed_xml
                    print(f"  Processed video frames for segment {segment_num}")
            
            # Update the persistent tracking with this iteration's replacements
            # CRITICAL: This must happen BEFORE merging so we track what was replaced
            for segment_num, visual_ids in iteration_replaced_visual_ids.items():
                replaced_visual_ids_by_segment[segment_num] = visual_ids
            # Update old asset URLs tracking - store old URLs for all replaced visuals
            for visual_id, old_url in iteration_old_asset_urls.items():
                # Store old URL if this visual_id was replaced in any segment
                if visual_id in [vid for vids in iteration_replaced_visual_ids.values() for vid in vids]:
                    old_asset_urls_by_visual_id[visual_id] = old_url
            # Merge replacements into existing segments
            for segment_num, replacement_xml in updated_segments.items():
                # Extract replacement URL for logging
                replacement_match = re.search(r'<replacement_visual_url>(.*?)</replacement_visual_url>', replacement_xml, re.DOTALL | re.IGNORECASE)
                replacement_url = replacement_match.group(1).strip() if replacement_match else "N/A"
                print(f"  Merging replacement for segment {segment_num} with URL: {replacement_url[:80]}...")
                
                final_graphics_definition = update_final_graphics_definition_with_replacements(
                    final_graphics_definition,
                    segment_num,
                    replacement_xml,
                )
            
            df.at[row_index, "final_graphics_definition"] = final_graphics_definition
            print(f"  Updated {len(updated_segments)} segment(s) in graphics definition")
            
            # Process video frames in the final merged graphics definition BEFORE saving
            # This converts any YouTube URLs with only start timestamps (still frames) to Drive images
            # This ensures that when we save to sheet, all YouTube frames have been converted to Drive images
            print(f"  Checking for video frames to process in final graphics definition...")
            processed_final_def = process_video_frames_in_text_format(final_graphics_definition, drive)
            if processed_final_def != final_graphics_definition:
                final_graphics_definition = processed_final_def
                df.at[row_index, "final_graphics_definition"] = final_graphics_definition
                print(f"  ✅ Processed video frames in final graphics definition (converted YouTube still frames to Drive images)")
            else:
                print(f"  No video frames found to process in final graphics definition")
            
            # Save to sheet immediately after revision and frame processing
            # This ensures the sheet always has the fully processed final_graphics_definition with Drive images
            if ws is not None:
                print("  Saving revision to sheet immediately (with processed video frames)...")
                with _sheet_lock:
                    save_to_sheet(ws, df)
                # CRITICAL: Refresh the dataframe row reference after saving to ensure we have the latest data
                # This ensures that when we read from df.loc[row_index] in the next attempt, we get the updated values
                row = df.loc[row_index]
            
            # CRITICAL: Update segments_map with merged data for next review/iteration
            # This ensures that follow-up reviews use the LATEST visuals (replacements), not stale original visuals
            # The segments_map is built from the updated final_graphics_definition which now contains the replacement visuals
            # in text format (parsed from the XML replacement format)
            segments_map = build_segment_visual_map(voiceover_text, final_graphics_definition)
            print(f"  ✅ Updated segments_map with latest visuals (including replacements) for next review")
            
            # Update revision tracking for this loop
            revision_tracking = update_revision_tracking(
                revision_tracking, criterion_name, attempt, iteration_replaced_visual_ids, segments_map
            )
        else:
            print("  WARNING: Revision produced no updates; moving to regeneration.")
            break

    # After review-revise loop completes (after MAX_REVIEW_ATTEMPTS iterations), do one final follow-up review
    # This checks if the last revision fixed the issues before moving to regeneration
    print(f"\n{'='*60}")
    print(f"Final Follow-up Review {criterion_name}: Row {row_index + 1} - Checking if last revision fixed issues")
    print(f"{'='*60}")
    
    # Refresh data for final review
    row = df.loc[row_index]
    final_graphics_definition = _safe_str(row.get("final_graphics_definition", ""))
    segments_map = build_segment_visual_map(voiceover_text, final_graphics_definition)
    segment_nums = list(segments_map.keys())
    
    # Build feedback string from previous failures for final follow-up review
    previous_feedback_lines = []
    if replaced_visual_ids_by_segment:
        for segment_num in replaced_visual_ids_by_segment.keys():
            if segment_num in last_feedback_by_segment:
                previous_feedback_lines.append(f"Segment {segment_num}:\n{last_feedback_by_segment[segment_num]}")
    else:
        for segment_num in segment_nums:
            if segment_num in last_feedback_by_segment:
                previous_feedback_lines.append(f"Segment {segment_num}:\n{last_feedback_by_segment[segment_num]}")
    previous_feedback = "\n\n".join(previous_feedback_lines)
    
    # Perform final follow-up review
    if replaced_visual_ids_by_segment:
        print(f"  Final follow-up review with {sum(len(vids) for vids in replaced_visual_ids_by_segment.values())} replaced visual(s) only")
        final_verdict, final_failures, _, _ = review_slide_segments_followup(
            course_name=course_name,
            target_audience=target_audience,
            topic_name=topic_name,
            subtopic_name=subtopic_name,
            slide_id=slide_id,
            slide_title=slide_title,
            slide_chunk=slide_chunk,
            segments_map=segments_map,
            visual_ids_by_segment=replaced_visual_ids_by_segment,
            old_asset_urls_by_visual_id=old_asset_urls_by_visual_id,
            previous_feedback=previous_feedback,
            drive=drive,
            llm=llm,
            conversation_history=conversation_history,
            criterion_name=criterion_name,
            visual_assignment_strategy=visual_assignment_strategy,
        )
    else:
        # Fall back to regular review if no replaced visuals
        print(f"  Final review (no replaced visuals to track)")
        final_verdict, final_failures, _, _ = review_slide_segments(
            prompt_template=prompt_template,
            course_name=course_name,
            target_audience=target_audience,
            topic_name=topic_name,
            subtopic_name=subtopic_name,
            slide_id=slide_id,
            slide_title=slide_title,
            slide_chunk=slide_chunk,
            segments_map=segments_map,
            segment_nums=segment_nums,
            drive=drive,
            llm=llm,
            conversation_history=conversation_history,
            criterion_name=criterion_name,
        )
    
    # If final review passes, we're done
    if final_verdict == "PASS":
        print(f"✓ {criterion_name} review PASSED for row {row_index + 1} after final follow-up review")
        # Mark remaining loops as "No replacement"
        row = df.loc[row_index]
        final_graphics_definition = _safe_str(row.get("final_graphics_definition", ""))
        segments_map = build_segment_visual_map(voiceover_text, final_graphics_definition)
        total_loops = MAX_REVIEW_ATTEMPTS + MAX_REGEN_ATTEMPTS
        for loop_num in range(MAX_REVIEW_ATTEMPTS + 1, total_loops + 1):  # Remaining regeneration loops (didn't run)
            revision_tracking = update_revision_tracking(
                revision_tracking, criterion_name, loop_num, {}, segments_map
            )
        return "PASS", revision_tracking
    
    # Final review failed - extract feedback for regeneration
    print(f"  Final follow-up review FAILED - proceeding to regeneration with latest feedback")
    failed_segments: List[int] = []
    feedback_by_segment: Dict[int, str] = {}
    for failure in final_failures:
        segment_id_text = failure.get("segment_id", "")
        segment_num = _parse_segment_marker(segment_id_text)
        # Handle "1 Visual for the whole Slide" case where LLM might return "SLIDE_1" instead of "SEGMENT 1"
        if not segment_num and visual_assignment_strategy == "1 Visual for the whole Slide":
            # If it's "SLIDE_X" format, map to segment 1
            if re.search(r"SLIDE_", segment_id_text, re.IGNORECASE):
                segment_num = 1
        if not segment_num:
            continue
        if segment_num not in failed_segments:
            failed_segments.append(segment_num)
        failing_visual_id = failure.get('failing_visual_ids', '')
        
        # Extract voiceover_part for the failing visual
        voiceover_part = ""
        segment = segments_map.get(segment_num, {})
        visual_steps = segment.get("visual_steps", [])
        for step in visual_steps:
            if step.get('visual_id') == failing_visual_id:
                voiceover_part = step.get('voiceover_part', '')
                break
        
        # Build failure text with voiceover part
        failure_text = f"Failing Visual: {failing_visual_id}"
        if voiceover_part:
            failure_text += f"\nVoiceover sentence to which this visual was assigned: {voiceover_part}"
        failure_text += f"\nReason: {failure.get('reason', '')}\nNeeded: {failure.get('needed_visual', '')}"
        
        if segment_num in feedback_by_segment:
            feedback_by_segment[segment_num] += "\n\n" + failure_text
        else:
            feedback_by_segment[segment_num] = failure_text
    
    if not failed_segments or not feedback_by_segment:
        print("  No failed segments or feedback found from final review; stopping.")
        # Mark all remaining loops as "No replacement"
        row = df.loc[row_index]
        final_graphics_definition = _safe_str(row.get("final_graphics_definition", ""))
        segments_map = build_segment_visual_map(voiceover_text, final_graphics_definition)
        for loop_num in range(MAX_REVIEW_ATTEMPTS + 1, 7):
            revision_tracking = update_revision_tracking(
                revision_tracking, criterion_name, loop_num, {}, segments_map
            )
        return "FAIL", revision_tracking
    
    # Use the feedback from the final follow-up review for regeneration
    print(f"\n{'='*60}")
    print(f"Regeneration {criterion_name}: Row {row_index + 1} - Using feedback from final follow-up review")
    print(f"{'='*60}")
    print(f"  Regenerating {len(failed_segments)} failed segment(s): {failed_segments}")
    print(f"  Using feedback from the final follow-up review (most recent feedback)")
    print(f"  Continuing conversation history from review-revise loop")
    
    # Track replaced visuals across regeneration attempts (accumulate across all attempts)
    all_replaced_visual_ids_by_segment: Dict[int, List[str]] = {}
    all_old_asset_urls_by_visual_id: Dict[str, str] = {}
    # Persist feedback across regeneration attempts for follow-up reviews
    regen_last_feedback_by_segment: Dict[int, str] = feedback_by_segment.copy()
    regen_last_failed_segments: List[int] = failed_segments.copy()
    
    for attempt in range(1, MAX_REGEN_ATTEMPTS + 1):
        print(f"\n  Regeneration attempt {attempt}/{MAX_REGEN_ATTEMPTS}")
        replaced_visual_ids_by_segment, old_asset_urls_by_visual_id = regenerate_failed_segments(
            row_index=row_index,
            row=row,
            df=df,
            course_name=course_name,
            target_audience=target_audience,
            drive=drive,
            llm=llm,
            failed_segments=failed_segments,
            feedback_by_segment=feedback_by_segment,
            ws=ws,
            use_only_drive_and_hvac=use_only_drive_and_hvac,
        )
        
        # Save to sheet immediately after regeneration (with processed video frames)
        # This ensures the sheet always has the fully processed final_graphics_definition with Drive images
        if ws is not None:
            print("  Saving regeneration to sheet immediately (with processed video frames)...")
            with _sheet_lock:
                save_to_sheet(ws, df)
        
        # Accumulate replaced visuals across attempts
        for segment_num, visual_ids in replaced_visual_ids_by_segment.items():
            if segment_num not in all_replaced_visual_ids_by_segment:
                all_replaced_visual_ids_by_segment[segment_num] = []
            all_replaced_visual_ids_by_segment[segment_num].extend(visual_ids)
        all_old_asset_urls_by_visual_id.update(old_asset_urls_by_visual_id)
        
        # CRITICAL: Refresh row reference and build segments_map from updated final_graphics_definition
        # This ensures that follow-up review uses the LATEST regenerated visuals, not stale original visuals
        # The final_graphics_definition now contains the regenerated visuals in text format (merged from XML)
        row = df.loc[row_index]
        final_graphics_definition = _safe_str(row.get("final_graphics_definition", ""))
        
        # Review the regenerated visuals using follow-up mechanism in the same conversation
        # The segments_map is built from the updated final_graphics_definition which contains the regenerated visuals
        segments_map = build_segment_visual_map(voiceover_text, final_graphics_definition, visual_assignment_strategy, slide_chunk)
        print(f"  ✅ Built segments_map with latest regenerated visuals for follow-up review")
        
        # Update revision tracking for regeneration loop (dynamic based on MAX_REVIEW_ATTEMPTS)
        regen_loop_num = MAX_REVIEW_ATTEMPTS + attempt  # First regen loop = MAX_REVIEW_ATTEMPTS + 1
        revision_tracking = update_revision_tracking(
            revision_tracking, criterion_name, regen_loop_num, replaced_visual_ids_by_segment, segments_map
        )
        
        # Build previous feedback for follow-up review (use persisted feedback from last attempt)
        previous_feedback_lines = []
        for segment_num in regen_last_failed_segments:
            if segment_num in regen_last_feedback_by_segment:
                previous_feedback_lines.append(f"Segment {segment_num}:\n{regen_last_feedback_by_segment[segment_num]}")
        previous_feedback = "\n\n".join(previous_feedback_lines)
        
        # Use follow-up review to continue the same conversation
        if conversation_history and replaced_visual_ids_by_segment:
            print(f"  Using follow-up review in same conversation for regenerated visuals")
            verdict, failures, _, conversation_history = review_slide_segments_followup(
                course_name=course_name,
                target_audience=target_audience,
                topic_name=topic_name,
                subtopic_name=subtopic_name,
                slide_id=slide_id,
                slide_title=slide_title,
                slide_chunk=slide_chunk,
                segments_map=segments_map,
                visual_ids_by_segment=replaced_visual_ids_by_segment,
                old_asset_urls_by_visual_id=old_asset_urls_by_visual_id,
                previous_feedback=previous_feedback,
                drive=drive,
                llm=llm,
                conversation_history=conversation_history,
                criterion_name=criterion_name,
                visual_assignment_strategy=visual_assignment_strategy,
            )
        else:
            # Fallback: if no conversation history or no replaced visuals, use regular review
            print(f" Using regular review (no conversation history or no replaced visuals)")
            segment_nums = failed_segments
            verdict, failures, _, conversation_history = review_slide_segments(
                prompt_template=prompt_template,
                course_name=course_name,
                target_audience=target_audience,
                topic_name=topic_name,
                subtopic_name=subtopic_name,
                slide_id=slide_id,
                slide_title=slide_title,
                slide_chunk=slide_chunk,
                segments_map=segments_map,
                segment_nums=segment_nums,
                drive=drive,
                llm=llm,
                conversation_history=conversation_history,
                criterion_name=criterion_name,
            )
        
        if verdict == "PASS":
            print(f"✓ {criterion_name} regeneration PASSED for row {row_index + 1} on attempt {attempt}")
            # Mark remaining regeneration loops as "No replacement"
            total_loops = MAX_REVIEW_ATTEMPTS + MAX_REGEN_ATTEMPTS
            for remaining_loop in range(regen_loop_num + 1, total_loops + 1):
                revision_tracking = update_revision_tracking(
                    revision_tracking, criterion_name, remaining_loop, {}, segments_map
                )
            return "PASS", revision_tracking
        
        # Update feedback for next regeneration attempt if still failing
        if failures:
            failed_segments = []
            feedback_by_segment = {}
            for failure in failures:
                segment_num = _parse_segment_marker(failure.get("segment_id", ""))
                if not segment_num:
                    continue
                if segment_num not in failed_segments:
                    failed_segments.append(segment_num)
                failing_visual_id = failure.get('failing_visual_ids', '')
                
                # Extract voiceover_part for the failing visual
                voiceover_part = ""
                segment = segments_map.get(segment_num, {})
                visual_steps = segment.get("visual_steps", [])
                for step in visual_steps:
                    if step.get('visual_id') == failing_visual_id:
                        voiceover_part = step.get('voiceover_part', '')
                        break
                
                # Build failure text with voiceover part
                failure_text = f"Failing Visual: {failing_visual_id}"
                if voiceover_part:
                    failure_text += f"\nVoiceover sentence to which this visual was assigned: {voiceover_part}"
                failure_text += f"\nReason: {failure.get('reason', '')}\nNeeded: {failure.get('needed_visual', '')}"
                
                if segment_num in feedback_by_segment:
                    feedback_by_segment[segment_num] += "\n\n" + failure_text
                else:
                    feedback_by_segment[segment_num] = failure_text
        
        # Persist feedback for next regeneration attempt's follow-up review
        regen_last_failed_segments = failed_segments
        regen_last_feedback_by_segment = feedback_by_segment.copy()
    

    return "FAIL", revision_tracking


def find_repeated_urls(group_segments, min_occurrences=4):
    """
    Find URLs that appear more than min_occurrences times across all slides/segments.
    
    :param group_segments: Map of slide_id -> segment_num -> segment data
    :param min_occurrences: Minimum number of occurrences to be considered repeated (default 4, meaning >3)
    :return: Dict mapping repeated URL -> list of (slide_id, segment_num, visual_id) tuples where it appears
    """
    url_occurrences: Dict[str, List[Tuple[str, int, str]]] = {}
    
    for slide_id, segments_map in group_segments.items():
        for segment_num, segment in segments_map.items():
            visual_steps = segment.get("visual_steps", [])
            for step in visual_steps:
                asset_url = step.get("asset", "").strip()
                if not asset_url:
                    continue
                visual_id = step.get("visual_id", "")
                if asset_url not in url_occurrences:
                    url_occurrences[asset_url] = []
                url_occurrences[asset_url].append((slide_id, segment_num, visual_id))
    
    # Filter to only URLs that appear more than min_occurrences times
    repeated_urls = {
        url: occurrences 
        for url, occurrences in url_occurrences.items() 
        if len(occurrences) >= min_occurrences
    }
    
    return repeated_urls


def filter_segments_by_url(group_segments, target_url):
    """
    Filter group_segments to only include segments that use the target URL.
    
    :param group_segments: Map of slide_id -> segment_num -> segment data
    :param target_url: URL to filter by
    :return: Tuple of (filtered_segments, target_keys) where target_keys is list of segment keys
    """
    filtered_segments: Dict[str, Dict[int, Dict[str, object]]] = {}
    target_keys: List[str] = []
    
    for slide_id, segments_map in group_segments.items():
        for segment_num, segment in segments_map.items():
            visual_steps = segment.get("visual_steps", [])
            # Filter visual_steps to only those using the target URL
            filtered_steps = [
                step for step in visual_steps 
                if step.get("asset", "").strip() == target_url
            ]
            
            # Only include segment if it has at least one visual using the target URL
            if filtered_steps:
                if slide_id not in filtered_segments:
                    filtered_segments[slide_id] = {}
                filtered_segments[slide_id][segment_num] = {
                    "vo_text": segment.get("vo_text", ""),
                    "visual_steps": filtered_steps,
                    "segment_text": segment.get("segment_text", ""),
                }
                target_keys.append(f"{slide_id}_SEGMENT_{segment_num}")
    
    return filtered_segments, target_keys


@traceable(
    metadata={
        "agent_name": "graphics_definition_v2",
        "step_name": "Review Agent",
        "function_name": "run_redundancy_loop_for_topic",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous")
    }
)
def run_redundancy_loop_for_topic(topic_key, row_indices, df, course_name, target_audience, drive, llm, use_only_drive_and_hvac=False):
    """
    Run redundancy review for a topic: find repeated URLs and run review-revise per URL.

    :param topic_key: Tuple of (topic_name, subtopic_name)
    :param row_indices: List of row indices for slides in the topic
    :param df: Dataframe (shared)
    :param course_name: Course name
    :param target_audience: Target audience
    :param drive: Google Drive instance
    :param llm: LLM model name
    :param use_only_drive_and_hvac: If True, skip web search and other-channels video search during regeneration.
    :return: "PASS" or "FAIL" overall status for the topic
    """
    
    topic_name, subtopic_name = topic_key
    
    # Build group_segments for all slides in the topic
    group_segments: Dict[str, Dict[int, Dict[str, object]]] = {}
    for row_index in row_indices:
        row = df.loc[row_index]
        voiceover_text = _safe_str(row.get("voiceover_segment", ""))
        final_graphics_definition = _safe_str(row.get("final_graphics_definition", ""))
        slide_id = f"SLIDE_{row_index + 1}"
        group_segments[slide_id] = build_segment_visual_map(voiceover_text, final_graphics_definition)

    # Find repeated URLs (>3 occurrences)
    repeated_urls = find_repeated_urls(group_segments, min_occurrences=4)
    
    if not repeated_urls:
        print(f"  No URLs found with more than 3 occurrences. Skipping redundancy review for topic {topic_name}")
        return "PASS"  # No redundancy issues if nothing is repeated
    
    print(f"  Found {len(repeated_urls)} repeated URL(s) with 4+ occurrences:")
    for url, occurrences in repeated_urls.items():
        print(f"    - {url[:80]}... ({len(occurrences)} occurrences)")
    
    # Run separate redundancy review loop for each repeated URL
    overall_status = "PASS"
    for repeated_url, occurrences in repeated_urls.items():
        print(f"\n{'='*60}")
        print(f"Reviewing redundancy for URL: {repeated_url[:80]}... ({len(occurrences)} occurrences)")
        print(f"{'='*60}")
        
        # Filter segments to only those using this repeated URL (for target_keys)
        filtered_segments, target_keys = filter_segments_by_url(group_segments, repeated_url)
        
        if not filtered_segments:
            print(f"  WARNING: No segments found using this URL after filtering. Skipping.")
            continue
        
        print(f"  Found {len(target_keys)} segment(s) using this URL out of {sum(len(segments_map) for segments_map in group_segments.values())} total segment(s) in topic")
        
        # Run redundancy review-revise loop for this specific repeated URL
        url_status = _run_redundancy_loop_for_url(
            topic_name=topic_name,
            subtopic_name=subtopic_name,
            repeated_url=repeated_url,
            all_segments=group_segments, 
            target_keys=target_keys,  
            row_indices=row_indices,
            df=df,
            course_name=course_name,
            target_audience=target_audience,
            drive=drive,
            llm=llm,
            use_only_drive_and_hvac=use_only_drive_and_hvac,
        )
        
        if url_status == "FAIL":
            overall_status = "FAIL"
    
    return overall_status


def _run_redundancy_loop_for_url(topic_name, subtopic_name, repeated_url, all_segments, target_keys, row_indices, df, course_name, target_audience, drive, llm, use_only_drive_and_hvac=False):
    """
    Run redundancy review-revise loop for a specific repeated URL.

    :param topic_name: Topic name
    :param subtopic_name: Subtopic name
    :param repeated_url: The repeated URL being evaluated
    :param all_segments: All segments in the topic (for full context in review_targets)
    :param target_keys: Only segments in this list will be evaluated (those using the repeated URL)
    :param row_indices: List of row indices for the topic
    :param df: Dataframe (shared)
    :param course_name: Course name
    :param target_audience: Target audience
    :param drive: Google Drive instance
    :param llm: LLM model name
    :param use_only_drive_and_hvac: If True, skip web search and other-channels video search during regeneration.
    :return: "PASS" if all occurrences are acceptable, "FAIL" if any need replacement
    """
    conversation_history: Optional[List[types.Content]] = None
    feedback_map: Dict[Tuple[int, int], str] = {}
    group_segments = all_segments.copy()  # Use all segments for context

    for attempt in range(1, MAX_REVIEW_ATTEMPTS + 1):
        print(f"\n{'='*60}")
        print(f"Review redundancy: topic {topic_name} attempt {attempt}/{MAX_REVIEW_ATTEMPTS}")
        print(f"{'='*60}")
        
        # Use follow-up review if we have conversation history (after first attempt with revisions)
        if conversation_history and attempt > 1:
            # Build feedback string from previous failures
            previous_feedback_lines = []
            for key in target_keys:
                match = re.search(r"SLIDE_(\d+)_SEGMENT_(\d+)", key)
                if match:
                    slide_index = int(match.group(1)) - 1
                    segment_num = int(match.group(2))
                    if (slide_index, segment_num) in feedback_map:
                        previous_feedback_lines.append(f"{key}:\n{feedback_map[(slide_index, segment_num)]}")
            previous_feedback = "\n\n".join(previous_feedback_lines)
            
            verdict, failures, _, conversation_history = review_topic_segments_followup(
                course_name=course_name,
                target_audience=target_audience,
                topic_name=topic_name,
                subtopic_name=subtopic_name,
                group_segments=group_segments,
                target_keys=target_keys,
                previous_feedback=previous_feedback,
                drive=drive,
                llm=llm,
                repeated_visual_url=repeated_url,
                conversation_history=conversation_history,
            )
        else:
            # First review or fresh start
            verdict, failures, _, conversation_history = review_topic_segments(
                course_name=course_name,
                target_audience=target_audience,
                topic_name=topic_name,
                subtopic_name=subtopic_name,
                group_segments=group_segments,
                target_keys=target_keys,
                drive=drive,
                llm=llm,
                repeated_visual_url=repeated_url,
                conversation_history=conversation_history,
            )
        
        if verdict == "PASS":
            print(f"✓ Redundancy review PASSED for topic {topic_name}")
            return "PASS"

        failed_keys = []
        # Update feedback_map with new failures (accumulate across iterations)
        for failure in failures:
            segment_key = failure.get("segment_key", "")
            match = re.search(r"SLIDE_(\d+)_SEGMENT_(\d+)", segment_key)
            if not match:
                continue
            slide_index = int(match.group(1)) - 1
            segment_num = int(match.group(2))
            if segment_key not in failed_keys:
                failed_keys.append(segment_key)
            failing_visual_id = failure.get('failing_visual_ids', '')
            
            # Extract voiceover_part for the failing visual
            voiceover_part = ""
            slide_id = f"SLIDE_{slide_index + 1}"
            segments_map = group_segments.get(slide_id, {})
            segment = segments_map.get(segment_num, {})
            visual_steps = segment.get("visual_steps", [])
            for step in visual_steps:
                if step.get('visual_id') == failing_visual_id:
                    voiceover_part = step.get('voiceover_part', '')
                    break
            
            # Build failure text with voiceover part
            failure_text = f"Failing Visual: {failing_visual_id}"
            if voiceover_part:
                failure_text += f"\nVoiceover sentence to which this visual was assigned: {voiceover_part}"
            failure_text += f"\nReason: {failure.get('reason', '')}\nNeeded: {failure.get('needed_visual', '')}"
            
            key = (slide_index, segment_num)
            if key in feedback_map:
                feedback_map[key] += "\n\n" + failure_text
            else:
                feedback_map[key] = failure_text

        if not failed_keys:
            print("  No failed keys found; stopping redundancy review loop.")
            break

        print(f"  Attempting revision for {len(failed_keys)} redundant segment(s)")
        target_keys = failed_keys

        for (slide_index, segment_num), feedback in feedback_map.items():
            row = df.loc[slide_index]
            slide_chunk = _safe_str(row.get("Slide Chunk", ""))
            visual_assignment_strategy = str(row.get("Visual Assignment Strategy", "Flexible, let the agent decide")).strip()
            if not visual_assignment_strategy or visual_assignment_strategy == "nan":
                visual_assignment_strategy = "Flexible, let the agent decide"
            
            segments_map = build_segment_visual_map(
                _safe_str(row.get("voiceover_segment", "")),
                _safe_str(row.get("final_graphics_definition", "")),
                visual_assignment_strategy,
                slide_chunk,
            )
            segment = segments_map.get(segment_num, {})
            current_visuals = "\n".join([
                f"{step.get('visual_id')} | When VO: \"{step.get('voiceover_part', '')}\" | Visual assigned: {step.get('asset', '')}"
                for step in segment.get("visual_steps", [])
            ])
            
            # For "1 Visual for the whole Slide", use slide_chunk as vo_text when segment_num == 1
            actual_vo_text = _safe_str(segment.get("vo_text", ""))
            if visual_assignment_strategy == "1 Visual for the whole Slide" and segment_num == 1:
                actual_vo_text = slide_chunk
            
            revised = revise_segment_visuals(
                course_name=course_name,
                target_audience=target_audience,
                topic_name=_safe_str(row.get("Topic", "")),
                subtopic_name=_safe_str(row.get("Subtopic", "")),
                slide_title=_safe_str(row.get("Slide Chunk Title", "")),
                slide_chunk=slide_chunk,
                vo_text=actual_vo_text,
                current_visuals=current_visuals,
                feedback=feedback,
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
            )
            if revised:
                df.at[slide_index, "final_graphics_definition"] = update_final_graphics_definition_with_replacements(
                    _safe_str(row.get("final_graphics_definition", "")),
                    segment_num,
                    revised,
                )
                print(f"    Updated SLIDE_{slide_index + 1} SEGMENT_{segment_num}")

        print("  Refreshing segment maps after revision...")
        for row_index in row_indices:
            row = df.loc[row_index]
            slide_id = f"SLIDE_{row_index + 1}"
            group_segments[slide_id] = build_segment_visual_map(
                _safe_str(row.get("voiceover_segment", "")),
                _safe_str(row.get("final_graphics_definition", "")),
            )

    # After review-revise loop completes (after 3 attempts), use the feedback from loop 3 for regeneration
    # feedback_map already contains the feedback from the last review attempt
    if not feedback_map:
        print("  No feedback found; stopping.")
        return "FAIL"
    
    # Use the feedback from the last review-revise attempt for regeneration
    print(f"\n{'='*60}")
    print(f"Regeneration redundancy: topic {topic_name} - Using feedback from review-revise loop")
    print(f"{'='*60}")
    print(f"  Regenerating {len(feedback_map)} redundant segment(s) across {len(set(slide_idx for slide_idx, _ in feedback_map.keys()))} slide(s)")
    print(f"  Using feedback from the last review attempt (loop 3)")
    
    for attempt in range(1, MAX_REGEN_ATTEMPTS + 1):
        print(f"\n  Regeneration attempt {attempt}/{MAX_REGEN_ATTEMPTS}")
        target_keys = [f"SLIDE_{slide_index + 1}_SEGMENT_{segment_num}" for slide_index, segment_num in feedback_map.keys()]

        grouped: Dict[int, Dict[int, str]] = {}
        for (slide_index, segment_num), feedback in feedback_map.items():
            grouped.setdefault(slide_index, {})[segment_num] = feedback

        for slide_index, feedback_by_segment in grouped.items():
            row = df.loc[slide_index]
            print(f"  Regenerating SLIDE_{slide_index + 1} segments: {list(feedback_by_segment.keys())}")
            # Note: Return values not used for redundancy regeneration as it uses different review mechanism
            _, _ = regenerate_failed_segments(
                row_index=slide_index,
                row=row,
                df=df,
                course_name=course_name,
                target_audience=target_audience,
                drive=drive,
                llm=llm,
                failed_segments=list(feedback_by_segment.keys()),
                feedback_by_segment=feedback_by_segment,
                ws=None,
                use_only_drive_and_hvac=use_only_drive_and_hvac,
            )
        
        # Review the regenerated visuals to check if they pass
        print("  Refreshing segment maps after regeneration...")
        # Rebuild full group_segments from dataframe
        full_group_segments: Dict[str, Dict[int, Dict[str, object]]] = {}
        for row_index in row_indices:
            row = df.loc[row_index]
            slide_id = f"SLIDE_{row_index + 1}"
            full_group_segments[slide_id] = build_segment_visual_map(
                _safe_str(row.get("voiceover_segment", "")),
                _safe_str(row.get("final_graphics_definition", "")),
            )
        
        # Re-filter segments to only those still using the repeated URL (in case some were replaced)
        filtered_segments_after_regen, target_keys_after_regen = filter_segments_by_url(full_group_segments, repeated_url)
        
        if not filtered_segments_after_regen:
            print(f"  All occurrences of repeated URL have been replaced. PASS for this URL.")
            return "PASS"
        
        # Update group_segments and target_keys for next iteration
        group_segments = filtered_segments_after_regen
        target_keys = target_keys_after_regen
        
        verdict, failures, _, _ = review_topic_segments(
            course_name=course_name,
            target_audience=target_audience,
            topic_name=topic_name,
            subtopic_name=subtopic_name,
            group_segments=group_segments,
            target_keys=target_keys,
            drive=drive,
            llm=llm,
            repeated_visual_url=repeated_url,
            conversation_history=None,
        )
        if verdict == "PASS":
            print(f"✓ Redundancy regeneration PASSED for topic {topic_name} on attempt {attempt}")
            return "PASS"
        
        # Update feedback for next regeneration attempt if still failing
        if failures:
            feedback_map = {}
            for failure in failures:
                segment_key = failure.get("segment_key", "")
                match = re.search(r"SLIDE_(\d+)_SEGMENT_(\d+)", segment_key)
                if not match:
                    continue
                slide_index = int(match.group(1)) - 1
                segment_num = int(match.group(2))
                failing_visual_id = failure.get('failing_visual_ids', '')
                
                # Extract voiceover_part for the failing visual
                voiceover_part = ""
                slide_id = f"SLIDE_{slide_index + 1}"
                segments_map = group_segments.get(slide_id, {})
                segment = segments_map.get(segment_num, {})
                visual_steps = segment.get("visual_steps", [])
                for step in visual_steps:
                    if step.get('visual_id') == failing_visual_id:
                        voiceover_part = step.get('voiceover_part', '')
                        break
                
                # Build failure text with voiceover part
                failure_text = f"Failing Visual: {failing_visual_id}"
                if voiceover_part:
                    failure_text += f"\nVoiceover sentence to which this visual was assigned: {voiceover_part}"
                failure_text += f"\nReason: {failure.get('reason', '')}\nNeeded: {failure.get('needed_visual', '')}"
                
                key = (slide_index, segment_num)
                if key in feedback_map:
                    feedback_map[key] += "\n\n" + failure_text
                else:
                    feedback_map[key] = failure_text

        print("  Refreshing segment maps after regeneration...")
        for row_index in row_indices:
            row = df.loc[row_index]
            slide_id = f"SLIDE_{row_index + 1}"
            group_segments[slide_id] = build_segment_visual_map(
                _safe_str(row.get("voiceover_segment", "")),
                _safe_str(row.get("final_graphics_definition", "")),
            )

    return "FAIL"


# Lock for thread-safe sheet operations
_sheet_lock = threading.Lock()


@traceable(
    metadata={
        "agent_name": "graphics_definition_v2",
        "step_name": "Review Agent",
        "function_name": "process_review_revise_row",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous"),
    }
)
def process_review_revise_row(row_index, df, course_name, target_audience, drive, llm, ws, review_cols, use_only_drive_and_hvac=False):
    """
    Process a single row's review-revise workflow (alignment and specificity reviews for one slide).

    :param row_index: The row index to process
    :param df: The dataframe (shared across threads, but each thread only modifies its own row)
    :param course_name: Course name
    :param target_audience: Target audience
    :param drive: Google Drive instance
    :param llm: Language model to use
    :param ws: Worksheet object
    :param review_cols: List of review columns
    :param use_only_drive_and_hvac: If True, skip web search and other-channels video search during regeneration.
    :return: None
    """
    
    try:
        row = df.loc[row_index]
        print("\n" + "=" * 80)
        print(f"Processing row {row_index + 1}")
        print(f"Slide Title: {_safe_str(row.get('Slide Chunk Title', ''))}")
        print("=" * 80)

        # Transition slides: skip review-revise; set columns and return
        slide_type = str(row.get("Slide Type", "")).strip().lower()
        if slide_type in ("transition", "transition slide"):
            df.at[row_index, "graphics_review_v2_notes"] = "-"
            df.at[row_index, "review_complete"] = "TRUE"
            df.at[row_index, "revision_tracking"] = "-"
            if ws is not None:
                with _sheet_lock:
                    save_to_sheet(ws, df)
            print(f"  Skipping row {row_index + 1}: Transition slide (review disabled)")
            return

        # Initialize revision tracking for this row
        voiceover_text = _safe_str(row.get("voiceover_segment", ""))
        final_graphics_definition = _safe_str(row.get("final_graphics_definition", ""))
        slide_chunk = _safe_str(row.get("Slide Chunk", ""))
        visual_assignment_strategy = str(row.get("Visual Assignment Strategy", "Flexible, let the agent decide")).strip()
        if not visual_assignment_strategy or visual_assignment_strategy == "nan":
            visual_assignment_strategy = "Flexible, let the agent decide"
        
        segments_map = build_segment_visual_map(voiceover_text, final_graphics_definition, visual_assignment_strategy, slide_chunk)
        revision_tracking = initialize_revision_tracking(segments_map)
        print(f"  Initialized revision tracking for {len(revision_tracking)} visual(s)")

        print("\n[STEP 1] Reviewing ALIGNMENT criterion...")
        alignment_status, revision_tracking = run_review_loop_for_slide(
            criterion_name="alignment",
            prompt_template=ALIGNMENT_REVIEW_PROMPT,
            row_index=row_index,
            row=row,
            df=df,
            course_name=course_name,
            target_audience=target_audience,
            drive=drive,
            llm=llm,
            ws=ws,
            revision_tracking=revision_tracking,
            use_only_drive_and_hvac=use_only_drive_and_hvac,
        )

        row = df.loc[row_index]
        # Update segments_map after alignment (kept for compatibility/debug parity with prior flow)
        final_graphics_definition = _safe_str(row.get("final_graphics_definition", ""))
        segments_map = build_segment_visual_map(voiceover_text, final_graphics_definition, visual_assignment_strategy, slide_chunk)

        # Specificity review loop intentionally disabled.
        # Rationale: alignment prompt now includes a light specificity/clarity check to avoid
        # a second revision pass that can overwrite good alignment replacements.
        specificity_status = "SKIPPED"
        # print("\n[STEP 2] Reviewing SPECIFICITY criterion...")
        # specificity_status, revision_tracking = run_review_loop_for_slide(
        #     criterion_name="specificity",
        #     prompt_template=SPECIFICITY_REVIEW_PROMPT,
        #     row_index=row_index,
        #     row=row,
        #     df=df,
        #     course_name=course_name,
        #     target_audience=target_audience,
        #     drive=drive,
        #     llm=llm,
        #     ws=ws,
        #     revision_tracking=revision_tracking,
        #     use_only_drive_and_hvac=use_only_drive_and_hvac,
        # )
        
        # Format and save revision tracking
        tracking_text = format_revision_tracking(revision_tracking)
        df.at[row_index, "revision_tracking"] = tracking_text
        print(f"  Saved revision tracking for {len(revision_tracking)} visual(s)")

        df.at[row_index, "graphics_review_v2_notes"] = f"alignment={alignment_status}"
        # Mark review as complete after alignment review is done.
        df.at[row_index, "review_complete"] = "TRUE"
        print(f"\nRow {row_index + 1} completed (alignment={alignment_status})")
        print(f"  Marked review_complete=TRUE for row {row_index + 1}")
        
        # Final check: Normalize YouTube URLs and convert video frames to Drive images
        # This ensures all YouTube links with timestamps are converted to Drive images before saving
        print(f"\n  Performing final YouTube URL normalization and frame conversion...")
        row = df.loc[row_index]  # Get fresh row data
        final_graphics_definition = _safe_str(row.get("final_graphics_definition", ""))
        
        if final_graphics_definition:
            # Step 1: Normalize t= parameter to start= parameter format
            normalized_def = normalize_youtube_timestamp_urls(final_graphics_definition)
            
            # Step 2: Process video frames (converts start= URLs to Drive images)
            # This handles both:
            # - Case 1: watch?v=...&start=... (normalized from t=)
            # - Case 2: embed/...?start=... (already in correct format)
            processed_def = process_video_frames_in_text_format(normalized_def, drive)
            
            if processed_def != final_graphics_definition:
                df.at[row_index, "final_graphics_definition"] = processed_def
                print(f"  ✅ Final graphics definition updated with Drive image URLs")
            else:
                print(f"  No YouTube frame URLs found to convert")
            
            # Step 3: Add (snapshot) label to Drive links from target folder
            row = df.loc[row_index]  # Refresh row data
            current_def = _safe_str(row.get("final_graphics_definition", ""))
            
            if current_def:
                labeled_def = add_snapshot_label_to_drive_links(current_def, drive)
                if labeled_def != current_def:
                    df.at[row_index, "final_graphics_definition"] = labeled_def
                    print(f"  ✅ Added (snapshot) labels to Drive links from target folder")
        
        # Save immediately to ensure review_complete is persisted (for resume logic)
        # Use lock to prevent concurrent writes to sheet
        if ws is not None:
            print(f"  Saving row {row_index + 1} to sheet immediately...")
            with _sheet_lock:
                save_to_sheet(ws, df)
            print(f"  ✓ Row {row_index + 1} saved successfully")
            
    except Exception as e:
        print(f"  ERROR processing row {row_index + 1}: {e}")
        # Mark row with error but still save
        df.at[row_index, "graphics_review_v2_notes"] = f"ERROR: {str(e)}"
        df.at[row_index, "review_complete"] = "FALSE"
        if ws is not None:
            with _sheet_lock:
                save_to_sheet(ws, df)
        raise  # Re-raise to be caught by the executor


@traceable(
    metadata={
        "agent_name": "graphics_definition_v2",
        "step_name": "Review and Revise Graphics Definition V2",
        "function_name": "run_review_and_revise_graphics_definition_v2_for_all_rows",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous"),
    }
)
def run_review_and_revise_graphics_definition_v2_for_all_rows(sheet, llm="gemini_3_flash_thinking", max_workers=50, use_only_drive_and_hvac=False):
    """
    Entry point: run alignment and specificity review-revise for all rows with voiceover and graphics definition.

    :param sheet: gspread sheet object
    :param llm: LLM model name
    :param max_workers: Number of parallel workers (1 for sequential)
    :param use_only_drive_and_hvac: If True, skip web search and other-channels video search during regeneration.
    :return: None
    """
    
    worksheet_name = "Slide Chunks"
    ws, df = get_sheet_data_and_df(sheet, worksheet_name)

    _, course_info_df = get_sheet_data_and_df(sheet, "Course info")
    course_name = _safe_str(course_info_df.loc[0, "Course Name"])
    target_audience = _safe_str(course_info_df.loc[0, "Target Audience & Industry"])

    review_cols = ["graphics_review_v2_notes", "review_complete", "revision_tracking"]
    for col in review_cols:
        if col not in df.columns:
            df[col] = ""

    print("Initializing Google Drive instance...")
    drive = get_drive_instance()
    if not drive:
        print("ERROR: Drive instance unavailable. Aborting review step.")
        return
    print("✓ Drive instance initialized successfully")

    rows_to_process = []
    for index, row in df.iterrows():
        voiceover_text = _safe_str(row.get("voiceover_segment", ""))
        final_graphics_definition = _safe_str(row.get("final_graphics_definition", ""))
        review_complete = _safe_str(row.get("review_complete", "")).strip().upper()
        if not voiceover_text or voiceover_text == "nan":
            continue
        if not final_graphics_definition or final_graphics_definition == "nan":
            continue
        # Skip rows that have already completed review (resume logic)
        if review_complete == "TRUE":
            print(f"  Skipping row {index + 1}: review_complete=TRUE")
            continue
        rows_to_process.append(index)

    if not rows_to_process:
        print("No rows to review. All rows already passed or missing data.")
        return

    print(f"\n{'='*80}")
    print(f"Starting Graphics Definition V2 Review & Revise")
    print(f"{'='*80}\n")

    progress = SmartProgressBar(total_tasks=len(rows_to_process), description="Graphics review (alignment)")

    # Process rows in parallel if max_workers > 1, otherwise sequential
    if max_workers > 1:
        print(f"🚀 Processing {len(rows_to_process)} row(s) in parallel with {max_workers} worker(s)...\n")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all rows for processing
            futures = {
                executor.submit(
                    process_review_revise_row,
                    row_index,
                    df,  # Shared dataframe - each thread only modifies its own row
                    course_name,
                    target_audience,
                    drive,
                    llm,
                    ws,
                    review_cols,
                    use_only_drive_and_hvac,
                ): row_index
                for row_index in rows_to_process
            }
            
            # Collect results as they complete
            for future in as_completed(futures):
                row_index = futures[future]
                try:
                    future.result()  # This will save inside the function
                    progress.update()
                    print(f"✓ Row {row_index + 1} completed and saved")
                except Exception as e:
                    print(f"❌ Error processing row {row_index + 1}: {e}")
                    progress.update()
                    # Error handling is done inside process_review_revise_row
    else:
        # Sequential processing (original behavior)
        print(f"Processing {len(rows_to_process)} row(s) sequentially...\n")
        for idx, row_index in enumerate(rows_to_process, 1):
            try:
                process_review_revise_row(
                    row_index,
                    df,
                    course_name,
                    target_audience,
                    drive,
                    llm,
                    ws,
                    review_cols,
                    use_only_drive_and_hvac,
                )
                progress.update()
            except Exception as e:
                print(f"❌ Error processing row {row_index + 1}: {e}")
                progress.update()

    print("\n" + "=" * 80)
    print("Saving alignment review results...")
    print("=" * 80)
    save_to_sheet(ws, df)
    format_worksheet(ws)

    # print("\n" + "=" * 80)
    # print("[STEP 3] Starting REDUNDANCY review across topic groups...")
    # print("=" * 80)
    # topic_groups: Dict[str, List[int]] = {}
    # for index, row in df.iterrows():
    #     topic_name = _safe_str(row.get("Topic", ""))
    #     if not _safe_str(row.get("final_graphics_definition", "")):
    #         continue
    #     if not topic_name:
    #         continue
    #     topic_groups.setdefault(topic_name, []).append(index)

    # print(f"Found {len(topic_groups)} topic group(s) to review for redundancy")
    # for topic_name, row_indices in topic_groups.items():
    #     if not row_indices:
    #         continue
    #     print(f"\nReviewing redundancy for topic: {topic_name} ({len(row_indices)} slide(s))")
    #     redundancy_status = run_redundancy_loop_for_topic(
    #         topic_key=(topic_name, ""),  # Pass empty subtopic for backward compatibility
    #         row_indices=row_indices,
    #         df=df,
    #         course_name=course_name,
    #         target_audience=target_audience,
    #         drive=drive,
    #         llm=llm,
    #     )
    #     for row_index in row_indices:
    #         df.at[row_index, "graphics_review_v2_notes"] = (
    #             _safe_str(df.at[row_index, "graphics_review_v2_notes"]) + f"; redundancy={redundancy_status}"
    #         ).strip("; ")
    #     print(f"  Final redundancy status for {topic_name}: {redundancy_status}")

    print("\n" + "=" * 80)
    print("Saving final results...")
    print("=" * 80)
    save_to_sheet(ws, df)
    format_worksheet(ws)

    try:
        hide_columns_by_name(ws, review_cols, df)
    except Exception as e:
        print(f"⚠️ Could not hide review columns: {e}")
    print("\n" + "=" * 80)
    print("✓ Graphics Definition V2 review/revise complete.")
    print("=" * 80)


def delete_review_and_revise_graphics_definition_v2(sheet):
    """
    Remove review-related columns from the Slide Chunks worksheet.

    :param sheet: The gspread sheet object
    :return: None
    """
    
    worksheet_name = "Slide Chunks"
    ws, df = get_sheet_data_and_df(sheet, worksheet_name)
    cols = ["review_complete", "graphics_review_v2_notes", "revision_tracking"]
    cols = [col for col in cols if col in df.columns]
    if cols:
        df = df.drop(columns=cols)
        clear_worksheet(ws)
        save_to_sheet(ws, df)
        print(f"Deleted review columns from '{worksheet_name}' worksheet")
    else:
        print(f"Review columns not found in '{worksheet_name}' worksheet")
