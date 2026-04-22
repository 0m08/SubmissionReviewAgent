from langsmith import traceable
import streamlit as st
from services.sheets_service import get_sheet_data_and_df, save_to_sheet, format_worksheet, clear_worksheet, merge_and_save_columns
from services.smart_progress_bar import SmartProgressBar
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
from services.helper_functions import build_video_part
from modules.chain import Chain
from dotenv import load_dotenv
from google import genai
from google.genai import types
from typing import List
from agents.graphics_definition_v2.aggregation_agent.aggregation_agent import (
    parse_urls_from_video_pool,
    parse_urls_from_video_pool_other_channels,
    parse_segments_from_voiceover,
    parse_video_url_timestamps,
    convert_watch_url_to_embed_url,
    get_drive_instance,
    invoke_gemini_multimodal,
)

load_dotenv()

MIN_BATCH_SIZE = 3
MAX_BATCH_SIZE = 5
SAVE_INTERVAL_ROWS = 5
SCORE_COLUMN_NAME = "video_score"


# Video scoring prompt to use when we want the visual assignment to be flexible or 1 visual per sentence
video_scoring_prompt = """You are an expert educational video evaluator specializing in the field of HVAC.

Your task is to review a provided set of video candidates and assign a relevance score to each video based on how well it visually supports a single voiceover sentence from an educational e-learning slide. This scoring will be used to shortlist the strongest candidate videos for further filtering and final visual selection in downstream steps.

A video’s score should reflect how clearly, directly, and instructionally it helps a learner understand the meaning and intent of the voiceover sentence. A video may support the full sentence or only a part of it; however, videos that support more important or central parts of the sentence, or provide clearer instructional value, should receive higher scores.

A video should be considered relevant even if any visually meaningful portion of the video can support the voiceover sentence. The entire video does not need to be relevant, as only a short segment or frame from the video can be used later.

These are the inputs:

<course_information>
Course name: {course_name}
Topic name: {topic_name}
Subtopic name: {subtopic_name}
</course_information>

<voiceover_sentence_for_which_to_assign_score_to_videos>
{vo_text}
</voiceover_sentence_for_which_to_assign_score_to_videos>

<whole_slide_context>
Slide Title: {slide_title}
Slide Content: "{slide_chunk}"
</whole_slide_context>

These are the video candidates for the voiceover sentence:

<video_candidates>
{video_candidates}
</video_candidates>

Instructions:

1) Core Scoring Objective
- Review all provided video candidates.
- Assign a score from 0 to 10 to each video based on how well it visually supports the voiceover sentence.
- Higher scores should be given to videos that clearly and directly help a learner understand the sentence, either fully or by strongly supporting an important part of it.

2) Meaning-Based Evaluation
- Judge each video based on the meaning and instructional intent of the voiceover sentence.
- Use the slide content to understand the full context and resolve references, pronouns, or implied meaning if needed.
- Do not assign scores based on general topic relevance alone.

3) Visual Grounding 
- Base all scoring decisions on what is visually shown in the video, not on what is being spoken in the video.
- Evaluate whether any portion of the video contains clear, usable visual content that supports the voiceover sentence.
- Do not rely on video titles, URLs, or metadata alone.

4) Instructional Clarity and Usefulness
- Prioritize videos that:
  - clearly show the key component, object, or concept being described
  - visually demonstrate the process, condition, or outcome mentioned in the sentence
  - contain visually clear segment(s) that can be directly used as clip(s) or still frame(s)
  - would make the explanation easier to understand for a learner
- Penalize videos that:
  - contain unclear, or hard-to-identify relevant segment(s)
  - are vague, generic, or only loosely related
  - require interpretation beyond what is visually shown
  - do not add meaningful instructional value
  - are completely irrelevant

5) Relative Ranking Within the Batch
- Treat the provided videos as a comparative set.
- Use a ranking mindset: some videos should clearly score higher than others.
- Avoid assigning identical scores unless two videos are truly indistinguishable in relevance and usefulness.
- Ensure that the scoring meaningfully differentiates stronger candidates from weaker ones.

6) Scoring Guidance
- 9–10: Highly relevant, contains clear and strong visual segment(s) that directly support the core idea or a key part of the sentence
- 7–8: Clearly relevant and useful, but the visual support may be less direct
- 5–6: Partially relevant or contains limited useful visual segment(s)
- 3–4: Weakly related or unclear visual connection to the sentence
- 0–2: Not relevant or does not meaningfully support the sentence

OUTPUT FORMAT:

Always provide your output strictly in the following format:

<output>

<evaluation_breakdown>

<voiceover_sentence_understanding>
Briefly explain what the voiceover sentence is communicating, using slide context to resolve any references or implied meaning if needed.
</voiceover_sentence_understanding>

<video_candidate_scan>
Create a numbered list of all provided video candidates and briefly describe what you see in each of the videos. Focus only on the visual content, not what is being spoken in the videos.
</video_candidate_scan>

<scoring_rationale>
Explain how you plan to apply the scoring criteria across the video candidates. Provide a detailed rationale for your scoring decision for each of the video candidate.
</scoring_rationale>

</evaluation_breakdown>

<final_scores>
(List of all video candidates with their scores in this exact format. Include the exact URL as provided, including timestamps if present.)
1. [Exact Video URL with timestamps as given in the 'video candidate' input] | Score: X/10
2. [Exact Video URL with timestamps as given in the 'video candidate' input] | Score: X/10
...
</final_scores>

</output>

(Ensure that you follow this exact XML format and do not add any extra text or comments outside the <output>, <evaluation_breakdown> and <final_scores> tags)
"""


# Video scoring prompt to use when we want 1 visual for the whole slide
video_scoring_prompt_for_entire_slide = """You are an expert educational video evaluator specializing in the field of HVAC.

Your task is to review a provided set of video candidates and assign a relevance score to each video based on how well it visually represents the overall meaning and instructional intent of an educational e-learning slide. This scoring will be used to shortlist the strongest candidate videos for further filtering and final visual selection in downstream steps.

A video’s score should reflect how clearly, directly, and instructionally it helps a learner understand the overall meaning of the slide. Videos that clearly represent the central concept, key takeaway, or most important visual idea of the slide should receive higher scores.

A video should be considered relevant even if any visually meaningful portion of the video can support the slide content. The entire video does not need to be relevant, as only a short segment or frame from the video can be used later. However, videos that contain clearer and more instructionally useful visual segment(s) representing the central concept of the slide should receive higher scores than videos that support only minor or secondary details.

These are the inputs:

<course_information>
Course name: {course_name}
Topic name: {topic_name}
Subtopic name: {subtopic_name}
</course_information>

<slide_content>
Slide Title: {slide_title}
Slide Content: "{slide_chunk}"
</slide_content>

These are the video candidates for the slide:

<video_candidates>
{video_candidates}
</video_candidates>

Instructions:

1) Core Scoring Objective
- Review all provided video candidates.
- Assign a score from 0 to 10 to each video based on how well it visually represents the overall meaning of the slide.
- Higher scores should be given to videos that clearly and directly capture the central idea, key takeaway, or most important concept of the slide.

2) Meaning-Based Evaluation
- Judge each video based on the overall meaning and instructional intent of the entire slide.
- Identify the primary concept, key takeaway, or dominant idea that the learner should understand from the slide.
- Do not assign scores based on general topic relevance alone.

3) Visual Grounding
- Base all scoring decisions on what is visually shown in the video, not on what is being spoken in the video.
- Evaluate whether any portion of the video contains clear, usable visual content that supports the slide content.
- Do not rely on video titles, URLs, or metadata alone.

4) Instructional Clarity and Representativeness
- Prioritize videos that:
  - clearly represent the central concept or main idea of the slide
  - provide a strong and direct visual summary of the slide content
  - contain visually clear segment(s) that can be directly used as clip(s) or still frame(s)
  - would make the overall explanation easier to understand for a learner at a glance
- Penalize videos that:
  - represent only minor or secondary details of the slide
  - contain unclear, or hard-to-identify relevant segment(s)
  - are too narrow, fragmented, or incomplete relative to the overall slide meaning
  - are vague, generic, or only loosely related
  - require interpretation beyond what is visually shown
  - do not add meaningful instructional value
  - are completely irrelevant

5) Relative Ranking Within the Batch
- Treat the provided videos as a comparative set.
- Use a ranking mindset: some videos should clearly score higher than others.
- Avoid assigning identical scores unless two videos are truly indistinguishable in relevance and usefulness.
- Ensure that the scoring meaningfully differentiates stronger candidates from weaker ones.

6) Scoring Guidance
- 9–10: Highly relevant, contains clear and strong visual segment(s) that directly represent the central idea or key takeaway of the slide
- 7–8: Clearly relevant and useful, but may not fully capture the most important concept of the slide
- 5–6: Partially relevant or represents only a limited aspect of the slide
- 3–4: Weakly related or unclear visual connection to the main idea of the slide
- 0–2: Not relevant or does not meaningfully represent the slide

OUTPUT FORMAT:

Always provide your output strictly in the following format:

<output>

<evaluation_breakdown>

<slide_content_understanding>
Briefly explain what the slide content is communicating and identify the central idea or key takeaway of the slide.
</slide_content_understanding>

<video_candidate_scan>
Create a numbered list of all provided video candidates and briefly describe what you see in each of the videos. Focus only on the visual content, not what is being spoken in the videos.
</video_candidate_scan>

<scoring_rationale>
Explain how you plan to apply the scoring criteria across the video candidates. Provide a detailed rationale for your scoring decision for each of the video candidate.
</scoring_rationale>

</evaluation_breakdown>

<final_scores>
(List of all video candidates with their scores in this exact format. Include the exact URL as provided, including timestamps if present.)
1. [Exact Video URL with timestamps as given in the 'video candidate' input] | Score: X/10
2. [Exact Video URL with timestamps as given in the 'video candidate' input] | Score: X/10
...
</final_scores>

</output>

(Ensure that you follow this exact XML format and do not add any extra text or comments outside the <output>, <evaluation_breakdown> and <final_scores> tags)
"""


# Prompt to use when we want the visual assignment to be flexible or 1 visual per sentence
video_selection_from_all_videos_prompt = """You are an expert educational video curator specializing in the field of HVAC.

Your task is to review a provided set of video candidates and identify all videos that are relevant to a single voiceover sentence from an educational e-learning slide. A video is considered relevant if it visually relates to, supports, or illustrates any concept, object, component, or idea mentioned or implied in the voiceover sentence.

These are the inputs:

<course_information>
Course name: {course_name}
Topic name: {topic_name}
Subtopic name: {subtopic_name}
</course_information>

This is the voiceover sentence that you need to select videos for:
<voiceover_sentence>
{vo_text}
</voiceover_sentence>

<whole_slide_context>
Slide Title: {slide_title}
Slide Content: {slide_chunk}
</whole_slide_context>

These are the video candidates for the voiceover sentence:

<video_candidates>
{video_candidates}
</video_candidates>

Instructions:

1) Relevance-Based Selection
   - Review all provided video candidates.
   - Select every video that is relevant to any part of the voiceover sentence.
   - A video may be relevant even if it represents only part of the sentence or provides supportive or illustrative context.
   - These selected videos will later be used for choosing and planning the visuals for this voiceover sentence.

2) Meaning-Based Judgment
   - Judge relevance based on the meaning and instructional intent of the voiceover sentence.
   - Use the full slide context to resolve any references, pronouns, or implied meaning if needed.
   - Do not select videos based on general topic relevance alone.

3) Inclusion Without Redundancy
   - Do not try to minimize the number of selected videos.
   - Include all videos that meaningfully support understanding of any part of the voiceover sentence.
   - Do not exclude a video solely because another video supports a similar concept.
   - However, avoid selecting videos that are purely redundant and do not add any new visual perspective, detail, or informational value beyond another selected video.

4) Visual Grounding
   - Base all decisions on what is actually visible in the provided videos and not on what is being spoken in the videos, since these videos will be used for creating visuals as the slide is narrated.
   - Do not rely on video titles, URLs, or metadata alone.

5) When in Doubt, Prioritize Selection
   - If you are uncertain or doubtful about whether any of the provided video candidates is relevant to the voiceover sentence, err on the side of inclusion rather than exclusion.
   - When it is unclear whether any of the provided video candidates relates to the content, it is better to select it than to reject it.
   - Prioritize selection when relevance is ambiguous, as having more candidate videos provides more options for later visual planning stages.

OUTPUT FORMAT:

Provide your output strictly in the following format:

<output>

<evaluation_breakdown>

<voiceover_sentence_understanding>
Briefly explain what the voiceover sentence is communicating, using slide context to resolve any references or implied meaning if needed.
</voiceover_sentence_understanding>

<video_candidate_scan>
Create a numbered list of all provided video candidates and briefly describe what you see in each of the videos.
</video_candidate_scan>

<relevance_decision>
Explain in detail which of the videos are relevant to the voiceover sentence and why, taking all the instructions into consideration. Also explain why the other videos are not relevant. Focus on what is actually visible in the videos and not on what is being spoken in the videos. Ensure that you cover all the videos in your explanation. It is okay for this section to be very long, verbose and detailed to fit all your reasoning and intermediate thinking needed to arrive at the best final decision.
</relevance_decision>

</evaluation_breakdown>

<selected_videos>
(List of all selected videos in this exact format. Note - Some videos may have timestamps, so you need to include the exact URL as it is with those start and end timestamps for such videos)
1. [Exact Video URL] 
2. [Exact Video URL]
...
(Continue the list of selected videos in the same format until you have covered all the videos)
</selected_videos>

</output>

(Ensure that you follow this exact XML format and do not add any extra text or comments outside the <output>, <evaluation_breakdown> and <selected_videos> tags)
"""


# Prompt to use when we want only one visual for the entire slide 
video_selection_from_all_videos_prompt_for_entire_slide = """You are an expert educational video curator specializing in the field of HVAC.

Your task is to review a provided set of video candidates and identify all videos that are relevant to the given educational e-learning slide. A video is considered relevant if it visually relates to, supports, or illustrates any concept, object, component, or idea mentioned or implied in any part of the slide content.

These are the inputs:

<course_information>
Course name: {course_name}
Topic name: {topic_name}
Subtopic name: {subtopic_name}
</course_information>

<slide_content>
Slide Title: {slide_title}
Slide Content: {slide_chunk}
</slide_content>

These are the video candidates for the slide:

<video_candidates>
{video_candidates}
</video_candidates>

Instructions:

1) Relevance-Based Selection
   - Review all provided video candidates.
   - Select every video that is relevant to any part of the slide content
   - A video may be relevant even if it represents only part of the slide content or provides supportive or illustrative context to any part of the slide content.
   - These selected videos will later be used for choosing and planning the visuals for the voiceover of this entire slide.

2) Meaning-Based Judgment
   - Judge relevance based on the meaning and instructional intent of the whole slide.
   - Do not select videos based on general topic relevance alone.

3) Inclusion Without Redundancy
   - Do not try to minimize the number of selected videos.
   - Include all videos that meaningfully support understanding of any part of the slide content.
   - Do not exclude a video solely because another video supports a similar concept.
   - However, avoid selecting videos that are purely redundant and do not add any new visual perspective, detail, or informational value beyond another selected video.

4) Visual Grounding
   - Base all decisions on what is actually visible in the provided videos and not on what is being spoken in the videos, since these videos will be used for creating visuals as the slide is narrated.
   - Do not rely on video titles, URLs, or metadata alone.

5) When in Doubt, Prioritize Selection
   - If you are uncertain or doubtful about whether any of the provided video candidates is relevant to any part of the slide content, err on the side of inclusion rather than exclusion.
   - When it is unclear whether any of the provided video candidates relates to the content, it is better to select it than to reject it.
   - Prioritize selection when relevance is ambiguous, as having more candidate videos provides more options for later visual planning stages.

OUTPUT FORMAT:

Provide your output strictly in the following format:

<output>

<evaluation_breakdown>

<slide_content_understanding>
Briefly explain what the slide content is communicating
</slide_content_understanding>

<video_candidate_scan>
Create a numbered list of all provided video candidates and briefly describe what you see in each of the videos.
</video_candidate_scan>

<relevance_decision>
Explain in detail which of the videos are relevant to any part of the slide content and why, taking all the instructions into consideration. Also explain why the other videos are not relevant. Focus on what is actually visible in the videos and not on what is being spoken in the videos. Ensure that you cover all the videos in your explanation. It is okay for this section to be very long, verbose and detailed to fit all your reasoning and intermediate thinking needed to arrive at the best final decision.
</relevance_decision>

</evaluation_breakdown>

<selected_videos>
(List of all selected videos in this exact format. Note - Some videos may have timestamps, so you need to include the exact URL as it is with those start and end timestamps for such videos)
1. [Exact Video URL] 
2. [Exact Video URL]
...
(Continue the list of selected videos in the same format until you have covered all the videos)
</selected_videos>

</output>

(Ensure that you follow this exact XML format and do not add any extra text or comments outside the <output>, <evaluation_breakdown> and <selected_videos> tags)
"""


# Regeneration prompt to use when we want the visual assignment to be flexible or 1 visual per sentence, with feedback for regeneration
video_selection_from_all_videos_prompt_with_feedback = """You are an expert educational video curator specializing in the field of HVAC.

Your task is to review a provided set of video candidates and identify all videos that are relevant to a single voiceover sentence from an educational e-learning slide, taking into account the feedback describing what visual requirements need to be met. The feedback contains the details of the previous visuals that were assigned for the voiceover sentence, what was wrong with it and what is needed instead.
A video is considered relevant if it visually relates to, supports, or illustrates any concept, object, component, or idea mentioned or implied in the voiceover sentence or addresses the visual requirements described in the feedback.

These are the inputs:

<course_information>
Course name: {course_name}
Topic name: {topic_name}
Subtopic name: {subtopic_name}
</course_information>

This is the voiceover sentence that you need to select videos for:
<voiceover_sentence>
{vo_text}
</voiceover_sentence>

<whole_slide_context>
Slide Title: {slide_title}
Slide Content: {slide_chunk}
</whole_slide_context>

<feedback>
{feedback}
</feedback>

These are the video candidates:

<video_candidates>
{video_candidates}
</video_candidates>

Instructions:

1) Feedback-Driven Relevance Selection
   - Carefully analyze the feedback to understand what visual requirements need to be met.
   - The feedback contains the details of the previous visuals that were assigned for the voiceover sentence, what was wrong with it and what is needed instead.
   - Select videos that relate to any part of the voiceover sentence OR address any of the feedback requirements.

2) Relevance-Based Selection
   - Review all provided video candidates.
   - Select every video that is relevant to any part of the voiceover sentence OR addresses any of the feedback requirements.
   - A video may be relevant even if it represents only part of the sentence or provides supportive or illustrative context.
   - These selected videos will later be used for choosing and planning the visuals for this voiceover sentence.

3) Meaning-Based Judgment
   - Judge relevance based on the meaning and instructional intent of the voiceover sentence.
   - Use the full slide context to resolve any references, pronouns, or implied meaning if needed.
   - Do not select videos based on general topic relevance alone.

4) Inclusion Without Redundancy
   - Do not try to minimize the number of selected videos.
   - Include all videos that meaningfully support understanding of any part of the voiceover sentence or address any of the feedback requirements.
   - Do not exclude a video solely because another video supports a similar concept.
   - However, avoid selecting videos that are purely redundant and do not add any new visual perspective, detail, or informational value beyond another selected video.

5) Visual Grounding
   - Base all decisions on what is actually visible in the provided videos and not on what is being spoken in the videos, since these videos will be used for creating visuals as the slide is narrated.
   - Do not rely on video titles, URLs, or metadata alone.

6) When in Doubt, Prioritize Selection
   - If you are uncertain or doubtful about whether any of the provided video candidates is relevant to the voiceover sentence or addresses any of the feedback requirements, err on the side of inclusion rather than exclusion.
   - When it is unclear whether any of the provided video candidates relates to the content or addresses the feedback, it is better to select it than to reject it.
   - Prioritize selection when relevance is ambiguous, as having more candidate videos provides more options for later visual planning stages.

OUTPUT FORMAT:

Provide your output strictly in the following format:

<output>

<evaluation_breakdown>

<voiceover_sentence_understanding>
Briefly explain what the voiceover sentence is communicating, using slide context to resolve any references or implied meaning if needed.
</voiceover_sentence_understanding>

<feedback_interpretation>
Explain what the feedback indicates is wrong with the previous visual(s) and what specific visual requirements need to be met.
</feedback_interpretation>

<video_candidate_scan>
Create a numbered list of all provided video candidates and briefly describe what you see in each of the videos.
</video_candidate_scan>

<relevance_decision>
Explain in detail which of the videos are relevant to any part of the voiceover sentence or address any of the feedback requirements, and why. Also explain why the other videos are not relevant. Focus on what is actually visible in the videos and not on what is being spoken in the videos. Ensure that you cover all the videos in your explanation. It is okay for this section to be very long, verbose and detailed to fit all your reasoning and intermediate thinking needed to arrive at the best final decision.
</relevance_decision>

</evaluation_breakdown>

<selected_videos>
(List of all selected videos in this exact format. Note - Some videos may have timestamps, so you need to include the exact URL as it is with those start and end timestamps for such videos)
1. [Exact Video URL] 
2. [Exact Video URL]
...
(Continue the list of selected videos in the same format until you have covered all the videos)
</selected_videos>

</output>

(Ensure that you follow this exact XML format and do not add any extra text or comments outside the <output>, <evaluation_breakdown> and <selected_videos> tags)
"""


# Regeneration prompt to use when we want only one visual for the entire slide, with feedback for regeneration
video_selection_from_all_videos_prompt_for_entire_slide_with_feedback = """You are an expert educational video curator specializing in the field of HVAC.

Your task is to review a provided set of video candidates and identify all videos that are relevant to the given educational e-learning slide, taking into account the feedback describing what visual requirements need to be met. The feedback contains the details of the previous visual that was assigned for the slide, what was wrong with it and what is needed instead. A video is considered relevant if it visually relates to, supports, or illustrates any concept, object, component, or idea mentioned or implied in any part of the slide content or addresses the visual requirements described in the feedback.

These are the inputs:

<course_information>
Course name: {course_name}
Topic name: {topic_name}
Subtopic name: {subtopic_name}
</course_information>

<slide_content>
Slide Title: {slide_title}
Slide Content: {slide_chunk}
</slide_content>

<feedback>
{feedback}
</feedback>

These are the video candidates:

<video_candidates>
{video_candidates}
</video_candidates>

Instructions:

1) Feedback-Driven Relevance Selection
   - Carefully analyze the feedback to understand what visual requirements need to be met.
   - The feedback contains the details of the previous visual that was assigned for the slide, what was wrong with it and what is needed instead.
   - Select videos that relate to any part of the slide content OR address any of the feedback requirements.

2) Relevance-Based Selection
   - Review all provided video candidates.
   - Select every video that is relevant to any part of the slide content OR addresses any of the feedback requirements.
   - A video may be relevant even if it represents only part of the slide content or provides supportive or illustrative context to any part of the slide content.
   - These selected videos will later be used for choosing and planning the visuals for the voiceover of this entire slide.

3) Meaning-Based Judgment
   - Judge relevance based on the meaning and instructional intent of the slide content.
   - Do not select videos based on general topic relevance alone.

4) Inclusion Without Redundancy
   - Do not try to minimize the number of selected videos.
   - Include all videos that meaningfully support understanding of any part of the slide content OR address any of the feedback requirements.
   - Do not exclude a video solely because another video supports a similar concept.
   - However, avoid selecting videos that are purely redundant and do not add any new visual perspective, detail, or informational value beyond another selected video.

5) Visual Grounding
   - Base all decisions on what is actually visible in the provided videos and not on what is being spoken in the videos, since these videos will be used for creating visuals as the slide is narrated.
   - Do not rely on video titles, URLs, or metadata alone.

6) When in Doubt, Prioritize Selection
   - If you are uncertain or doubtful about whether any of the provided video candidates is relevant to any part of the slide content or addresses any of the feedback requirements, err on the side of inclusion rather than exclusion.
   - When it is unclear whether any of the provided video candidates relates to the content or addresses the feedback, it is better to select it than to reject it.
   - Prioritize selection when relevance is ambiguous, as having more candidate videos provides more options for later visual planning stages.

OUTPUT FORMAT:

Provide your output strictly in the following format:

<output>

<evaluation_breakdown>

<slide_content_understanding>
Briefly explain what the slide content is communicating.
</slide_content_understanding>

<feedback_interpretation>
Explain what the feedback indicates is wrong with the previous visual and what specific visual requirements need to be met.
</feedback_interpretation>

<video_candidate_scan>
Create a numbered list of all provided video candidates and briefly describe what you see in each of the videos.
</video_candidate_scan>

<relevance_decision>
Explain in detail which of the videos are relevant to any part of the slide content OR address any of the feedback requirements, and why. Also explain why the other videos are not relevant. Focus on what is actually visible in the videos and not on what is being spoken in the videos. Ensure that you cover all the videos in your explanation. It is okay for this section to be very long, verbose and detailed to fit all your reasoning and intermediate thinking needed to arrive at the best final decision.
</relevance_decision>

</evaluation_breakdown>

<selected_videos>
(List of all selected videos in this exact format. Note - Some videos may have timestamps, so you need to include the exact URL as it is with those start and end timestamps for such videos)
1. [Exact Video URL] 
2. [Exact Video URL]
...
(Continue the list of selected videos in the same format until you have covered all the videos)
</selected_videos>

</output>

(Ensure that you follow this exact XML format and do not add any extra text or comments outside the <output>, <evaluation_breakdown> and <selected_videos> tags)
"""


def parse_video_items_from_pool_other_channels(video_pool_other_channels_text, segment_num):
    """
    Parse video items (with title, duration, channel, URL) from video_pool_other_channels for a specific segment.
    
    Format: "Title: {title} | Duration: {duration} | Channel: {channel} | URL: {url}"
    
    :param video_pool_other_channels_text: The video_pool_other_channels column content
    :param segment_num: Segment number to extract videos for
    :return: List of dictionaries with 'title', 'duration', 'channel', 'url' keys or empty list if no videos found
    """
    if not video_pool_other_channels_text or video_pool_other_channels_text.strip() == "" or video_pool_other_channels_text == "nan":
        return []
    
    # Find the segment section
    segment_pattern = rf'---SEGMENT_{segment_num}---\s*\n(.*?)(?=\n---SEGMENT_|\Z)'
    match = re.search(segment_pattern, video_pool_other_channels_text, re.DOTALL)
    
    if not match:
        return []
    
    segment_content = match.group(1).strip()
    items = []
    
    # Parse each line: "Title: ... | Duration: ... | Channel: ... | URL: ..."
    for line in segment_content.split('\n'):
        line = line.strip()
        if not line:
            continue
        
        # Parse the formatted line
        if " | URL: " in line:
            parts = line.split(" | URL: ", 1)
            if len(parts) == 2:
                metadata_part = parts[0].strip()
                url = parts[1].strip()
                
                # Parse metadata
                title = "Untitled"
                duration = ""
                channel = ""
                
                if "Title: " in metadata_part:
                    title_match = re.search(r'Title:\s*(.+?)(?:\s*\||$)', metadata_part)
                    if title_match:
                        title = title_match.group(1).strip()
                
                if "Duration: " in metadata_part:
                    duration_match = re.search(r'Duration:\s*(.+?)(?:\s*\||$)', metadata_part)
                    if duration_match:
                        duration = duration_match.group(1).strip()
                
                if "Channel: " in metadata_part:
                    channel_match = re.search(r'Channel:\s*(.+?)(?:\s*\||$)', metadata_part)
                    if channel_match:
                        channel = channel_match.group(1).strip()
                
                if url and url.startswith('http'):
                    items.append({
                        "title": title,
                        "duration": duration,
                        "channel": channel,
                        "url": url
                    })
        elif line.startswith('http'):
            # Fallback: if line is just a URL
            items.append({
                "title": "Untitled",
                "duration": "",
                "channel": "",
                "url": line
            })
    
    return items


def build_dynamic_batches(items, min_size=MIN_BATCH_SIZE, max_size=MAX_BATCH_SIZE):
    """
    Build balanced dynamic batches with size in [min_size, max_size] when possible.

    :param items: Input candidate list
    :param min_size: Minimum batch size target
    :param max_size: Maximum batch size target
    :return: List of candidate batches
    """
    n = len(items)
    if n <= 0:
        return []
    if n <= max_size:
        return [items]
    k_min = (n + max_size - 1) // max_size
    k_max = n // min_size
    if k_min > k_max:
        return [items]
    k = k_min
    base = n // k
    rem = n % k
    out, start = [], 0
    for i in range(k):
        size = base + (1 if i < rem else 0)
        out.append(items[start:start + size])
        start += size
    return out


def _extract_score_value(score_text):
    """
    Extract numeric score value from text containing X/10.

    :param score_text: Raw score text
    :return: Float score in [0, 10] or None
    """
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*/\s*10", score_text or "")
    if not match:
        return None
    try:
        return max(0.0, min(10.0, float(match.group(1))))
    except Exception:
        return None


def parse_scored_videos_text(final_scores_text):
    """
    Parse <final_scores> into (url, score) tuples.

    :param final_scores_text: Raw text from <final_scores> tag
    :return: List of tuples (video_url, score)
    """
    out = []
    for line in (final_scores_text or "").splitlines():
        raw = re.sub(r"^\d+\.\s*", "", line.strip())
        if not raw:
            continue
        url_match = re.search(r"https?://[^\s|]+", raw)
        score_val = _extract_score_value(raw)
        if url_match and score_val is not None:
            out.append((url_match.group(0).strip(), score_val))
    return out


def format_video_score_segment(segment_num, set_scored_items):
    """
    Format one segment's scored sets for video_score column.

    :param segment_num: Segment number
    :param set_scored_items: List of set outputs; each set is [(url, score), ...]
    :return: Formatted segment block text
    """
    lines = [f"---SEGMENT_{segment_num}---"]
    for set_idx, scored_items in enumerate(set_scored_items, 1):
        lines.append(f"Set {set_idx}:")
        for url, score in scored_items:
            try:
                score_num = float(score)
                score_text = str(int(score_num)) if score_num.is_integer() else f"{score_num}".rstrip("0").rstrip(".")
            except Exception:
                score_text = str(score)
            lines.append(f"Video Url: {url} | Score: {score_text}/10")
        lines.append("")
    return "\n".join(lines).strip()


def _shortlist_urls_from_segment_score_block(segment_block, top_n=2, min_score=4.0):
    """
    From one segment score block, shortlist URLs per set with score gating.

    :param segment_block: Segment block text with Set sections
    :param top_n: Base top-N to keep before tie expansion
    :param min_score: Minimum score cutoff for candidates during tie expansion
    :return: Deduplicated shortlisted URLs
    """
    
    shortlisted = []
    set_blocks = re.findall(r"Set\s+\d+\s*:\s*(.*?)(?=\nSet\s+\d+\s*:|\Z)", segment_block or "", flags=re.IGNORECASE | re.DOTALL)
    for set_block in set_blocks:
        scored = []
        for line in set_block.splitlines():
            raw = line.strip()
            if not raw:
                continue
            url_match = re.search(r"https?://[^\s|]+", raw)
            score_val = _extract_score_value(raw)
            if url_match and score_val is not None:
                scored.append((url_match.group(0).strip(), score_val))
        if not scored:
            continue
        scored.sort(key=lambda x: -x[1])
        if len(scored) <= top_n:
            chosen = scored
        else:
            cutoff = scored[top_n - 1][1]
            chosen = [x for x in scored if x[1] >= cutoff]
        # Drop low-score tie-expanded candidates; if none survive, force one best candidate.
        chosen = [x for x in chosen if x[1] >= min_score]
        if not chosen:
            chosen = [scored[0]]
        for url, _ in chosen:
            if url not in shortlisted:
                shortlisted.append(url)
    return shortlisted


def shortlist_video_urls_from_score_text(video_score_text, segment_num, top_n=2, min_score=4.0):
    """
    Parse video_score text and return shortlisted URLs for a segment.

    :param video_score_text: Full video_score column text
    :param segment_num: Segment number
    :param top_n: Base top-N to keep before tie expansion
    :param min_score: Minimum score cutoff for candidates during tie expansion
    :return: Deduplicated shortlisted URLs for the segment
    """
    
    if not video_score_text:
        return []
    segment_pattern = rf"---SEGMENT_{segment_num}---\s*\n(.*?)(?=\n---SEGMENT_|\Z)"
    match = re.search(segment_pattern, video_score_text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return []
    return _shortlist_urls_from_segment_score_block(match.group(1), top_n=top_n, min_score=min_score)


def score_videos_batch(vo_text, slide_title, slide_chunk, video_urls_pool, video_items_other_channels, course_name, topic_name, subtopic_name, drive, llm="gemini_3_flash_thinking", entire_slide=False):
    """
    Score one batch of video candidates and return parsed (url, score) tuples.

    :param vo_text: Voiceover sentence (segment mode)
    :param slide_title: Slide title
    :param slide_chunk: Slide content
    :param video_urls_pool: Candidate URLs from video_pool
    :param video_items_other_channels: Candidate metadata items from other channels
    :param course_name: Course name
    :param topic_name: Topic name
    :param subtopic_name: Subtopic name
    :param drive: Drive instance (reserved for parity/signature consistency)
    :param llm: Model name
    :param entire_slide: If True, use whole-slide scoring prompt
    :return: List of tuples (video_url, score)
    """
    
    total_videos = len(video_urls_pool) + len(video_items_other_channels)
    if total_videos == 0:
        return []
    video_candidates_list = []
    video_candidates_list.extend(video_urls_pool)
    for video_item in video_items_other_channels:
        url = video_item.get("url", "")
        if url:
            video_candidates_list.append(url)
    video_candidates_text = "\n".join([f"{i+1}. {url}" for i, url in enumerate(video_candidates_list)])
    parts: List[types.Part] = []

    if entire_slide:
        prompt_text = video_scoring_prompt_for_entire_slide.format(
            course_name=course_name,
            topic_name=topic_name,
            subtopic_name=subtopic_name,
            slide_title=slide_title,
            slide_chunk=slide_chunk,
            video_candidates=video_candidates_text,
        )
        # print(f"\n{'='*80}")
        # print(f"📝 FORMATTED VIDEO SCORING PROMPT (Entire Slide):")
        # print(f"{'='*80}")
        # print(prompt_text)
        # print(f"{'='*80}\n")
    else:
        prompt_text = video_scoring_prompt.format(
            course_name=course_name,
            topic_name=topic_name,
            subtopic_name=subtopic_name,
            vo_text=vo_text,
            slide_title=slide_title,
            slide_chunk=slide_chunk,
            video_candidates=video_candidates_text,
        )
        # print(f"\n{'='*80}")
        # print(f"📝 FORMATTED VIDEO SCORING PROMPT (Segment):")
        # print(f"{'='*80}")
        # print(prompt_text)
        # print(f"{'='*80}\n")

    idx = 1
    for video_url in video_urls_pool:
        clip_url, start_seconds, end_seconds = parse_video_url_timestamps(video_url)
        if clip_url:
            parts.append(
                types.Part(
                    text=f"\n--- Video {idx} of {total_videos} ---\nURL: {video_url}\n"
                )
            )
            parts.append(build_video_part(clip_url, start_seconds, end_seconds))
        idx += 1
    for video_item in video_items_other_channels:
        video_url = video_item.get("url", "")
        embed_url = convert_watch_url_to_embed_url(video_url)
        if embed_url:
            parts.append(
                types.Part(
                    text=f"\n--- Video {idx} of {total_videos} ---\nURL: {video_url}\n"
                )
            )
            parts.append(build_video_part(embed_url, start_seconds=None, end_seconds=None))
        idx += 1

    parts.append(types.Part(text=prompt_text))
    raw_text = invoke_gemini_multimodal(parts, llm=llm, temperature=0.1)
    
    print(f"\n{'='*80}")
    print(f"📤 VIDEO SCORING RESPONSE ({'Entire Slide' if entire_slide else 'Segment'}):")
    print(f"{'='*80}")
    print(raw_text)
    print(f"{'='*80}\n")
    parser_chain = Chain(llm=llm, tags=["final_scores"])
    parsed = parser_chain.extract_text_in_tags(raw_text)
    final_scores_text = parsed.get("final_scores", "")
    return parse_scored_videos_text(final_scores_text)




@traceable(
    metadata={
        "agent_name": "graphics_definition_v2",
        "step_name": "Video Selection from All Videos",
        "function_name": "select_videos_from_all_for_segment",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous")
    }
)
def select_videos_from_all_for_segment(vo_text, slide_title, slide_chunk, video_urls_pool, video_items_other_channels, course_name, topic_name, subtopic_name, drive, llm="gemini_3_flash_thinking", feedback=None):
    """
    Select relevant videos from all available videos for a single segment using vision model.
    
    :param vo_text: Voiceover text for the segment
    :param slide_title: Slide title
    :param slide_chunk: Full slide content
    :param video_urls_pool: List of video URLs from video_pool (with timestamps)
    :param video_items_other_channels: List of video items from video_pool_other_channels (with metadata)
    :param course_name: Course name
    :param topic_name: Topic name
    :param subtopic_name: Subtopic name
    :param drive: Google Drive instance
    :param llm: Language model to use
    :param feedback: Optional feedback for regeneration (if provided, uses feedback prompt)
    :return: Selected videos text or empty string if no videos available
    """
    print(f"\n{'─'*45}")
    print(f" 🎯 Selecting videos for segment")
    print(f"{'─'*45}")
    print(f"🤖 Using LLM model: {llm}")
    print(f"📝 VO text: \"{vo_text}\"")
    print(f"🎥 Available videos (pool): {len(video_urls_pool)}")
    print(f"🎬 Available videos (other channels): {len(video_items_other_channels)}")
    if feedback:
        print(f"📋 Using feedback-based selection")
    
    total_videos = len(video_urls_pool) + len(video_items_other_channels)
    
    if total_videos == 0:
        print(f"⚠️  No videos available, returning empty selection")
        return ""
    
    # Build video candidates list (all URLs)
    video_candidates_list = []
    # Add URLs from video_pool
    for video_url in video_urls_pool:
        video_candidates_list.append(video_url)
    # Add URLs from video_pool_other_channels
    for video_item in video_items_other_channels:
        video_url = video_item.get("url", "")
        if video_url:
            video_candidates_list.append(video_url)
    
    # Format video candidates as numbered list
    video_candidates_text = "\n".join([f"{i+1}. {url}" for i, url in enumerate(video_candidates_list)])
    
    # Build multimodal content with videos
    parts: List[types.Part] = []
    
    # Select prompt based on whether feedback is provided
    if feedback and feedback.strip():
        # Use feedback prompt for regeneration
        prompt_text = video_selection_from_all_videos_prompt_with_feedback.format(
            course_name=course_name,
            topic_name=topic_name,
            subtopic_name=subtopic_name,
            vo_text=vo_text,
            slide_title=slide_title,
            slide_chunk=slide_chunk,
            feedback=feedback.strip(),
            video_candidates=video_candidates_text,
            num_videos=total_videos
        )
        # Print the formatted feedback prompt
        # print(f"\n{'='*80}")
        # print(f"📝 FORMATTED VIDEO SELECTION PROMPT WITH FEEDBACK (Segment):")
        # print(f"{'='*80}")
        # print(prompt_text)
        # print(f"{'='*80}\n")
    else:
        # Use standard prompt for initial selection
        prompt_text = video_selection_from_all_videos_prompt.format(
            course_name=course_name,
            topic_name=topic_name,
            subtopic_name=subtopic_name,
            vo_text=vo_text,
            slide_title=slide_title,
            slide_chunk=slide_chunk,
            video_candidates=video_candidates_text,
            num_videos=total_videos
        )
        # Print the formatted standard prompt
        # print(f"\n{'='*80}")
        # print(f"📝 FORMATTED VIDEO SELECTION PROMPT (Segment):")
        # print(f"{'='*80}")
        # print(prompt_text)
        # print(f"{'='*80}\n")
    
    # Add videos from video_pool (with timestamps) - pass only the segment
    video_index = 1
    
    for video_url in video_urls_pool:
        print(f"📥 Processing video {video_index}/{total_videos}: {video_url[:50]}...")
        
        # Parse timestamps from URL
        clip_url, start_seconds, end_seconds = parse_video_url_timestamps(video_url)
        
        if clip_url:
            # Add label text with URL and timestamps
            label_text = f"\n--- Video {video_index} of {total_videos} ---\nURL: {video_url}"
            if start_seconds is not None and end_seconds is not None:
                label_text += f" | Timestamps: {start_seconds}s - {end_seconds}s"
            label_text += "\n"
            parts.append(types.Part(text=label_text))
            
            # Add video part with timestamps (only this segment)
            video_part = build_video_part(clip_url, start_seconds, end_seconds)
            parts.append(video_part)
            print(f"✅ Added video {video_index} (with timestamps): start={start_seconds}s, end={end_seconds}s")
        else:
            parts.append(
                types.Part(
                    text=f"\n--- Video {video_index} of {total_videos} ---\nURL: {video_url}\n[Failed to parse video URL]\n"
                )
            )
            print(f"⚠️ Failed to parse video URL: {video_url}")
        
        video_index += 1
    
    # Add videos from video_pool_other_channels (full videos)
    for video_item in video_items_other_channels:
        video_url = video_item.get("url", "")
        video_title = video_item.get("title", "Untitled")
        
        print(f"📥 Processing video {video_index}/{total_videos}: {video_title[:50]}...")
        
        # Convert watch URL to embed URL
        embed_url = convert_watch_url_to_embed_url(video_url)
        
        if embed_url:
            # Add label text with metadata
            label_text = f"\n--- Video {video_index} of {total_videos} ---\nTitle: {video_title}"
            if video_item.get("duration"):
                label_text += f" | Duration: {video_item.get('duration')}"
            if video_item.get("channel"):
                label_text += f" | Channel: {video_item.get('channel')}"
            label_text += f"\nURL: {video_url}\n"
            parts.append(types.Part(text=label_text))
            
            # Add video part without timestamps (full video)
            video_part = build_video_part(embed_url, start_seconds=None, end_seconds=None)
            parts.append(video_part)
            print(f"✅ Added video {video_index} (full video, no timestamps): {embed_url}")
        else:
            parts.append(
                types.Part(
                    text=f"\n--- Video {video_index} of {total_videos} ---\nTitle: {video_title}\nURL: {video_url}\n[Failed to convert video URL]\n"
                )
            )
            print(f"⚠️ Failed to convert video URL: {video_url}")
        
        video_index += 1
    
    # Add prompt text at the end
    parts.append(types.Part(text=prompt_text))
       
    print(f"🤖 Calling vision model with {len(video_urls_pool) + len(video_items_other_channels)} videos...")
    raw_text = invoke_gemini_multimodal(parts, llm=llm, temperature=0.1)
    
    # Print segment and response for debugging
    print(f"\nSegment: {vo_text}\n")
    print(f"📤 Video Selection Response from LLM:\n")
    print(raw_text)
    print(f"\n{'='*100}\n")
    
    # Parse the response
    parser_chain = Chain(llm=llm, tags=["selected_videos"])
    parsed = parser_chain.extract_text_in_tags(raw_text)
    selected_videos_text = parsed.get("selected_videos", "")
    
    if selected_videos_text:
        print(f"✅ Extracted selected videos ({len(selected_videos_text)} chars)")
    else:
        print(f"⚠️  Failed to extract selected videos from response")
    
    print(f"{'─'*45}\n")
    
    return selected_videos_text


@traceable(
    metadata={
        "agent_name": "graphics_definition_v2",
        "step_name": "Video Selection from All Videos",
        "function_name": "select_videos_from_all_for_entire_slide",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous")
    }
)
def select_videos_from_all_for_entire_slide(slide_title, slide_chunk, video_urls_pool, video_items_other_channels, course_name, topic_name, subtopic_name, drive, llm="gemini_3_flash_thinking", feedback=None):
    """
    Select relevant videos from all available videos for entire slide using vision model.
    
    :param slide_title: Slide title
    :param slide_chunk: Full slide content
    :param video_urls_pool: List of video URLs from video_pool (with timestamps)
    :param video_items_other_channels: List of video items from video_pool_other_channels (with metadata)
    :param course_name: Course name
    :param topic_name: Topic name
    :param subtopic_name: Subtopic name
    :param drive: Google Drive instance
    :param llm: Language model to use
    :param feedback: Optional feedback for regeneration (if provided, uses feedback prompt)
    :return: Selected videos text or empty string if no videos available
    """
    print(f"\n{'─'*45}")
    print(f" 🎯 Selecting videos for entire slide")
    print(f"{'─'*45}")
    print(f"🤖 Using LLM model: {llm}")
    print(f"📝 Slide Title: \"{slide_title}\"")
    print(f"🎥 Available videos (pool): {len(video_urls_pool)}")
    print(f"🎬 Available videos (other channels): {len(video_items_other_channels)}")
    if feedback:
        print(f"📋 Using feedback-based selection")
    
    total_videos = len(video_urls_pool) + len(video_items_other_channels)
    
    if total_videos == 0:
        print(f"⚠️  No videos available, returning empty selection")
        return ""
    
    # Build video candidates list (all URLs)
    video_candidates_list = []
    # Add URLs from video_pool
    for video_url in video_urls_pool:
        video_candidates_list.append(video_url)
    # Add URLs from video_pool_other_channels
    for video_item in video_items_other_channels:
        video_url = video_item.get("url", "")
        if video_url:
            video_candidates_list.append(video_url)
    
    # Format video candidates as numbered list
    video_candidates_text = "\n".join([f"{i+1}. {url}" for i, url in enumerate(video_candidates_list)])
    
    # Build multimodal content with videos
    parts: List[types.Part] = []
    
    # Select prompt based on whether feedback is provided
    if feedback and feedback.strip():
        # Use feedback prompt for regeneration
        prompt_text = video_selection_from_all_videos_prompt_for_entire_slide_with_feedback.format(
            course_name=course_name,
            topic_name=topic_name,
            subtopic_name=subtopic_name,
            slide_title=slide_title,
            slide_chunk=slide_chunk,
            feedback=feedback.strip(),
            video_candidates=video_candidates_text,
            num_videos=total_videos
        )
        # Print the formatted feedback prompt
        # print(f"\n{'='*80}")
        # print(f"📝 FORMATTED VIDEO SELECTION PROMPT WITH FEEDBACK (Entire Slide):")
        # print(f"{'='*80}")
        # print(prompt_text)
        # print(f"{'='*80}\n")
    else:
        # Use standard prompt for initial selection
        prompt_text = video_selection_from_all_videos_prompt_for_entire_slide.format(
            course_name=course_name,
            topic_name=topic_name,
            subtopic_name=subtopic_name,
            slide_title=slide_title,
            slide_chunk=slide_chunk,
            video_candidates=video_candidates_text,
            num_videos=total_videos
        )
        # Print the formatted standard prompt
        # print(f"\n{'='*80}")
        # print(f"📝 FORMATTED VIDEO SELECTION PROMPT (Entire Slide):")
        # print(f"{'='*80}")
        # print(prompt_text)
        # print(f"{'='*80}\n")
    
    # Add videos from video_pool (with timestamps) - pass only the segment
    video_index = 1
    
    for video_url in video_urls_pool:
        print(f"📥 Processing video {video_index}/{total_videos}: {video_url[:50]}...")
        
        # Parse timestamps from URL
        clip_url, start_seconds, end_seconds = parse_video_url_timestamps(video_url)
        
        if clip_url:
            # Add label text with URL and timestamps
            label_text = f"\n--- Video {video_index} of {total_videos} ---\nURL: {video_url}"
            if start_seconds is not None and end_seconds is not None:
                label_text += f" | Timestamps: {start_seconds}s - {end_seconds}s"
            label_text += "\n"
            parts.append(types.Part(text=label_text))
            
            # Add video part with timestamps (only this segment)
            video_part = build_video_part(clip_url, start_seconds, end_seconds)
            parts.append(video_part)
            print(f"✅ Added video {video_index} (with timestamps): start={start_seconds}s, end={end_seconds}s")
        else:
            parts.append(
                types.Part(
                    text=f"\n--- Video {video_index} of {total_videos} ---\nURL: {video_url}\n[Failed to parse video URL]\n"
                )
            )
            print(f"⚠️ Failed to parse video URL: {video_url}")
        
        video_index += 1
    
    # Add videos from video_pool_other_channels (full videos)
    for video_item in video_items_other_channels:
        video_url = video_item.get("url", "")
        video_title = video_item.get("title", "Untitled")
        
        print(f"📥 Processing video {video_index}/{total_videos}: {video_title[:50]}...")
        
        # Convert watch URL to embed URL
        embed_url = convert_watch_url_to_embed_url(video_url)
        
        if embed_url:
            # Add label text with metadata
            label_text = f"\n--- Video {video_index} of {total_videos} ---\nTitle: {video_title}"
            if video_item.get("duration"):
                label_text += f" | Duration: {video_item.get('duration')}"
            if video_item.get("channel"):
                label_text += f" | Channel: {video_item.get('channel')}"
            label_text += f"\nURL: {video_url}\n"
            parts.append(types.Part(text=label_text))
            
            # Add video part without timestamps (full video)
            video_part = build_video_part(embed_url, start_seconds=None, end_seconds=None)
            parts.append(video_part)
            print(f"✅ Added video {video_index} (full video, no timestamps): {embed_url}")
        else:
            parts.append(
                types.Part(
                    text=f"\n--- Video {video_index} of {total_videos} ---\nTitle: {video_title}\nURL: {video_url}\n[Failed to convert video URL]\n"
                )
            )
            print(f"⚠️ Failed to convert video URL: {video_url}")
        
        video_index += 1
    
    # Add prompt text at the end
    parts.append(types.Part(text=prompt_text))
       
    print(f"🤖 Calling vision model with {len(video_urls_pool) + len(video_items_other_channels)} videos...")
    raw_text = invoke_gemini_multimodal(parts, llm=llm, temperature=0.1)
    
    # Print slide and response for debugging
    print(f"\nSlide: {slide_title}\n")
    print(f"📤 Video Selection Response from LLM:\n")
    print(raw_text)
    print(f"\n{'='*100}\n")
    
    # Parse the response
    parser_chain = Chain(llm=llm, tags=["selected_videos"])
    parsed = parser_chain.extract_text_in_tags(raw_text)
    selected_videos_text = parsed.get("selected_videos", "")
    
    if selected_videos_text:
        print(f"✅ Extracted selected videos ({len(selected_videos_text)} chars)")
    else:
        print(f"⚠️  Failed to extract selected videos from response")
    
    print(f"{'─'*45}\n")
    
    return selected_videos_text


def format_selected_videos_for_segment(selected_videos_text, video_urls_pool, video_items_other_channels):
    """
    Parse and format selected videos for output to video_pool_filtered column.
    
    :param selected_videos_text: Text from <selected_videos> tag
    :param video_urls_pool: Original list of video URLs from video_pool
    :param video_items_other_channels: Original list of video items from video_pool_other_channels
    :return: List of formatted video lines matching original format
    """
    if not selected_videos_text or selected_videos_text.strip() == "":
        return []
    
    # Extract URLs from selected_videos_text
    selected_urls = []
    for line in selected_videos_text.split('\n'):
        line = line.strip()
        if not line:
            continue
        
        # Remove leading number and dot if present (format: "1. URL")
        if re.match(r'^\d+\.\s*', line):
            line = re.sub(r'^\d+\.\s*', '', line)
        
        # Extract URL
        if line.startswith('http'):
            selected_urls.append(line)
        elif "http" in line:
            # Try to extract URL from line
            url_match = re.search(r'https?://[^\s]+', line)
            if url_match:
                selected_urls.append(url_match.group(0))
    
    # Match selected URLs back to original format
    formatted_lines = []
    
    # Check video_pool URLs (simple URL format)
    for url in video_urls_pool:
        if url in selected_urls:
            formatted_lines.append(url)
    
    # Check video_pool_other_channels (formatted with metadata)
    for video_item in video_items_other_channels:
        video_url = video_item.get("url", "")
        if video_url in selected_urls:
            title = video_item.get("title", "Untitled")
            duration = video_item.get("duration", "")
            channel = video_item.get("channel", "")
            formatted_lines.append(f"Title: {title} | Duration: {duration} | Channel: {channel} | URL: {video_url}")
    
    return formatted_lines


@traceable(
    metadata={
        "agent_name": "graphics_definition_v2",
        "step_name": "Video Selection from All Videos",
        "function_name": "process_video_selection_segment",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous")
    }
)
def process_video_selection_segment(segment_idx, vo_text, slide_title, slide_chunk, video_pool, video_pool_other_channels, course_name, topic_name, subtopic_name, drive, llm="gemini_3_flash_thinking", shortlisted_urls=None):
    """
    Process a single segment: select relevant videos from all available videos.
    
    :param segment_idx: Segment index (1-based)
    :param vo_text: Voiceover text for the segment
    :param slide_title: Slide title
    :param slide_chunk: Full slide content
    :param video_pool: Video pool text (with timestamps)
    :param video_pool_other_channels: Video pool other channels text (with metadata)
    :param course_name: Course name
    :param topic_name: Topic name
    :param subtopic_name: Subtopic name
    :param drive: Google Drive instance
    :param llm: Language model to use
    :return: Tuple of (segment_idx, formatted_segment_text) or (segment_idx, None) if no selection generated
    """
    print(f"\n📦 Processing SEGMENT_{segment_idx}")
    
    # Get video URLs for this segment from both pools
    video_urls_pool = parse_urls_from_video_pool(video_pool, segment_idx)
    video_items_other_channels = parse_video_items_from_pool_other_channels(video_pool_other_channels, segment_idx)

    if shortlisted_urls:
        shortlist = set(shortlisted_urls)
        video_urls_pool = [u for u in video_urls_pool if u in shortlist]
        video_items_other_channels = [v for v in video_items_other_channels if v.get("url", "") in shortlist]
    
    print(f"🔗 Found {len(video_urls_pool)} videos from video_pool, {len(video_items_other_channels)} videos from video_pool_other_channels")
    print(f"📎 Total unique videos: {len(video_urls_pool) + len(video_items_other_channels)}")
    
    if not video_urls_pool and not video_items_other_channels:
        print(f"⚠️ No videos available for segment {segment_idx}, skipping")
        return segment_idx, None
    
    # Select videos for this segment
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
        llm=llm
    )
    
    if selected_videos_text:
        # Format the selected videos
        formatted_videos = format_selected_videos_for_segment(selected_videos_text, video_urls_pool, video_items_other_channels)
        if formatted_videos:
            # Format segment output: "---SEGMENT_N---\nURL\n..." or "---SEGMENT_N---\nTitle: ... | URL: ...\n..."
            segment_output = [f"---SEGMENT_{segment_idx}---"] + formatted_videos
            return segment_idx, '\n'.join(segment_output)
    
    return segment_idx, None


@traceable(
    metadata={
        "agent_name": "graphics_definition_v2",
        "step_name": "Video Selection from All Videos",
        "function_name": "process_video_selection_row",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous")
    }
)
def process_video_selection_row(index, row, course_name, drive, llm="gemini_3_flash_thinking"):
    """
    Process a single row: select videos for all segments and combine into video_pool_filtered output.
    
    :param index: Row index
    :param row: Pandas Series with row data
    :param course_name: Course name
    :param drive: Google Drive instance
    :param llm: Language model to use
    :return: Tuple of (index, video_pool_filtered_text) or (index, empty string) if no segments found
    """
    try:
        voiceover_segments = str(row.get("voiceover_segment", "")).strip()
        video_pool = str(row.get("video_pool", "")).strip()
        video_pool_other_channels = str(row.get("video_pool_other_channels", "")).strip()
        video_score_text = str(row.get(SCORE_COLUMN_NAME, "")).strip()
        slide_chunk = str(row.get("Slide Chunk", "")).strip()
        slide_title = str(row.get("Slide Chunk Title", "")).strip()
        topic_name = str(row.get("Topic", "")).strip()
        subtopic_name = str(row.get("Subtopic", "")).strip()
        
        # Get Visual Assignment Strategy
        visual_assignment_strategy = str(row.get("Visual Assignment Strategy", "Flexible, let the agent decide")).strip()
        if not visual_assignment_strategy or visual_assignment_strategy == "nan":
            visual_assignment_strategy = "Flexible, let the agent decide"
        
        # Skip if required fields are empty
        if not slide_chunk or slide_chunk == "nan":
            return index, ""
        
        # Handle "1 Visual for the whole Slide" case
        if visual_assignment_strategy == "1 Visual for the whole Slide":
            print(f"\n{'═'*50}")
            print(f"📋 Processing row {index}: Entire slide (1 visual)")
            print(f"{'═'*50}")
            
            # Get video URLs from SEGMENT_1 for both pools
            video_urls_pool = parse_urls_from_video_pool(video_pool, 1)
            video_items_other_channels = parse_video_items_from_pool_other_channels(video_pool_other_channels, 1)
            
            print(f"🔗 Found {len(video_urls_pool)} videos from video_pool, {len(video_items_other_channels)} videos from video_pool_other_channels")
            print(f"📎 Total unique videos: {len(video_urls_pool) + len(video_items_other_channels)}")

            shortlisted_urls = shortlist_video_urls_from_score_text(video_score_text, 1, top_n=2)
            if shortlisted_urls:
                shortlist = set(shortlisted_urls)
                video_urls_pool = [u for u in video_urls_pool if u in shortlist]
                video_items_other_channels = [v for v in video_items_other_channels if v.get("url", "") in shortlist]
            
            if not video_urls_pool and not video_items_other_channels:
                print(f"⚠️ No videos available for entire slide, skipping")
                return index, ""
            
            # Select videos for entire slide
            selected_videos_text = select_videos_from_all_for_entire_slide(
                slide_title=slide_title,
                slide_chunk=slide_chunk,
                video_urls_pool=video_urls_pool,
                video_items_other_channels=video_items_other_channels,
                course_name=course_name,
                topic_name=topic_name,
                subtopic_name=subtopic_name,
                drive=drive,
                llm=llm
            )
            
            if selected_videos_text:
                # Format the selected videos
                formatted_videos = format_selected_videos_for_segment(selected_videos_text, video_urls_pool, video_items_other_channels)
                if formatted_videos:
                    # Format output as SEGMENT_1: "---SEGMENT_1---\nURL\n..." or "---SEGMENT_1---\nTitle: ... | URL: ...\n..."
                    segment_output = [f"---SEGMENT_1---"] + formatted_videos
                    video_pool_filtered_text = '\n'.join(segment_output)
                    print(f"✅ Generated video selection for entire slide")
                    return index, video_pool_filtered_text
            
            return index, ""
        
        # Handle "1 Visual per Sentence" and "Flexible" cases - process segments individually
        if not voiceover_segments or voiceover_segments == "nan":
            return index, ""
        
        # Parse segments
        segments = parse_segments_from_voiceover(voiceover_segments)
        
        if not segments:
            return index, ""
        
        print(f"\n{'═'*50}")
        print(f"📋 Processing row {index}: {len(segments)} segments")
        print(f"{'═'*50}")
        
        # Execute all segments in parallel
        with ThreadPoolExecutor(max_workers=len(segments)) as executor:
            # Submit all segments
            futures = {
                executor.submit(
                    process_video_selection_segment,
                    segment_idx,
                    vo_text,
                    slide_title,
                    slide_chunk,
                    video_pool,
                    video_pool_other_channels,
                    course_name,
                    topic_name,
                    subtopic_name,
                    drive,
                    llm,
                    shortlist_video_urls_from_score_text(video_score_text, segment_idx, top_n=2),
                ): segment_idx
                for segment_idx, vo_text in segments
            }
            
            # Collect results as they complete
            segment_results = {}
            for future in as_completed(futures):
                segment_idx = futures[future]
                try:
                    seg_idx, formatted_segment = future.result()
                    if formatted_segment:
                        segment_results[seg_idx] = formatted_segment
                except Exception as e:
                    print(f"❌ Error processing segment {segment_idx}: {e}")
        
        # Format results in order (by segment_idx)
        segment_selections = []
        for segment_idx in sorted(segment_results.keys()):
            segment_selections.append(segment_results[segment_idx])
        
        # Combine all segment selections with double newline separator
        if segment_selections:
            video_pool_filtered_text = "\n\n".join(segment_selections)
            print(f"✅ Combined {len(segment_selections)} segment selections")
            return index, video_pool_filtered_text
        else:
            print(f"⚠️  No selections generated")
            return index, ""
        
    except Exception as e:
        print(f"Error processing row {index}: {e}")
        import traceback
        traceback.print_exc()
        return index, ""


def process_video_scoring_row(index, row, course_name, drive, llm="gemini_3_flash_thinking"):
    """
    Score video candidates per set and return formatted video_score text.

    :param index: Row index
    :param row: Row data
    :param course_name: Course name
    :param drive: Drive instance
    :param llm: Model name
    :return: Tuple (row_index, video_score_text)
    """
    try:
        voiceover_segments = str(row.get("voiceover_segment", "")).strip()
        video_pool = str(row.get("video_pool", "")).strip()
        video_pool_other_channels = str(row.get("video_pool_other_channels", "")).strip()
        slide_chunk = str(row.get("Slide Chunk", "")).strip()
        slide_title = str(row.get("Slide Chunk Title", "")).strip()
        topic_name = str(row.get("Topic", "")).strip()
        subtopic_name = str(row.get("Subtopic", "")).strip()
        visual_assignment_strategy = str(row.get("Visual Assignment Strategy", "Flexible, let the agent decide")).strip()
        if not visual_assignment_strategy or visual_assignment_strategy == "nan":
            visual_assignment_strategy = "Flexible, let the agent decide"
        if not slide_chunk or slide_chunk == "nan":
            return index, ""

        def _collect_sources(segment_num):
            return (
                parse_urls_from_video_pool(video_pool, segment_num),
                parse_video_items_from_pool_other_channels(video_pool_other_channels, segment_num),
            )

        segment_blocks = []
        if visual_assignment_strategy == "1 Visual for the whole Slide":
            pool_urls, other_items = _collect_sources(1)
            combined = [{"type": "pool", "url": u} for u in pool_urls] + [{"type": "other", "url": v.get("url", ""), "meta": v} for v in other_items if v.get("url", "")]
            batches = build_dynamic_batches(combined)
            set_results = []
            if batches:
                with ThreadPoolExecutor(max_workers=len(batches)) as executor:
                    futures = []
                    for batch in batches:
                        b_pool = [x["url"] for x in batch if x["type"] == "pool"]
                        b_other = [x["meta"] for x in batch if x["type"] == "other"]
                        futures.append(executor.submit(score_videos_batch, "", slide_title, slide_chunk, b_pool, b_other, course_name, topic_name, subtopic_name, drive, llm, True))
                    for future in futures:
                        set_results.append(future.result() or [])
            segment_blocks.append(format_video_score_segment(1, set_results))
        else:
            segments = parse_segments_from_voiceover(voiceover_segments)
            for segment_idx, vo_text in segments:
                pool_urls, other_items = _collect_sources(segment_idx)
                combined = [{"type": "pool", "url": u} for u in pool_urls] + [{"type": "other", "url": v.get("url", ""), "meta": v} for v in other_items if v.get("url", "")]
                batches = build_dynamic_batches(combined)
                set_results = []
                if batches:
                    with ThreadPoolExecutor(max_workers=len(batches)) as executor:
                        futures = []
                        for batch in batches:
                            b_pool = [x["url"] for x in batch if x["type"] == "pool"]
                            b_other = [x["meta"] for x in batch if x["type"] == "other"]
                            futures.append(executor.submit(score_videos_batch, vo_text, slide_title, slide_chunk, b_pool, b_other, course_name, topic_name, subtopic_name, drive, llm, False))
                        for future in futures:
                            set_results.append(future.result() or [])
                segment_blocks.append(format_video_score_segment(segment_idx, set_results))
        return index, "\n\n".join([b for b in segment_blocks if b.strip()])
    except Exception as e:
        print(f"Error processing video scoring row {index}: {e}")
        return index, ""


@traceable(
    metadata={
        "agent_name": "graphics_definition_v2",
        "step_name": "Video Scoring",
        "function_name": "run_video_scoring_for_all_rows",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous")
    }
)
def run_video_scoring_for_all_rows(sheet, llm="gemini_3_flash_thinking", max_workers=50, progress_callback=None, show_progress: bool = True, selected_topics=None):
    """
    Run video scoring for all eligible rows and write results to video_score.

    :param sheet: gspread sheet object
    :param llm: LLM model name
    :param max_workers: Row-level parallel workers
    :param progress_callback: Optional callback for external progress
    :param show_progress: Toggle internal Streamlit progress bar
    :return: None
    """
    
    worksheet_name = "Slide Chunks"
    drive = get_drive_instance()
    _, course_info_df = get_sheet_data_and_df(sheet, "Course info")
    course_name = course_info_df.loc[0, "Course Name"]
    _, df = get_sheet_data_and_df(sheet, worksheet_name)
    if SCORE_COLUMN_NAME not in df.columns:
        df[SCORE_COLUMN_NAME] = ""

    futures_map = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for index, row in df.iterrows():
            topic_name = str(row.get("Topic", "")).strip()
            if selected_topics and topic_name not in selected_topics:
                continue
            voiceover_segments = str(row.get("voiceover_segment", "")).strip()
            slide_chunk = str(row.get("Slide Chunk", "")).strip()
            existing_score = str(row.get(SCORE_COLUMN_NAME, "")).strip()
            video_pool = str(row.get("video_pool", "")).strip()
            video_pool_other_channels = str(row.get("video_pool_other_channels", "")).strip()
            if not voiceover_segments or voiceover_segments == "nan" or not slide_chunk or slide_chunk == "nan":
                continue
            if (not video_pool or video_pool == "nan") and (not video_pool_other_channels or video_pool_other_channels == "nan"):
                continue
            if existing_score and existing_score != "nan" and not existing_score.startswith("ERROR:"):
                continue
            futures_map[executor.submit(process_video_scoring_row, index, row, course_name, drive, llm)] = index

        if not futures_map:
            print("All rows already scored for video_score or no valid rows found.")
            return

        total_tasks = len(futures_map)
        progress = SmartProgressBar(total_tasks=total_tasks, description="Scoring videos", save_interval=SAVE_INTERVAL_ROWS) if show_progress else None
        completed_count = 0
        for future in as_completed(futures_map):
            index = futures_map[future]
            try:
                row_index, score_text = future.result()
                df.at[row_index, SCORE_COLUMN_NAME] = score_text
            except Exception as e:
                df.at[index, SCORE_COLUMN_NAME] = f"ERROR: {str(e)}"
            if progress is not None:
                progress.update()
                if progress.should_save():
                    merge_and_save_columns(sheet, worksheet_name, df, [SCORE_COLUMN_NAME])
            else:
                completed_count += 1
                if progress_callback:
                    progress_callback(1)
                if completed_count % SAVE_INTERVAL_ROWS == 0:
                    merge_and_save_columns(sheet, worksheet_name, df, [SCORE_COLUMN_NAME])
    merge_and_save_columns(sheet, worksheet_name, df, [SCORE_COLUMN_NAME])


def validate_video_pool_filtered_row(row):
    """
    Validate that video_pool_filtered matches voiceover_segment:
    - Row is not empty
    - All segments from voiceover_segment have results (unless "1 Visual for the whole Slide")
    - No gaps in segment numbering (must be sequential starting from 1)
    - For "1 Visual for the whole Slide", only SEGMENT_1 is expected
    
    :param row: Pandas Series with row data
    :return: Tuple (is_valid, error_message)
    """
    vo_segments_text = str(row.get("voiceover_segment", "")).strip()
    video_pool_filtered_text = str(row.get("video_pool_filtered", "")).strip()
    slide_type = str(row.get("Slide Type", "")).strip().lower()

    # Transition slides have no video candidates by design; empty video_pool_filtered is valid
    if slide_type in ("transition", "transition slide"):
        return True, None

    # Get Visual Assignment Strategy
    visual_assignment_strategy = str(row.get("Visual Assignment Strategy", "Flexible, let the agent decide")).strip()
    if not visual_assignment_strategy or visual_assignment_strategy == "nan":
        visual_assignment_strategy = "Flexible, let the agent decide"

    # Skip validation if voiceover_segment is empty
    if not vo_segments_text or vo_segments_text == "nan":
        return True, None

    # Skip validation if video_pool_filtered is empty (will be caught by retry logic)
    if not video_pool_filtered_text or video_pool_filtered_text == "nan" or video_pool_filtered_text.strip() == "":
        return False, "video_pool_filtered is empty"
    
    # Skip validation if it's an error marker
    if video_pool_filtered_text.startswith("ERROR:"):
        return False, "video_pool_filtered contains error marker"
    
    # Parse segment numbers from video_pool_filtered
    segment_pattern = r'---SEGMENT_(\d+)---'
    segment_numbers = [int(match) for match in re.findall(segment_pattern, video_pool_filtered_text)]
    
    if not segment_numbers:
        return False, "No segment markers found in video_pool_filtered"
    
    # Handle "1 Visual for the whole Slide" case - only expect SEGMENT_1
    if visual_assignment_strategy == "1 Visual for the whole Slide":
        if len(segment_numbers) != 1 or segment_numbers[0] != 1:
            return False, f"For '1 Visual for the whole Slide', expected only SEGMENT_1, found: {segment_numbers}"
        return True, None
    
    # For other strategies, validate against voiceover_segment count
    # Count segments in voiceover_segment (split by newline)
    vo_segments = [seg.strip() for seg in vo_segments_text.split('\n') if seg.strip()]
    expected_count = len(vo_segments)
    
    # Check count matches
    actual_count = len(segment_numbers)
    if actual_count != expected_count:
        return False, f"Segment count mismatch: expected {expected_count}, found {actual_count}"
    
    # Check for gaps (must be sequential starting from 1)
    segment_numbers_sorted = sorted(segment_numbers)
    expected_sequence = list(range(1, expected_count + 1))
    
    if segment_numbers_sorted != expected_sequence:
        missing = set(expected_sequence) - set(segment_numbers_sorted)
        extra = set(segment_numbers_sorted) - set(expected_sequence)
        error_parts = []
        if missing:
            error_parts.append(f"missing segments: {sorted(missing)}")
        if extra:
            error_parts.append(f"extra segments: {sorted(extra)}")
        return False, f"Segment numbering gap: {', '.join(error_parts)}"
    
    return True, None


@traceable(
    metadata={
        "agent_name": "graphics_definition_v2",
        "step_name": "Video Selection from All Videos",
        "function_name": "run_video_selection_from_all_videos_for_all_rows",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous")
    }
)
def run_video_selection_from_all_videos_for_all_rows(sheet, llm="gemini_3_flash_thinking", max_workers=50, progress_callback=None, show_progress: bool = True, selected_topics=None):
    """
    Select relevant videos from all available videos for all rows in the Slide Chunks sheet.
    
    :param sheet: The gspread sheet object.
    :param llm: Language model to use.
    :param max_workers: Number of parallel workers (default 3, lower due to video frame extraction).
    :param progress_callback: Optional callback invoked as each initial row completes.
    :param show_progress: If False, disable internal Streamlit progress bar (thread-safe for parallel outer steps).
    :return: None if initialization fails
    """
    print(f"\n{'='*80}")
    print(f"🤖 VIDEO SELECTION: Using LLM model: {llm}")
    print(f"{'='*80}\n")
    worksheet_name = "Slide Chunks"
    
    # Get drive instance
    drive = get_drive_instance()
    if not drive:
        print("⚠️ Drive instance not available. Some video frames may not be extracted.")
    
    # Fetch course info
    _, course_info_df = get_sheet_data_and_df(sheet, "Course info")
    course_name = course_info_df.loc[0, "Course Name"]
    
    # Load the worksheet and DataFrame
    worksheet, df = get_sheet_data_and_df(sheet, worksheet_name)
    
    # Ensure video_pool_filtered column exists
    if "video_pool_filtered" not in df.columns:
        df["video_pool_filtered"] = ""
    
    # Prepare for parallel processing - only process rows that need processing
    futures_map = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks first
        for index, row in df.iterrows():
            topic_name = str(row.get("Topic", "")).strip()
            if selected_topics and topic_name not in selected_topics:
                continue
            voiceover_segments = str(row.get("voiceover_segment", "")).strip()
            video_pool = str(row.get("video_pool", "")).strip()
            video_pool_other_channels = str(row.get("video_pool_other_channels", "")).strip()
            video_pool_filtered = str(row.get("video_pool_filtered", "")).strip()
            slide_type = str(row.get("Slide Type", "")).strip().lower()
            
            # Transition slides: no video filtering; leave video_pool_filtered empty
            if slide_type in ("transition", "transition slide"):
                continue
            
            # Skip if no video sources available
            if (not video_pool or video_pool == "nan") and (not video_pool_other_channels or video_pool_other_channels == "nan"):
                continue
            
            # Skip if video_pool_filtered is already filled
            # Rows marked with "ERROR:" should be retried on reruns.
            if video_pool_filtered and video_pool_filtered != "nan" and not str(video_pool_filtered).startswith("ERROR:"):
                print(f"⏭️ Skipping row {index + 2}: video_pool_filtered already filled.")
                continue
            
            # Submit task for processing
            future = executor.submit(process_video_selection_row, index, row, course_name, drive, llm)
            futures_map[future] = index
        
        # If no rows to process, return early
        if not futures_map:
            print("All rows already processed or no valid video sources found.")
            return
        
        # Progress tracking
        total_tasks = len(futures_map)
        save_interval = 3 
        progress = None
        completed_count = 0
        if show_progress:
            progress = SmartProgressBar(
                total_tasks=total_tasks,
                description="Selecting videos from all available videos",
                save_interval=save_interval,
            )
        
        # Collect results as they complete
        for future in as_completed(futures_map):
            index = futures_map[future]
            try:
                row_index, video_pool_filtered_text = future.result()
                
                # Update dataframe
                df.at[row_index, "video_pool_filtered"] = video_pool_filtered_text
                
                # Update progress
                if progress is not None:
                    progress.update()
                else:
                    completed_count += 1
                    if progress_callback:
                        progress_callback(1)
                
                # Save every 3 rows (more frequent due to longer processing)
                # Use merge_and_save_columns to avoid overwriting other parallel workers' columns
                if progress is not None:
                    if progress.should_save():
                        print(f'Saving partial progress to sheet after {progress.completed_count} tasks completed.')
                        merge_and_save_columns(sheet, worksheet_name, df, ["video_pool_filtered"])
                else:
                    if save_interval > 0 and completed_count % save_interval == 0:
                        print(f"Saving partial progress to sheet after {completed_count} tasks completed.")
                        merge_and_save_columns(sheet, worksheet_name, df, ["video_pool_filtered"])
            except Exception as e:
                print(f"Error getting result for row {index}: {e}")
                # Update dataframe with error marker so row is marked as processed
                df.at[index, "video_pool_filtered"] = f"ERROR: {str(e)}"
                if progress is not None:
                    progress.update()
                else:
                    completed_count += 1
                    if progress_callback:
                        progress_callback(1)

    # Save final results before validation (merge-safe)
    merge_and_save_columns(sheet, worksheet_name, df, ["video_pool_filtered"])

    # Validation and retry logic
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        # Reload dataframe to get latest state
        worksheet, df = get_sheet_data_and_df(sheet, worksheet_name)
        
        # Validate all rows and find invalid ones
        invalid_rows = []
        for index, row in df.iterrows():
            topic_name = str(row.get("Topic", "")).strip()
            if selected_topics and topic_name not in selected_topics:
                continue
            is_valid, error_msg = validate_video_pool_filtered_row(row)
            if not is_valid:
                invalid_rows.append((index, row, error_msg))
        
        if not invalid_rows:
            break
        
        retry_count += 1
        print(f"\n⚠️ Found {len(invalid_rows)} rows with invalid video_pool_filtered. Retrying (attempt {retry_count}/{max_retries})...")
        for idx, row, error in invalid_rows[:3]:  # Show first 3 errors
            print(f"  Row {idx}: {error}")
        
        # Clear video_pool_filtered for invalid rows
        for index, row, error_msg in invalid_rows:
            df.at[index, "video_pool_filtered"] = ""
        
        # Save cleared state (merge-safe)
        merge_and_save_columns(sheet, worksheet_name, df, ["video_pool_filtered"])
        
        # Retry processing invalid rows
        futures_map = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for index, row, error_msg in invalid_rows:
                topic_name = str(row.get("Topic", "")).strip()
                if selected_topics and topic_name not in selected_topics:
                    continue
                future = executor.submit(process_video_selection_row, index, row, course_name, drive, llm)
                futures_map[future] = index
            
            # Collect results
            for future in as_completed(futures_map):
                index = futures_map[future]
                try:
                    row_index, video_pool_filtered_text = future.result()
                    df.at[row_index, "video_pool_filtered"] = video_pool_filtered_text
                except Exception as e:
                    print(f"Error getting result for row {index} on retry: {e}")
                    df.at[index, "video_pool_filtered"] = f"ERROR: {str(e)}"
        
        # Save after retry (merge-safe)
        merge_and_save_columns(sheet, worksheet_name, df, ["video_pool_filtered"])
    
    if retry_count > 0:
        # Check final state
        worksheet, df = get_sheet_data_and_df(sheet, worksheet_name)
        final_invalid = []
        for index, row in df.iterrows():
            topic_name = str(row.get("Topic", "")).strip()
            if selected_topics and topic_name not in selected_topics:
                continue
            is_valid, error_msg = validate_video_pool_filtered_row(row)
            if not is_valid:
                final_invalid.append((index, error_msg))
        
        if final_invalid:
            print(f"⚠️ After {retry_count} retry attempt(s), {len(final_invalid)} rows still have invalid video_pool_filtered.")
            for idx, error in final_invalid[:5]:  # Show first 5
                print(f"  Row {idx}: {error}")
        else:
            print(f"✅ All rows validated after {retry_count} retry attempt(s).")
    
    # Final save to sheet (merge-safe)
    print('All video selections completed. Saving final DataFrame to sheet.')
    merge_and_save_columns(sheet, worksheet_name, df, ["video_pool_filtered"])
    print("✅ Video selection from all videos complete and saved to sheet.")

    if not show_progress and progress_callback and total_tasks > 0:
        progress_callback(1)


def delete_video_pool_filtered(sheet):
    """
    Remove the 'video_pool_filtered' column from the specified worksheet.
    
    :param sheet: The gspread sheet object.
    :return: None
    """
    worksheet_name = "Slide Chunks"
    ws, df = get_sheet_data_and_df(sheet, worksheet_name)
    if "video_pool_filtered" in df.columns:
        df = df.drop(columns=["video_pool_filtered"])
        clear_worksheet(ws)
        save_to_sheet(ws, df)
        print(f"🗑️ Deleted 'video_pool_filtered' column from '{worksheet_name}' worksheet")
    else:
        print(f"ℹ️ 'video_pool_filtered' column does not exist in '{worksheet_name}' worksheet")
