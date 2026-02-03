from langsmith import traceable
import streamlit as st
from services.sheets_service import get_sheet_data_and_df, save_to_sheet, format_worksheet, clear_worksheet
from services.smart_progress_bar import SmartProgressBar
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import requests
from io import BytesIO
from PIL import Image
from agents.vector_store_image_search.graphics_retriever_agent import pil_to_base64_data_uri
from agents.vector_store_image_search.create_vectorstore import download_image_from_drive
from services.video_clip_tools import build_video_part
from services.llm_service import log_token_usage
from dotenv import load_dotenv
import os
import base64
from services.drive_service import login_with_service_account
from pydrive2.drive import GoogleDrive
from google import genai
from google.genai import types
from typing import List, Dict, Optional, Tuple, Any
import tempfile
import subprocess
import urllib.parse

load_dotenv()

# Drive folder ID for storing extracted video frames
video_frames_drive_folder_id = "1sab6wSDPLj54q7KMumGwB1oZHRzXmVf-"


aggregation_agent_prompt = """You are a senior expert graphics designer specializing in the field of HVAC. Your role is to assemble the final, production-ready graphics definition for a single voiceover sentence so that it can be directly used by a graphics team to design the corresponding visuals for an educational e-learning slide. 
You will be given the voiceover sentence, the full course and slide context for reference, and a set of image and video candidates that have already been identified for this sentence by upstream agents, some of which may be only partially relevant or not ultimately suitable for use. Your task is to decide which visuals should be selected from the provided image and video candidates to best support the entire voiceover sentence visually, and whether those visuals should be images, video clips, still frames extracted from videos, or a combination of these. You will also be given a storyboard reference that describes the intended visual ideas for the slide; use it as guidance, but do not treat it as a strict template—your final decisions must be based on what is most clear and accurate given the provided candidates.

These are the inputs:

<course_information>
Course name: {course_name}
Topic name: {topic_name}
Subtopic name: {subtopic_name}
</course_information>

<voiceover_sentence_for_which_to_assemble_graphics_definition>
{vo_text}
</voiceover_sentence_for_which_to_assemble_graphics_definition>

<whole_slide_context>
Slide Title: {slide_title}
Slide Content: {slide_chunk}
</whole_slide_context>

<storyboard_reference>
{storyboard}
</storyboard_reference>

<image_candidates>
{image_candidates}
</image_candidates>

<video_candidates>
{video_candidates}
</video_candidates>

Instructions:

1. Scope and Decision Responsibility
   - Your task is to assemble a final graphics definition for the given voiceover sentence only.
   - Use the full slide context for reference and continuity awareness.
   - The assigned visuals are displayed on screen as the voiceover text for that segment is played/narrated.
   - When interpreting any part of the voiceover sentences, you must use the surrounding text in the slide content to understand the context and the instructional intent, before selecting visuals for the respective parts of the slide. Do not interpret a sentence, phrase or a clause literally in isolation. Always derive the meaning of a sentence, phrase or a clause from the surrounding text in the slide content.
   - Select visuals only from the provided image and video candidates.
   - The provided storyboard reference describes the intended visual plan for the slide. 
   - Use the storyboard to understand the visual intent and progression for the voiceover sentence for which you are assembling the graphics definition.
   - Aim to follow the storyboard’s visual idea and sequencing as closely as possible when suitable image or video candidates are available. 
   - If the available image and video candidates do not fully support a storyboard-suggested visual idea, adapt by selecting the most instructionally clear and relevant visuals based on the voiceover sentence and the available candidates.

2. Visual Coverage of the Entire Sentence  
   - First, understand the full meaning and instructional intent of the entire voiceover sentence.
   - Identify the key visual ideas that must be shown on screen for the sentence to be clearly understood
   - Select visuals so that the chosen visual or visuals, taken together, fully support the complete meaning of the voiceover sentence.

3. Allowed Visual Selection Forms
   - You may select one or more still images from the provided image candidates.
   - You may select one or more segments from the provided video candidates, including short portions of a video clip that are most relevant to the voiceover sentence.
   - You may select a specific still frame from a provided video clip and use it as a static image.
   - You may use a combination of still images, video clips, and still frames extracted from video clips, as long as the selected visuals collectively support the entire voiceover sentence.

4. Source-Specific Video Usage Constraints
   - Video candidates are divided into two distinct groups based on their source.
      - Videos listed under:
        "Videos from which you can use video clips (with timestamps) or still frames as images" may be used in any of the following ways:
        - short video clips with start and end timestamps
        - still frames extracted from the video

      - Video candidates listed under:
        "Videos from which you can ONLY use still frames as images (NOT playable video clips with timestamps)" have the following strict constraints:
        - You MUST NOT select them as playable video clips
        - You MUST NOT assign start–end timestamps
        - You MAY ONLY extract still frames and use them as static images
        - When using these videos, the asset MUST be represented as a video URL with a single start timestamp only

5. Time-Constrained Visual Design
   - Visuals are displayed only during the narration of the voiceover sentence.
   - Select the minimum number of visuals required to clearly support the sentence within this limited time.
   - Do not select lots of visuals or long video clips that cannot be realistically shown during the narration of the given sentence.

6. Alignment of Visuals to the Voiceover Sentence
   - For each selected visual, indicate which part of the voiceover sentence it should appear with during narration.
   - Align visuals to the natural progression of the sentence so that each visual appears when the corresponding idea is being spoken.
   - If multiple visuals are selected, ensure they are ordered and aligned in a way that makes the sentence easy to follow visually within the narration time.
   - Do not assign visuals to parts of the sentence that they do not clearly support.
   - If strict adherence to the storyboard’s narration-to-visual mapping is not feasible due to the available image or video candidates, you may adjust how visuals are aligned to narration parts, provided the final alignment remains instructionally clear and faithful to the meaning of the voiceover sentence.

7. Instructional Clarity Priority
   - Prioritize instructional clarity and accuracy over visual richness, variety or strict adherence to the storyboard.
   - When you find both a video clip and a still image that are equally clear, directly relevant, and instructionally effective for any part of the voiceover sentence, prefer using the video clip, since motion can add useful context. This is a guiding preference, not a strict rule - do not prioritize a video clip over an image if the video is only partially relevant, loosely related, or less effective than the still image at supporting the narration.
   - Always remember that the storyboard is a reference and not a strict template regarding the kind of visuals that should be used for the voiceover sentence. Try your best to follow the storyboard's visual ideas and sequencing, but do not be too rigid about it.

Strictly provide your output in the following format:

<output>

<evaluation_breakdown>

This section is your reasoning scratchpad used to analyze the voiceover sentence and the available visual candidates before producing the final graphics definition output. Use it to document your observations, reasoning, and decision process. Provide the following sections:

- Voiceover sentence understanding: 
Briefly explain, in your own words, what the voiceover sentence is communicating. Use the full slide context to resolve any references, pronouns, or implied meaning if needed.

- Storyboard reference scan: 
Explain the storyboard’s intended visual idea(s) that correspond to this voiceover sentence. Summarize what the storyboard is trying to show for this sentence in 1–3 concise bullet points.

- Image candidates scan:
Create a numbered list of all provided image candidates and briefly describe what you see in each of the image candidate.

- Video candidate scan: 
Create a numbered list of all provided video candidates and briefly describe what you see in each of the video candidate. Focus on explaining the visual content of the video, and not what is being spoken in the video. Split the list into two sections: one for videos from which video clips (with timestamps) or still frames as images can be used, and one for videos from which ONLY still frames as images can be used (NOT playable video clips with timestamps).

- Detailed overall analysis: 
Use this section to reason through how to assemble the final graphics definition for the given voiceover sentence.

Apply the instruction guidelines to:
- Carefully review each of the provided image and video candidate in detail before making any selection decisions.
- For video candidates, pay close attention to the visual content within the video to identify whether a video segment or a specific still frame from the video can be used as a suitable visual for the corresponding part of the voiceover sentence. Consider whether any visually clear, frame-worthy moments within the videos could be used as static images.
- Carefully review the storyboard reference to understand the intended visual idea(s) for this sentence, then evaluate how well the available image and video candidates can satisfy that intent.
- Decide whether to use still images, video segments, still frames from videos, or a combination of these. While considering the use of any specific video clip for a particular part of the voiceover sentence, determine exactly which portion of the video is visually relevant to help assign correct start and end timestamps for the video clip that you will select.
- If strict adherence to the storyboard’s narration-to-visual alignment is not feasible due to the available image or video candidates, explain how you plan to adjust the alignment to best support the voiceover sentence, while still maintaining instructionally clear and faithful to the meaning of the voiceover sentence.
- Ensure the selected visuals, taken together, fully support the entire meaning of the voiceover sentence.
- Account for the narration time of the voiceover sentence and how all selected visuals should realistically fit within that timeframe.
- Plan how the selected visuals align with the progression of the voiceover sentence.
- Consider the source-specific video usage constraints when selecting video candidates.
- Address any other reasoning considerations needed to arrive at a clear and instructionally useful final decision.

Document your reasoning, tradeoffs, and decision process as you work toward the final selection. It is ok for this evaluation breakdown section to be quite long to fit all your reasoning and intermediate thinking needed to arrive at the best final decision.

</evaluation_breakdown>

(Based on your above evaluation, provide the final graphics definition for the voiceover sentence in the following format)

<final_graphics_definition>

<visual_steps>

<visual_step>

<voiceover_part>
(Exact phrase or clause from the voiceover sentence that this visual aligns with)
</voiceover_part>

<visual_instruction>
Concise description of what appears on screen for this part of the narration, using only the selected visual asset.
</visual_instruction>

<asset>
The visual asset selected to use for this step, in one of the following forms:
- Image URL (Exact image URL as provided in the image candidates if an image is selected for this part of the voiceover sentence)
- Video URL with start and end timestamps (if a portion of a video clip is selected for this part of the voiceover sentence. Strictly use such example format of video URL with start and end timestamps: e.g. "https://www.youtube.com/embed/dQw4w9WgXc?start=10&end=20")
- Video URL with a single start timestamp (if a still frame extracted from a video clip is selected for this part of the voiceover sentence. Strictly use such example format of video URL with a single start timestamp indicating the frame timestamp: e.g. "https://www.youtube.com/embed/dQw4w9WgXc?start=10")
</asset>

<selection_justification>
Briefly explain how the selected visual clearly supports this specific part of the voiceover sentence, based on what is visibly shown in the asset. Do not refer to the visual number/index while giving the justification (eg. don't say "Image 9 supports the voiceover sentence because it shows...", instead say "the selectedimage supports the voiceover sentence because it shows...")
</selection_justification>

</visual_step>

<!-- Repeat <visual_step> as needed, in the sentence narration order -->

</visual_steps>

</final_graphics_definition>

</output>

(Ensure that you strictly follow this exact XML format in your output)
"""


aggregation_agent_regeneration_prompt = """You are a Graphics Definition Regeneration Agent specializing in the field of HVAC. Your task is to regenerate and assemble a complete, production-ready graphics definition for a SINGLE voiceover (VO) segment when one or more of its previously assigned visuals have failed review checks, by selecting the most appropriate visuals from newly generated candidate image and video pools that were specifically searched based on the review feedback.

You will be given the voiceover segment, the slide and course context for reference, explicit review feedback describing what was wrong with the failed visuals and what is required instead, and a pool of newly searched candidate visuals from which to select visuals that directly address the feedback requirements and fully support the voiceover segment.

Inputs:
These are the inputs for your regeneration:

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

This is the voiceover segment from the slide content for which the regeneration is needed:
<voiceover_segment>
{vo_text}
</voiceover_segment>

This is the explicit review feedback describing what was wrong with the previously failed visuals and what is required instead:
<feedback>
{feedback}
</feedback>

These are the newly searched candidate images and videos from which to select visuals that address the feedback requirements:

<candidates>

<image_candidates>
{image_candidates}
</image_candidates>

<video_candidates>
{video_candidates}
</video_candidates>

</candidates>

Instructions:

1. Scope and Regeneration Responsibility
   - Your task is to regenerate visuals ONLY for the specific part(s) of the voiceover segment that are addressed by the feedback (i.e., the failed visuals).
   - Use the slide content and the specific voiceover segment to understand the context and the instructional intent.
   - The assigned visuals are displayed on screen as the voiceover text for that segment is played/narrated.
   - When interpreting any part of the voiceover sentences, you must use the surrounding text in the slide content to understand the context and the instructional intent, before selecting visuals for the respective parts of the slide. Do not interpret a sentence, phrase or a clause literally in isolation. Always derive the meaning of a sentence, phrase or a clause from the surrounding text in the slide content.
   - The candidate visuals provided were specifically searched based on the feedback requirements, so they should be more targeted to addressing the issues described in the feedback.
   - The feedback explicitly identifies which visual(s) failed and what is needed instead. Your primary responsibility is to select visuals that address these specific feedback requirements.
   - IMPORTANT: Only select visuals for the part(s) of the voiceover segment mentioned in the feedback. Do not select visuals for parts of the segment that are not mentioned in the feedback (those visuals passed review and will be preserved).
   - Select visuals only from the provided candidate image and video pools.

2. Feedback-Driven Visual Selection
   - Carefully analyze the feedback to understand what was wrong with the previously failed visuals and what specific visual requirements must be satisfied.
   - Identify which specific part(s) of the voiceover segment the feedback is addressing (these are the only parts you need to select visuals for).
   - Extract from the feedback: which Visual ID(s) failed, which voiceover part(s) they were assigned to, what was wrong, and what is needed instead.
   - When selecting visuals, prioritize candidates that directly address the feedback requirements for the specific voiceover parts mentioned in the feedback.
   - Ensure that the selected visuals avoid the same issues that caused the previous visuals to fail.
   - If the feedback addresses multiple failed visuals, ensure your selection addresses all the feedback requirements collectively.
   - Focus your visual selection on the feedback-addressed parts; do not create visuals for other parts of the sentence unless they are mentioned in the feedback.

3. Visual Coverage for Feedback-Addressed Parts
   - Understand the specific meaning and instructional intent of the voiceover part(s) mentioned in the feedback.
   - Identify what visual characteristics are needed to satisfy the feedback requirements for these specific parts.
   - Select visuals that fully support the feedback-addressed parts of the voiceover segment.

4. Candidate Evaluation and Visual Selection
   - Carefully review all provided image and video candidates for this voiceover segment.
   - Evaluate each candidate against both the feedback requirements and the overall voiceover segment needs.
   - Select candidates that most directly and clearly satisfy the feedback while remaining aligned with the voiceover segment's instructional intent.
   - If no candidate fully satisfies the feedback, determine the closest acceptable alternative that still addresses the core feedback concerns.
   - Ensure the selected visuals are instructionally clear and effective for the respective parts of the voiceover segment.

5. Visual Form and Usage Constraints
   - You may select still images, video clips with timestamps, or still frames extracted from videos, using only the provided candidate visuals.
   - Choose the visual form (image, video clip, or still frame) that most clearly satisfies the feedback requirements and supports the narration timing of the relevant part(s) of the voiceover segment.
   - When you find both a video clip and a still image that equally satisfy the feedback for any part of the voiceover sentence that is mentioned in the feedback, prefer using the video clip, since motion can add useful context. This is a guiding preference, not a strict rule - do not prioritize a video clip over an image if the video is only partially relevant, loosely related, or less effective than the still image at supporting the narration and addressing the feedback.
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

6. Time-Constrained Visual Design
   - Visuals are displayed only during the narration of the voiceover sentence.
   - If selecting a video clip, ensure that the selected portion can realistically fit within the narration timing of the specific part of the voiceover segment being addressed by the feedback.

7. Alignment of Visuals to the Voiceover Sentence
   - For each selected visual, indicate which part of the voiceover sentence it should appear with during narration.
   - Align visuals to the natural progression of the sentence so that each visual appears when the corresponding idea is being spoken.
   - If multiple visuals are selected, ensure they are ordered and aligned in a way that makes the sentence easy to follow visually within the narration time.
   - Do not assign visuals to parts of the sentence that they do not clearly support.
   - Pay special attention to aligning visuals that address feedback requirements to the specific parts of the voiceover segment mentioned in the feedback.

Output:

Always provide your output strictly in the following format:

<output>

<evaluation_breakdown>

Use this section as a structured reasoning and scratchpad space for you to address the feedback and assemble a complete graphics definition for the voiceover segment. Use it to document your observations, reasoning, and decision process. Provide the following sections:

1. Voiceover Sentence Understanding
   - Briefly explain, in your own words, what the voiceover sentence is communicating. Use the slide content to resolve any references, pronouns, or implied meaning if needed.

2. Feedback Interpretation
   - Carefully analyze the feedback to identify what was wrong with the previously failed visual(s).
   - Extract from the feedback: which Visual ID(s) failed (e.g., S1V1, S2V3), which specific voiceover part(s) they were assigned to, what was wrong with them, and what visual characteristics are needed instead.
   - Identify the exact part(s) of the voiceover sentence that the feedback is addressing - these are the ONLY parts you need to select visuals for.
   - Clearly state the specific visual requirements that must be satisfied based on the feedback.

3. Image Candidates Scan
   - Create a numbered list of all provided image candidates and briefly describe what you see in each of the image candidate.

4. Video Candidates Scan
   - Create a numbered list of all provided video candidates and briefly describe what you see in each of the video candidate.
   - Focus on explaining the visual content of the video, and not what is being spoken in the video.
   - Split the list into two sections: one for videos from which video clips (with timestamps) or still frames as images can be used, and one for videos from which ONLY still frames as images can be used (NOT playable video clips with timestamps).

5. Candidate Fit Analysis Against Feedback
   - Compare all the provided candidates against the feedback requirements for the specific voiceover parts mentioned in the feedback.
   - Identify which candidate(s) most directly satisfy the feedback requirements for these specific parts and why.
   - Evaluate how well candidates address each specific issue mentioned in the feedback.
   - If multiple candidates partially satisfy a particular feedback requirement, reason about which one best addresses that requirement.
   - If no candidate fully satisfies any of the feedback requirements, determine the closest acceptable alternative and explain why it is acceptable despite not fully matching.

6. Video Timestamp / Frame Selection Thinking (only if selecting video)
   - If selecting a playable video clip that best satisfies any of the feedback requirements:
     - Determine exactly which portion of the video is visually relevant and will satisfy the feedback.
     - Plan the correct start and end timestamps for that portion, ensuring the selected video clip can realistically fit within the narration timing of the specific part of the voiceover segment being addressed by the feedback.
   - If selecting a still frame from a video, determine the moment (single timestamp) that captures the required visual so that it can be used as a static image.

7. Additional Analysis
   - Note any additional observations, thoughts or analysis that can help you arrive at the correct output and decision that addresses all the given feedback for the feedback-addressed parts.
   - Consider how the selected visuals avoid the same issues that caused the previous visuals to fail.
   - Confirm that you are only creating visuals for the parts mentioned in the feedback, not for other parts of the segment that passed review.

(It is ok for this section to be quite verbose, long and detailed as long as it helps you arrive at the correct output.)

</evaluation_breakdown>

(Based on your above evaluation, provide the replacement visuals in the following format. Only provide replacement visuals for the specific visual(s) that failed (identified by Visual ID in the feedback). Do not create visuals for parts of the segment that are not mentioned in the feedback, as those visuals passed review and will be preserved.)

<replacement_visuals>

<visual>

<visual_id>
(Provide the Visual ID of the failed visual that you are replacing, as specified in the feedback, e.g. S1V1, S2V3)
</visual_id>

<voiceover_part>
(Provide the exact portion of the voiceover text to which this failed visual was assigned, as specified in the feedback)
</voiceover_part>

<current_visual_url>
(Provide the URL of the current visual that we are replacing)
</current_visual_url>

<replacement_visual_url>
(Provide the URL of the replacement visual that you are selecting for this voiceover part, in one of the following forms:
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

Repeat one <visual> block per failed visual that needs to be replaced based on the feedback for this voiceover segment.

</replacement_visuals>

</output>

(Remember to strictly use this exact format for the output with the XML tags and structure, and nothing else.)
"""


def get_drive_instance():
    """
    Get Google Drive instance from session state or initialize from environment.
    
    :return: Google Drive instance
    """
    if "drive" in st.session_state:
        return st.session_state["drive"]
    
    try:
        key_bytes = base64.b64decode(os.environ["GDRIVE_SA_B64"])
        sa_json = key_bytes.decode()
        gauth = login_with_service_account(json_str=sa_json)
        gauth.ServiceAuth()
        drive = GoogleDrive(gauth)
        return drive
    except Exception as e:
        print(f"⚠️ Could not initialize Drive from environment: {e}")
        return None


def extract_drive_file_id(url):
    """
    Extract the Google Drive file ID from a URL.

    :param url: Google Drive URL
    :return: Google Drive file ID or None if not found
    """
    if not url:
        return None
    
    # Pattern for Google Drive URLs
    patterns = [
        r"/file/d/([a-zA-Z0-9-_]+)",
        r"id=([a-zA-Z0-9-_]+)",
        r"drive.google.com/open\?id=([a-zA-Z0-9-_]+)",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    
    return None


def is_drive_url(url):
    """
    Check if URL is a Google Drive URL.
    
    :param url: URL to check
    :return: True if URL is a Google Drive URL, False otherwise
    """
    if not url:
        return False
    return "drive.google.com" in url.lower() or "docs.google.com" in url.lower()


def download_image_from_web_url(url):
    """
    Download an image from a web URL and return as PIL Image.

    :param url: Image URL
    :return: PIL Image object or None if download fails
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        return img.convert("RGB")
    except Exception as e:
        print(f"⚠️ Error downloading image from {url}: {e}")
        return None


def load_image_from_drive_url(url, drive, title=""):
    """
    Load an image from Google Drive URL and return as PIL Image.

    :param url: Google Drive URL
    :param drive: Google Drive instance
    :param title: Image title (for logging)
    :return: PIL Image object or None if load fails
    """
    if not drive:
        print(f"⚠️ Drive instance not available for loading image: {title}")
        return None
    
    try:
        file_id = extract_drive_file_id(url)
        if not file_id:
            print(f"⚠️ Could not extract file ID from URL: {url}")
            return None
        
        pil_image = download_image_from_drive(drive, file_id)
        if pil_image:
            return pil_image
        return None
    except Exception as e:
        print(f"⚠️ Error loading image from Drive ({title}): {e}")
        return None


def load_image_from_url(url, drive, title=""):
    """
    Load an image from URL (Drive or web) and return as PIL Image.

    :param url: Image URL
    :param drive: Google Drive instance
    :param title: Image title (for logging)
    :return: PIL Image object or None if load fails
    """
    if is_drive_url(url):
        return load_image_from_drive_url(url, drive, title)
    else:
        return download_image_from_web_url(url)


def parse_urls_from_results(results_text, segment_num):
    """
    Parse titles and URLs from drive_results or web_results for a specific segment.
    
    Format: "Title: {title} | URL: {url}"
    
    :param results_text: The drive_results or web_results column content
    :param segment_num: Segment number to extract URLs for
    :return: List of dictionaries with 'title' and 'url' keys or empty list if no URLs found
    """
    if not results_text or results_text.strip() == "" or results_text == "nan":
        return []
    
    # Find the segment section
    segment_pattern = rf'---SEGMENT_{segment_num}---\s*\n(.*?)(?=\n---SEGMENT_|\Z)'
    match = re.search(segment_pattern, results_text, re.DOTALL)
    
    if not match:
        return []
    
    segment_content = match.group(1).strip()
    items = []
    
    # Parse each line: "Title: {title} | URL: {url}"
    for line in segment_content.split('\n'):
        line = line.strip()
        if not line:
            continue
        
        # Try to parse "Title: ... | URL: ..." format
        if " | URL: " in line:
            parts = line.split(" | URL: ", 1)
            if len(parts) == 2:
                title_part = parts[0]
                url = parts[1].strip()
                
                # Extract title (remove "Title: " prefix)
                if title_part.startswith("Title: "):
                    title = title_part[7:].strip()
                else:
                    title = title_part.strip()
                
                if url:
                    items.append({"title": title, "url": url})
        # Fallback: if line is just a URL (for backward compatibility)
        elif line.startswith('http'):
            items.append({"title": "Untitled", "url": line})
    
    return items


def parse_urls_from_video_pool(video_pool_text, segment_num):
    """
    Parse video URLs from video_pool column for a specific segment.
    
    :param video_pool_text: The video_pool column content
    :param segment_num: Segment number to extract URLs for
    :return: List of video URLs (YouTube embed URLs with timestamps) or empty list if no URLs found
    """
    if not video_pool_text or video_pool_text.strip() == "" or video_pool_text == "nan":
        return []
    
    # Find the segment section
    segment_pattern = rf'---SEGMENT_{segment_num}---\s*\n(.*?)(?=\n---SEGMENT_|\Z)'
    match = re.search(segment_pattern, video_pool_text, re.DOTALL)
    
    if not match:
        return []
    
    segment_content = match.group(1).strip()
    urls = []
    
    # Each line is a video URL
    for line in segment_content.split('\n'):
        line = line.strip()
        if line and line.startswith('http'):
            urls.append(line)
    
    return urls


def parse_urls_from_video_pool_other_channels(video_pool_other_channels_text, segment_num):
    """
    Parse video URLs from video_pool_other_channels column for a specific segment.
    
    Format: "Title: {title} | Duration: {duration} | Channel: {channel} | URL: {url}"
    
    :param video_pool_other_channels_text: The video_pool_other_channels column content
    :param segment_num: Segment number to extract URLs for
    :return: List of video URLs (YouTube watch URLs without timestamps) or empty list if no URLs found
    """
    if not video_pool_other_channels_text or video_pool_other_channels_text.strip() == "" or video_pool_other_channels_text == "nan":
        return []
    
    # Find the segment section
    segment_pattern = rf'---SEGMENT_{segment_num}---\s*\n(.*?)(?=\n---SEGMENT_|\Z)'
    match = re.search(segment_pattern, video_pool_other_channels_text, re.DOTALL)
    
    if not match:
        return []
    
    segment_content = match.group(1).strip()
    urls = []
    
    # Parse each line: "Title: ... | Duration: ... | Channel: ... | URL: ..."
    for line in segment_content.split('\n'):
        line = line.strip()
        if not line:
            continue
        
        # Try to extract URL from the formatted line
        # Format: "Title: ... | Duration: ... | Channel: ... | URL: ..."
        if " | URL: " in line:
            parts = line.split(" | URL: ", 1)
            if len(parts) == 2:
                url = parts[1].strip()
                if url and url.startswith('http'):
                    urls.append(url)
        elif " | URL:" in line:
            parts = line.split(" | URL:", 1)
            if len(parts) == 2:
                url = parts[1].strip()
                if url and url.startswith('http'):
                    urls.append(url)
        # Fallback: if line is just a URL
        elif line.startswith('http'):
            urls.append(line)
    
    return urls


def convert_watch_url_to_embed_url(watch_url):
    """
    Convert YouTube watch URL to embed URL.
    
    :param watch_url: YouTube watch URL (e.g., "https://www.youtube.com/watch?v=VIDEO_ID")
    :return: YouTube embed URL (e.g., "https://www.youtube.com/embed/VIDEO_ID") or original URL if conversion fails
    """
    if not watch_url:
        return watch_url
    
    # Extract video ID from watch URL
    video_id = extract_video_id_from_url(watch_url)
    if video_id:
        return f"https://www.youtube.com/embed/{video_id}"
    
    # If already an embed URL, return as is
    if 'youtube.com/embed' in watch_url:
        return watch_url
    
    # If conversion fails, return original
    return watch_url


def _build_asset_parts_for_failed_visual(asset_url, drive):
    """
    Build multimodal parts for a failed visual asset (image or video).
    
    :param asset_url: URL of the failed visual asset
    :param drive: Google Drive instance
    :return: List of parts (visual part only, no text label)
    """
    parts: List[types.Part] = []
    
    # Check if it's a YouTube URL
    if "youtube.com" in asset_url or "youtu.be" in asset_url:
        clip_url, start_seconds, end_seconds = parse_video_url_timestamps(asset_url)
        if not clip_url:
            # Try converting watch URL to embed URL
            clip_url = convert_watch_url_to_embed_url(asset_url)
        if clip_url:
            video_part = build_video_part(clip_url, start_seconds, end_seconds)
            if video_part:
                parts.append(video_part)
        return parts
    
    # For images, load from URL
    pil_image = load_image_from_url(asset_url, drive, "failed_visual")
    if pil_image:
        # Convert PIL image to bytes
        buffered = BytesIO()
        pil_image.convert("RGB").save(buffered, format="JPEG")
        image_bytes = buffered.getvalue()
        parts.append(types.Part(inline_data=types.Blob(mime_type="image/jpeg", data=image_bytes)))
    
    return parts


def parse_video_url_timestamps(video_url):
    """
    Parse start and end timestamps from YouTube embed URL.
    
    :param video_url: YouTube embed URL like "https://www.youtube.com/embed/{vid_id}?start=10&end=20"
    :return: Tuple of (clip_url, start_seconds, end_seconds) or (None, None, None) if parsing fails
    """
    if not video_url:
        return None, None, None
    
    # Extract start and end timestamps
    start_match = re.search(r'[?&]start=(\d+)', video_url)
    end_match = re.search(r'[?&]end=(\d+)', video_url)
    
    start_seconds = int(start_match.group(1)) if start_match else None
    end_seconds = int(end_match.group(1)) if end_match else None
    
    # Extract base URL without query params for file_uri
    base_url_match = re.search(r'(https://www\.youtube\.com/embed/[^?]+)', video_url)
    if not base_url_match:
        return None, None, None
    
    clip_url = base_url_match.group(1)
    
    return clip_url, start_seconds, end_seconds


def extract_video_id_from_url(youtube_url):
    """
    Extract video ID from YouTube embed URL.
    
    :param youtube_url: YouTube URL (embed or watch format)
    :return: Video ID string or None if extraction fails
    """
    if not youtube_url:
        return None
    
    # Handle embed URLs: https://www.youtube.com/embed/VIDEO_ID
    embed_match = re.search(r'youtube\.com/embed/([a-zA-Z0-9_-]+)', youtube_url)
    if embed_match:
        return embed_match.group(1)
    
    # Handle watch URLs: https://www.youtube.com/watch?v=VIDEO_ID
    watch_match = re.search(r'youtube\.com/watch\?v=([a-zA-Z0-9_-]+)', youtube_url)
    if watch_match:
        return watch_match.group(1)
    
    # Handle youtu.be URLs: https://youtu.be/VIDEO_ID
    short_match = re.search(r'youtu\.be/([a-zA-Z0-9_-]+)', youtube_url)
    if short_match:
        return short_match.group(1)
    
    return None


def extract_frame_from_youtube_video(video_id, timestamp_seconds, output_path):
    """
    Extract a single frame from a YouTube video at a specific timestamp.
    Uses yt-dlp to download the video and ffmpeg to extract the frame.
    
    :param video_id: YouTube video ID
    :param timestamp_seconds: Timestamp in seconds to extract frame at
    :param output_path: Path where the extracted frame will be saved
    :return: True if successful, False otherwise
    """
    try:
        youtube_url = f"https://www.youtube.com/watch?v={video_id}"
        temp_video_path = os.path.join(tempfile.gettempdir(), f"{video_id}_{timestamp_seconds}_temp.%(ext)s")
        
        try:
            # Use yt-dlp to download best quality video
            # We'll try to download just a segment around the timestamp using --download-sections
            # Format: download-sections "*start-end" (in seconds)
            segment_start = max(0, timestamp_seconds - 10)  # 10 seconds before
            segment_end = timestamp_seconds + 10  # 10 seconds after
            
            yt_dlp_cmd = [
                "yt-dlp",
                "-f", "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best[height<=720]",
                "--extractor-args", "youtube:player_client=android",
                "--download-sections", f"*{segment_start}-{segment_end}",
                "--merge-output-format", "mp4",
                "-o", temp_video_path,
                "--no-playlist",
                youtube_url
            ]
            
            result = subprocess.run(
                yt_dlp_cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            # If --download-sections fails, fall back to downloading full video
            if result.returncode != 0:
                print(f"⚠️ Segment download failed, trying full video download: {result.stderr[:200]}")
                temp_video_path = os.path.join(tempfile.gettempdir(), f"{video_id}_{timestamp_seconds}_temp.%(ext)s")
                yt_dlp_cmd = [
                    "yt-dlp",
                    "-f", "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best[height<=720]",
                    "--extractor-args", "youtube:player_client=android",
                    "--merge-output-format", "mp4",
                    "-o", temp_video_path,
                    "--no-playlist",
                    youtube_url
                ]
                
                result = subprocess.run(
                    yt_dlp_cmd,
                    capture_output=True,
                    text=True,
                    timeout=600  # 10 minute timeout for full video
                )
            
            # If still failing, try even lower quality as last resort
            if result.returncode != 0:
                print(f"⚠️ Android client failed, trying lower quality format: {result.stderr[:200]}")
                temp_video_path = os.path.join(tempfile.gettempdir(), f"{video_id}_{timestamp_seconds}_temp.%(ext)s")
                yt_dlp_cmd = [
                    "yt-dlp",
                    "-f", "best[height<=480][ext=mp4]/best[height<=480]/worst",
                    "--merge-output-format", "mp4",
                    "-o", temp_video_path,
                    "--no-playlist",
                    youtube_url
                ]
                
                result = subprocess.run(
                    yt_dlp_cmd,
                    capture_output=True,
                    text=True,
                    timeout=600  # 10 minute timeout
                )
            
            if result.returncode != 0:
                print(f"⚠️ yt-dlp failed for video {video_id} at {timestamp_seconds}s: {result.stderr[:500]}")
                return False
            
            # Find the actual downloaded file (yt-dlp adds extension)
            actual_video_path = None
            for ext in ['mp4', 'webm', 'mkv']:
                candidate = temp_video_path.replace('%(ext)s', ext)
                if os.path.exists(candidate):
                    actual_video_path = candidate
                    break
            
            if not actual_video_path:
                # Try to find any file starting with the temp name
                temp_dir = os.path.dirname(temp_video_path)
                base_name = os.path.basename(temp_video_path).replace('%(ext)s', '')
                for file in os.listdir(temp_dir):
                    if file.startswith(base_name.split('.')[0]):
                        actual_video_path = os.path.join(temp_dir, file)
                        break
            
            if not actual_video_path or not os.path.exists(actual_video_path):
                print(f"⚠️  Could not find downloaded video file for {video_id}")
                return False
            
            # Calculate relative timestamp if we downloaded a segment
            if '--download-sections' in str(yt_dlp_cmd):
                relative_timestamp = timestamp_seconds - segment_start
            else:
                relative_timestamp = timestamp_seconds
            
            # Extract frame using ffmpeg
            ffmpeg_cmd = [
                "ffmpeg",
                "-i", actual_video_path,
                "-ss", str(relative_timestamp),
                "-vframes", "1",
                "-q:v", "2",  # High quality JPEG (2 is very high quality, lower number = higher quality)
                "-y",  # Overwrite output file
                output_path
            ]
            
            result = subprocess.run(
                ffmpeg_cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode != 0:
                print(f"⚠️ ffmpeg failed to extract frame: {result.stderr[:500]}")
                return False
            
            return os.path.exists(output_path) and os.path.getsize(output_path) > 0
            
        finally:
            # Clean up temp video file(s)
            temp_dir = os.path.dirname(temp_video_path) if '%' not in temp_video_path else tempfile.gettempdir()
            base_name = os.path.basename(temp_video_path).replace('%(ext)s', '').split('.')[0]
            for file in os.listdir(temp_dir):
                if file.startswith(base_name) and (file.endswith('.mp4') or file.endswith('.webm') or file.endswith('.mkv')):
                    try:
                        os.remove(os.path.join(temp_dir, file))
                    except:
                        pass
        
    except subprocess.TimeoutExpired:
        print(f"⚠️ Timeout extracting frame from video {video_id} at {timestamp_seconds}s")
        return False
    except Exception as e:
        print(f"⚠️ Error extracting frame from video {video_id} at {timestamp_seconds}s: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_file_exists_in_drive(drive, folder_id, filename):
    """
    Check if a file with the given filename exists in the specified Drive folder.
    
    :param drive: Google Drive instance
    :param folder_id: Drive folder ID
    :param filename: Name of the file to check
    :return: Drive file object if exists, None otherwise
    """
    try:
        query = f"title='{filename}' and '{folder_id}' in parents and trashed=false"
        file_list = drive.ListFile({'q': query}).GetList()
        
        if file_list:
            return file_list[0]  # Return first matching file
        return None
    except Exception as e:
        print(f"⚠️ Error checking for existing file {filename} in Drive: {e}")
        return None


def upload_image_to_drive(image_path, filename, folder_id, drive):
    """
    Upload an image file to Google Drive folder.
    
    :param image_path: Local path to the image file
    :param filename: Name to use for the file in Drive
    :param folder_id: Drive folder ID to upload to
    :param drive: Google Drive instance
    :return: Shareable Drive URL or None if upload fails
    """
    try:
        # Check if file already exists
        existing_file = check_file_exists_in_drive(drive, folder_id, filename)
        if existing_file:
            file_id = existing_file['id']
            shareable_url = f"https://drive.google.com/file/d/{file_id}/view"
            print(f"✅ File {filename} already exists in Drive, reusing: {shareable_url}")
            return shareable_url
        
        # Upload new file
        file_drive = drive.CreateFile({
            'title': filename,
            'parents': [{'id': folder_id}]
        })
        file_drive.SetContentFile(image_path)
        file_drive.Upload()
        
        # Get shareable URL
        file_id = file_drive['id']
        shareable_url = f"https://drive.google.com/file/d/{file_id}/view"
        
        print(f"✅ Uploaded {filename} to Drive: {shareable_url}")
        return shareable_url
        
    except Exception as e:
        print(f"⚠️ Error uploading {filename} to Drive: {e}")
        return None


def process_video_frames_in_xml(graphics_definition_xml, drive, drive_folder_id=video_frames_drive_folder_id):
    """
    Find all YouTube URLs with only start timestamps (video frames) in the XML,
    extract frames, upload to Drive, and replace URLs in XML.
    
    :param graphics_definition_xml: XML string containing graphics definition
    :param drive: Google Drive instance
    :param drive_folder_id: Drive folder ID to store extracted frames
    :return: Modified XML with Drive URLs instead of YouTube frame URLs, or original XML if processing fails
    """
    if not graphics_definition_xml or not graphics_definition_xml.strip():
        return graphics_definition_xml
    
    if not drive:
        print("⚠️ Drive instance not available, skipping video frame extraction")
        return graphics_definition_xml
    
    try:
        # Find all <asset> tags
        asset_pattern = r'<asset>(.*?)</asset>'
        assets = re.findall(asset_pattern, graphics_definition_xml, re.DOTALL | re.IGNORECASE)
        
        # Dictionary to cache processed frames: {youtube_url: drive_url}
        processed_frames = {}
        
        # Process each asset URL
        for asset_content in assets:
            asset_url = asset_content.strip()
            
            # Check if it's a YouTube URL with only start parameter (no end)
            if 'youtube.com/embed' in asset_url or 'youtube.com/watch' in asset_url:
                # Parse URL to check if it has only start, no end
                parsed_url = urllib.parse.urlparse(asset_url)
                query_params = urllib.parse.parse_qs(parsed_url.query)
                
                has_start = 'start' in query_params or 'start' in asset_url
                has_end = 'end' in query_params or 'end' in asset_url
                
                # If it has start but no end, it's a frame to extract
                if has_start and not has_end:
                    # Check if we've already processed this URL
                    if asset_url in processed_frames:
                        continue
                    
                    # Extract video ID and timestamp
                    video_id = extract_video_id_from_url(asset_url)
                    start_match = re.search(r'[?&]start=(\d+)', asset_url)
                    timestamp = int(start_match.group(1)) if start_match else None
                    
                    if not video_id or timestamp is None:
                        print(f"⚠️ Could not parse video ID or timestamp from URL: {asset_url}")
                        continue
                    
                    # Generate filename
                    filename = f"{video_id}_frame_{timestamp}s.jpg"
                    
                    # Create temp file for extracted frame
                    temp_frame_path = os.path.join(tempfile.gettempdir(), filename)
                    
                    try:
                        print(f"🎬 Extracting frame from video {video_id} at {timestamp}s...")
                        # Extract frame
                        success = extract_frame_from_youtube_video(video_id, timestamp, temp_frame_path)
                        
                        if success and os.path.exists(temp_frame_path):
                            # Upload to Drive
                            drive_url = upload_image_to_drive(
                                temp_frame_path,
                                filename,
                                drive_folder_id,
                                drive
                            )
                            
                            if drive_url:
                                processed_frames[asset_url] = drive_url
                                print(f"✅ Successfully processed frame: {asset_url} -> {drive_url}")
                            else:
                                print(f"⚠️ Failed to upload frame for {asset_url}, keeping original URL")
                        else:
                            print(f"⚠️ Failed to extract frame for {asset_url}, keeping original URL")
                    finally:
                        # Clean up temp frame file
                        if os.path.exists(temp_frame_path):
                            try:
                                os.remove(temp_frame_path)
                            except:
                                pass
        
        # Replace all processed URLs in XML
        if processed_frames:
            modified_xml = graphics_definition_xml
            for youtube_url, drive_url in processed_frames.items():
                # Replace all occurrences of this YouTube URL in the XML
                modified_xml = modified_xml.replace(youtube_url, drive_url)
            
            print(f"✅ Replaced {len(processed_frames)} video frame URLs with Drive URLs")
            return modified_xml
        
        return graphics_definition_xml
        
    except Exception as e:
        print(f"⚠️ Error processing video frames in XML: {e}")
        import traceback
        traceback.print_exc()
        # Return original XML on error
        return graphics_definition_xml


def invoke_gemini_multimodal(parts, llm="gemini_3_flash_thinking", temperature=0.7):
    """
    Invoke the Gemini API with multimodal parts (images, videos, text).

    :param parts: List of Gemini Part objects (text, video, image, etc.)
    :param llm: Model identifier to use
    :param temperature: Temperature setting for generation
    :return: Text response from the model
    """
    # Map model identifier to actual model name
    model_mapping = {
        "gemini_3_flash_thinking": "gemini-3-flash-preview",
        "gemini_3_flash": "gemini-3-flash-preview",
        "gemini_3_pro": "gemini-2-pro",
        "gemini_2_5_flash": "gemini-2.5-flash",
        "gemini-3-flash-preview": "gemini-3-flash-preview",
    }
    actual_model = model_mapping.get(llm, llm)  # Default to llm if not in mapping
    

    client = genai.Client()
    response = client.models.generate_content(
        model=actual_model,
        contents=types.Content(parts=parts),
        config=types.GenerateContentConfig(
            temperature=temperature,
        ),
    )
    try:
        meta = getattr(response, "usage_metadata", None)
        input_tokens = getattr(meta, "prompt_token_count", 0) if meta else 0
        output_tokens = getattr(meta, "candidates_token_count", 0) if meta else 0
        log_token_usage(
            llm=llm,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            log_file="token_usage_log.csv",
        )
    except Exception as e:
        print(f"Token usage logging failed: {e}")
    if hasattr(response, "text") and response.text:
        return response.text
    if getattr(response, "candidates", None):
        first_candidate = response.candidates[0]
        if getattr(first_candidate, "content", None) and first_candidate.content.parts:
            part = first_candidate.content.parts[0]
            if hasattr(part, "text"):
                return part.text
    return str(response)


def parse_segments_from_voiceover(voiceover_text):
    """
    Parse segments from voiceover_segment column (newline-separated).
    
    :param voiceover_text: The voiceover_segment column content
    :return: List of tuples (segment_index, segment_text) where segment_index is 1-based or empty list if no segments found
    """
    if not voiceover_text or voiceover_text.strip() == "" or voiceover_text == "nan":
        return []
    
    segments = voiceover_text.strip().split('\n')
    return [(i + 1, seg.strip()) for i, seg in enumerate(segments) if seg.strip()]


@traceable(
    metadata={
        "agent_name": "graphics_definition_v2",
        "step_name": "Aggregation Agent",
        "function_name": "aggregate_graphics_definition_for_segment",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous")
    }
)
def aggregate_graphics_definition_for_segment(vo_text, slide_title, slide_chunk, image_items, video_urls, video_urls_other_channels, course_name, topic_name, subtopic_name, storyboard, drive, llm="gemini_3_flash_thinking", feedback=None, target_audience=None, failed_visuals=None):
    """
    Aggregate graphics definition for a single segment using images and videos.
    
    :param vo_text: Voiceover text for the segment
    :param slide_title: Slide title
    :param slide_chunk: Full slide content
    :param image_items: List of dicts with 'title' and 'url' keys
    :param video_urls: List of YouTube embed URLs with timestamps (from video_pool)
    :param video_urls_other_channels: List of YouTube watch URLs without timestamps (from video_pool_other_channels)
    :param course_name: Course name
    :param topic_name: Topic name
    :param subtopic_name: Subtopic name
    :param storyboard: Storyboard content from storyboard_planning column
    :param drive: Google Drive instance
    :param llm: Language model to use
    :param feedback: Optional revision feedback to correct previous failures (if provided, uses regeneration prompt)
    :param target_audience: Optional target audience (required when feedback is provided for regeneration)
    :param failed_visuals: Optional list of dicts with 'visual_id' and 'asset_url' keys for failed visuals to load as multimodal inputs
    :return: Tuple of (graphics_definition_xml, evaluation_breakdown) or (None, evaluation_breakdown) if generation fails
    """
    print(f"📝 Generating aggregation for segment: \"{vo_text[:60]}...\"")
    
    # Build image candidates text
    image_candidates_text = ""
    if image_items:
        image_candidates_text = "\n".join([
            f"{idx + 1}. {item.get('title', 'Untitled')} | URL: {item.get('url', '')}"
            for idx, item in enumerate(image_items)
        ])
    else:
        image_candidates_text = "No image candidates provided."
    
    # Build video candidates text with two sections
    video_candidates_text = ""
    
    # First section: video_pool (clips or frames)
    if video_urls:
        video_candidates_text += "Videos from which you can use video clips (with timestamps) or still frames as images\n"
        video_candidates_text += "\n".join([
            f"{idx + 1}. {url}"
            for idx, url in enumerate(video_urls)
        ])
        video_candidates_text += "\n\n"
    
    # Second section: video_pool_other_channels (frames only)
    if video_urls_other_channels:
        video_candidates_text += "Videos from which you can ONLY use still frames as images (NOT playable video clips with timestamps):\n"
        video_candidates_text += "\n".join([
            f"{idx + 1}. {url}"
            for idx, url in enumerate(video_urls_other_channels)
        ])
    
    if not video_candidates_text.strip():
        video_candidates_text = "No video candidates provided."
    
    # Use storyboard if provided, otherwise use empty string
    storyboard_text = storyboard if storyboard and storyboard.strip() and storyboard != "nan" else "No storyboard reference provided."
    
    # Select prompt based on whether feedback is provided (regeneration case)
    if feedback and feedback.strip():
        # Use regeneration-specific prompt
        target_audience_text = target_audience if target_audience else "General audience"
        prompt_text = aggregation_agent_regeneration_prompt.format(
            course_name=course_name,
            target_audience=target_audience_text,
            topic_name=topic_name,
            subtopic_name=subtopic_name,
            vo_text=vo_text,
            slide_title=slide_title,
            slide_chunk=slide_chunk,
            image_candidates=image_candidates_text,
            video_candidates=video_candidates_text,
            feedback=feedback.strip()
        )
    else:
        # Use standard aggregation prompt
        prompt_text = aggregation_agent_prompt.format(
            course_name=course_name,
            topic_name=topic_name,
            subtopic_name=subtopic_name,
            vo_text=vo_text,
            slide_title=slide_title,
            slide_chunk=slide_chunk,
            storyboard=storyboard_text,
            image_candidates=image_candidates_text,
            video_candidates=video_candidates_text
        )
    
    # # Print the formatted prompt
    # prompt_type = "REGENERATION" if feedback and feedback.strip() else "STANDARD"
    # print(f"\n{'='*80}")
    # print(f"📝 FORMATTED AGGREGATION {prompt_type} PROMPT:")
    # print(f"{'='*80}")
    # print(prompt_text)
    # print(f"{'='*80}\n")
    
    # Build multimodal parts: failed visuals (if any) + candidate images + candidate videos + text
    parts: List[types.Part] = []
    
    # Add failed visuals first (for regeneration case)
    if feedback and feedback.strip() and failed_visuals:
        for failed_visual in failed_visuals:
            visual_id = failed_visual.get("visual_id", "")
            asset_url = failed_visual.get("asset_url", "")
            if visual_id and asset_url:
                print(f"    Loading failed visual {visual_id}: {asset_url[:80]}...")
                # Create clear label for the failed visual
                label = types.Part(text=f"FAILED VISUAL - Visual ID: {visual_id} | URL: {asset_url}")
                parts.append(label)
                # Load the actual visual asset
                visual_parts = _build_asset_parts_for_failed_visual(asset_url, drive)
                if visual_parts:
                    parts.extend(visual_parts)
    
    # Add candidate images
    candidate_num = 1
    for idx, item in enumerate(image_items, start=1):
        image_url = item.get("url", "")
        image_title = item.get("title", f"Image {idx}")
        image_id = f"IMG_{candidate_num}"
        
        if image_url:
            # Load image
            pil_image = load_image_from_url(image_url, drive, image_title)
            if pil_image:
                # Convert PIL image to bytes
                buffered = BytesIO()
                pil_image.convert("RGB").save(buffered, format="JPEG")
                image_bytes = buffered.getvalue()
                # Add text label (matching revision prompt format)
                label_text = f"Image Candidate {candidate_num} ({image_id}): {image_title} | URL: {image_url}"
                parts.append(types.Part(text=label_text))
                # Add image part
                parts.append(types.Part(inline_data=types.Blob(mime_type="image/jpeg", data=image_bytes)))
                print(f"✅ Loaded image {idx}: {image_title}")
                candidate_num += 1
            else:
                print(f"⚠️ Failed to load image {idx}: {image_title}")
    
    # Add videos from video_pool (with timestamps) - can be used as clips or frames
    for video_url in video_urls:
        clip_url, start_seconds, end_seconds = parse_video_url_timestamps(video_url)
        if clip_url:
            video_id = f"VID_{candidate_num}"
            # Add text label (matching revision prompt format)
            label_text = f"Video Candidate {candidate_num} ({video_id}): Can be used as video clip (any part of this video with start and end timestamps) OR as still frame (extracted from any point in the video) | URL: {video_url}"
            parts.append(types.Part(text=label_text))
            # Add video part with timestamps
            video_part = build_video_part(clip_url, start_seconds, end_seconds)
            parts.append(video_part)
            print(f"✅ Added video {candidate_num} (with timestamps): start={start_seconds}s, end={end_seconds}s")
            candidate_num += 1
        else:
            print(f"⚠️ Failed to parse video URL: {video_url}")
    
    # Add videos from video_pool_other_channels (without timestamps - full video) - frames only
    for video_url in video_urls_other_channels:
        # Convert watch URL to embed URL
        embed_url = convert_watch_url_to_embed_url(video_url)
        if embed_url:
            video_id = f"VID_{candidate_num}"
            # Add text label (matching revision prompt format)
            label_text = f"Video Candidate {candidate_num} ({video_id}): Can be used ONLY as still frames (extracted from any point in the video) as images (NOT playable video clips with timestamps) | URL: {video_url}"
            parts.append(types.Part(text=label_text))
            # Add video part without timestamps (full video)
            video_part = build_video_part(embed_url, start_seconds=None, end_seconds=None)
            parts.append(video_part)
            print(f"✅ Added video {candidate_num} (full video, no timestamps): {embed_url}")
            candidate_num += 1
        else:
            print(f"⚠️ Failed to convert video URL: {video_url}")
    
    # Add prompt text at the end
    parts.append(types.Part(text=prompt_text))
        
    # Call LLM
    total_videos = len(video_urls) + len(video_urls_other_channels)
    try:
        print(f" 🤖 Calling {llm} with {len(image_items)} images and {total_videos} videos ({len(video_urls)} with timestamps, {len(video_urls_other_channels)} full videos)...")
        response_text = invoke_gemini_multimodal(parts, llm=llm, temperature=0.7)
        
        # Print the full response for debugging
        print(f"\n{'─'*80}")
        print(f"📤 Aggregation Agent Response from LLM for segment: \"{vo_text[:60]}...\"")
        print(f"{'─'*80}")
        print(response_text)
        print(f"{'─'*80}\n")
        
        # Extract <evaluation_breakdown> content
        eval_breakdown_match = re.search(
            r'<evaluation_breakdown>(.*?)</evaluation_breakdown>',
            response_text,
            re.DOTALL | re.IGNORECASE
        )
        
        evaluation_breakdown = ""
        if eval_breakdown_match:
            evaluation_breakdown = eval_breakdown_match.group(1).strip()
            print(f" ✅ Successfully extracted evaluation breakdown")
        else:
            print(f" ⚠️  Could not extract <evaluation_breakdown> from response")
        
        # Check for replacement_visuals (regeneration case) first
        replacement_visuals_match = re.search(
            r'<replacement_visuals>(.*?)</replacement_visuals>',
            response_text,
            re.DOTALL | re.IGNORECASE
        )
        
        if replacement_visuals_match:
            final_graphics_definition = f"<replacement_visuals>\n{replacement_visuals_match.group(1).strip()}\n</replacement_visuals>"
            print(f" ✅ Successfully extracted <replacement_visuals> (includes visual_instruction and selection_justification)")
            return final_graphics_definition, evaluation_breakdown
        
        # Check for final_graphics_definition (standard aggregation case)
        final_graphics_definition_match = re.search(
            r'<final_graphics_definition>(.*?)</final_graphics_definition>',
            response_text,
            re.DOTALL | re.IGNORECASE
        )
        
        if final_graphics_definition_match:
            final_graphics_definition = f"<final_graphics_definition>\n{final_graphics_definition_match.group(1).strip()}\n</final_graphics_definition>"
            print(f" ✅ Successfully extracted <final_graphics_definition>")
            return final_graphics_definition, evaluation_breakdown
        
        # If neither tag found, return None for graphics definition but still return evaluation breakdown
        print(f" ⚠️  Could not extract <replacement_visuals> or <final_graphics_definition> from response")
        return None, evaluation_breakdown
            
    except Exception as e:
        print(f" ❌ Error calling LLM: {e}")
        return None, ""


def format_aggregation_definition_for_sheet(vo_text, graphics_definition_xml, segment_num):
    """
    Format the aggregated graphics definition XML into the readable format for the sheet.
    
    :param vo_text: Voiceover text for the segment
    :param graphics_definition_xml: Graphics definition XML from <replacement_visuals> or <final_graphics_definition>
    :param segment_num: Segment number
    :return: Formatted graphics definition text or empty string if graphics definition XML is empty
    """
    if not graphics_definition_xml or graphics_definition_xml.strip() == "":
        return ""
    
    # Try new format first: <replacement_visuals> with <visual> blocks
    replacement_visuals_match = re.search(r'<replacement_visuals>(.*?)</replacement_visuals>', graphics_definition_xml, re.DOTALL | re.IGNORECASE)
    if replacement_visuals_match:
        # Extract all visual blocks from replacement_visuals
        visual_blocks_pattern = r'<visual>(.*?)</visual>'
        visual_blocks = re.findall(visual_blocks_pattern, replacement_visuals_match.group(1), re.DOTALL | re.IGNORECASE)
        
        if visual_blocks:
            formatted_parts = []
            
            # Segment header
            formatted_parts.append("=" * 80)
            formatted_parts.append(f"SEGMENT {segment_num}")
            formatted_parts.append("=" * 80)
            formatted_parts.append("")  # Empty line after header
            
            # Process each visual block
            for step_idx, visual_xml in enumerate(visual_blocks):
                # Extract components from each visual block
                voiceover_match = re.search(r'<voiceover_part>(.*?)</voiceover_part>', visual_xml, re.DOTALL | re.IGNORECASE)
                instruction_match = re.search(r'<visual_instruction>(.*?)</visual_instruction>', visual_xml, re.DOTALL | re.IGNORECASE)
                asset_match = re.search(r'<replacement_visual_url>(.*?)</replacement_visual_url>', visual_xml, re.DOTALL | re.IGNORECASE)
                justification_match = re.search(r'<selection_justification>(.*?)</selection_justification>', visual_xml, re.DOTALL | re.IGNORECASE)
                
                # Add separator between multiple visual steps (except before the first one)
                if step_idx > 0:
                    formatted_parts.append("----")
                    formatted_parts.append("")
                
                # When VO: (with quotes)
                if voiceover_match:
                    vo_part = voiceover_match.group(1).strip()
                    formatted_parts.append(f'When VO: "{vo_part}"')
                    formatted_parts.append("")  # One line gap
                
                # Visual Instructions:
                if instruction_match:
                    instruction = instruction_match.group(1).strip()
                    formatted_parts.append(f"Visual Instructions: {instruction}")
                    formatted_parts.append("")  # One line gap
                
                # Graphics to use:
                if asset_match:
                    asset_url = asset_match.group(1).strip()
                    formatted_parts.append(f"Graphics to use: {asset_url}")
                    formatted_parts.append("")  # One line gap
                
                # Selection Justification:
                if justification_match:
                    justification = justification_match.group(1).strip()
                    formatted_parts.append(f"Selection Justification: {justification}")
                    formatted_parts.append("")  # One line gap
            
            # Join all parts with newlines
            formatted_text = "\n".join(formatted_parts)
            return formatted_text
    
    # Fallback to old format: <final_graphics_definition> with <visual_step> blocks (for aggregation agent)
    visual_steps_pattern = r'<visual_step>(.*?)</visual_step>'
    visual_steps = re.findall(visual_steps_pattern, graphics_definition_xml, re.DOTALL | re.IGNORECASE)
    
    if not visual_steps:
        return ""
    
    # Build formatted output
    formatted_parts = []
    
    # Segment header
    formatted_parts.append("=" * 80)
    formatted_parts.append(f"SEGMENT {segment_num}")
    formatted_parts.append("=" * 80)
    formatted_parts.append("")  # Empty line after header
    
    # Process each visual step
    for step_idx, step_xml in enumerate(visual_steps):
        # Extract components from each visual step
        voiceover_match = re.search(r'<voiceover_part>(.*?)</voiceover_part>', step_xml, re.DOTALL | re.IGNORECASE)
        instruction_match = re.search(r'<visual_instruction>(.*?)</visual_instruction>', step_xml, re.DOTALL | re.IGNORECASE)
        asset_match = re.search(r'<asset>(.*?)</asset>', step_xml, re.DOTALL | re.IGNORECASE)
        justification_match = re.search(r'<selection_justification>(.*?)</selection_justification>', step_xml, re.DOTALL | re.IGNORECASE)
        
        # Add separator between multiple visual steps (except before the first one)
        if step_idx > 0:
            formatted_parts.append("----")
            formatted_parts.append("")
        
        # When VO: (with quotes)
        if voiceover_match:
            vo_part = voiceover_match.group(1).strip()
            formatted_parts.append(f'When VO: "{vo_part}"')
            formatted_parts.append("")  # One line gap
        
        # Visual Instructions:
        if instruction_match:
            instruction = instruction_match.group(1).strip()
            formatted_parts.append(f"Visual Instructions: {instruction}")
            formatted_parts.append("")  # One line gap
        
        # Graphics to use:
        if asset_match:
            asset_url = asset_match.group(1).strip()
            formatted_parts.append(f"Graphics to use: {asset_url}")
            formatted_parts.append("")  # One line gap
        
        # Selection Justification:
        if justification_match:
            justification = justification_match.group(1).strip()
            formatted_parts.append(f"Selection Justification: {justification}")
            formatted_parts.append("")  # One line gap
    
    # Join all parts with newlines
    formatted_text = "\n".join(formatted_parts)
    
    return formatted_text


@traceable(
    metadata={
        "agent_name": "graphics_definition_v2",
        "step_name": "Aggregation Agent",
        "function_name": "process_aggregation_segment",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous")
    }
)
def process_aggregation_segment(segment_idx, vo_text, slide_title, slide_chunk, drive_results_text, web_results_text, video_pool_text, video_pool_other_channels_text, storyboard_text, course_name, topic_name, subtopic_name, drive, llm="gemini_3_flash_thinking", feedback=None):
    """
    Process a single segment: aggregate graphics definition from images and videos.
    
    :param segment_idx: Segment index (1-based)
    :param vo_text: Voiceover text for the segment
    :param slide_title: Slide title
    :param slide_chunk: Full slide content
    :param drive_results_text: Drive results column content
    :param web_results_text: Web results column content
    :param video_pool_text: Video pool column content (with timestamps)
    :param video_pool_other_channels_text: Video pool other channels column content (without timestamps)
    :param storyboard_text: Storyboard content 
    :param course_name: Course name
    :param topic_name: Topic name
    :param subtopic_name: Subtopic name
    :param drive: Google Drive instance
    :param llm: Language model to use
    :param feedback: Optional revision feedback for this segment
    :return: Tuple of (segment_idx, formatted_segment_text, formatted_eval_breakdown) or (segment_idx, None, formatted_eval_breakdown) if no definition generated
    """
    print(f"\n📦 Processing SEGMENT_{segment_idx}")
    
    # Parse image items from drive_results and web_results for this segment
    drive_image_items = parse_urls_from_results(drive_results_text, segment_idx)
    web_image_items = parse_urls_from_results(web_results_text, segment_idx)
    
    # Combine both sources (drive_results + web_results)
    image_items = drive_image_items + web_image_items
    
    # Parse video items for this segment
    video_urls = parse_urls_from_video_pool(video_pool_text, segment_idx)
    video_urls_other_channels = parse_urls_from_video_pool_other_channels(video_pool_other_channels_text, segment_idx)
    
    print(f" 🖼️  Found {len(image_items)} image candidates ({len(drive_image_items)} from Drive, {len(web_image_items)} from Web)")
    print(f" 🎥 Found {len(video_urls)} video candidates (with timestamps)")
    print(f" 🎬 Found {len(video_urls_other_channels)} video candidates (other channels, full videos)")
    
    if not image_items and not video_urls and not video_urls_other_channels:
        print(f" ⚠️  No image or video candidates available for segment {segment_idx}, skipping")
        return segment_idx, None
    
    # Generate aggregated graphics definition
    graphics_definition_xml, evaluation_breakdown = aggregate_graphics_definition_for_segment(
        vo_text=vo_text,
        slide_title=slide_title,
        slide_chunk=slide_chunk,
        image_items=image_items,
        video_urls=video_urls,
        video_urls_other_channels=video_urls_other_channels,
        course_name=course_name,
        topic_name=topic_name,
        subtopic_name=subtopic_name,
        storyboard=storyboard_text,
        drive=drive,
        llm=llm,
        feedback=feedback,
    )
    
    if graphics_definition_xml:
        # Post-processing: Extract video frames and replace with Drive URLs
        print(f"🔄 Processing video frames in graphics definition...")
        graphics_definition_xml = process_video_frames_in_xml(
            graphics_definition_xml,
            drive,
            video_frames_drive_folder_id
        )
        
        # Format the definition for the sheet
        formatted_segment = format_aggregation_definition_for_sheet(
            vo_text,
            graphics_definition_xml,
            segment_idx
        )
        
        # Format evaluation breakdown for the sheet (with segment marker)
        formatted_eval_breakdown = ""
        if evaluation_breakdown:
            formatted_eval_breakdown = f"---SEGMENT_{segment_idx}---\n{evaluation_breakdown}"
        
        if formatted_segment:
            return segment_idx, formatted_segment, formatted_eval_breakdown
    
    # Return evaluation breakdown even if graphics definition failed
    formatted_eval_breakdown = ""
    if evaluation_breakdown:
        formatted_eval_breakdown = f"---SEGMENT_{segment_idx}---\n{evaluation_breakdown}"
    
    return segment_idx, None, formatted_eval_breakdown


@traceable(
    metadata={
        "agent_name": "graphics_definition_v2",
        "step_name": "Aggregation Agent",
        "function_name": "process_aggregation_row",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous")
    }
)
def process_aggregation_row(index, row, course_name, drive, llm="gemini_3_flash_thinking", max_workers=3):
    """
    Process a single row: aggregate graphics for all segments and combine into final definition.
    
    :param index: Row index
    :param row: Pandas Series with row data
    :param course_name: Course name
    :param drive: Google Drive instance
    :param llm: Language model to use
    :param max_workers: Max parallel workers for segment processing
    :return: Tuple of (index, final_graphics_definition_text, evaluation_breakdown_text) or (index, empty string, empty string) if no segments found
    """
    try:
        voiceover_text = str(row.get("voiceover_segment", "")).strip()
        drive_results_text = str(row.get("drive_results", "")).strip()
        web_results_text = str(row.get("web_results", "")).strip()
        video_pool_text = str(row.get("video_pool", "")).strip()
        video_pool_other_channels_text = str(row.get("video_pool_other_channels", "")).strip()
        storyboard_text = str(row.get("storyboard_planning", "")).strip()
        
        # Skip if voiceover_segment is empty
        if not voiceover_text or voiceover_text == "nan":
            return index, "", ""
        
        # Parse segments
        segments = parse_segments_from_voiceover(voiceover_text)
        if not segments:
            return index, "", ""
        
        # Get row data
        topic_name = str(row.get("Topic", "")).strip()
        subtopic_name = str(row.get("Subtopic", "")).strip()
        slide_title = str(row.get("Slide Chunk Title", "")).strip()
        slide_chunk = str(row.get("Slide Chunk", "")).strip()
        
        print(f"\n{'='*80}")
        print(f"📋 Row {index + 2}: Processing {len(segments)} segment(s)")
        print(f"{'='*80}")
        
        # Process all segments in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            
            # Submit all segments
            futures = {
                executor.submit(
                    process_aggregation_segment,
                    segment_idx,
                    vo_text,
                    slide_title,
                    slide_chunk,
                    drive_results_text,
                    web_results_text,
                    video_pool_text,
                    video_pool_other_channels_text,
                    storyboard_text,
                    course_name,
                    topic_name,
                    subtopic_name,
                    drive,
                    llm
                ): (segment_idx, vo_text)
                for segment_idx, vo_text in segments
            }
            
            # Collect results as they complete
            segment_results = {}
            eval_breakdown_results = {}
            for future in as_completed(futures):
                segment_idx, vo_text = futures[future]
                try:
                    result_idx, formatted_segment, formatted_eval_breakdown = future.result()
                    if formatted_segment:
                        segment_results[result_idx] = formatted_segment
                    if formatted_eval_breakdown:
                        eval_breakdown_results[result_idx] = formatted_eval_breakdown
                except Exception as e:
                    print(f"❌ Error processing segment {segment_idx} (\"{vo_text[:50]}...\"): {e}")
        
        # Combine all segments in order
        final_graphics_definition_text = ""
        if segment_results:
            all_segment_results = [
                segment_results[seg_idx]
                for seg_idx in sorted(segment_results.keys())
            ]
            final_graphics_definition_text = '\n\n'.join(all_segment_results)
        
        # Combine evaluation breakdowns in order
        evaluation_breakdown_text = ""
        if eval_breakdown_results:
            all_eval_breakdowns = [
                eval_breakdown_results[seg_idx]
                for seg_idx in sorted(eval_breakdown_results.keys())
            ]
            evaluation_breakdown_text = '\n\n'.join(all_eval_breakdowns)
        
        return index, final_graphics_definition_text, evaluation_breakdown_text
        
    except Exception as e:
        print(f"Error processing row {index}: {e}")
        return index, "", ""


def validate_final_graphics_definition_row(row):
    """
    Validate that final_graphics_definition matches voiceover_segment:
    - Row is not empty
    - All segments from voiceover_segment have results
    - No gaps in segment numbering (must be sequential starting from 1)
    
    :param row: Pandas Series with row data
    :return: Tuple (is_valid, error_message)
    """
    vo_segments_text = str(row.get("voiceover_segment", "")).strip()
    final_graphics_def_text = str(row.get("final_graphics_definition", "")).strip()
    
    # Skip validation if voiceover_segment is empty
    if not vo_segments_text or vo_segments_text == "nan":
        return True, None
    
    # Count segments in voiceover_segment (split by newline)
    vo_segments = [seg.strip() for seg in vo_segments_text.split('\n') if seg.strip()]
    expected_count = len(vo_segments)
    
    # Skip validation if final_graphics_definition is empty (will be caught by retry logic)
    if not final_graphics_def_text or final_graphics_def_text == "nan" or final_graphics_def_text.strip() == "":
        return False, "final_graphics_definition is empty"
    
    # Skip validation if it's an error marker
    if final_graphics_def_text.startswith("ERROR:"):
        return False, "final_graphics_definition contains error marker"
    
    # Parse segment numbers from final_graphics_definition
    # Format: "SEGMENT X" (not "---SEGMENT_X---")
    segment_pattern = r'SEGMENT\s+(\d+)'
    segment_numbers = [int(match) for match in re.findall(segment_pattern, final_graphics_def_text, re.IGNORECASE)]
    
    if not segment_numbers:
        return False, "No segment markers found in final_graphics_definition"
    
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
        "step_name": "Aggregation Agent",
        "function_name": "run_aggregation_agent_for_all_rows",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous")
    }
)
def run_aggregation_agent_for_all_rows(sheet, llm="gemini_3_flash_thinking", max_workers=5):
    """
    Run aggregation agent for all rows in the Slide Chunks sheet.

    :param sheet: The gspread sheet object.
    :param llm: The language model to use.
    :param max_workers: Number of parallel workers (default 5).
    :return: None
    """
    worksheet_name = "Slide Chunks"
    ws, df = get_sheet_data_and_df(sheet, worksheet_name)
    
    # Get course name from Course info tab
    _, course_info_df = get_sheet_data_and_df(sheet, "Course info")
    course_name = course_info_df.loc[0, "Course Name"]
    
    print(f"\n{'='*80}")
    print(f"🚀 Starting Aggregation Agent")
    print(f"📚 Course: {course_name}")
    print(f"📊 Processing {len(df)} row(s)")
    print(f"{'='*80}\n")
    
    # Get Drive instance
    drive = get_drive_instance()
    if not drive:
        print("❌ Could not initialize Google Drive. Aborting.")
        return
    
    # Ensure final_graphics_definition and evaluation_breakdown columns exist
    if "final_graphics_definition" not in df.columns:
        df["final_graphics_definition"] = ""
    if "evaluation_breakdown" not in df.columns:
        df["evaluation_breakdown"] = ""
    
    # Filter rows that have voiceover_segment but missing final_graphics_definition
    rows_to_process = []
    for index, row in df.iterrows():
        voiceover_segment = str(row.get("voiceover_segment", "")).strip()
        final_graphics_def = str(row.get("final_graphics_definition", "")).strip()
        
        # Skip if no voiceover_segment
        if not voiceover_segment or voiceover_segment == "nan":
            continue
        
        # Skip if final_graphics_definition is already filled
        if final_graphics_def and final_graphics_def != "nan":
            continue
        
        rows_to_process.append((index, row))
    
    if not rows_to_process:
        print("✅ No rows to process. All rows already have final_graphics_definition or missing voiceover_segment.")
        return
    
    print(f"📝 Processing {len(rows_to_process)} row(s) with missing final_graphics_definition\n")
    
    # Initialize progress bar
    total_tasks = len(rows_to_process)
    progress = SmartProgressBar(total_tasks, "Aggregation Agent")
    
    # Process rows in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all rows
        futures = {
            executor.submit(
                process_aggregation_row,
                index,
                row,
                course_name,
                drive,
                llm
            ): index
            for index, row in rows_to_process
        }
        
        # Collect results as they complete
        for future in as_completed(futures):
            index = futures[future]
            try:
                row_index, final_graphics_def_text, evaluation_breakdown_text = future.result()
                
                # Update dataframe
                df.at[row_index, "final_graphics_definition"] = final_graphics_def_text
                df.at[row_index, "evaluation_breakdown"] = evaluation_breakdown_text
                
                # Update progress
                progress.update()
                
                # Save immediately after each row completes
                print(f'Saving row {row_index + 2} to sheet immediately.')
                save_to_sheet(ws, df)
                format_worksheet(ws)
            except Exception as e:
                print(f"Error getting result for row {index}: {e}")
                # Update dataframe with error marker so row is marked as processed
                df.at[index, "final_graphics_definition"] = f"ERROR: {str(e)}"
                df.at[index, "evaluation_breakdown"] = f"ERROR: {str(e)}"
                progress.update()
                # Save immediately even on error
                print(f'Saving row {index + 2} (with error) to sheet immediately.')
                save_to_sheet(ws, df)
                format_worksheet(ws)

    # Save final results before validation
    save_to_sheet(ws, df)
    format_worksheet(ws)

    # Validation and retry logic
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        # Reload dataframe to get latest state
        ws, df = get_sheet_data_and_df(sheet, worksheet_name)
        
        # Validate all rows and find invalid ones
        invalid_rows = []
        for index, row in df.iterrows():
            is_valid, error_msg = validate_final_graphics_definition_row(row)
            if not is_valid:
                invalid_rows.append((index, row, error_msg))
        
        if not invalid_rows:
            break
        
        retry_count += 1
        print(f"\n⚠️ Found {len(invalid_rows)} rows with invalid final_graphics_definition. Retrying (attempt {retry_count}/{max_retries})...")
        for idx, row, error in invalid_rows[:3]:  # Show first 3 errors
            print(f"  Row {idx}: {error}")
        
        # Clear final_graphics_definition and evaluation_breakdown for invalid rows
        for index, row, error_msg in invalid_rows:
            df.at[index, "final_graphics_definition"] = ""
            df.at[index, "evaluation_breakdown"] = ""
        
        # Save cleared state
        save_to_sheet(ws, df)
        format_worksheet(ws)
        
        # Retry processing invalid rows
        futures = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for index, row, error_msg in invalid_rows:
                future = executor.submit(
                    process_aggregation_row,
                    index,
                    row,
                    course_name,
                    drive,
                    llm
                )
                futures[future] = index
            
            # Collect results
            for future in as_completed(futures):
                index = futures[future]
                try:
                    row_index, final_graphics_def_text, evaluation_breakdown_text = future.result()
                    df.at[row_index, "final_graphics_definition"] = final_graphics_def_text
                    df.at[row_index, "evaluation_breakdown"] = evaluation_breakdown_text
                    # Save immediately after each row completes in retry
                    print(f'Saving row {row_index + 2} (retry) to sheet immediately.')
                    save_to_sheet(ws, df)
                    format_worksheet(ws)
                except Exception as e:
                    print(f"Error getting result for row {index} on retry: {e}")
                    df.at[index, "final_graphics_definition"] = f"ERROR: {str(e)}"
                    df.at[index, "evaluation_breakdown"] = f"ERROR: {str(e)}"
                    # Save immediately even on error in retry
                    print(f'Saving row {index + 2} (retry, with error) to sheet immediately.')
                    save_to_sheet(ws, df)
                    format_worksheet(ws)
    
    if retry_count > 0:
        # Check final state
        ws, df = get_sheet_data_and_df(sheet, worksheet_name)
        final_invalid = []
        for index, row in df.iterrows():
            is_valid, error_msg = validate_final_graphics_definition_row(row)
            if not is_valid:
                final_invalid.append((index, error_msg))
        
        if final_invalid:
            print(f"⚠️ After {retry_count} retry attempt(s), {len(final_invalid)} rows still have invalid final_graphics_definition.")
            for idx, error in final_invalid[:5]:  # Show first 5
                print(f"  Row {idx}: {error}")
        else:
            print(f"✅ All rows validated after {retry_count} retry attempt(s).")
    
    # Final save to sheet
    print('All aggregation agent tasks completed. Saving final DataFrame to sheet.')
    save_to_sheet(ws, df)
    format_worksheet(ws)
    print("✅ Aggregation agent complete and saved to sheet.")


def delete_final_graphics_definition(sheet):
    """
    Remove the 'final_graphics_definition' column from the specified worksheet.
    
    :param sheet: The gspread sheet object.
    :return: None
    """
    worksheet_name = "Slide Chunks"
    ws, df = get_sheet_data_and_df(sheet, worksheet_name)
    if "final_graphics_definition" in df.columns:
        df = df.drop(columns=["final_graphics_definition"])
        clear_worksheet(ws)
        save_to_sheet(ws, df)
        print(f"🗑️ Deleted 'final_graphics_definition' column from '{worksheet_name}' worksheet")
    else:
        print(f"ℹ️ 'final_graphics_definition' column does not exist in '{worksheet_name}' worksheet")
