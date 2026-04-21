from langsmith import traceable
import streamlit as st
from services.sheets_service import get_sheet_data_and_df, save_to_sheet, format_worksheet, clear_worksheet
from services.smart_progress_bar import SmartProgressBar
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import requests
import time
from io import BytesIO
from PIL import Image
from agents.vector_store_image_search.graphics_retriever_agent import pil_to_base64_data_uri
from agents.vector_store_image_search.create_vectorstore import download_image_from_drive
from services.helper_functions import build_video_part
from services.llm_service import extract_token_usage, log_token_usage
from dotenv import load_dotenv
import os
import base64
from services.drive_service import login_with_service_account
from agents.graphics_asset_creation.gac_utils import get_or_create_drive_folder
from pydrive2.drive import GoogleDrive
from google import genai
from google.genai import types
from typing import List, Dict, Optional, Tuple, Any
import tempfile
import threading
import subprocess
import urllib.parse
import traceback

from agents.graphics_definition_v2.image_graphics_agent.reference_image_processor import process_reference_image_path

load_dotenv()

# Drive folder ID for storing extracted video frames
video_frames_drive_folder_id = "1sab6wSDPLj54q7KMumGwB1oZHRzXmVf-"


# Prompt to use when we want the visual assingment to be flexible
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
   - The assigned visuals will be displayed on screen as the voiceover text for that sentence is played/narrated.
   - When interpreting any part of the voiceover sentences, you must use the surrounding text in the slide content to understand the context and the instructional intent, before selecting visuals for the respective parts of the slide. Do not interpret a sentence, phrase or a clause literally in isolation. Always derive the meaning of a sentence, phrase or a clause from the surrounding text in the slide content.
   - Select visuals only from the provided image and video candidates.
   - The provided storyboard reference describes the intended visual plan for the slide. 
   - Use the storyboard to understand the visual intent and progression for the voiceover sentence for which you are assembling the graphics definition.
   - Aim to follow the storyboard’s visual idea and sequencing when suitable image or video candidates are available, but do not be too rigid about it.
   - If the available image and video candidates do not fully support any of the storyboard-suggested visual idea(s), adapt by selecting the most instructionally clear and relevant visuals based on the voiceover sentence and the available candidates.

2. Visual Coverage of the Entire Sentence  
   - First, understand the full meaning and instructional intent of the entire voiceover sentence.
   - Identify the key visual idea(s) that must be shown on screen for the sentence to be clearly understood.
   - Select visual(s) so that the chosen visual(s) fully support the complete meaning of the voiceover sentence.
   - Default to simplicity: it is acceptable, and often preferable, for a single visual to support a whole sentence when that visual clearly covers the entire idea. Only split the sentence into multiple distinct visual ideas when the narration clearly calls for a different visual (for example, the subject changes, a new diagram or scene is needed, or the learner must see something fundamentally different). Do not assign separate visuals for every small clause or phrase if you find a single visual that you select clearly supports the entire sentence.

3. Allowed Visual Selection Forms
   - You may select one or more still images from the provided image candidates.
   - You may select one or more segments from the provided video candidates, including short portions of a video clip that are most relevant to the voiceover sentence.
   - You may select a specific still frame from any provided video candidates and use it as a static image.
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
Briefly explain, in your own words, what the voiceover sentence is communicating. Use the full slide context to resolve any references, pronouns, or implied meaning if needed. Also shortly explain the overall instructional intent and meaning of the entire slide.

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
(Exact sentence or the exact phrase or clause from the voiceover sentence that this visual aligns with) 
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
Briefly explain how the selected visual clearly supports this specific part of the voiceover sentence, based on what is visibly shown in the asset. Do not refer to the visual number/index while giving the justification (eg. don't say "Image 9 supports the voiceover sentence because it shows...", instead say "the selected image supports the voiceover sentence because it shows...")
</selection_justification>

</visual_step>

<!-- Repeat <visual_step> if you find multiple distinct visual ideas in the voiceover sentence that require separate visuals -->

</visual_steps>

</final_graphics_definition>

</output>

(Ensure that you strictly follow this exact XML format in your output)
"""


# Prompt to use when we want exactly one visual assignment for each sentence
aggregation_agent_prompt_per_sentence = """You are a senior expert graphics designer specializing in the field of HVAC. Your role is to assemble the final, production-ready graphics definition for a single voiceover sentence so that it can be directly used by a graphics team to design the corresponding visuals for an educational e-learning slide. 
You will be given the voiceover sentence, the full course and slide context for reference, and a set of image and video candidates that have already been identified for this sentence by upstream agents, some of which may be only partially relevant or not ultimately suitable for use. Your task is to decide which SINGLE visual should be selected from the provided image and video candidates to best support the entire voiceover sentence visually. This visual can be an image, a video clip, or a still frame extracted from a video. You will also be given a storyboard reference that describes the intended visual ideas for the slide; use it as guidance, but do not treat it as a strict template—your final decision must be based on what is most clear and accurate given the provided candidates.

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
   - The assigned visual will be displayed on screen as the voiceover text for that sentence is played/narrated.
   - When interpreting the voiceover sentence, you must use the surrounding text in the slide content to understand the context and the instructional intent, before selecting a visual for the sentence. Do not interpret a sentence literally in isolation. Always derive the meaning of a sentence from the surrounding text in the slide content.
   - Select visuals only from the provided image and video candidates.
   - The provided storyboard reference describes the intended visual plan for the slide. 
   - Use the storyboard to understand the visual intent for the voiceover sentence for which you are assembling the graphics definition.
   - Aim to follow the storyboard's visual idea when suitable image or video candidates are available, but do not be too rigid about it.
   - If the available image and video candidates do not fully support a storyboard-suggested visual idea, adapt by selecting the most instructionally clear and relevant visual based on the voiceover sentence and the available candidates.

2. Visual Coverage of the Entire Sentence  
   - First, understand the full meaning and instructional intent of the entire voiceover sentence.
   - Identify the key visual idea that must be shown on screen for the sentence to be clearly understood.
   - IMPORTANT: You must strictly assign only 1 visual for this voiceover sentence from the provided image and video candidates. This single visual must fully support the complete meaning of the entire voiceover sentence.

3. Allowed Visual Selection Forms
   - You may select ONE still image from the provided image candidates for this voiceover sentence.
   - You may select ONE segment from the provided video candidates for this voiceover sentence.
   - You may select ONE specific still frame from any provided video candidates for this voiceover sentence and use it as a static image.
   - IMPORTANT: You must select exactly ONE visual (either one image, one video clip, or one still frame from a video). Do not select multiple visuals or combinations.

4. Source-Specific Video Usage Constraints
   - Video candidates are divided into two distinct groups based on their source.
      - Videos listed under:
        "Videos from which you can use video clips (with timestamps) or still frames as images" may be used in any of the following ways:
        - short video clip with start and end timestamps
        - still frame extracted from the video

      - Video candidates listed under:
        "Videos from which you can ONLY use still frames as images (NOT playable video clips with timestamps)" have the following strict constraints:
        - You MUST NOT select them as playable video clips
        - You MUST NOT assign start–end timestamps
        - You MAY ONLY extract still frame from the video and use it as a static image
        - When using these videos, the asset MUST be represented as a video URL with a single start timestamp only

5. Time-Constrained Visual Design
   - If you find a video clip that is the best visual for this voiceover sentence from the provided video candidates, determine exactly which portion of the video is visually relevant to help assign correct start and end timestamps for the video clip that you will select.
     - The visual is displayed only during the narration of the voiceover sentence.
     - Select a visual that can clearly support the entire sentence within this limited time.
     - Do not select a long video clip that cannot be realistically shown during the narration of the given sentence.

6. Alignment of Visual to the Voiceover Sentence
   - The selected visual should appear for the entire voiceover sentence during narration.
   - The visual must support the complete meaning of the sentence as it is spoken.
   - Since you are selecting only one visual for the entire sentence, it must be comprehensive enough to support the entire sentence meaning.

7. Instructional Clarity Priority
   - Prioritize instructional clarity and accuracy over visual richness or strict adherence to the storyboard.
   - When you find both a video clip and a still image that are equally clear, directly relevant, and instructionally effective for the voiceover sentence, prefer using the video clip, since motion can add useful context. This is a guiding preference, not a strict rule - do not prioritize a video clip over an image if the video is only partially relevant, loosely related, or less effective than the still image at supporting the narration.
   - Always remember that the storyboard is a reference and not a strict template regarding the kind of visual that should be used for the voiceover sentence. Try your best to follow the storyboard's visual idea, but do not be too rigid about it.

Strictly provide your output in the following format:

<output>

<evaluation_breakdown>

This section is your reasoning scratchpad used to analyze the voiceover sentence and the available visual candidates before producing the final graphics definition output. Use it to document your observations, reasoning, and decision process. Provide the following sections:

- Voiceover sentence understanding: 
Briefly explain, in your own words, what the voiceover sentence is communicating. Use the full slide context to resolve any references, pronouns, or implied meaning if needed.

- Storyboard reference scan: 
Explain the storyboard's intended visual idea that corresponds to this voiceover sentence. Summarize what the storyboard is trying to show for this sentence in 1–3 concise bullet points.

- Image candidates scan:
Create a numbered list of all provided image candidates and briefly describe what you see in each of the image candidate.

- Video candidate scan: 
Create a numbered list of all provided video candidates and briefly describe what you see in each of the video candidate. Focus on explaining the visual content of the video, and not what is being spoken in the video. Split the list into two sections: one for videos from which video clips (with timestamps) or still frames as images can be used, and one for videos from which ONLY still frames as images can be used (NOT playable video clips with timestamps).

- Detailed overall analysis: 
Use this section to reason through how to select the SINGLE best visual for the given voiceover sentence.

Apply the instruction guidelines to:
- Carefully review each of the provided image and video candidate in detail before making any selection decision.
- For video candidates, pay close attention to the visual content within the video to identify whether a video segment or a specific still frame from the video can be used as a suitable visual for the entire voiceover sentence. Consider whether any visually clear, frame-worthy moments within the videos could be used as static images.
- Carefully review the storyboard reference to understand the intended visual idea for this sentence, then evaluate how well the available image and video candidates can satisfy that intent.
- Decide which SINGLE visual (one image, one video clip, or one still frame from a video) best supports the entire voiceover sentence. While considering the use of any specific video clip, determine exactly which portion of the video is visually relevant to help assign correct start and end timestamps for the video clip that you will select.
- If the storyboard's visual idea is not feasible due to the available image or video candidates, explain how you plan to select the best alternative visual that still supports the voiceover sentence, while maintaining instructional clarity and faithfulness to the meaning of the voiceover sentence.
- Ensure the selected SINGLE visual fully supports the entire meaning of the voiceover sentence.
- Consider the source-specific video usage constraints when selecting video candidates.
- Address any other reasoning considerations needed to arrive at a clear and instructionally useful final decision.

Document your reasoning, tradeoffs, and decision process as you work toward selecting the single best visual. It is ok for this evaluation breakdown section to be quite long to fit all your reasoning and intermediate thinking needed to arrive at the best final decision.

</evaluation_breakdown>

(Based on your above evaluation, provide the final graphics definition for the voiceover sentence in the following format. IMPORTANT: You must provide exactly ONE visual_step.)

<final_graphics_definition>

<voiceover_part>
Provide the entire voiceover sentence here, as this single visual supports the complete sentence
</voiceover_part>

<visual_instruction>
Concise description of what appears on screen for this sentence, using only the selected visual asset.
</visual_instruction>

<asset>
The visual asset selected to use for this sentence, in one of the following forms:
- Image URL (Exact image URL as provided in the image candidates if an image is selected for this voiceover sentence)
- Video URL with start and end timestamps (if a portion of a video clip is selected for this voiceover sentence. Strictly use such example format of video URL with start and end timestamps: e.g. "https://www.youtube.com/embed/dQw4w9WgXc?start=10&end=20")
- Video URL with a single start timestamp (if a still frame extracted from a video clip is selected for this voiceover sentence. Strictly use such example format of video URL with a single start timestamp indicating the frame timestamp: e.g. "https://www.youtube.com/embed/dQw4w9WgXc?start=10")
</asset>

<selection_justification>
Briefly explain how the selected visual clearly supports the entire voiceover sentence, based on what is visibly shown in the asset. Do not refer to the visual number/index while giving the justification (eg. don't say "Image 9 supports the voiceover sentence because it shows...", instead say "the selected image supports the voiceover sentence because it shows..."
</selection_justification>

</final_graphics_definition>

</output>

(Ensure that you strictly follow this exact XML format in your output and provide exactly ONE visual_step)
"""


# Prompt to use when we want only one visual for the entire slide
aggregation_agent_prompt_for_entire_slide = """You are a senior expert graphics designer specializing in the field of HVAC. Your role is to assemble the final, production-ready graphics definition for an entire slide so that it can be directly used by a graphics team to design the corresponding visuals for an educational e-learning slide. 
You will be given the entire slide content, the full course context for reference, and a set of image and video candidates that have already been identified for this slide by upstream agents, some of which may be only partially relevant or not ultimately suitable for use. Your task is to decide which SINGLE visual should be selected from the provided image and video candidates to best support the entire slide visually as it is narrated. This visual can be an image, a video clip, or a still frame extracted from a video. You will also be given a storyboard reference that describes the intended visual idea for the entire slide; use it as guidance, but do not treat it as a strict template—your final decision must be based on what is most clear and accurate given the provided candidates.

These are the inputs:

<course_information>
Course name: {course_name}
Topic name: {topic_name}
Subtopic name: {subtopic_name}
</course_information>

<slide_content_for_which_to_assemble_graphics_definition>
Slide Title: {slide_title}
Slide Content: {slide_chunk}
</slide_content_for_which_to_assemble_graphics_definition>

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
   - Your task is to assemble a final graphics definition for the entire slide.
   - The assigned visual will be displayed on screen as the entire slide content is narrated.
   - When interpreting the slide content, you must understand the full context and instructional intent of the entire slide before selecting a visual.
   - Select visuals only from the provided image and video candidates.
   - The provided storyboard reference describes the intended visual idea for the entire slide. 
   - Use the storyboard to understand the visual intent for the slide for which you are assembling the graphics definition.
   - Aim to follow the storyboard's visual idea when suitable image or video candidates are available, but do not be too rigid about it.
   - If the available image and video candidates do not fully support a storyboard-suggested visual idea, adapt by selecting the most instructionally clear and relevant visual based on the slide content and the available candidates.

2. Visual Coverage of the Entire Slide  
   - First, understand the full meaning and instructional intent of the entire slide content.
   - Identify the key visual idea that must be shown on screen for the slide to be clearly understood as it is narrated.
   - IMPORTANT: You must strictly assign only 1 visual for the entire slide from the provided image and video candidates. This single visual must fully support the complete meaning of the entire slide content.

3. Allowed Visual Selection Forms
   - You may select ONE still image from the provided image candidates for the entire slide.
   - You may select ONE segment from the provided video candidates for the entire slide.
   - You may select ONE specific still frame from any provided video candidates for the entire slide and use it as a static image.
   - IMPORTANT: You must select exactly ONE visual (either one image, one video clip, or one still frame from a video). Do not select multiple visuals or combinations.

4. Source-Specific Video Usage Constraints
   - Video candidates are divided into two distinct groups based on their source.
      - Videos listed under:
        "Videos from which you can use video clips (with timestamps) or still frames as images" may be used in any of the following ways:
        - short video clip with start and end timestamps
        - still frame extracted from the video

      - Video candidates listed under:
        "Videos from which you can ONLY use still frames as images (NOT playable video clips with timestamps)" have the following strict constraints:
        - You MUST NOT select them as playable video clips
        - You MUST NOT assign start–end timestamps
        - You MAY ONLY extract still frame and use them as static image
        - When using these videos, the asset MUST be represented as a video URL with a single start timestamp only

5. Time-Constrained Visual Design
   - If you find a video clip that is the best visual for this entire slide from the provided video candidates, determine exactly which portion of the video is visually relevant to help assign correct start and end timestamps for the video clip that you will select.
     - The visual is displayed only during the narration of the entire slide.
     - Select a visual that can clearly support the entire slide within this limited time.
     - Do not select a long video clip that cannot be realistically shown during the narration of the entire slide.

6. Alignment of Visual to the Entire Slide
   - The selected visual should appear for the entire slide content during narration.
   - The visual must support the complete meaning of the slide as it is spoken.
   - Since you are selecting only one visual for the entire slide, it must be comprehensive enough to support the slide meaning.

7. Instructional Clarity Priority
   - Prioritize instructional clarity and accuracy over visual richness or strict adherence to the storyboard.
   - When you find both a video clip and a still image that are equally clear, directly relevant, and instructionally effective for the entire slide, prefer using the video clip, since motion can add useful context. This is a guiding preference, not a strict rule - do not prioritize a video clip over an image if the video is only partially relevant, loosely related, or less effective than the still image at supporting the narration.
   - Always remember that the storyboard is a reference and not a strict template regarding the kind of visual that should be used for the slide. Try your best to follow the storyboard's visual idea, but do not be too rigid about it.

Strictly provide your output in the following format:

<output>

<evaluation_breakdown>

This section is your reasoning scratchpad used to analyze the slide content and the available visual candidates before producing the final graphics definition output. Use it to document your observations, reasoning, and decision process. Provide the following sections:

- Slide content understanding: 
Briefly explain, in your own words, what the slide content is communicating. 

- Storyboard reference scan: 
Explain the storyboard's intended visual idea for the slide. Summarize what the storyboard is trying to show for this slide in 1–3 concise bullet points.

- Image candidates scan:
Create a numbered list of all provided image candidates and briefly describe what you see in each of the image candidate.

- Video candidate scan: 
Create a numbered list of all provided video candidates and briefly describe what you see in each of the video candidate. Focus on explaining the visual content of the video, and not what is being spoken in the video. Split the list into two sections: one for videos from which video clips (with timestamps) or still frames as images can be used, and one for videos from which ONLY still frames as images can be used (NOT playable video clips with timestamps).

- Detailed overall analysis: 
Use this section to reason through how to select the SINGLE best visual for the slide.

Apply the instruction guidelines to:
- Carefully review each of the provided image and video candidate in detail before making any selection decision.
- For video candidates, pay close attention to the visual content within the video to identify whether a video segment or a specific still frame from the video can be used as a suitable visual for the slide. Consider whether any visually clear, frame-worthy moments within the videos could be used as static images.
- Carefully review the storyboard reference to understand the intended visual idea for the entire slide, then evaluate how well the available image and video candidates can satisfy that intent.
- Decide which SINGLE visual (one image, one video clip, or one still frame from a video) best supports the entire slide. While considering the use of any specific video clip, determine exactly which portion of the video is visually relevant to help assign correct start and end timestamps for the video clip that you will select.
- If the storyboard's visual idea is not feasible due to the available image or video candidates, explain how you plan to select the best alternative visual that still supports the slide, while maintaining instructional clarity and faithfulness to the meaning of the slide content.
- Ensure the selected SINGLE visual fully supports the entire meaning of the slide content.
- Consider the source-specific video usage constraints when selecting video candidates.
- Address any other reasoning considerations needed to arrive at a clear and instructionally useful final decision.

Document your reasoning, tradeoffs, and decision process as you work toward selecting the single best visual. It is ok for this evaluation breakdown section to be quite long to fit all your reasoning and intermediate thinking needed to arrive at the best final decision.

</evaluation_breakdown>

(Based on your above evaluation, provide the final graphics definition for the entire slide in the following format. IMPORTANT: You must provide exactly ONE visual for the entire slide.)

<final_graphics_definition>

<voiceover_part>
Provide the entire slide content here, as this single visual supports the complete slide
</voiceover_part>

<visual_instruction>
Concise description of what appears on screen for this entire slide, using only the selected visual asset.
</visual_instruction>

<asset>
The visual asset selected to use for this entire slide, in one of the following forms:
- Image URL (Exact image URL as provided in the image candidates if an image is selected for this slide)
- Video URL with start and end timestamps (if a portion of a video clip is selected for this slide. Strictly use such example format of video URL with start and end timestamps: e.g. "https://www.youtube.com/embed/dQw4w9WgXc?start=10&end=20")
- Video URL with a single start timestamp (if a still frame extracted from a video clip is selected for this slide. Strictly use such example format of video URL with a single start timestamp indicating the frame timestamp: e.g. "https://www.youtube.com/embed/dQw4w9WgXc?start=10")
</asset>

<selection_justification>
Briefly explain how the selected visual clearly supports the entire slide content, based on what is visibly shown in the asset. Do not refer to the visual number/index while giving the justification (eg. don't say "Image 9 supports the voiceover sentence because it shows...", instead say "the selectedimage supports the voiceover sentence because it shows..."
</selection_justification>

</final_graphics_definition>

</output>

(Ensure that you strictly follow this exact XML format in your output and provide exactly ONE visual for the entire slide)
"""


# Prompt used by the reviser agent during regeneration, used when we have flexible visual assingment strategy
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
   - Your task is to replace visuals ONLY for the specific part(s) of the voiceover segment that are addressed by the feedback (i.e., the failed visuals).
   - Use the slide content and the specific voiceover segment to understand the whole context and the instructional intent.
   - The assigned visuals are displayed on screen as the voiceover text for that segment is narrated.
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

- Voiceover sentence understanding: 
Briefly explain, in your own words, what the voiceover sentence is communicating. Use the full slide context to resolve any references, pronouns, or implied meaning if needed. Also shortly explain the overall instructional intent and meaning of the entire slide.

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


# Prompt used by the reviser agent during regeneration, used when we have 1 visual per sentence visual assignment strategy
aggregation_agent_regeneration_prompt_for_one_visual_per_sentence = """You are a Graphics Definition Regeneration Agent specializing in the field of HVAC. Your task is to regenerate and assemble a complete, production-ready graphics definition for a SINGLE voiceover (VO) sentence when its previously assigned visual has failed review checks, by selecting the most appropriate SINGLE visual from newly generated candidate image and video pools that were specifically searched based on the review feedback.

You will be given the voiceover sentence, the slide and course context for reference, explicit review feedback describing what was wrong with the failed visual and what is required instead, and a pool of newly searched candidate visuals from which to select a SINGLE visual that directly addresses the feedback requirements and fully supports the entire voiceover sentence.

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

This is the voiceover sentence from the slide content for which the regeneration is needed:
<voiceover_segment>
{vo_text}
</voiceover_segment>

This is the explicit review feedback describing what was wrong with the previously failed visuals and what is required instead:
<feedback>
{feedback}
</feedback>

These are the newly searched candidate images and videos from which to select a SINGLE visual that best addresses the feedback requirements:

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
   - Your task is to replace visual for the voiceover sentence when its previously assigned visual has failed review checks.
   - IMPORTANT: You must strictly select only ONE replacement visual for this entire voiceover sentence. This single replacement visual must fully support the complete meaning of the entire voiceover sentence and address the feedback requirements.
   - Use the slide content and the specific voiceover sentence to understand the whole context and the instructional intent.
   - The assigned visual will be displayed on screen as the voiceover text for that sentence is narrated.
   - When interpreting the voiceover sentence, you must use the surrounding text in the slide content to understand the context and the instructional intent, before selecting a visual for the sentence. Do not interpret a sentence literally in isolation. Always derive the meaning of a sentence from the surrounding text in the slide content.
   - The candidate visuals provided were specifically searched based on the feedback requirements, so they should be more targeted to addressing the issues described in the feedback.
   - The feedback explicitly identifies the failed visual and what is needed instead. Your primary responsibility is to select a SINGLE replacement visual that addresses the feedback requirement while supporting the voiceover sentence.
   - Select visuals only from the provided candidate image and video pools.

2. Feedback-Driven Visual Selection
   - Carefully analyze the feedback to understand what was wrong with the previously failed visual and what specific visual requirements must be satisfied.
   - Extract from the feedback: which Visual ID failed, what was wrong with it, and what is needed instead.
   - IMPORTANT: You must select only ONE visual that addresses the feedback requirements. This single visual must support the entire voiceover sentence meaning.
   - Ensure that the selected replacement visual avoids the same issues that caused the previous visual to fail.
   - The single replacement visual you select must be comprehensive enough to support the entire sentence meaning and address the feedback concerns.

4. Candidate Evaluation and Visual Selection
   - Carefully review all provided image and video candidates for this voiceover sentence.
   - Evaluate each candidate against both the feedback requirements and the overall voiceover sentence needs.
   - Select the SINGLE candidate that most directly and clearly satisfies the feedback while remaining aligned with the voiceover sentence's complete instructional intent.
   - If no candidate fully satisfies the feedback requirements, determine the closest acceptable alternative that still addresses the core feedback concerns and supports the entire sentence.
   - Ensure the selected SINGLE visual is instructionally clear and effective for the entire voiceover sentence.

5. Visual Form and Usage Constraints
   - You may select ONE still image, ONE video clip with timestamps, or ONE still frame extracted from a video, using only the provided candidate visuals.
   - IMPORTANT: You must select exactly ONE visual (either one image, one video clip, or one still frame from a video). Do not select multiple visuals or combinations.
   - Choose the visual form (image, video clip, or still frame) that most clearly satisfies the feedback requirements..
   - When you find both a video clip and a still image that equally satisfy the feedback, prefer using the video clip, since motion can add useful context. This is a guiding preference, not a strict rule - do not prioritize a video clip over an image if the video is only partially relevant, loosely related, or less effective than the still image at addressing the feedback.
   - When selecting a video clip, identify the exact portion of the video that visually supports the required detail and assign appropriate start and end timestamps. Strictly use such example format of video URL with start and end timestamps: e.g. "https://www.youtube.com/embed/dQw4w9WgXc?start=10&end=20"
   - When selecting a still frame from a video so that it can be used as a static image, output the video URL with a single start timestamp only (no end timestamp). Strictly use such example format of video URL with a single start timestamp indicating the frame timestamp: e.g. "https://www.youtube.com/embed/dQw4w9WgXc?start=10"
   - Respect source-specific constraints when selecting video candidates from the provided candidate video pools.
     - Video candidates listed under:
        "Videos from which you can use video clips (with timestamps) or still frames as images" may be used in any of the following ways:
        - short video clip with start and end timestamps
        - still frame extracted from the video
     - Video candidates listed under:
        "Videos from which you can ONLY use still frames as images (NOT playable video clips with timestamps)" have the following strict constraints:
        - You MUST NOT select them as playable video clip
        - You MUST NOT assign start–end timestamps
        - You MAY ONLY extract still frame from the video and use it as a static image
        - When using these videos, the asset MUST be represented as a video URL with a single start timestamp only

6. Time-Constrained Visual Design
   - The visual is displayed only during the narration of the voiceover sentence.
   - If selecting a video clip, ensure that the selected portion can realistically fit within the narration timing of the entire voiceover sentence.
   - Select a visual that can clearly support the entire sentence within this limited time.

7. Alignment of Visual to the Voiceover Sentence
   - The selected SINGLE visual will appear for the entire voiceover sentence during narration.
   - The visual must support the complete meaning of the sentence as it is spoken.
   - Since you are selecting only one visual for the entire sentence, it must be comprehensive enough to support the entire sentence meaning and address all feedback requirements.

Output:

Always provide your output strictly in the following format:

<output>

<evaluation_breakdown>

Use this section as a structured reasoning and scratchpad space for you to address the feedback and select the SINGLE best visual for the voiceover sentence. Use it to document your observations, reasoning, and decision process. Provide the following sections:

- Voiceover sentence understanding: 
Briefly explain, in your own words, what the voiceover sentence is communicating. Use the full slide context to resolve any references, pronouns, or implied meaning if needed. Also shortly explain the overall instructional intent and meaning of the entire slide.

2. Feedback Interpretation
   - Carefully analyze the feedback to identify what was wrong with the previously failed visual.
   - Extract from the feedback: which Visual ID failed (e.g., S1V1), what was wrong with it, and what visual is needed instead.
   - Clearly state the specific visual requirements that must be satisfied based on the feedback.

3. Image Candidates Scan
   - Create a numbered list of all provided image candidates and briefly describe what you see in each of the image candidate.

4. Video Candidates Scan
   - Create a numbered list of all provided video candidates and briefly describe what you see in each of the video candidate.
   - Focus on explaining the visual content of the video, and not what is being spoken in the video.
   - Split the list into two sections: one for videos from which video clips (with timestamps) or still frames as images can be used, and one for videos from which ONLY still frames as images can be used (NOT playable video clips with timestamps).

5. Candidate Fit Analysis Against Feedback
   - Compare all the provided candidates against the feedback requirement.
   - Identify which SINGLE candidate most directly satisfies the feedback requirement.
   - Evaluate how well each candidate addresses the feedback issue.
   - If multiple candidates partially satisfy the feedback requirement, reason about which one best addresses the requirement.
   - If no candidate fully satisfies the feedback requirement, determine the closest acceptable alternative and explain why it is acceptable despite not fully matching.

6. Video Timestamp / Frame Selection Thinking (only if selecting video)
   - If selecting a playable video clip that best satisfies the feedback requirements:
     - Determine exactly which portion of the video is visually relevant and will satisfy the feedback.
     - Plan the correct start and end timestamps for that portion, ensuring the selected video clip can realistically fit within the narration timing of the voiceover sentence.
   - If selecting a still frame from a video, determine the moment (single timestamp) that captures the required visual so that it can be used as a static image.

7. Additional Analysis
   - Note any additional observations, thoughts or analysis that can help you arrive at the correct output and decision that addresses the given feedback with a single visual.
   - Consider how the selected SINGLE replacement visual avoids the same issues that caused the previous visual to fail.
   - Confirm that the selected visual supports the entire voiceover sentence meaning and addresses the feedback requirements.

(It is ok for this section to be quite verbose, long and detailed as long as it helps you arrive at the correct output.)

</evaluation_breakdown>

(Based on your above evaluation, provide the replacement visual in the following format)

<replacement_visual>

<visual>

<visual_id>
(Provide the Visual ID of the failed visual that you are replacing, as specified in the feedback, e.g. S1V1)
</visual_id>

<voiceover_part>
(Provide the entire voiceover sentence here for which we are replacing the visual for)
</voiceover_part>

<current_visual_url>
(Provide the URL of the current visual that we are replacing)
</current_visual_url>

<replacement_visual_url>
(Provide the URL of the replacement visual that you are selecting for this entire voiceover sentence, in one of the following forms:
- Image URL (Exact URL as provided in the image candidates if an image is selected for this voiceover sentence)
- Video URL with start and end timestamps (if a portion of a video clip is selected for this voiceover sentence. Strictly use such example format of video URL with start and end timestamps: e.g. "https://www.youtube.com/embed/dQw4w9WgXc?start=10&end=20")
- Video URL with a single start timestamp (if a still frame extracted from a video clip is selected for this voiceover sentence. Strictly use such example format of video URL with a single start timestamp indicating the frame timestamp: e.g. "https://www.youtube.com/embed/dQw4w9WgXc?start=10"))
</replacement_visual_url>

<visual_instruction>
(Provide a concise description of what is visibly shown in the replacement visual)
</visual_instruction>

<selection_justification>
(Provide a concise justification for why this replacement visual is the most appropriate for the entire voiceover sentence and how it addresses the feedback requirements)
</selection_justification>

</visual>

</replacement_visual>

</output>

(Remember to strictly use this exact format for the output with the XML tags and structure, and nothing else.)
"""


# Prompt used by the reviser agent during regeneration, used when we have one visual for the whole slide visual assignment strategy
aggregation_agent_regeneration_prompt_for_one_visual_per_slide = """You are a Graphics Definition Regeneration Agent specializing in the field of HVAC. Your task is to regenerate and assemble a complete, production-ready graphics definition for an entire slide when its previously assigned visual has failed review checks, by selecting the most appropriate SINGLE visual from newly generated candidate image and video pools that were specifically searched based on the review feedback.

You will be given the entire slide content, the course context for reference, explicit review feedback describing what was wrong with the previously assigned visual and what is required instead, and a pool of newly searched candidate visuals from which to select a SINGLE visual that directly addresses the feedback requirements and fully supports the entire slide content.

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

This is the explicit review feedback describing what was wrong with the previously assigned visual that failed the review check and what is required instead:
<feedback>
{feedback}
</feedback>

These are the newly searched candidate images and videos from which to select a SINGLE visual that best addresses the feedback requirements:

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
   - Your task is to replace visual for the entire slide when its previously assigned visual has failed review checks.
   - IMPORTANT: You must strictly select only ONE replacement visual for this entire slide. This single replacement visual must fully support the complete meaning of the entire slide content and address the feedback requirements.
   - Use the slide content to understand the whole context and the instructional intent.
   - The assigned visual will be displayed on screen as the entire slide content is narrated.
   - The candidate visuals provided were specifically searched based on the feedback requirements, so they should be more targeted to addressing the issues described in the feedback.
   - The feedback explicitly identifies the failed visual and what is needed instead. Your primary responsibility is to select a SINGLE replacement visual that addresses the feedback requirement while supporting the main instructional intent of the slide content.
   - Select visuals only from the provided candidate image and video pools.

2. Feedback-Driven Visual Selection
   - Carefully analyze the feedback to understand what was wrong with the previously failed visual and what specific visual requirement must be satisfied.
   - Extract from the feedback: which Visual ID failed, what was wrong with it, and what is needed instead.
   - IMPORTANT: You must select only ONE visual that addresses the feedback requirement. This single visual must support the main instructional intent of the slide content.
   - When selecting the single visual, prioritize candidates that directly address the feedback requirement while also supporting the main instructional intent of the slide content.
   - Ensure that the selected replacement visual avoids the same issues that caused the previous visual to fail.

3. Candidate Evaluation and Visual Selection
   - Carefully review all provided image and video candidates for this entire slide.
   - Evaluate each candidate against both the feedback requirements and the overall slide content needs.
   - Select the SINGLE candidate that most directly and clearly satisfies the feedback.
   - If no candidate fully satisfies the feedback requirement, determine the closest acceptable alternative that still addresses the core feedback concerns.
   - Ensure the selected SINGLE visual is instructionally clear and effective for the main instructional intent of the slide content.

4. Visual Form and Usage Constraints
   - You may select ONE still image, ONE video clip with timestamps, or ONE still frame extracted from a video, using only the provided candidate visuals.
   - IMPORTANT: You must select exactly ONE visual (either one image, one video clip, or one still frame from a video). Do not select multiple visuals or combinations.
   - Choose the visual form (image, video clip, or still frame) that most clearly satisfies the feedback requirements.
   - When you find both a video clip and a still image that equally satisfy the feedback, prefer using the video clip, since motion can add useful context. This is a guiding preference, not a strict rule - do not prioritize a video clip over an image if the video is only partially relevant, loosely related, or less effective than the still image at addressing the feedback.
   - When selecting a video clip, identify the exact portion of the video that visually supports the required detail and assign appropriate start and end timestamps. Strictly use such example format of video URL with start and end timestamps: e.g. "https://www.youtube.com/embed/dQw4w9WgXc?start=10&end=20"
   - When selecting a still frame from a video so that it can be used as a static image, output the video URL with a single start timestamp only (no end timestamp). Strictly use such example format of video URL with a single start timestamp indicating the frame timestamp: e.g. "https://www.youtube.com/embed/dQw4w9WgXc?start=10"
   - Respect source-specific constraints when selecting video candidates from the provided candidate video pools.
     - Video candidates listed under:
        "Videos from which you can use video clips (with timestamps) or still frames as images" may be used in any of the following ways:
        - short video clip with start and end timestamps
        - still frame extracted from the video
     - Video candidates listed under:
        "Videos from which you can ONLY use still frames as images (NOT playable video clips with timestamps)" have the following strict constraints:
        - You MUST NOT select them as playable video clip
        - You MUST NOT assign start–end timestamps
        - You MAY ONLY extract still frame from the video and use it as a static image
        - When using these videos, the asset MUST be represented as a video URL with a single start timestamp only

5. Time-Constrained Visual Design
   - The visual is displayed only during the narration of the entire slide.
   - If selecting a video clip, ensure that the selected portion can realistically fit within the narration timing of the entire slide.
   - Select a visual that can clearly support the entire slide within this limited time of the slide narration.

6. Alignment of Visual to the Entire Slide
   - The selected SINGLE visual will appear for the entire slide content during narration.
   - Since you are selecting only one visual for the entire slide, it must be comprehensive enough to support the main instructional intent of the slide content and address the feedback requirement.

Output:

Always provide your output strictly in the following format:

<output>

<evaluation_breakdown>

Use this section as a structured reasoning and scratchpad space for you to address the feedback and select the SINGLE best visual for the entire slide. Use it to document your observations, reasoning, and decision process. Provide the following sections:

1. Slide content understanding: 
   - Briefly explain, in your own words, what the slide content is communicating.

2. Feedback Interpretation
   - Carefully analyze the feedback to identify what was wrong with the previously failed visual.
   - Extract from the feedback: which Visual ID failed (e.g., S1V1), what was wrong with it, and what visual is needed instead.
   - Clearly state the specific visual requirements that must be satisfied based on the feedback.

3. Image Candidates Scan
   - Create a numbered list of all provided image candidates and briefly describe what you see in each of the image candidate.

4. Video Candidates Scan
   - Create a numbered list of all provided video candidates and briefly describe what you see in each of the video candidate.
   - Focus on explaining the visual content of the video, and not what is being spoken in the video.
   - Split the list into two sections: one for videos from which video clips (with timestamps) or still frames as images can be used, and one for videos from which ONLY still frames as images can be used (NOT playable video clips with timestamps).

5. Candidate Fit Analysis Against Feedback
   - Compare all the provided candidates against the feedback requirement.
   - Identify which SINGLE candidate most directly satisfies the feedback requirement.
   - Evaluate how well each candidate addresses the feedback issue.
   - If multiple candidates partially satisfy the feedback requirement, reason about which one best addresses the requirement.
   - If no candidate fully satisfies the feedback requirement, determine the closest acceptable alternative and explain why it is acceptable despite not fully matching.

6. Video Timestamp / Frame Selection Thinking (only if selecting video)
   - If selecting a playable video clip that best satisfies the feedback requirements:
     - Determine exactly which portion of the video is visually relevant and will satisfy the feedback.
     - Plan the correct start and end timestamps for that portion, ensuring the selected video clip can realistically fit within the narration timing of the entire slide.
   - If selecting a still frame from a video, determine the moment (single timestamp) that captures the required visual so that it can be used as a static image.

7. Additional Analysis
   - Note any additional observations, thoughts or analysis that can help you arrive at the correct output and decision that addresses the given feedback with a single visual.
   - Consider how the selected SINGLE replacement visual avoids the same issues that caused the previous visual to fail.
   - Confirm that the selected visual supports the main instructional intent of the slide content meaning and addresses the feedback requirements.

(It is ok for this section to be quite verbose, long and detailed as long as it helps you arrive at the correct output.)

</evaluation_breakdown>

(Based on your above evaluation, provide the replacement visual in the following format)

<replacement_visual>

<visual>

<visual_id>
(Provide the Visual ID of the failed visual that you are replacing, as specified in the feedback, e.g. S1V1)
</visual_id>

<voiceover_part>
(Provide the entire slide content as it is)
</voiceover_part>

<current_visual_url>
(Provide the URL of the current visual that we are replacing)
</current_visual_url>

<replacement_visual_url>
(Provide the URL of the replacement visual that you are selecting for this entire slide, in one of the following forms:
- Image URL (Exact URL as provided in the image candidates if an image is selected for this slide)
- Video URL with start and end timestamps (if a portion of a video clip is selected for this slide. Strictly use such example format of video URL with start and end timestamps: e.g. "https://www.youtube.com/embed/dQw4w9WgXc?start=10&end=20")
- Video URL with a single start timestamp (if a still frame extracted from a video clip is selected for this slide. Strictly use such example format of video URL with a single start timestamp indicating the frame timestamp: e.g. "https://www.youtube.com/embed/dQw4w9WgXc?start=10"))
</replacement_visual_url>

<visual_instruction>
(Provide a concise description of what is visibly shown in the replacement visual)
</visual_instruction>

<selection_justification>
(Provide a concise justification for why this replacement visual is the most appropriate for the entire slide and how it addresses the feedback requirements)
</selection_justification>

</visual>

</replacement_visual>

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


def parse_urls_from_image_pool(image_pool_text, segment_num):
    """
    Parse image items from image_pool column for a specific segment.
    
    Format: "Title: {title} | URL: {url}"
    
    :param image_pool_text: The image_pool column content
    :param segment_num: Segment number to extract images for
    :return: List of dictionaries with 'title' and 'url' keys or empty list if no images found
    """
    if not image_pool_text or image_pool_text.strip() == "" or image_pool_text == "nan":
        return []
    
    # Find the segment section
    segment_pattern = rf'---SEGMENT_{segment_num}---\s*\n(.*?)(?=\n---SEGMENT_|\Z)'
    match = re.search(segment_pattern, image_pool_text, re.DOTALL)
    
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
    
    urls = []
    
    if segment_num is None:
        # Get all segments - parse the entire text without segment filtering
        # Find all segment sections
        segment_pattern = r'---SEGMENT_\d+---\s*\n(.*?)(?=\n---SEGMENT_|\Z)'
        matches = re.findall(segment_pattern, video_pool_text, re.DOTALL)
        
        # Also check if there's content before the first segment marker
        first_segment_match = re.search(r'^(.*?)(?=\n---SEGMENT_|\Z)', video_pool_text, re.DOTALL)
        if first_segment_match:
            content_before = first_segment_match.group(1).strip()
            if content_before:
                matches.insert(0, content_before)
        
        # Parse all segments
        for segment_content in matches:
            segment_content = segment_content.strip()
            if not segment_content:
                continue
            
            # Each line is a video URL
            for line in segment_content.split('\n'):
                line = line.strip()
                if line and line.startswith('http'):
                    urls.append(line)
    else:
        # Find the segment section
        segment_pattern = rf'---SEGMENT_{segment_num}---\s*\n(.*?)(?=\n---SEGMENT_|\Z)'
        match = re.search(segment_pattern, video_pool_text, re.DOTALL)
        
        if not match:
            return []
        
        segment_content = match.group(1).strip()
        
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


def parse_video_items_from_pool_other_channels(video_pool_other_channels_text, segment_num):
    """
    Parse video items (with type, url, metadata) from video_pool_other_channels for a specific segment.
    Returns the same structure as parse_urls_from_video_pool_filtered for full_video items.

    Format: "Title: {title} | Duration: {duration} | Channel: {channel} | URL: {url}"

    :param video_pool_other_channels_text: The video_pool_other_channels column content
    :param segment_num: Segment number to extract videos for
    :return: List of dicts with 'type' ('full_video'), 'url', and 'metadata' (title, duration, channel)
    """
    if not video_pool_other_channels_text or video_pool_other_channels_text.strip() == "" or video_pool_other_channels_text == "nan":
        return []

    segment_pattern = rf'---SEGMENT_{segment_num}---\s*\n(.*?)(?=\n---SEGMENT_|\Z)'
    match = re.search(segment_pattern, video_pool_other_channels_text, re.DOTALL)
    if not match:
        return []

    segment_content = match.group(1).strip()
    items = []
    for line in segment_content.split('\n'):
        line = line.strip()
        if not line:
            continue
        if " | URL: " not in line:
            if line.startswith('http'):
                items.append({"type": "full_video", "url": line, "metadata": None})
            continue
        parts = line.split(" | URL: ", 1)
        if len(parts) != 2:
            continue
        metadata_part = parts[0].strip()
        url = parts[1].strip()
        if not url or not url.startswith('http'):
            continue
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
        items.append({
            "type": "full_video",
            "url": url,
            "metadata": {"title": title, "duration": duration, "channel": channel},
        })
    return items


def get_video_items_fallback_from_pools(video_pool_text, video_pool_other_channels_text, segment_num):
    """
    Build video items list from video_pool + video_pool_other_channels when video_pool_filtered is empty.
    Returns the same structure as parse_urls_from_video_pool_filtered (list of {type, url, metadata}).

    :param video_pool_text: The video_pool column content (embed URLs per segment)
    :param video_pool_other_channels_text: The video_pool_other_channels column content (metadata + URL lines)
    :param segment_num: Segment number to extract for
    :return: List of dicts with 'type' ('embed' or 'full_video'), 'url', and optional 'metadata'
    """
    items = []
    # Embed items from video_pool (clips or still frames)
    urls = parse_urls_from_video_pool(video_pool_text, segment_num)
    for url in urls:
        if url and url.strip():
            items.append({"type": "embed", "url": url.strip(), "metadata": None})
    # Full-video items from video_pool_other_channels (still frames only)
    other_items = parse_video_items_from_pool_other_channels(video_pool_other_channels_text, segment_num)
    items.extend(other_items)
    return items


def parse_urls_from_video_pool_filtered(video_pool_filtered_text, segment_num):
    """
    Parse video items from video_pool_filtered column for a specific segment.
    
    Handles two formats:
    1. Embed URL with timestamps: "https://www.youtube.com/embed/fd0kGz0XckE?start=120&end=136"
    2. Full video with metadata: "Title: ... | Duration: ... | Channel: ... | URL: https://www.youtube.com/watch?v=..."
    
    :param video_pool_filtered_text: The video_pool_filtered column content
    :param segment_num: Segment number to extract videos for
    :return: List of dictionaries with 'type' ('embed' or 'full_video'), 'url', and optional 'metadata' keys
    """
    if not video_pool_filtered_text or video_pool_filtered_text.strip() == "" or video_pool_filtered_text == "nan":
        return []
    
    # Find the segment section
    segment_pattern = rf'---SEGMENT_{segment_num}---\s*\n(.*?)(?=\n---SEGMENT_|\Z)'
    match = re.search(segment_pattern, video_pool_filtered_text, re.DOTALL)
    
    if not match:
        return []
    
    segment_content = match.group(1).strip()
    items = []
    
    # Parse each line
    for line in segment_content.split('\n'):
        line = line.strip()
        if not line:
            continue
        
        # Check if it's format 2: "Title: ... | Duration: ... | Channel: ... | URL: ..."
        if " | URL: " in line:
            parts = line.split(" | URL: ", 1)
            if len(parts) == 2:
                metadata_part = parts[0].strip()
                url = parts[1].strip()
                
                if url and url.startswith('http'):
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
                    
                    items.append({
                        "type": "full_video",
                        "url": url,
                        "metadata": {
                            "title": title,
                            "duration": duration,
                            "channel": channel
                        }
                    })
        # Check if it's format 1: Embed URL (starts with http and contains 'embed')
        elif line.startswith('http') and 'youtube.com/embed' in line:
            items.append({
                "type": "embed",
                "url": line,
                "metadata": None
            })
        # Fallback: if line is just a URL (treat as embed if it contains embed, otherwise as full video)
        elif line.startswith('http'):
            if 'youtube.com/embed' in line:
                items.append({
                    "type": "embed",
                    "url": line,
                    "metadata": None
                })
            else:
                items.append({
                    "type": "full_video",
                    "url": line,
                    "metadata": None
                })
    
    return items


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


def _parse_final_visual_choice(response_text):
    """Parse the A/B decision from the final comparison response."""
    if not response_text:
        return "A", ""
    chosen_match = re.search(
        r"<chosen_option>\s*([AB])\s*</chosen_option>",
        response_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    reason_match = re.search(
        r"<reason>\s*(.*?)\s*</reason>",
        response_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return (
        (chosen_match.group(1).strip().upper() if chosen_match else "A"),
        (reason_match.group(1).strip() if reason_match else ""),
    )


def _compare_final_visuals(
    gdv2_url,
    reference_url,
    vo_text,
    slide_title,
    slide_chunk,
    course_name,
    topic_name,
    subtopic_name,
    drive,
    llm="gemini_3_flash_thinking",
):
    """Pick the better final image between the GDv2 result and the reference-image pipeline result."""
    from agents.graphics_definition_v2.review_agent.review_and_revise import build_asset_parts

    if not gdv2_url or not reference_url:
        return gdv2_url or reference_url, "", ""
    if gdv2_url.strip() == reference_url.strip():
        return gdv2_url, "A", "Both final image URLs are identical."

    prompt = (
        "You are comparing two final visuals for the same voiceover segment. "
        "Use the actual image/video content, not the URL string, and choose the visual that best supports the narration and slide context.\n\n"
        f"Course name: {course_name or '(none)'}\n"
        f"Topic name: {topic_name or '(none)'}\n"
        f"Subtopic name: {subtopic_name or '(none)'}\n"
        f"Slide title: {slide_title or '(none)'}\n\n"
        f"Voiceover segment: {vo_text or '(none)'}\n"
        f"Slide chunk: {slide_chunk or '(none)'}\n\n"
        f"Option A URL: {gdv2_url}\n"
        f"Option B URL: {reference_url}\n\n"
        "Evaluate these criteria carefully:\n"
        "- Direct relevance to the narration moment\n"
        "- Specificity and instructional clarity\n"
        "- Whether the visual is loadable and visually coherent\n"
        "- Whether it better matches the slide context and avoids distractions\n\n"
        "Provide a brief structured comparison, then choose exactly one option. Return only this format:\n"
        "<decision>\n"
        "<evaluation_breakdown>\n"
        "- Segment Understanding: brief summary of what the narration needs\n"
        "- Option A Scan: what the GDv2 visual shows\n"
        "- Option B Scan: what the reference visual shows\n"
        "- Comparative Analysis: why one is better for this moment\n"
        "- Additional Analysis: any useful edge-case observations\n"
        "</evaluation_breakdown>\n"
        "<chosen_option>A|B</chosen_option>\n"
        "<reason>short reason</reason>\n"
        "</decision>"
    )

    parts = (
        build_asset_parts("A", gdv2_url, drive)
        + build_asset_parts("B", reference_url, drive)
        + [types.Part(text=prompt)]
    )
    response_text = invoke_gemini_multimodal(parts, llm=llm, temperature=0.7)
    chosen_option, reason = _parse_final_visual_choice(response_text)
    selected_url = gdv2_url if chosen_option == "A" else reference_url
    return selected_url, chosen_option, reason


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


def convert_video_url_to_embed(video_url):
    """
    Convert any YouTube URL to embed URL and extract timestamps.
    
    :param video_url: YouTube URL (watch, embed, youtu.be, etc.)
    :return: Tuple of (embed_url, start_seconds, end_seconds)
             Returns (None, None, None) if conversion fails
    """
    if not video_url:
        return None, None, None
    
    # If already an embed URL, parse timestamps from it
    if 'youtube.com/embed' in video_url:
        clip_url, start_seconds, end_seconds = parse_video_url_timestamps(video_url)
        if clip_url:
            # Reconstruct embed URL with timestamps if they exist
            embed_url = clip_url
            if start_seconds is not None:
                embed_url += f"?start={start_seconds}"
                if end_seconds is not None:
                    embed_url += f"&end={end_seconds}"
            return embed_url, start_seconds, end_seconds
        return None, None, None
    
    # Convert watch URL or other formats to embed URL
    video_id = extract_video_id_from_url(video_url)
    if not video_id:
        return None, None, None
    
    # Build embed URL
    embed_url = f"https://www.youtube.com/embed/{video_id}"
    
    # Try to extract timestamps from original URL if present
    start_match = re.search(r'[?&]start=(\d+)', video_url)
    end_match = re.search(r'[?&]end=(\d+)', video_url)
    
    start_seconds = int(start_match.group(1)) if start_match else None
    end_seconds = int(end_match.group(1)) if end_match else None
    
    # Add timestamps to embed URL if they exist
    if start_seconds is not None:
        embed_url += f"?start={start_seconds}"
        if end_seconds is not None:
            embed_url += f"&end={end_seconds}"
    
    return embed_url, start_seconds, end_seconds


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


def format_youtube_timestamp_note(total_seconds: int) -> str:
    """
    Human-readable timestamp for "(use the image at …)" lines, e.g. 70 -> "1m10s".
    """
    if total_seconds < 0:
        total_seconds = 0
    h = total_seconds // 3600
    rem = total_seconds % 3600
    m = rem // 60
    s = rem % 60
    if h > 0:
        return f"{h}h{m}m{s}s"
    if m > 0:
        return f"{m}m{s}s"
    return f"{s}s"


def _youtube_query_start_seconds(query_params) -> Optional[int]:
    """First integer seconds from start= or t= in parse_qs result."""
    if "start" in query_params and query_params["start"]:
        mm = re.search(r"\d+", str(query_params["start"][0]))
        if mm:
            return int(mm.group(0))
    if "t" in query_params and query_params["t"]:
        mm = re.search(r"\d+", str(query_params["t"][0]))
        if mm:
            return int(mm.group(0))
    return None


def _youtube_query_has_end(query_params) -> bool:
    if "end" not in query_params or not query_params["end"]:
        return False
    return bool(re.search(r"\d+", str(query_params["end"][0])))


def try_expand_youtube_single_timestamp_to_one_second_embed(url: str) -> Optional[str]:
    """
    If URL is a YouTube embed/watch/youtu.be link with a single moment (start or t=) and no end=,
    return canonical embed URL with start=n and end=n+1. Otherwise None.
    """
    if not url:
        return None
    raw = url.strip().rstrip(".,);\"'")
    video_id = extract_video_id_from_url(raw)
    if not video_id:
        return None
    parsed = urllib.parse.urlparse(raw)
    host = (parsed.netloc or "").lower()
    path = parsed.path or ""
    qs = urllib.parse.parse_qs(parsed.query)

    if _youtube_query_has_end(qs):
        return None

    start_sec = _youtube_query_start_seconds(qs)
    if start_sec is None:
        return None

    if "youtube.com" in host:
        if "/embed/" in path:
            if "start" not in qs:
                return None
        elif "/watch" in path or path.endswith("/watch"):
            if "start" not in qs and "t" not in qs:
                return None
        else:
            return None
    elif "youtu.be" in host:
        if "start" not in qs and "t" not in qs:
            return None
    else:
        return None

    return f"https://www.youtube.com/embed/{video_id}?start={start_sec}&end={start_sec + 1}"


def parse_youtube_embed_one_second_clip_start(url: str) -> Optional[int]:
    """
    If URL is youtube.com/embed with start=n and end=n+1, return n; else None.
    """
    if not url or "youtube.com/embed" not in url:
        return None
    raw = url.strip().rstrip(".,);\"'")
    parsed = urllib.parse.urlparse(raw)
    if "/embed/" not in (parsed.path or ""):
        return None
    qs = urllib.parse.parse_qs(parsed.query)
    if not _youtube_query_has_end(qs):
        return None
    start_sec = _youtube_query_start_seconds(qs)
    if start_sec is None:
        return None
    mm = re.search(r"\d+", str(qs["end"][0]))
    if not mm:
        return None
    end_sec = int(mm.group(0))
    if end_sec != start_sec + 1:
        return None
    return start_sec


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


def _format_bytes_human_readable(num_bytes: Optional[int]) -> str:
    """
    Convert bytes to a compact human-readable string.
    """
    if num_bytes is None:
        return "unknown"
    try:
        value = float(num_bytes)
    except Exception:
        return str(num_bytes)
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    idx = 0
    while value >= 1024 and idx < len(units) - 1:
        value /= 1024.0
        idx += 1
    return f"{value:.2f} {units[idx]}"


def _build_drive_diagnostics(drive, folder_id: Optional[str] = None) -> str:
    """
    Best-effort Drive diagnostics for failed uploads.
    Includes authenticated account email and storage usage.
    """
    details: List[str] = []
    try:
        about = drive.GetAbout() or {}
        user = about.get("user", {}) if isinstance(about, dict) else {}
        email = user.get("emailAddress")
        if email:
            details.append(f"auth_email={email}")

        quota_total = about.get("quotaBytesTotal")
        quota_used = about.get("quotaBytesUsed")
        total_int = int(quota_total) if quota_total not in (None, "") else None
        used_int = int(quota_used) if quota_used not in (None, "") else None
        free_int = None if total_int is None or used_int is None else max(total_int - used_int, 0)
        pct = None if total_int in (None, 0) or used_int is None else (used_int / total_int) * 100.0

        details.append(f"quota_used={_format_bytes_human_readable(used_int)}")
        details.append(f"quota_total={_format_bytes_human_readable(total_int)}")
        details.append(f"quota_free={_format_bytes_human_readable(free_int)}")
        if pct is not None:
            details.append(f"quota_used_pct={pct:.2f}%")
    except Exception as meta_err:
        details.append(f"drive_about_error={meta_err}")

    if folder_id:
        details.append(f"target_folder_id={folder_id}")
        try:
            folder = drive.CreateFile({"id": folder_id})
            folder.FetchMetadata(fields="title,owners(emailAddress),shared,teamDriveId,driveId")
            folder_title = folder.get("title")
            owners = folder.get("owners") or []
            owner_emails = [
                o.get("emailAddress")
                for o in owners
                if isinstance(o, dict) and o.get("emailAddress")
            ]
            if folder_title:
                details.append(f"target_folder_title={folder_title}")
            if owner_emails:
                details.append(f"target_folder_owners={','.join(owner_emails)}")
            shared_drive_id = folder.get("driveId") or folder.get("teamDriveId")
            if shared_drive_id:
                details.append(f"shared_drive_id={shared_drive_id}")
        except Exception as folder_meta_err:
            details.append(f"folder_meta_error={folder_meta_err}")

    return " | ".join(details)


def upload_image_to_drive(image_path, filename, folder_id, drive):
    """
    Upload an image file to Google Drive folder.

    :param image_path: Local path to the image file
    :param filename: Name to use for the file in Drive
    :param folder_id: Drive folder ID to upload to
    :param drive: Google Drive instance
    :return: (shareable_url, error_message). On success error_message is None; on failure url is None.
    """
    try:
        # Check if file already exists
        existing_file = check_file_exists_in_drive(drive, folder_id, filename)
        if existing_file:
            file_id = existing_file['id']
            shareable_url = f"https://drive.google.com/file/d/{file_id}/view"
            print(f"✅ File {filename} already exists in Drive, reusing: {shareable_url}")
            return shareable_url, None

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
        return shareable_url, None

    except Exception as e:
        print(f"⚠️ Error uploading {filename} to Drive: {e}")
        err = str(e).strip() or repr(e)
        diagnostics = _build_drive_diagnostics(drive, folder_id)
        if "quotaExceeded" in err or "storage quota" in err.lower():
            err = (
                f"{err} — The Google account that owns this Drive folder is out of storage "
                "(or the shared drive quota is exhausted). Free space or upload to a folder "
                f"in an account with available quota. [{diagnostics}]"
            )
        elif diagnostics:
            err = f"{err} [{diagnostics}]"
        return None, err


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
        asset_pattern = r"<asset>(.*?)</asset>"
        assets = re.findall(asset_pattern, graphics_definition_xml, re.DOTALL | re.IGNORECASE)
        
        # Dictionary to cache processed frames: {youtube_url: drive_url}
        processed_frames = {}
        
        # Process each asset URL
        for asset_content in assets:
            asset_url = asset_content.strip()
            
            # Check if it's a YouTube URL with only start parameter (no end)
            if "youtube.com/embed" in asset_url or "youtube.com/watch" in asset_url:
                # Parse URL to check if it has only start, no end
                parsed_url = urllib.parse.urlparse(asset_url)
                query_params = urllib.parse.parse_qs(parsed_url.query)

                has_start = "start" in query_params or "start" in asset_url
                has_end = "end" in query_params or "end" in asset_url

                # If it has start but no end, it's a frame to extract
                if has_start and not has_end:
                    # Check if we've already processed this URL
                    if asset_url in processed_frames:
                        continue
                    
                    # Extract video ID and timestamp
                    video_id = extract_video_id_from_url(asset_url)
                    start_match = re.search(r"[?&]start=(\d+)", asset_url)
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
                            drive_url, upload_err = upload_image_to_drive(
                                temp_frame_path,
                                filename,
                                drive_folder_id,
                                drive,
                            )

                            if drive_url:
                                processed_frames[asset_url] = drive_url
                                print(f"✅ Successfully processed frame: {asset_url} -> {drive_url}")
                            else:
                                detail = f" ({upload_err})" if upload_err else ""
                                print(
                                    f"⚠️ Failed to upload frame for {asset_url}{detail}, keeping original URL"
                                )
                        else:
                            print(f"⚠️ Failed to extract frame for {asset_url}, keeping original URL")
                    finally:
                        # Clean up temp frame file
                        if os.path.exists(temp_frame_path):
                            try:
                                os.remove(temp_frame_path)
                            except Exception:
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
        traceback.print_exc()
        # Return original XML on error
        return graphics_definition_xml


def expand_youtube_single_timestamp_clips_in_xml(graphics_definition_xml):
    """
    Convert YouTube single-timestamp URLs in <asset> (embed/watch/youtu.be with start or t=, no end)
    to 1-second embed clips: start=n&end=n+1.

    Does not require Drive. For extracting still frames to Drive images, use process_video_frames_in_xml.

    :param graphics_definition_xml: XML string containing graphics definition
    :return: Modified XML with expanded URLs, or original XML if nothing changed / on error
    """
    if not graphics_definition_xml or not graphics_definition_xml.strip():
        return graphics_definition_xml

    try:
        asset_pattern = r"<asset>(.*?)</asset>"
        assets = re.findall(asset_pattern, graphics_definition_xml, re.DOTALL | re.IGNORECASE)
        replacements = {}

        for asset_content in assets:
            asset_url = asset_content.strip()
            expanded = try_expand_youtube_single_timestamp_to_one_second_embed(asset_url)
            if expanded and expanded != asset_url:
                replacements[asset_url] = expanded
                print(f"🎬 XML <asset>: expanded single-timestamp URL to 1s clip: {asset_url[:80]}...")

        if replacements:
            modified_xml = graphics_definition_xml
            for old_u, new_u in replacements.items():
                modified_xml = modified_xml.replace(old_u, new_u)
            print(f"✅ Updated {len(replacements)} <asset> YouTube URL(s) to 1-second embed clips (XML)")
            return modified_xml

        return graphics_definition_xml

    except Exception as e:
        print(f"⚠️ Error expanding YouTube single-timestamp URLs in XML: {e}")
        traceback.print_exc()
        return graphics_definition_xml


def normalize_youtube_timestamp_urls(graphics_definition_text):
    """
    Expand YouTube single-timestamp links (watch + t=/start=, embed + start= only, youtu.be + t=)
    to embed URLs with start=n&end=n+1 anywhere they appear in the text.

    :param graphics_definition_text: Text string containing graphics definition
    :return: Modified text with expanded URLs, or original text if no changes needed
    """

    if not graphics_definition_text or not graphics_definition_text.strip():
        return graphics_definition_text

    try:
        url_token = re.compile(r"https?://[^\s]+")
        lines = graphics_definition_text.split("\n")
        out_lines = []
        total = 0

        for line in lines:
            new_line = line
            for m in reversed(list(url_token.finditer(line))):
                url = m.group(0)
                expanded = try_expand_youtube_single_timestamp_to_one_second_embed(url)
                if expanded:
                    new_line = new_line[: m.start()] + expanded + new_line[m.end() :]
                    total += 1
            out_lines.append(new_line)

        normalized_text = "\n".join(out_lines)
        if total:
            print(f"  Normalized {total} YouTube URL(s) to 1-second embed clips (start & end)")
        return normalized_text

    except Exception as e:
        print(f"⚠️ Error normalizing YouTube timestamp URLs: {e}")
        traceback.print_exc()
        return graphics_definition_text


def process_video_frames_in_text_format(graphics_definition_text, drive, drive_folder_id = video_frames_drive_folder_id):
    """
    On each "Graphics to use:" line: expand single-timestamp YouTube URLs to 1s embed clips and add
    "(use the image at …)" on the following line when missing. Also ensures that line exists for
    embed URLs that are already start=n&end=n+1 (e.g. after XML post-process + format).

    drive / drive_folder_id are unused but kept for API compatibility.

    Previous behavior (commented at end of function): extract frame, upload to Drive, replace URL.
    """
    if not graphics_definition_text or not graphics_definition_text.strip():
        return graphics_definition_text

    try:
        graphics_use_pattern = re.compile(
            r'(Graphics to use:\s*(?:\n\s*)?)(https?://[^\s\n]+)',
            re.IGNORECASE | re.MULTILINE,
        )
        text = graphics_definition_text
        matches = list(graphics_use_pattern.finditer(text))
        print(f"Found {len(matches)} 'Graphics to use:' URL(s) in graphics definition (text)")

        for m in reversed(matches):
            prefix, url = m.group(1), m.group(2).strip()
            new_url = url
            start_sec = None

            expanded = try_expand_youtube_single_timestamp_to_one_second_embed(url)
            if expanded:
                new_url = expanded
                start_sec = parse_youtube_embed_one_second_clip_start(expanded)
            else:
                start_sec = parse_youtube_embed_one_second_clip_start(url)
                if start_sec is not None:
                    new_url = url

            if start_sec is None:
                continue

            note_line = f"(use the image at {format_youtube_timestamp_note(start_sec)})"
            end_pos = m.end()
            rest = text[end_pos:]
            rest_nl = rest.lstrip("\n")
            already_note = rest_nl.lower().startswith("(use the image at")

            block = prefix + new_url
            if not already_note:
                block += "\n" + note_line
            text = text[: m.start()] + block + text[end_pos:]

            print(f"🎬 Text: YouTube 1s clip + note for Graphics line (start={start_sec}s)")

        if text != graphics_definition_text:
            print("✅ Updated Graphics to use block(s) with 1-second YouTube embed clips and notes (text)")
        return text

    except Exception as e:
        print(f"⚠️ Error processing YouTube clips in text format (aggregation): {e}")
        traceback.print_exc()
        return graphics_definition_text


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
        file_obj = drive.CreateFile({'id': file_id})
        file_obj.FetchMetadata()
        parents = file_obj.get('parents', [])
        parent_ids = [p.get('id') if isinstance(p, dict) else p for p in parents]
        return folder_id in parent_ids
    except Exception as e:
        print(f"⚠️ Error checking if file {file_id} is in folder {folder_id}: {e}")
        import traceback
        traceback.print_exc()
        return False


def add_snapshot_label_to_drive_links(graphics_definition_text, drive, target_folder_id = video_frames_drive_folder_id):
    """
    Add "(snapshot)" label to Drive links that belong to the target folder.

    :param graphics_definition_text: Text string containing graphics definition
    :param drive: Google Drive instance
    :param target_folder_id: Target folder ID to check (default: video_frames_drive_folder_id)
    :return: Modified text with "(snapshot)" labels added, or original text if no changes
    """
    
    if not graphics_definition_text or not graphics_definition_text.strip():
        return graphics_definition_text

    if not drive:
        print("⚠️ Drive instance not available, skipping snapshot label check (aggregation)")
        return graphics_definition_text

    try:
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
                print(f"Added (snapshot) label to {labels_added} Drive link(s) from target folder (aggregation)")
            else:
                print(f"Found {matches_found} Drive link(s) but none needed (snapshot) labels (aggregation)")

        modified_text = '\n'.join(result_lines)
        return modified_text if modified_text != graphics_definition_text else graphics_definition_text

    except Exception as e:
        print(f"⚠️ Error adding snapshot labels to Drive links (aggregation): {e}")
        import traceback
        traceback.print_exc()
        return graphics_definition_text


def invoke_gemini_multimodal(parts, llm="gemini_3_flash_thinking", temperature=0.7, max_retries= 3):
    """
    Invoke the Gemini API with multimodal parts (images, videos, text).

    :param parts: List of Gemini Part objects (text, video, image, etc.)
    :param llm: Model identifier to use
    :param temperature: Temperature setting for generation
    :param max_retries: Maximum retry attempts for transient/quota errors
    :return: Text response from the model
    """
    # Map model identifier to actual model name
    model_mapping = {
        "gemini_3_flash_thinking": "gemini-3-flash-preview",
        "gemini_3_flash": "gemini-3-flash-preview",
        "gemini_3_pro": "gemini-2-pro",
        "gemini_2_5_flash": "gemini-2.5-flash",
        "gemini_2_5_flash_lite": "gemini-2.5-flash-lite",
        "gemini-3-flash-preview": "gemini-3-flash-preview",
    }
    actual_model = model_mapping.get(llm, llm)  # Default to llm if not in mapping
    

    client = genai.Client()

    # === Thinking controls ===
    # Per Gemini docs: Gemini 2.5 models support thinking via `thinkingBudget`.
    # - thinking_budget = -1 => dynamic thinking (recommended default)
    # - thinking_budget = 0  => disable thinking
    thinking_config = None
    if "gemini-2.5" in str(actual_model):
        try:
            thinking_config = types.ThinkingConfig(thinking_budget=-1)
            print(f"🧠 Thinking enabled for model {actual_model} (dynamic budget)")
        except Exception as tc_err:
            # If the SDK surface changes, fall back to default behavior.
            print(f"⚠️ Could not enable thinking config: {tc_err}")
            thinking_config = None

    # Retry logic for quota/transient errors 
    retries = 0
    while retries < max_retries:
        try:
            response = client.models.generate_content(
                model=actual_model,
                contents=types.Content(role="user", parts=parts),
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    thinking_config=thinking_config,
                ),
            )
            break
        except Exception as e:
            error_str = str(e)

            is_quota_error = (
                "429" in error_str
                or "RESOURCE_EXHAUSTED" in error_str
                or "quota" in error_str.lower()
                or "quotaExceeded" in error_str
            )

            # Only retry quota/transient errors; otherwise raise immediately.
            if not is_quota_error:
                raise

            # Try to extract retry delay (Google GenAI errors sometimes include retryDelay).
            retry_delay = None
            try:
                if hasattr(e, "error") and isinstance(e.error, dict):
                    error_dict = e.error
                    if "details" in error_dict:
                        for detail in error_dict.get("details", []):
                            if isinstance(detail, dict) and "retryDelay" in detail:
                                # Sometimes retryDelay is a string like "32s"
                                raw = detail["retryDelay"]
                                if isinstance(raw, (int, float)):
                                    retry_delay = float(raw)
                                elif isinstance(raw, str):
                                    m = re.search(r"([\d.]+)", raw)
                                    if m:
                                        retry_delay = float(m.group(1))
                                break
            except Exception:
                retry_delay = None

            if retry_delay is None and ("retry in" in error_str.lower() or "retrydelay" in error_str.lower()):
                # Parse common formats: "retry in 32s", `"retryDelay": "32s"`
                m = re.search(r"retry\s+in\s+([\d.]+)\s*s", error_str, re.IGNORECASE)
                if m:
                    retry_delay = float(m.group(1))
                else:
                    m = re.search(r"retryDelay[\"']?\s*[:=]\s*[\"']?([\d.]+)s?", error_str, re.IGNORECASE)
                    if m:
                        retry_delay = float(m.group(1))

            if retry_delay is None:
                # Exponential backoff with a cap (seconds)
                retry_delay = min((2 ** retries) * 5, 60)
            else:
                # Add small buffer to be safe
                retry_delay = retry_delay + 1.0

            if retries < max_retries - 1:
                print(f"  ⚠️  Gemini quota/rate limit hit. Waiting {retry_delay:.1f}s before retry {retries + 1}/{max_retries}...")
                time.sleep(retry_delay)
                retries += 1
                continue
            else:
                print(f"  ❌ Max retries ({max_retries}) exceeded for Gemini quota/rate limit error.")
                raise
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
def aggregate_graphics_definition_for_segment(vo_text, slide_title, slide_chunk, image_items, video_items_filtered, course_name, topic_name, subtopic_name, storyboard, drive, llm="gemini_3_flash_thinking", feedback=None, target_audience=None, failed_visuals=None, visual_assignment_strategy="Flexible, let the agent decide", slide_type=""):
    """
    Aggregate graphics definition for a single segment using images and videos.
    
    :param vo_text: Voiceover text for the segment
    :param slide_title: Slide title
    :param slide_chunk: Full slide content
    :param image_items: List of dicts with 'title' and 'url' keys
    :param video_items_filtered: List of video items from video_pool_filtered, each with 'type' ('embed' or 'full_video'), 'url', and optional 'metadata'
    :param course_name: Course name
    :param topic_name: Topic name
    :param subtopic_name: Subtopic name
    :param storyboard: Storyboard content from storyboard_planning column
    :param drive: Google Drive instance
    :param llm: Language model to use
    :param feedback: Optional revision feedback to correct previous failures (if provided, uses regeneration prompt)
    :param target_audience: Optional target audience (required when feedback is provided for regeneration)
    :param failed_visuals: Optional list of dicts with 'visual_id' and 'asset_url' keys for failed visuals to load as multimodal inputs
    :param visual_assignment_strategy: Visual assignment strategy ("Flexible, let the agent decide", "1 Visual per Sentence", or "1 Visual for the whole Slide")
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
    
    # Build video candidates text with two sections based on video type
    video_candidates_text = ""
    
    # Separate videos by type
    embed_videos = [item for item in video_items_filtered if item.get("type") == "embed"]
    full_video_items = [item for item in video_items_filtered if item.get("type") == "full_video"]
    
    # First section: embed videos (clips or frames)
    if embed_videos:
        video_candidates_text += "Videos from which you can use video clips (with timestamps) or still frames as images\n"
        video_candidates_text += "\n".join([
            f"{idx + 1}. {item['url']}"
            for idx, item in enumerate(embed_videos)
        ])
        video_candidates_text += "\n\n"
    
    # Second section: full videos (frames only)
    if full_video_items:
        video_candidates_text += "Videos from which you can ONLY use still frames as images (NOT playable video clips with timestamps):\n"
        video_candidates_text += "\n".join([
            f"{idx + 1}. {item['url']}"
            for idx, item in enumerate(full_video_items)
        ])
    
    if not video_candidates_text.strip():
        video_candidates_text = "No video candidates provided."
        if str(slide_type).strip().lower() in ("transition", "transition slide"):
            video_candidates_text += " This is a Transition slide; only image candidates are available."
    
    # Use storyboard if provided, otherwise use empty string
    storyboard_text = storyboard if storyboard and storyboard.strip() and storyboard != "nan" else "No storyboard reference provided."
    
    # Select prompt based on whether feedback is provided (regeneration case) or visual_assignment_strategy
    if feedback and feedback.strip():
        # Use regeneration-specific prompt based on visual_assignment_strategy
        target_audience_text = target_audience
        visual_assignment_strategy = str(visual_assignment_strategy).strip()
        
        if visual_assignment_strategy == "1 Visual per Sentence":
            prompt_text = aggregation_agent_regeneration_prompt_for_one_visual_per_sentence.format(
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
            print(f"\n{'='*80}")
            print(f"📝 FORMATTED AGGREGATION REGENERATION PROMPT (1 Visual per Sentence):")
            print(f"{'='*80}")
            print(prompt_text)
            print(f"{'='*80}\n")
        elif visual_assignment_strategy == "1 Visual for the whole Slide":
            prompt_text = aggregation_agent_regeneration_prompt_for_one_visual_per_slide.format(
                course_name=course_name,
                target_audience=target_audience_text,
                topic_name=topic_name,
                subtopic_name=subtopic_name,
                slide_title=slide_title,
                slide_chunk=slide_chunk,
                image_candidates=image_candidates_text,
                video_candidates=video_candidates_text,
                feedback=feedback.strip()
            )
            print(f"\n{'='*80}")
            print(f"📝 FORMATTED AGGREGATION REGENERATION PROMPT (1 Visual for Entire Slide):")
            print(f"{'='*80}")
            print(prompt_text)
            print(f"{'='*80}\n")
        else:
            # Default: Flexible strategy
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
            print(f"\n{'='*80}")
            print(f"📝 FORMATTED AGGREGATION REGENERATION PROMPT (Flexible):")
            print(f"{'='*80}")
            print(prompt_text)
            print(f"{'='*80}\n")
    else:
        # Select prompt based on visual_assignment_strategy
        visual_assignment_strategy = str(visual_assignment_strategy).strip()
        if visual_assignment_strategy == "1 Visual per Sentence":
            # Use per-sentence prompt (1 visual per sentence)
            prompt_text = aggregation_agent_prompt_per_sentence.format(
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
            print(f"\n{'='*80}")
            print(f"📝 FORMATTED AGGREGATION PROMPT (1 Visual per Sentence):")
            print(f"{'='*80}")
            print(prompt_text)
            print(f"{'='*80}\n")
        else:
            # Use flexible prompt (default)
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
            print(f"\n{'='*80}")
            print(f"📝 FORMATTED AGGREGATION PROMPT (Flexible):")
            print(f"{'='*80}")
            print(prompt_text)
            print(f"{'='*80}\n")
    
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
    
    # Add videos from video_pool_filtered - process based on type
    for video_item in video_items_filtered:
        video_type = video_item.get("type")
        video_url = video_item.get("url")
        
        if not video_url:
            continue
        
        video_id = f"VID_{candidate_num}"
        
        if video_type == "embed":
            # Embed URL with timestamps - can be used as clips or frames
            clip_url, start_seconds, end_seconds = parse_video_url_timestamps(video_url)
            if clip_url:
                # Add text label
                label_text = f"Video Candidate {candidate_num} ({video_id}): Can be used as video clip (any part of this video with start and end timestamps) OR as still frame (extracted from any point in the video) | URL: {video_url}"
                parts.append(types.Part(text=label_text))
                # Add video part with timestamps
                video_part = build_video_part(clip_url, start_seconds, end_seconds)
                parts.append(video_part)
                print(f"✅ Added video {candidate_num} (embed with timestamps): start={start_seconds}s, end={end_seconds}s")
                candidate_num += 1
            else:
                print(f"⚠️ Failed to parse embed video URL: {video_url}")
        elif video_type == "full_video":
            # Full video - frames only
            # Convert watch URL to embed URL
            embed_url = convert_watch_url_to_embed_url(video_url)
            if embed_url:
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
    embed_count = len([item for item in video_items_filtered if item.get("type") == "embed"])
    full_video_count = len([item for item in video_items_filtered if item.get("type") == "full_video"])
    total_videos = len(video_items_filtered)
    try:
        print(f" 🤖 Calling {llm} with {len(image_items)} images and {total_videos} videos ({embed_count} with timestamps, {full_video_count} full videos)...")
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
        
        # Check for replacement_visual (singular - used for "1 Visual for the whole Slide" strategy)
        replacement_visual_match = re.search(
            r'<replacement_visual>(.*?)</replacement_visual>',
            response_text,
            re.DOTALL | re.IGNORECASE
        )
        
        if replacement_visual_match:
            # Wrap in replacement_visuals for consistency with downstream processing
            final_graphics_definition = f"<replacement_visuals>\n{replacement_visual_match.group(1).strip()}\n</replacement_visuals>"
            print(f" ✅ Successfully extracted <replacement_visual> (singular) and wrapped in <replacement_visuals>")
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


def aggregate_graphics_definition_for_entire_slide(slide_title, slide_chunk, image_items, video_items_filtered, course_name, topic_name, subtopic_name, storyboard, drive, llm="gemini_3_flash_thinking", slide_type=""):
    """
    Aggregate graphics definition for the entire slide using images and videos.
    
    :param slide_title: Slide title
    :param slide_chunk: Full slide content
    :param image_items: List of dicts with 'title' and 'url' keys
    :param video_items_filtered: List of video items from video_pool_filtered, each with 'type' ('embed' or 'full_video'), 'url', and optional 'metadata'
    :param course_name: Course name
    :param topic_name: Topic name
    :param subtopic_name: Subtopic name
    :param storyboard: Storyboard content from storyboard_planning column
    :param drive: Google Drive instance
    :param llm: Language model to use
    :return: Tuple of (graphics_definition_xml, evaluation_breakdown) or (None, evaluation_breakdown) if generation fails
    """
    print(f"📝 Generating aggregation for entire slide: \"{slide_title}\"")
    
    # Build image candidates text
    image_candidates_text = ""
    if image_items:
        image_candidates_text = "\n".join([
            f"{idx + 1}. {item.get('title', 'Untitled')} | URL: {item.get('url', '')}"
            for idx, item in enumerate(image_items)
        ])
    else:
        image_candidates_text = "No image candidates provided."
    
    # Build video candidates text with two sections based on video type
    video_candidates_text = ""
    
    # Separate videos by type
    embed_videos = [item for item in video_items_filtered if item.get("type") == "embed"]
    full_video_items = [item for item in video_items_filtered if item.get("type") == "full_video"]
    
    # First section: embed videos (clips or frames)
    if embed_videos:
        video_candidates_text += "Videos from which you can use video clips (with timestamps) or still frames as images\n"
        video_candidates_text += "\n".join([
            f"{idx + 1}. {item['url']}"
            for idx, item in enumerate(embed_videos)
        ])
        video_candidates_text += "\n\n"
    
    # Second section: full videos (frames only)
    if full_video_items:
        video_candidates_text += "Videos from which you can ONLY use still frames as images (NOT playable video clips with timestamps):\n"
        video_candidates_text += "\n".join([
            f"{idx + 1}. {item['url']}"
            for idx, item in enumerate(full_video_items)
        ])
    
    if not video_candidates_text.strip():
        video_candidates_text = "No video candidates provided."
        if str(slide_type).strip().lower() in ("transition", "transition slide"):
            video_candidates_text += " This is a Transition slide; only image candidates are available for you to select from"   
    
    # Use storyboard if provided, otherwise use empty string
    storyboard_text = storyboard if storyboard and storyboard.strip() and storyboard != "nan" else "No storyboard reference provided."
    
    # Use entire slide prompt
    prompt_text = aggregation_agent_prompt_for_entire_slide.format(
        course_name=course_name,
        topic_name=topic_name,
        subtopic_name=subtopic_name,
        slide_title=slide_title,
        slide_chunk=slide_chunk,
        storyboard=storyboard_text,
        image_candidates=image_candidates_text,
        video_candidates=video_candidates_text
    )
    
    print(f"\n{'='*80}")
    print(f"📝 FORMATTED AGGREGATION PROMPT (1 Visual for Entire Slide):")
    print(f"{'='*80}")
    print(prompt_text)
    print(f"{'='*80}\n")
    
    # Build multimodal parts: candidate images + candidate videos + text
    parts: List[types.Part] = []
    
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
                # Add text label
                label_text = f"Image Candidate {candidate_num} ({image_id}): {image_title} | URL: {image_url}"
                parts.append(types.Part(text=label_text))
                # Add image part
                parts.append(types.Part(inline_data=types.Blob(
                    mime_type="image/jpeg",
                    data=image_bytes
                )))
                print(f"✅ Added image {candidate_num}: {image_title}")
                candidate_num += 1
            else:
                print(f"⚠️ Failed to load image {image_title}")
    
    # Add videos from video_pool_filtered - process based on type
    for video_item in video_items_filtered:
        video_type = video_item.get("type")
        video_url = video_item.get("url")
        
        if not video_url:
            continue
        
        video_id = f"VID_{candidate_num}"
        
        if video_type == "embed":
            # Embed URL with timestamps - can be used as clips or frames
            clip_url, start_seconds, end_seconds = parse_video_url_timestamps(video_url)
            if clip_url:
                # Add text label
                label_text = f"Video Candidate {candidate_num} ({video_id}): Can be used as video clip (any part of this video with start and end timestamps) OR as still frame (extracted from any point in the video) | URL: {video_url}"
                parts.append(types.Part(text=label_text))
                # Add video part with timestamps
                video_part = build_video_part(clip_url, start_seconds, end_seconds)
                parts.append(video_part)
                print(f"✅ Added video {candidate_num} (embed with timestamps): start={start_seconds}s, end={end_seconds}s")
                candidate_num += 1
            else:
                print(f"⚠️ Failed to parse embed video URL: {video_url}")
        elif video_type == "full_video":
            # Full video - frames only
            # Convert watch URL to embed URL
            embed_url = convert_watch_url_to_embed_url(video_url)
            if embed_url:
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
    embed_count = len([item for item in video_items_filtered if item.get("type") == "embed"])
    full_video_count = len([item for item in video_items_filtered if item.get("type") == "full_video"])
    total_videos = len(video_items_filtered)
    try:
        print(f" 🤖 Calling {llm} with {len(image_items)} images and {total_videos} videos ({embed_count} with timestamps, {full_video_count} full videos)...")
        response_text = invoke_gemini_multimodal(parts, llm=llm, temperature=0.7)
        
        # # Print the full response for debugging
        # print(f"\n{'─'*80}")
        # print(f"📤 Aggregation Agent Response from LLM for entire slide: \"{slide_title}\"")
        # print(f"{'─'*80}")
        # print(response_text)
        # print(f"{'─'*80}\n")
        
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
        
        # Check for final_graphics_definition
        final_graphics_definition_match = re.search(
            r'<final_graphics_definition>(.*?)</final_graphics_definition>',
            response_text,
            re.DOTALL | re.IGNORECASE
        )
        
        if final_graphics_definition_match:
            final_graphics_definition = f"<final_graphics_definition>\n{final_graphics_definition_match.group(1).strip()}\n</final_graphics_definition>"
            print(f" ✅ Successfully extracted <final_graphics_definition>")
            return final_graphics_definition, evaluation_breakdown
        
        # If tag not found, return None for graphics definition but still return evaluation breakdown
        print(f" ⚠️  Could not extract <final_graphics_definition> from response")
        return None, evaluation_breakdown
            
    except Exception as e:
        print(f" ❌ Error calling LLM: {e}")
        return None, ""


def format_aggregation_definition_for_sheet(vo_text, graphics_definition_xml, segment_num, is_entire_slide=False):
    """
    Format the aggregated graphics definition XML into the readable format for the sheet.
    
    :param vo_text: Voiceover text for the segment (or slide content for entire slide)
    :param graphics_definition_xml: Graphics definition XML from <replacement_visuals> or <final_graphics_definition>
    :param segment_num: Segment number (or None for entire slide)
    :param is_entire_slide: Whether this is for entire slide (no segment header)
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
            if segment_num:
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
    
    # Check for <final_graphics_definition> format
    final_graphics_definition_match = re.search(r'<final_graphics_definition>(.*?)</final_graphics_definition>', graphics_definition_xml, re.DOTALL | re.IGNORECASE)
    if not final_graphics_definition_match:
        return ""
    
    final_graphics_content = final_graphics_definition_match.group(1).strip()
    
    # Check if it has <visual_steps> wrapper (flexible case) or direct content (per_sentence/entire_slide case)
    visual_steps_wrapper_match = re.search(r'<visual_steps>(.*?)</visual_steps>', final_graphics_content, re.DOTALL | re.IGNORECASE)
    
    if visual_steps_wrapper_match:
        # Flexible case: has <visual_steps> wrapper with multiple <visual_step> blocks
        visual_steps_pattern = r'<visual_step>(.*?)</visual_step>'
        visual_steps = re.findall(visual_steps_pattern, visual_steps_wrapper_match.group(1), re.DOTALL | re.IGNORECASE)
        
        if not visual_steps:
            return ""
        
        # Build formatted output
        formatted_parts = []
        
        # Segment header (always include if segment_num is provided)
        if segment_num:
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
    else:
        # Per sentence or entire slide case: direct content under <final_graphics_definition> (no visual_steps wrapper)
        # Extract components directly from final_graphics_content
        voiceover_match = re.search(r'<voiceover_part>(.*?)</voiceover_part>', final_graphics_content, re.DOTALL | re.IGNORECASE)
        instruction_match = re.search(r'<visual_instruction>(.*?)</visual_instruction>', final_graphics_content, re.DOTALL | re.IGNORECASE)
        asset_match = re.search(r'<asset>(.*?)</asset>', final_graphics_content, re.DOTALL | re.IGNORECASE)
        justification_match = re.search(r'<selection_justification>(.*?)</selection_justification>', final_graphics_content, re.DOTALL | re.IGNORECASE)
        
        if not (voiceover_match or instruction_match or asset_match):
            return ""
        
        formatted_parts = []
        
        # Segment header
        if segment_num:
            formatted_parts.append("=" * 80)
            formatted_parts.append(f"SEGMENT {segment_num}")
            formatted_parts.append("=" * 80)
            formatted_parts.append("")  # Empty line after header
        
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
def process_aggregation_segment(segment_idx, vo_text, slide_title, slide_chunk, image_pool_text, video_pool_filtered_text, drive_results_text, web_results_text, storyboard_text, course_name, topic_name, subtopic_name, drive, llm="gemini_3_flash_thinking", feedback=None, visual_assignment_strategy="Flexible, let the agent decide", video_pool_text="", video_pool_other_channels_text="", slide_type=""):
    """
    Process a single segment: aggregate graphics definition from images and videos.
    
    :param segment_idx: Segment index (1-based)
    :param vo_text: Voiceover text for the segment
    :param slide_title: Slide title
    :param slide_chunk: Full slide content
    :param image_pool_text: Image pool column content (filtered relevant images)
    :param video_pool_filtered_text: Video pool filtered column content (filtered relevant videos)
    :param drive_results_text: Drive results column content (fallback for images)
    :param web_results_text: Web results column content (fallback for images)
    :param storyboard_text: Storyboard content 
    :param course_name: Course name
    :param topic_name: Topic name
    :param subtopic_name: Subtopic name
    :param drive: Google Drive instance
    :param llm: Language model to use
    :param feedback: Optional revision feedback for this segment
    :param visual_assignment_strategy: Visual assignment strategy ("Flexible, let the agent decide", "1 Visual per Sentence", or "1 Visual for the whole Slide")
    :param video_pool_text: Optional; used as fallback when video_pool_filtered is empty
    :param video_pool_other_channels_text: Optional; used as fallback when video_pool_filtered is empty
    :return: Tuple of (segment_idx, formatted_segment_text, formatted_eval_breakdown) or (segment_idx, None, formatted_eval_breakdown) if no definition generated
    """
    print(f"\n📦 Processing SEGMENT_{segment_idx}")
    
    # Parse image items from image_pool for this segment
    image_items = parse_urls_from_image_pool(image_pool_text, segment_idx)
    used_fallback = False
    
    # Fallback: if no images in image_pool, check drive_results and web_results
    if not image_items:
        print(f" ⚠️  No images found in image_pool for segment {segment_idx}, falling back to drive_results and web_results")
        drive_image_items = parse_urls_from_results(drive_results_text, segment_idx)
        web_image_items = parse_urls_from_results(web_results_text, segment_idx)
        image_items = drive_image_items + web_image_items
        if image_items:
            used_fallback = True
            print(f" ✅ Found {len(image_items)} images from fallback sources ({len(drive_image_items)} from Drive, {len(web_image_items)} from Web)")
        else:
            print(f" ⚠️  No images found in fallback sources either")
    
    # Parse video items from video_pool_filtered for this segment
    video_items_filtered = parse_urls_from_video_pool_filtered(video_pool_filtered_text, segment_idx)
    # Fallback: if video_pool_filtered is empty, use video_pool + video_pool_other_channels combined
    if not video_items_filtered and (video_pool_text or video_pool_other_channels_text):
        video_items_filtered = get_video_items_fallback_from_pools(video_pool_text or "", video_pool_other_channels_text or "", segment_idx)
        if video_items_filtered:
            print(f" ⚠️  video_pool_filtered empty for segment {segment_idx}, using fallback: video_pool + video_pool_other_channels ({len(video_items_filtered)} video(s))")
    
    embed_count = len([item for item in video_items_filtered if item.get("type") == "embed"])
    full_video_count = len([item for item in video_items_filtered if item.get("type") == "full_video"])
    
    if image_items:
        if used_fallback:
            print(f" 🖼️  Found {len(image_items)} image candidates (from fallback: drive_results + web_results)")
        else:
            print(f" 🖼️  Found {len(image_items)} image candidates (from image_pool)")
    else:
        print(f" 🖼️  No image candidates found")
    print(f" 🎥 Found {len(video_items_filtered)} video candidates ({embed_count} embed with timestamps, {full_video_count} full videos)")
    
    if not image_items and not video_items_filtered:
        print(f" ⚠️  No image or video candidates available for segment {segment_idx}, skipping")
        return segment_idx, None
    
    # Generate aggregated graphics definition
    graphics_definition_xml, evaluation_breakdown = aggregate_graphics_definition_for_segment(
        vo_text=vo_text,
        slide_title=slide_title,
        slide_chunk=slide_chunk,
        image_items=image_items,
        video_items_filtered=video_items_filtered,
        course_name=course_name,
        topic_name=topic_name,
        subtopic_name=subtopic_name,
        storyboard=storyboard_text,
        drive=drive,
        llm=llm,
        feedback=feedback,
        visual_assignment_strategy=visual_assignment_strategy,
        slide_type=slide_type,
    )
    
    if graphics_definition_xml:
        # Post-processing: single-timestamp YouTube URLs -> 1s embed clips in <asset>
        print(f"🔄 Expanding single-timestamp YouTube URLs in graphics definition XML...")
        graphics_definition_xml = expand_youtube_single_timestamp_clips_in_xml(graphics_definition_xml)
        
        # Format the definition for the sheet
        formatted_segment = format_aggregation_definition_for_sheet(
            vo_text,
            graphics_definition_xml,
            segment_idx,
            is_entire_slide=False
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

def _inject_reference_link(gdv2_text: str, ref_link) -> str:
    """
    Inject 'Graphics to use (Reference): <url>' after the first 'Graphics to use: ...' 
    block (including any following notes like '(use the image at ...)') 
    found in gdv2_text.

    :param gdv2_text: The normal GDv2 formatted output text.
    :param ref_link: The processed reference image Drive URL, or None if pipeline failed.
    :return: Updated text with the reference key injected.
    """
    ref_value = ref_link if ref_link else "None"
    ref_line = f"Graphics to use (Reference): {ref_value}"

    lines = gdv2_text.split('\n')
    result_lines = []
    injected = False

    i = 0
    while i < len(lines):
        line = lines[i]
        result_lines.append(line)
        
        # Check if this is the start of a main Graphics block (exclude the reference line itself)
        if not injected and line.strip().startswith("Graphics to use:") and "(Reference)" not in line:
            # Look ahead for notes belonging to this block (starting with '(')
            next_idx = i + 1
            while next_idx < len(lines) and lines[next_idx].strip().startswith("("):
                result_lines.append(lines[next_idx])
                next_idx += 1
            
            # Inject the reference line after the block and all its corresponding notes
            result_lines.append(ref_line)
            injected = True
            i = next_idx # Advance to the line after the notes we just consumed
        else:
            i += 1

    if not injected:
        # GDv2 produced no output — append the ref line at the end so the cell
        # is non-empty only if gdv2_text itself is non-empty.
        if gdv2_text.strip():
            result_lines.append(ref_line)
        else:
            # Both GDv2 and ref results are missing — return empty so the outer
            # retry loop picks this row up.
            return ""

    return '\n'.join(result_lines)


@traceable(
    metadata={
        "agent_name": "graphics_definition_v2",
        "step_name": "Aggregation Agent",
        "function_name": "process_aggregation_row",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous")
    }
)
def process_aggregation_row(index, row, course_name, drive, llm="gemini_3_flash_thinking", max_workers=50, ref_drive_client=None, ref_output_folder_id=None, ref_drive_lock=None, df=None, precomputed_ref_link=None):
    """
    Process a single row: aggregate graphics for all segments and combine into final definition.
    
    :param index: Row index
    :param row: Pandas Series with row data
    :param course_name: Course name
    :param drive: Google Drive instance (GenAI)
    :param llm: Language model to use
    :param max_workers: Max parallel workers for segment processing
    :param ref_drive_client: GoogleDrive client for reference pipeline (fallback if no precomputed result)
    :param ref_output_folder_id: Folder ID for reference pipeline output
    :param ref_drive_lock: Lock for reference pipeline thread safety
    :param df: Full dataframe for context
    :param precomputed_ref_link: Pre-computed reference image Drive URL (from early-start thread launched
        right after Step 1). When provided, the inline background thread is skipped entirely.
    :return: Tuple of (index, final_graphics_definition_text, evaluation_breakdown_text, ref_link)
        where ref_link is the processed reference image URL (or None) to store separately.
    """
    try:
        final_graphics_definition_text = ""
        evaluation_breakdown_text = ""
        voiceover_text = str(row.get("voiceover_segment", "")).strip()
        image_pool_text = str(row.get("image_pool", "")).strip()
        video_pool_filtered_text = str(row.get("video_pool_filtered", "")).strip()
        video_pool_text = str(row.get("video_pool", "")).strip()
        video_pool_other_channels_text = str(row.get("video_pool_other_channels", "")).strip()
        drive_results_text = str(row.get("drive_results", "")).strip()
        web_results_text = str(row.get("web_results", "")).strip()
        storyboard_text = str(row.get("storyboard_planning", "")).strip()
        
        # Get Visual Assignment Strategy
        visual_assignment_strategy = str(row.get("Visual Assignment Strategy", "Flexible, let the agent decide")).strip()
        if not visual_assignment_strategy or visual_assignment_strategy == "nan":
            visual_assignment_strategy = "Flexible, let the agent decide"
            
        # Force "1 Visual for the whole Slide" if a Reference Image is present
        # This ensures consistency even if previous steps didn't update the sheet column correctly
        ref_image_url = str(row.get("Reference Image", "")).strip()
        if ref_image_url and ref_image_url.lower() != "nan":
            visual_assignment_strategy = "1 Visual for the whole Slide"
        
        # Get row data
        topic_name = str(row.get("Topic", "")).strip()
        subtopic_name = str(row.get("Subtopic", "")).strip()
        slide_title = str(row.get("Slide Chunk Title", "")).strip()
        slide_chunk = str(row.get("Slide Chunk", "")).strip()
        slide_type = str(row.get("Slide Type", "")).strip()
        if slide_type == "nan":
            slide_type = ""

        # ── REFERENCE PIPELINE: USE PRE-COMPUTED RESULT OR START INLINE THREAD ──
        # Priority:
        #   1. precomputed_ref_link — provided when the caller (run_aggregation_agent_for_all_rows)
        #      pre-launched the thread right after Step 1 so it ran in parallel with Steps 2-4.
        #   2. Inline background thread — fallback when called without a pre-computed result
        #      (e.g. during retry passes or standalone calls).
        _has_ref_image = ref_image_url and ref_image_url.lower() != "nan"
        ref_result_box = {}  # shared container used only for the inline fallback thread
        ref_thread = None

        if _has_ref_image and precomputed_ref_link is not None:
            # Fast path: the early-start thread already finished; use its result directly.
            if re.match(r"^https?://", str(precomputed_ref_link).strip(), flags=re.IGNORECASE):
                ref_result_box["link"] = precomputed_ref_link
            else:
                ref_result_box["link"] = None
            print(f"🔗 Row {index + 1}: Using pre-computed reference link: {precomputed_ref_link or '(none)'}")
        elif _has_ref_image and ref_drive_client and ref_output_folder_id and df is not None:
            # Fallback inline thread (retry path or standalone call)
            _ref_slide_chunk = slide_chunk  # capture for closure
            _ref_idx = index

            def _reference_pipeline_task():
                try:
                    result = process_reference_image_path(
                        df=df,
                        df_idx=_ref_idx,
                        ref_image_url=ref_image_url,
                        drive=ref_drive_client,
                        output_folder_id=ref_output_folder_id,
                        drive_lock=ref_drive_lock,
                        llm=llm,
                        voiceover_override=_ref_slide_chunk,
                    )
                    ref_result_box["result"] = result
                except Exception as _e:
                    print(f"⚠️ Row {_ref_idx + 1}: Reference pipeline thread error: {_e}")
                    ref_result_box["result"] = (False, None, None)

            ref_thread = threading.Thread(target=_reference_pipeline_task, daemon=True)
            ref_thread.start()
            print(f"🔗 Row {index + 1}: Reference pipeline started inline (parallel with GDv2 aggregation).") 

        # Handle "1 Visual for the whole Slide" case differently
        if visual_assignment_strategy == "1 Visual for the whole Slide":
            print(f"\n{'='*80}")
            print(f"📋 Row {index + 2}: Processing entire slide (1 visual for whole slide)")
            print(f"{'='*80}")
            
            # Parse image items from image_pool
            # For "entire slide" case, all candidates are under SEGMENT_1
            image_items = parse_urls_from_image_pool(image_pool_text, segment_num=1)  # Get all images from SEGMENT_1
            
            # Inject reference image as a primary candidate if present
            if ref_image_url and ref_image_url.lower() != "nan":
                if not any(item.get("url") == ref_image_url for item in image_items):
                    image_items.insert(0, {"title": "Reference Image from Storyboard", "url": ref_image_url})
            used_fallback = False
            
            # Fallback: if no images in image_pool, check drive_results and web_results
            if not image_items:
                print(f" ⚠️  No images found in image_pool for entire slide, falling back to drive_results and web_results")
                drive_image_items = parse_urls_from_results(drive_results_text, segment_num=1)
                web_image_items = parse_urls_from_results(web_results_text, segment_num=1)
                image_items = drive_image_items + web_image_items
                if image_items:
                    used_fallback = True
                    print(f" ✅ Found {len(image_items)} images from fallback sources ({len(drive_image_items)} from Drive, {len(web_image_items)} from Web)")
                else:
                    print(f" ⚠️  No images found in fallback sources either")
            
            # Parse video items from video_pool_filtered
            # For "entire slide" case, all candidates are under SEGMENT_1
            video_items_filtered = parse_urls_from_video_pool_filtered(video_pool_filtered_text, segment_num=1)  # Get all videos from SEGMENT_1
            # Fallback: if video_pool_filtered is empty, use video_pool + video_pool_other_channels combined
            if not video_items_filtered and (video_pool_text or video_pool_other_channels_text):
                video_items_filtered = get_video_items_fallback_from_pools(video_pool_text, video_pool_other_channels_text, 1)
                if video_items_filtered:
                    print(f" ⚠️  video_pool_filtered empty for entire slide, using fallback: video_pool + video_pool_other_channels ({len(video_items_filtered)} video(s))")
            
            embed_count = len([item for item in video_items_filtered if item.get("type") == "embed"])
            full_video_count = len([item for item in video_items_filtered if item.get("type") == "full_video"])
            
            if image_items:
                if used_fallback:
                    print(f" 🖼️  Found {len(image_items)} image candidates (from fallback: drive_results + web_results)")
                else:
                    print(f" 🖼️  Found {len(image_items)} image candidates (from image_pool)")
            else:
                print(f" 🖼️  No image candidates found")
            print(f" 🎥 Found {len(video_items_filtered)} video candidates ({embed_count} embed with timestamps, {full_video_count} full videos)")
            
            if not image_items and not video_items_filtered:
                print(f" ⚠️  No image or video candidates available for entire slide, skipping")
                return index, "", ""
            
            # Generate aggregated graphics definition for entire slide
            graphics_definition_xml, evaluation_breakdown = aggregate_graphics_definition_for_entire_slide(
                slide_title=slide_title,
                slide_chunk=slide_chunk,
                image_items=image_items,
                video_items_filtered=video_items_filtered,
                course_name=course_name,
                topic_name=topic_name,
                subtopic_name=subtopic_name,
                storyboard=storyboard_text,
                drive=drive,
                llm=llm,
                slide_type=slide_type,
            )
            
            if graphics_definition_xml:
                # Post-processing: single-timestamp YouTube URLs -> 1s embed clips in <asset>
                print(f"🔄 Expanding single-timestamp YouTube URLs in graphics definition XML...")
                graphics_definition_xml = expand_youtube_single_timestamp_clips_in_xml(graphics_definition_xml)
                
                # Format the definition for the sheet (entire slide, but still include SEGMENT 1 marker)
                formatted_segment = format_aggregation_definition_for_sheet(
                    slide_chunk,  # Use slide content as vo_text
                    graphics_definition_xml,
                    segment_num=1,  # Use segment 1 for entire slide to maintain consistency
                    is_entire_slide=True
                )
                
                # Text-level post-processing to mirror review/revise behavior
                if formatted_segment:
                    try:
                        # 1) Normalize YouTube URLs (t= → start=)
                        normalized = normalize_youtube_timestamp_urls(formatted_segment)
                        # 2) Graphics lines: 1s YouTube embed clips + "(use the image at …)" notes
                        processed = process_video_frames_in_text_format(normalized, drive)
                        # 3) Add "(snapshot)" labels for Drive links from the video frames folder
                        labeled = add_snapshot_label_to_drive_links(processed, drive)
                        formatted_segment = labeled
                    except Exception as e:
                        print(f"⚠️ Error during text-level post-processing for aggregation (entire slide): {e}")
                
                # Format evaluation breakdown for the sheet
                formatted_eval_breakdown = evaluation_breakdown if evaluation_breakdown else ""
                
                if formatted_segment:
                    final_graphics_definition_text = formatted_segment
                    evaluation_breakdown_text = formatted_eval_breakdown
                else:
                    final_graphics_definition_text = ""
                    evaluation_breakdown_text = formatted_eval_breakdown
            else:
                final_graphics_definition_text = ""
                evaluation_breakdown_text = evaluation_breakdown if evaluation_breakdown else ""
        
        else:
            # For "Flexible" and "1 Visual per Sentence" cases, process segments
            # Skip if voiceover_segment is empty
            if not voiceover_text or voiceover_text == "nan":
                return index, "", ""
            
            # Parse segments
            segments = parse_segments_from_voiceover(voiceover_text)
            if not segments:
                return index, "", ""
            
            print(f"\n{'='*80}")
            print(f"📋 Row {index + 2}: Processing {len(segments)} segment(s) (Strategy: {visual_assignment_strategy})")
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
                        image_pool_text,
                        video_pool_filtered_text,
                        drive_results_text,
                        web_results_text,
                        storyboard_text,
                        course_name,
                        topic_name,
                        subtopic_name,
                        drive,
                        llm,
                        None,  # feedback
                        visual_assignment_strategy,
                        video_pool_text,
                        video_pool_other_channels_text,
                        slide_type,
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
                
                try:
                    normalized = normalize_youtube_timestamp_urls(final_graphics_definition_text)
                    processed = process_video_frames_in_text_format(normalized, drive)
                    labeled = add_snapshot_label_to_drive_links(processed, drive)
                    final_graphics_definition_text = labeled
                except Exception as e:
                    print(f"⚠️ Error during text-level post-processing for aggregation (per-segment): {e}")
            
            # Combine evaluation breakdowns in order
            evaluation_breakdown_text = ""
            if eval_breakdown_results:
                all_eval_breakdowns = [
                    eval_breakdown_results[seg_idx]
                    for seg_idx in sorted(eval_breakdown_results.keys())
                ]
                evaluation_breakdown_text = '\n\n'.join(all_eval_breakdowns)

        # ── COLLECT REFERENCE PATH RESULT ────────────────────────────────────────
        # Resolve the processed reference image URL from whichever path ran:
        #   • Pre-computed path  → already in ref_result_box["link"]
        #   • Inline thread path → join thread, parse result into ref_result_box["link"]
        # The URL is NOT injected into final_graphics_definition here; instead it is
        # returned separately so run_aggregation_agent_for_all_rows can store it in
        # the 'reference_image_processed_url' column.  The Decide Final Visuals step
        # will compare the GDv2 image against this reference image and choose the best one.
        resolved_ref_link = ref_result_box.get("link")  # set by pre-computed path
        reference_eligible_for_compare = bool(
            resolved_ref_link and re.match(r"^https?://", str(resolved_ref_link).strip(), flags=re.IGNORECASE)
        )

        if ref_thread is not None:
            ref_thread.join()
            ref_result = ref_result_box.get("result")

            if ref_result is None:
                print(f"❌ [REF-PIPELINE] Row {index + 1}: ref_result is None — thread may have crashed silently.")
            else:
                is_relevant, ref_final_def, _ = ref_result
                print(f"🔍 [REF-PIPELINE] Row {index + 1}: is_relevant={is_relevant}, ref_final_def is {'SET' if ref_final_def else 'EMPTY/NONE'}")
                if ref_final_def:
                    print(f"   ref_final_def preview: {ref_final_def[:200]!r}")

                if is_relevant and ref_final_def:
                    graphics_match = re.search(
                        r'Graphics to use:\s*(https?://[^\s\n]+)', ref_final_def
                    )
                    if graphics_match:
                        resolved_ref_link = graphics_match.group(1).strip()
                        reference_eligible_for_compare = True
                        print(f"✅ [REF-PIPELINE] Row {index + 1}: Parsed ref_link = {resolved_ref_link}")
                    else:
                        print(f"⚠️ [REF-PIPELINE] Row {index + 1}: Could not parse URL from ref_final_def. Full text: {ref_final_def!r}")
                elif not is_relevant:
                    print(f"ℹ️ [REF-PIPELINE] Row {index + 1}: Reference image marked NOT relevant — skipping.")
                    resolved_ref_link = None
                    reference_eligible_for_compare = False

            status = "✅ succeeded" if resolved_ref_link else "⚠️ failed or not relevant"
            print(f"[REF-PIPELINE] Row {index + 1}: {status}. Storing in reference_image_processed_url column.")

        if reference_eligible_for_compare and final_graphics_definition_text:
            gdv2_urls = _extract_graphics_to_use_urls(final_graphics_definition_text)
            gdv2_url = gdv2_urls[0] if gdv2_urls else ""
            if gdv2_url and gdv2_url != resolved_ref_link:
                chosen_url, chosen_option, chosen_reason = _compare_final_visuals(
                    gdv2_url=gdv2_url,
                    reference_url=resolved_ref_link,
                    vo_text=voiceover_text,
                    slide_title=slide_title,
                    slide_chunk=slide_chunk,
                    course_name=course_name,
                    topic_name=topic_name,
                    subtopic_name=subtopic_name,
                    drive=drive,
                    llm=llm,
                )
                if chosen_url and chosen_url != gdv2_url:
                    final_graphics_definition_text = final_graphics_definition_text.replace(gdv2_url, chosen_url, 1)
                    print(
                        f"[FINAL-COMPARE] Row {index + 1}: chose reference visual ({chosen_option}) - {chosen_reason or 'no reason provided'}"
                    )
                else:
                    print(
                        f"[FINAL-COMPARE] Row {index + 1}: kept GDv2 visual ({chosen_option}) - {chosen_reason or 'no reason provided'}"
                    )

        return index, final_graphics_definition_text, evaluation_breakdown_text, resolved_ref_link
        
    except Exception as e:
        print(f"Error processing row {index}: {e}")
        return index, "", ""


def validate_final_graphics_definition_row(row):
    """
    Validate that final_graphics_definition matches voiceover_segment:
    - Row is not empty
    - All segments from voiceover_segment have results
    - No gaps in segment numbering (must be sequential starting from 1)
    - For "1 Visual for the whole Slide", expect exactly SEGMENT 1
    
    :param row: Pandas Series with row data
    :return: Tuple (is_valid, error_message)
    """
    vo_segments_text = str(row.get("voiceover_segment", "")).strip()
    final_graphics_def_text = str(row.get("final_graphics_definition", "")).strip()
    visual_assignment_strategy = str(row.get("Visual Assignment Strategy", "Flexible, let the agent decide")).strip()
    
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
    
    # Check if this should be treated as a single-visual slide
    ref_image = str(row.get("Reference Image", "")).strip()
    is_forced_single = (visual_assignment_strategy == "1 Visual for the whole Slide") or (ref_image and ref_image != "nan")

    # For single-visual strategy, segment markers are optional
    if is_forced_single:
        # For reference rows: verify that the normal GDv2 result is present.
        # 'Graphics to use:' (the standard key) must exist so GDv2 failures are
        # caught by the outer retry loop even when the reference key is written.
        if ref_image and ref_image != "nan":
            gdv2_key_present = bool(
                re.search(r'^Graphics to use:\s*\S', final_graphics_def_text, re.MULTILINE)
            )
            if not gdv2_key_present:
                return False, "Reference row is missing 'Graphics to use:' (GDv2 path failed or produced no result)"

        if segment_numbers and (len(segment_numbers) != 1 or segment_numbers[0] != 1):
             return False, f"Expected either no segment markers or exactly SEGMENT 1 for '1 Visual for the whole Slide', found: {segment_numbers}"

        return True, None
    
    # For other strategies, validate against voiceover_segment
    # Skip validation if voiceover_segment is empty
    if not vo_segments_text or vo_segments_text == "nan":
        return True, None
    
    # Count segments in voiceover_segment (split by newline)
    vo_segments = [seg.strip() for seg in vo_segments_text.split('\n') if seg.strip()]
    expected_count = len(vo_segments)
    
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


def _parse_iso8601_duration_to_seconds(duration):
    """Parse YouTube ISO8601 duration like PT1H2M3S to seconds."""
    if not duration:
        return None
    match = re.match(
        r"^P(?:(?P<days>\d+)D)?T(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?(?:(?P<seconds>\d+)S)?$",
        duration.strip(),
        re.IGNORECASE,
    )
    if not match:
        return None
    days = int(match.group("days") or 0)
    hours = int(match.group("hours") or 0)
    minutes = int(match.group("minutes") or 0)
    seconds = int(match.group("seconds") or 0)
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def _get_youtube_api_key():
    return (
        os.getenv("GCLOUD_YT_SEARCH_API_KEY_1")
        or os.getenv("GCLOUD_YT_SEARCH_API_KEY_2")
        or os.getenv("GCLOUD_YT_SEARCH_API_KEY_3")
    )


def _fetch_youtube_duration_seconds(video_id, api_key, timeout_sec = 12):
    """
    Fetch the duration of a YouTube video in seconds.
    """
    if not video_id or not api_key:
        return None
    try:
        resp = requests.get(
            "https://www.googleapis.com/youtube/v3/videos",
            params={"part": "contentDetails", "id": video_id, "key": api_key},
            timeout=timeout_sec,
        )
        resp.raise_for_status()
        payload = resp.json() if resp.content else {}
        items = payload.get("items", []) if isinstance(payload, dict) else []
        if not items:
            return None
        iso = items[0].get("contentDetails", {}).get("duration")
        return _parse_iso8601_duration_to_seconds(iso)
    except Exception:
        return None


def _extract_graphics_to_use_urls(text):
    urls = []
    if not text:
        return urls
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.lower().startswith("graphics to use:"):
            url = line.split(":", 1)[1].strip()
            if url:
                urls.append(url)
    return urls


def validate_youtube_clip_links_for_row(row, api_key, duration_cache = None):
    """
    Validate YouTube clip timestamps in final_graphics_definition against actual video duration.
    """
    
    if not api_key:
        return True, None
    final_graphics_def_text = str(row.get("final_graphics_definition", "")).strip()
    if not final_graphics_def_text or final_graphics_def_text == "nan":
        return True, None

    urls = _extract_graphics_to_use_urls(final_graphics_def_text)
    if not urls:
        return True, None

    if duration_cache is None:
        duration_cache = {}

    faulty_messages: List[str] = []
    for url in urls:
        if "youtube.com" not in url and "youtu.be" not in url:
            continue
        _, start_seconds, end_seconds = parse_video_url_timestamps(url)
        if start_seconds is None and end_seconds is None:
            continue

        video_id = extract_video_id_from_url(url)
        if not video_id:
            faulty_messages.append(f"could not parse video id for URL: {url}")
            continue

        if video_id in duration_cache:
            duration_seconds = duration_cache[video_id]
        else:
            duration_seconds = _fetch_youtube_duration_seconds(video_id, api_key)
            duration_cache[video_id] = duration_seconds

        if duration_seconds is None:
            faulty_messages.append(f"could not fetch duration for video_id={video_id}")
            continue
        if start_seconds is not None and start_seconds >= duration_seconds:
            faulty_messages.append(
                f"start={start_seconds}s is outside video duration={duration_seconds}s for {url}"
            )
            continue
        if end_seconds is not None and end_seconds > duration_seconds:
            faulty_messages.append(
                f"end={end_seconds}s is outside video duration={duration_seconds}s for {url}"
            )
            continue
        if start_seconds is not None and end_seconds is not None and end_seconds <= start_seconds:
            faulty_messages.append(f"invalid range start={start_seconds}s end={end_seconds}s for {url}")

    if faulty_messages:
        return False, " | ".join(faulty_messages[:3])
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
def run_aggregation_agent_for_all_rows(sheet, llm="gemini_3_flash_thinking", max_workers=50):
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
    
    # Setup Drive client for the Reference Image Pipeline.
    # Priority: (1) dedicated service-account from env var, (2) reuse the existing session drive.
    ref_drive_client = None
    ref_output_folder_id = None
    ref_drive_lock = threading.Lock()

    service_account_json = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
    if service_account_json:
        try:
            from services.drive_service import login_with_service_account
            from agents.graphics_asset_creation.gac_utils import get_or_create_drive_folder
            gauth = login_with_service_account(json_str=service_account_json)
            ref_drive_client = GoogleDrive(gauth)
            ref_output_folder_id = get_or_create_drive_folder(ref_drive_client, "Reference Image Pipeline GDv2")
        except Exception as auth_err:
            print(f"⚠️ [REF-PIPELINE] Service-account auth failed: {auth_err}")

    print(f"\n{'='*80}")
    print(f"🚀 Starting Aggregation Agent")
    print(f"📚 Course: {course_name}")
    print(f"📊 Processing {len(df)} row(s)")
    print(f"{'='*80}\n")

    # Get Drive instance (session-based)
    drive = get_drive_instance()
    if not drive:
        print("❌ Could not initialize Google Drive. Aborting.")
        return

    # Fallback: reuse the session drive client if dedicated auth above was unavailable
    if ref_drive_client is None and drive is not None:
        try:
            from agents.graphics_asset_creation.gac_utils import get_or_create_drive_folder
            ref_drive_client = drive
            ref_output_folder_id = get_or_create_drive_folder(ref_drive_client, "Reference Image Pipeline GDv2")
        except Exception as folder_err:
            print(f"⚠️ [REF-PIPELINE] Could not resolve output folder (reference pipeline disabled): {folder_err}")

    print(f"🔗 [REF-PIPELINE] Drive ready: {'yes' if ref_drive_client and ref_output_folder_id else 'no (reference images will be skipped)'}")
    
    # Ensure output columns exist
    if "final_graphics_definition" not in df.columns:
        df["final_graphics_definition"] = ""
    if "reference_image_processed_url" not in df.columns:
        df["reference_image_processed_url"] = ""
    
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

    # Precompute reference-image outputs first so the final comparison step can
    # reuse the saved URL from the sheet instead of re-running the reference path.
    ref_rows_to_process = []
    for index, row in rows_to_process:
        ref_image_url = str(row.get("Reference Image", "")).strip()
        if ref_image_url and ref_image_url.lower() != "nan":
            ref_rows_to_process.append((index, row, ref_image_url))

    if ref_rows_to_process and ref_drive_client and ref_output_folder_id:
        print(f"🧪 Precomputing {len(ref_rows_to_process)} reference-image row(s) in parallel before GDv2 aggregation...")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            ref_futures = {
                executor.submit(
                    process_reference_image_path,
                    df,
                    index,
                    ref_image_url,
                    ref_drive_client,
                    ref_output_folder_id,
                    ref_drive_lock,
                    llm,
                    str(row.get("voiceover_segment", "")).strip(),
                ): index
                for index, row, ref_image_url in ref_rows_to_process
            }

            for future in as_completed(ref_futures):
                index = ref_futures[future]
                try:
                    is_relevant, ref_final_def, _ = future.result()
                    ref_link = ""
                    if is_relevant and ref_final_def:
                        graphics_match = re.search(r'Graphics to use:\s*(https?://[^\s\n]+)', ref_final_def)
                        if graphics_match:
                            ref_link = graphics_match.group(1).strip()
                    df.at[index, "reference_image_processed_url"] = ref_link
                    print(f"[REF-PRECOMPUTE] Row {index + 1}: {'stored' if ref_link else 'empty'} reference_image_processed_url")
                except Exception as e:
                    print(f"[REF-PRECOMPUTE] Row {index + 1}: error while precomputing reference image: {e}")
                    df.at[index, "reference_image_processed_url"] = ""

        save_to_sheet(ws, df)
        format_worksheet(ws)
    
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
                llm,
                max_workers,
                ref_drive_client,
                ref_output_folder_id,
                ref_drive_lock,
                df,
                str(df.at[index, "reference_image_processed_url"]).strip(),
            ): index
            for index, row in rows_to_process
        }
        
        # Collect results as they complete
        for future in as_completed(futures):
            index = futures[future]
            try:
                row_index, final_graphics_def_text, _, ref_link = future.result()
                
                # Update dataframe
                df.at[row_index, "final_graphics_definition"] = final_graphics_def_text
                df.at[row_index, "reference_image_processed_url"] = ref_link or ""
                
                # Update progress
                progress.update()
                
                # Save immediately after each row completes
                print(f'Saving row {row_index + 2} to sheet immediately.')
                save_to_sheet(ws, df)
            except Exception as e:
                print(f"Error getting result for row {index}: {e}")
                # Update dataframe with error marker so row is marked as processed
                df.at[index, "final_graphics_definition"] = f"ERROR: {str(e)}"
                df.at[index, "reference_image_processed_url"] = ""
                progress.update()
                # Save immediately even on error
                print(f'Saving row {index + 2} (with error) to sheet immediately.')
                save_to_sheet(ws, df)

    # Save final results before validation
    save_to_sheet(ws, df)
    format_worksheet(ws)

    # Validation and retry logic
    youtube_api_key = _get_youtube_api_key()
    if youtube_api_key:
        print("🔍 Running post-aggregation YouTube clip validation.")
    else:
        print("⚠️ YouTube clip validation skipped: missing GCLOUD_YT_SEARCH_API_KEY_1/2/3.")

    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        # Reload dataframe to get latest state
        ws, df = get_sheet_data_and_df(sheet, worksheet_name)
        
        # Validate all rows and find invalid ones
        duration_cache: Dict[str, Optional[int]] = {}
        invalid_rows = []
        for index, row in df.iterrows():
            is_valid, error_msg = validate_final_graphics_definition_row(row)
            clip_valid, clip_error_msg = validate_youtube_clip_links_for_row(
                row, youtube_api_key, duration_cache
            )
            if not is_valid or not clip_valid:
                reasons: List[str] = []
                if not is_valid and error_msg:
                    reasons.append(error_msg)
                if not clip_valid and clip_error_msg:
                    reasons.append(f"YouTube clip validation failed: {clip_error_msg}")
                    print(f"⚠️ Faulty clip found in row {index + 2}: {clip_error_msg}")
                invalid_rows.append((index, row, " || ".join(reasons)))
        
        if not invalid_rows:
            if youtube_api_key:
                print("✅ YouTube clip validation completed: no faulty clip links found.")
            break
        
        retry_count += 1
        print(f"\n⚠️ Found {len(invalid_rows)} rows with invalid final_graphics_definition. Retrying (attempt {retry_count}/{max_retries})...")
        for idx, row, error in invalid_rows[:3]:  # Show first 3 errors
            print(f"  Row {idx + 2}: {error}")
        
        # Clear final_graphics_definition for invalid rows
        for index, row, error_msg in invalid_rows:
            df.at[index, "final_graphics_definition"] = ""
        
        # Save cleared state
        save_to_sheet(ws, df)
        format_worksheet(ws)
        
        # Retry processing invalid rows
        futures = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for index, row, error_msg in invalid_rows:
                print(f"🔁 Re-aggregating row {index + 2} after validation failure.")
                future = executor.submit(
                    process_aggregation_row,
                    index,
                    row,
                    course_name,
                    drive,
                    llm,
                    max_workers,
                    ref_drive_client,
                    ref_output_folder_id,
                    ref_drive_lock,
                    df,
                    str(df.at[index, "reference_image_processed_url"]).strip(),
                )
                futures[future] = index
            
            # Collect results
            for future in as_completed(futures):
                index = futures[future]
                try:
                    row_index, final_graphics_def_text, _, ref_link = future.result()
                    df.at[row_index, "final_graphics_definition"] = final_graphics_def_text
                    df.at[row_index, "reference_image_processed_url"] = ref_link or ""
                    # Save immediately after each row completes in retry
                    print(f'Saving row {row_index + 2} (retry) to sheet immediately.')
                    save_to_sheet(ws, df)
                except Exception as e:
                    print(f"Error getting result for row {index} on retry: {e}")
                    df.at[index, "final_graphics_definition"] = f"ERROR: {str(e)}"
                    df.at[index, "reference_image_processed_url"] = ""
                    # Save immediately even on error in retry
                    print(f'Saving row {index + 2} (retry, with error) to sheet immediately.')
                    save_to_sheet(ws, df)
    
    if retry_count > 0:
        # Check final state
        ws, df = get_sheet_data_and_df(sheet, worksheet_name)
        duration_cache: Dict[str, Optional[int]] = {}
        final_invalid = []
        for index, row in df.iterrows():
            is_valid, error_msg = validate_final_graphics_definition_row(row)
            clip_valid, clip_error_msg = validate_youtube_clip_links_for_row(
                row, youtube_api_key, duration_cache
            )
            if not is_valid or not clip_valid:
                reasons: List[str] = []
                if not is_valid and error_msg:
                    reasons.append(error_msg)
                if not clip_valid and clip_error_msg:
                    reasons.append(f"YouTube clip validation failed: {clip_error_msg}")
                final_invalid.append((index, " || ".join(reasons)))
        
        if final_invalid:
            print(f"⚠️ After {retry_count} retry attempt(s), {len(final_invalid)} rows still have invalid final_graphics_definition.")
            for idx, error in final_invalid[:5]:  # Show first 5
                print(f"  Row {idx + 2}: {error}")
        else:
            print(f"✅ All rows validated after {retry_count} retry attempt(s).")
            if youtube_api_key:
                print("✅ YouTube clip validation completed successfully after retries (no faulty clip links remain).")    
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
        cols_to_drop = ["final_graphics_definition"]
        if "evaluation_breakdown" in df.columns:
            cols_to_drop.append("evaluation_breakdown")
        df = df.drop(columns=cols_to_drop)
        clear_worksheet(ws)
        save_to_sheet(ws, df)
        print(f"🗑️ Deleted 'final_graphics_definition' column from '{worksheet_name}' worksheet")
    else:
        print(f"ℹ️ 'final_graphics_definition' column does not exist in '{worksheet_name}' worksheet")
