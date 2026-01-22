from langsmith import traceable
import streamlit as st
from services.sheets_service import get_sheet_data_and_df, save_to_sheet, format_worksheet, clear_worksheet
from services.smart_progress_bar import SmartProgressBar
import re
import requests
from io import BytesIO
from PIL import Image
from agents.vector_store_image_search.create_vectorstore import download_image_from_drive
from services.video_clip_tools import build_video_part
from dotenv import load_dotenv
import os
import base64
from services.drive_service import login_with_service_account
from pydrive2.drive import GoogleDrive
from google import genai
from google.genai import types
from typing import List, Dict, Optional, Tuple, Any

load_dotenv()




layout_agent_prompt = """You are a senior instructional visual designer specializing in HVAC e-learning content. Your task is to generate production-ready slide layout instructions from a finalized graphics definition.
You will be given the slide context and a graphics definition that already specifies the exact visual asset(s) to show for each narration part. Your responsibility is to decide:
- how those provided assets are arranged on a persistent slide canvas,
- how they transition at narration boundaries,
- which previously shown assets remain, move, or are removed to maintain visual continuity and clarity.

These are the inputs:

<course_information>
Course name: {course_name}
Topic name: {topic_name}
Subtopic name: {subtopic_name}
</course_information>

<slide_context>
Slide title: {slide_title}
Slide content that needs layout: {slide_content}
</slide_context>

<graphics_definition_for_this_slide_content>
{final_graphics_definition}
</graphics_definition_for_this_slide_content>

<previous_slide_layout>
{previous_slide_layout}
</previous_slide_layout>

Instructions:

1. Authority and Scope:
   - The graphics definition is final and authoritative.
   - Your role is limited strictly to layout planning.
   - You MUST NOT select, replace, modify, reinterpret, or reject any visual asset or visual concept defined in the graphics definition.
   - You MUST NOT introduce new visuals, annotations, emphasis, or instructional meaning.
   - You MUST generate layout instructions only for the exact narration parts and assets provided.

2. Persistent Canvas:
   - Treat the slide as a single, persistent canvas that evolves over time as narration progresses.
   - The canvas state at the end of one narration part becomes the starting canvas state for the next narration part.
   - If a previous_slide_layout is provided, treat it as the initial canvas state at the start of this slide.
   - Visuals are not reset between narration parts unless you explicitly remove them.
   - Maintain visual continuity by keeping, repositioning, or removing visuals only when necessary for clarity or space.

3. Visual Lifecycle and Coexistence
   - Each narration part has exactly one primary visual: the visual asset explicitly assigned to that narration part in the graphics definition.
   - The primary visual for the current narration part must always be introduced during that narration part.
   - Visuals introduced in earlier narration parts may remain on screen as secondary visuals if they provide context, continuity, or comparison.
   - Secondary visuals must not dominate the layout; they should be positioned or scaled in a way that keeps the current primary visual visually dominant.
   - Prefer showing one visual (the primary) or two visuals (the primary and at most one secondary, typically the visual assigned to the immediately preceding narration part) on the canvas. Show more than two only if required to preserve continuity across closely related narration parts.
   - If the narration clearly switches to a new example, location, or concept, you should generally remove previously shown visuals before introducing the new primary visual to avoid visual confusion.
   -  If the narration refines, reveals, or further explains the same object or concept, you should generally keep the previous visual and introduce the new visual alongside it to preserve continuity.

4. Narration-Driven Timing and Transitions
   - The voiceover narration defines the timeline for all visual changes.
   - Visual changes must occur only at narration boundaries defined in the graphics definition.
   - Do not introduce, remove, or reposition visuals mid-narration.
   - For each narration part, apply all layout actions at the start of that narration part.

Output Schema & Action Constraints 

This section defines the exact output schema, allowed actions, fields, and values you must use when generating layout instructions. You must follow these constraints strictly and produce output that conforms exactly to this schema.

1. Overall Structure
- The output must be composed of one or more <narration_part> blocks.
- Each <narration_part> represents a single narration boundary from the graphics definition.
- <narration_part> blocks must appear in the exact spoken order of the narration.
- Each <narration_part> must contain the following sections in this exact order:
  - <voiceover>
  - <canvas_state_before>
  - One or more <action> blocks
  - <canvas_state_after>
- No other sections or free-form text are permitted inside <narration_part>.

2. Voiceover
- <voiceover> must contain the exact narration text from the graphics definition.
- The text must match verbatim, including punctuation.
- Do not paraphrase, shorten, merge, or reorder narration text.

3. Canvas State Before
- <canvas_state_before> defines the complete set of visuals present on the canvas at the start of the narration part.
- It must list all visuals carried over from the previous narration part or from the previous slide.
- If no visuals are present (in cases where <previous_slide_layout> input is empty), the block must still be included and left empty.
- Each visual must be represented using a <visual> block containing:
  - asset
  - position

4. Actions
- Actions describe how the canvas changes at the narration boundary.
- Each <action> block applies to exactly one visual asset.
- Multiple <action> blocks may appear within a single narration part because a narration boundary can require multiple coordinated layout changes, such as:
  - removing visuals that are no longer relevant,
  - repositioning visuals that should remain as secondary context,
  - and introducing the new primary visual for the current narration part.
- Actions are executed in the order they appear, allowing the canvas to transition step-by-step from the previous state to the new state.
- All actions together must transform <canvas_state_before> into <canvas_state_after>.

Allowed Action Types
- Action: add
  - Introduces a new visual asset onto the canvas.
  - Required fields:
    - Asset
    - Position
    - Transition
  - Allowed values:
    - Asset: The link to the asset
    - Position: full | center | left | right
    - Transition: fade_in | none

- Action: keep
  - Keeps an existing visual exactly as it is.
  - Required fields:
    - Asset
    - Transition
  - Allowed values:
    - Asset: The link to the asset
    - Transition: none

- Action: move
  - Repositions an existing visual on the canvas.
  - Required fields:
    - Asset
    - Position
    - Transition
  - Allowed values:
    - Asset: The link to the asset
    - Position: full | center | left | right
    - Transition: slide_left | slide_right
  - Transition guidance for move actions:
    - When moving a visual to a new position, you MUST use a slide transition to create smooth animation instead of instant repositioning.
    - Use slide_left when moving to the left position.
    - Use slide_right when moving to the right position
    - Choose the slide direction based on the primary direction of movement from the current position to the new position.

- Action: remove
  - Removes a visual from the canvas.
  - Required fields:
    - Asset
    - Transition
  - Allowed values:
    - Asset: The link to the asset
    - Transition: fade_out

5. Required Action Constraints per Narration Part
- Each <narration_part> must include exactly one add action.
- There may be zero or more keep, move, or remove actions.
- The add action must reference the visual asset assigned to that narration part in the graphics definition.
- Actions must not reference assets that are not present in <canvas_state_before> or introduced via add.

6. Canvas State After
- <canvas_state_after> defines the complete set of visuals present on the canvas at the end of the narration part.
- It must include all visuals that remain visible after all actions are applied.
- It must not include visuals that were removed.
- Each visual must specify:
  - asset
  - position
- The listed positions must reflect the final layout after all actions.

7. Canvas Continuity Rules
- The <canvas_state_after> of a narration part becomes the <canvas_state_before> of the next narration part.
- For the first narration part of a slide, <canvas_state_before> must reflect the previous_slide_layout input.

8. Position Constraints
- At most one visual may use position: full.
- At most one visual may use position: center.
- left and right may coexist.
- full should be used only when the visual is intended to dominate the entire slide.

9. Strictness Requirements
- Do not introduce fields, actions, or values not explicitly defined above.
- Do not omit required sections.
- Do not infer or assume canvas state.
- Do not describe layout decisions outside the defined structure.

Output Format:

Always provide your output strictly in this exact format:

<output>

<evaluation_breakdown>
This section is a reasoning scratchpad for you plan the slide layout before producing the final layout instructions.

In this section, you should:
- Analyze the slide content and break it into narration parts as defined in the graphics definition.
- Review the visual assets assigned to each narration part and note what each visual depicts.
- Identify which visuals are primary for each narration part and which previously shown visuals, if any, may need to remain as secondary context.
- Consider the incoming canvas state from the previous slide or previous narration part.
- Decide which visuals should be removed, kept, moved, or added at each narration boundary.
- Plan how the canvas should evolve step by step to maintain clarity, continuity, and visual focus.
- Enter any other thoughts and analysis you have that will help you arrive at the correct output.

It is acceptable for this section to be very detailed and verbose to help you arrive at a correct and consistent layout plan.
</evaluation_breakdown>

<layout_instructions>

<narration_part>

<voiceover>
Exact narration part text from the graphics definition
</voiceover>

<canvas_state_before>
<visual>
Asset: Asset Link
Position: full | center | left | right
</visual>
<!-- repeat <visual> blocks as needed, or leave empty if no visuals are present -->
</canvas_state_before>

<action>
Action: add | keep | move | remove
Asset: Asset Link
Position: full | center | left | right        (only for add and move)
Transition: fade_in | fade_out | slide_left | slide_right | none
</action>

<!-- repeat <action> blocks as needed -->

<canvas_state_after>
<visual>
Asset: Asset Link
Position: full | center | left | right
</visual>
<!-- repeat <visual> blocks as needed -->
</canvas_state_after>

</narration_part>

<!-- repeat <narration_part> blocks in spoken order as given in the input-->

</layout_instructions>

</output>
"""


def get_drive_instance():
    """
    Get Google Drive instance from session state or initialize from environment.
    
    :return: Google Drive instance
    """
    # First try to get from session state
    if "drive" in st.session_state:
        return st.session_state["drive"]
    
    # Fallback: initialize from environment variables
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


def load_image_from_url(image_url: str, drive, title: str = "") -> Optional[Image.Image]:
    """
    Load image from URL (web URL or Google Drive URL).
    
    :param image_url: Image URL (web or Google Drive)
    :param drive: Google Drive instance
    :param title: Optional title for logging
    :return: PIL Image or None if loading fails
    """
    try:
        # Check if it's a Google Drive URL
        if "drive.google.com" in image_url:
            # Extract file ID from Drive URL
            file_id_match = re.search(r'/d/([a-zA-Z0-9_-]+)', image_url)
            if file_id_match:
                file_id = file_id_match.group(1)
                pil_image = download_image_from_drive(drive, file_id)
                return pil_image
            else:
                print(f"⚠️ Could not extract file ID from Drive URL: {image_url}")
                return None
        else:
            # Regular web URL
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(image_url, headers=headers, timeout=30)
            response.raise_for_status()
            pil_image = Image.open(BytesIO(response.content))
            return pil_image
    except Exception as e:
        print(f"⚠️ Error loading image from {image_url}: {e}")
        return None


def parse_video_url_timestamps(video_url: str) -> Tuple[Optional[str], Optional[int], Optional[int]]:
    """
    Parse video URL to extract base URL and timestamps.
    
    :param video_url: YouTube embed URL with optional timestamps
    :return: Tuple of (base_url, start_seconds, end_seconds)
    """
    try:
        # Match YouTube embed URL with timestamps
        # Format: https://www.youtube.com/embed/VIDEO_ID?start=X&end=Y
        match = re.match(r'(https://www\.youtube\.com/embed/[^?]+)\??(.*)', video_url)
        if not match:
            return None, None, None
        
        base_url = match.group(1)
        params_str = match.group(2) if match.group(2) else ""
        
        # Parse query parameters
        start_seconds = None
        end_seconds = None
        
        if params_str:
            params = dict(p.split('=') for p in params_str.split('&') if '=' in p)
            start_seconds = int(params.get('start', 0)) if 'start' in params else None
            end_seconds = int(params.get('end', 0)) if 'end' in params else None
        
        return base_url, start_seconds, end_seconds
    except Exception as e:
        print(f"⚠️ Error parsing video URL {video_url}: {e}")
        return None, None, None


def extract_last_narration_part_from_layout(layout_instructions):
    """
    Extract the last narration_part from the previous slide's layout_instructions.
    
    :param layout_instructions: The layout_instructions column content from previous slide
    :return: The last narration_part XML block, or None if not found
    """
    if not layout_instructions or layout_instructions.strip() == "" or layout_instructions == "nan":
        return None
    
    # Remove <layout_instructions> wrapper if present
    layout_content = layout_instructions
    layout_wrapper_match = re.search(
        r'<layout_instructions>(.*?)</layout_instructions>',
        layout_instructions,
        re.DOTALL | re.IGNORECASE
    )
    if layout_wrapper_match:
        layout_content = layout_wrapper_match.group(1)
    
    # Find all narration_part blocks
    narration_parts = re.findall(
        r'<narration_part>(.*?)</narration_part>',
        layout_content,
        re.DOTALL | re.IGNORECASE
    )
    
    if not narration_parts:
        return None
    
    # Return the last one
    last_part = narration_parts[-1]
    return f"<narration_part>{last_part}</narration_part>"


def extract_assets_from_canvas_state(canvas_state_text):
    """
    Extract asset URLs from canvas_state_after in a narration_part.
    
    :param canvas_state_text: The canvas_state_after section text
    :return: List of dicts with 'asset' and 'position' keys
    """
    assets = []
    
    # Find all visual blocks
    visual_pattern = r'<visual>(.*?)</visual>'
    visuals = re.findall(visual_pattern, canvas_state_text, re.DOTALL | re.IGNORECASE)
    
    for visual in visuals:
        # Extract asset URL
        asset_match = re.search(r'Asset:\s*(\S+)', visual, re.IGNORECASE)
        position_match = re.search(r'Position:\s*(\S+)', visual, re.IGNORECASE)
        
        if asset_match:
            asset_url = asset_match.group(1).strip()
            position = position_match.group(1).strip() if position_match else None
            
            assets.append({
                'asset': asset_url,
                'position': position
            })
    
    return assets


def extract_graphics_from_definition(final_graphics_definition: str) -> List[Dict[str, Any]]:
    """
    Extract all graphics (images and videos) from the final_graphics_definition text.
    
    Parses the structured format to find all "Graphics to use:" URLs and their associated
    VO parts and segments.
    
    :param final_graphics_definition: The final_graphics_definition column content
    :return: List of dicts with keys: 'segment', 'vo_part', 'url', 'type' (image/video/video_frame)
    """
    if not final_graphics_definition or final_graphics_definition.strip() == "" or final_graphics_definition == "nan":
        return []
    
    graphics = []
    current_segment = 0
    current_vo_part = ""
    
    # Split by segment markers
    segment_pattern = r'={50,}\s*SEGMENT\s+(\d+)\s*={50,}'
    segments = re.split(segment_pattern, final_graphics_definition)
    
    # Process each segment
    for i in range(1, len(segments), 2):
        if i + 1 < len(segments):
            segment_num = int(segments[i])
            segment_content = segments[i + 1]
            
            # Split segment content by "----" separator to get VO parts
            vo_parts = re.split(r'\n-{4,}\n', segment_content)
            
            for vo_part_content in vo_parts:
                if not vo_part_content.strip():
                    continue
                
                # Extract VO part text
                vo_match = re.search(r'When VO:\s*["\'](.+?)["\']', vo_part_content, re.DOTALL)
                vo_text = vo_match.group(1).strip() if vo_match else ""
                
                # Extract Graphics URL
                graphics_match = re.search(r'Graphics to use:\s*(\S+)', vo_part_content)
                if graphics_match:
                    url = graphics_match.group(1).strip()
                    
                    # Determine type
                    if 'youtube.com' in url or 'youtu.be' in url:
                        # Check if it has both start and end (video clip) or just start (still frame)
                        if 'start=' in url and 'end=' in url:
                            graphic_type = 'video'
                        elif 'start=' in url:
                            graphic_type = 'video_frame'
                        else:
                            graphic_type = 'video'
                    else:
                        graphic_type = 'image'
                    
                    graphics.append({
                        'segment': segment_num,
                        'vo_part': vo_text,
                        'url': url,
                        'type': graphic_type
                    })
    
    return graphics


def invoke_gemini_multimodal(parts: List[types.Part], llm: str = "gemini_3_flash_thinking", temperature: float = 0.7) -> str:
    """
    Invoke Gemini multimodal model with the given parts.
    
    :param parts: List of multimodal parts (text, images, videos)
    :param llm: Model name
    :param temperature: Temperature for generation
    :return: Response text
    """
    # Map LLM name to model ID
    model_map = {
        "gemini_3_flash_thinking": "gemini-3-flash-preview",
        "gemini_3_flash": "gemini-3-flash-preview",
        "gemini_2_5_flash": "gemini-2.5-flash",
        "gemini-2.5-flash": "gemini-2.5-flash",
        "gemini-3-flash-preview": "gemini-3-flash-preview",
    }
    
    model_id = model_map.get(llm, llm)  # Default to llm if not in mapping
    
    # Initialize client
    client = genai.Client()
    
    # Generate content
    response = client.models.generate_content(
        model=model_id,
        contents=types.Content(parts=parts),
        config=types.GenerateContentConfig(
            temperature=temperature,
        ),
    )
    
    # Handle response - try multiple ways to extract text
    if hasattr(response, "text") and response.text:
        return response.text
    if getattr(response, "candidates", None):
        first_candidate = response.candidates[0]
        if getattr(first_candidate, "content", None) and first_candidate.content.parts:
            part = first_candidate.content.parts[0]
            if hasattr(part, "text"):
                return part.text
    return str(response)


@traceable(
    metadata={
        "agent_name": "graphics_definition_v2",
        "step_name": "Layout Agent",
        "function_name": "generate_layout_for_row",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous")
    }
)
def generate_layout_for_row(
    course_name: str,
    topic_name: str,
    subtopic_name: str,
    slide_title: str,
    slide_content: str,
    final_graphics_definition: str,
    previous_slide_layout: Optional[str],
    drive,
    llm: str = "gemini_3_flash_thinking"
) -> Tuple[str, str]:
    """
    Generate layout instructions for a single row.
    
    :param course_name: Course name
    :param topic_name: Topic name
    :param subtopic_name: Subtopic name
    :param slide_title: Slide title
    :param slide_content: Full slide content
    :param final_graphics_definition: The final_graphics_definition column content
    :param previous_slide_layout: The layout_instructions from the previous slide (last narration_part)
    :param drive: Google Drive instance
    :param llm: Language model to use
    :return: Tuple of (layout_output, evaluation_breakdown)
    """
    print(f"\n{'='*80}")
    print(f"📐 Layout Agent - Processing: {slide_title}")
    print(f"{'='*80}")
    
    # Extract graphics from the definition
    graphics = extract_graphics_from_definition(final_graphics_definition)
    
    if not graphics:
        print("⚠️ No graphics found in final_graphics_definition")
        return "", ""
    
    print(f"📊 Found {len(graphics)} graphics to process")
    
    # Extract last narration_part from previous slide and its visuals
    previous_narration_part = None
    previous_assets = []
    
    if previous_slide_layout:
        print(f"📎 Previous slide layout provided (length: {len(previous_slide_layout)} chars)")
        previous_narration_part = extract_last_narration_part_from_layout(previous_slide_layout)
        if previous_narration_part:
            print(f"✅ Extracted last narration_part from previous slide (length: {len(previous_narration_part)} chars)")
            # Extract canvas_state_after from the last narration_part
            canvas_state_match = re.search(
                r'<canvas_state_after>(.*?)</canvas_state_after>',
                previous_narration_part,
                re.DOTALL | re.IGNORECASE
            )
            if canvas_state_match:
                previous_assets = extract_assets_from_canvas_state(canvas_state_match.group(1))
                print(f"📋 Found {len(previous_assets)} asset(s) from previous slide")
            else:
                print(f"⚠️ No canvas_state_after found in previous narration_part")
        else:
            print(f"⚠️ Could not extract narration_part from previous slide layout")
    else:
        print(f"ℹ️ No previous slide layout provided (first slide or no previous layout)")
    
    # Build multimodal parts
    parts: List[types.Part] = []
    
    # Process each graphic from current slide and add to parts (for graphics_definition section)
    current_slide_visuals = []
    for idx, graphic in enumerate(graphics, start=1):
        url = graphic['url']
        graphic_type = graphic['type']
        vo_part = graphic['vo_part']
        segment = graphic['segment']
        
        label = f"Graphic {idx} (Segment {segment})"
        if vo_part:
            label += f" - VO: \"{vo_part[:50]}...\""
        
        visual_parts = []
        if graphic_type == 'image':
            # Load image
            pil_image = load_image_from_url(url, drive, label)
            if pil_image:
                # Convert PIL image to bytes
                buffered = BytesIO()
                pil_image.convert("RGB").save(buffered, format="JPEG")
                image_bytes = buffered.getvalue()
                
                # Add text label
                visual_parts.append(types.Part(text=f"{label}\nURL: {url}"))
                # Add image part
                visual_parts.append(types.Part(inline_data=types.Blob(mime_type="image/jpeg", data=image_bytes)))
                print(f"✅ Loaded image: {label}")
            else:
                print(f"⚠️ Failed to load image: {label}")
                
        elif graphic_type == 'video':
            # Video clip with timestamps
            base_url, start_seconds, end_seconds = parse_video_url_timestamps(url)
            if base_url:
                # Add text label
                visual_parts.append(types.Part(text=f"{label} (Video Clip)\nURL: {url}"))
                # Add video part
                video_part = build_video_part(base_url, start_seconds, end_seconds)
                visual_parts.append(video_part)
                print(f"✅ Added video clip: {label} (start={start_seconds}s, end={end_seconds}s)")
            else:
                print(f"⚠️ Failed to parse video URL: {url}")
                
        elif graphic_type == 'video_frame':
            # Still frame from video (single timestamp)
            base_url, start_seconds, _ = parse_video_url_timestamps(url)
            if base_url and start_seconds is not None:
                # Add text label
                visual_parts.append(types.Part(text=f"{label} (Still Frame from Video)\nURL: {url}"))
                # Add video part with same start/end to get a single frame
                video_part = build_video_part(base_url, start_seconds, start_seconds + 1)
                visual_parts.append(video_part)
                print(f"✅ Added video frame: {label} (at {start_seconds}s)")
            else:
                print(f"⚠️ Failed to parse video frame URL: {url}")
    
        current_slide_visuals.extend(visual_parts)
    
    # Prepare previous slide visuals (to be inserted at previous_slide_layout section)
    previous_slide_visual_parts = []
    if previous_assets:
        for idx, asset_info in enumerate(previous_assets, start=1):
            asset_url = asset_info['asset']
            position = asset_info.get('position', 'unknown')
            
            # Determine type
            if 'youtube.com' in asset_url or 'youtu.be' in asset_url:
                if 'start=' in asset_url and 'end=' in asset_url:
                    graphic_type = 'video'
                elif 'start=' in asset_url:
                    graphic_type = 'video_frame'
                else:
                    graphic_type = 'video'
            else:
                graphic_type = 'image'
            
            label = f"Previous Slide Visual {idx} (Position: {position})"
            
            if graphic_type == 'image':
                pil_image = load_image_from_url(asset_url, drive, label)
                if pil_image:
                    buffered = BytesIO()
                    pil_image.convert("RGB").save(buffered, format="JPEG")
                    image_bytes = buffered.getvalue()
                    previous_slide_visual_parts.append(types.Part(text=f"{label}\nURL: {asset_url}"))
                    previous_slide_visual_parts.append(types.Part(inline_data=types.Blob(mime_type="image/jpeg", data=image_bytes)))
                    print(f"✅ Loaded previous slide image: {label}")
            elif graphic_type == 'video':
                base_url, start_seconds, end_seconds = parse_video_url_timestamps(asset_url)
                if base_url:
                    previous_slide_visual_parts.append(types.Part(text=f"{label} (Video Clip)\nURL: {asset_url}"))
                    video_part = build_video_part(base_url, start_seconds, end_seconds)
                    previous_slide_visual_parts.append(video_part)
                    print(f"✅ Added previous slide video clip: {label}")
            elif graphic_type == 'video_frame':
                base_url, start_seconds, _ = parse_video_url_timestamps(asset_url)
                if base_url and start_seconds is not None:
                    previous_slide_visual_parts.append(types.Part(text=f"{label} (Still Frame from Video)\nURL: {asset_url}"))
                    video_part = build_video_part(base_url, start_seconds, start_seconds + 1)
                    previous_slide_visual_parts.append(video_part)
                    print(f"✅ Added previous slide video frame: {label}")
    
    # Format prompt with previous_slide_layout
    previous_layout_text = previous_narration_part if previous_narration_part else ""
    
    prompt_text = layout_agent_prompt.format(
        course_name=course_name,
        topic_name=topic_name,
        subtopic_name=subtopic_name,
        slide_title=slide_title,
        slide_content=slide_content,
        final_graphics_definition=final_graphics_definition,
        previous_slide_layout=previous_layout_text
    )
    
    # Split prompt to insert visuals at the right places
    # Find positions of key sections
    graphics_def_end = prompt_text.find('</graphics_definition_for_this_slide_content>')
    previous_slide_start = prompt_text.find('<previous_slide_layout>')
    previous_slide_end = prompt_text.find('</previous_slide_layout>')
    
    # Build parts with visuals inserted contextually
    if graphics_def_end != -1 and previous_slide_start != -1 and previous_slide_end != -1:
        # Split into: before graphics_def, after graphics_def (before previous_slide), after previous_slide
        prompt_before_graphics = prompt_text[:graphics_def_end + len('</graphics_definition_for_this_slide_content>')]
        prompt_before_previous = prompt_text[graphics_def_end + len('</graphics_definition_for_this_slide_content>'):previous_slide_start + len('<previous_slide_layout>')]
        # Extract the content between the tags (the XML narration_part)
        previous_slide_content = prompt_text[previous_slide_start + len('<previous_slide_layout>'):previous_slide_end]
        prompt_after_previous = prompt_text[previous_slide_end:]
        
        parts = []
        # Add prompt up to end of graphics_definition
        parts.append(types.Part(text=prompt_before_graphics))
        # Insert current slide visuals right after graphics_definition
        parts.extend(current_slide_visuals)
        # Add prompt between graphics_definition and previous_slide_layout
        parts.append(types.Part(text=prompt_before_previous))
        # Insert previous slide visuals right at previous_slide_layout section
        parts.extend(previous_slide_visual_parts)
        # Add the XML content (narration_part) after the visuals
        if previous_slide_content.strip():
            parts.append(types.Part(text=previous_slide_content))
        # Add rest of prompt after previous_slide_layout
        parts.append(types.Part(text=prompt_after_previous))
    elif previous_slide_start != -1 and previous_slide_end != -1:
        # Only previous_slide_layout found, insert visuals there
        prompt_before = prompt_text[:previous_slide_start + len('<previous_slide_layout>')]
        prompt_after = prompt_text[previous_slide_end:]
        
        parts = []
        parts.append(types.Part(text=prompt_before))
        parts.extend(previous_slide_visual_parts)
        parts.append(types.Part(text=prompt_after))
        # Also add current slide visuals at the beginning
        parts = current_slide_visuals + parts
    else:
        # No special sections found, add visuals then prompt
        parts.extend(current_slide_visuals)
        parts.append(types.Part(text=prompt_text))
    
    # Print formatted prompt structure for debugging
    print(f"\n{'─'*80}")
    print(f"📋 Formatted Prompt Structure:")
    print(f"{'─'*80}")
    print(f"Total parts: {len(parts)}")
    for i, part in enumerate(parts, 1):
        if hasattr(part, 'text') and part.text:
            print(part.text)
            print()
        elif hasattr(part, 'inline_data') and part.inline_data and hasattr(part.inline_data, 'data') and part.inline_data.data:
            print(f"JPEG image ({len(part.inline_data.data)} bytes)")
        elif hasattr(part, 'file_data') and part.file_data:
            video_meta = getattr(part, 'video_metadata', None)
            if video_meta:
                start = getattr(video_meta, 'start_offset', 'N/A')
                end = getattr(video_meta, 'end_offset', 'N/A')
                file_uri = getattr(part.file_data, 'file_uri', 'N/A') if part.file_data else 'N/A'
                print(f"VIDEO: {file_uri} ({start} to {end})")
            else:
                file_uri = getattr(part.file_data, 'file_uri', 'N/A') if part.file_data else 'N/A'
                print(f"VIDEO: {file_uri}")
        else:
            print(f"[Unknown part type: {type(part)}]")
    print(f"{'─'*80}\n")
    
    # Call LLM
    try:
        print(f"🤖 Calling {llm} with {len(graphics)} graphics...")
        response_text = invoke_gemini_multimodal(parts, llm=llm, temperature=0.7)
        
        # Print full response
        print(f"\n{'─'*80}")
        print(f"📤 Layout Agent Response:")
        print(f"{'─'*80}")
        print(response_text)
        print(f"{'─'*80}\n")
        
        # Extract evaluation_breakdown
        eval_match = re.search(
            r'<evaluation_breakdown>(.*?)</evaluation_breakdown>',
            response_text,
            re.DOTALL | re.IGNORECASE
        )
        evaluation_breakdown = eval_match.group(1).strip() if eval_match else ""
        
        # Extract layout_instructions (not from <output> tag)
        layout_match = re.search(
            r'<layout_instructions>(.*?)</layout_instructions>',
            response_text,
            re.DOTALL | re.IGNORECASE
        )
        layout_output = layout_match.group(1).strip() if layout_match else ""
        
        return layout_output, evaluation_breakdown
        
    except Exception as e:
        print(f"❌ Error calling LLM: {e}")
        return "", ""


@traceable(
    metadata={
        "agent_name": "graphics_definition_v2",
        "step_name": "Layout Agent",
        "function_name": "run_layout_agent_for_all_rows",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous")
    }
)
def run_layout_agent_for_all_rows(sheet, llm: str = "gemini_3_flash_thinking"):
    """
    Run layout agent for all rows in the Slide Chunks sheet.
    Sequential processing (no parallelization).
    
    :param sheet: The gspread sheet object.
    :param llm: Language model to use.
    :return: None
    """
    worksheet_name = "Slide Chunks"
    ws, df = get_sheet_data_and_df(sheet, worksheet_name)
    
    # Get course name from Course info tab
    _, course_info_df = get_sheet_data_and_df(sheet, "Course info")
    course_name = course_info_df.loc[0, "Course Name"]
    
    print(f"\n{'='*80}")
    print(f"🚀 Starting Layout Agent")
    print(f"📚 Course: {course_name}")
    print(f"📊 Total rows: {len(df)}")
    print(f"{'='*80}\n")
    
    # Get Drive instance
    drive = get_drive_instance()
    if not drive:
        print("❌ Could not initialize Google Drive. Aborting.")
        return
    
    # Ensure layout_instructions and layout_evaluation columns exist
    if "layout_instructions" not in df.columns:
        df["layout_instructions"] = ""
    if "layout_evaluation" not in df.columns:
        df["layout_evaluation"] = ""
    
    # Filter rows that have final_graphics_definition but missing layout_instructions
    rows_to_process = []
    for index, row in df.iterrows():
        final_graphics_def = str(row.get("final_graphics_definition", "")).strip()
        layout_instructions = str(row.get("layout_instructions", "")).strip()
        
        # Skip if no final_graphics_definition
        if not final_graphics_def or final_graphics_def == "nan":
            continue
        
        # Skip if layout_instructions is already filled
        if layout_instructions and layout_instructions != "nan":
            continue
        
        rows_to_process.append((index, row))
    
    if not rows_to_process:
        print("✅ No rows to process. All rows already have layout_instructions or missing final_graphics_definition.")
        return
    
    print(f"📝 Processing {len(rows_to_process)} row(s) with missing layout_instructions\n")
    
    # Initialize progress bar
    total_tasks = len(rows_to_process)
    progress = SmartProgressBar(total_tasks, "Layout Agent")
    
    # Process rows sequentially (no parallelization)
    for idx, (index, row) in enumerate(rows_to_process):
        try:
            # Get row data
            topic_name = str(row.get("Topic", "")).strip()
            subtopic_name = str(row.get("Subtopic", "")).strip()
            slide_title = str(row.get("Slide Chunk Title", "")).strip()
            slide_content = str(row.get("Slide Chunk", "")).strip()
            final_graphics_def = str(row.get("final_graphics_definition", "")).strip()
            
            # Get previous slide's layout_instructions (from previous row)
            previous_slide_layout = None
            if idx > 0:
                # Get from previously processed row in this batch
                prev_index, prev_row = rows_to_process[idx - 1]
                previous_slide_layout = str(prev_row.get("layout_instructions", "")).strip()
                if not previous_slide_layout or previous_slide_layout == "nan":
                    previous_slide_layout = None
                # If not found in rows_to_process, check if we updated it in df
                if not previous_slide_layout:
                    prev_layout = str(df.at[prev_index, "layout_instructions"]).strip()
                    if prev_layout and prev_layout != "nan":
                        previous_slide_layout = prev_layout
            else:

                row_positions = df.index.tolist()
                try:
                    current_pos = row_positions.index(index)
                    if current_pos > 0:
                        prev_index = row_positions[current_pos - 1]
                        prev_layout = str(df.at[prev_index, "layout_instructions"]).strip()
                        if prev_layout and prev_layout != "nan":
                            previous_slide_layout = prev_layout
                except (ValueError, IndexError):
                    pass
            
            print(f"\n📋 Processing row {index + 2}: {slide_title}")
            if previous_slide_layout:
                print(f"📎 Using previous slide layout as context")
            
            # Generate layout
            layout_output, evaluation_breakdown = generate_layout_for_row(
                course_name=course_name,
                topic_name=topic_name,
                subtopic_name=subtopic_name,
                slide_title=slide_title,
                slide_content=slide_content,
                final_graphics_definition=final_graphics_def,
                previous_slide_layout=previous_slide_layout,
                drive=drive,
                llm=llm
            )
            
            # Update dataframe
            df.at[index, "layout_instructions"] = layout_output
            df.at[index, "layout_evaluation"] = evaluation_breakdown
            
            # Update progress
            progress.update()
            
            # Save after each row (sequential processing)
            print(f"💾 Saving progress after row {index + 2}...")
            save_to_sheet(ws, df)
            format_worksheet(ws)
            
        except Exception as e:
            print(f"❌ Error processing row {index}: {e}")
            df.at[index, "layout_instructions"] = f"ERROR: {str(e)}"
            df.at[index, "layout_evaluation"] = f"ERROR: {str(e)}"
            progress.update()
            
            # Save even on error
            save_to_sheet(ws, df)
            format_worksheet(ws)
    
    # Validation and retry logic
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        # Reload dataframe to get latest state
        ws, df = get_sheet_data_and_df(sheet, worksheet_name)
        
        # Find rows that need processing
        invalid_rows = []
        for index, row in df.iterrows():
            final_graphics_def = str(row.get("final_graphics_definition", "")).strip()
            layout_instructions = str(row.get("layout_instructions", "")).strip()
            
            # Skip if no final_graphics_definition
            if not final_graphics_def or final_graphics_def == "nan":
                continue
            
            # Check if layout_instructions is empty or error
            if not layout_instructions or layout_instructions == "nan" or layout_instructions.strip() == "" or layout_instructions.startswith("ERROR:"):
                invalid_rows.append((index, row))
        
        if not invalid_rows:
            break
        
        retry_count += 1
        print(f"\n⚠️ Found {len(invalid_rows)} rows with missing layout_instructions. Retrying (attempt {retry_count}/{max_retries})...")
        
        # Clear layout columns for invalid rows
        for index, row in invalid_rows:
            df.at[index, "layout_instructions"] = ""
            df.at[index, "layout_evaluation"] = ""
        
        # Save cleared state
        save_to_sheet(ws, df)
        format_worksheet(ws)
        
        # Retry processing invalid rows sequentially
        for idx, (index, row) in enumerate(invalid_rows):
            try:
                topic_name = str(row.get("Topic", "")).strip()
                subtopic_name = str(row.get("Subtopic", "")).strip()
                slide_title = str(row.get("Slide Chunk Title", "")).strip()
                slide_content = str(row.get("Slide Chunk", "")).strip()
                final_graphics_def = str(row.get("final_graphics_definition", "")).strip()
                
                # Get previous slide's layout_instructions
                previous_slide_layout = None
                if index > 0:
                    prev_row_data = df.iloc[index - 1]
                    prev_layout = str(prev_row_data.get("layout_instructions", "")).strip()
                    if prev_layout and prev_layout != "nan":
                        previous_slide_layout = prev_layout
                
                layout_output, evaluation_breakdown = generate_layout_for_row(
                    course_name=course_name,
                    topic_name=topic_name,
                    subtopic_name=subtopic_name,
                    slide_title=slide_title,
                    slide_content=slide_content,
                    final_graphics_definition=final_graphics_def,
                    previous_slide_layout=previous_slide_layout,
                    drive=drive,
                    llm=llm
                )
                
                df.at[index, "layout_instructions"] = layout_output
                df.at[index, "layout_evaluation"] = evaluation_breakdown
                
            except Exception as e:
                print(f"❌ Error on retry for row {index}: {e}")
                df.at[index, "layout_instructions"] = f"ERROR: {str(e)}"
                df.at[index, "layout_evaluation"] = f"ERROR: {str(e)}"
        
        # Save after retry
        save_to_sheet(ws, df)
        format_worksheet(ws)
    
    if retry_count > 0:
        # Check final state
        ws, df = get_sheet_data_and_df(sheet, worksheet_name)
        final_invalid = []
        for index, row in df.iterrows():
            final_graphics_def = str(row.get("final_graphics_definition", "")).strip()
            layout_instructions = str(row.get("layout_instructions", "")).strip()
            if (final_graphics_def and final_graphics_def != "nan" and
                (not layout_instructions or layout_instructions == "nan" or layout_instructions.strip() == "" or layout_instructions.startswith("ERROR:"))):
                final_invalid.append(index)
        
        if final_invalid:
            print(f"⚠️ After {retry_count} retry attempt(s), {len(final_invalid)} rows still have missing layout_instructions.")
        else:
            print(f"✅ All rows completed after {retry_count} retry attempt(s).")
    
    print("\n✅ Layout Agent complete.")


def delete_layout_columns(sheet):
    """
    Remove the 'layout_instructions' and 'layout_evaluation' columns from the worksheet.
    
    :param sheet: The gspread sheet object.
    :return: None
    """
    worksheet_name = "Slide Chunks"
    ws, df = get_sheet_data_and_df(sheet, worksheet_name)
    
    columns_to_delete = ["layout_instructions", "layout_evaluation"]
    deleted = []
    
    for col in columns_to_delete:
        if col in df.columns:
            df = df.drop(columns=[col])
            deleted.append(col)
    
    if deleted:
        clear_worksheet(ws)
        save_to_sheet(ws, df)
        print(f"🗑️ Deleted columns: {', '.join(deleted)} from '{worksheet_name}' worksheet")
    else:
        print(f"ℹ️ No layout columns found to delete in '{worksheet_name}' worksheet")

