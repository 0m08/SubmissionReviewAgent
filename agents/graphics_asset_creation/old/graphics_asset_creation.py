import os
import base64
from io import BytesIO
from PIL import Image
from typing import Optional, Tuple, Dict, List, Any
from google import genai
from google.genai import types
from dotenv import load_dotenv
import json
import streamlit as st
from pydantic import BaseModel, Field
from langsmith import traceable
# Import image editing functions for frame 2+ generation
from agents.image_editing.image_editing import generate_edited_image, review_edited_image
# Load environment variables
load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================
DEFAULT_GENERATOR_MODEL = "gemini-3-pro-image-preview" 
DEFAULT_REVIEWER_MODEL = "gemini-3-flash-preview"
MAX_CORRECTION_ROUNDS = 5

# Schema for Reviewer
class ReviewResult(BaseModel):
    approved: bool = Field(description="Boolean status. True if the generated graphics asset meets all technical, instructional, and scene definition requirements.")
    feedback: str = Field(description="Concise, objective description of any detected discrepancies, missing elements, or visual artifacts.")
    confidence_score: int = Field(description="Technical confidence score (0-100).")

def _get_client() -> Optional[genai.Client]:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("Missing GOOGLE_API_KEY")
        return None
    try:
        return genai.Client(api_key=api_key)
    except Exception as e:
        print(f"Failed to initialize Gemini client: {e}")
        return None

styling_guide = ("""
Design Tone - Balance of Informative, Professional, and Energetic.
Color Scheme - Primary Colors: White (#FFFFFF), Orange (#F05523), Marine Blue (#242052); Secondary Colors: Cool Grey (#F2F2F2), Blue Grey (#B9C9D1), Black (#000000)
Typefaces - Fira Sans
Appeal - Carefully select elements based on contrast, placement, angle, clarity, quality, and layout.

- Every image should be enhanced and color-corrected to make it look appealing.
- To convey the message or concept effectively, use the Principle of Visual Hierarchy.

Illustrations-type generation guidelines:
- They should be visually appealing and convey the message concisely.
- The illustrations should have a flat treatment while still conveying a sense of depth.
- The illustrations should not be overly childish, avoiding exaggerated expressions, comic proportions, and overly expressive poses.
- The illustrations should be simple and clear and should maintain a formal tone.

Diagrams and Charts generation guidelines:
- Choose the Right Chart/Diagram: Select the visual that best represents the data or concept (e.g., bar chart for comparisons, flow chart for processes).
- Keep it Simple: Avoid clutter; focus on the essential information.
- Clear Labels and Titles: Ensure all elements are clearly labeled and the chart/diagram has a descriptive title.
- Consistent Scale and Units: Use consistent scales and units to prevent misinterpretation.
- Highlight Key Data: Use color, annotations, or other visual cues to draw attention to important data points.
- Logical Flow: If using a diagram with a sequence, ensure the flow is logical and easy to follow.

Icons generation guidelines:
- Use simple line icons. Use a single color - orange (#F05523).
- Maintain the thickness and style of the icon.
- Label each icon with a relevant title in an orange (#F05523) solid base container.

Note:
- There shouldn’t be anything extra on-screen elements except the requirements of the graphics definition. No extra label, No extra graphic element, No unwanted movement
- The subject or object discussed in the graphics definition should be emphasized and occupy more space in the visual.

Checklist before the final result:
1) Check alignments
2) Check font size and color
3) Font casing in labels and naming
4) Correct spacing in a sentence-like space after, = . Etc.
5) Consistency in the design elements
6) Use of visual hierarchy principle
7) Tweak the contrast, clarity, levels, and exposure of any image to increase the appeal.

Master System Instructions
- Maximize Stage Utilization: Eliminate excessive blank space. Graphics must fill the "stage" to provide maximum detail. 
- Functional Perspective: Never use obscure or "artistic" angles that hide functionality. Use clear, top-down, or direct angles where screens and labels are legible. 
- Mobile-First Visibility: Icons and key elements must be large, bold, and high-contrast enough to be legible on a mobile phone in landscape mode. 
- Font Consistency: Maintain uniform font sizes across similar visual elements. Text must be readable on all screen sizes. 
- Layout Consistency: Repeated elements (e.g., fuses, boxes) must have identical sizing, alignment, and spacing. 
- Zero Tolerance for Typos: Thoroughly spell-check all text layers. (e.g., "Ventilation" not "Vantilation"). 
- Progressive Cognitive Load: Do not dump complex data at once. Start diagrams with minimal info and reveal details step-by-step. 
- Text Spacing & Legibility: Ensure distinct line height and letter spacing. Text must never feel cramped. 
- Annotation Uniformity: All arrows and pointers must share the same thickness, head size, and style. 
- De-Clutter: Remove all non-functional decoration. If a label, box, or icon does not serve a strict educational purpose, delete it. 
- Brand Palette Adherence: Strictly follow the brand color scheme. Use tints/tones of the palette for variety, but do not use off-brand default colors. 

""")

# =============================================================================
# DEBUG HELPER
# =============================================================================
def debug_print_history(history: List[types.Content], round_num: int):
    """Prints a readable summary of the conversation history."""
    print(f"\n{'='*20} DEBUG HISTORY CHECK (Round {round_num}) {'='*20}")
    if not history:
        print("History is empty.")
        return

    for i, content in enumerate(history):
        role = content.role.upper()
        print(f"\n[Turn {i+1}] Role: {role}")
        for p_idx, part in enumerate(content.parts):
            if hasattr(part, 'text') and part.text:
                preview = part.text[:100].replace('\n', ' ') + "..." if len(part.text) > 100 else part.text
                print(f"  - Part {p_idx+1}: [TEXT] \"{preview}\"")
            elif hasattr(part, 'inline_data') and part.inline_data:
                mime = part.inline_data.mime_type
                size = len(part.inline_data.data) if part.inline_data.data else 0
                print(f"  - Part {p_idx+1}: [IMAGE] Type: {mime}, Size: {size} bytes")
    print(f"{'='*60}\n")

# =============================================================================
# INTELLIGENT FRAME ANALYSIS
# =============================================================================

class FrameAnalysis(BaseModel):
    needs_multiple_frames: bool = Field(description="True if the graphics definition requires multiple frames for animation/transitions, False if a single complete image is sufficient.")
    num_frames: int = Field(description="Number of frames needed (1 if single frame, 2+ if animation sequence)")
    frames: List[str] = Field(description="Description of what each frame should contain. For multi-frame: cumulative descriptions (Frame 1: element A; Frame 2: element A + B; etc.). For single frame: one description with all elements.")
    reasoning: str = Field(description="Brief explanation of why this frame structure was chosen.")

def analyze_graphics_definition_for_frames(graphics_definition: str) -> List[str]:
    """
    Use LLM to intelligently analyze the graphics definition and determine
    if multiple frames are needed and what each frame should contain.
    
    Returns a list of frame descriptions. If only one frame is needed, returns
    a single-item list.
    """
    try:
        client = _get_client()
        if client is None:
            print("Warning: Could not get API client for frame analysis. Defaulting to single frame.")
            return ["Complete scene with all elements visible"]
        
        analysis_prompt = f"""<task>
Analyze the following graphics definition to determine if it REQUIRES multiple frames (animation) or can be handled with a SINGLE comprehensive frame.
DEFAULT TO SINGLE FRAME unless there is EXPLICIT and CLEAR evidence of sequential animation requirements.
</task>

<graphics_definition>
{graphics_definition}
</graphics_definition>

<strict_decision_criteria>
**SINGLE FRAME (DEFAULT)** - Use unless criteria below are met:
- All visual elements can be shown together
- The concept is explanatory, descriptive, or comparative
- No explicit temporal sequence or animation mentioned
- Elements are meant to coexist in the same scene
- Diagrams, charts, static illustrations, comparisons
- Even if showing "before/after", "comparison", or "multiple states" - use ONE frame with all states visible

**MULTIPLE FRAMES (ONLY IF ALL CONDITIONS MET):**
1. Graphics definition EXPLICITLY contains temporal keywords: "appears first", "then appears", "followed by", "next", "after that", "finally appears", "fades in", "transitions to"
2. Graphics Type is explicitly marked as "Animation" or "Animated Sequence"
3. The specification describes a TIME-BASED PROGRESSION where elements must appear sequentially over time
4. It is IMPOSSIBLE to show all elements together in one frame without losing the temporal narrative

**CRITICAL RULES:**
- If unsure → Choose SINGLE FRAME
- "Shows X, Y, and Z" → SINGLE FRAME with all three
- "Different states/scenarios" → SINGLE FRAME showing all states side-by-side
- "Process flow" or "workflow" → SINGLE FRAME flowchart
- "Before and after" → SINGLE FRAME split-screen
- Only choose multiple frames if the definition says "X appears, THEN Y appears, THEN Z appears" with clear temporal sequence

<analysis_approach>
1. **Scan for Temporal Keywords** - Look for explicit time-based language ("then", "next", "after", "followed by")
2. **Check Graphics Type** - Is it explicitly marked "Animation"?
3. **Evaluate if Single Frame Works** - Can all elements be shown together effectively?
4. **Default to Single** - Unless all three checks above indicate animation

**If multiple frames are needed, limit to 2-3 frames maximum. Avoid over-segmentation.**
</analysis_approach>

<frame_planning_process>
If SINGLE FRAME (default):
1. Include ALL visual elements in one comprehensive image
2. Use layout techniques: side-by-side, panels, split-screen, labeled sections
3. Ensure clarity through visual hierarchy and organization

If MULTIPLE FRAMES (rare - only when explicitly animated):
1. Keep to 2-3 frames maximum
2. Plan progression:
   - Frame 1: Initial state/elements
   - Frame 2: Next state/elements appearing
   - Frame 3 (if needed): Final state
3. Each frame must serve a clear temporal purpose
</frame_planning_process>

<output_requirements>
Return a structured analysis with:
- needs_multiple_frames: boolean (true ONLY if explicit animation requirements, false otherwise)
- num_frames: number (1 for most cases, 2-3 maximum if truly animated)
- frames: array of frame descriptions
- reasoning: explain why single or multiple frames chosen
</output_requirements>

<examples>
Example 1 - SINGLE FRAME (Multiple Elements):
Sentence: "The system includes servers, databases, and user interfaces"
Graphics Type: Diagram
Specification: Show servers, databases, user interfaces with connections

Analysis: Multiple elements but NO temporal sequence → SINGLE FRAME

Output: {{
  "needs_multiple_frames": false,
  "num_frames": 1,
  "frames": [
    "Complete system diagram showing all components (servers, databases, user interfaces) with connecting lines and labels"
  ],
  "reasoning": "No temporal keywords or animation requirement. All elements can be shown together in one comprehensive diagram."
}}

Example 2 - SINGLE FRAME (Process Flow):
Sentence: "Data flows from input through processing to output"
Graphics Type: Flowchart
Specification: Input box, processing box, output box, arrows showing flow

Analysis: Shows a process but NOT time-based animation → SINGLE FRAME flowchart

Output: {{
  "needs_multiple_frames": false,
  "num_frames": 1,
  "frames": [
    "Complete flowchart with input box, arrows pointing to processing box, arrows pointing to output box - all visible simultaneously"
  ],
  "reasoning": "Process flow is best shown as complete flowchart in single frame. No animation keywords present."
}}

Example 3 - MULTIPLE FRAMES (Explicit Animation):
Sentence: "Watch as the signal travels through the circuit"
Graphics Type: Animation
Specification: Circuit appears first, THEN signal starts at input, THEN signal travels through components, FINALLY signal reaches output

Analysis: Explicit temporal keywords ("THEN", "FINALLY") + marked as "Animation" → MULTIPLE FRAMES

Output: {{
  "needs_multiple_frames": true,
  "num_frames": 3,
  "frames": [
    "Frame 1: Complete circuit diagram without signal",
    "Frame 2: Circuit with signal at input and midway through components",
    "Frame 3: Circuit with signal reaching output"
  ],
  "reasoning": "Explicit temporal sequence with 'THEN' and 'FINALLY' keywords, plus marked as Animation type. True time-based progression required."
}}

Example 4 - SINGLE FRAME (Before/After Comparison):
Sentence: "This shows the difference between damaged and healthy tissue"
Graphics Type: Illustration
Specification: Show damaged tissue and healthy tissue for comparison

Analysis: Comparison of states but NOT temporal progression → SINGLE FRAME with split view

Output: {{
  "needs_multiple_frames": false,
  "num_frames": 1,
  "frames": [
    "Split-screen illustration: left side shows damaged tissue with labels, right side shows healthy tissue with labels"
  ],
  "reasoning": "Comparison scenario best shown in single frame with side-by-side layout. No temporal progression needed."
}}
</examples>

<final_reminder>
**BIAS TOWARD SINGLE FRAME** - Choose multiple frames ONLY if:
1. Explicit temporal keywords present ("then", "next", "after", "followed by")
2. Type explicitly marked "Animation"
3. Cannot show all elements together effectively

When in doubt → SINGLE FRAME is always the safer choice.
</final_reminder>

Analyze the provided graphics definition and return your structured frame generation plan.
"""
        
        print("Analyzing graphics definition to determine frame requirements...")
        
        response = client.models.generate_content(
            model=DEFAULT_REVIEWER_MODEL,  # Use Flash for fast analysis
            contents=[types.Content(role="user", parts=[types.Part.from_text(text=analysis_prompt)])],
            config=types.GenerateContentConfig(
                temperature=0.1,  # Lower temperature for more conservative/consistent decisions
                response_mime_type="application/json",
                response_schema=FrameAnalysis
            )
        )
        
        # Parse response
        if hasattr(response, 'parsed') and response.parsed:
            result = response.parsed
            if isinstance(result, dict):
                needs_multiple = result.get('needs_multiple_frames', False)
                frames = result.get('frames', [])
                reasoning = result.get('reasoning', '')
            else:
                needs_multiple = result.needs_multiple_frames
                frames = result.frames
                reasoning = result.reasoning
        else:
            data = json.loads(response.text)
            needs_multiple = data.get('needs_multiple_frames', False)
            frames = data.get('frames', [])
            reasoning = data.get('reasoning', '')
        
        print(f"Analysis: {'Multiple frames needed' if needs_multiple else 'Single frame sufficient'}")
        print(f"Reasoning: {reasoning}")
        print(f"Frames to generate: {len(frames)}")
        
        if not frames:
            print("Warning: No frames returned from analysis. Defaulting to single frame.")
            return ["Complete scene with all elements visible"]
        
        return frames
        
    except Exception as e:
        print(f"Error analyzing graphics definition: {e}")
        print("Defaulting to single complete frame.")
        return ["Complete scene with all elements visible"]

# =============================================================================
# PROFESSIONAL / TECHNICAL PROMPTS
# =============================================================================

def get_initial_prompt(graphics_definition: str, stage_description: str = None) -> str:
    """
    Formal prompt for the Generator to create graphics asset from definition and reference images.
    If stage_description is provided, generates a specific frame for that transition stage.
    """
    stage_instruction = ""
    if stage_description:
        stage_instruction = f"""\n<transition_stage>
Generate ONLY the visual state for this specific transition stage:
{stage_description}

Include ONLY the elements that should be visible at this stage. Do not show elements that appear in later stages.
</transition_stage>\n"""
    
    prompt = f"""Generate a graphics asset frame based on the scene definition and reference images.

<scene_definition>
{graphics_definition}
</scene_definition>
{stage_instruction}
<reference_images_usage>
Reference images show EXACT technical components to reproduce (not inspiration).
- Reproduce shapes, labels, text, proportions, connections, colors with 100% accuracy
- You can adapt scene layout/background/lighting
- You cannot simplify or modify the technical components themselves
</reference_images_usage>

<requirements>
1. Reproduce reference components exactly as shown (all details, labels, proportions)
2. Include all elements from scene definition  
3. Arrange per specified layout while preserving component accuracy
4. If previous frame exists: maintain exact visual consistency, only add new elements
5. Professional quality with consistent lighting and perspective
</requirements>

Generate the image frame with reference components reproduced exactly.
"""
    return prompt


def get_feedback_prompt(correction_feedback: str, stage_description: str = None) -> str:
    """
    Formal feedback prompt for the Generator.
    """
    stage_reminder = ""
    if stage_description:
        stage_reminder = f"\n\nRemember: This frame should show only: {stage_description}"
    
    return f"""<status>
The previous output was rejected during quality assurance review.
</status>

<detected_issues>
{correction_feedback}
</detected_issues>

<directive>
Regenerate the frame. Correct the specific issues listed above while maintaining adherence to all scene definition requirements for this stage.{stage_reminder}
</directive>"""


def get_reviewer_prompt(graphics_definition: str, stage_description: str = None, custom_criteria: Optional[str] = None) -> str:
    """
    Formal prompt for the Reviewer to assess the generated graphics asset.
    """
    stage_instruction = ""
    if stage_description:
        stage_instruction = f"""\n<transition_stage_requirements>
This frame represents a specific transition stage:
{stage_description}

Verify that ONLY the elements specified for this stage are present. Elements from later stages should NOT appear.
</transition_stage_requirements>\n"""
    
    prompt = f"""Review the generated graphics frame against requirements.

<scene_definition>
{graphics_definition}
</scene_definition>
{stage_instruction}
<review_criteria>
1. **Technical Accuracy** (Highest Priority): Reference components match exactly (shapes, labels, proportions, colors, details)
2. **Completeness**: All required elements present for this stage
3. **Stage Correctness**: Only this stage's elements shown (no premature elements)
4. **Frame Consistency**: Maintains previous frame's background/lighting/angle/style if applicable
5. **Layout**: Elements arranged per specifications
6. **Quality**: Professional appearance, no artifacts or inconsistencies
"""
    
    if custom_criteria and custom_criteria.strip():
        prompt += f"7. **Custom**: {custom_criteria}\n"
    
    prompt += """</review_criteria>

REJECT if: Technical components simplified/stylized, missing details/labels, incorrect proportions, or any required element missing.

Return JSON with:
- 'approved': true only if all criteria met
- 'feedback': Specific issues if rejected
- 'confidence_score': 0-100
"""
    return prompt

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def prepare_image_for_gemini(image: Image.Image, max_dimension: int = 2048) -> bytes:
    """Convert PIL Image to bytes for Gemini API with size optimization."""
    buffered = BytesIO()
    save_img = image.copy()
    
    # Resize if image is too large to prevent API errors
    width, height = save_img.size
    if width > max_dimension or height > max_dimension:
        ratio = min(max_dimension / width, max_dimension / height)
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        save_img = save_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        print(f"  Resized image from {width}x{height} to {new_width}x{new_height}")
    
    if save_img.mode != 'RGB':
        save_img = save_img.convert('RGB')
    
    # Use JPEG with good quality to reduce size while maintaining quality
    save_img.save(buffered, format="JPEG", quality=90)
    return buffered.getvalue()

def image_from_base64(base64_string: str) -> Image.Image:
    """Convert base64 string to PIL Image."""
    if ',' in base64_string:
        base64_string = base64_string.split(',', 1)[1]
    image_bytes = base64.b64decode(base64_string)
    return Image.open(BytesIO(image_bytes))

# =============================================================================
# GENERATOR AGENT
# =============================================================================

@traceable(
    metadata={
        "agent_name": "graphics_asset_creation",
        "step_name": "Generator",
        "function_name": "generate_graphics_asset",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous")
    }
)
def generate_graphics_asset(
    graphics_definition: str,
    reference_images: List[Image.Image] = None,
    correction_feedback: Optional[str] = None,
    chat_session = None,
    client = None,
    aspect_ratio: Optional[str] = None,
    image_size: str = "1K",
    stage_description: str = None,
    previous_frame: Image.Image = None
) -> Tuple[Image.Image, Dict, object, object]:
    """
    Generate graphics asset based on scene definition and optional reference images.
    
    Args:
        graphics_definition: Text description of the scene with all visual elements
        reference_images: List of reference images for style/concept guidance (optional)
        correction_feedback: Feedback from reviewer for regeneration (optional)
        chat_session: Chat session object for multi-turn generation (optional)
        client: Gemini client object (optional, will create if not provided)
        aspect_ratio: Aspect ratio for the generated image (optional)
        image_size: Size of the image to generate (default: "1K")
        stage_description: Specific transition stage to generate (optional)
        previous_frame: Previous frame to maintain visual consistency (optional)
    
    Returns:
        Tuple of (generated_image, metadata, chat_session, client)
    """
    try:
        if client is None:
            client = _get_client()
            if client is None:
                raise ValueError("Missing GOOGLE_API_KEY.")
        
        print(f"Starting graphics asset generation...")
        
        # Build image_config dynamically
        image_config_params = {"image_size": image_size}
        if aspect_ratio:
            image_config_params["aspect_ratio"] = aspect_ratio
        
        if not chat_session:
            # Round 1: Create chat session and send initial request
            print("Creating new chat session with full context...")
            if stage_description:
                print(f"  Generating frame for: {stage_description[:60]}...")
            
            chat_session = client.chats.create(
                model=DEFAULT_GENERATOR_MODEL,
                config=types.GenerateContentConfig(
                    system_instruction=f"Follow these styling guidelines carefully for generations: \n\n{styling_guide}",
                    response_modalities=['TEXT', 'IMAGE'],
                    tools=[{"google_search": {}}],
                    image_config=types.ImageConfig(**image_config_params)
                )
            )
            
            prompt = get_initial_prompt(graphics_definition, stage_description)
            
            # Build parts list: prompt first, then reference images
            parts = [types.Part.from_text(text=prompt)]
            
            # CRITICAL: Add previous frame FIRST to maintain visual consistency
            if previous_frame:
                print(f"Adding previous frame as reference for visual consistency...")
                prev_frame_bytes = prepare_image_for_gemini(previous_frame)
                parts.append(
                    types.Part.from_bytes(data=prev_frame_bytes, mime_type="image/jpeg")
                )
                print(f"  ✓ Previous frame attached - ensures same background, lighting, style")
            
            # Add reference images if provided (for style/concept guidance, limit to 3)
            if reference_images:
                limited_refs = reference_images[:3]
                print(f"Adding {len(limited_refs)} reference image(s) for style/concept guidance...")
                for idx, ref_img in enumerate(limited_refs):
                    image_bytes = prepare_image_for_gemini(ref_img)
                    parts.append(
                        types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
                    )
                if len(reference_images) > 3:
                    print(f"  Note: Using first 3 of {len(reference_images)} reference images")
                print(f"  ✓ {len(limited_refs)} reference images attached as inspiration")
            
            response = chat_session.send_message(parts)
        else:
            # Round 2+: Send feedback to existing chat session
            print("Sending feedback to existing chat session...")
            prompt = get_feedback_prompt(correction_feedback, stage_description)
            response = chat_session.send_message(prompt)
        
        if not response.candidates or not response.candidates[0].content:
            raise ValueError("Model returned no content.")

        # Extract Image & Raw Content
        generated_image = None
        model_content = response.candidates[0].content 
        
        # Root cause fix: parts can be None even when content exists
        # This happens when model returns malformed response or refuses generation
        if not hasattr(model_content, 'parts') or model_content.parts is None:
            raise ValueError(f"Model returned content without parts. This may be due to content policy violation or malformed response. Response: {str(response)[:200]}")
        
        if hasattr(model_content, 'parts'):
            for part in model_content.parts:
                if hasattr(part, 'inline_data') and part.inline_data:
                    try:
                        image_data = part.inline_data.data
                        generated_image = Image.open(BytesIO(image_data))
                        break
                    except Exception as e:
                        print(f"Error processing inline data: {e}")

        if generated_image is None and hasattr(response, 'text'):
            try:
                generated_image = image_from_base64(response.text)
            except:
                pass
                
        if generated_image is None:
            raise ValueError("No generated image found. The model may have refused the request.")
        
        # Metadata
        metadata = {
            "model": DEFAULT_GENERATOR_MODEL,
            "input_tokens": response.usage_metadata.prompt_token_count if hasattr(response, 'usage_metadata') else 0,
            "output_tokens": response.usage_metadata.candidates_token_count if hasattr(response, 'usage_metadata') else 0,
        }
        
        print("✓ Graphics asset generated")
        return generated_image, metadata, chat_session, client
        
    except Exception as e:
        st.error(f"Graphics asset generation failed: {str(e)}")
        raise

# =============================================================================
# REVIEWER AGENT
# =============================================================================

@traceable(
    metadata={
        "agent_name": "graphics_asset_creation",
        "step_name": "Reviewer",
        "function_name": "review_graphics_asset",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous")
    }
)
def review_graphics_asset(
    generated_image: Image.Image,
    graphics_definition: str,
    reference_images: List[Image.Image] = None,
    previous_feedback: Optional[str] = None,
    custom_criteria: Optional[str] = None,
    stage_description: str = None
) -> Tuple[bool, Optional[str], Dict]:
    """
    Review the generated graphics asset against the scene definition.
    
    Args:
        generated_image: The generated graphics asset to review
        graphics_definition: The scene definition requirements
        reference_images: Reference images that were provided (optional, for context)
        previous_feedback: Previous reviewer feedback (optional)
        custom_criteria: Additional custom review criteria (optional)
        stage_description: Specific transition stage for this frame (optional)
    
    Returns:
        Tuple of (is_approved, feedback, metadata)
    """
    try:
        client = _get_client()
        if client is None:
            raise ValueError("Missing GOOGLE_API_KEY.")
        
        print("Reviewing generated graphics asset...")
        
        generated_bytes = prepare_image_for_gemini(generated_image)
        
        base_prompt = get_reviewer_prompt(graphics_definition, stage_description, custom_criteria)
        if previous_feedback:
            base_prompt += f"\n\n<prior_defect_correction>\nThe previous iteration failed due to: {previous_feedback}.\nVerify that these specific issues have been resolved.\n</prior_defect_correction>"

        # Build parts: prompt + generated image + optional reference images for context
        parts = [
            types.Part.from_text(text=base_prompt),
            types.Part.from_bytes(data=generated_bytes, mime_type="image/png")
        ]
        
        # Add reference images for context (reviewer can see what references were available)
        if reference_images:
            for ref_img in reference_images:
                ref_bytes = prepare_image_for_gemini(ref_img)
                parts.append(types.Part.from_bytes(data=ref_bytes, mime_type="image/png"))

        # Structured Output with Low Temperature for Consistency
        response = client.models.generate_content(
            model=DEFAULT_REVIEWER_MODEL,
            contents=[types.Content(role="user", parts=parts)],
            config=types.GenerateContentConfig(
                system_instruction=f"Follow these styling guidelines carefully for review: \n\n{styling_guide}",
                temperature=0.2, 
                response_mime_type="application/json",
                response_schema=ReviewResult
            )
        )
        
        try:
            if hasattr(response, 'parsed') and response.parsed:
                result = response.parsed
                if isinstance(result, dict):
                    approved = result.get('approved', False)
                    feedback = result.get('feedback', '')
                    score = result.get('confidence_score', 0)
                else:
                    approved = result.approved
                    feedback = result.feedback
                    score = result.confidence_score
            else:
                data = json.loads(response.text)
                approved = data.get("approved", False)
                feedback = data.get("feedback", "")
                score = data.get("confidence_score", 0)
        except Exception as e:
            print(f"Parsing fallback: {e}")
            data = json.loads(response.text)
            approved = data["approved"]
            feedback = data["feedback"]
            score = 0
        
        metadata = {
            "model": DEFAULT_REVIEWER_MODEL,
            "input_tokens": response.usage_metadata.prompt_token_count if hasattr(response, 'usage_metadata') else 0,
            "output_tokens": response.usage_metadata.candidates_token_count if hasattr(response, 'usage_metadata') else 0,
            "confidence_score": score
        }
        
        return approved, feedback, metadata
        
    except Exception as e:
        st.error(f"Review failed: {str(e)}")
        raise

# =============================================================================
# FRAME EDITING WITH REFERENCES (for frames 2+)
# =============================================================================

@traceable(
    metadata={
        "agent_name": "graphics_asset_creation",
        "step_name": "Frame Editor with References",
        "function_name": "generate_frame_edit_with_references",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous")
    }
)
def generate_frame_edit_with_references(
    previous_frame: Image.Image,
    frame_description: str,
    reference_images: List[Image.Image] = None,
    correction_feedback: Optional[str] = None,
    chat_session = None,
    client = None,
    aspect_ratio: Optional[str] = None,
    image_size: str = "1K"
) -> Tuple[Image.Image, Dict, object, object]:
    """
    Edit the previous frame to create the next frame in the sequence.
    Includes reference images in context for style consistency.
    
    Args:
        previous_frame: The previous frame to edit
        frame_description: Description of what this frame should show
        reference_images: Original reference images for style/concept guidance
        correction_feedback: Feedback from reviewer for regeneration
        chat_session: Chat session object for multi-turn generation
        client: Gemini client object (optional, will create if not provided)
        aspect_ratio: Aspect ratio for the generated image
        image_size: Size of the image to generate
    
    Returns:
        Tuple of (generated_image, metadata, chat_session, client)
    """
    try:
        if client is None:
            client = _get_client()
            if client is None:
                raise ValueError("Missing GOOGLE_API_KEY.")
        
        print(f"Starting frame editing with reference images in context...")
        
        # Build image_config dynamically
        image_config_params = {"image_size": image_size}
        if aspect_ratio:
            image_config_params["aspect_ratio"] = aspect_ratio
        
        if not chat_session:
            # Round 1: Create chat session with full context
            print("Creating new chat session for frame editing...")
            
            chat_session = client.chats.create(
                model=DEFAULT_GENERATOR_MODEL,
                config=types.GenerateContentConfig(
                    system_instruction=f"Follow these styling guidelines carefully for generations: \n\n{styling_guide}",
                    response_modalities=['TEXT', 'IMAGE'],
                    tools=[{"google_search": {}}],
                    image_config=types.ImageConfig(**image_config_params)
                )
            )
            
            prompt = f"""Edit the previous frame to show the next animation stage.

<frame_requirements>
{frame_description}
</frame_requirements>

<reference_usage>
Reference images show EXACT components to reproduce (not inspiration).
- New technical components must match references 100% (shapes, labels, proportions, colors)
- You can adjust scene composition but not the components themselves
</reference_usage>

<constraints>
1. Maintain exact visual consistency with previous frame (background, lighting, angle, style)
2. Reproduce new technical components exactly from references
3. Add/modify only elements specified in frame requirements
4. Keep all existing elements unchanged
5. Ensure smooth transition from previous frame
</constraints>

Generate the edited frame with reference components reproduced exactly.
"""
            
            # Build parts list: previous frame + prompt + reference images
            parts = [types.Part.from_bytes(
                data=prepare_image_for_gemini(previous_frame), 
                mime_type="image/jpeg"
            )]
            parts.append(types.Part.from_text(text=prompt))
            
            # Add reference images for style consistency (limit to 3 to avoid API overload)
            if reference_images:
                # Limit to first 3 reference images to prevent API errors
                limited_refs = reference_images[:3]
                print(f"Adding {len(limited_refs)} reference image(s) for style consistency...")
                for idx, ref_img in enumerate(limited_refs):
                    image_bytes = prepare_image_for_gemini(ref_img)
                    parts.append(
                        types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
                    )
                if len(reference_images) > 3:
                    print(f"  Note: Using first 3 of {len(reference_images)} reference images to prevent API overload")
                print(f"  ✓ {len(limited_refs)} reference images attached")
            
            response = chat_session.send_message(parts)
        else:
            # Round 2+: Send feedback to existing chat session
            print("Sending feedback to existing chat session...")
            feedback_prompt = f"""<status>
The previous frame edit was rejected during review.
</status>

<detected_issues>
{correction_feedback}
</detected_issues>

<directive>
Re-edit the previous frame. Address the specific issues listed above while maintaining strict visual consistency with the base frame.
Ensure the frame requirements are met: {frame_description}
</directive>"""
            
            response = chat_session.send_message(feedback_prompt)

        if not response.candidates or not response.candidates[0].content:
            raise ValueError("Model returned no content.")

        # Extract Image & Raw Content
        generated_image = None
        model_content = response.candidates[0].content 
        
        # Root cause fix: parts can be None even when content exists
        if not hasattr(model_content, 'parts') or model_content.parts is None:
            raise ValueError(f"Model returned content without parts. This may be due to content policy violation or malformed response. Response: {str(response)[:200]}")
        
        if hasattr(model_content, 'parts'):
            for part in model_content.parts:
                if hasattr(part, 'inline_data') and part.inline_data:
                    try:
                        image_data = part.inline_data.data
                        generated_image = Image.open(BytesIO(image_data))
                        break
                    except Exception as e:
                        print(f"Error processing inline data: {e}")

        if generated_image is None and hasattr(response, 'text'):
            try:
                generated_image = image_from_base64(response.text)
            except:
                pass
                
        if generated_image is None:
            raise ValueError("No generated image found. The model may have refused the request.")
        
        # Metadata
        metadata = {
            "model": DEFAULT_GENERATOR_MODEL,
            "input_tokens": response.usage_metadata.prompt_token_count if hasattr(response, 'usage_metadata') else 0,
            "output_tokens": response.usage_metadata.candidates_token_count if hasattr(response, 'usage_metadata') else 0,
        }
        
        print("✓ Frame edit with references completed")
        return generated_image, metadata, chat_session, client
        
    except Exception as e:
        st.error(f"Frame editing with references failed: {str(e)}")
        raise

# =============================================================================
# MAIN LOOP
# =============================================================================

@traceable(
    metadata={
        "agent_name": "graphics_asset_creation",
        "step_name": "Main Loop",
        "function_name": "graphics_asset_creation_with_review_loop",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous")
    }
)
def graphics_asset_creation_with_review_loop(
    graphics_definition: str,
    reference_images: List[Image.Image] = None,
    quick_mode: bool = False,
    custom_criteria: Optional[str] = None,
    aspect_ratio: Optional[str] = None,
    image_size: str = "1K",
    generate_transitions: bool = True,
    pre_analyzed_frames: Optional[List[str]] = None
) -> Tuple[List[Image.Image], list, List[object], List[object]]:
    """
    Main loop for graphics asset creation with review and correction.
    Generates separate frames for each transition stage.
    
    Args:
        graphics_definition: Scene definition with all visual requirements
        reference_images: List of reference images to incorporate
        quick_mode: If True, skip review loop and generate once
        custom_criteria: Additional review criteria
        aspect_ratio: Aspect ratio for generated image
        image_size: Size of the image to generate
        generate_transitions: If True, generate separate frames for each transition stage
        pre_analyzed_frames: Optional pre-analyzed frame descriptions (if already analyzed in UI)
    
    Returns:
        Tuple of (list_of_final_images, ui_history_log, list_of_chat_sessions, list_of_clients)
    """
    print(f"Starting graphics asset creation with review loop... (Quick Mode: {quick_mode})")
    
    # Use pre-analyzed frames if provided, otherwise analyze now
    frame_descriptions = []
    if pre_analyzed_frames:
        frame_descriptions = pre_analyzed_frames
        print(f"Using pre-analyzed frame plan: {len(frame_descriptions)} frame(s)")
    elif generate_transitions:
        frame_descriptions = analyze_graphics_definition_for_frames(graphics_definition)
        print(f"Determined {len(frame_descriptions)} frame(s) needed")
    else:
        frame_descriptions = ["Complete scene with all elements visible"]
        print("Transition generation disabled - generating single complete frame")
    
    for i, frame_desc in enumerate(frame_descriptions):
        print(f"  Frame {i+1}: {frame_desc[:80]}...")
    
    all_final_images = []
    all_ui_history_log = []
    all_chat_sessions = []
    all_clients = []
    
    # =========================================================================
    # CRITICAL: Two-stage generation approach
    # - FRAME 1: Generate from scratch using Nano Banana Pro
    # - FRAMES 2+: Edit the previous frame using image editing workflow
    # - This ensures EXACT visual consistency by modifying the same base image
    # - Review loop validates each frame before proceeding to the next
    # =========================================================================
    for stage_idx, stage_desc in enumerate(frame_descriptions):
        stage_num = stage_idx + 1
        is_first_frame = (stage_idx == 0)
        
        print(f"\n{'='*60}")
        if is_first_frame:
            print(f"GENERATING FRAME 1 (Initial Generation)")
        else:
            print(f"EDITING TO CREATE FRAME {stage_num} (Image Editing from Frame {stage_num - 1})")
        if stage_desc:
            print(f"Frame content: {stage_desc[:100]}...")
        print(f"{'='*60}\n")
        
        chat_session = None
        client = None
        correction_feedback = None
        generated_image = None
        ui_history_log = []
        
        # =======================================================================
        # FRAME 1: Generate from scratch
        # FRAMES 2+: Edit the previous frame with new frame instructions
        # =======================================================================
        
        # Quick Mode: Single generation/edit without review
        if quick_mode:
            print(f"Quick Mode enabled for frame {stage_num} - skipping review loop")
            with st.spinner(f"{'Generating' if is_first_frame else 'Editing'} frame {stage_num}/{len(frame_descriptions)}..."):
                
                if is_first_frame:
                    # FRAME 1: Generate from scratch
                    generated_image, gen_metadata, chat_session, client = generate_graphics_asset(
                        graphics_definition=graphics_definition,
                        reference_images=reference_images,
                        correction_feedback=None,
                        chat_session=None,
                        client=None,
                        aspect_ratio=aspect_ratio,
                        image_size=image_size,
                        stage_description=stage_desc,
                        previous_frame=None
                    )
                else:
                    # FRAMES 2+: Edit previous frame with frame-specific instructions
                    # Include reference images in the context for style consistency
                    previous_frame = all_final_images[-1]
                    
                    # Build custom user content with: previous frame + instructions + reference images
                    generated_image, gen_metadata, chat_session, client = generate_frame_edit_with_references(
                        previous_frame=previous_frame,
                        frame_description=stage_desc,
                        reference_images=reference_images,
                        correction_feedback=None,
                        chat_session=None,
                        client=None,
                        aspect_ratio=aspect_ratio,
                        image_size=image_size
                    )
            
            ui_history_log.append({
                "round": 1,
                "type": "generation" if is_first_frame else "editing",
                "generated_image": generated_image,
                "metadata": gen_metadata,
                "stage": stage_num,
                "stage_description": stage_desc
            })
            
            print(f"✓ Quick mode {'generation' if is_first_frame else 'editing'} complete for frame {stage_num}")
            all_final_images.append(generated_image)
            all_ui_history_log.extend(ui_history_log)
            all_chat_sessions.append(chat_session)
            all_clients.append(client)
            continue
        
        # Normal Mode: Full review loop for this frame
        for round_num in range(MAX_CORRECTION_ROUNDS):
            print(f"\nFrame {stage_num} - Round {round_num + 1}/{MAX_CORRECTION_ROUNDS}")
            
            # 1. Generate or Edit
            with st.spinner(f"{'Generating' if is_first_frame else 'Editing'} frame {stage_num}/{len(frame_descriptions)} (Round {round_num + 1})..."):
                
                if is_first_frame:
                    # FRAME 1: Generate from scratch with full context
                    generated_image, gen_metadata, chat_session, client = generate_graphics_asset(
                        graphics_definition=graphics_definition,
                        reference_images=reference_images,
                        correction_feedback=correction_feedback,
                        chat_session=chat_session,
                        client=client,
                        aspect_ratio=aspect_ratio,
                        image_size=image_size,
                        stage_description=stage_desc,
                        previous_frame=None
                    )
                else:
                    # FRAMES 2+: Edit previous frame with frame-specific instructions
                    # Include reference images in the context for style consistency
                    previous_frame = all_final_images[-1]
                    
                    # Build custom user content with: previous frame + instructions + reference images
                    generated_image, gen_metadata, chat_session, client = generate_frame_edit_with_references(
                        previous_frame=previous_frame,
                        frame_description=stage_desc,
                        reference_images=reference_images,
                        correction_feedback=correction_feedback,
                        chat_session=chat_session,
                        client=client,
                        aspect_ratio=aspect_ratio,
                        image_size=image_size
                    )
            
            ui_history_log.append({
                "round": round_num + 1,
                "type": "generation" if is_first_frame else "editing",
                "generated_image": generated_image,
                "metadata": gen_metadata,
                "stage": stage_num,
                "stage_description": stage_desc
            })
            
            # 3. Review
            with st.spinner(f"Reviewing frame {stage_num}..."):
                if is_first_frame:
                    # Review generated frame
                    is_approved, feedback, review_metadata = review_graphics_asset(
                        generated_image=generated_image,
                        graphics_definition=graphics_definition,
                        reference_images=reference_images,
                        previous_feedback=correction_feedback,
                        custom_criteria=custom_criteria,
                        stage_description=stage_desc
                    )
                else:
                    # Review edited frame
                    previous_frame = all_final_images[-1]
                    editing_instructions = {
                        "additional_comments": f"Edit this image to show: {stage_desc}\n\nMaintain the exact same background, lighting, camera angle, and style."
                    }
                    is_approved, feedback, review_metadata = review_edited_image(
                        edited_image=generated_image,
                        reference_image=previous_frame,
                        editing_instructions=editing_instructions,
                        previous_feedback=correction_feedback
                    )
            
            ui_history_log.append({
                "round": round_num + 1,
                "type": "review",
                "approved": is_approved,
                "feedback": feedback,
                "metadata": review_metadata,
                "stage": stage_num,
                "stage_description": stage_desc
            })
            
            if is_approved:
                print(f"✓ Frame {stage_num} approved in round {round_num + 1}")
                all_final_images.append(generated_image)
                all_ui_history_log.extend(ui_history_log)
                all_chat_sessions.append(chat_session)
                all_clients.append(client)
                break
            else:
                print(f"Frame {stage_num} Round {round_num + 1} Rejected: {feedback[:100]}...")
                correction_feedback = feedback
        else:
            # Max rounds reached for this frame
            print(f"Max rounds reached for frame {stage_num}, using last generated image")
            all_final_images.append(generated_image)
            all_ui_history_log.extend(ui_history_log)
            all_chat_sessions.append(chat_session)
            all_clients.append(client)
    
    print(f"\n{'='*60}")
    print(f"✓ Completed generation of {len(all_final_images)} frame(s)")
    print(f"{'='*60}\n")
    
    return all_final_images, all_ui_history_log, all_chat_sessions, all_clients
