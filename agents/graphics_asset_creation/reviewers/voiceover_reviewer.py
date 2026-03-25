import os
import json
from io import BytesIO
from PIL import Image
from typing import Optional, Dict, Any, List, Callable
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from langsmith import traceable
import streamlit as st
import re

# Configuration
REVIEWER_MODEL = "gemini-3-flash-preview"

# Load styling guide
def _load_styling_guide() -> str:
    """Load styling guide from markdown file."""
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        styling_guide_path = os.path.join(current_dir, "..", "guide", "styling_guide.md")
        if os.path.exists(styling_guide_path):
            with open(styling_guide_path, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            print(f"Warning: Styling guide not found at {styling_guide_path}")
            return ""
    except Exception as e:
        print(f"Error loading styling guide: {e}")
        return ""

styling_guide = _load_styling_guide()

class RecommendedInstructions(BaseModel):
    """Editing instructions organized by category. All fields are optional."""
    subject_focus: str = Field(default="", description="Instructions to adjust or emphasize the main subject. STRICT RULE: Do NOT introduce or add any new components, objects, labels, or elements that are not already present in the current image. Only instruct on what already exists.")
    perspective_and_camera: str = Field(default="", description="Instructions to adjust viewing angle or perspective of existing elements only.")
    additional_comments: str = Field(default="", description="Any other editing instructions for existing elements only. Do NOT suggest adding new visual elements, annotations, text overlays, or callouts.")

class VoiceoverReviewResult(BaseModel):
    """Output schema for the Voiceover Reviewer Agent."""
    description: str = Field(description="Detailed description of the reference image.")
    analysis: str = Field(description="Analysis of the image in relation to the slide context and how well it demonstrates the voiceover.")
    verdict: str = Field(description="Verdict (Yes/No) on whether the image can be used as is to demonstrate the voiceover.")
    regression_detected: bool = Field(default=False, description="Set to true ONLY when the latest edit made the image worse compared to what was described in the previous round. False if this is the first review or if the edit improved or maintained quality.")
    recommended_instructions: RecommendedInstructions = Field(description="Recommended editing instructions organized by category. Only fill relevant categories, leave others empty. All fields should be empty if verdict is Yes.")

def _get_client() -> Optional[genai.Client]:
    """Initialize the Google GenAI client, checking environment and streamlit state."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        api_key = st.session_state.get("google_api_key")
    
    if not api_key:
        return None
        
    try:
        return genai.Client(api_key=api_key)
    except Exception as e:
        print(f"Failed to initialize Gemini client: {e}")
        return None

def call_llm_with_retry(func, *args, max_retries=3, initial_wait=2, **kwargs):
    """
    Retry an LLM call with exponential backoff.
    """
    import time
    last_exc = None
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_exc = e
            if attempt < max_retries - 1:
                wait_time = initial_wait * (2 ** attempt)
                print(f"  ⚠️ LLM call failed: {e}. Retrying in {wait_time}s... ({attempt+1}/{max_retries})")
                time.sleep(wait_time)
            else:
                print(f"  ❌ LLM call failed after {max_retries} attempts: {e}")
    raise last_exc

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
    
    if save_img.mode != 'RGB':
        save_img = save_img.convert('RGB')
    
    # Use JPEG with good quality to reduce size while maintaining quality
    save_img.save(buffered, format="JPEG", quality=90)
    return buffered.getvalue()


def build_review_prompt(slide_title: str = "", slide_content: str = "", voiceover: str = "", visual_instruction: str = "") -> str:
    """Build the core voiceover review prompt without output format instructions.
    
    Shared by both the standalone reviewer_agent and the automated pipeline.
    """
    return f"""You are a Voiceover–Visual Technical Accuracy Reviewer Agent specializing in educational engineering content.

Your task is to determine whether a provided reference visual can be used AS-IS to correctly demonstrate the associated voiceover in an educational slide. You must evaluate instructional correctness only, not aesthetics, presentation quality, or visual polish.

If the visual is not instructionally correct, you must generate precise, minimal corrective visual instructions that address only technical or mechanical mismatches.

INPUTS:

<slide_context>
<slide_title>{slide_title}</slide_title>
<slide_content>{slide_content}</slide_content>
<voiceover>{voiceover}</voiceover>
<visual_instruction>{visual_instruction}</visual_instruction>
</slide_context>

INSTRUCTIONS:

1) Visual Description Requirement
Provide a complete technical description of what is visible in the visual, including:
- All visible objects, components, and subcomponents
- Spatial layout and positioning relationships
- Colors, textures, materials, and visual attributes relevant to instruction
- Text, markings, or labels visible in the visual
- Background, environment, and setting
- Overall composition, framing, and camera perspective

The description must be detailed enough that a reader can reconstruct the visual without seeing it.

2) Voiceover–Visual Technical Alignment Analysis
Identify specific, measurable mismatches between the voiceover and the visual.

Include only objective, quantifiable observations such as:
- Components mentioned in the voiceover but not visible
- Components visible but ambiguous or unreadable
- Scale issues 
- Perspective and camera angle issues
- Legibility issues 
- Occlusion or cropping errors
- Mechanical inaccuracies or incorrect configurations

Do NOT include stylistic or aesthetic commentary.

3) Verdict
Answer the question:
Can this visual be used as-is to demonstrate the voiceover?

Allowed values:
- YES
- NO

4) Corrective Visual Instructions (ONLY IF verdict = NO)

Provide concise, non-overlapping, technically necessary instructions under the following categories.

subject_focus:
- Specify what must be zoomed, centered, or reframed
- Use measurable instructions 
- STRICT: Do NOT introduce new components or objects not visible in the current image

perspective_and_camera:
- ALWAYS start by identifying the current camera position from your description above
- State the change as: "Currently [current angle/view]. [Action] [exact degrees or axis] to achieve [target view]."
- Example: "Currently front-facing at eye level. Rotate 40° clockwise around vertical axis to a front-right view."
- Example: "Currently top-down overhead view. Lower camera 30° to achieve elevated 3/4 view."
- NEVER write a camera instruction without stating the current position it is changing FROM
- Use degrees for rotations, meters or percentage for translations, named viewpoints for reference

additional_comments:
- Lighting, environment, or composition changes ONLY if required to remove ambiguity or misinterpretation
- Do NOT suggest annotations, labels, arrows, highlights, outlines, glows, or text overlays
- Do NOT suggest adding any new visual elements not already present in the image

ANTI-FLUFF CONSTRAINTS (CRITICAL):

You MUST NOT:
- Suggest aesthetic improvements, polish, or presentation enhancements
- Suggest highlights, outlines, glows, callouts, arrows, labels, or text overlays
- Suggest changes that only make something "stand out" without correcting instructional error
- Suggest colors, fonts, or styling unless they fix factual ambiguity
- Use vague terms like "improve," "enhance," or "make clearer" without measurable corrective action
- **ADD OR INTRODUCE any new components, objects, labels, annotations, text overlays,
  or visual elements that are NOT already present in the current image. Instructions
  must only MODIFY, REMOVE, or REFRAME what already EXISTS in the image — never
  extend the scene with new content.**

Only propose visual changes if learner misunderstanding would occur without them.

If the visual is instructionally correct, leave all instruction categories empty.

<evaluation_breakdown>

Use this section as a structured reasoning and scratchpad space for you to evaluate visual–voiceover alignment for the slide.

- Slide Understanding: State in your own words what the slide is about and what the voiceover segments are trying to convey.
- Review of the visuals: Briefly describe what is visibly shown in visual image 
- Visual Alignment Analysis: Analyze whether the assigned visuals correctly support the respective part of the voiceover segment. 
- Additional Analysis: Note any additional observations, thoughts or analysis that can help you arrive at the correct output and verdict.
- Recommended Instructions: Detail each and every recommended instruction to be given to the editing agent to make the asset technically and functioanlly accurate. You need to detailed and careful with your instructions such that it doesn't induce any misinterpreation and ambiguity for the learner.   

(It is ok for this section to be quite verbose, long and detailed as long as it helps you arrive at the correct output.)

</evaluation_breakdown>

Provide your final answer strictly in the following XML format:

<output>

<description>
(Detailed technical description of the visual)
</description>

<analysis>
(Detailed technical alignment analysis)
</analysis>

<verdict>YES|NO</verdict>

<recommended_instructions>

<subject_focus>
(Empty if not required)
</subject_focus>

<perspective_and_camera>
(Empty if not required)
</perspective_and_camera>

<additional_comments>
(Empty if not required)
</additional_comments>

</recommended_instructions>

</output>"""


@traceable(
    metadata={
        "agent_name": "voiceover_reviewer",
        "model": REVIEWER_MODEL
    }
)
def reviewer_agent(
    reference_image: Image.Image,
    slide_title: str = "",
    slide_content: str = "",
    voiceover: str = "",
    visual_instruction: str = ""
) -> VoiceoverReviewResult:
    """
    Reviewer Agent that analyses a reference image and slide context.
    
    Args:
        reference_image: PIL Image of the reference.
        slide_title: Title of the slide.
        slide_content: Content/body of the slide.
        voiceover: Voiceover script text.
        visual_instruction: Visual instruction or requirement for the image.
        
    Returns:
        VoiceoverReviewResult containing description, analysis, verdict, and instructions.
    """
    client = _get_client()
    if client is None:
        raise ValueError("Google API Key not found. Please set GOOGLE_API_KEY environment variable or provides it in st.session_state['google_api_key'].")

    image_bytes = prepare_image_for_gemini(reference_image)
    
    prompt = build_review_prompt(slide_title=slide_title, slide_content=slide_content, voiceover=voiceover, visual_instruction=visual_instruction)
    
    parts = [
        types.Part.from_text(text=prompt),
        types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
    ]

    try:
        from agents.graphics_asset_creation.automated.llm_call_tracker import tracker as llm_tracker
        with llm_tracker.call(REVIEWER_MODEL, "Voiceover Reviewer") as usage:
            response = call_llm_with_retry(
                client.models.generate_content,
                model=REVIEWER_MODEL,
                contents=[types.Content(role="user", parts=parts)],
                config=types.GenerateContentConfig(
                    system_instruction=(
                        f"Follow these styling guidelines when evaluating visual quality and appropriateness:\n\n{styling_guide}\n\n"
                        "STRICT RULE — NO NEW COMPONENTS:\n"
                        "When generating corrective instructions you MUST NOT introduce, add, or suggest "
                        "any new components, objects, labels, annotations, text overlays, or visual elements "
                        "that are not already present in the current image. "
                        "Instructions may only MODIFY, REMOVE, or REFRAME existing elements — "
                        "never extend the scene with new content."
                        "Recommended instructions should be technically and functionally correct. Be careful with your instructions such that it doesn't induce any misinterpretation and ambiguity for the learner."
                    ),
                    thinking_config=types.ThinkingConfig(
                        thinking_level="high",
                    )
                )
            )
            usage.set_response(response)
        
        # Robust multi-format extraction logic
        text_response = response.text or ""
        
        # 1. Try XML extraction (Preferred)
        data = _extract_xml(text_response)
        
        # 2. Try JSON extraction if XML yielded nothing significant
        if not data.get("description") and not data.get("analysis"):
            json_data = _extract_json(text_response)
            if json_data:
                # Merge or use JSON data
                data.update({k: v for k, v in json_data.items() if v})

        # Ensure recommended_instructions is a dict for Pydantic
        if "recommended_instructions" not in data or not isinstance(data["recommended_instructions"], dict):
            data["recommended_instructions"] = {}

        # 3. Use Pydantic to validate and provide structure
        return VoiceoverReviewResult.model_validate(data)
            
    except Exception as e:
        # Return a graceful failure message 
        return VoiceoverReviewResult(
            description="Analysis failed.",
            analysis=f"The agent encountered an error during extraction: {str(e)}",
            verdict="No",
            recommended_instructions=RecommendedInstructions()
        )

def _extract_xml(text: str) -> Dict[str, Any]:
    """Extract XML tags from LLM response into a dictionary matching VoiceoverReviewResult."""
    data = {}
    
    # Extract top-level fields
    for tag in ["description", "analysis", "verdict"]:
        match = re.search(f"<{tag}>(.*?)</{tag}>", text, re.DOTALL | re.IGNORECASE)
        if match:
            data[tag] = match.group(1).strip()
        
    # Extract nested recommended_instructions
    instructions = {}
    instr_match = re.search(r"<recommended_instructions>(.*?)</recommended_instructions>", text, re.DOTALL | re.IGNORECASE)
    if instr_match:
        instr_text = instr_match.group(1)
        for tag in ["subject_focus", "perspective_and_camera", "additional_comments"]:
            tag_match = re.search(f"<{tag}>(.*?)</{tag}>", instr_text, re.DOTALL | re.IGNORECASE)
            if tag_match:
                content = tag_match.group(1).strip()
                # Filter out placeholder text and genuinely empty content
                if content and not content.startswith("(") and content.lower() not in ["empty if not required", "n/a", "none"]:
                    instructions[tag] = content
    
    # Always include recommended_instructions, even if empty
    data["recommended_instructions"] = instructions
        
    return data

def _extract_json(text: str) -> Dict[str, Any]:
    """Extract JSON from potential markdown text."""
    try:
        # Try finding markdown code block
        json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(1))
        
        # Try finding the first { and last }
        first_brace = text.find('{')
        last_brace = text.rfind('}')
        if first_brace != -1 and last_brace != -1:
            try:
                return json.loads(text[first_brace:last_brace+1])
            except:
                pass
            
        return json.loads(text)
    except Exception:
        return {}


