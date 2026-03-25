import os
import base64
import re
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
import requests

load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================
DEFAULT_GENERATOR_MODEL = "gemini-3-pro-image-preview" 
DEFAULT_REVIEWER_MODEL = "gemini-3-flash"
MAX_CORRECTION_ROUNDS = 5

# Schema for Reviewer
class ReviewResult(BaseModel):
    approved: bool = Field(description="Boolean status. True if the generated graphics asset meets all technical, instructional, scene definition and styling requirements.")
    feedback: str = Field(description="Concise, objective description of any detected discrepancies, missing elements, or visual artifacts.")
    confidence_score: int = Field(description="Technical confidence score (0-100).")

# Schema for Technical Accuracy Checker
class TechnicalAccuracyResult(BaseModel):
    is_accurate: bool = Field(description="True if the output is technically accurate according to the slide content and voiceover focus.")
    accuracy_score: int = Field(description="Technical accuracy score (0-100). Must be >= 90 to pass.")
    missing_components: List[str] = Field(default_factory=list, description="List of technical components mentioned in slide content but missing from the output.")
    incorrect_elements: List[str] = Field(default_factory=list, description="List of technical inaccuracies or misrepresentations found.")
    label_issues: List[str] = Field(default_factory=list, description="List of label/text accuracy issues.")
    recommendations: str = Field(description="Specific recommendations to improve technical accuracy.")

def _get_client() -> Optional[genai.Client]:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        error_msg = "Missing GOOGLE_API_KEY"
        print(error_msg)
        if hasattr(st, 'error'):
            try:
                st.error(error_msg)
            except:
                pass
        return None
    try:
        return genai.Client(api_key=api_key)
    except Exception as e:
        print(f"Failed to initialize Gemini client: {e}")
        return None

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
# STYLING GUIDE
# =============================================================================

def _load_styling_guide() -> str:
    """Load styling guide from markdown file."""
    try:
        # Get the directory of the current file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Navigate to the guide directory
        config_path = os.path.join(current_dir, "guide", "styling_guide.md")
        config_path = os.path.normpath(config_path)
        
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            print(f"Warning: Styling guide not found at {config_path}, using default")
            return "Follow professional design guidelines with clear, educational focus."
    except Exception as e:
        print(f"Error loading styling guide: {e}")
        return "Follow professional design guidelines with clear, educational focus."

# Load styling guide from markdown file
styling_guide = _load_styling_guide()

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

# =============================================================================
# TECHNICAL ACCURACY CHECKER - CRITICAL VALIDATION LAYER
# =============================================================================

@traceable(
    metadata={
        "agent_name": "technical_accuracy_checker",
        "step_name": "Technical Validation",
        "function_name": "check_technical_accuracy"
    }
)
def check_technical_accuracy(
    output_content: str,
    output_type: str,
    slide_chunk: str,
    slide_title: str,
    voiceover: str,
    visual_instruction: Optional[str] = None,
    output_image: Optional[Image.Image] = None
) -> TechnicalAccuracyResult:
    """
    CRITICAL: Validates technical accuracy of any output against slide content.
    Acts as a gatekeeper to ensure technical precision before passing to user.
    
    Args:
        output_content: The text output to validate (analysis, definition, etc.)
        output_type: Type of output being validated ('analysis', 'definition', 'image')
        slide_chunk: The slide content containing technical information
        slide_title: The slide title
        voiceover: The voiceover focus - what concept should be highlighted
        visual_instruction: Visual instruction provided by user (optional)
        output_image: Generated image for visual validation (optional)
    
    Returns:
        TechnicalAccuracyResult with validation status and detailed feedback
    """
    client = _get_client()
    if not client:
        print("Technical Accuracy Checker: Gemini client initialization failed.")
        # Return failed result if client unavailable
        return TechnicalAccuracyResult(
            is_accurate=False,
            accuracy_score=0,
            missing_components=["LLM client unavailable"],
            incorrect_elements=[],
            label_issues=[],
            recommendations="Fix API configuration to enable technical validation."
        )
    
    try:
        print(f"\n{'='*60}")
        print(f"TECHNICAL ACCURACY CHECK - {output_type.upper()}")
        print(f"{'='*60}")
        
        # Build validation prompt based on output type
        if output_type == "analysis":
            validation_prompt = f"""You are a CRITICAL technical accuracy validator. Your ONLY job is to verify technical accuracy.

SLIDE CONTEXT (SOURCE OF TRUTH):
Title: {slide_title}
Content: {slide_chunk}
Voiceover Focus: "{voiceover}"

OUTPUT TO VALIDATE:
{output_content}

CRITICAL VALIDATION TASK:
The output is a reference image analysis. Validate if the analysis correctly identifies technical elements that align with "{voiceover}" and the slide content.

CHECK FOR:
1. Does the analysis correctly identify technical components mentioned in the slide?
2. Are the suggested modifications technically accurate and relevant to "{voiceover}"?
3. Are there any technical misunderstandings or inaccuracies in the analysis?
4. Does the analysis focus on the right technical aspects?

STRICT CRITERIA:
- Accuracy score must be >= 90 to pass
- Any major technical misunderstanding = immediate fail
- Missing key technical concepts = fail
- Focus on technical correctness, not visual quality

Return structured validation result."""

        elif output_type == "definition":
            validation_prompt = f"""You are a CRITICAL technical accuracy validator. Your ONLY job is to verify technical accuracy.

SLIDE CONTEXT (SOURCE OF TRUTH):
Title: {slide_title}
Content: {slide_chunk}
Voiceover Focus: "{voiceover}"
Visual Instruction: {visual_instruction or "Not provided"}

GRAPHICS DEFINITION TO VALIDATE:
{output_content}

CRITICAL VALIDATION TASK:
The definition describes what image should be generated to illustrate "{voiceover}". Validate technical accuracy against slide content.

CHECK FOR:
1. COMPONENT ACCURACY: Are all technical components from the slide content correctly mentioned?
2. LABEL ACCURACY: Are technical terms, labels, and names used correctly?
3. RELATIONSHIP ACCURACY: Are connections and relationships between components technically correct?
4. SPECIFICATION ACCURACY: Are any numbers, measurements, or specs mentioned correctly?
5. VOICEOVER ALIGNMENT: Does the definition accurately represent "{voiceover}"?
6. TECHNICAL TERMINOLOGY: Is technical language used correctly?

IDENTIFY:
- Missing critical technical components that MUST be in the image
- Incorrect technical terms or misrepresentations
- Label inaccuracies (wrong names, typos in technical terms)
- Any technical impossibilities or contradictions

STRICT CRITERIA:
- Accuracy score must be >= 90 to pass
- Missing critical component mentioned in slide = fail
- Incorrect technical terminology = fail  
- Misrepresentation of "{voiceover}" = fail

Return structured validation result."""

        elif output_type == "image":
            validation_prompt = f"""
You are a Technical Auditor AI.

Your task is to analyze the provided image as a real, physical, engineered object or system.
You must judge the image for technical correctness, functional accuracy, internal logic, and safety implications.

SLIDE CONTEXT (SOURCE OF TRUTH):
Title: {slide_title}
Content: {slide_chunk}
Voiceover Focus: "{voiceover}"

GRAPHICS DEFINITION USED:
{output_content}

GENERATED IMAGE:
[Image will be provided]

Follow this process strictly:

1. OBJECT IDENTIFICATION
- Identify what the object/system is.
- State its intended real-world function.
- Identify major components visible in the image.

2. FUNCTIONAL INTERPRETATION
- Explain how the system is supposed to work in real operation.
- Describe flow paths, force transmission, pressure paths, electrical paths, or logical paths as applicable.
- Assume the object is real unless proven otherwise.

3. TECHNICAL VERIFICATION
- Verify whether the depicted configuration is mechanically, electrically, or physically correct.
- Confirm whether parts that appear connected should be connected in reality.
- Explicitly state what is correct.

4. ERROR & AMBIGUITY DETECTION
- Identify any incorrect, misleading, incomplete, or oversimplified representations.
- Call out ambiguous labeling, misleading arrows, colors, or annotations.
- Explain why a trained professional could misinterpret the image.

5. SAFETY & OPERATIONAL RISKS
- Identify any dangerous assumptions a viewer might make based on this image.
- Explain real-world consequences if the image is misunderstood or applied incorrectly.
- Distinguish between “technically wrong” and “technically unsafe”.

6. EDGE CASES & MISUSE SCENARIOS
- Describe what happens in edge conditions (e.g., partial operation, failure modes, incorrect usage).
- Identify scenarios where the image may lead to incorrect actions.

7. FINAL VERDICT
- Give a clear verdict using one of the following:
  - "Technically correct"
  - "Technically correct with misleading elements"
  - "Technically incorrect"
  - "Technically unsafe"
- Justify the verdict concisely.

IMPORTANT RULES:
- Do NOT assume user intent.
- Do NOT hallucinate hidden components.
- Do NOT simplify explanations; prefer precision over brevity.
- If something cannot be confirmed from the image alone, explicitly say so.
- If labeling is misleading but the hardware is correct, clearly separate those facts.

Tone: professional, precise, engineering-grade.
Audience: technicians, engineers, safety reviewers, educators.
"""

        else:
            raise ValueError(f"Unknown output_type: {output_type}")
        
        # Build parts for LLM request
        parts = [types.Part.from_text(text=validation_prompt)]
        
        # Add image if validating generated image
        if output_image and output_type == "image":
            # Directly call prepare_image_for_gemini (defined later in this file)
            image_bytes = prepare_image_for_gemini(output_image)
            parts.append(types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"))
        
        # Call LLM for validation
        response = client.models.generate_content(
            model=DEFAULT_REVIEWER_MODEL,
            contents=[types.Content(role="user", parts=parts)],
            config=types.GenerateContentConfig(
                temperature=0.2,  # Very low temperature for consistent, strict validation
                response_mime_type="application/json",
                response_schema=TechnicalAccuracyResult
            )
        )
        
        # Parse response
        if hasattr(response, 'parsed') and response.parsed:
            result = response.parsed
            if isinstance(result, dict):
                validation_result = TechnicalAccuracyResult(**result)
            else:
                validation_result = result
        else:
            data = json.loads(response.text)
            validation_result = TechnicalAccuracyResult(**data)
        
        # Log validation result
        status = "✅ PASSED" if validation_result.is_accurate else "❌ FAILED"
        print(f"\nValidation Result: {status}")
        print(f"Accuracy Score: {validation_result.accuracy_score}/100")
        
        if not validation_result.is_accurate:
            print(f"\n⚠️ TECHNICAL ACCURACY ISSUES DETECTED:")
            if validation_result.missing_components:
                print(f"Missing Components: {', '.join(validation_result.missing_components)}")
            if validation_result.incorrect_elements:
                print(f"Incorrect Elements: {', '.join(validation_result.incorrect_elements)}")
            if validation_result.label_issues:
                print(f"Label Issues: {', '.join(validation_result.label_issues)}")
            print(f"\nRecommendations: {validation_result.recommendations}")
        
        print(f"{'='*60}\n")
        
        return validation_result
        
    except Exception as e:
        print(f"Technical Accuracy Check Error: {e}")
        # Return failed result on error
        return TechnicalAccuracyResult(
            is_accurate=False,
            accuracy_score=0,
            missing_components=[],
            incorrect_elements=[f"Validation error: {str(e)}"],
            label_issues=[],
            recommendations="Fix validation error and retry."
        )

# =============================================================================
# REFERENCE IMAGE ANALYSIS
# =============================================================================

def analyze_reference_image_quality(reference_image_path: str, slide_chunk: str, slide_title: str, voiceover: str) -> Optional[str]:
    """
    Analyzes if a reference image is a good educational asset and provides insights on presentation.

    Args:
        reference_image_path (str): Path or URL to the reference image
        slide_chunk (str): The content of the slide chunk providing context
        slide_title (str): The title of the slide
        voiceover (str): The current line/segment of slide chunk in focus

    Returns:
        Optional[str]: Critical insights and recommendations or None if analysis fails
    """
    client = _get_client()
    if not client:
        print("Gemini client initialization failed.")
        return None

    try:
        # Load and prepare the image for analysis
        image_data = None
        
        if reference_image_path.startswith(("http://", "https://")):
            # Check if it's a Google Drive link
            if "drive.google.com" in reference_image_path:
                print("Detected Google Drive link, converting to direct download URL...")
                direct_url = convert_drive_link_to_direct(reference_image_path)
                if not direct_url:
                    print("Failed to convert Google Drive link to direct download URL.")
                    return None
                reference_image_path = direct_url
                print(f"Using direct URL: {direct_url}")
            
            # Handle web image (including converted Drive links)
            print(f"Downloading image from: {reference_image_path}")
            response = requests.get(reference_image_path, timeout=30)
            if response.status_code == 200:
                image_data = response.content
                print(f"Successfully downloaded {len(image_data)} bytes")
            else:
                print(f"Failed to download image. Status code: {response.status_code}")
                return None
        else:
            # Handle local file path
            if os.path.exists(reference_image_path):
                with open(reference_image_path, 'rb') as f:
                    image_data = f.read()
                print(f"Loaded local file: {len(image_data)} bytes")
            else:
                print(f"Image file not found: {reference_image_path}")
                return None

        if not image_data:
            print("No image data available for analysis.")
            return None

        # Check if this is a Drive link (indicating professionally prepared asset)
        is_drive_link = "drive.google.com" in str(reference_image_path)

        # Prepare the analysis prompt
        if is_drive_link:
            # More conservative analysis for Drive links - they're usually high-quality, pre-approved assets
            prompt = f"""You are evaluating a professionally prepared reference image. This image is likely already well-designed and closely aligned with requirements.

Primary Evaluation Target (voiceover): "{voiceover}"

Context (for reference only):
- Slide Title: {slide_title}
- Slide Content: {slide_chunk}

YOUR TASK: 
This reference image is likely already high-quality and appropriate. Assess if it effectively illustrates "{voiceover}". Only suggest minor refinements if absolutely necessary.

OUTPUT FORMAT:

If image is appropriate (most common case):
"This reference image effectively illustrates '{voiceover}'. Key strengths: [what specifically supports the concept]. The image can be used directly with minimal or no modifications."

If only minor improvements would help:
"This reference image is largely appropriate for '{voiceover}'. Optional minor enhancements:
- [only critical minor adjustment if needed]
- [only critical minor adjustment if needed]"

Be conservative - if the image communicates the concept adequately, approve it without unnecessary modifications. Focus on whether it serves the educational purpose.
Keep response brief and focused on "{voiceover}"."""
        else:
            # Standard analysis for non-Drive links
            prompt = f"""You are a critical educational graphics expert. Your primary focus is evaluating if this reference image effectively illustrates the voiceover concept: "{voiceover}"

Primary Evaluation Target (voiceover): "{voiceover}"

Context (for reference only):
- Slide Title: {slide_title}
- Slide Content: {slide_chunk}

YOUR TASK: 
Evaluate if the image directly and clearly illustrates "{voiceover}". The slide content is just background context - your analysis must focus specifically on whether the image supports this voiceover statement.

OUTPUT FORMAT - Provide direct modification suggestions:

If image clearly illustrates the voiceover:
"Key strengths: [what specifically supports "{voiceover}"]."

If modifications needed to better illustrate the voiceover:
- [specific modification to emphasize "{voiceover}"]
- [specific modification to clarify "{voiceover}"]
- [specific modification to highlight "{voiceover}"]

Examples of good modifications: "Add labels to identify components", "Emphasize the connection between X and Y with arrows", "Remove background clutter", "Show step-by-step progression", "Highlight the active element", "Add annotations explaining the process"
Examples are illustrative, not exhaustive or mandatory. Prefer the most appropriate visual solution for the specific voiceover.
Keep response focused on how well the image illustrates "{voiceover}" specifically."""

        # Detect MIME type from image data
        mime_type = "image/jpeg"  # default
        if image_data[:4] == b'\x89PNG':
            mime_type = "image/png"
        elif image_data[:2] == b'\xff\xd8':
            mime_type = "image/jpeg"
        elif image_data[:4] == b'RIFF' and image_data[8:12] == b'WEBP':
            mime_type = "image/webp"
        
        # Send image and prompt to Gemini 3 Flash
        response = client.models.generate_content(
            model=DEFAULT_REVIEWER_MODEL,
            contents=[
                types.Content(
                    role="user", 
                    parts=[
                        types.Part.from_text(text=prompt),
                        types.Part.from_bytes(data=image_data, mime_type=mime_type)
                    ]
                )
            ],
            config=types.GenerateContentConfig(
                temperature=0.3,  # Lower temperature for critical, objective analysis
                response_mime_type="text/plain"
            )
        )

        # Extract and return the analysis
        if not response.candidates or not response.candidates[0].content:
            print("Model returned no content.")
            return None

        model_content = response.candidates[0].content
        
        if not hasattr(model_content, 'parts') or model_content.parts is None:
            print("Model returned content without parts.")
            return None

        # Extract text from the first part
        for part in model_content.parts:
            if hasattr(part, 'text') and part.text:
                analysis_result = part.text.strip()
                
                # CRITICAL: Technical Accuracy Check
                print("\n🔍 Running Technical Accuracy Check on Analysis...")
                ta_result = check_technical_accuracy(
                    output_content=analysis_result,
                    output_type="analysis",
                    slide_chunk=slide_chunk,
                    slide_title=slide_title,
                    voiceover=voiceover
                )
                
                if not ta_result.is_accurate:
                    print(f"⚠️ Technical accuracy check FAILED (Score: {ta_result.accuracy_score}/100)")
                    print(f"Issues: {ta_result.recommendations}")
                    # Return with warning prefix
                    return f"⚠️ TECHNICAL ACCURACY WARNING (Score: {ta_result.accuracy_score}/100):\n{ta_result.recommendations}\n\n---ANALYSIS---\n{analysis_result}"
                else:
                    print(f"✅ Technical accuracy check PASSED (Score: {ta_result.accuracy_score}/100)")
                    return analysis_result
        
        print("No text content found in response parts.")
        return None

    except Exception as e:
        print(f"Error during image analysis: {e}")
        return None

# =============================================================================
# REFERENCE IMAGE PROCESSING
# =============================================================================

def convert_drive_link_to_direct(drive_link: str) -> Optional[str]:
    """
    Convert Google Drive share link to direct download link.
    
    Args:
        drive_link (str): Google Drive share link (e.g., https://drive.google.com/file/d/{FILE_ID}/view)
        
    Returns:
        Optional[str]: Direct download URL or None if conversion fails
    """
    import re
    
    # Pattern to extract file ID from various Google Drive link formats
    patterns = [
        r'drive\.google\.com/file/d/([a-zA-Z0-9_-]+)',  # /file/d/{id}/view
        r'drive\.google\.com/open\?id=([a-zA-Z0-9_-]+)',  # ?id={id}
        r'drive\.google\.com/uc\?id=([a-zA-Z0-9_-]+)',  # ?id={id}
    ]
    
    for pattern in patterns:
        match = re.search(pattern, drive_link)
        if match:
            file_id = match.group(1)
            # Return direct download URL
            return f"https://drive.google.com/uc?export=download&id={file_id}"
    
    print(f"Could not extract file ID from Drive link: {drive_link}")
    return None

def identify_link_type(link: str) -> str:
    """
    Identify if the link is a Google Drive link or a web link.
    
    Args:
        link (str): The reference image link
        
    Returns:
        str: "drive" for Google Drive links, "web" for web links, "unknown" for unrecognized
    """
    if not link:
        return "unknown"
    
    link = link.lower().strip()
    
    # Google Drive link patterns
    drive_patterns = [
        "drive.google.com"
    ]
    
    if any(pattern in link for pattern in drive_patterns):
        return "drive"
    
    # Web link patterns (http/https)
    if link.startswith(("http://", "https://")):
        return "web"
    
    return "unknown"

def process_drive_image(drive_link: str) -> Optional[bytes]:
    """
    Process Google Drive image link and download the image.
    
    Args:
        drive_link (str): Google Drive link to the image
        
    Returns:
        Optional[bytes]: Image data as bytes or None if processing fails
    """
    try:
        # Convert to direct download link
        direct_url = convert_drive_link_to_direct(drive_link)
        if not direct_url:
            print("Failed to convert Drive link to direct download URL")
            return None
        
        # Download the image
        print(f"Downloading from: {direct_url}")
        response = requests.get(direct_url, timeout=30)
        
        if response.status_code == 200:
            print(f"Successfully downloaded {len(response.content)} bytes from Drive")
            return response.content
        else:
            print(f"Failed to download. Status code: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"Error processing Drive image: {e}")
        return None

def process_web_image(web_link: str) -> Optional[bytes]:
    """
    Process web image link and download the image.
    
    Args:
        web_link (str): Web URL to the image
        
    Returns:
        Optional[bytes]: Image data as bytes or None if processing fails
    """
    try:
        print(f"Downloading from web: {web_link}")
        response = requests.get(web_link, timeout=30)
        
        if response.status_code == 200:
            print(f"Successfully downloaded {len(response.content)} bytes")
            return response.content
        else:
            print(f"Failed to download. Status code: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"Error processing web image: {e}")
        return None

# =============================================================================
# GENERATE GRAPHICS DEFINITION
# =============================================================================

def generate_graphics_definition(slide_chunk: str, slide_title: str, visual_instruction: str, voiceover: str, reference_analysis: Optional[str] = None, visual_style: str = "Illustration") -> Optional[str]:
    """
    Transforms a simple visual instruction into a detailed graphics definition based on the slide context.

    Args:
        slide_chunk (str): The content of the slide chunk providing context.
        slide_title (str): The title of the slide.
        visual_instruction (str): A simple visual instruction to be detailed.
        voiceover (str): The current line/segment of slide chunk in focus that should be emphasized.
        reference_analysis (Optional[str]): Optional analysis from analyze_reference_image_quality to guide improvements.
        visual_style (str): Style of the visual - either "Illustration" or "Real World Image". Default: "Illustration".

    Returns:
        Optional[str]: A detailed graphics definition in XML format with actionable to-dos or None if the operation fails.
    """
    client = _get_client()
    if not client:
        print("Gemini client initialization failed.")
        return None

    try:
        # Prepare the prompt for the Gemini model
        reference_guidance = ""
        if reference_analysis:
            reference_guidance = f"\n\nPriority Modifications from Reference Analysis:\n{reference_analysis}\nThese modifications must be incorporated first into the graphics definition."
        
        # Visual style instructions
        style_guidance = ""
        if visual_style == "Real World Image":
            style_guidance = "\n\n⚠️ VISUAL STYLE: REAL WORLD IMAGE - Use photorealistic descriptions, real-world objects, actual environments, natural lighting, and realistic textures. Describe as if photographing or filming actual physical subjects."
        else:  # Illustration
            style_guidance = "\n\n⚠️ VISUAL STYLE: ILLUSTRATION - Use stylized, simplified visual representations, graphic elements, icons, diagrams, and illustrated components. Describe as a designed graphic or diagram."
        
        prompt = f"""You are a technical graphics designer creating detailed graphics definitions for educational slide content.

CONTEXT:
Slide Title: {slide_title}
Slide Content: {slide_chunk}
Current Voiceover Focus: {voiceover}
Visual Instruction: {visual_instruction}
Visual Style: {visual_style}{style_guidance}
Modifications to be prioritized: {reference_guidance}

RESEARCH STEP:
Use Google Search to research technical accuracy of components mentioned in the slide content. Find: technical specifications, standard visual representations, correct terminology, safety symbols, and typical configurations. Incorporate findings into the definition.

TASK: Create a comprehensive, detailed graphics definition that describes exactly what should be generated to best illustrate "{voiceover}".

CRITICAL: Your definition MUST cover ALL of these aspects in a flowing, descriptive manner:

1. SCENE COMPOSITION & SETUP:
   - Overall scene structure and framing (e.g., "centered composition", "split-screen layout", "layered depth arrangement")
   - Perspective and viewing angle (e.g., "top-down view", "side elevation", "isometric perspective", "eye-level view")
   - Spatial organization and how elements are distributed across the frame

2. VISUAL ELEMENTS & COMPONENTS:
   - Primary elements that illustrate the voiceover concept "{voiceover}"
   - Secondary supporting elements with their exact shapes, forms, and representations
   - All technical components, objects, or subjects with precise descriptions
   - Any icons, symbols, or graphical representations needed

3. SPATIAL ARRANGEMENT & LAYOUT:
   - Exact positioning of each element (e.g., "left side", "upper right quadrant", "centered foreground")
   - Size relationships between elements (e.g., "server icon 2x larger than database icon")
   - Spacing and gaps between components
   - Alignment and grouping of related elements

4. CONNECTIONS & RELATIONSHIPS:
   - How elements connect or relate to each other
   - Arrows, lines, or connectors with direction and flow
   - Visual hierarchy showing what's most to least important
   - Cause-and-effect or sequential relationships

5. DEPTH & LAYERING:
   - Foreground elements (what's closest to viewer)
   - Middle-ground elements
   - Background elements and setting
   - How layers overlap or separate

6. ENVIRONMENT & SURROUNDINGS:
   - Background setting or context (e.g., "abstract gradient background", "data center environment", "clean white backdrop")
   - Atmospheric elements that provide context
   - Environmental details that support the concept
   - Surface or ground plane if applicable

7. LABELS & ANNOTATIONS:
   - Exact text labels for all components
   - Placement of each label relative to its element
   - Callouts, captions, or explanatory text
   - Titles or headings if needed

8. TECHNICAL ACCURACY:
   - Specific technical details from the slide content
   - Accurate representation of concepts, systems, or processes
   - Proper proportions and realistic representations
   - Reference modifications incorporated (if provided)

REQUIREMENTS:
- Start with priority modifications from reference analysis if provided
- Use precise, actionable language (e.g., "place a blue server icon in the upper left, label it 'Cloud Server', connect it with a black arrow pointing to...")
- Specify positioning, sizing, and relationships explicitly
- Include ALL visual elements needed to fully illustrate "{voiceover}"
- NO styling/color instructions - focus only on structural and content elements
- Be comprehensive yet clear - aim for 100-150 words
- If a section is not meaningfully required to illustrate "{voiceover}", keep it minimal and do not introduce unnecessary elements. Prioritize conceptual clarity over completeness.

OUTPUT FORMAT: Write a detailed, flowing paragraph that systematically covers all 8 aspects above. The definition should be so complete that someone could recreate the exact image from your description alone.

Generate the comprehensive graphics definition:"""

        # Use the Gemini 3 Flash model with Google Search to generate the graphics definition
        response = client.models.generate_content(
            model=DEFAULT_REVIEWER_MODEL,
            contents=[types.Content(role="user", parts=[types.Part.from_text(text=prompt)])],
            config=types.GenerateContentConfig(
                temperature=0.6,
                response_mime_type="text/plain",
                tools=[{"google_search": {}}]  # Enable Google Search for technical accuracy
            )
        )

        # Extract and return the generated text
        if not response.candidates or not response.candidates[0].content:
            print("Model returned no content.")
            return None

        model_content = response.candidates[0].content
        
        if not hasattr(model_content, 'parts') or model_content.parts is None:
            print("Model returned content without parts.")
            return None

        # Extract text from the first part
        for part in model_content.parts:
            if hasattr(part, 'text') and part.text:
                definition_result = f"<GraphicsDefinition>\n{part.text.strip()}\n</GraphicsDefinition>"
                
                # CRITICAL: Technical Accuracy Check
                print("\n🔍 Running Technical Accuracy Check on Graphics Definition...")
                ta_result = check_technical_accuracy(
                    output_content=definition_result,
                    output_type="definition",
                    slide_chunk=slide_chunk,
                    slide_title=slide_title,
                    voiceover=voiceover,
                    visual_instruction=visual_instruction
                )
                
                if not ta_result.is_accurate:
                    print(f"⚠️ Technical accuracy check FAILED (Score: {ta_result.accuracy_score}/100)")
                    print(f"Re-generating with corrections...")
                    
                    # Build correction guidance
                    correction_guidance = f"""
PREVIOUS DEFINITION HAD TECHNICAL ACCURACY ISSUES (Score: {ta_result.accuracy_score}/100):

Missing Components: {', '.join(ta_result.missing_components) if ta_result.missing_components else 'None'}
Incorrect Elements: {', '.join(ta_result.incorrect_elements) if ta_result.incorrect_elements else 'None'}
Label Issues: {', '.join(ta_result.label_issues) if ta_result.label_issues else 'None'}

CORRECTIONS NEEDED:
{ta_result.recommendations}

REGENERATE the graphics definition ensuring ALL technical components from the slide content are accurately included, labels are correct, and technical relationships are precise.
"""
                    
                    # Retry with corrections
                    retry_prompt = prompt + correction_guidance
                    retry_response = client.models.generate_content(
                        model=DEFAULT_REVIEWER_MODEL,
                        contents=[types.Content(role="user", parts=[types.Part.from_text(text=retry_prompt)])],
                        config=types.GenerateContentConfig(
                            temperature=0.6,
                            response_mime_type="text/plain",
                            tools=[{"google_search": {}}]  # Enable Google Search for retry as well
                        )
                    )
                    
                    if retry_response.candidates and retry_response.candidates[0].content:
                        retry_content = retry_response.candidates[0].content
                        if hasattr(retry_content, 'parts') and retry_content.parts:
                            for retry_part in retry_content.parts:
                                if hasattr(retry_part, 'text') and retry_part.text:
                                    corrected_definition = f"<GraphicsDefinition>\n{retry_part.text.strip()}\n</GraphicsDefinition>"
                                    print("✅ Corrected definition generated")
                                    return corrected_definition
                    
                    # If retry failed, return original with warning
                    return f"⚠️ TECHNICAL ACCURACY WARNING (Score: {ta_result.accuracy_score}/100):\n{ta_result.recommendations}\n\n{definition_result}"
                else:
                    print(f"✅ Technical accuracy check PASSED (Score: {ta_result.accuracy_score}/100)")
                    return definition_result
        
        print("No text content found in response parts.")
        return None

    except Exception as e:
        print(f"Error during graphics definition generation: {e}")
        return None

# =============================================================================
# IMAGE GENERATION WITH NANO BANANA PRO
# =============================================================================

def image_from_base64(base64_string: str) -> Image.Image:
    """Convert base64 string to PIL Image."""
    if ',' in base64_string:
        base64_string = base64_string.split(',', 1)[1]
    image_bytes = base64.b64decode(base64_string)
    return Image.open(BytesIO(image_bytes))

@traceable(
    metadata={
        "agent_name": "graphics_creation",
        "step_name": "Generator",
        "function_name": "generate_image_from_definition",
        "user_id": st.session_state.get("role", "anonymous") if hasattr(st, 'session_state') else "anonymous",
        "user_email": st.session_state.get("user_email", "anonymous") if hasattr(st, 'session_state') else "anonymous"
    }
)
def generate_image_from_definition(
    graphics_definition: str,
    slide_chunk: str,
    slide_title: str,
    voiceover: str,
    reference_image: Optional[Image.Image] = None,
    correction_feedback: Optional[str] = None,
    chat_session = None,
    client = None,
    aspect_ratio: Optional[str] = None,
    image_size: str = "1K"
) -> Tuple[Image.Image, Dict, object, object]:
    """
    Generate an image using Nano Banana Pro based on graphics definition and context.
    
    Args:
        graphics_definition: Detailed graphics definition (paragraph format)
        slide_chunk: The content of the slide chunk
        slide_title: The title of the slide
        voiceover: The voiceover text being illustrated
        reference_image: Optional reference image for style/content guidance
        correction_feedback: Feedback from reviewer for regeneration (optional)
        chat_session: Chat session object for multi-turn generation (optional)
        client: Gemini client object (optional, will create if not provided)
        aspect_ratio: Aspect ratio for the generated image (optional)
        image_size: Size of the image to generate (default: "1K")
    
    Returns:
        Tuple of (generated_image, metadata, chat_session, client)
    """
    try:
        if client is None:
            client = _get_client()
            if client is None:
                raise ValueError("Missing GOOGLE_API_KEY.")
        
        print(f"Starting image generation with Nano Banana Pro...")
        
        # Build image_config dynamically
        image_config_params = {"image_size": image_size}
        if aspect_ratio:
            image_config_params["aspect_ratio"] = aspect_ratio
        
        if not chat_session:
            # Round 1: Create chat session and send initial request
            print("Creating new chat session with full context...")
            
            chat_session = client.chats.create(
                model=DEFAULT_GENERATOR_MODEL,
                config=types.GenerateContentConfig(
                    system_instruction=f"Follow these styling guidelines carefully for generations: \n\n{styling_guide}",
                    response_modalities=['TEXT', 'IMAGE'],
                    tools=[{"google_search": {}}],
                    image_config=types.ImageConfig(**image_config_params)
                )
            )
            
            prompt = f"""Generate an educational graphics asset based on the provided context and requirements.

<slide_context>
Slide Title: {slide_title}
Slide Content: {slide_chunk}
Voiceover (Primary Focus): {voiceover}
</slide_context>

<graphics_requirements>
{graphics_definition}
</graphics_requirements>

<technical_research>
Use Google Search to verify technical accuracy of components before generating. Ensure correct labels, proportions, and configurations.
</technical_research>

<reference_image_usage>
The reference image (if provided) shows EXACT technical components, layout, or style to reproduce.
- Reproduce shapes, labels, text, proportions, connections, and visual elements with high accuracy
- Maintain the technical accuracy while enhancing visual appeal
- You can adapt scene composition but preserve component accuracy
</reference_image_usage>

<primary_objective>
Create an image that PRIMARILY ILLUSTRATES: "{voiceover}"
This voiceover statement is the central focus. All visual elements should support understanding this concept.
</primary_objective>

<requirements>
1. Follow the graphics requirements paragraph precisely
2. Verify technical accuracy with Google Search
3. Reproduce reference image components accurately
4. Emphasize the voiceover concept visually
5. Use clear visual hierarchy to guide attention
6. Include all specified labels, annotations, and elements
7. Maintain technical accuracy and educational clarity
8. Professional quality with consistent styling
</requirements>

Generate the educational graphics asset that best illustrates "{voiceover}".
"""
            
            # Build parts list: prompt first, then reference image if available
            parts = [types.Part.from_text(text=prompt)]
            
            # Add reference image if provided
            if reference_image:
                print(f"Adding reference image for style/content guidance...")
                image_bytes = prepare_image_for_gemini(reference_image)
                parts.append(
                    types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
                )
                print(f"  ✓ Reference image attached")
            
            response = chat_session.send_message(parts)
        else:
            # Round 2+: Send feedback to existing chat session
            print("Sending feedback to existing chat session...")
            feedback_prompt = f"""<status>
The previous output was rejected during quality assurance review.
</status>

<detected_issues>
{correction_feedback}
</detected_issues>

<directive>
Regenerate the image. Correct the specific issues listed above. Use Google Search to verify technical accuracy if needed.
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
        
        print("✓ Image generated successfully")
        return generated_image, metadata, chat_session, client
        
    except Exception as e:
        print(f"Image generation failed: {str(e)}")
        raise

@traceable(
    metadata={
        "agent_name": "graphics_creation",
        "step_name": "Reviewer",
        "function_name": "review_generated_image",
        "user_id": st.session_state.get("role", "anonymous") if hasattr(st, 'session_state') else "anonymous",
        "user_email": st.session_state.get("user_email", "anonymous") if hasattr(st, 'session_state') else "anonymous"
    }
)
def review_generated_image(
    generated_image: Image.Image,
    graphics_definition: str,
    slide_chunk: str,
    slide_title: str,
    voiceover: str,
    reference_image: Optional[Image.Image] = None,
    previous_feedback: Optional[str] = None,
    custom_criteria: Optional[str] = None
) -> Tuple[bool, Optional[str], Dict]:
    """
    Review the generated image against the graphics definition and requirements.
    
    Args:
        generated_image: The generated image to review
        graphics_definition: The graphics definition requirements
        slide_chunk: The content of the slide chunk
        slide_title: The title of the slide
        voiceover: The voiceover text that should be illustrated
        reference_image: Reference image that was provided (optional, for context)
        previous_feedback: Previous reviewer feedback (optional)
        custom_criteria: Additional custom review criteria (optional)
    
    Returns:
        Tuple of (is_approved, feedback, metadata)
    """
    try:
        client = _get_client()
        if client is None:
            raise ValueError("Missing GOOGLE_API_KEY.")
        
        print("Reviewing generated image...")
        
        generated_bytes = prepare_image_for_gemini(generated_image)
        
        base_prompt = f"""Review the generated image against the requirements and context.

<slide_context>
Slide Title: {slide_title}
Slide Content: {slide_chunk}
Voiceover (Primary Focus): {voiceover}
</slide_context>

<graphics_requirements>
{graphics_definition}
</graphics_requirements>

<review_criteria>
1. **Primary Focus** (Highest Priority): Image clearly illustrates "{voiceover}"
2. **Requirements Compliance**: All elements from graphics requirements are present
3. **Technical Accuracy**: Reference components match exactly if reference was provided
4. **Completeness**: All specified labels, annotations, and elements included
5. **Visual Hierarchy**: Proper emphasis on voiceover concept
6. **Educational Clarity**: Concept is clear and easy to understand
7. **Professional Quality**: No artifacts, consistent styling, appealing composition
"""
        
        if custom_criteria and custom_criteria.strip():
            base_prompt += f"8. **Custom Criteria**: {custom_criteria}\n"
        
        base_prompt += """</review_criteria>

<decision_guidelines>
APPROVE if:
- Image clearly illustrates the voiceover concept
- All graphics requirements are met
- Technical accuracy is maintained
- Professional quality standards met

REJECT if:
- Voiceover concept is unclear or not illustrated
- Missing required elements or labels
- Technical inaccuracies in reference components
- Poor visual quality or significant artifacts
</decision_guidelines>

Return JSON with:
- 'approved': true only if all criteria met
- 'feedback': Specific issues if rejected, or brief confirmation if approved
- 'confidence_score': 0-100 based on how well requirements are met
"""
        
        if previous_feedback:
            base_prompt += f"\n\n<prior_issues>\nThe previous iteration failed due to: {previous_feedback}.\nVerify that these specific issues have been resolved.\n</prior_issues>"

        # Build parts: prompt + generated image + optional reference image for context
        parts = [
            types.Part.from_text(text=base_prompt),
            types.Part.from_bytes(data=generated_bytes, mime_type="image/jpeg")
        ]
        
        # Add reference image for context
        if reference_image:
            ref_bytes = prepare_image_for_gemini(reference_image)
            parts.append(types.Part.from_bytes(data=ref_bytes, mime_type="image/jpeg"))

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
        
        # CRITICAL: Technical Accuracy Check - Only if visually approved
        if approved:
            print("\n🔍 Running Technical Accuracy Check on Generated Image...")
            ta_result = check_technical_accuracy(
                output_content=graphics_definition,
                output_type="image",
                slide_chunk=slide_chunk,
                slide_title=slide_title,
                voiceover=voiceover,
                output_image=generated_image
            )
            
            # Store TA details in metadata
            metadata["technical_accuracy_score"] = ta_result.accuracy_score
            metadata["technical_accuracy_details"] = {
                "missing_components": ta_result.missing_components,
                "incorrect_elements": ta_result.incorrect_elements,
                "label_issues": ta_result.label_issues,
                "recommendations": ta_result.recommendations
            }
            
            if not ta_result.is_accurate:
                print(f"⚠️ Technical accuracy check FAILED (Score: {ta_result.accuracy_score}/100)")
                print(f"Overriding visual approval due to technical inaccuracy")
                
                # Override approval
                approved = False
                
                # Build technical feedback
                technical_feedback = f"""TECHNICAL ACCURACY ISSUES (Score: {ta_result.accuracy_score}/100):

Missing Components: {', '.join(ta_result.missing_components) if ta_result.missing_components else 'None'}
Incorrect Elements: {', '.join(ta_result.incorrect_elements) if ta_result.incorrect_elements else 'None'}
Label Issues: {', '.join(ta_result.label_issues) if ta_result.label_issues else 'None'}

CRITICAL CORRECTIONS NEEDED:
{ta_result.recommendations}

The image must be regenerated with accurate technical representation."""
                
                feedback = technical_feedback
            else:
                print(f"✅ Technical accuracy check PASSED (Score: {ta_result.accuracy_score}/100)")
        else:
            # Visual approval failed, store empty TA data
            metadata["technical_accuracy_score"] = None
            metadata["technical_accuracy_details"] = None
        
        return approved, feedback, metadata
        
    except Exception as e:
        print(f"Review failed: {str(e)}")
        raise

@traceable(
    metadata={
        "agent_name": "graphics_creation",
        "step_name": "Main Loop",
        "function_name": "generate_image_with_review_loop",
        "user_id": st.session_state.get("role", "anonymous") if hasattr(st, 'session_state') else "anonymous",
        "user_email": st.session_state.get("user_email", "anonymous") if hasattr(st, 'session_state') else "anonymous"
    }
)
def generate_image_with_review_loop(
    graphics_definition: str,
    slide_chunk: str,
    slide_title: str,
    voiceover: str,
    reference_image: Optional[Image.Image] = None,
    quick_mode: bool = False,
    custom_criteria: Optional[str] = None,
    aspect_ratio: Optional[str] = None,
    image_size: str = "1K"
) -> Tuple[Image.Image, list]:
    """
    Main loop for image generation with review and correction.
    
    Args:
        graphics_definition: Detailed graphics definition (from generate_graphics_definition)
        slide_chunk: The content of the slide chunk
        slide_title: The title of the slide
        voiceover: The voiceover text being illustrated
        reference_image: Optional reference image for style/content guidance
        quick_mode: If True, skip review loop and generate once
        custom_criteria: Additional review criteria
        aspect_ratio: Aspect ratio for generated image
        image_size: Size of the image to generate
    
    Returns:
        Tuple of (final_image, ui_history_log)
    """
    print(f"Starting image generation with review loop... (Quick Mode: {quick_mode})")
    
    chat_session = None
    client = None
    correction_feedback = None
    generated_image = None
    ui_history_log = []
    
    # Quick Mode: Single generation without review
    if quick_mode:
        print("Quick Mode enabled - skipping review loop")
        generated_image, gen_metadata, chat_session, client = generate_image_from_definition(
            graphics_definition=graphics_definition,
            slide_chunk=slide_chunk,
            slide_title=slide_title,
            voiceover=voiceover,
            reference_image=reference_image,
            correction_feedback=None,
            chat_session=None,
            client=None,
            aspect_ratio=aspect_ratio,
            image_size=image_size
        )
        
        ui_history_log.append({
            "round": 1,
            "type": "generation",
            "generated_image": generated_image,
            "metadata": gen_metadata
        })
        
        print("✓ Quick mode generation complete")
        return generated_image, ui_history_log
    
    # Normal Mode: Full review loop
    for round_num in range(MAX_CORRECTION_ROUNDS):
        print(f"\nRound {round_num + 1}/{MAX_CORRECTION_ROUNDS}")
        
        # 1. Generate
        generated_image, gen_metadata, chat_session, client = generate_image_from_definition(
            graphics_definition=graphics_definition,
            slide_chunk=slide_chunk,
            slide_title=slide_title,
            voiceover=voiceover,
            reference_image=reference_image,
            correction_feedback=correction_feedback,
            chat_session=chat_session,
            client=client,
            aspect_ratio=aspect_ratio,
            image_size=image_size
        )
        
        ui_history_log.append({
            "round": round_num + 1,
            "type": "generation",
            "generated_image": generated_image,
            "metadata": gen_metadata
        })
        
        # 2. Review
        is_approved, feedback, review_metadata = review_generated_image(
            generated_image=generated_image,
            graphics_definition=graphics_definition,
            slide_chunk=slide_chunk,
            slide_title=slide_title,
            voiceover=voiceover,
            reference_image=reference_image,
            previous_feedback=correction_feedback,
            custom_criteria=custom_criteria
        )
        
        # Prepare history entry with TA data
        history_entry = {
            "round": round_num + 1,
            "type": "review",
            "approved": is_approved,
            "feedback": feedback,
            "metadata": review_metadata
        }
        
        # Add technical accuracy data if present
        if "technical_accuracy_score" in review_metadata and review_metadata.get("technical_accuracy_score") is not None:
            ta_score = review_metadata.get("technical_accuracy_score")
            history_entry["technical_accuracy"] = {
                "accuracy_score": ta_score,
                "is_accurate": ta_score >= 90
            }
            # Add detailed TA feedback if available
            if "technical_accuracy_details" in review_metadata and review_metadata.get("technical_accuracy_details"):
                ta_details = review_metadata['technical_accuracy_details']
                history_entry["technical_accuracy"].update({
                    "missing_components": ta_details.get("missing_components", []),
                    "incorrect_elements": ta_details.get("incorrect_elements", []),
                    "label_issues": ta_details.get("label_issues", [])
                })
        
        # Add visual score if present
        if "visual_quality_score" in review_metadata:
            history_entry["visual_score"] = review_metadata["visual_quality_score"]
        
        ui_history_log.append(history_entry)
        
        if is_approved:
            print(f"✓ Image approved in round {round_num + 1}")
            return generated_image, ui_history_log
        else:
            print(f"Round {round_num + 1} Rejected: {feedback[:100]}...")
            correction_feedback = feedback
    
    # Max rounds reached
    print(f"Max rounds reached, using last generated image")
    return generated_image, ui_history_log

