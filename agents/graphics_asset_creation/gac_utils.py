import os
import base64
import re
import pandas as pd
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
from services.llm_service import call_llm_with_retry

load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================
DEFAULT_GENERATOR_MODEL = "gemini-3-pro-image-preview"
DEFAULT_REVIEWER_MODEL = "gemini-3-flash-preview"
DEFAULT_SHARED_DRIVE_PARENT_FOLDER_ID = "1XnPRleC8pUQC2k9G7AT6rPw8rf277Ln9"

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

# Schema for Reviewer
class ReviewResult(BaseModel):
    approved: bool = Field(description="Boolean status. True if the generated graphics asset meets all technical, instructional, and scene definition requirements.")
    feedback: str = Field(description="Concise, objective description of any detected discrepancies, missing elements, or visual artifacts.")
    confidence_score: int = Field(description="Technical confidence score (0-100).")

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

def convert_drive_link_to_direct(drive_link: str) -> str:
    """Convert Google Drive share link to direct download link."""
    if "drive.google.com" not in drive_link:
        return drive_link

    # Extract file ID from various Google Drive link formats
    patterns = [
        r"https://drive\.google\.com/file/d/([a-zA-Z0-9_-]+)/",  # Standard share link
        r"https://drive\.google\.com/open\?id=([a-zA-Z0-9_-]+)",  # Open link
        r"id=([a-zA-Z0-9_-]+)",  # ID parameter
    ]

    file_id = None
    for pattern in patterns:
        match = re.search(pattern, drive_link)
        if match:
            file_id = match.group(1)
            break

    if file_id:
        return f"https://drive.google.com/uc?export=download&id={file_id}"
    else:
        print(f"Could not extract file ID from: {drive_link}")
        return drive_link

def process_drive_image(drive_link: str) -> Optional[bytes]:
    """Download image from Google Drive link."""
    try:
        direct_link = convert_drive_link_to_direct(drive_link)
        response = requests.get(direct_link, timeout=30)
        response.raise_for_status()
        return response.content
    except Exception as e:
        print(f"Error downloading from Google Drive: {e}")
        return None

def process_web_image(url: str) -> Optional[bytes]:
    """Download image from web URL."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        return response.content
    except Exception as e:
        print(f"Error downloading from web: {e}")
        return None

def identify_link_type(link: str) -> str:
    """Identify the type of link provided."""
    if not link:
        return "empty"

    link = link.strip()

    if "drive.google.com" in link:
        return "google_drive"
    elif link.startswith(("http://", "https://")):
        return "web_url"
    elif os.path.exists(link):
        return "local_file"
    else:
        return "unknown"

# =============================================================================
# TECHNICAL ACCURACY CHECKER
# =============================================================================

class TechnicalAccuracyResult(BaseModel):
    is_accurate: bool = Field(description="True if the output is technically accurate according to the slide content and voiceover focus.")
    accuracy_score: int = Field(description="Technical accuracy score (0-100). Must be >= 90 to pass.")
    missing_components: List[str] = Field(default_factory=list, description="List of technical components mentioned in slide content but missing from the output.")
    incorrect_elements: List[str] = Field(default_factory=list, description="List of technical inaccuracies or misrepresentations found.")
    label_issues: List[str] = Field(default_factory=list, description="List of label/text accuracy issues.")
    recommendations: str = Field(description="Specific recommendations to improve technical accuracy.")

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

        else:
            raise ValueError(f"Unknown output_type: {output_type}")

        # Build parts for LLM request
        parts = [types.Part.from_text(text=validation_prompt)]

        # Call LLM for validation
        from agents.graphics_asset_creation.automated.llm_call_tracker import tracker
        with tracker.call(DEFAULT_REVIEWER_MODEL, "Technical Accuracy Check") as usage:
            response = call_llm_with_retry(
                client.models.generate_content,
                model=DEFAULT_REVIEWER_MODEL,
                contents=[types.Content(role="user", parts=parts)],
                config=types.GenerateContentConfig(
                    temperature=0.2,  # Very low temperature for consistent, strict validation
                    response_mime_type="application/json",
                    response_schema=TechnicalAccuracyResult
                )
            )
            usage.set_response(response)

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
        print("Reference Image Analysis: Gemini client unavailable.")
        return None

    try:
        print(f"\n{'='*60}")
        print("ANALYZING REFERENCE IMAGE QUALITY")
        print(f"{'='*60}")

        # Load and prepare image
        if isinstance(reference_image_path, str):
            if reference_image_path.startswith(("http://", "https://")):
                if "drive.google.com" in reference_image_path:
                    image_data = process_drive_image(reference_image_path)
                else:
                    image_data = process_web_image(reference_image_path)
                if not image_data:
                    print("Failed to load reference image")
                    return None
                reference_image = Image.open(BytesIO(image_data))
            else:
                if not os.path.exists(reference_image_path):
                    print(f"Reference image file not found: {reference_image_path}")
                    return None
                reference_image = Image.open(reference_image_path)
        elif isinstance(reference_image_path, Image.Image):
            reference_image = reference_image_path
        else:
            print("Invalid reference image input")
            return None

        image_bytes = prepare_image_for_gemini(reference_image)

        analysis_prompt = f"""You are an expert educational graphics analyst. Analyze this reference image for its suitability as an educational asset.

SLIDE CONTEXT:
Title: {slide_title}
Content: {slide_chunk}
Voiceover Focus: {voiceover}

ANALYSIS REQUIREMENTS:
1. **Educational Value**: Does the image clearly illustrate the voiceover focus and slide content?
2. **Technical Accuracy**: Are all components, labels, and relationships technically correct?
3. **Visual Clarity**: Is the image clear, well-lit, and easy to understand?
4. **Presentation Quality**: Is it professional and suitable for educational use?

PROVIDE SPECIFIC RECOMMENDATIONS for any issues found. Focus on what changes would make this a better educational asset.

If the image is already excellent, state that clearly."""

        parts = [
            types.Part.from_text(text=analysis_prompt),
            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
        ]

        from agents.graphics_asset_creation.automated.llm_call_tracker import tracker
        with tracker.call(DEFAULT_REVIEWER_MODEL, "Reference Image Analysis") as usage:
            response = call_llm_with_retry(
                client.models.generate_content,
                model=DEFAULT_REVIEWER_MODEL,
                contents=[types.Content(role="user", parts=parts)],
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    max_output_tokens=1000
                )
            )
            usage.set_response(response)

        analysis_result = response.text.strip()
        print(f"Analysis Result: {analysis_result[:200]}...")
        return analysis_result

    except Exception as e:
        print(f"Reference Image Analysis Error: {e}")
        return None

# =============================================================================
# GRAPHICS DEFINITION GENERATION
# =============================================================================

def generate_graphics_definition(
    slide_title: str,
    slide_chunk: str,
    voiceover: str,
    visual_instruction: str,
    reference_image: Optional[Image.Image] = None
) -> Optional[str]:
    """
    Generate a detailed graphics definition for creating educational visuals.

    Args:
        slide_title: Title of the slide
        slide_chunk: Content of the slide
        voiceover: Voiceover text
        visual_instruction: Specific visual requirements
        reference_image: Optional reference image

    Returns:
        Graphics definition string or None if failed
    """
    client = _get_client()
    if not client:
        print("Graphics Definition Generation: Gemini client unavailable.")
        return None

    try:
        print(f"\n{'='*60}")
        print("GENERATING GRAPHICS DEFINITION")
        print(f"{'='*60}")

        prompt = f"""Create a detailed graphics definition for an educational image that illustrates the slide content.

SLIDE CONTEXT:
Title: {slide_title}
Content: {slide_chunk}
Voiceover: {voiceover}
Visual Instruction: {visual_instruction}

REQUIREMENTS:
- Define what should be shown in the image
- Specify technical components, labels, and relationships
- Include visual style and layout requirements
- Ensure technical accuracy and educational clarity
- Follow the styling guide: {styling_guide[:500]}...

OUTPUT FORMAT:
Provide a clear, detailed description of what the final image should contain."""

        parts = [types.Part.from_text(text=prompt)]

        if reference_image:
            image_bytes = prepare_image_for_gemini(reference_image)
            parts.append(types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"))

        response = client.models.generate_content(
            model=DEFAULT_GENERATOR_MODEL,
            contents=[types.Content(role="user", parts=parts)],
            config=types.GenerateContentConfig(
                temperature=0.4,
                max_output_tokens=2000
            )
        )

        definition = response.text.strip()
        print(f"Generated Definition: {definition[:200]}...")
        return definition

    except Exception as e:
        print(f"Graphics Definition Generation Error: {e}")
        return None

# =============================================================================
# IMAGE GENERATION WITH REVIEW LOOP
# =============================================================================

def generate_image_with_review_loop(
    graphics_definition: str,
    slide_title: str,
    slide_chunk: str,
    voiceover: str,
    visual_instruction: str,
    reference_image: Optional[Image.Image] = None,
    max_rounds: int = 5
) -> Tuple[Optional[Image.Image], List[Dict[str, Any]]]:
    """
    Generate image with iterative review and correction loop.

    Returns:
        Tuple of (final_image, history_list)
    """
    client = _get_client()
    if not client:
        print("Image Generation: Gemini client unavailable.")
        return None, []

    history = []
    current_image = None

    try:
        for round_num in range(1, max_rounds + 1):
            print(f"\n--- GENERATION ROUND {round_num} ---")

            prompt = f"""Generate an educational image based on this graphics definition.

GRAPHICS DEFINITION:
{graphics_definition}

SLIDE CONTEXT:
Title: {slide_title}
Content: {slide_chunk}
Voiceover: {voiceover}
Visual Instruction: {visual_instruction}

STYLING REQUIREMENTS:
{styling_guide}

TECHNICAL ACCURACY REQUIREMENTS:
- All components must be technically correct
- Labels and relationships must be accurate
- Image must clearly illustrate the voiceover focus"""

            if round_num > 1 and current_image:
                # Add review feedback to prompt
                last_review = history[-1].get('review', {})
                feedback = last_review.get('feedback', '')
                prompt += f"\n\nPREVIOUS ATTEMPT FEEDBACK:\n{feedback}\n\nPlease correct these issues in the new image."

            parts = [types.Part.from_text(text=prompt)]

            if reference_image and round_num == 1:
                ref_bytes = prepare_image_for_gemini(reference_image)
                parts.append(types.Part.from_bytes(data=ref_bytes, mime_type="image/jpeg"))
                parts.append(types.Part.from_text(text="Reference image for inspiration (do not copy directly)"))

            response = client.models.generate_content(
                model=DEFAULT_GENERATOR_MODEL,
                contents=[types.Content(role="user", parts=parts)],
                config=types.GenerateContentConfig(
                    temperature=0.7,
                    response_modalities=["image"]
                )
            )

            # Extract image from response
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'inline_data') and part.inline_data:
                    image_data = part.inline_data.data
                    current_image = Image.open(BytesIO(image_data))
                    break

            if not current_image:
                print("No image generated")
                return None, history

            # Technical accuracy check
            accuracy_result = check_technical_accuracy(
                output_content=graphics_definition,
                output_type="definition",
                slide_chunk=slide_chunk,
                slide_title=slide_title,
                voiceover=voiceover,
                visual_instruction=visual_instruction,
                output_image=current_image
            )

            round_data = {
                'round': round_num,
                'image': current_image,
                'review': {
                    'approved': accuracy_result.is_accurate,
                    'feedback': accuracy_result.recommendations,
                    'confidence_score': accuracy_result.accuracy_score
                }
            }
            history.append(round_data)

            if accuracy_result.is_accurate:
                print(f"✅ Image approved in round {round_num}")
                return current_image, history
            else:
                print(f"❌ Image rejected in round {round_num}: {accuracy_result.recommendations}")

        print("Max rounds reached without approval")
        return current_image, history

    except Exception as e:
        print(f"Image Generation Error: {e}")
        return None, history

# Additional stage configuration (mirrors gac.py)
EDITING_STAGES = {
    "stage1": {
        "name": "Geometry & Perspective (The Skeleton)",
        "description": "FIRST TRANSFORMATION: Establish the fundamental spatial structure. Change viewing angle, camera position, and overall geometric composition.",
        "fields": ["perspective_and_camera"]
    },
    "stage2": {
        "name": "Subject & Content (The Flesh)",
        "description": "SECOND TRANSFORMATION: Modify the primary subjects, objects, and compositional elements within the established geometry.",
        "fields": ["subject_focus"]
    },
    "stage3": {
        "name": "Environment & Lighting (The Atmosphere)",
        "description": "THIRD TRANSFORMATION: Set the environmental context and lighting conditions that define mood and realism.",
        "fields": ["lighting_and_environment", "background_setting"]
    },
    "stage4": {
        "name": "Stylization (The Skin)",
        "description": "FOURTH TRANSFORMATION: Apply the final visual style transformation. Convert to illustration, 3D render, or other distinct art style.",
        "fields": ["visual_style", "color_grading"]
    }
}

EDITING_OPTIONS = {
    "perspective_and_camera": "Perspective and Camera",
    "subject_focus": "Subject Focus",
    "lighting_and_environment": "Lighting and Environment",
    "background_setting": "Background Setting",
    "visual_style": "Visual Style",
    "color_grading": "Color Grading",
    "additional_comments": "Additional Comments",
}


def parse_subsegments(final_graphics_definition: str) -> List[dict]:
    """
    Parse the final_graphics_definition column to extract sub-segments.

    Args:
        final_graphics_definition: String containing multiple sub-segments separated by ----

    Returns:
        List of dictionaries with keys: voiceover_focus, visual_instruction, reference_link
    """
    if not final_graphics_definition or (hasattr(pd, 'isna') and pd.isna(final_graphics_definition)):
        return []

    subsegments = []

    # Split by segment separator if exists
    segments = re.split(r'={50,}[\r\n]+SEGMENT \d+[\r\n]+={50,}', str(final_graphics_definition))

    for segment in segments:
        if not segment.strip():
            continue

        # Split by ---- to get individual sub-segments
        parts = segment.split('----')

        for part in parts:
            if not part.strip():
                continue

            subseg = {}

            # Extract "When VO:"
            vo_match = re.search(r'When VO:\s*["\']?(.+?)["\']?(?:\n|$)', part, re.IGNORECASE)
            if vo_match:
                subseg['voiceover_focus'] = vo_match.group(1).strip()

            # Extract "Visual Instructions:"
            vi_match = re.search(
                r'Visual Instructions?:\s*(.+?)(?=\n\n|Graphics to use:|Selection Justification:|$)',
                part, re.IGNORECASE | re.DOTALL
            )
            if vi_match:
                subseg['visual_instruction'] = vi_match.group(1).strip()

            # Extract "Graphics to use:"
            # NOTE: (snapshot) may appear on the NEXT line after the URL.
            graphics_match = re.search(
                r'Graphics to use:\s*(.+?)(?:\n\s*\(snapshot\))?(?=\n\n|Selection Justification:|$)',
                part, re.IGNORECASE | re.DOTALL
            )
            if graphics_match:
                raw_link = graphics_match.group(1).strip()
                # Re-attach (snapshot) tag if it was on the next line
                snapshot_suffix_match = re.search(
                    r'Graphics to use:\s*.+?\n\s*(\(snapshot\))',
                    part, re.IGNORECASE
                )
                if snapshot_suffix_match:
                    raw_link = raw_link.split('\n')[0].strip() + ' (snapshot)'
                subseg['reference_link'] = raw_link

            # Only add if we have all three fields
            if all(k in subseg for k in ['voiceover_focus', 'visual_instruction', 'reference_link']):
                subsegments.append(subseg)

    return subsegments


def is_valid_web_image_link(url: str) -> bool:
    """
    Check if URL is a valid external web image link.

    - Allows Google Drive links that carry the `(snapshot)` tag.
    - Blocks links explicitly marked `(AI Generated)`.
    - Blocks YouTube, bare Google Drive/Docs links (no snapshot), and non-HTTP URLs.
    """
    if not url:
        return False

    try:
        # pandas NaN check without hard dependency on pd.isna at call time
        if pd.isna(url):
            return False
    except Exception:
        pass

    url = str(url).strip()

    # Explicitly skip links marked as AI-generated placeholders.
    if '(ai generated)' in url.lower():
        return False

    if not url.startswith(('http://', 'https://')):
        return False

    # Allow Drive links that have been confirmed as snapshots
    if '(snapshot)' in url.lower():
        return True

    # Block YouTube and plain Drive / Docs links
    if any(domain in url.lower() for domain in
           ['youtube.com', 'youtu.be', 'drive.google.com', 'docs.google.com']):
        return False

    return True


def get_or_create_drive_folder(drive, folder_name: str, parent_folder_id: str = None) -> Optional[str]:
    """
    Create a folder in Google Drive with collision-safe naming.

    Primary behavior:
    - Creates under shared-drive parent (provided parent_folder_id or default parent)
    - If folder name already exists in that parent, creates a suffixed name (e.g., "Name (2)")

    Fallback behavior:
    - If shared-drive creation fails, creates in My Drive root
    - Applies "anyone with link" read permission in fallback mode

    Args:
        drive: Authenticated PyDrive2 GoogleDrive instance
        folder_name: Base folder name to create
        parent_folder_id: Optional parent folder ID override

    Returns:
        Folder ID string, or None on failure
    """
    def _normalize_name(name: str) -> str:
        return (name or "").strip()

    def _find_available_name(existing_names: set, base_name: str) -> str:
        if base_name not in existing_names:
            return base_name
        suffix = 2
        while True:
            candidate = f"{base_name} ({suffix})"
            if candidate not in existing_names:
                return candidate
            suffix += 1

    def _list_existing_names_in_parent(target_parent_id: Optional[str], include_all_drives: bool) -> set:
        query = "mimeType='application/vnd.google-apps.folder' and trashed=false"
        if target_parent_id:
            query += f" and '{target_parent_id}' in parents"

        list_params: Dict[str, Any] = {'q': query}
        if include_all_drives:
            list_params.update({
                'supportsAllDrives': True,
                'includeItemsFromAllDrives': True,
            })

        file_list = drive.ListFile(list_params).GetList()
        return {str(f.get('title', '')).strip() for f in file_list if f.get('title')}

    def _create_folder(target_name: str, target_parent_id: Optional[str], shared_drive_mode: bool) -> str:
        folder_metadata: Dict[str, Any] = {
            'title': target_name,
            'mimeType': 'application/vnd.google-apps.folder'
        }
        if target_parent_id:
            folder_metadata['parents'] = [{'id': target_parent_id}]

        folder = drive.CreateFile(folder_metadata)
        if shared_drive_mode:
            folder.Upload(param={'supportsAllDrives': True})
        else:
            folder.Upload()
            # Required by fallback behavior: My Drive + link sharing.
            folder.InsertPermission({'type': 'anyone', 'value': 'anyone', 'role': 'reader'})
        return folder['id']

    try:
        base_name = _normalize_name(folder_name)
        if not base_name:
            raise ValueError("Folder name cannot be empty")

        target_parent_id = parent_folder_id or DEFAULT_SHARED_DRIVE_PARENT_FOLDER_ID

        # Primary path: create under shared-drive parent, with suffix if needed.
        try:
            existing_names = _list_existing_names_in_parent(
                target_parent_id=target_parent_id,
                include_all_drives=True,
            )
            final_name = _find_available_name(existing_names, base_name)
            folder_id = _create_folder(
                target_name=final_name,
                target_parent_id=target_parent_id,
                shared_drive_mode=True,
            )
            print(f"✓ Created shared-drive folder: {final_name} (ID: {folder_id})")
            return folder_id
        except Exception as shared_err:
            print(
                "Shared-drive folder creation failed. "
                f"Falling back to My Drive. Error: {shared_err}"
            )

        # Fallback path: create in My Drive root, with suffix + public link sharing.
        existing_names = _list_existing_names_in_parent(
            target_parent_id=None,
            include_all_drives=False,
        )
        final_name = _find_available_name(existing_names, base_name)
        folder_id = _create_folder(
            target_name=final_name,
            target_parent_id=None,
            shared_drive_mode=False,
        )
        print(f"✓ Created My Drive fallback folder: {final_name} (ID: {folder_id})")
        return folder_id

    except Exception as e:
        print(f"Error getting/creating folder '{folder_name}': {e}")
        return None


def upload_image_to_drive(
    image: Image.Image,
    filename: str,
    drive,
    folder_id: str = None
) -> str:
    """
    Upload a PIL Image to Google Drive and return its shareable view link.

    Args:
        image: PIL Image to upload
        filename: Filename for the uploaded file
        drive: Authenticated PyDrive2 GoogleDrive instance
        folder_id: Drive folder ID to upload into

    Returns:
        Shareable view URL string (https://drive.google.com/file/d/FILE_ID/view),
        or empty string on failure.
    """
    try:
        img_buffer = BytesIO()
        image.save(img_buffer, format='PNG')
        img_buffer.seek(0)

        metadata: Dict[str, Any] = {'title': filename, 'mimeType': 'image/png'}
        if folder_id:
            metadata['parents'] = [{'id': folder_id}]

        file = drive.CreateFile(metadata)
        file.content = img_buffer
        file.Upload()
        file.InsertPermission({'type': 'anyone', 'value': 'anyone', 'role': 'reader'})

        return f"https://drive.google.com/file/d/{file['id']}/view"

    except Exception as e:
        print(f"Error uploading '{filename}' to Drive: {e}")
        return ""


def create_graphics_asset_from_reference(
    reference_image,
    slide_title: str,
    slide_content: str,
    voiceover_focus: str,
    visual_instruction: str,
    quick_mode: bool = False,
    custom_review_criteria: Optional[str] = None,
    aspect_ratio: Optional[str] = None,
    image_size: str = "1K",
    use_multi_stage: bool = True
) -> Tuple[Optional[Image.Image], List[Dict[str, Any]]]:
    """
    Main workflow: load reference image, generate editing instructions, and
    return the edited graphics asset together with its history log.

    Args:
        reference_image: PIL Image, file path, or URL (including Google Drive)
        slide_title: Title of the slide
        slide_content: Body text of the slide
        voiceover_focus: The specific voiceover line being illustrated
        visual_instruction: Additional visual requirements from the author
        quick_mode: Skip the automated review loop for speed
        custom_review_criteria: Extra criteria for the reviewer agent
        aspect_ratio: Target aspect ratio (e.g. "16:9")
        image_size: Resolution setting ("1K" / "2K" / "4K")
        use_multi_stage: Whether to apply multi-stage editing (True = higher quality)

    Returns:
        (final_image, history_list) tuple
    """
    # Lazy import to avoid circular dependency
    try:
        from agents.graphics_asset_creation.image_editing.image_editing_grok import (
            image_editing_with_review_loop
        )
    except ImportError:
        print("create_graphics_asset_from_reference: image_editing_grok not available.")
        return None, []

    # Load the reference image if it isn't already a PIL Image
    if not isinstance(reference_image, Image.Image):
        if isinstance(reference_image, str) and reference_image.startswith(('http://', 'https://')):
            if 'drive.google.com' in reference_image:
                img_bytes = process_drive_image(reference_image)
            else:
                img_bytes = process_web_image(reference_image)
            if not img_bytes:
                print("create_graphics_asset_from_reference: failed to load reference image.")
                return None, []
            reference_image = Image.open(BytesIO(img_bytes))
        elif isinstance(reference_image, str):
            if not os.path.exists(reference_image):
                print(f"create_graphics_asset_from_reference: file not found: {reference_image}")
                return None, []
            reference_image = Image.open(reference_image)
        else:
            # Streamlit UploadedFile or similar file-like object
            try:
                reference_image = Image.open(reference_image)
            except Exception as e:
                print(f"create_graphics_asset_from_reference: cannot open reference image: {e}")
                return None, []

    editing_instructions = {
        "subject_focus": visual_instruction,
        "additional_comments": f"Slide: {slide_title}. Voiceover: {voiceover_focus}. {slide_content[:300]}"
    }

    return image_editing_with_review_loop(
        reference_image=reference_image,
        editing_instructions=editing_instructions,
        quick_mode=quick_mode,
        custom_criteria=custom_review_criteria,
        aspect_ratio=aspect_ratio,
        image_size=image_size,
    )


def display_editing_instructions(editing_instructions: Dict[str, str], show_stages: bool = True):
    """
    Display editing instructions in the Streamlit UI, grouped by processing stage.

    Args:
        editing_instructions: Dictionary of editing instructions
        show_stages: If True, group display by EDITING_STAGES; otherwise flat list
    """
    if not hasattr(st, 'expander'):
        return

    with st.expander("📝 Generated Editing Instructions", expanded=True):
        has_instructions = False

        if show_stages:
            for stage_key in ["stage1", "stage2", "stage3", "stage4"]:
                stage_info = EDITING_STAGES[stage_key]
                stage_fields = stage_info["fields"]

                stage_has_content = any(
                    field in editing_instructions
                    and editing_instructions[field]
                    and editing_instructions[field].strip()
                    for field in stage_fields
                )

                if stage_has_content:
                    has_instructions = True
                    st.markdown(f"### 🔧 {stage_info['name']}")
                    st.caption(stage_info['description'])

                    for field in stage_fields:
                        val = editing_instructions.get(field, "")
                        if val and val.strip():
                            option_name = EDITING_OPTIONS.get(field, field)
                            st.write(f"**{option_name}:**")
                            st.write(val)
                            st.write("")

                    st.divider()
        else:
            for key, value in editing_instructions.items():
                if value and value.strip():
                    has_instructions = True
                    option_name = EDITING_OPTIONS.get(key, key)
                    st.write(f"**{option_name}:**")
                    st.write(value)
                    st.write("")

        if not has_instructions:
            st.info("No edits needed — reference image is already suitable for this slide context.")


def process_sheet_batch(
    sheet_data,
    drive_service,
    output_tab: str = "Asset Creation Test",
    quick_mode: bool = True,
    image_size: str = "1K",
    drive_folder_name: str = "Graphics Asset Creation Output",
    max_workers: int = 2,
    use_multi_stage: bool = True
) -> List[Dict[str, Any]]:
    """
    Batch-process a list of sheet rows, generating graphics assets for each.

    Args:
        sheet_data: List of row dicts with slide_title, slide_content,
                    voiceover_focus, visual_instruction, reference_image keys
        drive_service: Authenticated PyDrive2 GoogleDrive instance
        output_tab: Name of the output worksheet tab
        quick_mode: Skip automated review for speed
        image_size: Resolution setting ("1K" / "2K" / "4K")
        drive_folder_name: Root Drive folder for outputs
        max_workers: Number of parallel workers (reserved for future use)
        use_multi_stage: Use multi-stage editing for higher quality

    Returns:
        List of result dicts with slide_title, status, image_link, history
    """
    results = []

    for row in sheet_data:
        slide_title = row.get('slide_title', '')
        slide_content = row.get('slide_content', '')
        voiceover_focus = row.get('voiceover_focus', '')
        visual_instruction = row.get('visual_instruction', '')
        reference_image = row.get('reference_image', None)

        try:
            print(f"\nProcessing: {slide_title}")
            final_image, history = create_graphics_asset_from_reference(
                reference_image=reference_image,
                slide_title=slide_title,
                slide_content=slide_content,
                voiceover_focus=voiceover_focus,
                visual_instruction=visual_instruction,
                quick_mode=quick_mode,
                image_size=image_size,
                use_multi_stage=use_multi_stage,
            )

            image_link = ""
            if final_image and drive_service:
                folder_id = get_or_create_drive_folder(drive_service, drive_folder_name)
                if folder_id:
                    safe_title = re.sub(r'[^\w\s-]', '', slide_title)[:50]
                    filename = f"{safe_title}.png"
                    image_link = upload_image_to_drive(final_image, filename, drive_service, folder_id)

            results.append({
                'slide_title': slide_title,
                'status': 'success' if final_image else 'failed',
                'image_link': image_link,
                'history': history,
            })

        except Exception as e:
            print(f"Error processing '{slide_title}': {e}")
            results.append({
                'slide_title': slide_title,
                'status': 'error',
                'error': str(e),
                'image_link': '',
                'history': [],
            })

    return results