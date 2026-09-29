
import os
import base64
import time
from io import BytesIO
from PIL import Image
from typing import Optional, Tuple, Dict, List, Any
from dotenv import load_dotenv
import json
import streamlit as st
from pydantic import BaseModel, Field
from langsmith import traceable

# Load environment variables
load_dotenv()

# Import xAI SDK (used for image generation)
try:
    import xai_sdk
    from xai_sdk.chat import user, image as chat_image
except ImportError:
    st.error("xai_sdk not found. Install with: pip install xai-sdk")
    xai_sdk = None

# Import Google GenAI SDK (used for image review)
try:
    from google import genai as google_genai
    from google.genai import types as google_types
except ImportError:
    google_genai = None
    google_types = None

# =============================================================================
# CONFIGURATION
# =============================================================================
DEFAULT_GENERATOR_MODEL = "grok-imagine-image"
DEFAULT_REVIEWER_MODEL = "gemini-3-flash-preview"  # Google Gemini Flash for review
MAX_CORRECTION_ROUNDS = 5

EDITING_OPTIONS = {
    "subject_focus": "Subject Focus",
    "visual_style": "Visual Style",
    "perspective_and_camera": "Perspective and Camera",
    "lighting_and_environment": "Lighting and Environment",
    "background_setting": "Background Setting",
    "color_grading": "Color Grading",
    "additional_comments": "Additional Comments"
}

# Schema for Reviewer
class ReviewResult(BaseModel):
    approved: bool = Field(description="Boolean status. True if the image meets all technical and instructional requirements.")
    feedback: str = Field(description="Concise, objective description of any detected discrepancies or artifacts.")
    confidence_score: int = Field(description="Technical confidence score (0-100).")

def _get_client() -> Optional[xai_sdk.Client]:
    """Get xAI client (for image generation) with API key from environment."""
    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        st.error("Missing XAI_API_KEY. Please set it in your .env file")
        return None
    try:
        return xai_sdk.Client(api_key=api_key, timeout=3600)
    except Exception as e:
        print(f"Failed to initialize xAI client: {e}")
        return None


def _get_gemini_client():
    """Get Google GenAI client (for review) with API key from environment."""
    if google_genai is None:
        raise ImportError("google-genai not installed. Run: pip install google-genai")
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        api_key = st.session_state.get("google_api_key")
    if not api_key:
        raise ValueError("Missing GOOGLE_API_KEY. Please set it in your .env file or provide it via st.session_state['google_api_key'].")
    try:
        return google_genai.Client(api_key=api_key)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Gemini client: {e}")


def _retry_api_call(fn, max_retries: int = 3, base_delay: float = 2.0, label: str = "API call"):
    """
    Retry an API call with exponential backoff.

    Args:
        fn: A zero-argument callable that performs the API call.
        max_retries: Maximum number of attempts (default 3).
        base_delay: Initial delay in seconds; doubles each retry.
        label: Human-readable label for log messages.

    Returns:
        The return value of *fn* on success.

    Raises:
        The last exception if all retries are exhausted.
    """
    last_exception = None
    for attempt in range(1, max_retries + 1):
        try:
            return fn()
        except Exception as e:
            last_exception = e
            if attempt < max_retries:
                delay = base_delay * (2 ** (attempt - 1))
                print(f"⚠️ {label} failed (attempt {attempt}/{max_retries}): {e}. "
                      f"Retrying in {delay:.1f}s...")
                time.sleep(delay)
            else:
                print(f"❌ {label} failed after {max_retries} attempts: {e}")
    raise last_exception

# =============================================================================
# PROFESSIONAL / TECHNICAL PROMPTS
# =============================================================================

def get_initial_prompt(editing_instructions: Dict[str, str]) -> str:
    """
    Formal prompt for the Generator.
    """
    prompt = """<objective>
Edit the provided reference image according to the specific user instructions below.
</objective>

<technical_constraints>
1. **Fidelity**: Maintain the original image's aspect ratio, perspective, and core visual characteristics unless explicitly instructed to change them.
2. **Grounding**: All generated elements must be visually consistent with the reference image's lighting, texture, and physics.
3. **Accuracy**: Do not introduce hallucinations, visual artifacts, or unrequested elements.
</technical_constraints>

<edit_requests>
"""
    for option_key, instruction in editing_instructions.items():
        if instruction and instruction.strip():
            option_name = EDITING_OPTIONS.get(option_key, option_key)
            prompt += f"\n- {option_name}: {instruction}\n"
    
    prompt += """</edit_requests>

<execution_guidelines>
Analyze the reference image and the instructions. Apply the edits seamlessly. Output the final image.
</execution_guidelines>
"""
    return prompt


def get_feedback_prompt(correction_feedback: str) -> str:
    """
    Formal feedback prompt for the Generator.
    """
    return f"""<status>
The previous output was rejected during quality assurance.
</status>

<detected_issues>
{correction_feedback}
</detected_issues>

<directive>
Regenerate the image. Correct the specific issues listed above while maintaining adherence to all original edit requests.
</directive>"""


def get_reviewer_prompt(editing_instructions: Dict[str, str], custom_criteria: Optional[str] = None) -> str:
    prompt = """<objective>
Perform a quality assurance review comparing the Reference Image (Image 1) and the Generated Edit (Image 2).
</objective>

<verification_criteria>
1. **Instruction Adherence**: Confirm that every specific edit request has been implemented correctly.
2. **Technical Quality**: Verify the absence of AI artifacts (smearing, geometric inconsistencies, floating objects).
3. **Consistency**: Ensure lighting, shadows, and perspective are consistent with the reference image.
"""
    
    if custom_criteria and custom_criteria.strip():
        prompt += f"4. **Custom Criteria**: {custom_criteria}\n"
    
    prompt += """</verification_criteria>

<edit_requests_to_verify>
"""
    for option_key, instruction in editing_instructions.items():
        if instruction and instruction.strip():
            option_name = EDITING_OPTIONS.get(option_key, option_key)
            prompt += f"\n- {option_name}: {instruction}\n"
    prompt += """</edit_requests_to_verify>

<output_requirements>
Return a structured JSON assessment with the following format:
{
  "approved": true/false,
  "feedback": "description of any issues or confirmation of quality",
  "confidence_score": 0-100
}
Set 'approved' to true only if the image is technically sound and fully adheres to instructions.
</output_requirements>
"""
    return prompt

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def prepare_image_for_grok(image: Image.Image, max_size_bytes: int = 3500000) -> str:
    """
    Convert PIL Image to base64 data URI for Grok API.
    Automatically compresses/resizes if image exceeds size limit.
    
    Args:
        image: PIL Image to convert
        max_size_bytes: Maximum size in bytes (default 3.5MB, API limit is 4MB)
        
    Returns:
        Base64 data URI string
    """
    save_img = image.copy()
    if save_img.mode not in ['RGB', 'RGBA']:
        save_img = save_img.convert('RGB')
    
    # Try original size first with JPEG (better compression than PNG)
    buffered = BytesIO()
    save_format = "JPEG" if save_img.mode == 'RGB' else "PNG"
    quality = 95
    
    if save_format == "JPEG":
        save_img.save(buffered, format=save_format, quality=quality, optimize=True)
    else:
        save_img.save(buffered, format=save_format, optimize=True)
    
    img_bytes = buffered.getvalue()
    original_size = len(img_bytes)
    
    # If too large, progressively reduce quality or resize
    if original_size > max_size_bytes:
        print(f"Image too large ({original_size / 1024 / 1024:.2f} MB), compressing...")
        
        # Try reducing quality first (JPEG only)
        if save_format == "JPEG":
            for quality in [85, 75, 65, 55]:
                buffered = BytesIO()
                save_img.save(buffered, format="JPEG", quality=quality, optimize=True)
                img_bytes = buffered.getvalue()
                if len(img_bytes) <= max_size_bytes:
                    print(f"Compressed to {len(img_bytes) / 1024 / 1024:.2f} MB (quality={quality})")
                    break
        
        # If still too large, resize the image
        if len(img_bytes) > max_size_bytes:
            scale_factor = (max_size_bytes / len(img_bytes)) ** 0.5
            new_width = int(save_img.width * scale_factor * 0.9)  # 0.9 for safety margin
            new_height = int(save_img.height * scale_factor * 0.9)
            
            print(f"Resizing from {save_img.width}x{save_img.height} to {new_width}x{new_height}")
            save_img = save_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            buffered = BytesIO()
            if save_format == "JPEG":
                save_img.save(buffered, format="JPEG", quality=85, optimize=True)
            else:
                save_img.save(buffered, format="PNG", optimize=True)
            
            img_bytes = buffered.getvalue()
            print(f"Final size: {len(img_bytes) / 1024 / 1024:.2f} MB")
    
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    mime_type = "image/jpeg" if save_format == "JPEG" else "image/png"
    return f"data:{mime_type};base64,{img_base64}"

def image_from_base64(base64_string: str) -> Image.Image:
    if ',' in base64_string:
        base64_string = base64_string.split(',', 1)[1]
    image_bytes = base64.b64decode(base64_string)
    return Image.open(BytesIO(image_bytes))

def image_from_url_or_base64(response) -> Image.Image:
    """
    Convert Grok response to PIL Image.
    Handles both URL and base64 responses.
    """
    if hasattr(response, 'image') and response.image:
        # Base64 response
        return Image.open(BytesIO(response.image))
    elif hasattr(response, 'url') and response.url:
        # URL response - need to download
        import requests
        img_response = requests.get(response.url)
        return Image.open(BytesIO(img_response.content))
    else:
        raise ValueError("No image found in response")

# =============================================================================
# GENERATOR AGENT
# =============================================================================

@traceable(
    metadata={
        "agent_name": "image_editing_grok",
        "step_name": "Generator",
        "function_name": "generate_edited_image",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous")
    }
)
def generate_edited_image(
    reference_image: Image.Image,
    editing_instructions: Dict[str, str],
    correction_feedback: Optional[str] = None,
    previous_image: Optional[Image.Image] = None,
    aspect_ratio: Optional[str] = None,
    image_size: str = "1K",
    system_instruction: Optional[str] = None
) -> Tuple[Image.Image, Dict, Optional[str]]:
    """
    Generate edited image using Grok's image editing capabilities.
    
    Args:
        reference_image: The original image to edit
        editing_instructions: Dictionary of editing instructions
        correction_feedback: Feedback from previous review (for corrections)
        previous_image: Previously generated image (for iterative editing)
        aspect_ratio: Desired aspect ratio (e.g., "16:9", "1:1", "auto")
        image_size: Image resolution ("1K", "2K", or "4K" - will be converted to Grok format)
        system_instruction: Optional system instruction
        
    Returns:
        Tuple of (edited_image, metadata, image_data_uri)
    """
    try:
        client = _get_client()
        if client is None:
            raise ValueError("Missing XAI_API_KEY.")
        
        print(f"Starting Grok image editing generation...")
        
        # Build the prompt
        if correction_feedback:
            prompt = get_feedback_prompt(correction_feedback)
        else:
            prompt = get_initial_prompt(editing_instructions)
        
        # Add system instruction if provided
        if system_instruction:
            prompt = f"{system_instruction}\n\n{prompt}"
        
        # Determine which image to use as source
        # For corrections, use the previous generated image
        # For initial edit, use the reference image
        source_image = previous_image if previous_image is not None else reference_image
        image_data_uri = prepare_image_for_grok(source_image)
        
        # Convert image_size to Grok resolution format (lowercase, max 2k)
        # Grok only supports "1k" and "2k"
        resolution = image_size.lower()
        if resolution not in ["1k", "2k"]:
            resolution = "1k"  # Default to 1k if unsupported size
        
        # Build parameters for image.sample()
        params = {
            "prompt": prompt,
            "model": DEFAULT_GENERATOR_MODEL,
            "image_url": image_data_uri,
            "resolution": resolution,
        }
        
        if aspect_ratio:
            params["aspect_ratio"] = aspect_ratio
        
        # Generate the edited image
        print(f"Calling Grok API with parameters: model={DEFAULT_GENERATOR_MODEL}, resolution={resolution}, aspect_ratio={aspect_ratio}")
        response = _retry_api_call(
            lambda: client.image.sample(**params),
            label="Grok image generation"
        )
        
        # Extract the edited image
        edited_image = image_from_url_or_base64(response)
        
        # Build metadata
        metadata = {
            "model": DEFAULT_GENERATOR_MODEL,
            "image_size": image_size,
            "resolution": resolution,
            "aspect_ratio": aspect_ratio,
            "moderation_passed": getattr(response, 'respect_moderation', True),
            "actual_model": getattr(response, 'model', DEFAULT_GENERATOR_MODEL)
        }
        
        print("✓ Image editing generated")
        return edited_image, metadata, image_data_uri
        
    except Exception as e:
        st.error(f"Image generation failed: {str(e)}")
        raise

# =============================================================================
# REVIEWER AGENT
# =============================================================================

@traceable(
    metadata={
        "agent_name": "image_editing_grok",
        "step_name": "Reviewer",
        "function_name": "review_edited_image",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous")
    }
)
def review_edited_image(
    reference_image: Image.Image,
    edited_image: Image.Image,
    editing_instructions: Dict[str, str],
    previous_feedback: Optional[str] = None,
    custom_criteria: Optional[str] = None,
    system_instruction: Optional[str] = None
) -> Tuple[bool, Optional[str], Dict]:
    """
    Review edited image using Google Gemini's vision capabilities.

    Args:
        reference_image: The original reference image
        edited_image: The generated edited image to review
        editing_instructions: The editing instructions that were provided
        previous_feedback: Feedback from previous review round
        custom_criteria: Additional custom review criteria
        system_instruction: Optional system instruction

    Returns:
        Tuple of (is_approved, feedback, metadata)
    """
    try:
        client = _get_gemini_client()
        print(f"Reviewing edited image with Gemini ({DEFAULT_REVIEWER_MODEL})...")

        # Convert images to JPEG bytes for Gemini
        def _to_jpeg_bytes(img: Image.Image) -> bytes:
            buf = BytesIO()
            save_img = img.copy()
            if save_img.mode != 'RGB':
                save_img = save_img.convert('RGB')
            # Downscale if very large to stay within Gemini limits
            max_dim = 2048
            w, h = save_img.size
            if w > max_dim or h > max_dim:
                ratio = min(max_dim / w, max_dim / h)
                save_img = save_img.resize((int(w * ratio), int(h * ratio)), Image.Resampling.LANCZOS)
            save_img.save(buf, format="JPEG", quality=90)
            return buf.getvalue()

        reference_bytes = _to_jpeg_bytes(reference_image)
        edited_bytes = _to_jpeg_bytes(edited_image)

        # Build the review prompt
        base_prompt = get_reviewer_prompt(editing_instructions, custom_criteria)
        if previous_feedback:
            base_prompt += (
                f"\n\n<prior_defect_correction>\nThe previous iteration failed due to: "
                f"{previous_feedback}.\nVerify that this specific issue has been resolved.\n"
                f"</prior_defect_correction>"
            )
        if system_instruction:
            base_prompt = f"{system_instruction}\n\n{base_prompt}"

        # Build multimodal parts: [text_intro, ref_image, edit_image, prompt]
        parts = [
            google_types.Part.from_text(text="Reference Image (Image 1):"),
            google_types.Part.from_bytes(data=reference_bytes, mime_type="image/jpeg"),
            google_types.Part.from_text(text="Generated Edit (Image 2):"),
            google_types.Part.from_bytes(data=edited_bytes, mime_type="image/jpeg"),
            google_types.Part.from_text(text=base_prompt),
        ]

        response = _retry_api_call(
            lambda: client.models.generate_content(
                model=DEFAULT_REVIEWER_MODEL,
                contents=[google_types.Content(role="user", parts=parts)],
                config=google_types.GenerateContentConfig(
                    temperature=0.1,
                )
            ),
            label="Gemini image review"
        )

        response_text = response.text or ""

        # Parse JSON from the response
        try:
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                json_str = response_text.split("```")[1].split("```")[0].strip()
            else:
                json_str = response_text

            data = json.loads(json_str)
            approved = bool(data.get("approved", False))
            feedback = data.get("feedback", "")
            score = int(data.get("confidence_score", 0))
        except Exception as parse_err:
            print(f"JSON parsing error: {parse_err}. Response snippet: {response_text[:200]}")
            # Keyword fallback
            response_lower = response_text.lower()
            approved = "\"approved\": true" in response_lower or "approved: true" in response_lower
            feedback = response_text
            score = 0

        metadata = {
            "model": DEFAULT_REVIEWER_MODEL,
            "confidence_score": score
        }

        print(f"Review result: Approved={approved}, Score={score}")
        return approved, feedback, metadata

    except Exception as e:
        st.error(f"Review failed: {str(e)}")
        raise

# =============================================================================
# MAIN LOOP
# =============================================================================

@traceable(
    metadata={
        "agent_name": "image_editing_grok",
        "step_name": "Main Loop",
        "function_name": "image_editing_with_review_loop",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous")
    }
)
def image_editing_with_review_loop(
    reference_image: Image.Image,
    editing_instructions: Dict[str, str],
    quick_mode: bool = False,
    custom_criteria: Optional[str] = None,
    aspect_ratio: Optional[str] = None,
    image_size: str = "1K",
    system_instruction: Optional[str] = None
) -> Tuple[Image.Image, list, list]:
    """
    Main loop for image editing with iterative review.
    
    Args:
        reference_image: The original image to edit
        editing_instructions: Dictionary of editing instructions
        quick_mode: If True, skip review loop (single generation)
        custom_criteria: Custom review criteria
        aspect_ratio: Desired aspect ratio
        image_size: Image resolution ("1K", "2K", or "4K")
        system_instruction: Optional system instruction
        
    Returns:
        Tuple of (final_edited_image, ui_history_log, conversation_history)
    """
    print(f"Starting Grok image editing with review loop... (Quick Mode: {quick_mode})")
    
    correction_feedback = None
    edited_image = None
    previous_image = None
    ui_history_log = []
    conversation_history = []  # For compatibility with original interface
    
    # Quick Mode: Single generation without review
    if quick_mode:
        print("Quick Mode enabled - skipping review loop")
        with st.spinner("Generating edited image with Grok..."):
            edited_image, gen_metadata, _ = generate_edited_image(
                reference_image=reference_image,
                editing_instructions=editing_instructions,
                correction_feedback=None,
                previous_image=None,
                aspect_ratio=aspect_ratio,
                image_size=image_size,
                system_instruction=system_instruction
            )
        
        ui_history_log.append({
            "round": 1,
            "type": "generation",
            "edited_image": edited_image,
            "metadata": gen_metadata
        })
        
        print("✓ Quick mode generation complete")
        return edited_image, ui_history_log, conversation_history
    
    # Normal Mode: Full review loop with iterative refinement
    for round_num in range(MAX_CORRECTION_ROUNDS):
        print(f"\nRound {round_num + 1}/{MAX_CORRECTION_ROUNDS}")
        
        # 1. Generate
        with st.spinner(f"Generating edited image... (Round {round_num + 1})"):
            edited_image, gen_metadata, _ = generate_edited_image(
                reference_image=reference_image,
                editing_instructions=editing_instructions,
                correction_feedback=correction_feedback,
                previous_image=previous_image,
                aspect_ratio=aspect_ratio,
                image_size=image_size,
                system_instruction=system_instruction
            )
        
        ui_history_log.append({
            "round": round_num + 1,
            "type": "generation",
            "edited_image": edited_image,
            "metadata": gen_metadata
        })
        
        # 2. Review
        with st.spinner("Reviewing edited image with Grok vision..."):
            is_approved, feedback, review_metadata = review_edited_image(
                reference_image=reference_image,
                edited_image=edited_image,
                editing_instructions=editing_instructions,
                previous_feedback=correction_feedback,
                custom_criteria=custom_criteria,
                system_instruction=system_instruction
            )
        
        ui_history_log.append({
            "round": round_num + 1,
            "type": "review",
            "approved": is_approved,
            "feedback": feedback,
            "metadata": review_metadata
        })
        
        if is_approved:
            print(f"✓ Approved in round {round_num + 1}")
            return edited_image, ui_history_log, conversation_history
        else:
            print(f"Round {round_num + 1} Rejected: {feedback[:100]}...")
            correction_feedback = feedback
            # Store the current image for the next iteration
            previous_image = edited_image
    
    print(f"Max rounds reached")
    return edited_image, ui_history_log, conversation_history

# =============================================================================
# BATCH GENERATION (Multiple Variations)
# =============================================================================

@traceable(
    metadata={
        "agent_name": "image_editing_grok",
        "step_name": "Batch Generator",
        "function_name": "generate_multiple_variations",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous")
    }
)
def generate_multiple_variations(
    reference_image: Image.Image,
    editing_instructions: Dict[str, str],
    n: int = 4,
    aspect_ratio: Optional[str] = None,
    image_size: str = "1K",
    system_instruction: Optional[str] = None
) -> List[Tuple[Image.Image, Dict]]:
    """
    Generate multiple variations of an edited image in a single request.
    
    Args:
        reference_image: The original image to edit
        editing_instructions: Dictionary of editing instructions
        n: Number of variations to generate (max 10)
        aspect_ratio: Desired aspect ratio
        image_size: Image resolution ("1K", "2K", or "4K")
        system_instruction: Optional system instruction
        
    Returns:
        List of tuples (edited_image, metadata)
    """
    try:
        client = _get_client()
        if client is None:
            raise ValueError("Missing XAI_API_KEY.")
        
        if n > 10:
            raise ValueError("Maximum 10 images per request")
        
        print(f"Generating {n} variations with Grok...")
        
        # Build the prompt
        prompt = get_initial_prompt(editing_instructions)
        if system_instruction:
            prompt = f"{system_instruction}\n\n{prompt}"
        
        # Convert image to data URI
        image_data_uri = prepare_image_for_grok(reference_image)
        
        # Convert image_size to Grok resolution format
        resolution = image_size.lower()
        if resolution not in ["1k", "2k"]:
            resolution = "1k"
        
        # Build parameters
        params = {
            "prompt": prompt,
            "model": DEFAULT_GENERATOR_MODEL,
            "image_url": image_data_uri,
            "resolution": resolution,
            "n": n
        }
        
        if aspect_ratio:
            params["aspect_ratio"] = aspect_ratio
        
        # Generate multiple images using sample_batch
        responses = _retry_api_call(
            lambda: client.image.sample_batch(**params),
            label="Grok batch generation"
        )
        
        # Process all responses
        results = []
        for i, response in enumerate(responses):
            edited_image = image_from_url_or_base64(response)
            metadata = {
                "model": DEFAULT_GENERATOR_MODEL,
                "image_size": image_size,
                "resolution": resolution,
                "aspect_ratio": aspect_ratio,
                "variation_index": i + 1,
                "moderation_passed": getattr(response, 'respect_moderation', True)
            }
            results.append((edited_image, metadata))
        
        print(f"✓ Generated {len(results)} variations")
        return results
        
    except Exception as e:
        st.error(f"Batch generation failed: {str(e)}")
        raise
