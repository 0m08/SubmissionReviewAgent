import os
import base64
from io import BytesIO
from PIL import Image
from typing import Optional, Tuple, Dict
from google import genai
from google.genai import types
from dotenv import load_dotenv
import json
import streamlit as st
from services.llm_service import log_token_usage

# Load environment variables
load_dotenv()
# Model configuration
DEFAULT_GENERATOR_MODEL = "gemini-3-pro-image-preview"
DEFAULT_REVIEWER_MODEL = "gemini-2.5-pro"
MAX_CORRECTION_ROUNDS = 5

def _get_client() -> Optional[genai.Client]:
    """Initialize Gemini client lazily to avoid import-time failures in main UI."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return None
    try:
        return genai.Client(api_key=api_key)
    except Exception as e:
        print(f"Failed to initialize Gemini client: {e}")
        return None


# =============================================================================
# PROMPTS
# =============================================================================

def get_generator_prompt(translation_direction: str, user_comment: Optional[str] = None, correction_feedback: Optional[str] = None) -> str:
    """
    Build the prompt for the generator agent.
    
    Args:
        translation_direction: Direction of translation
        user_comment: Optional user requirements
        correction_feedback: Optional feedback from reviewer
        
    Returns:
        Formatted prompt string
    """
    prompt = f"""
<task>
Translate all text in the image from {translation_direction}
</task>

<instructions>
- Replace original text completely (don't overlay)
- Keep original text color (black stays black, white stays white)
- Maintain exact style, font, position, and background
</instructions>"""
    
    if correction_feedback:
        prompt += f"""\n\n<feedback>
{correction_feedback}
</feedback>"""
    
    if user_comment:
        prompt += f"""\n\n<additional_requirements>
{user_comment}
</additional_requirements>"""
    
    return prompt


def get_reviewer_prompt(translation_direction: str, user_comment: Optional[str] = None) -> str:
    """
    Build the prompt for the reviewer agent.
    
    Args:
        translation_direction: Direction of translation
        user_comment: Optional user requirements
        
    Returns:
        Formatted prompt string
    """
    prompt = f"""<task>
Compare the two images and review the translation quality.
</task>

<context>
Translation direction: {translation_direction}
Image 1: Original
Image 2: Translated
</context>

<review_criteria>
For each text phrase:
- Quote original → translation
- Flag awkward/literal phrasings (even if understandable)
- Suggest better phrasing if needed

Check:
- Translation accuracy
- Visual consistency (font/style/layout)"""
    
    if user_comment:
        prompt += f"""\n- Meets requirement: "{user_comment}" """
    
    prompt += """
</review_criteria>

<example>
EN: "personnel shock protection" → ES: "protección contra descargas personales"
Issue: "descargas personales" sounds like person emits shock
Better: "protección personal contra descargas"
</example>

<instructions>
Set approved=false for ANY awkward phrasing. Give specific fix in "feedback".
</instructions>

<output_format>
Return JSON:
{
    "approved": false,
    "feedback": "Replace X with Y",
    "issues_found": ["specific issue"],
    "confidence_score": 75,
    "translation_accuracy": "phrase analysis",
    "visual_consistency": "assessment"
}
</output_format>"""
    return prompt


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def prepare_image_for_gemini(image: Image.Image) -> bytes:
    """
    Prepare image for Gemini API by converting to bytes.
    Preserves transparency if present.
    
    Args:
        image: PIL Image object
        
    Returns:
        Image as bytes
    """
    buffered = BytesIO()
    # Keep RGBA for transparency, convert other modes to RGB
    if image.mode in ('RGBA', 'LA', 'P'):
        # Preserve alpha channel
        if image.mode == 'P':
            image = image.convert('RGBA')
        image.save(buffered, format="PNG")
    else:
        # Convert to RGB for non-transparent images
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image.save(buffered, format="PNG")
    return buffered.getvalue()


def image_from_base64(base64_string: str) -> Image.Image:
    """
    Convert base64 string to PIL Image.
    
    Args:
        base64_string: Base64 encoded image string
        
    Returns:
        PIL Image object
    """
    # Remove data URI prefix if present
    if ',' in base64_string:
        base64_string = base64_string.split(',', 1)[1]
    
    image_bytes = base64.b64decode(base64_string)
    return Image.open(BytesIO(image_bytes))


# =============================================================================
# GENERATOR AGENT (NANO BANANA PRO)
# =============================================================================

def generate_translated_image(
    image: Image.Image,
    translation_direction: str,
    user_comment: Optional[str] = None,
    correction_feedback: Optional[str] = None
) -> Tuple[Image.Image, Dict]:
    """
    Generator Agent (Nano Banana Pro): Translates text in the image inline.
    
    Args:
        image: PIL Image to translate
        translation_direction: Translation direction ("Spanish to English" or "English to Spanish")
        user_comment: Optional user comment for translation context
        correction_feedback: Optional feedback from reviewer for corrections
        
    Returns:
        Tuple of (translated_image, metadata)
    """
    try:
        client = _get_client()
        if client is None:
            raise ValueError("Missing GOOGLE_API_KEY. Set it in environment or sidebar.")
        
        print(f"Starting translation: {translation_direction}")
        
        # Prepare the image
        image_bytes = prepare_image_for_gemini(image)
        
        # Build the prompt using the prompt function
        prompt = get_generator_prompt(translation_direction, user_comment, correction_feedback)
        
        # Make API call to Gemini with image generation
        response = client.models.generate_content(
            model=DEFAULT_GENERATOR_MODEL,
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_bytes(
                            data=image_bytes,
                            mime_type="image/png"
                        ),
                        types.Part.from_text(text=prompt)
                    ]
                )
            ],
            config=types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=8192,
            )
        )
        
        # Extract the generated image
        # The model should return an image in the response
        translated_image = None
        
        # Try to extract image from response
        if hasattr(response, 'candidates') and response.candidates:
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'inline_data') and part.inline_data:
                    # Extract image from inline data
                    image_data = part.inline_data.data
                    translated_image = Image.open(BytesIO(image_data))
                    break
                elif hasattr(part, 'file_data') and part.file_data:
                    # Handle file data if present
                    st.warning("File data returned, attempting to process...")
        
        # If no image found in structured response, try text as base64
        if translated_image is None and hasattr(response, 'text'):
            try:
                translated_image = image_from_base64(response.text)
            except:
                st.error("Could not extract image from response. Model may have returned text instead of image.")
                raise ValueError("No image found in model response")
        
        if translated_image is None:
            raise ValueError("No translated image could be extracted from the model response")
        
        # Get metadata
        input_tokens = response.usage_metadata.prompt_token_count if hasattr(response, 'usage_metadata') else 0
        output_tokens = response.usage_metadata.candidates_token_count if hasattr(response, 'usage_metadata') else 0
        metadata = {
            "model": DEFAULT_GENERATOR_MODEL,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
        
        # Log token usage
        try:
            log_token_usage(llm=DEFAULT_GENERATOR_MODEL, input_tokens=input_tokens, output_tokens=output_tokens)
        except Exception:
            pass
        
        print("✓ Translation generated")
        return translated_image, metadata
        
    except Exception as e:
        st.error(f"Translation generation failed: {str(e)}")
        raise


# =============================================================================
# REVIEWER AGENT (GEMINI 2.5 PRO)
# =============================================================================

def review_translated_image(
    original_image: Image.Image,
    translated_image: Image.Image,
    translation_direction: str,
    user_comment: Optional[str] = None
) -> Tuple[bool, Optional[str], Dict]:
    """
    Reviewer Agent (Gemini 2.5 Pro): Compares original and translated images for accuracy.
    
    Args:
        original_image: Original PIL Image
        translated_image: Translated PIL Image
        translation_direction: Translation direction
        user_comment: Optional user comment for context
        
    Returns:
        Tuple of (is_approved, feedback_if_not_approved, metadata)
    """
    try:
        client = _get_client()
        if client is None:
            raise ValueError("Missing GOOGLE_API_KEY. Set it in environment or sidebar.")
        
        print("Reviewing...")
        
        # Prepare both images
        original_bytes = prepare_image_for_gemini(original_image)
        translated_bytes = prepare_image_for_gemini(translated_image)
        
        # Build review prompt using the prompt function
        review_prompt = get_reviewer_prompt(translation_direction, user_comment)
        
        # Make API call to reviewer with both images
        response = client.models.generate_content(
            model=DEFAULT_REVIEWER_MODEL,
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text=review_prompt),
                        types.Part.from_bytes(
                            data=original_bytes,
                            mime_type="image/png"
                        ),
                        types.Part.from_bytes(
                            data=translated_bytes,
                            mime_type="image/png"
                        )
                    ]
                )
            ],
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=4096,
                response_mime_type="application/json"
            )
        )
        
        # Extract text from response properly
        response_text = None
        
        # Try response.text first
        if hasattr(response, 'text') and response.text:
            response_text = response.text
        # Try candidates
        elif hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            
            if hasattr(candidate, 'content') and candidate.content:
                if hasattr(candidate.content, 'parts') and candidate.content.parts:
                    # Combine all text parts
                    all_parts = []
                    for part in candidate.content.parts:
                        if hasattr(part, 'text') and part.text:
                            all_parts.append(part.text)
                    response_text = ''.join(all_parts) if all_parts else None
        
        if not response_text:
            raise ValueError("Empty or invalid response from reviewer model")
        
        # Parse JSON
        try:
            review_result = json.loads(response_text)
        except (json.JSONDecodeError, TypeError) as e:
            import re
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if json_match:
                try:
                    review_result = json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    pass
            
            if 'review_result' not in locals():
                json_match = re.search(r'\{[^\{\}]*(?:\{[^\{\}]*\}[^\{\}]*)*\}', response_text, re.DOTALL)
                if json_match:
                    try:
                        review_result = json.loads(json_match.group(0))
                    except json.JSONDecodeError:
                        pass
            
            if 'review_result' not in locals():
                raise ValueError(f"Could not parse JSON from response")
        
        # Validate required fields and provide defaults
        if "approved" not in review_result:
            raise ValueError("Invalid response: missing 'approved' field")
        
        # Get metadata
        input_tokens = response.usage_metadata.prompt_token_count if hasattr(response, 'usage_metadata') else 0
        output_tokens = response.usage_metadata.candidates_token_count if hasattr(response, 'usage_metadata') else 0
        metadata = {
            "model": DEFAULT_REVIEWER_MODEL,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "confidence_score": review_result.get("confidence_score", 0),
            "issues_found": review_result.get("issues_found", []),
            "translation_accuracy": review_result.get("translation_accuracy", ""),
            "visual_consistency": review_result.get("visual_consistency", "")
        }
        
        # Log token usage
        try:
            log_token_usage(llm=DEFAULT_REVIEWER_MODEL, input_tokens=input_tokens, output_tokens=output_tokens)
        except Exception:
            pass
        
        is_approved = review_result.get("approved", False)
        feedback = review_result.get("feedback")
        
        if is_approved:
            print(f"✓ Translation approved (Confidence: {metadata['confidence_score']}%)")
        else:
            print(f"Needs revision: {feedback[:100]}..." if feedback and len(feedback) > 100 else f"Needs revision: {feedback}")
        
        return is_approved, feedback, metadata
        
    except Exception as e:
        st.error(f"Review failed: {str(e)}")
        raise


# =============================================================================
# CORRECTION LOOP
# =============================================================================

def image_translation_with_review_loop(
    image: Image.Image,
    translation_direction: str,
    user_comment: Optional[str] = None
) -> Tuple[Image.Image, list]:
    """
    Main image translation pipeline with review and correction loop.
    
    Args:
        image: PIL Image to translate
        translation_direction: Translation direction
        user_comment: Optional user comment
        
    Returns:
        Tuple of (final_translated_image, history)
    """
    print(f"Starting translation: {translation_direction}")
    history = []
    correction_feedback = None
    translated_image = None
    
    for round_num in range(MAX_CORRECTION_ROUNDS):
        print(f"\nRound {round_num + 1}/{MAX_CORRECTION_ROUNDS}")
        
        # Generate translated image
        with st.spinner(f"Processing... (Round {round_num + 1}/{MAX_CORRECTION_ROUNDS})"):
            translated_image, gen_metadata = generate_translated_image(
                image=image,
                translation_direction=translation_direction,
                user_comment=user_comment,
                correction_feedback=correction_feedback
            )
        
        # Record generation in history
        history.append({
            "round": round_num + 1,
            "type": "generation",
            "translated_image": translated_image,
            "metadata": gen_metadata
        })
        
        # Review translation
        with st.spinner("Reviewing..."):
            is_approved, feedback, review_metadata = review_translated_image(
                original_image=image,
                translated_image=translated_image,
                translation_direction=translation_direction,
                user_comment=user_comment
            )
        
        # Record review in history
        history.append({
            "round": round_num + 1,
            "type": "review",
            "approved": is_approved,
            "feedback": feedback,
            "metadata": review_metadata
        })
        
        if is_approved:
            print(f"✓ Translation approved in round {round_num + 1}")
            return translated_image, history
        else:
            print(f"Round {round_num + 1}: Needs improvement")
            correction_feedback = feedback
    
    # Max rounds reached
    print(f"Max rounds ({MAX_CORRECTION_ROUNDS}) reached")
    return translated_image, history
