import os
import base64
from io import BytesIO
from PIL import Image
from typing import Optional, Tuple, Dict, List, Any
from openai import OpenAI
from dotenv import load_dotenv
import json
import streamlit as st
from pydantic import BaseModel, Field
from langsmith import traceable
import requests  # Added for URL fallback support
import concurrent.futures
import threading
import time

# Load environment variables
load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================
DEFAULT_GENERATOR_MODEL = "gpt-image-1.5"
DEFAULT_REVIEWER_MODEL = "gpt-5-mini"
MAX_CORRECTION_ROUNDS = 5

# Timeout for each individual LLM call (seconds). If exceeded, the call is
# cancelled and a fresh call is started automatically.
LLM_CALL_TIMEOUT_SECONDS = 180  # 3 minutes

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

def _get_client() -> Optional[OpenAI]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("Missing OPENAI_API_KEY")
        return None
    try:
        return OpenAI(api_key=api_key)
    except Exception as e:
        print(f"Failed to initialize OpenAI client: {e}")
        return None

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
Return a structured assessment. Set 'approved' to true only if the image is technically sound and fully adheres to instructions.
</output_requirements>
"""
    return prompt

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def prepare_image_for_upload(image: Image.Image, format: str = "PNG") -> BytesIO:
    """Prepares image bytes for OpenAI API upload with correct filename attribute."""
    buffered = BytesIO()
    save_img = image.copy()
    if save_img.mode != 'RGB':
        save_img = save_img.convert('RGB')

    save_img.save(buffered, format=format)
    buffered.seek(0)
    # Required for OpenAI SDK to detect MIME type correctly
    buffered.name = "image.png"
    return buffered

def encode_image_base64(image: Image.Image) -> str:
    """Encodes PIL image to base64 string for Vision API."""
    buffered = prepare_image_for_upload(image)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def image_from_base64(base64_string: str) -> Image.Image:
    if ',' in base64_string:
        base64_string = base64_string.split(',', 1)[1]
    image_bytes = base64.b64decode(base64_string)
    return Image.open(BytesIO(image_bytes))

def download_image_from_url(url: str) -> Image.Image:
    """Downloads an image from a URL and returns a PIL Image."""
    response = requests.get(url)
    response.raise_for_status()
    return Image.open(BytesIO(response.content))

# =============================================================================
# TIMEOUT-AWARE CALL WRAPPER
# =============================================================================

def _run_with_timeout(fn, timeout_seconds: int, *args, **kwargs):
    """
    Runs `fn(*args, **kwargs)` in a thread-pool executor with a hard timeout.
    If the call does not complete within `timeout_seconds`, raises TimeoutError
    so the caller can dispose the future and start a fresh call.

    Returns the result of fn on success.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(fn, *args, **kwargs)
        try:
            result = future.result(timeout=timeout_seconds)
            return result
        except concurrent.futures.TimeoutError:
            # Cancel the future (best-effort; the underlying thread cannot be
            # forcibly stopped in Python, but the caller will start a new one).
            future.cancel()
            raise TimeoutError(
                f"LLM call exceeded the {timeout_seconds}s timeout. "
                "Disposing previous call and starting a fresh one."
            )

# =============================================================================
# GENERATOR AGENT (GPT-Image-1.5)
# =============================================================================

@traceable(
    metadata={
        "agent_name": "image_editing",
        "step_name": "Generator",
        "model": DEFAULT_GENERATOR_MODEL
    }
)
def generate_edited_image(
    reference_image: Image.Image,
    editing_instructions: Dict[str, str],
    correction_feedback: Optional[str] = None,
    conversation_history: List[Any] = None,
    aspect_ratio: Optional[str] = None,
    image_size: str = "1024x1024",
    system_instruction: Optional[str] = None
) -> Tuple[Image.Image, Dict, Any, Any]:
    try:
        client = _get_client()
        if client is None:
            raise ValueError("Missing OPENAI_API_KEY.")

        print(f"Starting image editing generation with {DEFAULT_GENERATOR_MODEL}...")

        # 1. Construct the Prompt
        full_prompt = get_initial_prompt(editing_instructions)

        if correction_feedback:
            feedback_prompt = get_feedback_prompt(correction_feedback)
            full_prompt = f"{full_prompt}\n\n{feedback_prompt}"
            print("Processing feedback round (Prompt updated with defects)...")
        else:
            print("Processing initial generation...")

        if system_instruction:
            full_prompt = f"{system_instruction}\n\n{full_prompt}"

        # 2. Prepare Image
        image_bytes_io = prepare_image_for_upload(reference_image)

        # 3. Call OpenAI Image Edits API
        try:
            response = client.images.edit(
                model=DEFAULT_GENERATOR_MODEL,
                image=image_bytes_io,
                prompt=full_prompt,
                size="1024x1024",
                extra_body={"input_fidelity": "high"}
            )
        except Exception as e:
            print(f"Edit API call failed: {e}")
            raise

        # 4. Extract Result
        if not response.data:
            raise ValueError("Model returned no data.")

        data_item = response.data[0]
        edited_image = None

        # Check for Base64 (Preferred)
        if hasattr(data_item, 'b64_json') and data_item.b64_json:
            print("Received Base64 Image.")
            edited_image = image_from_base64(data_item.b64_json)
        # Check for URL (Fallback)
        elif hasattr(data_item, 'url') and data_item.url:
            print("Received Image URL. Downloading...")
            edited_image = download_image_from_url(data_item.url)
        else:
            # Try dictionary access if object is not standard pydantic model
            try:
                if isinstance(data_item, dict):
                    if data_item.get('b64_json'):
                        edited_image = image_from_base64(data_item['b64_json'])
                    elif data_item.get('url'):
                        edited_image = download_image_from_url(data_item['url'])
            except:
                pass

        if edited_image is None:
            raise ValueError("Failed to extract image from API response (No b64_json or url found).")

        # Metadata
        revised = getattr(data_item, 'revised_prompt', None)
        metadata = {
            "model": DEFAULT_GENERATOR_MODEL,
            "revised_prompt": revised,
            "prompt_used_length": len(full_prompt)
        }

        # Mock content
        user_content = {"role": "user", "content": full_prompt}
        model_content = {"role": "assistant", "content": "Image generated via Edit API"}

        print("✓ Image editing generated")
        return edited_image, metadata, model_content, user_content

    except Exception as e:
        st.error(f"Image generation failed: {str(e)}")
        raise


# =============================================================================
# FALLBACK GENERATOR — Direct gpt-image-1.5 call (no review loop)
# Called when all MAX_CORRECTION_ROUNDS are exhausted without success.
# =============================================================================

def generate_edited_image_openai_fallback(
    reference_image: Image.Image,
    editing_instructions: Dict[str, str],
    correction_feedback: Optional[str] = None,
    system_instruction: Optional[str] = None
) -> Tuple[Optional[Image.Image], Dict]:
    """
    A clean, standalone fallback that calls gpt-image-1.5 directly.
    Used as a last resort after MAX_CORRECTION_ROUNDS have all failed.
    Applies the same timeout-with-retry logic as the main generator.
    """
    print("⚠ Invoking gpt-image-1.5 fallback generator (all review rounds exhausted)...")

    def _do_call():
        return generate_edited_image(
            reference_image=reference_image,
            editing_instructions=editing_instructions,
            correction_feedback=correction_feedback,
            conversation_history=None,
            system_instruction=system_instruction
        )

    MAX_FALLBACK_ATTEMPTS = 3
    for attempt in range(1, MAX_FALLBACK_ATTEMPTS + 1):
        try:
            print(f"  Fallback attempt {attempt}/{MAX_FALLBACK_ATTEMPTS}...")
            result = _run_with_timeout(_do_call, LLM_CALL_TIMEOUT_SECONDS)
            edited_image, metadata, _, _ = result
            if edited_image is not None:
                print("✓ Fallback generation succeeded.")
                return edited_image, metadata
        except TimeoutError as te:
            print(f"  Fallback attempt {attempt} timed out: {te}")
        except Exception as e:
            print(f"  Fallback attempt {attempt} failed with error: {e}")

    print("✗ All fallback attempts failed. Returning None.")
    return None, {"model": DEFAULT_GENERATOR_MODEL, "fallback": True, "error": "all_fallback_attempts_failed"}


# =============================================================================
# TIMEOUT WRAPPER FOR generate_edited_image
# Each individual LLM call is bounded to LLM_CALL_TIMEOUT_SECONDS.
# If it times out, a fresh call is made transparently.
# =============================================================================

def generate_edited_image_with_timeout(
    reference_image: Image.Image,
    editing_instructions: Dict[str, str],
    correction_feedback: Optional[str] = None,
    conversation_history: List[Any] = None,
    aspect_ratio: Optional[str] = None,
    image_size: str = "1024x1024",
    system_instruction: Optional[str] = None,
    max_per_call_attempts: int = 3
) -> Tuple[Image.Image, Dict, Any, Any]:
    """
    Wraps generate_edited_image with a per-call timeout.
    If a call doesn't respond within LLM_CALL_TIMEOUT_SECONDS, it is disposed
    and a new call is automatically started (up to max_per_call_attempts times).
    Raises on total failure so the outer retry loop can handle it.
    """
    for attempt in range(1, max_per_call_attempts + 1):
        try:
            print(f"  [Call attempt {attempt}/{max_per_call_attempts}] Starting LLM call "
                  f"(timeout={LLM_CALL_TIMEOUT_SECONDS}s)...")
            result = _run_with_timeout(
                generate_edited_image,
                LLM_CALL_TIMEOUT_SECONDS,
                reference_image,
                editing_instructions,
                correction_feedback,
                conversation_history,
                aspect_ratio,
                image_size,
                system_instruction
            )
            return result
        except TimeoutError as te:
            print(f"  [Call attempt {attempt}] ⏱ Timeout: {te}")
            if attempt == max_per_call_attempts:
                raise
            print(f"  [Call attempt {attempt}] Disposing timed-out call, starting fresh...")
        except Exception as e:
            # Non-timeout errors bubble up immediately to the outer retry loop
            print(f"  [Call attempt {attempt}] ✗ Error: {e}")
            raise

# =============================================================================
# REVIEWER AGENT (GPT-4o)
# =============================================================================

@traceable(
    metadata={
        "agent_name": "image_editing",
        "step_name": "Reviewer",
        "model": DEFAULT_REVIEWER_MODEL
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
    try:
        client = _get_client()
        if client is None:
            raise ValueError("Missing OPENAI_API_KEY.")

        print("Reviewing edited image with GPT-4o...")

        ref_b64 = encode_image_base64(reference_image)
        edit_b64 = encode_image_base64(edited_image)

        base_prompt = get_reviewer_prompt(editing_instructions, custom_criteria)
        if previous_feedback:
            base_prompt += f"\n\n<prior_defect_correction>\nThe previous iteration failed due to: {previous_feedback}.\nVerify that this specific issue has been resolved.\n</prior_defect_correction>"

        system_prompt = "You are a senior QA Visual Editor."
        if system_instruction:
            system_prompt += f" {system_instruction}"

        response = client.beta.chat.completions.parse(
            model=DEFAULT_REVIEWER_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": base_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{ref_b64}", "detail": "high"}
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{edit_b64}", "detail": "high"}
                        }
                    ]
                }
            ],
            response_format=ReviewResult
        )

        result = response.choices[0].message.parsed

        metadata = {
            "model": DEFAULT_REVIEWER_MODEL,
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
            "confidence_score": result.confidence_score
        }

        return result.approved, result.feedback, metadata

    except Exception as e:
        st.error(f"Review failed: {str(e)}")
        raise

# =============================================================================
# MAIN LOOP
# =============================================================================

@traceable(
    metadata={
        "agent_name": "image_editing",
        "step_name": "Main Loop",
        "function_name": "image_editing_with_review_loop"
    }
)
def image_editing_with_review_loop(
    reference_image: Image.Image,
    editing_instructions: Dict[str, str],
    quick_mode: bool = False,
    custom_criteria: Optional[str] = None,
    aspect_ratio: Optional[str] = None,
    image_size: str = "1024x1024",
    system_instruction: Optional[str] = None
) -> Tuple[Image.Image, list, List[Any]]:

    print(f"Starting image editing with review loop... (Quick Mode: {quick_mode})")

    conversation_history = []
    correction_feedback = None
    edited_image = None
    ui_history_log = []

    # Quick Mode — single generation, no review loop
    if quick_mode:
        print("Quick Mode enabled - skipping review loop")
        with st.spinner("Generating edited image..."):
            edited_image, gen_metadata, model_content, user_content = generate_edited_image_with_timeout(
                reference_image=reference_image,
                editing_instructions=editing_instructions,
                correction_feedback=None,
                conversation_history=None,
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

        conversation_history.append(user_content)
        conversation_history.append(model_content)

        print("✓ Quick mode generation complete")
        return edited_image, ui_history_log, conversation_history

    # -------------------------------------------------------------------------
    # Normal Mode — up to MAX_CORRECTION_ROUNDS rounds with review.
    #
    # Retry strategy:
    #   • Each call to generate_edited_image is wrapped in a 3-minute timeout.
    #     If a call doesn't respond in time it is disposed and a fresh call is
    #     made (up to 3 per-call attempts per round).
    #   • If generation fails for a round (error or None result), the round is
    #     retried (the loop continues to the next iteration with the same
    #     correction_feedback).
    #   • After all MAX_CORRECTION_ROUNDS are exhausted without a successful
    #     approved image, a final fallback call to gpt-image-1.5 is made via
    #     generate_edited_image_openai_fallback. Whatever it produces becomes
    #     the final returned image.
    # -------------------------------------------------------------------------

    rounds_with_successful_generation = 0

    for round_num in range(MAX_CORRECTION_ROUNDS):
        print(f"\nRound {round_num + 1}/{MAX_CORRECTION_ROUNDS}")

        # 1. Generate (with per-call timeout + retry)
        gen_succeeded = False
        try:
            with st.spinner(f"Generating edited image... (Round {round_num + 1})"):
                edited_image, gen_metadata, model_content, user_content = generate_edited_image_with_timeout(
                    reference_image=reference_image,
                    editing_instructions=editing_instructions,
                    correction_feedback=correction_feedback,
                    conversation_history=conversation_history,
                    aspect_ratio=aspect_ratio,
                    image_size=image_size,
                    system_instruction=system_instruction
                )

            if edited_image is None:
                print(f"  Round {round_num + 1}: generation returned None — retrying next round...")
                ui_history_log.append({
                    "round": round_num + 1,
                    "type": "generation_failed",
                    "reason": "returned_none",
                    "metadata": {}
                })
                continue  # try the next round

            gen_succeeded = True
            rounds_with_successful_generation += 1

        except Exception as gen_err:
            print(f"  Round {round_num + 1}: generation error — {gen_err}. Retrying next round...")
            ui_history_log.append({
                "round": round_num + 1,
                "type": "generation_failed",
                "reason": str(gen_err),
                "metadata": {}
            })
            continue  # try the next round

        # 2. Update History (only on successful generation)
        conversation_history.append(user_content)
        conversation_history.append(model_content)

        ui_history_log.append({
            "round": round_num + 1,
            "type": "generation",
            "edited_image": edited_image,
            "metadata": gen_metadata
        })

        # 3. Review
        try:
            with st.spinner("Reviewing edited image..."):
                is_approved, feedback, review_metadata = review_edited_image(
                    reference_image=reference_image,
                    edited_image=edited_image,
                    editing_instructions=editing_instructions,
                    previous_feedback=correction_feedback,
                    custom_criteria=custom_criteria,
                    system_instruction=system_instruction
                )
        except Exception as rev_err:
            print(f"  Round {round_num + 1}: review error — {rev_err}. Continuing to next round...")
            ui_history_log.append({
                "round": round_num + 1,
                "type": "review_failed",
                "reason": str(rev_err),
                "metadata": {}
            })
            continue

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

    # -------------------------------------------------------------------------
    # All MAX_CORRECTION_ROUNDS exhausted without approval.
    # Invoke the gpt-image-1.5 fallback as a final attempt.
    # -------------------------------------------------------------------------
    print(f"\n⚠ All {MAX_CORRECTION_ROUNDS} rounds exhausted. Invoking gpt-image-1.5 fallback...")

    with st.spinner("All review rounds exhausted — running final gpt-image-1.5 fallback..."):
        fallback_image, fallback_metadata = generate_edited_image_openai_fallback(
            reference_image=reference_image,
            editing_instructions=editing_instructions,
            correction_feedback=correction_feedback,
            system_instruction=system_instruction
        )

    if fallback_image is not None:
        ui_history_log.append({
            "round": "fallback",
            "type": "generation",
            "edited_image": fallback_image,
            "metadata": fallback_metadata
        })
        print("✓ Returning fallback gpt-image-1.5 image as final result.")
        return fallback_image, ui_history_log, conversation_history

    # Absolute last resort — return whatever the last successful generation was
    # (could be None if every round failed outright).
    print("✗ Fallback also failed. Returning last available image (may be None).")
    return edited_image, ui_history_log, conversation_history