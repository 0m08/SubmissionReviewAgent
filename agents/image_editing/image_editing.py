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

# Load environment variables
load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================
DEFAULT_GENERATOR_MODEL = "gemini-3-pro-image-preview" 
DEFAULT_REVIEWER_MODEL = "gemini-3-flash-preview"
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

def prepare_image_for_gemini(image: Image.Image) -> bytes:
    buffered = BytesIO()
    save_img = image.copy()
    if save_img.mode != 'RGB':
        save_img = save_img.convert('RGB')
    save_img.save(buffered, format="PNG")
    return buffered.getvalue()

def image_from_base64(base64_string: str) -> Image.Image:
    if ',' in base64_string:
        base64_string = base64_string.split(',', 1)[1]
    image_bytes = base64.b64decode(base64_string)
    return Image.open(BytesIO(image_bytes))

# =============================================================================
# GENERATOR AGENT
# =============================================================================

@traceable(
    metadata={
        "agent_name": "image_editing",
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
    conversation_history: List[types.Content] = None,
    aspect_ratio: Optional[str] = None,
    image_size: str = "1K"
) -> Tuple[Image.Image, Dict, types.Content, types.Content]:
    try:
        client = _get_client()
        if client is None:
            raise ValueError("Missing GOOGLE_API_KEY.")
        
        print(f"Starting image editing generation...")
        
        user_content = None
        
        if not conversation_history:
            # Round 1: Full Prompt + Image
            print("Preparing initial request (Full Context)...")
            prompt = get_initial_prompt(editing_instructions)
            image_bytes = prepare_image_for_gemini(reference_image)
            
            user_content = types.Content(
                role="user",
                parts=[
                    types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                    types.Part.from_text(text=prompt)
                ]
            )
            full_history = [user_content]
        else:
            # Round 2+: Feedback Only
            print("Preparing feedback request (Diff Only)...")
            prompt = get_feedback_prompt(correction_feedback)
            
            user_content = types.Content(
                role="user",
                parts=[types.Part.from_text(text=prompt)]
            )
            full_history = conversation_history + [user_content]

        # Call Model
        # Build image_config dynamically
        image_config_params = {"image_size": image_size}
        if aspect_ratio:
            image_config_params["aspect_ratio"] = aspect_ratio
        
        response = client.models.generate_content(
            model=DEFAULT_GENERATOR_MODEL,
            contents=full_history,
            config=types.GenerateContentConfig(
                response_modalities=['TEXT', 'IMAGE'],
                tools=[{"google_search": {}}],
                image_config=types.ImageConfig(**image_config_params)
            )
        )
        
        if not response.candidates or not response.candidates[0].content:
            raise ValueError("Model returned no content.")

        # Extract Image & Raw Content
        edited_image = None
        model_content = response.candidates[0].content 
        
        if hasattr(model_content, 'parts'):
            for part in model_content.parts:
                if hasattr(part, 'inline_data') and part.inline_data:
                    try:
                        image_data = part.inline_data.data
                        edited_image = Image.open(BytesIO(image_data))
                        break
                    except Exception as e:
                        print(f"Error processing inline data: {e}")

        if edited_image is None and hasattr(response, 'text'):
            try:
                edited_image = image_from_base64(response.text)
            except:
                pass
                
        if edited_image is None:
            raise ValueError("No edited image found. The model may have refused the request.")
        
        # Metadata
        metadata = {
            "model": DEFAULT_GENERATOR_MODEL,
            "input_tokens": response.usage_metadata.prompt_token_count if hasattr(response, 'usage_metadata') else 0,
            "output_tokens": response.usage_metadata.candidates_token_count if hasattr(response, 'usage_metadata') else 0,
        }
        
        print("✓ Image editing generated")
        return edited_image, metadata, model_content, user_content
        
    except Exception as e:
        st.error(f"Image generation failed: {str(e)}")
        raise

# =============================================================================
# REVIEWER AGENT
# =============================================================================

@traceable(
    metadata={
        "agent_name": "image_editing",
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
    custom_criteria: Optional[str] = None
) -> Tuple[bool, Optional[str], Dict]:
    try:
        client = _get_client()
        if client is None:
            raise ValueError("Missing GOOGLE_API_KEY.")
        
        print("Reviewing edited image...")
        
        reference_bytes = prepare_image_for_gemini(reference_image)
        edited_bytes = prepare_image_for_gemini(edited_image)
        
        base_prompt = get_reviewer_prompt(editing_instructions, custom_criteria)
        if previous_feedback:
            base_prompt += f"\n\n<prior_defect_correction>\nThe previous iteration failed due to: {previous_feedback}.\nVerify that this specific issue has been resolved.\n</prior_defect_correction>"

        # Structured Output with Low Temperature for Consistency
        response = client.models.generate_content(
            model=DEFAULT_REVIEWER_MODEL,
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text=base_prompt),
                        types.Part.from_bytes(data=reference_bytes, mime_type="image/png"),
                        types.Part.from_bytes(data=edited_bytes, mime_type="image/png")
                    ]
                )
            ],
            config=types.GenerateContentConfig(
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
# MAIN LOOP
# =============================================================================

@traceable(
    metadata={
        "agent_name": "image_editing",
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
    image_size: str = "1K"
) -> Tuple[Image.Image, list, List[types.Content]]:
    print(f"Starting image editing with review loop... (Quick Mode: {quick_mode})")
    
    conversation_history: List[types.Content] = []
    correction_feedback = None
    edited_image = None
    ui_history_log = [] 
    
    # Quick Mode: Single generation without review
    if quick_mode:
        print("Quick Mode enabled - skipping review loop")
        with st.spinner("Generating edited image..."):
            edited_image, gen_metadata, model_content, user_content = generate_edited_image(
                reference_image=reference_image,
                editing_instructions=editing_instructions,
                correction_feedback=None,
                conversation_history=None,
                aspect_ratio=aspect_ratio,
                image_size=image_size
            )
        
        ui_history_log.append({
            "round": 1,
            "type": "generation",
            "edited_image": edited_image,
            "metadata": gen_metadata
        })
        
        # Build conversation history for potential manual feedback
        conversation_history.append(user_content)
        conversation_history.append(model_content)
        
        print("✓ Quick mode generation complete")
        return edited_image, ui_history_log, conversation_history
    
    # Normal Mode: Full review loop
    for round_num in range(MAX_CORRECTION_ROUNDS):
        print(f"\nRound {round_num + 1}/{MAX_CORRECTION_ROUNDS}")
        
        # 1. Generate
        with st.spinner(f"Generating edited image... (Round {round_num + 1})"):
            edited_image, gen_metadata, model_content, user_content = generate_edited_image(
                reference_image=reference_image,
                editing_instructions=editing_instructions,
                correction_feedback=correction_feedback,
                conversation_history=conversation_history,
                aspect_ratio=aspect_ratio,
                image_size=image_size
            )
        
        # 2. Update History
        conversation_history.append(user_content)
        conversation_history.append(model_content)
        
        # Debug Log
        # debug_print_history(conversation_history, round_num + 1)
        
        ui_history_log.append({
            "round": round_num + 1,
            "type": "generation",
            "edited_image": edited_image,
            "metadata": gen_metadata
        })
        
        # 3. Review
        with st.spinner("Reviewing edited image..."):
            is_approved, feedback, review_metadata = review_edited_image(
                reference_image=reference_image,
                edited_image=edited_image,
                editing_instructions=editing_instructions,
                previous_feedback=correction_feedback,
                custom_criteria=custom_criteria
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
    
    print(f"Max rounds reached")
    return edited_image, ui_history_log, conversation_history