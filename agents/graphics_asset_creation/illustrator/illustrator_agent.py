from __future__ import annotations
import os
from PIL import Image
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from langsmith import traceable

# ── Re-use shared utilities from gac_utils ─────────────────────────────────────
from agents.graphics_asset_creation.gac_utils import (
    DEFAULT_REVIEWER_MODEL as REVIEWER_MODEL,
    _get_client,
    call_llm_with_retry,
    prepare_image_for_gemini,
    _extract_json,
)

# ── Model configuration ───────────────────────────────────────────────────────
ILLUSTRATOR_MODEL = REVIEWER_MODEL           # gemini-3-flash-preview (analysis)

# ── Load styling guide ────────────────────────────────────────────────────────
def _load_styling_guide() -> str:
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.normpath(
            os.path.join(current_dir, "..", "guide", "styling_guide.md")
        )
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return f.read().strip()
    except Exception as exc:
        print(f"[ComplianceAgent] Could not load styling guide: {exc}")
    return ""


_STYLING_GUIDE = _load_styling_guide()


# ─────────────────────────────────────────────────────────────────────────────
# Output Schemas
# ─────────────────────────────────────────────────────────────────────────────

class IllustrationEditingInstructions(BaseModel):
    """
    Editing instructions that transform a photographic / rendered image into
    an illustration-style visual that self-explains the voiceover concept.

    All fields map directly to the downstream image_editing pipeline.
    Leave a field empty if no change is needed for that axis.
    """

    visual_style: str = Field(
        default="",
        description=(
            "CRITICAL: The image MUST be converted into an illustration, schematic, or "
            "technical diagram. Specify the high-level artistic approach, level of "
            "abstraction, rendering finish, and structural representation required."
        ),
    )
    lighting_and_environment: str = Field(
        default="",
        description=(
            "Lighting, shadow, and environmental directives that reinforce the core concept, "
            "control viewer focus, or establish the physical context of the subject."
        ),
    )
    color_grading: str = Field(
        default="",
        description=(
            "Colour palette, contrast, and grading changes that visually encode the concept "
            "or direct the learner's attention to semantic information."
        ),
    )
    additional_comments: str = Field(
        default="",
        description=(
            "Any remaining illustration directives: motion blur, vibration lines, "
            "directional arrows ON EXISTING elements (not new labels), cutaway views, "
            "transparency overlays on existing components, etc. "
            "Do NOT introduce new objects or text labels that are not already present."
        ),
    )

class IllustratorAgentResult(BaseModel):
    """Full output from the Illustrator Agent."""

    image_description: str = Field(
        description="Factual description of the input image: components, angle, style, lighting."
    )
    voiceover_gap_analysis: str = Field(
        description=(
            "Analysis of what the current image communicates vs. what the voiceover needs "
            "it to communicate. Identifies the illustration gap."
        ),
    )
    primary_concept: str = Field(
        description="The single dominant concept from the voiceover that the image must convey."
    )
    illustration_strategy: str = Field(
        description="High-level illustration strategy chosen (e.g. 'motion highlight', 'exploded view', 'thermal map')."
    )
    recommended_instructions: IllustrationEditingInstructions = Field(
        description=(
            "The concrete editing instructions — the primary recommendation "
            "that should be passed to the image_editing pipeline."
        ),
    )
    confidence_score: int = Field(
        default=0,
        description="0–100. How confident the agent is that these instructions will make the image self-explanatory.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# System Instruction
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_INSTRUCTION = """\
- Role: Illustrator Intelligence Agent — Illustration Specialist
- Develop a cohesive illustration strategy that strictly adheres to the provided STYLING AND TECHNICAL STANDARDS.
- Your analysis and recommendations must be compliant with the design tone, color scheme, and visual guidelines defined in the standards.
- Provide the complete IllustrationEditingInstructions (focusing on visual style, lighting, color, and additional illustration styling) as 'recommended_instructions'.
- Confidence score reflects how well the final image will be self-explanatory AND compliant with the standards.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Prompt Builders
# ─────────────────────────────────────────────────────────────────────────────

def _build_analysis_prompt(
    voiceover: str,
    slide_title: str = "",
    slide_content: str = "",
    visual_instruction: str = "",
) -> str:

    return f"""\
You are the Illustration Intelligence Agent. Your task is a two-stage cognitive exercise.

══════════════════════════════════════════════════════════════════
STAGE 1 — COMPREHENSION
══════════════════════════════════════════════════════════════════

IMAGE (attached above): Describe it in full technical detail.
Cover: subject, components, camera angle, visual style, lighting, colour palette,
environment, any branding or labels, and overall composition.

SLIDE CONTEXT:
• Title: {slide_title or 'N/A'}
• Content: {slide_content or 'N/A'}
• Voiceover: {voiceover}
• Visual instruction: {visual_instruction or 'N/A'}

VOICEOVER GAP ANALYSIS:
After describing the image, identify:
1. What is the PRIMARY concept the voiceover needs this image to convey?
2. Does the current image already convey that concept clearly WITHOUT narration?
3. What is the illustration gap — what visual evidence is missing or weak?

══════════════════════════════════════════════════════════════════
STAGE 2 — ILLUSTRATION DESIGN
══════════════════════════════════════════════════════════════════

Based on your gap analysis, design 2–3 CREATIVE illustration concepts.

For each concept, answer:
a) What is the illustration strategy? (e.g. motion highlight, thermal map, cutaway,
   exploded view, spotlighting, flow-colour coding, ghost overlay)
b) Why will this make the voiceover concept immediately visible without words?
c) What are the concrete editing instructions? (Fill all relevant fields with
   measurable, specific directives.)

STYLING GUIDE COMPLIANCE CHECK:
For your chosen instructions, verify compliance against the STYLING AND TECHNICAL STANDARDS:
- Does it use the correct HVAC color coding (Blue for cooling, Red/Orange for heating, etc.)?
- Is it mobile-first visibility compliant (bold, large, high-contrast)?
- Does it maintain "Industrial Clean" aesthetic (professional, not gritty)?
- Are PPE requirements (gloves, safety glasses) maintained?

CONCEPT RANKING:
Rank concepts from most to least effective. Lead with the one that creates
the clearest self-explanatory image with the least visual disruption.

STRICT PROHIBITIONS:
- Do NOT suggest adding or removing physical components, hardware, or objects.
  You must restrict changes to the aesthetics and style of existing assets.
- Text callouts, annotations, and arrows are ALLOWED but must be used wisely
  to support the technical concept.
- Maintain mechanical and functional accuracy — illustration style must not mislead
  a technically trained learner.

Now generate your analysis and recommendations as structured JSON matching
the IllustratorAgentResult schema exactly, ensuring all instructions are strictly compliant with the STYLING AND TECHNICAL STANDARDS provided in your system instructions.
""".strip()


# ─────────────────────────────────────────────────────────────────────────────
# Core Agent Function
# ─────────────────────────────────────────────────────────────────────────────

@traceable(metadata={"agent_name": "illustrator_agent", "step_name": "Illustration Intelligence"})
def illustrator_agent(
    image: Image.Image,
    voiceover: str,
    slide_title: str = "",
    slide_content: str = "",
    visual_instruction: str = "",
) -> IllustratorAgentResult:
    """
    Analyse an image against a voiceover and generate illustration-style editing
    instructions that make the image self-explanatory.

    Args:
        image:             PIL Image — the current graphics asset.
        voiceover:         The narration text this image must illustrate.
        slide_title:       Optional slide title for additional context.
        slide_content:     Optional slide body text for additional context.
        visual_instruction: Optional existing visual instruction from planner.

    Returns:
        IllustratorAgentResult containing gap analysis, creative ideas, and
        the recommended IllustrationEditingInstructions.
    """
    print("\n[ILLUSTRATOR AGENT] Starting illustration intelligence analysis...")
    print(f"  Voiceover: {voiceover[:120]}{'...' if len(voiceover) > 120 else ''}")

    client = _get_client()
    if not client:
        raise ValueError(
            "GOOGLE_API_KEY is missing. Illustrator Agent requires an LLM call."
        )

    # Build system instruction with styling guide if available
    system_instruction = _SYSTEM_INSTRUCTION
    if _STYLING_GUIDE:
        system_instruction += f"\n\nSTYLING AND TECHNICAL STANDARDS:\n{_STYLING_GUIDE}"

    image_bytes = prepare_image_for_gemini(image)

    prompt = _build_analysis_prompt(
        voiceover=voiceover,
        slide_title=slide_title,
        slide_content=slide_content,
        visual_instruction=visual_instruction,
    )

    print("[ILLUSTRATOR AGENT] Calling LLM for illustration analysis...")

    try:
        try:
            from agents.graphics_asset_creation.automated.llm_call_tracker import tracker as llm_tracker
            tracker_ctx = llm_tracker.call(ILLUSTRATOR_MODEL, "Illustrator Agent — Illustration Intelligence")
        except Exception:
            tracker_ctx = None  # tracker optional

        def _call():
            return call_llm_with_retry(
                client.models.generate_content,
                model=ILLUSTRATOR_MODEL,
                contents=[
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                            types.Part.from_text(text=prompt),
                        ],
                    )
                ],
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    response_mime_type="application/json",
                    response_schema=IllustratorAgentResult,
                    thinking_config=types.ThinkingConfig(
                        thinking_level="high",
                    ),
                ),
            )

        if tracker_ctx is not None:
            with tracker_ctx as usage:
                response = _call()
                usage.set_response(response)
        else:
            response = _call()

        # ── Parse result ──────────────────────────────────────────────────────
        parsed = getattr(response, "parsed", None)
        if parsed is not None and isinstance(parsed, IllustratorAgentResult):
            result = parsed
        elif parsed is not None and isinstance(parsed, dict):
            result = IllustratorAgentResult(**parsed)
        else:
            raw_text = (getattr(response, "text", None) or "").strip()
            if raw_text:
                try:
                    payload = _extract_json(raw_text)
                    result = IllustratorAgentResult(**payload)
                except Exception as parse_err:
                    print(f"[ILLUSTRATOR AGENT] JSON parse failed: {parse_err}")
                    result = _fallback_result(voiceover)
            else:
                result = _fallback_result(voiceover)

        _log_result(result)
        return result

    except Exception as exc:
        print(f"[ILLUSTRATOR AGENT] LLM call failed: {exc}")
        raise RuntimeError(f"Illustrator Agent LLM call failed: {exc}") from exc


# ─────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────────────────────────────────────

def _fallback_result(voiceover: str) -> IllustratorAgentResult:
    """Return a minimal fallback when the LLM fails to produce structured output."""
    return IllustratorAgentResult(
        image_description="Unable to analyse image — LLM response could not be parsed.",
        voiceover_gap_analysis=f"Voiceover: {voiceover}",
        primary_concept="Unknown — analysis failed.",
        illustration_strategy="N/A",
        recommended_instructions=IllustrationEditingInstructions(),
        confidence_score=0,
    )


def _log_result(result: IllustratorAgentResult) -> None:
    print(f"[ILLUSTRATOR AGENT] Primary concept: {result.primary_concept}")
    print(f"[ILLUSTRATOR AGENT] Strategy: {result.illustration_strategy}")
    print(f"[ILLUSTRATOR AGENT] Confidence: {result.confidence_score}/100")
    filled_fields = [
        f
        for f in result.recommended_instructions.model_fields
        if getattr(result.recommended_instructions, f, "")
    ]
    print(f"[ILLUSTRATOR AGENT] Recommended instructions fields filled: {filled_fields}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Convenience helper — flatten recommended_instructions to dict for pipeline
# ─────────────────────────────────────────────────────────────────────────────

def get_editing_instructions_dict(result: IllustratorAgentResult) -> dict:
    """
    Convert the recommended IllustrationEditingInstructions to a plain dict
    suitable for passing into image_editing.generate_edited_image().

    Returns:
        Dict with keys matching EDITING_OPTIONS in image_editing.py.
    """
    instr = result.recommended_instructions
    return {
        "visual_style":            instr.visual_style,
        "lighting_and_environment": instr.lighting_and_environment,
        "color_grading":           instr.color_grading,
        "additional_comments":     instr.additional_comments,
    }


@traceable(metadata={"agent_name": "illustrator_agent", "step_name": "Generate Illustrated Image"})
def generate_illustrated_image(
    reference_image: Image.Image,
    result: IllustratorAgentResult,
    image_size: str = "1K",
    aspect_ratio: str = "16:9",
    quick_mode: bool = True,
):
    """
    Run the image editing step using IllustratorAgentResult output.

    This wrapper keeps Step 2 generation flow in the illustrator module,
    so UI and automation can call one cohesive API.

    Returns:
        Tuple[edited_image, ui_log, conv_history] from image_editing_with_review_loop.
    """
    from agents.graphics_asset_creation.image_editing.image_editing import (
        image_editing_with_review_loop,
    )

    editing_instructions = {
        k: v
        for k, v in get_editing_instructions_dict(result).items()
        if isinstance(v, str) and v.strip()
    }

    if not editing_instructions:
        raise ValueError(
            "Illustrator Agent returned empty recommended instructions; cannot run image editing."
        )

    return image_editing_with_review_loop(
        reference_image=reference_image,
        editing_instructions=editing_instructions,
        quick_mode=quick_mode,
        image_size=image_size,
        aspect_ratio=aspect_ratio,
    )


@traceable(metadata={"agent_name": "illustrator_agent", "step_name": "Full Illustrator Pipeline"})
def run_illustrator_pipeline(
    image: Image.Image,
    voiceover: str,
    slide_title: str = "",
    slide_content: str = "",
    visual_instruction: str = "",
    image_size: str = "1K",
    aspect_ratio: str = "16:9",
    quick_mode: bool = True,
):
    """
    Single endpoint for Illustrator workflow.

    Runs:
    1) illustrator_agent analysis
    2) generate_illustrated_image editing pass

    Returns:
        Tuple[result, edited_image, ui_log, conv_history]
    """
    result = illustrator_agent(
        image=image,
        voiceover=voiceover,
        slide_title=slide_title,
        slide_content=slide_content,
        visual_instruction=visual_instruction,
    )

    edited_image, ui_log, conv_history = generate_illustrated_image(
        reference_image=image,
        result=result,
        image_size=image_size,
        aspect_ratio=aspect_ratio,
        quick_mode=quick_mode,
    )

    return result, edited_image, ui_log, conv_history
