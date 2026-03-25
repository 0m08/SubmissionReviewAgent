"""Stage C: Transformation Planner
================================
Single cognitive objective: Given the diagnosis from Stages A & B,
produce MINIMAL, SURGICAL editing instructions. 

MINIMAL = Fill the FEWEST fields possible - only what directly addresses diagnosed issues.
SURGICAL = When a field IS filled, provide detailed, specific instructions.

No image analysis, no Google Search, no verdict logic — just actionable instructions.
"""

from typing import Optional, List
from PIL import Image
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from langsmith import traceable

from agents.graphics_asset_creation.reviewers.voiceover_reviewer import (
    REVIEWER_MODEL,
    _get_client,
    _extract_json,
    prepare_image_for_gemini,
)

# ── Re-use the existing instruction schema ───────────────────────────────────

class TransformationInstructions(BaseModel):
    """Comprehensive transformation instructions to create a new synthetic image."""
    perspective_and_camera: str = Field(default="", description="Camera angle, position, and framing changes with precise measurements.")
    visual_style: str = Field(default="", description="Style transformation (photo→illustration, finish quality, detail level).")
    subject_focus: str = Field(default="", description="Components to REMOVE or PRESERVE from the existing image only. STRICT RULE: Do NOT introduce any new components, objects, labels, or elements that are not already present in the current image. Only instruct on what exists.")
    background_setting: str = Field(default="", description="Environment, setting, props, and spatial context — modify only what already exists in the image.")
    lighting_and_environment: str = Field(default="", description="Lighting setup with technical parameters.")
    color_grading: str = Field(default="", description="Color palette, saturation, tone adjustments.")
    additional_comments: str = Field(default="", description="Legal constraints (branding removal), safety notes. Do NOT suggest adding new elements.")


# ── System instruction (single focus) ────────────────────────────────────────

_SYSTEM_INSTRUCTION = """Transformation Planner: Generate minimal, direct editing instructions driven by visual anchor analysis.

CORE PRINCIPLE:
Visual anchors are the specific elements that make the current image recognizable as a derivative.
Your instructions MUST directly target UNCHANGED and PARTIALLY_CHANGED anchors to break derivative similarity.
CHANGED anchors are already differentiated — preserve them.

RULES:
1. READ every visual anchor description carefully — they explain WHY each element is recognizable
2. For each UNCHANGED anchor: generate a specific transformation that breaks its recognizability
3. For CHANGED or PARTIALLY_CHANGED anchors: do NOT modify — they already reduce similarity
4. Fill FEWEST fields possible — but each field must address at least one anchor
5. Write in direct imperative form — NO explanatory phrases
6. NO fluff: no "in order to", "this will", "to ensure", "for better"
7. State exact specifications only: angles, measurements, colors, component names
8. Each filled field must include at least one measurable quantitative parameter
9. AVOID repeating previous attempts — if history shows failed approaches, try alternatives

FIELD SELECTION:
- Similarity issue → perspective_and_camera only
- Branding issue → subject_focus or additional_comments only  
- Style issue → visual_style only
- Multiple distinct issues → still minimize field count

CAMERA ANGLE GUIDELINES:
- READ the image description first — identify the exact current camera position, angle, and viewpoint
- EVERY perspective_and_camera instruction MUST follow this 3-part structure:
  1. CURRENT STATE: "Currently [describe exact current angle/position from image description]."
  2. ACTION: "[Rotate/Elevate/Lower/Pan] [exact degrees or axis or distance]."
  3. TARGET STATE: "Result: [describe the resulting view]."
- Do NOT write camera instructions without anchoring them to the current described state
- Make realistic incremental changes — do not jump from front-facing to rear view in one step

Required format examples:
  ✓ "Currently front-facing at eye level. Rotate 40° clockwise around vertical axis. Result: front-right 3/4 view."
  ✓ "Currently elevated top-down view. Lower camera 25° toward horizontal. Result: elevated 3/4 view showing top and front face."
  ✓ "Currently right-side profile. Pan 20° left around vertical axis. Result: right 3/4 view with front face partially visible."
  ✗ "Rotate camera 35° clockwise."  ← INVALID — no current state anchor
  ✗ "Change to isometric view."  ← INVALID — no starting point specified

WRITING STYLE - Direct imperative, no fluff:
✓ "Rotate camera 35° clockwise from front-facing. Elevate 20° for right-side elevated view."
✓ "Remove 'BrandX' logo on left housing panel."
✓ "Convert to technical illustration. Matte industrial finish."
✗ "In order to differentiate from reference, rotate camera 35° clockwise..."
✗ "To ensure compliance with copyright requirements, remove the 'BrandX' logo..."

PROTECTED ANCHOR RULE (CRITICAL):
Some anchors are classified as breakable=false — these describe FUNCTIONAL EDUCATIONAL content.
Do NOT generate instructions that would alter or destroy them.
Examples of protected content:
- Specific pipe routing or wiring paths that are the subject of instruction
- Component configurations that demonstrate the technical concept
- Machinery layouts that define what the image teaches
- Spatial arrangements required to show a process or procedure
If an anchor from ANCHOR DESCRIPTIONS has breakable=false or describes functional layout/configuration → PRESERVE IT.
Only target breakable stylistic anchors (camera, lighting, material, branding, background, subject identity).

PROHIBITIONS:
- No justifications, rationales, or meta-commentary
- **STRICT: Do NOT add, introduce, or suggest any new components, objects, labels,
  annotations, text overlays, or visual elements that are NOT already present in
  the current image. Transformations modify what EXISTS — they do not extend the
  scene with new content.**
- Only fill fields that address diagnosed issues
- No extreme camera angles without prior moderate attempts
- No vague wording like "slightly", "a bit", or "more dynamic" without numbers

OUTPUT: TransformationInstructions JSON with minimal filled fields.
"""


# ── Public entry point ───────────────────────────────────────────────────────

@traceable(metadata={"agent_name": "transformation_planner"})
def transformation_planner_agent(
    image: Image.Image,
    image_description: str,
    similarity_analysis: str = "",
    technical_issues: str = "",
    visual_anchors: Optional[List[List[dict]]] = None,
    anchor_descriptions: str = "",
    anchor_breaking_suggestions: str = "",
    slide_title: str = "",
    voiceover: str = "",
    previous_instructions: Optional[List[TransformationInstructions]] = None,
    previous_similarity_scores: Optional[List[float]] = None,
    previous_context: str = "",
) -> TransformationInstructions:
    """
    Stage C — Generate editing instructions from prior diagnoses WITH VISUAL CONTEXT.

    Args:
        image: The actual image to analyze (NEW - for visual-informed decisions)
        image_description: Factual description of the current image (from Stage A).
        similarity_analysis: Similarity diagnostic text (from Stage A).
        technical_issues: Technical/styling issues text (from Stage B).
        visual_anchors: Per-reference detailed anchor lists (name/category/description/status/importance).
        anchor_descriptions: Formatted text of all anchor descriptions for context.
        slide_title: Educational context.
        voiceover: Educational context.
        previous_instructions: History of instructions from previous review cycles.
                              Used to avoid repeating failed attempts and make informed decisions.
        previous_similarity_scores: List of similarity scores from previous attempts (NEW).
                                   Used to detect convergence and adjust strategy.
        previous_context: Summary of previous edits from other agents.

    Returns:
        TransformationInstructions with only the fields that need changes filled.
    """
    print("\n[TRANSFORMATION PLANNER] Starting...")
    if previous_instructions:
        print(f"  [!] Aware of {len(previous_instructions)} previous attempt(s)")
    if previous_similarity_scores:
        print(f"  [!] Similarity trend: {previous_similarity_scores}")
    
    client = _get_client()
    if not client:
        raise ValueError("Client not initialized")

    # Format previous instructions history
    history_section = ""
    if previous_instructions:
        history_entries = []
        for idx, prev in enumerate(previous_instructions, 1):
            filled_fields = {f: getattr(prev, f) for f in prev.model_fields if getattr(prev, f, "")}
            if filled_fields:
                history_entries.append(f"Attempt {idx}:")
                for field, value in filled_fields.items():
                    history_entries.append(f"  • {field}: {value}")
        if history_entries:
            history_section = "\n".join(history_entries)
    
    # Build convergence feedback
    convergence_section = ""
    if previous_similarity_scores and len(previous_similarity_scores) > 1:
        latest = previous_similarity_scores[-1]
        previous = previous_similarity_scores[-2]
        trend = "IMPROVING" if latest < previous else "WORSENING"
        delta = abs(latest - previous)
        
        convergence_section = f"""
CONVERGENCE ANALYSIS:
Previous similarity scores: {[f"{s:.1f}%" for s in previous_similarity_scores]}
Latest: {latest:.1f}% (Previous: {previous:.1f}%)
Trend: {trend} (Delta: {delta:.1f}%)

STRATEGY GUIDANCE:
"""
        if trend == "IMPROVING":
            convergence_section += "- Continue current transformation approach\\n- Make incremental adjustments in same direction\\n- Don't change strategy drastically"
        else:
            convergence_section += "- CHANGE STRATEGY - current approach is not working\\n- Try a different transformation type\\n- If camera angle failed, try style/lighting/composition instead"
    elif previous_similarity_scores:
        convergence_section = f"\\nFirst transformation attempt. Previous similarity: {previous_similarity_scores[0]:.1f}%\\n"

    anchor_detail_section = ""
    if visual_anchors:
        lines = ["VISUAL ANCHOR DIAGNOSTICS:"]
        for idx, ref_anchors in enumerate(visual_anchors, 1):
            lines.append(f"- Reference {idx}:")
            if not ref_anchors:
                lines.append("  • No anchors detected")
                continue
            unchanged = [a for a in ref_anchors if isinstance(a, dict) and a.get("status") in ("unchanged", "identical")]
            partial   = [a for a in ref_anchors if isinstance(a, dict) and a.get("status") == "partially_changed"]
            changed   = [a for a in ref_anchors if isinstance(a, dict) and a.get("status") in ("changed", "fully_changed", "removed")]
            if unchanged:
                lines.append("  UNCHANGED (must transform):")
                for a in unchanged[:6]:
                    desc = a.get('description') or a.get('name', '?')
                    lines.append(f"    • [{a.get('category','?')}] {desc} (importance={a.get('importance','?')})")
            if partial:
                lines.append("  PARTIALLY CHANGED (consider further transformation):")
                for a in partial[:6]:
                    desc = a.get('description') or a.get('name', '?')
                    lines.append(f"    • [{a.get('category','?')}] {desc} (importance={a.get('importance','?')})")
            if changed:
                lines.append("  CHANGED (preserve — already differentiated):")
                for a in changed[:4]:
                    desc = a.get('description') or a.get('name', '?')
                    lines.append(f"    • [{a.get('category','?')}] {desc}")
            if not unchanged and not partial and not changed:
                lines.append("  • No structured anchors found")
        anchor_detail_section = "\n".join(lines)

    prompt = f"""Generate minimal transformation instructions. Write in direct imperative form - NO fluff.

═══════════════════════════════════════════════════════════════
📸 CURRENT IMAGE (Visual Context)
╔═══════════════════════════════════════════════════════════════
The image is attached above. Analyze it visually to understand:
- Current camera angle and perspective
- What elements are actually visible and prominent
- Current visual style (photo/render/illustration)
- Lighting setup and environment
- Spatial relationships and composition

Use this VISUAL ANALYSIS (not just the text description) to generate realistic, achievable instructions.

╔═══════════════════════════════════════════════════════════════
IMAGE DESCRIPTION (Text Summary):
╔═══════════════════════════════════════════════════════════════
{image_description}
{convergence_section}
╔═══════════════════════════════════════════════════════════════

═══════════════════════════════════════════════════════════════
SIMILARITY DIAGNOSIS:
═══════════════════════════════════════════════════════════════
{similarity_analysis if similarity_analysis else "No similarity issues."}

═══════════════════════════════════════════════════════════════
VISUAL ANCHOR DIAGNOSTICS:
═══════════════════════════════════════════════════════════════
{anchor_detail_section if anchor_detail_section else "No visual anchor diagnostics available."}

═══════════════════════════════════════════════════════════════
ANCHOR DESCRIPTIONS (detailed context — read carefully):
═══════════════════════════════════════════════════════════════
{anchor_descriptions if anchor_descriptions else "No anchor descriptions available."}

═══════════════════════════════════════════════════════════════
ANCHOR-BREAKING SUGGESTIONS (from dedicated analysis agent):
═══════════════════════════════════════════════════════════════
{anchor_breaking_suggestions if anchor_breaking_suggestions else "No suggestions available."}

These suggestions were generated by a specialist agent that deeply analyzed the
image and slide context. Use them as strong guidance — adopt or
adapt each suggestion into your instructions. Prioritize high-importance anchors.

═══════════════════════════════════════════════════════════════
TECHNICAL/STYLING ISSUES:
═══════════════════════════════════════════════════════════════
{technical_issues if technical_issues else "No technical issues."}

═══════════════════════════════════════════════════════════════
PREVIOUS EDITING ATTEMPTS:
═══════════════════════════════════════════════════════════════
{history_section if history_section else "This is the first review cycle - no previous attempts."}


═══════════════════════════════════════════════════════════════
EXTERNAL EDITING CONTEXT (FROM OTHER AGENTS):
═══════════════════════════════════════════════════════════════
{previous_context if previous_context else "No external context."}
IF the Accuracy Reviewer has explicitly restored a component, DO NOT remove it unless it is a blatant logo (trademarks). Functional components (windows, standard text, gauges) MUST be preserved if requested by Accuracy Reviewer.


═══════════════════════════════════════════════════════════════
CONTEXT:
═══════════════════════════════════════════════════════════════
Slide: {slide_title or "N/A"}
Voiceover: {voiceover or "N/A"}

═══════════════════════════════════════════════════════════════
TASK:
═══════════════════════════════════════════════════════════════
1. READ the VISUAL ANCHOR DIAGNOSTICS and ANCHOR DESCRIPTIONS above thoroughly
2. For each UNCHANGED anchor:
   Follow AGENT MODE: RECURSIVE ITERATION (Internal Logic):
   **Stage 1: The Integration Architect**
   - Review the 'Anchor-Breaking Suggestions' and 'Anchor Descriptions'.
   - Synthesize a transformation plan that maps these suggestions onto the current visual state.
   
   **Stage 2: The Technical Engineer (Accuracy Audit)**
   - Review Stage 1 against the 'External Editing Context' and 'Voiceover'.
   - **Audit Checklist**: 
     - Will this change violate a 'breakable=false' constraint? 
     - Does the new change maintain the technical and the functional accuracy of the asset?
     - Is the mechanical logic of the machine preserved?
   - **Veto/Refine**: Adjust any suggestion that compromises the industrial training utility.
   
   **Stage 3: The Quantitative Validator**
   - Convert the reconciled plan into final, measurable specifications.
   - Ensure zero "fluff" and 100% imperative technical language.

3. Do NOT touch CHANGED anchors — they already reduce similarity
4. VISUALLY ANALYZE the attached image to confirm anchor positions and current state
5. If camera angle change needed:
   a. OBSERVE the CURRENT angle/perspective in the actual image
   b. Plan REALISTIC incremental change (15-45° horizontal, 10-35° vertical)
   c. Describe transformation from observed current state → target angle
6. Make instructions QUANTITATIVE:
   - specific magnitudes (degrees, percentages, offsets, scale changes)
   - explicit composition shifts (crop, reposition, framing ratio)
   - at least one measurable change per filled field
7. Fill MINIMUM fields addressing the anchors
8. Write direct specs — NO explanatory text

Generate instructions that directly target the unchanged anchors (minimal fields, direct language):"""

    # Prepare image for visual context
    img_bytes = prepare_image_for_gemini(image)

    from agents.graphics_asset_creation.reviewers.voiceover_reviewer import call_llm_with_retry
    from agents.graphics_asset_creation.automated.llm_call_tracker import tracker as llm_tracker
    with llm_tracker.call(REVIEWER_MODEL, "Transformation Planner") as usage:
        response = call_llm_with_retry(
            client.models.generate_content,
            model=REVIEWER_MODEL,
            contents=[types.Content(role="user", parts=[
                types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
                types.Part.from_text(text=prompt)
            ])],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=TransformationInstructions,
                system_instruction=_SYSTEM_INSTRUCTION,
                thinking_config=types.ThinkingConfig(
                    thinking_level="high",
                )
            ),
        )
        usage.set_response(response)

    if hasattr(response, "parsed") and response.parsed:
        result = response.parsed
    else:
        payload = _extract_json(response.text)
        result = TransformationInstructions(**payload)

    filled = [f for f in result.model_fields if getattr(result, f, "")]
    print(f"[TRANSFORMATION PLANNER] Filled {len(filled)} field(s): {filled}")
    if convergence_section:
        print(f"                         Convergence-informed strategy applied")
    print(f"                         (Fewer is better - only what's necessary)\n")
    return result
