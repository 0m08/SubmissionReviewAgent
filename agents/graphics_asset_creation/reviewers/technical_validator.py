"""
Stage D: Technical Instruction Validator
=========================================
Single cognitive objective: Given the editing instructions produced by the
Transformation Planner (Stage C), verify that EVERY instruction is:
  1. Technically correct    — does not contradict engineering/physics reality
  2. Factually grounded     — measurements, materials, colors, orientations
                              match real-world standards (uses Google Search)
  3. Visually feasible      — achievable given the image's current visual state
  4. Voiceover-compatible   — does not destroy educational accuracy required
                              by the voiceover / slide context
  5. Non-contradictory      — instructions don't conflict with each other

If any instruction fails a check → CORRECT it in-place and document the fix.
Output is the SAME TransformationInstructions schema, corrected and editor-ready.

Position in pipeline:
  Stage A (Similarity) → Stage C (Transformation Planner) → Stage D (THIS) → Editor

Two-call architecture:
  Call 1: Google Search + free-text reasoning  →  genuine web grounding (no schema conflict)
  Call 2: Structured JSON extraction only      →  clean schema output (no tools conflict)
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
    prepare_image_for_gemini,
    call_llm_with_retry,
)
from agents.graphics_asset_creation.reviewers.transformation_planner import (
    TransformationInstructions,
)


# ── Output schema ──────────────────────────────────────────────────────────────

class ValidatedInstructions(BaseModel):
    """
    Corrected transformation instructions ready for the image editor.
    All fields mirror TransformationInstructions; corrections are applied
    in-place. validation_summary explains every change made.
    """
    perspective_and_camera:   str = Field(default="", description="Validated/corrected camera instruction.")
    visual_style:             str = Field(default="", description="Validated/corrected style instruction.")
    subject_focus:            str = Field(default="", description="Validated/corrected subject instruction.")
    background_setting:       str = Field(default="", description="Validated/corrected background instruction.")
    lighting_and_environment: str = Field(default="", description="Validated/corrected lighting instruction.")
    color_grading:            str = Field(default="", description="Validated/corrected color instruction.")
    additional_comments:      str = Field(default="", description="Validated/corrected comments.")
    validation_summary:       str = Field(
        default="",
        description=(
            "For each field: what was checked, what was found, and what was corrected. "
            "If no correction was needed for a field, state 'PASS'. "
            "If a field was corrected, state the original instruction, the error found, "
            "and the corrected instruction. Must also state which Google Search queries "
            "were executed and what they confirmed."
        )
    )
    corrections_made: bool = Field(
        default=False,
        description="True if ANY instruction field was modified from the planner's original."
    )


def _format_instructions(instr: TransformationInstructions) -> str:
    """Format instructions as a readable block for the prompt."""
    fields = {
        "perspective_and_camera":   instr.perspective_and_camera,
        "visual_style":             instr.visual_style,
        "subject_focus":            instr.subject_focus,
        "background_setting":       instr.background_setting,
        "lighting_and_environment": instr.lighting_and_environment,
        "color_grading":            instr.color_grading,
        "additional_comments":      instr.additional_comments,
    }
    lines = []
    for field, val in fields.items():
        if val and val.strip():
            lines.append(f"  [{field}]: {val.strip()}")
    return "\n".join(lines) if lines else "  (no instructions provided)"


# ── System instruction (Call 1 only) ───────────────────────────────────────────

_SYSTEM_INSTRUCTION = """You are a Technical Instruction Validator — a senior HVAC field technician
with 15+ years of on-ground installation and commissioning experience, acting as a
Technical Asset QA Engineer.

You think with the hands-on intuition of someone who has stood in front of actual
HVAC/R equipment: you know what looks wrong before you can articulate why, and
you always back that gut feel with verification — either from your own technical
knowledge or from a Google Search.

YOUR ROLE IN THE PIPELINE:
The Transformation Planner (previous stage) generated editing instructions to make
an image copyright-compliant. Your job is to validate those instructions for
technical correctness and feasibility BEFORE they reach the image editor.
The next stage blindly executes whatever you output — so every error you miss
becomes a bad edit.

════════════════════════════════════════════════════
VALIDATION MANDATE
════════════════════════════════════════════════════
For EACH non-empty instruction field, apply BOTH of the following:

1. YOUR OWN TECHNICAL KNOWLEDGE
   Draw on your expertise as a senior HVAC/R technician. You already know:
   - Safety color codes and which components they are mandated on
   - Refrigerant line sizing conventions, insulation requirements, flow directions
   - Component naming, identification, and functional roles
   - Manifold gauge conventions, wiring color standards, ductwork orientation rules
   - What is physically achievable from a given viewpoint vs. what is impossible
   - Normal operating pressures, temperatures, and material specs
   Use this knowledge to catch obvious errors immediately, without searching.

2. GOOGLE SEARCH — MANDATORY FOR ANY FACTUAL UNCERTAINTY
   For anything you are not 100% certain about, or want to verify with an
   authoritative source, run a Google Search BEFORE making a ruling.
   Do NOT guess. If you are uncertain, search first, then decide.
   Examples of when to search:
   - Color code for a specific refrigerant cylinder (ASHRAE 15)
   - Safety valve color mandates for a specific jurisdiction
   - Clearance requirements for a specific equipment type
   - Material or finish standards for a specific component
   - Whether a particular camera angle is feasible for a described subject
   Report every query you ran and what it confirmed in your analysis.

════════════════════════════════════════════════════
SYSTEMS PHYSICS & CAUSAL REASONING
════════════════════════════════════════════════════
You must validate the image's CAUSAL LOGIC. HVAC systems are interconnected:
1. CONNECTIVITY LOGIC: If a tool or gauge is disconnected (hoses off, wires dangling), 
   its readout must be at atmospheric/rest state.
   - Example: A manifold gauge on a shelf or with open hoses MUST show 0 PSI.
   - Example: A multimeter not touching contacts MUST show 0V.
   - If an instruction asks to "set gauge to 250 PSI" on a disconnected unit, FAIL IT.
2. OPERATIONAL STATE: Readouts must match the system's active state.
   - If the system is "Off/Static," high and low side pressures should equalize.
   - If the system is "Running," there must be a visible difference between high (red) 
     and low (blue) side pressures.
3. PHYSICAL INTEGRITY: No "Ghost Connections." If a hose is shown, it must have 
   a start and end point (e.g., from manifold to service port). 

════════════════════════════════════════════════════
CHECK CRITERIA — APPLY TO EVERY INSTRUCTION
════════════════════════════════════════════════════
1. Technical/engineering correctness — does it violate any standard, code, or physical law?
2. Causal Consistency — do tool readouts match their connectivity state in the image?
3. Component naming accuracy — are the right names used for the right parts?
4. Visual feasibility — can this be done from the current image viewpoint?
5. Measurement validity — are quantitative values realistic and proportionate?
6. Voiceover compatibility — does it preserve what the voiceover is teaching?
7. Inter-instruction consistency — do all instructions work together coherently?

════════════════════════════════════════════════════
CORRECTION PROTOCOL
════════════════════════════════════════════════════
PASS:   The instruction is correct. State "PASS" and copy it exactly.
FAIL:   Identify the violated standard/principle (inc. causal logic) → Search for confirmation →
        Write a corrected instruction that achieves the same copyright-breaking
        goal without the error → Document: original, error found, corrected text, source.

NEVER add instructions the planner did not include.
NEVER change the intent of an instruction without clear technical justification.
ALL corrected instructions must be as specific and measurable as the originals.
"""


# ── Public entry point ─────────────────────────────────────────────────────────

@traceable(metadata={"agent_name": "technical_instruction_validator"})
def technical_instruction_validator_agent(
    image: Image.Image,
    image_description: str,
    transformation_instructions: TransformationInstructions,
    slide_title: str = "",
    voiceover: str = "",
    visual_anchors_text: str = "",
    previous_context: str = "",
) -> TransformationInstructions:
    """
    Stage D: Validate and correct Transformation Planner instructions.

    Two-call architecture to avoid the google_search / response_schema conflict:

      Call 1 — Google Search tool enabled, free-text output.
               The model searches the web freely, reasons through each instruction,
               and produces a structured analysis in plain text.

      Call 2 — No tools, structured JSON output only.
               Takes Call 1's grounded analysis as input and extracts the final
               corrected instructions into the ValidatedInstructions schema.

    Returns:
        TransformationInstructions with corrections applied. If all pass, returns original.
    """
    client = _get_client()
    if not client:
        raise ValueError("Client not initialized")

    filled_fields = [
        f for f in transformation_instructions.model_fields
        if getattr(transformation_instructions, f, "")
    ]

    if not filled_fields:
        print("[TECHNICAL VALIDATOR] No instructions to validate — returning as-is.")
        return transformation_instructions

    print(f"\n[TECHNICAL VALIDATOR] Validating {len(filled_fields)} instruction field(s): {filled_fields}")

    formatted_instructions = _format_instructions(transformation_instructions)
    img_bytes = prepare_image_for_gemini(image)

    # Shared context block used in both calls
    shared_context = f"""IMAGE DESCRIPTION:
{image_description}

Slide Title: {slide_title or "N/A"}
Voiceover:   {voiceover or "N/A"}

SUBJECT ANCHORS:
{visual_anchors_text if visual_anchors_text else "No anchor descriptions available."}

PREVIOUS EDITING CONTEXT:
{previous_context if previous_context else "None."}

INSTRUCTIONS TO VALIDATE (from Transformation Planner):
{formatted_instructions}"""

    # ─────────────────────────────────────────────────────────────────────────
    # CALL 1 — Google Search + free-text analysis
    # Search grounding works here because there is NO response_schema conflict.
    # The model runs real Google searches and reasons freely in plain text.
    # ─────────────────────────────────────────────────────────────────────────
    call1_prompt = f"""{shared_context}

════════════════════════════════════════════════════
TASK: TECHNICAL VALIDATION ANALYSIS
════════════════════════════════════════════════════
For EACH non-empty instruction field listed above:

Step 1 — Apply your own HVAC/R technical knowledge.
Step 2 — For any uncertainty, run a Google Search before ruling. Do not guess.
Step 3 — Write your analysis using this format per field:

FIELD: [field_name]
STATUS: PASS | FAIL
SEARCHES RAN: [list every query you ran + what each returned; "none" if not needed]
ISSUE: [if FAIL — exactly which standard, code, or physical principle is violated]
CORRECTED INSTRUCTION: [if FAIL — the corrected text, same intent, equally precise]
REASONING: [1-3 sentences of evidence]

Cover every populated field. Be exhaustive — the next stage blindly executes your output."""

    from agents.graphics_asset_creation.automated.llm_call_tracker import tracker as llm_tracker

    print("[TECHNICAL VALIDATOR] Call 1 — Google Search + free-text analysis...")
    with llm_tracker.call(REVIEWER_MODEL, "Technical Validator (Search + Analysis)") as usage1:
        response1 = call_llm_with_retry(
            client.models.generate_content,
            model=REVIEWER_MODEL,
            contents=[types.Content(role="user", parts=[
                types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
                types.Part.from_text(text=call1_prompt),
            ])],
            config=types.GenerateContentConfig(
                tools=[{"google_search": {}}],   # ← Search works — no schema conflict here
                system_instruction=_SYSTEM_INSTRUCTION,
                thinking_config=types.ThinkingConfig(
                    thinking_level="high",        # ← Deep reasoning for rigorous audit
                ),
            ),
        )
        usage1.set_response(response1)

    analysis_text = response1.text.strip()
    print(f"[TECHNICAL VALIDATOR] Call 1 complete ({len(analysis_text)} chars of analysis).")

    # ─────────────────────────────────────────────────────────────────────────
    # CALL 2 — Structured JSON extraction only (no tools — no conflict)
    # Takes Call 1's grounded, search-backed analysis and maps it cleanly
    # into the ValidatedInstructions schema. No reasoning needed — just extraction.
    # ─────────────────────────────────────────────────────────────────────────
    call2_prompt = f"""A senior HVAC field technician has performed a grounded technical validation
of transformation instructions, using Google Search to verify factual claims.
Their full analysis is below. Extract the final validated instructions into JSON.

ORIGINAL INSTRUCTIONS:
{formatted_instructions}

TECHNICIAN'S VALIDATION ANALYSIS:
{analysis_text}

EXTRACTION RULES:
- PASS fields:  copy the original instruction EXACTLY — character for character.
- FAIL fields:  use the CORRECTED INSTRUCTION text from the analysis above.
- Missing fields (not in original): set to empty string "".
- corrections_made: true if ANY field was corrected/changed; false if all passed.
- validation_summary: concise per-field log. For each: PASS or CORRECTED, plus the
  key search query that confirmed the ruling. Keep it factual and brief."""

    print("[TECHNICAL VALIDATOR] Call 2 — Structured JSON extraction...")
    with llm_tracker.call(REVIEWER_MODEL, "Technical Validator (JSON Extraction)") as usage2:
        response2 = call_llm_with_retry(
            client.models.generate_content,
            model=REVIEWER_MODEL,
            contents=[types.Content(role="user", parts=[
                types.Part.from_text(text=call2_prompt),
            ])],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ValidatedInstructions,
                # No tools here — structured JSON output works cleanly
                thinking_config=types.ThinkingConfig(
                    thinking_level="low",   # Extraction only — no deep reasoning needed
                ),
            ),
        )
        usage2.set_response(response2)

    # Parse structured output
    if hasattr(response2, "parsed") and response2.parsed:
        validated: ValidatedInstructions = response2.parsed
    else:
        from agents.graphics_asset_creation.reviewers.voiceover_reviewer import _extract_json
        payload = _extract_json(response2.text)
        validated = ValidatedInstructions(**payload)

    # Log outcome
    if validated.corrections_made:
        print(f"[TECHNICAL VALIDATOR] WARNING: CORRECTIONS MADE — instructions updated before editor.")
        print(f"                       Summary: {validated.validation_summary[:300]}...")
    else:
        print(f"[TECHNICAL VALIDATOR] OK: All instructions passed — no corrections needed.")

    # Convert ValidatedInstructions -> TransformationInstructions (drop metadata fields)
    corrected = TransformationInstructions(
        perspective_and_camera   = validated.perspective_and_camera,
        visual_style             = validated.visual_style,
        subject_focus            = validated.subject_focus,
        background_setting       = validated.background_setting,
        lighting_and_environment = validated.lighting_and_environment,
        color_grading            = validated.color_grading,
        additional_comments      = validated.additional_comments,
    )

    return corrected
