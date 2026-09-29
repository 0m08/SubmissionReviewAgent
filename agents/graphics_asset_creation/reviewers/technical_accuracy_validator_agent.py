from __future__ import annotations

import json
import os
import re
from io import BytesIO
from typing import List, Optional, Tuple

from PIL import Image
from pydantic import BaseModel, Field
from google import genai
from google.genai import types


ANALYSIS_MODEL = "gemini-3-flash-preview"


class RegionVerdict(BaseModel):
    """One region-level classification from reference vs final comparison."""

    region_name: str = Field(default="", description="Human-readable region/component name.")
    category: str = Field(
        default="creative_liberty",
        description="Either 'creative_liberty' or 'technical_inaccuracy'.",
    )
    reason: str = Field(default="", description="Short explanation for the classification.")
    correction_instruction: str = Field(
        default="",
        description="Instruction only when category is technical_inaccuracy. Empty otherwise.",
    )


class ModuleEditingInstructions(BaseModel):
    """Separate schema for image editing payload."""

    subject_focus: str = ""
    visual_style: str = ""
    perspective_and_camera: str = ""
    lighting_and_environment: str = ""
    background_setting: str = ""
    color_grading: str = ""
    additional_comments: str = ""


class TechnicalValidationAnalysis(BaseModel):
    """Structured output for Agent 2 call-2 mapping."""

    reference_summary: str = Field(default="", description="What matters technically in input/reference image.")
    final_summary: str = Field(default="", description="What changed in the final image.")
    region_verdicts: List[RegionVerdict] = Field(default_factory=list)
    compromised_details: List[str] = Field(
        default_factory=list,
        description="Technical details from reference that are compromised in final output.",
    )
    ordered_corrections: List[str] = Field(
        default_factory=list,
        description="Ordered corrections to restore compromised technical details.",
    )
    verdict: str = Field(
        default="approved",
        description="Either 'approved' or 'correction_required'.",
    )
    justification: str = Field(default="", description="Short final justification.")
    free_text_analysis: str = Field(default="")
    module_editing_instructions: ModuleEditingInstructions = Field(
        default_factory=ModuleEditingInstructions,
        description="Exact payload sent to image_editing.generate_edited_image.",
    )


class TechnicalValidationResult(BaseModel):
    """Public return object for UI and pipeline usage."""

    analysis: TechnicalValidationAnalysis
    output_image: object
    was_modified: bool
    modification_notes: str = ""

    model_config = {"arbitrary_types_allowed": True}


def _build_call1_system_instruction() -> str:
    """Concise system instruction for Agent 2 technical-validation call flow."""
    return """
Role:
You are a senior HVAC technician and trainer with 15+ years of field experience.
You have expert knowledge of HVAC components, service workflows, and technician training visuals.

Mission:
- Compare reference vs final image and detect only meaningful technical compromises.
- Preserve educational-value and industry-relevant technical details from the reference image.

Rules:
1) Prioritize technical correctness over cosmetic preference.
2) For each region, classify into exactly one category: creative_liberty OR technical_inaccuracy.
3) Never assign both categories to the same region.
4) Creative liberties are allowed when they do not break technical meaning.
5) Only request corrections for confirmed technical inaccuracies.
6) Do not request style-only or aesthetic-only edits.
7) Keep conclusions concise, specific, and executable.
8) If uncertain on a technical standard, you may use Google Search.
""".strip()


def _get_client() -> genai.Client:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        try:
            import streamlit as st

            api_key = st.session_state.get("google_api_key")
        except Exception:
            api_key = None

    if not api_key:
        raise ValueError("GOOGLE_API_KEY is missing. Technical Accuracy Validator requires an LLM call.")

    try:
        return genai.Client(api_key=api_key)
    except Exception as exc:
        raise ValueError(f"Failed to initialize Gemini client: {exc}") from exc


def _to_bytes(image: Image.Image) -> bytes:
    buf = BytesIO()
    img = image.convert("RGB") if image.mode != "RGB" else image.copy()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def _text_from_response(response) -> str:
    text = (getattr(response, "text", None) or "").strip()
    if text:
        return text

    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return ""

    parts = getattr(candidates[0].content, "parts", [])
    chunks: List[str] = []
    for part in parts:
        part_text = getattr(part, "text", None)
        if part_text and str(part_text).strip():
            chunks.append(str(part_text).strip())

    return "\n".join(chunks).strip()


def _extract_json_payload(raw_text: str) -> dict:
    """Extract the first JSON object from LLM text and parse it."""
    text = (raw_text or "").strip()
    if not text:
        return {}

    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    candidate = fence_match.group(1).strip() if fence_match else text

    if candidate.startswith("{") and candidate.endswith("}"):
        return json.loads(candidate)

    obj_match = re.search(r"\{.*\}", candidate, re.DOTALL)
    if obj_match:
        return json.loads(obj_match.group(0))

    return json.loads(candidate)


def _ensure_text(value) -> str:
    """Best-effort coercion for LLM fields that may arrive as list/number/null."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = [str(x).strip() for x in value if str(x).strip()]
        return "\n".join(parts)
    return str(value)


def _ensure_text_list(value) -> List[str]:
    """Best-effort coercion for fields that should be a list of strings."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    text = str(value).strip()
    return [text] if text else []


def _normalize_analysis_payload(payload: dict) -> dict:
    """Normalize LLM JSON shape to the expected TechnicalValidationAnalysis schema."""
    if not isinstance(payload, dict):
        return {}

    normalized = dict(payload)

    normalized["reference_summary"] = _ensure_text(normalized.get("reference_summary", ""))
    normalized["final_summary"] = _ensure_text(normalized.get("final_summary", ""))
    normalized["justification"] = _ensure_text(normalized.get("justification", ""))
    normalized["free_text_analysis"] = _ensure_text(normalized.get("free_text_analysis", ""))

    normalized["compromised_details"] = _ensure_text_list(normalized.get("compromised_details"))
    normalized["ordered_corrections"] = _ensure_text_list(normalized.get("ordered_corrections"))

    module_raw = normalized.get("module_editing_instructions")
    module_obj = module_raw if isinstance(module_raw, dict) else {}
    normalized["module_editing_instructions"] = {
        "subject_focus": _ensure_text(module_obj.get("subject_focus", "")),
        "visual_style": _ensure_text(module_obj.get("visual_style", "")),
        "perspective_and_camera": _ensure_text(module_obj.get("perspective_and_camera", "")),
        "lighting_and_environment": _ensure_text(module_obj.get("lighting_and_environment", "")),
        "background_setting": _ensure_text(module_obj.get("background_setting", "")),
        "color_grading": _ensure_text(module_obj.get("color_grading", "")),
        "additional_comments": _ensure_text(module_obj.get("additional_comments", "")),
    }

    region_raw = normalized.get("region_verdicts")
    safe_regions: List[dict] = []
    if isinstance(region_raw, list):
        for item in region_raw:
            if isinstance(item, dict):
                safe_regions.append(
                    {
                        "region_name": _ensure_text(item.get("region_name", "")),
                        "category": _ensure_text(item.get("category", "creative_liberty")),
                        "reason": _ensure_text(item.get("reason", "")),
                        "correction_instruction": _ensure_text(item.get("correction_instruction", "")),
                    }
                )
    normalized["region_verdicts"] = safe_regions

    normalized["verdict"] = _ensure_text(normalized.get("verdict", "approved")).strip().lower()
    if normalized["verdict"] not in {"approved", "correction_required"}:
        normalized["verdict"] = "approved"

    return normalized


def _extract_section(text: str, header: str, next_headers: List[str]) -> str:
    """Extract a section body from call-1 formatted text."""
    if not text:
        return ""
    pattern = re.escape(header) + r"\s*(.*?)\s*(?=" + "|".join([re.escape(h) for h in next_headers]) + r"|\Z)"
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else ""


def _parse_bulleted_lines(section_text: str) -> List[str]:
    if not section_text:
        return []
    lines: List[str] = []
    for raw in section_text.splitlines():
        line = raw.strip()
        if not line:
            continue
        line = re.sub(r"^(?:[-*]|\d+[\).])\s*", "", line).strip()
        if line:
            lines.append(line)
    return lines


def _fallback_analysis_from_call1(free_text_analysis: str) -> dict:
    """Build minimal structured payload from call-1 text when call-2 mapping fails."""
    text = free_text_analysis or ""
    compromised_block = _extract_section(
        text,
        "4) Compromised details:",
        ["5) Ordered corrections:", "6) Final call:"],
    )
    corrections_block = _extract_section(
        text,
        "5) Ordered corrections:",
        ["6) Final call:"],
    )
    final_call_block = _extract_section(text, "6) Final call:", [])

    compromised = _parse_bulleted_lines(compromised_block)
    corrections = _parse_bulleted_lines(corrections_block)

    verdict = "approved"
    fc = final_call_block.upper()
    if "CORRECTION_REQUIRED" in fc:
        verdict = "correction_required"
    elif "APPROVED" in fc:
        verdict = "approved"
    elif corrections or compromised:
        verdict = "correction_required"

    return {
        "reference_summary": "",
        "final_summary": "",
        "region_verdicts": [],
        "compromised_details": compromised,
        "ordered_corrections": corrections,
        "verdict": verdict,
        "justification": "Recovered structured fields from call-1 analysis due to incomplete call-2 mapping.",
        "free_text_analysis": text,
        "module_editing_instructions": {
            "subject_focus": "",
            "visual_style": "",
            "perspective_and_camera": "",
            "lighting_and_environment": "",
            "background_setting": "",
            "color_grading": "",
            "additional_comments": "",
        },
    }


def _is_incomplete_call2_payload(payload: dict, free_text_analysis: str) -> bool:
    """Detect when call-2 failed to map despite call-1 detecting actionable issues."""
    if not isinstance(payload, dict) or not payload:
        return True

    call1_upper = (free_text_analysis or "").upper()
    call1_requires_fix = "CORRECTION_REQUIRED" in call1_upper

    has_compromised = bool(payload.get("compromised_details"))
    has_corrections = bool(payload.get("ordered_corrections"))
    verdict = str(payload.get("verdict", "")).strip().lower()

    if call1_requires_fix and verdict == "approved" and not has_compromised and not has_corrections:
        return True

    if call1_requires_fix and not has_compromised and not has_corrections:
        return True

    return False


def _build_call1_prompt(
    voiceover: str,
    slide_title: str = "",
    slide_content: str = "",
) -> str:
    return f"""
You are validating technical accuracy between two images.

Slide title:
{slide_title or 'N/A'}

Slide content:
{slide_content or 'N/A'}

Voiceover:
{voiceover or 'N/A'}

Follow this exact basic architecture:
Step 1: Identify educational-value HVAC technical details in Image 1 that must not be compromised.
Step 2: Check whether those details are preserved, visible, and correct in Image 2.
Step 3: If compromised, produce ordered corrections to restore technical accuracy.
Perform a skeptical technician audit of Image 2 as if you have 15+ years HVAC field experience. Assume Image 2 may be wrong until each critical anchor from Image 1 is verified (labels, units, ranges, markings, color coding, safety/retard indicators, readability).

Hard rules:
- For a specific region/component, classification must be exactly one category:
  creative_liberty OR technical_inaccuracy (never both).
- Be concise and practical.
- If uncertain on a technical HVAC standard, you may use Google Search before deciding.
- Preserve valid creative liberties.
- Only request corrections for confirmed technical inaccuracies.

Return free-text analysis only using this exact format:
1) Reference technical details (must-preserve):
2) Comparison findings (reference vs final):
3) Region classifications:
- Region: <name> | Category: creative_liberty|technical_inaccuracy | Reason: <short>
4) Compromised details:
5) Ordered corrections:
6) Final call: APPROVED or CORRECTION_REQUIRED
""".strip()


def _build_call2_prompt(
    voiceover: str,
    free_text_analysis: str,
    slide_title: str = "",
    slide_content: str = "",
) -> str:
    return f"""
Map the final corrected judgment into strict JSON.

Call-1 analysis:
{free_text_analysis}

Rules:
- verdict must be exactly 'approved' or 'correction_required'.
- Each region_verdict category must be exactly one of: 'creative_liberty' or 'technical_inaccuracy'.
- Never classify one region into both categories.
- correction_instruction must be empty for creative_liberty.
- If verdict is approved, ordered_corrections must be empty.
- If verdict is correction_required, ordered_corrections must be non-empty and specific.
- Re-check that compromised_details and ordered_corrections are technically consistent with both images.
- Keep output concise and operational.

Reference-faithful mapping requirements:
- reference_summary must include concrete technical anchors visible in Image 1 (labels, units, colors, ranges, markings, symbols).
- final_summary must describe what changed in Image 2 for those same anchors.
- Every ordered correction must specify:
    1) target region/component,
    2) exact attribute to restore (text/value/color/marking/layout cue),
    3) location cue (where on the object/scale),
    4) preservation guardrail (what must remain unchanged).
- Prefer verbatim restoration for critical technical text/units/ranges from Image 1.

Also prepare module_editing_instructions:
- Keep all fields empty except additional_comments.
- additional_comments must be a single STRING, not a list.
- additional_comments must be detailed, executable, and imitate Image 1 technical details exactly for corrected regions.
- additional_comments should include explicit restore directives for labels, units, ranges, and color coding where applicable.
- If no corrections are needed, keep additional_comments empty.

Return JSON only with the TechnicalValidationAnalysis schema fields.
""".strip()


def _build_reapproval_prompt(
    prior_analysis: TechnicalValidationAnalysis,
    voiceover: str,
    slide_title: str = "",
    slide_content: str = "",
) -> str:
    compromised = "\n".join([f"- {x}" for x in prior_analysis.compromised_details]) or "- None listed"
    corrections = "\n".join([f"{i}. {x}" for i, x in enumerate(prior_analysis.ordered_corrections, start=1)]) or "None"

    return f"""
You are doing a post-edit reapproval.

Image order in this call:
- Image 1: reference image
- Image 2: pre-edit final image (before correction)
- Image 3: corrected image (after correction)

Context:
- Voiceover: {voiceover or 'N/A'}
- Slide title: {slide_title or 'N/A'}
- Slide content: {slide_content or 'N/A'}

From prior decision:
Compromised details:
{compromised}

Ordered corrections that were applied:
{corrections}

Tasks:
1) Check if those compromised details actually changed in Image 3 vs Image 2.
2) Verify Image 3 remains faithful to Image 1 for technical HVAC correctness.
3) Decide approved vs correction_required.
4) If still failing, provide remaining issues and new ordered corrections.

Rules:
- Only two categories in spirit: either corrected (approved) or still technically inaccurate.
- Be strict and concise.
- Do not request style-only changes.

Return JSON with exactly:
- verdict
- changed_checks
- remaining_issues
- ordered_corrections
- justification
""".strip()


def _build_reapproval_delta_prompt(prior_analysis: TechnicalValidationAnalysis) -> str:
    """Compact follow-up prompt to avoid resending redundant context in reapproval chat."""
    compromised = "\n".join([f"- {x}" for x in prior_analysis.compromised_details]) or "- None listed"
    corrections = "\n".join([f"{i}. {x}" for i, x in enumerate(prior_analysis.ordered_corrections, start=1)]) or "None"

    return f"""
Use the already provided image order and session context.

From prior decision:
Compromised details:
{compromised}

Ordered corrections that were applied:
{corrections}

Now reapprove by checking what changed and what still fails.
Return JSON with exactly:
- verdict
- changed_checks
- remaining_issues
- ordered_corrections
- justification
""".strip()


def _build_module_editing_instructions(analysis: TechnicalValidationAnalysis) -> dict:
    payload = {
        "subject_focus": "",
        "visual_style": "",
        "perspective_and_camera": "",
        "lighting_and_environment": "",
        "background_setting": "",
        "color_grading": "",
        "additional_comments": "",
    }

    corrections = [c.strip() for c in analysis.ordered_corrections if c and c.strip()]
    compromised = [c.strip() for c in analysis.compromised_details if c and c.strip()]
    technical_region_fixes = [
        r for r in analysis.region_verdicts
        if (r.category or "").strip().lower() == "technical_inaccuracy" and (r.correction_instruction or "").strip()
    ]

    if corrections or compromised or technical_region_fixes:
        lines: List[str] = [
            "Objective: Restore technical fidelity so corrected regions match the reference image exactly.",
        ]

        if analysis.reference_summary.strip():
            lines.append(f"Reference anchors: {analysis.reference_summary.strip()}")
        if analysis.final_summary.strip():
            lines.append(f"Current deviations: {analysis.final_summary.strip()}")

        if compromised:
            lines.append("Compromised technical details to restore:")
            lines.extend([f"- {item}" for item in compromised])

        if technical_region_fixes:
            lines.append("Region-specific technical fixes:")
            for region in technical_region_fixes:
                lines.append(f"- {region.region_name or 'Unnamed region'}: {region.correction_instruction.strip()}")

        if corrections:
            lines.append("Ordered correction steps (apply exactly):")
            lines.extend([f"{i}. {txt}" for i, txt in enumerate(corrections, start=1)])

        lines.append("Hard constraints:")
        lines.append("- Restore technical labels/units/ranges/color coding to match the reference image for corrected regions.")
        lines.append("- Preserve geometry, perspective, and all regions not listed for correction.")
        lines.append("- Apply edits only at the named region/location cues in the correction steps; keep all other regions pixel-stable.")
        lines.append("- Do not rotate, flip, crop, reframe, or change camera angle/object orientation unless explicitly requested.")
        lines.append("- Do not add decorative or stylistic edits unrelated to technical restoration.")

        payload["additional_comments"] = "\n".join(lines)

    return payload


def _empty_module_editing_instructions() -> ModuleEditingInstructions:
    return ModuleEditingInstructions()


def analyse_technical_accuracy(
    reference_image: Image.Image,
    final_image: Image.Image,
    voiceover: str,
    slide_title: str = "",
    slide_content: str = "",
) -> TechnicalValidationAnalysis:
    """Two-call analysis: free-text multi-pass reasoning, then structured mapping."""

    print("[TechnicalAccuracy] Starting two-call analysis")

    client = _get_client()
    reference_bytes = _to_bytes(reference_image)
    final_bytes = _to_bytes(final_image)

    try:
        chat = client.chats.create(
            model=ANALYSIS_MODEL,
            config=types.GenerateContentConfig(
                tools=[{"google_search": {}}],
                thinking_config=types.ThinkingConfig(thinking_level="high"),
                system_instruction=_build_call1_system_instruction(),
            ),
        )

        # Call 1: free-text technical comparison
        print("[TechnicalAccuracy] Call 1: requesting free-text technical comparison")
        response1 = chat.send_message(
            [
                types.Part.from_text(text="Image 1: Reference image"),
                types.Part.from_bytes(data=reference_bytes, mime_type="image/jpeg"),
                types.Part.from_text(text="Image 2: Final output image"),
                types.Part.from_bytes(data=final_bytes, mime_type="image/jpeg"),
                types.Part.from_text(
                    text=_build_call1_prompt(
                        voiceover=voiceover,
                        slide_title=slide_title,
                        slide_content=slide_content,
                    )
                ),
            ]
        )
        free_text_analysis = _text_from_response(response1)
        print(f"[TechnicalAccuracy] Call 1 raw text present: {bool(free_text_analysis)}")
        if not free_text_analysis:
            free_text_analysis = "No free-text technical analysis was produced."
        print(f"[TechnicalAccuracy] Call 1 analysis (first 300 chars): {free_text_analysis[:300]}")

        # Call 2: re-check call-1 directly against images + map to structured output
        print("[TechnicalAccuracy] Call 2: re-checking and mapping to structured output")
        response2 = chat.send_message(
            [
                types.Part.from_text(text="Image 1: Reference image"),
                types.Part.from_bytes(data=reference_bytes, mime_type="image/jpeg"),
                types.Part.from_text(text="Image 2: Final output image"),
                types.Part.from_bytes(data=final_bytes, mime_type="image/jpeg"),
                types.Part.from_text(
                    text=_build_call2_prompt(
                        voiceover=voiceover,
                        free_text_analysis=free_text_analysis,
                        slide_title=slide_title,
                        slide_content=slide_content,
                    )
                ),
            ]
        )

        call2_text = _text_from_response(response2)
        print(f"[TechnicalAccuracy] Call 2 raw text present: {bool(call2_text)}")
        analysis_payload = _normalize_analysis_payload(_extract_json_payload(call2_text or "{}"))
        if _is_incomplete_call2_payload(analysis_payload, free_text_analysis):
            print("[TechnicalAccuracy] Call 2 mapping incomplete; recovering structured fields from call 1")
            analysis_payload = _normalize_analysis_payload(_fallback_analysis_from_call1(free_text_analysis))
        print(f"[TechnicalAccuracy] Call 2 parsed payload present: {bool(analysis_payload)}")
        analysis = TechnicalValidationAnalysis(**analysis_payload)

        analysis.free_text_analysis = free_text_analysis

        if analysis.verdict not in {"approved", "correction_required"}:
            analysis.verdict = "correction_required" if analysis.ordered_corrections else "approved"

        # Enforce single-category rule and correction mapping safety
        cleaned_regions: List[RegionVerdict] = []
        for region in analysis.region_verdicts:
            category = (region.category or "").strip().lower()
            if category not in {"creative_liberty", "technical_inaccuracy"}:
                category = "technical_inaccuracy" if (region.correction_instruction or "").strip() else "creative_liberty"
            region.category = category
            if category == "creative_liberty":
                region.correction_instruction = ""
            cleaned_regions.append(region)
        analysis.region_verdicts = cleaned_regions

        analysis.module_editing_instructions = ModuleEditingInstructions(**_build_module_editing_instructions(analysis))

        if analysis.verdict == "approved":
            analysis.ordered_corrections = []
            analysis.compromised_details = []
            analysis.module_editing_instructions = _empty_module_editing_instructions()

        print(f"[TechnicalAccuracy] Final verdict: {analysis.verdict}")
        print(f"[TechnicalAccuracy] Final compromised details count: {len(analysis.compromised_details)}")
        print(f"[TechnicalAccuracy] Final ordered corrections count: {len(analysis.ordered_corrections)}")

        return analysis
    except Exception as exc:
        print(f"[TechnicalAccuracy] Analysis failed: {exc}")
        raise RuntimeError(f"Technical accuracy validator LLM analysis failed: {exc}") from exc


def _reapprove_corrected_image(
    reference_image: Image.Image,
    pre_edit_image: Image.Image,
    corrected_image: Image.Image,
    prior_analysis: TechnicalValidationAnalysis,
    voiceover: str,
    slide_title: str = "",
    slide_content: str = "",
) -> dict:
    """Reapprove corrected image by checking whether prior compromised items were truly fixed."""

    print("[TechnicalAccuracy] Reapproval: starting post-edit verification")

    client = _get_client()
    reference_bytes = _to_bytes(reference_image)
    pre_edit_bytes = _to_bytes(pre_edit_image)
    corrected_bytes = _to_bytes(corrected_image)

    try:
        chat = client.chats.create(
            model=ANALYSIS_MODEL,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_level="high"),
            ),
        )
        # Seed once with full context; later turn sends only delta instructions.
        print("[TechnicalAccuracy] Reapproval: seeding chat with reference, pre-edit, and corrected images")
        chat.send_message(
            [
                types.Part.from_text(text="Image 1: Reference image"),
                types.Part.from_bytes(data=reference_bytes, mime_type="image/jpeg"),
                types.Part.from_text(text="Image 2: Pre-edit final image"),
                types.Part.from_bytes(data=pre_edit_bytes, mime_type="image/jpeg"),
                types.Part.from_text(text="Image 3: Corrected image"),
                types.Part.from_bytes(data=corrected_bytes, mime_type="image/jpeg"),
                types.Part.from_text(
                    text=_build_reapproval_prompt(
                        prior_analysis=prior_analysis,
                        voiceover=voiceover,
                        slide_title=slide_title,
                        slide_content=slide_content,
                    )
                ),
            ]
        )

        print("[TechnicalAccuracy] Reapproval: requesting compact delta recheck")
        response = chat.send_message(
            [
                types.Part.from_text(text=_build_reapproval_delta_prompt(prior_analysis=prior_analysis)),
            ]
        )

        reapproval_text = _text_from_response(response)
        print(f"[TechnicalAccuracy] Reapproval raw text present: {bool(reapproval_text)}")
        payload = _extract_json_payload(reapproval_text or "{}")
        print(f"[TechnicalAccuracy] Reapproval parsed payload present: {bool(payload)}")
        result = {
            "verdict": _ensure_text(payload.get("verdict", "approved")).strip().lower(),
            "changed_checks": _ensure_text_list(payload.get("changed_checks")),
            "remaining_issues": _ensure_text_list(payload.get("remaining_issues")),
            "ordered_corrections": _ensure_text_list(payload.get("ordered_corrections")),
            "justification": _ensure_text(payload.get("justification", "")),
        }

        if result["verdict"] not in {"approved", "correction_required"}:
            result["verdict"] = "correction_required" if result["remaining_issues"] else "approved"

        if result["verdict"] == "approved":
            result["remaining_issues"] = []
            result["ordered_corrections"] = []

        print(f"[TechnicalAccuracy] Reapproval verdict: {result['verdict']}")
        print(f"[TechnicalAccuracy] Reapproval remaining issues count: {len(result['remaining_issues'])}")

        return result
    except Exception as exc:
        print(f"[TechnicalAccuracy] Reapproval failed: {exc}")
        raise RuntimeError(f"Technical accuracy reapproval failed: {exc}") from exc


def _apply_corrections(
    image: Image.Image,
    analysis: TechnicalValidationAnalysis,
    reference_image: Optional[Image.Image] = None,
) -> Tuple[Image.Image, bool, str]:
    if analysis.verdict == "approved":
        print("[TechnicalAccuracy] Correction step skipped: verdict already approved")
        return image, False, "No technical inaccuracies detected."

    corrections = [c.strip() for c in analysis.ordered_corrections if c and c.strip()]
    if not corrections:
        print("[TechnicalAccuracy] Correction step skipped: no executable correction instructions")
        return image, False, "Correction was required but no executable correction instruction was produced."

    editing_instructions = _build_module_editing_instructions(analysis)
    analysis.module_editing_instructions = ModuleEditingInstructions(**editing_instructions)

    try:
        from agents.graphics_asset_creation.image_editing.image_editing import generate_edited_image

        print(
            f"[TechnicalAccuracy] Applying {len(corrections)} technical correction(s) via image editor "
            f"(target image only)"
        )
        edited, _, _, _ = generate_edited_image(
            reference_image=image,
            editing_instructions=editing_instructions,
            image_size="1K",
            system_instruction=(
                "Apply only the requested technical corrections to the target image. "
                "Position lock: edit only the explicitly named regions/location cues; do not shift edits to nearby areas. "
                "Do not rotate, flip, crop, reframe, change perspective, or change object orientation unless explicitly requested. "
                "Preserve all untouched regions exactly. Preserve valid creative liberties. "
                "Do not introduce annotations or extra components unless explicitly required by the instructions."
            ),
        )
    except Exception as exc:
        print(f"[TechnicalAccuracy] Image editing call failed: {exc}")
        return image, False, f"Image editing module call failed: {exc}"

    if edited is None:
        print("[TechnicalAccuracy] Image editing returned no image")
        return image, False, "Image editing module returned no image."

    print("[TechnicalAccuracy] Correction step completed successfully")
    return edited, True, f"Applied {len(corrections)} technical correction(s)."


def run_technical_accuracy_validator_agent(
    reference_image: Image.Image,
    final_image: Image.Image,
    voiceover: str,
    slide_title: str = "",
    slide_content: str = "",
) -> TechnicalValidationResult:
    """
    Agent 2 behavior:
    - Compare reference image vs final image for technical accuracy preservation.
    - Keep creative liberties untouched.
    - Correct only confirmed technical inaccuracies.
    """

    print("[TechnicalAccuracy] Pipeline started")

    analysis = analyse_technical_accuracy(
        reference_image=reference_image,
        final_image=final_image,
        voiceover=voiceover,
        slide_title=slide_title,
        slide_content=slide_content,
    )

    if analysis.verdict == "approved":
        print("[TechnicalAccuracy] Early exit: no technical corrections required")
        return TechnicalValidationResult(
            analysis=analysis,
            output_image=final_image,
            was_modified=False,
            modification_notes="No technical inaccuracies detected.",
        )

    # Apply correction and reapprove. If still failing, attempt one more correction pass.
    current_image = final_image
    final_output = final_image
    any_modified = False
    note_parts: List[str] = []

    for pass_index in range(2):
        print(f"[TechnicalAccuracy] Correction pass {pass_index + 1}/2")
        output_image, was_modified, notes = _apply_corrections(
            current_image,
            analysis,
            reference_image=reference_image,
        )
        note_parts.append(notes)
        final_output = output_image
        any_modified = any_modified or was_modified

        if not was_modified:
            break

        reapproval = _reapprove_corrected_image(
            reference_image=reference_image,
            pre_edit_image=current_image,
            corrected_image=output_image,
            prior_analysis=analysis,
            voiceover=voiceover,
            slide_title=slide_title,
            slide_content=slide_content,
        )

        note_parts.append(f"Reapproval: {reapproval.get('justification') or reapproval.get('verdict')}")

        if reapproval.get("verdict") == "approved":
            analysis.verdict = "approved"
            analysis.compromised_details = []
            analysis.ordered_corrections = []
            analysis.justification = reapproval.get("justification") or analysis.justification
            analysis.module_editing_instructions = _empty_module_editing_instructions()
            print("[TechnicalAccuracy] Reapproval approved: stopping correction loop")
            break

        analysis.verdict = "correction_required"
        analysis.compromised_details = [x.strip() for x in reapproval.get("remaining_issues", []) if x and x.strip()]
        analysis.ordered_corrections = [x.strip() for x in reapproval.get("ordered_corrections", []) if x and x.strip()]
        analysis.justification = reapproval.get("justification") or analysis.justification
        analysis.module_editing_instructions = ModuleEditingInstructions(**_build_module_editing_instructions(analysis))

        # No usable follow-up correction plan returned -> stop iteration
        if not analysis.ordered_corrections:
            print("[TechnicalAccuracy] Reapproval requested correction but returned no follow-up plan")
            break

        current_image = output_image

    print(f"[TechnicalAccuracy] Pipeline completed. Modified: {any_modified}; Final verdict: {analysis.verdict}")

    return TechnicalValidationResult(
        analysis=analysis,
        output_image=final_output,
        was_modified=any_modified,
        modification_notes=" | ".join([n for n in note_parts if n]),
    )
