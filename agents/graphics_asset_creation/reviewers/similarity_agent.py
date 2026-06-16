"""
Stage A: Similarity Agent
=========================
Single cognitive objective: Compare an edited image against reference images
and produce a derivative-risk verdict. No transformation instructions,
no styling-guide checks, no Google Search.
"""

import re
import time
import xml.etree.ElementTree as ET
from PIL import Image
from typing import List, Tuple, Dict, Any, Optional
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from langsmith import traceable

from agents.graphics_asset_creation.reviewers.voiceover_reviewer import (
    REVIEWER_MODEL,
    _get_client,
    prepare_image_for_gemini,
)


# ── Token extraction helper ─────────────────────────────────────────────────

def _extract_tokens(response) -> tuple:
    """Extract (input_tokens, output_tokens) from a Gemini API response."""
    input_t = 0
    output_t = 0
    if hasattr(response, 'usage_metadata') and response.usage_metadata:
        input_t = getattr(response.usage_metadata, 'prompt_token_count', 0) or 0
        output_t = getattr(response.usage_metadata, 'candidates_token_count', 0) or 0
    return input_t, output_t

# ── Output schema ────────────────────────────────────────────────────────────

class SimilarityResult(BaseModel):
    """Focused output: similarity assessment only."""
    description: str = Field(
        description="Factual description of the current image: subject, angle, style, environment."
    )
    similarity_scores: str = Field(
        default="",
        description="Compact summary, e.g. 'Reference 1: 72.0%'."
    )
    similarity_analysis: str = Field(
        default="",
        description="Per-reference JSON array with score, verdict, visual_analysis, recommendation."
    )
    visual_anchors_by_reference: List[List[Dict[str, Any]]] = Field(
        default_factory=list,
        description="Per-reference detailed visual anchors, each with name/category/description/status/importance/breakable."
    )
    anchor_descriptions: str = Field(
        default="",
        description="Formatted text of all anchor descriptions grouped by reference, ready for downstream agents."
    )
    anchor_breaking_suggestions: str = Field(
        default="",
        description="Per-anchor editing suggestions to break preserved/partial anchors, informed by image description and slide context."
    )
    verdict: str = Field(
        description="'Yes' if all references are ≤49 similarity (safe). 'No' otherwise."
    )
    # Token tracking for cost calculation
    input_tokens: int = Field(default=0, description="Total input tokens consumed by all Gemini calls in this agent.")
    output_tokens: int = Field(default=0, description="Total output tokens consumed by all Gemini calls in this agent.")


class VisualAnchor(BaseModel):
    name: str = ""
    category: str = "other"
    description: str = ""
    status: str = "partially_changed"
    importance: str = "medium"
    breakable: bool = True  # False = functional/educational content that must be preserved


class AnchorSummary(BaseModel):
    high_risk_anchors: List[str] = Field(default_factory=list)
    broken_anchors: List[str] = Field(default_factory=list)


def _parse_anchor_items(text: str) -> List[str]:
    if not text:
        return []

    raw = text.strip()
    if not raw:
        return []

    lines = [line.strip(" \t\r\n-•") for line in raw.splitlines() if line.strip()]
    if len(lines) == 1 and "," in lines[0]:
        lines = [part.strip(" \t\r\n-•") for part in lines[0].split(",") if part.strip()]

    cleaned = []
    for item in lines:
        lower = item.lower()
        if lower.startswith("list anchors") or lower in {"n/a", "none"}:
            continue
        cleaned.append(item)
    return cleaned

# ── Pairwise LLM comparison (kept from original) ────────────────────────────

_SIMILARITY_PROMPT = """<high_level_task>
You are a Copyright Derivative Risk Reviewer Agent. 
Your task is to compare a reference image and an edited/transformed image and assess whether the edited image is likely derived from the reference image for copyright purposes. 

You must balance compositional similarity with meaningful transformations that reduce copyright risk.
Your output must estimate derivative risk, not legal certainty.
</high_level_task>

<input>
<reference_image>
[Image 1: Original or reference image]
</reference_image>

<edited_image>
[Image 2: Edited, generated, or transformed image]
</edited_image>
</input>

<instructions>
Evaluate similarity using a Balanced Derivative Risk Framework.

1. Focus on EXPRESSIVE similarity, not functional or categorical similarity.
2. For creative works (art, illustrations, infographics): Composition and staging are dominant factors.
3. For standard industrial products and equipment: Basic form/shape similarity is LOW risk unless distinctive creative elements are copied.
4. Functional industrial layouts and standard architectural arrangements must be treated as LOW expressive similarity.
5. SIGNIFICANT transformations that reduce copyright risk include:
   - Branding/logo removal or replacement (MAJOR factor)
   - Camera angle or perspective changes (MAJOR factor)
   - Style changes (render vs photo, illustration style) (MODERATE factor)
   - Lighting and color palette changes (MODERATE factor)
   - Background/environment changes (MODERATE factor)
6. Apply a Recognizability Test: Would an average viewer recognize Image 2 as THIS SPECIFIC copyrighted image, or just the same category of object?
7. Detect true derivative patterns: tracing, layout reuse of creative compositions, distinctive creative design replication.
8. For industrial/commercial products: The same TYPE of object is not copyright infringement if branding differs and angle/presentation differs.
9. If similarity_score ≤ 49, verdict must be YES (safe).
10. If similarity_score ≥ 50, verdict must be NO (not safe).
11. Output must be in XML only. No markdown, no explanations outside XML.

12. VISUAL ANCHOR DETECTION (MANDATORY)
   - Identify visual anchors that make Image 1 recognizable as a specific copyrighted instance.
   - Anchors include but are not limited to:
     * Camera angle and projection type
     * Object spatial layout and symmetry
     * Subject silhouette and framing
     * Lighting model and direction
     * Background/environment class
     * Material/texture appearance
     * Distinctive UI or surface patterns
     * Composition geometry (centroid position, cropping, horizon line)
   - For each anchor, determine whether it is preserved, partially changed, or broken in Image 2.
   - Anchors must be explicitly listed in the output for downstream transformation planning.

13. ANCHOR BREAKABILITY CLASSIFICATION (MANDATORY)
   For each anchor, classify whether it is BREAKABLE or PROTECTED:

   BREAKABLE (breakable=true) — purely stylistic/incidental, safe to change:
   - Camera angle, perspective, or framing
   - Lighting style or direction
   - Color palette, material finish, or surface texture
   - Background or environment setting
   - Visual rendering style (photo vs illustration)
   - Subject identity (person's appearance, clothing, branding)
   - Compositional framing that is purely aesthetic

   PROTECTED (breakable=false) — functional/educational content, must NOT be broken:
   - Specific component layouts or configurations that teach how something works
   - Pipe routing, wiring paths, or mechanical arrangements that are the SUBJECT of instruction
   - Spatial relationships between technical components required to demonstrate a process
   - Equipment configurations that define the technical concept being taught
   - Any arrangement where altering it would change what the image teaches
   - HVAC component color codes used to distinguish pipe/duct/line types — these are industry-standard conventions that carry technical meaning
   - Technical labels, ratings, or color markings on HVAC/mechanical components (e.g. pressure ratings, flow direction arrows, BTU/capacity values, model or spec labels) — altering these changes the technical information being taught

   Rule: If the anchor describes HOW something LOOKS (style, angle, lighting) → BREAKABLE
   Rule: If the anchor describes WHAT IS THERE and its FUNCTIONAL ARRANGEMENT → PROTECTED

</instructions>

<evaluation_breakdown>
Step 1: Identify Image Type
- Creative/artistic work (higher copyright protection) vs Standard industrial/commercial product (lower protection)
- Unique creative composition vs Functional standard layout

Step 2: Evaluate Core Expressive Similarity (START at 100, subtract differences)
For creative works:
- Creative composition/layout differences (-0 to -30 points)
- Distinctive artistic design differences (-0 to -25 points)
- Camera angle and framing differences (-0 to -20 points)
- Visual style differences (-0 to -15 points)

For industrial/commercial products:
- Basic form similarity is acceptable if transformations are present
- Compositional differences (-0 to -20 points)  
- Angle/perspective differences (-0 to -25 points)
- Style rendering differences (-0 to -15 points)

Step 3: Apply MAJOR Transformation Deductions (subtract from score)
- All branding/logos removed or different: -25 to -35 points
- Significant camera angle change (>30 degrees): -20 to -30 points
- Major perspective shift (orthographic to isometric, etc.): -15 to -25 points
- Style transformation (photo→render, realistic→illustration): -15 to -20 points

Step 4: Apply MODERATE Transformation Deductions
- Lighting direction/quality significantly different: -10 to -15 points
- Color palette substantially changed: -10 to -15 points
- Background/environment completely different: -10 to -15 points
- Material/texture finish significantly altered: -5 to -10 points

Step 5: Apply Derivative Risk Overrides (INCREASE score only if applicable)
If ANY of the following are true, ADD points back or set minimum score:
- Appears directly traced or reconstructed with minimal changes: ADD +30 points
- Identical creative infographic template or artistic composition: ADD +25 points
- Distinctive non-functional creative design clearly copied: ADD +20 points
- Recognizable as THIS SPECIFIC copyrighted work (not just same category): ADD +15 points

Step 6: Industrial Product Special Rule
If image is industrial/commercial equipment AND:
- Branding is removed/changed
- Camera angle differs by >15 degrees
- At least 2 other transformations applied
THEN: Maximum score should be 45 (SAFE) unless distinctive creative elements are copied

Step 7: Final Recognizability Test
Would a typical observer identify Image 2 as a derivative of THIS SPECIFIC copyrighted Image 1, or just recognize it as the same category/type of product?
- If SPECIFIC image recognition: INCREASE score toward 50+
- If only category recognition: DECREASE score toward safe range

Step 8: Risk Level Mapping
- 85–100 → very_high
- 70–84 → high
- 50–69 → medium  
- 30–49 → low
- 0–29 → very_low
</evaluation_breakdown>

<output>
  <similarity_score>0-100</similarity_score>
  <verdict>YES or NO</verdict>

  <visual_analysis>
    Detailed comparison of composition, subject design, camera perspective, style, lighting, colors, and background.
  </visual_analysis>

  <visual_anchors>
    <anchor>
      <name>Detailed Anchor description</name>
      <category>camera|layout|lighting|background|material|silhouette|composition|ui_pattern|other</category>
      <description>Explanation of why this is a strong visual anchor for recognizing the original image.</description>
      <status>preserved|partially_changed|broken</status>
      <importance>high|medium|low</importance>
      <breakable>true|false</breakable>
    </anchor>
    (Repeat for all detected anchors)
  </visual_anchors>

  <anchor_summary>
    <high_risk_anchors>
      List anchors strongly preserved and contributing to derivative risk.
    </high_risk_anchors>
    <broken_anchors>
      List anchors significantly transformed or removed.
    </broken_anchors>
  </anchor_summary>

  <recommendation>
    Final assessment on derivative risk and whether the transformation is likely sufficient for copyright evasion.
  </recommendation>
</output>
"""


def _calculate_image_similarity(
    img1: Image.Image,
    img2: Image.Image,
    client: genai.Client,
) -> Tuple[float, Dict[str, Any]]:
    """
    Pairwise derivative-risk comparison via LLM visual analysis.
    Returns (score 0-100, metrics dict).
    """
    max_retries = 3
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            img1_bytes = prepare_image_for_gemini(img1)
            img2_bytes = prepare_image_for_gemini(img2)

            content_parts = [
                types.Part.from_bytes(data=img1_bytes, mime_type="image/jpeg"),
                types.Part.from_text(text="<reference_image>"),
                types.Part.from_bytes(data=img2_bytes, mime_type="image/jpeg"),
                types.Part.from_text(text="<edited_image>"),
                types.Part.from_text(text=_SIMILARITY_PROMPT),
            ]

            from agents.graphics_asset_creation.reviewers.voiceover_reviewer import call_llm_with_retry
            from agents.graphics_asset_creation.automated.llm_call_tracker import tracker as llm_tracker
            with llm_tracker.call(REVIEWER_MODEL, "Similarity Analysis (Pairwise)") as call_info:
                response = call_llm_with_retry(
                    client.models.generate_content,
                    model=REVIEWER_MODEL,
                    contents=[types.Content(role="user", parts=content_parts)],
                    config=types.GenerateContentConfig(
                        thinking_config=types.ThinkingConfig(
                            thinking_level="medium",
                        )
                    ),
                )
                call_info.set_response(response)

            response_text = response.text.strip()

            # Parse XML
            xml_start = response_text.find("<output>")
            xml_end = response_text.find("</output>") + len("</output>")
            if xml_start != -1 and xml_end > xml_start:
                response_text = response_text[xml_start:xml_end]

            try:
                root = ET.fromstring(response_text)
                similarity_score = float(root.find("similarity_score").text or 0)
                verdict = (root.find("verdict").text or "NO").strip()
                visual_analysis = (root.find("visual_analysis").text or "").strip()
                recommendation = (root.find("recommendation").text or "").strip()

                visual_anchors: List[Dict[str, str]] = []
                anchors_node = root.find("visual_anchors")
                if anchors_node is not None:
                    for anchor_node in anchors_node.findall("anchor"):
                        breakable_text = (anchor_node.findtext("breakable") or "true").strip().lower()
                        anchor_obj = VisualAnchor(
                            name=(anchor_node.findtext("name") or "").strip(),
                            category=(anchor_node.findtext("category") or "other").strip(),
                            description=(anchor_node.findtext("description") or "").strip(),
                            status=(anchor_node.findtext("status") or "partially_changed").strip(),
                            importance=(anchor_node.findtext("importance") or "medium").strip(),
                            breakable=(breakable_text != "false"),
                        )
                        visual_anchors.append(anchor_obj.model_dump())

                anchor_summary_node = root.find("anchor_summary")
                if anchor_summary_node is not None:
                    anchor_summary = AnchorSummary(
                        high_risk_anchors=_parse_anchor_items(anchor_summary_node.findtext("high_risk_anchors") or ""),
                        broken_anchors=_parse_anchor_items(anchor_summary_node.findtext("broken_anchors") or ""),
                    )
                else:
                    anchor_summary = AnchorSummary()
            except Exception:
                score_m = re.search(r"<similarity_score>(\d+(?:\.\d+)?)</similarity_score>", response_text)
                verdict_m = re.search(r"<verdict>(YES|NO)</verdict>", response_text, re.IGNORECASE)
                analysis_m = re.search(r"<visual_analysis>(.*?)</visual_analysis>", response_text, re.DOTALL)
                rec_m = re.search(r"<recommendation>(.*?)</recommendation>", response_text, re.DOTALL)

                similarity_score = float(score_m.group(1)) if score_m else 0.0
                verdict = verdict_m.group(1).upper() if verdict_m else "NO"
                visual_analysis = analysis_m.group(1).strip() if analysis_m else ""
                recommendation = rec_m.group(1).strip() if rec_m else ""

                visual_anchors: List[Dict[str, str]] = []
                for anchor_match in re.finditer(r"<anchor>(.*?)</anchor>", response_text, re.DOTALL | re.IGNORECASE):
                    block = anchor_match.group(1)
                    name_m = re.search(r"<name>(.*?)</name>", block, re.DOTALL | re.IGNORECASE)
                    category_m = re.search(r"<category>(.*?)</category>", block, re.DOTALL | re.IGNORECASE)
                    desc_m = re.search(r"<description>(.*?)</description>", block, re.DOTALL | re.IGNORECASE)
                    status_m = re.search(r"<status>(.*?)</status>", block, re.DOTALL | re.IGNORECASE)
                    importance_m = re.search(r"<importance>(.*?)</importance>", block, re.DOTALL | re.IGNORECASE)
                    breakable_m = re.search(r"<breakable>(.*?)</breakable>", block, re.DOTALL | re.IGNORECASE)
                    breakable_text = (breakable_m.group(1).strip().lower() if breakable_m else "true")
                    anchor_obj = VisualAnchor(
                        name=(name_m.group(1).strip() if name_m else ""),
                        category=(category_m.group(1).strip() if category_m else "other"),
                        description=(desc_m.group(1).strip() if desc_m else ""),
                        status=(status_m.group(1).strip() if status_m else "partially_changed"),
                        importance=(importance_m.group(1).strip() if importance_m else "medium"),
                        breakable=(breakable_text != "false"),
                    )
                    visual_anchors.append(anchor_obj.model_dump())

                high_risk_m = re.search(r"<high_risk_anchors>(.*?)</high_risk_anchors>", response_text, re.DOTALL | re.IGNORECASE)
                broken_m = re.search(r"<broken_anchors>(.*?)</broken_anchors>", response_text, re.DOTALL | re.IGNORECASE)
                anchor_summary = AnchorSummary(
                    high_risk_anchors=_parse_anchor_items(high_risk_m.group(1) if high_risk_m else ""),
                    broken_anchors=_parse_anchor_items(broken_m.group(1) if broken_m else ""),
                )

            return similarity_score, {
                "similarity_score": round(similarity_score, 2),
                "verdict": verdict,
                "visual_analysis": visual_analysis,
                "recommendation": recommendation,
                "visual_anchors": visual_anchors,
                "anchor_summary": anchor_summary.model_dump(),
            }

        except Exception as e:
            last_error = e
            if attempt < max_retries:
                print(f"  Similarity attempt {attempt} failed: {e}. Retrying...")
                time.sleep(1)
            else:
                print(f"  Similarity failed after {max_retries} attempts: {e}")

    return 0.0, {"error": str(last_error), "visual_analysis": "Failed to analyze similarity after multiple attempts"}


# ── Top-level description prompt (standalone, no tools) ──────────────────────

_DESCRIBE_PROMPT = """Provide a comprehensive, detailed description of this image covering ALL technical and visual aspects:

**TECHNICAL DETAILS:**
- Main subject: Identify equipment/object with specific type, model characteristics, or category
- Components visible: List all visible parts, features, controls, indicators, connectors, or technical elements
- Materials & finish: Describe surface materials, textures, wear patterns, condition
- Technical specifications visible: Any visible measurements, ratings, labels, or technical markings

**VISUAL & COMPOSITIONAL DETAILS:**
- Camera angle & perspective: Exact viewpoint (front, side, elevated, isometric, etc.) with approximate angles
- Framing & composition: How the subject is positioned, what's in focus, depth of field
- Visual style: Photography, technical illustration, 3D render, diagram, etc. — be specific about rendering quality
- Lighting: Direction, quality (soft/hard), shadows, highlights, color temperature
- Color palette: Dominant colors, saturation levels, overall tone

**CONTEXTUAL DETAILS:**
- Environment/background: Setting, location type, surrounding elements, props
- Branding/text: Any visible logos, labels, model numbers, warning stickers, or text
- Scale indicators: Objects or context that indicate size
- Notable features: Anything distinctive, unique, or particularly prominent

Provide 2-3 detailed paragraphs. Be factual and thorough — these details will guide transformation planning. Do NOT suggest improvements or judge quality."""


# ── Anchor-breaking suggestions LLM call ────────────────────────────────────

def _generate_anchor_breaking_suggestions(
    client: genai.Client,
    image_description: str,
    all_visual_anchors: List[List[Dict[str, str]]],
    slide_title: str = "",
    voiceover: str = "",
) -> str:
    """
    Dedicated LLM call: For each preserved/partially-changed anchor, generate
    a concrete editing suggestion that breaks its recognizability.

    Uses the detailed image description so the model understands the full
    visual context — not just the anchor name.
    """
    # Collect only BREAKABLE anchors that need breaking
    # Protected anchors (breakable=False) describe functional/educational content — skip them
    anchors_to_break: List[str] = []
    protected_anchors: List[str] = []
    for ref_idx, ref_anchors in enumerate(all_visual_anchors, 1):
        for a in ref_anchors:
            if not isinstance(a, dict):
                continue
            status = a.get("status", "").lower()
            if status not in ("preserved", "unchanged", "identical", "partially_changed"):
                continue
            name = a.get("name", "?")
            cat = a.get("category", "?")
            desc = a.get("description", "")
            imp = a.get("importance", "medium")
            is_breakable = a.get("breakable", True)  # default True for legacy anchors without field
            if not is_breakable:
                protected_anchors.append(f"- [{cat}] {name}")
                continue
            anchors_to_break.append(
                f"- Ref {ref_idx} | [{cat}] {name} | status={status} | importance={imp}\n  Description: {desc}"
            )

    if protected_anchors:
        print(f"[SIMILARITY AGENT] Skipping {len(protected_anchors)} protected (educational) anchors: {protected_anchors}")

    if not anchors_to_break:
        print(f"[SIMILARITY AGENT] No breakable anchors to process — all preserved anchors are educational/protected.")
        return ""  # Nothing to break

    prompt = f"""You are an expert image transformation strategist.

BELOW IS A DETAILED DESCRIPTION OF THE CURRENT IMAGE:
{image_description}

SLIDE CONTEXT:
- Title: {slide_title or 'N/A'}
- Voiceover: {voiceover or 'N/A'}

The following visual anchors are STILL recognizable from the reference image(s).
For EACH anchor below, suggest a specific, actionable editing instruction that
would break its recognizability while preserving the educational value for the
slide context above.

ANCHORS TO BREAK:
{chr(10).join(anchors_to_break)}

CRITICAL CONSTRAINTS — YOU MUST FOLLOW THESE:

1. **MINIMAL TRANSFORMATIONS ONLY** — Suggest changing ONE attribute (color, angle, material, texture) per anchor
2. **NO ADDITIONS** — NEVER suggest adding new components, objects, logos, labels, or elements
3. **NO WHOLESALE REPLACEMENTS** — NEVER say "replace X with Y" — instead say "change X's [attribute]"
4. **NO MULTI-STEP CHANGES** — One suggestion per anchor, not multiple stacked edits
5. **PRESERVE EDUCATIONAL PURPOSE** — If changing an anchor would break the slide's teaching goal, suggest ONLY surface-level changes (color, texture, minor angle adjustment)
6. **TRANSFORMATIONS > REMOVALS** — Prefer "change color to..." over "remove..."
7. **NO CAMERA/PERSPECTIVE CHANGES** — Do NOT suggest rotating camera, changing viewpoint, or repositioning subjects
8. **SPECIFIC MEASURABLE PARAMETERS** — Use degrees, color names, materials — no vague terms8. **CROSS-CHECK AGAINST SLIDE CONTEXT** — Before suggesting a change, verify: does this anchor describe content REQUIRED by the voiceover to teach the concept? If YES → skip it entirely, do NOT suggest anything for it
   - If voiceover says "pipes running horizontally to the unit" → pipe layout anchor is REQUIRED → skip
   - If voiceover says "technician inspecting the system" → camera angle and worker pose are INCIDENTAL → breakable
FORBIDDEN PATTERNS (DO NOT OUTPUT THESE):
❌ "Replace [object] with [different object]"
❌ "Add [new component]"
❌ "Move [object] to [different location]"
❌ "Rotate camera [degrees]"
❌ "Remove [element] and add [element]"
❌ Multiple changes in one suggestion

ALLOWED PATTERNS (USE THESE):
✅ "Change [object]'s color from [X] to [Y]"
✅ "Adjust [object]'s angle by [N] degrees clockwise"
✅ "Change [object]'s material from [X] to [Y]"
✅ "Modify [object]'s texture to [specific texture]"
✅ "Darken/lighten [object] by [amount]"

EVALUATION CHECKLIST (before generating each suggestion):
- READ the image description — what is actually present?
- READ the anchor description — why is it recognizable?
- READ the slide context — what must the image still demonstrate?
- Is this anchor CRITICAL to the educational message? If YES → suggest only surface changes
- Can I break this anchor with ONE attribute change? If NO → pick the most effective attribute
- Am I adding anything new? If YES → INVALID, try again
- Am I replacing an object? If YES → INVALID, try again

OUTPUT FORMAT (one per anchor, keep it tight):
Anchor: [anchor name]
Suggestion: [specific single-attribute editing instruction]

Generate MINIMAL suggestions now:"""

    try:
        from agents.graphics_asset_creation.reviewers.voiceover_reviewer import call_llm_with_retry
        from agents.graphics_asset_creation.automated.llm_call_tracker import tracker as llm_tracker
        with llm_tracker.call(REVIEWER_MODEL, "Anchor-Breaking Strategy") as call_info:
            response = call_llm_with_retry(
                client.models.generate_content,
                model=REVIEWER_MODEL,
                contents=[types.Content(role="user", parts=[
                    types.Part.from_text(text=prompt)
                ])],
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(
                        thinking_level="medium",
                    )
                ),
            )
            call_info.set_response(response)
        suggestions = response.text.strip()
        print(f"[SIMILARITY AGENT] Generated anchor-breaking suggestions ({len(anchors_to_break)} anchors)")
        return suggestions
    except Exception as e:
        print(f"[SIMILARITY AGENT] Anchor-breaking suggestions failed: {e}")
        return ""


# ── Public entry point ───────────────────────────────────────────────────────

@traceable(metadata={"agent_name": "similarity_agent"})
def similarity_agent(
    image: Image.Image,
    reference_images: Optional[List[Image.Image]] = None,
    slide_title: str = "",
    voiceover: str = "",
) -> SimilarityResult:
    """
    Stage A — Similarity-only assessment.

    Args:
        image: The edited / generated image to evaluate.
        reference_images: Original reference images to compare against.

    Returns:
        SimilarityResult with description, scores, and a safe/not-safe verdict.
    """
    import json as _json

    print("\n[SIMILARITY AGENT] Starting...")
    client = _get_client()
    if not client:
        raise ValueError("Client not initialized")

    # 1. Get a factual description of the image
    img_bytes = prepare_image_for_gemini(image)
    from agents.graphics_asset_creation.reviewers.voiceover_reviewer import call_llm_with_retry
    from agents.graphics_asset_creation.automated.llm_call_tracker import tracker as llm_tracker
    with llm_tracker.call(REVIEWER_MODEL, "Image Description") as call_info:
        desc_response = call_llm_with_retry(
            client.models.generate_content,
            model=REVIEWER_MODEL,
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
                        types.Part.from_text(text=_DESCRIBE_PROMPT),
                    ],
                )
            ],
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(
                    thinking_level="medium",
                )
            ),
        )
        call_info.set_response(desc_response)
    description = desc_response.text.strip()

    # 2. Pairwise similarity against each reference
    similarity_scores_raw: List[Tuple[int, float, Dict[str, Any]]] = []
    overall_verdict = "Yes"  # safe until proven otherwise

    if reference_images:
        print(f"  Comparing against {len(reference_images)} reference image(s)...")
        for idx, ref_img in enumerate(reference_images):
            score, metrics = _calculate_image_similarity(ref_img, image, client)
            similarity_scores_raw.append((idx + 1, score, metrics))
            print(f"    Reference {idx + 1} similarity: {score:.1f}%")
            if score >= 50:
                overall_verdict = "No"

    # 3. Assemble result
    summary_parts = []
    detailed = []
    all_visual_anchors: List[List[Dict[str, str]]] = []
    anchor_desc_parts: List[str] = []
    for idx, score, metrics in similarity_scores_raw:
        summary_parts.append(f"Reference {idx}: {score:.1f}%")
        ref_anchors = metrics.get("visual_anchors", [])
        if not isinstance(ref_anchors, list):
            ref_anchors = []
        all_visual_anchors.append(ref_anchors)

        # Print and collect anchor descriptions
        if ref_anchors:
            ref_desc_lines = [f"Reference {idx} anchors:"]
            for a in ref_anchors:
                if not isinstance(a, dict):
                    continue
                name = a.get("name", "?")
                cat = a.get("category", "?")
                desc = a.get("description", "")
                status = a.get("status", "?")
                imp = a.get("importance", "?")
                breakable = a.get("breakable", True)
                protection = "BREAKABLE" if breakable else "PROTECTED-educational"
                line = f"  [{cat}] {name} — {desc} (status={status}, importance={imp}, {protection})" if desc else f"  [{cat}] {name} (status={status}, importance={imp}, {protection})"
                ref_desc_lines.append(line)
            block = "\n".join(ref_desc_lines)
            print(f"    {block}")
            anchor_desc_parts.append(block)

        detailed.append({
            "reference_index": idx,
            "similarity_score": score,
            "verdict": metrics.get("verdict", "NO"),
            "visual_analysis": metrics.get("visual_analysis", ""),
            "recommendation": metrics.get("recommendation", ""),
            "visual_anchors": ref_anchors,
            "risk_level": (
                "CRITICAL - Very High Similarity" if score >= 95
                else "HIGH - Significant Similarity" if score >= 85
                else "MODERATE - Some Similarity" if score >= 70
                else "LOW - Acceptable Difference" if score >= 50
                else "VERY LOW - Safe"
            ),
        })

    anchor_descriptions_text = "\n".join(anchor_desc_parts) if anchor_desc_parts else ""

    # 4. Generate anchor-breaking suggestions (only when unsafe)
    anchor_breaking_text = ""
    if overall_verdict == "No" and all_visual_anchors:
        anchor_breaking_text = _generate_anchor_breaking_suggestions(
            client=client,
            image_description=description,
            all_visual_anchors=all_visual_anchors,
            slide_title=slide_title,
            voiceover=voiceover,
        )

    result = SimilarityResult(
        description=description,
        similarity_scores=", ".join(summary_parts),
        similarity_analysis=_json.dumps(detailed, indent=2) if detailed else "",
        visual_anchors_by_reference=all_visual_anchors,
        anchor_descriptions=anchor_descriptions_text,
        anchor_breaking_suggestions=anchor_breaking_text,
        verdict=overall_verdict,
    )

    print(f"[SIMILARITY AGENT] Verdict: {result.verdict}\n")
    return result
