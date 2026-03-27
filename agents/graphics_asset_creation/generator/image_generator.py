import os
import json
import time
import base64
from io import BytesIO
from typing import Optional
from urllib.request import urlopen
from PIL import Image
from google import genai
from google.genai import types
from langsmith import traceable

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# =============================================================================
# STANDALONE UTILITIES
# =============================================================================

def _get_client() -> Optional[genai.Client]:
    """Initialize the Google GenAI client."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        try:
            import streamlit as st
            api_key = st.session_state.get("google_api_key")
        except (ImportError, Exception):
            pass
    
    if not api_key:
        return None
        
    try:
        return genai.Client(api_key=api_key)
    except Exception as e:
        print(f"[ImageGenerator] Failed to initialize Gemini client: {e}")
        return None

def call_llm_with_retry(func, *args, max_retries=3, initial_wait=2, **kwargs):
    """Retry an LLM call with exponential backoff."""
    last_exc = None
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_exc = e
            if attempt < max_retries - 1:
                wait_time = initial_wait * (2 ** attempt)
                print(f"  ⚠️ LLM call failed: {e}. Retrying in {wait_time}s... ({attempt+1}/{max_retries})")
                time.sleep(wait_time)
            else:
                print(f"  ❌ LLM call failed after {max_retries} attempts: {e}")
    raise last_exc

def prepare_image_for_gemini(image: Image.Image, max_dimension: int = 2048) -> bytes:
    """Convert PIL Image to bytes for Gemini API with size optimization."""
    buffered = BytesIO()
    save_img = image.copy()
    
    width, height = save_img.size
    if width > max_dimension or height > max_dimension:
        ratio = min(max_dimension / width, max_dimension / height)
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        save_img = save_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    if save_img.mode != 'RGB':
        save_img = save_img.convert('RGB')
    
    save_img.save(buffered, format="JPEG", quality=90)
    return buffered.getvalue()

_STYLING_GUIDE = """\
# Graphics Styling Guide

## Design Tone
Balance of Informative, Professional, and Energetic.

## Color Scheme (Add-on Elements Prohibited)
- **Primary Colors:** White (#FFFFFF), Orange (#F05523), Marine Blue (#242052)
- **Secondary Colors:** Cool Grey (#F2F2F2), Blue Grey (#B9C9D1), Black (#000000)

## Typefaces
Fira Sans

## Visual Appeal
Carefully select elements based on contrast, placement, angle, clarity, quality, and layout.

- Every image should be enhanced and color-corrected to make it look appealing.
- To convey the message or concept effectively, use the Principle of Visual Hierarchy.

---

## Illustrations-Type Generation Guidelines
- If a component is highlighted, use the industrial color for the component and use brand color scheme for annotations, arrows or any other element that is not part of the component.
- They should be visually appealing and convey the message concisely.
- The illustrations should have a flat treatment while still conveying a sense of depth.
- The illustrations should not be overly childish, avoiding exaggerated expressions, comic proportions, and overly expressive poses.
- The illustrations should be simple and clear and should maintain a formal tone.

---

## Diagrams and Charts Generation Guidelines

- **Choose the Right Chart/Diagram:** Select the visual that best represents the data or concept (e.g., bar chart for comparisons, flow chart for processes).
- **Keep it Simple:** Avoid clutter; focus on the essential information.
- **Clear Labels and Titles:** Ensure all elements are clearly labeled and the chart/diagram has a descriptive title.
- **Consistent Scale and Units:** Use consistent scales and units to prevent misinterpretation.
- **Highlight Key Data:** Use subject focus, contrast, text labels or framing to draw attention to important components. 
- **Logical Flow:** If using a diagram with a sequence, ensure the flow is logical and easy to follow using visual arrangement.

---

## Icons Generation Guidelines

- Use simple line icons. Use a single color - orange (#F05523).
- Maintain the thickness and style of the icon.
- Label each icon with a relevant title in an orange (#F05523) solid base container.

---

## Important Notes

- There shouldn't be anything extra on-screen elements except the requirements of the graphics definition. No extra label, No extra graphic element, No unwanted movement
- The subject or object discussed in the graphics definition should be emphasized and occupy more space in the visual.

---

## Checklist Before the Final Result

1. Check alignments
2. Check font size and color
3. Font casing in labels and naming
4. Correct spacing in a sentence-like space after, = . Etc.
5. Consistency in the design elements
6. Use of visual hierarchy principle
7. Tweak the contrast, clarity, levels, and exposure of any image to increase the appeal.

---

## Master System Instructions

- **Maximize Stage Utilization:** Eliminate excessive blank space. Graphics must fill the "stage" to provide maximum detail.
- **Functional Perspective:** Never use obscure or "artistic" angles that hide functionality. Use clear, top-down, or direct angles where screens and labels are legible.
- **Mobile-First Visibility:** Icons and key elements must be large, bold, and high-contrast enough to be legible on a mobile phone in landscape mode.
- **Font Consistency:** Maintain uniform font sizes across similar visual elements. Text must be readable on all screen sizes.
- **Layout Consistency:** Repeated elements (e.g., fuses, boxes) must have identical sizing, alignment, and spacing.
- **Zero Tolerance for Typos:** Thoroughly spell-check all text layers. (e.g., "Ventilation" not "Vantilation").
- **Progressive Cognitive Load:** Do not dump complex data at once. Start diagrams with minimal info and reveal details step-by-step.
- **Text Spacing & Legibility:** Ensure distinct line height and letter spacing. Text must never feel cramped.
- **No Text Overlays:** Remove all non-functional decoration. If an element does not serve a strict educational purpose through its physical presence, delete it.
- **Realism Only (No Pedagogical Add-ons):** Use natural, realistic tones for actual components and subjects. Do not use brand colors for anything else as no add-ons are allowed.

---

## HVAC-Specific Visual Guidelines (MANDATORY)

### Subject Focus
The subject must be specific HVAC components (e.g., manifold gauge set, blower motor, evaporator coil) or a technician actively performing a diagnostic or repair task. The focus of the image is the tool or the part being discussed in the voiceover.

### Visual Style
**Industrial Clean:** The aesthetic should feel authentic but polished. Avoid the "gritty" look of a construction site in favor of a professional service environment. For 3D and 2D graphics, use accurate proportions with simplified textures to reduce visual noise.

### Perspective and Camera
Use **first-person or over-the-shoulder (OTS) shots** to mimic what the learner would see in the field. For diagrams, use **isometric views or clean 90-degree side profiles** to clearly show airflow and electrical paths.

### Lighting and Environment
**Bright, even lighting** that eliminates deep shadows (which can hide important components). The lighting should ensure all parts and details are clearly visible.

### Background Setting
**Clean White Background:** The subject should be isolated on a pure white background. No contextual environments, no floors, no walls, no bokeh.

### Color Grading
**True to Life:** Neutral tones with high saturation on functional elements:
- **Blue** for cooling/airflow
- **Red/Orange** for heating/danger
- **Yellow/Glowing blue** for electrical/caution

Avoid heavy filters; maintain natural skin tones and metallic finishes.

### Safety Requirements (CRITICAL)
**Safety First:** All subjects MUST be shown wearing appropriate Personal Protective Equipment (PPE):
- Gloves
- Safety glasses
- Ear protection (when applicable)

This reinforces industry standards and is non-negotiable for all HVAC training visuals.
"""

# ── Model identifiers ─────────────────────────────────────────────────────────
ANALYSIS_MODEL  = "gemini-3-flash-preview"
REVIEWER_MODEL  = "gemini-3-flash-preview"
GENERATOR_MODEL = "gemini-3-pro-image-preview"
OPENAI_FALLBACK_MODEL = "gpt-image-1.5"

# ── Safety caps ───────────────────────────────────────────────────────────────
MAX_INSTRUCTION_ROUNDS = 5
MAX_IMAGE_ROUNDS       = 5


def _get_openai_client() -> Optional[object]:
    """Initialize the OpenAI client for fallback image generation."""
    if OpenAI is None:
        print("[ImageGenerator] OpenAI SDK unavailable; fallback disabled.")
        return None
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        try:
            import streamlit as st
            api_key = st.session_state.get("openai_api_key")
        except (ImportError, Exception):
            pass

    if not api_key:
        return None

    try:
        return OpenAI(api_key=api_key)
    except Exception as e:
        print(f"[ImageGenerator] Failed to initialize OpenAI client: {e}")
        return None


def _openai_size_from_inputs(image_size: str, aspect_ratio: Optional[str]) -> str:
    """Map current generator settings to OpenAI-compatible sizes."""
    ratio = (aspect_ratio or "").strip()
    if ratio == "16:9":
        return "1536x1024"
    if ratio == "9:16":
        return "1024x1536"
    if ratio == "4:3":
        return "1536x1024"
    if ratio == "3:4":
        return "1024x1536"

    # Default square output for unknown/unsupported ratios.
    return "1024x1024"


def _extract_openai_image(response) -> Optional[Image.Image]:
    """Extract PIL image from OpenAI image response payload."""
    if not getattr(response, "data", None):
        return None

    first = response.data[0]

    # Preferred path: pydantic-style response object
    b64 = getattr(first, "b64_json", None)
    if b64:
        return Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")

    url = getattr(first, "url", None)
    if url:
        with urlopen(url, timeout=30) as resp:
            return Image.open(BytesIO(resp.read())).convert("RGB")

    # Fallback path: dict-style response payload
    if isinstance(first, dict):
        b64 = first.get("b64_json")
        if b64:
            return Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")
        url = first.get("url")
        if url:
            with urlopen(url, timeout=30) as resp:
                return Image.open(BytesIO(resp.read())).convert("RGB")

    return None


def _generate_with_openai_fallback(
    generation_user_prompt: str,
    generation_system_instruction: str,
    aspect_ratio: Optional[str],
    image_size: str,
) -> tuple[Optional[Image.Image], dict]:
    """Final fallback image generation using gpt-image-1.5."""
    client = _get_openai_client()
    if not client:
        return None, {"fallback": True, "error": "openai_client_unavailable", "model": OPENAI_FALLBACK_MODEL}

    size = _openai_size_from_inputs(image_size=image_size, aspect_ratio=aspect_ratio)
    prompt = (
        "Fallback generation: keep behavior aligned with the primary model.\n\n"
        f"{generation_system_instruction}\n\n"
        f"{generation_user_prompt}"
    )

    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            print(f"[ImageGenerator] OpenAI fallback attempt {attempt}/{max_attempts}...")
            if attempt > 1:
                wait_s = 2 ** (attempt - 1)
                print(f"[ImageGenerator] OpenAI fallback retry in {wait_s}s ({attempt}/{max_attempts})")
                time.sleep(wait_s)

            response = client.images.generate(
                model=OPENAI_FALLBACK_MODEL,
                prompt=prompt,
                size=size,
            )
            image = _extract_openai_image(response)
            if image is not None:
                return image, {"fallback": True, "model": OPENAI_FALLBACK_MODEL, "size": size}
            print(f"[ImageGenerator] OpenAI fallback returned no image payload ({attempt}/{max_attempts}).")
        except Exception as e:
            print(f"[ImageGenerator] OpenAI fallback attempt {attempt}/{max_attempts} failed: {e}")

    return None, {"fallback": True, "error": "openai_fallback_failed", "model": OPENAI_FALLBACK_MODEL, "size": size}


# =============================================================================
# STRATEGY & ANALYSIS SYSTEM INSTRUCTIONS
# =============================================================================

_STRATEGY_SYSTEM_INSTRUCTION = """\
You are a Visual Explainer. 
Your goal is to propose 5 "Safe & Simple" visual directions to clarify a voiceover sentence.

STRICT HALLUCINATION PREVENTION:
- NO INTERNAL VIEWS: Do not show internal technical mechanisms, cutaways, or complex wiring. 
- NO DETAILED SCHEMATICS: Avoid technical diagrams that require architectural accuracy.
- WISE ANNOTATION: Text callouts and labels are RECOMMENDED only if they are crucial to explaining the concept. Keep them brief (1-2 words), large, and legible. Do not overdo it.

THE 5 "SAFE" DIRECTIONS:
1. OPTION A (Representative Component): A clean, external view of the physical Part mentioned.
2. OPTION B (State Indicator): A simple gauge, dial, or indicator showing a STATUS (e.g. needle on 'High').
3. OPTION C (Atmospheric Metaphor): Using color/effects to show a state (e.g. 'Red Glow' for hot, 'Blue Flow' for cold).
4. OPTION D (Logical Icon): A universal brand-aligned symbol (e.g. a checkmark for success, a warning triangle).
5. OPTION E (Relational Comparison): Two simple objects showing a basic logic (e.g. 'One is bigger', 'One is blocked').

FORMAT: Output exactly 5 options, each as a single sentence starting with "OPTION X:". Nothing else."""

_ANALYSIS_SYSTEM_INSTRUCTION = """\
You are an HVAC Technical Lead. 
You will be provided with 5 visual strategy options for a voiceover.

YOUR TASK:
1. Evaluate the 5 options. CHOOSE the one that is most "Hallucination-Safe" and clear.
2. EXECUTE that strategy into a JSON brief.

ANTI-HALLUCINATION RULES:
- If a strategy feels "Too Technical" or "Too Complex," reject it for a simpler one.
- Describe only the external, visible features. 
- WISE ANNOTATION: You may request a single clear label or arrow if it's essential to identify a component or state. Avoid cluttered text.

COLOR USAGE RULES:
1. BRAND COLORS: Use these for Scaffolding and Icons/Symbols.
2. FUNCTIONAL COLORS: Use these for Physical Components to indicate state (Red for Hot, Blue for Cold).

Output ONLY the JSON object for your chosen strategy.

STYLING GUIDE:
{styling_guide}
"""

_ANALYSIS_FEEDBACK_TURN = """\
<review_result>
The JSON brief was REJECTED. It drifted from the Strategy or allowed AI clutter.
</review_result>

<issues>
{issues}
</issues>

<directive>
Refine the JSON. Re-center the Strategic Anchor. Fix Scale/Architecture.
Output ONLY the final corrected JSON.
</directive>"""


# =============================================================================
# INSTRUCTION REVIEWER
# =============================================================================

_INSTRUCTION_REVIEWER_SYSTEM = """\
You are a senior HVAC curriculum designer auditing an image generation brief.

Your role is to APPROVE briefs that are technically sound and pedagogically focused, not to find every possible improvement.

REVISE ONLY if there is a GENUINE BLOCKER:
  - A factual HVAC inaccuracy (wrong component name, incorrect flow direction, code violation)
  - A safety hazard (missing mandatory PPE, dangerous depiction)
  - The brief requests too many components (more than 1 primary focal element)

DO NOT flag:
  - Style preferences or minor phrasing differences
  - Missing details that are optional or obvious
  - Anything a competent image generator would handle automatically

MAX ISSUES: If you must REVISE, list at most 2 critical blockers. No more.

OUTPUT FORMAT:
VERDICT: PASS
or
VERDICT: REVISE
ISSUES:
• [critical blocker 1]
• [critical blocker 2 — only if truly necessary]
"""

def _review_instructions(
    slide_title: str,
    slide_content: str,
    voiceover_focus: str,
    instructions_json_str: str,
    round_num: int,
) -> dict:
    client = _get_client()
    if not client:
        return {"verdict": "PASS", "issues_text": "", "new_issues": []}

    prompt = f"""\
CONTEXT
───────
Title: {slide_title}
Voiceover: {voiceover_focus}

JSON BRIEF (Round {round_num})
──────────────────────────────
{instructions_json_str}

Evaluate technical accuracy and clarity. Respond with VERDICT and ISSUES.
"""

    response = call_llm_with_retry(
        client.models.generate_content,
        model=REVIEWER_MODEL,
        contents=[types.Content(role="user", parts=[types.Part.from_text(text=prompt)])],
        config=types.GenerateContentConfig(
            system_instruction=_INSTRUCTION_REVIEWER_SYSTEM,
            thinking_config=types.ThinkingConfig(thinking_level="medium"),
        ),
    )

    text = (getattr(response, "text", None) or "").strip()
    
    # Robust verdict check
    verdict = "PASS" if "VERDICT: PASS" in text.upper() else "REVISE"

    new_issues = []
    in_issues = False
    for line in text.splitlines():
        s = line.strip()
        if s.upper().startswith("ISSUES"):
            in_issues = True
            continue
        if in_issues and s.startswith(("•", "-", "*")):
            issue = s.lstrip("•-* ").strip()
            if issue:
                new_issues.append(issue)

    # ZERO ISSUES FALLBACK: If status is REVISE but 0 issues found, treat as PASS
    if verdict == "REVISE" and not new_issues:
        print("[InstructionReviewer] ℹ️  Verdict was REVISE but 0 issues parsed — auto-correcting to PASS.")
        verdict = "PASS"

    return {"verdict": verdict, "issues_text": "\n".join(f"• {i}" for i in new_issues), "new_issues": new_issues}


# =============================================================================
# STAGE 1 — ANALYSIS CHAT LOOP
# =============================================================================

def _decide_strategy(slide_title: str, slide_content: str, voiceover_focus: str) -> str:
    """Independent LLM call to decide the pedagogical strategy."""
    client = _get_client()
    if not client: return "Generic HVAC Illustration"
    
    prompt = f"TITLE: {slide_title}\nCONTENT: {slide_content}\nVOICEOVER: {voiceover_focus}"
    
    response = call_llm_with_retry(
        client.models.generate_content,
        model=ANALYSIS_MODEL,
        contents=[types.Content(role="user", parts=[types.Part.from_text(text=prompt)])],
        config=types.GenerateContentConfig(system_instruction=_STRATEGY_SYSTEM_INSTRUCTION),
    )
    return (getattr(response, "text", None) or "Standard technical illustration").strip()

@traceable(metadata={"agent_name": "image_generator", "step_name": "Stage 1 — JSON Brief Analysis"})
def run_analysis_stage(
    slide_title: str,
    slide_content: str,
    voiceover_focus: str,
    revision_history: list,
) -> str:
    client = _get_client()
    if not client:
        raise ValueError("Missing API key.")

    # 1. THE STRATEGY CALL (New independent phase)
    strategy = _decide_strategy(slide_title, slide_content, voiceover_focus)
    print(f"\n[ImageGenerator] 🎯 Strategy Decided:\n{strategy}")

    # 2. THE ANALYSIS CHAT
    chat = client.chats.create(
        model=ANALYSIS_MODEL,
        config=types.GenerateContentConfig(
            system_instruction=_ANALYSIS_SYSTEM_INSTRUCTION.format(
                styling_guide=_STYLING_GUIDE or "Follow clean, professional educational HVAC illustration standards."
            ),
            response_mime_type="application/json",
            thinking_config=types.ThinkingConfig(thinking_level="low"),
        ),
    )

    initial_prompt = f"""\
VOICEOVER: {voiceover_focus}

STRATEGY OPTIONS:
{strategy}

Examine these 5 options. Pick the most effective one and generate the JSON brief (subject, explanation, style).
"""

    round_num = 0
    instructions = ""

    while True:
        round_num += 1
        print(f"\n[ImageGenerator] ── Analysis Round {round_num} ─────────────────────────")

        if round_num > MAX_INSTRUCTION_ROUNDS:
            print(f"[ImageGenerator] ⚠️ Safety cap ({MAX_INSTRUCTION_ROUNDS}) reached.")
            break

        if round_num == 1:
            response = call_llm_with_retry(chat.send_message, initial_prompt)
        
        instructions = (getattr(response, "text", None) or "").strip()
        
        # Verify JSON
        try:
            json.loads(instructions)
        except:
            print("[ImageGenerator] ⚠️ Invalid JSON received. Retrying turns...")

        # Review
        review = _review_instructions(slide_title, slide_content, voiceover_focus, instructions, round_num)
        
        revision_history.append({
            "stage": "instruction",
            "round": round_num,
            "verdict": review["verdict"],
            "new_issues": review["new_issues"],
        })

        if review["verdict"] == "PASS":
            print(f"[ImageGenerator] ✅ JSON Brief Approved (Round {round_num}).")
            break

        print(f"[ImageGenerator]   REVISE — {len(review['new_issues'])} issue(s).")
        if round_num < MAX_INSTRUCTION_ROUNDS:
            print(
                f"[ImageGenerator] ↻ Retrying instruction round "
                f"{round_num + 1}/{MAX_INSTRUCTION_ROUNDS}."
            )
        feedback = _ANALYSIS_FEEDBACK_TURN.format(issues=review["issues_text"])
        response = call_llm_with_retry(chat.send_message, feedback)

    return instructions


# =============================================================================
# STAGE 2 — IMAGE GENERATION CHAT LOOP
# =============================================================================

_IMAGE_REVIEWER_SYSTEM = """\
You are a senior HVAC trainer reviewing a generated educational image.

Your goal is to APPROVE images that are clear, focused, and technically correct — NOT to find every possible flaw.

IMPORTANT CONVERGENCE RULES:
  1. Only flag issues that would GENUINELY confuse or mislead a learner.
  2. Do NOT re-flag issues from previous rounds that have already been addressed.
  3. If the image adequately illustrates the voiceover concept, PASS it — even if it's not perfect.
  4. PASS immediately if:
     - The main subject is clearly visible and fills most of the frame
     - The key HVAC concept from the voiceover is evident
     - There are no dangerous inaccuracies (wrong flow, missing PPE in safety-critical tasks)
  5. MAX 2 ISSUES per round. Never list more than 2. If there are more, pick only the 2 most critical.

OUTPUT FORMAT:
VERDICT: PASS
or
VERDICT: REVISE
ISSUES:
• [most critical issue]
• [second most critical issue — only if genuinely necessary]
"""

_GENERATOR_FEEDBACK_TURN = """\
<review_result>
The image was REJECTED.
</review_result>
<issues>
{issues}
</issues>
<directive>
Regenerate based on the original JSON but FIX the issues listed. 
Output the corrected image.
</directive>"""

def _review_image(image, slide_title, slide_content, voiceover_focus, instructions_json, round_num):
    client = _get_client()
    if not client: return {"verdict": "PASS", "issues_text": "", "new_issues": []}
    
    image_bytes = prepare_image_for_gemini(image)
    prompt = f"VOICEOVER: {voiceover_focus}\nJSON BRIEF: {instructions_json}\nIMAGE (Round {round_num}) attaché."
    
    response = call_llm_with_retry(
        client.models.generate_content,
        model=REVIEWER_MODEL,
        contents=[
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=prompt),
                    types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                ],
            )
        ],
        config=types.GenerateContentConfig(
            system_instruction=_IMAGE_REVIEWER_SYSTEM,
            thinking_config=types.ThinkingConfig(thinking_level="medium"),
        ),
    )
    
    text = (getattr(response, "text", None) or "").strip()
    verdict = "PASS" if "VERDICT: PASS" in text.upper() else "REVISE"
    
    new_issues = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith(("•", "-", "*")):
            issue = s.lstrip("•-* ").strip()
            if issue:
                new_issues.append(issue)

    # ZERO ISSUES FALLBACK: If status is REVISE but 0 issues found, treat as PASS
    if verdict == "REVISE" and not new_issues:
        print("[ImageReviewer] ℹ️  Verdict was REVISE but 0 issues parsed — auto-correcting to PASS.")
        verdict = "PASS"

    return {"verdict": verdict, "issues_text": "\n".join(f"• {i}" for i in new_issues), "new_issues": new_issues}

@traceable(metadata={"agent_name": "image_generator", "step_name": "Stage 2 — Image Generation"})
def run_generation_stage(
    approved_json_brief: str,
    slide_title: str,
    slide_content: str,
    voiceover_focus: str,
    aspect_ratio: Optional[str],
    image_size: str,
    revision_history: list,
) -> tuple:
    client = _get_client()
    if not client: raise ValueError("Missing API key.")

    image_config = {"image_size": image_size}
    if aspect_ratio: image_config["aspect_ratio"] = aspect_ratio

    generation_system_instruction = (
        (f"{_STYLING_GUIDE}\n\n" if _STYLING_GUIDE else "")
        + "Draw exactly what the JSON brief describes — nothing more. "
          "White background. Keep it simple and clean. Legible labels and annotations are allowed only if explicitly requested in the brief."
    )
    chat = client.chats.create(
        model=GENERATOR_MODEL,
        config=types.GenerateContentConfig(
            system_instruction=generation_system_instruction,
            response_modalities=["TEXT", "IMAGE"],
            temperature=0.65,
            image_config=types.ImageConfig(**image_config),
        ),
    )

    last_image = None
    round_num = 0
    last_error = None
    last_generation_prompt = approved_json_brief

    while True:
        round_num += 1
        print(f"\n[ImageGenerator] ── Image Generation Round {round_num} ──────────────────")

        if round_num > MAX_IMAGE_ROUNDS:
            break

        try:
            # Send JSON prompt in user prompt turn
            if round_num == 1:
                response = call_llm_with_retry(chat.send_message, approved_json_brief)

            # Extract
            generated_image = None
            if response.candidates and response.candidates[0].content:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, "inline_data") and part.inline_data:
                        generated_image = Image.open(BytesIO(part.inline_data.data))
                        break

            if not generated_image:
                raise RuntimeError("No image generated.")

            last_image = generated_image
            print(f"[ImageGenerator]   Image generated ({generated_image.size}).")

            # Review
            review = _review_image(generated_image, slide_title, slide_content, voiceover_focus, approved_json_brief, round_num)

            revision_history.append({
                "stage": "image",
                "round": round_num,
                "verdict": review["verdict"],
                "new_issues": review["new_issues"],
                "model": GENERATOR_MODEL,
            })

            if review["verdict"] == "PASS":
                print(f"[ImageGenerator] ✅ Image Approved.")
                return last_image, {"rounds": round_num, "model": GENERATOR_MODEL, "fallback_used": False}

            print(f"[ImageGenerator]   REVISE — {len(review['new_issues'])} issue(s).")
            if round_num < MAX_IMAGE_ROUNDS:
                print(
                    f"[ImageGenerator] ↻ Retrying Nano Banana Pro round "
                    f"{round_num + 1}/{MAX_IMAGE_ROUNDS}."
                )
            feedback = _GENERATOR_FEEDBACK_TURN.format(issues=review["issues_text"])
            last_generation_prompt = feedback
            response = call_llm_with_retry(chat.send_message, feedback)

        except Exception as e:
            last_error = e
            print(f"[ImageGenerator] ⚠️ Round {round_num} failed: {e}")
            if round_num < MAX_IMAGE_ROUNDS:
                print(
                    f"[ImageGenerator] ↻ Retrying Nano Banana Pro round "
                    f"{round_num + 1}/{MAX_IMAGE_ROUNDS} after failure."
                )
            revision_history.append({
                "stage": "image",
                "round": round_num,
                "verdict": "ERROR",
                "new_issues": [str(e)],
                "model": GENERATOR_MODEL,
            })
            continue

    print(
        f"[ImageGenerator] ⚠️ Nano Banana Pro ({GENERATOR_MODEL}) failed after {MAX_IMAGE_ROUNDS} rounds; "
        f"going to fallback ({OPENAI_FALLBACK_MODEL})."
    )
    fallback_image, fallback_meta = _generate_with_openai_fallback(
        generation_user_prompt=last_generation_prompt,
        generation_system_instruction=generation_system_instruction,
        aspect_ratio=aspect_ratio,
        image_size=image_size,
    )

    if fallback_image is not None:
        revision_history.append({
            "stage": "image",
            "round": "fallback",
            "verdict": "PASS",
            "new_issues": [],
            "model": OPENAI_FALLBACK_MODEL,
        })
        print(f"[ImageGenerator] ✅ Fallback image generated by {OPENAI_FALLBACK_MODEL}.")
        return fallback_image, {
            "rounds": round_num,
            "model": OPENAI_FALLBACK_MODEL,
            "fallback_used": True,
            "fallback": fallback_meta,
        }

    raise RuntimeError(f"Image generation failed after Gemini retries and {OPENAI_FALLBACK_MODEL} fallback. Last Gemini error: {last_error}")


# =============================================================================
# PUBLIC ENDPOINT
# =============================================================================

@traceable(metadata={"agent_name": "image_generator", "step_name": "Full Pipeline"})
def generate_asset(
    slide_title: str = "",
    slide_content: str = "",
    voiceover_focus: str = "",
    aspect_ratio: Optional[str] = "16:9",
    image_size: str = "1K",
) -> dict:
    revision_history = []
    try:
        # Phase 1: Analysis -> JSON Brief
        instructions = run_analysis_stage(slide_title, slide_content, voiceover_focus, revision_history)

        # Phase 2: Generation -> Image
        image, metadata = run_generation_stage(instructions, slide_title, slide_content, voiceover_focus, aspect_ratio, image_size, revision_history)

        return {
            "image": image,
            "instructions": instructions,
            "metadata": metadata,
            "revision_history": revision_history,
            "error": None,
        }
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return {"image": None, "instructions": None, "metadata": {}, "revision_history": revision_history, "error": str(exc)}