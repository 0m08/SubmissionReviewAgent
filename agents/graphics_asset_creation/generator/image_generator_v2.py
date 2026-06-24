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

# from agents.graphics_asset_creation.generator import model_config

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
                print(f"  ⚠️ LLM call failed: {e}. Retrying in {wait_time}s… ({attempt+1}/{max_retries})")
                time.sleep(wait_time)
            else:
                print(f"  ❌ LLM call failed after {max_retries} attempts: {e}")
    raise last_exc


def prepare_image_for_gemini(image: Image.Image, max_dimension: int = 2048) -> bytes:
    """Convert PIL Image to bytes for Gemini API with size optimisation."""
    buffered = BytesIO()
    save_img = image.copy()
    width, height = save_img.size
    if width > max_dimension or height > max_dimension:
        ratio = min(max_dimension / width, max_dimension / height)
        save_img = save_img.resize(
            (int(width * ratio), int(height * ratio)), Image.Resampling.LANCZOS
        )
    if save_img.mode != "RGB":
        save_img = save_img.convert("RGB")
    save_img.save(buffered, format="JPEG", quality=90)
    return buffered.getvalue()


# =============================================================================
# STYLING GUIDE  (contradictions resolved; "simple" language removed)
# =============================================================================

def _load_styling_guide() -> str:
    """Load styling guide from markdown file."""
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "..", "guide", "styling_guide.md")
        config_path = os.path.normpath(config_path)
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return f.read()
    except Exception as e:
        print(f"[ImageGenerator] Error loading styling guide from {config_path}: {e}")
    return "Follow professional, high-fidelity HVAC design guidelines."


_STYLING_GUIDE = _load_styling_guide()


# =============================================================================
# MODEL IDENTIFIERS & CAPS  (managed centrally in model_config.py)
# =============================================================================

ANALYSIS_MODEL  = "gemini-3-flash-preview"
REVIEWER_MODEL  = "gemini-3-flash-preview"
GENERATOR_MODEL = "gemini-3-pro-image-preview"
OPENAI_FALLBACK_MODEL = "gpt-image-1.5"

MAX_IMAGE_ROUNDS = 5


# =============================================================================
# OPENROUTER & OPENAI FALLBACK UTILITIES
# =============================================================================


def _extract_all_drive_ids_from_text(text: str) -> list[str]:
    if not text:
        return []
    print("[ImageGenerator] Scanning prompt for all Google Drive links...")
    import re
    # Patterns to match any drive.google.com link
    patterns = [
        r"https://drive\.google\.com/file/d/([a-zA-Z0-9_-]+)",
        r"https://drive\.google\.com/open\?id=([a-zA-Z0-9_-]+)",
        r"https://drive\.google\.com/uc\?id=([a-zA-Z0-9_-]+)",
        r"drive\.google\.com/file/d/([a-zA-Z0-9_-]+)",
        r"drive\.google\.com/open\?id=([a-zA-Z0-9_-]+)",
        r"drive\.google\.com/uc\?id=([a-zA-Z0-9_-]+)",
        # Support drive.usercontent.google.com download links
        r"drive\.usercontent\.google\.com/download\?id=([a-zA-Z0-9_-]+)",
        r"drive\.usercontent\.google\.com/uc\?id=([a-zA-Z0-9_-]+)",
        r"drive\.usercontent\.google\.com/open\?id=([a-zA-Z0-9_-]+)",
    ]
    seen = set()
    unique_ids = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            drive_id = match.group(1)
            if drive_id not in seen:
                seen.add(drive_id)
                unique_ids.append(drive_id)
    if unique_ids:
        print(f"[ImageGenerator] Found unique Google Drive IDs: {unique_ids}")
    else:
        print("[ImageGenerator] No Google Drive links found in user prompt.")
    return unique_ids


def _download_image_from_drive_as_b64(file_id: str) -> Optional[str]:
    import tempfile
    try:
        from services.drive_service import get_authenticated_drive_client
        print(f"[ImageGenerator] Reaching Google Drive to fetch file: {file_id}")
        drive = get_authenticated_drive_client()
        if not drive:
            print("[ImageGenerator] Could not build authenticated Drive client. Check OAUTH_CLIENT_ID / GOOGLE_OAUTH_REFRESH_TOKEN.")
            return None
        
        file_obj = drive.CreateFile({'id': file_id})
        file_obj.FetchMetadata()
        title = file_obj.get('title', 'untitled')
        mime_type = file_obj.get('mimeType', 'unknown')
        print(f"[ImageGenerator] Target file metadata retrieved. Title: '{title}', Mime-Type: '{mime_type}'")
        
        # Download the file content to a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name
        try:
            file_obj.GetContentFile(tmp_path)
            file_size = os.path.getsize(tmp_path)
            print(f"[ImageGenerator] Successfully downloaded file '{title}' ({file_size} bytes)")
            with open(tmp_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    except Exception as e:
        print(f"[ImageGenerator] Error downloading file {file_id} from Drive: {e}")
        import traceback
        traceback.print_exc()
        return None

def _extract_all_web_image_urls(text: str) -> list[str]:
    """Extract general web image URLs from a text prompt, ignoring Drive, Docs, and YouTube links."""
    if not text:
        return []
    import re
    urls = re.findall(r'https?://[^\s<>"]+|www\.[^\s<>"]+', text)
    image_urls = []
    for url in urls:
        if any(d in url for d in ["drive.google.com", "drive.usercontent.google.com", "docs.google.com/document", "docs.google.com/spreadsheets"]):
            continue
        if "youtube.com" in url or "youtu.be" in url:
            continue
        if url not in image_urls:
            image_urls.append(url)
    if image_urls:
        print(f"[ImageGenerator] Found candidate web image URLs: {image_urls}")
    return image_urls

def _download_web_image_as_b64(url: str) -> Optional[str]:
    """Download a general web image from a URL and return it as a base64-encoded string."""
    try:
        import requests
        import base64
        print(f"[ImageGenerator] Downloading web image from: {url}")
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
        response = requests.get(url, timeout=15, headers=headers)
        if response.status_code == 200:
            content_type = response.headers.get("content-type", "").lower()
            if "image" in content_type or any(ext in url.lower() for ext in [".png", ".jpg", ".jpeg", ".webp"]):
                encoded = base64.b64encode(response.content).decode("utf-8")
                print(f"[ImageGenerator] Successfully downloaded web image from: {url} (Content-Type: {content_type})")
                return encoded
            else:
                print(f"[ImageGenerator] WARNING: Web URL did not return an image content-type: {content_type}")
        else:
            print(f"[ImageGenerator] WARNING: Failed to download web image, HTTP {response.status_code}")
    except Exception as e:
        print(f"[ImageGenerator] ERROR downloading web image: {e}")
    return None

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
# PROMPT STRINGS  (all oversimplification guardrails removed)
# =============================================================================


_REQUIREMENTS_EXTRACTION_SYSTEM_INSTRUCTION = """\
You are an HVAC Technical Visual Director.
Your job is to analyze the slide context, voiceover, and optional human strategy to extract the core visual requirements.

Specifically, identify:
1. The CENTRAL VISUAL IDEA: The primary conceptual message that the image must convey.
2. THINGS TO BE INCLUDED: The physical components, equipment, context, or subjects that must appear in the image.
3. VISUAL REQUIREMENTS CHECKLIST: A checklist of concrete visual/technical rules to ensure the central idea is faithfully depicted.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ANNOTATION & LABEL RULES (STRICT CAP)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Do NOT include any requirements or instructions for annotations, text labels, callout boxes, or pointing arrows in the visual checklist unless they are explicitly requested in the HUMAN STRATEGY.
- Default to ZERO annotations (clean visual) to prevent visual clutter and confusion.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BACKGROUND & ENVIRONMENT RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- If the background/environment setting is not clearly defined or understandable from the slide title, content, voiceover, or human strategy, you MUST explicitly specify a "solid pure white background" in the requirements checklist.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HUMAN STRATEGY — PRIMARY DIRECTIVE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
If a "HUMAN STRATEGY" is provided in the input, it is the absolute source of truth.
- The Central Visual Idea must directly capture the human strategy.
- The checklist must be designed specifically to enforce that the human strategy is executed perfectly and not drifted from.

OUTPUT FORMAT:
Return a JSON object with this exact structure:
{{
  "central_idea": "...",
  "things_to_include": ["...", "..."],
  "checklist": [
    "Include...",
    "Ensure...",
    "Show..."
  ]
}}
Output ONLY the JSON object. Do not include markdown code block formatting (like ```json) or any extra explanation.

STYLING GUIDE:
{styling_guide}
"""


# ── Analysis ─────────────────────────────────────────────────────────────────
# OLD: instructed agent to choose "most Hallucination-Safe", reject complex
#      options, and limit annotations to 1 label/arrow.
# NEW: instructs agent to choose "most educationally effective", describe
#      complex visuals precisely, and use as many annotations as needed.

_ANALYSIS_SYSTEM_INSTRUCTION = """\
You are an HVAC Senior Curriculum Designer.
You will receive the extracted visual requirements (central idea, things to include, and a checklist) for a voiceover.
Produce a comprehensive JSON brief to guide the image generator model.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HUMAN STRATEGY — PRIMARY RULE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
If a "HUMAN STRATEGY" is present in the input:
  - You MUST design the JSON brief around it as the central subject and visual goal.
  - The checklist and composition notes must enforce the human strategy.

IF NO HUMAN STRATEGY IS PROVIDED:
  - Favour visual richness, realistic context, and scene complexity when the topic is macro or procedural.
  - Do not default to a closeup if the slide describes a process, environment, or workflow.

ACCURACY GATE:
- HVAC factual error: wrong component name, incorrect flow direction, physically impossible configuration.

ANNOTATION RULES:
- Do NOT add annotations unless they are absolutely necessary to explain the voiceover concept or explicitly requested in the Human Strategy.
- Default to 0 annotations for realistic, scenic, environmental, or procedural task visuals (keep them completely free of labels and arrows).
- If annotations are necessary, limit them strictly to the most critical details (maximum 2-3 labels or arrows total).
- Avoid requesting arrows unless they represent direct physical flow. No abstract, decorative, or floaty arrows.

JSON BRIEF SCHEMA (include all fields, especially the visual_requirements_checklist):
{{
  "visual_type": "component_portrait | cutaway | system_diagram | procedural_scene | comparison",
  "subject": "...",
  "key_components": ["...", "..."],
  "annotations": ["label: description", "..."],
  "perspective": "natural_view | first_person | eye_level | cutaway | standard_diagram",
  "background": "contextual_setting | room_interior | outdoor_setting | workshop | equipment_room | white",
  "functional_colors": {{"element": "color_reason"}},
  "technician_present": true | false,
  "ppe_required": true | false,
  "composition_notes": "...",
  "visual_requirements_checklist": ["requirement 1", "requirement 2", "..."]
}}

Output ONLY the JSON object.

STYLING GUIDE:
{styling_guide}
"""


# ── Analysis feedback turn ────────────────────────────────────────────────────
_ANALYSIS_FEEDBACK_TURN = """\
<review_result>
The JSON brief was REJECTED due to an HVAC factual error or safety omission.
</review_result>

<issues>
{issues}
</issues>

<directive>
Correct the factual errors listed above. Keep the visual richness and
complexity of the brief intact — do not simplify. Output ONLY the corrected JSON.
</directive>"""


# ── Instruction reviewer ──────────────────────────────────────────────────────
# OLD: rejected briefs with more than 1 focal element (forced icon-level output).
# NEW: only rejects for HVAC factual errors and genuine safety omissions.

_INSTRUCTION_REVIEWER_SYSTEM = """\
You are a senior HVAC curriculum designer auditing an image generation brief.

APPROVE the brief unless there is a GENUINE BLOCKER:
  - A factual HVAC inaccuracy (wrong component name, incorrect refrigerant flow
    direction, code violation, physically impossible configuration).
  - A safety hazard: a technician shown WITHOUT mandatory PPE (gloves + safety
    glasses) while performing a hands-on task.

DO NOT flag:
  - Visual complexity or the number of components shown.
  - Style preferences or minor phrasing.
  - Details an image model would handle automatically.
  - Anything that is merely "ambitious" — richness is the goal.

If you must REVISE, list at most 2 critical blockers. No more.

OUTPUT FORMAT:
VERDICT: PASS
or
VERDICT: REVISE
ISSUES:
• [critical blocker 1]
• [critical blocker 2 — only if truly necessary]
"""


# ── Image reviewer ────────────────────────────────────────────────────────────
# OLD: passed if "main subject fills most of the frame" — a very low bar.
# NEW: checks for educational completeness, annotation accuracy, and PPE.

_IMAGE_REVIEWER_SYSTEM = """\
You are a senior HVAC trainer reviewing a generated educational image.

PASS the image if ALL of the following are true:
  1. The primary subject clearly matches the voiceover concept.
  2. Annotations, labels, and arrows (if present) are correct, legible, point to the correct components, and follow the styling guidelines exactly:
     - The Box: A solid white rectangle with soft, rounded corners. It must be small in size, and if there are multiple boxes, they must all be of the exact same size.
     - The Border: A solid orange (#F05523) outline around the edges of the white box.
     - The Text: Plain, black text centered inside the box.
     - The Arrow (Connector): A straight orange (#F05523) arrow attached to the side of the box, pointing directly at the subject. The arrow must match the color and thickness of the box's border.
     - Physical equipment MUST use realistic/authentic colors and NOT orange/brand colors.
  3. There are no redundant, excessive, or irrelevant annotations/arrows that clutter the image. Realistic, scenic, environmental, or procedural task visuals should NOT contain arbitrary annotations/labels unless they are highly relevant or explicitly requested.
  4. If a technician is shown performing a hands-on task, PPE is visible.
  5. The image does not contain dangerous inaccuracies (wrong flow direction, wrong pressure gauge reading, incorrect component labels, or arrows pointing in physically impossible directions).
  6. The perspective and camera alignment are straight, level, and undistorted. There are no skewed lines, tilted horizons, wide-angle lens warping, or deformed geometries that make the equipment look warped, twisted, or distorted.

REVISE if any of those conditions fail.
Do NOT re-flag issues from previous rounds that have been addressed.
MAX 2 ISSUES per round — pick only the most critical.

OUTPUT FORMAT:
VERDICT: PASS
or
VERDICT: REVISE
ISSUES:
• [most critical issue]
• [second most critical — only if genuinely necessary]
"""

_GENERATOR_FEEDBACK_TURN = """
<review_result>
The image was REJECTED.
</review_result>
<issues>
{issues}
</issues>
<directive>
Regenerate based on the original JSON brief. Fix only the listed issues.
Keep the same level of detail and complexity — do not simplify the image.
Output the corrected image.
</directive>"""


# =============================================================================
# STAGE 1 — STRATEGY + ANALYSIS CHAT LOOP
# =============================================================================

def _extract_requirements(slide_title: str, slide_content: str, voiceover_focus: str, human_strategy: Optional[str] = None) -> str:
    """Independent LLM call to extract central ideas, included things, and visual requirements checklist."""
    client = _get_client()
    if not client:
        return "{}"
    prompt = f"TITLE: {slide_title}\nCONTENT: {slide_content}\nVOICEOVER: {voiceover_focus}"
    if human_strategy:
        prompt += f"\nHUMAN STRATEGY: {human_strategy}"
    response = call_llm_with_retry(
        client.models.generate_content,
        model=ANALYSIS_MODEL,
        contents=[types.Content(role="user", parts=[types.Part.from_text(text=prompt)])],
        config=types.GenerateContentConfig(
            system_instruction=_REQUIREMENTS_EXTRACTION_SYSTEM_INSTRUCTION.format(styling_guide=_STYLING_GUIDE),
            response_mime_type="application/json",
        ),
    )
    return (getattr(response, "text", None) or "{}").strip()


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

Check for HVAC factual errors and PPE omissions only. Respond with VERDICT and ISSUES.
"""
    response = call_llm_with_retry(
        client.models.generate_content,
        model=REVIEWER_MODEL,
        contents=[types.Content(role="user", parts=[types.Part.from_text(text=prompt)])],
        config=types.GenerateContentConfig(
            system_instruction=_INSTRUCTION_REVIEWER_SYSTEM,
            thinking_config=types.ThinkingConfig(thinking_level="low"),
        ),
    )

    text = (getattr(response, "text", None) or "").strip()
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

    if verdict == "REVISE" and not new_issues:
        print("[InstructionReviewer] ℹ️  REVISE with 0 issues — auto-correcting to PASS.")
        verdict = "PASS"

    return {"verdict": verdict, "issues_text": "\n".join(f"• {i}" for i in new_issues), "new_issues": new_issues}


@traceable(metadata={"agent_name": "image_generator_v2", "step_name": "Stage 1 — JSON Brief Analysis"})
def run_analysis_stage(
    slide_title: str,
    slide_content: str,
    voiceover_focus: str,
    revision_history: list,
    human_strategy: Optional[str] = None,
) -> str:
    client = _get_client()
    if not client:
        raise ValueError("Missing GOOGLE_API_KEY.")

    requirements_json = _extract_requirements(slide_title, slide_content, voiceover_focus, human_strategy)
    print(f"\n[ImageGenerator] 🎯 Extracted Requirements:\n{requirements_json}")
    return requirements_json


# =============================================================================
# STAGE 2 — IMAGE GENERATION CHAT LOOP
# =============================================================================

def _review_image(image, slide_title, slide_content, voiceover_focus, instructions_json, round_num, human_strategy=None):
    client = _get_client()
    if not client:
        return {"verdict": "PASS", "issues_text": "", "new_issues": []}

    image_bytes = prepare_image_for_gemini(image)
    prompt = (
        f"VOICEOVER: {voiceover_focus}\n"
        f"JSON BRIEF: {instructions_json}\n"
    )
    if human_strategy:
        prompt += f"HUMAN STRATEGY OVERRIDES: {human_strategy}\n"
    prompt += f"IMAGE (Round {round_num}) attached. Evaluate against all PASS criteria. Note: If the Human Strategy explicitly requests omitting elements (e.g., 'no text', 'no labels', 'no captions', 'no arrows'), do NOT flag their absence as an issue. Those omission instructions take absolute priority."
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

    if verdict == "REVISE" and not new_issues:
        print("[ImageReviewer] ℹ️  REVISE with 0 issues — auto-correcting to PASS.")
        verdict = "PASS"

    return {"verdict": verdict, "issues_text": "\n".join(f"• {i}" for i in new_issues), "new_issues": new_issues}


@traceable(metadata={"agent_name": "image_generator_v2", "step_name": "Stage 2 — Image Generation"})
def run_generation_stage(
    approved_json_brief: str,
    slide_title: str,
    slide_content: str,
    voiceover_focus: str,
    aspect_ratio: Optional[str],
    image_size: str,
    revision_history: list,
    human_strategy: Optional[str] = None,
    reference_url: Optional[str] = None,
) -> tuple:
    client = _get_client()
    if not client:
        raise ValueError("Missing GOOGLE_API_KEY.")

    image_config = {"image_size": image_size}
    if aspect_ratio:
        image_config["aspect_ratio"] = aspect_ratio

    checklist_str = ""
    try:
        brief_data = json.loads(approved_json_brief)
        checklist = brief_data.get("visual_requirements_checklist", [])
        if checklist and isinstance(checklist, list):
            checklist_str = "\n\nCENTRAL IDEA REQUIREMENTS CHECKLIST (MANDATORY):\n"
            for item in checklist:
                checklist_str += f"- [ ] {item}\n"
            checklist_str += "Verify that the generated image perfectly satisfies every item in the checklist above to ensure the central idea is followed."
    except Exception as e:
        print(f"[ImageGenerator] Note: could not parse visual checklist: {e}")

    # Build reference image inputs
    combined_brief = f"{approved_json_brief or ''}\n{voiceover_focus or ''}\n{human_strategy or ''}\n{slide_content or ''}\n{reference_url or ''}"
    
    # Extract and download Google Drive references
    drive_ids = _extract_all_drive_ids_from_text(combined_brief)
    b64_images = []
    for d_id in drive_ids:
        b64_img = _download_image_from_drive_as_b64(d_id)
        if b64_img:
            b64_images.append(b64_img)

    # Extract and download general web image references
    web_urls = _extract_all_web_image_urls(combined_brief)
    for url in web_urls:
        b64_img = _download_web_image_as_b64(url)
        if b64_img:
            b64_images.append(b64_img)

    image_parts = []
    for b64 in b64_images:
        try:
            img_bytes = base64.b64decode(b64)
            image_parts.append(types.Part.from_bytes(data=img_bytes, mime_type="image/png"))
        except Exception as e:
            print(f"[ImageGenerator] Error decoding reference image: {e}")

    generation_system_instruction = (
        "You are a professional technical visual creator.\n"
        "Your task is to render a high-quality educational HVAC image based on the user's prompt.\n"
        "You must strictly follow the visual style, technical specifications, safety requirements, and annotation design rules defined in the STYLING GUIDE below.\n\n"
        f"STYLING GUIDE:\n{_STYLING_GUIDE}"
    )

    if checklist_str:
        generation_system_instruction += checklist_str

    if human_strategy:
        generation_system_instruction += (
            f"\n\nCRITICAL DIRECTIVE OVERRIDES (from Human Strategy):\n"
            f"{human_strategy}\n"
            f"If the Human Strategy above explicitly requests omitting certain elements (e.g. 'no text', 'no labels', 'no captions', 'no annotations', 'no arrows'), "
            f"these constraints take absolute priority over all other guidelines. Do NOT render any of those omitted items under any circumstance."
        )

    # Image Editing Mode for 1 reference image
    if len(b64_images) == 1:
        print("[ImageGenerator] 1 reference image attached. Enabling Image Editing Mode.")
        generation_system_instruction += (
            "\n\nCRITICAL DIRECTIVE FOR IMAGE REFERENCE:\n"
            "An input/reference image has been attached to the user message. "
            "You MUST treat this attached image as the base template. Edit and modify this specific "
            "image according to the instructions in the prompt. Do NOT generate a brand new image from scratch. "
            "Instead, perform the edits (adding, removing, or changing elements) while maintaining the original background, "
            "layout, and details of the reference image.\n"
            "IMPORTANT: Make ONLY the edits explicitly requested by the user. Do NOT add any extra labels, annotations, "
            "arrows, text boxes, overlays, or decorative artifacts unless they are directly and specifically requested in the user's feedback. "
            "Preserve all other parts of the reference image exactly as they are."
        )
    elif len(b64_images) > 1:
        print(f"[ImageGenerator] {len(b64_images)} reference images attached. Analyzing user instructions to determine combination style.")
        generation_system_instruction += (
            "\n\nCRITICAL DIRECTIVE FOR MULTIPLE IMAGE REFERENCES:\n"
            "Multiple reference images have been attached to this request. "
            "You MUST carefully analyze the user's feedback/instructions to determine how to utilize them:\n"
            "1. If the user explicitly asks for a comparison, before/after, side-by-side, or split-screen layout, "
            "render the output as a clean split-screen collage according to the STYLING GUIDE collage rules.\n"
            "2. If the user describes a single cohesive scene or diagram that combines elements, components, or styles "
            "from the reference images (e.g. 'show unit A connected to unit B', 'put technician from image A next to unit in image B'), "
            "do NOT create a collage/split-screen. Instead, generate a single unified, cohesive image integrating "
            "the requested elements into a single scene.\n"
            "3. If the user does not specify a layout, default to generating a single cohesive scene that incorporates "
            "the referenced elements, rather than a split-screen, unless a comparison is clearly implied."
        )

    user_prompt = f"EXTRACTED REQUIREMENTS:\n{approved_json_brief}"
    if human_strategy:
        user_prompt += f"\n\nHUMAN STRATEGY:\n{human_strategy}"

    last_image = None
    last_error = None

    try:
        print(f"\n[ImageGenerator] Sending request to Gemini ({GENERATOR_MODEL}) for image generation...")
        initial_message = [user_prompt] + image_parts
        response = call_llm_with_retry(
            client.models.generate_content,
            model=GENERATOR_MODEL,
            contents=initial_message,
            config=types.GenerateContentConfig(
                system_instruction=generation_system_instruction,
                response_modalities=["TEXT", "IMAGE"],
                temperature=0.65,
                image_config=types.ImageConfig(**image_config),
            ),
        )

        generated_image = None
        if response.candidates and response.candidates[0].content:
            for part in response.candidates[0].content.parts:
                if hasattr(part, "inline_data") and part.inline_data:
                    generated_image = Image.open(BytesIO(part.inline_data.data))
                    break

        if not generated_image:
            raise RuntimeError("No image generated in Gemini response.")

        print(f"[ImageGenerator] ✅ Image generated successfully via Gemini ({generated_image.size}).")
        
        revision_history.append({
            "stage": "image",
            "round": 1,
            "verdict": "PASS",
            "new_issues": [],
            "model": GENERATOR_MODEL,
        })
        
        return generated_image, {"rounds": 1, "model": GENERATOR_MODEL, "fallback_used": False}

    except Exception as e:
        last_error = e
        print(f"[ImageGenerator] ⚠️ Gemini generation failed: {e}")
        revision_history.append({
            "stage": "image",
            "round": 1,
            "verdict": "ERROR",
            "new_issues": [str(e)],
            "model": GENERATOR_MODEL,
        })

    # Fallback to OpenAI if Gemini fails
    print(f"[ImageGenerator] ⚠️ Gemini failed; calling fallback model ({OPENAI_FALLBACK_MODEL})...")
    fallback_image, fallback_meta = _generate_with_openai_fallback(
        generation_user_prompt=user_prompt,
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
            "rounds": 1,
            "model": OPENAI_FALLBACK_MODEL,
            "fallback_used": True,
            "fallback": fallback_meta,
        }

    raise RuntimeError(
        f"[ImageGenerator] ❌ Image generation failed after Gemini attempt and {OPENAI_FALLBACK_MODEL} fallback. "
        f"Last Gemini error: {last_error}"
    )


# =============================================================================
# PUBLIC ENDPOINT  (identical signature to image_generator.generate_asset)
# =============================================================================

@traceable(metadata={"agent_name": "image_generator_v2", "step_name": "Full Pipeline"})
def generate_asset(
    slide_title: str = "",
    slide_content: str = "",
    voiceover_focus: str = "",
    aspect_ratio: Optional[str] = "16:9",
    image_size: str = "1K",
    human_strategy: Optional[str] = None,
    reference_url: Optional[str] = None,
) -> dict:
    revision_history = []
    try:
        instructions = run_analysis_stage(
            slide_title, slide_content, voiceover_focus, revision_history, human_strategy
        )
        image, metadata = run_generation_stage(
            instructions, slide_title, slide_content,
            voiceover_focus, aspect_ratio, image_size, revision_history, human_strategy, reference_url
        )
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
        return {
            "image": None,
            "instructions": None,
            "metadata": {},
            "revision_history": revision_history,
            "error": str(exc),
        }
