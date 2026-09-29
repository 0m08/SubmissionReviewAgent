import os
import json
import base64
import io
from typing import List, Optional, Tuple
from PIL import Image

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None


try:
    from langsmith import traceable
except ImportError:
    def traceable(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


def _pil_to_base64_url(img: Image.Image) -> str:
    """Convert PIL Image to base64 data URL."""
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{img_str}"


@traceable(name="Agent 1: Image Description Generator")
def generate_image_description(
    prompt: str,
    images: Optional[List[Image.Image]] = None,
    model_choice: str = "gpt-5.6-luna",
) -> str:
    """
    Agent 1: Image Description Generator.
    Takes an HVAC image asset and targeted prompt to generate a concise technical description.
    Supports GPT-5.6 Luna / GPT-4o and Gemini 3.7 Flash / Gemini 3.5 Flash.
    """
    images = images or []
    if not images and "extracted webpage content" not in prompt.lower():
        return "⚠️ No images provided for identification."

    is_gemini = "gemini" in model_choice.lower()

    if is_gemini:
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            return "⚠️ GEMINI_API_KEY or GOOGLE_API_KEY is missing. Please set it in your environment."
        if not genai:
            return "⚠️ google-genai package is not installed."

        try:
            client = genai.Client(api_key=api_key)
            contents = [prompt]
            contents.extend(images)

            # Map choice to model ID
            gemini_model = "gemini-3.7-flash" if "3.7" in model_choice else "gemini-3.5-flash"

            response = client.models.generate_content(
                model=gemini_model,
                contents=contents,
            )
            return (response.text or "No text response returned.").strip()
        except Exception as err:
            return f"❌ Gemini identification error: {err}"

    else:
        # GPT Models (e.g. gpt-5.6-luna, gpt-4o)
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return "⚠️ OPENAI_API_KEY is missing. Please set it in your environment."
        if not OpenAI:
            return "⚠️ openai package is not installed."

        try:
            client = OpenAI(api_key=api_key)
            user_content = [{"type": "text", "text": prompt}]
            for img in images:
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": _pil_to_base64_url(img)}
                })

            messages = [
                {"role": "user", "content": user_content}
            ]

            target_model = "gpt-5.6-luna" if "luna" in model_choice.lower() else "gpt-4o"
            kwargs = {"model": target_model, "messages": messages}
            if target_model == "gpt-5.6-luna":
                kwargs["reasoning_effort"] = "high"

            response = client.chat.completions.create(**kwargs)
            return (response.choices[0].message.content or "No text response returned.").strip()
        except Exception as err:
            return f"❌ GPT identification error: {err}"


@traceable(name="Agent 2: Scoring Judge")
def score_description_match(
    ai_description: str,
    human_description: str,
    model_choice: str = "gemini-3.7-flash",
) -> Tuple[str, str]:
    """
    Agent 2: Scoring Agent (Automated Evaluator).
    Compares the AI-generated image description against the ground-truth human description
    to issue a deterministic Yes / No match score.
    Powered by Gemini 3.7 Flash.
    Returns: (match_score, rationale) where match_score is 'Yes' or 'No'.
    """
    if not ai_description or not ai_description.strip() or ai_description.startswith("⚠️") or ai_description.startswith("❌"):
        return "No", "AI description is missing or contains an error."

    if not human_description or not human_description.strip():
        return "No", "Human ground-truth description is missing."

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return "No", "GEMINI_API_KEY or GOOGLE_API_KEY is missing."

    eval_prompt = f"""You are an automated QA judge evaluating whether an AI-generated image description matches a human ground-truth description.

Your core objective is to check if both descriptions carry the SAME CORE IDEA and technical meaning regarding the HVAC asset, component, or tool.

Rules:
1. Focus on whether the fundamental idea, technical object, component, or tool being described is essentially the same in both descriptions.
2. Allow variations in sentence structure, word choice, level of detail, or style.
3. If both descriptions convey the same core idea and describe the same primary HVAC component/asset, issue "Yes".
4. If the AI description describes a completely different component/tool, contradicts the human description, or fails to capture the main idea, issue "No".

Human Ground-Truth Description:
\"\"\"
{human_description.strip()}
\"\"\"

AI-Generated Model Description:
\"\"\"
{ai_description.strip()}
\"\"\"

Return strictly valid JSON with this format:
{{
  "match": "Yes" or "No",
  "rationale": "Brief 1-2 sentence explanation of why they carry or do not carry the same idea."
}}
"""

    if genai:
        try:
            client = genai.Client(api_key=api_key)
            gemini_model = "gemini-3.7-flash" if "3.7" in model_choice else "gemini-3.5-flash"
            response = client.models.generate_content(
                model=gemini_model,
                contents=[eval_prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                ) if types else None
            )
            raw_text = response.text or ""
            data = json.loads(raw_text)
            match_str = str(data.get("match", "No")).strip()
            rationale = str(data.get("rationale", "")).strip()

            # Ensure strict Yes or No output
            cleaned_match = "Yes" if match_str.lower() in ["yes", "true", "pass"] else "No"
            return cleaned_match, rationale
        except Exception as err:
            # Fallback text parsing if JSON parse failed
            print(f"[SCORING AGENT ERROR] {err}")

    # Fallback to OpenAI if Gemini client fails or unavailable
    if OpenAI and os.environ.get("OPENAI_API_KEY"):
        try:
            client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": eval_prompt}],
                response_format={"type": "json_object"},
            )
            data = json.loads(response.choices[0].message.content or "{}")
            match_str = str(data.get("match", "No")).strip()
            rationale = str(data.get("rationale", "")).strip()
            cleaned_match = "Yes" if match_str.lower() in ["yes", "true", "pass"] else "No"
            return cleaned_match, rationale
        except Exception as err:
            return "No", f"Error in scoring agent: {err}"

    return "No", "Could not invoke scoring agent API."


def identify_media(
    prompt: str,
    images: Optional[List[Image.Image]] = None,
    model_choice: str = "gpt-5.6-luna",
) -> str:
    """Backward-compatible function for single image description generation."""
    return generate_image_description(prompt=prompt, images=images, model_choice=model_choice)

