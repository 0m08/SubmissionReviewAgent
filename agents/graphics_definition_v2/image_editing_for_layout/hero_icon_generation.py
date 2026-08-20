import os
import re
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO

import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types
from langsmith import traceable
from PIL import Image

from agents.graphics_asset_creation.automated.llm_call_tracker import tracker
from agents.graphics_asset_creation.gac_utils import (
    DEFAULT_GENERATOR_MODEL,
    upload_image_to_drive,
)
from agents.graphics_asset_creation.image_editing.image_editing_openai import (
    image_from_base64,
)
from agents.graphics_asset_creation.reviewers.voiceover_reviewer import (
    call_llm_with_retry,
)
from agents.graphics_definition_v2.image_editing_for_layout.hero_animation_decision import (
    _ICONS_COLUMN,
    _PLAN_COLUMN,
    _hero_primary_asset_url,
    _is_video_asset_url,
    _scene_block,
)
from agents.graphics_definition_v2.image_editing_for_layout.hero_bbox_spatial import (
    split_hero_plan_scene_blocks,
)
from agents.graphics_definition_v2.image_editing_for_layout.image_edit_planning import (
    parse_scenes_from_slideshow_manifest,
)
from services.sheets_service import save_to_sheet
from services.smart_progress_bar import SmartProgressBar

load_dotenv()

hero_icon_generation_prompt = """Create one instructional overlay icon for an HVAC e-learning video player.

Icon meaning to depict:
{icon_concept}

This icon will be composited on top of a dimmed photograph. It must read clearly at small size on a phone in landscape.

## Style (mandatory)

- Simple line icon.
- Single color only: orange #F05523.
- Flat 2D geometric line art. Even stroke weight throughout. Rounded line caps and joins.
- Stroke thick enough to stay legible when the icon is small (medium-bold outline, not hairline, not chunky filled clipart).
- Centered in a square canvas. Generous even padding on all four sides (icon glyph occupies about 60–70% of the frame).
- Background: solid pure white #FFFFFF. No transparency, no gradient, no vignette, no drop shadow, no glow.
- No 3D, no bevel, no photorealism, no texture, no skeuomorphism.
- No cartoon characters, no mascots, no hands, no faces.
- No HVAC equipment photo. This is a symbol, not a product shot.

## Content rules

- Depict only the requested icon meaning. Do not add extra symbols.
- No text, letters, numbers, captions, titles, watermarks, or badges.
- No orange title bar or label container under the icon. Glyph only.
- No arrows pointing at empty space unless the concept itself is directional (for example airflow arrows).
- Keep the silhouette simple enough that a trainee can recognize it in under a second.

## Examples of correct look

- Warning: simple equilateral triangle outline with an exclamation mark, orange lines on white.
- Heat: simple flame outline.
- Cold: simple snowflake outline.
- Airflow: two or three parallel curved arrows.
- Water / moisture: simple droplet outline.
- Electricity: simple lightning bolt outline.
- Correct: simple check mark.
- Incorrect: simple X / cross.

## Do not

- Do not use marine blue, grey, black, or any second color on the glyph.
- Do not put the icon in a circle button, app-icon grid, or UI chrome unless the concept is specifically a badge.
- Do not generate multiple icons or a sheet of variants. One icon, one canvas.
- Do not add a logo.

Square 1:1 composition. White background. Orange #F05523 line icon only.
"""

HERO_ICON_DRIVE_FOLDER_ID = "1_cYkmnvDAUPardU1FyRHEU7Puwht6qZS"
_MAX_ICONS_PER_SCENE = 3
_ANIMATION_TYPE_RE = re.compile(
    r"<animation_type>\s*(.*?)\s*</animation_type>",
    re.DOTALL | re.IGNORECASE,
)
_ICON_RE = re.compile(r"<icon>(.*?)</icon>", re.DOTALL | re.IGNORECASE)
_ICON_CONCEPT_RE = re.compile(
    r"<icon_concept>\s*(.*?)\s*</icon_concept>",
    re.DOTALL | re.IGNORECASE,
)
_TARGET_RE = re.compile(
    r"<target_description>\s*(.*?)\s*</target_description>",
    re.DOTALL | re.IGNORECASE,
)
_PLACEMENT_RE = re.compile(
    r"<placement_hint>\s*(.*?)\s*</placement_hint>",
    re.DOTALL | re.IGNORECASE,
)


def _extract_tag_inner(block, pattern):
    """
    Return the first regex group from a plan block, stripped.

    :param block: Scene plan text or icon inner XML.
    :param pattern: Compiled regex with one capture group.
    :return: Captured string or empty string.
    """
    match = pattern.search(block or "")
    if not match:
        return ""
    return (match.group(1) or "").strip()


def _is_na(value):
    """
    True when a plan field is empty or the N/A sentinel.

    :param value: Raw tag inner text.
    :return: Whether the field should be ignored.
    """
    text = str(value or "").strip()
    return (not text) or text.upper() == "N/A"


def parse_icon_overlays_from_plan_block(block_text):
    """
    Parse icon overlay specs from one scene's hero_animation_plan block.

    :param block_text: Inner plan text for one scene.
    :return: List of dicts with icon_concept, target_description, placement_hint.
    """
    animation_type = _extract_tag_inner(block_text, _ANIMATION_TYPE_RE).lower()
    if animation_type != "icon_overlay":
        return []
    icons = []
    for inner in _ICON_RE.findall(block_text or ""):
        inner_stripped = (inner or "").strip()
        if _is_na(inner_stripped):
            continue
        concept = _extract_tag_inner(inner, _ICON_CONCEPT_RE)
        if _is_na(concept):
            continue
        icons.append(
            {
                "icon_concept": concept,
                "target_description": _extract_tag_inner(inner, _TARGET_RE),
                "placement_hint": _extract_tag_inner(inner, _PLACEMENT_RE),
            }
        )
        if len(icons) >= _MAX_ICONS_PER_SCENE:
            break
    return icons


def _get_gemini_client():
    """
    Google GenAI client for icon image generation.

    :return: Google GenAI client or None when API key is missing.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        api_key = st.session_state.get("google_api_key")
    if not api_key:
        print("Hero icon generation: Missing GOOGLE_API_KEY (or session google_api_key)")
        return None
    try:
        return genai.Client(api_key=api_key)
    except Exception as e:
        print(f"Hero icon generation: Failed to initialize Gemini client: {e}")
        return None


def _pil_from_white_background(image):
    """
    Flatten transparency onto white and return RGB.

    :param image: PIL image from the generator.
    :return: RGB PIL image.
    """
    if image.mode in ("RGBA", "LA"):
        background = Image.new("RGB", image.size, (255, 255, 255))
        alpha = image.split()[-1]
        background.paste(image.convert("RGBA"), mask=alpha)
        return background
    if image.mode != "RGB":
        return image.convert("RGB")
    return image


def _extract_pil_from_gemini_image_response(response):
    """
    Pull the first inline image out of a Gemini generate_content response.

    :param response: Gemini generate_content response.
    :return: RGB PIL image.
    """
    if not response or not response.candidates or not response.candidates[0].content:
        raise ValueError("Model returned no content.")

    image = None
    model_content = response.candidates[0].content
    if hasattr(model_content, "parts"):
        for part in model_content.parts:
            if hasattr(part, "inline_data") and part.inline_data:
                try:
                    image = Image.open(BytesIO(part.inline_data.data))
                    break
                except Exception as part_err:
                    print(f"Hero icon generation: error reading Gemini image part: {part_err}")

    if image is None and hasattr(response, "text"):
        try:
            image = image_from_base64(response.text)
        except Exception:
            pass

    if image is None:
        raise ValueError("No icon image found. The model may have refused the request.")
    return _pil_from_white_background(image)


def _should_retry_gemini_without_thinking(exc):
    """
    Return True if the Gemini error suggests retrying without thinking_config.

    :param exc: Exception from generate_content.
    :return: Whether a no-thinking retry is warranted.
    """
    msg = str(exc).lower()
    return any(
        s in msg
        for s in (
            "thinking",
            "thinking_config",
            "unsupported",
            "unknown field",
            "invalid_argument",
        )
    )


def generate_hero_icon_image(icon_concept, model=DEFAULT_GENERATOR_MODEL):
    """
    Generate one square overlay icon image from an icon_concept.

    :param icon_concept: Plain-language meaning of the icon.
    :param model: Gemini image model id.
    :return: RGB PIL image.
    """
    client = _get_gemini_client()
    if client is None:
        raise ValueError("Missing Google GenAI client (GOOGLE_API_KEY).")

    prompt_text = hero_icon_generation_prompt.format(icon_concept=icon_concept)
    user_content = types.Content(
        role="user",
        parts=[types.Part.from_text(text=prompt_text)],
    )
    image_config_obj = types.ImageConfig(image_size="1K", aspect_ratio="1:1")

    if "gemini-3" in (model or "").lower():
        configs_to_try = [
            types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"],
                image_config=image_config_obj,
                thinking_config=types.ThinkingConfig(thinking_level="high"),
            ),
            types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"],
                image_config=image_config_obj,
            ),
        ]
    else:
        configs_to_try = [
            types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"],
                image_config=image_config_obj,
            )
        ]

    response = None
    for idx, cfg in enumerate(configs_to_try):
        try:
            with tracker.call(model, "Hero Icon Overlay Generation") as usage:
                response = call_llm_with_retry(
                    client.models.generate_content,
                    model=model,
                    contents=[user_content],
                    config=cfg,
                )
                usage.set_response(response)
            break
        except Exception as e:
            if (
                idx == 0
                and len(configs_to_try) > 1
                and _should_retry_gemini_without_thinking(e)
            ):
                print(
                    "Hero icon generation: Gemini rejected thinking_config; "
                    "retrying generate_content without thinking_config."
                )
                continue
            print(f"Hero icon generation: Gemini {model} failed: {e}")
            raise

    return _extract_pil_from_gemini_image_response(response)


def _safe_filename_token(value):
    """
    Make a string safe for a Drive filename token.

    :param value: Raw scene id or similar.
    :return: Underscore-safe token.
    """
    token = re.sub(r"[^\w.-]+", "_", str(value or "").strip())
    return token or "unknown"


def _format_icon_field(value):
    """
    Sheet XML inner text for an optional icon field.

    :param value: Raw concept, target, or placement text.
    :return: Text or N/A.
    """
    text = str(value or "").strip()
    return text if text else "N/A"


def _format_icons_xml(icon_records):
    """
    Build the sheet XML for one icon_overlay scene.

    :param icon_records: List of dicts with concept, target, placement, url.
    :return: Inner XML string.
    """
    chunks = ["<icons>"]
    for record in icon_records:
        chunks.append(
            "<icon>\n"
            f"<icon_concept>{_format_icon_field(record.get('icon_concept'))}</icon_concept>\n"
            f"<target_description>{_format_icon_field(record.get('target_description'))}</target_description>\n"
            f"<placement_hint>{_format_icon_field(record.get('placement_hint'))}</placement_hint>\n"
            f"<url>{_format_icon_field(record.get('url'))}</url>\n"
            "</icon>"
        )
    chunks.append("</icons>")
    return "\n".join(chunks)


def generate_and_upload_hero_icon(scene_id, icon_index, icon_spec, drive, model=DEFAULT_GENERATOR_MODEL):
    """
    Generate one overlay icon and upload it to the edited-image Drive folder.

    :param scene_id: Scene id from slideshow_manifest.
    :param icon_index: 1-based icon index within the scene.
    :param icon_spec: Dict with icon_concept, target_description, placement_hint.
    :param drive: Drive client for upload.
    :param model: Gemini image model id.
    :return: Dict including generated Drive URL.
    """
    concept = icon_spec["icon_concept"]
    prompt_text = hero_icon_generation_prompt.format(icon_concept=concept)

    # print(f"\nHero Icon Generation - Scene ID: {scene_id} | Icon {icon_index}")
    # print("\n" + "=" * 80)
    # print("📝 FORMATTED ICON GENERATION PROMPT")
    # print("=" * 80)
    # print(prompt_text)
    # print("=" * 80 + "\n")

    if not drive:
        raise ValueError("Drive not available; cannot upload generated icon.")

    image = generate_hero_icon_image(concept, model=model)
    filename = (
        f"hero_icon_s{_safe_filename_token(scene_id)}"
        f"_i{icon_index}_{uuid.uuid4().hex[:8]}.png"
    )
    url = upload_image_to_drive(
        image, filename, drive, folder_id=HERO_ICON_DRIVE_FOLDER_ID
    )
    if not url:
        raise ValueError(f"Icon upload to Drive failed ({filename})")

    # print(f"📤 Hero Icon Generated URL: {url}\n")
    # print("=" * 100 + "\n")

    return {
        "icon_concept": concept,
        "target_description": icon_spec.get("target_description") or "",
        "placement_hint": icon_spec.get("placement_hint") or "",
        "url": url,
    }


@traceable(
    metadata={
        "agent_name": "graphics_definition_v2",
        "step_name": "Hero Overlay Animation Decision",
        "function_name": "process_hero_icon_generation_row",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous"),
    }
)
def process_hero_icon_generation_row(index, row, drive, model=DEFAULT_GENERATOR_MODEL):
    """
    Generate overlay icons for icon_overlay scenes on one Slide Chunks row.

    :param index: DataFrame row index.
    :param row: DataFrame row object with plan and manifest.
    :param drive: Drive client for upload.
    :param model: Gemini image model id.
    :return: Tuple of (index, overlays_text).
    """
    try:
        plan_text = str(row.get(_PLAN_COLUMN, "")).strip()
        if not plan_text or plan_text == "nan" or plan_text == "-":
            return index, "-"
        if plan_text.startswith("ERROR:"):
            return index, f"ERROR: plan not ready ({plan_text[:80]})"

        scenes = parse_scenes_from_slideshow_manifest(
            str(row.get("slideshow_manifest", "")).strip()
        )
        scene_by_id = {str(sc.get("id", "")).strip(): sc for sc in scenes}

        overlay_blocks = []
        for scene_id, block in split_hero_plan_scene_blocks(plan_text):
            icon_specs = parse_icon_overlays_from_plan_block(block)
            if not icon_specs:
                continue
            scene = scene_by_id.get(scene_id)
            asset_url = _hero_primary_asset_url(scene) if scene else ""
            if asset_url and _is_video_asset_url(asset_url):
                continue

            records = []
            for i, spec in enumerate(icon_specs, start=1):
                try:
                    records.append(
                        generate_and_upload_hero_icon(
                            scene_id=scene_id,
                            icon_index=i,
                            icon_spec=spec,
                            drive=drive,
                            model=model,
                        )
                    )
                except Exception as icon_err:
                    traceback.print_exc()
                    records.append(
                        {
                            "icon_concept": spec.get("icon_concept") or "",
                            "target_description": spec.get("target_description") or "",
                            "placement_hint": spec.get("placement_hint") or "",
                            "url": f"ERROR: {str(icon_err)}",
                        }
                    )
            overlay_blocks.append(_scene_block(scene_id, _format_icons_xml(records)))

        if not overlay_blocks:
            return index, "-"
        return index, "\n\n".join(overlay_blocks).strip()
    except Exception as e:
        print(f"Error hero icon generation row {index}: {e}")
        traceback.print_exc()
        return index, f"ERROR: {str(e)}"


def run_hero_icon_generation_phase(ws, df, drive, max_workers=50):
    """
    Phase 3 of Decide Hero Overlay Animation: write hero_icon_overlays.

    :param ws: Slide Chunks worksheet.
    :param df: Slide Chunks DataFrame (mutated in place).
    :param drive: Drive client.
    :param max_workers: Row-level parallel workers.
    :return: None
    """
    if _ICONS_COLUMN not in df.columns:
        df[_ICONS_COLUMN] = ""

    rows_to_process = []
    for index, row in df.iterrows():
        existing = str(row.get(_ICONS_COLUMN, "")).strip()
        plan_text = str(row.get(_PLAN_COLUMN, "")).strip()
        if str(row.get("Slide Type", "")).strip().lower() == "transition":
            if not existing or existing == "nan" or existing.startswith("ERROR:"):
                df.at[index, _ICONS_COLUMN] = "-"
            continue
        if existing and existing != "nan" and not existing.startswith("ERROR:"):
            continue
        if not plan_text or plan_text == "nan":
            continue
        rows_to_process.append((index, row))

    if not rows_to_process:
        save_to_sheet(ws, df)
        print("Hero icon generation: no rows to process.")
        return

    print(
        f"Hero icon generation: processing {len(rows_to_process)} row(s), "
        f"max_workers={max_workers}"
    )
    futures_map = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for index, row in rows_to_process:
            fut = executor.submit(process_hero_icon_generation_row, index, row, drive)
            futures_map[fut] = index

        progress = SmartProgressBar(
            total_tasks=len(futures_map),
            description="Hero icon overlays",
            save_interval=5,
        )
        for future in as_completed(futures_map):
            index = futures_map[future]
            try:
                row_index, overlays_text = future.result()
                df.at[row_index, _ICONS_COLUMN] = overlays_text
                progress.update()
                if progress.should_save():
                    save_to_sheet(ws, df)
            except Exception as e:
                print(f"Hero icon generation future error row {index}: {e}")
                df.at[index, _ICONS_COLUMN] = f"ERROR: {str(e)}"
                progress.update()

    save_to_sheet(ws, df)
    print("Hero icon generation: complete.")
