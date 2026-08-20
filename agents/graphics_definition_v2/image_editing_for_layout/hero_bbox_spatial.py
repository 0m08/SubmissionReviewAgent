import json
import re
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st
from dotenv import load_dotenv
from google.genai import types
from langsmith import traceable

from agents.graphics_definition_v2.aggregation_agent.aggregation_agent import (
    invoke_gemini_multimodal,
)
from agents.graphics_definition_v2.image_editing_for_layout.hero_animation_decision import (
    _COORDS_COLUMN,
    _PLAN_COLUMN,
    _hero_primary_asset_url,
    _is_video_asset_url,
    _scene_block,
)
from agents.graphics_definition_v2.image_editing_for_layout.image_edit_planning import (
    parse_scenes_from_slideshow_manifest,
)
from agents.graphics_definition_v2.review_agent.review_and_revise import (
    build_visual_part_only,
)
from services.sheets_service import save_to_sheet
from services.smart_progress_bar import SmartProgressBar

load_dotenv()

hero_bbox_spatial_prompt = """You are a spatial localization model. Locate regions in the provided still image.

Return bounding boxes for each listed target. Coordinates use Gemini spatial understanding format:
- box_2d is [ymin, xmin, ymax, xmax]
- Each value is an integer from 0 to 1000, normalized to the image height and width
- ymin < ymax and xmin < xmax
- The box must tightly enclose only the described target
- CRITICAL: Bounding boxes for different targets must NEVER overlap or intersect with each other. Keep them entirely distinct.

Targets (in order):

{targets_block}

Output a JSON array only. One object per target, same order, no markdown fences, no extra keys besides index, box_2d, and label:

[{{"index": 1, "box_2d": [ymin, xmin, ymax, xmax], "label": "short name"}}, ...]
"""

_SCENE_HEADER_RE = re.compile(r"---Scene ID:\s*(\S+)---", re.IGNORECASE)
_ANIMATION_TYPE_RE = re.compile(
    r"<animation_type>\s*(.*?)\s*</animation_type>",
    re.DOTALL | re.IGNORECASE,
)
_HIGHLIGHT_RE = re.compile(r"<highlight>(.*?)</highlight>", re.DOTALL | re.IGNORECASE)
_TARGET_RE = re.compile(
    r"<target_description>\s*(.*?)\s*</target_description>",
    re.DOTALL | re.IGNORECASE,
)
_SHAPE_RE = re.compile(
    r"<highlight_shape>\s*(.*?)\s*</highlight_shape>",
    re.DOTALL | re.IGNORECASE,
)


def _extract_tag_inner(block, pattern):
    """
    Return the first regex group from a plan block, stripped.

    :param block: Scene plan text.
    :param pattern: Compiled regex with one capture group.
    :return: Captured string or empty string.
    """
    match = pattern.search(block or "")
    if not match:
        return ""
    return (match.group(1) or "").strip()


def split_hero_plan_scene_blocks(plan_text):
    """
    Split a hero_animation_plan cell into (scene_id, inner_text) pairs.

    :param plan_text: Sheet cell value for hero_animation_plan.
    :return: List of (scene_id, block_text) tuples.
    """
    text = str(plan_text or "")
    matches = list(_SCENE_HEADER_RE.finditer(text))
    if not matches:
        return []
    blocks = []
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        scene_id = (match.group(1) or "").strip()
        blocks.append((scene_id, text[start:end].strip()))
    return blocks


def parse_bbox_highlights_from_plan_block(block_text):
    """
    Parse highlight targets from one scene's hero_animation_plan block.

    :param block_text: Inner plan text for one scene.
    :return: List of dicts with keys target_description and shape.
    """
    animation_type = _extract_tag_inner(block_text, _ANIMATION_TYPE_RE).lower()
    if animation_type != "bbox_highlight":
        return []
    highlights = []
    for inner in _HIGHLIGHT_RE.findall(block_text or ""):
        inner_stripped = (inner or "").strip()
        if not inner_stripped or inner_stripped.upper() == "N/A":
            continue
        target = _extract_tag_inner(inner, _TARGET_RE)
        shape = _extract_tag_inner(inner, _SHAPE_RE).lower()
        if not target or target.upper() == "N/A":
            continue
        if shape not in ("box", "circle"):
            shape = "box"
        highlights.append({"target_description": target, "shape": shape})
    return highlights


def _parse_box_2d(raw):
    """
    Normalize a model box_2d value to four ints in 0–1000.

    :param raw: List or string from the model.
    :return: [ymin, xmin, ymax, xmax] or None if invalid.
    """
    values = raw
    if isinstance(raw, str):
        numbers = re.findall(r"-?\d+", raw)
        values = numbers
    if not isinstance(values, (list, tuple)) or len(values) != 4:
        return None
    try:
        nums = [int(round(float(v))) for v in values]
    except (TypeError, ValueError):
        return None
    ymin, xmin, ymax, xmax = nums
    ymin = max(0, min(1000, ymin))
    xmin = max(0, min(1000, xmin))
    ymax = max(0, min(1000, ymax))
    xmax = max(0, min(1000, xmax))
    if ymin >= ymax or xmin >= xmax:
        return None
    return [ymin, xmin, ymax, xmax]


def parse_spatial_boxes_json(response_text, expected_count):
    """
    Parse Gemini spatial JSON into one box_2d list per expected target.

    :param response_text: Raw model response.
    :param expected_count: Number of highlight targets.
    :return: List of box_2d lists or None placeholders, length expected_count.
    """
    boxes = [None] * expected_count
    if not response_text:
        return boxes
    text = str(response_text).strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    array_match = re.search(r"\[.*\]", text, re.DOTALL)
    if not array_match:
        return boxes
    try:
        payload = json.loads(array_match.group(0))
    except json.JSONDecodeError:
        return boxes
    if isinstance(payload, dict):
        payload = payload.get("boxes") or payload.get("objects") or [payload]
    if not isinstance(payload, list):
        return boxes
    for i, item in enumerate(payload):
        if i >= expected_count:
            break
        if not isinstance(item, dict):
            continue
        idx = item.get("index")
        slot = i
        if idx is not None:
            try:
                slot = int(idx) - 1
            except (TypeError, ValueError):
                slot = i
        if slot < 0 or slot >= expected_count:
            continue
        parsed = _parse_box_2d(item.get("box_2d"))
        if parsed:
            boxes[slot] = parsed
    return boxes


def _format_highlights_xml(highlights, boxes):
    """
    Build the sheet XML for one bbox scene.

    :param highlights: List of highlight dicts with shape.
    :param boxes: Parallel list of box_2d lists or None.
    :return: Inner XML string.
    """
    chunks = ["<highlights>"]
    for highlight, box in zip(highlights, boxes):
        shape = highlight.get("shape") or "box"
        if box:
            box_text = ",".join(str(v) for v in box)
        else:
            box_text = "ERROR: missing box_2d"
        chunks.append(
            "<highlight>\n"
            f"<shape>{shape}</shape>\n"
            f"<box_2d>{box_text}</box_2d>\n"
            "</highlight>"
        )
    chunks.append("</highlights>")
    return "\n".join(chunks)


@traceable(
    metadata={
        "agent_name": "graphics_definition_v2",
        "step_name": "Hero Overlay Animation Decision",
        "function_name": "generate_hero_bbox_coordinates_for_scene",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous"),
    }
)
def generate_hero_bbox_coordinates_for_scene(scene_id, asset_url, highlights, drive, llm="gemini_3_flash_thinking"):
    """
    Locate bbox_highlight targets on one hero still using Gemini spatial understanding.

    :param scene_id: Scene id from slideshow_manifest.
    :param asset_url: Final hero image URL.
    :param highlights: List of dicts with target_description and shape.
    :param drive: Drive client for loading the image.
    :param llm: LLM identifier.
    :return: Inner XML string of <highlights> for this scene.
    """
    target_lines = []
    for i, highlight in enumerate(highlights, start=1):
        target_lines.append(f"{i}. {highlight['target_description']}")
    prompt_text = hero_bbox_spatial_prompt.format(targets_block="\n".join(target_lines))

    parts = []
    parts.append(types.Part(text=f"Hero image URL: {asset_url}\n"))
    visual_part = build_visual_part_only(asset_url, drive)
    if not visual_part:
        raise ValueError(f"Failed to load hero image for spatial locate, scene {scene_id}: {asset_url}")
    parts.append(visual_part)
    parts.append(types.Part(text=prompt_text))

    # print(f"\nHero BBox Spatial - Scene ID: {scene_id}")
    # print("\n" + "=" * 80)
    # print("📝 FORMATTED SPATIAL PROMPT")
    # print("=" * 80)
    # print(f"Assigned single-hero visual asset URL: {asset_url}")
    # print("[INLINE IMAGE attached]")
    # print(prompt_text)
    # print("=" * 80 + "\n")

    response_text = invoke_gemini_multimodal(parts, llm=llm, temperature=0.1)
    # print("📤 Hero BBox Spatial Full Response from LLM:\n")
    # print(response_text or "")
    # print("\n" + "=" * 100 + "\n")

    boxes = parse_spatial_boxes_json(response_text, len(highlights))
    return _format_highlights_xml(highlights, boxes)


@traceable(
    metadata={
        "agent_name": "graphics_definition_v2",
        "step_name": "Hero Overlay Animation Decision",
        "function_name": "process_hero_bbox_spatial_row",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous"),
    }
)
def process_hero_bbox_spatial_row(index, row, drive, llm="gemini_3_flash_thinking"):
    """
    Fill bbox coordinates for bbox_highlight scenes on one Slide Chunks row.

    :param index: DataFrame row index.
    :param row: DataFrame row object with plan and manifest.
    :param drive: Drive client for loading hero images.
    :param llm: LLM identifier.
    :return: Tuple of (index, coordinates_text).
    """
    try:
        plan_text = str(row.get(_PLAN_COLUMN, "")).strip()
        if not plan_text or plan_text == "nan" or plan_text == "-":
            return index, "-"
        if plan_text.startswith("ERROR:"):
            return index, f"ERROR: plan not ready ({plan_text[:80]})"

        scenes = parse_scenes_from_slideshow_manifest(str(row.get("slideshow_manifest", "")).strip())
        scene_by_id = {str(sc.get("id", "")).strip(): sc for sc in scenes}

        coord_blocks = []
        any_bbox = False
        for scene_id, block in split_hero_plan_scene_blocks(plan_text):
            highlights = parse_bbox_highlights_from_plan_block(block)
            if not highlights:
                continue
            any_bbox = True
            scene = scene_by_id.get(scene_id)
            asset_url = _hero_primary_asset_url(scene) if scene else ""
            if not asset_url:
                coord_blocks.append(_scene_block(scene_id, "ERROR: Missing primary_visual URL"))
                continue
            if _is_video_asset_url(asset_url):
                continue
            try:
                inner = generate_hero_bbox_coordinates_for_scene(
                    scene_id=scene_id,
                    asset_url=asset_url,
                    highlights=highlights,
                    drive=drive,
                    llm=llm,
                )
                coord_blocks.append(_scene_block(scene_id, inner))
            except Exception as scene_err:
                coord_blocks.append(_scene_block(scene_id, f"ERROR: {str(scene_err)}"))
                traceback.print_exc()

        if not any_bbox:
            return index, "-"
        return index, "\n\n".join(coord_blocks).strip()
    except Exception as e:
        print(f"Error hero bbox spatial row {index}: {e}")
        traceback.print_exc()
        return index, f"ERROR: {str(e)}"


def run_hero_bbox_spatial_phase(ws, df, drive, llm="gemini_3_flash_thinking", max_workers=50):
    """
    Phase 2 of Decide Hero Overlay Animation: write hero_bbox_coordinates.

    :param ws: Slide Chunks worksheet.
    :param df: Slide Chunks DataFrame (mutated in place).
    :param drive: Drive client.
    :param llm: LLM identifier.
    :param max_workers: Row-level parallel workers.
    :return: None
    """
    if _COORDS_COLUMN not in df.columns:
        df[_COORDS_COLUMN] = ""

    rows_to_process = []
    for index, row in df.iterrows():
        existing = str(row.get(_COORDS_COLUMN, "")).strip()
        plan_text = str(row.get(_PLAN_COLUMN, "")).strip()
        if str(row.get("Slide Type", "")).strip().lower() == "transition":
            if not existing or existing == "nan" or existing.startswith("ERROR:"):
                df.at[index, _COORDS_COLUMN] = "-"
            continue
        if existing and existing != "nan" and not existing.startswith("ERROR:"):
            continue
        if not plan_text or plan_text == "nan":
            continue
        rows_to_process.append((index, row))

    if not rows_to_process:
        save_to_sheet(ws, df)
        print("Hero bbox spatial: no rows to process.")
        return

    print(f"Hero bbox spatial: processing {len(rows_to_process)} row(s), max_workers={max_workers}")
    futures_map = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for index, row in rows_to_process:
            fut = executor.submit(process_hero_bbox_spatial_row, index, row, drive, llm)
            futures_map[fut] = index

        progress = SmartProgressBar(
            total_tasks=len(futures_map),
            description="Hero bbox coordinates",
            save_interval=5,
        )
        for future in as_completed(futures_map):
            index = futures_map[future]
            try:
                row_index, coords_text = future.result()
                df.at[row_index, _COORDS_COLUMN] = coords_text
                progress.update()
                if progress.should_save():
                    save_to_sheet(ws, df)
            except Exception as e:
                print(f"Hero bbox spatial future error row {index}: {e}")
                df.at[index, _COORDS_COLUMN] = f"ERROR: {str(e)}"
                progress.update()

    save_to_sheet(ws, df)
    print("Hero bbox spatial: complete.")
