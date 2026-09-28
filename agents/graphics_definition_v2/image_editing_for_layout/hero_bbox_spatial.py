import json
import re
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO

import streamlit as st
from dotenv import load_dotenv
from google.genai import types
from langsmith import traceable
from PIL import Image, ImageDraw, ImageFont

from agents.graphics_asset_creation.gac_utils import upload_image_to_drive
from agents.graphics_definition_v2.aggregation_agent.aggregation_agent import (
    load_image_from_url,
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
    invoke_gemini_multimodal,
)
from services.sheets_service import save_to_sheet
from services.smart_progress_bar import SmartProgressBar

load_dotenv()

_MAX_SPATIAL_REVIEW_ROUNDS = 3

# Hero-overlay Drive folder for bbox review previews.
HERO_BBOX_DRIVE_FOLDER_ID = "1_cYkmnvDAUPardU1FyRHEU7Puwht6qZS"

# Minimum gap between boxes in Gemini normalized coords (0–1000).
# 0 = flush shared edge is already illegal; >0 requires a visible gap.
_BBOX_MIN_GAP = 8

# Minimum readable box size in Gemini normalized coords (0–1000).
_BBOX_MIN_WIDTH = 55
_BBOX_MIN_HEIGHT = 55
_BBOX_MIN_AREA = 6000

hero_bbox_spatial_prompt = """You are a spatial localization model. Locate regions in the provided still image.

Return bounding boxes for each listed target. Coordinates use Gemini spatial understanding format:
- box_2d is [ymin, xmin, ymax, xmax]
- Each value is an integer from 0 to 1000, normalized to the image height and width
- ymin < ymax and xmin < xmax
- The box must tightly enclose the described target, but MUST remain large enough to read on screen
- CRITICAL size floor (normalized 0–1000): width >= {min_w}, height >= {min_h}, and width*height >= {min_area}. Never return hairline, ribbon, or pill-thin boxes. If the named part is inherently too small/thin to fill that floor on its own, DO NOT enlarge the box by pulling in unrelated nearby objects — omit that target entirely. Prefer fewer readable boxes over padding a tiny target with context we did not intend to emphasize
- CRITICAL: Bounding boxes for different targets must NEVER overlap, intersect, or touch each other. Do not let two boxes share a common edge or corner (e.g. one box's right edge flush against another's left edge). Leave a visible gap between every pair of boxes.
- Prefer fewer readable boxes over many tiny ones. One clear box is better than one good box plus one unreadable sliver

Targets (in order):

{targets_block}

Output a JSON array only. One object per target, same order, no markdown fences, no extra keys besides index, box_2d, and label:

[{{"index": 1, "box_2d": [ymin, xmin, ymax, xmax], "label": "short name"}}, ...]
"""

hero_bbox_spatial_refine_prompt = """You are a spatial localization model reviewing your own previous attempt.

We have overlaid your previously generated bounding boxes on top of the image in orange with numbered labels (e.g. [1], [2]) matching each target index.

Please carefully inspect the overlaid orange boxes and identify any mistakes:
1. Incorrect placement: Is the orange box off-target, misaligned, or not tightly enclosing the correct object?
2. Size issues: Is the orange box a thin horizontal/vertical sliver, ribbon, or tiny pill that a learner cannot clearly read? Each box MUST have width >= {min_w}, height >= {min_h}, and area (width*height) >= {min_area} in normalized 0–1000 coords. If a box is too small, DROP that highlight — do NOT enlarge it by adding unrelated nearby elements that were not meant to be emphasized. Never keep an unreadable sliver, and never "fix" size by padding with extra objects.
3. Interference: Are the orange boxes overlapping or intersecting each other? Each target's box must be entirely separate.
4. Touching / shared edges: Do any two orange boxes share a common side or corner (flush adjacency with no gap)? Even without overlap, edge-to-edge contact makes boxes look merged. Separate them with clear space between every pair.
5. Edge/Corner clipping: Is the orange box touching the slide's edges or corners awkwardly? Ensure the box fits cleanly.
6. Prefer fewer readable boxes: If one box is large and clear and another is a tiny leftover, revise by dropping the tiny one. One good box beats two boxes when the second is unreadable. Never enlarge a tiny box with unrelated context.

If ALL orange boxes pass every checklist item above, output ONLY:
{{"action": "keep"}}

If ANY checklist item fails for any box, output:
{{"action": "revise", "boxes": [{{"index": 1, "box_2d": [ymin, xmin, ymax, xmax], "label": "short name"}}, ...]}}

Rules for revise:
- box_2d is [ymin, xmin, ymax, xmax]
- Each value is an integer from 0 to 1000, normalized to the image height and width
- ymin < ymax and xmin < xmax
- One object per target, same order as the targets list
- Every returned box MUST meet the size floor (width >= {min_w}, height >= {min_h}, area >= {min_area}). If a target cannot meet that floor without adding unrelated objects, omit it from the revised list (keep only the readable boxes)
- CRITICAL: Bounding boxes for different targets must NEVER overlap, intersect, or touch each other. Keep them entirely distinct.
- Do NOT revise if all boxes already look correct and readable — prefer "keep" to avoid unnecessary changes.

Targets (in order):

{targets_block}

Output JSON only — no markdown fences.
"""


def _box_pair_gap(a, b):
    """
    Return (h_gap, v_gap) between two axis-aligned boxes in normalized coords.

    Negative gap means overlap on that axis. Separation distance is max(0, h_gap)
    when boxes already separate on the other axis, etc.
    """
    ay0, ax0, ay1, ax1 = a
    by0, bx0, by1, bx1 = b
    h_gap = max(ax0, bx0) - min(ax1, bx1)
    v_gap = max(ay0, by0) - min(ay1, by1)
    return h_gap, v_gap


def boxes_interfere(a, b, min_gap=_BBOX_MIN_GAP):
    """
    True when two boxes overlap, share an edge, or are closer than min_gap along a shared projection (stacked/side-by-side neighbors).

    Diagonally separated boxes (positive gap on both axes) do not interfere.
    """
    if not a or not b:
        return False
    h_gap, v_gap = _box_pair_gap(a, b)
    if h_gap < 0 and v_gap < 0:
        return True  # area overlap
    if h_gap < 0:
        # Overlap in X → neighbors vertically; require vertical gap
        return v_gap < min_gap
    if v_gap < 0:
        # Overlap in Y → neighbors horizontally; require horizontal gap
        return h_gap < min_gap
    # Positive gap on both axes = diagonal separation (no shared edge)
    return False


def find_bbox_interference_pairs(boxes, min_gap=_BBOX_MIN_GAP):
    """
    Return list of (i, j, kind, detail) for every interfering pair (0-based indices).

    kind is "overlap", "shared_edge", or "too_close".
    """
    pairs = []
    n = len(boxes)
    for i in range(n):
        if not boxes[i]:
            continue
        for j in range(i + 1, n):
            if not boxes[j]:
                continue
            if not boxes_interfere(boxes[i], boxes[j], min_gap=min_gap):
                continue
            h_gap, v_gap = _box_pair_gap(boxes[i], boxes[j])
            if h_gap < 0 and v_gap < 0:
                kind = "overlap"
                detail = (
                    f"boxes [{i + 1}] and [{j + 1}] overlap "
                    f"(h_gap={h_gap}, v_gap={v_gap})"
                )
            elif h_gap <= 0 or v_gap <= 0:
                kind = "shared_edge"
                detail = (
                    f"boxes [{i + 1}] and [{j + 1}] share/touch an edge "
                    f"(h_gap={h_gap}, v_gap={v_gap})"
                )
            else:
                kind = "too_close"
                detail = (
                    f"boxes [{i + 1}] and [{j + 1}] are closer than min gap {min_gap} "
                    f"(h_gap={h_gap}, v_gap={v_gap})"
                )
            pairs.append((i, j, kind, detail))
    return pairs


def format_bbox_interference_feedback(pairs):
    """Human/LLM-readable geometry failure text."""
    if not pairs:
        return ""
    lines = [
        "GEOMETRIC VALIDATION FAILED. These orange boxes are illegal and MUST be revised:",
    ]
    for _, _, _, detail in pairs:
        lines.append(f"- {detail}")
    lines.append(
        f"Requirements: no overlap, no shared edges, and at least {_BBOX_MIN_GAP} "
        "normalized units of clear gap between every pair. Shrink or reposition boxes "
        "so each target is tightly enclosed AND fully separated from every other box."
    )
    return "\n".join(lines)


def box_size_metrics(box):
    """Return (width, height, area) in normalized coords, or None if invalid."""
    if not box or len(box) != 4:
        return None
    ymin, xmin, ymax, xmax = box
    width = int(xmax) - int(xmin)
    height = int(ymax) - int(ymin)
    if width <= 0 or height <= 0:
        return None
    return width, height, width * height


def is_box_too_small(box, min_width=_BBOX_MIN_WIDTH, min_height=_BBOX_MIN_HEIGHT, min_area=_BBOX_MIN_AREA):
    """True when a box is too small or too thin to read on screen."""
    metrics = box_size_metrics(box)
    if not metrics:
        return True
    width, height, area = metrics
    return width < min_width or height < min_height or area < min_area


def find_undersized_boxes(boxes, min_width=_BBOX_MIN_WIDTH, min_height=_BBOX_MIN_HEIGHT, min_area=_BBOX_MIN_AREA):
    """
    Return list of (index, detail) for boxes below the readable-size floor (0-based).
    """
    bad = []
    for i, box in enumerate(boxes):
        if not box:
            continue
        metrics = box_size_metrics(box)
        if not metrics:
            bad.append((i, f"box [{i + 1}] is invalid / empty"))
            continue
        width, height, area = metrics
        reasons = []
        if width < min_width:
            reasons.append(f"width {width} < {min_width}")
        if height < min_height:
            reasons.append(f"height {height} < {min_height}")
        if area < min_area:
            reasons.append(f"area {area} < {min_area}")
        if reasons:
            bad.append(
                (
                    i,
                    f"box [{i + 1}] too small/thin ({', '.join(reasons)}; "
                    f"size {width}x{height})",
                )
            )
    return bad


def format_bbox_size_feedback(undersized):
    """Human/LLM-readable size failure text."""
    if not undersized:
        return ""
    lines = [
        "SIZE VALIDATION FAILED. These orange boxes are too small/thin to read and MUST be revised or dropped:",
    ]
    for _, detail in undersized:
        lines.append(f"- {detail}")
    lines.append(
        f"Requirements: width >= {_BBOX_MIN_WIDTH}, height >= {_BBOX_MIN_HEIGHT}, "
        f"area >= {_BBOX_MIN_AREA} (normalized 0–1000). DROP each failing box — do NOT "
        "enlarge it by pulling in unrelated nearby objects. Prefer one clear readable "
        "box over keeping (or padding) an unreadable sliver."
    )
    return "\n".join(lines)


def combine_bbox_validation_feedback(boxes):
    """Combine interference + size validation feedback for review prompts."""
    parts = []
    interference = find_bbox_interference_pairs(boxes)
    if interference:
        parts.append(format_bbox_interference_feedback(interference))
    undersized = find_undersized_boxes(boxes)
    if undersized:
        parts.append(format_bbox_size_feedback(undersized))
    return "\n\n".join(parts), interference, undersized


def drop_undersized_boxes(boxes):
    """
    Set undersized boxes to None (kept as slots for logging; formatter skips them).

    :return: (new_boxes, dropped_1based_indices)
    """
    result = list(boxes)
    dropped = []
    for i, _detail in find_undersized_boxes(result):
        result[i] = None
        dropped.append(i + 1)
    return result, dropped


def _clamp_box(box):
    """Clamp and order a box_2d to valid [ymin,xmin,ymax,xmax] ints in 0–1000."""
    if not box or len(box) != 4:
        return None
    y0, x0, y1, x1 = [int(round(v)) for v in box]
    y0 = max(0, min(1000, y0))
    x0 = max(0, min(1000, x0))
    y1 = max(0, min(1000, y1))
    x1 = max(0, min(1000, x1))
    if y1 <= y0 or x1 <= x0:
        return None
    return [y0, x0, y1, x1]


def separate_interfering_boxes(boxes, min_gap=_BBOX_MIN_GAP, max_iters=24):
    """
    Deterministically shrink interfering boxes until they no longer touch/overlap.

    Shrinks each conflicting pair along the axis of least overlap/separation deficit.
    Returns a new list (None slots preserved).
    """
    result = [list(b) if b else None for b in boxes]
    for _ in range(max_iters):
        pairs = find_bbox_interference_pairs(result, min_gap=min_gap)
        if not pairs:
            break
        for i, j, _, _ in pairs:
            a = result[i]
            b = result[j]
            if not a or not b:
                continue
            h_gap, v_gap = _box_pair_gap(a, b)
            # Prefer separating on the axis that needs the smaller adjustment.
            need_h = min_gap - h_gap
            need_v = min_gap - v_gap
            if need_h <= need_v:
                # Separate horizontally: push right edge of left box / left edge of right box
                half = max(1, (need_h + 1) // 2)
                if a[3] <= b[3]:
                    # a is leftish
                    a[3] = max(a[1] + 1, a[3] - half)
                    b[1] = min(b[3] - 1, b[1] + half)
                else:
                    b[3] = max(b[1] + 1, b[3] - half)
                    a[1] = min(a[3] - 1, a[1] + half)
            else:
                half = max(1, (need_v + 1) // 2)
                if a[2] <= b[2]:
                    a[2] = max(a[0] + 1, a[2] - half)
                    b[0] = min(b[2] - 1, b[0] + half)
                else:
                    b[2] = max(b[0] + 1, b[2] - half)
                    a[0] = min(a[2] - 1, a[0] + half)
            result[i] = _clamp_box(a)
            result[j] = _clamp_box(b)

    # Last resort: drop later box in each remaining conflict
    remaining = find_bbox_interference_pairs(result, min_gap=min_gap)
    drop = set()
    for i, j, _, _ in remaining:
        drop.add(j)
    if drop:
        for idx in drop:
            result[idx] = None
        print(
            f"⚠️ Bbox geometry: dropped box index(es) {[d + 1 for d in sorted(drop)]} "
            "after failed separation."
        )
    return result


def draw_bboxes_on_image(pil_img, boxes):
    """
    Draw orange bounding boxes and indices on a PIL image.
    
    :param pil_img: Original PIL Image.
    :param boxes: List of [ymin, xmin, ymax, xmax] lists (or None).
    :return: Annotated PIL Image.
    """
    draw_img = pil_img.copy().convert("RGB")
    draw = ImageDraw.Draw(draw_img)
    width, height = draw_img.size
    
    # Select size proportional to image height
    font_size = max(12, int(height * 0.022))
    
    # Try loading clean font
    font = None
    for font_name in ["arial.ttf", "LiberationSans-Regular.ttf", "Helvetica.ttf"]:
        try:
            font = ImageFont.truetype(font_name, size=font_size)
            break
        except IOError:
            continue
    if font is None:
        font = ImageFont.load_default()
        
    for idx, box in enumerate(boxes):
        if not box:
            continue
        ymin, xmin, ymax, xmax = box
        
        # Scale to pixels
        py_min = int((ymin / 1000.0) * height)
        px_min = int((xmin / 1000.0) * width)
        py_max = int((ymax / 1000.0) * height)
        px_max = int((xmax / 1000.0) * width)
        
        orange_color = (240, 85, 35) # RGB for #F05523
        
        # Draw thick orange border
        thickness = max(2, int(width * 0.005))
        for t in range(thickness):
            draw.rectangle(
                [px_min + t, py_min + t, px_max - t, py_max - t],
                outline=orange_color
            )
            
        # Draw index badge [idx+1]
        label_text = f"[{idx + 1}]"
        try:
            # Pillow 10+ uses textbbox
            left, top, right, bottom = draw.textbbox((0, 0), label_text, font=font)
            text_width = right - left
            text_height = bottom - top
        except AttributeError:
            text_width, text_height = draw.textsize(label_text, font=font)
            
        badge_left = px_min
        badge_top = max(0, py_min - text_height - 6)
        badge_right = px_min + text_width + 8
        badge_bottom = py_min
        
        # Draw badge background
        draw.rectangle([badge_left, badge_top, badge_right, badge_bottom], fill=orange_color)
        # Write white text
        draw.text((badge_left + 4, badge_top + 2), label_text, fill=(255, 255, 255), font=font)
        
    return draw_img


def _pil_to_jpeg_part(pil_image):
    """Encode a PIL image as a Gemini JPEG inline Part."""
    buffered = BytesIO()
    pil_image.convert("RGB").save(buffered, format="JPEG")
    return types.Part(inline_data=types.Blob(mime_type="image/jpeg", data=buffered.getvalue()))


def _upload_bbox_review_overlay(annotated_img, scene_id, review_round, drive):
    """
    Upload an annotated bbox review image to Drive for auditability.

    :return: Drive view URL or empty string.
    """
    if not drive or annotated_img is None:
        return ""
    scene_token = re.sub(r"[^A-Za-z0-9_-]+", "_", str(scene_id or "scene"))[:40]
    filename = f"bbox_review_s{scene_token}_r{review_round}_{uuid.uuid4().hex[:8]}.png"
    try:
        url = upload_image_to_drive(
            annotated_img,
            filename,
            drive,
            folder_id=HERO_BBOX_DRIVE_FOLDER_ID,
        )
        if url:
            print(f"📎 Bbox review overlay (scene {scene_id}, round {review_round}): {url}")
        else:
            print(
                f"⚠️ Bbox review overlay upload failed "
                f"(scene {scene_id}, round {review_round})."
            )
        return url or ""
    except Exception as err:
        print(
            f"⚠️ Bbox review overlay upload error "
            f"(scene {scene_id}, round {review_round}): {err}"
        )
        return ""


def _bbox_review_prompt_text(refine_prompt, review_round, geometry_feedback=""):
    """Review prompt for one in-chat review round."""
    geo = ""
    if geometry_feedback:
        geo = (
            "\n\nIMPORTANT — programmatic geometry check failed on the previous boxes. "
            "Do NOT output keep until these are fixed:\n"
            f"{geometry_feedback}\n"
        )
    if review_round <= 1:
        return refine_prompt + geo
    return (
        f"Review round {review_round} of {_MAX_SPATIAL_REVIEW_ROUNDS}. "
        "The attached image shows your latest orange box overlay after your previous revision.\n\n"
        f"{refine_prompt}"
        f"{geo}"
    )


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
_TRIGGER_PHRASE_RE = re.compile(
    r"<trigger_phrase>\s*(.*?)\s*</trigger_phrase>",
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
        if not target or target.upper() == "N/A":
            continue
        shape = "box"
        highlights.append({
            "target_description": target,
            "shape": shape,
            "trigger_phrase": _extract_tag_inner(inner, _TRIGGER_PHRASE_RE),
        })
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


_REFINE_KEEP_ACTIONS = frozenset(
    {"keep", "pass", "approve", "ok", "accept", "no_change", "unchanged"}
)


def parse_spatial_refine_response(response_text, expected_count, first_pass_boxes):
    """
    Parse pass-2 bbox review JSON (keep vs revise).

    :return: Tuple of (boxes list, kept_pass1 bool).
    """
    if not response_text:
        return first_pass_boxes, True

    text = str(response_text).strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()

    obj_match = re.search(r"\{.*\}", text, re.DOTALL)
    if obj_match:
        try:
            payload = json.loads(obj_match.group(0))
            if isinstance(payload, dict):
                action = str(payload.get("action", "")).lower().strip()
                if action in _REFINE_KEEP_ACTIONS:
                    return first_pass_boxes, True
                if action == "revise":
                    boxes_raw = payload.get("boxes") or payload.get("objects")
                    if boxes_raw is not None:
                        revised = parse_spatial_boxes_json(
                            json.dumps(boxes_raw), expected_count
                        )
                        final_boxes = []
                        for b1, b2 in zip(first_pass_boxes, revised):
                            final_boxes.append(b2 if b2 is not None else b1)
                        return final_boxes, False
                    return first_pass_boxes, True
        except json.JSONDecodeError:
            pass

    # Legacy: bare JSON array means revise
    revised = parse_spatial_boxes_json(response_text, expected_count)
    if any(b is not None for b in revised):
        final_boxes = []
        for b1, b2 in zip(first_pass_boxes, revised):
            final_boxes.append(b2 if b2 is not None else b1)
        return final_boxes, False
    return first_pass_boxes, True


def _format_highlights_xml(highlights, boxes):
    """
    Build the sheet XML for one bbox scene.

    :param highlights: List of highlight dicts with shape.
    :param boxes: Parallel list of box_2d lists or None.
    :return: Inner XML string.
    """
    chunks = ["<highlights>"]
    kept = 0
    for highlight, box in zip(highlights, boxes):
        if not box or is_box_too_small(box):
            continue
        shape = highlight.get("shape") or "box"
        box_text = ",".join(str(v) for v in box)
        trigger = highlight.get("trigger_phrase") or ""
        trigger_line = (
            f"<trigger_phrase>{trigger}</trigger_phrase>\n" if trigger else ""
        )
        chunks.append(
            "<highlight>\n"
            f"<shape>{shape}</shape>\n"
            f"<box_2d>{box_text}</box_2d>\n"
            f"{trigger_line}"
            "</highlight>"
        )
        kept += 1
    if kept == 0:
        # Preserve a clear signal when every proposed box was dropped.
        chunks.append("<!-- all highlights dropped: missing or below min size -->")
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
    Locate bbox_highlight targets on one hero still using one multi-turn Gemini chat.

    Turn 1: locate boxes on the original still.
    Turns 2–4: up to three in-chat review rounds (keep → exit; revise → update overlay).

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

    targets_block_text = "\n".join(target_lines)
    size_kwargs = {
        "min_w": _BBOX_MIN_WIDTH,
        "min_h": _BBOX_MIN_HEIGHT,
        "min_area": _BBOX_MIN_AREA,
        "targets_block": targets_block_text,
    }
    prompt_text = hero_bbox_spatial_prompt.format(**size_kwargs)
    refine_prompt = hero_bbox_spatial_refine_prompt.format(**size_kwargs)

    pil_image = load_image_from_url(asset_url, drive, "candidate")
    if not pil_image:
        raise ValueError(f"Failed to load hero image for spatial locate, scene {scene_id}: {asset_url}")

    conversation_history = None

    # print("\n" + "=" * 80)
    # print(f"📍 HERO BBOX SPATIAL — Scene {scene_id} — CHAT TURN 1 (locate)")
    # print("=" * 80)
    # print("📝 FORMATTED PROMPT")
    # print("-" * 80)
    # print(prompt_text)
    # print("-" * 80)

    locate_parts = [
        types.Part(text=f"Hero image URL: {asset_url}\n"),
        _pil_to_jpeg_part(pil_image),
        types.Part(text=prompt_text),
    ]
    response_text, conversation_history = invoke_gemini_multimodal(
        locate_parts,
        llm=llm,
        temperature=0.1,
        conversation_history=conversation_history,
    )

    # print("🤖 MODEL RESPONSE (turn 1 / locate)")
    # print("-" * 80)
    # print(response_text or "")
    # print("=" * 80 + "\n")

    current_boxes = parse_spatial_boxes_json(response_text, len(highlights))
    if not any(b is not None for b in current_boxes):
        return _format_highlights_xml(highlights, current_boxes)

    geometry_feedback = ""
    try:
        for review_round in range(1, _MAX_SPATIAL_REVIEW_ROUNDS + 1):
            geometry_feedback, interference, undersized = combine_bbox_validation_feedback(
                current_boxes
            )
            if interference or undersized:
                print(
                    f"⚠️ Scene {scene_id}: validation fail before review {review_round} — "
                    f"{len(interference)} interfere, {len(undersized)} undersized."
                )

            annotated_img = draw_bboxes_on_image(pil_image, current_boxes)
            review_url = _upload_bbox_review_overlay(
                annotated_img, scene_id, review_round, drive
            )
            review_text = _bbox_review_prompt_text(
                refine_prompt, review_round, geometry_feedback=geometry_feedback
            )
            review_header = (
                f"Hero image URL: {asset_url}\n"
                f"Review round {review_round} of {_MAX_SPATIAL_REVIEW_ROUNDS}. "
                "Orange boxes show your current result.\n"
            )
            if review_url:
                review_header += f"Saved review overlay URL: {review_url}\n"
            review_parts = [
                types.Part(text=review_header),
                _pil_to_jpeg_part(annotated_img),
                types.Part(text=review_text),
            ]

            review_response, conversation_history = invoke_gemini_multimodal(
                review_parts,
                llm=llm,
                temperature=0.1,
                conversation_history=conversation_history,
            )

            revised_boxes, kept = parse_spatial_refine_response(
                review_response, len(highlights), current_boxes
            )
            current_boxes = revised_boxes

            # LLM may say keep while boxes still share edges or are unreadably small.
            post_feedback, post_interference, post_undersized = combine_bbox_validation_feedback(
                current_boxes
            )
            if kept and not post_interference and not post_undersized:
                print(
                    f"✅ Scene {scene_id}: review round {review_round} — keep (exit chat)."
                )
                break
            if kept and (post_interference or post_undersized):
                geometry_feedback = post_feedback
                print(
                    f"🚫 Scene {scene_id}: review round {review_round} — model said keep, "
                    f"but validation still fails "
                    f"({len(post_interference)} interfere, {len(post_undersized)} undersized). "
                    "Forcing revise."
                )
                kept = False
            else:
                print(f"🔄 Scene {scene_id}: review round {review_round} — revise.")

            if review_round >= _MAX_SPATIAL_REVIEW_ROUNDS:
                print(
                    f"ℹ️ Scene {scene_id}: max review rounds ({_MAX_SPATIAL_REVIEW_ROUNDS}) "
                    "reached — applying deterministic separation + dropping undersized boxes."
                )
                if find_bbox_interference_pairs(current_boxes):
                    current_boxes = separate_interfering_boxes(current_boxes)
                current_boxes, dropped = drop_undersized_boxes(current_boxes)
                if dropped:
                    print(
                        f"⚠️ Scene {scene_id}: dropped undersized box index(es) {dropped}."
                    )
                final_overlay = draw_bboxes_on_image(pil_image, current_boxes)
                _upload_bbox_review_overlay(
                    final_overlay, scene_id, "final", drive
                )
    except Exception as err:
        print(
            f"Warning: Bbox in-chat review failed for scene {scene_id}, "
            f"using best boxes so far. Error: {err}"
        )
        traceback.print_exc()
        if find_bbox_interference_pairs(current_boxes):
            current_boxes = separate_interfering_boxes(current_boxes)

    # Final safety net even if review exited early with a miss.
    if find_bbox_interference_pairs(current_boxes):
        print(f"⚠️ Scene {scene_id}: final geometry pass — separating remaining conflicts.")
        current_boxes = separate_interfering_boxes(current_boxes)
    current_boxes, dropped = drop_undersized_boxes(current_boxes)
    if dropped:
        print(f"⚠️ Scene {scene_id}: final size pass — dropped box index(es) {dropped}.")
        try:
            final_overlay = draw_bboxes_on_image(pil_image, current_boxes)
            _upload_bbox_review_overlay(final_overlay, scene_id, "final", drive)
        except Exception:
            pass
    elif find_bbox_interference_pairs(current_boxes):
        try:
            final_overlay = draw_bboxes_on_image(pil_image, current_boxes)
            _upload_bbox_review_overlay(final_overlay, scene_id, "final", drive)
        except Exception:
            pass

    return _format_highlights_xml(highlights, current_boxes)


@traceable(
    metadata={
        "agent_name": "graphics_definition_v2",
        "step_name": "Hero Overlay Animation Decision",
        "function_name": "process_hero_bbox_spatial_row",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous"),
    }
)
def collect_bbox_scene_tasks(index, row, llm):
    """
    Build ordered scene specs and async tasks for bbox_highlight scenes on one row.

    :return: Tuple (ordered_specs, async_tasks, immediate_cell_value).
    """
    from agents.graphics_definition_v2.image_editing_for_layout.hero_scene_parallel import (
        PENDING,
    )

    plan_text = str(row.get(_PLAN_COLUMN, "")).strip()
    if not plan_text or plan_text == "nan" or plan_text == "-":
        return [], [], "-"
    if plan_text.startswith("ERROR:"):
        return [], [], f"ERROR: plan not ready ({plan_text[:80]})"

    scenes = parse_scenes_from_slideshow_manifest(str(row.get("slideshow_manifest", "")).strip())
    scene_by_id = {str(sc.get("id", "")).strip(): sc for sc in scenes}

    ordered_specs = []
    async_tasks = []
    any_bbox = False
    for sort_key, (scene_id, block) in enumerate(split_hero_plan_scene_blocks(plan_text), start=1):
        highlights = parse_bbox_highlights_from_plan_block(block)
        if not highlights:
            continue
        any_bbox = True
        scene = scene_by_id.get(scene_id)
        asset_url = _hero_primary_asset_url(scene) if scene else ""
        if not asset_url:
            ordered_specs.append((sort_key, scene_id, "ERROR: Missing primary_visual URL"))
            continue
        if _is_video_asset_url(asset_url):
            continue
        ordered_specs.append((sort_key, scene_id, PENDING))
        async_tasks.append(
            {
                "phase": "spatial",
                "row_index": index,
                "sort_key": sort_key,
                "scene_id": scene_id,
                "asset_url": asset_url,
                "highlights": highlights,
                "llm": llm,
            }
        )

    if not any_bbox:
        return [], [], "-"
    return ordered_specs, async_tasks, None


def worker_bbox_scene(task):
    """Run one bbox spatial scene task."""
    inner = generate_hero_bbox_coordinates_for_scene(
        scene_id=task["scene_id"],
        asset_url=task["asset_url"],
        highlights=task["highlights"],
        drive=task["drive"],
        llm=task["llm"],
    )
    return task["row_index"], task["sort_key"], task["scene_id"], inner


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
