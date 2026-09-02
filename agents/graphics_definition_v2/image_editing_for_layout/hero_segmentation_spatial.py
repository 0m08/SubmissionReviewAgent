"""Hero emphasis_style segmentation sub-agent.

For single-hero scenes whose hero_animation_plan selects animation_type =
emphasis_style, this module isolates the ONE described subject with Gemini
image segmentation, builds a full-image binary mask (white = subject, black =
background), uploads that mask PNG to Drive, and writes a hero_emphasis_overlays
cell the player uses to keep the subject sharp while the background recedes.
"""

import base64
import json
import os
import re
import traceback
import uuid
from io import BytesIO
from typing import List, Optional

import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types
from langsmith import traceable
from PIL import Image, ImageDraw, ImageEnhance
from pydantic import BaseModel, Field

from agents.graphics_asset_creation.gac_utils import upload_image_to_drive
from agents.graphics_definition_v2.aggregation_agent.aggregation_agent import (
    load_image_from_url,
)
from agents.graphics_definition_v2.image_editing_for_layout.hero_animation_decision import (
    _EMPHASIS_COLUMN,
    _PLAN_COLUMN,
    _hero_primary_asset_url,
    _is_video_asset_url,
    _scene_block,
)
from agents.graphics_definition_v2.image_editing_for_layout.hero_bbox_spatial import (
    _ANIMATION_TYPE_RE,
    _TARGET_RE,
    _extract_tag_inner,
    split_hero_plan_scene_blocks,
)
from agents.graphics_definition_v2.image_editing_for_layout.image_edit_planning import (
    parse_scenes_from_slideshow_manifest,
)
from services.llm_service import call_llm_with_retry
from services.sheets_service import save_to_sheet
from services.smart_progress_bar import SmartProgressBar

load_dotenv()

# Masks are stored alongside other hero overlay assets.
HERO_EMPHASIS_DRIVE_FOLDER_ID = "1_cYkmnvDAUPardU1FyRHEU7Puwht6qZS"

_SEGMENTATION_MODEL = "gemini-3.5-flash"

# Long side cap sent to Gemini — large hero stills often hit 503 deadline errors.
_MAX_SEGMENTATION_IMAGE_SIDE = 1536

# Retries for transient Gemini failures (503 UNAVAILABLE, deadline expired, etc.).
_SEGMENTATION_MAX_RETRIES = 5
_SEGMENTATION_RETRY_INITIAL_WAIT = 4

# In-chat review rounds after the initial locate/segment turn (1 locate + up to 3 reviews).
_MAX_SPATIAL_REVIEW_ROUNDS = 3

# Emphasis polygon vertex counts (prompt targets + post-response hard cap).
# Fewer points = stabler masks and less background bleed; emphasis is not cutout-grade.
_EMPHASIS_POLYGON_POINTS_LOCATE_MIN = 12
_EMPHASIS_POLYGON_POINTS_LOCATE_MAX = 22
_EMPHASIS_POLYGON_POINTS_REFINE_MIN = 14
_EMPHASIS_POLYGON_POINTS_REFINE_MAX = 28
_EMPHASIS_POLYGON_POINTS_HARD_MAX = 32

# Probability threshold (0–255) for turning a returned soft PNG mask into a hard binary mask (only used on the legacy base64 path).
_MASK_THRESHOLD = 127

_VALID_EMPHASIS_MODES = ("darken", "sepia")

# Player-style emphasis preview (matches human_feedback_app/player_config.py defaults).
_EMPHASIS_PREVIEW_BG_BRIGHTNESS = 0.52
_EMPHASIS_PREVIEW_VEIL_OPACITY = 0.26
_EMPHASIS_PREVIEW_SEPIA_VEIL_RGB = (92, 62, 22)
_EMPHASIS_PREVIEW_SEPIA_VEIL_OPACITY = 0.38

class _SegmentationMaskItem(BaseModel):
    box_2d: List[int] = Field(
        description="[ymin, xmin, ymax, xmax] integers normalized 0–1000",
        min_length=4,
        max_length=4,
    )
    mask: List[List[int]] = Field(
        description=(
            "Segmentation mask as a polygon: a list of [x, y] points tracing "
            "the target's outline (typically 12–28 points), each normalized 0–1000 "
            "relative to the full image"
        ),
    )
    label: str = Field(description="Short descriptive label for the target")


class _SegmentationMaskResponse(BaseModel):
    segmentation_masks: List[_SegmentationMaskItem] = Field(
        description="Exactly one mask entry for the requested target",
        min_length=1,
        max_length=1,
    )


class _SegmentationRefineResponse(BaseModel):
    action: str = Field(
        description='Use "keep" when the pass-1 mask and emphasis preview are correct; "revise" only when fixes are needed.',
    )
    segmentation_masks: Optional[List[_SegmentationMaskItem]] = Field(
        default=None,
        description="Required when action is revise. Omit or null when action is keep.",
    )


_REFINE_KEEP_ACTIONS = frozenset(
    {"keep", "pass", "approve", "ok", "accept", "no_change", "unchanged"}
)


# Polygon segmentation prompt for Gemini 3.5 Flash.
hero_segmentation_prompt = """Give the segmentation mask for the target described below in the provided image.

Target:
{target_block}

Output a JSON object with key "segmentation_masks" containing exactly one entry. The entry must contain:
- "box_2d": the 2D bounding box as [ymin, xmin, ymax, xmax] normalized 0–1000.
- "mask": the segmentation mask as a POLYGON — a list of [x, y] points tracing the target's real outline, each coordinate normalized 0–1000 relative to the full image. Use {polygon_min}–{polygon_max} points total. Place points evenly along the contour; on curves use more points, on straight edges use fewer.
- "label": a short descriptive label for the target.

Trace only the single described target. Prefer staying slightly INSIDE the true outline rather than expanding into background (slight under-segmentation is OK). Do not exceed {polygon_max} points. Output JSON only — no markdown fences.
"""

hero_segmentation_refine_prompt = """You are reviewing your own segmentation mask for an emphasis-style course animation.

You are given TWO images:
- Image 1: The original hero still (unchanged reference).
- Image 2: The emphasis PREVIEW — how this slide will look in the course player. ONLY the described target should stay sharp and full brightness; everything OUTSIDE the mask must be darkened.

Carefully inspect the emphasis preview (Image 2) against the checklist:
1. Extra mask / background bleed: Is any plain wall, floor, or empty space incorrectly sharp/bright when it should be darkened?
2. Bright islands: Are there small bright patches OUTSIDE the true target that should be part of the dark background?
3. Under-segmentation: Is any part of the described target incorrectly darkened when it should stay sharp?
4. Wrong object: Did you highlight the wrong object instead of ONLY the described target?
5. Edge quality: Only revise for blockiness if it clearly cuts off a major part of the target; minor faceting is acceptable.

If the emphasis preview is CORRECT and no checklist item fails, output ONLY:
{{"action": "keep"}}

If ANY checklist item fails, output:
{{"action": "revise", "segmentation_masks": [{{"box_2d": [ymin, xmin, ymax, xmax], "mask": [[x, y], ...], "label": "short name"}}]}}

Rules for revise:
- Exactly one entry in segmentation_masks for the described target.
- "box_2d": [ymin, xmin, ymax, xmax] normalized 0–1000
- "mask": polygon — {polygon_min}–{polygon_max} [x, y] points tracing ONLY the target outline. Fix bleed by pulling points INWARD; do not add extra points unless edges are clearly blocky.
- Prefer "keep" when bleed and target coverage are acceptable — do not revise for minor blockiness alone.
- Do NOT exceed {polygon_max} points.

Target (must match exactly):
{target_block}

Output JSON only — no markdown fences.
"""

_EMPHASIS_RE = re.compile(r"<emphasis>(.*?)</emphasis>", re.DOTALL | re.IGNORECASE)
_EMPHASIS_MODE_RE = re.compile(
    r"<emphasis_mode>\s*(.*?)\s*</emphasis_mode>",
    re.DOTALL | re.IGNORECASE,
)


def parse_emphasis_from_plan_block(block_text):
    """
    Parse the emphasis target and mode from one scene's hero_animation_plan block.

    :param block_text: Inner plan text for one scene.
    :return: Dict with target_description and emphasis_mode, or None when not emphasis_style.
    """
    animation_type = _extract_tag_inner(block_text, _ANIMATION_TYPE_RE).lower()
    if animation_type != "emphasis_style":
        return None

    inner_match = _EMPHASIS_RE.search(block_text or "")
    inner = (inner_match.group(1) if inner_match else block_text) or ""

    target = _extract_tag_inner(inner, _TARGET_RE)
    if not target or target.upper() == "N/A":
        return None

    mode = _extract_tag_inner(inner, _EMPHASIS_MODE_RE).lower()
    if mode not in _VALID_EMPHASIS_MODES:
        mode = "darken"

    return {"target_description": target, "emphasis_mode": mode}


def emphasis_cell_needs_processing(existing_text):
    """
    Return True when hero_emphasis_overlays should be (re)generated.

    Retries rows that are empty or contain any ERROR block (including
    ``---Scene ID: N---\\nERROR: ...`` which does not start with ERROR:).
    """
    existing = str(existing_text or "").strip()
    if not existing or existing.lower() == "nan" or existing == "-":
        return True
    return "ERROR:" in existing


def _emphasis_block_is_ok(block_text):
    """True when an existing emphasis scene block has a usable mask URL."""
    text = str(block_text or "").strip()
    if not text or "ERROR:" in text:
        return False
    match = re.search(r"<mask_url>(.*?)</mask_url>", text, re.DOTALL | re.IGNORECASE)
    if not match:
        return False
    url = match.group(1).strip()
    return bool(url) and url.upper() != "N/A" and not url.upper().startswith("ERROR")


def _prepare_image_for_segmentation(pil_image):
    """
    Downscale very large hero stills before the segmentation API call.

    :param pil_image: RGB PIL image at full resolution.
    :return: Tuple (image_for_api, scale) where scale is orig_max_side / sent_max_side.
    """
    w, h = pil_image.size
    max_side = max(w, h)
    if max_side <= _MAX_SEGMENTATION_IMAGE_SIDE:
        return pil_image, 1.0
    scale = _MAX_SEGMENTATION_IMAGE_SIDE / float(max_side)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = pil_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    print(
        f"Hero emphasis segmentation: downscaled {w}x{h} → {new_w}x{new_h} "
        f"(max side {_MAX_SEGMENTATION_IMAGE_SIDE}) to reduce API timeouts."
    )
    return resized, scale


def _strip_data_uri(raw):
    """
    Remove a leading data-URI prefix from a base64 mask string.

    :param raw: Raw mask string, possibly "data:image/png;base64,....".
    :return: Bare base64 payload.
    """
    text = str(raw or "").strip()
    if text.startswith("data:"):
        comma = text.find(",")
        if comma != -1:
            text = text[comma + 1:]
    return text


def _decode_mask_png(mask_b64):
    """
    Decode a base64 PNG probability mask into a grayscale PIL image.

    :param mask_b64: Base64 payload (data-URI prefix allowed).
    :return: PIL "L" image, or None on failure.
    """
    payload = _strip_data_uri(mask_b64)
    if not payload:
        return None
    try:
        raw = base64.b64decode(payload, validate=True)
    except Exception:
        return None
    if not raw.startswith(b"\x89PNG\r\n\x1a\n"):
        return None
    try:
        mask_img = Image.open(BytesIO(raw)).convert("L")
        return mask_img
    except Exception:
        return None


def _get_segmentation_client():
    """
    Google GenAI client for segmentation.

    Segmentation masks must use the Google AI API (API key). Vertex AI returnscraw <seg_*> tokens that cannot be decoded client-side (google-gemini/cookbook#798).

    :return: genai.Client or None when API key is missing.
    """
    if os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").strip().lower() in ("1", "true", "yes"):
        print(
            "⚠️ Hero emphasis segmentation: GOOGLE_GENAI_USE_VERTEXAI is enabled. "
            "Masks may be undecodable <seg_*> tokens — use API key auth instead."
        )
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        try:
            api_key = st.session_state.get("google_api_key")
        except (RuntimeError, AttributeError):
            api_key = None
    if not api_key:
        print("Hero emphasis segmentation: Missing GOOGLE_API_KEY / GEMINI_API_KEY")
        return None
    try:
        return genai.Client(api_key=api_key)
    except Exception as e:
        print(f"Hero emphasis segmentation: Failed to initialize Gemini client: {e}")
        return None


def _parse_box_2d(raw):
    """
    Normalize a model box_2d value to four ints in 0–1000.

    :param raw: List or string from the model.
    :return: [ymin, xmin, ymax, xmax] or None if invalid.
    """
    values = raw
    if isinstance(raw, str):
        values = re.findall(r"-?\d+", raw)
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


def _extract_json_payload(response_text):
    """
    Pull the JSON array/object out of a raw model response.

    :param response_text: Raw model response text.
    :return: Parsed Python object, or None.
    """
    if not response_text:
        return None
    text = str(response_text).strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    for pattern in (r"\[.*\]", r"\{.*\}"):
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                continue
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _normalize_segmentation_items(payload):
    """
    Flatten parsed JSON into a list of segmentation dicts.

    :param payload: Parsed JSON from Gemini.
    :return: List of dict items (may be empty).
    """
    if payload is None:
        return []
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if isinstance(payload, dict):
        for key in ("masks", "segmentation_masks", "items", "results"):
            nested = payload.get(key)
            if isinstance(nested, list):
                return [x for x in nested if isinstance(x, dict)]
        return [payload]
    return []


def _mask_is_seg_tokens(mask_val):
    """True when mask is an opaque seg-token string Gemini sometimes returns."""
    text = str(mask_val or "").strip()
    return text.startswith("<start_of_mask>") or text.startswith("<seg_")


def _raw_polygon_pairs(polygon):
    """
    Flatten a model polygon into (a, b) pairs in 0–1000 space.

    :param polygon: [[a,b], ...] or flat [a,b,a,b,...].
    :return: List of (float, float) pairs.
    """
    if not polygon or not isinstance(polygon, (list, tuple)):
        return []
    pairs = []
    if polygon and isinstance(polygon[0], (list, tuple)) and len(polygon[0]) >= 2:
        for pt in polygon:
            if len(pt) < 2:
                continue
            try:
                pairs.append((float(pt[0]), float(pt[1])))
            except (TypeError, ValueError):
                continue
    elif len(polygon) >= 6 and len(polygon) % 2 == 0:
        for i in range(0, len(polygon), 2):
            try:
                pairs.append((float(polygon[i]), float(polygon[i + 1])))
            except (TypeError, ValueError):
                continue
    return pairs


def _subsume_polygon_pairs(pairs, max_points):
    """
    Evenly subsample polygon vertices when the model returns too many.

    :param pairs: List of (a, b) in normalized space.
    :param max_points: Maximum vertex count to keep.
    :return: Subsampled list (order preserved).
    """
    n = len(pairs)
    if n <= max_points or max_points < 3:
        return pairs
    step = (n - 1) / float(max_points - 1)
    return [pairs[int(round(i * step))] for i in range(max_points)]


def _polygon_axis_order(pairs, box):
    """
    Decide whether polygon pairs are [x, y] or [y, x].

    :param pairs: List of (a, b) floats in 0–1000 space.
    :param box: Parsed [ymin, xmin, ymax, xmax] or None.
    :return: "xy" or "yx".
    """
    if not pairs:
        return "xy"
    parsed_box = _parse_box_2d(box)
    if not parsed_box:
        return "xy"
    ymin, xmin, ymax, xmax = parsed_box
    slack = 40

    def _in_box(x, y):
        return (xmin - slack) <= x <= (xmax + slack) and (ymin - slack) <= y <= (ymax + slack)

    xy_hits = sum(1 for a, b in pairs if _in_box(a, b))
    yx_hits = sum(1 for a, b in pairs if _in_box(b, a))
    if yx_hits > xy_hits:
        return "yx"
    return "xy"


def _polygon_points_to_pixels(polygon, image_w, image_h, box=None):
    """
    Convert a normalized polygon to pixel (x, y) tuples.

    Auto-detects [x, y] vs [y, x] using box_2d when available.

    :param polygon: Polygon from model.
    :param image_w: Image width in pixels.
    :param image_h: Image height in pixels.
    :param box: Optional box_2d [ymin, xmin, ymax, xmax] for axis detection.
    :return: List of (x, y) int tuples, or empty list.
    """
    pairs = _raw_polygon_pairs(polygon)
    if not pairs:
        return []
    orig_n = len(pairs)
    if orig_n > _EMPHASIS_POLYGON_POINTS_HARD_MAX:
        pairs = _subsume_polygon_pairs(pairs, _EMPHASIS_POLYGON_POINTS_HARD_MAX)
        print(
            f"🎯 Polygon subsampled {orig_n} → {len(pairs)} vertices "
            f"(hard max {_EMPHASIS_POLYGON_POINTS_HARD_MAX})"
        )
    order = _polygon_axis_order(pairs, box)
    pts = []
    for a, b in pairs:
        if order == "yx":
            x_n, y_n = b, a
        else:
            x_n, y_n = a, b
        x = max(0, min(image_w - 1, int(round(x_n / 1000.0 * image_w))))
        y = max(0, min(image_h - 1, int(round(y_n / 1000.0 * image_h))))
        pts.append((x, y))
    return pts, order


def _mask_from_polygon(polygon, image_w, image_h, box=None):
    """
    Rasterize a normalized polygon into a full-image grayscale mask.

    :param polygon: Polygon coordinates from the model.
    :param image_w: Image width.
    :param image_h: Image height.
    :param box: Optional box_2d used to detect [x,y] vs [y,x].
    :return: Tuple of (PIL "L" image or None, axis-order string).
    """
    converted = _polygon_points_to_pixels(polygon, image_w, image_h, box)
    if not converted:
        return None, "xy"
    pts, order = converted
    if len(pts) < 3:
        return None, order
    mask = Image.new("L", (image_w, image_h), 0)
    draw = ImageDraw.Draw(mask)
    draw.polygon(pts, fill=255)
    return mask, order


def _mask_from_base64_in_box(mask_b64, box, image_w, image_h):
    """
    Decode a box-sized base64 PNG mask and paste it onto a full-image canvas.

    :param mask_b64: Base64 PNG string.
    :param box: [ymin, xmin, ymax, xmax].
    :param image_w: Image width.
    :param image_h: Image height.
    :return: PIL "L" full-image mask or None.
    """
    parsed_box = _parse_box_2d(box)
    if not parsed_box:
        return None
    mask_img = _decode_mask_png(mask_b64)
    if mask_img is None:
        return None
    ymin, xmin, ymax, xmax = parsed_box
    px_min = int((xmin / 1000.0) * image_w)
    py_min = int((ymin / 1000.0) * image_h)
    px_max = int((xmax / 1000.0) * image_w)
    py_max = int((ymax / 1000.0) * image_h)
    box_w = max(1, px_max - px_min)
    box_h = max(1, py_max - py_min)
    resized = mask_img.resize((box_w, box_h), Image.BILINEAR)
    binary = resized.point(lambda p: 255 if p > _MASK_THRESHOLD else 0)
    full_mask = Image.new("L", (image_w, image_h), 0)
    full_mask.paste(binary, (px_min, py_min))
    return full_mask


def build_full_mask_from_segmentation(response_text, image_w, image_h):
    """
    Convert Gemini segmentation JSON into a full-image binary mask.

    Handles base64 PNG masks, polygon masks, and box-only fallback.

    :param response_text: Raw segmentation response.
    :param image_w: Hero image width in pixels.
    :param image_h: Hero image height in pixels.
    :return: Tuple of (PIL "L" mask or None, debug reason string).
    """

    payload = _extract_json_payload(response_text)
    items = _normalize_segmentation_items(payload)
    if not items:
        return None, "no_json"

    item = items[0]
    mask_val = item.get("mask")
    box = item.get("box_2d")

    if _mask_is_seg_tokens(mask_val):
        print(
            "⚠️ Gemini returned raw <seg_*> mask tokens — API did not expand to PNG. "
            "Use Google AI API key auth (not Vertex AI). See google-gemini/cookbook#798."
        )
        return None, "seg_tokens"

    # 1) Base64 PNG in mask field (must decode to a real PNG, not garbage)
    if isinstance(mask_val, str) and len(mask_val) > 80:
        full = _mask_from_base64_in_box(mask_val, box, image_w, image_h)
        if full is not None:
            return full, "base64_png"
        return None, "base64_decode_failed"

    # 2) Polygon list in mask field
    if isinstance(mask_val, (list, tuple)) and len(mask_val) >= 3:
        full, axis_order = _mask_from_polygon(mask_val, image_w, image_h, box)
        if full is not None:
            n_pts = len(_raw_polygon_pairs(mask_val))
            print(
                f"🎯 Polygon mask: {n_pts} model vertices, axis={axis_order} "
                f"([a,b] as {'[x,y]' if axis_order == 'xy' else '[y,x]'})"
            )
            return full, "polygon"
        return None, "polygon_invalid"

    return None, "no_mask_geometry"


def _safe_filename_token(value):
    """
    Make a string safe for a Drive filename token.

    :param value: Raw scene id or similar.
    :return: Underscore-safe token.
    """
    token = re.sub(r"[^\w.-]+", "_", str(value or "").strip())
    return token or "unknown"


def _pil_to_jpeg_part(pil_image):
    """Encode a PIL image as a Gemini JPEG inline Part."""
    buffered = BytesIO()
    pil_image.convert("RGB").save(buffered, format="JPEG")
    return types.Part(inline_data=types.Blob(mime_type="image/jpeg", data=buffered.getvalue()))


def _emphasis_review_prompt_text(refine_prompt, review_round, asset_url):
    """Build the review prompt for one in-chat emphasis review round."""
    if review_round <= 1:
        header = (
            f"Hero image URL: {asset_url}\n\n"
            "Image 1: original hero still (from turn 1).\n"
            "Image 2: emphasis preview (player-style composite from your latest mask).\n\n"
        )
        return header + refine_prompt
    header = (
        f"Hero image URL: {asset_url}\n\n"
        f"Review round {review_round} of {_MAX_SPATIAL_REVIEW_ROUNDS}. "
        "Image 1: original hero still. Image 2: your latest emphasis preview after the previous revision.\n\n"
    )
    return header + refine_prompt


def render_emphasis_preview(pil_rgb, mask_L, emphasis_mode="darken"):
    """
    Build a player-style emphasis preview (option A): sharp subject, darkened background.

    :param pil_rgb: Original hero still (RGB).
    :param mask_L: Full-image mask (white = subject, black = background).
    :param emphasis_mode: darken or sepia.
    :return: RGB PIL image matching course-player emphasis composite.
    """
    w, h = pil_rgb.size
    subject = mask_L.convert("L")
    bg_mask = Image.eval(subject, lambda p: 255 - p)

    dark = ImageEnhance.Brightness(pil_rgb.convert("RGB")).enhance(_EMPHASIS_PREVIEW_BG_BRIGHTNESS)
    dark_rgba = dark.convert("RGBA")
    mode = str(emphasis_mode or "darken").lower()
    if mode == "sepia":
        r, g, b = _EMPHASIS_PREVIEW_SEPIA_VEIL_RGB
        alpha = int(255 * _EMPHASIS_PREVIEW_SEPIA_VEIL_OPACITY)
        veil = Image.new("RGBA", (w, h), (r, g, b, alpha))
    else:
        alpha = int(255 * _EMPHASIS_PREVIEW_VEIL_OPACITY)
        veil = Image.new("RGBA", (w, h), (0, 0, 0, alpha))
    dark_rgba = Image.alpha_composite(dark_rgba, veil)

    out = pil_rgb.convert("RGBA")
    out.paste(dark_rgba, (0, 0), bg_mask)
    return out.convert("RGB")


def _resize_mask_to_original(full_mask, seg_w, seg_h, orig_w, orig_h):
    """
    Upscale a segmentation-space mask to original hero dimensions when needed.

    :return: PIL "L" mask at original size, or None.
    """
    if full_mask is None:
        return None
    if (orig_w, orig_h) != (seg_w, seg_h):
        return full_mask.resize((orig_w, orig_h), Image.Resampling.NEAREST)
    return full_mask


def _mask_from_segmentation_response(response_text, seg_w, seg_h):
    """
    Parse one segmentation response into a mask at segmentation resolution.

    :return: Tuple of (PIL "L" mask or None, reason string).
    """
    full_mask, mask_reason = build_full_mask_from_segmentation(response_text, seg_w, seg_h)
    return full_mask, mask_reason


def parse_emphasis_refine_response(response_text, seg_w, seg_h):
    """
    Parse pass-2 emphasis review JSON (keep vs revise).

    :return: Tuple of (mask or None, reason). reason "refine_keep" means keep pass-1 mask.
    """
    payload = _extract_json_payload(response_text)
    if isinstance(payload, dict):
        action = str(payload.get("action", "")).lower().strip()
        if action in _REFINE_KEEP_ACTIONS:
            return None, "refine_keep"
        if action == "revise":
            masks = payload.get("segmentation_masks")
            if masks:
                wrapped = json.dumps({"segmentation_masks": masks})
                return _mask_from_segmentation_response(wrapped, seg_w, seg_h)
            return None, "refine_revise_empty"
        if payload.get("segmentation_masks"):
            return _mask_from_segmentation_response(json.dumps(payload), seg_w, seg_h)
    return _mask_from_segmentation_response(response_text, seg_w, seg_h)


def _build_segmentation_thinking_config(model):
    """
    Thinking config for the segmentation model.



    :param model: Gemini model id.
    :return: types.ThinkingConfig or None.
    """
    model_id = str(model or "").lower()
    if model_id.startswith("gemini-2.5") or model_id.startswith("gemini-2-5"):
        return types.ThinkingConfig(thinking_budget=0)
    try:
        return types.ThinkingConfig(thinking_level="low")
    except Exception:
        return None


def _extract_segmentation_response_text(response):
    """Pull plain text from a Gemini generate_content response."""
    if hasattr(response, "text") and response.text:
        return response.text
    if getattr(response, "candidates", None):
        first = response.candidates[0]
        if getattr(first, "content", None) and first.content.parts:
            part = first.content.parts[0]
            if hasattr(part, "text"):
                return part.text
    return str(response)


def invoke_gemini_segmentation_chat(
    images,
    prompt_text,
    model=_SEGMENTATION_MODEL,
    response_schema=None,
    conversation_history=None,
):
    """
    One turn in a multi-turn segmentation chat.

    :param images: PIL RGB image or list of images for this user turn (optional).
    :param prompt_text: Prompt string for this turn.
    :param model: Gemini model id.
    :param response_schema: Pydantic schema for JSON output.
    :param conversation_history: Prior Content turns in this chat.
    :return: Tuple of (response_text, updated_conversation_history).
    """
    client = _get_segmentation_client()
    if client is None:
        raise ValueError("Gemini API key not available for segmentation")

    schema = response_schema or _SegmentationMaskResponse
    config_kwargs = dict(
        temperature=0.1,
        response_mime_type="application/json",
        response_schema=schema,
        max_output_tokens=8192,
    )
    thinking_config = _build_segmentation_thinking_config(model)
    if thinking_config is not None:
        config_kwargs["thinking_config"] = thinking_config
    config = types.GenerateContentConfig(**config_kwargs)

    user_parts = []
    if images is not None:
        img_list = images if isinstance(images, (list, tuple)) else [images]
        for img in img_list:
            user_parts.append(_pil_to_jpeg_part(img))
    user_parts.append(types.Part(text=prompt_text))

    contents = list(conversation_history or [])
    contents.append(types.Content(role="user", parts=user_parts))

    def _call():
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )
        return _extract_segmentation_response_text(response)

    response_text = call_llm_with_retry(
        _call,
        max_retries=_SEGMENTATION_MAX_RETRIES,
        initial_wait=_SEGMENTATION_RETRY_INITIAL_WAIT,
    )

    updated_history = list(conversation_history or [])
    updated_history.append(types.Content(role="user", parts=user_parts))
    if response_text:
        updated_history.append(types.Content(role="model", parts=[types.Part(text=response_text)]))
    return response_text, updated_history


def invoke_gemini_segmentation(images, prompt_text, model=_SEGMENTATION_MODEL, response_schema=None):
    """
    Single-turn segmentation call (backward compatible wrapper).

    :return: Response text (JSON).
    """
    response_text, _ = invoke_gemini_segmentation_chat(
        images,
        prompt_text,
        model=model,
        response_schema=response_schema,
        conversation_history=None,
    )
    return response_text


def _format_emphasis_xml(target_description, emphasis_mode, mask_url):
    """
    Build the sheet XML for one emphasis_style scene.

    :param target_description: Subject description used for segmentation.
    :param emphasis_mode: darken or sepia.
    :param mask_url: Drive URL of the uploaded mask PNG, or an ERROR string.
    :return: Inner XML string.
    """
    return (
        "<emphasis>\n"
        f"<target_description>{target_description}</target_description>\n"
        f"<emphasis_mode>{emphasis_mode}</emphasis_mode>\n"
        f"<mask_url>{mask_url}</mask_url>\n"
        "</emphasis>"
    )


@traceable(
    metadata={
        "agent_name": "graphics_definition_v2",
        "step_name": "Hero Overlay Animation Decision",
        "function_name": "generate_hero_emphasis_mask_for_scene",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous"),
    }
)
def generate_hero_emphasis_mask_for_scene(scene_id, asset_url, emphasis, drive, model=_SEGMENTATION_MODEL):
    """
    Segment the emphasis subject on one hero still and upload the mask to Drive.

    One Gemini chat per scene:
    - Turn 1: polygon segmentation on the original still
    - Turns 2–4: up to three in-chat review rounds (keep → exit; revise → update preview)

    :param scene_id: Scene id from slideshow_manifest.
    :param asset_url: Final hero image URL.
    :param emphasis: Dict with target_description and emphasis_mode.
    :param drive: Drive client for loading the image and uploading the mask.
    :param model: Gemini model id for segmentation.
    :return: Inner XML string of <emphasis> for this scene.
    """
    target_description = emphasis["target_description"]
    emphasis_mode = emphasis.get("emphasis_mode", "darken")
    scene_token = _safe_filename_token(scene_id)

    if not drive:
        raise ValueError("Drive not available; cannot upload emphasis mask.")

    pil_image = load_image_from_url(asset_url, drive, "candidate")
    if not pil_image:
        raise ValueError(
            f"Failed to load hero image for segmentation, scene {scene_id}: {asset_url}"
        )
    pil_image = pil_image.convert("RGB")
    orig_w, orig_h = pil_image.size
    seg_image, _scale = _prepare_image_for_segmentation(pil_image)
    seg_w, seg_h = seg_image.size

    prompt_text = hero_segmentation_prompt.format(
        target_block=target_description,
        polygon_min=_EMPHASIS_POLYGON_POINTS_LOCATE_MIN,
        polygon_max=_EMPHASIS_POLYGON_POINTS_LOCATE_MAX,
    )
    refine_prompt = hero_segmentation_refine_prompt.format(
        target_block=target_description,
        polygon_min=_EMPHASIS_POLYGON_POINTS_REFINE_MIN,
        polygon_max=_EMPHASIS_POLYGON_POINTS_REFINE_MAX,
    )
    conversation_history = None

    # print("\n" + "=" * 80)
    # print(f"🎯 HERO EMPHASIS SEGMENTATION — Scene {scene_id} — CHAT TURN 1 (locate)")
    # print(f"   mode={emphasis_mode}")
    # print("=" * 80)
    # print(f"🎯 TARGET: {target_description}")
    # print("-" * 80)
    # print("📝 FORMATTED PROMPT")
    # print("-" * 80)
    # print(prompt_text)
    # print("-" * 80)

    locate_response, conversation_history = invoke_gemini_segmentation_chat(
        seg_image,
        prompt_text,
        model=model,
        response_schema=_SegmentationMaskResponse,
        conversation_history=conversation_history,
    )

    # print("🤖 MODEL RESPONSE (turn 1 / locate, truncated to 800 chars)")
    # print("-" * 80)
    # print((locate_response or "")[:800])
    # print("=" * 80 + "\n")

    current_mask, final_reason = _mask_from_segmentation_response(locate_response, seg_w, seg_h)
    # print(f"🎯 Turn 1 mask build result: {final_reason}")

    if current_mask is not None:
        try:
            for review_round in range(1, _MAX_SPATIAL_REVIEW_ROUNDS + 1):
                preview_image = render_emphasis_preview(seg_image, current_mask, emphasis_mode)
                review_filename = (
                    f"emphasis_review_s{scene_token}_r{review_round}_{uuid.uuid4().hex[:8]}.png"
                )
                review_url = upload_image_to_drive(
                    preview_image,
                    review_filename,
                    drive,
                    folder_id=HERO_EMPHASIS_DRIVE_FOLDER_ID,
                )
                # if review_url:
                #     print(f"📎 Emphasis review preview (round {review_round}): {review_url}")
                # else:
                #     print(f"⚠️ Emphasis review preview upload failed (round {review_round}).")

                review_text = _emphasis_review_prompt_text(
                    refine_prompt, review_round, asset_url
                )

                # print("\n" + "=" * 80)
                # print(
                #     f"🔎 HERO EMPHASIS SEGMENTATION — Scene {scene_id} — "
                #     f"CHAT TURN {review_round + 1} (review {review_round}/{_MAX_SPATIAL_REVIEW_ROUNDS})"
                # )
                # print("=" * 80)
                # print("📝 FORMATTED PROMPT")
                # print("-" * 80)
                # print(review_text[:1200])
                # print("-" * 80)

                review_response, conversation_history = invoke_gemini_segmentation_chat(
                    [seg_image, preview_image],
                    review_text,
                    model=model,
                    response_schema=_SegmentationRefineResponse,
                    conversation_history=conversation_history,
                )

                # print(f"🤖 MODEL RESPONSE (review round {review_round}, truncated to 800 chars)")
                # print("-" * 80)
                # print((review_response or "")[:800])
                # print("=" * 80 + "\n")

                revised_mask, review_reason = parse_emphasis_refine_response(
                    review_response, seg_w, seg_h
                )
                print(f"🎯 Review round {review_round} result: {review_reason}")

                if review_reason == "refine_keep":
                    print(
                        f"✅ Scene {scene_id}: review round {review_round} — keep (exit chat)."
                    )
                    break

                if revised_mask is not None:
                    current_mask = revised_mask
                    final_reason = review_reason
                    print(f"🔄 Scene {scene_id}: review round {review_round} — revise.")
                else:
                    print(
                        f"⚠️ Scene {scene_id}: review round {review_round} revise invalid "
                        f"({review_reason}) — keeping previous mask."
                    )

                if review_round >= _MAX_SPATIAL_REVIEW_ROUNDS:
                    print(
                        f"ℹ️ Scene {scene_id}: max review rounds ({_MAX_SPATIAL_REVIEW_ROUNDS}) "
                        "reached — using latest mask as final (no further review)."
                    )
        except Exception as err:
            print(
                f"Warning: Emphasis in-chat review failed for scene {scene_id}, "
                f"using best mask so far. Error: {err}"
            )
            traceback.print_exc()

    final_mask = current_mask
    if final_mask is None:
        print(
            f"❌ Scene {scene_id}: no real segmentation mask ({final_reason}) — "
            "fallback disabled, recording ERROR."
        )
        return _format_emphasis_xml(
            target_description,
            emphasis_mode,
            f"ERROR: segmentation mask not returned ({final_reason})",
        )

    full_mask = _resize_mask_to_original(final_mask, seg_w, seg_h, orig_w, orig_h)
    print(f"🎯 Final mask source: {final_reason}")

    filename = f"hero_emphasis_mask_s{scene_token}_{uuid.uuid4().hex[:8]}.png"
    mask_url = upload_image_to_drive(
        full_mask, filename, drive, folder_id=HERO_EMPHASIS_DRIVE_FOLDER_ID
    )
    if not mask_url:
        return _format_emphasis_xml(
            target_description, emphasis_mode, "ERROR: mask upload to Drive failed"
        )

    return _format_emphasis_xml(target_description, emphasis_mode, mask_url)


def collect_emphasis_scene_tasks(index, row, model=_SEGMENTATION_MODEL):
    """
    Build ordered scene specs and async tasks for emphasis_style scenes on one row.

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

    existing_emphasis = str(row.get(_EMPHASIS_COLUMN, "")).strip()
    existing_by_scene = {}
    if existing_emphasis and existing_emphasis.lower() not in ("nan", "-"):
        existing_by_scene = dict(split_hero_plan_scene_blocks(existing_emphasis))

    ordered_specs = []
    async_tasks = []
    any_emphasis = False
    for sort_key, (scene_id, block) in enumerate(split_hero_plan_scene_blocks(plan_text), start=1):
        emphasis = parse_emphasis_from_plan_block(block)
        if not emphasis:
            continue
        any_emphasis = True
        prior = existing_by_scene.get(scene_id, "")
        if _emphasis_block_is_ok(prior):
            ordered_specs.append((sort_key, scene_id, prior))
            continue
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
                "phase": "emphasis",
                "row_index": index,
                "sort_key": sort_key,
                "scene_id": scene_id,
                "asset_url": asset_url,
                "emphasis": emphasis,
                "model": model,
            }
        )

    if not any_emphasis:
        return [], [], "-"
    return ordered_specs, async_tasks, None


def worker_emphasis_scene(task):
    """Run one emphasis segmentation scene task."""
    inner = generate_hero_emphasis_mask_for_scene(
        scene_id=task["scene_id"],
        asset_url=task["asset_url"],
        emphasis=task["emphasis"],
        drive=task["drive"],
        model=task["model"],
    )
    return task["row_index"], task["sort_key"], task["scene_id"], inner


@traceable(
    metadata={
        "agent_name": "graphics_definition_v2",
        "step_name": "Hero Overlay Animation Decision",
        "function_name": "process_hero_emphasis_segmentation_row",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous"),
    }
)
def process_hero_emphasis_segmentation_row(index, row, drive, model=_SEGMENTATION_MODEL):
    """
    Fill emphasis masks for emphasis_style scenes on one Slide Chunks row.

    :param index: DataFrame row index.
    :param row: DataFrame row object with plan and manifest.
    :param drive: Drive client for loading images and uploading masks.
    :param model: Gemini model id for segmentation.
    :return: Tuple of (index, emphasis_text).
    """
    try:
        plan_text = str(row.get(_PLAN_COLUMN, "")).strip()
        if not plan_text or plan_text == "nan" or plan_text == "-":
            return index, "-"
        if plan_text.startswith("ERROR:"):
            return index, f"ERROR: plan not ready ({plan_text[:80]})"

        scenes = parse_scenes_from_slideshow_manifest(str(row.get("slideshow_manifest", "")).strip())
        scene_by_id = {str(sc.get("id", "")).strip(): sc for sc in scenes}

        existing_emphasis = str(row.get(_EMPHASIS_COLUMN, "")).strip()
        existing_by_scene = {}
        if existing_emphasis and existing_emphasis.lower() not in ("nan", "-"):
            existing_by_scene = dict(split_hero_plan_scene_blocks(existing_emphasis))

        emphasis_blocks = []
        any_emphasis = False
        for scene_id, block in split_hero_plan_scene_blocks(plan_text):
            emphasis = parse_emphasis_from_plan_block(block)
            if not emphasis:
                continue
            any_emphasis = True
            prior = existing_by_scene.get(scene_id, "")
            if _emphasis_block_is_ok(prior):
                emphasis_blocks.append(_scene_block(scene_id, prior))
                continue
            scene = scene_by_id.get(scene_id)
            asset_url = _hero_primary_asset_url(scene) if scene else ""
            if not asset_url:
                emphasis_blocks.append(_scene_block(scene_id, "ERROR: Missing primary_visual URL"))
                continue
            if _is_video_asset_url(asset_url):
                continue
            try:
                inner = generate_hero_emphasis_mask_for_scene(
                    scene_id=scene_id,
                    asset_url=asset_url,
                    emphasis=emphasis,
                    drive=drive,
                    model=model,
                )
                emphasis_blocks.append(_scene_block(scene_id, inner))
            except Exception as scene_err:
                emphasis_blocks.append(_scene_block(scene_id, f"ERROR: {str(scene_err)}"))
                traceback.print_exc()

        if not any_emphasis:
            return index, "-"
        return index, "\n\n".join(emphasis_blocks).strip()
    except Exception as e:
        print(f"Error hero emphasis segmentation row {index}: {e}")
        traceback.print_exc()
        return index, f"ERROR: {str(e)}"


def run_hero_emphasis_segmentation_phase(ws, df, drive, model=_SEGMENTATION_MODEL, max_workers=50):
    """
    Standalone phase: write hero_emphasis_overlays for emphasis_style scenes.

    :param ws: Slide Chunks worksheet.
    :param df: Slide Chunks DataFrame (mutated in place).
    :param drive: Drive client.
    :param model: Gemini model id for segmentation.
    :param max_workers: Row-level parallel workers.
    :return: None
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if _EMPHASIS_COLUMN not in df.columns:
        df[_EMPHASIS_COLUMN] = ""

    rows_to_process = []
    for index, row in df.iterrows():
        existing = str(row.get(_EMPHASIS_COLUMN, "")).strip()
        plan_text = str(row.get(_PLAN_COLUMN, "")).strip()
        if str(row.get("Slide Type", "")).strip().lower() == "transition":
            if not existing or existing == "nan" or existing.startswith("ERROR:"):
                df.at[index, _EMPHASIS_COLUMN] = "-"
            continue
        if existing and existing != "nan" and not emphasis_cell_needs_processing(existing):
            continue
        if not plan_text or plan_text == "nan":
            continue
        rows_to_process.append((index, row))

    if not rows_to_process:
        save_to_sheet(ws, df)
        print("Hero emphasis segmentation: no rows to process.")
        return

    print(f"Hero emphasis segmentation: processing {len(rows_to_process)} row(s), max_workers={max_workers}")
    futures_map = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for index, row in rows_to_process:
            fut = executor.submit(process_hero_emphasis_segmentation_row, index, row, drive, model)
            futures_map[fut] = index

        progress = SmartProgressBar(
            total_tasks=len(futures_map),
            description="Hero emphasis masks",
            save_interval=5,
        )
        for future in as_completed(futures_map):
            index = futures_map[future]
            try:
                row_index, emphasis_text = future.result()
                df.at[row_index, _EMPHASIS_COLUMN] = emphasis_text
            except Exception as e:
                df.at[index, _EMPHASIS_COLUMN] = f"ERROR: {str(e)}"
                traceback.print_exc()
            progress.update()

    save_to_sheet(ws, df)
    print("Hero emphasis segmentation: complete.")
