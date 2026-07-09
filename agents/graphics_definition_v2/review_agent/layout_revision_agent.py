"""Human-driven slideshow layout revision for scene-based review."""

import html
import os
import re
import tempfile
import textwrap
from xml.etree import ElementTree as ET
from google.genai import types
from io import BytesIO
from PIL import Image, ImageDraw

from graphics_definition_v2_slideshow import safe_str
from human_feedback_app.backend.config import LLM_DEFAULT
from agents.graphics_definition_v2.review_agent.segmentation_based_reviser import build_segmentation_map_from_graphics
from agents.graphics_definition_v2.aggregation_agent.aggregation_agent import (
    parse_video_url_timestamps,
    convert_watch_url_to_embed_url,
    load_image_from_url,
    invoke_gemini_multimodal,
    check_file_exists_in_drive,
)
from services.helper_functions import build_video_part
from agents.graphics_definition_v2.slideshow_manifest.slideshow_manifest import (
    parse_when_vo_assigned_pairs,
    make_slot_narration_resolver_from_fgd,
    normalize_when_vo_line,
    urls_match_for_graphics_assignment,
)
from agents.graphics_definition_v2.slideshow_manifest.slideshow_video import (
    BG_COLOR,
    CANVAS_HEIGHT,
    CANVAS_WIDTH,
    SLOT_CORNER_RADIUS,
    compute_slot_rectangles,
    download_drive_image,
    download_web_image,
    is_drive_url,
    is_youtube_url,
    _draw_image_into_panel,
    _draw_missing_asset,
    _draw_panel,
    _draw_subtitle_bar,
    _draw_title_bar,
    _load_font,
    _round_pil_image,
    _scaled,
)
from human_feedback_app.backend.sheet_service import load_workbook, apply_layout_revision_to_sheet
from services.sheets_service import get_sheet_data_and_df

LAYOUT_SCENE_PREVIEW_DRIVE_FOLDER_ID = os.getenv(
    "LAYOUT_SCENE_PREVIEW_DRIVE_FOLDER_ID",
    "1an92bpldViG1KL25SPrPyC7sJ2ASYXpZ",
)

LAYOUT_REVISION_PROMPT = """You are a Slideshow Layout Revision Agent specializing in HVAC e-learning content.

Context & Concepts:

1. What is an e-learning slide?
   A slide is one unit of an online HVAC training course. It has a title, narration text (voiceover), and one or more visuals (images or video clips) that appear on screen while the learner hears that narration.

2. What is a Graphics Definition?
   The Graphics Definition (final_graphics_definition) is the authoritative list of which visual assets belong on this slide. It pairs each narration beat with an assigned asset URL in order: When Voiceover: ... then Assigned Visual: <url>. It answers *what* visuals play and *when* in the narration — not how they are arranged on the canvas.

3. What is layout?
   Layout is how those assigned visuals are composed on screen for a portion of the narration. The slideshow_manifest encodes layout as one or more scenes. Each scene has:
   - a layout template (e.g. side-by-side comparison, single hero image, grid),
   - a narration_span (which words are on screen during that composition),
   - slots that place each existing asset URL into a role (e.g. left_visual, right_visual, main_visual).

4. Where did this come from?
   Upstream agents segmented the slide narration, searched for and assigned visuals, planned layouts, and built the slideshow_manifest. The manifest is the machine-readable layout spec that video automation and the human review UI use to preview how the slide will look.

5. Why are we here?
   A human reviewer is checking this slide in a scene-based layout view. They left natural-language feedback describing layout changes for one or more scenes — for example, swapping which visual appears on the left vs right, or changing from a full-screen layout to a side-by-side comparison. Your job is to understand that feedback and produce a structured revision plan that downstream code applies to the manifest.

Your responsibility is layout composition: template choice, slot-to-URL assignment, and narration span boundaries for scenes. The reviewer is giving slide-level feedback in the layout review UI and expects concrete layout changes. Interpret their request in the most reasonable layout terms using the assigned assets already on this slide (URLs from final_graphics_definition).

<layout_template_library>
Allowed templates and slot roles:

1) single_visual_hero — one dominant visual for the scene (slot: primary_visual)
2) two_item_split_comparison — two visuals side by side (slots: left_visual, right_visual)
3) multi_panel_grid — three or four equal panels (slots: panel_1, panel_2, panel_3, optional panel_4 — consecutive, no gaps)
4) main_plus_supporting_inset — one large main visual plus one smaller inset (slots: main_visual, inset_visual)
</layout_template_library>

Inputs:

<course_information>
Course name: {course_name}
Target audience: {target_audience}
Topic name: {topic_name}
Subtopic name: {subtopic_name}
</course_information>

<slide_information>
Slide ID: {slide_id}
Slide Type: {slide_type}
Slide title: {slide_title}
Slide content: {slide_content}
</slide_information>

<final_graphics_definition>
{final_graphics_definition}
</final_graphics_definition>

<current_slideshow_manifest>

{current_slideshow_manifest}

</current_slideshow_manifest>

This is the scene ID that the human reviewer is providing feedback for:
<target_scene_id>
{target_scene_id}
</target_scene_id>

This is the human reviewer's feedback for the target scene:
<human_layout_feedback>
{human_layout_feedback}
</human_layout_feedback>

Instructions:

1. Scope and Responsibility
   - Convert <human_layout_feedback> (which targets <target_scene_id>) into layout operations.
   - Read the full <current_slideshow_manifest> for slide-wide context.
   - You only need to generate operations for the target scene <target_scene_id>.
   - If a template change in the target scene reduces the slot count, any "displaced" visuals will be automatically moved into new single_visual_hero scenes inserted immediately after the revised scene. Our backend code handles this automatically; you do not need to generate operations for creating those new scenes.
   - Narration spans for any new or modified scenes are recomputed automatically based on the final_graphics_definition; you do not need to specify narration spans.
   - Output an operation plan only; downstream code applies it and writes the updated manifest.

2. Interpreting Reviewer Language
   - Read the feedback in context of the target scene, manifest, final graphics definition, and multimodal previews.
   - Resolve informal references to visuals (e.g. "the diagram", "the clip") to concrete slot roles and asset URLs in the target scene.
   - Infer the intended layout change for the target scene: slot swaps, role reassignment, or template change.
   - Choose the allowed template from the layout library that best matches the reviewer's described composition.

3. Allowed Action Types and Operation Tags
   Every <operation> block uses the same XML tags. Fill only the tags that apply to the chosen action_type; leave all other tags empty.

   - SWAP_SLOTS — fill: action_type, scene_id, role_a, role_b, reason
   - REASSIGN_SLOT — fill: action_type, scene_id, role, asset_url, reason
   - SET_SCENE_TEMPLATE — fill: action_type, scene_id, new_template, slot_assignments, reason
     For slot_assignments, write one line per slot: role_name | exact_asset_url

4. Manifest Rules (validate your plan against these)
   - Every distinct URL in final_graphics_definition appears exactly once in the manifest.
   - Each scene uses one allowed template; slot count and role names match that template.
   - Scene ids are numeric strings 1, 2, 3, ... in narration order with no gaps.
   - Total slots across all scenes equals the number of visuals on the slide.

Output format (strict):

<output>

<evaluation_breakdown>

Use this section as a structured reasoning and scratchpad space for analyzing the feedback and before writing the layout revision plan. Provide the following sections:

1. Slide Understanding
- State in your own words what this slide is about and what the learner should take away.
- Use the full slide content to resolve context, continuation phrases, title-only narration, or implied meaning.

2. Assigned Asset Review
- List each assigned asset URL on this slide and briefly describe what is visibly shown in that asset (image or video).
- Focus only on what is actually visible in the multimodal previews; do not assume unseen details.
- Note the paired When VO narration for each asset where relevant.

3. Current Layout Analysis
- For each scene in <current_slideshow_manifest>, describe the current template, narration span, and which URL occupies each slot role.

4. Human Feedback Interpretation
- Summarize what layout changes the human reviewer is requesting.
- Identify which scene(s), slot roles, templates, or visuals their feedback refers to.
- Resolve any informal references (e.g. "the diagram", "the clip", "scene 2") to concrete scene ids, roles, and asset URLs.

5. Mapping to Layout Operations
- Analyze which predefined action types are needed to address the feedback: SWAP_SLOTS, REASSIGN_SLOT, or SET_SCENE_TEMPLATE.
- Explain why each chosen action is correct and which scene(s) it affects.
- Confirm the planned result preserves manifest rules: 1:1 URL mapping, valid templates, scene order, and total slot count.

6. Additional Analysis
- Note any additional observations, tradeoffs, edge cases, or analysis that helps you arrive at the correct revision plan.
- It is acceptable for this section to be detailed or verbose if needed for correctness.

</evaluation_breakdown>

(Based on your above evaluation, provide the output strictly in the following format.)

<layout_revision_plan>

<operations>

<operation>

<action_type>
SWAP_SLOTS | REASSIGN_SLOT | SET_SCENE_TEMPLATE
</action_type>

<scene_id>
(Numeric scene id from the manifest, e.g. 1, 2, 3.)
</scene_id>

<role_a>
(Fill only for SWAP_SLOTS. First slot role to swap, e.g. left_visual. Otherwise leave empty.)
</role_a>

<role_b>
(Fill only for SWAP_SLOTS. Second slot role to swap, e.g. right_visual. Otherwise leave empty.)
</role_b>

<role>
(Fill only for REASSIGN_SLOT. Target slot role, e.g. main_visual. Otherwise leave empty.)
</role>

<asset_url>
(Fill only for REASSIGN_SLOT. Exact asset URL copied from final_graphics_definition. Otherwise leave empty.)
</asset_url>

<new_template>
(Fill only for SET_SCENE_TEMPLATE. One of: single_visual_hero, two_item_split_comparison, multi_panel_grid, main_plus_supporting_inset. Otherwise leave empty.)
</new_template>

<slot_assignments>
(Fill only for SET_SCENE_TEMPLATE. One complete slot mapping for this scene after the template change. One line per slot in the form:
role_name | exact_asset_url
Otherwise leave empty.)
</slot_assignments>

<reason>
(Concise explanation explaining why this operation is needed.)
</reason>

</operation>

Repeat one <operation> block per required change so that all the human feedback is addressed.

</operations>

</layout_revision_plan>

</output>

(Use this exact XML format while providing your output. Do not provide any additional text outside the <output> block.)
"""


KNOWN_LAYOUT_ACTIONS = (
    "SWAP_SLOTS",
    "REASSIGN_SLOT",
    "SET_SCENE_TEMPLATE",
)

_SCENE_TEMPLATE_ALIASES = {
    "single_hero": "single_visual_hero",
    "hero": "single_visual_hero",
    "two_item_split": "two_item_split_comparison",
    "split_comparison": "two_item_split_comparison",
    "multi_panel": "multi_panel_grid",
    "grid": "multi_panel_grid",
    "main_plus_inset": "main_plus_supporting_inset",
    "main_plus_supporting": "main_plus_supporting_inset",
    "main_visual_plus_inset": "main_plus_supporting_inset",
}

ALLOWED_LAYOUT_TEMPLATES = (
    "single_visual_hero",
    "two_item_split_comparison",
    "multi_panel_grid",
    "main_plus_supporting_inset",
)


def _normalize_scene_template(template):
    """
    Map manifest template strings to one of the four canonical layout ids.

    :param template: Raw template name from manifest or LLM output
    :return: Canonical template id
    """
    t = re.sub(r"[\s\-]+", "_", (template or "").strip().lower())
    return _SCENE_TEMPLATE_ALIASES.get(t, t)


def _extract_template_name(raw):
    """
    Extract a canonical layout template id from LLM output text.

    :param raw: Raw new_template tag text
    :return: Canonical template id or empty string
    """
    text = (raw or "").strip().lower()
    if not text or text.startswith("("):
        for alias, canonical in _SCENE_TEMPLATE_ALIASES.items():
            if alias in text or canonical in text:
                return canonical
        return ""
    normalized = _normalize_scene_template(text)
    if normalized in ALLOWED_LAYOUT_TEMPLATES:
        return normalized
    for template in ALLOWED_LAYOUT_TEMPLATES:
        if template in text:
            return template
    return normalized


def _normalize_action_type(raw):
    """
    Extract one known layout action type from LLM output text.

    :param raw: Raw action_type tag text
    :return: Canonical action type or empty string
    """
    text = (raw or "").upper()
    for action in KNOWN_LAYOUT_ACTIONS:
        if action in text:
            return action
    return ""


def _normalize_scene_id(raw):
    """
    Extract a numeric scene id from LLM output text.

    :param raw: Raw scene_id tag text
    :return: Scene id string or empty string
    """
    text = (raw or "").strip()
    if not text:
        return ""
    match = re.search(r"\b(\d+)\b", text)
    return match.group(1) if match else ""


def _prepare_asset_url(url):
    """
    Normalize an asset URL for comparison and manifest writes.

    :param url: Raw asset URL string
    :return: Stripped, entity-decoded URL
    """
    return html.unescape((url or "").strip())


def _asset_urls_match(url_a, url_b):
    """
    Return True when two asset URLs refer to the same graphics asset.

    :param url_a: First asset URL
    :param url_b: Second asset URL
    :return: True if the URLs should be treated as equivalent
    """
    return urls_match_for_graphics_assignment(
        _prepare_asset_url(url_a),
        _prepare_asset_url(url_b),
    )


def _resolve_asset_url(asset_url, existing_urls):
    """
    Prefer an existing manifest/FGD URL when an assignment matches it.

    :param asset_url: Parsed assignment URL
    :param existing_urls: URLs already present in the scene before revision
    :return: Canonical URL string to write into the manifest
    """
    prepared = _prepare_asset_url(asset_url)
    for existing in existing_urls:
        if _asset_urls_match(prepared, existing):
            return existing
    return prepared


def _looks_like_asset_url(value):
    """
    Return True when a parsed value looks like an asset URL rather than prompt guidance.

    :param value: Candidate asset URL string
    :return: True if value resembles a URL
    """
    text = (value or "").strip().lower()
    if not text or text.startswith("("):
        return False
    return text.startswith("http") or "drive.google.com" in text or "youtu" in text


def _parse_slot_assignments(raw):
    """
    Parse slot assignment lines from SET_SCENE_TEMPLATE operations.

    :param raw: Raw slot_assignments tag text
    :return: List of role/asset assignment dicts
    """
    assignments = []
    for line in (raw or "").splitlines():
        line = line.strip()
        if not line or line.startswith("(") or "|" not in line:
            continue
        role_name, asset_url = line.split("|", 1)
        role_name = role_name.strip()
        asset_url = asset_url.strip()
        asset_url = _prepare_asset_url(asset_url)
        if not role_name or not _looks_like_asset_url(asset_url):
            continue
        assignments.append({"role": role_name, "asset": asset_url})
    return assignments


def _prepare_manifest_xml_text(manifest_xml):
    """
    Normalize sheet-exported manifest XML for parsing.

    :param manifest_xml: Raw slideshow_manifest cell value
    :return: Wrapped, entity-safe manifest XML string or empty string
    """
    text = (manifest_xml or "").strip()
    if not text or text == "nan" or text.startswith("ERROR:"):
        return ""
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]
    text = re.sub(r'""([^"<>]*)""', r'"\1"', text)
    if "<slideshow_manifest" not in text.lower():
        text = f"<slideshow_manifest>\n{text}\n</slideshow_manifest>"
    text = re.sub(r"&(?!(amp|lt|gt|quot|apos|#\d+|#x[0-9A-Fa-f]+);)", "&amp;", text)
    return text


def _load_manifest_root(manifest_xml):
    """
    Parse manifest XML into an ElementTree root.

    :param manifest_xml: Raw slideshow_manifest cell value
    :return: Parsed XML root element or None
    """
    prepared = _prepare_manifest_xml_text(manifest_xml)
    if not prepared:
        return None
    try:
        return ET.fromstring(prepared)
    except Exception:
        return None


def _serialize_manifest_root(root):
    """
    Serialize a manifest XML root back to the inner scene XML format used in sheets.

    :param root: Parsed manifest XML root
    :return: Serialized manifest XML string
    """
    tag = (root.tag or "").lower()
    if tag.endswith("slideshow_manifest"):
        chunks = []
        for child in list(root):
            chunks.append(ET.tostring(child, encoding="unicode"))
        return "\n\n".join(chunks).strip()
    return ET.tostring(root, encoding="unicode").strip()


def _manifest_scenes(root):
    """
    Return scene elements from a parsed manifest root.

    :param root: Parsed manifest XML root
    :return: List of scene XML elements
    """
    tag = (root.tag or "").lower()
    if tag.endswith("slideshow_manifest"):
        return root.findall("scene")
    if tag.endswith("scene"):
        return [root]
    return []


def _find_scene_element(root, scene_id):
    """
    Locate a scene element by id or numeric position.

    :param root: Parsed manifest XML root
    :param scene_id: Target scene id string
    :return: Matching scene element or None
    """
    scene_id = (scene_id or "").strip()
    if not scene_id:
        return None
    for scene_el in _manifest_scenes(root):
        if (scene_el.get("id") or "").strip() == scene_id:
            return scene_el
    try:
        scene_idx = int(scene_id) - 1
        scenes = _manifest_scenes(root)
        if 0 <= scene_idx < len(scenes):
            return scenes[scene_idx]
    except ValueError:
        pass
    return None


def _renumber_scene_ids(root):
    """
    Renumber scene ids to 1..N in narration order.

    :param root: Parsed manifest XML root
    :return: None
    """
    for idx, scene_el in enumerate(_manifest_scenes(root), start=1):
        scene_el.set("id", str(idx))


def _insert_hero_scenes_for_displaced(root, displaced_urls, narration_span="", after_scene_el=None):
    """
    Create single-visual hero scenes for assets displaced by a template change.

    :param root: Parsed manifest XML root
    :param displaced_urls: Asset URLs that need a new scene
    :param narration_span: Narration text to copy into new scenes
    :param after_scene_el: Insert new scenes immediately after this scene element
    :return: None
    """
    container = root if (root.tag or "").lower().endswith("slideshow_manifest") else root
    if after_scene_el is not None:
        try:
            insert_at = list(container).index(after_scene_el) + 1
        except ValueError:
            insert_at = len(container)
    else:
        insert_at = len(container)

    for url in displaced_urls:
        if not url:
            continue
        scene_el = ET.Element("scene")
        scene_el.set("template", "single_visual_hero")
        narr_el = ET.SubElement(scene_el, "narration_span")
        narr_el.text = narration_span or ""
        slot_el = ET.SubElement(scene_el, "slot")
        slot_el.set("role", "primary_visual")
        slot_el.set("asset", url)
        container.insert(insert_at, scene_el)
        insert_at += 1


def _primary_role_for_template(template):
    """
    Return the first slot role for a canonical layout template.

    :param template: Canonical template id
    :return: Primary slot role name
    """
    return {
        "single_visual_hero": "primary_visual",
        "two_item_split_comparison": "left_visual",
        "multi_panel_grid": "panel_1",
        "main_plus_supporting_inset": "main_visual",
    }.get(template, "primary_visual")


def parse_scenes_from_manifest(manifest_xml):
    """
    Parse the slideshow_manifest XML text into a list of scene dictionaries.

    :param manifest_xml: slideshow_manifest XML string
    :return: List of scene dictionaries
    """
    root = _load_manifest_root(manifest_xml)
    if root is None:
        return []

    tag = (root.tag or "").lower()
    if tag.endswith("slideshow_manifest"):
        scene_els = root.findall("scene")
    elif tag.endswith("scene"):
        scene_els = [root]
    else:
        return []

    scenes = []
    for idx, scene_el in enumerate(scene_els, start=1):
        narr_el = scene_el.find("narration_span")
        narration = ""
        if narr_el is not None and narr_el.text:
            narration = " ".join(narr_el.text.split())
        slots = [
            {
                "role": (slot_el.get("role") or "").strip(),
                "asset": (slot_el.get("asset") or "").strip(),
            }
            for slot_el in scene_el.findall("slot")
        ]
        scenes.append({
            "id": (scene_el.get("id") or str(idx)).strip() or str(idx),
            "template": (scene_el.get("template") or "").strip(),
            "narration": narration,
            "slots": slots
        })
    return scenes


def parse_layout_revision_operations(response_text):
    """
    Parse the XML output from the LLM into a structured list of layout operations.

    :param response_text: LLM layout revision plan XML text
    :return: List of parsed operation dictionaries
    """
    text = response_text or ""
    operations = []
    for block in re.findall(r"<operation>(.*?)</operation>", text, re.DOTALL | re.IGNORECASE):
        def tag(name):
            match = re.search(rf"<{name}>\s*(.*?)\s*</{name}>", block, re.DOTALL | re.IGNORECASE)
            return match.group(1).strip() if match else ""

        action_type = _normalize_action_type(tag("action_type"))
        if not action_type:
            continue

        scene_id = _normalize_scene_id(tag("scene_id"))
        slot_assignments = _parse_slot_assignments(tag("slot_assignments"))
        asset_url = tag("asset_url").strip()
        if asset_url and not _looks_like_asset_url(asset_url):
            asset_url = ""

        operations.append({
            "action_type": action_type,
            "scene_id": scene_id,
            "role_a": tag("role_a"),
            "role_b": tag("role_b"),
            "role": tag("role"),
            "asset_url": asset_url,
            "new_template": _extract_template_name(tag("new_template")),
            "slot_assignments": slot_assignments,
            "narration_span": tag("narration_span"),
            "reason": tag("reason")
        })
    return operations


def recompute_narration_spans_from_fgd(root, final_graphics_definition):
    """
    Recompute the narration text for all scenes in the manifest root based on the visuals assigned to each scene and their When VO text in FGD.

    :param root: Parsed manifest XML root
    :param final_graphics_definition: final graphics definition text
    :return: None
    """
    if not final_graphics_definition:
        return

    try:
        steps_by_segment = build_segmentation_map_from_graphics(final_graphics_definition)
    except Exception:
        steps_by_segment = {}

    # Build a flat list of steps in order
    flat_steps = []
    for seg_num in sorted(steps_by_segment.keys(), key=int):
        flat_steps.extend(steps_by_segment[seg_num])

    # To map slot asset to step voiceover, use URL matching (with Drive ID normalization)
    def urls_match(url_a, url_b):
        return _asset_urls_match(url_a, url_b)

    # For each scene in root, gather the steps whose asset matches any slot asset
    for scene in root.findall(".//scene"):
        scene_voiceover_parts = []
        # Let's find matches in flat_steps
        for step in flat_steps:
            step_asset = (step.get("asset") or "").strip()
            # Check if this step's asset is in any of this scene's slots
            matched = False
            for slot in scene.findall(".//slot"):
                slot_asset = (slot.get("asset") or "").strip()
                if urls_match(step_asset, slot_asset):
                    matched = True
                    break
            if matched:
                vo = (step.get("voiceover_part") or "").strip()
                if vo and vo not in scene_voiceover_parts:
                    scene_voiceover_parts.append(vo)

        if scene_voiceover_parts:
            # Join them together
            joined_vo = " ".join(scene_voiceover_parts).strip()
            # Clean up double spaces
            joined_vo = re.sub(r'\s+', ' ', joined_vo)
            # Find the narration_span element, or create it if missing, and update text
            narr_el = scene.find("narration_span")
            if narr_el is not None:
                narr_el.text = joined_vo
            else:
                new_narr = ET.SubElement(scene, "narration_span")
                new_narr.text = joined_vo


def apply_layout_revision_operations(manifest_xml, operations, final_graphics_definition=None):
    """
    Apply parsed layout operations sequentially to update the manifest XML.

    :param manifest_xml: original manifest XML string
    :param operations: list of operation dictionaries to apply
    :param final_graphics_definition: final graphics definition text
    :return: Updated slideshow_manifest XML string
    """
    root = _load_manifest_root(manifest_xml)
    if root is None:
        print("[layout_revision] Failed to parse manifest XML for revision")
        return manifest_xml or ""

    for op in operations:
        action_type = _normalize_action_type(op.get("action_type", ""))
        scene_id = _normalize_scene_id(op.get("scene_id", ""))
        if not action_type or not scene_id:
            continue

        scene_el = _find_scene_element(root, scene_id)
        if scene_el is None:
            print(f"[layout_revision] Warning: target scene '{scene_id}' not found in manifest")
            continue

        if action_type == "SWAP_SLOTS":
            role_a = (op.get("role_a") or "").strip()
            role_b = (op.get("role_b") or "").strip()
            slot_a, slot_b = None, None
            for slot in scene_el.findall("slot"):
                role = (slot.get("role") or "").strip()
                if role == role_a:
                    slot_a = slot
                elif role == role_b:
                    slot_b = slot
            if slot_a is not None and slot_b is not None:
                asset_a = slot_a.get("asset") or ""
                asset_b = slot_b.get("asset") or ""
                slot_a.set("asset", asset_b)
                slot_b.set("asset", asset_a)
                print(f"[layout_revision] Swapped slot '{role_a}' and '{role_b}' in scene '{scene_id}'")
            else:
                print(f"[layout_revision] SWAP_SLOTS failed: '{role_a}' or '{role_b}' slot not found")

        elif action_type == "REASSIGN_SLOT":
            role_name = (op.get("role") or "").strip()
            asset_url = (op.get("asset_url") or "").strip()
            if not role_name or not asset_url:
                continue
            slot_found = None
            for slot in scene_el.findall("slot"):
                if (slot.get("role") or "").strip() == role_name:
                    slot_found = slot
                    break
            if slot_found is not None:
                slot_found.set("asset", asset_url)
                print(f"[layout_revision] Reassigned slot '{role_name}' to '{asset_url}' in scene '{scene_id}'")
            else:
                slot_el = ET.SubElement(scene_el, "slot")
                slot_el.set("role", role_name)
                slot_el.set("asset", asset_url)
                print(f"[layout_revision] Created and assigned slot '{role_name}' to '{asset_url}' in scene '{scene_id}'")

        elif action_type == "SET_SCENE_TEMPLATE":
            new_template = _extract_template_name(op.get("new_template", ""))
            assignments = op.get("slot_assignments") or []
            if not new_template:
                print(f"[layout_revision] SET_SCENE_TEMPLATE skipped for scene '{scene_id}': missing template")
                continue

            old_urls = []
            source_narration = ""
            narr_el = scene_el.find("narration_span")
            if narr_el is not None and narr_el.text:
                source_narration = narr_el.text
            for slot in scene_el.findall("slot"):
                url = (slot.get("asset") or "").strip()
                if url:
                    old_urls.append(url)

            scene_el.set("template", new_template)

            assignments = list(assignments)
            displaced = []
            if assignments:
                displaced = [
                    url for url in old_urls
                    if not any(
                        _asset_urls_match(url, assignment.get("asset"))
                        for assignment in assignments
                        if assignment.get("asset")
                    )
                ]
            elif old_urls:
                assignments = [{
                    "role": _primary_role_for_template(new_template),
                    "asset": old_urls[0],
                }]
                displaced = old_urls[1:]

            for slot in list(scene_el.findall("slot")):
                scene_el.remove(slot)

            for assignment in assignments:
                role = (assignment.get("role") or "").strip()
                asset = _resolve_asset_url(assignment.get("asset") or "", old_urls)
                if not role or not asset:
                    continue
                slot_el = ET.SubElement(scene_el, "slot")
                slot_el.set("role", role)
                slot_el.set("asset", asset)

            if displaced:
                _insert_hero_scenes_for_displaced(root, displaced, source_narration, after_scene_el=scene_el)
                print(f"[layout_revision] Moved {len(displaced)} displaced asset(s) into new hero scene(s) after scene '{scene_id}'")

            print(f"[layout_revision] Updated template to '{new_template}' and set slots for scene '{scene_id}'")

    if final_graphics_definition:
        recompute_narration_spans_from_fgd(root, final_graphics_definition)

    _renumber_scene_ids(root)
    try:
        return _serialize_manifest_root(root)
    except Exception as e:
        print(f"[layout_revision] Serialization failed: {e}")
        return manifest_xml


_MULTIMODAL_SECTION_RULE = "----------------------------------------------------------------"


def _draw_video_placeholder_with_url(img, x, y, w, h, video_url):
    """
    Draw a video-slot placeholder panel with the YouTube URL for agent scene previews.

    :param img: RGBA canvas image to draw onto
    :param x: Panel left coordinate
    :param y: Panel top coordinate
    :param w: Panel width
    :param h: Panel height
    :param video_url: YouTube URL to display inside the placeholder
    :return: None
    """
    radius = _scaled(SLOT_CORNER_RADIUS, minimum=8)
    pad = max(12, _scaled(20, minimum=8))
    placeholder = Image.new("RGB", (w, h), (22, 28, 36))
    draw = ImageDraw.Draw(placeholder)
    title_font = _load_font(max(14, _scaled(22)), bold=True)
    body_font = _load_font(max(12, _scaled(18)), bold=False)
    title = "Video clip"
    draw.text((pad, pad), title, font=title_font, fill=(200, 210, 225))
    title_bbox = title_font.getbbox(title)
    url_y = pad + (title_bbox[3] - title_bbox[1]) + max(8, _scaled(12, minimum=6))
    url_text = (video_url or "").strip() or "YouTube URL unavailable"
    max_width = max(40, w - 2 * pad)
    wrapped = textwrap.wrap(url_text, width=max(24, max_width // 8))
    if len(wrapped) > 6:
        wrapped = wrapped[:5] + [wrapped[5][:40] + "…"]
    line_h = max(16, _scaled(22, minimum=14))
    for i, line in enumerate(wrapped):
        draw.text((pad, url_y + i * line_h), line, font=body_font, fill=(140, 155, 175))
    rounded = _round_pil_image(placeholder, radius)
    img.alpha_composite(rounded, (x, y))


def _load_slot_asset_for_agent_preview(asset_url, drive):
    """
    Resolve a slot asset for static agent scene previews without downloading YouTube clips.

    :param asset_url: Asset URL assigned to the slot
    :param drive: Google Drive client for image loading
    :return: Dictionary containing the kind of asset, the image if it was loaded, and the asset URL
    """
    if not asset_url:
        return {"kind": "missing", "image": None, "url": ""}
    if is_youtube_url(asset_url):
        return {"kind": "video_url", "image": None, "url": asset_url.strip()}
    if is_drive_url(asset_url):
        img = download_drive_image(asset_url, drive)
        if img:
            return {"kind": "image", "image": img, "url": asset_url}
        return {"kind": "missing", "image": None, "url": asset_url}
    img = download_web_image(asset_url)
    if img:
        return {"kind": "image", "image": img, "url": asset_url}
    return {"kind": "missing", "image": None, "url": asset_url}


def _make_agent_scene_preview_image(slide_title, slot_rects, asset_results, narration):
    """
    Bake a static scene preview image for LLM multimodal input.

    :param slide_title: Slide title shown in the scene title bar
    :param slot_rects: Ordered list of slot rectangle tuples
    :param asset_results: Loaded asset dicts per slot
    :param narration: Narration text for the subtitle bar
    :return: Static scene preview image or None
    """
    img = Image.new("RGBA", (CANVAS_WIDTH, CANVAS_HEIGHT), BG_COLOR + (255,))
    _draw_title_bar(img, slide_title)
    for (x, y, w, h) in slot_rects:
        _draw_panel(img, x, y, w, h)
    for (rect, asset) in zip(slot_rects, asset_results):
        if not rect:
            continue
        x, y, w, h = rect
        if not asset:
            _draw_missing_asset(img, x, y, w, h)
            continue
        kind = asset.get("kind")
        if kind == "image" and asset.get("image") is not None:
            _draw_image_into_panel(img, asset["image"], x, y, w, h)
        elif kind == "video_url":
            _draw_video_placeholder_with_url(img, x, y, w, h, asset.get("url", ""))
        else:
            _draw_missing_asset(img, x, y, w, h)
    _draw_subtitle_bar(img, narration)
    return img.convert("RGB")


def make_agent_scene_preview_image(slide_title, scene, drive, template_normalizer=None):
    """
    Build a composed scene screenshot for layout-revision agent multimodal input.

    :param slide_title: Slide title shown in the scene title bar
    :param scene: Parsed scene dict with template, narration, and slots
    :param drive: Google Drive client for image assets
    :param template_normalizer: Callable to map template string to canonical id
    :return: Composed scene screenshot image or None
    """
    template = (scene.get("template") or "single_visual_hero").strip()
    if template_normalizer:
        template = template_normalizer(template)
    else:
        template = re.sub(r"[\s\-]+", "_", template.lower())
    slots = scene.get("slots") or []
    if not slots:
        return None
    narration = scene.get("narration") or ""
    rects = compute_slot_rectangles(template, slots)
    asset_results = [
        _load_slot_asset_for_agent_preview((slot.get("asset") or "").strip(), drive)
        for slot in slots
    ]
    return _make_agent_scene_preview_image(slide_title, rects, asset_results, narration)


def _append_pil_jpeg_part(parts, pil_image):
    """
    Append a PIL image as a JPEG inline multimodal part.

    :param parts: Mutable list of Gemini Part objects
    :param pil_image: PIL image to encode, or None to skip
    :return: None
    """
    if pil_image is None:
        return
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG", quality=85)
    parts.append(
        types.Part(
            inline_data=types.Blob(mime_type="image/jpeg", data=buffered.getvalue())
        )
    )


def _scene_preview_drive_filename(slide_id, scene_id):
    """
    Build a stable Drive file name for a composed scene preview JPEG.

    :param slide_id: Slide identifier used in layout revision jobs
    :param scene_id: Scene id from the slideshow manifest
    :return: Stable Drive file name for the composed scene preview JPEG
    """
    safe_slide = re.sub(r"[^\w\-]+", "_", (slide_id or "slide").strip())
    safe_scene = re.sub(r"[^\w\-]+", "_", str(scene_id or "0").strip())
    return f"layout_scene_preview_{safe_slide}_scene_{safe_scene}.jpg"


def _upload_scene_preview_to_drive(pil_image, drive, folder_id, filename):
    """
    Upload a scene preview JPEG to Drive, replacing any same-named file in the folder.

    :param pil_image: PIL image to upload
    :param drive: Google Drive client
    :param folder_id: Destination Drive folder id
    :param filename: File name to use in Drive
    :return: Drive URL of the uploaded scene preview or None
    """
    if pil_image is None or not drive or not folder_id:
        return None
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            pil_image.save(tmp, format="JPEG", quality=85)
            tmp_path = tmp.name
        existing = check_file_exists_in_drive(drive, folder_id, filename)
        if existing:
            file_drive = drive.CreateFile({"id": existing["id"]})
        else:
            file_drive = drive.CreateFile({
                "title": filename,
                "parents": [{"id": folder_id}],
            })
        file_drive.SetContentFile(tmp_path)
        file_drive.Upload()
        file_id = file_drive.get("id")
        if file_id:
            drive_url = f"https://drive.google.com/file/d/{file_id}/view"
            print(f"[layout_revision] Uploaded scene preview to Drive: {drive_url}")
            return drive_url
    except Exception as e:
        print(f"[layout_revision] Failed to upload scene preview '{filename}' to Drive: {e}")
    finally:
        if tmp_path:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
    return None


def _append_scene_preview_part_from_drive(parts, drive_url, drive, filename):
    """
    Load a scene preview from Drive and append it as a JPEG multimodal part.

    :param parts: Mutable list of Gemini Part objects
    :param drive_url: Google Drive view URL for the preview image
    :param drive: Google Drive client
    :param filename: File name used for load logging
    :return: True if the scene preview was loaded and appended, False otherwise
    """
    if not drive_url:
        return False
    loaded = load_image_from_url(drive_url, drive, filename)
    if loaded is None:
        return False
    _append_pil_jpeg_part(parts, loaded.convert("RGB"))
    return True


def _append_scene_preview_part(parts, pil_image, drive, filename):
    """
    Save a scene preview to Drive and append the Drive-loaded image to multimodal parts.

    :param parts: Mutable list of Gemini Part objects
    :param pil_image: Composed scene preview image
    :param drive: Google Drive client
    :param filename: Drive file name for the preview JPEG
    :return: Drive URL of the uploaded scene preview or None
    """
    if pil_image is None:
        return None
    drive_url = _upload_scene_preview_to_drive(
        pil_image,
        drive,
        LAYOUT_SCENE_PREVIEW_DRIVE_FOLDER_ID,
        filename,
    )
    if drive_url and _append_scene_preview_part_from_drive(parts, drive_url, drive, filename):
        return drive_url
    print(f"[layout_revision] Drive upload/load failed for '{filename}', using inline JPEG fallback")
    _append_pil_jpeg_part(parts, pil_image)
    return drive_url


def _append_fgd_asset_media(parts, asset_url, drive, label_prefix=""):
    """
    Append image or video multimodal media for one FGD asset URL.

    :param parts: Mutable list of Gemini Part objects
    :param asset_url: Assigned asset URL from final graphics definition
    :param drive: Google Drive client for image loading
    :param label_prefix: Label prefix for image load logging
    :return: None
    """
    if not asset_url:
        return
    if is_youtube_url(asset_url):
        clip_url, start_seconds, end_seconds = parse_video_url_timestamps(asset_url)
        if not clip_url:
            clip_url = convert_watch_url_to_embed_url(asset_url)
        if clip_url:
            video_part = build_video_part(clip_url, start_seconds, end_seconds)
            if video_part:
                parts.append(video_part)
        return
    pil = load_image_from_url(asset_url, drive, label_prefix or "layout_revision_fgd")
    if pil:
        _append_pil_jpeg_part(parts, pil.convert("RGB"))


def _build_manifest_scene_text(scene, slot_voiceover_resolver):
    """
    Build prompt text for one manifest scene and its slots.

    :param scene: Parsed scene dict from the slideshow manifest
    :param slot_voiceover_resolver: Callable mapping slot asset URL to When VO text
    :return: String of prompt text for the scene and its slots
    """
    scene_id = scene["id"]
    template = _normalize_scene_template(scene["template"])
    narration = (scene.get("narration") or "").strip()
    lines = [
        f"Scene {scene_id} | Template: {template}",
        f"Narration Span: {narration}",
    ]
    for slot in scene.get("slots") or []:
        role = (slot.get("role") or "").strip()
        asset_url = (slot.get("asset") or "").strip()
        slot_vo = ""
        if slot_voiceover_resolver:
            slot_vo = (slot_voiceover_resolver(asset_url) or "").strip()
        if template == "single_visual_hero" and not slot_vo:
            slot_vo = narration
        lines.append(f"Slot Role: {role}")
        lines.append(f"Voiceover: {slot_vo}")
        lines.append(f"Asset URL: {asset_url}")
    return "\n".join(lines)


def _format_fgd_pair_for_prompt(when_vo, asset_url):
    """
    Format one narration beat and assigned visual URL for layout prompts.

    :param when_vo: Raw When VO text from final_graphics_definition
    :param asset_url: Assigned visual asset URL
    :return: Compact two-line prompt block
    """
    vo_text = normalize_when_vo_line(when_vo)
    return f"When Voiceover: {vo_text}\nAssigned Visual: {asset_url}"


def format_compact_fgd_for_layout_prompt(final_graphics_definition):
    """
    Reduce final_graphics_definition to ordered When Voiceover / Assigned Visual pairs.

    :param final_graphics_definition: Full final graphics definition cell text
    :return: Compact prompt text without segment headers or selection metadata
    """
    pairs = parse_when_vo_assigned_pairs(final_graphics_definition or "")
    if not pairs:
        return (final_graphics_definition or "").strip()
    return "\n\n".join(_format_fgd_pair_for_prompt(when_vo, asset_url) for when_vo, asset_url in pairs)


def build_layout_revision_multimodal_parts(manifest_xml, final_graphics_definition, drive, slide_title="", target_scene_id="", slide_id=""):
    """
    Build structured multimodal input with FGD assets, manifest screenshots, and target scene.

    :param manifest_xml: Slideshow manifest XML string
    :param final_graphics_definition: Final graphics definition text
    :param drive: Google Drive service instance
    :param slide_title: Slide title for scene preview title bar
    :param target_scene_id: Scene the reviewer submitted feedback for
    :param slide_id: Slide identifier used for sce
    :return: List of Gemini Part objects
    """
    parts = []
    fgd_pairs = parse_when_vo_assigned_pairs(final_graphics_definition or "")

    parts.append(types.Part(text="Final Graphics Definition for this Slide:\n"))
    for idx, (when_vo, asset_url) in enumerate(fgd_pairs, start=1):
        parts.append(
            types.Part(
                text=_format_fgd_pair_for_prompt(when_vo, asset_url) + "\n"
            )
        )
        _append_fgd_asset_media(parts, asset_url, drive, f"layout_revision_fgd_{idx}")
        parts.append(types.Part(text="---\n"))

    parts.append(types.Part(text=f"{_MULTIMODAL_SECTION_RULE}\n"))
    parts.append(types.Part(text="Current Slideshow Manifest for this Slide:\n"))

    slot_voiceover_resolver = None
    if final_graphics_definition:
        slot_voiceover_resolver = make_slot_narration_resolver_from_fgd(final_graphics_definition)

    scenes = parse_scenes_from_manifest(manifest_xml)
    scene_preview_drive_urls = {}
    for scene in scenes:
        scene_id = str(scene.get("id", "")).strip()
        parts.append(types.Part(text=_build_manifest_scene_text(scene, slot_voiceover_resolver) + "\n"))
        preview = make_agent_scene_preview_image(
            slide_title,
            scene,
            drive,
            template_normalizer=_normalize_scene_template,
        )
        filename = _scene_preview_drive_filename(slide_id, scene_id)
        drive_url = _append_scene_preview_part(parts, preview, drive, filename)
        if drive_url:
            scene_preview_drive_urls[scene_id] = drive_url
        parts.append(types.Part(text="---\n"))

    parts.append(types.Part(text=f"{_MULTIMODAL_SECTION_RULE}\n"))
    target_id = str(target_scene_id or "").strip()
    target_scene = next((sc for sc in scenes if str(sc.get("id", "")).strip() == target_id), None)
    parts.append(types.Part(text="Target Scene For which the Human has left the feedback for:\n"))
    if target_scene:
        target_template = _normalize_scene_template(target_scene.get("template", ""))
        parts.append(
            types.Part(
                text=f"Scene ID: {target_id}\nTemplate: {target_template}\n"
            )
        )
        filename = _scene_preview_drive_filename(slide_id, target_id)
        cached_url = scene_preview_drive_urls.get(target_id)
        if not cached_url or not _append_scene_preview_part_from_drive(parts, cached_url, drive, filename):
            target_preview = make_agent_scene_preview_image(
                slide_title,
                target_scene,
                drive,
                template_normalizer=_normalize_scene_template,
            )
            _append_scene_preview_part(parts, target_preview, drive, filename)
    else:
        parts.append(
            types.Part(
                text=f"Scene ID: {target_id}\nTemplate: (scene not found in manifest)\n"
            )
        )

    parts.append(types.Part(text=f"{_MULTIMODAL_SECTION_RULE}\n"))
    return parts


def validate_revised_manifest(manifest_xml, fgd):
    """
    Validate that the revised manifest meets strict structural and cardinality rules.

    :param manifest_xml: revised manifest XML string
    :param fgd: final graphics definition text
    :return: Tuple of (is_valid, error_message)
    """
    scenes = parse_scenes_from_manifest(manifest_xml)
    if not scenes:
        return False, "Failed to parse scenes from revised manifest or manifest is empty."
        
    pairs = parse_when_vo_assigned_pairs(fgd)
    fgd_urls = [url.strip() for _, url in pairs if url.strip()]
    
    manifest_urls = []
    for sc in scenes:
        for slot in sc["slots"]:
            manifest_urls.append(slot["asset"].strip())
            
    if len(manifest_urls) != len(fgd_urls):
        return False, f"Mismatch in visual count: final_graphics_definition has {len(fgd_urls)} but manifest has {len(manifest_urls)} slots."

    remaining_fgd_urls = list(fgd_urls)
    for manifest_url in manifest_urls:
        match_index = None
        for i, fgd_url in enumerate(remaining_fgd_urls):
            if _asset_urls_match(manifest_url, fgd_url):
                match_index = i
                break
        if match_index is None:
            return False, f"URL mismatch: manifest contains asset not in final_graphics_definition: {manifest_url}"
        remaining_fgd_urls.pop(match_index)

    if remaining_fgd_urls:
        return False, f"URL mismatch: missing in manifest: {remaining_fgd_urls}"
        
    allowed_roles = {
        "single_visual_hero": {"primary_visual"},
        "two_item_split_comparison": {"left_visual", "right_visual"},
        "multi_panel_grid": {"panel_1", "panel_2", "panel_3", "panel_4"},
        "main_plus_supporting_inset": {"main_visual", "inset_visual"}
    }
    
    for idx, sc in enumerate(scenes, start=1):
        template = _normalize_scene_template(sc["template"])
        if template not in allowed_roles:
            return False, f"Scene {idx} has invalid template: '{template}'"
        slots = sc["slots"]
        expected_set = allowed_roles[template]
        for slot in slots:
            role = slot["role"]
            if role not in expected_set:
                return False, f"Scene {idx} ({template}) has invalid slot role: '{role}'"
                
    for idx, sc in enumerate(scenes, start=1):
        if sc["id"] != str(idx):
            return False, f"Non-sequential scene ID: expected '{idx}' but got '{sc['id']}'"
            
    return True, ""


def plan_layout_revision(course_name, target_audience, topic_name, subtopic_name, slide_id, slide_type, slide_title, slide_content, final_graphics_definition, current_slideshow_manifest, target_scene_id, human_layout_feedback, drive, llm="gemini_3_flash_thinking"):
    """
    Call Gemini to revise the slideshow manifest layout plan based on human feedback.

    :param course_name: course name
    :param target_audience: target audience
    :param topic_name: topic name
    :param subtopic_name: subtopic name
    :param slide_id: slide identifier
    :param slide_type: slide type
    :param slide_title: slide title
    :param slide_content: slide content
    :param final_graphics_definition: final graphics definition text
    :param current_slideshow_manifest: current manifest XML string
    :param target_scene_id: ID of the scene being revised
    :param human_layout_feedback: human layout feedback text
    :param drive: Google Drive service instance
    :param llm: LLM model name to use
    :return: Tuple of (operations, raw_plan_response)
    """
    parts = build_layout_revision_multimodal_parts(
        current_slideshow_manifest,
        final_graphics_definition,
        drive,
        slide_title=slide_title,
        target_scene_id=target_scene_id,
        slide_id=slide_id,
    )
    
    compact_fgd = format_compact_fgd_for_layout_prompt(final_graphics_definition)

    prompt_text = LAYOUT_REVISION_PROMPT.format(
        course_name=course_name or "",
        target_audience=target_audience or "",
        topic_name=topic_name or "",
        subtopic_name=subtopic_name or "",
        slide_id=slide_id or "",
        slide_type=slide_type or "",
        slide_title=slide_title or "",
        slide_content=slide_content or "",
        final_graphics_definition=compact_fgd,
        current_slideshow_manifest=(current_slideshow_manifest or "").strip(),
        target_scene_id=target_scene_id or "",
        human_layout_feedback=human_layout_feedback or ""
    )

    print("\n" + "=" * 80)
    print("--- LAYOUT REVISER MULTIMODAL PARTS (TEXT LABELS & MEDIA) ---")
    for i, part in enumerate(parts, 1):
        text = getattr(part, "text", None) if part else None
        if text:
            print(f"[Part {i}] {text.strip()}")
        else:
            print(f"[Part {i}] <inline media (image or video)>")
    print("=" * 80 + "\n")

    print("\n" + "=" * 80)
    print("--- LAYOUT REVISER TEXT PROMPT ---")
    print(prompt_text)
    print("=" * 80 + "\n")

    parts.append(types.Part(text=prompt_text))

    print(f"--- LAYOUT REVISER LLM CALL (model: {llm}) ---")
    response_text = invoke_gemini_multimodal(parts, llm=llm, temperature=0.3)

    print("\n" + "=" * 80)
    print("--- LAYOUT REVISER RESPONSE ---")
    print(response_text or "")
    print("=" * 80 + "\n")
    
    operations = parse_layout_revision_operations(response_text)
    return operations, response_text


def run_layout_revision_for_row(session, row_index, scene_id, feedback, llm=None):
    """
    Plan, apply, and persist layout revision for a single spreadsheet row.

    :param session: UserSession object
    :param row_index: spreadsheet row index
    :param scene_id: ID of the scene being revised
    :param feedback: human feedback string
    :param llm: optional LLM override
    :return: Dict with operations, events, and raw_plan
    """
    llm = llm or LLM_DEFAULT
    
    _, df, _ = load_workbook(session)
    row = df.loc[row_index]

    _, course_info_df = get_sheet_data_and_df(session.sheet, "Course info")
    course_name = safe_str(course_info_df.iloc[0].get("Course Name", "")).strip()
    target_audience = safe_str(course_info_df.iloc[0].get("Target Audience & Industry", "")).strip()

    slide_title = safe_str(row.get("Slide Chunk Title", "")).strip()
    slide_content = safe_str(row.get("Slide Chunk", "")).strip()
    topic_name = safe_str(row.get("Topic", "")).strip()
    subtopic_name = safe_str(row.get("Subtopic", "")).strip()
    slide_type = safe_str(row.get("Slide Type", "")).strip()
    if slide_type == "nan":
        slide_type = ""
        
    final_graphics_definition = safe_str(row.get("final_graphics_definition", ""))
    current_manifest = safe_str(row.get("slideshow_manifest", ""))
    slide_id = f"SLIDE_{row_index + 1}"

    operations, raw_plan = plan_layout_revision(
        course_name=course_name,
        target_audience=target_audience,
        topic_name=topic_name,
        subtopic_name=subtopic_name,
        slide_id=slide_id,
        slide_type=slide_type,
        slide_title=slide_title,
        slide_content=slide_content,
        final_graphics_definition=final_graphics_definition,
        current_slideshow_manifest=current_manifest,
        target_scene_id=scene_id,
        human_layout_feedback=feedback,
        drive=session.drive,
        llm=llm
    )

    if not operations:
        apply_layout_revision_to_sheet(
            session,
            row_index=row_index,
            scene_id=scene_id,
            feedback=feedback,
            raw_plan=raw_plan,
            updated_slideshow_manifest=current_manifest,
            events=["No layout operations planned."]
        )
        return {"operations": [], "events": ["No layout operations planned."], "raw_plan": raw_plan, "manifest_applied": False}

    revised_manifest = apply_layout_revision_operations(current_manifest, operations, final_graphics_definition)
    
    is_valid, err = validate_revised_manifest(revised_manifest, final_graphics_definition)
    events = []
    manifest_applied = False
    if is_valid:
        events.append("Layout operations successfully applied and validated.")
        manifest_applied = True
    else:
        print(f"[layout_revision] Revised manifest failed validation: {err}. Falling back to original.")
        events.append(f"ERROR: Revised manifest failed validation: {err}. Falling back to original.")
        revised_manifest = current_manifest

    apply_layout_revision_to_sheet(
        session,
        row_index=row_index,
        scene_id=scene_id,
        feedback=feedback,
        raw_plan=raw_plan,
        updated_slideshow_manifest=revised_manifest,
        events=events
    )
    
    return {"operations": operations, "events": events, "raw_plan": raw_plan, "manifest_applied": manifest_applied}
