import re
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from xml.etree import ElementTree as ET

import streamlit as st
from dotenv import load_dotenv
from google.genai import types
from langsmith import traceable

from agents.graphics_definition_v2.aggregation_agent.aggregation_agent import (
    convert_watch_url_to_embed_url,
    get_drive_instance,
    invoke_gemini_multimodal,
    load_image_from_url,
    parse_video_url_timestamps,
)
from agents.graphics_definition_v2.slideshow_manifest.slideshow_manifest import (
    make_slot_narration_resolver_from_fgd,
)
from services.helper_functions import build_video_part
from services.sheets_service import (
    clear_worksheet,
    format_worksheet,
    get_sheet_data_and_df,
    save_to_sheet,
)
from services.smart_progress_bar import SmartProgressBar

load_dotenv()

image_edit_planning_prompt = """You are a senior graphics designer specializing in HVAC e-learning content. Your task is to review the selected visual assets for a single slideshow scene and create a structured edit plan that specifies whether each asset needs any instructional image edits before it is used in the final slideshow video.

In the final slideshow video, the selected visuals will be placed on the slide canvas according to the scene layout while the voiceover narration of the slide content plays in the background. Your edit decisions should therefore support what the learner needs to notice at that exact moment in the narration.

This edit plan will be used by a downstream image editing pipeline to apply only the required edits to the selected visuals. You do not edit images directly. Your responsibility is to decide what edits, if any, are needed for each visual asset so that the final scene is clearer, more instructionally useful, and better aligned with the narration and layout intent.

You will be given course context, slide context, one scene from the slideshow manifest, and the selected visual assets assigned to that scene. Review the scene carefully and decide whether each visual asset should remain unchanged or receive one or more specific edit instructions from the allowed edit types.

These are the inputs:

<course_information>
Course name: {course_name}
Target audience: {target_audience}
Topic name: {topic_name}
Subtopic name: {subtopic_name}
</course_information>

<slide_information>
Slide Type: {slide_type}
Slide Title: {slide_title}
Slide Content: {slide_content}
</slide_information>

<scene_information>
Scene ID: {scene_id}
Layout Template: {layout_template}
Narration Span: {narration_span}
</scene_information>

<scene_slots>
{scene_slots}
</scene_slots>

Instructions and Guidelines:

1. Core Responsibility
   - Your task is to create an edit plan for one slideshow scene only.
   - Review each visual asset assigned to the scene and decide whether it needs any instructional image edits before it is used in the final slideshow video.
   - You do not edit images directly. You only decide what edits are needed and describe them clearly for a downstream image editing pipeline.
   - Make edit decisions based on the narration span, the layout template, the slot role, and what is actually visible in each asset.
   - Use only the allowed edit types listed in this prompt.
   - Do not request edits that are decorative, unnecessary, or unrelated to learner understanding.
   - Prefer keeping the original asset unchanged when it already supports the narration clearly.

2. Layout Template Reference
   The layout template has already been assigned for this scene by an upstream layout agent. Use the reference below to understand the intended role of each slot and how editing decisions should respect the assigned layout.

   a) single_visual_hero
      - Description: One dominant visual supports the full scene.
      - Slot role: primary_visual
      - Editing implication: Edits should help the single visual clearly communicate the main narration idea without cluttering the full-screen composition.

   b) two_item_split_comparison
      - Description: Two visuals are shown side-by-side for direct comparison or paired explanation.
      - Slot roles: left_visual, right_visual
      - Editing implication: Edits should preserve the comparison. If labels, arrows, or highlights are needed, they should be balanced and consistent across both visuals when appropriate.

   c) multi_panel_grid
      - Description: Three or four visuals are shown as equal or near-equal panels.
      - Slot roles: panel_1, panel_2, panel_3, optional panel_4
      - Editing implication: Edits should remain minimal because each visual will appear smaller on screen. Avoid dense labels or multiple overlays inside a single panel.

   d) main_plus_supporting_inset
      - Description: One main visual is supported by one smaller inset/detail visual.
      - Slot roles: main_visual, inset_visual
      - Editing implication: The main visual should carry the main narration idea. The inset visual should clarify a specific detail. Avoid edits that make the inset feel more important than the main visual.

3. Scene and Narration Understanding
   - First, understand the instructional meaning of the scene’s narration span.
   - Use the full slide content to resolve context, continuation phrases, title-only narration, or implied meaning.
   - Do not interpret the narration span in isolation if the full slide content provides important context.
   - Identify what the learner needs to notice visually while this narration span is playing.
   - Use the assigned layout template and slot roles only as context for how the visuals will appear on screen.
   - Do not change, question, or reassign the layout template. The layout has already been selected by an upstream layout agent.
   - Do not introduce new instructional content that is not supported by the narration span or slide context.

4. Slot-by-Slot Visual Review
   - Review every slot listed in <scene_slots> independently.
   - For each slot, identify the slot role, asset URL, and whether the asset appears to be an image or a video.
   - Use the provided multimodal visual preview to inspect what is actually visible in the asset.
   - As you inspect each asset, mentally enumerate the visual anchors you would use to locate any potential edit target: position in the frame (top, bottom, center, left, right, upper-third, lower-right corner, etc.), nearby visible objects, colors, shapes, sizes, orientations, and any distinguishing details or conditions. These anchors are what you will reuse when writing target_description for the planned edits.
   - Base edit decisions only on visible content in the asset and the narration/layout intent.
   - Do not assume an object, component, label, or condition is present unless it is clearly visible.
   - If an asset is already clear, specific, and instructionally useful for its slot role, mark it as NO_EDIT.
   - If an asset is visually relevant but would be clearer with some edits, then only suggest the edits from the allowed edit types.
   - If an asset is a video URL, do not plan image edits for it. Mark it as SKIPPED_VIDEO.

5. Allowed Edit Types and When to Use Them
   - For each slot and its visual, choose only from the fixed edit types listed below.
   - Do not invent new edit types, aliases, or custom operation names.
   - A slot may have NO_EDIT, SKIPPED_VIDEO, or one or more instructional edit types if edits are clearly needed.

   Allowed edit types:

   a) NO_EDIT
      - Use when the asset is already clear, relevant, and instructionally useful without modification.
      - Use this by default when an edit would not meaningfully improve learner understanding.

   b) SKIPPED_VIDEO
      - Use when the asset is a video URL.
      - Do not request image edits for video assets in this planning step.

   c) ADD_TEXT_LABEL
     - Use when one or more specific visible objects, parts, conditions, or regions need to be named for learner clarity.
     - You may plan multiple text labels for the same asset when multiple visible targets need to be identified.
     - Each label must have its own target_description and label_text.
      - Label text must be short, direct, and instructional.
     - Do not add labels for obvious objects or decorative purposes.
     - Do not add too many labels if they would clutter the visual or make the final scene harder to understand.

   d) CROP_IMAGE
      - Use when removing unnecessary surrounding area would make the relevant object or region easier to see.
      - Do not crop out important context needed to understand the scene.
      - The crop instruction must clearly describe what should remain visible.

   e) ADD_ARROW
      - Use when the learner needs help locating a specific visible object, component, or region.
      - The arrow target must be clearly visible in the asset.
      - Do not use arrows for vague or assumed targets.

   f) ADD_HIGHLIGHT_CIRCLE_OR_BOX
      - Use when a specific region, object, defect, condition, or detail needs emphasis.
      - The highlight target must be clearly visible.
      - Prefer this for drawing attention without adding too much text.

   g) ADD_ICON
      - Use only when a simple symbol would clarify the meaning of the visual.
      - Do not add icons that are decorative or that introduce meaning not present in the narration.

   h) ADD_EMPHASIS
      - Use when the relevant area needs stronger visual focus, such as zoom emphasis, dimming unrelated background, or subtle visual emphasis.
      - Use only when the asset is relevant but the important detail may be missed without emphasis.
      - Do not use emphasis effects that distort the technical meaning of the asset.

6. Purposeful Editing and Clarity Rules
   - Plan edits when they can make the asset clearer, more instructionally useful, or easier to understand during the narration span.
   - Prefer edits that help the learner quickly notice the specific object, part, condition, action, or relationship being described.
   - Use NO_EDIT when the asset already communicates the narration clearly without modification.
   - Use multiple edits when they are genuinely needed, but make sure each edit has a clear instructional purpose.
   - Avoid decorative edits that only make the asset look more polished but do not improve learner understanding.
   - Avoid edits that create unnecessary clutter or make the important visual information harder to see.
   - When multiple assets appear in the same scene, make sure the planned edits support the overall scene layout and do not make one slot visually confusing or unintentionally dominant.
   - If multiple edits are needed for one asset, order them by instructional importance using priority_order.
   - Do not request edits that change the factual meaning of the visual or make the asset imply something different from what is shown.

7. Edit Targeting and Description Rules
   - For every planned edit, clearly describe the exact visible target that should be edited.
   - Treat target_description as standalone instructions for a separate downstream image editor that has the original image and your description, and nothing else. It does not see this plan, your reasoning, the narration span, or the slide content. The target must be locatable from target_description alone.
   - The target_description must identify what part of the asset the edit should apply to, such as a component, object, region, condition, flow path, or visual relationship.
   - Write target_description as multiple specific sentences. A single short phrase like "the central component", "the relevant flame", or "the gauge" is not acceptable.
   - Anchor the target with several of the following cues, using only what is clearly visible:
       - Position in the image (top, bottom, center, left, right, upper-third, lower-right corner, etc.).
       - Nearby visible landmarks (next to, above, between, attached to, behind, in front of specific visible objects).
       - Color, shape, size, orientation, or material of the target.
       - Its relationship to other visible parts (the pipe leading from X to Y, the gauge mounted on the side of Z, the third unit from the left, etc.).
       - The state or condition shown (open, closed, corroded, glowing, leaking, on, off, etc.) when it is relevant to the edit purpose.
   - If multiple similar objects are visible in the asset (multiple flames, multiple gauges, multiple pipes, multiple panels, etc.), disambiguate the exact intended one using position and adjacent landmarks, not just the object type.
   - If the target is a region rather than a single object, describe its boundaries using visible landmarks (from the top of X down to Y, the full area inside the dashed enclosure on the left, the band of pipes across the lower half of the image, etc.).
   - Do not use vague target descriptions like "highlight the important part" or "label the relevant area."
   - For ADD_TEXT_LABEL, provide the exact label_text that should appear on the asset, and also state where the label should be placed relative to the target (just above, to the right of, with a leader line from, etc.) so the label does not cover important detail.
   - For ADD_ARROW, describe both what the arrow should point to and, when useful, where the arrow should come from or be placed.
   - For ADD_HIGHLIGHT_CIRCLE_OR_BOX, specify whether a circle or box would be more appropriate when the choice is clear, and describe the approximate extent so the highlight encloses the right region without covering important neighboring detail or missing the target.
   - For CROP_IMAGE, describe what should remain visible after cropping and what unnecessary area can be removed, defining the kept region using visible landmarks.
   - For ADD_ICON, specify the icon meaning, such as warning, safety, airflow, water flow, correct, or incorrect, etc., and where it should be placed relative to the related visible object.
   - For ADD_EMPHASIS, describe the intended emphasis effect, such as zooming into a component, dimming unrelated background, or visually focusing attention on a specific region, and identify the exact region the emphasis applies to.

   Examples of acceptable target_description detail (for tone and specificity reference only, do not copy):
       - Good: "The pressure gauge mounted on the upper-right side of the boiler unit. It is a round dial with a red needle, sitting just above the horizontal pipe that exits the top of the unit. It is the only gauge visible in the upper half of the image."
       - Bad: "The pressure gauge."
       - Good: "The row of blue and orange flames at the center of the image, occupying roughly the middle third horizontally. The flames sit on top of a metal burner bar with circular ports. There are no other flames or fire visible elsewhere in the image."
       - Bad: "The flames in the middle."

8. Output Consistency Requirements
   - Provide one <slot_edit> block for every slot listed in <scene_slots>.
   - Preserve the exact slot_role value from the input.
   - Preserve the exact asset_url value from the input.
   - Each <slot_edit> block must contain the following fields:

     a) Slot Role:
        - The exact role of the slot from the input, such as primary_visual, left_visual, right_visual, panel_1, main_visual, or inset_visual.
        - Do not rename, paraphrase, or invent slot roles.

     b) Asset URL:
        - The exact asset URL from the input.
        - Do not shorten, rewrite, or replace the URL.

     c) Asset Type:
        - Use one of these exact values: image | video | unknown.
        - Use image for still image assets.
        - Use video for YouTube URLs or other video assets.
        - Use unknown only when the asset type cannot be confidently determined.

     d) <edit_required>
        - Use one of these exact values: YES | NO.
        - Use YES when one or more image edits should be applied to the asset.
        - Use NO when the asset should remain unchanged, when the asset is a video, or when no safe/useful edit can be planned.

     e) <edits>
        - This section must contain one or more <edit> blocks.
        - If edit_required is NO, provide exactly one <edit> block.
        - If the asset is a video, that single edit block must use SKIPPED_VIDEO.
        - If the asset is an image and no edit is needed, that single edit block must use NO_EDIT.
        - If edit_required is YES, provide one or more <edit> blocks using only the allowed instructional edit types.

   - Each <edit> block must contain the following fields:

     a) <priority_order>
        - The execution order for the edit.
        - Use numeric values as text: 1, 2, 3, etc.
        - For NO_EDIT or SKIPPED_VIDEO, use 0.

     b) <edit_type>
        - Use only one of the allowed edit types:
          NO_EDIT
          SKIPPED_VIDEO
          ADD_TEXT_LABEL
          CROP_IMAGE
          ADD_ARROW
          ADD_HIGHLIGHT_CIRCLE_OR_BOX
          ADD_ICON
          ADD_EMPHASIS

     c) <target_description>
        - Describe the exact visible target or region the edit should apply to.
        - Write multiple specific sentences. Include position in the image, nearby visible landmarks, and distinguishing details (color, shape, size, orientation, condition) so the downstream image editor can locate the target without any other context.
        - If multiple similar objects are visible, disambiguate the exact intended one using position and adjacent landmarks, not just the object type.
        - Do not write a single short phrase or a generic label such as "the central component" or "the relevant flame".
        - For NO_EDIT, write: None.
        - For SKIPPED_VIDEO, write: Video asset not edited.
        - For CROP_IMAGE, describe what should remain visible, what unnecessary area can be removed, and the approximate boundaries of the kept region using visible landmarks.
        - For ADD_TEXT_LABEL, ADD_ARROW, ADD_HIGHLIGHT_CIRCLE_OR_BOX, ADD_ICON, or ADD_EMPHASIS, describe the object, part, condition, or region to target, and where the overlay should be placed relative to it so it does not cover important detail.

     d) <label_text>
        - Required only for ADD_TEXT_LABEL.
        - For all other edit types, write: None.
        - Keep label text short, clear, and instructional.
        - If multiple labels are needed, create separate <edit> blocks for each label.

     e) <reason_for_edit>
        - Briefly explain why this edit is needed for learner understanding.
        - The reason should connect the edit to the narration span, slot role, or layout intent.
        - For NO_EDIT, explain why the asset is already clear enough.
        - For SKIPPED_VIDEO, explain that video assets are not edited in this image editing step.
     
     f) <must_preserve>
        - Describe any important visual context that must not be cropped out, covered, or obscured during editing.
        - For NO_EDIT or SKIPPED_VIDEO, write: None.
        - For crop, label, arrow, highlight, icon, or emphasis edits, mention the key object/context that should remain visible.

   - Do not leave required fields blank.
   - Do not add fields outside the required output schema.

Output:

Always provide your output strictly in the following format:

<output>

<evaluation_breakdown>

Use this section as a structured reasoning and scratchpad space to evaluate the assigned scene assets and decide whether each asset needs instructional image edits before producing the final scene edit plan. Use the following sections:

1. Assigned Asset Review
- List each assigned asset URL in the scene and briefly describe what is visibly shown in that asset. Focus only on what is actually visible, do not assume unseen details.

2. Scene Understanding
- Briefly explain what the scene narration is communicating.
- Use the full slide content to resolve any context, title-only narration, continuation phrases, or implied meaning.
- Explain what the learner needs to notice visually while this scene with its assigned visual is shown on the slide canvas.

3. Layout and Slot Context
- Explain the assigned layout template and how each slot role functions in this scene.
- Use the layout and slot roles to explain how the visuals will appear and how that affects edit planning.

4. Edit Need Analysis
- Analyze whether each slot needs editing.
- For each slot, explain whether the asset is already clear enough or whether edits such as labels, cropping, arrows, highlights, icons, or emphasis would improve learner understanding.
- Explain the specific visual target for any edit you are considering. Identify it by position in the image, nearby visible landmarks, and distinguishing details (color, shape, size, orientation, condition).
- If multiple similar objects are visible (multiple flames, gauges, pipes, panels, etc.), call out how you are disambiguating the exact intended target.
- Record these anchors here so you can reuse them when writing target_description in the final scene edit plan.

5. Final Edit Plan Rationale
- Summarize the final edit planning decision for each slot.
- Explain why each slot is marked for editing, no edit, or skipped because it is a video/non-editable asset.

6. Additional Analysis
- Note any additional observations, thoughts, tradeoffs, edge cases, or analysis that can help you arrive at a clear and useful edit plan.
- It is acceptable for this section to be detailed or verbose if needed for correctness.

</evaluation_breakdown>

(Based on your above evaluation, provide the final scene edit plan below.)

<scene_edit_plan>

<slot_edit>
Slot Role: The exact slot role from the input.
Asset URL: The exact asset URL from the input.
Asset Type: image | video | unknown

<edit_required>
YES | NO
</edit_required>

<edits>

<edit>

<priority_order>
Execution order for this edit. Use 1, 2, 3, etc. For NO_EDIT or SKIPPED_VIDEO, use 0.
</priority_order>

<edit_type>
ADD_TEXT_LABEL | CROP_IMAGE | ADD_ARROW | ADD_HIGHLIGHT_CIRCLE_OR_BOX | ADD_ICON | ADD_EMPHASIS | NO_EDIT | SKIPPED_VIDEO
</edit_type>

<target_description>
Describe the exact visible object, component, condition, or region to edit, using multiple specific sentences. Include position in the image, nearby visible landmarks, distinguishing details (color, shape, size, orientation, condition), and — for overlay edits — where the overlay should be placed relative to the target. Make it specific enough that a separate image editor can locate the target from this field alone, without any other context. Do not write a single short phrase. For NO_EDIT, write None. For SKIPPED_VIDEO, write Video asset not edited.
</target_description>

<label_text>
Exact label text for ADD_TEXT_LABEL. For all other edit types, write None.
</label_text>

<reason_for_edit>
Briefly explain why this edit is needed for learner understanding, or why no edit is needed.
</reason_for_edit>

<must_preserve>
Important visual context that must not be cropped, covered, or obscured. For NO_EDIT or SKIPPED_VIDEO, write None.
</must_preserve>

</edit>

<!-- Repeat <edit> if multiple edits are needed for this same asset -->

</edits>

</slot_edit>

<!-- Repeat <slot_edit> for every slot in the scene -->

</scene_edit_plan>

</output>

(Ensure that you strictly follow this exact output format. Do not add any extra text or comments outside the <output>, <evaluation_breakdown>, and <scene_edit_plan> sections.)
"""


def _is_youtube_url(url):
    """
    Check whether a URL points to YouTube.

    :param url: Candidate asset URL.
    :return: True when URL is YouTube/youtu.be, else False.
    """
    u = (url or "").lower()
    return "youtube.com" in u or "youtu.be" in u


def _normalize_attribute_quotes(text):
    """
    Normalize doubled XML attribute quotes from sheet-exported strings.

    :param text: Raw manifest text.
    :return: Text with attribute forms like id=""1"" normalized to id="1".
    """
    if not text:
        return text
    return re.sub(r'""([^"<>]*)""', r'"\1"', text)


def _escape_bare_ampersands(text):
    """
    Escape ampersands that are not valid XML entities.

    :param text: Raw or normalized XML text.
    :return: Entity-safe XML text for parser consumption.
    """
    if not text:
        return text
    return re.sub(r"&(?!(amp|lt|gt|quot|apos|#\d+|#x[0-9A-Fa-f]+);)", "&amp;", text)


def parse_scenes_from_slideshow_manifest(xml_text):
    """
    Parse slideshow_manifest content into scene dictionaries.

    :param xml_text: Manifest text from slideshow_manifest column (inner XML or wrapped root).
    :return: List of scene dicts with keys: id, template, narration, slots(list of {role, asset}).
    """
    if not xml_text or not str(xml_text).strip():
        return []
    text = str(xml_text).strip()
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]
    text = _normalize_attribute_quotes(text)
    if "<slideshow_manifest" not in text:
        text = f"<slideshow_manifest>\n{text}\n</slideshow_manifest>"
    safe_text = _escape_bare_ampersands(text)

    try:
        root = ET.fromstring(safe_text)
    except ET.ParseError:
        return []

    scenes = []
    for idx, scene_el in enumerate(root.findall("scene"), start=1):
        scene_id = (scene_el.get("id") or str(idx)).strip()
        template = (scene_el.get("template") or "").strip()
        narr_el = scene_el.find("narration_span")
        narration = (narr_el.text or "").strip() if narr_el is not None else ""
        slots = []
        for slot_el in scene_el.findall("slot"):
            role = (slot_el.get("role") or "").strip()
            asset = (slot_el.get("asset") or "").strip()
            if role and asset:
                slots.append({"role": role, "asset": asset})
        if template and slots:
            scenes.append({
                "id": scene_id,
                "template": template,
                "narration": narration,
                "slots": slots,
            })
    return scenes


def _scene_slots_text(slots, slot_voiceovers=None):
    """
    Build prompt-ready <scene_slots> inner content from parsed slot dicts.

    :param slots: List of slot dicts with role/asset.
    :param slot_voiceovers: Optional list parallel to slots with When VO text per slot (from FGD).
    :return: XML-like text block containing one <slot> block per slot.
    """
    chunks = []
    for si, slot in enumerate(slots or []):
        role = str(slot.get("role", "")).strip()
        asset = str(slot.get("asset", "")).strip()
        vo_line = ""
        if slot_voiceovers is not None and si < len(slot_voiceovers):
            vo_line = (slot_voiceovers[si] or "").strip()
        vo_block = ""
        if vo_line:
            vo_block = f"Voiceover for this visual: {vo_line}\n"
        chunks.append(
            "<slot>\n"
            f"Slot Role: {role}\n"
            f"{vo_block}"
            f"Asset URL: {asset}\n"
            "</slot>"
        )
    return "\n\n".join(chunks).strip()


def _append_scene_slot_multimodal_parts(parts, scene_id, slot_index, slot_role, asset_url, drive, when_vo_for_slot=None):
    """
    Append text label and multimodal asset part (image/video) for one slot.

    :param parts: Mutable list of Gemini Part objects.
    :param scene_id: Scene identifier for labeling.
    :param slot_index: 1-based slot index in scene.
    :param slot_role: Slot role name.
    :param asset_url: Asset URL assigned to slot.
    :param drive: Drive client for image loading.
    :param when_vo_for_slot: Voiceover line matched to this asset from final_graphics_definition, if any.
    :return: None
    """
    label = f"Scene ID: {scene_id}\n" f"Slot Role: {slot_role}\n"
    if when_vo_for_slot:
        label += f"Voiceover for this visual: {when_vo_for_slot}\n"
    label += f"Asset URL: {asset_url}\n"
    parts.append(types.Part(text=label))

    if _is_youtube_url(asset_url):
        clip_url, start_seconds, end_seconds = parse_video_url_timestamps(asset_url)
        if not clip_url:
            clip_url = convert_watch_url_to_embed_url(asset_url)
            start_seconds, end_seconds = None, None
        if clip_url:
            video_part = build_video_part(clip_url, start_seconds, end_seconds)
            if video_part:
                parts.append(video_part)
        return

    pil = load_image_from_url(asset_url, drive, f"scene_edit_plan_s{scene_id}_slot_{slot_index}")
    if pil:
        buffered = BytesIO()
        pil.convert("RGB").save(buffered, format="JPEG")
        parts.append(
            types.Part(
                inline_data=types.Blob(mime_type="image/jpeg", data=buffered.getvalue())
            )
        )


def parse_scene_edit_planning_response(response_text):
    """
    Extract inner <scene_edit_plan> and <evaluation_breakdown> from model response when present.

    :param response_text: Raw LLM response text.
    :return: Tuple(scene_edit_plan_inner_text_or_fallback, evaluation_breakdown_text).
    """
    if not response_text:
        return "", ""
    scene_plan_match = re.search(
        r"<scene_edit_plan>(.*?)</scene_edit_plan>",
        response_text,
        re.DOTALL | re.IGNORECASE,
    )
    eval_match = re.search(
        r"<evaluation_breakdown>(.*?)</evaluation_breakdown>",
        response_text,
        re.DOTALL | re.IGNORECASE,
    )
    scene_plan_text = scene_plan_match.group(1).strip() if scene_plan_match else response_text.strip()
    evaluation = eval_match.group(1).strip() if eval_match else ""
    return scene_plan_text, evaluation


@traceable(
    metadata={
        "agent_name": "graphics_definition_v2",
        "step_name": "Scene Edit Planning",
        "function_name": "generate_scene_edit_plan_for_scene",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous"),
    }
)
def generate_scene_edit_plan_for_scene(course_name, target_audience, topic_name, subtopic_name, slide_type, slide_title, slide_content, scene, drive, llm="gemini_3_flash_thinking", slot_narration_resolve=None):
    """
    Run image edit planning for one parsed scene.

    :param course_name: Course name.
    :param target_audience: Target audience text.
    :param topic_name: Topic name.
    :param subtopic_name: Subtopic name.
    :param slide_type: Slide Type value.
    :param slide_title: Slide title text.
    :param slide_content: Full slide content text.
    :param scene: Parsed scene dict (id, template, narration, slots).
    :param drive: Drive client for image loading.
    :param llm: LLM identifier for multimodal planning call.
    :param slot_narration_resolve: Optional callable(asset_url) returning When VO snippet from final_graphics_definition.
    :return: Tuple(scene_id, scene_edit_plan_inner_text, evaluation_breakdown_text).
    """
    scene_id = str(scene.get("id", "")).strip() or "1"
    layout_template = str(scene.get("template", "")).strip()
    narration_span = str(scene.get("narration", "")).strip()
    slots = scene.get("slots", []) or []
    slot_voiceovers = []
    for slot in slots:
        asset = str(slot.get("asset", "")).strip()
        vo = ""
        if slot_narration_resolve and asset:
            vo = slot_narration_resolve(asset) or ""
        slot_voiceovers.append(vo)
    scene_slots = _scene_slots_text(slots, slot_voiceovers)

    prompt_text = image_edit_planning_prompt.format(
        course_name=course_name or "",
        target_audience=target_audience or "",
        topic_name=topic_name or "",
        subtopic_name=subtopic_name or "",
        slide_type=slide_type or "",
        slide_title=slide_title or "",
        slide_content=slide_content or "",
        scene_id=scene_id,
        layout_template=layout_template,
        narration_span=narration_span,
        scene_slots=scene_slots or "",
    )

    parts = []
    for idx, slot in enumerate(slots, start=1):
        asset_url = str(slot.get("asset", "")).strip()
        slot_vo = slot_voiceovers[idx - 1] if idx <= len(slot_voiceovers) else ""
        _append_scene_slot_multimodal_parts(
            parts=parts,
            scene_id=scene_id,
            slot_index=idx,
            slot_role=str(slot.get("role", "")).strip(),
            asset_url=asset_url,
            drive=drive,
            when_vo_for_slot=slot_vo or None,
        )
    parts.append(types.Part(text=prompt_text))

    print(f"\nScene Edit Planning - Scene ID: {scene_id}, Template: {layout_template}")
    response_text = invoke_gemini_multimodal(parts, llm=llm, temperature=0.3)
    print("📤 Scene Edit Planning Full Response from LLM:\n")
    print(response_text or "")
    print("\n" + "=" * 100 + "\n")

    scene_plan_text, evaluation = parse_scene_edit_planning_response(response_text)
    return scene_id, scene_plan_text, evaluation


@traceable(
    metadata={
        "agent_name": "graphics_definition_v2",
        "step_name": "Scene Edit Planning",
        "function_name": "process_scene_edit_plan_row",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous"),
    }
)
def process_scene_edit_plan_row(index, row, course_name, target_audience, drive, llm="gemini_3_flash_thinking"):
    """
    Parse slideshow_manifest for one row and run planning once per scene.

    :param index: DataFrame row index.
    :param row: DataFrame row object.
    :param course_name: Course name from Course info.
    :param target_audience: Target audience from Course info.
    :param drive: Drive client for loading scene assets.
    :param llm: LLM identifier.
    :return: Tuple(index, scene_edit_plan_text).
    """
    
    try:
        manifest_text = str(row.get("slideshow_manifest", "")).strip()
        if not manifest_text or manifest_text == "nan" or manifest_text.startswith("ERROR:"):
            return index, ""

        scenes = parse_scenes_from_slideshow_manifest(manifest_text)
        if not scenes:
            return index, "ERROR: No scenes parsed from slideshow_manifest"

        topic_name = str(row.get("Topic", "")).strip()
        subtopic_name = str(row.get("Subtopic", "")).strip()
        slide_title = str(row.get("Slide Chunk Title", "")).strip()
        slide_content = str(row.get("Slide Chunk", "")).strip()
        slide_type = str(row.get("Slide Type", "")).strip()
        if slide_type == "nan":
            slide_type = ""

        fgd_text = str(row.get("final_graphics_definition", "")).strip()
        slot_narration_resolve = None
        if fgd_text and fgd_text != "nan":
            slot_narration_resolve = make_slot_narration_resolver_from_fgd(fgd_text)

        blocks = []
        for fallback_idx, scene in enumerate(scenes, start=1):
            try:
                scene_id, scene_output, _ = generate_scene_edit_plan_for_scene(
                    course_name=course_name,
                    target_audience=target_audience,
                    topic_name=topic_name,
                    subtopic_name=subtopic_name,
                    slide_type=slide_type,
                    slide_title=slide_title,
                    slide_content=slide_content,
                    scene=scene,
                    drive=drive,
                    llm=llm,
                    slot_narration_resolve=slot_narration_resolve,
                )
                header = f"---Scene ID: {scene_id or fallback_idx}---"
                blocks.append(f"{header}\n{scene_output}".strip())
            except Exception as scene_err:
                scene_id = str(scene.get("id", "")).strip() or str(fallback_idx)
                header = f"---Scene ID: {scene_id}---"
                err_msg = f"ERROR: {str(scene_err)}"
                blocks.append(f"{header}\n{err_msg}")
                traceback.print_exc()

        return index, "\n\n".join(blocks).strip()
    except Exception as e:
        print(f"Error scene edit planning row {index}: {e}")
        traceback.print_exc()
        return index, f"ERROR: {str(e)}"


@traceable(
    metadata={
        "agent_name": "graphics_definition_v2",
        "step_name": "Scene Edit Planning",
        "function_name": "run_scene_edit_planning_for_all_rows",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous"),
    }
)
def run_scene_edit_planning_for_all_rows(sheet, llm="gemini_3_flash_thinking", max_workers=50):
    """
    Generate scene_edit_plan from slideshow_manifest for Slide Chunks.

    :param sheet: gspread sheet object.
    :param llm: LLM identifier for scene edit planning.
    :param max_workers: Row-level parallel workers.
    :return: None
    """
    worksheet_name = "Slide Chunks"

    _, course_info_df = get_sheet_data_and_df(sheet, "Course info")
    course_name = str(course_info_df.loc[0, "Course Name"]).strip() if not course_info_df.empty else ""
    target_audience = str(course_info_df.loc[0, "Target Audience"]).strip() if "Target Audience" in course_info_df.columns and not course_info_df.empty else ""

    ws, df = get_sheet_data_and_df(sheet, worksheet_name)
    if "scene_edit_plan" not in df.columns:
        df["scene_edit_plan"] = ""

    drive = get_drive_instance()
    if not drive:
        print("Scene edit planning: Drive not available; some image loads may fail.")

    rows_to_process = []
    for index, row in df.iterrows():
        manifest = str(row.get("slideshow_manifest", "")).strip()
        existing = str(row.get("scene_edit_plan", "")).strip()
        if not manifest or manifest == "nan" or manifest.startswith("ERROR:"):
            continue
        if existing and existing != "nan" and not existing.startswith("ERROR:"):
            continue
        rows_to_process.append((index, row))

    if not rows_to_process:
        print("Scene edit planning: no rows to process.")
        return

    print(f"Scene edit planning: processing {len(rows_to_process)} row(s), max_workers={max_workers}")
    futures_map = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for index, row in rows_to_process:
            fut = executor.submit(
                process_scene_edit_plan_row,
                index,
                row,
                course_name,
                target_audience,
                drive,
                llm,
            )
            futures_map[fut] = index

        progress = SmartProgressBar(
            total_tasks=len(futures_map),
            description="Scene edit planning",
            save_interval=5,
        )

        for future in as_completed(futures_map):
            index = futures_map[future]
            try:
                row_index, scene_plan = future.result()
                df.at[row_index, "scene_edit_plan"] = scene_plan
                progress.update()
                if progress.should_save():
                    save_to_sheet(ws, df)
            except Exception as e:
                print(f"Scene edit planning future error row {index}: {e}")
                df.at[index, "scene_edit_plan"] = f"ERROR: {str(e)}"
                progress.update()

    save_to_sheet(ws, df)
    format_worksheet(ws)
    print("Scene edit planning: complete.")


def delete_scene_edit_plan_columns(sheet):
    """
    Remove scene_edit_plan column from Slide Chunks worksheet.

    :param sheet: gspread sheet object.
    :return: None
    """
    worksheet_name = "Slide Chunks"
    ws, df = get_sheet_data_and_df(sheet, worksheet_name)
    if "scene_edit_plan" not in df.columns:
        print(f"No scene edit planning column to delete on '{worksheet_name}'.")
        return
    df = df.drop(columns=["scene_edit_plan"])
    clear_worksheet(ws)
    save_to_sheet(ws, df)
    print(f"🗑️ Deleted column 'scene_edit_plan' from '{worksheet_name}' worksheet")