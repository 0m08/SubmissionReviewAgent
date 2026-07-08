import os
import re
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from io import BytesIO

import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types
from langsmith import traceable
from openai import OpenAI
from PIL import Image

from agents.graphics_asset_creation.automated.llm_call_tracker import tracker
from agents.graphics_asset_creation.gac_utils import upload_image_to_drive, styling_guide
from agents.graphics_asset_creation.image_editing.image_editing_openai import (
    download_image_from_url,
    image_from_base64,
    prepare_image_for_upload,
)
from agents.graphics_asset_creation.reviewers.voiceover_reviewer import (
    call_llm_with_retry,
    prepare_image_for_gemini,
)
from agents.graphics_definition_v2.aggregation_agent.aggregation_agent import (
    get_drive_instance,
    load_image_from_url,
)
from agents.graphics_definition_v2.image_editing_for_layout.image_edit_planning import (
    parse_scenes_from_slideshow_manifest,
)
from agents.graphics_definition_v2.slideshow_manifest.slideshow_manifest import (
    make_slot_narration_resolver_from_fgd,
)
from services.sheets_service import (
    clear_worksheet,
    format_worksheet,
    get_sheet_data_and_df,
    save_to_sheet,
)
from services.smart_progress_bar import SmartProgressBar

load_dotenv()

EDITED_IMAGE_DRIVE_FOLDER_ID = "1_cYkmnvDAUPardU1FyRHEU7Puwht6qZS"


#Allowed models for image editing:
# OpenAI: gpt-image-1.5, gpt-image-2
# Gemini: gemini-3.1-flash-image-preview, gemini-3-pro-image-preview
SCENE_IMAGE_EDIT_MODEL = "gemini-3-pro-image-preview"


SCENE_IMAGE_EDIT_REVIEW_MODEL = "gemini_3_flash_thinking"

MAX_EDIT_REVIEW_LOOPS = 3

_NON_INSTRUCTIONAL_EDIT_TYPES = frozenset({"NO_EDIT", "SKIPPED_VIDEO"})


def _resolve_scene_image_edit_review_model(model_id):
    """
    Resolve scene edit review alias to a concrete model id.

    :param model_id: Configured review model id or alias string.
    :return: Concrete model id string accepted by generate_content.
    """
    m = (model_id or "").strip()
    mapping = {
        "gemini_3_flash_thinking": "gemini-3-flash-preview",
        "gemini_3_flash": "gemini-3-flash-preview",
    }
    return mapping.get(m, m)

image_edit_execution_prompt = """You are a senior instructional image editing agent specializing in HVAC e-learning content. Your task is to apply the requested instructional edits to a single selected image asset based strictly on the provided edit plan.

This edited image will be used as a visual asset in a slideshow video, where the image will be placed on the slide canvas while the corresponding voiceover narration span plays in the background. Your edits should help the learner notice or understand the specific object, part, condition, action, relationship, or region that the edit plan identifies as instructionally important.

You must preserve the original image as much as possible. Do not redesign, restyle, reinterpret, or make unnecessary changes to the image. Apply only the edits explicitly requested in the edit plan, and keep all unrelated areas unchanged.

You will be given course context, slide context, the narration span for which this image will appear, the original asset URL, and the edit instructions for this image. The original image asset will also be provided as a multimodal image input. Review the image and apply only the requested edits.

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

<image_editing_context>
Voiceover Narration Span: {narration_span}
Original Assigned Visual Asset URL for this voiceover : {original_asset_url}
</image_editing_context>

<edit_instructions>
{edit_instructions}
</edit_instructions>

Instructions and Guidelines:

1. Core Responsibility
   - Your task is to apply the provided edit instructions to the provided input image only.
   - Apply only the edits explicitly listed in <edit_instructions>.
   - Do not create new edits, add extra visual elements, or make improvements that were not requested.
   - Do not redesign, restyle, or reinterpret the image.
   - Preserve the original image as closely as possible while applying the requested instructional edits.
   - Keep all unrelated areas of the image unchanged.
   - If multiple edits are provided, apply them in priority_order.
   - The final edited image should remain faithful to the original image, with only the requested changes added.

2. Context and Narration Understanding
   - Understand the narration span for which this image will appear in the slideshow video.
   - Use the full slide content to understand the instructional context, especially if the narration span is short, title-like, or depends on nearby slide text.
   - Use the course, topic, and subtopic information as supporting context for understanding the instructional purpose of the image.
   - Use the context to understand why the requested edits are needed, but do not use the context to invent new edits.
   - The edit instructions are the authority for what must be changed. The narration and slide context should only help you apply those edits accurately.

3. Image Fidelity and Change Control
   - Preserve the original image’s overall composition, perspective, proportions, lighting, visual style, and technical content.
   - Do not change the equipment, components, background, people, tools, or environment unless the edit instructions explicitly require it.
   - Keep all unedited regions as close to the original image as possible.
   - Do not add, remove, replace, or alter objects that are not part of the requested edits.
   - Do not crop, zoom, recolor, restyle, simplify, or enhance the image unless that specific edit is included in the edit instructions.
   - Do not make the image look like a new generated scene. The output should look like the original image with the requested instructional edits applied.
   - Do not obscure important technical details with labels, arrows, icons, highlights, or emphasis effects.
   - Follow any <must_preserve> instruction carefully so important visual context remains visible.

4. Edit Instruction Execution Rules
   - Read all edit instructions in <edit_instructions> before applying any changes.
   - Apply edits in the order specified by <priority_order>.
   - For each edit, use the <edit_type>, <target_description>, <label_text>, <reason_for_edit>, and <must_preserve> fields to understand exactly what to apply.
   - Treat <target_description> as the main source of truth for where the edit should be applied.
   - Use <label_text> only when the edit_type is ADD_TEXT_LABEL.
   - Use <reason_for_edit> only to understand the instructional purpose of the edit. Do not add extra edits based on the reason.
   - Follow <must_preserve> to avoid covering, cropping, or obscuring important visual information.
   - If two edit instructions affect the same area, apply them in a clean and readable way without cluttering the target region.
   - If an edit target cannot be clearly located in the image, apply the edit as accurately as possible based on visible evidence and the target description. Do not invent or alter the image to create the target.

5. Edit-Type Specific Rules
   - For ADD_TEXT_LABEL:
     - Add the exact text from <label_text>.
     - Keep the label short, readable, and placed close to the target without covering important visual details.
     - Use a clean instructional style that matches the original image and slideshow context.
     - If multiple labels are requested, keep placement consistent and avoid overlap.

   - For CROP_IMAGE:
     - Crop only as much as needed to make the target area easier to see.
     - Keep all objects, regions, or context mentioned in <must_preserve> visible.
     - Do not crop so tightly that the learner loses important context.

   - For ADD_ARROW:
     - Add an arrow pointing clearly to the target described in <target_description>.
     - Place the arrow so it is easy to follow and does not cover the target itself.
     - Do not add extra arrows beyond what is requested.

   - For ADD_HIGHLIGHT_CIRCLE_OR_BOX:
     - Add a clean circle or box around the target region described in <target_description>.
     - Use the shape that best fits the target region.
     - Keep the highlight clear but not visually overwhelming.

   - For ADD_ICON:
     - Add only the icon meaning requested or implied by the edit instruction.
     - Do not use icons that introduce new meaning not present in the edit instruction.
     - Simple line icons in a single color — orange (#F05523).
   - For ADD_EMPHASIS:
     - Apply the specific emphasis described in <target_description>, such as zoom focus, dimming unrelated background, or subtle visual focus.
     - Keep the emphasis effect instructional and restrained.
     - Do not distort the technical meaning or appearance of the image.

6. Visual Styling Rules
   - Use a clean instructional overlay style that is easy to read in an e-learning slideshow.
   - For all text labels, use Fira Sans.
   - Use the exact label text provided in <label_text>.
   - Label text must be large enough to read clearly on a slide canvas, but not so large that it covers important visual content.
   - Use orange (#F05523) for arrows, highlight circles, highlight boxes, and key emphasis overlays unless a different color is explicitly required in the edit instruction.
   - Use high contrast between labels/overlays and the image background so the edit is readable.
   - Place labels near their targets, but avoid covering the target or other important technical details.
   - Use simple straight or slightly curved arrows. Do not use decorative arrows.
   - Use clean, thin-to-medium stroke widths for arrows, circles, and boxes.
   - Keep icons simple, flat, and instructional. Do not use decorative or overly detailed icons.
   - Keep all overlays visually consistent across the image when multiple edits are applied.
   - Annotation Box & Border Styling:
     - The Box: A solid white rectangle with soft, rounded corners. The box must be small in size. If there are multiple annotation boxes in the image, all boxes must be of the exact same size.
     - The Border: A solid orange (#F05523) outline around the edges of the white box.
     - The Text: Plain, black text in Fira Sans font (bold for titles/labels; regular for body annotations), centered inside the box.
     - The Arrow (Connector): An orange (#F05523) arrow attached to any side of the box, pointing to the subject. The arrow must match the color and thickness of the box's border.
   - Split-Screen Collage Rule: Do not include annotations, text labels, callout boxes, or arrows when using a split-screen collage layout. Keep both halves of the image completely clean.

7. Readability and Placement Rules
   - All labels, arrows, icons, highlights, and emphasis effects must be clearly visible and easy to understand at slideshow viewing size.
   - Do not place labels, arrows, icons, or highlights over important technical details unless the edit specifically requires marking that exact area.
   - Keep labels close enough to their targets that the connection is clear.
   - When using arrows with labels, make sure the arrow clearly connects the label to the correct target.
   - Avoid overlapping labels, arrows, highlights, icons, or other overlays.
   - If the image background is busy or low-contrast, place the label in a readable area and use sufficient contrast so the text remains legible.
   - If multiple edits are applied, arrange them so the learner can still understand the image quickly without visual confusion.
   - Do not make overlays so large or visually dominant that they distract from the original image content.

<styling_guide>
{styling_guide}
</styling_guide>
"""


image_edit_review_prompt = """You are a senior instructional image edit review agent specializing in HVAC e-learning content. Your task is to review an edited image and determine whether the requested instructional edits were applied correctly, cleanly, and without damaging the original image.

CRITICAL REVIEW GUARDRAIL — READ FIRST AND STRICTLY ENFORCE:
- REJECT UNNECESSARY OVERLAYS: If the editor added redundant, obvious, or unnecessary labels, arrows, or highlight artifacts to a graphic that was already perfectly clear and instructionally complete in its original state, you MUST mark the image as FAIL.
- WHAT MAKES A GRAPHIC PERFECT: A graphic is perfect if it is a clean, professional photo, high-fidelity diagram, or realistic rendering that is self-explanatory. Adding arrows, highlight boxes, or basic labels (e.g., labeling an obvious 'air conditioner' or 'pipe') makes the visual look cluttered and amateurish.
- Revert feedback: In your failures feedback, instruct the editor: "Remove all added overlays/labels/arrows and revert the image back to its original state."

This edited image will be used as a visual asset in a slideshow video, where the image will be placed on the slide canvas while the corresponding voiceover narration span plays in the background. The image must therefore be clear, readable, visually clean, and instructionally useful for the narration moment.

You will be given the original image, the edited image, the edit instructions that were supposed to be applied, and the relevant course and slide context. Your responsibility is to compare the edited image against the original image and the edit instructions, then decide whether the edited image should PASS or FAIL.

A PASS means the requested edits were applied correctly and the edited image is clean, readable, faithful to the original image, and ready to use.

A FAIL means the edited image has meaningful issues that should be fixed before use. These issues may include incorrect edits, missing edits, wrong target selection, over-cropping, cropped-out text or context, unreadable labels, awkward highlights, unnecessary extra edits, visual artifacts, poor preservation of the original image, or any other visible problem that reduces instructional clarity or visual quality.

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

<image_editing_context>
Narration Span: {narration_span}
Original Asset URL: {original_asset_url}
Edited Asset URL: {edited_asset_url}
</image_editing_context>

<edit_instructions>
{edit_instructions}
</edit_instructions>

Instructions and Guidelines:

1. Core Responsibility
   - Inspect the provided edited image thoroughly.
   - Compare the edited image against the original image and the provided <edit_instructions>.
   - Determine whether the edited image is acceptable for use in the slideshow video.
   - A PASS should be given only when the requested edits are applied correctly, the image remains faithful to the original, and the final result is clear, readable, and instructionally useful.
   - A FAIL should be given when the edited image has meaningful problems that should be fixed before use.
   - Do not suggest new edits that were not requested unless they are needed to fix a visible fault introduced during editing.
   - **No Redundant Overlays**: Strongly reject edits that add unnecessary visual clutter, redundant labels, or extra artifacts to an already perfect and clear graphic.

2. Edit Instruction Compliance
   - Check whether every edit requested in <edit_instructions> was applied.
   - Check whether each edit was applied to the correct visible target described in <target_description>.
   - Check whether edits were applied in a way that supports the stated <reason_for_edit>.
   - Check whether all <must_preserve> requirements were respected.
   - If a requested edit is missing, applied to the wrong target, incomplete, or visibly different from what was requested, mark the image as FAIL.
   - If the edited image includes extra labels, arrows, highlights, icons, crops, or emphasis effects that were not requested, mark the image as FAIL unless the extra change is negligible and does not affect clarity.

3. Original Image Preservation
   - Compare the edited image with the original image.
   - The edited image should preserve the original composition, perspective, proportions, style, lighting, and technical content except where the requested edit required a change.
   - Check whether any important original text, label, object, component, person, tool, equipment, or background context was cropped out, covered, distorted, removed, or unintentionally changed.
   - Check whether the edited image looks like the original image with requested edits applied, not like a newly generated or redesigned image.
   - If unrelated parts of the image were changed in a meaningful way, mark the image as FAIL.
   - If the edit caused visual artifacts, distortion, poor image quality, or unnatural changes, mark the image as FAIL.

4. Crop Review Rules
   - If CROP_IMAGE was requested, check whether the crop focuses attention on the intended target without removing important context.
   - The crop must preserve all content listed in <must_preserve>.
   - The crop should not cut off important original text, labels, equipment, objects, or instructional visual context.
   - The crop should not be so tight that the image feels cramped or difficult to understand.
   - If CROP_IMAGE was not requested, the image should not be cropped or reframed in a meaningful way.
   - Mark the image as FAIL if the crop is too aggressive, cuts off important content, removes necessary context, or was applied when not requested.

5. Overlay Review Rules
   - Review any labels, arrows, highlight circles/boxes, icons, or emphasis effects added to the image.
   - Overlays must be placed on or near the correct target.
   - Overlays must be readable, clean, and visually clear at slideshow viewing size.
   - Overlays must not cover important technical details unless the requested edit specifically requires marking that exact area.
   - Labels should be legible, high contrast, and not cut off.
   - Arrows should clearly point to the correct target and should not obscure the target.
   - Highlight circles or boxes should fit the target region cleanly and should not look awkward, randomly placed, or visually excessive.
   - Icons should be simple, relevant, and not decorative.
   - Emphasis effects should help focus attention without distorting the image.
   - Mark the image as FAIL if overlays are unreadable, misplaced, excessive, visually awkward, overlapping, or harmful to image clarity.

6. Instructional Quality Review
   - Check whether the edited image helps the learner understand the narration span better than the original image and preserves the original instructional value.
   - The final image should be clear enough for the learner to quickly identify the target object, part, condition, action, relationship, or region that the edit was meant to clarify.
   - The image should not become more confusing, cluttered, or visually distracting after editing.
   - The image should remain appropriate for HVAC e-learning content and the target audience.
   - **Reject Unnecessary Overlays**: If the edit adds redundant, obvious, or unnecessary labels, arrows, or highlight artifacts to a graphic that was already perfectly clear and instructionally complete in its original state, you MUST mark the image as FAIL. In your feedback, instruct the editor: *"Remove all added overlays/labels/arrows and revert the image back to its original state."*
   - Mark the image as FAIL if the edit reduces instructional clarity, makes the image harder to understand, or distracts from the narration intent.

7. Open-Ended Fault Detection
   - In addition to the specific checks above, look for any other visible problem in the edited image that would make it unsuitable for use.
   - This includes but is not limited to unnatural artifacts, unexpected object changes, poor alignment, strange shapes, broken text, inconsistent styling, low readability, excessive visual clutter, or any issue that would make the image look unprofessional.

8. Failure Reporting and Revision Guidance
   - If the edited image has any meaningful issue(s), the verdict must be FAIL.
   - When giving a FAIL verdict, clearly describe the issue(s) that should be fixed before the image can be approved.
   - For each issue, explain what is wrong, why it is a problem, and what should be changed in the next revision.
   - The feedback must be specific and actionable so that a downstream image editing agent can use it directly for a revision pass.
   - If there are multiple issues, list them separately so they can be addressed clearly.

Output:

Always provide your output strictly in the following format:

<output>

<evaluation_breakdown>

Use this section as a structured reasoning and scratchpad space to compare the original image, edited image, edit instructions, and slide narration context before producing the final review verdict. Use the following sections:

1. Slide and Voiceover Understanding
- Briefly explain what the full slide is about.
- Explain what the narration span is communicating.

2. Original Image Review
- Briefly describe what is visibly shown in the original image.

3. Edited Image Review
- Briefly describe what is visibly shown in the edited image and what changes appear to have been made.

4. Edit Instruction Compliance
- Explain whether the requested edits from <edit_instructions> were applied correctly.

5. Original Image Preservation Review
- Explain whether the edited image preserved the important original content, composition, text, context, and technical meaning.

6. Visual Quality and Readability Review
- Explain whether edits like labels, arrows, highlights, icons, crops, or emphasis effects are clean, readable, correctly placed, and visually appropriate.

7. Additional Analysis
- Note any additional observations, visible faults, edge cases, or quality concerns that affect the verdict. It is ok for this section to be quite verbose and detailed as long as it allows you to do a thorough analysis and provide the correct output.

</evaluation_breakdown>

(Based on your above evaluation, provide the final review below.)

<review>

<verdict>
PASS | FAIL
</verdict>

(If the verdict is fail, then only provide the below fields)

<failures>

<failure>

<issue>
Clearly describe one specific issue found in the edited image.
</issue>

<reason>
Explain why this issue makes the edited image unsuitable or needs revision. 
</reason>

<revision_feedback>
Provide a detailed actionable feedback for how the editing agent should fix this issue in the next revision.
</revision_feedback>

</failure>

<!-- Repeat <failure> block for each separate issue. -->

</failures>

</review>

</output>

(Ensure that you strictly follow this exact output format. Do not add any extra text or comments outside the <output>, <evaluation_breakdown>, and <review> sections.)

<styling_guide>
{styling_guide}
</styling_guide>
"""


# Followup prompt to the editor when the reviewer rejects the previous edit.
edit_revision_followup_template = """The previous edited image you produced was reviewed and rejected during instructional quality assurance.

Below is the feedback received from the reviewer. Each <failure> block describes one issue, why it is a problem, and what should change in the next revision.

<reviewer_feedback>
{failures_xml}
</reviewer_feedback>

Your job is to now generate a new revised edited image so that all the failure blocks are addressed and resolved. Output only the new revised edited image.
"""


# Followup prompt to the reviewer when a revised edited image has been produced.
review_revision_followup_template = """A revised edited image has been produced based on all of your previous feedback.

Review this revised edited image against the original image and all the feedback that you had generated. Output your review in the same format as before.
Be very strict in your evaluation. Do not give a PASS verdict if any issue still exists even after multiple rounds of revisions. The verdict of "PASS" should only be assigned after you are completely satisfied that the revised image has all the issues addressed and resolved.

Remember: Strictly use the same output format as before while reviewing the revised image and providing your output.
"""


def _sanitize_filename_component(name, fallback="slide"):
    """
    Return a filesystem-safe filename fragment derived from slide title or similar.

    :param name: Raw title or label text.
    :param fallback: Value used when name is empty after sanitization.
    :return: Safe string for use inside a filename.
    """
    if not name:
        return fallback
    safe = re.sub(r"[^A-Za-z0-9._\-]+", "_", str(name)).strip("._")
    return safe or fallback


def _get_openai_client():
    """
    Build an OpenAI client for image edits (no Streamlit UI on failure).

    :return: OpenAI client or None when API key is missing.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Image edit execution: Missing OPENAI_API_KEY")
        return None
    try:
        return OpenAI(api_key=api_key)
    except Exception as e:
        print(f"Image edit execution: Failed to initialize OpenAI client: {e}")
        return None


def _get_gemini_client():
    """
    Google GenAI client for Nano Banana 2 (same key resolution as other Gemini callers).

    :return: Google GenAI client or None when API key is missing.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        api_key = st.session_state.get("google_api_key")
    if not api_key:
        print("Image edit execution: Missing GOOGLE_API_KEY (or session google_api_key)")
        return None
    try:
        return genai.Client(api_key=api_key)
    except Exception as e:
        print(f"Image edit execution: Failed to initialize Gemini client: {e}")
        return None


def _approx_gemini_aspect_ratio(reference_image):
    """
    Map image dimensions to a supported Gemini ImageConfig aspect_ratio string.

    :param reference_image: Source PIL image to approximate.
    :return: Aspect ratio label accepted by the image API.
    """
    w, h = reference_image.size
    if w <= 0 or h <= 0:
        return "4:3"
    ratio = w / h
    candidates = (
        (1.0, "1:1"),
        (4 / 3, "4:3"),
        (3 / 4, "3:4"),
        (16 / 9, "16:9"),
        (9 / 16, "9:16"),
        (3 / 2, "3:2"),
        (2 / 3, "2:3"),
    )
    return min(candidates, key=lambda ch: abs(ratio - ch[0]))[1]


def split_scene_edit_plan_by_scene(scene_edit_plan_text):
    """
    Split concatenated scene_edit_plan cell text into per-scene bodies.

    :param scene_edit_plan_text: Full scene_edit_plan column value.
    :return: List of (scene_id, scene_body_after_header) in document order.
    """
    text = (scene_edit_plan_text or "").strip()
    if not text:
        return []
    pattern = re.compile(r"---Scene ID:\s*(\d+)\s*---\s*", re.MULTILINE)
    matches = list(pattern.finditer(text))
    if not matches:
        return []
    out = []
    for i, m in enumerate(matches):
        scene_id = (m.group(1) or "").strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        out.append((scene_id, body))
    return out


def extract_slot_edit_blocks(scene_body):
    """
    Extract inner XML for each <slot_edit> block in one scene body.

    :param scene_body: Text after the scene header for one scene.
    :return: List of slot_edit inner strings (without outer <slot_edit> tags).
    """
    if not scene_body:
        return []
    found = re.findall(
        r"<slot_edit>(.*?)</slot_edit>",
        scene_body,
        flags=re.DOTALL | re.IGNORECASE,
    )
    return [s.strip() for s in found if s.strip()]


def _parse_multiline_field(slot_xml, label):
    """
    Read a single-line field value after Label colon in slot XML.

    :param slot_xml: One slot_edit inner block.
    :param label: Field label without trailing colon (e.g. Asset URL).
    :return: Stripped value or empty string.
    """
    m = re.search(rf"^{re.escape(label)}:\s*(.+)$", slot_xml, re.MULTILINE)
    return m.group(1).strip() if m else ""


def _parse_edit_required(slot_xml):
    """
    Parse <edit_required>YES|NO</edit_required> from slot XML.

    :param slot_xml: One slot_edit inner block.
    :return: Uppercase YES/NO or empty string if missing.
    """
    m = re.search(
        r"<edit_required>\s*(YES|NO)\s*</edit_required>",
        slot_xml,
        re.IGNORECASE | re.DOTALL,
    )
    return m.group(1).strip().upper() if m else ""


def _extract_edits_section_inner(slot_xml):
    """
    Return the inner text of the first <edits>...</edits> block.

    :param slot_xml: One slot_edit inner block.
    :return: Inner XML string (may contain multiple <edit> elements).
    """
    m = re.search(r"<edits>(.*?)</edits>", slot_xml, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""


def _extract_edit_blocks_xml(edits_inner):
    """
    Split edits inner XML into individual <edit>...</edit> fragments (with tags).

    :param edits_inner: Content inside <edits>.
    :return: List of full <edit>...</edit> strings.
    """
    if not edits_inner:
        return []
    return re.findall(r"(<edit>.*?</edit>)", edits_inner, re.DOTALL | re.IGNORECASE)


def _edit_type_from_edit_block(edit_xml):
    """
    Parse <edit_type> value from one <edit> fragment.

    :param edit_xml: Full <edit>...</edit> string.
    :return: Uppercased edit_type token or empty string.
    """
    m = re.search(
        r"<edit_type>\s*(.*?)\s*</edit_type>",
        edit_xml,
        re.DOTALL | re.IGNORECASE,
    )
    if not m:
        return ""
    return re.sub(r"\s+", " ", m.group(1)).strip().upper()


def slot_needs_instructional_image_edit(slot_xml):
    """
    Determine whether this slot should be sent to the configured image edit backend.

    Skips when edit_required is NO, asset is video, or all edits are NO_EDIT/SKIPPED_VIDEO.

    :param slot_xml: Inner content of one <slot_edit> block.
    :return: True when an instructional image edit should run.
    """
    asset_type = _parse_multiline_field(slot_xml, "Asset Type").lower()
    if asset_type == "video":
        return False
    if _parse_edit_required(slot_xml) != "YES":
        return False
    edits_inner = _extract_edits_section_inner(slot_xml)
    blocks = _extract_edit_blocks_xml(edits_inner)
    if not blocks:
        return False
    for blk in blocks:
        et = _edit_type_from_edit_block(blk)
        if et and et not in _NON_INSTRUCTIONAL_EDIT_TYPES:
            return True
    return False


def build_edit_instructions_payload(slot_xml):
    """
    Build the inner body for edit_instructions in the execution prompt template.

    :param slot_xml: Inner content of one slot_edit block.
    :return: String inserted into image_edit_execution_prompt at edit_instructions.
    """
    edits_inner = _extract_edits_section_inner(slot_xml)
    blocks = _extract_edit_blocks_xml(edits_inner)
    instructional = [
        b for b in blocks
        if _edit_type_from_edit_block(b) not in _NON_INSTRUCTIONAL_EDIT_TYPES
    ]
    inner = "\n\n".join(instructional) if instructional else "\n\n".join(blocks)
    return inner


def run_gpt_image_edit_with_prompt(reference_image, full_prompt, *, model):
    """
    Call OpenAI images.edit with a fully assembled text prompt.

    :param reference_image: PIL image to edit.
    :param full_prompt: Complete user prompt including edit instructions.
    :param model: OpenAI image model id (e.g. gpt-image-1.5, gpt-image-2).
    :return: Edited PIL image in RGB, or raises ValueError if client is missing, response has no image data, or the payload cannot be decoded.
    """
    client = _get_openai_client()
    if client is None:
        raise ValueError("Missing OpenAI client (OPENAI_API_KEY).")

    image_bytes_io = prepare_image_for_upload(reference_image)
    try:
        response = client.images.edit(
            model=model,
            image=image_bytes_io,
            prompt=full_prompt,
            size="1024x1024",
            # input_fidelity: match reference pixels; quality: OpenAI's latency/quality knob for gpt-image
            # (not the same as Gemini "thinking", but there is no separate reasoning parameter on this API).
            extra_body={"input_fidelity": "high", "quality": "high"},
        )
    except Exception as e:
        print(f"Image edit execution: OpenAI images.edit failed: {e}")
        raise

    if not response.data:
        raise ValueError("Model returned no image data.")

    data_item = response.data[0]
    edited_image = None
    if hasattr(data_item, "b64_json") and data_item.b64_json:
        edited_image = image_from_base64(data_item.b64_json)
    elif hasattr(data_item, "url") and data_item.url:
        edited_image = download_image_from_url(data_item.url)
    else:
        try:
            if isinstance(data_item, dict):
                if data_item.get("b64_json"):
                    edited_image = image_from_base64(data_item["b64_json"])
                elif data_item.get("url"):
                    edited_image = download_image_from_url(data_item["url"])
        except Exception:
            pass

    if edited_image is None:
        raise ValueError("Failed to extract edited image from API response.")

    if edited_image.mode != "RGB":
        edited_image = edited_image.convert("RGB")
    _print_scene_edit_llm_in_use(
        "OpenAI images.edit (scene image edit)",
        model,
        "openai",
        "thinking=n/a (not available on images.edit) | "
        "extra_body: input_fidelity=high, quality=high | size=1024x1024",
    )
    return edited_image


def run_gemini_image_edit_with_prompt(reference_image, full_prompt, *, model):
    """
    Call Google GenAI image-capable Gemini with reference image and full text prompt.

    :param reference_image: PIL image to edit.
    :param full_prompt: Complete user prompt including edit instructions.
    :param model: Gemini image model id (e.g. gemini-3.1-flash-image-preview, gemini-3-pro-image-preview).
    :return: Edited PIL image in RGB, or raises ValueError if client is missing, response has no usable image, or the model refuses the request.
    """
    client = _get_gemini_client()
    if client is None:
        raise ValueError("Missing Google GenAI client (GOOGLE_API_KEY).")

    image_bytes = prepare_image_for_gemini(reference_image)
    aspect_ratio = _approx_gemini_aspect_ratio(reference_image)
    parts = [
        types.Part.from_text(
            text="Original Image: Apply all requested edits from the edit plan to this image."
        ),
        types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
        types.Part.from_text(text=full_prompt),
    ]
    user_content = types.Content(role="user", parts=parts)
    image_config_obj = types.ImageConfig(image_size="1K", aspect_ratio=aspect_ratio)

    def _should_retry_gemini_without_thinking(exc):
        """Return True if the Gemini error suggests retrying without thinking_config."""
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
    used_config = None
    for idx, cfg in enumerate(configs_to_try):
        try:
            with tracker.call(model, "Scene Image Edit (Gemini)") as usage:
                response = call_llm_with_retry(
                    client.models.generate_content,
                    model=model,
                    contents=[user_content],
                    config=cfg,
                )
                usage.set_response(response)
            used_config = cfg
            break
        except Exception as e:
            if (
                idx == 0
                and len(configs_to_try) > 1
                and _should_retry_gemini_without_thinking(e)
            ):
                print(
                    "Image edit execution: Gemini rejected thinking_config; "
                    "retrying generate_content without thinking_config."
                )
                continue
            print(f"Image edit execution: Gemini {model} failed: {e}")
            raise

    assert response is not None
    _print_scene_edit_llm_in_use(
        "Gemini generate_content (scene image edit, run_scene_image_edit path)",
        model,
        "gemini",
        _gemini_thinking_label_for_config(used_config),
    )
    if not response.candidates or not response.candidates[0].content:
        raise ValueError("Model returned no content.")

    edited_image = None
    model_content = response.candidates[0].content
    if hasattr(model_content, "parts"):
        for part in model_content.parts:
            if hasattr(part, "inline_data") and part.inline_data:
                try:
                    edited_image = Image.open(BytesIO(part.inline_data.data))
                    break
                except Exception as part_err:
                    print(f"Image edit execution: error reading Gemini image part: {part_err}")

    if edited_image is None and hasattr(response, "text"):
        try:
            edited_image = image_from_base64(response.text)
        except Exception:
            pass

    if edited_image is None:
        raise ValueError("No edited image found. The model may have refused the request.")

    if edited_image.mode != "RGB":
        edited_image = edited_image.convert("RGB")
    return edited_image


def _scene_image_edit_provider(model_id):
    """
    Infer which API to call from SCENE_IMAGE_EDIT_MODEL.

    :param model_id: Image model id string (OpenAI or Gemini prefix).
    :return: The string openai or gemini.
    """
    m = (model_id or "").strip().lower()
    if m.startswith("gpt-image"):
        return "openai"
    if m.startswith("gemini"):
        return "gemini"
    raise ValueError(
        f"Unrecognized SCENE_IMAGE_EDIT_MODEL {model_id!r}. "
        "Use an OpenAI id starting with gpt-image- (e.g. gpt-image-1.5, gpt-image-2) "
        "or a Gemini id starting with gemini- (e.g. gemini-3.1-flash-image-preview, "
        "gemini-3-pro-image-preview)."
    )


def run_scene_image_edit_with_prompt(reference_image, full_prompt):
    """
    Run image edit using SCENE_IMAGE_EDIT_MODEL (OpenAI vs Gemini inferred from the id).

    :param reference_image: PIL image to edit.
    :param full_prompt: Complete user prompt.
    :return: Edited PIL image (RGB).
    """
    model = SCENE_IMAGE_EDIT_MODEL.strip()
    provider = _scene_image_edit_provider(model)
    if provider == "openai":
        return run_gpt_image_edit_with_prompt(reference_image, full_prompt, model=model)
    return run_gemini_image_edit_with_prompt(reference_image, full_prompt, model=model)


def _narration_by_scene_id(manifest_text):
    """
    Map scene id string to narration span from slideshow_manifest.

    :param manifest_text: slideshow_manifest column value.
    :return: Dict scene_id -> narration string.
    """
    scenes = parse_scenes_from_slideshow_manifest(manifest_text)
    out = {}
    for sc in scenes:
        sid = str(sc.get("id", "")).strip()
        if sid:
            out[sid] = str(sc.get("narration", "") or "").strip()
    return out


# =====================================================================
# Review-regenerate loop for Scene Edit Execution
# =====================================================================

_REVIEW_VERDICT_RE = re.compile(
    r"<verdict>\s*(PASS|FAIL)\s*</verdict>",
    re.DOTALL | re.IGNORECASE,
)
_REVIEW_FAILURES_RE = re.compile(
    r"<failures>.*?</failures>",
    re.DOTALL | re.IGNORECASE,
)
_FAILURES_OUTER_RE = re.compile(
    r"<failures\b[^>]*>\s*(.*?)\s*</failures>\s*\Z",
    re.DOTALL | re.IGNORECASE,
)


def _inner_failures_xml(failures_block):
    """
    Return only the XML inside the outer failures wrapper from the reviewer.

    :param failures_block: Full failures XML string or bare inner content.
    :return: Stripped inner XML, or the whole string stripped if no outer failures tags.
    """
    t = (failures_block or "").strip()
    if not t:
        return ""
    m = _FAILURES_OUTER_RE.match(t)
    if m:
        return m.group(1).strip()
    return t


def _parse_review_response(text):
    """
    Extract verdict and failures block from review model output.

    :param text: Raw model response text.
    :return: Tuple of verdict string, failures XML string including failures tags, and raw response text. Verdict is PASS, FAIL, or empty if not parsed.
    """
    raw = (text or "").strip()
    if not raw:
        return "", "", ""
    verdict_match = _REVIEW_VERDICT_RE.search(raw)
    verdict = verdict_match.group(1).strip().upper() if verdict_match else ""
    failures_match = _REVIEW_FAILURES_RE.search(raw)
    failures = failures_match.group(0).strip() if failures_match else ""
    return verdict, failures, raw


def _gemini_should_retry_without_thinking(exc):
    """
    Return True when a Gemini error suggests retrying without thinking_config.

    :param exc: Exception raised by the SDK.
    :return: True if the error message indicates thinking_config should be dropped.
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


def _gemini_thinking_label_for_config(cfg):
    """
    Describe whether generate_content was called with Gemini thinking_config.

    :param cfg: GenerateContentConfig used for the successful request, or None.
    :return: Short human-readable string for logs.
    """
    if cfg is None:
        return "thinking=unknown"
    tc = getattr(cfg, "thinking_config", None)
    if tc is None:
        return "thinking=off (request had no thinking_config)"
    level = getattr(tc, "thinking_level", None)
    if level is not None:
        return "thinking=on (thinking_level=%r)" % (level,)
    return "thinking=on (thinking_config present)"


def _print_scene_edit_llm_in_use(call_kind, model_id, backend, details):
    """
    Log which model/backend/options were used right after a prompt is sent to the API.

    :param call_kind: Short label for the call site.
    :param model_id: Model id passed to the provider.
    :param backend: Provider name, e.g. openai or gemini.
    :param details: Extra text, e.g. thinking label or OpenAI extra_body summary.
    :return: None
    """
    print(
        "Image edit execution — LLM in use — %s | backend=%s | model=%r | %s"
        % (call_kind, backend, model_id, details)
    )


def _gemini_image_edit_call_with_contents(client, model, contents, aspect_ratio=None):
    """
    Call Gemini generate_content for image output with a prebuilt contents list.

    Tries high thinking for gemini-3 models and falls back without it if the API rejects the field.

    :param client: Google GenAI client instance.
    :param model: Gemini model id.
    :param contents: List of Content objects forming the chat history.
    :param aspect_ratio: Optional aspect ratio label for ImageConfig.
    :return: Tuple of edited PIL image in RGB and model Content for appending to chat history.
    """
    image_config_kwargs = {"image_size": "1K"}
    if aspect_ratio:
        image_config_kwargs["aspect_ratio"] = aspect_ratio
    image_config_obj = types.ImageConfig(**image_config_kwargs)

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
    used_config = None
    for idx, cfg in enumerate(configs_to_try):
        try:
            with tracker.call(model, "Scene Image Edit (Gemini)") as usage:
                response = call_llm_with_retry(
                    client.models.generate_content,
                    model=model,
                    contents=contents,
                    config=cfg,
                )
                usage.set_response(response)
            used_config = cfg
            break
        except Exception as e:
            if (
                idx == 0
                and len(configs_to_try) > 1
                and _gemini_should_retry_without_thinking(e)
            ):
                print(
                    "Image edit execution: Gemini rejected thinking_config; "
                    "retrying generate_content without thinking_config."
                )
                continue
            print(f"Image edit execution: Gemini {model} failed: {e}")
            raise

    assert response is not None
    _print_scene_edit_llm_in_use(
        "Gemini generate_content (scene image edit, chat path)",
        model,
        "gemini",
        _gemini_thinking_label_for_config(used_config),
    )
    if not response.candidates or not response.candidates[0].content:
        raise ValueError("Model returned no content.")

    edited_image = None
    model_content = response.candidates[0].content
    if hasattr(model_content, "parts"):
        for part in model_content.parts:
            if hasattr(part, "inline_data") and part.inline_data:
                try:
                    edited_image = Image.open(BytesIO(part.inline_data.data))
                    break
                except Exception as part_err:
                    print(
                        f"Image edit execution: error reading Gemini image part: {part_err}"
                    )

    if edited_image is None and hasattr(response, "text"):
        try:
            edited_image = image_from_base64(response.text)
        except Exception:
            pass

    if edited_image is None:
        raise ValueError(
            "No edited image found. The model may have refused the request."
        )

    if edited_image.mode != "RGB":
        edited_image = edited_image.convert("RGB")
    return edited_image, model_content


def _gemini_review_call_with_contents(client, model, contents):
    """
    Call Gemini generate_content for text-only review output.

    Tries high thinking for gemini-3 models and falls back without it on rejection.

    :param client: Google GenAI client instance.
    :param model: Gemini review model id.
    :param contents: List of Content objects forming the chat history.
    :return: Tuple of response text and model Content for chat history extension.
    """
    if "gemini-3" in (model or "").lower():
        configs_to_try = [
            types.GenerateContentConfig(
                response_modalities=["TEXT"],
                thinking_config=types.ThinkingConfig(thinking_level="high"),
            ),
            types.GenerateContentConfig(
                response_modalities=["TEXT"],
            ),
        ]
    else:
        configs_to_try = [
            types.GenerateContentConfig(response_modalities=["TEXT"]),
        ]

    response = None
    used_config = None
    for idx, cfg in enumerate(configs_to_try):
        current_thinking_label = _gemini_thinking_label_for_config(cfg)
        print(
            "Image edit review: Gemini request config "
            f"(attempt {idx + 1}/{len(configs_to_try)}) | "
            f"model={model!r} | {current_thinking_label}"
        )
        try:
            with tracker.call(model, "Scene Image Edit Review (Gemini)") as usage:
                response = call_llm_with_retry(
                    client.models.generate_content,
                    model=model,
                    contents=contents,
                    config=cfg,
                )
                usage.set_response(response)
            used_config = cfg
            print(
                "Image edit review: Gemini request succeeded | "
                f"model={model!r} | {_gemini_thinking_label_for_config(used_config)}"
            )
            break
        except Exception as e:
            if (
                idx == 0
                and len(configs_to_try) > 1
                and _gemini_should_retry_without_thinking(e)
            ):
                print(
                    "Image edit review: Gemini rejected thinking_config; "
                    "retrying generate_content without thinking_config."
                )
                continue
            print(f"Image edit review: Gemini {model} failed: {e}")
            raise

    assert response is not None
    _print_scene_edit_llm_in_use(
        "Gemini generate_content (scene image edit review)",
        model,
        "gemini",
        _gemini_thinking_label_for_config(used_config),
    )
    if not response.candidates or not response.candidates[0].content:
        raise ValueError("Review model returned no content.")

    model_content = response.candidates[0].content
    response_text = ""
    if hasattr(model_content, "parts"):
        chunks = []
        for part in model_content.parts:
            if hasattr(part, "text") and part.text:
                chunks.append(part.text)
        response_text = "\n".join(chunks).strip()
    if not response_text and hasattr(response, "text") and response.text:
        response_text = (response.text or "").strip()
    return response_text, model_content


def _gemini_edit_first_turn(reference_image, full_prompt, *, model):
    """
    Begin a Gemini edit chat (Chat 1, turn 1).

    :param reference_image: PIL reference image to edit.
    :param full_prompt: Full image_edit_execution_prompt text for this slot.
    :param model: Gemini image model id.
    :return: Tuple of edited PIL image and history list of user and model Content turns.
    """
    client = _get_gemini_client()
    if client is None:
        raise ValueError("Missing Google GenAI client (GOOGLE_API_KEY).")

    image_bytes = prepare_image_for_gemini(reference_image)
    aspect_ratio = _approx_gemini_aspect_ratio(reference_image)
    parts = [
        types.Part.from_text(
            text="Original Image: Apply all requested edits from the edit plan to this image."
        ),
        types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
        types.Part.from_text(text=full_prompt),
    ]
    user_content = types.Content(role="user", parts=parts)
    edited_pil, model_content = _gemini_image_edit_call_with_contents(
        client, model, [user_content], aspect_ratio=aspect_ratio
    )
    history = [user_content, model_content]
    return edited_pil, history


def _gemini_edit_followup_turn(history, follow_up_text, *, model, original_image, last_edited_image, original_image_url, last_edited_image_url, follow_up_edit_loop_num, aspect_ratio=None):
    """
    Continue a Gemini edit chat with a multimodal follow-up turn.

    :param history: Existing list of Content objects, mutated in place.
    :param follow_up_text: Formatted edit_revision_followup_template text.
    :param model: Gemini image model id.
    :param original_image: PIL original reference image.
    :param last_edited_image: PIL image from the edit pass that failed review.
    :param original_image_url: URL string for the original asset.
    :param last_edited_image_url: URL string for the last edited upload.
    :param follow_up_edit_loop_num: 1-based edit loop number for this regeneration.
    :param aspect_ratio: Optional aspect ratio for ImageConfig (defaults from original_image).
    :return: Tuple of edited PIL image and updated history list.
    """
    client = _get_gemini_client()
    if client is None:
        raise ValueError("Missing Google GenAI client (GOOGLE_API_KEY).")

    if aspect_ratio is None:
        aspect_ratio = _approx_gemini_aspect_ratio(original_image)
    orig_bytes = prepare_image_for_gemini(original_image)
    edit_bytes = prepare_image_for_gemini(last_edited_image)
    parts = [
        types.Part.from_text(
            text=f"Follow-up prompt (edit loop {follow_up_edit_loop_num})."
        ),
        types.Part.from_text(text=f"Original Image: {original_image_url}"),
        types.Part.from_bytes(data=orig_bytes, mime_type="image/jpeg"),
        types.Part.from_text(text=f"Edited Image: {last_edited_image_url}"),
        types.Part.from_bytes(data=edit_bytes, mime_type="image/jpeg"),
        types.Part.from_text(text=follow_up_text),
    ]
    user_content = types.Content(
        role="user",
        parts=parts,
    )
    contents = list(history) + [user_content]
    edited_pil, model_content = _gemini_image_edit_call_with_contents(
        client, model, contents, aspect_ratio=aspect_ratio
    )
    history.append(user_content)
    history.append(model_content)
    return edited_pil, history


def _openai_edit_first_turn(reference_image, full_prompt, *, model):
    """
    Begin an OpenAI edit chat (stateless; state holds the base prompt only).

    :param reference_image: PIL reference image to edit.
    :param full_prompt: Full image_edit_execution_prompt text for this slot.
    :param model: OpenAI image model id.
    :return: Tuple of edited PIL image and state dict with key base_prompt for follow-up turns.
    """
    edited = run_gpt_image_edit_with_prompt(reference_image, full_prompt, model=model)
    return edited, {"base_prompt": full_prompt}


def _openai_edit_followup_turn(reference_image, state, follow_up_text, *, model, original_image_url=None, last_edited_image_url=None, follow_up_edit_loop_num=None):
    """
    Continue an OpenAI edit chat by re-sending the reference image and a composed prompt.

    OpenAI images.edit accepts one image file, so URLs for original and last edit are prepended as text; the API image is still the original reference only.

    :param reference_image: PIL reference image.
    :param state: Dict from _openai_edit_first_turn with base_prompt.
    :param follow_up_text: Short follow-up prompt containing review feedback.
    :param model: OpenAI image model id.
    :param original_image_url: Optional URL string echoed in the prompt.
    :param last_edited_image_url: Optional URL for the rejected edit (text only).
    :param follow_up_edit_loop_num: Optional loop label for the follow-up block.
    :return: Tuple of edited PIL image and the same state dict.
    """
    header_lines = []
    if follow_up_edit_loop_num is not None:
        header_lines.append(
            f"Follow-up prompt (edit loop {follow_up_edit_loop_num})."
        )
    if original_image_url:
        header_lines.append(f"Original Image: {original_image_url}")
    if last_edited_image_url:
        header_lines.append(f"Edited Image: {last_edited_image_url}")
    follow_block = follow_up_text.strip()
    if header_lines:
        follow_block = "\n\n".join(header_lines) + "\n\n" + follow_block
    composed_prompt = state["base_prompt"].rstrip() + "\n\n" + follow_block
    edited = run_gpt_image_edit_with_prompt(
        reference_image, composed_prompt, model=model
    )
    return edited, state


def _edit_chat_first_turn(reference_image, full_prompt, *, model):
    """
    Dispatch the first edit-chat turn to OpenAI or Gemini based on model id.

    :param reference_image: PIL reference image to edit.
    :param full_prompt: Full execution prompt for this slot.
    :param model: Edit model id (SCENE_IMAGE_EDIT_MODEL).
    :return: Tuple of edited PIL image and history or state object for follow-up turns.
    """
    provider = _scene_image_edit_provider(model)
    if provider == "openai":
        return _openai_edit_first_turn(reference_image, full_prompt, model=model)
    return _gemini_edit_first_turn(reference_image, full_prompt, model=model)


def _edit_chat_followup_turn(reference_image, history_or_state, follow_up_text, *, model, original_image_url=None, last_edited_image=None, last_edited_image_url=None, follow_up_edit_loop_num=None):
    """
    Dispatch a follow-up edit-chat turn to OpenAI or Gemini.

    :param reference_image: PIL reference image (OpenAI path must re-send it).
    :param history_or_state: Gemini history list or OpenAI state dict.
    :param follow_up_text: Short follow-up prompt with reviewer feedback.
    :param model: Edit model id.
    :param original_image_url: Original asset URL (Gemini multimodal + OpenAI text).
    :param last_edited_image: PIL of the edit output that failed review (Gemini only).
    :param last_edited_image_url: Drive URL for that edited image.
    :param follow_up_edit_loop_num: Edit loop number for this follow-up (e.g. 2, 3).
    :return: Tuple of edited PIL image and updated history or state.
    """
    provider = _scene_image_edit_provider(model)
    if provider == "openai":
        return _openai_edit_followup_turn(
            reference_image,
            history_or_state,
            follow_up_text,
            model=model,
            original_image_url=original_image_url,
            last_edited_image_url=last_edited_image_url,
            follow_up_edit_loop_num=follow_up_edit_loop_num,
        )
    return _gemini_edit_followup_turn(
        history_or_state,
        follow_up_text,
        model=model,
        original_image=reference_image,
        last_edited_image=last_edited_image,
        original_image_url=original_image_url or "",
        last_edited_image_url=last_edited_image_url or "",
        follow_up_edit_loop_num=follow_up_edit_loop_num or 0,
    )


def _review_chat_first_turn(original_image, edited_image, full_review_prompt, *, model):
    """
    Begin the reviewer chat (Chat 2, turn 1) on Gemini with multimodal inputs.

    :param original_image: PIL original reference image.
    :param edited_image: PIL first edited image to review.
    :param full_review_prompt: Full image_edit_review_prompt text for this slot.
    :param model: Gemini review model id.
    :return: Tuple of verdict string, failures XML string, raw response text, and history list.
    """
    client = _get_gemini_client()
    if client is None:
        raise ValueError("Missing Google GenAI client (GOOGLE_API_KEY).")

    orig_bytes = prepare_image_for_gemini(original_image)
    edit_bytes = prepare_image_for_gemini(edited_image)
    parts = [
        types.Part.from_text(
            text="Image 1 (ORIGINAL reference image, before any edits)."
        ),
        types.Part.from_bytes(data=orig_bytes, mime_type="image/jpeg"),
        types.Part.from_text(
            text="Image 2 (EDITED image to review, after applying the edit instructions)."
        ),
        types.Part.from_bytes(data=edit_bytes, mime_type="image/jpeg"),
        types.Part.from_text(text=full_review_prompt),
    ]
    user_content = types.Content(role="user", parts=parts)
    response_text, model_content = _gemini_review_call_with_contents(
        client, model, [user_content]
    )
    verdict, failures, raw = _parse_review_response(response_text)
    history = [user_content, model_content]
    return verdict, failures, raw, history


def _review_chat_followup_turn(new_edited_image, history, follow_up_text, *, model):
    """
    Continue the reviewer chat with a new revised edited image (Chat 2, later turns).

    :param new_edited_image: PIL revised edited image to review.
    :param history: Existing list of Content objects, mutated in place.
    :param follow_up_text: Short follow-up prompt.
    :param model: Gemini review model id.
    :return: Tuple of verdict string, failures XML string, raw response text, and history list.
    """
    client = _get_gemini_client()
    if client is None:
        raise ValueError("Missing Google GenAI client (GOOGLE_API_KEY).")

    edit_bytes = prepare_image_for_gemini(new_edited_image)
    parts = [
        types.Part.from_text(
            text="Revised edited image, replacing the previously reviewed edited image."
        ),
        types.Part.from_bytes(data=edit_bytes, mime_type="image/jpeg"),
        types.Part.from_text(text=follow_up_text),
    ]
    user_content = types.Content(role="user", parts=parts)
    contents = list(history) + [user_content]
    response_text, model_content = _gemini_review_call_with_contents(
        client, model, contents
    )
    verdict, failures, raw = _parse_review_response(response_text)
    history.append(user_content)
    history.append(model_content)
    return verdict, failures, raw, history


def _failures_for_followup(failures_block, raw_text):
    """
    Build inner XML for the editor follow-up template failures_xml placeholder.

    :param failures_block: Parsed failures XML from the reviewer, or empty.
    :param raw_text: Full raw review text when failures could not be parsed.
    :return: Inner XML string for edit_revision_followup_template, or one synthetic failure block built from raw_text.
    """
    if failures_block and failures_block.strip():
        return _inner_failures_xml(failures_block.strip())
    raw = (raw_text or "").strip()
    if not raw:
        return ""
    return (
        "<failure>\n"
        "  <issue>The previous review did not return a strictly structured "
        "verdict / failures block.</issue>\n"
        "  <reason>The reviewer's raw response is preserved below so you can "
        "still react to the most important concerns.</reason>\n"
        "  <revision_feedback>" + raw[:4000] + "</revision_feedback>\n"
        "</failure>"
    )


def _format_slot_outputs(loops, asset_url, image_index):
    """
    Build per-slot tracking lines and edit_review lines from loop records.

    :param loops: List of dicts from _run_edit_review_loop_for_slot with loop_num, edit_url, edit_error, verdict, failures, review_error, raw_review.
    :param asset_url: Original asset URL for this slot.
    :param image_index: 1-based image index within the scene.
    :return: Tuple of tracking_lines list and review_lines list (caller adds Image k headers for tracking).
    """
    tracking_lines = [f"Original Image: {asset_url}"]
    review_lines = []

    for r in loops:
        if r.get("edit_url"):
            tracking_lines.append(
                f"Edited Image after loop {r['loop_num']}: {r['edit_url']}"
            )
        elif r.get("edit_error"):
            tracking_lines.append(
                f"ERROR: Image edit (loop {r['loop_num']}) failed: {r['edit_error']}"
            )

    review_loops = [
        r for r in loops if r.get("verdict") or r.get("review_error")
    ]

    if not review_loops:
        review_lines.append(f"Image {image_index}:")
        first_edit_err = next(
            (r.get("edit_error") for r in loops if r.get("edit_error")),
            None,
        )
        if first_edit_err:
            review_lines.append(
                f"ERROR: Edit failed before review: {first_edit_err}"
            )
        else:
            review_lines.append("ERROR: No review available")
        return tracking_lines, review_lines

    only_one_pass = (
        len(review_loops) == 1
        and (review_loops[0].get("verdict") or "").upper() == "PASS"
    )
    if only_one_pass:
        review_lines.append(f"Image {image_index}:")
        review_lines.append("PASS")
        return tracking_lines, review_lines

    for i, r in enumerate(review_loops):
        if i > 0:
            review_lines.append("")
        review_lines.append(f"Image {image_index}, Loop {r['loop_num']}:")
        if r.get("review_error"):
            review_lines.append(f"ERROR: Review failed: {r['review_error']}")
            continue
        verdict = (r.get("verdict") or "").upper()
        if verdict == "PASS":
            review_lines.append("PASS")
            continue
        review_lines.append(verdict or "FAIL")
        failures = r.get("failures") or ""
        if failures.strip():
            review_lines.append(failures.strip())
    return tracking_lines, review_lines


def _run_edit_review_loop_for_slot(scene_id, image_index, asset_url, reference_image, full_edit_prompt, review_prompt_template, review_prompt_kwargs, edit_model, review_model, drive, safe_title, time_part):
    """
    Run edit, review, and optional regenerate loops for one instructional slot.

    :param scene_id: Scene id string from the edit plan.
    :param image_index: 1-based slot image index within the scene.
    :param asset_url: Original Drive or web URL for this slot.
    :param reference_image: PIL image loaded from asset_url.
    :param full_edit_prompt: Formatted image_edit_execution_prompt for this slot.
    :param review_prompt_template: image_edit_review_prompt template string with edited_asset_url placeholder.
    :param review_prompt_kwargs: Dict of format kwargs for the review template excluding edited_asset_url.
    :param edit_model: Image edit model id (SCENE_IMAGE_EDIT_MODEL).
    :param review_model: Review model id (SCENE_IMAGE_EDIT_REVIEW_MODEL).
    :param drive: Google Drive client for uploads.
    :param safe_title: Sanitized slide title fragment for filenames.
    :param time_part: Time string fragment for filenames.
    :return: Tuple of tracking_lines and review_lines from _format_slot_outputs.
    """
    loops = []

    # print("=" * 80)
    # print("Image edit execution — image_edit_execution_prompt (formatted)")
    # print(f"Scene ID {scene_id} | Image {image_index} | Initial editor turn (loop 1)")
    # print("=" * 80)
    # print(full_edit_prompt)
    # print("=" * 80)
    # print()

    rec = {
        "loop_num": 1,
        "edit_url": None,
        "edit_error": None,
        "verdict": None,
        "failures": "",
        "review_error": None,
        "raw_review": "",
    }

    try:
        edited_pil, edit_state = _edit_chat_first_turn(
            reference_image, full_edit_prompt, model=edit_model
        )
    except Exception as e:
        traceback.print_exc()
        rec["edit_error"] = f"{e}"
        loops.append(rec)
        return _format_slot_outputs(loops, asset_url, image_index)

    fname = (
        f"{safe_title}_{time_part}_{uuid.uuid4().hex[:8]}"
        f"_S{scene_id}_I{image_index}_L{rec['loop_num']}.png"
    )
    edited_url = upload_image_to_drive(
        edited_pil, fname, drive, folder_id=EDITED_IMAGE_DRIVE_FOLDER_ID
    )
    if not edited_url:
        rec["edit_error"] = "Edited image upload to Drive failed"
        loops.append(rec)
        return _format_slot_outputs(loops, asset_url, image_index)
    rec["edit_url"] = edited_url

    print("=" * 80)
    print("Image edit execution — editor model output (after image_edit_execution_prompt)")
    print(f"Scene ID {scene_id} | Image {image_index} | Loop 1")
    print("=" * 80)
    print(f"Edited image URL: {edited_url}")
    print("=" * 80)
    print()

    review_history = None
    full_review_prompt = review_prompt_template.format(
        **review_prompt_kwargs,
        edited_asset_url=edited_url,
    )
    # print("=" * 80)
    # print("Image edit execution — image_edit_review_prompt (formatted)")
    # print(f"Scene ID {scene_id} | Image {image_index} | Initial reviewer turn (loop 1)")
    # print("=" * 80)
    # print(full_review_prompt)
    # print("=" * 80)
    # print()
    try:
        verdict, failures, raw, review_history = _review_chat_first_turn(
            reference_image,
            edited_pil,
            full_review_prompt,
            model=review_model,
        )
        rec["verdict"] = verdict
        rec["failures"] = failures
        rec["raw_review"] = raw
        print(f"\n{'─'*80}")
        print(
            f"📤 Scene edit execution — reviewer LLM response "
            f"(Scene {scene_id}, Image {image_index}, loop 1; "
        )
        print(f"{'─'*80}")
        output_match = re.search(
            r"<output\b[^>]*>(.*?)</output>",
            (raw or ""),
            re.DOTALL | re.IGNORECASE,
        )
        if output_match:
            inner = output_match.group(1).strip()
            print(inner if inner else "(empty inside <output>)")
        else:
            print("⚠️  No <output>...</output> in response; full text below.")
            print((raw or "").strip() or "(empty reviewer response)")
        print(f"{'─'*80}")
        print(f"Parsed verdict: {verdict or '(none)'}")
        print()
    except Exception as e:
        traceback.print_exc()
        rec["review_error"] = f"{e}"
        loops.append(rec)
        return _format_slot_outputs(loops, asset_url, image_index)

    loops.append(rec)

    for _cycle in range(1, MAX_EDIT_REVIEW_LOOPS + 1):
        last = loops[-1]
        if (last.get("verdict") or "").upper() == "PASS":
            break
        if last.get("edit_error") or last.get("review_error"):
            break

        loop_num = last["loop_num"] + 1
        rec = {
            "loop_num": loop_num,
            "edit_url": None,
            "edit_error": None,
            "verdict": None,
            "failures": "",
            "review_error": None,
            "raw_review": "",
        }

        follow_up_edit_text = edit_revision_followup_template.format(
            failures_xml=_failures_for_followup(
                last.get("failures") or "",
                last.get("raw_review") or "",
            )
        )
        # print("=" * 80)
        # print("Image edit execution — edit_revision_followup_template (formatted)")
        # print(
        #     f"Scene ID {scene_id} | Image {image_index} | "
        #     f"Editor follow-up before loop {loop_num} edit (after loop {last['loop_num']} FAIL)"
        # )
        # print("=" * 80)
        # print(follow_up_edit_text)
        # print("=" * 80)
        # print()
        try:
            edited_pil, edit_state = _edit_chat_followup_turn(
                reference_image,
                edit_state,
                follow_up_edit_text,
                model=edit_model,
                original_image_url=asset_url,
                last_edited_image=edited_pil,
                last_edited_image_url=last.get("edit_url") or "",
                follow_up_edit_loop_num=loop_num,
            )
        except Exception as e:
            traceback.print_exc()
            rec["edit_error"] = f"{e}"
            loops.append(rec)
            break

        fname = (
            f"{safe_title}_{time_part}_{uuid.uuid4().hex[:8]}"
            f"_S{scene_id}_I{image_index}_L{loop_num}.png"
        )
        edited_url = upload_image_to_drive(
            edited_pil, fname, drive, folder_id=EDITED_IMAGE_DRIVE_FOLDER_ID
        )
        if not edited_url:
            rec["edit_error"] = "Edited image upload to Drive failed"
            loops.append(rec)
            break
        rec["edit_url"] = edited_url

        print("=" * 80)
        print(
            "Image edit execution — editor model output "
            "(after edit_revision_followup_template)"
        )
        print(f"Scene ID {scene_id} | Image {image_index} | Loop {loop_num}")
        print("=" * 80)
        print(f"Edited image URL: {edited_url}")
        print("=" * 80)
        print()

        # print("=" * 80)
        # print("Image edit execution — review_revision_followup_template")
        # print(
        #     f"Scene ID {scene_id} | Image {image_index} | "
        #     f"Reviewer follow-up re-review after loop {loop_num} edit"
        # )
        # print("=" * 80)
        # print(review_revision_followup_template)
        # print("=" * 80)
        # print()
        try:
            verdict, failures, raw, review_history = _review_chat_followup_turn(
                edited_pil,
                review_history,
                review_revision_followup_template,
                model=review_model,
            )
            rec["verdict"] = verdict
            rec["failures"] = failures
            rec["raw_review"] = raw
            print(f"\n{'─'*80}")
            print(
                f"📤 Scene edit execution — reviewer LLM response "
                f"(Scene {scene_id}, Image {image_index}, loop {loop_num}; "
                "content inside <output>...</output> when present)"
            )
            print(f"{'─'*80}")
            output_match = re.search(
                r"<output\b[^>]*>(.*?)</output>",
                (raw or ""),
                re.DOTALL | re.IGNORECASE,
            )
            if output_match:
                inner = output_match.group(1).strip()
                print(inner if inner else "(empty inside <output>)")
            else:
                print("⚠️  No <output>...</output> in response; full text below.")
                print((raw or "").strip() or "(empty reviewer response)")
            print(f"{'─'*80}")
            print(f"Parsed verdict: {verdict or '(none)'}")
            print()
        except Exception as e:
            traceback.print_exc()
            rec["review_error"] = f"{e}"
            loops.append(rec)
            break

        loops.append(rec)

    return _format_slot_outputs(loops, asset_url, image_index)


@traceable(
    metadata={
        "agent_name": "graphics_definition_v2",
        "step_name": "Scene Edit Execution",
        "function_name": "process_image_editing_execution_row",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous"),
    }
)
def process_image_editing_execution_row(index, row, course_name, target_audience, drive):
    """
    Run the edit-review loop per instructional slot for one sheet row.

    Builds image_editing_tracking and edit_review cell text for that row.

    :param index: DataFrame row index.
    :param row: DataFrame row object.
    :param course_name: Course name from Course info.
    :param target_audience: Target audience from Course info.
    :param drive: Drive client for loading originals and uploading edits.
    :return: Tuple of index, image_editing_tracking cell string, and edit_review cell string.
    """
    try:
        plan_text = str(row.get("scene_edit_plan", "")).strip()
        if not plan_text or plan_text == "nan" or plan_text.startswith("ERROR:"):
            return index, "", ""

        manifest_text = str(row.get("slideshow_manifest", "")).strip()
        narration_map = _narration_by_scene_id(manifest_text)

        fgd_text = str(row.get("final_graphics_definition", "")).strip()
        slot_vo_resolve = None
        if fgd_text and fgd_text != "nan":
            slot_vo_resolve = make_slot_narration_resolver_from_fgd(fgd_text)

        topic_name = str(row.get("Topic", "")).strip()
        subtopic_name = str(row.get("Subtopic", "")).strip()
        slide_title = str(row.get("Slide Chunk Title", "")).strip()
        slide_content = str(row.get("Slide Chunk", "")).strip()
        slide_type = str(row.get("Slide Type", "")).strip()
        if slide_type == "nan":
            slide_type = ""

        safe_title = _sanitize_filename_component(slide_title, fallback="slide")
        time_part = datetime.now().strftime("%H%M%S")

        scene_blocks = split_scene_edit_plan_by_scene(plan_text)
        if not scene_blocks:
            err = "ERROR: No ---Scene ID--- headers found in scene_edit_plan"
            return index, err, err

        edit_model = SCENE_IMAGE_EDIT_MODEL.strip()
        review_model = _resolve_scene_image_edit_review_model(
            SCENE_IMAGE_EDIT_REVIEW_MODEL
        )

        tracking_sections = []
        review_sections = []

        for scene_id, scene_body in scene_blocks:
            header = f"---Scene ID: {scene_id}---"
            scene_tracking = [header]
            scene_review = [header]
            slots = extract_slot_edit_blocks(scene_body)

            for k, slot_xml in enumerate(slots, start=1):
                asset_url = _parse_multiline_field(slot_xml, "Asset URL")

                if not slot_needs_instructional_image_edit(slot_xml):
                    scene_tracking.append(f"Image {k}:")
                    scene_tracking.append("No Edits made")
                    scene_tracking.append("")
                    scene_review.append(f"Image {k}:")
                    scene_review.append("No Edits made")
                    scene_review.append("")
                    continue

                edit_payload = build_edit_instructions_payload(slot_xml)
                slot_narration = ""
                if slot_vo_resolve:
                    slot_narration = slot_vo_resolve(asset_url) or ""
                narration_for_prompt = (
                    slot_narration or narration_map.get(scene_id, "")
                )
                full_edit_prompt = image_edit_execution_prompt.format(
                    course_name=course_name or "",
                    target_audience=target_audience or "",
                    topic_name=topic_name or "",
                    subtopic_name=subtopic_name or "",
                    slide_type=slide_type or "",
                    slide_title=slide_title or "",
                    slide_content=slide_content or "",
                    narration_span=narration_for_prompt,
                    original_asset_url=asset_url or "",
                    edit_instructions=edit_payload,
                    styling_guide=styling_guide,
                )

                ref = load_image_from_url(
                    asset_url,
                    drive,
                    title=f"edit_exec_row{index}_s{scene_id}_img{k}",
                )
                if ref is None:
                    err_line = f"ERROR: Could not load image from URL: {asset_url}"
                    scene_tracking.append(f"Image {k}:")
                    scene_tracking.append(err_line)
                    scene_tracking.append("")
                    scene_review.append(f"Image {k}:")
                    scene_review.append(err_line)
                    scene_review.append("")
                    continue

                review_prompt_kwargs = {
                    "course_name": course_name or "",
                    "target_audience": target_audience or "",
                    "topic_name": topic_name or "",
                    "subtopic_name": subtopic_name or "",
                    "slide_type": slide_type or "",
                    "slide_title": slide_title or "",
                    "slide_content": slide_content or "",
                    "narration_span": narration_for_prompt,
                    "original_asset_url": asset_url or "",
                    "edit_instructions": edit_payload,
                    "styling_guide": styling_guide,
                }

                tracking_lines, review_lines = _run_edit_review_loop_for_slot(
                    scene_id,
                    k,
                    asset_url or "",
                    ref,
                    full_edit_prompt,
                    image_edit_review_prompt,
                    review_prompt_kwargs,
                    edit_model,
                    review_model,
                    drive,
                    safe_title,
                    time_part,
                )

                scene_tracking.append(f"Image {k}:")
                scene_tracking.extend(tracking_lines)
                scene_tracking.append("")

                scene_review.extend(review_lines)
                scene_review.append("")

            tracking_sections.append("\n".join(scene_tracking).rstrip())
            review_sections.append("\n".join(scene_review).rstrip())

        return (
            index,
            "\n\n".join(tracking_sections).strip(),
            "\n\n".join(review_sections).strip(),
        )
    except Exception as e:
        print(f"Error image edit execution row {index}: {e}")
        traceback.print_exc()
        err = f"ERROR: {str(e)}"
        return index, err, err


@traceable(
    metadata={
        "agent_name": "graphics_definition_v2",
        "step_name": "Scene Edit Execution",
        "function_name": "run_image_editing_execution_for_all_rows",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous"),
    }
)
def run_image_editing_execution_for_all_rows(sheet, max_workers=50):
    """
    Populate image_editing_tracking and edit_review for Slide Chunks rows.

    :param sheet: gspread spreadsheet object.
    :param max_workers: Row-level thread pool size (default 50).
    :return: None
    """
    worksheet_name = "Slide Chunks"

    _, course_info_df = get_sheet_data_and_df(sheet, "Course info")
    course_name = (
        str(course_info_df.loc[0, "Course Name"]).strip()
        if not course_info_df.empty and "Course Name" in course_info_df.columns
        else ""
    )
    target_audience = (
        str(course_info_df.loc[0, "Target Audience & Industry"]).strip()
        if not course_info_df.empty
        else ""
    )

    ws, df = get_sheet_data_and_df(sheet, worksheet_name)
    if "image_editing_tracking" not in df.columns:
        df["image_editing_tracking"] = ""
    if "edit_review" not in df.columns:
        df["edit_review"] = ""

    model_id = SCENE_IMAGE_EDIT_MODEL.strip()
    provider = _scene_image_edit_provider(model_id)
    review_model_id = _resolve_scene_image_edit_review_model(
        SCENE_IMAGE_EDIT_REVIEW_MODEL
    )
    print(
        f"Image edit execution: edit model={model_id!r} (backend={provider}), "
        f"review model={review_model_id!r}, "
        f"max review-regenerate loops={MAX_EDIT_REVIEW_LOOPS}"
    )

    drive = get_drive_instance()
    if not drive:
        print("Image edit execution: Drive not available; loads/uploads may fail.")

    def _filled(val):
        """
        Return True if val is non-empty sheet content and not an error placeholder.

        :param val: Cell value from the DataFrame.
        :return: True if the value should count as already filled.
        """
        v = str(val or "").strip()
        return bool(v) and v != "nan" and not v.startswith("ERROR:")

    rows_to_process = []
    for index, row in df.iterrows():
        plan = str(row.get("scene_edit_plan", "")).strip()
        if not plan or plan == "nan" or plan.startswith("ERROR:"):
            continue
        if _filled(row.get("image_editing_tracking", "")) and _filled(
            row.get("edit_review", "")
        ):
            continue
        rows_to_process.append((index, row))

    if not rows_to_process:
        print(
            f"Image edit execution: no rows to process "
            f"(edit model={model_id!r}, backend={provider}, "
            f"review model={review_model_id!r})."
        )
        return

    print(
        f"Image edit execution: processing {len(rows_to_process)} row(s), "
        f"max_workers={max_workers}, folder={EDITED_IMAGE_DRIVE_FOLDER_ID}"
    )

    futures_map = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for index, row in rows_to_process:
            fut = executor.submit(
                process_image_editing_execution_row,
                index,
                row,
                course_name,
                target_audience,
                drive,
            )
            futures_map[fut] = index

        progress = SmartProgressBar(
            total_tasks=len(futures_map),
            description="Image edit execution",
            save_interval=5,
        )

        for future in as_completed(futures_map):
            index = futures_map[future]
            try:
                row_index, tracking, review = future.result()
                df.at[row_index, "image_editing_tracking"] = tracking
                df.at[row_index, "edit_review"] = review
                progress.update()
                if progress.should_save():
                    save_to_sheet(ws, df)
            except Exception as e:
                print(f"Image edit execution future error row {index}: {e}")
                err = f"ERROR: {str(e)}"
                df.at[index, "image_editing_tracking"] = err
                df.at[index, "edit_review"] = err
                progress.update()

    save_to_sheet(ws, df)
    format_worksheet(ws)
    print(
        f"Image edit execution: complete "
        f"(edit model={model_id!r}, review model={review_model_id!r})."
    )


def delete_image_editing_execution_columns(sheet):
    """
    Remove image_editing_tracking and edit_review from the Slide Chunks worksheet.

    :param sheet: gspread sheet object.
    :return: None
    """
    worksheet_name = "Slide Chunks"
    ws, df = get_sheet_data_and_df(sheet, worksheet_name)
    cols = ["image_editing_tracking", "edit_review"]
    existing = [c for c in cols if c in df.columns]
    if not existing:
        print(
            f"No column(s) {cols} to delete on '{worksheet_name}'."
        )
        return
    df = df.drop(columns=existing)
    clear_worksheet(ws)
    save_to_sheet(ws, df)
    for c in existing:
        print(f"Deleted column '{c}' from '{worksheet_name}' worksheet")
