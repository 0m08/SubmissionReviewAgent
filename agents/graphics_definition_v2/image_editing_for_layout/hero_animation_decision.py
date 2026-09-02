import re
import traceback
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st
from dotenv import load_dotenv
from google.genai import types
from langsmith import traceable

from agents.graphics_definition_v2.aggregation_agent.aggregation_agent import (
    get_drive_instance,
    invoke_gemini_multimodal,
)
from agents.graphics_definition_v2.image_editing_for_layout.image_edit_planning import (
    parse_scenes_from_slideshow_manifest,
)
from agents.graphics_definition_v2.review_agent.review_and_revise import (
    build_visual_part_only,
    is_drive_video_url,
    is_youtube_url,
)
from services.sheets_service import (
    clear_worksheet,
    format_worksheet,
    get_sheet_data_and_df,
    save_to_sheet,
)

load_dotenv()

hero_animation_decision_prompt = """You are a senior instructional designer specializing in the field of HVAC e-learning content.

## Context

Our courses play as narrated slideshows. Each slide is split into one or more **scenes** — short narration spans, each with its own on-screen layout showing the assigned visuals.
You are planning for a "single_visual_hero" scene: one image fills the slide while its voiceover line plays. There are no side-by-side panels or grids in this layout — just one hero visual and the narration span plays.
Your job is to decide whether the "video player" should add a short instructional overlay animation on top of that hero image — to help the learner see what the narration span is talking about (name a part, highlight some specific spot(s), show some symbol(s), show a callout card, or isolate one subject with an emphasis effect). These overlays will be drawn as animations at playback time; they will not be baked into the image file.
For most hero scenes, the image alone is enough — in that case you can choose **none**. Only pick an overlay when it clearly helps the learner connect words to what is on screen.
So your task is to decide, for ONE single-hero scene, whether the learner needs an instructional overlay animation on top of the hero image — and if so, which type to use.
You are deciding among single-hero overlay options only. Do not invent split, inset, multi-panel, or other layout animations.

## Inputs

These are the inputs:

<course_information>
Course name: {course_name}
Target audience: {target_audience}
Topic name: {topic_name}
Subtopic name: {subtopic_name}
</course_information>

<slide_information>
Slide Type: {slide_type}
Slide title: {slide_title}
Slide content: {slide_content}
</slide_information>

<single_hero_scene>
Narration span of this scene: {narration_span}
Visual URL: {primary_visual_url}
</single_hero_scene>

You will also receive a multimodal preview of this assigned asset for this single hero scene.

## Allowed animation types (choose exactly ONE)

Use only these six values for <animation_type>. No aliases. No invented types.

1) none

No instructional overlay on the hero image.

Choose none when:
- The image already communicates the narration point without annotation.
- The narration span is general or overview-level and does not call out one specific part, region, symbol, or message that needs an overlay.
- Adding a label, highlight, icon, callout card, or emphasis effect would clutter the visual or state the obvious.
- The slide is a title or transition moment, or a broad conceptual introduction.
- The target object is not clearly visible in the image.

This is the DEFAULT. Most single-hero scenes can be none.

2) text_label

The hero image shifts to make room (typically slides left), and a short text label or label card animates in on the remaining space.

Choose text_label when:
- The narration span names or identifies a specific visible object, component, part, condition, or region.
- The trainee needs the technical term tied to what they are looking at (e.g. "expansion valve", "condenser coil", "high-side port").
- A highlight would be insufficient because the trainee also needs the name.

Do NOT choose text_label when:
- The image already contains a clear, readable label for the same thing.
- Multiple labels would be needed in one scene (this animation supports one primary label per scene).
- A symbolic icon would communicate the idea better than text (use icon_overlay).
- The narration needs a short structured header + body callout (not just a part name) — use callout_card.
- Only pointing to a location is needed without naming it (use bbox_highlight).
- The narration needs the whole background to recede so one subject pops visually (use emphasis_style).

3) bbox_highlight

One or more rectangle outlines animate onto the hero image to emphasize specific visible regions.

You may output one highlight, or several, when the narration span clearly points to more than one distinct spot in the same still (e.g. two valves, a leak and the nearby fitting, supply vs return). Prefer as few highlights as will make the point. 

Choose bbox_highlight when:
- The narration span directs attention to one or more locations, details, defects, connection points, sub-regions, etc.
- Emphasis is enough — no new label text is required.
- Each target is a reasonably local area that is clearly visible and can be described separately.

Do NOT choose bbox_highlight when:
- The narration span requires naming the part (use text_label).
- The whole image is the subject and no sub-region matters (use none).
- A target is not clearly visible or cannot be described precisely enough for a spatial sub-agent to locate it later.
- The main point is symbolic meaning such as warning or airflow (use icon_overlay).
- The narration needs a structured header + body callout card (use callout_card).
- The learner needs one primary subject to stand out from a busy scene while the rest of the image dims or blurs (use emphasis_style).

4) icon_overlay

One or more simple instructional icons animate onto the hero image near the relevant objects or regions.

You may output one icon, or several, when the narration span clearly needs one or more than one symbol in the same still (e.g. danger symbol, heat vs cold, correct vs incorrect, airflow at two vents, etc). Prefer as few icons as will make the point. Typical count is 1. Use 2–3 only when the assigned narration span contrasts or lists those meanings. Do not exceed 3.

Choose icon_overlay when:
- The narration span implies a symbol more than a name: warning, safety, heat, cold, airflow, water or moisture, electricity, correct vs incorrect, check vs cross, etc.
- An icon clarifies meaning without adding readable text clutter.
- The related object or area is visible and the symbolic cue helps retention.

Do NOT choose icon_overlay when:
- The learner needs the exact technical term (use text_label).
- The narration span points to a precise sub-region without symbolic meaning (use bbox_highlight).
- The icon would be decorative or introduce meaning not supported by the narration span.
- The learner needs a readable header plus a short body line on a card (use callout_card), not a bare symbol.
- The narration needs one subject isolated from a busy background (use emphasis_style).

5) callout_card

One or more structured callout cards animate onto the hero image. Each card has an optional icon, a short header, and a short body line. Header and body are free-form — whatever the narration needs — not limited to a fixed set of words. Body text must be at most 5 words (strict; count every word).

You may output one callout, or several, when the narration span clearly needs more than one distinct header+body message in the same still. Prefer as few cards as will make the point. Typical count is 1. Use 2–3 only when the assigned narration span contrasts or lists those messages. Do not exceed 3. Give each card its own position so they do not stack on top of each other or cover the main subject.

Choose callout_card when:
- The narration span benefits from short structured card(s) (header + body) overlaid on the hero image, rather than only naming a part, only pointing at a region, or only showing a symbol.
- A header + body pair communicates the point better than text_label or icon_overlay alone.

Do NOT choose callout_card when:
- Only the technical name of a visible part is needed (use text_label).
- Only pointing at a region is needed (use bbox_highlight).
- Only a symbolic cue without readable header/body is needed (use icon_overlay).
- The narration needs one subject to pop from a busy scene without readable card text (use emphasis_style).
- The image already states the same message clearly, or the narration is general overview (use none).

6) emphasis_style

The hero image keeps one primary visible subject sharp and in full color while the rest of the scene is de-emphasized (darkened, desaturated, and/or blurred). A segmentation sub-agent will later isolate the subject using its real outline — not a rectangle box.

Choose emphasis_style when:
- The narration span focuses attention on ONE primary object, tool, piece of equipment, body part, action, etc. in an otherwise busy or distracting scene.
- Reducing visual noise around that subject would help the learner more than adding text, icons, callout cards, etc.
- The subject has a reasonably clear boundary that could be segmented (e.g. hacksaw and hands, gloves, a valve, a meter, etc.).
- The teaching goal is "look at this thing while everything else fades back" rather than "name this part" or "point at these spots."

Do NOT choose emphasis_style when:
- The narration requires readable label text (use text_label).
- One or multiple separate regions must be pointed out in the same scene using highlight boxes and they will serve a better purpose than emphasizing the subject (use bbox_highlight).
- A symbolic cue is enough without isolating the subject (use icon_overlay).
- A structured header + body card is needed (use callout_card).
- The whole image is already the focus, or the scene is not visually busy (use none).
- The subject is tiny, heavily occluded, or cannot be described clearly enough for segmentation.
- More than one unrelated subject must be emphasized at once in the same scene (pick the single most important one, or use bbox_highlight / none instead).

## Decision rules

1. Scope
   - You are planning for a scene whose layout template is single_visual_hero, meaning one image fills the slide while its narattion span plays.
   - If Slide Type is Transition, strictly output animation_type = none and explain that animations for transition slides are not planned here.

2. How to decide
   - Read the narration span first: what must the learner notice while this text is spoken? You can use the whole slide content as surrounding context, but remember you are only planning for the assigned narration span of this single hero scene and the assigned visual. The other parts of the slide content have been assigned their own layouts and visuals, so you don't need to worry about it apart from understanding the surrounding context, your main focus should be on the assigned single hero layout visual given in the input.
   - Inspect the assigned visual carefully. Determine whether it is already clear and easy to understand or whether additional animations are needed.
   - Choose the simplest option that improves understanding.
   - Pick exactly one animation type.

3. Disambiguation cheat sheet
   - Need to NAME something visible → text_label
   - Need to POINT to one or more regions without naming → bbox_highlight
   - Need one or more SYMBOLIC cues only (warning, airflow, temperature, etc.) → icon_overlay
   - Need a structured card with header + short body (optional icon) → callout_card
   - Need ONE subject to pop from a busy scene while the background recedes → emphasis_style
   - Image already sufficient → none

4. Callout card placement
   - When animation_type is callout_card, every card MUST have a rough on-screen <position> so it does not obstruct the main subject of the image.
   - One callout only: you may use any of the nine positions: top_left, top_center, top_right, center_left, center, center_right, bottom_left, bottom_center, bottom_right.
   - Two or more callouts in the same scene: use corner positions ONLY — top_left, top_right, bottom_left, bottom_right. Never use top_center, bottom_center, center, center_left, or center_right when multiple cards appear together.
   - Assign each card a different corner. Recommended patterns:
     - 2 cards: opposite corners (e.g. top_left + bottom_right, or top_right + bottom_left).
     - 3 cards: three distinct corners (e.g. top_left, top_right, bottom_left).
     - 4 cards: all four corners.
   - Never place two cards on the same row of the 9-grid (e.g. top_left + top_center).
   - Prefer empty or less-important sides of the visual. Avoid covering the main subject. Strictly ensure callouts do not obstruct the narration focus.

5. Emphasis style target
   - When animation_type is emphasis_style, pick exactly ONE compact, solid, cleanly-outlined object the narration is most about (e.g. a motor, a valve, a pipe, a gauge, a tool, a hand, a single person, one appliance, etc.).
   - The target MUST be a single segmentable object. Do NOT pick: a region or area ("the attic", "the room", "the left side"); diffuse/translucent phenomena ("air flow", "heat", "smoke", "steam"); a union of parts ("... and ... and ...", "all the X"); abstract/negative space ("the gaps", "the space between"); text, labels, arrows, or chart annotations; the whole scene.
   - If the narration is about a region or diffuse effect, still pick the single most relevant concrete object inside that region — never describe the region itself.
   - Keep <target_description> to a SINGLE short sentence: the object + a brief location + its color/shape. No compound "and" lists, no "including ...", no exclusions, no enumerations of parts.

## Output format

Provide your output strictly in this format:

<output>

<evaluation_breakdown>
Use this section as a structured reasoning and scratchpad space for you before producing the final output.

- Slide Understanding: Briefly state in your own words what the slide is about.
- Scene Understanding: Briefly explain what this narration span is communicating and what the trainee needs to notice on screen.
- Hero Asset Scan: Briefly describe what is actually visible in the hero image. Do not assume unseen details.
- Overlay Analysis: Decide and say whether you think an overlay is needed. If yes, name the visible target(s) and which type fits (none, text_label, bbox_highlight, icon_overlay, callout_card, emphasis_style) and why the others do not. If callout_card, also note which rough position(s) keeps the subject clear. If emphasis_style, name the one subject that should remain in full color.
- Additional Analysis: Any extra observations needed to justify the final choice.

It is acceptable for this section to be quite long and detailed if needed for correctness.
</evaluation_breakdown>

(Based on your above evaluation, provide the final output below in this exact format.)

<hero_animation_plan>

<animation_type>
none | text_label | bbox_highlight | icon_overlay | callout_card | emphasis_style
</animation_type>

<reason>
Why this selected animation type best supports leaner understanding, or why none is sufficient.
</reason>

<animation_details>
(Provide the following details for the selected animation type.)

<label_text>

Required only when animation_type is text_label. Put N/A in this field for other animation types. 
Short, direct, instructional label (typically 1–4 words). Pulled from or faithful to the narration span. Do not provide any other information here apart for the exact label text to use

</label_text>

<highlight>

Required when animation_type is bbox_highlight. Put N/A in this field for other animation types. Repeat one <highlight>...</highlight> tag per distinct region (1–3 times, never more than 3).

<target_description>
Describe this one visible target using position in the frame, nearby landmarks, color, shape, size, and distinguishing details. Write 2–4 sentences so a spatial sub-agent can locate this target without other context. Do not combine two regions in one <highlight>.
</target_description>

<highlight_shape>
box
</highlight_shape>

</highlight>

(Repeat the above <highlight>...</highlight> tag for each distinct region that needs to be highlighted.)

<icon>

Required when animation_type is icon_overlay. Put N/A in this field for other animation types. Repeat one <icon>...</icon> tag per distinct icon (1–3 times, never more than 3). 

<icon_concept>
Plain-language icon meaning, e.g. "warning triangle", "airflow arrows", "water droplet", "check mark for correct installation".
</icon_concept>

<target_description>
Describe the exact visible object or region this icon should relate to, using position in the frame, nearby landmarks, and distinguishing details. Do not combine two icons in one <icon>.
</target_description>

<placement_hint>
Where this icon should appear relative to the target, e.g. "above the condenser fan grille".
</placement_hint>

</icon>

(Repeat the above <icon>...</icon> tag for each distinct icon that needs to be animated.)

<callout>

Required when animation_type is callout_card. Put N/A in this field for other animation types. Repeat one <callout>...</callout> tag per distinct card (1–3 times, never more than 3).

<header>
Short header text for this card. Content is free-form — whatever fits the narration (a category, topic, condition, action, or other label). Keep it short. Do not put the full body sentence here.
</header>

<body>
Short instructional body line for this card. Strict limit: at most 5 words total (≤ 5).
</body>

<icon_concept>
Optional. Provide a description of the icon that needs to be animated. Ensure that your descrition is very clear since it will be used to generate the icon image. Put N/A if no icon is needed.
</icon_concept>

<position>
Exactly one position per card.
- If this scene has only ONE callout: any of top_left | top_center | top_right | center_left | center | center_right | bottom_left | bottom_center | bottom_right.
- If this scene has TWO OR MORE callouts: corner positions ONLY — top_left | top_right | bottom_left | bottom_right. Never top_center, bottom_center, center, center_left, or center_right. Each card must use a different corner.
</position>

</callout>

(Repeat the above <callout>...</callout> tag for each distinct callout card that needs to be animated.)

<emphasis>

Required when animation_type is emphasis_style. Put N/A in this field for other animation types. Output exactly one <emphasis>...</emphasis> block.

<target_description>
ONE compact, solid, cleanly-outlined object to segment. A SINGLE short sentence: object + brief location + color/shape.
</target_description>

<emphasis_mode>
How the non-subject background should be treated. Use exactly one of these values:
- darken — default for busy scenes; background dims/blurs while the subject stays in full color (Emphasis Style).
- sepia — background converts to sepia tone while the subject stays in full color (Selective Color Highlight).
Choose darken unless the narration clearly benefits from a warm sepia treatment.
</emphasis_mode>

</emphasis>

</animation_details>

</hero_animation_plan>

</output>

(Strictly follow the output format above. Do not add commentary, prose, or fields outside this schema.)
"""

_HERO_TEMPLATE = "single_visual_hero"
_DIRECT_VIDEO_FILE_SUFFIXES = (".mp4", ".webm", ".mov")
_PLAN_COLUMN = "hero_animation_plan"
_COORDS_COLUMN = "hero_bbox_coordinates"
_ICONS_COLUMN = "hero_icon_overlays"
_CALLOUTS_COLUMN = "hero_callout_overlays"
_EMPHASIS_COLUMN = "hero_emphasis_overlays"

# Retries for hero decision multimodal calls (503 deadline, rate limits, etc.).
_HERO_DECISION_MAX_RETRIES = 5

_VALID_PLAN_ANIMATION_TYPES = frozenset({
    "none",
    "text_label",
    "bbox_highlight",
    "icon_overlay",
    "callout_card",
    "emphasis_style",
})


def plan_cell_needs_processing(existing_text):
    """
    Return True when hero_animation_plan should be (re)generated.

    Retries cells that contain ERROR anywhere (e.g. ``---Scene ID: 1---\\nERROR: ...``).
    """
    existing = str(existing_text or "").strip()
    if not existing or existing.lower() == "nan" or existing == "-":
        return True
    return "ERROR:" in existing


def _plan_block_is_ok(block_text):
    """True when an existing plan scene block parsed successfully."""
    text = str(block_text or "").strip()
    if not text or "ERROR:" in text:
        return False
    match = re.search(
        r"<animation_type>\s*(.*?)\s*</animation_type>",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if not match:
        return False
    atype = match.group(1).strip().lower()
    return atype in _VALID_PLAN_ANIMATION_TYPES


def _normalize_scene_template(template):
    """
    Normalize a slideshow_manifest template label for comparison.

    :param template: Raw template string from a scene.
    :return: Normalized template token.
    """
    return str(template or "").strip().lower().replace("-", "_").replace(" ", "_")


def _is_video_asset_url(url):
    """
    Detect whether a URL is a YouTube, Drive, or direct video file.

    :param url: Candidate asset URL.
    :return: True when this hero visual should skip the decision prompt.
    """
    if not url:
        return False
    if is_youtube_url(url) or is_drive_video_url(url):
        return True
    u = str(url).lower()
    return any(suffix in u for suffix in _DIRECT_VIDEO_FILE_SUFFIXES)


def _is_single_visual_hero_scene(scene):
    """
    Return whether a parsed scene uses the single_visual_hero template.

    :param scene: Parsed scene dict from slideshow_manifest.
    :return: True when the scene template is single_visual_hero.
    """
    return _normalize_scene_template(scene.get("template")) == _HERO_TEMPLATE


def _hero_primary_asset_url(scene):
    """
    Return the primary visual URL for a single-hero scene.

    :param scene: Parsed scene dict with slots.
    :return: Asset URL string, or empty string if missing.
    """
    slots = scene.get("slots") or []
    for slot in slots:
        if str(slot.get("role", "")).strip() == "primary_visual":
            return str(slot.get("asset", "")).strip()
    if slots:
        return str(slots[0].get("asset", "")).strip()
    return ""


def _scene_block(scene_id, body):
    """
    Wrap one scene's output with the standard scene header used on the sheet.

    :param scene_id: Scene id from slideshow_manifest.
    :param body: Plan text for this scene.
    :return: Headered text block.
    """
    return f"---Scene ID: {scene_id}---\n{body}".strip()


def parse_hero_animation_decision_response(response_text):
    """
    Extract inner hero_animation_plan from the model response.

    :param response_text: Raw LLM response text.
    :return: Inner hero_animation_plan text, or the full response if the tag is missing.
    """
    if not response_text:
        return ""
    plan_match = re.search(
        r"<hero_animation_plan>(.*?)</hero_animation_plan>",
        response_text,
        re.DOTALL | re.IGNORECASE,
    )
    if not plan_match:
        return response_text.strip()
    return plan_match.group(1).strip()


def _append_hero_visual_multimodal_parts(parts, narration_span, asset_url, drive):
    """
    Append the preview label, then the still image, for one hero visual.

    :param parts: Mutable list of Gemini Part objects.
    :param narration_span: Narration span for this hero scene.
    :param asset_url: Assigned hero image URL.
    :param drive: Drive client for loading Drive-hosted images.
    :return: True when the image part was attached; False when load failed.
    """
    label = (
        f"Narration Span: {narration_span or '(empty)'}\n"
        f"Assigned single-hero visual asset URL: {asset_url}\n"
    )
    parts.append(types.Part(text=label))
    visual_part = build_visual_part_only(asset_url, drive)
    if not visual_part:
        parts.append(types.Part(text="[Assigned single-hero visual could not be loaded]"))
        return False
    parts.append(visual_part)
    return True


@traceable(
    metadata={
        "agent_name": "graphics_definition_v2",
        "step_name": "Hero Overlay Animation Decision",
        "function_name": "generate_hero_animation_decision_for_scene",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous"),
    }
)
def generate_hero_animation_decision_for_scene(course_name, target_audience, topic_name, subtopic_name, slide_type, slide_title, slide_content, scene, drive, llm="gemini_3_flash_thinking"):
    """
    Run the hero overlay animation decision prompt for one single-hero still.

    :param course_name: Course name from Course info.
    :param target_audience: Target audience from Course info.
    :param topic_name: Topic name.
    :param subtopic_name: Subtopic name.
    :param slide_type: Slide Type value.
    :param slide_title: Slide Chunk Title.
    :param slide_content: Full Slide Chunk text.
    :param scene: Parsed scene dict (id, template, narration, slots).
    :param drive: Drive client for loading the hero image.
    :param llm: LLM identifier for the multimodal call.
    :return: Tuple of (scene_id, plan_inner_text).
    """
    scene_id = str(scene.get("id", "")).strip() or "1"
    narration_span = str(scene.get("narration", "")).strip()
    primary_visual_url = _hero_primary_asset_url(scene)

    prompt_text = hero_animation_decision_prompt.format(
        course_name=course_name or "",
        target_audience=target_audience or "",
        topic_name=topic_name or "",
        subtopic_name=subtopic_name or "",
        slide_type=slide_type or "",
        slide_title=slide_title or "",
        slide_content=slide_content or "",
        narration_span=narration_span,
        primary_visual_url=primary_visual_url,
    )

    parts = []
    loaded = _append_hero_visual_multimodal_parts(
        parts=parts,
        narration_span=narration_span,
        asset_url=primary_visual_url,
        drive=drive,
    )
    if not loaded:
        raise ValueError(f"Failed to load hero image for scene {scene_id}: {primary_visual_url}")
    parts.append(types.Part(text=prompt_text))

    # print(f"\nHero Animation Decision - Scene ID: {scene_id}")
    # print("\n" + "=" * 80)
    # print("📝 FORMATTED PROMPT")
    # print("=" * 80)
    # print(f"Narration Span: {narration_span or '(empty)'}")
    # print(f"Assigned single-hero visual asset URL: {primary_visual_url}")
    # print("[INLINE IMAGE attached]")
    # print(prompt_text)
    # print("=" * 80 + "\n")

    response_text = invoke_gemini_multimodal(
        parts,
        llm=llm,
        temperature=0.3,
        max_retries=_HERO_DECISION_MAX_RETRIES,
    )
    # print("📤 Hero Animation Decision Full Response from LLM:\n")
    # print(response_text or "")
    # print("\n" + "=" * 100 + "\n")

    plan_text = parse_hero_animation_decision_response(response_text)
    return scene_id, plan_text


def collect_hero_decision_scene_tasks(index, row, course_name, target_audience, drive, llm):
    """
    Build ordered scene specs and async tasks for one Slide Chunks row (phase 1).

    :return: Tuple (ordered_specs, async_tasks, immediate_cell_value).
        immediate_cell_value is set when the whole row can be written without API calls
        (empty manifest, row error, or no hero scenes).
    """
    from agents.graphics_definition_v2.image_editing_for_layout.hero_bbox_spatial import (
        split_hero_plan_scene_blocks,
    )
    from agents.graphics_definition_v2.image_editing_for_layout.hero_scene_parallel import (
        PENDING,
    )

    manifest_text = str(row.get("slideshow_manifest", "")).strip()
    if not manifest_text or manifest_text == "nan" or manifest_text.startswith("ERROR:"):
        return [], [], ""

    scenes = parse_scenes_from_slideshow_manifest(manifest_text)
    if not scenes:
        return [], [], "ERROR: No scenes parsed from slideshow_manifest"

    topic_name = str(row.get("Topic", "")).strip()
    subtopic_name = str(row.get("Subtopic", "")).strip()
    slide_title = str(row.get("Slide Chunk Title", "")).strip()
    slide_content = str(row.get("Slide Chunk", "")).strip()
    slide_type = str(row.get("Slide Type", "")).strip()
    if slide_type == "nan":
        slide_type = ""

    existing_plan = str(row.get(_PLAN_COLUMN, "")).strip()
    existing_by_scene = {}
    if existing_plan and existing_plan.lower() not in ("nan", "-"):
        existing_by_scene = dict(split_hero_plan_scene_blocks(existing_plan))

    ordered_specs = []
    async_tasks = []
    ran_any = False
    for fallback_idx, scene in enumerate(scenes, start=1):
        scene_id = str(scene.get("id", "")).strip() or str(fallback_idx)
        if not _is_single_visual_hero_scene(scene):
            continue
        asset_url = _hero_primary_asset_url(scene)
        if not asset_url:
            ordered_specs.append((fallback_idx, scene_id, "ERROR: Missing primary_visual URL"))
            continue
        if _is_video_asset_url(asset_url):
            continue
        prior = existing_by_scene.get(scene_id, "")
        if _plan_block_is_ok(prior):
            ordered_specs.append((fallback_idx, scene_id, prior))
            continue
        ran_any = True
        ordered_specs.append((fallback_idx, scene_id, PENDING))
        async_tasks.append(
            {
                "phase": "plan",
                "row_index": index,
                "sort_key": fallback_idx,
                "scene_id": scene_id,
                "course_name": course_name,
                "target_audience": target_audience,
                "topic_name": topic_name,
                "subtopic_name": subtopic_name,
                "slide_type": slide_type,
                "slide_title": slide_title,
                "slide_content": slide_content,
                "scene": scene,
                "drive": drive,
                "llm": llm,
            }
        )

    if not ran_any and not ordered_specs:
        return [], [], "-"
    return ordered_specs, async_tasks, None


def worker_hero_decision_scene(task):
    """Run one phase-1 hero overlay decision scene task."""
    _, plan_text = generate_hero_animation_decision_for_scene(
        course_name=task["course_name"],
        target_audience=task["target_audience"],
        topic_name=task["topic_name"],
        subtopic_name=task["subtopic_name"],
        slide_type=task["slide_type"],
        slide_title=task["slide_title"],
        slide_content=task["slide_content"],
        scene=task["scene"],
        drive=task["drive"],
        llm=task["llm"],
    )
    return task["row_index"], task["sort_key"], task["scene_id"], plan_text


@traceable(
    metadata={
        "agent_name": "graphics_definition_v2",
        "step_name": "Hero Overlay Animation Decision",
        "function_name": "process_hero_animation_decision_row",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous"),
    }
)
def process_hero_animation_decision_row(index, row, course_name, target_audience, drive, llm="gemini_3_flash_thinking"):
    """
    Run hero overlay decisions for every still single-hero scene on one Slide Chunks row.

    :param index: DataFrame row index.
    :param row: DataFrame row object.
    :param course_name: Course name from Course info.
    :param target_audience: Target audience from Course info.
    :param drive: Drive client for loading hero images.
    :param llm: LLM identifier.
    :return: Tuple of (index, plan_text).
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

        existing_plan = str(row.get(_PLAN_COLUMN, "")).strip()
        existing_by_scene = {}
        if existing_plan and existing_plan.lower() not in ("nan", "-"):
            from agents.graphics_definition_v2.image_editing_for_layout.hero_bbox_spatial import (
                split_hero_plan_scene_blocks,
            )
            existing_by_scene = dict(split_hero_plan_scene_blocks(existing_plan))

        plan_blocks = []
        ran_any = False
        for fallback_idx, scene in enumerate(scenes, start=1):
            scene_id = str(scene.get("id", "")).strip() or str(fallback_idx)
            if not _is_single_visual_hero_scene(scene):
                continue
            asset_url = _hero_primary_asset_url(scene)
            if not asset_url:
                plan_blocks.append(_scene_block(scene_id, "ERROR: Missing primary_visual URL"))
                continue
            if _is_video_asset_url(asset_url):
                continue
            prior = existing_by_scene.get(scene_id, "")
            if _plan_block_is_ok(prior):
                plan_blocks.append(_scene_block(scene_id, prior))
                continue
            ran_any = True
            try:
                _, plan_text = generate_hero_animation_decision_for_scene(
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
                )
                plan_blocks.append(_scene_block(scene_id, plan_text))
            except Exception as scene_err:
                err_msg = f"ERROR: {str(scene_err)}"
                plan_blocks.append(_scene_block(scene_id, err_msg))
                traceback.print_exc()

        if not ran_any and not plan_blocks:
            return index, "-"

        return index, "\n\n".join(plan_blocks).strip()
    except Exception as e:
        print(f"Error hero animation decision row {index}: {e}")
        traceback.print_exc()
        return index, f"ERROR: {str(e)}"


def run_hero_spatial_and_icon_phases_parallel(ws, df, drive, llm="gemini_3_flash_thinking", max_workers=50, progress=None):
    """
    Run Phase 2 (bbox spatial) and Phase 3 (icon + callout + emphasis generation) in parallel.

    Scene tasks run in parallel within each row; up to max_workers rows run at once.
    A global API semaphore (30) caps concurrent Gemini/Drive scene workers.
    """
    from agents.graphics_definition_v2.image_editing_for_layout.hero_bbox_spatial import (
        collect_bbox_scene_tasks,
        worker_bbox_scene,
    )
    from agents.graphics_definition_v2.image_editing_for_layout.hero_icon_generation import (
        collect_callout_scene_tasks,
        collect_icon_scene_tasks,
        worker_callout_scene,
        worker_icon_scene,
    )
    from agents.graphics_definition_v2.image_editing_for_layout.hero_scene_parallel import (
        GLOBAL_API_SEMAPHORE_LIMIT,
        build_cached_row_text,
        execute_nested_row_scene_batch,
    )

    sheet_lock = threading.Lock()
    from agents.graphics_definition_v2.image_editing_for_layout.hero_segmentation_spatial import (
        collect_emphasis_scene_tasks,
        emphasis_cell_needs_processing,
        worker_emphasis_scene,
    )

    if _CALLOUTS_COLUMN not in df.columns:
        df[_CALLOUTS_COLUMN] = ""
    if _EMPHASIS_COLUMN not in df.columns:
        df[_EMPHASIS_COLUMN] = ""

    spatial_rows = []
    icon_rows = []
    callout_rows = []
    emphasis_rows = []
    for index, row in df.iterrows():
        slide_type = str(row.get("Slide Type", "")).strip().lower()
        plan_text = str(row.get(_PLAN_COLUMN, "")).strip()

        existing_coords = str(row.get(_COORDS_COLUMN, "")).strip()
        if slide_type == "transition":
            if not existing_coords or existing_coords == "nan" or existing_coords.startswith("ERROR:"):
                df.at[index, _COORDS_COLUMN] = "-"
        elif (
            (not existing_coords or existing_coords == "nan" or existing_coords.startswith("ERROR:"))
            and plan_text
            and plan_text != "nan"
        ):
            spatial_rows.append((index, row))

        existing_icons = str(row.get(_ICONS_COLUMN, "")).strip()
        if slide_type == "transition":
            if not existing_icons or existing_icons == "nan" or existing_icons.startswith("ERROR:"):
                df.at[index, _ICONS_COLUMN] = "-"
        elif (
            (not existing_icons or existing_icons == "nan" or existing_icons.startswith("ERROR:"))
            and plan_text
            and plan_text != "nan"
        ):
            icon_rows.append((index, row))

        existing_callouts = str(row.get(_CALLOUTS_COLUMN, "")).strip()
        if slide_type == "transition":
            if not existing_callouts or existing_callouts == "nan" or existing_callouts.startswith("ERROR:"):
                df.at[index, _CALLOUTS_COLUMN] = "-"
        elif (
            (not existing_callouts or existing_callouts == "nan" or existing_callouts.startswith("ERROR:"))
            and plan_text
            and plan_text != "nan"
        ):
            callout_rows.append((index, row))

        existing_emphasis = str(row.get(_EMPHASIS_COLUMN, "")).strip()
        if slide_type == "transition":
            if not existing_emphasis or existing_emphasis == "nan" or existing_emphasis.startswith("ERROR:"):
                df.at[index, _EMPHASIS_COLUMN] = "-"
        elif emphasis_cell_needs_processing(existing_emphasis) and plan_text and plan_text != "nan":
            emphasis_rows.append((index, row))

    row_jobs = []
    collectors = [
        ("spatial", spatial_rows, lambda i, r: collect_bbox_scene_tasks(i, r, llm)),
        ("icon", icon_rows, lambda i, r: collect_icon_scene_tasks(i, r)),
        ("callout", callout_rows, lambda i, r: collect_callout_scene_tasks(i, r)),
        ("emphasis", emphasis_rows, lambda i, r: collect_emphasis_scene_tasks(i, r)),
    ]
    column_by_phase = {
        "spatial": _COORDS_COLUMN,
        "icon": _ICONS_COLUMN,
        "callout": _CALLOUTS_COLUMN,
        "emphasis": _EMPHASIS_COLUMN,
    }

    for phase, rows, collect_fn in collectors:
        for index, row in rows:
            specs, tasks, immediate = collect_fn(index, row)
            if immediate is not None:
                df.at[index, column_by_phase[phase]] = immediate
            elif specs:
                if tasks:
                    row_jobs.append(
                        {
                            "phase": phase,
                            "row_index": index,
                            "ordered_specs": specs,
                            "tasks": tasks,
                            "drive": drive,
                            "llm": llm,
                        }
                    )
                else:
                    df.at[index, column_by_phase[phase]] = build_cached_row_text(
                        specs, _scene_block, "-"
                    )

    if not row_jobs:
        save_to_sheet(ws, df)
        print("Phase 2/3: no scene tasks (spatial, icon, callout, or emphasis).")
        return

    total_scenes = sum(len(job["tasks"]) for job in row_jobs)
    bbox_n = sum(1 for j in row_jobs if j["phase"] == "spatial" for _ in j["tasks"])
    icon_n = sum(1 for j in row_jobs if j["phase"] == "icon" for _ in j["tasks"])
    callout_n = sum(1 for j in row_jobs if j["phase"] == "callout" for _ in j["tasks"])
    emphasis_n = sum(1 for j in row_jobs if j["phase"] == "emphasis" for _ in j["tasks"])
    print(
        f"Phase 2/3: {total_scenes} scene task(s) in {len(row_jobs)} row job(s) "
        f"({bbox_n} bbox, {emphasis_n} emphasis, {icon_n} icon, {callout_n} callout)"
    )

    if progress is not None:
        progress.add_tasks(total_scenes)

    def on_row_complete(phase, row_index, text):
        df.at[row_index, column_by_phase[phase]] = text

    execute_nested_row_scene_batch(
        row_jobs=row_jobs,
        worker_by_phase={
            "spatial": worker_bbox_scene,
            "icon": worker_icon_scene,
            "callout": worker_callout_scene,
            "emphasis": worker_emphasis_scene,
        },
        scene_block_fn=_scene_block,
        on_row_complete=on_row_complete,
        max_row_workers=max_workers,
        api_semaphore_limit=GLOBAL_API_SEMAPHORE_LIMIT,
        progress=progress,
        log_label="Phase 2/3",
        log_every=5,
        on_checkpoint=lambda: save_to_sheet(ws, df),
        empty_fallback="-",
        sheet_lock=sheet_lock,
    )

    save_to_sheet(ws, df)
    print("Phase 2/3: complete.")


@traceable(
    metadata={
        "agent_name": "graphics_definition_v2",
        "step_name": "Hero Overlay Animation Decision",
        "function_name": "run_hero_animation_decision_for_all_rows",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous"),
    }
)
def run_hero_animation_decision_for_all_rows(sheet, llm="gemini_3_flash_thinking", max_workers=50):
    """
    Decide hero overlay animations, then generate spatial coordinates and icon overlays in parallel.

    :param sheet: gspread sheet object.
    :param llm: LLM identifier for the decision and spatial calls.
    :param max_workers: Outer row parallel workers (default 50); scenes parallel within each row.
    :return: None
    """

    worksheet_name = "Slide Chunks"

    print("Hero overlay step: starting phase 1 (overlay decisions)...")

    _, course_info_df = get_sheet_data_and_df(sheet, "Course info")
    course_name = str(course_info_df.loc[0, "Course Name"]).strip() if not course_info_df.empty else ""
    target_audience = (
        str(course_info_df.loc[0, "Target Audience & Industry"]).strip()
        if not course_info_df.empty
        else ""
    )

    ws, df = get_sheet_data_and_df(sheet, worksheet_name)
    if _PLAN_COLUMN not in df.columns:
        df[_PLAN_COLUMN] = ""
    if _COORDS_COLUMN not in df.columns:
        df[_COORDS_COLUMN] = ""
    if _ICONS_COLUMN not in df.columns:
        df[_ICONS_COLUMN] = ""
    if _CALLOUTS_COLUMN not in df.columns:
        df[_CALLOUTS_COLUMN] = ""
    if _EMPHASIS_COLUMN not in df.columns:
        df[_EMPHASIS_COLUMN] = ""

    drive = get_drive_instance()
    if not drive:
        print("Hero animation decision: Drive not available; some image loads may fail.")

    rows_to_process = []
    for index, row in df.iterrows():
        existing_plan = str(row.get(_PLAN_COLUMN, "")).strip()
        if str(row.get("Slide Type", "")).strip().lower() == "transition":
            if not existing_plan or existing_plan == "nan" or existing_plan.startswith("ERROR:"):
                df.at[index, _PLAN_COLUMN] = "-"
                df.at[index, _COORDS_COLUMN] = "-"
                df.at[index, _ICONS_COLUMN] = "-"
                df.at[index, _CALLOUTS_COLUMN] = "-"
                df.at[index, _EMPHASIS_COLUMN] = "-"
            continue
        manifest = str(row.get("slideshow_manifest", "")).strip()
        if not manifest or manifest == "nan" or manifest.startswith("ERROR:"):
            continue
        if existing_plan and existing_plan != "nan" and not plan_cell_needs_processing(existing_plan):
            continue
        rows_to_process.append((index, row))

    from agents.graphics_definition_v2.image_editing_for_layout.hero_scene_parallel import (
        GLOBAL_API_SEMAPHORE_LIMIT,
        OverlayStepProgress,
        build_cached_row_text,
        execute_nested_row_scene_batch,
    )

    progress = OverlayStepProgress(
        description="Hero overlay animation decision",
        save_interval=5,
    )

    sheet_lock = threading.Lock()

    plan_row_jobs = []
    for index, row in rows_to_process:
        specs, tasks, immediate = collect_hero_decision_scene_tasks(
            index, row, course_name, target_audience, drive, llm
        )
        if immediate is not None:
            df.at[index, _PLAN_COLUMN] = immediate
        elif specs:
            if tasks:
                plan_row_jobs.append(
                    {
                        "phase": "plan",
                        "row_index": index,
                        "ordered_specs": specs,
                        "tasks": tasks,
                    }
                )
            else:
                df.at[index, _PLAN_COLUMN] = build_cached_row_text(specs, _scene_block, "-")

    if rows_to_process:
        if plan_row_jobs:
            total_scenes = sum(len(j["tasks"]) for j in plan_row_jobs)
            progress.add_tasks(total_scenes)
            print(
                f"Phase 1: {total_scenes} scene task(s) in {len(plan_row_jobs)} row(s), "
                f"row_workers={max_workers}, api_semaphore={GLOBAL_API_SEMAPHORE_LIMIT}"
            )

            def on_plan_row_complete(_phase, row_index, text):
                df.at[row_index, _PLAN_COLUMN] = text

            execute_nested_row_scene_batch(
                row_jobs=plan_row_jobs,
                worker_by_phase={"plan": worker_hero_decision_scene},
                scene_block_fn=_scene_block,
                on_row_complete=on_plan_row_complete,
                max_row_workers=max_workers,
                api_semaphore_limit=GLOBAL_API_SEMAPHORE_LIMIT,
                progress=progress,
                log_label="Phase 1",
                log_every=5,
                on_checkpoint=lambda: save_to_sheet(ws, df),
                empty_fallback="-",
                sheet_lock=sheet_lock,
            )
            save_to_sheet(ws, df)
            print("Phase 1: complete.")
        else:
            print("Phase 1: all rows resolved from cache (no scene tasks).")
            save_to_sheet(ws, df)
    else:
        print("Phase 1: no rows to process.")

    print("Starting phase 2/3 (bbox, emphasis, icons, callouts)...")
    run_hero_spatial_and_icon_phases_parallel(ws, df, drive, llm=llm, max_workers=max_workers, progress=progress)
    progress.finish()
    format_worksheet(ws)
    print("Hero overlay step: complete.")


def delete_hero_animation_decision_columns(sheet):
    """
    Remove hero animation decision columns from the Slide Chunks worksheet.

    :param sheet: gspread sheet object.
    :return: None
    """
    worksheet_name = "Slide Chunks"
    ws, df = get_sheet_data_and_df(sheet, worksheet_name)
    columns_to_delete = [_PLAN_COLUMN, _COORDS_COLUMN, _ICONS_COLUMN, _CALLOUTS_COLUMN, _EMPHASIS_COLUMN]
    existing = [col for col in columns_to_delete if col in df.columns]
    if not existing:
        print(f"No hero animation decision columns to delete on '{worksheet_name}'.")
        return
    df = df.drop(columns=existing)
    clear_worksheet(ws)
    save_to_sheet(ws, df)
    print(f"Deleted columns {existing} from '{worksheet_name}' worksheet")
