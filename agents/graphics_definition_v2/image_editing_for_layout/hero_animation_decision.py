import re
import traceback
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
from services.smart_progress_bar import SmartProgressBar

load_dotenv()

hero_animation_decision_prompt = """You are a senior instructional designer specializing in the field of HVAC e-learning content.

## Context

Our courses play as narrated slideshows. Each slide is split into one or more **scenes** — short narration spans, each with its own on-screen layout showing the assigned visuals.
You are planning for a "single_visual_hero" scene: one image fills the slide while its voiceover line plays. There are no side-by-side panels or grids in this layout — just one hero visual and the narration span plays.
Your job is to decide whether the "video player" should add a short instructional overlay animation on top of that hero image — to help the learner see what the narration span is talking about (name a part, highlight some specific spot(s), or show some symbol(s)). These overlays will be drawn as animations at playback time; they will not be baked into the image file.
For most hero scenes, the image alone is enough — in that case you can choose **none**. Only pick an overlay when it clearly helps the learner connect words to what is on screen.
So your task is to decide, for ONE single-hero scene, whether the learner needs an instructional overlay animation on top of the hero image — and if so, which type to use.

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

Use only these four values for <animation_type>. No aliases. No invented types.

1) none

No instructional overlay on the hero image.

Choose none when:
- The image already communicates the narration point without annotation.
- The narration span is general or overview-level and does not call out one specific part, region, or symbol.
- Adding a label, highlight, or icon would clutter the visual or state the obvious.
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
- Only pointing to a location is needed without naming it (use bbox_highlight).

3) bbox_highlight

One or more rectangle or circle outlines animate onto the hero image to emphasize specific visible regions.

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
   - Need one or more SYMBOLIC cues (warning, flow, temperature, etc.) → icon_overlay
   - Image already sufficient → none

## Output format

Provide your output strictly in this format:

<output>

<evaluation_breakdown>
Use this section as a structured reasoning and scratchpad space for you before producing the final output.

- Slide Understanding: Briefly state in your own words what the slide is about.
- Scene Understanding: Briefly explain what this narration span is communicating and what the trainee needs to notice on screen.
- Hero Asset Scan: Briefly describe what is actually visible in the hero image. Do not assume unseen details.
- Overlay Analysis: Decide and say whether you think an overlay is needed. If yes, name the visible target(s) and which type fits (none, text_label, bbox_highlight, icon_overlay) and why the others do not.
- Additional Analysis: Any extra observations needed to justify the final choice.

It is acceptable for this section to be quite long and detailed if needed for correctness.
</evaluation_breakdown>

(Based on your above evaluation, provide the final output below in this exact format.)

<hero_animation_plan>

<animation_type>
none | text_label | bbox_highlight | icon_overlay
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
box | circle 
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

    response_text = invoke_gemini_multimodal(parts, llm=llm, temperature=0.3)
    # print("📤 Hero Animation Decision Full Response from LLM:\n")
    # print(response_text or "")
    # print("\n" + "=" * 100 + "\n")

    plan_text = parse_hero_animation_decision_response(response_text)
    return scene_id, plan_text


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


def run_hero_spatial_and_icon_phases_parallel(ws, df, drive, llm="gemini_3_flash_thinking", max_workers=50):
    """
    Run Phase 2 (bbox spatial coordinates) and Phase 3 (icon generation) in parallel.

    :param ws: Slide Chunks worksheet.
    :param df: Slide Chunks DataFrame (mutated in place).
    :param drive: Drive client.
    :param llm: LLM identifier.
    :param max_workers: Total thread pool workers shared by both phases.
    :return: None
    """
    from agents.graphics_definition_v2.image_editing_for_layout.hero_bbox_spatial import (
        process_hero_bbox_spatial_row,
    )
    from agents.graphics_definition_v2.image_editing_for_layout.hero_icon_generation import (
        process_hero_icon_generation_row,
    )

    # 1. Gather rows for Phase 2 (spatial)
    spatial_rows = []
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
        spatial_rows.append((index, row))

    # 2. Gather rows for Phase 3 (icons)
    icon_rows = []
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
        icon_rows.append((index, row))

    if not spatial_rows and not icon_rows:
        save_to_sheet(ws, df)
        print("Hero parallel phases: no rows to process for spatial coordinates or icon overlays.")
        return

    print(
        f"Hero parallel phases: processing {len(spatial_rows)} spatial row(s) and "
        f"{len(icon_rows)} icon row(s) with max_workers={max_workers}"
    )

    futures_map = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for index, row in spatial_rows:
            fut = executor.submit(process_hero_bbox_spatial_row, index, row, drive, llm)
            futures_map[fut] = (index, "spatial")

        for index, row in icon_rows:
            fut = executor.submit(process_hero_icon_generation_row, index, row, drive)
            futures_map[fut] = (index, "icon")

        progress = SmartProgressBar(
            total_tasks=len(futures_map),
            description="Hero spatial & icon overlays",
            save_interval=5,
        )

        for future in as_completed(futures_map):
            index, phase_type = futures_map[future]
            try:
                row_index, result_text = future.result()
                if phase_type == "spatial":
                    df.at[row_index, _COORDS_COLUMN] = result_text
                else:
                    df.at[row_index, _ICONS_COLUMN] = result_text
                progress.update()
                if progress.should_save():
                    save_to_sheet(ws, df)
            except Exception as e:
                print(f"Hero parallel phases future error in row {index} ({phase_type}): {e}")
                if phase_type == "spatial":
                    df.at[index, _COORDS_COLUMN] = f"ERROR: {str(e)}"
                else:
                    df.at[index, _ICONS_COLUMN] = f"ERROR: {str(e)}"
                progress.update()

    save_to_sheet(ws, df)
    print("Hero parallel phases: complete.")


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
    :param max_workers: Row-level parallel workers.
    :return: None
    """

    worksheet_name = "Slide Chunks"

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
            continue
        manifest = str(row.get("slideshow_manifest", "")).strip()
        if not manifest or manifest == "nan" or manifest.startswith("ERROR:"):
            continue
        if existing_plan and existing_plan != "nan" and not existing_plan.startswith("ERROR:"):
            continue
        rows_to_process.append((index, row))

    if rows_to_process:
        print(f"Hero animation decision phase 1: processing {len(rows_to_process)} row(s), max_workers={max_workers}")
        futures_map = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for index, row in rows_to_process:
                fut = executor.submit(
                    process_hero_animation_decision_row,
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
                description="Hero overlay animation decision",
                save_interval=5,
            )

            for future in as_completed(futures_map):
                index = futures_map[future]
                try:
                    row_index, plan_text = future.result()
                    df.at[row_index, _PLAN_COLUMN] = plan_text
                    progress.update()
                    if progress.should_save():
                        save_to_sheet(ws, df)
                except Exception as e:
                    print(f"Hero animation decision future error row {index}: {e}")
                    df.at[index, _PLAN_COLUMN] = f"ERROR: {str(e)}"
                    progress.update()
        save_to_sheet(ws, df)
    else:
        print("Hero animation decision phase 1: no rows to process.")

    print("Hero animation decision: executing spatial coordinates and icon generation in parallel.")
    run_hero_spatial_and_icon_phases_parallel(ws, df, drive, llm=llm, max_workers=max_workers)
    format_worksheet(ws)
    print("Hero animation decision: complete.")


def delete_hero_animation_decision_columns(sheet):
    """
    Remove hero animation decision columns from the Slide Chunks worksheet.

    :param sheet: gspread sheet object.
    :return: None
    """
    worksheet_name = "Slide Chunks"
    ws, df = get_sheet_data_and_df(sheet, worksheet_name)
    columns_to_delete = [_PLAN_COLUMN, _COORDS_COLUMN, _ICONS_COLUMN]
    existing = [col for col in columns_to_delete if col in df.columns]
    if not existing:
        print(f"No hero animation decision columns to delete on '{worksheet_name}'.")
        return
    df = df.drop(columns=existing)
    clear_worksheet(ws)
    save_to_sheet(ws, df)
    print(f"Deleted columns {existing} from '{worksheet_name}' worksheet")
