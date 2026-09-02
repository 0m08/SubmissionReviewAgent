import re
import traceback
import threading
import os
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

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
from agents.graphics_definition_v2.slideshow_manifest.slideshow_manifest import (
    normalize_when_vo_line,
    parse_when_vo_assigned_pairs,
    urls_match_for_graphics_assignment,
)
from services.sheets_service import (
    clear_worksheet,
    format_worksheet,
    get_sheet_data_and_df,
    merge_and_save_columns,
    save_to_sheet,
)

load_dotenv()

multivisual_animation_decision_prompt = """You are a senior instructional designer specializing in HVAC e-learning content.

## Context

Our courses play as narrated slideshows. Each slide is divided into one or more scenes — short narration spans, each with its own layout template.
You are planning for a multi-visual layout scene (one of: 'two_item_split_comparison', 'multi_panel_grid', or 'main_plus_supporting_inset'). These layouts display multiple images or videos side-by-side on the screen at the same time.

Your task is to decide whether this e-learning video player should display clean, professional text label card overlays on top of the visuals during this scene.
These overlay labels are drawn dynamically by the player at runtime at the bottom of each visual panel. 

For most scenes, the images alone are sufficient — in which case, choose **none**. Only choose **text_label** when explicitly naming or contrasting the on-screen components, concepts, or states is instructionally necessary for learner comprehension.

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

<multivisual_scene>
Layout Template: {layout_template}
Narration span of this scene: {narration_span}
</multivisual_scene>

<scene_slots>
{scene_slots}
</scene_slots>

You will also receive a composed, unified screenshot of the entire layout scene as it will appear on the course player canvas, showing all side-by-side/split visual panels loaded together inside their layout structure. Use this unified layout preview to see all visuals together as a whole, understand their spatial relationship, and decide if they require labeling to maintain a clean, symmetrical visual balance and learner understanding.

## Decision rules

1. Instructional Label Necessity:
   - Choose text_label only if labeling is essential to help the learner distinguish, contrast, or identify the parallel components or processes shown.
   - For example, if two split panels show a "Dirty Filter" vs "Clean Filter", or a "Scroll Compressor" vs "Reciprocating Compressor", labeling them is highly effective.
   - If the visuals are generic, transition-like, or already self-explanatory, choose **none**.
   - Your default should be none, unless you have a very good reason to add text labels for all the visuals in the scene

2. Mandatory Symmetry & Visual Balance Rule:
   - Symmetrical layout is a core pillar of professional design. It looks amateurish and unfinished if one panel carries a large orange label badge while sibling panels are empty.
   - Therefore, if any single visual in a multi-visual scene requires a text label, you MUST provide a corresponding text label for all other non-video visuals in that scene.**

3. Label Content Guidelines:
   - Labels must be short, direct, and instructional (1–3 words).
   - Use plain, strong technical terms matching or directly supporting the narration.

## Output format

Provide your output strictly in this XML-enclosed format:

<output>

<evaluation_breakdown>
Use this section as a structured reasoning and scratchpad space for you before producing the final output.

- Slide Understanding: Briefly state in your own words what the slide is about. Then, explain what the scene is communicating.
- Multi-Visual Layout & Asset Scan: Describe what is visible across the different slot panels in the composed layout preview. Are there distinct objects/states/components shown in the sibling panels?
- Overlay Necessity & Symmetry Plan: Analyze and decide whether text labels are instructionally necessary to name, identify, or contrast the parallel components. If you decide to add a label, list the balanced, symmetrical labels you will assign to ALL non-video slots.
- Additional Analysis: Any extra observations needed to justify the final choice.

It is acceptable for this section to be quite long and detailed if needed for correctness.
</evaluation_breakdown>

(Based on your above evaluation, provide the final output below in this exact format.)

<multivisual_animation_plan>

<animation_type>
none | text_label
</animation_type>

<reason>
Briefly explain why you chose the animation_type you did.
</reason>

<label_text>

Provide a semicolon-separated (';') list of labels matching the exact order of slots listed in <scene_slots>.
(For example, if the scene has two slots (left_visual and right_visual):
<label_text>CONDENSER COIL; EVAPORATOR COIL</label_text>

If the scene has three slots (panel_1, panel_2, panel_3):
<label_text>DIRTY FILTER; BLOWER FAN; CLEAN COIL</label_text>

If a slot is a video, or you chose animation_type = none, output N/A for those labels or the entire field:
<label_text>N/A</label_text>)

</label_text>

</multivisual_animation_plan>

</output>

Do not write any markdown formatting, commentary, or text outside the <output> block.
"""

_PLAN_COLUMN = "multivisual_animation_plan"
_DIRECT_VIDEO_FILE_SUFFIXES = (".mp4", ".webm", ".mov")

def _clean_asset_url(url):
    if not url:
        return ""
    # Strip whitespace and remove trailing parentheticals like " (AI Generated)"
    s = str(url).strip()
    m = re.match(r"(https?://[^\s\)\(]+)", s)
    if m:
        return m.group(1).strip()
    return s

def _make_robust_slot_narration_resolver(fgd_text):
    pairs = parse_when_vo_assigned_pairs(fgd_text or "")
    # Clean the paired URLs
    cleaned_pairs = []
    for vo, pu in pairs:
        cleaned_pairs.append((vo, _clean_asset_url(pu)))
    used_indices = set()

    def resolve(asset_url):
        clean_target = _clean_asset_url(asset_url)
        for j, (vo, pu) in enumerate(cleaned_pairs):
            if j in used_indices:
                continue
            if urls_match_for_graphics_assignment(clean_target, pu):
                used_indices.add(j)
                return normalize_when_vo_line(vo)
        return ""

    return resolve

def _normalize_scene_template(template):
    return str(template or "").strip().lower().replace("-", "_").replace(" ", "_")

def _is_video_asset_url(url):
    if not url:
        return False
    if is_youtube_url(url) or is_drive_video_url(url):
        return True
    u = str(url).lower()
    return any(suffix in u for suffix in _DIRECT_VIDEO_FILE_SUFFIXES)

def _is_multivisual_scene(scene):
    template = _normalize_scene_template(scene.get("template"))
    return template in ("two_item_split_comparison", "multi_panel_grid", "main_plus_supporting_inset")

def _scene_block(scene_id, body):
    return f"---Scene ID: {scene_id}---\n{body}".strip()

def parse_multivisual_animation_response(response_text):
    if not response_text:
        return ""
    plan_match = re.search(
        r"<multivisual_animation_plan>(.*?)</multivisual_animation_plan>",
        response_text,
        re.DOTALL | re.IGNORECASE,
    )
    if not plan_match:
        return response_text.strip()
    return plan_match.group(1).strip()

def _upload_composed_preview_to_drive(pil_image, drive, slide_title):
    if pil_image is None or not drive:
        return None
    folder_id = "1an92bpldViG1KL25SPrPyC7sJ2ASYXpZ"
    safe_title = re.sub(r"[^\w\-]+", "_", (slide_title or "slide").strip())
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{safe_title}_{timestamp}.jpg"
    tmp_path = None
    try:
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            pil_image.save(tmp, format="JPEG", quality=85)
            tmp_path = tmp.name
        
        file_drive = drive.CreateFile({
            "title": filename,
            "parents": [{"id": folder_id}],
        })
        file_drive.SetContentFile(tmp_path)
        file_drive.Upload()
        file_id = file_drive.get("id")
        if file_id:
            drive_url = f"https://drive.google.com/file/d/{file_id}/view"
            print(f"[multivisual_decision] Successfully uploaded composed preview to Drive: {drive_url}")
            return drive_url
    except Exception as e:
        print(f"[multivisual_decision] Failed to upload composed preview '{filename}' to Drive: {e}")
        traceback.print_exc()
    finally:
        if tmp_path:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
    return None

def _append_scene_slots_multimodal_parts(parts, scene, resolver, drive, slide_title=None):
    from agents.graphics_definition_v2.review_agent.layout_revision_agent import (
        make_agent_scene_preview_image,
    )

    # Try baking a unified composed layout screenshot first (with the slide title in the image)
    try:
        clean_scene = {**scene, "narration": ""}
        preview = make_agent_scene_preview_image(slide_title or "", clean_scene, drive)
        if preview is not None:
            # Save and upload to drive if slide_title and drive is provided
            if slide_title and drive:
                _upload_composed_preview_to_drive(preview, drive, slide_title)

            buffered = BytesIO()
            preview.save(buffered, format="JPEG", quality=85)
            parts.append(types.Part(text="Composed Layout Preview (as shown on player canvas):"))
            parts.append(
                types.Part(
                    inline_data=types.Blob(mime_type="image/jpeg", data=buffered.getvalue())
                )
            )
            print(f"[multivisual_decision] Successfully appended composed scene screenshot for scene {scene.get('id')}")
    except Exception as e:
        print(f"[multivisual_decision] Failed to bake scene preview image: {e}")

    # Regardless of composed preview success, always append individual slot images with exact scene_slots label format
    slots = scene.get("slots") or []
    for i, slot in enumerate(slots):
        role = slot.get("role", f"slot_{i+1}")
        asset_url = _clean_asset_url(slot.get("asset", ""))
        assigned_vo = resolver(asset_url) if asset_url else ""
        
        label = f"Slot Index: {i} | Role: {role} | Assigned VO: \"{assigned_vo}\" | Asset URL: {asset_url}\n"
        parts.append(types.Part(text=label))
        
        if _is_video_asset_url(asset_url):
            parts.append(types.Part(text="[Asset is a video, skipping image scan]"))
            continue
            
        visual_part = build_visual_part_only(asset_url, drive)
        if visual_part:
            parts.append(visual_part)
        else:
            parts.append(types.Part(text="[Visual could not be loaded]"))

@traceable(
    metadata={
        "agent_name": "graphics_definition_v2",
        "step_name": "Multi-Visual Overlay Animation Decision",
        "function_name": "generate_multivisual_animation_decision_for_scene",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous"),
    }
)
def generate_multivisual_animation_decision_for_scene(course_name, target_audience, topic_name, subtopic_name, slide_type, slide_title, slide_content, scene, final_graphics_definition, drive, llm="gemini_3_flash_thinking"):
    scene_id = str(scene.get("id", "")).strip() or "1"
    layout_template = str(scene.get("template", "")).strip()
    narration_span = str(scene.get("narration", "")).strip()
    slots = scene.get("slots") or []

    # If any slot is a video, skip LLM call and default to no text labels
    if any(_is_video_asset_url(slot.get("asset", "")) for slot in slots):
        formatted_plan = (
            "<multivisual_animation_plan>\n"
            "<animation_type>none</animation_type>\n"
            "<reason>A video asset is present in the scene slots; skipped overlay text label planning to preserve player video performance and layout symmetry.</reason>\n"
            "<label_text>N/A</label_text>\n"
            "</multivisual_animation_plan>"
        )
        return scene_id, formatted_plan

    # Build slot voiceover narration resolver for prompt text using the robust cleaned method
    prompt_resolver = _make_robust_slot_narration_resolver(final_graphics_definition)

    # Format slots info for prompt
    slots_block_lines = []
    for i, slot in enumerate(slots):
        role = slot.get("role", f"slot_{i+1}")
        asset = _clean_asset_url(slot.get("asset", ""))
        assigned_vo = prompt_resolver(asset) if asset else ""
        slots_block_lines.append(f"Slot Index: {i} | Role: {role} | Assigned VO: \"{assigned_vo}\" | Asset URL: {asset}")
    scene_slots_text = "\n".join(slots_block_lines)

    prompt_text = multivisual_animation_decision_prompt.format(
        course_name=course_name or "",
        target_audience=target_audience or "",
        topic_name=topic_name or "",
        subtopic_name=subtopic_name or "",
        slide_type=slide_type or "",
        slide_title=slide_title or "",
        slide_content=slide_content or "",
        layout_template=layout_template,
        narration_span=narration_span,
        scene_slots=scene_slots_text,
    )

    # Build a fresh slot voiceover narration resolver for appending multimodal parts
    multimodal_resolver = _make_robust_slot_narration_resolver(final_graphics_definition)

    parts = []
    _append_scene_slots_multimodal_parts(parts, scene, multimodal_resolver, drive, slide_title=slide_title)
    parts.append(types.Part(text=prompt_text))

    # print(f"\n" + "="*80)
    # print(f"🎬 MULTI-VISUAL ANIMATION DECISION - Scene ID: {scene_id}")
    # print("="*80)
    # print("📝 MULTIMODAL TEXT PARTS:")
    # for idx, p in enumerate(parts[:-1]):
    #     if p.text:
    #         print(f"Part {idx+1} (Text): {p.text.strip()}")
    #     elif p.inline_data:
    #         print(f"Part {idx+1} (Image/Video Blob): MIME: {p.inline_data.mime_type}")
    # print("\n📝 FORMATTED PROMPT:")
    # print("-" * 80)
    # print(prompt_text)
    # print("-" * 80)
    # print("="*80 + "\n")

    response_text = invoke_gemini_multimodal(parts, llm=llm, temperature=0.3)
    plan_text = parse_multivisual_animation_response(response_text)
    
    # Wrap with our custom tag for unified parsing if needed
    formatted_plan = f"<multivisual_animation_plan>\n<animation_type>{_extract_xml_tag(plan_text, 'animation_type')}</animation_type>\n<reason>{_extract_xml_tag(plan_text, 'reason')}</reason>\n<label_text>{_extract_xml_tag(plan_text, 'label_text')}</label_text>\n</multivisual_animation_plan>"
    return scene_id, formatted_plan

def _extract_xml_tag(text, tag):
    match = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else "N/A"

def collect_multivisual_decision_scene_tasks(
    index, row, course_name, target_audience, drive, llm="gemini_3_flash_thinking"
):
    """
    Build ordered scene specs and async tasks for multi-visual scenes on one row.

    :return: Tuple (ordered_specs, async_tasks, immediate_cell_value).
    """
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
    final_graphics_definition = str(row.get("final_graphics_definition", "")).strip()

    ordered_specs = []
    async_tasks = []
    for fallback_idx, scene in enumerate(scenes, start=1):
        scene_id = str(scene.get("id", "")).strip() or str(fallback_idx)
        if not _is_multivisual_scene(scene):
            continue
        ordered_specs.append((fallback_idx, scene_id, PENDING))
        async_tasks.append(
            {
                "phase": "multivisual",
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
                "final_graphics_definition": final_graphics_definition,
                "scene": scene,
                "drive": drive,
                "llm": llm,
            }
        )

    if not ordered_specs:
        return [], [], ""
    return ordered_specs, async_tasks, None


def worker_multivisual_decision_scene(task):
    """Run one multi-visual overlay decision scene task."""
    _, plan_text = generate_multivisual_animation_decision_for_scene(
        course_name=task["course_name"],
        target_audience=task["target_audience"],
        topic_name=task["topic_name"],
        subtopic_name=task["subtopic_name"],
        slide_type=task["slide_type"],
        slide_title=task["slide_title"],
        slide_content=task["slide_content"],
        scene=task["scene"],
        final_graphics_definition=task["final_graphics_definition"],
        drive=task["drive"],
        llm=task["llm"],
    )
    return task["row_index"], task["sort_key"], task["scene_id"], plan_text


@traceable(
    metadata={
        "agent_name": "graphics_definition_v2",
        "step_name": "Multi-Visual Overlay Animation Decision",
        "function_name": "process_multivisual_animation_decision_row",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous"),
    }
)
def process_multivisual_animation_decision_row(index, row, course_name, target_audience, drive, llm="gemini_3_flash_thinking"):
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
        final_graphics_definition = str(row.get("final_graphics_definition", "")).strip()

        plan_blocks = []
        ran_any = False
        for fallback_idx, scene in enumerate(scenes, start=1):
            scene_id = str(scene.get("id", "")).strip() or str(fallback_idx)
            if not _is_multivisual_scene(scene):
                continue
            ran_any = True
            try:
                _, plan_text = generate_multivisual_animation_decision_for_scene(
                    course_name=course_name,
                    target_audience=target_audience,
                    topic_name=topic_name,
                    subtopic_name=subtopic_name,
                    slide_type=slide_type,
                    slide_title=slide_title,
                    slide_content=slide_content,
                    scene=scene,
                    final_graphics_definition=final_graphics_definition,
                    drive=drive,
                    llm=llm,
                )
                plan_blocks.append(_scene_block(scene_id, plan_text))
            except Exception as se:
                print(f"Error in multi-visual decision for row {index+1} scene {scene_id}: {se}")
                plan_blocks.append(_scene_block(scene_id, f"ERROR: {str(se)}"))

        if not ran_any:
            return index, ""
        return index, "\n\n".join(plan_blocks)
    except Exception as e:
        traceback.print_exc()
        return index, f"ERROR: {str(e)}"

@traceable(
    metadata={
        "agent_name": "graphics_definition_v2",
        "step_name": "Multi-Visual Overlay Animation Decision",
        "function_name": "run_multivisual_animation_decision_for_all_rows",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous"),
    }
)
def run_multivisual_animation_decision_for_all_rows(sheet, llm="gemini_3_flash_thinking", max_workers=50):
    worksheet_name = "Slide Chunks"
    print("Multi-visual overlay step: starting...")
    ws, df = get_sheet_data_and_df(sheet, worksheet_name)
    
    if _PLAN_COLUMN not in df.columns:
        df[_PLAN_COLUMN] = ""

    _, course_info_df = get_sheet_data_and_df(sheet, "Course info")
    course_name = str(course_info_df.iloc[0].get("Course Name", "")).strip()
    target_audience = str(course_info_df.iloc[0].get("Target Audience & Industry", "")).strip()

    drive = get_drive_instance()
    rows_to_process = []
    for index, row in df.iterrows():
        manifest_text = str(row.get("slideshow_manifest", "")).strip()
        if not manifest_text or manifest_text == "nan" or manifest_text.startswith("ERROR:"):
            continue
        scenes = parse_scenes_from_slideshow_manifest(manifest_text)
        if any(_is_multivisual_scene(sc) for sc in scenes):
            # Check if we already have some plan, otherwise process
            existing_plan = str(row.get(_PLAN_COLUMN, "")).strip()
            # If we already have decisions, we append or update. To keep it simple, we re-run multi-visual scenes.
            rows_to_process.append((index, row))

    if rows_to_process:
        from agents.graphics_definition_v2.image_editing_for_layout.hero_bbox_spatial import (
            split_hero_plan_scene_blocks,
        )
        from agents.graphics_definition_v2.image_editing_for_layout.hero_scene_parallel import (
            GLOBAL_API_SEMAPHORE_LIMIT,
            OverlayStepProgress,
            build_cached_row_text,
            execute_nested_row_scene_batch,
        )

        sheet_lock = threading.Lock()

        def _merge_multivisual_row(row_index, plan_text):
            if not plan_text:
                return
            current_all_plans = str(df.at[row_index, _PLAN_COLUMN]).strip()
            if (
                current_all_plans
                and current_all_plans != "nan"
                and not current_all_plans.startswith("ERROR:")
            ):
                existing_blocks_dict = {
                    sid: body for sid, body in split_hero_plan_scene_blocks(current_all_plans)
                }
                for n_sid, n_body in split_hero_plan_scene_blocks(plan_text):
                    existing_blocks_dict[n_sid] = n_body
                merged_text = "\n\n".join(
                    _scene_block(sid, body)
                    for sid, body in sorted(existing_blocks_dict.items(), key=lambda x: x[0])
                )
                df.at[row_index, _PLAN_COLUMN] = merged_text
            else:
                df.at[row_index, _PLAN_COLUMN] = plan_text

        row_jobs = []
        for index, row in rows_to_process:
            specs, tasks, immediate = collect_multivisual_decision_scene_tasks(
                index, row, course_name, target_audience, drive, llm
            )
            if immediate is not None:
                if immediate:
                    df.at[index, _PLAN_COLUMN] = immediate
            elif specs:
                if tasks:
                    row_jobs.append(
                        {
                            "phase": "multivisual",
                            "row_index": index,
                            "ordered_specs": specs,
                            "tasks": tasks,
                        }
                    )
                else:
                    merged = build_cached_row_text(specs, _scene_block, "")
                    if merged:
                        _merge_multivisual_row(index, merged)

        progress = OverlayStepProgress(
            description="Multi-visual overlay animation decision",
            save_interval=5,
        )

        if row_jobs:
            total_scenes = sum(len(j["tasks"]) for j in row_jobs)
            progress.add_tasks(total_scenes)
            print(
                f"Multi-visual: {total_scenes} scene task(s) in {len(row_jobs)} row(s), "
                f"row_workers={max_workers}, api_semaphore={GLOBAL_API_SEMAPHORE_LIMIT}"
            )

            execute_nested_row_scene_batch(
                row_jobs=row_jobs,
                worker_by_phase={"multivisual": worker_multivisual_decision_scene},
                scene_block_fn=_scene_block,
                on_row_complete=lambda _phase, row_index, text: _merge_multivisual_row(
                    row_index, text
                ),
                max_row_workers=max_workers,
                api_semaphore_limit=GLOBAL_API_SEMAPHORE_LIMIT,
                progress=progress,
                log_label="Multi-visual",
                log_every=5,
                on_checkpoint=lambda: save_to_sheet(ws, df),
                empty_fallback="",
                sheet_lock=sheet_lock,
            )
        save_to_sheet(ws, df)
    else:
        print("Multi-visual: no rows to process.")
    format_worksheet(ws)
    print("Multi-visual overlay step: complete.")


def delete_multivisual_animation_decision_columns(sheet):
    """
    Remove multivisual animation decision columns from the Slide Chunks worksheet.

    :param sheet: gspread sheet object.
    :return: None
    """
    worksheet_name = "Slide Chunks"
    ws, df = get_sheet_data_and_df(sheet, worksheet_name)
    columns_to_delete = [_PLAN_COLUMN]
    existing = [col for col in columns_to_delete if col in df.columns]
    if not existing:
        print(f"No multivisual animation decision columns to delete on '{worksheet_name}'.")
        return
    df = df.drop(columns=existing)
    clear_worksheet(ws)
    save_to_sheet(ws, df)
    print(f"Deleted columns {existing} from '{worksheet_name}' worksheet")


def run_overlay_animation_decisions_for_all_rows(sheet, llm="gemini_3_flash_thinking", max_workers=50):
    """
    Orchestrates overlay animation decisions for both single visual hero layouts
    and multi-visual layouts (split comparison, panel grid, main plus inset).
    """
    from agents.graphics_definition_v2.image_editing_for_layout.hero_animation_decision import (
        run_hero_animation_decision_for_all_rows,
    )
    print("Overlay animation step: starting hero (single-visual) decisions...")
    run_hero_animation_decision_for_all_rows(sheet, llm=llm, max_workers=max_workers)

    print("Overlay animation step: starting multi-visual decisions...")
    run_multivisual_animation_decision_for_all_rows(sheet, llm=llm, max_workers=max_workers)

    print("Overlay animation step: complete.")


def delete_overlay_animation_decisions_columns(sheet):
    """
    Deletes both hero and multi-visual overlay animation planning columns.
    """
    from agents.graphics_definition_v2.image_editing_for_layout.hero_animation_decision import (
        delete_hero_animation_decision_columns,
    )
    print("Deleting Hero Overlay Animation columns...")
    delete_hero_animation_decision_columns(sheet)
    
    print("Deleting Multi-Visual Overlay Animation columns...")
    delete_multivisual_animation_decision_columns(sheet)
    
    print("All Overlay Animation columns deleted.")


