from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple

import streamlit as st
from google.genai import types
from langsmith import traceable

from services.sheets_service import (
    clear_worksheet,
    format_worksheet,
    get_sheet_data_and_df,
    merge_and_save_columns,
    merge_and_save_row_cells,
    save_to_sheet,
)
from services.smart_progress_bar import SmartProgressBar
from agents.graphics_definition_v2.review_agent.review_and_revise import (
    _safe_str,
    build_assets_for_segments,
    build_segment_visual_map,
    generate_search_queries_with_feedback,
    get_drive_instance,
    invoke_gemini_multimodal,
    parse_segmented_text,
    parse_segments_from_voiceover,
    replace_segment_block,
    revise_segment_visuals,
    process_segment_other_channels,
    process_web_search_segment,
)
from agents.graphics_definition_v2.aggregation_agent.aggregation_agent import (
    try_expand_youtube_single_timestamp_to_one_second_embed,
)


# Prompt to use when we have flexible or 1 visual per sentence visual assingment strategy
REVIEW_FINALIZED_VISUALS_PROMPT = """You are a Graphics Definition Review Agent specializing in the field of HVAC.

Your Task:
For the given slide, verify that the assigned visual(s) for the given voiceover segment(s) clearly and directly supports what the segment(s) is/are communicating at that moment in the slide narration.

Inputs:
These are the inputs for your evaluation:

<course_information>
Course name: {course_name}
Target audience: {target_audience}
Topic name: {topic_name}
Subtopic name: {subtopic_name}
</course_information>

<slide_information>
Slide ID: {slide_id}
Slide title: {slide_title}
Slide content: "{slide_chunk}"
</slide_information>

These are the voiceover segments of this slide and the assigned visuals for each segment:
<review_targets>
{assigned_visuals}
</review_targets>

Note: Visual IDs follow the format "S(segment_number)V(visual_number)". For example, "S1V3" means Segment 1, Visual 3 (the third visual assigned to segment 1 for its respecitve voiceover text), "S2V1" means Segment 2, Visual 1 (the first visual assigned to segment 2 for its respecitve voiceover text) and so on.

Instructions:
Follow the below evaluation rules to guide your evaluation:

1) Scope
   - Review each voiceover segment independently as the primary evaluation unit.
   - The assigned visuals are displayed on screen as the voiceover text for that segment is played/narrated.
   - You may use the full slide content and the sequence of voiceover segments to understand the intended meaning of a segment (for example, split sentences, pronouns, or continuation phrases).
   - When interpreting any part of the voiceover sentences, you must use the surrounding text in the slide content to understand the context and the instructional intent. Do not interpret a sentence, phrase or a clause literally in isolation. Always derive the meaning of a sentence, phrase or a clause from the surrounding text in the slide content.
   - Judge the relevance only between that segment's intended meaning and the visuals explicitly assigned to it.
   - Use only the provided assets and voiceover text; do not assume missing context beyond what is present in the slide.
   - Each visual asset has a Visual ID (for example, S2V1). Use these IDs when listing any failures.

2) What counts as PASS for a segment
   - The assigned visual(s) clearly show what is described by its respective voiceover text.
   - The visual(s) match the specific meaning of the segment as spoken, not just the general topic of the slide.
   - If multiple visuals are assigned to a segment, together they must fully support the segment’s meaning without introducing confusion or contradiction.
   - All the visuals are usable (loadable and interpretable).
   - A learner should be able to understand what the voiceover segment is referring to by looking at the assigned visual(s) at that moment.

3) What counts as FAIL for a segment
   A visual FAILS if any one of the following is true:
   - It shows something different from what the segment is describing.
   - It is too generic or loosely related and does not clearly support the segment meaning.
   - It contradicts the segment or implies a different instructional idea.
   - It is unusable (broken URL, non-loadable asset, or visually unclear/unreadable).
   - It only matches broad topic context but not its respective segment's instructional intent.

4) Slide-level verdict logic
   - The slide receives a PASS verdict only if ALL visuals assigned to it PASS.
   - If ANY one visual for this slide FAILS, the entire slide verdict must be FAIL.
   - Be extremely strict and critical in your evaluation to ensure that the visuals are correctly aligned with the voiceover segments.

5) Failure Reporting Requirements
   For every failed visual, you MUST:
   - Identify the voiceover segment ID
   - Quote the exact voiceover text
   - List the failing Visual ID
   - Clearly state why the visual does not align with the voiceover
   - Describe the specific visual requirements that would be required for the segment to PASS

Output Format:
Always provide your output strictly in the following format:

<evaluation_breakdown>

Use this section as a structured reasoning and scratchpad space for you to evaluate visual–voiceover alignment for the slide.

- Slide Understanding: State in your own words what the slide is about and what the voiceover segments are trying to convey.
- Review of the assigned visuals: For each segment, list the assigned Visual IDs and briefly describe what is visibly shown in each visual (image or video).
- Visual Alignment Analysis: For each segment, analyze whether the assigned visuals correctly support the respective part of the voiceover segment. 
- Additional Analysis: Note any additional observations, thoughts or analysis that can help you arrive at the correct output and verdict.

(It is ok for this section to be quite verbose, long and detailed as long as it helps you arrive at the correct output.)

</evaluation_breakdown>

(Based on your above evaluation, provide your output in the following format)

<review>

<verdict>
PASS|FAIL
</verdict>

(If the slide verdict is FAIL, provide the details of the failed segments in the following format)

<failures>

<failure>
<segment_id>
(Provide the segment number of the failed segment. e.g. SEGMENT 1)
</segment_id>

<segment_text>
(Provide the exact portion of voiceover text that is not correctly supported by this visual.)
</segment_text>

<failing_visual_id>
(Provide the Visual ID of the assigned visual that does not correctly support the voiceover segment. e.g. S1V3)
</failing_visual_id>

<current_visual_url>
(Provide exact URL of the failing current visual)
</current_visual_url>

<reason>
(Provide the reason why the assigned visual does not correctly support its voiceover text.)
</reason>

<needed_visual>
(Describe the visual requirements that is needed to correctly support the failed voiceover segment. Don't use words like "image" in this section since we are going to replace the faulty visual from a pool of image as well as video candidates. So prefer words like "visual" instead.)
</needed_visual>

</failure>

Repeat the <failure> block for each failed segment and its corresponding visual id. (Even if multiple visuals within the same segment fail, repeat the <failure> block separately for each failing visual.)

</failures>

</review>

(Use this exact XML format given above while providing your output)
"""

# Prompt to use when we have 1 visual for the whole slide visual assingment strategy
REVIEW_FINALIZED_VISUALS_PROMPT_FOR_ONE_VISUAL_PER_SLIDE = """You are a Graphics Definition Review Agent specializing in the field of HVAC.

Your Task:
For the given slide, verify whether the assigned visual for this whole slide clearly and directly supports what the full slide narration is communicating.

Inputs:
These are the inputs for your evaluation:

<course_information>
Course name: {course_name}
Target audience: {target_audience}
Topic name: {topic_name}
Subtopic name: {subtopic_name}
</course_information>

<slide_information>
Slide ID: {slide_id}
Slide title: {slide_title}
Slide content: "{slide_chunk}"
</slide_information>

This is the assigned visual for this whole slide:
<review_targets>
{assigned_visuals}
</review_targets>

Note: Visual IDs follow the format "S(segment_number)V(visual_number)". 

Instructions:
Follow the below evaluation rules to guide your evaluation:

1) Scope
   - Review the assigned visual for the whole slide.
   - The assigned visual is displayed on screen while the full slide content is narrated.
   - Use the full slide content to interpret the intended instructional meaning.
   - Judge relevance only between the slide's intended meaning and the assigned visual.
   - Use only the provided slide context and assigned visual; do not assume missing context.
   - The visual has a Visual ID (for example, S1V1). Use this ID when reporting any failures.
   - IMPORTANT: Know that we have been allowed to assign only one visual asset for this particular slide. So keep that in mind as you evaluate the alignment of the visual to the slide.

2) What counts as PASS for the slide
   - The assigned visual clearly and directly supports the full slide meaning.
   - The visual matches the specific instructional intent of the slide, not just broad topic relevance.
   - The visual does not contradict any important part of the slide content.
   - The visual is usable (loadable and interpretable).
   - A learner can understand what the slide is communicating by seeing this visual while narration plays.

3) What counts as FAIL for the slide
   The visual FAILS if any one of the following is true:
   - It shows something different from what the slide is describing.
   - It is generic or loosely related and does not clearly support the slide meaning.
   - It contradicts the slide or implies a different instructional idea.
   - It is unusable (broken URL, non-loadable asset, or visually unclear/unreadable).
   - It only matches broad topic context but not the slide's specific instructional intent.

4) Slide-level verdict logic
   - The slide receives PASS only if the assigned visual PASSES.
   - If the assigned visual FAILS, the slide verdict must be FAIL.
   - Be extremely strict and critical in your evaluation.

5) Failure Reporting Requirements
   If your verdict for the slide is FAIL, you MUST:
   - List the Segment ID 
   - List the failing Visual ID
   - Clearly state why the visual does not align with the voiceover content of the slide
   - Describe the specific visual requirement that will be required for the slide to PASS.

Output Format:
Always provide your output strictly in the following format:

<evaluation_breakdown>

Use this section as a structured reasoning and scratchpad space for evaluating visual–voiceover alignment for the slide.

- Slide Understanding: State in your own words what the slide is trying to teach.
- Review of the assigned visual: Briefly describe what is visibly shown in the assigned visual.
- Visual Alignment Analysis: Analyze whether the assigned visual correctly supports the slide meaning.
- Additional Analysis: Note any additional observations, thoughts or analysis that can help you arrive at the correct output and verdict.

(It is ok for this section to be quite verbose, long and detailed as long as it helps you arrive at the correct output.)

</evaluation_breakdown>

(Based on your above evaluation, provide your output in the following format)

<review>

<verdict>
PASS|FAIL
</verdict>

(If the slide verdict is FAIL, provide the details of the failed visual in the following format)

<failure>

<segment_id>
(Provide the segment number of the failed segment. e.g. SEGMENT 1)
</segment_id>

<vo_text>
(Provide the entire slide content text as it is.)
</vo_text>

<failing_visual_id>
(Provide the Visual ID of the assigned visual. e.g. S1V1)
</failing_visual_id>

<current_visual_url>
(Provide exact URL of the failing current visual)
</current_visual_url>

<reason>
(Provide the reason why the assigned visual does not correctly support the slide content.)
</reason>

<needed_visual>
(Describe the visual requirement that is needed to correctly support the slide content for this criteria. Don't use words like "image" in this section since we are going to replace the faulty visual from a pool of image as well as video candidates. So prefer words like "visual" instead.)
</needed_visual>

</failure>

</review>

(Use this exact XML format given above while providing your output)
"""



def get_review_finalized_visuals_prompt(
    course_name: str,
    target_audience: str,
    topic_name: str,
    subtopic_name: str,
    slide_id: str,
    slide_title: str,
    slide_chunk: str,
    assigned_visuals: str,
    prompt_template: str = REVIEW_FINALIZED_VISUALS_PROMPT,
) -> str:
    """
    Build the review prompt for finalized visuals for a full slide.
    """
    return prompt_template.format(
        course_name=course_name,
        target_audience=target_audience,
        topic_name=topic_name,
        subtopic_name=subtopic_name,
        slide_id=slide_id,
        slide_title=slide_title,
        slide_chunk=slide_chunk,
        assigned_visuals=assigned_visuals,
    )


def _extract_tag(text: str, tag: str) -> str:
    if not text:
        return ""
    match = re.search(
        rf"<{tag}>(.*?)</{tag}>",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return match.group(1).strip() if match else ""


def _parse_slide_review_response(text: str) -> Tuple[str, List[Dict[str, str]]]:
    verdict = (_extract_tag(text, "verdict") or "FAIL").strip().upper()
    failures: List[Dict[str, str]] = []

    failures_block = _extract_tag(text, "failures")
    failure_blocks = re.findall(
        r"<failure>(.*?)</failure>",
        failures_block or "",
        flags=re.IGNORECASE | re.DOTALL,
    )

    for block in failure_blocks:
        failures.append(
            {
                "segment_id": _extract_tag(block, "segment_id"),
                "segment_text": _extract_tag(block, "segment_text"),
                "failing_visual_id": _extract_tag(block, "failing_visual_id"),
                "current_visual_url": _extract_tag(block, "current_visual_url"),
                "reason": _extract_tag(block, "reason"),
                "needed_visual": _extract_tag(block, "needed_visual"),
            }
        )

    return verdict, failures


def _build_assigned_visuals_for_slide(
    segments_map: Dict[int, Dict[str, Any]],
    slide_id: str,
) -> str:
    lines: List[str] = []
    for segment_num in sorted(segments_map.keys()):
        segment = segments_map.get(segment_num, {})
        lines.append(f"{slide_id} SEGMENT {segment_num}")
        lines.append(f"VO: {segment.get('vo_text', '')}")

        visual_steps = segment.get("visual_steps", []) or []
        if not visual_steps:
            lines.append("No visuals assigned.")
        else:
            for step in visual_steps:
                lines.append(
                    f"{step.get('visual_id', '')} | "
                    f"When VO: {step.get('voiceover_part', '')} | "
                    f"Asset: {step.get('asset', '')}"
                )
        lines.append("")

    return "\n".join(lines).strip()


def _format_slide_review_output(
    segments_map: Dict[int, Dict[str, Any]],
    slide_verdict: str,
    failures: List[Dict[str, str]],
) -> str:
    parts: List[str] = []
    slide_verdict = (slide_verdict or "FAIL").upper()
    failure_by_visual_id = {
        (f.get("failing_visual_id") or "").strip(): f for f in (failures or [])
    }

    for segment_num in sorted(segments_map.keys()):
        segment = segments_map.get(segment_num, {})
        visual_steps = segment.get("visual_steps", []) or []

        parts.append(f"---SEGMENT_{segment_num}---")
        parts.append("")

        if not visual_steps:
            parts.append("No visuals assigned")
            parts.append("")
            continue

        for step in visual_steps:
            visual_id = (step.get("visual_id") or "").strip()
            failure = failure_by_visual_id.get(visual_id)
            if failure:
                reason = (failure.get("reason") or "").strip()
                feedback = (failure.get("needed_visual") or "").strip()
                parts.append(
                    f"{visual_id}: FAIL | Reason: {reason} | Feedback: {feedback}"
                )
            else:
                # If model gave FAIL but failed to return visual-level failures,
                # preserve strict behavior with generic fallback line.
                if slide_verdict == "FAIL" and visual_id not in failure_by_visual_id and failures == []:
                    parts.append(
                        f"{visual_id}: FAIL | Reason: Model returned FAIL without visual-level failure blocks | "
                        f"Feedback: Provide a clearer, segment-accurate visual."
                    )
                else:
                    parts.append(f"{visual_id}: PASS")

        parts.append("")  

    return "\n".join(parts).strip()


@traceable(
    metadata={
        "agent_name": "graphics_definition_v2",
        "step_name": "Review Finalized Visuals",
        "function_name": "process_review_finalized_visuals_row",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous"),
    }
)
def process_review_finalized_visuals_row(
    index: int,
    row: Dict[str, Any],
    course_name: str,
    target_audience: str,
    drive: Any,
    llm: str = "gemini_3_flash_thinking",
) -> Tuple[int, str]:
    slide_chunk = str(row.get("Slide Chunk", "")).strip()
    voiceover_text = str(row.get("voiceover_segment", "")).strip()
    final_graphics_definition = str(row.get("final_graphics_definition", "")).strip()
    topic_name = str(row.get("Topic", "")).strip()
    subtopic_name = str(row.get("Subtopic", "")).strip()
    slide_title = str(row.get("Slide Chunk Title", "")).strip() or "Untitled"
    slide_id = str(row.get("Slide ID", "")).strip() or f"SLIDE {index + 1}"
    visual_assignment_strategy = str(
        row.get("Visual Assignment Strategy", "Flexible, let the agent decide")
    ).strip()
    if not visual_assignment_strategy or visual_assignment_strategy == "nan":
        visual_assignment_strategy = "Flexible, let the agent decide"

    if (
        not slide_chunk
        or slide_chunk == "nan"
        or not final_graphics_definition
        or final_graphics_definition == "nan"
    ):
        return index, ""

    segments_map = build_segment_visual_map(
        voiceover_text=voiceover_text,
        final_graphics_definition=final_graphics_definition,
        visual_assignment_strategy=visual_assignment_strategy,
        slide_chunk=slide_chunk,
    )

    if not segments_map:
        return index, ""

    assigned_visuals = _build_assigned_visuals_for_slide(
        segments_map=segments_map,
        slide_id=slide_id,
    )
    prompt_template = (
        REVIEW_FINALIZED_VISUALS_PROMPT_FOR_ONE_VISUAL_PER_SLIDE
        if visual_assignment_strategy == "1 Visual for the whole Slide"
        else REVIEW_FINALIZED_VISUALS_PROMPT
    )
    prompt = get_review_finalized_visuals_prompt(
        course_name=course_name,
        target_audience=target_audience,
        topic_name=topic_name,
        subtopic_name=subtopic_name,
        slide_id=slide_id,
        slide_title=slide_title,
        slide_chunk=slide_chunk,
        assigned_visuals=assigned_visuals,
        prompt_template=prompt_template,
    )

    print(f"\n{'='*80}")
    print(f"📝 FORMATTED FINALIZED VISUAL REVIEW PROMPT ({slide_id}):")
    print(f"{'='*80}")
    print(prompt)
    print(f"{'='*80}\n")

    segment_nums = sorted(segments_map.keys())
    asset_parts = build_assets_for_segments(segments_map, segment_nums, drive)
    response_text, _ = invoke_gemini_multimodal(
        asset_parts + [types.Part(text=prompt)],
        llm=llm,
        conversation_history=None,
    )

    print(f"\n📤 Finalized Visual Review Response ({slide_id}):\n")
    print(response_text)
    print(f"\n{'='*100}\n")

    verdict, failures = _parse_slide_review_response(response_text)

    review_text = _format_slide_review_output(
        segments_map=segments_map,
        slide_verdict=verdict,
        failures=failures,
    )
    return index, review_text


@traceable(
    metadata={
        "agent_name": "graphics_definition_v2",
        "step_name": "Review Finalized Visuals",
        "function_name": "run_review_finalized_visuals_for_all_rows",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous"),
    }
)
def run_review_finalized_visuals_for_all_rows(
    sheet: Any,
    llm: str = "gemini_3_flash_thinking",
    max_workers: int = 50,
) -> None:
    worksheet_name = "Slide Chunks"

    _, course_info_df = get_sheet_data_and_df(sheet, "Course info")
    course_name = str(course_info_df.loc[0, "Course Name"]).strip()
    target_audience = str(course_info_df.loc[0, "Target Audience"]).strip() if "Target Audience" in course_info_df.columns else ""

    drive = get_drive_instance()
    if not drive:
        print("⚠️ Drive instance not available. Drive assets may fail to load in review.")

    worksheet, df = get_sheet_data_and_df(sheet, worksheet_name)
    if "final_visuals_review" not in df.columns:
        df["final_visuals_review"] = ""

    futures_map = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for index, row in df.iterrows():
            final_graphics_definition = str(row.get("final_graphics_definition", "")).strip()
            existing_review = str(row.get("final_visuals_review", "")).strip()

            if not final_graphics_definition or final_graphics_definition == "nan":
                continue
            if existing_review and existing_review != "nan":
                continue

            future = executor.submit(
                process_review_finalized_visuals_row,
                index,
                row,
                course_name,
                target_audience,
                drive,
                llm,
            )
            futures_map[future] = index

        if not futures_map:
            print("All rows already reviewed or no final_graphics_definition found.")
            return

        progress = SmartProgressBar(
            total_tasks=len(futures_map),
            description="Reviewing finalized visuals",
        )

        for future in as_completed(futures_map):
            index = futures_map[future]
            try:
                row_index, review_text = future.result()
                df.at[row_index, "final_visuals_review"] = review_text
                progress.update()
                merge_and_save_columns(sheet, worksheet_name, df, ["final_visuals_review"])
            except Exception as e:
                df.at[index, "final_visuals_review"] = f"ERROR: {str(e)}"
                progress.update()
                merge_and_save_columns(sheet, worksheet_name, df, ["final_visuals_review"])

    merge_and_save_columns(sheet, worksheet_name, df, ["final_visuals_review"])
    format_worksheet(worksheet)
    print("✅ Finalized visual review complete and saved to sheet.")


def delete_final_visuals_review(sheet: Any) -> None:
    worksheet_name = "Slide Chunks"
    ws, df = get_sheet_data_and_df(sheet, worksheet_name)
    if "final_visuals_review" in df.columns:
        df = df.drop(columns=["final_visuals_review"])
        clear_worksheet(ws)
        save_to_sheet(ws, df)
        print(f"🗑️ Deleted 'final_visuals_review' column from '{worksheet_name}' worksheet")
    else:
        print(f"ℹ️ 'final_visuals_review' column does not exist in '{worksheet_name}' worksheet")


@traceable(
    metadata={
        "agent_name": "graphics_definition_v2",
        "step_name": "Generate Alternative Visuals from Web",
        "function_name": "run_generate_alternative_visuals_from_web_for_all_rows",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous"),
    }
)
def run_generate_alternative_visuals_from_web_for_all_rows(
    sheet: Any,
    llm: str = "gemini_3_flash_thinking",
    max_workers: int = 50,
) -> None:
    """
    Generate alternative visuals from web/other channels for FAILED visuals.

    Flow:
    - Parse failed visuals from final_visuals_review
    - Regenerate queries and run web/other-channel retrieval for failed segments
    - Run existing regeneration aggregation flow to get replacements
    - Keep final_graphics_definition unchanged
    - Write replacement URL back into failed lines in final_visuals_review
    """
    worksheet_name = "Slide Chunks"
    worksheet, df = get_sheet_data_and_df(sheet, worksheet_name)

    original_columns = list(df.columns)
    if "web_results" not in df.columns:
        if "drive_results" in df.columns:
            df.insert(int(df.columns.get_loc("drive_results")) + 1, "web_results", "")
        else:
            df["web_results"] = ""

    if "video_pool_other_channels" not in df.columns:
        if "web_results" in df.columns:
            df.insert(int(df.columns.get_loc("web_results")) + 1, "video_pool_other_channels", "")
        else:
            df["video_pool_other_channels"] = ""

    if list(df.columns) != original_columns:
        save_to_sheet(worksheet, df)
        format_worksheet(worksheet)
        print("✅ Ensured fallback columns with placement near related search columns.")
    else:
        print("ℹ️ Fallback columns already exist and are ready.")

    # Process rows in parallel
    _, course_info_df = get_sheet_data_and_df(sheet, "Course info")
    course_name = str(course_info_df.loc[0, "Course Name"]).strip()
    target_audience = (
        str(course_info_df.loc[0, "Target Audience"]).strip()
        if "Target Audience" in course_info_df.columns
        else ""
    )

    futures_map = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for index, row in df.iterrows():
            final_review = _safe_str(row.get("final_visuals_review", "")).strip()
            final_graphics_definition = _safe_str(row.get("final_graphics_definition", "")).strip()
            if not final_review or final_review.lower() == "nan":
                print(f"⏭️ Skipping row {index + 1}: final_visuals_review is empty.")
                continue
            if not final_graphics_definition or final_graphics_definition.lower() == "nan":
                print(f"⏭️ Skipping row {index + 1}: final_graphics_definition is empty.")
                continue
            if "FAIL" not in final_review:
                print(f"⏭️ Skipping row {index + 1}: no FAIL visuals in final_visuals_review.")
                continue
            if not _has_unresolved_fail_lines(final_review):
                print(
                    f"⏭️ Skipping row {index + 1}: all FAIL visuals already have Replacement visual."
                )
                continue
            future = executor.submit(
                _process_alternative_visuals_row,
                index,
                row,
                course_name,
                target_audience,
                llm,
                sheet,
                worksheet_name,
            )
            futures_map[future] = index

        if not futures_map:
            print("✅ No rows with FAIL visuals found in final_visuals_review.")
            return

        progress = SmartProgressBar(
            total_tasks=len(futures_map),
            description="Generate alternative visuals from Web",
            save_interval=3,
        )

        for future in as_completed(futures_map):
            index = futures_map[future]
            try:
                row_index, updates = future.result()
                if updates:
                    for col, val in updates.items():
                        if col not in df.columns:
                            df[col] = ""
                        df.at[row_index, col] = val
                progress.update()
                if progress.should_save():
                    merge_and_save_columns(
                        sheet,
                        worksheet_name,
                        df,
                        [
                            "search_queries",
                            "web_results",
                            "video_pool_other_channels",
                            "drive_results",
                            "video_pool",
                            "image_pool",
                            "video_pool_filtered",
                            "final_visuals_review",
                        ],
                    )
            except Exception as e:
                print(f"❌ Row {index + 1}: alternative visual generation failed: {e}")
                progress.update()

    merge_and_save_columns(
        sheet,
        worksheet_name,
        df,
        [
            "search_queries",
            "web_results",
            "video_pool_other_channels",
            "drive_results",
            "video_pool",
            "image_pool",
            "video_pool_filtered",
            "final_visuals_review",
        ],
    )
    format_worksheet(worksheet)
    print("✅ Generate alternative visuals from Web completed.")


def _extract_failed_visuals_and_feedback(
    final_visuals_review: str,
) -> Tuple[List[int], Dict[int, str], List[str], Dict[int, List[str]]]:
    failed_segments: List[int] = []
    feedback_by_segment: Dict[int, List[str]] = {}
    failed_visual_ids: List[str] = []
    failed_visual_ids_by_segment: Dict[int, List[str]] = {}

    current_segment = None
    for raw_line in final_visuals_review.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        seg_match = re.match(r"---SEGMENT_(\d+)---", line, re.IGNORECASE)
        if seg_match:
            current_segment = int(seg_match.group(1))
            continue

        fail_match = re.match(r"^(S\d+V\d+):\s*FAIL\b(.*)$", line, re.IGNORECASE)
        if fail_match and current_segment is not None:
            visual_id = fail_match.group(1).strip()
            failed_visual_ids.append(visual_id)
            failed_visual_ids_by_segment.setdefault(current_segment, []).append(visual_id)
            if current_segment not in failed_segments:
                failed_segments.append(current_segment)
            rest = fail_match.group(2) or ""
            reason_match = re.search(
                r"\|\s*Reason:\s*(.*?)(?:\|\s*Feedback:|\|\s*Replacement visual:|$)",
                rest,
                re.IGNORECASE,
            )
            feedback_match = re.search(r"\|\s*Feedback:\s*(.*?)(?:\|\s*Replacement visual:|$)", rest, re.IGNORECASE)
            reason_text = reason_match.group(1).strip() if reason_match else ""
            feedback_text = feedback_match.group(1).strip() if feedback_match else ""
            if not reason_text:
                reason_text = "The currently assigned visual does not adequately support the intended meaning."
            if not feedback_text:
                feedback_text = "Find and assign a better visual that is relevant for this voiceover part."
            failure_text = (
                f"Failing Visual: {visual_id}\n"
                f"Reason: {reason_text}\n"
                f"Needed: {feedback_text}"
            )
            feedback_by_segment.setdefault(current_segment, []).append(failure_text)

    feedback_joined = {
        seg: "\n".join(items) for seg, items in feedback_by_segment.items()
    }
    return failed_segments, feedback_joined, failed_visual_ids, failed_visual_ids_by_segment


def _has_unresolved_fail_lines(final_visuals_review: str) -> bool:
    """
    Return True if at least one FAIL line still lacks a replacement URL.
    """
    for raw_line in (final_visuals_review or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if re.match(r"^S\d+V\d+:\s*FAIL\b", line, re.IGNORECASE):
            if re.search(r"\|\s*Replacement visual:\s*\S+", line, re.IGNORECASE):
                continue
            return True
    return False


def _get_visual_asset_map(segments_map: Dict[int, Dict[str, Any]]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for _, segment in segments_map.items():
        for step in segment.get("visual_steps", []) or []:
            vid = _safe_str(step.get("visual_id", "")).strip()
            url = _safe_str(step.get("asset", "")).strip()
            if vid and url:
                out[vid] = url
    return out


def _append_replacements_to_review_text(
    review_text: str,
    replacement_by_visual_id: Dict[str, str],
) -> str:
    lines: List[str] = []
    for raw_line in review_text.splitlines():
        line = raw_line.rstrip("\n")
        match = re.match(r"^(S\d+V\d+):\s*FAIL\b(.*)$", line, re.IGNORECASE)
        if not match:
            lines.append(line)
            continue
        visual_id = match.group(1).strip()
        replacement_url = replacement_by_visual_id.get(visual_id, "").strip()
        if not replacement_url:
            lines.append(line)
            continue
        line_wo_existing = re.sub(
            r"\s*\|\s*Replacement visual:\s*.*$",
            "",
            line,
            flags=re.IGNORECASE,
        )
        lines.append(f"{line_wo_existing} | Replacement visual: {replacement_url}")
    return "\n".join(lines)


def _normalize_replacement_visual_url(url: str) -> str:
    """
    Normalize replacement URL only for YouTube single-timestamp links by converting to 1-second embed clip.
    """
    clean = _safe_str(url).strip()
    if not clean:
        return clean
    expanded = try_expand_youtube_single_timestamp_to_one_second_embed(clean)
    return expanded or clean


def _process_alternative_visuals_row(
    row_index: int,
    row: Dict[str, Any],
    course_name: str,
    target_audience: str,
    llm: str,
    sheet: Any = None,
    worksheet_name: str = "Slide Chunks",
) -> Tuple[int, Dict[str, str]]:
    final_review = _safe_str(row.get("final_visuals_review", ""))
    final_graphics_definition_original = _safe_str(row.get("final_graphics_definition", ""))
    voiceover_text = _safe_str(row.get("voiceover_segment", ""))
    slide_chunk = _safe_str(row.get("Slide Chunk", ""))
    visual_assignment_strategy = _safe_str(row.get("Visual Assignment Strategy", "Flexible, let the agent decide"))
    if not visual_assignment_strategy or visual_assignment_strategy == "nan":
        visual_assignment_strategy = "Flexible, let the agent decide"

    (
        failed_segments,
        feedback_by_segment,
        failed_visual_ids,
        failed_visual_ids_by_segment,
    ) = _extract_failed_visuals_and_feedback(final_review)
    if not failed_segments:
        return row_index, {}

    print(f"🔁 Row {row_index + 1}: processing failed segments {failed_segments}")

    # Local dataframe copy so regeneration can mutate without touching shared df.
    # Always build a fresh local single-row frame to avoid cross-row contamination.
    import pandas as pd
    local_df = pd.DataFrame([dict(row)], index=[row_index])

    drive = get_drive_instance()
    if not drive:
        print(f"⚠️ Row {row_index + 1}: Drive not available, skipping row.")
        return row_index, {}

    slide_title = _safe_str(local_df.at[row_index, "Slide Chunk Title"])
    topic_name = _safe_str(local_df.at[row_index, "Topic"])
    subtopic_name = _safe_str(local_df.at[row_index, "Subtopic"])
    voiceover_segments = parse_segments_from_voiceover(voiceover_text)
    segments_map = build_segment_visual_map(
        voiceover_text=voiceover_text,
        final_graphics_definition=final_graphics_definition_original,
        visual_assignment_strategy=visual_assignment_strategy,
        slide_chunk=slide_chunk,
    )

    # 1) Regenerate search queries only for failed segments.
    for segment_num in failed_segments:
        vo_text = ""
        for seg_idx, seg_text in voiceover_segments:
            if seg_idx == segment_num:
                vo_text = seg_text
                break
        feedback = feedback_by_segment.get(segment_num, "")
        queries = generate_search_queries_with_feedback(
            course_name=course_name,
            target_audience=target_audience,
            topic_name=topic_name,
            subtopic_name=subtopic_name,
            slide_title=slide_title,
            slide_chunk=slide_chunk,
            vo_text=vo_text,
            feedback=feedback,
            llm=llm,
            segment_num=segment_num,
            segments_map=segments_map,
            drive=drive,
            visual_assignment_strategy=visual_assignment_strategy,
        )
        if queries:
            local_df.at[row_index, "search_queries"] = replace_segment_block(
                _safe_str(local_df.at[row_index, "search_queries"]),
                segment_num,
                queries,
            )
            if sheet is not None:
                merge_and_save_row_cells(
                    sheet,
                    worksheet_name,
                    row_index,
                    {"search_queries": _safe_str(local_df.at[row_index, "search_queries"])},
                )

    # 2) Run only web + other channel searches for failed segments.
    for segment_num in failed_segments:
        queries_map = parse_segmented_text(_safe_str(local_df.at[row_index, "search_queries"]))
        queries = queries_map.get(segment_num, [])
        if not queries:
            continue

        seg_num, web_results = process_web_search_segment(segment_num, queries)
        if web_results:
            lines = [line.strip() for line in web_results.splitlines() if line.strip()]
            local_df.at[row_index, "web_results"] = replace_segment_block(
                _safe_str(local_df.at[row_index, "web_results"]),
                segment_num,
                lines[1:] if lines and lines[0].startswith("---SEGMENT_") else lines,
            )
            if sheet is not None:
                merge_and_save_row_cells(
                    sheet,
                    worksheet_name,
                    row_index,
                    {"web_results": _safe_str(local_df.at[row_index, "web_results"])},
                )

        segment_sentence = ""
        for seg_idx, seg_text in voiceover_segments:
            if seg_idx == segment_num:
                segment_sentence = seg_text
                break
        seg_num, video_other = process_segment_other_channels(segment_num, queries, segment_sentence)
        if video_other:
            lines = [line.strip() for line in video_other.splitlines() if line.strip()]
            local_df.at[row_index, "video_pool_other_channels"] = replace_segment_block(
                _safe_str(local_df.at[row_index, "video_pool_other_channels"]),
                segment_num,
                lines[1:] if lines and lines[0].startswith("---SEGMENT_") else lines,
            )
            if sheet is not None:
                merge_and_save_row_cells(
                    sheet,
                    worksheet_name,
                    row_index,
                    {
                        "video_pool_other_channels": _safe_str(
                            local_df.at[row_index, "video_pool_other_channels"]
                        )
                    },
                )

    # 3) Use regeneration prompt, but constrain candidate inputs to web+other only.
    replacement_by_visual_id: Dict[str, str] = {}
    for segment_num in failed_segments:
        segment = segments_map.get(segment_num, {})
        visual_steps = segment.get("visual_steps", []) or []
        target_visual_ids = set(failed_visual_ids_by_segment.get(segment_num, []))
        current_visual_lines: List[str] = []
        for step in visual_steps:
            vid = _safe_str(step.get("visual_id", "")).strip()
            if vid and vid in target_visual_ids:
                current_visual_lines.append(
                    f'{vid} | When VO: "{_safe_str(step.get("voiceover_part", ""))}" | Visual assigned: {_safe_str(step.get("asset", ""))}'
                )
        if not current_visual_lines:
            continue
        current_visuals = "\n".join(current_visual_lines)
        vo_text = _safe_str(segment.get("vo_text", ""))
        feedback = feedback_by_segment.get(segment_num, "")
        replacement_xml_inner = revise_segment_visuals(
            course_name=course_name,
            target_audience=target_audience,
            topic_name=topic_name,
            subtopic_name=subtopic_name,
            slide_title=slide_title,
            slide_chunk=slide_chunk,
            vo_text=vo_text,
            current_visuals=current_visuals,
            feedback=feedback,
            image_pool_text="",  # force no image_pool/drive usage
            video_pool_filtered_text="",  # force no hvac-filtered usage
            drive_results_text="",  # force no drive fallback
            web_results_text=_safe_str(local_df.at[row_index, "web_results"]),
            segment_num=segment_num,
            drive=drive,
            llm=llm,
            visual_assignment_strategy=visual_assignment_strategy,
            video_pool_text="",  # force no hvac pool usage
            video_pool_other_channels_text=_safe_str(local_df.at[row_index, "video_pool_other_channels"]),
        )
        if not replacement_xml_inner:
            continue
        visual_blocks = re.findall(
            r"<visual>(.*?)</visual>",
            replacement_xml_inner,
            flags=re.IGNORECASE | re.DOTALL,
        )
        for block in visual_blocks:
            vid = _extract_tag(block, "visual_id").strip()
            rep_url = _extract_tag(block, "replacement_visual_url").strip()
            if vid and rep_url:
                replacement_by_visual_id[vid] = _normalize_replacement_visual_url(rep_url)

    # Build before/after visual asset maps to capture replacement URLs.
    # Only fallback-replacement URLs are appended; no primary-source replacement.

    updated_review = _append_replacements_to_review_text(final_review, replacement_by_visual_id)

    # Keep final_graphics_definition unchanged as requested.
    updates: Dict[str, str] = {
        "final_visuals_review": updated_review,
        "final_graphics_definition": final_graphics_definition_original,
    }

    # Keep regenerated search/candidate/pool outputs for fallback traceability.
    for col in [
        "search_queries",
        "web_results",
        "video_pool_other_channels",
    ]:
        if col in local_df.columns:
            updates[col] = _safe_str(local_df.at[row_index, col])

    if sheet is not None:
        merge_and_save_row_cells(
            sheet,
            worksheet_name,
            row_index,
            {
                "final_visuals_review": updates["final_visuals_review"],
                "final_graphics_definition": updates["final_graphics_definition"],
            },
        )

    return row_index, updates


def delete_replacement_visuals_from_final_visuals_review(sheet: Any) -> None:
    """
    Remove only appended 'Replacement visual' URLs from final_visuals_review.
    Keep PASS/FAIL verdicts, reasons, and feedback unchanged.
    """
    worksheet_name = "Slide Chunks"
    ws, df = get_sheet_data_and_df(sheet, worksheet_name)

    if "final_visuals_review" not in df.columns:
        print(f"ℹ️ 'final_visuals_review' column does not exist in '{worksheet_name}' worksheet")
        return

    changed_rows = 0
    for idx, val in df["final_visuals_review"].items():
        text_val = _safe_str(val)
        if not text_val or text_val.lower() == "nan":
            continue
        cleaned = re.sub(
            r"\s*\|\s*Replacement visual:\s*.*$",
            "",
            text_val,
            flags=re.IGNORECASE | re.MULTILINE,
        )
        if cleaned != text_val:
            df.at[idx, "final_visuals_review"] = cleaned
            changed_rows += 1

    save_to_sheet(ws, df)
    format_worksheet(ws)
    print(
        f"🧹 Removed replacement visual URLs from final_visuals_review for {changed_rows} row(s) in '{worksheet_name}'."
    )