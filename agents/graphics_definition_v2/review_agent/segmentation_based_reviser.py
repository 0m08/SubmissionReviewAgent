import re

from google.genai import types

from agents.graphics_definition_v2.slideshow_manifest.slideshow_manifest import generate_slideshow_manifest_for_row
from agents.graphics_definition_v2.review_agent.review_and_revise import (
    _build_segment_text_from_steps,
    build_asset_parts,
    build_final_graphics_definition,
    build_segment_visual_map,
    generate_search_queries_with_feedback,
    invoke_gemini_multimodal,
    parse_final_graphics_definition,
    parse_visual_steps,
)
from agents.graphics_definition_v2.image_graphics_agent.drive_search import (
    process_drive_search_segment,
)
from agents.graphics_definition_v2.image_graphics_agent.image_selection_from_all_images import (
    format_selected_images_for_segment,
    select_images_from_all_for_segment,
)
from agents.graphics_definition_v2.video_graphics_agent.youtube_video_search_from_queries import (
    process_video_search_segment,
)
from agents.graphics_definition_v2.video_graphics_agent.video_selection_from_all_videos import (
    format_selected_videos_for_segment,
    select_videos_from_all_for_segment,
)
from agents.graphics_definition_v2.aggregation_agent.aggregation_agent import (
    aggregate_graphics_definition_for_segment,
    expand_youtube_single_timestamp_clips_in_xml,
    parse_urls_from_results,
    parse_urls_from_video_pool,
    parse_urls_from_video_pool_filtered,
    process_video_frames_in_text_format,
)
from graphics_definition_v2_slideshow import detect_asset_type, safe_str
from services.sheets_service import get_sheet_data_and_df
from human_feedback_app.backend.config import LLM_DEFAULT
from human_feedback_app.backend.sheet_service import (
    apply_segmentation_revision_to_sheet,
    load_workbook,
)

SEGMENTATION_FEEDBACK_REVISION_PROMPT = """You are a Graphics Definition Segmentation Revision Agent specializing in the field of HVAC.

Context & Concepts:
1. What is a "Graphics Definition"?
   A Graphics Definition specifies which visual assets (images or video clips) must display on an e-learning slide and exactly which part of the voiceover narration (VO) triggers them.
2. Where did this come from?
   An automated AI agent previously parsed the slide content, segmented the narration text, and searched and assigned a visual asset to each segment.
3. Why are we here?
   A human reviewer is now reviewing these e-learning slides. They are providing natural-language feedback to correct and refine how the slide narration is divided and mapped to these visuals.

Your job is to understand the human reviewer's requested structural changes (such as merging split sentences so that they use a single visual, splitting a narration span, or copying/swapping assets), identify the affected visual IDs, and choose the correct predefined action(s) to align the graphics definition with the reviewer's intent.

Inputs:
These are the inputs for your segmentation revision planning:

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

These are the current voiceover segments and the respective assigned visuals for this slide:
<current_segmentation_map>
{current_segmentation_map}
</current_segmentation_map>

This is the human-provided segmentation feedback for this slide:
<human_segmentation_feedback>
{human_segmentation_feedback}
</human_segmentation_feedback>

Instructions:

1. Scope and Responsibility
   - Your task is to understand the human segmentation feedback and convert it into one or more structured operations.
   - If the human feedback begins with a `[Scope: VO Part X, ...]` indicator, it means the reviewer's feedback is specifically scoped and targeted to those voiceover parts (and their associated visuals). Focus your and changes on those parts.
   - Focus only on segmentation and visual assignment structure within this slide.
   - Segmentation feedback may ask to merge visual parts, reuse one existing visual for another voiceover part, split one visual into multiple voiceover parts, adjust which narration span a visual covers, or search for a new visual.
   - Use the full slide content and the sequence of voiceover segments to understand the reviewer's intent. Do not interpret a short phrase, pronoun, or clause in isolation.
   - Do not rewrite the slide narration. Preserve the original wording of voiceover text whenever you output a voiceover span.
   - Do not invent visual IDs or segment IDs that are not present in the current segmentation map.
   - Do not directly produce an edited graphics definition. Only produce the action plan that downstream code should apply.

2. Allowed Action Types
   You MUST choose from only the following action types based on the human feedback:

   - MERGE_VISUALS
     Use when two or more currently assigned visuals should be collapsed into one visual. This includes feedback like "join these parts", "merge part 1 and part 2", "use one visual for this whole sentence", "remove the extra split and keep the first visual", "combine these visuals and use this URL", etc.

   - REASSIGN_VISUAL
     Use when the structure should remain the same, but the asset from one visual should also be assigned to another visual. This includes feedback like "use voiceover part 1 visual for voiceover part 3", etc.

   - SPLIT_VISUAL
     Use when one currently assigned visual should be split into multiple visual steps covering different voiceover spans. This includes feedback like "split this visual into two parts", "show one visual for the first clause and another for the second clause", or "separate this sentence into two visuals", etc.

   - SET_VOICEOVER_SPAN
     Use when the asset should stay the same, but the exact narration text covered by a visual should change. This includes feedback like "this visual should cover only the first half", "extend this visual to cover the whole sentence", "part 2 should start from...", etc.

   - SEARCH_AND_ASSIGN
     Use when the reviewer asks the system to find or search for a new visual. Do not use this when they asked to reuse an assigned visual or pasted a URL.

   For MERGE_VISUALS and REASSIGN_VISUAL, set <source_visual_id> to the visual ID whose asset should be used (e.g. S1V1). If the reviewer pasted a URL to use, put that URL in <source_visual_id> instead. For SET_VOICEOVER_SPAN, leave <source_visual_id> empty.

3. Multiple Operations
   - The feedback may require more than one operation. If so, output multiple <operation> blocks in the order they should be applied.
   - If multiple independent voiceover parts need the same kind of fix, output one operation per independent fix.
   - If a single instruction affects multiple visuals together, output one operation containing all affected target visual IDs.
   - Prefer fewer, clearer operations when one operation fully captures the requested change.

4. Action Selection Rules
   - Prefer MERGE_VISUALS when the reviewer wants multiple existing visual parts to become one visual.
   - Prefer REASSIGN_VISUAL when the reviewer wants to reuse a visual asset but keep the visual steps separate.
   - Prefer SET_VOICEOVER_SPAN when the reviewer is only correcting the narration span covered by a visual.
   - Use SPLIT_VISUAL only when the reviewer clearly asks to create multiple visual parts from one existing visual.
   - Use SEARCH_AND_ASSIGN only when the reviewer asks to find or search for a new visual.
   - "Merge and use part 1 visual" → MERGE_VISUALS with <source_visual_id> set to that visual ID.
   - "Merge and use this URL: ..." → MERGE_VISUALS with <source_visual_id> set to the pasted URL.
   - "Find one visual for the whole sentence" → SEARCH_AND_ASSIGN with <search_intent> describing what to find.
   - If the feedback is not about segmentation, output an empty <operations> block.
   - Copy voiceover spans exactly from the slide content or current segmentation map.

Output:

Always provide your output strictly in the following format:

<output>

<evaluation_breakdown>

Use this section as a structured reasoning and scratchpad space for you to understand and address the human's feedback. Carefully think through the required segmentation structural changes and trace how they map to the current visuals. Provide the following sections:

1. Slide Understanding: State in your own words what the slide is about and what the voiceover segments are trying to convey. Then see the visuals that are currently assigned to the slide and briefly describe what you see in each of the visual.

2. Voiceover Segmentation & Assets Analysis
   - Analyze how the voiceover text is currently segmented and divided into steps. List each voiceover part and its assigned visual asset URL.
   - Identify which specific voiceover segments, visual IDs, and assets are referenced or affected by the human feedback.

2. Human Feedback Interpretation
   - Summarize what changes the human reviewer is requesting.
   - Identify whether the reviewer is asking to merge steps, split a step, reuse an asset/URL, change a voiceover span, or search for something new.
   - Identify any pasted URLs or referenced assets/visuals mentioned in the feedback.

3. Mapping to Predefined Action Types
   - Analyze the feedback and think through which all of the predefined action types are needed to address all the feedback requirements: MERGE_VISUALS, REASSIGN_VISUAL, SPLIT_VISUAL, SET_VOICEOVER_SPAN, or SEARCH_AND_ASSIGN.
   - Explain why the chosen action type(s) is/are correct for this feedback.

4. Targeting and Sourcing Strategy
   - Define which target visual ID(s) are the focus of the action.
   - If the action is MERGE_VISUALS or REASSIGN_VISUAL, identify which visual ID or pasted URL is the source asset (which asset should be kept or reused).
   - If the action is SET_VOICEOVER_SPAN, SPLIT_VISUAL, or MERGE_VISUALS, determine the exact voiceover text span(s) that the resulting visual step(s) should cover. Ensure these spans match the original voiceover wording exactly.

5. Additional Analysis: Note any additional observations, thoughts or analysis that can help you arrive at the correct output. It is ok for this section to be quite verbose, long and detailed as long as it helps you arrive at the correct output.

</evaluation_breakdown>

(Based on your above evaluation, provide the output strictly in following format)

<segmentation_plan>

<operations>

<operation>

<action_type>
MERGE_VISUALS|REASSIGN_VISUAL|SPLIT_VISUAL|SET_VOICEOVER_SPAN|SEARCH_AND_ASSIGN
</action_type>

<target_visual_ids>
(Affected visual IDs, one per line. For example:
- S1V1
- S1V2)
</target_visual_ids>

<source_visual_id>
(Visual ID whose asset to use, or a pasted URL. Leave empty for SET_VOICEOVER_SPAN or when the action alone is sufficient.)
</source_visual_id>

<new_voiceover_part>
(Exact voiceover span the resulting visual should cover. Preserve original wording.)
</new_voiceover_part>

<split_voiceover_parts>
(Fill only for SPLIT_VISUAL. Provide each new voiceover span on a separate "- " line. Otherwise leave empty.)
</split_voiceover_parts>

<search_intent>
(Fill only for SEARCH_AND_ASSIGN. Describe what kind of visual should be searched for. Otherwise leave empty.)
</search_intent>

<reason>
(One short sentence explaining why this operation is needed.)
</reason>

</operation>

Repeat one <operation> block per required operation so that all the human feedback is addressed.

</operations>

</segmentation_plan>

</output>

(Use this exact XML format while providing your output. Do not provide any additional text outside the <output> block.)
"""



_VISUAL_ID_RE = re.compile(r"^\s*S(\d+)V(\d+)\s*$", re.IGNORECASE)


def _is_url(value):
    """
    Check whether a string looks like an http(s) URL.

    :param value: String to test
    :return: True if the value starts with http:// or https://
    """

    return bool(re.match(r"^\s*https?://", value or "", re.IGNORECASE))


def _segment_num_of(visual_id):
    """
    Extract the segment number from a visual ID like 'S2V1'.

    :param visual_id: Visual ID string in the form S{seg}V{idx}
    :return: The segment number, or None if the ID does not match
    """

    match = _VISUAL_ID_RE.match(visual_id or "")
    return int(match.group(1)) if match else None


def build_segmentation_map_from_graphics(final_graphics_definition):
    """
    Parse final_graphics_definition into an ordered map of segment number to visual steps.

    :param final_graphics_definition: The final graphics definition text
    :return: Dict mapping segment number to a list of visual step dicts, each tagged with a stable visual_id (S{seg}V{idx})
    """

    segments = parse_final_graphics_definition(final_graphics_definition)
    steps_by_segment = {}
    for segment_num in sorted(segments.keys()):
        steps = parse_visual_steps(segments[segment_num])
        for idx, step in enumerate(steps, start=1):
            step["visual_id"] = f"S{segment_num}V{idx}"
        steps_by_segment[segment_num] = steps
    return steps_by_segment


def build_current_segmentation_map_text(steps_by_segment):
    """
    Render the segmentation map in the text format passed to the LLM.

    :param steps_by_segment: Dict mapping segment number to a list of visual step dicts
    :return: Formatted segmentation map string
    """

    lines = []
    vo_counter = 1
    for segment_num in sorted(steps_by_segment.keys()):
        steps = steps_by_segment[segment_num]
        lines.append(f"SEGMENT {segment_num}")
        lines.append("")
        if not steps:
            lines.append("No visuals assigned.")
            lines.append("")
            continue
        for step in steps:
            vo = (step.get("voiceover_part") or "").strip()
            asset = (step.get("asset") or "").strip()
            visual_id = step.get("visual_id", "")
            lines.append(f"Voiceover Part {vo_counter}")
            lines.append(f"When Voiceover: {vo}")
            lines.append(f"Assigned Visual ({visual_id}): {asset}")
            lines.append("")
            vo_counter += 1
    return "\n".join(lines).strip()


def build_segmentation_asset_parts(steps_by_segment, drive):
    """
    Build interleaved multimodal parts (text label + actual image/video) for every assigned visual.

    :param steps_by_segment: Dict mapping segment number to a list of visual step dicts
    :param drive: Google Drive instance
    :return: List of multimodal parts mirroring the segmentation map order
    """

    parts = []
    vo_counter = 1
    for segment_num in sorted(steps_by_segment.keys()):
        parts.append(types.Part(text=f"SEGMENT {segment_num}"))
        for step in steps_by_segment[segment_num]:
            visual_id = step.get("visual_id", "")
            vo = (step.get("voiceover_part") or "").strip()
            asset = (step.get("asset") or "").strip()
            parts.append(types.Part(text=f"Voiceover Part {vo_counter}\nWhen Voiceover: {vo}\nAssigned Visual ({visual_id}): {asset}"))
            if asset:
                parts.extend(build_asset_parts(visual_id, asset, drive))
            vo_counter += 1
    return parts


def parse_segmentation_operations(response_text):
    """
    Parse the <operations> block from the LLM output into structured operations.

    :param response_text: Raw LLM response text
    :return: List of operation dicts
    """

    text = response_text or ""
    operations = []
    for block in re.findall(r"<operation>(.*?)</operation>", text, re.DOTALL | re.IGNORECASE):
        def tag(name):
            """
            Extract the inner text of a named XML tag from the current operation block.

            :param name: Tag name to extract
            :return: Stripped tag content, or empty string if not found
            """

            match = re.search(rf"<{name}>\s*(.*?)\s*</{name}>", block, re.DOTALL | re.IGNORECASE)
            return match.group(1).strip() if match else ""

        def bullet_list(raw):
            """
            Parse a bullet/newline list into a list of cleaned strings.

            :param raw: Raw multi-line text
            :return: List of non-empty cleaned items
            """

            items = []
            for line in raw.splitlines():
                cleaned = line.strip().lstrip("-").strip()
                if cleaned:
                    items.append(cleaned)
            return items

        action_type = tag("action_type").upper()
        if not action_type:
            continue
        operations.append(
            {
                "action_type": action_type,
                "target_visual_ids": bullet_list(tag("target_visual_ids")),
                "source_visual_id": tag("source_visual_id"),
                "new_voiceover_part": tag("new_voiceover_part"),
                "split_voiceover_parts": bullet_list(tag("split_voiceover_parts")),
                "search_intent": tag("search_intent"),
                "reason": tag("reason"),
            }
        )
    return operations


def _find_step(steps_by_segment, visual_id):
    """
    Find a visual step by its visual ID.

    :param steps_by_segment: Dict mapping segment number to a list of visual step dicts
    :param visual_id: Visual ID to locate (e.g. S1V2)
    :return: The matching step dict, or None if not found
    """

    seg_num = _segment_num_of(visual_id)
    if seg_num is None:
        return None
    for step in steps_by_segment.get(seg_num, []):
        if (step.get("visual_id") or "").upper() == visual_id.strip().upper():
            return step
    return None


def _find_step_location(steps_by_segment, visual_id):
    """
    Find a visual step and its current location by visual ID.

    :param steps_by_segment: Dict mapping segment number to a list of visual step dicts
    :param visual_id: Visual ID to locate
    :return: Tuple of (segment number, step index, step dict), or (None, None, None)
    """

    wanted = (visual_id or "").strip().upper()
    if not wanted:
        return None, None, None
    for seg_num, steps in steps_by_segment.items():
        for idx, step in enumerate(steps):
            if (step.get("visual_id") or "").upper() == wanted:
                return seg_num, idx, step
    return None, None, None


def _resolve_source_asset(source_visual_id, original_asset_by_id):
    """
    Resolve a source asset from a visual ID or a pasted URL.

    :param source_visual_id: A visual ID (e.g. S1V1) or a pasted http(s) URL
    :param original_asset_by_id: Snapshot mapping of original visual_id to asset URL
    :return: The resolved asset URL, or empty string if unresolved
    """

    source = (source_visual_id or "").strip()
    if not source:
        return ""
    if _is_url(source):
        return source
    return original_asset_by_id.get(source.upper(), "")


def apply_segmentation_operations(final_graphics_definition, operations):
    """
    Apply parsed structural segmentation operations to a final graphics definition.

    :param final_graphics_definition: The current final graphics definition text
    :param operations: List of parsed operation dicts
    :return: Tuple of (updated_final_graphics_definition, event_log, affected_visual_ids, id_mapping)
    """

    steps_by_segment = build_segmentation_map_from_graphics(final_graphics_definition)
    if not steps_by_segment:
        return final_graphics_definition, ["No segments found in final_graphics_definition."], [], {}

    original_asset_by_id = {}
    original_ids_by_segment = {}
    for seg_num, steps in steps_by_segment.items():
        original_ids_by_segment[seg_num] = [(s.get("visual_id") or "").upper() for s in steps]
        for step in steps:
            original_asset_by_id[(step.get("visual_id") or "").upper()] = (step.get("asset") or "").strip()

    events = []
    affected = []

    def _flag(ids):
        """
        Record original visual IDs whose review state should be cleared.

        :param ids: List of visual IDs to flag as affected
        :return: None
        """

        for vid in ids:
            if vid and vid not in affected:
                affected.append(vid)

    for op in operations:
        action = (op.get("action_type") or "").upper()
        target_ids = [t for t in (op.get("target_visual_ids") or []) if t.strip()]
        source = (op.get("source_visual_id") or "").strip()
        new_vo = (op.get("new_voiceover_part") or "").strip()
        reason = (op.get("reason") or "").strip()

        if action == "SEARCH_AND_ASSIGN":
            events.append(f"SEARCH_AND_ASSIGN skipped (not implemented) | targets={','.join(target_ids)}")
            continue

        if not target_ids:
            events.append(f"{action} skipped (no target visuals) | reason={reason}")
            continue

        if action == "MERGE_VISUALS":
            target_set = {t.upper() for t in target_ids}
            retain_id = source.upper() if (source and not _is_url(source) and source.upper() in target_set) else target_ids[0].upper()

            retained = None
            retain_seg_num = None
            for s_num, steps in steps_by_segment.items():
                for s in steps:
                    if (s.get("visual_id") or "").upper() == retain_id:
                        retained = s
                        retain_seg_num = s_num
                        break
                if retained:
                    break

            if retained is None:
                for s_num, steps in steps_by_segment.items():
                    for s in steps:
                        if (s.get("visual_id") or "").upper() in target_set:
                            retained = s
                            retain_seg_num = s_num
                            retain_id = (s.get("visual_id") or "").upper()
                            break
                    if retained:
                        break

            if retained is None:
                events.append(f"MERGE_VISUALS skipped (no retain target found) | targets={','.join(target_ids)}")
                continue

            merged_vo = new_vo
            if not merged_vo:
                spans = []
                for s_num in sorted(steps_by_segment.keys()):
                    for s in steps_by_segment[s_num]:
                        if (s.get("visual_id") or "").upper() in target_set:
                            v_part = (s.get("voiceover_part") or "").strip()
                            if v_part and v_part not in spans:
                                spans.append(v_part)
                merged_vo = " ".join(spans).strip()
            retained["voiceover_part"] = merged_vo

            resolved_asset = _resolve_source_asset(source, original_asset_by_id)
            if resolved_asset:
                retained["asset"] = resolved_asset

            for s_num in list(steps_by_segment.keys()):
                steps_by_segment[s_num] = [
                    s for s in steps_by_segment[s_num]
                    if (s.get("visual_id") or "").upper() not in target_set
                    or (s.get("visual_id") or "").upper() == retain_id
                ]
            _flag(list(target_set))

            events.append(
                f"MERGE_VISUALS | retain={retain_id} | targets={','.join(target_ids)} | reason={reason}"
            )

        elif action == "REASSIGN_VISUAL":
            resolved_asset = _resolve_source_asset(source, original_asset_by_id)
            if not resolved_asset:
                events.append(f"REASSIGN_VISUAL skipped (no source asset) | targets={','.join(target_ids)}")
                continue
            applied = []
            for tid in target_ids:
                step = _find_step(steps_by_segment, tid)
                if step is not None:
                    step["asset"] = resolved_asset
                    applied.append(tid)
            _flag([t.upper() for t in applied])
            events.append(
                f"REASSIGN_VISUAL | targets={','.join(applied)} | source={source} | reason={reason}"
            )

        elif action == "SET_VOICEOVER_SPAN":
            if not new_vo:
                events.append(f"SET_VOICEOVER_SPAN skipped (no new span) | targets={','.join(target_ids)}")
                continue
            applied = []
            for tid in target_ids:
                step = _find_step(steps_by_segment, tid)
                if step is not None:
                    step["voiceover_part"] = new_vo
                    applied.append(tid)
            _flag([t.upper() for t in applied])
            events.append(
                f"SET_VOICEOVER_SPAN | targets={','.join(applied)} | reason={reason}"
            )

        elif action == "SPLIT_VISUAL":
            split_parts = [p for p in (op.get("split_voiceover_parts") or []) if p.strip()]
            if len(split_parts) < 2:
                events.append(f"SPLIT_VISUAL skipped (need >=2 spans) | targets={','.join(target_ids)}")
                continue
            tid = target_ids[0]
            seg_num = _segment_num_of(tid)
            steps = steps_by_segment.get(seg_num)
            if not steps:
                events.append(f"SPLIT_VISUAL skipped (segment not found) | target={tid}")
                continue
            idx = next((i for i, s in enumerate(steps) if (s.get("visual_id") or "").upper() == tid.upper()), None)
            if idx is None:
                events.append(f"SPLIT_VISUAL skipped (target not found) | target={tid}")
                continue
            base = steps[idx]
            new_steps = []
            for span in split_parts:
                new_steps.append(
                    {
                        "voiceover_part": span,
                        "visual_instruction": base.get("visual_instruction", ""),
                        "asset": base.get("asset", ""),
                        "selection_justification": base.get("selection_justification", ""),
                    }
                )
            steps_by_segment[seg_num] = steps[:idx] + new_steps + steps[idx + 1:]
            _flag(original_ids_by_segment.get(seg_num, []))
            events.append(
                f"SPLIT_VISUAL | SEGMENT {seg_num} | target={tid} | parts={len(split_parts)} | reason={reason}"
            )

        else:
            events.append(f"Unknown action '{action}' skipped | targets={','.join(target_ids)}")

    # Rebuild final_graphics_definition (reindex segments sequentially and visual IDs sequentially via order).
    non_empty_segments = {
        seg_num: steps
        for seg_num, steps in steps_by_segment.items()
        if steps
    }

    id_mapping = {}
    reindexed_segments = {}
    for new_seg_num, (old_seg_num, steps) in enumerate(sorted(non_empty_segments.items()), start=1):
        for idx, step in enumerate(steps, start=1):
            old_vid = step.get("visual_id")
            new_vid = f"S{new_seg_num}V{idx}"
            if old_vid:
                id_mapping[old_vid.upper()] = new_vid.upper()
            step["visual_id"] = new_vid
        reindexed_segments[new_seg_num] = steps

    new_segments_text = {}
    for seg_num, steps in reindexed_segments.items():
        new_segments_text[seg_num] = _build_segment_text_from_steps(seg_num, steps)

    updated = build_final_graphics_definition(new_segments_text)
    return updated, events, affected, id_mapping


def plan_segmentation_revision(course_name, target_audience, topic_name, subtopic_name, slide_id, slide_title, slide_chunk, final_graphics_definition, human_segmentation_feedback, drive, llm="gemini_3_flash_thinking"):
    """
    Run the multimodal segmentation planner over the slide and reviewer feedback.

    :param course_name: Course name
    :param target_audience: Target audience
    :param topic_name: Topic name
    :param subtopic_name: Subtopic name
    :param slide_id: Slide identifier
    :param slide_title: Slide title
    :param slide_chunk: Slide chunk text
    :param final_graphics_definition: Current final graphics definition text
    :param human_segmentation_feedback: Reviewer's segmentation feedback
    :param drive: Google Drive instance
    :param llm: LLM model name
    :return: Tuple of (operations, raw_response)
    """

    steps_by_segment = build_segmentation_map_from_graphics(final_graphics_definition)
    current_segmentation_map = build_current_segmentation_map_text(steps_by_segment)
    asset_parts = build_segmentation_asset_parts(steps_by_segment, drive)

    prompt = SEGMENTATION_FEEDBACK_REVISION_PROMPT.format(
        course_name=course_name,
        target_audience=target_audience,
        topic_name=topic_name,
        subtopic_name=subtopic_name,
        slide_id=slide_id,
        slide_title=slide_title,
        slide_chunk=slide_chunk,
        current_segmentation_map=current_segmentation_map,
        human_segmentation_feedback=human_segmentation_feedback,
    )

    print("\n" + "="*80)
    print("--- SEGMENTATION REVISER TEXT PROMPT ---")
    print(prompt)
    print("="*80 + "\n")

    print("--- SEGMENTATION REVISER MULTIMODAL PARTS (TEXT LABELS & MEDIA) ---")
    for i, part in enumerate(asset_parts, 1):
        text = getattr(part, "text", None) if part else None
        if text:
            print(f"[Part {i}] {text.strip()}")
        else:
            print(f"[Part {i}] <inline media (image or video)>")
    print("="*80 + "\n")

    parts = asset_parts + [types.Part(text=prompt)]
    response_text, _ = invoke_gemini_multimodal(parts, llm=llm)

    print("\n" + "="*80)
    print("--- SEGMENTATION REVISER RESPONSE ---")
    print(response_text)
    print("="*80 + "\n")

    operations = parse_segmentation_operations(response_text)
    return operations, response_text


def _extract_url_from_line(line):
    """
    Pull the asset URL out of a selector output line.

    :param line: A selector output line ('Title: X | URL: Y' or a bare URL)
    :return: The extracted URL, or empty string if none found
    """

    line = (line or "").strip()
    if not line:
        return ""
    if " URL: " in line:
        return line.split(" URL: ", 1)[1].strip()
    if "URL:" in line:
        return line.split("URL:", 1)[1].strip()
    match = re.search(r"https?://\S+", line)
    return match.group(0).strip() if match else ""


def _extract_first_asset_from_graphics_xml(graphics_definition_xml):
    """
    Extract the first selected asset URL from aggregation XML.

    :param graphics_definition_xml: Aggregation or replacement XML containing a selected asset URL
    :return: First asset URL, or empty string if none exists
    """

    xml = graphics_definition_xml or ""
    for tag_name in ("replacement_visual_url", "asset"):
        match = re.search(
            rf"<{tag_name}>(.*?)</{tag_name}>",
            xml,
            re.DOTALL | re.IGNORECASE,
        )
        if match:
            return match.group(1).strip()
    return ""


def _search_assign_one_target(target_visual_id, vo_text, original_asset, search_intent, segment_num, segments_map, course_name, target_audience, topic_name, subtopic_name, slide_title, slide_chunk, visual_assignment_strategy, drive, llm):
    """
    Run a fresh drive + HVAC-channel search for a single visual and return the best asset.

    :param target_visual_id: Visual ID being searched for
    :param vo_text: Voiceover text for the target visual
    :param original_asset: The current asset URL of the target visual
    :param search_intent: Reviewer's described search intent
    :param segment_num: Segment number of the target visual
    :param segments_map: Segment-to-visuals map for query generation context
    :param course_name: Course name
    :param target_audience: Target audience
    :param topic_name: Topic name
    :param subtopic_name: Subtopic name
    :param slide_title: Slide title
    :param slide_chunk: Slide chunk text
    :param visual_assignment_strategy: Visual assignment strategy
    :param drive: Google Drive instance
    :param llm: LLM model name
    :return: Tuple of (best_url or None, log_message)
    """

    feedback_str = (
        (search_intent or "Find and assign a more relevant visual for this voiceover part.")
        + f"\nFailing Visual: {target_visual_id}"
    )

    queries = generate_search_queries_with_feedback(
        course_name=course_name,
        target_audience=target_audience,
        topic_name=topic_name,
        subtopic_name=subtopic_name,
        slide_title=slide_title,
        slide_chunk=slide_chunk,
        vo_text=vo_text,
        feedback=feedback_str,
        llm=llm,
        segment_num=segment_num,
        segments_map=segments_map,
        drive=drive,
        visual_assignment_strategy=visual_assignment_strategy,
    )
    if not queries:
        return None, f"SEARCH {target_visual_id}: no queries generated"

    selected_image_items = []
    selected_video_items = []

    _, drive_out = process_drive_search_segment(segment_num, queries, drive)
    image_items = parse_urls_from_results(drive_out or "", segment_num)
    if image_items:
        urls = [i["url"] for i in image_items if i.get("url")]
        url_to_title = {i["url"]: i.get("title", "Untitled") for i in image_items if i.get("url")}
        selected = select_images_from_all_for_segment(
            vo_text=vo_text,
            slide_title=slide_title,
            slide_chunk=slide_chunk,
            image_urls=urls,
            url_to_title=url_to_title,
            course_name=course_name,
            topic_name=topic_name,
            subtopic_name=subtopic_name,
            drive=drive,
            llm=llm,
            feedback=feedback_str,
        )
        image_lines = format_selected_images_for_segment(selected) if selected else []
        selected_image_items = parse_urls_from_results(
            f"---SEGMENT_{segment_num}---\n" + "\n".join(image_lines),
            segment_num,
        )

    _, vpool_out = process_video_search_segment(segment_num, queries, drive)
    video_pool = parse_urls_from_video_pool(vpool_out or "", segment_num)
    if video_pool:
        selected = select_videos_from_all_for_segment(
            vo_text=vo_text,
            slide_title=slide_title,
            slide_chunk=slide_chunk,
            video_urls_pool=video_pool,
            video_items_other_channels=[],
            course_name=course_name,
            topic_name=topic_name,
            subtopic_name=subtopic_name,
            drive=drive,
            llm=llm,
            feedback=feedback_str,
        )
        video_lines = format_selected_videos_for_segment(selected, video_pool, []) if selected else []
        selected_video_items = parse_urls_from_video_pool_filtered(
            f"---SEGMENT_{segment_num}---\n" + "\n".join(video_lines),
            segment_num,
        )

    if not selected_image_items and not selected_video_items:
        return None, f"SEARCH {target_visual_id}: no selected image or video candidates"

    graphics_definition_xml, _ = aggregate_graphics_definition_for_segment(
        vo_text=vo_text,
        slide_title=slide_title,
        slide_chunk=slide_chunk,
        image_items=selected_image_items,
        video_items_filtered=selected_video_items,
        course_name=course_name,
        topic_name=topic_name,
        subtopic_name=subtopic_name,
        storyboard="",
        layout_plan="",
        drive=drive,
        llm=llm,
        feedback=feedback_str,
        target_audience=target_audience,
        failed_visuals=[{"visual_id": target_visual_id, "asset_url": original_asset}] if original_asset else None,
        visual_assignment_strategy=visual_assignment_strategy,
    )
    if graphics_definition_xml:
        graphics_definition_xml = expand_youtube_single_timestamp_clips_in_xml(graphics_definition_xml)
        best = _extract_first_asset_from_graphics_xml(graphics_definition_xml)
        if best:
            image_count = len(selected_image_items)
            video_count = len(selected_video_items)
            chosen_type = detect_asset_type(best)
            return best, f"SEARCH_AND_ASSIGN | {target_visual_id} | selected {chosen_type} from {image_count} image(s), {video_count} video(s)"

    return None, f"SEARCH {target_visual_id}: no suitable visual found"


def apply_search_assign_operations(final_graphics_definition, search_operations, course_name, target_audience, topic_name, subtopic_name, slide_title, slide_chunk, voiceover_text, visual_assignment_strategy, drive, llm):
    """
    Apply SEARCH_AND_ASSIGN operations to an already-restructured graphics definition.

    :param final_graphics_definition: The restructured final graphics definition text
    :param search_operations: List of SEARCH_AND_ASSIGN operation dicts
    :param course_name: Course name
    :param target_audience: Target audience
    :param topic_name: Topic name
    :param subtopic_name: Subtopic name
    :param slide_title: Slide title
    :param slide_chunk: Slide chunk text
    :param voiceover_text: Full voiceover segment text for the slide
    :param visual_assignment_strategy: Visual assignment strategy
    :param drive: Google Drive instance
    :param llm: LLM model name
    :return: Tuple of (new_final_graphics_definition, events, search_tracking, search_affected_visual_ids)
    """

    steps_by_segment = build_segmentation_map_from_graphics(final_graphics_definition)
    if not steps_by_segment:
        return final_graphics_definition, ["No segments found for search."], {}, []

    segments_map = build_segment_visual_map(
        voiceover_text, final_graphics_definition, visual_assignment_strategy, slide_chunk
    )

    events = []
    tracking = {}
    search_affected = []
    any_video_assigned = False

    for op in search_operations:
        search_intent = (op.get("search_intent") or "").strip()
        targets = [t for t in (op.get("target_visual_ids") or []) if t.strip()]
        new_vo = (op.get("new_voiceover_part") or "").strip()

        if len(targets) > 1 and new_vo:
            located_targets = []
            for tid in targets:
                seg_num, idx, step = _find_step_location(steps_by_segment, tid)
                if seg_num is None or step is None:
                    events.append(f"SEARCH merge target skipped (target {tid} not found in restructured slide)")
                    continue
                located_targets.append((tid, seg_num, idx, step))

            if not located_targets:
                events.append("SEARCH merge skipped (no targets found in restructured slide)")
                continue

            retained_id, retained_seg_num, _, retained_step = located_targets[0]
            original_asset = (retained_step.get("asset") or "").strip()
            retained_step["voiceover_part"] = new_vo

            removed_ids = {
                (tid or "").strip().upper()
                for tid, _, _, _ in located_targets[1:]
                if tid
            }
            for seg_num, steps in list(steps_by_segment.items()):
                steps_by_segment[seg_num] = [
                    step for step in steps
                    if (step.get("visual_id") or "").upper() not in removed_ids
                ]
            for removed_id in sorted(removed_ids):
                if removed_id not in search_affected:
                    search_affected.append(removed_id)

            best_url, log = _search_assign_one_target(
                target_visual_id=retained_id,
                vo_text=new_vo,
                original_asset=original_asset,
                search_intent=search_intent,
                segment_num=retained_seg_num,
                segments_map=segments_map,
                course_name=course_name,
                target_audience=target_audience,
                topic_name=topic_name,
                subtopic_name=subtopic_name,
                slide_title=slide_title,
                slide_chunk=slide_chunk,
                visual_assignment_strategy=visual_assignment_strategy,
                drive=drive,
                llm=llm,
            )
            events.append(log)
            events.append(
                "SEARCH_AND_ASSIGN merged targets into one visual step | "
                + f"retained={retained_id} removed={','.join(sorted(removed_ids))}"
            )
            if best_url:
                retained_step["asset"] = best_url
                tracking[retained_id] = {"original": original_asset, "after_revision": best_url}
                if detect_asset_type(best_url) == "video":
                    any_video_assigned = True
            continue

        for tid in targets:
            seg_num = _segment_num_of(tid)
            step = _find_step(steps_by_segment, tid)
            if seg_num is None or step is None:
                events.append(f"SEARCH skipped (target {tid} not found in restructured slide)")
                continue

            vo_text = (step.get("voiceover_part") or "").strip()
            original_asset = (step.get("asset") or "").strip()

            best_url, log = _search_assign_one_target(
                target_visual_id=tid,
                vo_text=vo_text,
                original_asset=original_asset,
                search_intent=search_intent,
                segment_num=seg_num,
                segments_map=segments_map,
                course_name=course_name,
                target_audience=target_audience,
                topic_name=topic_name,
                subtopic_name=subtopic_name,
                slide_title=slide_title,
                slide_chunk=slide_chunk,
                visual_assignment_strategy=visual_assignment_strategy,
                drive=drive,
                llm=llm,
            )
            events.append(log)
            if not best_url:
                continue

            step["asset"] = best_url
            tracking[tid] = {"original": original_asset, "after_revision": best_url}
            if detect_asset_type(best_url) == "video":
                any_video_assigned = True

    # Rebuild final_graphics_definition (reindex segments sequentially and visual IDs sequentially via order).
    non_empty_segments = {
        seg_num: steps
        for seg_num, steps in steps_by_segment.items()
        if steps
    }

    reindexed_segments = {}
    for new_seg_num, (old_seg_num, steps) in enumerate(sorted(non_empty_segments.items()), start=1):
        for idx, step in enumerate(steps, start=1):
            step["visual_id"] = f"S{new_seg_num}V{idx}"
        reindexed_segments[new_seg_num] = steps

    new_segments_text = {}
    for seg_num, steps in reindexed_segments.items():
        new_segments_text[seg_num] = _build_segment_text_from_steps(seg_num, steps)

    new_fgd = build_final_graphics_definition(new_segments_text)
    if any_video_assigned:
        new_fgd = process_video_frames_in_text_format(new_fgd, drive)
    return new_fgd, events, tracking, search_affected


def run_segmentation_revision_for_row(session, row_index, feedback, llm=None):
    """
    Plan and apply a full segmentation revision for one sheet row, then persist.

    :param session: Active user session
    :param row_index: Row index in the dataframe
    :param feedback: Reviewer's raw segmentation feedback
    :param llm: Optional LLM model name (defaults to LLM_DEFAULT)
    :return: Dict with operations, events, raw_plan, and search_tracking
    """

    llm = llm or LLM_DEFAULT
    _, df, _ = load_workbook(session)
    row = df.loc[row_index]

    _, course_info_df = get_sheet_data_and_df(session.sheet, "Course info")
    course_name = safe_str(course_info_df.iloc[0].get("Course Name", "")).strip()
    target_audience = safe_str(course_info_df.iloc[0].get("Target Audience & Industry", "")).strip()

    slide_title = safe_str(row.get("Slide Chunk Title", "")).strip()
    slide_chunk = safe_str(row.get("Slide Chunk", "")).strip()
    topic_name = safe_str(row.get("Topic", "")).strip()
    subtopic_name = safe_str(row.get("Subtopic", "")).strip()
    final_graphics_definition = safe_str(row.get("final_graphics_definition", ""))
    voiceover_text = safe_str(row.get("voiceover_segment", ""))
    visual_assignment_strategy = safe_str(row.get("Visual Assignment Strategy", "")).strip()
    if not visual_assignment_strategy or visual_assignment_strategy == "nan":
        visual_assignment_strategy = "Flexible, let the agent decide"
    slide_id = f"SLIDE_{row_index + 1}"

    operations, raw_plan = plan_segmentation_revision(
        course_name=course_name,
        target_audience=target_audience,
        topic_name=topic_name,
        subtopic_name=subtopic_name,
        slide_id=slide_id,
        slide_title=slide_title,
        slide_chunk=slide_chunk,
        final_graphics_definition=final_graphics_definition,
        human_segmentation_feedback=feedback,
        drive=session.drive,
        llm=llm,
    )

    if not operations:
        apply_segmentation_revision_to_sheet(
            session,
            row_index=row_index,
            feedback=feedback,
            raw_plan=raw_plan,
            updated_final_graphics_definition=final_graphics_definition,
            affected_visual_ids=[],
            events=["No segmentation operations planned."],
        )
        return {"operations": [], "events": [], "raw_plan": raw_plan}

    # Phase 1: structural ops.
    updated_fgd, events, affected_visual_ids, id_mapping = apply_segmentation_operations(
        final_graphics_definition, operations
    )

    # Phase 2: fresh search + assign (drive + HVAC channel), run on the restructured slide. Search targets record before/after tracking instead of being cleared.
    search_ops = [
        op for op in operations
        if (op.get("action_type") or "").upper() == "SEARCH_AND_ASSIGN"
    ]
    search_tracking = {}
    search_affected_visual_ids = []
    if search_ops:
        updated_fgd, search_events, search_tracking, search_affected_visual_ids = apply_search_assign_operations(
            updated_fgd,
            search_ops,
            course_name=course_name,
            target_audience=target_audience,
            topic_name=topic_name,
            subtopic_name=subtopic_name,
            slide_title=slide_title,
            slide_chunk=slide_chunk,
            voiceover_text=voiceover_text,
            visual_assignment_strategy=visual_assignment_strategy,
            drive=session.drive,
            llm=llm,
        )
        events.extend(search_events)
        for vid in search_affected_visual_ids:
            if vid not in affected_visual_ids:
                affected_visual_ids.append(vid)

    updated_manifest = None
    if "slideshow_manifest" in df.columns:
        print(f"[segmentation_reviser] Regenerating slideshow manifest for row {row_index + 1}...")
        try:
            slide_type = safe_str(row.get("Slide Type", "")).strip()
            layout_plan = safe_str(row.get("layout_plan", "")).strip()
            storyboard_planning = safe_str(row.get("storyboard_planning", "")).strip()

            updated_manifest, _ = generate_slideshow_manifest_for_row(
                course_name=course_name,
                topic_name=topic_name,
                subtopic_name=subtopic_name,
                slide_type=slide_type,
                slide_title=slide_title,
                slide_content=slide_chunk,
                layout_plan=layout_plan,
                storyboard_planning=storyboard_planning,
                final_graphics_definition=updated_fgd,
                drive=session.drive,
                llm=llm,
            )
            print("[segmentation_reviser] Successfully regenerated slideshow manifest.")
        except Exception as e:
            print(f"[segmentation_reviser] ERROR: Failed to regenerate slideshow manifest: {e}")
            updated_manifest = f"ERROR: Failed to regenerate manifest after segmentation revision: {str(e)}"

    apply_segmentation_revision_to_sheet(
        session,
        row_index=row_index,
        feedback=feedback,
        raw_plan=raw_plan,
        updated_final_graphics_definition=updated_fgd,
        affected_visual_ids=affected_visual_ids,
        search_tracking=search_tracking,
        events=events,
        updated_slideshow_manifest=updated_manifest,
        id_mapping=id_mapping,
    )
    return {
        "operations": operations,
        "events": events,
        "raw_plan": raw_plan,
        "search_tracking": search_tracking,
    }
