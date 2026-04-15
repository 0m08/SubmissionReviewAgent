import html
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st
from google.genai import types
from langsmith import traceable

from agents.graphics_definition_v2.aggregation_agent.aggregation_agent import (
    normalize_youtube_timestamp_urls,
    process_video_frames_in_text_format,
)
from agents.graphics_definition_v2.review_agent.review_and_revise import (
    _safe_str,
    add_snapshot_label_to_drive_links,
    build_asset_parts,
    build_segment_visual_map,
    get_drive_instance,
    invoke_gemini_multimodal,
    update_final_graphics_definition_with_replacements,
)
from services.sheets_service import (
    format_worksheet,
    clear_worksheet,
    get_sheet_data_and_df,
    merge_and_save_columns,
    save_to_sheet,
)
from services.smart_progress_bar import SmartProgressBar


DECIDE_FINAL_VISUALS_PROMPT = """You are a Graphics Definition Review Agent specializing in the field of HVAC.

Your Task:
For the given visual slot, compare the two assigned visual options and decide which one is the best visual to use for that specific part of narration.

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

<decision_context>
Visual ID under review: {visual_id}
Segment number: {segment_num}

Voiceover segment text:
"{segment_vo_text}"

Voiceover part assigned to this visual:
"{visual_part_for_this_visual}"

Option A URL:
{option_a_url}

Option B URL:
{option_b_url}
</decision_context>

You will receive two multimodal visual options for this same visual slot:
- Option A (current assigned visual)
- Option B (replacement visual)

Instructions:
Follow the below evaluation rules very strictly:

1) Scope
   - Evaluate only the provided visual ID.
   - Compare only the two provided options for this exact voiceover moment.
   - Use slide context to interpret meaning correctly (split sentences, pronouns, continuation phrases).
   - Do not judge based only on broad topic relevance.

2) What makes one option better
   - The better option must more clearly and directly support what the narration is communicating at that moment.
   - It must be more instructionally useful for the learner (clear, interpretable, specific).
   - It must not introduce contradiction, distraction, or unrelated instructional focus.
   - It must be usable (loadable and visually understandable).

3) Decision logic
   - Choose exactly one option: A or B.
   - If one option is clearly more accurate/specific, choose it.
   - If both are weak, choose the less incorrect option for this specific narration.

4) Failure awareness
   - Be strict and critical.
   - If Option A does not clearly support the assigned voiceover part but Option B does, choose B.
   - If Option B is generic, misleading, lower quality, or less relevant than A, choose A.

Output Format:
Always provide your output strictly in the following format:

<evaluation_breakdown>

Use this section as a structured reasoning and scratchpad space for comparison.

- Segment Understanding: Briefly explain what this narration segment is trying to communicate.
- Option A Scan: Briefly describe what is visibly shown in Option A.
- Option B Scan: Briefly describe what is visibly shown in Option B.
- Comparative Analysis: Compare A vs B specifically for this narration moment and explain which one aligns better and why.
- Additional Analysis: Any additional observations required to justify the final choice.

</evaluation_breakdown>

<decision>

<visual_id>
{visual_id}
</visual_id>

<chosen_option>
A|B
</chosen_option>

<chosen_visual_url>
(Provide exact URL of the chosen option)
</chosen_visual_url>

<reason>
(Provide a concise but clear reason why this option is better for this narration moment.)
</reason>

</decision>
"""


def get_decide_final_visuals_prompt(course_name, target_audience, topic_name, subtopic_name, slide_id, slide_title, slide_chunk, visual_id, segment_num, segment_vo_text, visual_part_for_this_visual, option_a_url, option_b_url):
    """
    Build final-visual decision prompt.

    :param course_name: Course name
    :param target_audience: Target audience
    :param topic_name: Topic name
    :param subtopic_name: Subtopic name
    :param slide_id: Slide ID
    :param slide_title: Slide title
    :param slide_chunk: Full slide content
    :param visual_id: Visual ID under decision
    :param segment_num: Segment number
    :param segment_vo_text: Segment voiceover text
    :param visual_part_for_this_visual: Voiceover part tied to visual
    :param option_a_url: Current assigned visual URL
    :param option_b_url: Replacement visual URL
    :return: Formatted prompt text
    """
    
    return DECIDE_FINAL_VISUALS_PROMPT.format(
        course_name=course_name,
        target_audience=target_audience,
        topic_name=topic_name,
        subtopic_name=subtopic_name,
        slide_id=slide_id,
        slide_title=slide_title,
        slide_chunk=slide_chunk,
        visual_id=visual_id,
        segment_num=segment_num,
        segment_vo_text=segment_vo_text or "(none)",
        visual_part_for_this_visual=visual_part_for_this_visual or "(none)",
        option_a_url=option_a_url or "(none)",
        option_b_url=option_b_url or "(none)",
    )


def _extract_tag(text, tag):
    """
    Extract XML tag contents from text.

    :param text: Source text
    :param tag: XML tag name
    :return: Tag content or empty string
    """
    
    if not text:
        return ""
    match = re.search(
        rf"<{tag}>(.*?)</{tag}>",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return match.group(1).strip() if match else ""


def _parse_decision_response(text):
    """
    Parse decision response and map choice labels.

    :param text: Model response text
    :return: Tuple of (chosen_option, reason)
    """
    
    block = text or ""
    m = re.search(r"<decision>.*?</decision>", block, flags=re.IGNORECASE | re.DOTALL)
    if m:
        block = m.group(0)
    opt = (_extract_tag(block, "chosen_option") or "").strip().upper()
    justification = (_extract_tag(block, "reason") or _extract_tag(block, "justification") or "").strip()
    # Map legacy wording if model returns it
    if opt == "PRIMARY":
        opt = "FIRST"
    elif opt == "FALLBACK":
        opt = "SECOND"
    elif opt == "A":
        opt = "FIRST"
    elif opt == "B":
        opt = "SECOND"
    return opt, justification


def _parse_fallback_candidates_from_review(review_text):
    """
    Parse failed visuals and replacement URLs from final_visuals_review.

    :param review_text: final_visuals_review text
    :return: List of candidate dicts
    """
    
    out = []
    current_segment = None
    for raw_line in review_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        seg_m = re.match(r"---SEGMENT_(\d+)---", line, re.IGNORECASE)
        if seg_m:
            current_segment = int(seg_m.group(1))
            continue
        fail_m = re.match(r"^(S\d+V\d+):\s*FAIL\b(.*)$", line, re.IGNORECASE)
        if not fail_m or current_segment is None:
            continue
        visual_id = fail_m.group(1).strip()
        rest = fail_m.group(2) or ""
        rep_m = re.search(r"\|\s*Replacement visual:\s*(.+)$", rest, re.IGNORECASE)
        if not rep_m:
            continue
        replacement_url = rep_m.group(1).strip()
        if not replacement_url or replacement_url.lower() == "nan":
            continue
        out.append(
            {
                "segment_num": current_segment,
                "visual_id": visual_id,
                "replacement_url": replacement_url,
                "review_line": line,
            }
        )
    return out


def _segment_num_from_visual_id(visual_id):
    """
    Extract segment number from visual ID.

    :param visual_id: Visual ID like S1V2
    :return: Segment number or None
    """
    
    m = re.match(r"^S(\d+)V\d+$", visual_id.strip(), re.IGNORECASE)
    return int(m.group(1)) if m else None


def _escape_xml_text(s):
    """
    Escape XML text content.

    :param s: Input string
    :return: XML-escaped string
    """
    
    return html.escape(s or "", quote=False)


def _build_replacement_xml_from_step(visual_id, step, new_asset_url):
    """
    Build replacement XML block for one visual.

    :param visual_id: Visual ID
    :param step: Existing parsed visual step dict
    :param new_asset_url: URL to place as replacement
    :return: replacement_visuals XML string
    """
    
    vo = _escape_xml_text(str(step.get("voiceover_part", "") or ""))
    vi = _escape_xml_text(str(step.get("visual_instruction", "") or ""))
    sj = _escape_xml_text(str(step.get("selection_justification", "") or ""))
    safe_url = new_asset_url.replace("&", "&amp;")
    return (
        "<replacement_visuals>\n"
        "<visual>\n"
        f"<visual_id>{_escape_xml_text(visual_id)}</visual_id>\n"
        f"<voiceover_part>{vo}</voiceover_part>\n"
        f"<replacement_visual_url>{safe_url}</replacement_visual_url>\n"
        f"<visual_instruction>{vi}</visual_instruction>\n"
        f"<selection_justification>{sj}</selection_justification>\n"
        "</visual>\n"
        "</replacement_visuals>"
    )


def post_process_final_graphics_definition_after_edit(text, drive):
    """
    Match review-revise final row post-processing.

    :param text: final_graphics_definition text
    :param drive: Google Drive instance
    :return: Post-processed final_graphics_definition text
    """
    
    if not text or not str(text).strip():
        return text
    t = normalize_youtube_timestamp_urls(text)
    t = process_video_frames_in_text_format(t, drive)
    if drive:
        t = add_snapshot_label_to_drive_links(t, drive)
    return t


@traceable(
    metadata={
        "agent_name": "graphics_definition_v2",
        "step_name": "Decide Final Visuals",
        "function_name": "process_decide_final_visuals_row",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous"),
    }
)
def process_decide_final_visuals_row(row_index, row, course_name, target_audience, llm, drive):
    """
    Process one row for final visual decision.

    :param row_index: Row index
    :param row: Row data
    :param course_name: Course name
    :param target_audience: Target audience
    :param llm: LLM model name
    :param drive: Google Drive instance
    :return: Tuple of (row_index, updates_dict)
    """
    
    slide_chunk = _safe_str(row.get("Slide Chunk", ""))
    voiceover_text = _safe_str(row.get("voiceover_segment", ""))
    topic_name = _safe_str(row.get("Topic", ""))
    subtopic_name = _safe_str(row.get("Subtopic", ""))
    slide_title = _safe_str(row.get("Slide Chunk Title", "")) or "Untitled"
    slide_id = _safe_str(row.get("Slide ID", "")) or f"SLIDE {row_index + 1}"
    final_review = _safe_str(row.get("final_visuals_review", ""))
    visual_assignment_strategy = _safe_str(
        row.get("Visual Assignment Strategy", "Flexible, let the agent decide")
    )
    if not visual_assignment_strategy or visual_assignment_strategy == "nan":
        visual_assignment_strategy = "Flexible, let the agent decide"

    fgd = _safe_str(row.get("final_graphics_definition", ""))
    if not fgd or fgd.lower() == "nan":
        return row_index, {}

    candidates = _parse_fallback_candidates_from_review(final_review)
    if not candidates:
        note = (
            "No Failed visuals"
        )
        return row_index, {
            "decision_of_final_visual": note,
            "final_graphics_definition": fgd,
        }

    log_lines = []
    decision_entries = []
    current_fgd = fgd

    for cand in candidates:
        segment_num = cand["segment_num"]
        visual_id = cand["visual_id"]
        fallback_url = cand["replacement_url"]

        if _segment_num_from_visual_id(visual_id) != segment_num:
            selected_url = ""
            log_lines.append(
                f"{visual_id}: Visual 1 (skipped) | Justification: Segment id mismatch vs visual id; kept current assignment."
            )
            decision_entries.append(
                {
                    "segment_num": str(segment_num),
                    "visual_id": visual_id,
                    "original_visual": selected_url,
                    "replacement_visual": fallback_url,
                    "selection": selected_url,
                }
            )
            continue

        segments_map = build_segment_visual_map(
            voiceover_text=voiceover_text,
            final_graphics_definition=current_fgd,
            visual_assignment_strategy=visual_assignment_strategy,
            slide_chunk=slide_chunk,
        )
        segment = segments_map.get(segment_num) or {}
        steps = segment.get("visual_steps") or []
        step = next((s for s in steps if (s.get("visual_id") or "").strip() == visual_id), None)
        if not step:
            selected_url = ""
            log_lines.append(
                f"{visual_id}: Visual 1 (skipped) | Justification: Visual not found in current final_graphics_definition."
            )
            decision_entries.append(
                {
                    "segment_num": str(segment_num),
                    "visual_id": visual_id,
                    "original_visual": selected_url,
                    "replacement_visual": fallback_url,
                    "selection": selected_url,
                }
            )
            continue

        primary_url = _safe_str(step.get("asset", "")).strip()
        if not primary_url:
            log_lines.append(
                f"{visual_id}: Visual 2 | Justification: No asset URL in definition for this slot; using the alternate visual."
            )
            decision_entries.append(
                {
                    "segment_num": str(segment_num),
                    "visual_id": visual_id,
                    "original_visual": primary_url,
                    "replacement_visual": fallback_url,
                    "selection": fallback_url,
                }
            )
            xml_fb = _build_replacement_xml_from_step(visual_id, step, fallback_url)
            current_fgd = update_final_graphics_definition_with_replacements(
                current_fgd, segment_num, xml_fb
            )
            continue

        if primary_url == fallback_url:
            log_lines.append(
                f"{visual_id}: Visual 1 | Justification: Both candidate URLs are identical; kept current assignment."
            )
            decision_entries.append(
                {
                    "segment_num": str(segment_num),
                    "visual_id": visual_id,
                    "original_visual": primary_url,
                    "replacement_visual": fallback_url,
                    "selection": primary_url,
                }
            )
            continue

        prompt = get_decide_final_visuals_prompt(
            course_name=course_name,
            target_audience=target_audience,
            topic_name=topic_name,
            subtopic_name=subtopic_name,
            slide_id=slide_id,
            slide_title=slide_title,
            slide_chunk=slide_chunk,
            visual_id=visual_id,
            segment_num=segment_num,
            segment_vo_text=_safe_str(segment.get("vo_text", "")),
            visual_part_for_this_visual=_safe_str(step.get("voiceover_part", "")),
            option_a_url=primary_url,
            option_b_url=fallback_url,
        )
        # print(f"\n{'='*80}")
        # print(f"📝 FORMATTED FINAL VISUAL DECISION PROMPT ({slide_id} {visual_id}):")
        # print(f"{'='*80}")
        # print(prompt)
        # print(f"{'='*80}\n")

        primary_parts = build_asset_parts(f"{visual_id} (visual 1 of 2)", primary_url, drive)
        fallback_parts = build_asset_parts(f"{visual_id} (visual 2 of 2)", fallback_url, drive)
        parts = primary_parts + fallback_parts + [types.Part(text=prompt)]
        print(f"Multimodal parts to be sent for {visual_id}:")
        for idx, part in enumerate(parts, 1):
            if hasattr(part, "text") and part.text:
                print(f"  Part {idx} (text): {part.text[:200]}{'...' if len(part.text) > 200 else ''}")
            elif hasattr(part, "inline_data"):
                print(f"  Part {idx} (image/video data)")
            else:
                print(f"  Part {idx} (other)")

        response_text, _ = invoke_gemini_multimodal(parts, llm=llm, conversation_history=None)
        print(f"\n📤 Final Visual Decision Response ({slide_id} {visual_id}):\n")
        print(response_text)
        print(f"\n{'='*100}\n")
        chosen, justification = _parse_decision_response(response_text)

        if chosen not in ("FIRST", "SECOND"):
            chosen = "FIRST"
            justification = (
                (justification + " " if justification else "")
                + "Model output did not specify FIRST|SECOND; defaulted to FIRST (current assignment)."
            ).strip()

        if chosen == "FIRST":
            log_lines.append(f"{visual_id}: Visual 1 | Justification: {justification}")
            decision_entries.append(
                {
                    "segment_num": str(segment_num),
                    "visual_id": visual_id,
                    "original_visual": primary_url,
                    "replacement_visual": fallback_url,
                    "selection": primary_url,
                }
            )
            continue

        log_lines.append(f"{visual_id}: Visual 2 | Justification: {justification}")
        decision_entries.append(
            {
                "segment_num": str(segment_num),
                "visual_id": visual_id,
                "original_visual": primary_url,
                "replacement_visual": fallback_url,
                "selection": fallback_url,
            }
        )
        xml_fb = _build_replacement_xml_from_step(visual_id, step, fallback_url)
        current_fgd = update_final_graphics_definition_with_replacements(
            current_fgd, segment_num, xml_fb
        )

    # Post-process entire definition once after all replacements
    processed_fgd = post_process_final_graphics_definition_after_edit(current_fgd, drive)

    decision_tracking_body = _format_decision_tracking_output(decision_entries)

    return row_index, {
        "final_graphics_definition": processed_fgd,
        "decision_of_final_visual": decision_tracking_body,
    }


def _format_decisions_sheet_output(candidates, log_lines):
    """
    Format debug decision logs by segment.

    :param candidates: Candidate list with segment mapping
    :param log_lines: Per-visual decision log lines
    :return: Formatted debug text
    """
    
    if not log_lines:
        return ""
    if len(candidates) != len(log_lines):
        return "\n".join(log_lines).strip()

    by_seg = {}
    for cand, ln in zip(candidates, log_lines):
        by_seg.setdefault(cand["segment_num"], []).append(ln)

    parts = []
    for seg in sorted(by_seg.keys()):
        parts.append(f"---SEGMENT_{seg}---")
        parts.append("")
        parts.extend(by_seg[seg])
        parts.append("")
    return "\n".join(parts).strip()


def _format_decision_tracking_output(entries):
    """
    Format tracking output for decision_of_final_visual column.

    :param entries: Decision entry dicts
    :return: Formatted tracking text
    """
    if not entries:
        return ""
    by_seg = {}
    for entry in entries:
        try:
            seg = int(entry.get("segment_num") or 0)
        except ValueError:
            seg = 0
        by_seg.setdefault(seg, []).append(entry)

    parts = []
    for seg in sorted(by_seg.keys()):
        parts.append(f"---SEGMENT_{seg}---")
        parts.append("")
        for entry in by_seg[seg]:
            vid = _safe_str(entry.get("visual_id", ""))
            parts.append(f"{vid}")
            parts.append(f"Original Visual: {_safe_str(entry.get('original_visual', ''))}")
            parts.append(f"Replacement Visual: {_safe_str(entry.get('replacement_visual', ''))}")
            parts.append(f"Selection: {_safe_str(entry.get('selection', ''))}")
            parts.append("")
    return "\n".join(parts).strip()


def _parse_decision_tracking_output(text):
    """
    Parse decision tracking text into structured entries.

    :param text: decision_of_final_visual text
    :return: Parsed decision entries
    """
    entries = []
    current_segment = None
    current_entry = {}

    def _flush_current():
        nonlocal current_entry
        if current_segment is None:
            return
        if current_entry.get("visual_id"):
            current_entry["segment_num"] = str(current_segment)
            entries.append(current_entry)
        current_entry = {}

    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        seg_match = re.match(r"---SEGMENT_(\d+)---", line, re.IGNORECASE)
        if seg_match:
            _flush_current()
            current_segment = int(seg_match.group(1))
            continue

        vid_match = re.match(r"^(S\d+V\d+):?$", line, re.IGNORECASE)
        if vid_match:
            _flush_current()
            current_entry["visual_id"] = vid_match.group(1).strip()
            continue

        if line.lower().startswith("original visual:"):
            current_entry["original_visual"] = line.split(":", 1)[1].strip()
            continue
        if line.lower().startswith("replacement visual:"):
            current_entry["replacement_visual"] = line.split(":", 1)[1].strip()
            continue
        if line.lower().startswith("selection:"):
            current_entry["selection"] = line.split(":", 1)[1].strip()
            continue

    _flush_current()
    return entries


@traceable(
    metadata={
        "agent_name": "graphics_definition_v2",
        "step_name": "Decide Final Visuals",
        "function_name": "run_decide_final_visuals_for_all_rows",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous"),
    }
)
def run_decide_final_visuals_for_all_rows(sheet, llm="gemini_3_flash_thinking", max_workers=50):
    """
    Run final-visual decision for all eligible rows.

    :param sheet: gspread sheet object
    :param llm: LLM model name
    :param max_workers: Parallel workers
    :return: None
    """
    
    worksheet_name = "Slide Chunks"
    _, course_info_df = get_sheet_data_and_df(sheet, "Course info")
    course_name = str(course_info_df.loc[0, "Course Name"]).strip()
    target_audience = (
        str(course_info_df.loc[0, "Target Audience"]).strip()
        if "Target Audience" in course_info_df.columns
        else ""
    )

    drive = get_drive_instance()
    if not drive:
        print("⚠️ Drive not available; snapshot labels and some asset loads may be skipped.")

    worksheet, df = get_sheet_data_and_df(sheet, worksheet_name)
    if "decision_of_final_visual" not in df.columns:
        if "final_visuals_review" in df.columns:
            df.insert(int(df.columns.get_loc("final_visuals_review")) + 1, "decision_of_final_visual", "")
        else:
            df["decision_of_final_visual"] = ""
        save_to_sheet(worksheet, df)
        format_worksheet(worksheet)

    futures_map = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for index, row in df.iterrows():
            existing = _safe_str(row.get("decision_of_final_visual", "")).strip()
            if existing and existing.lower() != "nan":
                continue
            fgd = _safe_str(row.get("final_graphics_definition", "")).strip()
            if not fgd or fgd.lower() == "nan":
                continue

            future = executor.submit(
                process_decide_final_visuals_row,
                index,
                dict(row),
                course_name,
                target_audience,
                llm,
                drive,
            )
            futures_map[future] = index

        if not futures_map:
            print("✅ No rows to process (already decided, or missing final_graphics_definition).")
            return

        progress = SmartProgressBar(
            total_tasks=len(futures_map),
            description="Decide final visuals (primary vs fallback)",
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
                merge_and_save_columns(
                    sheet,
                    worksheet_name,
                    df,
                    ["final_graphics_definition", "decision_of_final_visual"],
                )
            except Exception as e:
                print(f"❌ Row {index + 1}: decision step failed: {e}")
                progress.update()
                merge_and_save_columns(
                    sheet,
                    worksheet_name,
                    df,
                    ["final_graphics_definition", "decision_of_final_visual"],
                )

    merge_and_save_columns(
        sheet,
        worksheet_name,
        df,
        ["final_graphics_definition", "decision_of_final_visual"],
    )
    format_worksheet(worksheet)
    print("✅ Decide final visuals (primary vs fallback) completed.")


def delete_final_visual_decisions(sheet):
    """
    Revert selected replacements and delete decision tracking column.

    :param sheet: gspread sheet object
    :return: None
    """

    worksheet_name = "Slide Chunks"
    ws, df = get_sheet_data_and_df(sheet, worksheet_name)
    if "decision_of_final_visual" not in df.columns:
        print(f"ℹ️ 'decision_of_final_visual' does not exist in '{worksheet_name}' worksheet")
        return

    reverted_rows = 0
    reverted_visuals = 0

    for row_index, row in df.iterrows():
        tracking_text = _safe_str(row.get("decision_of_final_visual", ""))
        final_def = _safe_str(row.get("final_graphics_definition", ""))
        voiceover_text = _safe_str(row.get("voiceover_segment", ""))
        slide_chunk = _safe_str(row.get("Slide Chunk", ""))
        visual_assignment_strategy = _safe_str(
            row.get("Visual Assignment Strategy", "Flexible, let the agent decide")
        )
        if not visual_assignment_strategy or visual_assignment_strategy == "nan":
            visual_assignment_strategy = "Flexible, let the agent decide"

        if not tracking_text or tracking_text.lower() == "nan":
            continue
        if not final_def or final_def.lower() == "nan":
            continue

        entries = _parse_decision_tracking_output(tracking_text)
        if not entries:
            continue

        updated_fgd = final_def
        row_reverted_count = 0

        for entry in entries:
            visual_id = _safe_str(entry.get("visual_id", "")).strip()
            original_url = _safe_str(entry.get("original_visual", "")).strip()
            replacement_url = _safe_str(entry.get("replacement_visual", "")).strip()
            selected_url = _safe_str(entry.get("selection", "")).strip()
            try:
                segment_num = int(_safe_str(entry.get("segment_num", "")).strip())
            except ValueError:
                continue

            # Revert only visuals where replacement had been selected.
            if not visual_id or not original_url or not replacement_url or not selected_url:
                continue
            if selected_url != replacement_url:
                continue
            if original_url == replacement_url:
                continue

            segments_map = build_segment_visual_map(
                voiceover_text=voiceover_text,
                final_graphics_definition=updated_fgd,
                visual_assignment_strategy=visual_assignment_strategy,
                slide_chunk=slide_chunk,
            )
            segment = segments_map.get(segment_num) or {}
            step = next(
                (
                    s
                    for s in (segment.get("visual_steps") or [])
                    if _safe_str(s.get("visual_id", "")).strip() == visual_id
                ),
                None,
            )
            if not step:
                continue

            replacement_xml = _build_replacement_xml_from_step(visual_id, step, original_url)
            updated_fgd = update_final_graphics_definition_with_replacements(
                updated_fgd,
                segment_num,
                replacement_xml,
            )
            row_reverted_count += 1

        if row_reverted_count > 0:
            df.at[row_index, "final_graphics_definition"] = updated_fgd
            reverted_rows += 1
            reverted_visuals += row_reverted_count

    # Drop tracking column after reversion.
    df = df.drop(columns=["decision_of_final_visual"])
    clear_worksheet(ws)
    save_to_sheet(ws, df)
    print(
        f"🧹 Reverted {reverted_visuals} selected replacement visual(s) across {reverted_rows} row(s), "
        f"and deleted 'decision_of_final_visual' column from '{worksheet_name}' worksheet."
    )
