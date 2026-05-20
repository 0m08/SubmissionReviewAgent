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
from services.helper_functions import build_video_part
from services.sheets_service import (
    clear_worksheet,
    format_worksheet,
    get_sheet_data_and_df,
    save_to_sheet,
)
from services.smart_progress_bar import SmartProgressBar

load_dotenv()


slideshow_manifest_prompt = """You are a senior instructional graphics assembly agent specializing in HVAC e-learning content. Your task is to produce a slideshow manifest for one slide row: a single, machine-readable layout specification that downstream video automation will consume to arrange already selected visuals on screen while voiceover plays.

You do not choose new assets. You only assign URLs that already appear in the provided final graphics definition to predefined layout templates and scene narration spans. The manifest is the authoritative layout for this slide for automation.

You will be given course and slide context, the final graphics definition for the slide in the compact When VO: / Assigned Asset: pair format (narration then URL for each beat, in order), layout planning and storyboard text for reference, and the layout template library. You must not introduce or substitute any URL that is not already present in the final graphics definition text.

These are the inputs:

<course_information>
Course name: {course_name}
Topic name: {topic_name}
Subtopic name: {subtopic_name}
</course_information>

<slide_information>
Slide Type: {slide_type}
Slide Title: {slide_title}
Slide Content: {slide_content}
</slide_information>

<layout_planning_context>
{layout_plan}
</layout_planning_context>

<storyboard_planning_context>
{storyboard_planning}
</storyboard_planning_context>

<final_graphics_definition>
{final_graphics_definition}
</final_graphics_definition>

<layout_template_library>
The template attribute on each <scene> element must be exactly one of the following string literals (no other values, no aliases, no invented templates):

1) single_visual_hero
   - Use for one dominant full composition for the narration span of that scene.
   - Required slots (exactly 1): primary_visual

2) two_item_split_comparison
   - Use for two assets shown as a left/right comparison for the narration span of that scene.
   - Required slots (exactly 2): left_visual, right_visual

3) multi_panel_grid
   - Use for three or four parallel items of equal visual weight for the narration span of that scene.
   - Required slots: exactly 3 or 4 of: panel_1, panel_2, panel_3, panel_4 (use consecutive panel_1..panel_n with no gaps)

4) main_plus_supporting_inset
   - Use for one dominant asset plus one smaller supporting asset for the narration span of that scene.
   - Required slots (exactly 2): main_visual, inset_visual

</layout_template_library>

Instructions and Guidelines:

1. Scope and Responsibility
   - Produce one <slideshow_manifest> root element containing one or more <scene> elements in narration order.
   - Each scene must declare exactly one template attribute from the layout template library.
   - Each scene must declare which narration span it covers and which existing asset URLs fill which slot roles for that template.
   - Do not output commentary, prose layout descriptions, or fields outside the specified XML schema.

2. Asset URL Rules
   - Every asset attribute value on every <slot> element must be copied verbatim from an Assigned Asset: value in <final_graphics_definition> for this slide (the URL text only, exactly as given).
   - You must not add, invent, merge, or substitute URLs. You must not paraphrase or shorten URLs.

3. Parsing the Final Graphics Definition
   - The input is repeating pairs in order: When VO: with its narration, then Assigned Asset: with one URL for that beat.
   - Build scenes so that each scene's <narration_span> is the exact narration text that should be on screen for that composed layout.
   - Map each Assigned Asset: URL that appears in that scene to exactly one role allowed for that scene's template.

4. Scene Granularity
   - One scene corresponds to one composed screen layout for one contiguous narration span.
   - Use multiple scenes when the slide clearly needs a different template or a different set of assets for a later part of the narration.
   - Prefer fewer scenes when one template and one slot assignment cleanly covers multiple related When VO: beats.

5. Template and Slot Validity
   - For each scene, the number of <slot> elements and each role attribute must strictly match the chosen template per the layout template library.
   - Do not output slots with roles that are not listed for that template.
   - Do not output an empty asset attribute.

6. Use of Optional Context Blocks
   - <layout_planning_context> and <storyboard_planning_context> are optional references for grouping narration and choosing among valid templates.
   - If they conflict with the actual assigned assets, the assigned assets in final_graphics_definition always win: choose the template and slot mapping that best fits those URLs and the narration.

7. Transition Slide Type
   - If Slide Type is "Transition", you may use a single scene with template single_visual_hero when one asset supports the slide; still obey the URL rules.

Output:

Provide your output strictly in the following format:

<output>

<evaluation_breakdown>
Use this section as a structured reasoning scratchpad before writing the manifest.

Include:
- Brief slide intent in your own words.
- List of every URL you intend to use and which Assigned Asset: line (and paired When VO:) it came from in <final_graphics_definition>.
- For each scene: why this template was chosen and how each slot maps to narration and to a specific When VO beat.

It is acceptable for this section to be detailed if needed for correctness.
</evaluation_breakdown>

(Based on your above evaluation, provide the final manifest below.)

<slideshow_manifest>

<scene id="1" template="single_visual_hero">
<narration_span>
Exact narration text span this scene covers (from slide / When VO context).
</narration_span>
<slot role="primary_visual" asset="https://exact-url-copied-from-assigned-asset-line"/>
</scene>

<!-- Repeat <scene> as needed in narration order. scene id must be numeric strings 1, 2, 3, ... with no gaps. -->

</slideshow_manifest>

</output>

(Ensure that you strictly follow this exact XML format in your output. Do not add attributes or elements other than: output, evaluation_breakdown, slideshow_manifest, scene with attributes id and template, narration_span, slot with attributes role and asset.)
"""


_FGD_VO_CONTINUATION_SKIP_PREFIXES = (
    "visual instructions:",
    "selection justification:",
)


def _fgd_asset_line_prefix(line_lower_stripped):
    """
    Return which asset label starts this line, if any.

    :param line_lower_stripped: Lowercased stripped line text.
    :return: "assigned", "graphics", or None.
    """
    if line_lower_stripped.startswith("assigned asset:"):
        return "assigned"
    if line_lower_stripped.startswith("graphics to use:"):
        return "graphics"
    return None


def parse_when_vo_assigned_pairs(text):
    """
    Parse compact final_graphics_definition text into ordered (when_vo, asset_url) pairs.

    Expects blocks: When VO: ... then Assigned Asset: or Graphics to use: with URL
    (same line or following line). Lines like Visual Instructions are not part of When VO.

    :param text: Cell text from final_graphics_definition.
    :return: List of (when_vo_text, url) tuples in narration order.
    """
    if not text or str(text).strip() in ("", "nan"):
        return []
    lines = text.splitlines()
    pairs = []
    i = 0
    n = len(lines)
    while i < n:
        if not lines[i].strip().lower().startswith("when vo:"):
            i += 1
            continue
        vo_chunks = []
        first = lines[i]
        if ":" in first:
            after = first.split(":", 1)[1].strip()
            if after:
                vo_chunks.append(after)
        i += 1
        while i < n:
            ls = lines[i].strip()
            low = ls.lower()
            kind = _fgd_asset_line_prefix(low)
            if kind:
                url = ""
                if ":" in lines[i]:
                    url = lines[i].split(":", 1)[1].strip()
                if (not url or not url.startswith("http")) and i + 1 < n:
                    nxt = lines[i + 1].strip()
                    if nxt.startswith("http"):
                        url = nxt
                        i += 1
                if url.startswith("http"):
                    vo_text = "\n".join(vo_chunks).strip()
                    pairs.append((vo_text, url))
                i += 1
                break
            if not ls or ls == "----":
                i += 1
                continue
            if any(low.startswith(p) for p in _FGD_VO_CONTINUATION_SKIP_PREFIXES):
                i += 1
                continue
            vo_chunks.append(lines[i])
            i += 1
        continue
    return pairs


def _drive_file_id_from_url(url):
    """
    Extract Google Drive file id from a common Drive URL shape.

    :param url: Drive or other URL string.
    :return: File id or None.
    """
    if not url:
        return None
    m = re.search(r"/file/d/([a-zA-Z0-9_-]+)", url)
    if m:
        return m.group(1)
    m = re.search(r"[?&]id=([a-zA-Z0-9_-]+)", url)
    return m.group(1) if m else None


def urls_match_for_graphics_assignment(asset_url, assigned_url):
    """
    Return True when two URLs refer to the same graphics asset (Drive id or exact string).

    :param asset_url: URL from manifest or slot.
    :param assigned_url: URL from final_graphics_definition pair.
    :return: True if treated as the same asset.
    """
    a, b = (asset_url or "").strip(), (assigned_url or "").strip()
    if not a or not b:
        return False
    ida, idb = _drive_file_id_from_url(a), _drive_file_id_from_url(b)
    if ida and idb and ida == idb:
        return True
    return a.rstrip("/") == b.rstrip("/")


def normalize_when_vo_line(vo_text):
    """
    Strip surrounding quotes from parsed When VO text.

    :param vo_text: Raw When VO string from final_graphics_definition.
    :return: Cleaned string for prompts.
    """
    t = (vo_text or "").strip()
    if len(t) >= 2 and t[0] == t[-1] and t[0] in "'\"":
        return t[1:-1].strip()
    return t


def make_slot_narration_resolver_from_fgd(fgd_text):
    """
    Build a callable that maps each slot asset URL to its When VO beat from final_graphics_definition.

    Consumes (when_vo, url) pairs in document order so duplicate URLs map to consecutive beats.

    :param fgd_text: final_graphics_definition cell text for the slide row.
    :return: Callable ``resolve(asset_url)`` returning narration string or "" if unmatched.
    """
    pairs = parse_when_vo_assigned_pairs(fgd_text or "")
    used_indices = set()

    def resolve(asset_url):
        for j, (vo, pu) in enumerate(pairs):
            if j in used_indices:
                continue
            if urls_match_for_graphics_assignment(asset_url, pu):
                used_indices.add(j)
                return normalize_when_vo_line(vo)
        return ""

    return resolve


def _is_youtube_url(url):
    u = (url or "").lower()
    return "youtube.com" in u or "youtu.be" in u


def _append_asset_multimodal_parts(parts, beat_index, when_vo, asset_url, drive):
    """
    Append a text label and optional image or video part for one assigned asset.

    :param parts: List of Gemini Part objects to mutate.
    :param beat_index: 1-based index for logging labels.
    :param when_vo: Narration text for this beat.
    :param asset_url: Assigned asset URL.
    :param drive: Google Drive client for loading Drive-hosted images.
    :return: None
    """
    label = (
        f"Assigned beat {beat_index}\n\n"
        f"When VO:\n{when_vo or '(empty)'}\n\n"
        f"Assigned Asset:\n{asset_url}\n"
    )
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
    pil = load_image_from_url(asset_url, drive, f"slideshow_manifest_beat_{beat_index}")
    if pil:
        buffered = BytesIO()
        pil.convert("RGB").save(buffered, format="JPEG")
        parts.append(
            types.Part(
                inline_data=types.Blob(mime_type="image/jpeg", data=buffered.getvalue())
            )
        )


def parse_slideshow_manifest_response(response_text):
    """
    Extract inner slideshow_manifest XML and evaluation_breakdown text from model output.

    :param response_text: Raw model response string.
    :return: Tuple of (manifest_inner_xml, evaluation_text).
    """
    if not response_text:
        return "", ""
    eval_match = re.search(
        r"<evaluation_breakdown>(.*?)</evaluation_breakdown>",
        response_text,
        re.DOTALL | re.IGNORECASE,
    )
    man_match = re.search(
        r"<slideshow_manifest>(.*?)</slideshow_manifest>",
        response_text,
        re.DOTALL | re.IGNORECASE,
    )
    evaluation = eval_match.group(1).strip() if eval_match else ""
    manifest = man_match.group(1).strip() if man_match else ""
    return manifest, evaluation


@traceable(
    metadata={
        "agent_name": "graphics_definition_v2",
        "step_name": "Slideshow Manifest",
        "function_name": "generate_slideshow_manifest_for_row",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous"),
    }
)
def generate_slideshow_manifest_for_row(
    course_name,
    topic_name,
    subtopic_name,
    slide_type,
    slide_title,
    slide_content,
    layout_plan,
    storyboard_planning,
    final_graphics_definition,
    drive,
    llm="gemini_3_flash_thinking",
):
    """
    Build multimodal parts and call Gemini to produce the slideshow manifest for one slide row.

    :param course_name: The course name.
    :param topic_name: The topic name.
    :param subtopic_name: The subtopic name.
    :param slide_type: The slide type from the Slide Type column (e.g. Transition, Content).
    :param slide_title: The slide chunk title.
    :param slide_content: The slide chunk body text.
    :param layout_plan: Layout planning context from the layout_plan column.
    :param storyboard_planning: Storyboard text from the storyboard_planning column.
    :param final_graphics_definition: Compact When VO / Assigned Asset definition for the slide.
    :param drive: Google Drive instance for loading image assets.
    :param llm: The language model to use.
    :return: Tuple of (manifest_inner_xml, evaluation_breakdown_text).
    """
    layout_block = (
        layout_plan.strip()
        if layout_plan and str(layout_plan).strip() not in ("", "nan")
        else "No layout planning context provided."
    )
    storyboard_block = (
        storyboard_planning.strip()
        if storyboard_planning and str(storyboard_planning).strip() not in ("", "nan")
        else "No storyboard planning context provided."
    )
    fgd = (
        final_graphics_definition.strip()
        if final_graphics_definition and str(final_graphics_definition).strip() not in ("", "nan")
        else ""
    )

    prompt_text = slideshow_manifest_prompt.format(
        course_name=course_name or "",
        topic_name=topic_name or "",
        subtopic_name=subtopic_name or "",
        slide_type=slide_type or "",
        slide_title=slide_title or "",
        slide_content=slide_content or "",
        layout_plan=layout_block,
        storyboard_planning=storyboard_block,
        final_graphics_definition=fgd or "(empty)",
    )

    parts = []
    pairs = parse_when_vo_assigned_pairs(fgd)
    for idx, (vo, url) in enumerate(pairs, start=1):
        _append_asset_multimodal_parts(parts, idx, vo, url, drive)

    parts.append(types.Part(text=prompt_text))
    response_text = invoke_gemini_multimodal(parts, llm=llm, temperature=0.4)
    manifest_xml, evaluation = parse_slideshow_manifest_response(response_text)
    if not manifest_xml and response_text:
        manifest_xml = response_text.strip()
    return manifest_xml, evaluation


@traceable(
    metadata={
        "agent_name": "graphics_definition_v2",
        "step_name": "Slideshow Manifest",
        "function_name": "process_slideshow_manifest_row",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous"),
    }
)
def process_slideshow_manifest_row(index, row, course_name, drive, llm="gemini_3_flash_thinking"):
    """
    Process a single Slide Chunks row: read sheet columns and generate slideshow manifest output.

    :param index: The row index in the dataframe.
    :param row: The row data (pandas Series).
    :param course_name: The course name from Course info.
    :param drive: Google Drive instance for multimodal image loads.
    :param llm: The language model to use.
    :return: Tuple of (index, manifest_inner_xml). Empty manifest if skipped.
    """
    try:
        fgd = str(row.get("final_graphics_definition", "")).strip()
        if not fgd or fgd == "nan":
            return index, ""

        topic_name = str(row.get("Topic", "")).strip()
        subtopic_name = str(row.get("Subtopic", "")).strip()
        slide_title = str(row.get("Slide Chunk Title", "")).strip()
        slide_content = str(row.get("Slide Chunk", "")).strip()
        slide_type = str(row.get("Slide Type", "")).strip()
        if slide_type == "nan":
            slide_type = ""
        layout_plan = str(row.get("layout_plan", "")).strip()
        storyboard = str(row.get("storyboard_planning", "")).strip()

        manifest_xml, _ = generate_slideshow_manifest_for_row(
            course_name=course_name,
            topic_name=topic_name,
            subtopic_name=subtopic_name,
            slide_type=slide_type,
            slide_title=slide_title,
            slide_content=slide_content,
            layout_plan=layout_plan,
            storyboard_planning=storyboard,
            final_graphics_definition=fgd,
            drive=drive,
            llm=llm,
        )
        return index, manifest_xml
    except Exception as e:
        print(f"Error slideshow manifest row {index}: {e}")
        traceback.print_exc()
        return index, f"ERROR: {str(e)}"


@traceable(
    metadata={
        "agent_name": "graphics_definition_v2",
        "step_name": "Slideshow Manifest",
        "function_name": "run_slideshow_manifest_for_all_rows",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous"),
    }
)
def run_slideshow_manifest_for_all_rows(sheet, llm="gemini_3_flash_thinking", max_workers=50):
    """
    Generate slideshow_manifest for all rows that need it on Slide Chunks.

    :param sheet: The gspread sheet object.
    :param llm: The language model to use.
    :param max_workers: Number of parallel workers (default 50).
    :return: None
    """
    worksheet_name = "Slide Chunks"

    _, course_info_df = get_sheet_data_and_df(sheet, "Course info")
    course_name = course_info_df.loc[0, "Course Name"]

    ws, df = get_sheet_data_and_df(sheet, worksheet_name)

    drive = get_drive_instance()
    if not drive:
        print("Slideshow manifest: Drive not available; image assets may fail to load.")

    if "slideshow_manifest" not in df.columns:
        df["slideshow_manifest"] = ""

    rows_to_process = []
    for index, row in df.iterrows():
        fgd = str(row.get("final_graphics_definition", "")).strip()
        existing = str(row.get("slideshow_manifest", "")).strip()
        if not fgd or fgd == "nan":
            continue
        if existing and existing != "nan" and not existing.startswith("ERROR:"):
            continue
        rows_to_process.append((index, row))

    if not rows_to_process:
        print("Slideshow manifest: no rows to process (missing final_graphics_definition or already filled).")
        return

    print(f"Slideshow manifest: processing {len(rows_to_process)} row(s), max_workers={max_workers}")

    futures_map = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for index, row in rows_to_process:
            fut = executor.submit(
                process_slideshow_manifest_row,
                index,
                row,
                course_name,
                drive,
                llm,
            )
            futures_map[fut] = index

        progress = SmartProgressBar(
            total_tasks=len(futures_map),
            description="Slideshow manifest",
            save_interval=5,
        )

        for future in as_completed(futures_map):
            index = futures_map[future]
            try:
                row_index, manifest_xml = future.result()
                df.at[row_index, "slideshow_manifest"] = manifest_xml
                progress.update()
                if progress.should_save():
                    save_to_sheet(ws, df)
            except Exception as e:
                print(f"Slideshow manifest future error row {index}: {e}")
                df.at[index, "slideshow_manifest"] = f"ERROR: {str(e)}"
                progress.update()

    save_to_sheet(ws, df)
    format_worksheet(ws)
    print("Slideshow manifest: complete.")


def delete_slideshow_manifest_columns(sheet):
    """
    Remove slideshow_manifest column from Slide Chunks.

    :param sheet: The gspread sheet object.
    :return: None
    """
    worksheet_name = "Slide Chunks"
    ws, df = get_sheet_data_and_df(sheet, worksheet_name)
    if "slideshow_manifest" not in df.columns:
        print(f"No slideshow manifest column to delete on '{worksheet_name}'.")
        return
    df = df.drop(columns=["slideshow_manifest"])
    clear_worksheet(ws)
    save_to_sheet(ws, df)
    print(f"🗑️ Deleted column 'slideshow_manifest' from '{worksheet_name}' worksheet")


# ---------------------------------------------------------------------------
# image_editing_tracking → slideshow_manifest URL swap (post edit execution)
# ---------------------------------------------------------------------------


def _normalize_manifest_attribute_quotes(text):
    """
    Normalize doubled XML attribute quotes from sheet-exported strings.

    :param text: Raw manifest inner XML.
    :return: Text with attribute forms like id=""1"" normalized to id="1".
    """
    if not text:
        return text
    return re.sub(r'""([^"<>]*)""', r'"\1"', text)


def _escape_manifest_bare_ampersands(text):
    """
    Escape ampersands that are not valid XML entities.

    :param text: Raw manifest XML.
    :return: Entity-safe XML for parsing.
    """
    if not text:
        return text
    return re.sub(r"&(?!(amp|lt|gt|quot|apos|#\d+|#x[0-9A-Fa-f]+);)", "&amp;", text)


def split_image_editing_tracking_by_scene(tracking_text):
    """
    Split image_editing_tracking cell text into per-scene bodies (same header format as scene_edit_plan).

    :param tracking_text: Full image_editing_tracking column value.
    :return: List of (scene_id, scene_body_after_header) in document order.
    """
    text = (tracking_text or "").strip()
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


def parse_image_editing_tracking_edited_pairs(tracking_text):
    """
    Parse image_editing_tracking for Original+Edited image URL pairs.

    :param tracking_text: image_editing_tracking cell value.
    :return: List of dicts with keys scene_id (str), image_k (int), original_url, edited_url.
    """
    out = []
    text = (tracking_text or "").strip()
    if not text or text.startswith("ERROR:"):
        return out
    for scene_id, scene_body in split_image_editing_tracking_by_scene(text):
        parts = re.split(r"(?m)^Image (\d+):\s*$", scene_body)
        it = iter(parts[1:]) if len(parts) > 1 else iter([])
        for k_str, block in zip(it, it):
            try:
                k = int(k_str)
            except ValueError:
                continue
            body = (block or "").strip()
            if "No Edits made" in body and "Original Image:" not in body:
                continue
            om = re.search(r"^Original Image:\s*(\S+)", body, re.MULTILINE | re.IGNORECASE)
            edited_matches = re.findall(
                r"^Edited Image(?: after loop \d+)?:\s*(\S+)",
                body,
                re.MULTILINE | re.IGNORECASE,
            )
            if not om or not edited_matches:
                continue
            orig = om.group(1).strip()
            edited = edited_matches[-1].strip()
            if orig.startswith("http") and edited.startswith("http"):
                out.append({
                    "scene_id": str(scene_id).strip(),
                    "image_k": k,
                    "original_url": orig,
                    "edited_url": edited,
                })
    return out


def edited_url_pairs_from_image_editing_tracking(tracking_text):
    """
    Build (original_url, edited_url) pairs from image_editing_tracking cell text.

    :param tracking_text: image_editing_tracking column value.
    :return: List of (original_url, edited_url) tuples in document order.
    """
    return [
        (rec["original_url"], rec["edited_url"])
        for rec in parse_image_editing_tracking_edited_pairs(tracking_text)
    ]


_FGD_URL_IN_TEXT_RE = re.compile(r'https?://[^\s<>"\')\]]+', re.IGNORECASE)


def apply_url_replacements_to_final_graphics_definition(fgd_text, url_pairs):
    """
    Replace asset URLs in final_graphics_definition using (old_url, new_url) pairs.

    Scans http(s) URLs in the cell and swaps those that match each old_url via
    urls_match_for_graphics_assignment (Drive id or exact URL). Replacements run
    from end to start so earlier match indices stay valid.

    :param fgd_text: final_graphics_definition cell value.
    :param url_pairs: Iterable of (original_url, edited_url) from image edit tracking.
    :return: Tuple (updated_text, changed) where changed is True if at least one URL swapped.
    """
    text = str(fgd_text or "").strip()
    if not text or text in ("nan",) or text.startswith("ERROR:"):
        return text, False

    pairs_list = [(o, n) for o, n in (url_pairs or []) if o and n and o != n]
    if not pairs_list:
        return text, False

    matches = list(_FGD_URL_IN_TEXT_RE.finditer(text))
    if not matches:
        return text, False

    result = text
    changed = False
    for match in reversed(matches):
        found = match.group(0)
        start, end = match.start(), match.end()
        for old_url, new_url in pairs_list:
            if urls_match_for_graphics_assignment(old_url, found):
                if found != new_url:
                    result = result[:start] + new_url + result[end:]
                    changed = True
                break

    return result, changed


def apply_edited_asset_urls_to_slideshow_manifest_inner_xml(inner_xml, tracking_text):
    """
    Replace slot asset URLs in slideshow manifest inner XML with edited URLs from tracking.

    Uses urls_match_for_graphics_assignment so Drive URL variants still match. Order of
    replacement follows parse_image_editing_tracking_edited_pairs (first match wins per slot).

    :param inner_xml: slideshow_manifest column value (inner XML, one or more <scene> roots).
    :param tracking_text: image_editing_tracking cell value.
    :return: Updated inner XML string; unchanged if nothing to apply or parse fails.
    """
    pairs = parse_image_editing_tracking_edited_pairs(tracking_text)
    if not pairs:
        return (inner_xml or "").strip()

    text = str(inner_xml or "").strip()
    if not text or text in ("nan",) or text.startswith("ERROR:"):
        return text

    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]

    text = _normalize_manifest_attribute_quotes(text)
    wrapped = text
    if "<slideshow_manifest" not in wrapped.lower():
        wrapped = f"<slideshow_manifest>\n{wrapped}\n</slideshow_manifest>"
    safe = _escape_manifest_bare_ampersands(wrapped)

    try:
        root = ET.fromstring(safe)
    except ET.ParseError as e:
        print(f"apply_edited_urls_to_manifest: XML parse error: {e}")
        return (inner_xml or "").strip()

    tag = (root.tag or "").lower()
    if tag.endswith("slideshow_manifest"):
        scenes = root.findall("scene")
    elif tag.endswith("scene"):
        scenes = [root]
    else:
        print("apply_edited_urls_to_manifest: unexpected root tag, leaving manifest unchanged.")
        return (inner_xml or "").strip()

    for scene_el in scenes:
        for slot_el in scene_el.findall("slot"):
            asset = (slot_el.get("asset") or "").strip()
            if not asset:
                continue
            for rec in pairs:
                if urls_match_for_graphics_assignment(rec["original_url"], asset):
                    slot_el.set("asset", rec["edited_url"])
                    break

    if tag.endswith("slideshow_manifest"):
        chunks = []
        for child in list(root):
            chunks.append(ET.tostring(child, encoding="unicode"))
        return "\n\n".join(chunks).strip()
    return ET.tostring(root, encoding="unicode").strip()


def apply_url_replacements_to_slideshow_manifest_inner_xml(inner_xml, url_pairs):
    """
    Replace slot asset URLs in slideshow manifest inner XML using (old_url, new_url) pairs.

    :param inner_xml: slideshow_manifest column value (inner XML, one or more <scene> roots).
    :param url_pairs: Iterable of (old_url, new_url) tuples preserving caller order.
    :return: Tuple(updated_inner_xml, applied_pairs) where applied_pairs is a list of (old_url, new_url) tuples that actually replaced at least one slot.
    """
    pairs_list = [(o, n) for o, n in (url_pairs or []) if o and n and o != n]
    if not pairs_list:
        return (inner_xml or "").strip(), []

    text = str(inner_xml or "").strip()
    if not text or text in ("nan",) or text.startswith("ERROR:"):
        return text, []

    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]

    text = _normalize_manifest_attribute_quotes(text)
    wrapped = text
    if "<slideshow_manifest" not in wrapped.lower():
        wrapped = f"<slideshow_manifest>\n{wrapped}\n</slideshow_manifest>"
    safe = _escape_manifest_bare_ampersands(wrapped)

    try:
        root = ET.fromstring(safe)
    except ET.ParseError as e:
        print(f"apply_url_replacements_to_manifest: XML parse error: {e}")
        return (inner_xml or "").strip(), []

    tag = (root.tag or "").lower()
    if tag.endswith("slideshow_manifest"):
        scenes = root.findall("scene")
    elif tag.endswith("scene"):
        scenes = [root]
    else:
        print("apply_url_replacements_to_manifest: unexpected root tag, leaving manifest unchanged.")
        return (inner_xml or "").strip(), []

    applied_pairs = []
    applied_set = set()
    for scene_el in scenes:
        for slot_el in scene_el.findall("slot"):
            asset = (slot_el.get("asset") or "").strip()
            if not asset:
                continue
            for old_url, new_url in pairs_list:
                if urls_match_for_graphics_assignment(old_url, asset):
                    slot_el.set("asset", new_url)
                    key = (old_url, new_url)
                    if key not in applied_set:
                        applied_set.add(key)
                        applied_pairs.append(key)
                    break

    if not applied_pairs:
        return (inner_xml or "").strip(), []

    if tag.endswith("slideshow_manifest"):
        chunks = []
        for child in list(root):
            chunks.append(ET.tostring(child, encoding="unicode"))
        return "\n\n".join(chunks).strip(), applied_pairs
    return ET.tostring(root, encoding="unicode").strip(), applied_pairs


@traceable(
    metadata={
        "agent_name": "graphics_definition_v2",
        "step_name": "Apply edited URLs to slideshow manifest",
        "function_name": "process_apply_edited_urls_to_slideshow_manifest_row",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous"),
    },
)
def process_apply_edited_urls_to_slideshow_manifest_row(index, row):
    """
    For one Slide Chunks row, swap edited image URLs into slideshow_manifest and final_graphics_definition.

    :param index: DataFrame row index.
    :param row: DataFrame row.
    :return: Tuple (index, new_manifest_or_None, new_fgd_or_None). None per column means leave unchanged.
    """
    try:
        tracking = str(row.get("image_editing_tracking", "")).strip()
        if not tracking or tracking in ("nan",) or tracking.startswith("ERROR:"):
            return index, None, None

        url_pairs = edited_url_pairs_from_image_editing_tracking(tracking)
        if not url_pairs:
            return index, None, None

        new_manifest = None
        manifest = str(row.get("slideshow_manifest", "")).strip()
        if manifest and manifest not in ("nan",) and not manifest.startswith("ERROR:"):
            updated_manifest = apply_edited_asset_urls_to_slideshow_manifest_inner_xml(
                manifest, tracking,
            )
            if updated_manifest != manifest:
                new_manifest = updated_manifest

        new_fgd = None
        fgd = str(row.get("final_graphics_definition", "")).strip()
        if fgd and fgd not in ("nan",) and not fgd.startswith("ERROR:"):
            updated_fgd, fgd_changed = apply_url_replacements_to_final_graphics_definition(
                fgd, url_pairs,
            )
            if fgd_changed and updated_fgd != fgd:
                new_fgd = updated_fgd

        if new_manifest is None and new_fgd is None:
            return index, None, None
        return index, new_manifest, new_fgd
    except Exception as e:
        print(f"Error apply edited URLs row {index}: {e}")
        traceback.print_exc()
        return index, None, None


@traceable(
    metadata={
        "agent_name": "graphics_definition_v2",
        "step_name": "Apply edited URLs to slideshow manifest",
        "function_name": "run_apply_edited_urls_to_slideshow_manifest_for_all_rows",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous"),
    },
)
def run_apply_edited_urls_to_slideshow_manifest_for_all_rows(sheet, max_workers=30):
    """
    Update slideshow_manifest and final_graphics_definition with edited image URLs from tracking.

    Processes rows that have at least one Original/Edited pair in image_editing_tracking and
    a non-empty slideshow_manifest and/or final_graphics_definition to patch.

    :param sheet: gspread sheet object.
    :param max_workers: Parallel row workers.
    :return: None
    """
    worksheet_name = "Slide Chunks"
    ws, df = get_sheet_data_and_df(sheet, worksheet_name)

    if "slideshow_manifest" not in df.columns:
        df["slideshow_manifest"] = ""
    if "final_graphics_definition" not in df.columns:
        df["final_graphics_definition"] = ""
    if "image_editing_tracking" not in df.columns:
        print("Apply edited URLs: no image_editing_tracking column; nothing to do.")
        return

    rows_to_process = []
    for index, row in df.iterrows():
        tracking = str(row.get("image_editing_tracking", "")).strip()
        if not tracking or tracking in ("nan",) or tracking.startswith("ERROR:"):
            continue
        if not edited_url_pairs_from_image_editing_tracking(tracking):
            continue
        manifest = str(row.get("slideshow_manifest", "")).strip()
        fgd = str(row.get("final_graphics_definition", "")).strip()
        has_manifest = manifest and manifest not in ("nan",) and not manifest.startswith("ERROR:")
        has_fgd = fgd and fgd not in ("nan",) and not fgd.startswith("ERROR:")
        if not has_manifest and not has_fgd:
            continue
        rows_to_process.append((index, row))

    if not rows_to_process:
        print(
            "Apply edited URLs: no rows with edited tracking pairs and "
            "slideshow_manifest or final_graphics_definition."
        )
        return

    print(
        f"Apply edited URLs (manifest + FGD): processing {len(rows_to_process)} row(s), "
        f"max_workers={max_workers}"
    )

    futures_map = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for index, row in rows_to_process:
            fut = executor.submit(
                process_apply_edited_urls_to_slideshow_manifest_row,
                index,
                row,
            )
            futures_map[fut] = index

        progress = SmartProgressBar(
            total_tasks=len(futures_map),
            description="Apply edited URLs (manifest + FGD)",
            save_interval=5,
        )

        manifest_updates = 0
        fgd_updates = 0
        for future in as_completed(futures_map):
            index = futures_map[future]
            try:
                row_index, new_manifest, new_fgd = future.result()
                if new_manifest is not None:
                    df.at[row_index, "slideshow_manifest"] = new_manifest
                    manifest_updates += 1
                if new_fgd is not None:
                    df.at[row_index, "final_graphics_definition"] = new_fgd
                    fgd_updates += 1
                progress.update()
                if progress.should_save():
                    save_to_sheet(ws, df)
            except Exception as e:
                print(f"Apply edited URLs future error row {index}: {e}")
                progress.update()

    save_to_sheet(ws, df)
    format_worksheet(ws)
    print(
        f"Apply edited URLs: complete. "
        f"slideshow_manifest={manifest_updates}, final_graphics_definition={fgd_updates} row(s) updated."
    )


def delete_apply_edited_urls_to_slideshow_manifest(sheet):
    """
    No-op: edited URLs are merged into slideshow_manifest and final_graphics_definition.

    Re-run 'Generate Slideshow Manifest' from pre-edit FGD (or restore from backup) to reset URLs.

    :param sheet: gspread sheet object (unused).
    :return: None
    """
    print(
        "Apply edited URLs: delete skipped — re-run 'Generate Slideshow Manifest' or restore "
        "final_graphics_definition from backup to reset pre-edit asset URLs."
    )
