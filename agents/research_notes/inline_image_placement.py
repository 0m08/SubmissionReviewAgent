"""
Inline Image Placement — per-row agent that repositions existing markdown
image links (`![alt](url)`) next to the paragraph each image illustrates.

Runs as a standalone step in Section 4 of the Research Notes pipeline, after
the Checklist Based Review and Revise step and before the diff visualizer.

Per-row flow:
  1. Read the row's `research_notes` column directly.
  2. Skip rows with no image links or with video-format transcripts.
  3. Launch a LangGraph agent with tools: preview_image, str_replace, stop.
     The agent loads each image via preview_image and uses str_replace to
     move the inline link next to the matching paragraph. URLs and alt text
     are preserved verbatim (guarded by a post-agent set-equality check;
     mismatches are reverted).
"""
import re
import pandas as pd
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed

from langsmith import traceable
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain.messages import ToolMessage, HumanMessage
from langchain_core.rate_limiters import InMemoryRateLimiter

from services.sheets_service import (
    get_sheet_data_and_df,
    save_to_sheet,
    clear_worksheet,
    delete_worksheet,
    format_worksheet,
    create_or_read_worksheet,
    hide_worksheet_by_name,
)
from services.smart_progress_bar import SmartProgressBar
from services.web_page_loaders import DRIVE_FILE_LINK_RE
from services.crud_text_block_tools import preview_image
from agents.research_notes.review_revise_research_notes import (
    ResearchNotesState,
    rn_str_replace,
    rn_stop,
)


# ============================================================
# Helpers
# ============================================================

STATUS_COL = "inline_image_placement_status"
STATUS_DONE = "done"

_VIDEO_MARKERS = ("Link:", "Video_Id:", "Start:", "End:", "Transcript:")
_MD_IMAGE_RE = re.compile(r"!?\[.*?\]\((.*?)\)")
_IMAGE_EXT_RE = re.compile(r".*\.(png|jpg|jpeg|gif|svg|bmp|webp)(\?.*)?$", re.IGNORECASE)


def extract_inline_image_links(text: str) -> list:
    """Extract markdown image link URLs that are either standard image URLs
    (png/jpg/jpeg/gif/svg/bmp/webp) or Google Drive file links. Drive links
    are included here because `preview_image` can load them via PyDrive, even
    though the shared `extract_inline_image_links` utility filters them
    out for other callers.
    """
    if not text:
        return []
    # Collapse newlines that sometimes split markdown image links across lines.
    normalized = re.sub(
        r"!?\[.*?\]\([^\)]*\n[^\)]*\)",
        lambda m: m.group(0).replace("\n", ""),
        text,
    )
    links = []
    for url in _MD_IMAGE_RE.findall(normalized):
        if _IMAGE_EXT_RE.match(url) or DRIVE_FILE_LINK_RE.search(url):
            links.append(url)
    return links


def is_video_research_notes(research_notes: str) -> bool:
    """Return True if the research notes are a video transcript block."""
    if not research_notes:
        return False
    return all(marker in research_notes for marker in _VIDEO_MARKERS)


# ============================================================
# Prompt
# ============================================================

inline_image_placement_prompt = """You are repositioning existing inline markdown image links within a single piece of research notes so that each image sits immediately after the paragraph it visually illustrates. You are also improving the alt text so it describes what the image actually shows.

Course Name: {course_name}
Target Audience: {target_audience}

Current Research Notes:
<research_notes>
{research_notes}
</research_notes>

The following image link URLs were extracted from the research notes above. These are the exact URLs you must work with — do not invent new ones, do not drop any, and do not modify any URL:
<image_links>
{image_links}
</image_links>

Important notes about link format:
- The research notes may contain image links in either `[alt](url)` or `![alt](url)` form. Treat both as image links to be repositioned. Do not convert between the two forms — preserve whichever form (`[` vs `!`) each link already uses.
- Alt text (the part inside the square brackets, e.g. "image1", "image2") is often a generic placeholder. You should rename the alt text to briefly describe what the image actually shows (e.g. "furnace and water heater", "kitchen exhaust fan"). The alt text is the only thing you may rewrite.
- URLs must be preserved byte-for-byte.

Tools available:
- preview_image(image_url, intent): load the image so you can see what it shows. Call this once per URL before deciding placement and alt text.
- str_replace(old_text, new_text, intent): edit the research notes in place. old_text must appear exactly once.
- stop(reason): call when done.

Instructions:
1. For each URL in <image_links>, call preview_image to see what the image shows.
2. Identify which paragraph in the research notes the image best illustrates (subject match — the noun or process depicted).
3. Use str_replace to:
   a. Move each image link to sit next to the exact text/sentence it illustrates. The link can be placed inline — between sentences, at the end of a sentence, or in the middle of a paragraph — wherever it is most directly tied to the text. A new line is only appropriate when the image illustrates the whole paragraph rather than a specific sentence; prefer inline placement when a specific sentence or phrase is the subject.
   b. Rewrite the alt text to briefly describe the image contents. Keep the surrounding syntax exactly as it appeared in the source (`[...]` vs `![...]`), and keep the URL unchanged.
   c. If multiple image links are currently bunched together (e.g. `[image1](url1), [image2](url2), [image3](url3)`), split them and place each one next to the specific text it belongs to — they may belong to different sentences or different paragraphs.
4. If an image is already correctly positioned next to the right text and already has descriptive alt text, leave it alone.
5. When all image links have been considered, call `stop` with a brief summary.
"""


# ============================================================
# Per-row agent
# ============================================================

@traceable(metadata={
    "agent_name": "research_notes",
    "step_name": "Inline Image Placement",
    "function_name": "run_inline_image_placement_for_row",
    "user_id": st.session_state.get("role", "anonymous") if hasattr(st, "session_state") else "anonymous",
})
def run_inline_image_placement_for_row(
    row_id,
    research_notes,
    image_links,
    course_name,
    target_audience,
    llm="gemini_3_flash",
):
    """Run the placement agent on a single research-notes string.

    Returns the revised research notes. If the revised text has a different
    image-link set than the original, the original is returned unchanged as
    a safety net.
    """
    print(f"\n🖼️ Inline Image Placement — row {row_id} ({len(image_links)} image(s))")

    rate_limiter = InMemoryRateLimiter(
        requests_per_second=1,
        check_every_n_seconds=0.1,
        max_bucket_size=10,
    )

    from langchain_google_genai.chat_models import ChatGoogleGenerativeAI

    model = ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview",
        thinking_level="high",
        include_thoughts=True,
        rate_limiter=rate_limiter,
        max_retries=20,
    )

    @wrap_tool_call
    def handle_tool_errors(request, handler):
        try:
            return handler(request)
        except Exception as e:
            return ToolMessage(
                content=f"Tool error: Please check your input and try again.\n({str(e)})",
                tool_call_id=request.tool_call["id"],
            )

    tools = [preview_image, rn_str_replace, rn_stop]
    graph = create_agent(
        model=model,
        tools=tools,
        state_schema=ResearchNotesState,
        middleware=[handle_tool_errors],
    )

    image_links_block = "\n".join(f"- {url}" for url in image_links)
    prompt = inline_image_placement_prompt.format(
        course_name=course_name,
        target_audience=target_audience,
        research_notes=research_notes,
        image_links=image_links_block,
    )

    state = {
        "messages": [HumanMessage(content=prompt)],
        "research_notes": {"text": research_notes},
        "row_id": row_id,
    }

    try:
        final_state = graph.invoke(state, {"recursion_limit": 50})
    except Exception as e:
        print(f"⚠️ Inline Image Placement failed for row {row_id}: {e}. Keeping original text.")
        return research_notes

    revised = str(final_state["research_notes"]["text"])

    # Safety: image-link set must be byte-identical to the input set.
    original_set = set(image_links)
    revised_set = set(extract_inline_image_links(revised))
    if original_set != revised_set:
        print(
            f"⚠️ Row {row_id}: image-link set changed during placement "
            f"(added={sorted(revised_set - original_set)}, "
            f"removed={sorted(original_set - revised_set)}). Reverting to original."
        )
        return research_notes

    return revised


# ============================================================
# Orchestration
# ============================================================

@traceable(metadata={
    "agent_name": "research_notes",
    "step_name": "Inline Image Placement",
    "function_name": "run_inline_image_placement_for_all_rows",
    "user_id": st.session_state.get("role", "anonymous") if hasattr(st, "session_state") else "anonymous",
})
def run_inline_image_placement_for_all_rows(
    sheet,
    course_name,
    target_audience,
    worksheet_name="Final Outline",
    llm="gemini_3_flash",
    max_workers=10,
):
    """Main entry point wired into the research-notes pipeline."""
    research_notes_sheet, research_notes_df = get_sheet_data_and_df(
        sheet=sheet, sheet_name=worksheet_name
    )

    # Backup for delete-step restore.
    backup_ws_name = "Backup Final Outline Sheet for Delete step of Inline Image Placement"
    print(f"📋 Creating backup sheet '{backup_ws_name}' for delete step functionality...")
    backup_ws, _ = create_or_read_worksheet(sheet, backup_ws_name)
    clear_worksheet(backup_ws)
    save_to_sheet(backup_ws, research_notes_df)
    format_worksheet(backup_ws)
    try:
        hide_worksheet_by_name(sheet, backup_ws_name)
        print(f"✅ Backup sheet created and hidden successfully")
    except Exception as e:
        print(f"⚠️ Warning: Could not hide backup sheet: {e}")

    if "research_notes" not in research_notes_df.columns:
        print("⚠️ 'research_notes' column missing; nothing to do.")
        return

    # Ensure a tracking column exists so partial runs can resume.
    if STATUS_COL not in research_notes_df.columns:
        research_notes_df[STATUS_COL] = ""

    # Identify rows with image links to process.
    jobs = []
    skipped_done = 0
    for index, row in research_notes_df.iterrows():
        raw = row.get("research_notes", "")
        research_notes = "" if pd.isna(raw) else str(raw)
        if not research_notes.strip():
            continue
        if is_video_research_notes(research_notes):
            continue
        image_links = extract_inline_image_links(research_notes)
        if not image_links:
            continue
        status_raw = row.get(STATUS_COL, "")
        status = "" if pd.isna(status_raw) else str(status_raw).strip().lower()
        if status == STATUS_DONE:
            skipped_done += 1
            continue
        jobs.append((index, research_notes, image_links))

    if skipped_done:
        print(f"⏭️  Skipping {skipped_done} row(s) already marked '{STATUS_DONE}' in '{STATUS_COL}'")

    if not jobs:
        print("✅ No rows with inline image links to process — nothing to reposition.")
        return

    print(f"\n🖼️ Inline Image Placement: {len(jobs)} row(s) have image links")

    progress = SmartProgressBar(
        total_tasks=len(jobs),
        description="Inline Image Placement",
        save_interval=5,
    )

    save_interval = 5
    completed_since_save = 0

    futures_map = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for row_id, research_notes, image_links in jobs:
            future = executor.submit(
                run_inline_image_placement_for_row,
                row_id,
                research_notes,
                image_links,
                course_name,
                target_audience,
                llm,
            )
            futures_map[future] = row_id

        for future in as_completed(futures_map):
            row_id = futures_map[future]
            try:
                revised = future.result()
                if revised is not None:
                    research_notes_df.at[row_id, "research_notes"] = revised
                research_notes_df.at[row_id, STATUS_COL] = STATUS_DONE
            except Exception as e:
                print(f"⚠️ Row {row_id}: placement job raised {e}. Keeping original text.")
            progress.update()
            completed_since_save += 1
            if completed_since_save >= save_interval:
                try:
                    research_notes_sheet.clear()
                    save_to_sheet(worksheet=research_notes_sheet, df=research_notes_df)
                    completed_since_save = 0
                except Exception as e:
                    print(f"⚠️ Periodic save failed: {e}")

    research_notes_sheet.clear()
    save_to_sheet(worksheet=research_notes_sheet, df=research_notes_df)

    print("✅ Inline Image Placement completed.")


def delete_inline_image_placement(sheet, worksheet_name="Final Outline"):
    """Restore Final Outline from the Inline Image Placement backup and delete the backup."""
    backup_ws_name = "Backup Final Outline Sheet for Delete step of Inline Image Placement"

    try:
        final_ws, _ = get_sheet_data_and_df(sheet, worksheet_name)
    except Exception:
        print(f"⚠️ Final Outline sheet '{worksheet_name}' not found. Nothing to restore.")
        return

    try:
        backup_ws, backup_df = get_sheet_data_and_df(sheet, backup_ws_name)
        print(f"✅ Found Inline Image Placement backup with {len(backup_df)} rows")
    except Exception:
        print(f"⚠️ No Inline Image Placement backup found. Skipping restore.")
        return

    print(f"🔄 Restoring Final Outline from Inline Image Placement backup...")
    clear_worksheet(final_ws)
    if not backup_df.empty:
        save_to_sheet(final_ws, backup_df)

    print(f"🗑️ Cleaning up Inline Image Placement backup sheet...")
    try:
        delete_worksheet(sheet, backup_ws_name)
        print(f"✅ Backup sheet '{backup_ws_name}' deleted successfully")
    except Exception as e:
        print(f"⚠️ Warning: Could not delete Inline Image Placement backup sheet: {e}")
