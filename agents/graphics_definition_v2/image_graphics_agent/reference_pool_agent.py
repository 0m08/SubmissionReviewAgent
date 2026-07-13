import os
import re
import traceback
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from langsmith import traceable

from modules.chain import Chain
from services.sheets_service import get_sheet_data_and_df, save_to_sheet, format_worksheet
from services.smart_progress_bar import SmartProgressBar
from agents.graphics_definition_v2.image_graphics_agent.drive_search import (
    execute_drive_search_for_query,
    parse_search_queries_column,
    _get_drive_instance
)

load_dotenv()


def parse_reference_image_map(ref_map_text):
    """
    Parse the reference_image_map column to extract segments and their mapped image URLs.
    
    Format:
    SEGMENT_1:url1
    SEGMENT_2:url2
    
    :param ref_map_text: The reference_image_map column content
    :return: Dict of {segment_number: url}
    """
    if not ref_map_text or ref_map_text.strip() == "" or ref_map_text == "nan":
        return {}
    
    ref_map = {}
    for line in str(ref_map_text).splitlines():
        line = line.strip()
        if not line:
            continue
        if ":" in line:
            part1, part2 = line.split(":", 1)
            part1 = part1.strip()
            part2 = part2.strip()
            if part1.startswith("SEGMENT_"):
                try:
                    seg_num = int(part1.replace("SEGMENT_", ""))
                    ref_map[seg_num] = part2
                except ValueError:
                    pass
    return ref_map


# Prompt for selecting the single best candidate from the pool
best_candidate_selection_prompt = """You are an expert Graphics Selection Agent specializing in educational HVAC and technical courses.
Your task is to select the SINGLE best reference image from a list of candidates retrieved from our reference image pool.

Here is the context:
<course_info>
Course: {course_name}
Topic: {topic_name}
Subtopic: {subtopic_name}
</course_info>

<slide_context>
Slide Title: {slide_title}
Full Slide Content:
{slide_chunk}
</slide_context>

<layout_planning_context>
{layout_plan}
</layout_planning_context>

Voiceover Segment (the specific sentence to match):
"{vo_text}"

Here are the retrieved reference image candidates from our pool:
{candidates_list}

Instructions:
1. Analyze the voiceover segment and slide context. Determine the precise visual meaning and context needed for this segment.
2. Evaluate each candidate. Compare its title, description, and relevance score.
3. Use the layout planning context to ensure the selected visual aligns with the overall planned scene structure for the slide.
4. Compare them to find the single most relevant and accurate graphic that represents the voiceover segment.
5. Output your reasoning and the exact URL of the single best candidate.
6. If none of the candidates are relevant at all, output "NONE" in the best_candidate tag.

Always provide your output strictly in the following format:
<output>
<reasoning>
[Your step-by-step reasoning explaining why the chosen candidate is the most relevant, accurate, and helpful visual for the voiceover segment, and why others were not selected]
</reasoning>
<best_candidate>
[The exact URL/link of the selected candidate, e.g. https://drive.google.com/file/d/... OR "NONE" if none of the candidates are relevant]
</best_candidate>
</output>
"""


@traceable(
    metadata={
        "agent_name": "graphics_definition_v2",
        "step_name": "Map Reference Image Pool",
        "function_name": "process_reference_pool_row",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous")
    }
)
def process_reference_pool_row(index, row, drive, root_folder_id, course_name, llm="gemini_3_flash"):
    """
    Process a single row, query Google Drive vector store with each segment's queries,
    and map the single best candidate via Gemini 3 Flash.
    """
    try:
        # Get row context
        slide_chunk = str(row.get("Slide Chunk", "")).strip()
        slide_title = str(row.get("Slide Chunk Title", "")).strip() or str(row.get("Topic", "")).strip() or "Slide"
        topic_name = str(row.get("Topic", "")).strip()
        subtopic_name = str(row.get("Subtopic", "")).strip()
        layout_plan = str(row.get("layout_plan", "")).strip()
        if not layout_plan or layout_plan == "nan":
            layout_plan = ""
        
        vo_segments_text = str(row.get("voiceover_segment", "")).strip()
        if not vo_segments_text or vo_segments_text == "nan":
            return index, str(row.get("reference_image_map", "")).strip()

        vo_segments = [seg.strip() for seg in vo_segments_text.splitlines() if seg.strip()]
        if not vo_segments:
            return index, str(row.get("reference_image_map", "")).strip()

        # Parse existing reference_image_map
        existing_ref_map = parse_reference_image_map(row.get("reference_image_map", ""))
        
        # Parse search queries
        search_queries_text = str(row.get("search_queries", "")).strip()
        segments_queries_list = parse_search_queries_column(search_queries_text)
        segments_queries = {seg_num: queries for seg_num, queries in segments_queries_list}

        pool_mapped_references = {}

        for seg_idx, vo_text in enumerate(vo_segments, start=1):
            # If the segment already has a reference mapped, preserve it!
            if seg_idx in existing_ref_map:
                print(f"Row {index + 1} Segment {seg_idx} already has inline/existing reference image mapped. Keeping it.")
                continue

            # Check if we have search queries for this segment
            queries = segments_queries.get(seg_idx, [])
            if not queries:
                print(f"Row {index + 1} Segment {seg_idx} has no search queries formulated. Skipping pool lookup.")
                continue

            # Query vector store for each of the segment's queries
            all_candidates = []
            seen_urls = set()

            for query in queries:
                try:
                    query_results = execute_drive_search_for_query(
                        query=query,
                        drive=drive,
                        k=5,
                        root_folder_id=root_folder_id
                    )
                    for res in query_results:
                        url = res.get("url")
                        if url and url not in seen_urls:
                            seen_urls.add(url)
                            all_candidates.append(res)
                except Exception as query_err:
                    print(f"⚠️ Error searching vector store for Row {index + 1} Segment {seg_idx} query \"{query}\": {query_err}")

            if not all_candidates:
                print(f"No reference pool candidates found for Row {index + 1} Segment {seg_idx}")
                continue

            # Deduplicate & Sort candidates by relevance score descending
            all_candidates.sort(key=lambda x: x.get("relevance_score", 0.0), reverse=True)
            top_8_candidates = all_candidates[:8]

            # Format candidates for Gemini
            candidates_list_str = ""
            for idx, cand in enumerate(top_8_candidates, start=1):
                cand_title = cand.get("title", "Untitled")
                cand_url = cand.get("url", "")
                cand_metadata = cand.get("metadata", {})
                cand_desc = cand_metadata.get("description") or cand_metadata.get("name") or "No description available"
                cand_score = cand.get("relevance_score", 0.0)
                
                candidates_list_str += f"""
Candidate {idx}:
- Title: {cand_title}
- Description: {cand_desc}
- URL: {cand_url}
- Relevance Score: {cand_score:.4f}
------------------------------------
"""

            # Run Gemini 3 Flash call to evaluate candidates and select the single best one
            try:
                search_evaluator = Chain(llm=llm, tags=["reasoning", "best_candidate"])
                search_evaluator.add_message(
                    role="user",
                    content=best_candidate_selection_prompt.format(
                        course_name=course_name,
                        topic_name=topic_name,
                        subtopic_name=subtopic_name,
                        slide_title=slide_title,
                        slide_chunk=slide_chunk,
                        layout_plan=layout_plan,
                        vo_text=vo_text,
                        candidates_list=candidates_list_str
                    )
                )

                response = search_evaluator.run()
                best_url = response.get("best_candidate", "").strip()

                if best_url and best_url.upper() != "NONE" and best_url.startswith("http"):
                    pool_mapped_references[seg_idx] = best_url
                    print(f"🎯 Row {index + 1} Segment {seg_idx}: Successfully mapped pool reference: {best_url}")
                else:
                    print(f"ℹ️ Row {index + 1} Segment {seg_idx}: LLM decided no candidates were relevant.")

            except Exception as eval_err:
                print(f"❌ Failed to run Gemini selection for Row {index + 1} Segment {seg_idx}: {eval_err}")

        # Assemble the final reference_image_map lines (preserving inline references)
        new_map_lines = []
        for seg_idx in range(1, len(vo_segments) + 1):
            if seg_idx in existing_ref_map:
                new_map_lines.append(f"SEGMENT_{seg_idx}:{existing_ref_map[seg_idx]}")
            elif seg_idx in pool_mapped_references:
                new_map_lines.append(f"SEGMENT_{seg_idx}:{pool_mapped_references[seg_idx]}")

        final_ref_map_text = '\n'.join(new_map_lines)
        return index, final_ref_map_text

    except Exception as e:
        print(f"❌ Error processing reference pool row {index + 1}: {e}")
        return index, str(row.get("reference_image_map", "")).strip()


def resolve_reference_pool_root_folder_id(drive, root_folder_id):
    """
    Find the parent folder of 'Vectorstore files' inside the course's folder structure.
    Normally, this is either root_folder_id itself or root_folder_id -> 'Reference Image' -> 'Vector Store'.
    Now, it also supports root_folder_id -> 'Reference Image' -> 'Vectorstore files' directly.
    """
    if not drive or not root_folder_id:
        return root_folder_id

    # 1. First, check if 'Vectorstore files' exists directly under root_folder_id
    try:
        vectorstore_list = drive.ListFile({
            'q': f"title='Vectorstore files' and '{root_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
        }).GetList()
        if vectorstore_list:
            print(f"📁 Found 'Vectorstore files' directly under root_folder_id. Using original root_folder_id: {root_folder_id}")
            return root_folder_id
    except Exception as err:
        print(f"Error checking Vectorstore files under root_folder_id: {err}")

    # 2. Try to search for 'Reference Image' under root_folder_id
    try:
        ref_img_list = drive.ListFile({
            'q': f"title='Reference Image' and '{root_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
        }).GetList()
        if ref_img_list:
            ref_img_id = ref_img_list[0]['id']
            print(f"📁 Found 'Reference Image' folder: {ref_img_id}")
            
            # Check if 'Vectorstore files' exists DIRECTLY inside 'Reference Image'
            direct_vectorstore_list = drive.ListFile({
                'q': f"title='Vectorstore files' and '{ref_img_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
            }).GetList()
            if direct_vectorstore_list:
                print(f"🎯 Found 'Vectorstore files' directly inside 'Reference Image'. Using resolved root_folder_id: {ref_img_id}")
                return ref_img_id

            # Search for 'Vector Store' or 'VectorStore' under 'Reference Image'
            vec_store_list = drive.ListFile({
                'q': f"(title='Vector Store' or title='VectorStore') and '{ref_img_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
            }).GetList()
            if vec_store_list:
                for vs in vec_store_list:
                    vs_id = vs['id']
                    # Check if 'Vectorstore files' exists inside this vs_id
                    sub_vectorstore_list = drive.ListFile({
                        'q': f"title='Vectorstore files' and '{vs_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
                    }).GetList()
                    if sub_vectorstore_list:
                        print(f"🎯 Found 'Vectorstore files' inside '{vs['title']}'. Using resolved root_folder_id: {vs_id}")
                        return vs_id
                
                # Default to the first found 'Vector Store' folder under 'Reference Image'
                vs_id = vec_store_list[0]['id']
                print(f"🎯 Using resolved Vector Store folder ID: {vs_id}")
                return vs_id
    except Exception as err:
        print(f"Error resolving reference pool path: {err}")

    # Fallback to the original root_folder_id
    print(f"ℹ️ Fallback to original root_folder_id: {root_folder_id}")
    return root_folder_id


@traceable(
    metadata={
        "agent_name": "graphics_definition_v2",
        "step_name": "Map Reference Image Pool",
        "function_name": "run_reference_image_pool_mapping",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous")
    }
)
def run_reference_image_pool_mapping(sheet, llm="gemini_3_flash", max_workers=50, selected_topics=None):
    """
    Run Reference Image Pool Mapping for all slide segments.
    """
    print("\n" + "="*80)
    print("🚀 Starting Reference Image Pool Mapping Sub-Agent")
    print("="*80 + "\n")

    worksheet_name = "Slide Chunks"

    # Fetch course and folder info
    _, course_info_df = get_sheet_data_and_df(sheet, "Course info")
    course_name = course_info_df.loc[0, "Course Name"]

    root_folder_id = st.session_state.get("root_folder_id")
    if not root_folder_id:
        # Fallback to the default one if not in session state
        root_folder_id = '1QS6PmCESfgFWNNEpRJDUB0E-t8iMAatH'

    # Fetch drive instance
    drive = _get_drive_instance()
    if not drive:
        raise ValueError("Google Drive authentication failed. Cannot perform pool mapping.")

    # Resolve sub-folder for reference image vector store if nested
    resolved_root_id = resolve_reference_pool_root_folder_id(drive, root_folder_id)

    # Load Slide Chunks worksheet
    worksheet, df = get_sheet_data_and_df(sheet, worksheet_name)

    if "reference_image_map" not in df.columns:
        df["reference_image_map"] = ""

    # Prepare rows to process
    rows_to_process = []
    for index, row in df.iterrows():
        topic_name = str(row.get("Topic", "")).strip()
        if selected_topics and topic_name not in selected_topics:
            continue
        
        # We need both voiceover_segment and search_queries to be filled before we can run pool mapping
        vo_segments_text = str(row.get("voiceover_segment", "")).strip()
        search_queries_text = str(row.get("search_queries", "")).strip()

        if (not vo_segments_text or vo_segments_text == "nan" or 
            not search_queries_text or search_queries_text == "nan"):
            continue

        # Check if there are any segments in this row that are not already mapped
        vo_segments = [seg.strip() for seg in vo_segments_text.splitlines() if seg.strip()]
        existing_ref_map = parse_reference_image_map(row.get("reference_image_map", ""))

        if len(existing_ref_map) < len(vo_segments):
            rows_to_process.append((index, row))

    if not rows_to_process:
        print("All segments already have reference image mappings or missing required inputs (search queries/voiceover segments).")
        return

    print(f"Submitting {len(rows_to_process)} row(s) for reference pool mapping in parallel (max_workers={max_workers})")

    # Initialize progress tracker
    progress = SmartProgressBar(
        total_tasks=len(rows_to_process),
        description="Mapping Reference Image Pool",
        save_interval=5
    )

    # Process in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                process_reference_pool_row,
                index,
                row,
                drive,
                resolved_root_id,
                course_name,
                llm
            ): index
            for index, row in rows_to_process
        }

        for future in as_completed(futures):
            index = futures[future]
            try:
                row_index, final_ref_map_text = future.result()
                df.at[row_index, "reference_image_map"] = final_ref_map_text
                progress.update()

                # Save every interval
                if progress.should_save():
                    print(f"Saving partial progress after {progress.completed_count} tasks completed.")
                    save_to_sheet(worksheet, df)
            except Exception as e:
                print(f"❌ Error processing row {index + 1}: {e}")
                progress.update()

    # Final save and format
    save_to_sheet(worksheet, df)
    format_worksheet(worksheet)
    print("✅ Reference Image Pool Mapping completed successfully.")


def delete_reference_image_pool_mappings(sheet):
    """
    Delete any reference pool mapping results and restore to only inline mappings if any.
    """
    worksheet_name = "Slide Chunks"
    ws, df = get_sheet_data_and_df(sheet, worksheet_name)
    
    if "reference_image_map" in df.columns:
        print("🗑️ Resetting reference_image_map to only inline references...")
        
        for index, row in df.iterrows():
            slide_chunk_raw = str(row.get("slide_chunk_raw", "")).strip()
            if not slide_chunk_raw or slide_chunk_raw == "nan":
                slide_chunk_raw = str(row.get("Slide Chunk", "")).strip()
            
            # Re-parse inline segments from slide chunk
            # Any inline segment has format like [alt](url)
            inline_matches = re.findall(r'\[(.*?)\]\((https?://.*?)\)', slide_chunk_raw)
            
            if inline_matches:
                # Reconstruct inline maps
                vo_segments_text = str(row.get("voiceover_segment", "")).strip()
                vo_segments = [seg.strip() for seg in vo_segments_text.splitlines() if seg.strip()]
                
                # To be absolutely consistent with segment_slide.py split behavior:
                # Each inline match represents a chunk of text that contains a reference image at the end.
                # Let's rebuild the lines
                map_lines = []
                seg_pos = 0
                for alt_text, drive_url in inline_matches:
                    # In segment_slide.py, vo_text represents the text matching before the image.
                    # We can find which vo_segment ends with some part of the alt_text or represents it.
                    # As a simpler and safer approach: we can look up the URLs we parsed from the markdown,
                    # and if we can't align perfectly, we can match segment indices.
                    # Let's align by checking which segments correspond to the parsed list.
                    # Usually, there's a 1-to-1 matching.
                    seg_pos += 1
                    if seg_pos <= len(vo_segments):
                        map_lines.append(f"SEGMENT_{seg_pos}:{drive_url}")
                
                df.at[index, "reference_image_map"] = '\n'.join(map_lines)
            else:
                df.at[index, "reference_image_map"] = ""
                
        save_to_sheet(ws, df)
        format_worksheet(ws)
        print("✅ reference_image_map successfully restored to original inline-only state.")
    else:
        print("ℹ️ reference_image_map column does not exist.")
