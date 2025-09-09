from modules.chain import Chain
import time
import random
import json
import re
import tqdm as tqdm
from langchain.schema import Document
from agents.course_outline.video_search_tool.video_retriever import load_video_vector_db_retriever
from concurrent.futures import ThreadPoolExecutor, as_completed
from agents.course_outline.video_search_tool.video_retriever_agent import retrieve_relevant_docs
from services.sheets_service import get_sheet_data_and_df, save_to_sheet
from services.smart_progress_bar import SmartProgressBar
from services.helper_functions import create_and_populate_columns


video_intelligence_agent_system_prompt = """
You are a Video Intelligence Agent. Your task is to find the absolute best video segment(s) that directly and concisely address a specific Learning Objective (LO).

STEPS:
1. ANALYZE the provided Learning Objective (LO).
2. REVIEW the full transcript for each candidate YouTube video. The transcript is provided as a series of sequential chunks, each with its start and end timestamps. 
3. FOR EACH VIDEO, your goal is to find the SINGLE BEST CONTIGUOUS SEGMENT that teaches the LO. Do not simply return the first mention. A segemt can include timestamps from multiple chunks, as long as they are the best fit for the LO.
    - Look for the segment where the explanation is most clear, direct, and comprehensive.
    - The segment should be concise. Avoid including long intros, tangents, or conclusions that are not relevant. Do not cut short a perfect explanation just to save time. Thta's why you can include start time from one chunk and end time from a later chunk.
    - If the concept is explained perfectly in a 2-minute segment, do not return a 10-minute segment.
4. SELECT THE BEST VIDEO: Compare the best segments from all videos. Choose the one video whose segment best fulfills the LO. Only consider multiple videos if a single video is insufficient to cover the LO fully.
5. If no exact segment is found but a whole video is broadly relevant, return the full video instead.

OUTPUT FORMAT:
Wrap your results in the following tags:

<analysis>
Explain which video segment is best and why.
</analysis>

<verdict>
One sentence with the chosen video_id and reason.
</verdict>

<best_video_segments>
{
  "best_video_segments": [
    {
      "video_id": "<video_id>",
      "start": <start_seconds>,
      "end": <end_seconds>
    }
  ]
}
</best_video_segments>
"""


def get_chunks_by_video(chroma_db, video_id):
    """
    Fetch all transcript chunks for a given video_id from Chroma.
    """
    results = chroma_db.get(
        where={"video_id": video_id},
        include=["metadatas", "documents"]
    )

    docs = []
    for text, metadata in zip(results["documents"], results["metadatas"]):
        docs.append(Document(page_content=text, metadata=metadata))

    # Sort by start_time
    return sorted(docs, key=lambda d: d.metadata.get("start_time", 0))


def get_full_video_transcript(video_id, chroma_db):
    """
    Retrieve and combine transcript for a given video_id into a list of dicts.
    """
    docs_sorted = get_chunks_by_video(chroma_db, video_id)

    transcript = []
    for d in docs_sorted:
        transcript.append({
            "start_time": d.metadata.get("start_time", 0),
            "end_time": d.metadata.get("end_time", None),
            "text": d.metadata.get("text_0", "")
        })
    return transcript


def refine_video_selection(lo, candidate_video_ids, chroma_db, llm="gemini_2_flash"):
    """
    Given an LO and candidate video_ids, analyze full transcripts and return structured JSON.
    """
    # Build transcripts dict
    transcripts = {}
    for vid in candidate_video_ids:
        transcripts[vid] = get_full_video_transcript(vid, chroma_db)

    # Initialize agent
    refiner_agent = Chain(llm=llm, tags=["analysis", "verdict", "best_video_segments"])
    refiner_agent.add_message(
        role="system",
        content=video_intelligence_agent_system_prompt
    )
    refiner_agent.add_message(
        role="user",
        content=f"Learning Objective:\n{lo}\n\nCandidate transcripts:\n{json.dumps(transcripts, indent=2)}"
    )

    # Run model
    response = refiner_agent.run()

    # Normalize to string
    if hasattr(response, "content"):
        text = response.content
    elif isinstance(response, dict) and "content" in response:
        text = response["content"]
    else:
        text = str(response)

    # --- Extract inner content of <best_video_segments> ---
    match = re.search(r"<best_video_segments>\s*(.*?)\s*</best_video_segments>", text, re.DOTALL | re.IGNORECASE)
    if not match:
        raise ValueError(f"Could not find <best_video_segments> JSON block in response:\n{text}")

    best_segments_raw = match.group(1).strip()

    # --- Normalize escapes ---
    # Remove leading/trailing quotes if LLM returned it as a string literal
    if best_segments_raw.startswith('"') and best_segments_raw.endswith('"'):
        best_segments_raw = best_segments_raw[1:-1]

    # Replace escaped newlines and quotes
    best_segments_raw = best_segments_raw.encode("utf-8").decode("unicode_escape")

    # Parse JSON safely
    try:
        best_segments_data = json.loads(best_segments_raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Model did not return valid JSON inside <best_video_segments>:\n{best_segments_raw}") from e

    # Return unified structure with analysis + verdict text + parsed JSON
    analysis_match = re.search(r"<analysis>\s*(.*?)\s*</analysis>", text, re.DOTALL | re.IGNORECASE)
    verdict_match = re.search(r"<verdict>\s*(.*?)\s*</verdict>", text, re.DOTALL | re.IGNORECASE)

    return {
        "analysis": analysis_match.group(1).strip() if analysis_match else "",
        "verdict": verdict_match.group(1).strip() if verdict_match else "",
        "best_video_segments": best_segments_data.get("best_video_segments", [])
    }


def task_fn(task, course_name, target_audience, course_outline, drive, max_turns, llm):
    """
    Worker function: run retrieval and build YouTube embed URLs for agent-selected video IDs.
    Returns:
      - video_urls: list of YouTube embed URLs (for youtube_videos column only)
      - new_context_chunks: list of (vid_id, transcript_chunk) pairs
    """
    print(f"\nStarting task for LO: {task['learning_objective']}")
    time.sleep(random.uniform(0.6, 1.4))  # avoid rate spikes

    selected_video_ids, all_docs = retrieve_relevant_docs(
        course_name=course_name,
        target_audience=target_audience,
        course_outline=course_outline,
        drive=drive,
        topic_name=task['topic'],
        subtopic_name=task['subtopic'],
        learning_objective=task['learning_objective'],
        max_turns=max_turns,
        llm=llm
    )

    print(f"Retrieved {len(all_docs)} docs, agent selected {len(selected_video_ids)} IDs")

    video_urls = []
    new_context_chunks = []
    seen = set()

    for doc_id in selected_video_ids:
        if str(doc_id).isdigit():
            doc_index = int(doc_id)
            if doc_index < 0 or doc_index >= len(all_docs):
                print(f" Skipping invalid index {doc_index} (not in all_docs)")
                continue  

            context_doc = all_docs[doc_index]
            metadata = getattr(context_doc, "metadata", {})
            print(f"Metadata for doc {doc_index}: {metadata}")

            vid_id = metadata.get("video_id")
            if not vid_id:
                print(" Skipping: no video_id")
                continue
            if vid_id in seen:
                print(f"Skipping duplicate video_id {vid_id}")
                continue
            seen.add(vid_id)

            start_sec = metadata.get("start_time", 0)
            end_sec = metadata.get("end_time", None)
            if end_sec is not None:
                url = f"https://www.youtube.com/embed/{vid_id}?start={start_sec}&end={end_sec}"
            else:
                url = f"https://www.youtube.com/embed/{vid_id}?start={start_sec}"

            video_urls.append(url)

            transcript = metadata.get("text_0", "").strip()
            if transcript:
                chunk = f"============= Doc id: AUTO =============\n{transcript}\n\n"
                new_context_chunks.append((vid_id, chunk))

            print(f"Added URL: {url}")
        else:
            print(f" Skipping doc_id {doc_id} (not integer)")

    print(f"✔ Finished LO: {task['learning_objective']} → {len(video_urls)} URLs\n")

    return video_urls, new_context_chunks

def run_video_search_for_los(
    course_name: str,
    target_audience: str,
    course_outline: str,
    sheet,
    worksheet_name: str,
    drive,
    llm,
    max_turns: int = 3,
    verbose: bool = True,
    max_workers: int = 10
):
    """
    Safe version: always rebuilds all context columns from scratch.
    Skips rows where youtube_videos column already has video URLs.
    """
    course_outline_with_lo_sheet, course_outline_with_lo_df = get_sheet_data_and_df(
        sheet=sheet,
        sheet_name=worksheet_name
    )
    if "Learning Objectives" not in course_outline_with_lo_df.columns:
        raise ValueError("Sheet must contain a column named 'Learning Objectives'")

    for col in ["youtube_videos", "video_links"]:
        if col not in course_outline_with_lo_df.columns:
            course_outline_with_lo_df[col] = ""

    task_list = []
    for row_index, row in course_outline_with_lo_df.iterrows():
        lo = row.get("Learning Objectives")
        if not isinstance(lo, str) or not lo.strip():
            continue
        # 🚨 Skip rows that already have YouTube videos
        if row.get("youtube_videos") and str(row["youtube_videos"]).strip():
            continue
        task_list.append({
            "row_index": row_index,
            "learning_objective": lo.strip(),
            "topic": row.get("Topic", ""),
            "subtopic": row.get("Subtopic", ""),
        })

    total_tasks = len(task_list)
    print(f"Starting video retrieval for {total_tasks} Learning Objectives...")

    futures_map = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for task in task_list:
            future = executor.submit(
                task_fn,
                task,
                course_name,
                target_audience,
                course_outline,
                drive,
                max_turns,
                llm
            )
            futures_map[future] = task

        progress = SmartProgressBar(total_tasks=total_tasks, description="Percent complete", save_interval=5)
        for future in tqdm(as_completed(futures_map), total=total_tasks):
            task = futures_map[future]
            row_index = task["row_index"]
            lo = task["learning_objective"]

            try:
                video_urls, new_context_chunks = future.result()

                # youtube_videos
                existing_urls = str(course_outline_with_lo_df.at[row_index, "youtube_videos"]).splitlines()
                all_urls = list({*existing_urls, *video_urls})
                youtube_url_str = "\n".join(u for u in all_urls if u.strip())
                course_outline_with_lo_df.at[row_index, "youtube_videos"] = youtube_url_str

                all_video_ids = {u.split("/embed/")[-1].split("?")[0] for u in all_urls if "/embed/" in u}
                existing_video_links = str(course_outline_with_lo_df.at[row_index, "video_links"]).splitlines()
                already_linked_ids = {u.split("/embed/")[-1].split("?")[0] for u in existing_video_links}

                # full existing context
                existing_context = ""
                for col in sorted(course_outline_with_lo_df.columns):
                    if col.startswith("context") and isinstance(course_outline_with_lo_df.at[row_index, col], str):
                        existing_context += course_outline_with_lo_df.at[row_index, col]

                # Count existing Doc ids (0-indexed continuation)
                doc_count = existing_context.count("============= Doc id:")

                # -------------------- Build new context -------------------- #
                new_context_text = ""
                for vid_id, chunk in new_context_chunks:
                    if vid_id in all_video_ids and vid_id not in already_linked_ids:
                        next_doc_id = doc_count  # zero-based
                        numbered_chunk = chunk.replace("Doc id: AUTO", f"Doc id: {next_doc_id}")
                        new_context_text += numbered_chunk
                        doc_count += 1

                if new_context_text:
                    combined_text = existing_context + new_context_text
                    course_outline_with_lo_df = create_and_populate_columns(
                        df=course_outline_with_lo_df,
                        text=combined_text,
                        specific_index=row_index,
                        col_base_name="context",
                        chunk_size=49000
                    )

                progress.update()
                if progress.should_save():
                    save_to_sheet(course_outline_with_lo_sheet, course_outline_with_lo_df)

            except Exception as e:
                print(f"Error processing LO '{lo}' (row {row_index + 1}): {e}")

    print("All LOs processed. Saving final results.")
    save_to_sheet(course_outline_with_lo_sheet, course_outline_with_lo_df)
    return course_outline_with_lo_df["youtube_videos"].tolist()