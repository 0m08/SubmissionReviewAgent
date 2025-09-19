import time
import random
from tqdm import tqdm
from services.smart_progress_bar import SmartProgressBar
from concurrent.futures import ThreadPoolExecutor, as_completed
from services.helper_functions import create_and_populate_columns
from services.sheets_service import get_sheet_data_and_df, save_to_sheet, clear_worksheet
from agents.course_outline.video_search_tool.video_retriever_agent import retrieve_relevant_docs

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



from services.helper_functions import create_and_populate_columns


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
    
):
    """
    Safe version: always rebuilds all context columns from scratch.
    Skips rows where youtube_videos column already has video URLs.
    :param course_name: The name of the course.
    :param target_audience: The target audience of the course.
    :param course_outline: The outline of the course.
    :param sheet: The Google Sheets object.
    :param worksheet_name: The name of the worksheet containing learning objectives.
    : param drive: Authenticated GoogleDrive instance (PyDrive2).
    : param llm: The LLM to be used.
    : param max_turns: The maximum number of turns for the agent.
    : param verbose: Whether to print progress messages.
    : return: List of youtube_videos column values after processing.
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
        # Skip rows that already have YouTube videos
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
    max_workers = 10
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


def delete_youtube_videos(sheet, worksheet_name="Final Outline"):
    """Remove the youtube_videos column from the worksheet."""
    ws, df = get_sheet_data_and_df(sheet, worksheet_name)
    if "youtube_videos" in df.columns:
        df = df.drop(columns=["youtube_videos"])
        clear_worksheet(ws)
        save_to_sheet(ws, df)
