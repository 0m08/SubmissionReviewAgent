import time
import random
from tqdm import tqdm
from typing import List, Optional, Dict
from services.llm_service import llm_with_retry
from modules.chain import Chain, xml_check_and_fix
from services.smart_progress_bar import SmartProgressBar
from concurrent.futures import ThreadPoolExecutor, as_completed
from services.sheets_service import get_sheet_data_and_df, save_to_sheet
from agents.course_outline.video_search_tool.video_retriever import video_retriever

video_retriever_agent_prompt = """
You are an expert technical video evaluator, tasked with selecting relevant videos for both:
1. Specific learning objectives (when context is provided)
2. General search queries (when no context exists)

GENERAL GUIDELINES:
- For learning objectives: Prioritize direct instructional matches
- For general queries: Accept broader conceptual matches
- With context: Maintain alignment with provided topic/subtopic
- Without context: Focus on query intent

CONTEXT USAGE (when available):
1. Learning Objectives:
   - Use context to identify prerequisite knowledge
   - Ensure videos connect specific objectives to broader topics

2. General Queries:
   - Use context to disambiguate terms
   - Narrow results to relevant subdomains

Your primary goal is to identify videos that match or are reasonably close to the search query intent.
You should be flexible in accepting videos that are conceptually related, even if not a perfect match.
Only return no videos if they are completely unrelated to the query domain.

For each search turn, you will receive:
- The search query used to retrieve results
- A list of video search results with metadata (title, description, source)

Your responsibilities are as follows:

---

1. FLEXIBLE CONTENT EVALUATION
   - Prioritize finding useful videos over perfect matches
   - Accept videos that demonstrate related concepts or broader applications
   - Use video metadata (title, description, channel) to understand content
   - Consider partial matches as valid if they serve similar learning objectives

---

2. WHEN TO TERMINATE (Return Videos)

You MUST set verdict = TERMINATE and return videos if:
- Any video reasonably addresses the query (even partially)
- Videos cover similar concepts that would be useful for the query
- The content is in the same domain and could help understand the topic
- Only continue if videos are completely irrelevant to the domain

---

3. WHEN TO REFINE (Continue Searching)

Only set verdict = CONTINUE if:
- Videos are completely unrelated to HVAC/technical domain
- Content is about entirely different subjects
- No conceptual overlap exists with query intent
- When refining, maintain original learning objectives

---

4. QUERY REFINEMENT GUIDELINES

If refinement is absolutely necessary:
  - Keep core learning objectives intact
  - Only modify to clarify, not change subject
  - Add context rather than replace concepts
  - Preserve technical terminology
  - Follow original query format/style

---

5. FINAL FALLBACK

After 3 attempts, return ANY videos that are:
- In the same technical domain
- Cover related concepts
- Could be useful for similar learning objectives
Only return empty if content is completely non-technical/unrelated

---

RESPONSE FORMAT:

Respond using these exact tags:

<observations>
[Note how videos relate to query, even if partially]
[Highlight any useful aspects of imperfect matches]
</observations>

<verdict>
["TERMINATE" if videos are relevant or conceptually close
 "CONTINUE" ONLY if completely irrelevant]
</verdict>

<selected_indexes>
[Include ALL videos that could be useful, even partial matches]
[Empty only if absolutely no relevant content]
</selected_indexes>

<action>
[If CONTINUE: specific reason why all videos fail]
[If TERMINATE: note how videos meet needs]
</action>

<query>
[If CONTINUE: minimal refinement preserving intent]
[If TERMINATE: original query or "Accepted partial matches"]
</query>
"""

# -------------------- Helper Functions -------------------- #
def parse_selected_indexes(index_string: str) -> List[int]:
    """Parse string like '0, 2' into a list of integers."""
    try:
        return [int(i.strip()) for i in index_string.split(",") if i.strip().isdigit()]
    except Exception:
        return []

# -------------------- Video Retriever Agent -------------------- #
def video_retriever_agent(
    query: str,
    drive,
    llm,
    k: int = 20,
    max_turns: int = 3,
    filters: Optional[Dict] = None,
    verbose: bool = True,
     context: Optional[Dict] = None
) -> List[Dict]:
    """
    LLM-driven agent to evaluate videos and optionally refine queries.
    Returns relevant videos with metadata.
    """

     # Initialize context if not provided
    context = context or {}

    chain = Chain(
        llm=llm,
        tags=["observations", "verdict", "selected_indexes", "action", "query"],
        use_xml_checker=True
    )

    # Add context to the system prompt
    enhanced_prompt = video_retriever_agent_prompt

    if context:
        # Only add context header if we actually have context items
        context_items = []

        if context.get('topic'):
            context_items.append(f"- Broad Topic: {context['topic']}")
        if context.get('subtopic'):
            context_items.append(f"- Subtopic: {context['subtopic']}")

        if context_items:
            enhanced_prompt += "\n\nCONTEXTUAL INFORMATION:\n"
            enhanced_prompt += "\n".join(context_items)

            # Adaptive guidance based on query type
            if "learning objective" in query.lower() or "lo" in query.lower():
                enhanced_prompt += "\nUse this context to better understand the learning objective."
            else:
                enhanced_prompt += "\nUse this context to better understand the search query scope."

    original_query = query
    chain.add_message(role="system", content=video_retriever_agent_prompt)

    for turn in range(max_turns):
        if verbose:
            print(f"\nTurn {turn + 1}: Query = '{query}'")

        # Retrieve top-k videos using your existing video_retriever
        results = video_retriever(query=query, drive=drive, k=k, filters=filters)
        if not results:
            print("No videos retrieved.")
            return []

        # Prepare prompt content
        video_list_str = "\n".join(
            [f"{i}: {v['video_title']} ({v['channel']})" for i, v in enumerate(results)]
        )
        user_prompt = f"""
Original query: {original_query}
Candidate videos:
{video_list_str}
"""
        chain.add_message(role="user", content=user_prompt)

        # Call LLM
        llm_raw = llm_with_retry(chain.messages_list, llm_name=llm)
        if hasattr(llm_raw, "content"):
            raw_text = llm_raw.content
        elif isinstance(llm_raw, dict):
            raw_text = llm_raw.get("content") or llm_raw.get("text", "")
        else:
            raw_text = llm_raw

        # Validate XML
        if chain.use_xml_checker:
            content = xml_check_and_fix(raw_text, llm=llm)
        else:
            content = raw_text
        chain.add_message(role="ai", content=content)

        # Extract XML tags
        try:
            llm_response = chain.extract_text_in_tags(content)
        except Exception as e:
            print("Failed to parse LLM response.")
            print("Raw content:\n", content)
            raise e

        verdict = llm_response.get("verdict", "").strip().upper()
        if verbose:
            print(f"Verdict: {verdict}")

        if verdict == "TERMINATE":
            selected_indexes = parse_selected_indexes(llm_response.get("selected_indexes", ""))
            selected_videos = [results[i] for i in selected_indexes if i < len(results)]
            return selected_videos

        # Otherwise, refine query
        query = llm_response.get("query", "").strip()
        if not query:
            print("LLM returned CONTINUE but no refined query. Ending.")
            return []

    print("Max turns reached. No videos selected.")
    return []




def run_video_search_for_los(
    sheet,
    worksheet_name: str,
    drive,
    llm,
    k: int = 10,
    max_turns: int = 3,
    filters: Optional[Dict] = None,
    verbose: bool = True,
    max_workers: int = 10
):
    """
    Run video search for each Learning Objective (LO) in the Google Sheet.
    Each LO is processed in parallel, using SmartProgressBar for progress updates.
    Saves intermediate results after every few LOs and a final save.
    """

    # -------------------- Read the sheet -------------------- #
    course_outline_with_lo_sheet, course_outline_with_lo_df = get_sheet_data_and_df(
        sheet=sheet,
        sheet_name=worksheet_name
    )

    if "Learning Objectives" not in course_outline_with_lo_df.columns:
        raise ValueError("Sheet must contain a column named 'Learning Objectives'")

    pair_column = "youtube_videos"
    if pair_column not in course_outline_with_lo_df.columns:
        course_outline_with_lo_df[pair_column] = ""

    # -------------------- Prepare tasks -------------------- #
    # Prepare tasks with context
    task_list = []
    for row_index, row in course_outline_with_lo_df.iterrows():
        lo = row['Learning Objectives']
        if not isinstance(lo, str) or not lo.strip():
            continue
        if row.get(pair_column) and row[pair_column].strip():
            continue

        context = {
            'topic': row.get('Topic', ''),
            'subtopic': row.get('Subtopic', '')
        }
        task_list.append((row_index, lo.strip(), context))

    total_tasks = len(task_list)

    print(f"Starting video search for {total_tasks} Learning Objectives...")

    # -------------------- Progress bar -------------------- #
    save_interval = 10
    progress = SmartProgressBar(total_tasks=total_tasks, description="Percent complete", save_interval=save_interval)

    # -------------------- Parallel processing -------------------- #
    futures_map = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for row_index, lo, context in task_list:
            def task_fn(lo=lo):
                time.sleep(random.uniform(0.8, 1.6))  # staggered delay to reduce API pressure
                videos = video_retriever_agent(
                    query=lo,
                    drive=drive,
                    llm=llm,
                    k=k,
                    max_turns=max_turns,
                    filters=filters,
                    verbose=verbose,
                    context=context
                )
                video_urls = [v.get("video_url", v.get("url", "")) for v in videos]
                return video_urls

            future = executor.submit(task_fn)
            futures_map[future] = (row_index, lo)

        for future in tqdm(as_completed(futures_map), total=total_tasks):
            row_index, lo = futures_map[future]
            try:
                video_urls = future.result()
                video_str = "\n".join(video_urls) if video_urls else "No result"
                course_outline_with_lo_df.at[row_index, pair_column] = video_str

                # Save after each LO
                save_to_sheet(course_outline_with_lo_sheet, course_outline_with_lo_df)
                if verbose:
                    print(f"Processed LO: '{lo}' in row {row_index + 1}")

                # Progress bar update
                progress.update()
                if progress.should_save():
                    print(f"Saving intermediate results after {progress.completed_count} LOs.")
                    save_to_sheet(course_outline_with_lo_sheet, course_outline_with_lo_df)

            except Exception as e:
                print(f"Error processing LO '{lo}' (row {row_index + 1}): {e}")

    # -------------------- Final save -------------------- #
    print("All LOs processed. Saving final results.")
    save_to_sheet(course_outline_with_lo_sheet, course_outline_with_lo_df)

    return course_outline_with_lo_df[pair_column].tolist()
    