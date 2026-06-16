from services.sheets_service import get_sheet_data_and_df, save_to_sheet, clear_worksheet, delete_worksheet, get_worksheet_names
from modules.chain import Chain
from concurrent.futures import ThreadPoolExecutor, as_completed
import streamlit as st
from services.smart_progress_bar import SmartProgressBar
from langsmith import traceable
import re
from agents.slide_chunks.review_revise_slide_chunks import review_revise_slide_chunks


slide_chunks_generation_prompt_old = """You are a Slide Chunking Agent for an E-learning course. Your job is to take research notes for an entire topic and break them into slide-sized instructional chunks that flow naturally from start to finish.

Course Name: {course_name}
Target Audience: {target_audience}

Topic:
{topic}

Below are all subtopics under this topic, each with their learning objectives and research notes:
<subtopics>
{subtopics_content}
</subtopics>

Follow these rules when chunking the research notes into slides:

1. Source Format Recognition:
   Research notes come in two formats. Handle each differently:

   a. Document-Derived — structured text with paragraphs, bullets, or numbered points.
      - Rewrite into clear, narration-ready language as if a senior technician is walking a new team member through the task.
      - Each slide should cover one clear instructional step or closely related set of points.
      - Size each slide so it takes roughly 20-30 seconds to narrate aloud. This is a guideline, not a hard rule — some slides may be shorter or longer depending on the content.

   b. Transcript-Derived — line-by-line transcript segments with timestamps (e.g., '266': text).
      - Do NOT rewrite or paraphrase transcript content. Keep it exactly as-is.
      - Divide into natural segments based on shifts in ideas or instructional steps.
      - Each video chunk should be roughly 45-75 seconds, but can be shorter or longer to preserve natural flow. Do not exceed 90 seconds per chunk.
      - For each chunk, return: Title, Video_Id, Start timestamp, End timestamp, and the original Transcript lines with their timestamps.

2. Slide Structure:
   - Start with exactly ONE Transition slide for the topic. The hook must be drawn from the research notes — use a key fact, practical scenario, or problem stated in the source. Do not invent metaphors or claims. Avoid "In this section..." or "Imagine you're..." patterns.
   - Content slides form the body. Chunk by task or instructional step, not by individual facts. If a point is too thin to fill 20 seconds of narration on its own, merge it into the next related slide rather than letting it stand alone.
   - End with ONE Summary slide only if the topic has enough content to warrant it (3+ content slides). The summary must reference specific skills or procedures from the slides — not generic motivational statements. Do not use bullet points. Do not start with "In this section, we covered..."
   - Do NOT force a transition or summary slide for every subtopic. The topic flows as one instructional unit.

3. Tone and Language:
   - Sound like a real technician, not an AI. Use plain language and active voice.
   - Use action-oriented verbs: "check," "spot," "find" over "identify," "assess," "determine."
   - No buzzwords: "optimal," "leverage," "utilize," "facilitate."
   - Explain trade terms when first used.
   - Present tense, as if the learner is doing the task right now.

4. Content Integrity:
   - Use ONLY the content in the provided research notes. Do not add facts, examples, analogies, metaphors, or information not present in the source. This applies to ALL slide types including Transition and Summary slides.
   - Every specific fact, detail, measurement, and procedural step in the research notes must appear in the output. Do not drop details like positions, ratings, or specifications.
   - Do not remove or skip content unless it is clearly redundant or off-topic.
   - Your job is to restructure and present — not to expand or reduce.

5. Slide Titles:
   - Keep titles clear, concise, and varied.
   - Do not use the same title style on consecutive slides.
   - Narrator doesn't read the slide title. Thus, don't assume the title will be spoken aloud. It should be a clear label for the content, not a full sentence or question.
   - Therefore, content should not rely on the title to make sense. The slide should be understandable even if the title is hidden.

6. Flow:
   - Maintain a natural instructional progression across all subtopics within the topic.
   - Follow a problem → process → consequence narrative where it fits.
   - Every slide must support the learning objectives. Exclude tangential content.


Provide your output strictly in the following format:

<evaluation_breakdown>
Briefly plan your approach:
1. What is the topic about and what should the learner be able to do?
2. What source formats are present (document, transcript, or both)?
3. How will you structure the slide sequence across subtopics?
</evaluation_breakdown>

<slides>
All slides in sequence, separated by three dashes (---). Each slide should follow the specified format exactly. Do not include any explanatory text or commentary — only the slides themselves in the required format.

For document-derived slides:
    Subtopic: [Exact subtopic name this slide belongs to, from the input]
    Title: [Slide title]
    Slide Type: [Transition / Content / Summary]
    Content: [Instructional content in narration-ready language]

For transcript-derived slides:
    Subtopic: [Exact subtopic name this slide belongs to, from the input]
    Title: [Slide title]
    Slide Type: Video
    Video_Id: [Video ID from the input]
    Start: [Start timestamp]
    End: [End timestamp]
    Transcript:
      - '[timestamp]': [line of transcript]
      - '[timestamp]': [line of transcript]
      ...
</slides>

"""


slide_chunks_generation_prompt = """You are a Slide Chunking Agent for an E-learning course. Your job is to take research notes for an entire topic and break them into slide-sized instructional chunks that flow naturally from start to finish.

Course Name: {course_name}
Target Audience: {target_audience}

Topic:
{topic}

Below are all subtopics under this topic, each with their learning objectives and research notes:
<subtopics>
{subtopics_content}
</subtopics>

Follow these rules when chunking the research notes into slides:

1. Source Format Recognition:
   Research notes come in two formats. Handle each differently:

   a. Document-Derived — structured text with paragraphs, bullets, or numbered points.
      - Rewrite into clear, narration-ready language as if a senior technician is walking a new team member through the task.
      - Each slide should cover one clear instructional step or closely related set of points.
      - Size each slide so it takes roughly 20-30 seconds to narrate aloud. This is a guideline, not a hard rule — some slides may be shorter or longer depending on the content.

   b. Transcript-Derived — line-by-line transcript segments with timestamps (e.g., '266': text).
      - Do NOT rewrite or paraphrase transcript content. Keep it exactly as-is.
      - Divide into natural segments based on shifts in ideas or instructional steps.
      - Each video chunk should be roughly 45-75 seconds, but can be shorter or longer to preserve natural flow. Do not exceed 90 seconds per chunk.
      - For each chunk, return: Title, Video_Id, Start timestamp, End timestamp, and the original Transcript lines with their timestamps.

2. Slide Structure:

   a. Transition Slides — one per subtopic, placed at the start of each subtopic's slides.
      - Purpose: Orient the learner to what's coming and why it matters. Set the stage without teaching the lesson.
      - The hook must be drawn from the research notes — use a key fact, practical scenario, or problem stated in the source. Do not invent metaphors or claims.
      - CRITICAL: Do NOT include specific teaching content that belongs on the content slides that follow. A transition should make the learner curious or give them context, not deliver the core facts, diagnostic rules, procedural steps, or conclusions. If a content slide that follows would repeat something already stated in the transition, the transition has overstepped.
      - Avoid "In this section..." or "Imagine you're..." patterns.
      - Length: Keep it brief — 1-2 sentences. The transition is a teaser, not a lecture.

   b. Content Slides — form the instructional body of the topic.
      - Chunk by task or instructional step, not by individual facts.
      - Each content slide should contain enough substance to stand on its own — at minimum ~3 meaningful sentences of teaching content. If a point is too thin to fill 20 seconds of narration on its own, merge it into the next related slide covering related content from the same learning objective rather than letting it stand alone.
      - Conversely, do not overload a single slide with more than two distinct concepts or procedural steps. If a slide covers unrelated procedures, split it.
      - Each content slide's material should trace back to a single learning objective. Do not quietly merge content from two different learning objectives into one slide.
      - Don't compress too much into one slide just to reduce the total number of slides. It's better to have more slides that are clear and focused than fewer slides that are dense and hard to follow.

   c. Summary Slides — one per topic, placed at the end, only if the topic has enough content to warrant it (3+ content slides).
      - The summary must reference specific skills or procedures from the preceding slides. Do not use generic motivational statements.
      - The summary must cover ALL subtopics and key learning objectives from the preceding slides proportionally. Do not omit any subtopic.
      - CRITICAL: Do NOT introduce new terminology, claims, embellishments, or implications that are not present in the research notes. The summary restates what was taught — nothing more.
      - Do not use bullet points. Do not start with "In this section, we covered..."

3. Tone and Language:
   - Sound like a real technician, not an AI. Use plain language and active voice.
   - Use action-oriented verbs: "check," "spot," "find" over "identify," "assess," "determine."
   - No buzzwords: "optimal," "leverage," "utilize," "facilitate."
   - Explain trade terms when first used.
   - Present tense, as if the learner is doing the task right now.
   - If the research notes already use plain language, keep it as is. Do not rewrite for tone if the original is already clear and direct. Your job is to restructure and present — not to rewrite for tone if it's not needed.
   - See below paraphrasing rules:

<paraphrasing_rules>
1. Use plain language, not corporate jargon
2. Sound like a real technician, not an AI
3. Keep all factual information accurate (never add or remove facts)
4. Use active voice and short sentences
5. Explain trade terminology when first used
6. Be conversational but professional
7. No buzzwords like "optimal," "leverage," "utilize," "facilitate"
8. No hype or fluff - every sentence adds value

Examples:
BAD: "Technicians must de-energize the system to mitigate hazards prior to engaging with components."
GOOD: "Shut the power off before you touch anything—it keeps you safe."

BAD: "Ensure optimal airflow parameters."
GOOD: "Good airflow keeps the system running right."

BAD: "Apply pookie to the joints."
GOOD: "Apply mastic—also called pookie by tradesman—to seal the joints."
</paraphrasing_rules>

4. Content Integrity:
   - Use ONLY the content in the provided research notes. Do not add facts, examples, analogies, metaphors, or information not present in the source. This applies to ALL slide types including Transition and Summary slides.
   - Every specific fact, detail, measurement, and procedural step in the research notes must appear in the output. Do not drop details like positions, ratings, or specifications.
   - Do not remove or skip content unless it is clearly redundant or off-topic.
   - Your job is to restructure and present — not to expand or reduce.
   - Inline image links: Any markdown image links in the research notes — in either `[alt](url)` or `![alt](url)` form — must be carried through verbatim into the Content of the slide whose text covers the sentence or paragraph that the link sits next to. Do not delete, rewrite, drop, merge, or modify the URL or alt text of any existing image link. Keep the link format exactly as it appears in the source (same `[` vs `!` prefix, same URL, same alt text). Place each link inline with the Content text, in the same position relative to the surrounding sentence as it had in the source research notes. Image-link placement has already been handled by an upstream pipeline step — treat every `[...](...)` / `![...](...)` you see as load-bearing content to preserve exactly.

5. Slide Titles:
   - Keep titles clear, concise (2-5 words).
   - Titles should follow a logical naming theme based on the topic, subtopic, and learning objectives. For example, if a sequence of slides covers individual components, the titles should clearly name each component — consistency is good when it reflects the content structure.
   - Narrator doesn't read the slide title. Thus, don't assume the title will be spoken aloud. It should be a clear label for the content, not a full sentence or question.
   - Therefore, content should not rely on the title to make sense. The slide should be understandable even if the title is hidden.

6. Avoiding Redundancy:
   - No two slides should deliver the same core information, even in different words.
   - Pay special attention to transition + content slide pairs: if the transition already states a fact or conclusion, the following content slide should not restate it. Rework the transition to be less specific.
   - When the same concept appears in two different research note blocks (e.g., a brief mention in one block and a full explanation in another), the corresponding slides must differentiate their coverage. The first slide should be a brief setup; the second should add genuinely new depth. Do not repeat the same definition or explanation twice.

7. Flow and Continuity:
   - Maintain a natural instructional progression across all subtopics within the topic.
   - Follow a problem → process → consequence narrative where it fits.
   - Every slide must support the learning objectives. Exclude tangential content.
   - When a subtopic spans multiple content slides, ensure each slide logically leads into the next rather than reading as a disconnected series of paragraphs.
   - When there is a thematic shift within a subtopic (e.g., from diagnostic use to safety practices), acknowledge the shift with a bridging phrase rather than pivoting abruptly.

8. Subtopic Attribution:
   - Every slide must be labeled with the correct subtopic name — the subtopic from which the majority of its content originates.
   - Use consistent spelling and capitalization of subtopic names across all slides. If the source research notes contain a typo in a subtopic name, use the corrected form consistently.


Provide your output strictly in the following format:

<evaluation_breakdown>
Plan your approach here. It is okay for this section to be long to be thorough. 
</evaluation_breakdown>

<slides>
All slides in sequence, separated by three dashes (---). Each slide should follow the specified format exactly. Do not include any explanatory text or commentary — only the slides themselves in the required format.

For document-derived slides:
    Subtopic: [Exact subtopic name this slide belongs to, from the input — with consistent spelling]
    Title: [Slide title]
    Slide Type: [Transition / Content / Summary]
    Content: [Instructional content in narration-ready language]

For transcript-derived slides:
    Subtopic: [Exact subtopic name this slide belongs to, from the input — with consistent spelling]
    Title: [Slide title]
    Slide Type: Video
    Video_Id: [Video ID from the input]
    Start: [Start timestamp]
    End: [End timestamp]
    Transcript:
      - '[timestamp]': [line of transcript]
      - '[timestamp]': [line of transcript]
      ...
</slides>
"""


@traceable(
    metadata={
        "agent_name": "slide_chunks_v2",
        "step_name": "Generate Slide Chunks from Research Notes (Topic-Level)",
        "function_name": "generate_slide_chunks_for_topic",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous")
    }
)
def generate_slide_chunks_for_topic(course_name, target_audience, topic, subtopics_data, llm="gemini_3_flash"):
    """
    Generate slide chunks for an entire topic based on all its subtopics' research notes.

    :param course_name: Name of the course.
    :param target_audience: Intended audience of the course.
    :param topic: Topic name.
    :param subtopics_data: List of dicts with keys: subtopic, learning_objectives (list of str), research_notes (list of str).
    :param llm: Language model to use.
    :return: The extracted <slides> block text.
    """
    # Build the subtopics content block
    subtopics_content = _build_subtopics_content(subtopics_data)

    # Initialize the agent
    slide_chunking_agent = Chain(llm=llm, tags=['slides'])

    formatted_prompt = slide_chunks_generation_prompt.format(
        course_name=course_name,
        target_audience=target_audience,
        topic=topic,
        subtopics_content=subtopics_content
    )

    print(f"\n🔍 Slide Chunk Prompt for Topic '{topic}':\n")
    print(formatted_prompt)
    print("\n" + "=" * 100 + "\n")

    slide_chunking_agent.add_message(role="user", content=formatted_prompt)

    response = slide_chunking_agent.run()

    slide_chunks_text = response['slides'].strip()

    # Run review-revise quality loop
    print(f"\n🔄 Starting review-revise loop for topic '{topic}'...")
    slide_chunks_text = review_revise_slide_chunks(
        course_name=course_name,
        target_audience=target_audience,
        topic=topic,
        slide_chunks_text=slide_chunks_text,
        subtopics_data=subtopics_data,
        llm=llm,
        max_iterations=3,
        reviewer_llm="gemini_3_flash",
    )

    return slide_chunks_text


def _build_subtopics_content(subtopics_data):
    """
    Build the XML-tagged subtopics content block for the prompt.

    :param subtopics_data: List of dicts with keys: subtopic, learning_objectives, research_notes.
    :return: Formatted string with all subtopics, LOs, and research notes.
    """
    blocks = []
    for item in subtopics_data:
        subtopic = item["subtopic"]
        los = item["learning_objectives"]
        notes = item["research_notes"]

        lo_block = "\n".join(f"- {lo}" for lo in los)
        notes_block = "\n\n".join(notes)

        blocks.append(
            f"<subtopic>\n"
            f"Subtopic: {subtopic}\n\n"
            f"Learning Objectives:\n{lo_block}\n\n"
            f"Research Notes:\n{notes_block}\n"
            f"</subtopic>"
        )
    return "\n\n".join(blocks)


@traceable(
    metadata={
        "agent_name": "slide_chunks_v2",
        "step_name": "Generate Slide Chunks for All Topics",
        "function_name": "generate_slide_chunks_for_all_topics",
        "user_id": st.session_state.get("role", "anonymous"),
        "user_email": st.session_state.get("user_email", "anonymous")
    }
)
def generate_slide_chunks_for_all_topics(sheet, sheet_name, llm="gemini_3_flash", max_workers=5):
    """
    Generates slide chunks for each unique topic in the worksheet, processing all subtopics per topic together.

    :param sheet: The gspread sheet object.
    :param sheet_name: The worksheet name.
    :param llm: The language model to use.
    :param max_workers: Number of parallel workers (default 5).
    :return: None
    """
    # Fetch course info
    _, course_info_df = get_sheet_data_and_df(sheet, "Course info")
    course_name = course_info_df.loc[0, "Course Name"]
    target_audience = course_info_df.loc[0, "Target Audience & Industry"]

    # Load the worksheet and DataFrame
    worksheet, df = get_sheet_data_and_df(sheet, sheet_name)

    # Ensure slide_chunks column exists
    if "slide_chunks" not in df.columns:
        df["slide_chunks"] = ""

    # Group by Topic — build subtopics_data for each topic
    topic_groups = {}
    topic_first_indices = {}
    for index, row in df.iterrows():
        topic = row["Topic"]
        if topic not in topic_groups:
            topic_groups[topic] = []
            topic_first_indices[topic] = index

        # Collect subtopic info for this row
        subtopic = row["Subtopic"]
        lo = row["Learning Objectives"]
        rn = row.get("research_notes", "")

        # Find or create the subtopic entry within this topic
        existing = next((s for s in topic_groups[topic] if s["subtopic"] == subtopic), None)
        if existing:
            if lo and lo not in existing["learning_objectives"]:
                existing["learning_objectives"].append(lo)
            if rn and rn not in existing["research_notes"]:
                existing["research_notes"].append(rn)
        else:
            topic_groups[topic].append({
                "subtopic": subtopic,
                "learning_objectives": [lo] if lo else [],
                "research_notes": [rn] if rn else []
            })

    topic_names = list(topic_groups.keys())

    def process_topic(topic):
        subtopics_data = topic_groups[topic]
        slides_text = generate_slide_chunks_for_topic(
            course_name, target_audience, topic, subtopics_data, llm=llm
        )
        return slides_text

    # Parallel processing for all topics
    results = [None] * len(topic_names)
    progress = SmartProgressBar(total_tasks=len(topic_names), description="Generating slide chunks (topic-level)", save_interval=5)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {executor.submit(process_topic, topic): idx for idx, topic in enumerate(topic_names)}
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            slides_text = future.result()
            results[idx] = slides_text
            progress.update()
            if progress.should_save():
                # Save partial progress
                df["slide_chunks"] = ""
                for i, topic in enumerate(topic_names):
                    first_idx = topic_first_indices[topic]
                    df.at[first_idx, "slide_chunks"] = results[i] if results[i] is not None else ""
                save_to_sheet(worksheet, df)

    # Final save — place result in first row of each topic
    df["slide_chunks"] = ""
    for i, topic in enumerate(topic_names):
        first_idx = topic_first_indices[topic]
        df.at[first_idx, "slide_chunks"] = results[i] if results[i] is not None else ""

    save_to_sheet(worksheet, df)
    print("Slide chunk generation (topic-level) complete and saved to sheet.")


def delete_slide_chunks_generation_v2(sheet, worksheet_name="Final Outline"):
    """
    Remove the 'slide_chunks' column from the specified worksheet.
    :param sheet: The gspread sheet object.
    :param worksheet_name: The worksheet name (default 'Final Outline').
    :return: None
    """
    ws, df = get_sheet_data_and_df(sheet, worksheet_name)
    if "slide_chunks" in df.columns:
        df = df.drop(columns=["slide_chunks"])
        clear_worksheet(ws)
        save_to_sheet(ws, df)

    # Delete the checklist backup sheet if it exists
    backup_name = "Backup Slide Chunks Sheet for Delete step of Slide Chunks Checklist"
    sheet_names = get_worksheet_names(sheet)
    if backup_name in sheet_names:
        delete_worksheet(sheet, backup_name)
        print(f"Deleted backup sheet '{backup_name}' when deleting slide chunks generation")
